"""Command line interface: train, evaluate and recommend.

Hyperparameters below are the values used throughout the paper's reported
run. The test partition and both cold-start slices are touched only by
``evaluate``, and only once, which is what keeps the reported numbers honest.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from recommender.data import build_implicit_feedback, load_product_catalogue, load_reviews
from recommender.data.splitting import make_cold_start_split
from recommender.evaluation import evaluate, evaluate_cold_start
from recommender.features import build_feature_store
from recommender.models import (
    HybridWARPRecommender,
    IDOnlyMFRecommender,
    PopularityRecommender,
    RandomRecommender,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts"
DEFAULT_REVIEWS = PROJECT_ROOT / "data" / "reviews.csv"
DEFAULT_PRODUCTS = PROJECT_ROOT / "data" / "product_info.csv"

BEST_HYBRID = {
    "n_factors": 24,
    "n_epochs": 20,
    "learning_rate": 0.06,
    "reg": 0.0005,
    "max_sampled": 8,
    "max_bias_magnitude": 3.0,
}
BEST_ID_MF = {"n_factors": 32, "n_epochs": 40, "learning_rate": 0.08}


def _load(reviews_path: Path, products_path: Path, seed: int) -> tuple:
    reviews = load_reviews(reviews_path)
    products = load_product_catalogue(products_path)
    data = build_implicit_feedback(reviews)
    cold_split = make_cold_start_split(data, seed=seed)
    feature_store = build_feature_store(
        cold_split.train,
        data.frame,
        products,
        data.item_ids,
        cold_split.n_users,
        cold_split.n_items,
    )
    return data, cold_split, feature_store


def cmd_train(args: argparse.Namespace) -> int:
    data, split, feature_store = _load(args.reviews, args.products, args.seed)
    print(f"users={split.n_users}  items={split.n_items}  train={len(split.train)}")
    print(f"cold users={len(split.cold_user_ids)}  cold items={len(split.cold_item_ids)}")

    model = HybridWARPRecommender(random_state=args.seed, **BEST_HYBRID)
    model.fit(split, feature_store.user_matrix, feature_store.item_matrix, verbose=args.verbose)

    ARTIFACTS.mkdir(exist_ok=True)
    with (ARTIFACTS / "hybrid_warp.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "user_ids": data.user_ids,
                "item_ids": data.item_ids,
                "user_matrix": feature_store.user_matrix,
                "item_matrix": feature_store.item_matrix,
            },
            handle,
        )

    loss = model.loss_history
    print(f"Hybrid WARP trained: loss {loss[0]:.4f} -> {loss[-1]:.4f}")
    print(f"saved -> {ARTIFACTS / 'hybrid_warp.pkl'}")
    return 0


def _build_models(seed: int) -> dict:
    return {
        "Random": RandomRecommender(random_state=seed),
        "Popularity": PopularityRecommender(),
        "ID-only MF": IDOnlyMFRecommender(random_state=seed, **BEST_ID_MF),
    }


def cmd_evaluate(args: argparse.Namespace) -> int:
    _, split, feature_store = _load(args.reviews, args.products, args.seed)

    warm_rows = []
    cold_rows = []

    for name, model in _build_models(args.seed).items():
        model.fit(split)
        report = evaluate(model, split, k=args.k, partition=args.partition, model_name=name)
        warm_rows.append(report.as_row())
        for cold_report in evaluate_cold_start(model, split, k=args.k, model_name=name).values():
            cold_rows.append(cold_report.as_row())

    hybrid = HybridWARPRecommender(random_state=args.seed, **BEST_HYBRID)
    hybrid.fit(split, feature_store.user_matrix, feature_store.item_matrix)
    hybrid_report = evaluate(
        hybrid, split, k=args.k, partition=args.partition, model_name="Hybrid WARP"
    )
    warm_rows.append(hybrid_report.as_row())
    hybrid_cold_reports = evaluate_cold_start(hybrid, split, k=args.k, model_name="Hybrid WARP")
    for cold_report in hybrid_cold_reports.values():
        cold_rows.append(cold_report.as_row())

    for row in warm_rows:
        print(
            f"{row['model']:12} [{row['slice']:>10}] ndcg@{args.k}={row['ndcg']:.4f}  "
            f"recall={row['recall']:.4f}  map={row['map']:.4f}"
        )
    for row in cold_rows:
        print(
            f"{row['model']:12} [{row['slice']:>10}] ndcg@{args.k}={row['ndcg']:.4f}  "
            f"recall={row['recall']:.4f}  map={row['map']:.4f}"
        )

    ARTIFACTS.mkdir(exist_ok=True)
    pd.DataFrame(warm_rows).to_csv(ARTIFACTS / "results_warm.csv", index=False)
    pd.DataFrame(cold_rows).to_csv(ARTIFACTS / "results_cold_start.csv", index=False)
    print(f"\nsaved -> {ARTIFACTS / 'results_warm.csv'}, {ARTIFACTS / 'results_cold_start.csv'}")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    data, split, feature_store = _load(args.reviews, args.products, args.seed)

    user_lookup = {uid: idx for idx, uid in enumerate(data.user_ids)}
    if args.user_id not in user_lookup:
        raise SystemExit(f"author_id {args.user_id} is not in the filtered dataset.")
    user_index = user_lookup[args.user_id]

    model = HybridWARPRecommender(random_state=args.seed, **BEST_HYBRID)
    model.fit(split, feature_store.user_matrix, feature_store.item_matrix)
    history = set(split.train.loc[split.train["user_index"] == user_index, "item_index"])
    ranked = model.recommend(user_index, k=args.k, exclude=history)

    names: dict[str, str] = {}
    if args.products.exists():
        products = load_product_catalogue(args.products)
        names = dict(zip(products["product_id"], products["product_name"], strict=False))

    payload = []
    for rank, item_index in enumerate(ranked, start=1):
        product_id = str(data.item_ids[int(item_index)])
        payload.append({"rank": rank, "product_id": product_id, "name": names.get(product_id, "?")})

    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sephora-lightfm", description=__doc__)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Fit the hybrid WARP model and persist the artifact.")
    train.add_argument("--verbose", action="store_true")
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", help="Benchmark the hybrid model against the baselines.")
    ev.add_argument("--k", type=int, default=10)
    ev.add_argument("--partition", choices=["validation", "test"], default="test")
    ev.set_defaults(func=cmd_evaluate)

    rec = sub.add_parser("recommend", help="Top-k list for one user.")
    rec.add_argument("--user-id", type=str, required=True)
    rec.add_argument("--k", type=int, default=10)
    rec.set_defaults(func=cmd_recommend)

    args = parser.parse_args(argv)
    if not args.reviews.exists():
        raise SystemExit(
            f"{args.reviews} not found. Run: python scripts/download_data.py --synthetic"
        )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
