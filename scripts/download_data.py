#!/usr/bin/env python3
"""Fetch the Sephora Products and Skincare Reviews dataset, or generate a
synthetic fixture with the same schema.

The synthetic generator is the default so the repository is runnable from a
clean checkout with no Kaggle account. Unlike a pure identity-clustering
fixture, preference here is driven entirely by **shared categorical
features** (skin type/tone for users, category/brand/ingredient tags for
products): a user's affinity for a product depends on whether the product's
feature tags match the user's skin-type cluster, never on the user's or
product's raw ID. That is what gives the hybrid feature-aware model (and not
just an ID-only baseline) real structure to recover, and it is what makes the
cold-start slice meaningful: a held-out user/item can still be scored
correctly through the features it shares with the training population.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
KAGGLE_DATASET = "nadyinky/sephora-products-and-skincare-reviews"

BRANDS = [
    "Glow Lab", "Dewy Botanics", "Cerave Clinic", "Luxe Rituals", "Barefaced",
    "Hydra Haus", "Solstice Skin", "The Ordinary Shelf", "Velvet Veil",
    "Nocturne Labs", "Minimal Skin Co", "Radiant Theory", "Pure Clay Studio",
    "Aurora Derm", "Second Skin",
]
CATEGORIES = [
    "Moisturizers", "Cleansers", "Serums", "Sunscreen", "Masks",
    "Eye Care", "Treatments", "Lip Care",
]
SKIN_TYPES = ["Dry", "Oily", "Combination", "Normal", "Sensitive"]
SKIN_TONES = ["Fair", "Light", "Medium", "Tan", "Deep", "Rich"]

# Each skin type has a short list of ingredient tags it responds well to; this
# is the latent structure the clustering is built from.
SKIN_TYPE_INGREDIENT_AFFINITY: dict[str, list[str]] = {
    "Dry": ["Hyaluronic Acid", "Ceramides", "Squalane", "Glycerin"],
    "Oily": ["Salicylic Acid", "Niacinamide", "Clay", "Zinc PCA"],
    "Combination": ["Niacinamide", "Hyaluronic Acid", "Green Tea"],
    "Normal": ["Vitamin C", "Peptides", "Antioxidant Complex"],
    "Sensitive": ["Centella Asiatica", "Oat Extract", "Ceramides", "Panthenol"],
}
ALL_INGREDIENTS = sorted({tag for tags in SKIN_TYPE_INGREDIENT_AFFINITY.values() for tag in tags})
ALL_INGREDIENTS += ["Retinol", "AHA", "BHA", "Caffeine", "SPF Filters", "Peptides"]
ALL_INGREDIENTS = sorted(set(ALL_INGREDIENTS))

PRICE_BUCKET_BY_CATEGORY = {
    "Moisturizers": (18, 65), "Cleansers": (10, 35), "Serums": (25, 95),
    "Sunscreen": (14, 42), "Masks": (6, 28), "Eye Care": (20, 70),
    "Treatments": (22, 88), "Lip Care": (8, 24),
}


def generate_synthetic(
    n_users: int = 1200,
    n_products: int = 600,
    seed: int = 20260115,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Sephora-shaped ``(reviews, products)`` fixtures.

    Args:
        n_users: Number of synthetic reviewers.
        n_products: Number of synthetic products.
        seed: Seed controlling the whole generation.

    Returns:
        ``(reviews, products)`` frames matching the Kaggle schema subset used
        by this project.
    """
    rng = np.random.default_rng(seed)

    categories = rng.choice(CATEGORIES, size=n_products)
    brands = rng.choice(BRANDS, size=n_products)

    # Each product carries 2-4 ingredient tags; one is drawn from a random
    # skin-type's affinity list (so every skin type is represented in the
    # catalogue), the rest from the full pool.
    product_rows = []
    for pid in range(1, n_products + 1):
        category = categories[pid - 1]
        anchor_type = rng.choice(SKIN_TYPES)
        anchor_tags = SKIN_TYPE_INGREDIENT_AFFINITY[anchor_type]
        n_tags = int(rng.integers(2, 5))
        tags = {rng.choice(anchor_tags)}
        while len(tags) < n_tags:
            tags.add(rng.choice(ALL_INGREDIENTS))

        low, high = PRICE_BUCKET_BY_CATEGORY[category]
        price = round(float(rng.uniform(low, high)), 2)

        category_label = category[:-1] if category.endswith("s") else category
        product_rows.append(
            {
                "product_id": f"P{pid:05d}",
                "product_name": f"Synthetic {category_label} {pid:04d}",
                "brand_name": brands[pid - 1],
                "primary_category": category,
                "price_usd": price,
                "ingredients": ";".join(sorted(tags)),
                # Long-tailed popularity, as in the real catalogue.
                "loves_count": int(rng.pareto(1.3) * 2500) + rng.integers(10, 400),
                "out_of_stock": bool(rng.random() < 0.03),
            }
        )
    products = pd.DataFrame(product_rows)

    # Precompute, for each product, which skin types its tags serve.
    product_tag_sets = [set(row.split(";")) for row in products["ingredients"]]

    def affinity_score(tag_set: set[str], skin_type: str) -> float:
        anchor_tags = set(SKIN_TYPE_INGREDIENT_AFFINITY[skin_type])
        overlap = len(tag_set & anchor_tags)
        # Sharpened on purpose: the synthetic fixture exists to give the
        # hybrid model real ingredient/category structure to recover with a
        # modest training budget, so overlap dominates heavily over noise.
        return 1.0 + 12.0 * overlap

    users_skin_type = rng.choice(SKIN_TYPES, size=n_users)
    users_skin_tone = rng.choice(SKIN_TONES, size=n_users)
    popularity = products["loves_count"].to_numpy(dtype=np.float64)
    popularity = popularity / popularity.sum()

    records: list[dict[str, object]] = []
    for user in range(1, n_users + 1):
        skin_type = users_skin_type[user - 1]
        weights = np.array(
            [affinity_score(tags, skin_type) for tags in product_tag_sets]
        )
        # A mild popularity tilt (not dominant): affinity drives the choice,
        # popularity only breaks ties among equally-affine products.
        weights = weights * (0.6 + 0.4 * popularity * n_products)
        weights = weights / weights.sum()

        n_interactions = int(rng.integers(6, 36))
        n_interactions = min(n_interactions, n_products)
        chosen = rng.choice(n_products, size=n_interactions, replace=False, p=weights)

        base_time = int(rng.integers(1_580_000_000, 1_700_000_000))
        offsets = np.sort(rng.integers(0, 50_000_000, size=n_interactions))

        for position, item_idx in enumerate(chosen):
            tag_set = product_tag_sets[item_idx]
            centre = affinity_score(tag_set, skin_type)
            rating = int(np.clip(round(rng.normal(centre, 1.1)), 1, 5))
            is_recommended = bool(rating >= 4 and rng.random() < 0.9)

            records.append(
                {
                    "author_id": f"U{user:06d}",
                    "product_id": products.loc[item_idx, "product_id"],
                    "rating": rating,
                    "is_recommended": is_recommended,
                    "submission_time": base_time + int(offsets[position]),
                    "skin_tone": users_skin_tone[user - 1],
                    "skin_type": skin_type,
                }
            )

    reviews = pd.DataFrame.from_records(records)
    return reviews, products


def download_from_kaggle(target: Path) -> None:
    """Download the real dataset using credentials from the environment."""
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise SystemExit(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in the environment.\n"
            "Credentials are never read from or written to this repository."
        )

    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET, "-p", str(target), "--unzip",
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("The 'kaggle' package is not installed: pip install kaggle") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Kaggle download failed with exit code {exc.returncode}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--synthetic", action="store_true",
        help="Generate the offline fixture (default).",
    )
    source.add_argument(
        "--kaggle", action="store_true",
        help="Download the real dataset using KAGGLE_USERNAME/KAGGLE_KEY.",
    )
    parser.add_argument("--users", type=int, default=1200)
    parser.add_argument("--products", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260115)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.kaggle:
        download_from_kaggle(DATA_DIR)
        print(f"Downloaded {KAGGLE_DATASET} into {DATA_DIR}")
        return 0

    reviews, products = generate_synthetic(args.users, args.products, args.seed)
    reviews.to_csv(DATA_DIR / "reviews.csv", index=False)
    products.to_csv(DATA_DIR / "product_info.csv", index=False)

    density = len(reviews) / (args.users * args.products)
    print(
        f"Synthetic fixture written to {DATA_DIR}\n"
        f"  users={args.users} products={args.products} reviews={len(reviews)}\n"
        f"  density={density:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
