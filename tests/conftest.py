"""Shared pytest fixtures built from the synthetic Sephora fixture."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recommender.data import (
    build_implicit_feedback,
    leave_last_one_out,
    load_product_catalogue,
    load_reviews,
    make_cold_start_split,
)
from recommender.features import build_feature_store

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def _ensure_dataset() -> None:
    """Generate the synthetic fixture on demand so tests are self-contained."""
    if (DATA_DIR / "reviews.csv").exists() and (DATA_DIR / "product_info.csv").exists():
        return

    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from download_data import generate_synthetic

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reviews, products = generate_synthetic(n_users=400, n_products=180, seed=11)
    reviews.to_csv(DATA_DIR / "reviews.csv", index=False)
    products.to_csv(DATA_DIR / "product_info.csv", index=False)


@pytest.fixture(scope="session")
def raw_reviews() -> pd.DataFrame:
    _ensure_dataset()
    return load_reviews(DATA_DIR / "reviews.csv")


@pytest.fixture(scope="session")
def catalogue() -> pd.DataFrame:
    _ensure_dataset()
    return load_product_catalogue(DATA_DIR / "product_info.csv")


@pytest.fixture(scope="session")
def interactions(raw_reviews: pd.DataFrame):
    return build_implicit_feedback(raw_reviews)


@pytest.fixture(scope="session")
def split(interactions):
    return leave_last_one_out(interactions)


@pytest.fixture(scope="session")
def cold_split(interactions):
    return make_cold_start_split(interactions, cold_user_frac=0.08, cold_item_frac=0.08, seed=3)


@pytest.fixture(scope="session")
def feature_store(cold_split, interactions, catalogue):
    return build_feature_store(
        cold_split.train,
        interactions.frame,
        catalogue,
        interactions.item_ids,
        cold_split.n_users,
        cold_split.n_items,
    )
