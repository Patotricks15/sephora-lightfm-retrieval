"""Loading and implicit-feedback construction for the Sephora reviews dataset.

The real Kaggle dataset (``nadyinky/sephora-products-and-skincare-reviews``)
ships two kinds of files:

``product_info.csv``
    One row per product: ``product_id, brand_name, primary_category,
    price_usd, ingredients, loves_count, ...``.

``reviews.csv`` (several parts in the real export, merged here)
    One row per review: ``author_id, product_id, rating, is_recommended,
    submission_time, skin_tone, skin_type, ...``.

A review is treated as an implicit positive when the reviewer rated the
product 4 or 5 stars, or explicitly marked ``is_recommended``. Timestamps are
present (``submission_time``), unlike the Anime sibling project, so the
point-in-time ``interaction_order`` column is derived from real chronological
order rather than an append-order proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MIN_POSITIVE_RATING = 4


@dataclass(frozen=True)
class InteractionData:
    """Implicit-feedback interactions with contiguous internal indices.

    Attributes:
        frame: Interactions with ``user_index``, ``item_index`` and ordering.
        user_ids: Original ``author_id`` values, aligned to ``user_index``.
        item_ids: Original ``product_id`` values, aligned to ``item_index``.
    """

    frame: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return int(self.user_ids.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.item_ids.shape[0])

    def __len__(self) -> int:
        return int(self.frame.shape[0])


def load_reviews(path: str | Path) -> pd.DataFrame:
    """Read the raw reviews file and validate its contract."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/download_data.py --synthetic' first."
        )

    frame = pd.read_csv(path)
    required = {"author_id", "product_id", "rating", "is_recommended", "submission_time"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    return frame


def load_product_catalogue(path: str | Path) -> pd.DataFrame:
    """Read the item catalogue used for item features and popularity."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/download_data.py --synthetic' first."
        )

    frame = pd.read_csv(path)
    required = {"product_id", "brand_name", "primary_category", "price_usd", "ingredients"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame["ingredients"] = frame["ingredients"].fillna("")
    frame["price_usd"] = pd.to_numeric(frame["price_usd"], errors="coerce").fillna(0.0)
    return frame


def build_implicit_feedback(
    reviews: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> InteractionData:
    """Convert raw reviews into implicit feedback with contiguous indices.

    A row is a positive when ``rating >= MIN_POSITIVE_RATING`` or
    ``is_recommended`` is true. Everything else is dropped rather than used as
    an explicit negative -- the usual implicit-feedback assumption.

    Iterative core filtering removes users and items below the interaction
    floor; it loops because removing items can push a user below threshold.

    Args:
        reviews: Raw frame from :func:`load_reviews`.
        min_user_interactions: Minimum positives required to keep a user.
        min_item_interactions: Minimum positives required to keep an item.

    Returns:
        Indexed :class:`InteractionData`.

    Raises:
        ValueError: If filtering removes every interaction.
    """
    frame = reviews.copy()
    frame = frame.drop_duplicates(subset=["author_id", "product_id"], keep="first")

    is_recommended = frame["is_recommended"].astype(bool)
    is_highly_rated = frame["rating"] >= MIN_POSITIVE_RATING
    frame = frame.loc[is_recommended | is_highly_rated].copy()

    if frame.empty:
        raise ValueError("No positive interactions remain after implicit-feedback filtering.")

    while True:
        user_counts = frame["author_id"].value_counts()
        keep_users = user_counts[user_counts >= min_user_interactions].index
        item_counts = frame["product_id"].value_counts()
        keep_items = item_counts[item_counts >= min_item_interactions].index

        filtered = frame.loc[
            frame["author_id"].isin(keep_users) & frame["product_id"].isin(keep_items)
        ]
        if filtered.shape[0] == frame.shape[0]:
            frame = filtered
            break
        frame = filtered
        if frame.empty:
            raise ValueError(
                "Core filtering removed every interaction; lower the minimum thresholds."
            )

    frame = frame.sort_values(["author_id", "submission_time"], kind="mergesort").reset_index(
        drop=True
    )
    frame["interaction_order"] = frame.groupby("author_id").cumcount()

    user_ids = np.sort(frame["author_id"].unique())
    item_ids = np.sort(frame["product_id"].unique())
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    frame["user_index"] = frame["author_id"].map(user_map).astype(np.int32)
    frame["item_index"] = frame["product_id"].map(item_map).astype(np.int32)

    return InteractionData(frame=frame, user_ids=user_ids, item_ids=item_ids)
