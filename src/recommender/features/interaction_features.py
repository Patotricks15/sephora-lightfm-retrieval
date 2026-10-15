"""Aggregate features derived from the interaction log itself.

These are the features with real leakage risk: they are statistics over
reviews, so the rule is strict. ``user_review_count_bucket`` additionally
accepts ``cutoff_order`` and filters with ``<`` (never ``<=``) against each
user's own ``interaction_order`` when one is supplied, which is what
``tests/test_features_leakage.py`` exercises directly. Item-level aggregates
are instead made leakage-safe by only ever being called with the training
partition (the ``fit_on: train_only`` contract), matching the sibling
portfolio projects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def user_review_count_bucket(
    frame: pd.DataFrame, edges: list[int], cutoff_order: int | None = None
) -> pd.Series:
    """Bucketed count of each user's training reviews, point-in-time correct.

    Args:
        frame: Interaction frame carrying ``user_index`` and
            ``interaction_order``.
        edges: Ascending bucket boundaries on the raw count.
        cutoff_order: If given, only rows with ``interaction_order <
            cutoff_order`` count -- strictly less-than, never less-equal.

    Returns:
        ``user_index -> "review_count=<bucket>"``.
    """
    if cutoff_order is not None:
        frame = frame.loc[frame["interaction_order"] < cutoff_order]

    counts = frame.groupby("user_index").size()
    labels = [f"review_count={i}" for i in range(len(edges) + 1)]
    bucket_idx = np.digitize(counts.to_numpy(), edges)
    return pd.Series([labels[i] for i in bucket_idx], index=counts.index)


def item_popularity_bucket(train: pd.DataFrame, edges: list[int], n_items: int) -> pd.Series:
    """Bucketed training-partition popularity for every item (0 for unseen).

    Args:
        train: The training partition only.
        edges: Ascending bucket boundaries on the raw review count.
        n_items: Total item vocabulary size, so unseen (cold) items still get
            an explicit "unseen" bucket rather than a missing row.

    Returns:
        ``item_index -> "popularity=<bucket>"`` for every item in
        ``range(n_items)``.
    """
    counts = train.groupby("item_index").size().reindex(range(n_items), fill_value=0)
    labels = [f"popularity={i}" for i in range(len(edges) + 1)]
    bucket_idx = np.digitize(counts.to_numpy(), edges)
    return pd.Series([labels[i] for i in bucket_idx], index=counts.index)
