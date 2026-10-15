"""Context features: catalogue freshness relative to the training cutoff.

Unlike ``item_popularity_bucket``, this transform collapses the training
review count into a coarse two-state context (new arrival vs. established)
rather than a graded bucket, because its purpose is explanatory (the paper
and README point to it when discussing the cold-start slice) rather than a
precise popularity prior.
"""

from __future__ import annotations

import pandas as pd


def item_catalogue_context(
    train: pd.DataFrame, n_items: int, new_arrival_max_reviews: int
) -> pd.Series:
    """Flag items with few-to-zero training reviews as "new_arrival".

    Args:
        train: The training partition only.
        n_items: Total item vocabulary size.
        new_arrival_max_reviews: Items with at most this many training
            reviews are flagged ``context=new_arrival``; cold items (zero
            training reviews by construction) always fall in this bucket.

    Returns:
        ``item_index -> "context=new_arrival"`` or ``"context=established"``
        for every item in ``range(n_items)``.
    """
    counts = train.groupby("item_index").size().reindex(range(n_items), fill_value=0)
    return counts.map(
        lambda c: "context=new_arrival" if c <= new_arrival_max_reviews else "context=established"
    )
