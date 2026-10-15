"""Build the top-N payloads published to DynamoDB, plus the request-time
feature-based fallback for users/items that were not part of the last
training run at all.

Two serving paths exist, matching the architecture diagram:

1. **Precomputed path** (the common case): ``build_topn_payloads`` writes one
   ranked list per known user to DynamoDB after training; the Lambda handler
   is then a pure key lookup (see ``handler.py``).
2. **Request-time fallback** (truly new users/items): a user who signed up
   after the last training run has no precomputed row. ``score_new_user``
   builds their feature vector from profile attributes alone (skin
   type/tone -- whatever was captured at signup) via
   :func:`recommender.features.pipeline.recompute_user_tokens`, and scores
   the catalogue directly. This is slower than a DynamoDB GetItem, so it is
   meant to run only until the next scheduled retraining job folds the new
   user into the precomputed table.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from recommender.data.splitting import ColdStartSplit, Split
from recommender.models.hybrid_warp import HybridWARPRecommender


def build_topn_payloads(
    model: HybridWARPRecommender,
    split: Split | ColdStartSplit,
    item_ids: np.ndarray,
    user_ids: np.ndarray,
    k: int = 20,
    users: Iterable[int] | None = None,
    model_name: str = "hybrid_warp",
) -> list[dict[str, Any]]:
    """Materialize per-user top-``k`` lists in DynamoDB item shape.

    Args:
        model: A fitted :class:`HybridWARPRecommender`.
        split: Split whose training partition defines each user's history.
        item_ids: Original product ids aligned to internal item indices.
        user_ids: Original author ids aligned to internal user indices.
        k: List length to store.
        users: Internal user indices; defaults to every user in the split.
        model_name: Stored alongside the list for A/B attribution.

    Returns:
        One dict per user, ready for ``table.put_item``.

    Raises:
        ValueError: If ``k`` is not positive or the id arrays are misaligned.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if user_ids.shape[0] != split.n_users:
        raise ValueError(
            f"user_ids has {user_ids.shape[0]} entries but the split has {split.n_users} users."
        )
    if item_ids.shape[0] != split.n_items:
        raise ValueError(
            f"item_ids has {item_ids.shape[0]} entries but the split has {split.n_items} items."
        )

    history = {
        int(user): set(map(int, items))
        for user, items in split.train.groupby("user_index")["item_index"]
    }
    targets = range(split.n_users) if users is None else users

    payloads = []
    for user_index in targets:
        ranked = model.recommend(int(user_index), k=k, exclude=history.get(int(user_index), set()))
        payloads.append(
            {
                "user_id": str(user_ids[int(user_index)]),
                "model": model_name,
                "recommendations": [
                    {"rank": rank, "product_id": str(item_ids[int(item)])}
                    for rank, item in enumerate(ranked, start=1)
                ],
            }
        )
    return payloads


def score_new_user(
    model: HybridWARPRecommender,
    skin_type: str,
    skin_tone: str,
    item_matrix: sparse.csr_matrix,
    user_vocab: dict[str, int],
    k: int = 10,
) -> np.ndarray:
    """Score the catalogue for a brand-new user from profile attributes alone.

    This is the request-time cold-start fallback: it never touches the
    training interaction log, only the two demographic tokens a new user
    supplies at signup. It is intentionally independent of
    :class:`~recommender.features.pipeline.FeatureStore` so it can run
    without rebuilding the whole feature pipeline per request.

    Args:
        model: A fitted :class:`HybridWARPRecommender`.
        skin_type: Raw skin-type value, e.g. ``"Oily"``.
        skin_tone: Raw skin-tone value, e.g. ``"Deep"``.
        item_matrix: The item feature matrix the model was trained with.
        user_vocab: The user feature vocabulary the model was trained with.
        k: Unused, kept for interface symmetry with ``recommend``; callers
            should truncate the returned score vector themselves.

    Returns:
        A dense score vector over the full catalogue.
    """
    del k  # symmetry only; truncation happens in the caller
    tokens = [f"skin_type={skin_type}", f"skin_tone={skin_tone}"]
    cols = [user_vocab[tok] for tok in tokens if tok in user_vocab]
    if not cols:
        # Totally unknown attributes: no signal at all, fall back to item bias.
        _, bias = model.item_representations()
        return bias

    p_u = model.user_embeddings[cols].sum(axis=0)
    q, b = model.item_representations()
    return q @ p_u + b


def to_dataframe(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten payloads for inspection/export (e.g. a CSV artifact)."""
    rows = []
    for payload in payloads:
        for rec in payload["recommendations"]:
            rows.append(
                {
                    "user_id": payload["user_id"],
                    "model": payload["model"],
                    "rank": rec["rank"],
                    "product_id": rec["product_id"],
                }
            )
    return pd.DataFrame(rows)
