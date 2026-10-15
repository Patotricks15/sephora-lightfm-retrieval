"""Feature pipeline: builds the user/item feature matrices the hybrid model factorizes.

This module turns the declarative ``conf/features.yaml`` into two sparse
multi-hot matrices:

``user_matrix`` (``n_users x n_user_vocab``)
    Row ``u`` has a ``1`` in column ``v`` iff token ``v`` applies to user
    ``u``. The hybrid model's user representation is
    ``p_u = user_matrix[u] @ user_embeddings``, i.e. the sum of the embeddings
    of every applicable token -- exactly the equation in the paper.

``item_matrix`` (``n_items x n_item_vocab``)
    Same idea for items.

Both matrices are built once, from the training partition only where the spec
requires it, and reused by training, evaluation and serving alike -- the same
object answers "what does the model see" in every one of those contexts,
which is what makes training/serving parity testable rather than aspirational.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from recommender.features import (
    interaction_features,
    item_features,
    user_features,
)
from recommender.features.registry import validate_registry
from recommender.features.spec import FeatureSpec, load_feature_specs


def _params(specs: list[FeatureSpec]) -> dict[str, dict]:
    return {spec.transform: spec.params for spec in specs}


def _index_catalogue(catalogue: pd.DataFrame, item_ids: np.ndarray) -> pd.DataFrame:
    """Attach the internal ``item_index`` to raw product metadata."""
    item_map = {pid: idx for idx, pid in enumerate(item_ids)}
    indexed = catalogue.copy()
    indexed["item_index"] = indexed["product_id"].map(item_map)
    return indexed.dropna(subset=["item_index"]).astype({"item_index": int})


def _assemble_user_tokens(
    n_users: int,
    skin_type: pd.Series,
    skin_tone: pd.Series,
    identity: pd.Series,
    review_count: pd.Series,
    fallbacks: dict[str, str],
) -> list[list[str]]:
    tokens: list[list[str]] = []
    for idx in range(n_users):
        row = [
            skin_type.get(idx, fallbacks["user_skin_type"]),
            skin_tone.get(idx, fallbacks["user_skin_tone"]),
            review_count.get(idx, fallbacks["user_review_count_bucket"]),
        ]
        if idx in identity.index:
            row.append(identity.loc[idx])
        tokens.append(row)
    return tokens


def _assemble_item_tokens(
    n_items: int,
    brand: pd.Series,
    category: pd.Series,
    price_bucket: pd.Series,
    ingredients: pd.Series,
    identity: pd.Series,
    fallbacks: dict[str, str],
) -> list[list[str]]:
    """Build the item retrieval representation.

    Deliberately excludes ``item_popularity_bucket`` and
    ``item_catalogue_context`` even though both are declared, registered and
    leakage-tested like every other feature: every cold item has exactly zero
    training interactions by construction, so both aggregates collapse every
    cold item onto the same "unpopular" / "new_arrival" token. Including them
    here would teach the model to systematically suppress cold items, which
    is precisely the capability this project exists to test. The baselines
    already encode an item popularity prior through their own bias terms
    (which is just as unavailable for a cold item), so nothing about the
    comparison is lost by leaving these two aggregates out of the hybrid
    model's own representation. They remain fully part of the feature store
    for the schema/leakage tests and the paper's feature table.
    """
    tokens: list[list[str]] = []
    for idx in range(n_items):
        row = [
            brand.get(idx, fallbacks["item_brand"]),
            category.get(idx, fallbacks["item_category"]),
            price_bucket.get(idx, fallbacks["item_price_bucket"]),
        ]
        row.extend(ingredients.get(idx, []) or [fallbacks["item_ingredient_tags"]])
        if idx in identity.index:
            row.append(identity.loc[idx])
        tokens.append(row)
    return tokens


def _build_matrix(tokens: list[list[str]], vocab: dict[str, int]) -> sparse.csr_matrix:
    rows, cols = [], []
    for row_idx, row_tokens in enumerate(tokens):
        for tok in row_tokens:
            col = vocab.get(tok)
            if col is not None:
                rows.append(row_idx)
                cols.append(col)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(tokens), len(vocab)))


@dataclass
class FeatureStore:
    """Materialized user/item feature vocabularies and multi-hot matrices."""

    user_tokens: list[list[str]]
    item_tokens: list[list[str]]
    user_vocab: dict[str, int]
    item_vocab: dict[str, int]
    user_matrix: sparse.csr_matrix
    item_matrix: sparse.csr_matrix
    n_users: int
    n_items: int
    specs: list[FeatureSpec]

    def user_vector_tokens(self, user_index: int) -> list[str]:
        return self.user_tokens[user_index]

    def item_vector_tokens(self, item_index: int) -> list[str]:
        return self.item_tokens[item_index]

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """Export tidy per-entity token lists, for the feature table in the paper."""
        users = pd.DataFrame(
            {
                "user_index": np.arange(self.n_users),
                "tokens": [", ".join(t) for t in self.user_tokens],
            }
        )
        items = pd.DataFrame(
            {
                "item_index": np.arange(self.n_items),
                "tokens": [", ".join(t) for t in self.item_tokens],
            }
        )
        return {"users": users, "items": items}


def build_feature_store(
    train: pd.DataFrame,
    full_frame: pd.DataFrame,
    catalogue: pd.DataFrame,
    item_ids: np.ndarray,
    n_users: int,
    n_items: int,
    spec_path: str | Path | None = None,
) -> FeatureStore:
    """Materialize every declared feature into the two multi-hot matrices.

    Args:
        train: The warm-start training partition (``fit_on: train_only``
            features and identity vocabularies are built from this alone).
        full_frame: The complete interaction frame, used only for the static
            ``profile`` attributes (skin type/tone), which are not statistics
            and therefore not subject to the train-only rule.
        catalogue: Raw product metadata (``fit_on: catalogue`` features).
        item_ids: Original product ids aligned to internal item indices.
        n_users: Total user vocabulary size.
        n_items: Total item vocabulary size.
        spec_path: Optional override of ``conf/features.yaml``.

    Returns:
        A populated :class:`FeatureStore`.
    """
    specs = load_feature_specs(spec_path)
    validate_registry(specs)
    params = _params(specs)
    fallbacks = {spec.transform: spec.fallback for spec in specs}

    catalogue_indexed = _index_catalogue(catalogue, item_ids)

    skin_type = user_features.user_skin_type(full_frame)
    skin_tone = user_features.user_skin_tone(full_frame)
    identity_u = user_features.user_identity_token(train)
    review_count = interaction_features.user_review_count_bucket(
        train, edges=params["user_review_count_bucket"]["edges"]
    )

    brand = item_features.item_brand(catalogue_indexed)
    category = item_features.item_category(catalogue_indexed)
    price_bucket = item_features.item_price_bucket(
        catalogue_indexed, edges=params["item_price_bucket"]["edges"]
    )
    ingredients = item_features.item_ingredient_tags(
        catalogue_indexed, separator=params["item_ingredient_tags"].get("separator", ";")
    )
    identity_i = item_features.item_identity_token(train)
    # item_popularity_bucket and item_catalogue_context are declared, fitted
    # and leakage-tested, but deliberately not part of the retrieval
    # representation -- see the docstring of _assemble_item_tokens.

    user_tokens = _assemble_user_tokens(
        n_users, skin_type, skin_tone, identity_u, review_count, fallbacks
    )
    item_tokens = _assemble_item_tokens(
        n_items,
        brand,
        category,
        price_bucket,
        ingredients,
        identity_i,
        fallbacks,
    )

    user_vocab = {tok: i for i, tok in enumerate(sorted({t for row in user_tokens for t in row}))}
    item_vocab = {tok: i for i, tok in enumerate(sorted({t for row in item_tokens for t in row}))}

    return FeatureStore(
        user_tokens=user_tokens,
        item_tokens=item_tokens,
        user_vocab=user_vocab,
        item_vocab=item_vocab,
        user_matrix=_build_matrix(user_tokens, user_vocab),
        item_matrix=_build_matrix(item_tokens, item_vocab),
        n_users=n_users,
        n_items=n_items,
        specs=specs,
    )


def recompute_user_tokens(
    user_index: int,
    full_frame: pd.DataFrame,
    train: pd.DataFrame,
    spec_path: str | Path | None = None,
) -> list[str]:
    """Recompute one user's tokens independently of a pre-built :class:`FeatureStore`.

    This is the function the serving path calls for a user who was not part
    of the last training run at all (truly new, request-time cold start).
    ``tests/test_features_parity.py`` asserts this matches the batch-built
    store for users who *were* part of training, which is what catches
    training/serving skew.
    """
    specs = load_feature_specs(spec_path)
    params = _params(specs)
    fallbacks = {spec.transform: spec.fallback for spec in specs}

    skin_type = user_features.user_skin_type(full_frame)
    skin_tone = user_features.user_skin_tone(full_frame)
    identity = user_features.user_identity_token(train)
    review_count = interaction_features.user_review_count_bucket(
        train, edges=params["user_review_count_bucket"]["edges"]
    )

    tokens = [
        skin_type.get(user_index, fallbacks["user_skin_type"]),
        skin_tone.get(user_index, fallbacks["user_skin_tone"]),
        review_count.get(user_index, fallbacks["user_review_count_bucket"]),
    ]
    if user_index in identity.index:
        tokens.append(identity.loc[user_index])
    return tokens


def recompute_item_tokens(
    item_index: int,
    catalogue: pd.DataFrame,
    item_ids: np.ndarray,
    train: pd.DataFrame,
    n_items: int,
    spec_path: str | Path | None = None,
) -> list[str]:
    """Recompute one item's tokens independently of a pre-built :class:`FeatureStore`."""
    specs = load_feature_specs(spec_path)
    params = _params(specs)
    fallbacks = {spec.transform: spec.fallback for spec in specs}

    catalogue_indexed = _index_catalogue(catalogue, item_ids)
    brand = item_features.item_brand(catalogue_indexed)
    category = item_features.item_category(catalogue_indexed)
    price_bucket = item_features.item_price_bucket(
        catalogue_indexed, edges=params["item_price_bucket"]["edges"]
    )
    ingredients = item_features.item_ingredient_tags(
        catalogue_indexed, separator=params["item_ingredient_tags"].get("separator", ";")
    )
    identity = item_features.item_identity_token(train)
    # item_popularity_bucket / item_catalogue_context intentionally excluded;
    # see build_feature_store / _assemble_item_tokens for why.

    tokens = [
        brand.get(item_index, fallbacks["item_brand"]),
        category.get(item_index, fallbacks["item_category"]),
        price_bucket.get(item_index, fallbacks["item_price_bucket"]),
    ]
    tokens.extend(ingredients.get(item_index, []) or [fallbacks["item_ingredient_tags"]])
    if item_index in identity.index:
        tokens.append(identity.loc[item_index])
    return tokens
