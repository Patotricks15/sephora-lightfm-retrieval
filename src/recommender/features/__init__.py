"""Point-in-time feature engineering: the backbone representation of the hybrid model.

Unlike the ID-indexed sibling projects, these modules do not produce an
auxiliary signal added on top of a user/item embedding table -- they produce
the user and item vectors themselves. See ``conf/features.yaml`` and
``pipeline.py`` for the full contract.
"""

from __future__ import annotations

from recommender.features.pipeline import (
    FeatureStore,
    build_feature_store,
    recompute_item_tokens,
    recompute_user_tokens,
)
from recommender.features.spec import FeatureSpec, load_feature_specs

__all__ = [
    "FeatureStore",
    "build_feature_store",
    "recompute_item_tokens",
    "recompute_user_tokens",
    "FeatureSpec",
    "load_feature_specs",
]
