"""Registry binding declared feature names to their implementations."""

from __future__ import annotations

from collections.abc import Callable

from recommender.features import (
    context_features,
    interaction_features,
    item_features,
    user_features,
)
from recommender.features.spec import FeatureSpec

_MODULES = {
    "user_features": user_features,
    "item_features": item_features,
    "interaction_features": interaction_features,
    "context_features": context_features,
}

TRANSFORM_REGISTRY: dict[str, Callable[..., object]] = {
    "user_skin_type": user_features.user_skin_type,
    "user_skin_tone": user_features.user_skin_tone,
    "user_identity_token": user_features.user_identity_token,
    "item_brand": item_features.item_brand,
    "item_category": item_features.item_category,
    "item_price_bucket": item_features.item_price_bucket,
    "item_ingredient_tags": item_features.item_ingredient_tags,
    "item_identity_token": item_features.item_identity_token,
    "user_review_count_bucket": interaction_features.user_review_count_bucket,
    "item_popularity_bucket": interaction_features.item_popularity_bucket,
    "item_catalogue_context": context_features.item_catalogue_context,
}


def validate_registry(specs: list[FeatureSpec]) -> None:
    """Assert the declaration and the implementation agree in both directions.

    Args:
        specs: Parsed feature specifications.

    Raises:
        ValueError: If a declared feature has no implementation, an
            implemented transform is never declared, or a declared module
            does not match the module the transform actually lives in.
    """
    declared = {spec.transform for spec in specs}
    implemented = set(TRANSFORM_REGISTRY)

    undeclared = implemented - declared
    unimplemented = declared - implemented

    problems: list[str] = []
    if unimplemented:
        problems.append(f"declared but not implemented: {sorted(unimplemented)}")
    if undeclared:
        problems.append(f"implemented but not declared: {sorted(undeclared)}")

    for spec in specs:
        if spec.transform not in TRANSFORM_REGISTRY:
            continue
        module = _MODULES.get(spec.module)
        if module is None or not hasattr(module, spec.transform):
            problems.append(
                f"'{spec.name}' declares module '{spec.module}' but "
                f"'{spec.transform}' is not defined there"
            )

    if problems:
        raise ValueError("Feature registry mismatch -- " + "; ".join(problems))
