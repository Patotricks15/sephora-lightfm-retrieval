"""Schema validation tests for the declarative feature specification."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from recommender.features.registry import TRANSFORM_REGISTRY, validate_registry
from recommender.features.spec import FeatureSpec, load_feature_specs


def test_default_spec_loads_and_validates() -> None:
    specs = load_feature_specs()
    assert len(specs) == len(TRANSFORM_REGISTRY)
    validate_registry(specs)  # must not raise


def test_every_family_is_represented() -> None:
    specs = load_feature_specs()
    families = {spec.family for spec in specs}
    assert families == {"user", "item", "interaction", "context"}


def test_every_feature_documents_a_rationale() -> None:
    for spec in load_feature_specs():
        assert spec.rationale.strip(), f"{spec.name} has no rationale"


def test_identity_features_must_be_train_only() -> None:
    with pytest.raises(ValueError, match="train_only"):
        FeatureSpec(
            name="bad_identity",
            family="user",
            kind="identity",
            module="user_features",
            transform="user_identity_token",
            params={},
            fit_on="catalogue",
            point_in_time=False,
            fallback=None,
            rationale="test",
        )


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="family"):
        FeatureSpec(
            name="bad",
            family="nonsense",
            kind="categorical",
            module="user_features",
            transform="user_skin_type",
            params={},
            fit_on="profile",
            point_in_time=False,
            fallback=None,
            rationale="test",
        )


def test_missing_rationale_is_rejected() -> None:
    with pytest.raises(ValueError, match="rationale"):
        FeatureSpec(
            name="bad",
            family="user",
            kind="categorical",
            module="user_features",
            transform="user_skin_type",
            params={},
            fit_on="profile",
            point_in_time=False,
            fallback=None,
            rationale="   ",
        )


def test_duplicate_feature_name_is_rejected(tmp_path: Path) -> None:
    bad_yaml = textwrap.dedent(
        """
        version: 1
        features:
          - name: dup
            family: user
            kind: categorical
            module: user_features
            transform: user_skin_type
            params: {}
            fit_on: profile
            point_in_time: false
            fallback: "skin_type=unknown"
            rationale: "first"
          - name: dup
            family: user
            kind: categorical
            module: user_features
            transform: user_skin_tone
            params: {}
            fit_on: profile
            point_in_time: false
            fallback: "skin_tone=unknown"
            rationale: "second"
        """
    )
    path = tmp_path / "features.yaml"
    path.write_text(bad_yaml)

    with pytest.raises(ValueError, match="Duplicate feature"):
        load_feature_specs(path)


def test_registry_rejects_unimplemented_declaration(tmp_path: Path) -> None:
    bad_yaml = textwrap.dedent(
        """
        version: 1
        features:
          - name: ghost
            family: user
            kind: categorical
            module: user_features
            transform: does_not_exist
            params: {}
            fit_on: profile
            point_in_time: false
            fallback: null
            rationale: "test"
        """
    )
    path = tmp_path / "features.yaml"
    path.write_text(bad_yaml)

    with pytest.raises(ValueError, match="not implemented"):
        validate_registry(load_feature_specs(path))


def test_feature_store_matrices_match_vocab_dimensions(feature_store) -> None:
    assert feature_store.user_matrix.shape == (feature_store.n_users, len(feature_store.user_vocab))
    assert feature_store.item_matrix.shape == (feature_store.n_items, len(feature_store.item_vocab))
    assert feature_store.user_matrix.shape[0] == len(feature_store.user_tokens)
    assert feature_store.item_matrix.shape[0] == len(feature_store.item_tokens)


def test_every_user_and_item_has_at_least_one_token(feature_store) -> None:
    assert all(len(tokens) >= 1 for tokens in feature_store.user_tokens)
    assert all(len(tokens) >= 1 for tokens in feature_store.item_tokens)
