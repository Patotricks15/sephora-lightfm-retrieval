"""Serving-path tests: request parsing and the topn/feature-fallback helpers."""

from __future__ import annotations

import numpy as np
import pytest

from recommender.models import HybridWARPRecommender
from recommender.serving.handler import parse_request
from recommender.serving.topn import build_topn_payloads, score_new_user, to_dataframe


def test_parse_request_reads_query_string_parameters() -> None:
    user_id, k = parse_request({"queryStringParameters": {"user_id": "U000042", "k": "5"}})
    assert user_id == "U000042"
    assert k == 5


def test_parse_request_defaults_k() -> None:
    _, k = parse_request({"queryStringParameters": {"user_id": "U1"}})
    assert k > 0


def test_parse_request_rejects_missing_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        parse_request({"queryStringParameters": {}})


def test_parse_request_rejects_out_of_range_k() -> None:
    with pytest.raises(ValueError, match="k"):
        parse_request({"queryStringParameters": {"user_id": "U1", "k": "999"}})


def test_build_topn_payloads_shape(cold_split, feature_store, interactions) -> None:
    model = HybridWARPRecommender(n_factors=8, n_epochs=3, max_sampled=5, random_state=0)
    model.fit(cold_split, feature_store.user_matrix, feature_store.item_matrix)

    payloads = build_topn_payloads(
        model,
        cold_split,
        interactions.item_ids,
        interactions.user_ids,
        k=5,
        users=range(10),
    )
    assert len(payloads) == 10
    for payload in payloads:
        assert len(payload["recommendations"]) == 5
        assert payload["recommendations"][0]["rank"] == 1

    frame = to_dataframe(payloads)
    assert len(frame) == 50


def test_score_new_user_uses_profile_attributes_only(cold_split, feature_store) -> None:
    model = HybridWARPRecommender(n_factors=8, n_epochs=3, max_sampled=5, random_state=0)
    model.fit(cold_split, feature_store.user_matrix, feature_store.item_matrix)

    scores = score_new_user(
        model, "Oily", "Deep", feature_store.item_matrix, feature_store.user_vocab, k=10
    )
    assert scores.shape[0] == cold_split.n_items
    assert np.isfinite(scores).all()


def test_score_new_user_falls_back_for_unknown_attributes(cold_split, feature_store) -> None:
    model = HybridWARPRecommender(n_factors=8, n_epochs=3, max_sampled=5, random_state=0)
    model.fit(cold_split, feature_store.user_matrix, feature_store.item_matrix)

    scores = score_new_user(
        model, "NotARealSkinType", "NotARealTone", feature_store.item_matrix,
        feature_store.user_vocab, k=10,
    )
    _, bias = model.item_representations()
    np.testing.assert_allclose(scores, bias)
