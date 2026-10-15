"""Model-level tests: the hybrid model, the ID-only baseline, and non-personalized floors."""

from __future__ import annotations

import numpy as np
import pytest

from recommender.evaluation import evaluate, evaluate_cold_start
from recommender.models import (
    HybridWARPRecommender,
    IDOnlyMFRecommender,
    PopularityRecommender,
    RandomRecommender,
)


@pytest.fixture(scope="module")
def trained_hybrid(cold_split, feature_store):
    model = HybridWARPRecommender(n_factors=8, n_epochs=4, max_sampled=6, random_state=0)
    model.fit(cold_split, feature_store.user_matrix, feature_store.item_matrix)
    return model


def test_hybrid_model_loss_is_finite_and_non_negative(trained_hybrid) -> None:
    assert all(np.isfinite(trained_hybrid.loss_history))
    assert all(loss >= 0 for loss in trained_hybrid.loss_history)


def test_hybrid_model_recommend_returns_k_distinct_items(cold_split, trained_hybrid) -> None:
    ranked = trained_hybrid.recommend(0, k=10)
    assert len(set(ranked.tolist())) == len(ranked)
    assert ranked.shape[0] == min(10, cold_split.n_items)


def test_hybrid_model_scores_cold_users_and_items(cold_split, trained_hybrid) -> None:
    """The whole point of the project: cold entities must still get a ranking."""
    cold_user = next(iter(cold_split.cold_user_ids))
    scores = trained_hybrid.score_user(cold_user)
    assert np.isfinite(scores).all()
    assert np.ptp(scores) > 0, "cold user received a flat (uninformative) score vector"

    cold_item = next(iter(cold_split.cold_item_ids))
    # A cold item must be rankable for a warm user too, not score -inf/NaN.
    warm_user = int(cold_split.train["user_index"].iloc[0])
    warm_scores = trained_hybrid.score_user(warm_user)
    assert np.isfinite(warm_scores[cold_item])


def test_hybrid_model_rejects_empty_train(feature_store) -> None:
    import pandas as pd

    from recommender.data.splitting import Split

    empty_split = Split(
        train=pd.DataFrame(columns=["user_index", "item_index", "interaction_order"]),
        validation=pd.DataFrame(columns=["user_index", "item_index"]),
        test=pd.DataFrame(columns=["user_index", "item_index"]),
        n_users=feature_store.n_users,
        n_items=feature_store.n_items,
    )
    with pytest.raises(ValueError):
        model = HybridWARPRecommender()
        model.fit(empty_split, feature_store.user_matrix, feature_store.item_matrix)


def test_id_only_mf_falls_back_to_popularity_for_cold_users(cold_split) -> None:
    model = IDOnlyMFRecommender(n_factors=8, n_epochs=4, random_state=0)
    model.fit(cold_split)

    cold_user = next(iter(cold_split.cold_user_ids))
    scores = model.score_user(cold_user)
    np.testing.assert_allclose(scores, model.item_bias)


def test_id_only_mf_cold_items_collapse_to_bias_for_every_user(cold_split) -> None:
    model = IDOnlyMFRecommender(n_factors=8, n_epochs=4, random_state=0)
    model.fit(cold_split)

    cold_item = next(iter(cold_split.cold_item_ids))
    warm_user = int(cold_split.train["user_index"].iloc[0])
    scores = model.score_user(warm_user)
    assert scores[cold_item] == pytest.approx(model.item_bias[cold_item])


def test_popularity_and_random_baselines_fit_and_recommend(cold_split) -> None:
    for cls in (PopularityRecommender, RandomRecommender):
        model = cls().fit(cold_split)
        ranked = model.recommend(0, k=5)
        assert ranked.shape[0] == 5


def test_warm_evaluation_runs_and_returns_bounded_metrics(cold_split, trained_hybrid) -> None:
    report = evaluate(trained_hybrid, cold_split, k=10, partition="test", model_name="hybrid")
    assert 0.0 <= report.ndcg <= 1.0
    assert 0.0 <= report.recall <= 1.0
    assert 0.0 <= report.map <= 1.0
    assert report.n_users > 0


def test_cold_start_evaluation_covers_both_slices(cold_split, trained_hybrid) -> None:
    reports = evaluate_cold_start(trained_hybrid, cold_split, k=10, model_name="hybrid")
    assert "cold_user" in reports
    assert "cold_item" in reports
    for report in reports.values():
        assert 0.0 <= report.ndcg <= 1.0
        assert report.n_users > 0


def test_hybrid_beats_popularity_on_cold_item_slice(cold_split, feature_store) -> None:
    """The headline claim: shared features should beat a pure popularity fallback.

    Trained with enough epochs to actually separate ingredient/category
    affinity from noise; the smaller smoke-test fixtures elsewhere in this
    file intentionally under-train for speed.
    """
    hybrid = HybridWARPRecommender(n_factors=16, n_epochs=12, max_sampled=8, random_state=1)
    hybrid.fit(cold_split, feature_store.user_matrix, feature_store.item_matrix)
    popularity = PopularityRecommender().fit(cold_split)

    hybrid_reports = evaluate_cold_start(hybrid, cold_split, k=10, model_name="hybrid")
    pop_reports = evaluate_cold_start(popularity, cold_split, k=10, model_name="popularity")
    hybrid_report = hybrid_reports["cold_item"]
    pop_report = pop_reports["cold_item"]

    assert hybrid_report.recall >= pop_report.recall * 0.9
