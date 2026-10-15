"""Leakage tests.

These are the most important tests in the repository. Here the features are
not auxiliary signal, they are the model's representation, so a leak here
directly inflates every reported metric including the cold-start comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.data import build_implicit_feedback
from recommender.features import interaction_features, item_features, user_features


def test_future_review_does_not_change_past_review_count(split) -> None:
    """Injecting a review after the cut-off must not change a user's bucket.

    Direct operationalization of the point-in-time rule: the aggregate
    computed at cut-off C must be identical whether or not rows at
    interaction_order >= C exist in the frame.
    """
    train = split.train
    cutoff = int(train["interaction_order"].max())
    edges = [3, 8, 16]

    before = interaction_features.user_review_count_bucket(train, edges, cutoff_order=cutoff)

    victim = int(train["user_index"].iloc[0])
    future_row = train.iloc[[0]].copy()
    future_row["user_index"] = victim
    future_row["interaction_order"] = cutoff + 50
    polluted = pd.concat([train, future_row], ignore_index=True)

    after = interaction_features.user_review_count_bucket(polluted, edges, cutoff_order=cutoff)

    pd.testing.assert_series_equal(
        before.sort_index(), after.sort_index(), check_exact=True, check_index_type=False
    )


def test_cutoff_boundary_is_strictly_exclusive(split) -> None:
    """A review exactly at the cut-off belongs to the future, not the past."""
    train = split.train.copy()
    user = int(train["user_index"].iloc[0])
    user_rows = train.loc[train["user_index"] == user]
    cutoff = int(user_rows["interaction_order"].max())

    counts = interaction_features.user_review_count_bucket(train, [3, 8, 16], cutoff_order=cutoff)
    expected_count = int((user_rows["interaction_order"] < cutoff).sum())
    expected_bucket = f"review_count={int(np.digitize([expected_count], [3, 8, 16])[0])}"

    assert counts.get(user) == expected_bucket


def test_item_popularity_is_fitted_on_training_partition_only(split, interactions) -> None:
    """Popularity on train must differ from popularity on the full dataset.

    If identical, the training statistic would already encode the held-out
    interactions and the evaluation would be optimistic.
    """
    edges = [5, 15, 40]
    train_pop = interaction_features.item_popularity_bucket(
        split.train, edges, n_items=split.n_items
    )
    full_pop = interaction_features.item_popularity_bucket(
        interactions.frame, edges, n_items=split.n_items
    )

    assert not train_pop.equals(full_pop), (
        "train-only popularity buckets match full-data buckets; fit_on: train_only "
        "is not enforced"
    )


def test_cold_users_never_receive_an_identity_token(cold_split) -> None:
    """The identity vocabulary must be built from train, excluding cold users."""
    identity = user_features.user_identity_token(cold_split.train)

    for cold_user in cold_split.cold_user_ids:
        assert cold_user not in identity.index, (
            f"cold user {cold_user} has an identity token; the hybrid model "
            "could trivially memorize them, defeating the cold-start test"
        )


def test_cold_items_never_receive_an_identity_token(cold_split) -> None:
    identity = item_features.item_identity_token(cold_split.train)

    for cold_item in cold_split.cold_item_ids:
        assert cold_item not in identity.index, (
            f"cold item {cold_item} has an identity token; cold-start retrieval "
            "would not actually be exercising the shared-feature pathway"
        )


def test_cold_slice_interactions_are_absent_from_train(cold_split) -> None:
    """No interaction touching a cold user or cold item may appear in train."""
    train_users = set(cold_split.train["user_index"].unique().tolist())
    train_items = set(cold_split.train["item_index"].unique().tolist())

    assert not (train_users & set(cold_split.cold_user_ids))
    assert not (train_items & set(cold_split.cold_item_ids))


def test_warm_split_partitions_are_disjoint(split) -> None:
    def pairs(frame: pd.DataFrame) -> set[tuple[int, int]]:
        return set(
            zip(
                frame["user_index"].to_numpy().tolist(),
                frame["item_index"].to_numpy().tolist(),
                strict=True,
            )
        )

    train, validation, test = pairs(split.train), pairs(split.validation), pairs(split.test)

    assert not (train & validation)
    assert not (train & test)
    assert not (validation & test)


def test_test_interaction_is_the_users_latest(split) -> None:
    merged = split.test.merge(
        split.train.groupby("user_index")["interaction_order"].max().rename("train_max"),
        on="user_index",
        how="inner",
    )
    assert (merged["interaction_order"] > merged["train_max"]).all()


def test_no_duplicate_interactions_before_splitting(interactions) -> None:
    frame = interactions.frame
    assert not frame.duplicated(subset=["user_index", "item_index"]).any()


def test_shuffled_labels_destroy_ranking_quality(split, interactions) -> None:
    """A model trained on shuffled item assignments must not beat random."""
    from recommender.evaluation import ndcg_at_k
    from recommender.models import HybridWARPRecommender

    rng = np.random.default_rng(0)
    shuffled = split.train.copy()
    shuffled["item_index"] = rng.permutation(shuffled["item_index"].to_numpy())

    broken_split = type(split)(
        train=shuffled,
        validation=split.validation,
        test=split.test,
        n_users=split.n_users,
        n_items=split.n_items,
    )

    user_matrix = _identity_sparse(split.n_users)
    item_matrix = _identity_sparse(split.n_items)
    model = HybridWARPRecommender(n_factors=8, n_epochs=3, random_state=0)
    model.fit(broken_split, user_matrix, item_matrix)

    score = ndcg_at_k(model, broken_split, k=10)
    random_baseline = 10 / split.n_items

    assert score < max(0.08, random_baseline * 6), (
        f"NDCG@10 stayed at {score:.4f} after shuffling labels; a feature is leaking"
    )


def _identity_sparse(n: int):
    from scipy import sparse

    return sparse.identity(n, format="csr", dtype="float32")


def test_core_filter_rejects_impossible_thresholds(raw_reviews) -> None:
    with pytest.raises(ValueError):
        build_implicit_feedback(
            raw_reviews, min_user_interactions=10_000, min_item_interactions=10_000
        )
