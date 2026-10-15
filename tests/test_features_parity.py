"""Offline/online parity tests.

``recompute_user_tokens`` / ``recompute_item_tokens`` are the functions the
serving path calls (see ``src/recommender/serving/handler.py``) to build a
feature vector at request time, independently of the batch-built
:class:`~recommender.features.pipeline.FeatureStore` produced during
training. If the two ever disagree for a user/item that *was* part of
training, the model would be serving a different ranking than the one it was
evaluated on -- the classic training/serving skew bug.
"""

from __future__ import annotations

from recommender.features.pipeline import recompute_item_tokens, recompute_user_tokens


def test_warm_user_tokens_match_between_batch_and_recompute(
    feature_store, cold_split, interactions
) -> None:
    warm_users = sorted(set(cold_split.train["user_index"].unique().tolist()))[:25]

    for user_index in warm_users:
        recomputed = recompute_user_tokens(user_index, interactions.frame, cold_split.train)
        assert sorted(recomputed) == sorted(feature_store.user_tokens[user_index]), (
            f"user {user_index}: offline tokens {feature_store.user_tokens[user_index]} "
            f"!= recomputed tokens {recomputed}"
        )


def test_warm_item_tokens_match_between_batch_and_recompute(
    feature_store, cold_split, interactions, catalogue
) -> None:
    warm_items = sorted(set(cold_split.train["item_index"].unique().tolist()))[:25]

    for item_index in warm_items:
        recomputed = recompute_item_tokens(
            item_index, catalogue, interactions.item_ids, cold_split.train, cold_split.n_items
        )
        assert sorted(recomputed) == sorted(feature_store.item_tokens[item_index]), (
            f"item {item_index}: offline tokens {feature_store.item_tokens[item_index]} "
            f"!= recomputed tokens {recomputed}"
        )


def test_cold_user_recompute_has_no_identity_token(cold_split, interactions) -> None:
    cold_user = next(iter(cold_split.cold_user_ids))
    tokens = recompute_user_tokens(cold_user, interactions.frame, cold_split.train)
    assert not any(tok.startswith("user_id=") for tok in tokens)
    assert any(tok.startswith("skin_type=") for tok in tokens)


def test_cold_item_recompute_has_no_identity_token(cold_split, interactions, catalogue) -> None:
    cold_item = next(iter(cold_split.cold_item_ids))
    tokens = recompute_item_tokens(
        cold_item, catalogue, interactions.item_ids, cold_split.train, cold_split.n_items
    )
    assert not any(tok.startswith("item_id=") for tok in tokens)
    assert any(tok.startswith("brand=") for tok in tokens)
    assert any(tok.startswith("category=") for tok in tokens)


def test_feature_matrix_row_matches_token_list(feature_store) -> None:
    """The sparse matrix row for a user must have exactly its token columns set."""
    user_index = 0
    row = feature_store.user_matrix[user_index].toarray().ravel()
    expected_cols = {
        feature_store.user_vocab[tok]
        for tok in feature_store.user_tokens[user_index]
        if tok in feature_store.user_vocab
    }
    actual_cols = set(row.nonzero()[0].tolist())
    assert actual_cols == expected_cols
