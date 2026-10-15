"""Leave-last-one-out warm split, plus a dedicated cold-start holdout slice.

Two different notions of "holdout" are implemented and must not be confused:

``train`` / ``validation`` / ``test``
    The ordinary warm-start protocol: each *warm* user's last interaction goes
    to test, second-to-last to validation, the rest to train. Point-in-time
    correctness (see ``recommender.features``) is defined against this split.

``cold_user_eval`` / ``cold_item_eval`` / ``cold_both_eval``
    A disjoint population of users and items is withheld from ``train``
    *entirely* -- not just their last interaction, every one of them. The
    hybrid model has never seen their identity, so it can only score them
    through shared features (skin type/tone, brand/category/ingredients). An
    ID-only baseline has literally no row/column for them and can only fall
    back to popularity. This is the comparison the whole project is built to
    make honestly, so the split that produces it lives next to the ordinary
    one rather than being bolted on later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.data.loader import InteractionData


@dataclass(frozen=True)
class Split:
    """A leakage-safe warm-start partition of the interaction data."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int

    def train_matrix_entries(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(user_index, item_index)`` arrays of the train split."""
        return (
            self.train["user_index"].to_numpy(dtype=np.int64),
            self.train["item_index"].to_numpy(dtype=np.int64),
        )

    def user_positives(self) -> dict[int, set[int]]:
        """Map each user to the items seen during training, for negative sampling."""
        grouped = self.train.groupby("user_index")["item_index"]
        return {int(user): set(map(int, items)) for user, items in grouped}


@dataclass(frozen=True)
class ColdStartSplit:
    """Warm-start split plus a disjoint cold-start evaluation slice.

    Attributes:
        train/validation/test: The warm-start partitions. Interactions
            touching a cold user or a cold item never appear here.
        cold_user_eval: Interactions of cold (never-trained) users against
            warm items -- "new user, known item".
        cold_item_eval: Interactions of warm users against cold (never-
            trained) items -- "known user, new item".
        cold_both_eval: Interactions where both sides are cold -- the hardest
            slice, included for completeness but not required to beat
            popularity by any margin.
        cold_user_ids: Internal user indices withheld from training.
        cold_item_ids: Internal item indices withheld from training.
        n_users/n_items: Global vocabulary sizes, including cold entities
            (feature vectors must still be constructible for them).
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    cold_user_eval: pd.DataFrame
    cold_item_eval: pd.DataFrame
    cold_both_eval: pd.DataFrame
    cold_user_ids: frozenset[int]
    cold_item_ids: frozenset[int]
    n_users: int
    n_items: int

    def warm_split(self) -> Split:
        """Expose the warm-start partitions as a standalone :class:`Split`."""
        return Split(
            train=self.train,
            validation=self.validation,
            test=self.test,
            n_users=self.n_users,
            n_items=self.n_items,
        )

    def train_matrix_entries(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.train["user_index"].to_numpy(dtype=np.int64),
            self.train["item_index"].to_numpy(dtype=np.int64),
        )

    def user_positives(self) -> dict[int, set[int]]:
        grouped = self.train.groupby("user_index")["item_index"]
        return {int(user): set(map(int, items)) for user, items in grouped}


def leave_last_one_out(data: InteractionData, min_train_interactions: int = 3) -> Split:
    """Split per user: last interaction to test, second-to-last to validation.

    Users without enough history to leave at least ``min_train_interactions``
    rows in training contribute all of their interactions to training only,
    so they are never evaluated.

    Args:
        data: Indexed interactions carrying ``interaction_order``.
        min_train_interactions: Rows a user must retain to be evaluated.

    Returns:
        A :class:`Split` whose partitions are disjoint by construction.

    Raises:
        ValueError: If no user qualifies for evaluation.
    """
    return _leave_last_one_out_frame(data.frame, data.n_users, data.n_items, min_train_interactions)


def _leave_last_one_out_frame(
    frame: pd.DataFrame, n_users: int, n_items: int, min_train_interactions: int
) -> Split:
    frame = frame.sort_values(["user_index", "interaction_order"], kind="mergesort")

    position_from_end = frame.groupby("user_index").cumcount(ascending=False)
    user_sizes = frame.groupby("user_index")["item_index"].transform("size")

    evaluable = user_sizes >= (min_train_interactions + 2)

    is_test = evaluable & (position_from_end == 0)
    is_validation = evaluable & (position_from_end == 1)
    is_train = ~(is_test | is_validation)

    test = frame.loc[is_test].copy()
    validation = frame.loc[is_validation].copy()
    train = frame.loc[is_train].copy()

    if test.empty:
        raise ValueError(
            "No user has enough interactions to be evaluated; "
            "lower min_train_interactions or relax the core filter."
        )

    return Split(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        n_users=n_users,
        n_items=n_items,
    )


def make_cold_start_split(
    data: InteractionData,
    cold_user_frac: float = 0.08,
    cold_item_frac: float = 0.08,
    min_train_interactions: int = 3,
    seed: int = 42,
) -> ColdStartSplit:
    """Carve out a cold-start slice, then split the remainder warm-start style.

    Args:
        data: Indexed interactions.
        cold_user_frac: Fraction of users withheld from training entirely.
        cold_item_frac: Fraction of items withheld from training entirely.
        min_train_interactions: Passed through to the warm-start split.
        seed: Controls which users/items are selected as cold.

    Returns:
        A populated :class:`ColdStartSplit`.

    Raises:
        ValueError: If no user/item is left to withhold, or the warm split
            cannot produce an evaluable partition.
    """
    if not 0.0 < cold_user_frac < 0.5 or not 0.0 < cold_item_frac < 0.5:
        raise ValueError("cold_user_frac and cold_item_frac must be in (0, 0.5).")

    rng = np.random.default_rng(seed)
    n_cold_users = max(1, int(round(cold_user_frac * data.n_users)))
    n_cold_items = max(1, int(round(cold_item_frac * data.n_items)))

    cold_user_ids = frozenset(
        int(x) for x in rng.choice(data.n_users, size=n_cold_users, replace=False)
    )
    cold_item_ids = frozenset(
        int(x) for x in rng.choice(data.n_items, size=n_cold_items, replace=False)
    )

    frame = data.frame
    is_cold_user = frame["user_index"].isin(cold_user_ids)
    is_cold_item = frame["item_index"].isin(cold_item_ids)

    warm_pool = frame.loc[~is_cold_user & ~is_cold_item].copy()
    warm_split = _leave_last_one_out_frame(
        warm_pool, data.n_users, data.n_items, min_train_interactions
    )

    cold_user_eval = frame.loc[is_cold_user & ~is_cold_item].reset_index(drop=True)
    cold_item_eval = frame.loc[is_cold_item & ~is_cold_user].reset_index(drop=True)
    cold_both_eval = frame.loc[is_cold_user & is_cold_item].reset_index(drop=True)

    if cold_user_eval.empty or cold_item_eval.empty:
        raise ValueError(
            "Cold-start slice is empty; increase cold_user_frac/cold_item_frac "
            "or the size of the synthetic fixture."
        )

    return ColdStartSplit(
        train=warm_split.train,
        validation=warm_split.validation,
        test=warm_split.test,
        cold_user_eval=cold_user_eval,
        cold_item_eval=cold_item_eval,
        cold_both_eval=cold_both_eval,
        cold_user_ids=cold_user_ids,
        cold_item_ids=cold_item_ids,
        n_users=data.n_users,
        n_items=data.n_items,
    )
