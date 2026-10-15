"""Non-personalized baselines that every method must beat to justify its complexity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.splitting import ColdStartSplit, Split


@dataclass
class PopularityRecommender:
    """Recommends the globally most-reviewed products to everyone.

    This is also the cold-start fallback an ID-only system is reduced to: it
    has no way to personalize for a user or item it never trained on, so the
    popularity list is the best it can do. Reported explicitly in the
    cold-start table rather than left implicit.
    """

    item_scores: np.ndarray | None = None
    n_users: int = 0
    n_items: int = 0

    def fit(self, split: Split | ColdStartSplit) -> PopularityRecommender:
        self.n_users = split.n_users
        self.n_items = split.n_items

        counts = np.zeros(self.n_items, dtype=np.float64)
        _, items = split.train_matrix_entries()
        np.add.at(counts, items, 1.0)

        self.item_scores = counts
        return self

    def score_user(self, user_index: int) -> np.ndarray:  # noqa: ARG002
        if self.item_scores is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        return self.item_scores

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray:
        scores = self.score_user(user_index)
        if exclude:
            scores = scores.copy()
            scores[list(exclude)] = -np.inf

        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])]


@dataclass
class RandomRecommender:
    """Uniform random ordering; the floor every other model must clear."""

    random_state: int = 0
    n_users: int = 0
    n_items: int = 0

    def fit(self, split: Split | ColdStartSplit) -> RandomRecommender:
        self.n_users = split.n_users
        self.n_items = split.n_items
        return self

    def score_user(self, user_index: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state + int(user_index))
        return rng.random(self.n_items)

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray:
        scores = self.score_user(user_index)
        if exclude:
            scores = scores.copy()
            scores[list(exclude)] = -np.inf

        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])]
