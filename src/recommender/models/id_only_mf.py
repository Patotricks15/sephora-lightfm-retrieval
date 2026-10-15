"""ID-only matrix factorization baseline (BPR-style pairwise loss).

This model factorizes **identities**, exactly like the sibling portfolio
projects: a user is a row index, an item is a column index. It is the
contrast case for the whole project -- a user or item absent from training
has no row/column at all, so :meth:`score_user` falls back to the item bias
(a pure popularity signal) for anyone outside the fitted vocabulary. That
fallback is deliberate: it is what "no cold-start handling" looks like in
practice, and the cold-start evaluation slice exists specifically to quantify
how much worse it is than the hybrid model's feature-based retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.splitting import ColdStartSplit, Split


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        np.exp(np.clip(x, -500, 500)) / (1.0 + np.exp(np.clip(x, -500, 500))),
    )


@dataclass
class IDOnlyMFRecommender:
    """Bayesian Personalized Ranking matrix factorization over raw IDs.

    Args:
        n_factors: Latent dimensionality.
        n_epochs: Passes over the training interactions.
        learning_rate: SGD step size.
        reg_user: L2 penalty on user factors.
        reg_item: L2 penalty on item factors/bias.
        batch_size: Triplets per vectorized update.
        lr_decay: Multiplicative decay applied per epoch.
        random_state: Seed controlling initialization and sampling.
    """

    n_factors: int = 24
    n_epochs: int = 40
    learning_rate: float = 0.08
    reg_user: float = 0.01
    reg_item: float = 0.01
    batch_size: int = 2048
    lr_decay: float = 0.95
    random_state: int = 42

    user_factors: np.ndarray | None = None
    item_factors: np.ndarray | None = None
    item_bias: np.ndarray | None = None
    n_users: int = 0
    n_items: int = 0
    trained_users: frozenset[int] | None = None
    trained_items: frozenset[int] | None = None
    loss_history: list[float] | None = None
    _seen_keys: np.ndarray | None = None

    def fit(self, split: Split | ColdStartSplit) -> IDOnlyMFRecommender:
        users, items = split.train_matrix_entries()
        if users.size == 0:
            raise ValueError("Cannot fit on an empty training partition.")

        self.n_users = split.n_users
        self.n_items = split.n_items
        self.trained_users = frozenset(int(u) for u in np.unique(users))
        self.trained_items = frozenset(int(i) for i in np.unique(items))

        rng = np.random.default_rng(self.random_state)
        self.user_factors = rng.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (self.n_items, self.n_factors))
        self.item_bias = np.zeros(self.n_items)

        self._seen_keys = np.sort(users.astype(np.int64) * self.n_items + items.astype(np.int64))
        self.loss_history = []

        n_samples = users.shape[0]
        lr = self.learning_rate
        for _epoch in range(self.n_epochs):
            order = rng.permutation(n_samples)
            epoch_loss = 0.0

            for start in range(0, n_samples, self.batch_size):
                batch = order[start : start + self.batch_size]
                u = users[batch]
                i = items[batch]
                j = self._sample_negatives(rng, u)

                pu = self.user_factors[u]
                qi = self.item_factors[i]
                qj = self.item_factors[j]

                x_uij = np.einsum("bf,bf->b", pu, qi - qj) + self.item_bias[i] - self.item_bias[j]
                grad = _sigmoid(-x_uij)[:, None]
                epoch_loss += float(-np.log(np.maximum(_sigmoid(x_uij), 1e-12)).sum())

                grad_u = grad * (qi - qj) - self.reg_user * pu
                grad_i = grad * pu - self.reg_item * qi
                grad_j = -grad * pu - self.reg_item * qj
                grad_bi = grad[:, 0] - self.reg_item * self.item_bias[i]
                grad_bj = -grad[:, 0] - self.reg_item * self.item_bias[j]

                np.add.at(self.user_factors, u, lr * grad_u)
                np.add.at(self.item_factors, i, lr * grad_i)
                np.add.at(self.item_factors, j, lr * grad_j)
                np.add.at(self.item_bias, i, lr * grad_bi)
                np.add.at(self.item_bias, j, lr * grad_bj)

            mean_loss = epoch_loss / n_samples
            self.loss_history.append(mean_loss)
            lr *= self.lr_decay

        return self

    def _sample_negatives(self, rng: np.random.Generator, users: np.ndarray) -> np.ndarray:
        candidates = rng.integers(0, self.n_items, size=users.shape[0])
        for _ in range(16):
            keys = users.astype(np.int64) * self.n_items + candidates
            pos = np.searchsorted(self._seen_keys, keys)
            pos = np.minimum(pos, self._seen_keys.shape[0] - 1)
            collides = self._seen_keys[pos] == keys
            if not collides.any():
                break
            candidates[collides] = rng.integers(0, self.n_items, size=int(collides.sum()))
        return candidates

    def score_user(self, user_index: int) -> np.ndarray:
        """Return scores over the catalogue.

        Cold users fall back entirely to the item bias (popularity). Cold
        items are masked out of the dot product for every user, warm or not:
        an item that was never trained has a random, meaningless factor
        vector, so only its (also-untrained, zero) bias would be honest --
        in practice that collapses every cold item to the same score, which
        is exactly the "no personalisation available" fallback this baseline
        is supposed to represent.
        """
        self._check_fitted()
        untrained_items = (
            np.array(
                [i not in self.trained_items for i in range(self.n_items)], dtype=bool
            )
            if self.trained_items is not None
            else np.zeros(self.n_items, dtype=bool)
        )

        if self.trained_users is not None and user_index not in self.trained_users:
            # No row was ever trained for this user: the only signal left is
            # the learned popularity prior, i.e. the item bias.
            scores = self.item_bias.copy()
        else:
            scores = self.item_factors @ self.user_factors[user_index] + self.item_bias

        scores[untrained_items] = self.item_bias[untrained_items]
        return scores

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

    def _check_fitted(self) -> None:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
