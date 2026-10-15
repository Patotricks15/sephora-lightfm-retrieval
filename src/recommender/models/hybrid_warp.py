"""Hybrid feature-aware matrix factorization trained with a WARP ranking loss.

This is the LightFM algorithm (Kula, 2015) implemented from scratch in plain
NumPy/SciPy -- classical linear-algebra SGD over feature embeddings, no
hidden layers, no backpropagation through nonlinear activations, and no
dependency on the ``lightfm`` PyPI package.

Representations are feature sums, not per-identity rows:

    p_u = sum_{f in F_u} e_f        (user representation)
    q_i = sum_{g in F_i} e_g        (item representation)
    s(u, i) = <p_u, q_i> + b_i      (score; see note on user bias below)

``F_u``/``F_i`` are read directly from the ``user_matrix``/``item_matrix``
sparse multi-hot encodings built by :mod:`recommender.features.pipeline`, so
a user or item that was never part of training can still be scored, as long
as the matrix has a row for it (which it always does: :class:`FeatureStore`
is built over every user/item index, cold or warm; only the *identity*
feature column is restricted to users/items seen in training).

WARP loss (Weighted Approximate-Rank Pairwise): for a positive pair (u, i),
negatives j are sampled until ``s(u, j) > s(u, i) - 1`` is found (margin
violation) or ``max_sampled`` trials are exhausted. The number of trials N
implies a rank estimate ``rank(i) ~= floor((n_items - 1) / N)``: few trials
mean many items already violate the margin, i.e. i is ranked low; many
trials mean i is already hard to beat, i.e. ranked high. The hinge loss is
weighted by ``L(rank) = sum_{r=1}^{rank} 1/r`` so that correcting a badly
mis-ranked positive contributes much more gradient than nudging an
already-well-ranked one.

A per-user bias is intentionally omitted: it shifts every item's score for
that user by the same constant and therefore cancels out of the pairwise
margin ``s(u,i) - s(u,j)``, so it carries zero gradient under this loss
(the same argument the sibling BPR project makes for its own user bias).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from recommender.data.splitting import ColdStartSplit, Split


def _rows(matrix: sparse.csr_matrix, index: int) -> np.ndarray:
    """Column indices of the active (1-valued) features for one row."""
    return matrix.indices[matrix.indptr[index] : matrix.indptr[index + 1]]


@dataclass
class HybridWARPRecommender:
    """Hybrid feature-aware matrix factorization with WARP loss.

    Args:
        n_factors: Latent dimensionality ``d``.
        n_epochs: Passes over the training interactions.
        learning_rate: SGD step size.
        reg: L2 penalty applied to every touched embedding/bias.
        max_sampled: Maximum negative-sampling trials per positive (``N`` in
            the rank estimate) before giving up and using the last sample.
        init_scale: Standard deviation of the Gaussian initialization.
        lr_decay: Multiplicative decay applied to the step size each epoch.
        max_embedding_norm: Every touched feature embedding is rescaled back
            to this L2 norm after its update (LightFM's own implementation
            does the same). The WARP weight can be large right after a badly
            mis-ranked positive is found, and without this constraint the
            per-feature gradient steps compound across epochs and diverge --
            which is what happens if this is removed.
        random_state: Seed controlling initialization and sampling.
    """

    n_factors: int = 24
    n_epochs: int = 20
    learning_rate: float = 0.05
    reg: float = 0.0005
    max_sampled: int = 10
    init_scale: float = 0.05
    lr_decay: float = 0.95
    max_embedding_norm: float = 8.0
    random_state: int = 42

    user_embeddings: np.ndarray | None = None
    item_embeddings: np.ndarray | None = None
    item_bias: np.ndarray | None = None
    user_matrix: sparse.csr_matrix | None = None
    item_matrix: sparse.csr_matrix | None = None
    n_users: int = 0
    n_items: int = 0
    loss_history: list[float] = field(default_factory=list)
    _harmonic: np.ndarray | None = None
    _seen_keys: np.ndarray | None = None

    def fit(
        self,
        split: Split | ColdStartSplit,
        user_matrix: sparse.csr_matrix,
        item_matrix: sparse.csr_matrix,
        verbose: bool = False,
    ) -> HybridWARPRecommender:
        """Train on the training partition only.

        Args:
            split: Leakage-safe partition; only ``split.train`` is read.
            user_matrix: ``n_users x n_user_features`` multi-hot encoding
                (covers every user, cold or warm).
            item_matrix: ``n_items x n_item_features`` multi-hot encoding
                (covers every item, cold or warm).
            verbose: Print mean loss per epoch.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If the training partition is empty.
        """
        users, items = split.train_matrix_entries()
        if users.size == 0:
            raise ValueError("Cannot fit the hybrid model on an empty training partition.")

        self.n_users = split.n_users
        self.n_items = split.n_items
        self.user_matrix = user_matrix.tocsr()
        self.item_matrix = item_matrix.tocsr()

        rng = np.random.default_rng(self.random_state)
        n_user_features = self.user_matrix.shape[1]
        n_item_features = self.item_matrix.shape[1]
        self.user_embeddings = rng.normal(0, self.init_scale, (n_user_features, self.n_factors))
        self.item_embeddings = rng.normal(0, self.init_scale, (n_item_features, self.n_factors))
        self.item_bias = np.zeros(n_item_features)

        self._harmonic = np.cumsum(1.0 / np.arange(1, self.n_items + 1))

        self._seen_keys = np.sort(users.astype(np.int64) * self.n_items + items.astype(np.int64))
        self.loss_history = []

        n_samples = users.shape[0]
        lr = self.learning_rate
        for epoch in range(self.n_epochs):
            order = rng.permutation(n_samples)
            epoch_loss = 0.0

            for sample_idx in order:
                u = int(users[sample_idx])
                i = int(items[sample_idx])
                epoch_loss += self._warp_step(rng, u, i, lr)

            mean_loss = epoch_loss / n_samples
            self.loss_history.append(mean_loss)
            if verbose:
                print(f"  epoch {epoch + 1:3d}/{self.n_epochs}  loss={mean_loss:.5f}  lr={lr:.4f}")
            lr *= self.lr_decay

        return self

    def _entity_vector(
        self, matrix: sparse.csr_matrix, embeddings: np.ndarray, index: int
    ) -> np.ndarray:
        idx = _rows(matrix, index)
        if idx.size == 0:
            return np.zeros(embeddings.shape[1])
        return embeddings[idx].sum(axis=0)

    def _entity_bias(self, matrix: sparse.csr_matrix, bias: np.ndarray, index: int) -> float:
        idx = _rows(matrix, index)
        return float(bias[idx].sum()) if idx.size else 0.0

    def _score(self, user_vec: np.ndarray, item_index: int) -> float:
        q = self._entity_vector(self.item_matrix, self.item_embeddings, item_index)
        b = self._entity_bias(self.item_matrix, self.item_bias, item_index)
        return float(user_vec @ q) + b

    def _sample_negative(self, rng: np.random.Generator, user: int) -> int:
        return int(rng.integers(0, self.n_items))

    def _is_positive(self, user: int, item: int) -> bool:
        key = user * self.n_items + item
        pos = np.searchsorted(self._seen_keys, key)
        return pos < self._seen_keys.shape[0] and self._seen_keys[pos] == key

    def _warp_step(self, rng: np.random.Generator, u: int, i: int, lr: float) -> float:
        """One WARP update for a single positive pair; returns the hinge loss."""
        user_idx = _rows(self.user_matrix, u)
        pos_item_idx = _rows(self.item_matrix, i)
        p_u = (
            self.user_embeddings[user_idx].sum(axis=0)
            if user_idx.size
            else np.zeros(self.n_factors)
        )
        s_ui = self._score(p_u, i)

        negative_item = None
        trials = 0
        for trials in range(1, self.max_sampled + 1):  # noqa: B007
            candidate = self._sample_negative(rng, u)
            if self._is_positive(u, candidate):
                continue
            s_uj = self._score(p_u, candidate)
            if s_uj > s_ui - 1.0:
                negative_item = candidate
                break
        if negative_item is None:
            negative_item = candidate  # last sample, even if it wasn't a violation

        rank = max(1, (self.n_items - 1) // trials)
        weight = self._harmonic[min(rank, self.n_items) - 1]

        neg_item_idx = _rows(self.item_matrix, negative_item)
        q_i = (
            self.item_embeddings[pos_item_idx].sum(axis=0)
            if pos_item_idx.size
            else np.zeros(self.n_factors)
        )
        q_j = (
            self.item_embeddings[neg_item_idx].sum(axis=0)
            if neg_item_idx.size
            else np.zeros(self.n_factors)
        )
        s_uj = self._score(p_u, negative_item)

        margin = 1.0 - s_ui + s_uj
        if margin <= 0:
            return 0.0

        margin = min(margin, 10.0)  # bound the hinge so one bad triplet can't blow up the step
        loss = weight * margin

        # Gradients derived in the module docstring's score equations.
        if user_idx.size:
            delta_u = lr * weight * (q_i - q_j) - lr * self.reg * self.user_embeddings[user_idx]
            self.user_embeddings[user_idx] += delta_u
            self._clip_rows(self.user_embeddings, user_idx)
        if pos_item_idx.size:
            delta_i = lr * weight * p_u - lr * self.reg * self.item_embeddings[pos_item_idx]
            self.item_embeddings[pos_item_idx] += delta_i
            bias_reg = lr * self.reg * self.item_bias[pos_item_idx]
            self.item_bias[pos_item_idx] += lr * weight - bias_reg
            self._clip_rows(self.item_embeddings, pos_item_idx)
        if neg_item_idx.size:
            delta_j = -lr * weight * p_u - lr * self.reg * self.item_embeddings[neg_item_idx]
            self.item_embeddings[neg_item_idx] += delta_j
            bias_reg_j = lr * self.reg * self.item_bias[neg_item_idx]
            self.item_bias[neg_item_idx] += -lr * weight - bias_reg_j
            self._clip_rows(self.item_embeddings, neg_item_idx)

        return loss

    def _clip_rows(self, embeddings: np.ndarray, rows: np.ndarray) -> None:
        """Rescale touched embedding rows back under ``max_embedding_norm``."""
        norms = np.linalg.norm(embeddings[rows], axis=1)
        too_big = norms > self.max_embedding_norm
        if too_big.any():
            scale = self.max_embedding_norm / norms[too_big]
            embeddings[rows[too_big]] *= scale[:, None]

    def item_representations(self) -> tuple[np.ndarray, np.ndarray]:
        """Dense ``(n_items, n_factors)`` representations and bias vector."""
        self._check_fitted()
        q = self.item_matrix @ self.item_embeddings
        b = self.item_matrix @ self.item_bias
        return np.asarray(q), np.asarray(b)

    def score_user(self, user_index: int) -> np.ndarray:
        """Return scores over the full catalogue for one user (warm or cold)."""
        self._check_fitted()
        p_u = self._entity_vector(self.user_matrix, self.user_embeddings, user_index)
        q, b = self.item_representations()
        return q @ p_u + b

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray:
        """Return the top-``k`` item indices for a user."""
        scores = self.score_user(user_index)
        if exclude:
            scores = scores.copy()
            scores[list(exclude)] = -np.inf

        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])]

    def _check_fitted(self) -> None:
        if self.user_embeddings is None or self.item_embeddings is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
