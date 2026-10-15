"""Ranking metrics, including a dedicated cold-start evaluation protocol.

Every metric follows the same contract: for an evaluated user, rank the full
catalogue excluding their training history, then measure where their
held-out relevant item(s) land. The warm-start partitions
(``train``/``validation``/``test``) are leave-last-one-out, so each warm user
has exactly one relevant item. The cold-start slices can have several
relevant items per cold entity (every interaction a cold user/item has is
withheld from training), so the metrics are implemented for the general
multi-positive case and simply specialize correctly when there is only one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from recommender.data.splitting import ColdStartSplit, Split


class Recommender(Protocol):
    """Structural interface shared by every model in this repository."""

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class EvaluationReport:
    """Metrics for one model on one partition or slice."""

    model: str
    slice_name: str
    k: int
    ndcg: float
    recall: float
    map: float
    hit_rate: float
    coverage: float
    n_users: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "slice": self.slice_name,
            "k": self.k,
            "ndcg": round(self.ndcg, 4),
            "recall": round(self.recall, 4),
            "map": round(self.map, 4),
            "hit_rate": round(self.hit_rate, 4),
            "coverage": round(self.coverage, 4),
            "n_users": self.n_users,
        }


def _history_by_user(train: pd.DataFrame) -> dict[int, set[int]]:
    grouped = train.groupby("user_index")["item_index"]
    return {int(user): set(map(int, items)) for user, items in grouped}


def _relevant_by_user(frame: pd.DataFrame) -> dict[int, set[int]]:
    grouped = frame.groupby("user_index")["item_index"]
    return {int(user): set(map(int, items)) for user, items in grouped}


def _average_precision(ranked: np.ndarray, relevant: set[int]) -> float:
    hits = 0
    precision_sum = 0.0
    for position, item in enumerate(ranked, start=1):
        if int(item) in relevant:
            hits += 1
            precision_sum += hits / position
    denom = min(len(relevant), ranked.shape[0])
    return precision_sum / denom if denom else 0.0


def _evaluate_users(
    model: Recommender,
    relevant: Mapping[int, set[int]],
    history: Mapping[int, set[int]],
    n_items: int,
    k: int,
) -> dict[str, float]:
    ndcg_total = 0.0
    recall_total = 0.0
    map_total = 0.0
    hits = 0
    recommended_items: set[int] = set()

    users = list(relevant)
    for user in users:
        targets = relevant[user]
        ranked = model.recommend(user, k=k, exclude=history.get(user, set()))
        recommended_items.update(int(i) for i in ranked)

        hit_positions = [pos for pos, item in enumerate(ranked) if int(item) in targets]
        if hit_positions:
            hits += 1
        ndcg_total += sum(1.0 / np.log2(pos + 2) for pos in hit_positions) / _ideal_dcg(
            min(len(targets), k)
        )
        recall_total += len(hit_positions) / len(targets)
        map_total += _average_precision(ranked, targets)

    n_users = len(users)
    return {
        "ndcg": ndcg_total / n_users,
        "recall": recall_total / n_users,
        "map": map_total / n_users,
        "hit_rate": hits / n_users,
        "coverage": len(recommended_items) / n_items,
        "n_users": n_users,
    }


def _ideal_dcg(n_relevant: int) -> float:
    if n_relevant <= 0:
        return 1.0
    return float(sum(1.0 / np.log2(pos + 2) for pos in range(n_relevant)))


def evaluate(
    model: Recommender,
    split: Split | ColdStartSplit,
    k: int = 10,
    partition: str = "test",
    model_name: str = "model",
) -> EvaluationReport:
    """Compute ranking metrics for a model on a warm-start partition.

    Args:
        model: Any object exposing ``recommend``.
        split: The partition container.
        k: Cut-off.
        partition: ``"validation"`` or ``"test"``.
        model_name: Label used in the results table.

    Returns:
        A populated :class:`EvaluationReport`.

    Raises:
        ValueError: If the partition name is unknown or empty.
    """
    if partition not in {"validation", "test"}:
        raise ValueError(f"Unknown partition '{partition}'; expected 'validation' or 'test'.")
    frame: pd.DataFrame = getattr(split, partition)
    if frame.empty:
        raise ValueError(f"Partition '{partition}' is empty.")

    history = _history_by_user(split.train)
    relevant = _relevant_by_user(frame)
    metrics = _evaluate_users(model, relevant, history, split.n_items, k)

    return EvaluationReport(model=model_name, slice_name=partition, k=k, **metrics)


def evaluate_cold_start(
    model: Recommender,
    cold_split: ColdStartSplit,
    k: int = 10,
    model_name: str = "model",
) -> dict[str, EvaluationReport]:
    """Evaluate a model on both cold-start slices.

    Cold entities have no training history by construction, so the
    "exclude" set passed to ``recommend`` is always empty -- there is nothing
    to hide, which is exactly the point of the slice.

    Args:
        model: Any object exposing ``recommend``.
        cold_split: A :class:`ColdStartSplit`.
        k: Cut-off.
        model_name: Label used in the results table.

    Returns:
        ``{"cold_user": report, "cold_item": report, "cold_both": report}``,
        omitting any slice that is empty.
    """
    reports: dict[str, EvaluationReport] = {}
    empty_history: dict[int, set[int]] = {}

    for slice_name, frame in (
        ("cold_user", cold_split.cold_user_eval),
        ("cold_item", cold_split.cold_item_eval),
        ("cold_both", cold_split.cold_both_eval),
    ):
        if frame.empty:
            continue
        relevant = _relevant_by_user(frame)
        metrics = _evaluate_users(model, relevant, empty_history, cold_split.n_items, k)
        reports[slice_name] = EvaluationReport(
            model=model_name, slice_name=slice_name, k=k, **metrics
        )

    return reports


def ndcg_at_k(model: Recommender, split: Split, k: int = 10, partition: str = "test") -> float:
    """Convenience wrapper returning only NDCG@K."""
    return evaluate(model, split, k=k, partition=partition).ndcg
