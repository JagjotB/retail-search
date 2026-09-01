from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def dcg_at_k(relevances: Iterable[float], k: int = 10) -> float:
    values = np.asarray(list(relevances), dtype=float)[:k]
    if values.size == 0:
        return 0.0
    gains = np.power(2.0, values) - 1.0
    discounts = np.log2(np.arange(2, values.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances: Iterable[float], k: int = 10) -> float:
    values = np.asarray(list(relevances), dtype=float)
    ideal = dcg_at_k(np.sort(values)[::-1], k)
    return dcg_at_k(values, k) / ideal if ideal > 0 else 0.0


def ndcg_at_k_from_gains(gains: Iterable[float], k: int = 10) -> float:
    """Compute NDCG when the caller supplies gains directly (for TREC compatibility)."""
    values = np.asarray(list(gains), dtype=float)

    def discounted(items: np.ndarray) -> float:
        truncated = items[:k]
        if truncated.size == 0:
            return 0.0
        return float(np.sum(truncated / np.log2(np.arange(2, truncated.size + 2))))

    ideal = discounted(np.sort(values)[::-1])
    return discounted(values) / ideal if ideal > 0 else 0.0


def reciprocal_rank_at_k(relevances: Iterable[float], k: int = 10) -> float:
    for index, relevance in enumerate(list(relevances)[:k], 1):
        if relevance > 0:
            return 1.0 / index
    return 0.0


def recall_at_k(relevances: Iterable[float], k: int) -> float:
    values = np.asarray(list(relevances), dtype=float)
    relevant_total = int(np.sum(values > 0))
    if relevant_total == 0:
        return 0.0
    return float(np.sum(values[:k] > 0) / relevant_total)


def evaluate_rankings(
    frame: pd.DataFrame,
    score_column: str,
    query_column: str = "query_id",
    label_column: str = "relevance_label",
) -> dict[str, Any]:
    required = {query_column, label_column, score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")
    query_metrics: list[dict[str, float | str]] = []
    for query_id, rows in frame.groupby(query_column, sort=False):
        ordered = rows.sort_values(score_column, ascending=False, kind="stable")
        labels = ordered[label_column].to_numpy()
        query_metrics.append(
            {
                "query_id": str(query_id),
                "ndcg_at_10": ndcg_at_k(labels, 10),
                "mrr_at_10": reciprocal_rank_at_k(labels, 10),
                "recall_at_50": recall_at_k(labels, 50),
                "recall_at_100": recall_at_k(labels, 100),
            }
        )
    metrics = pd.DataFrame(query_metrics)
    if metrics.empty:
        raise ValueError("Cannot evaluate an empty query set")
    return {
        "query_count": len(metrics),
        "judgment_count": len(frame),
        **{column: float(metrics[column].mean()) for column in metrics.columns if column != "query_id"},
        "per_query": query_metrics,
    }


def evaluate_gain_ndcg(
    frame: pd.DataFrame,
    score_column: str,
    query_column: str = "query_id",
    gain_column: str = "relevance_gain",
) -> dict[str, Any]:
    required = {query_column, gain_column, score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing gain evaluation columns: {sorted(missing)}")
    per_query = []
    for query_id, rows in frame.groupby(query_column, sort=False):
        ordered = rows.sort_values(score_column, ascending=False, kind="stable")
        per_query.append(
            {
                "query_id": str(query_id),
                "ndcg_at_10": ndcg_at_k_from_gains(ordered[gain_column], 10),
            }
        )
    if not per_query:
        raise ValueError("Cannot evaluate an empty query set")
    return {
        "query_count": len(per_query),
        "judgment_count": len(frame),
        "ndcg_at_10": float(np.mean([row["ndcg_at_10"] for row in per_query])),
        "per_query": per_query,
    }
