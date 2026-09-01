from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd


def _sort_groups(
    features: pd.DataFrame,
    labels: Sequence[int],
    query_ids: Sequence[object],
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    order = np.argsort(np.asarray(query_ids).astype(str), kind="stable")
    sorted_features = features.iloc[order]
    sorted_labels = np.asarray(labels, dtype=np.int32)[order]
    sorted_queries = np.asarray(query_ids)[order]
    _, counts = np.unique(sorted_queries.astype(str), return_counts=True)
    return sorted_features, sorted_labels, counts.astype(int).tolist()


@dataclass
class TrainResult:
    model: lgb.LGBMRanker
    best_iteration: int
    feature_importance: dict[str, float]


def train_ranker(
    train_features: pd.DataFrame,
    train_labels: Sequence[int],
    train_query_ids: Sequence[object],
    validation_features: pd.DataFrame,
    validation_labels: Sequence[int],
    validation_query_ids: Sequence[object],
    parameters: dict[str, Any],
) -> TrainResult:
    train_x, train_y, train_groups = _sort_groups(train_features, train_labels, train_query_ids)
    valid_x, valid_y, valid_groups = _sort_groups(validation_features, validation_labels, validation_query_ids)
    model = lgb.LGBMRanker(
        objective=parameters.get("objective", "lambdarank"),
        metric=parameters.get("metric", "ndcg"),
        n_estimators=int(parameters.get("n_estimators", 350)),
        learning_rate=float(parameters.get("learning_rate", 0.045)),
        num_leaves=int(parameters.get("num_leaves", 31)),
        min_child_samples=int(parameters.get("min_child_samples", 40)),
        subsample=float(parameters.get("subsample", 0.9)),
        colsample_bytree=float(parameters.get("colsample_bytree", 0.9)),
        reg_lambda=float(parameters.get("reg_lambda", 1.0)),
        lambdarank_truncation_level=int(parameters.get("lambdarank_truncation_level", 10)),
        random_state=int(parameters.get("random_state", 20260901)),
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        train_x,
        train_y,
        group=train_groups,
        eval_set=[(valid_x, valid_y)],
        eval_group=[valid_groups],
        eval_at=[10],
        callbacks=[lgb.early_stopping(35, verbose=False), lgb.log_evaluation(0)],
    )
    importance = {
        name: float(value)
        for name, value in zip(train_features.columns, model.booster_.feature_importance("gain"), strict=True)
    }
    total = sum(importance.values()) or 1.0
    importance = {name: value / total for name, value in sorted(importance.items(), key=lambda item: -item[1])}
    return TrainResult(model, int(model.best_iteration_ or parameters.get("n_estimators", 350)), importance)


def train_final_ranker(
    features: pd.DataFrame,
    labels: Sequence[int],
    query_ids: Sequence[object],
    parameters: dict[str, Any],
    n_estimators: int,
) -> TrainResult:
    """Refit the selected ranker on train+validation without touching test labels."""
    final_x, final_y, final_groups = _sort_groups(features, labels, query_ids)
    model = lgb.LGBMRanker(
        objective=parameters.get("objective", "lambdarank"),
        metric=parameters.get("metric", "ndcg"),
        n_estimators=int(n_estimators),
        learning_rate=float(parameters.get("learning_rate", 0.045)),
        num_leaves=int(parameters.get("num_leaves", 31)),
        min_child_samples=int(parameters.get("min_child_samples", 40)),
        subsample=float(parameters.get("subsample", 0.9)),
        colsample_bytree=float(parameters.get("colsample_bytree", 0.9)),
        reg_lambda=float(parameters.get("reg_lambda", 1.0)),
        lambdarank_truncation_level=int(parameters.get("lambdarank_truncation_level", 10)),
        random_state=int(parameters.get("random_state", 20260901)),
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(final_x, final_y, group=final_groups)
    importance = {
        name: float(value)
        for name, value in zip(features.columns, model.booster_.feature_importance("gain"), strict=True)
    }
    total = sum(importance.values()) or 1.0
    importance = {
        name: value / total
        for name, value in sorted(importance.items(), key=lambda item: -item[1])
    }
    return TrainResult(model, int(n_estimators), importance)
