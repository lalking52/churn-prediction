from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


@dataclass
class BusinessConfig:
    retention_cost: float
    saved_revenue: float


def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if not 0 < k <= 1:
        raise ValueError("k must be in (0, 1]")
    n = len(y_true)
    cutoff = max(1, int(np.floor(k * n)))
    order = np.argsort(-y_prob)
    top_k = y_true[order][:cutoff]
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    return float(top_k.sum() / positives)


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if not 0 < k <= 1:
        raise ValueError("k must be in (0, 1]")
    n = len(y_true)
    cutoff = max(1, int(np.floor(k * n)))
    order = np.argsort(-y_prob)
    top_k = y_true[order][:cutoff]
    return float(top_k.mean())


def expected_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    business: BusinessConfig,
    threshold: float,
) -> float:
    treat = y_prob >= threshold
    true_churn = y_true == 1
    tp = (treat & true_churn).sum()
    fp = (treat & ~true_churn).sum()
    profit = tp * (business.saved_revenue - business.retention_cost) - fp * business.retention_cost
    return float(profit)


def best_threshold_by_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    business: BusinessConfig,
    thresholds: Iterable[float] | None = None,
) -> tuple[float, float]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_profit = -np.inf
    for t in thresholds:
        profit = expected_profit(y_true, y_prob, business, t)
        if profit > best_profit:
            best_profit = profit
            best_t = t
    return float(best_t), float(best_profit)


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_list: Iterable[float],
) -> dict[str, float]:
    unique_vals = np.unique(y_true)
    if len(unique_vals) < 2:
        metrics: dict[str, float] = {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "log_loss": float("nan"),
            "brier": float("nan"),
        }
    else:
        metrics = {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "log_loss": float(log_loss(y_true, y_prob)),
            "brier": float(brier_score_loss(y_true, y_prob)),
        }
    for k in k_list:
        metrics[f"recall@{int(k * 100)}"] = recall_at_k(y_true, y_prob, k)
        metrics[f"precision@{int(k * 100)}"] = precision_at_k(y_true, y_prob, k)
    return metrics
