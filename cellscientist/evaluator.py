from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .schemas import ContractError


def _safe_flat_pcc(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if not np.isfinite(denominator) or denominator <= 1e-12:
        return 0.0
    return float(np.sum(x * y) / denominator)


def _sample_pcc(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x = x - np.mean(x, axis=1, keepdims=True)
    y = y - np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x * y, axis=1)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    values = np.zeros(len(x), dtype=np.float64)
    valid = np.isfinite(denominator) & (denominator > 1e-12)
    values[valid] = numerator[valid] / denominator[valid]
    return float(np.mean(values))


def _topk_response_pcc(
    true_response: np.ndarray,
    predicted_response: np.ndarray,
    k: int,
) -> float:
    k = min(int(k), true_response.shape[1])
    top = np.argpartition(np.abs(true_response), -k, axis=1)[:, -k:]
    true_top = np.take_along_axis(true_response, top, axis=1)
    pred_top = np.take_along_axis(predicted_response, top, axis=1)
    return _sample_pcc(true_top, pred_top)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    morphology_pre: np.ndarray,
    train_target_mean: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    morphology_pre = np.asarray(morphology_pre, dtype=np.float32)
    train_target_mean = np.asarray(train_target_mean, dtype=np.float32)
    if y_true.shape != y_pred.shape or y_true.shape != morphology_pre.shape:
        raise ContractError(
            f"Prediction contract mismatch: true={y_true.shape}, "
            f"pred={y_pred.shape}, pre={morphology_pre.shape}"
        )
    if not np.all(np.isfinite(y_pred)):
        raise ContractError("Predictions contain NaN or infinity")

    residual = y_true - y_pred
    response_true = y_true - morphology_pre
    response_pred = y_pred - morphology_pre
    centered_true = y_true - train_target_mean
    centered_pred = y_pred - train_target_mean
    sum_squared = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    r2 = 0.0 if sum_squared <= 1e-12 else 1.0 - float(np.sum(residual**2)) / sum_squared
    metrics = {
        "pcc": _safe_flat_pcc(y_true, y_pred),
        "sample_pcc": _sample_pcc(y_true, y_pred),
        "mse": float(np.mean(residual**2)),
        "r2": float(r2),
        "centered_pcc": _safe_flat_pcc(centered_true, centered_pred),
        "response_pcc": _safe_flat_pcc(response_true, response_pred),
        "response_sample_pcc": _sample_pcc(response_true, response_pred),
        "response_mse": float(np.mean((response_true - response_pred) ** 2)),
        "top20_response_pcc": _topk_response_pcc(response_true, response_pred, 20),
        "top50_response_pcc": _topk_response_pcc(response_true, response_pred, 50),
    }
    for key, value in metrics.items():
        if not np.isfinite(value):
            raise ContractError(f"Non-finite metric {key}: {value}")
    return metrics


def metric_direction(metric: str) -> int:
    if metric in {"mse", "response_mse"}:
        return -1
    return 1


def objective_value(metrics: Dict[str, float], primary_metric: str) -> float:
    if primary_metric not in metrics:
        raise ContractError(f"Primary metric is absent: {primary_metric}")
    return metric_direction(primary_metric) * float(metrics[primary_metric])


def mean_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    values = list(rows)
    if not values:
        return {}
    common = set.intersection(*(set(row) for row in values))
    return {
        key: float(np.mean([row[key] for row in values]))
        for key in sorted(common)
    }

