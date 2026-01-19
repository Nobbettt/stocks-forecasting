"""Quantile forecast evaluation metrics."""

from __future__ import annotations

import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> np.ndarray:
    """Compute pinball loss for a single quantile."""
    q: float = float(quantile)
    if not (0.0 < q < 1.0):
        raise ValueError("quantile must be between 0 and 1 (exclusive)")

    yt: np.ndarray = np.asarray(y_true, dtype="float64")
    yp: np.ndarray = np.asarray(y_pred, dtype="float64")
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    delta: np.ndarray = yt - yp
    return np.maximum(q * delta, (q - 1.0) * delta)


def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error between true and predicted values."""
    yt: np.ndarray = np.asarray(y_true, dtype="float64")
    yp: np.ndarray = np.asarray(y_pred, dtype="float64")
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(yt - yp)))


def _coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Compute coverage rate of prediction interval."""
    yt: np.ndarray = np.asarray(y_true, dtype="float64")
    low: np.ndarray = np.asarray(y_lower, dtype="float64")
    high: np.ndarray = np.asarray(y_upper, dtype="float64")
    if yt.shape != low.shape or yt.shape != high.shape:
        raise ValueError("y_true, y_lower, and y_upper must have the same shape")
    return float(np.mean((yt >= low) & (yt <= high)))


def evaluate_quantile_forecasts(
    y_true: np.ndarray,
    forecasts: dict[float, np.ndarray],
    *,
    median_quantile: float = 0.5,
    coverage_interval: tuple[float, float] = (0.1, 0.9),
) -> dict[str, float]:
    """Evaluate quantile forecasts with CRPS, MAE, and coverage metrics."""
    if not forecasts:
        raise ValueError("forecasts must not be empty")

    yt: np.ndarray = np.asarray(y_true, dtype="float64").reshape(-1)
    if yt.size == 0:
        raise ValueError("y_true must not be empty")

    quantiles: list[float] = sorted(float(q) for q in forecasts.keys())
    for q in quantiles:
        if not (0.0 < q < 1.0):
            raise ValueError("all forecast quantiles must be between 0 and 1 (exclusive)")

    preds: dict[float, np.ndarray] = {float(q): np.asarray(forecasts[q], dtype="float64").reshape(-1) for q in forecasts}
    for q, yq in preds.items():
        if yq.shape != yt.shape:
            raise ValueError(f"forecast for quantile {q} has shape {yq.shape}, expected {yt.shape}")

    metrics: dict[str, float] = {}

    pinball_means: list[float] = []
    for q in quantiles:
        loss: np.ndarray = pinball_loss(yt, preds[q], q)
        mean_loss: float = float(np.mean(loss))
        metrics[f"pinball_q{q:g}"] = mean_loss
        pinball_means.append(mean_loss)

    metrics["crps_pinball"] = float(2.0 * float(np.mean(pinball_means)))

    if median_quantile in preds:
        metrics["mae_median"] = _mean_absolute_error(yt, preds[float(median_quantile)])

    low_q: float
    high_q: float
    low_q, high_q = coverage_interval
    if float(low_q) in preds and float(high_q) in preds:
        metrics["coverage_80"] = _coverage(yt, preds[float(low_q)], preds[float(high_q)])

    return metrics

