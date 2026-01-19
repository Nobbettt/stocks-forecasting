from __future__ import annotations

import numpy as np

from stocks_forecasting.metrics.quantile import evaluate_quantile_forecasts, pinball_loss


def test_pinball_loss_matches_known_values() -> None:
    y = np.array([1.0, 1.0, 1.0])
    yhat = np.array([0.0, 1.0, 2.0])
    # q=0.5 => absolute error / 2 per point.
    loss = pinball_loss(y, yhat, 0.5)
    assert loss.tolist() == [0.5, 0.0, 0.5]


def test_evaluate_quantile_forecasts_computes_required_metrics() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0])
    forecasts = {
        0.1: np.array([-1.0, 0.0, 1.0, 2.0]),
        0.5: np.array([0.0, 1.0, 2.0, 3.0]),
        0.9: np.array([1.0, 2.0, 3.0, 4.0]),
    }
    metrics = evaluate_quantile_forecasts(y, forecasts)

    assert "crps_pinball" in metrics
    assert metrics["mae_median"] == 0.0
    assert metrics["coverage_80"] == 1.0
    assert metrics["pinball_q0.1"] > 0.0
    assert metrics["pinball_q0.5"] == 0.0
    assert metrics["pinball_q0.9"] > 0.0

