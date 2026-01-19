from __future__ import annotations

import pandas as pd

from stocks_forecasting.config.models import ForecastingConfig
from stocks_forecasting.dataset.prepare_symbol import build_symbol_feature_frame


def test_build_symbol_feature_frame_adds_future_rows() -> None:
    cfg = ForecastingConfig()
    cfg.model.horizon_days = 5
    cfg.features.known_future.include_holidays = True

    prices = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC"),
            "open": [10.0] * 10,
            "high": [11.0] * 10,
            "low": [9.0] * 10,
            "close": [10.0 + i for i in range(10)],
            "volume": [100] * 10,
        }
    )

    built = build_symbol_feature_frame(prices, cfg, calendar=None)
    frame = built.frame

    assert built.horizon == 5
    assert len(frame) == 15
    assert frame["is_future"].sum() == 5
    assert frame["time"].is_monotonic_increasing

