from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocks_forecasting.config.models import BollingerConfig, MacdConfig, TechnicalIndicatorsConfig
from stocks_forecasting.features.technical_indicators import compute_technical_indicators


def test_compute_technical_indicators_basic_columns_and_length() -> None:
    prices = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], utc=True),
            "open": [10, 11, 12, 13],
            "high": [11, 12, 13, 14],
            "low": [9, 10, 11, 12],
            "close": [10, 11, 10, 12],
            "volume": [100, 100, 100, 100],
        }
    )
    cfg = TechnicalIndicatorsConfig(
        sma_windows=[2],
        ema_windows=[2],
        macd=MacdConfig(fast=2, slow=3, signal=2),
        rsi_period=2,
        bollinger=BollingerConfig(window=2, num_standard_deviations=2.0),
        atr_period=2,
        include_obv=True,
        volume_sma_window=2,
    )

    out = compute_technical_indicators(prices, cfg)
    assert len(out) == len(prices)
    assert out["time"].tolist() == prices["time"].tolist()

    expected_cols = {
        "sma_2",
        "ema_2",
        "macd_line_2_3",
        "macd_signal_2_3_2",
        "macd_hist_2_3_2",
        "rsi_2",
        "bb_mid_2",
        "bb_upper_2_2",
        "bb_lower_2_2",
        "atr_2",
        "obv",
        "volume_sma_2",
    }
    assert expected_cols.issubset(set(out.columns))


def test_compute_technical_indicators_obv_matches_expected() -> None:
    prices = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], utc=True),
            "open": [10, 11, 11, 12],
            "high": [11, 12, 12, 13],
            "low": [9, 10, 10, 11],
            "close": [10, 11, 10, 12],
            "volume": [100, 100, 100, 100],
        }
    )
    cfg = TechnicalIndicatorsConfig(sma_windows=[2], ema_windows=[2], include_obv=True)
    out = compute_technical_indicators(prices, cfg)
    # OBV starts at 0; up adds volume, down subtracts volume, flat adds 0.
    assert out["obv"].tolist() == [0.0, 100.0, 0.0, 100.0]


def test_compute_technical_indicators_bollinger_constant_series() -> None:
    prices = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "open": [10] * 5,
            "high": [10] * 5,
            "low": [10] * 5,
            "close": [10] * 5,
            "volume": [100] * 5,
        }
    )
    cfg = TechnicalIndicatorsConfig(
        sma_windows=[3],
        ema_windows=[3],
        rsi_period=3,
        bollinger=BollingerConfig(window=3, num_standard_deviations=2.0),
        atr_period=3,
        include_obv=False,
        volume_sma_window=3,
        macd=MacdConfig(fast=2, slow=3, signal=2),
    )
    out = compute_technical_indicators(prices, cfg)
    # After warmup, mid/upper/lower should equal 10 when std=0.
    assert out["bb_mid_3"].iloc[2] == pytest.approx(10.0)
    assert out["bb_upper_3_2"].iloc[2] == pytest.approx(10.0)
    assert out["bb_lower_3_2"].iloc[2] == pytest.approx(10.0)
    # ATR should be 0 for constant high/low/close.
    assert out["atr_3"].iloc[2] == pytest.approx(0.0)


def test_compute_technical_indicators_deduplicates_by_time_keep_last() -> None:
    prices = pd.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "open": [10, 99, 11],
            "high": [11, 100, 12],
            "low": [9, 98, 10],
            "close": [10, 99, 11],
            "volume": [100, 999, 100],
        }
    )
    cfg = TechnicalIndicatorsConfig(sma_windows=[2], ema_windows=[2], include_obv=False)
    out = compute_technical_indicators(prices, cfg)
    assert len(out) == 2
    assert out["time"].tolist() == pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True).tolist()
    # The last record for 2024-01-01 has close=99 and volume=999.
    assert out["sma_2"].isna().iloc[0]
    assert np.isfinite(out["sma_2"].iloc[1])

