from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocks_forecasting.features.target import TargetComputationError, compute_log_return_target


def test_compute_log_return_target_basic() -> None:
    prices = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True),
            "close": [100.0, 110.0, 121.0],
        }
    )
    out = compute_log_return_target(prices)
    assert list(out.columns) == ["time", "close", "log_return"]
    assert out["log_return"].isna().iloc[0]
    assert out["log_return"].iloc[1] == pytest.approx(np.log(110.0 / 100.0))
    assert out["log_return"].iloc[2] == pytest.approx(np.log(121.0 / 110.0))


def test_compute_log_return_target_sorts_and_deduplicates() -> None:
    prices = pd.DataFrame(
        {
            "time": ["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-01"],
            "close": [122.1, 110.0, 111.0, 100.0],
        }
    )
    out = compute_log_return_target(prices)
    assert out["time"].tolist() == pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True).tolist()
    assert out["close"].tolist() == [100.0, 111.0, 122.1]
    assert out["log_return"].iloc[1] == pytest.approx(np.log(111.0 / 100.0))
    assert out["log_return"].iloc[2] == pytest.approx(np.log(122.1 / 111.0))


def test_compute_log_return_target_non_positive_prices_become_nan() -> None:
    prices = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True),
            "close": [100.0, 0.0, 110.0],
        }
    )
    out = compute_log_return_target(prices)
    assert out["log_return"].isna().iloc[1]
    assert out["log_return"].isna().iloc[2]


def test_compute_log_return_target_empty_input() -> None:
    out = compute_log_return_target(pd.DataFrame(columns=["time", "close"]))
    assert list(out.columns) == ["time", "close", "log_return"]
    assert out.empty


def test_compute_log_return_target_missing_columns() -> None:
    with pytest.raises(TargetComputationError):
        compute_log_return_target(pd.DataFrame({"time": []}), price_column="close")

