from __future__ import annotations

import pandas as pd

from stocks_forecasting.config.models import EvaluationSplitConfig
from stocks_forecasting.dataset.time_splits import compute_time_split, slice_time_window


def test_compute_time_split_basic_ordering() -> None:
    cfg = EvaluationSplitConfig(train_months=18, val_months=3, test_months=3, gap_days=0)
    end = pd.Timestamp("2024-12-31", tz="UTC")
    split = compute_time_split(end, cfg)

    assert split.train_start <= split.train_end < split.val_start <= split.val_end < split.test_start <= split.test_end


def test_compute_time_split_gap_creates_excluded_window() -> None:
    cfg = EvaluationSplitConfig(train_months=18, val_months=3, test_months=3, gap_days=5)
    end = pd.Timestamp("2024-12-31", tz="UTC")
    split = compute_time_split(end, cfg)

    assert (split.val_start - split.train_end - pd.Timedelta(days=1)).days == 5
    assert (split.test_start - split.val_end - pd.Timedelta(days=1)).days == 5


def test_slice_time_window_inclusive_boundaries() -> None:
    frame = pd.DataFrame(
        {"time": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"), "x": range(10)}
    )
    out = slice_time_window(
        frame,
        start=pd.Timestamp("2024-01-03", tz="UTC"),
        end=pd.Timestamp("2024-01-05", tz="UTC"),
    )
    assert out["time"].min() == pd.Timestamp("2024-01-03", tz="UTC")
    assert out["time"].max() == pd.Timestamp("2024-01-05", tz="UTC")
    assert out["x"].tolist() == [2, 3, 4]
