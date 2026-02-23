"""Time-based train/val/test split computation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.tseries.offsets import DateOffset

from stocks_forecasting.config.models import EvaluationSplitConfig


class TimeSplitError(ValueError):
    """Raised when time split computation fails."""


@dataclass(frozen=True, slots=True)
class TimeSplit:
    """Time boundaries for train/val/test periods."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_iso_dict(self) -> dict[str, str]:
        """Convert boundaries to ISO format dict."""
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "val_start": self.val_start.isoformat(),
            "val_end": self.val_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def compute_time_split(series_end: pd.Timestamp, config: EvaluationSplitConfig) -> TimeSplit:
    """Compute train/val/test time boundaries working backward from series_end."""
    end: pd.Timestamp = _normalize_utc(series_end)

    test_start: pd.Timestamp = end - DateOffset(months=config.test_months)
    val_start: pd.Timestamp = test_start - DateOffset(months=config.val_months)
    train_start: pd.Timestamp = val_start - DateOffset(months=config.train_months)

    # Splits are inclusive windows. `gap_days` specifies how many whole days to exclude
    # between the end of one split and the start of the next.
    gap: pd.Timedelta = pd.Timedelta(days=int(config.gap_days) + 1)
    train_end: pd.Timestamp = val_start - gap
    val_end: pd.Timestamp = test_start - gap

    if not (train_start <= train_end < val_start <= val_end < test_start <= end):
        raise TimeSplitError(
            "Invalid time split boundaries (check months/gap_days relative to series_end)"
        )

    return TimeSplit(
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=end,
    )


def slice_time_window(frame: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Filter DataFrame to rows within [start, end] time window."""
    if frame.empty:
        return frame
    if "time" not in frame.columns:
        raise TimeSplitError("frame must contain a `time` column")

    start_ts: pd.Timestamp = _normalize_utc(start)
    end_ts: pd.Timestamp = _normalize_utc(end)
    if start_ts > end_ts:
        raise TimeSplitError("start must be <= end")

    out: pd.DataFrame = frame.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce").dt.normalize()
    out = out.dropna(subset=["time"])
    return out[(out["time"] >= start_ts) & (out["time"] <= end_ts)].copy()


def _normalize_utc(value: pd.Timestamp) -> pd.Timestamp:
    """Normalize timestamp to UTC midnight."""
    ts: pd.Timestamp = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()
