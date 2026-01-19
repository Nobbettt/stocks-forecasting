"""Known-future calendar features (day of week, holidays, etc.)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stocks_forecasting.calendars import TradingCalendar
from stocks_forecasting.config.models import KnownFutureConfig


class KnownFutureFeatureError(ValueError):
    """Raised when known-future feature computation fails."""


@dataclass(frozen=True, slots=True)
class HolidayGapFeatures:
    """Holiday gap features for a single session."""

    holiday_count_since_prev_session: int
    is_day_after_holiday: bool


def compute_known_future_features(
    times: pd.Series | pd.DatetimeIndex,
    config: KnownFutureConfig,
    *,
    calendar: TradingCalendar | None = None,
    time_column: str = "time",
) -> pd.DataFrame:
    """Compute calendar-based features known at forecast time."""
    index: pd.DatetimeIndex
    if isinstance(times, pd.DatetimeIndex):
        index = times
    else:
        index = pd.DatetimeIndex(pd.to_datetime(times, utc=True, errors="coerce"))

    index = index.dropna().sort_values()
    if index.empty:
        return pd.DataFrame(columns=[time_column])

    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    df = pd.DataFrame({time_column: index})

    if config.include_day_of_week:
        df["day_of_week"] = index.dayofweek.astype("int16")
    if config.include_day_of_month:
        df["day_of_month"] = index.day.astype("int16")
    if config.include_week_of_year:
        df["week_of_year"] = index.isocalendar().week.astype("int16")
    if config.include_month:
        df["month"] = index.month.astype("int16")
    if config.include_quarter:
        df["quarter"] = index.quarter.astype("int16")
    if config.include_is_month_end:
        df["is_month_end"] = index.is_month_end.astype("int8")
    if config.include_is_quarter_end:
        df["is_quarter_end"] = index.is_quarter_end.astype("int8")

    if config.include_holidays:
        holiday_count: pd.Series[int]
        day_after: pd.Series[bool]
        holiday_count, day_after = _compute_holiday_gap_features(index, calendar)
        df["holiday_count_since_prev_session"] = holiday_count.astype("int16")
        df["is_day_after_holiday"] = day_after.astype("int8")

    return df.reset_index(drop=True)


def _compute_holiday_gap_features(
    sessions: pd.DatetimeIndex, calendar: TradingCalendar | None
) -> tuple["pd.Series[int]", "pd.Series[bool]"]:
    """Compute holiday count and day-after-holiday flags for sessions."""
    if calendar is None:
        zeros: pd.Series[int] = pd.Series([0] * len(sessions), dtype="int64")
        falses: pd.Series[bool] = pd.Series([False] * len(sessions), dtype="bool")
        return zeros, falses

    sessions_norm: pd.DatetimeIndex = sessions.normalize()
    start: pd.Timestamp = sessions_norm.min() - pd.Timedelta(days=14)
    end: pd.Timestamp = sessions_norm.max() + pd.Timedelta(days=14)

    holiday_days: set[pd.Timestamp] = set(calendar.holidays_in_range(start, end).normalize())

    counts: list[int] = []
    after_flags: list[bool] = []

    for i, current in enumerate(sessions_norm):
        if i == 0:
            counts.append(0)
            after_flags.append(False)
            continue

        prev: pd.Timestamp = sessions_norm[i - 1]
        gap_start: pd.Timestamp = prev + pd.Timedelta(days=1)
        gap_end: pd.Timestamp = current - pd.Timedelta(days=1)
        if gap_start > gap_end:
            counts.append(0)
            after_flags.append(False)
            continue

        gap_days: pd.DatetimeIndex = pd.date_range(gap_start, gap_end, freq="D", tz="UTC").normalize()
        holiday_count: int = sum(1 for day in gap_days if day in holiday_days)
        counts.append(holiday_count)
        after_flags.append(holiday_count > 0)

    return pd.Series(counts, dtype="int64"), pd.Series(after_flags, dtype="bool")
