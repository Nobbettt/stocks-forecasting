from __future__ import annotations

import pandas as pd

from stocks_forecasting.config.models import KnownFutureConfig
from stocks_forecasting.features.known_future import compute_known_future_features


class StaticHolidayCalendar:
    def __init__(self, holidays: list[str]) -> None:
        self._holidays = pd.to_datetime(holidays, utc=True)

    def sessions_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:  # pragma: no cover
        raise NotImplementedError

    def holidays_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        start_day = start_ts.normalize()
        end_day = end_ts.normalize()
        idx = pd.DatetimeIndex(self._holidays).normalize()
        return idx[(idx >= start_day) & (idx <= end_day)]

    def next_sessions(self, after: pd.Timestamp, count: int) -> pd.DatetimeIndex:  # pragma: no cover
        raise NotImplementedError


def test_compute_known_future_features_basic_fields() -> None:
    times = pd.to_datetime(["2024-01-31", "2024-04-01"], utc=True)
    config = KnownFutureConfig(include_holidays=False)
    out = compute_known_future_features(times, config)
    assert out["day_of_month"].tolist() == [31, 1]
    assert out["week_of_year"].tolist() == [5, 14]
    assert out["month"].tolist() == [1, 4]
    assert out["quarter"].tolist() == [1, 2]
    assert out["is_month_end"].tolist() == [1, 0]


def test_compute_known_future_features_holiday_gap_count_and_flag() -> None:
    times = pd.to_datetime(["2024-07-03", "2024-07-05"], utc=True)
    config = KnownFutureConfig(
        include_day_of_week=False,
        include_day_of_month=False,
        include_week_of_year=False,
        include_month=False,
        include_quarter=False,
        include_is_month_end=False,
        include_is_quarter_end=False,
        include_holidays=True,
    )
    calendar = StaticHolidayCalendar(["2024-07-04"])
    out = compute_known_future_features(times, config, calendar=calendar)
    assert out["holiday_count_since_prev_session"].tolist() == [0, 1]
    assert out["is_day_after_holiday"].tolist() == [0, 1]
