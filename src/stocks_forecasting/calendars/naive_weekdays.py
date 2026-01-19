"""Naive weekday-only trading calendar (no exchange-specific holidays)."""

from __future__ import annotations

import pandas as pd

from stocks_forecasting.calendars.base import ensure_utc, normalize_utc_midnight


class NaiveWeekdayCalendar:
    """Naive calendar: Monday–Friday sessions, no exchange-specific holidays."""

    def sessions_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """Return all weekdays (Mon-Fri) between start and end."""
        start_day = normalize_utc_midnight(start)
        end_day = normalize_utc_midnight(end)
        return pd.date_range(start_day, end_day, freq="B", tz="UTC")

    def holidays_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """Return empty index (naive calendar has no holidays)."""
        _ = (start, end)
        return pd.DatetimeIndex([], tz="UTC")

    def next_sessions(self, after: pd.Timestamp, count: int) -> pd.DatetimeIndex:
        """Return next `count` weekdays after the given date."""
        if count <= 0:
            return pd.DatetimeIndex([], tz="UTC")

        after_day = normalize_utc_midnight(after)
        # Generate a bit more than needed; slice the first `count`.
        start = after_day + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=count * 3)
        sessions = pd.date_range(start, end, freq="B", tz="UTC")
        return ensure_utc(sessions[:count])

