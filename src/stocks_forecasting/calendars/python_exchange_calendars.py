"""Trading calendar backed by the exchange_calendars library."""

from __future__ import annotations

import pandas as pd

from stocks_forecasting.calendars.base import TradingCalendar, ensure_utc, normalize_utc_midnight


class PythonExchangeCalendarsCalendar(TradingCalendar):
    """Calendar using exchange_calendars for exchange-specific sessions/holidays."""

    def __init__(self, calendar_name: str) -> None:
        try:
            import exchange_calendars as ec  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Calendar provider requires `exchange-calendars` (import `exchange_calendars`). "
                "Install with `uv pip install exchange-calendars`."
            ) from exc

        self._calendar_name = calendar_name
        self._calendar = ec.get_calendar(calendar_name)

    @property
    def calendar_name(self) -> str:
        return self._calendar_name

    def sessions_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """Return exchange trading sessions between start and end."""
        start_day = normalize_utc_midnight(start).tz_localize(None)
        end_day = normalize_utc_midnight(end).tz_localize(None)
        sessions = self._calendar.sessions_in_range(start_day, end_day)
        return ensure_utc(pd.DatetimeIndex(sessions)).normalize()

    def holidays_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """Return exchange holidays (weekday non-sessions) between start and end."""
        start_day = normalize_utc_midnight(start)
        end_day = normalize_utc_midnight(end)

        sessions = self.sessions_in_range(start_day, end_day)
        weekdays = pd.date_range(start_day, end_day, freq="B", tz="UTC")
        holiday_days = weekdays.difference(sessions)
        return ensure_utc(holiday_days).normalize()

    def next_sessions(self, after: pd.Timestamp, count: int) -> pd.DatetimeIndex:
        """Return next `count` trading sessions after the given date."""
        if count <= 0:
            return pd.DatetimeIndex([], tz="UTC")

        after_day = normalize_utc_midnight(after)
        start = after_day + pd.Timedelta(days=1)
        # 400 calendar days is comfortably enough to find 30 trading sessions.
        # Clamp to calendar coverage to avoid DateOutOfBounds errors.
        end = start + pd.Timedelta(days=400)
        last_session = normalize_utc_midnight(pd.Timestamp(self._calendar.last_session))
        if end > last_session:
            end = last_session
        sessions = self.sessions_in_range(start, end)
        return sessions[:count]
