from __future__ import annotations

import pandas as pd
import pytest

from stocks_forecasting.calendars.python_exchange_calendars import PythonExchangeCalendarsCalendar


def test_exchange_calendars_next_sessions_accepts_pandas_utc_timezone() -> None:
    pytest.importorskip("exchange_calendars")
    cal = PythonExchangeCalendarsCalendar("XNYS")
    after = pd.Timestamp("2024-01-02", tz="UTC")
    nxt = cal.next_sessions(after, 5)

    assert len(nxt) == 5
    assert nxt.is_monotonic_increasing


def test_exchange_calendars_next_sessions_caps_range_to_calendar_coverage() -> None:
    pytest.importorskip("exchange_calendars")
    cal = PythonExchangeCalendarsCalendar("XNYS")

    last = pd.Timestamp(cal._calendar.last_session).tz_localize("UTC")  # type: ignore[attr-defined]
    after = last - pd.Timedelta(days=10)

    nxt = cal.next_sessions(after, 10)
    assert len(nxt) >= 1
    assert nxt.max() <= last.normalize()
