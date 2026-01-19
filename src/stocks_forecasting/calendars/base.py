"""Base protocol and utilities for trading calendars."""

from __future__ import annotations

from typing import Protocol

import pandas as pd
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")


class TradingCalendar(Protocol):
    """Provides trading sessions and market holidays for a given exchange/calendar."""

    def sessions_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """Return session dates between start and end (inclusive)."""

    def holidays_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """Return exchange holidays between start and end (inclusive), excluding weekends."""

    def next_sessions(self, after: pd.Timestamp, count: int) -> pd.DatetimeIndex:
        """Return the next `count` session dates after `after` (exclusive)."""


def normalize_utc_midnight(value: pd.Timestamp) -> pd.Timestamp:
    """Convert timestamp to UTC midnight (date-only, tz-aware)."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return timestamp.normalize()


def ensure_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ensure DatetimeIndex is UTC-aware."""
    if index.tz is None:
        return index.tz_localize(UTC)
    return index.tz_convert(UTC)
