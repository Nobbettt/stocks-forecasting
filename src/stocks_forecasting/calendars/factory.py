"""Factory for building trading calendars from config."""

from __future__ import annotations

from stocks_forecasting.calendars.base import TradingCalendar
from stocks_forecasting.calendars.naive_weekdays import NaiveWeekdayCalendar
from stocks_forecasting.calendars.python_exchange_calendars import PythonExchangeCalendarsCalendar
from stocks_forecasting.config.models import CalendarConfig, CalendarProvider, UnknownExchangePolicy


class CalendarError(ValueError):
    """Raised when calendar cannot be built or resolved."""


def build_trading_calendar(config: CalendarConfig, *, exchange_mic: str | None) -> TradingCalendar:
    """Build a trading calendar for an exchange.

    `exchange_mic` should typically come from `stocks.stocks.exchange_mic`.
    """

    if config.provider == CalendarProvider.naive_weekdays:
        return NaiveWeekdayCalendar()

    if config.provider != CalendarProvider.python_exchange_calendars:
        raise CalendarError(f"Unsupported calendar provider: {config.provider}")

    calendar_name = _resolve_calendar_name(exchange_mic, config)
    if calendar_name is not None:
        try:
            return PythonExchangeCalendarsCalendar(calendar_name)
        except ModuleNotFoundError as exc:
            raise CalendarError(str(exc)) from exc
        except Exception:
            return _fallback_calendar(config)

    return _fallback_calendar(config)


def _resolve_calendar_name(exchange_mic: str | None, config: CalendarConfig) -> str | None:
    """Map exchange MIC to calendar name via config, or use MIC directly."""
    if exchange_mic is None:
        return None
    exchange_mic = exchange_mic.strip()
    if not exchange_mic:
        return None
    return config.exchange_calendar_map.get(exchange_mic, exchange_mic)


def _fallback_calendar(config: CalendarConfig) -> TradingCalendar:
    """Return fallback calendar based on unknown_exchange_policy."""
    if config.unknown_exchange_policy == UnknownExchangePolicy.error:
        raise CalendarError("Unable to resolve exchange calendar and unknown_exchange_policy=error")

    if config.unknown_exchange_policy == UnknownExchangePolicy.fallback_naive_weekdays:
        return NaiveWeekdayCalendar()

    if config.unknown_exchange_policy == UnknownExchangePolicy.fallback_default_calendar:
        if not config.fallback_calendar:
            raise CalendarError(
                "unknown_exchange_policy=fallback_default_calendar requires `features.calendar.fallback_calendar`"
            )
        try:
            return PythonExchangeCalendarsCalendar(config.fallback_calendar)
        except ModuleNotFoundError as exc:
            raise CalendarError(str(exc)) from exc

    raise CalendarError(f"Unsupported unknown_exchange_policy: {config.unknown_exchange_policy}")
