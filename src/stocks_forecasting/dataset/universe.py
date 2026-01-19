"""Training universe construction from database."""

from __future__ import annotations

import pandas as pd

from stocks_forecasting.config.models import ForecastingConfig
from stocks_forecasting.dataset.metadata import add_market_cap_bucket
from stocks_forecasting.db import PostgresClient


class UniverseError(ValueError):
    """Raised when universe construction fails."""


def build_training_universe(client: PostgresClient, config: ForecastingConfig) -> pd.DataFrame:
    """Build eligible symbol universe filtered by minimum history requirement."""
    limit: int | None = config.universe.limit_symbols
    active_symbols: list[str] = client.fetch_active_symbols(limit=limit)
    if not active_symbols:
        raise UniverseError("No active symbols found in database")

    metadata: pd.DataFrame = client.fetch_stock_metadata(symbols=active_symbols)
    if metadata.empty:
        raise UniverseError("No metadata returned for active symbols")

    ranges: pd.DataFrame = client.fetch_price_ranges(price_type=config.data.price_type, symbols=active_symbols)
    if ranges.empty:
        raise UniverseError(f"No price history found for price_type={config.data.price_type}")

    merged: pd.DataFrame = metadata.merge(ranges, on="symbol", how="inner")
    merged = add_market_cap_bucket(merged)

    merged["start_time"] = pd.to_datetime(merged["start_time"], utc=True, errors="coerce")
    merged["end_time"] = pd.to_datetime(merged["end_time"], utc=True, errors="coerce")

    merged["history_days"] = (merged["end_time"] - merged["start_time"]).dt.days
    min_days: int = int(config.data.min_history_years * 365)
    eligible: pd.DataFrame = merged[(merged["history_days"] >= min_days) & merged["end_time"].notna() & merged["start_time"].notna()]

    eligible = eligible.sort_values("symbol").reset_index(drop=True)
    if eligible.empty:
        raise UniverseError(
            f"No symbols meet minimum history requirement ({config.data.min_history_years} years)"
        )
    return eligible

