"""Per-symbol feature frame construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stocks_forecasting.calendars import TradingCalendar
from stocks_forecasting.config.models import ForecastingConfig
from stocks_forecasting.features import compute_known_future_features, compute_log_return_target, compute_technical_indicators


@dataclass(frozen=True, slots=True)
class SymbolFeatureFrame:
    """Feature frame with observed data + horizon future rows."""

    frame: pd.DataFrame
    observed_end: pd.Timestamp
    horizon: int
    past_covariate_columns: list[str]
    future_covariate_columns: list[str]


def build_symbol_feature_frame(
    prices: pd.DataFrame,
    config: ForecastingConfig,
    *,
    calendar: TradingCalendar | None,
) -> SymbolFeatureFrame:
    """Build feature frame: target, indicators, known-future features, + horizon rows."""
    if prices.empty:
        raise ValueError("prices is empty")

    prices = prices.copy()
    prices["time"] = pd.to_datetime(prices["time"], utc=True, errors="coerce").dt.normalize()
    prices = prices.dropna(subset=["time"]).sort_values("time", kind="mergesort").drop_duplicates("time", keep="last")
    if prices.empty:
        raise ValueError("prices has no valid timestamps after normalization")

    observed_times: pd.DatetimeIndex = pd.DatetimeIndex(prices["time"])
    observed_end: pd.Timestamp = pd.Timestamp(observed_times.max()).tz_convert("UTC").normalize()

    target: pd.DataFrame = compute_log_return_target(prices, price_column=config.features.target.price_field.value)
    indicators: pd.DataFrame = compute_technical_indicators(prices, config.features.technical_indicators)
    past_covariate_columns: list[str] = [c for c in indicators.columns if c != "time"]

    features: pd.DataFrame = target.merge(indicators, on="time", how="left")

    horizon: int = int(config.model.horizon_days)
    if horizon < 1:
        raise ValueError("horizon_days must be >= 1")

    future_times: pd.DatetimeIndex
    if calendar is None:
        future_times = pd.date_range(observed_end + pd.Timedelta(days=1), periods=horizon, freq="B", tz="UTC")
    else:
        future_times = calendar.next_sessions(observed_end, horizon)

    all_times: pd.DatetimeIndex = observed_times.union(pd.DatetimeIndex(future_times)).sort_values()
    known_future: pd.DataFrame = compute_known_future_features(all_times, config.features.known_future, calendar=calendar)
    future_covariate_columns: list[str] = [c for c in known_future.columns if c != "time"]

    out: pd.DataFrame = known_future.merge(features, on="time", how="left")
    out["is_future"] = (out["time"] > observed_end).astype("int8")

    ordered: list[str] = ["time", "is_future", "close", "log_return"] + [
        c for c in out.columns if c not in {"time", "is_future", "close", "log_return"}
    ]
    out = out[ordered]
    return SymbolFeatureFrame(
        frame=out.reset_index(drop=True),
        observed_end=observed_end,
        horizon=horizon,
        past_covariate_columns=past_covariate_columns,
        future_covariate_columns=future_covariate_columns,
    )
