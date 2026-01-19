"""Target variable computation (log returns)."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TargetComputationError(ValueError):
    """Raised when target computation fails."""


def compute_log_return_target(
    prices: pd.DataFrame,
    *,
    time_column: str = "time",
    price_column: str = "close",
    output_column: str = "log_return",
) -> pd.DataFrame:
    """Compute daily log returns: log(P_t / P_{t-1}).

    The returned DataFrame contains `time_column`, `price_column`, and `output_column`,
    is sorted by time, and contains a NaN log return for the first observation.

    Notes:
    - Duplicate timestamps are de-duplicated by keeping the last record for a timestamp.
    - Non-positive prices result in NaN log returns for the affected row.
    """

    if time_column not in prices.columns:
        raise TargetComputationError(f"Missing required column: {time_column}")
    if price_column not in prices.columns:
        raise TargetComputationError(f"Missing required column: {price_column}")

    if prices.empty:
        return pd.DataFrame(columns=[time_column, price_column, output_column])

    df = prices[[time_column, price_column]].copy()
    df[time_column] = pd.to_datetime(df[time_column], utc=True, errors="coerce")
    df = df.dropna(subset=[time_column])
    if df.empty:
        return pd.DataFrame(columns=[time_column, price_column, output_column])

    df = df.sort_values(time_column, kind="mergesort")
    df = df.drop_duplicates(subset=[time_column], keep="last")

    df[price_column] = pd.to_numeric(df[price_column], errors="coerce")

    current: pd.Series[float] = pd.to_numeric(df[price_column], errors="coerce").astype("float64")
    previous: pd.Series[float] = current.shift(1)

    current_values: np.ndarray = current.to_numpy(dtype="float64", na_value=np.nan)
    previous_values: np.ndarray = previous.to_numpy(dtype="float64", na_value=np.nan)
    invalid: np.ndarray = (
        ~np.isfinite(current_values)
        | ~np.isfinite(previous_values)
        | (current_values <= 0)
        | (previous_values <= 0)
    )

    ratio: np.ndarray = np.full_like(current_values, np.nan)
    valid: np.ndarray = ~invalid
    ratio[valid] = current_values[valid] / previous_values[valid]
    log_returns: np.ndarray = np.log(ratio)

    df[output_column] = pd.Series(log_returns, index=df.index, dtype="float64")
    return df[[time_column, price_column, output_column]].reset_index(drop=True)
