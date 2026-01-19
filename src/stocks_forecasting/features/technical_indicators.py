"""Technical indicators computation (SMA, EMA, MACD, RSI, Bollinger, ATR, OBV)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stocks_forecasting.config.models import TechnicalIndicatorsConfig


class IndicatorComputationError(ValueError):
    """Raised when indicator computation fails."""


def compute_technical_indicators(
    prices: pd.DataFrame,
    config: TechnicalIndicatorsConfig,
    *,
    time_column: str = "time",
    high_column: str = "high",
    low_column: str = "low",
    close_column: str = "close",
    volume_column: str = "volume",
) -> pd.DataFrame:
    """Compute all configured technical indicators from OHLCV data."""
    if prices.empty:
        return pd.DataFrame(columns=[time_column])

    required: list[str] = [time_column, high_column, low_column, close_column, volume_column]
    missing: list[str] = [c for c in required if c not in prices.columns]
    if missing:
        raise IndicatorComputationError(f"Missing required columns: {', '.join(missing)}")

    df = prices[[time_column, high_column, low_column, close_column, volume_column]].copy()
    df[time_column] = pd.to_datetime(df[time_column], utc=True, errors="coerce")
    df = df.dropna(subset=[time_column])
    if df.empty:
        return pd.DataFrame(columns=[time_column])

    df = df.sort_values(time_column, kind="mergesort").drop_duplicates(subset=[time_column], keep="last")
    df[close_column] = pd.to_numeric(df[close_column], errors="coerce").astype("float64")
    df[high_column] = pd.to_numeric(df[high_column], errors="coerce").astype("float64")
    df[low_column] = pd.to_numeric(df[low_column], errors="coerce").astype("float64")
    df[volume_column] = pd.to_numeric(df[volume_column], errors="coerce")

    close: pd.Series[float] = df[close_column]
    high: pd.Series[float] = df[high_column]
    low: pd.Series[float] = df[low_column]
    volume: pd.Series[float] = df[volume_column].fillna(0).astype("float64")

    out = pd.DataFrame({time_column: df[time_column]})

    # Trend
    for window in config.sma_windows:
        out[f"sma_{window}"] = close.rolling(window, min_periods=window).mean()
    for window in config.ema_windows:
        out[f"ema_{window}"] = close.ewm(span=window, adjust=False, min_periods=window).mean()

    macd = _compute_macd(close, fast=config.macd.fast, slow=config.macd.slow, signal=config.macd.signal)
    out[f"macd_line_{config.macd.fast}_{config.macd.slow}"] = macd["macd"]
    out[f"macd_signal_{config.macd.fast}_{config.macd.slow}_{config.macd.signal}"] = macd["signal"]
    out[f"macd_hist_{config.macd.fast}_{config.macd.slow}_{config.macd.signal}"] = macd["hist"]

    # Momentum
    out[f"rsi_{config.rsi_period}"] = _compute_rsi(close, period=config.rsi_period)

    # Volatility
    bb = _compute_bollinger(
        close,
        window=config.bollinger.window,
        num_std=config.bollinger.num_standard_deviations,
    )
    out[f"bb_mid_{config.bollinger.window}"] = bb["mid"]
    out[f"bb_upper_{config.bollinger.window}_{config.bollinger.num_standard_deviations:g}"] = bb["upper"]
    out[f"bb_lower_{config.bollinger.window}_{config.bollinger.num_standard_deviations:g}"] = bb["lower"]

    out[f"atr_{config.atr_period}"] = _compute_atr(high, low, close, period=config.atr_period)

    # Volume
    if config.include_obv:
        out["obv"] = _compute_obv(close, volume)
    out[f"volume_sma_{config.volume_sma_window}"] = volume.rolling(
        config.volume_sma_window, min_periods=config.volume_sma_window
    ).mean()

    return out.reset_index(drop=True)


def _compute_macd(close: "pd.Series[float]", *, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """Compute MACD line, signal line, and histogram."""
    ema_fast: pd.Series[float] = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow: pd.Series[float] = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line: pd.Series[float] = ema_fast - ema_slow
    signal_line: pd.Series[float] = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist: pd.Series[float] = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def _compute_rsi(close: "pd.Series[float]", *, period: int) -> "pd.Series[float]":
    """Compute Relative Strength Index (RSI)."""
    delta: pd.Series[float] = close.diff()
    gain: pd.Series[float] = delta.clip(lower=0)
    loss: pd.Series[float] = -delta.clip(upper=0)

    avg_gain: pd.Series[float] = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss: pd.Series[float] = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs: pd.Series[float] = avg_gain / avg_loss
    rsi: pd.Series[float] = 100 - (100 / (1 + rs))

    rsi = rsi.mask(avg_loss == 0, 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), np.nan)
    return rsi.astype("float64")


def _compute_bollinger(close: "pd.Series[float]", *, window: int, num_std: float) -> pd.DataFrame:
    """Compute Bollinger Bands (middle, upper, lower)."""
    mid: pd.Series[float] = close.rolling(window, min_periods=window).mean()
    std: pd.Series[float] = close.rolling(window, min_periods=window).std(ddof=0)
    upper: pd.Series[float] = mid + num_std * std
    lower: pd.Series[float] = mid - num_std * std
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})


def _compute_atr(high: "pd.Series[float]", low: "pd.Series[float]", close: "pd.Series[float]", *, period: int) -> "pd.Series[float]":
    """Compute Average True Range (ATR)."""
    prev_close: pd.Series[float] = close.shift(1)
    tr: pd.Series[float] = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr: pd.Series[float] = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr.astype("float64")


def _compute_obv(close: "pd.Series[float]", volume: "pd.Series[float]") -> "pd.Series[float]":
    """Compute On-Balance Volume (OBV)."""
    direction: np.ndarray = np.sign(close.diff().fillna(0.0))
    signed_volume: pd.Series[float] = volume.where(direction >= 0, -volume)
    signed_volume = signed_volume.where(direction != 0, 0.0)
    return signed_volume.cumsum().astype("float64")
