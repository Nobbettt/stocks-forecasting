"""Market cap bucketing utilities."""

from __future__ import annotations

from typing import Final

import pandas as pd


MARKET_CAP_BUCKETS: Final[list[tuple[str, float | None, float | None]]] = [
    ("micro", None, 3e8),  # < $300M
    ("small", 3e8, 2e9),  # $300M–$2B
    ("mid", 2e9, 1e10),  # $2B–$10B
    ("large", 1e10, 2e11),  # $10B–$200B
    ("mega", 2e11, None),  # >= $200B
]


def market_cap_bucket(market_cap: float | int | None) -> str | None:
    """Classify market cap into bucket (micro, small, mid, large, mega)."""
    if market_cap is None:
        return None
    try:
        value = float(market_cap)
    except (TypeError, ValueError):
        return None
    if not pd.notna(value) or value <= 0:
        return None

    for name, lower, upper in MARKET_CAP_BUCKETS:
        if lower is not None and value < lower:
            continue
        if upper is not None and value >= upper:
            continue
        return name
    return None


def add_market_cap_bucket(
    metadata: pd.DataFrame, *, market_cap_column: str = "market_cap", output_column: str = "market_cap_bucket"
) -> pd.DataFrame:
    """Add market_cap_bucket column to metadata DataFrame."""
    if metadata.empty:
        return metadata.copy()

    if market_cap_column not in metadata.columns:
        raise ValueError(f"Missing required column: {market_cap_column}")

    out = metadata.copy()
    out[output_column] = out[market_cap_column].apply(market_cap_bucket)
    return out

