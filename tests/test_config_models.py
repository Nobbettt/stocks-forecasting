from __future__ import annotations

from stocks_forecasting.config.models import UniverseConfig


def test_universe_config_symbols_normalizes_and_deduplicates() -> None:
    cfg = UniverseConfig(symbols=[" aapl ", "AAPL", "msft", "", "  "])
    assert cfg.symbols == ["AAPL", "MSFT"]


def test_universe_config_symbols_empty_becomes_none() -> None:
    cfg = UniverseConfig(symbols=["", " "])
    assert cfg.symbols is None

