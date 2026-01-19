from __future__ import annotations

import pandas as pd

from stocks_forecasting.config.models import EvaluationSplitConfig
from stocks_forecasting.dataset.metadata import add_market_cap_bucket, market_cap_bucket
from stocks_forecasting.dataset.splits import stratified_symbol_split


def test_market_cap_bucket_thresholds() -> None:
    assert market_cap_bucket(None) is None
    assert market_cap_bucket(-1) is None
    assert market_cap_bucket(1) == "micro"
    assert market_cap_bucket(2.99e8) == "micro"
    assert market_cap_bucket(3e8) == "small"
    assert market_cap_bucket(1.99e9) == "small"
    assert market_cap_bucket(2e9) == "mid"
    assert market_cap_bucket(9.99e9) == "mid"
    assert market_cap_bucket(1e10) == "large"
    assert market_cap_bucket(1.99e11) == "large"
    assert market_cap_bucket(2e11) == "mega"


def test_add_market_cap_bucket_adds_column() -> None:
    df = pd.DataFrame({"symbol": ["A", "B"], "market_cap": [3e8, None]})
    out = add_market_cap_bucket(df)
    assert out["market_cap_bucket"].tolist() == ["small", None]


def test_stratified_symbol_split_no_overlap_and_full_coverage() -> None:
    df = pd.DataFrame(
        {
            "symbol": [f"S{i:03d}" for i in range(30)],
            "sector": ["Tech"] * 15 + ["Finance"] * 15,
            "industry": ["Soft"] * 10 + ["Semi"] * 5 + ["Bank"] * 15,
            "market_cap_bucket": ["mid"] * 30,
        }
    )
    cfg = EvaluationSplitConfig(
        symbol_train_ratio=0.7,
        symbol_val_ratio=0.15,
        symbol_test_ratio=0.15,
        stratify_by=["sector", "industry", "market_cap_bucket"],
    )
    splits = stratified_symbol_split(df, cfg, seed=42)

    all_symbols = set(df["symbol"].tolist())
    train = set(splits.train_symbols)
    val = set(splits.val_symbols)
    test = set(splits.test_symbols)

    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert train | val | test == all_symbols


def test_stratified_symbol_split_forces_test_only_symbols() -> None:
    df = pd.DataFrame(
        {
            "symbol": [f"S{i:03d}" for i in range(12)],
            "sector": ["Tech"] * 6 + ["Finance"] * 6,
            "industry": ["Soft"] * 3 + ["Semi"] * 3 + ["Bank"] * 6,
            "market_cap_bucket": ["mid"] * 12,
        }
    )
    cfg = EvaluationSplitConfig(
        symbol_train_ratio=0.7,
        symbol_val_ratio=0.15,
        symbol_test_ratio=0.15,
        stratify_by=["sector", "industry", "market_cap_bucket"],
        test_only_symbols=["S001", "S010"],
    )
    splits = stratified_symbol_split(df, cfg, seed=123)

    all_symbols = set(df["symbol"].tolist())
    train = set(splits.train_symbols)
    val = set(splits.val_symbols)
    test = set(splits.test_symbols)

    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert train | val | test == all_symbols

    assert "S001" in splits.test_symbols
    assert "S010" in splits.test_symbols
    assert "S001" not in splits.train_symbols
    assert "S001" not in splits.val_symbols
