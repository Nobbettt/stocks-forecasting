"""Stratified symbol splitting for train/val/test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from stocks_forecasting.config.models import EvaluationSplitConfig


class SplitError(ValueError):
    """Raised when symbol splitting fails."""


@dataclass(frozen=True, slots=True)
class SymbolSplits:
    """Result of stratified symbol split."""

    train_symbols: list[str]
    val_symbols: list[str]
    test_symbols: list[str]

    def all_symbols(self) -> list[str]:
        """Return all symbols across all splits."""
        return sorted(set(self.train_symbols + self.val_symbols + self.test_symbols))


def stratified_symbol_split(metadata: pd.DataFrame, config: EvaluationSplitConfig, *, seed: int) -> SymbolSplits:
    """Split symbols into train/val/test stratified by sector/industry/market cap."""
    if metadata.empty:
        raise SplitError("No symbols available for splitting")
    if "symbol" not in metadata.columns:
        raise SplitError("metadata must contain a `symbol` column")

    df: pd.DataFrame = metadata.copy()
    df["symbol"] = df["symbol"].astype(str)

    forced_test: set[str] = {str(s).strip() for s in config.test_only_symbols if str(s).strip()}
    if forced_test:
        all_symbols: set[str] = set(df["symbol"].tolist())
        unknown: list[str] = sorted(forced_test - all_symbols)
        if unknown:
            raise SplitError(f"test_only_symbols contain unknown symbols (not in metadata): {unknown[:10]}")
        df = df[~df["symbol"].isin(forced_test)].copy()
        if df.empty:
            raise SplitError("All symbols are marked as test-only; nothing left to split into train/val")

    stratify_by: Final[list[str]] = list(config.stratify_by)
    for column in stratify_by:
        if column not in df.columns:
            df[column] = "unknown"
        df[column] = df[column].fillna("unknown").astype(str)

    df["_stratum_key"] = df[stratify_by].agg("|".join, axis=1)

    rng: np.random.Generator = np.random.default_rng(seed)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    for _, group in df.groupby("_stratum_key", sort=True):
        symbols: list[str] = group["symbol"].astype(str).tolist()
        symbols = sorted(set(symbols))
        rng.shuffle(symbols)

        n_train: int
        n_val: int
        n_test: int
        n_train, n_val, n_test = _allocate_split_counts(
            len(symbols),
            train_ratio=config.symbol_train_ratio,
            val_ratio=config.symbol_val_ratio,
            test_ratio=config.symbol_test_ratio,
        )
        train.extend(symbols[:n_train])
        val.extend(symbols[n_train : n_train + n_val])
        test.extend(symbols[n_train + n_val : n_train + n_val + n_test])

    train_set: set[str] = set(train)
    val_set: set[str] = set(val)
    test_set: set[str] = set(test)

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise SplitError("Symbol split produced overlap between splits")

    all_symbols_final: set[str] = set(df["symbol"].astype(str).tolist())
    if forced_test:
        all_symbols_final |= forced_test
        test_set |= forced_test

    if train_set | val_set | test_set != all_symbols_final:
        missing: list[str] = sorted(all_symbols_final - (train_set | val_set | test_set))
        raise SplitError(f"Symbol split did not include all symbols; missing: {missing[:10]}")

    return SymbolSplits(train_symbols=sorted(train_set), val_symbols=sorted(val_set), test_symbols=sorted(test_set))


def _allocate_split_counts(n: int, *, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    """Allocate n items to train/val/test splits by ratio (handles rounding)."""
    if n < 0:
        raise SplitError("n must be >= 0")
    if n == 0:
        return 0, 0, 0

    ratios: np.ndarray = np.array([train_ratio, val_ratio, test_ratio], dtype="float64")
    if np.any(ratios <= 0):
        raise SplitError("split ratios must be > 0")
    ratios = ratios / ratios.sum()

    raw: np.ndarray = ratios * n
    base: np.ndarray = np.floor(raw).astype(int)
    remainder: int = n - int(base.sum())
    if remainder:
        frac: np.ndarray = raw - base
        order: np.ndarray = np.argsort(-frac)
        for i in range(remainder):
            base[int(order[i % 3])] += 1

    train_n: int
    val_n: int
    test_n: int
    train_n, val_n, test_n = (int(base[0]), int(base[1]), int(base[2]))
    if train_n + val_n + test_n != n:
        raise SplitError("internal error: split counts do not sum to n")
    return train_n, val_n, test_n
