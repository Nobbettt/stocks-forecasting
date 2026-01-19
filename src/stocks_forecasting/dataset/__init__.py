"""Dataset assembly and split logic."""

from stocks_forecasting.dataset.metadata import add_market_cap_bucket
from stocks_forecasting.dataset.prepare_symbol import build_symbol_feature_frame
from stocks_forecasting.dataset.splits import SymbolSplits, stratified_symbol_split
from stocks_forecasting.dataset.time_splits import TimeSplit, compute_time_split
from stocks_forecasting.dataset.universe import build_training_universe

__all__ = [
    "SymbolSplits",
    "TimeSplit",
    "add_market_cap_bucket",
    "build_symbol_feature_frame",
    "build_training_universe",
    "compute_time_split",
    "stratified_symbol_split",
]

