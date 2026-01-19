"""Configuration models and loaders for the forecasting system."""

from stocks_forecasting.config.models import ForecastingConfig
from stocks_forecasting.config.load import load_config

__all__ = ["ForecastingConfig", "load_config"]

