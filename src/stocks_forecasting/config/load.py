"""Config file loading (YAML/JSON) with environment variable expansion."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import yaml

from stocks_forecasting.config.models import ForecastingConfig


class ConfigError(ValueError):
    """Raised when a config file cannot be loaded or validated."""


def load_config(path: str | Path) -> ForecastingConfig:
    """Load and validate a forecasting config from a YAML or JSON file."""

    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise ConfigError(f"Config file not found: {resolved_path}")

    raw_text = resolved_path.read_text(encoding="utf-8")
    if resolved_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            raw = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML: {exc}") from exc
    elif resolved_path.suffix.lower() == ".json":
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON: {exc}") from exc
    else:
        raise ConfigError("Unsupported config format (use .yaml, .yml, or .json)")

    if not isinstance(raw, dict):
        raise ConfigError("Config file must parse to an object at the root level")

    expanded = _expand_env_vars(raw)
    try:
        return ForecastingConfig.model_validate(expanded)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value
