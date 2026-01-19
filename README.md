# Stocks Forecasting

Temporal Fusion Transformer (TFT) forecasting system for the Stocks platform.

This repo is intended to be used as a git submodule under `repos/` in the main `stocks` harness repo.

## Development (UV)

Create a virtualenv and install the package in editable mode:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Config

Example config: `configs/config.example.yaml`

Validate the config:

```bash
stocks-forecasting validate-config --config configs/config.example.yaml
```

### Trading Calendar & Holidays

This project does not rely on `stocks.market_holidays` (the table is currently empty).

- `features.calendar.provider`:
  - `python_exchange_calendars`: use the `exchange_calendars` Python library (pip package `exchange-calendars`) for exchange trading sessions and holidays
  - `naive_weekdays`: assume Monday–Friday are trading days (no holidays)
- `features.calendar.unknown_exchange_policy`: controls fallback behavior when an exchange cannot be mapped to a calendar.

## Useful commands

```bash
stocks-forecasting db-info --config configs/config.example.yaml
stocks-forecasting price-summary --config configs/config.example.yaml --symbol AAPL
stocks-forecasting target-preview --config configs/config.example.yaml --symbol AAPL
stocks-forecasting indicators-preview --config configs/config.example.yaml --symbol AAPL
stocks-forecasting known-future-preview --config configs/config.example.yaml --symbol AAPL --exchange-mic XNYS
stocks-forecasting make-splits --config configs/config.example.yaml --summary
stocks-forecasting prepare-symbol --config configs/config.example.yaml --symbol AAPL
```

## Training

Training requires Darts + PyTorch (optional dependency group `train`):

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,train]"
```

Train and write an artifacts bundle:

```bash
stocks-forecasting train --config configs/config.example.yaml
```
