# Stocks Forecasting System Architecture

A Temporal Fusion Transformer (TFT) based time-series forecasting system for predicting stock price movements.

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Procedural Workflow](#procedural-workflow)
  - [Phase 1: Configuration Loading](#phase-1-configuration-loading)
  - [Phase 2: Universe Construction](#phase-2-universe-construction)
  - [Phase 3: Train/Val/Test Splitting](#phase-3-trainvaltest-splitting)
  - [Phase 4: Feature Engineering](#phase-4-feature-engineering)
  - [Phase 5: Time Series Construction](#phase-5-time-series-construction)
  - [Phase 6: Model Training](#phase-6-model-training)
  - [Phase 7: Evaluation](#phase-7-evaluation)
  - [Phase 8: Artifact Bundle Creation](#phase-8-artifact-bundle-creation)
- [Component Reference](#component-reference)
- [Configuration Guide](#configuration-guide)
- [CLI Commands](#cli-commands)
- [Output Artifacts](#output-artifacts)

---

## Overview

The stocks-forecasting system predicts **30-day stock price log-returns** using a Temporal Fusion Transformer model with quantile regression. Key capabilities:

- **Quantile Predictions**: Generates confidence intervals via multiple quantiles (default: 10th, 50th, 90th percentiles)
- **Multi-Symbol Training**: Trains on multiple stocks simultaneously with stratified splitting
- **Rich Feature Engineering**: Technical indicators + calendar features + static metadata
- **Rigorous Evaluation**: Temporal and symbol-based holdout testing
- **Reproducible Artifacts**: Versioned model bundles with config snapshots

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Configuration | Pydantic 2.6+ |
| ML Framework | Darts (PyTorch backend) |
| Model | Temporal Fusion Transformer |
| Database | PostgreSQL |
| Calendars | exchange-calendars library |
| Data Processing | Pandas, NumPy |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Entry Point                                 │
│                         stocks-forecasting train                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Configuration Loading                               │
│                                                                              │
│   config.yaml ──► YAML Parser ──► Env Expansion ──► Pydantic Validation     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Universe Construction                               │
│                                                                              │
│   PostgreSQL ──► Active Symbols ──► Metadata Join ──► History Filter        │
│                                                                              │
│   Tables: stocks, stock_prices, sectors, industries, stock_industries       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Train/Val/Test Splitting                              │
│                                                                              │
│   ┌─────────────────────┐    ┌─────────────────────────────────────────┐   │
│   │    Time Split       │    │         Symbol Split                    │   │
│   │                     │    │                                         │   │
│   │  Train: 18 months   │    │  Stratified by:                        │   │
│   │  Val:   3 months    │    │    - sector                            │   │
│   │  Test:  3 months    │    │    - industry                          │   │
│   └─────────────────────┘    │    - market_cap_bucket                 │   │
│                              │                                         │   │
│                              │  Train: 70% │ Val: 15% │ Test: 15%     │   │
│                              └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Feature Engineering (Per Symbol)                          │
│                                                                              │
│   ┌───────────────┐   ┌───────────────────┐   ┌───────────────────────┐    │
│   │    Target     │   │ Technical Indic.  │   │   Calendar Features   │    │
│   │               │   │                   │   │                       │    │
│   │  log_return   │   │  SMA (20,50,200) │   │  day_of_week          │    │
│   │  = ln(C/C-1)  │   │  EMA (12,26)      │   │  day_of_month         │    │
│   │               │   │  MACD             │   │  week_of_year         │    │
│   └───────────────┘   │  RSI (14)         │   │  month, quarter       │    │
│                       │  Bollinger Bands  │   │  is_month/quarter_end │    │
│                       │  ATR (14)         │   │  holiday features     │    │
│                       │  OBV, Volume SMA  │   │                       │    │
│                       └───────────────────┘   └───────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Time Series Construction                                │
│                                                                              │
│   For each symbol, create Darts TimeSeries objects:                         │
│                                                                              │
│   ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│   │   Target    │  │ Past Covariates  │  │Future Covariates │              │
│   │             │  │                  │  │                  │              │
│   │ log_return  │  │ Technical        │  │ Calendar         │              │
│   │ series      │  │ indicators       │  │ features         │              │
│   └─────────────┘  └──────────────────┘  └──────────────────┘              │
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                      Static Covariates                              │   │
│   │   One-hot encoded: exchange, sector, industry, market_cap, country │   │
│   └────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Model Training                                     │
│                                                                              │
│   Temporal Fusion Transformer (TFT)                                         │
│                                                                              │
│   Input:  252 days history (past context)                                   │
│   Output: 30 days forecast (quantile predictions)                           │
│                                                                              │
│   Quantiles: [0.1, 0.5, 0.9] ──► Lower/Median/Upper bounds                  │
│   Likelihood: QuantileRegression                                            │
│   Early Stopping: patience=5                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Evaluation (Eval Mode Only)                             │
│                                                                              │
│   ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐   │
│   │Temporal Validation  │  │ Symbol Validation   │  │  Holdout Test    │   │
│   │                     │  │                     │  │                  │   │
│   │ Train symbols       │  │ Val symbols         │  │ Test symbols     │   │
│   │ Val time window     │  │ Val time window     │  │ Test time window │   │
│   └─────────────────────┘  └─────────────────────┘  └──────────────────┘   │
│                                                                              │
│   Metrics: CRPS, Pinball Loss, MAE, Coverage (% in 10-90 bounds)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Artifact Bundle                                      │
│                                                                              │
│   artifacts/tft_log_returns/{version}/                                      │
│   ├── manifest.json          # Training metadata                            │
│   ├── config.snapshot.json   # Full config at training time                 │
│   ├── metrics.json           # Evaluation results                           │
│   ├── model/                                                                │
│   │   └── tft_model.pt       # Trained model weights                        │
│   └── splits.json            # Train/val/test boundaries                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Procedural Workflow

### Phase 1: Configuration Loading

**Entry Point**: `stocks_forecasting/config/load.py`

When you run `stocks-forecasting train --config config.yaml`:

1. **File Reading**: The YAML/JSON config file is read from disk
2. **Environment Variable Expansion**: Placeholders like `${POSTGRES_PASSWORD}` are replaced with actual environment variable values
3. **Pydantic Validation**: The config is validated against the `ForecastingConfig` schema
4. **Return**: A validated, type-safe configuration object

```python
# Simplified flow
raw_yaml = read_file("config.yaml")
expanded = expand_env_vars(raw_yaml)  # ${VAR} → actual value
config = ForecastingConfig(**expanded)  # Pydantic validation
```

**Key Config Sections**:
- `database`: PostgreSQL connection settings
- `data`: Price type, minimum history requirements
- `features`: Target type, indicators, calendar settings
- `split`: Time and symbol split ratios
- `model`: TFT parameters (horizon, quantiles)
- `training`: Epochs, batch size, learning rate

---

### Phase 2: Universe Construction

**Entry Point**: `stocks_forecasting/dataset/universe.py`

The system builds a "universe" of eligible stocks for training:

1. **Connect to PostgreSQL**: Using credentials from config
2. **Fetch Active Symbols**: Query `stocks` table where `is_active = true`
3. **Fetch Metadata**: Join with sectors, industries for each symbol
4. **Fetch Price Ranges**: Get min/max dates and row counts from `stock_prices`
5. **Filter by History**: Remove symbols with less than `min_history_years` of data
6. **Add Market Cap Buckets**: Categorize into micro/small/mid/large/mega

```
PostgreSQL
    │
    ├── stocks (is_active=true)
    │      ↓
    ├── JOIN sectors, industries
    │      ↓
    ├── stock_prices (min/max dates)
    │      ↓
    └── Filter: history >= 2 years
           ↓
       Universe DataFrame
       [symbol, exchange_mic, sector, industry, market_cap_bucket, ...]
```

**Output**: A DataFrame with columns:
- `symbol`, `exchange_mic`, `country_code`, `currency`
- `sector`, `industry`, `market_cap`, `market_cap_bucket`
- `min_time`, `max_time`, `row_count`

---

### Phase 3: Train/Val/Test Splitting

**Entry Points**:
- `stocks_forecasting/dataset/time_splits.py`
- `stocks_forecasting/dataset/splits.py`

In evaluation mode, the system creates two types of splits:

#### Time Split

Works backward from the latest available date:

```
                    │◄─────── 18 mo ───────►│◄─ 3 mo ─►│◄─ 3 mo ─►│
Timeline: ─────────────────────────────────────────────────────────►
                    │      TRAIN            │   VAL    │   TEST   │
                    │                       │          │          │
                 train_start            val_start   test_start  as_of_date
```

#### Symbol Split (Stratified Splitting)

**What is Stratified Splitting?**

Stratified splitting ensures that each split (train/val/test) has the **same proportional representation** of different categories as the original dataset. Instead of randomly assigning symbols to splits, the algorithm groups symbols by their characteristics first, then samples proportionally from each group.

**Why Stratified Splitting Matters:**

Without stratification, random splitting could produce imbalanced splits:

```
PROBLEM: Random Split (unstratified)
─────────────────────────────────────
Original Universe:     Train (70%):         Test (15%):
  Tech: 40%             Tech: 75%  ⚠️        Tech: 10%  ⚠️
  Healthcare: 30%       Healthcare: 15%      Healthcare: 60%
  Finance: 30%          Finance: 10%         Finance: 30%

→ Model trains mostly on Tech, tested mostly on Healthcare
→ Evaluation doesn't reflect real-world performance
```

With stratification, proportions are preserved:

```
SOLUTION: Stratified Split
──────────────────────────
Original Universe:     Train (70%):         Test (15%):
  Tech: 40%             Tech: 40%  ✓         Tech: 40%  ✓
  Healthcare: 30%       Healthcare: 30%      Healthcare: 30%
  Finance: 30%          Finance: 30%         Finance: 30%

→ Each split mirrors the original distribution
→ Fair evaluation across all sectors
```

**How It Works:**

1. **Group symbols** by stratification columns (sector + industry + market_cap_bucket)
2. **Within each group**, randomly assign symbols to train/val/test
3. **Each group contributes proportionally** to each split

```
Example with 100 symbols:
─────────────────────────
Group: "Technology / Software / Large-cap" (20 symbols)
  → Train: 14 symbols (70%)
  → Val:    3 symbols (15%)
  → Test:   3 symbols (15%)

Group: "Healthcare / Biotech / Small-cap" (10 symbols)
  → Train:  7 symbols (70%)
  → Val:    1 symbol  (15%)
  → Test:   2 symbols (15%)

... and so on for each unique combination
```

**Stratification Columns (default):**

```python
stratify_by = ["sector", "industry", "market_cap_bucket"]
```

| Column | Example Values | Purpose |
|--------|---------------|---------|
| `sector` | Technology, Healthcare, Finance | Broad market segment |
| `industry` | Software, Biotech, Banking | Specific business type |
| `market_cap_bucket` | micro, small, mid, large, mega | Company size category |

**Split Ratios (default):**

```
train_symbols: 70%  →  Used for model training
val_symbols:   15%  →  Used for hyperparameter tuning / early stopping
test_symbols:  15%  →  Held out for final evaluation (never seen during training)
```

The split is **reproducible** via `random_seed` in config—same seed produces identical splits.

**Special Handling**:
- `test_only_symbols`: Specific symbols forced into test set (never used for training)
- Empty strata: Groups with few symbols are handled gracefully (may get 0 in some splits)

---

### Phase 4: Feature Engineering

**Entry Points**:
- `stocks_forecasting/features/target.py`
- `stocks_forecasting/features/technical_indicators.py`
- `stocks_forecasting/features/known_future.py`

For each symbol, the system computes:

#### 4.1 Target Variable

**Log-return**: The natural logarithm of price ratios

```
log_return_t = ln(close_t / close_{t-1})
```

- First observation is NaN (no previous price)
- Invalid/non-positive prices result in NaN
- Represents percentage change in logarithmic scale

#### 4.2 Technical Indicators (Past Covariates)

These are **observed** values that can only be known up to the current time:

| Indicator | Description | Default Parameters |
|-----------|-------------|-------------------|
| **SMA** | Simple Moving Average | windows: 20, 50, 200 |
| **EMA** | Exponential Moving Average | windows: 12, 26 |
| **MACD** | Moving Average Convergence Divergence | fast=12, slow=26, signal=9 |
| **RSI** | Relative Strength Index | period=14 |
| **Bollinger Bands** | Volatility bands | window=20, std=2.0 |
| **ATR** | Average True Range | period=14 |
| **OBV** | On-Balance Volume | - |
| **Volume SMA** | Volume moving average | window=20 |

#### 4.3 Calendar Features (Future Covariates)

These are **known in advance** and available at forecast time:

| Feature | Description |
|---------|-------------|
| `day_of_week` | 0-4 (Monday-Friday) |
| `day_of_month` | 1-31 |
| `week_of_year` | 1-52 |
| `month` | 1-12 |
| `quarter` | 1-4 |
| `is_month_end` | Boolean flag |
| `is_quarter_end` | Boolean flag |
| `sessions_since_holiday` | Days since last holiday |
| `is_day_after_holiday` | Boolean flag |

**Trading Calendar**: Uses `exchange-calendars` library for accurate trading sessions per exchange.

#### 4.4 Future Rows Generation

The system generates "future" rows for the forecast horizon:

```
┌─────────────────────────────────────────────────────────────────┐
│ Observed Data (is_future=0)                                     │
│ time | close | log_return | SMA_20 | RSI | day_of_week | ...   │
│ ─────────────────────────────────────────────────────────────── │
│ T-252│ 150.0 │   0.005    │  148.5 │ 55  │      1      │ ...   │
│ T-251│ 151.0 │   0.007    │  148.8 │ 57  │      2      │ ...   │
│  ...                                                            │
│ T    │ 180.0 │   0.003    │  175.2 │ 62  │      4      │ ...   │
├─────────────────────────────────────────────────────────────────┤
│ Future Data (is_future=1)                                       │
│ time | close | log_return | SMA_20 | RSI | day_of_week | ...   │
│ ─────────────────────────────────────────────────────────────── │
│ T+1  │  NaN  │    NaN     │  NaN   │ NaN │      0      │ ...   │
│ T+2  │  NaN  │    NaN     │  NaN   │ NaN │      1      │ ...   │
│  ...                                                            │
│ T+30 │  NaN  │    NaN     │  NaN   │ NaN │      3      │ ...   │
└─────────────────────────────────────────────────────────────────┘
```

---

### Phase 5: Time Series Construction

**Entry Point**: `stocks_forecasting/training/train_tft.py`

Each symbol's feature frame is converted to Darts `TimeSeries` objects:

```python
# For each symbol
target_series = TimeSeries.from_dataframe(
    df, time_col="time", value_cols=["log_return"]
)

past_covariates = TimeSeries.from_dataframe(
    df, time_col="time",
    value_cols=["sma_20", "ema_12", "rsi_14", "macd", ...]
)

future_covariates = TimeSeries.from_dataframe(
    df, time_col="time",
    value_cols=["day_of_week", "month", "is_month_end", ...]
)

# Static covariates (constant per symbol)
static_covariates = pd.DataFrame({
    "exchange_XNAS": [1], "exchange_XNYS": [0],
    "sector_Technology": [1], "sector_Healthcare": [0],
    ...
})
```

**One-Hot Encoding**: Static covariates are one-hot encoded. The encoder is **fit only on training symbols** to prevent data leakage.

---

### Phase 6: Model Training

**Entry Point**: `stocks_forecasting/training/train_tft.py`

The Temporal Fusion Transformer is configured and trained:

```python
model = TFTModel(
    input_chunk_length=252,    # ~1 year of trading days
    output_chunk_length=30,    # 30-day forecast
    likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
    batch_size=64,
    n_epochs=50,
    optimizer_kwargs={"lr": 0.0005},
    add_encoders=None,         # Using manual covariates
)

model.fit(
    series=training_series,           # List of target series
    past_covariates=past_cov_list,    # Technical indicators
    future_covariates=future_cov_list, # Calendar features
)
```

**Training Dynamics**:
- **Early Stopping**: Monitors validation loss, stops after 5 epochs without improvement
- **Batch Processing**: Processes multiple symbols simultaneously
- **GPU Acceleration**: Uses CUDA if available

---

### Phase 7: Evaluation

**Entry Point**: `stocks_forecasting/training/train_tft.py` (eval mode)

In evaluation mode, the model is tested on three segments:

#### 7.1 Temporal Validation
- **Symbols**: Training symbols
- **Time Window**: Validation period
- **Purpose**: Tests if model generalizes to future time periods

#### 7.2 Symbol Validation
- **Symbols**: Validation symbols (unseen during training)
- **Time Window**: Validation period
- **Purpose**: Tests if model generalizes to new stocks

#### 7.3 Holdout Test
- **Symbols**: Test symbols
- **Time Window**: Test period
- **Purpose**: True out-of-sample evaluation

#### Metrics Computed

| Metric | Description |
|--------|-------------|
| **CRPS** | Continuous Ranked Probability Score (overall forecast quality) |
| **Pinball Loss** | Per-quantile asymmetric loss |
| **MAE** | Mean Absolute Error (median quantile) |
| **Coverage** | % of actuals within 10th-90th percentile bounds |

```python
# Evaluation flow
for segment in [temporal_val, symbol_val, holdout_test]:
    forecasts = model.predict(horizon=30, series=segment_series)
    metrics = evaluate_quantile_forecasts(
        actuals=actual_returns,
        forecasts=forecasts,
        quantiles=[0.1, 0.5, 0.9]
    )
```

---

### Phase 8: Artifact Bundle Creation

**Entry Point**: `stocks_forecasting/artifacts/bundle.py`

After training (and optionally evaluation), a versioned artifact bundle is created:

```
artifacts/tft_log_returns/20250116T192345Z/
├── manifest.json
├── config.snapshot.json
├── metrics.json
├── model/
│   └── tft_model.pt
└── splits.json
```

**Version Format**: ISO 8601 UTC timestamp (e.g., `20250116T192345Z`)

---

## Component Reference

### Database Module

**Location**: `src/stocks_forecasting/db/postgres.py`

| Class/Function | Purpose |
|----------------|---------|
| `PostgresClient` | Connection pooling and query execution |
| `fetch_active_symbols()` | Get active stocks from `stocks` table |
| `fetch_stock_metadata()` | Join exchanges, sectors, industries |
| `fetch_price_ranges()` | Get min/max dates per symbol |
| `fetch_daily_prices()` | Fetch OHLCV data with time filtering |

### Features Module

**Location**: `src/stocks_forecasting/features/`

| File | Purpose |
|------|---------|
| `target.py` | Log-return computation |
| `technical_indicators.py` | SMA, EMA, MACD, RSI, Bollinger, ATR, OBV |
| `known_future.py` | Calendar/holiday features |

### Calendars Module

**Location**: `src/stocks_forecasting/calendars/`

| File | Purpose |
|------|---------|
| `base.py` | `TradingCalendar` protocol definition |
| `python_exchange_calendars.py` | Real trading calendars via library |
| `naive_weekdays.py` | Simple Mon-Fri fallback |
| `factory.py` | Calendar resolution by exchange MIC |

### Dataset Module

**Location**: `src/stocks_forecasting/dataset/`

| File | Purpose |
|------|---------|
| `universe.py` | Build eligible symbol list |
| `time_splits.py` | Compute train/val/test time boundaries |
| `splits.py` | Stratified symbol splitting |
| `prepare_symbol.py` | Assemble complete feature frame |
| `metadata.py` | Market cap bucketing |

### Training Module

**Location**: `src/stocks_forecasting/training/`

| File | Purpose |
|------|---------|
| `train_tft.py` | Main training orchestration |

### Metrics Module

**Location**: `src/stocks_forecasting/metrics/`

| File | Purpose |
|------|---------|
| `quantile.py` | Pinball loss, CRPS, coverage metrics |

### Artifacts Module

**Location**: `src/stocks_forecasting/artifacts/`

| File | Purpose |
|------|---------|
| `bundle.py` | Create and write artifact bundles |

---

## Configuration Guide

**Location**: `configs/config.example.yaml`

### Key Sections

```yaml
# Project metadata
project:
  name: stocks-forecasting
  mode: evaluation          # "evaluation" or "production"
  random_seed: 42           # For reproducibility

# Database connection
database:
  host: localhost
  port: 5432
  database: stocks
  user: postgres
  password: ${POSTGRES_PASSWORD}  # Environment variable expansion

# Data settings
data:
  price_type: 1d            # Daily prices
  min_history_years: 2      # Minimum required history

# Feature configuration
features:
  target:
    kind: log_return
    price_field: close

  technical_indicators:
    sma_windows: [20, 50, 200]
    ema_windows: [12, 26]
    macd: {fast: 12, slow: 26, signal: 9}
    rsi_period: 14
    bollinger: {window: 20, num_standard_deviations: 2.0}
    atr_period: 14
    include_obv: true
    volume_sma_window: 20

  known_future:
    include_day_of_week: true
    include_day_of_month: true
    include_week_of_year: true
    include_month: true
    include_quarter: true
    include_is_month_end: true
    include_is_quarter_end: true
    include_holidays: true

  calendar:
    provider: python_exchange_calendars
    unknown_exchange_policy: fallback_naive_weekdays

# Split configuration (evaluation mode)
split:
  evaluation:
    train_months: 18
    val_months: 3
    test_months: 3
    gap_days: 0
    symbol_train_ratio: 0.70
    symbol_val_ratio: 0.15
    symbol_test_ratio: 0.15
    stratify_by: [sector, industry, market_cap_bucket]

# Model configuration
model:
  type: tft
  input_chunk_length: 252
  horizon_days: 30
  quantiles: [0.1, 0.5, 0.9]

# Training parameters
training:
  max_epochs: 50
  batch_size: 64
  learning_rate: 0.0005
  early_stopping_patience: 5

# Artifact output
artifacts:
  root_dir: artifacts
  bundle_name: tft_log_returns
```

### Environment Variables

Use `${VAR_NAME}` syntax for sensitive values:

```yaml
database:
  password: ${POSTGRES_PASSWORD}
```

---

## CLI Commands

**Entry Point**: `stocks-forecasting` (defined in `pyproject.toml`)

### Configuration Commands

```bash
# Validate config syntax
stocks-forecasting validate-config --config config.yaml

# Print validated config (secrets masked)
stocks-forecasting print-config --config config.yaml

# Print config JSON schema
stocks-forecasting schema
```

### Data Inspection Commands

```bash
# Check database connectivity
stocks-forecasting db-info --config config.yaml

# Get price history stats for a symbol
stocks-forecasting price-summary --config config.yaml --symbol AAPL
```

### Feature Preview Commands

```bash
# Preview log-return target series
stocks-forecasting target-preview --config config.yaml --symbol AAPL

# Preview technical indicators
stocks-forecasting indicators-preview --config config.yaml --symbol AAPL

# Preview calendar features
stocks-forecasting known-future-preview --config config.yaml --symbol AAPL
```

### Dataset Commands

```bash
# Create and preview train/val/test splits
stocks-forecasting make-splits --config config.yaml --summary

# Build complete feature frame for a symbol
stocks-forecasting prepare-symbol --config config.yaml --symbol AAPL
```

### Training Command

```bash
# Train model and generate artifact bundle
stocks-forecasting train --config config.yaml

# Output: JSON with bundle paths
# {
#   "bundle": {
#     "root": "artifacts/tft_log_returns/20250116T192345Z",
#     "manifest_path": ".../manifest.json",
#     "config_snapshot_path": ".../config.snapshot.json",
#     "metrics_path": ".../metrics.json",
#     "model_dir": ".../model"
#   }
# }
```

---

## Output Artifacts

### Bundle Structure

```
artifacts/tft_log_returns/{version}/
├── manifest.json
├── config.snapshot.json
├── metrics.json
├── model/
│   └── tft_model.pt
└── splits.json
```

### manifest.json

Training metadata and configuration summary:

```json
{
  "created_at": "2025-01-16T19:23:45Z",
  "bundle_name": "tft_log_returns",
  "version": "20250116T192345Z",
  "project": {
    "version": "0.1.0",
    "mode": "evaluation",
    "random_seed": 42,
    "run_id": "abc123"
  },
  "model": {
    "type": "tft",
    "horizon_days": 30,
    "input_chunk_length": 252,
    "quantiles": [0.1, 0.5, 0.9]
  },
  "data": {
    "price_type": "1d",
    "min_history_years": 2,
    "as_of_date": "2024-12-31"
  },
  "training": {
    "symbols_used": 450,
    "symbols_skipped": 50
  },
  "splits": {
    "time_split": {
      "train_start": "2022-07-01",
      "train_end": "2024-01-01",
      "val_start": "2024-01-02",
      "val_end": "2024-04-01",
      "test_start": "2024-04-02",
      "test_end": "2024-07-01"
    },
    "symbol_split_counts": {
      "train": 315,
      "val": 67,
      "test": 68
    }
  },
  "features": {
    "target": {"kind": "log_return", "price_field": "close"},
    "past_covariates": ["sma_20", "sma_50", "ema_12", "rsi_14", ...],
    "future_covariates": ["day_of_week", "month", "is_month_end", ...],
    "static_covariates": ["exchange", "sector", "industry", "market_cap_bucket"]
  },
  "paths": {
    "model": "model/tft_model.pt",
    "splits": "splits.json"
  }
}
```

### metrics.json

Evaluation results (evaluation mode only):

```json
{
  "created_at": "2025-01-16T19:45:00Z",
  "mode": "evaluation",
  "quantiles": [0.1, 0.5, 0.9],
  "horizon": 30,
  "segments": {
    "temporal_validation": {
      "symbols": 315,
      "samples": 9450,
      "metrics": {
        "crps": 0.0123,
        "pinball_loss": {"q0.1": 0.008, "q0.5": 0.015, "q0.9": 0.009},
        "mae": 0.0187,
        "coverage": 0.82
      }
    },
    "symbol_validation": {
      "symbols": 67,
      "samples": 2010,
      "metrics": {...}
    },
    "holdout_test": {
      "symbols": 68,
      "samples": 2040,
      "metrics": {...}
    }
  }
}
```

### config.snapshot.json

Complete configuration at training time for reproducibility.

### splits.json

Train/val/test symbol assignments and time boundaries.

### model/tft_model.pt

PyTorch model weights (Darts checkpoint format).

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set environment variables
export POSTGRES_PASSWORD=your_password

# 3. Verify database connection
uv run stocks-forecasting db-info --config configs/config.example.yaml

# 4. Preview splits
uv run stocks-forecasting make-splits --config configs/config.example.yaml --summary

# 5. Train model
uv run stocks-forecasting train --config configs/config.example.yaml
```

---

## Design Decisions

### Why Log-Returns?

- **Stationarity**: Log-returns are approximately stationary, unlike raw prices
- **Interpretability**: Can be converted to percentage returns: `pct_return ≈ exp(log_return) - 1`
- **Additivity**: Multi-period returns are sums of log-returns

### Why Quantile Regression?

- **Uncertainty Quantification**: Provides full distribution, not just point estimates
- **Asymmetric Risk**: Different quantiles capture upside/downside risks
- **Robust Evaluation**: CRPS and coverage metrics assess calibration

### Why Stratified Splitting?

- **Balanced Representation**: Each split has similar sector/industry distributions
- **Prevents Leakage**: Symbols in test set never seen during training
- **Fair Evaluation**: Model isn't biased toward dominant sectors

### Why Temporal + Symbol Splits?

- **Temporal Generalization**: Tests if model works on future time periods
- **Symbol Generalization**: Tests if model works on unseen stocks
- **Realistic Evaluation**: Mimics real-world deployment scenario
