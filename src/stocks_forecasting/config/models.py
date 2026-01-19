"""Pydantic models for training/inference configuration."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator


class RunMode(str, Enum):
    """Training/inference run mode."""

    evaluation = "evaluation"
    production = "production"


class ProjectConfig(BaseModel):
    """Project-level settings: name, run mode, random seed."""

    name: str = Field(min_length=1, default="stocks-forecasting")
    mode: RunMode = RunMode.evaluation
    random_seed: int = 42
    run_id: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))


class PostgresConfig(BaseModel):
    """PostgreSQL connection settings. DSN takes precedence if provided."""

    dsn: str | None = None
    host: str = "localhost"
    port: int = Field(ge=1, le=65535, default=5432)
    database: str = "stocks"
    user: str = "postgres"
    password: SecretStr = SecretStr("postgres")
    db_schema: str = Field(default="stocks", validation_alias=AliasChoices("db_schema", "schema"))
    sslmode: str = Field(default="disable", min_length=1)


class UniverseSource(str, Enum):
    """Source for the stock universe."""

    active_db = "active_db"


class UniverseConfig(BaseModel):
    """Stock universe selection: source, explicit symbols, or limit."""

    source: UniverseSource = UniverseSource.active_db
    symbols: list[str] | None = None
    limit_symbols: int | None = Field(default=None, gt=0)


class DataConfig(BaseModel):
    """Data source settings: price type and minimum history requirements."""

    price_type: str = Field(default="1d", min_length=1)
    min_history_years: int = Field(default=2, ge=1)


class TargetKind(str, Enum):
    """Target variable type for forecasting."""

    log_return = "log_return"


class PriceField(str, Enum):
    """Price field to use for target computation."""

    close = "close"


class TargetConfig(BaseModel):
    """Target variable configuration."""

    kind: TargetKind = TargetKind.log_return
    price_field: PriceField = PriceField.close


class MacdConfig(BaseModel):
    """MACD indicator parameters."""

    fast: int = Field(default=12, gt=0)
    slow: int = Field(default=26, gt=0)
    signal: int = Field(default=9, gt=0)

    @model_validator(mode="after")
    def _validate_macd(self) -> "MacdConfig":
        if self.fast >= self.slow:
            raise ValueError("MACD requires fast < slow")
        return self


class BollingerConfig(BaseModel):
    """Bollinger Bands parameters."""

    window: int = Field(default=20, gt=0)
    num_standard_deviations: float = Field(
        default=2.0,
        gt=0,
        validation_alias=AliasChoices("num_standard_deviations", "num_std", "num_stddev"),
    )


class TechnicalIndicatorsConfig(BaseModel):
    """Technical indicator settings: SMAs, EMAs, MACD, RSI, Bollinger, ATR, OBV."""

    sma_windows: list[int] = Field(default_factory=lambda: [20, 50, 200])
    ema_windows: list[int] = Field(default_factory=lambda: [12, 26])
    macd: MacdConfig = Field(default_factory=MacdConfig)
    rsi_period: int = Field(default=14, gt=0)
    bollinger: BollingerConfig = Field(default_factory=BollingerConfig)
    atr_period: int = Field(default=14, gt=0)
    include_obv: bool = True
    volume_sma_window: int = Field(default=20, gt=0)

    @field_validator("sma_windows", "ema_windows")
    @classmethod
    def _validate_windows(cls, values: list[int]) -> list[int]:
        if not values:
            raise ValueError("must not be empty")
        for window in values:
            if window <= 0:
                raise ValueError("windows must be > 0")
        return sorted(set(values))


class KnownFutureConfig(BaseModel):
    """Known-future calendar features available at forecast time."""

    include_day_of_week: bool = True
    include_day_of_month: bool = True
    include_week_of_year: bool = True
    include_month: bool = True
    include_quarter: bool = True
    include_is_month_end: bool = True
    include_is_quarter_end: bool = True
    include_holidays: bool = True


class CalendarProvider(str, Enum):
    """Trading calendar provider."""

    python_exchange_calendars = "python_exchange_calendars"
    naive_weekdays = "naive_weekdays"


class UnknownExchangePolicy(str, Enum):
    """Policy when exchange MIC is unknown or unmapped."""

    error = "error"
    fallback_naive_weekdays = "fallback_naive_weekdays"
    fallback_default_calendar = "fallback_default_calendar"


class CalendarConfig(BaseModel):
    """Trading calendar provider and exchange mapping configuration."""

    provider: CalendarProvider = CalendarProvider.python_exchange_calendars
    fallback_calendar: str | None = Field(default=None, validation_alias=AliasChoices("fallback_calendar", "default_calendar"))
    exchange_calendar_map: dict[str, str] = Field(default_factory=dict)
    unknown_exchange_policy: UnknownExchangePolicy = UnknownExchangePolicy.fallback_naive_weekdays

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return {
            "exchange_calendars": CalendarProvider.python_exchange_calendars,
            "weekdays_only": CalendarProvider.naive_weekdays,
        }.get(value, value)

    @field_validator("unknown_exchange_policy", mode="before")
    @classmethod
    def _normalize_unknown_exchange_policy(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return {
            "weekdays_only": UnknownExchangePolicy.fallback_naive_weekdays,
            "use_default_calendar": UnknownExchangePolicy.fallback_default_calendar,
        }.get(value, value)


class FeaturesConfig(BaseModel):
    """Feature engineering configuration: target, indicators, calendar."""

    target: TargetConfig = Field(default_factory=TargetConfig)
    technical_indicators: TechnicalIndicatorsConfig = Field(default_factory=TechnicalIndicatorsConfig)
    known_future: KnownFutureConfig = Field(default_factory=KnownFutureConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)


class EvaluationSplitConfig(BaseModel):
    """Train/val/test split parameters for time and symbols."""

    train_months: int = Field(default=18, gt=0)
    val_months: int = Field(default=3, gt=0)
    test_months: int = Field(default=3, gt=0)
    gap_days: int = Field(default=0, ge=0)
    symbol_train_ratio: float = Field(default=0.70, gt=0, lt=1)
    symbol_val_ratio: float = Field(default=0.15, gt=0, lt=1)
    symbol_test_ratio: float = Field(default=0.15, gt=0, lt=1)
    test_only_symbols: list[str] = Field(default_factory=list)
    stratify_by: list[str] = Field(default_factory=lambda: ["sector", "industry", "market_cap_bucket"])

    @field_validator("test_only_symbols")
    @classmethod
    def _normalize_test_only_symbols(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            symbol = str(value).strip()
            if not symbol:
                continue
            cleaned.append(symbol)
        return sorted(set(cleaned))

    @model_validator(mode="after")
    def _validate_symbol_ratios(self) -> "EvaluationSplitConfig":
        total = self.symbol_train_ratio + self.symbol_val_ratio + self.symbol_test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("symbol split ratios must sum to 1.0")
        return self


class SplitConfig(BaseModel):
    """Data splitting configuration."""

    evaluation: EvaluationSplitConfig = Field(default_factory=EvaluationSplitConfig)


class ModelType(str, Enum):
    """Supported model architectures."""

    tft = "tft"


class ModelConfig(BaseModel):
    """Model architecture and prediction settings."""

    type: ModelType = ModelType.tft
    horizon_days: int = Field(default=30, ge=1, le=30)
    input_chunk_length: int = Field(default=252, ge=30, le=2000)
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])

    @field_validator("quantiles")
    @classmethod
    def _validate_quantiles(cls, values: list[float]) -> list[float]:
        if not values:
            raise ValueError("quantiles must not be empty")
        for q in values:
            if q <= 0 or q >= 1:
                raise ValueError("quantiles must be between 0 and 1 (exclusive)")
        return sorted(set(values))


class TrainingConfig(BaseModel):
    """Training hyperparameters: epochs, batch size, learning rate."""

    max_epochs: int = Field(default=50, gt=0)
    batch_size: int = Field(default=64, gt=0)
    learning_rate: float = Field(default=5e-4, gt=0)
    early_stopping_patience: int = Field(default=5, ge=0)


class ArtifactsConfig(BaseModel):
    """Model artifact storage configuration."""

    root_dir: str = Field(default="artifacts", min_length=1)
    bundle_name: str = Field(default="tft_log_returns", min_length=1, validation_alias=AliasChoices("bundle_name", "model_name"))


class ForecastingConfig(BaseModel):
    """Root configuration for the forecasting system."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    database: PostgresConfig = Field(default_factory=PostgresConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
