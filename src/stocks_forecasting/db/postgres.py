"""PostgreSQL client for fetching stock data and metadata."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator
from urllib.parse import quote_plus

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from stocks_forecasting.config.models import PostgresConfig


class DatabaseError(RuntimeError):
    """Raised when a database operation fails."""


@dataclass(frozen=True, slots=True)
class PriceSummary:
    """Summary statistics for a symbol's price history."""

    symbol: str
    rows: int
    start_time: datetime | None
    end_time: datetime | None
    last_time: datetime | None
    last_close: float | None


class PostgresClient:
    """Client for querying stocks database (symbols, metadata, prices)."""

    def __init__(self, config: PostgresConfig) -> None:
        self._config = config

    @property
    def config(self) -> PostgresConfig:
        return self._config

    def dsn(self) -> str:
        """Build connection string from config. DSN takes precedence if set."""
        if self._config.dsn:
            return self._config.dsn
        password: str = quote_plus(self._config.password.get_secret_value())
        sslmode: str = quote_plus(str(self._config.sslmode))
        return (
            f"postgresql://{self._config.user}:{password}"
            f"@{self._config.host}:{self._config.port}/{self._config.database}"
            f"?sslmode={sslmode}"
        )

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection]:
        """Context manager for database connections."""
        try:
            with psycopg.connect(self.dsn(), row_factory=dict_row) as conn:
                yield conn
        except psycopg.Error as exc:
            raise DatabaseError(str(exc)) from exc

    def fetch_active_symbols(self, limit: int | None = None) -> list[str]:
        """Fetch active stock symbols, optionally limited."""
        query: sql.Composed = sql.SQL("SELECT symbol FROM {}.{} WHERE is_active = true ORDER BY symbol").format(
            sql.Identifier(self._config.db_schema),
            sql.Identifier("stocks"),
        )

        if limit is not None:
            query = query + sql.SQL(" LIMIT {}").format(sql.Literal(limit))

        with self.connect() as conn:
            rows: list[dict[str, object]] = conn.execute(query).fetchall()
        return [row["symbol"] for row in rows]

    def count_active_symbols(self) -> int:
        """Count total active symbols in the database."""
        query: sql.Composed = sql.SQL("SELECT COUNT(*) AS count FROM {}.{} WHERE is_active = true").format(
            sql.Identifier(self._config.db_schema),
            sql.Identifier("stocks"),
        )
        with self.connect() as conn:
            row: dict[str, object] | None = conn.execute(query).fetchone()
        if row is None:
            return 0
        return int(row["count"])

    def fetch_stock_metadata(self, *, symbols: list[str] | None = None, limit: int | None = None) -> pd.DataFrame:
        """Fetch stock metadata (exchange, market cap, sector, industry)."""
        where: list[sql.Composed] = [sql.SQL("s.is_active = true")]
        params: dict[str, object] = {}

        if symbols is not None:
            where.append(sql.SQL("s.symbol = ANY({})").format(sql.Placeholder("symbols")))
            params["symbols"] = symbols

        query: sql.Composed = (
            sql.SQL(
                "SELECT "
                "  s.symbol, "
                "  s.exchange_mic, "
                "  s.market_cap, "
                "  s.country_code, "
                "  s.currency, "
                "  sec.name AS sector, "
                "  ind.name AS industry "
                "FROM {}.stocks s "
                "LEFT JOIN {}.stock_industries si "
                "  ON si.symbol = s.symbol AND si.is_primary = true "
                "LEFT JOIN {}.industries ind "
                "  ON ind.id = si.industry_id "
                "LEFT JOIN {}.sectors sec "
                "  ON sec.id = ind.sector_id "
                "WHERE "
            ).format(
                sql.Identifier(self._config.db_schema),
                sql.Identifier(self._config.db_schema),
                sql.Identifier(self._config.db_schema),
                sql.Identifier(self._config.db_schema),
            )
            + sql.SQL(" AND ").join(where)
            + sql.SQL(" ORDER BY s.symbol")
        )

        if limit is not None:
            query = query + sql.SQL(" LIMIT {}").format(sql.Literal(limit))

        with self.connect() as conn:
            rows: list[dict[str, object]] = conn.execute(query, params).fetchall()

        df: pd.DataFrame = pd.DataFrame(
            rows,
            columns=["symbol", "exchange_mic", "market_cap", "country_code", "currency", "sector", "industry"],
        )
        if df.empty:
            return df
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
        return df

    def fetch_price_ranges(self, *, price_type: str, symbols: list[str] | None = None) -> pd.DataFrame:
        """Fetch row count and date range per symbol for given price type."""
        where: list[sql.Composed] = [sql.SQL("s.is_active = true"), sql.SQL("sp.price_type = {}").format(sql.Placeholder("price_type"))]
        params: dict[str, object] = {"price_type": price_type}

        if symbols is not None:
            where.append(sql.SQL("sp.symbol = ANY({})").format(sql.Placeholder("symbols")))
            params["symbols"] = symbols

        query: sql.Composed = (
            sql.SQL(
                "SELECT "
                "  sp.symbol, "
                "  COUNT(*) AS rows, "
                "  MIN(sp.time) AS start_time, "
                "  MAX(sp.time) AS end_time "
                "FROM {}.stock_prices sp "
                "JOIN {}.stocks s "
                "  ON s.symbol = sp.symbol "
                "WHERE "
            ).format(sql.Identifier(self._config.db_schema), sql.Identifier(self._config.db_schema))
            + sql.SQL(" AND ").join(where)
            + sql.SQL(" GROUP BY sp.symbol ORDER BY sp.symbol")
        )

        with self.connect() as conn:
            rows: list[dict[str, object]] = conn.execute(query, params).fetchall()

        df: pd.DataFrame = pd.DataFrame(rows, columns=["symbol", "rows", "start_time", "end_time"])
        if df.empty:
            return df

        df["rows"] = pd.to_numeric(df["rows"], errors="coerce").fillna(0).astype("int64")
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
        df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
        return df

    def fetch_daily_prices(
        self,
        symbol: str,
        *,
        price_type: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV price data for a symbol within optional date range."""
        clauses: list[sql.Composed] = [
            sql.SQL("symbol = {}").format(sql.Placeholder("symbol")),
            sql.SQL("price_type = {}").format(sql.Placeholder("price_type")),
        ]

        if start_time is not None:
            clauses.append(sql.SQL("time >= {}").format(sql.Placeholder("start_time")))
        if end_time is not None:
            clauses.append(sql.SQL("time <= {}").format(sql.Placeholder("end_time")))

        query: sql.Composed = (
            sql.SQL(
                "SELECT time, open, high, low, close, volume "
                "FROM {}.{} "
                "WHERE "
            )
            .format(sql.Identifier(self._config.db_schema), sql.Identifier("stock_prices"))
            + sql.SQL(" AND ").join(clauses)
            + sql.SQL(" ORDER BY time")
        )

        params: dict[str, object] = {"symbol": symbol, "price_type": price_type}
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time

        with self.connect() as conn:
            rows: list[dict[str, object]] = conn.execute(query, params).fetchall()

        df: pd.DataFrame = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        if df.empty:
            return df

        df["time"] = pd.to_datetime(df["time"], utc=True)
        for column in ["open", "high", "low", "close"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
        return df

    def fetch_price_summary(self, symbol: str, *, price_type: str) -> PriceSummary:
        """Fetch summary statistics for a symbol's price history."""
        summary_query: sql.Composed = sql.SQL(
            "SELECT COUNT(*) AS rows, MIN(time) AS start_time, MAX(time) AS end_time "
            "FROM {}.{} "
            "WHERE symbol = {} AND price_type = {}"
        ).format(
            sql.Identifier(self._config.db_schema),
            sql.Identifier("stock_prices"),
            sql.Placeholder("symbol"),
            sql.Placeholder("price_type"),
        )

        last_query: sql.Composed = sql.SQL(
            "SELECT time, close "
            "FROM {}.{} "
            "WHERE symbol = {} AND price_type = {} "
            "ORDER BY time DESC "
            "LIMIT 1"
        ).format(
            sql.Identifier(self._config.db_schema),
            sql.Identifier("stock_prices"),
            sql.Placeholder("symbol"),
            sql.Placeholder("price_type"),
        )

        with self.connect() as conn:
            summary_row: dict[str, object] | None = conn.execute(summary_query, {"symbol": symbol, "price_type": price_type}).fetchone()
            last_row: dict[str, object] | None = conn.execute(last_query, {"symbol": symbol, "price_type": price_type}).fetchone()

        last_time: datetime | None = None
        last_close: float | None = None
        if last_row is not None:
            last_time = last_row["time"]  # type: ignore[assignment]
            last_close_raw: object = last_row["close"]
            last_close = float(last_close_raw) if last_close_raw is not None else None

        return PriceSummary(
            symbol=symbol,
            rows=int(summary_row["rows"]) if summary_row is not None else 0,
            start_time=summary_row["start_time"] if summary_row is not None else None,
            end_time=summary_row["end_time"] if summary_row is not None else None,
            last_time=last_time,
            last_close=last_close,
        )
