"""CLI entrypoints for the forecasting system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stocks_forecasting.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="stocks-forecasting")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate: argparse.ArgumentParser = subparsers.add_parser("validate-config", help="Validate a config file")
    validate.add_argument("--config", type=Path, required=True)

    db_info: argparse.ArgumentParser = subparsers.add_parser("db-info", help="Check DB connectivity and print basic info")
    db_info.add_argument("--config", type=Path, required=True)
    db_info.add_argument("--sample", type=int, default=10, help="Number of symbols to include as a sample")

    price_summary: argparse.ArgumentParser = subparsers.add_parser("price-summary", help="Summarize price history for a symbol")
    price_summary.add_argument("--config", type=Path, required=True)
    price_summary.add_argument("--symbol", required=True)

    target_preview: argparse.ArgumentParser = subparsers.add_parser("target-preview", help="Compute and preview the log-return target series")
    target_preview.add_argument("--config", type=Path, required=True)
    target_preview.add_argument("--symbol", required=True)
    target_preview.add_argument("--head", type=int, default=3)
    target_preview.add_argument("--tail", type=int, default=3)

    indicators_preview: argparse.ArgumentParser = subparsers.add_parser(
        "indicators-preview", help="Compute and preview technical indicators for a symbol"
    )
    indicators_preview.add_argument("--config", type=Path, required=True)
    indicators_preview.add_argument("--symbol", required=True)
    indicators_preview.add_argument("--head", type=int, default=3)
    indicators_preview.add_argument("--tail", type=int, default=3)

    known_future_preview: argparse.ArgumentParser = subparsers.add_parser(
        "known-future-preview", help="Compute and preview known-future calendar features"
    )
    known_future_preview.add_argument("--config", type=Path, required=True)
    known_future_preview.add_argument(
        "--exchange-mic",
        default=None,
        help="Optional exchange MIC (e.g. XNYS); required for python_exchange_calendars provider",
    )
    known_future_preview.add_argument("--symbol", required=True)
    known_future_preview.add_argument("--horizon", type=int, default=30)
    known_future_preview.add_argument("--head", type=int, default=3)
    known_future_preview.add_argument("--tail", type=int, default=3)

    splits: argparse.ArgumentParser = subparsers.add_parser("make-splits", help="Create evaluation symbol/time splits from the DB universe")
    splits.add_argument("--config", type=Path, required=True)
    splits.add_argument("--format", choices=["json"], default="json")
    splits.add_argument("--output", type=Path, default=None, help="Optional path to write the full splits JSON")
    splits.add_argument(
        "--summary",
        action="store_true",
        help="Print a small summary instead of the full symbol lists/time_splits",
    )

    prepare_symbol: argparse.ArgumentParser = subparsers.add_parser(
        "prepare-symbol", help="Build merged feature frame (observed + horizon future rows) for a symbol"
    )
    prepare_symbol.add_argument("--config", type=Path, required=True)
    prepare_symbol.add_argument("--symbol", required=True)
    prepare_symbol.add_argument("--head", type=int, default=3)
    prepare_symbol.add_argument("--tail", type=int, default=3)

    train: argparse.ArgumentParser = subparsers.add_parser("train", help="Train a model and write an artifacts bundle")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument(
        "--artifacts-root",
        type=Path,
        default=None,
        help="Optional override for artifacts root directory (defaults to artifacts.root_dir in config)",
    )

    dump: argparse.ArgumentParser = subparsers.add_parser("print-config", help="Print a validated config (secrets masked)")
    dump.add_argument("--config", type=Path, required=True)
    dump.add_argument("--format", choices=["json"], default="json")

    schema: argparse.ArgumentParser = subparsers.add_parser("schema", help="Print the config JSON schema")
    schema.add_argument("--format", choices=["json"], default="json")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse_args(argv)

    if args.command == "validate-config":
        _ = load_config(args.config)
        print("OK")
        return 0

    if args.command == "db-info":
        config = load_config(args.config)
        from stocks_forecasting.db import PostgresClient

        client = PostgresClient(config.database)
        count = client.count_active_symbols()
        sample = client.fetch_active_symbols(limit=args.sample) if args.sample > 0 else []
        print(
            json.dumps(
                {
                    "database": {
                        "host": config.database.host,
                        "port": config.database.port,
                        "database": config.database.database,
                        "db_schema": config.database.db_schema,
                    },
                    "active_symbols": {"count": count, "sample": sample},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "price-summary":
        config = load_config(args.config)
        from stocks_forecasting.db import PostgresClient

        client = PostgresClient(config.database)
        summary = client.fetch_price_summary(args.symbol, price_type=config.data.price_type)
        print(
            json.dumps(
                {
                    "symbol": summary.symbol,
                    "price_type": config.data.price_type,
                    "rows": summary.rows,
                    "start_time": summary.start_time.isoformat() if summary.start_time else None,
                    "end_time": summary.end_time.isoformat() if summary.end_time else None,
                    "last_time": summary.last_time.isoformat() if summary.last_time else None,
                    "last_close": summary.last_close,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "target-preview":
        config = load_config(args.config)
        from stocks_forecasting.db import PostgresClient
        from stocks_forecasting.features import compute_log_return_target
        import pandas as pd

        client = PostgresClient(config.database)
        prices = client.fetch_daily_prices(args.symbol, price_type=config.data.price_type)
        target = compute_log_return_target(prices, price_column=config.features.target.price_field.value)
        nan_count = int(target["log_return"].isna().sum()) if not target.empty else 0

        def format_records(frame: pd.DataFrame) -> list[dict[str, object]]:
            if frame.empty:
                return []
            rows = frame.copy()
            rows["time"] = rows["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            return list(rows.to_dict(orient="records"))

        head = format_records(target.head(max(args.head, 0)))
        tail = format_records(target.tail(max(args.tail, 0)))

        print(
            json.dumps(
                {
                    "symbol": args.symbol,
                    "price_type": config.data.price_type,
                    "rows": int(len(target)),
                    "nan_log_returns": nan_count,
                    "head": head,
                    "tail": tail,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "indicators-preview":
        config = load_config(args.config)
        from stocks_forecasting.db import PostgresClient
        from stocks_forecasting.features import compute_technical_indicators
        import pandas as pd

        client = PostgresClient(config.database)
        prices = client.fetch_daily_prices(args.symbol, price_type=config.data.price_type)
        indicators = compute_technical_indicators(prices, config.features.technical_indicators)

        def format_records(frame: pd.DataFrame) -> list[dict[str, object]]:
            if frame.empty:
                return []
            rows = frame.copy()
            if "time" in rows.columns:
                rows["time"] = pd.to_datetime(rows["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            return list(rows.to_dict(orient="records"))

        print(
            json.dumps(
                {
                    "symbol": args.symbol,
                    "price_type": config.data.price_type,
                    "rows": int(len(indicators)),
                    "columns": [c for c in indicators.columns if c != "time"],
                    "head": format_records(indicators.head(max(args.head, 0))),
                    "tail": format_records(indicators.tail(max(args.tail, 0))),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "known-future-preview":
        config = load_config(args.config)
        from stocks_forecasting.calendars import build_trading_calendar
        from stocks_forecasting.db import PostgresClient
        from stocks_forecasting.features import compute_known_future_features
        import pandas as pd

        if args.horizon < 1:
            raise SystemExit("--horizon must be >= 1")

        client = PostgresClient(config.database)
        prices = client.fetch_daily_prices(args.symbol, price_type=config.data.price_type)
        if prices.empty:
            raise SystemExit(f"No prices found for {args.symbol} (price_type={config.data.price_type})")

        times = pd.to_datetime(prices["time"], utc=True).sort_values()
        last_time = times.iloc[-1]
        calendar = build_trading_calendar(config.features.calendar, exchange_mic=args.exchange_mic)
        future = calendar.next_sessions(last_time, args.horizon)
        all_times = pd.DatetimeIndex(times).union(future).sort_values()

        features = compute_known_future_features(all_times, config.features.known_future, calendar=calendar)

        def format_records(frame: pd.DataFrame) -> list[dict[str, object]]:
            if frame.empty:
                return []
            rows = frame.copy()
            rows["time"] = pd.to_datetime(rows["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            return list(rows.to_dict(orient="records"))

        print(
            json.dumps(
                {
                    "symbol": args.symbol,
                    "price_type": config.data.price_type,
                    "last_observed_time": pd.to_datetime(last_time, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "horizon": args.horizon,
                    "exchange_mic": args.exchange_mic,
                    "rows": int(len(features)),
                    "columns": [c for c in features.columns if c != "time"],
                    "head": format_records(features.head(max(args.head, 0))),
                    "tail": format_records(features.tail(max(args.tail, 0))),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "make-splits":
        config = load_config(args.config)
        from stocks_forecasting.dataset import build_training_universe, compute_time_split, stratified_symbol_split
        from stocks_forecasting.db import PostgresClient

        client = PostgresClient(config.database)
        universe = build_training_universe(client, config)

        symbol_splits = stratified_symbol_split(universe, config.split.evaluation, seed=config.project.random_seed)

        test_only = set(config.split.evaluation.test_only_symbols)
        eligible_for_cutoff = universe if not test_only else universe[~universe["symbol"].isin(test_only)].copy()
        if eligible_for_cutoff.empty:
            raise SystemExit("No eligible symbols left to determine as_of_date cutoff (all are test_only_symbols)")
        as_of_date = eligible_for_cutoff["end_time"].min()
        time_split = compute_time_split(as_of_date, config.split.evaluation)

        payload: dict[str, object] = {
            "universe": {
                "price_type": config.data.price_type,
                "min_history_years": config.data.min_history_years,
                "eligible_symbols": int(len(universe)),
            },
            "as_of_date": as_of_date.isoformat() if as_of_date is not None else None,
            "symbol_splits": {
                "train": symbol_splits.train_symbols,
                "val": symbol_splits.val_symbols,
                "test": symbol_splits.test_symbols,
            },
            "time_split": time_split.to_iso_dict(),
        }

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if args.summary:
            summary_payload = {
                "universe": payload["universe"],
                "symbol_split_counts": {
                    "train": len(symbol_splits.train_symbols),
                    "val": len(symbol_splits.val_symbols),
                    "test": len(symbol_splits.test_symbols),
                },
                "as_of_date": payload["as_of_date"],
                "time_split": payload["time_split"],
                "output": str(args.output) if args.output is not None else None,
            }
            print(json.dumps(summary_payload, indent=2, sort_keys=True))
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "prepare-symbol":
        config = load_config(args.config)
        from stocks_forecasting.calendars import build_trading_calendar
        from stocks_forecasting.dataset import build_symbol_feature_frame
        from stocks_forecasting.db import PostgresClient
        import pandas as pd

        client = PostgresClient(config.database)
        meta = client.fetch_stock_metadata(symbols=[args.symbol])
        if meta.empty:
            raise SystemExit(f"No metadata found for symbol: {args.symbol}")
        exchange_mic = meta["exchange_mic"].iloc[0]
        calendar = build_trading_calendar(config.features.calendar, exchange_mic=exchange_mic)

        prices = client.fetch_daily_prices(args.symbol, price_type=config.data.price_type)
        if prices.empty:
            raise SystemExit(f"No prices found for {args.symbol} (price_type={config.data.price_type})")

        built = build_symbol_feature_frame(prices, config, calendar=calendar)
        frame = built.frame

        def format_records(frame: pd.DataFrame) -> list[dict[str, object]]:
            if frame.empty:
                return []
            rows = frame.copy()
            rows["time"] = pd.to_datetime(rows["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            return list(rows.to_dict(orient="records"))

        missing = {col: int(frame[col].isna().sum()) for col in frame.columns if col != "time"}
        payload = {
            "symbol": args.symbol,
            "exchange_mic": exchange_mic,
            "price_type": config.data.price_type,
            "observed_end": built.observed_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "horizon": built.horizon,
            "rows": int(len(frame)),
            "columns": [c for c in frame.columns if c != "time"],
            "missing_counts": missing,
            "head": format_records(frame.head(max(args.head, 0))),
            "tail": format_records(frame.tail(max(args.tail, 0))),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "train":
        config = load_config(args.config)
        if config.model.type.value != "tft":
            raise SystemExit(f"Unsupported model type: {config.model.type.value}")

        from stocks_forecasting.training.train_tft import TrainingError, train_tft

        try:
            paths = train_tft(config, artifacts_root=args.artifacts_root)
        except TrainingError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        payload = {
            "bundle": {
                "root": str(paths.root),
                "manifest_path": str(paths.manifest_path),
                "config_snapshot_path": str(paths.config_snapshot_path),
                "metrics_path": str(paths.metrics_path),
                "model_dir": str(paths.model_dir),
            }
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "print-config":
        config = load_config(args.config)
        print(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0

    if args.command == "schema":
        from stocks_forecasting.config.models import ForecastingConfig

        print(json.dumps(ForecastingConfig.model_json_schema(), indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")
