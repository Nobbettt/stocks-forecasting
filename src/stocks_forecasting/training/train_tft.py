"""TFT model training with Darts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_forecasting.artifacts import BundlePaths, create_bundle_paths, now_utc_iso, write_json
from stocks_forecasting.calendars import build_trading_calendar
from stocks_forecasting.config.models import ForecastingConfig, RunMode
from stocks_forecasting.dataset import build_symbol_feature_frame, build_training_universe, compute_time_split, stratified_symbol_split
from stocks_forecasting.dataset.time_splits import slice_time_window
from stocks_forecasting.db import PostgresClient
from stocks_forecasting.metrics import evaluate_quantile_forecasts


class TrainingError(RuntimeError):
    """Raised when model training fails."""


@dataclass(frozen=True, slots=True)
class PreparedSeries:
    """Prepared Darts TimeSeries for a single symbol."""

    symbol: str
    target: object
    past_covariates: object | None
    future_covariates: object | None
    static_covariates: pd.DataFrame | None
    timeline: pd.DataFrame


def train_tft(config: ForecastingConfig, *, artifacts_root: Path | None = None) -> BundlePaths:
    """Train TFT model and save versioned bundle (lazy darts import)."""
    try:
        from darts import TimeSeries  # type: ignore
        from darts.models import TFTModel  # type: ignore
        from darts.utils.likelihood_models import QuantileRegression  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TrainingError(
            "Training requires Darts with torch support. Install with:\n"
            "  uv pip install -e \".[train]\""
        ) from exc

    client: PostgresClient = PostgresClient(config.database)
    universe: pd.DataFrame = build_training_universe(client, config)

    time_split = None
    symbol_splits = None
    train_symbol_set: set[str] = set(universe["symbol"].astype(str).tolist())
    test_only: set[str] = set()

    if config.project.mode == RunMode.evaluation:
        test_only = set(config.split.evaluation.test_only_symbols)
        eligible_for_cutoff: pd.DataFrame = (
            universe if not test_only else universe[~universe["symbol"].isin(test_only)].copy()
        )
        if eligible_for_cutoff.empty:
            raise TrainingError(
                "No eligible symbols left to determine as_of_date cutoff (all are test_only_symbols)"
            )
        as_of_date: pd.Timestamp = pd.to_datetime(eligible_for_cutoff["end_time"].min(), utc=True).normalize()
        time_split = compute_time_split(as_of_date, config.split.evaluation)
        symbol_splits = stratified_symbol_split(universe, config.split.evaluation, seed=config.project.random_seed)
        train_symbol_set = set(symbol_splits.train_symbols)
    else:
        as_of_date = pd.to_datetime(universe["end_time"].min(), utc=True).normalize()

    static_cov_train_symbols: set[str] = (
        set(train_symbol_set) if config.project.mode == RunMode.evaluation else set(universe["symbol"].astype(str).tolist())
    )
    static_covariates: dict[str, pd.DataFrame] = _encode_static_covariates(universe, train_symbols=static_cov_train_symbols)

    series_by_symbol: dict[str, PreparedSeries] = {}
    skipped_train_symbols: dict[str, str] = {}
    min_train_rows: int = int(config.model.input_chunk_length + config.model.horizon_days)
    train_series: list[object] = []
    train_past_cov: list[object] | None = []
    train_future_cov: list[object] | None = []
    train_symbols_used: list[str] = []

    for row in universe.itertuples(index=False):
        symbol: str = str(getattr(row, "symbol"))
        exchange_mic: str | None = getattr(row, "exchange_mic")

        calendar = build_trading_calendar(config.features.calendar, exchange_mic=exchange_mic)
        prices: pd.DataFrame = client.fetch_daily_prices(symbol, price_type=config.data.price_type, end_time=as_of_date.to_pydatetime())
        if prices.empty:
            continue

        built = build_symbol_feature_frame(prices, config, calendar=calendar)
        frame: pd.DataFrame = built.frame.sort_values("time").reset_index(drop=True)
        frame["step"] = np.arange(len(frame), dtype="int32")

        observed: pd.DataFrame = frame[frame["is_future"] == 0].copy()

        # Remove rows with NaNs in target or past covariates (warmup window).
        required: list[str] = ["log_return", *built.past_covariate_columns]
        observed = observed.dropna(subset=required)
        if observed.empty:
            continue

        target_df: pd.DataFrame = observed[["step", "log_return"]]
        past_cov_df: pd.DataFrame | None = (
            observed[["step", *built.past_covariate_columns]] if built.past_covariate_columns else None
        )

        future_cov_df: pd.DataFrame | None = (
            frame[["step", *built.future_covariate_columns]] if built.future_covariate_columns else None
        )

        target_ts = TimeSeries.from_dataframe(target_df, time_col="step", value_cols=["log_return"]).astype(np.float32)
        past_cov_ts = (
            TimeSeries.from_dataframe(past_cov_df, time_col="step", value_cols=built.past_covariate_columns).astype(
                np.float32
            )
            if past_cov_df is not None
            else None
        )
        future_cov_ts = (
            TimeSeries.from_dataframe(future_cov_df, time_col="step", value_cols=built.future_covariate_columns).astype(
                np.float32
            )
            if future_cov_df is not None
            else None
        )

        scov: pd.DataFrame | None = static_covariates.get(symbol)
        if scov is not None:
            target_ts = target_ts.with_static_covariates(scov)
            if past_cov_ts is not None:
                past_cov_ts = past_cov_ts.with_static_covariates(scov)
            if future_cov_ts is not None:
                future_cov_ts = future_cov_ts.with_static_covariates(scov)

        series_by_symbol[symbol] = PreparedSeries(
            symbol=symbol,
            target=target_ts,
            past_covariates=past_cov_ts,
            future_covariates=future_cov_ts,
            static_covariates=scov,
            timeline=frame[["step", "time", "is_future"]].copy(),
        )

        should_train_symbol: bool = config.project.mode == RunMode.production or symbol in train_symbol_set
        if not should_train_symbol:
            continue

        training_source: pd.DataFrame = observed
        if config.project.mode == RunMode.evaluation:
            if time_split is None:
                raise TrainingError("internal error: time_split is None in evaluation mode")
            training_source = slice_time_window(observed, start=time_split.train_start, end=time_split.train_end)
            if len(training_source) < min_train_rows:
                skipped_train_symbols[symbol] = (
                    f"insufficient rows in train window (rows={len(training_source)}, required>={min_train_rows})"
                )
                continue

        train_target_df: pd.DataFrame = training_source[["step", "log_return"]]
        train_past_cov_df: pd.DataFrame | None = (
            training_source[["step", *built.past_covariate_columns]] if built.past_covariate_columns else None
        )

        train_target_ts = TimeSeries.from_dataframe(train_target_df, time_col="step", value_cols=["log_return"]).astype(
            np.float32
        )
        train_past_cov_ts = (
            TimeSeries.from_dataframe(train_past_cov_df, time_col="step", value_cols=built.past_covariate_columns).astype(
                np.float32
            )
            if train_past_cov_df is not None
            else None
        )
        train_future_cov_ts = future_cov_ts

        if scov is not None:
            train_target_ts = train_target_ts.with_static_covariates(scov)
            if train_past_cov_ts is not None:
                train_past_cov_ts = train_past_cov_ts.with_static_covariates(scov)
            if train_future_cov_ts is not None:
                train_future_cov_ts = train_future_cov_ts.with_static_covariates(scov)

        train_series.append(train_target_ts)
        train_past_cov.append(train_past_cov_ts)
        train_future_cov.append(train_future_cov_ts)
        train_symbols_used.append(symbol)

    if not train_symbols_used:
        details: str = ""
        if skipped_train_symbols:
            examples: list[tuple[str, str]] = sorted(skipped_train_symbols.items())[:10]
            details = "\nExamples:\n" + "\n".join(f"  - {sym}: {reason}" for sym, reason in examples)
        raise TrainingError(
            "No training symbols produced usable feature series. "
            "Consider reducing `model.input_chunk_length`, increasing `split.evaluation.train_months`, "
            "or reducing long-window indicators."
            f"{details}"
        )

    # Darts expects `None` rather than lists containing None.
    train_past_cov = _none_if_any_missing(train_past_cov)
    train_future_cov = _none_if_any_missing(train_future_cov)

    quantiles: list[float] = list(config.model.quantiles)
    model = TFTModel(
        input_chunk_length=int(config.model.input_chunk_length),
        output_chunk_length=int(config.model.horizon_days),
        batch_size=int(config.training.batch_size),
        n_epochs=int(config.training.max_epochs),
        likelihood=QuantileRegression(quantiles=quantiles),
        random_state=int(config.project.random_seed),
        add_relative_index=False,
    )

    model.fit(
        series=train_series,
        past_covariates=train_past_cov,
        future_covariates=train_future_cov,
        verbose=True,
    )

    version: str = config.project.run_id
    artifacts_root = artifacts_root or Path(config.artifacts.root_dir)
    paths: BundlePaths = create_bundle_paths(root_dir=artifacts_root, bundle_name=config.artifacts.bundle_name, version=version)

    paths.model_dir.mkdir(parents=True, exist_ok=True)
    model_path: Path = paths.model_dir / "tft_model.pt"
    model.save(str(model_path))

    splits_path: Path | None = None
    if config.project.mode == RunMode.evaluation and symbol_splits is not None and time_split is not None:
        splits_path = paths.root / "splits.json"
        write_json(
            splits_path,
            {
                "as_of_date": as_of_date.isoformat(),
                "time_split": time_split.to_iso_dict(),
                "symbol_splits": {
                    "train": symbol_splits.train_symbols,
                    "val": symbol_splits.val_symbols,
                    "test": symbol_splits.test_symbols,
                },
                "test_only_symbols": sorted(test_only),
            },
        )

    metrics_payload: dict[str, object] = {}
    if config.project.mode == RunMode.evaluation and time_split is not None and symbol_splits is not None:
        try:
            metrics_payload = _evaluate_model(
                model,
                TimeSeries,
                series_by_symbol,
                time_split=time_split,
                symbol_splits=symbol_splits,
                train_symbols_used=set(train_symbols_used),
                quantiles=list(config.model.quantiles),
                horizon=int(config.model.horizon_days),
            )
        except Exception as exc:  # pragma: no cover
            metrics_payload = {
                "created_at": now_utc_iso(),
                "mode": config.project.mode.value,
                "error": f"{exc.__class__.__name__}: {exc}",
            }

    manifest: dict[str, object] = {
        "created_at": now_utc_iso(),
        "bundle_name": config.artifacts.bundle_name,
        "version": version,
        "project": {
            "name": config.project.name,
            "mode": config.project.mode.value,
            "random_seed": config.project.random_seed,
            "run_id": config.project.run_id,
        },
        "model": {
            "type": config.model.type.value,
            "horizon_days": config.model.horizon_days,
            "input_chunk_length": config.model.input_chunk_length,
            "quantiles": quantiles,
        },
        "data": {
            "price_type": config.data.price_type,
            "min_history_years": config.data.min_history_years,
            "as_of_date": as_of_date.isoformat(),
        },
        "training": {
            "symbols_used": len(train_symbols_used),
            "symbols_skipped": len(skipped_train_symbols),
        },
        **(
            {
                "splits": {
                    "time_split": time_split.to_iso_dict(),
                    "symbol_split_counts": {
                        "train": len(symbol_splits.train_symbols),
                        "val": len(symbol_splits.val_symbols),
                        "test": len(symbol_splits.test_symbols),
                    },
                    "test_only_symbols": len(test_only),
                }
            }
            if config.project.mode == RunMode.evaluation and time_split is not None and symbol_splits is not None
            else {}
        ),
        "features": {
            "target": {"kind": config.features.target.kind.value, "price_field": config.features.target.price_field.value},
            "past_covariates": _common_past_covariate_columns(series_by_symbol, train_symbols_used),
            "future_covariates": _common_future_covariate_columns(series_by_symbol, train_symbols_used),
            "static_covariates": list(next(iter(static_covariates.values())).columns) if static_covariates else [],
        },
        "paths": {
            "model": str(model_path.relative_to(paths.root)),
            **({"splits": str(splits_path.relative_to(paths.root))} if splits_path is not None else {}),
        },
    }
    write_json(paths.manifest_path, manifest)
    write_json(paths.config_snapshot_path, config.model_dump(mode="json"))
    write_json(paths.metrics_path, metrics_payload)

    return paths


def _none_if_any_missing(items: list[object] | None) -> list[object] | None:
    """Return None if list contains any None values (Darts requirement)."""
    if items is None:
        return None
    if any(x is None for x in items):
        return None
    return items


def _encode_static_covariates(universe: pd.DataFrame, *, train_symbols: set[str]) -> dict[str, pd.DataFrame]:
    """One-hot encode static covariates (fit on training symbols only)."""
    if universe.empty:
        return {}

    static_cols: list[str] = ["exchange_mic", "sector", "industry", "country_code", "currency", "market_cap_bucket"]
    meta: pd.DataFrame = universe[["symbol", *static_cols]].copy()
    meta["symbol"] = meta["symbol"].astype(str)
    for col in static_cols:
        meta[col] = meta[col].fillna("unknown").astype(str)

    train_meta: pd.DataFrame = meta[meta["symbol"].isin(train_symbols)].copy()
    if train_meta.empty:
        return {}

    dummies: pd.DataFrame = pd.get_dummies(train_meta[static_cols], prefix=static_cols, dtype="int8")
    dummy_cols: list[str] = list(dummies.columns)

    out: dict[str, pd.DataFrame] = {}
    for row in meta.itertuples(index=False):
        symbol: str = str(getattr(row, "symbol"))
        row_df: pd.DataFrame = pd.DataFrame([{col: getattr(row, col) for col in static_cols}])
        row_dum: pd.DataFrame = pd.get_dummies(row_df[static_cols], prefix=static_cols, dtype="int8")
        row_dum = row_dum.reindex(columns=dummy_cols, fill_value=0)
        out[symbol] = row_dum
    return out


def _common_past_covariate_columns(series_by_symbol: dict[str, PreparedSeries], symbols: list[str]) -> list[str]:
    """Get intersection of past covariate columns across symbols."""
    columns: set[str] | None = None
    for symbol in symbols:
        item: PreparedSeries = series_by_symbol[symbol]
        past: object | None = item.past_covariates
        if past is None:
            continue
        current: set[str] = set(getattr(past, "components", []))  # type: ignore[attr-defined]
        columns = current if columns is None else columns & current
    return sorted(columns or [])


def _common_future_covariate_columns(series_by_symbol: dict[str, PreparedSeries], symbols: list[str]) -> list[str]:
    """Get intersection of future covariate columns across symbols."""
    columns: set[str] | None = None
    for symbol in symbols:
        item: PreparedSeries = series_by_symbol[symbol]
        fut: object | None = item.future_covariates
        if fut is None:
            continue
        current: set[str] = set(getattr(fut, "components", []))  # type: ignore[attr-defined]
        columns = current if columns is None else columns & current
    return sorted(columns or [])


def _evaluate_model(
    model: object,
    time_series_cls: object,
    series_by_symbol: dict[str, PreparedSeries],
    *,
    time_split: object,
    symbol_splits: object,
    train_symbols_used: set[str],
    quantiles: list[float],
    horizon: int,
) -> dict[str, object]:
    """Evaluate model on temporal validation, symbol validation, and holdout test segments."""
    quantiles_sorted: list[float] = sorted(float(q) for q in quantiles)
    stride: int = int(max(1, horizon))
    num_samples: int = int(max(200, len(quantiles_sorted) * 100))

    temporal_symbols: list[str] = sorted(train_symbols_used)

    segments: dict[str, dict[str, object]] = {
        "temporal_validation": _evaluate_segment(
            model,
            time_series_cls,
            series_by_symbol,
            name="temporal_validation",
            symbols=temporal_symbols,
            cutoff=getattr(time_split, "train_end"),
            window_start=getattr(time_split, "val_start"),
            window_end=getattr(time_split, "val_end"),
            quantiles=quantiles_sorted,
            horizon=horizon,
            stride=stride,
            num_samples=num_samples,
        ),
        "symbol_validation": _evaluate_segment(
            model,
            time_series_cls,
            series_by_symbol,
            name="symbol_validation",
            symbols=list(getattr(symbol_splits, "val_symbols")),
            cutoff=getattr(time_split, "train_end"),
            window_start=getattr(time_split, "val_start"),
            window_end=getattr(time_split, "val_end"),
            quantiles=quantiles_sorted,
            horizon=horizon,
            stride=stride,
            num_samples=num_samples,
        ),
        "holdout_test": _evaluate_segment(
            model,
            time_series_cls,
            series_by_symbol,
            name="holdout_test",
            symbols=list(getattr(symbol_splits, "test_symbols")),
            cutoff=getattr(time_split, "val_end"),
            window_start=getattr(time_split, "test_start"),
            window_end=getattr(time_split, "test_end"),
            quantiles=quantiles_sorted,
            horizon=horizon,
            stride=stride,
            num_samples=num_samples,
        ),
    }

    return {
        "created_at": now_utc_iso(),
        "mode": RunMode.evaluation.value,
        "quantiles": quantiles_sorted,
        "horizon": horizon,
        "stride": stride,
        "num_samples": num_samples,
        "segments": segments,
        "notes": {
            "crps": "crps_pinball is an approximation based on averaging pinball loss across quantiles",
            "evaluation": "rolling-origin evaluation with stride=horizon (not walk-forward retraining)",
        },
    }


def _evaluate_segment(
    model: object,
    time_series_cls: object,
    series_by_symbol: dict[str, PreparedSeries],
    *,
    name: str,
    symbols: list[str],
    cutoff: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    quantiles: list[float],
    horizon: int,
    stride: int,
    num_samples: int,
) -> dict[str, object]:
    """Evaluate model on a segment using rolling-origin forecasts."""
    requested: int = len(symbols)
    if requested == 0:
        return {"symbols_requested": 0, "points": 0, "metrics": {}, "by_horizon": {}}

    rows: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}
    forecasts_made: int = 0

    for symbol in symbols:
        item: PreparedSeries | None = series_by_symbol.get(symbol)
        if item is None:
            skipped[symbol] = "missing_series"
            continue

        try:
            target_df: pd.DataFrame = _timeseries_to_frame(item.target, index_name="step")
        except Exception as exc:
            skipped[symbol] = f"target_extract_failed:{exc.__class__.__name__}"
            continue

        target_cols: list[str] = [c for c in target_df.columns if c != "step"]
        if len(target_cols) != 1:
            skipped[symbol] = "target_not_univariate"
            continue
        target_col: str = target_cols[0]

        actual: pd.DataFrame = target_df.rename(columns={target_col: "y"})[["step", "y"]]
        actual = actual.merge(item.timeline[["step", "time", "is_future"]], on="step", how="left")
        actual = actual[(actual["is_future"] == 0) & actual["time"].notna()].copy()
        if actual.empty:
            skipped[symbol] = "no_observed_rows"
            continue

        cutoff_ts = pd.Timestamp(cutoff)
        if cutoff_ts.tzinfo is None:
            cutoff_ts = cutoff_ts.tz_localize("UTC")
        else:
            cutoff_ts = cutoff_ts.tz_convert("UTC")
        cutoff_ts = cutoff_ts.normalize()

        eligible: pd.DataFrame = actual[actual["time"] <= cutoff_ts]
        if eligible.empty:
            skipped[symbol] = "no_history_before_cutoff"
            continue

        base_step: int = int(pd.to_numeric(eligible["step"], errors="coerce").max())

        time_by_step = dict(zip(actual["step"].astype(int), pd.to_datetime(actual["time"], utc=True)))
        max_step: int = int(pd.to_numeric(actual["step"], errors="coerce").max())

        origin_steps: list[int] = list(range(base_step, max_step, int(stride)))
        origin_steps = [s for s in origin_steps if time_by_step.get(s) is not None and time_by_step[s] < window_end]
        if not origin_steps:
            skipped[symbol] = "no_origins_in_window"
            continue

        future_covariates: object | None = item.future_covariates

        for origin_step in origin_steps:
            series: object = item.target.drop_after(origin_step)  # type: ignore[attr-defined]
            if len(series) < 2:
                continue

            past_cov_series: object | None = None
            if item.past_covariates is not None:
                past_cov_series = item.past_covariates.drop_after(origin_step)  # type: ignore[attr-defined]

            try:
                forecast: object = model.predict(  # type: ignore[attr-defined]
                    n=int(horizon),
                    series=series,
                    past_covariates=past_cov_series,
                    future_covariates=future_covariates,
                    num_samples=int(num_samples),
                )
            except TypeError:
                forecast = model.predict(  # type: ignore[attr-defined]
                    n=int(horizon),
                    series=series,
                    past_covariates=past_cov_series,
                    future_covariates=future_covariates,
                )
            except Exception as exc:
                skipped[symbol] = f"predict_failed:{exc.__class__.__name__}"
                break

            try:
                pred: pd.DataFrame = _forecast_to_quantile_frame(forecast, quantiles, index_name="step")
            except Exception as exc:
                skipped[symbol] = f"forecast_extract_failed:{exc.__class__.__name__}"
                break

            origin_time = pd.Timestamp(time_by_step[origin_step]).isoformat()
            pred["origin_step"] = int(origin_step)
            pred["origin_time"] = origin_time

            merged = pred.merge(item.timeline[["step", "time", "is_future"]], on="step", how="left")
            merged = merged.merge(actual[["step", "y"]], on="step", how="left")
            merged = merged[(merged["time"] >= window_start) & (merged["time"] <= window_end)].copy()

            required_cols: list[str] = ["y", *[f"q{q:g}" for q in quantiles]]
            merged = merged.dropna(subset=required_cols)
            if merged.empty:
                continue

            merged["symbol"] = symbol
            rows.append(merged)
            forecasts_made += 1

    if not rows:
        return {
            "name": name,
            "symbols_requested": requested,
            "symbols_evaluated": 0,
            "symbols_skipped": len(skipped),
            "forecasts": forecasts_made,
            "points": 0,
            "metrics": {},
            "by_horizon": {},
        }

    results: pd.DataFrame = pd.concat(rows, ignore_index=True)
    y_true: np.ndarray = results["y"].to_numpy(dtype="float64")
    forecast_arrays: dict[float, np.ndarray] = {q: results[f"q{q:g}"].to_numpy(dtype="float64") for q in quantiles}

    overall: dict[str, float] = evaluate_quantile_forecasts(y_true, forecast_arrays)

    by_horizon: dict[str, dict[str, float]] = {}
    for step, group in results.groupby("horizon_step", sort=True):
        y_step: np.ndarray = group["y"].to_numpy(dtype="float64")
        f_step: dict[float, np.ndarray] = {q: group[f"q{q:g}"].to_numpy(dtype="float64") for q in quantiles}
        by_horizon[str(int(step))] = evaluate_quantile_forecasts(y_step, f_step)

    return {
        "name": name,
        "cutoff": pd.Timestamp(cutoff).isoformat(),
        "window_start": pd.Timestamp(window_start).isoformat(),
        "window_end": pd.Timestamp(window_end).isoformat(),
        "symbols_requested": requested,
        "symbols_evaluated": int(results["symbol"].nunique()),
        "symbols_skipped": len(skipped),
        "forecasts": forecasts_made,
        "points": int(len(results)),
        "metrics": overall,
        "by_horizon": by_horizon,
    }


def _timeseries_to_frame(ts: object, *, index_name: str = "time") -> pd.DataFrame:
    """Convert Darts TimeSeries to a DataFrame with an explicit index column."""
    df: pd.DataFrame = ts.to_dataframe()  # type: ignore[attr-defined]
    out: pd.DataFrame = df.reset_index()
    out = out.rename(columns={out.columns[0]: index_name})
    return out


def _forecast_to_quantile_frame(forecast: object, quantiles: list[float], *, index_name: str) -> pd.DataFrame:
    """Extract quantile forecasts from a stochastic Darts forecast TimeSeries."""

    out: pd.DataFrame | None = None
    for q in quantiles:
        try:
            q_ts: object = forecast.quantile(q)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            raise TrainingError(
                "Forecast object does not support quantile extraction; "
                "ensure `model.predict(..., num_samples=...)` produces a stochastic forecast."
            ) from exc

        q_df: pd.DataFrame = _timeseries_to_frame(q_ts, index_name=index_name)
        value_cols: list[str] = [c for c in q_df.columns if c != index_name]
        if len(value_cols) != 1:
            raise TrainingError("Expected a single component when extracting quantile forecasts")
        col: str = f"q{q:g}"
        q_df = q_df.rename(columns={value_cols[0]: col})
        out = q_df if out is None else out.merge(q_df, on=index_name, how="inner")

    if out is None or out.empty:
        raise TrainingError("No quantile forecasts extracted")

    out = out.sort_values(index_name).reset_index(drop=True)
    out["horizon_step"] = np.arange(1, len(out) + 1, dtype="int32")
    return out
