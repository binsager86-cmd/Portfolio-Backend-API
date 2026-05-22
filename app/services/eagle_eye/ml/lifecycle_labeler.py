from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from app.services.eagle_eye.adapter import TickerChartAdapter
from app.services.eagle_eye.indicators import compute_all_indicators
from app.services.eagle_eye.ml.leakage_audit import scan_source_for_leakage
from app.services.eagle_eye.ml.market_context import MarketContextBuilder
from app.services.eagle_eye.store import list_tickers_with_ohlcv, load_ohlcv


LIFECYCLE_STATES: tuple[str, ...] = (
    "DORMANT",
    "CONFIRMED",
    "MID",
    "MATURE",
    "EXHAUSTED",
)


INDICATOR_INVENTORY: dict[str, tuple[str, ...]] = {
    "trend": (
        "ema_8",
        "ema_21",
        "ema_50",
        "ema_100",
        "ema_200",
        "ema_ribbon_aligned",
        "macd_line",
        "macd_signal",
        "macd_histogram",
        "adx",
        "plus_di",
        "minus_di",
        "supertrend",
        "psar",
        "hull_ma",
        "linreg_slope",
        "ichimoku_cloud_pos",
        "ichimoku_tk_cross",
    ),
    "momentum": (
        "rsi",
        "rsi_divergence",
        "stoch_k",
        "stoch_d",
        "stoch_rsi",
        "williams_r",
        "cci",
        "roc",
        "tsi",
        "ao",
        "connors_rsi",
    ),
    "volatility": (
        "atr",
        "atr_percentile_252",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_pct_b",
        "bb_bandwidth",
        "bb_squeeze",
        "kc_upper",
        "kc_lower",
        "dc_upper",
        "dc_lower",
        "hist_vol_30d",
    ),
    "volume_flow": (
        "obv",
        "obv_slope_20",
        "obv_slope_60",
        "ad_line",
        "cmf",
        "mfi",
        "vwap",
        "vwap_distance_sigma",
        "rel_volume",
        "force_index",
        "eom",
        "klinger",
        "dollar_volume",
    ),
    "structure": (
        "zscore_20",
        "zscore_50",
        "zscore_200",
    ),
    "institutional": (
        "accumulation_score",
        "wyckoff_phase",
    ),
    "context": (
        "close",
        "high",
        "low",
        "volume",
        "index_return_1d",
        "index_return_3d",
        "index_return_5d",
        "index_return_10d",
        "index_return_20d",
        "index_return_60d",
        "index_vol_20d",
        "market_regime",
        "stock_beta_60d",
    ),
}


KNOWN_UNSAFE_INDICATOR_COLUMNS: tuple[str, ...] = (
    "swing_high",
    "swing_low",
    "chikou",
    "accumulation_score",
)


DEFAULT_RUN_LENGTH_DAYS = 35.0
DEFAULT_RUN_MAGNITUDE_PCT = 18.0
DEFAULT_RUN_DRAWDOWN_PCT = 8.0
MIN_BARS_FOR_LABELS = 260


@dataclass(frozen=True)
class RunSummary:
    start_date: pd.Timestamp
    peak_date: pd.Timestamp
    end_date: pd.Timestamp
    base_price: float
    peak_price: float
    peak_gain_pct: float
    days_to_peak: int
    max_drawdown_pct: float


@dataclass(frozen=True)
class LifecycleConfig:
    entry_mode: str = "confirmed"
    base_lookback_days: int = 20
    min_history_days: int = MIN_BARS_FOR_LABELS
    breakout_buffer: float = 1.005
    base_range_cap_pct: float = 0.22
    dormant_adx_max: float = 18.0
    weak_trend_adx_max: float = 20.0
    trend_adx_min: float = 18.0
    breakout_volume_min: float = 0.85
    confirmed_min_days: int = 2
    confirmed_min_gain_pct: float = 2.5
    confirmed_days_frac: float = 0.60
    confirmed_gain_frac: float = 0.75
    confirmed_extension_cap: float = 0.75
    confirmed_trend_score_min: float = 0.55
    confirmed_momentum_score_min: float = 0.52
    confirmed_volume_score_min: float = 0.45
    confirmed_adx_min: float = 18.0
    confirmed_rsi_min: float = 52.0
    confirmed_rsi_max: float = 76.0
    confirmed_rel_volume_min: float = 1.10
    confirmed_rel_volume_strong: float = 1.20
    confirmed_cmf_min: float = 0.10
    confirmed_cmf_strong: float = 0.25
    confirmed_volume_score_strong: float = 0.68
    confirmed_confirmation_score_min: float = 0.70
    early_days_frac: float = 0.40
    early_gain_frac: float = 0.60
    early_extension_cap: float = 0.70
    early_trend_score_min: float = 0.35
    early_momentum_score_min: float = 0.30
    mature_days_frac: float = 0.90
    mature_gain_frac: float = 0.85
    mature_extension_floor: float = 0.75
    exhausted_extension_frac: float = 1.30
    exhausted_score_min: float = 0.65
    exhausted_drawdown_floor_pct: float = 6.0
    stop_drawdown_frac: float = 1.15
    forward_confirm_days: int = 2
    backward_confirm_days: int = 3
    exhausted_relief_confirm_days: int = 4


def get_indicator_inventory() -> dict[str, tuple[str, ...]]:
    return dict(INDICATOR_INVENTORY)


def get_known_unsafe_indicator_columns() -> tuple[str, ...]:
    return KNOWN_UNSAFE_INDICATOR_COLUMNS


def load_lifecycle_inputs(
    ticker: str,
    *,
    start: Optional[date] = None,
    end: Optional[date] = None,
    include_market_context: bool = True,
) -> pd.DataFrame:
    price_frame = load_ohlcv(ticker, start=start, end=end)
    if price_frame is None or price_frame.empty:
        adapter = TickerChartAdapter()
        if end is None:
            end = date.today()
        if start is None:
            start = date(end.year - 6, end.month, min(end.day, 28))
        price_frame = adapter.get_ohlcv_daily(ticker, start, end)

    if price_frame is None or price_frame.empty:
        return _empty_lifecycle_frame()

    if len(price_frame) < 50:
        return _bootstrap_short_history_frame(ticker.upper(), price_frame)

    indicator_frame = compute_all_indicators(price_frame)
    indicator_frame = indicator_frame.copy()
    indicator_frame.index = pd.to_datetime(indicator_frame.index)

    if include_market_context:
        indicator_frame = _attach_market_context(indicator_frame)

    return indicator_frame


def label_lifecycle_states(
    ticker: str,
    *,
    ohlcv: Optional[pd.DataFrame] = None,
    indicator_frame: Optional[pd.DataFrame] = None,
    include_market_context: bool = True,
    config: Optional[LifecycleConfig] = None,
) -> pd.DataFrame:
    lifecycle_config = config or LifecycleConfig()

    if indicator_frame is None:
        if ohlcv is None:
            frame = load_lifecycle_inputs(
                ticker,
                include_market_context=include_market_context,
            )
        else:
            if ohlcv.empty or len(ohlcv) < 50:
                return _bootstrap_short_history_frame(ticker.upper(), ohlcv)
            frame = compute_all_indicators(ohlcv)
            frame.index = pd.to_datetime(frame.index)
            if include_market_context:
                frame = _attach_market_context(frame)
    else:
        frame = indicator_frame.copy()
        frame.index = pd.to_datetime(frame.index)

    if frame.empty:
        return _empty_lifecycle_frame()

    return _label_lifecycle_frame(ticker.upper(), frame, lifecycle_config)


def audit_labeler_source() -> list[str]:
    return scan_source_for_leakage(inspect.getsource(inspect.getmodule(audit_labeler_source)))


def audit_truncation_invariance(
    ticker: str,
    *,
    sample_size: int = 25,
    seed: int = 42,
    config: Optional[LifecycleConfig] = None,
) -> dict[str, Any]:
    raw = load_ohlcv(ticker)
    if raw is None or raw.empty:
        return {
            "ticker": ticker.upper(),
            "pass": False,
            "reason": "NO_OHLCV",
            "checked": 0,
            "mismatches": [],
        }

    full = label_lifecycle_states(ticker, ohlcv=raw, include_market_context=False, config=config)
    if full.empty:
        return {
            "ticker": ticker.upper(),
            "pass": False,
            "reason": "NO_LABELS",
            "checked": 0,
            "mismatches": [],
        }

    valid_dates = list(full.index[full["state"].notna()])
    if not valid_dates:
        return {
            "ticker": ticker.upper(),
            "pass": False,
            "reason": "NO_VALID_DATES",
            "checked": 0,
            "mismatches": [],
        }

    generator = np.random.default_rng(seed)
    sample_n = min(sample_size, len(valid_dates))
    sample_positions = sorted(generator.choice(len(valid_dates), size=sample_n, replace=False).tolist())
    mismatches: list[dict[str, Any]] = []

    for sample_position in sample_positions:
        sample_date = valid_dates[sample_position]
        truncated_raw = raw.loc[:sample_date].copy()
        truncated = label_lifecycle_states(
            ticker,
            ohlcv=truncated_raw,
            include_market_context=False,
            config=config,
        )
        if truncated.empty or sample_date not in truncated.index:
            mismatches.append(
                {
                    "date": sample_date.date().isoformat(),
                    "full_state": full.loc[sample_date, "state"],
                    "truncated_state": None,
                }
            )
            continue

        full_state = str(full.loc[sample_date, "state"])
        truncated_state = str(truncated.loc[sample_date, "state"])
        if full_state != truncated_state:
            mismatches.append(
                {
                    "date": sample_date.date().isoformat(),
                    "full_state": full_state,
                    "truncated_state": truncated_state,
                }
            )

    return {
        "ticker": ticker.upper(),
        "pass": len(mismatches) == 0,
        "checked": sample_n,
        "mismatches": mismatches,
    }


def count_completed_runs(ticker: str, *, config: Optional[LifecycleConfig] = None) -> int:
    labeled = label_lifecycle_states(ticker, include_market_context=False, config=config)
    if labeled.empty:
        return 0
    return int(labeled["is_move_end"].fillna(False).sum())


def completed_run_distribution(
    *,
    tickers: Optional[Sequence[str]] = None,
    config: Optional[LifecycleConfig] = None,
) -> dict[str, Any]:
    resolved_tickers = list(tickers or list_tickers_with_ohlcv())
    rows: list[dict[str, Any]] = []
    for ticker in resolved_tickers:
        raw = load_ohlcv(ticker)
        if raw is None or raw.empty or len(raw) < MIN_BARS_FOR_LABELS:
            continue
        runs = count_completed_runs(ticker, config=config)
        rows.append({"ticker": ticker.upper(), "completed_runs": runs})

    run_frame = pd.DataFrame(rows)
    if run_frame.empty:
        return {
            "frame": run_frame,
            "distribution": {">=20": 0, "10-19": 0, "<10": 0},
        }

    distribution = {
        ">=20": int((run_frame["completed_runs"] >= 20).sum()),
        "10-19": int(((run_frame["completed_runs"] >= 10) & (run_frame["completed_runs"] < 20)).sum()),
        "<10": int((run_frame["completed_runs"] < 10).sum()),
    }
    return {"frame": run_frame.sort_values("completed_runs", ascending=False), "distribution": distribution}


def _label_lifecycle_frame(
    ticker: str,
    frame: pd.DataFrame,
    config: LifecycleConfig,
) -> pd.DataFrame:
    labeled = frame.copy()
    labeled = labeled.sort_index()

    if len(labeled) < 50:
        return _empty_lifecycle_frame()

    labeled["prior_base_high"] = labeled["high"].rolling(config.base_lookback_days).max().shift(1)
    labeled["prior_base_low"] = labeled["low"].rolling(config.base_lookback_days).min().shift(1)
    labeled["base_range_pct"] = (
        (labeled["prior_base_high"] - labeled["prior_base_low"])
        / labeled["close"].shift(1).replace(0, np.nan)
    )
    labeled["ema_21_slope_5"] = labeled["ema_21"] - labeled["ema_21"].shift(5)
    labeled["adx_slope_5"] = labeled["adx"] - labeled["adx"].shift(5)
    labeled["macd_slope_3"] = labeled["macd_histogram"] - labeled["macd_histogram"].shift(3)
    labeled["rsi_slope_3"] = labeled["rsi"] - labeled["rsi"].shift(3)
    labeled["bb_bandwidth_delta_5"] = labeled["bb_bandwidth"] - labeled["bb_bandwidth"].shift(5)
    labeled["rolling_peak_close_20"] = labeled["close"].rolling(20).max()

    output_rows: list[dict[str, Any]] = []
    completed_runs: list[RunSummary] = []
    active_run: Optional[dict[str, Any]] = None
    run_counter = 0
    confirmed_state = "DORMANT"
    pending_state: Optional[str] = None
    pending_streak = 0

    for bar_date, row in labeled.iterrows():
        prior_stats = _summarize_runs(completed_runs)
        move_start = False
        move_end = False

        if active_run is None and _is_move_start(row, config):
            run_counter += 1
            move_start = True
            base_price = _coalesce_float(row.get("prior_base_low"), row.get("close"))
            active_run = {
                "run_id": f"{ticker}_{run_counter}",
                "start_date": bar_date,
                "base_price": base_price,
                "peak_price": _coalesce_float(row.get("close"), base_price),
                "peak_date": bar_date,
                "max_drawdown_pct": 0.0,
            }

        if active_run is not None:
            current_close = _coalesce_float(row.get("close"), active_run["base_price"])
            if current_close >= active_run["peak_price"]:
                active_run["peak_price"] = current_close
                active_run["peak_date"] = bar_date

            drawdown_pct = _pct_drop(active_run["peak_price"], current_close)
            active_run["max_drawdown_pct"] = max(active_run["max_drawdown_pct"], drawdown_pct)

            metrics = _build_active_run_metrics(row, active_run, prior_stats)
            raw_state = _classify_active_state(row, metrics, config)
            state, pending_state, pending_streak = _apply_state_hysteresis(
                current_state=confirmed_state,
                raw_state=raw_state,
                pending_state=pending_state,
                pending_streak=pending_streak,
                active_run=True,
                config=config,
            )
            confirmed_state = state

            if _is_move_end(row, metrics, prior_stats, config):
                move_end = True
                completed_runs.append(
                    RunSummary(
                        start_date=active_run["start_date"],
                        peak_date=active_run["peak_date"],
                        end_date=bar_date,
                        base_price=float(active_run["base_price"]),
                        peak_price=float(active_run["peak_price"]),
                        peak_gain_pct=float(metrics["gain_so_far_pct"] if current_close >= active_run["peak_price"] else _pct_gain(active_run["base_price"], active_run["peak_price"])),
                        days_to_peak=max(1, int((active_run["peak_date"] - active_run["start_date"]).days)),
                        max_drawdown_pct=float(active_run["max_drawdown_pct"]),
                    )
                )
                active_run = None
        else:
            metrics = _empty_metrics(prior_stats)
            raw_state = _classify_dormant_state(row, config)
            state = "DORMANT"
            confirmed_state = "DORMANT"
            pending_state = None
            pending_streak = 0

        output_rows.append(
            {
                "event_date": bar_date,
                "ticker": ticker,
                "state": state,
                "raw_state": raw_state,
                "is_move_start": move_start,
                "is_move_end": move_end,
                **metrics,
                "completed_runs_before_day": len(completed_runs),
            }
        )

    output = pd.DataFrame(output_rows).set_index("event_date")
    return output


def _attach_market_context(frame: pd.DataFrame) -> pd.DataFrame:
    context_builder = MarketContextBuilder()
    enriched = frame.reset_index()
    date_column = str(enriched.columns[0])
    if date_column != "event_date":
        enriched = enriched.rename(columns={date_column: "event_date"})
    enriched = context_builder.enrich(enriched, date_col="event_date")
    enriched["stock_beta_60d"] = context_builder.compute_rolling_beta(
        frame["close"],
        pd.DatetimeIndex(frame.index),
    ).values
    return enriched.set_index("event_date")


def _empty_lifecycle_frame() -> pd.DataFrame:
    columns = [
        "ticker",
        "state",
        "raw_state",
        "is_move_start",
        "is_move_end",
        "active_run",
        "run_id",
        "run_start_date",
        "base_price",
        "peak_price_so_far",
        "days_elapsed",
        "gain_so_far_pct",
        "drawdown_from_peak_pct",
        "typical_run_length_days",
        "typical_run_magnitude_pct",
        "typical_run_drawdown_pct",
        "duration_progress",
        "magnitude_progress",
        "extension_vs_norm",
        "trend_score",
        "momentum_score",
        "volume_score",
        "confirmation_score",
        "exhaustion_score",
        "market_regime",
        "index_return_20d",
        "index_vol_20d",
        "stock_beta_60d",
        "completed_runs_before_day",
    ]
    return pd.DataFrame(columns=columns)


def _bootstrap_short_history_frame(ticker: str, ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv is None or ohlcv.empty:
        return _empty_lifecycle_frame()

    index = pd.to_datetime(ohlcv.index)
    base = pd.DataFrame(index=index)
    base["ticker"] = ticker
    base["state"] = "DORMANT"
    base["raw_state"] = "DORMANT"
    base["is_move_start"] = False
    base["is_move_end"] = False
    base["active_run"] = False
    base["run_id"] = None
    base["run_start_date"] = None
    base["base_price"] = np.nan
    base["peak_price_so_far"] = np.nan
    base["days_elapsed"] = 0
    base["gain_so_far_pct"] = 0.0
    base["drawdown_from_peak_pct"] = 0.0
    base["typical_run_length_days"] = DEFAULT_RUN_LENGTH_DAYS
    base["typical_run_magnitude_pct"] = DEFAULT_RUN_MAGNITUDE_PCT
    base["typical_run_drawdown_pct"] = DEFAULT_RUN_DRAWDOWN_PCT
    base["duration_progress"] = 0.0
    base["magnitude_progress"] = 0.0
    base["extension_vs_norm"] = 0.0
    base["trend_score"] = 0.0
    base["momentum_score"] = 0.0
    base["volume_score"] = 0.0
    base["confirmation_score"] = 0.0
    base["exhaustion_score"] = 0.0
    base["market_regime"] = None
    base["index_return_20d"] = np.nan
    base["index_vol_20d"] = np.nan
    base["stock_beta_60d"] = np.nan
    base["completed_runs_before_day"] = 0

    if "volume" in ohlcv.columns:
        base["volume"] = pd.to_numeric(ohlcv["volume"], errors="coerce")

    return base


def _summarize_runs(completed_runs: Sequence[RunSummary]) -> dict[str, float]:
    if not completed_runs:
        return {
            "typical_run_length_days": DEFAULT_RUN_LENGTH_DAYS,
            "typical_run_magnitude_pct": DEFAULT_RUN_MAGNITUDE_PCT,
            "typical_run_drawdown_pct": DEFAULT_RUN_DRAWDOWN_PCT,
            "completed_run_count": 0.0,
        }

    lengths = np.array([run.days_to_peak for run in completed_runs], dtype=float)
    magnitudes = np.array([run.peak_gain_pct for run in completed_runs], dtype=float)
    drawdowns = np.array([run.max_drawdown_pct for run in completed_runs], dtype=float)
    return {
        "typical_run_length_days": float(np.median(lengths)),
        "typical_run_magnitude_pct": float(np.median(magnitudes)),
        "typical_run_drawdown_pct": float(np.median(drawdowns)),
        "completed_run_count": float(len(completed_runs)),
    }


def _is_move_start(row: pd.Series, config: LifecycleConfig) -> bool:
    close_value = _coalesce_float(row.get("close"), 0.0)
    prior_base_high = _coalesce_float(row.get("prior_base_high"), math.nan)
    if math.isnan(prior_base_high) or prior_base_high <= 0:
        return False

    breakout = close_value >= prior_base_high * config.breakout_buffer
    base_is_tight = _coalesce_float(row.get("base_range_pct"), math.inf) <= config.base_range_cap_pct
    trend_up = (
        close_value >= _coalesce_float(row.get("ema_21"), close_value)
        and _coalesce_float(row.get("ema_21"), close_value) >= _coalesce_float(row.get("ema_50"), close_value) * 0.995
        and _coalesce_float(row.get("ema_21_slope_5"), 0.0) >= 0
    )
    momentum_rising = (
        _coalesce_float(row.get("rsi"), 0.0) >= 52.0
        and _coalesce_float(row.get("macd_histogram"), 0.0) >= -0.05
        and (
            _coalesce_float(row.get("rsi_slope_3"), 0.0) >= 0.0
            or _coalesce_float(row.get("macd_slope_3"), 0.0) >= 0.0
        )
    )
    volume_support = (
        _coalesce_float(row.get("rel_volume"), 0.0) >= config.breakout_volume_min
        or _coalesce_float(row.get("cmf"), 0.0) >= -0.02
    )
    return bool(breakout and base_is_tight and trend_up and momentum_rising and volume_support)


def _build_active_run_metrics(
    row: pd.Series,
    active_run: dict[str, Any],
    prior_stats: dict[str, float],
) -> dict[str, Any]:
    current_close = _coalesce_float(row.get("close"), active_run["base_price"])
    base_price = float(active_run["base_price"])
    peak_price = float(active_run["peak_price"])
    days_elapsed = max(1, int((row.name - active_run["start_date"]).days) + 1)
    gain_so_far_pct = _pct_gain(base_price, current_close)
    drawdown_from_peak_pct = _pct_drop(peak_price, current_close)

    typical_length = max(prior_stats["typical_run_length_days"], 1.0)
    typical_magnitude = max(prior_stats["typical_run_magnitude_pct"], 1.0)
    typical_drawdown = max(prior_stats["typical_run_drawdown_pct"], 1.0)

    duration_progress = days_elapsed / typical_length
    magnitude_progress = gain_so_far_pct / typical_magnitude
    extension_vs_norm = max(duration_progress, magnitude_progress)

    trend_score = _average([
        _scale_between(_coalesce_float(row.get("adx"), 0.0), 15.0, 35.0),
        1.0 if _coalesce_float(row.get("plus_di"), 0.0) > _coalesce_float(row.get("minus_di"), 0.0) else 0.0,
        1.0 if _coalesce_float(row.get("close"), 0.0) > _coalesce_float(row.get("ema_21"), 0.0) else 0.0,
        1.0 if _coalesce_float(row.get("ema_21"), 0.0) > _coalesce_float(row.get("ema_50"), 0.0) else 0.0,
    ])
    momentum_score = _average([
        _scale_between(_coalesce_float(row.get("rsi"), 0.0), 50.0, 75.0),
        _scale_between(_coalesce_float(row.get("macd_histogram"), 0.0), 0.0, max(abs(_coalesce_float(row.get("macd_histogram"), 0.0)) * 2.0, 0.5)),
        _scale_between(_coalesce_float(row.get("rel_volume"), 0.0), 0.9, 1.8),
        _scale_between(_coalesce_float(row.get("cmf"), 0.0), -0.05, 0.2),
    ])
    volume_score = _average([
        _scale_between(_coalesce_float(row.get("rel_volume"), 0.0), 0.95, 1.6),
        _scale_between(_coalesce_float(row.get("cmf"), 0.0), -0.02, 0.15),
        1.0 if _coalesce_float(row.get("close"), 0.0) >= _coalesce_float(row.get("vwap"), math.inf) else 0.0,
    ])
    confirmation_score = _average([
        trend_score,
        momentum_score,
        volume_score,
    ])
    exhaustion_score = _average([
        _scale_between(extension_vs_norm, 0.9, 1.4),
        _scale_between(drawdown_from_peak_pct, typical_drawdown * 0.7, typical_drawdown * 1.4),
        1.0 if _coalesce_float(row.get("macd_slope_3"), 0.0) < 0.0 else 0.0,
        1.0 if _coalesce_float(row.get("rsi_slope_3"), 0.0) < 0.0 else 0.0,
        1.0 if _coalesce_float(row.get("cmf"), 0.0) < 0.0 else 0.0,
    ])

    return {
        "active_run": True,
        "run_id": active_run["run_id"],
        "run_start_date": active_run["start_date"].date().isoformat(),
        "base_price": base_price,
        "peak_price_so_far": peak_price,
        "days_elapsed": days_elapsed,
        "gain_so_far_pct": round(gain_so_far_pct, 4),
        "drawdown_from_peak_pct": round(drawdown_from_peak_pct, 4),
        "typical_run_length_days": round(typical_length, 4),
        "typical_run_magnitude_pct": round(typical_magnitude, 4),
        "typical_run_drawdown_pct": round(typical_drawdown, 4),
        "duration_progress": round(duration_progress, 4),
        "magnitude_progress": round(magnitude_progress, 4),
        "extension_vs_norm": round(extension_vs_norm, 4),
        "trend_score": round(trend_score, 4),
        "momentum_score": round(momentum_score, 4),
        "volume_score": round(volume_score, 4),
        "confirmation_score": round(confirmation_score, 4),
        "exhaustion_score": round(exhaustion_score, 4),
        "market_regime": row.get("market_regime"),
        "index_return_20d": _coalesce_float(row.get("index_return_20d"), math.nan),
        "index_vol_20d": _coalesce_float(row.get("index_vol_20d"), math.nan),
        "stock_beta_60d": _coalesce_float(row.get("stock_beta_60d"), math.nan),
    }


def _empty_metrics(prior_stats: dict[str, float]) -> dict[str, Any]:
    return {
        "active_run": False,
        "run_id": None,
        "run_start_date": None,
        "base_price": math.nan,
        "peak_price_so_far": math.nan,
        "days_elapsed": 0,
        "gain_so_far_pct": 0.0,
        "drawdown_from_peak_pct": 0.0,
        "typical_run_length_days": round(prior_stats["typical_run_length_days"], 4),
        "typical_run_magnitude_pct": round(prior_stats["typical_run_magnitude_pct"], 4),
        "typical_run_drawdown_pct": round(prior_stats["typical_run_drawdown_pct"], 4),
        "duration_progress": 0.0,
        "magnitude_progress": 0.0,
        "extension_vs_norm": 0.0,
        "trend_score": 0.0,
        "momentum_score": 0.0,
        "volume_score": 0.0,
        "confirmation_score": 0.0,
        "exhaustion_score": 0.0,
        "market_regime": None,
        "index_return_20d": math.nan,
        "index_vol_20d": math.nan,
        "stock_beta_60d": math.nan,
    }


def _classify_active_state(
    row: pd.Series,
    metrics: dict[str, Any],
    config: LifecycleConfig,
) -> str:
    if config.entry_mode == "legacy_early":
        return _classify_active_state_legacy_early(row, metrics, config)

    return _classify_active_state_confirmed(row, metrics, config)


def _classify_active_state_legacy_early(
    row: pd.Series,
    metrics: dict[str, Any],
    config: LifecycleConfig,
) -> str:
    early_days_cap = max(5.0, metrics["typical_run_length_days"] * config.early_days_frac)
    early_gain_cap = max(6.0, metrics["typical_run_magnitude_pct"] * config.early_gain_frac)
    early_drawdown_cap = max(4.0, metrics["typical_run_drawdown_pct"] * 0.75)
    mature_days_floor = max(10.0, metrics["typical_run_length_days"] * config.mature_days_frac)
    mature_gain_floor = max(10.0, metrics["typical_run_magnitude_pct"] * config.mature_gain_frac)
    exhausted_drawdown_floor = max(
        config.exhausted_drawdown_floor_pct,
        metrics["typical_run_drawdown_pct"],
    )

    if (
        metrics["days_elapsed"] <= early_days_cap
        and metrics["gain_so_far_pct"] <= early_gain_cap
        and metrics["extension_vs_norm"] <= config.early_extension_cap
        and metrics["drawdown_from_peak_pct"] <= early_drawdown_cap
        and metrics["trend_score"] >= config.early_trend_score_min
        and metrics["momentum_score"] >= config.early_momentum_score_min
    ):
        return "EARLY"

    if (
        metrics["extension_vs_norm"] >= config.exhausted_extension_frac
        and metrics["exhaustion_score"] >= config.exhausted_score_min
        and (
            _coalesce_float(row.get("macd_slope_3"), 0.0) < 0.0
            or _coalesce_float(row.get("rsi_slope_3"), 0.0) < 0.0
            or metrics["drawdown_from_peak_pct"] >= exhausted_drawdown_floor * 0.6
        )
    ) or (
        metrics["drawdown_from_peak_pct"] >= exhausted_drawdown_floor
        and _coalesce_float(row.get("macd_histogram"), 0.0) <= 0.0
        and _coalesce_float(row.get("rsi"), 50.0) <= 50.0
        and _coalesce_float(row.get("cmf"), 0.0) <= 0.0
    ):
        return "EXHAUSTED"

    if (
        metrics["days_elapsed"] >= mature_days_floor
        or metrics["gain_so_far_pct"] >= mature_gain_floor
        or metrics["extension_vs_norm"] >= config.mature_extension_floor
        or _coalesce_float(row.get("bb_pct_b"), 0.0) >= 0.9
        or _coalesce_float(row.get("zscore_50"), 0.0) >= 1.75
    ):
        return "MATURE"

    return "MID"


def _classify_active_state_confirmed(
    row: pd.Series,
    metrics: dict[str, Any],
    config: LifecycleConfig,
) -> str:
    mature_days_floor = max(10.0, metrics["typical_run_length_days"] * config.mature_days_frac)
    mature_gain_floor = max(10.0, metrics["typical_run_magnitude_pct"] * config.mature_gain_frac)
    exhausted_drawdown_floor = max(
        config.exhausted_drawdown_floor_pct,
        metrics["typical_run_drawdown_pct"],
    )

    exhausted = (
        (
            metrics["extension_vs_norm"] >= config.exhausted_extension_frac
            and metrics["exhaustion_score"] >= config.exhausted_score_min
            and (
                _coalesce_float(row.get("macd_slope_3"), 0.0) < 0.0
                or _coalesce_float(row.get("rsi_slope_3"), 0.0) < 0.0
                or metrics["drawdown_from_peak_pct"] >= exhausted_drawdown_floor * 0.6
            )
        )
        or (
            metrics["drawdown_from_peak_pct"] >= exhausted_drawdown_floor
            and _coalesce_float(row.get("macd_histogram"), 0.0) <= 0.0
            and _coalesce_float(row.get("rsi"), 50.0) <= 50.0
            and _coalesce_float(row.get("cmf"), 0.0) <= 0.0
        )
    )
    if exhausted:
        return "EXHAUSTED"

    mature = _is_mature_state_confirmed(row, metrics, config, mature_days_floor, mature_gain_floor)

    if _is_confirmed_entry(row, metrics, config, mature_days_floor, mature_gain_floor):
        return "CONFIRMED"

    if mature:
        return "MATURE"

    return "MID"


def _is_confirmed_entry(
    row: pd.Series,
    metrics: dict[str, Any],
    config: LifecycleConfig,
    mature_days_floor: float,
    mature_gain_floor: float,
) -> bool:
    rel_volume = _coalesce_float(row.get("rel_volume"), 0.0)
    cmf_value = _coalesce_float(row.get("cmf"), 0.0)
    adx_value = _coalesce_float(row.get("adx"), 0.0)
    plus_di = _coalesce_float(row.get("plus_di"), 0.0)
    minus_di = _coalesce_float(row.get("minus_di"), 0.0)
    macd_histogram = _coalesce_float(row.get("macd_histogram"), 0.0)
    macd_slope = _coalesce_float(row.get("macd_slope_3"), 0.0)
    rsi_value = _coalesce_float(row.get("rsi"), 50.0)
    rsi_slope = _coalesce_float(row.get("rsi_slope_3"), 0.0)

    confirmed_days_cap = max(7.0, metrics["typical_run_length_days"] * config.confirmed_days_frac)
    confirmed_gain_cap = max(8.0, metrics["typical_run_magnitude_pct"] * config.confirmed_gain_frac)
    confirmed_drawdown_cap = max(4.0, metrics["typical_run_drawdown_pct"] * 0.75)

    momentum_agreement = (
        metrics["days_elapsed"] >= config.confirmed_min_days
        and metrics["gain_so_far_pct"] >= config.confirmed_min_gain_pct
        and plus_di > minus_di
        and macd_histogram > 0.0
        and metrics["trend_score"] >= config.confirmed_trend_score_min
        and metrics["momentum_score"] >= config.confirmed_momentum_score_min
        and metrics["confirmation_score"] >= config.confirmed_confirmation_score_min
        and (
            adx_value >= max(config.confirmed_adx_min, config.trend_adx_min)
            or config.confirmed_rsi_min <= rsi_value <= config.confirmed_rsi_max
            or macd_slope >= 0.0
            or rsi_slope >= 0.0
        )
    )
    volume_agreement = (
        metrics["volume_score"] >= config.confirmed_volume_score_strong
        and (
            (rel_volume >= config.confirmed_rel_volume_min and cmf_value >= config.confirmed_cmf_min)
            or (rel_volume >= config.confirmed_rel_volume_strong and cmf_value >= -0.02)
            or cmf_value >= config.confirmed_cmf_strong
        )
    )
    not_late = (
        metrics["days_elapsed"] <= confirmed_days_cap
        and metrics["gain_so_far_pct"] <= confirmed_gain_cap
        and metrics["extension_vs_norm"] <= min(config.confirmed_extension_cap, config.mature_extension_floor)
        and metrics["drawdown_from_peak_pct"] <= confirmed_drawdown_cap
        and metrics["days_elapsed"] < mature_days_floor
        and metrics["gain_so_far_pct"] < mature_gain_floor
        and not _is_mature_state_confirmed(row, metrics, config, mature_days_floor, mature_gain_floor)
    )
    return bool(momentum_agreement and volume_agreement and not_late)


def _is_mature_state_confirmed(
    row: pd.Series,
    metrics: dict[str, Any],
    config: LifecycleConfig,
    mature_days_floor: float,
    mature_gain_floor: float,
) -> bool:
    bb_pct_b = _coalesce_float(row.get("bb_pct_b"), 0.0)
    zscore_50 = _coalesce_float(row.get("zscore_50"), 0.0)
    stretch_signal = (
        (bb_pct_b >= 0.95 or zscore_50 >= 2.0)
        and (
            metrics["extension_vs_norm"] >= config.mature_extension_floor * 0.85
            or metrics["gain_so_far_pct"] >= mature_gain_floor * 0.8
            or metrics["days_elapsed"] >= mature_days_floor * 0.8
        )
    )
    return bool(
        metrics["days_elapsed"] >= mature_days_floor
        or metrics["gain_so_far_pct"] >= mature_gain_floor
        or metrics["extension_vs_norm"] >= config.mature_extension_floor
        or stretch_signal
    )


def _classify_dormant_state(row: pd.Series, config: LifecycleConfig) -> str:
    adx_value = _coalesce_float(row.get("adx"), 0.0)
    close_value = _coalesce_float(row.get("close"), 0.0)
    prior_base_high = _coalesce_float(row.get("prior_base_high"), math.nan)
    base_range_pct = _coalesce_float(row.get("base_range_pct"), math.inf)

    if (
        adx_value <= config.dormant_adx_max
        and base_range_pct <= config.base_range_cap_pct * 1.2
        and (math.isnan(prior_base_high) or close_value < prior_base_high * config.breakout_buffer)
    ):
        return "DORMANT"
    return "DORMANT"


def _apply_state_hysteresis(
    *,
    current_state: str,
    raw_state: str,
    pending_state: Optional[str],
    pending_streak: int,
    active_run: bool,
    config: LifecycleConfig,
) -> tuple[str, Optional[str], int]:
    if not active_run:
        return "DORMANT", None, 0

    if raw_state == current_state:
        return current_state, None, 0

    if pending_state == raw_state:
        pending_streak += 1
    else:
        pending_state = raw_state
        pending_streak = 1

    required_days = _required_confirmation_days(current_state, raw_state, config)
    if pending_streak < required_days:
        return current_state, pending_state, pending_streak

    if not _transition_is_orderly(current_state, raw_state):
        return current_state, None, 0

    return raw_state, None, 0


def _required_confirmation_days(current_state: str, raw_state: str, config: LifecycleConfig) -> int:
    if current_state == raw_state:
        return 1

    if current_state == "EXHAUSTED" and raw_state in {"MATURE", "MID", "EARLY", "CONFIRMED"}:
        return config.exhausted_relief_confirm_days

    if _state_rank(raw_state) > _state_rank(current_state):
        return config.forward_confirm_days

    return config.backward_confirm_days


def _transition_is_orderly(current_state: str, raw_state: str) -> bool:
    if current_state == raw_state:
        return True
    if current_state == "DORMANT":
        return raw_state in {"EARLY", "CONFIRMED", "MID", "MATURE"}
    if current_state in {"EARLY", "CONFIRMED"}:
        return raw_state in {"MID", "MATURE", "DORMANT"}
    if current_state == "MID":
        return raw_state in {"EARLY", "CONFIRMED", "MATURE", "EXHAUSTED", "DORMANT"}
    if current_state == "MATURE":
        return raw_state in {"MID", "EXHAUSTED", "DORMANT"}
    if current_state == "EXHAUSTED":
        return raw_state in {"MATURE", "DORMANT"}
    return True


def _state_rank(state: str) -> int:
    return {
        "DORMANT": 0,
        "EARLY": 1,
        "CONFIRMED": 1,
        "MID": 2,
        "MATURE": 3,
        "EXHAUSTED": 4,
    }.get(state, 0)


def _is_move_end(
    row: pd.Series,
    metrics: dict[str, Any],
    prior_stats: dict[str, float],
    config: LifecycleConfig,
) -> bool:
    if metrics["days_elapsed"] < 3:
        return False

    typical_drawdown = max(prior_stats["typical_run_drawdown_pct"], config.exhausted_drawdown_floor_pct)
    trend_broken = (
        _coalesce_float(row.get("close"), 0.0) < _coalesce_float(row.get("ema_21"), 0.0)
        and _coalesce_float(row.get("macd_histogram"), 0.0) <= 0.0
        and _coalesce_float(row.get("rsi"), 50.0) < 48.0
    )
    deeper_break = (
        _coalesce_float(row.get("close"), 0.0) < _coalesce_float(row.get("ema_50"), 0.0)
        and _coalesce_float(row.get("cmf"), 0.0) < 0.0
    )
    drawdown_breach = metrics["drawdown_from_peak_pct"] >= max(
        config.exhausted_drawdown_floor_pct,
        typical_drawdown * config.stop_drawdown_frac,
    )
    return bool((trend_broken and drawdown_breach) or deeper_break)


def _coalesce_float(value: Any, default: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(converted) or math.isinf(converted):
        return default
    return converted


def _pct_gain(base_price: float, current_price: float) -> float:
    if base_price <= 0:
        return 0.0
    return (current_price / base_price - 1.0) * 100.0


def _pct_drop(peak_price: float, current_price: float) -> float:
    if peak_price <= 0:
        return 0.0
    return max(0.0, (peak_price - current_price) / peak_price * 100.0)


def _average(values: Sequence[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return 0.0
    return float(sum(clean) / len(clean))


def _scale_between(value: float, low: float, high: float) -> float:
    if math.isnan(value):
        return 0.0
    if high <= low:
        return 0.0
    clipped = min(max(value, low), high)
    return float((clipped - low) / (high - low))