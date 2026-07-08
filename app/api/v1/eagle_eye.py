"""
Eagle Eye API Router.

Exposes the Kuwait stock lifecycle rating system.

Endpoints:
  GET  /eagle-eye/scanner              — rated stock universe (filterable)
  GET  /eagle-eye/stocks/{ticker}      — full single-stock analysis
  GET  /eagle-eye/stocks/{ticker}/dna  — behavioral DNA
  GET  /eagle-eye/stocks/{ticker}/events — historical move events
  POST /eagle-eye/refresh              — queue background recompute
  GET  /eagle-eye/regime               — current market regime
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from app.api.deps import get_current_user, require_admin
from app.core.security import TokenData
from app.schemas.eagle_eye import (
    BehavioralDNAResponse,
    DNAResponse,
    DNASetupBarResponse,
    DNASetupExampleResponse,
    DNASetupForwardOutcomeResponse,
    DNASetupObservationResponse,
    DNAWindowProfileResponse,
    EventsListResponse,
    FullStockAnalysis,
    MoveEventResponse,
    RatedStock,
    RefreshRequest,
    RefreshResponse,
    RegimeResponse,
    ScannerResponse,
    SignalBreakdown,
    SignalReliabilityResponse,
    StockAnalysisResponse,
    SupportResistanceLevel,
    ThresholdProfileResponse,
    VolumeContextSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eagle-eye", tags=["Eagle Eye"])

# ---------------------------------------------------------------------------
# In-memory cache — keyed by "<TICKER>:<ISO_DATE>" to allow daily staleness
# ---------------------------------------------------------------------------
_cache: Dict[str, dict] = {}
_DNA_CACHE: Dict[str, dict] = {}
_EVENTS_CACHE: Dict[str, list] = {}
_DNA_BUILD_LOCK = threading.Lock()
_DNA_BUILD_IN_PROGRESS: set[str] = set()
_RECOMPUTE_LOCK = threading.Lock()
_RECOMPUTE_IN_PROGRESS = False
_RECOMPUTE_LAST_ATTEMPT_AT = 0.0

_LOOKBACK_YEARS = 5
_RECOMPUTE_COOLDOWN_SEC = 300

# ---------------------------------------------------------------------------
# Performance caches
# ---------------------------------------------------------------------------
# Stock meta map: ticker → StockMeta.  Rebuilt at most once every 10 minutes.
# Avoids hitting the analysis_stocks DB table on every /scanner request.
_META_MAP_CACHE: Optional[Dict[str, object]] = None
_META_MAP_CACHE_AT: float = 0.0
_META_MAP_TTL_SEC: float = 600.0  # 10 minutes

# Fundamentals map: ticker -> {pe_ratio, book_value_per_share, eps}.
# Built from local tables (stocks + stock_metrics) and refreshed every 10 min.
_FUNDAMENTALS_MAP_CACHE: Optional[Dict[str, Dict[str, Optional[float]]]] = None
_FUNDAMENTALS_MAP_CACHE_AT: float = 0.0
_FUNDAMENTALS_MAP_TTL_SEC: float = 600.0  # 10 minutes

# Latest close map: ticker -> latest cached close (fils for KSE universe).
_LATEST_CLOSE_MAP_CACHE: Optional[Dict[str, float]] = None
_LATEST_CLOSE_MAP_CACHE_AT: float = 0.0
_LATEST_CLOSE_MAP_TTL_SEC: float = 600.0  # 10 minutes

# Scanner response cache: assembled List[RatedStock] held for 30 s so
# rapid re-fetches (focus events, filter clicks) return instantly.
_SCANNER_RESP_CACHE: Optional[list] = None
_SCANNER_RESP_CACHE_AT: float = 0.0
_SCANNER_RESP_TTL_SEC: float = 30.0  # 30 seconds

# Market regime response cache: breadth changes slowly intraday and the
# endpoint is hit on every Eagle Eye screen load.
_REGIME_RESP_CACHE: Optional[dict] = None
_REGIME_RESP_CACHE_AT: float = 0.0
_REGIME_RESP_TTL_SEC: float = 600.0  # 10 minutes


def _get_meta_map() -> Dict[str, object]:
    """Return a cached ticker → StockMeta lookup, rebuilt at most every 10 min."""
    global _META_MAP_CACHE, _META_MAP_CACHE_AT
    now = time.time()
    if _META_MAP_CACHE is not None and (now - _META_MAP_CACHE_AT) < _META_MAP_TTL_SEC:
        return _META_MAP_CACHE
    try:
        from app.services.eagle_eye.adapter import TickerChartAdapter
        adapter = TickerChartAdapter()
        meta_map = {s.ticker: s for s in adapter.list_stocks()}
        _META_MAP_CACHE = meta_map
        _META_MAP_CACHE_AT = now
        return meta_map
    except Exception:
        logger.warning("Could not refresh meta map; using stale cache or empty dict")
        return _META_MAP_CACHE or {}


def _normalize_symbol(raw: object) -> str:
    sym = str(raw or "").upper().strip()
    if sym.endswith(".KW"):
        sym = sym[:-3]
    return sym


def _get_fundamentals_map() -> Dict[str, Dict[str, Optional[float]]]:
    """Return cached ticker fundamentals used by scanner table columns.

    Source priority:
    1) TickerChart QuotesSnapShot (P/E, LTM EPS, BVPS)
    2) ``stocks.pe_ratio``
    3) Latest ``stock_metrics`` values for:
         - "Book Value / Share"
         - "EPS" (used to derive P/E when direct PE is missing)
    4) Latest ``ml_fundamentals`` snapshot (fills remaining gaps)
    """
    global _FUNDAMENTALS_MAP_CACHE, _FUNDAMENTALS_MAP_CACHE_AT

    now = time.time()
    if (
        _FUNDAMENTALS_MAP_CACHE is not None
        and (now - _FUNDAMENTALS_MAP_CACHE_AT) < _FUNDAMENTALS_MAP_TTL_SEC
    ):
        return _FUNDAMENTALS_MAP_CACHE

    fmap: Dict[str, Dict[str, Optional[float]]] = {}

    try:
        from app.core.database import column_exists, query_all
        from app.services import tickerchart_service as tc

        # Primary source: TickerChart QuotesSnapShot fundamentals.
        # Scanner universe is Kuwait-focused, so symbols are resolved as KSE.
        for raw_ticker in _get_meta_map().keys():
            ticker = _normalize_symbol(raw_ticker)
            if not ticker:
                continue

            fmap.setdefault(
                ticker,
                {
                    "pe_ratio": None,
                    "book_value_per_share": None,
                    "eps": None,
                },
            )

            try:
                pe_val = _safe_float(tc.read_quotes_snapshot_pe(ticker, "KSE"))
                eps_val = _safe_float(tc.read_quotes_snapshot_ltm_eps(ticker, "KSE", price_divisor=1000.0))
                bvps_val = _safe_float(tc.read_quotes_snapshot_bvps(ticker, "KSE", price_divisor=1000.0))
            except Exception as exc:
                logger.debug("TickerChart snapshot fundamentals unavailable for %s: %s", ticker, exc)
                continue

            if pe_val is not None:
                fmap[ticker]["pe_ratio"] = pe_val
            if eps_val is not None:
                fmap[ticker]["eps"] = eps_val
            if bvps_val is not None:
                fmap[ticker]["book_value_per_share"] = bvps_val

        has_stocks_symbol = column_exists("stocks", "symbol")
        has_stocks_pe = column_exists("stocks", "pe_ratio")

        if has_stocks_symbol:
            pe_select = "pe_ratio" if has_stocks_pe else "NULL AS pe_ratio"
            rows = query_all(f"SELECT symbol, {pe_select} FROM stocks", ())
            for r in rows or []:
                ticker = _normalize_symbol(r.get("symbol"))
                if not ticker:
                    continue
                fmap.setdefault(
                    ticker,
                    {
                        "pe_ratio": None,
                        "book_value_per_share": None,
                        "eps": None,
                    },
                )
                pe_val = _safe_float(r.get("pe_ratio"))
                if fmap[ticker].get("pe_ratio") is None and pe_val is not None:
                    fmap[ticker]["pe_ratio"] = pe_val

        # Pull latest EPS + BVPS per ticker from stock_metrics.
        has_metric_name = column_exists("stock_metrics", "metric_name")
        has_metric_value = column_exists("stock_metrics", "metric_value")
        has_metric_stock_id = column_exists("stock_metrics", "stock_id")
        has_metric_id = column_exists("stock_metrics", "id")
        has_metric_period_end = column_exists("stock_metrics", "period_end_date")
        has_metric_year = column_exists("stock_metrics", "fiscal_year")
        has_metric_q = column_exists("stock_metrics", "fiscal_quarter")
        has_metric_created = column_exists("stock_metrics", "created_at")
        has_stocks_id = column_exists("stocks", "id")

        metric_filter_values = (
            "'book value / share',"
            "'book value per share',"
            "'book value/share',"
            "'book value per-share',"
            "'bvps',"
            "'eps',"
            "'earnings per share'"
        )

        def _nulls_last_desc(expr: str) -> str:
            # Cross-db nulls-last ordering that works for text/date/timestamp/numeric.
            return f"CASE WHEN {expr} IS NULL THEN 1 ELSE 0 END, {expr} DESC"

        def _metric_order_expr(alias: str) -> str:
            parts: list[str] = []
            if has_metric_period_end:
                parts.append(_nulls_last_desc(f"{alias}.period_end_date"))
            if has_metric_year:
                parts.append(_nulls_last_desc(f"{alias}.fiscal_year"))
            if has_metric_q:
                parts.append(_nulls_last_desc(f"{alias}.fiscal_quarter"))
            if has_metric_created:
                parts.append(_nulls_last_desc(f"{alias}.created_at"))
            if has_metric_id:
                parts.append(f"{alias}.id DESC")
            if not parts:
                parts.append(f"LOWER({alias}.metric_name) ASC")
            return ",\n                                ".join(parts)

        metric_order = _metric_order_expr("sm")

        def _merge_metric_rows(rows: list) -> None:
            for r in rows or []:
                ticker = _normalize_symbol(r.get("symbol"))
                if not ticker:
                    continue
                metric_name = str(r.get("metric_name") or "").lower().strip()
                metric_name_norm = " ".join(metric_name.replace("/", " ").replace("-", " ").split())
                metric_val = _safe_float(r.get("metric_value"))
                if metric_val is None:
                    continue

                fmap.setdefault(
                    ticker,
                    {
                        "pe_ratio": None,
                        "book_value_per_share": None,
                        "eps": None,
                    },
                )
                is_bvps = (
                    metric_name_norm == "bvps"
                    or ("book value" in metric_name_norm and "share" in metric_name_norm)
                )
                is_eps = metric_name_norm == "eps" or "earnings per share" in metric_name_norm
                if is_bvps:
                    if fmap[ticker].get("book_value_per_share") is None:
                        fmap[ticker]["book_value_per_share"] = metric_val
                elif is_eps:
                    if fmap[ticker].get("eps") is None:
                        fmap[ticker]["eps"] = metric_val

        has_metrics_table = (
            has_metric_name
            and has_metric_value
            and has_metric_stock_id
        )

        if has_metrics_table and has_stocks_symbol and has_stocks_id:
            try:
                metric_rows = query_all(
                    f"""
                    WITH ranked AS (
                        SELECT
                            s.symbol AS symbol,
                            LOWER(sm.metric_name) AS metric_name,
                            sm.metric_value AS metric_value,
                            ROW_NUMBER() OVER (
                                PARTITION BY s.symbol, LOWER(sm.metric_name)
                                ORDER BY
                                    {metric_order}
                            ) AS rn
                        FROM stock_metrics sm
                        JOIN stocks s ON s.id = sm.stock_id
                        WHERE LOWER(sm.metric_name) IN ({metric_filter_values})
                    )
                    SELECT symbol, metric_name, metric_value
                    FROM ranked
                    WHERE rn = 1
                    """,
                    (),
                )
                _merge_metric_rows(metric_rows)
            except Exception as exc:
                logger.warning("stock_metrics->stocks fundamentals query failed: %s", exc)

        has_master_id = column_exists("stocks_master", "id")
        has_master_ticker = column_exists("stocks_master", "ticker")
        if has_metrics_table and has_master_id and has_master_ticker:
            try:
                metric_rows_master = query_all(
                    f"""
                    WITH ranked AS (
                        SELECT
                            smt.ticker AS symbol,
                            LOWER(sm.metric_name) AS metric_name,
                            sm.metric_value AS metric_value,
                            ROW_NUMBER() OVER (
                                PARTITION BY smt.ticker, LOWER(sm.metric_name)
                                ORDER BY
                                    {metric_order}
                            ) AS rn
                        FROM stock_metrics sm
                        JOIN stocks_master smt ON smt.id = sm.stock_id
                        WHERE LOWER(sm.metric_name) IN ({metric_filter_values})
                    )
                    SELECT symbol, metric_name, metric_value
                    FROM ranked
                    WHERE rn = 1
                    """,
                    (),
                )
                _merge_metric_rows(metric_rows_master)
            except Exception as exc:
                logger.warning("stock_metrics->stocks_master fundamentals query failed: %s", exc)

        # Fill any remaining gaps from latest ml_fundamentals snapshots.
        has_mlf_ticker = column_exists("ml_fundamentals", "stock_ticker")
        has_mlf_disclosure = column_exists("ml_fundamentals", "disclosure_date")
        has_mlf_id = column_exists("ml_fundamentals", "id")
        has_mlf_period_end = column_exists("ml_fundamentals", "period_end_date")
        has_mlf_created = column_exists("ml_fundamentals", "created_at")
        has_mlf_pe = column_exists("ml_fundamentals", "pe_ratio")
        has_mlf_eps = column_exists("ml_fundamentals", "eps")
        has_mlf_bvps = column_exists("ml_fundamentals", "book_value_per_share")

        mlf_order_parts: list[str] = []
        if has_mlf_disclosure:
            mlf_order_parts.append(_nulls_last_desc("disclosure_date"))
        if has_mlf_period_end:
            mlf_order_parts.append(_nulls_last_desc("period_end_date"))
        if has_mlf_created:
            mlf_order_parts.append(_nulls_last_desc("created_at"))
        if has_mlf_id:
            mlf_order_parts.append("id DESC")
        if not mlf_order_parts:
            mlf_order_parts.append("UPPER(TRIM(stock_ticker)) ASC")
        mlf_order = ",\n                                ".join(mlf_order_parts)

        if has_mlf_ticker and has_mlf_disclosure and has_mlf_id:
            try:
                mlf_rows = query_all(
                    f"""
                    WITH ranked AS (
                        SELECT
                            stock_ticker AS symbol,
                            {'pe_ratio' if has_mlf_pe else 'NULL'} AS pe_ratio,
                            {'eps' if has_mlf_eps else 'NULL'} AS eps,
                            {'book_value_per_share' if has_mlf_bvps else 'NULL'} AS book_value_per_share,
                            ROW_NUMBER() OVER (
                                PARTITION BY UPPER(TRIM(stock_ticker))
                                ORDER BY
                                    {mlf_order}
                            ) AS rn
                        FROM ml_fundamentals
                    )
                    SELECT symbol, pe_ratio, eps, book_value_per_share
                    FROM ranked
                    WHERE rn = 1
                    """,
                    (),
                )

                for r in mlf_rows or []:
                    ticker = _normalize_symbol(r.get("symbol"))
                    if not ticker:
                        continue

                    fmap.setdefault(
                        ticker,
                        {
                            "pe_ratio": None,
                            "book_value_per_share": None,
                            "eps": None,
                        },
                    )

                    pe_val = _safe_float(r.get("pe_ratio"))
                    eps_val = _safe_float(r.get("eps"))
                    bvps_val = _safe_float(r.get("book_value_per_share"))

                    if fmap[ticker].get("pe_ratio") is None and pe_val is not None:
                        fmap[ticker]["pe_ratio"] = pe_val
                    if fmap[ticker].get("eps") is None and eps_val is not None:
                        fmap[ticker]["eps"] = eps_val
                    if fmap[ticker].get("book_value_per_share") is None and bvps_val is not None:
                        fmap[ticker]["book_value_per_share"] = bvps_val
            except Exception as exc:
                logger.warning("ml_fundamentals fallback query failed: %s", exc)

        _FUNDAMENTALS_MAP_CACHE = fmap
        _FUNDAMENTALS_MAP_CACHE_AT = now
        return fmap
    except Exception as exc:
        logger.warning("Could not refresh fundamentals map; using stale cache or empty dict: %s", exc)
        return _FUNDAMENTALS_MAP_CACHE or {}


def _cache_key(ticker: str, as_of: Optional[date] = None) -> str:
    d = (as_of or date.today()).isoformat()
    return f"{ticker.upper()}:{d}"


def _get_latest_close_map() -> Dict[str, float]:
    """Return cached ticker -> latest close map from ee_ohlcv_cache."""
    global _LATEST_CLOSE_MAP_CACHE, _LATEST_CLOSE_MAP_CACHE_AT

    now = time.time()
    if (
        _LATEST_CLOSE_MAP_CACHE is not None
        and (now - _LATEST_CLOSE_MAP_CACHE_AT) < _LATEST_CLOSE_MAP_TTL_SEC
    ):
        return _LATEST_CLOSE_MAP_CACHE

    close_map: Dict[str, float] = {}
    try:
        from app.core.database import query_all

        rows = query_all(
            """
            SELECT c.ticker, c.close
            FROM ee_ohlcv_cache c
            JOIN (
                SELECT ticker, MAX(bar_date) AS max_bar_date
                FROM ee_ohlcv_cache
                GROUP BY ticker
            ) mx
              ON mx.ticker = c.ticker
             AND mx.max_bar_date = c.bar_date
            """,
            (),
        )

        for row in rows or []:
            ticker = _normalize_symbol(row.get("ticker"))
            close_val = _safe_float(row.get("close"))
            if ticker and close_val is not None:
                close_map[ticker] = close_val

        _LATEST_CLOSE_MAP_CACHE = close_map
        _LATEST_CLOSE_MAP_CACHE_AT = now
        return close_map
    except Exception as exc:
        logger.warning("Could not refresh latest close map; using stale cache or empty dict: %s", exc)
        return _LATEST_CLOSE_MAP_CACHE or {}


def _is_computed_today(computed_at: object, today_iso: Optional[str] = None) -> bool:
    """Treat both date-only and timestamp values as fresh for the same day."""
    if computed_at is None:
        return False
    value = str(computed_at).strip()
    if not value:
        return False
    day = today_iso or date.today().isoformat()
    return (value[:10] == day) if len(value) >= 10 else (value == day)


def _safe_float(v) -> Optional[float]:
    """Coerce to float; return None for NaN, Inf, or anything non-numeric."""
    if v is None:
        return None
    try:
        import math
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _derive_pe_from_price_eps(last_price_fils: Optional[float], eps_kwd: Optional[float]) -> Optional[float]:
    """Derive P/E using KSE price units (fils) and EPS in KWD."""
    if last_price_fils is None or eps_kwd is None:
        return None
    if last_price_fils <= 0 or eps_kwd <= 0:
        return None
    # KSE prices are stored in fils; EPS is in KWD.
    derived = (last_price_fils / 1000.0) / eps_kwd
    if derived <= 0 or derived > 5000:
        return None
    return derived


def _extract_mce_from_reason(reason: object) -> Optional[float]:
    """Parse MCE value from disable reason text, if present."""
    if reason is None:
        return None
    match = re.search(r"\bMCE\s*=\s*([0-9]+(?:\.[0-9]+)?)", str(reason), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _build_live_feature_vector(feature_row: Dict[str, object], feature_names: List[str]) -> List[float]:
    """Build a model-ordered feature vector from a live feature row."""
    out: List[float] = []
    for name in feature_names:
        val = feature_row.get(name)
        fv = _safe_float(val)
        out.append(float("nan") if fv is None else float(fv))
    return out


def predict_confidence(ticker: str, ohlcv_df, as_of: date) -> Optional[float]:
    """Retired in the Phase 1 rules-primary rebuild."""
    del ticker, ohlcv_df, as_of
    return None


def _predict_ml_signal(ticker: str, ohlcv_df, as_of: date) -> Optional[Dict[str, float]]:
    """Retired compatibility wrapper for legacy ML payloads."""
    del ticker, ohlcv_df, as_of
    return None


def _predict_ml_opportunity_score(ticker: str, ohlcv_df, as_of: date) -> Optional[float]:
    """Retired compatibility wrapper for phase-regression output."""
    del ticker, ohlcv_df, as_of
    return None


def _build_threshold_profile_response(
    threshold_profile: dict,
    fallback_total_setups: int,
) -> ThresholdProfileResponse:
    hits = threshold_profile.get("occurrences", threshold_profile.get("hits", 0))
    sample_count = threshold_profile.get("sample_count", hits)
    total_setups = threshold_profile.get("sample_count", threshold_profile.get("total_count", fallback_total_setups))
    return ThresholdProfileResponse(
        threshold_pct=threshold_profile.get("threshold_pct", 0),
        success_rate=threshold_profile.get("success_rate", 0),
        sample_count=sample_count,
        total_count=total_setups,
        hits=hits,
        total_setups=total_setups,
        median_bars_to_hit=threshold_profile.get("median_bars_to_hit"),
        avg_win_pct=threshold_profile.get("avg_gain_on_hits_pct", threshold_profile.get("avg_gain_pct", threshold_profile.get("avg_win_pct"))),
        avg_loss_pct=threshold_profile.get("avg_loss_pct"),
        avg_gain_all_pct=threshold_profile.get("avg_gain_all_pct"),
        avg_gain_on_hits_pct=threshold_profile.get("avg_gain_on_hits_pct", threshold_profile.get("avg_gain_pct", threshold_profile.get("avg_win_pct"))),
    )


def _build_window_profile_response(window_profile: dict) -> DNAWindowProfileResponse:
    setup_count = window_profile.get("setup_count", 0)
    threshold_profiles = [
        _build_threshold_profile_response(threshold_profile, setup_count)
        for threshold_profile in window_profile.get("threshold_profiles", [])
    ]
    return DNAWindowProfileResponse(
        horizon_days=window_profile.get("horizon_days", 0),
        setup_count=setup_count,
        history_status=window_profile.get("history_status", "ok"),
        confidence_floor=window_profile.get("confidence_floor", 5),
        confidence_tier=window_profile.get("confidence_tier", "TOO_THIN"),
        confidence_label=window_profile.get("confidence_label", "Too thin"),
        percentages_visible=window_profile.get("percentages_visible", False),
        threshold_profiles=threshold_profiles,
    )


_BAR_FLOAT_FIELDS = (
    "open", "high", "low", "close", "volume", "rel_volume",
    "rsi", "macd_line", "macd_signal", "macd_histogram",
    "adx", "plus_di", "minus_di",
)


def _build_setup_example_response(setup_example: dict) -> DNASetupExampleResponse:
    bars: list[DNASetupBarResponse] = []
    for raw_bar in setup_example.get("bars", []):
        sanitized = {"date": str(raw_bar.get("date", ""))}
        for field in _BAR_FLOAT_FIELDS:
            sanitized[field] = _safe_float(raw_bar.get(field))
        # Fix bars where open was stored as 0 (API didn't report it).
        # Use close as a neutral fallback so the chart never shows 0.
        if not sanitized.get("open") and sanitized.get("close"):
            sanitized["open"] = sanitized["close"]
        bars.append(DNASetupBarResponse(**sanitized))

    observations = [
        DNASetupObservationResponse(**observation)
        for observation in setup_example.get("observations", [])
    ]
    forward_outcomes = {
        key: DNASetupForwardOutcomeResponse(**outcome)
        for key, outcome in setup_example.get("forward_outcomes", {}).items()
    }
    return DNASetupExampleResponse(
        setup_date=setup_example.get("setup_date", ""),
        setup_window_start_date=setup_example.get("setup_window_start_date", ""),
        setup_window_end_date=setup_example.get("setup_window_end_date", ""),
        setup_bar_index=setup_example.get("setup_bar_index", 0),
        setup_window_start_index=setup_example.get("setup_window_start_index", 0),
        setup_window_end_index=setup_example.get("setup_window_end_index", 0),
        available_forward_bars=setup_example.get("available_forward_bars", 0),
        bars=bars,
        observations=observations,
        forward_outcomes=forward_outcomes,
    )


def _trigger_eagle_eye_recompute(reason: str, *, force: bool = False) -> bool:
    """Best-effort background recompute trigger with per-process cooldown."""
    global _RECOMPUTE_IN_PROGRESS, _RECOMPUTE_LAST_ATTEMPT_AT

    now = time.time()
    with _RECOMPUTE_LOCK:
        if _RECOMPUTE_IN_PROGRESS:
            logger.info("Eagle Eye recompute already in progress; skip trigger (%s)", reason)
            return False
        if not force and (now - _RECOMPUTE_LAST_ATTEMPT_AT) < _RECOMPUTE_COOLDOWN_SEC:
            logger.info("Eagle Eye recompute cooldown active; skip trigger (%s)", reason)
            return False
        _RECOMPUTE_IN_PROGRESS = True
        _RECOMPUTE_LAST_ATTEMPT_AT = now

    def _runner() -> None:
        global _RECOMPUTE_IN_PROGRESS
        try:
            from app.services.eagle_eye.ingest import run_nightly_recompute

            result = run_nightly_recompute(dna_refresh=False, verbose=False)
            logger.info("Eagle Eye background recompute finished (%s): %s", reason, result)
        except Exception:
            logger.exception("Eagle Eye background recompute failed (%s)", reason)
        finally:
            with _RECOMPUTE_LOCK:
                _RECOMPUTE_IN_PROGRESS = False

    try:
        thread = threading.Thread(
            target=_runner,
            daemon=True,
            name=f"ee_recompute_{reason}",
        )
        thread.start()
        logger.info("Eagle Eye background recompute triggered (%s)", reason)
        return True
    except Exception:
        with _RECOMPUTE_LOCK:
            _RECOMPUTE_IN_PROGRESS = False
        logger.exception("Could not start Eagle Eye background recompute (%s)", reason)
        return False


def _trigger_dna_build(ticker: str, cache_key: str) -> bool:
    """Best-effort background DNA build for a single ticker."""
    with _DNA_BUILD_LOCK:
        if ticker in _DNA_BUILD_IN_PROGRESS:
            logger.info("DNA build already in progress for %s", ticker)
            return False
        _DNA_BUILD_IN_PROGRESS.add(ticker)

    def _runner() -> None:
        try:
            from app.services.eagle_eye.ingest import build_dna_for_ticker

            rebuilt = build_dna_for_ticker(ticker)
            if rebuilt is not None:
                _DNA_CACHE.pop(cache_key, None)
                logger.info("Background DNA build finished for %s", ticker)
            else:
                logger.info("Background DNA build found insufficient data for %s", ticker)
        except Exception:
            logger.exception("Background DNA build failed for %s", ticker)
        finally:
            with _DNA_BUILD_LOCK:
                _DNA_BUILD_IN_PROGRESS.discard(ticker)

    thread = threading.Thread(
        target=_runner,
        daemon=True,
        name=f"ee_dna_{ticker}",
    )
    thread.start()
    logger.info("Background DNA build triggered for %s", ticker)
    return True


# ---------------------------------------------------------------------------
# Shared analysis helper
# ---------------------------------------------------------------------------

def _run_analysis(ticker: str) -> Optional[dict]:
    """
    Execute the Eagle Eye pipeline for a single ticker and cache the result.

    Fast path: checks ee_ratings_cache for a row computed today.
    Falls back to live TickerChart fetch + indicator computation.
    Returns a plain dict containing all analysis outputs, or None on failure.
    """
    key = _cache_key(ticker)
    if key in _cache:
        return _cache[key]

    # ── DB fast path: today's pre-computed rating ──
    try:
        from app.services.eagle_eye.rating_engine import (
            is_stock_active,
        )
        from app.services.eagle_eye.store import load_ohlcv, load_rating

        cached_row = load_rating(ticker)
        if cached_row and _is_computed_today(cached_row.get("computed_at")):
            ohlcv_cached = load_ohlcv(ticker)
            if ohlcv_cached is None or len(ohlcv_cached) == 0:
                raise ValueError(f"Missing cached OHLCV for {ticker}")
            if not is_stock_active(ticker, ohlcv_cached):
                return None

            indicators = cached_row.get("indicators_json") or {}
            if isinstance(indicators, str):
                import json
                indicators = json.loads(indicators)

            if not isinstance(indicators, dict):
                indicators = {}

            supports = cached_row.get("supports_json") or []
            resistances = cached_row.get("resistances_json") or []
            # Always refresh display close from cached OHLCV so detail price tracks
            # daily ingestion even if a prior rating row had stale indicators_json.
            latest_close = _safe_float(ohlcv_cached["close"].iloc[-1])
            if latest_close is not None:
                indicators["close"] = latest_close
            entry = {
                "entry_primary": cached_row.get("entry_primary"),
                "entry_aggressive": cached_row.get("entry_aggressive"),
                "entry_conservative": cached_row.get("entry_conservative"),
                "plan_state": "ACTIVE",
                "plan_reason": None,
                "conditional_entry": None,
                "stop_loss": cached_row.get("stop_loss"),
                "tp1": cached_row.get("tp1"),
                "tp1_probability": cached_row.get("tp1_probability"),
                "tp2": cached_row.get("tp2"),
                "tp2_probability": cached_row.get("tp2_probability"),
                "tp3": cached_row.get("tp3"),
                "tp3_probability": cached_row.get("tp3_probability"),
                "risk_reward_ratio": None,
                "gain_pct_to_tp1": None,
            }
            try:
                from app.services.eagle_eye.adapter import TickerChartAdapter
                from app.services.eagle_eye.rating_engine import compute_entry_stop_targets

                adapter = TickerChartAdapter()
                end_d = date.today()
                start_d = end_d - timedelta(days=_LOOKBACK_YEARS * 365 + 60)
                df = adapter.get_ohlcv_daily(ticker, start_d, end_d)
                if df is not None and len(df) >= 30 and isinstance(indicators, dict):
                    entry = compute_entry_stop_targets(
                        df,
                        indicators,
                        {"supports": supports, "resistances": resistances},
                        stage=cached_row.get("stage"),
                    )
            except Exception as exc:
                logger.debug("Live trade-plan refresh miss for %s: %s", ticker, exc)
            result = {
                "ticker": ticker.upper(),
                "stage": cached_row.get("stage"),
                "rating": cached_row.get("rating"),
                "confidence": cached_row.get("confidence"),
                "ml_score": cached_row.get("ml_score"),
                "thesis": cached_row.get("thesis"),
                "supports": supports,
                "resistances": resistances,
                "entry": entry,
                "indicators": indicators,
                "days_of_history": cached_row.get("days_of_history"),
                "computed_at": cached_row.get("computed_at"),
            }
            _cache[key] = result
            return result
    except Exception as exc:
        logger.debug("DB rating cache miss for %s: %s", ticker, exc)

    # ── Live compute fallback ──
    try:
        from app.services.eagle_eye.adapter import TickerChartAdapter
        from app.services.eagle_eye.indicators import compute_all_indicators
        from app.services.eagle_eye.rating_engine import (
            compute_entry_stop_targets,
            compute_support_resistance,
            compute_volume_context,
            generate_thesis,
            is_stock_active,
        )
        from app.services.eagle_eye.scoring.explanation_engine import explain
        from app.services.eagle_eye.scoring.family_scores import compute_family_scores
        from app.services.eagle_eye.scoring.recommendation_engine import generate_recommendation
        from app.services.eagle_eye.stage_classifier import classify_stage_with_confidence

        adapter = TickerChartAdapter()
        end_d = date.today()
        start_d = end_d - timedelta(days=_LOOKBACK_YEARS * 365 + 60)

        df = adapter.get_ohlcv_daily(ticker, start_d, end_d)
        if df is None or len(df) < 30:
            return None
        if not is_stock_active(ticker, df):
            return None

        indicators_df = compute_all_indicators(df)
        if indicators_df is None or len(indicators_df) == 0:
            return None

        latest = indicators_df.iloc[-1].to_dict()

        family_scores = compute_family_scores(latest)
        stage, stage_conf = classify_stage_with_confidence(latest, family_scores=family_scores)

        recommendation_payload = generate_recommendation(
            latest,
            family_scores=family_scores,
            total_score=float(family_scores.get("total_score", 50.0)),
            stage=stage,
            stage_conf=stage_conf,
            pattern_match=None,
            data_quality=_safe_float(latest.get("data_quality_score")) or 50.0,
        )

        confidence = float(recommendation_payload["confidence"])
        rating = str(recommendation_payload["recommendation"])
        ml_score = None
        ml_proba = None

        # Keep volume context for display only (no post-model multipliers).
        volume_context = compute_volume_context(df, stage)

        sr = compute_support_resistance(df, latest)
        et = compute_entry_stop_targets(df, latest, sr, stage=stage)
        explanation = explain(recommendation_payload, latest, pattern_match=None)
        top_signals = explanation.get("why_supporting", [])[:2]
        thesis = generate_thesis(ticker, rating, stage, latest, dna=None, top_signals_fired=top_signals)

        result = {
            "ticker": ticker.upper(),
            "stage": stage,
            "stage_confidence": recommendation_payload.get("stage_confidence"),
            "rating": rating,
            "recommendation": rating,
            "confidence": confidence,
            "ml_score": ml_score,
            "ml_proba": ml_proba,
            "thesis": thesis,
            "supports": sr.get("supports", []),
            "resistances": sr.get("resistances", []),
            "entry": et,
            "indicators": latest,
            "family_scores": family_scores,
            "why_supporting": explanation.get("why_supporting", []),
            "why_conflicting": explanation.get("why_conflicting", []),
            "what_invalidates": explanation.get("what_invalidates", []),
            "veto_reasons": recommendation_payload.get("veto_reasons", []),
            "volume_context": volume_context,
            "days_of_history": len(df),
            "computed_at": datetime.utcnow().date().isoformat(),
        }
        _cache[key] = result
        return result

    except Exception as exc:
        logger.warning("Eagle Eye analysis failed for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# GET /eagle-eye/scanner
# ---------------------------------------------------------------------------

@router.get("/scanner", response_model=ScannerResponse, summary="Scan all Kuwait stocks")
async def get_scanner(
    sector: Optional[str] = Query(None, description="Filter by sector"),
    tier: Optional[str] = Query(None, description="Filter by market tier"),
    min_confidence: float = Query(0.0, ge=0, le=100, description="Minimum confidence score"),
    limit: int = Query(200, ge=1, le=500, description="Maximum number of stocks to return"),
    _user: TokenData = Depends(get_current_user),
):
    """
    Return a rated list of Kuwait stocks, optionally filtered by sector, tier, and
    minimum confidence. Reads from ee_ratings_cache (pre-computed nightly) for
    instant response; falls back to live per-stock computation when the cache is empty.

    Performance notes:
    - StockMeta lookup is cached for 10 minutes (_get_meta_map).
    - The assembled response list is cached for 30 seconds for rapid re-fetches.
    - min_confidence filtering is pushed to SQL in load_all_ratings().
    - limit defaults to 200 so the frontend always receives the full universe
      and can do all secondary filtering client-side without extra round trips.
    """
    global _SCANNER_RESP_CACHE, _SCANNER_RESP_CACHE_AT

    # ── 30-second response cache for identical unfiltered requests ───────────
    # Sector/tier filters bypass the cache so they remain correct.
    now = time.time()
    use_resp_cache = (
        not sector
        and not tier
        and min_confidence == 0.0
        and _SCANNER_RESP_CACHE is not None
        and (now - _SCANNER_RESP_CACHE_AT) < _SCANNER_RESP_TTL_SEC
    )
    if use_resp_cache:
        cached = _SCANNER_RESP_CACHE
        return ScannerResponse(status="ok", count=len(cached), stocks=cached)

    # ── DB fast path: read pre-computed ratings ──────────────────────────────
    # NOTE: Live compute fallback removed — it fetched OHLCV for 100+ stocks
    # synchronously in an async handler, blocking the event loop for minutes.
    # The background warmup (started at app startup) populates ee_ratings_cache.
    # Return warming_up immediately when cache is cold so the UI stays responsive.
    try:
        from app.services.eagle_eye.rating_engine import is_stock_active
        from app.services.eagle_eye.store import load_all_ratings, load_ohlcv

        db_rows = load_all_ratings(min_confidence=min_confidence, limit=limit)
        if not db_rows:
            # Cache is cold — retrigger a best-effort background warmup and respond immediately.
            _trigger_eagle_eye_recompute("scanner_cache_cold")
            logger.info("Eagle Eye scanner: cache cold, returning warming_up status")
            return ScannerResponse(status="warming_up", count=0, stocks=[])

        # Keep scanner consistent with detail endpoint freshness policy.
        # Detail view only trusts same-day ee_ratings_cache rows before falling
        # back to live compute. Apply the same rule here to avoid CONF mismatch
        # between list and detail screens.
        today_iso = date.today().isoformat()
        fresh_rows = [row for row in db_rows if _is_computed_today(row.get("computed_at"), today_iso)]
        if not fresh_rows:
            _trigger_eagle_eye_recompute("scanner_cache_stale")
            logger.info("Eagle Eye scanner: cache stale, returning warming_up status")
            return ScannerResponse(status="warming_up", count=0, stocks=[])
        db_rows = fresh_rows

        # ── Use cached meta map (rebuilt at most every 10 min) ───────────────
        meta_map = _get_meta_map()
        fundamentals_map = _get_fundamentals_map()
        latest_close_map = _get_latest_close_map()

        results: List[RatedStock] = []
        for row in db_rows:
            t = str(row.get("ticker") or "").upper()
            meta = meta_map.get(t)
            row_sector = str(row.get("sector") or (meta.sector if meta else "Kuwait"))
            row_name = str(row.get("name_en") or (meta.name_en if meta else t))
            row_tier = meta.market_tier if meta else "premier"

            if sector and row_sector.lower() != sector.lower():
                continue
            if tier and row_tier.lower() != tier.lower():
                continue

            conf = float(row.get("confidence") or 0.0)

            if row.get("ml_score") is not None and conf > 60.0:
                try:
                    live_df = load_ohlcv(t)
                    if not is_stock_active(t, live_df):
                        continue
                    if live_df is not None and len(live_df) >= 20:
                        low_20d = _safe_float(live_df["low"].tail(20).min())
                        close_now = _safe_float(live_df["close"].iloc[-1])
                        if low_20d is not None and close_now is not None and low_20d > 0:
                            ext_20d = (close_now / low_20d - 1.0) * 100.0
                            if ext_20d > 30.0:
                                conf = min(conf, 30.0)
                                row["rating"] = "HOLD"
                except Exception as exc:
                    logger.debug("Scanner safety recheck skipped for %s: %s", t, exc)

            vc_raw = row.get("volume_context") or {}
            vc_summary = VolumeContextSummary(
                relative_volume=float(vc_raw.get("relative_volume") or 1.0),
                liquidity_tier=str(vc_raw.get("liquidity_tier") or "TRADEABLE"),
                is_volume_confirmed=bool(vc_raw.get("is_volume_confirmed", True)),
                volume_character=str(vc_raw.get("volume_character") or "NEUTRAL"),
                volume_trend_5d=str(vc_raw.get("volume_trend_5d") or "NEUTRAL"),
            ) if vc_raw else None

            fmeta = fundamentals_map.get(t, {})
            bvps = _safe_float(fmeta.get("book_value_per_share"))
            # Use the most recent close from OHLCV for the scanner "Current" field.
            # Fallback to rating-row last_price when no OHLCV bar is available.
            last_price = _safe_float(latest_close_map.get(t))
            if last_price is None:
                last_price = _safe_float(row.get("last_price"))

            eps_latest = _safe_float(fmeta.get("eps"))
            pe_ratio = _derive_pe_from_price_eps(last_price, eps_latest)
            if pe_ratio is None:
                pe_ratio = _safe_float(fmeta.get("pe_ratio"))

            results.append(RatedStock(
                ticker=t,
                name_en=row_name,
                sector=row_sector,
                stage=row.get("stage"),
                rating=row.get("rating"),
                confidence=conf,
                thesis=row.get("thesis"),
                entry_primary=row.get("entry_primary"),
                stop_loss=row.get("stop_loss"),
                tp1=row.get("tp1"),
                last_price=last_price,
                book_value_per_share=round(bvps, 3) if bvps is not None else None,
                pe_ratio=round(pe_ratio, 2) if pe_ratio is not None else None,
                computed_at=row.get("computed_at"),
                volume_context=vc_summary,
            ))

        # Cache the unfiltered response for 30 s
        if not sector and not tier and min_confidence == 0.0:
            _SCANNER_RESP_CACHE = results
            _SCANNER_RESP_CACHE_AT = now

        return ScannerResponse(status="ok", count=len(results), stocks=results)

    except Exception as exc:
        logger.warning("DB scanner failed: %s", exc)
        return ScannerResponse(status="error", count=0, stocks=[])


# ---------------------------------------------------------------------------
# GET /eagle-eye/stocks/{ticker}
# ---------------------------------------------------------------------------

@router.get("/stocks/{ticker}", response_model=StockAnalysisResponse, summary="Full stock analysis")
async def get_stock_analysis(
    ticker: str,
    portfolio_kwd: float = Query(0.0, description="Portfolio size in KWD for position sizing"),
    _user: TokenData = Depends(get_current_user),
):
    """
    Return full Eagle Eye analysis for a single Kuwait ticker.
    Includes stage, rating, confidence, SR levels, entry/stop/targets, and signals.
    """
    t = ticker.upper().strip()
    analysis = _run_analysis(t)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"No data found for ticker '{t}'")

    et = analysis.get("entry", {})
    ind = analysis.get("indicators", {})

    # Build signal breakdown from top indicator categories
    signals: List[SignalBreakdown] = []
    _SIGNAL_KEYS = [
        ("rsi", "RSI"),
        ("macd_histogram", "MACD Histogram"),
        ("adx", "ADX"),
        ("cmf", "Chaikin Money Flow"),
        ("accumulation_score", "Accumulation Score"),
        ("obv_slope_20", "OBV Slope"),
        ("ema_ribbon_aligned", "EMA Ribbon"),
        ("bb_squeeze", "Bollinger Squeeze"),
        ("mfi", "Money Flow Index"),
        ("supertrend_signal", "Supertrend"),
    ]
    for key, desc in _SIGNAL_KEYS:
        v = ind.get(key)
        if v is not None:
            signals.append(SignalBreakdown(
                signal=key,
                fired=bool(v) if isinstance(v, (bool, int)) else v > 0 if isinstance(v, float) else False,
                value=float(v) if isinstance(v, (int, float)) else None,
                description=desc,
            ))

    # Position sizing (optional — only if portfolio_kwd provided)
    pos_size_pct = pos_size_kwd = liq_capped = req_confirm = None
    if portfolio_kwd > 0 and et.get("plan_state") == "ACTIVE":
        from app.services.eagle_eye.rating_engine import compute_position_size
        entry_p = et.get("entry_primary", 0.0)
        stop_p = et.get("stop_loss", entry_p * 0.95)
        avg_turn = float(ind.get("avg_daily_turnover_kwd", portfolio_kwd * 0.01) or portfolio_kwd * 0.01)
        sizing = compute_position_size(
            analysis["confidence"], entry_p, stop_p, portfolio_kwd, avg_turn
        )
        pos_size_pct = sizing["size_pct"]
        pos_size_kwd = sizing["suggested_kwd"]
        liq_capped = sizing["liquidity_capped"]
        req_confirm = sizing["requires_confirmation"]

    sr_supports = [SupportResistanceLevel(**s) for s in analysis.get("supports", [])]
    sr_resistances = [SupportResistanceLevel(**r) for r in analysis.get("resistances", [])]

    data = FullStockAnalysis(
        ticker=analysis["ticker"],
        name_en=analysis["ticker"],  # name resolved by adapter if available
        sector="Kuwait",
        stage=analysis["stage"],
        rating=analysis["rating"],
        confidence=analysis["confidence"],
        thesis=analysis["thesis"],
        supports=sr_supports,
        resistances=sr_resistances,
        entry_primary=et.get("entry_primary"),
        entry_aggressive=et.get("entry_aggressive"),
        entry_conservative=et.get("entry_conservative"),
        plan_state=et.get("plan_state", "ACTIVE"),
        plan_reason=et.get("plan_reason"),
        conditional_entry=et.get("conditional_entry"),
        stop_loss=et.get("stop_loss"),
        tp1=et.get("tp1"),
        tp1_probability=et.get("tp1_probability"),
        tp2=et.get("tp2"),
        tp2_probability=et.get("tp2_probability"),
        tp3=et.get("tp3"),
        tp3_probability=et.get("tp3_probability"),
        risk_reward_ratio=et.get("risk_reward_ratio"),
        gain_pct_to_tp1=et.get("gain_pct_to_tp1"),
        position_size_pct=pos_size_pct,
        position_size_kwd=pos_size_kwd,
        liquidity_capped=liq_capped,
        requires_confirmation=req_confirm,
        signals=signals,
        computed_at=analysis.get("computed_at"),
        days_of_history=analysis.get("days_of_history"),
    )
    return StockAnalysisResponse(status="ok", data=data)


# ---------------------------------------------------------------------------
# GET /eagle-eye/stocks/{ticker}/dna
# ---------------------------------------------------------------------------

@router.get("/stocks/{ticker}/dna", summary="Behavioral DNA")
async def get_stock_dna(
    ticker: str,
    _user: TokenData = Depends(get_current_user),
):
    """
    Return the behavioral DNA for a Kuwait ticker — historical success rates,
    signal reliability profiles, and dominant setup pattern.

    Returns HTTP 200 with status="pending" when the DNA pipeline has not yet
    finished computing this ticker. The client should display a friendly
    "Computing..." state rather than an error.
    """
    from app.services.eagle_eye.store import load_dna

    t = ticker.upper().strip()
    cache_key = f"dna:{t}"

    try:
        return await _get_stock_dna_inner(t, cache_key, load_dna)
    except Exception as exc:
        logger.exception("DNA endpoint crashed for %s — evicting cache and returning error", t)
        # Evict potentially corrupt in-memory entry so the next request retries cleanly
        _DNA_CACHE.pop(cache_key, None)
        return JSONResponse(
            status_code=200,
            content={
                "status": "error",
                "message": f"Failed to build DNA response for {t}: {exc}",
                "ticker": t,
            },
        )


async def _get_stock_dna_inner(t: str, cache_key: str, load_dna):
    """Inner implementation of get_stock_dna — separated so the outer function can catch all exceptions."""
    # 1. Fast path — in-memory cache
    if cache_key in _DNA_CACHE:
        dna_dict = _DNA_CACHE[cache_key]
        try:
            return DNAResponse(status="ok", data=BehavioralDNAResponse(**dna_dict))
        except Exception:
            # Cached entry is stale/broken — fall through to rebuild
            logger.warning("Corrupt DNA cache entry for %s; rebuilding", t)
            _DNA_CACHE.pop(cache_key, None)

    # 2. Check the DB store (written by the nightly Phase-2 pipeline)
    try:
        stored = load_dna(t)
    except Exception as exc:
        logger.warning("load_dna failed for %s: %s", t, exc)
        stored = None

    needs_refresh = bool(
        stored
        and (
            "history_status" not in stored
            or "setup_signals" not in stored
            or "setup_horizon_days" not in stored
            or "signal_stats" not in stored
            or "window_profiles" not in stored
            or "setup_examples" not in stored
            or "default_window_days" not in stored
            # New per-stock ML fields (added with multi-horizon upgrade).
            # Force a DNA rebuild for any stored entry that pre-dates this.
            or "optimal_hold_window_days" not in stored
        )
    )

    # Also force a rebuild when stored setup-example bars have null open values —
    # this indicates the DNA was persisted before OHLC data was sourced from
    # TickerChart (the authoritative price feed).
    if stored and not needs_refresh:
        examples = stored.get("setup_examples") or []
        for ex in examples:
            bars = ex.get("bars") or []
            if bars and any(b.get("open") is None for b in bars):
                logger.info(
                    "DNA for %s has null open values in setup bars — forcing rebuild from TickerChart", t
                )
                needs_refresh = True
                break

    if stored is None or needs_refresh:
        build_started = _trigger_dna_build(t, cache_key)
        if stored is None:
            message = (
                f"Computing Behavioral DNA for {t}. Check back shortly."
                if build_started
                else f"Behavioral DNA for {t} is still computing. Check back shortly."
            )
            return JSONResponse(
                status_code=200,
                content={
                    "status": "pending",
                    "message": message,
                    "ticker": t,
                },
            )
        logger.info("Serving stored DNA for %s while background refresh runs", t)

    if stored is None:
        return JSONResponse(
            status_code=200,
            content={
                "status": "unavailable",
                "message": (
                    "Insufficient price history to compute Behavioral DNA "
                    "for this stock."
                ),
                "ticker": t,
            },
        )

    # 3. Build response from stored DNA dict
    profiles: List[ThresholdProfileResponse] = [
        _build_threshold_profile_response(tp, stored.get("total_events_studied", 0))
        for tp in stored.get("profiles_by_threshold", [])
    ]

    signal_stats: List[SignalReliabilityResponse] = []
    setup_signal_stats = stored.get("signal_stats", [])
    if setup_signal_stats and isinstance(setup_signal_stats[0], dict):
        signal_stats = [
            SignalReliabilityResponse(
                signal=s.get("signal", ""),
                reliability_pct=s.get("reliability_pct"),
                presence_pct=s.get("presence_pct", s.get("reliability_pct")),
                fired_count=s.get("fired_count", 0),
                total_events=s.get("total_events"),
                total_setups=s.get("total_setups", stored.get("total_events_studied", 0)),
                avg_lead_days=s.get("avg_lead_days"),
                false_positive_rate=s.get("false_positive_rate"),
                discriminative_power=s.get("discriminative_power"),
            )
            for s in setup_signal_stats
            if s.get("signal")
        ]
    most_reliable = stored.get("most_reliable_signals_overall", [])
    if not signal_stats and most_reliable and isinstance(most_reliable[0], dict):
        signal_stats = [
            SignalReliabilityResponse(
                signal=s.get("signal", ""),
                reliability_pct=s.get("reliability_pct", 0),
                presence_pct=s.get("reliability_pct", 0),
                fired_count=s.get("fired_count", 0),
                total_events=s.get("total_events", stored.get("total_events_studied", 0)),
                total_setups=s.get("total_events", stored.get("total_events_studied", 0)),
                avg_lead_days=s.get("avg_lead_days"),
                false_positive_rate=s.get("false_positive_rate"),
                discriminative_power=s.get("discriminative_power"),
            )
            for s in most_reliable
            if s.get("signal")
        ]

    if most_reliable and isinstance(most_reliable[0], dict):
        most_reliable = [s.signal for s in signal_stats]

    window_profiles = [
        _build_window_profile_response(window_profile)
        for window_profile in stored.get("window_profiles", [])
    ]
    if not window_profiles and stored.get("setup_horizon_days"):
        window_profiles = [
            DNAWindowProfileResponse(
                horizon_days=stored.get("setup_horizon_days", 0),
                setup_count=stored.get("total_events_studied", 0),
                history_status=stored.get("history_status", "ok"),
                confidence_floor=stored.get("confidence_floor", 20),
                confidence_tier="ESTABLISHED" if stored.get("history_status") != "INSUFFICIENT_HISTORY" else "TOO_THIN",
                confidence_label="Established" if stored.get("history_status") != "INSUFFICIENT_HISTORY" else "Too thin",
                percentages_visible=stored.get("history_status") != "INSUFFICIENT_HISTORY",
                threshold_profiles=profiles,
            )
        ]

    setup_examples = []
    for raw_ex in stored.get("setup_examples", []):
        try:
            setup_examples.append(_build_setup_example_response(raw_ex))
        except Exception as ex_err:
            logger.warning("Skipping malformed setup example (%s): %s", t, ex_err)

    dna_response = BehavioralDNAResponse(
        ticker=t,
        total_events_analyzed=stored.get("total_events_studied", 0),
        history_status=stored.get("history_status", "ok"),
        setup_signals=stored.get("setup_signals", []),
        setup_horizon_days=stored.get("setup_horizon_days"),
        default_window_days=stored.get("default_window_days", stored.get("setup_horizon_days")),
        available_window_days=stored.get("available_window_days", [stored.get("setup_horizon_days")] if stored.get("setup_horizon_days") else []),
        confidence_floor=stored.get("confidence_floor", 5),
        most_reliable_signals=most_reliable[:10],
        signal_stats=signal_stats[:10],
        threshold_profiles=profiles,
        window_profiles=window_profiles,
        setup_examples=setup_examples,
        dominant_pattern=stored.get("dominant_pattern"),
        computed_at=stored.get("computed_at", datetime.utcnow().date().isoformat()),
    )
    _DNA_CACHE[cache_key] = dna_response.model_dump()
    return DNAResponse(status="ok", data=dna_response)


# ---------------------------------------------------------------------------
# GET /eagle-eye/stocks/{ticker}/events
# ---------------------------------------------------------------------------

@router.get("/stocks/{ticker}/events", response_model=EventsListResponse, summary="Historical move events")
async def get_stock_events(
    ticker: str,
    _user: TokenData = Depends(get_current_user),
):
    """
    Return the list of historically detected move events (breakouts, breakdowns,
    reversals, fakeouts) for the given Kuwait ticker.
    """
    t = ticker.upper().strip()
    cache_key = f"events:{t}"
    if cache_key in _EVENTS_CACHE:
        ev_list = _EVENTS_CACHE[cache_key]
        return EventsListResponse(status="ok", ticker=t, count=len(ev_list), events=ev_list)

    try:
        from app.services.eagle_eye.adapter import TickerChartAdapter
        from app.services.eagle_eye.move_detector import detect_fakeouts, detect_moves

        adapter = TickerChartAdapter()
        end_d = date.today()
        start_d = end_d - timedelta(days=_LOOKBACK_YEARS * 365 + 60)
        df = adapter.get_ohlcv_daily(t, start_d, end_d)
        if df is None or len(df) < 30:
            raise HTTPException(status_code=404, detail=f"Insufficient data for ticker '{t}'")

        moves = detect_moves(df)
        fakeouts = detect_fakeouts(df, moves)

        ev_list: List[MoveEventResponse] = []
        for e in moves:
            ev_list.append(MoveEventResponse(
                date=str(getattr(e, "date", "unknown")),
                event_type=getattr(e, "event_type", "move"),
                magnitude_pct=float(getattr(e, "magnitude_pct", 0.0)),
                duration_bars=int(getattr(e, "duration_bars", 1)),
                volume_confirmation=bool(getattr(e, "volume_confirmation", False)),
                description=getattr(e, "description", None),
            ))
        for e in fakeouts:
            ev_list.append(MoveEventResponse(
                date=str(getattr(e, "date", "unknown")),
                event_type="fakeout",
                magnitude_pct=float(getattr(e, "magnitude_pct", 0.0)),
                duration_bars=int(getattr(e, "duration_bars", 1)),
                volume_confirmation=False,
                description=getattr(e, "description", None),
            ))

        ev_list.sort(key=lambda x: x.date, reverse=True)
        _EVENTS_CACHE[cache_key] = [e.model_dump() for e in ev_list]
        return EventsListResponse(status="ok", ticker=t, count=len(ev_list), events=ev_list)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Events detection failed for %s", t)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /eagle-eye/refresh
# ---------------------------------------------------------------------------

@router.post("/refresh", response_model=RefreshResponse, summary="Queue background recompute")
async def refresh_stocks(
    body: RefreshRequest,
    _user: TokenData = Depends(get_current_user),
):
    """
    Invalidate the in-memory cache for the specified tickers and queue a
    background recompute so ee_ratings_cache is refreshed.

    Returns a job_id and estimated_minutes (0.5 min per ticker as a rough guide).
    """
    invalidated = 0
    for ticker in body.tickers:
        t = ticker.upper().strip()
        key = _cache_key(t)
        dna_key = f"dna:{t}"
        events_key = f"events:{t}"
        for k in (key, dna_key, events_key):
            if k in _cache:
                del _cache[k]
                invalidated += 1
            if dna_key in _DNA_CACHE:
                del _DNA_CACHE[dna_key]
            if events_key in _EVENTS_CACHE:
                del _EVENTS_CACHE[events_key]

    # Spawn a background thread to re-run the full nightly pipeline so
    # ee_ratings_cache is refreshed without blocking this response.
    try:
        _trigger_eagle_eye_recompute("manual_refresh", force=True)
        logger.info("Eagle Eye background recompute requested for %d ticker(s)", len(body.tickers))
    except Exception as exc:
        logger.warning("Could not start Eagle Eye background recompute: %s", exc)

    job_id = str(uuid.uuid4())
    return RefreshResponse(
        status="ok",
        job_id=job_id,
        tickers_queued=len(body.tickers),
        estimated_minutes=round(len(body.tickers) * 0.5, 1) or 0.5,
    )


# ===========================================================================
# SIMULATOR ENDPOINTS
# ===========================================================================

def _sim_portfolio_summary(portfolio: dict) -> dict:
    """Compute aggregate metrics for one simulator portfolio."""
    from app.core.database import query_all, query_val

    pid = portfolio["id"]

    closed_rows = query_all(
        """SELECT pnl_pct, pnl_kwd, exit_reason, entry_stage, entry_confidence
           FROM simulator_positions
           WHERE portfolio_id = ? AND status IN ('CLOSED', 'OVERRIDDEN')""",
        (pid,),
    )
    closed = [dict(r.items()) for r in closed_rows] if closed_rows else []

    open_rows = query_all(
        "SELECT id FROM simulator_positions WHERE portfolio_id = ? AND status = 'OPEN'",
        (pid,),
    )
    open_count = len(open_rows) if open_rows else 0

    wins = [r for r in closed if float(r.get("pnl_pct") or 0) > 0]
    losses = [r for r in closed if float(r.get("pnl_pct") or 0) <= 0]
    win_rate = (len(wins) / len(closed) * 100) if closed else 0

    avg_win = (sum(float(r.get("pnl_pct") or 0) for r in wins) / len(wins)) if wins else 0
    avg_loss = (sum(abs(float(r.get("pnl_pct") or 0)) for r in losses) / len(losses)) if losses else 0
    profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if (avg_loss * len(losses)) > 0 else 0

    # Max drawdown from snapshots
    snap_rows = query_all(
        "SELECT drawdown_from_peak_pct FROM simulator_daily_snapshots WHERE portfolio_id = ?",
        (pid,),
    )
    drawdowns = [float(dict(r.items()).get("drawdown_from_peak_pct") or 0) for r in snap_rows] if snap_rows else [0]
    max_drawdown = min(drawdowns)

    # Equity curve (last 30 snapshots)
    equity_rows = query_all(
        """SELECT date, total_value_kwd, cumulative_return_pct
           FROM simulator_daily_snapshots
           WHERE portfolio_id = ?
           ORDER BY date DESC LIMIT 30""",
        (pid,),
    )
    equity_curve = [
        {"date": dict(r.items())["date"], "value": float(dict(r.items()).get("total_value_kwd") or 0),
         "return_pct": float(dict(r.items()).get("cumulative_return_pct") or 0)}
        for r in (equity_rows or [])
    ]
    equity_curve.reverse()

    # Cumulative return
    starting = float(portfolio.get("starting_capital_kwd") or 10000)
    total = float(portfolio.get("total_value_kwd") or starting)
    cumulative_return_pct = ((total - starting) / starting * 100) if starting > 0 else 0

    # live_since: date when portfolio was last reset (updated_at from DB)
    raw_live_since = portfolio.get("updated_at") or portfolio.get("created_at")
    if raw_live_since:
        live_since = str(raw_live_since).split(" ")[0].split("T")[0]
    else:
        live_since = None

    return {
        "id": pid,
        "strategy_name": portfolio.get("strategy_name"),
        "starting_capital_kwd": starting,
        "cash_balance_kwd": float(portfolio.get("cash_balance_kwd") or 0),
        "total_value_kwd": total,
        "cumulative_return_pct": round(cumulative_return_pct, 2),
        "open_positions_count": open_count,
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "equity_curve": equity_curve,
        "live_since": live_since,
    }


def _get_all_sim_portfolios() -> list:
    from app.core.database import query_all
    rows = query_all("SELECT * FROM simulator_portfolios ORDER BY id", ())
    return [dict(r.items()) for r in rows] if rows else []


def _get_sim_portfolio_by_strategy(strategy_name: str) -> Optional[dict]:
    from app.core.database import query_one
    row = query_one(
        "SELECT * FROM simulator_portfolios WHERE strategy_name = ?",
        (strategy_name.upper(),),
    )
    return dict(row.items()) if row else None


# ---------------------------------------------------------------------------
# GET /eagle-eye/simulator/portfolios
# ---------------------------------------------------------------------------

@router.get("/simulator/portfolios", summary="All 3 simulator portfolios overview")
async def get_simulator_portfolios(_user: TokenData = Depends(get_current_user)):
    portfolios = _get_all_sim_portfolios()
    summaries = [_sim_portfolio_summary(p) for p in portfolios]
    return {"status": "ok", "portfolios": summaries}


# ---------------------------------------------------------------------------
# GET /eagle-eye/simulator/compare
# ---------------------------------------------------------------------------

@router.get("/simulator/compare", summary="Side-by-side strategy comparison")
async def get_simulator_compare(_user: TokenData = Depends(get_current_user)):
    portfolios = _get_all_sim_portfolios()
    summaries = {p["strategy_name"]: _sim_portfolio_summary(p) for p in portfolios}
    return {"status": "ok", "strategies": summaries}


# ---------------------------------------------------------------------------
# GET /eagle-eye/simulator/portfolios/{strategy_name}
# ---------------------------------------------------------------------------

@router.get("/simulator/portfolios/{strategy_name}", summary="Full strategy detail")
async def get_simulator_portfolio_detail(
    strategy_name: str,
    _user: TokenData = Depends(get_current_user),
):
    from app.core.database import query_all

    portfolio = _get_sim_portfolio_by_strategy(strategy_name)
    if portfolio is None:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")

    summary = _sim_portfolio_summary(portfolio)
    pid = portfolio["id"]

    # Full equity curve
    eq_rows = query_all(
        """SELECT date, cash_balance_kwd, open_positions_value_kwd,
                  total_value_kwd, daily_pnl_kwd, cumulative_return_pct,
                  drawdown_from_peak_pct, open_position_count
           FROM simulator_daily_snapshots
           WHERE portfolio_id = ?
           ORDER BY date ASC""",
        (pid,),
    )
    equity_curve = [dict(r.items()) for r in eq_rows] if eq_rows else []

    # Open positions
    open_rows = query_all(
        """SELECT id, ticker, entry_date, entry_price, shares, shares_remaining,
                  size_kwd, entry_confidence, entry_stage, entry_rating, entry_thesis,
                  planned_stop_loss, planned_tp1, planned_tp2, planned_tp3,
                  max_unrealized_gain_pct, max_unrealized_loss_pct, created_at
           FROM simulator_positions
           WHERE portfolio_id = ? AND status = 'OPEN'
           ORDER BY entry_date DESC""",
        (pid,),
    )
    open_positions = [dict(r.items()) for r in open_rows] if open_rows else []

    # Recent closed trades
    closed_rows = query_all(
        """SELECT id, ticker, entry_date, entry_price, exit_date, exit_price,
                  exit_reason, pnl_kwd, pnl_pct, days_held,
                  entry_confidence, entry_stage, entry_rating
           FROM simulator_positions
           WHERE portfolio_id = ? AND status IN ('CLOSED', 'OVERRIDDEN')
           ORDER BY exit_date DESC LIMIT 50""",
        (pid,),
    )
    closed_trades = [dict(r.items()) for r in closed_rows] if closed_rows else []

    # Considered-not-taken count
    considered_count_row = query_all(
        "SELECT COUNT(*) as cnt FROM simulator_considered_trades WHERE portfolio_id = ?",
        (pid,),
    )
    considered_count = int(dict(considered_count_row[0].items()).get("cnt") or 0) if considered_count_row else 0

    # Breakdown by stage at entry
    stage_rows = query_all(
        """SELECT entry_stage, COUNT(*) as trades,
                  AVG(pnl_pct) as avg_pnl, SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins
           FROM simulator_positions
           WHERE portfolio_id = ? AND status IN ('CLOSED', 'OVERRIDDEN')
           GROUP BY entry_stage""",
        (pid,),
    )
    by_stage = [dict(r.items()) for r in stage_rows] if stage_rows else []

    # Breakdown by exit reason
    reason_rows = query_all(
        """SELECT exit_reason, COUNT(*) as cnt, AVG(pnl_pct) as avg_pnl
           FROM simulator_positions
           WHERE portfolio_id = ? AND status IN ('CLOSED', 'OVERRIDDEN')
           GROUP BY exit_reason""",
        (pid,),
    )
    by_exit_reason = [dict(r.items()) for r in reason_rows] if reason_rows else []

    return {
        "status": "ok",
        "summary": summary,
        "equity_curve": equity_curve,
        "open_positions": open_positions,
        "recent_closed_trades": closed_trades,
        "considered_not_taken_count": considered_count,
        "breakdown_by_stage": by_stage,
        "breakdown_by_exit_reason": by_exit_reason,
    }


# ---------------------------------------------------------------------------
# GET /eagle-eye/simulator/portfolios/{strategy_name}/trades
# ---------------------------------------------------------------------------

@router.get("/simulator/portfolios/{strategy_name}/trades", summary="All trades (paginated)")
async def get_simulator_trades(
    strategy_name: str,
    status: Optional[str] = Query(None, description="OPEN | CLOSED | OVERRIDDEN"),
    ticker: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    _user: TokenData = Depends(get_current_user),
):
    from app.core.database import query_all

    portfolio = _get_sim_portfolio_by_strategy(strategy_name)
    if portfolio is None:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")

    pid = portfolio["id"]
    offset = (page - 1) * page_size
    conditions = ["portfolio_id = ?"]
    params: list = [pid]

    if status:
        conditions.append("status = ?")
        params.append(status.upper())
    if ticker:
        conditions.append("ticker = ?")
        params.append(ticker.upper())

    where = " AND ".join(conditions)
    rows = query_all(
        f"""SELECT * FROM simulator_positions WHERE {where}
            ORDER BY COALESCE(exit_date, entry_date) DESC
            LIMIT ? OFFSET ?""",
        tuple(params) + (page_size, offset),
    )
    trades = []
    for r in (rows or []):
        d = dict(r.items())
        for json_col in ("entry_signal_breakdown", "entry_indicators_snapshot"):
            if d.get(json_col) and isinstance(d[json_col], str):
                try:
                    d[json_col] = json.loads(d[json_col])
                except Exception:
                    d[json_col] = {}
        trades.append(d)

    count_row = query_all(f"SELECT COUNT(*) as cnt FROM simulator_positions WHERE {where}", tuple(params))
    total = int(dict(count_row[0].items()).get("cnt") or 0) if count_row else 0

    return {"status": "ok", "total": total, "page": page, "page_size": page_size, "trades": trades}


# ---------------------------------------------------------------------------
# GET /eagle-eye/simulator/portfolios/{strategy_name}/performance
# ---------------------------------------------------------------------------

@router.get("/simulator/portfolios/{strategy_name}/performance", summary="Aggregate analytics")
async def get_simulator_performance(
    strategy_name: str,
    _user: TokenData = Depends(get_current_user),
):
    from app.core.database import query_all
    import math as _math

    portfolio = _get_sim_portfolio_by_strategy(strategy_name)
    if portfolio is None:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")

    pid = portfolio["id"]
    closed_rows = query_all(
        """SELECT pnl_pct, pnl_kwd, days_held, entry_stage, entry_confidence,
                  entry_rating, exit_reason, ticker
           FROM simulator_positions
           WHERE portfolio_id = ? AND status IN ('CLOSED', 'OVERRIDDEN')""",
        (pid,),
    )
    closed = [dict(r.items()) for r in closed_rows] if closed_rows else []

    wins = [r for r in closed if float(r.get("pnl_pct") or 0) > 0]
    losses = [r for r in closed if float(r.get("pnl_pct") or 0) <= 0]

    # Sharpe-like ratio using daily snapshots
    daily_rows = query_all(
        "SELECT daily_pnl_kwd FROM simulator_daily_snapshots WHERE portfolio_id = ? ORDER BY date",
        (pid,),
    )
    daily_returns = [float(dict(r.items()).get("daily_pnl_kwd") or 0) for r in (daily_rows or [])]
    starting = float(portfolio.get("starting_capital_kwd") or 10000)
    daily_pct = [r / starting * 100 for r in daily_returns]
    if len(daily_pct) > 1:
        mean_r = sum(daily_pct) / len(daily_pct)
        variance = sum((x - mean_r) ** 2 for x in daily_pct) / len(daily_pct)
        std_r = _math.sqrt(variance)
        sharpe = (mean_r / std_r * _math.sqrt(252)) if std_r > 0 else 0
    else:
        sharpe = 0

    # By confidence band
    bands = [(55, 65), (65, 75), (75, 85), (85, 100)]
    by_confidence = []
    for lo, hi in bands:
        band_trades = [r for r in closed if lo <= float(r.get("entry_confidence") or 0) < hi]
        band_wins = [r for r in band_trades if float(r.get("pnl_pct") or 0) > 0]
        by_confidence.append({
            "band": f"{lo}-{hi}",
            "trades": len(band_trades),
            "wins": len(band_wins),
            "win_rate": round(len(band_wins) / len(band_trades) * 100, 1) if band_trades else 0,
            "avg_pnl_pct": round(sum(float(r.get("pnl_pct") or 0) for r in band_trades) / len(band_trades), 2) if band_trades else 0,
        })

    # By stage
    stage_map: dict = {}
    for r in closed:
        s = r.get("entry_stage") or "UNKNOWN"
        if s not in stage_map:
            stage_map[s] = {"stage": s, "trades": 0, "wins": 0, "total_pnl": 0}
        stage_map[s]["trades"] += 1
        pnl = float(r.get("pnl_pct") or 0)
        if pnl > 0:
            stage_map[s]["wins"] += 1
        stage_map[s]["total_pnl"] += pnl
    by_stage = [
        {**v, "win_rate": round(v["wins"] / v["trades"] * 100, 1) if v["trades"] else 0,
         "avg_pnl_pct": round(v["total_pnl"] / v["trades"], 2) if v["trades"] else 0}
        for v in stage_map.values()
    ]

    # By exit reason
    reason_map: dict = {}
    for r in closed:
        reason = r.get("exit_reason") or "UNKNOWN"
        if reason not in reason_map:
            reason_map[reason] = {"exit_reason": reason, "count": 0, "avg_pnl": 0, "total_pnl": 0}
        reason_map[reason]["count"] += 1
        reason_map[reason]["total_pnl"] += float(r.get("pnl_pct") or 0)
    for v in reason_map.values():
        v["avg_pnl"] = round(v["total_pnl"] / v["count"], 2) if v["count"] else 0
    by_exit_reason = list(reason_map.values())

    avg_duration = (sum(int(r.get("days_held") or 0) for r in closed) / len(closed)) if closed else 0

    return {
        "status": "ok",
        "strategy_name": strategy_name.upper(),
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 2) if closed else 0,
        "avg_win_pct": round(sum(float(r.get("pnl_pct") or 0) for r in wins) / len(wins), 2) if wins else 0,
        "avg_loss_pct": round(sum(abs(float(r.get("pnl_pct") or 0)) for r in losses) / len(losses), 2) if losses else 0,
        "avg_trade_duration_days": round(avg_duration, 1),
        "sharpe_like_ratio": round(sharpe, 2),
        "by_confidence_band": by_confidence,
        "by_stage": by_stage,
        "by_exit_reason": by_exit_reason,
    }


# ---------------------------------------------------------------------------
# POST /eagle-eye/simulator/positions/{position_id}/close
# ---------------------------------------------------------------------------

@router.post("/simulator/positions/{position_id}/close", summary="Manual override close")
async def close_simulator_position(
    position_id: int,
    body: dict,
    user: TokenData = Depends(get_current_user),
):
    """Close an open simulator position at the provided price (manual override)."""
    current_price = body.get("current_price")
    if current_price is None or float(current_price) <= 0:
        raise HTTPException(status_code=422, detail="current_price must be a positive number")

    try:
        from app.services.eagle_eye.simulator import get_engine
        result = get_engine().manual_override_close(position_id, float(current_price))
        return {"status": "ok", **result}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Simulator manual close failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# GET /eagle-eye/simulator/activity   — recent feed across all 3 strategies
# ---------------------------------------------------------------------------

@router.get("/simulator/activity", summary="Recent activity across all strategies")
async def get_simulator_activity(
    limit: int = Query(20, ge=1, le=100),
    _user: TokenData = Depends(get_current_user),
):
    from app.core.database import query_all

    rows = query_all(
        """SELECT sp.strategy_name, pos.ticker, pos.status, pos.entry_date,
                  pos.exit_date, pos.exit_reason, pos.pnl_kwd, pos.pnl_pct,
                  pos.entry_stage
           FROM simulator_positions pos
           JOIN simulator_portfolios sp ON sp.id = pos.portfolio_id
           WHERE pos.status IN ('CLOSED', 'OVERRIDDEN')
           ORDER BY pos.exit_date DESC
           LIMIT ?""",
        (limit,),
    )
    entries = query_all(
        """SELECT sp.strategy_name, pos.ticker, 'ENTERED' as action,
                  pos.entry_date as event_date, pos.size_kwd, pos.entry_stage, pos.entry_confidence
           FROM simulator_positions pos
           JOIN simulator_portfolios sp ON sp.id = pos.portfolio_id
           ORDER BY pos.entry_date DESC
           LIMIT ?""",
        (limit,),
    )

    exits = [{"action": "EXIT", **dict(r.items())} for r in (rows or [])]
    opens = [{"action": "ENTRY", **dict(r.items())} for r in (entries or [])]
    feed = sorted(exits + opens, key=lambda x: x.get("exit_date") or x.get("event_date") or "", reverse=True)[:limit]

    return {"status": "ok", "feed": feed}


# ---------------------------------------------------------------------------
# POST /eagle-eye/simulator/reset   — clear stale simulator state
# ---------------------------------------------------------------------------

@router.post("/simulator/reset", summary="Reset simulator data and restart from today")
async def reset_simulator_now(
    run_after_reset: bool = Query(False, description="Run one simulation cycle after reset"),
    _admin: TokenData = Depends(require_admin),
):
    try:
        from app.services.eagle_eye.simulator import get_engine

        engine = get_engine()
        result = engine.reset_all()
        if run_after_reset:
            result["run_result"] = engine.run_daily()
        return {"status": "ok", "result": result}
    except Exception as exc:
        logger.exception("Simulator reset failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /eagle-eye/simulator/run   — manual trigger (admin / testing)
# ---------------------------------------------------------------------------

@router.post("/simulator/run", summary="Manually trigger simulator daily run")
async def run_simulator_now(
    user: TokenData = Depends(get_current_user),
):
    try:
        from app.services.eagle_eye.simulator import get_engine
        result = get_engine().run_daily()
        return {"status": "ok", "result": result}
    except Exception as exc:
        logger.exception("Manual simulator run failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    job_id = str(uuid.uuid4())
    est_minutes = round(len(body.tickers) * 0.5, 1)
    return RefreshResponse(
        status="ok",
        job_id=job_id,
        tickers_queued=len(body.tickers),
        estimated_minutes=est_minutes,
    )


# ---------------------------------------------------------------------------
# GET /eagle-eye/regime
# ---------------------------------------------------------------------------

@router.get("/regime", response_model=RegimeResponse, summary="Current market regime")
async def get_market_regime(
    _user: TokenData = Depends(get_current_user),
):
    """
    Return the current macro regime classification for the Kuwait market.

    Regime is derived from:
      - Breadth (% of KSE stocks above 50-day MA)
      - Oil price trend (Brent crude proxy)

    Falls back to NEUTRAL when data is unavailable.
    """
    global _REGIME_RESP_CACHE, _REGIME_RESP_CACHE_AT

    now = time.time()
    if (
        _REGIME_RESP_CACHE is not None
        and (now - _REGIME_RESP_CACHE_AT) < _REGIME_RESP_TTL_SEC
    ):
        return RegimeResponse(**_REGIME_RESP_CACHE)

    def _compute_regime_response() -> RegimeResponse:
        import json

        from app.core.database import query_all
        from app.services.eagle_eye.adapter import TickerChartAdapter
        from app.services.eagle_eye.store import load_ohlcv

        adapter = TickerChartAdapter()
        end_d = date.today()
        start_d = end_d - timedelta(days=120)

        stocks_meta = adapter.list_stocks()
        if not stocks_meta:
            return RegimeResponse(
                status="ok",
                regime="NEUTRAL",
                last_updated=datetime.utcnow().date().isoformat(),
            )

        sample_tickers = [meta.ticker.upper() for meta in stocks_meta[:30]]
        indicators_map: dict[str, dict] = {}
        if sample_tickers:
            placeholders = ",".join("?" for _ in sample_tickers)
            rows = query_all(
                f"SELECT ticker, indicators_json FROM ee_ratings_cache WHERE ticker IN ({placeholders})",
                tuple(sample_tickers),
            )
            for row in rows:
                raw = row.get("indicators_json")
                if not raw:
                    continue
                try:
                    indicators_map[str(row.get("ticker") or "").upper()] = json.loads(raw)
                except Exception:
                    continue

        above_50ma_count = 0
        checked = 0
        for meta in stocks_meta[:30]:  # sample 30 stocks for breadth
            try:
                latest = indicators_map.get(meta.ticker.upper()) or {}
                ema50 = latest.get("ema50") if isinstance(latest, dict) else None
                close = latest.get("close") if isinstance(latest, dict) else None

                if ema50 is None or close is None:
                    from app.services.eagle_eye.indicators import compute_all_indicators

                    df = load_ohlcv(meta.ticker, start=start_d, end=end_d)
                    if df is None or len(df) < 52:
                        continue
                    ind_df = compute_all_indicators(df)
                    latest = ind_df.iloc[-1]
                    ema50 = latest.get("ema50") if isinstance(latest, dict) else getattr(latest, "ema50", None)
                    close = latest.get("close") if isinstance(latest, dict) else getattr(latest, "close", None)

                if ema50 and close and close > ema50:
                    above_50ma_count += 1
                checked += 1
            except Exception:
                continue

        breadth_pct = round(above_50ma_count / max(checked, 1) * 100, 1) if checked else 50.0

        if breadth_pct >= 60:
            regime = "RISK_ON"
        elif breadth_pct <= 35:
            regime = "RISK_OFF"
        else:
            regime = "NEUTRAL"

        return RegimeResponse(
            status="ok",
            regime=regime,
            breadth_pct_above_50ma=breadth_pct,
            brent_trend="neutral",
            pmi_trend="neutral",
            last_updated=datetime.utcnow().date().isoformat(),
        )

    try:
        response = await asyncio.to_thread(_compute_regime_response)
        _REGIME_RESP_CACHE = response.model_dump()
        _REGIME_RESP_CACHE_AT = now
        return response

    except Exception as exc:
        logger.warning("Regime calculation failed: %s", exc)
        response = RegimeResponse(
            status="ok",
            regime="NEUTRAL",
            last_updated=datetime.utcnow().date().isoformat(),
        )
        _REGIME_RESP_CACHE = response.model_dump()
        _REGIME_RESP_CACHE_AT = now
        return response


# ---------------------------------------------------------------------------
# Addendum A.1 — ML eligibility summary for frontend Settings page
# ---------------------------------------------------------------------------

@router.get(
    "/ml/eligibility-summary",
    summary="ML eligibility coverage summary (Settings page)",
)
async def get_ml_eligibility_summary(
    _user: TokenData = Depends(get_current_user),
):
    """
    Return a compact summary of how many stocks are ML-eligible, rules-only,
    and watch-only.  Used by the frontend Settings page.

    Example response::

        {
            "status": "ok",
            "total": 139,
            "ml_eligible": 62,
            "rules_only": 59,
            "watch_only": 18,
            "label": "62 of 139 stocks are ML-eligible. 59 are rules-only. 18 are watch-only."
        }
    """
    from app.services.eagle_eye.ml.eligibility_report import get_eligibility_summary_for_frontend

    counts = get_eligibility_summary_for_frontend()
    label = (
        f"{counts['ml_eligible']} of {counts['total']} stocks are ML-eligible. "
        f"{counts['rules_only']} are rules-only. "
        f"{counts['watch_only']} are watch-only."
    )
    return {"status": "ok", **counts, "label": label}


# ---------------------------------------------------------------------------
# Phase 3 — ML band display endpoints
# ---------------------------------------------------------------------------

def _resolve_ml_display_state() -> tuple[bool, bool, Optional[str], bool]:
    """Return (enabled, auto_disabled, disabled_reason, config_enabled).

    Includes a one-time self-heal for legacy false-disable rows where MCE was
    historically recorded on a 0-100 scale (e.g., 40.027) instead of 0-1.
    """
    from app.core.config import get_settings
    from app.core.database import exec_sql, query_one

    settings = get_settings()
    state = query_one(
        "SELECT auto_disabled, disabled_reason FROM ml_display_state WHERE id = 1",
        (),
    )
    auto_disabled = bool(state and state["auto_disabled"]) if state else False
    disabled_reason = state["disabled_reason"] if state else None

    if auto_disabled and disabled_reason:
        legacy_mce = _extract_mce_from_reason(disabled_reason)
        if legacy_mce is not None and legacy_mce > 1.0:
            exec_sql(
                """
                UPDATE ml_display_state
                   SET auto_disabled = 0,
                       disabled_at = NULL,
                       disabled_reason = NULL,
                       updated_at = CURRENT_TIMESTAMP
                 WHERE id = 1
                """,
                (),
            )
            logger.info(
                "Eagle Eye: cleared legacy ML auto-disable state (reason=%s)",
                disabled_reason,
            )
            auto_disabled = False
            disabled_reason = None

    config_enabled = settings.ENABLE_ML_DISPLAY
    enabled = config_enabled and not auto_disabled
    return enabled, auto_disabled, disabled_reason, config_enabled

@router.get("/ml/display-state", summary="ML display kill-switch state")
async def get_ml_display_state(
    _user: TokenData = Depends(get_current_user),
):
    """
    Return the current ML display state.

    Response::

        {
            "enabled": true,          // ENABLE_ML_DISPLAY setting
            "auto_disabled": false,   // auto-disable monitor flag
            "disabled_reason": null   // reason if auto_disabled=true
        }
    """
    enabled, auto_disabled, disabled_reason, config_enabled = _resolve_ml_display_state()
    return {
        "enabled": enabled,
        "config_enabled": config_enabled,
        "auto_disabled": auto_disabled,
        "disabled_reason": disabled_reason,
    }


@router.get("/ml/bands", summary="ML band scores for all SHADOW stocks")
async def get_ml_bands(
    _user: TokenData = Depends(get_current_user),
):
    """
    Return ML band labels for all 14 SHADOW-roster stocks.

    When ML display is disabled (kill-switch or auto-disable), all band values
    are returned as null and *enabled* is false.

    Response::

        {
            "enabled": true,
            "disclaimer": "⚠️ ML signal in evaluation...",
            "bands": [
                {"ticker": "AAYANRE", "band": "HIGH", "color": "#10B981", ...},
                ...
            ]
        }
    """
    from app.core.database import query_one
    from app.services.eagle_eye.ml.band_display import band_for_display, DISCLAIMER_TEXT
    from app.services.eagle_eye.ml.shadow_runner import SHADOW_ROSTER

    ml_enabled, _, _, _ = _resolve_ml_display_state()

    bands = []
    for ticker in SHADOW_ROSTER:
        if not ml_enabled:
            bands.append({
                "ticker": ticker,
                "band": None,
                "color": None,
                "emoji": None,
                "short_label": None,
            })
            continue

        row = query_one(
            """
            SELECT band_label, calibrated_prob, log_date
              FROM ml_shadow_log
             WHERE stock_ticker = ?
               AND band_label IS NOT NULL
             ORDER BY log_date DESC
             LIMIT 1
            """,
            (ticker,),
        )
        if row and row["band_label"]:
            display = band_for_display(row["band_label"])
            bands.append({
                "ticker": ticker,
                "band": row["band_label"],
                "color": display["color"],
                "emoji": display["emoji"],
                "short_label": display["short"],
                "as_of": row["log_date"],
                "calibrated_prob": row["calibrated_prob"],
            })
        else:
            bands.append({
                "ticker": ticker,
                "band": "INSUFFICIENT_DATA",
                "color": "#9CA3AF",
                "emoji": "—",
                "short_label": "N/A",
                "as_of": None,
                "calibrated_prob": None,
            })

    return {
        "enabled": ml_enabled,
        "disclaimer": DISCLAIMER_TEXT,
        "bands": bands,
    }


@router.post("/ml/display-state/re-enable", summary="Manually re-enable ML display (admin)")
async def re_enable_ml_display_state(
    _admin: TokenData = Depends(require_admin),
):
    """Admin-only override to clear ML auto-disable state immediately."""
    from app.services.eagle_eye.ml.auto_disable_monitor import re_enable_display

    re_enable_display()
    return {
        "status": "ok",
        "enabled": True,
        "config_enabled": True,
        "auto_disabled": False,
        "disabled_reason": None,
    }


@router.get("/ml/bands/{ticker}", summary="Full ML band card for one stock")
async def get_ml_band_for_ticker(
    ticker: str,
    _user: TokenData = Depends(get_current_user),
):
    """
    Return the full ML band card for a single ticker.

    Includes band label, thresholds, calibrated probability, a BORDERLINE
    verdict when the stock is within 5% of a band boundary, and the
    mandatory disclaimer text.

    Returns 404 if the ticker is not in the SHADOW roster.
    Returns null band fields if ML display is disabled.
    """
    from app.core.database import query_one
    from app.services.eagle_eye.ml.band_display import band_for_display, DISCLAIMER_TEXT
    from app.services.eagle_eye.ml.shadow_runner import SHADOW_ROSTER

    ticker = ticker.upper()
    if ticker not in SHADOW_ROSTER:
        raise HTTPException(status_code=404, detail=f"{ticker} is not in the ML SHADOW roster")

    ml_enabled, _, _, config_enabled = _resolve_ml_display_state()

    if not ml_enabled:
        return {
            "ticker": ticker,
            "enabled": False,
            "band": None,
            "disclaimer": DISCLAIMER_TEXT
            if not config_enabled
            else "⚠️ ML signals temporarily disabled.",
        }

    row = query_one(
        """
        SELECT band_label, calibrated_prob, raw_prob, log_date,
               band_low_threshold, band_high_threshold, rule_stage
          FROM ml_shadow_log
         WHERE stock_ticker = ?
           AND band_label IS NOT NULL
         ORDER BY log_date DESC
         LIMIT 1
        """,
        (ticker,),
    )

    if not row:
        return {
            "ticker": ticker,
            "enabled": True,
            "band": "INSUFFICIENT_DATA",
            "calibrated_prob": None,
            "as_of": None,
            "disclaimer": DISCLAIMER_TEXT,
            "verdict": None,
        }

    band_label = row["band_label"]
    display = band_for_display(band_label)
    cal_prob = row["calibrated_prob"]
    low_thr = row["band_low_threshold"]
    high_thr = row["band_high_threshold"]

    # BORDERLINE verdict: within 5 percentage points of a band boundary
    verdict = None
    if cal_prob is not None and low_thr is not None and high_thr is not None:
        distance_to_boundary = min(
            abs(float(cal_prob) - float(low_thr)),
            abs(float(cal_prob) - float(high_thr)),
        )
        if distance_to_boundary < 0.05:
            verdict = "BORDERLINE"

    return {
        "ticker": ticker,
        "enabled": True,
        "band": band_label,
        "color": display["color"],
        "emoji": display["emoji"],
        "calibrated_prob": cal_prob,
        "raw_prob": row["raw_prob"],
        "band_low_threshold": low_thr,
        "band_high_threshold": high_thr,
        "rule_stage": row["rule_stage"],
        "verdict": verdict,
        "as_of": row["log_date"],
        "disclaimer": DISCLAIMER_TEXT,
        "methodology_link": "/eagle-eye/methodology",
    }


@router.get("/ml/methodology", summary="ML methodology explanation")
async def get_ml_methodology(
    _user: TokenData = Depends(get_current_user),
):
    """
    Return human-readable methodology text for the ML band display.

    Used by the Methodology screen in the mobile app.
    """
    return {
        "title": "Eagle Eye ML Bands — Methodology",
        "phase": "Phase 3: Shadow Evaluation",
        "status": "Experimental — not for trading decisions",
        "disclaimer": "⚠️ ML signal in evaluation — do not use for trading decisions yet.",
        "sections": [
            {
                "heading": "What are ML bands?",
                "body": (
                    "Eagle Eye ML bands classify each SHADOW-roster stock as LOW, MEDIUM, or HIGH "
                    "based on a LightGBM binary classifier trained on historical breakout events. "
                    "The classifier outputs a calibrated probability of a ≥10% move within 20 trading days."
                ),
            },
            {
                "heading": "How are bands computed?",
                "body": (
                    "Each stock's calibrated probability is compared against the 33rd and 67th percentile "
                    "of its own last 90 days of shadow scores.  Stocks below the 33rd percentile are LOW, "
                    "between percentiles are MEDIUM, and above the 67th are HIGH.  "
                    "Fewer than 30 days of history returns INSUFFICIENT_DATA."
                ),
            },
            {
                "heading": "What does BORDERLINE mean?",
                "body": (
                    "A BORDERLINE verdict appears when a stock's probability is within 5 percentage "
                    "points of a band boundary.  This signals that the classification is less certain "
                    "and should be treated with extra caution."
                ),
            },
            {
                "heading": "Phase 3 constraints",
                "body": (
                    "No model will be promoted to LIVE status during Phase 3.  "
                    "Shadow scoring runs daily (Sun–Thu) after Boursa market close.  "
                    "Models are reviewed weekly.  Automatic disabling occurs if calibration "
                    "error exceeds 30%, multiple rollbacks occur, or scoring jobs fail repeatedly."
                ),
            },
            {
                "heading": "Covered stocks (14)",
                "body": "AAYANRE, ALTIJARIA, ARGAN, BOURSA, FACIL, IFA, JAZEERA, JTC, KCEM, KPPC, MKHZN, OOREDOO, URC, WARBACAP",
            },
        ],
    }
