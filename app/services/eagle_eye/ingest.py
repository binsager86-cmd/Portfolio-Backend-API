"""
Eagle Eye Phase 3 — Live Data Ingestion Pipeline.

Three sequential phases, each independently re-runnable:

  Phase 1 — ingest_all_ohlcv()
      Fetch 3 years of daily OHLCV for every KSE stock and store to
      ee_ohlcv_cache. Refresh policy is overlap-based: every run re-fetches
      and overwrites the trailing cached sessions to absorb late exchange
      corrections, while older history remains untouched.

  Phase 2 — build_all_dna()
      Run the forensic pipeline on the cached OHLCV to build
      BehavioralDNA profiles stored in ee_dna_profiles.
      Expensive (500+ bars × 70+ stocks) — intended for weekly runs.

  Phase 3 — compute_all_ratings()
      Rate every stock using indicators computed from ee_ohlcv_cache.
      Results are stored to ee_ratings_cache so the scanner endpoint
      returns instantly without hitting TickerChart.

  Orchestrator — run_nightly_recompute(dna_refresh=False)
      Phases 1 + 3 every trading day.
      Phases 1 + 2 + 3 on Sundays (when dna_refresh=True).

  init_schema()
      Call once at startup to create tables if missing.
"""
from __future__ import annotations

import logging
import math
import os
import time
import warnings
from datetime import date, datetime, timedelta
from typing import List, Optional
import uuid

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    _LOOKBACK_BARS = int(os.getenv("EE_RATINGS_LOOKBACK_BARS", "320"))
except (TypeError, ValueError):
    _LOOKBACK_BARS = 320

# Keep enough warmup for long-horizon rolling/EMA indicators while avoiding
# full-history recompute cost on every ticker.
RATINGS_INDICATOR_LOOKBACK_BARS = max(50, _LOOKBACK_BARS)


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _build_live_feature_vector(feature_row: dict, feature_names: List[str]) -> List[float]:
    vec: List[float] = []
    for name in feature_names:
        fv = _safe_float(feature_row.get(name))
        vec.append(float("nan") if fv is None else float(fv))
    return vec


def predict_confidence(ticker: str, ohlcv_df, as_of: date) -> Optional[float]:
    """Retired in Phase 1 rebuild (rules-primary pipeline)."""
    del ticker, ohlcv_df, as_of
    return None


def _predict_ml_signal(ticker: str, ohlcv_df, as_of: date):
    """Retired compatibility wrapper for legacy ML payloads."""
    del ticker, ohlcv_df, as_of
    return None


def _predict_ml_opportunity_score(ticker: str, ohlcv_df, as_of: date):
    """Retired compatibility wrapper for phase-regression output."""
    del ticker, ohlcv_df, as_of
    return None


# ---------------------------------------------------------------------------
# Schema initializer — safe to call at every startup
# ---------------------------------------------------------------------------

def init_schema() -> None:
    """Create Eagle Eye DB tables (OHLCV/DNA/ratings) and ML tables if they do not exist."""
    from app.services.eagle_eye.store import ensure_tables
    from app.services.eagle_eye.ml.db_tables import ensure_ml_tables
    from app.services.eagle_eye.ml.macro_features import write_data_gaps_report
    ensure_tables()
    ensure_ml_tables()
    write_data_gaps_report()


# ---------------------------------------------------------------------------
# Phase 1 — OHLCV ingestion
# ---------------------------------------------------------------------------

def ingest_all_ohlcv(verbose: bool = False) -> dict:
    """
    Fetch and cache 3 years of daily OHLCV for every stock returned by
    TickerChartAdapter.list_stocks().

    Refresh policy: always re-fetch a trailing overlap window per ticker and
    overwrite those bars in cache, so recent exchange revisions are captured.

    Returns a summary dict: {ok, skipped, errors, insufficient, gaps}.
    """
    from app.core.config import get_settings
    from app.services.eagle_eye.adapter import TickerChartAdapter
    from app.services.eagle_eye.store import (
        ensure_tables,
        get_latest_ohlcv_date,
        get_trailing_ohlcv_start_date,
        log_compute,
        save_ohlcv,
    )

    ensure_tables()
    settings = get_settings()
    if not (settings.TICKERCHART_USERNAME or "").strip() or not (settings.TICKERCHART_PASSWORD or "").strip():
        msg = (
            "TickerChart credentials are not configured; "
            "Eagle Eye OHLCV warmup cannot populate the scanner cache"
        )
        logger.error(msg)
        log_compute("ohlcv_fetch", None, "error", msg)
        return {"ok": 0, "skipped": 0, "errors": 1, "insufficient": [], "gaps": [], "error": msg}

    adapter = TickerChartAdapter()
    stocks = adapter.list_stocks()
    total_stocks = len(stocks)

    today = date.today()
    history_start = today - timedelta(days=3 * 365 + 60)  # 3 years + buffer
    trailing_refresh_sessions = 10
    phase_t0 = time.time()

    stats: dict = {"ok": 0, "skipped": 0, "errors": 0, "insufficient": [], "gaps": []}

    if verbose:
        print(f"[EagleEye] Ingesting OHLCV for {total_stocks} stocks ({history_start} -> {today})")
        print("=" * 70)

    for idx, stock in enumerate(stocks, start=1):
        ticker = stock.ticker
        try:
            last_date = get_latest_ohlcv_date(ticker)

            # Re-fetch trailing sessions every run so late corrections overwrite
            # stale bars in cache.
            if last_date:
                overlap_start = get_trailing_ohlcv_start_date(
                    ticker,
                    trailing_sessions=trailing_refresh_sessions,
                )
                if overlap_start is None:
                    overlap_start = last_date - timedelta(days=21)
                fetch_start = max(history_start, overlap_start)
            else:
                fetch_start = history_start

            if fetch_start > today:
                fetch_start = today

            if verbose:
                print(
                    f"  [{idx}/{total_stocks}] [{ticker}] fetching {fetch_start} -> {today} "
                    f"(refresh overlap {trailing_refresh_sessions} sessions) ...",
                    end=" ",
                    flush=True,
                )

            df = adapter.get_ohlcv_daily(ticker, fetch_start, today)

            if df is None or df.empty:
                if verbose:
                    print("no data")
                if last_date is None:
                    stats["insufficient"].append(ticker)
                    log_compute("ohlcv_fetch", ticker, "skip", "no data returned")
                else:
                    stats["skipped"] += 1
                continue

            # Gap detection (informational only — does not block storage)
            gaps = _detect_gaps(ticker, df)
            if gaps:
                stats["gaps"].extend(gaps)
                if verbose:
                    print(f"  [WARN] {len(gaps)} gap(s) detected", end=" ")

            n = save_ohlcv(ticker, df)
            stats["ok"] += 1
            log_compute("ohlcv_fetch", ticker, "ok", f"{n} bars stored")

            if verbose:
                print(f"stored {n} bars")

        except Exception as exc:
            logger.warning("[%s] OHLCV ingest failed: %s", ticker, exc)
            stats["errors"] += 1
            log_compute("ohlcv_fetch", ticker, "error", str(exc)[:300])
            if verbose:
                print(f"  [{ticker}] ERROR: {exc}")
        finally:
            if verbose and (idx % 10 == 0 or idx == total_stocks):
                elapsed = time.time() - phase_t0
                print(
                    f"  [progress] ingest {idx}/{total_stocks} complete "
                    f"(ok={stats['ok']} skipped={stats['skipped']} errors={stats['errors']}) "
                    f"elapsed={elapsed:.1f}s"
                )

    if verbose:
        print(
            f"\n[EagleEye] OHLCV done: {stats['ok']} ok, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )
        if stats["insufficient"]:
            print(f"  No data: {stats['insufficient']}")

    return stats


def _detect_gaps(ticker: str, df) -> List[str]:
    """Return descriptions of consecutive-bar gaps > 7 calendar days."""
    if len(df) < 2:
        return []
    gaps = []
    dates = sorted(df.index)
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        if gap > 7:
            gaps.append(
                f"{ticker}: {dates[i-1].date()} -> {dates[i].date()} ({gap}d gap)"
            )
    return gaps


# ---------------------------------------------------------------------------
# Phase 2 — DNA profiles
# ---------------------------------------------------------------------------

def build_all_dna(verbose: bool = False) -> dict:
    """
    Run the forensic pipeline on cached OHLCV to build BehavioralDNA
    profiles for every ticker in ee_ohlcv_cache.

    Requires Phase 1 to have completed first.
    Tickers with fewer than CONFIG.MIN_HISTORY_DAYS_REQUIRED bars are skipped.

    Returns a summary dict: {ok, skipped, errors, insufficient}.
    """
    from app.services.eagle_eye.config import CONFIG
    from app.services.eagle_eye.dna_extractor import (
        DNA_CONFIDENCE_FLOOR,
        DNA_DEFAULT_WINDOW_DAYS,
        DNA_WINDOW_OPTIONS,
        dna_to_dict,
        extract_dna,
    )
    from app.services.eagle_eye.indicators import compute_all_indicators
    from app.services.eagle_eye.move_detector import detect_fakeouts, detect_moves
    from app.services.eagle_eye.recorder import record_all_events
    from app.services.eagle_eye.store import (
        ensure_tables, list_tickers_with_ohlcv, load_ohlcv, log_compute, save_dna,
    )

    ensure_tables()
    tickers = list_tickers_with_ohlcv()

    stats: dict = {"ok": 0, "skipped": 0, "errors": 0, "insufficient": []}

    if verbose:
        print(
            f"[EagleEye] Building DNA for {len(tickers)} tickers "
            f"(need >= {CONFIG.MIN_HISTORY_DAYS_REQUIRED} bars)"
        )
        print("=" * 70)

    for ticker in tickers:
        try:
            df = load_ohlcv(ticker)

            if len(df) < CONFIG.MIN_HISTORY_DAYS_REQUIRED:
                stats["skipped"] += 1
                stats["insufficient"].append(f"{ticker} ({len(df)} bars)")
                log_compute(
                    "dna_build", ticker, "skip",
                    f"only {len(df)} bars (need {CONFIG.MIN_HISTORY_DAYS_REQUIRED})"
                )
                if verbose:
                    print(f"  [{ticker}] SKIP: only {len(df)} bars")
                continue

            if verbose:
                print(f"  [{ticker}] {len(df)} bars ...", end=" ", flush=True)

            ind_df = compute_all_indicators(df)

            moves = detect_moves(ticker, df)
            fakeouts = detect_fakeouts(ticker, df)
            all_events = moves + fakeouts

            snapshots = record_all_events(all_events, ind_df)

            dna = extract_dna(
                ticker,
                snapshots,
                [],
                indicators_df=ind_df,
                horizon_days=DNA_DEFAULT_WINDOW_DAYS,
                min_setup_occurrences=DNA_CONFIDENCE_FLOOR,
                window_days=DNA_WINDOW_OPTIONS,
            )
            if dna is None:
                stats["skipped"] += 1
                log_compute("dna_build", ticker, "skip", "< 3 real events found")
                if verbose:
                    print("skipped (< 3 events)")
                continue

            dna_dict = dna_to_dict(dna)
            save_dna(
                ticker=ticker,
                dna_dict=dna_dict,
                total_events=dna.total_events_studied,
                dominant_pattern=dna.personality_tag,
            )

            stats["ok"] += 1
            log_compute(
                "dna_build", ticker, "ok",
                f"{dna.total_events_studied} events, pattern={dna.personality_tag}"
            )

            if verbose:
                print(
                    f"DNA built ({dna.total_events_studied} events, "
                    f"pattern={dna.personality_tag})"
                )

        except Exception as exc:
            logger.exception("[%s] DNA build failed", ticker)
            stats["errors"] += 1
            log_compute("dna_build", ticker, "error", str(exc)[:300])
            if verbose:
                print(f"  [{ticker}] ERROR: {exc}")

    if verbose:
        print(
            f"\n[EagleEye] DNA done: {stats['ok']} ok, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )

    return stats


def build_dna_for_ticker(ticker: str) -> Optional[dict]:
    """
    Compute and persist Behavioral DNA for a single ticker on-demand.

    Always attempts a live TickerChart fetch first so the DNA chart bars
    contain accurate, up-to-date OHLC values.  Falls back to the OHLCV DB
    cache only when TickerChart is unavailable or returns insufficient data.
    Returns the raw DNA dict on success, or None if insufficient data.
    """
    from datetime import date, timedelta

    from app.services.eagle_eye.config import CONFIG
    from app.services.eagle_eye.dna_extractor import (
        DNA_CONFIDENCE_FLOOR,
        DNA_DEFAULT_WINDOW_DAYS,
        DNA_WINDOW_OPTIONS,
        dna_to_dict,
        extract_dna,
    )
    from app.services.eagle_eye.indicators import compute_all_indicators
    from app.services.eagle_eye.move_detector import detect_fakeouts, detect_moves
    from app.services.eagle_eye.recorder import record_all_events
    from app.services.eagle_eye.store import (
        ensure_tables,
        load_ohlcv,
        log_compute,
        save_dna,
        save_ohlcv,
    )

    ensure_tables()
    ticker = ticker.upper()

    # 1. Always try a live TickerChart fetch first — this guarantees the DNA
    #    chart bars contain accurate OHLC data from the authoritative source.
    df = None
    try:
        from app.services.eagle_eye.adapter import TickerChartAdapter

        adapter = TickerChartAdapter()
        end_d = date.today()
        start_d = end_d - timedelta(days=3 * 365 + 90)
        fetched = adapter.get_ohlcv_daily(ticker, start_d, end_d)
        if fetched is not None and len(fetched) >= CONFIG.MIN_HISTORY_DAYS_REQUIRED:
            df = fetched
            # Persist fresh data back to the OHLCV cache so other pipelines benefit
            try:
                save_ohlcv(ticker, df)
            except Exception as save_exc:
                logger.warning("Could not persist fresh OHLCV for %s: %s", ticker, save_exc)
            logger.info("DNA build [%s]: using live TickerChart data (%d bars)", ticker, len(df))
    except Exception as exc:
        logger.warning("Live TickerChart OHLCV fetch for DNA failed [%s]: %s", ticker, exc)

    # 2. Fall back to the DB cache when live fetch is unavailable or too sparse
    if df is None or len(df) < CONFIG.MIN_HISTORY_DAYS_REQUIRED:
        cached = load_ohlcv(ticker)
        if len(cached) > (len(df) if df is not None else 0):
            df = cached
            logger.info("DNA build [%s]: using OHLCV cache (%d bars)", ticker, len(df))

    if len(df) < CONFIG.MIN_HISTORY_DAYS_REQUIRED:
        log_compute(
            "dna_build", ticker, "skip",
            f"only {len(df)} bars (need {CONFIG.MIN_HISTORY_DAYS_REQUIRED})"
        )
        return None

    try:
        ind_df = compute_all_indicators(df)
        moves = detect_moves(ticker, df)
        fakeouts = detect_fakeouts(ticker, df)
        all_events = moves + fakeouts
        snapshots = record_all_events(all_events, ind_df)

        dna = extract_dna(
            ticker,
            snapshots,
            [],
            indicators_df=ind_df,
            horizon_days=DNA_DEFAULT_WINDOW_DAYS,
            min_setup_occurrences=DNA_CONFIDENCE_FLOOR,
            window_days=DNA_WINDOW_OPTIONS,
        )
        if dna is None:
            log_compute("dna_build", ticker, "skip", "< 3 real events found")
            return None

        dna_dict = dna_to_dict(dna)
        save_dna(
            ticker=ticker,
            dna_dict=dna_dict,
            total_events=dna.total_events_studied,
            dominant_pattern=dna.personality_tag,
        )
        log_compute(
            "dna_build", ticker, "ok",
            f"{dna.total_events_studied} events, pattern={dna.personality_tag}"
        )
        return dna_dict

    except Exception as exc:
        logger.exception("[%s] on-demand DNA build failed", ticker)
        log_compute("dna_build", ticker, "error", str(exc)[:300])
        return None


def _build_premier_market_proxy(premier_tickers: List[str]) -> Optional[pd.Series]:
    """Build traded-value weighted close proxy for Premier market regime context."""
    if not premier_tickers:
        return None

    from app.core.database import query_all

    placeholders = ", ".join(["?"] * len(premier_tickers))
    rows = query_all(
        f"""
        SELECT bar_date, ticker, close, turnover_kwd
        FROM ee_ohlcv_cache
        WHERE ticker IN ({placeholders})
        ORDER BY bar_date
        """,
        tuple(premier_tickers),
    )
    if not rows:
        return None

    normalized_rows = []
    for row in rows:
        if hasattr(row, "items"):
            normalized_rows.append(dict(row.items()))
        elif isinstance(row, (tuple, list)) and len(row) >= 4:
            normalized_rows.append(
                {
                    "bar_date": row[0],
                    "ticker": row[1],
                    "close": row[2],
                    "turnover_kwd": row[3],
                }
            )

    if not normalized_rows:
        return None

    frame = pd.DataFrame(normalized_rows)
    if frame.empty or "bar_date" not in frame.columns:
        return None

    frame["bar_date"] = pd.to_datetime(frame["bar_date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["turnover_kwd"] = pd.to_numeric(frame["turnover_kwd"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["bar_date", "close"])
    if frame.empty:
        return None

    weighted_sum = (frame["close"] * frame["turnover_kwd"]).groupby(frame["bar_date"]).sum()
    weight_sum = frame["turnover_kwd"].groupby(frame["bar_date"]).sum()
    simple_mean = frame["close"].groupby(frame["bar_date"]).mean()

    proxy = (weighted_sum / weight_sum.replace(0, np.nan)).where(weight_sum > 0, simple_mean).sort_index()
    return proxy.astype(float)

def compute_all_ratings(verbose: bool = False) -> dict:
    """
    Rate every stock in ee_ohlcv_cache using the Eagle Eye rating engine.

    Reads OHLCV from the DB (no TickerChart calls).
    Populates ee_ratings_cache so the scanner endpoint is instant.

    Returns a summary dict: {ok, skipped, errors}.
    """
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
    from app.services.eagle_eye import stage_classifier as stage_classifier_module
    from app.services.eagle_eye.stage_classifier import classify_stage_with_confidence
    from app.core.database import exec_sql
    from app.services.eagle_eye.store import (
        ensure_tables, list_tickers_with_ohlcv, load_ohlcv,
        log_compute, save_rating,
    )

    ensure_tables()
    run_started = datetime.now().isoformat(timespec="seconds")
    run_date = run_started[:10]
    run_id = f"rating_run_{uuid.uuid4().hex[:12]}"
    try:
        ingest_mtime = int(os.path.getmtime(__file__))
    except OSError:
        ingest_mtime = 0
    stage_file = getattr(stage_classifier_module, "__file__", None)
    try:
        stage_mtime = int(os.path.getmtime(stage_file)) if stage_file else 0
    except OSError:
        stage_mtime = 0
    latest_code_mtime = max(ingest_mtime, stage_mtime)
    code_fingerprint = (
        f"ingest:{ingest_mtime};stage_classifier:{stage_mtime};latest:{latest_code_mtime}"
    )

    log_compute(
        "rating_run",
        None,
        "start",
        f"run_id={run_id} run_started={run_started} code={code_fingerprint}",
    )

    # Full refresh prevents stale rows from previous taxonomy/version runs.
    exec_sql("DELETE FROM ee_ratings_cache", ())
    tickers = list_tickers_with_ohlcv()
    today_str = run_date

    # Build ticker -> StockMeta map for names/sectors
    adapter = TickerChartAdapter()
    stock_meta = {s.ticker: s for s in adapter.list_stocks()}
    premier_tickers = [
        t for t, meta in stock_meta.items()
        if str(getattr(meta, "market_tier", "premier") or "premier").strip().upper() == "PREMIER"
    ]
    market_proxy = _build_premier_market_proxy(premier_tickers)

    stats: dict = {"ok": 0, "skipped": 0, "errors": 0}
    total_tickers = len(tickers)
    phase_t0 = time.time()
    verbose_per_ticker = os.getenv("EE_VERBOSE_PER_TICKER", "0").strip() == "1"

    def _save_placeholder_rating(symbol: str, reason: str, df=None) -> None:
        meta = stock_meta.get(symbol)
        name_en = meta.name_en if meta else symbol
        sector = meta.sector if meta else "Kuwait"
        market_tier = (meta.market_tier if meta and meta.market_tier else "premier").upper()

        stage_map = {
            "insufficient_history": "INSUFFICIENT_HISTORY",
            "inactive_or_delisted": "INACTIVE_OR_DELISTED",
            "indicator_unavailable": "INDICATOR_UNAVAILABLE",
        }
        thesis_map = {
            "insufficient_history": "Insufficient price history for full Eagle Eye scoring."
                                   " Kept in scanner as watchlist-only.",
            "inactive_or_delisted": "Stock appears inactive or delisted based on recent market activity."
                                     " Kept in scanner as watchlist-only.",
            "indicator_unavailable": "Indicators are currently unavailable for this symbol."
                                    " Kept in scanner as watchlist-only.",
        }

        days_of_history = 0
        last_close = None
        if df is not None and len(df) > 0:
            days_of_history = int(len(df))
            try:
                last_close = _safe_float(df["close"].iloc[-1])
            except Exception:
                last_close = None

        indicators = {"close": last_close} if last_close is not None else {}

        result = {
            "ticker": symbol.upper(),
            "market_tier": market_tier,
            "stage": stage_map.get(reason, "DATA_ISSUE"),
            "rating": "AVOID",
            "recommendation": "AVOID",
            "confidence": 0.0,
            "ml_score": None,
            "thesis": thesis_map.get(reason, "Data unavailable; kept in scanner as watchlist-only."),
            "supports": [],
            "resistances": [],
            "entry": {
                "entry_primary": None,
                "entry_aggressive": None,
                "entry_conservative": None,
                "stop_loss": None,
                "tp1": None,
                "tp1_probability": None,
                "tp2": None,
                "tp2_probability": None,
                "tp3": None,
                "tp3_probability": None,
            },
            "indicators": indicators,
            "family_scores": {
                "liquidity": 0.0,
                "trend": 0.0,
                "momentum": 0.0,
                "geometry": 0.0,
                "risk_reward": 0.0,
                "total_score": 0.0,
            },
            "stage_confidence": 0.0,
            "pattern_match": {
                "takeoff_similarity": 0.0,
                "crash_similarity": 0.0,
                "neutral_similarity": 1.0,
                "nearest_analogs": [],
            },
            "why_supporting": [],
            "why_conflicting": ["Insufficient data for stage/recommendation computation"],
            "what_invalidates": [],
            "veto_reasons": [reason],
            "data_quality_score": 0.0,
            "volume_context": {
                "relative_volume": 0.0,
                "relative_volume_percentile": 0.0,
                "liquidity_tier": "WATCH_ONLY",
                "is_volume_confirmed": False,
                "volume_character": "STALE",
                "volume_trend_5d": "NEUTRAL",
            },
            "days_of_history": days_of_history,
            "computed_at": run_started,
            "computed_date": run_date,
            "run_id": run_id,
            "run_started_at": run_started,
            "code_fingerprint": code_fingerprint,
        }

        save_rating(symbol, name_en, sector, result)
        log_compute("rating_run", symbol, "skip", reason)

    if verbose:
        print(f"[EagleEye] Computing ratings for {total_tickers} tickers")
        print(
            f"[EagleEye] Indicator compute window: "
            f"last {RATINGS_INDICATOR_LOOKBACK_BARS} bars per ticker"
        )
        if verbose_per_ticker:
            print("[EagleEye] Verbose detail: per-ticker lines enabled")
        else:
            print("[EagleEye] Verbose detail: progress-only (set EE_VERBOSE_PER_TICKER=1 for per-ticker lines)")
        print("=" * 70)

    for idx, ticker in enumerate(tickers, start=1):
        try:
            df = load_ohlcv(ticker)
            # Keep this in sync with compute_all_indicators minimum history requirement.
            if df is None or len(df) < 50:
                stats["skipped"] += 1
                _save_placeholder_rating(ticker, "insufficient_history", df)
                continue

            if not is_stock_active(ticker, df):
                stats["skipped"] += 1
                _save_placeholder_rating(ticker, "inactive_or_delisted", df)
                continue

            indicator_input = (
                df.tail(RATINGS_INDICATOR_LOOKBACK_BARS)
                if len(df) > RATINGS_INDICATOR_LOOKBACK_BARS
                else df
            )

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                    ind_df = compute_all_indicators(indicator_input, market_close=market_proxy)
            except ValueError as exc:
                if "Need at least 50 bars" in str(exc):
                    stats["skipped"] += 1
                    _save_placeholder_rating(ticker, "insufficient_history", df)
                    continue
                raise

            if ind_df is None or len(ind_df) == 0:
                stats["skipped"] += 1
                _save_placeholder_rating(ticker, "indicator_unavailable", df)
                continue

            latest = ind_df.iloc[-1].to_dict()
            family_scores = compute_family_scores(latest)
            stage, stage_conf = classify_stage_with_confidence(latest, family_scores=family_scores)

            # Keep volume context for UI/debug visibility.
            volume_context = compute_volume_context(df, stage)

            recommendation_payload = generate_recommendation(
                latest,
                family_scores=family_scores,
                total_score=float(family_scores.get("total_score", 50.0)),
                stage=stage,
                stage_conf=stage_conf,
                pattern_match=None,  # Phase 1: rules primary, no pattern-memory adjustment yet.
                data_quality=_safe_float(latest.get("data_quality_score")) or 50.0,
            )

            confidence = float(recommendation_payload["confidence"])
            rating = str(recommendation_payload["recommendation"])
            ml_score = None
            tier = volume_context["liquidity_tier"]

            sr = compute_support_resistance(df, latest)
            et = compute_entry_stop_targets(df, latest, sr, stage=stage)
            explanation = explain(recommendation_payload, latest, pattern_match=None)
            top_supporting = explanation.get("why_supporting", [])[:2]
            thesis = generate_thesis(
                ticker,
                rating,
                stage,
                latest,
                dna=None,
                top_signals_fired=top_supporting,
            )

            meta = stock_meta.get(ticker)
            name_en = meta.name_en if meta else ticker
            sector = meta.sector if meta else "Kuwait"
            market_tier = (meta.market_tier if meta and meta.market_tier else "premier").upper()

            result = {
                "ticker": ticker.upper(),
                "market_tier": market_tier,
                "stage": stage,
                "stage_confidence": recommendation_payload.get("stage_confidence"),
                "rating": rating,
                "recommendation": rating,
                "confidence": confidence,
                "ml_score": ml_score,
                "thesis": thesis,
                "supports": sr.get("supports", []),
                "resistances": sr.get("resistances", []),
                "entry": et,
                "indicators": latest,
                "family_scores": family_scores,
                "pattern_match": recommendation_payload.get("pattern_match", {}),
                "why_supporting": explanation.get("why_supporting", []),
                "why_conflicting": explanation.get("why_conflicting", []),
                "what_invalidates": explanation.get("what_invalidates", []),
                "veto_reasons": recommendation_payload.get("veto_reasons", []),
                "data_quality_score": recommendation_payload.get("data_quality_score"),
                "volume_context": volume_context,
                "days_of_history": len(df),
                "computed_at": run_started,
                "computed_date": run_date,
                "run_id": run_id,
                "run_started_at": run_started,
                "code_fingerprint": code_fingerprint,
            }

            save_rating(ticker, name_en, sector, result)

            # ── Signal logger (observation only — must not block rating) ─────
            try:
                from app.services.eagle_eye.ml.signal_logger import log_considered_signal as _log_sig

                _entered = rating == "BUY"
                _would_have_entered = _entered
                _skip_reason = None
                if not _entered:
                    veto_reasons = recommendation_payload.get("veto_reasons") or []
                    veto_text = " ".join(str(v).lower() for v in veto_reasons)
                    if "distribution" in veto_text or "markdown" in veto_text:
                        _skip_reason = "STAGE_NOT_ALLOWED"
                    elif (
                        "data quality" in veto_text
                        or "infrequently" in veto_text
                        or "near-zero volume" in veto_text
                    ):
                        _skip_reason = "LIQUIDITY_GATE"
                    elif "market bearish" in veto_text:
                        _skip_reason = "CIRCUIT_BREAKER"
                    elif veto_reasons:
                        _skip_reason = "OTHER"
                    else:
                        _skip_reason = "BELOW_CONFIDENCE_THRESHOLD"

                # Sanitize latest snapshot: coerce numpy types, replace NaN/Inf with None
                def _jv(v):
                    if v is None:
                        return None
                    try:
                        f = float(v)
                        return None if (f != f or abs(f) == math.inf) else f
                    except (TypeError, ValueError):
                        return str(v)

                _feature_snapshot = {
                    "stage": stage,
                    "recommendation": rating,
                    "tier": tier,
                    "phase_score": None,
                    "confidence_pre_adj": float(confidence),
                    "dampener_fired": False,
                    **{k: _jv(v) for k, v in latest.items()},
                }
                _log_sig(
                    ticker=ticker,
                    signal_date=today_str,
                    rule_score=float(confidence),
                    would_have_entered=_would_have_entered,
                    skip_reason=_skip_reason,
                    features=_feature_snapshot,
                )
            except Exception as _log_exc:
                logger.warning("[%s] log_considered_signal failed: %s", ticker, _log_exc)
            # ── End signal logger ─────────────────────────────────────────────

            stats["ok"] += 1
            log_compute(
                "rating_run", ticker, "ok",
                f"confidence={confidence:.1f} rating={rating} stage={stage}"
            )

            if verbose and verbose_per_ticker:
                print(f"  [{ticker}] {rating} (conf={confidence:.0f}%) stage={stage}")

        except Exception as exc:
            logger.exception("[%s] rating computation/persistence failed", ticker)
            stats["errors"] += 1
            log_compute("rating_run", ticker, "error", str(exc)[:300])
        finally:
            if verbose and (idx % 10 == 0 or idx == total_tickers):
                elapsed = time.time() - phase_t0
                print(
                    f"  [progress] ratings {idx}/{total_tickers} complete "
                    f"(ok={stats['ok']} skipped={stats['skipped']} errors={stats['errors']}) "
                    f"elapsed={elapsed:.1f}s"
                )

    if verbose:
        print(
            f"\n[EagleEye] Ratings done: {stats['ok']} rated, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )

    log_compute(
        "rating_run",
        None,
        "summary",
        (
            f"run_id={run_id} run_started={run_started} "
            f"ok={stats['ok']} skipped={stats['skipped']} errors={stats['errors']}"
        ),
    )

    return stats


# ---------------------------------------------------------------------------
# Nightly orchestrator — entry point for the APScheduler job
# ---------------------------------------------------------------------------

def run_nightly_recompute(dna_refresh: bool = False, verbose: bool = False) -> dict:
    """
    Nightly pipeline orchestrator called by the background scheduler.

    Phase 1 (OHLCV) and Phase 3 (ratings) run every trading day.
    Phase 2 (DNA) is optional — set *dna_refresh=True* when a full DNA refresh
    should be included in the nightly run.

    Never raises; exceptions are logged and captured in the return dict.
    """
    logger.info(
        "Eagle Eye nightly recompute starting (dna_refresh=%s)", dna_refresh
    )
    t0 = time.time()
    from app.core.database import query_val
    from app.services.eagle_eye.store import log_compute

    ohlcv_stats: dict = {}
    dna_stats: dict = {}
    rating_stats: dict = {}

    log_compute("nightly_recompute", None, "start", f"dna_refresh={dna_refresh}")

    if verbose:
        print(f"[EagleEye] Nightly recompute started (dna_refresh={dna_refresh})")

    if verbose:
        print("[EagleEye] Phase 1/3: ingest_all_ohlcv ...")
    phase_t0 = time.time()

    try:
        ohlcv_stats = ingest_all_ohlcv(verbose=verbose)
    except Exception as exc:
        logger.error("Eagle Eye OHLCV ingest failed: %s", exc)
        ohlcv_stats = {"error": str(exc)}
    if verbose:
        print(
            f"[EagleEye] Phase 1/3 done in {time.time() - phase_t0:.1f}s: "
            f"{ohlcv_stats}"
        )

    if dna_refresh:
        if verbose:
            print("[EagleEye] Phase 2/3: build_all_dna ...")
        phase_t0 = time.time()
        try:
            dna_stats = build_all_dna(verbose=verbose)
        except Exception as exc:
            logger.error("Eagle Eye DNA build failed: %s", exc)
            dna_stats = {"error": str(exc)}
        if verbose:
            print(
                f"[EagleEye] Phase 2/3 done in {time.time() - phase_t0:.1f}s: "
                f"{dna_stats}"
            )

    if verbose:
        print("[EagleEye] Phase 3/3: compute_all_ratings ...")
    phase_t0 = time.time()
    try:
        rating_stats = compute_all_ratings(verbose=verbose)
    except Exception as exc:
        logger.error("Eagle Eye rating run failed: %s", exc)
        rating_stats = {"error": str(exc)}
    if verbose:
        print(
            f"[EagleEye] Phase 3/3 done in {time.time() - phase_t0:.1f}s: "
            f"{rating_stats}"
        )

    elapsed = round(time.time() - t0, 1)
    cache_rows = int(query_val("SELECT COUNT(*) FROM ee_ratings_cache", ()) or 0)
    summary = (
        "elapsed_sec=%s ohlcv_ok=%s ohlcv_errors=%s ratings_ok=%s "
        "ratings_errors=%s cache_rows=%s"
    ) % (
        elapsed,
        ohlcv_stats.get("ok", 0),
        ohlcv_stats.get("errors", 0),
        rating_stats.get("ok", 0),
        rating_stats.get("errors", 0),
        cache_rows,
    )
    if cache_rows == 0:
        logger.error("Eagle Eye nightly recompute finished with empty ratings cache: %s", summary)
        log_compute("nightly_recompute", None, "error", summary)
    else:
        logger.info("Eagle Eye nightly recompute finished in %.1fs (%s)", elapsed, summary)
        log_compute("nightly_recompute", None, "ok", summary)

    if verbose:
        print(f"[EagleEye] Nightly recompute complete in {elapsed:.1f}s")
        print(f"[EagleEye] Summary: {summary}")

    return {
        "elapsed_sec": elapsed,
        "ohlcv": ohlcv_stats,
        "dna": dna_stats,
        "ratings": rating_stats,
        "cache_rows": cache_rows,
    }
