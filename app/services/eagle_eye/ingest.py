"""
Eagle Eye Phase 3 — Live Data Ingestion Pipeline.

Three sequential phases, each independently re-runnable:

  Phase 1 — ingest_all_ohlcv()
      Fetch 3 years of daily OHLCV for every KSE stock and store to
      ee_ohlcv_cache. Incremental: only fetches bars after the most
      recent cached date, so reruns are fast.

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
import time
from datetime import date, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)


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

    Incremental: only fetches bars after the latest cached date for each
    ticker, so repeated calls on the same day are virtually free.

    Returns a summary dict: {ok, skipped, errors, insufficient, gaps}.
    """
    from app.core.config import get_settings
    from app.services.eagle_eye.adapter import TickerChartAdapter
    from app.services.eagle_eye.store import (
        ensure_tables, get_latest_ohlcv_date, log_compute, save_ohlcv,
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

    today = date.today()
    history_start = today - timedelta(days=3 * 365 + 60)  # 3 years + buffer

    stats: dict = {"ok": 0, "skipped": 0, "errors": 0, "insufficient": [], "gaps": []}

    if verbose:
        print(f"[EagleEye] Ingesting OHLCV for {len(stocks)} stocks ({history_start} → {today})")
        print("=" * 70)

    for stock in stocks:
        ticker = stock.ticker
        try:
            last_date = get_latest_ohlcv_date(ticker)

            # Already up to date?
            if last_date is not None and last_date >= today:
                stats["skipped"] += 1
                log_compute("ohlcv_fetch", ticker, "skip", f"up to date: {last_date}")
                if verbose:
                    print(f"  [{ticker}] already up to date ({last_date})")
                continue

            # Incremental: only request missing bars
            fetch_start = (last_date + timedelta(days=1)) if last_date else history_start

            if verbose:
                print(f"  [{ticker}] fetching {fetch_start} → {today} ...", end=" ", flush=True)

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
                    print(f"  ⚠  {len(gaps)} gap(s) detected", end=" ")

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
                f"{ticker}: {dates[i-1].date()} → {dates[i].date()} ({gap}d gap)"
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

    Tries the OHLCV DB cache first; falls back to a live TickerChart fetch
    when the cache is too sparse (< MIN_HISTORY_DAYS_REQUIRED bars).
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
    )

    ensure_tables()
    ticker = ticker.upper()

    # 1. Try cached OHLCV
    df = load_ohlcv(ticker)

    # 2. Fall back to live fetch when cache is too sparse
    if len(df) < CONFIG.MIN_HISTORY_DAYS_REQUIRED:
        try:
            from app.services.eagle_eye.adapter import TickerChartAdapter

            adapter = TickerChartAdapter()
            end_d = date.today()
            start_d = end_d - timedelta(days=3 * 365 + 90)
            fetched = adapter.get_ohlcv_daily(ticker, start_d, end_d)
            if fetched is not None and len(fetched) > len(df):
                df = fetched
        except Exception as exc:
            logger.warning("Live OHLCV fetch for DNA failed [%s]: %s", ticker, exc)

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
        classify_stage,
        compute_confidence,
        compute_entry_stop_targets,
        compute_rating,
        compute_support_resistance,
        compute_volume_context,
        generate_thesis,
    )
    from app.services.eagle_eye.store import (
        ensure_tables, list_tickers_with_ohlcv, load_ohlcv,
        log_compute, save_rating,
    )

    ensure_tables()
    tickers = list_tickers_with_ohlcv()
    today_str = date.today().isoformat()

    # Build ticker → StockMeta map for names/sectors
    adapter = TickerChartAdapter()
    stock_meta = {s.ticker: s for s in adapter.list_stocks()}

    stats: dict = {"ok": 0, "skipped": 0, "errors": 0}

    if verbose:
        print(f"[EagleEye] Computing ratings for {len(tickers)} tickers")
        print("=" * 70)

    for ticker in tickers:
        try:
            df = load_ohlcv(ticker)
            if len(df) < 30:
                stats["skipped"] += 1
                continue

            ind_df = compute_all_indicators(df)
            if ind_df is None or len(ind_df) == 0:
                stats["skipped"] += 1
                continue

            latest = ind_df.iloc[-1].to_dict()

            stage = classify_stage(latest)
            confidence = compute_confidence(latest, stage, dna=None)
            _confidence_raw = confidence  # snapshot before vol-context/dampener adjustments
            _dampener_fired = False

            # ── Volume context + confidence multiplier ───────────────────
            volume_context = compute_volume_context(df, stage)
            tier = volume_context["liquidity_tier"]
            _LIQUIDITY_MULTIPLIERS = {"ILLIQUID": 0.50, "WATCH_ONLY": 0.70}
            if tier in _LIQUIDITY_MULTIPLIERS:
                confidence = confidence * _LIQUIDITY_MULTIPLIERS[tier]
            elif not volume_context["is_volume_confirmed"]:
                confidence = confidence * 0.85
            elif volume_context["relative_volume_percentile"] > 80:
                confidence = min(confidence * 1.10, 100)
            confidence = round(min(confidence, 100), 2)

            # ── Thin-volume-on-rise dampener (net liquidity check) ───────
            if "dollar_volume" in ind_df.columns and len(ind_df) >= 21:
                _dv = ind_df["dollar_volume"]
                _median_dv = float(_dv.rolling(20).median().shift(1).iloc[-1])
                _today_dv = float(_dv.iloc[-1])
                _rel_liq = _today_dv / _median_dv if _median_dv > 0 else 0.0
                _today_ret = (
                    float((df["close"].iloc[-1] / df["close"].iloc[-2]) - 1)
                    if len(df) >= 2 else 0.0
                )
                if _rel_liq < 0.5 and _today_ret > 0.02:
                    confidence = min(confidence, 60)
                    _dampener_fired = True
                    log_compute(
                        "rating_run", ticker, "dampened",
                        f"thin_volume_on_rise: rel_liq={_rel_liq:.2f} "
                        f"ret={_today_ret:.3f} conf_capped=60",
                    )

            rating = compute_rating(confidence)
            sr = compute_support_resistance(df, latest)
            et = compute_entry_stop_targets(df, latest, sr, stage=stage)
            thesis = generate_thesis(
                ticker, rating, stage, latest, dna=None, top_signals_fired=[]
            )

            meta = stock_meta.get(ticker)
            name_en = meta.name_en if meta else ticker
            sector = meta.sector if meta else "Kuwait"
            market_tier = (meta.market_tier if meta and meta.market_tier else "premier").upper()

            result = {
                "ticker": ticker.upper(),
                "market_tier": market_tier,
                "stage": stage,
                "rating": rating,
                "confidence": confidence,
                "thesis": thesis,
                "supports": sr.get("supports", []),
                "resistances": sr.get("resistances", []),
                "entry": et,
                "indicators": latest,
                "volume_context": volume_context,
                "days_of_history": len(df),
                "computed_at": today_str,
            }

            save_rating(ticker, name_en, sector, result)

            # ── Signal logger (observation only — must not block rating) ─────
            try:
                from app.services.eagle_eye.ml.signal_logger import log_considered_signal as _log_sig
                from app.services.eagle_eye.config import CONFIG as _cfg

                _entered = rating in ("BUY", "STRONG_BUY")
                # would_have_entered: True if raw signal crossed threshold,
                # even if a gate (liquidity/dampener) brought it below.
                _would_have_entered = _entered or (_confidence_raw >= _cfg.BUY_CONFIDENCE)
                _skip_reason = None
                if not _entered:
                    if _dampener_fired or tier in ("ILLIQUID", "WATCH_ONLY"):
                        _skip_reason = "LIQUIDITY_GATE"
                    elif stage in ("DISTRIBUTION_TOPPING", "MARKDOWN_DECLINE"):
                        _skip_reason = "STAGE_NOT_ALLOWED"
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
                    "tier": tier,
                    "confidence_pre_adj": float(_confidence_raw),
                    "dampener_fired": bool(_dampener_fired),
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
                f"confidence={confidence:.1f} rating={rating}"
            )

            if verbose:
                print(f"  [{ticker}] {rating} (conf={confidence:.0f}%) stage={stage}")

        except Exception as exc:
            logger.exception("[%s] rating computation/persistence failed", ticker)
            stats["errors"] += 1
            log_compute("rating_run", ticker, "error", str(exc)[:300])

    if verbose:
        print(
            f"\n[EagleEye] Ratings done: {stats['ok']} rated, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
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

    try:
        ohlcv_stats = ingest_all_ohlcv(verbose=verbose)
    except Exception as exc:
        logger.error("Eagle Eye OHLCV ingest failed: %s", exc)
        ohlcv_stats = {"error": str(exc)}

    if dna_refresh:
        try:
            dna_stats = build_all_dna(verbose=verbose)
        except Exception as exc:
            logger.error("Eagle Eye DNA build failed: %s", exc)
            dna_stats = {"error": str(exc)}

    try:
        rating_stats = compute_all_ratings(verbose=verbose)
    except Exception as exc:
        logger.error("Eagle Eye rating run failed: %s", exc)
        rating_stats = {"error": str(exc)}

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

    return {
        "elapsed_sec": elapsed,
        "ohlcv": ohlcv_stats,
        "dna": dna_stats,
        "ratings": rating_stats,
        "cache_rows": cache_rows,
    }
