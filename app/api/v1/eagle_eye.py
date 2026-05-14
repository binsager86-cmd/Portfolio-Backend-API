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

import logging
import uuid
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.schemas.eagle_eye import (
    BehavioralDNAResponse,
    DNAResponse,
    EventsListResponse,
    FullStockAnalysis,
    MoveEventResponse,
    RatedStock,
    RefreshRequest,
    RefreshResponse,
    RegimeResponse,
    ScannerResponse,
    SignalBreakdown,
    StockAnalysisResponse,
    SupportResistanceLevel,
    ThresholdProfileResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eagle-eye", tags=["Eagle Eye"])

# ---------------------------------------------------------------------------
# In-memory cache — keyed by "<TICKER>:<ISO_DATE>" to allow daily staleness
# ---------------------------------------------------------------------------
_cache: Dict[str, dict] = {}
_DNA_CACHE: Dict[str, dict] = {}
_EVENTS_CACHE: Dict[str, list] = {}

_LOOKBACK_YEARS = 5


def _cache_key(ticker: str, as_of: Optional[date] = None) -> str:
    d = (as_of or date.today()).isoformat()
    return f"{ticker.upper()}:{d}"


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
        from app.services.eagle_eye.store import load_rating

        cached_row = load_rating(ticker)
        if cached_row and cached_row.get("computed_at") == date.today().isoformat():
            indicators = cached_row.get("indicators_json") or {}
            if isinstance(indicators, str):
                import json
                indicators = json.loads(indicators)
            result = {
                "ticker": ticker.upper(),
                "stage": cached_row.get("stage"),
                "rating": cached_row.get("rating"),
                "confidence": cached_row.get("confidence"),
                "thesis": cached_row.get("thesis"),
                "supports": cached_row.get("supports_json") or [],
                "resistances": cached_row.get("resistances_json") or [],
                "entry": {
                    "entry_primary": cached_row.get("entry_primary"),
                    "entry_aggressive": cached_row.get("entry_aggressive"),
                    "entry_conservative": cached_row.get("entry_conservative"),
                    "stop_loss": cached_row.get("stop_loss"),
                    "tp1": cached_row.get("tp1"),
                    "tp1_probability": cached_row.get("tp1_probability"),
                    "tp2": cached_row.get("tp2"),
                    "tp2_probability": cached_row.get("tp2_probability"),
                    "tp3": cached_row.get("tp3"),
                    "tp3_probability": cached_row.get("tp3_probability"),
                },
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
            classify_stage,
            compute_confidence,
            compute_entry_stop_targets,
            compute_rating,
            compute_support_resistance,
            generate_thesis,
        )

        adapter = TickerChartAdapter()
        end_d = date.today()
        start_d = end_d - timedelta(days=_LOOKBACK_YEARS * 365 + 60)

        df = adapter.get_ohlcv_daily(ticker, start_d, end_d)
        if df is None or len(df) < 30:
            return None

        indicators_df = compute_all_indicators(df)
        if indicators_df is None or len(indicators_df) == 0:
            return None

        latest = indicators_df.iloc[-1].to_dict()

        stage = classify_stage(latest)
        confidence = compute_confidence(latest, stage, dna=None)
        rating = compute_rating(confidence)
        sr = compute_support_resistance(df, latest)
        et = compute_entry_stop_targets(df, latest, sr, stage=stage)
        thesis = generate_thesis(ticker, rating, stage, latest, dna=None, top_signals_fired=[])

        result = {
            "ticker": ticker.upper(),
            "stage": stage,
            "rating": rating,
            "confidence": confidence,
            "thesis": thesis,
            "supports": sr.get("supports", []),
            "resistances": sr.get("resistances", []),
            "entry": et,
            "indicators": latest,
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
    limit: int = Query(50, ge=1, le=200, description="Maximum number of stocks to return"),
    _user: TokenData = Depends(get_current_user),
):
    """
    Return a rated list of Kuwait stocks, optionally filtered by sector, tier, and
    minimum confidence. Reads from ee_ratings_cache (pre-computed nightly) for
    instant response; falls back to live per-stock computation when the cache is empty.
    """
    try:
        from app.services.eagle_eye.adapter import TickerChartAdapter
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Eagle Eye service unavailable: {exc}")

    # ── DB fast path: read pre-computed ratings ──────────────────────
    try:
        from app.services.eagle_eye.store import load_all_ratings

        db_rows = load_all_ratings()
        if db_rows:
            # Build StockMeta map for names/sectors (quick, from adapter)
            adapter = TickerChartAdapter()
            meta_map = {s.ticker: s for s in adapter.list_stocks()}

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
                if conf < min_confidence:
                    continue

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
                    last_price=row.get("last_price"),
                    computed_at=row.get("computed_at"),
                ))
                if len(results) >= limit:
                    break

            if results:
                return ScannerResponse(status="ok", count=len(results), stocks=results)
    except Exception as exc:
        logger.debug("DB scanner fast path failed, using live compute: %s", exc)

    # ── Live compute fallback (first run / cache empty) ───────────────
    adapter = TickerChartAdapter()
    stocks_meta = adapter.list_stocks()

    # Apply sector / tier filter before computing
    if sector:
        stocks_meta = [s for s in stocks_meta if s.sector.lower() == sector.lower()]
    if tier:
        stocks_meta = [s for s in stocks_meta if s.market_tier.lower() == tier.lower()]

    results: List[RatedStock] = []
    for meta in stocks_meta[:limit * 2]:  # compute extra to allow confidence filter
        analysis = _run_analysis(meta.ticker)
        if analysis is None:
            continue
        if analysis["confidence"] < min_confidence:
            continue

        et = analysis.get("entry", {})
        results.append(RatedStock(
            ticker=analysis["ticker"],
            name_en=meta.name_en,
            sector=meta.sector,
            stage=analysis["stage"],
            rating=analysis["rating"],
            confidence=analysis["confidence"],
            thesis=analysis["thesis"],
            entry_primary=et.get("entry_primary"),
            stop_loss=et.get("stop_loss"),
            tp1=et.get("tp1"),
            last_price=float(analysis["indicators"].get("close") or 0) or None,
            computed_at=analysis.get("computed_at"),
        ))
        if len(results) >= limit:
            break

    # Sort by confidence descending
    results.sort(key=lambda x: -(x.confidence or 0))
    return ScannerResponse(status="ok", count=len(results), stocks=results)


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
    if portfolio_kwd > 0:
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
        stop_loss=et.get("stop_loss"),
        tp1=et.get("tp1"),
        tp1_probability=et.get("tp1_probability"),
        tp2=et.get("tp2"),
        tp2_probability=et.get("tp2_probability"),
        tp3=et.get("tp3"),
        tp3_probability=et.get("tp3_probability"),
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

    # 1. Fast path — in-memory cache
    if cache_key in _DNA_CACHE:
        dna_dict = _DNA_CACHE[cache_key]
        return DNAResponse(status="ok", data=BehavioralDNAResponse(**dna_dict))

    # 2. Check the DB store (written by the nightly Phase-2 pipeline)
    try:
        stored = load_dna(t)
    except Exception as exc:
        logger.warning("load_dna failed for %s: %s", t, exc)
        stored = None

    if stored is None:
        # Pipeline hasn't built the DNA for this ticker yet — return pending
        return JSONResponse(
            status_code=200,
            content={
                "status": "pending",
                "message": (
                    "Behavioral DNA is still being computed for this stock. "
                    "Check back in a few minutes."
                ),
                "ticker": t,
            },
        )

    # 3. Build response from stored DNA dict
    profiles: List[ThresholdProfileResponse] = []
    for tp in stored.get("profiles_by_threshold", []):
        profiles.append(ThresholdProfileResponse(
            threshold_pct=tp.get("threshold_pct", 0),
            success_rate=tp.get("success_rate", 0),
            sample_count=tp.get("sample_count", 0),
            median_bars_to_hit=tp.get("median_bars_to_hit"),
            avg_win_pct=tp.get("avg_win_pct"),
            avg_loss_pct=tp.get("avg_loss_pct"),
        ))

    most_reliable = stored.get("most_reliable_signals_overall", [])
    if most_reliable and isinstance(most_reliable[0], dict):
        most_reliable = [s.get("signal", "") for s in most_reliable if s.get("signal")]

    dna_response = BehavioralDNAResponse(
        ticker=t,
        total_events_analyzed=stored.get("total_events_studied", 0),
        most_reliable_signals=most_reliable[:10],
        threshold_profiles=profiles,
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
        import threading
        from app.services.eagle_eye.ingest import run_nightly_recompute

        thread = threading.Thread(
            target=run_nightly_recompute,
            kwargs={"dna_refresh": False, "verbose": False},
            daemon=True,
            name="ee_refresh_bg",
        )
        thread.start()
        logger.info(
            "Eagle Eye background recompute triggered for %d ticker(s)",
            len(body.tickers),
        )
    except Exception as exc:
        logger.warning("Could not start Eagle Eye background recompute: %s", exc)

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
    try:
        from app.services.eagle_eye.adapter import TickerChartAdapter

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

        from app.services.eagle_eye.indicators import compute_all_indicators

        above_50ma_count = 0
        checked = 0
        for meta in stocks_meta[:30]:  # sample 30 stocks for breadth
            try:
                df = adapter.get_ohlcv_daily(meta.ticker, start_d, end_d)
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

    except Exception as exc:
        logger.warning("Regime calculation failed: %s", exc)
        return RegimeResponse(
            status="ok",
            regime="NEUTRAL",
            last_updated=datetime.utcnow().date().isoformat(),
        )
