"""
trend_hold_batch.py -- daily batch orchestration for trend_hold_engine.py.

Runs the pure trend_hold_engine (Donchian/EMA-cross entry, chandelier stop,
scale-out) across every scanner ticker using the live ee_ohlcv_cache data,
and upserts each ticker's latest decision into ee_trend_hold_state.

Kept separate from trend_hold_engine.py so that module stays pure (no DB or
orchestration concerns) -- mirrors the existing split in app/services/
eagle_eye/ between rating_engine.py (pure scoring) and ingest.py
(orchestration: iterate tickers, load data, persist results).

ee_ratings_cache (~857 rows/ticker across ~141 tickers, ~105K rows total) is
small enough that recomputing the full replay from scratch every run is fast
(low seconds for the whole universe) -- no incremental/carry-forward state is
needed here, unlike a system processing years of tick data. Each run simply
takes the latest row of replay_symbol()'s output as "today's decision" and
upserts it, the same "recompute fresh, keep only the latest" pattern
ee_ratings_cache itself already uses.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)

# EMA50/ATR14/RSI14/ADX14/Donchian-40 all need real warm-up history before
# they produce anything but NaN; skip tickers too new to feed them rather
# than spend time running the full feature pipeline for a guaranteed no-op.
MIN_SESSIONS_REQUIRED = 60


def _adapt_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Adapt store.load_ohlcv()'s shape (DatetimeIndex named "date"; columns
    open/high/low/close/volume/turnover_kwd) to what
    trend_hold_engine.compute_daily_features() expects (trade_date/open/
    high/low/close/volume/value_kwd columns).
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["trade_date", "open", "high", "low", "close", "volume", "value_kwd"])
    df = raw.reset_index().rename(columns={"date": "trade_date", "turnover_kwd": "value_kwd"})
    df["trade_date"] = df["trade_date"].astype(str).str.slice(0, 10)
    return df


def run_trend_hold_scan() -> Dict[str, Any]:
    """
    Score every scanner ticker with trend_hold_engine and upsert the latest
    decision into ee_trend_hold_state.

    Never raises -- per-ticker failures are isolated and counted, matching
    compute_all_ratings()'s resilience pattern (one bad symbol's OHLCV or a
    transient error never blocks the rest of the universe).
    """
    from app.services.eagle_eye.store import (
        list_tickers_with_ohlcv,
        load_ohlcv,
        save_trend_hold_state,
        save_trend_hold_state_snapshot,
    )
    from app.services.eagle_eye_v2.trend_hold_engine import compute_daily_features, replay_symbol

    t0 = time.time()
    tickers = list_tickers_with_ohlcv()
    stats: Dict[str, Any] = {"scored": 0, "skipped": 0, "errors": 0, "expected": len(tickers)}

    for ticker in tickers:
        try:
            raw = load_ohlcv(ticker)
            if raw is None or len(raw) < MIN_SESSIONS_REQUIRED:
                stats["skipped"] += 1
                continue
            features = compute_daily_features(_adapt_ohlcv(raw))
            rows = replay_symbol(features)
            if not rows:
                stats["skipped"] += 1
                continue
            save_trend_hold_state(ticker, rows[-1])
            save_trend_hold_state_snapshot(ticker, rows[-1])
            stats["scored"] += 1
        except Exception as exc:
            logger.warning("trend_hold_batch: error scoring %s: %s", ticker, exc)
            stats["errors"] += 1

    stats["elapsed_sec"] = round(time.time() - t0, 2)
    logger.info(
        "trend_hold_batch: scored=%d/%d skipped=%d errors=%d in %.1fs",
        stats["scored"], stats["expected"], stats["skipped"], stats["errors"], stats["elapsed_sec"],
    )
    return stats
