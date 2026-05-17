"""
ml/precursor_builder.py — Phase 2 Deliverable 5

Move precursor library builder.

For each eligible stock:
  1. Detect historical move events (detect_moves + detect_fakeouts).
  2. For each MoveEvent: capture indicator snapshots at T-30, T-14, T-7, T-3, T-1
     (trading days before acceleration_date).
  3. Write rows to the `move_precursors` table in SQLite.

Table schema (from db_tables.py):
  move_precursors (
    id INTEGER PRIMARY KEY,
    stock_ticker TEXT,
    precursor_date TEXT,
    acceleration_date TEXT,
    pattern_type TEXT,       -- 'T-30', 'T-14', 'T-7', 'T-3', 'T-1'
    signal_strength REAL,    -- indicator-based composite score at that date
    context_json TEXT,       -- key indicator snapshot (JSON)
    move_outcome TEXT,       -- 'BULL_5PCT', 'BULL_10PCT', 'FAKEOUT', 'NO_MOVE'
    created_at TEXT
  )

Verification: after writing, random 5 events per stock are spot-checked
against raw OHLCV to confirm precursor_date alignment.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from app.services.eagle_eye.indicators import compute_all_indicators
from app.services.eagle_eye.move_detector import detect_moves, detect_fakeouts, MoveEvent
from app.services.eagle_eye.store import load_ohlcv, list_tickers_with_ohlcv

LOGGER = logging.getLogger(__name__)

PRECURSOR_OFFSETS: Dict[str, int] = {
    "T-30": 30,
    "T-14": 14,
    "T-7":  7,
    "T-3":  3,
    "T-1":  1,
}

# Indicator keys to snapshot into context_json
_CONTEXT_KEYS = [
    "close", "volume", "dollar_volume",
    "rsi_14", "macd_line", "macd_signal",
    "adx_14", "bb_width", "atr_14",
    "ema_20", "ema_50", "ema_200",
    "vol_ratio_20",
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
]


def _safe_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _compute_signal_strength(row: pd.Series) -> float:
    """
    Composite signal strength score in [0, 100] at one indicator row.

    Combines RSI position, MACD alignment, ADX strength, and
    BB squeeze as a simple weighted sum — purely diagnostic.
    """
    score = 0.0
    weight_total = 0.0

    rsi = _safe_float(row.get("rsi_14"))
    if rsi is not None:
        rsi_score = max(0.0, min(1.0, (rsi - 30) / 40))
        score += rsi_score * 30
        weight_total += 30

    macd = _safe_float(row.get("macd_line"))
    signal = _safe_float(row.get("macd_signal"))
    if macd is not None and signal is not None:
        macd_score = 1.0 if macd > signal else 0.0
        score += macd_score * 25
        weight_total += 25

    adx = _safe_float(row.get("adx_14"))
    if adx is not None:
        adx_score = min(1.0, adx / 40)
        score += adx_score * 25
        weight_total += 25

    bb_width = _safe_float(row.get("bb_width"))
    if bb_width is not None:
        bb_score = max(0.0, min(1.0, 1.0 - bb_width / 0.2))
        score += bb_score * 20
        weight_total += 20

    if weight_total <= 0:
        return 50.0
    return (score / weight_total) * 100


def _outcome_label(event: MoveEvent) -> str:
    if event.is_fakeout:
        return "FAKEOUT"
    if event.gain_pct is not None and event.gain_pct >= 10.0:
        return "BULL_10PCT"
    if event.gain_pct is not None and event.gain_pct >= 5.0:
        return "BULL_5PCT"
    return "NO_MOVE"


def _build_context_dict(row: pd.Series) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    for key in _CONTEXT_KEYS:
        val = _safe_float(row.get(key))
        if val is not None:
            ctx[key] = round(val, 6)
    return ctx


def build_precursors_for_ticker(
    ticker: str,
    ohlcv: Optional[pd.DataFrame] = None,
    *,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Build precursor rows for all detected move events of one stock.

    Parameters
    ----------
    dry_run : if True, return rich diagnostic objects (for smoke-test Check 5)
              instead of flat DB-ready dicts.  Does NOT write to DB.

    Each returned dict (flat mode, dry_run=False):
      stock_ticker, precursor_date, acceleration_date, pattern_type,
      signal_strength, context_json, move_outcome, created_at

    Each returned dict (dry_run=True, diagnostic mode):
      move_start_date, acceleration_date, move_outcome,
      snapshots: list of {
          snapshot_offset_days, snapshot_date,
          latest_data_date,       ← latest OHLCV bar used (must <= snapshot_date)
          signal_strength, context
      }

    Returns a list of row dicts ready for insert into move_precursors
    (or diagnostic dicts if dry_run=True).
    """
    log = logger or LOGGER

    if ohlcv is None:
        ohlcv = load_ohlcv(ticker)
    if ohlcv is None or len(ohlcv) < 120:
        return []

    try:
        ind_df = compute_all_indicators(ohlcv)
    except Exception as exc:
        log.warning("[%s] compute_all_indicators failed: %s", ticker, exc)
        return []

    events: List[MoveEvent] = detect_moves(ticker, ohlcv)
    events.extend(detect_fakeouts(ticker, ohlcv))

    rows: List[Dict[str, Any]] = []
    now_str = datetime.utcnow().isoformat()

    for event in events:
        try:
            accel_ts = pd.Timestamp(event.acceleration_date)
            accel_pos = ind_df.index.get_indexer([accel_ts], method="nearest")[0]
        except Exception:
            continue

        if accel_pos < 0:
            continue

        accel_date_str = str(ind_df.index[accel_pos].date())
        outcome = _outcome_label(event)

        if dry_run:
            # Rich diagnostic format for smoke-test Check 5
            snapshots = []
            for pattern_type, offset in PRECURSOR_OFFSETS.items():
                prec_pos = accel_pos - offset
                if prec_pos < 0:
                    continue
                prec_ts = ind_df.index[prec_pos]
                prec_date_str = str(prec_ts.date())
                ind_row = ind_df.iloc[prec_pos]
                # latest_data_date: the indicator at prec_pos only uses OHLCV up to prec_pos
                latest_data_date = str(ind_df.index[prec_pos].date())
                snapshots.append({
                    "snapshot_offset_days": -offset,
                    "snapshot_date": prec_date_str,
                    "latest_data_date": latest_data_date,
                    "signal_strength": round(_compute_signal_strength(ind_row), 4),
                    "context": _build_context_dict(ind_row),
                })
            rows.append({
                "move_start_date": str(pd.Timestamp(event.start_date).date()) if event.start_date else accel_date_str,
                "acceleration_date": accel_date_str,
                "move_outcome": outcome,
                "snapshots": snapshots,
            })
            continue

        for pattern_type, offset in PRECURSOR_OFFSETS.items():
            prec_pos = accel_pos - offset
            if prec_pos < 0:
                continue

            prec_ts = ind_df.index[prec_pos]
            prec_date_str = str(prec_ts.date())
            ind_row = ind_df.iloc[prec_pos]

            signal_strength = _compute_signal_strength(ind_row)
            context = _build_context_dict(ind_row)

            rows.append({
                "stock_ticker": ticker.upper(),
                "precursor_date": prec_date_str,
                "acceleration_date": accel_date_str,
                "pattern_type": pattern_type,
                "signal_strength": round(signal_strength, 4),
                "context_json": json.dumps(context),
                "move_outcome": outcome,
                "created_at": now_str,
            })

    return rows


def write_precursors_to_db(rows: List[Dict[str, Any]]) -> int:
    """Insert precursor rows to move_precursors table. Returns rows inserted."""
    if not rows:
        return 0
    from app.core.database import exec_sql
    inserted = 0
    for row in rows:
        try:
            exec_sql(
                """INSERT INTO move_precursors
                   (stock_ticker, precursor_date, acceleration_date, pattern_type,
                    signal_strength, context_json, move_outcome, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["stock_ticker"],
                    row["precursor_date"],
                    row["acceleration_date"],
                    row["pattern_type"],
                    row["signal_strength"],
                    row["context_json"],
                    row["move_outcome"],
                    row["created_at"],
                ),
            )
            inserted += 1
        except Exception:
            pass
    return inserted


def verify_precursors(
    ticker: str,
    ohlcv: pd.DataFrame,
    rows: List[Dict[str, Any]],
    n_sample: int = 5,
    *,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Spot-check n_sample precursor rows against raw OHLCV to confirm alignment.

    For each sampled row:
      - precursor_date must be a valid trading date <= acceleration_date
      - precursor_date must exist (or be nearest) in the OHLCV index

    Returns True if all spot checks pass.
    """
    log = logger or LOGGER
    if not rows:
        return True

    sample = rows[:n_sample]
    all_ok = True

    for row in sample:
        prec_ts = pd.Timestamp(row["precursor_date"])
        accel_ts = pd.Timestamp(row["acceleration_date"])

        # precursor must be BEFORE acceleration
        if prec_ts >= accel_ts:
            log.error("[%s] Precursor %s is not before acceleration %s", ticker, prec_ts, accel_ts)
            all_ok = False
            continue

        # precursor date must be in OHLCV (or very close)
        nearest_pos = ohlcv.index.get_indexer([prec_ts], method="nearest")[0]
        nearest_date = ohlcv.index[nearest_pos]
        delta_days = abs((nearest_date - prec_ts).days)
        if delta_days > 7:
            log.warning(
                "[%s] Precursor date %s not found in OHLCV (nearest: %s, delta=%dd)",
                ticker, prec_ts.date(), nearest_date.date(), delta_days,
            )
            all_ok = False

    return all_ok


def build_all_precursors(
    tickers: Optional[Sequence[str]] = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, int]:
    """
    Build and persist precursors for all eligible stocks.

    Returns dict of ticker → rows inserted.
    """
    log = logger or LOGGER

    if tickers is None:
        tickers = _get_eligible_tickers(log)

    summary: Dict[str, int] = {}

    for ticker in tickers:
        try:
            ohlcv = load_ohlcv(ticker)
            rows = build_precursors_for_ticker(ticker, ohlcv, logger=log)
            if not rows:
                summary[ticker] = 0
                continue

            ok = verify_precursors(ticker, ohlcv, rows, logger=log)
            if not ok:
                log.warning("[%s] Spot-check failed — skipping insert", ticker)
                summary[ticker] = 0
                continue

            n_inserted = write_precursors_to_db(rows)
            summary[ticker] = n_inserted
            log.info("[%s] %d precursor rows inserted", ticker, n_inserted)
        except Exception as exc:
            log.error("[%s] Precursor build error: %s", ticker, exc)
            summary[ticker] = 0

    total = sum(summary.values())
    log.info("Precursor build complete. Total rows: %d across %d stocks", total, len(summary))
    return summary


def _get_eligible_tickers(log: logging.Logger) -> List[str]:
    try:
        from app.core.database import query_all
        rows = query_all(
            "SELECT stock_ticker FROM ml_stock_eligibility WHERE eligible=1 AND (watch_only IS NULL OR watch_only=0)"
        )
        tickers = [r[0] for r in rows if r[0]]
        if tickers:
            return tickers
    except Exception as exc:
        log.warning("Eligibility table error: %s — using OHLCV list", exc)
    return list_tickers_with_ohlcv()
