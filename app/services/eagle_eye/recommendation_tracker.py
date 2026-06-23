"""
Eagle Eye - Recommendation Tracker & Weekly Review System.

Captures every rating the system produces (append-only history),
tracks forward outcomes for BUY/SELL signals, and generates
weekly performance review reports.

Integration:
    Called once after each compute_all_ratings() run:
        from app.services.eagle_eye.recommendation_tracker import post_compute_snapshot
        post_compute_snapshot(run_id=run_id, run_date=run_date)

Tables created (idempotent):
    ee_rating_snapshots    - Daily append-only history of all ratings
    ee_signal_tracker      - Tracks BUY/SELL signals and their forward P&L
    ee_weekly_reviews      - Weekly summary reports (JSON blobs)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table DDL (idempotent, called from ensure_tracker_tables)
# ---------------------------------------------------------------------------

def ensure_tracker_tables() -> None:
    """Create tracking tables if they don't exist. Safe to call repeatedly."""
    from app.core.database import exec_sql

    # 1. Daily snapshot of every rating (append-only, never upserted)
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_rating_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date   TEXT    NOT NULL,
            run_id          TEXT,
            ticker          TEXT    NOT NULL,
            rating          TEXT    NOT NULL,
            confidence      REAL,
            stage           TEXT,
            last_price      REAL,
            entry_primary   REAL,
            stop_loss       REAL,
            tp1             REAL,
            tp2             REAL,
            risk_reward_ratio REAL,
            ml_score        REAL,
            liquidity_score REAL,
            trend_score     REAL,
            momentum_score  REAL,
            geometry_score  REAL,
            rr_score        REAL,
            risky_near_resistance INTEGER DEFAULT 0,
            indicators_json TEXT,
            created_at      INTEGER NOT NULL
        )
        """,
        (),
    )
    exec_sql(
        "CREATE INDEX IF NOT EXISTS idx_snapshots_date_ticker "
        "ON ee_rating_snapshots (snapshot_date, ticker)",
        (),
    )
    exec_sql(
        "CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_date "
        "ON ee_rating_snapshots (ticker, snapshot_date)",
        (),
    )

    # 2. Signal tracker - one row per BUY/SELL signal event
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_signal_tracker (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT    NOT NULL,
            signal_type     TEXT    NOT NULL,
            signal_date     TEXT    NOT NULL,
            signal_price    REAL,
            confidence      REAL,
            stage           TEXT,
            entry_primary   REAL,
            stop_loss       REAL,
            tp1             REAL,
            risk_reward_ratio REAL,
            ml_score        REAL,
            run_id          TEXT,
            status          TEXT    DEFAULT 'OPEN',
            price_1d        REAL,
            price_3d        REAL,
            price_5d        REAL,
            price_10d       REAL,
            price_20d       REAL,
            pnl_1d_pct      REAL,
            pnl_3d_pct      REAL,
            pnl_5d_pct      REAL,
            pnl_10d_pct     REAL,
            pnl_20d_pct     REAL,
            max_gain_20d    REAL,
            max_drawdown_20d REAL,
            hit_tp1         INTEGER DEFAULT 0,
            hit_stop        INTEGER DEFAULT 0,
            outcome_label   TEXT,
            closed_date     TEXT,
            closed_price    REAL,
            notes           TEXT,
            created_at      INTEGER NOT NULL
        )
        """,
        (),
    )
    exec_sql(
        "CREATE INDEX IF NOT EXISTS idx_signals_ticker_date "
        "ON ee_signal_tracker (ticker, signal_date)",
        (),
    )
    exec_sql(
        "CREATE INDEX IF NOT EXISTS idx_signals_status "
        "ON ee_signal_tracker (status)",
        (),
    )

    # 3. Weekly review summaries
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_weekly_reviews (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start      TEXT    NOT NULL,
            week_end        TEXT    NOT NULL,
            review_json     TEXT    NOT NULL,
            created_at      INTEGER NOT NULL
        )
        """,
        (),
    )
    exec_sql(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_reviews_week "
        "ON ee_weekly_reviews (week_start)",
        (),
    )


# ---------------------------------------------------------------------------
# 1. Post-compute snapshot capture
# ---------------------------------------------------------------------------

def post_compute_snapshot(run_id: str, run_date: str) -> dict:
    """
    Call this ONCE after compute_all_ratings() completes.

    Reads all current ratings from ee_ratings_cache and:
      a) Appends them to ee_rating_snapshots (history)
      b) Creates new signal tracker entries for fresh BUY/SELL signals
      c) Updates forward outcomes for previously-tracked signals

    Returns: {snapshots_saved, new_signals, outcomes_updated}
    """
    ensure_tracker_tables()

    stats = {"snapshots_saved": 0, "new_signals": 0, "outcomes_updated": 0}

    # --- Step A: Snapshot all current ratings ---
    stats["snapshots_saved"] = _capture_daily_snapshot(run_id, run_date)

    # --- Step B: Detect new BUY/SELL signals ---
    stats["new_signals"] = _detect_new_signals(run_id, run_date)

    # --- Step C: Update forward outcomes for open signals ---
    stats["outcomes_updated"] = _update_signal_outcomes(run_date)

    logger.info(
        "Recommendation tracker: %d snapshots, %d new signals, %d outcomes updated",
        stats["snapshots_saved"],
        stats["new_signals"],
        stats["outcomes_updated"],
    )
    return stats


def _capture_daily_snapshot(run_id: str, run_date: str) -> int:
    """Append all current ratings to the snapshot history table."""
    from app.core.database import exec_sql, query_all

    # Check if we already captured this date (idempotent)
    existing = query_all(
        "SELECT COUNT(*) as cnt FROM ee_rating_snapshots WHERE snapshot_date = ?",
        (run_date,),
    )
    if existing and existing[0].get("cnt", 0) > 0:
        logger.info("Snapshot for %s already exists, skipping", run_date)
        return 0

    rows = query_all(
        """
        SELECT ticker, rating, confidence, stage, last_price,
               entry_primary, stop_loss, tp1, tp2,
               risk_reward_ratio, ml_score, risky_near_resistance,
               indicators_json, run_id
        FROM ee_ratings_cache
        """,
        (),
    )
    if not rows:
        return 0

    now_ts = int(time.time())
    count = 0

    for row in rows:
        # Extract family scores from indicators_json if available
        ind = {}
        try:
            ind = json.loads(row.get("indicators_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            pass

        exec_sql(
            """
            INSERT INTO ee_rating_snapshots (
                snapshot_date, run_id, ticker, rating, confidence, stage,
                last_price, entry_primary, stop_loss, tp1, tp2,
                risk_reward_ratio, ml_score,
                liquidity_score, trend_score, momentum_score,
                geometry_score, rr_score,
                risky_near_resistance, indicators_json, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_date,
                run_id or row.get("run_id"),
                row["ticker"],
                row.get("rating"),
                _sf(row.get("confidence")),
                row.get("stage"),
                _sf(row.get("last_price")),
                _sf(row.get("entry_primary")),
                _sf(row.get("stop_loss")),
                _sf(row.get("tp1")),
                _sf(row.get("tp2")),
                _sf(row.get("risk_reward_ratio")),
                _sf(row.get("ml_score")),
                _sf(ind.get("liquidity")),
                _sf(ind.get("trend")),
                _sf(ind.get("momentum")),
                _sf(ind.get("geometry")),
                _sf(ind.get("risk_reward")),
                int(row.get("risky_near_resistance") or 0),
                row.get("indicators_json"),
                now_ts,
            ),
        )
        count += 1

    return count


def _detect_new_signals(run_id: str, run_date: str) -> int:
    """
    Compare today's ratings to yesterday's. A NEW signal is:
      - BUY that wasn't BUY yesterday (or first appearance)
      - SELL/STRONG_SELL/REDUCE that wasn't SELL yesterday
    """
    from app.core.database import query_all, exec_sql

    # Get today's ratings
    today = query_all(
        """
        SELECT ticker, rating, confidence, stage, last_price,
               entry_primary, stop_loss, tp1, risk_reward_ratio, ml_score
        FROM ee_rating_snapshots
        WHERE snapshot_date = ?
        """,
        (run_date,),
    )
    if not today:
        return 0

    # Get yesterday's ratings (most recent snapshot before today)
    yesterday = query_all(
        """
        SELECT ticker, rating
        FROM ee_rating_snapshots
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date)
            FROM ee_rating_snapshots
            WHERE snapshot_date < ?
        )
        """,
        (run_date,),
    )
    prev_ratings = {r["ticker"]: r["rating"] for r in (yesterday or [])}

    BUY_RATINGS = {"BUY", "STRONG_BUY"}
    SELL_RATINGS = {"SELL", "STRONG_SELL", "REDUCE"}

    now_ts = int(time.time())
    count = 0

    for row in today:
        ticker = row["ticker"]
        rating = row.get("rating", "")
        prev = prev_ratings.get(ticker, "")

        signal_type = None
        if rating in BUY_RATINGS and prev not in BUY_RATINGS:
            signal_type = "BUY"
        elif rating in SELL_RATINGS and prev not in SELL_RATINGS:
            signal_type = "SELL"

        if signal_type is None:
            continue

        # Check if we already tracked this exact signal today
        existing = query_all(
            """
            SELECT id FROM ee_signal_tracker
            WHERE ticker = ? AND signal_date = ? AND signal_type = ?
            """,
            (ticker, run_date, signal_type),
        )
        if existing:
            continue

        exec_sql(
            """
            INSERT INTO ee_signal_tracker (
                ticker, signal_type, signal_date, signal_price, confidence,
                stage, entry_primary, stop_loss, tp1, risk_reward_ratio,
                ml_score, run_id, status, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ticker,
                signal_type,
                run_date,
                _sf(row.get("last_price")),
                _sf(row.get("confidence")),
                row.get("stage"),
                _sf(row.get("entry_primary")),
                _sf(row.get("stop_loss")),
                _sf(row.get("tp1")),
                _sf(row.get("risk_reward_ratio")),
                _sf(row.get("ml_score")),
                run_id,
                "OPEN",
                now_ts,
            ),
        )
        count += 1

    return count


def _update_signal_outcomes(current_date: str) -> int:
    """
    For all OPEN signals, check if we now have forward price data.
    Updates 1d/3d/5d/10d/20d P&L, max gain/drawdown, TP1 hit, stop hit.
    """
    from app.core.database import query_all, exec_sql
    from app.services.eagle_eye.store import load_ohlcv

    open_signals = query_all(
        "SELECT * FROM ee_signal_tracker WHERE status = 'OPEN'",
        (),
    )
    if not open_signals:
        return 0

    count = 0
    for sig in open_signals:
        ticker = sig["ticker"]
        signal_price = sig.get("signal_price")
        signal_date = sig.get("signal_date")

        if not signal_price or signal_price <= 0 or not signal_date:
            continue

        df = load_ohlcv(ticker)
        if df is None or len(df) < 5:
            continue

        # Find the signal date field. OHLCV may expose date as a column or index.
        df = df.copy()
        date_col = None
        date_candidates = ["date", "bar_date", "timestamp", "datetime", "index"]
        for col in date_candidates:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            df = df.reset_index()
            for col in date_candidates:
                if col in df.columns:
                    date_col = col
                    break

        if date_col is None:
            logger.debug("Skipping %s signal outcome update: no date column/index", ticker)
            continue

        df = df.reset_index(drop=True)

        df[date_col] = df[date_col].astype(str).str[:10]
        signal_rows = df[df[date_col] == signal_date[:10]]
        if signal_rows.empty:
            # Try to find nearest date after signal
            mask = df[date_col] >= signal_date[:10]
            if mask.any():
                signal_idx = df[mask].index[0]
            else:
                continue
        else:
            signal_idx = signal_rows.index[0]

        future = df.iloc[signal_idx + 1:]
        if len(future) < 1:
            continue

        # Compute forward P&L at various horizons
        updates = {}
        for days, col_price, col_pnl in [
            (1, "price_1d", "pnl_1d_pct"),
            (3, "price_3d", "pnl_3d_pct"),
            (5, "price_5d", "pnl_5d_pct"),
            (10, "price_10d", "pnl_10d_pct"),
            (20, "price_20d", "pnl_20d_pct"),
        ]:
            if len(future) >= days:
                fwd_price = float(future.iloc[days - 1]["close"])
                fwd_pnl = ((fwd_price / signal_price) - 1) * 100
                updates[col_price] = round(fwd_price, 4)
                updates[col_pnl] = round(fwd_pnl, 4)

        # Max gain and max drawdown within 20 days
        fwd_20 = future.iloc[:20] if len(future) >= 20 else future
        if len(fwd_20) > 0:
            max_high = float(fwd_20["high"].max())
            min_low = float(fwd_20["low"].min())
            updates["max_gain_20d"] = round(
                ((max_high / signal_price) - 1) * 100, 4
            )
            updates["max_drawdown_20d"] = round(
                ((1 - min_low / signal_price)) * 100, 4
            )

        # Check TP1 hit and stop hit
        tp1 = sig.get("tp1")
        stop = sig.get("stop_loss")
        if tp1 and tp1 > 0 and len(fwd_20) > 0:
            if float(fwd_20["high"].max()) >= tp1:
                updates["hit_tp1"] = 1
        if stop and stop > 0 and len(fwd_20) > 0:
            if float(fwd_20["low"].min()) <= stop:
                updates["hit_stop"] = 1

        # Outcome label
        pnl_20 = updates.get("pnl_20d_pct")
        if pnl_20 is not None:
            if sig["signal_type"] == "BUY":
                if pnl_20 > 5:
                    updates["outcome_label"] = "STRONG_WIN"
                elif pnl_20 > 0:
                    updates["outcome_label"] = "WIN"
                elif pnl_20 > -3:
                    updates["outcome_label"] = "FLAT"
                else:
                    updates["outcome_label"] = "LOSS"
            else:  # SELL signal
                if pnl_20 < -5:
                    updates["outcome_label"] = "STRONG_WIN"
                elif pnl_20 < 0:
                    updates["outcome_label"] = "WIN"
                elif pnl_20 < 3:
                    updates["outcome_label"] = "FLAT"
                else:
                    updates["outcome_label"] = "LOSS"

        # Close signal if 20d data is available
        if updates.get("pnl_20d_pct") is not None:
            updates["status"] = "CLOSED"
            updates["closed_price"] = updates.get("price_20d")
            if len(future) >= 20:
                updates["closed_date"] = str(future.iloc[19][date_col])[:10]

        if updates:
            set_clauses = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [sig["id"]]
            exec_sql(
                f"UPDATE ee_signal_tracker SET {set_clauses} WHERE id = ?",
                tuple(values),
            )
            count += 1

    return count


# ---------------------------------------------------------------------------
# 2. Weekly review report generator
# ---------------------------------------------------------------------------

def generate_weekly_review(
    week_end_date: Optional[str] = None,
    save_to_db: bool = True,
) -> dict:
    """
    Generate a comprehensive weekly review report.

    Covers:
      - Rating distribution changes (this week vs last week)
      - All BUY/SELL signals issued this week
      - Outcome tracking for signals that matured (20d forward)
      - Hit rates (TP1, stop loss, win/loss)
      - Confidence calibration (are high-confidence signals better?)
      - Best and worst calls
      - Rating stability (churn rate)

    Args:
        week_end_date: ISO date string (defaults to today)
        save_to_db: persist to ee_weekly_reviews table

    Returns: the full review dict
    """
    ensure_tracker_tables()
    from app.core.database import exec_sql, query_all

    if week_end_date is None:
        week_end_date = datetime.now().strftime("%Y-%m-%d")

    we = datetime.strptime(week_end_date, "%Y-%m-%d")
    ws = we - timedelta(days=6)
    week_start = ws.strftime("%Y-%m-%d")
    prev_ws = (ws - timedelta(days=7)).strftime("%Y-%m-%d")
    prev_we = (ws - timedelta(days=1)).strftime("%Y-%m-%d")

    review = {
        "period": {"week_start": week_start, "week_end": week_end_date},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # --- Section 1: Rating distribution (latest snapshot this week vs last week) ---
    review["distribution"] = _compare_distributions(
        week_start, week_end_date, prev_ws, prev_we
    )

    # --- Section 2: Signals issued this week ---
    review["signals_this_week"] = _get_week_signals(week_start, week_end_date)

    # --- Section 3: Matured signal outcomes ---
    review["matured_outcomes"] = _get_matured_outcomes(week_start, week_end_date)

    # --- Section 4: Hit rates and performance ---
    review["performance"] = _compute_performance_stats()

    # --- Section 5: Confidence calibration ---
    review["confidence_calibration"] = _compute_confidence_calibration()

    # --- Section 6: Best and worst calls ---
    review["best_worst"] = _get_best_worst_calls()

    # --- Section 7: Rating stability / churn ---
    review["stability"] = _compute_rating_churn(week_start, week_end_date)

    # --- Section 8: Actionable summary ---
    review["summary"] = _build_executive_summary(review)

    # Persist
    if save_to_db:
        exec_sql(
            """
            INSERT INTO ee_weekly_reviews (week_start, week_end, review_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (week_start) DO UPDATE SET
                week_end = excluded.week_end,
                review_json = excluded.review_json,
                created_at = excluded.created_at
            """,
            (week_start, week_end_date, json.dumps(review, default=str), int(time.time())),
        )

    return review


# ---------------------------------------------------------------------------
# Review sub-computations
# ---------------------------------------------------------------------------

def _compare_distributions(
    ws: str, we: str, prev_ws: str, prev_we: str
) -> dict:
    """Compare rating distribution between this week and last week."""
    from app.core.database import query_all

    # Latest snapshot this week
    this_week = query_all(
        """
        SELECT rating, COUNT(*) as cnt
        FROM ee_rating_snapshots
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date)
            FROM ee_rating_snapshots
            WHERE snapshot_date BETWEEN ? AND ?
        )
        GROUP BY rating ORDER BY cnt DESC
        """,
        (ws, we),
    )

    # Latest snapshot last week
    last_week = query_all(
        """
        SELECT rating, COUNT(*) as cnt
        FROM ee_rating_snapshots
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date)
            FROM ee_rating_snapshots
            WHERE snapshot_date BETWEEN ? AND ?
        )
        GROUP BY rating ORDER BY cnt DESC
        """,
        (prev_ws, prev_we),
    )

    this_dist = {r["rating"]: r["cnt"] for r in (this_week or [])}
    last_dist = {r["rating"]: r["cnt"] for r in (last_week or [])}
    all_ratings = sorted(set(list(this_dist.keys()) + list(last_dist.keys())))

    comparison = []
    for rating in all_ratings:
        curr = this_dist.get(rating, 0)
        prev = last_dist.get(rating, 0)
        comparison.append({
            "rating": rating,
            "this_week": curr,
            "last_week": prev,
            "change": curr - prev,
        })

    return {
        "this_week": this_dist,
        "last_week": last_dist,
        "comparison": comparison,
    }


def _get_week_signals(ws: str, we: str) -> dict:
    """All BUY/SELL signals issued this week."""
    from app.core.database import query_all

    signals = query_all(
        """
        SELECT ticker, signal_type, signal_date, signal_price,
               confidence, stage, entry_primary, stop_loss, tp1,
               risk_reward_ratio, ml_score
        FROM ee_signal_tracker
        WHERE signal_date BETWEEN ? AND ?
        ORDER BY signal_date, ticker
        """,
        (ws, we),
    )

    buys = [s for s in (signals or []) if s["signal_type"] == "BUY"]
    sells = [s for s in (signals or []) if s["signal_type"] == "SELL"]

    return {
        "total": len(signals or []),
        "buys": len(buys),
        "sells": len(sells),
        "buy_signals": buys,
        "sell_signals": sells,
    }


def _get_matured_outcomes(ws: str, we: str) -> dict:
    """Signals that reached 20-day maturity this week."""
    from app.core.database import query_all

    matured = query_all(
        """
        SELECT ticker, signal_type, signal_date, signal_price,
               confidence, pnl_1d_pct, pnl_5d_pct, pnl_10d_pct, pnl_20d_pct,
               max_gain_20d, max_drawdown_20d, hit_tp1, hit_stop, outcome_label
        FROM ee_signal_tracker
        WHERE closed_date BETWEEN ? AND ?
        ORDER BY pnl_20d_pct DESC
        """,
        (ws, we),
    )

    return {
        "total_matured": len(matured or []),
        "signals": matured or [],
    }


def _compute_performance_stats() -> dict:
    """Overall hit rates across all closed signals."""
    from app.core.database import query_all

    closed = query_all(
        """
        SELECT signal_type, outcome_label, pnl_20d_pct, pnl_5d_pct,
               max_gain_20d, max_drawdown_20d, hit_tp1, hit_stop, confidence
        FROM ee_signal_tracker
        WHERE status = 'CLOSED'
        """,
        (),
    )
    if not closed:
        return {"total_closed": 0, "message": "No matured signals yet"}

    buys = [s for s in closed if s["signal_type"] == "BUY"]
    sells = [s for s in closed if s["signal_type"] == "SELL"]

    def _stats(signals: list) -> dict:
        if not signals:
            return {"count": 0}
        wins = [s for s in signals if s.get("outcome_label") in ("WIN", "STRONG_WIN")]
        losses = [s for s in signals if s.get("outcome_label") == "LOSS"]
        pnls = [s["pnl_20d_pct"] for s in signals if s.get("pnl_20d_pct") is not None]
        tp1_hits = sum(1 for s in signals if s.get("hit_tp1"))
        stop_hits = sum(1 for s in signals if s.get("hit_stop"))

        return {
            "count": len(signals),
            "win_rate": round(len(wins) / len(signals) * 100, 1) if signals else 0,
            "loss_rate": round(len(losses) / len(signals) * 100, 1) if signals else 0,
            "avg_pnl_20d": round(sum(pnls) / len(pnls), 2) if pnls else 0,
            "median_pnl_20d": round(sorted(pnls)[len(pnls) // 2], 2) if pnls else 0,
            "best_pnl_20d": round(max(pnls), 2) if pnls else 0,
            "worst_pnl_20d": round(min(pnls), 2) if pnls else 0,
            "tp1_hit_rate": round(tp1_hits / len(signals) * 100, 1) if signals else 0,
            "stop_hit_rate": round(stop_hits / len(signals) * 100, 1) if signals else 0,
            "avg_max_gain": round(
                sum(s["max_gain_20d"] for s in signals if s.get("max_gain_20d") is not None)
                / max(1, sum(1 for s in signals if s.get("max_gain_20d") is not None)),
                2,
            ),
            "avg_max_drawdown": round(
                sum(s["max_drawdown_20d"] for s in signals if s.get("max_drawdown_20d") is not None)
                / max(1, sum(1 for s in signals if s.get("max_drawdown_20d") is not None)),
                2,
            ),
        }

    return {
        "total_closed": len(closed),
        "buy_performance": _stats(buys),
        "sell_performance": _stats(sells),
    }


def _compute_confidence_calibration() -> dict:
    """
    Are higher-confidence signals actually better?
    Buckets: <50, 50-60, 60-70, 70-80, 80+
    """
    from app.core.database import query_all

    closed = query_all(
        """
        SELECT confidence, pnl_20d_pct, outcome_label, signal_type
        FROM ee_signal_tracker
        WHERE status = 'CLOSED' AND signal_type = 'BUY'
        """,
        (),
    )
    if not closed or len(closed) < 5:
        return {"message": "Not enough closed signals for calibration"}

    buckets = [
        ("below_50", 0, 50),
        ("50_60", 50, 60),
        ("60_70", 60, 70),
        ("70_80", 70, 80),
        ("above_80", 80, 101),
    ]

    result = {}
    for name, lo, hi in buckets:
        group = [s for s in closed if lo <= (s.get("confidence") or 0) < hi]
        if not group:
            result[name] = {"count": 0}
            continue
        wins = sum(1 for s in group if s.get("outcome_label") in ("WIN", "STRONG_WIN"))
        pnls = [s["pnl_20d_pct"] for s in group if s.get("pnl_20d_pct") is not None]
        result[name] = {
            "count": len(group),
            "win_rate": round(wins / len(group) * 100, 1),
            "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else 0,
        }

    return result


def _get_best_worst_calls() -> dict:
    """Top 5 best and worst BUY signals by 20d P&L."""
    from app.core.database import query_all

    best = query_all(
        """
        SELECT ticker, signal_date, signal_price, confidence, pnl_20d_pct,
               max_gain_20d, outcome_label
        FROM ee_signal_tracker
        WHERE status = 'CLOSED' AND signal_type = 'BUY'
              AND pnl_20d_pct IS NOT NULL
        ORDER BY pnl_20d_pct DESC LIMIT 5
        """,
        (),
    )
    worst = query_all(
        """
        SELECT ticker, signal_date, signal_price, confidence, pnl_20d_pct,
               max_drawdown_20d, outcome_label
        FROM ee_signal_tracker
        WHERE status = 'CLOSED' AND signal_type = 'BUY'
              AND pnl_20d_pct IS NOT NULL
        ORDER BY pnl_20d_pct ASC LIMIT 5
        """,
        (),
    )
    return {"best_buys": best or [], "worst_buys": worst or []}


def _compute_rating_churn(ws: str, we: str) -> dict:
    """How many stocks changed rating during the week (stability check)."""
    from app.core.database import query_all

    snapshots = query_all(
        """
        SELECT DISTINCT snapshot_date FROM ee_rating_snapshots
        WHERE snapshot_date BETWEEN ? AND ?
        ORDER BY snapshot_date
        """,
        (ws, we),
    )
    dates = [s["snapshot_date"] for s in (snapshots or [])]
    if len(dates) < 2:
        return {"days_with_snapshots": len(dates), "message": "Need 2+ days for churn"}

    first = dates[0]
    last = dates[-1]

    first_ratings = query_all(
        "SELECT ticker, rating FROM ee_rating_snapshots WHERE snapshot_date = ?",
        (first,),
    )
    last_ratings = query_all(
        "SELECT ticker, rating FROM ee_rating_snapshots WHERE snapshot_date = ?",
        (last,),
    )

    first_map = {r["ticker"]: r["rating"] for r in (first_ratings or [])}
    last_map = {r["ticker"]: r["rating"] for r in (last_ratings or [])}

    all_tickers = set(list(first_map.keys()) + list(last_map.keys()))
    changes = []
    for t in sorted(all_tickers):
        r1 = first_map.get(t, "N/A")
        r2 = last_map.get(t, "N/A")
        if r1 != r2:
            changes.append({"ticker": t, "from": r1, "to": r2})

    return {
        "days_with_snapshots": len(dates),
        "first_date": first,
        "last_date": last,
        "total_tickers": len(all_tickers),
        "changed_count": len(changes),
        "churn_rate_pct": round(len(changes) / max(1, len(all_tickers)) * 100, 1),
        "changes": changes[:30],  # cap at 30 for readability
    }


def _build_executive_summary(review: dict) -> dict:
    """Build a plain-language summary from the review data."""
    perf = review.get("performance", {})
    sigs = review.get("signals_this_week", {})
    dist = review.get("distribution", {})
    stab = review.get("stability", {})

    buy_perf = perf.get("buy_performance", {})
    total_closed = perf.get("total_closed", 0)

    lines = []

    # Signals issued
    if sigs.get("total", 0) > 0:
        lines.append(
            f"Issued {sigs['buys']} BUY and {sigs['sells']} SELL signals this week."
        )
    else:
        lines.append("No new signals issued this week.")

    # Performance
    if total_closed > 0 and buy_perf.get("count", 0) > 0:
        lines.append(
            f"BUY track record: {buy_perf['win_rate']}% win rate across "
            f"{buy_perf['count']} closed signals, avg {buy_perf['avg_pnl_20d']}% "
            f"return at 20 days."
        )
        if buy_perf.get("tp1_hit_rate", 0) > 0:
            lines.append(f"TP1 hit rate: {buy_perf['tp1_hit_rate']}%.")

    # Distribution shift
    this_buys = dist.get("this_week", {}).get("BUY", 0)
    last_buys = dist.get("last_week", {}).get("BUY", 0)
    if this_buys != last_buys:
        direction = "up" if this_buys > last_buys else "down"
        lines.append(
            f"BUY count {direction} from {last_buys} to {this_buys} "
            f"({this_buys - last_buys:+d})."
        )

    # Stability
    churn = stab.get("churn_rate_pct", 0)
    if churn > 20:
        lines.append(f"Rating churn is high ({churn}%) - review for instability.")
    elif churn > 0:
        lines.append(f"Rating churn: {churn}% of stocks changed rating.")

    return {
        "headline": lines[0] if lines else "Weekly review generated.",
        "details": lines,
    }


# ---------------------------------------------------------------------------
# 3. Utility: Export review as formatted report
# ---------------------------------------------------------------------------

def export_weekly_report(
    week_end_date: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate the review and export as a formatted markdown report.

    Returns the markdown string. Also writes to output_path if provided.
    """
    review = generate_weekly_review(week_end_date)

    lines = []
    lines.append("# Eagle Eye - Weekly Performance Review")
    lines.append(f"**Period:** {review['period']['week_start']} to {review['period']['week_end']}")
    lines.append(f"**Generated:** {review['generated_at']}")
    lines.append("")

    # Executive summary
    summary = review.get("summary", {})
    lines.append("## Executive Summary")
    for detail in summary.get("details", []):
        lines.append(f"- {detail}")
    lines.append("")

    # Distribution
    dist = review.get("distribution", {})
    comp = dist.get("comparison", [])
    if comp:
        lines.append("## Rating Distribution")
        lines.append("| Rating | This Week | Last Week | Change |")
        lines.append("|--------|-----------|-----------|--------|")
        for c in comp:
            change_str = f"{c['change']:+d}" if c["change"] != 0 else "-"
            lines.append(
                f"| {c['rating']} | {c['this_week']} | {c['last_week']} | {change_str} |"
            )
        lines.append("")

    # Signals this week
    sigs = review.get("signals_this_week", {})
    if sigs.get("buy_signals"):
        lines.append("## BUY Signals Issued")
        lines.append("| Ticker | Date | Price | Confidence | Stage | R:R | TP1 |")
        lines.append("|--------|------|-------|------------|-------|-----|-----|")
        for s in sigs["buy_signals"]:
            lines.append(
                f"| {s['ticker']} | {s['signal_date']} | {s.get('signal_price', '-')} "
                f"| {s.get('confidence', '-')} | {s.get('stage', '-')} "
                f"| {s.get('risk_reward_ratio', '-')} | {s.get('tp1', '-')} |"
            )
        lines.append("")

    if sigs.get("sell_signals"):
        lines.append("## SELL Signals Issued")
        lines.append("| Ticker | Date | Price | Confidence | Stage |")
        lines.append("|--------|------|-------|------------|-------|")
        for s in sigs["sell_signals"]:
            lines.append(
                f"| {s['ticker']} | {s['signal_date']} | {s.get('signal_price', '-')} "
                f"| {s.get('confidence', '-')} | {s.get('stage', '-')} |"
            )
        lines.append("")

    # Performance
    perf = review.get("performance", {})
    bp = perf.get("buy_performance", {})
    if bp.get("count", 0) > 0:
        lines.append("## BUY Signal Performance (all-time)")
        lines.append(f"- **Total closed:** {bp['count']}")
        lines.append(f"- **Win rate:** {bp['win_rate']}%")
        lines.append(f"- **Avg 20d return:** {bp['avg_pnl_20d']}%")
        lines.append(f"- **Best:** {bp['best_pnl_20d']}% / **Worst:** {bp['worst_pnl_20d']}%")
        lines.append(f"- **TP1 hit rate:** {bp['tp1_hit_rate']}%")
        lines.append(f"- **Stop hit rate:** {bp['stop_hit_rate']}%")
        lines.append(f"- **Avg max gain (20d):** {bp['avg_max_gain']}%")
        lines.append(f"- **Avg max drawdown (20d):** {bp['avg_max_drawdown']}%")
        lines.append("")

    # Confidence calibration
    cal = review.get("confidence_calibration", {})
    if not cal.get("message"):
        lines.append("## Confidence Calibration (BUY signals)")
        lines.append("| Confidence Bucket | Count | Win Rate | Avg P&L |")
        lines.append("|-------------------|-------|----------|---------|")
        for bucket, data in cal.items():
            if data.get("count", 0) > 0:
                label = bucket.replace("_", "-")
                lines.append(
                    f"| {label} | {data['count']} | {data['win_rate']}% | {data['avg_pnl']}% |"
                )
        lines.append("")

    # Best/worst
    bw = review.get("best_worst", {})
    if bw.get("best_buys"):
        lines.append("## Top Calls")
        for s in bw["best_buys"][:3]:
            lines.append(
                f"- **{s['ticker']}** ({s['signal_date']}): "
                f"+{s['pnl_20d_pct']}% in 20d (conf {s['confidence']})"
            )
    if bw.get("worst_buys"):
        lines.append("")
        lines.append("## Worst Calls")
        for s in bw["worst_buys"][:3]:
            lines.append(
                f"- **{s['ticker']}** ({s['signal_date']}): "
                f"{s['pnl_20d_pct']}% in 20d (conf {s['confidence']})"
            )
    lines.append("")

    # Stability
    stab = review.get("stability", {})
    if stab.get("changed_count", 0) > 0:
        lines.append("## Rating Changes This Week")
        lines.append(f"**{stab['changed_count']}** of {stab['total_tickers']} stocks "
                      f"changed rating ({stab['churn_rate_pct']}% churn)")
        if stab.get("changes"):
            lines.append("")
            lines.append("| Ticker | From | To |")
            lines.append("|--------|------|----|")
            for ch in stab["changes"][:20]:
                lines.append(f"| {ch['ticker']} | {ch['from']} | {ch['to']} |")
        lines.append("")

    md = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

    return md


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sf(val) -> Optional[float]:
    """Safe float conversion."""
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None
