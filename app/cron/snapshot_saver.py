"""
Snapshot Saver — scheduled job that saves a daily portfolio snapshot.

Runs after the price update job so snapshots reflect fresh prices.
Reuses the same logic as POST /api/v1/tracker/save-snapshot but
bypasses JWT auth (runs as an internal system job).
"""

import logging
import time
from datetime import date

from app.core.database import query_df, query_val, exec_sql
from app.services.portfolio_service import get_total_portfolio_value
from app.services.fx_service import convert_to_kwd

logger = logging.getLogger(__name__)

_last_run: dict = {}


def _sum_deposits_kwd(uid: int, deposit_date: str) -> float:
    """Sum active deposits for a specific date, converting each to KWD."""
    rows = query_df(
        """SELECT amount, COALESCE(currency, 'KWD') AS currency
           FROM cash_deposits
           WHERE user_id = ? AND deposit_date = ?
             AND COALESCE(is_deleted, 0) = 0""",
        (uid, deposit_date),
    )
    if rows.empty:
        return 0.0
    total = 0.0
    for _, r in rows.iterrows():
        total += convert_to_kwd(float(r["amount"]), str(r["currency"]))
    return round(total, 3)


def run_snapshot_save(user_id: int = 1) -> dict:
    """
    Save today's portfolio snapshot (identical to the manual save button).

    Steps:
      1. Calculate live portfolio value from holdings
      2. Compute deposit cash, daily movement, derived metrics
      3. Upsert into portfolio_snapshots
      4. Recalculate all snapshots for consistency

    Returns dict with run info.
    """
    logger.info("📸 Scheduled snapshot save starting (user_id=%d)…", user_id)

    try:
        today = str(date.today())

        # 1. Live portfolio value
        unified = get_total_portfolio_value(user_id)
        portfolio_value = unified["total_value_kwd"]

        # 2. Deposit cash for today
        deposit_cash = float(_sum_deposits_kwd(user_id, today))

        # 3. Previous snapshot
        prev = query_df(
            """SELECT portfolio_value, accumulated_cash FROM portfolio_snapshots
               WHERE user_id = ? AND snapshot_date < ?
               ORDER BY snapshot_date DESC LIMIT 1""",
            (user_id, today),
        )
        prev_val = float(prev.iloc[0]["portfolio_value"]) if not prev.empty else 0.0
        prev_acc_raw = prev.iloc[0]["accumulated_cash"] if not prev.empty else 0.0
        prev_accumulated = float(prev_acc_raw) if prev_acc_raw is not None and not (
            isinstance(prev_acc_raw, float) and prev_acc_raw != prev_acc_raw
        ) else 0.0
        daily_movement = round(portfolio_value - prev_val, 3) if prev_val > 0 else 0.0

        # 4. Beginning difference
        first_val = query_val(
            """SELECT portfolio_value FROM portfolio_snapshots
               WHERE user_id = ? ORDER BY snapshot_date ASC LIMIT 1""",
            (user_id,),
        )
        baseline_value = float(first_val) if first_val else portfolio_value
        beginning_difference = round(portfolio_value - baseline_value, 3)

        # 5. Accumulated cash
        accumulated_cash = round(prev_accumulated + deposit_cash, 3)

        # 6. Derived metrics
        net_gain = round(beginning_difference - accumulated_cash, 3)
        roi_percent = round(net_gain / accumulated_cash * 100, 2) if accumulated_cash > 0 else 0.0
        change_percent = round(daily_movement / prev_val * 100, 2) if prev_val else 0.0

        now = int(time.time())

        # 7. Upsert
        existing_id = query_val(
            "SELECT id FROM portfolio_snapshots WHERE user_id = ? AND snapshot_date = ?",
            (user_id, today),
        )

        if existing_id:
            exec_sql(
                """UPDATE portfolio_snapshots SET
                    portfolio_value = ?, daily_movement = ?,
                    beginning_difference = ?, deposit_cash = ?,
                    accumulated_cash = ?, net_gain = ?,
                    roi_percent = ?, change_percent = ?,
                    created_at = ?
                   WHERE id = ? AND user_id = ?""",
                (portfolio_value, daily_movement,
                 beginning_difference, deposit_cash,
                 accumulated_cash, net_gain,
                 roi_percent, change_percent, now,
                 existing_id, user_id),
            )
            action = "updated"
        else:
            exec_sql(
                """INSERT INTO portfolio_snapshots
                   (user_id, snapshot_date, portfolio_value, daily_movement,
                    beginning_difference, deposit_cash, accumulated_cash,
                    net_gain, roi_percent, change_percent, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, today, portfolio_value, daily_movement,
                 beginning_difference, deposit_cash, accumulated_cash,
                 net_gain, roi_percent, change_percent, now),
            )
            action = "created"

        # 8. Recalculate all snapshots
        from app.api.v1.tracker import recalculate_all_snapshots
        recalculate_all_snapshots(user_id)

        run_info = {
            "timestamp": now,
            "snapshot_date": today,
            "portfolio_value": portfolio_value,
            "action": action,
            "success": True,
        }
        logger.info(
            "📸 Scheduled snapshot %s for %s — value: %.3f KWD",
            action, today, portfolio_value,
        )

    except Exception as exc:
        run_info = {
            "timestamp": int(time.time()),
            "error": str(exc),
            "success": False,
        }
        logger.error("📸 Scheduled snapshot save FAILED: %s", exc)

    _last_run.update(run_info)
    return run_info


def get_last_run() -> dict:
    """Return info about the last snapshot save run."""
    return dict(_last_run)
