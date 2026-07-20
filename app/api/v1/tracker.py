"""
Portfolio Tracker API v1 — save/delete daily portfolio snapshots.

Mirrors the Streamlit ``ui_portfolio_tracker()`` logic:
  - Save today's snapshot: calculates live portfolio value from
    build_portfolio_table(), adds manual cash, then inserts into
    portfolio_snapshots.
  - Delete all snapshots for the authenticated user.

NOTE: The read endpoints (GET /analytics/snapshots) already exist in
      analytics.py.  This router only handles write operations.
"""

import time
import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.exceptions import BadRequestError
from app.core.database import query_df, query_val, exec_sql, exec_sql_fetchone, add_column_if_missing, transaction
from app.services.portfolio_service import build_portfolio_table, get_total_portfolio_value
from app.services.fx_service import convert_to_kwd

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tracker", tags=["Portfolio Tracker"])


# ── Reusable helpers (called from cash.py too) ───────────────────────

def _ensure_deposit_adjustment_col() -> None:
    """Add deposit_adjustment column to portfolio_snapshots if missing."""
    try:
        add_column_if_missing("portfolio_snapshots", "deposit_adjustment", "REAL DEFAULT 0")
    except Exception as exc:
        logger.warning("Could not ensure deposit_adjustment column: %s", exc)


def _cash_flow_kwd(row) -> float:
    amount = float(row["amount"] or 0.0)
    source = str(row.get("source", "deposit") or "deposit").strip().lower()
    currency = str(row.get("currency", "KWD") or "KWD").strip().upper()
    fx_rate = row.get("fx_rate_at_deposit")
    if source == "withdrawal":
        amount = -amount
    if currency == "USD" and fx_rate:
        return amount * float(fx_rate)
    return convert_to_kwd(amount, currency)


def recalculate_all_snapshots(uid: int) -> int:
    """
    Recalculate all derived columns for every snapshot of a user.

    Phase 0 — Undo deposit_adjustment corruption (one-time repair):
      Previous code wrongly added deposit amounts to portfolio_value via
      a ``deposit_adjustment`` column.  Reverse that: subtract the stored
      adjustment from portfolio_value and zero the column.

    Phase 1 — Deposit reconciliation:
      Re-sync deposit_cash from cash_deposits for every snapshot date.

    Phase 2 — Derived metrics (Streamlit formulas):
      1. daily_movement = current − previous value
      2. beginning_difference = current − first snapshot value
      3. accumulated_cash = running sum of deposit_cash
      4. net_gain = beginning_difference − accumulated_cash
      5. change_percent = daily_movement / previous × 100
      6. roi_percent = net_gain / accumulated_cash × 100

    Returns the number of snapshots updated.
    """
    _ensure_deposit_adjustment_col()
    with transaction():
        # ── Phase 0: Undo deposit_adjustment corruption ──────────────────
        exec_sql(
            """UPDATE portfolio_snapshots
               SET portfolio_value = portfolio_value - COALESCE(deposit_adjustment, 0),
                   deposit_adjustment = 0
               WHERE user_id = ? AND COALESCE(deposit_adjustment, 0) != 0""",
            (uid,),
        )

        dep_dates = query_df(
            """SELECT DISTINCT deposit_date
               FROM cash_deposits
               WHERE user_id = ?
                 AND COALESCE(include_in_analysis, 1) = 1
                 AND COALESCE(is_deleted, 0) = 0""",
            (uid,),
        )
        deposit_by_date = {
            dd["deposit_date"]: _sum_deposits_kwd(uid, dd["deposit_date"])
            for _, dd in dep_dates.iterrows()
        }

        df = query_df(
            """SELECT id, snapshot_date, portfolio_value
               FROM portfolio_snapshots
               WHERE user_id = ?
               ORDER BY snapshot_date ASC, id ASC""",
            (uid,),
        )
        if df.empty:
            return 0

        first_value = float(df.iloc[0]["portfolio_value"])
        first_flow = round(float(deposit_by_date.get(df.iloc[0]["snapshot_date"], 0.0)), 3)
        prev_value = 0.0
        running_accumulated = 0.0
        updated = 0

        for idx, row in df.iterrows():
            snap_id = int(row["id"])
            snap_date = row["snapshot_date"]
            pv = float(row["portfolio_value"])
            deposit_cash = round(float(deposit_by_date.get(snap_date, 0.0)), 3)

            if idx == 0:
                daily_movement = 0.0
                change_percent = 0.0
            else:
                daily_movement = round(pv - prev_value - deposit_cash, 3)
                change_percent = round(daily_movement / prev_value * 100, 2) if prev_value > 0 else 0.0

            beginning_difference = round(pv - first_value, 3)
            running_accumulated += deposit_cash
            accumulated_cash = round(running_accumulated, 3)
            net_external_after_baseline = running_accumulated - first_flow
            net_gain = round(beginning_difference - net_external_after_baseline, 3)
            roi_base = first_value + max(net_external_after_baseline, 0.0)
            roi_percent = round(net_gain / roi_base * 100, 2) if roi_base > 0 and idx != 0 else 0.0

            exec_sql(
                """UPDATE portfolio_snapshots
                   SET deposit_cash = ?, daily_movement = ?, beginning_difference = ?,
                       accumulated_cash = ?, net_gain = ?,
                       change_percent = ?, roi_percent = ?
                   WHERE id = ? AND user_id = ?""",
                (deposit_cash, daily_movement, beginning_difference,
                 accumulated_cash, net_gain,
                 change_percent, roi_percent,
                 snap_id, uid),
            )
            prev_value = pv
            updated += 1

        return updated


def _sum_deposits_kwd(uid: int, deposit_date: str) -> float:
    """Sum active deposits for a specific date, converting each to KWD.

    Matches Streamlit which always stores deposit_cash in KWD
    (``convert_to_kwd(amount, currency)``).
    """
    rows = query_df(
        """SELECT amount, COALESCE(currency, 'KWD') AS currency,
                  COALESCE(source, 'deposit') AS source,
                  fx_rate_at_deposit
           FROM cash_deposits
           WHERE user_id = ? AND deposit_date = ?
             AND COALESCE(include_in_analysis, 1) = 1
             AND COALESCE(is_deleted, 0) = 0""",
        (uid, deposit_date),
    )
    if rows.empty:
        return 0.0
    total = 0.0
    for _, r in rows.iterrows():
        total += _cash_flow_kwd(r)
    return round(total, 3)


def _calculate_accumulated_cash(uid: int) -> float:
    """Return total portfolio cash across all portfolios, converted to KWD.

    Mirrors Streamlit's ``calculate_accumulated_cash()`` which reads the
    current balance from ``portfolio_cash``.
    """
    cash_df = query_df(
        """SELECT balance, COALESCE(currency, 'KWD') AS currency
           FROM portfolio_cash
           WHERE user_id = ?""",
        (uid,),
    )
    if cash_df.empty:
        return 0.0
    total = 0.0
    for _, cr in cash_df.iterrows():
        bal = float(cr["balance"]) if cr["balance"] else 0.0
        total += convert_to_kwd(bal, str(cr["currency"]))
    return round(total, 3)


def _create_snapshot_for_deposit(uid: int, deposit_date: str, deposit_kwd: float) -> None:
    """Create a new snapshot on a date triggered by a deposit.

    Mirrors Streamlit's behaviour: when a deposit is added for a date
    without a snapshot, a full snapshot is created with the LIVE
    portfolio value so that TWR calculations remain accurate.
    """
    import time as _time

    # 1. Live portfolio value — single source of truth
    unified = get_total_portfolio_value(uid)
    portfolio_value = unified["total_value_kwd"]

    # 2. Previous snapshot for daily_movement
    prev = query_df(
        """SELECT portfolio_value FROM portfolio_snapshots
           WHERE user_id = ? AND snapshot_date < ?
           ORDER BY snapshot_date DESC LIMIT 1""",
        (uid, deposit_date),
    )
    prev_val = float(prev.iloc[0]["portfolio_value"]) if not prev.empty else 0.0
    daily_movement = round(portfolio_value - prev_val, 3) if prev_val > 0 else 0.0

    # 3. Beginning difference (vs first snapshot)
    first_val = query_val(
        """SELECT portfolio_value FROM portfolio_snapshots
           WHERE user_id = ? ORDER BY snapshot_date ASC LIMIT 1""",
        (uid,),
    )
    baseline_value = float(first_val) if first_val else portfolio_value
    beginning_difference = round(portfolio_value - baseline_value, 3)

    # 4. Accumulated cash from portfolio_cash (matches Streamlit calculate_accumulated_cash)
    accumulated_cash = _calculate_accumulated_cash(uid)

    # 5. Derived metrics
    net_gain = round(beginning_difference - accumulated_cash, 3)
    roi_percent = round(net_gain / accumulated_cash * 100, 2) if accumulated_cash > 0 else 0.0
    change_percent = round(daily_movement / prev_val * 100, 2) if prev_val > 0 else 0.0

    exec_sql(
        """INSERT INTO portfolio_snapshots
           (user_id, snapshot_date, portfolio_value, daily_movement,
            beginning_difference, deposit_cash, accumulated_cash,
            net_gain, roi_percent, change_percent, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (uid, deposit_date, portfolio_value, daily_movement,
         beginning_difference, deposit_kwd, accumulated_cash,
         net_gain, roi_percent, change_percent, int(_time.time())),
    )
    logger.info(
        "Created snapshot for deposit date %s (user %s): value=%s, deposit_cash=%s",
        deposit_date, uid, portfolio_value, deposit_kwd,
    )


def sync_deposit_to_snapshot(uid: int, deposit_date: str) -> None:
    """
    Re-sum all active deposits for *deposit_date* (converted to KWD),
    write the total into the matching snapshot's ``deposit_cash`` column,
    then recalculate all snapshots so accumulated_cash / net_gain /
    roi_percent cascade.

    Mirrors Streamlit's deposit-triggered snapshot sync:
      - If a snapshot exists for the date → update its deposit_cash
        and recalculate all snapshots.
      - If no snapshot exists → **create one** with the live portfolio
        value (matching Streamlit's behaviour for TWR accuracy).
    """
    # Total deposits for this specific date (KWD-converted)
    day_total_kwd = _sum_deposits_kwd(uid, deposit_date)

    # Check if a snapshot exists for this date
    snap_id = query_val(
        "SELECT id FROM portfolio_snapshots WHERE user_id = ? AND snapshot_date = ?",
        (uid, deposit_date),
    )

    if snap_id:
        # Update the deposit_cash column for this snapshot (KWD)
        exec_sql(
            "UPDATE portfolio_snapshots SET deposit_cash = ? WHERE id = ? AND user_id = ?",
            (day_total_kwd, snap_id, uid),
        )

        # Recalculate all snapshots — this reconciles portfolio_value
        # using created_at comparison so only stale snapshots are adjusted.
        recalculate_all_snapshots(uid)
        logger.info("Synced deposit_cash=%s KWD for snapshot %s (user %s)", day_total_kwd, deposit_date, uid)
    else:
        # Create a new snapshot for this deposit date (matches Streamlit)
        try:
            _create_snapshot_for_deposit(uid, deposit_date, day_total_kwd)
            # Recalculate all snapshots so the new one integrates properly
            recalculate_all_snapshots(uid)
        except Exception as exc:
            logger.warning("Failed to create snapshot for deposit date %s: %s", deposit_date, exc)


# ── Schema ───────────────────────────────────────────────────────────

class SnapshotManual(BaseModel):
    """Manually supply snapshot values (optional override)."""
    snapshot_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    portfolio_value: Optional[float] = None
    deposit_cash: Optional[float] = None
    notes: Optional[str] = None


# ── Save today's snapshot (live calc) ────────────────────────────────

@router.post("/save-snapshot", status_code=201)
async def save_snapshot(
    body: Optional[SnapshotManual] = None,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Calculate and save today's portfolio snapshot.

    1. Iterates all portfolios → build_portfolio_table() → sum market values (KWD)
    2. Adds manual cash from portfolio_cash table
    3. Computes deposit_cash from cash_deposits
    4. Calculates daily_movement vs previous snapshot
    5. Inserts into portfolio_snapshots

    If body.portfolio_value is provided, it overrides the live calculation.
    """
    uid = current_user.user_id
    today = (body.snapshot_date if body and body.snapshot_date else str(date.today()))

    # ── 1. Live portfolio value — single source of truth ──────────
    if body and body.portfolio_value is not None:
        portfolio_value = body.portfolio_value
    else:
        unified = get_total_portfolio_value(uid)
        portfolio_value = unified["total_value_kwd"]

    # ── 2. Deposit cash (today's deposits only, converted to KWD) ──
    if body and body.deposit_cash is not None:
        deposit_cash = body.deposit_cash
    else:
        deposit_cash = _sum_deposits_kwd(uid, today)
    deposit_cash = float(deposit_cash)

    # ── 3. Previous snapshot for daily_movement ──────────────────────
    prev = query_df(
        """SELECT portfolio_value, accumulated_cash FROM portfolio_snapshots
           WHERE user_id = ? AND snapshot_date < ?
           ORDER BY snapshot_date DESC LIMIT 1""",
        (uid, today),
    )
    prev_val = float(prev.iloc[0]["portfolio_value"]) if not prev.empty else 0.0
    prev_acc_raw = prev.iloc[0]["accumulated_cash"] if not prev.empty else 0.0
    prev_accumulated = float(prev_acc_raw) if prev_acc_raw is not None and not (isinstance(prev_acc_raw, float) and prev_acc_raw != prev_acc_raw) else 0.0
    daily_movement = round(portfolio_value - prev_val, 3) if prev_val > 0 else 0.0

    # ── 4. Beginning difference (vs first ever snapshot) ─────────────
    first_val = query_val(
        """SELECT portfolio_value FROM portfolio_snapshots
           WHERE user_id = ? ORDER BY snapshot_date ASC LIMIT 1""",
        (uid,),
    )
    baseline_value = float(first_val) if first_val else portfolio_value
    beginning_difference = round(portfolio_value - baseline_value, 3)

    # ── 5. Accumulated cash (prev + today's deposit) ─────────────────
    accumulated_cash = round(prev_accumulated + deposit_cash, 3)

    # ── 6. Derived metrics (Streamlit formulas) ──────────────────────
    net_gain = round(beginning_difference - accumulated_cash, 3)
    roi_percent = round(net_gain / accumulated_cash * 100, 2) if accumulated_cash > 0 else 0.0
    change_percent = round(daily_movement / prev_val * 100, 2) if prev_val else 0.0

    now = int(time.time())

    # ── 7. Atomic upsert (replace if same date) ──────────────────────
    existing_id = query_val(
        "SELECT id FROM portfolio_snapshots WHERE user_id = ? AND snapshot_date = ?",
        (uid, today),
    )
    with transaction():
        upserted = exec_sql_fetchone(
            """INSERT INTO portfolio_snapshots
               (user_id, snapshot_date, portfolio_value, daily_movement,
                beginning_difference, deposit_cash, accumulated_cash,
                net_gain, roi_percent, change_percent, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, snapshot_date) DO UPDATE SET
                    portfolio_value = excluded.portfolio_value,
                    daily_movement = excluded.daily_movement,
                    beginning_difference = excluded.beginning_difference,
                    deposit_cash = excluded.deposit_cash,
                    accumulated_cash = excluded.accumulated_cash,
                    net_gain = excluded.net_gain,
                    roi_percent = excluded.roi_percent,
                    change_percent = excluded.change_percent,
                    created_at = excluded.created_at
               RETURNING id""",
            (uid, today, portfolio_value, daily_movement,
             beginning_difference, deposit_cash, accumulated_cash,
             net_gain, roi_percent, change_percent, now),
        )
    snapshot_id = upserted[0]
    action = "updated" if existing_id else "created"

    # ── 8. Auto-recalculate ALL snapshots (Streamlit parity) ─────────
    # Mirrors the Streamlit recalculate button: rebuilds deposit_cash
    # from cash_deposits, recomputes accumulated_cash as a running sum,
    # and cascades net_gain / roi_percent across the full chain.
    try:
        recalculate_all_snapshots(uid)
        logger.info("Auto-recalculated snapshots after save for user %d", uid)
    except Exception as exc:
        logger.warning("Auto-recalculate after save failed: %s", exc)

    # ── 9. Re-read final values after recalculate ────────────────────
    final = query_df(
        """SELECT portfolio_value, daily_movement, beginning_difference,
                  deposit_cash, accumulated_cash, net_gain,
                  roi_percent, change_percent
           FROM portfolio_snapshots
           WHERE id = ? AND user_id = ?""",
        (snapshot_id, uid),
    )
    if not final.empty:
        r = final.iloc[0]
        portfolio_value = float(r["portfolio_value"])
        daily_movement = float(r["daily_movement"]) if r["daily_movement"] else 0.0
        beginning_difference = float(r["beginning_difference"]) if r["beginning_difference"] else 0.0
        deposit_cash = float(r["deposit_cash"]) if r["deposit_cash"] else 0.0
        accumulated_cash = float(r["accumulated_cash"]) if r["accumulated_cash"] else 0.0
        net_gain = float(r["net_gain"]) if r["net_gain"] else 0.0
        roi_percent = float(r["roi_percent"]) if r["roi_percent"] else 0.0
        change_percent = float(r["change_percent"]) if r["change_percent"] else 0.0

    return {
        "status": "ok",
        "data": {
            "id": snapshot_id,
            "snapshot_date": today,
            "portfolio_value": portfolio_value,
            "daily_movement": daily_movement,
            "beginning_difference": beginning_difference,
            "deposit_cash": deposit_cash,
            "accumulated_cash": accumulated_cash,
            "net_gain": net_gain,
            "roi_percent": roi_percent,
            "change_percent": change_percent,
            "action": action,
            "message": f"Snapshot {action} for {today}",
        },
    }


# ── Delete all snapshots ─────────────────────────────────────────────

@router.delete("/snapshots")
async def delete_all_snapshots(
    current_user: TokenData = Depends(get_current_user),
):
    """
    Permanently delete ALL portfolio snapshots for the authenticated user.
    This cannot be undone.
    """
    uid = current_user.user_id

    count = query_val(
        "SELECT COUNT(*) FROM portfolio_snapshots WHERE user_id = ?",
        (uid,),
    ) or 0

    exec_sql(
        "DELETE FROM portfolio_snapshots WHERE user_id = ?",
        (uid,),
    )

    return {
        "status": "ok",
        "data": {
            "deleted_count": count,
            "message": f"Deleted {count} snapshots",
        },
    }


# ── Delete single snapshot ───────────────────────────────────────────

@router.delete("/snapshots/{snapshot_id}")
async def delete_snapshot(
    snapshot_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Delete a single portfolio snapshot by ID."""
    from app.core.exceptions import NotFoundError

    existing = query_val(
        "SELECT id FROM portfolio_snapshots WHERE id = ? AND user_id = ?",
        (snapshot_id, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("Snapshot", snapshot_id)

    exec_sql(
        "DELETE FROM portfolio_snapshots WHERE id = ? AND user_id = ?",
        (snapshot_id, current_user.user_id),
    )

    return {"status": "ok", "data": {"id": snapshot_id, "message": "Snapshot deleted"}}


# ── Recalculate all snapshot metrics ─────────────────────────────────

@router.post("/recalculate")
async def recalculate_snapshots_endpoint(
    current_user: TokenData = Depends(get_current_user),
):
    """
    Recalculate all derived columns for every snapshot.

    Delegates to the reusable ``recalculate_all_snapshots()`` helper.
    """
    updated = recalculate_all_snapshots(current_user.user_id)
    return {
        "status": "ok",
        "data": {
            "updated": updated,
            "message": f"Recalculated {updated} snapshots",
        },
    }
