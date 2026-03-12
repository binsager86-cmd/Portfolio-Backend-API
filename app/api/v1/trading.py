"""
Trading Section API v1 — CFA/IFRS-compliant trading summary.

Mirrors the Streamlit ``ui_trading_section()`` business logic:
  - WAC (Weighted Average Cost) per (symbol, portfolio)
  - Unrealized P&L for open positions
  - Realized P&L for sell transactions (stored > runtime fallback)
  - 12 summary metrics (buys, sells, deposits, withdrawals, P&L, dividends, fees, etc.)
  - Enriched transaction list with avg_cost, pnl, pnl_pct, status, value columns

All heavy computation happens server-side so the mobile client receives
ready-to-render data.
"""

import io
import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.database import query_df, column_exists, add_column_if_missing, exec_sql
from app.services.fx_service import (
    convert_to_kwd,
    safe_float,
    get_usd_kwd_rate,
    PORTFOLIO_CCY,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["Trading"])


def _soft_delete_filter(alias: str = "") -> str:
    """SQL fragment to exclude soft-deleted rows."""
    if not column_exists("transactions", "is_deleted"):
        return ""
    prefix = f"{alias}." if alias else ""
    return f" AND COALESCE({prefix}is_deleted, 0) = 0"


# ── WAC Engine ───────────────────────────────────────────────────────

def _build_position_state(user_id: int):
    """
    CFA/IFRS-compliant WAC calculation — processes ALL non-deleted
    transactions chronologically per (symbol, portfolio).

    Returns (position_state, txn_state) dicts identical to ui.py logic.
    """
    soft_del = _soft_delete_filter()
    sql = f"""
        SELECT id, stock_symbol, COALESCE(portfolio, 'KFH') as portfolio,
               txn_type, txn_date,
               COALESCE(shares, 0) as shares,
               COALESCE(purchase_cost, 0) as purchase_cost,
               COALESCE(sell_value, 0) as sell_value,
               COALESCE(fees, 0) as fees,
               COALESCE(bonus_shares, 0) as bonus_shares,
               COALESCE(cash_dividend, 0) as cash_dividend,
               COALESCE(reinvested_dividend, 0) as reinvested_dividend
        FROM transactions
        WHERE user_id = ? {soft_del}
        ORDER BY txn_date ASC, id ASC
    """
    df = query_df(sql, (user_id,))

    position_state = {}  # (symbol, portfolio) -> state dict
    txn_state = {}       # txn_id -> snapshot at that point

    if df.empty:
        return position_state, txn_state

    for _, txn in df.iterrows():
        sym = txn["stock_symbol"]
        port = txn["portfolio"]
        key = (sym, port)
        txn_id = int(txn["id"])
        typ = txn["txn_type"]

        if key not in position_state:
            position_state[key] = {
                "total_shares": 0.0,
                "total_cost": 0.0,
                "avg_cost": 0.0,
                "last_known_avg_cost": 0.0,   # persists even after full sell
                "realized_pnl": 0.0,
                "dividends_received": 0.0,
                "position_open": False,
            }

        st = position_state[key]

        if typ == "Buy":
            buy_shares = float(txn["shares"])
            buy_cost = float(txn["purchase_cost"])
            fees = float(txn["fees"])
            bonus_on_buy = float(txn["bonus_shares"]) if txn["bonus_shares"] and float(txn["bonus_shares"]) > 0 else 0

            st["total_cost"] += buy_cost + fees
            st["total_shares"] += buy_shares
            if bonus_on_buy > 0:
                st["total_shares"] += bonus_on_buy
            st["total_cost"] = max(st["total_cost"], 0)
            st["total_shares"] = max(st["total_shares"], 0)
            st["avg_cost"] = st["total_cost"] / st["total_shares"] if st["total_shares"] > 0 else 0
            st["last_known_avg_cost"] = st["avg_cost"] if st["avg_cost"] > 0 else st["last_known_avg_cost"]
            st["position_open"] = True
            txn_state[txn_id] = {
                "avg_cost_at_time": st["avg_cost"] or st["last_known_avg_cost"],
                "realized_pnl": 0,
                "cost_basis": st["total_cost"],
                "shares_held": st["total_shares"],
            }

        elif typ == "Sell":
            sell_shares = float(txn["shares"])
            sell_value = float(txn["sell_value"])
            fees = float(txn["fees"])
            avg_cost_at_sale = st["avg_cost"] or st["last_known_avg_cost"]
            proceeds = sell_value - fees
            cost_of_sold = sell_shares * avg_cost_at_sale
            realized_increment = proceeds - cost_of_sold
            st["realized_pnl"] += realized_increment
            st["total_cost"] -= cost_of_sold
            st["total_shares"] -= sell_shares
            if st["total_shares"] > 0:
                st["avg_cost"] = st["total_cost"] / st["total_shares"]
                st["position_open"] = True
            else:
                st["avg_cost"] = 0
                st["total_cost"] = 0
                st["total_shares"] = 0
                st["position_open"] = False
            st["total_cost"] = max(st["total_cost"], 0)
            st["total_shares"] = max(st["total_shares"], 0)
            st["last_known_avg_cost"] = avg_cost_at_sale if avg_cost_at_sale > 0 else st["last_known_avg_cost"]
            txn_state[txn_id] = {
                "avg_cost_at_time": avg_cost_at_sale,
                "realized_pnl": realized_increment,
                "cost_basis": st["total_cost"],
                "shares_held": st["total_shares"],
            }

        elif typ in ("Bonus Shares", "Bonus", "Stock Split"):
            bonus = float(txn["bonus_shares"]) if txn["bonus_shares"] else float(txn["shares"])
            st["total_shares"] += bonus
            st["total_cost"] = max(st["total_cost"], 0)
            st["total_shares"] = max(st["total_shares"], 0)
            st["avg_cost"] = st["total_cost"] / st["total_shares"] if st["total_shares"] > 0 else 0
            st["position_open"] = st["total_shares"] > 0
            # Use current avg_cost if > 0, else fall back to last known
            effective_avg = st["avg_cost"] or st["last_known_avg_cost"]
            if st["avg_cost"] > 0:
                st["last_known_avg_cost"] = st["avg_cost"]
            txn_state[txn_id] = {
                "avg_cost_at_time": effective_avg,
                "realized_pnl": 0,
                "cost_basis": st["total_cost"],
                "shares_held": st["total_shares"],
            }

        elif typ in ("DIVIDEND_ONLY", "Dividend"):
            dividend_amount = float(txn["cash_dividend"]) if txn["cash_dividend"] else 0
            st["dividends_received"] += dividend_amount
            bonus_in_div = float(txn["bonus_shares"]) if txn["bonus_shares"] and float(txn["bonus_shares"]) > 0 else 0
            if bonus_in_div > 0:
                st["total_shares"] += bonus_in_div
                st["avg_cost"] = st["total_cost"] / st["total_shares"] if st["total_shares"] > 0 else 0
                st["position_open"] = st["total_shares"] > 0
            st["total_cost"] = max(st["total_cost"], 0)
            st["total_shares"] = max(st["total_shares"], 0)
            # Dividends always carry the position's avg_cost (current or last known)
            effective_avg = st["avg_cost"] or st["last_known_avg_cost"]
            txn_state[txn_id] = {
                "avg_cost_at_time": effective_avg,
                "realized_pnl": 0,
                "dividend": dividend_amount,
                "cost_basis": st["total_cost"],
                "shares_held": st["total_shares"],
            }

        else:
            st["total_cost"] = max(st["total_cost"], 0)
            st["total_shares"] = max(st["total_shares"], 0)
            effective_avg = st["avg_cost"] or st["last_known_avg_cost"]
            txn_state[txn_id] = {
                "avg_cost_at_time": effective_avg,
                "realized_pnl": 0,
                "cost_basis": st["total_cost"],
                "shares_held": st["total_shares"],
            }

    return position_state, txn_state


# ── Main endpoint ────────────────────────────────────────────────────

@router.get("/trading-summary")
async def trading_summary(
    portfolio: Optional[str] = Query(None, description="Filter by portfolio"),
    txn_type: Optional[str] = Query(None, description="Filter by transaction type"),
    search: Optional[str] = Query(None, description="Search symbol/notes/portfolio"),
    source: Optional[str] = Query(None, description="Filter by source (MANUAL, UPLOAD, etc.)"),
    date_from: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    current_user: TokenData = Depends(get_current_user),
):
    """
    CFA/IFRS-compliant Trading Section summary — matches Streamlit exactly.

    Returns:
      - summary: 12 metric values (buys, sells, deposits, withdrawals,
        unrealized/realized/total P&L, dividends, fees, net cash flow, return %)
      - transactions: enriched list with avg_cost, pnl, pnl_pct, status, value
      - pagination info
    """
    user_id = current_user.user_id

    # 0) Ensure optional columns exist (additive migration — matches Streamlit init_db)
    _OPTIONAL_COLS = [
        ("avg_cost_at_txn", "REAL"),
        ("realized_pnl_at_txn", "REAL"),
        ("cost_basis_at_txn", "REAL"),
        ("shares_held_at_txn", "REAL"),
        ("is_deleted", "INTEGER DEFAULT 0"),
        ("source", "TEXT"),
        ("source_reference", "TEXT"),
        ("category", "TEXT"),
    ]
    for col_name, col_type in _OPTIONAL_COLS:
        try:
            add_column_if_missing("transactions", col_name, col_type)
        except Exception:
            pass  # Best-effort — table may not exist yet

    # 1) Build WAC position state
    position_state, txn_state = _build_position_state(user_id)

    # 2) Fetch all transactions with current prices
    #    Build SELECT dynamically so missing columns get COALESCE(NULL, 0)
    soft_del = _soft_delete_filter("t")

    has_avg_cost = column_exists("transactions", "avg_cost_at_txn")
    has_realized  = column_exists("transactions", "realized_pnl_at_txn")
    has_cost_basis = column_exists("transactions", "cost_basis_at_txn")
    has_shares_held = column_exists("transactions", "shares_held_at_txn")
    has_category = column_exists("transactions", "category")

    avg_cost_col   = "t.avg_cost_at_txn"       if has_avg_cost    else "NULL AS avg_cost_at_txn"
    realized_col   = "t.realized_pnl_at_txn"   if has_realized    else "NULL AS realized_pnl_at_txn"
    cost_basis_col = "t.cost_basis_at_txn"      if has_cost_basis  else "NULL AS cost_basis_at_txn"
    shares_held_col = "t.shares_held_at_txn"    if has_shares_held else "NULL AS shares_held_at_txn"
    category_col   = "t.category"               if has_category    else "NULL AS category"

    txn_sql = f"""
        SELECT
            t.id,
            t.stock_symbol AS symbol,
            t.txn_date AS date,
            COALESCE(t.portfolio, s.portfolio, 'KFH') AS portfolio,
            t.txn_type AS type,
            {category_col},
            COALESCE(t.shares, 0) AS quantity,
            COALESCE(t.purchase_cost, 0) AS purchase_cost,
            COALESCE(t.sell_value, 0) AS sell_value,
            COALESCE(t.fees, 0) AS fees,
            COALESCE(t.cash_dividend, 0) AS dividend,
            COALESCE(t.bonus_shares, 0) AS bonus_shares,
            t.reinvested_dividend,
            t.notes,
            COALESCE(t.source, 'MANUAL') AS source,
            t.source_reference,
            COALESCE(s.current_price, 0) AS current_price,
            COALESCE(s.name, t.stock_symbol) AS company_name,
            s.id AS stock_id,
            {avg_cost_col},
            {realized_col},
            {cost_basis_col},
            {shares_held_col}
        FROM transactions t
        LEFT JOIN stocks s ON UPPER(t.stock_symbol) = UPPER(s.symbol) AND t.user_id = s.user_id
        WHERE t.user_id = ? {soft_del}
        ORDER BY t.txn_date DESC, t.id DESC
    """
    df = query_df(txn_sql, (user_id,))

    if df.empty:
        return {
            "status": "ok",
            "data": {
                "summary": _empty_summary(),
                "transactions": [],
                "pagination": {"page": 1, "page_size": page_size, "total_items": 0, "total_pages": 1},
            },
        }

    # 3) Build price map fallback (same as ui.py)
    price_sql = """
        SELECT UPPER(symbol) as symbol_upper, symbol, current_price
        FROM stocks WHERE user_id = ? AND current_price > 0
    """
    price_df = query_df(price_sql, (user_id,))
    price_map = {row["symbol_upper"]: row["current_price"] for _, row in price_df.iterrows()}

    def lookup_price(row):
        if row["current_price"] > 0:
            return row["current_price"]
        sym = str(row["symbol"]).upper() if row["symbol"] else ""
        if not sym:
            return 0
        if sym in price_map:
            return price_map[sym]
        if sym.endswith(".KW") and sym[:-3] in price_map:
            return price_map[sym[:-3]]
        if f"{sym}.KW" in price_map:
            return price_map[f"{sym}.KW"]
        for stock_sym, price in price_map.items():
            if sym in stock_sym or stock_sym in sym:
                return price
        return 0

    df["current_price"] = df.apply(lookup_price, axis=1)

    # 4) Assign avg_cost per row — EVERY row gets the WAC snapshot
    #    Priority: stored DB value > runtime WAC engine > last_known > related symbol
    def get_avg_cost(row):
        stored = row.get("avg_cost_at_txn")
        if stored is not None and pd.notna(stored) and stored > 0:
            return float(stored)
        txn_id = row.get("id")
        # WAC engine records avg_cost_at_time for ALL txn types
        if txn_id in txn_state:
            v = txn_state[txn_id].get("avg_cost_at_time", 0)
            if v > 0:
                return v
        # Fallback: current or last_known position WAC
        sym, port = row.get("symbol", ""), row.get("portfolio", "KFH")
        key = (sym, port)
        if key in position_state:
            ac = position_state[key].get("avg_cost", 0) or position_state[key].get("last_known_avg_cost", 0)
            if ac > 0:
                return ac
        # Cross-symbol fallback: e.g. "AGILITY PLC" → check "AGILITY" position
        # Handles corporate restructurings / spin-offs
        for (ps_sym, ps_port), ps_st in position_state.items():
            if ps_port == port and ps_sym != sym:
                if ps_sym in sym or sym in ps_sym:
                    ac = ps_st.get("last_known_avg_cost", 0) or ps_st.get("avg_cost", 0)
                    if ac > 0:
                        return ac
        return 0

    df["avg_cost"] = df.apply(get_avg_cost, axis=1)

    # 4b) Backfill: persist computed values for rows that don't have them.
    #     Matches Streamlit recalculate_and_store_avg_costs() — stores all 4 columns:
    #     avg_cost_at_txn, realized_pnl_at_txn, cost_basis_at_txn, shares_held_at_txn
    if has_avg_cost:
        backfill_count = 0
        for _, row in df.iterrows():
            txn_id = int(row["id"])
            snap = txn_state.get(txn_id, {})
            updates = {}

            # avg_cost_at_txn
            stored_ac = row.get("avg_cost_at_txn")
            if (stored_ac is None or pd.isna(stored_ac) or stored_ac == 0) and row["avg_cost"] > 0:
                updates["avg_cost_at_txn"] = round(float(row["avg_cost"]), 8)

            # realized_pnl_at_txn (Sell transactions)
            if has_realized:
                stored_rpnl = row.get("realized_pnl_at_txn")
                snap_rpnl = snap.get("realized_pnl", 0)
                if (stored_rpnl is None or pd.isna(stored_rpnl)) and snap_rpnl != 0:
                    updates["realized_pnl_at_txn"] = round(float(snap_rpnl), 4)

            # cost_basis_at_txn
            if has_cost_basis:
                stored_cb = row.get("cost_basis_at_txn")
                snap_cb = snap.get("cost_basis")
                if (stored_cb is None or pd.isna(stored_cb)) and snap_cb is not None:
                    updates["cost_basis_at_txn"] = round(float(snap_cb), 4)

            # shares_held_at_txn
            if has_shares_held:
                stored_sh = row.get("shares_held_at_txn")
                snap_sh = snap.get("shares_held")
                if (stored_sh is None or pd.isna(stored_sh)) and snap_sh is not None:
                    updates["shares_held_at_txn"] = round(float(snap_sh), 6)

            if updates:
                set_clause = ", ".join(f"{k} = ?" for k in updates)
                vals = list(updates.values()) + [txn_id]
                try:
                    exec_sql(f"UPDATE transactions SET {set_clause} WHERE id = ?", tuple(vals))
                    backfill_count += 1
                except Exception as exc:
                    logger.warning("Backfill id=%s failed: %s", txn_id, exc)

        if backfill_count > 0:
            logger.info("Backfilled WAC columns for %d rows", backfill_count)

    # 5) Realized P&L per row
    def get_realized_pnl(row):
        if row.get("type") != "Sell":
            return 0
        stored = row.get("realized_pnl_at_txn")
        if stored is not None and pd.notna(stored):
            return float(stored)
        txn_id = row.get("id")
        if txn_id in txn_state:
            return txn_state[txn_id].get("realized_pnl", 0)
        return 0

    df["stored_realized_pnl"] = df.apply(get_realized_pnl, axis=1)

    # 6) Derived columns
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    def calc_price(r):
        if r["type"] == "Buy" and r["quantity"] and r["quantity"] > 0:
            return r["purchase_cost"] / r["quantity"]
        if r["type"] == "Sell" and r["quantity"] and r["quantity"] > 0:
            return r["sell_value"] / r["quantity"]
        return 0

    df["price"] = df.apply(calc_price, axis=1)

    df["sell_price"] = df.apply(
        lambda r: r["sell_value"] / r["quantity"] if r["type"] == "Sell" and r["quantity"] and r["quantity"] > 0 else 0,
        axis=1,
    )

    def calc_value(r):
        if r["type"] == "Buy":
            return r["purchase_cost"]
        if r["type"] in ("Sell", "Withdrawal"):
            return r["sell_value"]
        if r["type"] in ("DIVIDEND_ONLY", "Dividend"):
            return r["dividend"]
        if r["type"] == "Deposit" or r.get("category") == "FLOW_IN":
            return r["purchase_cost"] + r["sell_value"]
        return 0

    df["value"] = df.apply(calc_value, axis=1)

    def get_status(r):
        t = r["type"]
        key = (r.get("symbol", ""), r.get("portfolio", "KFH"))
        if t == "Sell":
            return "Realized"
        elif t == "Buy":
            if key in position_state and not position_state[key].get("position_open", False):
                return "Closed"
            return "Unrealized"
        elif t in ("DIVIDEND_ONLY", "Dividend"):
            return "Income"
        elif t in ("Bonus Shares", "Bonus"):
            return "Bonus"
        return ""

    df["status"] = df.apply(get_status, axis=1)

    # 7) P&L calculation (CFA-compliant)
    def calc_pnl(r):
        typ = r.get("type", "")
        txn_id = r.get("id")
        qty = float(r.get("quantity", 0) or 0)
        key = (r.get("symbol", ""), r.get("portfolio", "KFH"))

        if typ == "Buy":
            if key in position_state and not position_state[key].get("position_open", False):
                return 0
            cp = float(r.get("current_price", 0) or 0)
            ac = float(r.get("avg_cost", 0) or 0)
            if cp > 0 and qty > 0 and ac > 0:
                return (cp - ac) * qty
            if cp > 0 and qty > 0:
                pc = float(r.get("purchase_cost", 0) or 0)
                if pc > 0:
                    return (cp - pc / qty) * qty
            return 0

        elif typ == "Sell":
            stored = r.get("stored_realized_pnl", 0)
            if stored != 0:
                return stored
            if txn_id in txn_state:
                return txn_state[txn_id].get("realized_pnl", 0)
            sv = float(r.get("sell_value", 0) or 0)
            ac = float(r.get("avg_cost", 0) or 0)
            fees = float(r.get("fees", 0) or 0)
            if qty > 0 and ac > 0:
                return (sv - fees) - ac * qty
            return 0

        elif typ in ("DIVIDEND_ONLY", "Dividend"):
            return float(r.get("dividend", 0) or 0)

        return 0

    def calc_pnl_pct(r):
        typ = r.get("type", "")
        pnl = r.get("pnl", 0)
        qty = float(r.get("quantity", 0) or 0)
        ac = float(r.get("avg_cost", 0) or 0)
        key = (r.get("symbol", ""), r.get("portfolio", "KFH"))

        if typ == "Buy":
            if key in position_state and not position_state[key].get("position_open", False):
                return 0
            if ac > 0 and qty > 0:
                return (pnl / (ac * qty)) * 100
            pc = float(r.get("purchase_cost", 0) or 0)
            if pc > 0:
                return (pnl / pc) * 100
            return 0

        elif typ == "Sell":
            if ac <= 0:
                txn_id = r.get("id")
                if txn_id in txn_state:
                    ac = txn_state[txn_id].get("avg_cost_at_time", 0)
            if ac <= 0:
                stored = r.get("avg_cost_at_txn")
                if stored is not None and pd.notna(stored):
                    ac = float(stored)
            if ac > 0 and qty > 0:
                return (pnl / (ac * qty)) * 100
            return 0

        return 0

    df["pnl"] = df.apply(calc_pnl, axis=1)
    df["pnl_pct"] = df.apply(calc_pnl_pct, axis=1)

    # ── Apply date + portfolio filters BEFORE summary so cards reflect them ──

    summary_df = df.copy()
    if portfolio:
        summary_df = summary_df[summary_df["portfolio"] == portfolio]
    if date_from:
        try:
            dt_from = pd.Timestamp(date_from)
            summary_df = summary_df[summary_df["date"] >= dt_from]
        except Exception:
            pass
    if date_to:
        try:
            dt_to = pd.Timestamp(date_to)
            summary_df = summary_df[summary_df["date"] <= dt_to]
        except Exception:
            pass

    # ── Helper: convert amount to KWD based on portfolio ─────────────

    def _to_kwd(amount: float, portfolio_name: str) -> float:
        """Convert a native-currency amount to KWD using portfolio's currency."""
        ccy = PORTFOLIO_CCY.get(portfolio_name, "KWD")
        return convert_to_kwd(amount, ccy)

    # ── Summary metrics (CFA-compliant, matches ui.py) ───────────────
    #    All values converted to KWD for unified total.

    buy_df = summary_df[summary_df["type"] == "Buy"]
    sell_df = summary_df[summary_df["type"] == "Sell"]
    withdrawal_df = summary_df[(summary_df["type"] == "Withdrawal") | (summary_df["category"] == "FLOW_OUT")]
    dividend_df = summary_df[summary_df["type"].isin(["DIVIDEND_ONLY", "Dividend"])]

    # Convert buy/sell/withdrawal/fee/dividend amounts to KWD
    total_buys = float(buy_df.apply(
        lambda r: _to_kwd(float(r["purchase_cost"] or 0), r.get("portfolio", "KFH")), axis=1
    ).sum()) if not buy_df.empty else 0
    total_sells = float(sell_df.apply(
        lambda r: _to_kwd(float(r["sell_value"] or 0), r.get("portfolio", "KFH")), axis=1
    ).sum()) if not sell_df.empty else 0
    total_withdrawals = float(withdrawal_df.apply(
        lambda r: _to_kwd(float(r["sell_value"] or 0), r.get("portfolio", "KFH")), axis=1
    ).sum()) if not withdrawal_df.empty else 0
    total_fees = float(summary_df.apply(
        lambda r: _to_kwd(float(r["fees"] or 0), r.get("portfolio", "KFH")), axis=1
    ).sum())

    # Deposits from cash_deposits table (source of truth), converted to KWD
    # Separate positive (deposits) from negative (withdrawals) — matches overview logic
    dep_sql = """
        SELECT amount, currency, portfolio, deposit_date FROM cash_deposits
        WHERE user_id = ? AND include_in_analysis = 1
          AND COALESCE(is_deleted, 0) = 0
    """
    dep_params = [user_id]
    if portfolio:
        dep_sql += " AND portfolio = ?"
        dep_params.append(portfolio)
    dep_df = query_df(dep_sql, tuple(dep_params))
    if not dep_df.empty:
        # Apply date filter to deposits too
        dep_df["deposit_date"] = pd.to_datetime(dep_df["deposit_date"], errors="coerce")
        if date_from:
            try:
                dep_df = dep_df[dep_df["deposit_date"] >= pd.Timestamp(date_from)]
            except Exception:
                pass
        if date_to:
            try:
                dep_df = dep_df[dep_df["deposit_date"] <= pd.Timestamp(date_to)]
            except Exception:
                pass
        dep_df["amount_kwd"] = dep_df.apply(
            lambda r: convert_to_kwd(float(r["amount"]), r.get("currency", "KWD") or "KWD"), axis=1
        )
        # Only count positive amounts as deposits (negative = withdrawals)
        pos_mask = dep_df["amount_kwd"] > 0
        total_deposits = float(dep_df.loc[pos_mask, "amount_kwd"].sum())
        deposit_count = int(pos_mask.sum())
        # Add negative amounts (from cash_deposits) to withdrawals
        neg_mask = dep_df["amount_kwd"] < 0
        total_withdrawals += float(dep_df.loc[neg_mask, "amount_kwd"].abs().sum())
    else:
        total_deposits = 0.0
        deposit_count = 0

    # Realized P&L — convert each sell's realized PnL to KWD
    if "stored_realized_pnl" in summary_df.columns and summary_df["stored_realized_pnl"].notna().any():
        total_realized_pnl = float(summary_df.apply(
            lambda r: _to_kwd(float(r.get("stored_realized_pnl", 0) or 0), r.get("portfolio", "KFH")),
            axis=1,
        ).sum())
    else:
        total_realized_pnl = sum(
            _to_kwd(s["realized_pnl"], port)
            for (sym, port), s in position_state.items()
        )

    # Dividends — convert to KWD
    total_dividends = float(summary_df.apply(
        lambda r: _to_kwd(float(r["dividend"] or 0), r.get("portfolio", "KFH")), axis=1
    ).sum())

    # Unrealized P&L from open positions — convert to KWD
    total_unrealized_pnl = 0.0
    if "shares_held_at_txn" in summary_df.columns and summary_df["shares_held_at_txn"].notna().any():
        latest = summary_df.groupby(["symbol", "portfolio"]).first().reset_index()
        for _, r in latest.iterrows():
            sh = safe_float(r.get("shares_held_at_txn"), 0)
            cb = safe_float(r.get("cost_basis_at_txn"), 0)
            cp = safe_float(r.get("current_price"), 0)
            port = r.get("portfolio", "KFH")
            if sh > 0 and cp > 0:
                native_pnl = cp * sh - cb
                total_unrealized_pnl += _to_kwd(native_pnl, port)
    else:
        for (sym, port), st in position_state.items():
            if st.get("position_open") and st["total_shares"] > 0:
                prices = summary_df[(summary_df["symbol"] == sym) & (summary_df["portfolio"] == port)]["current_price"].dropna()
                if prices.empty:
                    prices = summary_df[summary_df["symbol"] == sym]["current_price"].dropna()
                cp = float(prices.iloc[0]) if not prices.empty else 0
                if cp > 0:
                    native_pnl = cp * st["total_shares"] - st["total_cost"]
                    total_unrealized_pnl += _to_kwd(native_pnl, port)

    total_pnl = total_unrealized_pnl + total_realized_pnl
    total_return = total_pnl + total_dividends
    net_cash_flow = total_sells + total_deposits + total_dividends - total_buys - total_withdrawals - total_fees
    total_return_pct = (total_return / total_buys * 100) if total_buys > 0 else 0

    summary = {
        "total_buys": round(total_buys, 3),
        "buy_count": len(buy_df),
        "total_sells": round(total_sells, 3),
        "sell_count": len(sell_df),
        "total_deposits": round(total_deposits, 3),
        "deposit_count": deposit_count,
        "total_withdrawals": round(total_withdrawals, 3),
        "withdrawal_count": len(withdrawal_df),
        "unrealized_pnl": round(total_unrealized_pnl, 3),
        "realized_pnl": round(total_realized_pnl, 3),
        "total_pnl": round(total_pnl, 3),
        "total_dividends": round(total_dividends, 3),
        "dividend_count": len(dividend_df),
        "total_fees": round(total_fees, 3),
        "net_cash_flow": round(net_cash_flow, 3),
        "total_return_pct": round(total_return_pct, 2),
        "total_transactions": len(summary_df),
        "total_trades": len(buy_df) + len(sell_df),
        "currency": "KWD",
    }

    # ── Apply filters to transaction list ────────────────────────────

    filtered = df.copy()

    if portfolio:
        filtered = filtered[filtered["portfolio"] == portfolio]
    if txn_type:
        filtered = filtered[filtered["type"] == txn_type]
    if source:
        filtered = filtered[filtered["source"].astype(str).str.upper() == source.upper()]
    if date_from:
        try:
            dt_from = pd.Timestamp(date_from)
            filtered = filtered[filtered["date"] >= dt_from]
        except Exception:
            pass
    if date_to:
        try:
            dt_to = pd.Timestamp(date_to)
            filtered = filtered[filtered["date"] <= dt_to]
        except Exception:
            pass
    if search:
        q = search.strip().lower()
        words = q.split()
        for w in words:
            mask = (
                filtered["symbol"].astype(str).str.lower().str.contains(w, na=False)
                | filtered["company_name"].astype(str).str.lower().str.contains(w, na=False)
                | filtered["portfolio"].astype(str).str.lower().str.contains(w, na=False)
                | filtered["type"].astype(str).str.lower().str.contains(w, na=False)
                | filtered["notes"].astype(str).str.lower().str.contains(w, na=False)
                | filtered["source"].astype(str).str.lower().str.contains(w, na=False)
            )
            filtered = filtered[mask]

    total_filtered = len(filtered)
    total_pages = max(1, (total_filtered + page_size - 1) // page_size)
    offset = (page - 1) * page_size
    page_df = filtered.iloc[offset: offset + page_size]

    # ── Serialize transactions ───────────────────────────────────────

    cols = [
        "id", "date", "symbol", "company_name", "stock_id", "portfolio", "type", "status", "source",
        "quantity", "avg_cost", "price", "current_price", "sell_price",
        "value", "pnl", "pnl_pct", "fees", "dividend", "bonus_shares", "notes",
    ]
    records = []
    for _, r in page_df.iterrows():
        rec = {}
        for c in cols:
            v = r.get(c)
            if c == "date":
                rec[c] = v.strftime("%Y-%m-%d") if pd.notna(v) else None
            elif isinstance(v, (float, int)):
                rec[c] = round(float(v), 4) if pd.notna(v) else 0
            else:
                rec[c] = str(v) if pd.notna(v) else None
        # Show current_price only for unrealized positions
        if rec.get("status") != "Unrealized":
            rec["current_price"] = 0
        records.append(rec)

    return {
        "status": "ok",
        "data": {
            "summary": summary,
            "transactions": records,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_filtered,
                "total_pages": total_pages,
            },
        },
    }


@router.patch("/rename-stock")
async def rename_stock_by_symbol(
    symbol: str = Query(..., description="Stock symbol to rename"),
    name: str = Query(..., description="New display name"),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Rename a stock by symbol — mirrors Streamlit 'Edit Stock Details' expander.
    Updates the stock name for the current user.
    """
    user_id = current_user.user_id
    row = query_df(
        "SELECT id, name FROM stocks WHERE UPPER(symbol) = UPPER(?) AND user_id = ?",
        (symbol, user_id),
    )
    if row.empty:
        from app.core.exceptions import NotFoundError
        raise NotFoundError("Stock", symbol)

    stock_id = int(row.iloc[0]["id"])
    exec_sql(
        "UPDATE stocks SET name = ? WHERE id = ? AND user_id = ?",
        (name.strip(), stock_id, user_id),
    )
    return {
        "status": "ok",
        "data": {
            "stock_id": stock_id,
            "symbol": symbol,
            "name": name.strip(),
            "message": "Stock name updated",
        },
    }


def _empty_summary() -> dict:
    return {
        "total_buys": 0, "buy_count": 0,
        "total_sells": 0, "sell_count": 0,
        "total_deposits": 0, "deposit_count": 0,
        "total_withdrawals": 0, "withdrawal_count": 0,
        "unrealized_pnl": 0, "realized_pnl": 0, "total_pnl": 0,
        "total_dividends": 0, "dividend_count": 0,
        "total_fees": 0, "net_cash_flow": 0,
        "total_return_pct": 0, "total_transactions": 0, "total_trades": 0,
        "currency": "KWD",
    }


# ── Recalculate endpoint ────────────────────────────────────────────

@router.post("/trading-recalculate")
async def trading_recalculate(current_user: TokenData = Depends(get_current_user)):
    """
    Recalculate and store avg_cost, realized_pnl, cost_basis, shares_held
    for ALL transactions. Matches Streamlit recalculate_and_store_avg_costs().
    """
    user_id = current_user.user_id
    soft_del = _soft_delete_filter()

    # Ensure columns exist
    for col_name, col_type in [
        ("avg_cost_at_txn", "REAL"), ("realized_pnl_at_txn", "REAL"),
        ("cost_basis_at_txn", "REAL"), ("shares_held_at_txn", "REAL"),
    ]:
        try:
            add_column_if_missing("transactions", col_name, col_type)
        except Exception:
            pass

    # Get distinct (symbol, portfolio) groups
    group_sql = f"""
        SELECT DISTINCT stock_symbol, COALESCE(portfolio, 'KFH') as portfolio
        FROM transactions
        WHERE user_id = ? {soft_del}
          AND stock_symbol IS NOT NULL AND stock_symbol != ''
    """
    groups = query_df(group_sql, (user_id,))

    stats = {"updated": 0, "positions_processed": 0, "errors": []}

    for _, grp in groups.iterrows():
        symbol, portfolio = grp["stock_symbol"], grp["portfolio"]
        try:
            txn_sql = f"""
                SELECT id, txn_type, txn_date,
                    COALESCE(shares, 0) as shares,
                    COALESCE(purchase_cost, 0) as purchase_cost,
                    COALESCE(sell_value, 0) as sell_value,
                    COALESCE(fees, 0) as fees,
                    COALESCE(bonus_shares, 0) as bonus_shares,
                    COALESCE(cash_dividend, 0) as cash_dividend
                FROM transactions
                WHERE user_id = ? AND stock_symbol = ? AND COALESCE(portfolio, 'KFH') = ?
                    {soft_del}
                ORDER BY txn_date ASC, id ASC
            """
            txns = query_df(txn_sql, (user_id, symbol, portfolio))

            total_shares = 0.0
            total_cost = 0.0

            for _, txn in txns.iterrows():
                txn_id = int(txn["id"])
                typ = txn["txn_type"]
                shares = float(txn["shares"])
                pc = float(txn["purchase_cost"])
                sv = float(txn["sell_value"])
                fees = float(txn["fees"])
                bs = float(txn["bonus_shares"])

                realized_pnl = 0.0
                avg_cost = 0.0

                typ_upper = (typ or "").upper()

                if typ_upper == "BUY":
                    total_cost += pc + fees
                    total_shares += shares
                    if bs > 0:
                        total_shares += bs
                    avg_cost = total_cost / total_shares if total_shares > 0 else 0

                elif typ_upper == "SELL":
                    if total_shares > 0 and shares > 0:
                        avg_before = total_cost / total_shares
                        proceeds = sv - fees
                        cost_sold = avg_before * shares
                        realized_pnl = proceeds - cost_sold
                        total_cost -= cost_sold
                        total_shares -= shares
                        avg_cost = avg_before
                    else:
                        avg_cost = 0

                elif typ_upper in ("BONUS SHARES", "BONUS", "STOCK SPLIT",
                                   "DIVIDEND_ONLY", "DIVIDEND"):
                    if bs > 0:
                        total_shares += bs
                    elif typ_upper in ("BONUS SHARES", "BONUS", "STOCK SPLIT") and shares > 0:
                        total_shares += shares
                    avg_cost = total_cost / total_shares if total_shares > 0 else 0
                else:
                    avg_cost = total_cost / total_shares if total_shares > 0 else 0

                total_cost = max(total_cost, 0)
                total_shares = max(total_shares, 0)

                exec_sql(
                    "UPDATE transactions SET avg_cost_at_txn=?, realized_pnl_at_txn=?, "
                    "cost_basis_at_txn=?, shares_held_at_txn=? WHERE id=?",
                    (round(avg_cost, 8), round(realized_pnl, 4),
                     round(total_cost, 4), round(total_shares, 6), txn_id),
                )
                stats["updated"] += 1

            stats["positions_processed"] += 1
        except Exception as e:
            stats["errors"].append(f"{symbol}/{portfolio}: {str(e)}")

    return {"status": "ok", "data": stats}


# ── Export endpoint ──────────────────────────────────────────────────

@router.get("/trading-export")
async def trading_export(current_user: TokenData = Depends(get_current_user)):
    """
    Export all transactions as Excel (.xlsx).
    Matches Streamlit's Download Trading History button.
    """
    user_id = current_user.user_id
    soft_del = _soft_delete_filter("t")

    sql = f"""
        SELECT
            t.id, t.txn_date AS date, t.stock_symbol AS symbol,
            COALESCE(t.portfolio, 'KFH') AS portfolio,
            t.txn_type AS type,
            COALESCE(t.shares, 0) AS quantity,
            COALESCE(t.purchase_cost, 0) AS purchase_cost,
            COALESCE(t.sell_value, 0) AS sell_value,
            COALESCE(t.fees, 0) AS fees,
            COALESCE(t.cash_dividend, 0) AS dividend,
            COALESCE(t.bonus_shares, 0) AS bonus_shares,
            t.notes,
            COALESCE(t.source, 'MANUAL') AS source,
            t.avg_cost_at_txn AS avg_cost,
            t.realized_pnl_at_txn AS realized_pnl
        FROM transactions t
        WHERE t.user_id = ? {soft_del}
        ORDER BY t.txn_date DESC, t.id DESC
    """
    df = query_df(sql, (user_id,))

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Trading History", index=False)
    buf.seek(0)

    today = date.today().isoformat()
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="transactions_{today}.xlsx"'},
    )
