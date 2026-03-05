"""
Backup / Restore Service — Excel import/export.

Provides:
  - export_portfolio_excel()  → generates an Excel workbook with all data
  - import_transactions_excel() → imports transactions from Excel upload
"""

import io
import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from app.core.database import query_df, exec_sql, exec_sql_fetchone
from app.services.fx_service import PORTFOLIO_CCY

logger = logging.getLogger(__name__)


def export_portfolio_excel(user_id: int) -> io.BytesIO:
    """
    Export all user data to an Excel workbook with multiple sheets.

    Sheets:
      - Transactions
      - Cash Deposits
      - Stocks
      - Portfolio Snapshots
      - Position Snapshots

    Returns:
        BytesIO buffer containing the .xlsx file.
    """
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Transactions
        tx_df = query_df(
            """SELECT id, portfolio, stock_symbol, txn_date, txn_type,
                      shares, purchase_cost, sell_value, bonus_shares,
                      cash_dividend, reinvested_dividend, fees,
                      broker, reference, notes, created_at
               FROM transactions
               WHERE user_id = ? AND COALESCE(is_deleted, 0) = 0
               ORDER BY txn_date ASC""",
            (user_id,),
        )
        tx_df.to_excel(writer, sheet_name="Transactions", index=False)

        # Cash Deposits
        cd_df = query_df(
            """SELECT id, portfolio, deposit_date, amount, currency,
                      bank_name, notes, created_at
               FROM cash_deposits
               WHERE user_id = ? AND COALESCE(is_deleted, 0) = 0
               ORDER BY deposit_date ASC""",
            (user_id,),
        )
        cd_df.to_excel(writer, sheet_name="Cash Deposits", index=False)

        # Stocks
        st_df = query_df(
            """SELECT symbol, name, portfolio, currency, current_price,
                      last_updated, price_source, tradingview_symbol,
                      tradingview_exchange
               FROM stocks
               WHERE user_id = ?
               ORDER BY portfolio, symbol""",
            (user_id,),
        )
        st_df.to_excel(writer, sheet_name="Stocks", index=False)

        # Portfolio Snapshots
        ps_df = query_df(
            """SELECT snapshot_date, portfolio_value, daily_movement,
                      beginning_difference, deposit_cash, accumulated_cash,
                      net_gain, change_percent, roi_percent,
                      twr_percent, mwrr_percent
               FROM portfolio_snapshots
               WHERE user_id = ?
               ORDER BY snapshot_date DESC""",
            (user_id,),
        )
        ps_df.to_excel(writer, sheet_name="Portfolio Snapshots", index=False)

    buffer.seek(0)
    return buffer


def import_transactions_excel(
    user_id: int,
    file_bytes: bytes,
    portfolio: Optional[str] = None,
    sheet_name: Optional[str] = "Transactions",
) -> Dict[str, Any]:
    """
    Full bulk restore from an exported Excel backup.

    Imports ALL sheets found in the file that match known tables:
      - Transactions (or the *sheet_name* override)
      - Stocks
      - Cash Deposits
      - Portfolio Snapshots

    If *portfolio* is None the value is read from each row's ``portfolio``
    column.  When supplied it overrides every row.

    Returns dict with imported / skipped / errors / warnings plus per-sheet
    breakdown.
    """
    result: Dict[str, Any] = {
        "imported": 0,
        "skipped": 0,
        "errors": [],
        "warnings": [],
        "sheets": {},
    }

    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        available_sheets = xls.sheet_names
    except Exception as exc:
        result["errors"].append({"row": 0, "error": f"Failed to read Excel: {exc}"})
        return result

    now = int(time.time())
    total_imported = 0
    total_skipped = 0

    # ── 1. Stocks ────────────────────────────────────────────────
    stocks_sheet = _find_sheet(available_sheets, ["Stocks", "stocks"])
    if stocks_sheet:
        df = pd.read_excel(xls, sheet_name=stocks_sheet)
        df.columns = [_norm_col(c) for c in df.columns]
        s_imp = s_upd = s_skip = 0

        for _, row in df.iterrows():
            try:
                symbol = _safe_str(row, "symbol")
                if not symbol:
                    s_skip += 1
                    continue

                stock_name = _safe_str(row, "name") or _safe_str(row, "company_name") or symbol
                port = portfolio or _safe_str(row, "portfolio") or "KFH"
                ccy = _safe_str(row, "currency") or "KWD"
                price = _safe_num(row, "current_price")

                # Upsert: check if exists
                existing = exec_sql_fetchone(
                    "SELECT id FROM stocks WHERE symbol = ? AND user_id = ?",
                    (symbol, user_id),
                )

                if not existing:
                    exec_sql(
                        """INSERT INTO stocks
                           (user_id, symbol, name, portfolio, currency,
                            current_price, last_updated)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (user_id, symbol, stock_name, port, ccy, price, now),
                    )
                    s_imp += 1
                else:
                    exec_sql(
                        """UPDATE stocks
                           SET name=?, portfolio=?, currency=?, current_price=?, last_updated=?
                           WHERE symbol=? AND user_id=?""",
                        (stock_name, port, ccy, price, now, symbol, user_id),
                    )
                    s_upd += 1
            except Exception as exc:
                s_skip += 1
                result["errors"].append({"sheet": "Stocks", "row": _ + 2, "error": str(exc)})

        total_imported += s_imp + s_upd
        result["sheets"]["stocks"] = {"new": s_imp, "updated": s_upd, "skipped": s_skip}

    # ── 2. Transactions ──────────────────────────────────────────
    txn_sheet = _find_sheet(available_sheets, [sheet_name or "Transactions", "Transactions", "transactions"])
    if txn_sheet:
        df = pd.read_excel(xls, sheet_name=txn_sheet)
        df.columns = [_norm_col(c) for c in df.columns]
        t_imp = t_skip = 0

        for idx, row in df.iterrows():
            try:
                symbol = (
                    _safe_str(row, "stock_symbol")
                    or _safe_str(row, "symbol")
                    or _safe_str(row, "company")
                    or _safe_str(row, "ticker")
                )
                if not symbol:
                    t_skip += 1
                    continue

                row_port = portfolio or _safe_str(row, "portfolio") or "KFH"
                txn_date = _safe_date(row, "txn_date") or _safe_date(row, "date") or _safe_date(row, "trade_date")
                txn_type = _safe_str(row, "txn_type") or _safe_str(row, "type") or "Buy"

                exec_sql(
                    """INSERT INTO transactions
                       (user_id, portfolio, stock_symbol, txn_date, txn_type,
                        shares, purchase_cost, sell_value, bonus_shares,
                        cash_dividend, reinvested_dividend, fees,
                        broker, reference, notes,
                        category, is_deleted, created_at)
                       VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?, ?,?,?, ?,0,?)""",
                    (
                        user_id,
                        row_port,
                        symbol,
                        txn_date,
                        txn_type,
                        _safe_num(row, "shares"),
                        _safe_num(row, "purchase_cost"),
                        _safe_num(row, "sell_value"),
                        _safe_num(row, "bonus_shares"),
                        _safe_num(row, "cash_dividend"),
                        _safe_num(row, "reinvested_dividend"),
                        _safe_num(row, "fees"),
                        _safe_str(row, "broker"),
                        _safe_str(row, "reference"),
                        _safe_str(row, "notes"),
                        _safe_str(row, "category") or "portfolio",
                        now,
                    ),
                )
                t_imp += 1
            except Exception as exc:
                t_skip += 1
                result["errors"].append({"sheet": "Transactions", "row": idx + 2, "error": str(exc)})

        total_imported += t_imp
        total_skipped += t_skip
        result["sheets"]["transactions"] = {"imported": t_imp, "skipped": t_skip}

    # ── 3. Cash Deposits ─────────────────────────────────────────
    cash_sheet = _find_sheet(available_sheets, ["Cash Deposits", "cash_deposits"])
    if cash_sheet:
        df = pd.read_excel(xls, sheet_name=cash_sheet)
        df.columns = [_norm_col(c) for c in df.columns]
        c_imp = c_skip = 0

        for idx, row in df.iterrows():
            try:
                row_port = portfolio or _safe_str(row, "portfolio") or "KFH"
                amount = _safe_num(row, "amount")
                dep_date = (
                    _safe_date(row, "deposit_date")
                    or _safe_date(row, "date")
                    or time.strftime("%Y-%m-%d")
                )

                exec_sql(
                    """INSERT INTO cash_deposits
                       (user_id, portfolio, deposit_date, amount, currency,
                        bank_name, notes, is_deleted, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                    (
                        user_id,
                        row_port,
                        dep_date,
                        amount,
                        _safe_str(row, "currency") or "KWD",
                        _safe_str(row, "bank_name") or _safe_str(row, "source") or "",
                        _safe_str(row, "notes"),
                        now,
                    ),
                )
                c_imp += 1
            except Exception as exc:
                c_skip += 1
                result["errors"].append({"sheet": "Cash Deposits", "row": idx + 2, "error": str(exc)})

        total_imported += c_imp
        total_skipped += c_skip
        result["sheets"]["cash_deposits"] = {"imported": c_imp, "skipped": c_skip}

    # ── 4. Portfolio Snapshots ───────────────────────────────────
    snap_sheet = _find_sheet(available_sheets, ["Portfolio Snapshots", "portfolio_snapshots"])
    if snap_sheet:
        df = pd.read_excel(xls, sheet_name=snap_sheet)
        df.columns = [_norm_col(c) for c in df.columns]
        sn_imp = sn_skip = 0

        for idx, row in df.iterrows():
            try:
                snap_date = _safe_date(row, "snapshot_date")
                if not snap_date:
                    sn_skip += 1
                    continue

                # Skip duplicates
                existing = exec_sql_fetchone(
                    "SELECT id FROM portfolio_snapshots WHERE snapshot_date=? AND user_id=?",
                    (snap_date, user_id),
                )
                if existing:
                    sn_skip += 1
                    continue

                pv = _safe_num(row, "portfolio_value")
                dep_cash = _safe_num(row, "deposit_cash")

                exec_sql(
                    """INSERT INTO portfolio_snapshots
                       (user_id, snapshot_date, portfolio_value, daily_movement,
                        beginning_difference, deposit_cash, accumulated_cash,
                        net_gain, change_percent, roi_percent,
                        twr_percent, mwrr_percent, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        snap_date,
                        pv,
                        _safe_num(row, "daily_movement"),
                        _safe_num(row, "beginning_difference"),
                        dep_cash,
                        _safe_num(row, "accumulated_cash"),
                        _safe_num(row, "net_gain"),
                        _safe_num(row, "change_percent"),
                        _safe_num(row, "roi_percent"),
                        _safe_num(row, "twr_percent"),
                        _safe_num(row, "mwrr_percent"),
                        now,
                    ),
                )
                sn_imp += 1
            except Exception as exc:
                sn_skip += 1
                result["errors"].append({"sheet": "Snapshots", "row": idx + 2, "error": str(exc)})

        total_imported += sn_imp
        total_skipped += sn_skip
        result["sheets"]["portfolio_snapshots"] = {"imported": sn_imp, "skipped": sn_skip}

    result["imported"] = total_imported
    result["skipped"] = total_skipped

    # ── Auto-recalculate snapshot metrics after import ────────────
    try:
        from app.api.v1.tracker import recalculate_all_snapshots
        updated = recalculate_all_snapshots(user_id)
        if updated > 0:
            result["warnings"].append(f"Auto-recalculated {updated} snapshots to fill derived metrics.")
            logger.info("Auto-recalculated %d snapshots for user %d after import", updated, user_id)
    except Exception as exc:
        logger.warning("Auto-recalculate after import failed: %s", exc)

    return result


# ── Private helpers ──────────────────────────────────────────────────

def _norm_col(col: str) -> str:
    """Normalize column name: lowercase, strip, replace spaces with underscores."""
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def _find_sheet(available: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first sheet name from *candidates* that exists in *available*."""
    lower_map = {s.lower(): s for s in available}
    for c in candidates:
        if c and c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _safe_str(row, col: str, default: str = "") -> str:
    """Get a string cell value, returning *default* for NaN / None / 'nan'."""
    val = row.get(col)
    if val is None:
        return default
    if isinstance(val, float) and pd.isna(val):
        return default
    s = str(val).strip()
    return default if s.lower() == "nan" else s


def _safe_num(row, col: str, default: float = 0.0) -> float:
    """Get a numeric cell value, returning *default* for NaN / None / empty."""
    val = row.get(col)
    if val is None:
        return default
    try:
        if isinstance(val, float) and pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_date(row, col: str) -> str:
    """Get a date cell value as ISO string (YYYY-MM-DD)."""
    val = row.get(col)
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    # pandas Timestamp / datetime
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    s = str(val).strip()
    if not s or s.lower() == "nan" or s.lower() == "nat":
        return ""
    # Try ISO parse
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
        return s  # return as-is; DB will store the raw string
