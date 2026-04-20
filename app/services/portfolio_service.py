"""
Portfolio Service — CFA/IFRS-compliant portfolio calculations.

Extracted from ui.py — NO Streamlit dependencies.
All business logic lives here; routes are thin wrappers.

Class-based API:
    svc = PortfolioService(user_id=1)
    overview = svc.get_overview()
    holdings, totals = svc.get_all_holdings(portfolio="KFH")
    perf = svc.calculate_performance(period="ytd")

Backward-compatible module-level functions are kept at the bottom
so existing route imports continue to work unmodified.
"""

import logging
import re
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

from app.core.database import query_df, query_val, query_one, get_conn, exec_sql, column_exists
from app.services.fx_service import (
    convert_to_kwd,
    safe_float,
    get_usd_kwd_rate,
    PORTFOLIO_CCY,
    DEFAULT_USD_TO_KWD,
)

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────

def _soft_delete_filter(table_alias: str = "") -> str:
    """SQL fragment to exclude soft-deleted rows (if column exists)."""
    if not column_exists("transactions", "is_deleted"):
        return ""
    prefix = f"{table_alias}." if table_alias else ""
    return f" AND COALESCE({prefix}is_deleted, 0) = 0"


def _parse_stockanalysis_pe(page_text: str) -> Optional[float]:
    """Extract P/E ratio from StockAnalysis statistics page HTML payload."""
    # StockAnalysis embeds stats in Svelte payload fragments like: id:"pe" ... hover:"12.34"
    m = re.search(r'id:"pe"[^}]*hover:"([^\"]+)"', page_text)
    if not m:
        return None
    raw = (m.group(1) or "").replace(",", "").replace("%", "").strip()
    if not raw or raw.lower() in {"n/a", "na", "-", "—"}:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return round(val, 2) if val > 0 else None


def _fetch_pe_from_stockanalysis(symbol: str, currency: str) -> Optional[float]:
    """Fetch P/E from stockanalysis.com statistics page for KW and US stocks."""
    base = re.sub(r"\.KW$", "", (symbol or "").strip(), flags=re.IGNORECASE)
    if not base:
        return None

    is_us = (currency or "").upper() == "USD"
    if is_us:
        # US path: /stocks/{symbol}/statistics/
        url = f"https://stockanalysis.com/stocks/{base.lower()}/statistics/"
    else:
        # Kuwait path: /quote/kwse/{SYMBOL}/statistics/
        url = f"https://stockanalysis.com/quote/kwse/{base.upper()}/statistics/"

    try:
        resp = httpx.get(
            url,
            timeout=12,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                )
            },
        )
        if resp.status_code != 200:
            logger.debug("StockAnalysis P/E: %s returned %s", url, resp.status_code)
            return None
        return _parse_stockanalysis_pe(resp.text)
    except Exception as exc:
        logger.debug("StockAnalysis P/E fetch failed for %s: %s", symbol, exc)
        return None


# ── Standalone WAC engine (testable without class) ───────────────────

def compute_holdings_avg_cost(tx: pd.DataFrame) -> dict:
    """
    CFA/IFRS-Compliant Weighted Average Cost Method.

    Transaction rules (matches ui.py L9316-9416 exactly):
      Buy  → cost += purchase_cost + fees;  shares += qty
      Sell → avg = cost / shares;  realized_pnl += (sell_value − fees) − avg × qty
      Bonus→ shares += bonus (zero cost ⇒ dilutes avg_cost)

    Returns dict:
        shares, cost_basis, avg_cost, cash_div, bonus_shares,
        reinv, realized_pnl, position_open
    """
    _empty = {
        "shares": 0.0,
        "cost_basis": 0.0,
        "avg_cost": 0.0,
        "cash_div": 0.0,
        "bonus_shares": 0.0,
        "reinv": 0.0,
        "realized_pnl": 0.0,
        "position_open": False,
    }

    if tx is None or tx.empty:
        return _empty

    t = tx.copy()
    t["txn_date"] = t["txn_date"].fillna("")
    t["created_at"] = t["created_at"].fillna(0)
    t = t.sort_values(["txn_date", "created_at", "id"], ascending=[True, True, True])

    shares = 0.0
    cost = 0.0
    realized_pnl = 0.0

    cash_div = float(t.get("cash_dividend", pd.Series([0], dtype=float)).fillna(0).sum())
    bonus_total = float(t.get("bonus_shares", pd.Series([0], dtype=float)).fillna(0).sum())
    reinv = float(t.get("reinvested_dividend", pd.Series([0], dtype=float)).fillna(0).sum())

    for _, r in t.iterrows():
        typ = str(r.get("txn_type", ""))
        sh = safe_float(r.get("shares", 0), 0.0)
        fees = safe_float(r.get("fees", 0), 0.0)
        buy_cost = safe_float(r.get("purchase_cost", 0), 0.0)
        sell_value = safe_float(r.get("sell_value", 0), 0.0)
        bonus = safe_float(r.get("bonus_shares", 0), 0.0)

        if typ == "Buy":
            shares += sh
            cost += buy_cost + fees
        elif typ == "Sell":
            if shares > 0 and sh > 0:
                avg = cost / shares
                proceeds = sell_value - fees
                cost_of_sold = avg * sh
                realized_pnl += proceeds - cost_of_sold
                cost -= cost_of_sold
                shares -= sh

        if bonus > 0:
            shares += bonus

    shares = max(shares, 0.0)

    if shares <= 0:
        return {
            **_empty,
            "cash_div": float(cash_div),
            "bonus_shares": float(bonus_total),
            "reinv": float(reinv),
            "realized_pnl": float(realized_pnl),
        }

    cost = max(cost, 0.0)
    return {
        "shares": float(shares),
        "cost_basis": float(cost),
        "avg_cost": float(cost / shares),
        "cash_div": float(cash_div),
        "bonus_shares": float(bonus_total),
        "reinv": float(reinv),
        "realized_pnl": float(realized_pnl),
        "position_open": True,
    }


# =====================================================================
#   PortfolioService — class wrapping ALL portfolio business logic
# =====================================================================

class PortfolioService:
    """
    CFA/IFRS-compliant portfolio calculations.
    Extracted from ui.py — NO Streamlit dependencies.

    Encapsulates:
      - Holdings & WAC cost basis
      - Portfolio table building
      - Cash reconciliation (5-source UNION ALL)
      - Realized profit details (dual-path WAC)
      - Overview aggregation
      - TWR / MWRR performance
      - Sharpe & Sortino risk ratios
    """

    def __init__(self, user_id: int):
        self.user_id = user_id

    # ------------------------------------------------------------------
    #  Holdings from transactions
    # ------------------------------------------------------------------

    def get_current_holdings(
        self,
        portfolio: Optional[str] = None,
        include_closed: bool = False,
    ) -> pd.DataFrame:
        """
        Single Source of Truth for position calculations.

        Returns DataFrame with:
            portfolio, stock_symbol, total_bought, total_sold, total_bonus,
            current_holding, total_cost, total_dividends, total_sell_value
        """
        soft_del = _soft_delete_filter()

        having = (
            "HAVING (SUM(CASE WHEN txn_type = 'Buy' THEN COALESCE(shares,0) ELSE 0 END)"
            " + SUM(COALESCE(bonus_shares,0))"
            " - SUM(CASE WHEN txn_type = 'Sell' THEN COALESCE(shares,0) ELSE 0 END)) > 0.001"
            if not include_closed
            else ""
        )
        pf = f"AND portfolio = '{portfolio}'" if portfolio else ""

        sql = f"""
            SELECT
                portfolio,
                stock_symbol,
                SUM(CASE WHEN txn_type='Buy'  THEN COALESCE(shares,0) ELSE 0 END) AS total_bought,
                SUM(CASE WHEN txn_type='Sell' THEN COALESCE(shares,0) ELSE 0 END) AS total_sold,
                SUM(COALESCE(bonus_shares,0))                                       AS total_bonus,
                SUM(CASE WHEN txn_type='Buy'  THEN COALESCE(shares,0) ELSE 0 END)
                  + SUM(COALESCE(bonus_shares,0))
                  - SUM(CASE WHEN txn_type='Sell' THEN COALESCE(shares,0) ELSE 0 END) AS current_holding,
                SUM(CASE WHEN txn_type='Buy' THEN COALESCE(purchase_cost,0) ELSE 0 END) AS total_cost,
                SUM(COALESCE(cash_dividend,0))                                            AS total_dividends,
                SUM(CASE WHEN txn_type='Sell' THEN COALESCE(sell_value,0) ELSE 0 END)    AS total_sell_value
            FROM transactions
            WHERE user_id = ? {pf} {soft_del}
            GROUP BY portfolio, stock_symbol
            {having}
            ORDER BY portfolio, stock_symbol
        """
        return query_df(sql, (self.user_id,))

    # ------------------------------------------------------------------
    #  Build portfolio table (per-portfolio)
    # ------------------------------------------------------------------

    def build_portfolio_table(self, portfolio_name: str) -> pd.DataFrame:
        """
        Build the full holdings table for *portfolio_name*.

        Mirrors ui.py ``build_portfolio_table()`` (L9418-9582):
          1. Fetch all transactions for portfolio
          2. Fetch stock metadata (price, name, currency)
          3. Run WAC engine per symbol
          4. Calculate unrealized P&L, total P&L, weights

        Returns DataFrame with 22+ columns per holding.
        """
        soft_del = _soft_delete_filter()

        all_txs = query_df(
            f"""
            SELECT
                id, TRIM(stock_symbol) AS stock_symbol, txn_date, txn_type,
                purchase_cost, sell_value, shares,
                bonus_shares, cash_dividend,
                price_override, planned_cum_shares,
                reinvested_dividend, fees,
                broker, reference, notes, created_at, portfolio
            FROM transactions
            WHERE user_id = ? AND COALESCE(category,'portfolio')='portfolio'
                  AND portfolio = ?
                  {soft_del}
            ORDER BY txn_date ASC, created_at ASC, id ASC
            """,
            (self.user_id, portfolio_name),
        )

        if all_txs.empty:
            return pd.DataFrame()

        unique_symbols = [s.strip() for s in all_txs["stock_symbol"].str.strip().unique()]

        # Fetch stock metadata
        stock_lookup: Dict[str, dict] = {}
        has_pe_col = column_exists("stocks", "pe_ratio")
        if unique_symbols:
            ph = ",".join(["?" for _ in unique_symbols])
            pe_select = ", pe_ratio" if has_pe_col else ""
            meta_df = query_df(
                f"""
                SELECT
                    TRIM(symbol) AS symbol,
                    COALESCE(name,'')            AS name,
                    COALESCE(current_price,0)    AS current_price,
                    COALESCE(portfolio,'KFH')    AS portfolio,
                    COALESCE(currency,'KWD')     AS currency,
                    tradingview_symbol, tradingview_exchange,
                    yf_ticker{pe_select}
                FROM stocks
                WHERE TRIM(symbol) IN ({ph}) AND user_id = ?
                """,
                tuple(unique_symbols) + (self.user_id,),
            )
            if not meta_df.empty:
                for _, srow in meta_df.iterrows():
                    stock_lookup[srow["symbol"].strip()] = {
                        "name": srow["name"],
                        "current_price": srow["current_price"],
                        "portfolio": srow["portfolio"],
                        "currency": srow["currency"],
                        "yf_ticker": srow.get("yf_ticker") or None,
                        "pe_ratio": srow.get("pe_ratio") or None,
                    }

        # Build rows per symbol
        rows: List[dict] = []
        for sym in unique_symbols:
            sym = sym.strip()
            meta = stock_lookup.get(sym, {
                "name": sym,
                "current_price": 0.0,
                "portfolio": portfolio_name,
                "currency": "USD" if portfolio_name == "USA" else "KWD",
            })

            cp = safe_float(meta.get("current_price", 0), 0.0)
            currency = meta.get("currency", "KWD")
            if portfolio_name == "USA":
                currency = "USD"

            tx = all_txs[all_txs["stock_symbol"].str.strip() == sym].copy()
            h = compute_holdings_avg_cost(tx)

            qty = h["shares"]
            if qty <= 0.001:
                continue

            total_cost = round(h["cost_basis"], 3)
            avg_cost = round(h["avg_cost"], 6)
            mkt_price = cp
            mkt_value = round(qty * mkt_price, 3)

            # Unrealized P&L = (Market Price − Avg Cost) × Shares
            unreal = (
                round((mkt_price - avg_cost) * qty, 3)
                if qty > 0 and mkt_price > 0
                else 0.0
            )
            realized_pnl = round(h["realized_pnl"], 3)
            cash_div = round(h["cash_div"], 3)
            bonus_sh = h["bonus_shares"]
            reinv_div = round(h["reinv"], 3)

            yield_pct = (cash_div / total_cost) if total_cost > 0 else 0.0
            total_pnl = round(unreal + realized_pnl + cash_div, 3)
            pnl_pct = (total_pnl / total_cost) if total_cost > 0 else 0.0

            display_name = meta.get("name") or sym

            # P/E ratio from stocks table. If missing, fetch from StockAnalysis and persist.
            pe_ratio = meta.get("pe_ratio")
            if pe_ratio is None:
                fetched_pe = _fetch_pe_from_stockanalysis(sym, currency)
                if fetched_pe is not None:
                    pe_ratio = fetched_pe
                    try:
                        exec_sql(
                            """
                            UPDATE stocks
                            SET pe_ratio = ?, last_updated = ?
                            WHERE TRIM(symbol) = ? AND user_id = ?
                            """,
                            (fetched_pe, int(time.time()), sym, self.user_id),
                        )
                    except Exception as exc:
                        logger.debug("Unable to persist StockAnalysis P/E for %s: %s", sym, exc)

            mkt_val_kwd = convert_to_kwd(mkt_value, currency)
            unreal_kwd = convert_to_kwd(unreal, currency)
            total_pnl_kwd = convert_to_kwd(total_pnl, currency)
            total_cost_kwd = convert_to_kwd(total_cost, currency)

            rows.append({
                "company": f"{display_name} - {sym}".strip(),
                "symbol": sym,
                "pe_ratio": pe_ratio,
                "shares_qty": qty,
                "avg_cost": avg_cost,
                "total_cost": total_cost,
                "market_price": mkt_price,
                "market_value": mkt_value,
                "unrealized_pnl": unreal,
                "realized_pnl": realized_pnl,
                "cash_dividends": cash_div,
                "reinvested_dividends": reinv_div,
                "bonus_dividend_shares": bonus_sh,
                "bonus_share_value": round(bonus_sh * mkt_price, 3),
                "dividend_yield_on_cost_pct": yield_pct,
                "total_pnl": total_pnl,
                "pnl_pct": pnl_pct,
                "currency": currency,
                "market_value_kwd": mkt_val_kwd,
                "unrealized_pnl_kwd": unreal_kwd,
                "total_pnl_kwd": total_pnl_kwd,
                "total_cost_kwd": total_cost_kwd,
            })

        df = pd.DataFrame(rows)

        if not df.empty:
            total_cost_sum = float(df["total_cost"].sum())
            df["weight_by_cost"] = df["total_cost"].apply(
                lambda x: (x / total_cost_sum) if total_cost_sum > 0 else 0.0
            )
            df["weighted_dividend_yield_on_cost"] = (
                df["dividend_yield_on_cost_pct"] * df["weight_by_cost"]
            )
            df = df.sort_values("total_cost", ascending=False).reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    #  Cash reconciliation (5-source UNION ALL — ui.py L10517-10640)
    # ------------------------------------------------------------------

    def recalc_portfolio_cash(
        self,
        force_override: bool = False,
        deposit_delta: Optional[float] = None,
        delta_portfolio: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Recalculate absolute cash balance for each portfolio.

        Formula:
            Cash = Σ Deposits − Σ Buy Costs + Σ Sell Values + Σ Dividends − Σ Fees

        Respects manual_override flag unless *force_override* is True.

        When *deposit_delta* is provided for a *delta_portfolio* that has
        manual_override=1, the stored balance is incremented by the delta
        instead of being skipped.  This ensures deposits / deletes always
        update the displayed cash even under manual override.

        Returns dict { portfolio_name: balance }.
        """
        conn = get_conn()
        try:
            cur = conn.cursor()

            manual_overrides: set = set()
            if not force_override:
                try:
                    cur.execute(
                        "SELECT portfolio FROM portfolio_cash "
                        "WHERE user_id = ? AND manual_override = 1",
                        (self.user_id,),
                    )
                    manual_overrides = {row[0] for row in cur.fetchall()}
                except Exception:
                    pass  # table may not exist yet

            agg_sql = """
                SELECT portfolio, SUM(net_change) AS total_change
                FROM (
                    SELECT portfolio,
                           CASE WHEN LOWER(COALESCE(source,'deposit')) = 'withdrawal'
                                THEN -1 * COALESCE(amount,0)
                                ELSE COALESCE(amount,0)
                           END AS net_change
                    FROM cash_deposits
                    WHERE user_id = ?
                      AND COALESCE(include_in_analysis,1) = 1
                      AND COALESCE(is_deleted,0) = 0

                    UNION ALL

                    SELECT portfolio, -1 * COALESCE(purchase_cost,0) AS net_change
                    FROM transactions
                    WHERE user_id = ? AND txn_type = 'Buy'
                      AND COALESCE(category,'portfolio') = 'portfolio'
                      AND COALESCE(is_deleted,0) = 0

                    UNION ALL

                    SELECT portfolio, COALESCE(sell_value,0) AS net_change
                    FROM transactions
                    WHERE user_id = ? AND txn_type = 'Sell'
                      AND COALESCE(category,'portfolio') = 'portfolio'
                      AND COALESCE(is_deleted,0) = 0

                    UNION ALL

                    SELECT portfolio, COALESCE(cash_dividend,0) AS net_change
                    FROM transactions
                    WHERE user_id = ? AND COALESCE(cash_dividend,0) > 0
                      AND COALESCE(category,'portfolio') = 'portfolio'
                      AND COALESCE(is_deleted,0) = 0

                    UNION ALL

                    SELECT portfolio, -1 * COALESCE(fees,0) AS net_change
                    FROM transactions
                    WHERE user_id = ? AND COALESCE(fees,0) > 0
                      AND COALESCE(category,'portfolio') = 'portfolio'
                      AND COALESCE(is_deleted,0) = 0
                ) AS cash_movements
                GROUP BY portfolio
            """
            cur.execute(agg_sql, (self.user_id,) * 5)
            results = cur.fetchall()

            balances: Dict[str, float] = {}
            now = int(time.time())

            for row in results:
                pf = row[0]
                balance = float(row[1]) if row[1] else 0.0
                if pf is None:
                    continue
                if pf in manual_overrides:
                    # If a deposit_delta is provided for THIS portfolio,
                    # increment the manual-override balance by the delta
                    # so deposits/deletes always affect displayed cash.
                    if deposit_delta is not None and delta_portfolio == pf:
                        try:
                            cur.execute(
                                "SELECT balance FROM portfolio_cash "
                                "WHERE user_id = ? AND portfolio = ?",
                                (self.user_id, pf),
                            )
                            old_row = cur.fetchone()
                            old_bal = float(old_row[0] or 0) if old_row else 0.0
                            new_bal = old_bal + deposit_delta
                            now_ts = int(time.time())
                            cur.execute(
                                "UPDATE portfolio_cash SET balance=?, last_updated=? "
                                "WHERE user_id=? AND portfolio=?",
                                (new_bal, now_ts, self.user_id, pf),
                            )
                            balances[pf] = new_bal
                            logger.info(
                                "Manual-override cash for %s updated: %.3f + %.3f = %.3f",
                                pf, old_bal, deposit_delta, new_bal,
                            )
                        except Exception as exc:
                            logger.debug("manual override delta update skipped: %s", exc)
                    continue
                balances[pf] = balance

                try:
                    cur.execute(
                        "SELECT 1 FROM portfolio_cash WHERE user_id = ? AND portfolio = ?",
                        (self.user_id, pf),
                    )
                    if cur.fetchone():
                        cur.execute(
                            "UPDATE portfolio_cash SET balance=?, last_updated=? "
                            "WHERE user_id=? AND portfolio=?",
                            (balance, now, self.user_id, pf),
                        )
                    else:
                        pf_ccy = PORTFOLIO_CCY.get(pf, "KWD")
                        cur.execute(
                            "INSERT INTO portfolio_cash "
                            "(user_id, portfolio, balance, currency, last_updated, manual_override) "
                            "VALUES (?,?,?,?,?,0)",
                            (self.user_id, pf, balance, pf_ccy, now),
                        )
                except Exception as exc:
                    logger.debug("portfolio_cash upsert skipped: %s", exc)

            # Zero out any non-manual portfolios that had no activity
            # (e.g. all transactions deleted → aggregate returned no rows)
            computed_portfolios = set(balances.keys())
            try:
                cur.execute(
                    "SELECT portfolio FROM portfolio_cash "
                    "WHERE user_id = ? AND COALESCE(manual_override, 0) = 0",
                    (self.user_id,),
                )
                for row in cur.fetchall():
                    pf = row[0]
                    if pf and pf not in computed_portfolios:
                        cur.execute(
                            "UPDATE portfolio_cash SET balance = 0, last_updated = ? "
                            "WHERE user_id = ? AND portfolio = ?",
                            (now, self.user_id, pf),
                        )
                        balances[pf] = 0.0
                        logger.info("Zeroed stale cash for portfolio %s", pf)
            except Exception as exc:
                logger.debug("stale portfolio_cash cleanup skipped: %s", exc)

            conn.commit()
            return balances

        except Exception as exc:
            logger.error("recalc_portfolio_cash failed: %s", exc)
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    #  Realized profit details (ui.py L20204-20398)
    # ------------------------------------------------------------------

    def calculate_realized_profit_details(self) -> dict:
        """
        Calculate realized profit from ALL sell transactions.

        Dual-path strategy (matches ui.py exactly):
          1. Use stored ``realized_pnl_at_txn`` if available in DB
          2. Fallback: runtime WAC calculation per (symbol, portfolio)

        Returns dict:
            total_realized_kwd, total_profit_kwd, total_loss_kwd, details (list)
        """
        soft_del = _soft_delete_filter("t")
        sql = f"""
            SELECT
                t.id, t.stock_symbol, t.txn_date, t.txn_type, t.shares,
                t.purchase_cost, t.sell_value, t.realized_pnl_at_txn,
                t.avg_cost_at_txn,
                COALESCE(t.portfolio, s.portfolio, 'KFH') AS portfolio,
                COALESCE(s.currency, 'KWD') AS currency,
                COALESCE(t.category, 'portfolio') AS category
            FROM transactions t
            LEFT JOIN stocks s
                ON UPPER(t.stock_symbol) = UPPER(s.symbol) AND s.user_id = t.user_id
            WHERE t.user_id = ? {soft_del}
            ORDER BY t.stock_symbol, t.portfolio, t.txn_date ASC, t.id ASC
        """

        try:
            df = query_df(sql, (self.user_id,))
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            return {
                "total_realized_kwd": 0.0,
                "total_profit_kwd": 0.0,
                "total_loss_kwd": 0.0,
                "details": [],
            }

        has_stored = (
            "realized_pnl_at_txn" in df.columns
            and df["realized_pnl_at_txn"].notna().any()
        )

        total_realized_kwd = 0.0
        total_profit_kwd = 0.0
        total_loss_kwd = 0.0
        details: List[dict] = []

        if has_stored:
            sells = df[df["txn_type"] == "Sell"].copy()
            for _, row in sells.iterrows():
                stored_pnl = row.get("realized_pnl_at_txn")
                if pd.isna(stored_pnl):
                    continue
                profit = float(stored_pnl)
                ccy = row.get("currency", "KWD")
                profit_kwd = convert_to_kwd(profit, ccy)
                total_realized_kwd += profit_kwd
                if profit_kwd >= 0:
                    total_profit_kwd += profit_kwd
                else:
                    total_loss_kwd += profit_kwd
                details.append({
                    "id": int(row["id"]),
                    "symbol": row["stock_symbol"],
                    "portfolio": row["portfolio"],
                    "txn_date": row["txn_date"],
                    "shares": safe_float(row["shares"]),
                    "sell_value": safe_float(row["sell_value"]),
                    "avg_cost_at_txn": safe_float(row.get("avg_cost_at_txn")),
                    "realized_pnl": profit,
                    "realized_pnl_kwd": profit_kwd,
                    "currency": ccy,
                    "source": "stored",
                })
        else:
            position_basis: Dict[Tuple[str, str], dict] = {}
            for _, row in df.iterrows():
                sym = str(row["stock_symbol"]).strip()
                pf = str(row.get("portfolio", "KFH"))
                typ = str(row["txn_type"])
                qty = safe_float(row.get("shares"), 0.0)
                ccy = row.get("currency", "KWD")
                key = (sym, pf)

                if key not in position_basis:
                    position_basis[key] = {"qty": 0.0, "total_cost": 0.0, "currency": ccy}

                if typ == "Buy":
                    cost = safe_float(row.get("purchase_cost"), 0.0)
                    position_basis[key]["qty"] += qty
                    position_basis[key]["total_cost"] += cost

                elif typ == "Sell" and qty > 0:
                    cur_qty = position_basis[key]["qty"]
                    cur_cost = position_basis[key]["total_cost"]
                    if cur_qty > 0:
                        avg_cost_ps = cur_cost / cur_qty
                        cost_of_sold = avg_cost_ps * qty
                        proceeds = safe_float(row.get("sell_value"), 0.0)
                        profit = proceeds - cost_of_sold
                        profit_kwd = convert_to_kwd(profit, ccy)

                        total_realized_kwd += profit_kwd
                        if profit_kwd >= 0:
                            total_profit_kwd += profit_kwd
                        else:
                            total_loss_kwd += profit_kwd

                        position_basis[key]["qty"] -= qty
                        position_basis[key]["total_cost"] -= cost_of_sold

                        details.append({
                            "id": int(row["id"]),
                            "symbol": sym,
                            "portfolio": pf,
                            "txn_date": row["txn_date"],
                            "shares": qty,
                            "sell_value": proceeds,
                            "avg_cost_at_txn": avg_cost_ps,
                            "realized_pnl": profit,
                            "realized_pnl_kwd": profit_kwd,
                            "currency": ccy,
                            "source": "calculated",
                        })

        return {
            "total_realized_kwd": round(total_realized_kwd, 3),
            "total_profit_kwd": round(total_profit_kwd, 3),
            "total_loss_kwd": round(total_loss_kwd, 3),
            "details": details,
        }

    # ------------------------------------------------------------------
    #  Portfolio overview (transaction aggregates)
    # ------------------------------------------------------------------

    def get_portfolio_overview(self, portfolio_id: Optional[int] = None) -> dict:
        """
        Comprehensive portfolio overview aggregated from the Streamlit-schema
        tables: ``transactions`` and ``cash_deposits``.

        Deposits/withdrawals      → cash_deposits
        Invested/divested/divs    → transactions (Buy/Sell/cash_dividend)
        Fees                      → transactions.fees
        """
        result: dict = {
            "total_deposits": 0.0,
            "total_withdrawals": 0.0,
            "net_deposits": 0.0,
            "total_invested": 0.0,
            "total_divested": 0.0,
            "total_dividends": 0.0,
            "cash_balance": 0.0,
            "total_fees": 0.0,
            "transaction_count": 0,
            "by_portfolio": {},
        }

        soft_del = _soft_delete_filter()

        try:
            conn = get_conn()
            cur = conn.cursor()

            # --- Deposits per portfolio (from cash_deposits) ---
            dep_sql = """
                SELECT portfolio,
                       COALESCE(SUM(CASE WHEN LOWER(COALESCE(source,'deposit')) != 'withdrawal' THEN amount ELSE 0 END), 0) AS deposits,
                       COALESCE(SUM(CASE WHEN LOWER(COALESCE(source,'deposit')) = 'withdrawal' THEN amount ELSE 0 END), 0) AS withdrawals
                FROM cash_deposits
                WHERE user_id = ?
                  AND COALESCE(include_in_analysis, 1) = 1
                  AND COALESCE(is_deleted, 0) = 0
                GROUP BY portfolio
            """
            cur.execute(dep_sql, (self.user_id,))
            dep_by_pf: Dict[str, dict] = {}
            for row in cur.fetchall():
                pf = row[0] or "KFH"
                dep_by_pf[pf] = {
                    "deposits": float(row[1] or 0),
                    "withdrawals": float(row[2] or 0),
                }

            # --- Transaction aggregates per portfolio ---
            txn_sql = f"""
                SELECT portfolio,
                       COALESCE(SUM(CASE WHEN txn_type='Buy'  THEN COALESCE(purchase_cost,0) ELSE 0 END), 0) AS total_invested,
                       COALESCE(SUM(CASE WHEN txn_type='Sell' THEN COALESCE(sell_value,0)    ELSE 0 END), 0) AS total_divested,
                       COALESCE(SUM(COALESCE(cash_dividend,0)), 0) AS total_dividends,
                       COALESCE(SUM(COALESCE(fees,0)),          0) AS total_fees,
                       COUNT(*) AS txn_count
                FROM transactions
                WHERE user_id = ?
                  AND COALESCE(category,'portfolio') = 'portfolio'
                  {soft_del}
                GROUP BY portfolio
            """
            cur.execute(txn_sql, (self.user_id,))
            txn_by_pf: Dict[str, dict] = {}
            for row in cur.fetchall():
                pf = row[0] or "KFH"
                txn_by_pf[pf] = {
                    "total_invested": float(row[1] or 0),
                    "total_divested": float(row[2] or 0),
                    "total_dividends": float(row[3] or 0),
                    "total_fees": float(row[4] or 0),
                    "txn_count": int(row[5] or 0),
                }

            # --- Merge data from both sources ---
            all_pfs = set(dep_by_pf.keys()) | set(txn_by_pf.keys())
            for pf in all_pfs:
                if pf not in PORTFOLIO_CCY:
                    continue  # skip unknown portfolio names
                ccy = PORTFOLIO_CCY.get(pf, "KWD")
                dep = dep_by_pf.get(pf, {"deposits": 0.0, "withdrawals": 0.0})
                txn = txn_by_pf.get(pf, {
                    "total_invested": 0.0, "total_divested": 0.0,
                    "total_dividends": 0.0, "total_fees": 0.0, "txn_count": 0,
                })

                deposits = dep["deposits"]
                withdrawals = dep["withdrawals"]
                invested = txn["total_invested"]
                divested = txn["total_divested"]
                dividends = txn["total_dividends"]
                fees = txn["total_fees"]
                txn_count = txn["txn_count"]

                result["by_portfolio"][pf] = {
                    "currency": ccy,
                    "total_deposits": deposits,
                    "total_withdrawals": withdrawals,
                    "net_deposits": deposits - withdrawals,
                    "total_invested": invested,
                    "total_divested": divested,
                    "total_dividends": dividends,
                    "total_fees": fees,
                    "transaction_count": txn_count,
                    # per-portfolio P&L placeholders filled by get_overview
                    "unrealized_pnl_kwd": 0.0,
                    "realized_pnl_kwd": 0.0,
                }

                result["total_deposits"] += convert_to_kwd(deposits, ccy)
                result["total_withdrawals"] += convert_to_kwd(withdrawals, ccy)
                result["total_invested"] += convert_to_kwd(invested, ccy)
                result["total_divested"] += convert_to_kwd(divested, ccy)
                result["total_dividends"] += convert_to_kwd(dividends, ccy)
                result["total_fees"] += convert_to_kwd(fees, ccy)
                result["transaction_count"] += txn_count

            result["net_deposits"] = (
                result["total_deposits"] - result["total_withdrawals"]
            )
            conn.close()

        except Exception as exc:
            logger.error("get_portfolio_overview error: %s", exc)
            result["error"] = str(exc)

        return result

    # ------------------------------------------------------------------
    #  Portfolio market value
    # ------------------------------------------------------------------

    def get_portfolio_value(self) -> dict:
        """Current market value of all portfolios, with KWD totals."""
        result: dict = {"total_value_kwd": 0.0, "by_portfolio": {}}

        for pname in PORTFOLIO_CCY:
            df = self.build_portfolio_table(pname)
            if df.empty:
                continue
            ccy = PORTFOLIO_CCY.get(pname, "KWD")
            port_value = float(
                df.loc[df["shares_qty"] > 0, "market_value"].sum()
            )
            holding_count = int((df["shares_qty"] > 0.001).sum())
            result["by_portfolio"][pname] = {
                "currency": ccy,
                "market_value": port_value,
                "market_value_kwd": convert_to_kwd(port_value, ccy),
                "holding_count": holding_count,
            }
            result["total_value_kwd"] += convert_to_kwd(port_value, ccy)

        return result

    # ------------------------------------------------------------------
    #  Total portfolio value — SINGLE SOURCE OF TRUTH
    # ------------------------------------------------------------------

    def get_total_portfolio_value(self) -> dict:
        """Calculate the total portfolio value (stocks + cash) in KWD.

        This is the **canonical** function used by Overview, Holdings,
        and Snapshot.  It mirrors the Streamlit save-snapshot logic:

        1. Sum stock market values from ``build_portfolio_table()``
           for each portfolio, converted to KWD.
        2. Add manual cash from ``portfolio_cash`` table, converted
           to KWD.
        3. ``total_value_kwd = stocks_kwd + cash_kwd``.

        Returns a dict with ``total_value_kwd``, ``stocks_kwd``,
        ``cash_kwd``, ``by_portfolio`` (stock breakdown), and
        ``accounts`` (cash breakdown).
        """
        values = self.get_portfolio_value()   # stocks only
        accounts = self.get_account_balances()  # cash only

        stocks_kwd = values["total_value_kwd"]
        cash_kwd = accounts["total_cash_kwd"]

        return {
            "total_value_kwd": round(stocks_kwd + cash_kwd, 3),
            "stocks_kwd": round(stocks_kwd, 3),
            "cash_kwd": round(cash_kwd, 3),
            "by_portfolio": values["by_portfolio"],
            "accounts": accounts["accounts"],
        }

    # ------------------------------------------------------------------
    #  Account balances (external_accounts)
    # ------------------------------------------------------------------

    def get_account_balances(self, portfolio_id: Optional[int] = None) -> dict:
        """
        Cash balances — always computed LIVE from the canonical formula:

            Cash = Σ Deposits − Σ Buy Costs + Σ Sell Values + Σ Dividends − Σ Fees

        This mirrors Streamlit's approach of computing cash fresh every time
        so that deposits, withdrawals, and transactions are immediately
        reflected in total portfolio value.

        Manual-override balances stored in ``portfolio_cash`` are respected
        (recalc_portfolio_cash skips those portfolios).
        """
        result: dict = {"total_cash_kwd": 0.0, "accounts": []}

        try:
            # Recalculate non-manual portfolios (respects manual_override)
            balances = self.recalc_portfolio_cash()  # force_override=False

            for pf, balance in balances.items():
                if pf not in PORTFOLIO_CCY:
                    continue
                ccy = PORTFOLIO_CCY.get(pf, "KWD")
                bal_kwd = convert_to_kwd(balance, ccy)
                result["accounts"].append({
                    "id": None,
                    "name": f"{pf} Cash",
                    "balance": balance,
                    "currency": ccy,
                    "balance_kwd": bal_kwd,
                    "last_reconciled": None,
                    "portfolio_name": pf,
                })
                result["total_cash_kwd"] += bal_kwd

            # Include manually-overridden portfolios that recalc skipped
            covered = set(balances.keys())
            try:
                conn = get_conn()
                cur = conn.cursor()
                cur.execute(
                    "SELECT portfolio, balance, currency FROM portfolio_cash "
                    "WHERE user_id = ? AND manual_override = 1",
                    (self.user_id,),
                )
                for row in cur.fetchall():
                    pf = row[0]
                    if pf not in PORTFOLIO_CCY or pf in covered:
                        continue
                    balance = float(row[1] or 0)
                    ccy = row[2] if len(row) > 2 and row[2] else PORTFOLIO_CCY.get(pf, "KWD")
                    bal_kwd = convert_to_kwd(balance, ccy)
                    result["accounts"].append({
                        "id": None,
                        "name": f"{pf} Cash",
                        "balance": balance,
                        "currency": ccy,
                        "balance_kwd": bal_kwd,
                        "last_reconciled": None,
                        "portfolio_name": pf,
                    })
                    result["total_cash_kwd"] += bal_kwd
                conn.close()
            except Exception:
                pass  # manual_override column may not exist

        except Exception as exc:
            logger.error("get_account_balances error: %s", exc)
            result["error"] = str(exc)

        return result

    # ------------------------------------------------------------------
    #  Complete overview (entry point for Overview tab)
    # ------------------------------------------------------------------

    def get_overview(self, portfolio_filter: Optional[str] = None) -> dict:
        """
        Combine transaction aggregates, market values, and account balances
        into a single payload for the frontend Overview screen.

        All monetary values in KWD.

        Uses ``get_total_portfolio_value()`` as the single source of truth
        so that Overview, Holdings, and Snapshot always agree.
        """
        overview = self.get_portfolio_overview()
        unified = self.get_total_portfolio_value()

        net_deposits = overview["net_deposits"]
        portfolio_value = unified["stocks_kwd"]
        cash_balance = unified["cash_kwd"]
        total_value = unified["total_value_kwd"]

        # Enrich by_portfolio with P&L from build_portfolio_table
        for pname in list(overview["by_portfolio"].keys()):
            ccy = PORTFOLIO_CCY.get(pname, "KWD")
            df = self.build_portfolio_table(pname)
            if not df.empty:
                unrealized = float(df["unrealized_pnl"].sum())
                realized = float(df["realized_pnl"].sum()) if "realized_pnl" in df.columns else 0.0
                dividends = float(df["cash_dividends"].sum()) if "cash_dividends" in df.columns else 0.0
                total_cost = float(df["total_cost"].sum()) if "total_cost" in df.columns else 0.0
                holding_count = int((df["shares_qty"] > 0.001).sum())
            else:
                unrealized = realized = dividends = total_cost = 0.0
                holding_count = 0

            overview["by_portfolio"][pname]["unrealized_pnl_kwd"] = convert_to_kwd(unrealized, ccy)
            overview["by_portfolio"][pname]["realized_pnl_kwd"] = convert_to_kwd(realized, ccy)
            overview["by_portfolio"][pname]["dividends_kwd"] = convert_to_kwd(dividends, ccy)
            overview["by_portfolio"][pname]["holding_count"] = holding_count
            overview["by_portfolio"][pname]["total_cost_kwd"] = convert_to_kwd(total_cost, ccy)

            # Merge market value data from unified calculation
            pv = unified["by_portfolio"].get(pname, {})
            overview["by_portfolio"][pname]["market_value"] = pv.get("market_value", 0.0)
            overview["by_portfolio"][pname]["market_value_kwd"] = pv.get("market_value_kwd", 0.0)

        # Use enriched by_portfolio as portfolio_values so PortfolioCard
        # has market_value, market_value_kwd, total_cost_kwd, holding_count, currency
        enriched_portfolio_values = {}
        for pname, bp_data in overview["by_portfolio"].items():
            enriched_portfolio_values[pname] = {
                "currency": bp_data.get("currency", "KWD"),
                "market_value": bp_data.get("market_value", 0.0),
                "market_value_kwd": bp_data.get("market_value_kwd", 0.0),
                "holding_count": bp_data.get("holding_count", 0),
                "total_cost_kwd": bp_data.get("total_cost_kwd", 0.0),
                "unrealized_pnl_kwd": bp_data.get("unrealized_pnl_kwd", 0.0),
                "realized_pnl_kwd": bp_data.get("realized_pnl_kwd", 0.0),
                "dividends_kwd": bp_data.get("dividends_kwd", 0.0),
            }

        return {
            "total_deposits": overview["total_deposits"],
            "total_withdrawals": overview["total_withdrawals"],
            "net_deposits": net_deposits,
            "total_invested": overview["total_invested"],
            "total_divested": overview["total_divested"],
            "total_dividends": overview["total_dividends"],
            "total_fees": overview["total_fees"],
            "transaction_count": overview["transaction_count"],
            "portfolio_value": portfolio_value,
            "cash_balance": cash_balance,
            "total_value": total_value,
            "total_gain": total_value - net_deposits,
            "roi_percent": (
                ((total_value / net_deposits) - 1) * 100
                if net_deposits > 0
                else 0.0
            ),
            "by_portfolio": overview["by_portfolio"],
            "portfolio_values": enriched_portfolio_values,
            "accounts": unified["accounts"],
            "usd_kwd_rate": get_usd_kwd_rate(),
            # Daily movement: live total value (stocks+cash) vs previous snapshot (matches Streamlit)
            **self._calc_daily_movement(total_value),
            # CAGR inputs: first deposit amount and date
            **self._calc_cagr_inputs(total_value),
            # MWRR (IRR) — calculated inline so the overview always has it
            "mwrr_percent": self._safe_mwrr(),
        }

    # ------------------------------------------------------------------
    #  Daily movement — LIVE (matches Streamlit)
    # ------------------------------------------------------------------

    def _calc_daily_movement(self, live_total_value: float) -> dict:
        """
        Compute daily movement as live_total_value - previous_snapshot.portfolio_value.
        Matches Streamlit's approach: live_portfolio_value (stocks + cash) compared
        against the previous day's stored snapshot portfolio_value (also stocks + cash).

        Uses ``snapshot_date < today`` so the comparison baseline is always
        yesterday's snapshot, regardless of whether today's snapshot exists.
        """
        from datetime import date as _date

        try:
            today_str = str(_date.today())
            prev = query_one(
                """SELECT portfolio_value
                   FROM portfolio_snapshots
                   WHERE user_id = ?
                     AND snapshot_date < ?
                   ORDER BY snapshot_date DESC
                   LIMIT 1""",
                (self.user_id, today_str),
            )
            if prev:
                prev_pv = prev[0] or 0.0
                if prev_pv > 0:
                    movement = live_total_value - prev_pv
                    pct = (movement / prev_pv) * 100
                    return {"daily_movement": movement, "daily_movement_pct": pct}
        except Exception:
            pass
        return {"daily_movement": 0.0, "daily_movement_pct": 0.0}

    # ------------------------------------------------------------------
    #  CAGR inputs — first deposit (matches Streamlit)
    # ------------------------------------------------------------------

    def _calc_cagr_inputs(self, live_total_value: float) -> dict:
        """
        Provide CAGR data matching Streamlit's calculation.

        Per CFA / GIPS: CAGR is NOT a performance metric — it ignores
        intermediate cash flows.  V_start = first deposit, V_end = current
        portfolio value, t = years since first deposit.

        Returns dict with cagr_percent pre-calculated, plus the raw inputs.
        """
        try:
            row = query_one(
                """SELECT deposit_date, amount
                   FROM cash_deposits
                   WHERE user_id = ?
                     AND deposit_date IS NOT NULL
                     AND deposit_date > '1971-01-01'
                     AND amount > 0
                     AND (include_in_analysis = 1 OR include_in_analysis IS NULL)
                     AND (is_deleted IS NULL OR is_deleted = 0)
                   ORDER BY deposit_date ASC
                   LIMIT 1""",
                (self.user_id,),
            )
            if row and row[0] and row[1]:
                from datetime import date as _date_type

                first_deposit_amount = float(row[1])
                first_deposit_date = str(row[0])
                d = _date_type.fromisoformat(first_deposit_date)
                today = _date_type.today()
                days = (today - d).days
                years = days / 365.25

                cagr_pct = 0.0
                if first_deposit_amount > 0 and years > 0 and live_total_value > 0:
                    cagr_pct = ((live_total_value / first_deposit_amount) ** (1 / years) - 1) * 100

                return {
                    "cagr_percent": cagr_pct,
                    "cagr_first_deposit": first_deposit_amount,
                    "cagr_first_date": first_deposit_date,
                }
        except Exception:
            pass
        return {"cagr_percent": 0.0, "cagr_first_deposit": 0.0, "cagr_first_date": None}

    # ------------------------------------------------------------------
    #  Safe MWRR wrapper for overview (never raises)
    # ------------------------------------------------------------------

    def _safe_mwrr(self) -> Optional[float]:
        """Compute MWRR for the overview response; returns None on any error."""
        try:
            return self.calculate_mwrr()
        except Exception as exc:
            logger.warning("_safe_mwrr: %s", exc)
            return None

    # ------------------------------------------------------------------
    #  External cash flows — shared by TWR & MWRR
    # ------------------------------------------------------------------

    def _get_external_flows(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        include_dividends: bool = False,
    ) -> pd.DataFrame:
        """Collect external cash flows from source tables (matches Streamlit).

        Returns DataFrame with columns: ``date``, ``amount``, ``type``.

        Sign convention (before sign flip):
          - DEPOSIT  amounts are positive in the DB
          - DIVIDEND amounts are positive in the DB
          - WITHDRAWAL amounts are positive in the DB

        ``calculate_twr`` and ``calculate_mwrr`` apply their own sign
        conventions.

        Sources (identical to Streamlit ui.py L21230-L21420):
          1. ``cash_deposits`` — deposits & withdrawals
          2. ``transactions`` where txn_type in (Deposit, Withdrawal,
             Transfer In, Transfer Out)
          3. ``transactions`` where cash_dividend > 0  (MWRR only)

        All USD amounts are converted to KWD.
        """
        frames: list = []
        date_filter = ""
        params: list = [self.user_id]
        if start_date:
            date_filter += " AND {col} >= ?"
        if end_date:
            date_filter += " AND {col} <= ?"

        def _params(col: str):
            p = [self.user_id]
            if start_date:
                p.append(start_date.isoformat())
            if end_date:
                p.append(end_date.isoformat())
            return tuple(p)

        # 1. Cash deposits / withdrawals
        try:
            dep_sql = f"""
                SELECT deposit_date AS date, amount,
                       COALESCE(currency, 'KWD') AS currency,
                       CASE WHEN amount >= 0 THEN 'DEPOSIT' ELSE 'WITHDRAWAL' END AS type
                FROM cash_deposits
                WHERE user_id = ?
                  AND deposit_date IS NOT NULL
                  AND amount != 0
                  AND deposit_date > '1971-01-01'
                  AND COALESCE(include_in_analysis, 1) = 1
                  AND COALESCE(is_deleted, 0) = 0
                  {date_filter.replace('{col}', 'deposit_date')}
            """
            dep_df = query_df(dep_sql, _params("deposit_date"))
            if not dep_df.empty:
                # Flip negative amounts (withdrawals) to positive
                wd_mask = dep_df["type"] == "WITHDRAWAL"
                if wd_mask.any():
                    dep_df.loc[wd_mask, "amount"] = dep_df.loc[wd_mask, "amount"].abs()
                # Convert USD → KWD
                usd_mask = dep_df["currency"].str.upper() == "USD"
                if usd_mask.any():
                    rate = get_usd_kwd_rate()
                    dep_df.loc[usd_mask, "amount"] = dep_df.loc[usd_mask, "amount"] * rate
                dep_df = dep_df[["date", "amount", "type"]]
                frames.append(dep_df)
        except Exception as exc:
            logger.warning("_get_external_flows: cash_deposits query failed: %s", exc)

        # 2. Ledger deposits (txn_type = 'Deposit' / category = 'FLOW_IN')
        try:
            ldep_sql = f"""
                SELECT txn_date AS date, purchase_cost AS amount, 'DEPOSIT' AS type
                FROM transactions
                WHERE user_id = ?
                  AND (txn_type = 'Deposit' OR category = 'FLOW_IN')
                  AND COALESCE(is_deleted, 0) = 0
                  {date_filter.replace('{col}', 'txn_date')}
            """
            ldep_df = query_df(ldep_sql, _params("txn_date"))
            if not ldep_df.empty:
                frames.append(ldep_df[["date", "amount", "type"]])
        except Exception as exc:
            logger.warning("_get_external_flows: ledger deposits query failed: %s", exc)

        # 3. Withdrawals (txn_type = 'Withdrawal' / category = 'FLOW_OUT')
        try:
            wd_sql = f"""
                SELECT txn_date AS date, sell_value AS amount, 'WITHDRAWAL' AS type
                FROM transactions
                WHERE user_id = ?
                  AND (txn_type = 'Withdrawal' OR category = 'FLOW_OUT')
                  AND COALESCE(is_deleted, 0) = 0
                  {date_filter.replace('{col}', 'txn_date')}
            """
            wd_df = query_df(wd_sql, _params("txn_date"))
            if not wd_df.empty:
                frames.append(wd_df[["date", "amount", "type"]])
        except Exception as exc:
            logger.warning("_get_external_flows: withdrawals query failed: %s", exc)

        # 4. Transfer In
        try:
            tin_sql = f"""
                SELECT txn_date AS date, COALESCE(purchase_cost, 0) AS amount,
                       'DEPOSIT' AS type
                FROM transactions
                WHERE user_id = ?
                  AND txn_type = 'Transfer In'
                  AND COALESCE(is_deleted, 0) = 0
                  {date_filter.replace('{col}', 'txn_date')}
            """
            tin_df = query_df(tin_sql, _params("txn_date"))
            if not tin_df.empty:
                frames.append(tin_df[["date", "amount", "type"]])
        except Exception as exc:
            logger.warning("_get_external_flows: transfer-in query failed: %s", exc)

        # 5. Transfer Out
        try:
            tout_sql = f"""
                SELECT txn_date AS date, COALESCE(sell_value, 0) AS amount,
                       'WITHDRAWAL' AS type
                FROM transactions
                WHERE user_id = ?
                  AND txn_type = 'Transfer Out'
                  AND COALESCE(is_deleted, 0) = 0
                  {date_filter.replace('{col}', 'txn_date')}
            """
            tout_df = query_df(tout_sql, _params("txn_date"))
            if not tout_df.empty:
                frames.append(tout_df[["date", "amount", "type"]])
        except Exception as exc:
            logger.warning("_get_external_flows: transfer-out query failed: %s", exc)

        # 6. Cash dividends (MWRR only — per CFA, dividends are
        #    investment returns, NOT external flows for TWR)
        if include_dividends:
            try:
                div_sql = f"""
                    SELECT txn_date AS date, COALESCE(cash_dividend, 0) AS amount,
                           'DIVIDEND' AS type
                    FROM transactions
                    WHERE user_id = ?
                      AND COALESCE(cash_dividend, 0) > 0
                      AND txn_date IS NOT NULL
                      AND txn_date > '1971-01-01'
                      AND COALESCE(is_deleted, 0) = 0
                      {date_filter.replace('{col}', 'txn_date')}
                """
                div_df = query_df(div_sql, _params("txn_date"))
                if not div_df.empty:
                    frames.append(div_df[["date", "amount", "type"]])
            except Exception as exc:
                logger.warning("_get_external_flows: dividends query failed: %s", exc)

        if not frames:
            return pd.DataFrame(columns=["date", "amount", "type"])

        combined = pd.concat(frames, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined["amount"] = combined["amount"].astype(float)
        return combined.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    #  TWR — Time-Weighted Return (GIPS Modified Dietz)
    # ------------------------------------------------------------------

    def calculate_twr(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        portfolio: Optional[str] = None,
    ) -> Optional[float]:
        """
        Time-Weighted Return using Modified Dietz (GIPS midpoint weighting).

        Matches Streamlit's TWR logic:
        - Subperiods defined by snapshots in ``portfolio_snapshots``
        - External flows: deposits, withdrawals, transfers (NOT dividends)
        - Flows are summed per subperiod and midpoint-weighted
        - Appends a virtual "today" endpoint with LIVE portfolio value
          so the return reflects current market prices (CFA-compliant).

        Subperiod return:
            R_i = (MV_end − MV_begin − CF) / (MV_begin + CF × 0.5)

        Geometric linking:
            TWR = Π(1 + R_i) − 1

        Returns percentage (e.g. 11.74 for 11.74%).
        """
        df = self._fetch_snapshots(start_date, end_date, portfolio)
        if df is None or len(df) < 2:
            return None

        # Append a virtual "today" data point with live portfolio value
        # so TWR covers inception-to-now (CFA: ending MV = current market value)
        live = self.get_total_portfolio_value()
        live_value = live.get("total_value_kwd", 0.0)
        today_ts = pd.Timestamp.now().normalize()  # midnight today
        last_snap_date = df.iloc[-1]["snapshot_date"]
        if live_value > 0 and today_ts > last_snap_date:
            today_row = pd.DataFrame(
                [{"snapshot_date": today_ts, "portfolio_value": live_value}]
            )
            df = pd.concat([df, today_row], ignore_index=True)

        # Get external flows (deposits/withdrawals only — no dividends per GIPS)
        flows = self._get_external_flows(start_date, end_date, include_dividends=False)

        cumulative = 1.0
        for i in range(1, len(df)):
            v_begin = float(df.iloc[i - 1]["portfolio_value"] or 0)
            v_end = float(df.iloc[i]["portfolio_value"] or 0)
            d_begin = df.iloc[i - 1]["snapshot_date"]
            d_end = df.iloc[i]["snapshot_date"]

            # Sum external flows in (begin, end] — deposits positive, withdrawals negative
            if not flows.empty:
                mask = (flows["date"] > d_begin) & (flows["date"] <= d_end)
                period_flows = flows[mask]
                net_cf = 0.0
                for _, f in period_flows.iterrows():
                    if f["type"] == "DEPOSIT":
                        net_cf += float(f["amount"])
                    elif f["type"] in ("WITHDRAWAL",):
                        net_cf -= float(f["amount"])
            else:
                net_cf = 0.0

            adjusted_begin = v_begin + net_cf * 0.5
            if adjusted_begin <= 0:
                continue

            sub_return = (v_end - v_begin - net_cf) / adjusted_begin
            cumulative *= (1 + sub_return)

        return round((cumulative - 1) * 100, 4)

    # ------------------------------------------------------------------
    #  MWRR — Money-Weighted Return (XIRR Newton + bisection)
    # ------------------------------------------------------------------

    def calculate_mwrr(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        portfolio: Optional[str] = None,
    ) -> Optional[float]:
        """
        Money-Weighted Rate of Return (XIRR) — CFA Level III compliant.

        Per the CFA curriculum (GIPS & Performance Measurement):
        MWRR is the internal rate of return (IRR) that equates the PV of all
        cash flows (deposits, withdrawals, dividends) to the terminal market
        value.  It is equivalent to dollar-weighted return and captures both
        the timing *and* magnitude of portfolio cash flows.

        Algorithm: XIRR via Newton-Raphson with bisection fallback.

        Cash flow sign convention (investor's perspective):
        - Deposits / Transfers In  → negative (money *from* investor)
        - Withdrawals / Transfers Out → positive (money *to* investor)
        - Cash dividends received → positive (return *to* investor)
        - Terminal portfolio value → positive (simulated liquidation)

        Day-count: ACT/365.25 (standard for XIRR, matches Excel / CFA)

        Terminal value: LIVE market value (stocks + cash in KWD).
        Fallback: latest portfolio_snapshot value.

        Returns annualised IRR as percentage (e.g. 11.74 for 11.74%),
        or None if insufficient data.
        """
        # ── 1. Terminal value ────────────────────────────────────────
        # CFA-compliant: use LIVE portfolio value (mark-to-market).
        # Fall back to the latest snapshot if live is unavailable.
        current_value = 0.0
        try:
            live = self.get_total_portfolio_value()
            current_value = live.get("total_value_kwd", 0.0)
        except Exception as exc:
            logger.warning("MWRR: live portfolio value failed: %s", exc)

        if current_value <= 0:
            # Fallback: latest snapshot portfolio_value
            try:
                snap = query_one(
                    """SELECT portfolio_value FROM portfolio_snapshots
                       WHERE user_id = ? ORDER BY snapshot_date DESC LIMIT 1""",
                    (self.user_id,),
                )
                if snap and snap[0]:
                    current_value = float(snap[0])
                    logger.info("MWRR: using snapshot fallback value %.2f", current_value)
            except Exception as exc:
                logger.warning("MWRR: snapshot fallback failed: %s", exc)

        if current_value <= 0:
            logger.warning("MWRR: terminal value is 0; cannot compute IRR")
            return None

        # ── 2. Collect external cash flows ───────────────────────────
        flows = self._get_external_flows(start_date, end_date, include_dividends=True)
        if flows.empty:
            logger.warning("MWRR: no external cash flows found")
            return None

        # ── 3. Build signed cash flow series ─────────────────────────
        cf_dates: list = []
        cf_amounts: list = []

        for _, row in flows.iterrows():
            amt = float(row["amount"])
            if amt == 0:
                continue
            cf_type = str(row["type"]).upper()
            if cf_type == "DEPOSIT":
                cf_amounts.append(-abs(amt))       # money leaving investor
            elif cf_type in ("DIVIDEND", "WITHDRAWAL"):
                cf_amounts.append(abs(amt))         # money returning
            else:
                continue
            cf_dates.append(pd.to_datetime(row["date"]))

        # Terminal value (simulated full liquidation at market)
        today = pd.Timestamp.now()
        cf_dates.append(today)
        cf_amounts.append(abs(current_value))

        if len(cf_dates) < 2:
            logger.warning("MWRR: fewer than 2 cash flows after filtering")
            return None

        # Must have at least one negative AND one positive flow
        if not (any(c < 0 for c in cf_amounts) and any(c > 0 for c in cf_amounts)):
            logger.warning("MWRR: cash flows are all same sign; IRR undefined")
            return None

        # ── 4. Aggregate same-day flows ──────────────────────────────
        pairs = sorted(zip(cf_dates, cf_amounts), key=lambda x: x[0])
        combined_dates: list = []
        combined_amounts: list = []
        prev_dt = None
        running = 0.0
        for dt, a in pairs:
            if prev_dt is None:
                prev_dt, running = dt, a
            elif dt == prev_dt:
                running += a
            else:
                combined_dates.append(prev_dt)
                combined_amounts.append(running)
                prev_dt, running = dt, a
        if prev_dt is not None:
            combined_dates.append(prev_dt)
            combined_amounts.append(running)

        cf_dates = combined_dates
        cf_amounts = combined_amounts

        # ── 5. Year-fraction (ACT/365.25) ────────────────────────────
        t0 = cf_dates[0]
        year_fracs = [(dt - t0).days / 365.25 for dt in cf_dates]

        # ── 6. XIRR solver ──────────────────────────────────────────
        def npv(r: float) -> float:
            if r <= -1.0:
                return float("inf")
            return sum(a / ((1.0 + r) ** t) for a, t in zip(cf_amounts, year_fracs))

        def npv_d(r: float) -> float:
            if r <= -1.0:
                return float("inf")
            return sum(-t * a / ((1.0 + r) ** (t + 1.0)) for a, t in zip(cf_amounts, year_fracs))

        # 6a. Newton-Raphson (primary) — initial guess 10%, max 200 iter
        rate = 0.10
        converged = False
        for _ in range(200):
            f = npv(rate)
            fp = npv_d(rate)
            if abs(fp) < 1e-14:
                break
            r_next = max(-0.9999, min(rate - f / fp, 100.0))
            if abs(r_next - rate) < 1e-10:
                if abs(npv(r_next)) < 0.01:
                    converged = True
                    rate = r_next
                break
            rate = r_next

        if converged:
            result = round(rate * 100, 4)
            logger.info("MWRR (Newton): %.4f%% from %d flows", result, len(cf_dates))
            return result

        # 6b. Bisection fallback — wider search
        lo, hi = -0.9999, 10.0
        npv_lo = npv(lo)
        if npv_lo * npv(hi) > 0:
            for test_hi in [20.0, 50.0, 100.0]:
                if npv(lo) * npv(test_hi) < 0:
                    hi = test_hi
                    break
            else:
                if abs(npv(rate)) < 1.0 and -0.99 < rate < 100:
                    result = round(rate * 100, 4)
                    logger.info("MWRR (approx): %.4f%%", result)
                    return result
                logger.warning("MWRR: bisection bracket not found")
                return None

        for _ in range(1000):
            mid = (lo + hi) / 2.0
            npv_mid = npv(mid)
            if abs(npv_mid) < 1e-8:
                result = round(mid * 100, 4)
                logger.info("MWRR (bisection): %.4f%%", result)
                return result
            if npv(lo) * npv_mid < 0:
                hi = mid
            else:
                lo = mid

        result = round((lo + hi) / 2.0 * 100, 4)
        logger.info("MWRR (bisection final): %.4f%%", result)
        return result

    # ------------------------------------------------------------------
    #  Performance (combines TWR + MWRR + ROI for a period)
    # ------------------------------------------------------------------

    def calculate_performance(
        self,
        period: str = "all",
        portfolio: Optional[str] = None,
    ) -> dict:
        """
        Calculate TWR, MWRR, and ROI for a time period.

        period: "all", "ytd", "1y", "6m", "3m", "1m"
        """
        today = date.today()
        start: Optional[date] = None

        if period == "ytd":
            start = today.replace(month=1, day=1)
        elif period == "1y":
            start = today - timedelta(days=365)
        elif period == "6m":
            start = today - timedelta(days=182)
        elif period == "3m":
            start = today - timedelta(days=91)
        elif period == "1m":
            start = today - timedelta(days=30)

        twr = self.calculate_twr(start, today, portfolio)
        mwrr = self.calculate_mwrr(start, today, portfolio)

        df = self._fetch_snapshots(start, today, portfolio)
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            earliest = df.iloc[0]
            start_val = float(earliest.get("portfolio_value") or 0)
            end_val = float(latest.get("portfolio_value") or 0)
            net_dep = float(latest.get("deposit_cash") or 0)
            net_gain = float(latest.get("net_gain") or 0)
            roi = float(latest.get("roi_percent") or 0)
        else:
            start_val = end_val = net_dep = net_gain = roi = 0.0

        return {
            "period": period,
            "start_date": start.isoformat() if start else None,
            "end_date": today.isoformat(),
            "twr_percent": twr,
            "mwrr_percent": mwrr,
            "roi_percent": roi,
            "total_gain_kwd": net_gain,
            "starting_value": start_val,
            "ending_value": end_val,
            "net_deposits": net_dep,
            "snapshots_count": len(df) if df is not None else 0,
        }

    # ------------------------------------------------------------------
    #  Sharpe Ratio  (ui.py L20059-20111)
    # ------------------------------------------------------------------

    def calculate_sharpe_ratio(self, rf_rate: float) -> Optional[float]:
        """
        Sharpe Ratio from portfolio snapshot returns.

        Sharpe = mean(Rp − Rf) / std(Rp − Rf) × √N

        Where N = annualization factor:
          252 (daily), 52 (weekly), 12 (monthly)

        Period Rf = (1 + rf_annual)^(1/N) − 1

        rf_rate must be provided by the caller (user sets it manually).
        """
        df = query_df(
            "SELECT snapshot_date, portfolio_value "
            "FROM portfolio_snapshots "
            "WHERE user_id = ? ORDER BY snapshot_date ASC",
            (self.user_id,),
        )
        if df.empty or len(df) < 2:
            return None

        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        avg_days = df["snapshot_date"].diff().dt.days.mean()

        # Edge case: NaN or non-positive avg_days → default to daily
        # (matches Streamlit: avg_days = 1 fallback)
        if pd.isna(avg_days) or avg_days <= 0:
            avg_days = 1

        if avg_days > 25:
            annual_factor = 12   # monthly
        elif avg_days > 5:
            annual_factor = 52   # weekly
        else:
            annual_factor = 252  # daily

        df["period_return"] = df["portfolio_value"].pct_change()
        df = df.dropna(subset=["period_return"])

        if df.empty:
            return None

        period_rf = (1 + rf_rate) ** (1 / annual_factor) - 1
        df["excess_return"] = df["period_return"] - period_rf

        mean_excess = df["excess_return"].mean()
        std_excess = df["excess_return"].std()

        if std_excess == 0:
            return 0.0

        return float(round((mean_excess / std_excess) * np.sqrt(annual_factor), 4))

    # ------------------------------------------------------------------
    #  Sortino Ratio  (ui.py L20133-20195)
    # ------------------------------------------------------------------

    def calculate_sortino_ratio(self, mar: float = 0.0) -> Optional[float]:
        """
        Sortino Ratio — penalizes only downside volatility.

        Matches Streamlit's calculate_sortino_ratio() (ui.py L20133-20196).

        Sortino = mean(Rp − MAR) / downside_std × √N

        Where downside_std = std(min(Rp − MAR, 0))  [population std, ddof=0]
        MAR (Minimum Acceptable Return) defaults to 0% (break-even).
        """
        df = query_df(
            "SELECT snapshot_date, portfolio_value "
            "FROM portfolio_snapshots "
            "WHERE user_id = ? ORDER BY snapshot_date ASC",
            (self.user_id,),
        )
        if df.empty or len(df) < 2:
            return None

        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        avg_days = df["snapshot_date"].diff().dt.days.mean()

        # Edge case: NaN or non-positive avg_days → default to daily
        # (matches Streamlit: avg_days = 1 fallback)
        if pd.isna(avg_days) or avg_days <= 0:
            avg_days = 1

        if avg_days > 25:
            annual_factor = 12
        elif avg_days > 5:
            annual_factor = 52
        else:
            annual_factor = 252

        df["period_return"] = df["portfolio_value"].pct_change()
        df = df.dropna(subset=["period_return"])

        if df.empty:
            return None

        df["excess_return"] = df["period_return"] - mar
        negative_returns = np.minimum(df["excess_return"].values, 0)
        downside_std = float(np.std(negative_returns))

        if downside_std == 0:
            return 10.0  # capped — no downside volatility observed

        mean_excess = df["excess_return"].mean()
        return float(round((mean_excess / downside_std) * np.sqrt(annual_factor), 4))

    # ------------------------------------------------------------------
    #  All holdings (cross-portfolio convenience)
    # ------------------------------------------------------------------

    def get_all_holdings(
        self, portfolio: Optional[str] = None,
    ) -> Tuple[List[dict], dict]:
        """
        Fetch holdings across all (or one) portfolio(s).

        Returns (holdings_list, totals_dict).
        """
        portfolios = [portfolio] if portfolio else list(PORTFOLIO_CCY.keys())

        all_holdings: List[dict] = []
        totals = {
            "total_market_value_kwd": 0.0,
            "total_cost_kwd": 0.0,
            "total_unrealized_pnl_kwd": 0.0,
            "total_realized_pnl_kwd": 0.0,
            "total_pnl_kwd": 0.0,
            "total_dividends_kwd": 0.0,
        }

        for pname in portfolios:
            df = self.build_portfolio_table(pname)
            if df.empty:
                continue
            for _, row in df.iterrows():
                h = row.to_dict()
                all_holdings.append(h)
                ccy = h.get("currency", "KWD")
                totals["total_market_value_kwd"] += float(h.get("market_value_kwd", 0))
                totals["total_cost_kwd"] += float(h.get("total_cost_kwd", 0))
                totals["total_unrealized_pnl_kwd"] += float(h.get("unrealized_pnl_kwd", 0))
                totals["total_realized_pnl_kwd"] += convert_to_kwd(
                    float(h.get("realized_pnl", 0)), ccy,
                )
                totals["total_pnl_kwd"] += float(h.get("total_pnl_kwd", 0))
                totals["total_dividends_kwd"] += convert_to_kwd(
                    float(h.get("cash_dividends", 0)), ccy,
                )

        return all_holdings, totals

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _fetch_snapshots(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        portfolio: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch portfolio_snapshots filtered by date range."""
        conditions = ["user_id = ?"]
        params: list = [self.user_id]

        # Note: portfolio_snapshots has no 'portfolio' column;
        # snapshots are aggregated per-user.
        if start_date:
            conditions.append("snapshot_date >= ?")
            params.append(start_date.isoformat())
        if end_date:
            conditions.append("snapshot_date <= ?")
            params.append(end_date.isoformat())

        where = " AND ".join(conditions)
        df = query_df(
            f"""
            SELECT snapshot_date, portfolio_value, deposit_cash,
                   net_gain, change_percent, roi_percent,
                   twr_percent, mwrr_percent
            FROM portfolio_snapshots
            WHERE {where}
            ORDER BY snapshot_date ASC
            """,
            tuple(params),
        )
        if df.empty:
            return None

        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        return df


# =====================================================================
#   Backward-compatible module-level functions
#   (so existing route imports keep working unmodified)
# =====================================================================

def get_current_holdings(
    user_id: int, portfolio: Optional[str] = None, include_closed: bool = False,
) -> pd.DataFrame:
    return PortfolioService(user_id).get_current_holdings(portfolio, include_closed)


def build_portfolio_table(portfolio_name: str, user_id: int) -> pd.DataFrame:
    return PortfolioService(user_id).build_portfolio_table(portfolio_name)


def get_portfolio_overview(user_id: int, portfolio_id: Optional[int] = None) -> dict:
    return PortfolioService(user_id).get_portfolio_overview(portfolio_id)


def get_portfolio_value(user_id: int) -> dict:
    return PortfolioService(user_id).get_portfolio_value()


def get_account_balances(user_id: int, portfolio_id: Optional[int] = None) -> dict:
    return PortfolioService(user_id).get_account_balances(portfolio_id)


def get_total_portfolio_value(user_id: int) -> dict:
    return PortfolioService(user_id).get_total_portfolio_value()


def get_complete_overview(user_id: int) -> dict:
    return PortfolioService(user_id).get_overview()
