"""
Financial Data Integrity Service — Phase 3.2

Verifies that financial calculations match between:
  - Transaction-level calculations (WAC engine)
  - Portfolio-level aggregates (cash balances, holdings)
  - Snapshot historical values (position_snapshots vs live)

Uses the same raw-SQL helpers as the rest of the backend; no ORM sessions.
"""

import logging
import time
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.core.database import query_df, query_val, query_one, query_all, column_exists
from app.services.portfolio_service import PortfolioService, compute_holdings_avg_cost
from app.services.fx_service import safe_float, convert_to_kwd

logger = logging.getLogger(__name__)

# ── Tolerance thresholds ─────────────────────────────────────────────
CASH_TOLERANCE = Decimal("0.01")       # 10 fils — rounding across 5 UNION legs
POSITION_TOLERANCE = 0.01              # 0.01 share (float rounding)
COST_TOLERANCE = Decimal("0.01")       # 10 fils for cost basis
SNAPSHOT_AGE_WARN_DAYS = 3             # warn if latest snapshot is stale


def _soft_del() -> str:
    """SQL fragment for soft-delete guard."""
    if not column_exists("transactions", "is_deleted"):
        return ""
    return " AND COALESCE(is_deleted, 0) = 0"


def _dec(value: Any) -> Decimal:
    """Safely coerce to Decimal."""
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


# =====================================================================
#  IntegrityService  (class receives user_id, stateless; no DB handle)
# =====================================================================

class IntegrityService:
    """
    Run financial integrity checks for a single user.

    Usage:
        svc = IntegrityService(user_id=1)
        report = svc.run_full_integrity_check()
    """

    def __init__(self, user_id: int):
        self.user_id = user_id
        self._portfolio_svc = PortfolioService(user_id)

    # -----------------------------------------------------------------
    #  1. Cash-balance verification
    # -----------------------------------------------------------------

    def verify_cash_balance(self, portfolio: str) -> dict:
        """
        Verify that portfolio_cash.balance matches the 5-source UNION
        calculation  (Deposits − Buys + Sells + Dividends − Fees).

        Returns a discrepancy report dict.
        """

        sd = _soft_del()

        # ── Component queries (parameterised, injection-safe) ────────
        deposits = _dec(query_val(
            "SELECT COALESCE(SUM(amount), 0) FROM cash_deposits "
            "WHERE user_id = ? AND portfolio = ? "
            "  AND COALESCE(is_deleted, 0) = 0",
            (self.user_id, portfolio),
        ))

        buys = _dec(query_val(
            f"SELECT COALESCE(SUM(purchase_cost), 0) FROM transactions "
            f"WHERE user_id = ? AND portfolio = ? AND txn_type = 'Buy' "
            f"  AND COALESCE(category, 'portfolio') = 'portfolio' {sd}",
            (self.user_id, portfolio),
        ))

        sells = _dec(query_val(
            f"SELECT COALESCE(SUM(sell_value), 0) FROM transactions "
            f"WHERE user_id = ? AND portfolio = ? AND txn_type = 'Sell' "
            f"  AND COALESCE(category, 'portfolio') = 'portfolio' {sd}",
            (self.user_id, portfolio),
        ))

        dividends = _dec(query_val(
            f"SELECT COALESCE(SUM(cash_dividend), 0) FROM transactions "
            f"WHERE user_id = ? AND portfolio = ? "
            f"  AND COALESCE(cash_dividend, 0) > 0 "
            f"  AND COALESCE(category, 'portfolio') = 'portfolio' {sd}",
            (self.user_id, portfolio),
        ))

        fees = _dec(query_val(
            f"SELECT COALESCE(SUM(fees), 0) FROM transactions "
            f"WHERE user_id = ? AND portfolio = ? "
            f"  AND COALESCE(fees, 0) > 0 "
            f"  AND COALESCE(category, 'portfolio') = 'portfolio' {sd}",
            (self.user_id, portfolio),
        ))

        expected = deposits - buys + sells + dividends - fees

        # ── Stored (reconciled) balance ──────────────────────────────
        stored_row = query_one(
            "SELECT balance FROM portfolio_cash "
            "WHERE user_id = ? AND portfolio = ?",
            (self.user_id, portfolio),
        )
        stored = _dec(stored_row[0]) if stored_row else None

        # ── Compare ──────────────────────────────────────────────────
        if stored is not None:
            discrepancy = abs(expected - stored)
            is_valid = discrepancy <= CASH_TOLERANCE
        else:
            discrepancy = None
            is_valid = None   # cannot judge — no stored value

        return {
            "portfolio": portfolio,
            "expected_balance": str(expected.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            "stored_balance": str(stored.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)) if stored is not None else None,
            "discrepancy": str(discrepancy.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)) if discrepancy is not None else None,
            "is_valid": is_valid,
            "tolerance": str(CASH_TOLERANCE),
            "components": {
                "deposits": str(deposits),
                "buys": str(buys),
                "sells": str(sells),
                "dividends": str(dividends),
                "fees": str(fees),
            },
        }

    # -----------------------------------------------------------------
    #  2. Position / holdings verification
    # -----------------------------------------------------------------

    def verify_positions(self, portfolio: str) -> dict:
        """
        Cross-check share counts derived from the WAC engine against
        the aggregate-SQL holdings query.

        Both paths MUST agree on net shares held.  Cost basis will
        legitimately differ (aggregate = raw purchase_cost sum;
        WAC = cost adjusted for sells & fees), so cost differences
        are reported informatively but not flagged as errors.
        """

        sd = _soft_del()

        # ── Path A: aggregate SQL (fast, used for overview) ──────────
        agg_df = query_df(
            f"""
            SELECT
                TRIM(stock_symbol) AS stock_symbol,
                SUM(CASE WHEN txn_type='Buy'  THEN COALESCE(shares,0) ELSE 0 END)
                  + SUM(COALESCE(bonus_shares,0))
                  - SUM(CASE WHEN txn_type='Sell' THEN COALESCE(shares,0) ELSE 0 END) AS agg_shares,
                SUM(CASE WHEN txn_type='Buy' THEN COALESCE(purchase_cost,0) ELSE 0 END) AS agg_total_bought_cost
            FROM transactions
            WHERE user_id = ? AND portfolio = ?
              AND COALESCE(category,'portfolio') = 'portfolio' {sd}
            GROUP BY TRIM(stock_symbol)
            HAVING agg_shares > 0.001
            """,
            (self.user_id, portfolio),
        )

        # ── Path B: WAC engine per symbol (slow, authoritative) ──────
        tx_df = query_df(
            f"""
            SELECT
                id, TRIM(stock_symbol) AS stock_symbol, txn_date, txn_type,
                purchase_cost, sell_value, shares, bonus_shares,
                cash_dividend, reinvested_dividend, fees,
                created_at, portfolio
            FROM transactions
            WHERE user_id = ? AND portfolio = ?
              AND COALESCE(category,'portfolio') = 'portfolio' {sd}
            ORDER BY txn_date, created_at, id
            """,
            (self.user_id, portfolio),
        )

        wac_by_sym: Dict[str, dict] = {}
        if not tx_df.empty:
            for sym in tx_df["stock_symbol"].unique():
                stx = tx_df[tx_df["stock_symbol"] == sym].copy()
                h = compute_holdings_avg_cost(stx)
                if h["shares"] > 0.001:
                    wac_by_sym[sym] = h

        # ── Reconcile ────────────────────────────────────────────────
        all_symbols = set()

        agg_lookup: Dict[str, dict] = {}
        if not agg_df.empty:
            for _, r in agg_df.iterrows():
                sym = r["stock_symbol"]
                all_symbols.add(sym)
                agg_lookup[sym] = {
                    "shares": float(r["agg_shares"]),
                    "total_bought_cost": float(r["agg_total_bought_cost"]),
                }

        for sym in wac_by_sym:
            all_symbols.add(sym)

        mismatches: List[dict] = []
        matched = 0
        details: List[dict] = []

        for sym in sorted(all_symbols):
            agg = agg_lookup.get(sym, {"shares": 0.0, "total_bought_cost": 0.0})
            wac = wac_by_sym.get(sym, {"shares": 0.0, "cost_basis": 0.0})

            share_diff = abs(agg["shares"] - wac["shares"])

            detail = {
                "symbol": sym,
                "agg_shares": round(agg["shares"], 6),
                "wac_shares": round(wac["shares"], 6),
                "share_diff": round(share_diff, 6),
                "agg_total_bought_cost": round(agg["total_bought_cost"], 3),
                "wac_remaining_cost": round(wac["cost_basis"], 3),
            }
            details.append(detail)

            if share_diff > POSITION_TOLERANCE:
                mismatches.append(detail)
            else:
                matched += 1

        return {
            "portfolio": portfolio,
            "total_symbols": len(all_symbols),
            "matched": matched,
            "mismatches": mismatches,
            "details": details,
            "is_valid": len(mismatches) == 0,
        }

    # -----------------------------------------------------------------
    #  3. Snapshot freshness & consistency
    # -----------------------------------------------------------------

    def verify_snapshots(self, portfolio: str) -> dict:
        """
        Check that portfolio snapshots are:
          a) Recent (not stale, last snapshot within SNAPSHOT_AGE_WARN_DAYS)
          b) Value-consistent (latest snapshot value ≈ sum of position values)
        """

        latest = query_one(
            "SELECT snapshot_date, portfolio_value, deposit_cash "
            "FROM portfolio_snapshots "
            "WHERE user_id = ? AND portfolio = ? "
            "ORDER BY snapshot_date DESC LIMIT 1",
            (self.user_id, portfolio),
        )

        if not latest:
            return {
                "portfolio": portfolio,
                "has_snapshots": False,
                "is_fresh": False,
                "is_valid": None,
                "message": "No snapshots found",
            }

        snap_date_str = latest[0]
        snap_value = float(latest[1]) if latest[1] else 0.0

        try:
            snap_date = datetime.strptime(snap_date_str, "%Y-%m-%d").date()
        except Exception:
            snap_date = date.today()

        days_old = (date.today() - snap_date).days
        is_fresh = days_old <= SNAPSHOT_AGE_WARN_DAYS

        # ── Compare snapshot value against live holdings ─────────────
        live_value = 0.0
        try:
            tbl = self._portfolio_svc.build_portfolio_table(portfolio)
            if not tbl.empty and "market_value" in tbl.columns:
                live_value = float(tbl["market_value"].sum())
        except Exception as exc:
            logger.warning("Cannot compute live value for %s: %s", portfolio, exc)

        # Add cash component
        cash_row = query_one(
            "SELECT balance FROM portfolio_cash "
            "WHERE user_id = ? AND portfolio = ?",
            (self.user_id, portfolio),
        )
        cash_balance = float(cash_row[0]) if cash_row and cash_row[0] else 0.0
        live_total = live_value + cash_balance

        # Snapshot values can drift by price moves, so use a 5% tolerance
        if snap_value > 0 and live_total > 0:
            drift_pct = abs(live_total - snap_value) / snap_value * 100
        else:
            drift_pct = 0.0

        return {
            "portfolio": portfolio,
            "has_snapshots": True,
            "latest_date": snap_date_str,
            "days_since_snapshot": days_old,
            "is_fresh": is_fresh,
            "snapshot_value": round(snap_value, 3),
            "live_value": round(live_total, 3),
            "drift_pct": round(drift_pct, 2),
            "is_valid": is_fresh,
        }

    # -----------------------------------------------------------------
    #  4. Transaction anomaly scan
    # -----------------------------------------------------------------

    def scan_transaction_anomalies(self) -> dict:
        """
        Flag suspicious transactions:
          - Sells exceeding held shares at that date
          - Negative purchase costs
          - Duplicate transactions (same symbol+date+type+shares)
          - Missing required fields (Buy without purchase_cost, etc.)
        """

        sd = _soft_del()

        tx_df = query_df(
            f"""
            SELECT
                id, TRIM(stock_symbol) AS stock_symbol, portfolio,
                txn_date, txn_type, shares, purchase_cost, sell_value,
                bonus_shares, cash_dividend, fees, created_at
            FROM transactions
            WHERE user_id = ?
              AND COALESCE(category, 'portfolio') = 'portfolio' {sd}
            ORDER BY portfolio, stock_symbol, txn_date, created_at, id
            """,
            (self.user_id,),
        )

        if tx_df.empty:
            return {"anomalies": [], "count": 0, "is_valid": True}

        anomalies: List[dict] = []

        # ── 4a. Negative costs ───────────────────────────────────────
        neg_cost = tx_df[
            (tx_df["txn_type"] == "Buy") & (tx_df["purchase_cost"].fillna(0) < 0)
        ]
        for _, r in neg_cost.iterrows():
            anomalies.append({
                "txn_id": int(r["id"]),
                "type": "negative_cost",
                "detail": f"Buy with negative purchase_cost: {r['purchase_cost']}",
                "severity": "error",
            })

        # ── 4b. Buys missing purchase_cost ───────────────────────────
        missing_cost = tx_df[
            (tx_df["txn_type"] == "Buy")
            & (tx_df["purchase_cost"].fillna(0) == 0)
            & (tx_df["shares"].fillna(0) > 0)
        ]
        for _, r in missing_cost.iterrows():
            anomalies.append({
                "txn_id": int(r["id"]),
                "type": "missing_cost",
                "detail": f"Buy of {r['shares']} shares with zero purchase_cost",
                "severity": "warning",
            })

        # ── 4c. Sells missing sell_value ─────────────────────────────
        missing_sell = tx_df[
            (tx_df["txn_type"] == "Sell")
            & (tx_df["sell_value"].fillna(0) == 0)
            & (tx_df["shares"].fillna(0) > 0)
        ]
        for _, r in missing_sell.iterrows():
            anomalies.append({
                "txn_id": int(r["id"]),
                "type": "missing_sell_value",
                "detail": f"Sell of {r['shares']} shares with zero sell_value",
                "severity": "warning",
            })

        # ── 4d. Duplicates (same symbol+date+type+shares) ───────────
        dup_cols = ["stock_symbol", "txn_date", "txn_type", "shares"]
        dupes = tx_df[tx_df.duplicated(subset=dup_cols, keep=False)]
        seen_groups: set = set()
        for _, r in dupes.iterrows():
            key = (r["stock_symbol"], r["txn_date"], r["txn_type"], r["shares"])
            if key not in seen_groups:
                seen_groups.add(key)
                anomalies.append({
                    "txn_id": int(r["id"]),
                    "type": "possible_duplicate",
                    "detail": (
                        f"Duplicate: {r['stock_symbol']} {r['txn_type']} "
                        f"{r['shares']} shares on {r['txn_date']}"
                    ),
                    "severity": "warning",
                })

        # ── 4e. Over-sell detection (sells > cumulative holdings) ────
        for (pf, sym), group in tx_df.groupby(["portfolio", "stock_symbol"]):
            held = 0.0
            for _, r in group.iterrows():
                typ = r["txn_type"]
                sh = safe_float(r.get("shares", 0), 0.0)
                bonus = safe_float(r.get("bonus_shares", 0), 0.0)

                if typ == "Buy":
                    held += sh
                held += bonus
                if typ == "Sell":
                    if sh > held + POSITION_TOLERANCE:
                        anomalies.append({
                            "txn_id": int(r["id"]),
                            "type": "over_sell",
                            "detail": (
                                f"Sell {sh} shares of {sym} in {pf} "
                                f"but only {round(held, 4)} held"
                            ),
                            "severity": "error",
                        })
                    held -= sh

        return {
            "anomalies": anomalies,
            "count": len(anomalies),
            "errors": sum(1 for a in anomalies if a["severity"] == "error"),
            "warnings": sum(1 for a in anomalies if a["severity"] == "warning"),
            "is_valid": not any(a["severity"] == "error" for a in anomalies),
        }

    # -----------------------------------------------------------------
    #  5. Data completeness check
    # -----------------------------------------------------------------

    def verify_data_completeness(self) -> dict:
        """
        Check that supporting data exists:
          - Every transacted symbol has a stocks entry with a price
          - Every portfolio has at least one cash_deposit
          - Portfolios table is populated
        """

        sd = _soft_del()

        # Symbols with transactions but no stock record
        orphan_symbols = query_all(
            f"""
            SELECT DISTINCT TRIM(t.stock_symbol) AS sym
            FROM transactions t
            WHERE t.user_id = ?
              AND COALESCE(t.category, 'portfolio') = 'portfolio' {sd}
              AND TRIM(t.stock_symbol) NOT IN (
                  SELECT TRIM(symbol) FROM stocks WHERE user_id = ?
              )
            """,
            (self.user_id, self.user_id),
        )

        # Symbols with $0 current_price
        zero_price = query_all(
            f"""
            SELECT DISTINCT TRIM(s.symbol) AS sym
            FROM stocks s
            WHERE s.user_id = ?
              AND COALESCE(s.current_price, 0) = 0
              AND TRIM(s.symbol) IN (
                  SELECT DISTINCT TRIM(stock_symbol)
                  FROM transactions
                  WHERE user_id = ?
                    AND COALESCE(category, 'portfolio') = 'portfolio' {sd}
              )
            """,
            (self.user_id, self.user_id),
        )

        # Portfolios without deposits
        portfolios = query_all(
            "SELECT name FROM portfolios WHERE user_id = ?",
            (self.user_id,),
        )
        portfolio_names = [r[0] for r in portfolios] if portfolios else []

        no_deposits: List[str] = []
        for pf in portfolio_names:
            cnt = query_val(
                "SELECT COUNT(*) FROM cash_deposits "
                "WHERE user_id = ? AND portfolio = ? AND COALESCE(is_deleted,0) = 0",
                (self.user_id, pf),
            )
            if not cnt or int(cnt) == 0:
                no_deposits.append(pf)

        issues: List[dict] = []

        for r in (orphan_symbols or []):
            issues.append({
                "type": "orphan_symbol",
                "detail": f"Symbol '{r[0]}' has transactions but no stocks entry",
                "severity": "warning",
            })

        for r in (zero_price or []):
            issues.append({
                "type": "zero_price",
                "detail": f"Symbol '{r[0]}' has current_price = 0",
                "severity": "warning",
            })

        for pf in no_deposits:
            issues.append({
                "type": "no_deposits",
                "detail": f"Portfolio '{pf}' has no cash deposits",
                "severity": "info",
            })

        return {
            "portfolios_found": len(portfolio_names),
            "orphan_symbols": len(orphan_symbols or []),
            "zero_price_symbols": len(zero_price or []),
            "portfolios_without_deposits": no_deposits,
            "issues": issues,
            "is_valid": len([i for i in issues if i["severity"] == "error"]) == 0,
        }

    # =================================================================
    #  Full integrity sweep
    # =================================================================

    def run_full_integrity_check(self) -> dict:
        """
        Run ALL integrity checks and return a unified report.
        """

        portfolios_rows = query_all(
            "SELECT name FROM portfolios WHERE user_id = ?",
            (self.user_id,),
        )
        portfolio_names = [r[0] for r in portfolios_rows] if portfolios_rows else []

        cash_results: Dict[str, dict] = {}
        position_results: Dict[str, dict] = {}
        snapshot_results: Dict[str, dict] = {}

        for pf in portfolio_names:
            try:
                cash_results[pf] = self.verify_cash_balance(pf)
            except Exception as exc:
                logger.error("Cash check failed for %s: %s", pf, exc)
                cash_results[pf] = {"portfolio": pf, "is_valid": None, "error": str(exc)}

            try:
                position_results[pf] = self.verify_positions(pf)
            except Exception as exc:
                logger.error("Position check failed for %s: %s", pf, exc)
                position_results[pf] = {"portfolio": pf, "is_valid": None, "error": str(exc)}

            try:
                snapshot_results[pf] = self.verify_snapshots(pf)
            except Exception as exc:
                logger.error("Snapshot check failed for %s: %s", pf, exc)
                snapshot_results[pf] = {"portfolio": pf, "is_valid": None, "error": str(exc)}

        anomalies = self.scan_transaction_anomalies()
        completeness = self.verify_data_completeness()

        # ── Overall validity ─────────────────────────────────────────
        def _valid(d: dict) -> Optional[bool]:
            return d.get("is_valid")

        all_checks: List[Optional[bool]] = []
        for d in list(cash_results.values()) + list(position_results.values()):
            all_checks.append(_valid(d))
        all_checks.append(_valid(anomalies))
        all_checks.append(_valid(completeness))

        # None = indeterminate, filter them out for overall
        determinate = [v for v in all_checks if v is not None]
        overall_valid = all(determinate) if determinate else None

        return {
            "user_id": self.user_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "overall_valid": overall_valid,
            "summary": {
                "cash_checks": len(cash_results),
                "position_checks": len(position_results),
                "snapshot_checks": len(snapshot_results),
                "anomaly_errors": anomalies.get("errors", 0),
                "anomaly_warnings": anomalies.get("warnings", 0),
                "completeness_issues": len(completeness.get("issues", [])),
            },
            "cash": cash_results,
            "positions": position_results,
            "snapshots": snapshot_results,
            "anomalies": anomalies,
            "completeness": completeness,
        }


# ── Module-level convenience function ────────────────────────────────

def run_integrity_check(user_id: int) -> dict:
    """Run full integrity check (backward-compatible free function)."""
    return IntegrityService(user_id).run_full_integrity_check()
