"""
Fundamental Analysis — shared helpers, constants, and utility functions.

Extracted from the monolithic fundamental.py to enable modular imports.
"""

import re
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.exceptions import NotFoundError
from app.core.database import query_all, query_one, query_val, exec_sql

logger = logging.getLogger(__name__)

# ── PDF file storage ─────────────────────────────────────────────────

_PDF_UPLOAD_DIR = Path(__file__).resolve().parents[3] / "uploads" / "pdfs"


def _get_pdf_dir(stock_id: int) -> Path:
    """Return (and create) the per-stock PDF directory."""
    d = _PDF_UPLOAD_DIR / str(stock_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Helper: convert camelCase SA keys to readable display names ──────

def _camel_to_display(key: str) -> str:
    """Convert camelCase key to 'Title Case Display Name'.

    e.g. 'stockBasedCompensation' -> 'Stock Based Compensation'
         'netIncomeCF' -> 'Net Income CF'
         'totalOpex' -> 'Total Opex'
    """
    import re as _re
    # Insert space before uppercase letters that follow lowercase letters or
    # before a run of uppercase letters followed by a lowercase letter
    spaced = _re.sub(r'([a-z])([A-Z])', r'\1 \2', key)
    spaced = _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    return spaced[:1].upper() + spaced[1:]


# ── Canonical line-item code map ─────────────────────────────────────

# Standard line item codes for normalization
# Canonical code map — all values are lowercase for consistency across years.
# Uses the same canonical forms as _MERGE_CODES in extraction_service.py.
_CANONICAL_CODES: Dict[str, str] = {
    # Income statement
    "revenue": "revenue", "total_revenue": "revenue", "net_revenue": "revenue",
    "sales": "revenue", "net_sales": "revenue",
    "cost_of_revenue": "cost_of_revenue", "cost_of_sales": "cost_of_revenue",
    "cost_of_goods_sold": "cost_of_revenue", "cogs": "cost_of_revenue",
    "gross_profit": "gross_profit",
    "selling_general_administrative": "sga", "sga": "sga",
    "selling_general_and_administrative": "sga",
    "research_and_development": "r_and_d", "r_and_d": "r_and_d", "r&d": "r_and_d",
    "operating_expenses": "operating_expenses", "total_operating_expenses": "operating_expenses",
    "operating_income": "operating_income", "operating_profit": "operating_income",
    "income_from_operations": "operating_income",
    "interest_expense": "interest_expense", "finance_costs": "interest_expense",
    "finance_cost": "interest_expense",
    "other_income": "other_income", "other_income_expense": "other_income",
    "income_before_tax": "income_before_tax", "profit_before_tax": "income_before_tax",
    "income_tax": "income_tax", "income_tax_expense": "income_tax",
    "tax_expense": "income_tax", "taxation": "income_tax",
    "net_income": "net_income", "net_profit": "net_income",
    "profit_for_the_year": "net_income", "profit_for_the_period": "net_income",
    "profit_for_year": "net_income",
    "net_income_attributable_to_shareholders": "net_income",
    "eps_basic": "eps_basic", "basic_eps": "eps_basic",
    "basic_earnings_per_share": "eps_basic", "earnings_per_share_basic": "eps_basic",
    "basic_and_diluted_earnings_per_share": "eps_basic",
    "basic_and_diluted_earnings_per_sha": "eps_basic",
    "earnings_per_share": "eps_basic",
    "eps_diluted": "eps_diluted", "diluted_eps": "eps_diluted",
    "diluted_earnings_per_share": "eps_diluted", "earnings_per_share_diluted": "eps_diluted",
    "ebitda": "ebitda",
    "depreciation_and_amortization": "depreciation_amortization",
    "depreciation_amortization": "depreciation_amortization",
    # Balance sheet
    "cash": "cash", "cash_and_cash_equivalents": "cash", "cash_equivalents": "cash",
    "cash_and_bank_balances": "cash", "cash_and_balances_with_banks": "cash",
    "accounts_receivable": "accounts_receivable", "trade_receivables": "accounts_receivable",
    "receivables": "accounts_receivable", "trade_and_other_receivables": "accounts_receivable",
    "inventory": "inventory", "inventories": "inventory",
    "other_current_assets": "other_current_assets",
    "total_current_assets": "total_current_assets",
    "property_plant_equipment": "ppe_net", "property_plant_and_equipment": "ppe_net",
    "ppe_net": "ppe_net", "fixed_assets": "ppe_net",
    "property_and_equipment": "ppe_net",
    "goodwill": "goodwill",
    "intangible_assets": "intangible_assets", "intangibles": "intangible_assets",
    "total_non_current_assets": "total_non_current_assets",
    "total_assets": "total_assets",
    "accounts_payable": "accounts_payable", "trade_payables": "accounts_payable",
    "trade_and_other_payables": "accounts_payable",
    "short_term_debt": "short_term_debt", "current_portion_of_debt": "short_term_debt",
    "current_portion_of_long_term_debt": "short_term_debt",
    "short_term_borrowings": "short_term_debt",
    "total_current_liabilities": "total_current_liabilities",
    "long_term_debt": "long_term_debt", "long_term_borrowings": "long_term_debt",
    "non_current_borrowings": "long_term_debt",
    "total_non_current_liabilities": "total_non_current_liabilities",
    "total_liabilities": "total_liabilities",
    "common_stock": "share_capital", "share_capital": "share_capital",
    "issued_capital": "share_capital",
    "retained_earnings": "retained_earnings",
    "share_premium": "share_premium",
    "statutory_reserve": "statutory_reserve",
    "voluntary_reserve": "voluntary_reserve",
    "general_reserve": "general_reserve",
    "treasury_shares": "treasury_shares", "treasury_shares_equity": "treasury_shares",
    "total_equity": "total_equity", "total_shareholders_equity": "total_equity",
    "total_stockholders_equity": "total_equity",
    "equity_attributable_to_shareholders": "total_equity",
    "total_liabilities_and_equity": "total_liabilities_and_equity",
    "total_liabilities_and_shareholders_equity": "total_liabilities_and_equity",
    "total_liabilities_equity": "total_liabilities_and_equity",
    # Cash flow
    "net_income_cf": "net_income_cf",
    "cash_from_operations": "cash_from_operations",
    "cash_from_operating_activities": "cash_from_operations",
    "net_cash_from_operating_activities": "cash_from_operations",
    "net_cash_used_in_operating_activities": "cash_from_operations",
    "cash_used_in_operating_activities": "cash_from_operations",
    "capital_expenditures": "capital_expenditures", "capex": "capital_expenditures",
    "purchase_of_property_plant_equipment": "capital_expenditures",
    "purchase_of_property_plant_and_equipment": "capital_expenditures",
    "other_investing_activities": "other_investing",
    "other_investing": "other_investing",
    "cash_from_investing": "cash_from_investing",
    "cash_from_investing_activities": "cash_from_investing",
    "net_cash_from_investing_activities": "cash_from_investing",
    "net_cash_used_in_investing_activities": "cash_from_investing",
    "cash_used_in_investing_activities": "cash_from_investing",
    "debt_issued": "debt_issued", "proceeds_from_borrowings": "debt_issued",
    "debt_repaid": "debt_repaid", "repayment_of_borrowings": "debt_repaid",
    "dividends_paid": "dividends_paid", "dividend_paid": "dividends_paid",
    "cash_from_financing": "cash_from_financing",
    "cash_from_financing_activities": "cash_from_financing",
    "net_cash_from_financing_activities": "cash_from_financing",
    "net_cash_used_in_financing_activities": "cash_from_financing",
    "cash_used_in_financing_activities": "cash_from_financing",
    "net_change_in_cash": "net_change_in_cash",
    "net_change_cash": "net_change_in_cash",
    "net_increase_decrease_in_cash": "net_change_in_cash",
    "changes_in_working_capital": "changes_in_working_capital",
    "changes_working_capital": "changes_in_working_capital",
    # Free Cash Flow variants
    "free_cash_flow": "free_cash_flow",
    "fcf": "free_cash_flow",
    "unlevered_free_cash_flow": "free_cash_flow",
    "levered_free_cash_flow": "levered_free_cash_flow",
    "free_cash_flow_margin": "free_cash_flow_margin",
    "fcf_margin": "free_cash_flow_margin",
    "free_cash_flow_per_share": "free_cash_flow_per_share",
    "fcf_per_share": "free_cash_flow_per_share",
    "free_cash_flow_growth": "free_cash_flow_growth",
    "operating_cash_flow": "cash_from_operations",
    # Equity statement
    "shares_outstanding": "shares_diluted", "share_count": "share_count",
    "shares_basic": "shares_basic", "shares_diluted": "shares_diluted",
    # GCC / Kuwait specific
    "cost_of_operations": "cost_of_revenue",
    "general_and_administrative_expenses": "general_and_admin",
    "general_and_administrative_expens": "general_and_admin",
    "selling_expenses": "selling_expenses",
    "selling_and_distribution_expenses": "selling_expenses",
    "finance_charges": "finance_charges", "finance_charge": "finance_charges",
    "profit_before_contribution_to_kfas": "profit_before_deductions",
    "profit_before_contribution_to_kuwait_foundation_for_advancement_of_sciences": "profit_before_deductions",
    "profit_before_contribution_to_kuwait_foundation_for_the_advancement_of_sciences": "profit_before_deductions",
    "contribution_to_kfas": "contribution_to_kfas",
    "contribution_kfas": "contribution_to_kfas",
    "contribution_to_kuwait_foundation_for_advancement_of_sciences": "contribution_to_kfas",
    "contribution_to_kuwait_foundation_for_the_advancement_of_sciences": "contribution_to_kfas",
    "national_labour_support_tax": "nlst", "nlst": "nlst",
    "national_labor_support_tax": "nlst",
    "zakat": "zakat",
    "directors_remuneration": "directors_remuneration",
    "directors_fees": "directors_remuneration",
    "board_of_directors_remuneration": "directors_remuneration",
    "share_of_profit_of_associates": "share_results_associates",
    "share_of_loss_of_associates": "share_results_associates",
    "share_of_results_of_associates": "share_results_associates",
    "share_of_profit_loss_of_associates": "share_results_associates",
}


def _normalize_key(raw_key: str) -> str:
    """Map an AI-extracted key to a canonical lowercase line item code."""
    import re as _re
    k = raw_key.strip().lower().replace(" ", "_").replace("-", "_")
    k = _re.sub(r"_+", "_", k).strip("_")
    # Direct lookup in canonical map
    if k in _CANONICAL_CODES:
        return _CANONICAL_CODES[k]
    # Try without trailing _total, _net etc.
    for suffix in ("_total", "_net", "_and_equivalents"):
        if k.endswith(suffix):
            trimmed = k[: -len(suffix)].rstrip("_")
            if trimmed in _CANONICAL_CODES:
                return _CANONICAL_CODES[trimmed]
    # Fallback: clean lowercase snake_case (never uppercase)
    return k


# ════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ════════════════════════════════════════════════════════════════════

def _verify_stock_owner(stock_id: int, user_id: int) -> None:
    row = query_val(
        "SELECT id FROM analysis_stocks WHERE id = ? AND user_id = ?",
        (stock_id, user_id),
    )
    if not row:
        raise NotFoundError("Analysis Stock", str(stock_id))


# ── Metrics calculation helpers ──────────────────────────────────────

def _load_items_for_period(stock_id: int, period_end_date: str) -> Dict[str, float]:
    """Flatten all line items across all statement types for one period.

    Keys are uppercased so callers can use ``_get("REVENUE")`` regardless
    of whether the DB stores ``revenue`` or ``REVENUE``.
    """
    rows = query_all(
        """SELECT li.line_item_code, li.amount
           FROM financial_line_items li
           JOIN financial_statements fs ON fs.id = li.statement_id
           WHERE fs.stock_id = ? AND fs.period_end_date = ?""",
        (stock_id, period_end_date),
    )
    items: Dict[str, float] = {}
    for r in rows:
        code = (r[0] if isinstance(r, (tuple, list)) else r["line_item_code"]).upper()
        amount = r[1] if isinstance(r, (tuple, list)) else r["amount"]
        # Keep first occurrence (don't overwrite with duplicates)
        if code not in items:
            items[code] = amount
    return items


def _upsert_metric(
    stock_id: int, fiscal_year: int, period_end_date: str,
    metric_type: str, metric_name: str, metric_value: float,
    fiscal_quarter: Optional[int] = None,
) -> None:
    now = int(time.time())
    existing = query_val(
        "SELECT id FROM stock_metrics WHERE stock_id = ? AND metric_name = ? AND period_end_date = ?",
        (stock_id, metric_name, period_end_date),
    )
    if existing:
        exec_sql(
            """UPDATE stock_metrics
               SET fiscal_year=?, fiscal_quarter=?, metric_type=?, metric_value=?, created_at=?
               WHERE id=?""",
            (fiscal_year, fiscal_quarter, metric_type, metric_value, now, existing),
        )
    else:
        exec_sql(
            """INSERT INTO stock_metrics
               (stock_id, fiscal_year, fiscal_quarter, period_end_date,
                metric_type, metric_name, metric_value, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (stock_id, fiscal_year, fiscal_quarter, period_end_date,
             metric_type, metric_name, metric_value, now),
        )


def _resolve_yf_ticker(symbol: str, user_id: int = None) -> str:
    """Resolve a raw symbol to its yfinance-compatible ticker.

    Priority: 1) stocks table yf_ticker, 2) KUWAIT_STOCKS list, 3) raw symbol.
    """
    # Try the stocks table (populated when user added the stock)
    if user_id:
        row = query_one(
            "SELECT yf_ticker FROM stocks WHERE symbol = ? AND user_id = ? AND yf_ticker IS NOT NULL",
            (symbol, user_id),
        )
    else:
        row = query_one(
            "SELECT yf_ticker FROM stocks WHERE symbol = ? AND yf_ticker IS NOT NULL LIMIT 1",
            (symbol,),
        )
    if row:
        yft = row[0] if isinstance(row, (tuple, list)) else row.get("yf_ticker")
        if yft:
            return yft

    # Check the hardcoded stock lists (covers peers not in the user's portfolio)
    from app.data.stock_lists import KUWAIT_STOCKS
    upper = symbol.upper()
    for s in KUWAIT_STOCKS:
        if s["symbol"].upper() == upper:
            return s["yf_ticker"]

    return symbol
