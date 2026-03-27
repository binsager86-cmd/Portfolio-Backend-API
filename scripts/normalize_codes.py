"""One-time migration: normalize all line_item_codes to lowercase canonical form.

Run from backend-api/:
  ..\\mobile-app\\.venv\\Scripts\\python.exe scripts/normalize_codes.py
"""
import re
import sqlite3
import sys
from pathlib import Path

DB_PATH = str((Path(__file__).resolve().parent.parent.parent / "dev_portfolio.db"))

# Same canonical map as in fundamental.py — all values lowercase
_CANONICAL_CODES = {
    # Income statement
    "revenue": "revenue", "total_revenue": "revenue", "net_revenue": "revenue",
    "sales": "revenue", "net_sales": "revenue",
    "cost_of_revenue": "cost_of_revenue", "cost_of_sales": "cost_of_revenue",
    "cost_of_goods_sold": "cost_of_revenue", "cogs": "cost_of_revenue",
    "cost_of_operations": "cost_of_revenue",
    "gross_profit": "gross_profit",
    "selling_general_administrative": "sga", "sga": "sga",
    "selling_general_and_administrative": "sga",
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
    "total_current_assets": "total_current_assets",
    "property_plant_equipment": "ppe_net", "property_plant_and_equipment": "ppe_net",
    "ppe_net": "ppe_net", "fixed_assets": "ppe_net", "property_and_equipment": "ppe_net",
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
    "cash_from_operations": "cash_from_operations",
    "cash_from_operating_activities": "cash_from_operations",
    "net_cash_from_operating_activities": "cash_from_operations",
    "net_cash_used_in_operating_activities": "cash_from_operations",
    "cash_used_in_operating_activities": "cash_from_operations",
    "capital_expenditures": "capital_expenditures", "capex": "capital_expenditures",
    "purchase_of_property_plant_equipment": "capital_expenditures",
    "purchase_of_property_plant_and_equipment": "capital_expenditures",
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
    # GCC specific
    "general_and_administrative_expenses": "general_and_admin",
    "general_and_administrative_expens": "general_and_admin",
    "selling_expenses": "selling_expenses",
    "selling_and_distribution_expenses": "selling_expenses",
    "finance_charges": "finance_charges", "finance_charge": "finance_charges",
    "profit_before_contribution_to_kfas": "profit_before_deductions",
    "contribution_to_kfas": "contribution_to_kfas",
    "contribution_kfas": "contribution_to_kfas",
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


def normalize_key(raw_key: str) -> str:
    k = raw_key.strip().lower().replace(" ", "_").replace("-", "_")
    k = re.sub(r"_+", "_", k).strip("_")
    if k in _CANONICAL_CODES:
        return _CANONICAL_CODES[k]
    for suffix in ("_total", "_net", "_and_equivalents"):
        if k.endswith(suffix):
            trimmed = k[: -len(suffix)].rstrip("_")
            if trimmed in _CANONICAL_CODES:
                return _CANONICAL_CODES[trimmed]
    return k


def main():
    dry_run = "--dry-run" in sys.argv

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Get all distinct codes
    rows = conn.execute("SELECT DISTINCT line_item_code FROM financial_line_items").fetchall()
    print(f"Total distinct codes: {len(rows)}")

    # Build update mapping
    updates = {}
    for r in rows:
        old_code = r[0]
        new_code = normalize_key(old_code)
        if old_code != new_code:
            updates[old_code] = new_code

    print(f"Codes to normalize: {len(updates)}")

    if not updates:
        print("Nothing to do.")
        return

    # Show all changes
    for old, new in sorted(updates.items()):
        print(f"  {old:55s} -> {new}")

    if dry_run:
        print("\n[DRY RUN] No changes applied.")
        return

    # Apply updates
    cur = conn.cursor()
    total_rows = 0
    for old_code, new_code in updates.items():
        cur.execute(
            "UPDATE financial_line_items SET line_item_code = ? WHERE line_item_code = ?",
            (new_code, old_code),
        )
        total_rows += cur.rowcount

    # Also normalize cashflow_staged_rows.normalized_code
    try:
        staged = conn.execute(
            "SELECT DISTINCT normalized_code FROM cashflow_staged_rows WHERE normalized_code IS NOT NULL"
        ).fetchall()
        staged_updates = 0
        for r in staged:
            old = r[0]
            new = normalize_key(old)
            if old != new:
                cur.execute(
                    "UPDATE cashflow_staged_rows SET normalized_code = ? WHERE normalized_code = ?",
                    (new, old),
                )
                staged_updates += cur.rowcount
        print(f"  Staged rows normalized: {staged_updates}")
    except Exception:
        pass

    conn.commit()

    # Deduplicate: if two rows in the same statement have the same code, keep the first
    dupes = conn.execute("""
        SELECT li.statement_id, li.line_item_code, COUNT(*) as cnt
        FROM financial_line_items li
        GROUP BY li.statement_id, li.line_item_code
        HAVING cnt > 1
    """).fetchall()
    if dupes:
        print(f"\nDeduplicating {len(dupes)} duplicate code groups...")
        for d in dupes:
            stmt_id, code, cnt = d
            # Keep the row with lowest order_index, delete the rest
            conn.execute("""
                DELETE FROM financial_line_items
                WHERE statement_id = ? AND line_item_code = ?
                  AND rowid NOT IN (
                    SELECT MIN(rowid) FROM financial_line_items
                    WHERE statement_id = ? AND line_item_code = ?
                  )
            """, (stmt_id, code, stmt_id, code))
            print(f"  stmt={stmt_id} code={code}: removed {cnt - 1} duplicate(s)")
        conn.commit()

    conn.close()
    print(f"\nDone: {total_rows} rows updated across {len(updates)} code changes.")


if __name__ == "__main__":
    main()
