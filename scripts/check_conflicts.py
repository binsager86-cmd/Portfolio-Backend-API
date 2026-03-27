"""Check for duplicate conflicts before migration."""
import sqlite3, re
from collections import defaultdict

DB = r"C:\Users\Sager\OneDrive\Desktop\portfolio_app\mobile-migration\dev_portfolio.db"
conn = sqlite3.connect(DB)

rows = conn.execute("""
    SELECT li.statement_id, fs.stock_id, fs.statement_type, fs.fiscal_year,
           li.line_item_code, li.amount
    FROM financial_line_items li
    JOIN financial_statements fs ON li.statement_id = fs.id
    ORDER BY li.statement_id, li.order_index
""").fetchall()

stmts = defaultdict(list)
for r in rows:
    stmts[r[0]].append({"code": r[4], "amount": r[5], "stock": r[1], "type": r[2], "year": r[3]})

mapping = {
    "common_stock": "share_capital", "cash_equivalents": "cash",
    "cash_and_cash_equivalents": "cash",
    "treasury_shares_equity": "treasury_shares",
    "total_liabilities_equity": "total_liabilities_and_equity",
    "net_change_cash": "net_change_in_cash",
    "changes_working_capital": "changes_in_working_capital",
    "contribution_kfas": "contribution_to_kfas",
    "board_of_directors_remuneration": "directors_remuneration",
}

conflicts = []
for stmt_id, items in stmts.items():
    seen = {}
    for item in items:
        old = item["code"]
        k = old.strip().lower().replace(" ", "_").replace("-", "_")
        k = re.sub(r"_+", "_", k).strip("_")
        new_code = mapping.get(k, k)
        if new_code in seen:
            conflicts.append(
                f"  stmt={stmt_id} stock={item['stock']} {item['type']} {item['year']}: "
                f"{seen[new_code]} + {old} -> {new_code}"
            )
        seen[new_code] = old

if conflicts:
    print(f"Found {len(conflicts)} conflicts:")
    for c in conflicts:
        print(c)
else:
    print("No conflicts found - safe to migrate.")
