"""Quick script to check database state."""
from app.core.database import query_all, query_val

# List all tables
tables = query_all("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
print("=== TABLES ===")
for t in tables:
    print(f"  {t['name']}")

print("\n=== ROW COUNTS ===")
key_tables = [
    "users", "stocks", "transactions", "cash_deposits",
    "analysis_stocks", "financial_statements", "financial_line_items",
    "extraction_cache", "portfolios", "assets", "prices",
    "daily_snapshots", "ledger_entries",
]
for tbl in key_tables:
    try:
        c = query_val(f"SELECT COUNT(*) FROM [{tbl}]")
        print(f"  {tbl}: {c} rows")
    except Exception as e:
        print(f"  {tbl}: (missing)")

# Check analysis_stocks detail
print("\n=== ANALYSIS STOCKS ===")
try:
    rows = query_all("SELECT id, user_id, symbol, company_name FROM analysis_stocks")
    for r in rows:
        print(f"  id={r['id']} user={r['user_id']} {r['symbol']} - {r['company_name']}")
except Exception:
    print("  (table missing)")

# Check financial_statements detail
print("\n=== FINANCIAL STATEMENTS ===")
try:
    rows = query_all("SELECT id, stock_id, statement_type, period_end_date, source_file FROM financial_statements ORDER BY id")
    for r in rows:
        print(f"  id={r['id']} stock={r['stock_id']} {r['statement_type']} {r['period_end_date']} src={r['source_file']}")
except Exception:
    print("  (table missing)")

# Check financial_line_items count per statement
print("\n=== LINE ITEMS PER STATEMENT ===")
try:
    rows = query_all("""
        SELECT fs.id, fs.statement_type, fs.period_end_date, COUNT(li.id) as item_count
        FROM financial_statements fs
        LEFT JOIN financial_line_items li ON li.statement_id = fs.id
        GROUP BY fs.id
        ORDER BY fs.id
    """)
    for r in rows:
        print(f"  stmt {r['id']}: {r['statement_type']} {r['period_end_date']} -> {r['item_count']} items")
except Exception:
    print("  (tables missing)")

# Check DB file path
from app.core.config import get_settings
s = get_settings()
print(f"\n=== DB PATH: {s.DATABASE_URL} ===")
