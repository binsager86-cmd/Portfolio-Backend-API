from app.core.database import query_all

rows = query_all("SELECT id, user_id, symbol, company_name FROM analysis_stocks")
print("=== ANALYSIS STOCKS OWNERSHIP ===")
for r in rows:
    print(f"  stock_id={r['id']} user_id={r['user_id']} {r['symbol']} - {r['company_name']}")

print()
users = query_all("SELECT id, email FROM users")
print("=== USERS ===")
for u in users:
    print(f"  user_id={u['id']} email={u['email']}")
