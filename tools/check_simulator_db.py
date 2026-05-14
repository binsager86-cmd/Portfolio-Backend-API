import sys
sys.path.insert(0, ".")
from app.core.database import query_all

tables = query_all("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'simulator%'")
print("Simulator tables:", [t["name"] for t in tables])

for tbl in ["simulator_portfolios", "simulator_positions"]:
    cols = query_all(f"PRAGMA table_info({tbl})")
    print(f"\n{tbl} columns:")
    for c in cols:
        print(f"  {c['cid']} {c['name']} {c['type']}")

rows = query_all("SELECT * FROM simulator_portfolios")
print("\nsimulator_portfolios rows:")
for r in rows:
    print(" ", dict(r))
