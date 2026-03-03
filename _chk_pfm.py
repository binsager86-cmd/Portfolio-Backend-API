import sqlite3
conn = sqlite3.connect('dev_portfolio.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pfm%'")
tables = [r[0] for r in cur.fetchall()]
print("PFM tables:", tables)
for t in tables:
    cur.execute(f"PRAGMA table_info({t})")
    cols = [(r[1], r[2]) for r in cur.fetchall()]
    print(f"  {t}: {cols}")
conn.close()
