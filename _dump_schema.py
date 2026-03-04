"""Dump all table schemas from dev_portfolio.db."""
import sqlite3

DB = r"C:\Users\Sager\OneDrive\Desktop\portfolio_app\mobile-migration\dev_portfolio.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

tables = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
for t in tables:
    cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
    print(f"\n=== {t} ({len(cols)} cols) ===")
    for c in cols:
        print(f"  {c[1]:30s} {c[2]:15s} {'PK' if c[5] else ''}")
conn.close()
