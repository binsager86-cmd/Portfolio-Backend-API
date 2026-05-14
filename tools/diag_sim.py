"""Quick diagnostic for simulator entry price vs actual OHLCV prices."""
from app.core.database import query_all

print("=== SENERGY / TIJARA ratings ===")
rows = query_all(
    "SELECT ticker, stage, confidence, entry_primary, last_price, stop_loss, tp1 "
    "FROM ee_ratings_cache WHERE ticker IN ('SENERGY','TIJARA') ORDER BY ticker"
)
for r in rows:
    print(dict(r))

print("\n=== SENERGY OHLCV (first 5 dates) ===")
rows = query_all(
    "SELECT bar_date, open, high, low, close FROM ee_ohlcv_cache "
    "WHERE ticker='SENERGY' ORDER BY bar_date LIMIT 5"
)
for r in rows:
    print(dict(r))

print("\n=== Positions summary (worst losses) ===")
rows = query_all(
    "SELECT ticker, entry_date, entry_price, exit_price, pnl_pct, exit_reason "
    "FROM simulator_positions WHERE pnl_pct IS NOT NULL ORDER BY pnl_pct LIMIT 10"
)
for r in rows:
    print(dict(r))
