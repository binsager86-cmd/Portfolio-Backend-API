"""Reset simulator data to baseline for re-run."""
from app.core.database import exec_sql, query_all

exec_sql("DELETE FROM simulator_positions")
exec_sql("DELETE FROM simulator_daily_snapshots")
exec_sql("DELETE FROM simulator_considered_trades")
exec_sql("UPDATE simulator_portfolios SET cash_balance_kwd = 10000, total_value_kwd = 10000, updated_at = datetime('now')")
print("Reset complete.")
for r in query_all("SELECT strategy_name, cash_balance_kwd, total_value_kwd FROM simulator_portfolios ORDER BY id"):
    print(dict(r))
