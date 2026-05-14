"""Full simulator report."""
from app.core.database import query_all, query_one

print("=" * 70)
print("EAGLE EYE PAPER TRADING SIMULATOR — BACKFILL REPORT")
print("=" * 70)

# Portfolio summary
print("\n--- PORTFOLIO SUMMARY ---")
rows = query_all("""
    SELECT p.strategy_name,
           p.starting_capital_kwd, p.total_value_kwd, p.cash_balance_kwd,
           ROUND((p.total_value_kwd - p.starting_capital_kwd) / p.starting_capital_kwd * 100, 2) AS ret_pct,
           (SELECT COUNT(*) FROM simulator_positions sp WHERE sp.portfolio_id = p.id) AS total_trades,
           (SELECT COUNT(*) FROM simulator_positions sp WHERE sp.portfolio_id = p.id AND sp.status = 'OPEN') AS open_pos,
           (SELECT COUNT(*) FROM simulator_positions sp WHERE sp.portfolio_id = p.id AND sp.pnl_pct > 0) AS wins,
           (SELECT COUNT(*) FROM simulator_positions sp WHERE sp.portfolio_id = p.id AND sp.pnl_pct <= 0 AND sp.status='CLOSED') AS losses,
           (SELECT ROUND(AVG(sp.pnl_pct),2) FROM simulator_positions sp WHERE sp.portfolio_id = p.id AND sp.pnl_pct > 0 AND sp.status='CLOSED') AS avg_win,
           (SELECT ROUND(AVG(sp.pnl_pct),2) FROM simulator_positions sp WHERE sp.portfolio_id = p.id AND sp.pnl_pct <= 0 AND sp.status='CLOSED') AS avg_loss,
           (SELECT ROUND(MIN(total_value_kwd),2) FROM simulator_daily_snapshots sn WHERE sn.portfolio_id = p.id) AS min_value,
           (SELECT ROUND(MAX(total_value_kwd),2) FROM simulator_daily_snapshots sn WHERE sn.portfolio_id = p.id) AS max_value
    FROM simulator_portfolios p ORDER BY p.id
""")

for r in rows:
    r = dict(r)
    wins = r['wins'] or 0
    losses = r['losses'] or 0
    total_closed = wins + losses
    wr = round(wins / total_closed * 100, 1) if total_closed > 0 else 0
    print(f"\n  {r['strategy_name']}")
    print(f"    Capital : {r['starting_capital_kwd']:>10,.2f} KWD  →  {r['total_value_kwd']:>10,.2f} KWD  ({r['ret_pct']:+.2f}%)")
    print(f"    Cash    : {r['cash_balance_kwd']:>10,.2f} KWD")
    print(f"    Trades  : {r['total_trades']} total  |  {r['open_pos']} open  |  {total_closed} closed")
    print(f"    W/L     : {wins}W / {losses}L  =  {wr}% win rate")
    print(f"    Avg Win : {r['avg_win'] or 0:+.2f}%    Avg Loss: {r['avg_loss'] or 0:+.2f}%")
    print(f"    Range   : {r['min_value']} – {r['max_value']} KWD")

# Top winners across all strategies
print("\n\n--- TOP 10 WINNERS (closed positions) ---")
rows = query_all("""
    SELECT sp.ticker, sp.entry_date, sp.exit_date, sp.entry_price, sp.exit_price,
           sp.pnl_pct, sp.exit_reason, p.strategy_name
    FROM simulator_positions sp JOIN simulator_portfolios p ON p.id = sp.portfolio_id
    WHERE sp.status = 'CLOSED' AND sp.pnl_pct > 0
    ORDER BY sp.pnl_pct DESC LIMIT 10
""")
for r in rows:
    r = dict(r)
    print(f"  {r['strategy_name']:12s} {r['ticker']:10s}  {r['entry_date']} → {r['exit_date']}  "
          f"entry={r['entry_price']:.2f}  exit={r['exit_price']:.2f}  {r['pnl_pct']:+.2f}%  [{r['exit_reason']}]")

# Top losers
print("\n\n--- TOP 10 LOSERS (closed positions) ---")
rows = query_all("""
    SELECT sp.ticker, sp.entry_date, sp.exit_date, sp.entry_price, sp.exit_price,
           sp.pnl_pct, sp.exit_reason, p.strategy_name
    FROM simulator_positions sp JOIN simulator_portfolios p ON p.id = sp.portfolio_id
    WHERE sp.status = 'CLOSED' AND sp.pnl_pct IS NOT NULL
    ORDER BY sp.pnl_pct ASC LIMIT 10
""")
for r in rows:
    r = dict(r)
    print(f"  {r['strategy_name']:12s} {r['ticker']:10s}  {r['entry_date']} → {r['exit_date']}  "
          f"entry={r['entry_price']:.2f}  exit={r['exit_price']:.2f}  {r['pnl_pct']:+.2f}%  [{r['exit_reason']}]")

# Exit reason breakdown
print("\n\n--- EXIT REASON BREAKDOWN ---")
rows = query_all("""
    SELECT exit_reason, COUNT(*) as cnt,
           ROUND(AVG(pnl_pct),2) as avg_pnl,
           ROUND(MIN(pnl_pct),2) as min_pnl,
           ROUND(MAX(pnl_pct),2) as max_pnl
    FROM simulator_positions WHERE status='CLOSED' AND pnl_pct IS NOT NULL
    GROUP BY exit_reason ORDER BY cnt DESC
""")
for r in rows:
    r = dict(r)
    print(f"  {r['exit_reason']:20s}  {r['cnt']:3d} trades  avg={r['avg_pnl']:+.2f}%  "
          f"min={r['min_pnl']:+.2f}%  max={r['max_pnl']:+.2f}%")

# Most traded tickers
print("\n\n--- MOST TRADED TICKERS (all strategies combined) ---")
rows = query_all("""
    SELECT ticker, COUNT(*) as trades,
           ROUND(AVG(pnl_pct),2) as avg_pnl,
           SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins
    FROM simulator_positions WHERE status='CLOSED' AND pnl_pct IS NOT NULL
    GROUP BY ticker ORDER BY trades DESC LIMIT 15
""")
for r in rows:
    r = dict(r)
    wr = round(r['wins'] / r['trades'] * 100) if r['trades'] > 0 else 0
    print(f"  {r['ticker']:12s}  {r['trades']:3d} trades  avg={r['avg_pnl']:+.2f}%  wr={wr}%")

# Snapshot count
row = query_one("SELECT COUNT(*) as cnt FROM simulator_daily_snapshots")
pos_row = query_one("SELECT COUNT(*) as cnt FROM simulator_positions")
cons_row = query_one("SELECT COUNT(*) as cnt FROM simulator_considered_trades")
print(f"\n\n--- DATABASE SUMMARY ---")
print(f"  Daily snapshots   : {dict(row)['cnt']}")
print(f"  Total positions   : {dict(pos_row)['cnt']}")
print(f"  Considered trades : {dict(cons_row)['cnt']}")
print("=" * 70)
