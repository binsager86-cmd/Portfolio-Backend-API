"""Check and restore missing TRADING stocks."""
from app.core.database import query_df, query_val, exec_sql

# Check what TRADING stocks exist
df = query_df("SELECT id, symbol, portfolio FROM stocks WHERE user_id=1 AND portfolio='TRADING'", ())
print("Current TRADING stocks:")
print(df.to_string() if not df.empty else "  (none)")

# Expected TRADING stocks
expected = {
    47: ("KRE.KW", 0.379),
    48: ("NIH.KW", 0.130),
    49: ("SANAM.KW", 0.222),
}

for stock_id, (symbol, price) in expected.items():
    exists = query_val("SELECT id FROM stocks WHERE id=? AND user_id=1", (stock_id,))
    if not exists:
        exec_sql(
            """INSERT INTO stocks (id, user_id, symbol, name, portfolio, currency, current_price, created_at)
               VALUES (?, 1, ?, ?, 'TRADING', 'KWD', ?, 0)""",
            (stock_id, symbol, symbol, price),
        )
        print(f"  Restored {symbol} (id={stock_id})")
    else:
        print(f"  {symbol} (id={stock_id}) already exists")

# Final state
df2 = query_df("SELECT id, symbol, portfolio FROM stocks WHERE user_id=1 ORDER BY symbol", ())
print("\nAll stocks:")
print(df2.to_string())
