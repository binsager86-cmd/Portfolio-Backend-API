from app.core.database import query_df, query_val

# Simulate what the endpoint does for user_id=2
uid = 2
df = query_df("SELECT * FROM analysis_stocks WHERE user_id = ? ORDER BY symbol", (uid,))
print(f"Stocks for user {uid}: {len(df)} rows")
if not df.empty:
    for _, row in df.iterrows():
        sid = int(row["id"])
        sym = row["symbol"]
        stmt_count = query_val(
            "SELECT COUNT(*) FROM financial_statements WHERE stock_id = ?", (sid,)
        )
        li_count = query_val(
            "SELECT COUNT(*) FROM financial_line_items WHERE statement_id IN "
            "(SELECT id FROM financial_statements WHERE stock_id = ?)", (sid,)
        )
        print(f"  stock_id={sid} {sym} -> {stmt_count} stmts, {li_count} line items")

# Also check user_id=1
uid = 1
df1 = query_df("SELECT * FROM analysis_stocks WHERE user_id = ? ORDER BY symbol", (uid,))
print(f"\nStocks for user {uid}: {len(df1)} rows")
if not df1.empty:
    for _, row in df1.iterrows():
        sid = int(row["id"])
        sym = row["symbol"]
        stmt_count = query_val(
            "SELECT COUNT(*) FROM financial_statements WHERE stock_id = ?", (sid,)
        )
        print(f"  stock_id={sid} {sym} -> {stmt_count} stmts")
