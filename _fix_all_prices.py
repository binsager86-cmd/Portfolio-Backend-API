"""Fix prices for all users — re-run price updater with corrected normalization."""
from app.services.price_service import update_all_prices
from app.core.database import query_df

# Find all user IDs that have stocks
users = query_df("SELECT DISTINCT user_id FROM stocks WHERE current_price > 0", ())
print(f"Users with stocks: {users['user_id'].tolist()}")

for uid in users["user_id"].tolist():
    uid = int(uid)
    print(f"\n--- Updating prices for user_id={uid} ---")
    r = update_all_prices(user_id=uid, only_with_holdings=True)
    print(f"  Updated: {r.updated}, Failed: {r.failed}, Skipped: {r.skipped}")
    for d in r.details:
        print(f"  {d['symbol']}: {d.get('price','?')} {d['status']}")

# Verify
print("\n--- Verification ---")
df = query_df("SELECT user_id, symbol, current_price, currency FROM stocks WHERE current_price > 0 ORDER BY user_id, symbol", ())
print(df.to_string())
