from app.services.eagle_eye.store import load_all_ratings
rows = load_all_ratings()
print(f"Total in ee_ratings_cache: {len(rows)}")
for r in rows[:20]:
    ticker = r.get("ticker", "?")
    conf = r.get("confidence", 0)
    rating = r.get("rating", "-")
    print(f"  {ticker:15s} conf={conf:.0f}  rating={rating}")
