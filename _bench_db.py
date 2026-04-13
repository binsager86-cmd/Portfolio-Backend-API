"""Quick DB query timing test."""
import time
import sys
sys.path.insert(0, '.')
from app.core.database import SessionLocal
from app.models.news import NewsArticle

db = SessionLocal()

# Test 1: Count all
t0 = time.time()
total = db.query(NewsArticle).count()
print(f"COUNT all: {total} in {time.time()-t0:.3f}s")

# Test 2: Count with ORDER BY and LIMIT
t0 = time.time()
rows = db.query(NewsArticle).order_by(NewsArticle.published_at.desc()).limit(15).all()
print(f"ORDER+LIMIT 15: {len(rows)} in {time.time()-t0:.3f}s")

# Test 3: Count + ORDER + LIMIT
t0 = time.time()
total = db.query(NewsArticle).count()
rows = db.query(NewsArticle).order_by(NewsArticle.published_at.desc()).offset(0).limit(15).all()
print(f"COUNT+ORDER+LIMIT: {total}/{len(rows)} in {time.time()-t0:.3f}s")

# Test 4: Count with category filter
t0 = time.time()
q = db.query(NewsArticle).filter(NewsArticle.category == 'dividend')
cnt = q.count()
items = q.order_by(NewsArticle.published_at.desc()).limit(15).all()
print(f"DIVIDEND count+fetch: {cnt}/{len(items)} in {time.time()-t0:.3f}s")

db.close()
