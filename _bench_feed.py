"""Precise feed timing."""
import httpx, time

# Login once
r = httpx.post("http://localhost:8004/api/v1/auth/login/form", data={"username": "admin", "password": "admin123"}, timeout=10)
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Warm up
httpx.get("http://localhost:8004/api/v1/news/feed?limit=1", headers=headers, timeout=30)

# Test 1: default feed
t0 = time.time()
r = httpx.get("http://localhost:8004/api/v1/news/feed", headers=headers, timeout=30)
print(f"Feed (default): {time.time()-t0:.3f}s - {r.json()['totalAvailable']} items")

# Test 2: dividend
t0 = time.time()
r = httpx.get("http://localhost:8004/api/v1/news/feed?categories=dividend", headers=headers, timeout=30)
print(f"Feed (dividend): {time.time()-t0:.3f}s - {r.json()['totalAvailable']} items")

# Test 3: symbol
t0 = time.time()
r = httpx.get("http://localhost:8004/api/v1/news/feed?symbols=KFH", headers=headers, timeout=30)
print(f"Feed (KFH): {time.time()-t0:.3f}s - {r.json()['totalAvailable']} items")

# Test 4: page 100
t0 = time.time()
r = httpx.get("http://localhost:8004/api/v1/news/feed?cursor=1500", headers=headers, timeout=30)
print(f"Feed (page 100): {time.time()-t0:.3f}s - {r.json()['totalAvailable']} items")
