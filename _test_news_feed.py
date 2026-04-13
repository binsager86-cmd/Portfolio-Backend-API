"""Test the news feed endpoint."""
import httpx
import json
import time

# Login
r = httpx.post("http://localhost:8004/api/v1/auth/login", json={"email": "admin", "password": "admin123"}, timeout=10)
if r.status_code != 200:
    r = httpx.post("http://localhost:8004/api/v1/auth/login/form", data={"username": "admin", "password": "admin123"}, timeout=10)

token = r.json().get("access_token", "")
headers = {"Authorization": f"Bearer {token}"}

# Test feed - all categories
print("=== /feed (default) ===")
t0 = time.time()
r = httpx.get("http://localhost:8004/api/v1/news/feed", headers=headers, timeout=30)
print(f"Status: {r.status_code} in {time.time()-t0:.2f}s")
data = r.json()
print(f"Total: {data['totalAvailable']}, Items: {len(data['items'])}")
if data["items"]:
    print(f"First: {data['items'][0]['publishedAt']} - {data['items'][0]['title'][:80]}")
    print(f"Last:  {data['items'][-1]['publishedAt']} - {data['items'][-1]['title'][:80]}")

# Test feed - dividend category
print("\n=== /feed (dividend) ===")
r = httpx.get("http://localhost:8004/api/v1/news/feed?categories=dividend", headers=headers, timeout=30)
data = r.json()
print(f"Total: {data['totalAvailable']}, Items: {len(data['items'])}")

# Test feed - regulatory category
print("\n=== /feed (regulatory) ===")
r = httpx.get("http://localhost:8004/api/v1/news/feed?categories=regulatory", headers=headers, timeout=30)
data = r.json()
print(f"Total: {data['totalAvailable']}, Items: {len(data['items'])}")

# Test feed - specific symbol
print("\n=== /feed (symbol=KFH) ===")
r = httpx.get("http://localhost:8004/api/v1/news/feed?symbols=KFH", headers=headers, timeout=30)
data = r.json()
print(f"Total: {data['totalAvailable']}, Items: {len(data['items'])}")

# Test pagination
print("\n=== /feed (page 2) ===")
r = httpx.get("http://localhost:8004/api/v1/news/feed?cursor=15", headers=headers, timeout=30)
data = r.json()
print(f"Total: {data['totalAvailable']}, Items: {len(data['items'])}, Next: {data['nextPageCursor']}")

# Test history
print("\n=== /history ===")
r = httpx.get("http://localhost:8004/api/v1/news/history?limit=50", headers=headers, timeout=30)
data = r.json()
print(f"Total: {data['totalItems']}, Pages: {data['totalPages']}")
