"""
Full CRUD verification — creates, reads, updates, deletes for each new resource.
"""
import requests, json

BASE = "http://127.0.0.1:8003"

def auth():
    r = requests.post(f"{BASE}/api/v1/auth/login", json={"username": "tester99", "password": "Test123!"})
    if r.status_code == 401:
        r = requests.post(f"{BASE}/api/v1/auth/register", json={"username": "tester99", "password": "Test123!"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}

def api(method, path, expected=200, **kw):
    r = getattr(requests, method)(f"{BASE}{path}", **kw)
    ok = r.status_code == expected
    tag = "OK" if ok else "FAIL"
    body = r.json() if r.headers.get("content-type","").startswith("application/json") else r.text[:60]
    print(f"  [{tag}] {method.upper():6s} {path} -> {r.status_code}")
    if not ok:
        print(f"         Expected {expected}, got: {json.dumps(body, default=str)[:200]}")
    return r, ok

h = auth()
all_ok = True

# ── Clean up from previous runs ─────────────────────────────────────
print("=== CLEANUP ===")
# Delete test stock if exists
r = requests.get(f"{BASE}/api/v1/stocks/by-symbol/CRUDTEST", headers=h)
if r.status_code == 200:
    sid = r.json()["data"]["id"]
    requests.delete(f"{BASE}/api/v1/stocks/{sid}", headers=h)
    print(f"  Cleaned up stock CRUDTEST (id={sid})")

# Delete test security if exists
r = requests.get(f"{BASE}/api/v1/securities/KSE:CRUDTEST", headers=h)
if r.status_code == 200:
    requests.delete(f"{BASE}/api/v1/securities/KSE:CRUDTEST", headers=h)
    print("  Cleaned up security KSE:CRUDTEST")

# ── STOCKS CRUD ──────────────────────────────────────────────────────
print("\n=== STOCKS CRUD ===")

# Create
r, ok = api("post", "/api/v1/stocks", 201, headers=h,
    json={"symbol": "CRUDTEST", "name": "CRUD Test Stock", "portfolio": "KFH", "currency": "KWD"})
all_ok &= ok
stock_id = r.json()["data"]["id"]
print(f"  Created stock id={stock_id}")

# Read
r, ok = api("get", f"/api/v1/stocks/{stock_id}", 200, headers=h)
all_ok &= ok
assert r.json()["data"]["name"] == "CRUD Test Stock"

# Read by symbol
r, ok = api("get", "/api/v1/stocks/by-symbol/CRUDTEST", 200, headers=h)
all_ok &= ok

# Update
r, ok = api("put", f"/api/v1/stocks/{stock_id}", 200, headers=h,
    json={"name": "CRUD Updated", "current_price": 1.5})
all_ok &= ok

# Verify update
r, ok = api("get", f"/api/v1/stocks/{stock_id}", 200, headers=h)
all_ok &= ok
assert r.json()["data"]["name"] == "CRUD Updated"
assert r.json()["data"]["current_price"] == 1.5

# Delete
r, ok = api("delete", f"/api/v1/stocks/{stock_id}", 200, headers=h)
all_ok &= ok

# Verify delete
r, ok = api("get", f"/api/v1/stocks/{stock_id}", 404, headers=h)
all_ok &= ok

# ── SECURITIES CRUD ──────────────────────────────────────────────────
print("\n=== SECURITIES CRUD ===")

# Create
r, ok = api("post", "/api/v1/securities", 201, headers=h,
    json={"canonical_ticker": "CRUDTEST", "exchange": "KSE", "display_name": "CRUD Test Sec", "currency": "KWD"})
all_ok &= ok
sec_id = r.json()["data"]["security_id"]
print(f"  Created security_id={sec_id}")

# Read
r, ok = api("get", f"/api/v1/securities/{sec_id}", 200, headers=h)
all_ok &= ok
assert r.json()["data"]["display_name"] == "CRUD Test Sec"

# Update
r, ok = api("put", f"/api/v1/securities/{sec_id}", 200, headers=h,
    json={"display_name": "CRUD Updated Sec"})
all_ok &= ok

# Verify update
r, ok = api("get", f"/api/v1/securities/{sec_id}", 200, headers=h)
all_ok &= ok
assert r.json()["data"]["display_name"] == "CRUD Updated Sec"

# Add alias
r, ok = api("post", f"/api/v1/securities/{sec_id}/aliases", 201, headers=h,
    json={"alias_name": "CRUDTEST_ALIAS", "alias_type": "user_input"})
all_ok &= ok

# List aliases
r, ok = api("get", f"/api/v1/securities/{sec_id}/aliases", 200, headers=h)
all_ok &= ok
assert r.json()["data"]["count"] >= 1

# Delete alias
r, ok = api("delete", f"/api/v1/securities/{sec_id}/aliases/CRUDTEST_ALIAS", 200, headers=h)
all_ok &= ok

# Delete security
r, ok = api("delete", f"/api/v1/securities/{sec_id}", 200, headers=h)
all_ok &= ok

# Verify delete
r, ok = api("get", f"/api/v1/securities/{sec_id}", 404, headers=h)
all_ok &= ok

# ── CASH DEPOSIT CRUD ────────────────────────────────────────────────
print("\n=== CASH DEPOSIT CRUD ===")

# Create
r, ok = api("post", "/api/v1/cash/deposits", 201, headers=h,
    json={"portfolio": "KFH", "deposit_date": "2025-01-15", "amount": 1000, "currency": "KWD", "bank_name": "Test Bank"})
all_ok &= ok
dep_id = r.json()["data"]["id"]
print(f"  Created deposit id={dep_id}")

# Read
r, ok = api("get", f"/api/v1/cash/deposits/{dep_id}", 200, headers=h)
all_ok &= ok

# Update
r, ok = api("put", f"/api/v1/cash/deposits/{dep_id}", 200, headers=h,
    json={"amount": 2000})
all_ok &= ok

# Delete (soft)
r, ok = api("delete", f"/api/v1/cash/deposits/{dep_id}", 200, headers=h)
all_ok &= ok

# Restore
r, ok = api("post", f"/api/v1/cash/deposits/{dep_id}/restore", 200, headers=h)
all_ok &= ok

# Clean up - delete again
api("delete", f"/api/v1/cash/deposits/{dep_id}", 200, headers=h)

# ── TRANSACTION CRUD ─────────────────────────────────────────────────
print("\n=== TRANSACTION CRUD ===")

# Create
r, ok = api("post", "/api/v1/portfolio/transactions", 201, headers=h,
    json={"portfolio": "KFH", "stock_symbol": "CRUDTEST", "txn_date": "2025-01-15",
          "txn_type": "Buy", "shares": 100, "purchase_cost": 500})
all_ok &= ok
txn_id = r.json()["data"]["id"]
print(f"  Created transaction id={txn_id}")

# Read
r, ok = api("get", f"/api/v1/portfolio/transactions/{txn_id}", 200, headers=h)
all_ok &= ok

# Update
r, ok = api("put", f"/api/v1/portfolio/transactions/{txn_id}", 200, headers=h,
    json={"shares": 200})
all_ok &= ok

# Delete (soft)
r, ok = api("delete", f"/api/v1/portfolio/transactions/{txn_id}", 200, headers=h)
all_ok &= ok

# Restore
r, ok = api("post", f"/api/v1/portfolio/transactions/{txn_id}/restore", 200, headers=h)
all_ok &= ok

# Clean up
api("delete", f"/api/v1/portfolio/transactions/{txn_id}", 200, headers=h)

# ── TRACKER ──────────────────────────────────────────────────────────
print("\n=== TRACKER ===")
r, ok = api("post", "/api/v1/tracker/save-snapshot", 201, headers=h)
all_ok &= ok

# ── SUMMARY ──────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"{'✅  ALL CRUD TESTS PASSED!' if all_ok else '❌  SOME TESTS FAILED'}")
print(f"{'='*50}")
