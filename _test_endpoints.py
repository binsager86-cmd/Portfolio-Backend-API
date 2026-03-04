"""
Endpoint smoke-test — exercises every new router.
Run:  python _test_endpoints.py
"""
import requests, json, sys

BASE = "http://127.0.0.1:8003"
TOKEN = None

def auth():
    """Register or login, return bearer token."""
    global TOKEN
    # try login first
    r = requests.post(f"{BASE}/api/v1/auth/login", json={"username": "tester99", "password": "Test123!"})
    if r.status_code == 401:
        r = requests.post(f"{BASE}/api/v1/auth/register", json={"username": "tester99", "password": "Test123!"})
    data = r.json()
    TOKEN = data["access_token"]
    print(f"[AUTH] Logged in as {data.get('username','?')}  (user_id={data.get('user_id')})")
    return {"Authorization": f"Bearer {TOKEN}"}

def test(method, path, expected_codes=(200,), **kwargs):
    url = f"{BASE}{path}"
    r = getattr(requests, method)(url, **kwargs)
    ok = r.status_code in expected_codes
    status = "PASS" if ok else "FAIL"
    detail = ""
    try:
        body = r.json()
        if isinstance(body, dict):
            detail = json.dumps(body, default=str)[:200]
        elif isinstance(body, list):
            detail = f"[{len(body)} items]"
    except Exception:
        detail = r.text[:120]
    print(f"  [{status}] {method.upper():6s} {path}  -> {r.status_code}  {detail}")
    return r

# ── Auth ─────────────────────────────────────────────────────────────
h = auth()

# ── Existing: Portfolio ──────────────────────────────────────────────
print("\n=== PORTFOLIO ===")
test("get", "/api/v1/portfolio/overview", headers=h)
test("get", "/api/v1/portfolio/holdings", headers=h)
test("get", "/api/v1/portfolio/accounts", headers=h)
test("get", "/api/v1/portfolio/fx-rate", headers=h)

# ── Existing: Transactions CRUD ──────────────────────────────────────
print("\n=== TRANSACTIONS ===")
test("get", "/api/v1/portfolio/transactions?limit=3", headers=h)

# ── Existing: Cash Deposits ──────────────────────────────────────────
print("\n=== CASH DEPOSITS ===")
test("get", "/api/v1/cash/deposits?limit=3", headers=h)

# ── Existing: Analytics ──────────────────────────────────────────────
print("\n=== ANALYTICS ===")
test("get", "/api/v1/analytics/performance?portfolio=KFH", headers=h)
test("get", "/api/v1/analytics/snapshots", headers=h)

# ── NEW: Dividends ───────────────────────────────────────────────────
print("\n=== DIVIDENDS (new) ===")
test("get", "/api/v1/dividends", headers=h)
test("get", "/api/v1/dividends/by-stock", headers=h)

# ── NEW: Securities ──────────────────────────────────────────────────
print("\n=== SECURITIES (new) ===")
test("get", "/api/v1/securities", headers=h)
# Create
r = test("post", "/api/v1/securities", headers=h, expected_codes=(201, 409),
         json={"canonical_ticker": "TEST.KW", "display_name": "Test Corp", "exchange": "KSE", "currency": "KWD"})
if r.status_code == 201:
    sec_id = r.json().get("data", {}).get("security_id")
    print(f"    Created security_id={sec_id}")
    # Get
    test("get", f"/api/v1/securities/{sec_id}", headers=h)
    # Update
    test("put", f"/api/v1/securities/{sec_id}", headers=h, json={"company_name": "Test Corp Updated"})
    # Aliases
    test("get", f"/api/v1/securities/{sec_id}/aliases", headers=h)
    test("post", f"/api/v1/securities/{sec_id}/aliases", headers=h, expected_codes=(201, 409),
         json={"alias_name": "TESTCORP", "alias_type": "user_input"})
    test("delete", f"/api/v1/securities/{sec_id}/aliases/TESTCORP", headers=h, expected_codes=(200, 204))
    # Delete security
    test("delete", f"/api/v1/securities/{sec_id}", headers=h, expected_codes=(200, 204))
else:
    print("    (skipping CRUD — security already exists or creation failed)")

# ── NEW: Stocks ──────────────────────────────────────────────────────
print("\n=== STOCKS (new) ===")
test("get", "/api/v1/stocks", headers=h)
test("get", "/api/v1/stocks?portfolio=KFH", headers=h)
# Create
r = test("post", "/api/v1/stocks", headers=h, expected_codes=(201, 409),
         json={"symbol": "SMOKTEST", "name": "Smoke Test Inc", "portfolio": "KFH", "currency": "KWD"})
if r.status_code == 201:
    st_id = r.json().get("data", {}).get("id")
    print(f"    Created stock id={st_id}")
    test("get", f"/api/v1/stocks/{st_id}", headers=h)
    test("get", "/api/v1/stocks/by-symbol/SMOKTEST", headers=h)
    test("put", f"/api/v1/stocks/{st_id}", headers=h, json={"name": "Smoke Test Updated"})
    test("delete", f"/api/v1/stocks/{st_id}", headers=h, expected_codes=(200, 204))
else:
    print("    (skipping CRUD — stock may already exist)")

# ── NEW: Tracker ─────────────────────────────────────────────────────
print("\n=== TRACKER (new) ===")
r = test("post", "/api/v1/tracker/save-snapshot", headers=h, expected_codes=(200, 201))
if r.status_code in (200, 201):
    data = r.json()
    snap_id = None
    if isinstance(data, dict) and "rows" in data:
        rows = data["rows"]
        if rows:
            snap_id = rows[0].get("id")
    if snap_id:
        test("delete", f"/api/v1/tracker/snapshots/{snap_id}", headers=h, expected_codes=(200, 204))

# ── NEW: Backup ──────────────────────────────────────────────────────
print("\n=== BACKUP (new) ===")
test("get", "/api/v1/backup/export", headers=h)

# ── Integrity ────────────────────────────────────────────────────────
print("\n=== INTEGRITY ===")
test("get", "/api/v1/integrity/check", headers=h)
test("get", "/api/v1/integrity/completeness", headers=h)

print("\n✅  Smoke test complete!")
