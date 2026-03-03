"""End-to-end test: deposit creates -> total_value increases -> delete -> reverts."""
import requests

BASE = 'http://localhost:8004/api'

# Login
r = requests.post(f'{BASE}/auth/login/json', json={'username':'sager alsager','password':'Admin123!'})
token = r.json()['access_token']
h = {'Authorization': f'Bearer {token}'}

def unwrap(resp_json):
    if isinstance(resp_json, dict) and 'data' in resp_json:
        return resp_json['data']
    return resp_json

# Get overview BEFORE
r = requests.get(f'{BASE}/v1/portfolio/overview', headers=h)
ov = unwrap(r.json())
print('=== BEFORE ===')
print(f"total_value: {ov.get('total_value')}")
print(f"portfolio_value: {ov.get('portfolio_value')}")
print(f"cash_balance: {ov.get('cash_balance')}")

# Check cash balances via analytics endpoint
r = requests.get(f'{BASE}/v1/analytics/cash-balances', headers=h)
bals = unwrap(r.json())
print('\n=== CASH BALANCES ===')
if isinstance(bals, list):
    for b in bals:
        if isinstance(b, dict):
            print(f"  {b.get('portfolio','?')}: {b.get('balance','?')} (override={b.get('manual_override','?')})")
elif isinstance(bals, dict):
    for k, v in bals.items():
        print(f"  {k}: {v}")

# Create test deposit
print('\n=== CREATE TEST DEPOSIT (100 KWD to KFH) ===')
dep = {'portfolio':'KFH','amount':100,'deposit_date':'2025-01-15','bank_name':'Test Bank','notes':'e2e test'}
r = requests.post(f'{BASE}/v1/cash/deposits', headers=h, json=dep)
resp = unwrap(r.json())
print(f"Status: {r.status_code}")
dep_id = resp.get('id')
print(f"deposit id: {dep_id}")
print(f"cash_balance returned: {resp.get('cash_balance')}")
print(f"total_value returned: {resp.get('total_value')}")

# Get overview AFTER
r = requests.get(f'{BASE}/v1/portfolio/overview', headers=h)
ov2 = unwrap(r.json())
print('\n=== AFTER DEPOSIT (overview) ===')
print(f"total_value: {ov2.get('total_value')}")
print(f"cash_balance: {ov2.get('cash_balance')}")
before_total = ov.get('total_value') or 0
after_total = ov2.get('total_value') or 0
diff = after_total - before_total
print(f"Diff total: {diff:.3f}")
if abs(diff - 100) < 0.01:
    print("PASS: Total increased by ~100 KWD")
else:
    print(f"WARN: Expected +100, got +{diff:.3f}")

# Delete test deposit
if dep_id:
    print(f'\n=== DELETE TEST DEPOSIT id={dep_id} ===')
    r = requests.delete(f'{BASE}/v1/cash/deposits/{dep_id}', headers=h)
    print(f"Status: {r.status_code}")
    r = requests.get(f'{BASE}/v1/portfolio/overview', headers=h)
    ov3 = unwrap(r.json())
    print('\n=== AFTER DELETE ===')
    print(f"total_value: {ov3.get('total_value')}")
    after_del = ov3.get('total_value') or 0
    diff2 = after_del - before_total
    print(f"Diff from original: {diff2:.3f}")
    if abs(diff2) < 0.01:
        print("PASS: Total reverted to original")
    else:
        print(f"WARN: Expected 0 diff, got {diff2:.3f}")
else:
    print("ERROR: No deposit ID returned, cannot delete")
