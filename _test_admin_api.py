"""Test admin API endpoint after fix."""
import requests
import json

resp = requests.post('http://localhost:8004/api/v1/auth/login', json={
    'username': 'Admin', 'password': 's.0400148sager'
})
print("Login status:", resp.status_code)
data = resp.json()
token = data.get('access_token') or data.get('data', {}).get('access_token')

headers = {'Authorization': f'Bearer {token}'}

# Test users
resp2 = requests.get('http://localhost:8004/api/v1/admin/users', headers=headers)
print("Users status:", resp2.status_code)
if resp2.status_code == 200:
    d = resp2.json()
    print(f"Count: {d['count']}")
    for u in d['users'][:5]:
        print(f"  id={u['id']} username={u['username']} name={u['name']} "
              f"portfolio={u['portfolio_value']} growth={u['growth_value']} "
              f"last_login={u['last_login']} txns={u['transaction_count']}")
else:
    print("Error:", resp2.text[:500])

# Test activities
resp3 = requests.get('http://localhost:8004/api/v1/admin/activities', headers=headers)
print(f"\nActivities status: {resp3.status_code}")
if resp3.status_code == 200:
    d3 = resp3.json()
    print(f"Total: {d3['total']}, Count on page: {d3['count']}")
    for a in d3['activities'][:3]:
        print(f"  {a['txn_date']} {a['txn_type']} {a['stock_symbol']} "
              f"shares={a['shares']} value={a['value']} user=@{a['username']}")
else:
    print("Error:", resp3.text[:500])
