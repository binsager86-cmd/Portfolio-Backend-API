import requests, json
r = requests.post('http://localhost:8004/api/auth/login/json', json={'username':'sager alsager','password':'Admin123!'})
token = r.json()['access_token']
h = {'Authorization': f'Bearer {token}'}
r2 = requests.get('http://localhost:8004/api/portfolio/holdings', headers=h)
d = r2.json()['data']
for x in d['holdings']:
    if x.get('currency') == 'USD':
        print(json.dumps({k: x[k] for k in ['company','currency','shares_qty','avg_cost','total_cost','market_price','market_value','market_value_kwd','total_cost_kwd','unrealized_pnl','unrealized_pnl_kwd','total_pnl','total_pnl_kwd']}, indent=2))
