import requests
r = requests.post('http://localhost:8004/api/auth/login/json', json={'username':'sager alsager','password':'Admin123!'})
token = r.json()['access_token']
h = {'Authorization': f'Bearer {token}'}
# All holdings
r = requests.get('http://localhost:8004/api/portfolio/holdings', headers=h)
d = r.json()['data']
print(f'Total Count: {d["count"]}')
for hld in d['holdings']:
    ccy = hld.get('currency','?')
    line = f'  {hld["company"]}: ccy={ccy} qty={hld["shares_qty"]} avg={hld["avg_cost"]:.3f} cost={hld["total_cost"]:.2f} mkt_price={hld["market_price"]:.3f} mkt_val={hld["market_value"]:.2f}'
    if ccy == 'USD':
        line += f' mkt_val_kwd={hld["market_value_kwd"]:.2f} cost_kwd={hld["total_cost_kwd"]:.2f}'
    print(line)
