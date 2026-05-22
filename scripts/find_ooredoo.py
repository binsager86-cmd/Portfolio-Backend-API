import json, os

cache = r'C:\Users\Sager\AppData\Local\UniTicker\TCLive\Cache\MarketInfo.json'
data = json.load(open(cache))
companies = (data.get('COMPANIES') or {}).get('VALUES') or {}

keywords = ['OOREDOO', 'OOR', 'ZAIN', 'WATANIYA', 'VIVA']
for cid, row in companies.items():
    if not isinstance(row, list) or len(row) < 2:
        continue
    ticker = str(row[0]).upper()
    for kw in keywords:
        if kw in ticker:
            print(f'ID={cid}  ticker={row[0]}  market={row[1]}')
            break
