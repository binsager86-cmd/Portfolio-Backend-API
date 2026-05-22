import json

data = json.load(open(r'C:\Users\Sager\AppData\Local\UniTicker\TCLive\Cache\CategoriesAndCompaniesData.json'))
companies = data.get('Companies') or []

print(f'Total companies: {len(companies)}')
print('Fields:', list(companies[0].keys()) if companies else 'N/A')

# Find NBK, Ooredoo by TickerID
for c in companies:
    ticker = str(c.get('TickerID') or '').upper()
    market = str(c.get('MarketAbrv') or '').upper()
    cid = c.get('ID')
    if ticker in ('NBK', 'OOREDOO', 'KFH', 'ZAIN', 'BOURSA', 'BOUBYAN', 'WARBA') and market == 'KSE':
        name = c.get('EnglishName', '')
        print(f'TickerID={ticker}  ID={cid}  Market={market}  Name={name}')
