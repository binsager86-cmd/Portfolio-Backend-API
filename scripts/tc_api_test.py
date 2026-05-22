import hashlib, urllib.request, ssl, random, json

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

PREFIX = 'RX_06_01_15_TC'

def sign(path_and_query):
    return hashlib.md5((PREFIX + path_and_query).encode()).hexdigest()

def call_signed(host, path, params):
    pq = path + '?' + params
    h = sign(pq)
    url = 'https://' + host + pq + '&h=' + h
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'TickerChartLive/4.8.7.33'})
        resp = urllib.request.urlopen(req, context=ctx, timeout=10)
        return resp.status, resp.read().decode('utf-8', 'ignore')
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8', 'ignore')[:400]
    except Exception as e:
        return 0, str(e)

def call_plain(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'TickerChartLive/4.8.7.33'})
        resp = urllib.request.urlopen(req, context=ctx, timeout=10)
        return resp.status, resp.read().decode('utf-8', 'ignore')
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8', 'ignore')[:400]
    except Exception as e:
        return 0, str(e)

rand = random.randint(100000000, 2000000000)
t = '2026-05-18'
u = 'sager123'
bp = 'user_name=' + u + '&language=ENGLISH&version=4.8.7.33&rand=' + str(rand) + '&t=' + t

# 1. Get full streamers to find HISTORICAL_PRICES_PATH per market
print("=== STREAMERS - HISTORICAL PRICES PATH ===")
s, b = call_signed('www.tickerchart.com', '/m/v2/tickerchart/streamers', bp)
print("Status:", s)
servers = {}
if s == 200:
    data = json.loads(b)
    servers = data['response']['STREAMING_SERVER']
    for k, v in servers.items():
        print("  " + v['ABB'] + " -> " + v.get('HISTORICAL_PRICES_PATH', 'N/A'))
        if v.get('INDICATORS_SERVER'):
            print("    IND: " + v['INDICATORS_SERVER'] + ":" + v.get('INDICATORS_PORT','?'))

# 2. Test ondemandDataLoader for KSE (NBK = NBK.KSE)
print()
print("=== HISTORICAL PRICES: NBK.KSE (1day, 30days) ===")
# From streamers, KSE historical prices path
kse_hist_host = None
for k, v in servers.items():
    if v.get('ABB') == 'KSE':
        hp = v.get('HISTORICAL_PRICES_PATH', '')
        if hp.startswith('https://'):
            kse_hist_host = hp.split('/')[2]
            kse_hist_base = '/'.join(hp.split('/')[3:])
        print("  KSE hist path:", hp)

# Try the ondemandDataLoader with no h (seen in memory without h param)
ohlcv_params = 'user_name=' + u + '&language=ENGLISH&symbol=NBK.KSE&interval=1day&period=30day&version=4.8.7.33&rand=' + str(rand) + '&t=' + t
ohlcv_params_h = ohlcv_params  # first try without h
if kse_hist_host:
    path_ohlcv = '/' + kse_hist_base
    url_no_h = 'https://' + kse_hist_host + path_ohlcv + '?' + ohlcv_params
    s3, b3 = call_plain(url_no_h)
    print("Without h - Status:", s3, "Body:", b3[:300])
    if s3 != 200:
        s3b, b3b = call_signed(kse_hist_host, path_ohlcv, ohlcv_params)
        print("With h - Status:", s3b, "Body:", b3b[:300])

# Also try livedata.tickerchart.net directly
print()
print("=== livedata.tickerchart.net ondemandDataLoader ===")
url_ld = 'https://livedata.tickerchart.net/tcdata/ondemandDataLoader.php?' + ohlcv_params
s4, b4 = call_plain(url_ld)
print("No-h Status:", s4, b4[:300])
if s4 != 200:
    s4b, b4b = call_signed('livedata.tickerchart.net', '/tcdata/ondemandDataLoader.php', ohlcv_params)
    print("With-h Status:", s4b, b4b[:300])

# 3. Try index data (KSE index)
print()
print("=== KSE INDEX DATA ===")
idx_params = 'user_name=' + u + '&language=ENGLISH&symbol=KWSE.KSE&interval=1day&period=30day&version=4.8.7.33&rand=' + str(rand) + '&t=' + t
if kse_hist_host:
    url_idx = 'https://' + kse_hist_host + path_ohlcv + '?' + idx_params
    si, bi = call_plain(url_idx)
    print("KWSE index no-h:", si, bi[:300])
    if si != 200:
        si2, bi2 = call_signed(kse_hist_host, path_ohlcv, idx_params)
        print("KWSE index with-h:", si2, bi2[:300])

# 4. Try AdvanceDeclineDataLoader
print()
print("=== ADVANCE/DECLINE ===")
ad_params = 'user_name=' + u + '&language=ENGLISH&market=KSE&version=4.8.7.33&rand=' + str(rand) + '&t=' + t
url_ad = 'https://livedata.tickerchart.net/tcdata/AdvanceDeclineDataLoader.php?' + ad_params
s5, b5 = call_plain(url_ad)
print("A/D no-h:", s5, b5[:300])

# 5. Market info (confirmed working)
print()
print("=== MARKET INFO ===")
s6, b6 = call_signed('www.tickerchart.com', '/m/v2/tickerchart/desktop/market-info', bp)
print("Status:", s6)
if s6 == 200:
    data6 = json.loads(b6)
    for m in data6['response']['markets']:
        print("  market_id=" + m['market_id'] + " abb=" + m['abb'] + " name=" + m['name'])
