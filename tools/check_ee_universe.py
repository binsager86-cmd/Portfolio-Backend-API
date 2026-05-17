"""Check Eagle Eye universe and ratings cache size."""
import sys, os
sys.path.insert(0, ".")
from app.core.database import query_all

ratings = query_all("SELECT COUNT(*) as cnt FROM ee_ratings_cache")
print(f"ee_ratings_cache rows: {ratings[0]['cnt']}")

try:
    stocks = query_all("SELECT COUNT(*) as cnt FROM analysis_stocks WHERE exchange IN ('KW','KSE') OR currency='KWD'")
    print(f"analysis_stocks (KW universe): {stocks[0]['cnt']}")
    sample = query_all("SELECT symbol FROM analysis_stocks WHERE exchange IN ('KW','KSE') OR currency='KWD' LIMIT 5")
    print(f"Sample tickers: {[r['symbol'] for r in sample]}")
except Exception as e:
    print(f"analysis_stocks error: {e}")

try:
    sm = query_all("SELECT COUNT(*) as cnt FROM stocks_master")
    print(f"stocks_master total: {sm[0]['cnt']}")
    sm_kw = query_all("SELECT COUNT(*) as cnt FROM stocks_master WHERE market='KW' OR exchange='KSE'")
    print(f"stocks_master (KW): {sm_kw[0]['cnt']}")
except Exception as e:
    print(f"stocks_master error: {e}")

# Show what the adapter would use
from app.services.eagle_eye.adapter import TickerChartAdapter
adapter = TickerChartAdapter()
stocks_list = adapter.list_stocks()
print(f"\nadapter.list_stocks() returns: {len(stocks_list)} stocks")
if stocks_list:
    print(f"First 5: {[s.ticker for s in stocks_list[:5]]}")
    print(f"Last 5:  {[s.ticker for s in stocks_list[-5:]]}")
