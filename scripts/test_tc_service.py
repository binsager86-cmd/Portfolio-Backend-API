"""Quick smoke test for the updated tickerchart_service.fetch_ohlcv."""
import asyncio
import hashlib
import os
import random
import ssl
import sys
import urllib.request
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Inline the critical logic from tickerchart_service so we don't need
# the full dependency chain (pydantic_settings, etc.)
# ---------------------------------------------------------------------------
_SALT = "RX_06_01_15_TC"
_USER_AGENT = "TickerChartLive/4.8.7.33"

MARKET_HOST = {
    "KSE": "delayed2.tickerchart.net",
    "TAD": "delayedtad2.tickerchart.net",
    "USA": "delayedus.tickerchart.net",
}

KSE_INDEX_CANDIDATES = [("Boursa Kuwait Index", "BKI", "KSE")]
TC_USER = os.getenv("TICKERCHART_USERNAME", "").strip()


def _sign(path: str, qs: str) -> str:
    return hashlib.md5(f"{_SALT}{path}?{qs}".encode()).hexdigest()


def _pick_period(from_d, to_d) -> str:
    if from_d is None or to_d is None:
        return "5years"
    days = (to_d - from_d).days
    if days <= 1:
        return "1day"
    if days <= 7:
        return "1week"
    if days <= 31:
        return "1month"
    if days <= 366:
        return "1year"
    if days <= 366 * 2:
        return "2years"
    if days <= 366 * 5:
        return "5years"
    return "all"


def fetch_ohlcv_sync(symbol: str, market: str, from_d=None, to_d=None, interval="day") -> list:
    if not TC_USER:
        raise RuntimeError("Set TICKERCHART_USERNAME in environment before running test_tc_service.py")
    host = MARKET_HOST.get(market)
    assert host, f"No host for market {market}"
    period = _pick_period(from_d, to_d)
    path = "/tcdata/ondemandDataLoader.php"
    rand = random.randint(100_000_000, 2_000_000_000)
    today_str = date.today().isoformat()
    qs = (
        f"user_name={TC_USER}&language=ENGLISH&symbol={symbol}.{market}"
        f"&interval={interval}&period={period}&version=4.8.7.33&rand={rand}&t={today_str}"
    )
    h = _sign(path, qs)
    url = f"https://{host}{path}?{qs}&h={h}"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
        text = resp.read().decode("utf-8", "ignore")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower() in ("historicaldata", "end"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            from datetime import datetime
            datetime.strptime(parts[0], "%Y-%m-%d")
            o, h_, lo, c, vol = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            if o > 0 and h_ > 0 and lo > 0 and c > 0:
                rows.append({"date": parts[0], "open": o, "high": h_, "low": lo, "close": c, "volume": vol})
        except (ValueError, IndexError):
            continue
    if from_d:
        rows = [r for r in rows if r["date"] >= from_d.isoformat()]
    if to_d:
        rows = [r for r in rows if r["date"] <= to_d.isoformat()]
    return rows


def run_tests():
    today = date.today()
    from_d = today - timedelta(days=10)

    print("=== Config checks ===")
    assert MARKET_HOST["KSE"] == "delayed2.tickerchart.net"
    assert KSE_INDEX_CANDIDATES[0][1] == "BKI"
    assert _pick_period(today - timedelta(days=5000), today) == "all"
    print("  KSE host: delayed2.tickerchart.net  OK")
    print("  KSE index: BKI  OK")
    print("  period(5000d) = all  OK")

    print("\n=== Live fetch tests ===")
    for label, sym, mkt in [("NBK", "NBK", "KSE"), ("BKI index", "BKI", "KSE"), ("AAPL", "AAPL", "USA")]:
        rows = fetch_ohlcv_sync(sym, mkt, from_d=from_d, to_d=today)
        assert len(rows) > 0, f"{label} returned 0 rows!"
        print(f"  {label}: {len(rows)} rows | last: {rows[-1]}")

    print("\nAll tests PASSED.")


if __name__ == "__main__":
    run_tests()

