"""TickerChart Live wrapper.

Authenticates against TickerChart's mobile API, signs every request with the
recovered MD5 query-string signature, fetches OHLCV from the per-market data
host and returns rows in the EODHD-compatible shape the mobile WhaleRadar
already consumes.

Signature algorithm (recovered via runtime BCryptHashData hook):
    h = md5("RX_06_01_15_TC" + path + "?" + query_string_without_h)
"""
from __future__ import annotations

import hashlib
import logging
import random
import time
from datetime import date, datetime
from typing import Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── Constants recovered from TickerChart Live 4.8.7.31 ──────────────
_VERSION = "4.8.7.31"
_SALT = "RX_06_01_15_TC"
_USER_AGENT = "RestSharp/4.8.7.31"
_LOGIN_HOST = "www.tickerchart.com"
_LOGIN_PATH = "/m/v2/tickerchart/live/login"

# Per-market historical-prices host (from /m/v2/tickerchart/streamers capture).
# Suffix is the abbreviation we pass to ondemandDataLoader.php as `<SYMBOL>.<ABB>`.
_MARKET_HOST: dict[str, str] = {
    "KSE": "livedata.tickerchart.net",       # Kuwait
    "TAD": "delayedtad2.tickerchart.net",    # Tadawul (Saudi)
    "DFM": "delayed2.tickerchart.net",       # Dubai
    "ADX": "delayed2.tickerchart.net",       # Abu Dhabi
    "DSM": "delayed2.tickerchart.net",       # Doha (Qatar)
    "EGY": "delayed2.tickerchart.net",       # Egypt
    "USA": "delayedus.tickerchart.net",
    "FRX": "livedata06.tickerchart.net",
}

# Mobile-side suffix → TickerChart abbreviation.
_SUFFIX_MAP: dict[str, str] = {
    "KW": "KSE",        # mobile sends KFH.KW
    "KSE": "KSE",
    "BK": "KSE",
    "SR": "TAD",
    "TADAWUL": "TAD",
    "DFM": "DFM",
    "ADX": "ADX",
    "QSE": "DSM",
    "DSM": "DSM",
    "EGY": "EGY",
    "EGX": "EGY",
    "US": "USA",
    "USA": "USA",
}


# ── Token cache ──────────────────────────────────────────────────────
_token_cache: dict[str, object] = {"token": None, "expires": 0.0}


def _sign(path: str, query_pairs: list[tuple[str, str]]) -> tuple[str, str]:
    """Return (final_query_string_with_h, h)."""
    qs = "&".join(f"{k}={v}" for k, v in query_pairs)
    plain = f"{_SALT}{path}?{qs}"
    h = hashlib.md5(plain.encode("utf-8")).hexdigest()
    return f"{qs}&h={h}", h


def _common_params() -> list[tuple[str, str]]:
    return [
        ("version", _VERSION),
        ("rand", str(random.randint(1, 2_147_483_647))),
        ("t", date.today().isoformat()),
    ]


def _resolve_market(suffix: Optional[str]) -> Optional[str]:
    if not suffix:
        return None
    return _SUFFIX_MAP.get(suffix.strip().upper())


def split_symbol(symbol: str, exchange: Optional[str], country: Optional[str]) -> Optional[tuple[str, str]]:
    """Translate a mobile symbol like 'KFH.KW' or ('KFH', exchange='KW') to ('KFH', 'KSE')."""
    if not symbol:
        return None
    sym = symbol.strip().upper()
    if "." in sym:
        base, _, suf = sym.partition(".")
        market = _resolve_market(suf)
        if base and market:
            return base, market
        return None
    base = sym
    market = _resolve_market(exchange) or _resolve_market(country)
    if base and market:
        return base, market
    return None


# ── Auth ─────────────────────────────────────────────────────────────
async def _login(client: httpx.AsyncClient) -> str:
    settings = get_settings()
    username = (settings.TICKERCHART_USERNAME or "").strip()
    password = (settings.TICKERCHART_PASSWORD or "").strip()
    if not username or not password:
        raise RuntimeError("TICKERCHART_USERNAME / TICKERCHART_PASSWORD not configured")

    # TickerChart accepts the password base64-encoded.
    import base64
    pw_b64 = base64.b64encode(password.encode("utf-8")).decode("ascii")

    qs_pairs = _common_params()
    final_qs, _ = _sign(_LOGIN_PATH, qs_pairs)
    url = f"https://{_LOGIN_HOST}{_LOGIN_PATH}?{final_qs}"

    resp = await client.post(
        url,
        json={"username": username, "password": pw_b64},
        headers={"User-Agent": _USER_AGENT, "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict) or not body.get("success"):
        raise RuntimeError(f"TickerChart login failed: {body!r}")
    token = (body.get("response") or {}).get("token")
    if not token:
        raise RuntimeError("TickerChart login returned no token")
    return token


async def _get_token() -> str:
    now = time.time()
    cached = _token_cache.get("token")
    if cached and float(_token_cache.get("expires", 0)) > now:
        return str(cached)
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        token = await _login(client)
    _token_cache["token"] = token
    _token_cache["expires"] = now + 8 * 3600  # session is good ≥ several hours; refresh every 8 h
    return token


# ── OHLCV ────────────────────────────────────────────────────────────
def _pick_period(from_d: Optional[date], to_d: Optional[date]) -> str:
    """Pick the smallest TickerChart `period` that covers the requested range."""
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
    return "10years"


async def fetch_ohlcv(
    base_symbol: str,
    market_abb: str,
    from_d: Optional[date] = None,
    to_d: Optional[date] = None,
    interval: str = "day",
) -> list[dict]:
    """Return list of EODHD-shaped rows: {date, open, high, low, close, volume}.

    Re-logs in once if the cached token is rejected.
    """
    host = _MARKET_HOST.get(market_abb)
    if host is None:
        raise ValueError(f"Unsupported market: {market_abb}")

    period = _pick_period(from_d, to_d)
    path = "/tcdata/ondemandDataLoader.php"
    user_name = (get_settings().TICKERCHART_USERNAME or "").strip()

    async def _do_request(token: str) -> httpx.Response:
        qs_pairs = [
            ("user_name", user_name),
            ("language", "ENGLISH"),
            ("symbol", f"{base_symbol}.{market_abb}"),
            ("interval", interval),
            ("period", period),
        ] + _common_params()
        final_qs, _ = _sign(path, qs_pairs)
        url = f"https://{host}{path}?{final_qs}"
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            return await client.get(
                url,
                headers={
                    "User-Agent": _USER_AGENT,
                    "Authorization": f"TcToken{token}",
                },
            )

    token = await _get_token()
    resp = await _do_request(token)
    if resp.status_code in (401, 403):
        logger.info("TickerChart token rejected, re-logging in")
        _token_cache["token"] = None
        token = await _get_token()
        resp = await _do_request(token)
    resp.raise_for_status()

    rows = _parse_ondemand_csv(resp.text)
    # Apply requested date window (TickerChart returns whole period buckets)
    if from_d is not None:
        rows = [r for r in rows if r["date"] >= from_d.isoformat()]
    if to_d is not None:
        rows = [r for r in rows if r["date"] <= to_d.isoformat()]
    return rows


def _parse_ondemand_csv(text: str) -> list[dict]:
    """Parse the text/plain response of ondemandDataLoader.php.

    Format:
        HistoricalData
        YYYY-MM-DD,open,high,low,close,volume,value,trades,flag
        ...
    Lines may include trailing fields we don't need; we keep only OHLCV.
    """
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower() == "historicaldata":
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        d = parts[0].strip()
        # Reject anything that isn't an ISO date — guards against header lines
        try:
            datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            continue
        try:
            out.append({
                "date": d,
                "open": float(parts[1] or 0),
                "high": float(parts[2] or 0),
                "low": float(parts[3] or 0),
                "close": float(parts[4] or 0),
                "volume": float(parts[5] or 0),
            })
        except ValueError:
            continue
    
    # Deduplicate by date (ex-dividend dates may appear twice)
    # Keep the entry with highest volume for each date
    out.sort(key=lambda r: r["date"])
    deduped: dict[str, dict] = {}
    for row in out:
        d = row["date"]
        if d not in deduped:
            deduped[d] = row
        else:
            # Keep the one with higher volume (actual trading data)
            if row["volume"] > deduped[d]["volume"]:
                deduped[d] = row
    
    return sorted(deduped.values(), key=lambda r: r["date"])
