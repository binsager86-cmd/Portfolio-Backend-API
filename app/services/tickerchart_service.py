"""TickerChart Live wrapper.

Authenticates against TickerChart's mobile API, signs every request with the
recovered MD5 query-string signature, fetches OHLCV from the per-market data
host and returns rows in the EODHD-compatible shape the mobile WhaleRadar
already consumes.

Signature algorithm (recovered via runtime BCryptHashData hook):
    h = md5("RX_06_01_15_TC" + path + "?" + query_string_without_h)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Network guardrails for outbound TickerChart calls used by recompute paths.
# Keep finite connect/read timeouts and bounded retries so one slow symbol
# cannot stall the full nightly pipeline.
_TC_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=20.0)
_TC_FETCH_MAX_ATTEMPTS = 2
_TC_FETCH_RETRY_BACKOFF_SEC = 0.4

# ── Constants recovered from TickerChart Live 4.8.7.31 ──────────────
_VERSION = "4.8.7.31"
_SALT = "RX_06_01_15_TC"
_USER_AGENT = "RestSharp/4.8.7.31"
_LOGIN_HOST = "www.tickerchart.com"
_LOGIN_PATH = "/m/v2/tickerchart/live/login"
_DESKTOP_MARKET_INFO_PATH = "/m/v2/tickerchart/desktop/market-info"
_FINANCIAL_FIELD_PATH = "/m/v2/tickerchart/financial-field/company/"

# Per-market historical-prices host (from /m/v2/tickerchart/streamers capture).
# Suffix is the abbreviation we pass to ondemandDataLoader.php as `<SYMBOL>.<ABB>`.
_MARKET_HOST: dict[str, str] = {
    "KSE": "delayed2.tickerchart.net",      # Kuwait
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
    # Standard exchange names → USA (stocks created via quick-scan may use these)
    "NYSE": "USA",
    "NASDAQ": "USA",
    "AMEX": "USA",
    "ARCX": "USA",   # NYSE Arca
    "BATS": "USA",
}


# ── Token cache ──────────────────────────────────────────────────────
_token_cache: dict[str, object] = {"token": None, "expires": 0.0}
_company_id_cache: dict[str, object] = {"entries": None, "expires": 0.0}
_kse_market_tier_cache: Optional[dict[str, str]] = None
_quotes_snapshot_cache: dict[str, object] = {
    "entries": None,
    "path": None,
    "mtime": None,
    "expires": 0.0,
}
_factset_eps_cache: dict[tuple[str, str], dict[str, object]] = {}
_FACTSET_EPS_TTL_SEC = 6 * 3600.0
_FACTSET_EPS_MISS_TTL_SEC = 300.0

# L3 signal cache stores the Kuwait market tier classification per symbol.
_KSE_MARKET_TIER_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "l3_signals"

# Some symbols never got a tier persisted in the cache because the signal run
# stopped early on insufficient history. These fallback labels come from the
# backend's generated eligibility reports for the same Kuwait universe.
_KSE_MARKET_TIER_FALLBACKS: dict[str, str] = {
    "ALFTAQA": "PREMIER",
    "BKIKWT": "MAIN",
    "TROLLEY": "PREMIER",
}


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


def _market_info_cache_candidates() -> list[Path]:
    candidates: list[Path] = []
    settings = get_settings()

    env_path = (
        (settings.TICKERCHART_MARKET_INFO_PATH or "").strip()
        or os.getenv("TICKERCHART_MARKET_INFO_PATH", "").strip()
    )
    if env_path:
        candidates.append(Path(env_path))

    if os.name == "nt":
        local_appdata = os.getenv("LOCALAPPDATA", "").strip()
        if local_appdata:
            candidates.append(Path(local_appdata) / "UniTicker" / "TCLive" / "Cache" / "MarketInfo.json")
        candidates.append(Path.home() / "AppData" / "Local" / "UniTicker" / "TCLive" / "Cache" / "MarketInfo.json")
    else:
        candidates.append(Path("/var/lib/tickerchart/cache/MarketInfo.json"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _load_kse_market_tiers() -> dict[str, str]:
    global _kse_market_tier_cache

    if _kse_market_tier_cache is not None:
        return _kse_market_tier_cache

    tiers: dict[str, str] = {}
    if _KSE_MARKET_TIER_CACHE_DIR.is_dir():
        for cache_file in _KSE_MARKET_TIER_CACHE_DIR.glob("*.json"):
            try:
                with cache_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, ValueError) as exc:
                logger.debug("Failed to read market tier cache %s: %s", cache_file, exc)
                continue

            symbol = str(payload.get("ticker") or cache_file.stem).strip().upper()
            market_tier = str(payload.get("market_tier") or "").strip().upper()
            if symbol and market_tier in {"PREMIER", "MAIN"}:
                tiers[symbol] = market_tier

    for symbol, market_tier in _KSE_MARKET_TIER_FALLBACKS.items():
        tiers.setdefault(symbol, market_tier)

    _kse_market_tier_cache = tiers
    return tiers


def _parse_company_map(payload: Any) -> dict[tuple[str, str], int]:
    companies = None
    if isinstance(payload, dict):
        companies = (payload.get("COMPANIES") or {}).get("VALUES")
    if not isinstance(companies, dict):
        return {}

    mapping: dict[tuple[str, str], int] = {}
    for company_id, row in companies.items():
        if not isinstance(row, list) or len(row) < 2:
            continue
        ticker = str(row[0] or "").strip().upper()
        market = str(row[1] or "").strip().upper()
        if not ticker or not market:
            continue
        try:
            mapping[(ticker, market)] = int(company_id)
        except (TypeError, ValueError):
            continue
    return mapping


def _load_company_map_from_disk() -> dict[tuple[str, str], int]:
    for path in _market_info_cache_candidates():
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            logger.debug("Failed to read TickerChart market info cache %s: %s", path, exc)
            continue

        mapping = _parse_company_map(payload)
        if mapping:
            logger.info("Loaded TickerChart market info cache from %s", path)
            return mapping

    return {}


async def _fetch_company_map_remote() -> dict[tuple[str, str], int]:
    url = f"https://{_LOGIN_HOST}{_DESKTOP_MARKET_INFO_PATH}"
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": _USER_AGENT,
                    "Accept": "application/json, text/plain, */*",
                },
            )
    except httpx.HTTPError as exc:
        logger.debug("TickerChart market info request failed: %s", exc)
        return {}

    if resp.status_code == 403:
        logger.debug("TickerChart market info endpoint returned 403")
        return {}

    try:
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("TickerChart market info parse failed: %s", exc)
        return {}

    return _parse_company_map(payload)


async def _get_company_map() -> dict[tuple[str, str], int]:
    now = time.time()
    cached_entries = _company_id_cache.get("entries")
    if isinstance(cached_entries, dict) and float(_company_id_cache.get("expires", 0)) > now:
        return cached_entries

    entries = await _fetch_company_map_remote()
    if not entries:
        entries = _load_company_map_from_disk()

    _company_id_cache["entries"] = entries
    _company_id_cache["expires"] = now + (12 * 3600 if entries else 300)
    return entries


async def resolve_company_id(base_symbol: str, market_abb: str) -> Optional[int]:
    """Resolve a TickerChart company id from symbol + market abbreviation."""
    base = (base_symbol or "").strip().upper()
    market = (market_abb or "").strip().upper()
    if not base or not market:
        return None

    mapping = await _get_company_map()
    return mapping.get((base, market))


def _resolve_company_id_sync(base_symbol: str, market_abb: str) -> Optional[int]:
    """Resolve company id without requiring an async context."""
    base = (base_symbol or "").strip().upper()
    market = (market_abb or "").strip().upper()
    if not base or not market:
        return None

    now = time.time()
    cached_entries = _company_id_cache.get("entries")
    if not isinstance(cached_entries, dict) or float(_company_id_cache.get("expires", 0.0)) <= now:
        entries = _load_company_map_from_disk()
        _company_id_cache["entries"] = entries
        _company_id_cache["expires"] = now + (12 * 3600 if entries else 300)
        cached_entries = entries

    if not isinstance(cached_entries, dict):
        return None

    company_id = cached_entries.get((base, market))
    try:
        return int(company_id) if company_id is not None else None
    except (TypeError, ValueError):
        return None


def _extract_latest_factset_eps(payload: Any) -> Optional[float]:
    """Extract latest positive EPS value from FactSet payload shapes."""
    if isinstance(payload, dict):
        best_dt: Optional[date] = None
        best_val: Optional[float] = None

        for raw_dt, raw_val in payload.items():
            val = _coerce_float(raw_val)
            if val is None or val <= 0:
                continue

            dt: Optional[date] = None
            try:
                dt = datetime.fromisoformat(str(raw_dt)).date()
            except (TypeError, ValueError):
                dt = None

            if best_val is None:
                best_val = val
                best_dt = dt
                continue

            if dt is not None and (best_dt is None or dt > best_dt):
                best_val = val
                best_dt = dt

        return best_val

    if isinstance(payload, list):
        # Defensive parser for list/object payload variants.
        best_dt: Optional[date] = None
        best_val: Optional[float] = None
        for row in payload:
            if not isinstance(row, dict):
                continue
            fields = row.get("fields")
            if not isinstance(fields, dict):
                continue
            val = _coerce_float(fields.get("ff_eps_basic"))
            if val is None or val <= 0:
                continue

            dt: Optional[date] = None
            raw_dt = row.get("date")
            try:
                dt = datetime.fromisoformat(str(raw_dt)).date()
            except (TypeError, ValueError):
                dt = None

            if best_val is None:
                best_val = val
                best_dt = dt
                continue

            if dt is not None and (best_dt is None or dt > best_dt):
                best_val = val
                best_dt = dt

        return best_val

    return None


def fetch_factset_ltm_eps(base_symbol: str, market_abb: str, period: int = 25) -> Optional[float]:
    """Fetch latest TTM EPS from TickerChart FactSet feed through cacheserver."""
    sym = (base_symbol or "").strip().upper()
    mkt = (market_abb or "").strip().upper()
    if not sym or not mkt:
        return None

    cache_key = (sym, mkt)
    now = time.time()
    cached = _factset_eps_cache.get(cache_key)
    if isinstance(cached, dict) and float(cached.get("expires", 0.0)) > now:
        val = _coerce_float(cached.get("value"))
        return val if val is not None and val > 0 else None

    company_id = _resolve_company_id_sync(sym, mkt)
    if company_id is None:
        _factset_eps_cache[cache_key] = {"value": None, "expires": now + _FACTSET_EPS_MISS_TTL_SEC}
        return None

    path = f"/factset-feed/financial-field/company/{company_id}"
    query_pairs = [
        ("field", "ff_eps_basic"),
        ("period-type", "ltm"),
        ("period", str(max(1, int(period)))),
    ] + _common_params()
    final_qs, _ = _sign(path, query_pairs)

    factset_url = f"https://factset.tickerchart.net{path}?{final_qs}"
    cache_url = "https://cacheserver.tickerchart.net/"

    eps_val: Optional[float] = None
    try:
        resp = httpx.get(
            cache_url,
            params={"url": factset_url},
            headers={
                "Accept": "application/json, text/json, text/x-json, text/javascript, application/xml, text/xml",
                "User-Agent": _USER_AGENT,
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        eps_val = _extract_latest_factset_eps(resp.json())
    except Exception as exc:
        logger.debug("FactSet LTM EPS fetch failed for %s.%s: %s", sym, mkt, exc)

    _factset_eps_cache[cache_key] = {
        "value": eps_val,
        "expires": now + (_FACTSET_EPS_TTL_SEC if eps_val is not None else _FACTSET_EPS_MISS_TTL_SEC),
    }
    return eps_val


def _extract_indicator_values(payload: Any) -> list[float]:
    tracked_keys = {
        "value",
        "indicatorvalue",
        "datavalue",
        "amount",
        "actual",
        "actualvalue",
        "result",
        "y",
    }
    values: list[float] = []

    def visit(node: Any, parent_key: Optional[str] = None) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).strip().lower()
                if lowered in tracked_keys:
                    numeric = _coerce_float(value)
                    if numeric is not None:
                        values.append(numeric)
                        continue
                visit(value, lowered)
            return

        if isinstance(node, list):
            for item in node:
                visit(item, parent_key)
            return

        if parent_key in tracked_keys:
            numeric = _coerce_float(node)
            if numeric is not None:
                values.append(numeric)

    visit(payload)
    return values


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
    # API returns the token with "TcToken" prefix already included.
    # Strip it so callers can safely prepend it via f"TcToken{token}".
    if isinstance(token, str) and token.startswith("TcToken"):
        token = token[len("TcToken"):]
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


async def fetch_ltm_eps(base_symbol: str, market_abb: str) -> Optional[float]:
    """Fetch TickerChart's ff_eps_basic(ltm) value for a symbol."""
    company_id = await resolve_company_id(base_symbol, market_abb)
    if company_id is None:
        return None

    async def _do_request(token: str) -> httpx.Response:
        qs_pairs = [
            ("companyID", str(company_id)),
            ("financialIndicatorId", "ff_eps_basic"),
            ("reportRange", "ltm"),
        ] + _common_params()
        final_qs, _ = _sign(_FINANCIAL_FIELD_PATH, qs_pairs)
        url = f"https://{_LOGIN_HOST}{_FINANCIAL_FIELD_PATH}?{final_qs}"
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            return await client.get(
                url,
                headers={
                    "User-Agent": _USER_AGENT,
                    "Accept": "application/json, text/plain, */*",
                    "Authorization": f"TcToken{token}",
                },
            )

    token = await _get_token()
    resp = await _do_request(token)
    if resp.status_code in (401, 403):
        logger.info("TickerChart token rejected for LTM EPS fetch, re-logging in")
        _token_cache["token"] = None
        token = await _get_token()
        resp = await _do_request(token)
    resp.raise_for_status()

    try:
        payload: Any = resp.json()
    except ValueError:
        return _coerce_float(resp.text)

    if isinstance(payload, dict) and payload.get("success") is False:
        logger.warning(
            "TickerChart LTM EPS request failed for %s.%s: %s",
            base_symbol,
            market_abb,
            payload,
        )
        return None

    data = payload.get("response") if isinstance(payload, dict) and "response" in payload else payload
    values = _extract_indicator_values(data)
    for value in reversed(values):
        if abs(value) > 1e-12:
            return value
    return None


# ── FlatFiles PE reader ──────────────────────────────────────────────
# TickerChart caches FactSet PE indicator data in local binary files.
# Each 40-byte record: date(float64 OLE) + open(f32) + high(f32) + low(f32) + close(f32) + 16 bytes padding.
# Multiple records per day represent different PE variants (basic/diluted/normalized).
# The close PE of the first record per day is used as the primary LTM PE value.
#
# Mapping: (base_symbol, market_abb) → FactSet file ID used in the FlatFiles directory name.
# New mappings are discovered by watching which .dat file is modified when TC loads a company chart.
_PE_FLATFILES_MAP: dict[tuple[str, str], int] = {
    ("NBK", "KSE"): 10315,   # confirmed: file modified when NBK chart was opened 2025-05-18
    ("KFH", "KSE"): 13470,   # confirmed: closest match to snapshot PE=21.25 on 2024-04-08 (delta=0.33)
}

# Symbols for which auto-discovery has already been attempted and failed; avoids
# re-scanning the flatfiles directory on every request for unknown stocks.
_PE_FLATFILES_FAILED_SYMBOLS: set[tuple[str, str]] = set()

# Top-level FlatFiles directories that may contain PE data, ordered by data recency preference.
# aa3ba → longest history (2023-present); a22ce → shorter window; df1650 → may have older data.
_PE_FLATFILES_PARENT_DIRS = [
    "aa3ba405d27847645e3d",
    "a22ce12861bfed7af141",
    "df1650f22ae3fee9d671",
]
_PE_FLATFILES_SUBDIR = "320800594fe439050088"

def _resolve_flatfiles_base() -> Path:
    settings = get_settings()
    configured_path = (
        (settings.TICKERCHART_FLATFILES_PATH or "").strip()
        or os.getenv("TC_FLATFILES_PATH", "").strip()
        or os.getenv("TICKERCHART_FLATFILES_PATH", "").strip()
    )
    if configured_path:
        return Path(configured_path)

    if os.name == "nt":
        local_appdata = os.getenv("LOCALAPPDATA", "").strip()
        if local_appdata:
            return Path(local_appdata) / "UniTicker" / "TCLive" / "FlatFiles"
        return Path.home() / "AppData" / "Local" / "UniTicker" / "TCLive" / "FlatFiles"

    # Linux/macOS default. Override with TICKERCHART_FLATFILES_PATH or TC_FLATFILES_PATH.
    return Path("/var/lib/tickerchart/flatfiles")


_TC_FLATFILES_BASE = _resolve_flatfiles_base()


def _ole_date_to_python(ole_val: float) -> Optional[date]:
    """Convert OLE Automation date (days since 1899-12-30) to Python date."""
    import math
    try:
        if math.isnan(ole_val) or math.isinf(ole_val):
            return None
        d = date(1899, 12, 30) + __import__("datetime").timedelta(days=int(ole_val))
        if 1990 <= d.year <= 2040:
            return d
    except (OverflowError, ValueError, OSError):
        pass
    return None


def _load_quotes_snapshot_entries(snapshot_path: Path) -> Optional[dict]:
    """Load QuotesSnapShot.bin once and reuse it for repeated symbol lookups."""
    now = time.time()

    cached_entries = _quotes_snapshot_cache.get("entries")
    cached_path = _quotes_snapshot_cache.get("path")
    if (
        isinstance(cached_entries, dict)
        and cached_path == str(snapshot_path)
        and float(_quotes_snapshot_cache.get("expires", 0.0)) > now
    ):
        return cached_entries

    if not snapshot_path.exists():
        return None

    # Revalidate by mtime first to avoid repeatedly parsing the same JSON.
    try:
        current_mtime = snapshot_path.stat().st_mtime
    except OSError:
        current_mtime = None

    if (
        isinstance(cached_entries, dict)
        and cached_path == str(snapshot_path)
        and _quotes_snapshot_cache.get("mtime") == current_mtime
    ):
        _quotes_snapshot_cache["expires"] = now + 5.0
        return cached_entries

    try:
        payload = json.loads(snapshot_path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    _quotes_snapshot_cache["entries"] = payload
    _quotes_snapshot_cache["path"] = str(snapshot_path)
    _quotes_snapshot_cache["mtime"] = current_mtime
    _quotes_snapshot_cache["expires"] = now + 5.0
    return payload


def _read_quotes_snapshot_entry(base_symbol: str, market_abb: str) -> Optional[dict]:
    """Return the raw QuotesSnapShot.bin entry for a stock, or None."""
    snapshot_path = _TC_FLATFILES_BASE.parent / "Cache" / "QuotesSnapShot.bin"
    data = _load_quotes_snapshot_entries(snapshot_path)
    if not isinstance(data, dict):
        return None
    return data.get(f"QO.{base_symbol.upper()}.{market_abb.upper()}")


def _read_quotes_snapshot_pe(base_symbol: str, market_abb: str) -> Optional[float]:
    """
    Read the live P/E for a stock from TickerChart's QuotesSnapShot.bin.

    The snapshot is a JSON file keyed by "QO.{SYMBOL}.{MARKET}" with a ``p_e``
    field that stores the ratio multiplied by 1000 (e.g. 21250 → 21.25×).
    Returns None when the file is absent, the key is missing, or PE <= 0.
    """
    entry = _read_quotes_snapshot_entry(base_symbol, market_abb)
    if not entry:
        return None
    try:
        pe_val = float(entry["p_e"]) / 1000.0
    except (KeyError, TypeError, ValueError):
        return None
    return pe_val if pe_val > 0 else None


def read_quotes_snapshot_pe(base_symbol: str, market_abb: str) -> Optional[float]:
    """Public wrapper for the live QuotesSnapShot P/E reader."""
    return _read_quotes_snapshot_pe(base_symbol, market_abb)


def read_quotes_snapshot_ltm_eps(base_symbol: str, market_abb: str, price_divisor: float = 1000.0) -> Optional[float]:
    """
    Read the current LTM EPS from TickerChart's QuotesSnapShot.bin.

    Tries the ``financial_ff$eps$basic_ltm_0`` field first; falls back to
    deriving EPS from live price (``last`` in fils) ÷ P/E (``p_e`` ÷ 1000).

    ``price_divisor`` converts the raw ``last`` field to the stock's reporting
    currency unit (1000 for KSE fils→KWD, 1.0 for USD already in dollars).

    Returns None when data is unavailable or EPS ≤ 0.
    """
    entry = _read_quotes_snapshot_entry(base_symbol, market_abb)
    if not entry:
        return None

    # Prefer the direct FactSet LTM EPS field
    eps_raw = entry.get("financial_ff$eps$basic_ltm_0")
    if eps_raw is not None:
        try:
            eps = float(eps_raw)
            if eps > 0:
                return eps
        except (TypeError, ValueError):
            pass

    # Fallback: derive from live price and P/E
    try:
        pe = float(entry["p_e"]) / 1000.0
        price = float(entry["last"]) / price_divisor
        if pe > 0 and price > 0:
            return price / pe
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass

    return None


def read_quotes_snapshot_bvps(base_symbol: str, market_abb: str, price_divisor: float = 1000.0) -> Optional[float]:
    """
    Read the current Book Value Per Share from TickerChart's QuotesSnapShot.bin.

    Tries direct ``bookvalue`` first, then derives BVPS from
    ``last`` / ``price_book`` (where ``price_book`` is scaled by 1000).
    Returns None when no usable value is available.
    """
    entry = _read_quotes_snapshot_entry(base_symbol, market_abb)
    if not entry:
        return None

    # Most Kuwait entries include bookvalue directly (per-share, same currency unit).
    for key in ("bookvalue", "book_value", "bookValue"):
        raw_val = entry.get(key)
        if raw_val is None:
            continue
        try:
            bvps_direct = float(raw_val)
        except (TypeError, ValueError):
            continue
        if bvps_direct > 0:
            return bvps_direct

    # Fallback derivation: BVPS = price / P/B.
    # Snapshot stores `last` as fils for KSE and `price_book` as ratio * 1000.
    try:
        price = float(entry["last"]) / price_divisor
        price_to_book = float(entry["price_book"]) / 1000.0
        if price > 0 and price_to_book > 0:
            bvps = price / price_to_book
            if bvps > 0:
                return bvps
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass

    return None


def auto_discover_pe_flatfile(base_symbol: str, market_abb: str) -> Optional[int]:
    """
    Auto-discover the FactSet flatfile ID for a stock.

    Algorithm:
    1. Read the stock's current PE from QuotesSnapShot.bin.
    2. Scan every .dat file in the PE subdirectory.
    3. For each file look at the last 20 records; keep those dated within 30 days.
    4. Pick the file whose most-recent PE value is closest to the snapshot PE
       AND within ±0.5 of it.
    5. On success register the mapping and return the ID; otherwise return None.

    The result (success or failure) is memoised so the scan runs at most once
    per (symbol, market) pair per process lifetime.
    """
    import struct

    live_pe = _read_quotes_snapshot_pe(base_symbol, market_abb)
    if live_pe is None:
        logger.debug("QuotesSnapshot: no PE for %s.%s", base_symbol, market_abb)
        return None

    today = date.today()
    known_ids = set(_PE_FLATFILES_MAP.values())
    best_id: Optional[int] = None
    best_delta = float("inf")

    for parent_dir in _PE_FLATFILES_PARENT_DIRS:
        pe_dir = _TC_FLATFILES_BASE / parent_dir / _PE_FLATFILES_SUBDIR
        if not pe_dir.exists():
            continue
        for dat_file in sorted(pe_dir.glob("*.dat")):
            try:
                factset_id = int(dat_file.stem)
            except ValueError:
                continue
            if factset_id in known_ids:
                continue  # already claimed by another symbol
            try:
                raw = dat_file.read_bytes()
            except OSError:
                continue
            n = len(raw) // 40
            if n == 0:
                continue
            # Inspect last 20 records only — avoids reading entire large files
            recent_pe: Optional[float] = None
            for i in range(max(0, n - 20), n):
                try:
                    ole_val, _, _, _, close_pe = struct.unpack_from("<dffff", raw, i * 40)
                except struct.error:
                    break
                d = _ole_date_to_python(ole_val)
                if d is None or (today - d).days > 30:
                    continue
                if 3.0 < close_pe < 200.0:
                    recent_pe = close_pe
            if recent_pe is None:
                continue
            delta = abs(recent_pe - live_pe)
            if delta < best_delta:
                best_delta = delta
                best_id = factset_id

    if best_id is not None and best_delta < 0.5:
        register_pe_flatfiles_mapping(base_symbol, market_abb, best_id)
        logger.info(
            "Auto-discovered PE flatfile for %s.%s: id=%d "
            "(snapshot_pe=%.2f, file_pe≈%.2f, delta=%.3f)",
            base_symbol, market_abb, best_id,
            live_pe, live_pe - best_delta, best_delta,
        )
        return best_id

    logger.debug(
        "PE flatfile auto-discovery failed for %s.%s "
        "(best_delta=%.3f, threshold=0.5)",
        base_symbol, market_abb, best_delta,
    )
    return None


def fetch_pe_from_flatfiles(
    base_symbol: str,
    market_abb: str,
) -> dict[date, float]:
    """
    Read pre-computed daily P/E LTM values from TickerChart's local FlatFiles cache.

    Returns a dict mapping trading date → PE close value.
    Returns an empty dict if no mapping is known or no cache file exists.

    The cache is populated whenever the user views the PE indicator for a
    company in TickerChart; the data reflects FactSet's "Price to Earnings
    (Last Twelve Months)" series.
    """
    import struct

    sym_upper = base_symbol.upper()
    mkt_upper = market_abb.upper()
    factset_id = _PE_FLATFILES_MAP.get((sym_upper, mkt_upper))
    if factset_id is None:
        # Symbol-only fallback: handle cases where the same company is stored with
        # a different exchange label (e.g. KFH saved as exchange="US" instead of "KSE").
        for (map_sym, map_mkt), fid in _PE_FLATFILES_MAP.items():
            if map_sym == sym_upper:
                factset_id = fid
                logger.debug(
                    "FlatFiles PE market fallback: %s.%s resolved via %s.%s (id=%d)",
                    sym_upper, mkt_upper, map_sym, map_mkt, fid,
                )
                break
    if factset_id is None:
        # Auto-discovery: cross-reference QuotesSnapshot live PE with flatfile scan.
        # Skip if we already attempted and failed for this (symbol, market) pair.
        sym_mkt_key = (sym_upper, mkt_upper)
        if sym_mkt_key not in _PE_FLATFILES_FAILED_SYMBOLS:
            discovered = auto_discover_pe_flatfile(base_symbol, market_abb)
            if discovered is not None:
                factset_id = discovered
            else:
                _PE_FLATFILES_FAILED_SYMBOLS.add(sym_mkt_key)
    if factset_id is None:
        logger.debug("No FlatFiles PE mapping for %s.%s (auto-discovery failed)", base_symbol, market_abb)
        return {}

    daily_pe: dict[date, tuple[float, float]] = {}  # (high_pe, low_pe) per day

    for parent_dir in _PE_FLATFILES_PARENT_DIRS:
        dat_path = _TC_FLATFILES_BASE / parent_dir / _PE_FLATFILES_SUBDIR / f"{factset_id}.dat"
        if not dat_path.exists():
            continue
        try:
            raw = dat_path.read_bytes()
        except OSError as exc:
            logger.warning("Cannot read PE FlatFile %s: %s", dat_path, exc)
            continue

        n_records = len(raw) // 40
        for i in range(n_records):
            try:
                ole_val, _o, _h, _l, close_pe = struct.unpack_from("<dffff", raw, i * 40)
            except struct.error:
                break

            trading_date = _ole_date_to_python(ole_val)
            if trading_date is None:
                continue

            # Accept only plausible PE values (3–200 avoids noise / padding zeros)
            if not (3.0 < close_pe < 200.0):
                continue

            # Keep first record seen for each date; store as (close_pe, close_pe) tuple
            # — the OHLC fields in the flatfile are intraday PE variants clustered
            # within ~0.1× of each other, so close_pe is the canonical daily PE.
            if trading_date not in daily_pe:
                daily_pe[trading_date] = (close_pe, close_pe)

    logger.info(
        "FlatFiles PE for %s.%s: %d daily records loaded (factset_id=%d)",
        base_symbol,
        market_abb,
        len(daily_pe),
        factset_id,
    )
    return daily_pe


def register_pe_flatfiles_mapping(base_symbol: str, market_abb: str, factset_id: int) -> None:
    """
    Register a new (symbol, market) → FactSet file ID mapping at runtime.
    Call this when a new mapping is discovered via the file watcher.
    """
    _PE_FLATFILES_MAP[(base_symbol.upper(), market_abb.upper())] = factset_id
    logger.info(
        "Registered FlatFiles PE mapping: %s.%s → %d", base_symbol, market_abb, factset_id
    )


# ── OHLCV ────────────────────────────────────────────────────────────
def _pick_period(from_d: Optional[date], to_d: Optional[date], interval: str = "day") -> str:
    """Pick a TickerChart `period` that preserves enough candles for the use case."""
    if from_d is None or to_d is None:
        return "5years"
    days = (to_d - from_d).days

    # TickerChart's `period=1week` can collapse day-interval responses to a
    # single candle, which breaks day-over-day calculations (change, movers).
    # Use at least a monthly bucket for short day-range requests.
    if interval == "day" and days <= 7:
        return "1month"

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


async def fetch_ohlcv(
    base_symbol: str,
    market_abb: str,
    from_d: Optional[date] = None,
    to_d: Optional[date] = None,
    interval: str = "day",
    client: Optional[httpx.AsyncClient] = None,
) -> list[dict]:
    """Return list of EODHD-shaped rows: {date, open, high, low, close, volume}.

    Re-tries transient failures with finite bounded attempts.
    """
    host = _MARKET_HOST.get(market_abb)
    if host is None:
        raise ValueError(f"Unsupported market: {market_abb}")

    period = _pick_period(from_d, to_d, interval=interval)
    path = "/tcdata/ondemandDataLoader.php"
    user_name = (get_settings().TICKERCHART_USERNAME or "").strip()

    qs_pairs = [
        ("user_name", user_name),
        ("language", "ENGLISH"),
        ("symbol", f"{base_symbol}.{market_abb}"),
        ("interval", interval),
        ("period", period),
    ] + _common_params()
    final_qs, _ = _sign(path, qs_pairs)
    url = f"https://{host}{path}?{final_qs}"

    resp: Optional[httpx.Response] = None
    for attempt in range(1, _TC_FETCH_MAX_ATTEMPTS + 1):
        try:
            if client is None:
                async with httpx.AsyncClient(timeout=_TC_HTTP_TIMEOUT, follow_redirects=True) as request_client:
                    resp = await request_client.get(
                        url,
                        headers={"User-Agent": _USER_AGENT},
                    )
            else:
                resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()
            break
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            if attempt >= _TC_FETCH_MAX_ATTEMPTS:
                raise
            logger.warning(
                "TickerChart OHLCV retry %s/%s for %s.%s: %s",
                attempt,
                _TC_FETCH_MAX_ATTEMPTS,
                base_symbol,
                market_abb,
                exc,
            )
            await asyncio.sleep(_TC_FETCH_RETRY_BACKOFF_SEC * attempt)

    if resp is None:
        return []

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
            row: dict = {
                "date": d,
                "open": float(parts[1] or 0),
                "high": float(parts[2] or 0),
                "low": float(parts[3] or 0),
                "close": float(parts[4] or 0),
                "volume": float(parts[5] or 0),
            }
            # Capture optional value and trades columns when present
            if len(parts) > 6 and parts[6].strip():
                try:
                    row["value"] = float(parts[6].strip() or 0)
                except ValueError:
                    pass
            if len(parts) > 7 and parts[7].strip():
                try:
                    row["trades"] = int(float(parts[7].strip() or 0))
                except ValueError:
                    pass
            out.append(row)
        except ValueError:
            continue
    
    # Strip phantom rows: ex-dividend entries with invalid price fields.
    # Keep rows where high/low/close are all present and positive.
    # Some feeds legitimately provide open=0 for older candles; we normalize
    # those opens below so historical coverage is preserved.
    def _has_price(r: dict) -> bool:
        return r["high"] > 0 and r["low"] > 0 and r["close"] > 0

    out = [r for r in out if _has_price(r)]

    # Deduplicate by date — ex-dividend dates sometimes still appear twice
    # with real prices. Keep the entry with the highest volume.
    out.sort(key=lambda r: r["date"])
    deduped: dict[str, dict] = {}
    for row in out:
        d = row["date"]
        if d not in deduped:
            deduped[d] = row
        else:
            if row["volume"] > deduped[d]["volume"]:
                deduped[d] = row

    cleaned = sorted(deduped.values(), key=lambda r: r["date"])

    # Normalize zero opens to previous close (or same-bar close for first row).
    # This keeps candles usable for charting/indicators without discarding rows.
    prev_close: Optional[float] = None
    for row in cleaned:
        if row["open"] <= 0:
            if prev_close is not None and prev_close > 0:
                row["open"] = prev_close
            else:
                row["open"] = row["close"]
        if row["close"] > 0:
            prev_close = row["close"]

    return cleaned


# ── Order Book / Market Depth ────────────────────────────────────────
async def fetch_order_book(
    base_symbol: str,
    market_abb: str,
    depth: int = 20,
) -> dict:
    """Fetch real-time order book (market depth) snapshot.
    
    Returns format matching TickerChart Market Depth view:
    {
        "symbol": "IFAHR",
        "market": "KSE",
        "timestamp": "2026-05-04T21:35:00Z",
        "bids": [
            {"price": 890.0, "volume": 5000},
            {"price": 886.0, "volume": 5000},
            ...
        ],
        "asks": [
            {"price": 908.0, "volume": 3500},
            {"price": 909.0, "volume": 849},
            ...
        ],
        "total_bid_volume": 29010,
        "total_ask_volume": 182254
    }
    """
    host = _MARKET_HOST.get(market_abb)
    if host is None:
        raise ValueError(f"Unsupported market: {market_abb}")

    # Try multiple possible endpoints for market depth
    endpoints = [
        "/tcdata/marketdepth.php",
        "/tcdata/orderbook.php",
        "/tcdata/level2.php",
        "/m/v2/marketdepth",
    ]
    
    user_name = (get_settings().TICKERCHART_USERNAME or "").strip()
    token = await _get_token()
    
    for path in endpoints:
        try:
            qs_pairs = [
                ("user_name", user_name),
                ("symbol", f"{base_symbol}.{market_abb}"),
                ("depth", str(depth)),
                ("language", "ENGLISH"),
            ] + _common_params()
            final_qs, _ = _sign(path, qs_pairs)
            url = f"https://{host}{path}?{final_qs}"
            
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.get(
                    url,
                    headers={
                        "User-Agent": _USER_AGENT,
                        "Authorization": f"TcToken{token}",
                    },
                )
                
                if resp.status_code == 200:
                    # Found working endpoint
                    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else None
                    if data:
                        # Parse TickerChart format to our standard format
                        return _parse_order_book_response(data, base_symbol, market_abb)
                    
                    # If CSV/text format, parse it
                    text = resp.text
                    if text and "BID" in text.upper() or "ASK" in text.upper():
                        return _parse_order_book_csv(text, base_symbol, market_abb)
                        
        except Exception as e:
            logger.debug(f"Endpoint {path} failed: {e}")
            continue
    
    # If all endpoints fail, raise error
    raise RuntimeError(
        f"No working order book endpoint found for {base_symbol}.{market_abb}. "
        "Market depth may not be available for this symbol or market."
    )


def _parse_order_book_response(data: dict, symbol: str, market: str) -> dict:
    """Parse JSON response from TickerChart market depth API."""
    from datetime import datetime
    
    # Try multiple possible JSON structures
    bids_raw = data.get("bids", data.get("bid", data.get("buy", [])))
    asks_raw = data.get("asks", data.get("ask", data.get("sell", [])))
    
    bids = []
    asks = []
    
    # Parse bids (descending by price)
    for item in bids_raw:
        if isinstance(item, dict):
            price = float(item.get("price", item.get("p", 0)))
            volume = float(item.get("volume", item.get("v", item.get("qty", 0))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            price = float(item[0])
            volume = float(item[1])
        else:
            continue
        
        if price > 0 and volume > 0:
            bids.append({"price": price, "volume": volume})
    
    # Parse asks (ascending by price)
    for item in asks_raw:
        if isinstance(item, dict):
            price = float(item.get("price", item.get("p", 0)))
            volume = float(item.get("volume", item.get("v", item.get("qty", 0))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            price = float(item[0])
            volume = float(item[1])
        else:
            continue
        
        if price > 0 and volume > 0:
            asks.append({"price": price, "volume": volume})
    
    # Sort bids descending (highest price first)
    bids.sort(key=lambda x: x["price"], reverse=True)
    # Sort asks ascending (lowest price first)
    asks.sort(key=lambda x: x["price"])
    
    total_bid = sum(b["volume"] for b in bids)
    total_ask = sum(a["volume"] for a in asks)
    
    return {
        "symbol": symbol,
        "market": market,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "bids": bids,
        "asks": asks,
        "total_bid_volume": total_bid,
        "total_ask_volume": total_ask,
    }


def _parse_order_book_csv(text: str, symbol: str, market: str) -> dict:
    """Parse CSV/text format order book response."""
    from datetime import datetime
    
    bids = []
    asks = []
    current_section = None
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        
        upper = line.upper()
        if "BID" in upper:
            current_section = "bid"
            continue
        elif "ASK" in upper or "OFFER" in upper:
            current_section = "ask"
            continue
        
        # Parse price,volume lines
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                price = float(parts[0].strip())
                volume = float(parts[1].strip())
                
                if price > 0 and volume > 0:
                    if current_section == "bid":
                        bids.append({"price": price, "volume": volume})
                    elif current_section == "ask":
                        asks.append({"price": price, "volume": volume})
            except ValueError:
                continue
    
    # Sort bids descending, asks ascending
    bids.sort(key=lambda x: x["price"], reverse=True)
    asks.sort(key=lambda x: x["price"])
    
    total_bid = sum(b["volume"] for b in bids)
    total_ask = sum(a["volume"] for a in asks)
    
    return {
        "symbol": symbol,
        "market": market,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "bids": bids,
        "asks": asks,
        "total_bid_volume": total_bid,
        "total_ask_volume": total_ask,
    }


# ── KSE Market Snapshot ──────────────────────────────────────────────

# Confirmed working TickerChart symbols for the Kuwait market indices shown on
# the Boursa Kuwait market page.
# Each entry: (display_name, tc_base_symbol, tc_market_abb)
_KSE_INDEX_CANDIDATES: list[tuple[str, str, str]] = [
    ("Premier Market", "BKP", "KSE"),
    ("BK Main 50", "BKM50", "KSE"),
    ("Main Market", "BKM", "KSE"),
    ("All-Share", "BKA", "KSE"),
]


async def _fetch_index_row(
    display_name: str,
    tc_symbol: str,
    market_abb: str,
    from_d: date,
    to_d: date,
) -> Optional[dict]:
    """Attempt to fetch one index via TickerChart OHLCV. Returns None on any failure."""
    try:
        rows = await fetch_ohlcv(tc_symbol, market_abb, from_d=from_d, to_d=to_d, interval="day")
        if not rows:
            return None
        rows_sorted = sorted(rows, key=lambda r: r["date"])
        last_row = rows_sorted[-1]
        prev_row = rows_sorted[-2] if len(rows_sorted) >= 2 else None
        last = float(last_row.get("close") or 0)
        if last == 0:
            return None
        change: Optional[float] = None
        change_pct: Optional[float] = None
        if prev_row:
            prev = float(prev_row.get("close") or 0)
            if prev > 0:
                change = last - prev
                change_pct = round((change / prev) * 100, 4)
        return {
            "name": display_name,
            "value": last,
            "change": change,
            "changePercent": change_pct,
            "volume": float(last_row.get("volume") or 0) or None,
            "value_traded": float(last_row.get("value") or 0) or None,
            "trades": int(last_row.get("trades") or 0) or None,
        }
    except Exception as exc:
        logger.debug("Index fetch failed for %s.%s: %s", tc_symbol, market_abb, exc)
        return None


async def fetch_kse_market_snapshot(symbols: list[str], stock_name_map: dict[str, str]) -> dict:
    """
    Fetch a market-wide snapshot for Kuwait (KSE) from TickerChart.

    Concurrently fetches the last 7 calendar days of OHLCV for every supplied
    symbol, computes per-stock day changes from the last two trading candles,
    then aggregates gainers / losers / movers and total market statistics.

    Returns a dict in the same shape as the legacy Boursa Kuwait scraper.
    """
    import asyncio as _asyncio
    from datetime import timedelta

    today_d = date.today()
    from_d = today_d - timedelta(days=7)
    market_tiers = _load_kse_market_tiers()

    # ── Fetch all stocks concurrently over one shared connection pool ───
    # Creating one AsyncClient per symbol causes connection churn and can
    # queue requests behind timeout retries. A bounded shared pool preserves
    # concurrency without overwhelming the provider or local socket table.
    async with httpx.AsyncClient(
        timeout=_TC_HTTP_TIMEOUT,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
    ) as market_client:
        async def _safe_fetch(symbol: str) -> tuple[str, list[dict]]:
            try:
                rows = await fetch_ohlcv(
                    symbol,
                    "KSE",
                    from_d=from_d,
                    to_d=today_d,
                    interval="day",
                    client=market_client,
                )
                return symbol, rows
            except Exception as exc:
                logger.debug("Market snapshot: fetch failed for %s: %s", symbol, exc)
                return symbol, []

        stock_results = await _asyncio.gather(*[_safe_fetch(sym) for sym in symbols])

    # ── Fetch indices concurrently ───────────────────────────────────
    index_tasks = [
        _fetch_index_row(name, tc_sym, mkt, from_d, today_d)
        for name, tc_sym, mkt in _KSE_INDEX_CANDIDATES
    ]
    index_results = await _asyncio.gather(*index_tasks)
    indices = [r for r in index_results if r is not None]
    index_by_name = {str(item.get("name") or ""): item for item in indices}

    # ── Compute per-stock stats ──────────────────────────────────────
    stocks: list[dict] = []
    total_volume = 0.0
    total_value = 0.0
    total_trades = 0
    per_market_totals: dict[str, dict[str, float | int]] = {
        "PREMIER": {"volume": 0.0, "value_traded": 0.0, "trades": 0, "count": 0},
        "MAIN": {"volume": 0.0, "value_traded": 0.0, "trades": 0, "count": 0},
    }

    for symbol, rows in stock_results:
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["date"])
        today_row = rows_sorted[-1]
        prev_row = rows_sorted[-2] if len(rows_sorted) >= 2 else None

        last = float(today_row.get("close") or 0)
        volume = float(today_row.get("volume") or 0)
        # TickerChart reports KSE turnover in fils. Convert to KWD before
        # exposing it as `value_traded` in the market payload.
        value = float(today_row.get("value") or (last * volume)) / 1000.0
        trades = int(today_row.get("trades") or 0)

        if last == 0:
            continue  # skip stocks with no price data

        change: Optional[float] = None
        change_pct: Optional[float] = None
        if prev_row:
            prev = float(prev_row.get("close") or 0)
            if prev > 0:
                change = round(last - prev, 4)
                change_pct = round((change / prev) * 100, 4)

        total_volume += volume
        total_value += value
        total_trades += trades

        market_tier = market_tiers.get(symbol.strip().upper())
        if market_tier in per_market_totals:
            per_market_totals[market_tier]["volume"] += volume
            per_market_totals[market_tier]["value_traded"] += value
            per_market_totals[market_tier]["trades"] += trades
            per_market_totals[market_tier]["count"] += 1

        stocks.append({
            "symbol": symbol,
            "name": stock_name_map.get(symbol, symbol),
            "last": last,
            "change": change,
            "changePercent": change_pct,
            "volume": volume,
            "value": value,
        })

    # ── Classify movers ──────────────────────────────────────────────
    with_change = [s for s in stocks if s["changePercent"] is not None]
    gainers = [s for s in with_change if s["changePercent"] > 0]
    losers = [s for s in with_change if s["changePercent"] < 0]
    neutral_count = len(stocks) - len(gainers) - len(losers)

    top_gainers = sorted(gainers, key=lambda s: s["changePercent"], reverse=True)[:5]
    top_losers = sorted(losers, key=lambda s: s["changePercent"])[:5]
    top_value_list = sorted(stocks, key=lambda s: s["value"], reverse=True)[:10]

    def _to_mover(s: dict) -> dict:
        return {
            "symbol": s["symbol"],
            "last": s["last"],
            "change": s["change"],
            "changePercent": s["changePercent"],
            "volume": s["volume"],
        }

    def _to_per_market_summary(market_tier: str) -> dict:
        totals = per_market_totals[market_tier]
        if not totals["count"]:
            return {"volume": None, "value_traded": None, "trades": None, "market_cap": None}
        return {
            "volume": float(totals["volume"]),
            "value_traded": float(totals["value_traded"]),
            "trades": int(totals["trades"]) or None,
            "market_cap": None,
        }

    def _to_index_summary(index_name: str, fallback: dict) -> dict:
        index_row = index_by_name.get(index_name) or {}
        if index_row.get("volume") is None and index_row.get("value_traded") is None and index_row.get("trades") is None:
            return fallback
        return {
            "volume": index_row.get("volume"),
            "value_traded": index_row.get("value_traded"),
            "trades": index_row.get("trades"),
            "market_cap": None,
        }

    all_share_summary = _to_index_summary(
        "All-Share",
        {
            "volume": total_volume,
            "value_traded": total_value,
            "trades": total_trades or None,
            "market_cap": None,
        },
    )

    return {
        "indices": indices,
        "market_summary": {
            "volume": all_share_summary["volume"],
            "value_traded": all_share_summary["value_traded"],
            "trades": all_share_summary["trades"],
            "market_cap": all_share_summary["market_cap"],
            "gainers": len(gainers),
            "losers": len(losers),
            "neutral": neutral_count,
            "stock_gainers": len(gainers),
            "stock_losers": len(losers),
        },
        "premier_summary": _to_index_summary("Premier Market", _to_per_market_summary("PREMIER")),
        "main_summary": _to_index_summary("Main Market", _to_per_market_summary("MAIN")),
        "top_gainers": [_to_mover(s) for s in top_gainers],
        "top_losers": [_to_mover(s) for s in top_losers],
        "top_value": [_to_mover(s) for s in top_value_list],
        "sectors": [],
        "date": today_d.isoformat(),
        "status": "open",
    }
