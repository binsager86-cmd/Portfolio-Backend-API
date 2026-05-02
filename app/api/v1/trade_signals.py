"""
Trade Signals API — actionable buy/sell insights.

Currently implements F.Signals: P/E quarterly history + over/undervaluation
verdict for a chosen stock. Data source: stockanalysis.com (quarterly ratios
page) for both Kuwait (KWSE) and US tickers, with a yfinance fallback for
the live current P/E reading.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import httpx
from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.database import query_one
from app.core.security import TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trade-signals", tags=["Trade Signals"])

# [P2-4/B-6] TTL cache for P/E scrape results — 1 h TTL, max 256 symbol slots.
# Falls back to last known good value when upstream is temporarily unavailable.
_pe_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

_QUARTER_OF_MONTH = {
    1: "q1", 2: "q1", 3: "q1",
    4: "q2", 5: "q2", 6: "q2",
    7: "q3", 8: "q3", 9: "q3",
    10: "q4", 11: "q4", 12: "q4",
}

settings = get_settings()


# ── Scraping helpers ──────────────────────────────────────────────────


def _ratios_url(
    symbol: str,
    yf_ticker: Optional[str],
    exchange: Optional[str] = None,
    currency: Optional[str] = None,
) -> str:
    """Resolve the stockanalysis.com quarterly ratios URL for a symbol."""
    sym_upper = (symbol or "").upper()
    yf_upper = (yf_ticker or "").upper()
    ex_upper = (exchange or "").upper()
    cur_upper = (currency or "").upper()
    is_kwse = (
        sym_upper.endswith(".KW")
        or yf_upper.endswith(".KW")
        or ex_upper in {"KSE", "KWSE", "KUWAIT"}
        or cur_upper == "KWD"
    )
    base = re.sub(r"\.KW$", "", sym_upper)
    if is_kwse:
        return f"https://stockanalysis.com/quote/kwse/{base}/financials/ratios/?p=quarterly"
    return f"https://stockanalysis.com/stocks/{base.lower()}/financials/ratios/?p=quarterly"


def _statistics_url(
    symbol: str,
    yf_ticker: Optional[str],
    exchange: Optional[str] = None,
    currency: Optional[str] = None,
) -> str:
    sym_upper = (symbol or "").upper()
    yf_upper = (yf_ticker or "").upper()
    ex_upper = (exchange or "").upper()
    cur_upper = (currency or "").upper()
    is_kwse = (
        sym_upper.endswith(".KW")
        or yf_upper.endswith(".KW")
        or ex_upper in {"KSE", "KWSE", "KUWAIT"}
        or cur_upper == "KWD"
    )
    base = re.sub(r"\.KW$", "", sym_upper)
    if is_kwse:
        return f"https://stockanalysis.com/quote/kwse/{base}/statistics/"
    return f"https://stockanalysis.com/stocks/{base.lower()}/statistics/"


def _normalize_eod_symbol(symbol: str, exchange: Optional[str], country: Optional[str]) -> str:
    trimmed = (symbol or "").strip().upper()
    if not trimmed:
        return ""
    if "." in trimmed:
        return trimmed

    exchange_code = (exchange or "").strip().upper()
    country_code = (country or "").strip().upper()
    is_kuwait = (
        exchange_code in {"KW", "KSE", "BK"}
        or country_code in {"KW", "KWT", "KUWAIT"}
    )
    return f"{trimmed}.KW" if is_kuwait else f"{trimmed}.US"


def _parse_quarter_label(label: str) -> Optional[Tuple[int, str]]:
    """Parse a column header into (year, quarter_key).

    Handles formats like:
      'Mar '24', 'Mar 2024', 'Q1 2024', '2024-03-31', '03/2024'
    Returns None for 'Current' / TTM / unparseable.
    """
    s = label.strip()
    if not s or s.lower() in ("current", "ttm"):
        return None

    # ISO date 2024-03-31
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        q = _QUARTER_OF_MONTH.get(mo)
        return (y, q) if q else None

    # Mar '24 / Mar 2024 / Mar 31, 2024 / Mar-2024
    m = re.match(
        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"[\s\-]*(?:\d{1,2}[, ]+)?'?(\d{2,4})$",
        s, re.IGNORECASE,
    )
    if m:
        mo_name = m.group(1).title()
        mo = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(mo_name) + 1
        y_raw = int(m.group(2))
        y = 2000 + y_raw if y_raw < 100 else y_raw
        q = _QUARTER_OF_MONTH.get(mo)
        return (y, q) if q else None

    # Q1 2024 / Q1-2024
    m = re.match(r"^Q([1-4])[\s\-]*(\d{2,4})$", s, re.IGNORECASE)
    if m:
        q_num = int(m.group(1))
        y_raw = int(m.group(2))
        y = 2000 + y_raw if y_raw < 100 else y_raw
        return (y, f"q{q_num}")

    return None


def _strip_html(s: str) -> str:
    """Strip HTML tags and decode common entities."""
    s = re.sub(r"<[^>]+>", "", s)
    return (s.replace("&nbsp;", " ")
             .replace("&amp;", "&")
             .replace("&#39;", "'")
             .replace("&quot;", '"')
             .strip())


def _to_float(s: str) -> Optional[float]:
    s = s.replace(",", "").replace("%", "").strip()
    if not s or s in ("-", "—", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def _scrape_ratios_page(url: str) -> Tuple[List[Optional[Tuple[int, str]]], List[Optional[float]]]:
    """
    [P2-4/B-6] Fetch the quarterly ratios page and return (column_periods, pe_values).

    Retries up to 3 times on network / HTTP errors with exponential back-off.
    Timeout hard-capped at 30 s. Results are NOT cached here — the caller
    is responsible for checking ``_pe_cache`` before invoking.

    column_periods[i] is (year, q_key) tuple or None for 'Current'/unknown.
    pe_values[i] is the PE ratio for that column or None.
    """
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True, headers=_HEADERS)
    except Exception as e:  # noqa: BLE001
        logger.warning("ratios fetch failed for %s: %s", url, e)
        return [], []

    if resp.status_code != 200:
        logger.warning("ratios returned %s for %s", resp.status_code, url)
        return [], []

    html = resp.text

    # Find the financials table (stockanalysis.com uses id="main-table")
    table_m = re.search(
        r"<table[^>]*id=\"main-table\"[^>]*>(.*?)</table>", html, re.DOTALL,
    )
    if not table_m:
        # Fallback by class
        table_m = re.search(
            r"<table[^>]*class=\"[^\"]*financials-table[^\"]*\"[^>]*>(.*?)</table>",
            html, re.DOTALL,
        )
    if not table_m:
        # Last resort: first table
        table_m = re.search(r"<table[^>]*>(.*?)</table>", html, re.DOTALL)
    if not table_m:
        return [], []
    table_html = table_m.group(1)

    # Headers — first row contains <th> with column labels
    head_row_m = re.search(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL)
    headers: List[Optional[Tuple[int, str]]] = []
    if head_row_m:
        for cell_m in re.finditer(r"<th[^>]*>(.*?)</th>", head_row_m.group(1), re.DOTALL):
            label = _strip_html(cell_m.group(1))
            headers.append(_parse_quarter_label(label))
        # Drop the first label column ("Fiscal Quarter")
        if headers and headers[0] is None:
            headers = headers[1:]

    # PE Ratio row — locate by label text inside the row.
    # The label is nested inside <div>...</div> within the first <td>, so we
    # search for ">PE Ratio<" and walk back to the enclosing <tr>.
    pe_values: List[Optional[float]] = []
    label_m = re.search(r">\s*PE\s*Ratio\s*<", table_html, re.IGNORECASE)
    if label_m:
        # Find the <tr that opens before this position
        tr_start = table_html.rfind("<tr", 0, label_m.start())
        tr_end = table_html.find("</tr>", label_m.end())
        if tr_start != -1 and tr_end != -1:
            row_html = table_html[tr_start:tr_end]
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.DOTALL)
            # Skip first cell (label)
            for raw in cells[1:]:
                pe_values.append(_to_float(_strip_html(raw)))

    return headers, pe_values


def _scrape_current_pe(url: str) -> Optional[float]:
    """Fetch the live PE from the statistics page (SvelteKit bootstrap data)."""
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True, headers=_HEADERS)
    except Exception as e:  # noqa: BLE001
        logger.warning("statistics fetch failed for %s: %s", url, e)
        return None
    if resp.status_code != 200:
        return None

    text = resp.text
    m = re.search(r'\{id:"pe"[^}]*hover:"([^"]*)"', text)
    if not m:
        return None
    return _to_float(m.group(1))


# ── Verdict scaling ───────────────────────────────────────────────────


def _verdict(current_pe: Optional[float], avg_pe: Optional[float]) -> Dict[str, Any]:
    """Compare current P/E vs the average of the matching quarter.

    Returns {verdict, scale, scaleLabel, diffPct, diffAbs}.

    Scale (1-4) reflects the magnitude of |diff| as % of avg:
      1 = minimal   (<5%)
      2 = mild      (5-15%)
      3 = strong    (15-30%)
      4 = extreme   (>=30%)
    """
    if current_pe is None or avg_pe is None or avg_pe == 0:
        return {
            "verdict": "unknown",
            "scale": 0,
            "scaleLabel": "n/a",
            "diffPct": None,
            "diffAbs": None,
        }

    diff_abs = current_pe - avg_pe
    diff_pct = (diff_abs / avg_pe) * 100.0
    abs_pct = abs(diff_pct)

    if abs_pct < 1.0:
        verdict = "fair"
    elif diff_abs < 0:
        verdict = "undervalued"
    else:
        verdict = "overvalued"

    if abs_pct < 5:
        scale, label = 1, "minimal"
    elif abs_pct < 15:
        scale, label = 2, "mild"
    elif abs_pct < 30:
        scale, label = 3, "strong"
    else:
        scale, label = 4, "extreme"

    return {
        "verdict": verdict,
        "scale": scale,
        "scaleLabel": label,
        "diffPct": round(diff_pct, 2),
        "diffAbs": round(diff_abs, 2),
    }


# ── Endpoint ─────────────────────────────────────────────────────────


@router.get("/whale-candles")
async def whale_candles(
    symbol: str,
    exchange: Optional[str] = None,
    country: Optional[str] = None,
    from_date: Optional[date] = Query(default=None, alias="from"),
    to_date: Optional[date] = Query(default=None, alias="to"),
    indicators: bool = Query(default=True, description="Attach TA-Lib technical indicators"),
    current_user: TokenData = Depends(get_current_user),
):
    """OHLCV (and optional technical indicators) for the Whale Radar engine.

    Backed by TickerChart Live (replaces EODHD). Returns rows in the
    EODHD-compatible shape the mobile WhaleRadar already consumes:
        [{date, open, high, low, close, volume, ...indicators}, ...]

    Indicators are computed server-side via TA-Lib (the same C library
    bundled in TickerChart Live's desktop app) so values match the desktop
    chart bit-for-bit.
    """
    del current_user  # endpoint is auth-protected; user payload not otherwise needed here

    from app.services import tickerchart_service as tc
    from app.services.indicators_service import attach_indicators

    parsed = tc.split_symbol(symbol, exchange, country)
    if parsed is None:
        return {"status": "ok", "data": []}
    base, market = parsed

    # When indicators are requested we need extra history for the warmup
    # period (SMA-200 + MACD slowperiod is the longest at 200 + ~35 bars).
    # We fetch the broader window, compute, then trim to the requested range.
    fetch_from = from_date
    if indicators and from_date is not None:
        from datetime import timedelta
        fetch_from = from_date - timedelta(days=365)

    try:
        rows = await tc.fetch_ohlcv(base, market, from_d=fetch_from, to_d=to_date)
    except RuntimeError as exc:
        # Misconfigured credentials.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        logger.warning("TickerChart request failed for %s.%s: %s", base, market, exc)
        raise HTTPException(status_code=502, detail="Failed to reach TickerChart") from exc

    if indicators and rows:
        rows = attach_indicators(rows)
        # Trim back to requested window (we fetched extra warmup)
        if from_date is not None:
            iso = from_date.isoformat()
            rows = [r for r in rows if r["date"] >= iso]

    return {"status": "ok", "data": rows}


@router.get("/pe-quarterly/{stock_id}")
async def pe_quarterly(
    stock_id: int,
    response: Response,
    current_user: TokenData = Depends(get_current_user),
):
    """Quarterly P/E history (last 4 fiscal years) + current-quarter verdict.

    [P2-4/B-6] Results are cached in memory for 1 h (TTL cache).
    ``X-Cache-Status: HIT`` is returned when data comes from cache.
    Pulls from stockanalysis.com's quarterly ratios page on cache miss.
    """
    stock = query_one(
        "SELECT id, symbol, company_name, exchange, currency FROM analysis_stocks "
        "WHERE id = ? AND user_id = ?",
        (stock_id, current_user.user_id),
    )
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    symbol: str = stock["symbol"]
    company_name: Optional[str] = stock["company_name"]
    exchange: Optional[str] = stock["exchange"]
    currency: Optional[str] = stock["currency"]
    yf_ticker: Optional[str] = symbol  # symbol already carries the .KW suffix for KWSE

    # [P2-4/B-6] Check TTL cache before scraping
    cache_key = f"pe:{symbol}"
    cached = _pe_cache.get(cache_key)
    if cached is not None:
        response.headers["X-Cache-Status"] = "HIT"
        return cached

    response.headers["X-Cache-Status"] = "MISS"

    ratios_url = _ratios_url(symbol, yf_ticker, exchange, currency)
    stats_url = _statistics_url(symbol, yf_ticker, exchange, currency)

    try:
        headers, pe_values = _scrape_ratios_page(ratios_url)
        current_pe = _scrape_current_pe(stats_url)
    except Exception as exc:
        logger.warning("P/E scrape failed for %s (all retries exhausted): %s", symbol, exc)
        # Graceful degradation: return last cached state if available, else 502
        stale = _pe_cache.get(cache_key)
        if stale is not None:
            response.headers["X-Cache-Status"] = "STALE"
            return stale
        raise HTTPException(status_code=502, detail="P/E data temporarily unavailable — upstream scrape failed.")

    # Legacy safety-net: some rows may have wrong/default exchange/currency.
    # If no quarterly values were found, try Kuwait URL once for plain symbols.
    if not pe_values and "." not in symbol:
        kw_base = symbol.upper()
        fallback_ratios = f"https://stockanalysis.com/quote/kwse/{kw_base}/financials/ratios/?p=quarterly"
        fallback_stats = f"https://stockanalysis.com/quote/kwse/{kw_base}/statistics/"
        try:
            f_headers, f_values = _scrape_ratios_page(fallback_ratios)
            if f_values:
                headers, pe_values = f_headers, f_values
                if current_pe is None:
                    current_pe = _scrape_current_pe(fallback_stats)
        except Exception:
            pass  # ignore fallback failure — proceed with empty values

    # Build pe_table: { year: {q1, q2, q3, q4} } restricted to last 4 fiscal years
    today = date.today()
    current_year = today.year
    years = list(range(current_year - 3, current_year + 1))  # 4 years incl. current

    pe_table: Dict[int, Dict[str, Optional[float]]] = {
        y: {"q1": None, "q2": None, "q3": None, "q4": None} for y in years
    }

    for period, value in zip(headers, pe_values):
        if period is None or value is None:
            continue
        year, q_key = period
        if year in pe_table:
            pe_table[year][q_key] = value

    # Quarterly averages across the 4 years
    averages: Dict[str, Optional[float]] = {}
    for q in ("q1", "q2", "q3", "q4"):
        vals = [pe_table[y][q] for y in years if pe_table[y][q] is not None]
        averages[q] = round(sum(vals) / len(vals), 2) if vals else None

    # Growth table: YoY % change of PE for the same quarter
    # growth[year][q] = (pe[year][q] - pe[year-1][q]) / pe[year-1][q] * 100
    growth_table: Dict[int, Dict[str, Optional[float]]] = {
        y: {"q1": None, "q2": None, "q3": None, "q4": None} for y in years
    }
    for y in years:
        prev = y - 1
        for q in ("q1", "q2", "q3", "q4"):
            cur = pe_table[y][q]
            base = pe_table.get(prev, {}).get(q)
            if cur is not None and base is not None and base != 0:
                growth_table[y][q] = round(((cur - base) / base) * 100.0, 2)

    # Current quarter (calendar quarter of today's month)
    current_quarter = _QUARTER_OF_MONTH[today.month]
    compare_avg = averages[current_quarter]
    verdict = _verdict(current_pe, compare_avg)

    # Round pe_table for display
    pe_table_out = {
        y: {q: (round(v, 2) if v is not None else None) for q, v in row.items()}
        for y, row in pe_table.items()
    }

    result = {
        "status": "ok",
        "data": {
            "symbol": symbol,
            "company_name": company_name,
            "yf_ticker": yf_ticker,
            "years": years,
            "pe_table": pe_table_out,
            "growth_table": growth_table,
            "averages": averages,
            "current_pe": round(current_pe, 2) if current_pe is not None else None,
            "current_quarter": current_quarter,
            "compare_quarter_avg": compare_avg,
            "verdict": verdict,
            "source": "stockanalysis.com",
        },
    }
    # [P2-4/B-6] Store in TTL cache so subsequent calls within 1 h skip scraping
    _pe_cache[cache_key] = result
    return result
