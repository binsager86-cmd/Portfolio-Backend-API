"""
Trade Signals API — actionable buy/sell insights.

Currently implements F.Signals: P/E quarterly history + over/undervaluation
verdict for a chosen stock. Historical quarterly ratios come from
stockanalysis.com. The live current P/E prefers TickerChart close price plus
TickerChart ff_eps_basic(ltm), falls back to the latest stored EPS snapshot,
and finally falls back to stockanalysis.com when the local formula path cannot
be resolved.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import httpx
from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.api.deps import get_current_user, require_admin
from app.core.config import get_settings
from app.core.database import query_all, query_one
from app.core.security import TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trade-signals", tags=["Trade Signals"])

# [P2-4/B-6] TTL cache for P/E scrape results — 1 h TTL, max 256 symbol slots.
# Falls back to last known good value when upstream is temporarily unavailable.
_pe_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)

# Quarter Movement: 1 h TTL, max 256 symbol slots.
_quarter_movement_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)

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


def _derived_financial_data_version(stock_id: int) -> str:
    """Version marker for user-owned inputs that feed derived signal caches."""
    row = query_one(
        """
        SELECT
            COALESCE(MAX(ast.updated_at), 0) AS stock_updated_at,
            COALESCE(MAX(fs.created_at), 0) AS statement_created_at,
            COALESCE(MAX(fli.edited_at), 0) AS line_item_edited_at,
            COALESCE(MAX(sm.created_at), 0) AS metric_created_at,
            COUNT(DISTINCT fs.id) AS statement_count,
            COUNT(DISTINCT fli.id) AS line_item_count,
            COUNT(DISTINCT sm.id) AS metric_count
        FROM analysis_stocks ast
        LEFT JOIN financial_statements fs ON fs.stock_id = ast.id
        LEFT JOIN financial_line_items fli ON fli.statement_id = fs.id
        LEFT JOIN stock_metrics sm ON sm.stock_id = ast.id
        WHERE ast.id = ?
        """,
        (stock_id,),
    ) or {}
    parts = [
        row.get("stock_updated_at") or 0,
        row.get("statement_created_at") or 0,
        row.get("line_item_edited_at") or 0,
        row.get("metric_created_at") or 0,
        row.get("statement_count") or 0,
        row.get("line_item_count") or 0,
        row.get("metric_count") or 0,
    ]
    return ":".join(str(part) for part in parts)


def _derived_cache_key(prefix: str, user_id: int, stock_id: int, symbol: str) -> str:
    version = _derived_financial_data_version(stock_id)
    return f"{prefix}:v4:user:{user_id}:stock:{stock_id}:symbol:{symbol}:fdv:{version}"


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

    # stockanalysis.com splits the ratios page into several section tables
    # (e.g. id="main-table-total-valuation", "main-table-price-ratios",
    # "main-table-ev-ratios", ...). The "PE Ratio" row only lives in one of
    # them, so scan every table on the page and pick the one that actually
    # contains a "PE Ratio" row label instead of assuming the first table
    # (or the first "main-table*"/"financials-table" match) is the right one.
    candidate_tables: List[str] = [
        m.group(1) for m in re.finditer(r"<table[^>]*>(.*?)</table>", html, re.DOTALL)
    ]
    table_html: Optional[str] = None
    for candidate in candidate_tables:
        if re.search(r">\s*PE\s*Ratio\s*<", candidate, re.IGNORECASE):
            table_html = candidate
            break
    if table_html is None and candidate_tables:
        # Fall back to the first table so headers can still be parsed even
        # if the PE row lookup below comes up empty.
        table_html = candidate_tables[0]
    if table_html is None:
        return [], []

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


def _latest_positive_eps_snapshot(stock_id: int) -> Tuple[Optional[float], str]:
    snapshots, eps_source = _resolve_eps_snapshots_for_stock(stock_id)
    for snapshot in reversed(snapshots):
        try:
            eps_value = float(snapshot.get("eps_value"))
        except (AttributeError, TypeError, ValueError):
            continue
        if eps_value > 0:
            return eps_value, eps_source
    return None, eps_source


async def _resolve_current_pe(
    stock_id: int,
    symbol: str,
    exchange: Optional[str],
    currency: Optional[str],
    stats_url: str,
) -> Tuple[Optional[float], str]:
    from datetime import timedelta

    from app.services import tickerchart_service as tc

    parsed = tc.split_symbol(symbol, exchange, None)
    normalized_close: Optional[float] = None

    if parsed is not None:
        base_symbol, market = parsed
        try:
            rows = await tc.fetch_ohlcv(
                base_symbol,
                market,
                from_d=date.today() - timedelta(days=14),
                to_d=None,
            )
        except (RuntimeError, ValueError, httpx.HTTPError) as exc:
            logger.warning("TickerChart current close fetch failed for %s: %s", symbol, exc)
        else:
            latest_close: Optional[float] = None
            for row in sorted(rows, key=lambda item: item.get("date") or ""):
                try:
                    close_value = float(row.get("close") or 0)
                except (AttributeError, TypeError, ValueError):
                    continue
                if close_value > 0:
                    latest_close = close_value

            if latest_close is not None:
                price_divisor = 1000.0 if (
                    market == "KSE"
                    or (currency or "").upper() == "KWD"
                    or (exchange or "").upper() in {"KSE", "KWSE", "KUWAIT"}
                ) else 1.0
                normalized_close = latest_close / price_divisor

        if normalized_close is not None and normalized_close > 0:
            try:
                tickerchart_eps = await tc.fetch_ltm_eps(base_symbol, market)
            except (RuntimeError, ValueError, httpx.HTTPError) as exc:
                logger.warning("TickerChart LTM EPS fetch failed for %s: %s", symbol, exc)
            else:
                if tickerchart_eps is not None and tickerchart_eps > 0:
                    return (
                        normalized_close / tickerchart_eps,
                        "stockanalysis.com history + tickerchart close / tickerchart ff_eps_basic(ltm)",
                    )

    ttm_eps, eps_source = _latest_positive_eps_snapshot(stock_id)
    if ttm_eps is not None and normalized_close is not None and normalized_close > 0:
        return (
            normalized_close / ttm_eps,
            f"stockanalysis.com history + tickerchart close / {eps_source} EPS",
        )

    return await asyncio.to_thread(_scrape_current_pe, stats_url), "stockanalysis.com"


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
    Pulls historical ratios from stockanalysis.com's quarterly ratios page on
    cache miss and derives the live current P/E locally when possible.
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
    cache_key = _derived_cache_key("pe", current_user.user_id, stock_id, symbol)
    cached = _pe_cache.get(cache_key)
    if cached is not None:
        response.headers["X-Cache-Status"] = "HIT"
        return cached

    response.headers["X-Cache-Status"] = "MISS"

    ratios_url = _ratios_url(symbol, yf_ticker, exchange, currency)
    stats_url = _statistics_url(symbol, yf_ticker, exchange, currency)

    try:
        headers, pe_values = await asyncio.to_thread(_scrape_ratios_page, ratios_url)
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
            f_headers, f_values = await asyncio.to_thread(_scrape_ratios_page, fallback_ratios)
            if f_values:
                headers, pe_values = f_headers, f_values
        except Exception:
            pass  # ignore fallback failure — proceed with empty values

    current_pe, source = await _resolve_current_pe(
        stock_id=stock_id,
        symbol=symbol,
        exchange=exchange,
        currency=currency,
        stats_url=stats_url,
    )

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
            "source": source,
        },
    }
    # [P2-4/B-6] Store in TTL cache so subsequent calls within 1 h skip scraping
    _pe_cache[cache_key] = result
    return result


# ── Kuwait Multi-Factor Signal Engine ────────────────────────────────────────


@router.get("/kuwait-signal")
async def kuwait_signal(
    symbol: str = Query(..., pattern=r"^[A-Z0-9][A-Z0-9.]{0,11}$", description="Stock symbol (e.g. NBK, ZAIN)"),
    exchange: Optional[str] = Query(default="KSE"),
    country: Optional[str] = Query(default=None),
    segment: str = Query(default="PREMIER", description="PREMIER | MAIN | AUCTION"),
    account_equity: float = Query(default=100_000.0, description="Account size in KWD for position sizing"),
    delay_hours: int = Query(default=0, ge=0, description="Hours since signal was generated (confidence decay)"),
    wins: Optional[int] = Query(default=None, description="Recent winning trades count (Bayesian calibration)"),
    total_trades: Optional[int] = Query(default=None, description="Recent total trades count (Bayesian calibration)"),
    current_user: TokenData = Depends(get_current_user),
):
    """Multi-factor technical trade signal for Kuwait Premier Market stocks.

    Fetches 2-year OHLCV history from TickerChart, computes full indicator
    suite via TA-Lib, then runs the Kuwait Signal Engine:

    • Liquidity filter (ADTV, spread proxy, active-days, wash-trade check)
    • 3-state HMM regime detection (Bullish / Neutral / Bearish)
    • Confluence scoring: trend + momentum + volume/flow + S/R + risk-reward
    • Dynamic regime-based weight adjustments
    • CVaR-adjusted position sizing (liquidity-aware Kelly fraction)
    • Probability calibration (isotonic regression + Bayesian updating)
    • Time-based confidence decay (T+24h → 85 %, T+48h → 65 %, T+72h → 0 %)
    • Circuit-breaker and Kuwait tick-grid alignment on all price levels

    Returns the canonical signal JSON schema (see Section 6 of spec).
    """
    del current_user

    from datetime import timedelta

    from app.services import tickerchart_service as tc
    from app.services.indicators_service import attach_indicators
    from app.services.signal_engine.engine.signal_generator import generate_kuwait_signal

    parsed = tc.split_symbol(symbol, exchange, country)
    if parsed is None:
        raise HTTPException(status_code=400, detail=f"Cannot resolve symbol '{symbol}' to a TickerChart market")
    base, market = parsed

    # Fetch 2 years of history to ensure sufficient warmup for HMM training
    # and long-period indicators (SMA-200 needs 200 bars + signal engine needs 250+)
    from datetime import date as _date
    fetch_from = _date.today() - timedelta(days=730)

    try:
        rows = await tc.fetch_ohlcv(base, market, from_d=fetch_from, to_d=None)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        logger.warning("TickerChart request failed for %s.%s: %s", base, market, exc)
        raise HTTPException(status_code=502, detail="Failed to reach TickerChart data provider") from exc

    if not rows:
        raise HTTPException(status_code=404, detail=f"No price data returned for {symbol}")

    # Fill short gaps (≤ 3 Kuwait trading sessions) before indicator computation
    from app.services.signal_engine.data.preprocessing import forward_fill_gaps
    rows = forward_fill_gaps(rows)

    # Attach TA-Lib indicators (same as whale-candles endpoint)
    rows = attach_indicators(rows)

    # Optional Bayesian calibration context
    recent_performance: Optional[dict] = None
    if wins is not None and total_trades is not None and total_trades > 0:
        recent_performance = {"wins": wins, "total": total_trades}

    signal = await generate_kuwait_signal(
        rows=rows,
        stock_code=base,
        segment=segment.upper(),
        account_equity=account_equity,
        delay_hours=delay_hours,
        recent_performance=recent_performance,
    )

    return {"status": "ok", "data": signal}


@router.get("/technical-batch/latest")
async def technical_batch_latest(
    limit: int = Query(default=300, ge=1, le=1000),
    current_user: TokenData = Depends(get_current_user),
):
    """Return the latest stored technical universe batch run + score rows."""
    del current_user
    from app.services.technical_batch_service import get_latest_run

    return {"status": "ok", "data": get_latest_run(limit=limit)}


@router.get("/technical-batch/{run_id}")
async def technical_batch_by_id(
    run_id: int,
    limit: int = Query(default=300, ge=1, le=1000),
    current_user: TokenData = Depends(get_current_user),
):
    """Return a specific technical universe batch run + score rows."""
    del current_user
    from app.services.technical_batch_service import get_run_by_id

    data = get_run_by_id(run_id, limit=limit)
    if not data.get("run"):
        raise HTTPException(status_code=404, detail=f"Technical batch run {run_id} not found")
    return {"status": "ok", "data": data}


@router.post("/technical-batch/run")
async def technical_batch_run(
    background: bool = Query(default=True, description="Run in background and return immediately"),
    segment: str = Query(default="PREMIER", description="Segment label used by signal model"),
    max_concurrency: int = Query(default=4, ge=1, le=8),
    limit: Optional[int] = Query(default=None, ge=1, le=500),
    current_user: TokenData = Depends(require_admin),
):
    """Trigger technical universe scoring for all configured Kuwait symbols."""
    from app.services.technical_batch_service import kickoff_batch_background, run_batch_once

    if background:
        payload = kickoff_batch_background(
            triggered_by="manual",
            requested_by_user_id=current_user.user_id,
            segment=segment,
            max_concurrency=max_concurrency,
            limit=limit,
        )
        return {"status": "ok", "data": payload}

    payload = await run_batch_once(
        triggered_by="manual",
        requested_by_user_id=current_user.user_id,
        segment=segment,
        max_concurrency=max_concurrency,
        limit=limit,
    )
    return {"status": "ok", "data": payload}


# ── Quarter Movement ──────────────────────────────────────────────────────────

_QM_START_DATE = "2023-01-01"


def _is_subunit_eps_code(code_str: object) -> bool:
    if not isinstance(code_str, str):
        return False
    low = code_str.lower()
    return "fils" in low or "cents" in low or "halala" in low


# ── yfinance quarterly EPS fallback ──────────────────────────────────────────

_yf_eps_cache: Dict[str, List[Dict]] = {}


def _fetch_yfinance_quarterly_eps(symbol: str, exchange: Optional[str]) -> List[Dict]:
    """Fetch quarterly EPS from yfinance for stocks that have no local EPS data.

    Returns list of {period_end_date: ISO str, eps_value: float}, oldest first.
    Values are in the stock's native currency unit (KWD for KSE, USD for US).
    """
    cache_key = f"{symbol}:{exchange}"
    if cache_key in _yf_eps_cache:
        return _yf_eps_cache[cache_key]

    # Build yfinance ticker symbol
    exch = (exchange or "").upper()
    if exch in {"KSE", "KWSE", "KUWAIT"}:
        yf_ticker = f"{symbol}.KW"
    else:
        yf_ticker = symbol  # US stocks use bare symbol on Yahoo Finance

    try:
        import yfinance as yf

        tk = yf.Ticker(yf_ticker)
        qi = tk.quarterly_income_stmt
        if qi is None or qi.empty:
            _yf_eps_cache[cache_key] = []
            return []

        # Prefer Diluted EPS, fall back to Basic EPS
        eps_row = None
        for candidate in ("Diluted EPS", "Basic EPS"):
            if candidate in qi.index:
                eps_row = qi.loc[candidate]
                break

        if eps_row is None:
            _yf_eps_cache[cache_key] = []
            return []

        snapshots: List[Dict] = []
        for col, val in eps_row.items():
            if val is None or (hasattr(val, "__class__") and val.__class__.__name__ == "float" and str(val) == "nan"):
                continue
            try:
                import math
                fval = float(val)
                if math.isnan(fval):
                    continue
                period_str = str(col)[:10]  # "YYYY-MM-DD" from Timestamp
                snapshots.append({"period_end_date": period_str, "eps_value": fval})
            except (TypeError, ValueError):
                continue

        snapshots.sort(key=lambda r: r["period_end_date"])
        _yf_eps_cache[cache_key] = snapshots
        return snapshots

    except Exception as exc:
        logger.warning("yfinance EPS fetch failed for %s: %s", yf_ticker, exc)
        _yf_eps_cache[cache_key] = []
        return []


def _resolve_eps_snapshots_for_stock(stock_id: int) -> Tuple[List[Dict], str]:
    """
    Retrieve all available EPS snapshots from stored fundamentals for this stock.

    Prefers stock_metrics EPS rows, then falls back to financial statement
    line items. No network fallback is used here because quarter movement
    should stay aligned with the app's stored fundamentals path.

    Each returned dict has: period_end_date (ISO str), eps_value (float).
    """
    def _coerce_snapshots(rows: List[Dict], value_key: str = "eps_value") -> List[Dict]:
        snapshots: List[Dict] = []
        for row in rows:
            eps_val = row.get(value_key)
            period_end = row.get("period_end_date")
            if eps_val is None or not period_end:
                continue
            try:
                snapshots.append({
                    "period_end_date": str(period_end)[:10],
                    "eps_value": float(eps_val),
                })
            except (TypeError, ValueError):
                continue
        return snapshots

    metric_eps_rows = query_all(
        """
        SELECT period_end_date,
               metric_value AS eps_value
        FROM   stock_metrics
        WHERE  stock_id = ?
          AND  metric_name = 'EPS'
          AND  period_end_date IS NOT NULL
          AND  metric_value IS NOT NULL
        ORDER  BY period_end_date ASC
        """,
        (stock_id,),
    )
    metric_snapshots = _coerce_snapshots(metric_eps_rows)
    if metric_snapshots:
        return metric_snapshots, "stock_metrics"

    db_eps_rows = query_all(
        """
        SELECT li.amount AS eps_value,
               li.line_item_code,
               fs.period_end_date
        FROM   financial_line_items li
        JOIN   financial_statements fs ON fs.id = li.statement_id
        WHERE  fs.stock_id = ?
          AND  fs.statement_type = 'income'
          AND  li.amount IS NOT NULL
                    AND  (
                                     UPPER(li.line_item_code) IN ('EPS_DILUTED', 'EPS_BASIC')
                                OR LOWER(li.line_item_code) LIKE '%earnings_per_share%'
                                OR LOWER(li.line_item_code) LIKE '%eps_%'
                             )
          AND  fs.period_end_date IS NOT NULL
        ORDER  BY fs.period_end_date ASC,
                  CASE WHEN UPPER(li.line_item_code) = 'EPS_DILUTED' THEN 0 ELSE 1 END ASC
        """,
        (stock_id,),
    )

    if db_eps_rows:
        deduped_rows: List[Dict] = []
        seen_dates: set[str] = set()
        for row in db_eps_rows:
            period_end = str(row.get("period_end_date") or "")[:10]
            if not period_end or period_end in seen_dates:
                continue
            seen_dates.add(period_end)
            eps_val = row.get("eps_value")
            if eps_val is not None and _is_subunit_eps_code(row.get("line_item_code")):
                try:
                    eps_val = float(eps_val) / 1000.0
                except (TypeError, ValueError):
                    continue
            deduped_rows.append({"period_end_date": period_end, "eps_value": eps_val})

        snapshots = _coerce_snapshots(deduped_rows)
        if snapshots:
            return snapshots, "financials"

    # ── yfinance fallback: fetch quarterly EPS from Yahoo Finance ────────────
    stock_row = query_one(
        "SELECT symbol, exchange FROM analysis_stocks WHERE id = ?",
        (stock_id,),
    )
    if stock_row:
        yf_snapshots = _fetch_yfinance_quarterly_eps(
            stock_row["symbol"], stock_row["exchange"]
        )
        if yf_snapshots:
            return yf_snapshots, "yfinance"

    return [], "none"


@router.get("/quarter-movement/{stock_id}")
async def quarter_movement(
    stock_id: int,
    response: Response,
    current_user: TokenData = Depends(get_current_user),
):
    """Quarterly price & P/E movement analysis + expected price forecast.

    Fetches OHLCV from TickerChart from 2023-01-01, computes quarterly
    high/low percentage changes from each quarter's baseline closing price
    (Module 1), quarterly highest/lowest daily P/E ratios (Module 2), and
    three-method expected price forecasts for the active quarter (Module 3).

    Results are cached in memory for 1 h (TTL cache).
    ``X-Cache-Status: HIT`` is returned when data comes from cache.

    Spec §7.4: retries TickerChart up to 3× with exponential back-off;
    serves stale cache on exhausted retries.
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
    currency: Optional[str] = stock["currency"]
    exchange: Optional[str] = stock["exchange"]

    cache_key = _derived_cache_key("qm", current_user.user_id, stock_id, symbol)
    cached = _quarter_movement_cache.get(cache_key)
    if cached is not None:
        response.headers["X-Cache-Status"] = "HIT"
        return cached

    response.headers["X-Cache-Status"] = "MISS"

    from app.services import tickerchart_service as tc

    parsed = tc.split_symbol(symbol, exchange, None)
    if parsed is None:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resolve symbol '{symbol}' to a TickerChart market",
        )
    base_symbol, market = parsed

    from datetime import datetime as _dt

    start_d = _dt.strptime(_QM_START_DATE, "%Y-%m-%d").date()
    today = date.today()

    # Spec §7.4: retry TickerChart up to 3× with 2 s back-off between attempts
    daily_records: List[Dict] = []
    last_exc: Optional[Exception] = None
    for _attempt in range(3):
        try:
            daily_records = await tc.fetch_ohlcv(base_symbol, market, from_d=start_d, to_d=None)
            last_exc = None
            break
        except (httpx.HTTPError, RuntimeError) as exc:
            last_exc = exc
            import asyncio
            await asyncio.sleep(2)

    if last_exc is not None or not daily_records:
        stale = _quarter_movement_cache.get(cache_key)
        if stale is not None:
            response.headers["X-Cache-Status"] = "STALE"
            return stale
        if last_exc is not None:
            raise HTTPException(status_code=502, detail="Failed to reach TickerChart data provider") from last_exc
        raise HTTPException(status_code=404, detail=f"No price data returned for {symbol}")

    # ── Run the three modules ─────────────────────────────────────────────
    from app.services.quarter_movement import (
        ExpectedPriceForecastModule,
        QuarterlyPERatioMovementModule,
        QuarterlyPriceMovementModule,
    )
    from app.services.quarter_movement.price_module import _active_quarter_for_date

    # PE price divisor: KSE/KWD prices are in fils (÷1000 → KWD)
    pe_price_divisor = 1000.0 if (
        market == "KSE"
        or (currency or "").upper() == "KWD"
        or (exchange or "").upper() in {"KSE", "KWSE", "KUWAIT"}
    ) else 1.0

    module_one_result = QuarterlyPriceMovementModule().compute(daily_records, today)
    eps_snapshots, eps_source = _resolve_eps_snapshots_for_stock(stock_id)

    # ── Module 2: exclusively use TickerChart FlatFiles PE data ──────────
    # Auto-discovery is attempted inside fetch_pe_from_flatfiles when the
    # symbol is not yet in the map; it cross-references QuotesSnapShot.bin.
    flatfiles_pe = tc.fetch_pe_from_flatfiles(base_symbol, market)

    # ── Gap-fill: extend flatfile PE to present via live QuotesSnapshot P/E ──
    # TickerChart flatfiles cache historical PE up to ~March 2025. For trading
    # days after the last cached date, we derive daily PE by scaling the live
    # P/E from QuotesSnapShot.bin by each day's price relative to the most
    # recent OHLCV close — all TickerChart data, no external sources.
    #
    #   pe_day = (price_day / price_ref) × live_pe
    #
    # where live_pe is the current P/E from QuotesSnapshot and price_ref is
    # the most recent OHLCV close (the price TickerChart used to compute live_pe).
    # We use each day's OHLCV high/low for better quarterly high/low accuracy.
    _GAP_FILL_THRESHOLD_DAYS = 30
    if daily_records:
        last_flatfile_date = max(flatfiles_pe.keys()) if flatfiles_pe else None
        import datetime as _dt_mod
        cutoff = today - _dt_mod.timedelta(days=_GAP_FILL_THRESHOLD_DAYS)
        if last_flatfile_date is None or last_flatfile_date < cutoff:
            live_pe = tc._read_quotes_snapshot_pe(base_symbol, market)
            if live_pe and live_pe > 0:
                # Reference close = most recent OHLCV close (TickerChart price)
                ref_close_raw: Optional[float] = None
                for _row in reversed(daily_records):
                    _c = _row.get("close")
                    if _c and float(_c) > 0:
                        ref_close_raw = float(_c)
                        break
                if ref_close_raw and ref_close_raw > 0:
                    ref_close = ref_close_raw / pe_price_divisor
                    gap_anchor = last_flatfile_date if last_flatfile_date else date(1900, 1, 1)
                    for row in daily_records:
                        row_date_str = row.get("date")
                        if not row_date_str:
                            continue
                        try:
                            row_date = date.fromisoformat(row_date_str)
                        except ValueError:
                            continue
                        if row_date <= gap_anchor:
                            continue
                        close_val = row.get("close")
                        if close_val is None:
                            continue
                        try:
                            close_p = float(close_val)
                            high_p = float(row.get("high") or close_val)
                            low_p = float(row.get("low") or close_val)
                        except (TypeError, ValueError):
                            continue
                        if close_p <= 0:
                            continue
                        # Scale live P/E by price ratio — no EPS needed
                        high_pe = (high_p / pe_price_divisor / ref_close) * live_pe
                        low_pe = (low_p / pe_price_divisor / ref_close) * live_pe
                        if 1.0 < low_pe < high_pe < 500.0 or 1.0 < high_pe <= low_pe < 500.0:
                            flatfiles_pe[row_date] = (max(high_pe, low_pe), min(high_pe, low_pe))

    module_two = QuarterlyPERatioMovementModule()
    if flatfiles_pe:
        module_two_result = module_two.compute_from_pe_series(flatfiles_pe, today)
        ttm_eps_source = "flatfiles"
    else:
        module_two_result = module_two.compute(
            daily_records,
            eps_snapshots,
            today,
            price_divisor=pe_price_divisor,
        )
        ttm_eps_source = eps_source

    _today_close: Optional[float] = None
    if daily_records:
        for _r in reversed(daily_records):
            _c2 = _r.get("close")
            if _c2 and float(_c2) > 0:
                _today_close = float(_c2)
                break
    current_price = round(_today_close / pe_price_divisor, 3) if _today_close else None
    # Derive implied TTM EPS from the live TickerChart P/E + most recent OHLCV
    # close so Module 3 (Method 2 expected price) can produce a forecast.
    _live_pe_for_eps = tc._read_quotes_snapshot_pe(base_symbol, market)
    if _live_pe_for_eps and _live_pe_for_eps > 0 and _today_close:
        module_two_result["ttm_eps"] = round((_today_close / pe_price_divisor) / _live_pe_for_eps, 6)
        ttm_eps_source = "tickerchart_live"
    elif module_two_result.get("ttm_eps") is None and eps_snapshots:
        try:
            module_two_result["ttm_eps"] = float(eps_snapshots[-1]["eps_value"])
            ttm_eps_source = eps_source
        except (KeyError, TypeError, ValueError):
            pass

    active_year, active_quarter_key = _active_quarter_for_date(today)
    module_three_result = ExpectedPriceForecastModule().compute(
        active_quarter_key=active_quarter_key,
        baseline_price=module_one_result["active_quarter_baseline_price"],
        price_movement_means=module_one_result["price_movement_means"],
        pe_movement_means=module_two_result["pe_movement_means"],
        trailing_twelve_months_eps=module_two_result["ttm_eps"],
    )

    _QUARTER_LABEL = {"q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4"}

    result = {
        "status": "ok",
        "data": {
            "symbol": symbol,
            "company_name": company_name,
            "currency": currency,
            # Active quarter context
            "active_quarter": _QUARTER_LABEL[active_quarter_key],
            "active_quarter_key": active_quarter_key,
            "active_year": active_year,
            "current_price": current_price,
            "baseline_price": module_one_result["active_quarter_baseline_price"],
            # Module 1 outputs
            "years": module_one_result["years"],
            "price_movement_table": module_one_result["price_movement_table"],
            "price_movement_means": module_one_result["price_movement_means"],
            # Module 2 outputs
            "pe_movement_table": module_two_result["pe_movement_table"],
            "pe_movement_means": module_two_result["pe_movement_means"],
            "ttm_eps": module_two_result["ttm_eps"],
            "ttm_eps_source": ttm_eps_source,
            "eps_coverage": module_two_result["eps_coverage"],
            # Module 3 outputs
            "method_one_expected_price": module_three_result["method_one_expected_price"],
            "method_one_expected_low_price": module_three_result["method_one_expected_low_price"],
            "method_two_expected_price": module_three_result["method_two_expected_price"],
            "method_two_expected_low_price": module_three_result["method_two_expected_low_price"],
            "consensus_expected_price": module_three_result["consensus_expected_price"],
            "consensus_expected_low_price": module_three_result["consensus_expected_low_price"],
            "method_one_inputs": module_three_result["method_one_inputs"],
            "method_two_inputs": module_three_result["method_two_inputs"],
            # Metadata
            "data_source": "tickerchart",
            "last_updated": today.isoformat(),
            "stale": False,
        },
    }

    _quarter_movement_cache[cache_key] = result
    return result


@router.post("/quarter-movement/cache/clear")
async def quarter_movement_cache_clear(
    current_user: TokenData = Depends(get_current_user),
):
    """Clear all cached quarter-movement results so the next request re-fetches
    fresh PE data from the FlatFiles (or EPS fallback).

    Useful after adding a new flatfile PE mapping or updating EPS data.
    """
    del current_user
    count = len(_quarter_movement_cache)
    _quarter_movement_cache.clear()
    return {"status": "ok", "cleared": count}

