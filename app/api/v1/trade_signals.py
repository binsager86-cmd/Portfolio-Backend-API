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
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user
from app.core.database import query_one
from app.core.security import TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trade-signals", tags=["Trade Signals"])

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


def _scrape_ratios_page(url: str) -> Tuple[List[Optional[Tuple[int, str]]], List[Optional[float]]]:
    """Fetch the quarterly ratios page and return (column_periods, pe_values).

    column_periods[i] is (year, q_key) tuple or None for 'Current'/unknown.
    pe_values[i] is the PE ratio for that column or None.
    """
    try:
        resp = httpx.get(url, timeout=20, follow_redirects=True, headers=_HEADERS)
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


@router.get("/pe-quarterly/{stock_id}")
async def pe_quarterly(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Quarterly P/E history (last 4 fiscal years) + current-quarter verdict.

    Pulls from stockanalysis.com's quarterly ratios page and the live
    statistics page for the current P/E reading.
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

    ratios_url = _ratios_url(symbol, yf_ticker, exchange, currency)
    stats_url = _statistics_url(symbol, yf_ticker, exchange, currency)

    headers, pe_values = _scrape_ratios_page(ratios_url)
    current_pe = _scrape_current_pe(stats_url)

    # Legacy safety-net: some rows may have wrong/default exchange/currency.
    # If no quarterly values were found, try Kuwait URL once for plain symbols.
    if not pe_values and "." not in symbol:
        kw_base = symbol.upper()
        fallback_ratios = f"https://stockanalysis.com/quote/kwse/{kw_base}/financials/ratios/?p=quarterly"
        fallback_stats = f"https://stockanalysis.com/quote/kwse/{kw_base}/statistics/"
        f_headers, f_values = _scrape_ratios_page(fallback_ratios)
        if f_values:
            headers, pe_values = f_headers, f_values
            if current_pe is None:
                current_pe = _scrape_current_pe(fallback_stats)

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

    return {
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
