"""StockAnalysis fundamentals helpers.

Provides lightweight scraping helpers for trailing EPS and Book Value Per Share
used by Eagle Eye daily fundamentals refresh.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
import threading
from typing import Optional

import httpx
from cachetools import TTLCache

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

_TRAILING_CACHE: TTLCache = TTLCache(maxsize=2048, ttl=6 * 3600)
_HTTP_CLIENT: Optional[httpx.Client] = None
_HTTP_CLIENT_LOCK = threading.Lock()


def _is_kwse_symbol(symbol: str, market_abb: str | None) -> bool:
    sym = (symbol or "").upper().strip()
    market = (market_abb or "").upper().strip()
    return sym.endswith(".KW") or market in {"KSE", "KWSE", "KUWAIT"}


def _build_urls(symbol: str, market_abb: str | None) -> tuple[str, str, str, str]:
    sym = (symbol or "").upper().strip()
    base = re.sub(r"\.KW$", "", sym)
    if _is_kwse_symbol(sym, market_abb):
        return (
            f"https://stockanalysis.com/quote/kwse/{base}/financials/?p=trailing",
            f"https://stockanalysis.com/quote/kwse/{base}/financials/balance-sheet/?p=trailing",
            f"https://stockanalysis.com/quote/kwse/{base}/financials/ratios/?p=trailing",
            f"https://stockanalysis.com/quote/kwse/{base}/financials/metrics/?p=trailing",
        )
    base_us = re.sub(r"[^a-zA-Z0-9-]", "-", base).strip("-").lower()
    return (
        f"https://stockanalysis.com/stocks/{base_us}/financials/?p=trailing",
        f"https://stockanalysis.com/stocks/{base_us}/financials/balance-sheet/?p=trailing",
        f"https://stockanalysis.com/stocks/{base_us}/financials/ratios/?p=trailing",
        f"https://stockanalysis.com/stocks/{base_us}/financials/metrics/?p=trailing",
    )


def _strip_html(value: str) -> str:
    txt = re.sub(r"<[^>]+>", "", value)
    return (
        txt.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
        .strip()
    )


def _to_float(value: str) -> Optional[float]:
    s = (value or "").strip().replace(",", "").replace("\u2212", "-")
    if not s or s in {"-", "--", "—", "N/A", "n/a"}:
        return None

    negative_parentheses = s.startswith("(") and s.endswith(")")
    if negative_parentheses:
        s = s[1:-1].strip()

    if s.endswith("%"):
        s = s[:-1].strip()

    multiplier = 1.0
    if s:
        suffix = s[-1].upper()
        if suffix in {"K", "M", "B", "T"}:
            multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[suffix]
            s = s[:-1].strip()

    try:
        out = float(s) * multiplier
    except ValueError:
        return None
    if negative_parentheses:
        out = -out
    return out


def _extract_row_first_numeric(page_html: str, labels: tuple[str, ...]) -> Optional[float]:
    for label in labels:
        match = re.search(rf">\s*{re.escape(label)}\s*<", page_html, re.IGNORECASE)
        if not match:
            continue

        tr_start = page_html.rfind("<tr", 0, match.start())
        tr_end = page_html.find("</tr>", match.end())
        if tr_start == -1 or tr_end == -1:
            continue
        row_html = page_html[tr_start:tr_end]

        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.DOTALL)
        if len(cells) <= 1:
            continue

        for raw_cell in cells[1:]:
            title_match = re.search(r'title="([^"]+)"', raw_cell)
            if title_match:
                parsed_title = _to_float(title_match.group(1))
                if parsed_title is not None:
                    return parsed_title

            parsed = _to_float(_strip_html(raw_cell))
            if parsed is not None:
                return parsed

    return None


def _fetch_page(url: str) -> str:
    response = _get_http_client().get(url)
    response.raise_for_status()
    return response.text


def _get_http_client() -> httpx.Client:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        return _HTTP_CLIENT

    with _HTTP_CLIENT_LOCK:
        if _HTTP_CLIENT is None:
            _HTTP_CLIENT = httpx.Client(
                timeout=20.0,
                follow_redirects=True,
                headers=_HEADERS,
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            )
    return _HTTP_CLIENT


def _extract_statistics_title_value(page_html: str, label: str) -> Optional[float]:
    match = re.search(rf">\s*{re.escape(label)}\s*<", page_html, re.IGNORECASE)
    if not match:
        return None

    tr_start = page_html.rfind("<tr", 0, match.start())
    tr_end = page_html.find("</tr>", match.end())
    if tr_start == -1 or tr_end == -1:
        return None

    row_html = page_html[tr_start:tr_end]
    title_match = re.search(r'title="([^"]+)"', row_html)
    if title_match:
        parsed = _to_float(title_match.group(1))
        if parsed is not None:
            return parsed

    return _extract_row_first_numeric(page_html[tr_start:tr_end + 5], (label,))


def _extract_statistics_trailing_pe(page_html: str) -> Optional[float]:
    # Example text: "The trailing PE ratio is 26.87."
    match = re.search(r"trailing\s+pe\s+ratio\s+is\s+([0-9]+(?:\.[0-9]+)?)", page_html, re.IGNORECASE)
    if not match:
        return None
    try:
        val = float(match.group(1))
    except ValueError:
        return None
    return val if val > 0 else None


def fetch_trailing_eps_bvps(symbol: str, market_abb: str | None = "KSE") -> dict[str, Optional[float]]:
    """Fetch trailing EPS (basic) and BVPS from StockAnalysis.

    Returns dict: {"eps": float|None, "book_value_per_share": float|None, "pe_ratio": float|None}
    """
    cache_key = f"{(symbol or '').upper()}:{(market_abb or '').upper()}"
    cached = _TRAILING_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    financials_url, balance_sheet_url, ratios_url, metrics_url = _build_urls(symbol, market_abb)
    stats_url = financials_url.replace("/financials/?p=trailing", "/statistics/")
    eps: Optional[float] = None
    bvps: Optional[float] = None
    pe_ratio: Optional[float] = None

    # Primary precise source: statistics page title attributes and valuation text.
    try:
        stats_html = _fetch_page(stats_url)
        eps = _extract_statistics_title_value(stats_html, "Earnings Per Share (EPS)")
        bvps = _extract_statistics_title_value(stats_html, "Book Value Per Share")
        pe_ratio = _extract_statistics_trailing_pe(stats_html)
    except Exception as exc:
        logger.debug("StockAnalysis statistics fetch failed for %s: %s", symbol, exc)

    if eps is None:
        try:
            income_html = _fetch_page(financials_url)
            eps = _extract_row_first_numeric(income_html, ("EPS (Basic)", "EPS (Diluted)"))
        except Exception as exc:
            logger.debug("StockAnalysis EPS fetch failed for %s: %s", symbol, exc)

    bvps_labels = (
        "Book Value Per Share",
        "Book Value / Share",
        "Book Value Per Share (MRQ)",
    )
    for bvps_url in (balance_sheet_url, ratios_url, metrics_url):
        if bvps is not None:
            break
        try:
            page_html = _fetch_page(bvps_url)
            bvps = _extract_row_first_numeric(page_html, bvps_labels)
        except Exception as exc:
            logger.debug("StockAnalysis BVPS fetch failed for %s (%s): %s", symbol, bvps_url, exc)

    result = {
        "eps": eps,
        "book_value_per_share": bvps,
        "pe_ratio": pe_ratio,
    }
    _TRAILING_CACHE[cache_key] = dict(result)
    return result


def fetch_trailing_eps_bvps_batch(
    symbols: list[str],
    market_abb: str | None = "KSE",
    max_workers: int = 4,
) -> dict[str, dict[str, Optional[float]]]:
    """Fetch trailing fundamentals in a low-overhead bounded-concurrency batch.

    - Reuses a shared HTTP client connection pool.
    - Uses modest worker count to keep CPU/network load predictable.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in symbols or []:
        sym = str(raw or "").upper().strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        cleaned.append(sym)

    if not cleaned:
        return {}

    workers = max(1, min(int(max_workers or 1), 8))
    results: dict[str, dict[str, Optional[float]]] = {}

    if workers == 1 or len(cleaned) == 1:
        for sym in cleaned:
            try:
                results[sym] = fetch_trailing_eps_bvps(sym, market_abb)
            except Exception as exc:
                logger.debug("StockAnalysis batch fetch failed for %s: %s", sym, exc)
                results[sym] = {"eps": None, "book_value_per_share": None, "pe_ratio": None}
        return results

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sa-fund") as pool:
        future_to_symbol = {
            pool.submit(fetch_trailing_eps_bvps, sym, market_abb): sym
            for sym in cleaned
        }
        for fut in as_completed(future_to_symbol):
            sym = future_to_symbol[fut]
            try:
                results[sym] = fut.result()
            except Exception as exc:
                logger.debug("StockAnalysis batch fetch failed for %s: %s", sym, exc)
                results[sym] = {"eps": None, "book_value_per_share": None, "pe_ratio": None}

    return results
