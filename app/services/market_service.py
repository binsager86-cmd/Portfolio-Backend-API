"""
Market Data Service — scrapes Boursa Kuwait homepage for market summary data.

Uses Playwright (headless Chromium) to render the JS-heavy page and extract:
  • Market indices (Premier, Main, All-Share, BK Main 50)
  • Volume / Value / Trades
  • Top Gainers, Top Losers, Top Value
  • Sector indices
  • Market status (open/closed) and date

Results are cached daily in the market_data table.
"""

import json
import logging
import re
import time
from datetime import datetime

logger = logging.getLogger(__name__)

BOURSA_URL = "https://www.boursakuwait.com.kw/en"


def _parse_number(raw: str) -> float | None:
    """Parse a formatted number like '9,214.11' → 9214.11."""
    try:
        return float(raw.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _parse_change(text: str) -> dict:
    """Parse change string like '-42.19' and '%' like '-0.46%'."""
    return {
        "change": _parse_number(text) if text else None,
    }


def _parse_mw_movers(text: str) -> list[dict]:
    """Parse movers data from Market Watch page text.

    The mover table has 7 columns per row:
        Symbol, Last, Chg, Chg%, Value, Volume, Trades
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Find data start: right after the "Trades" header in the movers area
    start = None
    for i, line in enumerate(lines):
        if line == "Trades" and i > 30:
            start = i + 1
            break
    if start is None:
        return []

    items: list[dict] = []
    j = start
    while j + 6 < len(lines) and len(items) < 10:
        symbol = lines[j]
        # Stop at end-of-table markers
        if symbol in ("Watch List", "More Columns", "All Sectors") or symbol.startswith("Code"):
            break

        last_val = _parse_number(lines[j + 1])
        chg = _parse_number(lines[j + 2].replace("+", ""))
        if lines[j + 2].strip().startswith("-"):
            chg = -abs(chg) if chg else chg
        chg_pct = _parse_number(lines[j + 3].replace("%", "").replace("+", ""))
        if lines[j + 3].strip().startswith("-"):
            chg_pct = -abs(chg_pct) if chg_pct else chg_pct
        volume = _parse_number(lines[j + 5])

        items.append({
            "symbol": symbol,
            "last": last_val,
            "change": chg,
            "changePercent": chg_pct,
            "volume": volume,
        })
        j += 7

    return items


def _scrape_market_data() -> dict:
    """Launch headless browser, navigate to Boursa Kuwait, extract market data."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()

            # ── Step 1: Homepage for indices, summary, sectors ──
            page.goto(BOURSA_URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(10000)
            homepage_text = page.inner_text("body")

            # ── Step 2: Market Watch for All-Share movers ──
            page.goto(
                BOURSA_URL + "/securities/prices-and-screens/market-watch",
                wait_until="domcontentloaded",
                timeout=60000,
            )
            page.wait_for_timeout(12000)

            # Switch to "All-Share" market context
            allshare = page.get_by_text("All-Share", exact=True)
            if allshare.count() > 0:
                allshare.first.click()
                page.wait_for_timeout(5000)

            # Cycle through mover tabs and capture text for each
            movers: dict[str, list] = {}
            for tab_name, key in [
                ("Top Gainers", "top_gainers"),
                ("Top Losers", "top_losers"),
                ("Top Value", "top_value"),
            ]:
                tab = page.get_by_text(tab_name, exact=True)
                if tab.count() > 0:
                    tab.first.click()
                    page.wait_for_timeout(3000)
                    mw_text = page.inner_text("body")
                    movers[key] = _parse_mw_movers(mw_text)
        finally:
            browser.close()

    data = _parse_page_text(homepage_text)
    # Override movers with Market Watch All-Share data
    for key in ("top_gainers", "top_losers", "top_value"):
        if movers.get(key):
            data[key] = movers[key]
    return data


def _parse_page_text(text: str) -> dict:
    """Parse the rendered page text into structured market data."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    data: dict = {
        "indices": [],
        "market_summary": {},
        "premier_summary": {},
        "main_summary": {},
        "top_gainers": [],
        "top_losers": [],
        "top_value": [],
        "sectors": [],
        "date": None,
        "status": None,
    }

    # ── Extract date and status ──
    for line in lines:
        m = re.match(r"(\d{1,2}\s+\w+\s+\d{4})\s+(Open|Closed)", line, re.I)
        if m:
            data["date"] = m.group(1).strip()
            data["status"] = m.group(2).strip().lower()
            break

    # ── Extract indices (deduplicate by name, keep first valid match) ──
    index_names = ["Premier Market", "BK Main 50", "Main Market", "All-Share"]
    seen_indices: set[str] = set()
    i = 0
    while i < len(lines):
        for name in index_names:
            if lines[i] == name and name not in seen_indices and i + 1 < len(lines):
                value = _parse_number(lines[i + 1])
                if value is None:
                    break  # skip entries with no numeric value
                change = None
                change_pct = None
                # Next lines should be change and change%
                j = i + 2
                while j < min(i + 5, len(lines)):
                    val = lines[j]
                    if "%" in val:
                        pct_str = val.replace("%", "").replace("+", "").strip()
                        change_pct = _parse_number(pct_str)
                        break
                    else:
                        parsed = _parse_number(val.replace("+", ""))
                        if parsed is not None and change is None:
                            change = parsed
                            if val.strip().startswith("-"):
                                change = -abs(change)
                    j += 1
                seen_indices.add(name)
                data["indices"].append({
                    "name": name,
                    "value": value,
                    "change": change,
                    "changePercent": change_pct,
                })
                break
        i += 1

    # ── Extract Premier Market summary (Volume, Value, Trades, Market Cap) ──
    summary_labels = {
        "Volume": "volume",
        "Value": "value_traded",
        "Value Traded": "value_traded",
        "Trades": "trades",
        "Market Cap": "market_cap",
    }
    found_keys: set[str] = set()
    for label, key in summary_labels.items():
        if key in found_keys:
            continue
        for idx, line in enumerate(lines):
            if line == label and idx + 1 < len(lines):
                val = _parse_number(lines[idx + 1])
                if val is not None:
                    data["market_summary"][key] = val
                    found_keys.add(key)
                    break

    # ── Extract Premier Market detail section (tab-delimited lines) ──
    # Page has "Premier Market" followed by lines like "Trades\t9,814"
    premier_labels = {"Trades": "trades", "Volume": "volume",
                      "Value Traded": "value_traded", "Market Cap": "market_cap"}
    for idx, line in enumerate(lines):
        # The 3rd occurrence of "Premier Market" starts the detail section
        if line == "Premier Market" and idx + 1 < len(lines):
            # Check if next line contains tab-delimited stats
            next_line = lines[idx + 1]
            if "\t" in next_line:
                j = idx + 1
                while j < len(lines) and "\t" in lines[j]:
                    parts = lines[j].split("\t")
                    if len(parts) == 2:
                        lbl = parts[0].strip()
                        val = _parse_number(parts[1].strip())
                        if lbl in premier_labels and val is not None:
                            data["premier_summary"][premier_labels[lbl]] = val
                    j += 1
                break

    # ── Compute Main Market stats (overall minus Premier) ──
    overall = data["market_summary"]
    premier = data["premier_summary"]
    if premier:
        for key in ("trades", "volume", "value_traded"):
            ov = overall.get(key, 0) or 0
            pr = premier.get(key, 0) or 0
            data["main_summary"][key] = max(ov - pr, 0)

    # ── Extract Top Gainers / Losers / Value ──
    section_map = {
        "Top Gainers": "top_gainers",
        "Top Losers": "top_losers",
        "Top Value": "top_value",
    }

    for section_label, key in section_map.items():
        try:
            start = None
            for idx, line in enumerate(lines):
                if line == section_label:
                    start = idx + 1
                    break
            if start is None:
                continue

            # Skip header labels
            header_skip = {"Symbol", "Last", "Chg", "Chg%", "Volume", "Value"}
            while start < len(lines) and lines[start] in header_skip:
                start += 1

            items = []
            j = start
            while j < len(lines) and len(items) < 5:
                symbol = lines[j]
                # Stop if we hit another section
                if symbol in section_map or symbol.startswith("Sector") or symbol.startswith("Top "):
                    break
                if symbol.startswith("*"):
                    j += 1
                    continue

                # Read: last, chg, chg%, volume/value
                if j + 4 < len(lines):
                    last_val = _parse_number(lines[j + 1])
                    chg = _parse_number(lines[j + 2].replace("+", ""))
                    if lines[j + 2].strip().startswith("-"):
                        chg = -abs(chg) if chg else chg
                    chg_pct = _parse_number(lines[j + 3].replace("%", "").replace("+", ""))
                    if lines[j + 3].strip().startswith("-"):
                        chg_pct = -abs(chg_pct) if chg_pct else chg_pct
                    volume_or_value = _parse_number(lines[j + 4])
                    items.append({
                        "symbol": symbol,
                        "last": last_val,
                        "change": chg,
                        "changePercent": chg_pct,
                        "volume": volume_or_value,
                    })
                    j += 5
                else:
                    break

            data[key] = items
        except Exception:
            pass

    # ── Extract Sector Indices ──
    try:
        sector_start = None
        for idx, line in enumerate(lines):
            if line == "Sector Indices":
                sector_start = idx + 1
                break

        if sector_start is not None:
            # Skip headers
            header_skip = {"Sector", "Chg%", "Last"}
            while sector_start < len(lines) and lines[sector_start] in header_skip:
                sector_start += 1

            j = sector_start
            while j + 2 < len(lines) and len(data["sectors"]) < 15:
                name = lines[j]
                # Stop if we hit non-sector content
                if name.startswith("*") or name.startswith("Company") or name.startswith("View"):
                    break
                chg_pct = _parse_number(lines[j + 1].replace("+", ""))
                if lines[j + 1].strip().startswith("-"):
                    chg_pct = -abs(chg_pct) if chg_pct else chg_pct
                last_val = _parse_number(lines[j + 2])
                if last_val and last_val > 10:  # Sanity check: sector values > 10
                    data["sectors"].append({
                        "name": name,
                        "changePercent": chg_pct,
                        "last": last_val,
                    })
                    j += 3
                else:
                    break
    except Exception:
        pass

    # ── Count gainers / losers / neutral from sector data ──
    gainers_count = sum(1 for s in data["sectors"] if (s.get("changePercent") or 0) > 0)
    losers_count = sum(1 for s in data["sectors"] if (s.get("changePercent") or 0) < 0)
    neutral_count = sum(1 for s in data["sectors"] if (s.get("changePercent") or 0) == 0)
    data["market_summary"]["gainers"] = gainers_count
    data["market_summary"]["losers"] = losers_count
    data["market_summary"]["neutral"] = neutral_count

    # If top_gainers/losers are available, use their count for stock-level stats
    data["market_summary"]["stock_gainers"] = len(data.get("top_gainers", []))
    data["market_summary"]["stock_losers"] = len(data.get("top_losers", []))

    return data


def get_market_data(force_refresh: bool = False) -> dict:
    """
    Return cached market data for today, or scrape fresh if stale/missing.

    Cache strategy: one row per trade_date in market_data table.
    On weekends or holidays, returns the latest available cached data.
    """
    from app.core.database import query_one, exec_sql

    today = datetime.utcnow().strftime("%Y-%m-%d")

    if not force_refresh:
        row = query_one(
            "SELECT data_json, fetched_at FROM market_data WHERE trade_date = ? ORDER BY fetched_at DESC LIMIT 1",
            (today,),
        )
        if row:
            cached = json.loads(row["data_json"])
            cached["_cached"] = True
            cached["_fetched_at"] = row["fetched_at"]
            return cached

    # Scrape fresh data
    try:
        data = _scrape_market_data()
        data["_fetched_at"] = int(time.time())
        data["_trade_date"] = today

        # Append snapshot (preserve historical data)
        exec_sql(
            """
            INSERT INTO market_data (trade_date, data_json, fetched_at)
            VALUES (?, ?, ?)
            """,
            (today, json.dumps(data), int(time.time())),
        )
        data["_cached"] = False
        return data

    except Exception as e:
        logger.error("Market data scrape failed: %s", e, exc_info=True)
        # Fall back to most recent cached data
        row = query_one(
            "SELECT data_json, fetched_at FROM market_data ORDER BY trade_date DESC, fetched_at DESC LIMIT 1"
        )
        if row:
            cached = json.loads(row["data_json"])
            cached["_cached"] = True
            cached["_stale"] = True
            cached["_fetched_at"] = row["fetched_at"]
            return cached
        raise


def get_market_history(
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 30,
) -> list[dict]:
    """
    Return historical market snapshots (latest per trade date, most recent first).

    Parameters
    ----------
    start_date : optional YYYY-MM-DD
    end_date   : optional YYYY-MM-DD
    limit      : max rows (default 30)
    """
    from app.core.database import query_df

    conditions = []
    params: list = []
    if start_date:
        conditions.append("trade_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("trade_date <= ?")
        params.append(end_date)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # Get the latest snapshot per trade date
    df = query_df(
        f"""
        SELECT trade_date, data_json, fetched_at
        FROM market_data
        {where}
        ORDER BY trade_date DESC, fetched_at DESC
        """,
        tuple(params),
    )

    if df.empty:
        return []

    # Keep only the latest snapshot per trade date
    df = df.drop_duplicates(subset="trade_date", keep="first")
    df = df.head(limit)

    rows = []
    for _, r in df.iterrows():
        data = json.loads(r["data_json"])
        data["_trade_date"] = r["trade_date"]
        data["_fetched_at"] = r["fetched_at"]
        rows.append(data)

    return rows
