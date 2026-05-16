"""
Market Data Service — fetches Kuwait (KSE) market data from TickerChart Live.

Replaces the legacy Playwright / Boursa Kuwait scraper with concurrent OHLCV
calls to the TickerChart API.  One authenticated request per listed stock is
made in parallel; day change is computed from the last two trading candles.

Results are cached daily in the market_data table (same schema as before).
"""

import asyncio
import json
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)



def _build_degraded_market_payload(trade_date: str, exception: Exception) -> dict:
    """Return a safe fallback payload when TickerChart and cache are unavailable."""
    return {
        "indices": [],
        "market_summary": {
            "gainers": 0,
            "losers": 0,
            "neutral": 0,
            "stock_gainers": 0,
            "stock_losers": 0,
        },
        "premier_summary": {},
        "main_summary": {},
        "top_gainers": [],
        "top_losers": [],
        "top_value": [],
        "sectors": [],
        "date": trade_date,
        "status": "unavailable",
        "_cached": False,
        "_stale": True,
        "_degraded": True,
        "_trade_date": trade_date,
        "_fetched_at": int(time.time()),
        "_error": "market_data_unavailable",
        "_error_type": exception.__class__.__name__,
    }


async def get_market_data(force_refresh: bool = False) -> dict:
    """
    Return cached market data for today, or fetch fresh from TickerChart if stale/missing.

    Cache strategy: one row per trade_date in market_data table.
    On weekends or holidays, returns the latest available cached data.
    """
    from app.core.database import query_one, exec_sql
    from app.data.stock_lists import KUWAIT_STOCKS
    from app.services.tickerchart_service import fetch_kse_market_snapshot

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

    # Fetch fresh data from TickerChart
    try:
        symbols = [s["symbol"] for s in KUWAIT_STOCKS]
        stock_name_map = {s["symbol"]: s["name"] for s in KUWAIT_STOCKS}
        data = await fetch_kse_market_snapshot(symbols, stock_name_map)
        data["_fetched_at"] = int(time.time())
        data["_trade_date"] = today

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
        logger.error("Market data fetch (TickerChart) failed: %s", e, exc_info=True)
        # Fall back to most recent cached data
        row = query_one(
            "SELECT data_json, fetched_at FROM market_data ORDER BY trade_date DESC, fetched_at DESC LIMIT 1"
        )
        if row:
            cached = json.loads(row["data_json"])
            cached["_cached"] = True
            cached["_stale"] = True
            cached["_fetched_at"] = row["fetched_at"]
            logger.warning("Serving stale cached market snapshot because live fetch failed")
            return cached
        logger.warning("No cached market snapshot available; serving degraded empty payload")
        return _build_degraded_market_payload(today, e)


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
