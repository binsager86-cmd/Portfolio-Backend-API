"""Sector regime detector — Banking Lead-Lag Filter.

Kuwait banking stocks consistently lead/lag the broader market.
When the banking sector is trending strongly, non-banking mid-caps
tend to follow with a 1-3 session lag — boosting their momentum weight.

Uses a proxy basket of the 3 largest Kuwait banks by ADTV:
  NBK  (National Bank of Kuwait)
  KFH  (Kuwait Finance House)
  BOUBYAN (Boubyan Bank)

The composite trend score is the simple average of their individual
trend_raw outputs from compute_trend_score().
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Banking proxy basket — top 3 by ADTV on Premier Market
_BANKING_PROXY: list[str] = ["NBK", "KFH", "BOUBYAN"]

# Cache: {date_str → result_dict} — refreshed once per trading session
_cache: dict[str, dict[str, Any]] = {}


async def fetch_banking_index_regime(
    lookback_days: int = 30,
) -> dict[str, Any]:
    """Return the composite banking sector trend for the Lead-Lag filter.

    Result is cached per trading day — subsequent calls within the same
    day return instantly.

    Returns:
        {
            "trend_raw":  float,    # 0-100 composite trend score
            "available":  bool,     # False when data could not be fetched
            "symbols_used": list,   # which proxy stocks contributed
            "data_as_of": str,      # date string of latest bar used
        }
    """
    today = date.today().isoformat()
    if today in _cache:
        return _cache[today]

    result = await _compute_banking_regime(lookback_days)
    _cache.clear()
    _cache[today] = result
    return result


async def _compute_banking_regime(lookback_days: int) -> dict[str, Any]:
    """Fetch OHLCV for each proxy stock and compute composite trend score."""
    from app.services import tickerchart_service
    from app.services.signal_engine.models.technical.trend_score import compute_trend_score

    # Import indicator attachment lazily to avoid circular imports
    try:
        from app.services.indicators_service import attach_indicators
    except ImportError:
        logger.warning("indicators_service not available; banking regime unavailable")
        return _unavailable()

    to_date = date.today()
    # Fetch extra bars so indicators have warm-up data
    from_date = to_date - timedelta(days=lookback_days + 60)

    scores: list[float] = []
    symbols_used: list[str] = []
    data_as_of: str = ""

    for symbol in _BANKING_PROXY:
        try:
            rows = await tickerchart_service.fetch_ohlcv(
                base_symbol=symbol,
                market_abb="KSE",
                from_d=from_date,
                to_d=to_date,
                interval="day",
            )
            if not rows or len(rows) < 20:
                continue

            rows_with_indicators = attach_indicators(rows)
            trend_raw, _ = compute_trend_score(rows_with_indicators)
            scores.append(float(trend_raw))
            symbols_used.append(symbol)
            if rows_with_indicators:
                data_as_of = rows_with_indicators[-1].get("date", data_as_of)

        except Exception as exc:
            logger.debug("Banking regime: could not fetch %s — %s", symbol, exc)
            continue

    if not scores:
        return _unavailable()

    composite = round(sum(scores) / len(scores), 1)
    logger.debug(
        "Banking regime: composite_trend=%.1f  symbols=%s", composite, symbols_used
    )
    return {
        "trend_raw": composite,
        "available": True,
        "symbols_used": symbols_used,
        "data_as_of": data_as_of,
    }


def _unavailable() -> dict[str, Any]:
    return {
        "trend_raw": 0.0,
        "available": False,
        "symbols_used": [],
        "data_as_of": "",
    }
