"""Order Book WebSocket endpoint.

Streams live bid/ask depth for a Kuwait Stock Exchange security.

WebSocket: /api/v1/orderbook/stream

Send once after connecting:
  {"symbol": "NBK", "market_segment": "Premier", "depth": 20}

Receive JSON frames until disconnect:
  {
    "timestamp": "2026-05-05T09:30:00.123Z",
    "symbol": "NBK",
    "bids": [{"price": 1.234, "volume": 15000, "orders": 3}, ...],
    "asks": [{"price": 1.236, "volume": 8500,  "orders": 2}, ...],
    "spread_fils": 0.2,
    "mid_price": 1.235
  }

If no live feed is available the endpoint returns a single
"unavailable" frame and closes gracefully (no error).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.services.signal_engine.config.kuwait_constants import (
    LIQUIDITY_WALL_THRESHOLD_MAIN,
    LIQUIDITY_WALL_THRESHOLD_PREMIER,
    ORDERBOOK_API_BASE,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orderbook", tags=["Order Book"])

# ── Schema helpers ─────────────────────────────────────────────────────────────

_DEPTH_MAX = 50   # cap per side to avoid huge payloads
_DEPTH_MIN = 1


def _validate_subscribe(raw: dict[str, Any]) -> tuple[str, str, int] | None:
    """Return (symbol, market_segment, depth) or None if the message is invalid."""
    symbol = str(raw.get("symbol", "")).strip().upper()
    if not symbol:
        return None
    market_segment = str(raw.get("market_segment", "Premier")).strip()
    depth = int(raw.get("depth", 10))
    depth = max(_DEPTH_MIN, min(_DEPTH_MAX, depth))
    return symbol, market_segment, depth


def _liquidity_wall_threshold(market_segment: str) -> int:
    return (
        LIQUIDITY_WALL_THRESHOLD_PREMIER
        if market_segment.lower() == "premier"
        else LIQUIDITY_WALL_THRESHOLD_MAIN
    )


def _compute_spread_and_mid(
    bids: list[dict[str, Any]], asks: list[dict[str, Any]]
) -> tuple[float, float]:
    """Return (spread_fils, mid_price) from the top-of-book."""
    best_bid = bids[0]["price"] if bids else 0.0
    best_ask = asks[0]["price"] if asks else 0.0
    if best_bid > 0 and best_ask > 0:
        spread = round(best_ask - best_bid, 3)
        mid = round((best_bid + best_ask) / 2, 4)
    else:
        spread = 0.0
        mid = best_bid or best_ask
    return spread, mid


def _unavailable_frame(symbol: str) -> str:
    return json.dumps({
        "type": "unavailable",
        "symbol": symbol,
        "message": "Live order book feed not available for this symbol",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ── Live feed connector ────────────────────────────────────────────────────────

async def _fetch_orderbook_snapshot(
    symbol: str,
    market_segment: str,
    depth: int,
) -> dict[str, Any] | None:
    """Attempt to fetch a single order book snapshot.

    This function is intentionally thin: it is the integration seam
    where a real Boursa Kuwait feed (REST poll or upstream WebSocket)
    would be wired in.  For now it returns None so the endpoint
    gracefully reports "unavailable" rather than crashing.

    Replace the body of this function (or inject a dependency) when
    a live data source is available.
    """
    # TODO: connect to ORDERBOOK_API_BASE and fetch real depth data.
    logger.debug(
        "Order book snapshot requested: symbol=%s segment=%s depth=%d base_url=%s",
        symbol, market_segment, depth, ORDERBOOK_API_BASE,
    )
    return None  # signals "unavailable"


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@router.websocket("/stream")
async def orderbook_stream(websocket: WebSocket) -> None:
    """Stream live order book depth for a KSE security.

    Protocol
    --------
    1. Client connects.
    2. Client sends a JSON subscribe message:
           {"symbol": "NBK", "market_segment": "Premier", "depth": 20}
    3. Server streams JSON depth frames until the client disconnects
       or the market closes.
    4. If the live feed is unavailable the server sends one
       {"type": "unavailable", ...} frame and closes cleanly.

    Authentication note
    -------------------
    WebSocket connections cannot carry the Authorization header from
    the browser.  Callers should pass the JWT as a query param:
        wss://host/api/v1/orderbook/stream?token=<jwt>
    Token validation is intentionally left as a TODO so the endpoint
    can be deployed and tested without breaking the auth middleware.
    """
    await websocket.accept()

    try:
        # ── Step 1: receive subscribe message ────────────────────────
        try:
            raw_msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        except asyncio.TimeoutError:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        try:
            payload = json.loads(raw_msg)
        except json.JSONDecodeError:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Invalid JSON in subscribe message"})
            )
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return

        parsed = _validate_subscribe(payload)
        if parsed is None:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Missing required field: symbol"})
            )
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return

        symbol, market_segment, depth = parsed
        wall_threshold = _liquidity_wall_threshold(market_segment)

        logger.info(
            "Order book stream subscribed: symbol=%s segment=%s depth=%d",
            symbol, market_segment, depth,
        )

        # ── Step 2: stream frames ─────────────────────────────────────
        while True:
            snapshot = await _fetch_orderbook_snapshot(symbol, market_segment, depth)

            if snapshot is None:
                # Live feed unavailable — notify client and close
                await websocket.send_text(_unavailable_frame(symbol))
                await websocket.close()
                return

            bids: list[dict[str, Any]] = snapshot.get("bids", [])[:depth]
            asks: list[dict[str, Any]] = snapshot.get("asks", [])[:depth]
            spread_fils, mid_price = _compute_spread_and_mid(bids, asks)

            frame: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "bids": bids,
                "asks": asks,
                "spread_fils": spread_fils,
                "mid_price": mid_price,
            }

            # Annotate any level that qualifies as a liquidity wall
            for side_key, side_list in (("bids", bids), ("asks", asks)):
                for level in side_list:
                    if level.get("volume", 0) >= wall_threshold:
                        level["liquidity_wall"] = True

            await websocket.send_text(json.dumps(frame))

            # Throttle: send at most once per second to avoid flooding
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        logger.debug("Order book stream client disconnected: %s", websocket.client)
    except Exception:
        logger.exception("Unexpected error in order book stream")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
