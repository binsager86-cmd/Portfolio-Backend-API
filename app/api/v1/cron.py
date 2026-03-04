"""
Cron / Scheduler API v1 — price update trigger + status.

Protected by CRON_SECRET_KEY (header or query param).
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Header, Query, HTTPException

from app.core.config import get_settings
from app.services.price_service import update_all_prices

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/cron", tags=["Cron / Scheduler"])

# ── In-memory last-run tracking ──────────────────────────────────────
_last_run: dict = {}


def _verify_cron_key(
    x_cron_key: Optional[str] = Header(None, alias="X-Cron-Key"),
    key: Optional[str] = Query(None),
) -> None:
    """Accept the secret via Header ``X-Cron-Key`` or Query ``?key=``."""
    secret = settings.CRON_SECRET_KEY
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="CRON_SECRET_KEY is not configured on the server.",
        )
    provided = x_cron_key or key
    if provided != secret:
        raise HTTPException(status_code=403, detail="Invalid cron key.")


@router.post("/update-prices")
async def trigger_price_update(
    x_cron_key: Optional[str] = Header(None, alias="X-Cron-Key"),
    key: Optional[str] = Query(None),
    user_id: int = Query(1, description="User whose stocks to update"),
    only_holdings: bool = Query(True, description="Only update stocks with positive holdings"),
):
    """
    Trigger a full price refresh.

    Pass CRON_SECRET_KEY as Header ``X-Cron-Key`` or query ``?key=``.
    """
    _verify_cron_key(x_cron_key, key)

    logger.info("🚀 Price update triggered (user_id=%d)", user_id)
    result = update_all_prices(user_id=user_id, only_with_holdings=only_holdings)

    _last_run.update({
        "timestamp": int(time.time()),
        "result": result.to_dict(),
    })

    return {
        "status": "ok",
        "message": f"Updated {result.updated}/{result.stocks_found} prices in {result.elapsed_sec:.1f}s",
        "data": result.to_dict(),
    }


@router.get("/status")
async def cron_status():
    """Return the last price-update run info (no auth required)."""
    return {
        "status": "ok",
        "cron_key_configured": bool(settings.CRON_SECRET_KEY),
        "scheduler_enabled": settings.PRICE_UPDATE_ENABLED,
        "schedule": f"{settings.PRICE_UPDATE_HOUR:02d}:{settings.PRICE_UPDATE_MINUTE:02d} Asia/Kuwait",
        "last_run": _last_run if _last_run else None,
    }
