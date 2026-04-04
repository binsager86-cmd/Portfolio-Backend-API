"""
Cron / Scheduler API v1 — price update + snapshot save triggers + status.

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
_last_snapshot_run: dict = {}


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
    """Return the last price-update and snapshot run info (no auth required)."""
    return {
        "status": "ok",
        "cron_key_configured": bool(settings.CRON_SECRET_KEY),
        "scheduler_enabled": settings.PRICE_UPDATE_ENABLED,
        "schedule": f"{settings.PRICE_UPDATE_HOUR:02d}:{settings.PRICE_UPDATE_MINUTE:02d} Asia/Kuwait",
        "last_price_update": _last_run if _last_run else None,
        "last_snapshot_save": _last_snapshot_run if _last_snapshot_run else None,
    }


@router.post("/save-snapshot")
async def trigger_snapshot_save(
    x_cron_key: Optional[str] = Header(None, alias="X-Cron-Key"),
    key: Optional[str] = Query(None),
    user_id: int = Query(1, description="User whose snapshot to save"),
):
    """
    Trigger a portfolio snapshot save (same as the Save Snapshot button).

    Pass CRON_SECRET_KEY as Header ``X-Cron-Key`` or query ``?key=``.
    """
    _verify_cron_key(x_cron_key, key)

    from app.cron.snapshot_saver import run_snapshot_save

    logger.info("📸 Snapshot save triggered via API (user_id=%d)", user_id)
    result = run_snapshot_save(user_id=user_id)

    _last_snapshot_run.update(result)

    if result.get("success"):
        return {
            "status": "ok",
            "message": f"Snapshot {result.get('action', 'saved')} for {result.get('snapshot_date')} — value: {result.get('portfolio_value', 0):.3f} KWD",
            "data": result,
        }
    else:
        raise HTTPException(status_code=500, detail=result.get("error", "Snapshot save failed"))


@router.post("/update-prices-and-snapshot")
async def trigger_price_update_and_snapshot(
    x_cron_key: Optional[str] = Header(None, alias="X-Cron-Key"),
    key: Optional[str] = Query(None),
    user_id: int = Query(1, description="User whose stocks to update and snapshot to save"),
):
    """
    Trigger a full price refresh followed by a snapshot save.

    This is the same as the daily scheduled job — useful for manual testing
    or external cron services.
    """
    _verify_cron_key(x_cron_key, key)

    from app.cron.snapshot_saver import run_snapshot_save

    logger.info("🚀 Price update + snapshot triggered via API (user_id=%d)", user_id)

    price_result = update_all_prices(user_id=user_id)
    _last_run.update({
        "timestamp": int(time.time()),
        "result": price_result.to_dict(),
    })

    snapshot_result = run_snapshot_save(user_id=user_id)
    _last_snapshot_run.update(snapshot_result)

    return {
        "status": "ok",
        "message": f"Prices updated ({price_result.updated}/{price_result.stocks_found}), snapshot {snapshot_result.get('action', 'saved')}",
        "data": {
            "prices": price_result.to_dict(),
            "snapshot": snapshot_result,
        },
    }
