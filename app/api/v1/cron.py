"""
Cron / Scheduler API v1 — price update + snapshot save triggers + status.

Protected by CRON_SECRET_KEY (header or query param).
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Header, Query, HTTPException

from app.core.config import get_settings
from app.core.database import query_all
from app.services.price_service import update_all_prices

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/cron", tags=["Cron / Scheduler"])

# ── In-memory last-run tracking ──────────────────────────────────────
_last_run: dict = {}
_last_snapshot_run: dict = {}


def _resolve_user_ids(user_id: int) -> list[int]:
    """Return a list of user IDs to process.

    ``user_id=0`` means **all users** that have at least one stock.
    Any positive value means just that single user.
    """
    if user_id > 0:
        return [user_id]
    rows = query_all(
        "SELECT DISTINCT user_id FROM stocks WHERE symbol IS NOT NULL AND symbol != ''"
    )
    return [int(r[0]) for r in rows] if rows else [1]


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
    user_id: int = Query(0, description="User whose stocks to update (0 = all users)"),
    only_holdings: bool = Query(True, description="Only update stocks with positive holdings"),
):
    """
    Trigger a full price refresh.

    Pass CRON_SECRET_KEY as Header ``X-Cron-Key`` or query ``?key=``.
    Use ``user_id=0`` (default) to update prices for **all** users.
    """
    _verify_cron_key(x_cron_key, key)

    user_ids = _resolve_user_ids(user_id)
    logger.info("🚀 Price update triggered for user_ids=%s", user_ids)

    all_results = {}
    total_updated = 0
    total_found = 0
    for uid in user_ids:
        result = update_all_prices(user_id=uid, only_with_holdings=only_holdings)
        all_results[uid] = result.to_dict()
        total_updated += result.updated
        total_found += result.stocks_found

    _last_run.update({
        "timestamp": int(time.time()),
        "user_ids": user_ids,
        "result": all_results,
    })

    return {
        "status": "ok",
        "message": f"Updated {total_updated}/{total_found} prices across {len(user_ids)} user(s)",
        "data": all_results,
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
    user_id: int = Query(0, description="User whose snapshot to save (0 = all users)"),
):
    """
    Trigger a portfolio snapshot save (same as the Save Snapshot button).

    Pass CRON_SECRET_KEY as Header ``X-Cron-Key`` or query ``?key=``.
    Use ``user_id=0`` (default) to save snapshots for **all** users.
    """
    _verify_cron_key(x_cron_key, key)

    from app.cron.snapshot_saver import run_snapshot_save

    user_ids = _resolve_user_ids(user_id)
    logger.info("📸 Snapshot save triggered via API for user_ids=%s", user_ids)

    all_results = {}
    for uid in user_ids:
        all_results[uid] = run_snapshot_save(user_id=uid)

    _last_snapshot_run.update({
        "timestamp": int(time.time()),
        "user_ids": user_ids,
        "result": all_results,
    })

    failures = [uid for uid, r in all_results.items() if not r.get("success")]
    if failures:
        return {
            "status": "partial" if len(failures) < len(user_ids) else "error",
            "message": f"Snapshot save failed for user(s): {failures}",
            "data": all_results,
        }
    return {
        "status": "ok",
        "message": f"Snapshots saved for {len(user_ids)} user(s)",
        "data": all_results,
    }


@router.post("/update-prices-and-snapshot")
async def trigger_price_update_and_snapshot(
    x_cron_key: Optional[str] = Header(None, alias="X-Cron-Key"),
    key: Optional[str] = Query(None),
    user_id: int = Query(0, description="User whose stocks to update and snapshot to save (0 = all users)"),
):
    """
    Trigger a full price refresh followed by a snapshot save.

    This is the same as the daily scheduled job — useful for manual testing
    or external cron services.
    Use ``user_id=0`` (default) to process **all** users.
    """
    _verify_cron_key(x_cron_key, key)

    from app.cron.snapshot_saver import run_snapshot_save

    user_ids = _resolve_user_ids(user_id)
    logger.info("🚀 Price update + snapshot triggered via API for user_ids=%s", user_ids)

    all_price_results = {}
    all_snapshot_results = {}
    total_updated = 0
    total_found = 0

    for uid in user_ids:
        price_result = update_all_prices(user_id=uid)
        all_price_results[uid] = price_result.to_dict()
        total_updated += price_result.updated
        total_found += price_result.stocks_found

        snapshot_result = run_snapshot_save(user_id=uid)
        all_snapshot_results[uid] = snapshot_result

    _last_run.update({
        "timestamp": int(time.time()),
        "user_ids": user_ids,
        "result": all_price_results,
    })
    _last_snapshot_run.update({
        "timestamp": int(time.time()),
        "user_ids": user_ids,
        "result": all_snapshot_results,
    })

    return {
        "status": "ok",
        "message": f"Prices updated ({total_updated}/{total_found}), snapshots saved for {len(user_ids)} user(s)",
        "data": {
            "prices": all_price_results,
            "snapshots": all_snapshot_results,
        },
    }
