"""
Cron / Scheduler API v1 — price update + snapshot save triggers + status.

Protected by CRON_SECRET_KEY in the X-Cron-Key header.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, Query, HTTPException
from starlette.concurrency import run_in_threadpool

from app.api.deps import require_admin

from app.core.config import get_settings
from app.core.database import query_all
from app.cron.job_locks import run_with_job_lock
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
) -> None:
    """Accept the secret only via Header ``X-Cron-Key``."""
    secret = settings.CRON_SECRET_KEY
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="CRON_SECRET_KEY is not configured on the server.",
        )
    if x_cron_key != secret:
        raise HTTPException(status_code=403, detail="Invalid cron key.")


@router.post("/update-prices")
async def trigger_price_update(
    x_cron_key: Optional[str] = Header(None, alias="X-Cron-Key"),
    user_id: int = Query(0, description="User whose stocks to update (0 = all users)"),
    only_holdings: bool = Query(True, description="Only update stocks with positive holdings"),
):
    """
    Trigger a full price refresh.

    Pass CRON_SECRET_KEY as Header ``X-Cron-Key``.
    Use ``user_id=0`` (default) to update prices for **all** users.
    """
    _verify_cron_key(x_cron_key)

    def _run():
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

    try:
        return await run_in_threadpool(lambda: run_with_job_lock("daily_price_and_snapshot", _run))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/status")
async def cron_status(_admin=Depends(require_admin)):
    """Return the last price-update and snapshot run info (admin only)."""
    last_fundamentals_run = None
    try:
        from app.cron.fundamentals_updater import get_last_run as get_fundamentals_last_run

        last_fundamentals_run = get_fundamentals_last_run() or None
    except Exception:
        last_fundamentals_run = None

    return {
        "status": "ok",
        "cron_key_configured": bool(settings.CRON_SECRET_KEY),
        "scheduler_enabled": settings.PRICE_UPDATE_ENABLED,
        "schedule": f"{settings.PRICE_UPDATE_HOUR:02d}:{settings.PRICE_UPDATE_MINUTE:02d} Asia/Kuwait",
        "last_price_update": _last_run if _last_run else None,
        "last_snapshot_save": _last_snapshot_run if _last_snapshot_run else None,
        "last_fundamentals_refresh": last_fundamentals_run,
    }


@router.post("/save-snapshot")
async def trigger_snapshot_save(
    x_cron_key: Optional[str] = Header(None, alias="X-Cron-Key"),
    user_id: int = Query(0, description="User whose snapshot to save (0 = all users)"),
):
    """
    Trigger a portfolio snapshot save (same as the Save Snapshot button).

    Pass CRON_SECRET_KEY as Header ``X-Cron-Key``.
    Use ``user_id=0`` (default) to save snapshots for **all** users.
    """
    _verify_cron_key(x_cron_key)

    from app.cron.snapshot_saver import run_snapshot_save

    def _run():
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

    try:
        return await run_in_threadpool(lambda: run_with_job_lock("daily_price_and_snapshot", _run))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/update-prices-and-snapshot")
async def trigger_price_update_and_snapshot(
    x_cron_key: Optional[str] = Header(None, alias="X-Cron-Key"),
    user_id: int = Query(0, description="User whose stocks to update and snapshot to save (0 = all users)"),
):
    """
    Trigger a full price refresh followed by a snapshot save.

    This is the same as the daily scheduled job — useful for manual testing
    or external cron services.
    Use ``user_id=0`` (default) to process **all** users.
    """
    _verify_cron_key(x_cron_key)

    from app.cron.fundamentals_updater import run_tickerchart_fundamentals_update
    from app.cron.snapshot_saver import run_snapshot_save

    def _run():
        fundamentals_result = run_tickerchart_fundamentals_update()

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
            "message": (
                f"Fundamentals refreshed, prices updated ({total_updated}/{total_found}), "
                f"snapshots saved for {len(user_ids)} user(s)"
            ),
            "data": {
                "fundamentals": fundamentals_result,
                "prices": all_price_results,
                "snapshots": all_snapshot_results,
            },
        }

    try:
        return await run_in_threadpool(lambda: run_with_job_lock("daily_price_and_snapshot", _run))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/notify-portfolio-updates")
async def trigger_portfolio_alerts(
    x_cron_key: Optional[str] = Header(None, alias="X-Cron-Key"),
    user_id: int = Query(0, description="User to notify (0 = all users with stocks)"),
):
    """
    Manually dispatch portfolio-update push notifications.

    The same dispatch is invoked automatically after the daily price+snapshot
    job; this endpoint is for manual testing / re-runs.
    Honors per-user notification preferences (``dailyPriceUpdates`` and
    ``portfolioUpdates``).
    """
    _verify_cron_key(x_cron_key)

    from app.services.portfolio_alerts import notify_portfolio_updates_for_users

    user_ids = _resolve_user_ids(user_id)
    logger.info("📲 Portfolio alert dispatch triggered for user_ids=%s", user_ids)
    result = notify_portfolio_updates_for_users(user_ids)
    return {
        "status": "ok",
        "message": f"Dispatched portfolio alerts to {len(user_ids)} user(s); {result.get('total_sent', 0)} push(es) sent",
        "data": result,
    }


@router.post("/update-fundamentals")
async def trigger_fundamentals_update(
    x_cron_key: Optional[str] = Header(None, alias="X-Cron-Key"),
):
    """
    Refresh Eagle Eye fundamentals (PE, EPS, BVPS) for the entire universe.

    Reads fresh data from StockAnalysis/TickerChart and upserts into
    ``ml_fundamentals``. Call this once on a fresh deployment to ensure
    B/V and P/E columns are populated before the nightly scheduler fires.
    """
    _verify_cron_key(x_cron_key)

    from app.cron.fundamentals_updater import run_tickerchart_fundamentals_update

    logger.info("📘 Fundamentals update triggered via API")
    result = run_tickerchart_fundamentals_update()
    return {
        "status": "ok" if result.get("success") else "error",
        "message": (
            f"Fundamentals refreshed: {result.get('upserted', 0)} upserted, "
            f"{result.get('skipped_existing', 0)} skipped, "
            f"{result.get('failed', 0)} failed"
        ),
        "data": result,
    }


@router.post("/eagle-eye-recompute")
async def trigger_eagle_eye_recompute(
    x_cron_key: Optional[str] = Header(None, alias="X-Cron-Key"),
    dna_refresh: bool = Query(False, description="Also rebuild DNA profiles (slow)"),
):
    """
    Trigger a full Eagle Eye nightly recompute (OHLCV fetch + ratings).

    Equivalent to the scheduled 14:05 job. Use this to force a refresh
    immediately after deploying a code change or after populating the
    fundamentals table for the first time.
    """
    _verify_cron_key(x_cron_key)

    import threading
    from app.services.eagle_eye.ingest import run_nightly_recompute

    logger.info("👁️ Eagle Eye recompute triggered via API (dna_refresh=%s)", dna_refresh)

    def _run() -> None:
        try:
            run_nightly_recompute(dna_refresh=dna_refresh, verbose=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("👁️ Eagle Eye recompute via API failed: %s", exc)

    threading.Thread(target=_run, daemon=True).start()

    return {
        "status": "accepted",
        "message": (
            "Eagle Eye recompute started in background. "
            "Ratings will update progressively over the next ~20 minutes."
        ),
    }


@router.post("/trend-hold-recompute")
async def trigger_trend_hold_recompute(
    x_cron_key: Optional[str] = Header(None, alias="X-Cron-Key"),
):
    """
    Trigger a full trend-hold engine scan + Trend-Hold Book paper-trading
    step immediately. Equivalent to the scheduled 14:15 + 14:18 Asia/Kuwait
    jobs. Use this to force a run right now -- e.g. right after deploying a
    code change, without waiting for the next scheduled cycle -- and to
    check the returned counts/log for what actually happened.
    """
    _verify_cron_key(x_cron_key)

    import threading

    logger.info("📈 Trend-hold recompute triggered via API")

    def _run() -> None:
        try:
            from app.services.eagle_eye_v2.trend_hold_batch import run_trend_hold_scan
            scan_summary = run_trend_hold_scan()
            logger.info("📈 Trend-hold scan via API: %s", scan_summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("📈 Trend-hold scan via API failed: %s", exc)
            return
        try:
            from app.services.eagle_eye_v2.trend_hold_book import run_trend_hold_book_step
            book_summary = run_trend_hold_book_step()
            logger.info("📈 Trend-hold book step via API: %s", book_summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("📈 Trend-hold book step via API failed: %s", exc)

    threading.Thread(target=_run, daemon=True).start()

    return {
        "status": "accepted",
        "message": (
            "Trend-hold scan + book step started in background. "
            "Check server logs or GET /api/v1/trend-hold-book/portfolio in ~1 minute."
        ),
    }
