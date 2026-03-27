"""
Scheduler setup — APScheduler configuration for recurring tasks.

Called from main.py lifespan to start/stop the scheduler.
"""

import logging
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_scheduler = None


def start_scheduler() -> None:
    """
    Initialize and start the APScheduler background scheduler.

    Always schedules the stale extraction-job sweep.
    Adds the daily price update job when PRICE_UPDATE_ENABLED is set.
    """
    global _scheduler

    settings = get_settings()

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        logger.warning(
            "apscheduler not installed — scheduler will NOT run.\n"
            "  Install with: pip install apscheduler"
        )
        return

    _scheduler = BackgroundScheduler(daemon=True)

    # ── Periodic stale extraction-job sweep (every 5 min) ────────
    try:
        from app.api.v1.fundamental import recover_stale_jobs
        _scheduler.add_job(
            recover_stale_jobs,
            trigger=IntervalTrigger(minutes=5),
            id="stale_extraction_sweep",
            name="Stale extraction job sweep",
            replace_existing=True,
        )
        logger.info("🔄 Stale extraction-job sweep scheduled (every 5 min)")
    except Exception as exc:
        logger.warning("Could not schedule stale-job sweep: %s", exc)

    # ── Daily price update (optional) ────────────────────────────
    if settings.PRICE_UPDATE_ENABLED:
        from app.cron.price_updater import run_price_update

        price_trigger = CronTrigger(
            hour=settings.PRICE_UPDATE_HOUR,
            minute=settings.PRICE_UPDATE_MINUTE,
            timezone="Asia/Kuwait",
        )
        _scheduler.add_job(
            run_price_update,
            trigger=price_trigger,
            id="daily_price_update",
            name="Daily price update",
            replace_existing=True,
        )
        logger.info(
            "🕐 Price update scheduled — daily at %02d:%02d Asia/Kuwait",
            settings.PRICE_UPDATE_HOUR,
            settings.PRICE_UPDATE_MINUTE,
        )
    else:
        logger.info("⏸  Price scheduler disabled (PRICE_UPDATE_ENABLED=False)")

    _scheduler.start()
    logger.info("🕐 Scheduler started")


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        logger.info("🕐 Price scheduler stopped")
        _scheduler = None


def get_scheduler():
    """Return the scheduler instance (or None if not started)."""
    return _scheduler
