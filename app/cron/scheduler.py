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

    Adds the daily price update job based on settings.
    """
    global _scheduler

    settings = get_settings()

    if not settings.PRICE_UPDATE_ENABLED:
        logger.info("⏸  Price scheduler disabled (PRICE_UPDATE_ENABLED=False)")
        return

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning(
            "apscheduler not installed — daily price scheduler will NOT run.\n"
            "  Install with: pip install apscheduler"
        )
        return

    from app.cron.price_updater import run_price_update

    _scheduler = BackgroundScheduler(daemon=True)
    trigger = CronTrigger(
        hour=settings.PRICE_UPDATE_HOUR,
        minute=settings.PRICE_UPDATE_MINUTE,
        timezone="Asia/Kuwait",
    )
    _scheduler.add_job(
        run_price_update,
        trigger=trigger,
        id="daily_price_update",
        name="Daily price update",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "🕐 Price scheduler started — runs daily at %02d:%02d Asia/Kuwait",
        settings.PRICE_UPDATE_HOUR,
        settings.PRICE_UPDATE_MINUTE,
    )


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
