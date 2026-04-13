"""
Scheduler setup — APScheduler configuration for recurring tasks.

Called from main.py lifespan to start/stop the scheduler.

Daily workflow (when PRICE_UPDATE_ENABLED):
  1. Price update runs at PRICE_UPDATE_HOUR:PRICE_UPDATE_MINUTE (Asia/Kuwait)
  2. Snapshot save runs SNAPSHOT_DELAY_MINUTES later (default 5 min)
     — ensures the snapshot reflects the freshly-fetched prices.
"""

import logging
import os
import sys
import tempfile
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_scheduler = None
_lock_fd = None  # file descriptor for cross-worker lock


def _run_daily_price_then_snapshot(user_id: int | None = None) -> dict:
    """
    Combined daily job: refresh prices, then save snapshot.

    If *user_id* is None (the default for the scheduler), runs for
    **every** user that has at least one stock — so all users benefit
    from the automated daily update.

    By running both in a single job we guarantee ordering
    (snapshot always uses the freshest prices).
    Also updates the in-memory tracking dicts in the cron API router
    so the /status endpoint reflects the last scheduler run.
    """
    import time
    from app.core.database import query_all
    from app.cron.price_updater import run_price_update
    from app.cron.snapshot_saver import run_snapshot_save

    # Determine which users to process
    if user_id is not None:
        user_ids = [user_id]
    else:
        rows = query_all(
            "SELECT DISTINCT user_id FROM stocks WHERE symbol IS NOT NULL AND symbol != ''"
        )
        user_ids = [int(r[0]) for r in rows] if rows else [1]
        logger.info("🔄 Scheduler: updating prices for %d user(s): %s", len(user_ids), user_ids)

    all_price_results = {}
    all_snapshot_results = {}

    for uid in user_ids:
        price_result = run_price_update(user_id=uid)
        snapshot_result = run_snapshot_save(user_id=uid)
        all_price_results[uid] = price_result
        all_snapshot_results[uid] = snapshot_result

    # Update the cron API status tracking so /status shows scheduler runs
    try:
        from app.api.v1.cron import _last_run, _last_snapshot_run
        _last_run.update({
            "timestamp": int(time.time()),
            "source": "scheduler",
            "user_ids": user_ids,
            "result": {
                uid: r.to_dict() if hasattr(r, "to_dict") else r
                for uid, r in all_price_results.items()
            },
        })
        _last_snapshot_run.update({
            "timestamp": int(time.time()),
            "source": "scheduler",
            "user_ids": user_ids,
            "result": all_snapshot_results,
        })
    except Exception:
        pass  # non-critical — don't let tracking break the job

    return {"price": all_price_results, "snapshot": all_snapshot_results}


def _acquire_scheduler_lock() -> bool:
    """
    Try to acquire an exclusive file lock so only ONE gunicorn worker
    (or one process) runs the scheduler.

    Returns True if lock acquired, False otherwise.
    """
    global _lock_fd
    lock_path = os.path.join(tempfile.gettempdir(), "portfolio_scheduler.lock")
    try:
        _lock_fd = open(lock_path, "w")
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(_lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd.write(str(os.getpid()))
        _lock_fd.flush()
        return True
    except (OSError, IOError):
        # Another worker already holds the lock
        if _lock_fd:
            _lock_fd.close()
            _lock_fd = None
        return False


def start_scheduler() -> None:
    """
    Initialize and start the APScheduler background scheduler.

    Uses a file lock to ensure only one gunicorn/uvicorn worker starts
    the scheduler (prevents duplicate job runs in multi-worker setups).

    Always schedules the stale extraction-job sweep.
    Adds the daily price-update + snapshot job when PRICE_UPDATE_ENABLED is set.
    """
    global _scheduler

    if not _acquire_scheduler_lock():
        logger.info("🕐 Scheduler skipped — another worker already owns it (pid %d)", os.getpid())
        return

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

    # ── News polling (adaptive: 15s market hours / 5m off-hours) ───
    try:
        from app.cron.news_poller import start_news_poller
        start_news_poller()
        logger.info("📰 Adaptive news poller started (15s market / 5m off-hours)")
    except Exception as exc:
        logger.warning("Could not start news poller: %s", exc)

    # ── Daily price update + snapshot save ────────────────────────
    if settings.PRICE_UPDATE_ENABLED:
        price_trigger = CronTrigger(
            hour=settings.PRICE_UPDATE_HOUR,
            minute=settings.PRICE_UPDATE_MINUTE,
            timezone="Asia/Kuwait",
        )
        _scheduler.add_job(
            _run_daily_price_then_snapshot,
            trigger=price_trigger,
            id="daily_price_and_snapshot",
            name="Daily price update + snapshot save",
            replace_existing=True,
        )
        logger.info(
            "🕐 Daily price update + snapshot scheduled — daily at %02d:%02d Asia/Kuwait",
            settings.PRICE_UPDATE_HOUR,
            settings.PRICE_UPDATE_MINUTE,
        )
    else:
        logger.info("⏸  Price scheduler disabled (PRICE_UPDATE_ENABLED=False)")

    _scheduler.start()
    logger.info("🕐 Scheduler started")


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler, news poller, and release the lock."""
    global _scheduler, _lock_fd
    # Stop news poller thread
    try:
        from app.cron.news_poller import stop_news_poller
        stop_news_poller()
    except Exception:
        pass
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        logger.info("🕐 Price scheduler stopped")
        _scheduler = None
    if _lock_fd is not None:
        try:
            _lock_fd.close()
        except Exception:
            pass
        _lock_fd = None


def get_scheduler():
    """Return the scheduler instance (or None if not started)."""
    return _scheduler
