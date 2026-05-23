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
import threading
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_scheduler = None
_lock_fd = None  # file descriptor for cross-worker lock


def _queue_portfolio_news_alerts() -> None:
    """Run async portfolio-news alerts from a sync scheduler context."""
    import asyncio
    from app.services.portfolio_alerts import notify_portfolio_news_alerts

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(notify_portfolio_news_alerts())
        return
    except RuntimeError:
        pass

    def _runner() -> None:
        try:
            asyncio.run(notify_portfolio_news_alerts())
        except Exception as exc:
            logger.warning("Portfolio news alert task failed: %s", exc)

    threading.Thread(target=_runner, daemon=True).start()


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

    # Fire portfolio-update push notifications (best-effort; never blocks
    # the daily job). Honors per-user notification preferences.
    try:
        from app.services.portfolio_alerts import notify_portfolio_updates_for_users
        alerts_result = notify_portfolio_updates_for_users(user_ids)
        logger.info(
            "📲 Portfolio alerts dispatched: %d push(es) sent across %d user(s)",
            alerts_result.get("total_sent", 0),
            len(user_ids),
        )
    except Exception as exc:
        logger.warning("Portfolio alert dispatch failed: %s", exc)

    try:
        _queue_portfolio_news_alerts()
        logger.info("📰 Queued portfolio news alert dispatcher")
    except Exception as exc:
        logger.warning("Portfolio news alert queue failed: %s", exc)

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


def _run_daily_technical_batch() -> dict:
    """Daily job: run technical scoring across the Kuwait stock universe."""
    settings = get_settings()
    from app.services.technical_batch_service import run_batch_sync

    logger.info(
        "📊 Scheduler: starting daily technical batch (%02d:%02d Asia/Kuwait)",
        settings.TECHNICAL_BATCH_HOUR,
        settings.TECHNICAL_BATCH_MINUTE,
    )

    try:
        result = run_batch_sync(
            triggered_by="scheduler",
            segment=settings.TECHNICAL_BATCH_SEGMENT,
            max_concurrency=settings.TECHNICAL_BATCH_MAX_CONCURRENCY,
        )
        logger.info("📊 Scheduler: technical batch finished: %s", result.get("message"))
        return result
    except Exception as exc:
        logger.warning("📊 Scheduler: technical batch failed: %s", exc)
        return {"status": "error", "message": str(exc)}


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

    # ── Nightly data-retention sweep (03:00 Asia/Kuwait) ────────
    try:
        from app.services.compliance_service import enforce_data_retention
        _scheduler.add_job(
            enforce_data_retention,
            trigger=CronTrigger(hour=3, minute=0, timezone="Asia/Kuwait"),
            id="data_retention_sweep",
            name="Nightly audit data retention",
            kwargs={"retention_days": 365},
            misfire_grace_time=3600,
            replace_existing=True,
        )
        logger.info("🗑️  Nightly data retention scheduled (03:00 Asia/Kuwait, 365-day window)")
    except Exception as exc:
        logger.warning("Could not schedule data retention sweep: %s", exc)

    # ── Daily technical universe scoring ───────────────────────────
    if settings.TECHNICAL_BATCH_ENABLED:
        technical_trigger = CronTrigger(
            hour=settings.TECHNICAL_BATCH_HOUR,
            minute=settings.TECHNICAL_BATCH_MINUTE,
            timezone="Asia/Kuwait",
        )
        _scheduler.add_job(
            _run_daily_technical_batch,
            trigger=technical_trigger,
            id="daily_technical_batch",
            name="Daily technical universe batch scoring",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        logger.info(
            "📊 Daily technical batch scheduled — daily at %02d:%02d Asia/Kuwait",
            settings.TECHNICAL_BATCH_HOUR,
            settings.TECHNICAL_BATCH_MINUTE,
        )
    else:
        logger.info("⏸  Technical batch scheduler disabled (TECHNICAL_BATCH_ENABLED=False)")

    # ── Eagle Eye nightly recompute (Sun–Thu, after Boursa close) ────
    try:
        from app.services.eagle_eye.ingest import run_nightly_recompute

        def _run_eagle_eye_intraday_refresh() -> None:
            """Intraday ratings refresh — runs near market close (13:15) to
            capture late-session signals before the post-close recompute."""
            run_nightly_recompute(dna_refresh=False, verbose=False)

        def _run_eagle_eye_daily() -> None:
            """Nightly incremental OHLCV fetch + ratings (no DNA — that runs weekly on Sundays)."""
            run_nightly_recompute(dna_refresh=False, verbose=False)

        def _run_eagle_eye_dna() -> None:
            """Weekly full recompute including DNA profiles (Sundays)."""
            run_nightly_recompute(dna_refresh=True, verbose=False)

        # Sun–Thu at 13:15 Asia/Kuwait — intraday refresh near Boursa close
        _scheduler.add_job(
            _run_eagle_eye_intraday_refresh,
            trigger=CronTrigger(
                day_of_week="sun,mon,tue,wed,thu",
                hour=13,
                minute=15,
                timezone="Asia/Kuwait",
            ),
            id="eagle_eye_intraday_refresh",
            name="Eagle Eye intraday refresh",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

        # Daily at 14:05 Asia/Kuwait Sun–Thu (Boursa closes ~13:45)
        _scheduler.add_job(
            _run_eagle_eye_daily,
            trigger=CronTrigger(
                day_of_week="sun,mon,tue,wed,thu",
                hour=14,
                minute=5,
                timezone="Asia/Kuwait",
            ),
            id="eagle_eye_nightly",
            name="Eagle Eye nightly recompute",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

        # Sunday at 14:30 Asia/Kuwait — full DNA rebuild (after daily job)
        _scheduler.add_job(
            _run_eagle_eye_dna,
            trigger=CronTrigger(
                day_of_week="sun",
                hour=14,
                minute=30,
                timezone="Asia/Kuwait",
            ),
            id="eagle_eye_weekly_dna",
            name="Eagle Eye weekly DNA refresh",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

        # ── Simulator daily run (Sun–Thu 14:20 — after rating recompute) ──
        def _run_eagle_eye_simulator() -> None:
            """Paper trading simulator: exits → entries → snapshot for all 3 strategies."""
            try:
                from app.services.eagle_eye.simulator import get_engine
                result = get_engine().run_daily()
                logger.info("📈 Simulator daily run complete: %s", result)
            except Exception as _exc:
                logger.warning("📈 Simulator daily run failed: %s", _exc)

        _scheduler.add_job(
            _run_eagle_eye_simulator,
            trigger=CronTrigger(
                day_of_week="sun,mon,tue,wed,thu",
                hour=14,
                minute=20,
                timezone="Asia/Kuwait",
            ),
            id="eagle_eye_simulator_daily",
            name="Eagle Eye paper trading simulator",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

        logger.info(
            "Eagle Eye jobs scheduled "
            "(Sun–Thu 13:15 intraday; Sun–Thu 14:05 nightly; DNA rebuild Sun 14:30; Simulator 14:20 Asia/Kuwait)"
        )
    except Exception as exc:
        logger.warning("Could not schedule Eagle Eye jobs: %s", exc)

    # ── Phase 3: ML shadow runner (Sun–Thu 14:30) ─────────────────────────
    try:
        from app.services.eagle_eye.ml.shadow_runner import run_shadow_scoring
        from app.services.eagle_eye.ml.auto_disable_monitor import run_auto_disable_check
        from app.services.eagle_eye.ml.weekly_review import run_weekly_review

        def _run_ml_shadow_scoring() -> None:
            try:
                summary = run_shadow_scoring()
                logger.info("🤖 ML shadow scoring complete: %s", summary)
            except Exception as _exc:
                logger.warning("🤖 ML shadow scoring failed: %s", _exc)

        def _run_ml_auto_disable() -> None:
            try:
                result = run_auto_disable_check()
                if result.get("triggered"):
                    logger.warning("🚨 ML auto-disable triggered: %s", result)
                else:
                    logger.info("✅ ML auto-disable check passed")
            except Exception as _exc:
                logger.warning("ML auto-disable check failed: %s", _exc)

        def _run_ml_weekly_review() -> None:
            try:
                path = run_weekly_review()
                logger.info("📋 ML weekly review report: %s", path)
            except Exception as _exc:
                logger.warning("ML weekly review failed: %s", _exc)

        # Shadow scoring — Sun–Thu at ML_SHADOW_HOUR:ML_SHADOW_MINUTE (default 14:30)
        _scheduler.add_job(
            _run_ml_shadow_scoring,
            trigger=CronTrigger(
                day_of_week="sun,mon,tue,wed,thu",
                hour=settings.ML_SHADOW_HOUR,
                minute=settings.ML_SHADOW_MINUTE,
                timezone="Asia/Kuwait",
            ),
            id="ml_shadow_runner",
            name="ML shadow scoring runner",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

        # Auto-disable monitor — Sun–Thu at 14:45 (runs after shadow scoring)
        _scheduler.add_job(
            _run_ml_auto_disable,
            trigger=CronTrigger(
                day_of_week="sun,mon,tue,wed,thu",
                hour=14,
                minute=45,
                timezone="Asia/Kuwait",
            ),
            id="ml_auto_disable_monitor",
            name="ML auto-disable monitor",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

        # Weekly review — Sunday at 15:00
        _scheduler.add_job(
            _run_ml_weekly_review,
            trigger=CronTrigger(
                day_of_week=settings.ML_WEEKLY_REVIEW_DAY,
                hour=15,
                minute=0,
                timezone="Asia/Kuwait",
            ),
            id="ml_weekly_review",
            name="ML weekly flagged-stock review",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

        logger.info(
            "🤖 ML Phase 3 jobs scheduled "
            "(shadow %02d:%02d Sun–Thu; auto-disable 14:45 Sun–Thu; weekly review %s 15:00 Asia/Kuwait)",
            settings.ML_SHADOW_HOUR,
            settings.ML_SHADOW_MINUTE,
            settings.ML_WEEKLY_REVIEW_DAY,
        )
    except Exception as exc:
        logger.warning("Could not schedule ML Phase 3 jobs: %s", exc)

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
