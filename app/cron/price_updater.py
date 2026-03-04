"""
Price Updater — the actual update logic invoked by the scheduler.

Delegates to ``services/price_service.py`` for the heavy lifting.
This module adds logging, state tracking, and error recovery.
"""

import logging
import time

from app.services.price_service import update_all_prices

logger = logging.getLogger(__name__)

# Module-level state for last run info
_last_run: dict = {}


def run_price_update(user_id: int = 1) -> dict:
    """
    Execute a full price update cycle.

    Returns:
        dict with timestamp and result summary.
    """
    logger.info("⏰ Scheduled price update starting (user_id=%d)…", user_id)

    try:
        result = update_all_prices(user_id=user_id)

        run_info = {
            "timestamp": int(time.time()),
            "result": result.to_dict(),
            "success": True,
        }

        logger.info(
            "⏰ Scheduled update done: %d/%d updated in %.1fs",
            result.updated, result.stocks_found, result.elapsed_sec,
        )

    except Exception as exc:
        run_info = {
            "timestamp": int(time.time()),
            "error": str(exc),
            "success": False,
        }
        logger.error("⏰ Scheduled update FAILED: %s", exc)

    _last_run.update(run_info)
    return run_info


def get_last_run() -> dict:
    """Return info about the last price update run."""
    return dict(_last_run)
