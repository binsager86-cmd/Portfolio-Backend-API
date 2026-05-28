"""
Daily TickerChart fundamentals updater.

Persists one point-in-time row per ticker/day into ``ml_fundamentals`` so
scanner fallback data is refreshed automatically even when historical sources
are sparse.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.core.database import exec_sql
from app.services import tickerchart_service as tc
from app.services.eagle_eye.adapter import TickerChartAdapter

logger = logging.getLogger(__name__)

_last_run: dict = {}


def _today_kuwait_iso() -> str:
    try:
        return datetime.now(ZoneInfo("Asia/Kuwait")).date().isoformat()
    except Exception:
        return date.today().isoformat()


def _round_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def run_tickerchart_fundamentals_update() -> dict:
    """Refresh today's PE/BVPS/EPS snapshot for the Eagle Eye universe."""
    started_at = time.time()
    disclosure_date = _today_kuwait_iso()
    source = "tickerchart_snapshot_daily"

    logger.info("📘 Daily fundamentals refresh starting (date=%s)", disclosure_date)

    try:
        universe = TickerChartAdapter().list_stocks()
    except Exception as exc:
        run_info = {
            "timestamp": int(time.time()),
            "success": False,
            "error": f"universe_load_failed: {exc}",
        }
        _last_run.update(run_info)
        logger.warning("📘 Daily fundamentals refresh failed to load universe: %s", exc)
        return run_info

    seen: set[str] = set()
    upserted = 0
    no_data = 0
    failed = 0

    for meta in universe:
        ticker = str(getattr(meta, "ticker", "") or "").upper().strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)

        try:
            pe_ratio = _round_or_none(tc.read_quotes_snapshot_pe(ticker, "KSE"), 4)
            eps = _round_or_none(tc.read_quotes_snapshot_ltm_eps(ticker, "KSE", price_divisor=1000.0), 6)
            bvps = _round_or_none(tc.read_quotes_snapshot_bvps(ticker, "KSE", price_divisor=1000.0), 6)
        except Exception as exc:
            failed += 1
            logger.debug("TickerChart snapshot read failed for %s: %s", ticker, exc)
            continue

        if pe_ratio is None and eps is None and bvps is None:
            no_data += 1
            continue

        try:
            exec_sql(
                """
                INSERT INTO ml_fundamentals (
                    stock_ticker,
                    disclosure_date,
                    period_end_date,
                    source,
                    pe_ratio,
                    eps,
                    book_value_per_share
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stock_ticker, disclosure_date, source)
                DO UPDATE SET
                    period_end_date = excluded.period_end_date,
                    pe_ratio = excluded.pe_ratio,
                    eps = excluded.eps,
                    book_value_per_share = excluded.book_value_per_share,
                    created_at = CURRENT_TIMESTAMP
                """,
                (
                    ticker,
                    disclosure_date,
                    disclosure_date,
                    source,
                    pe_ratio,
                    eps,
                    bvps,
                ),
            )
            upserted += 1
        except Exception as exc:
            failed += 1
            logger.debug("Fundamentals upsert failed for %s: %s", ticker, exc)

    run_info = {
        "timestamp": int(time.time()),
        "success": True,
        "date": disclosure_date,
        "source": source,
        "universe": len(seen),
        "upserted": upserted,
        "no_data": no_data,
        "failed": failed,
        "elapsed_sec": round(time.time() - started_at, 2),
    }
    _last_run.update(run_info)

    logger.info(
        "📘 Daily fundamentals refresh done: upserted=%d no_data=%d failed=%d elapsed=%.2fs",
        upserted,
        no_data,
        failed,
        run_info["elapsed_sec"],
    )
    return run_info


def get_last_run() -> dict:
    """Return info about the last fundamentals refresh run."""
    return dict(_last_run)
