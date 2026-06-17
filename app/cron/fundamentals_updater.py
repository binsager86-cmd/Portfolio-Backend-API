"""
Daily TickerChart fundamentals updater.

Persists one point-in-time row per ticker/day into ``ml_fundamentals``.
EPS is refreshed from the TickerChart FactSet LTM feed daily (with
QuotesSnapShot EPS fallback when live EPS is unavailable).
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.core.config import get_settings
from app.core.database import exec_sql, query_all
from app.services.stockanalysis_service import fetch_trailing_eps_bvps_batch
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
    """Refresh today's PE/BVPS/EPS values for the Eagle Eye universe."""
    started_at = time.time()
    disclosure_date = _today_kuwait_iso()
    source = "stockanalysis_primary_daily"
    settings = get_settings()

    logger.info(
        "📘 Daily fundamentals refresh starting (date=%s, eps/bvps=StockAnalysis primary, pe=TickerChart price/eps)",
        disclosure_date,
    )

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
    skipped_existing = 0
    upserted = 0
    no_data = 0
    failed = 0
    eps_from_stockanalysis = 0
    bvps_from_stockanalysis = 0
    eps_missing = 0
    pe_from_price_eps = 0

    existing_rows = query_all(
        """
        SELECT stock_ticker, eps, book_value_per_share, pe_ratio
        FROM ml_fundamentals
        WHERE disclosure_date = ? AND source = ?
        """,
        (disclosure_date, source),
    )
    existing_today: dict[str, dict] = {}
    for r in existing_rows or []:
        t = str((r.get("stock_ticker") if hasattr(r, "get") else r[0]) or "").upper().strip()
        if not t:
            continue
        existing_today[t] = {
            "eps": (r.get("eps") if hasattr(r, "get") else r[1]),
            "book_value_per_share": (r.get("book_value_per_share") if hasattr(r, "get") else r[2]),
            "pe_ratio": (r.get("pe_ratio") if hasattr(r, "get") else r[3]),
        }

    to_fetch: list[str] = []

    for meta in universe:
        ticker = str(getattr(meta, "ticker", "") or "").upper().strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)

        ready = existing_today.get(ticker)
        if ready and ready.get("eps") is not None and ready.get("book_value_per_share") is not None and ready.get("pe_ratio") is not None:
            skipped_existing += 1
            continue

        to_fetch.append(ticker)

    sa_batch = fetch_trailing_eps_bvps_batch(
        to_fetch,
        "KSE",
        max_workers=settings.STOCKANALYSIS_MAX_WORKERS,
    )

    read_last_price = getattr(tc, "read_quotes_snapshot_last_price", None)

    for ticker in to_fetch:
        pe_snapshot = None
        last_price = None

        try:
            pe_snapshot = _round_or_none(tc.read_quotes_snapshot_pe(ticker, "KSE"), 3)
        except Exception as exc:
            logger.debug("TickerChart snapshot PE read failed for %s: %s", ticker, exc)

        if callable(read_last_price):
            try:
                last_price = _round_or_none(read_last_price(ticker, "KSE", price_divisor=1000.0), 6)
            except Exception as exc:
                logger.debug("TickerChart snapshot last-price read failed for %s: %s", ticker, exc)

        sa = sa_batch.get(ticker) or {}
        eps_stockanalysis = _round_or_none(sa.get("eps"), 3)
        bvps_stockanalysis = _round_or_none(sa.get("book_value_per_share"), 3)
        pe_stockanalysis = _round_or_none(sa.get("pe_ratio"), 3)

        eps = eps_stockanalysis
        bvps = bvps_stockanalysis
        pe_ratio = None

        if eps_stockanalysis is not None:
            eps_from_stockanalysis += 1
        else:
            eps_missing += 1

        if bvps_stockanalysis is not None:
            bvps_from_stockanalysis += 1

        if pe_stockanalysis is not None and pe_stockanalysis > 0:
            pe_ratio = pe_stockanalysis
        elif last_price is not None and eps is not None and eps > 0:
            pe_ratio = _round_or_none(last_price / eps, 3)
            if pe_ratio is not None:
                pe_from_price_eps += 1
        if pe_ratio is None:
            pe_ratio = pe_snapshot

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
        "skipped_existing": skipped_existing,
        "fetched": len(to_fetch),
        "upserted": upserted,
        "no_data": no_data,
        "failed": failed,
        "eps_from_stockanalysis": eps_from_stockanalysis,
        "bvps_from_stockanalysis": bvps_from_stockanalysis,
        "eps_missing": eps_missing,
        "pe_from_price_over_eps": pe_from_price_eps,
        "elapsed_sec": round(time.time() - started_at, 2),
    }
    _last_run.update(run_info)

    logger.info(
        (
            "📘 Daily fundamentals refresh done: universe=%d fetched=%d skipped_existing=%d "
            "upserted=%d no_data=%d failed=%d eps_stockanalysis=%d "
            "bvps_stockanalysis=%d pe_price_over_eps=%d eps_missing=%d elapsed=%.2fs"
        ),
        len(seen),
        len(to_fetch),
        skipped_existing,
        upserted,
        no_data,
        failed,
        eps_from_stockanalysis,
        bvps_from_stockanalysis,
        pe_from_price_eps,
        eps_missing,
        run_info["elapsed_sec"],
    )
    return run_info


def get_last_run() -> dict:
    """Return info about the last fundamentals refresh run."""
    return dict(_last_run)
