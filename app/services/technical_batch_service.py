"""Daily Technical Analysis universe batch scoring.

Runs Kuwait signal scoring across the configured stock universe, stores results,
and serves latest-run snapshots for fast UI rendering.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, timedelta
from typing import Any, Optional

from app.core.config import get_settings
from app.core.database import exec_sql, exec_sql_returning_id, query_all, query_one
from app.data.stock_lists import KUWAIT_STOCKS

logger = logging.getLogger(__name__)

_SCHEMA_INIT = False
_BACKGROUND_TASKS: set[asyncio.Task] = set()

DEFAULT_MAX_CONCURRENCY = 4
MAX_CONCURRENCY = 8
DEFAULT_SEGMENT = "PREMIER"
STALE_RUN_TIMEOUT_SECONDS = 60 * 60
STALL_NO_PROGRESS_SECONDS = 10 * 60
PER_SYMBOL_TIMEOUT_SECONDS = 45


def _ensure_schema() -> None:
    """Create batch run/result tables if missing."""
    global _SCHEMA_INIT
    if _SCHEMA_INIT:
        return

    settings = get_settings()
    pk = "SERIAL PRIMARY KEY" if settings.use_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"

    exec_sql(
        f"""
        CREATE TABLE IF NOT EXISTS technical_analysis_runs (
            id {pk},
            started_at BIGINT NOT NULL,
            finished_at BIGINT,
            status TEXT NOT NULL,
            triggered_by TEXT NOT NULL,
            requested_by_user_id INTEGER,
            segment TEXT NOT NULL DEFAULT 'PREMIER',
            total_symbols INTEGER NOT NULL DEFAULT 0,
            processed_symbols INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            failed_count INTEGER NOT NULL DEFAULT 0,
            message TEXT
        )
        """
    )

    exec_sql(
        f"""
        CREATE TABLE IF NOT EXISTS technical_analysis_scores (
            id {pk},
            run_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            company_name TEXT,
            segment TEXT,
            signal TEXT,
            reason TEXT,
            trend_score INTEGER,
            momentum_score INTEGER,
            buying_pressure_score INTEGER,
            key_price_level_score INTEGER,
            overall_score INTEGER,
            raw_technical_score INTEGER,
            risk_adjusted_score INTEGER,
            error TEXT,
            created_at BIGINT NOT NULL,
            UNIQUE(run_id, symbol)
        )
        """
    )

    exec_sql(
        "CREATE INDEX IF NOT EXISTS ix_ta_runs_started_at ON technical_analysis_runs(started_at)"
    )
    exec_sql(
        "CREATE INDEX IF NOT EXISTS ix_ta_runs_status ON technical_analysis_runs(status)"
    )
    exec_sql(
        "CREATE INDEX IF NOT EXISTS ix_ta_scores_run_id ON technical_analysis_scores(run_id)"
    )
    exec_sql(
        "CREATE INDEX IF NOT EXISTS ix_ta_scores_symbol ON technical_analysis_scores(symbol)"
    )

    _SCHEMA_INIT = True


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(round(float(value)))
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _close_stale_running_runs() -> None:
    """Auto-close stale/stalled jobs so new scans are not blocked forever."""
    now = int(time.time())
    stale_rows = query_all(
        "SELECT id, started_at, processed_symbols, success_count, failed_count "
        "FROM technical_analysis_runs "
        "WHERE status = 'running'"
    )
    for row in stale_rows:
        started_at = _to_int(row.get("started_at"))
        if started_at is None:
            continue
        age_seconds = now - started_at

        run_id = _to_int(row.get("id"))
        if run_id is None:
            continue

        score_stats = query_one(
            "SELECT "
            "COUNT(*) AS processed, "
            "SUM(CASE WHEN error IS NULL OR error = '' THEN 1 ELSE 0 END) AS success, "
            "MAX(created_at) AS last_write "
            "FROM technical_analysis_scores "
            "WHERE run_id = ?",
            (run_id,),
        ) or {}

        processed = _to_int(score_stats.get("processed"))
        success = _to_int(score_stats.get("success"))
        if processed is None:
            processed = _to_int(row.get("processed_symbols")) or 0
        if success is None:
            success = _to_int(row.get("success_count")) or 0
        failed = max(0, processed - success)

        last_write_ts = _to_int(score_stats.get("last_write"))
        no_progress_seconds = age_seconds if last_write_ts is None else max(0, now - last_write_ts)

        timed_out = age_seconds > STALE_RUN_TIMEOUT_SECONDS
        stalled = no_progress_seconds > STALL_NO_PROGRESS_SECONDS
        if not (timed_out or stalled):
            continue

        if timed_out:
            reason = f"Auto-closed stale run after timeout ({STALE_RUN_TIMEOUT_SECONDS // 60}m)"
            logger.warning(
                "Technical batch: auto-closing timed-out run %s after %ss (processed=%s)",
                run_id,
                age_seconds,
                processed,
            )
        else:
            reason = f"Auto-closed stalled run after no progress ({STALL_NO_PROGRESS_SECONDS // 60}m)"
            logger.warning(
                "Technical batch: auto-closing stalled run %s after %ss without progress (processed=%s)",
                run_id,
                no_progress_seconds,
                processed,
            )

        _finish_run(
            run_id,
            status="failed",
            processed_symbols=processed,
            success_count=success,
            failed_count=failed,
            message=reason,
        )


def _load_universe(limit: Optional[int] = None) -> list[dict[str, str]]:
    """Return a unique, sorted Kuwait stock universe."""
    seen: set[str] = set()
    out: list[dict[str, str]] = []

    for stock in KUWAIT_STOCKS:
        symbol = str(stock.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(
            {
                "symbol": symbol,
                "name": str(stock.get("name") or symbol).strip(),
            }
        )

    out.sort(key=lambda x: x["symbol"])
    if limit is not None and limit > 0:
        return out[:limit]
    return out


def _serialize_run(row: Any) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "started_at": _to_int(row["started_at"]),
        "finished_at": _to_int(row.get("finished_at")),
        "status": str(row["status"]),
        "triggered_by": str(row.get("triggered_by") or ""),
        "requested_by_user_id": _to_int(row.get("requested_by_user_id")),
        "segment": str(row.get("segment") or DEFAULT_SEGMENT),
        "total_symbols": _to_int(row.get("total_symbols")) or 0,
        "processed_symbols": _to_int(row.get("processed_symbols")) or 0,
        "success_count": _to_int(row.get("success_count")) or 0,
        "failed_count": _to_int(row.get("failed_count")) or 0,
        "message": row.get("message"),
    }


def _serialize_score_row(row: Any) -> dict[str, Any]:
    return {
        "symbol": str(row["symbol"]),
        "company_name": row.get("company_name"),
        "segment": row.get("segment"),
        "signal": row.get("signal"),
        "reason": row.get("reason"),
        "trend_directional": _to_int(row.get("trend_score")) or 0,
        "speed_momentum": _to_int(row.get("momentum_score")) or 0,
        "buying_pressure": _to_int(row.get("buying_pressure_score")) or 0,
        "key_price_level": _to_int(row.get("key_price_level_score")) or 0,
        "overall_score": _to_int(row.get("overall_score")) or 0,
        "raw_technical_score": _to_int(row.get("raw_technical_score")),
        "risk_adjusted_score": _to_int(row.get("risk_adjusted_score")),
        "error": row.get("error"),
    }


def get_active_run() -> Optional[dict[str, Any]]:
    _ensure_schema()
    _close_stale_running_runs()
    row = query_one(
        "SELECT * FROM technical_analysis_runs "
        "WHERE status = 'running' "
        "ORDER BY started_at DESC, id DESC "
        "LIMIT 1"
    )
    if not row:
        return None
    return _serialize_run(row)


def get_latest_run(limit: int = 300) -> dict[str, Any]:
    _ensure_schema()
    _close_stale_running_runs()
    safe_limit = max(1, min(1000, int(limit or 300)))

    run_row = query_one(
        "SELECT * FROM technical_analysis_runs ORDER BY started_at DESC, id DESC LIMIT 1"
    )
    if not run_row:
        return {"run": None, "rows": []}

    run = _serialize_run(run_row)
    score_rows = query_all(
        "SELECT symbol, company_name, segment, signal, reason, trend_score, momentum_score, "
        "buying_pressure_score, key_price_level_score, overall_score, raw_technical_score, "
        "risk_adjusted_score, error "
        "FROM technical_analysis_scores "
        "WHERE run_id = ? "
        "ORDER BY CASE WHEN raw_technical_score IS NULL THEN 1 ELSE 0 END, raw_technical_score DESC, symbol ASC "
        "LIMIT ?",
        (run["id"], safe_limit),
    )
    rows = [_serialize_score_row(r) for r in score_rows]
    return {"run": run, "rows": rows}


def get_run_by_id(run_id: int, limit: int = 300) -> dict[str, Any]:
    _ensure_schema()
    _close_stale_running_runs()
    safe_limit = max(1, min(1000, int(limit or 300)))

    run_row = query_one("SELECT * FROM technical_analysis_runs WHERE id = ?", (run_id,))
    if not run_row:
        return {"run": None, "rows": []}

    run = _serialize_run(run_row)
    score_rows = query_all(
        "SELECT symbol, company_name, segment, signal, reason, trend_score, momentum_score, "
        "buying_pressure_score, key_price_level_score, overall_score, raw_technical_score, "
        "risk_adjusted_score, error "
        "FROM technical_analysis_scores "
        "WHERE run_id = ? "
        "ORDER BY CASE WHEN raw_technical_score IS NULL THEN 1 ELSE 0 END, raw_technical_score DESC, symbol ASC "
        "LIMIT ?",
        (run_id, safe_limit),
    )
    rows = [_serialize_score_row(r) for r in score_rows]
    return {"run": run, "rows": rows}


def _create_run(
    *,
    triggered_by: str,
    requested_by_user_id: Optional[int],
    total_symbols: int,
    segment: str,
) -> int:
    now = int(time.time())
    return exec_sql_returning_id(
        "INSERT INTO technical_analysis_runs "
        "(started_at, status, triggered_by, requested_by_user_id, segment, total_symbols, "
        "processed_symbols, success_count, failed_count, message) "
        "VALUES (?, 'running', ?, ?, ?, ?, 0, 0, 0, ?)",
        (
            now,
            triggered_by,
            requested_by_user_id,
            segment,
            total_symbols,
            "Batch scoring started",
        ),
    )


def _update_run_progress(
    run_id: int,
    *,
    processed_symbols: int,
    success_count: int,
    failed_count: int,
    message: str,
) -> None:
    exec_sql(
        "UPDATE technical_analysis_runs "
        "SET processed_symbols = ?, success_count = ?, failed_count = ?, message = ? "
        "WHERE id = ?",
        (processed_symbols, success_count, failed_count, message, run_id),
    )


def _finish_run(
    run_id: int,
    *,
    status: str,
    processed_symbols: int,
    success_count: int,
    failed_count: int,
    message: str,
) -> None:
    exec_sql(
        "UPDATE technical_analysis_runs "
        "SET status = ?, processed_symbols = ?, success_count = ?, failed_count = ?, "
        "message = ?, finished_at = ? "
        "WHERE id = ?",
        (
            status,
            processed_symbols,
            success_count,
            failed_count,
            message,
            int(time.time()),
            run_id,
        ),
    )


def _insert_score(run_id: int, score: dict[str, Any]) -> None:
    exec_sql(
        "INSERT INTO technical_analysis_scores "
        "(run_id, symbol, company_name, segment, signal, reason, trend_score, momentum_score, "
        "buying_pressure_score, key_price_level_score, overall_score, raw_technical_score, "
        "risk_adjusted_score, error, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            score["symbol"],
            score.get("company_name"),
            score.get("segment") or DEFAULT_SEGMENT,
            score.get("signal"),
            score.get("reason"),
            score.get("trend_score"),
            score.get("momentum_score"),
            score.get("buying_pressure_score"),
            score.get("key_price_level_score"),
            score.get("overall_score"),
            score.get("raw_technical_score"),
            score.get("risk_adjusted_score"),
            score.get("error"),
            int(time.time()),
        ),
    )


def _failed_symbol_row(
    *,
    symbol: str,
    company_name: str,
    segment: str,
    error: str,
    signal: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build a table-ready failed row so every symbol is accounted for."""
    return {
        "symbol": symbol,
        "company_name": company_name,
        "segment": segment.upper(),
        "signal": signal,
        "reason": reason,
        "trend_score": None,
        "momentum_score": None,
        "buying_pressure_score": None,
        "key_price_level_score": None,
        "overall_score": None,
        "raw_technical_score": None,
        "risk_adjusted_score": None,
        "error": (error or "unknown_error")[:300],
    }


async def _score_one_symbol(
    symbol: str,
    company_name: str,
    segment: str,
    account_equity: float,
) -> dict[str, Any]:
    """Score one symbol and return table-ready values."""
    from app.services import tickerchart_service as tc
    from app.services.indicators_service import attach_indicators
    from app.services.signal_engine.data.preprocessing import forward_fill_gaps
    from app.services.signal_engine.engine.signal_generator import generate_kuwait_signal

    fetch_from = date.today() - timedelta(days=730)

    try:
        parsed = tc.split_symbol(symbol, "KSE", None)
        if parsed is None:
            raise RuntimeError("symbol_resolution_failed")

        base, market = parsed
        rows = await tc.fetch_ohlcv(base, market, from_d=fetch_from, to_d=None)
        if not rows:
            raise RuntimeError("no_price_data")

        rows = forward_fill_gaps(rows)
        rows = attach_indicators(rows)

        signal = await generate_kuwait_signal(
            rows=rows,
            stock_code=base,
            segment=segment.upper(),
            account_equity=account_equity,
            delay_hours=0,
        )

        component_scores = signal.get("component_scores") or {}
        trend_raw = _to_int(((component_scores.get("trend") or {}).get("raw")))
        momentum_raw = _to_int(((component_scores.get("momentum") or {}).get("raw")))
        volume_raw = _to_int(((component_scores.get("volume_flow") or {}).get("raw")))
        sr_raw = _to_int(((component_scores.get("support_resistance") or {}).get("raw")))

        confluence = signal.get("confluence_details") or {}
        score_breakdown = signal.get("score_breakdown") or {}
        four_scores = confluence.get("four_scores") or {}
        overall_four = four_scores.get("overall") or {}

        # Daily batch UI dual-overall semantics:
        # - raw_technical_score  -> combined score WITHOUT directional trend adjustment
        # - risk_adjusted_score  -> combined score WITH directional trend adjustment
        combined_no_adjustment = _to_int(signal.get("combined_score_unadjusted_directional"))
        combined_with_adjustment = _to_int(signal.get("combined_score_adjusted_directional"))

        # Backfill from score_breakdown for mixed/older payload variants.
        if combined_no_adjustment is None:
            combined_no_adjustment = _to_int(score_breakdown.get("combined_unadjusted_directional"))
        if combined_with_adjustment is None:
            combined_with_adjustment = _to_int(score_breakdown.get("combined_adjusted_directional"))

        # In current engine raw_technical_score matches combined adjusted directional.
        if combined_with_adjustment is None:
            combined_with_adjustment = _to_int(signal.get("raw_technical_score"))

        # Legacy fallback path for older runs that only had four-score overall values.
        if combined_no_adjustment is None:
            combined_no_adjustment = _to_int(overall_four.get("base_score"))
        if combined_with_adjustment is None:
            combined_with_adjustment = _to_int(overall_four.get("score"))

        # Final fallback from older confluence totals.
        if combined_no_adjustment is None:
            combined_no_adjustment = _to_int(confluence.get("total_score_raw"))
        if combined_with_adjustment is None:
            combined_with_adjustment = _to_int(confluence.get("total_score"))

        # Safety: keep both columns populated whenever one side exists.
        if combined_no_adjustment is None:
            combined_no_adjustment = combined_with_adjustment
        if combined_with_adjustment is None:
            combined_with_adjustment = combined_no_adjustment

        overall_score = combined_with_adjustment

        return {
            "symbol": symbol,
            "company_name": company_name,
            "segment": segment.upper(),
            "signal": signal.get("signal"),
            "reason": signal.get("reason"),
            "trend_score": trend_raw,
            "momentum_score": momentum_raw,
            "buying_pressure_score": volume_raw,
            "key_price_level_score": sr_raw,
            "overall_score": overall_score,
            "raw_technical_score": combined_no_adjustment,
            "risk_adjusted_score": combined_with_adjustment,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        err_code = str(exc)[:300]
        logger.warning("Technical batch: %s failed: %s", symbol, err_code)
        if err_code in {"no_price_data", "symbol_resolution_failed"}:
            return _failed_symbol_row(
                symbol=symbol,
                company_name=company_name,
                segment=segment,
                error=err_code,
                signal="NO_DATA",
                reason=err_code,
            )
        return _failed_symbol_row(
            symbol=symbol,
            company_name=company_name,
            segment=segment,
            error=err_code,
        )


async def _execute_run(
    *,
    run_id: int,
    universe: list[dict[str, str]],
    segment: str,
    max_concurrency: int,
    account_equity: float,
) -> dict[str, Any]:
    sem = asyncio.Semaphore(max(1, min(MAX_CONCURRENCY, max_concurrency)))
    accounted_symbols: set[str] = set()
    loop_exception: Exception | None = None

    async def _worker(entry: dict[str, str]) -> dict[str, Any]:
        async with sem:
            symbol = entry["symbol"]
            company_name = entry["name"]
            try:
                return await asyncio.wait_for(
                    _score_one_symbol(
                        symbol=symbol,
                        company_name=company_name,
                        segment=segment,
                        account_equity=account_equity,
                    ),
                    timeout=PER_SYMBOL_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Technical batch: %s timed out after %ss",
                    symbol,
                    PER_SYMBOL_TIMEOUT_SECONDS,
                )
                return _failed_symbol_row(
                    symbol=symbol,
                    company_name=company_name,
                    segment=segment,
                    error="symbol_timeout",
                    signal="NO_DATA",
                    reason="symbol_timeout",
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Technical batch worker crashed for %s", symbol)
                return _failed_symbol_row(
                    symbol=symbol,
                    company_name=company_name,
                    segment=segment,
                    error=f"worker_exception:{exc}",
                )

    tasks = [asyncio.create_task(_worker(entry)) for entry in universe]

    total = len(universe)
    processed = 0
    success = 0
    failed = 0

    try:
        for task in asyncio.as_completed(tasks):
            result = await task
            symbol = str(result.get("symbol") or "")

            try:
                _insert_score(run_id, result)
            except Exception as insert_exc:  # noqa: BLE001
                logger.exception(
                    "Technical batch run %s: failed inserting row for %s",
                    run_id,
                    symbol or "<unknown>",
                )
                if loop_exception is None:
                    loop_exception = insert_exc
                continue

            if symbol:
                accounted_symbols.add(symbol)

            processed += 1
            if result.get("error"):
                failed += 1
            else:
                success += 1

            if processed == 1 or processed % 10 == 0 or processed == total:
                _update_run_progress(
                    run_id,
                    processed_symbols=processed,
                    success_count=success,
                    failed_count=failed,
                    message=f"Processed {processed}/{total}",
                )
    except Exception as exc:  # noqa: BLE001
        loop_exception = exc
        logger.exception("Technical batch run %s interrupted during processing", run_id)
    finally:
        # Ensure all tasks are finalized and no task remains running in the loop.
        await asyncio.gather(*tasks, return_exceptions=True)

    # Reconcile missing symbols so the run accounts for the full universe.
    if len(accounted_symbols) < total:
        reason = "batch_interrupted" if loop_exception else "symbol_not_accounted"
        missing_entries = [
            entry for entry in universe if entry["symbol"] not in accounted_symbols
        ]
        for entry in missing_entries:
            fallback = _failed_symbol_row(
                symbol=entry["symbol"],
                company_name=entry["name"],
                segment=segment,
                error=reason,
                signal="NO_DATA",
                reason=reason,
            )
            try:
                _insert_score(run_id, fallback)
                accounted_symbols.add(entry["symbol"])
            except Exception as insert_exc:  # noqa: BLE001
                logger.exception(
                    "Technical batch run %s: failed inserting reconciliation row for %s",
                    run_id,
                    entry["symbol"],
                )
                if loop_exception is None:
                    loop_exception = insert_exc

    # Recompute counts from persisted rows to keep run metadata consistent.
    stats = query_one(
        "SELECT "
        "COUNT(*) AS processed, "
        "SUM(CASE WHEN error IS NULL OR error = '' THEN 1 ELSE 0 END) AS success "
        "FROM technical_analysis_scores "
        "WHERE run_id = ?",
        (run_id,),
    ) or {}

    processed = _to_int(stats.get("processed")) or 0
    success = _to_int(stats.get("success")) or 0
    failed = max(0, processed - success)

    if processed < total:
        status = "failed"
        finish_message = f"Incomplete: accounted {processed}/{total}, success {success}, failed {failed}"
    elif loop_exception is not None:
        status = "failed"
        finish_message = f"Completed with interruption: {success} success, {failed} failed"
    else:
        status = "completed" if success > 0 else "failed"
        finish_message = f"Completed: {success} success, {failed} failed"

    _finish_run(
        run_id,
        status=status,
        processed_symbols=processed,
        success_count=success,
        failed_count=failed,
        message=finish_message,
    )

    return {
        "run_id": run_id,
        "status": status,
        "total_symbols": total,
        "processed_symbols": processed,
        "success_count": success,
        "failed_count": failed,
        "message": finish_message,
    }


async def run_batch_once(
    *,
    triggered_by: str,
    requested_by_user_id: Optional[int] = None,
    segment: str = DEFAULT_SEGMENT,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    limit: Optional[int] = None,
    account_equity: float = 100_000.0,
) -> dict[str, Any]:
    """Run one full universe batch synchronously in the current event loop."""
    _ensure_schema()

    active = get_active_run()
    if active:
        return {
            "accepted": False,
            "already_running": True,
            "run": active,
            "message": "A technical batch is already running",
        }

    universe = _load_universe(limit=limit)
    if not universe:
        raise ValueError("No stocks available for technical batch scoring")

    run_id = _create_run(
        triggered_by=triggered_by,
        requested_by_user_id=requested_by_user_id,
        total_symbols=len(universe),
        segment=segment.upper(),
    )

    try:
        summary = await _execute_run(
            run_id=run_id,
            universe=universe,
            segment=segment,
            max_concurrency=max_concurrency,
            account_equity=account_equity,
        )
        run_data = get_run_by_id(run_id, limit=10).get("run")
        return {
            "accepted": True,
            "already_running": False,
            "run": run_data,
            "summary": summary,
            "message": "Technical batch run completed",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Technical batch run %s failed", run_id)
        _finish_run(
            run_id,
            status="failed",
            processed_symbols=0,
            success_count=0,
            failed_count=len(universe),
            message=f"Batch failed: {exc}",
        )
        raise


def kickoff_batch_background(
    *,
    triggered_by: str,
    requested_by_user_id: Optional[int] = None,
    segment: str = DEFAULT_SEGMENT,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    limit: Optional[int] = None,
    account_equity: float = 100_000.0,
) -> dict[str, Any]:
    """Create a run and execute it in a background asyncio task."""
    _ensure_schema()

    active = get_active_run()
    if active:
        return {
            "accepted": False,
            "already_running": True,
            "run": active,
            "message": "A technical batch is already running",
        }

    universe = _load_universe(limit=limit)
    if not universe:
        raise ValueError("No stocks available for technical batch scoring")

    run_id = _create_run(
        triggered_by=triggered_by,
        requested_by_user_id=requested_by_user_id,
        total_symbols=len(universe),
        segment=segment.upper(),
    )

    async def _runner() -> None:
        try:
            await _execute_run(
                run_id=run_id,
                universe=universe,
                segment=segment,
                max_concurrency=max_concurrency,
                account_equity=account_equity,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Background technical batch run %s failed", run_id)
            _finish_run(
                run_id,
                status="failed",
                processed_symbols=0,
                success_count=0,
                failed_count=len(universe),
                message=f"Batch failed: {exc}",
            )

    task = asyncio.create_task(_runner())
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)

    run_data = get_run_by_id(run_id, limit=10).get("run")
    return {
        "accepted": True,
        "already_running": False,
        "run": run_data,
        "message": "Technical batch run started",
    }


def run_batch_sync(
    *,
    triggered_by: str = "scheduler",
    segment: str = DEFAULT_SEGMENT,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    limit: Optional[int] = None,
    account_equity: float = 100_000.0,
) -> dict[str, Any]:
    """Run batch scoring from sync contexts (APScheduler thread)."""
    return asyncio.run(
        run_batch_once(
            triggered_by=triggered_by,
            requested_by_user_id=None,
            segment=segment,
            max_concurrency=max_concurrency,
            limit=limit,
            account_equity=account_equity,
        )
    )
