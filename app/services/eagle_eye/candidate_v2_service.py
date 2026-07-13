from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.database import get_conn
from app.data.stock_lists import KUWAIT_STOCKS
from app.services import tickerchart_service as tc

RUNNER_VERSION = "phase0_candidate_v2_runner_1"
PARSER_VERSION = "tc_ondemand_v2"
SCHEMA_VERSION = "candidate_v2_schema_1"
VENDOR = "tickerchart"
RAW_PAYLOAD_VERSION = "vendor_raw_v1"
IDENTITY_ADJUSTMENT_VERSION = "identity_v1"

FAILPOINTS = {
    "25pct",
    "50pct",
    "90pct",
    "after_raw_before_lineage",
    "after_anomaly_before_completion",
}


@dataclass
class LineageInfo:
    commit: str
    dirty: bool
    diff_hash: str | None


@dataclass
class SymbolRunResult:
    symbol: str
    run_id: str
    status: str
    rows_written: int
    anomaly_count: int
    reason_code: str | None
    parse_failed: int
    no_vendor_data: bool


def _now_ts() -> int:
    return int(datetime.now(tz=UTC).timestamp())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _iso_to_ts(date_str: str) -> int | None:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
        return int(dt.timestamp())
    except ValueError:
        return None


def _resolve_lineage(repo_root: Path) -> LineageInfo:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to resolve git commit: {exc}") from exc

    dirty_text = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True)
    dirty = bool(dirty_text.strip())
    diff_hash: str | None = None
    if dirty:
        diff = subprocess.check_output(["git", "diff"], cwd=repo_root)
        diff_hash = _sha256_bytes(diff)

    if not commit:
        raise RuntimeError("Fail-closed lineage: commit missing")

    return LineageInfo(commit=commit, dirty=dirty, diff_hash=diff_hash)


def _run_coro_sync(coro):
    result_box: list[Any] = []
    exc_box: list[Exception] = []

    def _target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box.append(loop.run_until_complete(coro))
        except Exception as exc:  # noqa: BLE001
            exc_box.append(exc)
        finally:
            loop.close()

    import threading

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=90)
    if thread.is_alive():
        raise TimeoutError("Vendor fetch timed out after 90s")
    if exc_box:
        raise exc_box[0]
    return result_box[0] if result_box else None


def connect(db_path: Path) -> Any:
    settings = get_settings()
    if settings.use_postgres:
        return get_conn()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def ensure_schema(conn: Any) -> None:
    stmts = [
        """
        CREATE TABLE IF NOT EXISTS ee_schema_metadata (
            schema_version TEXT PRIMARY KEY,
            applied_at INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_ingestion_runs_v2 (
            run_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            started_at INTEGER NOT NULL,
            completed_at INTEGER,
            status TEXT NOT NULL,
            reason_code TEXT,
            reason_json TEXT,
            rows_written INTEGER NOT NULL DEFAULT 0,
            anomaly_count INTEGER NOT NULL DEFAULT 0,
            parse_failed_count INTEGER NOT NULL DEFAULT 0,
            request_hash TEXT,
            payload_hash TEXT,
            code_commit TEXT NOT NULL,
            dirty_worktree INTEGER NOT NULL DEFAULT 0,
            source_package_hash TEXT,
            parser_version TEXT NOT NULL,
            runner_version TEXT NOT NULL,
            environment TEXT NOT NULL,
            db_schema_version TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_failure_audit_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            failpoint TEXT,
            error_text TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_ohlcv_raw (
            symbol TEXT NOT NULL,
            trade_date INTEGER NOT NULL,
            vendor TEXT NOT NULL,
            payload_version TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            value_kwd REAL NOT NULL,
            payload_hash TEXT NOT NULL,
            run_id TEXT NOT NULL,
            ingested_at INTEGER NOT NULL,
            PRIMARY KEY (symbol, trade_date, vendor, payload_version)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_anomaly_events_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            event_start_date INTEGER NOT NULL,
            event_end_date INTEGER NOT NULL,
            classification TEXT NOT NULL,
            decision_owner TEXT,
            evidence_json TEXT,
            adjustment_version TEXT,
            resolution_status TEXT NOT NULL DEFAULT 'open',
            created_at INTEGER NOT NULL,
            run_id TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_corporate_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            announcement_date INTEGER,
            effective_date INTEGER,
            action_type TEXT NOT NULL,
            official_factor REAL,
            evidence_source TEXT,
            approval_status TEXT NOT NULL,
            action_version TEXT NOT NULL,
            approved_by TEXT,
            approved_at INTEGER,
            created_at INTEGER NOT NULL,
            UNIQUE(symbol, action_version)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_ohlcv_adjusted (
            symbol TEXT NOT NULL,
            trade_date INTEGER NOT NULL,
            adjustment_version TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            value_kwd REAL NOT NULL,
            source_raw_identity TEXT NOT NULL,
            source_raw_hash TEXT NOT NULL,
            corporate_action_version TEXT,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (symbol, trade_date, adjustment_version)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_indicators_v2 (
            symbol TEXT NOT NULL,
            trade_date INTEGER NOT NULL,
            series_kind TEXT NOT NULL,
            adjustment_version TEXT,
            payload_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (symbol, trade_date, series_kind, adjustment_version)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_symbol_reconciliation_v2 (
            symbol TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            raw_rows INTEGER NOT NULL,
            parse_failed_count INTEGER NOT NULL,
            anomaly_events INTEGER NOT NULL,
            no_vendor_data INTEGER NOT NULL,
            fully_unusable INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_indicator_readiness_v2 (
            symbol TEXT PRIMARY KEY,
            total_bars INTEGER NOT NULL,
            valid_rsi_count INTEGER NOT NULL,
            valid_adx_count INTEGER NOT NULL,
            valid_sma200_count INTEGER NOT NULL,
            valid_252_count INTEGER NOT NULL,
            earliest_model_ready_date INTEGER,
            latest_feature_completeness REAL NOT NULL,
            eligibility_status TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_benchmark_boundary_registry_v2 (
            benchmark_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            proposed_as_of_date INTEGER,
            owner_approved_as_of_date INTEGER,
            chart_reference TEXT,
            purpose TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_pit_features_v2 (
            benchmark_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            as_of_date INTEGER NOT NULL,
            source_trade_date INTEGER,
            feature_json TEXT NOT NULL,
            approval_status TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (benchmark_id, symbol)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_pit_outcomes_v2 (
            benchmark_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            as_of_date INTEGER NOT NULL,
            outcome_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (benchmark_id, symbol)
        )
        """,
    ]

    for stmt in stmts:
        conn.execute(stmt)

    conn.execute(
        "INSERT OR REPLACE INTO ee_schema_metadata (schema_version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, _now_ts()),
    )
    conn.commit()


def _classify_event(prev_ts: int, cur_ts: int, prev_close: float, cur_close: float) -> tuple[str, dict[str, Any]]:
    jump_abs = abs((cur_close / prev_close) - 1.0) if prev_close > 0 else 0.0
    gap_days = max(1, int((cur_ts - prev_ts) / 86400))

    if jump_abs >= 0.75:
        cls = "scale anomaly"
    elif gap_days > 20:
        cls = "missing-session or suspension effect"
    elif jump_abs >= 0.35:
        cls = "possible corporate action"
    elif jump_abs >= 0.25:
        cls = "unresolved"
    else:
        cls = "valid market sequence"

    return cls, {
        "jump_abs": jump_abs,
        "gap_days": gap_days,
        "prior_close": prev_close,
        "next_close": cur_close,
    }


def _prepare_rows(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    parsed: list[dict[str, Any]] = []
    parse_failed = 0
    for row in raw_rows:
        ts = _iso_to_ts(str(row.get("date") or ""))
        if ts is None:
            parse_failed += 1
            continue
        try:
            parsed.append(
                {
                    "trade_date": ts,
                    "open": float(row.get("open") or 0.0),
                    "high": float(row.get("high") or 0.0),
                    "low": float(row.get("low") or 0.0),
                    "close": float(row.get("close") or 0.0),
                    "volume": float(row.get("volume") or 0.0),
                    "value_kwd": float(row.get("value") or 0.0),
                }
            )
        except Exception:  # noqa: BLE001
            parse_failed += 1

    parsed.sort(key=lambda x: x["trade_date"])

    events: list[dict[str, Any]] = []
    prev = None
    for row in parsed:
        if prev is not None and prev["close"] > 0 and row["close"] > 0:
            cls, details = _classify_event(prev["trade_date"], row["trade_date"], prev["close"], row["close"])
            if cls != "valid market sequence":
                events.append(
                    {
                        "event_start_date": prev["trade_date"],
                        "event_end_date": row["trade_date"],
                        "classification": cls,
                        "evidence": details,
                    }
                )
        prev = row

    return parsed, events, parse_failed


def _insert_failure_audit(conn: Any, run_id: str, symbol: str, failpoint: str | None, exc: Exception) -> None:
    conn.execute(
        "INSERT INTO ee_failure_audit_v2 (run_id, symbol, failpoint, error_text, created_at) VALUES (?, ?, ?, ?, ?)",
        (run_id, symbol, failpoint, str(exc)[:1000], _now_ts()),
    )
    conn.commit()


def _status_for_symbol(rows_written: int, no_vendor: bool, parse_failed: int, anomaly_count: int) -> tuple[str, str | None]:
    if no_vendor:
        return "no_vendor_data", "no_vendor_data"
    if rows_written == 0 and parse_failed > 0:
        return "failed_validation", "parse_failed"
    if rows_written == 0:
        return "failed_validation", "fully_unusable"
    if anomaly_count > 0:
        return "event_quarantined", "anomaly_events_detected"
    return "completed", None


def ingest_symbol_rows(
    conn: Any,
    *,
    symbol: str,
    rows: list[dict[str, Any]],
    lineage: LineageInfo,
    failpoint: str | None,
    environment: str,
    request_hash: str,
) -> SymbolRunResult:
    run_id = str(uuid.uuid4())
    started = _now_ts()
    payload_hash = _sha256_text(json.dumps(rows, ensure_ascii=True, sort_keys=True))

    conn.execute(
        """
        INSERT INTO ee_ingestion_runs_v2 (
            run_id, symbol, started_at, status, request_hash, payload_hash,
            code_commit, dirty_worktree, source_package_hash, parser_version,
            runner_version, environment, db_schema_version
        ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            symbol,
            started,
            request_hash,
            payload_hash,
            lineage.commit,
            1 if lineage.dirty else 0,
            lineage.diff_hash,
            PARSER_VERSION,
            RUNNER_VERSION,
            environment,
            SCHEMA_VERSION,
        ),
    )
    conn.commit()

    if failpoint and failpoint not in FAILPOINTS:
        raise ValueError(f"Unsupported failpoint: {failpoint}")

    parsed, events, parse_failed = _prepare_rows(rows)

    if not rows:
        status, reason = _status_for_symbol(0, True, parse_failed, 0)
        conn.execute(
            "UPDATE ee_ingestion_runs_v2 SET completed_at=?, status=?, reason_code=?, reason_json=?, parse_failed_count=? WHERE run_id=?",
            (
                _now_ts(),
                status,
                reason,
                json.dumps({"reason": "vendor returned no rows"}, ensure_ascii=True),
                parse_failed,
                run_id,
            ),
        )
        conn.commit()
        return SymbolRunResult(symbol, run_id, status, 0, 0, reason, parse_failed, True)

    rows_written = 0
    anomaly_count = 0

    try:
        conn.execute("BEGIN")
        total = len(parsed)
        for idx, row in enumerate(parsed, start=1):
            raw_identity = f"{symbol}:{row['trade_date']}:{VENDOR}:{RAW_PAYLOAD_VERSION}"
            raw_hash = _sha256_text(json.dumps(row, ensure_ascii=True, sort_keys=True))
            conn.execute(
                """
                INSERT OR REPLACE INTO ee_ohlcv_raw (
                    symbol, trade_date, vendor, payload_version,
                    open, high, low, close, volume, value_kwd,
                    payload_hash, run_id, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    row["trade_date"],
                    VENDOR,
                    RAW_PAYLOAD_VERSION,
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                    row["value_kwd"],
                    raw_hash,
                    run_id,
                    _now_ts(),
                ),
            )

            conn.execute(
                """
                INSERT OR REPLACE INTO ee_ohlcv_adjusted (
                    symbol, trade_date, adjustment_version,
                    open, high, low, close, volume, value_kwd,
                    source_raw_identity, source_raw_hash,
                    corporate_action_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    row["trade_date"],
                    IDENTITY_ADJUSTMENT_VERSION,
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                    row["value_kwd"],
                    raw_identity,
                    raw_hash,
                    "none",
                    _now_ts(),
                ),
            )

            rows_written += 1
            pct = int((idx / max(total, 1)) * 100)
            if failpoint == "25pct" and pct >= 25:
                raise RuntimeError("FAILPOINT_25PCT")
            if failpoint == "50pct" and pct >= 50:
                raise RuntimeError("FAILPOINT_50PCT")
            if failpoint == "90pct" and pct >= 90:
                raise RuntimeError("FAILPOINT_90PCT")

        if failpoint == "after_raw_before_lineage":
            raise RuntimeError("FAILPOINT_AFTER_RAW_BEFORE_LINEAGE")

        for ev in events:
            conn.execute(
                """
                INSERT INTO ee_anomaly_events_v2 (
                    symbol, event_start_date, event_end_date,
                    classification, decision_owner, evidence_json,
                    adjustment_version, resolution_status, created_at, run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    ev["event_start_date"],
                    ev["event_end_date"],
                    ev["classification"],
                    "system_candidate_v2",
                    json.dumps(ev["evidence"], ensure_ascii=True),
                    "pending",
                    "open",
                    _now_ts(),
                    run_id,
                ),
            )
            anomaly_count += 1

        # Mandatory SANAM handling: preserve raw, create CA ledger + unapproved preview.
        if symbol == "SANAM" and anomaly_count > 0:
            action_version = "sanam_ca_v1_proposed"
            conn.execute(
                """
                INSERT OR REPLACE INTO ee_corporate_actions (
                    symbol, announcement_date, effective_date, action_type,
                    official_factor, evidence_source, approval_status,
                    action_version, approved_by, approved_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    None,
                    events[0]["event_end_date"],
                    "split_or_reverse_split",
                    None,
                    "tickerchart_jump_detection",
                    "proposed",
                    action_version,
                    None,
                    None,
                    _now_ts(),
                ),
            )

            # Proposed preview only; not approved.
            factor = 1.0
            if events:
                jump_abs = float(events[0]["evidence"].get("jump_abs") or 0.0)
                factor = max(0.1, min(10.0, 1.0 + jump_abs))
            for row in parsed:
                raw_identity = f"{symbol}:{row['trade_date']}:{VENDOR}:{RAW_PAYLOAD_VERSION}"
                raw_hash = _sha256_text(json.dumps(row, ensure_ascii=True, sort_keys=True))
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ee_ohlcv_adjusted (
                        symbol, trade_date, adjustment_version,
                        open, high, low, close, volume, value_kwd,
                        source_raw_identity, source_raw_hash,
                        corporate_action_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        row["trade_date"],
                        "sanam_preview_unapproved_v1",
                        row["open"] / factor,
                        row["high"] / factor,
                        row["low"] / factor,
                        row["close"] / factor,
                        row["volume"],
                        row["value_kwd"],
                        raw_identity,
                        raw_hash,
                        action_version,
                        _now_ts(),
                    ),
                )

        if failpoint == "after_anomaly_before_completion":
            raise RuntimeError("FAILPOINT_AFTER_ANOMALY_BEFORE_COMPLETION")

        status, reason = _status_for_symbol(rows_written, False, parse_failed, anomaly_count)
        conn.execute(
            """
            UPDATE ee_ingestion_runs_v2
            SET completed_at=?, status=?, reason_code=?, reason_json=?,
                rows_written=?, anomaly_count=?, parse_failed_count=?
            WHERE run_id=?
            """,
            (
                _now_ts(),
                status,
                reason,
                json.dumps({"anomaly_count": anomaly_count}, ensure_ascii=True),
                rows_written,
                anomaly_count,
                parse_failed,
                run_id,
            ),
        )

        conn.execute("COMMIT")
        conn.commit()
        return SymbolRunResult(symbol, run_id, status, rows_written, anomaly_count, reason, parse_failed, False)

    except Exception as exc:  # noqa: BLE001
        conn.execute("ROLLBACK")
        conn.execute(
            """
            UPDATE ee_ingestion_runs_v2
            SET completed_at=?, status='failed_transaction', reason_code='failed_transaction',
                reason_json=?, rows_written=0, anomaly_count=0, parse_failed_count=?
            WHERE run_id=?
            """,
            (
                _now_ts(),
                json.dumps({"error": str(exc)}, ensure_ascii=True),
                parse_failed,
                run_id,
            ),
        )
        conn.commit()
        _insert_failure_audit(conn, run_id, symbol, failpoint, exc)
        return SymbolRunResult(symbol, run_id, "failed_transaction", 0, 0, "failed_transaction", parse_failed, False)


def _fetch_symbol_rows(symbol: str) -> list[dict[str, Any]]:
    try:
        return _run_coro_sync(tc.fetch_ohlcv(symbol, "KSE", interval="day")) or []
    except Exception:
        return []


def _compute_indicator_readiness(conn: Any, symbol: str) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT trade_date FROM ee_ohlcv_raw WHERE symbol=? ORDER BY trade_date",
        (symbol,),
    ).fetchall()
    total = len(rows)
    if total == 0:
        return {
            "symbol": symbol,
            "total_bars": 0,
            "valid_rsi_count": 0,
            "valid_adx_count": 0,
            "valid_sma200_count": 0,
            "valid_252_count": 0,
            "earliest_model_ready_date": None,
            "latest_feature_completeness": 0.0,
            "eligibility_status": "no_data",
        }

    valid_rsi = max(0, total - 14)
    valid_adx = max(0, total - 19)
    valid_sma200 = max(0, total - 199)
    valid_252 = max(0, total - 251)

    earliest_ready = int(rows[251]["trade_date"]) if total >= 252 else None
    completeness = min(1.0, valid_252 / max(total, 1))

    last_td = int(rows[-1]["trade_date"])
    stale_cutoff = int((datetime.now(tz=UTC) - timedelta(days=45)).timestamp())

    if total < 200:
        status = "insufficient_history"
    elif total < 252:
        status = "partial_features"
    elif last_td < stale_cutoff:
        status = "stale_history"
    else:
        status = "model_ready"

    return {
        "symbol": symbol,
        "total_bars": total,
        "valid_rsi_count": valid_rsi,
        "valid_adx_count": valid_adx,
        "valid_sma200_count": valid_sma200,
        "valid_252_count": valid_252,
        "earliest_model_ready_date": earliest_ready,
        "latest_feature_completeness": completeness,
        "eligibility_status": status,
    }


def _upsert_reconciliation(conn: Any, symbols: list[str]) -> None:
    for symbol in symbols:
        raw_rows = int(
            conn.execute("SELECT COUNT(1) FROM ee_ohlcv_raw WHERE symbol=?", (symbol,)).fetchone()[0]
        )
        parse_failed = int(
            conn.execute(
                "SELECT COALESCE(SUM(parse_failed_count),0) FROM ee_ingestion_runs_v2 WHERE symbol=?",
                (symbol,),
            ).fetchone()[0]
        )
        anomaly_events = int(
            conn.execute("SELECT COUNT(1) FROM ee_anomaly_events_v2 WHERE symbol=?", (symbol,)).fetchone()[0]
        )
        no_vendor = int(
            conn.execute(
                "SELECT COUNT(1) FROM ee_ingestion_runs_v2 WHERE symbol=? AND status='no_vendor_data'",
                (symbol,),
            ).fetchone()[0]
        )

        if raw_rows > 0 and anomaly_events > 0:
            status = "event_quarantined"
            fully_unusable = 0
        elif raw_rows > 0:
            status = "loaded_raw"
            fully_unusable = 0
        elif no_vendor > 0:
            status = "no_vendor_data"
            fully_unusable = 1
        elif parse_failed > 0:
            status = "parse_failed"
            fully_unusable = 1
        else:
            status = "fully_unusable"
            fully_unusable = 1

        conn.execute(
            """
            INSERT OR REPLACE INTO ee_symbol_reconciliation_v2 (
                symbol, status, raw_rows, parse_failed_count,
                anomaly_events, no_vendor_data, fully_unusable, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                status,
                raw_rows,
                parse_failed,
                anomaly_events,
                1 if no_vendor > 0 else 0,
                fully_unusable,
                _now_ts(),
            ),
        )


def _seed_benchmark_registry(conn: Any) -> None:
    now = _now_ts()
    proposals = [
        ("pit_bpcc_r1", "BPCC", "2026-07-08", "chart_bpcc_redline_r1", "Phase0 benchmark red line"),
        ("pit_sanam_r1", "SANAM", "2026-07-08", "chart_sanam_redline_r1", "Phase0 benchmark red line"),
        ("pit_tijara_r1", "TIJARA", "2026-07-08", "chart_tijara_redline_r1", "Phase0 benchmark red line"),
        ("pit_zain_r1", "ZAIN", "2026-07-08", "chart_zain_redline_r1", "Phase0 benchmark red line"),
        ("pit_mabanee_r1", "MABANEE", "2026-07-08", "chart_mabanee_redline_r1", "Phase0 benchmark red line"),
    ]
    for bench_id, symbol, proposed_date, chart_ref, purpose in proposals:
        proposed_ts = _iso_to_ts(proposed_date)
        conn.execute(
            """
            INSERT OR REPLACE INTO ee_benchmark_boundary_registry_v2 (
                benchmark_id, symbol, proposed_as_of_date, owner_approved_as_of_date,
                chart_reference, purpose, status, created_at, updated_at
            ) VALUES (?, ?, ?, COALESCE((SELECT owner_approved_as_of_date FROM ee_benchmark_boundary_registry_v2 WHERE benchmark_id=?), NULL), ?, ?, ?, COALESCE((SELECT created_at FROM ee_benchmark_boundary_registry_v2 WHERE benchmark_id=?), ?), ?)
            """,
            (
                bench_id,
                symbol,
                proposed_ts,
                bench_id,
                chart_ref,
                purpose,
                "pending_owner_approval",
                bench_id,
                now,
                now,
            ),
        )


def _build_pit(conn: Any) -> None:
    rows = conn.execute(
        "SELECT benchmark_id, symbol, proposed_as_of_date, owner_approved_as_of_date, status FROM ee_benchmark_boundary_registry_v2 ORDER BY benchmark_id"
    ).fetchall()

    for row in rows:
        symbol = str(row["symbol"])
        approved = row["owner_approved_as_of_date"]
        proposed = row["proposed_as_of_date"]
        as_of = int(approved if approved is not None else proposed)
        approval_status = "approved" if approved is not None else "proposed_unapproved"

        src = conn.execute(
            """
            SELECT trade_date, close, volume, value_kwd
            FROM ee_ohlcv_raw
            WHERE symbol=? AND trade_date<=?
            ORDER BY trade_date DESC
            LIMIT 1
            """,
            (symbol, as_of),
        ).fetchone()

        if src is None:
            feature = {
                "symbol": symbol,
                "as_of_date": as_of,
                "status": "no_data",
                "reason": "no rows on or before boundary",
            }
            src_trade_date = None
        else:
            src_trade_date = int(src["trade_date"])
            count = int(
                conn.execute(
                    "SELECT COUNT(1) FROM ee_ohlcv_raw WHERE symbol=? AND trade_date<=?",
                    (symbol, src_trade_date),
                ).fetchone()[0]
            )
            feature = {
                "symbol": symbol,
                "as_of_date": as_of,
                "source_trade_date": src_trade_date,
                "left_of_line_count": count,
                "close": float(src["close"]),
                "volume": float(src["volume"]),
                "value_kwd": float(src["value_kwd"]),
            }

        conn.execute(
            """
            INSERT OR REPLACE INTO ee_pit_features_v2 (
                benchmark_id, symbol, as_of_date, source_trade_date,
                feature_json, approval_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["benchmark_id"],
                symbol,
                as_of,
                src_trade_date,
                json.dumps(feature, ensure_ascii=True),
                approval_status,
                _now_ts(),
            ),
        )

        conn.execute(
            """
            INSERT OR REPLACE INTO ee_pit_outcomes_v2 (
                benchmark_id, symbol, as_of_date, outcome_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                row["benchmark_id"],
                symbol,
                as_of,
                json.dumps({"future_outcomes_reserved": True}, ensure_ascii=True),
                _now_ts(),
            ),
        )


def _build_blast_radius_report(conn: Any, symbol: str) -> dict[str, Any]:
    rows_raw = conn.execute(
        "SELECT trade_date, close FROM ee_ohlcv_raw WHERE symbol=? ORDER BY trade_date",
        (symbol,),
    ).fetchall()
    rows_adj = conn.execute(
        "SELECT trade_date, close FROM ee_ohlcv_adjusted WHERE symbol=? AND adjustment_version='sanam_preview_unapproved_v1' ORDER BY trade_date",
        (symbol,),
    ).fetchall()

    if not rows_raw or not rows_adj:
        return {"symbol": symbol, "status": "unavailable"}

    def sma(values: list[float], period: int) -> list[float | None]:
        out: list[float | None] = []
        for i in range(len(values)):
            if i + 1 < period:
                out.append(None)
            else:
                win = values[i - period + 1 : i + 1]
                out.append(sum(win) / period)
        return out

    raw_close = [float(r["close"]) for r in rows_raw]
    adj_close = [float(r["close"]) for r in rows_adj]

    raw_sma200 = sma(raw_close, 200)
    adj_sma200 = sma(adj_close, 200)

    diffs: list[float] = []
    for rv, av in zip(raw_sma200, adj_sma200):
        if rv is None or av is None:
            continue
        diffs.append(abs(rv - av))

    return {
        "symbol": symbol,
        "rows_raw": len(rows_raw),
        "rows_adjusted_preview": len(rows_adj),
        "sma200_abs_diff_max": max(diffs) if diffs else 0.0,
        "sma200_abs_diff_mean": (sum(diffs) / len(diffs)) if diffs else 0.0,
        "note": "preview only; official factor not approved",
    }


def _build_census(conn: Any) -> dict[str, Any]:
    checks = {
        "sqlite_integrity": conn.execute("PRAGMA integrity_check").fetchone()[0],
        "duplicate_raw_pk": conn.execute(
            "SELECT COUNT(1) FROM (SELECT symbol, trade_date, vendor, payload_version, COUNT(1) c FROM ee_ohlcv_raw GROUP BY 1,2,3,4 HAVING c>1)"
        ).fetchone()[0],
        "invalid_ohlc": conn.execute(
            "SELECT COUNT(1) FROM ee_ohlcv_raw WHERE high < low OR open<=0 OR high<=0 OR low<=0 OR close<=0"
        ).fetchone()[0],
        "negative_volume_value": conn.execute(
            "SELECT COUNT(1) FROM ee_ohlcv_raw WHERE volume < 0 OR value_kwd < 0"
        ).fetchone()[0],
        "future_rows": conn.execute(
            "SELECT COUNT(1) FROM ee_ohlcv_raw WHERE trade_date > ?",
            (_now_ts() + 86400,),
        ).fetchone()[0],
        "missing_lineage": conn.execute(
            "SELECT COUNT(1) FROM ee_ingestion_runs_v2 WHERE code_commit IS NULL OR TRIM(code_commit)=''"
        ).fetchone()[0],
        "unknown_placeholder_lineage": conn.execute(
            "SELECT COUNT(1) FROM ee_ingestion_runs_v2 WHERE LOWER(code_commit)='unknown'"
        ).fetchone()[0],
        "orphan_ingestion_runs": conn.execute(
            "SELECT COUNT(1) FROM ee_ingestion_runs_v2 r LEFT JOIN ee_ohlcv_raw o ON o.run_id=r.run_id WHERE r.status IN ('completed','completed_with_warnings','event_quarantined') AND o.run_id IS NULL"
        ).fetchone()[0],
        "non_completed_run_residue": conn.execute(
            "SELECT COUNT(1) FROM ee_ohlcv_raw o JOIN ee_ingestion_runs_v2 r ON r.run_id=o.run_id WHERE r.status NOT IN ('completed','completed_with_warnings','event_quarantined')"
        ).fetchone()[0],
        "raw_adjusted_coexistence": conn.execute(
            "SELECT COUNT(1) FROM ee_ohlcv_raw r JOIN ee_ohlcv_adjusted a ON a.symbol=r.symbol AND a.trade_date=r.trade_date"
        ).fetchone()[0],
        "corporate_action_version_integrity": conn.execute(
            "SELECT COUNT(1) FROM (SELECT symbol, action_version, COUNT(1) c FROM ee_corporate_actions GROUP BY symbol, action_version HAVING c>1)"
        ).fetchone()[0],
        "stale_symbols": conn.execute(
            "SELECT COUNT(1) FROM ee_indicator_readiness_v2 WHERE eligibility_status='stale_history'"
        ).fetchone()[0],
        "no_data_symbols": conn.execute(
            "SELECT COUNT(1) FROM ee_indicator_readiness_v2 WHERE eligibility_status='no_data'"
        ).fetchone()[0],
        "event_level_quarantines": conn.execute(
            "SELECT COUNT(1) FROM ee_anomaly_events_v2"
        ).fetchone()[0],
        "pit_symbols_present": conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM ee_pit_features_v2"
        ).fetchone()[0],
        "reconciled_symbol_count": conn.execute(
            "SELECT COUNT(1) FROM ee_symbol_reconciliation_v2"
        ).fetchone()[0],
    }
    return checks


def run_candidate_v2(*, db_path: Path, output_dir: Path, failpoint: str | None = None) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    lineage = _resolve_lineage(repo_root)
    settings = get_settings()

    if not settings.use_postgres and db_path.exists():
        db_path.unlink()

    conn = connect(db_path)
    ensure_schema(conn)

    symbols = sorted({str(s.get("symbol") or "").upper().strip() for s in KUWAIT_STOCKS if str(s.get("symbol") or "").strip()})
    env = str(os.getenv("ENVIRONMENT") or "development").strip().lower() or "development"

    results: list[SymbolRunResult] = []
    for symbol in symbols:
        request_hash = _sha256_text(json.dumps({"symbol": symbol, "interval": "day"}, ensure_ascii=True, sort_keys=True))
        rows = _fetch_symbol_rows(symbol)
        result = ingest_symbol_rows(
            conn,
            symbol=symbol,
            rows=rows,
            lineage=lineage,
            failpoint=failpoint,
            environment=env,
            request_hash=request_hash,
        )
        results.append(result)

    _upsert_reconciliation(conn, symbols)

    for symbol in symbols:
        r = _compute_indicator_readiness(conn, symbol)
        conn.execute(
            """
            INSERT OR REPLACE INTO ee_indicator_readiness_v2 (
                symbol, total_bars, valid_rsi_count, valid_adx_count,
                valid_sma200_count, valid_252_count, earliest_model_ready_date,
                latest_feature_completeness, eligibility_status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["symbol"],
                r["total_bars"],
                r["valid_rsi_count"],
                r["valid_adx_count"],
                r["valid_sma200_count"],
                r["valid_252_count"],
                r["earliest_model_ready_date"],
                r["latest_feature_completeness"],
                r["eligibility_status"],
                _now_ts(),
            ),
        )

    _seed_benchmark_registry(conn)
    _build_pit(conn)

    blast_radius = _build_blast_radius_report(conn, "SANAM")
    census = _build_census(conn)

    summary = {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "database_path": str(db_path),
        "schema_version": SCHEMA_VERSION,
        "lineage": {
            "commit": lineage.commit,
            "dirty_worktree": lineage.dirty,
            "source_package_hash": lineage.diff_hash,
            "parser_version": PARSER_VERSION,
            "runner_version": RUNNER_VERSION,
            "environment": env,
        },
        "symbols_requested": len(symbols),
        "run_status_counts": {
            k: int(v)
            for k, v in conn.execute(
                "SELECT status, COUNT(1) FROM ee_ingestion_runs_v2 GROUP BY status"
            ).fetchall()
        },
        "reconciliation_counts": {
            k: int(v)
            for k, v in conn.execute(
                "SELECT status, COUNT(1) FROM ee_symbol_reconciliation_v2 GROUP BY status"
            ).fetchall()
        },
        "sanam_blast_radius": blast_radius,
        "census": census,
        "pass_recommendation": False,
        "pass_blockers": [
            "Owner benchmark approvals pending",
            "Corporate-action official factors pending where unresolved",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    recon = [dict(r) for r in conn.execute("SELECT * FROM ee_symbol_reconciliation_v2 ORDER BY symbol").fetchall()]
    (output_dir / "candidate_v2_reconciliation.json").write_text(json.dumps(recon, ensure_ascii=True, indent=2), encoding="utf-8")

    pit = [dict(r) for r in conn.execute("SELECT * FROM ee_pit_features_v2 ORDER BY benchmark_id").fetchall()]
    for row in pit:
        row["feature_json"] = json.loads(str(row.get("feature_json") or "{}"))
    (output_dir / "candidate_v2_pit_features.json").write_text(json.dumps(pit, ensure_ascii=True, indent=2), encoding="utf-8")

    readiness = [dict(r) for r in conn.execute("SELECT * FROM ee_indicator_readiness_v2 ORDER BY symbol").fetchall()]
    (output_dir / "candidate_v2_indicator_readiness.json").write_text(json.dumps(readiness, ensure_ascii=True, indent=2), encoding="utf-8")

    actions = [dict(r) for r in conn.execute("SELECT * FROM ee_corporate_actions ORDER BY symbol, action_version").fetchall()]
    (output_dir / "candidate_v2_corporate_actions.json").write_text(json.dumps(actions, ensure_ascii=True, indent=2), encoding="utf-8")

    anomalies = [dict(r) for r in conn.execute("SELECT * FROM ee_anomaly_events_v2 ORDER BY symbol, event_start_date").fetchall()]
    for row in anomalies:
        row["evidence_json"] = json.loads(str(row.get("evidence_json") or "{}"))
    (output_dir / "candidate_v2_anomaly_events.json").write_text(json.dumps(anomalies, ensure_ascii=True, indent=2), encoding="utf-8")

    conn.commit()
    conn.close()
    return summary
