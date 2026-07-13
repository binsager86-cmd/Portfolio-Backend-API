from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException

from app.core.database import exec_sql, query_all, query_one, query_val
from app.core.config import get_settings
from app.core.security import TokenData
from app.services.eagle_eye.audit_service import create_event
from app.data.stock_lists import KUWAIT_STOCKS

logger = logging.getLogger(__name__)
PARSER_VERSION = "ee_ohlcv_parser_v3"


def _current_environment() -> str:
    env = str(get_settings().ENVIRONMENT or "development").strip().lower()
    return env or "development"


def _current_commit() -> str:
    return str(os.getenv("GIT_COMMIT") or os.getenv("COMMIT_SHA") or "unknown").strip() or "unknown"


def _is_synthetic_fixture_source(path: str | None, source: str) -> bool:
    src = str(source or "").strip().lower()
    if src.startswith("debug") or src.startswith("synthetic") or src.startswith("fixture") or src.startswith("test"):
        return True
    p = str(path or "").replace("\\", "/").lower()
    return "/tests/fixtures/" in p and "/synthetic_" in p


def _real_market_symbols() -> set[str]:
    symbols = {str(s.get("symbol") or "").upper().replace(".KW", "").strip() for s in KUWAIT_STOCKS}
    symbols.discard("")
    try:
        rows = query_all(
            """
            SELECT DISTINCT symbol FROM analysis_stocks
            WHERE (exchange IN ('KW', 'KSE') OR currency = 'KWD')
            """,
            (),
        ) or []
    except Exception:
        rows = []
    for row in rows:
        symbols.add(str(row.get("symbol") or "").upper().replace(".KW", "").strip())
    symbols.discard("")
    return symbols


def _source_priority(source_type: str) -> int:
    levels = {
        "vendor_raw": 90,
        "vendor_corrected": 80,
        "manual_correction": 70,
        "csv_import": 50,
        "csv_fixture": 10,
        "debug": 5,
        "synthetic": 1,
    }
    return levels.get(str(source_type or "").strip().lower(), 30)


def _is_trusted_source_type(source_type: str) -> bool:
    return str(source_type or "").strip().lower() in {"vendor_raw", "vendor_corrected", "manual_correction"}


def _emit_ingest_reject_audit(
    *,
    symbol: str,
    trade_date: int,
    existing_type: str,
    incoming_type: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        create_event(
            {
                "action": "data_ingest_rejected",
                "entity_type": "symbol",
                "entity_id": str(symbol or "").upper(),
                "change_type": "data",
                "risk_level": "medium",
                "source": "ingest_guard",
                "metadata": {
                    "trade_date": int(trade_date),
                    "existing_source_type": existing_type,
                    "incoming_source_type": incoming_type,
                    "reason": reason,
                    **(metadata or {}),
                },
                "concept_version": CONCEPT_VERSION,
            },
            TokenData(user_id=0, username="system", is_admin=True),
        )
    except Exception:
        logger.exception("Failed to write data_ingest_rejected audit event")


def _begin_ingestion_run(
    *,
    source_type: str,
    source_ref: str,
    payload_hash: str,
    request_parameters_hash: str | None,
    synthetic_flag: int,
) -> str:
    run_id = str(uuid.uuid4())
    exec_sql(
        """
        INSERT INTO ee_ingestion_runs (
            run_id, environment, source_type, source_ref,
            request_parameters_hash, payload_hash,
            code_commit, parser_version, synthetic_flag,
            started_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            _current_environment(),
            source_type,
            source_ref,
            request_parameters_hash,
            payload_hash,
            _current_commit(),
            PARSER_VERSION,
            int(synthetic_flag),
            now_ts(),
            "running",
        ),
    )
    return run_id


def _finalize_ingestion_run(run_id: str, rows_written: int, status: str = "completed") -> None:
    exec_sql(
        """
        UPDATE ee_ingestion_runs
        SET completed_at = ?, rows_written = ?, status = ?
        WHERE run_id = ?
        """,
        (now_ts(), int(rows_written), status, run_id),
    )


def _upsert_ohlcv_row(
    *,
    symbol: str,
    trade_date: int,
    open_v: float,
    high_v: float,
    low_v: float,
    close_v: float,
    volume_v: float,
    value_kwd_v: float,
    source: str,
    source_type: str,
    source_ref: str,
    run_id: str,
    request_parameters_hash: str | None,
    payload_hash: str,
    synthetic_flag: int,
    adjustment_status: str,
    corporate_action_version: str,
    approved_change_request_id: int | None = None,
) -> bool:
    existing = query_one(
        """
        SELECT source_type, data_environment, synthetic_flag,
               payload_hash, adjustment_status, corporate_action_version,
               source_ref, ingestion_run_id
        FROM ee_ohlcv
        WHERE symbol = ? AND trade_date = ?
        """,
        (symbol, trade_date),
    )

    if str(source_type).strip().lower() == "manual_correction" and approved_change_request_id is None:
        raise HTTPException(status_code=400, detail="manual_correction requires approved_change_request_id")

    if existing is not None:
        existing_type = str(existing.get("source_type") or "")
        existing_env = str(existing.get("data_environment") or "")
        existing_syn = int(existing.get("synthetic_flag") or 0)
        existing_payload_hash = str(existing.get("payload_hash") or "")
        existing_adjustment_status = str(existing.get("adjustment_status") or "raw_unadjusted")

        # Re-ingestion of identical trusted payload must be idempotent.
        if (
            _is_trusted_source_type(existing_type)
            and _is_trusted_source_type(source_type)
            and existing_payload_hash
            and existing_payload_hash == str(payload_hash or "")
            and existing_adjustment_status == str(adjustment_status or "raw_unadjusted")
        ):
            return False

        # Raw and adjusted bars must never overwrite one another.
        if existing_adjustment_status != str(adjustment_status or "raw_unadjusted"):
            _emit_ingest_reject_audit(
                symbol=symbol,
                trade_date=trade_date,
                existing_type=existing_type,
                incoming_type=str(source_type),
                reason="raw_adjusted_cross_overwrite_rejected",
                metadata={
                    "existing_adjustment_status": existing_adjustment_status,
                    "incoming_adjustment_status": str(adjustment_status or "raw_unadjusted"),
                },
            )
            return False

        if existing_env == "production" and existing_syn == 0 and _source_priority(source_type) < _source_priority(existing_type):
            _emit_ingest_reject_audit(
                symbol=symbol,
                trade_date=trade_date,
                existing_type=existing_type,
                incoming_type=str(source_type),
                reason="lower_priority_source_rejected",
            )
            return False

        if existing_type in {"vendor_raw", "vendor_corrected", "manual_correction"} and source_type in {"csv_fixture", "debug", "synthetic"}:
            _emit_ingest_reject_audit(
                symbol=symbol,
                trade_date=trade_date,
                existing_type=existing_type,
                incoming_type=str(source_type),
                reason="trusted_row_preserved_against_fixture_or_debug",
            )
            return False

        # Conflicting trusted payload must be quarantined (or versioned). We quarantine.
        if (
            _is_trusted_source_type(existing_type)
            and _is_trusted_source_type(source_type)
            and existing_payload_hash
            and existing_payload_hash != str(payload_hash or "")
        ):
            now = now_ts()
            reason = {
                "type": "trusted_payload_conflict",
                "trade_date": int(trade_date),
                "existing_source_type": existing_type,
                "incoming_source_type": str(source_type),
                "existing_payload_hash": existing_payload_hash,
                "incoming_payload_hash": str(payload_hash or ""),
            }
            exec_sql(
                """
                INSERT INTO ee_data_quality_quarantine (
                    symbol, status, reason_json, first_flagged_at, last_flagged_at
                ) VALUES (?, 'quarantined', ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    status='quarantined',
                    reason_json=excluded.reason_json,
                    last_flagged_at=excluded.last_flagged_at,
                    cleared_at=NULL,
                    change_request_id=NULL,
                    cleared_by_user_id=NULL
                """,
                (symbol, json.dumps([reason], ensure_ascii=True), now, now),
            )
            _emit_ingest_reject_audit(
                symbol=symbol,
                trade_date=trade_date,
                existing_type=existing_type,
                incoming_type=str(source_type),
                reason="trusted_payload_conflict_quarantined",
            )
            return False

    exec_sql(
        """
        INSERT INTO ee_ohlcv (
            symbol, trade_date, open, high, low, close, raw_close, adjusted_close,
            volume, value_kwd, value_unit, source, source_type, source_ref,
            data_environment, ingestion_run_id, request_parameters_hash,
            payload_hash, code_commit, parser_version, synthetic_flag,
            adjustment_status, corporate_action_version, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, trade_date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            raw_close = excluded.raw_close,
            adjusted_close = excluded.adjusted_close,
            volume = excluded.volume,
            value_kwd = excluded.value_kwd,
            value_unit = excluded.value_unit,
            source = excluded.source,
            source_type = excluded.source_type,
            source_ref = excluded.source_ref,
            data_environment = excluded.data_environment,
            ingestion_run_id = excluded.ingestion_run_id,
            request_parameters_hash = excluded.request_parameters_hash,
            payload_hash = excluded.payload_hash,
            code_commit = excluded.code_commit,
            parser_version = excluded.parser_version,
            synthetic_flag = excluded.synthetic_flag,
            adjustment_status = excluded.adjustment_status,
            corporate_action_version = excluded.corporate_action_version,
            ingested_at = excluded.ingested_at
        """,
        (
            symbol,
            int(trade_date),
            float(open_v),
            float(high_v),
            float(low_v),
            float(close_v),
            float(close_v),
            float(close_v),
            float(volume_v),
            float(value_kwd_v),
            "kwd",
            source,
            source_type,
            source_ref,
            _current_environment(),
            run_id,
            request_parameters_hash,
            payload_hash,
            _current_commit(),
            PARSER_VERSION,
            int(synthetic_flag),
            adjustment_status,
            corporate_action_version,
            now_ts(),
        ),
    )
    return True

CONCEPT_VERSION = "ee-2.1.2"
_NOW_TS_OVERRIDE: int | None = None

KEY_TARGET_AREAS: dict[str, str] = {
    "base_min_sessions": "scanner",
    "base_max_width_pct": "scanner",
    "volume_breakout_mult": "scanner",
    "obv_divergence_lookback": "scanner",
    "rsi_regime": "scanner",
    "adx_trigger": "scanner",
    "cmf_floor": "scanner",
    "atr_squeeze_pctile": "scanner",
    "trend_join_window": "scanner",
    "base_drift_invalidate_sessions": "scanner",
    "avoid_reclaim_clear_closes": "scanner",
    "exit_cooldown_sessions": "scanner",
    "pilot_enabled": "entry_exit",
    "climax_partial": "entry_exit",
    "ml_gate_enabled": "ml_overlay",
    "ml_min_labeled_signals": "ml_overlay",
    "ml_prob_min": "ml_overlay",
    "min_daily_value_kwd": "risk_management",
    "risk_per_trade": "risk_management",
    "max_positions": "risk_management",
    "max_sector_concentration": "risk_management",
    "max_portfolio_heat": "risk_management",
    "allow_self_review": "api_contract",
    "bt_commission_bps": "scheduler",
    "bt_slippage_bps": "scheduler",
    "gap_check_window_days": "scheduler",
    "max_session_gap_days": "scheduler",
    "validated_history_start": "scheduler",
    "pipeline_mode": "scheduler",
}

DEFAULT_ENGINE_CONFIG: dict[str, Any] = {
    "base_min_sessions": 60,
    "base_max_width_pct": 0.18,
    "volume_breakout_mult": 2.5,
    "obv_divergence_lookback": 40,
    "rsi_regime": 55,
    "adx_trigger": 22,
    "cmf_floor": 0.05,
    "atr_squeeze_pctile": 0.20,
    "trend_join_window": 40,
    "base_drift_invalidate_sessions": 10,
    "avoid_reclaim_clear_closes": 2,
    "exit_cooldown_sessions": 10,
    "pilot_enabled": True,
    "climax_partial": True,
    "ml_gate_enabled": False,
    "ml_min_labeled_signals": 150,
    "ml_prob_min": 0.45,
    "min_daily_value_kwd": 100000.0,
    "risk_per_trade": 0.01,
    "max_positions": 8,
    "max_sector_concentration": 0.40,
    "max_portfolio_heat": 0.06,
    "allow_self_review": False,
    "bt_commission_bps": 25,
    "bt_slippage_bps": 30,
    "gap_check_window_days": 30,
    "max_session_gap_days": 7,
    "validated_history_start": "2021-01-01",
    "pipeline_mode": "paper",
}

REQUIRED_CONFIG_KEYS: set[str] = {
    "base_min_sessions",
    "base_max_width_pct",
    "volume_breakout_mult",
    "rsi_regime",
    "adx_trigger",
    "cmf_floor",
    "atr_squeeze_pctile",
    "trend_join_window",
    "base_drift_invalidate_sessions",
    "avoid_reclaim_clear_closes",
    "exit_cooldown_sessions",
    "pilot_enabled",
    "min_daily_value_kwd",
    "max_positions",
    "ml_gate_enabled",
    "ml_min_labeled_signals",
    "ml_prob_min",
    "bt_commission_bps",
    "bt_slippage_bps",
}


class ConfigKeyMissing(KeyError):
    def __init__(self, key: str):
        super().__init__(f"Missing required config key: {key}")
        self.key = key


def get_cfg(cfg: dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise ConfigKeyMissing(key)
    return cfg[key]


def validate_runtime_config_keys(cfg: dict[str, Any], required_keys: set[str] | None = None) -> None:
    required = required_keys or REQUIRED_CONFIG_KEYS
    missing = [k for k in sorted(required) if k not in cfg]
    if missing:
        raise ConfigKeyMissing(", ".join(missing))


def validate_engine_config_presence(required_keys: set[str] | None = None) -> None:
    required = required_keys or REQUIRED_CONFIG_KEYS
    rows = query_all("SELECT key FROM ee_engine_config", ())
    present = {str(r.get("key")) for r in rows or []}
    missing = [k for k in sorted(required) if k not in present]
    if missing:
        raise ConfigKeyMissing(", ".join(missing))


def now_ts() -> int:
    if _NOW_TS_OVERRIDE is not None:
        return int(_NOW_TS_OVERRIDE)
    return int(datetime.now(tz=timezone.utc).timestamp())


def set_now_ts_override(ts: int | None) -> None:
    global _NOW_TS_OVERRIDE
    _NOW_TS_OVERRIDE = int(ts) if ts is not None else None


def ensure_schema() -> None:
    stmts = [
        """
        CREATE TABLE IF NOT EXISTS ee_ohlcv (
            symbol TEXT NOT NULL, trade_date INTEGER NOT NULL,
            open REAL, high REAL, low REAL, close REAL, raw_close REAL, adjusted_close REAL,
            volume REAL, value_kwd REAL, value_unit TEXT,
            source TEXT NOT NULL DEFAULT 'feed', source_type TEXT, source_ref TEXT,
            data_environment TEXT, ingestion_run_id TEXT, request_parameters_hash TEXT,
            payload_hash TEXT, code_commit TEXT, parser_version TEXT,
            synthetic_flag INTEGER NOT NULL DEFAULT 0,
            adjustment_status TEXT NOT NULL DEFAULT 'raw_unadjusted',
            corporate_action_version TEXT NOT NULL DEFAULT 'none',
            ingested_at INTEGER NOT NULL,
            PRIMARY KEY (symbol, trade_date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_ingestion_runs (
            run_id TEXT PRIMARY KEY,
            environment TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            request_parameters_hash TEXT,
            payload_hash TEXT,
            code_commit TEXT,
            parser_version TEXT,
            synthetic_flag INTEGER NOT NULL DEFAULT 0,
            started_at INTEGER NOT NULL,
            completed_at INTEGER,
            rows_written INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'running'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_indicators (
            symbol TEXT NOT NULL, trade_date INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            concept_version TEXT NOT NULL,
            PRIMARY KEY (symbol, trade_date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_symbol_state (
            symbol TEXT PRIMARY KEY,
            phase TEXT NOT NULL DEFAULT 'NEUTRAL',
            phase_since INTEGER NOT NULL,
            base_high REAL, base_low REAL, base_start INTEGER,
            last_score REAL, avoid_until INTEGER,
            updated_at INTEGER NOT NULL,
            state_json TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL, symbol TEXT NOT NULL, trade_date INTEGER NOT NULL,
            signal_type TEXT NOT NULL,
            phase_from TEXT, phase_to TEXT,
            score REAL, price REAL, stop_price REAL, evidence_json TEXT NOT NULL,
            concept_version TEXT NOT NULL, config_hash TEXT NOT NULL,
            audit_event_id INTEGER,
            outcome_label TEXT, outcome_return REAL, outcome_at INTEGER
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_ee_signals_symbol ON ee_signals(symbol, trade_date DESC)",
        "CREATE INDEX IF NOT EXISTS idx_ee_signals_type ON ee_signals(signal_type, created_at DESC)",
        """
        CREATE TABLE IF NOT EXISTS ee_ratings (
            symbol TEXT NOT NULL, trade_date INTEGER NOT NULL,
            score REAL NOT NULL, band TEXT NOT NULL, components_json TEXT NOT NULL,
            concept_version TEXT NOT NULL,
            PRIMARY KEY (symbol, trade_date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL, opened_at INTEGER NOT NULL, closed_at INTEGER,
            status TEXT NOT NULL DEFAULT 'open',
            tranches_json TEXT NOT NULL, avg_entry REAL, stop_price REAL, trail_price REAL,
            exit_reason TEXT, realized_return REAL,
            signal_id INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_engine_config (
            key TEXT PRIMARY KEY, value_json TEXT NOT NULL,
            updated_at INTEGER NOT NULL, updated_by_user_id INTEGER,
            change_request_id INTEGER
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            started_at INTEGER NOT NULL,
            ended_at INTEGER,
            symbols_json TEXT NOT NULL,
            start_date INTEGER NOT NULL,
            end_date INTEGER NOT NULL,
            config_hash TEXT NOT NULL,
            concept_version TEXT NOT NULL,
            report_json TEXT,
            status TEXT NOT NULL DEFAULT 'running'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_backtest_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            opened_at INTEGER NOT NULL,
            closed_at INTEGER,
            side TEXT NOT NULL DEFAULT 'long',
            tranches_json TEXT NOT NULL,
            avg_entry REAL,
            avg_exit REAL,
            gross_return REAL,
            net_return REAL,
            exit_reason TEXT,
            meta_json TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ee_data_quality_quarantine (
            symbol TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'quarantined',
            reason_json TEXT NOT NULL,
            first_flagged_at INTEGER NOT NULL,
            last_flagged_at INTEGER NOT NULL,
            cleared_at INTEGER,
            change_request_id INTEGER,
            cleared_by_user_id INTEGER
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_ee_dq_status ON ee_data_quality_quarantine(status)",
    ]

    for stmt in stmts:
        exec_sql(stmt, ())

    from app.core.database import add_column_if_missing as _acim
    _acim("ee_ohlcv", "value_unit", "TEXT")
    _acim("ee_ohlcv", "raw_close", "REAL")
    _acim("ee_ohlcv", "adjusted_close", "REAL")
    _acim("ee_ohlcv", "source_type", "TEXT")
    _acim("ee_ohlcv", "source_ref", "TEXT")
    _acim("ee_ohlcv", "data_environment", "TEXT")
    _acim("ee_ohlcv", "ingestion_run_id", "TEXT")
    _acim("ee_ohlcv", "request_parameters_hash", "TEXT")
    _acim("ee_ohlcv", "payload_hash", "TEXT")
    _acim("ee_ohlcv", "code_commit", "TEXT")
    _acim("ee_ohlcv", "parser_version", "TEXT")
    _acim("ee_ohlcv", "synthetic_flag", "INTEGER")
    _acim("ee_ohlcv", "adjustment_status", "TEXT")
    _acim("ee_ohlcv", "corporate_action_version", "TEXT")
    try:
        _acim("ee_ohlcv_cache", "value_unit", "TEXT")
    except Exception:
        pass

    _seed_default_config()
    validate_engine_config_presence()


def _seed_default_config() -> None:
    ts = now_ts()
    for key, value in DEFAULT_ENGINE_CONFIG.items():
        exec_sql(
            """
            INSERT INTO ee_engine_config (key, value_json, updated_at, updated_by_user_id, change_request_id)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO NOTHING
            """,
            (key, json.dumps(value, ensure_ascii=True), ts, 0, None),
        )


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").upper().strip()
    if s.endswith(".KW"):
        s = s[:-3]
    return s


def get_active_config() -> dict[str, Any]:
    rows = query_all("SELECT key, value_json FROM ee_engine_config", ())
    cfg = dict(DEFAULT_ENGINE_CONFIG)
    for row in rows or []:
        key = str(row.get("key"))
        raw = row.get("value_json")
        try:
            cfg[key] = json.loads(str(raw))
        except Exception:
            logger.warning("Invalid ee_engine_config JSON for key=%s", key)
    validate_runtime_config_keys(cfg)
    return cfg


def get_config_hash(config: dict[str, Any] | None = None) -> str:
    payload = config or get_active_config()
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_config_with_meta() -> dict[str, Any]:
    cfg = get_active_config()
    return {
        "config": cfg,
        "config_hash": get_config_hash(cfg),
        "concept_version": CONCEPT_VERSION,
        "advice": False,
    }


def get_validated_history_start(cfg: dict[str, Any] | None = None) -> date:
    """Return the inclusive history start date used by scan/backtest pipelines."""
    raw = (cfg or get_active_config()).get("validated_history_start", "2021-01-01")
    text = str(raw or "2021-01-01").strip()
    try:
        return date.fromisoformat(text[:10])
    except Exception:
        return date(2021, 1, 1)


def _validated_start_ts(cfg: dict[str, Any] | None = None) -> int:
    day = get_validated_history_start(cfg)
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def _normalize_value_kwd(native_value: float | None, close_fils: float | None, volume: float | None) -> float:
    """Convert feed-native value from fils to KWD; OHLC prices remain in fils."""
    if native_value is not None:
        return float(native_value) / 1000.0
    if close_fils is None or volume is None:
        return 0.0
    return (float(close_fils) * float(volume)) / 1000.0


def repair_value_units(
    actor: TokenData | None = None,
    source: str = "manual",
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Normalize legacy fils-scale traded value rows to KWD across all symbols."""
    ensure_schema()
    actor = actor or TokenData(user_id=0, username="system", is_admin=True)
    trace_id = trace_id or str(uuid.uuid4())

    rows = query_all(
        """
        SELECT symbol, COUNT(1) AS n
        FROM ee_ohlcv
        WHERE value_kwd IS NOT NULL
          AND COALESCE(value_unit, '') <> 'kwd'
        GROUP BY symbol
        ORDER BY symbol ASC
        """,
        (),
    )

    total_rows = 0
    per_symbol: list[dict[str, Any]] = []
    for row in rows or []:
        symbol = str(row.get("symbol") or "").upper()
        touched = int(row.get("n") or 0)
        if not symbol or touched <= 0:
            continue
        exec_sql(
            """
            UPDATE ee_ohlcv
            SET value_kwd = value_kwd / 1000.0,
                value_unit = 'kwd'
            WHERE symbol = ?
              AND value_kwd IS NOT NULL
              AND COALESCE(value_unit, '') <> 'kwd'
            """,
            (symbol,),
        )
        total_rows += touched
        event = create_event(
            {
                "action": "data_repair",
                "entity_type": "symbol",
                "entity_id": symbol,
                "change_type": "data",
                "risk_level": "medium",
                "trace_id": trace_id,
                "source": source,
                "metadata": {
                    "repair": "value_kwd_fils_to_kwd",
                    "rows_touched": touched,
                },
                "concept_version": CONCEPT_VERSION,
            },
            actor,
        )
        per_symbol.append(
            {
                "symbol": symbol,
                "rows_touched": touched,
                "audit_event_id": event.get("id"),
            }
        )

    return {
        "trace_id": trace_id,
        "symbols_repaired": len(per_symbol),
        "rows_touched": total_rows,
        "per_symbol": per_symbol,
        "advice": False,
    }


def update_config(
    values: dict[str, Any],
    target_area: str,
    change_request_id: int,
    actor: TokenData,
) -> dict[str, Any]:
    if not values:
        raise HTTPException(status_code=400, detail="No config values provided")

    unknown = [k for k in values.keys() if k not in KEY_TARGET_AREAS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unsupported config keys: {', '.join(sorted(unknown))}")

    mismatched = [k for k, area in KEY_TARGET_AREAS.items() if k in values and area != target_area]
    if mismatched:
        raise HTTPException(
            status_code=409,
            detail=f"Change request target_area '{target_area}' does not match keys: {', '.join(sorted(mismatched))}",
        )

    cr = query_one(
        "SELECT id, status, target_area FROM ee_change_requests WHERE id = ?",
        (change_request_id,),
    )
    if not cr:
        raise HTTPException(status_code=404, detail="Change request not found")

    if str(cr.get("status") or "") != "approved":
        raise HTTPException(status_code=400, detail="Change request must be approved")

    if str(cr.get("target_area") or "") != target_area:
        raise HTTPException(status_code=400, detail="Change request target_area does not match payload")

    before = get_active_config()
    ts = now_ts()
    for key, value in values.items():
        exec_sql(
            """
            INSERT INTO ee_engine_config (key, value_json, updated_at, updated_by_user_id, change_request_id)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at,
                updated_by_user_id = excluded.updated_by_user_id,
                change_request_id = excluded.change_request_id
            """,
            (key, json.dumps(value, ensure_ascii=True), ts, actor.user_id, change_request_id),
        )

    after = get_active_config()
    audit = create_event(
        {
            "action": "config_update",
            "entity_type": "engine_config",
            "entity_id": "eagle_eye",
            "change_type": "config",
            "before_state": before,
            "after_state": after,
            "risk_level": "high",
            "metadata": {
                "change_request_id": change_request_id,
                "target_area": target_area,
                "updated_keys": sorted(values.keys()),
            },
            "concept_version": CONCEPT_VERSION,
            "source": "api",
        },
        actor,
    )

    return {
        "updated": sorted(values.keys()),
        "change_request_id": change_request_id,
        "audit_event_id": audit.get("id"),
        **get_config_with_meta(),
    }


def sync_from_legacy_cache(run_date_ts: int, source: str = "feed") -> int:
    rows = query_all(
        """
        SELECT ticker, bar_date, open, high, low, close, volume, turnover_kwd
        FROM ee_ohlcv_cache
        """,
        (),
    )
    inserted = 0
    for row in rows or []:
        symbol = _normalize_symbol(row.get("ticker"))
        if not symbol:
            continue
        bar_date = str(row.get("bar_date") or "")
        if not bar_date:
            continue
        try:
            dt = datetime.fromisoformat(bar_date)
        except Exception:
            continue
        trade_date = int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp())
        exec_sql(
            """
            INSERT INTO ee_ohlcv (
                symbol, trade_date, open, high, low, close, volume, value_kwd, source, ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, trade_date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                value_kwd = excluded.value_kwd,
                source = excluded.source,
                ingested_at = excluded.ingested_at
            """,
            (
                symbol,
                trade_date,
                row.get("open"),
                row.get("high"),
                row.get("low"),
                row.get("close"),
                row.get("volume"),
                row.get("turnover_kwd"),
                source,
                run_date_ts,
            ),
        )
        inserted += 1
    return inserted


def list_data_quality_quarantine(status: str = "quarantined") -> list[dict[str, Any]]:
    rows = query_all(
        """
        SELECT symbol, status, reason_json, first_flagged_at, last_flagged_at,
               cleared_at, change_request_id, cleared_by_user_id
        FROM ee_data_quality_quarantine
        WHERE status = ?
        ORDER BY last_flagged_at DESC, symbol ASC
        """,
        (status,),
    )
    out: list[dict[str, Any]] = []
    for row in rows or []:
        try:
            reasons = json.loads(str(row.get("reason_json") or "[]"))
        except Exception:
            reasons = []
        out.append(
            {
                "symbol": row.get("symbol"),
                "status": row.get("status"),
                "reasons": reasons if isinstance(reasons, list) else [],
                "first_flagged_at": row.get("first_flagged_at"),
                "last_flagged_at": row.get("last_flagged_at"),
                "cleared_at": row.get("cleared_at"),
                "change_request_id": row.get("change_request_id"),
                "cleared_by_user_id": row.get("cleared_by_user_id"),
            }
        )
    return out


def get_quarantined_symbols() -> set[str]:
    rows = query_all(
        "SELECT symbol FROM ee_data_quality_quarantine WHERE status = 'quarantined'",
        (),
    )
    return {str(r.get("symbol") or "").upper() for r in rows or [] if r.get("symbol")}


def _run_coro_sync(coro):
    """Run async TickerChart call from sync service code safely."""
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

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=60)
    if thread.is_alive():
        raise TimeoutError("TickerChart fetch timed out after 60s")
    if exc_box:
        raise exc_box[0]
    return result_box[0] if result_box else None


def ingest_tickerchart(
    symbols: list[str],
    start: datetime | None = None,
    end: datetime | None = None,
    source: str = "manual",
    actor: TokenData | None = None,
) -> dict[str, Any]:
    """Ingest OHLCV from TickerChart into ee_ohlcv with mandatory quality gates."""
    ensure_schema()
    actor = actor or TokenData(user_id=0, username="system", is_admin=True)
    trace_id = str(uuid.uuid4())
    cfg = get_active_config()
    max_gap_days = int(cfg.get("max_session_gap_days", 7) or 7)
    gap_check_window_days = int(cfg.get("gap_check_window_days", 30) or 30)
    validated_start = _validated_start_ts(cfg)

    # Never ingest on behalf of an unsupported non-paper mode path.
    mode = str(cfg.get("pipeline_mode", "paper") or "paper").strip().lower()
    if mode not in {"paper", "live"}:
        raise HTTPException(status_code=400, detail=f"Unsupported pipeline_mode: {mode}")

    start_d = start.date() if isinstance(start, datetime) else None
    end_d = end.date() if isinstance(end, datetime) else None

    normalized = sorted({str(s or "").upper().replace(".KW", "").strip() for s in symbols if str(s or "").strip()})
    rows_upserted = 0
    anomalies_count = 0
    quarantine_hits = 0
    processed: list[dict[str, Any]] = []
    request_params = {
        "start": start_d.isoformat() if start_d else None,
        "end": end_d.isoformat() if end_d else None,
        "interval": "day",
        "market": "KSE",
        "symbols": normalized,
    }
    request_parameters_hash = hashlib.sha256(
        json.dumps(request_params, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_payload_hash = hashlib.sha256(
        json.dumps({"source": source, "symbols": normalized}, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_id = _begin_ingestion_run(
        source_type="vendor_raw",
        source_ref="tickerchart:/tcdata/ondemandDataLoader.php",
        payload_hash=run_payload_hash,
        request_parameters_hash=request_parameters_hash,
        synthetic_flag=0,
    )

    from app.services import tickerchart_service as tc
    from app.services.eagle_eye.audit_service import create_event

    for symbol in normalized:
        try:
            raw_rows = _run_coro_sync(tc.fetch_ohlcv(symbol, "KSE", from_d=start_d, to_d=end_d, interval="day")) or []
        except Exception as exc:
            anomalies = [{"type": "fetch_error", "error": str(exc)[:240]}]
            anomalies_count += len(anomalies)
            quarantine_hits += 1
            now = now_ts()
            exec_sql(
                """
                INSERT INTO ee_data_quality_quarantine (
                    symbol, status, reason_json, first_flagged_at, last_flagged_at
                ) VALUES (?, 'quarantined', ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    status='quarantined',
                    reason_json=excluded.reason_json,
                    last_flagged_at=excluded.last_flagged_at,
                    cleared_at=NULL,
                    change_request_id=NULL,
                    cleared_by_user_id=NULL
                """,
                (symbol, json.dumps(anomalies, ensure_ascii=True), now, now),
            )
            create_event(
                {
                    "action": "data_quality_alert",
                    "entity_type": "symbol",
                    "entity_id": symbol,
                    "change_type": "data",
                    "risk_level": "high",
                    "trace_id": trace_id,
                    "source": source,
                    "requires_follow_up": True,
                    "metadata": {"anomalies": anomalies},
                    "concept_version": CONCEPT_VERSION,
                },
                actor,
            )
            processed.append({"symbol": symbol, "rows": 0, "quarantined": True, "anomalies": anomalies})
            continue

        seen_dates: set[str] = set()
        prev_td: int | None = None
        prev_close: float | None = None
        anomalies: list[dict[str, Any]] = []
        cleaned: list[tuple[int, float, float, float, float, float, float, str]] = []
        recent_window_cutoff = int((datetime.now(timezone.utc) - timedelta(days=gap_check_window_days)).timestamp())

        for row in sorted(raw_rows, key=lambda r: str(r.get("date") or "")):
            d = str(row.get("date") or "").strip()
            if not d:
                continue
            if d in seen_dates:
                anomalies.append({"type": "duplicate_trade_date", "date": d})
                continue
            seen_dates.add(d)

            try:
                dt = datetime.fromisoformat(d)
            except Exception:
                anomalies.append({"type": "invalid_trade_date", "date": d})
                continue

            td = int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp())
            o = float(row.get("open") or 0.0)
            h = float(row.get("high") or 0.0)
            l = float(row.get("low") or 0.0)
            c = float(row.get("close") or 0.0)
            v = float(row.get("volume") or 0.0)
            has_value = row.get("value") is not None
            val = _normalize_value_kwd(float(row.get("value") or 0.0) if has_value else None, c, v)
            src = "tickerchart" if has_value else "tickerchart_est_value"

            if td <= 0:
                anomalies.append({"type": "regressing_trade_date", "date": d})
                continue
            if prev_td is not None and td <= prev_td:
                anomalies.append({"type": "regressing_trade_date", "date": d})
            if c <= 0 or h <= 0 or l <= 0 or o <= 0:
                anomalies.append({"type": "non_positive_price", "date": d})
            if h < l:
                anomalies.append({"type": "high_below_low", "date": d, "high": h, "low": l})
            if (row.get("volume") is None) != (row.get("value") is None):
                anomalies.append({"type": "inconsistent_volume_value_presence", "date": d})
            if prev_td is not None and td >= recent_window_cutoff:
                gap_days = int((td - prev_td) / 86400)
                if gap_days > max_gap_days:
                    anomalies.append({"type": "session_gap", "date": d, "gap_days": gap_days, "max_gap_days": max_gap_days})
            if prev_close is not None and prev_close > 0 and td >= validated_start:
                jump = abs((c / prev_close) - 1.0)
                if jump > 0.25:
                    anomalies.append(
                        {
                            "type": "price_jump_gt_25pct",
                            "date": d,
                            "jump_abs": jump,
                            "pct": jump * 100.0,
                            "prior_close": prev_close,
                            "rejected_close": c,
                        }
                    )

            prev_td = td
            prev_close = c
            cleaned.append((td, o, h, l, c, v, val, src))

        if anomalies:
            anomalies_count += len(anomalies)
            quarantine_hits += 1
            now = now_ts()
            exec_sql(
                """
                INSERT INTO ee_data_quality_quarantine (
                    symbol, status, reason_json, first_flagged_at, last_flagged_at
                ) VALUES (?, 'quarantined', ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    status='quarantined',
                    reason_json=excluded.reason_json,
                    last_flagged_at=excluded.last_flagged_at,
                    cleared_at=NULL,
                    change_request_id=NULL,
                    cleared_by_user_id=NULL
                """,
                (symbol, json.dumps(anomalies, ensure_ascii=True), now, now),
            )
            create_event(
                {
                    "action": "data_quality_alert",
                    "entity_type": "symbol",
                    "entity_id": symbol,
                    "change_type": "data",
                    "risk_level": "high",
                    "trace_id": trace_id,
                    "source": source,
                    "requires_follow_up": True,
                    "metadata": {"anomalies": anomalies},
                    "concept_version": CONCEPT_VERSION,
                },
                actor,
            )
            processed.append({"symbol": symbol, "rows": 0, "quarantined": True, "anomalies": anomalies})
            continue

        # Auto-clear stale quarantine when an ingest run passes all quality gates.
        exec_sql(
            "UPDATE ee_data_quality_quarantine SET status='cleared', cleared_at=?, last_flagged_at=? WHERE symbol=? AND status='quarantined'",
            (now_ts(), now_ts(), symbol),
        )

        upserted = 0
        for td, o, h, l, c, v, val, src in cleaned:
            row_payload_hash = hashlib.sha256(
                json.dumps(
                    {
                        "symbol": symbol,
                        "trade_date": td,
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": v,
                        "value_kwd": val,
                        "source": src,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            wrote = _upsert_ohlcv_row(
                symbol=symbol,
                trade_date=td,
                open_v=o,
                high_v=h,
                low_v=l,
                close_v=c,
                volume_v=v,
                value_kwd_v=val,
                source=src,
                source_type="vendor_raw",
                source_ref="tickerchart:/tcdata/ondemandDataLoader.php",
                run_id=run_id,
                request_parameters_hash=request_parameters_hash,
                payload_hash=row_payload_hash,
                synthetic_flag=0,
                adjustment_status="raw_unadjusted",
                corporate_action_version="none",
            )
            if wrote:
                upserted += 1

        rows_upserted += upserted
        processed.append({"symbol": symbol, "rows": upserted, "quarantined": False, "anomalies": []})

    min_td = query_val(
        "SELECT MIN(trade_date) FROM ee_ohlcv WHERE symbol IN (" + ",".join(["?"] * len(normalized)) + ")",
        tuple(normalized),
    ) if normalized else None
    max_td = query_val(
        "SELECT MAX(trade_date) FROM ee_ohlcv WHERE symbol IN (" + ",".join(["?"] * len(normalized)) + ")",
        tuple(normalized),
    ) if normalized else None

    from app.services.eagle_eye.audit_service import create_event

    ingest_event = create_event(
        {
            "action": "data_ingest",
            "entity_type": "pipeline",
            "entity_id": f"tickerchart:{trace_id}",
            "change_type": "data",
            "risk_level": "high" if anomalies_count else "low",
            "trace_id": trace_id,
            "source": source,
            "metadata": {
                "symbols": normalized,
                "rows_upserted": rows_upserted,
                "date_range": {"min_trade_date": min_td, "max_trade_date": max_td},
                "anomalies_count": anomalies_count,
                "quarantined_symbols": quarantine_hits,
                "pipeline_mode": mode,
            },
            "concept_version": CONCEPT_VERSION,
        },
        actor,
    )

    _finalize_ingestion_run(run_id, rows_upserted, status="completed")

    return {
        "trace_id": trace_id,
        "symbols": normalized,
        "rows_upserted": rows_upserted,
        "anomalies_count": anomalies_count,
        "quarantined_symbols": quarantine_hits,
        "audit_event_id": ingest_event.get("id"),
        "processed": processed,
        "advice": False,
    }


def list_symbols() -> list[str]:
    rows = query_all("SELECT DISTINCT symbol FROM ee_ohlcv ORDER BY symbol", ())
    return [str(r.get("symbol")) for r in rows or [] if r.get("symbol")]


def load_symbol_ohlcv(symbol: str) -> list[dict[str, Any]]:
    rows = query_all(
        """
        SELECT trade_date, open, high, low, close, volume, value_kwd
        FROM ee_ohlcv
        WHERE symbol = ?
        ORDER BY trade_date
        """,
        (_normalize_symbol(symbol),),
    )
    return [dict(r) for r in rows or []]


def latest_trade_date() -> int | None:
    value = query_val("SELECT MAX(trade_date) FROM ee_ohlcv", ())
    return int(value) if value is not None else None


def _normalize_csv_columns(columns: list[str]) -> dict[str, str]:
    aliases = {
        "date": {"date", "trade_date", "session_date", "تاريخ", "التاريخ"},
        "open": {"open", "o", "افتتاح", "الافتتاح"},
        "high": {"high", "h", "اعلى", "الأعلى", "اعلي"},
        "low": {"low", "l", "ادنى", "الأدنى", "ادني"},
        "close": {"close", "c", "اغلاق", "الإغلاق", "اغلاق*"},
        "volume": {"volume", "vol", "qty", "كمية", "الكمية"},
        "value": {"value", "turnover", "value_kwd", "قيمة", "القيمة", "التداول"},
    }
    lowered = {c: re.sub(r"\s+", "", str(c).strip().lower()) for c in columns}
    out: dict[str, str] = {}
    for canonical, names in aliases.items():
        for original, key in lowered.items():
            if key in names:
                out[canonical] = original
                break
    return out


def load_ohlcv_csv(path: str, symbol: str, actor: TokenData | None = None, source: str = "csv") -> dict[str, Any]:
    ensure_schema()
    actor = actor or TokenData(user_id=0, username="system", is_admin=True)

    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding="utf-8-sig")

    cols = _normalize_csv_columns(list(df.columns))
    required = {"date", "open", "high", "low", "close", "volume", "value"}
    missing = sorted(required - set(cols.keys()))
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV missing required columns: {', '.join(missing)}")

    data = pd.DataFrame(
        {
            "date": df[cols["date"]],
            "open": df[cols["open"]],
            "high": df[cols["high"]],
            "low": df[cols["low"]],
            "close": df[cols["close"]],
            "volume": df[cols["volume"]],
            "value": df[cols["value"]],
        }
    )

    data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "value"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")

    if data.empty:
        raise HTTPException(status_code=400, detail="CSV contains no valid OHLC rows")

    if not data["date"].is_monotonic_increasing:
        raise HTTPException(status_code=400, detail="CSV dates are not monotonic increasing")

    if (data[["open", "high", "low", "close"]] < 0).any().any():
        raise HTTPException(status_code=400, detail="CSV contains negative prices")

    sym = _normalize_symbol(symbol)
    synthetic_fixture = _is_synthetic_fixture_source(path, source)
    if synthetic_fixture and sym in _real_market_symbols():
        _emit_ingest_reject_audit(
            symbol=sym,
            trade_date=0,
            existing_type="n/a",
            incoming_type="csv_fixture" if synthetic_fixture else "csv_import",
            reason="synthetic_or_debug_source_rejected_for_real_symbol",
            metadata={"path": str(path), "source": str(source)},
        )
        raise HTTPException(
            status_code=400,
            detail=f"Synthetic/debug fixture symbol is not allowed for real market ticker: {sym}",
        )

    request_parameters_hash = hashlib.sha256(
        json.dumps({"path": str(path), "source": source, "symbol": sym}, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    file_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    run_id = _begin_ingestion_run(
        source_type="csv_fixture" if synthetic_fixture else "csv_import",
        source_ref=f"file:{Path(path).resolve().as_posix()}",
        payload_hash=file_hash,
        request_parameters_hash=request_parameters_hash,
        synthetic_flag=1 if synthetic_fixture else 0,
    )

    quality_events = []

    session_gaps = data["date"].diff().dt.days.fillna(1)
    gap_rows = data.loc[session_gaps > 7]
    for _, row in gap_rows.iterrows():
        quality_events.append(
            {
                "type": "gap",
                "date": int(row["date"].replace(tzinfo=timezone.utc).timestamp()),
                "days": int(session_gaps.loc[row.name]),
            }
        )

    ret = data["close"].pct_change().abs().fillna(0.0)
    jump_rows = data.loc[ret > 0.25]
    for _, row in jump_rows.iterrows():
        quality_events.append(
            {
                "type": "jump",
                "date": int(row["date"].replace(tzinfo=timezone.utc).timestamp()),
                "return_abs": float(ret.loc[row.name]),
            }
        )

    inserted = 0
    forced_fail_after = int(os.getenv("EE_FAIL_BATCH_AFTER_ROWS") or "0")
    try:
        for _, row in data.iterrows():
            trade_date = int(row["date"].replace(tzinfo=timezone.utc).timestamp())
            row_payload_hash = hashlib.sha256(
                json.dumps(
                    {
                        "symbol": sym,
                        "trade_date": trade_date,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"] or 0.0),
                        "value": float(row["value"] or 0.0),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            wrote = _upsert_ohlcv_row(
                symbol=sym,
                trade_date=trade_date,
                open_v=float(row["open"]),
                high_v=float(row["high"]),
                low_v=float(row["low"]),
                close_v=float(row["close"]),
                volume_v=float(row["volume"] or 0.0),
                value_kwd_v=float(row["value"] or 0.0),
                source=source,
                source_type="csv_fixture" if synthetic_fixture else "csv_import",
                source_ref=f"file:{Path(path).resolve().as_posix()}",
                run_id=run_id,
                request_parameters_hash=request_parameters_hash,
                payload_hash=row_payload_hash,
                synthetic_flag=1 if synthetic_fixture else 0,
                adjustment_status="raw_unadjusted",
                corporate_action_version="none",
            )
            if wrote:
                inserted += 1

            if forced_fail_after > 0 and inserted >= forced_fail_after:
                raise RuntimeError("EE_FORCED_BATCH_FAILURE")
    except Exception:
        # Compensating rollback: ensure failed batches leave no partial writes.
        exec_sql("DELETE FROM ee_ohlcv WHERE ingestion_run_id = ?", (run_id,))
        _finalize_ingestion_run(run_id, 0, status="failed")
        raise

    for ev in quality_events:
        create_event(
            {
                "action": "data_quality_alert",
                "entity_type": "symbol",
                "entity_id": sym,
                "change_type": "data",
                "risk_level": "high",
                "source": "manual",
                "requires_follow_up": True,
                "metadata": {"quality": ev, "path": path},
                "concept_version": CONCEPT_VERSION,
            },
            actor,
        )

    _finalize_ingestion_run(run_id, inserted, status="completed")

    return {
        "symbol": sym,
        "rows": inserted,
        "quality_alerts": len(quality_events),
        "advice": False,
    }
