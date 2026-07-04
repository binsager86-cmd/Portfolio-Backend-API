from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import HTTPException

from app.core.database import exec_sql, query_all, query_one, query_val
from app.core.security import TokenData
from app.services.eagle_eye.audit_service import create_event

logger = logging.getLogger(__name__)

CONCEPT_VERSION = "ee-2.1.0-verification"
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
    "accumulation_min_score": "scanner",
    "breakout_min_score": "scanner",
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
    "accumulation_min_score": 60,
    "breakout_min_score": 70,
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
}


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
            open REAL, high REAL, low REAL, close REAL, volume REAL, value_kwd REAL,
            source TEXT NOT NULL DEFAULT 'feed', ingested_at INTEGER NOT NULL,
            PRIMARY KEY (symbol, trade_date)
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
    ]

    for stmt in stmts:
        exec_sql(stmt, ())

    _seed_default_config()


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
    for _, row in data.iterrows():
        trade_date = int(row["date"].replace(tzinfo=timezone.utc).timestamp())
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
                sym,
                trade_date,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"] or 0.0),
                float(row["value"] or 0.0),
                source,
                now_ts(),
            ),
        )
        inserted += 1

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

    return {
        "symbol": sym,
        "rows": inserted,
        "quality_alerts": len(quality_events),
        "advice": False,
    }
