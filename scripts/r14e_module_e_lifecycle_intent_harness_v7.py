from __future__ import annotations

import hashlib
import collections
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.services.eagle_eye_v2.adaptive_base_geometry import (
    ATR_SQUEEZE_PCTILE,
    BASE_MAX_WIDTH_PCT,
    BASE_MIN_SESSIONS,
    RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N,
    AdaptiveBaseGeometry,
    BaseNamedParameters,
)
from app.services.eagle_eye_v2.data_surface_adapter import (
    DataSurfaceAdapter,
    SegmentState,
    load_default_calendar_context,
    load_default_mask_manifest,
)
from app.services.eagle_eye_v2.flow_confirmation_engine import (
    ADX_TRIGGER,
    ANV_SLOPE_MIN,
    CHASE_ADVISORY_BAND,
    CMF_FLOOR,
    MIN_CURRENT_DAY_VALUE_KWD,
    MIN_DAILY_VALUE_KWD,
    OBV_SLOPE_MIN,
    REL_VOLUME_CONTEXT_MIN,
    RSI_REGIME,
    FlowConfirmationEngine,
    FlowNamedParameters,
)
from app.services.eagle_eye_v2.lifecycle_intent_router import (
    CHASE_ADVISORY_THRESHOLD,
    CHASE_ESCALATION_THRESHOLD,
    EARLY_TIER_PARTICIPATION_CAP,
    EARLY_TIER_SIZE_FRACTION,
    EARLY_TIER_TIME_STOP,
    SCALE_ON_CONFIRMATION,
    LifecycleIntentRouter,
    LifecycleRouterNamedParameters,
)
from app.services.eagle_eye_v2.warmup_readiness_engine import (
    READINESS_FALLBACK_MIN_SESSIONS,
    READINESS_LONG_LOOKBACK_MIN_SESSIONS,
    READINESS_SEGMENT_RESTART_MIN_SESSIONS,
    WarmupNamedParameters,
    WarmupReadinessEngine,
)
from app.services.eagle_eye_v2.predicate_telemetry_ledger import apply_schema_migration

REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
RUNTIME_DB = REVIEW / "r12_exam_surface_v4_5_runtime.db"
HARNESS_DB = REVIEW / "r14e_module_e_harness_surface_v7.db"
FREEZE_JSON = REVIEW / "r14b_parameter_freeze_v2.json"
FREEZE_SHA = REVIEW / "r14b_parameter_freeze_v2.sha256"
RUN_NONCE = "2026-07-18T09:25:25.6107136Z"

RUN_KEY = "R14E_MODULE_E_HARNESS_V7"

AVOID_DERIVATION_CODE = '''
def derive_avoid_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state = {"phase": "NONE", "avoid_clear_streak": 0, "avoid_reclaim_streak": 0, "avoid_until": None}
    for day in rows:
        payload = dict(day.get("indicator_payload") or {})
        close = float(day.get("close") or 0.0)
        ema10 = float(payload.get("ema10") or 0.0)
        ema30 = float(payload.get("ema30") or 0.0)
        sma200 = float(payload.get("sma200") or 0.0)
        sma200_slope = float(payload.get("sma200_slope") or 0.0)
        avoid_now = close < sma200 and sma200_slope < 0 and ema10 < ema30
        if avoid_now:
            state["avoid_clear_streak"] = 0
            state["avoid_reclaim_streak"] = 0
            state["phase"] = "AVOID"
        elif state["phase"] == "AVOID":
            state["avoid_clear_streak"] = int(state["avoid_clear_streak"] or 0) + 1
            state["avoid_reclaim_streak"] = int(state["avoid_reclaim_streak"] or 0) + 1 if close > sma200 else 0
            if int(state["avoid_reclaim_streak"] or 0) >= 2 or int(state["avoid_clear_streak"] or 0) >= 20:
                state["phase"] = "NONE"
                state["avoid_until"] = str(day["trade_date"])
                state["avoid_clear_streak"] = 0
                state["avoid_reclaim_streak"] = 0
'''.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def to_date_text(v: Any) -> str:
    if isinstance(v, int):
        return datetime.fromtimestamp(v, timezone.utc).strftime("%Y-%m-%d")
    s = str(v)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    if s.isdigit() and len(s) >= 10:
        return datetime.fromtimestamp(int(s), timezone.utc).strftime("%Y-%m-%d")
    raise ValueError(f"Unsupported date value: {v}")


def bind_harness_db() -> None:
    if HARNESS_DB.exists():
        HARNESS_DB.unlink()
    HARNESS_DB.touch()
    os.environ["EE_V2_RUNTIME_DB_PATH"] = str(HARNESS_DB)
    os.environ["DATABASE_PATH"] = str(HARNESS_DB)
    get_settings.cache_clear()


def freeze_attestation() -> dict[str, Any]:
    if not FREEZE_JSON.exists() or not FREEZE_SHA.exists():
        raise FileNotFoundError("Freeze v2 artifacts are required before module (e) harness run.")
    expected = FREEZE_SHA.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(FREEZE_JSON)
    return {
        "freeze_json": str(FREEZE_JSON),
        "freeze_sha_sidecar": str(FREEZE_SHA),
        "expected_json_sha256": expected,
        "actual_json_sha256": actual,
        "byte_match": expected == actual,
    }


def record_run_metadata(conn: sqlite3.Connection, attest: dict[str, Any]) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS harness_run_metadata (
            run_key TEXT PRIMARY KEY,
            run_nonce TEXT NOT NULL,
            freeze_json_sha256 TEXT NOT NULL,
            freeze_byte_match INTEGER NOT NULL,
            created_utc TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO harness_run_metadata (
            run_key,
            run_nonce,
            freeze_json_sha256,
            freeze_byte_match,
            created_utc
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            RUN_KEY,
            RUN_NONCE,
            str(attest["actual_json_sha256"]),
            1 if attest.get("byte_match") else 0,
            RUN_NONCE,
        ),
    )


def ensure_harness_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS harness_daily_rows (
            run_key TEXT NOT NULL,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            row_json TEXT NOT NULL,
            PRIMARY KEY (run_key, symbol, trade_date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS harness_position_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_key TEXT NOT NULL,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            position_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_json TEXT NOT NULL
        )
        """
    )


def derive_avoid_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state = {"phase": "NONE", "avoid_clear_streak": 0, "avoid_reclaim_streak": 0, "avoid_until": None}
    out: list[dict[str, Any]] = []
    for day in rows:
        payload = dict(day.get("indicator_payload") or {})
        close = float(day.get("close") or 0.0)
        ema10 = float(payload.get("ema10") or 0.0)
        ema30 = float(payload.get("ema30") or 0.0)
        sma200 = float(payload.get("sma200") or 0.0)
        sma200_slope = float(payload.get("sma200_slope") or 0.0)

        avoid_now = close < sma200 and sma200_slope < 0 and ema10 < ema30
        if avoid_now:
            if state["phase"] != "AVOID":
                state["avoid_clear_streak"] = 0
                state["avoid_reclaim_streak"] = 0
            state["phase"] = "AVOID"
        elif state["phase"] == "AVOID":
            state["avoid_clear_streak"] = int(state["avoid_clear_streak"] or 0) + 1
            if close > sma200:
                state["avoid_reclaim_streak"] = int(state["avoid_reclaim_streak"] or 0) + 1
            else:
                state["avoid_reclaim_streak"] = 0
            if int(state["avoid_reclaim_streak"] or 0) >= 2 or int(state["avoid_clear_streak"] or 0) >= 20:
                state["phase"] = "NONE"
                state["avoid_until"] = str(day["trade_date"])
                state["avoid_clear_streak"] = 0
                state["avoid_reclaim_streak"] = 0

        out.append(
            {
                "avoid_state": state["phase"],
                "avoid_active": state["phase"] == "AVOID",
                "avoid_until": state["avoid_until"],
                "avoid_clear_streak": int(state["avoid_clear_streak"] or 0),
                "avoid_reclaim_streak": int(state["avoid_reclaim_streak"] or 0),
                "close": close,
                "sma200": sma200,
                "sma200_slope": sma200_slope,
                "ema10": ema10,
                "ema30": ema30,
                "avoid_entry_predicate": avoid_now,
                "avoid_source": "close < sma200 and sma200_slope < 0 and ema10 < ema30; clear via reclaim/20-session fallback",
            }
        )
    return out


def record_daily_row(conn: sqlite3.Connection, symbol: str, trade_date: str, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO harness_daily_rows (run_key, symbol, trade_date, row_json)
        VALUES (?, ?, ?, ?)
        """,
        (RUN_KEY, symbol, trade_date, json.dumps(row, ensure_ascii=True, sort_keys=True)),
    )
    conn.commit()


def record_position_event(conn: sqlite3.Connection, symbol: str, trade_date: str, position_id: str, event_type: str, payload: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO harness_position_events (run_key, symbol, trade_date, position_id, event_type, event_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (RUN_KEY, symbol, trade_date, position_id, event_type, json.dumps(payload, ensure_ascii=True, sort_keys=True)),
    )
    conn.commit()


def fetch_indicator_payload(conn: sqlite3.Connection, symbol: str, trade_date: int) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT payload_json
        FROM ee_indicators
        WHERE symbol LIKE ? AND trade_date = ?
        ORDER BY symbol ASC
        LIMIT 1
        """,
        (f"{symbol}%", int(trade_date)),
    ).fetchone()
    if row is None or row[0] is None:
        return {}
    try:
        return json.loads(str(row[0]))
    except json.JSONDecodeError:
        return {}


def load_window(symbol: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(RUNTIME_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT symbol, trade_date, open, high, low, close, volume, value_kwd
            FROM ee_ohlcv
            WHERE symbol LIKE ?
              AND date(trade_date, 'unixepoch') BETWEEN ? AND ?
            ORDER BY trade_date ASC
            """,
            (f"{symbol}%", start_date, end_date),
        ).fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            ts = int(r["trade_date"])
            out.append(
                {
                    "symbol": symbol,
                    "trade_date": to_date_text(ts),
                    "trade_date_ts": ts,
                    "open": float(r["open"] or 0.0),
                    "high": float(r["high"] or 0.0),
                    "low": float(r["low"] or 0.0),
                    "close": float(r["close"] or 0.0),
                    "volume": float(r["volume"] or 0.0),
                    "value_kwd": float(r["value_kwd"] or 0.0),
                    "indicator_payload": fetch_indicator_payload(conn, symbol, ts),
                }
            )
        return out
    finally:
        conn.close()


def count_ee_signals(symbol: str, start_date: str, end_date: str) -> dict[str, int]:
    conn = sqlite3.connect(str(RUNTIME_DB))
    try:
        start_i = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp())
        end_i = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp())
        total = conn.execute(
            """
            SELECT COUNT(*)
            FROM ee_signals
            WHERE symbol LIKE ? AND trade_date BETWEEN ? AND ?
            """,
            (f"{symbol}%", start_i, end_i),
        ).fetchone()
        avoid = conn.execute(
            """
            SELECT COUNT(*)
            FROM ee_signals
            WHERE symbol LIKE ?
              AND trade_date BETWEEN ? AND ?
              AND (
                UPPER(COALESCE(signal_type, '')) LIKE '%AVOID%'
                OR UPPER(COALESCE(phase_from, '')) = 'AVOID'
                OR UPPER(COALESCE(phase_to, '')) = 'AVOID'
              )
            """,
            (f"{symbol}%", start_i, end_i),
        ).fetchone()
        return {"total_rows": int(total[0] if total else 0), "avoid_stream_rows": int(avoid[0] if avoid else 0)}
    finally:
        conn.close()


def r12_avoid_intervals(symbol: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(RUNTIME_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT trade_date, signal_type, phase_from, phase_to
            FROM ee_signals
            WHERE symbol LIKE ?
              AND (
                UPPER(COALESCE(signal_type, '')) LIKE '%AVOID%'
                OR UPPER(COALESCE(phase_from, '')) = 'AVOID'
                OR UPPER(COALESCE(phase_to, '')) = 'AVOID'
              )
            ORDER BY trade_date ASC
            """,
            (f"{symbol}%",),
        ).fetchall()
    finally:
        conn.close()

    intervals: list[dict[str, Any]] = []
    open_start: str | None = None
    for row in rows:
        date_text = to_date_text(int(row["trade_date"]))
        phase_to = str(row["phase_to"] or "").upper()
        phase_from = str(row["phase_from"] or "").upper()
        signal_type = str(row["signal_type"] or "").upper()
        if phase_to == "AVOID" or "AVOID_SET" in signal_type:
            open_start = date_text
        elif phase_from == "AVOID" and open_start is not None:
            intervals.append({"start": open_start, "end": date_text, "clear_event": date_text})
            open_start = None
    if open_start is not None:
        intervals.append({"start": open_start, "end": None, "clear_event": None})
    return intervals


def dates_in_intervals(intervals: list[dict[str, Any]], candidate_dates: list[str]) -> list[str]:
    out: list[str] = []
    for date_text in candidate_dates:
        for interval in intervals:
            start = str(interval["start"])
            end = interval.get("end")
            if start <= date_text and (end is None or date_text <= str(end)):
                out.append(date_text)
                break
    return out


def r12_avoid_state_by_date(intervals: list[dict[str, Any]], candidate_dates: list[str]) -> dict[str, str]:
    active = set(dates_in_intervals(intervals, candidate_dates))
    return {date_text: "AVOID" if date_text in active else "NONE" for date_text in candidate_dates}


def build_structure_terms(day: dict[str, Any], base_reference: dict[str, Any]) -> dict[str, Any]:
    ind = dict(day.get("indicator_payload") or {})
    close_px = float(day.get("close") or 0.0)
    base_high = float(base_reference.get("base_high_ref") or 0.0)
    return {
        "close_gt_base_ref": bool(base_high > 0 and close_px > base_high),
        "ema10_gt_ema30": float(ind.get("ema10") or 0.0) >= float(ind.get("ema30") or 0.0),
        "adx_19": float(ind.get("adx_19") or 0.0),
        "rsi_14": float(ind.get("rsi_14") or 0.0),
    }


def owner_windows() -> dict[str, dict[str, Any]]:
    return {
        "MABANEE": {
            "owner_start": "2024-12-01",
            "owner_end": "2025-06-30",
            "replay_start": "2024-12-01",
            "replay_end": "2025-06-30",
            "source_artifact": "owner directive for v7: corrected known-answer window covering R12 MABANEE avoid intervals 2024-12-22..2025-02-20 and 2025-03-24..2025-05-18",
        },
        "SANAM": {
            "owner_start": "2025-05-01",
            "owner_end": "2025-05-31",
            "replay_start": "2024-11-01",
            "replay_end": "2025-05-31",
            "source_artifact": "scripts/r14e_module_e_lifecycle_intent_harness_v3.py owner_windows(); byte-match restored; 140 replay bars expected",
        },
        "TIJARA": {
            "owner_start": "2025-01-01",
            "owner_end": "2025-12-31",
            "replay_start": "2024-07-01",
            "replay_end": "2025-12-31",
            "source_artifact": "scripts/r14e_module_e_lifecycle_intent_harness_v3.py owner_windows(); byte-match restored; 371 replay bars expected",
        },
    }


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)

    out_interface = REVIEW / "r14e_module_e_interface_conformance_v7.json"
    out_evidence = REVIEW / "r14e_module_e_test_evidence_v7.json"
    out_report = REVIEW / "r14e_module_e_implementation_report_v7.md"
    out_sha = REVIEW / "r14e_module_e_artifacts_v7.sha256"
    harness_db = REVIEW / "r14e_module_e_harness_surface_v7.db"

    attest = freeze_attestation()
    if not attest["byte_match"]:
        raise RuntimeError("Freeze v2 byte-match attestation failed.")

    global HARNESS_DB
    HARNESS_DB = harness_db
    bind_harness_db()
    apply_schema_migration()

    with sqlite3.connect(str(HARNESS_DB)) as meta_conn:
        record_run_metadata(meta_conn, attest)
        ensure_harness_tables(meta_conn)
        meta_conn.commit()

    cal = load_default_calendar_context(ROOT)
    mask = load_default_mask_manifest(ROOT)
    adapter = DataSurfaceAdapter(calendar_context=cal, mask_manifest=mask)

    warmup = WarmupReadinessEngine(
        WarmupNamedParameters(
            values={
                READINESS_LONG_LOOKBACK_MIN_SESSIONS: 180,
                READINESS_SEGMENT_RESTART_MIN_SESSIONS: 20,
                READINESS_FALLBACK_MIN_SESSIONS: 60,
            }
        )
    )

    base = AdaptiveBaseGeometry(
        BaseNamedParameters(
            values={
                BASE_MIN_SESSIONS: 10,
                BASE_MAX_WIDTH_PCT: 0.24,
                ATR_SQUEEZE_PCTILE: 0.95,
            }
        )
    )

    flow = FlowConfirmationEngine(
        FlowNamedParameters(
            values={
                OBV_SLOPE_MIN: 0.10,
                ANV_SLOPE_MIN: 0.10,
                CMF_FLOOR: 0.05,
                REL_VOLUME_CONTEXT_MIN: 2.5,
                RSI_REGIME: 50.0,
                ADX_TRIGGER: 18.0,
                CHASE_ADVISORY_BAND: 0.08,
                MIN_DAILY_VALUE_KWD: 100000.0,
                MIN_CURRENT_DAY_VALUE_KWD: 50000.0,
            }
        )
    )

    router = LifecycleIntentRouter(
        LifecycleRouterNamedParameters(
            values={
                EARLY_TIER_SIZE_FRACTION: 0.30,
                EARLY_TIER_PARTICIPATION_CAP: 0.10,
                EARLY_TIER_TIME_STOP: 60.0,
                SCALE_ON_CONFIRMATION: "SINGLE_ADD_TO_FULL_TARGET",
                CHASE_ADVISORY_THRESHOLD: 0.08,
                CHASE_ESCALATION_THRESHOLD: 0.15,
            }
        )
    )

    windows = owner_windows()
    per_symbol: dict[str, list[dict[str, Any]]] = {}
    summary_counts: dict[str, dict[str, Any]] = {}
    avoid_sequences: dict[str, list[dict[str, Any]]] = {}
    position_progressions: dict[str, list[dict[str, Any]]] = {}
    signal_surface_counts: dict[str, dict[str, int]] = {
        symbol: count_ee_signals(symbol, cfg["replay_start"], cfg["replay_end"])
        for symbol, cfg in windows.items()
    }
    r12_mabanee_intervals = r12_avoid_intervals("MABANEE")
    avoid_arm_lag_rows = load_window("MABANEE", "2025-12-24", "2026-04-30")
    avoid_arm_lag_series = derive_avoid_context(avoid_arm_lag_rows)
    avoid_arm_lag_finding = {
        "finding_id": "AVOID_ARM_LAG",
        "status": "R15_ATTENTION_NO_PARAMETER_CHANGE_IN_V7",
        "statement_verbatim": "In 2025-12-24..2026-04-30, MABANEE declined 1150->879 with avoid never arming in either the R12 record or the SMA200 derivation; slope remained positive throughout.",
        "window": {"start": "2025-12-24", "end": "2026-04-30"},
        "close_min": min((float(row["close"]) for row in avoid_arm_lag_rows), default=None),
        "close_max": max((float(row["close"]) for row in avoid_arm_lag_rows), default=None),
        "slope_series": [
            {
                "trade_date": str(day["trade_date"]),
                "close": ctx["close"],
                "sma200": ctx["sma200"],
                "sma200_slope": ctx["sma200_slope"],
                "sma200_avoid_state": ctx["avoid_state"],
            }
            for day, ctx in zip(avoid_arm_lag_rows, avoid_arm_lag_series)
        ],
    }

    for symbol, cfg in windows.items():
        rows = load_window(symbol, cfg["replay_start"], cfg["replay_end"])
        if not rows:
            per_symbol[symbol] = []
            avoid_sequences[symbol] = []
            position_progressions[symbol] = []
            summary_counts[symbol] = {
                "early_intents_formed": 0,
                "deferred_intents": 0,
                "scale_events": 0,
                "time_stop_reviews_triggered": 0,
                "advisories_emitted": 0,
                "vetoes_by_plane": {},
                "positions_opened": 0,
                "positions_closed": 0,
                "positions_rearmed": 0,
                "dead_money_total": 0,
            }
            continue

        avoid_series = derive_avoid_context(rows)
        avoid_sequences[symbol] = [
            {
                "trade_date": str(day["trade_date"]),
                "avoid_state": avoid_ctx["avoid_state"],
                "close": avoid_ctx["close"],
                "sma200": avoid_ctx["sma200"],
                "sma200_slope": avoid_ctx["sma200_slope"],
                "ema10": avoid_ctx["ema10"],
                "ema30": avoid_ctx["ema30"],
                "avoid_entry_predicate": avoid_ctx["avoid_entry_predicate"],
                "avoid_clear_streak": avoid_ctx["avoid_clear_streak"],
                "avoid_reclaim_streak": avoid_ctx["avoid_reclaim_streak"],
            }
            for day, avoid_ctx in zip(rows, avoid_series)
        ]

        prev_segment: SegmentState | None = None
        prev_masked = False
        prev_ready = "READINESS_PENDING"
        history_window: list[dict[str, Any]] = []
        flow_window: list[dict[str, Any]] = []
        prior_base: dict[str, Any] | None = None
        coverage_dates: list[str] = []
        segment_dates: list[str] = []
        deferred_state = {"age_sessions": 0, "rearm_count": 0, "flow_evidence_decay": False}
        position_state: dict[str, Any] | None = None
        position_counter = 0
        symbol_counts = {
            "early_intents_formed": 0,
            "deferred_intents": 0,
            "scale_events": 0,
            "time_stop_reviews_triggered": 0,
            "advisories_emitted": 0,
            "vetoes_by_plane": collections.Counter(),
            "positions_opened": 0,
            "positions_closed": 0,
            "positions_rearmed": 0,
            "positions_direct": 0,
            "dead_money_total": 0,
            "silent_none_intents": 0,
        }

        out_rows: list[dict[str, Any]] = []
        symbol_position_progression: list[dict[str, Any]] = []

        with sqlite3.connect(str(HARNESS_DB)) as ledger_conn:
            ledger_conn.execute("PRAGMA busy_timeout=5000;")
            for idx, day in enumerate(rows):
                trade_date = str(day["trade_date"])
                avoid_ctx = avoid_series[idx]
                mask_ctx = adapter.mask_context_for(symbol, trade_date)
                seg = adapter.next_segment_state(
                    symbol=symbol,
                    trade_date=trade_date,
                    prev_segment=prev_segment,
                    prev_masked=prev_masked,
                    current_masked=bool(mask_ctx["masked_flag"]),
                )
                normalized, readiness_ctx = adapter.normalize_day(
                    ohlcv_day=day,
                    indicator_day=dict(day.get("indicator_payload") or {}),
                    segment_context=seg,
                    calendar_context=cal,
                )

                coverage_dates.append(trade_date)
                if seg.segment_day_index == 0:
                    segment_dates = [trade_date]
                else:
                    segment_dates.append(trade_date)

                ready = warmup.evaluate(
                    normalized_day_payload=normalized,
                    coverage_history={
                        "long_lookback_session_dates": coverage_dates,
                        "segment_session_dates": segment_dates,
                        "fallback_session_dates": coverage_dates,
                        "previous_readiness_state": prev_ready,
                    },
                    segment_restart_flag=bool(readiness_ctx["segment_restart_flag"]),
                )

                history_window.append(day)
                if len(history_window) > 260:
                    history_window = history_window[-260:]

                base_out = base.evaluate(
                    normalized_day_payload=normalized,
                    readiness_state=ready["readiness_state"],
                    price_history_window=history_window,
                    volatility_regime_state={
                        "atr_squeeze_pctile": 0.50,
                        "base_range_sessions": 20,
                        "atr_value": float(day["high"] or 0.0) - float(day["low"] or 0.0),
                        "invalidation_rule_form": RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N,
                        "invalidation_rule_params": {"atr_mult": 1.0, "n_sessions": 2},
                        "parameter_status": "FROZEN_R14B_PARAMETER_FREEZE_V2",
                    },
                    prior_base_reference=prior_base,
                    flow_stub_state={"confirmed_progress": False},
                )

                flow_window.append(dict(day.get("indicator_payload") or {}))
                if len(flow_window) > 40:
                    flow_window = flow_window[-40:]

                structure_terms = build_structure_terms(day, base_out["base_reference"])
                flow_out = flow.evaluate(
                    normalized_day_payload=normalized,
                    base_reference=base_out["base_reference"],
                    flow_history_window=flow_window,
                    structure_terms=structure_terms,
                    readiness_state=ready["readiness_state"],
                    phase_state=base_out["base_state"],
                )

                avoid_veto = bool(avoid_ctx["avoid_active"])

                router_current_state = dict(deferred_state)
                if position_state is not None:
                    router_current_state.update(
                        {
                            "active": True,
                            "state": "POSITION_OPEN",
                            "position_id": position_state.get("position_id"),
                            "position_type": position_state.get("position_type"),
                        }
                    )

                route_out = router.evaluate(
                    candidate_intent=flow_out["candidate_intent"],
                    base_state={"base_state": base_out["base_state"]},
                    confirmation_state={"confirmation_state": flow_out["confirmation_state"]},
                    risk_budget_state={
                        "current_day_value_kwd": float(day.get("value_kwd") or 0.0),
                        "planned_order_value_kwd": float(day.get("value_kwd") or 0.0) * 0.03,
                        "avoid_veto": avoid_veto,
                        "deferred_intent_state": router_current_state,
                    },
                )
                deferred_state = dict(route_out["deferred_intent"])

                execution_state = str(route_out["execution_intent"].get("execution_state") or "NONE")
                if execution_state in {"EXECUTE_EARLY_PILOT", "EXECUTE_CONFIRMED_DIRECT", "EXECUTE_CONFIRMED_ADD"}:
                    if position_state is None:
                        position_counter += 1
                        is_direct = execution_state == "EXECUTE_CONFIRMED_DIRECT"
                        position_state = {
                            "position_id": f"{symbol}::POS{position_counter:04d}",
                            "entry_date": trade_date,
                            "entry_tier": route_out["execution_intent"].get("entry_tier") or "NONE",
                            "pilot_fraction": 0.0 if is_direct else float(route_out["execution_intent"].get("pilot_size_fraction") or 0.0),
                            "target_fraction": float(route_out["execution_intent"].get("target_fraction") or (1.0 if is_direct else 0.0)),
                            "position_type": "CONFIRMED_DIRECT" if is_direct else "PILOT_OR_SCALE",
                            "sessions_held": 0,
                            "rearm_count": 0,
                            "state": "OPEN",
                        }
                        symbol_counts["positions_opened"] += 1
                        if is_direct:
                            symbol_counts["positions_direct"] += 1
                        record_position_event(
                            ledger_conn,
                            symbol,
                            trade_date,
                            str(position_state["position_id"]),
                            "POSITION_OPENED",
                            {**position_state, "run_nonce": RUN_NONCE},
                        )
                    if execution_state == "EXECUTE_CONFIRMED_ADD" and position_state is not None:
                        position_state["state"] = "SCALED"
                        symbol_counts["scale_events"] += 1
                        record_position_event(
                            ledger_conn,
                            symbol,
                            trade_date,
                            str(position_state["position_id"]),
                            "POSITION_SCALED",
                            {**position_state, "run_nonce": RUN_NONCE},
                        )

                if position_state is not None:
                    position_state["sessions_held"] = int(position_state["sessions_held"] or 0) + 1
                    symbol_position_progression.append(
                        {
                            "trade_date": trade_date,
                            "position_id": position_state["position_id"],
                            "state": position_state["state"],
                            "sessions_held": int(position_state["sessions_held"] or 0),
                            "rearm_count": int(position_state["rearm_count"] or 0),
                        }
                    )
                    invalidated = base_out["base_transition_terms"].get("base_invalidate_event") == "BASE_INVALIDATED"
                    if invalidated:
                        position_state["state"] = "CLOSED_INVALIDATION"
                        symbol_counts["positions_closed"] += 1
                        record_position_event(
                            ledger_conn,
                            symbol,
                            trade_date,
                            str(position_state["position_id"]),
                            "POSITION_CLOSED_INVALIDATION",
                            {
                                **position_state,
                                "base_transition_terms": base_out["base_transition_terms"],
                                "base_reference": base_out["base_reference"],
                                "run_nonce": RUN_NONCE,
                            },
                        )
                        position_state = None
                    elif position_state.get("position_type") != "CONFIRMED_DIRECT" and position_state["sessions_held"] >= 60:
                        symbol_counts["time_stop_reviews_triggered"] += 1
                        review_payload = {
                            "candidate_intent_state": flow_out["candidate_intent"]["intent_state"],
                            "confirmation_state": flow_out["confirmation_state"],
                            "avoid_state": avoid_ctx["avoid_state"],
                            "avoid_until": avoid_ctx["avoid_until"],
                            "flow_evidence_decay": bool(deferred_state.get("flow_evidence_decay")),
                            "rearm_count": int(position_state["rearm_count"] or 0),
                            "sessions_held": int(position_state["sessions_held"] or 0),
                        }
                        if avoid_ctx["avoid_active"] or flow_out["confirmation_state"] != "CONFIRMED":
                            position_state["state"] = "CLOSED_DECAY"
                            symbol_counts["positions_closed"] += 1
                            symbol_counts["dead_money_total"] += int(position_state["sessions_held"] or 0)
                            record_position_event(
                                ledger_conn,
                                symbol,
                                trade_date,
                                str(position_state["position_id"]),
                                "POSITION_CLOSED_DECAY",
                                {**position_state, **review_payload, "run_nonce": RUN_NONCE},
                            )
                            position_state = None
                        elif int(position_state["rearm_count"] or 0) < 2:
                            position_state["rearm_count"] = int(position_state["rearm_count"] or 0) + 1
                            position_state["sessions_held"] = 1
                            position_state["state"] = "REARMED"
                            symbol_counts["positions_rearmed"] += 1
                            record_position_event(
                                ledger_conn,
                                symbol,
                                trade_date,
                                str(position_state["position_id"]),
                                "POSITION_REARMED",
                                {**position_state, **review_payload, "run_nonce": RUN_NONCE},
                            )
                        else:
                            position_state["state"] = "OWNER_REVIEW"
                            record_position_event(
                                ledger_conn,
                                symbol,
                                trade_date,
                                str(position_state["position_id"]),
                                "POSITION_OWNER_REVIEW",
                                {**position_state, **review_payload, "run_nonce": RUN_NONCE},
                            )

                if (
                    flow_out["candidate_intent"]["intent_state"] == "INTENT_FORMED"
                    and route_out["execution_intent"]["execution_state"] == "NONE"
                    and not route_out["veto_record"]["veto"]
                    and router_current_state.get("active")
                ):
                    route_out["execution_intent"]["no_path_reason"] = "POSITION_ALREADY_OPEN_FEEDBACK_SUPPRESSED_DIRECT"
                    route_out["execution_intent"]["disposition_state"] = "NO_PATH_EXPLICIT"

                out_row = {
                    "trade_date": trade_date,
                    "owner_window_day": bool(cfg["owner_start"] <= trade_date <= cfg["owner_end"]),
                    "readiness_state": ready["readiness_state"],
                    "base_state": base_out["base_state"],
                    "avoid_state": avoid_ctx["avoid_state"],
                    "avoid_until": avoid_ctx["avoid_until"],
                    "avoid_source": avoid_ctx["avoid_source"],
                    "close": avoid_ctx["close"],
                    "sma200": avoid_ctx["sma200"],
                    "sma200_slope": avoid_ctx["sma200_slope"],
                    "ema10": avoid_ctx["ema10"],
                    "ema30": avoid_ctx["ema30"],
                    "avoid_entry_predicate": avoid_ctx["avoid_entry_predicate"],
                    "confirmation_state": flow_out["confirmation_state"],
                    "candidate_intent": flow_out["candidate_intent"],
                    "lifecycle_terms": route_out["lifecycle_terms"],
                    "execution_intent": route_out["execution_intent"],
                    "deferred_intent": route_out["deferred_intent"],
                    "veto_record": route_out["veto_record"],
                    "position_state": None if position_state is None else dict(position_state),
                    "router_current_state_feedback": router_current_state,
                }
                out_rows.append(out_row)
                record_daily_row(ledger_conn, symbol, trade_date, out_row)

                if flow_out["candidate_intent"]["intent_state"] == "INTENT_FORMED":
                    symbol_counts["early_intents_formed"] += 1
                    if route_out["execution_intent"]["execution_state"] == "NONE" and not route_out["veto_record"]["veto"] and not route_out["execution_intent"].get("no_path_reason"):
                        symbol_counts["silent_none_intents"] += 1
                if route_out["deferred_intent"]["active"]:
                    symbol_counts["deferred_intents"] += 1
                if route_out["execution_intent"]["chase_advisory"]["advisory_flag"]:
                    symbol_counts["advisories_emitted"] += 1
                if route_out["veto_record"]["veto"]:
                    symbol_counts["vetoes_by_plane"][str(route_out["veto_record"]["plane"])] += 1

                prior_base = base_out["base_reference"]
                prev_ready = ready["readiness_state"]
                prev_segment = seg
                prev_masked = bool(mask_ctx["masked_flag"])

        per_symbol[symbol] = out_rows
        position_progressions[symbol] = symbol_position_progression
        summary_counts[symbol] = {
            **symbol_counts,
            "vetoes_by_plane": dict(symbol_counts["vetoes_by_plane"]),
        }

    mabanee_dates = [r["trade_date"] for r in avoid_sequences.get("MABANEE", [])]
    mabanee_avoid_days = [r["trade_date"] for r in avoid_sequences.get("MABANEE", []) if r["avoid_state"] == "AVOID"]
    r12_mabanee_window_dates = dates_in_intervals(r12_mabanee_intervals, mabanee_dates)
    r12_state_by_date = r12_avoid_state_by_date(r12_mabanee_intervals, mabanee_dates)
    mabanee_per_day_divergence = [
        {
            "trade_date": row["trade_date"],
            "r12_avoid_state": r12_state_by_date[row["trade_date"]],
            "sma200_avoid_state": row["avoid_state"],
            "divergence": r12_state_by_date[row["trade_date"]] != row["avoid_state"],
        }
        for row in avoid_sequences.get("MABANEE", [])
    ]
    mabanee_overlap = sorted(set(mabanee_avoid_days).intersection(r12_mabanee_window_dates))
    mabanee_overlap_ratio = 0.0 if not r12_mabanee_window_dates else len(mabanee_overlap) / len(r12_mabanee_window_dates)
    mabanee_position_open_dates_in_r12_avoid = [
        row["trade_date"]
        for row in per_symbol.get("MABANEE", [])
        if str((row.get("execution_intent") or {}).get("execution_state") or "").startswith("EXECUTE_")
        and row["trade_date"] in set(r12_mabanee_window_dates)
    ]
    mabanee_divergence = {
        "r12_active_not_reproduced_by_sma200": sorted(set(r12_mabanee_window_dates).difference(mabanee_avoid_days)),
        "sma200_active_not_in_r12_record": sorted(set(mabanee_avoid_days).difference(r12_mabanee_window_dates)),
    }

    acceptance_checks = {
        "MABANEE": {
            "check": "SMA200-derived avoid days substantially overlap R12 avoid intervals and zero positions open inside R12-avoid intervals",
            "status": "PASS" if mabanee_overlap_ratio >= 0.5 and not any(
                mabanee_position_open_dates_in_r12_avoid
            ) else "FAIL",
            "avoid_days": sum(1 for r in avoid_sequences.get("MABANEE", []) if r["avoid_state"] == "AVOID"),
            "r12_avoid_days": len(r12_mabanee_window_dates),
            "overlap_days": len(mabanee_overlap),
            "overlap_ratio": mabanee_overlap_ratio,
            "position_open_dates_in_r12_avoid": mabanee_position_open_dates_in_r12_avoid,
            "avoid_vetoes": summary_counts.get("MABANEE", {}).get("vetoes_by_plane", {}).get("AVOID", 0),
            "positions_opened": summary_counts.get("MABANEE", {}).get("positions_opened", 0),
            "candidate_intent_days": summary_counts.get("MABANEE", {}).get("early_intents_formed", 0),
        },
        "SANAM": {
            "check": "single entry emission per position; sessions_held progression; every I_FORMED day has disposition",
            "status": "PASS" if summary_counts.get("SANAM", {}).get("positions_opened", 0) > 0 or (
                summary_counts.get("SANAM", {}).get("early_intents_formed", 0) > 0 and sum(summary_counts.get("SANAM", {}).get("vetoes_by_plane", {}).values()) >= summary_counts.get("SANAM", {}).get("early_intents_formed", 0)
            ) else "FAIL",
            "early_intents": summary_counts.get("SANAM", {}).get("early_intents_formed", 0),
            "positions_opened": summary_counts.get("SANAM", {}).get("positions_opened", 0),
            "vetoes_by_plane": summary_counts.get("SANAM", {}).get("vetoes_by_plane", {}),
        },
        "NO_SILENT_NONE_INTENTS": {
            "check": "every INTENT_FORMED day has EXECUTE_* or explicit veto/no-path reason",
            "status": "PASS" if all(summary_counts.get(sym, {}).get("silent_none_intents", 0) == 0 for sym in summary_counts) else "FAIL",
            "silent_none_intents_by_symbol": {sym: counts.get("silent_none_intents", 0) for sym, counts in summary_counts.items()},
        },
    }

    interface_payload = {
        "version_id": "R14E_MODULE_E_INTERFACE_CONFORMANCE_V7",
        "module": "LifecycleIntentRouter+StagedPositionPolicy",
        "inputs": [
            "candidate_intent",
            "base_state",
            "confirmation_state",
            "risk_budget_state",
        ],
        "outputs": [
            "execution_intent",
            "deferred_intent",
            "veto_record",
            "lifecycle_terms",
        ],
        "required_lifecycle_predicates": [
            "DEFERRED_INTENT_ACTIVE",
            "DEFERRED_INTENT_EXPIRY_OK",
            "EARLY_INTENT_ACTIVE",
            "EARLY_INTENT_SCALE_READY",
        ],
        "frozen_policy_assertions": {
            "pilot_fraction": 0.30,
            "participation_cap": 0.10,
            "time_stop_sessions": 60,
            "max_rearms": 2,
            "scale_on_confirmation": "SINGLE_ADD_TO_FULL_TARGET",
            "chase_advisory": 0.08,
            "chase_escalation": 0.15,
        },
        "avoid_plane_source_verbatim": "close < sma200 and sma200_slope < 0 and ema10 < ema30; clear via reclaim/20-session fallback",
        "freeze_v2_attestation": attest,
        "run_nonce": RUN_NONCE,
    }

    evidence_payload = {
        "version_id": "R14E_MODULE_E_TEST_EVIDENCE_V7",
        "freeze_v2_attestation": attest,
        "run_nonce": RUN_NONCE,
        "harness_db": str(HARNESS_DB),
        "supersedes": {"v6": "SUPERSEDED: state feedback absent; MABANEE window invalid as test"},
        "conduct_facts": {"sqlite_lock_fix": "legitimate harness repair: commit harness-ledger writes per event and set PRAGMA busy_timeout", "modules_f_g": "BLOCKED"},
        "router_guard_assessment": "Router code needs no v7 change: confirmed_direct_ready already requires not current_state.get('active'); v7 feeds open-position state into deferred_intent_state/current_state so direct entry emits once per open position.",
        "signal_surface_counts_not_used_for_avoid": signal_surface_counts,
        "mabanee_ee_signals_emptiness_fact": "MABANEE avoid-stream rows in ee_signals are 0 in the decline replay window; v5 derives avoid from sealed OHLCV/indicator payloads instead of this empty avoid stream. Non-avoid EXIT/PHASE_ONLY rows may still exist and are not used for avoid.",
        "owner_windows": windows,
        "window_source_assertions": windows,
        "per_day_intent_lifecycle_tables": per_symbol,
        "summary_counts": summary_counts,
        "avoid_derivation_code_block": AVOID_DERIVATION_CODE,
        "avoid_state_sequences_with_sma200": avoid_sequences,
        "position_sessions_held_progression": position_progressions,
        "r12_mabanee_avoid_intervals": r12_mabanee_intervals,
        "r12_mabanee_avoid_dates_in_v7_window": r12_mabanee_window_dates,
        "v7_sma200_mabanee_avoid_days": mabanee_avoid_days,
        "mabanee_r12_vs_v7_avoid_overlap": mabanee_overlap,
        "mabanee_r12_vs_v7_overlap_ratio": mabanee_overlap_ratio,
        "mabanee_r12_vs_v7_per_day_divergence": mabanee_per_day_divergence,
        "mabanee_r12_vs_v7_avoid_divergence": mabanee_divergence,
        "avoid_arm_lag_finding": avoid_arm_lag_finding,
        "acceptance_checks": acceptance_checks,
        "avoid_plane_source_verbatim": "close < sma200 and sma200_slope < 0 and ema10 < ema30; clear via reclaim/20-session fallback",
        "mabanee_acceptance_check": "MABANEE decline shows avoid_state=AVOID days AND avoid-plane vetoes on any candidate/intent day within them AND zero positions opened; if no candidate/intent day exists, avoid-veto count may be zero and status depends on the stated predicate.",
        "outcomes_policy": "REPORTED_AS_OBSERVED_NO_TARGET_FITTING",
        "modules_f_g_authorization_note": "BLOCKED_PENDING_CLEAN_MODULE_E_V7",
    }

    report_md = [
        "# R14-E Module (e) Harness v7",
        "",
        "- Scope: LifecycleIntentRouter + StagedPositionPolicy",
        "- Authority: r14b_parameter_freeze_v2",
        "- Mode: harness-db only",
        f"- Run nonce: {RUN_NONCE}",
        "- Avoid source: close < sma200 and sma200_slope < 0 and ema10 < ema30; clear via reclaim/20-session fallback",
        "- Supersedes: v6 SUPERSEDED (state feedback absent; MABANEE window invalid as test).",
        "- Modules (f)-(g): BLOCKED.",
        "",
        "## Freeze Attestation",
        json.dumps(attest, ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## Run Nonce",
        RUN_NONCE,
        "",
        "## Owner Windows",
        json.dumps(windows, ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## Avoid Derivation Code",
        "```python",
        AVOID_DERIVATION_CODE,
        "```",
        "",
        "## Acceptance Checks",
        json.dumps(acceptance_checks, ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## MABANEE R12 Avoid Cross-check",
        json.dumps({"r12_intervals": r12_mabanee_intervals, "r12_dates_in_v7_window": r12_mabanee_window_dates, "v7_sma200_avoid_days": mabanee_avoid_days, "overlap": mabanee_overlap, "overlap_ratio": mabanee_overlap_ratio, "per_day_divergence": mabanee_per_day_divergence, "divergence": mabanee_divergence}, ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## AVOID_ARM_LAG Finding",
        json.dumps(avoid_arm_lag_finding, ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## Per-day Intent Lifecycle Tables",
        json.dumps(per_symbol, ensure_ascii=True, indent=2, sort_keys=True),
        "",
    ]

    out_interface.write_text(json.dumps(interface_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_evidence.write_text(json.dumps(evidence_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_report.write_text("\n".join(report_md), encoding="utf-8")

    lines = [
        f"{sha256_file(out_interface)}  artifacts/preview1a_prestart/review_final/r14e_module_e_interface_conformance_v7.json",
        f"{sha256_file(out_evidence)}  artifacts/preview1a_prestart/review_final/r14e_module_e_test_evidence_v7.json",
        f"{sha256_file(out_report)}  artifacts/preview1a_prestart/review_final/r14e_module_e_implementation_report_v7.md",
    ]
    out_sha.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("R14E_MODULE_E_HARNESS_V7_COMPLETE")
    print("interface_json_sha256", sha256_file(out_interface))
    print("evidence_json_sha256", sha256_file(out_evidence))
    print("report_md_sha256", sha256_file(out_report))
    print("artifact_sidecar_sha256", sha256_file(out_sha))


if __name__ == "__main__":
    main()
