from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import difflib
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import r14e_module_e_lifecycle_intent_harness_v41a as v41a
import r14e_module_e_lifecycle_intent_harness_v7 as v7
import r16_2_candidate_state_machine as sm
from app.core.config import get_settings
from app.services.eagle_eye_v2.adaptive_base_geometry import ATR_SQUEEZE_PCTILE, BASE_MAX_WIDTH_PCT, BASE_MIN_SESSIONS, RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N, AdaptiveBaseGeometry, BaseNamedParameters
from app.services.eagle_eye_v2.data_surface_adapter import DataSurfaceAdapter, SegmentState, load_default_calendar_context, load_default_mask_manifest
from app.services.eagle_eye_v2.flow_confirmation_engine import ADX_TRIGGER, ANV_SLOPE_MIN, CHASE_ADVISORY_BAND, CMF_FLOOR, MIN_CURRENT_DAY_VALUE_KWD, MIN_DAILY_VALUE_KWD, OBV_SLOPE_MIN, REL_VOLUME_CONTEXT_MIN, RSI_REGIME, FlowConfirmationEngine, FlowNamedParameters
from app.services.eagle_eye_v2.warmup_readiness_engine import READINESS_FALLBACK_MIN_SESSIONS, READINESS_LONG_LOOKBACK_MIN_SESSIONS, READINESS_SEGMENT_RESTART_MIN_SESSIONS, WarmupNamedParameters, WarmupReadinessEngine
from app.services.eagle_eye_v2.predicate_telemetry_ledger import apply_schema_migration

PIVOT_CONFIRMATION_LAG_SESSIONS = 3
SIGNIFICANT_PIVOT_ATR_MULT = 1.5
CHANDELIER_ATR_MULT = 2.75
TIME_STOP_MFE_WAIVER_PCT = 0.08

RUN_KEY_PREFIX = "R16_2_HARNESS_V52"
SANDBOX = Path(r"C:\ee_sandbox\harness_v52")
EXPORT = SANDBOX / "export"
BASELINE_EXPORT = Path(r"C:\ee_sandbox\harness_v41\export\f5_universe_baseline.txt")
BASELINE_DB = Path(r"C:\ee_sandbox\harness_v41\harness_v41B_2026-07-20T150021_572797Z.db")
REQUIRED_SHA256 = v41a.REQUIRED_SHA256

BASELINE_SUM_SQL = """
WITH opened AS (
    SELECT
        e.symbol,
        e.position_id,
        json_extract(e.event_json, '$.entry_date') AS entry_date,
        CAST(json_extract(d.row_json, '$.close') AS REAL) AS entry_close
    FROM harness_position_events e
    JOIN harness_daily_rows d
        ON d.run_key = e.run_key
     AND d.symbol = e.symbol
     AND d.trade_date = json_extract(e.event_json, '$.entry_date')
    WHERE e.run_key = 'R14E_MODULE_E_HARNESS_V41B'
        AND e.event_type = 'POSITION_OPENED'
), closed AS (
    SELECT
        e.position_id,
        e.event_type,
        CAST(json_extract(e.event_json, '$.exit_close') AS REAL) AS exit_close
    FROM harness_position_events e
    WHERE e.run_key = 'R14E_MODULE_E_HARNESS_V41B'
        AND e.event_type <> 'POSITION_OPENED'
), mark_close AS (
    SELECT
        symbol,
        CAST(json_extract(row_json, '$.close') AS REAL) AS mark_close
    FROM harness_daily_rows
    WHERE run_key = 'R14E_MODULE_E_HARNESS_V41B'
        AND trade_date = '2026-07-09'
), position_returns AS (
    SELECT
        o.symbol,
        o.position_id,
        o.entry_date,
        o.entry_close,
        COALESCE(c.exit_close, m.mark_close) AS valuation_close,
        CASE WHEN c.position_id IS NULL THEN 'OPEN_MARKED' ELSE c.event_type END AS valuation_type,
        ((COALESCE(c.exit_close, m.mark_close) / o.entry_close) - 1.0) * 100.0 AS pnl_pct
    FROM opened o
    LEFT JOIN closed c ON c.position_id = o.position_id
    LEFT JOIN mark_close m ON m.symbol = o.symbol
)
SELECT
    COUNT(*) AS positions_valued,
    SUM(CASE WHEN valuation_type = 'OPEN_MARKED' THEN 1 ELSE 0 END) AS open_marked_count,
    SUM(pnl_pct) AS baseline_sum_pnl_pct
FROM position_returns;
""".strip()


class PivotEngine:
    """Layer-1 pivot engine. A pivot at index i is usable only after i + k sessions."""

    def __init__(self, k: int = PIVOT_CONFIRMATION_LAG_SESSIONS, atr_mult: float = SIGNIFICANT_PIVOT_ATR_MULT) -> None:
        self.k = k
        self.atr_mult = atr_mult
        self.rows: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []
        self.significant_highs: list[dict[str, Any]] = []
        self.significant_lows: list[dict[str, Any]] = []
        self.last_opposite: dict[str, dict[str, Any]] = {}
        self.last_markup_swing_low: dict[str, Any] | None = None

    def update(self, row: dict[str, Any], lifecycle_state: str) -> dict[str, Any]:
        self.rows.append(row)
        idx = len(self.rows) - 1
        center_idx = idx - self.k
        if center_idx >= self.k:
            pivot = self._formed_pivot(center_idx)
            if pivot:
                self.pending.append(pivot)
        usable_now = [p for p in self.pending if idx >= int(p["formed_index"]) + self.k]
        self.pending = [p for p in self.pending if p not in usable_now]
        for pivot in usable_now:
            self._accept_if_significant(pivot, lifecycle_state)
        last_high = self.significant_highs[-1] if self.significant_highs else None
        prior_high = self.significant_highs[-2] if len(self.significant_highs) >= 2 else None
        last_low = self.significant_lows[-1] if self.significant_lows else None
        return {
            "last_sig_high": last_high,
            "prior_sig_high": prior_high,
            "last_sig_low": last_low,
            "last_markup_swing_low": self.last_markup_swing_low,
            "obv_at_last_high_pivot": None if last_high is None else last_high.get("obv"),
            "obv_at_prior_high_pivot": None if prior_high is None else prior_high.get("obv"),
        }

    def _formed_pivot(self, center_idx: int) -> dict[str, Any] | None:
        window = self.rows[center_idx - self.k : center_idx + self.k + 1]
        center = self.rows[center_idx]
        if float(center["high"]) == max(float(r["high"]) for r in window):
            return {"kind": "HIGH", "price": float(center["high"]), "date": center["date"], "formed_index": center_idx, "atr14": float(center.get("atr14") or 0.0), "obv": float(center.get("obv") or 0.0)}
        if float(center["low"]) == min(float(r["low"]) for r in window):
            return {"kind": "LOW", "price": float(center["low"]), "date": center["date"], "formed_index": center_idx, "atr14": float(center.get("atr14") or 0.0), "obv": float(center.get("obv") or 0.0)}
        return None

    def _accept_if_significant(self, pivot: dict[str, Any], lifecycle_state: str) -> None:
        opposite = self.last_opposite.get("LOW" if pivot["kind"] == "HIGH" else "HIGH")
        self.last_opposite[pivot["kind"]] = pivot
        if opposite is None:
            return
        significant = abs(float(pivot["price"]) - float(opposite["price"])) >= self.atr_mult * max(float(pivot.get("atr14") or 0.0), 0.0)
        if not significant:
            return
        if pivot["kind"] == "HIGH":
            self.significant_highs.append(pivot)
        else:
            self.significant_lows.append(pivot)
            if lifecycle_state == "MARKUP_ACTIVE":
                self.last_markup_swing_low = pivot

    def reset_cycle(self) -> None:
        self.last_markup_swing_low = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text_with_sidecar(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8", newline="")
    digest = sha256_file(path)
    (path.with_suffix(path.suffix + ".sha256")).write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def ro_uri(path: Path) -> str:
    return "file:" + path.resolve().as_posix() + "?mode=ro"


def compute_baseline_sum() -> dict[str, Any]:
    with sqlite3.connect(ro_uri(BASELINE_DB), uri=True) as conn:
        row = conn.execute(BASELINE_SUM_SQL).fetchone()
    return {"sql": BASELINE_SUM_SQL, "positions_valued": int(row[0]), "open_marked_count": int(row[1]), "baseline_sum_pnl_pct": float(row[2])}


def assert_sealed_input() -> dict[str, Any]:
    actual = sha256_file(v41a.SEALED_DB).lower()
    with sqlite3.connect(ro_uri(v41a.SEALED_DB), uri=True) as conn:
        canonical_count = conn.execute("SELECT COUNT(DISTINCT original_symbol) FROM ee_symbol_segment_map").fetchone()[0]
        segment_count = conn.execute("SELECT COUNT(DISTINCT segment_symbol) FROM ee_symbol_segment_map").fetchone()[0]
        absent = {symbol: conn.execute("SELECT COUNT(*) FROM ee_symbol_segment_map WHERE original_symbol=?", (symbol,)).fetchone()[0] for symbol in ("ERESCO", "ONON")}
    if actual != REQUIRED_SHA256:
        raise RuntimeError(f"sealed input SHA mismatch: {actual} != {REQUIRED_SHA256}")
    if int(canonical_count) != 139 or int(segment_count) != 309 or any(absent.values()):
        raise RuntimeError(f"universe assertion failed canonical={canonical_count} segment={segment_count} absent={absent}")
    return {"actual_sha256": actual, "required_sha256": REQUIRED_SHA256, "canonical_count": int(canonical_count), "segment_count": int(segment_count), "absent_assertions": absent}


def owner_windows() -> dict[str, dict[str, str]]:
    windows: dict[str, dict[str, str]] = {}
    for symbol, segments in v41a.SEGMENT_MAP.items():
        starts = [int(row["start_trade_date"]) for row in segments]
        ends = [int(row["end_trade_date"]) for row in segments]
        start = v7.to_date_text(min(starts))
        end = v7.to_date_text(max(ends))
        windows[symbol] = {"replay_start": start, "replay_end": end, "owner_start": start, "owner_end": end}
    return dict(sorted(windows.items()))


def configure_runtime_db(harness_db: Path) -> None:
    if harness_db.exists():
        harness_db.unlink()
    harness_db.parent.mkdir(parents=True, exist_ok=True)
    harness_db.touch()
    os.environ["EE_V2_RUNTIME_DB_PATH"] = str(harness_db)
    os.environ["DATABASE_PATH"] = str(harness_db)
    get_settings.cache_clear()
    v7.HARNESS_DB = harness_db
    apply_schema_migration()


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS r16_daily_rows (run_key TEXT, symbol TEXT, trade_date TEXT, row_json TEXT, PRIMARY KEY(run_key, symbol, trade_date))")
    conn.execute("CREATE TABLE IF NOT EXISTS r16_position_events (id INTEGER PRIMARY KEY AUTOINCREMENT, run_key TEXT, symbol TEXT, trade_date TEXT, position_id TEXT, event_type TEXT, event_json TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS r16_run_metadata (run_key TEXT PRIMARY KEY, run_nonce TEXT, variant TEXT, sealed_sha256 TEXT, freeze_sha256 TEXT, row_count INTEGER)")


def record_daily(conn: sqlite3.Connection, run_key: str, symbol: str, trade_date: str, row: dict[str, Any]) -> None:
    conn.execute("INSERT OR REPLACE INTO r16_daily_rows VALUES (?, ?, ?, ?)", (run_key, symbol, trade_date, json.dumps(row, ensure_ascii=True, sort_keys=True)))
    conn.commit()


def record_event(conn: sqlite3.Connection, run_key: str, symbol: str, trade_date: str, position_id: str, event_type: str, payload: dict[str, Any]) -> None:
    conn.execute("INSERT INTO r16_position_events (run_key, symbol, trade_date, position_id, event_type, event_json) VALUES (?, ?, ?, ?, ?, ?)", (run_key, symbol, trade_date, position_id, event_type, json.dumps(payload, ensure_ascii=True, sort_keys=True)))
    conn.commit()


def build_context(day: dict[str, Any], base_out: dict[str, Any], flow_out: dict[str, Any], pivots: dict[str, Any], machine_state: dict[str, Any], flag_breakout: bool, variant: str) -> dict[str, Any]:
    ind = dict(day.get("indicator_payload") or {})
    base_ref = dict(base_out.get("base_reference") or {})
    base_top = base_ref.get("base_high_ref")
    high = float(day.get("high") or 0.0)
    base_mfe = 0.0 if not base_top else max(0.0, (high / float(base_top)) - 1.0)
    ema30_slope = float(ind.get("ema30_slope") or 0.0)
    return {
        "variant": variant,
        "date": str(day["trade_date"]),
        "close": float(day.get("close") or 0.0),
        "high": high,
        "low": float(day.get("low") or 0.0),
        "base_state": base_out.get("base_state"),
        "base_valid": base_ref.get("base_validity_state") == "VALID",
        "base_top_ref": base_top,
        "base_low_ref": base_ref.get("base_low_ref"),
        "base_reference_id": base_ref.get("base_reference_id"),
        "base_recovery_seq": base_ref.get("base_recovery_seq"),
        "base_mfe": base_mfe,
        "confirmation_state": flow_out.get("confirmation_state"),
        "candidate_intent_state": (flow_out.get("candidate_intent") or {}).get("intent_state"),
        "ema10": float(ind.get("ema10") or day.get("close") or 0.0),
        "ema30": float(ind.get("ema30") or day.get("close") or 0.0),
        "ema30_slope_5s": ema30_slope,
        "atr14": float(ind.get("atr_14") or max(0.0, float(day.get("high") or 0.0) - float(day.get("low") or 0.0))),
        "obv": float(ind.get("obv") or 0.0),
        "usable_pivots": pivots,
        "flag_breakout": flag_breakout,
        "machine_lifecycle_state_before": machine_state.get("lifecycle_state"),
    }


def flag_breakout(rows: deque[dict[str, Any]], current: dict[str, Any]) -> bool:
    if len(rows) < 5:
        return False
    window = list(rows)[-15:]
    for n in range(5, min(15, len(window)) + 1):
        chunk = window[-n:]
        high = max(float(r["high"]) for r in chunk)
        low = min(float(r["low"]) for r in chunk)
        atr = max(float(current.get("atr14") or 0.0), 0.0)
        if high - low <= 2.0 * atr and float(current["close"]) > high:
            return True
    return False


def run_variant(variant: str) -> dict[str, Any]:
    run_nonce = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    run_key = f"{RUN_KEY_PREFIX}_{variant}"
    harness_db = SANDBOX / f"harness_v52{variant}_{run_nonce.replace(':', '').replace('.', '_')}.db"
    sealed = assert_sealed_input()
    SANDBOX.mkdir(parents=True, exist_ok=True)
    shutil.copy2(v41a.SOURCE_REVIEW / "r14b_parameter_freeze_v2.json", SANDBOX / "r14b_parameter_freeze_v2.json")
    shutil.copy2(v41a.SOURCE_REVIEW / "r14b_parameter_freeze_v2.sha256", SANDBOX / "r14b_parameter_freeze_v2.sha256")
    freeze_actual = sha256_file(SANDBOX / "r14b_parameter_freeze_v2.json")
    freeze_expected = (SANDBOX / "r14b_parameter_freeze_v2.sha256").read_text(encoding="utf-8").strip().split()[0]
    configure_runtime_db(harness_db)
    cal = load_default_calendar_context(ROOT)
    mask = load_default_mask_manifest(ROOT)
    adapter = DataSurfaceAdapter(calendar_context=cal, mask_manifest=mask)
    warmup = WarmupReadinessEngine(WarmupNamedParameters(values={READINESS_LONG_LOOKBACK_MIN_SESSIONS: 180, READINESS_SEGMENT_RESTART_MIN_SESSIONS: 20, READINESS_FALLBACK_MIN_SESSIONS: 60}))
    base = AdaptiveBaseGeometry(BaseNamedParameters(values={BASE_MIN_SESSIONS: 10, BASE_MAX_WIDTH_PCT: 0.24, ATR_SQUEEZE_PCTILE: 0.95}))
    flow = FlowConfirmationEngine(FlowNamedParameters(values={OBV_SLOPE_MIN: 0.10, ANV_SLOPE_MIN: 0.10, CMF_FLOOR: 0.05, REL_VOLUME_CONTEXT_MIN: 2.5, RSI_REGIME: 50.0, ADX_TRIGGER: 18.0, CHASE_ADVISORY_BAND: 0.08, MIN_DAILY_VALUE_KWD: 100000.0, MIN_CURRENT_DAY_VALUE_KWD: 50000.0}))
    windows = owner_windows()
    row_count = 0
    with sqlite3.connect(str(harness_db)) as conn:
        conn.execute("PRAGMA busy_timeout=5000")
        ensure_tables(conn)
        conn.commit()
        for symbol, cfg in windows.items():
            rows = v41a.load_window(symbol, cfg["replay_start"], cfg["replay_end"])
            prev_segment: SegmentState | None = None
            prev_masked = False
            prev_ready = "READINESS_PENDING"
            history_window: list[dict[str, Any]] = []
            flow_window: list[dict[str, Any]] = []
            prior_base: dict[str, Any] | None = None
            coverage_dates: list[str] = []
            segment_dates: list[str] = []
            machine = sm.initial_state(variant)
            pivot = PivotEngine()
            base_recovery_stamps: dict[str, int] = {}
            flag_rows: deque[dict[str, Any]] = deque(maxlen=16)
            for day in rows:
                trade_date = str(day["trade_date"])
                mask_ctx = adapter.mask_context_for(symbol, trade_date)
                seg = adapter.next_segment_state(symbol=symbol, trade_date=trade_date, prev_segment=prev_segment, prev_masked=prev_masked, current_masked=bool(mask_ctx["masked_flag"]))
                normalized, readiness_ctx = adapter.normalize_day(ohlcv_day=day, indicator_day=dict(day.get("indicator_payload") or {}), segment_context=seg, calendar_context=cal)
                coverage_dates.append(trade_date)
                segment_dates = [trade_date] if seg.segment_day_index == 0 else [*segment_dates, trade_date]
                ready = warmup.evaluate(normalized_day_payload=normalized, coverage_history={"long_lookback_session_dates": coverage_dates, "segment_session_dates": segment_dates, "fallback_session_dates": coverage_dates, "previous_readiness_state": prev_ready}, segment_restart_flag=bool(readiness_ctx["segment_restart_flag"]))
                history_window.append(day)
                history_window = history_window[-260:]
                base_out = base.evaluate(normalized_day_payload=normalized, readiness_state=ready["readiness_state"], price_history_window=history_window, volatility_regime_state={"atr_squeeze_pctile": 0.50, "base_range_sessions": 20, "atr_value": float(day["high"] or 0.0) - float(day["low"] or 0.0), "invalidation_rule_form": RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N, "invalidation_rule_params": {"atr_mult": 1.0, "n_sessions": 2}, "parameter_status": "FROZEN_R14B_PARAMETER_FREEZE_V2"}, prior_base_reference=prior_base, flow_stub_state={"confirmed_progress": False})
                base_ref = dict(base_out.get("base_reference") or {})
                base_id = str(base_ref.get("base_reference_id") or "")
                if base_id and base_ref.get("base_validity_state") == "VALID" and base_id not in base_recovery_stamps:
                    base_recovery_stamps[base_id] = int(machine.get("recovery_seq") or 0)
                if base_id and base_id in base_recovery_stamps:
                    base_out = {**base_out, "base_reference": {**base_ref, "base_recovery_seq": base_recovery_stamps[base_id]}}
                flow_window.append(dict(day.get("indicator_payload") or {}))
                flow_window = flow_window[-40:]
                flow_out = flow.evaluate(normalized_day_payload=normalized, base_reference=base_out["base_reference"], flow_history_window=flow_window, structure_terms=v7.build_structure_terms(day, base_out["base_reference"]), readiness_state=ready["readiness_state"], phase_state=base_out["base_state"])
                ind = dict(day.get("indicator_payload") or {})
                pivot_row = {"date": trade_date, "high": day["high"], "low": day["low"], "atr14": float(ind.get("atr_14") or 0.0), "obv": float(ind.get("obv") or 0.0)}
                usable_pivots = pivot.update(pivot_row, str(machine.get("lifecycle_state")))
                current_for_flag = {"close": day["close"], "high": day["high"], "low": day["low"], "atr14": float(ind.get("atr_14") or 0.0)}
                flag = flag_breakout(flag_rows, current_for_flag)
                ctx = build_context(day, base_out, flow_out, usable_pivots, machine, flag, variant)
                before_position = None if machine.get("position") is None else dict(machine["position"])
                machine, actions = sm.step(machine, ctx)
                for action in actions:
                    if action.get("type") == "OPEN_POSITION":
                        pid = f"{symbol}::{action['position_id']}"
                        machine["position"]["position_id"] = pid
                        record_event(conn, run_key, symbol, trade_date, pid, "POSITION_OPENED", {**action, "position_id": pid, "run_nonce": run_nonce})
                    elif action.get("type") == "CLOSE_POSITION":
                        pid = str(action.get("position_id") or (before_position or {}).get("position_id") or "UNKNOWN")
                        if not pid.startswith(symbol + "::"):
                            pid = f"{symbol}::{pid}"
                        record_event(conn, run_key, symbol, trade_date, pid, str(action.get("exit_reason") or "EXIT"), {**action, "position_id": pid, "run_nonce": run_nonce})
                    elif action.get("type") == "RESET_CYCLE_REFERENCES":
                        pivot.reset_cycle()
                execution_actions = [a for a in actions if a.get("type") in {"OPEN_POSITION", "CLOSE_POSITION"}]
                avoid_tier = next((a.get("avoid_tier") for a in actions if a.get("type") == "DAILY_STATE"), "NONE")
                disposition_state = disposition_for_day(avoid_tier, execution_actions)
                daily = {"date": trade_date, "close": ctx["close"], "state": machine.get("lifecycle_state"), "avoid_tier": avoid_tier, "disposition_state": disposition_state, "confirmation_state": ctx["confirmation_state"], "candidate_intent_state": ctx["candidate_intent_state"], "execution": execution_actions, "position": machine.get("position"), "base_state": ctx["base_state"], "usable_pivots": usable_pivots}
                record_daily(conn, run_key, symbol, trade_date, daily)
                row_count += 1
                prior_base = base_out["base_reference"]
                prev_ready = ready["readiness_state"]
                prev_segment = seg
                prev_masked = bool(mask_ctx["masked_flag"])
                flag_rows.append(current_for_flag)
            conn.commit()
        conn.execute("INSERT OR REPLACE INTO r16_run_metadata VALUES (?, ?, ?, ?, ?, ?)", (run_key, run_nonce, variant, sealed["actual_sha256"], freeze_actual, row_count))
        conn.commit()
    return {"variant": variant, "run_key": run_key, "run_nonce": run_nonce, "harness_db": str(harness_db), "row_count": row_count, "sealed": sealed, "freeze_byte_match": freeze_actual == freeze_expected, "freeze_sha256": freeze_actual}


def disposition_for_day(avoid_tier: str, execution_actions: list[dict[str, Any]]) -> str:
    if execution_actions:
        action = execution_actions[0]
        return str(action.get("entry_reason") or action.get("exit_reason") or action.get("type") or "EXECUTED")
    if avoid_tier == "AVOID_SOFT":
        return "VETOED_AVOID_SOFT"
    if avoid_tier == "AVOID_HARD":
        return "VETOED_AVOID_HARD"
    return "NONE"


def table_line(row: sqlite3.Row) -> str:
    js = json.loads(row["row_json"])
    pos = js.get("position") or {}
    executions = js.get("execution") or []
    close_action = next((a for a in executions if a.get("type") == "CLOSE_POSITION"), None)
    open_action = next((a for a in executions if a.get("type") == "OPEN_POSITION"), None)
    exec_state = "OPEN" if open_action else ("EXIT" if close_action else "NONE")
    disp = js.get("disposition_state") or (close_action or open_action or {}).get("entry_reason") or (close_action or {}).get("exit_reason") or "-"
    held = pos.get("sessions_held") if pos else "-"
    pos_state = "OPEN" if pos else "-"
    exit_reason = (close_action or {}).get("exit_reason") or "-"
    return f"{js['date']}|{js['close']}|{js['state']}|{js['avoid_tier']}|{js['confirmation_state']}|{js['candidate_intent_state']}|{exec_state}|{disp}|{pos_state}|{held}|{exit_reason}"


def export_variant(result: dict[str, Any]) -> dict[str, str]:
    EXPORT.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    windows = {"sanam": ("SANAM", "2026-01-01", "2026-07-09"), "tijara": ("TIJARA", "2025-09-01", "2026-07-09"), "mabanee": ("MABANEE", "2025-11-01", "2026-07-09")}
    with sqlite3.connect(result["harness_db"]) as conn:
        conn.row_factory = sqlite3.Row
        for slug, (symbol, start, end) in windows.items():
            rows = conn.execute("SELECT row_json FROM r16_daily_rows WHERE run_key=? AND symbol=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date", (result["run_key"], symbol, start, end)).fetchall()
            lines = [table_line(r) for r in rows]
            path = EXPORT / f"v52{result['variant']}_{slug}_daily.txt"
            hashes[path.name] = write_text_with_sidecar(path, "\n".join(lines) + "\n")
        universe_path = EXPORT / f"v52{result['variant']}_universe.txt"
        lines = universe_rows(conn, result["run_key"])
        hashes[universe_path.name] = write_text_with_sidecar(universe_path, "\n".join(lines) + "\n")
    return hashes


def parse_f5_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("F5_ROW|"):
            continue
        obj = json.loads(line.split("|", 1)[1])
        rows[str(obj["symbol"])] = obj
    return rows


def export_vs_baseline(results: list[dict[str, Any]], baseline_sum: dict[str, Any]) -> str:
    baseline = parse_f5_rows(BASELINE_EXPORT)
    lines = ["R16_2_V52_VS_BASELINE"]
    for result in results:
        variant_path = EXPORT / f"v52{result['variant']}_universe.txt"
        candidate = parse_f5_rows(variant_path)
        base_global = baseline.get("GLOBAL") or baseline.get("GLOBAL_TOTALS") or {}
        cand_global = candidate.get("GLOBAL", {})
        comparison = {
            "variant": result["variant"],
            "baseline_symbol": base_global.get("symbol"),
            "baseline_positions_opened": base_global.get("positions_opened"),
            "candidate_positions_opened": cand_global.get("positions_opened"),
            "baseline_timestop": base_global.get("exited_timestop"),
            "candidate_timestop_stagnant": cand_global.get("timestop_stagnant"),
            "baseline_invalidation": base_global.get("exited_invalidation"),
            "candidate_structural_chandelier": cand_global.get("structural_chandelier"),
            "candidate_avoid_hard": cand_global.get("avoid_hard"),
            "baseline_open_sessions": base_global.get("total_open_sessions"),
            "candidate_open_sessions": cand_global.get("total_open_sessions"),
            "baseline_sum_pnl_pct": baseline_sum["baseline_sum_pnl_pct"],
            "baseline_positions_valued_for_sum": baseline_sum["positions_valued"],
            "baseline_open_marked_count_for_sum": baseline_sum["open_marked_count"],
            "candidate_sum_pnl_pct": cand_global.get("sum_pnl_pct"),
        }
        lines.append("COMPARE|" + json.dumps(comparison, sort_keys=True))
    return write_text_with_sidecar(EXPORT / "v52_vs_baseline.txt", "\n".join(lines) + "\n")


def event_chain(conn: sqlite3.Connection, run_key: str, symbol: str) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT trade_date, event_type, event_json FROM r16_position_events WHERE run_key=? AND symbol=? ORDER BY trade_date, id", (run_key, symbol)).fetchall()
    return [{"date": r[0], "event_type": r[1], **json.loads(r[2])} for r in rows]


def avoid_tier_changes(conn: sqlite3.Connection, run_key: str, symbol: str) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT row_json FROM r16_daily_rows WHERE run_key=? AND symbol=? ORDER BY trade_date", (run_key, symbol)).fetchall()
    out: list[dict[str, Any]] = []
    prev = None
    for (row_json,) in rows:
        row = json.loads(row_json)
        tier = row.get("avoid_tier")
        if tier != prev:
            out.append({"date": row.get("date"), "close": row.get("close"), "avoid_tier": tier})
            prev = tier
    return out


def export_audit_report(results: list[dict[str, Any]], checks: dict[str, Any]) -> str:
    lines = ["R16_2_V52_D4_EVIDENCE"]
    registered_findings = {
        "R1": "SOFT_EVIDENCE_WIPE — reset_cycle cleared significant_highs/lows; S1/S3 blinded post-recovery; G1=null. Spec defect (owner-side), A4.1(c) pivot chain restarts overbroad.",
        "R2": "STALE_LIFECYCLE_PROMOTION — _sync_base_lifecycle promotes NEUTRAL→MARKUP_ACTIVE from base-retirement records predating the last MARKDOWN recovery; MABANEE re-entered a markdown via M1 (04-29 @ 1006, 40 exposure days to 963).",
        "R3": "RATIFIED-LAYER_DIVERGENCE — TIJARA/SANAM v5.1 Layer-1 divergence was isolated to the added UPWARD_RETIREMENT_MFE_THRESHOLD key; v5.2 drops it to byte-match v7 invocation semantics.",
    }
    lines.append("REGISTERED_FINDINGS|" + json.dumps(registered_findings, sort_keys=True))
    for result in results:
        with sqlite3.connect(result["harness_db"]) as conn:
            for symbol in ("MABANEE", "SANAM", "TIJARA"):
                if symbol == "MABANEE":
                    payload = {"variant": result["variant"], "symbol": symbol, "avoid_tier_changes": avoid_tier_changes(conn, result["run_key"], symbol)}
                else:
                    chain = event_chain(conn, result["run_key"], symbol)
                    payload = {"variant": result["variant"], "symbol": symbol, "position_event_chain": chain}
                    if symbol == "TIJARA":
                        may_events = [e for e in chain if str(e.get("date", ""))[:7] == "2026-05"]
                        payload["may_shakeout_verdict"] = "EVENTS_PRESENT" if may_events else "NO_POSITION_EVENT_IN_MAY_2026"
                lines.append("D4|" + json.dumps(payload, sort_keys=True))
    lines.append("E2_SELF_AUDIT|" + json.dumps({"candidate_module_pure": True, "side_effects_confined_to_harness": True, "new_files_only": ["scripts/r16_2_candidate_state_machine.py", "scripts/r16_2_harness_v52.py"], "H_avoid_reference_survives_cycle_boundary": "NO — _update_markdown_recovery increments recovery_seq, calls _expire_cycle_references, and emits RESET_CYCLE_REFERENCES; harness PivotEngine.reset_cycle clears only last_markup_swing_low while preserving soft evidence.", "I_intent_none_when_ratified_formed_intent": "NO_BY_RECORDING_DESIGN — build_context records FlowConfirmationEngine candidate_intent_state before tier disposition; daily rows store candidate_intent_state regardless of VETOED_AVOID_SOFT/HARD disposition.", "J_adoption_promotion_seq_distinction": "BASE_VALID adoption is ratified-layer-authoritative and restamps originating_base_recovery_seq to current recovery_seq; BASE_RETIRED promotion remains gated by originating_base_recovery_seq == recovery_seq. Fixture K evidence attached.", "K_fixture_evidence": {"K_adoption_restamps_then_promotes": bool(checks.get("K_adoption_restamps_then_promotes")), "K_retired_without_current_adoption_neutral": bool(checks.get("K_retired_without_current_adoption_neutral"))}}, sort_keys=True))
    return write_text_with_sidecar(EXPORT / "v52_audit_report.txt", "\n".join(lines) + "\n")


def export_run_integrity(results: list[dict[str, Any]], checks: dict[str, Any], hashes: dict[str, dict[str, str]]) -> str:
    integrity: list[dict[str, Any]] = []
    expected_rows = 151642
    for result in results:
        integrity.extend([
            {"check": "I1_SEALED_SHA", "variant": result["variant"], "pass": result["sealed"]["actual_sha256"] == REQUIRED_SHA256},
            {"check": "I2_UNIVERSE_ASSERTIONS", "variant": result["variant"], "pass": result["sealed"]["canonical_count"] == 139 and result["sealed"]["segment_count"] == 309 and not any(result["sealed"]["absent_assertions"].values())},
            {"check": "I3_FREEZE_SHA", "variant": result["variant"], "pass": bool(result["freeze_byte_match"])},
            {"check": "I4_ROW_PARITY", "variant": result["variant"], "pass": result["row_count"] == expected_rows, "row_count": result["row_count"]},
            {"check": "I5_EXPORT_SIDECARS", "variant": result["variant"], "pass": all((EXPORT / name).with_suffix((EXPORT / name).suffix + ".sha256").exists() for name in hashes[result["variant"]])},
            {"check": "I6_UNIT_CHECKS", "variant": result["variant"], "pass": all(checks.values())},
            {"check": "I7_R16_CONSTANTS", "variant": result["variant"], "pass": PIVOT_CONFIRMATION_LAG_SESSIONS == 3 and SIGNIFICANT_PIVOT_ATR_MULT == 1.5 and CHANDELIER_ATR_MULT == 2.75 and TIME_STOP_MFE_WAIVER_PCT == 0.08},
        ])
    return write_text_with_sidecar(EXPORT / "v52_run_integrity.txt", "\n".join("RUN_INTEGRITY|" + json.dumps(g, sort_keys=True) for g in integrity) + "\n")


def load_daily(conn: sqlite3.Connection, run_key: str, symbol: str, start: str | None = None, end: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT row_json FROM r16_daily_rows WHERE run_key=? AND symbol=?"
    params: list[Any] = [run_key, symbol]
    if start is not None:
        sql += " AND trade_date >= ?"
        params.append(start)
    if end is not None:
        sql += " AND trade_date <= ?"
        params.append(end)
    sql += " ORDER BY trade_date"
    return [json.loads(r[0]) for r in conn.execute(sql, params).fetchall()]


def load_events(conn: sqlite3.Connection, run_key: str, symbol: str, start: str | None = None, end: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT trade_date, event_type, event_json FROM r16_position_events WHERE run_key=? AND symbol=?"
    params: list[Any] = [run_key, symbol]
    if start is not None:
        sql += " AND trade_date >= ?"
        params.append(start)
    if end is not None:
        sql += " AND trade_date <= ?"
        params.append(end)
    sql += " ORDER BY trade_date, id"
    return [{"date": r[0], "event_type": r[1], **json.loads(r[2])} for r in conn.execute(sql, params).fetchall()]


def position_capture(events: list[dict[str, Any]], daily: list[dict[str, Any]], symbol: str) -> float | None:
    if not events:
        return None
    closes = {row["date"]: float(row["close"] or 0.0) for row in daily}
    total = 0.0
    open_positions: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event_type") == "POSITION_OPENED":
            open_positions[str(event["position_id"])] = event
        elif str(event.get("event_type", "")).startswith("EXIT"):
            total += float(event.get("pnl_pct") or 0.0)
            open_positions.pop(str(event.get("position_id")), None)
    mark_close = closes.get("2026-07-09")
    if mark_close is not None:
        for event in open_positions.values():
            entry_close = float(event.get("entry_close") or 0.0)
            if entry_close > 0.0:
                total += ((mark_close / entry_close) - 1.0) * 100.0
    return total


def tijara_may_verdict(events_2026: list[dict[str, Any]]) -> str:
    may_exits = [e for e in events_2026 if str(e.get("date", ""))[:7] == "2026-05" and str(e.get("event_type", "")).startswith("EXIT")]
    if not may_exits:
        return "SURVIVED"
    first_exit = may_exits[0]
    exit_date = str(first_exit["date"])
    reentries = [e for e in events_2026 if e.get("event_type") == "POSITION_OPENED" and str(e.get("date")) > exit_date]
    if not reentries:
        return "EXITED_NO_REENTRY"
    from datetime import date
    delta = date.fromisoformat(str(reentries[0]["date"])) - date.fromisoformat(exit_date)
    return f"EXITED+REENTERED_WITHIN_{delta.days}"


def acceptance_gates_for_variant(result: dict[str, Any], baseline_sum: dict[str, Any], grep_output: str) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    with sqlite3.connect(result["harness_db"]) as conn:
        mabanee = load_daily(conn, result["run_key"], "MABANEE", "2025-12-01", "2026-07-09")
        first_soft = next((r["date"] for r in mabanee if r.get("avoid_tier") == "AVOID_SOFT"), None)
        exposure = [{"date": r["date"], "close": r["close"]} for r in mabanee if r.get("position")]
        exposure_below_1000 = [r for r in exposure if float(r["close"] or 0.0) < 1000.0]
        sanam_daily = load_daily(conn, result["run_key"], "SANAM", "2026-01-01", "2026-07-09")
        sanam_events = load_events(conn, result["run_key"], "SANAM", "2026-01-01", "2026-07-09")
        sanam_first = next((e for e in sanam_events if e.get("event_type") == "POSITION_OPENED"), None)
        sanam_capture = position_capture(sanam_events, sanam_daily, "SANAM")
        tijara_daily = load_daily(conn, result["run_key"], "TIJARA", "2026-01-01", "2026-07-09")
        tijara_events_2026 = load_events(conn, result["run_key"], "TIJARA", "2026-01-01", "2026-07-09")
        tijara_capture = position_capture(tijara_events_2026, tijara_daily, "TIJARA")
        tijara_closed_2021_2025 = [e for e in load_events(conn, result["run_key"], "TIJARA", "2021-01-01", "2025-12-31") if str(e.get("event_type", "")).startswith("EXIT")]
        universe = parse_f5_rows(EXPORT / f"v52{result['variant']}_universe.txt").get("GLOBAL", {})
    total_rows = max(int(result.get("row_count") or 0), 1)
    clock_share = float(universe.get("total_open_sessions") or 0) / total_rows
    worst = float(universe.get("worst_position_pnl_pct") or 0.0)
    candidate_sum = float(universe.get("sum_pnl_pct") or 0.0)
    gates.append({"gate": "G1", "criterion": "MABANEE soft-arm date <= 2026-02-28", "measured_value": first_soft, "pass": first_soft is not None and first_soft <= "2026-02-28"})
    gates.append({"gate": "G2", "criterion": "MABANEE exposure audit 2025-12-01..2026-07-09; no exposure below 1000", "measured_value": {"open_position_days": exposure, "exposure_below_1000": exposure_below_1000, "any_exposure_below_1000": bool(exposure_below_1000)}, "pass": not exposure_below_1000})
    gates.append({"gate": "G3", "criterion": "SANAM first-entry date/price + capture >= +57.2%", "measured_value": {"first_entry_date": None if sanam_first is None else sanam_first.get("date"), "first_entry_price": None if sanam_first is None else sanam_first.get("entry_close"), "capture_pct": sanam_capture}, "pass": sanam_first is not None and sanam_first.get("date") == "2026-04-15" and abs(float(sanam_first.get("entry_close") or 0.0) - 229.0) < 1e-9 and sanam_capture is not None and sanam_capture >= 57.2})
    gates.append({"gate": "G4", "criterion": "TIJARA per-position chain + total capture >= +101.1% + May shakeout verdict", "measured_value": {"position_event_chain": tijara_events_2026, "total_capture_pct": tijara_capture, "may_shakeout_verdict": tijara_may_verdict(tijara_events_2026)}, "pass": tijara_capture is not None and tijara_capture >= 101.1 and tijara_may_verdict(tijara_events_2026) in {"SURVIVED"} | {f"EXITED+REENTERED_WITHIN_{i}" for i in range(0, 31)}})
    gates.append({"gate": "G5", "criterion": "TIJARA closed-position count 2021-2025 <= 4", "measured_value": len(tijara_closed_2021_2025), "pass": len(tijara_closed_2021_2025) <= 4})
    gates.append({"gate": "G6", "criterion": "clock-share <= 40%, worst >= -57.5%, sum >= computed baseline sum", "measured_value": {"clock_share": clock_share, "worst_position_pnl_pct": worst, "candidate_sum_pnl_pct": candidate_sum, "baseline_sum_pnl_pct": baseline_sum["baseline_sum_pnl_pct"]}, "pass": clock_share <= 0.40 and worst >= -57.5 and candidate_sum >= baseline_sum["baseline_sum_pnl_pct"]})
    gates.append({"gate": "G7", "criterion": "BR-A grep output verbatim", "measured_value": grep_output, "pass": grep_output.strip() == ""})
    for gate in gates:
        gate["variant"] = result["variant"]
    return gates


def export_acceptance_gates(results: list[dict[str, Any]], baseline_sum: dict[str, Any], grep_output: str) -> str:
    gates: list[dict[str, Any]] = []
    for result in results:
        gates.extend(acceptance_gates_for_variant(result, baseline_sum, grep_output))
    text = "\n".join(f"{g['gate']}|{g['variant']}|{g['criterion']}|{json.dumps(g['measured_value'], sort_keys=True)}|{'PASS' if g['pass'] else 'FAIL'}" for g in gates) + "\n"
    return write_text_with_sidecar(EXPORT / "v52_acceptance_gates.txt", text)


def export_diff() -> str:
    pairs = [(ROOT / "scripts" / "r16_candidate_state_machine.py", ROOT / "scripts" / "r16_2_candidate_state_machine.py"), (ROOT / "scripts" / "r16_harness_v5.py", ROOT / "scripts" / "r16_2_harness_v52.py")]
    lines: list[str] = ["R16.1_DIFFS"]
    for old, new in pairs:
        old_lines = old.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = new.read_text(encoding="utf-8").splitlines(keepends=True)
        lines.extend(difflib.unified_diff(old_lines, new_lines, fromfile=str(old.relative_to(ROOT)), tofile=str(new.relative_to(ROOT))))
    return write_text_with_sidecar(EXPORT / "r16_2_diff.txt", "".join(lines) + "\n")


def export_text_artifact(name: str, text: str) -> str:
    return write_text_with_sidecar(EXPORT / name, text if text.endswith("\n") else text + "\n")


def export_existing_artifact(path: Path) -> str:
    EXPORT.mkdir(parents=True, exist_ok=True)
    target = EXPORT / path.name
    shutil.copy2(path, target)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists():
        shutil.copy2(sidecar, EXPORT / sidecar.name)
    return sha256_file(target)


def universe_rows(conn: sqlite3.Connection, run_key: str) -> list[str]:
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM r16_daily_rows WHERE run_key=? ORDER BY symbol", (run_key,))]
    lines: list[str] = []
    global_counts = Counter()
    pnls: list[float] = []
    for symbol in symbols:
        events = conn.execute("SELECT event_type, event_json FROM r16_position_events WHERE run_key=? AND symbol=?", (run_key, symbol)).fetchall()
        daily = conn.execute("SELECT row_json FROM r16_daily_rows WHERE run_key=? AND symbol=?", (run_key, symbol)).fetchall()
        row_pnls = [float(json.loads(e[1]).get("pnl_pct") or 0.0) for e in events if e[0].startswith("EXIT")]
        open_mark = marked_open_pnls(conn, run_key, symbol)
        row_pnls.extend(open_mark)
        exits = Counter(e[0] for e in events if e[0] != "POSITION_OPENED")
        obj = {"symbol": symbol, "positions_opened": sum(1 for e in events if e[0] == "POSITION_OPENED"), "exits": dict(exits), "timestop_stagnant": exits.get("EXITED_TIMESTOP_STAGNANT", 0), "structural_chandelier": exits.get("EXIT_STRUCTURAL_EMA30_2C", 0) + exits.get("EXIT_CHANDELIER", 0), "avoid_hard": exits.get("EXIT_AVOID_HARD", 0), "invalidation": exits.get("POSITION_CLOSED_INVALIDATION", 0), "avoid_soft_days": sum(1 for r in daily if json.loads(r[0]).get("avoid_tier") == "AVOID_SOFT"), "avoid_hard_days": sum(1 for r in daily if json.loads(r[0]).get("avoid_tier") == "AVOID_HARD"), "total_open_sessions": sum(1 for r in daily if json.loads(r[0]).get("position")), "best_position_pnl_pct": max(row_pnls) if row_pnls else None, "worst_position_pnl_pct": min(row_pnls) if row_pnls else None, "sum_pnl_pct": sum(row_pnls)}
        lines.append("F5_ROW|" + json.dumps(obj, sort_keys=True))
        for k in ("positions_opened", "timestop_stagnant", "structural_chandelier", "avoid_hard", "invalidation", "avoid_soft_days", "avoid_hard_days", "total_open_sessions"):
            global_counts[k] += int(obj[k])
        pnls.extend(row_pnls)
    glob = {"symbol": "GLOBAL", **dict(global_counts), "best_position_pnl_pct": max(pnls) if pnls else None, "worst_position_pnl_pct": min(pnls) if pnls else None, "sum_pnl_pct": sum(pnls)}
    lines.append("F5_ROW|" + json.dumps(glob, sort_keys=True))
    return lines


def marked_open_pnls(conn: sqlite3.Connection, run_key: str, symbol: str) -> list[float]:
    last = conn.execute("SELECT row_json FROM r16_daily_rows WHERE run_key=? AND symbol=? ORDER BY trade_date DESC LIMIT 1", (run_key, symbol)).fetchone()
    if not last:
        return []
    js = json.loads(last[0])
    pos = js.get("position") or {}
    if not pos:
        return []
    entry = float(pos.get("entry_close") or 0.0)
    close = float(js.get("close") or 0.0)
    return [] if entry <= 0 else [((close / entry) - 1.0) * 100.0]


def run_unit_checks() -> dict[str, Any]:
    pe = PivotEngine()
    rows = [10, 11, 12, 15, 12, 11, 10]
    seen = []
    for i, px in enumerate(rows):
        seen.append(pe.update({"date": str(i), "high": px, "low": px - 1, "atr14": 1, "obv": i}, "MARKUP_ACTIVE"))
    lag_ok = all(not s.get("last_sig_high") for s in seen[:6])
    st = sm.initial_state("A")
    st["position"] = {"position_id": "POS0001", "entry_date": "1", "entry_close": 10.0, "sessions_held": 1, "max_close": 10.0, "mfe": 0.0}
    st["lifecycle_state"] = "MARKUP_ACTIVE"
    ctx = {"date": "10", "close": 8.0, "high": 8.2, "low": 7.8, "ema30": 9.0, "ema10": 8.5, "ema30_slope_5s": -1, "atr14": 1, "base_top_ref": 9.5, "base_low_ref": 7, "base_valid": False, "base_state": "BASE_RETIRED", "base_mfe": 0.3, "confirmation_state": "NOT_CONFIRMED", "candidate_intent_state": "INTENT_NONE", "usable_pivots": {"last_markup_swing_low": {"price": 8.5}}, "flag_breakout": False}
    _, actions = sm.step(st, ctx)
    hard_ok = any(a.get("exit_reason") == "EXIT_AVOID_HARD" for a in actions)
    st2 = sm.initial_state("A")
    st2["lifecycle_state"] = "MARKUP_ACTIVE"
    st2["position"] = {"position_id": "POS0001", "entry_date": "1", "entry_close": 10.0, "sessions_held": 59, "max_close": 12.0, "mfe": 0.2}
    ctx2 = {**ctx, "close": 12.0, "ema30": 10.0, "usable_pivots": {}}
    _, actions2 = sm.step(st2, ctx2)
    waive_ok = not any(a.get("exit_reason") == "EXITED_TIMESTOP_STAGNANT" for a in actions2)
    st3 = sm.initial_state("A")
    st3["lifecycle_state"] = "BASE_VALID"
    st3["position"] = {"position_id": "POS0002", "entry_date": "1", "entry_close": 10.0, "sessions_held": 59, "max_close": 10.2, "mfe": 0.02}
    ctx3 = {**ctx, "close": 10.2, "ema30": 9.0, "usable_pivots": {}, "base_state": "BASE_VALID", "base_valid": True}
    _, actions3 = sm.step(st3, ctx3)
    stagnant_ok = any(a.get("exit_reason") == "EXITED_TIMESTOP_STAGNANT" for a in actions3)
    st4 = sm.initial_state("A")
    st4.update({"lifecycle_state": "MARKDOWN", "originating_base_top": 100.0, "originating_base_low": 80.0, "originating_base_id": "dead", "hard_base_top_breach_streak": 2})
    reset_actions = []
    for i in range(5):
        st4, actions4 = sm.step(st4, {**ctx, "date": str(i), "close": 110.0, "ema30": 100.0, "usable_pivots": {}, "base_state": "NO_BASE", "base_valid": False})
        reset_actions.extend(actions4)
    recovery_reset_ok = st4["lifecycle_state"] == "NEUTRAL" and st4["originating_base_top"] is None and any(a.get("type") == "RESET_CYCLE_REFERENCES" for a in reset_actions)
    pe2 = PivotEngine()
    pe2.significant_highs = [{"price": 20.0, "obv": 10.0}, {"price": 15.0, "obv": 9.0}]
    pe2.significant_lows = [{"price": 8.0, "obv": 7.0}]
    pe2.last_opposite = {"HIGH": {"price": 15.0}, "LOW": {"price": 8.0}}
    pe2.last_markup_swing_low = {"price": 8.0}
    pe2.reset_cycle()
    persistent_pivots = {"last_sig_high": pe2.significant_highs[-1], "prior_sig_high": pe2.significant_highs[-2], "last_markup_swing_low": pe2.last_markup_swing_low}
    post_recovery_s1_ok = pe2.last_markup_swing_low is None and len(pe2.significant_highs) == 2 and sm._avoid_soft_conditions({**ctx, "close": 9.0, "ema30": 10.0, "ema30_slope_5s": -1.0, "usable_pivots": persistent_pivots})["S1_LOWER_HIGH"]
    st5 = sm.initial_state("A")
    st5.update({"lifecycle_state": "BASE_VALID", "recovery_seq": 1, "originating_base_recovery_seq": 0, "originating_base_top": 10.0})
    st5, _ = sm.step(st5, {**ctx, "base_state": "BASE_RETIRED", "base_valid": False, "base_mfe": 0.5, "base_recovery_seq": 0, "usable_pivots": {}})
    stale_seq_retirement_no_promote = st5["lifecycle_state"] != "MARKUP_ACTIVE"
    st6 = sm.initial_state("A")
    st6.update({"lifecycle_state": "NEUTRAL", "recovery_seq": 1})
    st6, _ = sm.step(st6, {**ctx, "base_state": "BASE_VALID", "base_valid": True, "base_mfe": 0.0, "base_recovery_seq": 0, "base_reference_id": "BASE-K", "usable_pivots": {}})
    k_adopted_state = st6["lifecycle_state"]
    k_adopted_seq = st6.get("originating_base_recovery_seq")
    st6, _ = sm.step(st6, {**ctx, "base_state": "BASE_RETIRED", "base_valid": False, "base_mfe": 0.5, "base_recovery_seq": 0, "base_reference_id": "BASE-K", "usable_pivots": {}})
    k_promoted_state = st6["lifecycle_state"]
    k_adoption_restamps_then_promotes = k_adopted_state == "BASE_VALID" and k_adopted_seq == 1 and k_promoted_state == "MARKUP_ACTIVE"
    st7 = sm.initial_state("A")
    st7.update({"lifecycle_state": "NEUTRAL", "recovery_seq": 1})
    st7, _ = sm.step(st7, {**ctx, "base_state": "BASE_RETIRED", "base_valid": False, "base_mfe": 0.5, "base_recovery_seq": 0, "base_reference_id": "BASE-K", "usable_pivots": {}})
    k_no_adoption_retired_neutral = st7["lifecycle_state"] == "NEUTRAL"
    return {"pivot_usable_lag": lag_ok, "avoid_hard_forced_exit": hard_ok, "time_stop_waiver": waive_ok, "time_stop_stagnant_exit": stagnant_ok, "markdown_recovery_expires_refs": recovery_reset_ok, "post_recovery_s1_persistent": post_recovery_s1_ok, "stale_seq_retirement_no_promote": stale_seq_retirement_no_promote, "K_adoption_restamps_then_promotes": k_adoption_restamps_then_promotes, "K_retired_without_current_adoption_neutral": k_no_adoption_retired_neutral}


def conformance_map() -> dict[str, str]:
    targets = {
        "A1": (run_variant, "Layer 1 invokes AdaptiveBaseGeometry and FlowConfirmationEngine black boxes"),
        "A2": (sm._sync_base_lifecycle, "base retirement upward MFE promotes to MARKUP_ACTIVE"),
        "A3": (PivotEngine, "lag-filtered significant pivot infrastructure"),
        "A4": (sm._avoid_soft_conditions, "soft conditions plus hard avoid helpers"),
        "A5": (sm._entry_signal, "MARKUP_ACTIVE pullback and flag entries"),
        "A6": (sm._variant_exit_actions, "variant exits and progress time-stop"),
    }
    out = {}
    for key, (obj, desc) in targets.items():
        lines = inspect.getsourcelines(obj)
        path = inspect.getsourcefile(obj) or "unknown"
        out[key] = f"{desc}: {Path(path).name} lines {lines[1]}-{lines[1] + len(lines[0]) - 1}"
    out["R5"] = "AVOID_HARD arming event forced exit recorded same session close as EXIT_AVOID_HARD: r16_2_candidate_state_machine.py _force_exit_if_open/_close_position"
    out["SessionContext additions"] = "variant, base_reference_id, flag_breakout, machine_lifecycle_state_before"
    return out


def main() -> None:
    baseline_sum = compute_baseline_sum()
    print("R2_BASELINE_SUM_SQL")
    print(baseline_sum["sql"])
    print("R2_BASELINE_SUM_RESULT")
    print(json.dumps({k: v for k, v in baseline_sum.items() if k != "sql"}, indent=2, sort_keys=True))
    print("PHASE_A_CONFORMANCE_MAP")
    conformance = conformance_map()
    print(json.dumps(conformance, indent=2, sort_keys=True))
    checks = run_unit_checks()
    print("B3_UNIT_CHECKS")
    print(json.dumps(checks, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise RuntimeError("unit sanity checks failed")
    results = [run_variant("A"), run_variant("B")]
    print("C_RUN_SUMMARIES")
    print(json.dumps(results, indent=2, sort_keys=True))
    all_hashes = {r["variant"]: export_variant(r) for r in results}
    grep = subprocess.run(["git", "grep", "-n", "-E", "SANAM|TIJARA|MABANEE", "--", "scripts/r16_2_candidate_state_machine.py"], capture_output=True, text=True)
    grep_output = grep.stdout.strip()
    all_hashes["comparison"] = {"v52_vs_baseline.txt": export_vs_baseline(results, baseline_sum)}
    all_hashes["audit"] = {"v52_audit_report.txt": export_audit_report(results, checks)}
    all_hashes["acceptance_gates"] = {"v52_acceptance_gates.txt": export_acceptance_gates(results, baseline_sum, grep_output)}
    all_hashes["run_integrity"] = {"v52_run_integrity.txt": export_run_integrity(results, checks, {r["variant"]: all_hashes[r["variant"]] for r in results})}
    all_hashes["diff"] = {"r16_2_diff.txt": export_diff()}
    all_hashes["conformance"] = {"v52_conformance_map.txt": export_text_artifact("v52_conformance_map.txt", json.dumps(conformance, indent=2, sort_keys=True))}
    all_hashes["grep"] = {"v52_br_a_grep_output.txt": export_text_artifact("v52_br_a_grep_output.txt", grep_output)}
    all_hashes["unit"] = {"v52_b3_unit_checks.txt": export_text_artifact("v52_b3_unit_checks.txt", json.dumps(checks, indent=2, sort_keys=True))}
    all_hashes["baseline"] = {"v52_baseline_sum_sql.txt": export_text_artifact("v52_baseline_sum_sql.txt", json.dumps(baseline_sum, indent=2, sort_keys=True))}
    all_hashes["r3_inv_b"] = {"r3_inv_diff.txt": export_existing_artifact(SANDBOX / "r3_inv_diff.txt"), "r3_inv_diff.txt.sha256": export_existing_artifact(SANDBOX / "r3_inv_diff.txt.sha256")}
    print("D_EXPORT_HASHES")
    print(json.dumps(all_hashes, indent=2, sort_keys=True))
    print("E1_BR_A_GREP_OUTPUT")
    print(grep_output)
    print("E1_ACCEPTANCE_GATES")
    print((EXPORT / "v52_acceptance_gates.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()