from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import r14e_module_e_lifecycle_intent_harness_v41a as v41a
import r14e_module_e_lifecycle_intent_harness_v7 as v7
import r16_3_candidate_state_machine as sm
import r16_3_harness_v53 as harness
from app.services.eagle_eye import indicator_service as ee_indicator_service
from app.services.eagle_eye_v2.adaptive_base_geometry import (
    ATR_SQUEEZE_PCTILE,
    BASE_MAX_WIDTH_PCT,
    BASE_MIN_SESSIONS,
    RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N,
    AdaptiveBaseGeometry,
    BaseNamedParameters,
)
from app.services.eagle_eye_v2.data_surface_adapter import DataSurfaceAdapter, SegmentState, load_default_calendar_context, load_default_mask_manifest
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
from app.services.eagle_eye_v2.warmup_readiness_engine import (
    READINESS_FALLBACK_MIN_SESSIONS,
    READINESS_LONG_LOOKBACK_MIN_SESSIONS,
    READINESS_SEGMENT_RESTART_MIN_SESSIONS,
    WarmupNamedParameters,
    WarmupReadinessEngine,
)

FORWARD_DB_DEFAULT = ROOT / "artifacts" / "preview1a_prestart" / "forward_surface_gate_live_full.db"
FORWARD_DB_FALLBACK = ROOT / "artifacts" / "preview1a_prestart" / "review_final" / "forward_surface_gate_live_full.db"
SEALED_DB_DEFAULT = ROOT / "artifacts" / "preview1a_prestart" / "review_final" / "r12_exam_surface_v4_5_runtime.db"
EXPORT_DIR = Path(r"C:\ee_sandbox\harness_v53\export")


def resolve_forward_db(provided: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if provided is not None:
        candidates.append(Path(provided))
    candidates.extend([FORWARD_DB_DEFAULT, FORWARD_DB_FALLBACK])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else FORWARD_DB_DEFAULT


def configure_runtime_db(runtime_db: Path | None = None) -> Path:
    target = runtime_db or (Path(tempfile.gettempdir()) / "ee_forward_replay_runtime.db")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            target.unlink()
        except PermissionError:
            # File is locked, try with a unique suffix
            import random
            target = target.parent / f"ee_forward_replay_runtime_{random.randint(10000, 99999)}.db"
    target.touch()
    os.environ["EE_V2_RUNTIME_DB_PATH"] = str(target)
    os.environ["DATABASE_PATH"] = str(target)
    get_settings.cache_clear()
    from app.services.eagle_eye_v2.predicate_telemetry_ledger import apply_schema_migration

    apply_schema_migration()
    return target


def ro_uri(path: Path) -> str:
    return "file:" + path.resolve().as_posix() + "?mode=ro"


def segment_map() -> dict[str, list[str]]:
    with sqlite3.connect(ro_uri(SEALED_DB_DEFAULT), uri=True) as conn:
        rows = conn.execute(
            "SELECT original_symbol, segment_symbol FROM ee_symbol_segment_map ORDER BY original_symbol, segment_id"
        ).fetchall()
    out: dict[str, list[str]] = {}
    for original_symbol, segment_symbol in rows:
        out.setdefault(str(original_symbol).upper(), []).append(str(segment_symbol).upper())
    return out


def canonical_by_segment() -> dict[str, str]:
    cmap: dict[str, str] = {}
    for original_symbol, segments in segment_map().items():
        for segment_symbol in segments:
            cmap[segment_symbol.upper()] = original_symbol.upper()
    return cmap


def fetch_indicator_payload(conn: sqlite3.Connection, segment_symbol: str, trade_date: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT payload_json FROM ee_indicators WHERE symbol = ? AND trade_date = ?",
        (segment_symbol, int(trade_date)),
    ).fetchone()
    if row is None or row[0] is None:
        return {}
    try:
        payload = json.loads(str(row[0]))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def load_sealed_rows(symbol: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    segments = segment_map().get(symbol.upper(), [])
    if not segments:
        return []
    placeholders = ",".join("?" for _ in segments)
    with sqlite3.connect(ro_uri(SEALED_DB_DEFAULT), uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT symbol, trade_date, open, high, low, close, volume, value_kwd
            FROM ee_ohlcv
            WHERE symbol IN ({placeholders})
              AND date(trade_date, 'unixepoch') BETWEEN ? AND ?
            ORDER BY trade_date ASC, symbol ASC
            """,
            (*segments, start_date, end_date),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            trade_ts = int(row["trade_date"])
            segment_symbol = str(row["symbol"]).upper()
            payload = fetch_indicator_payload(conn, segment_symbol, trade_ts)
            out.append(
                {
                    "symbol": symbol.upper(),
                    "segment_symbol": segment_symbol,
                    "trade_date": v7.to_date_text(trade_ts),
                    "trade_date_ts": trade_ts,
                    "open": float(row["open"] or 0.0),
                    "high": float(row["high"] or 0.0),
                    "low": float(row["low"] or 0.0),
                    "close": float(row["close"] or 0.0),
                    "volume": float(row["volume"] or 0.0),
                    "value_kwd": float(row["value_kwd"] or 0.0),
                    "indicator_payload": payload,
                }
            )
    return out


def recompute_indicator_payloads_for_forward_rows(symbol: str, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    original = ee_indicator_service.load_symbol_ohlcv
    payloads_by_trade_date: dict[str, dict[str, Any]] = {}
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_rows.append(
            {
                "symbol": str(row.get("symbol") or symbol).upper(),
                "trade_date": int(row.get("trade_date_ts") or row.get("trade_date_epoch") or 0),
                "open": float(row.get("open") or 0.0),
                "high": float(row.get("high") or 0.0),
                "low": float(row.get("low") or 0.0),
                "close": float(row.get("close") or 0.0),
                "volume": float(row.get("volume") or 0.0),
                "value_kwd": float(row.get("value_kwd") or 0.0),
            }
        )
    ee_indicator_service.load_symbol_ohlcv = lambda requested_symbol: [
        r for r in normalized_rows if str(r.get("symbol") or "").upper() == str(requested_symbol).upper()
    ]
    try:
        results = ee_indicator_service.compute_symbol_indicators(symbol.upper())
        for result in results:
            payloads_by_trade_date[str(result.trade_date)] = result.payload
    finally:
        ee_indicator_service.load_symbol_ohlcv = original
    return payloads_by_trade_date


def load_forward_rows(symbol: str, start_date: str, end_date: str, forward_db: Path = FORWARD_DB_DEFAULT) -> list[dict[str, Any]]:
    if not forward_db.exists():
        raise FileNotFoundError(f"forward surface db not found: {forward_db}")
    canonical_by_seg = canonical_by_segment()
    query = """
        SELECT symbol, trade_date, row_json
        FROM forward_surface_rows
        WHERE run_key = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date ASC, symbol ASC
    """
    with sqlite3.connect(str(forward_db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, ("FORWARD_SURFACE", start_date, end_date)).fetchall()

    filtered: list[dict[str, Any]] = []
    for row in rows:
        segment_symbol = str(row["symbol"]).upper()
        canonical = canonical_by_seg.get(segment_symbol, segment_symbol)
        if canonical.upper() != symbol.upper():
            continue
        payload = json.loads(row["row_json"])
        trade_date = str(payload.get("session") or row["trade_date"]).strip()
        trade_ts = int(v7.to_date_text(trade_date).replace("-", "")) if False else None
        try:
            trade_ts = int(datetime_from_text(trade_date).timestamp())
        except Exception:
            trade_ts = 0
        filtered.append(
            {
                "symbol": canonical.upper(),
                "segment_symbol": segment_symbol,
                "trade_date": trade_date,
                "trade_date_ts": trade_ts,
                "open": float(payload.get("open") or 0.0),
                "high": float(payload.get("high") or 0.0),
                "low": float(payload.get("low") or 0.0),
                "close": float(payload.get("close") or 0.0),
                "volume": float(payload.get("volume") or 0.0),
                "value_kwd": float(payload.get("turnover_kwd") or 0.0),
            }
        )
    if not filtered:
        return []
    indicator_map = recompute_indicator_payloads_for_forward_rows(symbol, filtered)
    for item in filtered:
        item["indicator_payload"] = indicator_map.get(str(item["trade_date_ts"]), {})
    return filtered


def datetime_from_text(value: str):
    return datetime.strptime(str(value), "%Y-%m-%d").replace(tzinfo=timezone.utc)


def segment_first_available_session(symbol: str) -> str:
    segments = segment_map().get(symbol.upper(), [])
    if not segments:
        return "1970-01-01"
    placeholders = ",".join("?" for _ in segments)
    with sqlite3.connect(ro_uri(SEALED_DB_DEFAULT), uri=True) as conn:
        row = conn.execute(
            f"SELECT MIN(trade_date) FROM ee_ohlcv WHERE symbol IN ({placeholders})",
            tuple(segments),
        ).fetchone()
    if row is None or row[0] is None:
        return "1970-01-01"
    return v7.to_date_text(int(row[0]))


def symbol_replay_window(symbol: str, start_date: str | None = None, end_date: str | None = None) -> tuple[str, str]:
    cfg = harness.owner_windows().get(symbol.upper(), {})
    if not cfg:
        cfg = v41a.owner_windows().get(symbol.upper(), {})
    load_from = str(cfg.get("replay_start") or cfg.get("owner_start") or segment_first_available_session(symbol) or "1970-01-01")
    effective_end = str(end_date or cfg.get("replay_end") or cfg.get("owner_end") or "2026-07-09")
    return load_from, effective_end


def continuous_symbol_rows(symbol: str, start_date: str, end_date: str, forward_db: Path = FORWARD_DB_DEFAULT) -> list[dict[str, Any]]:
    load_from, effective_end = symbol_replay_window(symbol, start_date, end_date)
    sealed_rows = load_sealed_rows(symbol, load_from, effective_end)
    if effective_end > "2026-07-09":
        forward_rows = load_forward_rows(symbol, "2026-07-10", effective_end, forward_db)
    else:
        forward_rows = []
    rows = sealed_rows + forward_rows
    rows.sort(key=lambda r: (r.get("trade_date"), r.get("segment_symbol", "")))
    return rows


def replay_symbol(symbol: str, rows: list[dict[str, Any]], *, variant: str = "A") -> list[dict[str, Any]]:
    runtime_db = configure_runtime_db()
    print(f"RUNTIME_DB|{runtime_db}")
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

    prev_segment: SegmentState | None = None
    prev_masked = False
    prev_ready = "READINESS_PENDING"
    history_window: list[dict[str, Any]] = []
    flow_window: list[dict[str, Any]] = []
    prior_base: dict[str, Any] | None = None
    coverage_dates: list[str] = []
    segment_dates: list[str] = []
    flag_rows: deque[dict[str, Any]] = deque(maxlen=16)
    machine = sm.initial_state(variant)
    pivot = harness.PivotEngine(harness.PIVOT_CONFIRMATION_LAG_SESSIONS, harness.SIGNIFICANT_PIVOT_ATR_MULT)
    base_recovery_stamps: dict[str, int] = {}
    out: list[dict[str, Any]] = []

    for day in rows:
        trade_date = str(day["trade_date"])
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
        segment_dates = [trade_date] if seg.segment_day_index == 0 else [*segment_dates, trade_date]
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
        base_ref = dict(base_out.get("base_reference") or {})
        base_id = str(base_ref.get("base_reference_id") or "")
        if base_id and base_ref.get("base_validity_state") == "VALID" and base_id not in base_recovery_stamps:
            base_recovery_stamps[base_id] = int(machine.get("recovery_seq") or 0)
        if base_id and base_id in base_recovery_stamps:
            base_out = {**base_out, "base_reference": {**base_ref, "base_recovery_seq": base_recovery_stamps[base_id]}}
        flow_window.append(dict(day.get("indicator_payload") or {}))
        flow_window = flow_window[-40:]
        flow_out = flow.evaluate(
            normalized_day_payload=normalized,
            base_reference=base_out["base_reference"],
            flow_history_window=flow_window,
            structure_terms=v7.build_structure_terms(day, base_out["base_reference"]),
            readiness_state=ready["readiness_state"],
            phase_state=base_out["base_state"],
        )
        ind = dict(day.get("indicator_payload") or {})
        pivot_row = {"date": trade_date, "high": day["high"], "low": day["low"], "atr14": float(ind.get("atr_14") or 0.0), "obv": float(ind.get("obv") or 0.0)}
        usable_pivots = pivot.update(pivot_row, str(machine.get("lifecycle_state")))
        current_for_flag = {"close": day["close"], "high": day["high"], "low": day["low"], "atr14": float(ind.get("atr_14") or 0.0)}
        flag = harness.flag_breakout(flag_rows, current_for_flag)
        ctx = harness.build_context(day, base_out, flow_out, usable_pivots, machine, flag, variant)
        before_position = None if machine.get("position") is None else dict(machine["position"])
        machine, actions = sm.step(machine, ctx)
        for action in actions:
            if action.get("type") == "RESET_CYCLE_REFERENCES":
                pivot.reset_cycle()
        execution_actions = [a for a in actions if a.get("type") in {"OPEN_POSITION", "CLOSE_POSITION"}]
        avoid_tier = next((a.get("avoid_tier") for a in actions if a.get("type") == "DAILY_STATE"), "NONE")
        disposition_state = harness.disposition_for_day(avoid_tier, execution_actions)
        daily = {
            "date": trade_date,
            "close": ctx["close"],
            "state": machine.get("lifecycle_state"),
            "avoid_tier": avoid_tier,
            "disposition_state": disposition_state,
            "confirmation_state": ctx["confirmation_state"],
            "candidate_intent_state": ctx["candidate_intent_state"],
            "execution": execution_actions,
            "position": machine.get("position"),
            "base_state": ctx["base_state"],
            "usable_pivots": usable_pivots,
        }
        out.append(
            {
                "date": trade_date,
                "close": ctx["close"],
                "state": machine.get("lifecycle_state"),
                "tier": avoid_tier,
                "confirmation_state": ctx["confirmation_state"],
                "candidate_intent_state": ctx["candidate_intent_state"],
                "position": machine.get("position"),
                "daily": daily,
            }
        )
        prior_base = base_out["base_reference"]
        prev_ready = ready["readiness_state"]
        prev_segment = seg
        prev_masked = bool(mask_ctx["masked_flag"])
        flag_rows.append(current_for_flag)
    return out


def format_replay_line(row: dict[str, Any]) -> str:
    pos = row.get("position") or {}
    pos_state = "OPEN" if pos else "-"
    held = pos.get("sessions_held") if pos else "-"
    return f"{row['date']}|{row['close']}|{row['state']}|{row['tier']}|{row['confirmation_state']}|{row['candidate_intent_state']}|{pos_state}|{held}"


def export_line_from_daily(daily: dict[str, Any]) -> str:
    pos = daily.get("position") or {}
    executions = daily.get("execution") or []
    close_action = next((a for a in executions if a.get("type") == "CLOSE_POSITION"), None)
    open_action = next((a for a in executions if a.get("type") == "OPEN_POSITION"), None)
    exec_state = "OPEN" if open_action else ("EXIT" if close_action else "NONE")
    disp = daily.get("disposition_state") or (close_action or open_action or {}).get("entry_reason") or (close_action or {}).get("exit_reason") or "-"
    held = pos.get("sessions_held") if pos else "-"
    pos_state = "OPEN" if pos else "-"
    exit_reason = (close_action or {}).get("exit_reason") or "-"
    return f"{daily['date']}|{daily['close']}|{daily['state']}|{daily['avoid_tier']}|{daily['confirmation_state']}|{daily['candidate_intent_state']}|{exec_state}|{disp}|{pos_state}|{held}|{exit_reason}"


def parse_export_daily(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 11:
            continue
        date = parts[0]
        rows[date] = {
            "date": date,
            "close": parts[1],
            "state": parts[2],
            "tier": parts[3],
            "conf": parts[4],
            "intent": parts[5],
            "exec_state": parts[6],
            "disp": parts[7],
            "pos": parts[8],
            "held": parts[9],
            "exit_reason": parts[10] if len(parts) > 10 else "-",
        }
    return rows


def replay_projection(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        pos = row.get("position") or {}
        pos_state = "OPEN" if pos else "-"
        held = pos.get("sessions_held") if pos else "-"
        out[str(row["date"])] = {
            "date": str(row["date"]),
            "close": str(row["close"]),
            "state": str(row["state"]),
            "tier": str(row["tier"]),
            "conf": str(row["confirmation_state"]),
            "intent": str(row["candidate_intent_state"]),
            "pos": pos_state,
            "held": str(held),
        }
    return out


def compare_export_intersection(symbol: str, rows: list[dict[str, Any]], *, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
    expected_path = EXPORT_DIR / f"v53A_{symbol.lower()}_daily.txt"
    if not expected_path.exists():
        raise FileNotFoundError(f"expected export file not found: {expected_path}")

    expected_map = parse_export_daily(expected_path)
    actual_map = replay_projection(rows)
    if start_date is not None and end_date is not None:
        filtered_expected = {d: v for d, v in expected_map.items() if start_date <= d <= end_date}
        filtered_actual = {d: v for d, v in actual_map.items() if start_date <= d <= end_date}
    else:
        filtered_expected = expected_map
        filtered_actual = actual_map

    intersection = sorted(set(filtered_expected) & set(filtered_actual))
    compared = []
    differing = []
    for date in intersection:
        expected = filtered_expected[date]
        actual = filtered_actual[date]
        expected_key = {
            "state": expected["state"],
            "tier": expected["tier"],
            "conf": expected["conf"],
            "intent": expected["intent"],
            "pos": expected["pos"],
            "held": expected["held"],
        }
        actual_key = {
            "state": actual["state"],
            "tier": actual["tier"],
            "conf": actual["conf"],
            "intent": actual["intent"],
            "pos": actual["pos"],
            "held": actual["held"],
        }
        match = expected_key == actual_key
        compared.append({"date": date, "expected": expected_key, "actual": actual_key, "match": match})
        if not match:
            differing.append({"date": date, "expected": expected, "actual": actual})
        print(f"DATE_COMPARE|date={date}|expected={expected_key}|actual={actual_key}|{'MATCH' if match else 'DIFFER'}")

    summary = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "compared_n": len(compared),
        "matched_n": sum(1 for item in compared if item["match"]),
        "differing_n": len(differing),
        "first_5_differing": differing[:5],
    }
    print(f"SUMMARY|symbol={symbol}|compared_n={summary['compared_n']}|matched_n={summary['matched_n']}|differing_n={summary['differing_n']}")
    if differing:
        print("FIRST_5_DIFFERING|" + json.dumps(differing[:5], ensure_ascii=True, sort_keys=True))
    else:
        print("FIRST_5_DIFFERING|none")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the ratified sealed and forward surface through the frozen state machine.")
    parser.add_argument("--symbols", default="SANAM,TIJARA,MABANEE")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--print-from", default=None)
    parser.add_argument("--end", default="2026-07-09")
    parser.add_argument("--sealed-db", default=str(SEALED_DB_DEFAULT))
    parser.add_argument("--forward-db", default=str(FORWARD_DB_DEFAULT))
    parser.add_argument("--validate-export", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    forward_db = resolve_forward_db(args.forward_db)
    if not forward_db.exists():
        raise FileNotFoundError(f"forward DB not found: {forward_db}")

    print("FORWARD_REPLAY_POLICY|source=sealed ee_indicators for sealed rows|forward_indicator_source=compute_symbol_indicators from app.services.eagle_eye.indicator_service|reason=forward_surface_rows are raw OHLCV only and must be fed through the same indicator engine; no alternate source is allowed")
    print(f"INPUT|symbols={','.join(symbols)}|start={args.start}|end={args.end}|sealed_db={args.sealed_db}|forward_db={forward_db}")

    for symbol in symbols:
        load_from, effective_end = symbol_replay_window(symbol, args.start, args.end)
        print_from = args.print_from or args.start or load_from
        rows = continuous_symbol_rows(symbol, load_from, effective_end, forward_db)
        print(
            "LOAD_SUMMARY|symbol={}|load_from={}|print_from={}|effective_end={}"
            "|sealed_rows={}|forward_rows={}|join_date=2026-07-10|total_rows={}"
            "|policy=MIN(ee_ohlcv.trade_date) for segment_symbol; warmup replay uses all actual sessions before print boundary".format(
                symbol,
                load_from,
                print_from,
                effective_end,
                sum(1 for r in rows if r.get("segment_symbol") and r.get("trade_date") <= "2026-07-09"),
                sum(1 for r in rows if r.get("trade_date") > "2026-07-09"),
                len(rows),
            )
        )
        replay_rows = replay_symbol(symbol, rows)
        print(f"TABLE|symbol={symbol}")
        for row in replay_rows:
            if row["date"] >= print_from and row["date"] <= effective_end:
                print(format_replay_line(row))
        if args.validate_export:
            compare_export_intersection(symbol, replay_rows, start_date=args.start, end_date=effective_end)
            full_window_compare = compare_export_intersection(symbol, replay_symbol(symbol, continuous_symbol_rows(symbol, load_from, effective_end, forward_db)), start_date=args.start, end_date=effective_end)
            print(f"FULL_WINDOW_SUMMARY|symbol={symbol}|compared_n={full_window_compare['compared_n']}|matched_n={full_window_compare['matched_n']}|differing_n={full_window_compare['differing_n']}")


if __name__ == "__main__":
    main()
