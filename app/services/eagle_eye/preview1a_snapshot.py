from __future__ import annotations

import json
import sqlite3
from typing import Any


def _load_json(raw: Any) -> Any:
    try:
        return json.loads(str(raw or "null"))
    except Exception:
        return None


def build_symbol_snapshot(conn: sqlite3.Connection, symbol: str, trade_date: int) -> dict[str, Any]:
    row_i = conn.execute(
        "SELECT payload_json FROM ee_indicators WHERE symbol = ? AND trade_date = ?",
        (symbol, trade_date),
    ).fetchone()
    row_r = conn.execute(
        "SELECT score, band, components_json FROM ee_ratings WHERE symbol = ? AND trade_date = ?",
        (symbol, trade_date),
    ).fetchone()
    row_s = conn.execute(
        "SELECT phase, phase_since, base_high, base_low, state_json FROM ee_symbol_state WHERE symbol = ?",
        (symbol,),
    ).fetchone()
    row_sig = conn.execute(
        "SELECT signal_type, phase_from, phase_to, price, stop_price, evidence_json "
        "FROM ee_signals WHERE symbol = ? AND trade_date = ? ORDER BY id DESC LIMIT 1",
        (symbol, trade_date),
    ).fetchone()
    row_pos = conn.execute(
        "SELECT status, opened_at, closed_at, avg_entry, stop_price, trail_price, exit_reason "
        "FROM ee_positions WHERE symbol = ? ORDER BY id DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    row_src = conn.execute(
        "SELECT source_type, adjustment_status, corporate_action_version FROM ee_ohlcv WHERE symbol = ? AND trade_date = ?",
        (symbol, trade_date),
    ).fetchone()

    state_json = _load_json(row_s[4]) if row_s else {}
    lifecycle = state_json.get("phase_lifecycle_log", []) if isinstance(state_json, dict) else []

    return {
        "symbol": symbol,
        "trade_date": int(trade_date),
        "feature_vector": _load_json(row_i[0]) if row_i else None,
        "lifecycle_phase": row_s[0] if row_s else None,
        "sessions_in_phase": (int(trade_date) - int(row_s[1])) // 86400 if row_s and row_s[1] else None,
        "previous_phase": state_json.get("phase_lifecycle_last_event", {}).get("old", {}).get("phase") if isinstance(state_json, dict) else None,
        "transition_date": state_json.get("phase_lifecycle_last_event", {}).get("bar") if isinstance(state_json, dict) else None,
        "rating_total": float(row_r[0]) if row_r else None,
        "rating_components": _load_json(row_r[2]) if row_r else None,
        "base_levels": {"base_high": row_s[2] if row_s else None, "base_low": row_s[3] if row_s else None},
        "gate_outcomes": (_load_json(row_sig[5]) or {}) if row_sig else None,
        "signal": row_sig[0] if row_sig else None,
        "position_status": row_pos[0] if row_pos else None,
        "entry": row_pos[3] if row_pos else None,
        "stop": row_pos[4] if row_pos else None,
        "exit_state": row_pos[6] if row_pos else None,
        "transition_history": lifecycle,
        "user_facing_output": {
            "band": row_r[1] if row_r else None,
            "phase": row_s[0] if row_s else None,
            "signal": row_sig[0] if row_sig else None,
        },
        "source_series_type": row_src[1] if row_src else None,
        "adjustment_version_as_of": row_src[2] if row_src else None,
    }
