from __future__ import annotations

import json
import uuid
from typing import Any

from app.core.database import exec_sql, query_all, query_one, query_val
from app.core.security import TokenData
from app.services.eagle_eye.audit_service import create_event
from app.services.eagle_eye.entry_exit_service import close_open_position, maybe_open_or_add_position, update_trailing_stop
from app.services.eagle_eye.indicator_service import load_latest_indicator
from app.services.eagle_eye.market_data_service import CONCEPT_VERSION, get_cfg, get_config_hash
from app.services.eagle_eye.ml_service import apply_ml_gate
from app.services.eagle_eye.risk_service import can_open_new_position, liquidity_filter_at

PHASES = {
    "NEUTRAL",
    "BASE_FORMING",
    "ACCUMULATION",
    "BREAKOUT_WATCH",
    "BREAKOUT_CONFIRMED",
    "MARKUP",
    "DISTRIBUTION_WARNING",
    "EXIT",
    "AVOID",
}

ALLOWED_PHASE_TRANSITIONS = {
    "NEUTRAL": {"BASE_FORMING", "MARKUP", "AVOID", "NEUTRAL"},
    "BASE_FORMING": {"ACCUMULATION", "BREAKOUT_WATCH", "NEUTRAL", "AVOID", "BASE_FORMING"},
    "ACCUMULATION": {"BREAKOUT_WATCH", "NEUTRAL", "AVOID", "ACCUMULATION"},
    "BREAKOUT_WATCH": {"BREAKOUT_CONFIRMED", "ACCUMULATION", "NEUTRAL", "AVOID", "BREAKOUT_WATCH"},
    "BREAKOUT_CONFIRMED": {"MARKUP", "ACCUMULATION", "EXIT", "AVOID", "BREAKOUT_CONFIRMED"},
    "MARKUP": {"DISTRIBUTION_WARNING", "EXIT", "AVOID", "MARKUP"},
    "DISTRIBUTION_WARNING": {"EXIT", "MARKUP", "AVOID", "DISTRIBUTION_WARNING"},
    "EXIT": {"NEUTRAL", "AVOID", "EXIT"},
    "AVOID": {"AVOID", "NEUTRAL", "BASE_FORMING", "ACCUMULATION", "BREAKOUT_WATCH"},
}


def _json_load(raw: Any) -> dict[str, Any]:
    try:
        out = json.loads(str(raw or "{}"))
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def assert_valid_transition(current_phase: str, next_phase: str) -> None:
    allowed = ALLOWED_PHASE_TRANSITIONS.get(current_phase, set())
    if next_phase not in allowed:
        raise ValueError(f"Illegal phase transition: {current_phase} -> {next_phase}")


def _system_actor() -> TokenData:
    return TokenData(user_id=0, username="system", is_admin=True)


def _load_recent_indicators(symbol: str, trade_date: int, limit: int = 140) -> list[dict[str, Any]]:
    rows = query_all(
        """
        SELECT trade_date, payload_json
        FROM ee_indicators
        WHERE symbol = ? AND trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
        """,
        (symbol, trade_date, limit),
    )
    out: list[dict[str, Any]] = []
    for row in reversed(rows or []):
        payload = _json_load(row.get("payload_json"))
        payload["trade_date"] = int(row.get("trade_date") or 0)
        out.append(payload)
    return out


def _phase_set(state: dict[str, Any], new_phase: str, trade_date: int) -> None:
    current = str(state.get("phase") or "NEUTRAL")
    assert_valid_transition(current, new_phase)
    if current != new_phase:
        state["phase"] = new_phase
        state["phase_since"] = trade_date


def get_symbol_state(symbol: str) -> dict[str, Any] | None:
    row = query_one("SELECT * FROM ee_symbol_state WHERE symbol = ?", (symbol,))
    if not row:
        return None
    return {
        "symbol": row.get("symbol"),
        "phase": row.get("phase"),
        "phase_since": row.get("phase_since"),
        "base_high": row.get("base_high"),
        "base_low": row.get("base_low"),
        "base_start": row.get("base_start"),
        "last_score": row.get("last_score"),
        "avoid_until": row.get("avoid_until"),
        "updated_at": row.get("updated_at"),
        "state_json": _json_load(row.get("state_json")),
    }


def _upsert_state(state: dict[str, Any]) -> None:
    exec_sql(
        """
        INSERT INTO ee_symbol_state (
            symbol, phase, phase_since, base_high, base_low, base_start,
            last_score, avoid_until, updated_at, state_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            phase = excluded.phase,
            phase_since = excluded.phase_since,
            base_high = excluded.base_high,
            base_low = excluded.base_low,
            base_start = excluded.base_start,
            last_score = excluded.last_score,
            avoid_until = excluded.avoid_until,
            updated_at = excluded.updated_at,
            state_json = excluded.state_json
        """,
        (
            state["symbol"],
            state["phase"],
            state["phase_since"],
            state.get("base_high"),
            state.get("base_low"),
            state.get("base_start"),
            state.get("last_score"),
            state.get("avoid_until"),
            state["updated_at"],
            _json_dump(state.get("state_json", {})),
        ),
    )


def _resolve_warmup_context(
    history: list[dict[str, Any]],
    trade_date: int,
    coverage_start_date: int | None,
    coverage_sessions: int | None,
) -> tuple[int | None, int]:
    warmup_ready_date: int | None = None
    if coverage_start_date is not None and int(coverage_start_date) > 0:
        warmup_ready_date = int(coverage_start_date)
    else:
        for row in history:
            td = int(row.get("trade_date") or 0)
            if td <= 0 or td > trade_date:
                continue
            sma200 = float(row.get("sma200") or 0.0)
            range_low_120 = float(row.get("range_low_120") or 0.0)
            range_high_120 = float(row.get("range_high_120") or 0.0)
            if sma200 > 0 and range_low_120 > 0 and range_high_120 > 0:
                warmup_ready_date = td
                break

    if warmup_ready_date is None:
        return None, 0

    if coverage_sessions is not None:
        sessions = max(0, int(coverage_sessions))
    else:
        sessions = sum(1 for row in history if warmup_ready_date <= int(row.get("trade_date") or 0) <= trade_date)
    return warmup_ready_date, sessions


def upsert_symbol_state(state: dict[str, Any]) -> None:
    _upsert_state(state)


def _risk_level(signal_type: str) -> str:
    if signal_type in {"BREAKOUT_CONFIRMED", "EXIT"}:
        return "high"
    if signal_type == "ACCUMULATION_ALERT":
        return "medium"
    return "low"


_BREAKOUT_STATE_KEYS = {
    "confirming",
    "breakout_confirmed_at",
    "breakout_base_high",
    "breakout_entry_price",
    "ema30_armed",
    "below_ema30_streak",
    "max_close",
    "trail_price",
}


def _clear_avoid_state(state_json: dict[str, Any]) -> None:
    state_json.pop("avoid_clear_streak", None)
    state_json.pop("avoid_reclaim_streak", None)
    state_json.pop("avoid_until", None)
    state_json.pop("pre_avoid_phase", None)
    state_json.pop("pre_avoid_base_high", None)
    state_json.pop("pre_avoid_base_low", None)


def _append_phase_lifecycle_event(
    state_json: dict[str, Any],
    trade_date: int,
    action: str,
    old_phase: str | None,
    new_phase: str | None,
    reason: str,
) -> dict[str, Any]:
    event = {
        "bar": int(trade_date),
        "action": str(action),
        "old": {"phase": old_phase},
        "new": {"phase": new_phase},
        "reason": str(reason),
    }
    log = state_json.get("phase_lifecycle_log") if isinstance(state_json.get("phase_lifecycle_log"), list) else []
    log.append(event)
    state_json["phase_lifecycle_log"] = log[-200:]
    state_json["phase_lifecycle_last_event"] = event
    return event


def _append_base_lifecycle_event(
    state_json: dict[str, Any],
    trade_date: int,
    action: str,
    old_high: float | None,
    old_low: float | None,
    new_high: float | None,
    new_low: float | None,
    reason: str,
) -> dict[str, Any]:
    event = {
        "bar": int(trade_date),
        "action": str(action),
        "old": {"base_high": old_high, "base_low": old_low},
        "new": {"base_high": new_high, "base_low": new_low},
        "reason": str(reason),
    }
    log = state_json.get("base_lifecycle_log") if isinstance(state_json.get("base_lifecycle_log"), list) else []
    log.append(event)
    state_json["base_lifecycle_log"] = log[-200:]
    state_json["base_lifecycle_last_event"] = event
    return event


def _clear_base_breakout_state(
    state: dict[str, Any],
    state_json: dict[str, Any],
    trade_date: int,
    action: str,
    reason: str,
) -> dict[str, Any]:
    old_high = float(state.get("base_high") or 0.0) if state.get("base_high") is not None else None
    old_low = float(state.get("base_low") or 0.0) if state.get("base_low") is not None else None
    state["base_high"] = None
    state["base_low"] = None
    state["base_start"] = None
    for key in _BREAKOUT_STATE_KEYS:
        state_json.pop(key, None)
    state_json["base_breakdown_streak"] = 0
    state_json["base_drift_up_streak"] = 0
    return _append_base_lifecycle_event(
        state_json,
        trade_date,
        action,
        old_high,
        old_low,
        None,
        None,
        reason,
    )


def _emit_signal(
    symbol: str,
    trade_date: int,
    signal_type: str,
    phase_from: str,
    phase_to: str,
    score: float,
    price: float,
    stop_price: float | None,
    evidence: dict[str, Any],
    config_hash: str,
    trace_id: str,
) -> int:
    actor = _system_actor()
    audit = create_event(
        {
            "action": "phase_transition",
            "entity_type": "symbol",
            "entity_id": symbol,
            "change_type": "operation",
            "before_state": {"phase": phase_from},
            "after_state": {"phase": phase_to},
            "risk_level": _risk_level(signal_type),
            "trace_id": trace_id,
            "source": "scheduler",
            "metadata": {
                "signal_type": signal_type,
                "evidence": evidence,
            },
            "requires_follow_up": signal_type == "BREAKOUT_CONFIRMED",
            "concept_version": CONCEPT_VERSION,
        },
        actor,
    )

    exec_sql(
        """
        INSERT INTO ee_signals (
            created_at, symbol, trade_date, signal_type, phase_from, phase_to,
            score, price, stop_price, evidence_json, concept_version, config_hash, audit_event_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trade_date,
            symbol,
            trade_date,
            signal_type,
            phase_from,
            phase_to,
            score,
            price,
            stop_price,
            _json_dump(evidence),
            CONCEPT_VERSION,
            config_hash,
            audit.get("id"),
        ),
    )

    row = query_one("SELECT id FROM ee_signals WHERE symbol = ? AND trade_date = ? AND signal_type = ? ORDER BY id DESC LIMIT 1", (symbol, trade_date, signal_type))
    return int(row.get("id") or 0) if row else 0


def evaluate_symbol(
    symbol: str,
    trade_date: int,
    score: float,
    config: dict[str, Any],
    trace_id: str | None = None,
    indicator_payload: dict[str, Any] | None = None,
    indicator_history: list[dict[str, Any]] | None = None,
    state_override: dict[str, Any] | None = None,
    persist_state: bool = True,
    liquidity_snapshot: tuple[bool, dict[str, Any]] | None = None,
    coverage_start_date: int | None = None,
    coverage_sessions: int | None = None,
) -> dict[str, Any]:
    trace_id = trace_id or str(uuid.uuid4())
    payload = indicator_payload if indicator_payload is not None else load_latest_indicator(symbol, trade_date)
    if not payload:
        return {"symbol": symbol, "status": "no_indicator"}

    history = indicator_history if indicator_history is not None else _load_recent_indicators(symbol, trade_date, 140)
    if not history:
        return {"symbol": symbol, "status": "no_history"}

    prev = history[-2] if len(history) >= 2 else None

    state = state_override or get_symbol_state(symbol) or {
        "symbol": symbol,
        "phase": "NEUTRAL",
        "phase_since": trade_date,
        "base_high": None,
        "base_low": None,
        "base_start": None,
        "last_score": None,
        "avoid_until": None,
        "updated_at": trade_date,
        "state_json": {},
    }

    warmup_ready_date, sessions_since_warmup = _resolve_warmup_context(
        history,
        trade_date,
        coverage_start_date,
        coverage_sessions,
    )

    if warmup_ready_date is None or trade_date < warmup_ready_date:
        state_json = state.get("state_json", {})
        state["phase"] = "NEUTRAL"
        state["phase_since"] = trade_date
        state["last_score"] = score
        state["updated_at"] = trade_date
        state_json["warmup_ready_date"] = warmup_ready_date
        state_json["warmup_sessions"] = 0
        state_json["last_phase_reason"] = "warmup_pending"

        signal_id = 0
        emitted_signal_type: str | None = None
        if not bool(state_json.get("warmup_note_emitted")):
            state_json["warmup_note_emitted"] = True
            config_hash = get_config_hash(config)
            signal_id = _emit_signal(
                symbol=symbol,
                trade_date=trade_date,
                signal_type="PHASE_ONLY",
                phase_from="NEUTRAL",
                phase_to="NEUTRAL",
                score=score,
                price=float(payload.get("close") or 0.0),
                stop_price=None,
                evidence={
                    **payload,
                    "warmup": True,
                    "reason": "warmup_pending",
                    "warmup_ready_date": warmup_ready_date,
                },
                config_hash=config_hash,
                trace_id=trace_id,
            )
            emitted_signal_type = "PHASE_ONLY"

        state["state_json"] = state_json
        if persist_state:
            _upsert_state(state)
        return {
            "symbol": symbol,
            "phase": "NEUTRAL",
            "transition": None,
            "signal_id": signal_id,
            "signal_type": emitted_signal_type,
            "score": score,
            "state": state,
            "reason": "warmup_pending",
        }

    old_phase = str(state["phase"])
    close = float(payload.get("close") or 0.0)
    ema10 = float(payload.get("ema10") or 0.0)
    ema30 = float(payload.get("ema30") or 0.0)
    sma200 = float(payload.get("sma200") or 0.0)
    sma200_slope = float(payload.get("sma200_slope") or 0.0)
    rel_volume = float(payload.get("rel_volume") or 0.0)
    adx = float(payload.get("adx_19") or 0.0)
    plus_di = float(payload.get("plus_di") or 0.0)
    minus_di = float(payload.get("minus_di") or 0.0)
    rsi = float(payload.get("rsi_14") or 0.0)
    cmf = float(payload.get("cmf_10") or 0.0)
    range_high_120 = float(payload.get("range_high_120") or payload.get("range_high_60") or 0.0)
    range_low_60 = float(payload.get("range_low_60") or 0.0)
    width = float(payload.get("range_width_pct") or 9.0)
    atr_pct_pctile_raw = payload.get("atr_pct_percentile_252")
    atr_pct_pctile = float(atr_pct_pctile_raw) if atr_pct_pctile_raw is not None else None

    if liquidity_snapshot is not None:
        liquidity_ok, liquidity_meta = liquidity_snapshot
    else:
        liquidity_ok, liquidity_meta = liquidity_filter_at(
            symbol,
            trade_date,
            float(get_cfg(config, "min_daily_value_kwd")),
        )

    state_json = state.get("state_json", {})
    state_json.pop("last_phase_reason", None)
    resumed_after_avoid = False
    state_base_high = state.get("base_high")
    state_base_low = state.get("base_low")

    if state["phase"] == "AVOID" and state_base_high is not None and state_base_low is not None:
        avoid_base_high_ref = float(state_base_high or 0.0)
        avoid_base_low_ref = float(state_base_low or 0.0)

        if avoid_base_low_ref > 0 and close < (avoid_base_low_ref * 0.98):
            state_json["base_breakdown_streak"] = int(state_json.get("base_breakdown_streak") or 0) + 1
        else:
            state_json["base_breakdown_streak"] = 0
        if int(state_json.get("base_breakdown_streak") or 0) >= 2:
            state_json["last_phase_reason"] = "base_invalidated_during_avoid"
            payload["base_invalidation_reason"] = "breakdown"
            _clear_base_breakout_state(
                state,
                state_json,
                trade_date,
                "base_invalidated",
                "breakdown_2x_close_lt_base_low_98pct_during_avoid",
            )

        if state_base_high is not None:
            gap_pct_base = max(0.0, (float(payload.get("open") or close) - avoid_base_high_ref) / avoid_base_high_ref) if avoid_base_high_ref > 0 else 0.0
            qualifying_breakout = (
                close > avoid_base_high_ref
                and rel_volume >= float(get_cfg(config, "volume_breakout_mult"))
                and ema10 > ema30
                and gap_pct_base <= 0.08
                and bool(liquidity_ok)
            )
            if close > avoid_base_high_ref and not qualifying_breakout:
                state_json["base_drift_up_streak"] = int(state_json.get("base_drift_up_streak") or 0) + 1
            else:
                state_json["base_drift_up_streak"] = 0
            if int(state_json.get("base_drift_up_streak") or 0) >= int(config.get("base_drift_invalidate_sessions", 10)):
                state_json["last_phase_reason"] = "base_invalidated_during_avoid"
                payload["base_invalidation_reason"] = "structure"
                _clear_base_breakout_state(
                    state,
                    state_json,
                    trade_date,
                    "base_invalidated",
                    "structure_drift_outside_base_without_qual_breakout_during_avoid",
                )

    avoid = close < sma200 and sma200_slope < 0 and ema10 < ema30
    if avoid:
        if old_phase in {"BASE_FORMING", "ACCUMULATION"}:
            state_json["pre_avoid_phase"] = old_phase
            state_json["pre_avoid_base_high"] = state.get("base_high")
            state_json["pre_avoid_base_low"] = state.get("base_low")
        state_json["avoid_clear_streak"] = 0
        state_json["avoid_reclaim_streak"] = 0
        _phase_set(state, "AVOID", trade_date)
    elif old_phase == "AVOID":
        clear_streak = int(state_json.get("avoid_clear_streak") or 0) + 1
        state_json["avoid_clear_streak"] = clear_streak
        reclaim_close = close > sma200
        if reclaim_close:
            state_json["avoid_reclaim_streak"] = int(state_json.get("avoid_reclaim_streak") or 0) + 1
        else:
            state_json["avoid_reclaim_streak"] = 0

        clear_rule: str | None = None
        if int(state_json.get("avoid_reclaim_streak") or 0) >= int(config.get("avoid_reclaim_clear_closes", 2)):
            clear_rule = "reclaim"
        elif clear_streak >= 20:
            clear_rule = "fallback"

        if clear_rule is not None:
            prev_phase = state["phase"]
            retained_base = state.get("base_high") is not None and state.get("base_low") is not None
            pre_avoid_phase = str(state_json.get("pre_avoid_phase") or "BASE_FORMING")
            if retained_base:
                _phase_set(state, pre_avoid_phase, trade_date)
                resumed_after_avoid = True
                state_json["last_phase_reason"] = "avoid_reclaimed" if clear_rule == "reclaim" else "avoid_fallback_resume"
                _append_phase_lifecycle_event(
                    state_json,
                    trade_date,
                    "avoid_cleared_resume",
                    prev_phase,
                    state["phase"],
                    f"rule={clear_rule}",
                )
            else:
                _phase_set(state, "NEUTRAL", trade_date)
                state_json["last_phase_reason"] = "avoid_cleared_invalidated"
                _append_phase_lifecycle_event(
                    state_json,
                    trade_date,
                    "avoid_cleared_neutral",
                    prev_phase,
                    state["phase"],
                    f"rule={clear_rule}",
                )
            transition = (prev_phase, state["phase"])
            _clear_avoid_state(state_json)

    transition = None
    signal_type = None
    base_min_sessions = int(get_cfg(config, "base_min_sessions"))
    range_low_120 = float(payload.get("range_low_120") or range_low_60 or 0.0)
    low = float(payload.get("low") or close)
    high = float(payload.get("high") or close)
    atr = float(payload.get("atr_14") or 0.0)
    base_high_ref = float(state.get("base_high") or 0.0)
    base_low_ref = float(state.get("base_low") or range_low_60 or 0.0)

    if old_phase != state["phase"]:
        transition = (old_phase, state["phase"])
        signal_type = "AVOID_SET" if state["phase"] == "AVOID" else None
        old_phase = state["phase"]

    if state["phase"] != "AVOID":
        closes_60 = [float(h.get("close") or 0.0) for h in history[-60:]]
        base_high_60 = float(payload.get("range_high_60") or 0.0)
        base_low_60 = float(payload.get("range_low_60") or 0.0)
        sessions_in_range = sum(1 for c in closes_60 if base_low_60 <= c <= base_high_60)

        if (transition is None or resumed_after_avoid) and state["phase"] in {"BASE_FORMING", "ACCUMULATION", "BREAKOUT_WATCH"}:
            if base_low_ref > 0 and close < (base_low_ref * 0.98):
                state_json["base_breakdown_streak"] = int(state_json.get("base_breakdown_streak") or 0) + 1
            else:
                state_json["base_breakdown_streak"] = 0
            if int(state_json.get("base_breakdown_streak") or 0) >= 2:
                prev_phase = state["phase"]
                _phase_set(state, "NEUTRAL", trade_date)
                transition = (prev_phase, state["phase"])
                signal_type = None
                state_json["last_phase_reason"] = "base_invalidated"
                payload["base_invalidation_reason"] = "breakdown"
                _clear_base_breakout_state(
                    state,
                    state_json,
                    trade_date,
                    "base_invalidated",
                    "breakdown_2x_close_lt_base_low_98pct",
                )

        if state["phase"] in {"BASE_FORMING", "ACCUMULATION"} and base_high_ref > 0:
            gap_pct_base = max(0.0, (float(payload.get("open") or close) - base_high_ref) / base_high_ref)
            qualifying_breakout = (
                close > base_high_ref
                and rel_volume >= float(get_cfg(config, "volume_breakout_mult"))
                and ema10 > ema30
                and gap_pct_base <= 0.08
                and bool(liquidity_ok)
            )
            if close > base_high_ref and not qualifying_breakout:
                state_json["base_drift_up_streak"] = int(state_json.get("base_drift_up_streak") or 0) + 1
            else:
                state_json["base_drift_up_streak"] = 0

            if int(state_json.get("base_drift_up_streak") or 0) >= int(config.get("base_drift_invalidate_sessions", 10)):
                prev_phase = state["phase"]
                _phase_set(state, "NEUTRAL", trade_date)
                transition = (prev_phase, state["phase"])
                signal_type = None
                state_json["last_phase_reason"] = "base_invalidated"
                payload["base_invalidation_reason"] = "structure"
                _clear_base_breakout_state(
                    state,
                    state_json,
                    trade_date,
                    "base_invalidated",
                    "structure_drift_outside_base_without_qual_breakout",
                )

        if (transition is None or resumed_after_avoid) and state["phase"] in {"BASE_FORMING", "ACCUMULATION", "BREAKOUT_WATCH"} and base_high_ref > 0 and base_low_ref > 0:
            if base_high_60 > 0 and base_low_60 > 0 and base_high_60 < base_high_ref and base_low_60 > base_low_ref:
                old_high = base_high_ref
                old_low = base_low_ref
                state["base_high"] = base_high_60
                state["base_low"] = base_low_60
                base_high_ref = float(state.get("base_high") or base_high_ref)
                base_low_ref = float(state.get("base_low") or base_low_ref)
                _append_base_lifecycle_event(
                    state_json,
                    trade_date,
                    "base_ratchet",
                    old_high,
                    old_low,
                    base_high_ref,
                    base_low_ref,
                    "tightened_60_session_band",
                )

        # 2) Active-position exit logic first if already in distribution warning.
        if (transition is None or resumed_after_avoid) and state["phase"] == "DISTRIBUTION_WARNING":
            below = close < ema30
            if below:
                state_json["below_ema30_streak"] = int(state_json.get("below_ema30_streak") or 0) + 1
            else:
                state_json["below_ema30_streak"] = 0

            two_close_exit = (
                bool(state_json.get("ema30_armed"))
                and int(state_json.get("below_ema30_streak") or 0) >= 2
                and rel_volume >= 1.2
            )

            confirmed_at = int(state_json.get("breakout_confirmed_at") or 0)
            time_stop = False
            if confirmed_at:
                post = [h for h in history if int(h.get("trade_date") or 0) >= confirmed_at]
                if len(post) >= 40:
                    entry = float(state_json.get("breakout_entry_price") or close)
                    time_stop = close < (entry + atr)

            exit_now = (
                two_close_exit
                or close < float(state.get("state_json", {}).get("trail_price") or 0.0)
                or (ema10 < ema30 and bool(payload.get("distribution_divergence")))
                or (close < sma200 and sma200_slope < 0)
                or time_stop
            )
            if exit_now:
                prev_phase = state["phase"]
                _phase_set(state, "EXIT", trade_date)
                transition = (prev_phase, state["phase"])
                signal_type = "EXIT"
                state_json["last_phase_reason"] = "distribution_exit"

        # 3) Breakout confirmation with two-tier mandatory + confirmatory scoring.
        if (transition is None or resumed_after_avoid) and state["phase"] == "BREAKOUT_WATCH":
            confirming = state_json.get("confirming") if isinstance(state_json.get("confirming"), dict) else None

            adx_5_back = float(history[-5].get("adx_19") or adx) if len(history) >= 5 else adx
            macd_cross_recent = False
            if len(history) >= 6:
                for i in range(max(1, len(history) - 5), len(history)):
                    prev_row = history[i - 1]
                    row = history[i]
                    if (
                        float(prev_row.get("macd_line") or 0.0) <= float(prev_row.get("macd_signal") or 0.0)
                        and float(row.get("macd_line") or 0.0) > float(row.get("macd_signal") or 0.0)
                    ):
                        macd_cross_recent = True
                        break

            day_range = max(0.0, high - low)
            close_top40 = True if day_range == 0 else close >= (low + 0.6 * day_range)
            rsi_rising = True if prev is None else rsi > float(prev.get("rsi_14") or rsi)
            gap_pct_base = 0.0 if base_high_ref <= 0 else max(0.0, (float(payload.get("open") or close) - base_high_ref) / base_high_ref)

            mandatory = {
                "M1_close_gt_base": base_high_ref > 0 and close > base_high_ref,
                "M2_rel_volume": rel_volume >= float(get_cfg(config, "volume_breakout_mult")),
                "M3_ema10_gt_ema30": ema10 > ema30,
                "M4_chase_guard": base_high_ref > 0 and gap_pct_base <= 0.08,
                "M5_liquidity": liquidity_ok,
            }

            confirm_flags = {
                "C1_rsi": rsi >= float(get_cfg(config, "rsi_regime")),
                "C2_rsi_rising": rsi_rising,
                "C3_adx_di": adx >= float(get_cfg(config, "adx_trigger")) and plus_di > minus_di,
                "C4_adx_accel": adx > adx_5_back,
                "C5_macd": (float(payload.get("macd_hist") or 0.0) > 0) or macd_cross_recent,
                "C6_close_top40": close_top40,
            }
            c_score = sum(1 for v in confirm_flags.values() if v)

            if confirming is None and all(mandatory.values()):
                confirming = {
                    "start_trade_date": trade_date,
                    "bars": 0,
                    "scores": [],
                }

            if confirming is not None:
                bars = int(confirming.get("bars") or 0) + 1
                confirming["bars"] = bars
                scores = confirming.get("scores") if isinstance(confirming.get("scores"), list) else []
                scores.append({"trade_date": trade_date, "c_score": c_score, "flags": confirm_flags})
                confirming["scores"] = scores[-3:]
                state_json["confirming"] = confirming

                if close < (base_high_ref * 0.99):
                    state_json.pop("confirming", None)
                    state_json["last_phase_reason"] = "confirming_revert_below_base"
                else:
                    ml_ok, ml_prob = apply_ml_gate(payload, config)
                    if c_score >= 4 and ml_ok and score >= 70:
                        prev_phase = state["phase"]
                        _phase_set(state, "BREAKOUT_CONFIRMED", trade_date)
                        transition = (prev_phase, state["phase"])
                        signal_type = "BREAKOUT_CONFIRMED"
                        state_json["breakout_confirmed_at"] = trade_date
                        state_json["breakout_base_high"] = base_high_ref
                        state_json["breakout_entry_price"] = close
                        state_json["last_phase_reason"] = "confirming_success"
                        state_json.pop("confirming", None)
                        if ml_prob is not None:
                            payload["ml_prob"] = ml_prob
                    elif bars >= 3:
                        state_json.pop("confirming", None)
                        state_json["last_phase_reason"] = "confirming_expired"

            payload["confirming_mandatory"] = mandatory
            payload["confirming_c_flags"] = confirm_flags
            payload["confirming_c_score"] = c_score
            payload["confirming_base_high_ref"] = base_high_ref
            payload["confirming_base_high_source"] = "state"

        # 4) Accumulation from BASE.
        if (transition is None or resumed_after_avoid) and state["phase"] == "BASE_FORMING":
            cmf_hist = [float(h.get("cmf_10") or 0.0) for h in history[-10:]]
            cmf_hits = sum(1 for x in cmf_hist if x > float(get_cfg(config, "cmf_floor")))
            accumulation_gate = bool(payload.get("accumulation_divergence")) or (
                abs(float(payload.get("price_slope_40") or 0.0)) < 0.02
                and (
                    float(payload.get("obv_slope_40") or 0.0) > 0.10
                    or float(payload.get("anv_slope_40") or 0.0) > 0.10
                )
            )
            squeeze_ok = float(payload.get("bb_width") or 1.0) <= 0.12
            if atr_pct_pctile is not None:
                squeeze_ok = squeeze_ok or (atr_pct_pctile <= float(get_cfg(config, "atr_squeeze_pctile")))
            if (
                accumulation_gate
                and cmf_hits >= 5
                and squeeze_ok
                and (close >= ema30 or close >= 0.97 * sma200)
                and liquidity_ok
                and score >= 60
            ):
                prev_phase = state["phase"]
                _phase_set(state, "ACCUMULATION", trade_date)
                transition = (prev_phase, state["phase"])
                signal_type = "ACCUMULATION_ALERT"
                state_json["had_accumulation_phase"] = True
                state_json["last_phase_reason"] = "accumulation_gate"

        # 5) Watch trigger from ACCUMULATION with frozen base-high reference.
        if (transition is None or resumed_after_avoid) and state["phase"] in {"ACCUMULATION", "BASE_FORMING"}:
            recent_5 = history[-5:]
            rv_hits = sum(1 for h in recent_5 if float(h.get("rel_volume") or 0.0) >= 1.5)
            near_base_with_build = base_high_ref > 0 and close >= (0.97 * base_high_ref) and rv_hits >= 2
            if near_base_with_build:
                prev_phase = state["phase"]
                _phase_set(state, "BREAKOUT_WATCH", trade_date)
                transition = (prev_phase, state["phase"])
                state_json["last_phase_reason"] = "watch_trigger"

        # 6) Pass-2 ordering amendment: while NEUTRAL, evaluate trend-join before
        # base detection inside the join window.
        if (transition is None or resumed_after_avoid) and state["phase"] == "NEUTRAL":
            trend_join_window = int(get_cfg(config, "trend_join_window"))
            if coverage_sessions is not None:
                sessions_since_start = int(coverage_sessions)
            elif coverage_start_date is not None:
                sessions_since_start = int(
                    query_val(
                        "SELECT COUNT(1) FROM ee_ohlcv WHERE symbol = ? AND trade_date BETWEEN ? AND ?",
                        (symbol, int(coverage_start_date), trade_date),
                    )
                    or 0
                )
            else:
                sessions_since_start = int(
                    query_val(
                        "SELECT COUNT(1) FROM ee_ohlcv WHERE symbol = ? AND trade_date <= ?",
                        (symbol, trade_date),
                    )
                    or 0
                )

            if (
                sessions_since_start <= trend_join_window
                and close > sma200
                and sma200_slope > 0
                and ema10 > ema30
                and range_low_120 > 0
                and close >= (range_low_120 * 1.15)
            ):
                prev_phase = state["phase"]
                _phase_set(state, "MARKUP", trade_date)
                transition = (prev_phase, state["phase"])
                state_json["joined_externally"] = True
                state_json["warmup_ready_date"] = int(coverage_start_date or 0) if coverage_start_date else None
                state_json["warmup_sessions"] = int(sessions_since_start)
                state_json["max_close"] = max(float(state_json.get("max_close") or close), close)
                state_json["last_phase_reason"] = "trend_join_early_coverage"

        # 7) Base detection from NEUTRAL (after trend-join check; freeze bounds on entry).
        if transition is None and state["phase"] == "NEUTRAL":
            if (
                sma200 > 0
                and ema30 > 0
                and
                width <= float(get_cfg(config, "base_max_width_pct"))
                and sessions_in_range >= base_min_sessions
                and base_low_60 <= close <= base_high_60
            ):
                prev_phase = state["phase"]
                _phase_set(state, "BASE_FORMING", trade_date)
                transition = (prev_phase, state["phase"])
                old_high = float(state.get("base_high") or 0.0) if state.get("base_high") is not None else None
                old_low = float(state.get("base_low") or 0.0) if state.get("base_low") is not None else None
                state["base_high"] = base_high_60
                state["base_low"] = range_low_60
                state["base_start"] = trade_date
                base_high_ref = float(state.get("base_high") or base_high_ref)
                base_low_ref = float(state.get("base_low") or base_low_ref)
                state_json["last_phase_reason"] = "base_detected"
                _append_base_lifecycle_event(
                    state_json,
                    trade_date,
                    "base_freeze",
                    old_high,
                    old_low,
                    base_high_ref,
                    base_low_ref,
                    "entered_base_forming",
                )

        # 8) Exit is not terminal; re-arm to NEUTRAL after cooldown sessions.
        if state["phase"] == "EXIT":
            cooldown_sessions = int(get_cfg(config, "exit_cooldown_sessions"))
            exit_start = int(state.get("phase_since") or trade_date)
            bars_since_exit = sum(1 for h in history if exit_start <= int(h.get("trade_date") or 0) <= trade_date)
            elapsed_sessions = max(0, bars_since_exit - 1)
            if elapsed_sessions >= cooldown_sessions:
                prev_phase = state["phase"]
                _phase_set(state, "NEUTRAL", trade_date)
                transition = (prev_phase, state["phase"])
                state_json["last_phase_reason"] = "exit_cooldown_rearm"
                _clear_base_breakout_state(
                    state,
                    state_json,
                    trade_date,
                    "base_cleared",
                    "exit_rearm",
                )
                _clear_avoid_state(state_json)

        confirmed_at = int(state_json.get("breakout_confirmed_at") or 0)
        if (transition is None or resumed_after_avoid) and confirmed_at:
            confirm_ix = None
            for i, h in enumerate(history):
                if int(h.get("trade_date") or 0) == confirmed_at:
                    confirm_ix = i
                    break
            if confirm_ix is not None:
                sessions_since = len(history) - 1 - confirm_ix
                if 0 < sessions_since <= 5:
                    base_high = float(state_json.get("breakout_base_high") or state.get("base_high") or 0.0)
                    if close < (base_high * 0.97) and rel_volume >= 1.5:
                        prev_phase = state["phase"]
                        _phase_set(state, "ACCUMULATION", trade_date)
                        transition = (prev_phase, state["phase"])
                        signal_type = "BREAKOUT_FAILED"
                        close_open_position(symbol, trade_date, "breakout_failed", close)

        if (transition is None or resumed_after_avoid) and state["phase"] == "BREAKOUT_CONFIRMED":
            confirmed_at = int(state_json.get("breakout_confirmed_at") or 0)
            if confirmed_at:
                post = [h for h in history if int(h.get("trade_date") or 0) >= confirmed_at]
                if len(post) >= 10:
                    held = sum(1 for h in post[-10:] if float(h.get("close") or 0.0) >= float(h.get("ema30") or 0.0))
                    if held >= 8:
                        prev_phase = state["phase"]
                        _phase_set(state, "MARKUP", trade_date)
                        transition = (prev_phase, state["phase"])

    if (transition is None or resumed_after_avoid) and state["phase"] == "MARKUP":
        day_range = max(0.0, high - low)
        close_bottom40 = True if day_range == 0 else close <= (low + 0.4 * day_range)
        climax = rel_volume >= 4 and day_range >= (2.5 * atr) and close_bottom40
        rsi_collapse = False
        if len(history) >= 5:
            recent = history[-5:]
            had_high = any(float(h.get("rsi_14") or 0.0) > 75 for h in recent)
            rsi_collapse = had_high and rsi < 60
        cmf_5_neg = False
        if len(history) >= 5:
            cmf_5_neg = all(float(h.get("cmf_10") or 0.0) < -0.05 for h in history[-5:])
        distribution = bool(payload.get("distribution_divergence")) or climax or rsi_collapse or cmf_5_neg
        if distribution:
            prev_phase = state["phase"]
            _phase_set(state, "DISTRIBUTION_WARNING", trade_date)
            transition = (prev_phase, state["phase"])
            signal_type = "DISTRIBUTION_WARNING"
            state_json["ema30_armed"] = True
            state_json["below_ema30_streak"] = 0

    if (transition is None or resumed_after_avoid) and state["phase"] == "MARKUP":
        anchor = float(state.get("state_json", {}).get("max_close") or close)
        anchor = max(anchor, close)
        atr = float(payload.get("atr_14") or 0.0)
        trail = anchor - (3.0 * atr)
        state_json["max_close"] = anchor
        state_json["trail_price"] = trail
        update_trailing_stop(symbol, trail)

    if (transition is None or resumed_after_avoid) and state["phase"] == "DISTRIBUTION_WARNING":
        anchor = float(state.get("state_json", {}).get("max_close") or close)
        anchor = max(anchor, close)
        atr = float(payload.get("atr_14") or 0.0)
        trail = anchor - (2.0 * atr)
        state_json["max_close"] = anchor
        state_json["trail_price"] = trail
        update_trailing_stop(symbol, trail)

    state["last_score"] = score
    state["updated_at"] = trade_date
    state["state_json"] = state_json
    if persist_state:
        _upsert_state(state)

    signal_id = 0
    emitted_signal_type: str | None = None
    if transition:
        config_hash = get_config_hash(config)
        stop_price = None
        if signal_type == "ACCUMULATION_ALERT":
            stop_price = max(0.0, min(base_low_ref, close - 2.0 * float(payload.get("atr_14") or 0.0)))
        if signal_type == "BREAKOUT_CONFIRMED":
            stop_price = max(0.0, min(base_high_ref * 0.97, close - 2.0 * float(payload.get("atr_14") or 0.0)))

        effective_signal_type = signal_type or "PHASE_ONLY"
        suppression_reason: str | None = None
        if signal_type in {"ACCUMULATION_ALERT", "BREAKOUT_CONFIRMED", "ADD_ON_PULLBACK"}:
            allowed, reason = can_open_new_position(score, int(get_cfg(config, "max_positions")))
            if not allowed:
                effective_signal_type = "SIGNAL_SUPPRESSED_RISK"
                suppression_reason = reason

        signal_id = _emit_signal(
            symbol=symbol,
            trade_date=trade_date,
            signal_type=effective_signal_type,
            phase_from=transition[0],
            phase_to=transition[1],
            score=score,
            price=close,
            stop_price=stop_price,
            evidence={
                **payload,
                "liquidity": liquidity_meta,
                "score": score,
                "had_accumulation_phase": bool(state_json.get("had_accumulation_phase")),
                "joined_externally": bool(state_json.get("joined_externally")),
                "base_lifecycle_event": state_json.get("base_lifecycle_last_event"),
                "suppressed_reason": suppression_reason,
                "attempted_signal_type": signal_type,
            },
            config_hash=config_hash,
            trace_id=trace_id,
        )
        emitted_signal_type = effective_signal_type

        if signal_type in {"ACCUMULATION_ALERT", "BREAKOUT_CONFIRMED", "ADD_ON_PULLBACK"}:
            if emitted_signal_type != "SIGNAL_SUPPRESSED_RISK":
                maybe_open_or_add_position(
                    symbol,
                    signal_type,
                    signal_id,
                    trade_date,
                    close,
                    stop_price or 0.0,
                    bool(get_cfg(config, "pilot_enabled")),
                )

        if signal_type == "EXIT":
            close_open_position(symbol, trade_date, "distribution_or_trend_break", close)

    return {
        "symbol": symbol,
        "phase": state["phase"],
        "transition": transition,
        "signal_id": signal_id,
        "signal_type": emitted_signal_type,
        "score": score,
        "state": state,
        "reason": state_json.get("last_phase_reason"),
    }


def list_watchlist() -> list[dict[str, Any]]:
    rows = query_all(
        """
        SELECT s.symbol, s.phase, s.base_high, s.base_low, s.last_score, s.updated_at,
               r.band, r.score,
               i.payload_json
        FROM ee_symbol_state s
        LEFT JOIN ee_ratings r
          ON r.symbol = s.symbol AND r.trade_date = (
              SELECT MAX(trade_date) FROM ee_ratings r2 WHERE r2.symbol = s.symbol
          )
        LEFT JOIN ee_indicators i
          ON i.symbol = s.symbol AND i.trade_date = (
              SELECT MAX(trade_date) FROM ee_indicators i2 WHERE i2.symbol = s.symbol
          )
        WHERE s.phase IN ('ACCUMULATION', 'BREAKOUT_WATCH')
        ORDER BY COALESCE(r.score, s.last_score, 0) DESC, s.symbol ASC
        """,
        (),
    )
    out: list[dict[str, Any]] = []
    for row in rows or []:
        evidence = _json_load(row.get("payload_json"))
        out.append(
            {
                "symbol": row.get("symbol"),
                "phase": row.get("phase"),
                "score": row.get("score") if row.get("score") is not None else row.get("last_score"),
                "band": row.get("band"),
                "base_high": row.get("base_high"),
                "base_low": row.get("base_low"),
                "updated_at": row.get("updated_at"),
                "evidence": {
                    "accumulation_divergence": evidence.get("accumulation_divergence"),
                    "cmf_10": evidence.get("cmf_10"),
                    "atr_pct_percentile_252": evidence.get("atr_pct_percentile_252"),
                    "rel_volume": evidence.get("rel_volume"),
                },
                "advice": False,
            }
        )
    return out
