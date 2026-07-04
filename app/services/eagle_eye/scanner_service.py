from __future__ import annotations

import json
import uuid
from typing import Any

from app.core.database import exec_sql, query_all, query_one
from app.core.security import TokenData
from app.services.eagle_eye.audit_service import create_event
from app.services.eagle_eye.entry_exit_service import close_open_position, maybe_open_or_add_position, update_trailing_stop
from app.services.eagle_eye.indicator_service import load_latest_indicator
from app.services.eagle_eye.market_data_service import CONCEPT_VERSION, get_config_hash
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
    "NEUTRAL": {"BASE_FORMING", "AVOID", "NEUTRAL"},
    "BASE_FORMING": {"ACCUMULATION", "NEUTRAL", "AVOID", "BASE_FORMING"},
    "ACCUMULATION": {"BREAKOUT_WATCH", "NEUTRAL", "AVOID", "ACCUMULATION"},
    "BREAKOUT_WATCH": {"BREAKOUT_CONFIRMED", "ACCUMULATION", "NEUTRAL", "AVOID", "BREAKOUT_WATCH"},
    "BREAKOUT_CONFIRMED": {"MARKUP", "ACCUMULATION", "EXIT", "AVOID", "BREAKOUT_CONFIRMED"},
    "MARKUP": {"DISTRIBUTION_WARNING", "EXIT", "AVOID", "MARKUP"},
    "DISTRIBUTION_WARNING": {"EXIT", "MARKUP", "AVOID", "DISTRIBUTION_WARNING"},
    "EXIT": {"NEUTRAL", "AVOID", "EXIT"},
    "AVOID": {"AVOID", "NEUTRAL"},
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


def _risk_level(signal_type: str) -> str:
    if signal_type in {"BREAKOUT_CONFIRMED", "EXIT"}:
        return "high"
    if signal_type == "ACCUMULATION_ALERT":
        return "medium"
    return "low"


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
) -> dict[str, Any]:
    trace_id = trace_id or str(uuid.uuid4())
    payload = load_latest_indicator(symbol, trade_date)
    if not payload:
        return {"symbol": symbol, "status": "no_indicator"}

    history = _load_recent_indicators(symbol, trade_date, 140)
    if not history:
        return {"symbol": symbol, "status": "no_history"}

    prev = history[-2] if len(history) >= 2 else None

    state = get_symbol_state(symbol) or {
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
    atr_pct_pctile = float(payload.get("atr_pct_percentile_252") or 1.0)

    liquidity_ok, liquidity_meta = liquidity_filter_at(
        symbol,
        trade_date,
        float(config.get("min_daily_value_kwd", 100000.0)),
    )

    state_json = state.get("state_json", {})
    avoid = close < sma200 and sma200_slope < 0 and ema10 < ema30
    if avoid:
        state_json["avoid_clear_streak"] = 0
        _phase_set(state, "AVOID", trade_date)
    elif old_phase == "AVOID":
        clear_streak = int(state_json.get("avoid_clear_streak") or 0) + 1
        state_json["avoid_clear_streak"] = clear_streak
        if clear_streak >= 20:
            _phase_set(state, "NEUTRAL", trade_date)

    transition = None
    signal_type = None
    base_min_sessions = int(config.get("base_min_sessions", 60))
    range_low_120 = float(payload.get("range_low_120") or range_low_60 or 0.0)
    low = float(payload.get("low") or close)
    high = float(payload.get("high") or close)
    atr = float(payload.get("atr_14") or 0.0)
    base_high_ref = float(state.get("base_high") or range_high_120 or 0.0)
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

        if state["phase"] == "NEUTRAL":
            if (
                width <= float(config.get("base_max_width_pct", 0.18))
                and sessions_in_range >= base_min_sessions
                and base_low_60 <= close <= base_high_60
            ):
                prev_phase = state["phase"]
                _phase_set(state, "BASE_FORMING", trade_date)
                transition = (prev_phase, state["phase"])
                state["base_high"] = range_high_120
                state["base_low"] = range_low_60
                state["base_start"] = trade_date
                base_high_ref = float(state.get("base_high") or base_high_ref)
                base_low_ref = float(state.get("base_low") or base_low_ref)

        if state["phase"] == "BASE_FORMING":
            cmf_hist = [float(h.get("cmf_10") or 0.0) for h in history[-10:]]
            cmf_hits = sum(1 for x in cmf_hist if x > float(config.get("cmf_floor", 0.05)))
            accumulation_gate = bool(payload.get("accumulation_divergence")) or (
                abs(float(payload.get("price_slope_40") or 0.0)) < 0.03
                and (
                    float(payload.get("obv_slope_40") or 0.0) > 0.10
                    or float(payload.get("anv_slope_40") or 0.0) > 0.10
                )
            )
            if (
                accumulation_gate
                and cmf_hits >= 5
                and (atr_pct_pctile <= float(config.get("atr_squeeze_pctile", 0.20)) or float(payload.get("bb_width") or 1.0) <= 0.12)
                and (close >= ema30 or close >= 0.97 * sma200)
                and liquidity_ok
                and score >= 60
            ):
                prev_phase = state["phase"]
                _phase_set(state, "ACCUMULATION", trade_date)
                transition = (prev_phase, state["phase"])
                signal_type = "ACCUMULATION_ALERT"

        if state["phase"] == "ACCUMULATION":
            recent_5 = history[-5:]
            rv_hits = sum(1 for h in recent_5 if float(h.get("rel_volume") or 0.0) >= 1.5)
            if close >= (0.97 * base_high_ref) and rv_hits >= 2:
                prev_phase = state["phase"]
                _phase_set(state, "BREAKOUT_WATCH", trade_date)
                transition = (prev_phase, state["phase"])

        if state["phase"] == "BREAKOUT_WATCH":
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
            gap_pct = 0.0 if range_high_120 <= 0 else max(0.0, (float(payload.get("open") or close) - range_high_120) / range_high_120)
            chase_guard = gap_pct <= 0.08
            breakout = (
                close > base_high_ref
                and rel_volume >= float(config.get("volume_breakout_mult", 2.5))
                and ema10 > ema30
                and float(payload.get("ema10_slope") or 0.0) > 0
                and rsi >= float(config.get("rsi_regime", 55))
                and rsi_rising
                and adx >= float(config.get("adx_trigger", 22))
                and plus_di > minus_di
                and adx > adx_5_back
                and (float(payload.get("macd_hist") or 0.0) > 0 or macd_cross_recent)
                and close_top40
                and chase_guard
                and liquidity_ok
            )
            if breakout:
                ml_ok, ml_prob = apply_ml_gate(payload, config)
                if ml_ok and score >= 70:
                    prev_phase = state["phase"]
                    _phase_set(state, "BREAKOUT_CONFIRMED", trade_date)
                    transition = (prev_phase, state["phase"])
                    signal_type = "BREAKOUT_CONFIRMED"
                    state_json["breakout_confirmed_at"] = trade_date
                    state_json["breakout_base_high"] = base_high_ref
                    state_json["breakout_entry_price"] = close
                    if ml_prob is not None:
                        payload["ml_prob"] = ml_prob

        confirmed_at = int(state_json.get("breakout_confirmed_at") or 0)
        if confirmed_at:
            confirm_ix = None
            for i, h in enumerate(history):
                if int(h.get("trade_date") or 0) == confirmed_at:
                    confirm_ix = i
                    break
            if confirm_ix is not None:
                sessions_since = len(history) - 1 - confirm_ix
                if 0 < sessions_since <= 5:
                    base_high = float(state_json.get("breakout_base_high") or range_high_120)
                    if close < (base_high * 0.97) and rel_volume >= 1.5:
                        prev_phase = state["phase"]
                        _phase_set(state, "ACCUMULATION", trade_date)
                        transition = (prev_phase, state["phase"])
                        signal_type = "BREAKOUT_FAILED"
                        close_open_position(symbol, trade_date, "breakout_failed", close)

        if state["phase"] == "BREAKOUT_CONFIRMED":
            confirmed_at = int(state_json.get("breakout_confirmed_at") or 0)
            if confirmed_at:
                post = [h for h in history if int(h.get("trade_date") or 0) >= confirmed_at]
                if len(post) >= 10:
                    held = sum(1 for h in post[-10:] if float(h.get("close") or 0.0) >= float(h.get("ema30") or 0.0))
                    if held >= 8:
                        prev_phase = state["phase"]
                        _phase_set(state, "MARKUP", trade_date)
                        transition = (prev_phase, state["phase"])

    if state["phase"] == "MARKUP":
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

    if state["phase"] == "DISTRIBUTION_WARNING":
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

    if state["phase"] == "MARKUP":
        anchor = float(state.get("state_json", {}).get("max_close") or close)
        anchor = max(anchor, close)
        atr = float(payload.get("atr_14") or 0.0)
        trail = anchor - (3.0 * atr)
        state_json["max_close"] = anchor
        state_json["trail_price"] = trail
        update_trailing_stop(symbol, trail)

    if state["phase"] == "DISTRIBUTION_WARNING":
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
    _upsert_state(state)

    signal_id = 0
    if transition:
        config_hash = get_config_hash(config)
        stop_price = None
        if signal_type == "ACCUMULATION_ALERT":
            stop_price = max(0.0, min(base_low_ref, close - 2.0 * float(payload.get("atr_14") or 0.0)))
        if signal_type == "BREAKOUT_CONFIRMED":
            stop_price = max(0.0, min(base_high_ref * 0.97, close - 2.0 * float(payload.get("atr_14") or 0.0)))

        signal_id = _emit_signal(
            symbol=symbol,
            trade_date=trade_date,
            signal_type=signal_type or "PHASE_ONLY",
            phase_from=transition[0],
            phase_to=transition[1],
            score=score,
            price=close,
            stop_price=stop_price,
            evidence={**payload, "liquidity": liquidity_meta, "score": score},
            config_hash=config_hash,
            trace_id=trace_id,
        )

        if signal_type in {"ACCUMULATION_ALERT", "BREAKOUT_CONFIRMED", "ADD_ON_PULLBACK"}:
            allowed, reason = can_open_new_position(score, int(config.get("max_positions", 8)))
            if allowed:
                maybe_open_or_add_position(
                    symbol,
                    signal_type,
                    signal_id,
                    trade_date,
                    close,
                    stop_price or 0.0,
                    bool(config.get("pilot_enabled", True)),
                )
            else:
                _emit_signal(
                    symbol=symbol,
                    trade_date=trade_date,
                    signal_type="SIGNAL_SUPPRESSED_RISK",
                    phase_from=transition[0],
                    phase_to=transition[1],
                    score=score,
                    price=close,
                    stop_price=stop_price,
                    evidence={"reason": reason, "score": score},
                    config_hash=config_hash,
                    trace_id=trace_id,
                )

        if signal_type == "EXIT":
            close_open_position(symbol, trade_date, "distribution_or_trend_break", close)

    return {
        "symbol": symbol,
        "phase": state["phase"],
        "transition": transition,
        "signal_id": signal_id,
        "score": score,
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
