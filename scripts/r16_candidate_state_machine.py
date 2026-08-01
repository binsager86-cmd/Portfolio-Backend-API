from __future__ import annotations

from copy import deepcopy
from typing import Any


PIVOT_CONFIRMATION_LAG_SESSIONS = 3
SIGNIFICANT_PIVOT_ATR_MULT = 1.5
CHANDELIER_ATR_MULT = 2.75
TIME_STOP_MFE_WAIVER_PCT = 0.08

SOFT_CLEAR_SESSIONS = 5
EMA30_LOSS_SLOPE_SESSIONS = 5
FLOW_DIVERGENCE_LOOKBACK_SESSIONS = 40
BASE_RETIREMENT_MFE_THRESHOLD = 0.20
TIME_STOP_RECHECK_SESSIONS = 20


def initial_state(variant: str) -> dict[str, Any]:
    return {
        "lifecycle_state": "NEUTRAL",
        "variant": variant,
        "position": None,
        "position_counter": 0,
        "originating_base_top": None,
        "originating_base_low": None,
        "originating_base_id": None,
        "last_base_state": "NO_BASE",
        "soft_clear_streak": 0,
        "hard_base_top_breach_streak": 0,
        "markdown_recovery_streak": 0,
        "pullback_touch_age": None,
        "ema30_loss_streak": 0,
    }


def step(machine_state: dict[str, Any], ctx: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pure R16 per-session transition. All pivots in ctx are already lag-filtered by Layer 1."""
    state = deepcopy(machine_state) if machine_state else initial_state(str(ctx.get("variant") or "A"))
    actions: list[dict[str, Any]] = []

    _sync_base_lifecycle(state, ctx)
    soft_conditions = _avoid_soft_conditions(ctx)
    avoid_soft = _update_avoid_soft(state, soft_conditions)
    avoid_hard, hard_reason = _update_avoid_hard(state, ctx)
    if avoid_hard:
        actions.extend(_force_exit_if_open(state, ctx, hard_reason))
        state["lifecycle_state"] = "MARKDOWN" if _close_below_ema30(ctx) else "NEUTRAL"
    elif state["lifecycle_state"] == "MARKDOWN":
        _update_markdown_recovery(state, ctx)

    if state.get("position") is not None:
        _advance_position(state, ctx)
        actions.extend(_variant_exit_actions(state, ctx))
        if state.get("position") is not None:
            actions.extend(_progress_time_stop_actions(state, ctx))

    if state.get("position") is None and not avoid_soft and not avoid_hard:
        entry_reason = _entry_signal(state, ctx)
        if entry_reason:
            actions.append(_open_position(state, ctx, entry_reason))

    actions.append(
        {
            "type": "DAILY_STATE",
            "state": state.get("lifecycle_state"),
            "avoid_tier": _avoid_tier(avoid_soft, avoid_hard),
            "soft_conditions": soft_conditions,
            "hard_reason": hard_reason if avoid_hard else "NONE",
        }
    )
    return state, actions


def _sync_base_lifecycle(state: dict[str, Any], ctx: dict[str, Any]) -> None:
    base_state = str(ctx.get("base_state") or "NO_BASE")
    state["last_base_state"] = base_state
    if base_state in {"BASE_FORMING", "NO_BASE"} and state.get("lifecycle_state") in {"NEUTRAL", "BASE_FORMING"}:
        state["lifecycle_state"] = "BASE_FORMING" if base_state == "BASE_FORMING" else "NEUTRAL"
    if bool(ctx.get("base_valid")) and state.get("lifecycle_state") in {"NEUTRAL", "BASE_FORMING", "RE_BASE"}:
        state["lifecycle_state"] = "BASE_VALID"
        _capture_base_reference(state, ctx)
    if base_state == "BASE_RETIRED" and float(ctx.get("base_mfe") or 0.0) >= BASE_RETIREMENT_MFE_THRESHOLD:
        state["lifecycle_state"] = "MARKUP_ACTIVE"
        _capture_base_reference(state, ctx)
    elif base_state == "BASE_RETIRED" and state.get("lifecycle_state") == "BASE_VALID":
        state["lifecycle_state"] = "NEUTRAL"


def _capture_base_reference(state: dict[str, Any], ctx: dict[str, Any]) -> None:
    base_top = ctx.get("base_top_ref")
    if base_top is not None:
        state["originating_base_top"] = float(base_top)
    base_low = ctx.get("base_low_ref")
    if base_low is not None:
        state["originating_base_low"] = float(base_low)
    if ctx.get("base_reference_id"):
        state["originating_base_id"] = str(ctx.get("base_reference_id"))


def _avoid_soft_conditions(ctx: dict[str, Any]) -> dict[str, bool]:
    pivots = ctx.get("usable_pivots") or {}
    last_high = pivots.get("last_sig_high") or {}
    prior_high = pivots.get("prior_sig_high") or {}
    s1 = _num(last_high.get("price")) is not None and _num(prior_high.get("price")) is not None and float(last_high["price"]) < float(prior_high["price"])
    s2 = _close_below_ema30(ctx) and float(ctx.get("ema30_slope_5s") or 0.0) < 0.0
    last_obv = _num(pivots.get("obv_at_last_high_pivot"))
    prior_obv = _num(pivots.get("obv_at_prior_high_pivot"))
    equal_or_higher_high = _num(last_high.get("price")) is not None and _num(prior_high.get("price")) is not None and float(last_high["price"]) >= float(prior_high["price"])
    s3 = bool(equal_or_higher_high and last_obv is not None and prior_obv is not None and last_obv < prior_obv)
    return {"S1_LOWER_HIGH": s1, "S2_TREND_LOSS": s2, "S3_FLOW_DIVERGENCE": s3}


def _update_avoid_soft(state: dict[str, Any], conditions: dict[str, bool]) -> bool:
    active = sum(1 for value in conditions.values() if value) >= 2 and state.get("lifecycle_state") in {"BASE_VALID", "MARKUP_ACTIVE", "AVOID_SOFT"}
    if active:
        state["lifecycle_state"] = "AVOID_SOFT"
        state["soft_clear_streak"] = 0
        return True
    if state.get("lifecycle_state") == "AVOID_SOFT":
        state["soft_clear_streak"] = int(state.get("soft_clear_streak") or 0) + 1
        if int(state["soft_clear_streak"]) >= SOFT_CLEAR_SESSIONS:
            state["lifecycle_state"] = "MARKUP_ACTIVE"
            state["soft_clear_streak"] = 0
            return False
        return True
    state["soft_clear_streak"] = 0
    return False


def _update_avoid_hard(state: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, str]:
    pivots = ctx.get("usable_pivots") or {}
    swing_low = pivots.get("last_markup_swing_low") or {}
    close_px = float(ctx.get("close") or 0.0)
    if _num(swing_low.get("price")) is not None and close_px < float(swing_low["price"]):
        state["lifecycle_state"] = "AVOID_HARD"
        return True, "H1_CLOSE_BELOW_MARKUP_SWING_LOW"
    base_top = state.get("originating_base_top") if state.get("originating_base_top") is not None else ctx.get("base_top_ref")
    if _num(base_top) is not None and close_px < float(base_top):
        state["hard_base_top_breach_streak"] = int(state.get("hard_base_top_breach_streak") or 0) + 1
    else:
        state["hard_base_top_breach_streak"] = 0
    if int(state.get("hard_base_top_breach_streak") or 0) >= 2:
        state["lifecycle_state"] = "AVOID_HARD"
        return True, "H2_TWO_CLOSES_BELOW_BASE_TOP"
    return False, "NONE"


def _update_markdown_recovery(state: dict[str, Any], ctx: dict[str, Any]) -> None:
    if float(ctx.get("close") or 0.0) > float(ctx.get("ema30") or 0.0):
        state["markdown_recovery_streak"] = int(state.get("markdown_recovery_streak") or 0) + 1
    else:
        state["markdown_recovery_streak"] = 0
    if int(state.get("markdown_recovery_streak") or 0) >= 5:
        state["lifecycle_state"] = "NEUTRAL"
        state["markdown_recovery_streak"] = 0


def _advance_position(state: dict[str, Any], ctx: dict[str, Any]) -> None:
    position = state.get("position") or {}
    close_px = float(ctx.get("close") or 0.0)
    position["sessions_held"] = int(position.get("sessions_held") or 0) + 1
    position["max_close"] = max(float(position.get("max_close") or close_px), close_px)
    entry_close = float(position.get("entry_close") or close_px)
    position["mfe"] = max(float(position.get("mfe") or 0.0), (close_px / entry_close) - 1.0 if entry_close else 0.0)
    state["position"] = position


def _variant_exit_actions(state: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    position = state.get("position")
    if not position:
        return []
    variant = str(state.get("variant") or "A").upper()
    close_px = float(ctx.get("close") or 0.0)
    if variant == "A":
        if _close_below_ema30(ctx):
            state["ema30_loss_streak"] = int(state.get("ema30_loss_streak") or 0) + 1
        else:
            state["ema30_loss_streak"] = 0
        if int(state.get("ema30_loss_streak") or 0) >= 2:
            return _close_position(state, ctx, "EXIT_STRUCTURAL_EMA30_2C")
    else:
        trail = float(position.get("max_close") or close_px) - CHANDELIER_ATR_MULT * float(ctx.get("atr14") or 0.0)
        if close_px < trail:
            return _close_position(state, ctx, "EXIT_CHANDELIER")
    return []


def _progress_time_stop_actions(state: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    position = state.get("position")
    if not position:
        return []
    held = int(position.get("sessions_held") or 0)
    if held < 60 or (held - 60) % TIME_STOP_RECHECK_SESSIONS != 0:
        return []
    if float(position.get("mfe") or 0.0) < TIME_STOP_MFE_WAIVER_PCT and state.get("lifecycle_state") != "MARKUP_ACTIVE":
        return _close_position(state, ctx, "EXITED_TIMESTOP_STAGNANT")
    return []


def _entry_signal(state: dict[str, Any], ctx: dict[str, Any]) -> str | None:
    base_entry = bool(ctx.get("candidate_intent_state") == "INTENT_FORMED" and ctx.get("confirmation_state") == "CONFIRMED" and ctx.get("base_valid"))
    if state.get("lifecycle_state") == "BASE_VALID" and base_entry:
        return "BASE_CONFIRMED_DIRECT"
    if state.get("lifecycle_state") != "MARKUP_ACTIVE":
        return None
    pullback = _pullback_entry(state, ctx)
    if pullback:
        return pullback
    if bool(ctx.get("flag_breakout")):
        return "MARKUP_FLAG_BREAKOUT"
    return None


def _pullback_entry(state: dict[str, Any], ctx: dict[str, Any]) -> str | None:
    low_px = float(ctx.get("low") or 0.0)
    ema10 = float(ctx.get("ema10") or 0.0)
    ema30 = float(ctx.get("ema30") or 0.0)
    close_px = float(ctx.get("close") or 0.0)
    touched = low_px <= max(ema10, ema30) and low_px >= min(ema10, ema30) or low_px <= ema30
    if touched:
        state["pullback_touch_age"] = 0
    elif state.get("pullback_touch_age") is not None:
        state["pullback_touch_age"] = int(state["pullback_touch_age"]) + 1
    if state.get("pullback_touch_age") is not None and int(state["pullback_touch_age"]) <= 3 and close_px >= ema10:
        state["pullback_touch_age"] = None
        return "MARKUP_PULLBACK_EMA_BAND"
    if state.get("pullback_touch_age") is not None and int(state["pullback_touch_age"]) > 3:
        state["pullback_touch_age"] = None
    return None


def _open_position(state: dict[str, Any], ctx: dict[str, Any], reason: str) -> dict[str, Any]:
    state["position_counter"] = int(state.get("position_counter") or 0) + 1
    position_id = f"POS{int(state['position_counter']):04d}"
    close_px = float(ctx.get("close") or 0.0)
    state["position"] = {
        "position_id": position_id,
        "entry_date": ctx.get("date"),
        "entry_close": close_px,
        "sessions_held": 0,
        "max_close": close_px,
        "mfe": 0.0,
        "entry_reason": reason,
    }
    return {"type": "OPEN_POSITION", "position_id": position_id, "entry_reason": reason, "entry_close": close_px, "date": ctx.get("date")}


def _force_exit_if_open(state: dict[str, Any], ctx: dict[str, Any], reason: str) -> list[dict[str, Any]]:
    if not state.get("position"):
        return []
    return _close_position(state, ctx, "EXIT_AVOID_HARD", detail=reason)


def _close_position(state: dict[str, Any], ctx: dict[str, Any], reason: str, detail: str | None = None) -> list[dict[str, Any]]:
    position = state.get("position")
    if not position:
        return []
    close_px = float(ctx.get("close") or 0.0)
    entry_close = float(position.get("entry_close") or 0.0)
    pnl = 0.0 if entry_close <= 0.0 else ((close_px / entry_close) - 1.0) * 100.0
    action = {
        "type": "CLOSE_POSITION",
        "position_id": position.get("position_id"),
        "exit_reason": reason,
        "exit_detail": detail or reason,
        "exit_close": close_px,
        "exit_date": ctx.get("date"),
        "entry_date": position.get("entry_date"),
        "entry_close": entry_close,
        "sessions_held": int(position.get("sessions_held") or 0),
        "pnl_pct": pnl,
    }
    state["position"] = None
    state["ema30_loss_streak"] = 0
    return [action]


def _avoid_tier(soft: bool, hard: bool) -> str:
    if hard:
        return "AVOID_HARD"
    if soft:
        return "AVOID_SOFT"
    return "NONE"


def _close_below_ema30(ctx: dict[str, Any]) -> bool:
    return float(ctx.get("close") or 0.0) < float(ctx.get("ema30") or 0.0)


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
