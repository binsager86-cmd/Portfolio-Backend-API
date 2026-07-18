from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.services.eagle_eye_v2.predicate_telemetry_ledger import append_row

BASE_GEOMETRY_WIDTH_OK = "BASE_GEOMETRY_WIDTH_OK"
BASE_DWELL_OK = "BASE_DWELL_OK"
BASE_RANGE_INCLUSION_OK = "BASE_RANGE_INCLUSION_OK"
BASE_VOLATILITY_REGIME_OK = "BASE_VOLATILITY_REGIME_OK"
BASE_REFERENCE_ADVANCE_OK = "BASE_REFERENCE_ADVANCE_OK"

BASE_MIN_SESSIONS = "base_min_sessions"
BASE_MAX_WIDTH_PCT = "base_max_width_pct"
ATR_SQUEEZE_PCTILE = "atr_squeeze_pctile"
UPWARD_RETIREMENT_MFE_THRESHOLD = "UPWARD_RETIREMENT_MFE_THRESHOLD"

RULE_CLOSE_BELOW_BASE_LOW_N = "CLOSE_BELOW_BASE_LOW_N"
RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N = "CLOSE_BELOW_BASE_LOW_BY_ATR_X_N"
RULE_VOL_BREAK_AND_RANGE_BREAK = "VOL_BREAK_AND_RANGE_BREAK"
RULE_TIME_STALE_AND_FLOW_DECAY = "TIME_STALE_AND_FLOW_DECAY"


@dataclass(frozen=True)
class BaseNamedParameters:
    values: dict[str, float]

    def require(self, name: str) -> float:
        if name not in self.values:
            raise ValueError(f"Missing named base parameter: {name}")
        return float(self.values[name])


class AdaptiveBaseGeometry:
    """Regime-aware base detection and lifecycle transitions for module (c)."""

    def __init__(self, named_parameters: BaseNamedParameters) -> None:
        self.params = named_parameters

    def evaluate(
        self,
        *,
        normalized_day_payload: dict[str, Any],
        readiness_state: str,
        price_history_window: list[dict[str, Any]],
        volatility_regime_state: dict[str, Any],
        prior_base_reference: dict[str, Any] | None,
        flow_stub_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        min_sessions = int(self.params.require(BASE_MIN_SESSIONS))
        max_width = float(self.params.require(BASE_MAX_WIDTH_PCT))
        atr_squeeze = float(self.params.require(ATR_SQUEEZE_PCTILE))
        invalidation_rule_form = str(volatility_regime_state.get("invalidation_rule_form") or RULE_CLOSE_BELOW_BASE_LOW_N).upper()
        invalidation_rule_params = dict(volatility_regime_state.get("invalidation_rule_params") or {})

        window = price_history_window
        range_sessions = int(volatility_regime_state.get("base_range_sessions") or len(window) or 1)
        range_sessions = max(1, range_sessions)
        range_window = window[-range_sessions:] if window else []
        highs = [float(r.get("high") or 0.0) for r in range_window] if range_window else [float(normalized_day_payload.get("high") or 0.0)]
        lows = [float(r.get("low") or 0.0) for r in range_window] if range_window else [float(normalized_day_payload.get("low") or 0.0)]

        high_ref = max(highs) if highs else 0.0
        low_ref = min(lows) if lows else 0.0
        width_pct = 0.0 if low_ref <= 0 else (high_ref - low_ref) / low_ref
        dwell_sessions = len(window)

        close_px = float(normalized_day_payload.get("close") or 0.0)
        range_inclusion_ok = low_ref <= close_px <= high_ref if high_ref >= low_ref else False
        vol_pctile = float(volatility_regime_state.get("atr_squeeze_pctile") or 0.0)
        volatility_ok = vol_pctile <= atr_squeeze
        width_ok = width_pct <= max_width
        dwell_ok = dwell_sessions >= min_sessions

        freeze_eligible = width_ok and dwell_ok and range_inclusion_ok and volatility_ok and readiness_state != "READINESS_PENDING"

        base_reference = dict(prior_base_reference) if prior_base_reference else None
        if base_reference is not None and not base_reference.get("base_reference_id"):
            base_reference = None
        base_state = "NO_BASE"
        transition = {
            "base_freeze_event": "NONE",
            "base_rachet_event": "NONE",
            "base_invalidate_event": "NONE",
            "no_freeze_reason": "NONE",
        }

        if base_reference is None:
            if freeze_eligible:
                base_reference = {
                    "base_reference_id": f"{normalized_day_payload['symbol']}::{normalized_day_payload['trade_date']}::BASE01",
                    "base_high_ref": float(high_ref),
                    "base_low_ref": float(low_ref),
                    "base_origin_date": normalized_day_payload["trade_date"],
                    "base_validity_state": "VALID",
                    "base_retirement_reason": "NONE",
                    "invalidation_rule_form": invalidation_rule_form,
                    "invalidation_rule_state": {},
                }
                base_state = "BASE_FROZEN"
                transition["base_freeze_event"] = "BASE_FROZEN"
            else:
                base_state = "BASE_FORMING"
                reasons: list[str] = []
                if readiness_state == "READINESS_PENDING":
                    reasons.append("READINESS_PENDING")
                if not width_ok:
                    reasons.append("WIDTH_NOT_OK")
                if not dwell_ok:
                    reasons.append("DWELL_NOT_OK")
                if not range_inclusion_ok:
                    reasons.append("RANGE_NOT_OK")
                if not volatility_ok:
                    reasons.append("VOL_REGIME_NOT_OK")
                transition["no_freeze_reason"] = ",".join(reasons) if reasons else "UNSPECIFIED"
        else:
            validity_state = str(base_reference.get("base_validity_state") or "").upper()
            if validity_state == "RETIRED" and str(base_reference.get("base_retirement_reason") or "").startswith("RETIRED_SUPERSEDED_BY_MARKUP"):
                base_reference = None
                if freeze_eligible:
                    base_reference = {
                        "base_reference_id": f"{normalized_day_payload['symbol']}::{normalized_day_payload['trade_date']}::BASE01",
                        "base_high_ref": float(high_ref),
                        "base_low_ref": float(low_ref),
                        "base_origin_date": normalized_day_payload["trade_date"],
                        "base_validity_state": "VALID",
                        "base_retirement_reason": "NONE",
                        "invalidation_rule_form": invalidation_rule_form,
                        "invalidation_rule_state": {},
                    }
                    base_state = "BASE_FROZEN"
                    transition["base_freeze_event"] = "BASE_FROZEN"
                else:
                    base_state = "BASE_FORMING"
            elif validity_state == "RETIRED":
                base_state = "BASE_RETIRED"
            else:
                base_reference["base_validity_state"] = "VALID"
                base_state = "BASE_VALID"
                flow_confirmed_progress = bool((flow_stub_state or {}).get("confirmed_progress"))
                if flow_confirmed_progress and close_px > float(base_reference.get("base_high_ref") or 0.0):
                    base_reference["base_high_ref"] = float(close_px)
                    transition["base_rachet_event"] = "BASE_REFERENCE_ADVANCE_OK"

                retire, retire_reason, next_rule_state = self._evaluate_invalidation(
                    rule_form=str(base_reference.get("invalidation_rule_form") or invalidation_rule_form),
                    rule_params=invalidation_rule_params,
                    rule_state=base_reference.get("invalidation_rule_state"),
                    close_px=close_px,
                    base_low_ref=float(base_reference.get("base_low_ref") or 0.0),
                    base_high_ref=float(base_reference.get("base_high_ref") or 0.0),
                    vol_pctile=vol_pctile,
                    atr_value=float(volatility_regime_state.get("atr_value") or 0.0),
                    flow_confirmed_progress=flow_confirmed_progress,
                )
                if not retire:
                    upward_retirement = self._evaluate_upward_retirement(
                        rule_params=volatility_regime_state,
                        rule_state=next_rule_state,
                        high_px=float(normalized_day_payload.get("high") or close_px),
                        base_high_ref=float(base_reference.get("base_high_ref") or 0.0),
                    )
                    if upward_retirement is not None:
                        retire, retire_reason, next_rule_state = upward_retirement
                base_reference["invalidation_rule_state"] = next_rule_state
                if retire:
                    base_reference["base_validity_state"] = "RETIRED"
                    base_reference["base_retirement_reason"] = retire_reason
                    transition["base_invalidate_event"] = "BASE_INVALIDATED"
                    base_state = "BASE_RETIRED"

        if base_reference is None:
            base_reference = {
                "base_reference_id": None,
                "base_high_ref": None,
                "base_low_ref": None,
                "base_origin_date": None,
                "base_validity_state": "NONE",
                "base_retirement_reason": transition["no_freeze_reason"],
                "invalidation_rule_form": invalidation_rule_form,
                "invalidation_rule_state": {},
            }

        self._append_base_predicate(
            normalized_day_payload=normalized_day_payload,
            predicate_name=BASE_GEOMETRY_WIDTH_OK,
            predicate_value=width_pct,
            threshold_param=BASE_MAX_WIDTH_PCT,
            predicate_pass=width_ok,
            readiness_state=readiness_state,
            base_reference=base_reference,
            transition=transition,
            extra_context={"dwell_sessions": dwell_sessions, "vol_pctile": vol_pctile},
        )
        self._append_base_predicate(
            normalized_day_payload=normalized_day_payload,
            predicate_name=BASE_DWELL_OK,
            predicate_value=float(dwell_sessions),
            threshold_param=BASE_MIN_SESSIONS,
            predicate_pass=dwell_ok,
            readiness_state=readiness_state,
            base_reference=base_reference,
            transition=transition,
            extra_context={"width_pct": width_pct, "vol_pctile": vol_pctile},
        )
        self._append_base_predicate(
            normalized_day_payload=normalized_day_payload,
            predicate_name=BASE_RANGE_INCLUSION_OK,
            predicate_value=close_px,
            threshold_param="base_range_window",
            predicate_pass=range_inclusion_ok,
            readiness_state=readiness_state,
            base_reference=base_reference,
            transition=transition,
            extra_context={"range_low": low_ref, "range_high": high_ref},
        )
        self._append_base_predicate(
            normalized_day_payload=normalized_day_payload,
            predicate_name=BASE_VOLATILITY_REGIME_OK,
            predicate_value=vol_pctile,
            threshold_param=ATR_SQUEEZE_PCTILE,
            predicate_pass=volatility_ok,
            readiness_state=readiness_state,
            base_reference=base_reference,
            transition=transition,
            extra_context={"width_pct": width_pct, "dwell_sessions": dwell_sessions},
        )
        self._append_base_predicate(
            normalized_day_payload=normalized_day_payload,
            predicate_name=BASE_REFERENCE_ADVANCE_OK,
            predicate_value=1.0 if transition["base_rachet_event"] == "BASE_REFERENCE_ADVANCE_OK" else 0.0,
            threshold_param="flow_confirmed_progress",
            predicate_pass=transition["base_rachet_event"] == "BASE_REFERENCE_ADVANCE_OK",
            readiness_state=readiness_state,
            base_reference=base_reference,
            transition=transition,
            extra_context={"flow_stub": flow_stub_state or {}},
            namespace="lifecycle",
        )

        return {
            "base_state": base_state,
            "base_transition_terms": transition,
            "base_reference": base_reference,
        }

    @staticmethod
    def _evaluate_invalidation(
        *,
        rule_form: str,
        rule_params: dict[str, Any],
        rule_state: Any,
        close_px: float,
        base_low_ref: float,
        base_high_ref: float,
        vol_pctile: float,
        atr_value: float,
        flow_confirmed_progress: bool,
    ) -> tuple[bool, str, dict[str, Any]]:
        state: dict[str, Any] = dict(rule_state) if isinstance(rule_state, dict) else {}

        if rule_form == RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N:
            atr_mult = float(rule_params.get("atr_mult") or 1.0)
            n_sessions = max(1, int(rule_params.get("n_sessions") or 1))
            threshold = base_low_ref - (atr_mult * max(0.0, atr_value))
            streak = int(state.get("streak") or 0)
            streak = streak + 1 if close_px < threshold else 0
            state["streak"] = streak
            state["threshold"] = threshold
            retire = streak >= n_sessions
            reason = f"{RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N}(atr_mult={atr_mult},n={n_sessions})"
            return retire, reason, state

        if rule_form == RULE_VOL_BREAK_AND_RANGE_BREAK:
            vol_break_pctile = float(rule_params.get("vol_break_pctile") or 0.9)
            n_sessions = max(1, int(rule_params.get("n_sessions") or 1))
            streak = int(state.get("streak") or 0)
            breached = (close_px < base_low_ref) and (vol_pctile >= vol_break_pctile)
            streak = streak + 1 if breached else 0
            state["streak"] = streak
            retire = streak >= n_sessions
            reason = f"{RULE_VOL_BREAK_AND_RANGE_BREAK}(vol_break_pctile={vol_break_pctile},n={n_sessions})"
            return retire, reason, state

        if rule_form == RULE_TIME_STALE_AND_FLOW_DECAY:
            min_age_sessions = max(1, int(rule_params.get("min_age_sessions") or 40))
            flow_decay_n = max(1, int(rule_params.get("flow_decay_n") or 8))
            age = int(state.get("age_sessions") or 0) + 1
            flow_streak = int(state.get("flow_decay_streak") or 0)
            flow_streak = flow_streak + 1 if not flow_confirmed_progress else 0
            state["age_sessions"] = age
            state["flow_decay_streak"] = flow_streak
            retire = age >= min_age_sessions and flow_streak >= flow_decay_n and close_px < base_high_ref
            reason = f"{RULE_TIME_STALE_AND_FLOW_DECAY}(age>={min_age_sessions},flow_decay_n={flow_decay_n})"
            return retire, reason, state

        # Default form: close below base low for N consecutive sessions.
        n_sessions = max(1, int(rule_params.get("n_sessions") or 1))
        streak = int(state.get("streak") or 0)
        streak = streak + 1 if close_px < base_low_ref else 0
        state["streak"] = streak
        retire = streak >= n_sessions
        reason = f"{RULE_CLOSE_BELOW_BASE_LOW_N}(n={n_sessions})"
        return retire, reason, state

    @staticmethod
    def _evaluate_upward_retirement(
        *,
        rule_params: dict[str, Any],
        rule_state: dict[str, Any],
        high_px: float,
        base_high_ref: float,
    ) -> tuple[bool, str, dict[str, Any]] | None:
        if UPWARD_RETIREMENT_MFE_THRESHOLD not in rule_params:
            return None
        threshold = float(rule_params[UPWARD_RETIREMENT_MFE_THRESHOLD])
        state = dict(rule_state)
        upward = dict(state.get("upward_retirement") or {})
        age = int(upward.get("age_sessions") or 0) + 1
        mfe = 0.0 if base_high_ref <= 0.0 else max(float(upward.get("mfe") or 0.0), (high_px / base_high_ref) - 1.0)
        upward["age_sessions"] = age
        upward["mfe"] = mfe
        upward["threshold"] = threshold
        state["upward_retirement"] = upward
        retire = age <= 120 and mfe >= threshold
        if not retire:
            return False, "NONE", state
        reason = f"RETIRED_SUPERSEDED_BY_MARKUP({UPWARD_RETIREMENT_MFE_THRESHOLD}={threshold},sessions<=120)"
        return True, reason, state

    def _append_base_predicate(
        self,
        *,
        normalized_day_payload: dict[str, Any],
        predicate_name: str,
        predicate_value: float,
        threshold_param: str,
        predicate_pass: bool,
        readiness_state: str,
        base_reference: dict[str, Any],
        transition: dict[str, Any],
        extra_context: dict[str, Any],
        namespace: str = "base",
    ) -> None:
        payload = {
            "symbol": normalized_day_payload["symbol"],
            "trade_date": normalized_day_payload["trade_date"],
            "segment_id": normalized_day_payload["segment_id"],
            "segment_day_index": int(normalized_day_payload.get("segment_day_index") or 0),
            "phase_before": readiness_state,
            "phase_after": readiness_state,
            "readiness_state": readiness_state,
            "readiness_transition_event": "NO_TRANSITION",
            "readiness_transition_from_state": readiness_state,
            "readiness_transition_to_state": readiness_state,
            "segment_restart_flag": 0,
            "masked_context_flag": 1 if bool(normalized_day_payload.get("masked_context", {}).get("masked_flag")) else 0,
            "lookback_long_sessions": 0,
            "lookback_segment_sessions": int(normalized_day_payload.get("segment_day_index") or 0) + 1,
            "lookback_fallback_sessions": 0,
            "base_reference_id": base_reference.get("base_reference_id"),
            "intent_id": None,
            "predicate_namespace": namespace,
            "predicate_name": predicate_name,
            "predicate_value": float(predicate_value),
            "predicate_threshold_parameter": threshold_param,
            "predicate_pass": 1 if predicate_pass else 0,
            "recoverability_state": "RECOVERABLE",
            "recoverability_reason": str(transition.get("no_freeze_reason") or "NONE"),
            "source_payload_fields": "price_history_window,volatility_regime_state,flow_stub_state,PROVISIONAL_PENDING_PARAMETER_GATE",
            "base_reference_version": "PROVISIONAL_PENDING_PARAMETER_GATE",
            "base_reference_origin": "ADAPTIVE_BASE_GEOMETRY",
            "base_reference_current_flag": 1 if base_reference.get("base_validity_state") == "VALID" else 0,
            "extension_pct_vs_current_valid_reference": 0.0,
            "chase_advisory_flag": 0,
            "current_day_value_kwd": float(normalized_day_payload.get("value_kwd") or 0.0),
            "trailing_liquidity_context_value": 0.0,
            "early_tier_flag": 0,
            "dead_money_sessions": 0,
            "flow_obv_slope_40": 0.0,
            "flow_anv_slope_40": 0.0,
            "flow_accumulation_divergence": 0.0,
            "accumulation_context_ok": 0,
            "participation_cap_pct": 0.0,
            "pilot_size_fraction": 0.0,
            "time_stop_sessions": 0,
            "entry_tier": "NONE",
            "flow_evidence_snapshot": json.dumps(extra_context, ensure_ascii=True, sort_keys=True),
            "current_valid_reference_value": float(base_reference.get("base_high_ref") or 0.0),
        }
        append_row("daily_term_row", payload)
