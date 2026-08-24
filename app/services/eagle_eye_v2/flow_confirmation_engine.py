from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.services.eagle_eye_v2.predicate_telemetry_ledger import append_row

FLOW_OBV_SLOPE_OK = "FLOW_OBV_SLOPE_OK"
FLOW_ANV_SLOPE_OK = "FLOW_ANV_SLOPE_OK"
FLOW_ACCUMULATION_DIVERGENCE_OK = "FLOW_ACCUMULATION_DIVERGENCE_OK"
ACCUMULATION_CONTEXT_OK = "ACCUMULATION_CONTEXT_OK"

CONFIRM_FLOW_CORE_OK = "CONFIRM_FLOW_CORE_OK"
CONFIRM_STRUCTURE_OK = "CONFIRM_STRUCTURE_OK"
CONFIRM_RELATIVE_VOLUME_CONTEXT_OK = "CONFIRM_RELATIVE_VOLUME_CONTEXT_OK"
CONFIRM_CHASE_GUARD_OK = "CONFIRM_CHASE_GUARD_OK"
CURRENT_DAY_LIQUIDITY_OK = "CURRENT_DAY_LIQUIDITY_OK"
LIQUIDITY_CONTEXT_OK = "LIQUIDITY_CONTEXT_OK"

OBV_SLOPE_MIN = "obv_slope_min"
ANV_SLOPE_MIN = "anv_slope_min"
REL_VOLUME_CONTEXT_MIN = "volume_breakout_mult"
CMF_FLOOR = "cmf_floor"
RSI_REGIME = "rsi_regime"
ADX_TRIGGER = "adx_trigger"
CHASE_ADVISORY_BAND = "CHASE_ADVISORY_BAND"
MIN_DAILY_VALUE_KWD = "min_daily_value_kwd"
MIN_CURRENT_DAY_VALUE_KWD = "min_current_day_value_kwd"


@dataclass(frozen=True)
class FlowNamedParameters:
    values: dict[str, float]

    def require(self, name: str) -> float:
        if name not in self.values:
            raise ValueError(f"Missing named flow parameter: {name}")
        return float(self.values[name])


class FlowConfirmationEngine:
    """Evaluate flow and confirmation predicates and emit daily telemetry rows."""

    def __init__(self, named_parameters: FlowNamedParameters) -> None:
        self.params = named_parameters

    def evaluate(
        self,
        *,
        normalized_day_payload: dict[str, Any],
        base_reference: dict[str, Any],
        flow_history_window: list[dict[str, Any]],
        structure_terms: dict[str, Any],
        readiness_state: str,
        phase_state: str,
    ) -> dict[str, Any]:
        today = flow_history_window[-1] if flow_history_window else {}

        obv_slope = float(today.get("obv_slope_40") or 0.0)
        anv_slope = float(today.get("anv_slope_40") or 0.0)
        accumulation_div = bool(today.get("accumulation_divergence"))
        rel_volume = float(today.get("rel_volume") or 0.0)
        cmf_10 = float(today.get("cmf_10") or 0.0)
        adx_19 = float(today.get("adx_19") or 0.0)
        rsi_14 = float(today.get("rsi_14") or 0.0)

        trailing_values = [float(x.get("value_kwd") or 0.0) for x in flow_history_window[-20:] if x.get("value_kwd") is not None]
        trailing_liq = (sum(trailing_values) / len(trailing_values)) if trailing_values else 0.0
        current_day_liq = float(normalized_day_payload.get("value_kwd") or 0.0)

        obv_ok = obv_slope >= self.params.require(OBV_SLOPE_MIN)
        anv_ok = anv_slope >= self.params.require(ANV_SLOPE_MIN)
        acc_div_ok = accumulation_div
        accumulation_ok = obv_ok or anv_ok or acc_div_ok

        structure_ok = bool(structure_terms.get("close_gt_base_ref")) and bool(structure_terms.get("ema10_gt_ema30"))

        rel_ctx_ok = rel_volume >= self.params.require(REL_VOLUME_CONTEXT_MIN)

        has_current_valid_reference = bool(base_reference.get("base_reference_id")) and bool(base_reference.get("base_validity_state") == "VALID")
        base_high_ref = float(base_reference.get("base_high_ref") or 0.0)
        if has_current_valid_reference and base_high_ref > 0.0:
            extension_pct = max(0.0, (float(normalized_day_payload.get("close") or 0.0) - base_high_ref) / base_high_ref)
        else:
            extension_pct = 0.0
        chase_band = self.params.require(CHASE_ADVISORY_BAND)
        chase_advisory_flag = 1 if (has_current_valid_reference and extension_pct > chase_band) else 0
        chase_guard_ok = has_current_valid_reference

        current_day_liq_ok = current_day_liq >= self.params.require(MIN_CURRENT_DAY_VALUE_KWD)
        trailing_liq_ok = trailing_liq >= self.params.require(MIN_DAILY_VALUE_KWD)

        flow_core_ok = accumulation_ok and (cmf_10 >= self.params.require(CMF_FLOOR))
        structure_gate_ok = structure_ok and (adx_19 >= self.params.require(ADX_TRIGGER)) and (rsi_14 >= self.params.require(RSI_REGIME))

        confirmation_ok = flow_core_ok and structure_gate_ok and chase_guard_ok and current_day_liq_ok
        if confirmation_ok:
            confirmation_state = "CONFIRMED"
            reason = "FLOW_STRUCTURE_CHASE_LIQUIDITY_PASS"
        else:
            confirmation_state = "NOT_CONFIRMED"
            reason = "PREDICATE_BLOCK"

        intent_state = "INTENT_FORMED" if confirmation_ok else "INTENT_NONE"
        candidate_intent = {
            "intent_id": f"{normalized_day_payload['symbol']}::{normalized_day_payload['trade_date']}::FLOW_INTENT_V1",
            "intent_state": intent_state,
            "symbol": normalized_day_payload["symbol"],
            "trade_date": normalized_day_payload["trade_date"],
            "base_reference_id": base_reference.get("base_reference_id"),
            "confirmation_state": confirmation_state,
            "entry_tier": "BREAKOUT_CONFIRMED_ENTRY" if confirmation_ok else "NONE",
            "chase_advisory_flag": chase_advisory_flag,
            "extension_pct_vs_current_valid_reference": extension_pct,
            "context_flags": {
                "relative_volume_context_ok": rel_ctx_ok,
                "liquidity_context_ok": trailing_liq_ok,
            },
            "reason": reason,
        }

        confirmation_terms = {
            "flow_obv_slope_ok": obv_ok,
            "flow_anv_slope_ok": anv_ok,
            "flow_accumulation_divergence_ok": acc_div_ok,
            "accumulation_context_ok": accumulation_ok,
            "confirm_flow_core_ok": flow_core_ok,
            "confirm_structure_ok": structure_gate_ok,
            "confirm_relative_volume_context_ok": rel_ctx_ok,
            "confirm_chase_guard_ok": chase_guard_ok,
            "current_day_liquidity_ok": current_day_liq_ok,
            "liquidity_context_ok": trailing_liq_ok,
            "chase_advisory_flag": chase_advisory_flag,
            "extension_pct_vs_current_valid_reference": extension_pct,
        }
        confirmation_gates = [
            {"name": FLOW_OBV_SLOPE_OK, "value": obv_slope, "threshold": self.params.require(OBV_SLOPE_MIN), "pass": obv_ok},
            {"name": FLOW_ANV_SLOPE_OK, "value": anv_slope, "threshold": self.params.require(ANV_SLOPE_MIN), "pass": anv_ok},
            {"name": FLOW_ACCUMULATION_DIVERGENCE_OK, "value": acc_div_ok, "threshold": True, "pass": acc_div_ok},
            {"name": CONFIRM_FLOW_CORE_OK, "value": cmf_10, "threshold": self.params.require(CMF_FLOOR), "pass": flow_core_ok},
            {"name": CONFIRM_STRUCTURE_OK, "value": structure_gate_ok, "threshold": "close_gt_base_ref_and_ema_and_rsi_adx", "pass": structure_gate_ok},
            {"name": CONFIRM_RELATIVE_VOLUME_CONTEXT_OK, "value": rel_volume, "threshold": self.params.require(REL_VOLUME_CONTEXT_MIN), "pass": rel_ctx_ok},
            {"name": CONFIRM_CHASE_GUARD_OK, "value": has_current_valid_reference, "threshold": "current_valid_base_reference", "pass": chase_guard_ok},
            {"name": CURRENT_DAY_LIQUIDITY_OK, "value": current_day_liq, "threshold": self.params.require(MIN_CURRENT_DAY_VALUE_KWD), "pass": current_day_liq_ok},
            {"name": LIQUIDITY_CONTEXT_OK, "value": trailing_liq, "threshold": self.params.require(MIN_DAILY_VALUE_KWD), "pass": trailing_liq_ok},
        ]

        self._append_predicate(
            normalized_day_payload=normalized_day_payload,
            readiness_state=readiness_state,
            phase_state=phase_state,
            base_reference=base_reference,
            predicate_namespace="accumulation",
            predicate_name=FLOW_OBV_SLOPE_OK,
            predicate_value=obv_slope,
            threshold_param=OBV_SLOPE_MIN,
            predicate_pass=obv_ok,
            extension_pct=extension_pct,
            chase_advisory_flag=chase_advisory_flag,
            trailing_liq=trailing_liq,
            flow_snapshot=today,
            candidate_intent=candidate_intent,
            accumulation_ok=accumulation_ok,
        )
        self._append_predicate(
            normalized_day_payload=normalized_day_payload,
            readiness_state=readiness_state,
            phase_state=phase_state,
            base_reference=base_reference,
            predicate_namespace="accumulation",
            predicate_name=FLOW_ANV_SLOPE_OK,
            predicate_value=anv_slope,
            threshold_param=ANV_SLOPE_MIN,
            predicate_pass=anv_ok,
            extension_pct=extension_pct,
            chase_advisory_flag=chase_advisory_flag,
            trailing_liq=trailing_liq,
            flow_snapshot=today,
            candidate_intent=candidate_intent,
            accumulation_ok=accumulation_ok,
        )
        self._append_predicate(
            normalized_day_payload=normalized_day_payload,
            readiness_state=readiness_state,
            phase_state=phase_state,
            base_reference=base_reference,
            predicate_namespace="accumulation",
            predicate_name=FLOW_ACCUMULATION_DIVERGENCE_OK,
            predicate_value=1.0 if accumulation_div else 0.0,
            threshold_param="accumulation_divergence_true",
            predicate_pass=acc_div_ok,
            extension_pct=extension_pct,
            chase_advisory_flag=chase_advisory_flag,
            trailing_liq=trailing_liq,
            flow_snapshot=today,
            candidate_intent=candidate_intent,
            accumulation_ok=accumulation_ok,
        )
        self._append_predicate(
            normalized_day_payload=normalized_day_payload,
            readiness_state=readiness_state,
            phase_state=phase_state,
            base_reference=base_reference,
            predicate_namespace="accumulation",
            predicate_name=ACCUMULATION_CONTEXT_OK,
            predicate_value=1.0 if accumulation_ok else 0.0,
            threshold_param="obv_or_anv_or_divergence",
            predicate_pass=accumulation_ok,
            extension_pct=extension_pct,
            chase_advisory_flag=chase_advisory_flag,
            trailing_liq=trailing_liq,
            flow_snapshot=today,
            candidate_intent=candidate_intent,
            accumulation_ok=accumulation_ok,
        )

        self._append_predicate(
            normalized_day_payload=normalized_day_payload,
            readiness_state=readiness_state,
            phase_state=phase_state,
            base_reference=base_reference,
            predicate_namespace="confirmation",
            predicate_name=CONFIRM_FLOW_CORE_OK,
            predicate_value=cmf_10,
            threshold_param=CMF_FLOOR,
            predicate_pass=flow_core_ok,
            extension_pct=extension_pct,
            chase_advisory_flag=chase_advisory_flag,
            trailing_liq=trailing_liq,
            flow_snapshot=today,
            candidate_intent=candidate_intent,
            accumulation_ok=accumulation_ok,
        )
        self._append_predicate(
            normalized_day_payload=normalized_day_payload,
            readiness_state=readiness_state,
            phase_state=phase_state,
            base_reference=base_reference,
            predicate_namespace="confirmation",
            predicate_name=CONFIRM_STRUCTURE_OK,
            predicate_value=1.0 if structure_gate_ok else 0.0,
            threshold_param="close_gt_base_ref_and_ema_and_rsi_adx",
            predicate_pass=structure_gate_ok,
            extension_pct=extension_pct,
            chase_advisory_flag=chase_advisory_flag,
            trailing_liq=trailing_liq,
            flow_snapshot=today,
            candidate_intent=candidate_intent,
            accumulation_ok=accumulation_ok,
        )
        self._append_predicate(
            normalized_day_payload=normalized_day_payload,
            readiness_state=readiness_state,
            phase_state=phase_state,
            base_reference=base_reference,
            predicate_namespace="confirmation",
            predicate_name=CONFIRM_RELATIVE_VOLUME_CONTEXT_OK,
            predicate_value=rel_volume,
            threshold_param=REL_VOLUME_CONTEXT_MIN,
            predicate_pass=rel_ctx_ok,
            extension_pct=extension_pct,
            chase_advisory_flag=chase_advisory_flag,
            trailing_liq=trailing_liq,
            flow_snapshot=today,
            candidate_intent=candidate_intent,
            accumulation_ok=accumulation_ok,
        )
        self._append_predicate(
            normalized_day_payload=normalized_day_payload,
            readiness_state=readiness_state,
            phase_state=phase_state,
            base_reference=base_reference,
            predicate_namespace="confirmation",
            predicate_name=CONFIRM_CHASE_GUARD_OK,
            predicate_value=extension_pct,
            threshold_param=CHASE_ADVISORY_BAND,
            predicate_pass=chase_guard_ok,
            extension_pct=extension_pct,
            chase_advisory_flag=chase_advisory_flag,
            trailing_liq=trailing_liq,
            flow_snapshot=today,
            candidate_intent=candidate_intent,
            accumulation_ok=accumulation_ok,
        )
        self._append_predicate(
            normalized_day_payload=normalized_day_payload,
            readiness_state=readiness_state,
            phase_state=phase_state,
            base_reference=base_reference,
            predicate_namespace="confirmation",
            predicate_name=CURRENT_DAY_LIQUIDITY_OK,
            predicate_value=current_day_liq,
            threshold_param=MIN_CURRENT_DAY_VALUE_KWD,
            predicate_pass=current_day_liq_ok,
            extension_pct=extension_pct,
            chase_advisory_flag=chase_advisory_flag,
            trailing_liq=trailing_liq,
            flow_snapshot=today,
            candidate_intent=candidate_intent,
            accumulation_ok=accumulation_ok,
        )
        self._append_predicate(
            normalized_day_payload=normalized_day_payload,
            readiness_state=readiness_state,
            phase_state=phase_state,
            base_reference=base_reference,
            predicate_namespace="confirmation",
            predicate_name=LIQUIDITY_CONTEXT_OK,
            predicate_value=trailing_liq,
            threshold_param=MIN_DAILY_VALUE_KWD,
            predicate_pass=trailing_liq_ok,
            extension_pct=extension_pct,
            chase_advisory_flag=chase_advisory_flag,
            trailing_liq=trailing_liq,
            flow_snapshot=today,
            candidate_intent=candidate_intent,
            accumulation_ok=accumulation_ok,
        )

        return {
            "confirmation_state": confirmation_state,
            "confirmation_terms": confirmation_terms,
            "confirmation_gates": confirmation_gates,
            "candidate_intent": candidate_intent,
        }

    def _append_predicate(
        self,
        *,
        normalized_day_payload: dict[str, Any],
        readiness_state: str,
        phase_state: str,
        base_reference: dict[str, Any],
        predicate_namespace: str,
        predicate_name: str,
        predicate_value: float,
        threshold_param: str,
        predicate_pass: bool,
        extension_pct: float,
        chase_advisory_flag: int,
        trailing_liq: float,
        flow_snapshot: dict[str, Any],
        candidate_intent: dict[str, Any],
        accumulation_ok: bool,
    ) -> None:
        row = {
            "symbol": normalized_day_payload["symbol"],
            "trade_date": normalized_day_payload["trade_date"],
            "segment_id": normalized_day_payload["segment_id"],
            "segment_day_index": int(normalized_day_payload.get("segment_day_index") or 0),
            "phase_before": phase_state,
            "phase_after": phase_state,
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
            "intent_id": candidate_intent.get("intent_id"),
            "predicate_namespace": predicate_namespace,
            "predicate_name": predicate_name,
            "predicate_value": float(predicate_value),
            "predicate_threshold_parameter": threshold_param,
            "predicate_pass": 1 if predicate_pass else 0,
            "recoverability_state": "RECOVERABLE",
            "recoverability_reason": "PROVISIONAL_PENDING_PARAMETER_GATE",
            "source_payload_fields": "ee_indicators.payload_json,flow_history_window,structure_terms,PROVISIONAL_PENDING_PARAMETER_GATE",
            "base_reference_version": "PROVISIONAL_PENDING_PARAMETER_GATE",
            "base_reference_origin": "FLOW_CONFIRMATION_ENGINE",
            "base_reference_current_flag": 1 if base_reference.get("base_validity_state") == "VALID" else 0,
            "extension_pct_vs_current_valid_reference": float(extension_pct),
            "chase_advisory_flag": int(chase_advisory_flag),
            "current_day_value_kwd": float(normalized_day_payload.get("value_kwd") or 0.0),
            "trailing_liquidity_context_value": float(trailing_liq),
            "early_tier_flag": 0,
            "dead_money_sessions": 0,
            "flow_obv_slope_40": float(flow_snapshot.get("obv_slope_40") or 0.0),
            "flow_anv_slope_40": float(flow_snapshot.get("anv_slope_40") or 0.0),
            "flow_accumulation_divergence": 1.0 if bool(flow_snapshot.get("accumulation_divergence")) else 0.0,
            "accumulation_context_ok": 1 if accumulation_ok else 0,
            "participation_cap_pct": 0.0,
            "pilot_size_fraction": 0.0,
            "time_stop_sessions": 0,
            "entry_tier": str(candidate_intent.get("entry_tier") or "NONE"),
            "flow_evidence_snapshot": json.dumps(flow_snapshot, ensure_ascii=True, sort_keys=True),
            "current_valid_reference_value": float(base_reference.get("base_high_ref") or 0.0),
        }
        append_row("daily_term_row", row)
