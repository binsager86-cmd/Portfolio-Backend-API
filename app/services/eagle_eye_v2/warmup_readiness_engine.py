from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.eagle_eye_v2.predicate_telemetry_ledger import append_row

READINESS_LONG_LOOKBACK_READY = "READINESS_LONG_LOOKBACK_READY"
READINESS_SEGMENT_RESTART_READY = "READINESS_SEGMENT_RESTART_READY"
READINESS_FALLBACK_ELIGIBLE = "READINESS_FALLBACK_ELIGIBLE"

READINESS_LONG_LOOKBACK_MIN_SESSIONS = "READINESS_LONG_LOOKBACK_MIN_SESSIONS"
READINESS_SEGMENT_RESTART_MIN_SESSIONS = "READINESS_SEGMENT_RESTART_MIN_SESSIONS"
READINESS_FALLBACK_MIN_SESSIONS = "READINESS_FALLBACK_MIN_SESSIONS"


@dataclass(frozen=True)
class WarmupNamedParameters:
    values: dict[str, int]

    def require(self, name: str) -> int:
        if name not in self.values:
            raise ValueError(f"Missing named readiness parameter: {name}")
        return int(self.values[name])


class WarmupReadinessEngine:
    """Evaluate readiness predicates and persist every predicate term to ledger."""

    def __init__(self, named_parameters: WarmupNamedParameters) -> None:
        self.params = named_parameters

    def _predicate_row(
        self,
        *,
        normalized_day_payload: dict[str, Any],
        predicate_name: str,
        predicate_value: float,
        threshold_param: str,
        predicate_pass: bool,
        phase_before: str,
        phase_after: str,
        readiness_state: str,
        readiness_reason: str,
        readiness_transition_event: str,
        readiness_transition_from_state: str,
        readiness_transition_to_state: str,
        segment_restart_flag: bool,
        long_sessions: int,
        segment_sessions: int,
        fallback_sessions: int,
    ) -> dict[str, Any]:
        return {
            "symbol": normalized_day_payload["symbol"],
            "trade_date": normalized_day_payload["trade_date"],
            "segment_id": normalized_day_payload["segment_id"],
            "segment_day_index": int(normalized_day_payload.get("segment_day_index") or 0),
            "phase_before": phase_before,
            "phase_after": phase_after,
            "readiness_state": readiness_state,
            "readiness_transition_event": readiness_transition_event,
            "readiness_transition_from_state": readiness_transition_from_state,
            "readiness_transition_to_state": readiness_transition_to_state,
            "segment_restart_flag": 1 if segment_restart_flag else 0,
            "masked_context_flag": 1 if bool(normalized_day_payload.get("masked_context", {}).get("masked_flag")) else 0,
            "lookback_long_sessions": int(long_sessions),
            "lookback_segment_sessions": int(segment_sessions),
            "lookback_fallback_sessions": int(fallback_sessions),
            "base_reference_id": None,
            "intent_id": None,
            "predicate_namespace": "warmup",
            "predicate_name": predicate_name,
            "predicate_value": float(predicate_value),
            "predicate_threshold_parameter": threshold_param,
            "predicate_pass": 1 if predicate_pass else 0,
            "recoverability_state": "RECOVERABLE",
            "recoverability_reason": readiness_reason,
            "source_payload_fields": "coverage_history,segment_restart_flag,masked_context",
            "base_reference_version": None,
            "base_reference_origin": None,
            "base_reference_current_flag": 0,
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
            "flow_evidence_snapshot": "{}",
            "current_valid_reference_value": 0.0,
        }

    def evaluate(
        self,
        *,
        normalized_day_payload: dict[str, Any],
        coverage_history: dict[str, Any],
        segment_restart_flag: bool,
    ) -> dict[str, str]:
        long_sessions = self._resolve_count(
            coverage_history,
            primary_key="long_lookback_sessions",
            container_key="long_lookback_session_dates",
        )
        segment_sessions = self._resolve_count(
            coverage_history,
            primary_key="segment_sessions",
            container_key="segment_session_dates",
        )
        fallback_sessions = self._resolve_count(
            coverage_history,
            primary_key="fallback_sessions",
            container_key="fallback_session_dates",
        )

        long_min = self.params.require(READINESS_LONG_LOOKBACK_MIN_SESSIONS)
        seg_restart_min = self.params.require(READINESS_SEGMENT_RESTART_MIN_SESSIONS)
        fallback_min = self.params.require(READINESS_FALLBACK_MIN_SESSIONS)

        p_long = long_sessions >= long_min
        p_seg = (not segment_restart_flag) or (segment_sessions >= seg_restart_min)
        p_fallback = fallback_sessions >= fallback_min

        if p_long and p_seg:
            readiness_state = "READY"
            readiness_reason = "LONG_LOOKBACK_AND_SEGMENT_RESTART_READY"
        elif p_fallback:
            readiness_state = "READINESS_LIMITED"
            readiness_reason = "FALLBACK_ELIGIBLE"
        else:
            readiness_state = "READINESS_PENDING"
            readiness_reason = "INSUFFICIENT_COVERAGE"

        prev_state = str(coverage_history.get("previous_readiness_state") or "READINESS_PENDING")
        if prev_state != readiness_state:
            transition_event = f"{prev_state}_TO_{readiness_state}"
        else:
            transition_event = "NO_TRANSITION"
        transition_from_state = prev_state
        transition_to_state = readiness_state

        append_row(
            "daily_term_row",
            self._predicate_row(
                normalized_day_payload=normalized_day_payload,
                predicate_name=READINESS_LONG_LOOKBACK_READY,
                predicate_value=float(long_sessions),
                threshold_param=READINESS_LONG_LOOKBACK_MIN_SESSIONS,
                phase_before=prev_state,
                phase_after=readiness_state,
                predicate_pass=p_long,
                readiness_state=readiness_state,
                readiness_reason=readiness_reason,
                readiness_transition_event=transition_event,
                readiness_transition_from_state=transition_from_state,
                readiness_transition_to_state=transition_to_state,
                segment_restart_flag=segment_restart_flag,
                long_sessions=long_sessions,
                segment_sessions=segment_sessions,
                fallback_sessions=fallback_sessions,
            ),
        )
        append_row(
            "daily_term_row",
            self._predicate_row(
                normalized_day_payload=normalized_day_payload,
                predicate_name=READINESS_SEGMENT_RESTART_READY,
                predicate_value=float(segment_sessions),
                threshold_param=READINESS_SEGMENT_RESTART_MIN_SESSIONS,
                phase_before=prev_state,
                phase_after=readiness_state,
                predicate_pass=p_seg,
                readiness_state=readiness_state,
                readiness_reason=readiness_reason,
                readiness_transition_event=transition_event,
                readiness_transition_from_state=transition_from_state,
                readiness_transition_to_state=transition_to_state,
                segment_restart_flag=segment_restart_flag,
                long_sessions=long_sessions,
                segment_sessions=segment_sessions,
                fallback_sessions=fallback_sessions,
            ),
        )
        append_row(
            "daily_term_row",
            self._predicate_row(
                normalized_day_payload=normalized_day_payload,
                predicate_name=READINESS_FALLBACK_ELIGIBLE,
                predicate_value=float(fallback_sessions),
                threshold_param=READINESS_FALLBACK_MIN_SESSIONS,
                phase_before=prev_state,
                phase_after=readiness_state,
                predicate_pass=p_fallback,
                readiness_state=readiness_state,
                readiness_reason=readiness_reason,
                readiness_transition_event=transition_event,
                readiness_transition_from_state=transition_from_state,
                readiness_transition_to_state=transition_to_state,
                segment_restart_flag=segment_restart_flag,
                long_sessions=long_sessions,
                segment_sessions=segment_sessions,
                fallback_sessions=fallback_sessions,
            ),
        )

        return {
            "readiness_state": readiness_state,
            "readiness_reason": readiness_reason,
            "readiness_transition_event": transition_event,
        }

    @staticmethod
    def _resolve_count(
        coverage_history: dict[str, Any],
        *,
        primary_key: str,
        container_key: str,
    ) -> int:
        container = coverage_history.get(container_key)
        if isinstance(container, (list, tuple, set)):
            return len(container)
        if isinstance(container, dict):
            items = container.get("dates")
            if isinstance(items, (list, tuple, set)):
                return len(items)

        value = coverage_history.get(primary_key)
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, (list, tuple, set)):
            return len(value)
        return 0
