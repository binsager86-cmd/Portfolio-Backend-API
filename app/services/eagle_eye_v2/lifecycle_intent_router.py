from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.eagle_eye_v2.staged_position_policy import (
    CHASE_ADVISORY_THRESHOLD,
    CHASE_ESCALATION_THRESHOLD,
    EARLY_TIER_PARTICIPATION_CAP,
    EARLY_TIER_SIZE_FRACTION,
    EARLY_TIER_TIME_STOP,
    SCALE_ON_CONFIRMATION,
    StagedPolicyNamedParameters,
    StagedPositionPolicy,
)

DEFERRED_INTENT_ACTIVE = "DEFERRED_INTENT_ACTIVE"
DEFERRED_INTENT_EXPIRY_OK = "DEFERRED_INTENT_EXPIRY_OK"
EARLY_INTENT_ACTIVE = "EARLY_INTENT_ACTIVE"
EARLY_INTENT_SCALE_READY = "EARLY_INTENT_SCALE_READY"


@dataclass(frozen=True)
class LifecycleRouterNamedParameters:
    values: dict[str, float | str]

    def require_float(self, name: str) -> float:
        if name not in self.values:
            raise ValueError(f"Missing lifecycle parameter: {name}")
        return float(self.values[name])


class LifecycleIntentRouter:
    """Route candidate intents into deferred/early/confirmed execution intents."""

    def __init__(self, named_parameters: LifecycleRouterNamedParameters) -> None:
        self.params = named_parameters
        self.staged = StagedPositionPolicy(
            StagedPolicyNamedParameters(
                values={
                    EARLY_TIER_SIZE_FRACTION: self.params.values.get(EARLY_TIER_SIZE_FRACTION, 0.30),
                    EARLY_TIER_PARTICIPATION_CAP: self.params.values.get(EARLY_TIER_PARTICIPATION_CAP, 0.10),
                    EARLY_TIER_TIME_STOP: self.params.values.get(EARLY_TIER_TIME_STOP, 60.0),
                    SCALE_ON_CONFIRMATION: self.params.values.get(SCALE_ON_CONFIRMATION, "SINGLE_ADD_TO_FULL_TARGET"),
                    CHASE_ADVISORY_THRESHOLD: self.params.values.get(CHASE_ADVISORY_THRESHOLD, 0.08),
                    CHASE_ESCALATION_THRESHOLD: self.params.values.get(CHASE_ESCALATION_THRESHOLD, 0.15),
                }
            )
        )

    def evaluate(
        self,
        *,
        candidate_intent: dict[str, Any],
        base_state: dict[str, Any],
        confirmation_state: dict[str, Any],
        risk_budget_state: dict[str, Any],
    ) -> dict[str, Any]:
        current_state = dict(risk_budget_state.get("deferred_intent_state") or {})
        deferred_age = int(current_state.get("age_sessions") or 0)
        rearm_count = int(current_state.get("rearm_count") or 0)
        flow_decay = bool(current_state.get("flow_evidence_decay"))

        avoid_veto = bool(risk_budget_state.get("avoid_veto"))
        candidate_state = str(candidate_intent.get("intent_state") or "INTENT_NONE")
        base_valid = str(base_state.get("base_state") or "").upper() in {"BASE_VALID", "BASE_FROZEN"}
        confirmed = str(confirmation_state.get("confirmation_state") or "").upper() == "CONFIRMED"

        deferred_active = False
        deferred_expiry_ok = True
        veto_record: dict[str, Any] = {"veto": False, "plane": "NONE", "reason": "NONE"}

        if avoid_veto:
            veto_record = {"veto": True, "plane": "AVOID", "reason": "AVOID_PLANE_ACTIVE"}
        elif candidate_state == "INTENT_FORMED" and base_valid and not confirmed:
            deferred_active = True
            deferred_age += 1
            max_sessions = int(self.params.require_float(EARLY_TIER_TIME_STOP))
            if deferred_age > max_sessions:
                if flow_decay:
                    deferred_active = False
                    deferred_expiry_ok = False
                    veto_record = {"veto": True, "plane": "TIME_STOP", "reason": "FLOW_EVIDENCE_DECAY"}
                elif rearm_count < 2:
                    rearm_count += 1
                    deferred_age = 1
                else:
                    deferred_expiry_ok = False
                    veto_record = {"veto": True, "plane": "TIME_STOP", "reason": "OWNER_REVIEW_REQUIRED"}

        deferred_intent = {
            "active": deferred_active,
            "age_sessions": deferred_age,
            "rearm_count": rearm_count,
            "expiry_ok": deferred_expiry_ok,
            "state": "DEFERRED_INTENT" if deferred_active else "NONE",
            "base_reference_id": candidate_intent.get("base_reference_id"),
        }

        extension_pct = float(candidate_intent.get("extension_pct_vs_current_valid_reference") or 0.0)
        staged = self.staged.evaluate(
            confirmation_state=str(confirmation_state.get("confirmation_state") or "NOT_CONFIRMED"),
            deferred_intent=deferred_intent,
            risk_budget_state=risk_budget_state,
            extension_pct=extension_pct,
        )

        early_active = deferred_active and staged.get("cap_ok", True)
        early_scale_ready = bool(staged.get("scale_action", {}).get("scale_ready"))
        confirmed_direct_ready = bool(
            candidate_state == "INTENT_FORMED"
            and base_valid
            and confirmed
            and not current_state.get("active")
            and not deferred_active
        )

        execution_state = "NONE"
        entry_tier = "NONE"
        if early_scale_ready and not veto_record.get("veto"):
            execution_state = "EXECUTE_CONFIRMED_ADD"
            entry_tier = "BREAKOUT_CONFIRMED_ENTRY"
        elif early_active and not veto_record.get("veto"):
            execution_state = "EXECUTE_EARLY_PILOT"
            entry_tier = "EARLY_ACCUMULATION_ENTRY"
        elif confirmed_direct_ready and not veto_record.get("veto"):
            execution_state = "EXECUTE_CONFIRMED_DIRECT"
            entry_tier = "BREAKOUT_CONFIRMED_ENTRY"

        execution_intent = {
            "execution_state": execution_state,
            "entry_tier": entry_tier,
            "pilot_size_fraction": staged.get("early_entry", {}).get("pilot_size_fraction", 0.0),
            "target_fraction": 1.0 if execution_state == "EXECUTE_CONFIRMED_DIRECT" else 0.0,
            "participation_cap_pct": staged.get("early_entry", {}).get("participation_cap_pct", 0.0),
            "dead_money_sessions": deferred_age if deferred_active else 0,
            "time_stop_sessions": None if execution_state == "EXECUTE_CONFIRMED_DIRECT" else staged.get("early_entry", {}).get("time_stop_sessions", 60),
            "chase_advisory": staged.get("chase_band", {}),
        }

        lifecycle_terms = {
            DEFERRED_INTENT_ACTIVE: deferred_active,
            DEFERRED_INTENT_EXPIRY_OK: deferred_expiry_ok,
            EARLY_INTENT_ACTIVE: early_active,
            EARLY_INTENT_SCALE_READY: early_scale_ready,
        }

        return {
            "execution_intent": execution_intent,
            "deferred_intent": deferred_intent,
            "veto_record": veto_record,
            "lifecycle_terms": lifecycle_terms,
        }
