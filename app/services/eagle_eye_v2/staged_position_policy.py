from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EARLY_TIER_SIZE_FRACTION = "EARLY_TIER_SIZE_FRACTION"
EARLY_TIER_PARTICIPATION_CAP = "EARLY_TIER_PARTICIPATION_CAP"
EARLY_TIER_TIME_STOP = "EARLY_TIER_TIME_STOP"
SCALE_ON_CONFIRMATION = "SCALE_ON_CONFIRMATION"
CHASE_ADVISORY_THRESHOLD = "CHASE_ADVISORY_THRESHOLD"
CHASE_ESCALATION_THRESHOLD = "CHASE_ESCALATION_THRESHOLD"


@dataclass(frozen=True)
class StagedPolicyNamedParameters:
    values: dict[str, float | str]

    def require_float(self, name: str) -> float:
        if name not in self.values:
            raise ValueError(f"Missing staged policy parameter: {name}")
        return float(self.values[name])

    def require_text(self, name: str) -> str:
        if name not in self.values:
            raise ValueError(f"Missing staged policy parameter: {name}")
        return str(self.values[name])


class StagedPositionPolicy:
    """Two-tier staged execution policy frozen at the R14-B parameter gate."""

    def __init__(self, named_parameters: StagedPolicyNamedParameters) -> None:
        self.params = named_parameters

    def evaluate(
        self,
        *,
        confirmation_state: str,
        deferred_intent: dict[str, Any],
        risk_budget_state: dict[str, Any],
        extension_pct: float,
    ) -> dict[str, Any]:
        pilot_fraction = self.params.require_float(EARLY_TIER_SIZE_FRACTION)
        participation_cap = self.params.require_float(EARLY_TIER_PARTICIPATION_CAP)
        time_stop_sessions = int(self.params.require_float(EARLY_TIER_TIME_STOP))
        scale_rule = self.params.require_text(SCALE_ON_CONFIRMATION)
        chase_threshold = self.params.require_float(CHASE_ADVISORY_THRESHOLD)
        escalate_threshold = self.params.require_float(CHASE_ESCALATION_THRESHOLD)

        day_value = float(risk_budget_state.get("current_day_value_kwd") or 0.0)
        planned_order_value = float(risk_budget_state.get("planned_order_value_kwd") or 0.0)

        cap_ok = True
        cap_ratio = 0.0
        if day_value > 0.0:
            cap_ratio = planned_order_value / day_value
            cap_ok = cap_ratio <= participation_cap

        early_open = bool(deferred_intent.get("active")) and cap_ok
        early_entry = {
            "entry_tier": "EARLY_ACCUMULATION_ENTRY" if early_open else "NONE",
            "pilot_size_fraction": pilot_fraction if early_open else 0.0,
            "participation_cap_pct": participation_cap,
            "participation_ratio_used": cap_ratio,
            "time_stop_sessions": time_stop_sessions,
            "time_stop_review_semantics": "FLOW_EVIDENCE_DECAY_ONLY_REARM_MAX_2",
        }

        scale_ready = bool(confirmation_state == "CONFIRMED" and deferred_intent.get("active"))
        scale_action = {
            "scale_ready": scale_ready,
            "scale_rule": scale_rule,
            "execution_tier": "BREAKOUT_CONFIRMED_ENTRY" if scale_ready else "NONE",
        }

        return {
            "early_entry": early_entry,
            "scale_action": scale_action,
            "chase_band": {
                "advisory_threshold": chase_threshold,
                "escalation_threshold": escalate_threshold,
                "advisory_flag": extension_pct > chase_threshold,
                "escalation_flag": extension_pct > escalate_threshold,
                "extension_pct_vs_current_valid_reference": extension_pct,
            },
            "cap_ok": cap_ok,
        }
