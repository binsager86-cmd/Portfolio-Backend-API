# R14 Design Spec v1.1

Supersedes: R14 Design Spec v1

Amendment focus:
- Explicitly answer F8b by making reference advancement during confirmed accumulation a first-class lifecycle requirement.
- Require chase guard evaluation against the current valid reference, not the original freeze reference.
- The 2025-05-18 final probe resolved the surfaced blocker to M5_liquidity, not a composite/ML layer; F8c is therefore not established as a new mechanism.
- Any veto-capable post-mandatory layer must still be fully telemetried if retained, and liquidity authority must remain explicit in telemetry and acceptance criteria.

New/strengthened design principles:
- Base-reference lifecycle includes freeze, ratchet/advance, invalidate, retire, and current-valid-reference designation.
- Chase guard consumes the current valid reference only.
- Deferred intent and confirmation share a common reference object so confirmation latency cannot structurally create an unwinnable race.
- No unauditable gate may retain blocking authority.

State-machine additions:
- lifecycle terms: ['BASE_REFERENCE_PRESENT', 'BASE_REFERENCE_VALID', 'DEFERRED_INTENT_ACTIVE', 'DEFERRED_INTENT_EXPIRY_OK', 'BASE_REFERENCE_ADVANCE_OK', 'CHASE_GUARD_CURRENT_REF_OK']

Telemetry additions:
- daily_term_row: ['symbol', 'trade_date', 'segment_id', 'phase_before', 'phase_after', 'readiness_state', 'base_reference_id', 'intent_id', 'predicate_namespace', 'predicate_name', 'predicate_value', 'predicate_threshold_parameter', 'predicate_pass', 'recoverability_state', 'recoverability_reason', 'source_payload_fields', 'base_reference_version', 'base_reference_origin', 'base_reference_current_flag']

Updated finding-response map:
- F8a: Solved by persistent base references plus readiness-aware base freeze so missing-reference disarm cannot persist silently.
- F8b: Solved by advancing current-valid references during confirmed accumulation and referencing chase guard to the current valid reference, not the original freeze.
- F8c: Not established by sealed evidence; however, any remaining veto-capable post-mandatory authority must be fully telemetried as named predicates if retained.

Final-probe note:
- Identified blocker on 2025-05-18: M5_liquidity

Updated R15 acceptance criteria:
- SANAM: ['The 2025-05-08 through 2025-05-21 window produces at least one confirmed entry event.', 'Width-rule dominance in the owner window drops below one-quarter of blocking rows.', 'The 2025-05-18 session specifically must produce BREAKOUT_CONFIRMED or an explicit, fully-telemetried veto naming its blocking term.', 'No stale original-freeze chase guard may block a day where the current valid reference has advanced through confirmed accumulation.']
- TIJARA: ['PHASE_PROGRESSED_NO_CANDIDATE share over 2024-2026 falls below one-quarter of trading days.', 'At least one owner-window breakout cluster produces BREAKOUT_CONFIRMED or DEFERRED_INTENT instead of persistent BREAKOUT_WATCH stagnation.', 'At least one 2025 high-volume cluster must operate with an explicit valid base reference rather than unresolved M1 disarm.']

R14-B and R15 remain NOT AUTHORIZED.
