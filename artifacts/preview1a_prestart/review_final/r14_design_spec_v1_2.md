# R14 Design Spec v1.2

Supersedes: R14 Design Spec v1.1

Amendment focus:
- Encode owner ruling CHASE_POLICY = TOLERANT_WITH_ADVISORY.
- Entry beyond the chase reference is permitted when flow confirmation holds; extended entries emit advisory telemetry only.
- Encode the M5/F9 result: current and arriving liquidity must be weighted directly; trailing baseline is context, never sole veto authority.

New/strengthened design principles:
- Chase policy is tolerant with advisory, not hard rejection, when confirmation core is satisfied.
- Advisory telemetry records extension_pct versus the current valid reference and references a named advisory-band parameter.
- Execution-liquidity assessment must combine current-day liquidity, short-horizon liquidity trend, and trailing context; trailing context cannot be sole veto authority.

State-machine additions:
- confirmation terms: ['CONFIRM_FLOW_CORE_OK', 'CONFIRM_STRUCTURE_OK', 'CONFIRM_RELATIVE_VOLUME_CONTEXT_OK', 'CONFIRM_CHASE_GUARD_OK', 'CONFIRM_LIQUIDITY_OK', 'CHASE_EXTENSION_ADVISORY_ONLY', 'CURRENT_DAY_LIQUIDITY_OK', 'LIQUIDITY_CONTEXT_OK']

Telemetry additions:
- daily_term_row: ['symbol', 'trade_date', 'segment_id', 'phase_before', 'phase_after', 'readiness_state', 'base_reference_id', 'intent_id', 'predicate_namespace', 'predicate_name', 'predicate_value', 'predicate_threshold_parameter', 'predicate_pass', 'recoverability_state', 'recoverability_reason', 'source_payload_fields', 'base_reference_version', 'base_reference_origin', 'base_reference_current_flag', 'extension_pct_vs_current_valid_reference', 'chase_advisory_flag', 'current_day_value_kwd', 'trailing_liquidity_context_value']
- execution_outcome_row: ['symbol', 'trade_date', 'candidate_intent_state', 'execution_state', 'veto_plane', 'veto_reason', 'opened_trade_flag', 'trade_id', 'chase_advisory_emitted', 'chase_advisory_extension_pct']

Updated finding-response map:
- F9: Solved by treating current and arriving liquidity as decisive execution evidence, with trailing baseline used as context rather than sole veto authority.

M5 forensic note:
- CONFIRMED: Trailing-window liquidity baseline lagged breakout-day liquidity and acted as sole veto on at least one high-volume Set A day.

Updated R15 acceptance criteria:
- SANAM: ['The 2025-05-08 through 2025-05-21 window produces at least one confirmed entry event.', 'Width-rule dominance in the owner window drops below one-quarter of blocking rows.', 'The 2025-05-18 session specifically must produce BREAKOUT_CONFIRMED or an explicit, fully-telemetried veto naming its blocking term.', 'No stale original-freeze chase guard may block a day where the current valid reference has advanced through confirmed accumulation.', 'The 2025-05-18 session must produce a confirmed entry across all layers.', 'No Set A high-volume day may be blocked by trailing-liquidity alone when same-day value exceeds the execution-size parameter.']
- BPCC: ['The 2025-04-22 near-miss cluster yields a candidate intent or confirmed entry without requiring a same-bar multiple shock.', 'M2-only blockade frequency materially declines in the owner window.', 'Extended entries beyond the current valid reference must emit advisory telemetry rather than hard rejection when flow confirmation holds.']

R14-B and R15 remain NOT AUTHORIZED.
