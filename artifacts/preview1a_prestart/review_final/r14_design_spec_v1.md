# R14 Design Spec v1

Authorization status:
- R14-A: AUTHORIZED (design spec only)
- R14-B: NOT AUTHORIZED
- R15: NOT AUTHORIZED

Design constraints:
- Zero engine or scanner code changes in this batch.
- No numeric threshold values in the design; every threshold-bearing rule is represented as a named config parameter to be frozen at the R14-B gate.
- Set B remains excluded from any exposure decisions until R14-B authorization.

Architecture blueprint:
- Skeleton: stateful lifecycle and full daily predicate telemetry.
- Confirmation core: accumulation-window flow confirmation using already-computed flow evidence and breakout structure.
- Base module: adaptive, volatility-regime-aware base geometry.
- Avoid plane: preserved in authority.
- Warmup module: explicit readiness states with fallback behavior.

Module boundaries and interfaces:
- DataSurfaceAdapter: Provide day-normalized input payloads and segment-aware readiness context.
  inputs=['ohlcv_day', 'indicator_day', 'segment_context', 'calendar_context']
  outputs=['normalized_day_payload', 'readiness_context']
- WarmupReadinessEngine: Determine readiness state, warmup fallback, and reset semantics for new listings and segment restarts.
  inputs=['normalized_day_payload', 'coverage_history', 'segment_restart_flag']
  outputs=['readiness_state', 'readiness_reason', 'readiness_transition_event']
- AdaptiveBaseGeometry: Detect, freeze, ratchet, invalidate, and retire bases using regime-aware geometry.
  inputs=['normalized_day_payload', 'readiness_state', 'price_history_window', 'volatility_regime_state']
  outputs=['base_state', 'base_transition_terms', 'base_reference']
- FlowConfirmationEngine: Score confirmation using accumulation-window flow evidence plus breakout structure.
  inputs=['normalized_day_payload', 'base_reference', 'flow_history_window', 'structure_terms']
  outputs=['confirmation_state', 'confirmation_terms', 'candidate_intent']
- LifecycleIntentRouter: Persist candidate intent, survive delayed volume arrival, and hand off to risk/capacity layer.
  inputs=['candidate_intent', 'base_state', 'confirmation_state', 'risk_budget_state']
  outputs=['execution_intent', 'deferred_intent', 'veto_record']
- AvoidAuthorityPlane: Retain current avoid authority semantics as a veto plane on top of the architecture.
  inputs=['normalized_day_payload', 'trend_state']
  outputs=['avoid_state', 'avoid_veto']
- PredicateTelemetryLedger: Persist every predicate term, every day, for every symbol.
  inputs=['all_module_terms', 'state_transitions', 'execution_outcomes']
  outputs=['daily_term_row', 'state_snapshot_row', 'audit_row']

State machine:
- States: ['READINESS_PENDING', 'READINESS_LIMITED', 'NEUTRAL', 'BASE_FORMING', 'ACCUMULATION', 'BREAKOUT_WATCH', 'BREAKOUT_CONFIRMED', 'MARKUP', 'DISTRIBUTION_WARNING', 'EXIT', 'AVOID', 'DEFERRED_INTENT']
- Transition principles:
  - All transitions are gated by named predicates only.
  - All threshold-bearing terms are represented by named configuration parameters, not literal values.
  - Readiness states are explicit and segment-aware.
  - Base references persist as first-class state objects.
  - Candidate intent may persist independently of immediate execution eligibility.
- Named predicate namespaces:
  - warmup: ['READINESS_LONG_LOOKBACK_READY', 'READINESS_SEGMENT_RESTART_READY', 'READINESS_FALLBACK_ELIGIBLE']
  - base: ['BASE_GEOMETRY_WIDTH_OK', 'BASE_DWELL_OK', 'BASE_RANGE_INCLUSION_OK', 'BASE_VOLATILITY_REGIME_OK']
  - accumulation: ['FLOW_OBV_SLOPE_OK', 'FLOW_ANV_SLOPE_OK', 'FLOW_ACCUMULATION_DIVERGENCE_OK', 'ACCUMULATION_CONTEXT_OK']
  - watch: ['WATCH_NEAR_BASE_OK', 'WATCH_FLOW_PERSISTENCE_OK', 'WATCH_STRUCTURE_OK']
  - confirmation: ['CONFIRM_FLOW_CORE_OK', 'CONFIRM_STRUCTURE_OK', 'CONFIRM_RELATIVE_VOLUME_CONTEXT_OK', 'CONFIRM_CHASE_GUARD_OK', 'CONFIRM_LIQUIDITY_OK']
  - lifecycle: ['BASE_REFERENCE_PRESENT', 'BASE_REFERENCE_VALID', 'DEFERRED_INTENT_ACTIVE', 'DEFERRED_INTENT_EXPIRY_OK']
  - avoid: ['AVOID_CONDITION_ACTIVE']

Telemetry schema:
- daily_term_row: ['symbol', 'trade_date', 'segment_id', 'phase_before', 'phase_after', 'readiness_state', 'base_reference_id', 'intent_id', 'predicate_namespace', 'predicate_name', 'predicate_value', 'predicate_threshold_parameter', 'predicate_pass', 'recoverability_state', 'recoverability_reason', 'source_payload_fields']
- daily_state_snapshot: ['symbol', 'trade_date', 'readiness_state', 'phase_state', 'base_reference_snapshot', 'intent_snapshot', 'avoid_state', 'risk_budget_state']
- execution_outcome_row: ['symbol', 'trade_date', 'candidate_intent_state', 'execution_state', 'veto_plane', 'veto_reason', 'opened_trade_flag', 'trade_id']

Finding response map:
- F1: Solved by FlowConfirmationEngine replacing same-bar multiple as sole confirmation core.
- F2: Solved by AdaptiveBaseGeometry replacing fixed-width geometry with regime-aware geometry.
- F3: Addressed by LifecycleIntentRouter and persistent base references; to be validated in R15.
- F4: Solved by WarmupReadinessEngine with explicit readiness states and fallback.
- F5: Preserved by AvoidAuthorityPlane retaining avoid veto semantics.
- F6: Explained by moving emphasis from veto gates to candidate-intent and confirmation persistence.
- F7: Solved by PredicateTelemetryLedger persisting every term every day.
- F8: Tested by combining adaptive base geometry with persistent base references and deferred intent.

R15 acceptance criteria:
- TIJARA:
  - PHASE_PROGRESSED_NO_CANDIDATE share over 2024-2026 falls below one-quarter of trading days.
  - At least one owner-window breakout cluster produces BREAKOUT_CONFIRMED or DEFERRED_INTENT instead of persistent BREAKOUT_WATCH stagnation.
- SANAM:
  - The 2025-05-08 through 2025-05-21 window produces at least one confirmed entry event.
  - Width-rule dominance in the owner window drops below one-quarter of blocking rows.
- BPCC:
  - The 2025-04-22 near-miss cluster yields a candidate intent or confirmed entry without requiring a same-bar multiple shock.
  - M2-only blockade frequency materially declines in the owner window.
- ZAIN:
  - Owner-window no-candidate share falls below one-third of trading days.
  - At least one breakout-above-owner-threshold cluster yields BREAKOUT_CONFIRMED or DEFERRED_INTENT.
- MABANEE:
  - The decline interval remains avoided with zero long entries during the avoid-dominant regime.
  - Avoid veto authority remains sufficient to block downtrend participation.

Citations:
- Findings artifacts:
  - artifacts/preview1a_prestart/review_final/r13_findings_of_record_v1_1.md
  - artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json
  - artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json
- Code refs:
  - app/services/eagle_eye/scanner_service.py#L713
  - app/services/eagle_eye/scanner_service.py#L718
  - app/services/eagle_eye/scanner_service.py#L859
  - app/services/eagle_eye/scanner_service.py#L108
  - app/services/eagle_eye/scanner_service.py#L288
  - app/services/eagle_eye/scanner_service.py#L770

R14-B and R15 remain NOT AUTHORIZED.
