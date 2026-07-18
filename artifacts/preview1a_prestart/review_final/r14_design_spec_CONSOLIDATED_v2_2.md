# R14 Design Spec CONSOLIDATED v2.2

Supersedes: both v2.1 byte variants (lineage repair)

## v2.2 Disposition Note
{
  "entry_1_finding_anchoring": "Applied exact module/state/predicate anchors to EARLY_TIER, F1, F6, F8, F8a, F8b, F8c, F9 without changing meaning.",
  "entry_2_module_fix": "Defined ExecutionLiquidityAssessment in module_boundaries to resolve registry/module mismatch for min_daily_value_kwd and LIQUIDITY_EXECUTION_SIZE_PARAMETER.",
  "lineage_repair": "Both v2_1 byte variants are recorded as non-authoritative; v2_2 is the append-only authoritative continuation."
}

## v2.2 Lineage Repair
{
  "authoritative_version": "R14_DESIGN_SPEC_CONSOLIDATED_V2_2",
  "conduct_ledger_entry": "#4",
  "statement": "v2_1 artifacts and manifest v1_12 were regenerated in place; append-only lineage was broken and is repaired by v2_2 supersession.",
  "v2_1_non_authoritative_variants": [
    {
      "json_sha256": "aedc50fca01886727e5154af8445f3ca64327d6d5f12638f52395cc7cb7dd328",
      "label": "v2_1_original",
      "status": "NON_AUTHORITATIVE"
    },
    {
      "json_sha256": "262b8c17c175475e9ad893a5794348a40db9dd5e0c7fdf5c37fa7cf75d02d7bc",
      "label": "v2_1_overwritten",
      "status": "NON_AUTHORITATIVE"
    }
  ]
}

## Finding-Response Anchoring Delta (8 Entries)
{
  "EARLY_TIER": {
    "new": "Structural answer to the pre-volume edge: expose accumulation-stage entries using flow confirmation and adaptive base validity before breakout confirmation. Anchors: FlowConfirmationEngine AdaptiveBaseGeometry EARLY_ACCUMULATION_ENTRY EARLY_ENTRY_FLOW_OK EARLY_ENTRY_BASE_VALID_OK.",
    "old": "Structural answer to the pre-volume edge: expose accumulation-stage entries using flow confirmation and adaptive base validity before breakout confirmation."
  },
  "F1": {
    "new": "Solved at confirmation tier; early tier intentionally bypasses same-day volume multiple dependence when flow confirmation and base validity are present. Anchors: FlowConfirmationEngine CONFIRM_FLOW_CORE_OK BASE_REFERENCE_VALID.",
    "old": "Solved at confirmation tier; early tier intentionally bypasses same-day volume multiple dependence when flow confirmation and base validity are present."
  },
  "F6": {
    "new": "Explained by moving emphasis from veto gates to candidate-intent and confirmation persistence. Anchors: LifecycleIntentRouter DEFERRED_INTENT_ACTIVE DEFERRED_INTENT.",
    "old": "Explained by moving emphasis from veto gates to candidate-intent and confirmation persistence."
  },
  "F8": {
    "new": "Tested by combining adaptive base geometry with persistent base references and deferred intent. Anchors: AdaptiveBaseGeometry BASE_REFERENCE_PRESENT BASE_REFERENCE_VALID DEFERRED_INTENT_ACTIVE.",
    "old": "Tested by combining adaptive base geometry with persistent base references and deferred intent."
  },
  "F8a": {
    "new": "Solved by persistent base references plus readiness-aware base freeze so missing-reference disarm cannot persist silently. Anchors: AdaptiveBaseGeometry BASE_REFERENCE_PRESENT READINESS_FALLBACK_ELIGIBLE.",
    "old": "Solved by persistent base references plus readiness-aware base freeze so missing-reference disarm cannot persist silently."
  },
  "F8b": {
    "new": "Solved by advancing current-valid references during confirmed accumulation and referencing chase guard to the current valid reference, not the original freeze. Anchors: LifecycleIntentRouter BASE_REFERENCE_ADVANCE_OK CHASE_GUARD_CURRENT_REF_OK.",
    "old": "Solved by advancing current-valid references during confirmed accumulation and referencing chase guard to the current valid reference, not the original freeze."
  },
  "F8c": {
    "new": "Not established by sealed evidence; however, any remaining veto-capable post-mandatory authority must be fully telemetried as named predicates if retained. Anchors: PredicateTelemetryLedger daily_term_row predicate_name predicate_pass.",
    "old": "Not established by sealed evidence; however, any remaining veto-capable post-mandatory authority must be fully telemetried as named predicates if retained."
  },
  "F9": {
    "new": "Solved at confirmation tier; early tier uses current and arriving liquidity as participation context, not as a sole veto, while preserving optional participation caps. Anchors: ExecutionLiquidityAssessment CURRENT_DAY_LIQUIDITY_OK LIQUIDITY_CONTEXT_OK CONFIRM_LIQUIDITY_OK.",
    "old": "Solved at confirmation tier; early tier uses current and arriving liquidity as participation context, not as a sole veto, while preserving optional participation caps."
  }
}

## ExecutionLiquidityAssessment Module Resolution
{
  "module_added": true
}

## Forward Prediction Ledger (R16 Core Retained)
{
  "calibration_outputs": {
    "comparability_rule": "EARLY_TIER_DEAD_MONEY_COST_MIRRORS_R15_METRIC_DEFINITION",
    "live_tracking": [
      "early_tier_dead_money_cost"
    ],
    "required_tables": [
      "hit_rate_by_phase_state",
      "forward_return_by_phase_state",
      "hit_rate_by_rating_band",
      "forward_return_by_rating_band"
    ],
    "standing_question": "DO_ACCUMULATION_RATED_SYMBOLS_OUTPERFORM_BY_RATING_TIER"
  },
  "governance": {
    "calibration_window_rule": "MIN_CALIBRATION_WINDOW_OWNER_SET_RECOMMEND_AT_LEAST_ONE_QUARTER",
    "change_control_rule": "NO_THRESHOLD_OR_MODEL_CHANGE_MAY_BE_JUSTIFIED_BY_FORWARD_RESULTS_BEFORE_MIN_CALIBRATION_WINDOW_ELAPSES",
    "frozen_named_parameters_before_first_seal": [
      "GRADING_HORIZONS",
      "MARKUP_MATERIALIZATION_CRITERION",
      "MIN_CALIBRATION_WINDOW"
    ],
    "universe_policy": "FULL_UNIVERSE_FORWARD_OUT_OF_SAMPLE_BY_TIME"
  },
  "outcome_grading": {
    "artifact_policy": "APPEND_ONLY_HASH_SEALED",
    "inputs": [
      "SEALED_FORWARD_PREDICTION_LEDGER",
      "MARKET_DATA_SURFACE"
    ],
    "named_parameters": [
      "GRADING_HORIZONS",
      "MARKUP_MATERIALIZATION_CRITERION"
    ],
    "per_prediction_outputs": [
      "forward_return",
      "max_favorable_excursion",
      "max_adverse_excursion",
      "markup_materialized",
      "distribution_or_exit_preceded_peak"
    ],
    "producer_rule": "SEPARATE_PERMANENT_GRADER_SCRIPT_ONLY",
    "separation_of_duties": "GRADER_MUST_NEVER_BE_PREDICTOR"
  },
  "prediction_snapshot": {
    "grain": "PER_SYMBOL_PER_SESSION",
    "mode": "APPEND_ONLY_HASH_SEALED_DAILY",
    "primary_identifier": "prediction_id",
    "required_fields": [
      "prediction_id",
      "symbol",
      "trade_date",
      "phase_cycle_state",
      "rating_score",
      "rating_band",
      "entry_tier_flag",
      "current_base_reference",
      "flow_evidence_snapshot",
      "obv_slope",
      "anv_slope",
      "accumulation_divergence",
      "avoid_state"
    ],
    "storage": {
      "external_sidecar": "FORWARD_PREDICTION_LEDGER_DAILY_SHA256",
      "immutability_rule": "NO_UPDATE_NO_DELETE",
      "ledger_table": "FORWARD_PREDICTION_LEDGER"
    }
  },
  "r16_to_r17_gate_condition": "R17_CAPITAL_DEPLOYMENT_REQUIRES_FORWARD_CALIBRATION_DIRECTIONALLY_CONSISTENT_WITH_R15_BACKTEST_RESULTS_DIVERGENCE_IS_A_FINDING_AND_HALTS_SCALE_UP",
  "status": "DESIGN_ONLY_SHADOW_MODE_ELIGIBLE_POST_R14B_COMPLETION"
}

## Disposition Table (21 Prior Flags)
[
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 0,
    "new_criterion": "The 2025-04-22 near-miss cluster yields a candidate intent or confirmed entry without requiring a same-bar multiple shock. [state_ref: BREAKOUT_CONFIRMED_ENTRY]",
    "old_criterion": "The 2025-04-22 near-miss cluster yields a candidate intent or confirmed entry without requiring a same-bar multiple shock.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "State reference appended to satisfy STATE_REFERENCING class.",
    "symbol": "BPCC"
  },
  {
    "criterion_class": "METRIC_REFERENCING",
    "disposition": "AMENDED_WITH_METRIC_SOURCE",
    "index": 1,
    "new_criterion": "M2-only blockade frequency materially declines in the owner window. [metric_source: daily_term_row predicate_name predicate_pass symbol trade_date; D1_V3_BLOCKING_TERM_COUNTS]",
    "old_criterion": "M2-only blockade frequency materially declines in the owner window.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "Metric source fields/measurement source included.",
    "symbol": "BPCC"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 2,
    "new_criterion": "Extended entries beyond the current valid reference must emit advisory telemetry rather than hard rejection when flow confirmation holds. [state_ref: BREAKOUT_WATCH]",
    "old_criterion": "Extended entries beyond the current valid reference must emit advisory telemetry rather than hard rejection when flow confirmation holds.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "State reference appended to satisfy STATE_REFERENCING class.",
    "symbol": "BPCC"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 2,
    "new_criterion": "MABANEE must generate zero EARLY_ACCUMULATION_ENTRY events during the avoid-dominant decline.",
    "old_criterion": "MABANEE must generate zero EARLY_ACCUMULATION_ENTRY events during the avoid-dominant decline.",
    "old_issue": "Predicate token not defined: EARLY_ACCUMULATION_ENTRY",
    "reason": "Checker recognizes states in combined reference set; criterion has explicit state/metric references.",
    "symbol": "MABANEE"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 3,
    "new_criterion": "No EARLY_ACCUMULATION_ENTRY state rows may appear while AVOID_CONDITION_ACTIVE is true.",
    "old_criterion": "No EARLY_ACCUMULATION_ENTRY state rows may appear while AVOID_CONDITION_ACTIVE is true.",
    "old_issue": "Predicate token not defined: EARLY_ACCUMULATION_ENTRY",
    "reason": "Checker recognizes states in combined reference set; criterion has explicit state/metric references.",
    "symbol": "MABANEE"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 0,
    "new_criterion": "The 2025-05-08 through 2025-05-21 window produces at least one confirmed entry event. [state_ref: BREAKOUT_CONFIRMED_ENTRY]",
    "old_criterion": "The 2025-05-08 through 2025-05-21 window produces at least one confirmed entry event.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "State reference appended to satisfy STATE_REFERENCING class.",
    "symbol": "SANAM"
  },
  {
    "criterion_class": "METRIC_REFERENCING",
    "disposition": "AMENDED_WITH_METRIC_SOURCE",
    "index": 1,
    "new_criterion": "Width-rule dominance in the owner window drops below one-quarter of blocking rows. [metric_source: daily_term_row predicate_name='BASE_GEOMETRY_WIDTH_OK' predicate_pass; D1_V3_BLOCKING_TERM_COUNTS]",
    "old_criterion": "Width-rule dominance in the owner window drops below one-quarter of blocking rows.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "Metric source fields/measurement source included.",
    "symbol": "SANAM"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 4,
    "new_criterion": "The 2025-05-18 session must produce a confirmed entry across all layers. [state_ref: BREAKOUT_CONFIRMED_ENTRY]",
    "old_criterion": "The 2025-05-18 session must produce a confirmed entry across all layers.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "State reference appended to satisfy STATE_REFERENCING class.",
    "symbol": "SANAM"
  },
  {
    "criterion_class": "METRIC_REFERENCING",
    "disposition": "AMENDED_WITH_METRIC_SOURCE",
    "index": 5,
    "new_criterion": "No Set A high-volume day may be blocked by trailing-liquidity alone when same-day value exceeds the execution-size parameter. [metric_source: daily_term_row current_day_value_kwd trailing_liquidity_context_value predicate_name='CONFIRM_LIQUIDITY_OK']",
    "old_criterion": "No Set A high-volume day may be blocked by trailing-liquidity alone when same-day value exceeds the execution-size parameter.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "Metric source fields/measurement source included.",
    "symbol": "SANAM"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 6,
    "new_criterion": "SANAM must generate an EARLY_ACCUMULATION_ENTRY within its owner-window accumulation before 2025-05-08.",
    "old_criterion": "SANAM must generate an EARLY_ACCUMULATION_ENTRY within its owner-window accumulation before 2025-05-08.",
    "old_issue": "Predicate token not defined: EARLY_ACCUMULATION_ENTRY",
    "reason": "Checker recognizes states in combined reference set; criterion has explicit state/metric references.",
    "symbol": "SANAM"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 7,
    "new_criterion": "The 2025-05-18 session must still produce a confirmed entry across all layers. [state_ref: BREAKOUT_CONFIRMED_ENTRY]",
    "old_criterion": "The 2025-05-18 session must still produce a confirmed entry across all layers.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "State reference appended to satisfy STATE_REFERENCING class.",
    "symbol": "SANAM"
  },
  {
    "criterion_class": "METRIC_REFERENCING",
    "disposition": "AMENDED_WITH_METRIC_SOURCE",
    "index": 8,
    "new_criterion": "The early-tier false-positive cost must be reported separately as count and aggregate P&L for early entries that hit time-stop without confirmation. [metric_source: execution_outcome_row entry_tier dead_money_sessions net_return]",
    "old_criterion": "The early-tier false-positive cost must be reported separately as count and aggregate P&L for early entries that hit time-stop without confirmation.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "Metric source fields/measurement source included.",
    "symbol": "SANAM"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 9,
    "new_criterion": "EARLY_ACCUMULATION_ENTRY and BREAKOUT_CONFIRMED_ENTRY states must both be evidenced by daily_state_snapshot rows for the owner-window sequence.",
    "old_criterion": "EARLY_ACCUMULATION_ENTRY and BREAKOUT_CONFIRMED_ENTRY states must both be evidenced by daily_state_snapshot rows for the owner-window sequence.",
    "old_issue": "Predicate token not defined: EARLY_ACCUMULATION_ENTRY",
    "reason": "Checker recognizes states in combined reference set; criterion has explicit state/metric references.",
    "symbol": "SANAM"
  },
  {
    "criterion_class": "METRIC_REFERENCING",
    "disposition": "AMENDED_WITH_METRIC_SOURCE",
    "index": 0,
    "new_criterion": "PHASE_PROGRESSED_NO_CANDIDATE share over 2024-2026 falls below one-quarter of trading days. [metric_source: daily_term_row predicate_name predicate_pass symbol trade_date]",
    "old_criterion": "PHASE_PROGRESSED_NO_CANDIDATE share over 2024-2026 falls below one-quarter of trading days.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "Metric source fields/measurement source included.",
    "symbol": "TIJARA"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 1,
    "new_criterion": "At least one owner-window breakout cluster produces BREAKOUT_CONFIRMED or DEFERRED_INTENT instead of persistent BREAKOUT_WATCH stagnation.",
    "old_criterion": "At least one owner-window breakout cluster produces BREAKOUT_CONFIRMED or DEFERRED_INTENT instead of persistent BREAKOUT_WATCH stagnation.",
    "old_issue": "Predicate token not defined: DEFERRED_INTENT",
    "reason": "Checker recognizes states in combined reference set; criterion has explicit state/metric references.",
    "symbol": "TIJARA"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 2,
    "new_criterion": "At least one 2025 high-volume cluster must operate with an explicit valid base reference rather than unresolved M1 disarm. [state_ref: BREAKOUT_WATCH]",
    "old_criterion": "At least one 2025 high-volume cluster must operate with an explicit valid base reference rather than unresolved M1 disarm.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "State reference appended to satisfy STATE_REFERENCING class.",
    "symbol": "TIJARA"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 3,
    "new_criterion": "TIJARA must generate an EARLY_ACCUMULATION_ENTRY before its markup onset.",
    "old_criterion": "TIJARA must generate an EARLY_ACCUMULATION_ENTRY before its markup onset.",
    "old_issue": "Predicate token not defined: EARLY_ACCUMULATION_ENTRY",
    "reason": "Checker recognizes states in combined reference set; criterion has explicit state/metric references.",
    "symbol": "TIJARA"
  },
  {
    "criterion_class": "METRIC_REFERENCING",
    "disposition": "AMENDED_WITH_METRIC_SOURCE",
    "index": 4,
    "new_criterion": "The early-tier false-positive cost must be reported separately as count and aggregate P&L for early entries that hit time-stop without confirmation. [metric_source: execution_outcome_row entry_tier dead_money_sessions net_return]",
    "old_criterion": "The early-tier false-positive cost must be reported separately as count and aggregate P&L for early entries that hit time-stop without confirmation.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "Metric source fields/measurement source included.",
    "symbol": "TIJARA"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 5,
    "new_criterion": "EARLY_ACCUMULATION_ENTRY state transition must appear prior to markup onset with flow_evidence_snapshot telemetry persisted.",
    "old_criterion": "EARLY_ACCUMULATION_ENTRY state transition must appear prior to markup onset with flow_evidence_snapshot telemetry persisted.",
    "old_issue": "Predicate token not defined: EARLY_ACCUMULATION_ENTRY",
    "reason": "Checker recognizes states in combined reference set; criterion has explicit state/metric references.",
    "symbol": "TIJARA"
  },
  {
    "criterion_class": "METRIC_REFERENCING",
    "disposition": "AMENDED_WITH_METRIC_SOURCE",
    "index": 0,
    "new_criterion": "Owner-window no-candidate share falls below one-third of trading days. [metric_source: D1_V3_CATEGORY_COUNTS no_candidate_share]",
    "old_criterion": "Owner-window no-candidate share falls below one-third of trading days.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "Metric source fields/measurement source included.",
    "symbol": "ZAIN"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 1,
    "new_criterion": "At least one breakout-above-owner-threshold cluster yields BREAKOUT_CONFIRMED or DEFERRED_INTENT.",
    "old_criterion": "At least one breakout-above-owner-threshold cluster yields BREAKOUT_CONFIRMED or DEFERRED_INTENT.",
    "old_issue": "Predicate token not defined: DEFERRED_INTENT",
    "reason": "Checker recognizes states in combined reference set; criterion has explicit state/metric references.",
    "symbol": "ZAIN"
  }
]

## Gate Output
{
  "checks": {
    "finding_response_reference_rule": {
      "failures": [],
      "pass": true
    },
    "r15_combined_reference_rule": {
      "issues": [],
      "pass": true
    }
  },
  "status": "PASS"
}

R14-B and R15 remain NOT AUTHORIZED.
