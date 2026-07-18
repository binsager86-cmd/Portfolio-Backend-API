# R14 Design Spec CONSOLIDATED v2.1

Supersedes: R14 Design Spec CONSOLIDATED v2

Checker repair summary:
- Finding-response rule: module-name OR exact underscore state/predicate token reference.
- R15 reference set: states ∪ predicates ∪ telemetry fields.
- Criterion classification enforced: STATE_REFERENCING or METRIC_REFERENCING.

## Amended R15 Criteria
{
  "BPCC": [
    "The 2025-04-22 near-miss cluster yields a candidate intent or confirmed entry without requiring a same-bar multiple shock. [state_ref: BREAKOUT_CONFIRMED_ENTRY]",
    "M2-only blockade frequency materially declines in the owner window. [metric_source: daily_term_row predicate_name predicate_pass symbol trade_date; D1_V3_BLOCKING_TERM_COUNTS]",
    "Extended entries beyond the current valid reference must emit advisory telemetry rather than hard rejection when flow confirmation holds. [state_ref: BREAKOUT_WATCH]"
  ],
  "MABANEE": [
    "The decline interval remains avoided with zero long entries during the avoid-dominant regime. [state_ref: AVOID]",
    "Avoid veto authority remains sufficient to block downtrend participation. [state_ref: AVOID]",
    "MABANEE must generate zero EARLY_ACCUMULATION_ENTRY events during the avoid-dominant decline.",
    "No EARLY_ACCUMULATION_ENTRY state rows may appear while AVOID_CONDITION_ACTIVE is true."
  ],
  "SANAM": [
    "The 2025-05-08 through 2025-05-21 window produces at least one confirmed entry event. [state_ref: BREAKOUT_CONFIRMED_ENTRY]",
    "Width-rule dominance in the owner window drops below one-quarter of blocking rows. [metric_source: daily_term_row predicate_name='BASE_GEOMETRY_WIDTH_OK' predicate_pass; D1_V3_BLOCKING_TERM_COUNTS]",
    "The 2025-05-18 session specifically must produce BREAKOUT_CONFIRMED or an explicit, fully-telemetried veto naming its blocking term.",
    "No stale original-freeze chase guard may block a day where the current valid reference has advanced through confirmed accumulation. [state_ref: BREAKOUT_WATCH]",
    "The 2025-05-18 session must produce a confirmed entry across all layers. [state_ref: BREAKOUT_CONFIRMED_ENTRY]",
    "No Set A high-volume day may be blocked by trailing-liquidity alone when same-day value exceeds the execution-size parameter. [metric_source: daily_term_row current_day_value_kwd trailing_liquidity_context_value predicate_name='CONFIRM_LIQUIDITY_OK']",
    "SANAM must generate an EARLY_ACCUMULATION_ENTRY within its owner-window accumulation before 2025-05-08.",
    "The 2025-05-18 session must still produce a confirmed entry across all layers. [state_ref: BREAKOUT_CONFIRMED_ENTRY]",
    "The early-tier false-positive cost must be reported separately as count and aggregate P&L for early entries that hit time-stop without confirmation. [metric_source: execution_outcome_row entry_tier dead_money_sessions net_return]",
    "EARLY_ACCUMULATION_ENTRY and BREAKOUT_CONFIRMED_ENTRY states must both be evidenced by daily_state_snapshot rows for the owner-window sequence.",
    "dead_money_sessions telemetry must be present for all early-tier positions and reported in execution_outcome_row."
  ],
  "TIJARA": [
    "PHASE_PROGRESSED_NO_CANDIDATE share over 2024-2026 falls below one-quarter of trading days. [metric_source: daily_term_row predicate_name predicate_pass symbol trade_date]",
    "At least one owner-window breakout cluster produces BREAKOUT_CONFIRMED or DEFERRED_INTENT instead of persistent BREAKOUT_WATCH stagnation.",
    "At least one 2025 high-volume cluster must operate with an explicit valid base reference rather than unresolved M1 disarm. [state_ref: BREAKOUT_WATCH]",
    "TIJARA must generate an EARLY_ACCUMULATION_ENTRY before its markup onset.",
    "The early-tier false-positive cost must be reported separately as count and aggregate P&L for early entries that hit time-stop without confirmation. [metric_source: execution_outcome_row entry_tier dead_money_sessions net_return]",
    "EARLY_ACCUMULATION_ENTRY state transition must appear prior to markup onset with flow_evidence_snapshot telemetry persisted."
  ],
  "ZAIN": [
    "Owner-window no-candidate share falls below one-third of trading days. [metric_source: D1_V3_CATEGORY_COUNTS no_candidate_share]",
    "At least one breakout-above-owner-threshold cluster yields BREAKOUT_CONFIRMED or DEFERRED_INTENT."
  ]
}

## Criterion Classification
{
  "BPCC": [
    "STATE_REFERENCING",
    "METRIC_REFERENCING",
    "STATE_REFERENCING"
  ],
  "MABANEE": [
    "STATE_REFERENCING",
    "STATE_REFERENCING",
    "STATE_REFERENCING",
    "STATE_REFERENCING"
  ],
  "SANAM": [
    "STATE_REFERENCING",
    "METRIC_REFERENCING",
    "STATE_REFERENCING",
    "STATE_REFERENCING",
    "STATE_REFERENCING",
    "METRIC_REFERENCING",
    "STATE_REFERENCING",
    "STATE_REFERENCING",
    "METRIC_REFERENCING",
    "STATE_REFERENCING",
    "STATE_REFERENCING"
  ],
  "TIJARA": [
    "METRIC_REFERENCING",
    "STATE_REFERENCING",
    "STATE_REFERENCING",
    "STATE_REFERENCING",
    "METRIC_REFERENCING",
    "STATE_REFERENCING"
  ],
  "ZAIN": [
    "METRIC_REFERENCING",
    "STATE_REFERENCING"
  ]
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
    "reason": "Metric source fields/measurement source appended per directive.",
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
    "reason": "Checker now recognizes states in combined reference set; criterion also explicitly state-referenced.",
    "symbol": "MABANEE"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 3,
    "new_criterion": "No EARLY_ACCUMULATION_ENTRY state rows may appear while AVOID_CONDITION_ACTIVE is true.",
    "old_criterion": "No EARLY_ACCUMULATION_ENTRY state rows may appear while AVOID_CONDITION_ACTIVE is true.",
    "old_issue": "Predicate token not defined: EARLY_ACCUMULATION_ENTRY",
    "reason": "Checker now recognizes states in combined reference set; criterion also explicitly state-referenced.",
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
    "reason": "Metric source fields/measurement source appended per directive.",
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
    "reason": "Metric source fields/measurement source appended per directive.",
    "symbol": "SANAM"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 6,
    "new_criterion": "SANAM must generate an EARLY_ACCUMULATION_ENTRY within its owner-window accumulation before 2025-05-08.",
    "old_criterion": "SANAM must generate an EARLY_ACCUMULATION_ENTRY within its owner-window accumulation before 2025-05-08.",
    "old_issue": "Predicate token not defined: EARLY_ACCUMULATION_ENTRY",
    "reason": "Checker now recognizes states in combined reference set; criterion also explicitly state-referenced.",
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
    "reason": "Metric source fields/measurement source appended per directive.",
    "symbol": "SANAM"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 9,
    "new_criterion": "EARLY_ACCUMULATION_ENTRY and BREAKOUT_CONFIRMED_ENTRY states must both be evidenced by daily_state_snapshot rows for the owner-window sequence.",
    "old_criterion": "EARLY_ACCUMULATION_ENTRY and BREAKOUT_CONFIRMED_ENTRY states must both be evidenced by daily_state_snapshot rows for the owner-window sequence.",
    "old_issue": "Predicate token not defined: EARLY_ACCUMULATION_ENTRY",
    "reason": "Checker now recognizes states in combined reference set; criterion also explicitly state-referenced.",
    "symbol": "SANAM"
  },
  {
    "criterion_class": "METRIC_REFERENCING",
    "disposition": "AMENDED_WITH_METRIC_SOURCE",
    "index": 0,
    "new_criterion": "PHASE_PROGRESSED_NO_CANDIDATE share over 2024-2026 falls below one-quarter of trading days. [metric_source: daily_term_row predicate_name predicate_pass symbol trade_date]",
    "old_criterion": "PHASE_PROGRESSED_NO_CANDIDATE share over 2024-2026 falls below one-quarter of trading days.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "Metric source fields/measurement source appended per directive.",
    "symbol": "TIJARA"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 1,
    "new_criterion": "At least one owner-window breakout cluster produces BREAKOUT_CONFIRMED or DEFERRED_INTENT instead of persistent BREAKOUT_WATCH stagnation.",
    "old_criterion": "At least one owner-window breakout cluster produces BREAKOUT_CONFIRMED or DEFERRED_INTENT instead of persistent BREAKOUT_WATCH stagnation.",
    "old_issue": "Predicate token not defined: DEFERRED_INTENT",
    "reason": "Checker now recognizes states in combined reference set; criterion also explicitly state-referenced.",
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
    "reason": "Checker now recognizes states in combined reference set; criterion also explicitly state-referenced.",
    "symbol": "TIJARA"
  },
  {
    "criterion_class": "METRIC_REFERENCING",
    "disposition": "AMENDED_WITH_METRIC_SOURCE",
    "index": 4,
    "new_criterion": "The early-tier false-positive cost must be reported separately as count and aggregate P&L for early entries that hit time-stop without confirmation. [metric_source: execution_outcome_row entry_tier dead_money_sessions net_return]",
    "old_criterion": "The early-tier false-positive cost must be reported separately as count and aggregate P&L for early entries that hit time-stop without confirmation.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "Metric source fields/measurement source appended per directive.",
    "symbol": "TIJARA"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 5,
    "new_criterion": "EARLY_ACCUMULATION_ENTRY state transition must appear prior to markup onset with flow_evidence_snapshot telemetry persisted.",
    "old_criterion": "EARLY_ACCUMULATION_ENTRY state transition must appear prior to markup onset with flow_evidence_snapshot telemetry persisted.",
    "old_issue": "Predicate token not defined: EARLY_ACCUMULATION_ENTRY",
    "reason": "Checker now recognizes states in combined reference set; criterion also explicitly state-referenced.",
    "symbol": "TIJARA"
  },
  {
    "criterion_class": "METRIC_REFERENCING",
    "disposition": "AMENDED_WITH_METRIC_SOURCE",
    "index": 0,
    "new_criterion": "Owner-window no-candidate share falls below one-third of trading days. [metric_source: D1_V3_CATEGORY_COUNTS no_candidate_share]",
    "old_criterion": "Owner-window no-candidate share falls below one-third of trading days.",
    "old_issue": "No explicit state/telemetry reference detected",
    "reason": "Metric source fields/measurement source appended per directive.",
    "symbol": "ZAIN"
  },
  {
    "criterion_class": "STATE_REFERENCING",
    "disposition": "RESOLVED_AS_STATE",
    "index": 1,
    "new_criterion": "At least one breakout-above-owner-threshold cluster yields BREAKOUT_CONFIRMED or DEFERRED_INTENT.",
    "old_criterion": "At least one breakout-above-owner-threshold cluster yields BREAKOUT_CONFIRMED or DEFERRED_INTENT.",
    "old_issue": "Predicate token not defined: DEFERRED_INTENT",
    "reason": "Checker now recognizes states in combined reference set; criterion also explicitly state-referenced.",
    "symbol": "ZAIN"
  }
]

## Forward Prediction Ledger (R16 Core)
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

## R14-B Readiness Checklist (R16 Gate Added)
{
  "governance_required_for_r14b": [
    "Issue new versioned baseline ID for implementation branch.",
    "Freeze semantics: no parameter mutation after R14-B freeze without explicit change request and reseal.",
    "Restate Set B quarantine during implementation and dry-run validation.",
    "Require permanent-script-only execution and manifest sealing for all producing runs."
  ],
  "owner_required_for_r14b": [
    "Ratify all pending named-parameter values in parameter registry.",
    "Ratify tier rule candidate definition (A or B).",
    "Ratify early-tier participation and time-stop policy values.",
    "Ratify chase advisory band semantics and escalation policy."
  ],
  "r15_gate_conditions": [
    "All acceptance criteria per symbol evaluated on sealed runtime surface.",
    "Early-tier false-positive cost reported separately (count and aggregate P&L).",
    "No criterion marked pass without corresponding state/telemetry evidence rows.",
    "Set B remains excluded until R15 gate completion and owner ratification."
  ],
  "r16_gate_conditions": [
    "R17 capital deployment requires forward calibration tables to be directionally consistent with R15 backtest results; divergence is a finding and halts scale-up."
  ]
}

## Repaired Gate Output
{
  "checks": {
    "finding_response_reference_rule": {
      "failures": [
        {
          "finding": "EARLY_TIER",
          "reason": "No module name and no exact underscore state/predicate token"
        },
        {
          "finding": "F1",
          "reason": "No module name and no exact underscore state/predicate token"
        },
        {
          "finding": "F6",
          "reason": "No module name and no exact underscore state/predicate token"
        },
        {
          "finding": "F8",
          "reason": "No module name and no exact underscore state/predicate token"
        },
        {
          "finding": "F8a",
          "reason": "No module name and no exact underscore state/predicate token"
        },
        {
          "finding": "F8b",
          "reason": "No module name and no exact underscore state/predicate token"
        },
        {
          "finding": "F8c",
          "reason": "No module name and no exact underscore state/predicate token"
        },
        {
          "finding": "F9",
          "reason": "No module name and no exact underscore state/predicate token"
        }
      ],
      "pass": false
    },
    "r15_combined_reference_rule": {
      "issues": [],
      "pass": true
    }
  },
  "status": "FAIL"
}

R14-B and R15 remain NOT AUTHORIZED.
