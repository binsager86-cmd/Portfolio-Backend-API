# R14-B Parameter Freeze v1

{
  "authority": {
    "governing_design_doc": {
      "json_sha256": "9a5f1facdf1fc222239e6304afadb1e420f4d22fb506f23d97af52b64cb4b52b",
      "md_sha256": "dedf5361c25b40df5b0ece8dbfeb9f360e81cf5facfdb3a2305dcf6c8b31de4e",
      "version": "R14_DESIGN_SPEC_CONSOLIDATED_V2_2"
    },
    "implementation_liberty_note": "No implementation liberty beyond governing text without owner directive.",
    "owner_ratification_received": true
  },
  "baseline": {
    "implementation_baseline_id": "EE_V2_20260713T170715Z",
    "new_module_path": "app/services/eagle_eye_v2",
    "r11_baseline_status": "UNTOUCHED_ARCHIVED",
    "supersession_rule": "Old engine is never edited; only superseded by isolated v2 module path."
  },
  "conduct_rules_reaffirmed": [
    "Permanent scripts only",
    "Append-only artifacts",
    "Frozen verifiers",
    "No self-declared gate passage",
    "No temp-and-delete"
  ],
  "owner_ratified_values_verbatim": {
    "CHASE_ADVISORY_BAND": "advisory flag > 0.08 extension vs current valid reference; escalation flag > 0.15",
    "EARLY_TIER_PARTICIPATION_CAP": "0.10 (fraction of daily traded value)",
    "EARLY_TIER_SIZE_FRACTION": "0.30 (fraction of full target position)",
    "EARLY_TIER_TIME_STOP": "60 sessions, REVIEW semantics: at expiry re-evaluate flow predicates; exit only on flow-evidence decay; else re-arm clock; max 2 re-arms then OWNER_REVIEW state",
    "GRADING_HORIZONS": "[20, 60, 120] sessions",
    "MARKUP_MATERIALIZATION_CRITERION": "max favorable excursion >= +0.20 within 120 sessions of prediction",
    "MIN_CALIBRATION_WINDOW": "63 sessions",
    "SCALE_ON_CONFIRMATION": "SINGLE_ADD_TO_FULL_TARGET at BREAKOUT_CONFIRMED_ENTRY",
    "TIER_RULE": "CANDIDATE_A (HIGH >= 500000 KWD median daily value; MID >= 100000 KWD; else LOW; one-time sanity check vs live tier profile during R14-B)"
  },
  "quarantine_and_prohibitions": {
    "no_backtests_no_stat_runs_no_r15_preview_no_r16_execution": true,
    "r15_authorization": "NOT_AUTHORIZED",
    "r16_authorization": "NOT_AUTHORIZED",
    "set_b_quarantine": "EXPLICITLY_RESTATED_NO_PARAMETER_VALUE_MAY_BE_CHOSEN_USING_SET_B",
    "threshold_mutation_post_freeze": "PROHIBITED_UNLESS_CHANGE_REQUEST_APPROVED"
  },
  "remaining_parameters_requiring_r14b_parameter_gate": [
    {
      "family": "base_geometry",
      "name": "base_min_sessions",
      "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE"
    },
    {
      "family": "base_geometry",
      "name": "base_max_width_pct",
      "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE"
    },
    {
      "family": "base_geometry",
      "name": "atr_squeeze_pctile",
      "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE"
    },
    {
      "family": "confirmation_thresholds",
      "name": "cmf_floor",
      "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE"
    },
    {
      "family": "confirmation_thresholds",
      "name": "volume_breakout_mult",
      "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE"
    },
    {
      "family": "confirmation_thresholds",
      "name": "rsi_regime",
      "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE"
    },
    {
      "family": "confirmation_thresholds",
      "name": "adx_trigger",
      "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE"
    },
    {
      "family": "liquidity_execution_size",
      "name": "LIQUIDITY_EXECUTION_SIZE_PARAMETER",
      "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE"
    },
    {
      "family": "ml_floor",
      "name": "ml_prob_min",
      "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE"
    }
  ],
  "time_stop_review_semantics": {
    "clock_rearm_rule": "REARM_WHEN_FLOW_HOLDS",
    "exit_condition": "FLOW_EVIDENCE_DECAY_ONLY",
    "max_clock_rearms": 2,
    "post_rearm_terminal_state": "OWNER_REVIEW",
    "re_evaluate_at_sessions": 60
  },
  "version_id": "R14B_PARAMETER_FREEZE_V1"
}
