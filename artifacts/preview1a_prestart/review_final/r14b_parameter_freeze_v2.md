# R14-B Parameter Freeze v2

{
  "authority": {
    "governing_design_doc": {
      "json_sha256": "9a5f1facdf1fc222239e6304afadb1e420f4d22fb506f23d97af52b64cb4b52b",
      "md_sha256": "dedf5361c25b40df5b0ece8dbfeb9f360e81cf5facfdb3a2305dcf6c8b31de4e",
      "version": "R14_DESIGN_SPEC_CONSOLIDATED_V2_2"
    },
    "implementation_liberty_note": "No implementation liberty beyond governing text without owner directive.",
    "owner_ratification_received": true,
    "owner_ratification_status": "OWNER_RATIFIED_AT_PARAMETER_GATE"
  },
  "baseline": {
    "implementation_baseline_id": "EE_V2_20260714T163910Z",
    "new_module_path": "app/services/eagle_eye_v2",
    "r11_baseline_status": "UNTOUCHED_ARCHIVED",
    "supersession_rule": "Old engine is never edited; only superseded by isolated v2 module path."
  },
  "conduct_rules_reaffirmed": [
    "Permanent scripts only",
    "Append-only artifacts",
    "Frozen verifiers",
    "No self-declared gate passage",
    "No temp-and-delete",
    "Set B quarantine remains in force",
    "Canonical surface unchanged",
    "Recurrence counting continues"
  ],
  "extension_mode": "APPEND_ONLY",
  "module_authorizations": {
    "build_order": [
      "e",
      "f",
      "g"
    ],
    "module_e": "AUTHORIZED",
    "module_f": "AUTHORIZED_TO_FOLLOW_ON_MODULE_E_REVIEW_PASS",
    "module_g": "AUTHORIZED_TO_FOLLOW_ON_MODULE_E_REVIEW_PASS"
  },
  "r14b_parameter_gate_ratifications_v2": {
    "flow_core_composition": {
      "blocking_authority": "NONE_UNTIL_OWNER_RATIFICATION_OF_CONDITIONAL_ANALYSIS",
      "interim_wiring": "OBV_ANV_SLOPE_CORE_DRIVES_EARLY_TIER_DETECTION_PREDICATES",
      "set_b_quarantine_reaffirmed": true,
      "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
      "value": "DEFERRED_PENDING_CONDITIONAL_ANALYSIS"
    },
    "frozen_parameters": {
      "LIQUIDITY_EXECUTION_SIZE_PARAMETER": {
        "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
        "value": 0.1
      },
      "adx_trigger": {
        "owner_amendment": {
          "rationale": "Consistency with cited EX_SET_B p55~=25 distribution.",
          "supersedes_proposal": 15.0
        },
        "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
        "value": 18.0
      },
      "atr_squeeze_pctile": {
        "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
        "value": 0.95
      },
      "base_max_width_pct": {
        "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
        "value": 0.24
      },
      "base_min_sessions": {
        "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
        "value": 10
      },
      "cmf_floor": {
        "authority": "TELEMETRY_ONLY",
        "status": "OWNER_RATIFIED_AT_PARAMETER_GATE_PENDING_FLOW_CORE_DECISION",
        "value": 0.05
      },
      "ml_prob_min": {
        "authority": "NON_BLOCKING",
        "principle": "F8c_no_unauditable_veto",
        "status": "OWNER_RATIFIED_AT_PARAMETER_GATE_UNTIL_AUDITABLE_ML_SURFACE",
        "value": 0.55
      },
      "rsi_regime": {
        "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
        "value": 50.0
      },
      "volume_breakout_mult": {
        "authority": "CONTEXT_ONLY_NEVER_SOLE_VETO",
        "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
        "value": 2.5
      }
    },
    "invalidation_rule": {
      "evidence": {
        "gate_table_row": {
          "base_count": 522,
          "false_persistence_pct": 0.0,
          "median_life": 93.5,
          "survive60_pct": 57.09,
          "tiers": "all tiers"
        },
        "source": "r14b_parameter_gate_evidence_v1"
      },
      "name": "INVALIDATION_RULE",
      "parameters": {
        "atr_mult": 1.0,
        "n_sessions": 2
      },
      "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
      "value": "CLOSE_BELOW_BASE_LOW_BY_ATR_X_N"
    }
  },
  "supersedes": "R14B_PARAMETER_FREEZE_V1_BY_EXTENSION_ONLY",
  "v1_restated_values_unchanged": {
    "restated_from": "R14B_PARAMETER_FREEZE_V1",
    "status": "IN_FORCE_UNCHANGED",
    "values": {
      "CHASE_ADVISORY_BAND": "advisory flag > 0.08 extension vs current valid reference; escalation flag > 0.15",
      "EARLY_TIER_PARTICIPATION_CAP": "0.10 (fraction of daily traded value)",
      "EARLY_TIER_SIZE_FRACTION": "0.30 (fraction of full target position)",
      "EARLY_TIER_TIME_STOP": "60 sessions, REVIEW semantics: at expiry re-evaluate flow predicates; exit only on flow-evidence decay; else re-arm clock; max 2 re-arms then OWNER_REVIEW state",
      "GRADING_HORIZONS": "[20, 60, 120] sessions",
      "MARKUP_MATERIALIZATION_CRITERION": "max favorable excursion >= +0.20 within 120 sessions of prediction",
      "MIN_CALIBRATION_WINDOW": "63 sessions",
      "SCALE_ON_CONFIRMATION": "SINGLE_ADD_TO_FULL_TARGET at BREAKOUT_CONFIRMED_ENTRY",
      "TIER_RULE": "CANDIDATE_A (HIGH >= 500000 KWD median daily value; MID >= 100000 KWD; else LOW; one-time sanity check vs live tier profile during R14-B)"
    }
  },
  "version_id": "R14B_PARAMETER_FREEZE_V2"
}
