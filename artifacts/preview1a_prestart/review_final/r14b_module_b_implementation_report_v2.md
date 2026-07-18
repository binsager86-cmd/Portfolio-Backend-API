# R14-B Module (b) Implementation Report v2

Boundary: DataSurfaceAdapter + WarmupReadinessEngine

Set B distinction: THURAYA replay here is sealed historical data-surface plumbing verification only, not parameter selection.

## File Hashes
[
  {
    "path": "app/services/eagle_eye_v2/data_surface_adapter.py",
    "sha256": "3997920b6a14067080cf2585e5caf7553d87f3011c3ec1453bc0f22dfa5b7217",
    "size_bytes": 6408
  },
  {
    "path": "app/services/eagle_eye_v2/predicate_telemetry_ledger.py",
    "sha256": "432dc18a499401e9254a432ecfac2769d36b7da20c2dd419d6600c12b7b6f45a",
    "size_bytes": 12095
  },
  {
    "path": "app/services/eagle_eye_v2/telemetry_schema.py",
    "sha256": "36a0b5abfdd9c929bbc9cd2ed2b33946c237d62565d6149a4292e438af258aa8",
    "size_bytes": 2155
  },
  {
    "path": "app/services/eagle_eye_v2/warmup_readiness_engine.py",
    "sha256": "e03d9797a6451ff5751a4a88b15113175c8729abec2dfc0be9fb25cfdb017546",
    "size_bytes": 9114
  },
  {
    "path": "scripts/r14b_module_b_adapter_readiness_harness_v2.py",
    "sha256": "32080955cbc3a4d5541f7be2f681ca1aa6e91c9a476ff0c72bbe5fd62e78ef31",
    "size_bytes": 28688
  },
  {
    "path": "scripts/r15_surface_binding_v1.py",
    "sha256": "43841a4b92295e155eaaf0a2720a3e3df2a0dac5a49fb1c20d3cd2660f5d2a34",
    "size_bytes": 4133
  }
]

## R15 Surface Binding
- Binding artifact: r15_surface_binding_v1.json
- Canonical EE_V2 runtime DB: C:\Users\Sager\OneDrive\Desktop\portfolio_app\mobile-migration\backend-api-main-release\artifacts\preview1a_prestart\review_final\ee_v2_runtime_surface_r15_v1.db

## Boundary Artifacts
- r14b_module_b_interface_conformance_v2.json
- r14b_module_b_test_evidence_v2.json
- r14b_module_b_harness_output_v2.log

## Seam Surface Tables (THURAYA)
- Includes masked_context_flag, segment_day_index reset rows, dated readiness transition fields, and lookback values from persisted daily_term_row rows.

### THURAYA Suspension Interval Surface
```json
{
  "interval": {
    "end_date": "2025-07-27",
    "source_final_class": "SUSPECTED_CORPORATE_ACTION",
    "source_rule": "R-3",
    "span_days": 4,
    "start_date": "2025-07-24"
  },
  "no_cross_seam_samples": [
    {
      "curr_lookback_segment_sessions": 1,
      "curr_segment_day_index": 0,
      "curr_segment_id": "THURAYA::SEG0002",
      "prev_segment_day_index": 2,
      "prev_segment_id": "THURAYA::SEG0001",
      "prev_trade_date": "2025-07-23",
      "trade_date": "2025-07-24"
    },
    {
      "curr_lookback_segment_sessions": 1,
      "curr_segment_day_index": 0,
      "curr_segment_id": "THURAYA::SEG0003",
      "prev_segment_day_index": 0,
      "prev_segment_id": "THURAYA::SEG0002",
      "prev_trade_date": "2025-07-24",
      "trade_date": "2025-07-25"
    },
    {
      "curr_lookback_segment_sessions": 1,
      "curr_segment_day_index": 0,
      "curr_segment_id": "THURAYA::SEG0004",
      "prev_segment_day_index": 0,
      "prev_segment_id": "THURAYA::SEG0003",
      "prev_trade_date": "2025-07-25",
      "trade_date": "2025-07-26"
    },
    {
      "curr_lookback_segment_sessions": 1,
      "curr_segment_day_index": 0,
      "curr_segment_id": "THURAYA::SEG0005",
      "prev_segment_day_index": 0,
      "prev_segment_id": "THURAYA::SEG0004",
      "prev_trade_date": "2025-07-26",
      "trade_date": "2025-07-27"
    }
  ],
  "reset_rows": [
    {
      "lookback_segment_sessions": 1,
      "segment_day_index": 0,
      "segment_id": "THURAYA::SEG0002",
      "trade_date": "2025-07-24"
    },
    {
      "lookback_segment_sessions": 1,
      "segment_day_index": 0,
      "segment_id": "THURAYA::SEG0003",
      "trade_date": "2025-07-25"
    },
    {
      "lookback_segment_sessions": 1,
      "segment_day_index": 0,
      "segment_id": "THURAYA::SEG0004",
      "trade_date": "2025-07-26"
    },
    {
      "lookback_segment_sessions": 1,
      "segment_day_index": 0,
      "segment_id": "THURAYA::SEG0005",
      "trade_date": "2025-07-27"
    }
  ],
  "window_rows": [
    {
      "lookback_fallback_sessions": 82,
      "lookback_long_sessions": 122,
      "lookback_segment_sessions": 3,
      "masked_context_flag": false,
      "phase_after": "READINESS_LIMITED",
      "phase_before": "READINESS_LIMITED",
      "readiness_state": "READINESS_LIMITED",
      "readiness_transition_event": "NO_TRANSITION",
      "readiness_transition_from_state": "READINESS_LIMITED",
      "readiness_transition_to_state": "READINESS_LIMITED",
      "segment_day_index": 2,
      "segment_id": "THURAYA::SEG0001",
      "segment_restart_flag": false,
      "symbol": "THURAYA",
      "trade_date": "2025-07-23",
      "triggering_predicate_values": {
        "READINESS_FALLBACK_ELIGIBLE": 82.0,
        "READINESS_LONG_LOOKBACK_READY": 122.0,
        "READINESS_SEGMENT_RESTART_READY": 3.0
      }
    },
    {
      "lookback_fallback_sessions": 80,
      "lookback_long_sessions": 120,
      "lookback_segment_sessions": 1,
      "masked_context_flag": true,
      "phase_after": "READINESS_LIMITED",
      "phase_before": "READINESS_LIMITED",
      "readiness_state": "READINESS_LIMITED",
      "readiness_transition_event": "NO_TRANSITION",
      "readiness_transition_from_state": "READINESS_LIMITED",
      "readiness_transition_to_state": "READINESS_LIMITED",
      "segment_day_index": 0,
      "segment_id": "THURAYA::SEG0002",
      "segment_restart_flag": true,
      "symbol": "THURAYA",
      "trade_date": "2025-07-24",
      "triggering_predicate_values": {
        "READINESS_FALLBACK_ELIGIBLE": 80.0,
        "READINESS_LONG_LOOKBACK_READY": 120.0,
        "READINESS_SEGMENT_RESTART_READY": 1.0
      }
    },
    {
      "lookback_fallback_sessions": 80,
      "lookback_long_sessions": 120,
      "lookback_segment_sessions": 1,
      "masked_context_flag": true,
      "phase_after": "READINESS_LIMITED",
      "phase_before": "READINESS_LIMITED",
      "readiness_state": "READINESS_LIMITED",
      "readiness_transition_event": "NO_TRANSITION",
      "readiness_transition_from_state": "READINESS_LIMITED",
      "readiness_transition_to_state": "READINESS_LIMITED",
      "segment_day_index": 0,
      "segment_id": "THURAYA::SEG0003",
      "segment_restart_flag": true,
      "symbol": "THURAYA",
      "trade_date": "2025-07-25",
      "triggering_predicate_values": {
        "READINESS_FALLBACK_ELIGIBLE": 80.0,
        "READINESS_LONG_LOOKBACK_READY": 120.0,
        "READINESS_SEGMENT_RESTART_READY": 1.0
      }
    },
    {
      "lookback_fallback_sessions": 80,
      "lookback_long_sessions": 120,
      "lookback_segment_sessions": 1,
      "masked_context_flag": true,
      "phase_after": "READINESS_LIMITED",
      "phase_before": "READINESS_LIMITED",
      "readiness_state": "READINESS_LIMITED",
      "readiness_transition_event": "NO_TRANSITION",
      "readiness_transition_from_state": "READINESS_LIMITED",
      "readiness_transition_to_state": "READINESS_LIMITED",
      "segment_day_index": 0,
      "segment_id": "THURAYA::SEG0004",
      "segment_restart_flag": true,
      "symbol": "THURAYA",
      "trade_date": "2025-07-26",
      "triggering_predicate_values": {
        "READINESS_FALLBACK_ELIGIBLE": 80.0,
        "READINESS_LONG_LOOKBACK_READY": 120.0,
        "READINESS_SEGMENT_RESTART_READY": 1.0
      }
    },
    {
      "lookback_fallback_sessions": 80,
      "lookback_long_sessions": 120,
      "lookback_segment_sessions": 1,
      "masked_context_flag": true,
      "phase_after": "READINESS_LIMITED",
      "phase_before": "READINESS_LIMITED",
      "readiness_state": "READINESS_LIMITED",
      "readiness_transition_event": "NO_TRANSITION",
      "readiness_transition_from_state": "READINESS_LIMITED",
      "readiness_transition_to_state": "READINESS_LIMITED",
      "segment_day_index": 0,
      "segment_id": "THURAYA::SEG0005",
      "segment_restart_flag": true,
      "symbol": "THURAYA",
      "trade_date": "2025-07-27",
      "triggering_predicate_values": {
        "READINESS_FALLBACK_ELIGIBLE": 80.0,
        "READINESS_LONG_LOOKBACK_READY": 120.0,
        "READINESS_SEGMENT_RESTART_READY": 1.0
      }
    },
    {
      "lookback_fallback_sessions": 83,
      "lookback_long_sessions": 123,
      "lookback_segment_sessions": 4,
      "masked_context_flag": false,
      "phase_after": "READINESS_LIMITED",
      "phase_before": "READINESS_LIMITED",
      "readiness_state": "READINESS_LIMITED",
      "readiness_transition_event": "NO_TRANSITION",
      "readiness_transition_from_state": "READINESS_LIMITED",
      "readiness_transition_to_state": "READINESS_LIMITED",
      "segment_day_index": 3,
      "segment_id": "THURAYA::SEG0001",
      "segment_restart_flag": false,
      "symbol": "THURAYA",
      "trade_date": "2025-07-28",
      "triggering_predicate_values": {
        "READINESS_FALLBACK_ELIGIBLE": 80.0,
        "READINESS_LONG_LOOKBACK_READY": 120.0,
        "READINESS_SEGMENT_RESTART_READY": 1.0
      }
    }
  ]
}
```

### THURAYA 2026-06-28 Interval Surface
```json
{
  "interval": {
    "end_date": "2026-06-29",
    "source_final_class": "SUSPECTED_CORPORATE_ACTION",
    "source_rule": "R-3",
    "span_days": 2,
    "start_date": "2026-06-28"
  },
  "no_cross_seam_samples": [
    {
      "curr_lookback_segment_sessions": 1,
      "curr_segment_day_index": 0,
      "curr_segment_id": "THURAYA::SEG0007",
      "prev_segment_day_index": 5,
      "prev_segment_id": "THURAYA::SEG0006",
      "prev_trade_date": "2026-06-27",
      "trade_date": "2026-06-28"
    },
    {
      "curr_lookback_segment_sessions": 1,
      "curr_segment_day_index": 0,
      "curr_segment_id": "THURAYA::SEG0008",
      "prev_segment_day_index": 0,
      "prev_segment_id": "THURAYA::SEG0007",
      "prev_trade_date": "2026-06-28",
      "trade_date": "2026-06-29"
    }
  ],
  "reset_rows": [
    {
      "lookback_segment_sessions": 1,
      "segment_day_index": 0,
      "segment_id": "THURAYA::SEG0007",
      "trade_date": "2026-06-28"
    },
    {
      "lookback_segment_sessions": 1,
      "segment_day_index": 0,
      "segment_id": "THURAYA::SEG0008",
      "trade_date": "2026-06-29"
    }
  ],
  "window_rows": [
    {
      "lookback_fallback_sessions": 85,
      "lookback_long_sessions": 125,
      "lookback_segment_sessions": 6,
      "masked_context_flag": false,
      "phase_after": "READINESS_LIMITED",
      "phase_before": "READINESS_LIMITED",
      "readiness_state": "READINESS_LIMITED",
      "readiness_transition_event": "NO_TRANSITION",
      "readiness_transition_from_state": "READINESS_LIMITED",
      "readiness_transition_to_state": "READINESS_LIMITED",
      "segment_day_index": 5,
      "segment_id": "THURAYA::SEG0006",
      "segment_restart_flag": false,
      "symbol": "THURAYA",
      "trade_date": "2026-06-27",
      "triggering_predicate_values": {
        "READINESS_FALLBACK_ELIGIBLE": 85.0,
        "READINESS_LONG_LOOKBACK_READY": 125.0,
        "READINESS_SEGMENT_RESTART_READY": 6.0
      }
    },
    {
      "lookback_fallback_sessions": 80,
      "lookback_long_sessions": 120,
      "lookback_segment_sessions": 1,
      "masked_context_flag": true,
      "phase_after": "READINESS_LIMITED",
      "phase_before": "READINESS_LIMITED",
      "readiness_state": "READINESS_LIMITED",
      "readiness_transition_event": "NO_TRANSITION",
      "readiness_transition_from_state": "READINESS_LIMITED",
      "readiness_transition_to_state": "READINESS_LIMITED",
      "segment_day_index": 0,
      "segment_id": "THURAYA::SEG0007",
      "segment_restart_flag": true,
      "symbol": "THURAYA",
      "trade_date": "2026-06-28",
      "triggering_predicate_values": {
        "READINESS_FALLBACK_ELIGIBLE": 80.0,
        "READINESS_LONG_LOOKBACK_READY": 120.0,
        "READINESS_SEGMENT_RESTART_READY": 1.0
      }
    },
    {
      "lookback_fallback_sessions": 80,
      "lookback_long_sessions": 120,
      "lookback_segment_sessions": 1,
      "masked_context_flag": true,
      "phase_after": "READINESS_LIMITED",
      "phase_before": "READINESS_LIMITED",
      "readiness_state": "READINESS_LIMITED",
      "readiness_transition_event": "NO_TRANSITION",
      "readiness_transition_from_state": "READINESS_LIMITED",
      "readiness_transition_to_state": "READINESS_LIMITED",
      "segment_day_index": 0,
      "segment_id": "THURAYA::SEG0008",
      "segment_restart_flag": true,
      "symbol": "THURAYA",
      "trade_date": "2026-06-29",
      "triggering_predicate_values": {
        "READINESS_FALLBACK_ELIGIBLE": 80.0,
        "READINESS_LONG_LOOKBACK_READY": 120.0,
        "READINESS_SEGMENT_RESTART_READY": 1.0
      }
    },
    {
      "lookback_fallback_sessions": 87,
      "lookback_long_sessions": 127,
      "lookback_segment_sessions": 8,
      "masked_context_flag": false,
      "phase_after": "READINESS_LIMITED",
      "phase_before": "READINESS_LIMITED",
      "readiness_state": "READINESS_LIMITED",
      "readiness_transition_event": "NO_TRANSITION",
      "readiness_transition_from_state": "READINESS_LIMITED",
      "readiness_transition_to_state": "READINESS_LIMITED",
      "segment_day_index": 7,
      "segment_id": "THURAYA::SEG0001",
      "segment_restart_flag": false,
      "symbol": "THURAYA",
      "trade_date": "2026-06-30",
      "triggering_predicate_values": {
        "READINESS_FALLBACK_ELIGIBLE": 80.0,
        "READINESS_LONG_LOOKBACK_READY": 120.0,
        "READINESS_SEGMENT_RESTART_READY": 1.0
      }
    }
  ]
}
```

## Transition Persistence Check
```json
{
  "expected_predicate_rows": 138,
  "observed_predicate_names": [
    "READINESS_FALLBACK_ELIGIBLE",
    "READINESS_LONG_LOOKBACK_READY",
    "READINESS_SEGMENT_RESTART_READY"
  ],
  "observed_warmup_rows": 390,
  "pass": true,
  "required_predicate_names": [
    "READINESS_FALLBACK_ELIGIBLE",
    "READINESS_LONG_LOOKBACK_READY",
    "READINESS_SEGMENT_RESTART_READY"
  ],
  "transition_rows_with_date_symbol_segment_state": 390
}
```

## Harness Output (Verbatim)
```text
R14B_MODULE_B_HARNESS_V2_START
SURFACE_BOUND C:\Users\Sager\OneDrive\Desktop\portfolio_app\mobile-migration\backend-api-main-release\artifacts\preview1a_prestart\review_final\ee_v2_runtime_surface_r15_v1.db
DDL_APPLIED count=16
SOURCE_TABLE ee_ohlcv
SYMBOLS SANAM,THURAYA,AAYAN
THURAYA_INTERVALS {"june_interval": {"end_date": "2026-06-29", "source_final_class": "SUSPECTED_CORPORATE_ACTION", "source_rule": "R-3", "span_days": 2, "start_date": "2026-06-28"}, "suspension_interval": {"end_date": "2025-07-27", "source_final_class": "SUSPECTED_CORPORATE_ACTION", "source_rule": "R-3", "span_days": 4, "start_date": "2025-07-24"}}
TRIGGER_CHECK pass=True
PREDICATE_LEDGER_CHECK pass=True
TRANSITION_ROW_PERSISTENCE rows=390
SIDECAR_CHAIN_ADVANCE rows=18
R14B_MODULE_B_HARNESS_V2_COMPLETE

```
