# R14-B Module (b) Implementation Report v3

Boundary: DataSurfaceAdapter + WarmupReadinessEngine on dedicated harness DB only.

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
    "path": "scripts/r14b_module_b_adapter_readiness_harness_v3.py",
    "sha256": "f468c10b58f369ce982c88e73242499670e085c995a388c4f5a1259cfb14fd83",
    "size_bytes": 19427
  }
]

## Harness DB
C:\Users\Sager\OneDrive\Desktop\portfolio_app\mobile-migration\backend-api-main-release\artifacts\preview1a_prestart\review_final\r14b_module_b_harness_surface_v3.db

## Seam Evidence (Real Bars Only)
```json
{
  "june_interval": {
    "gap_calendar_absence": {
      "absence_is_reported_as_absence": true,
      "calendar_verified_holiday_dates": [],
      "gap_dates": [
        "2026-06-28",
        "2026-06-29"
      ],
      "gap_session_count": 2,
      "missing_real_bar_count": 2,
      "missing_real_bar_dates": [
        "2026-06-28",
        "2026-06-29"
      ],
      "real_bar_count_in_gap": 0
    },
    "interval": {
      "end_date": "2026-06-29",
      "interval_id": "THURAYA::2026-06-28::2026-06-29::R-3",
      "source_final_class": "SUSPECTED_CORPORATE_ACTION",
      "source_rule": "R-3",
      "span_days": 2,
      "start_date": "2026-06-28"
    },
    "interval_id": "THURAYA::2026-06-28::2026-06-29::R-3",
    "post_gap_real_bar_table": [
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 1,
        "masked_context_flag": false,
        "phase_after": "READINESS_LIMITED",
        "phase_before": "READY",
        "readiness_state": "READINESS_LIMITED",
        "readiness_transition_event": "READY_TO_READINESS_LIMITED",
        "readiness_transition_from_state": "READY",
        "readiness_transition_to_state": "READINESS_LIMITED",
        "segment_day_index": 0,
        "segment_id": "THURAYA::SEG0003",
        "segment_restart_flag": true,
        "symbol": "THURAYA",
        "trade_date": "2026-06-30",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 1.0
        }
      },
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 2,
        "masked_context_flag": false,
        "phase_after": "READY",
        "phase_before": "READINESS_LIMITED",
        "readiness_state": "READY",
        "readiness_transition_event": "READINESS_LIMITED_TO_READY",
        "readiness_transition_from_state": "READINESS_LIMITED",
        "readiness_transition_to_state": "READY",
        "segment_day_index": 1,
        "segment_id": "THURAYA::SEG0003",
        "segment_restart_flag": false,
        "symbol": "THURAYA",
        "trade_date": "2026-07-01",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 2.0
        }
      },
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 3,
        "masked_context_flag": false,
        "phase_after": "READY",
        "phase_before": "READY",
        "readiness_state": "READY",
        "readiness_transition_event": "NO_TRANSITION",
        "readiness_transition_from_state": "READY",
        "readiness_transition_to_state": "READY",
        "segment_day_index": 2,
        "segment_id": "THURAYA::SEG0003",
        "segment_restart_flag": false,
        "symbol": "THURAYA",
        "trade_date": "2026-07-02",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 3.0
        }
      }
    ],
    "pre_gap_real_bar_table": [
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 4,
        "masked_context_flag": false,
        "phase_after": "READY",
        "phase_before": "READY",
        "readiness_state": "READY",
        "readiness_transition_event": "NO_TRANSITION",
        "readiness_transition_from_state": "READY",
        "readiness_transition_to_state": "READY",
        "segment_day_index": 3,
        "segment_id": "THURAYA::SEG0002",
        "segment_restart_flag": false,
        "symbol": "THURAYA",
        "trade_date": "2026-06-23",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 4.0
        }
      },
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 5,
        "masked_context_flag": false,
        "phase_after": "READY",
        "phase_before": "READY",
        "readiness_state": "READY",
        "readiness_transition_event": "NO_TRANSITION",
        "readiness_transition_from_state": "READY",
        "readiness_transition_to_state": "READY",
        "segment_day_index": 4,
        "segment_id": "THURAYA::SEG0002",
        "segment_restart_flag": false,
        "symbol": "THURAYA",
        "trade_date": "2026-06-24",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 5.0
        }
      },
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 6,
        "masked_context_flag": false,
        "phase_after": "READY",
        "phase_before": "READY",
        "readiness_state": "READY",
        "readiness_transition_event": "NO_TRANSITION",
        "readiness_transition_from_state": "READY",
        "readiness_transition_to_state": "READY",
        "segment_day_index": 5,
        "segment_id": "THURAYA::SEG0002",
        "segment_restart_flag": false,
        "symbol": "THURAYA",
        "trade_date": "2026-06-25",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 6.0
        }
      }
    ],
    "restart_transition_rows_with_date": [
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 1,
        "masked_context_flag": false,
        "phase_after": "READINESS_LIMITED",
        "phase_before": "READY",
        "readiness_state": "READINESS_LIMITED",
        "readiness_transition_event": "READY_TO_READINESS_LIMITED",
        "readiness_transition_from_state": "READY",
        "readiness_transition_to_state": "READINESS_LIMITED",
        "segment_day_index": 0,
        "segment_id": "THURAYA::SEG0003",
        "segment_restart_flag": true,
        "symbol": "THURAYA",
        "trade_date": "2026-06-30",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 1.0
        }
      }
    ]
  },
  "suspension_interval": {
    "gap_calendar_absence": {
      "absence_is_reported_as_absence": true,
      "calendar_verified_holiday_dates": [],
      "gap_dates": [
        "2021-05-02",
        "2021-05-03"
      ],
      "gap_session_count": 2,
      "missing_real_bar_count": 2,
      "missing_real_bar_dates": [
        "2021-05-02",
        "2021-05-03"
      ],
      "real_bar_count_in_gap": 0
    },
    "interval": {
      "end_date": "2021-05-03",
      "interval_id": "THURAYA::2021-05-02::2021-05-03::R-3",
      "source_final_class": "SUSPECTED_CORPORATE_ACTION",
      "source_rule": "R-3",
      "span_days": 2,
      "start_date": "2021-05-02"
    },
    "interval_id": "THURAYA::2021-05-02::2021-05-03::R-3",
    "post_gap_real_bar_table": [
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 1,
        "masked_context_flag": false,
        "phase_after": "READINESS_LIMITED",
        "phase_before": "READY",
        "readiness_state": "READINESS_LIMITED",
        "readiness_transition_event": "READY_TO_READINESS_LIMITED",
        "readiness_transition_from_state": "READY",
        "readiness_transition_to_state": "READINESS_LIMITED",
        "segment_day_index": 0,
        "segment_id": "THURAYA::SEG0002",
        "segment_restart_flag": true,
        "symbol": "THURAYA",
        "trade_date": "2021-05-06",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 1.0
        }
      },
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 2,
        "masked_context_flag": false,
        "phase_after": "READY",
        "phase_before": "READINESS_LIMITED",
        "readiness_state": "READY",
        "readiness_transition_event": "READINESS_LIMITED_TO_READY",
        "readiness_transition_from_state": "READINESS_LIMITED",
        "readiness_transition_to_state": "READY",
        "segment_day_index": 1,
        "segment_id": "THURAYA::SEG0002",
        "segment_restart_flag": false,
        "symbol": "THURAYA",
        "trade_date": "2021-05-09",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 2.0
        }
      },
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 3,
        "masked_context_flag": false,
        "phase_after": "READY",
        "phase_before": "READY",
        "readiness_state": "READY",
        "readiness_transition_event": "NO_TRANSITION",
        "readiness_transition_from_state": "READY",
        "readiness_transition_to_state": "READY",
        "segment_day_index": 2,
        "segment_id": "THURAYA::SEG0002",
        "segment_restart_flag": false,
        "symbol": "THURAYA",
        "trade_date": "2021-05-10",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 3.0
        }
      }
    ],
    "pre_gap_real_bar_table": [
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 1,
        "masked_context_flag": false,
        "phase_after": "READINESS_LIMITED",
        "phase_before": "READINESS_PENDING",
        "readiness_state": "READINESS_LIMITED",
        "readiness_transition_event": "READINESS_PENDING_TO_READINESS_LIMITED",
        "readiness_transition_from_state": "READINESS_PENDING",
        "readiness_transition_to_state": "READINESS_LIMITED",
        "segment_day_index": 0,
        "segment_id": "THURAYA::SEG0001",
        "segment_restart_flag": true,
        "symbol": "THURAYA",
        "trade_date": "2021-04-27",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 1.0
        }
      },
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 2,
        "masked_context_flag": false,
        "phase_after": "READY",
        "phase_before": "READINESS_LIMITED",
        "readiness_state": "READY",
        "readiness_transition_event": "READINESS_LIMITED_TO_READY",
        "readiness_transition_from_state": "READINESS_LIMITED",
        "readiness_transition_to_state": "READY",
        "segment_day_index": 1,
        "segment_id": "THURAYA::SEG0001",
        "segment_restart_flag": false,
        "symbol": "THURAYA",
        "trade_date": "2021-04-28",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 2.0
        }
      },
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 3,
        "masked_context_flag": false,
        "phase_after": "READY",
        "phase_before": "READY",
        "readiness_state": "READY",
        "readiness_transition_event": "NO_TRANSITION",
        "readiness_transition_from_state": "READY",
        "readiness_transition_to_state": "READY",
        "segment_day_index": 2,
        "segment_id": "THURAYA::SEG0001",
        "segment_restart_flag": false,
        "symbol": "THURAYA",
        "trade_date": "2021-04-29",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 3.0
        }
      }
    ],
    "restart_transition_rows_with_date": [
      {
        "lookback_fallback_sessions": 80,
        "lookback_long_sessions": 220,
        "lookback_segment_sessions": 1,
        "masked_context_flag": false,
        "phase_after": "READINESS_LIMITED",
        "phase_before": "READY",
        "readiness_state": "READINESS_LIMITED",
        "readiness_transition_event": "READY_TO_READINESS_LIMITED",
        "readiness_transition_from_state": "READY",
        "readiness_transition_to_state": "READINESS_LIMITED",
        "segment_day_index": 0,
        "segment_id": "THURAYA::SEG0002",
        "segment_restart_flag": true,
        "symbol": "THURAYA",
        "trade_date": "2021-05-06",
        "triggering_predicate_values": {
          "READINESS_FALLBACK_ELIGIBLE": 80.0,
          "READINESS_LONG_LOOKBACK_READY": 220.0,
          "READINESS_SEGMENT_RESTART_READY": 1.0
        }
      }
    ]
  }
}
```

## Harness Output (Verbatim)
```text
R14B_MODULE_B_HARNESS_V3_START
HARNESS_DB C:\Users\Sager\OneDrive\Desktop\portfolio_app\mobile-migration\backend-api-main-release\artifacts\preview1a_prestart\review_final\r14b_module_b_harness_surface_v3.db
DDL_APPLIED count=16
PROCESSED_REAL_BAR_DATES 12
MASKED_REAL_BARS_IN_SLICE 0
R14B_MODULE_B_HARNESS_V3_COMPLETE

```
