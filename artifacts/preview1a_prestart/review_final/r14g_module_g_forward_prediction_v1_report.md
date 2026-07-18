# R14-G Module (g) ForwardPredictionLedger v1

- RUN_NONCE: 2026-07-18T12:41:36.9182244Z
- Freeze v2 byte-match: True
- Overall acceptance: PASS
- Predictions: 142
- Grades: 142
- SANAM 2025-05-15 verdict: PENDING_HORIZON
- Writer/grader separation: prediction reader opened mode=ro; prediction table UPDATE/DELETE trigger-blocked.

## Acceptance
{
  "EVERY_EVENT_DAY_HAS_PREDICTION_ROW": {
    "expected_events": 142,
    "extra": [],
    "missing": [],
    "prediction_rows": 142,
    "status": "PASS"
  },
  "SANAM_2025_05_15_MATERIALIZATION_POLICY": {
    "grade_row": {
      "grade_date": "2026-07-18T12:41:36.9182244Z",
      "grade_status": "PENDING_HORIZON",
      "grader_version": "R14G_PREDICTION_GRADER_V1",
      "materialization_verdict": "PENDING_HORIZON",
      "mfe_120": null,
      "prediction_date": "2025-05-15",
      "prediction_id": "4ff17bde7dcd8625be34b34b6dd8572c",
      "return_120": null,
      "return_20": null,
      "return_60": null,
      "sealed_data_last_date": "2025-05-29",
      "symbol": "SANAM"
    },
    "rule": "MATERIALIZED only if 120-session MFE inside sealed data clears +20%; otherwise NOT_MATERIALIZED or PENDING_HORIZON without reaching",
    "status": "PASS",
    "verdict": "PENDING_HORIZON"
  },
  "WRITER_GRADER_SEPARATION_ATTESTED": {
    "prediction_reader_write_attempt": {
      "prediction_reader_write_attempt": "attempt to write a readonly database"
    },
    "status": "PASS",
    "structural_enforcement": "prediction_grader opens ee_v2_forward_predictions through SQLite URI mode=ro and exposes no prediction-table write method; writer table UPDATE/DELETE are trigger-blocked",
    "writer_update_delete_attempts": {
      "delete": "append-only table: ee_v2_forward_predictions delete blocked",
      "update": "append-only table: ee_v2_forward_predictions update blocked"
    }
  },
  "ZERO_GRADEABLE_BUT_UNGRADED": {
    "gradeable": 24,
    "grades": 142,
    "predictions": 142,
    "status": "PASS"
  }
}
