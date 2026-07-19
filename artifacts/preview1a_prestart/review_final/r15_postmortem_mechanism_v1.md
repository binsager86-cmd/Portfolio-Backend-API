# R15 Postmortem Mechanism v1

Mode: ROWS_ONLY

Sources:

- sealed_attempt_2_db: `artifacts/preview1a_prestart/review_final/r15_exam_v2_harness.db`
- sealed_attempt_2_report: `artifacts/preview1a_prestart/review_final/r15_exam_report_v2.md`
- note: `execution_outcome_row` count in sealed_attempt_2_db = 0; position ledger rows below are copied from sealed_attempt_2_report `Position Ledger With Sessions Held`.

## Position Ledger Rows

position_id|symbol|entry_date|tier|sessions_held|final_event
---|---|---|---|---:|---
SANAM|NO_ROWS|NO_ROWS|NO_ROWS|NO_ROWS|NO_ROWS
TIJARA::POS0001|TIJARA|2021-10-11|EARLY_ACCUMULATION_ENTRY|13|CLOSE_INVALIDATION
TIJARA::POS0002|TIJARA|2021-11-22|EARLY_ACCUMULATION_ENTRY|60|CLOSE_TIME_STOP
TIJARA::POS0003|TIJARA|2022-02-21|EARLY_ACCUMULATION_ENTRY|49|CLOSE_INVALIDATION
TIJARA::POS0004|TIJARA|2022-08-23|EARLY_ACCUMULATION_ENTRY|24|CLOSE_INVALIDATION

## Disposition Rows: SANAM 2025-05-08..2025-05-29

source_table: `ee_v2_forward_predictions`

row_count: 0

date|prediction_id|intent_state|execution_state|suppression_veto_reason
---|---|---|---|---
NO_ROWS|NO_ROWS|NO_ROWS|NO_ROWS|NO_ROWS

## Disposition Rows: TIJARA 2024-09-01..2024-12-31

source_table: `ee_v2_forward_predictions`

row_count: 0

date|prediction_id|intent_state|execution_state|suppression_veto_reason
---|---|---|---|---
NO_ROWS|NO_ROWS|NO_ROWS|NO_ROWS|NO_ROWS

## Position Open At Criterion Window Start

symbol|window_start|position_open|sessions_held|source_row
---|---|---|---:|---
SANAM|2025-05-08|NO|0|SANAM 2025-05-08 READY BASE_RETIRED NONE INTENT_NONE NOT_CONFIRMED NONE NONE NONE 0
TIJARA|2024-09-01|NO|0|TIJARA 2024-09-01 READY BASE_RETIRED NONE INTENT_NONE NOT_CONFIRMED NONE NONE NONE 0
