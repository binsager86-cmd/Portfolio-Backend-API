# R15-REM Upward Retirement Evidence v2

Mode: READ_ONLY_SEALED_R15_EXAM_DB_PLUS_RUNTIME_OHLCV_LOOKUP

Exam artifact mutation: NONE

## Threshold 0.15

Early execution rows: 304
Would suppress: 196 (0.6447368421052632)
Extension before: {"count": 304, "max": 3.0847457627118646, "median": 0.2840501792114696, "p75": 2.5084745762711864}
Extension after: {"count": 108, "max": 0.08888888888888889, "median": 0.0, "p75": 0.017955801104972375}
Upward-retired unique early bases: 2 / 5
Reformed above retired high: 2 (1.0); median sessions: 62.5
Suppressed by symbol: {"BPCC": 0, "MABANEE": 0, "SANAM": 146, "TIJARA": 50, "ZAIN": 0}

## Threshold 0.20

Early execution rows: 304
Would suppress: 196 (0.6447368421052632)
Extension before: {"count": 304, "max": 3.0847457627118646, "median": 0.2840501792114696, "p75": 2.5084745762711864}
Extension after: {"count": 108, "max": 0.08888888888888889, "median": 0.0, "p75": 0.017955801104972375}
Upward-retired unique early bases: 2 / 5
Reformed above retired high: 2 (1.0); median sessions: 62.0
Suppressed by symbol: {"BPCC": 0, "MABANEE": 0, "SANAM": 146, "TIJARA": 50, "ZAIN": 0}

## Threshold 0.25

Early execution rows: 304
Would suppress: 196 (0.6447368421052632)
Extension before: {"count": 304, "max": 3.0847457627118646, "median": 0.2840501792114696, "p75": 2.5084745762711864}
Extension after: {"count": 108, "max": 0.08888888888888889, "median": 0.0, "p75": 0.017955801104972375}
Upward-retired unique early bases: 2 / 5
Reformed above retired high: 2 (1.0); median sessions: 59.5
Suppressed by symbol: {"BPCC": 0, "MABANEE": 0, "SANAM": 146, "TIJARA": 50, "ZAIN": 0}

## Criterion iii Options

{
  "execution_rows": 53,
  "option_a_candidate_state_strict": {
    "candidate_intent_days": 249,
    "no_candidate_count": 954,
    "no_candidate_rate": 0.7930174563591023,
    "verdict_under_25pct_rule": "FAIL",
    "wording": "TIJARA no-candidate means candidate_intent.intent_state != INTENT_FORMED; veto days remain no-candidate days."
  },
  "option_b_explicit_disposition_coverage": {
    "explicit_disposition_rows": 411,
    "no_record_count": 792,
    "no_record_rate": 0.6583541147132169,
    "verdict_under_25pct_rule": "FAIL",
    "wording": "TIJARA no-candidate means no sealed forward-prediction disposition row; veto rows count as explicit non-entry dispositions."
  },
  "prediction_rows": 411,
  "suppression_rows": 196,
  "total_days": 1203,
  "veto_rows": 162
}
