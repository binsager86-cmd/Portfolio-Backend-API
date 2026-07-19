# R15 Criteria Re-Ratification Sheet v1

Status: DRAFT_REQUIRES_OWNER_RATIFICATION_BEFORE_ATTEMPT_2

R15_ATTEMPT_1 remains FAIL_OF_RECORD and is not rerun by this sheet.

Evidence source for computed numbers: sealed `r15_exam_v1_harness.db`, opened read-only by `r15rem_upward_retirement_evidence_v2.py`; exam artifacts were not mutated.

## Revised Criteria Options

1. Criterion i: SANAM must show a confirmed, non-stale-base entry or an explicit no-entry disposition tied to a newly ratified base lifecycle rule. Owner must choose exact wording before attempt 2.
2. Criterion ii: Wording must distinguish true thesis capture from stale-base early pilots. `PASS_ON_WORDING_HOLLOW` is not acceptable for attempt 2.
3. Criterion iii: TIJARA must be judged under one owner-ratified no-candidate definition before attempt 2. Both computed options remain FAIL under the existing `<25%` threshold:

Option A, candidate-state strict:

- Wording: TIJARA no-candidate means `candidate_intent.intent_state != INTENT_FORMED`; veto days remain no-candidate days.
- Total days: 1,203.
- Candidate-intent days: 249 = 53 execution rows + 196 suppression rows.
- No-candidate count: 954.
- No-candidate rate: 0.7930174563591023.
- Verdict under `<25%`: FAIL.

Option B, explicit-disposition coverage:

- Wording: TIJARA no-candidate means no sealed forward-prediction disposition row; veto rows count as explicit non-entry dispositions.
- Total days: 1,203.
- Explicit disposition rows: 411 = 53 execution rows + 196 suppression rows + 162 veto rows.
- No-record count: 792.
- No-record rate: 0.6583541147132169.
- Verdict under `<25%`: FAIL.

4. Criterion iv: MABANEE avoid protection remains a preservation criterion. Existing genuine PASS must not be degraded.
5. Criterion v: Pilot cost criterion must include duplicate-pilot suppression and capital-days, with failed pilot count and drawdown reported separately.

## Ratification Gate

Attempt 2 is not authorized until the owner ratifies this sheet or a successor sheet and explicitly authorizes `R15_ATTEMPT_2`.