# R15 Attempt 2 Criteria Of Record v1

Status: PRE_REGISTERED_FINAL_BEFORE_ATTEMPT_2_EXECUTION

R15 attempt 2 is the second and final attempt under the stopping rule.

These criteria are final for attempt 2. They must not be revised after execution. FAIL is reported as FAIL.

## Criteria

1. SANAM confirmed entry occurs within `2025-05-08..2025-05-29` from a then-live, non-retired base.
2. SANAM early-tier entry occurs within `2025-03-01..2025-05-07` from a then-live base. Pre-registered expectation: FAIL under current detection timing (`FLOW_CORE_LAG`). Record actual first-intent date regardless.
3. TIJARA confirmed entry occurs within `2024-09-01..2024-12-31` from a then-live base.
4. MABANEE has zero early entries inside decline windows `2024-12-22..2025-02-20` and `2025-03-24..2025-05-18`. This is a preservation criterion.
5. Early-tier false-positive cost is computed and reported: failed-pilot count, capital-days, realized-drawdown min, plus duplicate-pilot suppression counts.

## Execution Constraints

- Use freeze v2 plus amendments 1 and 2 byte-match attestation as an execution gate.
- Use the same five Set A symbols and same full windows as attempt 1.
- Use a fresh `RUN_NONCE`.
- Use live-wired prediction ledger writes inline.
- Run grader after prediction ledger completion, bounded by sealed data.
- Harness DB only; canonical and Set B untouched.
- No tuning.
- No reruns after results are seen.