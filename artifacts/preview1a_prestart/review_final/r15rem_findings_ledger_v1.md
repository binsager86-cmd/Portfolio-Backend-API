# R15-REM Findings Ledger v1

Mode: APPEND_ONLY

## R15_ATTEMPT_1

Status: FAIL_OF_RECORD

RUN_NONCE: 2026-07-18T11:16:01.688511Z

Stopping-rule status: FAILURE_1_OF_2

Criteria:

- i: FAIL
- ii: PASS_ON_WORDING_HOLLOW
- iii: FAIL
- iv: PASS_GENUINE
- v: PASS

Evidence anchors:

- Criterion i: SANAM 2025-05-18 emitted `SUPPRESSION_RESTRAINT`, not confirmed entry. Prediction ID `d750da9228f714381b7a64623938c191`.
- Criterion ii: SANAM first early pilot was `08c114b2a28e59cc3fbc1e27848e6b40` on 2023-11-16 at 168.0, referencing `SANAM::2021-06-23::BASE01`, base_high_ref 59.0, extension 1.847457627118644.
- Criterion iii: TIJARA full-window no-candidate criterion failed; see `r15_exam_report_v1.md` Per-Criterion Evidence.
- Criterion iv: MABANEE had zero early entries inside the decline windows; total avoid-veto rows across the sealed R15 attempt were 1,568. Sample MABANEE avoid-veto prediction IDs: `211375425be5bcf10052f0631fcbcc47`, `694e84f03b10e47966071b9be57695c6`, `dc2dbed6aa9dbf82e9e865001b976a60`.
- Criterion v: failed pilots 125, capital-days 4,537.5, min drawdown -37.9%.

## New Findings

### BASE_REFERENCE_STALENESS

Invalidation retires bases downward only; no upward retirement on markup materialization, allowing bases to persist years into markup and early-tier detection to fire at stale-reference extension. Evidence anchor: prediction ID `08c114b2a28e59cc3fbc1e27848e6b40`, SANAM 2023-11-16, base `SANAM::2021-06-23::BASE01`, extension 1.847457627118644.

### PILOT_REENTRY_UNGUARDED

`EXECUTE_EARLY_PILOT` path lacks a same-symbol position-open guard; daily re-pilots occur while a pilot is already open and materially drive failed-pilot count. Evidence anchors include MABANEE `8c3a928c78778973a079c094ab6b8efa` 2021-10-11, `4772bec4b95230653938714fdcc01a5e` 2021-10-12, `ffcfd3bbeda22b414b27cf785b71986a` 2021-10-13; BPCC `16529355c4c62c308de475e7c9036abb` 2021-10-12 and `4eba1171414cd9cbf5e8243a5e9a42af` 2021-11-14.