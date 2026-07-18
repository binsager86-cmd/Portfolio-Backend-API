# R14 Design Spec CONSOLIDATED v2.2 Addendum 1

## Module (e) Confirmed Direct Entry

Status: owner-authorized module-code remediation.

Authority: append-only addendum to `r14_design_spec_CONSOLIDATED_v2_2`.

Motivating defect record: `r14e_module_e_test_evidence_v5.json` recorded 65 of 65 `INTENT_FORMED` rows across SANAM and TIJARA falling through to `execution_state=NONE`. The observed condition was `INTENT_FORMED AND BASE_VALID AND CONFIRMED` with no veto, while `LifecycleIntentRouter` only opened the deferred/early path when `not confirmed`.

### CONFIRMED_DIRECT_ENTRY

Trigger:

- `candidate_intent.intent_state == INTENT_FORMED`
- `base_state in {BASE_VALID, BASE_FROZEN}`
- `confirmation_state == CONFIRMED`
- no active deferred intent already exists
- no veto is active

Precedence:

- Existing deferred/early/scale paths remain unchanged.
- If a deferred intent is already active, the deferred path takes precedence and direct entry must not be selected.

Action:

- emit `execution_state = EXECUTE_CONFIRMED_DIRECT`
- emit `entry_tier = BREAKOUT_CONFIRMED_ENTRY`
- full target entry, represented by `target_fraction = 1.0`
- `pilot_size_fraction = 0.0`; no pilot is created
- participation cap remains `0.10`
- chase advisory remains evaluated from extension percentage
- no 60-session time-stop applies to the confirmed-direct position
- confirmed-direct positions are governed by invalidation and exit lifecycle semantics

Non-goals:

- Do not alter deferred intent formation.
- Do not alter early pilot sizing.
- Do not alter confirmed add scaling from an existing deferred/early state.
- Do not tune avoid logic to match historical records.