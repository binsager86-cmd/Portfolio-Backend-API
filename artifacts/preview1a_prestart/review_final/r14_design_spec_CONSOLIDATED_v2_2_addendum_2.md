# R14 Design Spec CONSOLIDATED v2.2 Addendum 2

Status: owner-authorized R15 remediation cycle 1.

Authority: append-only addendum to `r14_design_spec_CONSOLIDATED_v2_2`.

## Pilot Re-Entry Guard

`EXECUTE_EARLY_PILOT` requires no open position for the same symbol.

While a position is open, intent days emit explicit suppression disposition `POSITION_ALREADY_OPEN_FEEDBACK_SUPPRESSED_PILOT`, mirroring the direct-path guard `POSITION_ALREADY_OPEN_FEEDBACK_SUPPRESSED_DIRECT`.

## Upward Base Retirement

A base is `RETIRED_SUPERSEDED_BY_MARKUP` when markup materializes from it, defined by the already-frozen materialization criterion: MFE >= +0.20 from `base_high_ref` within 120 sessions.

Retirement is permanent. New bases may then form at higher structure per existing geometry rules.

Parameter form: `UPWARD_RETIREMENT_MFE_THRESHOLD`.

Parameter value: `PENDING_GATE`.

Presumptive candidate: `0.20`.

Ratification status: pending owner ratification after Part C evidence.