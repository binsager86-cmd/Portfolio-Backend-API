# R14B Parameter Freeze v2 Amendment 2

Amendment ID: `R14B_PARAMETER_FREEZE_V2_AMENDMENT_2`

Status: `OWNER_RATIFIED_BY_DELEGATION`

Append-only: true

Amends: `R14B_PARAMETER_FREEZE_V2`

## Ratified Parameter

`UPWARD_RETIREMENT_MFE_THRESHOLD = 0.20`

A base is `RETIRED_SUPERSEDED_BY_MARKUP` when MFE from `base_high_ref` is greater than or equal to `+0.20` within 120 sessions of markup onset.

Retirement is permanent.

The materialization definition is a single shared definition with the forward grader and the frozen materialization criterion.

## Evidence Citation

Source: `r15rem_upward_retirement_evidence_v2`

- `196/304` stale early executions suppressed at all candidate thresholds.
- Extension p75 reduced from `2.5084745762711864` to `0.017955801104972375`.
- Suppression concentrated in `SANAM=146` and `TIJARA=50`.
- Re-formation `2/2`, median approximately `62` sessions.

## Conduct

- Canonical surface unchanged.
- Set B quarantine reaffirmed.
- Extension mode: append-only.