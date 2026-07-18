# R14 Design Spec v1.3

Supersedes: R14 Design Spec v1.2

Owner ruling:
- The architecture must catch the accumulation stage, not only confirmed breakouts.

Two-tier entry model:
- EARLY_ACCUMULATION_ENTRY
- BREAKOUT_CONFIRMED_ENTRY
- Early tier is gated by flow-confirmation predicates over the accumulation window plus base validity from AdaptiveBaseGeometry.
- Early tier does not use same-day volume multiples or trailing-liquidity vetoes as entry gates by design.

Staged position policy:
- EARLY_TIER_SIZE_FRACTION
- EARLY_TIER_PARTICIPATION_CAP
- EARLY_TIER_TIME_STOP
- SCALE_ON_CONFIRMATION

Authority rules:
- AvoidAuthorityPlane retains full veto over early entries.
- Early entries emit telemetry rows including flow evidence values at entry and DEAD_MONEY tracking.

Lifecycle wiring:
- Early entry and deferred intent are one mechanism at two trigger points.
- Base-reference ratcheting applies from early entry onward.

Updated R15 acceptance criteria:
- SANAM: ['The 2025-05-08 through 2025-05-21 window produces at least one confirmed entry event.', 'Width-rule dominance in the owner window drops below one-quarter of blocking rows.', 'The 2025-05-18 session specifically must produce BREAKOUT_CONFIRMED or an explicit, fully-telemetried veto naming its blocking term.', 'No stale original-freeze chase guard may block a day where the current valid reference has advanced through confirmed accumulation.', 'The 2025-05-18 session must produce a confirmed entry across all layers.', 'No Set A high-volume day may be blocked by trailing-liquidity alone when same-day value exceeds the execution-size parameter.', 'SANAM must generate an EARLY_ACCUMULATION_ENTRY within its owner-window accumulation before 2025-05-08.', 'The 2025-05-18 session must still produce a confirmed entry across all layers.', 'The early-tier false-positive cost must be reported separately as count and aggregate P&L for early entries that hit time-stop without confirmation.']
- TIJARA: ['PHASE_PROGRESSED_NO_CANDIDATE share over 2024-2026 falls below one-quarter of trading days.', 'At least one owner-window breakout cluster produces BREAKOUT_CONFIRMED or DEFERRED_INTENT instead of persistent BREAKOUT_WATCH stagnation.', 'At least one 2025 high-volume cluster must operate with an explicit valid base reference rather than unresolved M1 disarm.', 'TIJARA must generate an EARLY_ACCUMULATION_ENTRY before its markup onset.', 'The early-tier false-positive cost must be reported separately as count and aggregate P&L for early entries that hit time-stop without confirmation.']
- MABANEE: ['The decline interval remains avoided with zero long entries during the avoid-dominant regime.', 'Avoid veto authority remains sufficient to block downtrend participation.', 'MABANEE must generate zero EARLY_ACCUMULATION_ENTRY events during the avoid-dominant decline.']

False-positive cost:
- Count and aggregate P&L of early entries that hit time-stop without confirmation must be reported separately.

Finding-response map:
- EARLY_TIER: Structural answer to the pre-volume edge: expose accumulation-stage entries using flow confirmation and adaptive base validity before breakout confirmation.
- F1: Solved at confirmation tier; early tier intentionally bypasses same-day volume multiple dependence when flow confirmation and base validity are present.
- F9: Solved at confirmation tier; early tier uses current and arriving liquidity as participation context, not as a sole veto, while preserving optional participation caps.

Early-tier notes:
{
  "SANAM_owner_window_before_breakout": 99,
  "TIJARA_owner_window_before_markup": 102,
  "false_positive_cost_tracking": true,
  "m5_context": {
    "canonical_day": "SANAM 2025-05-18",
    "statement": "Trailing-window liquidity baseline lagged breakout-day liquidity and acted as sole veto on at least one high-volume Set A day.",
    "status": "CONFIRMED"
  },
  "source_artifacts": [
    "artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json",
    "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
    "artifacts/preview1a_prestart/review_final/r13_m5_liquidity_forensic_v1.json"
  ],
  "value_thesis": "Catch accumulation stage, not only confirmed breakouts."
}

R14-B and R15 remain NOT AUTHORIZED.
