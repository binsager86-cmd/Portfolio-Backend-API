# R14-F Module (f) AvoidAuthorityPlane v1

- RUN_NONCE: 2026-07-18T09:34:56.0297303Z
- Freeze v2 byte-match: True
- Acceptance: PASS
- Module (e) closure scope note: Module (e) evidence exercises entry, holding, suppression, and avoid-veto lifecycle; exit lifecycle is out of scope and untested; positions opened in replay never close.
- Source rule: close < sma200 and sma200_slope < 0 and ema10 < ema30; clear via reclaim/20-session fallback
- Interval note: MABANEE boundary dates 2025-02-20 and 2025-05-18 registered; no tuning.
- Module (g): BLOCKED_PENDING_MODULE_F_REVIEW.

## Counts
{
  "MABANEE": {
    "avoid_days": 77,
    "mismatches": 0,
    "rows": 138
  },
  "SANAM": {
    "avoid_days": 0,
    "mismatches": 0,
    "rows": 140
  },
  "TIJARA": {
    "avoid_days": 0,
    "mismatches": 0,
    "rows": 371
  }
}

## Mismatches
{
  "MABANEE": [],
  "SANAM": [],
  "TIJARA": []
}
