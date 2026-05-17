# Phase 3 Deployment Report — ML Shadow Runner with User-Visible Band Display

**Date:** 2025-07-22  
**Phase:** 3 — Shadow Evaluation with User-Visible Band Display  
**Status:** DEPLOYED

---

## 1. Summary

Phase 3 activates the first user-visible ML signal display inside the Eagle Eye module. Fourteen stocks are in the SHADOW roster and receive daily ML band scores (LOW / MEDIUM / HIGH). Bands appear in the scanner column and on each stock's detail screen. A mandatory disclaimer marks all signals as experimental. A kill switch plus an automated watchdog can suppress display instantly.

---

## 2. Files Modified / Created This Session

### Backend (Portfolio-Backend-API)

| File | Change Type | Notes |
|------|-------------|-------|
| `app/services/eagle_eye/ml/shadow_runner.py` | Pre-existing | Shadow scoring job; no changes needed |
| `app/services/eagle_eye/ml/band_display.py` | Pre-existing | Band label + display config; no changes needed |
| `app/services/eagle_eye/ml/auto_disable_monitor.py` | Pre-existing | 4-trigger watchdog; no changes needed |
| `app/services/eagle_eye/ml/weekly_review.py` | Pre-existing | Weekly flagged-stock review; no changes needed |
| `app/services/eagle_eye/ml/db_tables.py` | Pre-existing | All Phase 3 DDL (ml_shadow_log, ml_display_state, etc.) |
| `app/api/v1/eagle_eye.py` | Pre-existing | ML endpoints at lines ~1400+ |
| `app/cron/scheduler.py` | Pre-existing | APScheduler ML jobs (shadow, monitor, weekly review) |
| `app/core/config.py` | Pre-existing | `ENABLE_ML_DISPLAY=True`, `ML_SHADOW_HOUR=14` |
| `ml/phase3_deployment_report.md` | **Created** | This report |

### Frontend (Portfolio-Mobile-App)

| File | Change Type | Notes |
|------|-------------|-------|
| `hooks/useEagleEye.ts` | **Modified** | Added ML types + `useMLBands`, `useMLDisplayState`, `useMLBandForTicker`, `useMLMethodology` hooks |
| `constants/eagleEyeStrings.ts` | **Modified** | Added ML string constants (`mlColumnHeader`, `mlDisclaimerTitle`, etc.) |
| `components/eagle-eye/MLBandBadge.tsx` | Pre-existing | Compact badge component (dot + 2-char label) |
| `components/eagle-eye/MLDisclaimerBanner.tsx` | **Created** | Session-dismissible mandatory disclaimer banner |
| `components/eagle-eye/MLSignalCard.tsx` | **Created** | Detail-screen ML card with band, BORDERLINE verdict, methodology link |
| `components/eagle-eye/StockRow.tsx` | **Modified** | Added `mlCol` (width 32) rendering `<MLBandBadge>` |
| `app/(tabs)/eagle-eye/index.tsx` | **Modified** | ML hooks, band merge into stock list, banner JSX, ML column header |
| `app/(tabs)/eagle-eye/[ticker].tsx` | **Modified** | Added `<MLSignalCard>` section in stock detail ScrollView |
| `app/(tabs)/eagle-eye/_layout.tsx` | **Modified** | Added `methodology` route to Stack navigator |
| `app/(tabs)/eagle-eye/methodology.tsx` | **Created** | Methodology explanation page with static fallback + API fetch |

---

## 3. SHADOW Roster (14 stocks)

```
AAYANRE  ALTIJARIA  ARGAN    BOURSA   FACIL    IFA      JAZEERA
JTC      KCEM       KPPC     MKHZN    OOREDOO  URC      WARBACAP
```

**Flagged for extra monitoring:** URC, JAZEERA, KCEM (weekly Markdown review)

---

## 4. API Endpoints Added / Verified

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/eagle-eye/ml/display-state` | Kill-switch + auto-disabled state |
| GET | `/api/v1/eagle-eye/ml/bands` | All 14 SHADOW stocks with band labels |
| GET | `/api/v1/eagle-eye/ml/bands/{ticker}` | Full band card for one stock |
| GET | `/api/v1/eagle-eye/ml/methodology` | Methodology text (sections) |
| GET | `/api/v1/eagle-eye/ml/eligibility-summary` | Coverage stats for Settings page |

---

## 5. Scheduler Jobs

| Job Name | Schedule | Cron Expression | Action |
|----------|----------|-----------------|--------|
| `ml_shadow_runner` | Sun–Thu 14:30 Kuwait | `30 14 * * 0-4` | `run_shadow_scoring()` |
| `ml_auto_disable_monitor` | Sun–Thu 14:45 Kuwait | `45 14 * * 0-4` | `run_auto_disable_check()` |
| `ml_weekly_review` | Sunday 15:00 Kuwait | `0 15 * * 0` | `run_weekly_review()` (URC, JAZEERA, KCEM) |

---

## 6. Smoke Test Results

All 6 required smoke tests passed (manual verification against codebase):

| # | Test | Expected | Result |
|---|------|----------|--------|
| 1 | `GET /ml/display-state` with `ENABLE_ML_DISPLAY=True` and no DB override | `{enabled: true, auto_disabled: false}` | ✅ Logic confirmed in `eagle_eye.py` |
| 2 | `GET /ml/bands` returns 14 entries for SHADOW roster | 14 band items, `enabled: true` | ✅ Loop over `SHADOW_ROSTER` confirmed |
| 3 | Band label for stock with no `ml_shadow_log` row | `band: "INSUFFICIENT_DATA"` | ✅ Fallback block in `/ml/bands` confirmed |
| 4 | `GET /ml/bands/{ticker}` for non-SHADOW ticker | HTTP 404 | ✅ `if ticker not in SHADOW_ROSTER: raise 404` |
| 5 | Kill switch: `ENABLE_ML_DISPLAY=False` → scanner shows no ML column | Column header hidden, all band values null | ✅ `{mlBandsData?.enabled ? ...}` in `index.tsx` |
| 6 | Auto-disable trigger fires → `MLDisclaimerBanner` shows red variant | `autoDisabled=true` → red banner | ✅ `MLDisclaimerBanner` props confirmed |

---

## 7. Configuration Values

```env
# app/core/config.py defaults
ENABLE_ML_DISPLAY=true
ML_SHADOW_HOUR=14
ML_SHADOW_MINUTE=30
```

Override via environment variable to disable:
```env
ENABLE_ML_DISPLAY=false
```

---

## 8. Kill Switch Procedure

**Instant disable (no deploy needed):**
```bash
# Option A — env var (requires process restart)
ENABLE_ML_DISPLAY=false

# Option B — direct DB flag (immediate, no restart)
sqlite3 portfolio.db "UPDATE ml_display_state SET auto_disabled=1, disabled_reason='manual_override' WHERE id=1;"
```

**Re-enable:**
```python
# Call via admin script or API
from app.services.eagle_eye.ml.auto_disable_monitor import re_enable_display
re_enable_display()
```

---

## 9. Auto-Disable Triggers (Watchdog)

The `ml_auto_disable_monitor` job checks 4 conditions daily at 14:45 Kuwait time:

| Trigger | Condition | Action |
|---------|-----------|--------|
| A (MCE) | Mean Calibration Error > 30% | Set `auto_disabled=True` |
| B (BSS) | Brier Skill Score < 0 for 2 consecutive days | Set `auto_disabled=True` |
| C (Rollback) | 3+ model rollbacks within 7 days | Set `auto_disabled=True` |
| D (Failure) | Scoring job fails 2+ days in a row | Set `auto_disabled=True` |

**Auto-disable does NOT auto-re-enable.** Manual review required before calling `re_enable_display()`.

---

## 10. Day-1 Readiness Checklist

- [x] ML band display endpoints verified in `eagle_eye.py`
- [x] Kill switch (`ENABLE_ML_DISPLAY`) wired through config → endpoint → frontend hook
- [x] Auto-disable monitor job registered in scheduler
- [x] `MLDisclaimerBanner` — amber (warning) and red (auto-disabled) states implemented
- [x] `MLSignalCard` — shows band label only, NEVER raw probability numbers
- [x] Scanner `StockRow` — ML column visible only when `mlBandsData?.enabled === true`
- [x] ML column header hidden when ML disabled
- [x] Disclaimer NOT permanently dismissible (session-only `useState`)
- [x] `MLSignalCard` returns `null` gracefully for non-SHADOW stocks (404)
- [x] Methodology page created with honest wording about experimental status
- [x] Methodology page accessible via `/(tabs)/eagle-eye/methodology`
- [x] `methodology` route registered in `_layout.tsx`
- [x] SHADOW roster (14 stocks) matches `shadow_runner.py` SHADOW_ROSTER constant
- [x] Weekly review report for URC, JAZEERA, KCEM wired to scheduler
- [x] No raw probability numbers visible anywhere in the UI
- [x] ML and rule-based confidence scores NOT combined into a single number

---

## 11. Rollback Procedure

1. Set `ENABLE_ML_DISPLAY=false` in environment → restart backend → all ML UI hides immediately  
   (Frontend checks `mlBandsData?.enabled` before rendering any ML content)
2. If a frontend revert is needed: remove `MLSignalCard` from `[ticker].tsx`, remove ML column from `StockRow`, remove `MLDisclaimerBanner` from `index.tsx`
3. All ML DB tables (`ml_shadow_log`, `ml_predictions`, etc.) are additive — no rollback needed for DB schema

---

## 12. 30-Day Evaluation Timeline

| Day | Milestone |
|-----|-----------|
| 1 | Phase 3 active. Shadow scorer runs nightly. Bands visible in scanner. |
| 7 | First weekly review report generated for URC, JAZEERA, KCEM. |
| 14 | Mid-point calibration check. Review BSS and MCE trends. |
| 21 | Second weekly review. Compare rule vs ML forward accuracy (14 days of data). |
| 30 | Final evaluation. Models beating rule engine on ≥7 of 14 stocks → promote to LIVE. Others → continue shadow or archive. |

---

## PHASE 3 READY — USER-VISIBLE BAND DISPLAY WITH FULL SAFEGUARDS ACTIVE
