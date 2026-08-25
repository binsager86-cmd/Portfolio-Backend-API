# Reconciliation Report (Read-Only)

Date: 2026-08-02
Scope: Post-SIM-APP-2 provenance and untouchables verification
Mode: Read-only (no code edits, no commits)

## 1) Intervening Commits Since Last-Known SIM-APP-2 Anchors

Anchors used:
- backend-api: 94254f6
- mobile-app: c725612

### backend-api range: 94254f6..HEAD
- 17ae3dc | fix(fundamental): add lazy metrics ensure endpoint | author: binsager86-cmd | date: 2026-08-01 14:58:53 +0300

### mobile-app range: c725612..HEAD
- 28dbfc1 | fix(fundamental): lazy metrics ensure and period consistency | author: binsager86-cmd | date: 2026-08-01 14:59:58 +0300

## 2) Per-Commit File Lists and Sensitive-Touch Flags

Sensitive patterns checked:
- simulator/* code
- sealed_imports
- projection
- ledger.py
- frozen candidate files
- MANIFEST.json

### Commit 17ae3dc (backend-api)
Files:
- app/api/v1/fundamental_legacy.py
- tests/unit/test_fundamental_metrics_ensure.py

Sensitive-touch flag: NO

### Commit 28dbfc1 (mobile-app)
Files:
- README.md
- app/(tabs)/fundamental-analysis.tsx
- docs/fundamental-statement-period-rules.md
- services/api/analytics/metrics.ts
- src/features/fundamental-analysis/components/ComparisonPanel.native.tsx
- src/features/fundamental-analysis/components/ComparisonPanel.tsx
- src/features/fundamental-analysis/components/MetricsPanel.tsx
- src/features/fundamental-analysis/components/StatementsPanel.tsx
- src/features/fundamental-analysis/hooks/useStatementsTableState.ts
- src/features/fundamental-analysis/metricCalculation.ts
- src/features/fundamental-analysis/statementPeriodSelection.ts

Sensitive-touch flag: NO

## 3) Origin Classification

Assessment:
- Intervening commits are not recognizable as SIM-APP-2 continuation by message/content.
- They appear to be fundamental-analysis fixes.

Classification:
- Unknown origin relative to SIM-APP-2 chain (not identified as prior SIM-APP-2 commits).

## 4) Untouchables Verification

### Frozen candidate hashes vs freeze v3 registry
- Candidate state machine hash:
  - actual: d16afb2ffa7faf80dfe2ad3d64034403589c7a21ed35b0fd09bd958954cf2eeb
  - expected prefix: d16afb2f...
  - result: MATCH

- Candidate harness hash:
  - actual: 968625754efd1deb35259bc749ad583e2514e33efe46205186351a9692be1eee
  - expected prefix: 96862575...
  - result: MATCH

### Live ledger
- ledger path: F:\eagle_eye_archive\simulator\ee_sim_ledger.db
- current ledger SHA256:
  - 91d0808f1f8ec823c0a76e92c8efd696b8cd48f825c7572c91f8bcea3a400a0f

Row counts:
- transactions: 0
- daily_valuations: 0
- decision_log: 0
- guard_trips: 0
- monthly_hashes: 0

## 5) Verdict

Verdict: ANOMALY

Reason:
- Intervening commits are unrecognized relative to SIM-APP-2 chain.
- Frozen hashes are intact and no sensitive simulator/freeze files were touched in the intervening commits.

Recommendation:
1. Owner review/confirm the two intervening fundamental commits as intentional.
2. If confirmed intentional, continuation can proceed from current HEAD with a fresh SIM-APP-2 baseline note.
3. If not intentional, stop continuation and reconcile branch history first.

Escalation rule check:
- Frozen-hash drift trigger NOT hit (no hash mismatch).
