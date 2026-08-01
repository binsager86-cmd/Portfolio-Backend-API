# EAGLE EYE - PREVIEW-1A (Revised After PRE-START REVIEW Corrections)

Status: EXECUTION HELD. Replay unauthorized.
Authority: HISTORICAL_REPLAY_NON_CANONICAL diagnostic only.
Canonical obligations unchanged: CRD-0 and CG-A remain active.

## 0. Hard hold

No replay, no engine changes, no tuning, no database population, and no benchmark approval actions are authorized by this document.

## 1. Deterministic baseline matrix (frozen engine evidence)

Canonical deterministic-engine test inventory for PREVIEW-1A baseline:
- tests/integration/test_eagle_eye_regression.py
- tests/unit/test_auto_disable_monitor.py
- tests/unit/test_candidate_v2_atomicity.py
- tests/unit/test_candidate_v2_raw_adjusted_separation.py
- tests/unit/test_eagle_eye_adapter.py
- tests/unit/test_eagle_eye_audit_service.py
- tests/unit/test_eagle_eye_indicator_service.py
- tests/unit/test_eagle_eye_phase_e.py
- tests/unit/test_eagle_eye_r9_ingest.py
- tests/unit/test_eagle_eye_store.py
- tests/unit/test_r11_ingest_guards.py

Evidence files:
- canonical test-file inventory: artifacts/preview1a_prestart/corrections/deterministic_engine_test_files.txt
- collected node-ID inventory (73, exit 0): artifacts/preview1a_prestart/corrections/deterministic_nodeids_only.txt
- collect raw output: artifacts/preview1a_prestart/corrections/deterministic_collect_output.txt
- collect exit code: artifacts/preview1a_prestart/corrections/deterministic_collect_exit_code.txt
- baseline run raw terminal output: artifacts/preview1a_prestart/corrections/deterministic_baseline_output.txt
- JUnit XML: artifacts/preview1a_prestart/corrections/deterministic_baseline_junit.xml
- baseline run exit code: artifacts/preview1a_prestart/corrections/deterministic_baseline_exit_code.txt
- deterministic test-file hashes: artifacts/preview1a_prestart/corrections/deterministic_test_file_hashes.txt
- deterministic engine-file hashes: artifacts/preview1a_prestart/corrections/engine_file_hashes_current.txt
- expected versus actual: artifacts/preview1a_prestart/corrections/expected_vs_actual.json

## 2. Collection-error resolution

Full-repo collect-only remains non-zero and is documented only as forensic input:
- full collect output: artifacts/preview1a_prestart/corrections/full_collect_output.txt
- full collect exit code: artifacts/preview1a_prestart/corrections/full_collect_exit_code.txt

Intentional exclusions and binding reasons (for deterministic baseline scope) are frozen in:
- artifacts/preview1a_prestart/corrections/intentional_exclusions.json

Rule: deterministic baseline collection MUST remain exit code 0 and zero collection errors.

## 3. Dirty-tree forensic identity (no cleanup applied)

Required forensic outputs:
- full git status: artifacts/preview1a_prestart/corrections/git_status_full.txt
- complete git diff: artifacts/preview1a_prestart/corrections/git_diff_full.patch
- untracked inventory: artifacts/preview1a_prestart/corrections/git_untracked_files.txt
- diff SHA-256: artifacts/preview1a_prestart/corrections/git_diff_full.patch.sha256
- dirty file classification (every entry): artifacts/preview1a_prestart/corrections/dirty_tree_classification.json
- deterministic-engine diff proof versus baseline commit: artifacts/preview1a_prestart/corrections/dirty_tree_classification.json
- proposed immutable baseline snapshot identity: artifacts/preview1a_prestart/corrections/proposed_immutable_baseline_snapshot_identity.json

## 4. Milestone registry governance state

All records are explicitly stamped:
PROVISIONAL_SOURCE_UNAPPROVED_CANDIDATE_V2

Registry artifacts:
- JSON: artifacts/preview1a_prestart/proposed_milestone_registry_v1.json
- CSV: artifacts/preview1a_prestart/proposed_milestone_registry_v1.csv
- registry summary + SHA-256: artifacts/preview1a_prestart/prestart_registry_summary.json

No milestone is owner-approved by generation alone.

## 5. SANAM milestone restriction (unapproved)

No adjusted-series approval and no inferred factor are permitted.

SANAM CA status artifact:
- artifacts/preview1a_prestart/corrections/sanam_milestone_ca_status.json

This artifact returns:
- which milestones cross unresolved CA windows
- RAW_DIAGNOSTIC_ONLY vs PIT_INVALID_CA_UNRESOLVED per milestone
- proposed corporate-action-safe cycle windows (pre-CA and post-CA+200 sessions)

## 6. Real disposable replay DB specification (not prepared, not populated)

Replay DB specification is declared in manifest and remains pending authorization:
- schema/migration source: candidate_v2 schema blueprint only, instantiated on disposable DB at execution time
- dataset source: TickerChart ingest campaign for authorized symbols only
- dataset_id: generated once at execution start; immutable thereafter
- lineage manifest: required (request hash, payload hash, commit, parser/runtime versions, run IDs)
- input immutability: append-only raw, no overwrite; all corrections via new version records
- preparation procedure: create DB -> apply schema -> integrity checks -> ingest campaign -> post-prep integrity checks
- post-prep integrity tests: sqlite integrity, duplicate guard, OHLC validity, lineage completeness, orphan-run absence
- final SHA-256: computed only after preparation completes

No preparation or population performed yet.

## 7. Explicit execution assumptions (locked pre-run)

Full formulas and numeric values are frozen in:
- artifacts/preview1a_prestart/corrections/execution_assumptions_preview1a_v1.json

Includes explicit definitions for:
- signal timing
- entry and exit execution price formulas
- spread and slippage values
- liquidity participation formula and limits
- partial-fill policy
- suspension handling
- price-limit handling
- missing next-session price handling
- corporate-action-invalid period handling

## 8. Set C owner-approval evidence

Set C proposal and evidence artifacts:
- proposal: artifacts/preview1a_prestart/set_c_symbol_proposal_v1.json
- proposal SHA-256 sidecar: artifacts/preview1a_prestart/set_c_symbol_proposal_v1.json.sha256
- owner approval evidence package: artifacts/preview1a_prestart/corrections/set_c_owner_approval_evidence.json

For each Set C symbol the evidence package includes:
- evaluation window
- category
- objective selection reason
- data-series version
- corporate-action status
- data evidence rows
- explicit confirmation that no engine output was viewed during selection

## 9. Governance label

Every PREVIEW-1A output must carry:
HISTORICAL_REPLAY_NON_CANONICAL

PREVIEW-1A cannot be represented as live prediction, profitability proof, statistical validation, production readiness, or proof of edge. R12 remains exclusive for profitability/edge.

## 10. Start condition

Replay remains blocked until owner explicitly approves this revised document and the associated pre-start manifest.
