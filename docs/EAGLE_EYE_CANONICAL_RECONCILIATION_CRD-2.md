# EAGLE EYE — CANONICAL RECONCILIATION DIRECTIVE v2 (CRD-2)

## Project constitution. Supersedes CRD-1 and all prior chains. Owner approval activates it.

## Sequence: CRD-0 -> R11-CG-A -> R11-CC -> R12 -> R13 -> separate implementation CRs. R11-CG-B (production) runs parallel, gates deployment only.

---

## 0. Precedence & conflict resolution

1. This CRD-2.
2. External reviewer's R11 standards (data engineering).
3. Master Implementation Directive (long-term roadmap only; no license to modify the current engine).
4. P1-P7/Phase E/R8-R10B chain (authority on the frozen engine and its defect fences).

Taxonomy conflicts (13-state vs 8-state vs current) are frozen until the R13 mapping CR.

## 0A. Evidence hierarchy (binding on all adjudications)

1. Official Boursa Kuwait records and company announcements.
2. Trusted vendor raw payloads.
3. Reproducible internal calculations.
4. Owner chart/market-knowledge adjudication.
5. Model inference.

The owner is the final APPROVAL authority on every gate, and the primary audit layer, but does not substitute for official evidence on corporate-action terms, adjustment factors, or trading-calendar facts.

## 0B. Database naming (mandatory vocabulary)

forensic_contaminated_db (frozen evidence, never unfrozen/copied) · clean_candidate_db (R11-CC target) · production_db (DigitalOcean-deployed only; a local app DB is NOT "production") · test_temp_db (per-session disposable).

## 1. Frozen assets

- Deterministic v2 engine (scanner/entry/exit/risk/rating), thresholds, evaluation order.
- Baseline identity is pinned by artifact, not test count: git commit + dirty-tree status, engine-file inventory with SHA-256 per file, canonical test node-ID inventory with test-file hashes, machine-generated JUnit result, expected outcome. "47 tests"/"54 tests" are descriptive labels only. Producing this pinning manifest is the first task of CRD-0.
- forensic_contaminated_db and all forensic artifacts.
- MABANEE as FULL-LIFECYCLE benchmark (entry AND exit), per owner's A3 correction; Master Directive wording overridden.
- Constitution: no tuning to pass tests; verbatim machine-generated evidence; abstention/NO_TRADE always valid; failures reported honestly; evidence hierarchy §0A.

## CRD-0 — PRE-GATE: Unauthorized Candidate V2 Containment (evidence-only; no corrective implementation)

The Candidate V2 work created after the freeze acknowledgment is neither accepted nor rejected by this directive.

1. Preserve all Candidate V2 code/tests/DBs/artifacts unmodified.
2. Record: current commit, git status, complete diff vs last owner-approved baseline commit (identified and hashed), DB SHA-256s, source-file SHA-256s, artifact inventory.
3. Isolate on branch/patch-set r11_candidate_v2_unapproved.
4. Produce the frozen-baseline pinning manifest (§1).
5. Review each component against R11-CG-A standards; adjudicate ADOPT / ADOPT_WITH_AMENDMENT / REJECT / DEFER with rationale.
6. Passing tests do not confer adoption.
7. No destructive reset, deletion, or merge without owner approval.

Deliver the adjudication table; STOP.

## R11-CG-A — Local Engineering Closure (blocks R11-CC)

1. Atomic ingestion: per-batch transaction (run record, OHLCV, lineage, conflicts, quarantines); injected failures at 25/50/90% leave zero partial rows of any kind + durable failure audit. Compensating cleanup ≠ atomicity.
2. Raw/adjusted separation: immutable append-only ee_ohlcv_raw; versioned ee_ohlcv_adjusted linked to source raw rows and the corporate-action ledger (event terms from official records per §0A, source doc, factor, reviewer, effective date). Tests: coexistence; raw immutability under rebuild; new CA version -> new adjusted version; explicit stream selection by consumers.
3. Indicator stream policy: every indicator record carries source_series_type, adjustment_version, calculation_version, as_of_date; no silent adjustment-version changes.
4. Isolation: test/debug physically cannot resolve production_db regardless of flags; guards at app AND database layer; CI/pytest hard-fail on production paths; break-glass access short-lived, audited, read-only by default.
5. Ten guard cases (real-symbol synthetic/debug/CSV rejection incl. with debug opt-in; migration-script guard; trusted-row precedence over fixture rows; rejected-overwrite audit event; idempotent trusted re-ingest; conflicting-trusted versioning/quarantine; raw/adjusted non-overwrite; atomic rollback) with machine-generated evidence: --junitxml artifact, raw terminal output, exit code, commit hash, test-file and guard-module SHA-256s, Python/OS versions. No hand-written pass counts.
6. Lineage on every bar: environment, dataset_id, source_vendor/type, ingestion_run_id, ingested_at, request_parameters_hash, raw_payload_hash, code_commit, adjustment_status, quality_status. No cross-source/environment overwrite on bare (symbol, trade_date).
7. Namespace proof: TST_/DBG_ fixtures only; repo scan report listing every remaining real-ticker occurrence and why it is safe (docs/forensics/case-study code exempt); fixtures rejected if symbol exists in the security master; no executable test/debug path can load synthetic data under TIJARA/BPCC/ZAIN/SANAM/MABANEE.
8. Contamination causation: exact writing process, DB path, run identified; leak reproduced in a disposable DB; guard demonstrably prevents recurrence.
9. BPCC fixture forensics: git history + hashes; verdict stated as confirmed cause or "unresolved"; never likely-as-proven.
10. Final local diff for review: schema, transactions, guards, conftest, source-priority rules, CSV-import path, migrations, SQLite+PostgreSQL, all tests. STOP.

## R11-CG-B — Production Runtime Closure (parallel; gates DEPLOYMENT only, never local research)

Deployed commit, environment name, DB engine, redacted destination/fingerprint, DATABASE_URL presence, persistent-volume mount, API/worker/scheduled-job DB configuration, per-process startup fingerprints, production rollback plan. Local .env files are not proof. No DigitalOcean action until CG-B passes AND owner approves.

## R11-CC — Clean Development Candidate (after CG-A)

1. New clean_candidate_db from migrations; nothing copied from any existing DB.
2. One controlled rebuild CAMPAIGN: one dataset_id, one parent campaign record, transactionally isolated child ingestion runs (per symbol/batch), complete reconciliation of every requested symbol, one frozen campaign manifest + dataset hash. Reproducible from manifest.
3. Event-level anomaly handling (anti-selection-bias): jumps/gaps/suspected CAs create EVENT records; trusted RAW bars remain stored unless unparseable or fundamentally invalid; an unresolved event may block adjusted-series approval, window-crossing indicators, and statistical eligibility for the affected interval; never auto-discard a symbol's history. Every symbol reconciled as raw_loaded_clean / raw_loaded_with_events / no_vendor_data / parse_failed / fully_unusable_with_reason.
4. Indicators recalculated from zero. Full census exported: duplicates, OHLC validity (low<=open,close<=high; non-negative fields; zero-volume classified), non-trading-day census, missing-session analysis, >20% close-change review against the CA ledger (limit context: +10%/-5%), lineage completeness, raw/adjusted check, warmup completeness.
5. Five case-study landmark table + point-in-time red-line benchmark snapshots + broad-market anomaly table. Owner anchor verification is the exit gate.

## R12 — Frozen Deterministic Baseline Examination (after owner verifies CC)

1. Current frozen engine vs clean_candidate_db: full point-in-time eligible universe (winners, failures, suspensions, delistings as data permits), realistic execution (next-executable-session entry, spread, costs, liquidity-dependent slippage, ±10%/-5% limit constraints, partial fills, capacity).
2. Benchmark parity: buy-and-hold, 20/60 breakout, MA trend, relative-volume breakout, equal-weight universe, random top-k; ALL with identical universe, as-of information, execution delay, cost/slippage model, liquidity constraints, capital, position limits, suspension/limit treatment. Random top-k over multiple fixed seeds with confidence intervals.
3. Metrics: net expectancy, profit factor, hit rate/payoff, drawdown, time under water, turnover, capacity, top-rank precision, probability calibration, regime/liquidity/year stability, symbol concentration, overfitting control.
4. Verdict handling: result preserved permanently as Baseline v2 verdict. No modifying the tested engine and rerunning against the same untouched examination period; no optimizing against observed failures. Descriptive failure analysis permitted. Later research requires: new version, documented hypothesis, train/dev periods separate from the locked exam period, a NEW untouched examination period, owner-approved CR. The original verdict stays visible.
5. Live ledger starts prospectively at activation; reconstructed predictions labeled historical_replay, never presented as issued-live.

## R13 — Proposals only

1. State-taxonomy mapping CR, inventory first: extract the authoritative existing state set from source enums, transition code, DB values, API schemas, tests, and historical prediction records; reconcile legacy names; produce usage counts; THEN propose old->new mapping. Any mapping sketched in prior documents is conceptual, not instruction. Owner resolves 13-state vs 8-state here; each genuinely new state arrives via its own rules+tests+trace CR.
2. Three-model architecture proposal (entry / continuation / exit-hazard) with Review-§9 economics labels (ATR targets, next-executable-session pricing), Review-§10 validation (anchored walk-forward, purging/embargo, ablations), Review-§8 feature families; absorbs the queued family_scores harvest. This is Phase F, upgraded.
3. T1/T2/T3 conditional models and the analog engine: proposals only, after the three-model core is approved.

## Prohibitions (scoped)

Until R13 completes AND a separate owner-approved ML Implementation CR exists: no ML training/deployment, no target-model or analog-model implementation, no lifecycle-logic changes, no threshold/weight changes. Never: unfreeze forensic_contaminated_db; rewrite historical predictions; deploy without CG-B + owner approval. Tuning proposals always follow the governance template (gap, N, significance, out-of-sample proof, version bump). Owner approval queue: CRs 2-6, CR-7 cancellation, CR-8 reconciliation against the clean candidate, SANAM 2023-09-05 adjudication, SANAM adjustment factor; all resolved via the CA ledger with official terms.

## Roles

Owner: sole approver; audit layer per §0A.
External reviewer docs: engineering standard + roadmap.
Claude: canonical author + adversarial reviewer.
Code agent: executes strictly in sequence, stops at every gate, pastes machine-generated evidence, and, after Candidate V2, is on notice that unauthorized work enters canon only through CRD-0-style adjudication.
