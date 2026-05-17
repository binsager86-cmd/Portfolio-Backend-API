# Eagle Eye — Production Migration Master Plan

> Reference document, not a single agent prompt. Lays out the full path from local SQLite + cached data to a production deployment on your existing VPS alongside the portfolio app. Each phase has its own prompt file written separately (Phase A.1 first, the rest written as you reach them).

---

## Context

- **Existing app:** portfolio app already runs in production on a self-hosted VPS with PostgreSQL
- **Eagle Eye current state:** all code lives in the portfolio app's backend (`mobile-migration/backend-api/app/services/eagle_eye/`) but uses local SQLite for caching
- **Goal:** migrate Eagle Eye's data layer to the production PostgreSQL, ship it as part of the existing app's deployment, add scheduled jobs and monitoring
- **Your ops experience:** limited — agent must be explicit and produce rollback-able changes

---

## Guiding Principles

1. **Never break what works.** The portfolio app is in production. Every change must be additive (new tables, new endpoints, new code paths) until verified. Only after verification does old code get removed.

2. **One phase at a time.** Each phase ends with the system in a working, deployable state. No half-migrations left mid-flight.

3. **Every change has a rollback.** If something breaks in production, the agent must have left a documented way back to the last working state.

4. **Production database is sacred.** No experimental scripts touch production data directly. All DB changes go through migration files reviewed before applying.

5. **Verify before promoting.** Each phase ends with verification on a staging copy or a feature-flagged code path, not a hot prod switch.

---

## Phase Overview (in order)

| Phase | Title | Estimated time | Reversible? |
|---|---|---|---|
| **A.1** | Database + Dependencies + Config (foundation) | 2-3 days | Yes |
| **A.2** | Schema migration system (Alembic) | 1-2 days | Yes |
| **A.3** | Data ingestion pipeline (production-safe) | 2-3 days | Yes |
| **A.4** | Model artifact storage and loading | 1-2 days | Yes |
| **A.5** | Scheduled jobs (nightly retrain, daily ingestion) | 2-3 days | Yes |
| **A.6** | Logging, monitoring, alerts | 2-3 days | Yes |
| **A.7** | Deployment automation (CI/CD or manual playbook) | 1-2 days | Yes |
| **A.8** | Staged rollout + feature flags | 1-2 days | Yes |

**Total realistic timeline:** 12-20 working days. Don't try to compress this. Each phase has acceptance criteria — if a phase isn't done, the next phase can't start.

---

## Phase A.1 — Database + Dependencies + Config (the foundation)

**Goal:** make the codebase able to run against production PostgreSQL with pinned dependencies and externalized config, without changing any runtime behavior.

**Scope:**
- Replace SQLite-specific SQL with portable SQL or use an abstraction layer
- Add `requirements.txt` with pinned versions for everything Eagle Eye and ML need
- Move all hardcoded paths and credentials to environment variables
- Add a connection layer that uses the existing portfolio-app PostgreSQL instance
- Add a local Postgres-via-Docker setup for development that matches production

**Out of scope:**
- Migrating data (Phase A.3)
- Schema migration tooling (Phase A.2)
- Scheduled jobs (Phase A.5)
- Monitoring (Phase A.6)
- Deployment (Phase A.7)

**Acceptance criteria:**
- All Eagle Eye and ML code runs against a local Postgres container with no behavior change
- `pip install -r requirements.txt` reproduces the exact environment used in development
- Zero hardcoded paths or credentials in the code
- Existing tests still pass against Postgres
- Rollback path: revert the commit, redeploy, system goes back to SQLite

**See:** `phase_a1_database_deps_config.md` (separate prompt file)

---

## Phase A.2 — Schema Migration System

**Goal:** track all database schema changes through Alembic migrations so they can be applied incrementally and rolled back.

**Scope:**
- Initialize Alembic in the backend-api repo if not already present
- Convert every `CREATE TABLE` in current `db_tables.py`, `signal_logger.py`, `move_detector.py`, etc. into Alembic migrations
- Generate a baseline migration that represents the current state
- Document the migration workflow (how to create a new migration, apply, rollback)

**Out of scope:**
- Actually running migrations on production (Phase A.7)
- Data migration from SQLite cache (Phase A.3)

**Acceptance criteria:**
- A fresh, empty Postgres database can be brought to current schema via `alembic upgrade head`
- A migration can be rolled back via `alembic downgrade -1` cleanly
- New tables added in code automatically prompt for a new migration during PR review
- Migration files are version-controlled
- Rollback path: roll back the migrations, schema returns to previous state

**Important:** Eagle Eye created ~20 tables across Phase 1 and 2 (ml_models, ml_predictions, considered_signals, etc.). Each needs a migration. Existing portfolio app tables should NOT be touched — only Eagle Eye tables get migrations in this phase.

---

## Phase A.3 — Data Ingestion Pipeline (production-safe)

**Goal:** make the daily data ingestion (TickerChart OHLCV, fundamentals, corporate events) work safely in production with retries, rate limits, and error handling.

**Scope:**
- Wrap TickerChart API calls with retry logic and exponential backoff
- Add rate limit handling (current code probably hammers the API; production needs to respect quotas)
- Add idempotency: re-running ingestion for the same day shouldn't duplicate rows
- Move from "fetch all on demand" to "incremental daily fetch"
- Add a one-time data backfill script that loads existing SQLite cache into production Postgres
- Handle network failures gracefully — partial ingestion shouldn't corrupt the DB

**Out of scope:**
- Scheduling the ingestion (Phase A.5)
- Alerting on ingestion failures (Phase A.6)

**Acceptance criteria:**
- Daily ingestion can be triggered manually and completes successfully on production-sized data
- If the API returns errors, the script logs and retries cleanly
- Re-running the same day's ingestion produces zero new rows (idempotent)
- The SQLite cache can be migrated to production Postgres without data loss
- Rollback path: revert ingestion code, drop newly added rows from a clean cutoff timestamp

---

## Phase A.4 — Model Artifact Storage and Loading

**Goal:** define where trained ML models, calibrators, and pattern indices live in production, and how the runtime loads them.

**Scope:**
- Choose a model storage location on the VPS (e.g., `/var/eagle_eye/models/`)
- Add a model loading layer that knows how to find the current LIVE/SHADOW model per stock
- Version-tag model artifacts (timestamp, training data range, model_id)
- Add a fallback: if a model fails to load, the stock automatically falls back to rule-based scoring
- Make sure the deployment process knows to upload model artifacts (or generate them on the production server)

**Out of scope:**
- Automated retraining (Phase A.5)
- S3 or cloud storage (deferred — local filesystem is fine for now given the VPS deployment)

**Acceptance criteria:**
- All trained models can be loaded by the runtime from the defined location
- A missing or corrupted model artifact doesn't crash the system — it falls back to rules
- Model versioning lets you know which model was used to score which prediction
- Rollback path: point the runtime back to the previous model version

---

## Phase A.5 — Scheduled Jobs

**Goal:** automate the daily ingestion, daily scoring, and periodic retraining.

**Scope:**
- Decide on scheduling mechanism (systemd timers vs cron — recommend systemd timers for better logging on a VPS)
- Schedule daily ingestion (typically early morning, after market close + overnight processing)
- Schedule daily scoring/rating recompute (after ingestion completes)
- Schedule monthly model retraining (per the Phase 2 brief)
- Schedule the forward-fill job for `considered_signals.realized_outcome_20d`
- Schedule the calibration health monitor (Phase 6 of original brief)
- Schedule auto-rollback monitor (per Section 0.10 of original brief)

**Out of scope:**
- Alerting on job failures (Phase A.6 covers this)

**Acceptance criteria:**
- Each scheduled job has a defined trigger, runtime, and expected duration
- A failed job is logged and creates a follow-up state — it doesn't silently skip the next day
- Jobs run in the correct order (ingestion → scoring → simulator update)
- Job runs are tracked in a `job_runs` table for audit
- Rollback path: disable the systemd timer, schedule reverts to manual

---

## Phase A.6 — Logging, Monitoring, Alerts

**Goal:** make sure you know when something goes wrong in production within hours, not days.

**Scope:**
- Centralized application logging (structured logs to a known location)
- Metrics on key things: number of stocks scored today, number of new ratings, number of SHADOW predictions, calibration error trend
- Alerts (email or Slack or Telegram) for: ingestion failure, scoring failure, model auto-rollback, calibration drift past warning threshold, scheduled job missing its window
- A status dashboard (can be a simple HTML page or admin endpoint) showing system health
- Database query for "is anything currently broken"

**Out of scope:**
- Full APM (application performance monitoring) — over-engineered for this stage
- User-facing notifications

**Acceptance criteria:**
- A simulated ingestion failure produces an alert within 5 minutes
- A status endpoint shows: last ingestion run, last scoring run, current model count by status, recent errors
- Logs are queryable for the last 30 days
- Rollback path: alerts can be disabled via env flag without code changes

---

## Phase A.7 — Deployment Automation

**Goal:** make deployments to production a known, scripted, reversible process.

**Scope:**
- Document the current deployment process for the portfolio app (the agent investigates and writes this down)
- Add Eagle Eye to the deployment flow without disrupting the existing app
- Create a deployment playbook: pre-deployment checklist, deployment steps, post-deployment verification
- If using Git-based deployment: branching strategy, PR requirements, deployment branch
- If using container-based: Dockerfile, build process, container registry

**Out of scope:**
- Full CI/CD with automated testing on every push — over-engineered for a solo developer at this stage

**Acceptance criteria:**
- A deployment can be triggered from a documented set of steps (manual or scripted)
- Deployment is reversible: previous version can be restored within 15 minutes
- Database migrations run as part of deployment (not manually after)
- The Eagle Eye deployment does NOT bring down the portfolio app

---

## Phase A.8 — Staged Rollout + Feature Flags

**Goal:** make sure new features (especially ML scoring) can be enabled gradually and disabled instantly if they cause problems.

**Scope:**
- Add a feature flag system (can be as simple as environment variables or DB-backed flags)
- Flag: `enable_ml_shadow_scoring` (off by default, turn on after Phase 2 + verification)
- Flag: `enable_eagle_eye_scanner_in_app` (controls whether the scanner is visible to users)
- Flag: `enable_simulator_in_app` (controls simulator visibility)
- Flag: `enable_auto_rollback_monitor` (controls whether auto-rollback fires)
- Document which flag controls what, and the procedure to flip each one

**Out of scope:**
- Per-user feature flags (over-engineered for this stage)

**Acceptance criteria:**
- Every major Eagle Eye component can be turned off via a flag without code changes
- Flag changes take effect within 5 minutes
- Default flag state is "off" for anything new and untested
- Rollback path: flip the flag off, problem disappears

---

## Decision Points Along the Way

After each phase, you have a decision point:

- **After Phase A.1:** Foundation is ready. Decide whether to continue or pause based on Phase 2 results.
- **After Phase A.3:** Production data ingestion is working. You could stop here and run Eagle Eye manually for a while.
- **After Phase A.5:** Scheduled jobs are running. You could stop here and add monitoring manually.
- **After Phase A.7:** System is fully deployed. You could defer Phase A.8 if not needed.

**The point:** you don't have to finish all 8 phases to derive value. Each phase ends with a working system at a higher level of production-readiness than before.

---

## What Could Go Wrong

Honest list of things that have killed production migrations like this before:

1. **Datetime timezone mismatches.** SQLite is naive about timezones. PostgreSQL forces you to be explicit. Some part of the codebase will assume UTC, another will assume local time, and ratings will be off by hours. Phase A.1 catches this.

2. **JSON column type differences.** SQLite stores JSON as TEXT. PostgreSQL has `JSON` and `JSONB`. If code reads/writes JSON columns assuming TEXT, it may break or perform poorly on Postgres. Phase A.1 catches this.

3. **Parquet file paths.** Your training matrices are stored as `.parquet` files in `ml_training_matrix/v1/{TICKER}/data.parquet`. These need to exist in production too. Either you ship them or regenerate them. Phase A.4 handles this.

4. **API credentials leak.** TickerChart credentials currently might be in a config file checked into git. Phase A.1 explicitly checks for and externalizes these.

5. **Postgres connection pool exhaustion.** SQLite is single-connection. Postgres has connection limits. If the code opens new connections per call, you'll exhaust the pool. Phase A.1 covers this.

6. **Migration ordering bugs.** Eagle Eye created tables over multiple phases. If migrations are out of order, the schema in production won't match dev. Phase A.2 catches this.

7. **Scheduled job overlap.** If a daily job takes longer than expected and the next day's run starts before the previous one finishes, you can get race conditions. Phase A.5 includes lock handling.

8. **Loss of model artifacts.** If a deployment doesn't preserve trained models, you lose all of Phase 2's work. Phase A.4 prevents this.

---

## Reading Order

1. Read this document fully — understand the scope before starting
2. Read `phase_a1_database_deps_config.md` (your next actionable prompt)
3. After Phase A.1 completes, the next phase prompt will be written based on what was learned
4. Each subsequent phase prompt is written when you're ready for it, not all upfront

This is the master plan. The Phase A.1 prompt is in a separate file.
