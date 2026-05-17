# Phase A.1 — Database + Dependencies + Config

> Actionable prompt. Paste this to the agent after the Phase 2 run completes (or in parallel if the user explicitly opts in). Estimated 2-3 days of agent work. Foundation only — no deployment, no scheduling, no monitoring yet.

---

## Context for the Agent

The Eagle Eye system (`backend-api/app/services/eagle_eye/`) currently uses SQLite for all caching and ML state. The existing portfolio app already runs in production on a self-hosted VPS with PostgreSQL. The goal of this phase is to make Eagle Eye **production-ready at the code level** by:

1. Migrating database access to PostgreSQL
2. Pinning every dependency
3. Externalizing all configuration (paths, credentials, connection strings)
4. Creating a local development environment that matches production

**This phase does NOT migrate any data, deploy anything, or change runtime behavior.** It only makes the codebase capable of running against Postgres when configured to do so. Defaults remain backward-compatible.

The user has **limited operational experience** — be explicit, document every command, and provide rollback steps for everything.

---

## Hard Rules

1. **Do not break SQLite mode.** During this phase, the code must continue to work with SQLite if `DATABASE_URL=sqlite:///...` is set. Postgres becomes an option, not a forced switch. Backward compatibility through this phase, removable in Phase A.2.

2. **Do not migrate any data yet.** Empty production schema is fine for testing. Data migration is Phase A.3.

3. **Do not deploy anything to the actual production server.** All work is local. The production VPS is not touched in Phase A.1.

4. **Every secret stays out of git.** API keys, database passwords, connection strings — all in environment variables or a `.env` file (which is gitignored). If the agent finds any secret currently committed, that's a finding to surface — do not silently fix.

5. **No new abstractions just for fun.** This is a migration phase. We're not redesigning the database access layer. If something works in SQLite, port it to Postgres in the simplest way that maintains correctness.

6. **Every database change must be reversible.** No `DROP TABLE` operations. No `ALTER TABLE DROP COLUMN`. If a column is wrong, leave it and add a new one. (Phase A.2 will set up proper migration tooling; until then, use additive-only changes.)

7. **Test against a local Postgres container.** Do not connect to production Postgres for testing. Use Docker to spin up a local Postgres instance that mimics production.

---

## Step 1 — Inventory Current State

Before changing anything, produce a `phase_a1_baseline.md` document covering:

### 1.1 Database access points

Find every place in the codebase where SQLite-specific code exists:
- `app/core/database.py` — the main DB module
- Any direct `sqlite3.connect(...)` calls
- Any SQL with SQLite-specific syntax: `INSERT OR IGNORE`, `INSERT OR REPLACE`, `PRAGMA`, `AUTOINCREMENT`, `BLOB` handling
- Any hardcoded `.db` file paths

List each occurrence with file:line. This is the migration target list.

### 1.2 Hardcoded values

Find every hardcoded:
- File path (e.g., `/Users/.../portfolio_app/...`, `mobile-migration/...`, `ml_training_matrix/v1/...`)
- API credential (TickerChart username/password/token if present in source)
- URL or endpoint
- Magic number that looks like a config value

List with file:line.

### 1.3 Current dependencies

Generate a snapshot of the current Python environment:
```bash
pip freeze > phase_a1_current_packages.txt
```

Identify which packages are actually used by Eagle Eye code (vs. installed but unused). Use `pipreqs` or manual inspection.

### 1.4 Schema inventory

List every table that Eagle Eye creates:
- Tables defined in `ml/db_tables.py`
- Tables defined in `signal_logger.py`
- Tables defined in `move_detector.py`, `simulator.py`, or anywhere else
- Tables created implicitly by SQLAlchemy or similar

For each table: name, columns, primary key, indexes, current row count in your dev SQLite. This is the production schema target.

### 1.5 Identify SQLite-specific behavior

Specifically look for:
- Reliance on SQLite's permissive typing (e.g., storing strings in INTEGER columns)
- Use of `rowid` column (SQLite-specific)
- Lack of explicit timezone handling on datetime columns
- JSON stored as TEXT (should become JSONB in Postgres)
- Boolean stored as 0/1 INTEGER (should become BOOLEAN in Postgres)

Each finding becomes an item in a migration checklist.

**Output:** `phase_a1_baseline.md` with all sections above. Do not start coding changes until this exists and is reviewed.

---

## Step 2 — Set Up Local Postgres for Development

The user has limited ops experience. Make this dead simple.

### 2.1 Create a `docker-compose.yml` in the backend-api directory

```yaml
version: '3.8'
services:
  postgres-dev:
    image: postgres:16
    container_name: eagle_eye_postgres_dev
    environment:
      POSTGRES_USER: eagle_eye_dev
      POSTGRES_PASSWORD: eagle_eye_dev_password
      POSTGRES_DB: eagle_eye_dev
    ports:
      - "5433:5432"  # 5433 to avoid clash with any local Postgres on 5432
    volumes:
      - eagle_eye_pgdata:/var/lib/postgresql/data

volumes:
  eagle_eye_pgdata:
```

(Adjust if a `docker-compose.yml` already exists for the portfolio app — add a new service, don't overwrite.)

### 2.2 Write `dev_postgres_setup.md`

A short markdown document the user can follow:

```
# Local Postgres Development Setup

1. Install Docker Desktop if not already installed
2. From the backend-api directory:
   docker-compose up -d postgres-dev
3. To check it's running:
   docker ps  (should show eagle_eye_postgres_dev)
4. To stop:
   docker-compose stop postgres-dev
5. To reset (drops all data):
   docker-compose down -v postgres-dev
   docker-compose up -d postgres-dev
6. Connection string for the .env file:
   DATABASE_URL=postgresql://eagle_eye_dev:eagle_eye_dev_password@localhost:5433/eagle_eye_dev
```

This is what the user reads when they need to spin up Postgres locally.

---

## Step 3 — Externalize Configuration

### 3.1 Create `.env.example` (committed) and `.env` (gitignored)

`.env.example` shows the structure with placeholder values:
```
# Database
DATABASE_URL=sqlite:///./dev_portfolio.db
# DATABASE_URL=postgresql://eagle_eye_dev:eagle_eye_dev_password@localhost:5433/eagle_eye_dev

# TickerChart API
TICKERCHART_USERNAME=
TICKERCHART_PASSWORD=
TICKERCHART_API_KEY=

# File paths
ML_MATRIX_ROOT=./ml_training_matrix
ML_MODEL_ROOT=./ml_models
ML_REPORTS_ROOT=./reports

# Eagle Eye behavior
EAGLE_EYE_ENV=development  # development | staging | production
EAGLE_EYE_LOG_LEVEL=INFO
```

`.env` is the actual file (NOT committed) with real values.

Ensure `.env` is in `.gitignore`. If not, add it.

If any secret is currently committed in git history, **stop and report** — that's a security finding, not something to silently fix. The user needs to decide whether to rotate the credential.

### 3.2 Create `app/core/config.py`

A single module that loads environment variables once and exposes them as a settings object. Use `pydantic-settings` if available, otherwise plain `os.environ.get`.

```python
class Settings(BaseSettings):
    database_url: str
    tickerchart_username: str | None = None
    tickerchart_password: str | None = None
    tickerchart_api_key: str | None = None
    ml_matrix_root: str = "./ml_training_matrix"
    ml_model_root: str = "./ml_models"
    ml_reports_root: str = "./reports"
    eagle_eye_env: str = "development"
    eagle_eye_log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
```

### 3.3 Replace hardcoded values throughout the codebase

For every item found in Step 1.2, replace with a config reference:
- Hardcoded paths → `settings.ml_matrix_root` etc.
- Hardcoded credentials → `settings.tickerchart_username` etc.

Run grep to confirm no hardcoded paths remain.

---

## Step 4 — Database Access Layer Migration

### 4.1 Choose the approach

Two options:
- **Option 1:** SQLAlchemy (full ORM or just Core)
- **Option 2:** Raw `psycopg2` / `psycopg3` connections

Pick **whatever the existing portfolio app uses**. Eagle Eye should match the parent app's choice — don't introduce a new ORM if the app uses raw SQL, or vice versa.

If the portfolio app uses both or it's unclear, default to whatever's already in the Eagle Eye code (likely `sqlite3` calls). Wrap them with a common interface that can target either SQLite or Postgres based on `DATABASE_URL`.

### 4.2 Replace SQLite-specific SQL

For each finding from Step 1.5:

| SQLite | PostgreSQL |
|---|---|
| `INSERT OR IGNORE INTO ... VALUES (...)` | `INSERT INTO ... VALUES (...) ON CONFLICT DO NOTHING` |
| `INSERT OR REPLACE INTO ...` | `INSERT INTO ... ON CONFLICT (...) DO UPDATE SET ...` |
| `AUTOINCREMENT` | `SERIAL` or `BIGSERIAL` (or `GENERATED ALWAYS AS IDENTITY` in newer Postgres) |
| `?` parameter placeholders | `%s` (psycopg) or `$1` (asyncpg) — match the driver |
| `BLOB` | `BYTEA` |
| `REAL` | `DOUBLE PRECISION` |
| `INTEGER PRIMARY KEY` | `BIGSERIAL PRIMARY KEY` |
| `BOOLEAN as 0/1 INTEGER` | `BOOLEAN` |
| `JSON as TEXT` | `JSONB` |

Important: do NOT silently convert. For each table, the column types are likely sensitive (e.g., the `considered_signals.full_feature_snapshot_json` column needs to become JSONB, and code that reads it as a string will need adjustment). Make a checklist per table.

### 4.3 Datetime handling

Audit every datetime column. In Postgres, use `TIMESTAMP WITH TIME ZONE` (`TIMESTAMPTZ`) for any datetime that represents a real-world moment. Use `DATE` for date-only fields (e.g., `signal_date`, `event_date`).

Ensure all Python code creates timezone-aware datetimes (`datetime.now(timezone.utc)`) when writing to TIMESTAMPTZ columns. Naive datetimes silently shift in Postgres.

### 4.4 Make tables creatable in both dialects

Each table's `CREATE TABLE` statement needs to work against both SQLite and Postgres for the duration of Phase A.1 (Phase A.2 replaces this with Alembic migrations).

Either:
- Use a SQL dialect that's portable (mostly works but loses Postgres-specific types like JSONB)
- Or write two versions and select based on the DB driver in use

For simplicity, the agent can write a `db_tables_postgres.py` next to `db_tables.py` and dispatch based on `DATABASE_URL`. This is temporary — Phase A.2 replaces both with Alembic migrations.

---

## Step 5 — Pin Dependencies

### 5.1 Create / update `requirements.txt`

In the backend-api directory, create or update `requirements.txt` with:
- Every package currently used by Eagle Eye and the broader backend
- Each pinned to an exact version (e.g., `pandas==2.2.0`, not `pandas>=2.0`)
- Grouped by purpose with comments:

```
# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.25

# Eagle Eye core
pandas==2.2.0
numpy==1.26.3

# ML
lightgbm==4.3.0
scikit-learn==1.4.0
joblib==1.3.2

# Config
pydantic==2.5.3
pydantic-settings==2.1.0
python-dotenv==1.0.0

# ... (every dependency, exact versions)
```

### 5.2 Test reproducibility

In a fresh virtual environment:
```bash
python -m venv test_env
source test_env/bin/activate  # or .\test_env\Scripts\activate on Windows
pip install -r requirements.txt
```

Then run the Eagle Eye test suite. If anything fails because of a missing dependency, add it to `requirements.txt`.

### 5.3 Create `requirements-dev.txt` for dev-only packages

Things like `pytest`, `black`, `mypy`, anything needed for development but not production.

---

## Step 6 — Verify End to End

### 6.1 Schema verification

With local Postgres running:
1. Apply schema (Eagle Eye's table creation should work against an empty Postgres database)
2. Confirm every table is present
3. Confirm every index is present
4. Confirm every constraint is present

### 6.2 Smoke test the existing pipelines

Run, in order, against local Postgres:
1. The Phase 1 leakage audit tests (`tests/test_leakage_audit.py`)
2. The Phase 1 verification script
3. The Phase 2 smoke test on NBK (if Phase 2 has completed by now)
4. A single rating computation for one stock

If anything fails, fix the root cause — do not edit the test to pass.

### 6.3 Backward compatibility check

Switch `DATABASE_URL` back to SQLite. Re-run the same tests. All should still pass. This confirms backward compatibility is preserved.

### 6.4 Performance sanity check

For a single stock rating computation:
- SQLite mode: measure time
- Postgres mode: measure time

Postgres should not be more than 2x slower than SQLite on this workload. If it is, there's likely a query that's missing an index or doing N+1.

---

## Step 7 — Document and Hand Off

Write `phase_a1_completion_report.md` containing:

1. List of every file modified, with one-line description of why
2. The new env vars and their purpose
3. The schema diff (what changed between SQLite and Postgres versions)
4. Test results: every test that ran, with pass/fail status
5. Known issues: anything that works but isn't ideal (will be addressed in later phases)
6. Rollback procedure: exact steps to revert this phase
7. Operator runbook: how to spin up local Postgres, where the .env lives, how to switch between SQLite and Postgres modes

---

## Final Output

At the end of Phase A.1, the user should be able to:

```bash
# Start local Postgres
docker-compose up -d postgres-dev

# Set the database URL
export DATABASE_URL=postgresql://eagle_eye_dev:eagle_eye_dev_password@localhost:5433/eagle_eye_dev

# Run the same Eagle Eye code as before, now against Postgres
python -m app.services.eagle_eye.ingest

# Run tests
pytest tests/
```

And get identical functional behavior to SQLite mode, with all configuration externalized and dependencies pinned.

**Status line at end of completion report:**

```
PHASE A.1 COMPLETE — CODEBASE PRODUCTION-READY FOR DATABASE / DEPS / CONFIG
NEXT: Phase A.2 (Alembic schema migration tooling)
```

---

## Hard Don'ts

- Do NOT migrate any cached data from SQLite to Postgres in this phase (that's A.3)
- Do NOT connect to or modify the production Postgres database
- Do NOT add monitoring, logging frameworks, or scheduling (those are A.5/A.6)
- Do NOT deploy anything to the actual VPS
- Do NOT remove the SQLite code path — keep both working through this phase
- Do NOT rename any tables or columns (Phase A.2 can do refactors via migrations)
- Do NOT add new features. This is migration only.

---

## Escalation

The agent should stop and ask before:
- Touching any file under `app/services/eagle_eye/` that doesn't currently exist in the dev environment (i.e., the agent must already see the file before editing)
- Connecting to any database other than the local Docker Postgres
- Adding any dependency not currently in use
- Making any decision about ORM / SQL approach without confirming what the existing portfolio app uses
- Encountering committed secrets in git
- Finding that the schema differs substantially between SQLite and what the brief assumes
