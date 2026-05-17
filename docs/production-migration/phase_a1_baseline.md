# Phase A.1 Baseline — Production Migration Inventory

_Date: 2026-05-16_  
_Status: READ-ONLY AUDIT — no code changes made_  
_Purpose: Pre-migration inventory required before any Phase A.1 changes_

---

## Executive Summary

The Eagle Eye backend has **partial PostgreSQL awareness** already built into `app/core/database.py` (dual-mode connections, `?`→`%s` conversion via `_pg_sql_named()`). However, **15+ files in the ML layer bypass this central layer entirely** via direct `sqlite3.connect()` calls. Those files are the primary migration target.

**No committed secrets found.** `.env` (real credentials) is gitignored. `.env.production` is committed but contains only placeholders.

**Estimated migration surface:**
- 15 files with `sqlite3.connect()` → need connection centralization
- 7 `INSERT OR REPLACE` → need `ON CONFLICT DO UPDATE` for Postgres
- 6 `PRAGMA` statements → must be removed/conditioned
- ~14 tables with JSON stored as TEXT → candidates for JSONB
- 8 `portfolio.db` fallback references → need env var replacement
- No ML packages in current venv (lightgbm, scikit-learn, joblib) → need `requirements.txt`

---

## Section 1.1 — SQLite-Specific Database Access Points

### Direct `sqlite3.connect()` Calls (15 occurrences)

| File | Line | Purpose |
|------|------|---------|
| `app/core/database.py` | 348 | Central `get_connection()` helper — context-managed |
| `app/core/database.py` | 366 | Central `get_conn()` helper — non-context-managed |
| `app/services/eagle_eye/store.py` | 165 | OHLCV bulk upsert (raw executemany bypass for performance) |
| `app/services/eagle_eye/ml/evaluator_v2.py` | 445 | ML evaluator — direct connection per run |
| `app/services/eagle_eye/ml/feature_builder.py` | 735 | Schema inspection (PRAGMA table_info) |
| `app/services/eagle_eye/ml/pattern_store.py` | 353 | Pattern vector store read |
| `app/services/eagle_eye/ml/pattern_store.py` | 391 | Pattern vector store write |
| `app/services/eagle_eye/ml/run_phase2.py` | 117 | Phase 2 runner connection |
| `app/services/eagle_eye/ml/tier_resolver.py` | 28 | Tier resolver — uses `settings.database_abs_path` |
| `app/services/eagle_eye/ml/precursor_builder.py` | 346 | Precursor builder read |
| `app/services/eagle_eye/ml/precursor_builder.py` | 377 | Precursor builder write |
| `app/services/eagle_eye/ml/trainer_v2.py` | 672 | ML trainer connection |
| `app/services/eagle_eye/ml/training_matrix.py` | 301 | Training matrix read |
| `app/services/eagle_eye/ml/training_matrix.py` | 517 | Training matrix write |
| `app/services/eagle_eye/ml/training_matrix.py` | 682 | Training matrix checkpoint |

**Note:** `app/core/database.py` already has PostgreSQL conditional logic. All files OUTSIDE of `app/core/` bypass the central layer.

### INSERT OR REPLACE → ON CONFLICT Needed (7 occurrences)

| File | Line | Table |
|------|------|-------|
| `app/services/eagle_eye/store.py` | 170 | `ee_ohlcv_cache` (executemany bulk) |
| `app/services/eagle_eye/ml/evaluator_v2.py` | 462 | `ml_model_metrics` |
| `app/services/eagle_eye/ml/pattern_store.py` | 179 | `pattern_vector_store` |
| `app/services/eagle_eye/ml/run_phase2.py` | 139 | `ml_models` |
| `app/services/eagle_eye/ml/precursor_builder.py` | 256 | `move_precursors` |
| `app/services/eagle_eye/ml/trainer_v2.py` | 700 | `ml_models` |
| `app/services/eagle_eye/ml/training_matrix.py` | 520 | `features_audit` |

**Migration:** Each becomes `INSERT INTO ... ON CONFLICT (...) DO UPDATE SET ...` with the conflict column named explicitly.

### PRAGMA Statements (6 occurrences)

| File | Line | PRAGMA | Migration Action |
|------|------|--------|-----------------|
| `app/core/database.py` | 135 | `PRAGMA journal_mode=WAL` | Already conditioned on SQLite; remove for Postgres |
| `app/core/database.py` | 136 | `PRAGMA foreign_keys=ON` | Already conditioned on SQLite; remove for Postgres |
| `app/core/database.py` | 211 | `PRAGMA journal_mode=WAL` | Same as above |
| `app/core/database.py` | 542 | `PRAGMA table_info({table})` | Replace with `information_schema.columns` query for Postgres |
| `app/services/eagle_eye/store.py` | 167 | `PRAGMA journal_mode=WAL` | Wrap with dialect check |
| `app/services/eagle_eye/ml/feature_builder.py` | 754 | `PRAGMA table_info({table})` | Replace with `information_schema.columns` for Postgres |

### ? Parameter Placeholders

All Eagle Eye and core code uses `?` placeholders. `app/core/database.py` already translates these via `_pg_sql_named()` (converts `?` → `%s` for psycopg2). **The central layer is already handled.** The issue is the 15 ML files that bypass the central layer — they use raw `sqlite3` directly, which means their `?` placeholders also need to be converted or they need to be routed through the central layer.

### Hardcoded `.db` File Paths

| File | Line | Path | Action Required |
|------|------|------|-----------------|
| `app/core/config.py` | 25 | `../dev_portfolio.db` | Default config — OK for dev, blocked by env var in prod |
| `app/main.py` | 84 | `mobile-migration/dev_portfolio.db` | Error message string only — not a runtime path |
| `app/services/eagle_eye/ml/evaluator_v2.py` | 444 | `portfolio.db` (fallback) | **Replace with `settings.database_abs_path`** |
| `app/services/eagle_eye/ml/pattern_store.py` | 350, 390 | `portfolio.db` (fallback) | **Replace with `settings.database_abs_path`** |
| `app/services/eagle_eye/ml/precursor_builder.py` | 342, 376 | `portfolio.db` (fallback) | **Replace with `settings.database_abs_path`** |
| `app/services/eagle_eye/ml/run_phase2.py` | 116 | `portfolio.db` (fallback) | **Replace with `settings.database_abs_path`** |
| `app/services/eagle_eye/ml/training_matrix.py` | 295, 516, 681 | `portfolio.db` (fallback) | **Replace with `settings.database_abs_path`** |
| `app/services/eagle_eye/ml/trainer_v2.py` | 671 | `portfolio.db` (fallback) | **Replace with `settings.database_abs_path`** |

---

## Section 1.2 — Hardcoded Values and Credentials

### Credentials and Secrets

| Finding | Severity | Details |
|---------|----------|---------|
| `.env` exists locally with `TICKERCHART_USERNAME=sager123`, `TICKERCHART_PASSWORD=sager949`, `EODHD_API_TOKEN=682cc20d36b014...` | INFO | `.env` is gitignored. **NOT committed to git.** Local dev only. No action required but rotate if this becomes a shared repo. |
| `.env.production` is committed to git | INFO | Contains **placeholders only** (`CHANGE_ME_TO_A_RANDOM_64_CHAR_HEX_STRING`, empty API keys). No real credentials exposed. Template is safe to commit. |
| SMTP_USER `binsager.86@gmail.com` in local `.env` | INFO | Local `.env`, not committed. Dev email, not production credential. |

**Verdict: No security finding requiring stop-and-report. All real credentials are in gitignored `.env`. Template `.env.production` is safe.**

### Relative Paths for ML Output Directories

All ML output paths use `Path(__file__).resolve().parents[N]` — dynamically resolved to repo root. **Not hardcoded absolute paths.** These ARE environment-dependent (the paths must exist at runtime) but are NOT broken by machine changes.

Paths that must exist in production:
- `mobile-migration/backend-api/ml_training_matrix/v1/{TICKER}/` — training matrices
- `mobile-migration/backend-api/ml_models/` — trained model artifacts
- `mobile-migration/backend-api/ml_models/pattern_indices/` — pattern NN indices
- `mobile-migration/backend-api/reports/` — eligibility and diagnostic reports

These need to be created on the production VPS as part of Phase A.4.

---

## Section 1.3 — Current Dependencies

From `pip freeze` in the active venv (`.venv/Scripts/pip`):

```
# Web framework
fastapi==0.136.0
uvicorn==0.45.0
gunicorn==22.0.0
starlette==0.52.1
python-multipart==0.0.26

# Database
SQLAlchemy==2.0.49
psycopg2-binary==2.9.12     # PostgreSQL driver already installed
alembic==1.18.4              # Schema migrations already available
asyncpg==0.31.0

# Core data
pandas==2.3.3
numpy==2.4.4

# Config / environment
pydantic==2.13.3
pydantic-settings==2.14.0
python-dotenv==1.2.2

# Auth / security
bcrypt==4.3.0
python-jose==3.5.0
cryptography==43.0.3

# HTTP
httpx==0.28.1
requests==2.33.1
tenacity==9.1.4

# AI / AI services
google-genai==1.73.1

# Scheduling
APScheduler==3.11.2

# Monitoring
sentry-sdk==2.58.0
prometheus-fastapi-instrumentator==7.1.0
prometheus_client==0.25.0

# Dev / test
pytest==9.0.3
pytest-asyncio==1.3.0
ruff==0.15.12

# Other
openpyxl==3.1.5
PyMuPDF==1.27.2.2
yfinance==0.2.66
pytz==2026.1.post1
tzdata==2026.1
```

### Missing ML Packages (CRITICAL GAP)

The following packages are required by Eagle Eye ML but **NOT present in the current venv**:

| Package | Used In | Needed For |
|---------|---------|-----------|
| `lightgbm` | `ml/trainer_v2.py`, `ml/model_store.py` | Model training and inference |
| `scikit-learn` | `ml/evaluator_v2.py`, `ml/trainer_v2.py` | Calibration, metrics |
| `joblib` | `ml/model_store.py` | Model serialization |
| `scipy` | `ml/evaluator_v2.py` (likely) | Statistical functions |

These were installed separately (via `python -m pip install` during Phase 1 debugging) but are not persisted in a `requirements.txt`. They will be missing on a fresh deployment.

**Action for Step 5:** Pin these in `requirements.txt`.

---

## Section 1.4 — Schema Inventory (Eagle Eye Tables Only)

_Portfolio app tables in `app/core/schema.py` (24 tables) are out of scope for Phase A.1 — they are already managed by Alembic migrations. Only Eagle Eye-specific tables are listed here._

### Eagle Eye Core (`app/services/eagle_eye/store.py`)

| Table | Columns | JSON Columns | Primary Key |
|-------|---------|--------------|-------------|
| `ee_ohlcv_cache` | ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at | — | (ticker, bar_date) |
| `ee_dna_profiles` | ticker, dna_json, computed_at | dna_json TEXT | ticker |
| `ee_ratings_cache` | ticker, name_en, sector, stage, rating, confidence, thesis, entry_primary, entry_aggressive, entry_conservative, stop_loss, tp1/tp2/tp3 + probabilities, last_price, supports_json, resistances_json, signals_json, indicators_json, days_of_history, computed_at, updated_at, volume_context_json | supports_json, resistances_json, signals_json, indicators_json, **volume_context_json** (5 TEXT JSON columns) | ticker |
| `ee_compute_log` | id, run_type, ticker, status, message, run_at | — | AUTOINCREMENT |

### Eagle Eye ML (`app/services/eagle_eye/ml/db_tables.py`)

| Table | Primary Key | JSON/BLOB Columns | Notes |
|-------|-------------|-------------------|-------|
| `ml_models` | model_id TEXT | — | CHECK on status |
| `ml_model_metrics` | AUTOINCREMENT | — | FK → ml_models |
| `ml_predictions` | prediction_id TEXT | probability_surface_json TEXT, early_move_risk_json TEXT, analogues_json TEXT, top_features_json TEXT | 4 JSON TEXT columns |
| `ml_shadow_log` | AUTOINCREMENT | — | FK → ml_models |
| `model_lifecycle_log` | AUTOINCREMENT | metadata_json TEXT | CHECK on action |
| `features_audit` | (feature_name, feature_version) | — | CHECK on leakage_verdict |
| `data_lineage_log` | AUTOINCREMENT | — | CHECK on action |
| `ml_fundamentals` | AUTOINCREMENT | — | UNIQUE (stock_ticker, disclosure_date) |
| `ml_corporate_events` | AUTOINCREMENT | raw_json TEXT | UNIQUE constraint |
| `ml_stock_eligibility` | stock_ticker TEXT | — | — |
| `move_precursors` | AUTOINCREMENT | context_json TEXT | — |
| `pattern_vector_store` | AUTOINCREMENT | **vector_blob BLOB** | UNIQUE (stock_ticker, vector_date) |
| `ml_calibration_health` | AUTOINCREMENT | reliability_json TEXT | FK → ml_models |
| `considered_signals` | signal_id TEXT | **full_feature_snapshot_json TEXT** | CHECK on would_have_entered |

### Simulator (`app/services/eagle_eye/simulator.py`)

| Table | Primary Key | JSON Columns | Notes |
|-------|-------------|--------------|-------|
| `simulator_portfolios` | AUTOINCREMENT | — | — |
| `simulator_positions` | AUTOINCREMENT | entry_signal_breakdown TEXT, entry_indicators_snapshot TEXT | NUMERIC(8,2) for entry_relative_volume |
| `simulator_daily_snapshots` | AUTOINCREMENT | — | UNIQUE (portfolio_id, date) |
| `simulator_considered_trades` | AUTOINCREMENT | — | — |

**Total Eagle Eye tables: 22** (4 core + 14 ML + 4 simulator)

---

## Section 1.5 — SQLite-Specific Behavior Flags

### JSON Stored as TEXT → JSONB in PostgreSQL

| Table | Column(s) | Migration Risk |
|-------|-----------|---------------|
| `ee_ratings_cache` | supports_json, resistances_json, signals_json, indicators_json, volume_context_json | HIGH — app reads/writes these as strings via `json.loads()`; JSONB requires Python dict pass-through |
| `ee_dna_profiles` | dna_json | MEDIUM |
| `ml_predictions` | probability_surface_json, early_move_risk_json, analogues_json, top_features_json | MEDIUM |
| `model_lifecycle_log` | metadata_json | LOW |
| `ml_corporate_events` | raw_json | LOW |
| `move_precursors` | context_json | LOW |
| `pattern_vector_store` | metadata_json | LOW |
| `ml_calibration_health` | reliability_json | LOW |
| `considered_signals` | full_feature_snapshot_json | MEDIUM — read by signal_logger and training pipeline |
| `simulator_positions` | entry_signal_breakdown, entry_indicators_snapshot | LOW |

**Total: 15 JSON TEXT columns across 10 tables.**

Decision for Phase A.1: keep as `TEXT` in PostgreSQL for now (simplest migration, backward compatible). Switch to `JSONB` in a future Alembic migration after Phase A.2 tooling is in place.

### BLOB Column → BYTEA in PostgreSQL

| Table | Column | Content |
|-------|--------|---------|
| `pattern_vector_store` | vector_blob | Pickled numpy array (NN pattern index) |

**Migration:** Change to `BYTEA` in Postgres DDL. Python `bytes` objects are transparently handled by psycopg2 as BYTEA.

### AUTOINCREMENT → SERIAL/IDENTITY in PostgreSQL

All 20+ AUTOINCREMENT primary keys need to become `SERIAL PRIMARY KEY` or `BIGSERIAL PRIMARY KEY` in Postgres. `app/core/schema.py` already has this logic (lines 23-24 show dialect branching).

**Status:** `app/core/schema.py` already handles this via dialect check. The Eagle Eye tables in `ml/db_tables.py` and `simulator.py` do NOT — they all use `INTEGER PRIMARY KEY AUTOINCREMENT`. These need the same dialect branching.

### Datetime Columns — Timezone Handling

| Pattern | Tables | Issue |
|---------|--------|-------|
| `TEXT DEFAULT (datetime('now'))` | audit_events (compliance_service.py:34) | SQLite dialect function — doesn't work in Postgres |
| `INTEGER` epoch timestamps | audit_log, token_blacklist, users, most app tables | No issue — epoch integers are DB-agnostic |
| `TEXT` ISO strings | ee_ohlcv_cache.bar_date, ee_ratings_cache.computed_at, simulator_positions.entry_date | Will work in Postgres TEXT columns but miss timezone enforcement |

**Action:** Replace `datetime('now')` with `CURRENT_TIMESTAMP` (ANSI SQL, works in both). For Phase A.1 keep as TEXT in Postgres. Add TIMESTAMPTZ in Phase A.2 Alembic migrations.

### Boolean Stored as INTEGER

Many tables use `INTEGER` for boolean fields (e.g., `would_have_entered`, `outcome_filled`, `tp1_hit`, `eligible`). This works transparently in Postgres (integer 0/1 is accepted in non-strict mode) but is not type-safe. Note for Alembic migration in Phase A.2 — add proper BOOLEAN columns then.

### rowid References

No direct `rowid` references found in application code. SQLite tables without explicit PKs use rowid implicitly — all Eagle Eye tables have explicit PKs. **No migration risk.**

### `app/core/database.py` — Existing Postgres Support Inventory

| Feature | Status |
|---------|--------|
| Dual-mode connection (`sqlite3` vs `psycopg2`) | Implemented at lines 340-370 |
| `?` → `%s` parameter conversion (`_pg_sql_named()`) | Implemented at lines 187-205 |
| WAL mode conditioned on SQLite only | Implemented (wrapped in dialect check) |
| Connection pooling via SQLAlchemy | Implemented for PostgreSQL path |
| Row factory (`sqlite3.Row` for SQLite, dict wrapper for PG) | Implemented |
| `add_column_if_missing()` helper | Uses `PRAGMA table_info` for SQLite — needs Postgres branch |
| `column_exists()` helper | Uses `PRAGMA table_info` for SQLite — needs Postgres branch |

---

## Readiness Assessment

### What's Already Done (Pre-Phase-A.1)

- `psycopg2-binary` and `asyncpg` are already installed
- `alembic` is already installed
- `app/core/database.py` has dual-mode connection logic
- `?` → `%s` conversion already exists
- `app/core/config.py` already reads `DATABASE_URL` from env
- `app/core/schema.py` already has `SERIAL` / `AUTOINCREMENT` dialect branching

### What Needs to Be Done in Phase A.1

| Priority | Task | Files Affected |
|----------|------|---------------|
| P0 | Route all ML `sqlite3.connect()` calls through central layer | 13 ML files |
| P0 | Fix 8 `portfolio.db` hardcoded fallbacks | 5 ML files |
| P0 | Add missing ML packages to `requirements.txt` (lightgbm, scikit-learn, joblib) | New file |
| P1 | Replace `INSERT OR REPLACE` with `ON CONFLICT DO UPDATE` | 7 files |
| P1 | Remove/condition `PRAGMA` statements | 6 occurrences |
| P1 | Fix `datetime('now')` in DDL | compliance_service.py:34 |
| P1 | Add Postgres branch to `add_column_if_missing()` / `column_exists()` (uses PRAGMA) | database.py |
| P1 | Add `SERIAL` branching to Eagle Eye DDL (simulator.py, db_tables.py) | 2 files |
| P2 | Create `.env.example` (committed template with all keys) | New file |
| P2 | Create `docker-compose.yml` for local Postgres dev | New file |
| P2 | Create `requirements-dev.txt` for test/lint tools | New file |

### Known Risks

1. **`pattern_vector_store.vector_blob BLOB`** — pickled numpy. BYTEA in Postgres is fine but test round-trip.
2. **`ee_ohlcv_cache` bulk executemany** — `store.py:165` uses raw `sqlite3` for performance. For Postgres, use `psycopg2.extras.execute_values()` or `COPY`. Performance-sensitive.
3. **ML files open their own connections** — if Postgres connection pool has a limit of 10, and 13 ML files each open a connection during training, pool exhaustion is a real risk. Connection sharing through central layer is the fix.
4. **Training matrix `.parquet` files** — not a DB concern but must exist on the production VPS. Phase A.4 addresses this.

---

## Sign-Off Required Before Step 2

_Per Phase A.1 brief: "Do not start coding changes until this [baseline] exists and is reviewed."_

**Please review and confirm before proceeding to Step 2 (Docker Postgres setup) and Step 3 (code changes).**

Items requiring your decision:
1. **Postgres connection approach for ML layer:** Route all ML files through `app/core/database.py`'s central layer (recommended) OR give each ML file its own Postgres connection string from settings? Recommendation: central layer — avoids 13 separate connection configs.
2. **JSON → JSONB:** Defer JSONB migration to Phase A.2 (recommended — keep TEXT, simpler) OR do it now? Recommendation: defer.
3. **`ee_ohlcv_cache` bulk upsert performance:** Use `psycopg2.extras.execute_values()` (fast) OR standard `ON CONFLICT` (simple)? Recommendation: `execute_values` for the OHLCV table, standard for all others.
