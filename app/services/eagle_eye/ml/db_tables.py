"""
ml/db_tables.py — Phase 1: ML database table definitions.

Creates all tables required by the Eagle Eye ML pipeline.  Every DDL
statement is idempotent (CREATE TABLE IF NOT EXISTS / CREATE INDEX IF
NOT EXISTS) so this module can be called safely on every startup.

Tables
------
ml_models             : one row per trained model version per stock
ml_model_metrics      : metric time-series for each model
ml_predictions        : raw + calibrated scores at inference time
ml_shadow_log         : per-day shadow-vs-rule scores with outcomes
model_lifecycle_log   : every lifecycle event (train/promote/rollback)
features_audit        : per-feature leakage audit registry
data_lineage_log      : ingest / dedup / drop audit trail
ml_fundamentals       : point-in-time fundamentals (disclosure date)
ml_corporate_events   : structured corporate event calendar
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL: list[str] = [
    # ── Core model registry ───────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS ml_models (
        model_id               TEXT    PRIMARY KEY,
        stock_ticker           TEXT    NOT NULL,
        version                TEXT    NOT NULL,
        trained_at             TEXT    NOT NULL,
        training_window_start  TEXT    NOT NULL,
        training_window_end    TEXT    NOT NULL,
        oot_window_start       TEXT    NOT NULL,
        oot_window_end         TEXT    NOT NULL,
        status                 TEXT    NOT NULL
            CHECK(status IN ('TRAINING','SHADOW','LIVE','LIVE_WITH_REGIME_CAVEAT','ARCHIVED','FAILED_GATE')),
        parent_model_id        TEXT,
        n_events               INTEGER,
        created_at             TEXT    DEFAULT (datetime('now')),
        notes                  TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ml_models_ticker ON ml_models(stock_ticker)",
    "CREATE INDEX IF NOT EXISTS idx_ml_models_status  ON ml_models(status)",

    # ── Per-model metric rows ─────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS ml_model_metrics (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id     TEXT    NOT NULL REFERENCES ml_models(model_id),
        metric_name  TEXT    NOT NULL,
        metric_value REAL,
        window_type  TEXT    NOT NULL
            CHECK(window_type IN ('cv','oot','shadow','live')),
        measured_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ml_metrics_model ON ml_model_metrics(model_id)",
    "CREATE INDEX IF NOT EXISTS idx_ml_metrics_window ON ml_model_metrics(window_type)",

    # ── Inference predictions (one row per stock-date scored) ─────────────
    """
    CREATE TABLE IF NOT EXISTS ml_predictions (
        prediction_id    TEXT    PRIMARY KEY,
        model_id         TEXT    NOT NULL REFERENCES ml_models(model_id),
        stock_ticker     TEXT    NOT NULL,
        pred_date        TEXT    NOT NULL,
        raw_prob         REAL,
        calibrated_prob  REAL,
        rating_bucket    TEXT,
        features_hash    TEXT,
        created_at       TEXT    DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ml_pred_ticker_date ON ml_predictions(stock_ticker, pred_date)",

    # ── Shadow mode logging ───────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS ml_shadow_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id        TEXT    NOT NULL REFERENCES ml_models(model_id),
        stock_ticker    TEXT    NOT NULL,
        log_date        TEXT    NOT NULL,
        ml_score        REAL,
        ml_bucket       TEXT,
        rule_score      REAL,
        rule_bucket     TEXT,
        actual_outcome  INTEGER,
        outcome_filled  INTEGER DEFAULT 0,
        created_at      TEXT    DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_shadow_ticker_date ON ml_shadow_log(stock_ticker, log_date)",
    "CREATE INDEX IF NOT EXISTS idx_shadow_model ON ml_shadow_log(model_id)",

    # ── Human-auditable lifecycle log ─────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS model_lifecycle_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        logged_at       TEXT    NOT NULL DEFAULT (datetime('now')),
        action          TEXT    NOT NULL
            CHECK(action IN ('TRAIN','SHADOW_START','PROMOTE','ROLLBACK','ARCHIVE','RETRAIN','FAILED_GATE','AUTO_DISABLE')),
        stock_ticker    TEXT    NOT NULL,
        model_id        TEXT,
        reason          TEXT,
        human_approver  TEXT,
        metadata_json   TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_lifecycle_ticker ON model_lifecycle_log(stock_ticker)",
    "CREATE INDEX IF NOT EXISTS idx_lifecycle_action  ON model_lifecycle_log(action)",

    # ── Feature leakage audit registry ───────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS features_audit (
        feature_name       TEXT    NOT NULL,
        feature_version    TEXT    NOT NULL DEFAULT 'v1',
        leakage_verdict    TEXT    NOT NULL
            CHECK(leakage_verdict IN ('CLEAN','LEAKY','REVIEW','DROPPED')),
        audit_notes        TEXT,
        created_at         TEXT    DEFAULT (datetime('now')),
        updated_at         TEXT    DEFAULT (datetime('now')),
        PRIMARY KEY (feature_name, feature_version)
    )
    """,

    # ── Data lineage / ingest audit ───────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS data_lineage_log (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        logged_at        TEXT    NOT NULL DEFAULT (datetime('now')),
        source           TEXT    NOT NULL,
        action           TEXT    NOT NULL
            CHECK(action IN ('INGEST','DEDUP','DROP','FILTER','FLAG')),
        stock_ticker     TEXT,
        records_affected INTEGER,
        notes            TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_lineage_ticker ON data_lineage_log(stock_ticker)",

    # ── Point-in-time fundamentals ────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS ml_fundamentals (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_ticker         TEXT    NOT NULL,
        disclosure_date      TEXT    NOT NULL,
        period_end_date      TEXT,
        source               TEXT,
        pe_ratio             REAL,
        pb_ratio             REAL,
        eps                  REAL,
        book_value_per_share REAL,
        market_cap_kwd       REAL,
        dividend_yield_pct   REAL,
        payout_ratio_pct     REAL,
        roe_pct              REAL,
        roa_pct              REAL,
        debt_equity_ratio    REAL,
        revenue_kwd          REAL,
        net_income_kwd       REAL,
        created_at           TEXT    DEFAULT (datetime('now')),
        UNIQUE (stock_ticker, disclosure_date, source)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker_date ON ml_fundamentals(stock_ticker, disclosure_date)",

    # ── Structured corporate events ───────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS ml_corporate_events (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_ticker         TEXT    NOT NULL,
        event_type           TEXT    NOT NULL
            CHECK(event_type IN ('DIVIDEND','CAPITAL_INCREASE','AGM_EGM','RESULTS')),
        announcement_date    TEXT    NOT NULL,
        event_date           TEXT,
        ex_date              TEXT,
        amount               REAL,
        notes                TEXT,
        raw_json             TEXT,
        created_at           TEXT    DEFAULT (datetime('now')),
        UNIQUE (stock_ticker, event_type, announcement_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_corp_events_ticker ON ml_corporate_events(stock_ticker, event_type)",
    "CREATE INDEX IF NOT EXISTS idx_corp_events_date   ON ml_corporate_events(announcement_date)",

    # ── Phase 3: ML display kill-switch state (single row, id=1) ─────────
    """
    CREATE TABLE IF NOT EXISTS ml_display_state (
        id              INTEGER PRIMARY KEY DEFAULT 1 CHECK(id = 1),
        auto_disabled   INTEGER NOT NULL DEFAULT 0,
        disabled_at     TEXT,
        disabled_reason TEXT,
        updated_at      TEXT    DEFAULT (datetime('now'))
    )
    """,
    "INSERT INTO ml_display_state (id, auto_disabled) VALUES (1, 0) ON CONFLICT (id) DO NOTHING",

    # ── Phase 3: Daily shadow vs rule comparison log ───────────────────
    """
    CREATE TABLE IF NOT EXISTS phase3_evaluation_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        log_date        TEXT    NOT NULL,
        stock_ticker    TEXT    NOT NULL,
        model_id        TEXT,
        band_label      TEXT,
        rule_rating     TEXT,
        rule_confidence REAL,
        agreement       INTEGER,
        created_at      TEXT    DEFAULT (datetime('now')),
        UNIQUE(log_date, stock_ticker)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_p3eval_ticker_date ON phase3_evaluation_log(log_date, stock_ticker)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_shadow_model_date ON ml_shadow_log(model_id, log_date)",

    # ── Per-stock filter eligibility cache ────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS ml_stock_eligibility (
        stock_ticker       TEXT    PRIMARY KEY,
        eligible           INTEGER NOT NULL DEFAULT 0,
        reason             TEXT,
        n_move_events      INTEGER,
        n_trading_days     INTEGER,
        liquidity_tier     TEXT,
        median_daily_vol   REAL,
        watch_only         INTEGER DEFAULT 0,
        last_evaluated     TEXT    DEFAULT (datetime('now'))
    )
    """,

    # ── Move precursor patterns ───────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS move_precursors (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_ticker     TEXT    NOT NULL,
        precursor_date   TEXT    NOT NULL,
        pattern_type     TEXT    NOT NULL,
        signal_strength  REAL,
        context_json     TEXT,
        move_outcome     REAL,
        outcome_filled   INTEGER NOT NULL DEFAULT 0,
        created_at       TEXT    DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_precursors_ticker ON move_precursors(stock_ticker, precursor_date)",

    # ── Per-stock embedding / analogue vector store ───────────────────────
    """
    CREATE TABLE IF NOT EXISTS pattern_vector_store (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_ticker     TEXT    NOT NULL,
        vector_date      TEXT    NOT NULL,
        vector_blob      BLOB    NOT NULL,
        vector_dim       INTEGER NOT NULL,
        metadata_json    TEXT,
        created_at       TEXT    DEFAULT (datetime('now')),
        UNIQUE (stock_ticker, vector_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pvs_ticker ON pattern_vector_store(stock_ticker)",

    # ── Calibration health snapshots ──────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS ml_calibration_health (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id          TEXT    NOT NULL REFERENCES ml_models(model_id),
        evaluated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
        n_predictions     INTEGER,
        brier_score       REAL,
        ece_score         REAL,
        reliability_json  TEXT,
        drift_flag        INTEGER NOT NULL DEFAULT 0,
        notes             TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cal_health_model ON ml_calibration_health(model_id)",

    # ── Considered-signal log (Addendum A.4) ──────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS considered_signals (
        signal_id                   TEXT    PRIMARY KEY,
        stock_ticker                TEXT    NOT NULL,
        signal_date                 TEXT    NOT NULL,
        rule_score                  REAL    NOT NULL,
        would_have_entered          INTEGER NOT NULL CHECK(would_have_entered IN (0,1)),
        skip_reason                 TEXT,
        full_feature_snapshot_json  TEXT,
        realized_outcome_20d        REAL,
        outcome_filled              INTEGER NOT NULL DEFAULT 0,
        created_at                  INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cs_ticker_date ON considered_signals(stock_ticker, signal_date)",
    "CREATE INDEX IF NOT EXISTS idx_cs_outcome_filled ON considered_signals(outcome_filled)",

    # ── Ride Quality Model: active position state ─────────────────────────
    # Tracks entry price and running peak for each open position so the ride
    # evaluator can compute drawdown_from_peak without replaying full history.
    """
    CREATE TABLE IF NOT EXISTS ee_ride_state (
        ticker               TEXT    NOT NULL,
        entry_date           TEXT    NOT NULL,
        entry_price          REAL    NOT NULL,
        running_peak_price   REAL,
        last_evaluated       TEXT,
        last_action          TEXT    CHECK(last_action IN ('HOLD','ADD','EXIT')),
        created_at           TEXT    DEFAULT (datetime('now')),
        PRIMARY KEY (ticker, entry_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ride_state_ticker ON ee_ride_state(ticker)",

    # ── Ride Quality Model: evaluation log ────────────────────────────────
    # One row per ticker per day for active ride evaluations — audit trail
    # and for future A/B comparison vs rules-only decisions.
    """
    CREATE TABLE IF NOT EXISTS ride_quality_log (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker                TEXT    NOT NULL,
        eval_date             TEXT    NOT NULL,
        entry_date            TEXT    NOT NULL,
        entry_price           REAL    NOT NULL,
        current_price         REAL,
        days_held             INTEGER,
        unrealized_pct        REAL,
        peak_gain_pct         REAL,
        drawdown_from_peak    REAL,
        ride_action           TEXT    CHECK(ride_action IN ('HOLD','ADD','EXIT')),
        ride_confidence       REAL,
        p_hold                REAL,
        p_add                 REAL,
        p_exit                REAL,
        remaining_upside_est  REAL,
        model_source          TEXT,
        created_at            TEXT    DEFAULT (datetime('now')),
        UNIQUE (ticker, eval_date, entry_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_rql_ticker_date ON ride_quality_log(ticker, eval_date)",
]


# ---------------------------------------------------------------------------
# Addendum A schema migrations (additive only — safe on existing DBs)
# ---------------------------------------------------------------------------

# ALTER TABLE ADD COLUMN statements for columns added in Addendum A.
# SQLite ignores "duplicate column" errors via the try/except in
# _apply_migrations().  For PostgreSQL, we catch the relevant error too.
_ADDENDUM_A_MIGRATIONS: list[tuple[str, str]] = [
    # ml_models: regime_strategy column + allow LIVE_WITH_REGIME_CAVEAT status
    #   (the CHECK constraint on status cannot be updated in SQLite for existing
    #    rows; it is updated in the CREATE TABLE DDL above for fresh installs.
    #    Runtime enforcement is done via application-level validation.)
    ("ml_models",         "ALTER TABLE ml_models ADD COLUMN regime_strategy TEXT"),

    # ml_model_metrics: regime tag
    ("ml_model_metrics",  "ALTER TABLE ml_model_metrics ADD COLUMN regime TEXT"),

    # ml_predictions: extended scoring columns
    ("ml_predictions",    "ALTER TABLE ml_predictions ADD COLUMN probability_surface_json TEXT"),
    ("ml_predictions",    "ALTER TABLE ml_predictions ADD COLUMN early_move_risk_json TEXT"),
    ("ml_predictions",    "ALTER TABLE ml_predictions ADD COLUMN regime TEXT"),
    ("ml_predictions",    "ALTER TABLE ml_predictions ADD COLUMN analogues_json TEXT"),
    ("ml_predictions",    "ALTER TABLE ml_predictions ADD COLUMN top_features_json TEXT"),

    # ml_shadow_log: regime tag
    ("ml_shadow_log",     "ALTER TABLE ml_shadow_log ADD COLUMN regime TEXT"),

    # Phase 3 — ml_shadow_log extended shadow runner columns
    ("ml_shadow_log",     "ALTER TABLE ml_shadow_log ADD COLUMN raw_prob REAL"),
    ("ml_shadow_log",     "ALTER TABLE ml_shadow_log ADD COLUMN calibrated_prob REAL"),
    ("ml_shadow_log",     "ALTER TABLE ml_shadow_log ADD COLUMN band_label TEXT"),
    ("ml_shadow_log",     "ALTER TABLE ml_shadow_log ADD COLUMN rule_stage TEXT"),
    ("ml_shadow_log",     "ALTER TABLE ml_shadow_log ADD COLUMN rule_confidence REAL"),
    ("ml_shadow_log",     "ALTER TABLE ml_shadow_log ADD COLUMN features_hash TEXT"),

    # Phase 3 — ml_predictions band display columns
    ("ml_predictions",    "ALTER TABLE ml_predictions ADD COLUMN band_label TEXT"),
    ("ml_predictions",    "ALTER TABLE ml_predictions ADD COLUMN band_low_threshold REAL"),
    ("ml_predictions",    "ALTER TABLE ml_predictions ADD COLUMN band_high_threshold REAL"),
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ensure_ml_tables() -> None:
    """Create all ML tables and apply additive migrations.  Safe to call on every startup."""
    from app.core.database import exec_sql

    for ddl in _DDL:
        stmt = ddl.strip()
        if not stmt:
            continue
        try:
            exec_sql(stmt, ())
        except Exception as exc:  # noqa: BLE001
            logger.warning("ML DDL warning (non-fatal): %s — %s", stmt[:60], exc)

    _apply_migrations()
    _migrate_lifecycle_log_constraint()
    logger.info("ML tables verified / created (including Addendum A schema).")


def _apply_migrations() -> None:
    """
    Run additive ALTER TABLE migrations.  Each migration is wrapped in
    try/except so it is safe to call repeatedly — duplicate-column errors
    from SQLite ('duplicate column name') and PostgreSQL ('already exists')
    are silently ignored.
    """
    from app.core.database import exec_sql

    for _table, stmt in _ADDENDUM_A_MIGRATIONS:
        try:
            exec_sql(stmt, ())
        except Exception as exc:
            msg = str(exc).lower()
            if "duplicate column" in msg or "already exists" in msg:
                pass  # column exists — expected on re-runs
            else:
                logger.warning("Migration warning (%s): %s", _table, exc)


def _migrate_lifecycle_log_constraint() -> None:
    """
    Addendum B: add AUTO_DISABLE to model_lifecycle_log action CHECK constraint.

    SQLite does not support ALTER TABLE … MODIFY COLUMN, so we recreate the
    table via the standard rename-swap pattern.  Safe to call repeatedly — it
    checks the existing DDL first and skips if AUTO_DISABLE is already present.
    """
    from app.core.database import get_connection

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_lifecycle_log'"
            )
            row = cur.fetchone()
            if row is None or "AUTO_DISABLE" in (row[0] or ""):
                return  # table absent (fresh install already has it) or already migrated

            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_lifecycle_log_p3 (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    logged_at       TEXT    NOT NULL DEFAULT (datetime('now')),
                    action          TEXT    NOT NULL
                        CHECK(action IN ('TRAIN','SHADOW_START','PROMOTE','ROLLBACK',
                                         'ARCHIVE','RETRAIN','FAILED_GATE','AUTO_DISABLE')),
                    stock_ticker    TEXT    NOT NULL,
                    model_id        TEXT,
                    reason          TEXT,
                    human_approver  TEXT,
                    metadata_json   TEXT
                )
            """)
            cur.execute(
                "INSERT OR IGNORE INTO model_lifecycle_log_p3 "
                "SELECT * FROM model_lifecycle_log"
            )
            cur.execute("DROP TABLE model_lifecycle_log")
            cur.execute(
                "ALTER TABLE model_lifecycle_log_p3 RENAME TO model_lifecycle_log"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_lifecycle_ticker "
                "ON model_lifecycle_log(stock_ticker)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_lifecycle_action "
                "ON model_lifecycle_log(action)"
            )
            conn.commit()
            logger.info("Addendum B: model_lifecycle_log constraint updated to include AUTO_DISABLE.")
    except Exception as exc:
        logger.warning("Addendum B migration skipped (non-fatal): %s", exc)


def log_lifecycle(
    *,
    action: str,
    stock_ticker: str,
    model_id: str | None = None,
    reason: str | None = None,
    human_approver: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Write a row to model_lifecycle_log.  Called by trainer, shadow runner, rollback."""
    import json
    from app.core.database import exec_sql

    meta_json = json.dumps(metadata, default=str) if metadata else None
    exec_sql(
        """
        INSERT INTO model_lifecycle_log
            (action, stock_ticker, model_id, reason, human_approver, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (action, stock_ticker, model_id, reason, human_approver, meta_json),
    )


def log_data_lineage(
    *,
    source: str,
    action: str,
    stock_ticker: str | None = None,
    records_affected: int = 0,
    notes: str | None = None,
) -> None:
    """Write a row to data_lineage_log."""
    from app.core.database import exec_sql

    exec_sql(
        """
        INSERT INTO data_lineage_log
            (source, action, stock_ticker, records_affected, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (source, action, stock_ticker, records_affected, notes),
    )
