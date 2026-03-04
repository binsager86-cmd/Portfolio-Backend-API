"""
Database Schema Initialization — ensures ALL tables exist at startup.

Called from main.py lifespan.  Every CREATE is IF NOT EXISTS so it's
safe to run on every boot, even against a production database that
already has the tables.

Supports both SQLite (AUTOINCREMENT) and PostgreSQL (SERIAL).
"""

import logging
from app.core.config import get_settings
from app.core.database import exec_sql, add_column_if_missing

logger = logging.getLogger(__name__)

settings = get_settings()


def _pk() -> str:
    """Return the correct auto-increment primary key syntax."""
    if settings.use_postgres:
        return "SERIAL PRIMARY KEY"
    return "INTEGER PRIMARY KEY AUTOINCREMENT"


def ensure_all_tables() -> None:
    """
    Idempotently create every table the application needs.

    Tables are grouped by domain; each block is wrapped in its own
    try/except so a failure in one group doesn't prevent the rest.
    """
    PK = _pk()

    # ── 1. Auth & Users ──────────────────────────────────────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS users (
                id          {PK},
                username    TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name        TEXT,
                created_at  INTEGER,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until INTEGER,
                last_failed_login INTEGER
            )
        """)
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS token_blacklist (
                id              {PK},
                jti             TEXT NOT NULL UNIQUE,
                user_id         INTEGER,
                blacklisted_at  INTEGER NOT NULL,
                expires_at      INTEGER NOT NULL
            )
        """)
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS audit_log (
                id              {PK},
                user_id         INTEGER,
                action          TEXT NOT NULL,
                resource_type   TEXT,
                resource_id     INTEGER,
                details         TEXT,
                ip_address      TEXT,
                user_agent      TEXT,
                created_at      INTEGER NOT NULL
            )
        """)
        logger.info("✅  Auth tables ensured (users, token_blacklist, audit_log)")
    except Exception as e:
        logger.warning("⚠️  Auth tables creation skipped: %s", e)

    # ── 2. Portfolios ────────────────────────────────────────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS portfolios (
                id          {PK},
                user_id     INTEGER NOT NULL,
                name        TEXT NOT NULL,
                currency    TEXT DEFAULT 'KWD',
                description TEXT,
                created_at  INTEGER
            )
        """)
        logger.info("✅  portfolios table ensured")
    except Exception as e:
        logger.warning("⚠️  portfolios table creation skipped: %s", e)

    # ── 3. Stocks ────────────────────────────────────────────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS stocks (
                id                  {PK},
                user_id             INTEGER NOT NULL,
                symbol              TEXT NOT NULL,
                name                TEXT,
                portfolio           TEXT,
                currency            TEXT DEFAULT 'KWD',
                current_price       REAL,
                last_updated        INTEGER,
                price_source        TEXT,
                tradingview_symbol  TEXT,
                tradingview_exchange TEXT,
                market_cap          REAL,
                sector              TEXT,
                industry            TEXT,
                yf_ticker           TEXT
            )
        """)
        logger.info("✅  stocks table ensured")
    except Exception as e:
        logger.warning("⚠️  stocks table creation skipped: %s", e)

    # ── 4. Transactions ──────────────────────────────────────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS transactions (
                id                  {PK},
                user_id             INTEGER NOT NULL,
                portfolio           TEXT NOT NULL,
                stock_symbol        TEXT NOT NULL,
                txn_date            TEXT,
                txn_type            TEXT NOT NULL,
                shares              REAL,
                purchase_cost       REAL,
                sell_value          REAL,
                bonus_shares        REAL,
                cash_dividend       REAL,
                reinvested_dividend REAL,
                fees                REAL,
                price_override      REAL,
                planned_cum_shares  REAL,
                broker              TEXT,
                reference           TEXT,
                notes               TEXT,
                category            TEXT DEFAULT 'portfolio',
                source              TEXT,
                fx_rate_at_txn      REAL,
                is_deleted          INTEGER DEFAULT 0,
                deleted_at          INTEGER,
                created_at          INTEGER
            )
        """)
        # Additive columns used by trading recalculations
        for col in ("avg_cost_at_txn", "realized_pnl_at_txn",
                     "cost_basis_at_txn", "shares_held_at_txn"):
            add_column_if_missing("transactions", col, "REAL")
        logger.info("✅  transactions table ensured")
    except Exception as e:
        logger.warning("⚠️  transactions table creation skipped: %s", e)

    # ── 5. Cash Deposits ─────────────────────────────────────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS cash_deposits (
                id                  {PK},
                user_id             INTEGER NOT NULL,
                portfolio           TEXT NOT NULL,
                deposit_date        TEXT NOT NULL,
                amount              REAL NOT NULL,
                currency            TEXT DEFAULT 'KWD',
                bank_name           TEXT,
                source              TEXT DEFAULT 'deposit',
                deposit_type        TEXT DEFAULT 'deposit',
                notes               TEXT,
                description         TEXT,
                comments            TEXT,
                include_in_analysis INTEGER DEFAULT 1,
                fx_rate_at_deposit  REAL,
                is_deleted          INTEGER DEFAULT 0,
                deleted_at          INTEGER,
                created_at          INTEGER
            )
        """)
        logger.info("✅  cash_deposits table ensured")
    except Exception as e:
        logger.warning("⚠️  cash_deposits table creation skipped: %s", e)

    # ── 6. Portfolio Cash (computed balances) ─────────────────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS portfolio_cash (
                id              {PK},
                user_id         INTEGER NOT NULL,
                portfolio       TEXT NOT NULL,
                balance         REAL,
                currency        TEXT DEFAULT 'KWD',
                last_updated    INTEGER,
                manual_override INTEGER DEFAULT 0
            )
        """)
        logger.info("✅  portfolio_cash table ensured")
    except Exception as e:
        logger.warning("⚠️  portfolio_cash table creation skipped: %s", e)

    # ── 7. Portfolio Snapshots (tracker / analytics) ─────────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id                  {PK},
                user_id             INTEGER NOT NULL,
                portfolio           TEXT,
                snapshot_date       TEXT NOT NULL,
                portfolio_value     REAL,
                daily_movement      REAL,
                beginning_difference REAL,
                deposit_cash        REAL,
                accumulated_cash    REAL,
                net_gain            REAL,
                change_percent      REAL,
                roi_percent         REAL,
                twr_percent         REAL,
                mwrr_percent        REAL,
                created_at          INTEGER
            )
        """)
        logger.info("✅  portfolio_snapshots table ensured")
    except Exception as e:
        logger.warning("⚠️  portfolio_snapshots table creation skipped: %s", e)

    # ── 8. Position Snapshots ────────────────────────────────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS position_snapshots (
                id                      {PK},
                user_id                 INTEGER NOT NULL,
                stock_id                INTEGER,
                stock_symbol            TEXT,
                portfolio_id            INTEGER,
                snapshot_date           TEXT NOT NULL,
                total_shares            REAL,
                total_cost              REAL,
                avg_cost                REAL,
                realized_pnl            REAL,
                cash_dividends_received REAL,
                status                  TEXT DEFAULT 'OPEN'
            )
        """)
        logger.info("✅  position_snapshots table ensured")
    except Exception as e:
        logger.warning("⚠️  position_snapshots table creation skipped: %s", e)

    # ── 9. Securities Master ─────────────────────────────────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS securities_master (
                security_id         TEXT PRIMARY KEY,
                user_id             INTEGER NOT NULL,
                exchange            TEXT NOT NULL,
                canonical_ticker    TEXT NOT NULL,
                display_name        TEXT,
                isin                TEXT,
                currency            TEXT DEFAULT 'KWD',
                country             TEXT DEFAULT 'KW',
                sector              TEXT,
                status              TEXT DEFAULT 'active',
                created_at          INTEGER
            )
        """)
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS security_aliases (
                id          {PK},
                security_id INTEGER,
                user_id     INTEGER,
                alias_name  TEXT,
                alias_type  TEXT,
                valid_from  TEXT,
                valid_until TEXT,
                created_at  INTEGER
            )
        """)
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS stocks_master (
                id          {PK},
                symbol      TEXT NOT NULL,
                name        TEXT NOT NULL,
                exchange    TEXT,
                currency    TEXT DEFAULT 'KWD'
            )
        """)
        logger.info("✅  Securities tables ensured (securities_master, security_aliases, stocks_master)")
    except Exception as e:
        logger.warning("⚠️  Securities tables creation skipped: %s", e)

    # ── 10. PFM (Personal Financial Management) ─────────────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS pfm_snapshots (
                id          {PK},
                user_id     INTEGER NOT NULL,
                snapshot_date TEXT NOT NULL,
                notes       TEXT,
                total_assets     REAL DEFAULT 0,
                total_liabilities REAL DEFAULT 0,
                net_worth        REAL DEFAULT 0,
                created_at  INTEGER
            )
        """)
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS pfm_assets (
                id          {PK},
                snapshot_id INTEGER NOT NULL,
                user_id     INTEGER NOT NULL,
                asset_type  TEXT NOT NULL,
                category    TEXT NOT NULL,
                name        TEXT NOT NULL,
                quantity    REAL,
                price       REAL,
                currency    TEXT DEFAULT 'KWD',
                value_kwd   REAL DEFAULT 0,
                created_at  INTEGER
            )
        """)
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS pfm_liabilities (
                id          {PK},
                snapshot_id INTEGER NOT NULL,
                user_id     INTEGER NOT NULL,
                category    TEXT NOT NULL,
                amount_kwd  REAL DEFAULT 0,
                is_current  INTEGER DEFAULT 0,
                is_long_term INTEGER DEFAULT 0,
                created_at  INTEGER
            )
        """)
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS pfm_income_expenses (
                id              {PK},
                snapshot_id     INTEGER NOT NULL,
                user_id         INTEGER NOT NULL,
                kind            TEXT NOT NULL,
                category        TEXT NOT NULL,
                monthly_amount  REAL DEFAULT 0,
                is_finance_cost INTEGER DEFAULT 0,
                is_gna          INTEGER DEFAULT 0,
                sort_order      INTEGER DEFAULT 0,
                created_at      INTEGER
            )
        """)
        logger.info("✅  PFM tables ensured")
    except Exception as e:
        logger.warning("⚠️  PFM tables creation skipped: %s", e)

    # ── 11. User Settings ────────────────────────────────────────────
    try:
        exec_sql("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id       INTEGER NOT NULL,
                setting_key   TEXT NOT NULL,
                setting_value TEXT NOT NULL,
                updated_at    INTEGER,
                PRIMARY KEY (user_id, setting_key)
            )
        """)
        logger.info("✅  user_settings table ensured")
    except Exception as e:
        logger.warning("⚠️  user_settings table creation skipped: %s", e)

    # ── 12. External accounts & portfolio transactions ───────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS external_accounts (
                id                  {PK},
                user_id             INTEGER NOT NULL,
                portfolio_id        INTEGER,
                name                TEXT NOT NULL,
                account_type        TEXT,
                currency            TEXT DEFAULT 'KWD',
                current_balance     REAL DEFAULT 0,
                last_reconciled_date TEXT,
                notes               TEXT,
                created_at          INTEGER
            )
        """)
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS portfolio_transactions (
                id              {PK},
                user_id         INTEGER NOT NULL,
                portfolio_id    INTEGER NOT NULL,
                account_id      INTEGER,
                stock_id        INTEGER,
                txn_type        TEXT NOT NULL,
                txn_date        TEXT NOT NULL,
                amount          REAL DEFAULT 0,
                shares          REAL,
                price_per_share REAL,
                fees            REAL DEFAULT 0,
                currency        TEXT DEFAULT 'KWD',
                fx_rate         REAL,
                symbol          TEXT,
                description     TEXT,
                reference       TEXT,
                notes           TEXT,
                is_deleted      INTEGER DEFAULT 0,
                deleted_at      INTEGER,
                created_at      INTEGER,
                updated_at      INTEGER
            )
        """)
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS ledger_entries (
                id              {PK},
                asset_id        INTEGER,
                entry_date      TEXT,
                entry_type      TEXT,
                quantity        REAL,
                price_per_unit  REAL,
                total_value     REAL,
                fees            REAL,
                notes           TEXT,
                created_at      INTEGER
            )
        """)
        logger.info("✅  Supplementary tables ensured (external_accounts, portfolio_transactions, ledger_entries)")
    except Exception as e:
        logger.warning("⚠️  Supplementary tables creation skipped: %s", e)

    # ── 13. Additive column migrations ───────────────────────────────
    try:
        add_column_if_missing("stocks", "yf_ticker", "TEXT")
        add_column_if_missing("users", "failed_login_attempts", "INTEGER DEFAULT 0")
        add_column_if_missing("users", "locked_until", "INTEGER")
        add_column_if_missing("users", "last_failed_login", "INTEGER")
        # Google OAuth: store email separately so email-registered users
        # can be matched when they later sign in via Google.
        add_column_if_missing("users", "email", "TEXT")
        add_column_if_missing("users", "google_sub", "TEXT")
        logger.info("✅  Additive column migrations applied")
    except Exception as e:
        logger.warning("⚠️  Additive column migrations skipped: %s", e)

    logger.info("🏁  Schema initialization complete — all tables ensured")
