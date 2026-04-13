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
        # Additive columns for stocks (may be missing from older schemas)
        for col, ctype in [
            ("yf_ticker", "TEXT"),
            ("price_source", "TEXT"),
            ("tradingview_symbol", "TEXT"),
            ("tradingview_exchange", "TEXT"),
            ("market_cap", "REAL"),
            ("sector", "TEXT"),
            ("industry", "TEXT"),
        ]:
            add_column_if_missing("stocks", col, ctype)
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
        # Additive columns — ensure all columns exist even if table
        # was created from an older Streamlit schema
        for col, ctype in [
            ("category", "TEXT"),
            ("source", "TEXT"),
            ("fx_rate_at_txn", "REAL"),
            ("is_deleted", "INTEGER"),
            ("deleted_at", "INTEGER"),
            ("avg_cost_at_txn", "REAL"),
            ("realized_pnl_at_txn", "REAL"),
            ("cost_basis_at_txn", "REAL"),
            ("shares_held_at_txn", "REAL"),
        ]:
            add_column_if_missing("transactions", col, ctype)
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
        # Backfill columns added after initial schema
        add_column_if_missing("pfm_snapshots", "total_assets", "REAL DEFAULT 0")
        add_column_if_missing("pfm_snapshots", "total_liabilities", "REAL DEFAULT 0")
        add_column_if_missing("pfm_snapshots", "net_worth", "REAL DEFAULT 0")
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

    # ── 13. News Articles ────────────────────────────────────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS news_articles (
                id              {PK},
                news_id         TEXT UNIQUE NOT NULL,
                title           TEXT NOT NULL,
                summary         TEXT,
                source          TEXT NOT NULL DEFAULT 'boursa_kuwait',
                category        TEXT NOT NULL DEFAULT 'company_announcement',
                published_at    TIMESTAMP NOT NULL,
                url             TEXT,
                related_symbols TEXT,
                sentiment       TEXT NOT NULL DEFAULT 'neutral',
                impact          TEXT NOT NULL DEFAULT 'informational',
                language        TEXT NOT NULL DEFAULT 'en',
                is_verified     INTEGER NOT NULL DEFAULT 1,
                attachments_json TEXT,
                fetched_at      TIMESTAMP NOT NULL
            )
        """)
        logger.info("✅  news_articles table ensured")
    except Exception as e:
        logger.warning("⚠️  news_articles table creation skipped: %s", e)

    # ── 14. Market Data Cache ────────────────────────────────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS market_data (
                id              {PK},
                trade_date      TEXT NOT NULL,
                data_json       TEXT NOT NULL,
                fetched_at      INTEGER NOT NULL,
                UNIQUE(trade_date)
            )
        """)
        logger.info("✅  market_data table ensured")
    except Exception as e:
        logger.warning("⚠️  market_data table creation skipped: %s", e)

    # ── 15. Push Tokens (Expo push notification tokens) ──────────────
    try:
        exec_sql(f"""
            CREATE TABLE IF NOT EXISTS push_tokens (
                id              {PK},
                user_id         INTEGER NOT NULL,
                token           TEXT UNIQUE NOT NULL,
                platform        TEXT NOT NULL DEFAULT 'unknown',
                created_at      TIMESTAMP NOT NULL,
                updated_at      TIMESTAMP NOT NULL
            )
        """)
        logger.info("✅  push_tokens table ensured")
    except Exception as e:
        logger.warning("⚠️  push_tokens table creation skipped: %s", e)

    # ── 16. Additive column migrations ───────────────────────────────
    try:
        # -- users --
        add_column_if_missing("users", "failed_login_attempts", "INTEGER DEFAULT 0")
        add_column_if_missing("users", "locked_until", "INTEGER")
        add_column_if_missing("users", "last_failed_login", "INTEGER")
        add_column_if_missing("users", "email", "TEXT")
        add_column_if_missing("users", "google_sub", "TEXT")
        add_column_if_missing("users", "is_admin", "INTEGER DEFAULT 0")

        # -- ensure "Admin" user has admin privileges --
        exec_sql("UPDATE users SET is_admin = 1 WHERE username = 'Admin' AND COALESCE(is_admin, 0) = 0")

        # -- stocks --
        add_column_if_missing("stocks", "yf_ticker", "TEXT")

        # -- cash_deposits (production PG may have older schema) --
        add_column_if_missing("cash_deposits", "bank_name", "TEXT")
        add_column_if_missing("cash_deposits", "source", "TEXT DEFAULT 'deposit'")
        add_column_if_missing("cash_deposits", "deposit_type", "TEXT DEFAULT 'deposit'")
        add_column_if_missing("cash_deposits", "notes", "TEXT")
        add_column_if_missing("cash_deposits", "description", "TEXT")
        add_column_if_missing("cash_deposits", "comments", "TEXT")
        add_column_if_missing("cash_deposits", "include_in_analysis", "INTEGER DEFAULT 1")
        add_column_if_missing("cash_deposits", "fx_rate_at_deposit", "REAL")
        add_column_if_missing("cash_deposits", "is_deleted", "INTEGER DEFAULT 0")
        add_column_if_missing("cash_deposits", "deleted_at", "INTEGER")
        add_column_if_missing("cash_deposits", "created_at", "INTEGER")

        # -- transactions --
        add_column_if_missing("transactions", "is_deleted", "INTEGER DEFAULT 0")
        add_column_if_missing("transactions", "deleted_at", "INTEGER")
        add_column_if_missing("transactions", "category", "TEXT DEFAULT 'portfolio'")

        # -- portfolio_cash --
        add_column_if_missing("portfolio_cash", "manual_override", "INTEGER DEFAULT 0")
        add_column_if_missing("portfolio_cash", "last_updated", "INTEGER")

        logger.info("✅  Additive column migrations applied")
    except Exception as e:
        logger.warning("⚠️  Additive column migrations skipped: %s", e)

    # ── 17. PostgreSQL: drop stale NOT NULL constraints ──────────────
    # Production PG tables may have been created with older schemas that
    # used NOT NULL on columns now expected to be nullable.  SQLite has
    # no ALTER COLUMN, so this block is PG-only.
    if settings.use_postgres:
        _drop_stale_not_null_constraints()

    # ── 18. Production indexes ───────────────────────────────────────
    # PostgreSQL does NOT auto-index foreign-key columns.  These indexes
    # ensure common query patterns are fast on both SQLite and PG.
    _ensure_indexes()

    # ── 19. PostgreSQL: upgrade REAL → DOUBLE PRECISION ──────────────
    # PG REAL is 4-byte (~7 digits); financial data needs 8-byte (~15).
    if settings.use_postgres:
        _upgrade_real_to_float8()

    logger.info("🏁  Schema initialization complete — all tables ensured")


def _drop_stale_not_null_constraints() -> None:
    """Drop NOT NULL from columns that should be nullable in PostgreSQL.

    This is idempotent — running it on an already-nullable column is a
    no-op in PostgreSQL.
    """
    nullable_columns = {
        "cash_deposits": [
            "bank_name", "source", "deposit_type", "notes", "description",
            "comments", "fx_rate_at_deposit", "is_deleted", "deleted_at",
            "created_at", "currency",
        ],
        "transactions": [
            "txn_date", "shares", "purchase_cost", "sell_value",
            "bonus_shares", "cash_dividend", "reinvested_dividend", "fees",
            "price_override", "planned_cum_shares", "broker", "reference",
            "notes", "category", "source", "fx_rate_at_txn",
            "is_deleted", "deleted_at", "created_at",
        ],
        "portfolio_cash": [
            "balance", "currency", "last_updated", "manual_override",
        ],
        "portfolio_snapshots": [
            "portfolio", "portfolio_value", "daily_movement",
            "beginning_difference", "deposit_cash", "accumulated_cash",
            "net_gain", "change_percent", "roi_percent",
            "twr_percent", "mwrr_percent", "created_at",
        ],
        "stocks": [
            "name", "portfolio", "currency", "current_price",
            "last_updated", "price_source", "tradingview_symbol",
            "tradingview_exchange", "market_cap", "sector", "industry",
            "yf_ticker",
        ],
    }

    for table, cols in nullable_columns.items():
        for col in cols:
            try:
                exec_sql(
                    f"ALTER TABLE {table} ALTER COLUMN {col} DROP NOT NULL"
                )
            except Exception:
                pass  # column already nullable or doesn't exist — fine
    logger.info("✅  PostgreSQL NOT NULL constraints relaxed on optional columns")


def _ensure_indexes() -> None:
    """Create performance indexes for production workloads.

    PostgreSQL does NOT auto-index foreign-key columns.  These indexes
    ensure common query patterns (filter by user_id, date, symbol) are
    fast on both SQLite and PostgreSQL.
    """
    indexes = [
        # Auth & users
        ("idx_token_bl_user",        "token_blacklist",        "user_id"),
        ("idx_audit_user",           "audit_log",              "user_id"),
        ("idx_audit_created",        "audit_log",              "created_at"),
        # Portfolios
        ("idx_portfolios_user",      "portfolios",             "user_id"),
        # Stocks
        ("idx_stocks_user",          "stocks",                 "user_id"),
        ("idx_stocks_symbol",        "stocks",                 "symbol"),
        # Transactions
        ("idx_txn_user",             "transactions",           "user_id"),
        ("idx_txn_symbol",           "transactions",           "stock_symbol"),
        ("idx_txn_user_ptf",         "transactions",           "user_id, portfolio"),
        ("idx_txn_date",             "transactions",           "txn_date"),
        # Cash deposits
        ("idx_cashdep_user",         "cash_deposits",          "user_id"),
        ("idx_cashdep_user_ptf",     "cash_deposits",          "user_id, portfolio"),
        # Portfolio cash
        ("idx_ptfcash_user",         "portfolio_cash",         "user_id"),
        # Portfolio snapshots
        ("idx_snap_user",            "portfolio_snapshots",    "user_id"),
        ("idx_snap_date",            "portfolio_snapshots",    "snapshot_date"),
        ("idx_snap_user_date",       "portfolio_snapshots",    "user_id, snapshot_date"),
        # Position snapshots
        ("idx_possn_user",           "position_snapshots",     "user_id"),
        ("idx_possn_symbol",         "position_snapshots",     "stock_symbol"),
        ("idx_possn_date",           "position_snapshots",     "snapshot_date"),
        # Securities
        ("idx_secmaster_user",       "securities_master",      "user_id"),
        ("idx_secalias_secid",       "security_aliases",       "security_id"),
        # PFM
        ("idx_pfmsnap_user",         "pfm_snapshots",          "user_id"),
        ("idx_pfmasset_snap",        "pfm_assets",             "snapshot_id"),
        ("idx_pfmliab_snap",         "pfm_liabilities",        "snapshot_id"),
        ("idx_pfminc_snap",          "pfm_income_expenses",    "snapshot_id"),
        # News
        ("idx_news_published",       "news_articles",          "published_at"),
        ("idx_news_category",        "news_articles",          "category"),
        # Push tokens
        ("idx_pushtok_user",         "push_tokens",            "user_id"),
        # External accounts & portfolio transactions
        ("idx_extacc_user",          "external_accounts",      "user_id"),
        ("idx_ptxn_user",            "portfolio_transactions", "user_id"),
        ("idx_ptxn_ptf",             "portfolio_transactions", "portfolio_id"),
        ("idx_ptxn_date",            "portfolio_transactions", "txn_date"),
    ]

    for idx_name, table, columns in indexes:
        try:
            exec_sql(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({columns})"
            )
        except Exception:
            pass  # table may not exist yet on partial schema
    logger.info("✅  Production indexes ensured")


def _upgrade_real_to_float8() -> None:
    """Upgrade REAL (4-byte) columns to DOUBLE PRECISION (8-byte) on PG.

    PostgreSQL REAL is only 4-byte (~7 significant digits).  For financial
    data (portfolio values, costs, prices) DOUBLE PRECISION (8-byte, ~15
    digits) provides the accuracy needed in production.

    Idempotent — running on an already-upgraded column is harmless.
    """
    upgrades: dict[str, list[str]] = {
        "stocks": ["current_price", "market_cap"],
        "transactions": [
            "shares", "purchase_cost", "sell_value", "bonus_shares",
            "cash_dividend", "reinvested_dividend", "fees",
            "price_override", "planned_cum_shares", "fx_rate_at_txn",
            "avg_cost_at_txn", "realized_pnl_at_txn",
            "cost_basis_at_txn", "shares_held_at_txn",
        ],
        "cash_deposits": ["amount", "fx_rate_at_deposit"],
        "portfolio_cash": ["balance"],
        "portfolio_snapshots": [
            "portfolio_value", "daily_movement", "beginning_difference",
            "deposit_cash", "accumulated_cash", "net_gain",
            "change_percent", "roi_percent", "twr_percent", "mwrr_percent",
        ],
        "position_snapshots": [
            "total_shares", "total_cost", "avg_cost",
            "realized_pnl", "cash_dividends_received",
        ],
        "pfm_snapshots": ["total_assets", "total_liabilities", "net_worth"],
        "pfm_assets": ["quantity", "price", "value_kwd"],
        "pfm_liabilities": ["amount_kwd"],
        "pfm_income_expenses": ["monthly_amount"],
        "external_accounts": ["current_balance"],
        "portfolio_transactions": [
            "amount", "shares", "price_per_share", "fees", "fx_rate",
        ],
        "ledger_entries": [
            "quantity", "price_per_unit", "total_value", "fees",
        ],
    }

    for table, cols in upgrades.items():
        for col in cols:
            try:
                exec_sql(
                    f'ALTER TABLE "{table}" ALTER COLUMN "{col}" '
                    f"TYPE DOUBLE PRECISION"
                )
            except Exception:
                pass  # column doesn't exist or already correct type
    logger.info("✅  PostgreSQL REAL → DOUBLE PRECISION upgrade applied")
