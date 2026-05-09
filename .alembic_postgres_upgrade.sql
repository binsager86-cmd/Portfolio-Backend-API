BEGIN;

CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL, 
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- Running upgrade  -> 15d6c76a75e2

CREATE TABLE audit_log (
    id SERIAL NOT NULL, 
    user_id INTEGER, 
    action VARCHAR(100) NOT NULL, 
    resource_type VARCHAR(50), 
    resource_id INTEGER, 
    details TEXT, 
    ip_address VARCHAR(45), 
    user_agent VARCHAR(500), 
    created_at INTEGER NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_audit_log_action ON audit_log (action);

CREATE INDEX ix_audit_log_created_at ON audit_log (created_at);

CREATE INDEX ix_audit_log_user_id ON audit_log (user_id);

CREATE TABLE ledger_entries (
    id SERIAL NOT NULL, 
    asset_id INTEGER, 
    entry_date VARCHAR(20), 
    entry_type VARCHAR(30), 
    quantity FLOAT, 
    price_per_unit FLOAT, 
    total_value FLOAT, 
    fees FLOAT, 
    notes TEXT, 
    created_at INTEGER, 
    PRIMARY KEY (id)
);

CREATE TABLE news_articles (
    id SERIAL NOT NULL, 
    news_id VARCHAR(100) NOT NULL, 
    title TEXT NOT NULL, 
    summary TEXT, 
    source VARCHAR(50) NOT NULL, 
    category VARCHAR(50) NOT NULL, 
    published_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    url TEXT, 
    related_symbols TEXT, 
    sentiment VARCHAR(20) NOT NULL, 
    impact VARCHAR(20) NOT NULL, 
    language VARCHAR(5) NOT NULL, 
    is_verified INTEGER NOT NULL, 
    attachments_json TEXT, 
    fetched_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    content_hash VARCHAR(32), 
    PRIMARY KEY (id)
);

CREATE INDEX ix_news_articles_content_hash ON news_articles (content_hash);

CREATE UNIQUE INDEX ix_news_articles_news_id ON news_articles (news_id);

CREATE INDEX ix_news_articles_published_at ON news_articles (published_at);

CREATE TABLE securities_master (
    id SERIAL NOT NULL, 
    symbol VARCHAR(20) NOT NULL, 
    name VARCHAR(200) NOT NULL, 
    exchange VARCHAR(20), 
    currency VARCHAR(10) NOT NULL, 
    asset_type VARCHAR(20), 
    sector VARCHAR(100), 
    industry VARCHAR(100), 
    country VARCHAR(50), 
    yahoo_symbol VARCHAR(30), 
    tradingview_symbol VARCHAR(50), 
    tradingview_exchange VARCHAR(20), 
    isin VARCHAR(20), 
    market_cap FLOAT, 
    outstanding_shares FLOAT, 
    is_active INTEGER NOT NULL, 
    created_at INTEGER, 
    updated_at INTEGER, 
    PRIMARY KEY (id)
);

CREATE TABLE stocks_master (
    id SERIAL NOT NULL, 
    symbol VARCHAR(50) NOT NULL, 
    name VARCHAR(200) NOT NULL, 
    exchange VARCHAR(20), 
    currency VARCHAR(10) NOT NULL, 
    PRIMARY KEY (id)
);

CREATE TABLE token_blacklist (
    id SERIAL NOT NULL, 
    jti VARCHAR(64) NOT NULL, 
    user_id INTEGER NOT NULL, 
    blacklisted_at INTEGER NOT NULL, 
    expires_at INTEGER NOT NULL, 
    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX ix_token_blacklist_jti ON token_blacklist (jti);

CREATE INDEX ix_token_blacklist_user_id ON token_blacklist (user_id);

CREATE TABLE users (
    id SERIAL NOT NULL, 
    username VARCHAR(200) NOT NULL, 
    password_hash TEXT NOT NULL, 
    name VARCHAR(200), 
    created_at INTEGER, 
    is_admin INTEGER DEFAULT '0' NOT NULL, 
    failed_login_attempts INTEGER DEFAULT '0' NOT NULL, 
    locked_until INTEGER, 
    last_failed_login INTEGER, 
    PRIMARY KEY (id), 
    UNIQUE (username)
);

CREATE TABLE cash_deposits (
    id SERIAL NOT NULL, 
    user_id INTEGER NOT NULL, 
    portfolio VARCHAR(50) NOT NULL, 
    deposit_date VARCHAR(10) NOT NULL, 
    amount FLOAT NOT NULL, 
    currency VARCHAR(10) NOT NULL, 
    bank_name VARCHAR(100), 
    source VARCHAR(20), 
    deposit_type VARCHAR(20), 
    notes TEXT, 
    description TEXT, 
    comments TEXT, 
    include_in_analysis INTEGER, 
    fx_rate_at_deposit FLOAT, 
    is_deleted INTEGER, 
    deleted_at INTEGER, 
    created_at INTEGER, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE pfm_snapshots (
    id SERIAL NOT NULL, 
    user_id INTEGER NOT NULL, 
    snapshot_date VARCHAR(10) NOT NULL, 
    notes TEXT, 
    total_assets FLOAT NOT NULL, 
    total_liabilities FLOAT NOT NULL, 
    net_worth FLOAT NOT NULL, 
    created_at INTEGER, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE portfolio_cash (
    id SERIAL NOT NULL, 
    user_id INTEGER NOT NULL, 
    portfolio VARCHAR(50) NOT NULL, 
    balance FLOAT, 
    currency VARCHAR(10) NOT NULL, 
    last_updated INTEGER, 
    manual_override INTEGER NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE portfolio_snapshots (
    id SERIAL NOT NULL, 
    user_id INTEGER NOT NULL, 
    portfolio VARCHAR(50), 
    snapshot_date VARCHAR(10) NOT NULL, 
    portfolio_value FLOAT, 
    daily_movement FLOAT, 
    beginning_difference FLOAT, 
    deposit_cash FLOAT, 
    accumulated_cash FLOAT, 
    net_gain FLOAT, 
    change_percent FLOAT, 
    roi_percent FLOAT, 
    twr_percent FLOAT, 
    mwrr_percent FLOAT, 
    created_at INTEGER, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE portfolios (
    id SERIAL NOT NULL, 
    user_id INTEGER NOT NULL, 
    name VARCHAR(50) NOT NULL, 
    currency VARCHAR(10) NOT NULL, 
    description TEXT, 
    created_at INTEGER, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE position_snapshots (
    id SERIAL NOT NULL, 
    user_id INTEGER NOT NULL, 
    stock_id INTEGER, 
    stock_symbol VARCHAR(50), 
    portfolio_id INTEGER, 
    snapshot_date VARCHAR(10) NOT NULL, 
    total_shares FLOAT, 
    total_cost FLOAT, 
    avg_cost FLOAT, 
    realized_pnl FLOAT, 
    cash_dividends_received FLOAT, 
    status VARCHAR(20), 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE stocks (
    id SERIAL NOT NULL, 
    user_id INTEGER NOT NULL, 
    symbol VARCHAR(50) NOT NULL, 
    name VARCHAR(200), 
    portfolio VARCHAR(50), 
    currency VARCHAR(10), 
    current_price FLOAT, 
    last_updated INTEGER, 
    price_source VARCHAR(100), 
    tradingview_symbol VARCHAR(50), 
    tradingview_exchange VARCHAR(50), 
    market_cap FLOAT, 
    sector VARCHAR(100), 
    industry VARCHAR(100), 
    pe_ratio FLOAT, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE transactions (
    id SERIAL NOT NULL, 
    user_id INTEGER NOT NULL, 
    portfolio VARCHAR(50) NOT NULL, 
    stock_symbol VARCHAR(50) NOT NULL, 
    txn_date VARCHAR(20), 
    txn_type VARCHAR(20) NOT NULL, 
    shares FLOAT, 
    purchase_cost FLOAT, 
    sell_value FLOAT, 
    bonus_shares FLOAT, 
    cash_dividend FLOAT, 
    reinvested_dividend FLOAT, 
    fees FLOAT, 
    price_override FLOAT, 
    planned_cum_shares FLOAT, 
    broker VARCHAR(100), 
    reference VARCHAR(200), 
    notes TEXT, 
    category VARCHAR(50), 
    is_deleted INTEGER, 
    deleted_at INTEGER, 
    created_at INTEGER, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE external_accounts (
    id SERIAL NOT NULL, 
    user_id INTEGER NOT NULL, 
    portfolio_id INTEGER, 
    name VARCHAR(100) NOT NULL, 
    account_type VARCHAR(50), 
    currency VARCHAR(10) NOT NULL, 
    current_balance FLOAT NOT NULL, 
    last_reconciled_date VARCHAR(10), 
    notes TEXT, 
    created_at INTEGER, 
    PRIMARY KEY (id), 
    FOREIGN KEY(portfolio_id) REFERENCES portfolios (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE pfm_assets (
    id SERIAL NOT NULL, 
    snapshot_id INTEGER NOT NULL, 
    user_id INTEGER NOT NULL, 
    asset_type VARCHAR(50) NOT NULL, 
    category VARCHAR(100) NOT NULL, 
    name VARCHAR(200) NOT NULL, 
    quantity FLOAT, 
    price FLOAT, 
    currency VARCHAR(10) NOT NULL, 
    value_kwd FLOAT NOT NULL, 
    created_at INTEGER, 
    PRIMARY KEY (id), 
    FOREIGN KEY(snapshot_id) REFERENCES pfm_snapshots (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE pfm_income_expenses (
    id SERIAL NOT NULL, 
    snapshot_id INTEGER NOT NULL, 
    user_id INTEGER NOT NULL, 
    kind VARCHAR(20) NOT NULL, 
    category VARCHAR(100) NOT NULL, 
    monthly_amount FLOAT NOT NULL, 
    is_finance_cost INTEGER NOT NULL, 
    is_gna INTEGER NOT NULL, 
    sort_order INTEGER NOT NULL, 
    created_at INTEGER, 
    PRIMARY KEY (id), 
    FOREIGN KEY(snapshot_id) REFERENCES pfm_snapshots (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE pfm_liabilities (
    id SERIAL NOT NULL, 
    snapshot_id INTEGER NOT NULL, 
    user_id INTEGER NOT NULL, 
    category VARCHAR(100) NOT NULL, 
    amount_kwd FLOAT NOT NULL, 
    is_current INTEGER NOT NULL, 
    is_long_term INTEGER NOT NULL, 
    created_at INTEGER, 
    PRIMARY KEY (id), 
    FOREIGN KEY(snapshot_id) REFERENCES pfm_snapshots (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE portfolio_transactions (
    id SERIAL NOT NULL, 
    user_id INTEGER NOT NULL, 
    portfolio_id INTEGER NOT NULL, 
    account_id INTEGER, 
    stock_id INTEGER, 
    txn_type VARCHAR(20) NOT NULL, 
    txn_date VARCHAR(20) NOT NULL, 
    amount FLOAT, 
    shares FLOAT, 
    price_per_share FLOAT, 
    fees FLOAT, 
    currency VARCHAR(10), 
    fx_rate FLOAT, 
    symbol VARCHAR(50), 
    description TEXT, 
    reference VARCHAR(200), 
    notes TEXT, 
    is_deleted INTEGER, 
    deleted_at INTEGER, 
    created_at INTEGER, 
    updated_at INTEGER, 
    PRIMARY KEY (id), 
    FOREIGN KEY(portfolio_id) REFERENCES portfolios (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE TABLE security_aliases (
    id SERIAL NOT NULL, 
    security_id INTEGER, 
    user_id INTEGER, 
    alias_name TEXT, 
    alias_type TEXT, 
    valid_from TEXT, 
    valid_until TEXT, 
    created_at INTEGER, 
    PRIMARY KEY (id)
);

CREATE TABLE user_settings (
    user_id INTEGER NOT NULL, 
    setting_key TEXT NOT NULL, 
    setting_value TEXT NOT NULL, 
    updated_at INTEGER, 
    PRIMARY KEY (user_id, setting_key)
);

CREATE TABLE market_data (
    id SERIAL NOT NULL, 
    trade_date TEXT NOT NULL, 
    data_json TEXT NOT NULL, 
    fetched_at INTEGER NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX idx_market_data_trade_date ON market_data (trade_date, fetched_at);

INSERT INTO alembic_version (version_num) VALUES ('15d6c76a75e2') RETURNING alembic_version.version_num;

-- Running upgrade 15d6c76a75e2 -> a3f9c1d2e8b4

