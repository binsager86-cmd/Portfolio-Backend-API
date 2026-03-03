"""
Test Helpers & Factories — reusable data-creation utilities.

Provides:
  - Transaction factory (buy, sell, bonus, dividend)
  - Cash deposit factory
  - Snapshot factory
  - Second-user creation for isolation tests
  - Known-value test data builders for calculation verification
"""

import sqlite3
import time
import os
from datetime import date, timedelta
from typing import Optional


def get_test_db() -> sqlite3.Connection:
    """Get a connection to the test database."""
    db_path = os.environ["DATABASE_PATH"]
    return sqlite3.connect(db_path, check_same_thread=False)


# ── Transaction Factory ──────────────────────────────────────────────

def create_transaction(
    user_id: int = 1,
    portfolio: str = "KFH",
    stock_symbol: str = "TEST.KW",
    txn_date: str = "2024-01-15",
    txn_type: str = "Buy",
    shares: float = 100,
    purchase_cost: Optional[float] = 1000.0,
    sell_value: Optional[float] = None,
    bonus_shares: float = 0,
    cash_dividend: float = 0,
    reinvested_dividend: float = 0,
    fees: float = 0,
    category: str = "portfolio",
) -> int:
    """Insert a transaction and return its ID."""
    conn = get_test_db()
    cur = conn.cursor()
    now = int(time.time())
    cur.execute(
        """INSERT INTO transactions
           (user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, bonus_shares, cash_dividend,
            reinvested_dividend, fees, category, is_deleted, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
        (
            user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, bonus_shares, cash_dividend,
            reinvested_dividend, fees, category, now,
        ),
    )
    txn_id = cur.lastrowid
    conn.commit()
    conn.close()
    return txn_id


def create_buy(
    user_id: int = 1,
    portfolio: str = "KFH",
    symbol: str = "TEST.KW",
    txn_date: str = "2024-01-15",
    shares: float = 100,
    cost: float = 1000.0,
    fees: float = 0,
    bonus: float = 0,
) -> int:
    """Convenience: create a Buy transaction."""
    return create_transaction(
        user_id=user_id,
        portfolio=portfolio,
        stock_symbol=symbol,
        txn_date=txn_date,
        txn_type="Buy",
        shares=shares,
        purchase_cost=cost,
        sell_value=None,
        bonus_shares=bonus,
        fees=fees,
    )


def create_sell(
    user_id: int = 1,
    portfolio: str = "KFH",
    symbol: str = "TEST.KW",
    txn_date: str = "2024-06-01",
    shares: float = 50,
    sell_value: float = 600.0,
    fees: float = 0,
) -> int:
    """Convenience: create a Sell transaction."""
    return create_transaction(
        user_id=user_id,
        portfolio=portfolio,
        stock_symbol=symbol,
        txn_date=txn_date,
        txn_type="Sell",
        shares=shares,
        purchase_cost=None,
        sell_value=sell_value,
        fees=fees,
    )


def create_dividend(
    user_id: int = 1,
    portfolio: str = "KFH",
    symbol: str = "TEST.KW",
    txn_date: str = "2024-12-01",
    cash_dividend: float = 50.0,
) -> int:
    """Convenience: create a dividend transaction."""
    return create_transaction(
        user_id=user_id,
        portfolio=portfolio,
        stock_symbol=symbol,
        txn_date=txn_date,
        txn_type="Buy",
        shares=0,
        purchase_cost=0,
        sell_value=None,
        cash_dividend=cash_dividend,
    )


# ── Cash Deposit Factory ────────────────────────────────────────────

def create_deposit(
    user_id: int = 1,
    portfolio: str = "KFH",
    deposit_date: str = "2024-01-01",
    amount: float = 5000.0,
    currency: str = "KWD",
    bank_name: Optional[str] = None,
    deposit_type: str = "deposit",
    notes: Optional[str] = None,
) -> int:
    """Insert a cash deposit and return its ID."""
    conn = get_test_db()
    cur = conn.cursor()
    now = int(time.time())
    cur.execute(
        """INSERT INTO cash_deposits
           (user_id, portfolio, deposit_date, amount, currency,
            bank_name, deposit_type, notes, is_deleted, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
        (user_id, portfolio, deposit_date, amount, currency,
         bank_name, deposit_type, notes, now),
    )
    dep_id = cur.lastrowid
    conn.commit()
    conn.close()
    return dep_id


# ── Stock Factory ────────────────────────────────────────────────────

def create_stock(
    user_id: int = 1,
    symbol: str = "TEST.KW",
    name: str = "Test Company",
    portfolio: str = "KFH",
    currency: str = "KWD",
    current_price: float = 1.500,
) -> int:
    """Insert a stock entry and return its ID."""
    conn = get_test_db()
    cur = conn.cursor()
    now = int(time.time())
    cur.execute(
        """INSERT INTO stocks
           (user_id, symbol, name, portfolio, currency, current_price, last_updated)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (user_id, symbol, name, portfolio, currency, current_price, now),
    )
    stock_id = cur.lastrowid
    conn.commit()
    conn.close()
    return stock_id


# ── Snapshot Factory ─────────────────────────────────────────────────

def create_snapshot(
    user_id: int = 1,
    portfolio: str = "KFH",
    snapshot_date: Optional[str] = None,
    portfolio_value: float = 10000.0,
    deposit_cash: float = 8000.0,
    daily_movement: float = 0.0,
    net_gain: float = 0.0,
) -> int:
    """Insert a portfolio snapshot and return its ID."""
    if snapshot_date is None:
        snapshot_date = date.today().isoformat()
    conn = get_test_db()
    cur = conn.cursor()
    now = int(time.time())
    cur.execute(
        """INSERT INTO portfolio_snapshots
           (user_id, portfolio, snapshot_date, portfolio_value,
            deposit_cash, daily_movement, net_gain, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, portfolio, snapshot_date, portfolio_value,
         deposit_cash, daily_movement, net_gain, now),
    )
    snap_id = cur.lastrowid
    conn.commit()
    conn.close()
    return snap_id


# ── Second User for Isolation Tests ─────────────────────────────────

_user2_created = False


def ensure_user2(test_client=None) -> dict:
    """
    Create user2 for multi-user isolation tests.
    Returns dict with user_id, username, and auth_headers.
    Uses direct DB insert + token creation to avoid rate limits.
    """
    global _user2_created

    from app.core.security import hash_password, create_access_token

    if not _user2_created:
        conn = get_test_db()
        cur = conn.cursor()
        # Check if user2 already exists
        cur.execute("SELECT id FROM users WHERE username = 'user2'")
        row = cur.fetchone()
        if not row:
            pw_hash = hash_password("user2pass789")
            now = int(time.time())
            cur.execute(
                "INSERT INTO users (username, password_hash, name, created_at) VALUES (?, ?, ?, ?)",
                ("user2", pw_hash, "User Two", now),
            )
            user2_id = cur.lastrowid
            # Create a portfolio for user2
            cur.execute(
                "INSERT INTO portfolios (user_id, name, currency, created_at) VALUES (?, 'KFH', 'KWD', ?)",
                (user2_id, now),
            )
            conn.commit()
        else:
            user2_id = row[0]
        conn.close()
        _user2_created = True
    else:
        conn = get_test_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = 'user2'")
        user2_id = cur.fetchone()[0]
        conn.close()

    token = create_access_token(user2_id, "user2")
    return {
        "user_id": user2_id,
        "username": "user2",
        "headers": {"Authorization": f"Bearer {token}"},
    }


# ── Known-Value Test Data Builders ───────────────────────────────────

def seed_twr_reference_data(user_id: int = 1, portfolio: str = "KFH"):
    """
    Seed snapshot data for TWR verification against CFA reference values.

    Scenario:
      - Start: 100,000 portfolio value, 100,000 deposits
      - Day 30: deposit 10,000 → value grows to 105,000
      - Day 60: value is 112,000, deposits now 110,000
      - Day 90: value is 115,000, deposits still 110,000

    Expected TWR ≈ 4.23% (CFA Modified Dietz chain-linked)
    """
    base_date = date(2024, 1, 1)
    snapshots = [
        (base_date.isoformat(), 100000.0, 100000.0),
        ((base_date + timedelta(days=30)).isoformat(), 105000.0, 110000.0),
        ((base_date + timedelta(days=60)).isoformat(), 112000.0, 110000.0),
        ((base_date + timedelta(days=90)).isoformat(), 115000.0, 110000.0),
    ]

    conn = get_test_db()
    cur = conn.cursor()
    now = int(time.time())
    for snap_date, value, deposits in snapshots:
        cur.execute(
            """INSERT INTO portfolio_snapshots
               (user_id, portfolio, snapshot_date, portfolio_value, deposit_cash, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, portfolio, snap_date, value, deposits, now),
        )
    conn.commit()
    conn.close()


def seed_wac_complex_scenario(user_id: int = 1) -> list:
    """
    Seed transactions for a complex WAC scenario with known expected values.

    Scenario:
      Buy 100 @ 10.00 (cost=1000, fees=10) → avg=10.10
      Buy 200 @ 12.00 (cost=2400, fees=20) → avg=11.43
      Bonus 30 shares → avg=10.39 (diluted)
      Sell 150 @ 15.00 (proceeds=2250, fees=15) → realized_pnl = 2235 - 1558.79 = 676.21
      Buy 50 @ 14.00 (cost=700, fees=5) → new avg

    Returns list of transaction IDs.
    """
    ids = []
    ids.append(create_transaction(
        user_id=user_id, portfolio="KFH", stock_symbol="COMPLEX.KW",
        txn_date="2024-01-10", txn_type="Buy", shares=100,
        purchase_cost=1000.0, fees=10.0,
    ))
    ids.append(create_transaction(
        user_id=user_id, portfolio="KFH", stock_symbol="COMPLEX.KW",
        txn_date="2024-02-15", txn_type="Buy", shares=200,
        purchase_cost=2400.0, fees=20.0,
    ))
    ids.append(create_transaction(
        user_id=user_id, portfolio="KFH", stock_symbol="COMPLEX.KW",
        txn_date="2024-03-20", txn_type="Buy", shares=0,
        purchase_cost=0, bonus_shares=30, fees=0,
    ))
    ids.append(create_transaction(
        user_id=user_id, portfolio="KFH", stock_symbol="COMPLEX.KW",
        txn_date="2024-04-25", txn_type="Sell", shares=150,
        purchase_cost=None, sell_value=2250.0, fees=15.0,
    ))
    ids.append(create_transaction(
        user_id=user_id, portfolio="KFH", stock_symbol="COMPLEX.KW",
        txn_date="2024-05-30", txn_type="Buy", shares=50,
        purchase_cost=700.0, fees=5.0,
    ))
    return ids


def cleanup_test_data(
    table: str,
    user_id: Optional[int] = None,
    symbol: Optional[str] = None,
):
    """
    Remove test data from a specific table.
    Useful for cleaning up between tests.
    """
    conn = get_test_db()
    cur = conn.cursor()
    conditions = []
    params = []
    if user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)
    if symbol is not None and table == "transactions":
        conditions.append("stock_symbol = ?")
        params.append(symbol)
    where = " AND ".join(conditions) if conditions else "1=1"
    cur.execute(f"DELETE FROM {table} WHERE {where}", tuple(params))
    conn.commit()
    conn.close()
