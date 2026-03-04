"""
User Onboarding Service — sets up default data for newly registered users.

Called after successful registration (both email/password and Google Sign-In).

Creates:
  - Default portfolios (KFH, BBYN, USA) with correct currencies
  - Default user settings (risk-free rate, etc.)
  - Initial portfolio_cash rows (zero balances)

All operations are idempotent — safe to call multiple times for the
same user_id without creating duplicates.
"""

import logging
import time
from typing import Optional

from app.core.database import exec_sql, query_val

logger = logging.getLogger(__name__)

# ── Default portfolios matching the Streamlit app ────────────────────
DEFAULT_PORTFOLIOS = [
    {"name": "KFH",  "currency": "KWD"},
    {"name": "BBYN", "currency": "KWD"},
    {"name": "USA",  "currency": "USD"},
]

# ── Default settings ────────────────────────────────────────────────
DEFAULT_SETTINGS = [
    ("rf_rate", "0.04"),          # Risk-free rate 4%
    ("default_currency", "KWD"),
]


def setup_new_user(user_id: int, username: Optional[str] = None) -> None:
    """
    One-time setup for a freshly registered user.

    Creates default portfolios, settings, and initial cash balances.
    Idempotent — skips creation if rows already exist.
    """
    label = username or f"user#{user_id}"
    logger.info("🆕  Setting up new user: %s (id=%d)", label, user_id)

    _create_default_portfolios(user_id)
    _create_default_settings(user_id)
    _create_initial_cash_balances(user_id)

    logger.info("✅  User setup complete for %s (id=%d)", label, user_id)


def _create_default_portfolios(user_id: int) -> None:
    """Create the three default portfolios if they don't already exist."""
    now = int(time.time())
    for p in DEFAULT_PORTFOLIOS:
        existing = query_val(
            "SELECT id FROM portfolios WHERE user_id = ? AND name = ?",
            (user_id, p["name"]),
        )
        if existing:
            continue
        try:
            exec_sql(
                "INSERT INTO portfolios (user_id, name, currency, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, p["name"], p["currency"], now),
            )
            logger.info("  📁 Created portfolio '%s' (%s) for user %d",
                         p["name"], p["currency"], user_id)
        except Exception as e:
            logger.warning("  ⚠️ Portfolio '%s' creation failed: %s", p["name"], e)


def _create_default_settings(user_id: int) -> None:
    """Seed default user_settings rows."""
    now = int(time.time())
    for key, value in DEFAULT_SETTINGS:
        existing = query_val(
            "SELECT 1 FROM user_settings WHERE user_id = ? AND setting_key = ?",
            (user_id, key),
        )
        if existing:
            continue
        try:
            exec_sql(
                "INSERT INTO user_settings (user_id, setting_key, setting_value, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, key, value, now),
            )
        except Exception as e:
            logger.warning("  ⚠️ Setting '%s' creation failed: %s", key, e)


def _create_initial_cash_balances(user_id: int) -> None:
    """Create zero-balance portfolio_cash rows so queries don't fail."""
    now = int(time.time())
    portfolio_currencies = {p["name"]: p["currency"] for p in DEFAULT_PORTFOLIOS}
    for name, ccy in portfolio_currencies.items():
        existing = query_val(
            "SELECT id FROM portfolio_cash WHERE user_id = ? AND portfolio = ?",
            (user_id, name),
        )
        if existing:
            continue
        try:
            exec_sql(
                "INSERT INTO portfolio_cash (user_id, portfolio, balance, currency, "
                "last_updated, manual_override) VALUES (?, ?, 0, ?, ?, 0)",
                (user_id, name, ccy, now),
            )
        except Exception as e:
            logger.warning("  ⚠️ portfolio_cash '%s' creation failed: %s", name, e)
