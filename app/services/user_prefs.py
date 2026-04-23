"""
User Preferences — per-user UI settings (expertise level, language, feature flags).

Stored in the existing ``user_settings`` key-value table under setting_key
``user_prefs`` as JSON. No schema migration required.

Mirrors the mobile-app ``UserPreferences`` shape (minus ``notifications``,
which has its own row under ``notification_prefs``).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

SETTING_KEY = "user_prefs"

# Canonical defaults — keep in sync with mobile-app/src/store/userPrefsStore.ts
DEFAULT_PREFS: dict[str, Any] = {
    "expertiseLevel": "normal",        # normal | intermediate | advanced
    "language": "en",                  # en | ar
    "showAdvancedMetrics": False,
    "enableShariaFilter": False,
    "dividendFocus": False,
}

VALID_KEYS = set(DEFAULT_PREFS.keys())
_VALID_EXPERTISE = {"normal", "intermediate", "advanced"}
_VALID_LANG = {"en", "ar"}


def _coerce(key: str, value: Any) -> Any:
    """Validate + coerce a single preference value to its canonical type."""
    if key == "expertiseLevel":
        return value if value in _VALID_EXPERTISE else DEFAULT_PREFS[key]
    if key == "language":
        return value if value in _VALID_LANG else DEFAULT_PREFS[key]
    # Remaining keys are booleans
    return bool(value)


def _normalize(raw: Any) -> dict[str, Any]:
    out = dict(DEFAULT_PREFS)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in VALID_KEYS:
                out[k] = _coerce(k, v)
    return out


def get_prefs(db, user_id: int) -> dict[str, Any]:
    """Fetch a user's UI prefs (returns defaults if none stored)."""
    try:
        row = db.execute(
            text(
                "SELECT setting_value FROM user_settings "
                "WHERE user_id = :uid AND setting_key = :k"
            ),
            {"uid": user_id, "k": SETTING_KEY},
        ).fetchone()
        if not row or not row[0]:
            return dict(DEFAULT_PREFS)
        try:
            parsed = json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return dict(DEFAULT_PREFS)
        return _normalize(parsed)
    except Exception as e:
        logger.warning("user_prefs.get_prefs failed for user %s: %s", user_id, e)
        return dict(DEFAULT_PREFS)


def set_prefs(db, user_id: int, partial: dict[str, Any]) -> dict[str, Any]:
    """Upsert a user's UI prefs (partial — merges with existing)."""
    current = get_prefs(db, user_id)
    for k, v in (partial or {}).items():
        if k in VALID_KEYS and v is not None:
            current[k] = _coerce(k, v)

    payload = json.dumps(current)
    now = int(time.time())

    try:
        db.execute(
            text(
                "DELETE FROM user_settings "
                "WHERE user_id = :uid AND setting_key = :k"
            ),
            {"uid": user_id, "k": SETTING_KEY},
        )
        db.execute(
            text(
                "INSERT INTO user_settings (user_id, setting_key, setting_value, updated_at) "
                "VALUES (:uid, :k, :v, :ts)"
            ),
            {"uid": user_id, "k": SETTING_KEY, "v": payload, "ts": now},
        )
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning("user_prefs.set_prefs failed for user %s: %s", user_id, e)

    return current
