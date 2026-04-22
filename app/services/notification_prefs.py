"""
Notification Preferences — per-user toggles for which push categories to receive.

Stored in the existing `user_settings` key-value table under setting_key
``notification_prefs`` as JSON. No schema migration required.

Defaults: all categories enabled. A missing row means "send everything".
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

SETTING_KEY = "notification_prefs"

# Canonical keys mirror the mobile-app `NotificationPreferences` shape.
# Keep this list in sync with mobile-app/src/store/userPrefsStore.ts.
DEFAULT_PREFS: dict[str, bool] = {
    "newsNotifications": True,
    "portfolioUpdates": True,
    "priceAlerts": True,
    "dailyPriceUpdates": True,
}

VALID_KEYS = set(DEFAULT_PREFS.keys())


def _normalize(raw: Any) -> dict[str, bool]:
    """Coerce any stored value into the canonical {key: bool} shape with defaults."""
    out = dict(DEFAULT_PREFS)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in VALID_KEYS:
                out[k] = bool(v)
    return out


def get_prefs(db, user_id: int) -> dict[str, bool]:
    """Fetch a user's notification prefs (returns defaults if none stored)."""
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
        logger.warning("get_prefs failed for user %s: %s", user_id, e)
        return dict(DEFAULT_PREFS)


def set_prefs(db, user_id: int, partial: dict[str, Any]) -> dict[str, bool]:
    """Upsert a user's notification prefs (partial — merges with existing)."""
    current = get_prefs(db, user_id)
    for k, v in (partial or {}).items():
        if k in VALID_KEYS:
            current[k] = bool(v)

    payload = json.dumps(current)
    now = int(time.time())

    # SQLite + Postgres compatible upsert via DELETE + INSERT (table PK is
    # composite (user_id, setting_key) so this is atomic-enough for our needs).
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
        logger.warning("set_prefs failed for user %s: %s", user_id, e)

    return current


def filter_users_by_pref(db, user_ids: list[int], pref_key: str) -> list[int]:
    """
    Given a list of candidate user_ids and a pref key, return the subset
    whose preference for that key is enabled (defaulting to enabled when
    no record exists).
    """
    if not user_ids or pref_key not in VALID_KEYS:
        return list(user_ids)

    candidate_set = {int(u) for u in user_ids}

    try:
        # Fetch all rows for this setting key, filter in Python. Bounded
        # by the number of users who've explicitly saved prefs (small).
        rows = db.execute(
            text(
                "SELECT user_id, setting_value FROM user_settings "
                "WHERE setting_key = :k"
            ),
            {"k": SETTING_KEY},
        ).fetchall()
    except Exception as e:
        logger.warning("filter_users_by_pref query failed: %s — defaulting to all enabled", e)
        return list(user_ids)

    explicit: dict[int, bool] = {}
    for uid, raw in rows:
        uid_int = int(uid)
        if uid_int not in candidate_set:
            continue
        try:
            parsed = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        explicit[uid_int] = bool(_normalize(parsed).get(pref_key, True))

    # Users with no row default to True (enabled).
    return [uid for uid in user_ids if explicit.get(int(uid), True)]
