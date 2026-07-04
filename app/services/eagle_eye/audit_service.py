"""Eagle Eye audit/change-management service layer.

Contains:
- Data design bootstrap (DDL)
- Serialization helpers
- Change lifecycle transition logic
- Summary aggregations for audit/reporting
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from fastapi import HTTPException

from app.core.database import exec_sql, exec_sql_returning_id, get_connection, query_all, query_one, query_val
from app.core.security import TokenData

logger = logging.getLogger(__name__)

AUDIT_STATUSES = {
    "draft",
    "proposed",
    "needs_changes",
    "approved",
    "rejected",
    "implemented",
    "cancelled",
}

ALLOWED_TRANSITIONS = {
    "draft": {"proposed", "cancelled"},
    "proposed": {"needs_changes", "approved", "rejected", "cancelled"},
    "needs_changes": {"proposed", "cancelled"},
    "approved": {"implemented", "cancelled"},
    "rejected": set(),
    "implemented": set(),
    "cancelled": set(),
}


def now_ts() -> int:
    return int(time.time())


def to_json(value: Any) -> str:
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def parse_json(value: Any, context: str = "unknown") -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        logger.warning("Corrupt JSON payload in %s", context)
        return None


def row_get(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except Exception:
        try:
            d = dict(row)
            return d.get(key, default)
        except Exception:
            return default


def ensure_schema() -> None:
    stmts = [
        """
        CREATE TABLE IF NOT EXISTS ee_audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time INTEGER NOT NULL,
            actor_user_id INTEGER,
            actor_username TEXT,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT,
            change_type TEXT NOT NULL DEFAULT 'operation',
            before_state TEXT,
            after_state TEXT,
            rationale TEXT,
            risk_level TEXT NOT NULL DEFAULT 'low',
            trace_id TEXT,
            source TEXT NOT NULL DEFAULT 'api',
            metadata_json TEXT,
            concept_version TEXT,
            requires_follow_up INTEGER NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_ee_audit_event_time ON ee_audit_events(event_time DESC)",
        "CREATE INDEX IF NOT EXISTS idx_ee_audit_entity ON ee_audit_events(entity_type, entity_id)",
        "CREATE INDEX IF NOT EXISTS idx_ee_audit_action ON ee_audit_events(action)",
        """
        CREATE TABLE IF NOT EXISTS ee_change_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            requested_by_user_id INTEGER NOT NULL,
            requested_by_username TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            target_area TEXT NOT NULL,
            change_category TEXT NOT NULL,
            proposed_payload_json TEXT,
            status TEXT NOT NULL DEFAULT 'draft',
            reviewed_by_user_id INTEGER,
            reviewed_by_username TEXT,
            review_notes TEXT,
            approved_at INTEGER,
            rejected_at INTEGER,
            effective_from INTEGER,
            effective_to INTEGER,
            supersedes_request_id INTEGER
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_ee_change_status ON ee_change_requests(status)",
        "CREATE INDEX IF NOT EXISTS idx_ee_change_created ON ee_change_requests(created_at DESC)",
        """
        CREATE TABLE IF NOT EXISTS ee_change_status_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER NOT NULL,
            changed_at INTEGER NOT NULL,
            changed_by_user_id INTEGER NOT NULL,
            changed_by_username TEXT NOT NULL,
            old_status TEXT NOT NULL,
            new_status TEXT NOT NULL,
            note TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_ee_change_hist_req ON ee_change_status_history(request_id, changed_at DESC)",
    ]

    for sql in stmts:
        exec_sql(sql)


def serialize_event(row: Any) -> dict[str, Any]:
    return {
        "id": row_get(row, "id"),
        "event_time": row_get(row, "event_time"),
        "actor_user_id": row_get(row, "actor_user_id"),
        "actor_username": row_get(row, "actor_username"),
        "action": row_get(row, "action"),
        "entity_type": row_get(row, "entity_type"),
        "entity_id": row_get(row, "entity_id"),
        "change_type": row_get(row, "change_type"),
        "before_state": parse_json(row_get(row, "before_state"), f"ee_audit_events.id={row_get(row, 'id')}.before_state"),
        "after_state": parse_json(row_get(row, "after_state"), f"ee_audit_events.id={row_get(row, 'id')}.after_state"),
        "rationale": row_get(row, "rationale"),
        "risk_level": row_get(row, "risk_level"),
        "trace_id": row_get(row, "trace_id"),
        "source": row_get(row, "source"),
        "metadata": parse_json(row_get(row, "metadata_json"), f"ee_audit_events.id={row_get(row, 'id')}.metadata_json"),
        "concept_version": row_get(row, "concept_version"),
        "requires_follow_up": bool(row_get(row, "requires_follow_up", 0)),
    }


def serialize_change_request(row: Any) -> dict[str, Any]:
    return {
        "id": row_get(row, "id"),
        "created_at": row_get(row, "created_at"),
        "updated_at": row_get(row, "updated_at"),
        "requested_by_user_id": row_get(row, "requested_by_user_id"),
        "requested_by_username": row_get(row, "requested_by_username"),
        "title": row_get(row, "title"),
        "description": row_get(row, "description"),
        "target_area": row_get(row, "target_area"),
        "change_category": row_get(row, "change_category"),
        "proposed_payload": parse_json(
            row_get(row, "proposed_payload_json"),
            f"ee_change_requests.id={row_get(row, 'id')}.proposed_payload_json",
        ),
        "status": row_get(row, "status"),
        "reviewed_by_user_id": row_get(row, "reviewed_by_user_id"),
        "reviewed_by_username": row_get(row, "reviewed_by_username"),
        "review_notes": row_get(row, "review_notes"),
        "approved_at": row_get(row, "approved_at"),
        "rejected_at": row_get(row, "rejected_at"),
        "effective_from": row_get(row, "effective_from"),
        "effective_to": row_get(row, "effective_to"),
        "supersedes_request_id": row_get(row, "supersedes_request_id"),
    }


def insert_status_history(
    request_id: int,
    changed_by: TokenData,
    old_status: str,
    new_status: str,
    note: Optional[str],
) -> None:
    exec_sql(
        """
        INSERT INTO ee_change_status_history (
            request_id, changed_at, changed_by_user_id, changed_by_username,
            old_status, new_status, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request_id,
            now_ts(),
            changed_by.user_id,
            changed_by.username,
            old_status,
            new_status,
            note,
        ),
    )


def create_event(payload: dict[str, Any], user: TokenData) -> dict[str, Any]:
    event_id = exec_sql_returning_id(
        """
        INSERT INTO ee_audit_events (
            event_time, actor_user_id, actor_username, action, entity_type, entity_id,
            change_type, before_state, after_state, rationale, risk_level, trace_id,
            source, metadata_json, concept_version, requires_follow_up
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now_ts(),
            user.user_id,
            user.username,
            payload["action"],
            payload["entity_type"],
            payload.get("entity_id"),
            payload.get("change_type", "operation"),
            to_json(payload.get("before_state")),
            to_json(payload.get("after_state")),
            payload.get("rationale"),
            payload.get("risk_level", "low"),
            payload.get("trace_id"),
            payload.get("source", "api"),
            to_json(payload.get("metadata")),
            payload.get("concept_version"),
            1 if payload.get("requires_follow_up") else 0,
        ),
    )

    row = query_one("SELECT * FROM ee_audit_events WHERE id = ?", (event_id,))
    return serialize_event(row)


def list_events(
    action: Optional[str],
    entity_type: Optional[str],
    risk_level: Optional[str],
    since: Optional[int],
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    where_parts = []
    params: list[Any] = []

    if action:
        where_parts.append("action = ?")
        params.append(action)
    if entity_type:
        where_parts.append("entity_type = ?")
        params.append(entity_type)
    if risk_level:
        where_parts.append("risk_level = ?")
        params.append(risk_level)
    if since:
        where_parts.append("event_time >= ?")
        params.append(since)

    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    total = int(query_val(f"SELECT COUNT(1) FROM ee_audit_events {where_sql}", tuple(params)) or 0)

    rows = query_all(
        f"""
        SELECT *
        FROM ee_audit_events
        {where_sql}
        ORDER BY event_time DESC, id DESC
        LIMIT ? OFFSET ?
        """,
        tuple(params + [limit, offset]),
    )

    return [serialize_event(r) for r in rows], total


def create_change_request(payload: dict[str, Any], user: TokenData) -> dict[str, Any]:
    now = now_ts()

    request_id = exec_sql_returning_id(
        """
        INSERT INTO ee_change_requests (
            created_at, updated_at, requested_by_user_id, requested_by_username,
            title, description, target_area, change_category, proposed_payload_json,
            status, effective_from, effective_to, supersedes_request_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now,
            now,
            user.user_id,
            user.username,
            payload["title"],
            payload["description"],
            payload["target_area"],
            payload["change_category"],
            to_json(payload.get("proposed_payload")),
            payload.get("status", "draft"),
            payload.get("effective_from"),
            payload.get("effective_to"),
            payload.get("supersedes_request_id"),
        ),
    )

    insert_status_history(request_id, user, "(created)", payload.get("status", "draft"), "created")

    row = query_one("SELECT * FROM ee_change_requests WHERE id = ?", (request_id,))
    return serialize_change_request(row)


def list_change_requests(
    status: Optional[str],
    target_area: Optional[str],
    requested_by_user_id: Optional[int],
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    where_parts = []
    params: list[Any] = []

    if status:
        where_parts.append("status = ?")
        params.append(status)
    if target_area:
        where_parts.append("target_area = ?")
        params.append(target_area)
    if requested_by_user_id is not None:
        where_parts.append("requested_by_user_id = ?")
        params.append(requested_by_user_id)

    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    total = int(query_val(f"SELECT COUNT(1) FROM ee_change_requests {where_sql}", tuple(params)) or 0)

    rows = query_all(
        f"""
        SELECT *
        FROM ee_change_requests
        {where_sql}
        ORDER BY created_at DESC, id DESC
        LIMIT ? OFFSET ?
        """,
        tuple(params + [limit, offset]),
    )

    return [serialize_change_request(r) for r in rows], total


def get_change_request(request_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = query_one("SELECT * FROM ee_change_requests WHERE id = ?", (request_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Change request not found")

    history_rows = query_all(
        """
        SELECT request_id, changed_at, changed_by_user_id, changed_by_username,
               old_status, new_status, note
        FROM ee_change_status_history
        WHERE request_id = ?
        ORDER BY changed_at DESC, id DESC
        """,
        (request_id,),
    )

    history = [
        {
            "request_id": row_get(h, "request_id"),
            "changed_at": row_get(h, "changed_at"),
            "changed_by_user_id": row_get(h, "changed_by_user_id"),
            "changed_by_username": row_get(h, "changed_by_username"),
            "old_status": row_get(h, "old_status"),
            "new_status": row_get(h, "new_status"),
            "note": row_get(h, "note"),
        }
        for h in history_rows
    ]

    return serialize_change_request(row), history


def change_status(
    request_id: int,
    actor: TokenData,
    new_status: str,
    note: Optional[str] = None,
    set_reviewer: bool = False,
) -> dict[str, Any]:
    row = query_one("SELECT * FROM ee_change_requests WHERE id = ?", (request_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Change request not found")

    old_status = str(row_get(row, "status"))
    if new_status not in AUDIT_STATUSES:
        raise HTTPException(status_code=400, detail=f"Unsupported status: {new_status}")

    if new_status == old_status:
        raise HTTPException(status_code=400, detail="Status is already set to this value")

    if new_status not in ALLOWED_TRANSITIONS.get(old_status, set()):
        raise HTTPException(status_code=409, detail=f"Invalid transition: {old_status} -> {new_status}")

    if set_reviewer and actor.user_id == int(row_get(row, "requested_by_user_id") or 0) and not _allow_self_review():
        raise HTTPException(status_code=403, detail="Self-review is not allowed")

    now = now_ts()
    approved_at = now if new_status == "approved" else row_get(row, "approved_at")
    rejected_at = now if new_status == "rejected" else row_get(row, "rejected_at")

    if set_reviewer:
        reviewed_by_user_id = actor.user_id
        reviewed_by_username = actor.username
        review_notes = note
    else:
        reviewed_by_user_id = row_get(row, "reviewed_by_user_id")
        reviewed_by_username = row_get(row, "reviewed_by_username")
        review_notes = row_get(row, "review_notes")

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
        """
        UPDATE ee_change_requests
        SET
            status = ?,
            updated_at = ?,
            reviewed_by_user_id = ?,
            reviewed_by_username = ?,
            review_notes = ?,
            approved_at = ?,
            rejected_at = ?
        WHERE id = ? AND status = ?
        """,
        (
            new_status,
            now,
            reviewed_by_user_id,
            reviewed_by_username,
            review_notes,
            approved_at,
            rejected_at,
            request_id,
            old_status,
        ),
    )
        rowcount = int(getattr(cur, "rowcount", 0) or 0)
        conn.commit()

    if rowcount == 0:
        raise HTTPException(status_code=409, detail="Concurrent modification, reload and retry")

    insert_status_history(request_id, actor, old_status, new_status, note)

    updated = query_one("SELECT * FROM ee_change_requests WHERE id = ?", (request_id,))
    return serialize_change_request(updated)


def get_summary(days: int) -> dict[str, Any]:
    since = now_ts() - (days * 24 * 60 * 60)

    total_events = int(query_val("SELECT COUNT(1) FROM ee_audit_events WHERE event_time >= ?", (since,)) or 0)

    risk_rows = query_all(
        """
        SELECT risk_level, COUNT(1) AS c
        FROM ee_audit_events
        WHERE event_time >= ?
        GROUP BY risk_level
        """,
        (since,),
    )

    status_rows = query_all(
        """
        SELECT status, COUNT(1) AS c
        FROM ee_change_requests
        WHERE updated_at >= ?
        GROUP BY status
        """,
        (since,),
    )

    all_time_status_rows = query_all(
        """
        SELECT status, COUNT(1) AS c
        FROM ee_change_requests
        GROUP BY status
        """
    )

    latest_high_risk = query_all(
        """
        SELECT id, event_time, action, entity_type, entity_id, risk_level
        FROM ee_audit_events
        WHERE risk_level IN ('high', 'critical')
        ORDER BY event_time DESC, id DESC
        LIMIT 20
        """
    )

    return {
        "window_days": days,
        "total_events": total_events,
        "events_by_risk": {str(row_get(r, "risk_level")): int(row_get(r, "c", 0) or 0) for r in risk_rows},
        "change_requests_by_status": {str(row_get(r, "status")): int(row_get(r, "c", 0) or 0) for r in status_rows},
        "change_requests_by_status_all_time": {
            str(row_get(r, "status")): int(row_get(r, "c", 0) or 0) for r in all_time_status_rows
        },
        "high_risk_recent": [
            {
                "id": row_get(e, "id"),
                "event_time": row_get(e, "event_time"),
                "action": row_get(e, "action"),
                "entity_type": row_get(e, "entity_type"),
                "entity_id": row_get(e, "entity_id"),
                "risk_level": row_get(e, "risk_level"),
            }
            for e in latest_high_risk
        ],
    }


def _allow_self_review() -> bool:
    try:
        row = query_one("SELECT value_json FROM ee_engine_config WHERE key = ?", ("allow_self_review",))
    except Exception:
        return False
    if not row:
        return False
    parsed = parse_json(row_get(row, "value_json"), "ee_engine_config.key=allow_self_review.value_json")
    if isinstance(parsed, bool):
        return parsed
    if isinstance(parsed, dict):
        return bool(parsed.get("enabled", False))
    return False


def get_design() -> dict[str, Any]:
    return {
        "module": "Eagle Eye Audit",
        "purpose": [
            "Track immutable Eagle Eye audit events",
            "Manage concept-change lifecycle with approvals",
            "Provide reviewable status history and accountability",
        ],
        "tables": [
            {
                "name": "ee_audit_events",
                "description": "Append-only audit trail for Eagle Eye actions/config/model changes",
                "key_fields": [
                    "event_time",
                    "actor_user_id",
                    "action",
                    "entity_type",
                    "risk_level",
                    "before_state",
                    "after_state",
                    "concept_version",
                ],
            },
            {
                "name": "ee_change_requests",
                "description": "Primary change-management record for concept changes",
                "key_fields": [
                    "title",
                    "target_area",
                    "change_category",
                    "status",
                    "reviewed_by_user_id",
                ],
            },
            {
                "name": "ee_change_status_history",
                "description": "Immutable workflow transition log for each request",
                "key_fields": ["request_id", "old_status", "new_status", "changed_by_user_id", "note"],
            },
        ],
        "lifecycle": sorted(AUDIT_STATUSES),
        "allowed_transitions": {k: sorted(v) for k, v in ALLOWED_TRANSITIONS.items()},
    }
