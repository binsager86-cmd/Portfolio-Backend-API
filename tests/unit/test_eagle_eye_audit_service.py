from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from app.core.database import exec_sql
from app.core.security import TokenData
from app.schemas.eagle_eye_audit import ChangeTransitionRequest
from app.services.eagle_eye import audit_service
from app.services.eagle_eye.audit_service import (
    change_status,
    create_change_request,
    create_event,
    ensure_schema,
    get_change_request,
    get_summary,
    list_change_requests,
    parse_json,
)
from app.api.v1.eagle_eye_audit import api_transition_change_request


@pytest.fixture(autouse=True)
def _reset_audit_tables():
    ensure_schema()
    for table in (
        "ee_change_status_history",
        "ee_change_requests",
        "ee_audit_events",
    ):
        exec_sql(f"DELETE FROM {table}")
    yield


def _user(user_id: int, is_admin: bool = False) -> TokenData:
    return TokenData(user_id=user_id, username=f"u{user_id}", is_admin=is_admin)


def _base_payload(status: str = "draft") -> dict:
    return {
        "title": "Scanner threshold update",
        "description": "Adjust scanner thresholds after review",
        "target_area": "scanner",
        "change_category": "enhancement",
        "status": status,
    }


def test_create_change_request_history_marks_created_state():
    req = create_change_request(_base_payload(status="draft"), _user(1))
    _, history = get_change_request(int(req["id"]))

    assert history
    assert history[0]["old_status"] == "(created)"
    assert history[0]["new_status"] == "draft"


def test_list_change_requests_filters_user_id_zero():
    create_change_request(_base_payload(), _user(1))

    rows, total = list_change_requests(
        status=None,
        target_area=None,
        requested_by_user_id=0,
        limit=100,
        offset=0,
    )

    assert total == 0
    assert rows == []


def test_change_status_self_review_blocked_by_default():
    req = create_change_request(_base_payload(status="proposed"), _user(1))

    with pytest.raises(HTTPException) as exc:
        change_status(
            request_id=int(req["id"]),
            actor=_user(1, is_admin=True),
            new_status="approved",
            note="self approval",
            set_reviewer=True,
        )

    assert exc.value.status_code == 403


def test_change_status_concurrency_guard_returns_409(monkeypatch):
    req = create_change_request(_base_payload(status="proposed"), _user(1))

    class _DummyCursor:
        rowcount = 0

        def execute(self, *args, **kwargs):
            return None

    class _DummyConn:
        def cursor(self):
            return _DummyCursor()

        def commit(self):
            return None

    class _DummyContext:
        def __enter__(self):
            return _DummyConn()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(audit_service, "get_connection", lambda: _DummyContext())

    with pytest.raises(HTTPException) as exc:
        change_status(
            request_id=int(req["id"]),
            actor=_user(2, is_admin=True),
            new_status="approved",
            note="reviewed",
            set_reviewer=True,
        )

    assert exc.value.status_code == 409
    assert "Concurrent modification" in str(exc.value.detail)


def test_get_summary_scoped_and_all_time_status_counts():
    req = create_change_request(_base_payload(status="draft"), _user(1))

    old_ts = int(time.time()) - (10 * 24 * 60 * 60)
    exec_sql("UPDATE ee_change_requests SET updated_at = ? WHERE id = ?", (old_ts, req["id"]))

    create_change_request(_base_payload(status="proposed"), _user(2))

    summary = get_summary(days=1)

    assert summary["change_requests_by_status"].get("draft", 0) == 0
    assert summary["change_requests_by_status"].get("proposed", 0) == 1
    assert summary["change_requests_by_status_all_time"].get("draft", 0) == 1
    assert summary["change_requests_by_status_all_time"].get("proposed", 0) == 1


def test_parse_json_logs_corrupt_payload(caplog):
    with caplog.at_level("WARNING"):
        parsed = parse_json("{invalid", "ee_audit_events.id=1.metadata_json")

    assert parsed is None
    assert "Corrupt JSON payload" in caplog.text
    assert "ee_audit_events.id=1.metadata_json" in caplog.text


def test_create_event_does_not_call_ensure_schema(monkeypatch):
    ensure_schema()

    def _fail(*args, **kwargs):
        raise AssertionError("ensure_schema should not be called during create_event")

    monkeypatch.setattr(audit_service, "ensure_schema", _fail)

    event = create_event(
        {
            "action": "phase_transition",
            "entity_type": "symbol",
            "entity_id": "ZAIN",
            "after_state": {"phase": "ACCUMULATION"},
        },
        _user(1),
    )

    assert event["action"] == "phase_transition"


def test_transition_endpoint_blocks_non_owner_non_admin():
    req = create_change_request(_base_payload(status="draft"), _user(1))

    with pytest.raises(HTTPException) as exc:
        api_transition_change_request(
            request_id=int(req["id"]),
            payload=ChangeTransitionRequest(new_status="proposed", note="submit"),
            current_user=_user(2, is_admin=False),
        )

    assert exc.value.status_code == 403
