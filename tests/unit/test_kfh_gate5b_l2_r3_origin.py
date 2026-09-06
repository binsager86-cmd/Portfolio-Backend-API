"""Gate 5B-L2-R3 exact local Origin boundary tests."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_connector.kfh_gate5b.local_api import (
    CONNECT_AND_FETCH_PATH,
    HEALTH_PATH,
    LOCAL_SAHAM_ORIGIN,
    NONCE_HEADER,
    canonicalize_local_origin,
    create_local_test_app,
)

NONCE = "GATE5B-L2-R3-LOCAL-NONCE-1234567890"
BODY = {
    "username": "SYNTHETIC-USER",
    "password": "SYNTHETIC-PASSWORD",
    "fromDate": "20251001",
    "toDate": "20260902",
}


class BoundaryOnlyController:
    """A zero-I/O controller proving the request passed only the local boundary."""

    def __init__(self) -> None:
        self.connect_calls = 0
        self.close_calls = 0

    async def connect(self, _payload: Any) -> dict[str, Any]:
        self.connect_calls += 1
        return {
            "connection": "DISCONNECTED",
            "result": "IDLE",
            "failure": None,
            "stage": "BOUNDARY_TEST_ONLY",
            "financialWrites": 0,
            "sqliteWrites": 0,
            "postgresqlWrites": 0,
        }

    async def close(self) -> dict[str, Any]:
        self.close_calls += 1
        return {"result": "IDLE"}


def client_for(
    controller: BoundaryOnlyController,
    *,
    allowed_origin: str = LOCAL_SAHAM_ORIGIN,
    configured: bool = True,
) -> TestClient:
    app = create_local_test_app(
        controller=controller,
        nonce=NONCE,
        allowed_origin=allowed_origin,
        allowed_origin_configured=configured,
    )
    return TestClient(app, client=("127.0.0.1", 49152))


def assert_origin_rejected(origin: str, reason: str, category: str) -> None:
    controller = BoundaryOnlyController()
    with client_for(controller) as client:
        response = client.get(
            HEALTH_PATH,
            headers={"Origin": origin, NONCE_HEADER: NONCE},
        )
    assert response.status_code == 403
    assert response.json()["failure"] == "ORIGIN_REJECTED"
    assert response.json()["failureReason"] == reason
    assert response.json()["originFailureCategory"] == category
    assert response.json()["originStatus"] == "REJECTED"
    assert controller.connect_calls == 0


def test_exact_localhost_origin_health_is_accepted_and_reports_loaded_config() -> None:
    controller = BoundaryOnlyController()
    with client_for(controller) as client:
        response = client.get(
            HEALTH_PATH,
            headers={"Origin": LOCAL_SAHAM_ORIGIN, NONCE_HEADER: NONCE},
        )
    assert response.status_code == 200
    assert response.json() == {
        "status": "OK",
        "localTestEnabled": True,
        "previewHandoffVersion": 1,
        "serverAllowedOrigin": LOCAL_SAHAM_ORIGIN,
        "serverAllowedOriginConfigured": True,
    }
    assert response.headers["access-control-allow-origin"] == LOCAL_SAHAM_ORIGIN
    assert response.headers["access-control-allow-origin"] != "*"
    assert "access-control-allow-credentials" not in response.headers
    assert controller.connect_calls == 0


def test_trailing_slash_configuration_canonicalizes_for_cors_and_validation() -> None:
    controller = BoundaryOnlyController()
    with client_for(controller, allowed_origin=f"{LOCAL_SAHAM_ORIGIN}/") as client:
        response = client.get(
            HEALTH_PATH,
            headers={"Origin": LOCAL_SAHAM_ORIGIN, NONCE_HEADER: NONCE},
        )
        preflight = client.options(
            CONNECT_AND_FETCH_PATH,
            headers={
                "Origin": LOCAL_SAHAM_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": f"Content-Type,{NONCE_HEADER}",
            },
        )
        assert client.app.state.allowed_origin == LOCAL_SAHAM_ORIGIN
    assert canonicalize_local_origin(f"{LOCAL_SAHAM_ORIGIN}/") == LOCAL_SAHAM_ORIGIN
    assert response.status_code == 200
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == LOCAL_SAHAM_ORIGIN


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:8081",
        "http://localhost:8082",
        "https://localhost:8081",
        "https://foreign.example:443",
    ],
)
def test_valid_but_unauthorized_origins_are_not_allowed(origin: str) -> None:
    assert_origin_rejected(origin, "NOT_ALLOWED", "ORIGIN_NOT_ALLOWED")


@pytest.mark.parametrize(
    "origin",
    [
        "not-an-origin",
        "http://localhost",
        "http://localhost:8081/path",
        "http://localhost:8081?query=yes",
        "http://localhost:8081#fragment",
    ],
)
def test_malformed_origins_are_format_invalid(origin: str) -> None:
    assert_origin_rejected(origin, "FORMAT_INVALID", "ORIGIN_FORMAT_INVALID")


def test_missing_origin_is_rejected_before_nonce_or_controller() -> None:
    controller = BoundaryOnlyController()
    with client_for(controller) as client:
        response = client.get(HEALTH_PATH, headers={NONCE_HEADER: NONCE})
    assert response.status_code == 403
    assert response.json()["failureReason"] == "MISSING"
    assert response.json()["originStatus"] == "REJECTED"
    assert response.json()["nonceStatus"] == "NOT_OBSERVED"
    assert controller.connect_calls == 0


def test_127_loopback_is_rejected_by_default_but_can_be_explicitly_authorized() -> None:
    loopback = "http://127.0.0.1:8081"
    assert_origin_rejected(loopback, "NOT_ALLOWED", "ORIGIN_NOT_ALLOWED")
    controller = BoundaryOnlyController()
    with client_for(controller, allowed_origin=loopback) as client:
        response = client.get(
            HEALTH_PATH,
            headers={"Origin": loopback, NONCE_HEADER: NONCE},
        )
    assert response.status_code == 200
    assert response.json()["serverAllowedOrigin"] == loopback


def test_connect_uses_same_origin_source_without_kfh_or_database_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_sqlite(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("SQLite must not be opened by the Origin boundary test")

    monkeypatch.setattr(sqlite3, "connect", forbidden_sqlite)
    controller = BoundaryOnlyController()
    with client_for(controller) as client:
        response = client.post(
            CONNECT_AND_FETCH_PATH,
            headers={"Origin": LOCAL_SAHAM_ORIGIN, NONCE_HEADER: NONCE},
            json=BODY,
        )
    assert response.status_code == 200
    assert controller.connect_calls == 1
    assert response.json()["originStatus"] == "ACCEPTED"
    assert response.json()["serverAllowedOrigin"] == LOCAL_SAHAM_ORIGIN
    assert response.json()["serverAllowedOriginConfigured"] is True
    assert response.json()["financialWrites"] == 0
    assert response.json()["sqliteWrites"] == 0
    assert response.json()["postgresqlWrites"] == 0
    assert response.headers["access-control-allow-origin"] == LOCAL_SAHAM_ORIGIN


def test_rejected_origin_never_reaches_kfh_controller_or_body() -> None:
    controller = BoundaryOnlyController()
    with client_for(controller) as client:
        response = client.post(
            CONNECT_AND_FETCH_PATH,
            headers={"Origin": "http://localhost:9999", NONCE_HEADER: NONCE},
            content=b'{"username":"must-not-be-read"',
        )
    assert response.status_code == 403
    assert response.json()["failureReason"] == "NOT_ALLOWED"
    assert response.json()["bodyStatus"] == "NOT_OBSERVED"
    assert response.json()["financialWrites"] == 0
    assert response.json()["sqliteWrites"] == 0
    assert response.json()["postgresqlWrites"] == 0
    assert controller.connect_calls == 0
