"""Gate 5B-L2-R4 browser Private Network Access boundary tests."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_connector.kfh_gate5b.local_api import (
    CONNECT_AND_FETCH_PATH,
    HEALTH_PATH,
    LOCAL_SAHAM_ORIGIN,
    NONCE_HEADER,
    create_local_test_app,
)

NONCE = "GATE5B-L2-R4-PNA-NONCE-1234567890"
UNAUTHORIZED_ORIGIN = "http://localhost:8082"
STATUS_PATH = "/local-test/kfh/status"
CLOSE_PATH = "/local-test/kfh/close"
BODY = {
    "username": "SYNTHETIC-USER",
    "password": "SYNTHETIC-PASSWORD",
    "fromDate": "20251001",
    "toDate": "20260902",
}


class NoIoController:
    """Protected-route probe that cannot launch Chromium, KFH, or storage."""

    def __init__(self) -> None:
        self.connect_calls = 0
        self.status_calls = 0
        self.close_calls = 0

    async def connect(self, _payload: Any) -> dict[str, Any]:
        self.connect_calls += 1
        return {
            "connection": "AUTHENTICATING",
            "result": "READING",
            "failure": None,
            "operationAccepted": True,
            "operationActive": True,
            "operationStage": "OPERATION_ACCEPTED",
            "financialWrites": 0,
            "sqliteWrites": 0,
            "postgresqlWrites": 0,
        }

    async def status(self) -> dict[str, Any]:
        self.status_calls += 1
        return {
            "connection": "DISCONNECTED",
            "result": "IDLE",
            "operationActive": False,
            "operationStage": "IDLE",
            "financialWrites": 0,
            "sqliteWrites": 0,
            "postgresqlWrites": 0,
        }

    async def close(self) -> dict[str, Any]:
        self.close_calls += 1
        return {
            "connection": "DISCONNECTED",
            "result": "IDLE",
            "operationActive": False,
            "operationStage": "DISCONNECTED",
            "financialWrites": 0,
            "sqliteWrites": 0,
            "postgresqlWrites": 0,
        }


def client_for(controller: NoIoController) -> TestClient:
    app = create_local_test_app(
        controller=controller,
        nonce=NONCE,
        allowed_origin=LOCAL_SAHAM_ORIGIN,
        allowed_origin_configured=True,
    )
    return TestClient(app, client=("127.0.0.1", 49152))


def pna_headers(
    method: str,
    *,
    origin: str = LOCAL_SAHAM_ORIGIN,
    requested_headers: str = NONCE_HEADER,
) -> dict[str, str]:
    return {
        "Origin": origin,
        "Access-Control-Request-Method": method,
        "Access-Control-Request-Private-Network": "true",
        "Access-Control-Request-Headers": requested_headers,
    }


def request_headers() -> dict[str, str]:
    return {"Origin": LOCAL_SAHAM_ORIGIN, NONCE_HEADER: NONCE}


def test_valid_pna_health_preflight_returns_exact_narrow_cors_headers() -> None:
    controller = NoIoController()
    with client_for(controller) as client:
        response = client.options(HEALTH_PATH, headers=pna_headers("GET"))

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == LOCAL_SAHAM_ORIGIN
    assert response.headers["access-control-allow-private-network"] == "true"
    assert "GET" in response.headers["access-control-allow-methods"]
    assert NONCE_HEADER.lower() in response.headers["access-control-allow-headers"].lower()
    assert response.headers["access-control-allow-origin"] != "*"
    assert "access-control-allow-credentials" not in response.headers
    assert controller.connect_calls == 0


def test_unauthorized_origin_pna_preflight_remains_rejected() -> None:
    controller = NoIoController()
    with client_for(controller) as client:
        response = client.options(
            HEALTH_PATH,
            headers=pna_headers("GET", origin=UNAUTHORIZED_ORIGIN),
        )

    assert response.status_code == 403
    assert response.json()["failure"] == "ORIGIN_REJECTED"
    assert response.json()["failureReason"] == "NOT_ALLOWED"
    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-private-network" not in response.headers
    assert controller.connect_calls == 0


def test_normal_non_pna_cors_preflight_remains_accepted() -> None:
    controller = NoIoController()
    headers = pna_headers("GET")
    headers.pop("Access-Control-Request-Private-Network")
    with client_for(controller) as client:
        response = client.options(HEALTH_PATH, headers=headers)

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == LOCAL_SAHAM_ORIGIN
    assert "access-control-allow-private-network" not in response.headers


def test_pna_preflight_does_not_bypass_nonce_and_valid_health_succeeds() -> None:
    controller = NoIoController()
    with client_for(controller) as client:
        preflight = client.options(HEALTH_PATH, headers=pna_headers("GET"))
        missing = client.get(HEALTH_PATH, headers={"Origin": LOCAL_SAHAM_ORIGIN})
        invalid = client.get(
            HEALTH_PATH,
            headers={"Origin": LOCAL_SAHAM_ORIGIN, NONCE_HEADER: "invalid-nonce-value"},
        )
        healthy = client.get(HEALTH_PATH, headers=request_headers())

    assert preflight.status_code == 200
    assert missing.status_code == 403
    assert missing.json()["failure"] == "NONCE_REJECTED"
    assert invalid.status_code == 403
    assert invalid.json()["failure"] == "NONCE_REJECTED"
    assert healthy.status_code == 200
    assert healthy.json() == {
        "status": "OK",
        "localTestEnabled": True,
        "previewHandoffVersion": 1,
        "serverAllowedOrigin": LOCAL_SAHAM_ORIGIN,
        "serverAllowedOriginConfigured": True,
    }


def test_pna_preserves_connect_status_close_and_zero_write_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_sqlite(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("PNA route tests must not open SQLite")

    monkeypatch.setattr(sqlite3, "connect", forbidden_sqlite)
    controller = NoIoController()
    with client_for(controller) as client:
        connect_preflight = client.options(
            CONNECT_AND_FETCH_PATH,
            headers=pna_headers("POST", requested_headers=f"Content-Type,{NONCE_HEADER}"),
        )
        status_preflight = client.options(STATUS_PATH, headers=pna_headers("GET"))
        close_preflight = client.options(CLOSE_PATH, headers=pna_headers("POST"))
        connected = client.post(
            CONNECT_AND_FETCH_PATH,
            headers=request_headers(),
            json=BODY,
        )
        status = client.get(STATUS_PATH, headers=request_headers())
        closed = client.post(CLOSE_PATH, headers=request_headers())

    for preflight in (connect_preflight, status_preflight, close_preflight):
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == LOCAL_SAHAM_ORIGIN
        assert preflight.headers["access-control-allow-private-network"] == "true"
    assert connected.status_code == 200
    assert connected.json()["operationStage"] == "OPERATION_ACCEPTED"
    assert status.status_code == 200
    assert closed.status_code == 200
    assert (controller.connect_calls, controller.status_calls, controller.close_calls) == (1, 1, 2)
    for response in (connected, status, closed):
        serialized = json.dumps(response.json())
        assert response.json()["financialWrites"] == 0
        assert response.json()["sqliteWrites"] == 0
        assert response.json()["postgresqlWrites"] == 0
        for secret in BODY.values():
            assert secret not in serialized
