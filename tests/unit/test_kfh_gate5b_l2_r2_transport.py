"""Gate 5B-L2-R2 local transport boundary tests; no real KFH is contacted."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi.testclient import TestClient

from local_connector.kfh_gate5b.local_api import (
    CONNECT_AND_FETCH_PATH,
    HEALTH_PATH,
    LOCAL_SAHAM_ORIGIN,
    NONCE_HEADER,
    create_local_test_app,
)

NONCE = "LOCAL-R2-CONTRACT-NONCE-1234567890"
MOBILE_ROOT = Path(__file__).resolve().parents[3] / "mobile-app"
TRANSPORT_PROBE = MOBILE_ROOT / "scripts" / "kfh-local-transport-probe.cjs"


def base_diagnostics(**overrides: Any) -> dict[str, Any]:
    return {
        "connection": "LOGIN_REQUIRED",
        "result": "IDLE",
        "failure": None,
        "stage": "LOGIN_SUBMITTED",
        "operationAccepted": False,
        "operationActive": False,
        "operationStage": "IDLE",
        "startAckReceived": False,
        "failedStage": None,
        "loginOrchestrationStarted": False,
        "loginSubmitTriggered": True,
        "requestsSent": 0,
        "responsesSeen": 0,
        "pagesCompleted": 0,
        "financialWrites": 0,
        "sqliteWrites": 0,
        "postgresqlWrites": 0,
        **overrides,
    }


class TransportController:
    def __init__(self) -> None:
        self.connect_invocations = 0

    async def connect(self, payload: Any) -> dict[str, Any]:
        self.connect_invocations += 1
        assert payload.username.get_secret_value() == "TEST_USER"
        assert payload.password.get_secret_value() == "TEST_PASSWORD"
        return base_diagnostics(
            connection="AUTHENTICATING",
            result="READING",
            stage="OPERATION_ACCEPTED",
            operationAccepted=True,
            operationActive=True,
            operationStage="OPERATION_ACCEPTED",
            loginOrchestrationStarted=False,
            loginSubmitTriggered=False,
        )

    async def status(self) -> dict[str, Any]:
        return base_diagnostics(
            connection="READY",
            result="PASS",
            stage="PREVIEW_READY",
            operationAccepted=True,
            operationActive=False,
            operationStage="PREVIEW_READY",
        )

    async def submit_otp(self, _payload: Any) -> dict[str, Any]:
        return base_diagnostics()

    async def close(self) -> dict[str, Any]:
        return base_diagnostics(connection="DISCONNECTED")


def local_client(controller: Any) -> TestClient:
    app = create_local_test_app(controller=controller, nonce=NONCE)
    return TestClient(
        app,
        client=("127.0.0.1", 49152),
        headers={"Origin": LOCAL_SAHAM_ORIGIN},
    )


def valid_body() -> dict[str, str]:
    return {
        "username": "TEST_USER",
        "password": "TEST_PASSWORD",
        "fromDate": "20251001",
        "toDate": "20260902",
    }

def assert_subset(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert expected.items() <= actual.items()


def test_health_is_exact_origin_nonce_protected_and_no_store() -> None:
    with local_client(TransportController()) as client:
        missing_nonce = client.get(HEALTH_PATH)
        accepted = client.get(HEALTH_PATH, headers={NONCE_HEADER: NONCE})
    assert missing_nonce.status_code == 403
    assert missing_nonce.json()["failure"] == "NONCE_REJECTED"
    assert missing_nonce.json()["nonceStatus"] == "REJECTED"
    assert accepted.status_code == 200
    assert accepted.json() == {
        "status": "OK",
        "localTestEnabled": True,
        "previewHandoffVersion": 1,
        "serverAllowedOrigin": LOCAL_SAHAM_ORIGIN,
        "serverAllowedOriginConfigured": False,
    }
    assert accepted.headers["cache-control"] == "no-store"
    assert accepted.headers["pragma"] == "no-cache"
    assert accepted.headers["access-control-allow-origin"] == LOCAL_SAHAM_ORIGIN
    assert accepted.headers["access-control-allow-origin"] != "*"


def test_exact_connect_route_is_registered_once_and_old_routes_are_absent() -> None:
    app = create_local_test_app(controller=TransportController(), nonce=NONCE)
    exact = [
        route
        for route in app.routes
        if getattr(route, "path", None) == CONNECT_AND_FETCH_PATH
        and "POST" in (getattr(route, "methods", set()) or set())
    ]
    paths = {getattr(route, "path", None) for route in app.routes}
    assert app.state.connect_and_fetch_route_registered is True
    assert len(exact) == 1
    assert not paths.intersection(
        {
            "/local-test/kfh/connect",
            "/local-test/kfh/read-statement",
            "/local-test/kfh/connect-and-read",
        }
    )


def test_cors_preflight_allows_only_exact_saham_origin_without_credentials() -> None:
    with local_client(TransportController()) as client:
        allowed = client.options(
            CONNECT_AND_FETCH_PATH,
            headers={
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": f"Content-Type,{NONCE_HEADER}",
            },
        )
        foreign = client.options(
            CONNECT_AND_FETCH_PATH,
            headers={
                "Origin": "https://foreign.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": f"Content-Type,{NONCE_HEADER}",
            },
        )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == LOCAL_SAHAM_ORIGIN
    assert "access-control-allow-credentials" not in allowed.headers
    assert foreign.status_code == 403
    assert foreign.json()["failure"] == "ORIGIN_REJECTED"
    assert foreign.json()["originStatus"] == "REJECTED"
    assert "access-control-allow-origin" not in foreign.headers


def test_boundary_statuses_identify_first_observed_failure_without_false_values() -> None:
    controller = TransportController()
    with local_client(controller) as client:
        wrong_nonce = client.post(
            CONNECT_AND_FETCH_PATH,
            headers={NONCE_HEADER: "STALE-NONCE-1234567890"},
            json=valid_body(),
        )
        wrong_type = client.post(
            CONNECT_AND_FETCH_PATH,
            headers={NONCE_HEADER: NONCE, "Content-Type": "text/plain"},
            content=json.dumps(valid_body()),
        )
        malformed = client.post(
            CONNECT_AND_FETCH_PATH,
            headers={NONCE_HEADER: NONCE, "Content-Type": "application/json"},
            content="{malformed",
        )
        invalid_model = client.post(
            CONNECT_AND_FETCH_PATH,
            headers={NONCE_HEADER: NONCE},
            json={"username": "TEST_USER"},
        )
        accepted = client.post(
            CONNECT_AND_FETCH_PATH,
            headers={NONCE_HEADER: NONCE},
            json=valid_body(),
        )

    assert_subset(wrong_nonce.json(), {
        "failure": "NONCE_REJECTED",
        "failedStage": "NONCE",
        "originStatus": "ACCEPTED",
        "nonceStatus": "REJECTED",
        "contentTypeStatus": "NOT_OBSERVED",
        "bodyStatus": "NOT_OBSERVED",
    })
    assert_subset(wrong_type.json(), {
        "failure": "CONTENT_TYPE_REJECTED",
        "failedStage": "CONTENT_TYPE",
        "originStatus": "ACCEPTED",
        "nonceStatus": "ACCEPTED",
        "contentTypeStatus": "REJECTED",
        "bodyStatus": "NOT_OBSERVED",
    })
    assert_subset(malformed.json(), {
        "failure": "REQUEST_BODY_REJECTED",
        "failedStage": "REQUEST_BODY",
        "originStatus": "ACCEPTED",
        "nonceStatus": "ACCEPTED",
        "contentTypeStatus": "ACCEPTED",
        "bodyStatus": "REJECTED",
    })
    assert_subset(invalid_model.json(), {
        "failure": "REQUEST_VALIDATION_FAILED",
        "failedStage": "REQUEST_VALIDATION",
        "bodyStatus": "PARSED",
        "requestValidated": False,
    })
    assert_subset(accepted.json(), {
        "originStatus": "ACCEPTED",
        "nonceStatus": "ACCEPTED",
        "contentTypeStatus": "ACCEPTED",
        "bodyStatus": "PARSED",
        "requestValidated": True,
        "loginOrchestrationStarted": False,
    })
    assert controller.connect_invocations == 1


def test_backend_500_is_sanitized_and_no_store() -> None:
    class ExplodingController(TransportController):
        async def connect(self, _payload: Any) -> dict[str, Any]:
            raise RuntimeError("raw traceback with secret")

    with local_client(ExplodingController()) as client:
        response = client.post(
            CONNECT_AND_FETCH_PATH,
            headers={NONCE_HEADER: NONCE},
            json=valid_body(),
        )
    assert response.status_code == 500
    assert response.json()["failure"] == "LOCAL_API_HTTP_ERROR"
    assert response.json()["failedStage"] == "LOCAL_API_HTTP"
    assert "raw traceback" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"


def test_real_typescript_client_health_then_fastapi_connect_invokes_mock_once() -> None:
    controller = TransportController()
    requests: list[dict[str, Any]] = []
    process = subprocess.Popen(
        ["node", str(TRANSPORT_PROBE)],
        cwd=MOBILE_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    final: dict[str, Any] | None = None
    with local_client(controller) as client:
        while True:
            line = process.stdout.readline()
            assert line, process.stderr.read()
            message = json.loads(line)
            if message["kind"] == "RESULT":
                final = message["value"]
                break
            assert message["kind"] == "REQUEST"
            parsed = urlsplit(message["url"])
            assert (parsed.scheme, parsed.hostname, parsed.port) == (
                "http",
                "127.0.0.1",
                8765,
            )
            assert message["headers"]["Origin"] == LOCAL_SAHAM_ORIGIN
            requests.append(message)
            response = client.request(
                message["method"],
                parsed.path,
                headers=message["headers"],
                content=message["body"],
            )
            process.stdin.write(
                json.dumps({"status": response.status_code, "body": response.json()})
                + "\n"
            )
            process.stdin.flush()
    process.stdin.close()
    return_code = process.wait(timeout=10)
    stderr = process.stderr.read()

    assert return_code == 0, stderr
    assert [urlsplit(request["url"]).path for request in requests] == [
        HEALTH_PATH,
        CONNECT_AND_FETCH_PATH,
        "/local-test/kfh/status",
    ]
    assert requests[0]["body"] is None
    assert json.loads(requests[1]["body"]) == valid_body()
    assert final == {
        "healthStage": "LOCAL_CONNECTOR_READY",
        "healthHttpStatusClass": "2XX",
        "healthOriginStatus": "ACCEPTED",
        "healthNonceStatus": "ACCEPTED",
        "result": "READING",
        "operationAccepted": True,
        "operationActive": True,
        "operationStage": "OPERATION_ACCEPTED",
        "startAckReceived": True,
        "polledResult": "PASS",
        "polledOperationStage": "PREVIEW_READY",
        "originStatus": "ACCEPTED",
        "nonceStatus": "ACCEPTED",
        "contentTypeStatus": "ACCEPTED",
        "bodyStatus": "PARSED",
        "requestValidated": True,
        "loginOrchestrationStarted": False,
        "financialWrites": 0,
        "sqliteWrites": 0,
        "postgresqlWrites": 0,
    }
    assert controller.connect_invocations == 1
    assert "TEST_PASSWORD" not in json.dumps(final)
    assert "0000-OPAQUE-ACCOUNT" not in json.dumps(final)
