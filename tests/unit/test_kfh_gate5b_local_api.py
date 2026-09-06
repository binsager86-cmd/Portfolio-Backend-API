"""Gate 5B local/dev UI API boundary tests; no real browser is opened."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import local_connector.kfh_gate5b.local_api as local_api
from local_connector.kfh_gate3a.state import KfhAuthState
from local_connector.kfh_gate5b.adapter import Gate5BLiveBridgeFailureError
from local_connector.kfh_gate5b.local_api import (
    LOCAL_HOST,
    LOCAL_PORT,
    LOCAL_SAHAM_ORIGIN,
    NONCE_HEADER,
    KfhLocalTestController,
    LocalConnectAndFetchRequest,
    create_local_test_app,
    redact_local_credentials,
)

NONCE = "LOCAL-TEST-NONCE-ONLY-1234567890"


@pytest.mark.parametrize(
    ("bridge_code", "expected"),
    [
        ("PAGINATION_PROTOCOL_DRIFT", ("PAGINATION_PROTOCOL_DRIFT", "PAGINATION")),
        ("PAGINATION_TOTAL_DRIFT", ("PAGINATION_TOTAL_DRIFT", "PAGINATION")),
        ("PAGINATION_CURSOR_FAILED", ("PAGINATION_CURSOR_FAILED", "PAGINATION")),
        ("PAGINATION_LIMIT_FAILED", ("PAGINATION_LIMIT_FAILED", "PAGINATION")),
    ],
)
def test_precise_typescript_pagination_failures_remain_classified(
    bridge_code: str,
    expected: tuple[str, str],
) -> None:
    error = Gate5BLiveBridgeFailureError(bridge_code, None, {})
    assert KfhLocalTestController._classify(error) == expected


def diagnostics(connection: str = "READY") -> dict[str, Any]:
    return {
        "connection": connection,
        "result": "IDLE",
        "failure": None,
        "stage": "IDLE",
        "usernameAutofillConfirmed": False,
        "passwordAutofillConfirmed": False,
        "loginSubmitTriggered": False,
        "otpSubmitTriggered": False,
        "visibleKfhBrowser": False,
        "gate3aReady": connection == "READY",
        "authenticatedHed": connection == "READY",
        "authenticatedSocket": connection == "READY",
        "requestsSent": 0,
        "responsesSeen": 0,
        "pagesCompleted": 0,
        "financialWrites": 0,
        "sqliteWrites": 0,
        "postgresqlWrites": 0,
    }


class StubController:
    def __init__(self) -> None:
        self.read_payload: Any | None = None
        self.closed = 0
        self.connect_received = False
        self.connect_invocations = 0
        self.selected_handle: str | None = None
        self.preview_payload: dict[str, Any] | None = None

    async def connect(self, payload: Any) -> dict[str, Any]:
        self.connect_invocations += 1
        self.connect_received = bool(
            payload.username.get_secret_value() and payload.password.get_secret_value()
        )
        return diagnostics("LOGIN_REQUIRED")

    async def status(self) -> dict[str, Any]:
        return diagnostics()

    async def select_account(self, payload: Any) -> dict[str, Any]:
        self.selected_handle = payload.handle
        return diagnostics("READY")

    async def submit_otp(self, payload: Any) -> dict[str, Any]:
        assert payload.verification_code.get_secret_value()
        return diagnostics("AUTHENTICATING")

    async def take_preview(self) -> dict[str, Any] | None:
        return self.preview_payload

    async def acknowledge_preview(self) -> dict[str, str]:
        self.preview_payload = None
        return {"status": "OK"}

    async def close(self) -> dict[str, Any]:
        self.closed += 1
        return diagnostics("DISCONNECTED")


def client_for(
    controller: StubController,
    *,
    host: str = "127.0.0.1",
    include_origin: bool = True,
) -> TestClient:
    app = create_local_test_app(controller=controller, nonce=NONCE)
    headers = {"Origin": LOCAL_SAHAM_ORIGIN} if include_origin else {}
    return TestClient(app, client=(host, 49152), headers=headers)


def test_local_api_exposes_only_one_click_and_lifecycle_routes_and_requires_nonce() -> None:
    controller = StubController()
    with client_for(controller) as client:
        assert client.app.state.request_body_logging is False
        paths = {
            route.path
            for route in client.app.routes
            if route.path.startswith("/local-test/kfh/")
        }
        assert paths == {
            "/local-test/kfh/health",
            "/local-test/kfh/connect-and-fetch",
            "/local-test/kfh/status",
            "/local-test/kfh/otp",
            "/local-test/kfh/select-account",
            "/local-test/kfh/preview",
            "/local-test/kfh/preview/ack",
            "/local-test/kfh/close",
        }
        assert all("{" not in path and "}" not in path for path in paths)
        assert client.get("/local-test/kfh/status").status_code == 403
        response = client.get(
            "/local-test/kfh/status", headers={NONCE_HEADER: NONCE}
        )
        assert response.status_code == 200
        assert response.json()["connection"] == "READY"
        assert response.headers["access-control-allow-origin"] == LOCAL_SAHAM_ORIGIN
        assert response.headers["access-control-allow-origin"] != "*"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["pragma"] == "no-cache"


def test_foreign_or_missing_browser_origin_is_rejected_before_credentials() -> None:
    body = {
        "username": "SYNTHETIC-USER",
        "password": "SYNTHETIC-PASSWORD",
        "fromDate": "20251001",
        "toDate": "20260902",
    }
    foreign_controller = StubController()
    with client_for(foreign_controller) as client:
        response = client.post(
            "/local-test/kfh/connect-and-fetch",
            headers={NONCE_HEADER: NONCE, "Origin": "https://foreign.example"},
            json=body,
        )
        assert response.status_code == 403
        assert response.json()["invalidField"] == "ORIGIN"
        assert response.json()["failureReason"] == "FORMAT_INVALID"
        assert response.json()["originFailureCategory"] == "ORIGIN_FORMAT_INVALID"
        assert response.json()["originStatus"] == "REJECTED"
        assert response.json()["nonceStatus"] == "NOT_OBSERVED"
        assert response.json()["contentTypeStatus"] == "NOT_OBSERVED"
        assert response.json()["bodyStatus"] == "NOT_OBSERVED"
        assert "access-control-allow-origin" not in response.headers
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["pragma"] == "no-cache"
        assert foreign_controller.connect_received is False

    missing_controller = StubController()
    with client_for(missing_controller, include_origin=False) as client:
        response = client.post(
            "/local-test/kfh/connect-and-fetch", headers={NONCE_HEADER: NONCE}, json=body
        )
        assert response.status_code == 403
        assert response.json()["invalidField"] == "ORIGIN"
        assert response.json()["failureReason"] == "MISSING"
        assert response.json()["originStatus"] == "REJECTED"
        assert response.json()["nonceStatus"] == "NOT_OBSERVED"
        assert response.json()["contentTypeStatus"] == "NOT_OBSERVED"
        assert response.json()["bodyStatus"] == "NOT_OBSERVED"
        assert missing_controller.connect_received is False


def test_only_approved_origin_cors_preflight_succeeds_without_wildcard() -> None:
    with client_for(StubController()) as client:
        response = client.options(
            "/local-test/kfh/connect-and-fetch",
            headers={
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": f"Content-Type,{NONCE_HEADER}",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == LOCAL_SAHAM_ORIGIN
        assert response.headers["access-control-allow-origin"] != "*"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["pragma"] == "no-cache"


def test_credentials_are_accepted_only_in_json_post_body() -> None:
    controller = StubController()
    body = {
        "username": "SYNTHETIC-USER",
        "password": "SYNTHETIC-PASSWORD",
        "fromDate": "20251001",
        "toDate": "20260902",
    }
    with client_for(controller) as client:
        query = client.post(
            "/local-test/kfh/connect-and-fetch?username=SYNTHETIC-USER",
            headers={NONCE_HEADER: NONCE},
            json=body,
        )
        assert query.status_code == 400

        header = client.post(
            "/local-test/kfh/connect-and-fetch",
            headers={NONCE_HEADER: NONCE, "X-KFH-Password": "SYNTHETIC-PASSWORD"},
            json=body,
        )
        assert header.status_code == 400

        get = client.get(
            "/local-test/kfh/connect-and-fetch?password=SYNTHETIC-PASSWORD",
            headers={NONCE_HEADER: NONCE},
        )
        assert get.status_code == 400

        wrong_type = client.post(
            "/local-test/kfh/connect-and-fetch",
            headers={NONCE_HEADER: NONCE, "Content-Type": "text/plain"},
            content='{"username":"SYNTHETIC-USER","password":"SYNTHETIC-PASSWORD"}',
        )
        assert wrong_type.status_code == 415
        assert controller.connect_received is False
        for response in (query, header, get, wrong_type):
            assert "SYNTHETIC" not in response.text
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["pragma"] == "no-cache"


def test_connect_accepts_credentials_only_with_nonce_and_never_echoes_them() -> None:
    controller = StubController()
    body = {
        "username": "SYNTHETIC-USER",
        "password": "SYNTHETIC-PASSWORD",
        "fromDate": "20251001",
        "toDate": "20260902",
    }
    with client_for(controller) as client:
        assert client.post("/local-test/kfh/connect-and-fetch", json=body).status_code == 403
        response = client.post(
            "/local-test/kfh/connect-and-fetch",
            headers={NONCE_HEADER: NONCE},
            json=body,
        )
        assert response.status_code == 200
        assert controller.connect_received is True
        assert "SYNTHETIC-USER" not in response.text
        assert "SYNTHETIC-PASSWORD" not in response.text


def test_credential_validation_and_redaction_never_echo_secret_values() -> None:
    with client_for(StubController()) as client:
        response = client.post(
            "/local-test/kfh/connect-and-fetch",
            headers={NONCE_HEADER: NONCE},
            json={
                "username": "SYNTHETIC-USER",
                "password": "SYNTHETIC-PASSWORD",
                "otp": "FORBIDDEN-OTP",
            },
        )
        assert response.status_code == 422
        assert response.json()["failure"] == "REQUEST_VALIDATION_FAILED"
        assert "SYNTHETIC" not in response.text
        assert "FORBIDDEN-OTP" not in response.text

    redacted = redact_local_credentials(
        {
            "username": "PRIVATE",
            "userName": "PRIVATE",
            "password": "PRIVATE",
            "passwd": "PRIVATE",
            "pwd": "PRIVATE",
            "otp": "PRIVATE",
            "pin": "PRIVATE",
            "safe": "VISIBLE",
        }
    )
    assert set(redacted.values()) == {"[REDACTED]", "VISIBLE"}


@pytest.mark.parametrize(
    ("missing_field", "invalid_field"),
    [
        ("username", "USERNAME"),
        ("password", "PASSWORD"),
    ],
)
def test_required_connect_fields_report_only_field_and_missing_reason(
    missing_field: str, invalid_field: str
) -> None:
    body = {
        "username": "TEST_USER",
        "password": "TEST_PASSWORD",
        "fromDate": "20251001",
        "toDate": "20260902",
    }
    body.pop(missing_field)
    controller = StubController()
    with client_for(controller) as client:
        response = client.post(
            "/local-test/kfh/connect-and-fetch",
            headers={NONCE_HEADER: NONCE},
            json=body,
        )
    assert response.status_code == 422
    assert response.json()["invalidField"] == invalid_field
    assert response.json()["failureReason"] == "MISSING"
    assert controller.connect_invocations == 0
    assert "TEST_" not in response.text


@pytest.mark.parametrize(
    ("field", "invalid_field"),
    [("fromDate", "FROM_DATE"), ("toDate", "TO_DATE")],
)
def test_invalid_calendar_date_reports_format_without_starting_browser(
    field: str, invalid_field: str
) -> None:
    body = {
        "username": "TEST_USER",
        "password": "TEST_PASSWORD",
        "fromDate": "20251001",
        "toDate": "20260902",
    }
    body[field] = "20260231"
    controller = StubController()
    with client_for(controller) as client:
        response = client.post(
            "/local-test/kfh/connect-and-fetch",
            headers={NONCE_HEADER: NONCE},
            json=body,
        )
    assert response.status_code == 422
    assert response.json()["invalidField"] == invalid_field
    assert response.json()["failureReason"] == "INVALID_FORMAT"
    assert controller.connect_invocations == 0


def test_reversed_dates_fail_out_of_range_before_browser() -> None:
    controller = StubController()
    with client_for(controller) as client:
        response = client.post(
            "/local-test/kfh/connect-and-fetch",
            headers={NONCE_HEADER: NONCE},
            json={
                "username": "TEST_USER",
                "password": "TEST_PASSWORD",
                "fromDate": "20260902",
                "toDate": "20251001",
            },
        )
    assert response.status_code == 422
    assert response.json()["invalidField"] == "FROM_DATE"
    assert response.json()["failureReason"] == "OUT_OF_RANGE"
    assert controller.connect_invocations == 0


def test_selected_account_is_strict_opaque_handle_and_preserves_leading_zeroes() -> None:
    controller = StubController()
    with client_for(controller) as client:
        accepted = client.post(
            "/local-test/kfh/select-account",
            headers={NONCE_HEADER: NONCE},
            json={"handle": "0000-OPAQUE-HANDLE"},
        )
        rejected = client.post(
            "/local-test/kfh/select-account",
            headers={NONCE_HEADER: NONCE},
            json={"handle": 1234},
        )
    assert accepted.status_code == 200
    assert controller.selected_handle == "0000-OPAQUE-HANDLE"
    assert rejected.status_code == 422
    assert rejected.json()["invalidField"] == "ACCOUNT_HANDLE"
    assert rejected.json()["failureReason"] == "WRONG_TYPE"


def test_unexpected_connect_field_is_rejected_without_browser_start() -> None:
    controller = StubController()
    with client_for(controller) as client:
        response = client.post(
            "/local-test/kfh/connect-and-fetch",
            headers={NONCE_HEADER: NONCE},
            json={
                "username": "TEST_USER",
                "password": "TEST_PASSWORD",
                "fromDate": "20251001",
                "toDate": "20260902",
                "session": "FORBIDDEN",
            },
        )
    assert response.status_code == 422
    assert response.json()["invalidField"] == "REQUEST_SHAPE"
    assert response.json()["failureReason"] == "UNEXPECTED_FIELD"
    assert controller.connect_invocations == 0
    assert "FORBIDDEN" not in response.text


def test_content_type_and_nonce_failures_are_field_specific_and_value_free() -> None:
    body = '{"username":"TEST_USER"}'
    controller = StubController()
    with client_for(controller) as client:
        wrong_type = client.post(
            "/local-test/kfh/connect-and-fetch",
            headers={NONCE_HEADER: NONCE, "Content-Type": "text/plain"},
            content=body,
        )
        missing_nonce = client.post(
            "/local-test/kfh/connect-and-fetch",
            headers={"Content-Type": "application/json"},
            content=body,
        )
        wrong_nonce = client.post(
            "/local-test/kfh/connect-and-fetch",
            headers={NONCE_HEADER: "WRONG-NONCE", "Content-Type": "application/json"},
            content=body,
        )
    assert (wrong_type.json()["invalidField"], wrong_type.json()["failureReason"]) == (
        "CONTENT_TYPE",
        "INVALID_FORMAT",
    )
    assert (missing_nonce.json()["invalidField"], missing_nonce.json()["failureReason"]) == (
        "NONCE",
        "MISSING",
    )
    assert (wrong_nonce.json()["invalidField"], wrong_nonce.json()["failureReason"]) == (
        "NONCE",
        "INVALID_FORMAT",
    )
    assert controller.connect_invocations == 0
    for response in (wrong_type, missing_nonce, wrong_nonce):
        assert "TEST_USER" not in response.text


def test_local_api_rejects_non_loopback_clients() -> None:
    with client_for(StubController(), host="192.168.1.20") as client:
        response = client.get(
            "/local-test/kfh/status", headers={NONCE_HEADER: NONCE}
        )
        assert response.status_code == 403


def test_obsolete_second_step_read_route_is_absent() -> None:
    with client_for(StubController()) as client:
        response = client.post(
            "/local-test/kfh/read-statement",
            headers={NONCE_HEADER: NONCE},
            json={
                "securityAccount": "SYNTHETIC-ACCOUNT",
                "fromDate": "20251001",
                "toDate": "20260902",
            },
        )
    assert response.status_code == 404
    assert "SYNTHETIC-ACCOUNT" not in response.text


def test_local_server_is_fixed_loopback_and_has_no_database_or_write_dependency() -> None:
    assert LOCAL_HOST == "127.0.0.1"
    assert LOCAL_PORT == 8765
    module = Path(inspect.getfile(KfhLocalTestController))
    source = module.read_text(encoding="utf-8")
    forbidden = (
        "0.0.0.0",
        "import sqlalchemy",
        "import sqlite3",
        "import psycopg",
        "from app.models",
        "from app.core.database",
        "broker_import",
        "cash_deposits",
        "portfolio_cash",
        "place_order",
        "cancel_order",
        '"/send"',
        '"/raw"',
        '"/ws"',
        '"/message"',
        '"/evaluate"',
    )
    assert not any(value in source.lower() for value in forbidden)


def test_controller_public_surface_contains_no_raw_or_trading_operation() -> None:
    public = {
        name
        for name, member in inspect.getmembers(KfhLocalTestController, inspect.isfunction)
        if not name.startswith("_")
    }
    assert public == {
        "connect",
        "select_account",
        "submit_otp",
        "status",
        "take_preview",
        "acknowledge_preview",
        "close",
    }


def test_preview_handoff_is_retryable_until_ack_then_cleared() -> None:
    controller = StubController()
    controller.preview_payload = {
        "brokerAccount": "KFH-LOCAL-" + ("a" * 64),
        "cashLogs": [],
        "unsettledCashLogs": [],
        "statementSummary": {
            "currency": "KWD",
            "open_balance": "0",
            "close_balance": "0",
            "total_deposit": "0",
            "total_withdrawal": "0",
            "total_buy": "0",
            "total_sell": "0",
            "total_other": "0",
            "vat_amount": "0",
        },
    }
    with client_for(controller) as client:
        assert client.get("/local-test/kfh/preview").status_code == 403
        first = client.get(
            "/local-test/kfh/preview", headers={NONCE_HEADER: NONCE}
        )
        assert first.status_code == 200
        assert first.headers["cache-control"] == "no-store"
        assert first.headers["pragma"] == "no-cache"
        assert first.headers["access-control-allow-origin"] == LOCAL_SAHAM_ORIGIN
        assert first.headers["access-control-allow-origin"] != "*"
        second = client.get(
            "/local-test/kfh/preview", headers={NONCE_HEADER: NONCE}
        )
        assert second.status_code == 200
        assert second.json() == first.json()
        ack = client.post(
            "/local-test/kfh/preview/ack",
            headers={NONCE_HEADER: NONCE},
            json={},
        )
        assert ack.status_code == 200
        assert ack.json() == {"status": "OK"}
        assert ack.headers["cache-control"] == "no-store"
        after_ack = client.get(
            "/local-test/kfh/preview", headers={NONCE_HEADER: NONCE}
        )
        assert after_ack.status_code == 409
        assert after_ack.json() == {"status": "NOT_READY"}


@pytest.mark.asyncio
async def test_controller_clears_backend_credential_references_without_logging(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class FakeRuntime:
        def __init__(self, *, on_statement_response_frame: Any) -> None:
            self.on_statement_response_frame = on_statement_response_frame

        async def _submit_login_credentials(
            self, username: str, password: str
        ) -> dict[str, bool]:
            assert username == "SYNTHETIC-USER"
            assert password == "SYNTHETIC-PASSWORD"
            return {
                "usernameAutofillConfirmed": True,
                "passwordAutofillConfirmed": True,
                "loginSubmitTriggered": True,
            }

        async def _clear_login_dom_credentials(self) -> None:
            return None

        async def _clear_otp_dom_credentials(self) -> None:
            return None

        async def _interactive_challenge_present(self) -> bool:
            return False

        def _visible_browser_enabled(self) -> bool:
            return False

        async def _authenticated_context_status(self) -> str:
            return "NOT_AVAILABLE"

        async def _authenticated_context_diagnostics(self) -> dict[str, bool | str]:
            return {
                "status": "NOT_AVAILABLE",
                "authenticatedHed": False,
                "authenticatedSocket": False,
            }

    class FakeConnector:
        def __init__(self, runtime: FakeRuntime) -> None:
            self.runtime = runtime

        async def connect(self) -> None:
            return None

        def status(self) -> SimpleNamespace:
            return SimpleNamespace(state=KfhAuthState.LOGIN_REQUIRED)

        async def wait_for_ready(self, timeout_seconds: int) -> SimpleNamespace:
            assert timeout_seconds == 300
            return SimpleNamespace(state=KfhAuthState.LOGIN_REQUIRED)

        async def logout(self) -> None:
            return None

    monkeypatch.setattr(local_api, "Gate5BLiveBrowserRuntime", FakeRuntime)
    monkeypatch.setattr(local_api, "KfhGate3AConnector", FakeConnector)
    credentials = LocalConnectAndFetchRequest(
        username="SYNTHETIC-USER",
        password="SYNTHETIC-PASSWORD",
        fromDate="20251001",
        toDate="20260902",
    )
    controller = KfhLocalTestController()

    response = await controller.connect(credentials)
    await asyncio.sleep(0)

    assert response["connection"] == "AUTHENTICATING"
    assert credentials.username.get_secret_value() == ""
    assert credentials.password.get_secret_value() == ""
    assert "SYNTHETIC-USER" not in caplog.text
    assert "SYNTHETIC-PASSWORD" not in caplog.text
    await controller.close()
