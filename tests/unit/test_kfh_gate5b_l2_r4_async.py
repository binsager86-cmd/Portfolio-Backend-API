from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

import local_connector.kfh_gate5b.local_api as local_api
from local_connector.kfh_gate3a.state import KfhAuthState
from local_connector.kfh_gate5b.local_api import (
    CONNECT_AND_FETCH_PATH,
    HEALTH_PATH,
    NONCE_HEADER,
    KfhLocalTestController,
    create_local_test_app,
)

NONCE = "R4-NON-BLOCKING-NONCE-1234567890"
ORIGIN = "http://localhost:8081"
SELECTED_ACCOUNT = "0000-R4-SYNTHETIC-ACCOUNT"
REQUEST_BODY = {
    "username": "R4-SYNTHETIC-USER",
    "password": "R4-SYNTHETIC-PASSWORD",
    "fromDate": "20251001",
    "toDate": "20260902",
}


class DelayedRuntime:
    delay_seconds = 10.0
    instances = 0
    login_consumed = False
    login_dom_cleared = False
    otp_dom_cleared = False

    def __init__(self, *, on_statement_response_frame: Any) -> None:
        del on_statement_response_frame
        type(self).instances += 1

    @staticmethod
    def _visible_browser_enabled() -> bool:
        return False

    async def _submit_login_credentials(self, username: str, password: str) -> dict[str, bool]:
        assert username == REQUEST_BODY["username"]
        assert password == REQUEST_BODY["password"]
        type(self).login_consumed = True
        assert DelayedConnector.latest is not None
        DelayedConnector.latest.state = KfhAuthState.READY
        return {
            "usernameAutofillConfirmed": True,
            "passwordAutofillConfirmed": True,
            "loginSubmitTriggered": True,
        }

    async def _interactive_challenge_present(self) -> bool:
        return False

    async def _clear_login_dom_credentials(self) -> None:
        type(self).login_dom_cleared = True

    async def _clear_otp_dom_credentials(self) -> None:
        type(self).otp_dom_cleared = True

    async def _authenticated_context_diagnostics(self) -> dict[str, bool | str]:
        return {
            "status": "AVAILABLE",
            "authenticatedHed": True,
            "authenticatedSocket": True,
        }

    async def _authenticated_context_status(self) -> str:
        return "AVAILABLE"

    async def _mark_gate3a_ready(self) -> None:
        return None

    async def _discovered_accounts(self) -> list[dict[str, Any]]:
        return [
            {
                "secAccNum": SELECTED_ACCOUNT,
                "portNme": "R4 PORTFOLIO",
                "curr": "KWD",
                "isDefaultAccount": True,
            }
        ]


class DelayedConnector:
    latest: DelayedConnector | None = None
    instances = 0
    logout_calls = 0

    def __init__(self, runtime: DelayedRuntime) -> None:
        del runtime
        type(self).instances += 1
        type(self).latest = self
        self.state = KfhAuthState.DISCONNECTED

    async def connect(self) -> Any:
        self.state = KfhAuthState.OPENING_KFH
        await asyncio.sleep(DelayedRuntime.delay_seconds)
        self.state = KfhAuthState.LOGIN_REQUIRED
        return self.status()

    def status(self) -> Any:
        return type("Snapshot", (), {"state": self.state})()

    async def logout(self) -> Any:
        type(self).logout_calls += 1
        self.state = KfhAuthState.DISCONNECTED
        return self.status()


async def successful_read(
    adapter: Any,
    request: dict[str, Any],
    *,
    account_currency: str,
) -> dict[str, Any]:
    del adapter
    assert account_currency == "KWD"
    assert request["secAccNum"] == SELECTED_ACCOUNT
    assert request["startSeq"] == 0
    assert request["totalNoRec"] == 20
    return {
        "financialWritesPerformed": 0,
        "requestStartSeqProgression": [0, 20, 40, 60],
        "responseCashLogsCounts": [19, 20, 7, 0],
        "isNxtPagAvailSequence": [1, 1, 1, 0],
        "finalResponseObserved": True,
    }


@pytest.fixture(autouse=True)
def delayed_orchestration(monkeypatch: pytest.MonkeyPatch) -> None:
    DelayedRuntime.delay_seconds = 10.0
    DelayedRuntime.instances = 0
    DelayedRuntime.login_consumed = False
    DelayedRuntime.login_dom_cleared = False
    DelayedRuntime.otp_dom_cleared = False
    DelayedConnector.latest = None
    DelayedConnector.instances = 0
    DelayedConnector.logout_calls = 0
    monkeypatch.setattr(local_api, "Gate5BLiveBrowserRuntime", DelayedRuntime)
    monkeypatch.setattr(local_api, "KfhGate3AConnector", DelayedConnector)
    monkeypatch.setattr(local_api, "run_typescript_cash_statement_read", successful_read)


def make_app() -> tuple[Any, KfhLocalTestController]:
    controller = KfhLocalTestController()
    return create_local_test_app(nonce=NONCE, controller=controller), controller


def headers() -> dict[str, str]:
    return {"Origin": ORIGIN, NONCE_HEADER: NONCE}


@pytest.mark.asyncio
async def test_ten_second_browser_start_acknowledges_before_http_timeout_then_passes() -> None:
    app, controller = make_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        started = time.perf_counter()
        response = await client.post(CONNECT_AND_FETCH_PATH, headers=headers(), json=REQUEST_BODY)
        acknowledgement_seconds = time.perf_counter() - started

        assert response.status_code == 200
        assert acknowledgement_seconds < 1.0
        acknowledgement = response.json()
        assert acknowledgement["operationAccepted"] is True
        assert acknowledgement["operationActive"] is True
        assert acknowledgement["operationStage"] == "OPERATION_ACCEPTED"
        assert acknowledgement["failure"] is None
        assert "LOCAL_API_TIMEOUT" not in json.dumps(acknowledgement)
        for secret in REQUEST_BODY.values():
            assert secret not in json.dumps(acknowledgement)

        first_status = await client.get("/local-test/kfh/status", headers=headers())
        assert first_status.status_code == 200
        assert first_status.json()["operationActive"] is True

        deadline = asyncio.get_running_loop().time() + 12.0
        selected = False
        terminal: dict[str, Any] | None = None
        while asyncio.get_running_loop().time() < deadline:
            value = (await client.get("/local-test/kfh/status", headers=headers())).json()
            if not selected and value["operationStage"] == "AWAITING_ACCOUNT_SELECTION":
                assert len(value["availableAccounts"]) == 1
                option = value["availableAccounts"][0]
                assert set(option) == {"handle", "curr", "isDefaultAccount"}
                assert option["curr"] == "KWD"
                assert option["isDefaultAccount"] is True
                selection = await client.post(
                    "/local-test/kfh/select-account",
                    headers=headers(),
                    json={"handle": option["handle"]},
                )
                assert selection.status_code == 200
                selected = True
                continue
            if value["result"] == "PASS" and value["operationActive"] is False:
                terminal = value
                break
            await asyncio.sleep(0.1)

        assert selected is True
        assert terminal is not None
        assert terminal["operationStage"] == "PREVIEW_READY"
        assert terminal["requestStarts"] == [0, 20, 40, 60]
        assert terminal["financialWrites"] == 0
        assert terminal["sqliteWrites"] == 0
        assert terminal["postgresqlWrites"] == 0
        assert DelayedRuntime.login_consumed is True
        assert DelayedRuntime.login_dom_cleared is True
        assert DelayedRuntime.otp_dom_cleared is True
        await controller.close()


@pytest.mark.asyncio
async def test_fifteen_second_login_keeps_health_status_and_close_responsive() -> None:
    DelayedRuntime.delay_seconds = 15.0
    app, controller = make_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        acknowledgement = await client.post(
            CONNECT_AND_FETCH_PATH, headers=headers(), json=REQUEST_BODY
        )
        assert acknowledgement.status_code == 200
        await asyncio.sleep(0)

        for path in (HEALTH_PATH, "/local-test/kfh/status"):
            started = time.perf_counter()
            response = await client.get(path, headers=headers())
            assert time.perf_counter() - started < 1.0
            assert response.status_code == 200

        duplicate_started = time.perf_counter()
        duplicate = await client.post(
            CONNECT_AND_FETCH_PATH, headers=headers(), json=REQUEST_BODY
        )
        assert time.perf_counter() - duplicate_started < 1.0
        assert duplicate.status_code == 409
        assert duplicate.json()["failure"] == "OPERATION_ALREADY_ACTIVE"
        assert duplicate.json()["operationAccepted"] is False
        assert DelayedRuntime.instances == 1
        assert DelayedConnector.instances == 1

        close_started = time.perf_counter()
        closed = await client.post("/local-test/kfh/close", headers=headers())
        assert time.perf_counter() - close_started < 1.0
        assert closed.status_code == 200
        assert closed.json()["connection"] == "DISCONNECTED"
        assert closed.json()["operationActive"] is False

        after_close = await client.get("/local-test/kfh/status", headers=headers())
        assert after_close.json()["operationActive"] is False
        assert controller._operation_task is None
        assert DelayedConnector.logout_calls == 1
        assert DelayedRuntime.login_dom_cleared is True
        assert DelayedRuntime.otp_dom_cleared is True


class SlowHedRuntime(DelayedRuntime):
    """Authenticated HED context becomes AVAILABLE only after a few reads,
    matching real KFH: it is populated passively from later /wstrs traffic
    and is not guaranteed to exist the instant READY is reached."""

    available_after_call = 3
    calls = 0

    async def _authenticated_context_diagnostics(self) -> dict[str, bool | str]:
        type(self).calls += 1
        if type(self).calls < type(self).available_after_call:
            return {"status": "NOT_AVAILABLE", "authenticatedHed": False, "authenticatedSocket": False}
        return {"status": "AVAILABLE", "authenticatedHed": True, "authenticatedSocket": True}


@pytest.mark.asyncio
async def test_authenticated_hed_context_is_awaited_not_failed_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SlowHedRuntime.calls = 0
    SlowHedRuntime.available_after_call = 3
    DelayedRuntime.delay_seconds = 0.0
    monkeypatch.setattr(local_api, "Gate5BLiveBrowserRuntime", SlowHedRuntime)
    monkeypatch.setattr(KfhLocalTestController, "AUTHENTICATED_HED_WAIT_SECONDS", 2.0)
    app, controller = make_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        await client.post(CONNECT_AND_FETCH_PATH, headers=headers(), json=REQUEST_BODY)

        deadline = asyncio.get_running_loop().time() + 3.0
        selected = False
        terminal: dict[str, Any] | None = None
        while asyncio.get_running_loop().time() < deadline:
            value = (await client.get("/local-test/kfh/status", headers=headers())).json()
            if not selected and value["operationStage"] == "AWAITING_ACCOUNT_SELECTION":
                await client.post(
                    "/local-test/kfh/select-account",
                    headers=headers(),
                    json={"handle": value["availableAccounts"][0]["handle"]},
                )
                selected = True
                continue
            if value["operationActive"] is False:
                terminal = value
                break
            await asyncio.sleep(0.05)

        assert terminal is not None
        assert terminal["result"] == "PASS"
        assert terminal["failure"] is None
        assert SlowHedRuntime.calls >= SlowHedRuntime.available_after_call
        await controller.close()


@pytest.mark.asyncio
async def test_authenticated_hed_context_never_available_fails_after_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SlowHedRuntime.calls = 0
    SlowHedRuntime.available_after_call = 10_000
    DelayedRuntime.delay_seconds = 0.0
    monkeypatch.setattr(local_api, "Gate5BLiveBrowserRuntime", SlowHedRuntime)
    monkeypatch.setattr(KfhLocalTestController, "AUTHENTICATED_HED_WAIT_SECONDS", 0.2)
    app, controller = make_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        await client.post(CONNECT_AND_FETCH_PATH, headers=headers(), json=REQUEST_BODY)

        deadline = asyncio.get_running_loop().time() + 2.0
        selected = False
        terminal: dict[str, Any] | None = None
        while asyncio.get_running_loop().time() < deadline:
            value = (await client.get("/local-test/kfh/status", headers=headers())).json()
            if not selected and value["operationStage"] == "AWAITING_ACCOUNT_SELECTION":
                await client.post(
                    "/local-test/kfh/select-account",
                    headers=headers(),
                    json={"handle": value["availableAccounts"][0]["handle"]},
                )
                selected = True
                continue
            if value["operationActive"] is False:
                terminal = value
                break
            await asyncio.sleep(0.05)

        assert terminal is not None
        assert terminal["result"] == "FAILED_CLOSED"
        assert terminal["failure"] == "AUTHENTICATED_SOCKET_NOT_AVAILABLE"
        assert terminal["financialWrites"] == 0
        await controller.close()
