"""Real frontend serializer to FastAPI one-click contract regression test."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import local_connector.kfh_gate5b.local_api as local_api
from local_connector.kfh_gate3a.state import KfhAuthState
from local_connector.kfh_gate5b.local_api import (
    LOCAL_SAHAM_ORIGIN,
    NONCE_HEADER,
    KfhLocalTestController,
    create_local_test_app,
)

NONCE = "LOCAL-CONTRACT-NONCE-1234567890"
CONTRACT_PROBE = (
    Path(__file__).resolve().parents[3]
    / "mobile-app"
    / "scripts"
    / "kfh-local-contract-probe.cjs"
)


class ContractRuntime:
    login_submits = 0

    def __init__(self, *, on_statement_response_frame: Any) -> None:
        self.on_statement_response_frame = on_statement_response_frame

    async def _submit_login_credentials(
        self, username: str, password: str
    ) -> dict[str, bool]:
        assert username == "TEST_USER"
        assert password == "TEST_PASSWORD"
        type(self).login_submits += 1
        return {
            "usernameAutofillConfirmed": True,
            "passwordAutofillConfirmed": True,
            "loginSubmitTriggered": True,
        }

    async def _interactive_challenge_present(self) -> bool:
        return False

    async def _mark_gate3a_ready(self) -> None:
        return None

    async def _clear_login_dom_credentials(self) -> None:
        return None

    async def _clear_otp_dom_credentials(self) -> None:
        return None

    async def _authenticated_context_status(self) -> str:
        return "AVAILABLE"

    async def _authenticated_context_diagnostics(self) -> dict[str, bool | str]:
        return {
            "status": "AVAILABLE",
            "authenticatedHed": True,
            "authenticatedSocket": True,
        }

    async def _discovered_accounts(self) -> list[dict[str, object]]:
        return [
            {
                "secAccNum": "0000-OPAQUE-ACCOUNT",
                "portNme": "CONTRACT PORTFOLIO",
                "curr": "KWD",
                "isDefaultAccount": True,
            }
        ]

    def _visible_browser_enabled(self) -> bool:
        return False


class ContractConnector:
    def __init__(self, runtime: ContractRuntime) -> None:
        self.runtime = runtime

    async def connect(self) -> None:
        return None

    def status(self) -> SimpleNamespace:
        return SimpleNamespace(state=KfhAuthState.READY)

    async def logout(self) -> None:
        return None


@pytest.mark.asyncio
async def test_actual_frontend_serialization_reaches_browser_once_and_preserves_account(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    probe = subprocess.run(
        ["node", str(CONTRACT_PROBE)],
        check=True,
        capture_output=True,
        text=True,
    )
    frontend_payload = json.loads(probe.stdout)
    captured_request: dict[str, object] = {}

    async def fake_read(
        _adapter: Any,
        request: dict[str, object],
        *,
        account_currency: str,
    ) -> dict[str, object]:
        assert account_currency == "KWD"
        captured_request.update(request)
        return {
            "financialWritesPerformed": 0,
            "requestStartSeqProgression": [0],
            "responseCashLogsCounts": [0],
            "isNxtPagAvailSequence": [0],
            "finalResponseObserved": True,
        }

    ContractRuntime.login_submits = 0
    monkeypatch.setattr(local_api, "Gate5BLiveBrowserRuntime", ContractRuntime)
    monkeypatch.setattr(local_api, "KfhGate3AConnector", ContractConnector)
    monkeypatch.setattr(local_api, "run_typescript_cash_statement_read", fake_read)
    controller = KfhLocalTestController()
    app = create_local_test_app(controller=controller, nonce=NONCE)

    with TestClient(
        app,
        client=("127.0.0.1", 49152),
        headers={"Origin": LOCAL_SAHAM_ORIGIN},
    ) as client:
        response = client.post(
            "/local-test/kfh/connect-and-fetch",
            headers={NONCE_HEADER: NONCE, "Content-Type": "application/json"},
            content=probe.stdout,
        )
        assert response.status_code == 200
        result = response.json()
        for _attempt in range(50):
            result = client.get(
                "/local-test/kfh/status", headers={NONCE_HEADER: NONCE}
            ).json()
            if result["operationStage"] == "AWAITING_ACCOUNT_SELECTION":
                break
            time.sleep(0.01)
        assert result["operationStage"] == "AWAITING_ACCOUNT_SELECTION"
        # Saham receives only an opaque handle and safe fields - never the
        # real secAccNum/portNme, which stay server-side only.
        assert len(result["availableAccounts"]) == 1
        option = result["availableAccounts"][0]
        assert set(option) == {"handle", "curr", "isDefaultAccount"}
        assert isinstance(option["handle"], str) and option["handle"]
        assert option["curr"] == "KWD"
        assert option["isDefaultAccount"] is True
        selected = client.post(
            "/local-test/kfh/select-account",
            headers={NONCE_HEADER: NONCE, "Content-Type": "application/json"},
            json={"handle": option["handle"]},
        )
        assert selected.status_code == 200
        for _attempt in range(50):
            result = client.get(
                "/local-test/kfh/status", headers={NONCE_HEADER: NONCE}
            ).json()
            if result["result"] == "PASS":
                break
            time.sleep(0.01)

    assert frontend_payload == {
        "username": "TEST_USER",
        "password": "TEST_PASSWORD",
        "fromDate": "20251001",
        "toDate": "20260902",
    }
    assert ContractRuntime.login_submits == 1
    assert result["loginOrchestrationStarted"] is True
    assert result["result"] == "PASS"
    assert captured_request["secAccNum"] == "0000-OPAQUE-ACCOUNT"
    assert captured_request["frmDate"] == "20251001"
    assert captured_request["toDate"] == "20260902"
    assert result["financialWrites"] == 0
    assert result["sqliteWrites"] == 0
    assert result["postgresqlWrites"] == 0
    assert "TEST_PASSWORD" not in caplog.text
    assert "0000-OPAQUE-ACCOUNT" not in caplog.text
