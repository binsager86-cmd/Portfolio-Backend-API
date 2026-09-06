"""Gate 5B-L2 headless, Saham-only authentication-to-read orchestration tests."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import local_connector.kfh_gate5b.local_api as local_api
from local_connector.kfh_gate3a.state import KfhAuthState
from local_connector.kfh_gate5b.browser import (
    Gate5BLiveBrowserRuntime,
    _gate5b_browser_launch_options,
)
from local_connector.kfh_gate5b.local_api import (
    ACCOUNT_DISCOVERY_SUPPORTED,
    ACCOUNT_SELECTION_MODE,
    KfhLocalTestController,
    LocalConnectAndFetchRequest,
    LocalOtpRequest,
)


class ReadyRuntime:
    def __init__(self, *, on_statement_response_frame: Any) -> None:
        self.on_statement_response_frame = on_statement_response_frame
        self.login_clears = 0
        self.otp_clears = 0

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

    async def _interactive_challenge_present(self) -> bool:
        return False

    async def _mark_gate3a_ready(self) -> None:
        return None

    async def _clear_login_dom_credentials(self) -> None:
        self.login_clears += 1

    async def _clear_otp_dom_credentials(self) -> None:
        self.otp_clears += 1

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
                "secAccNum": "SYNTHETIC-ACCOUNT",
                "portNme": "SYNTHETIC PORTFOLIO",
                "curr": "KWD",
                "isDefaultAccount": True,
            }
        ]

    def _visible_browser_enabled(self) -> bool:
        return False


class ReadyConnector:
    def __init__(self, runtime: ReadyRuntime) -> None:
        self.runtime = runtime

    async def connect(self) -> None:
        return None

    def status(self) -> SimpleNamespace:
        return SimpleNamespace(state=KfhAuthState.READY)

    async def logout(self) -> None:
        return None


@pytest.mark.asyncio
async def test_one_saham_action_authenticates_then_reads_with_saham_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            "requestStartSeqProgression": [0, 20, 40, 60],
            "responseCashLogsCounts": [19, 20, 7, 0],
            "isNxtPagAvailSequence": [1, 1, 1, 0],
            "finalResponseObserved": True,
        }

    monkeypatch.setattr(local_api, "Gate5BLiveBrowserRuntime", ReadyRuntime)
    monkeypatch.setattr(local_api, "KfhGate3AConnector", ReadyConnector)
    monkeypatch.setattr(local_api, "run_typescript_cash_statement_read", fake_read)
    credentials = LocalConnectAndFetchRequest(
        username="SYNTHETIC-USER",
        password="SYNTHETIC-PASSWORD",
        fromDate="20251001",
        toDate="20260902",
    )
    controller = KfhLocalTestController()

    first = await controller.connect(credentials)
    assert first["visibleKfhBrowser"] is False
    assert first["operationAccepted"] is True
    assert first["operationActive"] is True
    assert first["operationStage"] == "OPERATION_ACCEPTED"
    assert first["loginSubmitTriggered"] is False
    assert credentials.username.get_secret_value() == ""
    assert credentials.password.get_secret_value() == ""

    result = first
    for _attempt in range(50):
        result = await controller.status()
        if result["operationStage"] == "AWAITING_ACCOUNT_SELECTION":
            break
        await asyncio.sleep(0.01)

    assert result["operationStage"] == "AWAITING_ACCOUNT_SELECTION"
    # Saham receives only an opaque handle and safe fields - never the real
    # secAccNum/portNme, which stay server-side only.
    assert len(result["availableAccounts"]) == 1
    option = result["availableAccounts"][0]
    assert set(option) == {"handle", "curr", "isDefaultAccount"}
    assert isinstance(option["handle"], str) and option["handle"]
    assert option["curr"] == "KWD"
    assert option["isDefaultAccount"] is True

    from local_connector.kfh_gate5b.local_api import LocalAccountSelectionRequest

    selected = await controller.select_account(
        LocalAccountSelectionRequest(handle=option["handle"])
    )
    assert selected["operationStage"] == "FETCHING_STATEMENT"

    for _attempt in range(50):
        result = await controller.status()
        if result["result"] == "PASS":
            break
        await asyncio.sleep(0.01)

    assert result["result"] == "PASS"
    assert result["stage"] == "PREVIEW_READY"
    assert result["requestStarts"] == [0, 20, 40, 60]
    assert result["financialWrites"] == 0
    assert result["sqliteWrites"] == 0
    assert result["postgresqlWrites"] == 0
    assert captured_request == {
        "secAccNum": "SYNTHETIC-ACCOUNT",
        "frmDate": "20251001",
        "toDate": "20260902",
        "sortMode": 0,
        "startSeq": 0,
        "totalNoRec": 20,
    }
    await controller.close()


def test_headless_is_default_and_visible_browser_requires_separate_debug_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KFH_LOCAL_DEBUG_VISIBLE_BROWSER", raising=False)
    assert _gate5b_browser_launch_options()["headless"] is True
    monkeypatch.setenv("KFH_LOCAL_DEBUG_VISIBLE_BROWSER", "true")
    assert _gate5b_browser_launch_options()["headless"] is False


@pytest.mark.asyncio
async def test_otp_is_submitted_once_and_cleared_from_backend_request() -> None:
    class OtpRuntime:
        def __init__(self) -> None:
            self.submits = 0

        async def _submit_otp(self, verification_code: str) -> dict[str, bool]:
            assert verification_code == "123456"
            self.submits += 1
            return {"otpSubmitTriggered": True}

    class OtpConnector:
        def status(self) -> SimpleNamespace:
            return SimpleNamespace(state=KfhAuthState.OTP_REQUIRED)

    controller = KfhLocalTestController()
    runtime = OtpRuntime()
    controller._runtime = runtime  # type: ignore[assignment]
    controller._connector = OtpConnector()  # type: ignore[assignment]
    request = LocalOtpRequest(verificationCode="123456")

    result = await controller.submit_otp(request)

    assert result["connection"] == "OTP_REQUIRED"
    assert result["otpSubmitTriggered"] is True
    assert runtime.submits == 1
    assert request.verification_code.get_secret_value() == ""


@pytest.mark.asyncio
async def test_interactive_challenge_returns_explicit_fail_closed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ChallengeRuntime(ReadyRuntime):
        async def _interactive_challenge_present(self) -> bool:
            return True

    class AuthenticatingConnector(ReadyConnector):
        def status(self) -> SimpleNamespace:
            return SimpleNamespace(state=KfhAuthState.AUTHENTICATING)

    monkeypatch.setattr(local_api, "Gate5BLiveBrowserRuntime", ChallengeRuntime)
    monkeypatch.setattr(local_api, "KfhGate3AConnector", AuthenticatingConnector)
    controller = KfhLocalTestController()
    await controller.connect(
        LocalConnectAndFetchRequest(
            username="SYNTHETIC-USER",
            password="SYNTHETIC-PASSWORD",
            fromDate="20251001",
            toDate="20260902",
        )
    )
    await asyncio.sleep(0)
    result = await controller.status()
    assert result["connection"] == "INTERACTIVE_VERIFICATION_REQUIRED"
    assert result["failure"] == "INTERACTIVE_VERIFICATION_REQUIRED"
    assert result["result"] == "FAILED_CLOSED"
    await controller.close()


def test_account_discovery_is_supported_from_verified_protocol_evidence() -> None:
    """Real KFH evidence (2026-09): the post-login secAccLst response
    already carries secAccNum/portNme/curr/isDefaultAccount per account, so
    the owner never has to type an account identifier by hand."""
    assert ACCOUNT_DISCOVERY_SUPPORTED is True
    assert ACCOUNT_SELECTION_MODE == "DISCOVERED"
    with pytest.raises(ValueError):
        LocalConnectAndFetchRequest(
            username="SYNTHETIC-USER",
            password="SYNTHETIC-PASSWORD",
            securityAccount="SYNTHETIC-ACCOUNT",
            fromDate="20251001",
            toDate="20260902",
        )


@pytest.mark.asyncio
async def test_account_selection_rejects_any_value_not_in_the_discovered_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An account can never be guessed, typed, or otherwise injected - only
    one of KFH's own reported accounts may be selected."""
    from local_connector.kfh_gate5b.local_api import LocalAccountSelectionRequest

    monkeypatch.setattr(local_api, "Gate5BLiveBrowserRuntime", ReadyRuntime)
    monkeypatch.setattr(local_api, "KfhGate3AConnector", ReadyConnector)
    controller = KfhLocalTestController()
    await controller.connect(
        LocalConnectAndFetchRequest(
            username="SYNTHETIC-USER",
            password="SYNTHETIC-PASSWORD",
            fromDate="20251001",
            toDate="20260902",
        )
    )
    for _attempt in range(50):
        result = await controller.status()
        if result["operationStage"] == "AWAITING_ACCOUNT_SELECTION":
            break
        await asyncio.sleep(0.01)
    assert result["operationStage"] == "AWAITING_ACCOUNT_SELECTION"

    rejected = await controller.select_account(
        LocalAccountSelectionRequest(handle="NOT-A-REAL-DISCOVERED-HANDLE")
    )
    assert rejected["result"] == "FAILED_CLOSED"
    assert rejected["failure"] == "ACCOUNT_SELECTION_REJECTED"
    assert rejected["operationStage"] == "FAILED_CLOSED"
    await controller.close()


def test_l2_source_has_no_statement_ui_navigation_or_broadened_surface() -> None:
    public = {
        name
        for name, member in inspect.getmembers(Gate5BLiveBrowserRuntime, inspect.isfunction)
        if not name.startswith("_")
    }
    assert public == {"open", "send_cash_statement"}
    source = Path(inspect.getfile(Gate5BLiveBrowserRuntime)).read_text(encoding="utf-8")
    for forbidden in (
        "click_statement",
        "click_view",
        "click_next",
        "sendRawMessage",
        "placeOrder",
        "cancelOrder",
        '"msgTyp": 1',
        "storage_state=",
    ):
        assert forbidden not in source
