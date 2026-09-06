"""Gate 3A isolated browser authentication boundary tests using fake data only."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from local_connector.kfh_gate3a.browser import (
    BROWSER_CONTEXT_OPTIONS,
    BROWSER_LAUNCH_OPTIONS,
    PlaywrightBrowserRuntime,
)
from local_connector.kfh_gate3a.connector import KfhGate3AConnector
from local_connector.kfh_gate3a.observer import KfhAuthenticationObserver
from local_connector.kfh_gate3a.policy import KfhApprovedAction, is_allowed_kfh_url
from local_connector.kfh_gate3a.state import KfhAuthState
from local_connector.kfh_gate3a.stdio_service import PUBLIC_COMMANDS, _validated_request


class FakeSession:
    def __init__(
        self,
        *,
        status: int | None = 200,
        login: bool = True,
        otp: bool = False,
        auth_failed: bool = False,
        auth_ui_count: int = 0,
        login_sequence: list[bool] | None = None,
    ) -> None:
        self.http_status = status
        self.login = login
        self.otp = otp
        self.auth_failed = auth_failed
        self.auth_ui_count = auth_ui_count
        self.closed = False
        self.logout_clicked = False
        self.__login_sequence = list(login_sequence) if login_sequence else None

    async def goto_kfh(self) -> int | None:
        return self.http_status

    async def login_ui_active(self) -> bool:
        if self.__login_sequence:
            return self.__login_sequence.pop(0)
        return self.login

    async def otp_ui_active(self) -> bool:
        return self.otp

    async def auth_failed_ui_active(self) -> bool:
        return self.auth_failed

    async def authenticated_ui_signal_count(self) -> int:
        return self.auth_ui_count

    async def logout(self) -> None:
        self.logout_clicked = True

    async def close(self) -> None:
        self.closed = True

    def is_closed(self) -> bool:
        return self.closed


class FakeRuntime:
    def __init__(self, sessions: list[FakeSession] | None = None, error: Exception | None = None) -> None:
        self.sessions = sessions or [FakeSession()]
        self.error = error
        self.open_count = 0
        self.on_inbound_frame = None
        self.on_closed = None
        self.on_document_failure = None

    async def open(self, **callbacks):
        self.open_count += 1
        if self.error:
            raise self.error
        self.on_inbound_frame = callbacks["on_inbound_frame"]
        self.on_closed = callbacks["on_closed"]
        self.on_document_failure = callbacks["on_document_failure"]
        return self.sessions[self.open_count - 1]


async def wait_for_state(
    connector: KfhGate3AConnector,
    expected: KfhAuthState,
    attempts: int = 50,
) -> None:
    for _ in range(attempts):
        if connector.status().state == expected:
            return
        await __import__("asyncio").sleep(0.002)
    assert connector.status().state == expected


@pytest.mark.asyncio
async def test_connector_opens_dedicated_headful_ephemeral_kfh_browser():
    runtime = FakeRuntime()
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    snapshot = await connector.connect()
    assert runtime.open_count == 1
    assert snapshot.state == KfhAuthState.LOGIN_REQUIRED
    assert BROWSER_LAUNCH_OPTIONS["headless"] is False
    assert "user_data_dir" not in BROWSER_LAUNCH_OPTIONS
    assert "storage_state" not in BROWSER_CONTEXT_OPTIONS
    source = inspect.getsource(PlaywrightBrowserRuntime)
    assert "launch_persistent_context" not in source
    assert "connect_over_cdp" not in source
    await connector.close()


def test_normal_browser_data_and_non_kfh_origins_are_inaccessible():
    assert is_allowed_kfh_url("https://trading.kfhtrade.com/") is True
    assert is_allowed_kfh_url("https://accounts.example.test/private") is False
    assert is_allowed_kfh_url("https://mail.example.test/") is False


@pytest.mark.asyncio
async def test_success_requires_independent_websocket_and_authenticated_ui_signals():
    session = FakeSession(login=False, auth_ui_count=1)
    runtime = FakeRuntime([session])
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    await connector.connect()
    runtime.on_inbound_frame(
        json.dumps(
            {
                "msgGrp": 5,
                "msgTyp": 101,
                "response": {"DAT": {"authSts": 1, "sesnId": "FAKE-SECRET"}},
            }
        )
    )
    await wait_for_state(connector, KfhAuthState.READY)
    await connector.close()


@pytest.mark.asyncio
async def test_websocket_auth_signal_alone_is_sufficient_without_dom_markers():
    """Regression: DOM markers are English-only text and never render on the
    Arabic-language KFH UI, so a real authenticated session must still be
    recognized from the WebSocket authSts=1 signal alone."""
    session = FakeSession(login=False, auth_ui_count=0)
    runtime = FakeRuntime([session])
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    await connector.connect()
    runtime.on_inbound_frame(
        json.dumps(
            {
                "msgGrp": 5,
                "msgTyp": 101,
                "response": {"DAT": {"authSts": 1, "sesnId": "FAKE-SECRET"}},
            }
        )
    )
    await wait_for_state(connector, KfhAuthState.READY)
    await connector.close()


@pytest.mark.asyncio
async def test_single_transient_login_ui_read_after_ready_does_not_expire_session():
    """Regression: the monitor polls the live page every poll_interval with no
    debounce. On a real, actively-rendering dashboard, a single transient
    DOM read (a re-render, a loading flicker right after authentication) must
    not be enough to kill an otherwise-healthy READY session before the
    statement fetch can start. Only a persistent (2-consecutive-poll) signal
    should trigger SESSION_EXPIRED."""

    class FlickerOnceSession(FakeSession):
        def __init__(self, *, flicker_at_call: int, **kwargs) -> None:
            super().__init__(**kwargs)
            self.calls = 0
            self.flicker_at_call = flicker_at_call

        async def login_ui_active(self) -> bool:
            self.calls += 1
            return self.calls == self.flicker_at_call

    session = FlickerOnceSession(flicker_at_call=10, login=False, auth_ui_count=0)
    runtime = FakeRuntime([session])
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    await connector.connect()
    runtime.on_inbound_frame(
        json.dumps(
            {
                "msgGrp": 5,
                "msgTyp": 101,
                "response": {"DAT": {"authSts": 1, "sesnId": "FAKE-SECRET"}},
            }
        )
    )
    await wait_for_state(connector, KfhAuthState.READY)
    await __import__("asyncio").sleep(0.3)
    assert connector.status().state == KfhAuthState.READY
    assert session.calls >= session.flicker_at_call
    await connector.close()


@pytest.mark.asyncio
async def test_single_authenticated_dom_label_is_not_sufficient_proof():
    session = FakeSession(login=False, auth_ui_count=1)
    connector = KfhGate3AConnector(FakeRuntime([session]), poll_interval=0.001)
    await connector.connect()
    await __import__("asyncio").sleep(0.02)
    assert connector.status().state == KfhAuthState.AUTHENTICATING
    await connector.close()


@pytest.mark.asyncio
async def test_failed_login_and_otp_required_states():
    failed = FakeSession(login=True, auth_failed=True)
    connector = KfhGate3AConnector(FakeRuntime([failed]), poll_interval=0.001)
    await connector.connect()
    await wait_for_state(connector, KfhAuthState.AUTH_FAILED)
    await connector.close()

    otp = FakeSession(login=False, otp=True)
    connector = KfhGate3AConnector(FakeRuntime([otp]), poll_interval=0.001)
    await connector.connect()
    await wait_for_state(connector, KfhAuthState.OTP_REQUIRED)
    await connector.close()


@pytest.mark.asyncio
async def test_browser_closed_network_interruption_and_kfh_unavailable():
    runtime = FakeRuntime()
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    await connector.connect()
    runtime.on_closed()
    await wait_for_state(connector, KfhAuthState.BROWSER_CLOSED)
    await connector.close()

    runtime = FakeRuntime()
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    await connector.connect()
    runtime.on_document_failure()
    await wait_for_state(connector, KfhAuthState.NETWORK_ERROR)
    await connector.close()

    connector = KfhGate3AConnector(FakeRuntime(error=ConnectionError("offline")))
    assert (await connector.connect()).state == KfhAuthState.NETWORK_ERROR
    await connector.close()

    connector = KfhGate3AConnector(FakeRuntime([FakeSession(status=503)]))
    assert (await connector.connect()).state == KfhAuthState.KFH_UNAVAILABLE
    await connector.close()


@pytest.mark.asyncio
async def test_session_expiration_logout_and_reconnect_flow():
    first = FakeSession(login=False, auth_ui_count=2)
    second = FakeSession(login=True)
    runtime = FakeRuntime([first, second])
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    await connector.connect()
    await wait_for_state(connector, KfhAuthState.READY)
    first.login = True
    first.auth_ui_count = 0
    await wait_for_state(connector, KfhAuthState.SESSION_EXPIRED)

    assert (await connector.reconnect()).state == KfhAuthState.LOGIN_REQUIRED
    assert first.closed is True
    assert runtime.open_count == 2
    assert (await connector.logout()).state == KfhAuthState.DISCONNECTED
    assert second.logout_clicked is True


def test_authentication_observer_discards_secrets_and_never_observes_outbound_frames():
    observer = KfhAuthenticationObserver()
    secret = "FAKE-PASSWORD-NEVER-STORE"
    observer.observe_inbound_frame(
        json.dumps(
            {
                "msgGrp": 5,
                "msgTyp": 101,
                "DAT": {"authSts": 1, "pwd": secret, "otp": "000000", "sesnId": "FAKE"},
            }
        )
    )
    assert observer.authenticated is True
    assert not hasattr(observer, "__dict__")
    assert secret not in repr(observer)
    assert not hasattr(observer, "observe_outbound_frame")


@pytest.mark.asyncio
async def test_logs_are_state_only_and_redact_by_exclusion(caplog):
    session = FakeSession(login=False, auth_ui_count=1)
    runtime = FakeRuntime([session])
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    with caplog.at_level("INFO", logger="saham.kfh_gate3a"):
        await connector.connect()
        runtime.on_inbound_frame(
            json.dumps(
                {
                    "msgGrp": 5,
                    "msgTyp": 101,
                    "DAT": {"authSts": 1, "pwd": "FAKE-PASSWORD", "sesnId": "FAKE-SESSION"},
                }
            )
        )
        await wait_for_state(connector, KfhAuthState.READY)
    log_text = caplog.text
    assert "READY" in log_text
    assert "FAKE-PASSWORD" not in log_text
    assert "FAKE-SESSION" not in log_text
    await connector.close()


def test_public_api_has_no_generic_browser_websocket_or_trading_capability():
    public = {name for name, _ in inspect.getmembers(KfhGate3AConnector, inspect.isfunction) if not name.startswith("_")}
    assert public == {"connect", "status", "wait_for_ready", "logout", "reconnect", "close"}
    forbidden = {
        "buy",
        "sell",
        "placeOrder",
        "cancel",
        "amend",
        "transfer",
        "withdraw",
        "sendRawMessage",
        "sendWebSocketMessage",
        "executeJavascriptOnKfh",
        "clickArbitrarySelector",
    }
    assert public.isdisjoint(forbidden)
    assert PUBLIC_COMMANDS == {"connect", "status", "wait_for_ready", "logout", "reconnect", "close"}
    assert KfhApprovedAction.__members__.keys() == {
        "LOGIN",
        "STATEMENTS",
        "PORTFOLIO",
        "ACCOUNT_SUMMARY",
        "LOGOUT",
    }


def test_stdio_boundary_rejects_credential_fields_and_arbitrary_request_ids():
    assert _validated_request('{"id":1,"method":"connect"}') == (1, "connect")
    with pytest.raises(ValueError):
        _validated_request('{"id":1,"method":"connect","password":"FAKE"}')
    with pytest.raises(ValueError):
        _validated_request('{"id":"FAKE-SECRET","method":"status"}')


def test_kfh_gate3a_source_contains_no_password_persistence_or_raw_login_sender():
    package_root = Path(__file__).parents[2] / "local_connector" / "kfh_gate3a"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package_root.glob("*.py"))
    forbidden = (
        "DAT.lgnNme",
        "DAT.pwd",
        "msgTyp 1",
        "storage_state=",
        "localStorage",
        "sessionStorage",
        "AsyncStorage",
        "send_websocket",
        "sendRawMessage",
    )
    for value in forbidden:
        assert value not in source
