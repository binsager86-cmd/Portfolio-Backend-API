"""Gate 5A must be authentication-transparent to the closed Gate 3A connector."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from local_connector.kfh_gate3a.connector import KfhGate3AConnector
from local_connector.kfh_gate3a.state import KfhAuthState
from local_connector.kfh_gate5a.browser import (
    Gate5AAuthDiagnostics,
    reduce_gate3a_compatible_auth_observation,
    route_gate5a_inbound_frame,
)

SANITIZED_AUTH_SUCCESS = '{"msgGrp":5,"msgTyp":101,"DAT":{"authSts":1}}'
NESTED_AUTH_SUCCESS = json.dumps(
    {
        "HED": {"msgGrp": 5, "msgTyp": 101},
        "DAT": {"authSts": 1, "sesnId": "FAKE-MUST-DROP", "otp": "FAKE-MUST-DROP"},
    }
)


class FakeSession:
    def __init__(self) -> None:
        self.login = True
        self.auth_ui_count = 0
        self.auth_failed = False
        self.closed = False

    async def goto_kfh(self):
        return 200

    async def login_ui_active(self):
        return self.login

    async def otp_ui_active(self):
        return False

    async def auth_failed_ui_active(self):
        return self.auth_failed

    async def authenticated_ui_signal_count(self):
        return self.auth_ui_count

    async def logout(self):
        return None

    async def close(self):
        self.closed = True

    def is_closed(self):
        return self.closed


class FakeRuntime:
    def __init__(self, *, wrapped: bool) -> None:
        self.wrapped = wrapped
        self.session = FakeSession()
        self.on_auth_frame = None
        self.on_closed = None
        self.on_document_failure = None
        self.statement_frames: list[str | bytes] = []
        self.diagnostics = Gate5AAuthDiagnostics(browser_origin_allowed=True)

    async def open(self, **callbacks):
        self.on_auth_frame = callbacks["on_inbound_frame"]
        self.on_closed = callbacks["on_closed"]
        self.on_document_failure = callbacks["on_document_failure"]
        return self.session

    def emit_auth_success(self) -> None:
        if self.wrapped:
            route_gate5a_inbound_frame(
                NESTED_AUTH_SUCCESS,
                on_auth_frame=self.on_auth_frame,
                on_statement_response_frame=self.statement_frames.append,
                diagnostics=self.diagnostics,
            )
        else:
            self.on_auth_frame(SANITIZED_AUTH_SUCCESS)

    def emit_statement(self) -> None:
        frame = json.dumps(
            {"HED": {"msgGrp": 2, "msgTyp": 107}, "DAT": {"unqReqId": "FAKE"}}
        )
        if self.wrapped:
            route_gate5a_inbound_frame(
                frame,
                on_auth_frame=self.on_auth_frame,
                on_statement_response_frame=self.statement_frames.append,
                diagnostics=self.diagnostics,
            )


async def wait_for_state(connector: KfhGate3AConnector, expected: KfhAuthState) -> None:
    for _ in range(100):
        if connector.status().state == expected:
            return
        await asyncio.sleep(0.002)
    assert connector.status().state == expected


def state_sequence(caplog) -> list[str]:
    sequence = [KfhAuthState.DISCONNECTED.value]
    for record in caplog.records:
        if record.name != "saham.kfh_gate3a" or not record.message.startswith("KFH_GATE3A_STATE "):
            continue
        sequence.append(json.loads(record.message.split(" ", 1)[1])["state"])
    return sequence


async def run_branch(caplog, *, wrapped: bool, branch: str) -> list[str]:
    caplog.clear()
    runtime = FakeRuntime(wrapped=wrapped)
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    with caplog.at_level("INFO", logger="saham.kfh_gate3a"):
        await connector.connect()
        if branch in {"READY", "SESSION_EXPIRED"}:
            runtime.session.login = False
            runtime.session.auth_ui_count = 1
            await asyncio.sleep(0.005)
            runtime.emit_auth_success()
            await wait_for_state(connector, KfhAuthState.READY)
            if branch == "SESSION_EXPIRED":
                runtime.session.login = True
                runtime.session.auth_ui_count = 0
                await wait_for_state(connector, KfhAuthState.SESSION_EXPIRED)
        elif branch == "AUTH_FAILED":
            runtime.session.auth_failed = True
            await wait_for_state(connector, KfhAuthState.AUTH_FAILED)
        elif branch == "BROWSER_CLOSED":
            runtime.on_closed()
            await wait_for_state(connector, KfhAuthState.BROWSER_CLOSED)
        elif branch == "NETWORK_ERROR":
            runtime.on_document_failure()
            await wait_for_state(connector, KfhAuthState.NETWORK_ERROR)
        sequence = state_sequence(caplog)
        await connector.close()
    return sequence


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "branch",
    ["READY", "AUTH_FAILED", "BROWSER_CLOSED", "SESSION_EXPIRED", "NETWORK_ERROR"],
)
async def test_gate5a_wrapper_has_identical_gate3a_state_sequence(caplog, branch):
    baseline = await run_branch(caplog, wrapped=False, branch=branch)
    wrapped = await run_branch(caplog, wrapped=True, branch=branch)
    assert wrapped == baseline


@pytest.mark.asyncio
async def test_valid_reduced_auth_signal_reaches_ready_and_malformed_cannot(caplog):
    runtime = FakeRuntime(wrapped=True)
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    await connector.connect()
    runtime.session.login = False
    runtime.session.auth_ui_count = 1
    await asyncio.sleep(0.005)
    runtime.emit_auth_success()
    await wait_for_state(connector, KfhAuthState.READY)
    await connector.close()

    malformed = reduce_gate3a_compatible_auth_observation(
        json.dumps({"HED": {"msgGrp": 5, "msgTyp": 101}, "DAT": {"authSts": "bad"}})
    )
    assert malformed.response_seen is True
    assert malformed.sanitized_frame is None

    malformed_runtime = FakeRuntime(wrapped=True)
    malformed_connector = KfhGate3AConnector(malformed_runtime, poll_interval=0.001)
    await malformed_connector.connect()
    malformed_runtime.session.login = False
    malformed_runtime.session.auth_ui_count = 1
    await asyncio.sleep(0.005)
    route_gate5a_inbound_frame(
        json.dumps(
            {"HED": {"msgGrp": 5, "msgTyp": 101}, "DAT": {"authSts": "bad"}}
        ),
        on_auth_frame=malformed_runtime.on_auth_frame,
        on_statement_response_frame=malformed_runtime.statement_frames.append,
        diagnostics=malformed_runtime.diagnostics,
    )
    await asyncio.sleep(0.02)
    assert malformed_connector.status().state == KfhAuthState.AUTHENTICATING
    await malformed_connector.close()


def test_non_auth_frames_are_not_forwarded_and_statement_path_is_separate():
    auth_frames: list[str | bytes] = []
    statement_frames: list[str | bytes] = []
    diagnostics = Gate5AAuthDiagnostics()
    unrelated = json.dumps({"HED": {"msgGrp": 9, "msgTyp": 999}, "DAT": {"value": 1}})
    route_gate5a_inbound_frame(
        unrelated,
        on_auth_frame=auth_frames.append,
        on_statement_response_frame=statement_frames.append,
        diagnostics=diagnostics,
    )
    assert auth_frames == []
    assert statement_frames == []

    statement = json.dumps({"HED": {"msgGrp": 2, "msgTyp": 107}, "DAT": {}})
    route_gate5a_inbound_frame(
        statement,
        on_auth_frame=auth_frames.append,
        on_statement_response_frame=statement_frames.append,
        diagnostics=diagnostics,
    )
    assert auth_frames == []
    assert statement_frames == [statement]
    assert diagnostics.auth_response_seen is False


@pytest.mark.asyncio
async def test_statement_observation_cannot_change_authentication_state():
    runtime = FakeRuntime(wrapped=True)
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    await connector.connect()
    runtime.session.login = False
    runtime.session.auth_ui_count = 1
    runtime.emit_statement()
    await asyncio.sleep(0.02)
    assert connector.status().state == KfhAuthState.AUTHENTICATING
    assert len(runtime.statement_frames) == 1
    await connector.close()


def test_gate3a_r1_candidate_source_and_test_manifest_is_exact():
    root = Path(__file__).parents[2]
    paths = [Path("local_connector/__init__.py")]
    paths.extend(
        path.relative_to(root)
        for path in (root / "local_connector" / "kfh_gate3a").glob("*.py")
    )
    paths.append(Path("tests/unit/test_kfh_gate3a_connector.py"))
    paths.append(Path("tests/unit/test_kfh_gate3a_r1.py"))
    entries = []
    for relative_path in sorted(paths, key=lambda value: value.as_posix()):
        digest = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        entries.append(f"{digest}  {relative_path.as_posix()}")
    manifest = ("\n".join(entries) + "\n").encode()
    assert hashlib.sha256(manifest).hexdigest() == (
        "9ce5f2a37d6d47c867f0284599fd0582ca955837d7514e377417b1fcdc8afe7d"
    )
