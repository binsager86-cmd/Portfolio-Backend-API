"""Gate 5A Debug R2 pre-auth lifecycle and sanitized final-record tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_connector.kfh_gate5a import ui_debug
from local_connector.kfh_gate5a.browser import (
    reduce_gate3a_compatible_auth_observation,
)
from local_connector.kfh_gate5a.ui_debug import Gate5ATempUiDebugger


class _Locator:
    first: _Locator

    def __init__(self) -> None:
        self.first = self

    async def count(self) -> int:
        return 0

    async def is_visible(self, timeout: int) -> bool:
        return False


class _Page:
    def __init__(self) -> None:
        self.url = "https://trading.kfhtrade.com/app?private=DROP-ME"
        self.frames = [self]
        self.main_frame = self

    async def evaluate(self, expression: str) -> str:
        assert expression == "document.readyState"
        return "complete"

    def get_by_text(self, pattern, exact: bool) -> _Locator:
        return _Locator()

    def locator(self, selector: str) -> _Locator:
        return _Locator()


class _Session:
    def __init__(self, *, login: bool = True) -> None:
        self.login = login

    async def login_ui_active(self) -> bool:
        return self.login

    async def otp_ui_active(self) -> bool:
        return False

    async def auth_failed_ui_active(self) -> bool:
        return False

    async def authenticated_ui_signal_count(self) -> int:
        return 0


def _records(debugger: Gate5ATempUiDebugger) -> list[dict]:
    return [
        json.loads(line)
        for line in debugger.path.read_text(encoding="utf-8").splitlines()
    ]


def _none_auth():
    return SimpleNamespace(
        response_seen=False, status_success=False, sanitized_frame=None
    )


def _success_auth():
    return reduce_gate3a_compatible_auth_observation(
        json.dumps(
            {
                "HED": {"msgGrp": 5, "msgTyp": 101},
                "DAT": {"authSts": 1, "password": "RAW-PASSWORD-MUST-DROP"},
            }
        )
    )


@pytest.mark.asyncio
async def test_r2_no_login_timeout_has_nonempty_lifecycle_and_final(tmp_path):
    debugger = Gate5ATempUiDebugger(tmp_path / "no-login.jsonl")
    page = _Page()
    debugger.browser_opened(page, _Session(login=True))
    await debugger.document_loaded(page.url)
    final = await debugger.finalize("LOGIN_REQUIRED")
    events = [record["event"] for record in _records(debugger)]
    assert events[0] == "DEBUG_STARTED"
    assert events[-1] == "DEBUG_FINAL"
    assert final["resultCategory"] == "LOGIN_NOT_COMPLETED"


def test_r2_browser_opened_event_is_written_immediately(tmp_path):
    debugger = Gate5ATempUiDebugger(tmp_path / "browser.jsonl")
    debugger.browser_opened(_Page(), _Session())
    assert [record["event"] for record in _records(debugger)][:2] == [
        "DEBUG_STARTED",
        "BROWSER_OPENED",
    ]


@pytest.mark.asyncio
async def test_r2_document_load_records_origin_only(tmp_path):
    debugger = Gate5ATempUiDebugger(tmp_path / "document.jsonl")
    page = _Page()
    debugger.browser_opened(page, _Session())
    await debugger.document_loaded(page.url)
    record = next(
        item for item in _records(debugger) if item["event"] == "KFH_DOCUMENT_LOADED"
    )
    assert record == {
        "event": "KFH_DOCUMENT_LOADED",
        "origin": "https://trading.kfhtrade.com",
    }


def test_r2_trade_socket_records_symbolic_role(tmp_path):
    debugger = Gate5ATempUiDebugger(tmp_path / "trade.jsonl")
    role = debugger.websocket_opened("wss://trading.kfhtrade.com/wstrs")
    assert role == "TRADE"
    assert _records(debugger)[-1] == {
        "event": "KFH_WEBSOCKET_OPENED",
        "socketRole": "TRADE",
    }


def test_r2_full_websocket_url_is_never_retained(tmp_path):
    debugger = Gate5ATempUiDebugger(tmp_path / "url.jsonl")
    debugger.websocket_opened(
        "wss://trading.kfhtrade.com/wstrs?token=FULL-URL-SECRET#fragment"
    )
    payload = debugger.path.read_text(encoding="utf-8")
    assert "FULL-URL-SECRET" not in payload
    assert "wss://" not in payload
    assert "token=" not in payload


def test_r2_raw_inbound_frame_is_never_retained(tmp_path):
    debugger = Gate5ATempUiDebugger(tmp_path / "raw.jsonl")
    raw = '{"HED":{"msgGrp":9},"DAT":{"password":"RAW-FRAME-SECRET"}}'
    debugger.observe_inbound_frame("TRADE", raw, _none_auth())
    payload = debugger.path.read_text(encoding="utf-8")
    assert "RAW-FRAME-SECRET" not in payload
    assert '"HED"' not in payload
    assert '"DAT"' not in payload


def test_r2_auth_5_101_is_reduced_to_identity_and_boolean(tmp_path):
    debugger = Gate5ATempUiDebugger(tmp_path / "auth.jsonl")
    debugger.observe_inbound_frame("TRADE", "RAW-PAYLOAD-MUST-DROP", _success_auth())
    records = _records(debugger)
    assert records[-2] == {
        "event": "AUTH_RESPONSE_IDENTITY_SEEN",
        "msgGrp": 5,
        "msgTyp": 101,
    }
    assert records[-1] == {"event": "AUTH_STATUS_REDUCED", "success": True}
    assert "RAW-PAYLOAD-MUST-DROP" not in debugger.path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_r2_auth_status_one_starts_ui_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_debug, "SAMPLE_OFFSETS_SECONDS", (0,))
    debugger = Gate5ATempUiDebugger(tmp_path / "sampling.jsonl")
    debugger.observe_inbound_frame("TRADE", "discard", _success_auth())
    debugger.start(_Page(), _Session(login=False))
    assert await debugger.wait(timeout_seconds=1)
    events = [record["event"] for record in _records(debugger)]
    assert "AUTH_SUCCESS_TRIGGER" in events
    assert "UI_SIGNAL_SAMPLE" in events


@pytest.mark.asyncio
async def test_r2_ui_sampling_cannot_start_before_auth_success(tmp_path):
    debugger = Gate5ATempUiDebugger(tmp_path / "gated.jsonl")
    debugger.start(_Page(), _Session())
    await debugger.finalize("AUTHENTICATING")
    events = [record["event"] for record in _records(debugger)]
    assert "AUTH_SUCCESS_TRIGGER" not in events
    assert "UI_SIGNAL_SAMPLE" not in events


@pytest.mark.asyncio
async def test_r2_owner_marker_has_no_authentication_effect(tmp_path):
    debugger = Gate5ATempUiDebugger(tmp_path / "marker.jsonl")
    debugger.owner_visual_login_marker()
    final = await debugger.finalize("AUTHENTICATING")
    assert final["ownerVisualLoginMarker"] is True
    assert final["auth"] == {
        "responseIdentitySeen": False,
        "statusExtracted": False,
        "statusSuccess": False,
    }


@pytest.mark.asyncio
async def test_r2_timeout_always_appends_debug_final(tmp_path):
    debugger = Gate5ATempUiDebugger(tmp_path / "timeout.jsonl")
    assert await debugger.wait(timeout_seconds=0.001) is False
    await debugger.finalize("AUTHENTICATING")
    assert _records(debugger)[-1]["event"] == "DEBUG_FINAL"


@pytest.mark.asyncio
async def test_r2_exception_always_appends_debug_final(tmp_path):
    debugger = Gate5ATempUiDebugger(tmp_path / "exception.jsonl")
    debugger.fail_safely()
    final = await debugger.finalize("CONNECTOR_ERROR")
    assert final["resultCategory"] == "DEBUG_FAILURE"
    assert _records(debugger)[-1]["event"] == "DEBUG_FINAL"


@pytest.mark.asyncio
async def test_r2_debug_records_prohibit_secret_fields_and_values(tmp_path):
    debugger = Gate5ATempUiDebugger(tmp_path / "secrets.jsonl")
    debugger.websocket_opened(
        "wss://trading.kfhtrade.com/wstrs?session=DROP-SESSION&account=DROP-ACCOUNT"
    )
    debugger.observe_inbound_frame(
        "TRADE",
        json.dumps(
            {
                "username": "DROP-USERNAME",
                "password": "DROP-PASSWORD",
                "otp": "DROP-OTP",
                "cookies": "DROP-COOKIE",
            }
        ),
        _none_auth(),
    )
    await debugger.finalize("AUTHENTICATING")
    payload = debugger.path.read_text(encoding="utf-8")
    for value in (
        "DROP-SESSION",
        "DROP-ACCOUNT",
        "DROP-USERNAME",
        "DROP-PASSWORD",
        "DROP-OTP",
        "DROP-COOKIE",
    ):
        assert value not in payload


def test_gate3a_r1_candidate_manifest_includes_repair_suite():
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
