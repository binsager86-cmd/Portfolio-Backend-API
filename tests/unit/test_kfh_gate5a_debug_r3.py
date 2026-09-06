"""Gate 5A Debug R3 login-detector forensic tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from local_connector.kfh_gate3a.browser import LOGIN_MARKERS
from local_connector.kfh_gate5a import ui_debug
from local_connector.kfh_gate5a.ui_debug import (
    LOGIN_SIGNAL_NAMES,
    Gate5ATempUiDebugger,
    _login_signal_results,
)


class _Locator:
    def __init__(
        self,
        *,
        count: int = 0,
        visible: bool = False,
        box: bool = False,
        ancestor_visible: bool = False,
    ) -> None:
        self._count = count
        self._visible = visible
        self._box = box
        self._ancestor_visible = ancestor_visible
        self.first = self

    async def count(self) -> int:
        return self._count

    async def is_visible(self, timeout: int) -> bool:
        return self._visible

    async def bounding_box(self, timeout: int):
        if not self._box:
            return None
        return {"width": 20, "height": 10}

    def locator(self, selector: str):
        assert selector == "xpath=.."
        return _Locator(count=1, visible=self._ancestor_visible)


class _Page:
    def __init__(self, login_locators: list[_Locator]) -> None:
        self._login_locators = login_locators
        self._text_call = 0
        self.url = "https://trading.kfhtrade.com/app?token=DROP-URL-TOKEN"
        self.frames = [self]
        self.main_frame = self

    def get_by_text(self, pattern, exact: bool):
        assert exact is False
        if self._text_call < len(self._login_locators):
            locator = self._login_locators[self._text_call]
        else:
            locator = _Locator()
        self._text_call += 1
        return locator

    def locator(self, selector: str):
        return _Locator()

    async def evaluate(self, expression: str) -> str:
        assert expression == "document.readyState"
        return "complete"


class _Session:
    def __init__(self, *, login_active: bool, authenticated_count: int) -> None:
        self.login_active = login_active
        self.authenticated_count = authenticated_count

    async def login_ui_active(self) -> bool:
        return self.login_active

    async def authenticated_ui_signal_count(self) -> int:
        return self.authenticated_count


def _auth_success():
    return SimpleNamespace(
        response_seen=True,
        status_success=True,
        sanitized_frame='{"msgGrp":5,"msgTyp":101,"DAT":{"authSts":1}}',
    )


def _records(debugger: Gate5ATempUiDebugger) -> list[dict]:
    return [
        json.loads(line)
        for line in debugger.path.read_text(encoding="utf-8").splitlines()
    ]


def test_r3_symbolic_signals_cover_exact_sealed_login_markers():
    assert tuple(LOGIN_SIGNAL_NAMES) == LOGIN_MARKERS
    assert tuple(LOGIN_SIGNAL_NAMES.values()) == (
        "LOGIN_TEXT_MARKER",
        "SIGN_IN_TEXT_MARKER",
        "USERNAME_TEXT_MARKER",
        "USER_NAME_TEXT_MARKER",
    )


@pytest.mark.asyncio
async def test_r3_presence_visibility_box_and_ancestor_are_separate_booleans():
    page = _Page(
        [
            _Locator(count=3, visible=False, box=False, ancestor_visible=True),
            _Locator(count=1, visible=True, box=True, ancestor_visible=True),
            _Locator(),
            _Locator(),
        ]
    )
    results = await _login_signal_results(page)
    assert results[0] == {
        "signal": "LOGIN_TEXT_MARKER",
        "matched": True,
        "visible": False,
        "matchCount": 3,
        "hasNonzeroBoundingBox": False,
        "ancestorVisible": True,
    }
    assert results[1]["visible"] is True
    assert results[1]["hasNonzeroBoundingBox"] is True


@pytest.mark.asyncio
async def test_r3_sample_records_closed_decision_matrix_without_changing_it(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ui_debug, "SAMPLE_OFFSETS_SECONDS", (0,))
    debugger = Gate5ATempUiDebugger(tmp_path / "matrix.jsonl")
    debugger.observe_inbound_frame("TRADE", "discard", _auth_success())
    debugger.start(
        _Page([_Locator(visible=True, count=1), _Locator(), _Locator(), _Locator()]),
        _Session(login_active=True, authenticated_count=2),
    )
    assert await debugger.wait(timeout_seconds=1)
    sample = next(record for record in _records(debugger) if record["event"] == "UI_SIGNAL_SAMPLE")
    assert sample["closedLoginUiActive"] is True
    assert sample["decisionMatrix"] == {
        "authProtocolSuccess": True,
        "loginUiActive": True,
        "authenticatedUiSignalCount": 2,
        "wouldGate3AAuthenticate": False,
    }


@pytest.mark.asyncio
async def test_r3_visible_marker_classifies_visible_login_ui_after_auth(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ui_debug, "SAMPLE_OFFSETS_SECONDS", (0,))
    debugger = Gate5ATempUiDebugger(tmp_path / "visible.jsonl")
    debugger.observe_inbound_frame("TRADE", "discard", _auth_success())
    debugger.start(
        _Page([_Locator(count=1, visible=True, box=True), _Locator(), _Locator(), _Locator()]),
        _Session(login_active=True, authenticated_count=2),
    )
    assert await debugger.wait(timeout_seconds=1)
    assert debugger.summary("AUTHENTICATING", SimpleNamespace(
        auth_response_seen=True, auth_status_success=True, login_ui_inactive=False
    ))["rootCauseCategory"] == "VISIBLE_LOGIN_UI_REMAINS_AFTER_AUTH"


@pytest.mark.asyncio
async def test_r3_hidden_matches_classify_hidden_login_dom_false_positive(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ui_debug, "SAMPLE_OFFSETS_SECONDS", (0,))
    debugger = Gate5ATempUiDebugger(tmp_path / "hidden.jsonl")
    debugger.observe_inbound_frame("TRADE", "discard", _auth_success())
    debugger.start(
        _Page([_Locator(count=1, visible=False), _Locator(), _Locator(), _Locator()]),
        _Session(login_active=True, authenticated_count=2),
    )
    assert await debugger.wait(timeout_seconds=1)
    final = await debugger.finalize("AUTHENTICATING")
    assert final["rootCauseCategory"] == "HIDDEN_LOGIN_DOM_FALSE_POSITIVE"


@pytest.mark.asyncio
async def test_r3_unmatched_contradiction_is_not_claimed_as_proven(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ui_debug, "SAMPLE_OFFSETS_SECONDS", (0,))
    debugger = Gate5ATempUiDebugger(tmp_path / "insufficient.jsonl")
    debugger.observe_inbound_frame("TRADE", "discard", _auth_success())
    debugger.start(
        _Page([_Locator(), _Locator(), _Locator(), _Locator()]),
        _Session(login_active=True, authenticated_count=2),
    )
    assert await debugger.wait(timeout_seconds=1)
    final = await debugger.finalize("AUTHENTICATING")
    assert final["rootCauseCategory"] == "LOGIN_UI_DETECTOR_CAUSE_NOT_PROVEN"
    assert final["rootCauseCategory"] != "CLOSED_GATE3A_SIGNAL_AVAILABLE"


@pytest.mark.asyncio
async def test_r3_output_retains_no_page_text_attributes_or_secret_values(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ui_debug, "SAMPLE_OFFSETS_SECONDS", (0,))
    debugger = Gate5ATempUiDebugger(tmp_path / "secure.jsonl")
    debugger.observe_inbound_frame(
        "TRADE",
        '{"password":"DROP-PASSWORD","otp":"DROP-OTP"}',
        _auth_success(),
    )
    debugger.start(
        _Page([_Locator(count=1, visible=True), _Locator(), _Locator(), _Locator()]),
        _Session(login_active=True, authenticated_count=2),
    )
    assert await debugger.wait(timeout_seconds=1)
    await debugger.finalize("AUTHENTICATING")
    payload = debugger.path.read_text(encoding="utf-8")
    for forbidden in (
        "DROP-PASSWORD",
        "DROP-OTP",
        "DROP-URL-TOKEN",
        "Sign In",
        "User Name",
        "inputValue",
        "outerHTML",
    ):
        assert forbidden not in payload
