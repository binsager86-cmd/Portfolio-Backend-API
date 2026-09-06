"""Temporary Gate 5A UI diagnostics must remain local, bounded, and sanitized."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from local_connector.kfh_gate5a import ui_debug
from local_connector.kfh_gate5a.browser import Gate5AAuthDiagnostics
from local_connector.kfh_gate5a.ui_debug import Gate5ATempUiDebugger


class FakeLocator:
    def __init__(self, count: int, visible: bool) -> None:
        self._count = count
        self._visible = visible
        self.first = self

    async def count(self) -> int:
        return self._count

    async def is_visible(self, timeout: int) -> bool:
        assert timeout == 200
        return self._visible


class FakeScope:
    def __init__(
        self,
        *,
        visible_text: tuple[str, ...] = (),
        selector_counts: dict[str, tuple[int, bool]] | None = None,
    ) -> None:
        self.visible_text = visible_text
        self.selector_counts = selector_counts or {}

    def get_by_text(self, pattern, exact: bool):
        assert exact is False
        matches = [text for text in self.visible_text if pattern.search(text)]
        return FakeLocator(len(matches), bool(matches))

    def locator(self, selector: str):
        count, visible = self.selector_counts.get(selector, (0, False))
        return FakeLocator(count, visible)


class FakePage(FakeScope):
    def __init__(self, *, child_frame: FakeScope | None = None) -> None:
        super().__init__(
            visible_text=("Cash Statement", "Logout"),
            selector_counts={"nav, [role='navigation']": (1, True)},
        )
        self.url = "https://trading.kfhtrade.com/app?sesnId=SECRET-MUST-DROP"
        self.main_frame = self
        self.frames = [self] + ([child_frame] if child_frame else [])

    async def evaluate(self, expression: str):
        assert expression == "document.readyState"
        return "complete"


class FakeClosedGate3ASession:
    async def login_ui_active(self) -> bool:
        return False

    async def authenticated_ui_signal_count(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_temp_debug_jsonl_is_symbolic_sanitized_and_origin_only(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_debug, "SAMPLE_OFFSETS_SECONDS", (0,))
    path = tmp_path / "gate5a-ui-debug-test.jsonl"
    debugger = Gate5ATempUiDebugger(path)
    diagnostics = Gate5AAuthDiagnostics(
        login_ui_inactive=True,
        auth_response_seen=True,
        auth_status_success=True,
        browser_origin_allowed=True,
    )
    debugger.observe_inbound_frame(
        "TRADE",
        "{}",
        SimpleNamespace(
            response_seen=True,
            status_success=True,
            sanitized_frame='{"msgGrp":5,"msgTyp":101,"DAT":{"authSts":1}}',
        ),
    )
    debugger.start(FakePage(), FakeClosedGate3ASession())
    assert await debugger.wait(timeout_seconds=1) is True

    payload = path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in payload.splitlines()]
    record = next(item for item in records if item["event"] == "UI_SIGNAL_SAMPLE")
    assert record["sampleOffsetSeconds"] == 0
    assert record["closedGate3A"]["signalCount"] == 0
    assert record["pageState"]["currentOrigin"] == "https://trading.kfhtrade.com"
    assert record["pageState"]["numberOfFrames"] == 1
    assert record["pageState"]["authenticatedMarkerFoundInChildFrame"] is False
    assert debugger.summary("AUTHENTICATING", diagnostics)["rootCauseCategory"] == (
        "STALE_OR_INCORRECT_AUTHENTICATED_UI_SELECTORS"
    )
    for forbidden in (
        "SECRET-MUST-DROP",
        "Cash Statement",
        "sesnId",
        "password",
        "otp",
    ):
        assert forbidden.lower() not in payload.lower()

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert {key.lower() for key in keys(record)}.isdisjoint({"hed", "dat"})


@pytest.mark.asyncio
async def test_child_frame_marker_is_reported_without_frame_content(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_debug, "SAMPLE_OFFSETS_SECONDS", (0,))
    child = FakeScope(visible_text=("المحفظة",))
    debugger = Gate5ATempUiDebugger(tmp_path / "child-frame.jsonl")
    debugger.observe_inbound_frame(
        "TRADE",
        "{}",
        SimpleNamespace(
            response_seen=True,
            status_success=True,
            sanitized_frame='{"msgGrp":5,"msgTyp":101,"DAT":{"authSts":1}}',
        ),
    )
    debugger.start(FakePage(child_frame=child), FakeClosedGate3ASession())
    assert await debugger.wait(timeout_seconds=1) is True
    summary = debugger.summary("AUTHENTICATING", Gate5AAuthDiagnostics())
    assert summary["authenticatedMarkerFoundInChildFrame"] is True
    assert summary["rootCauseCategory"] == "AUTHENTICATED_MARKERS_IN_CHILD_FRAME"
    payload = debugger.path.read_text(encoding="utf-8")
    assert "المحفظة" not in payload


def test_default_debug_path_is_under_system_temp_not_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_debug, "gettempdir", lambda: str(tmp_path))
    path = ui_debug.create_temp_debug_path()
    try:
        assert path.parent == tmp_path / "saham-kfh"
        assert path.name.startswith("gate5a-ui-debug-")
        assert path.suffix == ".jsonl"
    finally:
        if path.exists():
            path.unlink()
        if path.parent.exists():
            path.parent.rmdir()
