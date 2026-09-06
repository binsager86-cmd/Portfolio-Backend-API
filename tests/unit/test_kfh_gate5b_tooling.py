"""Gate 5B evidence-tool safety hardening."""

from __future__ import annotations

import pytest

from local_connector.kfh_gate5a.live_capture import (
    _formal_fixture_allowed,
    _wait_for_explicit_end,
)


class _Runtime:
    async def inspect_pagination_ui(self) -> dict:
        return {}


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["DONE", "END"])
async def test_done_and_end_are_returned_as_distinct_commands(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: command)
    assert await _wait_for_explicit_end(_Runtime()) == command


def test_only_done_can_allow_formal_fixture_evaluation() -> None:
    assert _formal_fixture_allowed(
        "DONE", completed_page_count=4, new_query_detected=False
    )
    assert not _formal_fixture_allowed(
        "END", completed_page_count=4, new_query_detected=False
    )
    assert not _formal_fixture_allowed(
        "DONE", completed_page_count=1, new_query_detected=False
    )
    assert not _formal_fixture_allowed(
        "DONE", completed_page_count=4, new_query_detected=True
    )
