# ruff: noqa: E402

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# Ensure backend root is on the path so app.* imports resolve
_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from app.api.v1 import eagle_eye
from app.services.eagle_eye import ingest


@pytest.fixture(autouse=True)
def _reset_recompute_state() -> None:
    eagle_eye._RECOMPUTE_IN_PROGRESS = False
    eagle_eye._RECOMPUTE_LAST_ATTEMPT_AT = 0.0


@pytest.mark.asyncio
async def test_get_scanner_retriggers_warmup_when_cache_is_cold(monkeypatch: pytest.MonkeyPatch) -> None:
    triggered: list[tuple[str, bool]] = []

    monkeypatch.setattr("app.services.eagle_eye.store.load_all_ratings", lambda: [])
    monkeypatch.setattr(
        eagle_eye,
        "_trigger_eagle_eye_recompute",
        lambda reason, force=False: triggered.append((reason, force)) or True,
    )

    response = await eagle_eye.get_scanner(_user=None)

    assert response.status == "warming_up"
    assert response.count == 0
    assert triggered == [("scanner_cache_cold", False)]


def test_trigger_eagle_eye_recompute_respects_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []

    class FakeThread:
        def __init__(self, *, target, daemon: bool, name: str) -> None:
            self._target = target
            self.daemon = daemon
            self.name = name

        def start(self) -> None:
            started.append(self.name)
            self._target()

    monkeypatch.setattr(eagle_eye.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        "app.services.eagle_eye.ingest.run_nightly_recompute",
        lambda dna_refresh=False, verbose=False: {"cache_rows": 12},
    )

    assert eagle_eye._trigger_eagle_eye_recompute("cold_cache_test") is True
    assert eagle_eye._trigger_eagle_eye_recompute("cold_cache_test") is False
    assert started == ["ee_recompute_cold_cache_test"]


def test_ingest_all_ohlcv_fails_fast_without_tickerchart_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compute_logs: list[tuple[str | None, str, str]] = []

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: SimpleNamespace(TICKERCHART_USERNAME="", TICKERCHART_PASSWORD=""),
    )
    monkeypatch.setattr("app.services.eagle_eye.store.ensure_tables", lambda: None)
    monkeypatch.setattr(
        "app.services.eagle_eye.store.log_compute",
        lambda run_type, ticker, status, message="": compute_logs.append((ticker, status, message)),
    )

    result = ingest.ingest_all_ohlcv()

    assert result["errors"] == 1
    assert "credentials" in result["error"].lower()
    assert compute_logs and compute_logs[0][1] == "error"
