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

    monkeypatch.setattr(
        "app.services.eagle_eye.store.load_all_ratings",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        eagle_eye,
        "_trigger_eagle_eye_recompute",
        lambda reason, force=False, **kwargs: triggered.append((reason, force)) or True,
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


def test_run_nightly_recompute_fails_fast_when_ohlcv_has_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    compute_calls: list[str] = []

    monkeypatch.setattr(ingest, "check_ratings_cache_drop", lambda run_id: None)
    monkeypatch.setattr(
        ingest,
        "ingest_all_ohlcv",
        lambda verbose=False, progress_callback=None: {
            "ok": 0,
            "skipped": 0,
            "errors": 1,
            "insufficient": [],
            "gaps": [],
            "error": "TickerChart credentials are not configured",
        },
    )

    def fake_compute_all_ratings(*args, **kwargs):
        compute_calls.append("called")
        return {"ok": 12, "skipped": 0, "errors": 0}

    monkeypatch.setattr(ingest, "compute_all_ratings", fake_compute_all_ratings)
    monkeypatch.setattr(ingest, "build_all_dna", lambda verbose=False: {"ok": 0, "errors": 0})
    monkeypatch.setattr("app.core.database.query_val", lambda *args, **kwargs: 0)
    monkeypatch.setattr("app.services.eagle_eye.store.log_compute", lambda *args, **kwargs: None)

    result = ingest.run_nightly_recompute(verbose=False)

    assert result["status"] == "failure"
    assert result["ohlcv"]["errors"] == 1
    assert compute_calls == []


def test_run_nightly_recompute_checks_cache_drop_even_when_ingest_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """AUD-004: the row-count guard must run even when OHLCV ingest aborts the run."""
    guard_calls: list[str] = []

    monkeypatch.setattr(ingest, "check_ratings_cache_drop", lambda run_id: guard_calls.append(run_id))
    monkeypatch.setattr(
        ingest,
        "ingest_all_ohlcv",
        lambda verbose=False, progress_callback=None: {
            "ok": 0,
            "skipped": 0,
            "errors": 1,
            "insufficient": [],
            "gaps": [],
            "error": "TickerChart credentials are not configured",
        },
    )
    monkeypatch.setattr("app.core.database.query_val", lambda *args, **kwargs: 0)
    monkeypatch.setattr("app.services.eagle_eye.store.log_compute", lambda *args, **kwargs: None)

    result = ingest.run_nightly_recompute(verbose=False)

    assert result["status"] == "failure"
    assert len(guard_calls) == 1


def test_run_nightly_recompute_fails_when_rating_run_is_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "check_ratings_cache_drop", lambda run_id: None)
    monkeypatch.setattr(
        ingest,
        "ingest_all_ohlcv",
        lambda verbose=False, progress_callback=None: {
            "ok": 10,
            "skipped": 0,
            "errors": 0,
            "insufficient": [],
            "gaps": [],
        },
    )
    monkeypatch.setattr(
        ingest,
        "compute_all_ratings",
        lambda verbose=False, progress_callback=None: {"ok": 8, "expected": 10, "skipped": 0, "errors": 0},
    )
    monkeypatch.setattr(ingest, "build_all_dna", lambda verbose=False: {"ok": 0, "errors": 0})
    monkeypatch.setattr("app.core.database.query_val", lambda *args, **kwargs: 9)
    monkeypatch.setattr("app.services.eagle_eye.store.log_compute", lambda *args, **kwargs: None)

    result = ingest.run_nightly_recompute(verbose=False)

    assert result["status"] == "failure"
    assert result["ratings"]["ok"] == 8
    assert result["ratings"]["expected"] == 10


def test_run_nightly_recompute_ok_with_realistic_healthy_skip_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    """AUD-001 T1: a healthy run with legitimate skips (real historical shape:
    135 ok, 6 skipped, 0 errors, 141 expected) must report status "ok", not
    "failure". Skipped tickers (insufficient history / inactive / indicator
    unavailable) are covered, not missing."""
    monkeypatch.setattr(ingest, "check_ratings_cache_drop", lambda run_id: None)
    monkeypatch.setattr(
        ingest,
        "ingest_all_ohlcv",
        lambda verbose=False, progress_callback=None: {
            "ok": 141,
            "skipped": 0,
            "errors": 0,
            "insufficient": [],
            "gaps": [],
        },
    )
    monkeypatch.setattr(
        ingest,
        "compute_all_ratings",
        lambda verbose=False, progress_callback=None: {"ok": 135, "skipped": 6, "errors": 0, "expected": 141},
    )
    monkeypatch.setattr(ingest, "build_all_dna", lambda verbose=False: {"ok": 0, "errors": 0})
    monkeypatch.setattr("app.core.database.query_val", lambda *args, **kwargs: 141)
    monkeypatch.setattr("app.services.eagle_eye.store.log_compute", lambda *args, **kwargs: None)

    result = ingest.run_nightly_recompute(verbose=False)

    assert result["status"] == "ok"


def test_run_nightly_recompute_fails_when_run_is_interrupted_mid_way(monkeypatch: pytest.MonkeyPatch) -> None:
    """AUD-001 T2: an interrupted run (85 of 141 tickers ever attempted, the
    remainder never reached) must still report status "failure", even though
    none of the 85 attempted tickers themselves errored."""
    monkeypatch.setattr(ingest, "check_ratings_cache_drop", lambda run_id: None)
    monkeypatch.setattr(
        ingest,
        "ingest_all_ohlcv",
        lambda verbose=False, progress_callback=None: {
            "ok": 141,
            "skipped": 0,
            "errors": 0,
            "insufficient": [],
            "gaps": [],
        },
    )
    monkeypatch.setattr(
        ingest,
        "compute_all_ratings",
        lambda verbose=False, progress_callback=None: {"ok": 80, "skipped": 5, "errors": 0, "expected": 141},
    )
    monkeypatch.setattr(ingest, "build_all_dna", lambda verbose=False: {"ok": 0, "errors": 0})
    monkeypatch.setattr("app.core.database.query_val", lambda *args, **kwargs: 85)
    monkeypatch.setattr("app.services.eagle_eye.store.log_compute", lambda *args, **kwargs: None)

    result = ingest.run_nightly_recompute(verbose=False)

    assert result["status"] == "failure"
