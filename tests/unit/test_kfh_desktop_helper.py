"""Desktop helper boundary and pairing tests.

The dev/test tool's origin check (kfh_gate5b.local_api) requires an
explicit port in the Origin header - proven by
test_kfh_gate5b_l2_r3_origin.py::test_malformed_origins_are_format_invalid,
which asserts "http://localhost" (no port) is rejected as malformed. Real
browsers omit the port for a site's default port, so that dev-only check
can never accept a real deployed site's Origin header. These tests cover
the desktop helper's corrected origin handling and its per-installation
pairing nonce - both new, both security-relevant, neither exercised by the
dev/test tool's own suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_connector.kfh_desktop_helper.app import (
    HELPER_PATH_PREFIX,
    NONCE_HEADER,
    canonicalize_browser_origin,
    create_desktop_helper_app,
)
from local_connector.kfh_desktop_helper.pairing import (
    MIN_NONCE_LENGTH,
    load_or_create_pairing_nonce,
)

NONCE = "DESKTOP-HELPER-TEST-NONCE-1234567890"
PROD_ORIGIN = "https://portfolioproapp.com"
BODY = {
    "username": "SYNTHETIC-USER",
    "password": "SYNTHETIC-PASSWORD",
    "fromDate": "20251001",
    "toDate": "20260902",
}


class BoundaryOnlyController:
    """A zero-I/O controller proving the request passed only the boundary."""

    def __init__(self) -> None:
        self.connect_calls = 0
        self.close_calls = 0

    async def connect(self, _payload: Any) -> dict[str, Any]:
        self.connect_calls += 1
        return {"connection": "DISCONNECTED", "result": "IDLE", "failure": None}

    async def close(self) -> dict[str, Any]:
        self.close_calls += 1
        return {"result": "IDLE"}


def client_for(controller: BoundaryOnlyController, *, allowed_origin: str = PROD_ORIGIN) -> TestClient:
    app = create_desktop_helper_app(controller=controller, nonce=NONCE, allowed_origin=allowed_origin)
    return TestClient(app, client=("127.0.0.1", 49152))


class TestCanonicalizeBrowserOrigin:
    def test_https_with_no_port_defaults_to_443(self) -> None:
        assert canonicalize_browser_origin("https://portfolioproapp.com") == "https://portfolioproapp.com:443"

    def test_http_with_no_port_defaults_to_80(self) -> None:
        assert canonicalize_browser_origin("http://example.com") == "http://example.com:80"

    def test_explicit_port_is_preserved(self) -> None:
        assert canonicalize_browser_origin("http://localhost:8081") == "http://localhost:8081"

    def test_host_is_lowercased(self) -> None:
        assert canonicalize_browser_origin("https://Portfolioproapp.com") == "https://portfolioproapp.com:443"

    @pytest.mark.parametrize(
        "origin",
        [
            "not-an-origin",
            "https://portfolioproapp.com/path",
            "https://portfolioproapp.com?query=yes",
            "https://portfolioproapp.com#fragment",
            "ftp://portfolioproapp.com",
            "https://user:pass@portfolioproapp.com",
        ],
    )
    def test_malformed_origins_still_rejected(self, origin: str) -> None:
        with pytest.raises(ValueError):
            canonicalize_browser_origin(origin)


def test_real_production_origin_with_no_port_is_accepted() -> None:
    """This is the exact case the dev/test tool's boundary cannot handle:
    a real browser's Origin header for a site on its default HTTPS port."""
    controller = BoundaryOnlyController()
    with client_for(controller) as client:
        response = client.get(
            f"{HELPER_PATH_PREFIX}health",
            headers={"Origin": PROD_ORIGIN, NONCE_HEADER: NONCE},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "OK"


@pytest.mark.parametrize(
    "origin",
    ["https://evil.example", "http://portfolioproapp.com", "https://portfolioproapp.com:9999"],
)
def test_unauthorized_origins_are_rejected(origin: str) -> None:
    controller = BoundaryOnlyController()
    with client_for(controller) as client:
        response = client.get(
            f"{HELPER_PATH_PREFIX}health",
            headers={"Origin": origin, NONCE_HEADER: NONCE},
        )
    assert response.status_code == 403
    assert response.json()["failure"] == "ORIGIN_REJECTED"


def test_missing_origin_is_rejected() -> None:
    controller = BoundaryOnlyController()
    with client_for(controller) as client:
        response = client.get(f"{HELPER_PATH_PREFIX}health", headers={NONCE_HEADER: NONCE})
    assert response.status_code == 403
    assert response.json()["failureReason"] == "MISSING"


def test_missing_or_wrong_nonce_is_rejected() -> None:
    controller = BoundaryOnlyController()
    with client_for(controller) as client:
        missing = client.get(f"{HELPER_PATH_PREFIX}health", headers={"Origin": PROD_ORIGIN})
        wrong = client.get(
            f"{HELPER_PATH_PREFIX}health",
            headers={"Origin": PROD_ORIGIN, NONCE_HEADER: "wrong-nonce-value-1234567890"},
        )
    assert missing.status_code == 403
    assert missing.json()["failure"] == "NONCE_REJECTED"
    assert wrong.status_code == 403
    assert wrong.json()["failure"] == "NONCE_REJECTED"


def test_non_loopback_client_is_rejected_before_anything_else() -> None:
    controller = BoundaryOnlyController()
    app = create_desktop_helper_app(controller=controller, nonce=NONCE, allowed_origin=PROD_ORIGIN)
    with TestClient(app, client=("203.0.113.5", 49152)) as client:
        response = client.get(
            f"{HELPER_PATH_PREFIX}health",
            headers={"Origin": PROD_ORIGIN, NONCE_HEADER: NONCE},
        )
    assert response.status_code == 403
    assert controller.connect_calls == 0


def test_connect_reaches_the_controller_only_once_boundary_passes() -> None:
    controller = BoundaryOnlyController()
    with client_for(controller) as client:
        response = client.post(
            f"{HELPER_PATH_PREFIX}connect-and-fetch",
            headers={"Origin": PROD_ORIGIN, NONCE_HEADER: NONCE},
            json=BODY,
        )
    assert response.status_code == 200
    assert controller.connect_calls == 1


def test_nonce_shorter_than_32_chars_is_refused_at_construction() -> None:
    with pytest.raises(ValueError):
        create_desktop_helper_app(nonce="too-short", allowed_origin=PROD_ORIGIN)


class TestPairingNonce:
    def test_generates_and_persists_a_nonce_on_first_run(self, tmp_path: Path) -> None:
        nonce = load_or_create_pairing_nonce(tmp_path)
        assert len(nonce) >= MIN_NONCE_LENGTH
        assert (tmp_path / "kfh_helper_pairing.txt").exists()

    def test_reuses_the_same_nonce_on_subsequent_runs(self, tmp_path: Path) -> None:
        first = load_or_create_pairing_nonce(tmp_path)
        second = load_or_create_pairing_nonce(tmp_path)
        assert first == second

    def test_replaces_a_corrupted_or_too_short_stored_value(self, tmp_path: Path) -> None:
        pairing_file = tmp_path / "kfh_helper_pairing.txt"
        pairing_file.parent.mkdir(parents=True, exist_ok=True)
        pairing_file.write_text("short", encoding="utf-8")
        nonce = load_or_create_pairing_nonce(tmp_path)
        assert len(nonce) >= MIN_NONCE_LENGTH
        assert nonce != "short"
