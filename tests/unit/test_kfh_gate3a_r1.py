"""Gate 3A-R1 regression tests for the login-UI false-positive repair."""

from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

import pytest
from playwright.async_api import Page, async_playwright

from local_connector.kfh_gate3a.browser import (
    LOGIN_PASSWORD_CONTROL_SELECTOR,
    LOGIN_USERNAME_CONTROL_SELECTOR,
    PlaywrightBrowserSession,
)
from local_connector.kfh_gate3a.connector import KfhGate3AConnector
from local_connector.kfh_gate3a.state import KfhAuthState


class _Item:
    def __init__(self, visible: bool) -> None:
        self.visible = visible

    async def is_visible(self, timeout: int) -> bool:
        assert timeout == 200
        return self.visible

    async def click(self, timeout: int) -> None:
        return None


class _Locator:
    def __init__(self, visibilities: list[bool]) -> None:
        self.visibilities = visibilities
        self.first = _Item(visibilities[0] if visibilities else False)

    async def count(self) -> int:
        return len(self.visibilities)

    def nth(self, index: int) -> _Item:
        return _Item(self.visibilities[index])


class _Response:
    status = 200


class _Page:
    def __init__(
        self,
        *,
        visible_text: set[str] | None = None,
        username_control: bool = False,
        password_control: bool = False,
    ) -> None:
        self.visible_text = visible_text or set()
        self.username_control = username_control
        self.password_control = password_control
        self.closed = False

    async def goto(self, url: str, *, wait_until: str, timeout: int):
        return _Response()

    def get_by_text(self, pattern, exact: bool):
        matches = [text for text in self.visible_text if pattern.search(text)]
        return _Locator([True for _match in matches])

    def locator(self, selector: str):
        if selector == LOGIN_USERNAME_CONTROL_SELECTOR:
            return _Locator([self.username_control] if self.username_control else [])
        if selector == LOGIN_PASSWORD_CONTROL_SELECTOR:
            return _Locator([self.password_control] if self.password_control else [])
        return _Locator([])

    def is_closed(self) -> bool:
        return self.closed


class _Closable:
    async def close(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class _Runtime:
    def __init__(self, page: _Page) -> None:
        self.page = page
        self.on_inbound_frame = None
        self.on_closed = None
        self.on_document_failure = None

    async def open(self, **callbacks):
        self.on_inbound_frame = callbacks["on_inbound_frame"]
        self.on_closed = callbacks["on_closed"]
        self.on_document_failure = callbacks["on_document_failure"]
        return PlaywrightBrowserSession(
            _Closable(), _Closable(), _Closable(), self.page
        )

    def emit_auth_success(self) -> None:
        self.on_inbound_frame(
            json.dumps(
                {"msgGrp": 5, "msgTyp": 101, "DAT": {"authSts": 1}}
            )
        )


async def _wait_for_state(
    connector: KfhGate3AConnector, expected: KfhAuthState
) -> None:
    for _ in range(100):
        if connector.status().state == expected:
            return
        await asyncio.sleep(0.002)
    assert connector.status().state == expected


def _session(page: _Page) -> PlaywrightBrowserSession:
    return PlaywrightBrowserSession(_Closable(), _Closable(), _Closable(), page)


@pytest.mark.asyncio
async def test_r1_real_r3_regression_login_label_does_not_block_ready():
    page = _Page(visible_text={"Login"})
    assert await _session(page).login_ui_active() is False
    runtime = _Runtime(page)
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    await connector.connect()
    page.visible_text.update({"Statements", "Buying Power"})
    runtime.emit_auth_success()
    await _wait_for_state(connector, KfhAuthState.READY)
    await connector.close()


@pytest.mark.asyncio
async def test_r1_actual_login_controls_establish_active_login_ui():
    page = _Page(
        visible_text={"Login"}, username_control=True, password_control=True
    )
    assert await _session(page).login_ui_active() is True


@pytest.mark.asyncio
async def test_r1_partial_login_page_never_reaches_ready():
    page = _Page(
        visible_text={"Login"}, username_control=True, password_control=True
    )
    connector = KfhGate3AConnector(_Runtime(page), poll_interval=0.001)
    assert (await connector.connect()).state == KfhAuthState.LOGIN_REQUIRED
    await asyncio.sleep(0.02)
    assert connector.status().state == KfhAuthState.LOGIN_REQUIRED
    await connector.close()


@pytest.mark.asyncio
async def test_r1_generic_authenticated_page_login_label_is_insufficient():
    page = _Page(visible_text={"Login", "Statements", "Buying Power"})
    assert await _session(page).login_ui_active() is False


@pytest.mark.asyncio
async def test_r1_single_authenticated_marker_without_protocol_stays_authenticating():
    page = _Page(visible_text={"Statements"})
    connector = KfhGate3AConnector(_Runtime(page), poll_interval=0.001)
    await connector.connect()
    await asyncio.sleep(0.02)
    assert connector.status().state == KfhAuthState.AUTHENTICATING
    await connector.close()


@pytest.mark.asyncio
async def test_r1_otp_flow_is_unchanged():
    page = _Page(visible_text={"One Time Password"})
    connector = KfhGate3AConnector(_Runtime(page), poll_interval=0.001)
    await connector.connect()
    await _wait_for_state(connector, KfhAuthState.OTP_REQUIRED)
    await connector.close()


@pytest.mark.asyncio
async def test_r1_auth_failed_flow_is_unchanged():
    page = _Page(
        visible_text={"Login", "Invalid password"},
        username_control=True,
        password_control=True,
    )
    connector = KfhGate3AConnector(_Runtime(page), poll_interval=0.001)
    await connector.connect()
    await _wait_for_state(connector, KfhAuthState.AUTH_FAILED)
    await connector.close()


@pytest.mark.asyncio
async def test_r1_session_expired_detection_remains_fail_safe():
    page = _Page()
    runtime = _Runtime(page)
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    await connector.connect()
    page.visible_text = {"Statements"}
    runtime.emit_auth_success()
    await _wait_for_state(connector, KfhAuthState.READY)
    page.visible_text = {"Login"}
    page.username_control = True
    page.password_control = True
    await _wait_for_state(connector, KfhAuthState.SESSION_EXPIRED)
    await connector.close()


@pytest.mark.asyncio
async def test_r1_browser_close_and_network_signals_are_unchanged():
    runtime = _Runtime(_Page())
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    await connector.connect()
    runtime.on_closed()
    await _wait_for_state(connector, KfhAuthState.BROWSER_CLOSED)
    await connector.close()


@asynccontextmanager
async def _fixture_page(html: str) -> AsyncIterator[Page]:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html)
        try:
            yield page
        finally:
            await browser.close()


AUTHENTICATED_DASHBOARD_DOM = """
<!doctype html>
<html lang="ar"><body>
  <input type="text" placeholder="بحث" />
  <input type="text" placeholder="رمز الشركة" />
  <input type="text" placeholder="أفضل السعر" />
  <div>BKA 8,847.93</div>
</body></html>
"""


@pytest.mark.asyncio
async def test_r2_dashboard_search_and_order_fields_do_not_trigger_login_ui_active():
    """Regression: the real authenticated KFH dashboard has several visible
    input[type='text'] fields (search, order entry) unrelated to login. The
    prior type-based selector matched them, so login_ui_active() incorrectly
    reported True right after a real successful authentication, which the
    monitor loop then misread as a session-expired regression back to the
    login page. Only the deterministic #txtUsername/#txtPassword IDs from the
    live login page should count."""
    async with _fixture_page(AUTHENTICATED_DASHBOARD_DOM) as page:
        session = PlaywrightBrowserSession(None, None, None, page)
        assert await session.login_ui_active() is False

    runtime = _Runtime(_Page())
    connector = KfhGate3AConnector(runtime, poll_interval=0.001)
    await connector.connect()
    runtime.on_document_failure()
    await _wait_for_state(connector, KfhAuthState.NETWORK_ERROR)
    await connector.close()


def test_r1_detector_never_reads_or_listens_to_credential_values():
    source = inspect.getsource(PlaywrightBrowserSession)
    for forbidden in (
        "input_value",
        ".value",
        "keydown",
        "add_event_listener",
        "serialize",
        "storage_state",
    ):
        assert forbidden not in source
