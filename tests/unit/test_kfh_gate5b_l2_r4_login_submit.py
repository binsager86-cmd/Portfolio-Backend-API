"""Gate 5B-L2-R4 login-submit selector tests from the sanitized live DOM."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from playwright.async_api import Page, async_playwright

from local_connector.kfh_gate3a.browser import (
    LOGIN_PASSWORD_CONTROL_SELECTOR,
    LOGIN_USERNAME_CONTROL_SELECTOR,
)
from local_connector.kfh_gate5b.browser import (
    KFH_LOGIN_SUBMIT_CONTROL_SELECTOR,
    KFH_OTP_CONTROL_SELECTOR,
    _clear_login_handoff,
    _fixed_login_controls,
    _KfhLoginHandoff,
)

ARABIC_LOGIN_DOM = """
<!doctype html>
<html lang="ar"><body>
  <input id="txtUsername" name="username" type="text" placeholder="اسم المستخدم">
  <input id="txtPassword" name="password" type="password" placeholder="كلمة السر">
  <button id="btnLogin" type="button">تسجيل الدخول</button>
  <input id="otpLoginPin" type="text" hidden>
  <button id="btnOtpLogin" type="button" hidden>تحقق</button>
</body></html>
"""


@asynccontextmanager
async def fixture_page(html: str) -> AsyncIterator[Page]:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html)
        try:
            yield page
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_observed_arabic_dom_resolves_username_password_and_btn_login() -> None:
    async with fixture_page(ARABIC_LOGIN_DOM) as page:
        controls = await _fixed_login_controls(page)
        assert controls is not None
        username, password, submit = controls
        assert await username.get_attribute("id") == "txtUsername"
        assert await password.get_attribute("id") == "txtPassword"
        assert await submit.get_attribute("id") == "btnLogin"
        assert await submit.get_attribute("type") == "button"
        assert await page.locator(KFH_LOGIN_SUBMIT_CONTROL_SELECTOR).count() == 1


@pytest.mark.asyncio
async def test_normal_login_never_resolves_btn_otp_login() -> None:
    async with fixture_page(ARABIC_LOGIN_DOM) as page:
        controls = await _fixed_login_controls(page)
        assert controls is not None
        assert await controls[2].get_attribute("id") == "btnLogin"
        assert await controls[2].get_attribute("id") != "btnOtpLogin"
        assert await page.locator("button#btnOtpLogin").count() == 1
        assert await page.locator(KFH_OTP_CONTROL_SELECTOR).count() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["Login", "Sign In"])
async def test_exact_english_button_fallback_remains_supported(label: str) -> None:
    html = f"""
    <input id="txtUsername" name="username" type="text">
    <input id="txtPassword" name="password" type="password">
    <button type="button">{label}</button>
    """
    async with fixture_page(html) as page:
        controls = await _fixed_login_controls(page)
        assert controls is not None
        assert await controls[2].text_content() == label


@pytest.mark.asyncio
async def test_submit_input_fallback_remains_supported() -> None:
    html = """
    <input id="txtUsername" name="username" type="text">
    <input id="txtPassword" name="password" type="password">
    <input id="legacyLogin" type="submit" value="Login">
    """
    async with fixture_page(html) as page:
        controls = await _fixed_login_controls(page)
        assert controls is not None
        assert await controls[2].get_attribute("id") == "legacyLogin"


@pytest.mark.asyncio
async def test_no_accepted_submit_fails_closed_without_arbitrary_button() -> None:
    html = """
    <input id="txtUsername" name="username" type="text">
    <input id="txtPassword" name="password" type="password">
    <button id="unrelated" type="button">Continue</button>
    <button id="btnOtpLogin" type="button">Login</button>
    """
    async with fixture_page(html) as page:
        assert await _fixed_login_controls(page) is None


@pytest.mark.asyncio
async def test_disabled_btn_login_fails_actionability_validation() -> None:
    html = """
    <input id="txtUsername" name="username" type="text">
    <input id="txtPassword" name="password" type="password">
    <button id="btnLogin" type="button" disabled>تسجيل الدخول</button>
    """
    async with fixture_page(html) as page:
        assert await _fixed_login_controls(page) is None


@pytest.mark.asyncio
async def test_username_and_password_selectors_are_deterministic_ids() -> None:
    """Regression: the prior type-based selectors (input[type='text'], etc.)
    matched unrelated visible inputs elsewhere in the app (e.g. the dashboard's
    search/order-entry fields), causing login_ui_active() false positives after
    a real successful authentication. Deterministic IDs, confirmed via KFH DOM
    forensic evidence, avoid that."""
    assert LOGIN_USERNAME_CONTROL_SELECTOR == "input#txtUsername"
    assert LOGIN_PASSWORD_CONTROL_SELECTOR == "input#txtPassword"
    async with fixture_page(ARABIC_LOGIN_DOM) as page:
        assert await page.locator(LOGIN_USERNAME_CONTROL_SELECTOR).count() == 1
        assert await page.locator(LOGIN_PASSWORD_CONTROL_SELECTOR).count() == 1


@pytest.mark.asyncio
async def test_stale_login_controls_never_block_cleanup() -> None:
    """Regression: KFH navigates away from the login view after a real
    authentication, so #txtUsername/#txtPassword are commonly already
    detached by the time cleanup runs. Cleanup must resolve in around a
    second, not Playwright's much longer default actionability timeout,
    so the statement reader is never delayed behind best-effort cleanup."""
    async with fixture_page(ARABIC_LOGIN_DOM) as page:
        controls = await _fixed_login_controls(page)
        assert controls is not None
        username_control, password_control, _submit_control = controls
        handoff = _KfhLoginHandoff(
            username_control=username_control,
            password_control=password_control,
            username_autofill_confirmed=True,
            password_autofill_confirmed=True,
        )

        # Simulate KFH navigating away from the login view: the controls
        # are removed from the DOM entirely (detached, never becomes
        # visible/actionable again).
        await page.evaluate(
            "() => { document.getElementById('txtUsername').remove(); "
            "document.getElementById('txtPassword').remove(); }"
        )

        started = time.perf_counter()
        await _clear_login_handoff(handoff)
        elapsed = time.perf_counter() - started

        assert elapsed < 5.0
