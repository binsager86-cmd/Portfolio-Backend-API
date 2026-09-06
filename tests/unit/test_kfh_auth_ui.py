"""Fixed-origin one-shot KFH login form interaction tests."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from local_connector.kfh_gate3a.browser import (
    LOGIN_PASSWORD_CONTROL_SELECTOR,
    LOGIN_USERNAME_CONTROL_SELECTOR,
)
from local_connector.kfh_gate5b.browser import (
    KFH_LOGIN_SUBMIT_CONTROL_SELECTOR,
    KFH_OTP_CONTROL_SELECTOR,
    Gate5BLiveBrowserRuntime,
    KfhLoginAutofillError,
    _clear_login_handoff,
    _clear_otp_handoff,
    _submit_fixed_kfh_login,
    _submit_fixed_kfh_otp,
)


class Control:
    def __init__(
        self,
        *,
        visible: bool = True,
        click_fails: bool = False,
        read_override: str | None = None,
        on_click: Callable[[], Awaitable[None]] | None = None,
        element_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.visible = visible
        self.click_fails = click_fails
        self.read_override = read_override
        self.on_click = on_click
        self.element_id = element_id
        self.enabled = enabled
        self.fills: list[str] = []
        self.value = ""
        self.clicks = 0
        self.click_task: asyncio.Task[None] | None = None

    async def is_visible(self, timeout: int) -> bool:
        assert timeout == 200
        return self.visible

    async def fill(self, value: str, timeout: int | None = None) -> None:
        self.fills.append(value)
        self.value = value

    async def input_value(self) -> str:
        return self.read_override if self.read_override is not None else self.value

    async def get_attribute(self, name: str) -> str | None:
        assert name == "id"
        return self.element_id

    async def is_enabled(self, timeout: int) -> bool:
        assert timeout == 200
        return self.enabled

    async def click(self, timeout: int, trial: bool = False) -> None:
        if trial:
            assert timeout == 5_000
            if not self.enabled:
                raise RuntimeError("control is disabled")
            return
        assert timeout == 5_000
        self.clicks += 1
        if self.click_fails:
            raise RuntimeError("raw browser failure")
        if self.on_click is not None:
            self.click_task = asyncio.create_task(self.on_click())


class Locator:
    def __init__(self, controls: list[Control]) -> None:
        self.controls = controls

    async def count(self) -> int:
        return len(self.controls)

    def nth(self, index: int) -> Control:
        return self.controls[index]


class Page:
    def __init__(
        self,
        *,
        url: str = "https://trading.kfhtrade.com/",
        username: list[Control] | None = None,
        password: list[Control] | None = None,
        submit: list[Control] | None = None,
        otp: list[Control] | None = None,
        otp_submit: list[Control] | None = None,
        ready_after_checks: int = 0,
    ) -> None:
        self.url = url
        self.username = username if username is not None else [Control()]
        self.password = password if password is not None else [Control()]
        self.submit = submit if submit is not None else [Control()]
        self.otp = otp if otp is not None else [Control()]
        self.otp_submit = otp_submit if otp_submit is not None else [Control()]
        self.locator_calls: list[str] = []
        self.role_calls: list[tuple[str, bool]] = []
        self.ready_after_checks = ready_after_checks
        self.username_checks = 0

    def locator(self, selector: str) -> Locator:
        self.locator_calls.append(selector)
        if selector == LOGIN_USERNAME_CONTROL_SELECTOR:
            self.username_checks += 1
            if self.username_checks <= self.ready_after_checks:
                return Locator([])
            return Locator(self.username)
        if selector == LOGIN_PASSWORD_CONTROL_SELECTOR:
            return Locator(self.password)
        if selector == KFH_OTP_CONTROL_SELECTOR:
            return Locator(self.otp)
        if selector == KFH_LOGIN_SUBMIT_CONTROL_SELECTOR:
            return Locator([])
        if selector == "input[type='submit']":
            return Locator([])
        raise AssertionError("arbitrary selector attempted")

    def get_by_role(self, role: str, *, name: Any, exact: bool) -> Locator:
        pattern = getattr(name, "pattern", "")
        assert pattern in {r"^(login|sign in)$", r"^(verify|continue|submit)$"}
        self.role_calls.append((role, exact))
        return Locator(self.submit if pattern == r"^(login|sign in)$" else self.otp_submit)

    def is_closed(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_fixed_login_fills_only_username_password_and_submit() -> None:
    page = Page()
    handoff = await _submit_fixed_kfh_login(
        page, "SYNTHETIC-USER", "SYNTHETIC-PASSWORD"
    )
    assert handoff.public_confirmation() == {
        "usernameAutofillConfirmed": True,
        "passwordAutofillConfirmed": True,
        "loginSubmitTriggered": True,
    }
    assert page.username[0].fills == ["SYNTHETIC-USER"]
    assert page.password[0].fills == ["SYNTHETIC-PASSWORD"]
    assert page.submit[0].clicks == 1
    assert page.locator_calls == [
        LOGIN_USERNAME_CONTROL_SELECTOR,
        LOGIN_PASSWORD_CONTROL_SELECTOR,
        KFH_LOGIN_SUBMIT_CONTROL_SELECTOR,
    ]
    assert page.role_calls == [("button", True)]
    await _clear_login_handoff(handoff)
    assert page.username[0].value == ""
    assert page.password[0].value == ""


@pytest.mark.asyncio
async def test_async_kfh_login_handler_consumes_values_before_dom_clear() -> None:
    username = Control()
    password = Control()
    consumed: list[str] = []

    async def consume_after_click() -> None:
        await asyncio.sleep(0.01)
        consumed.extend([username.value, password.value])

    submit = Control(on_click=consume_after_click)
    page = Page(username=[username], password=[password], submit=[submit])
    handoff = await _submit_fixed_kfh_login(
        page, "SYNTHETIC-USER", "SYNTHETIC-PASSWORD"
    )

    assert username.value == "SYNTHETIC-USER"
    assert password.value == "SYNTHETIC-PASSWORD"
    assert submit.click_task is not None
    await submit.click_task
    assert consumed == ["SYNTHETIC-USER", "SYNTHETIC-PASSWORD"]

    await _clear_login_handoff(handoff)
    assert username.value == ""
    assert password.value == ""


@pytest.mark.asyncio
async def test_login_waits_for_delayed_fixed_controls_then_submits_once() -> None:
    page = Page(ready_after_checks=2)
    handoff = await _submit_fixed_kfh_login(
        page,
        "SYNTHETIC-USER",
        "SYNTHETIC-PASSWORD",
        timeout_seconds=0.1,
        poll_interval=0,
    )
    assert page.username_checks == 3
    assert page.submit[0].clicks == 1
    await _clear_login_handoff(handoff)


@pytest.mark.asyncio
async def test_wrong_origin_rejects_before_any_field_interaction() -> None:
    page = Page(url="https://example.invalid/login")
    with pytest.raises(KfhLoginAutofillError, match="KFH_LOGIN_ORIGIN_REJECTED"):
        await _submit_fixed_kfh_login(page, "SYNTHETIC-USER", "SYNTHETIC-PASSWORD")
    assert page.locator_calls == []


@pytest.mark.asyncio
async def test_missing_or_ambiguous_login_fields_fail_closed() -> None:
    for page in (Page(username=[]), Page(password=[]), Page(username=[Control(), Control()])):
        with pytest.raises(KfhLoginAutofillError, match="KFH_LOGIN_FIELDS_NOT_FOUND"):
            await _submit_fixed_kfh_login(
                page,
                "SYNTHETIC-USER",
                "SYNTHETIC-PASSWORD",
                timeout_seconds=0,
            )


@pytest.mark.asyncio
async def test_autofill_mismatch_prevents_submit_and_clears_controls() -> None:
    page = Page(password=[Control(read_override="MISMATCH")])
    with pytest.raises(KfhLoginAutofillError, match="KFH_LOGIN_AUTOFILL_FAILED"):
        await _submit_fixed_kfh_login(page, "SYNTHETIC-USER", "SYNTHETIC-PASSWORD")
    assert page.submit[0].clicks == 0
    assert page.username[0].value == ""
    assert page.password[0].value == ""


@pytest.mark.asyncio
async def test_submit_failure_clears_dom_controls_and_returns_only_safe_code() -> None:
    page = Page(submit=[Control(click_fails=True)])
    with pytest.raises(KfhLoginAutofillError) as captured:
        await _submit_fixed_kfh_login(page, "SYNTHETIC-USER", "SYNTHETIC-PASSWORD")
    assert str(captured.value) == "KFH_LOGIN_SUBMIT_FAILED"
    assert page.username[0].fills[-1] == ""
    assert page.password[0].fills[-1] == ""
    assert "SYNTHETIC" not in str(captured.value)


@pytest.mark.asyncio
async def test_runtime_rejects_duplicate_login_submit_attempt() -> None:
    page = Page()
    runtime = Gate5BLiveBrowserRuntime(on_statement_response_frame=lambda _frame: None)
    runtime._Gate5BLiveBrowserRuntime__page = page
    first = await runtime._submit_login_credentials(
        "SYNTHETIC-USER", "SYNTHETIC-PASSWORD"
    )
    assert first["loginSubmitTriggered"] is True
    with pytest.raises(KfhLoginAutofillError, match="KFH_LOGIN_SUBMIT_FAILED"):
        await runtime._submit_login_credentials(
            "SYNTHETIC-USER", "SYNTHETIC-PASSWORD"
        )
    assert page.submit[0].clicks == 1
    await runtime._clear_login_dom_credentials()


@pytest.mark.asyncio
async def test_fixed_otp_is_transient_and_consumed_before_dom_clear() -> None:
    otp = Control()
    consumed: list[str] = []

    async def consume_after_click() -> None:
        await asyncio.sleep(0.01)
        consumed.append(otp.value)

    page = Page(otp=[otp], otp_submit=[Control(on_click=consume_after_click)])
    handoff = await _submit_fixed_kfh_otp(page, "123456")
    assert otp.value == "123456"
    assert page.otp_submit[0].clicks == 1
    assert page.otp_submit[0].click_task is not None
    await page.otp_submit[0].click_task
    assert consumed == ["123456"]
    await _clear_otp_handoff(handoff)
    assert otp.value == ""


def test_auth_ui_exposes_no_arbitrary_browser_selector_or_protocol_surface() -> None:
    public = {
        name
        for name, member in inspect.getmembers(Gate5BLiveBrowserRuntime, inspect.isfunction)
        if not name.startswith("_")
    }
    assert public == {"open", "send_cash_statement"}
    source = Path(inspect.getfile(Gate5BLiveBrowserRuntime)).read_text(encoding="utf-8")
    for forbidden in (
        "sendRawMessage",
        "querySelector(rawSelector)",
        "getPage",
        "getBrowser",
        'msgTyp: 1',
        '"msgTyp": 1',
        "placeOrder",
        "cancelOrder",
    ):
        assert forbidden not in source
