"""Headful, ephemeral Playwright runtime with no normal-profile access."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

from .policy import KFH_START_URL, KfhApprovedAction, is_allowed_kfh_url, require_approved_action

FrameCallback = Callable[[str | bytes], None]
SignalCallback = Callable[[], None]

LOGIN_MARKERS = ("Login", "Sign In", "Username", "User Name")
LOGIN_SUPPORTING_MARKERS = LOGIN_MARKERS[1:]
LOGIN_USERNAME_CONTROL_SELECTOR = "input#txtUsername"
LOGIN_PASSWORD_CONTROL_SELECTOR = "input#txtPassword"
OTP_MARKERS = ("One Time Password", "OTP", "Verification Code")
AUTH_FAILED_MARKERS = ("Invalid username", "Invalid password", "Authentication failed", "Login failed")
AUTHENTICATED_MARKERS = ("Statements", "Portfolio", "Account Summary", "Buying Power")
BROWSER_LAUNCH_OPTIONS = {
    "headless": False,
    "args": [
        "--disable-extensions",
        "--disable-sync",
        "--disable-background-networking",
        "--disable-features=AutofillServerCommunication,PasswordManagerOnboarding",
    ],
}
BROWSER_CONTEXT_OPTIONS = {
    "accept_downloads": False,
    "service_workers": "block",
}


class BrowserSession(Protocol):
    async def goto_kfh(self) -> int | None: ...
    async def login_ui_active(self) -> bool: ...
    async def otp_ui_active(self) -> bool: ...
    async def auth_failed_ui_active(self) -> bool: ...
    async def authenticated_ui_signal_count(self) -> int: ...
    async def logout(self) -> None: ...
    async def close(self) -> None: ...
    def is_closed(self) -> bool: ...


class BrowserRuntime(Protocol):
    async def open(
        self,
        *,
        on_inbound_frame: FrameCallback,
        on_closed: SignalCallback,
        on_document_failure: SignalCallback,
    ) -> BrowserSession: ...


class PlaywrightBrowserSession:
    def __init__(self, playwright: Any, browser: Any, context: Any, page: Any) -> None:
        self.__playwright = playwright
        self.__browser = browser
        self.__context = context
        self.__page = page

    async def goto_kfh(self) -> int | None:
        require_approved_action(KfhApprovedAction.LOGIN)
        response = await self.__page.goto(KFH_START_URL, wait_until="domcontentloaded", timeout=45_000)
        return response.status if response else None

    async def __visible(self, text: str) -> bool:
        try:
            locator = self.__page.get_by_text(re.compile(re.escape(text), re.IGNORECASE), exact=False)
            return await locator.first.is_visible(timeout=200)
        except Exception:
            return False

    async def __any_visible(self, markers: tuple[str, ...]) -> bool:
        for marker in markers:
            if await self.__visible(marker):
                return True
        return False

    async def __any_selector_visible(self, selector: str) -> bool:
        """Inspect control presence/visibility only; never read input values."""
        try:
            locator = self.__page.locator(selector)
            for index in range(await locator.count()):
                if await locator.nth(index).is_visible(timeout=200):
                    return True
        except Exception:
            return False
        return False

    async def login_ui_active(self) -> bool:
        password_control_visible = await self.__any_selector_visible(
            LOGIN_PASSWORD_CONTROL_SELECTOR
        )
        if not password_control_visible:
            return False
        username_control_visible = await self.__any_selector_visible(
            LOGIN_USERNAME_CONTROL_SELECTOR
        )
        supporting_marker_visible = await self.__any_visible(LOGIN_SUPPORTING_MARKERS)
        return username_control_visible or supporting_marker_visible

    async def otp_ui_active(self) -> bool:
        return await self.__any_visible(OTP_MARKERS)

    async def auth_failed_ui_active(self) -> bool:
        return await self.__any_visible(AUTH_FAILED_MARKERS)

    async def authenticated_ui_signal_count(self) -> int:
        count = 0
        for marker in AUTHENTICATED_MARKERS:
            count += int(await self.__visible(marker))
        return count

    async def logout(self) -> None:
        require_approved_action(KfhApprovedAction.LOGOUT)
        logout = self.__page.get_by_text(re.compile(r"^(logout|sign out)$", re.IGNORECASE)).first
        await logout.click(timeout=5_000)

    async def close(self) -> None:
        await self.__context.close()
        await self.__browser.close()
        await self.__playwright.stop()

    def is_closed(self) -> bool:
        return self.__page.is_closed()


class PlaywrightBrowserRuntime:
    """Creates a new temporary Chromium process and incognito context per connection."""

    async def open(
        self,
        *,
        on_inbound_frame: FrameCallback,
        on_closed: SignalCallback,
        on_document_failure: SignalCallback,
    ) -> BrowserSession:
        from playwright.async_api import async_playwright

        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(**BROWSER_LAUNCH_OPTIONS)
        context = await browser.new_context(**BROWSER_CONTEXT_OPTIONS)

        async def route_request(route: Any) -> None:
            if is_allowed_kfh_url(route.request.url):
                await route.continue_()
            else:
                await route.abort("blockedbyclient")

        await context.route("**/*", route_request)
        page = await context.new_page()
        page.on("close", lambda: on_closed())

        def websocket_opened(websocket: Any) -> None:
            if not is_allowed_kfh_url(websocket.url.replace("wss://", "https://", 1)):
                return
            websocket.on("framereceived", on_inbound_frame)

        page.on("websocket", websocket_opened)
        page.on(
            "requestfailed",
            lambda request: on_document_failure() if request.resource_type == "document" else None,
        )
        return PlaywrightBrowserSession(playwright, browser, context, page)
