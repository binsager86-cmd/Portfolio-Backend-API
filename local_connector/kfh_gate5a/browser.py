"""Authentication-transparent Gate 5A browser wrapper around closed Gate 3A."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from local_connector.kfh_gate3a.browser import (
    BROWSER_CONTEXT_OPTIONS,
    BROWSER_LAUNCH_OPTIONS,
    BrowserSession,
    FrameCallback,
    PlaywrightBrowserSession,
    SignalCallback,
)
from local_connector.kfh_gate3a.policy import is_allowed_kfh_url

from .ui_debug import Gate5ATempUiDebugger

PAGINATION_UI_DIAGNOSTIC_SCRIPT = r"""
() => {
  const visible = (element) => {
    const style = window.getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== "none" && style.visibility !== "hidden" &&
      Number(style.opacity || "1") !== 0 && rect.width > 0 && rect.height > 0;
  };
  const disabled = (element) => Boolean(
    element.disabled || element.getAttribute("aria-disabled") === "true" ||
    element.classList.contains("disabled")
  );
  const inScrollable = (element) => {
    for (let current = element.parentElement; current; current = current.parentElement) {
      const style = window.getComputedStyle(current);
      if (/(auto|scroll)/.test(style.overflowY) && current.scrollHeight > current.clientHeight) {
        return true;
      }
    }
    return false;
  };
  const summarize = (elements) => ({
    matched: elements.length > 0,
    visible: elements.some(visible),
    count: elements.length,
    disabled: elements.length > 0 && elements.every(disabled),
    belowCurrentViewport: elements.some((element) => element.getBoundingClientRect().top >= window.innerHeight),
    insideScrollableContainer: elements.some(inScrollable),
  });
  const controls = Array.from(document.querySelectorAll(
    "button,a,[role='button'],input[type='button'],input[type='submit']"
  ));
  const label = (element) => String(
    element.getAttribute("aria-label") || element.getAttribute("title") ||
    element.value || element.textContent || ""
  ).trim();
  const nextText = controls.filter((element) => /^next$/i.test(label(element)));
  const previousText = controls.filter((element) => /^previous$/i.test(label(element)));
  const nextCandidates = controls.filter((element) =>
    /next/i.test(label(element)) || /(^|[-_\s])next($|[-_\s])/i.test(String(element.className || ""))
  );
  const forward = controls.filter((element) => /^(>|›)$/.test(label(element)));
  const doubleForward = controls.filter((element) => /^(>>|»)$/.test(label(element)));
  const paginationContainers = Array.from(document.querySelectorAll(
    "[class*='pagination' i],[class*='pager' i],[role='navigation'][aria-label*='pag' i]"
  ));
  const pageNumbers = paginationContainers.flatMap((container) =>
    Array.from(container.querySelectorAll("button,a,[role='button']"))
      .filter((element) => /^\d+$/.test(label(element)))
  );
  const tableFooterControls = Array.from(document.querySelectorAll(
    "tfoot button,tfoot a,[class*='table-footer' i] button,[class*='table-footer' i] a"
  ));
  const statementScrollContainerPresent = Array.from(document.querySelectorAll("table"))
    .some((table) => {
      for (let current = table.parentElement; current; current = current.parentElement) {
        const style = window.getComputedStyle(current);
        if (/(auto|scroll)/.test(style.overflowY) && current.scrollHeight > current.clientHeight) {
          return true;
        }
      }
      return false;
    });
  return {
    NEXT_TEXT: summarize(nextText),
    NEXT_BUTTON_CANDIDATE: summarize(nextCandidates),
    PREVIOUS_TEXT: summarize(previousText),
    FORWARD_CHEVRON: summarize(forward),
    DOUBLE_FORWARD_CHEVRON: summarize(doubleForward),
    PAGINATION_CONTAINER: summarize(paginationContainers),
    PAGE_NUMBER_CONTROLS: summarize(pageNumbers),
    TABLE_FOOTER_CONTROLS: summarize(tableFooterControls),
    STATEMENT_SCROLL_CONTAINER: {present: statementScrollContainerPresent},
  };
}
"""

UI_CANDIDATE_NAMES = (
    "NEXT_TEXT",
    "NEXT_BUTTON_CANDIDATE",
    "PREVIOUS_TEXT",
    "FORWARD_CHEVRON",
    "DOUBLE_FORWARD_CHEVRON",
    "PAGINATION_CONTAINER",
    "PAGE_NUMBER_CONTROLS",
    "TABLE_FOOTER_CONTROLS",
)
UI_CANDIDATE_FIELDS = frozenset(
    {
        "matched",
        "visible",
        "count",
        "disabled",
        "belowCurrentViewport",
        "insideScrollableContainer",
    }
)


def sanitize_pagination_ui_diagnostic(value: Any) -> dict[str, Any]:
    """Accept only the fixed R2 boolean/count diagnostic schema."""
    if not isinstance(value, dict) or set(value) != {
        *UI_CANDIDATE_NAMES,
        "STATEMENT_SCROLL_CONTAINER",
    }:
        raise ValueError("Invalid pagination UI diagnostic fields")
    result: dict[str, Any] = {}
    for name in UI_CANDIDATE_NAMES:
        candidate = value.get(name)
        if not isinstance(candidate, dict) or set(candidate) != UI_CANDIDATE_FIELDS:
            raise ValueError("Invalid pagination UI candidate fields")
        count = candidate.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Invalid pagination UI candidate count")
        if any(
            not isinstance(candidate.get(field), bool)
            for field in UI_CANDIDATE_FIELDS - {"count"}
        ):
            raise ValueError("Invalid pagination UI candidate boolean")
        result[name] = dict(candidate)
    scroll = value.get("STATEMENT_SCROLL_CONTAINER")
    if not isinstance(scroll, dict) or set(scroll) != {"present"} or not isinstance(
        scroll.get("present"), bool
    ):
        raise ValueError("Invalid statement scroll-container diagnostic")
    result["STATEMENT_SCROLL_CONTAINER"] = dict(scroll)
    return result


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _decoded_frame(frame: str | bytes) -> Any | None:
    if not isinstance(frame, str) or len(frame) > 5_000_000:
        return None
    try:
        return json.loads(frame)
    except (json.JSONDecodeError, TypeError):
        return None


def _find_auth_status(value: Any) -> int | None:
    if isinstance(value, dict):
        if "authSts" in value:
            status = _integer(value.get("authSts"))
            if status is not None:
                return status
        for child in value.values():
            status = _find_auth_status(child)
            if status is not None:
                return status
    elif isinstance(value, list):
        for child in value:
            status = _find_auth_status(child)
            if status is not None:
                return status
    return None


def _identity(candidate: dict[str, Any]) -> tuple[int | None, int | None]:
    hed = candidate.get("HED")
    if isinstance(hed, dict):
        return _integer(hed.get("msgGrp")), _integer(hed.get("msgTyp"))
    return _integer(candidate.get("msgGrp")), _integer(candidate.get("msgTyp"))


def _contains_message(frame: str | bytes, group: int, message_type: int) -> bool:
    decoded = _decoded_frame(frame)
    if decoded is None:
        return False
    return any(_identity(candidate) == (group, message_type) for candidate in _walk(decoded))


@dataclass(frozen=True, slots=True)
class Gate3ACompatibleAuthObservation:
    response_seen: bool
    status_success: bool
    sanitized_frame: str | None


def reduce_gate3a_compatible_auth_observation(
    frame: str | bytes,
) -> Gate3ACompatibleAuthObservation:
    """Return only Gate 3A's exact top-level 5/101 shape; discard all else."""
    decoded = _decoded_frame(frame)
    if decoded is None:
        return Gate3ACompatibleAuthObservation(False, False, None)
    for candidate in _walk(decoded):
        if _identity(candidate) != (5, 101):
            continue
        status = _find_auth_status(candidate)
        if status is None:
            return Gate3ACompatibleAuthObservation(True, False, None)
        sanitized = json.dumps(
            {"msgGrp": 5, "msgTyp": 101, "DAT": {"authSts": status}},
            separators=(",", ":"),
        )
        return Gate3ACompatibleAuthObservation(True, status == 1, sanitized)
    return Gate3ACompatibleAuthObservation(False, False, None)


@dataclass(slots=True)
class Gate5AAuthDiagnostics:
    login_ui_inactive: bool | None = None
    authenticated_ui_signal_count: int = 0
    auth_response_seen: bool = False
    auth_status_extracted: bool = False
    auth_status_success: bool = False
    browser_origin_allowed: bool = False

    def public_dict(self, final_state: str) -> dict[str, bool | int | str | None]:
        return {
            "loginUiInactive": self.login_ui_inactive,
            "authenticatedUiSignal1": self.authenticated_ui_signal_count >= 1,
            "authenticatedUiSignal2": self.authenticated_ui_signal_count >= 2,
            "authenticatedUiSignalCount": self.authenticated_ui_signal_count,
            "authResponseSeen": self.auth_response_seen,
            "authStatusExtracted": self.auth_status_extracted,
            "authStatusSuccess": self.auth_status_success,
            "browserOriginAllowed": self.browser_origin_allowed,
            "gate3aDecision": "AUTHENTICATED" if final_state == "READY" else final_state,
            "readyTransition": final_state == "READY",
        }


def route_gate5a_inbound_frame(
    frame: str | bytes,
    *,
    on_auth_frame: FrameCallback,
    on_statement_response_frame: FrameCallback,
    diagnostics: Gate5AAuthDiagnostics,
) -> Gate3ACompatibleAuthObservation:
    """Keep auth and statement paths separate; never forward unrelated raw frames."""
    auth = reduce_gate3a_compatible_auth_observation(frame)
    if auth.response_seen:
        diagnostics.auth_response_seen = True
        diagnostics.auth_status_success = auth.status_success
        if auth.sanitized_frame is not None:
            diagnostics.auth_status_extracted = True
            on_auth_frame(auth.sanitized_frame)
    if _contains_message(frame, 2, 107):
        on_statement_response_frame(frame)
    return auth


def route_gate5a_outbound_frame(
    frame: str | bytes,
    *,
    on_statement_request_frame: FrameCallback,
) -> None:
    if _contains_message(frame, 2, 7):
        on_statement_request_frame(frame)


class _DiagnosticBrowserSession:
    """Records booleans/counts while preserving Gate 3A session behavior."""

    def __init__(
        self,
        delegate: PlaywrightBrowserSession,
        page: Any,
        diagnostics: Gate5AAuthDiagnostics,
        ui_debugger: Gate5ATempUiDebugger | None,
    ) -> None:
        self.__delegate = delegate
        self.__page = page
        self.__diagnostics = diagnostics
        self.__ui_debugger = ui_debugger

    async def goto_kfh(self) -> int | None:
        status = cast(int | None, await self.__delegate.goto_kfh())
        self.__diagnostics.browser_origin_allowed = is_allowed_kfh_url(self.__page.url)
        if self.__ui_debugger:
            await self.__ui_debugger.document_loaded(self.__page.url)
        return status

    async def login_ui_active(self) -> bool:
        active = cast(bool, await self.__delegate.login_ui_active())
        self.__diagnostics.login_ui_inactive = not active
        return active

    async def otp_ui_active(self) -> bool:
        return cast(bool, await self.__delegate.otp_ui_active())

    async def auth_failed_ui_active(self) -> bool:
        return cast(bool, await self.__delegate.auth_failed_ui_active())

    async def authenticated_ui_signal_count(self) -> int:
        count = cast(int, await self.__delegate.authenticated_ui_signal_count())
        self.__diagnostics.authenticated_ui_signal_count = count
        return count

    async def logout(self) -> None:
        await self.__delegate.logout()

    async def close(self) -> None:
        await self.__delegate.close()

    def is_closed(self) -> bool:
        return cast(bool, self.__delegate.is_closed())


class Gate5APassiveBrowserRuntime:
    """Adds filtered passive observers without changing Gate 3A authentication."""

    def __init__(
        self,
        *,
        on_statement_request_frame: FrameCallback,
        on_statement_response_frame: FrameCallback,
        diagnostics: Gate5AAuthDiagnostics | None = None,
        ui_debugger: Gate5ATempUiDebugger | None = None,
    ) -> None:
        self.__on_statement_request_frame = on_statement_request_frame
        self.__on_statement_response_frame = on_statement_response_frame
        self.diagnostics = diagnostics or Gate5AAuthDiagnostics()
        self.__ui_debugger = ui_debugger
        self.__page: Any | None = None

    async def inspect_pagination_ui(self) -> dict[str, Any]:
        """Read fixed booleans/counts only; never scroll or click the KFH page."""
        if self.__page is None or not is_allowed_kfh_url(self.__page.url):
            raise RuntimeError("KFH Statement page is not available for diagnostics")
        raw = await self.__page.evaluate(PAGINATION_UI_DIAGNOSTIC_SCRIPT)
        return sanitize_pagination_ui_diagnostic(raw)

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
        self.__page = page
        page.on("close", lambda: on_closed())
        delegate = PlaywrightBrowserSession(playwright, browser, context, page)
        diagnostic_session = _DiagnosticBrowserSession(
            delegate, page, self.diagnostics, self.__ui_debugger
        )
        if self.__ui_debugger:
            self.__ui_debugger.browser_opened(page, diagnostic_session)

        def websocket_opened(websocket: Any) -> None:
            allowed = is_allowed_kfh_url(websocket.url.replace("wss://", "https://", 1))
            self.diagnostics.browser_origin_allowed = (
                self.diagnostics.browser_origin_allowed or allowed
            )
            if not allowed:
                return

            role = (
                self.__ui_debugger.websocket_opened(websocket.url)
                if self.__ui_debugger
                else None
            )

            def inbound_frame(frame: str | bytes) -> None:
                auth = route_gate5a_inbound_frame(
                    frame,
                    on_auth_frame=on_inbound_frame,
                    on_statement_response_frame=self.__on_statement_response_frame,
                    diagnostics=self.diagnostics,
                )
                if self.__ui_debugger and role is not None:
                    self.__ui_debugger.observe_inbound_frame(role, frame, auth)
                if self.__ui_debugger and self.diagnostics.auth_status_success:
                    self.__ui_debugger.start(page, diagnostic_session)

            websocket.on("framereceived", inbound_frame)
            websocket.on(
                "framesent",
                lambda frame: route_gate5a_outbound_frame(
                    frame,
                    on_statement_request_frame=self.__on_statement_request_frame,
                ),
            )
            if self.__ui_debugger and role is not None:
                websocket.on(
                    "close", lambda: self.__ui_debugger.websocket_closed(role)
                )

        page.on("websocket", websocket_opened)
        page.on(
            "requestfailed",
            lambda request: on_document_failure() if request.resource_type == "document" else None,
        )
        return diagnostic_session
