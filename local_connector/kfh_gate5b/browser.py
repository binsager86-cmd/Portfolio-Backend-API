"""Fixed-purpose KFH Cash Statement browser transport for Gate 5B-L1."""

from __future__ import annotations

import asyncio
import os
import re
from contextlib import suppress
from typing import Any, TypedDict
from urllib.parse import urlsplit

from local_connector.kfh_gate3a.browser import (
    BROWSER_CONTEXT_OPTIONS,
    BROWSER_LAUNCH_OPTIONS,
    LOGIN_PASSWORD_CONTROL_SELECTOR,
    LOGIN_USERNAME_CONTROL_SELECTOR,
    BrowserSession,
    FrameCallback,
    PlaywrightBrowserSession,
    SignalCallback,
)
from local_connector.kfh_gate3a.policy import KFH_ALLOWED_ORIGINS, is_allowed_kfh_url
from local_connector.kfh_gate5a.browser import (
    Gate5AAuthDiagnostics,
    route_gate5a_inbound_frame,
)


class KfhDiscoveredAccount(TypedDict):
    """One of the owner's own KFH trading accounts, for selection only."""

    secAccNum: str
    portNme: str
    curr: str
    isDefaultAccount: bool


KFH_OTP_CONTROL_SELECTOR = (
    "input[autocomplete='one-time-code'], input[name*='otp' i], input[id*='otp' i]"
)
KFH_LOGIN_SUBMIT_CONTROL_SELECTOR = "button#btnLogin"
KFH_LOGIN_SUBMIT_CONTROL_ID = "btnLogin"
KFH_OTP_SUBMIT_CONTROL_ID = "btnOtpLogin"
KFH_INTERACTIVE_CHALLENGE_MARKERS = (
    "captcha",
    "security challenge",
    "interactive verification",
)


def _gate5b_browser_launch_options() -> dict[str, object]:
    options = {**BROWSER_LAUNCH_OPTIONS}
    options["headless"] = os.environ.get("KFH_LOCAL_DEBUG_VISIBLE_BROWSER") != "true"
    return options

CASH_STATEMENT_SOCKET_HOOK = r"""
(() => {
  const key = Symbol.for("saham.kfh.cash-statement.socket.v1");
  if (window[key]) return;
  const NativeWebSocket = window.WebSocket;
  const state = {
    tradeSockets: new Set(),
    ready: false,
    boundSocket: null,
    authenticatedHed: null,
    ambiguous: false,
    boundSocketClosed: false,
    successfulAuthSocket: null,
    successfulAuthAmbiguous: false,
    // Per-socket accumulator of whichever identity fields have been
    // observed so far, merged from BOTH inbound (server) and outbound
    // (KFH page's own requests) /wstrs traffic. Real KFH evidence shows
    // sesnId/usrId are echoed by the server, but ver/clVer are declared
    // only by the client itself and never appear in server responses -
    // so identity must be assembled incrementally from both directions
    // rather than requiring one single frame to carry all four fields.
    identityBySocket: new Map(),
    // The owner's own trading accounts, as reported by KFH's own post-login
    // account listing (real evidence: DAT.secAccLst). Populated once, the
    // first time it is observed. Only the four fields the account selector
    // needs are retained - never the full ~47-field KFH record.
    discoveredAccounts: undefined,
  };
  const authenticatedFields = ["ver", "clVer", "sesnId", "usrId"];
  const validScalar = (value) =>
    (typeof value === "string" && value.length > 0) ||
    (typeof value === "number" && Number.isFinite(value));
  const hasCompleteIdentity = (identity) =>
    Boolean(identity) && authenticatedFields.every((field) => validScalar(identity[field]));
  state.hasCompleteIdentity = hasCompleteIdentity;
  const mergeIdentity = (socket, hed) => {
    if (!hed || typeof hed !== "object") return;
    let accumulated = state.identityBySocket.get(socket);
    if (!accumulated) {
      accumulated = {};
      state.identityBySocket.set(socket, accumulated);
    }
    for (const field of authenticatedFields) {
      const value = hed[field];
      if (!validScalar(value)) continue;
      if (field in accumulated && accumulated[field] !== value) {
        // Conflicting value for the same field on the same socket: fail
        // closed rather than silently trust either one.
        state.ambiguous = true;
        continue;
      }
      accumulated[field] = value;
    }
  };
  const observeSuccessfulAuthentication = (socket, value) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) return;
    if (Number(value.HED?.msgGrp) === 5 && Number(value.HED?.msgTyp) === 101 &&
        Number(value.DAT?.authSts) === 1) {
      if (!state.successfulAuthSocket) {
        state.successfulAuthSocket = socket;
      } else if (state.successfulAuthSocket !== socket) {
        state.successfulAuthAmbiguous = true;
      }
    }
  };
  // Merges identity fields from any frame, in either direction, at any
  // time - not gated on `ready` - so identity assembly starts from the
  // socket's very first message rather than only after Gate 3A-R1 READY.
  const observeFrame = (socket, value) => {
    if (Array.isArray(value)) {
      value.forEach((child) => observeFrame(socket, child));
    } else if (value && typeof value === "object") {
      if (value.HED && typeof value.HED === "object") {
        mergeIdentity(socket, value.HED);
      }
      if (state.discoveredAccounts === undefined && Array.isArray(value.secAccLst)) {
        const accounts = value.secAccLst
          .filter((entry) => entry && typeof entry === "object" && typeof entry.secAccNum === "string" && entry.secAccNum)
          .map((entry) => ({
            secAccNum: entry.secAccNum,
            portNme: typeof entry.portNme === "string" ? entry.portNme : "",
            curr: typeof entry.curr === "string" ? entry.curr.toUpperCase() : "",
            isDefaultAccount: entry.isDefaultAccount === true,
          }));
        if (accounts.length > 0) state.discoveredAccounts = accounts;
      }
      Object.values(value).forEach((child) => observeFrame(socket, child));
    }
  };
  function WrappedWebSocket(...args) {
    const socket = Reflect.construct(NativeWebSocket, args, new.target || NativeWebSocket);
    try {
      if (new URL(socket.url).pathname.toLowerCase() === "/wstrs") {
        state.tradeSockets.add(socket);
        socket.addEventListener("message", (event) => {
          if (typeof event.data !== "string") return;
          try {
            const value = JSON.parse(event.data);
            observeFrame(socket, value);
            if (!state.ready) observeSuccessfulAuthentication(socket, value);
          } catch {
            // Non-JSON traffic cannot establish authenticated HED context.
          }
        });
        const nativeSend = socket.send.bind(socket);
        socket.send = (data) => {
          if (typeof data === "string") {
            try {
              observeFrame(socket, JSON.parse(data));
            } catch {
              // Outbound non-JSON traffic carries no identity information.
            }
          }
          return nativeSend(data);
        };
        socket.addEventListener("close", () => {
          state.tradeSockets.delete(socket);
          if (state.boundSocket === socket) {
            state.boundSocketClosed = true;
            state.authenticatedHed = null;
          }
        });
      }
    } catch {
      // Unknown sockets are never retained or used.
    }
    return socket;
  }
  Object.setPrototypeOf(WrappedWebSocket, NativeWebSocket);
  WrappedWebSocket.prototype = NativeWebSocket.prototype;
  Object.defineProperty(window, key, {
    value: state,
    enumerable: false,
    configurable: false,
    writable: false,
  });
  window.WebSocket = WrappedWebSocket;
})();
"""

CASH_STATEMENT_DISCOVERED_ACCOUNTS_SCRIPT = r"""
() => {
  const state = window[Symbol.for("saham.kfh.cash-statement.socket.v1")];
  return state && state.discoveredAccounts !== undefined ? state.discoveredAccounts : null;
}
"""

CASH_STATEMENT_MARK_READY_SCRIPT = r"""
() => {
  const state = window[Symbol.for("saham.kfh.cash-statement.socket.v1")];
  if (!state) return false;
  state.ready = true;
  state.boundSocket = state.successfulAuthAmbiguous ? null : state.successfulAuthSocket;
  if (state.boundSocket) {
    if (!state.identityBySocket.has(state.boundSocket)) {
      state.identityBySocket.set(state.boundSocket, {});
    }
    // Same object reference as the accumulator: identity fields observed
    // on later frames (either direction) continue to fill this in place.
    state.authenticatedHed = state.identityBySocket.get(state.boundSocket);
  } else {
    state.authenticatedHed = null;
  }
  state.ambiguous = state.ambiguous || state.successfulAuthAmbiguous;
  state.boundSocketClosed = false;
  state.successfulAuthSocket = null;
  return true;
}
"""

CASH_STATEMENT_CONTEXT_STATUS_SCRIPT = r"""
() => {
  const state = window[Symbol.for("saham.kfh.cash-statement.socket.v1")];
  if (!state || !state.ready) return "NOT_AVAILABLE";
  if (state.ambiguous) return "AMBIGUOUS";
  if (state.boundSocketClosed) return "CLOSED";
  if (!state.boundSocket || !state.hasCompleteIdentity(state.authenticatedHed) ||
      state.boundSocket.readyState !== WebSocket.OPEN) return "NOT_AVAILABLE";
  return "AVAILABLE";
}
"""

CASH_STATEMENT_CONTEXT_DIAGNOSTICS_SCRIPT = r"""
() => {
  const state = window[Symbol.for("saham.kfh.cash-statement.socket.v1")];
  const socketOpen = Boolean(
    state && [...state.tradeSockets].some(
      (candidate) => candidate.readyState === WebSocket.OPEN
    )
  );
  let status = "NOT_AVAILABLE";
  if (state?.ambiguous) status = "AMBIGUOUS";
  else if (state?.boundSocketClosed) status = "CLOSED";
  else if (state?.ready && state?.boundSocket && state.hasCompleteIdentity(state.authenticatedHed) &&
           state.boundSocket.readyState === WebSocket.OPEN) status = "AVAILABLE";
  return {
    status,
    authenticatedHed: status === "AVAILABLE",
    authenticatedSocket: socketOpen,
  };
}
"""

CASH_STATEMENT_SEND_SCRIPT = r"""
(request) => {
  const expected = [
    "frmDate", "secAccNum", "sortMode", "startSeq", "toDate", "totalNoRec", "unqReqId"
  ];
  const keys = Object.keys(request).sort();
  if (JSON.stringify(keys) !== JSON.stringify(expected)) {
    throw new Error("Cash Statement request fields rejected");
  }
  if (typeof request.secAccNum !== "string" || !request.secAccNum.trim()) {
    throw new Error("Cash Statement account rejected");
  }
  if (!/^\d{8}$/.test(request.frmDate) || !/^\d{8}$/.test(request.toDate)) {
    throw new Error("Cash Statement date range rejected");
  }
  if (!Number.isInteger(request.startSeq) || request.startSeq < 0 ||
      request.startSeq % 20 !== 0 || request.totalNoRec !== 20 ||
      request.sortMode !== 0 || typeof request.unqReqId !== "string" ||
      !request.unqReqId) {
    throw new Error("Cash Statement paging request rejected");
  }
  const state = window[Symbol.for("saham.kfh.cash-statement.socket.v1")];
  if (!state || !state.ready || !state.boundSocket ||
      !state.hasCompleteIdentity(state.authenticatedHed)) {
    throw new Error("AUTHENTICATED_CASH_STATEMENT_CONTEXT_NOT_AVAILABLE");
  }
  if (state.ambiguous) throw new Error("KFH_AUTHENTICATED_SOCKET_AMBIGUOUS");
  if (state.boundSocketClosed || state.boundSocket.readyState !== WebSocket.OPEN) {
    throw new Error("KFH_SESSION_EXPIRED");
  }
  const socket = state.boundSocket;
  const authenticatedHed = state.authenticatedHed;
  socket.send(JSON.stringify({
    HED: {
      ver: authenticatedHed.ver,
      msgGrp: 2,
      msgTyp: 7,
      chnlId: 30,
      clVer: authenticatedHed.clVer,
      sesnId: authenticatedHed.sesnId,
      usrId: authenticatedHed.usrId,
    },
    DAT: {
      secAccNum: request.secAccNum,
      frmDate: request.frmDate,
      toDate: request.toDate,
      sortMode: request.sortMode,
      startSeq: request.startSeq,
      totalNoRec: request.totalNoRec,
      unqReqId: request.unqReqId,
    },
  }));
  return true;
}
"""


class KfhLoginAutofillError(RuntimeError):
    """Sanitized fixed-login autofill failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _KfhLoginHandoff:
    """Private DOM handles retained only until KFH consumes the login form."""

    __slots__ = (
        "username_control",
        "password_control",
        "username_autofill_confirmed",
        "password_autofill_confirmed",
        "login_submit_triggered",
    )

    def __init__(
        self,
        *,
        username_control: Any,
        password_control: Any,
        username_autofill_confirmed: bool,
        password_autofill_confirmed: bool,
    ) -> None:
        self.username_control = username_control
        self.password_control = password_control
        self.username_autofill_confirmed = username_autofill_confirmed
        self.password_autofill_confirmed = password_autofill_confirmed
        self.login_submit_triggered = True

    def public_confirmation(self) -> dict[str, bool]:
        return {
            "usernameAutofillConfirmed": self.username_autofill_confirmed,
            "passwordAutofillConfirmed": self.password_autofill_confirmed,
            "loginSubmitTriggered": self.login_submit_triggered,
        }


class _KfhOtpHandoff:
    """Private OTP DOM handle retained only until KFH consumes the challenge."""

    __slots__ = ("otp_control",)

    def __init__(self, otp_control: Any) -> None:
        self.otp_control = otp_control


async def _one_visible(locator: Any) -> Any | None:
    visible: list[Any] = []
    for index in range(await locator.count()):
        candidate = locator.nth(index)
        if await candidate.is_visible(timeout=200):
            visible.append(candidate)
    return visible[0] if len(visible) == 1 else None


async def _one_actionable_login_submit(
    locator: Any, *, expected_id: str | None = None
) -> Any | None:
    control = await _one_visible(locator)
    if control is None:
        return None
    try:
        control_id = await control.get_attribute("id")
        if control_id == KFH_OTP_SUBMIT_CONTROL_ID:
            return None
        if expected_id is not None and control_id != expected_id:
            return None
        if not await control.is_enabled(timeout=200):
            return None
        await control.click(timeout=5_000, trial=True)
    except Exception:
        return None
    return control


async def _fixed_login_controls(page: Any) -> tuple[Any, Any, Any] | None:
    username_control = await _one_visible(page.locator(LOGIN_USERNAME_CONTROL_SELECTOR))
    password_control = await _one_visible(page.locator(LOGIN_PASSWORD_CONTROL_SELECTOR))
    submit_control = await _one_actionable_login_submit(
        page.locator(KFH_LOGIN_SUBMIT_CONTROL_SELECTOR),
        expected_id=KFH_LOGIN_SUBMIT_CONTROL_ID,
    )
    if submit_control is None:
        submit_control = await _one_actionable_login_submit(
        page.get_by_role(
            "button",
            name=re.compile(r"^(login|sign in)$", re.IGNORECASE),
            exact=True,
        )
    )
    if submit_control is None:
        submit_control = await _one_actionable_login_submit(
            page.locator("input[type='submit']")
        )
    if username_control is None or password_control is None or submit_control is None:
        return None
    return username_control, password_control, submit_control


async def _wait_for_fixed_login_controls(
    page: Any, *, timeout_seconds: float, poll_interval: float
) -> tuple[Any, Any, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while True:
        parsed = urlsplit(page.url)
        origin = f"{parsed.scheme}://{parsed.netloc}".lower()
        if origin not in KFH_ALLOWED_ORIGINS:
            raise KfhLoginAutofillError("KFH_LOGIN_ORIGIN_REJECTED")
        with suppress(Exception):
            controls = await _fixed_login_controls(page)
            if controls is not None:
                return controls
        if loop.time() >= deadline:
            raise KfhLoginAutofillError("KFH_LOGIN_FIELDS_NOT_FOUND")
        await asyncio.sleep(poll_interval)


# Best-effort only: KFH navigates away from the login view after a real
# authentication, so these controls are commonly already detached. Cleanup
# must never block the statement reader behind Playwright's much longer
# default actionability timeout - a stale/detached control is treated as
# already gone, not retried.
_DOM_CLEANUP_TIMEOUT_MS = 1_000


async def _clear_login_handoff(handoff: _KfhLoginHandoff | None) -> None:
    if handoff is None:
        return
    with suppress(Exception):
        await handoff.username_control.fill("", timeout=_DOM_CLEANUP_TIMEOUT_MS)
    with suppress(Exception):
        await handoff.password_control.fill("", timeout=_DOM_CLEANUP_TIMEOUT_MS)


async def _clear_otp_handoff(handoff: _KfhOtpHandoff | None) -> None:
    if handoff is None:
        return
    with suppress(Exception):
        await handoff.otp_control.fill("", timeout=_DOM_CLEANUP_TIMEOUT_MS)


async def _submit_fixed_kfh_login(
    page: Any,
    username: str,
    password: str,
    *,
    timeout_seconds: float = 15,
    poll_interval: float = 0.1,
) -> _KfhLoginHandoff:
    """Wait, verify, and submit only the fixed controls on the approved origin."""
    parsed = urlsplit(page.url)
    origin = f"{parsed.scheme}://{parsed.netloc}".lower()
    if origin not in KFH_ALLOWED_ORIGINS:
        raise KfhLoginAutofillError("KFH_LOGIN_ORIGIN_REJECTED")

    username_control: Any | None = None
    password_control: Any | None = None
    submitted = False
    try:
        username_control, password_control, submit_control = await _wait_for_fixed_login_controls(
            page,
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
        )
        try:
            await username_control.fill(username)
            await password_control.fill(password)
            username_matches = await username_control.input_value() == username
            password_matches = await password_control.input_value() == password
        except Exception as error:
            raise KfhLoginAutofillError("KFH_LOGIN_AUTOFILL_FAILED") from error
        if not username_matches or not password_matches:
            raise KfhLoginAutofillError("KFH_LOGIN_AUTOFILL_FAILED")

        await submit_control.click(timeout=5_000)
        submitted = True
        return _KfhLoginHandoff(
            username_control=username_control,
            password_control=password_control,
            username_autofill_confirmed=username_matches,
            password_autofill_confirmed=password_matches,
        )
    except KfhLoginAutofillError:
        raise
    except Exception as error:
        code = "KFH_LOGIN_SUBMIT_FAILED" if username_control is not None else "KFH_LOGIN_FIELDS_NOT_FOUND"
        raise KfhLoginAutofillError(code) from error
    finally:
        if not submitted and username_control is not None:
            with suppress(Exception):
                await username_control.fill("")
        if not submitted and password_control is not None:
            with suppress(Exception):
                await password_control.fill("")
        username = ""
        password = ""


async def _submit_fixed_kfh_otp(
    page: Any,
    verification_code: str,
    *,
    timeout_seconds: float = 15,
    poll_interval: float = 0.1,
) -> _KfhOtpHandoff:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    otp_control: Any | None = None
    submitted = False
    try:
        while True:
            parsed = urlsplit(page.url)
            origin = f"{parsed.scheme}://{parsed.netloc}".lower()
            if origin not in KFH_ALLOWED_ORIGINS:
                raise KfhLoginAutofillError("KFH_LOGIN_ORIGIN_REJECTED")
            with suppress(Exception):
                otp_control = await _one_visible(page.locator(KFH_OTP_CONTROL_SELECTOR))
                submit_control = await _one_visible(
                    page.get_by_role(
                        "button",
                        name=re.compile(r"^(verify|continue|submit)$", re.IGNORECASE),
                        exact=True,
                    )
                )
                if otp_control is not None and submit_control is not None:
                    break
            if loop.time() >= deadline:
                raise KfhLoginAutofillError("KFH_OTP_FIELDS_NOT_FOUND")
            await asyncio.sleep(poll_interval)

        await otp_control.fill(verification_code)
        if await otp_control.input_value() != verification_code:
            raise KfhLoginAutofillError("KFH_OTP_AUTOFILL_FAILED")
        await submit_control.click(timeout=5_000)
        submitted = True
        return _KfhOtpHandoff(otp_control)
    except KfhLoginAutofillError:
        raise
    except Exception as error:
        raise KfhLoginAutofillError("KFH_OTP_SUBMIT_FAILED") from error
    finally:
        if not submitted and otp_control is not None:
            with suppress(Exception):
                await otp_control.fill("")
        verification_code = ""


class Gate5BLiveBrowserRuntime:
    """Adds one fixed Cash Statement send operation to an ephemeral Gate 3A session."""

    def __init__(self, *, on_statement_response_frame: FrameCallback) -> None:
        self.__on_statement_response_frame = on_statement_response_frame
        self.__page: Any | None = None
        self.__diagnostics = Gate5AAuthDiagnostics()
        self.__login_handoff: _KfhLoginHandoff | None = None
        self.__otp_handoff: _KfhOtpHandoff | None = None
        self.__visible_debug_browser = (
            os.environ.get("KFH_LOCAL_DEBUG_VISIBLE_BROWSER") == "true"
        )

    async def send_cash_statement(self, request: dict[str, object]) -> None:
        """Construct and send only KFH Cash Statement 2/7 through the trade socket."""
        page = self.__page
        if page is None or page.is_closed() or not is_allowed_kfh_url(page.url):
            raise RuntimeError("KFH Cash Statement browser session is unavailable")
        sent = await page.evaluate(CASH_STATEMENT_SEND_SCRIPT, request)
        if sent is not True:
            raise RuntimeError("KFH Cash Statement request was not sent")

    async def _mark_gate3a_ready(self) -> None:
        """Promote the exact successful-auth socket after Gate 3A-R1 is READY."""
        page = self.__page
        if page is None or page.is_closed() or not is_allowed_kfh_url(page.url):
            raise RuntimeError("KFH Cash Statement browser session is unavailable")
        marked = await page.evaluate(CASH_STATEMENT_MARK_READY_SCRIPT)
        if marked is not True:
            raise RuntimeError("KFH authenticated context observer is unavailable")

    async def _discovered_accounts(self) -> list[KfhDiscoveredAccount] | None:
        """The owner's own trading accounts, as reported by KFH's own
        post-login account listing. Returns None while not yet observed."""
        page = self.__page
        if page is None or page.is_closed() or not is_allowed_kfh_url(page.url):
            return None
        value = await page.evaluate(CASH_STATEMENT_DISCOVERED_ACCOUNTS_SCRIPT)
        if not isinstance(value, list):
            return None
        accounts: list[KfhDiscoveredAccount] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            sec_acc_num = entry.get("secAccNum")
            if not isinstance(sec_acc_num, str) or not sec_acc_num:
                continue
            accounts.append(
                {
                    "secAccNum": sec_acc_num,
                    "portNme": str(entry.get("portNme") or ""),
                    "curr": str(entry.get("curr") or ""),
                    "isDefaultAccount": entry.get("isDefaultAccount") is True,
                }
            )
        return accounts

    async def _authenticated_context_status(self) -> str:
        """Return sanitized status; authenticated HED values never leave the page."""
        page = self.__page
        if page is None or page.is_closed() or not is_allowed_kfh_url(page.url):
            return "CLOSED"
        status = await page.evaluate(CASH_STATEMENT_CONTEXT_STATUS_SCRIPT)
        if status not in {"AVAILABLE", "NOT_AVAILABLE", "AMBIGUOUS", "CLOSED"}:
            return "NOT_AVAILABLE"
        return str(status)

    async def _authenticated_context_diagnostics(self) -> dict[str, bool | str]:
        """Return only allowlisted booleans and status; never HED values."""
        page = self.__page
        fallback: dict[str, bool | str] = {
            "status": "CLOSED",
            "authenticatedHed": False,
            "authenticatedSocket": False,
        }
        if page is None or page.is_closed() or not is_allowed_kfh_url(page.url):
            return fallback
        value = await page.evaluate(CASH_STATEMENT_CONTEXT_DIAGNOSTICS_SCRIPT)
        if not isinstance(value, dict):
            return fallback
        status = value.get("status")
        if status not in {"AVAILABLE", "NOT_AVAILABLE", "AMBIGUOUS", "CLOSED"}:
            return fallback
        return {
            "status": str(status),
            "authenticatedHed": value.get("authenticatedHed") is True,
            "authenticatedSocket": value.get("authenticatedSocket") is True,
        }

    async def _submit_login_credentials(
        self, username: str, password: str
    ) -> dict[str, bool]:
        """One fixed origin/field/submit operation; no selector API is exposed."""
        page = self.__page
        if page is None or page.is_closed():
            raise KfhLoginAutofillError("KFH_LOGIN_PAGE_NOT_FOUND")
        if self.__login_handoff is not None:
            raise KfhLoginAutofillError("KFH_LOGIN_SUBMIT_FAILED")
        handoff = await _submit_fixed_kfh_login(page, username, password)
        self.__login_handoff = handoff
        return handoff.public_confirmation()

    async def _clear_login_dom_credentials(self) -> None:
        """Clear KFH DOM values only after consumption, outcome, timeout, or close."""
        handoff = self.__login_handoff
        self.__login_handoff = None
        await _clear_login_handoff(handoff)

    async def _submit_otp(self, verification_code: str) -> dict[str, bool]:
        page = self.__page
        if page is None or page.is_closed():
            raise KfhLoginAutofillError("KFH_LOGIN_PAGE_NOT_FOUND")
        if self.__otp_handoff is not None:
            raise KfhLoginAutofillError("KFH_OTP_SUBMIT_FAILED")
        handoff = await _submit_fixed_kfh_otp(page, verification_code)
        self.__otp_handoff = handoff
        return {"otpSubmitTriggered": True}

    async def _clear_otp_dom_credentials(self) -> None:
        handoff = self.__otp_handoff
        self.__otp_handoff = None
        await _clear_otp_handoff(handoff)

    async def _interactive_challenge_present(self) -> bool:
        page = self.__page
        if page is None or page.is_closed() or not is_allowed_kfh_url(page.url):
            return False
        for marker in KFH_INTERACTIVE_CHALLENGE_MARKERS:
            try:
                locator = page.get_by_text(re.compile(re.escape(marker), re.IGNORECASE), exact=False)
                if await locator.first.is_visible(timeout=200):
                    return True
            except Exception:
                continue
        return False

    def _visible_browser_enabled(self) -> bool:
        return self.__visible_debug_browser

    async def open(
        self,
        *,
        on_inbound_frame: FrameCallback,
        on_closed: SignalCallback,
        on_document_failure: SignalCallback,
    ) -> BrowserSession:
        from playwright.async_api import async_playwright

        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(**_gate5b_browser_launch_options())
        context = await browser.new_context(**BROWSER_CONTEXT_OPTIONS)
        await context.add_init_script(CASH_STATEMENT_SOCKET_HOOK)

        async def route_request(route: Any) -> None:
            if is_allowed_kfh_url(route.request.url):
                await route.continue_()
            else:
                await route.abort("blockedbyclient")

        await context.route("**/*", route_request)
        page = await context.new_page()
        self.__page = page
        def browser_closed() -> None:
            self.__login_handoff = None
            self.__otp_handoff = None
            on_closed()

        page.on("close", browser_closed)

        def websocket_opened(websocket: Any) -> None:
            if not is_allowed_kfh_url(websocket.url.replace("wss://", "https://", 1)):
                return

            def inbound_frame(frame: str | bytes) -> None:
                route_gate5a_inbound_frame(
                    frame,
                    on_auth_frame=on_inbound_frame,
                    on_statement_response_frame=self.__on_statement_response_frame,
                    diagnostics=self.__diagnostics,
                )

            websocket.on("framereceived", inbound_frame)

        page.on("websocket", websocket_opened)
        page.on(
            "requestfailed",
            lambda request: on_document_failure()
            if request.resource_type == "document"
            else None,
        )
        return PlaywrightBrowserSession(playwright, browser, context, page)
