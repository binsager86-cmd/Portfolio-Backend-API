"""Gate 3A connector orchestrator exposing lifecycle operations only."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress

from .browser import BrowserRuntime, BrowserSession, PlaywrightBrowserRuntime
from .observer import KfhAuthenticationObserver
from .state import KfhAuthState, KfhConnectionSnapshot, KfhStateMachine

logger = logging.getLogger("saham.kfh_gate3a")


class KfhGate3AConnector:
    """No credentials, browser handles, selectors, or raw messages cross this API."""

    def __init__(self, runtime: BrowserRuntime | None = None, *, poll_interval: float = 0.5) -> None:
        self.__runtime = runtime or PlaywrightBrowserRuntime()
        self.__poll_interval = poll_interval
        self.__machine = KfhStateMachine()
        self.__observer = KfhAuthenticationObserver()
        self.__session: BrowserSession | None = None
        self.__monitor: asyncio.Task[None] | None = None
        self.__closed_signal = False
        self.__document_failure = False
        self.__login_ui_reappeared_streak = 0

    def __transition(self, target: KfhAuthState, reason_code: str | None = None) -> None:
        snapshot = self.__machine.transition(target, reason_code)
        logger.info(
            "KFH_GATE3A_STATE %s",
            json.dumps(snapshot.public_dict(), separators=(",", ":")),
        )

    async def connect(self) -> KfhConnectionSnapshot:
        if self.__machine.snapshot.state != KfhAuthState.DISCONNECTED:
            await self.close()
        self.__observer = KfhAuthenticationObserver()
        self.__closed_signal = False
        self.__document_failure = False
        self.__login_ui_reappeared_streak = 0
        self.__transition(KfhAuthState.OPENING_KFH)
        try:
            self.__session = await self.__runtime.open(
                on_inbound_frame=self.__observer.observe_inbound_frame,
                on_closed=self.__mark_browser_closed,
                on_document_failure=self.__mark_document_failure,
            )
            status = await self.__session.goto_kfh()
            if status is not None and status >= 500:
                self.__transition(KfhAuthState.KFH_UNAVAILABLE, "HTTP_UNAVAILABLE")
                return self.status()
            if await self.__session.login_ui_active():
                self.__transition(KfhAuthState.LOGIN_REQUIRED)
            else:
                self.__transition(KfhAuthState.AUTHENTICATING)
            self.__monitor = asyncio.create_task(self.__monitor_authentication())
        except (TimeoutError, ConnectionError, OSError):
            self.__transition(KfhAuthState.NETWORK_ERROR, "NETWORK_FAILURE")
        except Exception:
            self.__transition(KfhAuthState.CONNECTOR_ERROR, "BROWSER_START_FAILED")
        return self.status()

    def status(self) -> KfhConnectionSnapshot:
        return self.__machine.snapshot

    async def wait_for_ready(self, timeout_seconds: float = 300) -> KfhConnectionSnapshot:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while loop.time() < deadline:
            snapshot = self.status()
            if snapshot.state in {
                KfhAuthState.READY,
                KfhAuthState.AUTH_FAILED,
                KfhAuthState.SESSION_EXPIRED,
                KfhAuthState.BROWSER_CLOSED,
                KfhAuthState.KFH_UNAVAILABLE,
                KfhAuthState.NETWORK_ERROR,
                KfhAuthState.CONNECTOR_ERROR,
            }:
                return snapshot
            await asyncio.sleep(self.__poll_interval)
        return self.status()

    async def logout(self) -> KfhConnectionSnapshot:
        if self.__session and not self.__session.is_closed():
            try:
                await self.__session.logout()
            except Exception:
                pass
        await self.close()
        return self.status()

    async def reconnect(self) -> KfhConnectionSnapshot:
        await self.close()
        return await self.connect()

    async def close(self) -> None:
        current_task = asyncio.current_task()
        if self.__monitor and self.__monitor is not current_task:
            self.__monitor.cancel()
            with suppress(asyncio.CancelledError):
                await self.__monitor
        self.__monitor = None
        if self.__session:
            with suppress(Exception):
                await self.__session.close()
        self.__session = None
        if self.__machine.snapshot.state != KfhAuthState.DISCONNECTED:
            self.__transition(KfhAuthState.DISCONNECTED)

    def __mark_browser_closed(self) -> None:
        self.__closed_signal = True

    def __mark_document_failure(self) -> None:
        self.__document_failure = True

    async def __monitor_authentication(self) -> None:
        while self.__session:
            try:
                if self.__closed_signal or self.__session.is_closed():
                    self.__transition(KfhAuthState.BROWSER_CLOSED, "USER_CLOSED_BROWSER")
                    return
                if self.__document_failure:
                    self.__transition(KfhAuthState.NETWORK_ERROR, "DOCUMENT_NETWORK_FAILURE")
                    return
                login_active = await self.__session.login_ui_active()
                if self.__machine.snapshot.state == KfhAuthState.READY and login_active:
                    # Require the login UI to be observed on two consecutive
                    # polls before treating this as a real session expiry.
                    # A single transient read (a re-render, a loading state
                    # right after authentication) must not kill a healthy
                    # session before the statement fetch can even start.
                    self.__login_ui_reappeared_streak += 1
                    if self.__login_ui_reappeared_streak >= 2:
                        self.__transition(KfhAuthState.SESSION_EXPIRED, "LOGIN_UI_RETURNED")
                        return
                    await asyncio.sleep(self.__poll_interval)
                    continue
                self.__login_ui_reappeared_streak = 0
                if self.__observer.failed or await self.__session.auth_failed_ui_active():
                    self.__transition(KfhAuthState.AUTH_FAILED, "KFH_REJECTED_LOGIN")
                    return
                if await self.__session.otp_ui_active():
                    if self.__machine.snapshot.state != KfhAuthState.OTP_REQUIRED:
                        self.__transition(KfhAuthState.OTP_REQUIRED)
                else:
                    ui_signals = await self.__session.authenticated_ui_signal_count()
                    independently_authenticated = (
                        self.__observer.authenticated and not login_active
                    ) or (ui_signals >= 2 and not login_active)
                    if independently_authenticated:
                        if self.__machine.snapshot.state not in {
                            KfhAuthState.AUTHENTICATED,
                            KfhAuthState.READY,
                        }:
                            self.__transition(KfhAuthState.AUTHENTICATED)
                        if self.__machine.snapshot.state != KfhAuthState.READY:
                            self.__transition(KfhAuthState.READY)
                    elif not login_active and self.__machine.snapshot.state in {
                        KfhAuthState.LOGIN_REQUIRED,
                        KfhAuthState.OTP_REQUIRED,
                    }:
                        self.__transition(KfhAuthState.AUTHENTICATING)
                await asyncio.sleep(self.__poll_interval)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.__transition(KfhAuthState.CONNECTOR_ERROR, "STATE_DETECTION_FAILED")
                return
