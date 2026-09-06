"""Temporary local-only sanitized UI-signal diagnostics for Gate 5A."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from urllib.parse import urlsplit

from local_connector.kfh_gate3a.browser import AUTHENTICATED_MARKERS, LOGIN_MARKERS

SAMPLE_OFFSETS_SECONDS = (0, 1, 2, 5, 10, 15)

CLOSED_SIGNAL_NAMES = {
    "Statements": "STATEMENTS_SIGNAL",
    "Portfolio": "PORTFOLIO_SIGNAL",
    "Account Summary": "ACCOUNT_SUMMARY_SIGNAL",
    "Buying Power": "BUYING_POWER_SIGNAL",
}

LOGIN_SIGNAL_NAMES = dict(
    zip(
        LOGIN_MARKERS,
        (
            "LOGIN_TEXT_MARKER",
            "SIGN_IN_TEXT_MARKER",
            "USERNAME_TEXT_MARKER",
            "USER_NAME_TEXT_MARKER",
        ),
        strict=True,
    )
)

SOCKET_ROLE_TRADE = "TRADE"
SOCKET_ROLE_PRICE = "PRICE"
SOCKET_ROLE_UNKNOWN = "UNKNOWN_ALLOWED_KFH_SOCKET"


@dataclass(frozen=True, slots=True)
class _TextCandidate:
    name: str
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SelectorCandidate:
    name: str
    selector: str


TEXT_CANDIDATES = (
    _TextCandidate("STATEMENTS_EN", ("Statements", "Statement")),
    _TextCandidate("PORTFOLIO_EN", ("Portfolio",)),
    _TextCandidate("ACCOUNT_SUMMARY_EN", ("Account Summary",)),
    _TextCandidate("BUYING_POWER_EN", ("Buying Power",)),
    _TextCandidate("LOGOUT_SIGN_OUT_EN", ("Logout", "Log out", "Sign out")),
    _TextCandidate("STATEMENTS_AR", ("كشف الحساب", "كشوفات الحساب", "كشوفات")),
    _TextCandidate("PORTFOLIO_AR", ("المحفظة",)),
    _TextCandidate("ACCOUNT_SUMMARY_AR", ("ملخص الحساب",)),
    _TextCandidate("BUYING_POWER_AR", ("القوة الشرائية",)),
    _TextCandidate("LOGOUT_AR", ("تسجيل الخروج",)),
)

SELECTOR_CANDIDATES = (
    _SelectorCandidate("AUTH_NAV_CONTAINER", "nav, [role='navigation']"),
    _SelectorCandidate("STATEMENT_TAB_CONTAINER", "[role='tablist'], [role='tab']"),
    _SelectorCandidate(
        "PORTFOLIO_ACCOUNT_WIDGET",
        "[id*='portfolio' i], [class*='portfolio' i], "
        "[id*='account' i], [class*='account' i]",
    ),
)


def create_temp_debug_path() -> Path:
    directory = Path(gettempdir()) / "saham-kfh"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return directory / f"gate5a-ui-debug-{timestamp}.jsonl"


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "NON_HTTP_ORIGIN"
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def socket_role(url: str) -> str:
    """Reduce an allowed KFH WebSocket URL to a path-only symbolic role."""
    path = urlsplit(url).path.lower()
    if path == "/wstrs":
        return SOCKET_ROLE_TRADE
    if path == "/wsqs":
        return SOCKET_ROLE_PRICE
    return SOCKET_ROLE_UNKNOWN


def _json_frame(frame: str | bytes) -> bool:
    if not isinstance(frame, str) or len(frame) > 5_000_000:
        return False
    try:
        json.loads(frame)
    except (json.JSONDecodeError, TypeError):
        return False
    return True


async def _locator_result(locator: Any) -> dict[str, bool | int]:
    try:
        count = await locator.count()
        visible = count > 0 and await locator.first.is_visible(timeout=200)
    except Exception:
        count = 0
        visible = False
    return {"matched": count > 0, "visible": visible, "matchCount": count}


async def _text_candidate_result(scope: Any, candidate: _TextCandidate) -> dict[str, Any]:
    total_count = 0
    visible = False
    for value in candidate.values:
        locator = scope.get_by_text(re.compile(re.escape(value), re.IGNORECASE), exact=False)
        result = await _locator_result(locator)
        total_count += int(result["matchCount"])
        visible = visible or bool(result["visible"])
    return {
        "candidate": candidate.name,
        "found": total_count > 0,
        "visible": visible,
        "matchCount": total_count,
    }


async def _selector_candidate_result(
    scope: Any, candidate: _SelectorCandidate
) -> dict[str, Any]:
    result = await _locator_result(scope.locator(candidate.selector))
    return {
        "candidate": candidate.name,
        "found": result["matched"],
        "visible": result["visible"],
        "matchCount": result["matchCount"],
    }


async def _candidate_results(scope: Any) -> list[dict[str, Any]]:
    text_results = [
        await _text_candidate_result(scope, candidate) for candidate in TEXT_CANDIDATES
    ]
    selector_results = [
        await _selector_candidate_result(scope, candidate)
        for candidate in SELECTOR_CANDIDATES
    ]
    return text_results + selector_results


async def _login_signal_result(
    page: Any, marker: str, symbolic_name: str
) -> dict[str, Any]:
    """Inspect one sealed Gate 3A marker while retaining booleans only."""
    locator = page.get_by_text(
        re.compile(re.escape(marker), re.IGNORECASE), exact=False
    )
    try:
        count = await locator.count()
    except Exception:
        count = 0

    visible = False
    has_nonzero_bounding_box = False
    ancestor_visible = False
    if count > 0:
        with suppress(Exception):
            visible = bool(await locator.first.is_visible(timeout=200))
        with suppress(Exception):
            box = await locator.first.bounding_box(timeout=200)
            has_nonzero_bounding_box = bool(
                box and box.get("width", 0) > 0 and box.get("height", 0) > 0
            )
        with suppress(Exception):
            ancestor = locator.first.locator("xpath=..")
            ancestor_visible = bool(await ancestor.is_visible(timeout=200))

    return {
        "signal": symbolic_name,
        "matched": count > 0,
        "visible": visible,
        "matchCount": count,
        "hasNonzeroBoundingBox": has_nonzero_bounding_box,
        "ancestorVisible": ancestor_visible,
    }


async def _login_signal_results(page: Any) -> list[dict[str, Any]]:
    return [
        await _login_signal_result(page, marker, symbolic_name)
        for marker, symbolic_name in LOGIN_SIGNAL_NAMES.items()
    ]


class Gate5ATempUiDebugger:
    """Writes a bounded, sanitized JSONL file outside the repository."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or create_temp_debug_path()
        self.path.touch(exist_ok=False)
        self._task: asyncio.Task[None] | None = None
        self._completed = asyncio.Event()
        self._samples: list[dict[str, Any]] = []
        self._debug_failed = False
        self._finalized = False
        self._final_record: dict[str, Any] | None = None
        self._page: Any | None = None
        self._closed_gate3a_session: Any | None = None
        self._browser_opened = False
        self._document_loaded = False
        self._socket_state = {
            SOCKET_ROLE_TRADE: "NOT_OBSERVED",
            SOCKET_ROLE_PRICE: "NOT_OBSERVED",
        }
        self._trade_inbound_frame_count = 0
        self._price_inbound_frame_count = 0
        self._unrelated_json_frame_count = 0
        self._unparsed_frame_count = 0
        self._auth_response_seen = False
        self._auth_status_extracted = False
        self._auth_status_success = False
        self._owner_visual_marker = False
        self._pre_auth_ui_state: dict[str, Any] | None = None
        self._append({"event": "DEBUG_STARTED"})
        self._progress("DEBUG_STARTED")

    @property
    def completed(self) -> asyncio.Event:
        return self._completed

    @property
    def samples(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._samples)

    @property
    def ui_sampling_started(self) -> bool:
        return self._task is not None

    @property
    def finalized(self) -> bool:
        return self._finalized

    def browser_opened(self, page: Any, closed_gate3a_session: Any) -> None:
        self._page = page
        self._closed_gate3a_session = closed_gate3a_session
        if self._browser_opened:
            return
        self._browser_opened = True
        self._append({"event": "BROWSER_OPENED"})
        self._progress("BROWSER_OPENED")

    async def document_loaded(self, url: str) -> None:
        if not self._document_loaded:
            self._document_loaded = True
            self._append({"event": "KFH_DOCUMENT_LOADED", "origin": _origin(url)})
            self._progress("KFH_DOCUMENT_LOADED")
        await self.record_pre_auth_ui_state()

    def websocket_opened(self, url: str) -> str:
        role = socket_role(url)
        if role in self._socket_state:
            self._socket_state[role] = "OPENED"
        self._append({"event": "KFH_WEBSOCKET_OPENED", "socketRole": role})
        self._progress(f"KFH_WEBSOCKET_OPENED {role}")
        return role

    def websocket_closed(self, role: str) -> None:
        if role in self._socket_state:
            self._socket_state[role] = "CLOSED"
        self._append({"event": "KFH_WEBSOCKET_CLOSED", "socketRole": role})
        self._progress(f"KFH_WEBSOCKET_CLOSED {role}")

    def observe_inbound_frame(self, role: str, frame: str | bytes, auth: Any) -> None:
        """Reduce a frame to counters/auth booleans and immediately discard content."""
        if role == SOCKET_ROLE_TRADE:
            self._trade_inbound_frame_count += 1
        elif role == SOCKET_ROLE_PRICE:
            self._price_inbound_frame_count += 1

        if bool(auth.response_seen):
            self._auth_response_seen = True
            self._append(
                {"event": "AUTH_RESPONSE_IDENTITY_SEEN", "msgGrp": 5, "msgTyp": 101}
            )
            self._progress("AUTH_RESPONSE_IDENTITY_SEEN 5/101")
            if auth.sanitized_frame is not None:
                self._auth_status_extracted = True
                self._auth_status_success = bool(auth.status_success)
                self._append(
                    {"event": "AUTH_STATUS_REDUCED", "success": self._auth_status_success}
                )
                result = "SUCCESS" if self._auth_status_success else "NOT_SUCCESS"
                self._progress(f"AUTH_STATUS_REDUCED {result}")
            return

        if _json_frame(frame):
            self._unrelated_json_frame_count += 1
        else:
            self._unparsed_frame_count += 1

    def owner_visual_login_marker(self) -> None:
        if self._owner_visual_marker:
            return
        self._owner_visual_marker = True
        self._append({"event": "OWNER_VISUAL_LOGIN_MARKER"})
        self._progress("OWNER_VISUAL_LOGIN_MARKER")

    def start(self, page: Any, closed_gate3a_session: Any) -> None:
        if not self._auth_status_success or self._task is not None:
            return
        self._page = page
        self._closed_gate3a_session = closed_gate3a_session
        self._append({"event": "AUTH_SUCCESS_TRIGGER"})
        self._progress("AUTH_SUCCESS_TRIGGER; UI sampling T+0..15s")
        self._task = asyncio.create_task(self._run(page, closed_gate3a_session))

    async def wait(self, timeout_seconds: float = 600) -> bool:
        try:
            await asyncio.wait_for(self._completed.wait(), timeout_seconds)
            return True
        except TimeoutError:
            return False

    async def record_pre_auth_ui_state(self) -> None:
        if (
            self._auth_status_success
            or self._page is None
            or self._closed_gate3a_session is None
        ):
            return
        page = self._page
        session = self._closed_gate3a_session
        try:
            ready_state = await page.evaluate("document.readyState")
        except Exception:
            ready_state = "UNAVAILABLE"
        if ready_state not in {"loading", "interactive", "complete", "UNAVAILABLE"}:
            ready_state = "UNAVAILABLE"
        try:
            state = {
                "loginUiActive": bool(await session.login_ui_active()),
                "otpUiActive": bool(await session.otp_ui_active()),
                "authFailedUiActive": bool(await session.auth_failed_ui_active()),
                "currentOrigin": _origin(page.url),
                "documentReadyState": ready_state,
                "numberOfFrames": len(page.frames),
            }
        except Exception:
            state = {
                "loginUiActive": None,
                "otpUiActive": None,
                "authFailedUiActive": None,
                "currentOrigin": _origin(getattr(page, "url", "")),
                "documentReadyState": ready_state,
                "numberOfFrames": len(getattr(page, "frames", ())),
            }
        self._pre_auth_ui_state = state
        self._append({"event": "PRE_AUTH_UI_STATE", **state})

    async def _run(self, page: Any, closed_gate3a_session: Any) -> None:
        try:
            previous_offset = 0
            for offset in SAMPLE_OFFSETS_SECONDS:
                await asyncio.sleep(offset - previous_offset)
                previous_offset = offset
                sample = await self._sample(page, closed_gate3a_session, offset)
                self._samples.append(sample)
                self._append(sample)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._debug_failed = True
            self._append({"event": "DEBUG_FAILURE", "failed": True})
            self._progress("DEBUG_FAILURE")
        finally:
            self._completed.set()

    async def _sample(
        self, page: Any, closed_gate3a_session: Any, offset: int
    ) -> dict[str, Any]:
        closed_login_ui_active = await closed_gate3a_session.login_ui_active()
        login_signals = await _login_signal_results(page)
        closed_signal_count = await closed_gate3a_session.authenticated_ui_signal_count()
        would_gate3a_authenticate = (
            self._auth_status_success
            and closed_signal_count >= 1
            and not closed_login_ui_active
        ) or (closed_signal_count >= 2 and not closed_login_ui_active)
        closed_signals = []
        for marker in AUTHENTICATED_MARKERS:
            locator = page.get_by_text(
                re.compile(re.escape(marker), re.IGNORECASE), exact=False
            )
            result = await _locator_result(locator)
            closed_signals.append(
                {"signal": CLOSED_SIGNAL_NAMES[marker], **result}
            )

        main_candidates = await _candidate_results(page.main_frame)
        child_marker_found = False
        for frame in page.frames[1:]:
            frame_candidates = await _candidate_results(frame)
            child_marker_found = child_marker_found or any(
                candidate["visible"] for candidate in frame_candidates
            )

        try:
            ready_state = await page.evaluate("document.readyState")
        except Exception:
            ready_state = "UNAVAILABLE"
        if ready_state not in {"loading", "interactive", "complete", "UNAVAILABLE"}:
            ready_state = "UNAVAILABLE"

        return {
            "event": "UI_SIGNAL_SAMPLE",
            "sampleOffsetSeconds": offset,
            "closedLoginUiActive": closed_login_ui_active,
            "loginSignals": login_signals,
            "closedGate3A": {
                "signalCount": closed_signal_count,
                "signals": closed_signals,
            },
            "decisionMatrix": {
                "authProtocolSuccess": self._auth_status_success,
                "loginUiActive": closed_login_ui_active,
                "authenticatedUiSignalCount": closed_signal_count,
                "wouldGate3AAuthenticate": would_gate3a_authenticate,
            },
            "tempCandidates": main_candidates,
            "pageState": {
                "currentOrigin": _origin(page.url),
                "documentReadyState": ready_state,
                "numberOfFrames": len(page.frames),
                "mainFrameAuthenticatedCandidateCount": sum(
                    1 for candidate in main_candidates if candidate["visible"]
                ),
                "authenticatedMarkerFoundInChildFrame": child_marker_found,
            },
        }

    async def finalize(
        self, final_state: str, result_category: str | None = None
    ) -> dict[str, Any]:
        if self._final_record is not None:
            return self._final_record
        await self.record_pre_auth_ui_state()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        category = result_category or self._result_category(final_state)
        root_cause = self._login_detector_root_cause()
        record = {
            "event": "DEBUG_FINAL",
            "resultCategory": category,
            "failureStage": self._failure_stage(category),
            "rootCauseCategory": root_cause,
            "finalState": final_state,
            "browserOpened": self._browser_opened,
            "kfhDocumentLoaded": self._document_loaded,
            "socketHealth": {
                "trade": self._socket_state[SOCKET_ROLE_TRADE],
                "price": self._socket_state[SOCKET_ROLE_PRICE],
            },
            "counters": {
                "tradeInboundFrameCount": self._trade_inbound_frame_count,
                "priceInboundFrameCount": self._price_inbound_frame_count,
                "unrelatedJsonFrameCount": self._unrelated_json_frame_count,
                "unparsedFrameCount": self._unparsed_frame_count,
            },
            "auth": {
                "responseIdentitySeen": self._auth_response_seen,
                "statusExtracted": self._auth_status_extracted,
                "statusSuccess": self._auth_status_success,
            },
            "ownerVisualLoginMarker": self._owner_visual_marker,
            "uiSamplingStarted": self.ui_sampling_started,
            "uiSampleCount": len(self._samples),
            "preAuthUiState": self._pre_auth_ui_state,
        }
        self._append(record)
        self._final_record = record
        self._finalized = True
        self._page = None
        self._closed_gate3a_session = None
        self._completed.set()
        self._progress(f"DEBUG_FINAL {category}")
        return record

    def _login_detector_root_cause(self) -> str | None:
        if not self._samples:
            if self._auth_status_success:
                return "LOGIN_UI_DETECTOR_CAUSE_NOT_PROVEN"
            return None
        sample = self._samples[-1]
        decision = sample.get("decisionMatrix", {})
        contradictory = (
            bool(decision.get("authProtocolSuccess"))
            and int(decision.get("authenticatedUiSignalCount", 0)) >= 1
            and bool(decision.get("loginUiActive"))
        )
        if not contradictory:
            return None
        signals = sample.get("loginSignals", [])
        if any(bool(signal.get("visible")) for signal in signals):
            return "VISIBLE_LOGIN_UI_REMAINS_AFTER_AUTH"
        if any(bool(signal.get("matched")) for signal in signals):
            return "HIDDEN_LOGIN_DOM_FALSE_POSITIVE"
        if signals:
            return "LOGIN_UI_DETECTOR_CAUSE_NOT_PROVEN"
        return "LOGIN_UI_DETECTOR_FALSE_POSITIVE_OR_STICKY_LOGIN_UI"

    def _result_category(self, final_state: str) -> str:
        if final_state == "READY":
            return "READY"
        if self._debug_failed:
            return "DEBUG_FAILURE"
        if self._auth_status_success:
            if self._samples:
                return "AUTH_SUCCESS_UI_SIGNAL_FAILURE"
            return "AUTH_SUCCESS_UI_DEBUG_STARTED"
        if self._auth_response_seen:
            if not self._auth_status_extracted:
                return "AUTH_STATUS_NOT_EXTRACTED"
            return "AUTH_STATUS_NOT_SUCCESS"
        if self._pre_auth_ui_state and self._pre_auth_ui_state["loginUiActive"] is True:
            return "LOGIN_NOT_COMPLETED"
        if self._socket_state[SOCKET_ROLE_TRADE] == "NOT_OBSERVED":
            return "TRADE_SOCKET_NOT_OBSERVED"
        if self._trade_inbound_frame_count == 0:
            return "NO_TRADE_SOCKET_INBOUND_TRAFFIC"
        return "AUTH_5_101_NOT_OBSERVED"

    @staticmethod
    def _failure_stage(result_category: str) -> str | None:
        if result_category == "TRADE_SOCKET_NOT_OBSERVED":
            return "A"
        if result_category == "NO_TRADE_SOCKET_INBOUND_TRAFFIC":
            return "B"
        if result_category == "AUTH_5_101_NOT_OBSERVED":
            return "C"
        if result_category in {
            "AUTH_STATUS_NOT_EXTRACTED",
            "AUTH_STATUS_NOT_SUCCESS",
        }:
            return "D"
        if result_category in {
            "AUTH_SUCCESS_UI_DEBUG_STARTED",
            "AUTH_SUCCESS_UI_SIGNAL_FAILURE",
        }:
            return "E"
        return None

    def fail_safely(self) -> None:
        self._debug_failed = True
        self._append({"event": "DEBUG_FAILURE", "failed": True})
        self._progress("DEBUG_FAILURE")

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True))
            handle.write("\n")

    @staticmethod
    def _progress(message: str) -> None:
        print(f"GATE5A_DEBUG {message}")

    def summary(self, final_state: str, auth_diagnostics: Any) -> dict[str, Any]:
        final_sample = self._samples[-1] if self._samples else None
        login_detector_root_cause = self._login_detector_root_cause()
        if self._debug_failed or final_sample is None:
            root_cause = login_detector_root_cause or self._result_category(final_state)
            closed_count = None
            candidates: dict[str, bool] = {}
            child_frame = None
            timing: list[dict[str, int | None]] = []
        else:
            closed_count = final_sample["closedGate3A"]["signalCount"]
            candidates = {
                candidate["candidate"]: bool(candidate["visible"])
                for candidate in final_sample["tempCandidates"]
            }
            child_frame = final_sample["pageState"][
                "authenticatedMarkerFoundInChildFrame"
            ]
            timing = [
                {
                    "sampleOffsetSeconds": sample["sampleOffsetSeconds"],
                    "authenticatedUiSignalCount": sample["closedGate3A"][
                        "signalCount"
                    ],
                    "closedLoginUiActive": sample["closedLoginUiActive"],
                    "wouldGate3AAuthenticate": sample["decisionMatrix"][
                        "wouldGate3AAuthenticate"
                    ],
                }
                for sample in self._samples
            ]
            earlier_counts = [
                sample["closedGate3A"]["signalCount"] for sample in self._samples[:-1]
            ]
            if login_detector_root_cause:
                root_cause = login_detector_root_cause
            elif child_frame and closed_count == 0:
                root_cause = "AUTHENTICATED_MARKERS_IN_CHILD_FRAME"
            elif closed_count == 0 and any(candidates.values()):
                root_cause = "STALE_OR_INCORRECT_AUTHENTICATED_UI_SELECTORS"
            elif closed_count > 0 and earlier_counts and not any(earlier_counts):
                root_cause = "DELAYED_KFH_RENDERING"
            elif closed_count > 0:
                root_cause = "CLOSED_GATE3A_SIGNAL_AVAILABLE"
            else:
                root_cause = "NO_SUPPORTED_AUTHENTICATED_UI_MARKER_OBSERVED"

        return {
            "authProtocol": {
                "authResponseSeen": bool(auth_diagnostics.auth_response_seen),
                "authStatusSuccess": bool(auth_diagnostics.auth_status_success),
            },
            "loginUiInactive": auth_diagnostics.login_ui_inactive,
            "closedGate3ASignalCount": closed_count,
            "tempCandidatesVisible": candidates,
            "authenticatedMarkerFoundInChildFrame": child_frame,
            "loginSignals": final_sample["loginSignals"] if final_sample else [],
            "decisionMatrix": (
                final_sample["decisionMatrix"] if final_sample else None
            ),
            "pageState": final_sample["pageState"] if final_sample else None,
            "timing": timing,
            "finalState": final_state,
            "rootCauseCategory": root_cause,
        }
