"""Loopback-only local/dev UI bridge for the Gate 5B read-only test."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import secrets
import sys
from contextlib import suppress
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_core import PydanticCustomError
from starlette.responses import JSONResponse

from local_connector.kfh_gate3a.connector import KfhGate3AConnector
from local_connector.kfh_gate3a.state import KfhAuthState

from .adapter import (
    Gate5BLiveAdapterError,
    Gate5BLiveAuthenticatedContextError,
    Gate5BLiveBridgeFailureError,
    Gate5BLiveCorrelationError,
    Gate5BLiveResponseStatusError,
    Gate5BLiveSessionExpiredError,
    KfhCashStatementSessionAdapter,
)
from .bridge import run_typescript_cash_statement_read
from .browser import Gate5BLiveBrowserRuntime, KfhDiscoveredAccount, KfhLoginAutofillError

LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 8765
LOCAL_SAHAM_ORIGIN = "http://localhost:8081"
ACCOUNT_DISCOVERY_SUPPORTED = True
ACCOUNT_SELECTION_MODE = "DISCOVERED"
NONCE_HEADER = "X-KFH-Local-Nonce"
CONNECT_AND_FETCH_PATH = "/local-test/kfh/connect-and-fetch"
SELECT_ACCOUNT_PATH = "/local-test/kfh/select-account"
PREVIEW_PATH = "/local-test/kfh/preview"
PREVIEW_ACK_PATH = "/local-test/kfh/preview/ack"
PREVIEW_HANDOFF_VERSION = 1
HEALTH_PATH = "/local-test/kfh/health"
OPERATION_STAGES = frozenset(
    {
        "IDLE",
        "OPERATION_ACCEPTED",
        "OPENING_KFH",
        "WAITING_FOR_LOGIN_PAGE",
        "LOGIN_SUBMITTED",
        "AUTHENTICATING",
        "OTP_REQUIRED",
        "AUTHENTICATED",
        "AWAITING_ACCOUNT_SELECTION",
        "FETCHING_STATEMENT",
        "PAGINATING",
        "PREVIEW_READY",
        "FAILED_CLOSED",
        "CANCELLING",
        "DISCONNECTED",
    }
)
BOUNDARY_DECISIONS = frozenset({"NOT_OBSERVED", "ACCEPTED", "REJECTED"})
BODY_STATUSES = frozenset({"NOT_OBSERVED", "PARSED", "REJECTED"})
INVALID_FIELDS = frozenset(
    {
        "REQUEST_SHAPE",
        "USERNAME",
        "PASSWORD",
        "ACCOUNT_HANDLE",
        "FROM_DATE",
        "TO_DATE",
        "CONTENT_TYPE",
        "ORIGIN",
        "NONCE",
    }
)
FAILURE_REASONS = frozenset(
    {
        "MISSING",
        "WRONG_TYPE",
        "INVALID_FORMAT",
        "FORMAT_INVALID",
        "NOT_ALLOWED",
        "OUT_OF_RANGE",
        "UNEXPECTED_FIELD",
    }
)

ORIGIN_FAILURE_CATEGORIES = frozenset(
    {"ORIGIN_FORMAT_INVALID", "ORIGIN_NOT_ALLOWED"}
)


def canonicalize_local_origin(value: str) -> str:
    """Return a strict URL origin with a required explicit port."""

    if not isinstance(value, str) or not value:
        raise ValueError("origin must be a non-empty string")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ValueError("origin is malformed") from error
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or parsed.hostname is None
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("origin must contain only scheme, hostname, and explicit port")
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    return f"{parsed.scheme.lower()}://{host}:{port}"

FAILURE_CODES = frozenset(
    {
        "AUTHENTICATED_HED_CONTEXT_NOT_AVAILABLE",
        "AUTHENTICATED_SOCKET_NOT_AVAILABLE",
        "AUTHENTICATED_SOCKET_AMBIGUOUS",
        "AUTHENTICATED_SOCKET_CLOSED",
        "REQUEST_VALIDATION_FAILED",
        "ORIGIN_REJECTED",
        "NONCE_REJECTED",
        "CONTENT_TYPE_REJECTED",
        "REQUEST_BODY_REJECTED",
        "LOCAL_API_HTTP_ERROR",
        "REQUEST_SEND_FAILED",
        "RESPONSE_TIMEOUT",
        "SESSION_EXPIRED",
        "RESPONSE_PROTOCOL_IDENTITY_FAILED",
        "RESPONSE_CHANNEL_FAILED",
        "RESPONSE_STATUS_FAILED",
        "RESPONSE_CORRELATION_FAILED",
        "RESPONSE_SCHEMA_FAILED",
        "PAGINATION_INTERRUPTED",
        "PAGINATION_PROTOCOL_DRIFT",
        "PAGINATION_TOTAL_DRIFT",
        "PAGINATION_CURSOR_FAILED",
        "PAGINATION_LIMIT_FAILED",
        "LIFECYCLE_CLOSE_FAILED",
        "INTERNAL_BRIDGE_FAILED",
        "UNCLASSIFIED_SAFE_FAILURE",
        "KFH_LOGIN_PAGE_NOT_FOUND",
        "KFH_LOGIN_ORIGIN_REJECTED",
        "KFH_LOGIN_FIELDS_NOT_FOUND",
        "KFH_LOGIN_AUTOFILL_FAILED",
        "KFH_LOGIN_SUBMIT_FAILED",
        "KFH_LOGIN_REJECTED",
        "KFH_LOGIN_TIMEOUT",
        "KFH_OTP_FIELDS_NOT_FOUND",
        "KFH_OTP_AUTOFILL_FAILED",
        "KFH_OTP_SUBMIT_FAILED",
        "INTERACTIVE_VERIFICATION_REQUIRED",
        "OPERATION_ALREADY_ACTIVE",
        "ACCOUNT_DISCOVERY_FAILED",
        "ACCOUNT_SELECTION_REJECTED",
        "KFH_UNAVAILABLE",
        "NETWORK_LOST",
        "USER_CANCELLED",
        "OTP_FAILED",
        "LIVE_READ_FAILED",
    }
)

# The TypeScript reader's own allowlisted KfhConnectorError codes (see
# kfhConnectorErrors.ts / kfh-gate5b-live-bridge.cjs), mapped to this
# module's FAILURE_CODES so a real bridge failure is never collapsed into
# UNCLASSIFIED_SAFE_FAILURE. "UNEXPECTED_MESSAGE" is folded into the
# existing RESPONSE_SCHEMA_FAILED bucket; the bridge's own untyped-error
# fallback code ("LIVE_READ_FAILED") is preserved as its own code so it is
# still distinguishable from a genuinely unclassified Python-side failure.
TS_BRIDGE_FAILURE_CODES: dict[str, tuple[str, str]] = {
    "SESSION_EXPIRED": ("SESSION_EXPIRED", "SESSION"),
    "PAGINATION_INTERRUPTED": ("PAGINATION_INTERRUPTED", "PAGINATION"),
    "PAGINATION_PROTOCOL_DRIFT": ("PAGINATION_PROTOCOL_DRIFT", "PAGINATION"),
    "PAGINATION_TOTAL_DRIFT": ("PAGINATION_TOTAL_DRIFT", "PAGINATION"),
    "PAGINATION_CURSOR_FAILED": ("PAGINATION_CURSOR_FAILED", "PAGINATION"),
    "PAGINATION_LIMIT_FAILED": ("PAGINATION_LIMIT_FAILED", "PAGINATION"),
    "KFH_UNAVAILABLE": ("KFH_UNAVAILABLE", "TRANSPORT"),
    "NETWORK_LOST": ("NETWORK_LOST", "TRANSPORT"),
    "USER_CANCELLED": ("USER_CANCELLED", "USER"),
    "OTP_FAILED": ("OTP_FAILED", "OTP"),
    "UNEXPECTED_MESSAGE": ("RESPONSE_SCHEMA_FAILED", "RESPONSE"),
    "LIVE_READ_FAILED": ("LIVE_READ_FAILED", "INTERNAL"),
}

REDACTED_CREDENTIAL_KEYS = frozenset(
    {
        "username",
        "userName",
        "password",
        "passwd",
        "pwd",
        "otp",
        "pin",
        "verificationCode",
    }
)


def redact_local_credentials(value: Any) -> Any:
    """Defensive redaction for any future local diagnostic boundary."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key in REDACTED_CREDENTIAL_KEYS else redact_local_credentials(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_local_credentials(child) for child in value]
    return value


class LocalConnectAndFetchRequest(BaseModel):
    """One-shot secrets accepted only by the existing local connect operation."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    username: SecretStr = Field(min_length=1, max_length=128)
    password: SecretStr = Field(min_length=1, max_length=256)
    from_date: str = Field(alias="fromDate")
    to_date: str = Field(alias="toDate")

    @field_validator("from_date", "to_date")
    @classmethod
    def validate_protocol_date(cls, value: str) -> str:
        return LocalReadRequest.validate_protocol_date(value)

    @model_validator(mode="after")
    def validate_date_range(self) -> LocalConnectAndFetchRequest:
        if self.from_date > self.to_date:
            raise PydanticCustomError(
                "date_range", "date range reversed", {"field": "fromDate"}
            )
        return self


class LocalOtpRequest(BaseModel):
    """One transient KFH verification code; never retained or persisted."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    verification_code: SecretStr = Field(
        alias="verificationCode", min_length=1, max_length=16
    )


class LocalAccountSelectionRequest(BaseModel):
    """The owner's chosen KFH account, by opaque handle only - never the
    real secAccNum, which this local API never accepts from the client."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    handle: str = Field(min_length=1, max_length=64, strict=True)


class LocalReadRequest(BaseModel):
    """The complete and only app-supplied read shape."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    security_account: str = Field(alias="securityAccount", max_length=128)
    from_date: str = Field(alias="fromDate")
    to_date: str = Field(alias="toDate")

    @field_validator("from_date", "to_date")
    @classmethod
    def validate_protocol_date(cls, value: str) -> str:
        if not re.fullmatch(r"\d{8}", value):
            raise ValueError("invalid date")
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError as error:
            raise ValueError("invalid date") from error
        return value


def _base_diagnostics() -> dict[str, Any]:
    return {
        "connection": "DISCONNECTED",
        "result": "IDLE",
        "failure": None,
        "stage": "IDLE",
        "operationAccepted": False,
        "operationActive": False,
        "operationStage": "IDLE",
        "startAckReceived": False,
        "failedStage": None,
        "invalidField": None,
        "failureReason": None,
        "clientRequestStarted": False,
        "localApiConnectionEstablished": False,
        "httpResponseObserved": False,
        "httpStatusClass": "NONE",
        "originStatus": "NOT_OBSERVED",
        "originFailureCategory": None,
        "clientOrigin": None,
        "serverAllowedOrigin": None,
        "serverAllowedOriginConfigured": None,
        "nonceStatus": "NOT_OBSERVED",
        "contentTypeStatus": "NOT_OBSERVED",
        "bodyStatus": "NOT_OBSERVED",
        "requestValidated": False,
        "clientOriginMatch": None,
        "clientNoncePresent": None,
        "serverNonceConfigured": True,
        "localApiProcess": "AVAILABLE",
        "usernamePresent": False,
        "passwordPresent": False,
        "securityAccountPresent": False,
        "fromDateValid": False,
        "toDateValid": False,
        "loginOrchestrationStarted": False,
        "usernameAutofillConfirmed": False,
        "passwordAutofillConfirmed": False,
        "loginSubmitTriggered": False,
        "otpSubmitTriggered": False,
        "visibleKfhBrowser": False,
        "accountDiscoverySupported": ACCOUNT_DISCOVERY_SUPPORTED,
        "accountSelectionMode": ACCOUNT_SELECTION_MODE,
        "gate3aReady": False,
        "authenticatedHed": False,
        "authenticatedSocket": False,
        "requestsSent": 0,
        "responsesSeen": 0,
        "pagesCompleted": 0,
        "summaryConsistency": None,
        "summaryVariantFields": [],
        "financialWrites": 0,
        "sqliteWrites": 0,
        "postgresqlWrites": 0,
    }


_FIELD_DIAGNOSTICS = {
    "username": ("USERNAME", "usernamePresent"),
    "password": ("PASSWORD", "passwordPresent"),
    "handle": ("ACCOUNT_HANDLE", None),
    "fromDate": ("FROM_DATE", "fromDateValid"),
    "toDate": ("TO_DATE", "toDateValid"),
}


def _validation_failure(
    invalid_field: str,
    reason: str,
    *,
    failure_code: str = "REQUEST_VALIDATION_FAILED",
    origin_status: str = "NOT_OBSERVED",
    nonce_status: str = "NOT_OBSERVED",
    content_type_status: str = "NOT_OBSERVED",
    body_status: str = "NOT_OBSERVED",
    failed_stage: str = "REQUEST_VALIDATION",
    origin_failure_category: str | None = None,
    server_allowed_origin: str | None = None,
    server_allowed_origin_configured: bool | None = None,
) -> dict[str, Any]:
    field = invalid_field if invalid_field in INVALID_FIELDS else "REQUEST_SHAPE"
    safe_reason = reason if reason in FAILURE_REASONS else "INVALID_FORMAT"
    safe_failure = failure_code if failure_code in FAILURE_CODES else "REQUEST_VALIDATION_FAILED"
    safe_origin = origin_status if origin_status in BOUNDARY_DECISIONS else "NOT_OBSERVED"
    safe_nonce = nonce_status if nonce_status in BOUNDARY_DECISIONS else "NOT_OBSERVED"
    safe_content_type = (
        content_type_status
        if content_type_status in BOUNDARY_DECISIONS
        else "NOT_OBSERVED"
    )
    safe_body = body_status if body_status in BODY_STATUSES else "NOT_OBSERVED"
    safe_origin_category = (
        origin_failure_category
        if origin_failure_category in ORIGIN_FAILURE_CATEGORIES
        else None
    )
    payload = _base_diagnostics()
    payload.update(
        {
            "result": "FAILED_CLOSED",
            "failure": safe_failure,
            "stage": "REQUEST_BOUNDARY",
            "operationStage": "FAILED_CLOSED",
            "failedStage": failed_stage,
            "invalidField": field,
            "failureReason": safe_reason,
            "clientRequestStarted": True,
            "localApiConnectionEstablished": True,
            "originStatus": safe_origin,
            "originFailureCategory": safe_origin_category,
            "serverAllowedOrigin": server_allowed_origin,
            "serverAllowedOriginConfigured": server_allowed_origin_configured,
            "nonceStatus": safe_nonce,
            "contentTypeStatus": safe_content_type,
            "bodyStatus": safe_body,
        }
    )
    return payload


def _model_validation_failure(errors: list[dict[str, Any]]) -> dict[str, Any]:
    malformed_json = any(str(error.get("type", "")) == "json_invalid" for error in errors)
    payload = _validation_failure(
        "REQUEST_SHAPE",
        "INVALID_FORMAT",
        failure_code=(
            "REQUEST_BODY_REJECTED" if malformed_json else "REQUEST_VALIDATION_FAILED"
        ),
        origin_status="ACCEPTED",
        nonce_status="ACCEPTED",
        content_type_status="ACCEPTED",
        body_status="REJECTED" if malformed_json else "PARSED",
        failed_stage="REQUEST_BODY" if malformed_json else "REQUEST_VALIDATION",
    )
    if malformed_json:
        return payload
    payload.update(
        {
            "usernamePresent": True,
            "passwordPresent": True,
            "securityAccountPresent": True,
            "fromDateValid": True,
            "toDateValid": True,
        }
    )
    first_field = "REQUEST_SHAPE"
    first_reason = "INVALID_FORMAT"
    for index, error in enumerate(errors):
        error_type = str(error.get("type", ""))
        location = error.get("loc", ())
        leaf = str(location[-1]) if location else ""
        if error_type == "date_range":
            field, flag = "FROM_DATE", "fromDateValid"
            reason = "OUT_OF_RANGE"
        elif error_type == "extra_forbidden":
            field, flag, reason = "REQUEST_SHAPE", None, "UNEXPECTED_FIELD"
        else:
            field, flag = _FIELD_DIAGNOSTICS.get(leaf, ("REQUEST_SHAPE", None))
            if error_type == "missing":
                reason = "MISSING"
            elif error_type.endswith("_type"):
                reason = "WRONG_TYPE"
            elif error_type in {"string_too_long", "string_too_short"}:
                reason = "OUT_OF_RANGE"
            else:
                reason = "INVALID_FORMAT"
        if flag is not None:
            payload[flag] = False
        if index == 0:
            first_field, first_reason = field, reason
    payload["invalidField"] = first_field
    payload["failureReason"] = first_reason
    return payload

class KfhLocalTestController:
    """One ephemeral browser and one read at a time; no persistence dependency."""

    AUTHENTICATED_HED_WAIT_SECONDS = 15

    def __init__(self) -> None:
        self._runtime: Gate5BLiveBrowserRuntime | None = None
        self._connector: KfhGate3AConnector | None = None
        self._adapter: KfhCashStatementSessionAdapter | None = None
        self._operation_task: asyncio.Task[None] | None = None
        self._operation_lock = asyncio.Lock()
        self._diagnostics = _base_diagnostics()
        self._terminal_connection: str | None = None
        self._pending_statement_request: LocalReadRequest | None = None
        # Ephemeral, single-use: opaque handle -> the owner's real secAccNum.
        # Never sent to the client; cleared once selection resolves.
        self._account_handles: dict[str, tuple[str, str]] = {}
        # Minimized statement fields only. Retained in process memory until
        # Saham acknowledges successful backend staging, a new operation
        # begins, or the connector closes. Never included in diagnostics.
        self._pending_preview: dict[str, Any] | None = None

    def _increment_request(self) -> None:
        self._diagnostics["requestsSent"] += 1

    def _increment_response(self) -> None:
        self._diagnostics["responsesSeen"] += 1
        self._diagnostics["pagesCompleted"] += 1
        self._set_operation_stage("PAGINATING")

    def _set_operation_stage(self, stage: str) -> None:
        self._diagnostics["stage"] = stage
        self._diagnostics["operationStage"] = (
            stage if stage in OPERATION_STAGES else "FAILED_CLOSED"
        )

    def _operation_is_active(self) -> bool:
        if self._diagnostics.get("operationStage") == "AWAITING_ACCOUNT_SELECTION":
            return True
        task = self._operation_task
        return task is not None and not task.done()

    async def connect(self, credentials: LocalConnectAndFetchRequest) -> dict[str, Any]:
        async with self._operation_lock:
            username = credentials.username.get_secret_value()
            password = credentials.password.get_secret_value()
            statement_request = LocalReadRequest(
                securityAccount="",
                fromDate=credentials.from_date,
                toDate=credentials.to_date,
            )
            try:
                if self._operation_is_active():
                    rejected = _base_diagnostics()
                    rejected.update(
                        {
                            "connection": self._diagnostics["connection"],
                            "result": "FAILED_CLOSED",
                            "failure": "OPERATION_ALREADY_ACTIVE",
                            "stage": "OPERATION_ALREADY_ACTIVE",
                            "failedStage": "OPERATION_START",
                            "operationAccepted": False,
                            "operationActive": True,
                            "operationStage": self._diagnostics["operationStage"],
                            "clientRequestStarted": True,
                            "localApiConnectionEstablished": True,
                            "originStatus": "ACCEPTED",
                            "nonceStatus": "ACCEPTED",
                            "contentTypeStatus": "ACCEPTED",
                            "bodyStatus": "PARSED",
                            "requestValidated": True,
                        }
                    )
                    return rejected
                self._diagnostics = _base_diagnostics()
                self._pending_preview = None
                self._diagnostics.update(
                    {
                        "connection": "AUTHENTICATING",
                        "result": "READING",
                        "stage": "OPERATION_ACCEPTED",
                        "operationAccepted": True,
                        "operationActive": True,
                        "operationStage": "OPERATION_ACCEPTED",
                        "clientRequestStarted": True,
                        "localApiConnectionEstablished": True,
                        "originStatus": "ACCEPTED",
                        "nonceStatus": "ACCEPTED",
                        "contentTypeStatus": "ACCEPTED",
                        "bodyStatus": "PARSED",
                        "requestValidated": True,
                        "usernamePresent": True,
                        "passwordPresent": True,
                        "securityAccountPresent": False,
                        "fromDateValid": True,
                        "toDateValid": True,
                    }
                )
                self._terminal_connection = None
                self._operation_task = asyncio.create_task(
                    self._run_connect_and_fetch(username, password, statement_request),
                    name="kfh-connect-and-fetch",
                )
                return dict(self._diagnostics)
            finally:
                credentials.username = SecretStr("")
                credentials.password = SecretStr("")

    async def _run_connect_and_fetch(
        self,
        username: str,
        password: str,
        statement_request: LocalReadRequest,
    ) -> None:
        current_task = asyncio.current_task()
        try:
            await self._close_resources_unlocked()

            def statement_response(frame: str | bytes) -> None:
                adapter = self._adapter
                if adapter is not None:
                    adapter._observe_statement_response(frame)

            runtime = Gate5BLiveBrowserRuntime(
                on_statement_response_frame=statement_response
            )
            connector = KfhGate3AConnector(runtime)
            self._runtime = runtime
            self._connector = connector
            self._diagnostics["visibleKfhBrowser"] = runtime._visible_browser_enabled()
            self._adapter = KfhCashStatementSessionAdapter(
                runtime,
                ready=lambda: connector.status().state == KfhAuthState.READY,
                authenticated_context_status=runtime._authenticated_context_status,
                on_request_sent=self._increment_request,
                on_response_accepted=self._increment_response,
                timeout_seconds=45,
            )
            self._set_operation_stage("OPENING_KFH")
            self._diagnostics["loginOrchestrationStarted"] = True
            await connector.connect()
            self._set_operation_stage("WAITING_FOR_LOGIN_PAGE")
            try:
                confirmation = await runtime._submit_login_credentials(username, password)
            finally:
                username = ""
                password = ""
            self._diagnostics.update(confirmation)
            self._set_operation_stage("LOGIN_SUBMITTED")
            await self._authenticate_and_fetch(connector, runtime, statement_request)
        except asyncio.CancelledError:
            raise
        except KfhLoginAutofillError as error:
            self._fail(error.code, "LOGIN")
            await self._close_resources_unlocked()
        except Exception:
            self._fail("KFH_LOGIN_SUBMIT_FAILED", "LOGIN")
            await self._close_resources_unlocked()
        finally:
            username = ""
            password = ""
            statement_request.security_account = ""
            if self._operation_task is current_task:
                self._diagnostics["operationActive"] = False

    async def _authenticate_and_fetch(
        self,
        connector: KfhGate3AConnector,
        runtime: Gate5BLiveBrowserRuntime,
        statement_request: LocalReadRequest,
    ) -> None:
        authenticated = False
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 300
            while loop.time() < deadline and connector is self._connector:
                if await runtime._interactive_challenge_present():
                    self._terminal_connection = "INTERACTIVE_VERIFICATION_REQUIRED"
                    self._fail(
                        "INTERACTIVE_VERIFICATION_REQUIRED", "INTERACTIVE_VERIFICATION"
                    )
                    return
                snapshot = connector.status()
                if snapshot.state == KfhAuthState.READY:
                    authenticated = True
                    break
                if snapshot.state == KfhAuthState.AUTH_FAILED:
                    self._fail("KFH_LOGIN_REJECTED", "LOGIN")
                    return
                if snapshot.state == KfhAuthState.KFH_UNAVAILABLE:
                    self._fail("KFH_LOGIN_PAGE_NOT_FOUND", "LOGIN")
                    return
                if snapshot.state in {
                    KfhAuthState.SESSION_EXPIRED,
                    KfhAuthState.BROWSER_CLOSED,
                    KfhAuthState.NETWORK_ERROR,
                    KfhAuthState.CONNECTOR_ERROR,
                }:
                    self._fail("KFH_LOGIN_SUBMIT_FAILED", "LOGIN")
                    return
                await asyncio.sleep(0.25)
            if not authenticated:
                self._fail("KFH_LOGIN_TIMEOUT", "LOGIN")
                return
            with suppress(Exception):
                await runtime._mark_gate3a_ready()
            self._set_operation_stage("AUTHENTICATED")
        finally:
            with suppress(Exception):
                await runtime._clear_login_dom_credentials()
            with suppress(Exception):
                await runtime._clear_otp_dom_credentials()
            if not authenticated:
                statement_request.security_account = ""

        if connector is not self._connector:
            return
        await self._await_account_selection(runtime, statement_request)

    async def _await_account_selection(
        self, runtime: Gate5BLiveBrowserRuntime, statement_request: LocalReadRequest
    ) -> None:
        """Wait for KFH's own post-login account list, then pause the
        operation for the owner to choose one. Never guesses an account."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 30
        accounts: list[KfhDiscoveredAccount] | None = None
        while loop.time() < deadline and self._connector is not None:
            accounts = await runtime._discovered_accounts()
            if accounts:
                break
            await asyncio.sleep(0.5)
        if not accounts:
            self._account_handles = {}
            self._fail("ACCOUNT_DISCOVERY_FAILED", "ACCOUNT_DISCOVERY")
            return
        # Saham never receives the owner's real secAccNum/portNme - only an
        # opaque per-operation handle and the safe fields (currency, the
        # genuine KFH default flag) needed to choose between accounts.
        handles: dict[str, str] = {}
        options: list[dict[str, Any]] = []
        for account in accounts:
            handle = secrets.token_urlsafe(9)
            handles[handle] = (account["secAccNum"], account["curr"])
            options.append(
                {
                    "handle": handle,
                    "curr": account["curr"],
                    "isDefaultAccount": account["isDefaultAccount"],
                }
            )
        self._account_handles = handles
        self._pending_statement_request = statement_request
        self._diagnostics["availableAccounts"] = options
        self._diagnostics["result"] = "READING"
        self._set_operation_stage("AWAITING_ACCOUNT_SELECTION")

    async def select_account(self, request: LocalAccountSelectionRequest) -> dict[str, Any]:
        async with self._operation_lock:
            pending = self._pending_statement_request
            handles = self._account_handles
            if (
                self._connector is None
                or self._diagnostics.get("operationStage") != "AWAITING_ACCOUNT_SELECTION"
                or pending is None
                or not handles
            ):
                return self._fail("ACCOUNT_SELECTION_REJECTED", "ACCOUNT_SELECTION")
            chosen_account = handles.get(request.handle)
            if chosen_account is None:
                return self._fail("ACCOUNT_SELECTION_REJECTED", "ACCOUNT_SELECTION")
            security_account, account_currency = chosen_account
            pending.security_account = security_account
            account_token = hashlib.sha256(
                security_account.strip().upper().encode("utf-8")
            ).hexdigest()
            self._pending_statement_request = None
            self._account_handles = {}
            self._diagnostics["securityAccountPresent"] = True
            self._diagnostics["result"] = "READING"
            self._set_operation_stage("FETCHING_STATEMENT")
            self._operation_task = asyncio.create_task(
                self._select_account_and_fetch(
                    pending, account_currency, account_token
                ),
                name="kfh-select-account-and-fetch",
            )
            return dict(self._diagnostics)

    async def _select_account_and_fetch(
        self,
        statement_request: LocalReadRequest,
        account_currency: str,
        account_token: str,
    ) -> None:
        current_task = asyncio.current_task()
        try:
            await self._read_statement(
                statement_request, account_currency, account_token
            )
        finally:
            statement_request.security_account = ""
            account_token = ""
            if self._operation_task is current_task:
                self._diagnostics["operationActive"] = False

    async def submit_otp(self, request: LocalOtpRequest) -> dict[str, Any]:
        async with self._operation_lock:
            verification_code = request.verification_code.get_secret_value()
            try:
                connector = self._connector
                runtime = self._runtime
                if (
                    connector is None
                    or runtime is None
                    or connector.status().state != KfhAuthState.OTP_REQUIRED
                ):
                    return self._fail("KFH_OTP_SUBMIT_FAILED", "OTP")
                confirmation = await runtime._submit_otp(verification_code)
                self._diagnostics.update(confirmation)
                self._set_operation_stage("AUTHENTICATING")
                return await self.status()
            except KfhLoginAutofillError as error:
                return self._fail(error.code, "OTP")
            except Exception:
                return self._fail("KFH_OTP_SUBMIT_FAILED", "OTP")
            finally:
                verification_code = ""
                request.verification_code = SecretStr("")

    @staticmethod
    def _display_connection(state: KfhAuthState) -> str:
        if state == KfhAuthState.READY:
            return "READY"
        if state == KfhAuthState.LOGIN_REQUIRED:
            return "LOGIN_REQUIRED"
        if state == KfhAuthState.OTP_REQUIRED:
            return "OTP_REQUIRED"
        if state in {
            KfhAuthState.OPENING_KFH,
            KfhAuthState.AUTHENTICATING,
            KfhAuthState.AUTHENTICATED,
        }:
            return "AUTHENTICATING"
        return "DISCONNECTED"

    async def status(self) -> dict[str, Any]:
        connector = self._connector
        state = connector.status().state if connector else KfhAuthState.DISCONNECTED
        connection = self._display_connection(state)
        if self._terminal_connection is not None:
            connection = self._terminal_connection
        if connection == "LOGIN_REQUIRED" and self._diagnostics["loginSubmitTriggered"] is True:
            connection = "AUTHENTICATING"
        self._diagnostics.update(
            {
                "connection": connection,
                "gate3aReady": state == KfhAuthState.READY,
                "operationActive": self._operation_is_active(),
            }
        )
        if self._diagnostics["result"] != "FAILED_CLOSED":
            if state == KfhAuthState.OTP_REQUIRED:
                self._set_operation_stage("OTP_REQUIRED")
            elif state == KfhAuthState.READY and self._diagnostics["result"] not in {
                "READING",
                "PASS",
            }:
                self._set_operation_stage("AUTHENTICATED")
            elif (
                self._diagnostics["loginSubmitTriggered"] is True
                and self._diagnostics["result"] not in {"READING", "PASS"}
            ):
                self._set_operation_stage("AUTHENTICATING")
        return dict(self._diagnostics)

    def _fail(self, code: str, stage: str) -> dict[str, Any]:
        safe_code = code if code in FAILURE_CODES else "UNCLASSIFIED_SAFE_FAILURE"
        self._diagnostics.update(
            {
                "result": "FAILED_CLOSED",
                "failure": safe_code,
                "stage": stage,
                "failedStage": stage,
                "operationStage": "FAILED_CLOSED",
            }
        )
        return dict(self._diagnostics)

    @staticmethod
    def _classify(error: Exception) -> tuple[str, str]:
        if isinstance(error, Gate5BLiveAuthenticatedContextError):
            if "AMBIGUOUS" in str(error):
                return "AUTHENTICATED_SOCKET_AMBIGUOUS", "AUTHENTICATED_CONTEXT"
            return "AUTHENTICATED_HED_CONTEXT_NOT_AVAILABLE", "AUTHENTICATED_CONTEXT"
        if isinstance(error, Gate5BLiveSessionExpiredError):
            return "SESSION_EXPIRED", "SESSION"
        if isinstance(error, Gate5BLiveCorrelationError):
            return "RESPONSE_CORRELATION_FAILED", "RESPONSE_CORRELATION"
        if isinstance(error, Gate5BLiveResponseStatusError):
            return error.code, "RESPONSE_HEADER"
        if isinstance(error, Gate5BLiveBridgeFailureError):
            return TS_BRIDGE_FAILURE_CODES.get(
                error.ts_code, ("LIVE_READ_FAILED", "INTERNAL")
            )
        if isinstance(error, Gate5BLiveAdapterError):
            controlled = str(error)
            if "timed out" in controlled:
                return "RESPONSE_TIMEOUT", "RESPONSE"
            if "transport rejected" in controlled:
                return "REQUEST_SEND_FAILED", "REQUEST"
            if "bridge" in controlled:
                return "INTERNAL_BRIDGE_FAILED", "BRIDGE"
        return "UNCLASSIFIED_SAFE_FAILURE", "INTERNAL"

    async def _read_statement(
        self,
        request: LocalReadRequest,
        account_currency: str,
        account_token: str,
    ) -> dict[str, Any]:
        connector = self._connector
        runtime = self._runtime
        adapter = self._adapter
        if connector is None or runtime is None or adapter is None:
            request.security_account = ""
            return self._fail("SESSION_EXPIRED", "SESSION")
        if connector.status().state != KfhAuthState.READY:
            request.security_account = ""
            return self._fail("SESSION_EXPIRED", "SESSION")

        # The authenticated HED context is populated passively from KFH's own
        # later /wstrs traffic (see docs/kfh-gate5b-l1-r1-preparation-record.md,
        # "Authenticated HED acquisition method") - it is not guaranteed to be
        # available the instant READY is reached. Give it a brief window to
        # arrive naturally before treating it as unavailable.
        context = await runtime._authenticated_context_diagnostics()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.AUTHENTICATED_HED_WAIT_SECONDS
        while (
            context["status"] == "NOT_AVAILABLE"
            and loop.time() < deadline
            and connector.status().state == KfhAuthState.READY
        ):
            await asyncio.sleep(0.25)
            context = await runtime._authenticated_context_diagnostics()
        self._diagnostics.update(
            {
                "connection": "READY",
                "gate3aReady": True,
                "authenticatedHed": context["authenticatedHed"] is True,
                "authenticatedSocket": context["authenticatedSocket"] is True,
            }
        )
        if context["status"] == "AMBIGUOUS":
            request.security_account = ""
            return self._fail(
                "AUTHENTICATED_SOCKET_AMBIGUOUS", "AUTHENTICATED_CONTEXT"
            )
        if context["status"] == "CLOSED":
            request.security_account = ""
            return self._fail(
                "AUTHENTICATED_SOCKET_CLOSED", "AUTHENTICATED_CONTEXT"
            )
        if context["authenticatedHed"] is not True:
            code = (
                "AUTHENTICATED_HED_CONTEXT_NOT_AVAILABLE"
                if context["authenticatedSocket"] is True
                else "AUTHENTICATED_SOCKET_NOT_AVAILABLE"
            )
            request.security_account = ""
            return self._fail(code, "AUTHENTICATED_CONTEXT")

        self._diagnostics.update(
            {
                "result": "READING",
                "failure": None,
                "requestsSent": 0,
                "responsesSeen": 0,
                "pagesCompleted": 0,
            }
        )
        self._set_operation_stage("FETCHING_STATEMENT")
        security_account = request.security_account
        try:
            live = await run_typescript_cash_statement_read(
                adapter,
                {
                    "secAccNum": security_account,
                    "frmDate": request.from_date,
                    "toDate": request.to_date,
                    "sortMode": 0,
                    "startSeq": 0,
                    "totalNoRec": 20,
                },
                account_currency=account_currency,
            )
            if live.get("financialWritesPerformed") != 0:
                return self._fail("UNCLASSIFIED_SAFE_FAILURE", "ZERO_WRITE_GUARD")
            self._diagnostics.update(
                {
                    "result": "PASS",
                    "failure": None,
                    "requestStarts": list(live["requestStartSeqProgression"]),
                    "cashLogCounts": list(live["responseCashLogsCounts"]),
                    "continuation": list(live["isNxtPagAvailSequence"]),
                    "pagesCompleted": len(live["requestStartSeqProgression"]),
                    "finalResponse": live["finalResponseObserved"] is True,
                    "summaryConsistency": (
                        live.get("summaryConsistency")
                        if live.get("summaryConsistency") in {"STABLE", "PAGE_VARIANT"}
                        else None
                    ),
                    "summaryVariantFields": [
                        field
                        for field in live.get("summaryVariantFields", [])
                        if field
                        in {
                            "openBal",
                            "closeBal",
                            "totDeposit",
                            "totWithdrawal",
                            "totBuy",
                            "totSell",
                            "totOther",
                            "vatAmount",
                        }
                    ],
                    "financialWrites": 0,
                    "sqliteWrites": 0,
                    "postgresqlWrites": 0,
                    # TEMPORARY diagnostic only (Gate 6A field-mapping
                    # evidence): field NAMES only, never values. Remove
                    # once the live schema decoder is proven.
                    "firstCashLogFieldNames": live.get("firstCashLogFieldNames"),
                    "firstUnsettledCashLogFieldNames": live.get(
                        "firstUnsettledCashLogFieldNames"
                    ),
                }
            )
            preview_payload = live.get("previewPayload")
            if isinstance(preview_payload, dict):
                self._pending_preview = {
                    **preview_payload,
                    # Stable per KFH account but opaque outside this process.
                    # The backend hashes it again and scopes identity by user.
                    "brokerAccount": f"KFH-LOCAL-{account_token}",
                }
            self._set_operation_stage("PREVIEW_READY")
            return dict(self._diagnostics)
        except Exception as error:
            self._pending_preview = None
            # TEMPORARY diagnostic only: server console, never sent to the
            # client. Only the exception class and, for our own structured
            # bridge failures, the pre-vetted static/field-name-based code
            # and detail are printed - never a raw exception message, which
            # could otherwise carry account, payload, or balance data from
            # an unclassified library exception. Remove once pagination is
            # proven stable across multiple pages.
            if isinstance(error, Gate5BLiveBridgeFailureError):
                evidence = error.evidence
                request_starts = evidence.get("requestStartSeqProgression", [])
                cash_log_counts = evidence.get("responseCashLogsCounts", [])
                continuation = evidence.get("isNxtPagAvailSequence", [])
                self._diagnostics.update(
                    {
                        "requestStarts": list(request_starts),
                        "cashLogCounts": list(cash_log_counts),
                        "continuation": list(continuation),
                        "requestsSent": len(request_starts),
                        "responsesSeen": len(cash_log_counts),
                        "pagesCompleted": len(cash_log_counts),
                        "finalResponse": (
                            evidence.get("finalResponseObserved") is True
                        ),
                        "financialWrites": 0,
                        "sqliteWrites": 0,
                        "postgresqlWrites": 0,
                    }
                )
                print(
                    f"[kfh-diagnostic] _read_statement failed: "
                    f"ts_code={error.ts_code!r} detail={error.detail!r}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[kfh-diagnostic] _read_statement failed: {type(error).__name__}",
                    file=sys.stderr,
                )
            code, stage = self._classify(error)
            return self._fail(code, stage)
        finally:
            security_account = ""
            request.security_account = ""

    async def take_preview(self) -> dict[str, Any] | None:
        """Read the transient statement cache; never expose it via status."""
        async with self._operation_lock:
            if (
                self._diagnostics.get("result") != "PASS"
                or self._diagnostics.get("operationStage") != "PREVIEW_READY"
                or self._pending_preview is None
            ):
                return None
            return dict(self._pending_preview)

    async def acknowledge_preview(self) -> dict[str, str]:
        """Clear cached rows only after Saham has staged its review batch."""
        async with self._operation_lock:
            self._pending_preview = None
            return {"status": "OK"}

    async def close(self) -> dict[str, Any]:
        async with self._operation_lock:
            self._set_operation_stage("CANCELLING")
            failed = not await self._close_unlocked()
            self._pending_statement_request = None
            self._account_handles = {}
            self._pending_preview = None
            self._diagnostics = _base_diagnostics()
            self._terminal_connection = None
            self._set_operation_stage("DISCONNECTED")
            if failed:
                self._diagnostics.update(
                    {
                        "result": "FAILED_CLOSED",
                        "failure": "LIFECYCLE_CLOSE_FAILED",
                        "stage": "CLOSE",
                    }
                )
            return {**self._diagnostics, "connection": "DISCONNECTED"}

    async def _close_unlocked(self) -> bool:
        task = self._operation_task
        self._operation_task = None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        return await self._close_resources_unlocked()

    async def _close_resources_unlocked(self) -> bool:
        success = True
        adapter = self._adapter
        connector = self._connector
        runtime = self._runtime
        self._adapter = None
        self._connector = None
        self._runtime = None
        if runtime is not None:
            try:
                await runtime._clear_login_dom_credentials()
                await runtime._clear_otp_dom_credentials()
            except Exception:
                success = False
        if adapter is not None:
            try:
                await adapter.close()
            except Exception:
                success = False
        if connector is not None:
            try:
                await connector.logout()
            except Exception:
                success = False
        return success


def create_local_test_app(
    *,
    controller: KfhLocalTestController | Any | None = None,
    nonce: str,
    allowed_origin: str = LOCAL_SAHAM_ORIGIN,
    allowed_origin_configured: bool = False,
) -> FastAPI:
    if len(nonce) < 16:
        raise ValueError("KFH local-test nonce must contain at least 16 characters")
    canonical_allowed_origin = canonicalize_local_origin(allowed_origin)
    local_controller = controller or KfhLocalTestController()
    app = FastAPI(
        title="Saham local KFH read-only test",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.request_body_logging = False
    app.state.allowed_origin = canonical_allowed_origin
    app.state.allowed_origin_configured = allowed_origin_configured
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[canonical_allowed_origin],
        allow_credentials=False,
        allow_private_network=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", NONCE_HEADER],
    )

    @app.exception_handler(RequestValidationError)
    async def sanitized_validation_failure(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        payload = _model_validation_failure(error.errors())
        return JSONResponse(status_code=422, content=payload)

    @app.exception_handler(HTTPException)
    async def sanitized_http_failure(
        _request: Request, error: HTTPException
    ) -> JSONResponse:
        if error.detail == "LOCAL_NONCE_MISSING":
            payload = _validation_failure(
                "NONCE",
                "MISSING",
                failure_code="NONCE_REJECTED",
                origin_status="ACCEPTED",
                nonce_status="REJECTED",
                failed_stage="NONCE",
            )
        elif error.detail == "LOCAL_NONCE_REJECTED":
            payload = _validation_failure(
                "NONCE",
                "INVALID_FORMAT",
                failure_code="NONCE_REJECTED",
                origin_status="ACCEPTED",
                nonce_status="REJECTED",
                failed_stage="NONCE",
            )
        else:
            payload = _validation_failure(
                "REQUEST_SHAPE", "INVALID_FORMAT", failed_stage="REQUEST_VALIDATION"
            )
        return JSONResponse(status_code=error.status_code, content=payload)

    @app.middleware("http")
    async def hardened_local_browser_boundary(request: Request, call_next: Any) -> Any:
        def no_store(response: Any) -> Any:
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
            return response

        host = request.client.host if request.client else ""
        if host not in {"127.0.0.1", "::1"}:
            return no_store(
                JSONResponse(
                    status_code=403,
                    content=_validation_failure(
                        "ORIGIN",
                        "INVALID_FORMAT",
                        failure_code="ORIGIN_REJECTED",
                        failed_stage="LOCAL_API_CONNECTION",
                    ),
                )
            )

        # Origin is evaluated before nonce, content type, or request-body parsing.
        supplied_origin = request.headers.get("origin")
        if supplied_origin is None:
            return no_store(
                JSONResponse(
                    status_code=403,
                    content=_validation_failure(
                        "ORIGIN",
                        "MISSING",
                        failure_code="ORIGIN_REJECTED",
                        origin_status="REJECTED",
                        failed_stage="ORIGIN",
                        server_allowed_origin=canonical_allowed_origin,
                        server_allowed_origin_configured=allowed_origin_configured,
                    ),
                )
            )
        try:
            canonical_supplied_origin = canonicalize_local_origin(supplied_origin)
        except ValueError:
            return no_store(
                JSONResponse(
                    status_code=403,
                    content=_validation_failure(
                        "ORIGIN",
                        "FORMAT_INVALID",
                        failure_code="ORIGIN_REJECTED",
                        origin_status="REJECTED",
                        failed_stage="ORIGIN_FORMAT",
                        origin_failure_category="ORIGIN_FORMAT_INVALID",
                        server_allowed_origin=canonical_allowed_origin,
                        server_allowed_origin_configured=allowed_origin_configured,
                    ),
                )
            )
        if canonical_supplied_origin != canonical_allowed_origin:
            return no_store(
                JSONResponse(
                    status_code=403,
                    content=_validation_failure(
                        "ORIGIN",
                        "NOT_ALLOWED",
                        failure_code="ORIGIN_REJECTED",
                        origin_status="REJECTED",
                        failed_stage="ORIGIN_ALLOWLIST",
                        origin_failure_category="ORIGIN_NOT_ALLOWED",
                        server_allowed_origin=canonical_allowed_origin,
                        server_allowed_origin_configured=allowed_origin_configured,
                    ),
                )
            )

        credential_names = {
            "username",
            "password",
            "passwd",
            "pwd",
            "otp",
            "pin",
            "verificationcode",
        }
        if any(key.lower() in credential_names for key in request.query_params):
            return no_store(
                JSONResponse(
                    status_code=400,
                    content=_validation_failure(
                        "REQUEST_SHAPE",
                        "UNEXPECTED_FIELD",
                        failure_code="REQUEST_BODY_REJECTED",
                        origin_status="ACCEPTED",
                        failed_stage="REQUEST_BODY",
                    ),
                )
            )
        if any(
            key.lower() in credential_names
            or any(key.lower().endswith(f"-{name}") for name in credential_names)
            for key in request.headers
        ):
            return no_store(
                JSONResponse(
                    status_code=400,
                    content=_validation_failure(
                        "REQUEST_SHAPE",
                        "UNEXPECTED_FIELD",
                        failure_code="REQUEST_BODY_REJECTED",
                        origin_status="ACCEPTED",
                        failed_stage="REQUEST_BODY",
                    ),
                )
            )

        # Browser preflight is authorized only by the exact Origin/CORS allowlist.
        if request.method == "OPTIONS":
            return no_store(await call_next(request))

        if request.url.path.startswith("/local-test/kfh/"):
            supplied_nonce = request.headers.get(NONCE_HEADER)
            if supplied_nonce is None:
                return no_store(
                    JSONResponse(
                        status_code=403,
                        content=_validation_failure(
                            "NONCE",
                            "MISSING",
                            failure_code="NONCE_REJECTED",
                            origin_status="ACCEPTED",
                            nonce_status="REJECTED",
                            failed_stage="NONCE",
                        ),
                    )
                )
            if not secrets.compare_digest(supplied_nonce, nonce):
                return no_store(
                    JSONResponse(
                        status_code=403,
                        content=_validation_failure(
                            "NONCE",
                            "INVALID_FORMAT",
                            failure_code="NONCE_REJECTED",
                            origin_status="ACCEPTED",
                            nonce_status="REJECTED",
                            failed_stage="NONCE",
                        ),
                    )
                )

        json_post_paths = {
            CONNECT_AND_FETCH_PATH,
            "/local-test/kfh/otp",
            SELECT_ACCOUNT_PATH,
            PREVIEW_ACK_PATH,
        }
        if request.url.path in json_post_paths:
            if request.method != "POST":
                return no_store(
                    JSONResponse(
                        status_code=405,
                        content=_validation_failure(
                            "REQUEST_SHAPE",
                            "INVALID_FORMAT",
                            origin_status="ACCEPTED",
                            nonce_status="ACCEPTED",
                            failed_stage="REQUEST_VALIDATION",
                        ),
                    )
                )
            content_type = (
                request.headers.get("content-type", "")
                .split(";", 1)[0]
                .strip()
                .lower()
            )
            if content_type != "application/json":
                return no_store(
                    JSONResponse(
                        status_code=415,
                        content=_validation_failure(
                            "CONTENT_TYPE",
                            "INVALID_FORMAT",
                            failure_code="CONTENT_TYPE_REJECTED",
                            origin_status="ACCEPTED",
                            nonce_status="ACCEPTED",
                            content_type_status="REJECTED",
                            failed_stage="CONTENT_TYPE",
                        ),
                    )
                )

        try:
            response = await call_next(request)
        except Exception:
            payload = _base_diagnostics()
            payload.update(
                {
                    "result": "FAILED_CLOSED",
                    "failure": "LOCAL_API_HTTP_ERROR",
                    "stage": "LOCAL_TRANSPORT",
                    "failedStage": "LOCAL_API_HTTP",
                    "clientRequestStarted": True,
                    "localApiConnectionEstablished": True,
                    "originStatus": "ACCEPTED",
                    "nonceStatus": "ACCEPTED",
                    "contentTypeStatus": (
                        "ACCEPTED"
                        if request.url.path in json_post_paths
                        else "NOT_OBSERVED"
                    ),
                }
            )
            return no_store(JSONResponse(status_code=500, content=payload))
        return no_store(response)

    async def require_nonce(
        supplied: str | None = Header(default=None, alias=NONCE_HEADER),
    ) -> None:
        if supplied is None:
            raise HTTPException(status_code=403, detail="LOCAL_NONCE_MISSING")
        if not secrets.compare_digest(supplied, nonce):
            raise HTTPException(status_code=403, detail="LOCAL_NONCE_REJECTED")

    guard = [Depends(require_nonce)]

    @app.get(HEALTH_PATH, dependencies=guard)
    async def health() -> dict[str, bool | int | str]:
        return {
            "status": "OK",
            "localTestEnabled": True,
            "previewHandoffVersion": PREVIEW_HANDOFF_VERSION,
            "serverAllowedOrigin": canonical_allowed_origin,
            "serverAllowedOriginConfigured": allowed_origin_configured,
        }

    @app.post(CONNECT_AND_FETCH_PATH, dependencies=guard)
    async def connect(payload: LocalConnectAndFetchRequest) -> Any:
        result = await local_controller.connect(payload)
        response_payload = {
            **result,
            "clientRequestStarted": True,
            "localApiConnectionEstablished": True,
            "originStatus": "ACCEPTED",
            "serverAllowedOrigin": canonical_allowed_origin,
            "serverAllowedOriginConfigured": allowed_origin_configured,
            "nonceStatus": "ACCEPTED",
            "contentTypeStatus": "ACCEPTED",
            "bodyStatus": "PARSED",
            "requestValidated": True,
        }
        if result.get("failure") == "OPERATION_ALREADY_ACTIVE":
            return JSONResponse(status_code=409, content=response_payload)
        return response_payload

    @app.get("/local-test/kfh/status", dependencies=guard)
    async def status() -> dict[str, Any]:
        return await local_controller.status()

    @app.post("/local-test/kfh/otp", dependencies=guard)
    async def submit_otp(payload: LocalOtpRequest) -> dict[str, Any]:
        return await local_controller.submit_otp(payload)

    @app.post(SELECT_ACCOUNT_PATH, dependencies=guard)
    async def select_account(payload: LocalAccountSelectionRequest) -> dict[str, Any]:
        return await local_controller.select_account(payload)

    @app.get(PREVIEW_PATH, dependencies=guard)
    async def take_preview() -> Any:
        preview = await local_controller.take_preview()
        if preview is None:
            return JSONResponse(
                status_code=409,
                content={"status": "NOT_READY"},
            )
        return {"status": "OK", "preview": preview}

    @app.post(PREVIEW_ACK_PATH, dependencies=guard)
    async def acknowledge_preview() -> dict[str, str]:
        return await local_controller.acknowledge_preview()

    @app.post("/local-test/kfh/close", dependencies=guard)
    async def close() -> dict[str, Any]:
        return await local_controller.close()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await local_controller.close()

    exact_connect_routes = [
        route
        for route in app.routes
        if getattr(route, "path", None) == CONNECT_AND_FETCH_PATH
        and "POST" in (getattr(route, "methods", set()) or set())
    ]
    forbidden_routes = {
        "/local-test/kfh/connect",
        "/local-test/kfh/read-statement",
        "/local-test/kfh/connect-and-read",
    }
    registered_paths = {getattr(route, "path", None) for route in app.routes}
    if len(exact_connect_routes) != 1 or registered_paths.intersection(forbidden_routes):
        raise RuntimeError("KFH local-test route contract is not exact")
    app.state.connect_and_fetch_route_registered = True

    return app

def main() -> None:
    if os.environ.get("KFH_LOCAL_TEST_ENABLED") != "true":
        raise SystemExit("KFH_LOCAL_TEST_ENABLED=true is required")
    nonce = os.environ.get("KFH_LOCAL_TEST_NONCE", "")
    if len(nonce) < 16:
        raise SystemExit("KFH_LOCAL_TEST_NONCE with at least 16 characters is required")
    import uvicorn

    configured_origin = os.environ.get("KFH_LOCAL_TEST_ALLOWED_ORIGIN")
    allowed_origin = configured_origin or LOCAL_SAHAM_ORIGIN
    try:
        canonical_allowed_origin = canonicalize_local_origin(allowed_origin)
    except ValueError as error:
        raise SystemExit("KFH_LOCAL_TEST_ALLOWED_ORIGIN is not a valid URL origin") from error
    if canonical_allowed_origin != LOCAL_SAHAM_ORIGIN:
        raise SystemExit(f"KFH_LOCAL_TEST_ALLOWED_ORIGIN must be {LOCAL_SAHAM_ORIGIN}")
    app = create_local_test_app(
        nonce=nonce,
        allowed_origin=canonical_allowed_origin,
        allowed_origin_configured=configured_origin is not None,
    )
    uvicorn.run(app, host=LOCAL_HOST, port=LOCAL_PORT, access_log=False)


if __name__ == "__main__":
    main()
