"""Narrow transient KFH Cash Statement session adapter for Gate 5B-L1."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

REQUEST_FIELDS = frozenset(
    {
        "secAccNum",
        "frmDate",
        "toDate",
        "sortMode",
        "startSeq",
        "totalNoRec",
        "unqReqId",
    }
)


class Gate5BLiveAdapterError(RuntimeError):
    """Sanitized fail-closed live adapter error."""


class Gate5BLiveCorrelationError(Gate5BLiveAdapterError):
    """A 2/107 response did not match the one outstanding request."""


class Gate5BLiveSessionExpiredError(Gate5BLiveAdapterError):
    """Gate 3A left READY during a Cash Statement request."""


class Gate5BLiveAuthenticatedContextError(Gate5BLiveAdapterError):
    """The authenticated HED context was unavailable or ambiguous."""


class Gate5BLiveResponseStatusError(Gate5BLiveAdapterError):
    """A Cash Statement response had a non-success or invalid HED."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class Gate5BLiveBridgeFailureError(Gate5BLiveAdapterError):
    """The TypeScript reader failed closed with its own allowlisted code.

    `ts_code` and `detail` come from kfh-gate5b-live-bridge.cjs, which only
    ever emits a fixed KfhConnectorError code plus a static or field-name
    based message (see KfhReadOnlyConnector) - never raw account, balance,
    or transaction values.
    """

    def __init__(
        self,
        ts_code: str,
        detail: str | None,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"KFH bridge failed closed (code={ts_code!r})")
        self.ts_code = ts_code
        self.detail = detail
        self.evidence = evidence or {}


class CashStatementBrowserTransport(Protocol):
    async def send_cash_statement(self, request: dict[str, object]) -> None: ...


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
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


def _identity(candidate: dict[str, Any]) -> tuple[int | None, int | None]:
    hed = candidate.get("HED")
    if isinstance(hed, dict):
        return _integer(hed.get("msgGrp")), _integer(hed.get("msgTyp"))
    return _integer(candidate.get("msgGrp")), _integer(candidate.get("msgTyp"))


def _cash_statement_response(frame: str | bytes) -> dict[str, Any] | None:
    if not isinstance(frame, str) or len(frame) > 5_000_000:
        return None
    try:
        decoded = json.loads(frame)
    except (json.JSONDecodeError, TypeError):
        return None
    for candidate in _walk(decoded):
        if _identity(candidate) != (2, 107):
            continue
        hed = candidate.get("HED")
        if not isinstance(hed, dict):
            raise Gate5BLiveResponseStatusError("RESPONSE_PROTOCOL_IDENTITY_FAILED")
        if _integer(hed.get("chnlId")) != 30:
            raise Gate5BLiveResponseStatusError("RESPONSE_CHANNEL_FAILED")
        if _integer(hed.get("resSts")) != 0 or _integer(hed.get("errCode")) != 0:
            raise Gate5BLiveResponseStatusError("RESPONSE_STATUS_FAILED")
        response = candidate.get("response")
        dat = response.get("DAT") if isinstance(response, dict) else candidate.get("DAT")
        if isinstance(dat, dict):
            return {"msgGrp": 2, "msgTyp": 107, "response": {"DAT": dat}}
        raise Gate5BLiveResponseStatusError("RESPONSE_SCHEMA_FAILED")
    return None


def _validated_request(value: dict[str, Any]) -> dict[str, object]:
    if set(value) != REQUEST_FIELDS:
        raise Gate5BLiveAdapterError("Cash Statement request fields rejected")
    account = value.get("secAccNum")
    from_date = value.get("frmDate")
    to_date = value.get("toDate")
    request_id = value.get("unqReqId")
    if not isinstance(account, str) or not account.strip() or len(account) > 128:
        raise Gate5BLiveAdapterError("Cash Statement account rejected")
    if not isinstance(from_date, str) or not re.fullmatch(r"\d{8}", from_date):
        raise Gate5BLiveAdapterError("Cash Statement date range rejected")
    if not isinstance(to_date, str) or not re.fullmatch(r"\d{8}", to_date):
        raise Gate5BLiveAdapterError("Cash Statement date range rejected")
    if value.get("sortMode") != 0:
        raise Gate5BLiveAdapterError("Cash Statement sort mode rejected")
    start = value.get("startSeq")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0 or start % 20:
        raise Gate5BLiveAdapterError("Cash Statement cursor rejected")
    if value.get("totalNoRec") != 20:
        raise Gate5BLiveAdapterError("Cash Statement capacity rejected")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise Gate5BLiveAdapterError("Cash Statement correlation rejected")
    return dict(value)


class KfhCashStatementSessionAdapter:
    """Only request_cash_statement() and close() form its public session surface."""

    def __init__(
        self,
        transport: CashStatementBrowserTransport,
        *,
        ready: Callable[[], bool],
        authenticated_context_status: Callable[[], Awaitable[str]] | None = None,
        on_request_sent: Callable[[], None] | None = None,
        on_response_accepted: Callable[[], None] | None = None,
        timeout_seconds: float = 30,
    ) -> None:
        self.__transport = transport
        self.__ready = ready
        self.__authenticated_context_status = authenticated_context_status
        self.__on_request_sent = on_request_sent
        self.__on_response_accepted = on_response_accepted
        self.__timeout_seconds = timeout_seconds
        self.__closed = False
        self.__outstanding_id: str | None = None
        self.__response: asyncio.Future[dict[str, Any]] | None = None

    async def __context_status(self) -> str:
        callback = self.__authenticated_context_status
        if callback is None:
            return "AVAILABLE"
        try:
            status = await callback()
        except Exception:
            return "NOT_AVAILABLE"
        if status in {"AVAILABLE", "NOT_AVAILABLE", "AMBIGUOUS", "CLOSED"}:
            return status
        return "NOT_AVAILABLE"

    @staticmethod
    def __raise_for_context_status(status: str) -> None:
        if status == "AVAILABLE":
            return
        if status == "CLOSED":
            raise Gate5BLiveSessionExpiredError("KFH authenticated socket closed")
        if status == "AMBIGUOUS":
            raise Gate5BLiveAuthenticatedContextError(
                "AUTHENTICATED_CASH_STATEMENT_SOCKET_AMBIGUOUS"
            )
        raise Gate5BLiveAuthenticatedContextError(
            "AUTHENTICATED_CASH_STATEMENT_CONTEXT_NOT_AVAILABLE"
        )

    async def request_cash_statement(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.__closed:
            raise Gate5BLiveAdapterError("Cash Statement session is closed")
        if not self.__ready():
            raise Gate5BLiveSessionExpiredError("KFH session is not READY")
        self.__raise_for_context_status(await self.__context_status())
        if self.__outstanding_id is not None:
            raise Gate5BLiveAdapterError("Concurrent Cash Statement request rejected")

        validated = _validated_request(request)
        request_id = str(validated["unqReqId"])
        loop = asyncio.get_running_loop()
        response: asyncio.Future[dict[str, Any]] = loop.create_future()
        self.__outstanding_id = request_id
        self.__response = response

        async def session_monitor() -> None:
            while not response.done():
                if not self.__ready():
                    response.set_exception(
                        Gate5BLiveSessionExpiredError("KFH session left READY")
                    )
                    return
                status = await self.__context_status()
                if status != "AVAILABLE":
                    try:
                        self.__raise_for_context_status(status)
                    except Gate5BLiveAdapterError as error:
                        response.set_exception(error)
                    return
                await asyncio.sleep(0.1)

        monitor = asyncio.create_task(session_monitor())
        try:
            try:
                await self.__transport.send_cash_statement(validated)
            except Exception as error:
                status = await self.__context_status()
                if status != "AVAILABLE":
                    self.__raise_for_context_status(status)
                raise Gate5BLiveAdapterError(
                    "Cash Statement transport rejected the request"
                ) from error
            if self.__on_request_sent is not None:
                self.__on_request_sent()
            return await asyncio.wait_for(response, timeout=self.__timeout_seconds)
        except TimeoutError as error:
            raise Gate5BLiveAdapterError("Cash Statement response timed out") from error
        finally:
            monitor.cancel()
            self.__outstanding_id = None
            self.__response = None

    async def close(self) -> None:
        self.__closed = True
        response = self.__response
        if response is not None and not response.done():
            response.set_exception(Gate5BLiveAdapterError("Cash Statement session closed"))
        self.__outstanding_id = None
        self.__response = None

    def _observe_statement_response(self, frame: str | bytes) -> None:
        try:
            envelope = _cash_statement_response(frame)
        except Gate5BLiveResponseStatusError as error:
            response = self.__response
            if response is not None and not response.done():
                response.set_exception(error)
            return
        if envelope is None:
            return
        response = self.__response
        outstanding_id = self.__outstanding_id
        if response is None or response.done() or outstanding_id is None:
            return
        dat = envelope["response"]["DAT"]
        if str(dat.get("unqReqId", "")) != outstanding_id:
            response.set_exception(
                Gate5BLiveCorrelationError("Cash Statement response correlation failed")
            )
            return
        if self.__on_response_accepted is not None:
            self.__on_response_accepted()
        response.set_result(envelope)
