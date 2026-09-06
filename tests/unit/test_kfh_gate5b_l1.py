"""Gate 5B-L1 narrow live Cash Statement adapter tests using synthetic data."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import async_playwright

from local_connector.kfh_gate5b.adapter import (
    Gate5BLiveAdapterError,
    Gate5BLiveAuthenticatedContextError,
    Gate5BLiveBridgeFailureError,
    Gate5BLiveCorrelationError,
    Gate5BLiveResponseStatusError,
    Gate5BLiveSessionExpiredError,
    KfhCashStatementSessionAdapter,
)
from local_connector.kfh_gate5b.bridge import (
    BRIDGE_PATH,
    run_typescript_cash_statement_read,
)
from local_connector.kfh_gate5b.browser import (
    CASH_STATEMENT_CONTEXT_STATUS_SCRIPT,
    CASH_STATEMENT_MARK_READY_SCRIPT,
    CASH_STATEMENT_SEND_SCRIPT,
    CASH_STATEMENT_SOCKET_HOOK,
    Gate5BLiveBrowserRuntime,
)

FAKE_NATIVE_WEBSOCKET_JS = r"""
() => {
  window.__sentFrames = [];
  class FakeWebSocket {
    constructor(url) {
      this.url = url;
      this.readyState = 1;
      this._listeners = {};
    }
    addEventListener(type, handler) {
      (this._listeners[type] ||= []).push(handler);
    }
    send(data) {
      window.__sentFrames.push(data);
    }
    __emit(data) {
      (this._listeners["message"] || []).forEach((handler) => handler({ data }));
    }
  }
  FakeWebSocket.OPEN = 1;
  window.WebSocket = FakeWebSocket;
}
"""
from local_connector.kfh_gate5b.evidence import (
    audit_live_evidence,
    build_live_evidence,
    write_live_evidence,
)


def request(start: int = 0, request_id: str = "SYNTHETIC-ID") -> dict[str, Any]:
    return {
        "secAccNum": "SYNTHETIC-ACCOUNT",
        "frmDate": "20251001",
        "toDate": "20260902",
        "sortMode": 0,
        "startSeq": start,
        "totalNoRec": 20,
        "unqReqId": request_id,
    }


def response_frame(
    request_id: str,
    *,
    count: int = 0,
    has_next: int = 0,
    total: int = 1,
    channel: int = 30,
    response_status: int = 0,
    error_code: int = 0,
) -> str:
    return json.dumps(
        {
            "HED": {
                "msgGrp": 2,
                "msgTyp": 107,
                "chnlId": channel,
                "resSts": response_status,
                "errCode": error_code,
            },
            "DAT": {
                "curr": "KWD",
                "openBal": "0.000",
                "closeBal": "0.000",
                "totDeposit": "0.000",
                "totWithdrawal": "0.000",
                "totBuy": "0.000",
                "totSell": "0.000",
                "totOther": "0.000",
                "vatAmount": "0.000",
                "totalNoRec": total,
                "isNxtPagAvail": has_next,
                "pageNo": -1,
                "totalPages": -1,
                "unqReqId": request_id,
                "cashLogs": [
                    {
                        "date": "20260101",
                        "trnsType": "SYNTHETIC",
                        "trnsRef": f"SYNTHETIC-{index}",
                    }
                    for index in range(count)
                ],
                "unsettledCashLogs": [],
            },
        }
    )


class FakeTransport:
    def __init__(self) -> None:
        self.on_send = None
        self.sent: list[dict[str, object]] = []

    async def send_cash_statement(self, value: dict[str, object]) -> None:
        self.sent.append(dict(value))
        if self.on_send:
            self.on_send(value)


@pytest.mark.asyncio
async def test_adapter_exposes_only_narrow_request_and_close_surface() -> None:
    public = {
        name
        for name, member in inspect.getmembers(
            KfhCashStatementSessionAdapter, inspect.isfunction
        )
        if not name.startswith("_")
    }
    assert public == {"request_cash_statement", "close"}


def test_outbound_browser_script_is_fixed_to_cash_statement_2_7_30() -> None:
    compact = "".join(CASH_STATEMENT_SEND_SCRIPT.split())
    for field in ("ver", "clVer", "sesnId", "usrId"):
        assert f"{field}:authenticatedHed.{field}" in compact
    assert "msgGrp:2" in compact
    assert "msgTyp:7" in compact
    assert "chnlId:30" in compact
    assert "socket.send(JSON.stringify" in compact
    assert "constsocket=state.boundSocket" in compact
    assert ".at(-1)" not in compact
    forbidden = (
        "sendRawMessage",
        "sendMessage",
        "sendProtocol",
        "placeOrder",
        "cancelOrder",
        "transfer",
        "withdraw",
        "password",
        "otp",
    )
    assert not any(value in CASH_STATEMENT_SEND_SCRIPT for value in forbidden)


def test_authenticated_context_promotes_only_successful_auth_socket_then_stays_exact() -> None:
    """Real KFH evidence (sanitized field-name samples, 2026-09) shows
    sesnId/usrId are echoed by the server but ver/clVer are declared only in
    the client's own outbound frames and never appear in server responses.
    So identity is assembled incrementally from BOTH directions on the one
    socket proven authenticated via the real 5/101 authSts=1 signal, rather
    than requiring one single inbound frame to carry all four fields."""
    compact_hook = "".join(CASH_STATEMENT_SOCKET_HOOK.split())
    assert "Number(value.HED?.msgGrp)===5" in compact_hook
    assert "Number(value.HED?.msgTyp)===101" in compact_hook
    assert "Number(value.DAT?.authSts)===1" in compact_hook
    assert "state.successfulAuthSocket=socket" in compact_hook
    assert "state.successfulAuthSocket!==socket" in compact_hook
    assert "state.successfulAuthAmbiguous=true" in compact_hook
    assert 'socket.addEventListener("message"' in compact_hook
    # Identity must be merged from the KFH page's own outbound requests too
    # (ver/clVer never appear inbound) - the native send must still always
    # be forwarded exactly, observation is passive.
    assert "constnativeSend=socket.send.bind(socket)" in compact_hook
    assert "returnnativeSend(data)" in compact_hook
    assert "state.identityBySocket" in compact_hook
    assert "state.ambiguous=true" in compact_hook
    assert "state.boundSocketClosed=true" in compact_hook
    compact_ready = "".join(CASH_STATEMENT_MARK_READY_SCRIPT.split())
    assert "state.ready=true" in compact_ready
    assert "state.boundSocket=state.successfulAuthAmbiguous?null:state.successfulAuthSocket" in compact_ready
    assert "AMBIGUOUS" in CASH_STATEMENT_CONTEXT_STATUS_SCRIPT
    assert "CLOSED" in CASH_STATEMENT_CONTEXT_STATUS_SCRIPT
    assert "state.hasCompleteIdentity(state.authenticatedHed)" in CASH_STATEMENT_CONTEXT_STATUS_SCRIPT


@pytest.mark.asyncio
async def test_identity_assembles_from_inbound_and_outbound_traffic_end_to_end() -> None:
    """Real KFH /wstrs evidence: server responses (inbound) reliably carry
    sesnId/usrId but never ver/clVer; ver/clVer are only ever declared by
    the client's own outbound requests. This proves the full hook - real
    browser execution, not just source text - correctly withholds
    AVAILABLE until identity is complete, then unblocks once the KFH page's
    own outbound traffic supplies the missing fields, all while the native
    send is still always forwarded untouched."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.evaluate(FAKE_NATIVE_WEBSOCKET_JS)
            await page.evaluate(CASH_STATEMENT_SOCKET_HOOK)
            await page.evaluate(
                "() => { window.__socket = new window.WebSocket('wss://trading.kfhtrade.com/wstrs'); }"
            )

            # Inbound auth-success signal, matching real evidence: no ver/clVer.
            await page.evaluate(
                "(frame) => window.__socket.__emit(frame)",
                json.dumps({"HED": {"msgGrp": 5, "msgTyp": 101, "sesnId": "SYN-SESSION"},
                            "DAT": {"authSts": 1}}),
            )
            assert await page.evaluate(CASH_STATEMENT_MARK_READY_SCRIPT) is True
            assert await page.evaluate(CASH_STATEMENT_CONTEXT_STATUS_SCRIPT) == "NOT_AVAILABLE"

            # Inbound dashboard-population response, matching real evidence:
            # sesnId + usrId present, still no ver/clVer.
            await page.evaluate(
                "(frame) => window.__socket.__emit(frame)",
                json.dumps({"HED": {"msgGrp": 3, "msgTyp": 102, "sesnId": "SYN-SESSION",
                                     "usrId": "SYN-USER"}, "DAT": {"portfls": []}}),
            )
            assert await page.evaluate(CASH_STATEMENT_CONTEXT_STATUS_SCRIPT) == "NOT_AVAILABLE"

            # KFH's own page sends an outbound request carrying ver/clVer.
            # The native send must still be forwarded exactly.
            outbound = json.dumps({"HED": {"ver": "SYN-1.0", "msgGrp": 3, "msgTyp": 2,
                                            "chnlId": 30, "clVer": "SYN-2.0",
                                            "sesnId": "SYN-SESSION", "usrId": "SYN-USER"},
                                    "DAT": {}})
            await page.evaluate("(frame) => window.__socket.send(frame)", outbound)
            assert await page.evaluate("() => window.__sentFrames") == [outbound]
            assert await page.evaluate(CASH_STATEMENT_CONTEXT_STATUS_SCRIPT) == "AVAILABLE"

            # The real statement send now succeeds end-to-end with the
            # assembled identity, and the native transport still receives it.
            statement_request = request()
            await page.evaluate(CASH_STATEMENT_SEND_SCRIPT, statement_request)
            sent = await page.evaluate("() => window.__sentFrames")
            second = json.loads(sent[1])
            assert second["HED"] == {
                "ver": "SYN-1.0", "msgGrp": 2, "msgTyp": 7, "chnlId": 30,
                "clVer": "SYN-2.0", "sesnId": "SYN-SESSION", "usrId": "SYN-USER",
            }
        finally:
            await browser.close()


def test_authenticated_values_are_never_logged_or_persisted() -> None:
    source = CASH_STATEMENT_SOCKET_HOOK + CASH_STATEMENT_SEND_SCRIPT
    for forbidden in ("console.", "localStorage", "sessionStorage", "indexedDB"):
        assert forbidden not in source


@pytest.mark.asyncio
async def test_incomplete_authenticated_context_cannot_send() -> None:
    transport = FakeTransport()

    async def unavailable() -> str:
        return "NOT_AVAILABLE"

    adapter = KfhCashStatementSessionAdapter(
        transport,
        ready=lambda: True,
        authenticated_context_status=unavailable,
    )
    with pytest.raises(
        Gate5BLiveAuthenticatedContextError,
        match="AUTHENTICATED_CASH_STATEMENT_CONTEXT_NOT_AVAILABLE",
    ):
        await adapter.request_cash_statement(request())
    assert transport.sent == []


@pytest.mark.asyncio
async def test_all_other_outbound_shapes_are_rejected() -> None:
    adapter = KfhCashStatementSessionAdapter(FakeTransport(), ready=lambda: True)
    invalid = request()
    invalid["msgGrp"] = 3
    with pytest.raises(Gate5BLiveAdapterError, match="fields rejected"):
        await adapter.request_cash_statement(invalid)


@pytest.mark.asyncio
async def test_correlated_2_107_response_completes_and_id_is_cleared() -> None:
    transport = FakeTransport()
    adapter = KfhCashStatementSessionAdapter(transport, ready=lambda: True)
    transport.on_send = lambda value: adapter._observe_statement_response(
        response_frame(str(value["unqReqId"]))
    )
    envelope = await adapter.request_cash_statement(request())
    assert envelope["msgGrp"] == 2
    assert envelope["msgTyp"] == 107
    assert adapter._KfhCashStatementSessionAdapter__outstanding_id is None


@pytest.mark.asyncio
async def test_mismatched_response_id_fails_closed_and_is_cleared() -> None:
    transport = FakeTransport()
    adapter = KfhCashStatementSessionAdapter(transport, ready=lambda: True)
    transport.on_send = lambda _value: adapter._observe_statement_response(
        response_frame("OTHER-SYNTHETIC-ID")
    )
    with pytest.raises(Gate5BLiveCorrelationError):
        await adapter.request_cash_statement(request())
    assert adapter._KfhCashStatementSessionAdapter__outstanding_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"channel": 31}, "RESPONSE_CHANNEL_FAILED"),
        ({"response_status": 1}, "RESPONSE_STATUS_FAILED"),
        ({"error_code": 7}, "RESPONSE_STATUS_FAILED"),
    ],
)
async def test_invalid_response_hed_or_status_fails_closed(
    overrides: dict[str, int], expected_code: str
) -> None:
    transport = FakeTransport()
    adapter = KfhCashStatementSessionAdapter(transport, ready=lambda: True)
    transport.on_send = lambda value: adapter._observe_statement_response(
        response_frame(str(value["unqReqId"]), **overrides)
    )
    with pytest.raises(Gate5BLiveResponseStatusError, match=expected_code):
        await adapter.request_cash_statement(request())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("AMBIGUOUS", Gate5BLiveAuthenticatedContextError),
        ("CLOSED", Gate5BLiveSessionExpiredError),
    ],
)
async def test_ambiguous_or_closed_authenticated_socket_fails_closed(
    status: str, expected: type[Exception]
) -> None:
    async def context_status() -> str:
        return status

    transport = FakeTransport()
    adapter = KfhCashStatementSessionAdapter(
        transport,
        ready=lambda: True,
        authenticated_context_status=context_status,
    )
    with pytest.raises(expected):
        await adapter.request_cash_statement(request())
    assert transport.sent == []


@pytest.mark.asyncio
async def test_only_one_request_can_be_outstanding() -> None:
    adapter = KfhCashStatementSessionAdapter(
        FakeTransport(), ready=lambda: True, timeout_seconds=1
    )
    first = asyncio.create_task(adapter.request_cash_statement(request()))
    await asyncio.sleep(0)
    with pytest.raises(Gate5BLiveAdapterError, match="Concurrent"):
        await adapter.request_cash_statement(request(20, "SECOND-ID"))
    await adapter.close()
    with pytest.raises(Gate5BLiveAdapterError, match="session closed"):
        await first


@pytest.mark.asyncio
async def test_session_expiry_and_malformed_response_fail_safely() -> None:
    with pytest.raises(Gate5BLiveSessionExpiredError):
        await KfhCashStatementSessionAdapter(
            FakeTransport(), ready=lambda: False
        ).request_cash_statement(request())

    ready = True
    transport = FakeTransport()
    adapter = KfhCashStatementSessionAdapter(
        transport, ready=lambda: ready, timeout_seconds=0.5
    )

    def expire(_value: dict[str, object]) -> None:
        nonlocal ready
        adapter._observe_statement_response("not-json")
        ready = False

    transport.on_send = expire
    with pytest.raises(Gate5BLiveSessionExpiredError):
        await adapter.request_cash_statement(request())


@pytest.mark.asyncio
async def test_close_rejects_future_requests() -> None:
    adapter = KfhCashStatementSessionAdapter(FakeTransport(), ready=lambda: True)
    await adapter.close()
    with pytest.raises(Gate5BLiveAdapterError, match="closed"):
        await adapter.request_cash_statement(request())


def test_runtime_exposes_no_browser_page_websocket_or_generic_send_handle() -> None:
    public = {
        name
        for name, member in inspect.getmembers(Gate5BLiveBrowserRuntime, inspect.isfunction)
        if not name.startswith("_")
    }
    assert public == {"open", "send_cash_statement"}
    assert public.isdisjoint({"page", "browser", "context", "websocket", "evaluate"})


def test_package_has_no_financial_repository_or_write_path() -> None:
    package = Path(__file__).parents[2] / "local_connector" / "kfh_gate5b"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    forbidden = (
        "broker_import",
        "cash_deposits",
        "portfolio_cash",
        "confirmKfhImportBatch",
        "app.models",
        "app.services.kfh_sync",
        "place_order",
        "cancel_order",
    )
    assert not any(value in source for value in forbidden)


def valid_live() -> dict[str, Any]:
    return {
        "requestStartSeqProgression": [0, 20, 40, 60],
        "requestPageCapacity": 20,
        "responseCashLogsCounts": [19, 20, 7, 0],
        "responseUnsettledCounts": [0, 0, 0, 0],
        "isNxtPagAvailSequence": [1, 1, 1, 0],
        "responseTotalSequence": [69, 69, 69, 69],
        "correlationStrategy": "FRESH_UNIQUE_OPAQUE_ID_PER_PAGE",
        "allResponsesCorrelated": True,
        "oneOutstandingRequestMaximum": 1,
        "finalResponseObserved": True,
        "finalIsNextPageAvailable": 0,
        "partialRead": False,
        "financialWritesPerformed": 0,
    }


def test_sanitized_evidence_contains_no_real_identifiers_or_financial_payload(
    tmp_path: Path,
) -> None:
    evidence = build_live_evidence(
        valid_live(),
        gate3a_ready=True,
        browser_closed_successfully=True,
        new_run_restored_session=False,
    )
    audit_live_evidence(evidence)
    path = tmp_path / "kfh_gate5b_live_read_redacted_evidence_v1.json"
    digest = write_live_evidence(path, evidence)
    serialized = path.read_text(encoding="utf-8")
    assert len(digest) == 64
    assert evidence["liveReadPass"] is True
    assert "SYNTHETIC-ACCOUNT" not in serialized
    assert "SYNTHETIC-ID" not in serialized
    assert "trnsRef" not in serialized
    assert evidence["financialWritesPerformed"] == 0


def test_failed_live_conditions_cannot_write_passing_evidence(tmp_path: Path) -> None:
    live = valid_live()
    live["allResponsesCorrelated"] = False
    evidence = build_live_evidence(
        live,
        gate3a_ready=True,
        browser_closed_successfully=True,
        new_run_restored_session=False,
    )
    assert evidence["liveReadPass"] is False
    with pytest.raises(ValueError, match="failed read"):
        write_live_evidence(tmp_path / "forbidden.json", evidence)


@pytest.mark.asyncio
async def test_private_bridge_runs_actual_typescript_reader_with_fresh_ids() -> None:
    process = await asyncio.create_subprocess_exec(
        "node",
        str(BRIDGE_PATH),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    start = {
        "type": "start",
        "query": {
            "secAccNum": "SYNTHETIC-ACCOUNT",
            "frmDate": "20251001",
            "toDate": "20260902",
            "sortMode": 0,
            "startSeq": 0,
            "totalNoRec": 20,
        },
    }
    process.stdin.write((json.dumps(start) + "\n").encode())
    await process.stdin.drain()
    shapes = [(19, 1), (20, 1), (7, 1), (0, 0)]
    starts: list[int] = []
    request_ids: list[str] = []
    complete: dict[str, Any] | None = None
    for count, has_next in shapes:
        message = json.loads(await process.stdout.readline())
        assert message["type"] == "cash_statement_request"
        wire_request = message["request"]
        starts.append(wire_request["startSeq"])
        request_ids.append(wire_request["unqReqId"])
        envelope = json.loads(
            response_frame(
                wire_request["unqReqId"], count=count, has_next=has_next, total=69
            )
        )
        normalized = {
            "msgGrp": envelope["HED"]["msgGrp"],
            "msgTyp": envelope["HED"]["msgTyp"],
            "response": {"DAT": envelope["DAT"]},
        }
        process.stdin.write(
            (
                json.dumps(
                    {"type": "cash_statement_response", "envelope": normalized}
                )
                + "\n"
            ).encode()
        )
        await process.stdin.drain()
    complete = json.loads(await process.stdout.readline())
    assert complete["type"] == "complete"
    await process.wait()

    evidence = complete["evidence"]
    assert starts == [0, 20, 40, 60]
    assert len(set(request_ids)) == 4
    assert evidence["correlationStrategy"] == "FRESH_UNIQUE_OPAQUE_ID_PER_PAGE"
    assert evidence["allResponsesCorrelated"] is True
    assert evidence["oneOutstandingRequestMaximum"] == 1
    assert evidence["financialWritesPerformed"] == 0
    serialized = json.dumps(complete)
    assert "SYNTHETIC-ACCOUNT" not in serialized
    assert not any(request_id in serialized for request_id in request_ids)


@pytest.mark.asyncio
async def test_python_adapter_bridge_executes_actual_reader_end_to_end() -> None:
    transport = FakeTransport()
    adapter = KfhCashStatementSessionAdapter(transport, ready=lambda: True)

    def reply(value: dict[str, object]) -> None:
        start = int(value["startSeq"])
        adapter._observe_statement_response(
            response_frame(
                str(value["unqReqId"]),
                count=1,
                has_next=1 if start == 0 else 0,
                total=21,
            )
        )

    transport.on_send = reply
    evidence = await run_typescript_cash_statement_read(
        adapter,
        {
            "secAccNum": "SYNTHETIC-ACCOUNT",
            "frmDate": "20251001",
            "toDate": "20260902",
            "sortMode": 0,
            "startSeq": 0,
            "totalNoRec": 20,
        },
    )
    assert evidence["requestStartSeqProgression"] == [0, 20]
    assert evidence["allResponsesCorrelated"] is True
    assert evidence["finalResponseObserved"] is True
    assert evidence["financialWritesPerformed"] == 0


@pytest.mark.asyncio
async def test_bridge_failure_preserves_only_sanitized_page_progress() -> None:
    transport = FakeTransport()
    adapter = KfhCashStatementSessionAdapter(transport, ready=lambda: True)

    def reply(value: dict[str, object]) -> None:
        start = int(value["startSeq"])
        adapter._observe_statement_response(
            response_frame(
                str(value["unqReqId"]),
                count=1,
                has_next=1 if start == 0 else 0,
                total=21 if start == 0 else 22,
            )
        )

    transport.on_send = reply
    with pytest.raises(Gate5BLiveBridgeFailureError) as captured:
        await run_typescript_cash_statement_read(
            adapter,
            {
                "secAccNum": "SYNTHETIC-ACCOUNT",
                "frmDate": "20251001",
                "toDate": "20260902",
                "sortMode": 0,
                "startSeq": 0,
                "totalNoRec": 20,
            },
        )

    error = captured.value
    assert error.ts_code == "PAGINATION_TOTAL_DRIFT"
    assert error.evidence["requestStartSeqProgression"] == [0, 20]
    assert error.evidence["responseCashLogsCounts"] == [1, 1]
    assert error.evidence["isNxtPagAvailSequence"] == [1, 0]
    assert error.evidence["partialRead"] is True
    assert error.evidence["financialWritesPerformed"] == 0
    serialized = json.dumps(error.evidence)
    assert "SYNTHETIC-ACCOUNT" not in serialized
    assert "unqReqId" not in serialized
