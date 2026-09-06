"""Gate 5A-R2 first-page telemetry and passive UI diagnostic tests."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from local_connector.kfh_gate5a.browser import (
    Gate5APassiveBrowserRuntime,
    route_gate5a_outbound_frame,
)
from local_connector.kfh_gate5a.capture import (
    Gate5ACaptureError,
    KfhGate5APassiveCapture,
    audit_sanitized_attempt_record,
    write_attempt_record,
)

ROOT = Path(__file__).resolve().parents[2]


def request(*, request_id: str = "PRIVATE-REQUEST", start_seq: int = 0) -> str:
    return json.dumps(
        {
            "HED": {"msgGrp": 2, "msgTyp": 7, "chnlId": 30},
            "DAT": {
                "secAccNum": "PRIVATE-ACCOUNT",
                "frmDate": "20260101",
                "toDate": "20260831",
                "sortMode": 1,
                "startSeq": start_seq,
                "totalNoRec": 20,
                "unqReqId": request_id,
            },
        }
    )


def response(*, request_id: str = "PRIVATE-REQUEST", has_next: int) -> str:
    return json.dumps(
        {
            "HED": {"msgGrp": 2, "msgTyp": 107, "resSts": 1, "errCode": 0},
            "DAT": {
                "totalNoRec": 27,
                "isNxtPagAvail": has_next,
                "pageNo": -1,
                "totalPages": -1,
                "unqReqId": request_id,
                "cashLogs": [
                    {
                        "date": "20260831",
                        "trnsRef": "PRIVATE-TRANSACTION-REFERENCE",
                        "amount": "PRIVATE-FINANCIAL-VALUE",
                    }
                ],
                "unsettledCashLogs": [],
            },
        }
    )


def first_page(has_next: int, callback=None) -> KfhGate5APassiveCapture:
    capture = KfhGate5APassiveCapture(on_first_page=callback)
    capture.activate_after_gate3a_ready()
    capture.observe_request_frame(request())
    capture.observe_response_frame(response(has_next=has_next))
    return capture


def test_first_page_summary_emitted_after_correlated_request_and_response():
    emitted = []
    capture = first_page(1, emitted.append)

    assert capture.completed_page_count == 1
    assert len(emitted) == 1
    assert emitted[0]["firstPageCaptured"] is True
    assert emitted[0]["requestResponseCorrelated"] is True
    assert emitted[0]["startSeq"] == 0
    assert emitted[0]["requestPageCapacity"] == 20
    assert emitted[0]["responseTotalMatchingRecords"] == 27


def test_first_page_summary_contains_no_secrets_or_financial_values():
    summary = first_page(1).first_page_summary
    assert summary is not None
    serialized = json.dumps(summary)
    for secret in (
        "PRIVATE-REQUEST",
        "PRIVATE-ACCOUNT",
        "PRIVATE-TRANSACTION-REFERENCE",
        "PRIVATE-FINANCIAL-VALUE",
        "unqReqId",
        "secAccNum",
    ):
        assert secret not in serialized


def test_no_continuation_produces_range_not_multipage():
    summary = first_page(0).first_page_summary
    assert summary is not None
    assert summary["kfhProtocolSaysMorePages"] is False
    assert summary["rangeDisposition"] == "RANGE_NOT_MULTIPAGE"


def test_continuation_produces_protocol_more_pages_expectation():
    summary = first_page(1).first_page_summary
    assert summary is not None
    assert summary["kfhProtocolSaysMorePages"] is True
    assert summary["rangeDisposition"] == "KFH_PROTOCOL_SAYS_MORE_PAGES"


def test_single_page_evidence_may_be_retained_without_gate_pass(tmp_path):
    record = first_page(1).build_sanitized_attempt_record()
    assert record["result"]["captureValid"] is True
    assert record["result"]["multiPageObserved"] is False
    assert record["result"]["gate5aPass"] is False
    assert record["result"]["paginationAuthorized"] is False
    assert record["result"]["queryCaptureOrigin"] == "OWNER_ARMED_VIEW_ATTEMPT"
    assert record["result"]["repeatedPageRequestCount"] == 0
    path = tmp_path / "cash_statement_attempt_redacted.json"
    assert len(write_attempt_record(path, record)) == 64
    assert path.exists()


def test_first_page_attempt_cannot_authorize_pagination():
    record = first_page(1).build_sanitized_attempt_record()
    record["result"]["paginationAuthorized"] = True
    with pytest.raises(Gate5ACaptureError, match="cannot authorize pagination"):
        audit_sanitized_attempt_record(record)


def test_attempt_record_rejects_arbitrary_text_in_date_range():
    record = first_page(0).build_sanitized_attempt_record()
    record["pages"][0]["dateRange"] = "PRIVATE-ARBITRARY-TEXT"
    with pytest.raises(Gate5ACaptureError, match="date range"):
        audit_sanitized_attempt_record(record)


@pytest.mark.asyncio
async def test_pagination_ui_diagnostic_reads_booleans_without_clicking():
    candidate = {
        "matched": False,
        "visible": False,
        "count": 0,
        "disabled": False,
        "belowCurrentViewport": False,
        "insideScrollableContainer": False,
    }
    diagnostic = {
        name: dict(candidate)
        for name in (
            "NEXT_TEXT",
            "NEXT_BUTTON_CANDIDATE",
            "PREVIOUS_TEXT",
            "FORWARD_CHEVRON",
            "DOUBLE_FORWARD_CHEVRON",
            "PAGINATION_CONTAINER",
            "PAGE_NUMBER_CONTROLS",
            "TABLE_FOOTER_CONTROLS",
        )
    }
    diagnostic["STATEMENT_SCROLL_CONTAINER"] = {"present": True}

    class FakePage:
        url = "https://trading.kfhtrade.com/statement"
        click_count = 0

        async def evaluate(self, script):
            assert "click(" not in script
            return diagnostic

        async def click(self, *_args, **_kwargs):
            self.click_count += 1

    page = FakePage()
    runtime = Gate5APassiveBrowserRuntime(
        on_statement_request_frame=lambda _frame: None,
        on_statement_response_frame=lambda _frame: None,
    )
    runtime._Gate5APassiveBrowserRuntime__page = page
    observed = await runtime.inspect_pagination_ui()

    assert observed == diagnostic
    assert page.click_count == 0


def test_gate5a_has_no_generated_websocket_message_surface():
    parameters = set(inspect.signature(route_gate5a_outbound_frame).parameters)
    assert parameters == {"frame", "on_statement_request_frame"}
    observed = []
    result = route_gate5a_outbound_frame(
        request(), on_statement_request_frame=observed.append
    )
    assert result is None
    assert observed == [request()]


def test_gate3a_r1_closure_record_remains_unchanged():
    record = (ROOT / "docs" / "kfh-gate3a-r1-gate-record.md").read_text(
        encoding="utf-8"
    )
    assert "**CLOSED / PASSED**" in record
    assert "54a8751dd79ba2144a8ee90c7a32496c04d869e4fcd3df50c449b07a273d43d4" in record
    assert "Supersession scope | Original Gate 3A login-UI detector only" in record


def test_kfh_auto_sync_source_default_remains_false():
    source = (ROOT / "app" / "core" / "config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    defaults = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "KFH_AUTO_SYNC_ENABLED"
        and isinstance(node.value, ast.Constant)
    ]
    assert defaults == [False]
