"""Gate 5A-R2.1 explicit arming and retry/query-separation tests."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from local_connector.kfh_gate5a.browser import route_gate5a_outbound_frame
from local_connector.kfh_gate5a.capture import (
    Gate5ACaptureError,
    KfhGate5APassiveCapture,
)
from local_connector.kfh_gate5a.live_capture import _arm_capture, _wait_for_arm

ROOT = Path(__file__).resolve().parents[2]


def request(
    start_seq: int,
    request_id: str,
    *,
    from_date: str = "20260101",
    to_date: str = "20260831",
    sort_mode: int = 1,
    capacity: int = 20,
) -> str:
    return json.dumps(
        {
            "HED": {"msgGrp": 2, "msgTyp": 7, "chnlId": 30},
            "DAT": {
                "secAccNum": "PRIVATE-ACCOUNT",
                "frmDate": from_date,
                "toDate": to_date,
                "sortMode": sort_mode,
                "startSeq": start_seq,
                "totalNoRec": capacity,
                "unqReqId": request_id,
            },
        }
    )


def response(
    request_id: str,
    *,
    references: list[str],
    has_next: int,
    total: int = 25,
) -> str:
    return json.dumps(
        {
            "HED": {"msgGrp": 2, "msgTyp": 107, "resSts": 1, "errCode": 0},
            "DAT": {
                "totalNoRec": total,
                "isNxtPagAvail": has_next,
                "pageNo": -1,
                "totalPages": -1,
                "unqReqId": request_id,
                "cashLogs": [
                    {
                        "date": f"202608{31 - index:02d}",
                        "trnsRef": reference,
                        "amount": "PRIVATE-FINANCIAL-VALUE",
                    }
                    for index, reference in enumerate(references)
                ],
                "unsettledCashLogs": [],
            },
        }
    )


def add_first_page(capture: KfhGate5APassiveCapture) -> None:
    capture.observe_request_frame(request(0, "PRIVATE-REQUEST-1"))
    capture.observe_response_frame(
        response(
            "PRIVATE-REQUEST-1",
            references=[f"PRIVATE-REF-{index}" for index in range(17)],
            has_next=1,
        )
    )


def test_no_statement_frames_before_arm_are_captured():
    capture = KfhGate5APassiveCapture()
    capture.observe_request_frame(request(0, "PRE-ARM-REQUEST"))
    capture.observe_response_frame(
        response("PRE-ARM-REQUEST", references=["PRE-ARM-REF"], has_next=0)
    )
    assert capture.completed_page_count == 0
    with pytest.raises(Gate5ACaptureError, match="No correlated"):
        capture.build_sanitized_attempt_record()


def test_arm_itself_performs_no_page_or_network_action():
    class PassiveCaptureProbe:
        activation_count = 0

        def activate_after_gate3a_ready(self):
            self.activation_count += 1

    probe = PassiveCaptureProbe()
    assert _arm_capture(probe) is None
    assert probe.activation_count == 1


@pytest.mark.asyncio
async def test_runner_requires_exact_arm_command(monkeypatch, capsys):
    commands = iter(["arm", "ARM"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    capture = KfhGate5APassiveCapture()
    await _wait_for_arm(capture)
    output = capsys.readouterr().out
    assert "CAPTURE_NOT_ARMED" in output
    assert "GATE 5A CAPTURE ARMED" in output
    assert "NOW CLICK KFH VIEW" in output
    assert capture.active is True


def test_first_request_after_arm_becomes_logical_page_one():
    summaries = []
    capture = KfhGate5APassiveCapture(on_first_page=summaries.append)
    _arm_capture(capture)
    add_first_page(capture)
    assert capture.completed_page_count == 1
    assert summaries[0]["startSeq"] == 0
    assert summaries[0]["queryCaptureOrigin"] == "OWNER_ARMED_VIEW_ATTEMPT"
    assert summaries[0]["repeatedPageRequestCount"] == 0


def test_completed_same_query_page_zero_request_is_classified_as_retry():
    capture = KfhGate5APassiveCapture()
    _arm_capture(capture)
    add_first_page(capture)
    capture.observe_request_frame(request(0, "PRIVATE-RETRY-1"))
    assert capture.repeated_page_request_count == 1
    assert capture.first_page_summary["repeatedPageRequestCount"] == 1


def test_same_page_request_while_outstanding_fails_closed_without_new_page():
    capture = KfhGate5APassiveCapture()
    _arm_capture(capture)
    capture.observe_request_frame(request(0, "PRIVATE-REQUEST-1"))
    capture.observe_request_frame(request(0, "PRIVATE-REQUEST-2"))
    assert capture.completed_page_count == 0
    with pytest.raises(Gate5ACaptureError, match="DUPLICATE_OUTSTANDING_REQUEST"):
        capture.build_sanitized_attempt_record()


def test_retry_is_not_counted_as_second_page():
    capture = KfhGate5APassiveCapture()
    _arm_capture(capture)
    add_first_page(capture)
    capture.observe_request_frame(request(0, "PRIVATE-RETRY-1"))
    assert capture.completed_page_count == 1


def test_retry_response_correlates_without_contaminating_page_evidence():
    capture = KfhGate5APassiveCapture()
    _arm_capture(capture)
    add_first_page(capture)
    capture.observe_request_frame(request(0, "PRIVATE-RETRY-1"))
    capture.observe_response_frame(
        response(
            "PRIVATE-RETRY-1",
            references=["RETRY-ONLY-PRIVATE-REF"],
            has_next=1,
        )
    )
    assert capture.completed_page_count == 1
    assert capture.raw_correlation_identifiers_retained is False
    record = capture.build_sanitized_attempt_record()
    assert record["pages"][0]["cashLogsCount"] == 17
    assert record["result"]["repeatedPageRequestCount"] == 1
    assert "RETRY-ONLY-PRIVATE-REF" not in json.dumps(record)


def test_same_query_new_start_sequence_becomes_continuation_page():
    capture = KfhGate5APassiveCapture()
    _arm_capture(capture)
    add_first_page(capture)
    capture.observe_request_frame(request(17, "PRIVATE-REQUEST-2"))
    capture.observe_response_frame(
        response("PRIVATE-REQUEST-2", references=["PAGE-2-REF"], has_next=0)
    )
    assert capture.completed_page_count == 2
    assert capture.repeated_page_request_count == 0


def test_different_query_signature_start_zero_is_new_query_not_continuation():
    events = []
    capture = KfhGate5APassiveCapture(on_new_query=lambda: events.append("NEW"))
    _arm_capture(capture)
    add_first_page(capture)
    capture.observe_request_frame(
        request(0, "PRIVATE-NEW-QUERY", from_date="20250101")
    )
    assert events == ["NEW"]
    assert capture.new_query_detected is True
    assert capture.current_capture_attempt_stopped is True
    assert capture.completed_page_count == 1


def test_different_query_cannot_be_merged_into_current_attempt():
    capture = KfhGate5APassiveCapture()
    _arm_capture(capture)
    add_first_page(capture)
    capture.observe_request_frame(
        request(0, "PRIVATE-NEW-QUERY", to_date="20260901")
    )
    record = capture.build_sanitized_attempt_record()
    assert len(record["pages"]) == 1
    assert record["result"]["newQueryDetected"] is True
    assert record["result"]["currentCaptureAttemptStopped"] is True


def test_real_request_ids_exist_only_while_correlation_is_outstanding():
    capture = KfhGate5APassiveCapture()
    _arm_capture(capture)
    capture.observe_request_frame(request(0, "PRIVATE-REQUEST-1"))
    assert capture.raw_correlation_identifiers_retained is True
    capture.observe_response_frame(
        response("PRIVATE-REQUEST-1", references=["PRIVATE-REF"], has_next=0)
    )
    assert capture.raw_correlation_identifiers_retained is False
    capture.build_sanitized_attempt_record()
    assert capture.raw_correlation_identifiers_retained is False


def test_no_raw_frames_are_persisted():
    capture = KfhGate5APassiveCapture()
    _arm_capture(capture)
    add_first_page(capture)
    serialized = json.dumps(capture.build_sanitized_attempt_record())
    assert '"HED"' not in serialized
    assert '"DAT"' not in serialized
    assert "PRIVATE-REQUEST" not in serialized
    assert "PRIVATE-ACCOUNT" not in serialized


def test_no_financial_values_are_persisted():
    capture = KfhGate5APassiveCapture()
    _arm_capture(capture)
    add_first_page(capture)
    serialized = json.dumps(capture.build_sanitized_attempt_record())
    assert "PRIVATE-FINANCIAL-VALUE" not in serialized
    assert "PRIVATE-REF" not in serialized
    assert "amount" not in serialized


def test_no_websocket_sending_surface_was_introduced():
    parameters = set(inspect.signature(route_gate5a_outbound_frame).parameters)
    assert parameters == {"frame", "on_statement_request_frame"}
    assert "send" not in parameters


def test_kfh_auto_sync_remains_false():
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
