"""Gate 5A evidence-model tests; synthetic values make no production claim."""

from __future__ import annotations

import json

import pytest

from local_connector.kfh_gate5a.capture import (
    REQUEST_PAGE_CAPACITY,
    RETURNED_CASH_LOG_COUNT,
    RETURNED_CASH_PLUS_UNSETTLED_COUNT,
    Gate5ACaptureError,
    KfhGate5APassiveCapture,
    audit_sanitized_fixture,
    write_fixture,
)


def request(start_seq: int, request_id: str, *, capacity: int) -> str:
    return json.dumps(
        {
            "HED": {"msgGrp": 2, "msgTyp": 7, "chnlId": 30},
            "DAT": {
                "secAccNum": "FAKE-ACCOUNT-NOT-RETAINED",
                "frmDate": "20260101",
                "toDate": "20260831",
                "sortMode": 1,
                "startSeq": start_seq,
                "totalNoRec": capacity,
                "unqReqId": request_id,
            },
        }
    )


def response(
    page: int,
    request_id: str,
    references: list[str],
    has_next: int,
    *,
    response_total: int,
    unsettled_count: int = 0,
) -> str:
    base_day = 30 - ((page - 1) * 10)
    return json.dumps(
        {
            "HED": {"msgGrp": 2, "msgTyp": 107, "resSts": 1, "errCode": 0},
            "DAT": {
                "totalNoRec": response_total,
                "isNxtPagAvail": has_next,
                "pageNo": -1,
                "totalPages": -1,
                "unqReqId": request_id,
                "cashLogs": [
                    {
                        "date": f"202608{base_day - index:02d}",
                        "trnsRef": reference,
                        "amount": "SECRET-FINANCIAL-VALUE",
                    }
                    for index, reference in enumerate(references)
                ],
                "unsettledCashLogs": [
                    {"date": f"202607{28 - index:02d}", "trnsRef": f"UNSETTLED-{page}-{index}"}
                    for index in range(unsettled_count)
                ],
            },
        }
    )


def add_page(
    capture: KfhGate5APassiveCapture,
    page: int,
    start_seq: int,
    references: list[str],
    has_next: int,
    *,
    capacity: int,
    response_total: int,
    unsettled_count: int = 0,
) -> None:
    request_id = f"REAL-REQUEST-{page}"
    capture.observe_request_frame(request(start_seq, request_id, capacity=capacity))
    capture.observe_response_frame(
        response(
            page,
            request_id,
            references,
            has_next,
            response_total=response_total,
            unsettled_count=unsettled_count,
        )
    )


def test_two_page_capture_is_retained_as_valid_but_cursor_rule_remains_equivalent(tmp_path):
    capture = KfhGate5APassiveCapture()
    capture.activate_after_gate3a_ready()
    add_page(capture, 1, 0, ["REAL-A", "REAL-B"], 1, capacity=2, response_total=3)
    add_page(capture, 2, 2, ["REAL-C"], 0, capacity=2, response_total=3)

    fixture = capture.build_sanitized_fixture()
    proof = fixture["proof"]
    assert proof["captureValid"] is True
    assert proof["paginationBehaviorObserved"] is True
    assert proof["cursorRuleStatus"] == "OBSERVATIONALLY_EQUIVALENT"
    assert proof["paginationRuleVerified"] is False
    assert proof["compatibleCursorRules"] == [
        RETURNED_CASH_LOG_COUNT,
        REQUEST_PAGE_CAPACITY,
        RETURNED_CASH_PLUS_UNSETTLED_COUNT,
    ]
    assert proof["gate5aPass"] is False
    fixture_path = tmp_path / "cash_statement_pagination_real_redacted_evidence_v1.json"
    assert len(write_fixture(fixture_path, fixture)) == 64
    assert fixture_path.exists()


def test_three_pages_do_not_falsely_prove_returned_count_when_capacity_is_equivalent():
    capture = KfhGate5APassiveCapture()
    capture.activate_after_gate3a_ready()
    add_page(capture, 1, 0, [f"P1-{index}" for index in range(20)], 1, capacity=20, response_total=47)
    add_page(capture, 2, 20, [f"P2-{index}" for index in range(20)], 1, capacity=20, response_total=47)
    add_page(capture, 3, 40, [f"P3-{index}" for index in range(7)], 0, capacity=20, response_total=47)

    proof = capture.build_sanitized_fixture()["proof"]
    assert proof["compatibleCursorRules"] == [
        RETURNED_CASH_LOG_COUNT,
        REQUEST_PAGE_CAPACITY,
        RETURNED_CASH_PLUS_UNSETTLED_COUNT,
    ]
    assert proof["cursorRuleStatus"] == "OBSERVATIONALLY_EQUIVALENT"
    assert proof["paginationRuleVerified"] is False


def test_distinguishing_nonfinal_continuation_can_uniquely_verify_cursor_rule():
    capture = KfhGate5APassiveCapture()
    capture.activate_after_gate3a_ready()
    add_page(
        capture,
        1,
        0,
        [f"P1-{index}" for index in range(7)],
        1,
        capacity=20,
        response_total=8,
        unsettled_count=2,
    )
    add_page(capture, 2, 7, ["P2-0"], 0, capacity=20, response_total=8)

    proof = capture.build_sanitized_fixture()["proof"]
    assert proof["compatibleCursorRules"] == [RETURNED_CASH_LOG_COUNT]
    assert proof["distinguishingContinuationObserved"] is True
    assert proof["cursorRuleStatus"] == "VERIFIED"
    assert proof["paginationRuleVerified"] is True


def test_request_capacity_and_response_total_are_distinct_and_unsettled_empty_is_ambiguous():
    capture = KfhGate5APassiveCapture()
    capture.activate_after_gate3a_ready()
    add_page(capture, 1, 0, ["A", "B"], 1, capacity=20, response_total=3)
    add_page(capture, 2, 2, ["C"], 0, capacity=20, response_total=3)

    fixture = capture.build_sanitized_fixture()
    assert fixture["pages"][0]["request"]["DAT"]["requestPageCapacity"] == 20
    assert fixture["pages"][0]["response"]["DAT"]["responseTotalMatchingRecords"] == 3
    assert fixture["proof"]["requestPageCapacitySequence"] == [20, 20]
    assert fixture["proof"]["responseTotalMatchingRecordsSequence"] == [3, 3]
    assert fixture["proof"]["unsettledTotalSemantics"] == "NOT_DISTINGUISHABLE_FROM_CAPTURE"


def test_wrong_correlation_and_interrupted_capture_fail_closed():
    capture = KfhGate5APassiveCapture()
    capture.activate_after_gate3a_ready()
    capture.observe_request_frame(request(0, "REQ-1", capacity=20))
    capture.observe_response_frame(
        response(1, "WRONG", ["R1"], 1, response_total=2)
    )
    with pytest.raises(Gate5ACaptureError, match="outstanding request ID"):
        capture.build_sanitized_fixture()

    interrupted = KfhGate5APassiveCapture()
    interrupted.activate_after_gate3a_ready()
    add_page(interrupted, 1, 0, ["R1"], 1, capacity=20, response_total=2)
    interrupted.observe_request_frame(request(1, "REQ-2", capacity=20))
    with pytest.raises(Gate5ACaptureError, match="PAGINATION_INTERRUPTED"):
        interrupted.build_sanitized_fixture()


def test_frames_before_gate3a_ready_are_not_observed():
    capture = KfhGate5APassiveCapture()
    capture.observe_request_frame(request(0, "AUTH-PHASE-REQUEST", capacity=20))
    capture.activate_after_gate3a_ready()
    with pytest.raises(Gate5ACaptureError, match="at least two"):
        capture.build_sanitized_fixture()


def test_security_audit_rejects_secret_fields_and_raw_identifiers():
    capture = KfhGate5APassiveCapture()
    capture.activate_after_gate3a_ready()
    add_page(capture, 1, 0, ["REAL-A", "REAL-B"], 1, capacity=2, response_total=3)
    add_page(capture, 2, 2, ["REAL-C"], 0, capacity=2, response_total=3)
    fixture = capture.build_sanitized_fixture()
    serialized = json.dumps(fixture)
    for secret in ("FAKE-ACCOUNT", "REAL-REQUEST", "REAL-A", "SECRET-FINANCIAL-VALUE"):
        assert secret not in serialized

    fixture["pages"][0]["request"]["DAT"]["password"] = "MUST-REJECT"
    with pytest.raises(Gate5ACaptureError, match="secret field"):
        audit_sanitized_fixture(fixture)


def test_reused_request_id_is_observed_without_forcing_unique_behavior():
    capture = KfhGate5APassiveCapture()
    capture.activate_after_gate3a_ready()
    capture.observe_request_frame(request(0, "REUSED", capacity=2))
    capture.observe_response_frame(response(1, "REUSED", ["A", "B"], 1, response_total=3))
    capture.observe_request_frame(request(2, "REUSED", capacity=2))
    capture.observe_response_frame(response(2, "REUSED", ["C"], 0, response_total=3))

    proof = capture.build_sanitized_fixture()["proof"]
    assert proof["requestIdBehavior"] == "REUSED_ACROSS_PAGES"
    assert proof["requestIdsUniquePerPage"] is False
    assert proof["requestIdsEchoedByResponse"] is True


def test_completed_page_repeat_is_retry_but_ambiguous_outstanding_id_fails_closed():
    repeated_cursor = KfhGate5APassiveCapture()
    repeated_cursor.activate_after_gate3a_ready()
    add_page(repeated_cursor, 1, 0, ["A"], 1, capacity=2, response_total=2)
    repeated_cursor.observe_request_frame(request(0, "REQ-2", capacity=2))
    assert repeated_cursor.completed_page_count == 1
    assert repeated_cursor.repeated_page_request_count == 1

    ambiguous_id = KfhGate5APassiveCapture()
    ambiguous_id.activate_after_gate3a_ready()
    ambiguous_id.observe_request_frame(request(0, "SAME", capacity=2))
    ambiguous_id.observe_request_frame(request(2, "SAME", capacity=2))
    with pytest.raises(Gate5ACaptureError, match="outstanding"):
        ambiguous_id.build_sanitized_fixture()


def test_maximum_page_and_record_guards_fail_closed():
    pages = KfhGate5APassiveCapture(max_pages=2)
    pages.activate_after_gate3a_ready()
    add_page(pages, 1, 0, ["A"], 1, capacity=1, response_total=3)
    add_page(pages, 2, 1, ["B"], 1, capacity=1, response_total=3)
    pages.observe_request_frame(request(2, "REQ-3", capacity=1))
    with pytest.raises(Gate5ACaptureError, match="maximum-page"):
        pages.build_sanitized_fixture()

    records = KfhGate5APassiveCapture(max_records=2)
    records.activate_after_gate3a_ready()
    records.observe_request_frame(request(0, "REQ-1", capacity=3))
    records.observe_response_frame(
        response(1, "REQ-1", ["A", "B", "C"], 0, response_total=3)
    )
    with pytest.raises(Gate5ACaptureError, match="maximum-record"):
        records.build_sanitized_fixture()


def test_cross_page_duplicate_is_reported_without_claiming_boundary_completeness():
    capture = KfhGate5APassiveCapture()
    capture.activate_after_gate3a_ready()
    add_page(capture, 1, 0, ["A", "B"], 1, capacity=2, response_total=4)
    add_page(capture, 2, 2, ["B", "C"], 0, capacity=2, response_total=4)

    proof = capture.build_sanitized_fixture()["proof"]
    assert proof["duplicateReferencesAcrossPages"] == ["TX_REF_0002"]
    assert proof["boundaryTransitions"][0]["duplicateAtBoundary"] is True
    assert proof["boundaryAnalysis"] == "NOT_PROVEN"
    assert proof["gate5aPass"] is False


def test_fixture_audit_rejects_unapproved_record_fields():
    capture = KfhGate5APassiveCapture()
    capture.activate_after_gate3a_ready()
    add_page(capture, 1, 0, ["A", "B"], 1, capacity=2, response_total=3)
    add_page(capture, 2, 2, ["C"], 0, capacity=2, response_total=3)
    fixture = capture.build_sanitized_fixture()
    fixture["pages"][0]["response"]["DAT"]["records"][0]["amount"] = "1.000"
    with pytest.raises(Gate5ACaptureError, match="record fields"):
        audit_sanitized_fixture(fixture)
