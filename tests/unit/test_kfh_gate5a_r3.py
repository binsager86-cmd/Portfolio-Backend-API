"""Gate 5A-R3 pagination-domain proof tests using synthetic sanitized values."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from local_connector.kfh_gate5a.capture import (
    REQUEST_PAGE_CAPACITY,
    RETURNED_CASH_PLUS_UNSETTLED_COUNT,
    Gate5ACaptureError,
    KfhGate5APassiveCapture,
)


def request(start_seq: int, *, request_id: str = "PRIVATE-CORRELATION-9981") -> str:
    return json.dumps(
        {
            "HED": {"msgGrp": 2, "msgTyp": 7, "chnlId": 30},
            "DAT": {
                "secAccNum": "SYNTHETIC-ACCOUNT",
                "frmDate": "20250101",
                "toDate": "20260831",
                "sortMode": 0,
                "startSeq": start_seq,
                "totalNoRec": 20,
                "unqReqId": request_id,
            },
        }
    )


def response(
    request_id: str,
    *,
    count: int,
    has_next: int,
    offset: int,
    total: int = 69,
    unsettled_count: int = 0,
) -> str:
    first_date = date(2026, 8, 31)
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
                        "date": (first_date - timedelta(days=offset + index)).strftime(
                            "%Y%m%d"
                        ),
                        "trnsRef": f"SYNTHETIC-REF-{offset + index}",
                        "amount": "NOT-RETAINED",
                    }
                    for index in range(count)
                ],
                "unsettledCashLogs": [
                    {
                        "date": "20260101",
                        "trnsRef": f"SYNTHETIC-UNSETTLED-{offset}-{index}",
                    }
                    for index in range(unsettled_count)
                ],
            },
        }
    )


def real_shape_fixture() -> dict:
    capture = KfhGate5APassiveCapture()
    capture.activate_after_gate3a_ready()
    shapes = (
        (0, 19, 1, 0),
        (20, 20, 1, 19),
        (40, 7, 1, 39),
        (60, 0, 0, 46),
    )
    for start_seq, count, has_next, offset in shapes:
        capture.observe_request_frame(request(start_seq))
        capture.observe_response_frame(
            response(
                "PRIVATE-CORRELATION-9981",
                count=count,
                has_next=has_next,
                offset=offset,
            )
        )
    return capture.build_sanitized_fixture()


def test_real_shape_uniquely_verifies_capacity_cursor_and_window_coverage():
    fixture = real_shape_fixture()
    proof = fixture["proof"]

    assert proof["startSeqProgression"] == [0, 20, 40, 60]
    assert proof["requestPageCapacitySequence"] == [20, 20, 20, 20]
    assert proof["cashLogsCounts"] == [19, 20, 7, 0]
    assert proof["unsettledCashLogsCounts"] == [0, 0, 0, 0]
    assert proof["isNxtPagAvailSequence"] == [1, 1, 1, 0]
    assert proof["responseTotalMatchingRecordsSequence"] == [69, 69, 69, 69]
    assert proof["compatibleCursorRules"] == [REQUEST_PAGE_CAPACITY]
    assert proof["cursorRuleStatus"] == "VERIFIED"
    assert proof["paginationRuleVerified"] is True
    assert proof["expectedPaginationWindows"] == 4
    assert proof["observedPaginationWindows"] == 4
    assert proof["paginationWindowCoverageComplete"] is True
    assert proof["gate5aPass"] is True


def test_response_total_is_pagination_domain_not_returned_cash_log_total():
    proof = real_shape_fixture()["proof"]
    assert proof["responseTotalStable"] is True
    assert proof["responseTotalValue"] == 69
    assert proof["cashLogsCombinedCount"] == 46
    assert proof["unsettledCombinedCount"] == 0
    assert proof["cashLogsEqualResponseTotal"] is False
    assert proof["cashPlusUnsettledEqualResponseTotal"] is False
    assert proof["responseTotalSemantics"] == "PAGINATION_DOMAIN_TOTAL"
    assert proof["gate5aPass"] is True


def test_empty_fourth_window_is_valid_terminal_page_and_page_numbers_are_not_used():
    fixture = real_shape_fixture()
    final_response = fixture["pages"][3]["response"]["DAT"]
    assert final_response["cashLogsCount"] == 0
    assert final_response["isNxtPagAvail"] == 0
    assert final_response["pageNo"] == -1
    assert final_response["totalPages"] == -1
    assert fixture["proof"]["paginationWindowCoverageComplete"] is True


def test_reused_request_id_is_observed_without_retaining_actual_value():
    fixture = real_shape_fixture()
    proof = fixture["proof"]
    assert proof["requestIdBehavior"] == "REUSED_ACROSS_PAGES"
    assert proof["requestIdsUniquePerPage"] is False
    assert proof["requestIdsEchoedByResponse"] is True
    assert "PRIVATE-CORRELATION-9981" not in json.dumps(fixture)


def test_real_shape_preserves_newest_order_and_no_cross_page_duplicates():
    proof = real_shape_fixture()["proof"]
    assert proof["sortOrder"] == "NEWEST_TO_OLDEST"
    assert proof["duplicateReferencesAcrossPages"] == []
    assert len(proof["boundaryTransitions"]) == 3
    assert all(
        transition["duplicateAtBoundary"] is False
        for transition in proof["boundaryTransitions"]
    )


def test_generic_distinguishing_logic_can_eliminate_candidates_via_unsettled_count():
    capture = KfhGate5APassiveCapture()
    capture.activate_after_gate3a_ready()
    capture.observe_request_frame(request(0, request_id="ONE"))
    capture.observe_response_frame(
        response(
            "ONE",
            count=20,
            has_next=1,
            offset=0,
            total=23,
            unsettled_count=2,
        )
    )
    capture.observe_request_frame(request(22, request_id="TWO"))
    capture.observe_response_frame(
        response("TWO", count=1, has_next=0, offset=22, total=23)
    )

    proof = capture.build_sanitized_fixture()["proof"]
    assert proof["distinguishingContinuationObserved"] is True
    assert proof["compatibleCursorRules"] == [RETURNED_CASH_PLUS_UNSETTLED_COUNT]
    assert proof["paginationRuleVerified"] is True
    assert proof["paginationWindowCoverageComplete"] is False
    assert proof["gate5aPass"] is False


def test_uncorrelated_retry_prevents_formal_pagination_fixture():
    capture = KfhGate5APassiveCapture()
    capture.activate_after_gate3a_ready()
    capture.observe_request_frame(request(0, request_id="PAGE-1"))
    capture.observe_response_frame(
        response("PAGE-1", count=20, has_next=1, offset=0, total=40)
    )
    capture.observe_request_frame(request(0, request_id="RETRY-PENDING"))
    capture.observe_request_frame(request(20, request_id="PAGE-2"))
    capture.observe_response_frame(
        response("PAGE-2", count=20, has_next=0, offset=20, total=40)
    )
    with pytest.raises(Gate5ACaptureError, match="repeated-page request"):
        capture.build_sanitized_fixture()


def test_pagination_domain_total_label_requires_complete_window_coverage():
    capture = KfhGate5APassiveCapture()
    capture.activate_after_gate3a_ready()
    capture.observe_request_frame(request(0, request_id="PAGE-1"))
    capture.observe_response_frame(
        response("PAGE-1", count=19, has_next=1, offset=0, total=69)
    )
    capture.observe_request_frame(request(20, request_id="PAGE-2"))
    capture.observe_response_frame(
        response("PAGE-2", count=20, has_next=0, offset=19, total=69)
    )

    proof = capture.build_sanitized_fixture()["proof"]
    assert proof["responseTotalStable"] is True
    assert proof["requestPageCapacityStable"] is True
    assert proof["paginationWindowCoverageComplete"] is False
    assert proof["responseTotalSemantics"] == "NOT_PROVEN"
    assert proof["gate5aPass"] is False


def test_required_r3_proof_fields_are_present():
    proof = real_shape_fixture()["proof"]
    required = {
        "startSeqProgression",
        "requestPageCapacitySequence",
        "cashLogsCounts",
        "unsettledCashLogsCounts",
        "isNxtPagAvailSequence",
        "responseTotalMatchingRecordsSequence",
        "expectedPaginationWindows",
        "observedPaginationWindows",
        "paginationWindowCoverageComplete",
        "compatibleCursorRules",
        "cursorRuleStatus",
        "paginationRuleVerified",
        "requestIdBehavior",
        "sortOrder",
        "duplicateReferencesAcrossPages",
        "boundaryTransitions",
        "captureValid",
        "gate5aPass",
    }
    assert required <= set(proof)
