"""Regression lock for the owner-generated sanitized Gate 5A live fixture."""

from __future__ import annotations

import json
from pathlib import Path

from local_connector.kfh_gate5a.capture import (
    REQUEST_PAGE_CAPACITY,
    audit_sanitized_fixture,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "kfh"
    / "cash_statement_pagination_real_redacted_evidence_v1.json"
)


def load_fixture() -> dict:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    audit_sanitized_fixture(fixture)
    return fixture


def test_owner_live_fixture_satisfies_revised_gate5a_proof():
    fixture = load_fixture()
    proof = fixture["proof"]

    assert fixture["fixtureType"] == (
        "KFH_CASH_STATEMENT_PAGINATION_REAL_REDACTED_EVIDENCE_V1"
    )
    assert fixture["captureMethod"] == "OWNER_ARMED_KFH_UI_VIA_GATE_3A_R1"
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
    assert proof["responseTotalSemantics"] == "PAGINATION_DOMAIN_TOTAL"
    assert proof["requestIdBehavior"] == "REUSED_ACROSS_PAGES"
    assert proof["requestIdsUniquePerPage"] is False
    assert proof["requestIdsEchoedByResponse"] is True
    assert proof["sortOrder"] == "NEWEST_TO_OLDEST"
    assert proof["duplicateReferencesAcrossPages"] == []
    assert proof["captureValid"] is True
    assert proof["gate5aPass"] is True


def test_owner_live_fixture_contains_no_raw_private_or_financial_payload():
    fixture = load_fixture()
    serialized = json.dumps(fixture)
    forbidden_keys = {
        "username",
        "password",
        "otp",
        "sessionId",
        "userId",
        "token",
        "cookie",
        "amount",
        "balance",
        "orderNumber",
        "particulars",
        "rawFrame",
        "rawWebSocketPayload",
    }
    assert not any(f'"{key}"' in serialized for key in forbidden_keys)
    assert fixture["security"]["authenticationDataRetained"] is False
    assert fixture["security"]["completeFinancialPayloadsRetained"] is False
    assert all(
        page["request"]["DAT"]["secAccNum"] == "<REDACTED_ACCOUNT>"
        for page in fixture["pages"]
    )
    assert all(
        page["request"]["DAT"]["unqReqId"].startswith(
            "<REDACTED_REQUEST_ID_PAGE_"
        )
        for page in fixture["pages"]
    )
