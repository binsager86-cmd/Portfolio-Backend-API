"""Build and persist only sanitized Gate 5B-L1 live-read evidence."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

EVIDENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "kfh"
    / "kfh_gate5b_live_read_redacted_evidence_v1.json"
)

REQUIRED_LIVE_FIELDS = {
    "requestStartSeqProgression",
    "requestPageCapacity",
    "responseCashLogsCounts",
    "responseUnsettledCounts",
    "isNxtPagAvailSequence",
    "responseTotalSequence",
    "correlationStrategy",
    "allResponsesCorrelated",
    "oneOutstandingRequestMaximum",
    "finalResponseObserved",
    "finalIsNextPageAvailable",
    "partialRead",
    "financialWritesPerformed",
}


def build_live_evidence(
    live: dict[str, Any],
    *,
    gate3a_ready: bool,
    browser_closed_successfully: bool,
    new_run_restored_session: bool,
) -> dict[str, Any]:
    if set(live) != REQUIRED_LIVE_FIELDS:
        raise ValueError("Gate 5B-L1 sanitized evidence fields rejected")
    starts = live["requestStartSeqProgression"]
    capacity = live["requestPageCapacity"]
    continuations = live["isNxtPagAvailSequence"]
    counts = live["responseCashLogsCounts"]
    unsettled = live["responseUnsettledCounts"]
    totals = live["responseTotalSequence"]
    if not all(isinstance(value, list) for value in (starts, continuations, counts, unsettled, totals)):
        raise ValueError("Gate 5B-L1 evidence sequences rejected")
    if not starts or not (
        len(starts) == len(continuations) == len(counts) == len(unsettled) == len(totals)
    ):
        raise ValueError("Gate 5B-L1 evidence sequence lengths rejected")
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise ValueError("Gate 5B-L1 request capacity rejected")
    for sequence in (starts, continuations, counts, unsettled, totals):
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in sequence
        ):
            raise ValueError("Gate 5B-L1 numeric evidence rejected")
    expected_starts = [index * capacity for index in range(len(starts))]
    progression_valid = capacity == 20 and starts == expected_starts
    continuation_valid = continuations[:-1] == [1] * (len(starts) - 1) and continuations[-1] == 0
    live_read_pass = bool(
        gate3a_ready
        and progression_valid
        and continuation_valid
        and live["correlationStrategy"] == "FRESH_UNIQUE_OPAQUE_ID_PER_PAGE"
        and live["allResponsesCorrelated"] is True
        and live["oneOutstandingRequestMaximum"] == 1
        and live["finalResponseObserved"] is True
        and live["finalIsNextPageAvailable"] == 0
        and live["partialRead"] is False
        and live["financialWritesPerformed"] == 0
        and browser_closed_successfully
        and new_run_restored_session is False
    )
    return {
        "fixtureType": "KFH_GATE5B_LIVE_READ_REDACTED_EVIDENCE_V1",
        "captureMethod": "OWNER_CONTROLLED_SAHAM_AUTOMATIC_READ_VIA_GATE3A_R1",
        "evidenceDate": date.today().isoformat(),
        "gate3aReady": gate3a_ready,
        "requestStartSeqProgression": starts,
        "requestPageCapacity": capacity,
        "responseCashLogsCounts": counts,
        "responseUnsettledCounts": unsettled,
        "isNxtPagAvailSequence": continuations,
        "responseTotalSequence": totals,
        "correlationStrategy": live["correlationStrategy"],
        "allResponsesCorrelated": live["allResponsesCorrelated"],
        "oneOutstandingRequestMaximum": live["oneOutstandingRequestMaximum"],
        "finalResponseObserved": live["finalResponseObserved"],
        "finalIsNextPageAvailable": live["finalIsNextPageAvailable"],
        "partialRead": live["partialRead"],
        "financialWritesPerformed": live["financialWritesPerformed"],
        "browserClosedSuccessfully": browser_closed_successfully,
        "newRunRestoredSession": new_run_restored_session,
        "liveReadPass": live_read_pass,
        "security": {
            "realIdentifiersDiscarded": True,
            "accountIdentifiersRetained": False,
            "requestIdentifiersRetained": False,
            "sessionOrUserIdentifiersRetained": False,
            "tokensRetained": False,
            "transactionReferencesRetained": False,
            "amountsOrBalancesRetained": False,
            "rawFramesRetained": False,
            "financialWritesPerformed": 0,
        },
    }


def audit_live_evidence(evidence: dict[str, Any]) -> None:
    if evidence.get("fixtureType") != "KFH_GATE5B_LIVE_READ_REDACTED_EVIDENCE_V1":
        raise ValueError("Gate 5B-L1 evidence type rejected")
    if evidence.get("liveReadPass") is not True:
        raise ValueError("Gate 5B-L1 evidence cannot be written for a failed read")
    forbidden = {
        "secaccnum",
        "unqreqid",
        "sessionid",
        "sesnid",
        "userid",
        "username",
        "password",
        "otp",
        "token",
        "cookie",
        "transactionref",
        "particulars",
        "rawframe",
        "ordernumber",
    }

    def audit_keys(value: Any) -> None:
        if isinstance(value, dict):
            if any(str(key).lower() in forbidden for key in value):
                raise ValueError("Gate 5B-L1 evidence contains a forbidden field")
            for child in value.values():
                audit_keys(child)
        elif isinstance(value, list):
            for child in value:
                audit_keys(child)

    audit_keys(evidence)


def write_live_evidence(path: Path, evidence: dict[str, Any]) -> str:
    audit_live_evidence(evidence)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return hashlib.sha256(payload.encode()).hexdigest()
