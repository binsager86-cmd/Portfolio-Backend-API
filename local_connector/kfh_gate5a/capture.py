"""Sanitize KFH-generated Cash Statement pagination frames in memory."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any


class Gate5ACaptureError(RuntimeError):
    """The live evidence is incomplete, unsafe, or cannot prove pagination."""


RETURNED_CASH_LOG_COUNT = "RETURNED_CASH_LOG_COUNT"
REQUEST_PAGE_CAPACITY = "REQUEST_PAGE_CAPACITY"
RETURNED_CASH_PLUS_UNSETTLED_COUNT = "RETURNED_CASH_PLUS_UNSETTLED_COUNT"
CURSOR_RULES = (
    RETURNED_CASH_LOG_COUNT,
    REQUEST_PAGE_CAPACITY,
    RETURNED_CASH_PLUS_UNSETTLED_COUNT,
)

FirstPageCallback = Callable[[dict[str, Any]], None]
RetryCallback = Callable[[int], None]
NewQueryCallback = Callable[[], None]


def _integer(value: Any, name: str, *, non_negative: bool = True) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise Gate5ACaptureError(f"Invalid captured {name}") from error
    if non_negative and parsed < 0:
        raise Gate5ACaptureError(f"Invalid captured {name}")
    return parsed


def _frame_json(frame: str | bytes) -> Any | None:
    if not isinstance(frame, str) or len(frame) > 5_000_000:
        return None
    try:
        return json.loads(frame)
    except (json.JSONDecodeError, TypeError):
        return None


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _message_parts(value: Any, group: int, message_type: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
    for candidate in _walk(value):
        hed = candidate.get("HED")
        dat = candidate.get("DAT")
        if (
            isinstance(hed, dict)
            and isinstance(dat, dict)
            and _maybe_integer(hed.get("msgGrp")) == group
            and _maybe_integer(hed.get("msgTyp")) == message_type
        ):
            return hed, dat

        if _maybe_integer(candidate.get("msgGrp")) != group or _maybe_integer(candidate.get("msgTyp")) != message_type:
            continue
        for envelope_name in ("request", "response"):
            envelope = candidate.get(envelope_name)
            if isinstance(envelope, dict) and isinstance(envelope.get("DAT"), dict):
                return candidate, envelope["DAT"]
        if isinstance(candidate.get("DAT"), dict):
            return candidate, candidate["DAT"]
    return None


def _maybe_integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compact_date(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if len(text) != 8 or not text.isdigit():
        raise Gate5ACaptureError(f"Invalid captured {name}")
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError as error:
        raise Gate5ACaptureError(f"Invalid captured {name}") from error
    return text


def _request_label(page_index: int) -> str:
    return f"<REDACTED_REQUEST_ID_PAGE_{page_index}>"


def _request_page_label(page_index: int) -> str:
    return f"REQUEST_PAGE_{page_index}"


def _date_order_key(value: Any) -> str | None:
    text = str(value or "").strip()
    digits = "".join(character for character in text if character.isdigit())
    return digits if len(digits) >= 8 else None


@dataclass(frozen=True, slots=True)
class QuerySignature:
    frm_date: str
    to_date: str
    sort_mode: int
    request_page_capacity: int


@dataclass(slots=True)
class _PendingPage:
    real_request_id: str
    request_id_digest: bytes
    public_request: dict[str, Any]
    public_response: dict[str, Any] | None = None


@dataclass(slots=True)
class _PendingRetry:
    real_request_id: str
    start_seq: int
    response_correlated: bool = False


class KfhGate5APassiveCapture:
    """Records only redacted proof fields and never constructs or sends requests."""

    def __init__(
        self,
        *,
        max_pages: int = 100,
        max_records: int = 10_000,
        on_first_page: FirstPageCallback | None = None,
        on_retry: RetryCallback | None = None,
        on_new_query: NewQueryCallback | None = None,
    ) -> None:
        if max_pages < 2 or max_records < 1:
            raise ValueError("Invalid Gate 5A capture limits")
        self.__active = False
        self.__pages: list[_PendingPage] = []
        self.__reference_labels: dict[str, str] = {}
        self.__max_pages = max_pages
        self.__max_records = max_records
        self.__capture_error: Gate5ACaptureError | None = None
        self.__on_first_page = on_first_page
        self.__on_retry = on_retry
        self.__on_new_query = on_new_query
        self.__first_page_summary: dict[str, Any] | None = None
        self.__query_signature: QuerySignature | None = None
        self.__retry_requests: list[_PendingRetry] = []
        self.__repeated_page_request_count = 0
        self.__new_query_detected = False
        self.__current_capture_attempt_stopped = False
        self.__request_id_salt = secrets.token_bytes(32)

    @property
    def active(self) -> bool:
        return self.__active

    @property
    def completed_page_count(self) -> int:
        return sum(page.public_response is not None for page in self.__pages)

    @property
    def first_page_summary(self) -> dict[str, Any] | None:
        if self.__first_page_summary is None:
            return None
        return dict(self.__first_page_summary)

    @property
    def repeated_page_request_count(self) -> int:
        return self.__repeated_page_request_count

    @property
    def new_query_detected(self) -> bool:
        return self.__new_query_detected

    @property
    def current_capture_attempt_stopped(self) -> bool:
        return self.__current_capture_attempt_stopped

    @property
    def raw_correlation_identifiers_retained(self) -> bool:
        return any(page.real_request_id for page in self.__pages) or any(
            retry.real_request_id for retry in self.__retry_requests
        )

    def activate_after_gate3a_ready(self) -> None:
        self.__active = True

    def observe_request_frame(self, frame: str | bytes) -> None:
        if self.__capture_error:
            return
        try:
            self.__observe_request_frame(frame)
        except Gate5ACaptureError as error:
            self.__capture_error = error

    def __observe_request_frame(self, frame: str | bytes) -> None:
        if not self.__active:
            return
        decoded = _frame_json(frame)
        parts = _message_parts(decoded, 2, 7) if decoded is not None else None
        if not parts:
            return
        hed, dat = parts
        request_id = str(dat.get("unqReqId", "")).strip()
        if not request_id:
            raise Gate5ACaptureError("Captured Cash Statement request has no unqReqId")
        if any(
            page.real_request_id == request_id and page.public_response is None
            for page in self.__pages
        ) or any(
            retry.real_request_id == request_id and not retry.response_correlated
            for retry in self.__retry_requests
        ):
            raise Gate5ACaptureError(
                "KFH reused a request ID while an earlier request was outstanding"
            )
        frm_date = _compact_date(dat.get("frmDate"), "request frmDate")
        to_date = _compact_date(dat.get("toDate"), "request toDate")
        sort_mode = _integer(dat.get("sortMode"), "request sortMode", non_negative=False)
        request_page_capacity = _integer(dat.get("totalNoRec"), "request totalNoRec")
        query_signature = QuerySignature(
            frm_date=frm_date,
            to_date=to_date,
            sort_mode=sort_mode,
            request_page_capacity=request_page_capacity,
        )
        start_seq = _integer(dat.get("startSeq"), "request startSeq")

        if self.__query_signature is None:
            self.__query_signature = query_signature
        elif query_signature != self.__query_signature:
            self.__new_query_detected = True
            self.__current_capture_attempt_stopped = True
            self.__active = False
            if self.__on_new_query is not None:
                self.__on_new_query()
            return

        existing_page = next(
            (
                page
                for page in self.__pages
                if page.public_request["DAT"]["startSeq"] == start_seq
            ),
            None,
        )
        if existing_page is not None:
            if existing_page.public_response is None or any(
                retry.start_seq == start_seq and not retry.response_correlated
                for retry in self.__retry_requests
            ):
                raise Gate5ACaptureError("DUPLICATE_OUTSTANDING_REQUEST")
            self.__retry_requests.append(_PendingRetry(request_id, start_seq))
            self.__repeated_page_request_count += 1
            if self.__first_page_summary is not None:
                self.__first_page_summary["repeatedPageRequestCount"] = (
                    self.__repeated_page_request_count
                )
            if self.__on_retry is not None:
                self.__on_retry(self.__repeated_page_request_count)
            return

        if any(page.public_response is None for page in self.__pages):
            raise Gate5ACaptureError("PAGINATION_REQUEST_WHILE_PREVIOUS_PAGE_OUTSTANDING")
        if len(self.__pages) >= self.__max_pages:
            raise Gate5ACaptureError("Gate 5A capture exceeded the maximum-page guard")
        page_index = len(self.__pages) + 1
        public_request = {
            "HED": {
                "msgGrp": _integer(hed.get("msgGrp"), "request HED.msgGrp"),
                "msgTyp": _integer(hed.get("msgTyp"), "request HED.msgTyp"),
                "chnlId": _integer(hed.get("chnlId"), "request HED.chnlId"),
            },
            "DAT": {
                "secAccNum": "<REDACTED_ACCOUNT>",
                "frmDate": frm_date,
                "toDate": to_date,
                "sortMode": sort_mode,
                "startSeq": start_seq,
                "requestPageCapacity": request_page_capacity,
                "requestPageLabel": _request_page_label(page_index),
                "unqReqId": _request_label(page_index),
            },
        }
        request_id_digest = sha256(
            self.__request_id_salt + request_id.encode("utf-8")
        ).digest()
        self.__pages.append(
            _PendingPage(request_id, request_id_digest, public_request)
        )

    def observe_response_frame(self, frame: str | bytes) -> None:
        if self.__capture_error:
            return
        try:
            self.__observe_response_frame(frame)
        except Gate5ACaptureError as error:
            self.__capture_error = error

    def __observe_response_frame(self, frame: str | bytes) -> None:
        if not self.__active:
            return
        decoded = _frame_json(frame)
        parts = _message_parts(decoded, 2, 107) if decoded is not None else None
        if not parts:
            return
        hed, dat = parts
        response_request_id = str(dat.get("unqReqId", "")).strip()
        retry = next(
            (
                candidate
                for candidate in self.__retry_requests
                if candidate.real_request_id == response_request_id
                and not candidate.response_correlated
            ),
            None,
        )
        if retry is not None:
            cash_logs = dat.get("cashLogs")
            unsettled_logs = dat.get("unsettledCashLogs", [])
            if not isinstance(cash_logs, list) or not isinstance(unsettled_logs, list):
                raise Gate5ACaptureError("Captured retry response logs are invalid")
            if len(cash_logs) + len(unsettled_logs) > self.__max_records:
                raise Gate5ACaptureError("Gate 5A retry exceeded the maximum-record guard")
            retry.response_correlated = True
            retry.real_request_id = ""
            return
        page = next(
            (
                candidate
                for candidate in self.__pages
                if candidate.real_request_id == response_request_id
                and candidate.public_response is None
            ),
            None,
        )
        if page is None:
            raise Gate5ACaptureError("Cash Statement response did not match an outstanding request ID")

        cash_logs = dat.get("cashLogs")
        unsettled_logs = dat.get("unsettledCashLogs", [])
        if not isinstance(cash_logs, list) or not isinstance(unsettled_logs, list):
            raise Gate5ACaptureError("Captured Cash Statement response logs are invalid")
        captured_count = sum(
            candidate.public_response["DAT"]["cashLogsCount"]
            + candidate.public_response["DAT"]["unsettledCashLogsCount"]
            for candidate in self.__pages
            if candidate.public_response is not None
        )
        if captured_count + len(cash_logs) + len(unsettled_logs) > self.__max_records:
            raise Gate5ACaptureError("Gate 5A capture exceeded the maximum-record guard")
        page_index = self.__pages.index(page) + 1
        sanitized_records = [self.__sanitize_record(record) for record in cash_logs]
        page.public_response = {
            "HED": {
                "msgGrp": _integer(hed.get("msgGrp"), "response HED.msgGrp"),
                "msgTyp": _integer(hed.get("msgTyp"), "response HED.msgTyp"),
                "resSts": hed.get("resSts"),
                "errCode": hed.get("errCode"),
            },
            "DAT": {
                "responseTotalMatchingRecords": _integer(
                    dat.get("totalNoRec"), "response totalNoRec"
                ),
                "isNxtPagAvail": _integer(dat.get("isNxtPagAvail"), "response isNxtPagAvail"),
                "pageNo": _integer(dat.get("pageNo"), "response pageNo", non_negative=False),
                "totalPages": _integer(dat.get("totalPages"), "response totalPages", non_negative=False),
                "cashLogsCount": len(cash_logs),
                "unsettledCashLogsCount": len(unsettled_logs),
                "firstTransactionRef": sanitized_records[0]["transactionRef"] if sanitized_records else None,
                "lastTransactionRef": sanitized_records[-1]["transactionRef"] if sanitized_records else None,
                "records": sanitized_records,
                "requestPageLabel": _request_page_label(page_index),
                "unqReqId": _request_label(page_index),
            },
        }
        page.real_request_id = ""
        if page_index == 1:
            self.__first_page_summary = self.__build_first_page_summary(page)
            if self.__on_first_page is not None:
                self.__on_first_page(dict(self.__first_page_summary))

    def __build_first_page_summary(self, page: _PendingPage) -> dict[str, Any]:
        response = page.public_response
        if response is None:
            raise Gate5ACaptureError("First Cash Statement response is incomplete")
        request_dat = page.public_request["DAT"]
        response_dat = response["DAT"]
        has_next = response_dat["isNxtPagAvail"] == 1
        return {
            "firstPageCaptured": True,
            "startSeq": request_dat["startSeq"],
            "requestPageCapacity": request_dat["requestPageCapacity"],
            "sortMode": request_dat["sortMode"],
            "dateRange": f'{request_dat["frmDate"]}..{request_dat["toDate"]}',
            "cashLogsCount": response_dat["cashLogsCount"],
            "unsettledCashLogsCount": response_dat["unsettledCashLogsCount"],
            "responseTotalMatchingRecords": response_dat[
                "responseTotalMatchingRecords"
            ],
            "isNxtPagAvail": response_dat["isNxtPagAvail"],
            "pageNo": response_dat["pageNo"],
            "totalPages": response_dat["totalPages"],
            "requestResponseCorrelated": True,
            "queryCaptureOrigin": "OWNER_ARMED_VIEW_ATTEMPT",
            "repeatedPageRequestCount": self.__repeated_page_request_count,
            "kfhProtocolSaysMorePages": has_next,
            "rangeDisposition": (
                "KFH_PROTOCOL_SAYS_MORE_PAGES"
                if has_next
                else "RANGE_NOT_MULTIPAGE"
            ),
        }

    def __sanitize_record(self, value: Any) -> dict[str, str | None]:
        if not isinstance(value, dict):
            raise Gate5ACaptureError("Captured cashLogs entry is not an object")
        real_reference = str(value.get("trnsRef", "")).strip()
        if not real_reference:
            raise Gate5ACaptureError("Captured cashLogs entry has no trnsRef")
        if real_reference not in self.__reference_labels:
            self.__reference_labels[real_reference] = f"TX_REF_{len(self.__reference_labels) + 1:04d}"
        return {
            "transactionRef": self.__reference_labels[real_reference],
            "date": str(value.get("date", "")).strip() or None,
        }

    def build_sanitized_fixture(self) -> dict[str, Any]:
        try:
            return self.__build_sanitized_fixture()
        finally:
            # Raw request IDs and transaction references are required only for
            # in-memory correlation/equality mapping and are not retained after
            # fixture construction.
            for page in self.__pages:
                page.real_request_id = ""
                page.request_id_digest = b""
            for retry in self.__retry_requests:
                retry.real_request_id = ""
            self.__request_id_salt = b""
            self.__reference_labels.clear()

    def build_sanitized_attempt_record(self) -> dict[str, Any]:
        """Retain safe protocol facts without creating Gate 5A proof."""
        try:
            if self.__capture_error:
                raise self.__capture_error
            completed = [page for page in self.__pages if page.public_response is not None]
            if not completed:
                raise Gate5ACaptureError(
                    "No correlated Cash Statement request/response pair was captured"
                )
            if len(completed) != len(self.__pages):
                raise Gate5ACaptureError(
                    "PAGINATION_INTERRUPTED: not every captured request has a response"
                )
            pages: list[dict[str, Any]] = []
            for index, page in enumerate(completed, 1):
                request_dat = page.public_request["DAT"]
                response = page.public_response
                if response is None:
                    raise Gate5ACaptureError(
                        "PAGINATION_INTERRUPTED: completed page has no response"
                    )
                response_dat = response["DAT"]
                pages.append(
                    {
                        "page": index,
                        "startSeq": request_dat["startSeq"],
                        "requestPageCapacity": request_dat["requestPageCapacity"],
                        "sortMode": request_dat["sortMode"],
                        "dateRange": f'{request_dat["frmDate"]}..{request_dat["toDate"]}',
                        "cashLogsCount": response_dat["cashLogsCount"],
                        "unsettledCashLogsCount": response_dat[
                            "unsettledCashLogsCount"
                        ],
                        "responseTotalMatchingRecords": response_dat[
                            "responseTotalMatchingRecords"
                        ],
                        "isNxtPagAvail": response_dat["isNxtPagAvail"],
                        "pageNo": response_dat["pageNo"],
                        "totalPages": response_dat["totalPages"],
                        "requestResponseCorrelated": True,
                    }
                )
            record = {
                "recordType": "KFH_CASH_STATEMENT_ATTEMPT_REDACTED_V1",
                "evidenceDate": date.today().isoformat(),
                "captureMethod": "OWNER_CONTROLLED_PASSIVE_KFH_UI_VIA_GATE_3A_R1",
                "pages": pages,
                "result": {
                    "captureValid": True,
                    "multiPageObserved": len(pages) >= 2,
                    "gate5aPass": False,
                    "paginationAuthorized": False,
                    "queryCaptureOrigin": "OWNER_ARMED_VIEW_ATTEMPT",
                    "repeatedPageRequestObserved": (
                        self.__repeated_page_request_count > 0
                    ),
                    "repeatedPageRequestCount": self.__repeated_page_request_count,
                    "newQueryDetected": self.__new_query_detected,
                    "currentCaptureAttemptStopped": (
                        self.__current_capture_attempt_stopped
                    ),
                },
                "security": {
                    "identifiersRetained": False,
                    "transactionValuesRetained": False,
                    "cashTotalsRetained": False,
                    "rawFramesRetained": False,
                    "authenticationDataRetained": False,
                },
            }
            audit_sanitized_attempt_record(record)
            return record
        finally:
            for page in self.__pages:
                page.real_request_id = ""
                page.request_id_digest = b""
            for retry in self.__retry_requests:
                retry.real_request_id = ""
            self.__request_id_salt = b""
            self.__reference_labels.clear()

    def __build_sanitized_fixture(self) -> dict[str, Any]:
        if self.__capture_error:
            raise self.__capture_error
        if len(self.__pages) < 2:
            raise Gate5ACaptureError(
                "Gate 5A requires at least two KFH-generated statement pages; "
                f"observed {len(self.__pages)}"
            )
        if any(page.public_response is None for page in self.__pages):
            raise Gate5ACaptureError("PAGINATION_INTERRUPTED: not every captured request has a response")
        if any(not retry.response_correlated for retry in self.__retry_requests):
            raise Gate5ACaptureError(
                "PAGINATION_INTERRUPTED: a repeated-page request has no correlated response"
            )

        request_id_digests = [page.request_id_digest for page in self.__pages]
        pages: list[dict[str, Any]] = []
        for index, page in enumerate(self.__pages, 1):
            response = page.public_response
            if response is None:
                raise Gate5ACaptureError(
                    "PAGINATION_INTERRUPTED: logical page has no response"
                )
            pages.append(
                {"page": index, "request": page.public_request, "response": response}
            )
        starts = [page["request"]["DAT"]["startSeq"] for page in pages]
        counts = [page["response"]["DAT"]["cashLogsCount"] for page in pages]
        unsettled_counts = [page["response"]["DAT"]["unsettledCashLogsCount"] for page in pages]
        continuations = [page["response"]["DAT"]["isNxtPagAvail"] for page in pages]
        request_capacities = [
            page["request"]["DAT"]["requestPageCapacity"] for page in pages
        ]
        response_total_records = [
            page["response"]["DAT"]["responseTotalMatchingRecords"] for page in pages
        ]
        ranges = [
            (page["request"]["DAT"]["frmDate"], page["request"]["DAT"]["toDate"])
            for page in pages
        ]
        sort_modes = [page["request"]["DAT"]["sortMode"] for page in pages]

        if continuations[:-1] != [1] * (len(continuations) - 1) or continuations[-1] != 0:
            raise Gate5ACaptureError("Final-page continuation behavior was not proven")
        if len(set(ranges)) != 1 or len(set(sort_modes)) != 1:
            raise Gate5ACaptureError("Date range or sortMode changed between captured pages")

        compatible_rules = set(CURSOR_RULES)
        distinguishing_transition = False
        for index in range(len(starts) - 1):
            actual_next = starts[index + 1]
            candidates = {
                RETURNED_CASH_LOG_COUNT: starts[index] + counts[index],
                REQUEST_PAGE_CAPACITY: starts[index] + request_capacities[index],
                RETURNED_CASH_PLUS_UNSETTLED_COUNT: (
                    starts[index] + counts[index] + unsettled_counts[index]
                ),
            }
            matching_rules = {
                rule for rule, candidate in candidates.items() if candidate == actual_next
            }
            compatible_rules.intersection_update(matching_rules)
            if (
                continuations[index] == 1
                and len(set(candidates.values())) > 1
                and len(matching_rules) < len(candidates)
            ):
                distinguishing_transition = True

        compatible_cursor_rules = [
            rule for rule in CURSOR_RULES if rule in compatible_rules
        ]
        pagination_rule_verified = (
            len(compatible_cursor_rules) == 1 and distinguishing_transition
        )
        if pagination_rule_verified:
            cursor_rule_status = "VERIFIED"
        elif compatible_cursor_rules:
            cursor_rule_status = "OBSERVATIONALLY_EQUIVALENT"
        else:
            cursor_rule_status = "UNEXPLAINED"

        cash_only_total = sum(counts)
        unsettled_total = sum(unsettled_counts)
        cash_plus_unsettled_total = cash_only_total + unsettled_total
        response_total_stable = len(set(response_total_records)) == 1
        response_total = response_total_records[0] if response_total_stable else None
        cash_only_matches = response_total_stable and response_total == cash_only_total
        cash_plus_unsettled_matches = (
            response_total_stable and response_total == cash_plus_unsettled_total
        )
        if not any(unsettled_counts):
            unsettled_total_semantics = "NOT_DISTINGUISHABLE_FROM_CAPTURE"
        elif cash_only_matches and not cash_plus_unsettled_matches:
            unsettled_total_semantics = "CASH_ONLY_TOTAL"
        elif cash_plus_unsettled_matches and not cash_only_matches:
            unsettled_total_semantics = "CASH_PLUS_UNSETTLED_TOTAL"
        else:
            unsettled_total_semantics = "NO_UNIQUE_MATCH"

        request_capacity_stable = (
            len(set(request_capacities)) == 1 and request_capacities[0] > 0
        )
        stable_capacity = request_capacities[0] if request_capacity_stable else None
        expected_pagination_windows = (
            (response_total + stable_capacity - 1) // stable_capacity
            if response_total_stable
            and response_total is not None
            and stable_capacity is not None
            and response_total > 0
            else None
        )
        expected_start_progression = (
            [stable_capacity * index for index in range(expected_pagination_windows)]
            if stable_capacity is not None and expected_pagination_windows is not None
            else None
        )
        continuation_chain_complete = (
            continuations[:-1] == [1] * (len(continuations) - 1)
            and continuations[-1] == 0
        )
        pagination_window_coverage_complete = (
            response_total_stable
            and request_capacity_stable
            and expected_pagination_windows is not None
            and starts == expected_start_progression
            and len(pages) == expected_pagination_windows
            and continuation_chain_complete
        )
        response_total_semantics = (
            "PAGINATION_DOMAIN_TOTAL"
            if pagination_window_coverage_complete
            else "NOT_PROVEN"
        )

        all_records = [
            record
            for proof_page in pages
            for record in proof_page["response"]["DAT"]["records"]
        ]
        reference_pages: dict[str, set[int]] = {}
        for proof_page in pages:
            for record in proof_page["response"]["DAT"]["records"]:
                reference_pages.setdefault(record["transactionRef"], set()).add(
                    proof_page["page"]
                )
        duplicates = sorted(
            reference
            for reference, page_numbers in reference_pages.items()
            if len(page_numbers) > 1
        )
        boundary_transitions = []
        for index in range(len(pages) - 1):
            current_records = pages[index]["response"]["DAT"]["records"]
            next_records = pages[index + 1]["response"]["DAT"]["records"]
            current_last = current_records[-1]["transactionRef"] if current_records else None
            next_first = next_records[0]["transactionRef"] if next_records else None
            boundary_transitions.append(
                {
                    "fromPage": pages[index]["page"],
                    "toPage": pages[index + 1]["page"],
                    "lastTransactionRef": current_last,
                    "nextFirstTransactionRef": next_first,
                    "duplicateAtBoundary": bool(
                        current_last is not None and current_last == next_first
                    ),
                }
            )
        sort_order = self.__sort_order(all_records)
        capture_valid = True
        pagination_behavior_observed = len(pages) >= 2
        request_ids_unique = len(set(request_id_digests)) == len(request_id_digests)
        gate_pass = (
            capture_valid
            and pagination_rule_verified
            and pagination_window_coverage_complete
            and not duplicates
            and not self.__new_query_detected
            and sort_order in {"NEWEST_TO_OLDEST", "OLDEST_TO_NEWEST"}
        )

        fixture = {
            "fixtureType": "KFH_CASH_STATEMENT_PAGINATION_REAL_REDACTED_EVIDENCE_V1",
            "evidenceDate": date.today().isoformat(),
            "captureMethod": "OWNER_ARMED_KFH_UI_VIA_GATE_3A_R1",
            "confidence": "SANITIZED_REAL_KFH_OBSERVATION",
            "pages": pages,
            "proof": {
                "pagesCaptured": len(pages),
                "startSeqProgression": starts,
                "cashLogsCounts": counts,
                "unsettledCashLogsCounts": unsettled_counts,
                "isNxtPagAvailSequence": continuations,
                "requestPageCapacitySequence": request_capacities,
                "responseTotalMatchingRecordsSequence": response_total_records,
                "compatibleCursorRules": compatible_cursor_rules,
                "cursorRuleStatus": cursor_rule_status,
                "distinguishingContinuationObserved": distinguishing_transition,
                "paginationRuleVerified": pagination_rule_verified,
                "cashOnlyCombinedRecordCount": cash_only_total,
                "cashPlusUnsettledCombinedRecordCount": cash_plus_unsettled_total,
                "cashOnlyTotalMatchesResponse": cash_only_matches,
                "cashPlusUnsettledTotalMatchesResponse": cash_plus_unsettled_matches,
                "responseTotalStable": response_total_stable,
                "responseTotalValue": response_total,
                "cashLogsCombinedCount": cash_only_total,
                "unsettledCombinedCount": unsettled_total,
                "cashLogsEqualResponseTotal": cash_only_matches,
                "cashPlusUnsettledEqualResponseTotal": cash_plus_unsettled_matches,
                "responseTotalSemantics": response_total_semantics,
                "unsettledTotalSemantics": unsettled_total_semantics,
                "requestPageCapacityStable": request_capacity_stable,
                "expectedPaginationWindows": expected_pagination_windows,
                "observedPaginationWindows": len(pages),
                "expectedStartSeqProgression": expected_start_progression,
                "paginationWindowCoverageComplete": (
                    pagination_window_coverage_complete
                ),
                "requestIdBehavior": (
                    "UNIQUE_PER_PAGE" if request_ids_unique else "REUSED_ACROSS_PAGES"
                ),
                "requestIdsUniquePerPage": request_ids_unique,
                "requestIdsEchoedByResponse": True,
                "queryCaptureOrigin": "OWNER_ARMED_VIEW_ATTEMPT",
                "repeatedPageRequestObserved": (
                    self.__repeated_page_request_count > 0
                ),
                "repeatedPageRequestCount": self.__repeated_page_request_count,
                "newQueryDetected": self.__new_query_detected,
                "currentCaptureAttemptStopped": (
                    self.__current_capture_attempt_stopped
                ),
                "sortOrder": sort_order,
                "duplicateReferencesAcrossPages": duplicates,
                "boundaryTransitions": boundary_transitions,
                "boundaryAnalysis": "CHRONOLOGICAL_WITHOUT_CROSS_PAGE_DUPLICATES"
                if sort_order in {"NEWEST_TO_OLDEST", "OLDEST_TO_NEWEST"}
                and not duplicates
                else "NOT_PROVEN",
                "captureValid": capture_valid,
                "paginationBehaviorObserved": pagination_behavior_observed,
                "gate5aPass": gate_pass,
            },
            "security": {
                "account": "<REDACTED_ACCOUNT>",
                "requestIds": "REPLACED_WITH_STABLE_PAGE_LABELS",
                "transactionReferences": "REPLACED_WITH_STABLE_FAKE_IDENTIFIERS",
                "authenticationDataRetained": False,
                "completeFinancialPayloadsRetained": False,
            },
        }
        audit_sanitized_fixture(fixture)
        return fixture

    @staticmethod
    def __sort_order(records: list[dict[str, Any]]) -> str:
        keys = [_date_order_key(record.get("date")) for record in records]
        if len(keys) < 2 or any(key is None for key in keys):
            return "UNDETERMINED"
        values = [key for key in keys if key is not None]
        if len(set(values)) == 1:
            return "UNDETERMINED"
        if all(values[index] >= values[index + 1] for index in range(len(values) - 1)):
            return "NEWEST_TO_OLDEST"
        if all(values[index] <= values[index + 1] for index in range(len(values) - 1)):
            return "OLDEST_TO_NEWEST"
        return "NON_MONOTONIC"


FORBIDDEN_FIXTURE_KEYS = frozenset(
    {
        "username",
        "password",
        "otp",
        "mobile",
        "mobilenumber",
        "sesnid",
        "sessionid",
        "usrid",
        "userid",
        "ssotoken",
        "xapptoken",
        "cookie",
        "cookies",
        "rawframe",
        "rawwebsocketpayload",
        "authenticationframe",
        "inputvalue",
    }
)


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def audit_sanitized_fixture(fixture: dict[str, Any]) -> None:
    """Reject secret-bearing fields and any non-redacted protocol identifiers."""
    for candidate in _walk(fixture):
        for key in candidate:
            if _normalized_key(key) in FORBIDDEN_FIXTURE_KEYS:
                raise Gate5ACaptureError("Security-redaction audit rejected a secret field")
    for page in fixture.get("pages", []):
        if set(page) != {"page", "request", "response"}:
            raise Gate5ACaptureError("Security-redaction audit rejected page fields")
        if set(page.get("request", {})) != {"HED", "DAT"}:
            raise Gate5ACaptureError("Security-redaction audit rejected request fields")
        if set(page.get("response", {})) != {"HED", "DAT"}:
            raise Gate5ACaptureError("Security-redaction audit rejected response fields")
        request_dat = page.get("request", {}).get("DAT", {})
        response_dat = page.get("response", {}).get("DAT", {})
        if set(request_dat) != {
            "secAccNum",
            "frmDate",
            "toDate",
            "sortMode",
            "startSeq",
            "requestPageCapacity",
            "requestPageLabel",
            "unqReqId",
        }:
            raise Gate5ACaptureError("Security-redaction audit rejected request DAT fields")
        if set(response_dat) != {
            "responseTotalMatchingRecords",
            "isNxtPagAvail",
            "pageNo",
            "totalPages",
            "cashLogsCount",
            "unsettledCashLogsCount",
            "firstTransactionRef",
            "lastTransactionRef",
            "records",
            "requestPageLabel",
            "unqReqId",
        }:
            raise Gate5ACaptureError("Security-redaction audit rejected response DAT fields")
        if request_dat.get("secAccNum") != "<REDACTED_ACCOUNT>":
            raise Gate5ACaptureError("Security-redaction audit rejected an account identifier")
        page_number = page.get("page")
        expected_request_label = _request_label(page_number)
        expected_page_label = _request_page_label(page_number)
        if request_dat.get("requestPageLabel") != expected_page_label:
            raise Gate5ACaptureError("Security-redaction audit rejected a request-page label")
        if response_dat.get("requestPageLabel") != expected_page_label:
            raise Gate5ACaptureError("Security-redaction audit rejected a response-page label")
        if request_dat.get("unqReqId") != expected_request_label:
            raise Gate5ACaptureError("Security-redaction audit rejected a request identifier")
        if response_dat.get("unqReqId") != expected_request_label:
            raise Gate5ACaptureError("Security-redaction audit rejected a response identifier")
        for record in response_dat.get("records", []):
            if set(record) != {"transactionRef", "date"}:
                raise Gate5ACaptureError("Security-redaction audit rejected record fields")
            reference = record.get("transactionRef")
            suffix = reference.removeprefix("TX_REF_") if isinstance(reference, str) else ""
            if len(suffix) != 4 or not suffix.isdigit():
                raise Gate5ACaptureError(
                    "Security-redaction audit rejected a transaction reference"
                )
        for boundary_name in ("firstTransactionRef", "lastTransactionRef"):
            reference = response_dat.get(boundary_name)
            if reference is None:
                continue
            suffix = reference.removeprefix("TX_REF_") if isinstance(reference, str) else ""
            if len(suffix) != 4 or not suffix.isdigit():
                raise Gate5ACaptureError(
                    "Security-redaction audit rejected a boundary transaction reference"
                )


ATTEMPT_PAGE_FIELDS = frozenset(
    {
        "page",
        "startSeq",
        "requestPageCapacity",
        "sortMode",
        "dateRange",
        "cashLogsCount",
        "unsettledCashLogsCount",
        "responseTotalMatchingRecords",
        "isNxtPagAvail",
        "pageNo",
        "totalPages",
        "requestResponseCorrelated",
    }
)


def audit_sanitized_attempt_record(record: dict[str, Any]) -> None:
    """Allow only non-identifying counts and paging facts in R2 attempts."""
    if set(record) != {
        "recordType",
        "evidenceDate",
        "captureMethod",
        "pages",
        "result",
        "security",
    }:
        raise Gate5ACaptureError("Attempt-record audit rejected top-level fields")
    for candidate in _walk(record):
        for key in candidate:
            if _normalized_key(key) in FORBIDDEN_FIXTURE_KEYS:
                raise Gate5ACaptureError("Attempt-record audit rejected a secret field")
    if record.get("recordType") != "KFH_CASH_STATEMENT_ATTEMPT_REDACTED_V1":
        raise Gate5ACaptureError("Attempt-record audit rejected its record type")
    pages = record.get("pages")
    if not isinstance(pages, list) or not pages:
        raise Gate5ACaptureError("Attempt-record audit requires at least one page")
    for page in pages:
        if not isinstance(page, dict) or set(page) != ATTEMPT_PAGE_FIELDS:
            raise Gate5ACaptureError("Attempt-record audit rejected page fields")
        if page.get("requestResponseCorrelated") is not True:
            raise Gate5ACaptureError("Attempt-record audit rejected uncorrelated evidence")
        integer_fields = ATTEMPT_PAGE_FIELDS - {
            "dateRange",
            "requestResponseCorrelated",
        }
        if any(
            isinstance(page.get(field), bool) or not isinstance(page.get(field), int)
            for field in integer_fields
        ):
            raise Gate5ACaptureError("Attempt-record audit rejected a paging scalar")
        if page.get("isNxtPagAvail") not in {0, 1}:
            raise Gate5ACaptureError("Attempt-record audit rejected a continuation flag")
        date_range = page.get("dateRange")
        if not isinstance(date_range, str) or len(date_range) != 18:
            raise Gate5ACaptureError("Attempt-record audit rejected a date range")
        from_date, separator, to_date = date_range.partition("..")
        if separator != "..":
            raise Gate5ACaptureError("Attempt-record audit rejected a date range")
        _compact_date(from_date, "attempt fromDate")
        _compact_date(to_date, "attempt toDate")
    result = record.get("result", {})
    if not isinstance(result, dict) or set(result) != {
        "captureValid",
        "multiPageObserved",
        "gate5aPass",
        "paginationAuthorized",
        "queryCaptureOrigin",
        "repeatedPageRequestObserved",
        "repeatedPageRequestCount",
        "newQueryDetected",
        "currentCaptureAttemptStopped",
    }:
        raise Gate5ACaptureError("Attempt-record audit rejected result fields")
    if result.get("captureValid") is not True:
        raise Gate5ACaptureError("Attempt-record audit rejected invalid evidence")
    if result.get("gate5aPass") is not False:
        raise Gate5ACaptureError("An attempt record cannot pass Gate 5A")
    if result.get("paginationAuthorized") is not False:
        raise Gate5ACaptureError("An attempt record cannot authorize pagination")
    if result.get("queryCaptureOrigin") != "OWNER_ARMED_VIEW_ATTEMPT":
        raise Gate5ACaptureError("Attempt-record capture origin is invalid")
    retry_count = result.get("repeatedPageRequestCount")
    if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
        raise Gate5ACaptureError("Attempt-record retry count is invalid")
    if result.get("repeatedPageRequestObserved") is not (retry_count > 0):
        raise Gate5ACaptureError("Attempt-record retry declaration is invalid")
    new_query_detected = result.get("newQueryDetected")
    attempt_stopped = result.get("currentCaptureAttemptStopped")
    if not isinstance(new_query_detected, bool) or not isinstance(attempt_stopped, bool):
        raise Gate5ACaptureError("Attempt-record query-state declaration is invalid")
    if attempt_stopped is not new_query_detected:
        raise Gate5ACaptureError("Attempt-record new-query stop declaration is invalid")
    if result.get("multiPageObserved") is not (len(pages) >= 2):
        raise Gate5ACaptureError("Attempt-record page-count declaration is invalid")
    security = record.get("security", {})
    if not isinstance(security, dict) or set(security) != {
        "identifiersRetained",
        "transactionValuesRetained",
        "cashTotalsRetained",
        "rawFramesRetained",
        "authenticationDataRetained",
    } or any(value is not False for value in security.values()):
        raise Gate5ACaptureError("Attempt-record audit rejected security declarations")


def write_fixture(path: Path, fixture: dict[str, Any]) -> str:
    """Write valid sanitized evidence even when it cannot authorize Gate 5A."""
    if fixture.get("proof", {}).get("captureValid") is not True:
        raise Gate5ACaptureError("Gate 5A capture is invalid; evidence fixture not written")
    audit_sanitized_fixture(fixture)
    payload = json.dumps(fixture, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return sha256(payload.encode("utf-8")).hexdigest()


def write_attempt_record(path: Path, record: dict[str, Any]) -> str:
    """Persist a sanitized diagnostic attempt that cannot authorize Gate 5A."""
    audit_sanitized_attempt_record(record)
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return sha256(payload.encode("utf-8")).hexdigest()
