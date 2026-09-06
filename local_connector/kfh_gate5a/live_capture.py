"""Owner-operated passive Gate 5A capture using KFH's own pagination UI."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from local_connector.kfh_gate3a.connector import KfhGate3AConnector

from .browser import Gate5APassiveBrowserRuntime
from .capture import (
    Gate5ACaptureError,
    KfhGate5APassiveCapture,
    write_attempt_record,
    write_fixture,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "kfh"
    / "cash_statement_pagination_real_redacted_evidence_v1.json"
)
ATTEMPT_DIRECTORY = Path(__file__).resolve().parents[2] / "docs" / "evidence" / "kfh"


def _print_first_page_summary(summary: dict[str, Any]) -> None:
    print("FIRST_PAGE_CAPTURED = YES")
    print(f'QUERY_CAPTURE_ORIGIN = {summary["queryCaptureOrigin"]}')
    print("REQUEST:")
    print(f'startSeq = {summary["startSeq"]}')
    print(f'requestPageCapacity = {summary["requestPageCapacity"]}')
    print(f'sortMode = {summary["sortMode"]}')
    print(f'dateRange = {summary["dateRange"]}')
    print("RESPONSE:")
    print(f'cashLogsCount = {summary["cashLogsCount"]}')
    print(f'unsettledCashLogsCount = {summary["unsettledCashLogsCount"]}')
    print(
        "responseTotalMatchingRecords = "
        f'{summary["responseTotalMatchingRecords"]}'
    )
    print(f'isNxtPagAvail = {summary["isNxtPagAvail"]}')
    print(f'pageNo = {summary["pageNo"]}')
    print(f'totalPages = {summary["totalPages"]}')
    print("REQUEST_RESPONSE_CORRELATED = YES")
    print(f'repeatedPageRequestCount = {summary["repeatedPageRequestCount"]}')
    if summary["kfhProtocolSaysMorePages"]:
        print("KFH_PROTOCOL_SAYS_MORE_PAGES = YES")
        print("OWNER_ACTION = LOCATE KFH NEXT/PAGINATION CONTROL")
        print("MORE PAGES EXIST ACCORDING TO KFH.")
    else:
        print("KFH_PROTOCOL_SAYS_MORE_PAGES = NO")
        print("THIS RANGE CANNOT PROVIDE MULTI_PAGE_EVIDENCE")
        print("RANGE_NOT_MULTIPAGE")


def _print_ui_diagnostic(diagnostic: dict[str, Any]) -> bool:
    print("PAGINATION_UI_DIAGNOSTIC")
    candidate_located = False
    for name, value in diagnostic.items():
        print(f"{name}:")
        for field, field_value in value.items():
            rendered = str(field_value).lower() if isinstance(field_value, bool) else field_value
            print(f"{field}={rendered}")
        if name != "STATEMENT_SCROLL_CONTAINER":
            candidate_located = candidate_located or bool(value["matched"])
    if not candidate_located:
        print("PAGINATION_UI_CONTROL_NOT_LOCATED")
    return candidate_located


async def _wait_for_explicit_end(
    runtime: Gate5APassiveBrowserRuntime,
) -> Literal["DONE", "END"]:
    while True:
        command = (
            await asyncio.to_thread(
                input,
                "Type SCAN to inspect controls again, DONE after the final page, "
                "or END to stop this attempt: ",
            )
        ).strip().upper()
        if command == "SCAN" or not command:
            _print_ui_diagnostic(await runtime.inspect_pagination_ui())
            if not command:
                print("EMPTY_INPUT_DOES_NOT_END_CAPTURE")
            continue
        if command == "DONE":
            return "DONE"
        if command == "END":
            return "END"
        print("UNKNOWN_COMMAND; use SCAN, DONE, or END")


def _formal_fixture_allowed(
    owner_completion: Literal["DONE", "END"],
    *,
    completed_page_count: int,
    new_query_detected: bool,
) -> bool:
    return (
        owner_completion == "DONE"
        and completed_page_count >= 2
        and not new_query_detected
    )


def _arm_capture(capture: KfhGate5APassiveCapture) -> None:
    """Arm passive observation only; this function has no browser/network surface."""
    capture.activate_after_gate3a_ready()


async def _wait_for_arm(capture: KfhGate5APassiveCapture) -> None:
    while True:
        command = (
            await asyncio.to_thread(
                input,
                "Type ARM when the date range is ready and you are about to "
                "click View: ",
            )
        ).strip()
        if command == "ARM":
            _arm_capture(capture)
            print("GATE 5A CAPTURE ARMED")
            print("NOW CLICK KFH VIEW")
            return
        print("CAPTURE_NOT_ARMED; exact command required: ARM")


async def main() -> None:
    first_page_captured = asyncio.Event()

    def first_page_callback(summary: dict[str, Any]) -> None:
        _print_first_page_summary(summary)
        first_page_captured.set()

    def retry_callback(count: int) -> None:
        print("REPEATED_PAGE_REQUEST / RETRY")
        print("repeatedPageRequestObserved = true")
        print(f"repeatedPageRequestCount = {count}")

    def new_query_callback() -> None:
        print("NEW_QUERY_DETECTED")
        print("CURRENT_CAPTURE_ATTEMPT_STOPPED")
        print("End this attempt and rerun Gate 5A-R2.1 for the new query.")

    capture = KfhGate5APassiveCapture(
        on_first_page=first_page_callback,
        on_retry=retry_callback,
        on_new_query=new_query_callback,
    )
    runtime = Gate5APassiveBrowserRuntime(
        on_statement_request_frame=capture.observe_request_frame,
        on_statement_response_frame=capture.observe_response_frame,
    )
    connector = KfhGate3AConnector(runtime)
    print("Opening the isolated, ephemeral Gate 3A KFH browser.")
    print("Enter credentials and OTP only in the visible KFH-controlled page.")
    await connector.connect()
    snapshot = await connector.wait_for_ready(timeout_seconds=300)
    print(f"STATE = {snapshot.state.value}")
    if snapshot.state.value != "READY":
        await connector.close()
        raise SystemExit("Gate 5A capture stopped: Gate 3A did not reach READY")

    print("READY_CONFIRMATION = READY")
    print("Navigate to: English -> Trade -> Statement")
    print("Select the desired date range.")
    print("DO NOT CLICK VIEW YET.")
    print("Opening the Trade menu for Statement navigation is allowed; do not place a trade.")
    print("Select a range expected to contain more than the first-page capacity.")
    print("Do not open order entry, cancel/modify, transfers, withdrawals, transaction-password, or Level-2 screens.")
    await _wait_for_arm(capture)
    print("Use only KFH's own pagination control through the final page.")

    try:
        await asyncio.wait_for(first_page_captured.wait(), timeout=600)
        summary = capture.first_page_summary
        if summary is None:
            raise Gate5ACaptureError("First-page summary was not retained")

        if summary["kfhProtocolSaysMorePages"]:
            await asyncio.sleep(1)
            _print_ui_diagnostic(await runtime.inspect_pagination_ui())
            print("The generic UI scan is advisory and need not recognize KFH's custom pager.")
            print("Manually use KFH's visible native left/right pagination arrows.")
            print("Continue through the final KFH page, then type DONE.")
            owner_completion = await _wait_for_explicit_end(runtime)
        else:
            await asyncio.to_thread(
                input,
                "This range is not multi-page. Press Enter to retain the sanitized "
                "attempt and stop: ",
            )

            owner_completion = "END"

        if owner_completion == "END":
            attempt = capture.build_sanitized_attempt_record()
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            attempt_path = ATTEMPT_DIRECTORY / (
                f"cash_statement_attempt_redacted_{timestamp}.json"
            )
            attempt_hash = write_attempt_record(attempt_path, attempt)
            print(f"SANITIZED_ATTEMPT_RECORD = {attempt_path}")
            print(f"ATTEMPT_RECORD_SHA256 = {attempt_hash}")
            print("GATE5A_CONCLUSION = ABORTED_BY_OWNER")
            print("FORMAL_FIXTURE_WRITTEN = NO")
        elif _formal_fixture_allowed(
            owner_completion,
            completed_page_count=capture.completed_page_count,
            new_query_detected=capture.new_query_detected,
        ):
            fixture = capture.build_sanitized_fixture()
            fixture_hash = write_fixture(FIXTURE_PATH, fixture)
            print(json.dumps(fixture["proof"], indent=2, sort_keys=True))
            print(f"SANITIZED_FIXTURE = {FIXTURE_PATH}")
            print(f"FIXTURE_SHA256 = {fixture_hash}")
            if fixture["proof"]["gate5aPass"]:
                print("GATE5A_CONCLUSION = PASS_CRITERIA_SATISFIED_OWNER_CLOSURE_REQUIRED")
            else:
                print("GATE5A_CONCLUSION = VALID_REAL_EVIDENCE_RETAINED_GATE_BLOCKED")
        else:
            attempt = capture.build_sanitized_attempt_record()
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            attempt_path = ATTEMPT_DIRECTORY / (
                f"cash_statement_attempt_redacted_{timestamp}.json"
            )
            attempt_hash = write_attempt_record(attempt_path, attempt)
            print(f"SANITIZED_ATTEMPT_RECORD = {attempt_path}")
            print(f"ATTEMPT_RECORD_SHA256 = {attempt_hash}")
            print("captureValid = true")
            print(
                "multiPageObserved = "
                f'{str(capture.completed_page_count >= 2).lower()}'
            )
            print("gate5aPass = false")
            print(f"repeatedPageRequestCount = {capture.repeated_page_request_count}")
            if capture.new_query_detected:
                print("GATE5A_CONCLUSION = BLOCKED -- NEW_QUERY_DETECTED")
            else:
                print("GATE5A_CONCLUSION = BLOCKED -- MULTI_PAGE_NOT_OBSERVED")
            print("No pagination implementation is authorized.")
    except TimeoutError:
        print("GATE5A_CONCLUSION = BLOCKED -- FIRST_PAGE_NOT_CAPTURED_WITHIN_TIMEOUT")
        print("NEW_FIXTURE_WRITTEN = NO")
    except (Gate5ACaptureError, RuntimeError, ValueError) as error:
        print(f"GATE5A_CONCLUSION = BLOCKED -- {error}")
        print("NEW_FIXTURE_WRITTEN = NO")
        print("No pagination implementation is authorized.")
    finally:
        print(f"FINAL_CAPTURED_PAGES = {capture.completed_page_count}")
        await connector.logout()


if __name__ == "__main__":
    asyncio.run(main())
