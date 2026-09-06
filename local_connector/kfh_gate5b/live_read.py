"""Owner-controlled one-shot Saham Cash Statement live-read validation."""

from __future__ import annotations

import asyncio
import getpass
import re
from typing import Any

from local_connector.kfh_gate3a.connector import KfhGate3AConnector
from local_connector.kfh_gate3a.state import KfhAuthState

from .adapter import (
    Gate5BLiveAdapterError,
    Gate5BLiveAuthenticatedContextError,
    KfhCashStatementSessionAdapter,
)
from .bridge import run_typescript_cash_statement_read
from .browser import Gate5BLiveBrowserRuntime
from .evidence import EVIDENCE_PATH, build_live_evidence, write_live_evidence


def _protocol_date(prompt: str) -> str:
    value = input(prompt).strip()
    if not re.fullmatch(r"\d{8}", value):
        raise Gate5BLiveAdapterError("Cash Statement date must be YYYYMMDD")
    return value


def _print_sanitized_result(evidence: dict[str, Any], evidence_hash: str) -> None:
    print("LIVE_READ_STARTED = YES")
    print("STATE = READY")
    print(f'REQUEST_STARTS = {evidence["requestStartSeqProgression"]}')
    print(f'REQUEST_CAPACITY = {evidence["requestPageCapacity"]}')
    print(f'RESPONSE_CASH_LOG_COUNTS = {evidence["responseCashLogsCounts"]}')
    print(f'RESPONSE_UNSETTLED_COUNTS = {evidence["responseUnsettledCounts"]}')
    print(f'CONTINUATION_SEQUENCE = {evidence["isNxtPagAvailSequence"]}')
    print(f'RESPONSE_TOTAL_SEQUENCE = {evidence["responseTotalSequence"]}')
    print(f'CORRELATION_STRATEGY = {evidence["correlationStrategy"]}')
    print(f'CORRELATION_SUCCESS_PER_PAGE = {str(evidence["allResponsesCorrelated"]).lower()}')
    print(f'FINAL_RESPONSE_OBSERVED = {str(evidence["finalResponseObserved"]).lower()}')
    print(f'FINAL_IS_NEXT_PAGE_AVAILABLE = {evidence["finalIsNextPageAvailable"]}')
    print(f'PARTIAL_READ = {str(evidence["partialRead"]).lower()}')
    print(f'FINANCIAL_WRITES = {evidence["financialWritesPerformed"]}')
    print(f'BROWSER_CLOSED_SUCCESSFULLY = {str(evidence["browserClosedSuccessfully"]).lower()}')
    print(f'NEW_RUN_RESTORED_SESSION = {str(evidence["newRunRestoredSession"]).lower()}')
    print(f'LIVE_READ_PASS = {str(evidence["liveReadPass"]).lower()}')
    print(f"SANITIZED_EVIDENCE_FILE = {EVIDENCE_PATH}")
    print(f"SANITIZED_EVIDENCE_SHA256 = {evidence_hash}")


async def _new_run_restores_no_session() -> bool:
    connector = KfhGate3AConnector()
    try:
        snapshot = await connector.connect()
        if snapshot.state != KfhAuthState.LOGIN_REQUIRED:
            raise Gate5BLiveAdapterError(
                "Fresh ephemeral browser did not prove LOGIN_REQUIRED"
            )
        return False
    finally:
        await connector.close()


async def main() -> None:
    adapter_holder: dict[str, KfhCashStatementSessionAdapter] = {}

    def statement_response(frame: str | bytes) -> None:
        adapter = adapter_holder.get("adapter")
        if adapter is not None:
            adapter._observe_statement_response(frame)

    runtime = Gate5BLiveBrowserRuntime(on_statement_response_frame=statement_response)
    connector = KfhGate3AConnector(runtime)
    adapter: KfhCashStatementSessionAdapter | None = None
    live: dict[str, Any] | None = None
    gate3a_ready = False
    browser_closed = False

    print("Opening the isolated, ephemeral Gate 3A-R1 KFH browser.")
    print("Enter username, password, and OTP only in the visible KFH-controlled page.")
    print("No credentials, session state, or financial records will be retained.")
    await connector.connect()
    snapshot = await connector.wait_for_ready(timeout_seconds=300)
    if snapshot.state != KfhAuthState.READY:
        await connector.close()
        print("GATE5B_L1_RESULT = FAILED_CLOSED -- GATE3A_NOT_READY")
        return
    gate3a_ready = True
    await runtime._mark_gate3a_ready()
    print("STATE = READY")
    print("Navigate to English -> Trade -> Statement and select the intended range.")
    print("DO NOT click View and DO NOT click any pagination arrow.")

    try:
        account = getpass.getpass(
            "Enter the selected KFH security account for this transient read (hidden): "
        ).strip()
        from_date = _protocol_date("From date YYYYMMDD: ")
        to_date = _protocol_date("To date YYYYMMDD: ")
        command = input("Type READ to let Saham retrieve all pages automatically: ").strip()
        if command != "READ":
            raise Gate5BLiveAdapterError("Exact READ command was not supplied")

        adapter = KfhCashStatementSessionAdapter(
            runtime,
            ready=lambda: connector.status().state == KfhAuthState.READY,
            authenticated_context_status=runtime._authenticated_context_status,
            timeout_seconds=45,
        )
        adapter_holder["adapter"] = adapter
        live = await run_typescript_cash_statement_read(
            adapter,
            {
                "secAccNum": account,
                "frmDate": from_date,
                "toDate": to_date,
                "sortMode": 0,
                "startSeq": 0,
                "totalNoRec": 20,
            },
        )
    except Gate5BLiveAuthenticatedContextError as error:
        print(str(error))
    except (Gate5BLiveAdapterError, OSError, TimeoutError):
        print("GATE5B_L1_RESULT = FAILED_CLOSED -- CASH_STATEMENT_READ_FAILED")
    finally:
        account = ""
        if adapter is not None:
            await adapter.close()
        adapter_holder.clear()
        final_snapshot = await connector.logout()
        browser_closed = final_snapshot.state == KfhAuthState.DISCONNECTED

    if live is None:
        return

    try:
        restored = await _new_run_restores_no_session()
    except Gate5BLiveAdapterError:
        print("GATE5B_L1_RESULT = FAILED_CLOSED -- EPHEMERAL_SESSION_CHECK_FAILED")
        return
    evidence = build_live_evidence(
        live,
        gate3a_ready=gate3a_ready,
        browser_closed_successfully=browser_closed,
        new_run_restored_session=restored,
    )
    if not evidence["liveReadPass"]:
        print("GATE5B_L1_RESULT = FAILED_CLOSED -- PASS_CONDITIONS_NOT_MET")
        return
    evidence_hash = write_live_evidence(EVIDENCE_PATH, evidence)
    _print_sanitized_result(evidence, evidence_hash)
    print("GATE5B_L1_RESULT = PASSED -- OWNER_CLOSURE_REQUIRED")


if __name__ == "__main__":
    asyncio.run(main())
