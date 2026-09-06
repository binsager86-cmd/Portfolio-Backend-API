"""Private local bridge from Python browser transport to the TypeScript reader."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from .adapter import (
    Gate5BLiveAdapterError,
    Gate5BLiveBridgeFailureError,
    KfhCashStatementSessionAdapter,
)

BRIDGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "mobile-app"
    / "scripts"
    / "kfh-gate5b-live-bridge.cjs"
)

_FAILURE_SEQUENCE_FIELDS = frozenset(
    {
        "requestStartSeqProgression",
        "responseCashLogsCounts",
        "responseUnsettledCounts",
        "isNxtPagAvailSequence",
        "responseTotalSequence",
    }
)

_PREVIEW_LOG_FIELDS = frozenset(
    {
        "date",
        "trnsType",
        "trnsRef",
        "particulars",
        "dtails",
        "amount",
        "settleDate",
        "trnsCurr",
        "qty",
        "price",
        "commission",
        "feeWithTax",
        "vatAmount",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "currency",
        "open_balance",
        "close_balance",
        "total_deposit",
        "total_withdrawal",
        "total_buy",
        "total_sell",
        "total_other",
        "vat_amount",
    }
)


def _validated_preview(value: Any) -> dict[str, Any]:
    """Validate the minimized private handoff before it can reach localhost."""
    if not isinstance(value, dict) or set(value) != {
        "cashLogs",
        "unsettledCashLogs",
        "statementSummary",
    }:
        raise Gate5BLiveAdapterError("Gate 5B preview shape was invalid")

    def logs(name: str) -> list[dict[str, str | int | float]]:
        rows = value.get(name)
        if not isinstance(rows, list) or len(rows) > 10_000:
            raise Gate5BLiveAdapterError("Gate 5B preview rows were invalid")
        validated: list[dict[str, str | int | float]] = []
        for row in rows:
            if not isinstance(row, dict) or not set(row).issubset(_PREVIEW_LOG_FIELDS):
                raise Gate5BLiveAdapterError("Gate 5B preview row was invalid")
            if any(
                not isinstance(item, (str, int, float)) or isinstance(item, bool)
                for item in row.values()
            ):
                raise Gate5BLiveAdapterError("Gate 5B preview scalar was invalid")
            validated.append(dict(row))
        return validated

    summary = value.get("statementSummary")
    if (
        not isinstance(summary, dict)
        or set(summary) != _SUMMARY_FIELDS
        or not all(isinstance(item, str) for item in summary.values())
    ):
        raise Gate5BLiveAdapterError("Gate 5B statement summary was invalid")
    return {
        "cashLogs": logs("cashLogs"),
        "unsettledCashLogs": logs("unsettledCashLogs"),
        "statementSummary": dict(summary),
    }


def _sanitize_failure_evidence(value: Any) -> dict[str, Any]:
    """Accept only aggregate counters/flags; never forward raw KFH payload data."""
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for field in _FAILURE_SEQUENCE_FIELDS:
        sequence = value.get(field)
        if (
            isinstance(sequence, list)
            and len(sequence) <= 100
            and all(isinstance(item, int) and not isinstance(item, bool) for item in sequence)
        ):
            sanitized[field] = list(sequence)
    for field in (
        "allResponsesCorrelated",
        "finalResponseObserved",
        "partialRead",
    ):
        if isinstance(value.get(field), bool):
            sanitized[field] = value[field]
    writes = value.get("financialWritesPerformed")
    if isinstance(writes, int) and not isinstance(writes, bool):
        sanitized["financialWritesPerformed"] = writes
    return sanitized


async def _write_message(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    writer.write((json.dumps(message, separators=(",", ":")) + "\n").encode())
    await writer.drain()


async def run_typescript_cash_statement_read(
    adapter: KfhCashStatementSessionAdapter,
    query: dict[str, Any],
    *,
    account_currency: str = "",
    timeout_seconds: float = 180,
) -> dict[str, Any]:
    """Run the real TS reader; raw request/response data stays inside local pipes."""
    process = await asyncio.create_subprocess_exec(
        "node",
        str(BRIDGE_PATH),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdin = process.stdin
    stdout = process.stdout
    if stdin is None or stdout is None:
        raise Gate5BLiveAdapterError("Gate 5B-L1 bridge could not start")

    async def exchange() -> dict[str, Any]:
        await _write_message(
            stdin,
            {"type": "start", "query": query, "accountCurrency": account_currency},
        )
        while True:
            line = await stdout.readline()
            if not line:
                raise Gate5BLiveAdapterError("Gate 5B-L1 bridge ended before completion")
            try:
                message = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise Gate5BLiveAdapterError("Gate 5B-L1 bridge output was invalid") from error
            if not isinstance(message, dict):
                raise Gate5BLiveAdapterError("Gate 5B-L1 bridge output was invalid")
            message_type = message.get("type")
            if message_type == "cash_statement_request":
                request = message.get("request")
                if not isinstance(request, dict):
                    raise Gate5BLiveAdapterError("Gate 5B-L1 request was invalid")
                envelope = await adapter.request_cash_statement(request)
                await _write_message(
                    stdin,
                    {"type": "cash_statement_response", "envelope": envelope},
                )
                continue
            if message_type == "complete":
                evidence = message.get("evidence")
                if not isinstance(evidence, dict):
                    raise Gate5BLiveAdapterError("Gate 5B-L1 evidence was invalid")
                return {
                    **evidence,
                    "previewPayload": _validated_preview(message.get("preview")),
                }
            if message_type == "failed":
                # The bridge's own code/detail are static or field-name-based
                # (see kfh-gate5b-live-bridge.cjs), never raw financial data -
                # carried as structured attributes so the real allowlisted
                # failure code survives instead of being collapsed.
                code = message.get("code")
                detail = message.get("detail")
                raise Gate5BLiveBridgeFailureError(
                    code if isinstance(code, str) else "LIVE_READ_FAILED",
                    detail if isinstance(detail, str) else None,
                    _sanitize_failure_evidence(message.get("evidence")),
                )
            raise Gate5BLiveAdapterError("Gate 5B-L1 bridge message was rejected")

    try:
        return await asyncio.wait_for(exchange(), timeout_seconds)
    finally:
        stdin.close()
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
        await process.wait()
