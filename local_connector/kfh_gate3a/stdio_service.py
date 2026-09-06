"""Fixed-command local stdio boundary for Saham to control Gate 3A."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from .connector import KfhGate3AConnector

PUBLIC_COMMANDS = frozenset({"connect", "status", "wait_for_ready", "logout", "reconnect", "close"})


def _validated_request(line: str) -> tuple[int, str]:
    if len(line) > 4096:
        raise ValueError("KFH connector request is too large")
    request: dict[str, Any] = json.loads(line)
    if not isinstance(request, dict) or set(request) != {"id", "method"}:
        raise ValueError("Invalid KFH connector request fields")
    request_id = request["id"]
    method = request["method"]
    if isinstance(request_id, bool) or not isinstance(request_id, int):
        raise ValueError("KFH connector request ID must be an integer")
    if not isinstance(method, str) or method not in PUBLIC_COMMANDS:
        raise ValueError("Unsupported KFH connector command")
    return request_id, method


async def _execute(connector: KfhGate3AConnector, method: str) -> dict[str, str | None]:
    if method not in PUBLIC_COMMANDS:
        raise ValueError("Unsupported KFH connector command")
    if method == "connect":
        snapshot = await connector.connect()
    elif method == "status":
        snapshot = connector.status()
    elif method == "wait_for_ready":
        snapshot = await connector.wait_for_ready()
    elif method == "logout":
        snapshot = await connector.logout()
    elif method == "reconnect":
        snapshot = await connector.reconnect()
    else:
        await connector.close()
        snapshot = connector.status()
    return snapshot.public_dict()


async def serve() -> None:
    connector = KfhGate3AConnector()
    try:
        while line := await asyncio.to_thread(sys.stdin.readline):
            request_id: int | None = None
            try:
                request_id, method = _validated_request(line)
                result = await _execute(connector, method)
                response = {"id": request_id, "ok": True, "result": result}
            except Exception:
                response = {
                    "id": request_id,
                    "ok": False,
                    "error": {"code": "INVALID_CONNECTOR_COMMAND"},
                }
            print(json.dumps(response, separators=(",", ":")), flush=True)
    finally:
        await connector.close()


if __name__ == "__main__":
    asyncio.run(serve())
