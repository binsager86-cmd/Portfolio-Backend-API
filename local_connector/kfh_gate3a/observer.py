"""Secret-discarding observer for inbound KFH authentication success only."""

from __future__ import annotations

import json
from typing import Any


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _find_auth_status(value: Any) -> int | None:
    if isinstance(value, dict):
        if "authSts" in value:
            status = _integer(value.get("authSts"))
            if status is not None:
                return status
        for child in value.values():
            status = _find_auth_status(child)
            if status is not None:
                return status
    elif isinstance(value, list):
        for child in value:
            status = _find_auth_status(child)
            if status is not None:
                return status
    return None


class KfhAuthenticationObserver:
    """Stores booleans only; inbound frame payloads are discarded immediately."""

    __slots__ = ("_authenticated", "_failed")

    def __init__(self) -> None:
        self._authenticated = False
        self._failed = False

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    @property
    def failed(self) -> bool:
        return self._failed

    def observe_inbound_frame(self, frame: str | bytes) -> None:
        if not isinstance(frame, str) or len(frame) > 1_000_000:
            return
        try:
            decoded = json.loads(frame)
        except (json.JSONDecodeError, TypeError):
            return
        for candidate in _walk(decoded):
            if _integer(candidate.get("msgGrp")) != 5 or _integer(candidate.get("msgTyp")) != 101:
                continue
            status = _find_auth_status(candidate)
            if status == 1:
                self._authenticated = True
                self._failed = False
            elif status is not None:
                self._failed = True
            return
