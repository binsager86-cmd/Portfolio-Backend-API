"""Explicit Gate 3A authentication state machine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class KfhAuthState(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    OPENING_KFH = "OPENING_KFH"
    LOGIN_REQUIRED = "LOGIN_REQUIRED"
    AUTHENTICATING = "AUTHENTICATING"
    OTP_REQUIRED = "OTP_REQUIRED"
    AUTHENTICATED = "AUTHENTICATED"
    READY = "READY"
    AUTH_FAILED = "AUTH_FAILED"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    BROWSER_CLOSED = "BROWSER_CLOSED"
    KFH_UNAVAILABLE = "KFH_UNAVAILABLE"
    NETWORK_ERROR = "NETWORK_ERROR"
    CONNECTOR_ERROR = "CONNECTOR_ERROR"


TRANSITIONS: dict[KfhAuthState, frozenset[KfhAuthState]] = {
    KfhAuthState.DISCONNECTED: frozenset({KfhAuthState.OPENING_KFH}),
    KfhAuthState.OPENING_KFH: frozenset(
        {
            KfhAuthState.LOGIN_REQUIRED,
            KfhAuthState.AUTHENTICATING,
            KfhAuthState.BROWSER_CLOSED,
            KfhAuthState.KFH_UNAVAILABLE,
            KfhAuthState.NETWORK_ERROR,
            KfhAuthState.CONNECTOR_ERROR,
            KfhAuthState.DISCONNECTED,
        }
    ),
    KfhAuthState.LOGIN_REQUIRED: frozenset(
        {
            KfhAuthState.AUTHENTICATING,
            KfhAuthState.OTP_REQUIRED,
            KfhAuthState.AUTH_FAILED,
            KfhAuthState.BROWSER_CLOSED,
            KfhAuthState.KFH_UNAVAILABLE,
            KfhAuthState.NETWORK_ERROR,
            KfhAuthState.CONNECTOR_ERROR,
            KfhAuthState.DISCONNECTED,
        }
    ),
    KfhAuthState.AUTHENTICATING: frozenset(
        {
            KfhAuthState.LOGIN_REQUIRED,
            KfhAuthState.OTP_REQUIRED,
            KfhAuthState.AUTHENTICATED,
            KfhAuthState.AUTH_FAILED,
            KfhAuthState.BROWSER_CLOSED,
            KfhAuthState.KFH_UNAVAILABLE,
            KfhAuthState.NETWORK_ERROR,
            KfhAuthState.CONNECTOR_ERROR,
            KfhAuthState.DISCONNECTED,
        }
    ),
    KfhAuthState.OTP_REQUIRED: frozenset(
        {
            KfhAuthState.AUTHENTICATING,
            KfhAuthState.AUTHENTICATED,
            KfhAuthState.AUTH_FAILED,
            KfhAuthState.BROWSER_CLOSED,
            KfhAuthState.NETWORK_ERROR,
            KfhAuthState.CONNECTOR_ERROR,
            KfhAuthState.DISCONNECTED,
        }
    ),
    KfhAuthState.AUTHENTICATED: frozenset(
        {
            KfhAuthState.READY,
            KfhAuthState.SESSION_EXPIRED,
            KfhAuthState.BROWSER_CLOSED,
            KfhAuthState.NETWORK_ERROR,
            KfhAuthState.CONNECTOR_ERROR,
            KfhAuthState.DISCONNECTED,
        }
    ),
    KfhAuthState.READY: frozenset(
        {
            KfhAuthState.SESSION_EXPIRED,
            KfhAuthState.BROWSER_CLOSED,
            KfhAuthState.NETWORK_ERROR,
            KfhAuthState.CONNECTOR_ERROR,
            KfhAuthState.DISCONNECTED,
        }
    ),
}

for terminal in (
    KfhAuthState.AUTH_FAILED,
    KfhAuthState.SESSION_EXPIRED,
    KfhAuthState.BROWSER_CLOSED,
    KfhAuthState.KFH_UNAVAILABLE,
    KfhAuthState.NETWORK_ERROR,
    KfhAuthState.CONNECTOR_ERROR,
):
    TRANSITIONS[terminal] = frozenset({KfhAuthState.DISCONNECTED, KfhAuthState.OPENING_KFH})


@dataclass(frozen=True, slots=True)
class KfhConnectionSnapshot:
    state: KfhAuthState
    reason_code: str | None = None

    def public_dict(self) -> dict[str, str | None]:
        value = asdict(self)
        value["state"] = self.state.value
        return value


class KfhStateMachine:
    def __init__(self) -> None:
        self._snapshot = KfhConnectionSnapshot(KfhAuthState.DISCONNECTED)

    @property
    def snapshot(self) -> KfhConnectionSnapshot:
        return self._snapshot

    def transition(
        self,
        target: KfhAuthState,
        reason_code: str | None = None,
    ) -> KfhConnectionSnapshot:
        current = self._snapshot.state
        if target == current:
            return self._snapshot
        if target not in TRANSITIONS[current]:
            raise RuntimeError(f"Invalid KFH Gate 3A transition: {current.value} -> {target.value}")
        self._snapshot = KfhConnectionSnapshot(target, reason_code)
        return self._snapshot
