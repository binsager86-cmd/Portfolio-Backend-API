from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from app.services.eagle_eye_v2.simulator.constants import INITIAL_CAPITAL_KWD


class PortfolioName(StrEnum):
    BUY = "BUY"
    WATCHLIST = "WATCHLIST"


class TransactionType(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    VOID = "VOID"


class DecisionKind(StrEnum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    VETO = "VETO"
    DAILY_STATE = "DAILY_STATE"
    GUARD_TRIP = "GUARD_TRIP"


@dataclass(frozen=True)
class MarketSession:
    symbol: str
    session: str
    open_price: float | None
    close_price: float | None
    ingestion_ts: str
    decision_close_ts: str
    suspended: bool = False


@dataclass(frozen=True)
class FrozenEvent:
    symbol: str
    decision_session: str
    kind: DecisionKind
    reason: str
    action: dict[str, Any]
    state_snapshot: dict[str, Any]
    would_have_entry_reason: str | None = None
    veto_tier: str | None = None


@dataclass(frozen=True)
class PendingOrder:
    portfolio: PortfolioName
    symbol: str
    side: TransactionType
    decision_session: str
    earliest_fill_session: str
    reason: str
    state_snapshot: dict[str, Any]
    target_notional_kwd: float | None = None
    source_event_id: str | None = None
    missed_fill_sessions: int = 0


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    opened_session: str
    reason: str


@dataclass
class PortfolioState:
    name: PortfolioName
    cash_kwd: float = INITIAL_CAPITAL_KWD
    positions: dict[str, Position] = field(default_factory=dict)

    def nav(self, closes: dict[str, float]) -> float:
        value = self.cash_kwd
        for symbol, position in self.positions.items():
            value += position.quantity * float(closes.get(symbol, position.avg_cost))
        return value

    def open_position_count(self) -> int:
        return len(self.positions)


def parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)
