from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

from app.services.eagle_eye_v2.simulator.constants import (
    COMMISSION_RATE,
    ENTRY_REASON_ALIASES,
    ENTRY_REASONS,
    EXIT_REASONS,
    MAX_CONCURRENT_POSITIONS,
    POSITION_SIZE_FRACTION,
)
from app.services.eagle_eye_v2.simulator.ledger import SimulatorLedger
from app.services.eagle_eye_v2.simulator.models import (
    DecisionKind,
    FrozenEvent,
    MarketSession,
    PendingOrder,
    PortfolioName,
    PortfolioState,
    Position,
    TransactionType,
)


@dataclass
class SessionExecutionResult:
    transactions: list[int] = field(default_factory=list)
    decisions: list[int] = field(default_factory=list)
    guard_trips: list[int] = field(default_factory=list)
    pending_orders: list[PendingOrder] = field(default_factory=list)


def canonical_entry_reason(reason: str | None) -> str:
    raw = str(reason or "").strip().upper()
    return ENTRY_REASON_ALIASES.get(raw, raw)


class PaperPortfolioEngine:
    def __init__(self, ledger: SimulatorLedger) -> None:
        self.ledger = ledger
        self.portfolios = {
            PortfolioName.BUY: PortfolioState(PortfolioName.BUY),
            PortfolioName.WATCHLIST: PortfolioState(PortfolioName.WATCHLIST),
        }
        self.pending_orders: list[PendingOrder] = []

    def process_session(self, session: str, market_sessions: dict[str, MarketSession], events: Iterable[FrozenEvent]) -> SessionExecutionResult:
        result = SessionExecutionResult()
        self._fill_due_orders(session, market_sessions, result)
        for event in events:
            market_session = market_sessions[event.symbol.upper()]
            result.decisions.append(self.ledger.append_decision(event, market_session))
            self._schedule_from_event(event, result)
        self._write_valuations(session, market_sessions)
        result.pending_orders = list(self.pending_orders)
        return result

    def _schedule_from_event(self, event: FrozenEvent, result: SessionExecutionResult) -> None:
        if event.kind == DecisionKind.ENTRY and canonical_entry_reason(event.reason) in ENTRY_REASONS:
            self._schedule_entry(PortfolioName.BUY, event, result)
        elif event.kind == DecisionKind.VETO and canonical_entry_reason(event.would_have_entry_reason or event.reason) in ENTRY_REASONS:
            self._schedule_entry(PortfolioName.WATCHLIST, event, result)
        elif event.kind == DecisionKind.EXIT:
            self._schedule_exit(PortfolioName.BUY, event, result)
            self._schedule_exit(PortfolioName.WATCHLIST, event, result)
        elif event.kind == DecisionKind.DAILY_STATE:
            watchlist_exit_reason = self._watchlist_exit_reason_from_daily_state(event)
            if watchlist_exit_reason:
                self._schedule_exit(PortfolioName.WATCHLIST, self._event_with_reason(event, watchlist_exit_reason), result)

    @staticmethod
    def _watchlist_exit_reason_from_daily_state(event: FrozenEvent) -> str | None:
        action = event.action
        published_reason = str(action.get("watchlist_exit_reason") or action.get("published_exit_reason") or action.get("structural_exit_reason") or "").upper()
        if published_reason in EXIT_REASONS:
            return published_reason
        if str(action.get("avoid_tier") or event.veto_tier or "").upper() == "AVOID_HARD":
            return "EXIT_AVOID_HARD"
        return None

    def _schedule_entry(self, portfolio_name: PortfolioName, event: FrozenEvent, result: SessionExecutionResult) -> None:
        portfolio = self.portfolios[portfolio_name]
        symbol = event.symbol.upper()
        if symbol in portfolio.positions:
            return
        if portfolio.open_position_count() >= MAX_CONCURRENT_POSITIONS:
            return
        nav = portfolio.nav({symbol: float(event.action.get("entry_close") or event.action.get("close") or 0.0)})
        target_notional = nav * POSITION_SIZE_FRACTION
        self.pending_orders.append(
            PendingOrder(
                portfolio=portfolio_name,
                symbol=symbol,
                side=TransactionType.BUY,
                decision_session=event.decision_session,
                earliest_fill_session=event.decision_session,
                reason=canonical_entry_reason(event.would_have_entry_reason or event.reason),
                state_snapshot=event.state_snapshot,
                target_notional_kwd=target_notional,
                source_event_id=str(event.action.get("position_id") or ""),
            )
        )
        result.pending_orders = list(self.pending_orders)

    def _schedule_exit(self, portfolio_name: PortfolioName, event: FrozenEvent, result: SessionExecutionResult) -> None:
        portfolio = self.portfolios[portfolio_name]
        symbol = event.symbol.upper()
        if symbol not in portfolio.positions:
            return
        self.pending_orders.append(
            PendingOrder(
                portfolio=portfolio_name,
                symbol=symbol,
                side=TransactionType.SELL,
                decision_session=event.decision_session,
                earliest_fill_session=event.decision_session,
                reason=str(event.reason),
                state_snapshot=event.state_snapshot,
                source_event_id=str(event.action.get("position_id") or ""),
            )
        )
        result.pending_orders = list(self.pending_orders)

    def _fill_due_orders(self, session: str, market_sessions: dict[str, MarketSession], result: SessionExecutionResult) -> None:
        still_pending: list[PendingOrder] = []
        for order in self.pending_orders:
            if session <= order.decision_session:
                still_pending.append(order)
                continue
            market_session = market_sessions.get(order.symbol)
            if market_session is None or market_session.open_price is None or float(market_session.open_price) <= 0.0 or market_session.suspended:
                still_pending.append(replace(order, missed_fill_sessions=order.missed_fill_sessions + 1))
                continue
            tx_id = self._fill_order(order, market_session, order.missed_fill_sessions)
            if tx_id is not None:
                result.transactions.append(tx_id)
        self.pending_orders = still_pending

    def _fill_order(self, order: PendingOrder, market_session: MarketSession, suspension_gap: int) -> int | None:
        portfolio = self.portfolios[order.portfolio]
        price = float(market_session.open_price or 0.0)
        if price <= 0.0:
            return None
        if order.side == TransactionType.BUY:
            if order.symbol in portfolio.positions or portfolio.open_position_count() >= MAX_CONCURRENT_POSITIONS:
                return None
            gross = min(float(order.target_notional_kwd or 0.0), portfolio.cash_kwd / (1.0 + COMMISSION_RATE))
            if gross <= 0.0:
                return None
            commission = gross * COMMISSION_RATE
            quantity = gross / price
            portfolio.cash_kwd -= gross + commission
            portfolio.positions[order.symbol] = Position(order.symbol, quantity, price, order.decision_session, order.reason)
            return self.ledger.append_transaction(
                portfolio=order.portfolio.value,
                transaction_type=TransactionType.BUY,
                symbol=order.symbol,
                quantity=quantity,
                price=price,
                gross_value_kwd=gross,
                commission_kwd=commission,
                net_cash_delta_kwd=-(gross + commission),
                decision_session=order.decision_session,
                fill_session=market_session.session,
                reason=order.reason,
                market_session=market_session,
                state_snapshot=order.state_snapshot,
                source_event_id=order.source_event_id,
                suspension_gap_sessions=suspension_gap,
            )
        position = portfolio.positions.get(order.symbol)
        if position is None:
            return None
        gross = position.quantity * price
        commission = gross * COMMISSION_RATE
        portfolio.cash_kwd += gross - commission
        del portfolio.positions[order.symbol]
        return self.ledger.append_transaction(
            portfolio=order.portfolio.value,
            transaction_type=TransactionType.SELL,
            symbol=order.symbol,
            quantity=position.quantity,
            price=price,
            gross_value_kwd=gross,
            commission_kwd=commission,
            net_cash_delta_kwd=gross - commission,
            decision_session=order.decision_session,
            fill_session=market_session.session,
            reason=order.reason,
            market_session=market_session,
            state_snapshot=order.state_snapshot,
            source_event_id=order.source_event_id,
            suspension_gap_sessions=suspension_gap,
        )

    def _write_valuations(self, session: str, market_sessions: dict[str, MarketSession]) -> None:
        closes = {symbol: float(row.close_price or 0.0) for symbol, row in market_sessions.items() if row.close_price is not None}
        all_symbols = sorted(set(market_sessions) | {symbol for portfolio in self.portfolios.values() for symbol in portfolio.positions})
        for portfolio in self.portfolios.values():
            nav = portfolio.nav(closes)
            for symbol in all_symbols:
                position = portfolio.positions.get(symbol)
                quantity = 0.0 if position is None else position.quantity
                close_price = closes.get(symbol, 0.0 if position is None else position.avg_cost)
                self.ledger.append_daily_valuation(
                    portfolio=portfolio.name.value,
                    symbol=symbol,
                    session=session,
                    quantity=quantity,
                    close_price=close_price,
                    market_value_kwd=quantity * close_price,
                    cash_kwd=portfolio.cash_kwd,
                    nav_kwd=nav,
                    state_snapshot={"positions": sorted(portfolio.positions), "portfolio": portfolio.name.value},
                )


    @staticmethod
    def _event_with_reason(event: FrozenEvent, reason: str) -> FrozenEvent:
        return FrozenEvent(
            symbol=event.symbol,
            decision_session=event.decision_session,
            kind=DecisionKind.EXIT,
            reason=reason,
            action=event.action,
            state_snapshot=event.state_snapshot,
            would_have_entry_reason=event.would_have_entry_reason,
            veto_tier=event.veto_tier,
        )
