"""Portfolio accounting engine with proper sign conventions."""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, List
from datetime import date
from simulation.domain.models import (
    Position,
    PositionState,
    Order,
    OrderSide,
    TradeRecord,
    EagleEyeRating,
)


# Monetary precision for all cash calculations
CASH_PRECISION = Decimal("0.01")
PRICE_PRECISION = Decimal("0.0001")


class PortfolioAccounting:
    """Portfolio accounting engine with canonical sign conventions.
    
    Sign Convention (REQUIRED):
    - quantity: Always positive on an order
    - price: Always positive
    - gross_amount = quantity * price
    - Transaction type determines cash direction (BUY vs SELL)
    - Commission, tax, slippage: Positive expense values
    
    BUY accounting:
        cash_after = cash_before - gross_amount - commission - slippage
        new_quantity = old_quantity + purchased_quantity
        new_cost_basis = old_cost_basis + gross_amount + commission + slippage
        new_average_cost = new_cost_basis / new_quantity
    
    SELL accounting:
        cash_after = cash_before + gross_amount - commission - slippage
        new_quantity = old_quantity - sold_quantity
        realized_pnl = gross_amount - costs - (avg_cost * sold_quantity)
    """

    def __init__(self, initial_cash: Decimal):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.completed_trades: List[TradeRecord] = []
        self.total_commissions = Decimal("0")
        self.total_slippage = Decimal("0")
        self.total_deposits = Decimal("0")
        self.total_dividends = Decimal("0")
        self.total_buy_costs = Decimal("0")        # Gross buy amounts + costs
        self.total_sell_proceeds = Decimal("0")    # Net sell proceeds
        self.initial_cash = initial_cash

    def get_position(self, symbol: str) -> Position:
        """Get or create position."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal("0"),
                cost_basis=Decimal("0"),
                average_cost=Decimal("0"),
                state=PositionState.FLAT,
            )
        return self.positions[symbol]

    def execute_buy(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal = Decimal("0"),
        slippage: Decimal = Decimal("0"),
    ) -> tuple[bool, str]:
        """Execute BUY order with proper accounting.
        
        Returns: (success, reason)
        """
        # Validate inputs
        if quantity <= 0 or price <= 0:
            return False, "Invalid quantity or price"

        gross_amount = (quantity * price).quantize(CASH_PRECISION, ROUND_HALF_UP)
        total_cost = gross_amount + commission + slippage

        # Check cash availability
        if total_cost > self.cash:
            return False, f"Insufficient cash: need {total_cost}, have {self.cash}"

        # Deduct cash (BUY reduces cash)
        self.cash = (self.cash - total_cost).quantize(CASH_PRECISION, ROUND_HALF_UP)
        self.total_commissions += commission
        self.total_slippage += slippage
        self.total_buy_costs += total_cost

        # Update position
        pos = self.get_position(symbol)
        old_quantity = pos.quantity
        old_cost_basis = pos.cost_basis

        # New cost basis includes all acquisition costs
        new_cost_basis = old_cost_basis + gross_amount + commission + slippage
        new_quantity = old_quantity + quantity

        pos.quantity = new_quantity
        pos.cost_basis = new_cost_basis.quantize(CASH_PRECISION, ROUND_HALF_UP)

        if new_quantity > 0:
            pos.average_cost = (new_cost_basis / new_quantity).quantize(
                PRICE_PRECISION, ROUND_HALF_UP
            )
        pos.current_price = price
        pos.state = PositionState.OPEN

        return True, "BUY executed"

    def execute_sell(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal = Decimal("0"),
        slippage: Decimal = Decimal("0"),
        signal_rating: EagleEyeRating = EagleEyeRating.SELL,
        signal_date: date = None,
    ) -> tuple[bool, str, Optional[TradeRecord]]:
        """Execute SELL order with proper accounting.
        
        Returns: (success, reason, completed_trade_or_none)
        """
        # Validate inputs
        if quantity <= 0 or price <= 0:
            return False, "Invalid quantity or price", None

        pos = self.get_position(symbol)

        # Oversell protection
        if quantity > pos.quantity:
            return (
                False,
                f"Oversell: selling {quantity} but only {pos.quantity} held",
                None,
            )

        # Calculate proceeds and costs
        gross_proceeds = (quantity * price).quantize(CASH_PRECISION, ROUND_HALF_UP)
        net_proceeds = (gross_proceeds - commission - slippage).quantize(
            CASH_PRECISION, ROUND_HALF_UP
        )

        # Calculate realized P&L
        cost_of_sold_shares = (
            pos.average_cost * quantity
        ).quantize(CASH_PRECISION, ROUND_HALF_UP)
        realized_pnl_gross = (gross_proceeds - cost_of_sold_shares).quantize(
            CASH_PRECISION, ROUND_HALF_UP
        )
        realized_pnl_net = (realized_pnl_gross - commission - slippage).quantize(
            CASH_PRECISION, ROUND_HALF_UP
        )

        if cost_of_sold_shares > 0:
            realized_pnl_pct = (
                (realized_pnl_net / cost_of_sold_shares) * Decimal("100")
            ).quantize(Decimal("0.01"), ROUND_HALF_UP)
        else:
            realized_pnl_pct = Decimal("0")

        # Add to cash (SELL increases cash)
        self.cash = (self.cash + net_proceeds).quantize(
            CASH_PRECISION, ROUND_HALF_UP
        )
        self.total_commissions += commission
        self.total_slippage += slippage
        self.total_sell_proceeds += net_proceeds

        # Update position
        old_quantity = pos.quantity
        new_quantity = old_quantity - quantity
        pos.quantity = new_quantity

        # Update cost basis (reduce by sold portion only)
        cost_basis_to_remove = (
            pos.average_cost * quantity
        ).quantize(CASH_PRECISION, ROUND_HALF_UP)
        pos.cost_basis = (pos.cost_basis - cost_basis_to_remove).quantize(
            CASH_PRECISION, ROUND_HALF_UP
        )

        # Average cost remains unchanged for remaining position
        if new_quantity == 0:
            pos.quantity = Decimal("0")
            pos.cost_basis = Decimal("0")
            pos.average_cost = Decimal("0")
            pos.state = PositionState.FLAT
        pos.current_price = price

        # Record completed trade
        trade = TradeRecord(
            symbol=symbol,
            signal_rating=signal_rating,
            entry_quantity=quantity,
            exit_quantity=quantity,
            entry_price=pos.average_cost,
            exit_price=price,
            gross_entry_cost=cost_of_sold_shares,
            entry_commission=Decimal("0"),  # Allocate original entry commission
            gross_exit_proceeds=gross_proceeds,
            exit_commission=commission,
            exit_slippage=slippage,
            realized_pnl_gross=realized_pnl_gross,
            realized_pnl_net=realized_pnl_net,
            realized_pnl_pct=realized_pnl_pct,
            status="CLOSED",
        )

        if signal_date:
            trade.exit_signal_date = signal_date

        self.completed_trades.append(trade)

        return True, "SELL executed", trade

    def record_dividend(
        self, symbol: str, amount: Decimal
    ) -> None:
        """Record dividend payment."""
        self.cash = (self.cash + amount).quantize(CASH_PRECISION, ROUND_HALF_UP)
        self.total_dividends += amount

    def record_deposit(self, amount: Decimal) -> None:
        """Record cash deposit."""
        self.cash = (self.cash + amount).quantize(CASH_PRECISION, ROUND_HALF_UP)
        self.total_deposits += amount

    def mark_positions_to_market(self, prices: Dict[str, Decimal]) -> None:
        """Update all positions with current market prices."""
        for symbol, price in prices.items():
            pos = self.get_position(symbol)
            pos.update_market_price(price)

    def get_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value: cash + invested value."""
        invested_value = Decimal("0")
        for pos in self.positions.values():
            if pos.quantity > 0:
                invested_value += pos.current_value

        return (self.cash + invested_value).quantize(
            CASH_PRECISION, ROUND_HALF_UP
        )

    def get_invested_value(self) -> Decimal:
        """Calculate total value of all open positions."""
        invested_value = Decimal("0")
        for pos in self.positions.values():
            if pos.quantity > 0:
                invested_value += pos.current_value
        return invested_value.quantize(CASH_PRECISION, ROUND_HALF_UP)

    def get_unrealized_pnl(self) -> Decimal:
        """Total unrealized P&L across all positions."""
        unrealized = Decimal("0")
        for pos in self.positions.values():
            if pos.quantity > 0:
                unrealized += pos.unrealized_pnl
        return unrealized.quantize(CASH_PRECISION, ROUND_HALF_UP)

    def get_realized_pnl(self) -> Decimal:
        """Total realized P&L from completed trades."""
        total = Decimal("0")
        for trade in self.completed_trades:
            total += trade.realized_pnl_net
        return total.quantize(CASH_PRECISION, ROUND_HALF_UP)

    def reconcile_cash(self) -> tuple[bool, Decimal]:
        """Reconcile cash ledger.
        
        Returns: (reconciled, error_amount)
        
        Expected cash equation:
            initial_cash
            + deposits
            + dividends
            + sell_proceeds (net)
            - buy_costs (gross + commission + slippage)
            = ending_cash
        """
        expected_cash = (
            self.initial_cash
            + self.total_deposits
            + self.total_dividends
            + self.total_sell_proceeds
            - self.total_buy_costs
        ).quantize(CASH_PRECISION, ROUND_HALF_UP)

        error = abs(expected_cash - self.cash)

        # Tolerance: 0.01 (one cent) 
        reconciled = error <= Decimal("0.01")

        return reconciled, error

    def reconcile_equity(self, reference_value: Decimal) -> tuple[bool, Decimal]:
        """Reconcile total equity against reference.
        
        Returns: (reconciled, error_amount)
        """
        calculated_equity = self.get_portfolio_value()
        error = abs(calculated_equity - reference_value)
        reconciled = error <= Decimal("0.01")
        return reconciled, error
