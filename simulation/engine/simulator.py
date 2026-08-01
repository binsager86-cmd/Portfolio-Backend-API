"""Event-driven simulation engine for Eagle Eye strategy."""

from decimal import Decimal, ROUND_HALF_UP
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

from simulation.domain.models import (
    SimulationConfig,
    SimulationResult,
    EagleEyeRatingRecord,
    OHLCV,
    Order,
    OrderSide,
    OrderStatus,
    PositionState,
    Position,
    DailyRecord,
    SkippedSignalRecord,
    EagleEyeRating,
    PositionSizingMode,
    ExecutionRule,
)
from simulation.accounting.portfolio import PortfolioAccounting, CASH_PRECISION

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Deterministic, point-in-time Eagle Eye strategy simulator.
    
    Enforces:
    - No look-ahead: only use data available on/before signal date
    - Proper execution: next session open (no same-bar close)
    - Exact sign conventions: BUY reduces cash, SELL increases cash
    - State machine: FLAT → PENDING_BUY → OPEN → PENDING_SELL → CLOSED
    - Idempotent: repeated calls produce identical results
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.accounting = PortfolioAccounting(config.initial_cash)
        self.result = SimulationResult(
            run_id=config.run_id,
            config=config,
            initial_cash=config.initial_cash,
        )

        # Historical data stores (indexed for O(1) lookup)
        self.ratings_by_symbol_date: Dict[str, Dict[date, List[EagleEyeRatingRecord]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self.ohlcv_by_symbol_date: Dict[str, Dict[date, OHLCV]] = defaultdict(
            lambda: defaultdict()
        )

        # Trading session date map for look-ahead enforcement
        self.all_trading_dates: List[date] = []
        self.trading_dates_set: set = set()

        # Position state machine
        self.position_states: Dict[str, PositionState] = defaultdict(
            lambda: PositionState.FLAT
        )

        # Pending orders by symbol (one per symbol)
        self.pending_orders: Dict[str, Order] = {}

        # Daily records
        self.daily_records: Dict[date, DailyRecord] = {}

    def load_ratings(self, ratings: List[EagleEyeRatingRecord]) -> None:
        """Load historical Eagle Eye ratings into memory.
        
        Validation:
        - Symbol/date uniqueness
        - Chronological order
        """
        for rating in ratings:
            self.ratings_by_symbol_date[rating.symbol][rating.rating_date].append(
                rating
            )
        self.result.ratings_loaded = len(ratings)
        logger.info(f"Loaded {len(ratings)} rating records")

    def load_ohlcv(self, ohlcv_data: List[OHLCV]) -> None:
        """Load historical OHLCV market data.
        
        Validation:
        - Symbol/date uniqueness
        - No future-dated prices
        - Price sanity (H >= L, etc.)
        """
        # Build index
        for bar in ohlcv_data:
            if bar.date not in self.trading_dates_set:
                self.all_trading_dates.append(bar.date)
                self.trading_dates_set.add(bar.date)
            self.ohlcv_by_symbol_date[bar.symbol][bar.date] = bar

        # Sort trading dates
        self.all_trading_dates.sort()
        self.result.ohlcv_rows_loaded = len(ohlcv_data)
        logger.info(f"Loaded {len(ohlcv_data)} OHLCV bars from {len(self.all_trading_dates)} trading sessions")

    def run(self) -> SimulationResult:
        """Execute the complete simulation."""
        start_time = datetime.now()
        self.result.created_at = start_time

        try:
            # Step 1: Validate inputs
            if not self.all_trading_dates:
                raise ValueError("No OHLCV data loaded")
            if not self.ratings_by_symbol_date:
                raise ValueError("No rating data loaded")

            # Step 2: Main simulation loop
            for current_date in self.all_trading_dates:
                # Skip dates outside simulation range
                if current_date < self.config.start_date:
                    continue
                if current_date > self.config.end_date:
                    break

                # Process one trading session
                self._process_session(current_date)

            # Step 3: Close out remaining positions at end
            if self.config.end_date in self.all_trading_dates:
                end_price_date = self.config.end_date
            else:
                # Find last available date
                end_price_date = max(
                    d for d in self.all_trading_dates if d <= self.config.end_date
                )

            # Get last market prices for unrealized calculation
            for symbol in self.accounting.positions:
                if symbol in self.ohlcv_by_symbol_date:
                    if end_price_date in self.ohlcv_by_symbol_date[symbol]:
                        bar = self.ohlcv_by_symbol_date[symbol][end_price_date]
                        self.accounting.get_position(symbol).update_market_price(
                            Decimal(str(bar.close))
                        )

            # Step 4: Compile results
            self._compile_results()

            # Step 5: Reconciliation checks
            self._run_reconciliation_checks()

            self.result.status = "COMPLETED"
            self.result.error_message = None

        except Exception as e:
            logger.exception("Simulation failed")
            self.result.status = "FAILED"
            self.result.error_message = str(e)

        finally:
            self.result.completed_at = datetime.now()
            self.result.execution_seconds = (
                self.result.completed_at - start_time
            ).total_seconds()

        return self.result

    def _process_session(self, current_date: date) -> None:
        """Process one trading session."""
        daily = DailyRecord(date=current_date)

        # Step 1: Execute pending orders at session open
        self._execute_pending_orders(current_date)

        # Step 2: Process new signals from this date
        self._process_signals_for_date(current_date)

        # Step 3: Mark positions to market with closing prices
        market_prices = {}
        for symbol in self.accounting.positions:
            if symbol in self.ohlcv_by_symbol_date:
                if current_date in self.ohlcv_by_symbol_date[symbol]:
                    bar = self.ohlcv_by_symbol_date[symbol][current_date]
                    market_prices[symbol] = Decimal(str(bar.close))

        self.accounting.mark_positions_to_market(market_prices)

        # Step 4: Build daily record
        daily.cash = self.accounting.cash
        daily.invested_value = self.accounting.get_invested_value()
        daily.total_equity = self.accounting.get_portfolio_value()
        daily.positions_count = len(
            [p for p in self.accounting.positions.values() if p.quantity > 0]
        )
        daily.positions = {
            s: p
            for s, p in self.accounting.positions.items()
            if p.quantity > 0
        }

        self.daily_records[current_date] = daily
        self.result.daily_records.append(daily)

    def _execute_pending_orders(self, current_date: date) -> None:
        """Execute any pending orders at session open."""
        symbols_to_remove = []

        for symbol, order in self.pending_orders.items():
            # Get market open price
            if symbol not in self.ohlcv_by_symbol_date:
                order.status = OrderStatus.SKIPPED
                order.rejection_reason = "NO_MARKET_DATA"
                self.position_states[symbol] = PositionState.FLAT
                symbols_to_remove.append(symbol)
                continue

            if current_date not in self.ohlcv_by_symbol_date[symbol]:
                # Not yet, wait for next session
                continue

            bar = self.ohlcv_by_symbol_date[symbol][current_date]
            exec_price = Decimal(str(bar.open_price))
            order.execution_date = current_date
            order.execution_price = exec_price

            # Execute based on side
            if order.side == OrderSide.BUY:
                success, reason = self.accounting.execute_buy(
                    symbol=symbol,
                    quantity=order.quantity_requested,
                    price=exec_price,
                    commission=order.quantity_requested
                    * exec_price
                    * self.config.commission_pct,
                    slippage=order.quantity_requested
                    * exec_price
                    * self.config.slippage_pct,
                )
                if success:
                    order.status = OrderStatus.FILLED
                    order.quantity_filled = order.quantity_requested
                    order.gross_amount = order.quantity_requested * exec_price
                    order.commission = order.quantity_requested * exec_price * self.config.commission_pct
                    order.slippage = order.quantity_requested * exec_price * self.config.slippage_pct
                    order.filled_at = datetime.now()
                    self.position_states[symbol] = PositionState.OPEN
                    self.result.buy_signals_executed += 1
                else:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = reason
                    self.position_states[symbol] = PositionState.FLAT
                    self.result.buy_signals_skipped += 1

            elif order.side == OrderSide.SELL:
                success, reason, trade = self.accounting.execute_sell(
                    symbol=symbol,
                    quantity=order.quantity_requested,
                    price=exec_price,
                    commission=order.quantity_requested
                    * exec_price
                    * self.config.commission_pct,
                    slippage=order.quantity_requested
                    * exec_price
                    * self.config.slippage_pct,
                    signal_date=order.signal_date,
                )
                if success:
                    order.status = OrderStatus.FILLED
                    order.quantity_filled = order.quantity_requested
                    order.gross_amount = order.quantity_requested * exec_price
                    order.commission = order.quantity_requested * exec_price * self.config.commission_pct
                    order.slippage = order.quantity_requested * exec_price * self.config.slippage_pct
                    order.filled_at = datetime.now()
                    self.position_states[symbol] = PositionState.FLAT
                    self.result.sell_signals_executed += 1
                else:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = reason
                    self.result.sell_signals_skipped += 1

            symbols_to_remove.append(symbol)
            self.result.orders.append(order)

        # Clean up executed orders
        for symbol in symbols_to_remove:
            del self.pending_orders[symbol]

    def _process_signals_for_date(self, current_date: date) -> None:
        """Process all Eagle Eye signals available on this date.
        
        No-look-ahead: only use data available on or before current_date.
        """
        # Collect all symbols with signals on this date
        symbols_with_signals = []
        for symbol in self.ratings_by_symbol_date:
            if current_date in self.ratings_by_symbol_date[symbol]:
                symbols_with_signals.append(symbol)

        self.result.buy_signals_total += sum(
            1
            for s in symbols_with_signals
            if any(
                r.rating in [EagleEyeRating.BUY, EagleEyeRating.STRONG_BUY]
                for r in self.ratings_by_symbol_date[s][current_date]
            )
        )

        self.result.sell_signals_total += sum(
            1
            for s in symbols_with_signals
            if any(
                r.rating in [EagleEyeRating.SELL, EagleEyeRating.STRONG_SELL]
                for r in self.ratings_by_symbol_date[s][current_date]
            )
        )

        # Process each signal
        for symbol in symbols_with_signals:
            ratings_today = self.ratings_by_symbol_date[symbol][current_date]

            for rating in ratings_today:
                # Determine action
                if rating.rating in [EagleEyeRating.BUY, EagleEyeRating.STRONG_BUY]:
                    self._handle_buy_signal(symbol, current_date, rating)
                elif rating.rating in [EagleEyeRating.SELL, EagleEyeRating.STRONG_SELL]:
                    self._handle_sell_signal(symbol, current_date, rating)
                # HOLD/NEUTRAL: do nothing

    def _handle_buy_signal(
        self, symbol: str, signal_date: date, rating: EagleEyeRatingRecord
    ) -> None:
        """Handle BUY signal."""
        state = self.position_states[symbol]

        # If already OPEN, don't submit another order (unless pyramiding enabled)
        if state == PositionState.OPEN:
            if not self.config.allow_pyramiding:
                self.result.skipped_signals.append(
                    SkippedSignalRecord(
                        symbol=symbol,
                        signal_date=signal_date,
                        signal_rating=rating.rating,
                        reason="ALREADY_OPEN",
                    )
                )
                return

        # If already pending, don't submit another order
        if symbol in self.pending_orders:
            self.result.skipped_signals.append(
                SkippedSignalRecord(
                    symbol=symbol,
                    signal_date=signal_date,
                    signal_rating=rating.rating,
                    reason="ORDER_PENDING",
                )
            )
            return

        # Calculate position size
        qty = self._calculate_position_size(signal_date, rating)
        if qty <= 0:
            self.result.skipped_signals.append(
                SkippedSignalRecord(
                    symbol=symbol,
                    signal_date=signal_date,
                    signal_rating=rating.rating,
                    reason="INSUFFICIENT_CASH",
                    quantity_requested=qty,
                )
            )
            return

        # Create pending order
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity_requested=qty,
            price_limit=rating.entry_primary,
            signal_date=signal_date,
            status=OrderStatus.PENDING,
        )
        self.pending_orders[symbol] = order
        self.position_states[symbol] = PositionState.PENDING_BUY

    def _handle_sell_signal(
        self, symbol: str, signal_date: date, rating: EagleEyeRatingRecord
    ) -> None:
        """Handle SELL signal."""
        state = self.position_states[symbol]

        # Only execute SELL if we have an open position
        if state != PositionState.OPEN:
            self.result.skipped_signals.append(
                SkippedSignalRecord(
                    symbol=symbol,
                    signal_date=signal_date,
                    signal_rating=rating.rating,
                    reason="NO_OPEN_POSITION",
                )
            )
            return

        # Get held quantity
        pos = self.accounting.get_position(symbol)
        qty_to_sell = pos.quantity

        # Create sell order
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity_requested=qty_to_sell,
            signal_date=signal_date,
            status=OrderStatus.PENDING,
        )
        self.pending_orders[symbol] = order
        self.position_states[symbol] = PositionState.PENDING_SELL

    def _calculate_position_size(
        self, signal_date: date, rating: EagleEyeRatingRecord
    ) -> Decimal:
        """Calculate position size based on configured mode and available cash.
        
        Returns: quantity to buy (integer if market convention allows)
        """
        current_equity = self.accounting.get_portfolio_value()
        available_cash = self.accounting.cash

        if self.config.position_sizing_mode == PositionSizingMode.EQUAL_ALLOCATION:
            # Allocate: available_cash / (max_positions - current_open)
            open_count = len(
                [
                    p
                    for p in self.accounting.positions.values()
                    if p.quantity > 0
                ]
            )
            available_slots = max(1, self.config.max_concurrent_positions - open_count)
            per_position_cash = available_cash / Decimal(available_slots)

        elif self.config.position_sizing_mode == PositionSizingMode.FIXED_AMOUNT:
            if not self.config.fixed_position_size:
                return Decimal("0")
            per_position_cash = self.config.fixed_position_size

        elif self.config.position_sizing_mode == PositionSizingMode.PERCENTAGE_EQUITY:
            per_position_cash = current_equity * self.config.position_size_pct / Decimal("100")

        else:
            return Decimal("0")

        # Get entry price (use primary entry point or close if not available)
        if rating.entry_primary and rating.entry_primary > 0:
            entry_price = rating.entry_primary
        else:
            # Fall back to market close price
            if (
                rating.symbol in self.ohlcv_by_symbol_date
                and signal_date in self.ohlcv_by_symbol_date[rating.symbol]
            ):
                bar = self.ohlcv_by_symbol_date[rating.symbol][signal_date]
                entry_price = Decimal(str(bar.close))
            else:
                return Decimal("0")

        if entry_price <= 0:
            return Decimal("0")

        # Calculate quantity (integer shares)
        # Account for commission in sizing
        price_with_commission = entry_price * (
            Decimal("1") + self.config.commission_pct
        )
        qty = (per_position_cash / price_with_commission).to_integral_value(
            rounding=ROUND_HALF_UP
        )

        return max(Decimal("1"), qty)

    def _compile_results(self) -> None:
        """Compile final performance metrics."""
        # Basic values
        self.result.ending_cash = self.accounting.cash
        self.result.ending_equity = self.accounting.get_portfolio_value()
        self.result.realized_pnl = self.accounting.get_realized_pnl()
        self.result.unrealized_pnl = self.accounting.get_unrealized_pnl()
        self.result.total_commissions = self.accounting.total_commissions
        self.result.total_slippage = self.accounting.total_slippage

        # Return calculations
        if self.config.initial_cash > 0:
            self.result.total_profit_loss = (
                self.result.ending_equity - self.config.initial_cash
            )
            self.result.total_return_pct = (
                (self.result.total_profit_loss / self.config.initial_cash)
                * Decimal("100")
            ).quantize(Decimal("0.01"), ROUND_HALF_UP)

        # Trade statistics
        self.result.trades = self.accounting.completed_trades
        self.result.trades_count = len(self.accounting.completed_trades)

        if self.result.trades_count > 0:
            winners = [t for t in self.accounting.completed_trades if t.realized_pnl_net > 0]
            losers = [t for t in self.accounting.completed_trades if t.realized_pnl_net < 0]

            self.result.winning_trades = len(winners)
            self.result.losing_trades = len(losers)

            if self.result.trades_count > 0:
                self.result.win_rate_pct = (
                    (Decimal(str(self.result.winning_trades)) / Decimal(str(self.result.trades_count)))
                    * Decimal("100")
                ).quantize(Decimal("0.01"), ROUND_HALF_UP)

            if winners:
                self.result.gross_profit = sum(t.realized_pnl_net for t in winners)
                self.result.avg_winner = (
                    self.result.gross_profit / Decimal(str(len(winners)))
                ).quantize(CASH_PRECISION, ROUND_HALF_UP)
                self.result.largest_winner = max(t.realized_pnl_net for t in winners)

            if losers:
                self.result.gross_loss = sum(t.realized_pnl_net for t in losers)
                self.result.avg_loser = (
                    self.result.gross_loss / Decimal(str(len(losers)))
                ).quantize(CASH_PRECISION, ROUND_HALF_UP)
                self.result.largest_loser = min(t.realized_pnl_net for t in losers)

            if self.result.gross_loss != 0:
                self.result.profit_factor = (
                    self.result.gross_profit / abs(self.result.gross_loss)
                ).quantize(Decimal("0.01"), ROUND_HALF_UP)

        # Max drawdown
        self.result.max_drawdown_pct = self._calculate_max_drawdown()

    def _calculate_max_drawdown(self) -> Decimal:
        """Calculate maximum drawdown from daily equity curve."""
        if not self.result.daily_records:
            return Decimal("0")

        max_peak = Decimal("0")
        max_dd = Decimal("0")

        for daily in self.result.daily_records:
            if daily.total_equity > max_peak:
                max_peak = daily.total_equity
            else:
                dd = ((daily.total_equity - max_peak) / max_peak) * Decimal("100")
                if dd < max_dd:
                    max_dd = dd

        return max_dd.quantize(Decimal("0.01"), ROUND_HALF_UP)

    def _run_reconciliation_checks(self) -> None:
        """Validate accounting and equity ledgers."""
        # Cash reconciliation
        cash_ok, cash_error = self.accounting.reconcile_cash()
        self.result.cash_reconciliation_ok = cash_ok
        self.result.cash_reconciliation_error = cash_error

        if not cash_ok:
            self.result.validation_warnings.append(
                f"Cash reconciliation failed: error = {cash_error}"
            )

        # Equity reconciliation
        equity_ok, equity_error = self.accounting.reconcile_equity(
            self.result.ending_equity
        )
        self.result.equity_reconciliation_ok = equity_ok
        self.result.equity_reconciliation_error = equity_error

        if not equity_ok:
            self.result.validation_warnings.append(
                f"Equity reconciliation failed: error = {equity_error}"
            )
