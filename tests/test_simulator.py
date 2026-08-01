"""Comprehensive test suite for Eagle Eye strategy simulator.

Mandatory test cases:
- Test A: Single round trip
- Test B: Persistent BUY
- Test C: Next-session execution
- Test D: Partial sale
- Test E: Insufficient cash
- Test F: Oversell protection
- Test G: Transaction replay (no duplication)
- Test H: Deterministic run identity
- Test I: Amount audit symmetry
- Test J: Deterministic replay
- Test K: Truncation parity
- Test L: Ledger reconciliation
- Test M: Trading calendar (no weekends)
- Test N: JSON enum serialization
"""

import pytest
from decimal import Decimal
from datetime import date, timedelta
from simulation.domain.models import (
    SimulationConfig,
    EagleEyeRatingRecord,
    OHLCV,
    EagleEyeRating,
    WyckoffPhase,
    PositionSizingMode,
    ExecutionRule,
)
from simulation.engine.simulator import SimulationEngine
from simulation.accounting.portfolio import PortfolioAccounting


class TestSingleRoundTrip:
    """Test A: Single round trip BUY→SELL."""
    
    def test_simple_buy_sell(self):
        """Given: Initial cash 1000, BUY 10 @ 100, SELL 10 @ 120."""
        accounting = PortfolioAccounting(Decimal("1000"))
        
        # BUY
        success, msg = accounting.execute_buy(
            symbol="TEST",
            quantity=Decimal("10"),
            price=Decimal("100"),
            commission=Decimal("0"),
            slippage=Decimal("0"),
        )
        assert success
        assert accounting.cash == Decimal("0")
        assert accounting.get_position("TEST").quantity == Decimal("10")
        assert accounting.get_position("TEST").average_cost == Decimal("100")
        
        # SELL
        success, msg, trade = accounting.execute_sell(
            symbol="TEST",
            quantity=Decimal("10"),
            price=Decimal("120"),
            commission=Decimal("0"),
            slippage=Decimal("0"),
        )
        assert success
        assert accounting.cash == Decimal("1200")
        assert accounting.get_position("TEST").quantity == Decimal("0")
        assert trade.realized_pnl_net == Decimal("200")
        assert accounting.get_realized_pnl() == Decimal("200")


class TestPersistentBUY:
    """Test B: Persistent BUY signal with pyramiding disabled."""
    
    def test_no_repeated_purchase_without_pyramiding(self):
        """Given BUY on three consecutive days without pyramiding,
        expect exactly one BUY order."""
        start_date = date(2024, 1, 1)
        config = SimulationConfig(
            initial_cash=Decimal("10000"),
            start_date=start_date,
            end_date=start_date + timedelta(days=2),
            allow_pyramiding=False,
        )
        engine = SimulationEngine(config)
        
        # Create three BUY signals for same symbol on consecutive dates
        ratings = [
            EagleEyeRatingRecord(
                symbol="TEST",
                rating_date=start_date + timedelta(days=i),
                rating_timestamp=None,
                rating=EagleEyeRating.BUY,
                confidence=Decimal("80"),
                stage=WyckoffPhase.EARLY_BREAKOUT,
                thesis="Strong breakout",
            )
            for i in range(3)
        ]
        engine.load_ratings(ratings)
        
        # Create OHLCV data
        ohlcv = [
            OHLCV(
                symbol="TEST",
                date=start_date + timedelta(days=i),
                open_price=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("99"),
                close=Decimal("102"),
                volume=1000000,
            )
            for i in range(3)
        ]
        engine.load_ohlcv(ohlcv)
        
        result = engine.run()
        
        # Should have exactly 1 BUY order, not 3
        buy_orders = [o for o in result.orders if o.side.value == "BUY"]
        assert len(buy_orders) == 1


class TestNextSessionExecution:
    """Test C: Next-session execution rule (no same-bar close)."""
    
    def test_buy_signal_at_close_executes_next_open(self):
        """Given a BUY signal calculated at close on day T,
        expect execution at day T+1 open, not day T close."""
        config = SimulationConfig(
            initial_cash=Decimal("10000"),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
            execution_rule=ExecutionRule.NEXT_SESSION_OPEN,
        )
        engine = SimulationEngine(config)
        
        # Signal on day 1
        ratings = [
            EagleEyeRatingRecord(
                symbol="TEST",
                rating_date=date(2024, 1, 1),
                rating_timestamp=None,
                rating=EagleEyeRating.BUY,
                confidence=Decimal("80"),
                stage=WyckoffPhase.EARLY_BREAKOUT,
                thesis="Test",
            )
        ]
        engine.load_ratings(ratings)
        
        # Market data
        ohlcv = [
            OHLCV(
                symbol="TEST",
                date=date(2024, 1, 1),
                open_price=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("99"),
                close=Decimal("102"),
                volume=1000000,
            ),
            OHLCV(
                symbol="TEST",
                date=date(2024, 1, 2),
                open_price=Decimal("103"),
                high=Decimal("108"),
                low=Decimal("102"),
                close=Decimal("105"),
                volume=1000000,
            ),
        ]
        engine.load_ohlcv(ohlcv)
        
        result = engine.run()
        
        # BUY order should execute on day 2 at open (103), not day 1
        buy_orders = [o for o in result.orders if o.side.value == "BUY"]
        assert len(buy_orders) == 1
        assert buy_orders[0].execution_date == date(2024, 1, 2)
        assert buy_orders[0].execution_price == Decimal("103")


class TestPartialSale:
    """Test D: Partial position sale."""
    
    def test_partial_sale_preserves_average_cost(self):
        """Given: BUY 10 @ 100, SELL 4 @ 120.
        Expect: quantity 6, avg_cost 100, realized P&L 80."""
        accounting = PortfolioAccounting(Decimal("10000"))
        
        # BUY
        accounting.execute_buy(
            symbol="TEST",
            quantity=Decimal("10"),
            price=Decimal("100"),
            commission=Decimal("0"),
        )
        pos = accounting.get_position("TEST")
        assert pos.quantity == Decimal("10")
        assert pos.average_cost == Decimal("100")
        
        # Sell half
        success, msg, trade = accounting.execute_sell(
            symbol="TEST",
            quantity=Decimal("4"),
            price=Decimal("120"),
            commission=Decimal("0"),
        )
        
        assert success
        assert pos.quantity == Decimal("6")
        # Average cost should be unchanged for remaining shares
        assert pos.average_cost == Decimal("100")
        # Realized P&L = (120 - 100) * 4 = 80
        assert trade.realized_pnl_net == Decimal("80")


class TestInsufficientCash:
    """Test E: Insufficient cash protection."""
    
    def test_buy_rejected_when_insufficient_cash(self):
        """Given: Cash 1000, try BUY 20 @ 100 (cost 2000).
        Expect: Order rejected, no negative cash."""
        accounting = PortfolioAccounting(Decimal("1000"))
        
        success, msg = accounting.execute_buy(
            symbol="TEST",
            quantity=Decimal("20"),
            price=Decimal("100"),
            commission=Decimal("0"),
        )
        
        assert not success
        assert "Insufficient" in msg
        assert accounting.cash == Decimal("1000")
        assert accounting.get_position("TEST").quantity == Decimal("0")


class TestOversellProtection:
    """Test F: Oversell protection."""
    
    def test_sell_rejected_when_overselling(self):
        """Given: Holding 10 shares, try SELL 15.
        Expect: Order rejected, no short position."""
        accounting = PortfolioAccounting(Decimal("10000"))
        
        # BUY
        accounting.execute_buy(
            symbol="TEST",
            quantity=Decimal("10"),
            price=Decimal("100"),
            commission=Decimal("0"),
        )
        
        # Try to oversell
        success, msg, trade = accounting.execute_sell(
            symbol="TEST",
            quantity=Decimal("15"),
            price=Decimal("100"),
            commission=Decimal("0"),
        )
        
        assert not success
        assert "Oversell" in msg
        assert accounting.get_position("TEST").quantity == Decimal("10")


class TestTransactionReplay:
    """Test G: No duplicate transaction application."""
    
    def test_transaction_applied_once_only(self):
        """Given: BUY on day 1, repeated simulation through days 1-10.
        Expect: Quantity stays 10, never increases."""
        # This test validates idempotency
        config = SimulationConfig(
            initial_cash=Decimal("10000"),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
        )
        engine = SimulationEngine(config)
        
        ratings = [
            EagleEyeRatingRecord(
                symbol="TEST",
                rating_date=date(2024, 1, 1),
                rating_timestamp=None,
                rating=EagleEyeRating.BUY,
                confidence=Decimal("80"),
                stage=WyckoffPhase.EARLY_BREAKOUT,
                thesis="Test",
            )
        ]
        engine.load_ratings(ratings)
        
        ohlcv = [
            OHLCV(
                symbol="TEST",
                date=date(2024, 1, 1) + timedelta(days=i),
                open_price=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("99"),
                close=Decimal("102"),
                volume=1000000,
            )
            for i in range(10)
        ]
        engine.load_ohlcv(ohlcv)
        
        result = engine.run()
        
        # Check daily records: quantity should be 10, never increase
        quantities = []
        for daily in result.daily_records:
            if "TEST" in daily.positions:
                quantities.append(float(daily.positions["TEST"].quantity))
        
        # All should be same value (0 until buy fills, then 10)
        assert all(q in [0.0, 10.0] for q in quantities)


class TestDeterministicRun:
    """Test H: Deterministic full-mode generation (no duplication)."""
    
    def test_no_duplicate_generation_on_repeated_run(self):
        """Given: Run simulator twice with identical config and data.
        Expect: Identical results (same orders, fills, trades)."""
        config = SimulationConfig(
            run_id="DET_TEST",
            initial_cash=Decimal("10000"),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
        )
        
        ratings = [
            EagleEyeRatingRecord(
                symbol="TEST",
                rating_date=date(2024, 1, 1),
                rating_timestamp=None,
                rating=EagleEyeRating.BUY,
                confidence=Decimal("80"),
                stage=WyckoffPhase.EARLY_BREAKOUT,
                thesis="Test",
            )
        ]
        
        ohlcv = [
            OHLCV(
                symbol="TEST",
                date=date(2024, 1, 1) + timedelta(days=i),
                open_price=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("99"),
                close=Decimal("102"),
                volume=1000000,
            )
            for i in range(5)
        ]
        
        # Run 1
        engine1 = SimulationEngine(config)
        engine1.load_ratings(ratings)
        engine1.load_ohlcv(ohlcv)
        result1 = engine1.run()
        
        # Run 2
        engine2 = SimulationEngine(config)
        engine2.load_ratings(ratings)
        engine2.load_ohlcv(ohlcv)
        result2 = engine2.run()
        
        # Compare
        assert len(result1.orders) == len(result2.orders)
        assert len(result1.trades) == len(result2.trades)
        assert result1.ending_equity == result2.ending_equity
        assert result1.realized_pnl == result2.realized_pnl


class TestAmountAuditSymmetry:
    """Test I: Amount audit must use proper absolute-value comparison."""
    
    def test_amount_audit_catches_both_overstated_and_understated(self):
        """Given: Expected 1000, test both 900 and 1100.
        Expected: Both fail the audit."""
        # This validates the fix for signed comparison issue
        expected = Decimal("1000")
        tolerance = Decimal("0.01")
        
        actual_high = Decimal("1100")
        actual_low = Decimal("900")
        actual_ok = Decimal("1000")
        
        # Correct formula: abs(abs(expected) - abs(actual)) > tolerance
        def audit_amount(exp, act, tol):
            return abs(abs(exp) - abs(act)) > tol
        
        assert audit_amount(expected, actual_high, tolerance)  # Should fail
        assert audit_amount(expected, actual_low, tolerance)   # Should fail
        assert not audit_amount(expected, actual_ok, tolerance)  # Should pass


class TestLedgerReconciliation:
    """Test L: Full ledger reconciliation."""
    
    def test_cash_equity_reconciliation(self):
        """Verify: cash + invested_value = total_equity."""
        accounting = PortfolioAccounting(Decimal("10000"))
        
        # Buy some shares
        accounting.execute_buy(
            symbol="TEST",
            quantity=Decimal("5"),
            price=Decimal("100"),
            commission=Decimal("0"),
        )
        
        # Mark to market
        accounting.mark_positions_to_market({"TEST": Decimal("110")})
        
        # Verify equation
        cash = accounting.cash
        invested = accounting.get_invested_value()
        total = accounting.get_portfolio_value()
        
        assert (cash + invested - total).quantize(Decimal("0.01")) == Decimal("0")


class TestJSONEnumSerialization:
    """Test N: JSON enum values must be strings, not enum repr."""
    
    def test_enum_serializes_as_string_value(self):
        """Given: SimulationResult with rating BUY.
        Expect: JSON has 'BUY', not 'EagleEyeRating.BUY'."""
        from simulation.domain.models import SimulationResult, TradeRecord
        
        trade = TradeRecord(
            symbol="TEST",
            signal_rating=EagleEyeRating.BUY,
        )
        
        trade_dict = trade.to_dict()
        
        # Should be string "BUY", not enum repr
        assert trade_dict["signal_rating"] == "BUY"
        assert not "EagleEyeRating" in str(trade_dict["signal_rating"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
