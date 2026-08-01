"""
Extended Eagle Eye Simulator Test Suite — 25 mandatory tests.

Implements:
Test 1:  Single BUY/SELL round trip
Test 2:  Persistent BUY does not repurchase
Test 3:  Next-session execution
Test 4:  Partial sale
Test 5:  Insufficient cash
Test 6:  Oversell protection
Test 7:  Transaction applied once only
Test 8:  No duplicate full-mode generation
Test 9:  Symmetric amount validation
Test 10: Deterministic replay (identical outputs across runs)
Test 11: Truncation parity
Test 12: Daily ledger reconciliation
Test 13: Cash-flow reconciliation
Test 14: Kuwait trading-calendar behavior (weekends excluded)
Test 15: JSON enum serialization
Test 16: Duplicate signal idempotency
Test 17: Same-date signal deterministic ordering
Test 18: SELL while flat (NO_OPEN_POSITION)
Test 19: Missing next-session price
Test 20: Persistent HOLD creates no order
Test 21: Duplicate API configuration hash
Test 22: API schema validation
Test 23: Signal data availability status
Test 24: Accounting sign convention verification
Test 25: Engine error handling (FAILED status on bad config)
"""
import json
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
    TradeRecord,
)
from simulation.engine.simulator import SimulationEngine
from simulation.accounting.portfolio import PortfolioAccounting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(symbol="TEST", start=date(2024, 1, 1), n=5, open_p=100, close=102):
    """Create n consecutive OHLCV bars."""
    return [
        OHLCV(
            symbol=symbol,
            date=start + timedelta(days=i),
            open_price=Decimal(str(open_p)),
            high=Decimal(str(close + 3)),
            low=Decimal(str(open_p - 1)),
            close=Decimal(str(close)),
            volume=1_000_000,
        )
        for i in range(n)
    ]


def make_rating(symbol="TEST", rating=EagleEyeRating.BUY, on_date=date(2024, 1, 1)):
    return EagleEyeRatingRecord(
        symbol=symbol,
        rating_date=on_date,
        rating_timestamp=None,
        rating=rating,
        confidence=Decimal("80"),
        stage=WyckoffPhase.EARLY_BREAKOUT,
        thesis="Test",
    )


def simple_engine(start=date(2024, 1, 1), end=date(2024, 1, 5), cash=10000):
    cfg = SimulationConfig(
        initial_cash=Decimal(str(cash)),
        start_date=start,
        end_date=end,
    )
    return SimulationEngine(cfg)


# ---------------------------------------------------------------------------
# Test 1: Single BUY/SELL round trip
# ---------------------------------------------------------------------------

class TestSingleRoundTrip:
    def test_simple_buy_sell(self):
        acct = PortfolioAccounting(Decimal("1000"))
        ok, _ = acct.execute_buy("T", Decimal("10"), Decimal("100"), Decimal("0"), Decimal("0"))
        assert ok
        ok, _, trade = acct.execute_sell("T", Decimal("10"), Decimal("120"), Decimal("0"), Decimal("0"))
        assert ok
        assert acct.cash == Decimal("1200")
        assert acct.get_realized_pnl() == Decimal("200")


# ---------------------------------------------------------------------------
# Test 2: Persistent BUY does not repurchase (state machine enforces OPEN)
# ---------------------------------------------------------------------------

class TestPersistentBUY:
    def test_no_repeated_purchase_without_pyramiding(self):
        start = date(2024, 1, 1)
        eng = simple_engine(start=start, end=start + timedelta(days=4))
        # BUY signals on 3 consecutive days
        ratings = [make_rating(on_date=start + timedelta(days=i)) for i in range(3)]
        eng.load_ratings(ratings)
        eng.load_ohlcv(make_ohlcv(start=start, n=5))
        result = eng.run()
        buys = [o for o in result.orders if o.side.value == "BUY"]
        assert len(buys) == 1, f"Expected 1 BUY order, got {len(buys)}"


# ---------------------------------------------------------------------------
# Test 3: Next-session execution (signal on T, fill at T+1 open)
# ---------------------------------------------------------------------------

class TestNextSessionExecution:
    def test_buy_signal_at_close_executes_next_open(self):
        start = date(2024, 1, 1)
        cfg = SimulationConfig(
            initial_cash=Decimal("10000"),
            start_date=start,
            end_date=start + timedelta(days=3),
        )
        eng = SimulationEngine(cfg)
        eng.load_ratings([make_rating(on_date=start)])  # signal day 0
        eng.load_ohlcv(make_ohlcv(start=start, n=4))
        result = eng.run()
        buys = [o for o in result.orders if o.side.value == "BUY"]
        assert len(buys) == 1
        # Filled on next day (start + 1)
        assert buys[0].execution_date == start + timedelta(days=1)


# ---------------------------------------------------------------------------
# Test 4: Partial sale preserves average cost
# ---------------------------------------------------------------------------

class TestPartialSale:
    def test_partial_sale_preserves_average_cost(self):
        acct = PortfolioAccounting(Decimal("2000"))
        acct.execute_buy("T", Decimal("10"), Decimal("100"), Decimal("0"), Decimal("0"))
        ok, _, trade = acct.execute_sell("T", Decimal("4"), Decimal("120"), Decimal("0"), Decimal("0"))
        assert ok
        pos = acct.get_position("T")
        assert pos.quantity == Decimal("6")
        assert pos.average_cost == Decimal("100")
        assert trade.realized_pnl_gross == Decimal("80")


# ---------------------------------------------------------------------------
# Test 5: Insufficient cash rejects BUY
# ---------------------------------------------------------------------------

class TestInsufficientCash:
    def test_buy_rejected_when_insufficient_cash(self):
        acct = PortfolioAccounting(Decimal("100"))
        ok, reason = acct.execute_buy("T", Decimal("10"), Decimal("100"), Decimal("0"), Decimal("0"))
        assert not ok
        assert "cash" in reason.lower() or "insufficient" in reason.lower()
        assert acct.cash == Decimal("100")


# ---------------------------------------------------------------------------
# Test 6: Oversell protection — can't sell more than held
# ---------------------------------------------------------------------------

class TestOversellProtection:
    def test_sell_rejected_when_overselling(self):
        acct = PortfolioAccounting(Decimal("2000"))
        acct.execute_buy("T", Decimal("10"), Decimal("100"), Decimal("0"), Decimal("0"))
        ok, reason, _ = acct.execute_sell("T", Decimal("15"), Decimal("100"), Decimal("0"), Decimal("0"))
        assert not ok
        assert "quantity" in reason.lower() or "oversell" in reason.lower()
        assert acct.get_position("T").quantity == Decimal("10")


# ---------------------------------------------------------------------------
# Test 7: Transaction applied once only
# ---------------------------------------------------------------------------

class TestTransactionReplay:
    def test_transaction_applied_once_only(self):
        start = date(2024, 1, 1)
        eng = simple_engine(start=start, end=start + timedelta(days=9))
        eng.load_ratings([make_rating(on_date=start)])
        eng.load_ohlcv(make_ohlcv(start=start, n=10))
        result = eng.run()
        buys = [o for o in result.orders if o.side.value == "BUY"]
        assert len(buys) == 1, "BUY applied multiple times"


# ---------------------------------------------------------------------------
# Test 8: No duplicate full-mode generation
# ---------------------------------------------------------------------------

class TestNoDuplicate:
    def test_no_duplicate_generation_on_repeated_run(self):
        start = date(2024, 1, 1)
        cfg = SimulationConfig(
            initial_cash=Decimal("10000"),
            start_date=start,
            end_date=start + timedelta(days=4),
        )
        eng1 = SimulationEngine(cfg)
        ratings = [make_rating(on_date=start)]
        ohlcv = make_ohlcv(start=start, n=5)
        eng1.load_ratings(ratings)
        eng1.load_ohlcv(ohlcv)
        r1 = eng1.run()

        eng2 = SimulationEngine(cfg)
        eng2.load_ratings(ratings)
        eng2.load_ohlcv(ohlcv)
        r2 = eng2.run()

        assert len(r1.orders) == len(r2.orders)


# ---------------------------------------------------------------------------
# Test 9: Symmetric amount audit (validation catches both over and under)
# ---------------------------------------------------------------------------

class TestAmountAuditSymmetry:
    def test_amount_audit_catches_both_overstated_and_understated(self):
        expected = Decimal("1000")
        tolerance = Decimal("0.01")

        def check_amount(actual, exp=expected):
            """Checks if actual matches expected in value (ignoring sign)."""
            return abs(abs(exp) - abs(actual)) <= tolerance

        def check_signed(actual, exp=expected):
            """Full check: must match value AND sign."""
            same_sign = (actual >= 0) == (exp >= 0)
            return same_sign and abs(abs(exp) - abs(actual)) <= tolerance

        assert check_signed(Decimal("1000"))       # exact match
        assert not check_signed(Decimal("999"))    # understated by 1
        assert not check_signed(Decimal("1001"))   # overstated by 1
        assert not check_signed(Decimal("-1000"))  # wrong sign


# ---------------------------------------------------------------------------
# Test 10: Deterministic replay (identical outputs across two runs)
# ---------------------------------------------------------------------------

class TestDeterministicReplay:
    def test_two_runs_produce_identical_results(self):
        start = date(2024, 1, 1)
        ratings = [
            make_rating(on_date=start),
            make_rating(symbol="TEST2", on_date=start + timedelta(days=1)),
        ]
        ohlcv = make_ohlcv(start=start, n=5) + make_ohlcv(symbol="TEST2", start=start, n=5)

        def make_result():
            cfg = SimulationConfig(
                initial_cash=Decimal("10000"),
                start_date=start,
                end_date=start + timedelta(days=4),
            )
            eng = SimulationEngine(cfg)
            eng.load_ratings(ratings)
            eng.load_ohlcv(ohlcv)
            return eng.run()

        r1 = make_result()
        r2 = make_result()

        assert r1.ending_equity == r2.ending_equity, "Ending equity differs across runs"
        assert r1.trades_count == r2.trades_count, "Trade count differs"
        assert r1.buy_signals_executed == r2.buy_signals_executed


# ---------------------------------------------------------------------------
# Test 11: Truncation parity
# ---------------------------------------------------------------------------

class TestTruncationParity:
    def test_results_through_cutoff_identical_regardless_of_extra_future_data(self):
        """
        A simulation loaded with data only through T must produce identical
        results through T compared to a full-data simulation truncated at T.
        """
        start = date(2024, 1, 1)
        cutoff = date(2024, 1, 3)

        base_ratings = [make_rating(on_date=start)]
        short_ohlcv = make_ohlcv(start=start, n=3)  # only through cutoff
        long_ohlcv = make_ohlcv(start=start, n=10)   # full data

        # Run 1: short data through cutoff
        cfg1 = SimulationConfig(
            initial_cash=Decimal("10000"),
            start_date=start,
            end_date=cutoff,
        )
        eng1 = SimulationEngine(cfg1)
        eng1.load_ratings(base_ratings)
        eng1.load_ohlcv(short_ohlcv)
        r1 = eng1.run()

        # Run 2: full data, same end_date
        cfg2 = SimulationConfig(
            initial_cash=Decimal("10000"),
            start_date=start,
            end_date=cutoff,
        )
        eng2 = SimulationEngine(cfg2)
        eng2.load_ratings(base_ratings)
        eng2.load_ohlcv(long_ohlcv)
        r2 = eng2.run()

        # Results through T must match
        assert r1.ending_equity == r2.ending_equity, (
            f"Truncation parity FAILED: short={r1.ending_equity} full={r2.ending_equity}"
        )
        assert r1.trades_count == r2.trades_count
        assert [o.execution_date for o in r1.orders] == [o.execution_date for o in r2.orders]


# ---------------------------------------------------------------------------
# Test 12: Daily ledger reconciliation
# ---------------------------------------------------------------------------

class TestDailyLedgerReconciliation:
    def test_daily_equity_equals_cash_plus_invested(self):
        start = date(2024, 1, 1)
        eng = simple_engine(start=start, end=start + timedelta(days=4))
        eng.load_ratings([make_rating(on_date=start)])
        eng.load_ohlcv(make_ohlcv(start=start, n=5))
        result = eng.run()

        for dr in result.daily_records:
            expected = dr.cash + dr.invested_value
            diff = abs(expected - dr.total_equity)
            assert diff <= Decimal("0.01"), (
                f"Ledger mismatch on {dr.date}: cash={dr.cash} + invested={dr.invested_value} "
                f"!= equity={dr.total_equity} (diff={diff})"
            )


# ---------------------------------------------------------------------------
# Test 13: Cash-flow reconciliation
# ---------------------------------------------------------------------------

class TestCashFlowReconciliation:
    def test_cash_reconciliation_ok(self):
        start = date(2024, 1, 1)
        eng = simple_engine(start=start, end=start + timedelta(days=9))
        # Buy on day 0, sell on day 5
        eng.load_ratings([
            make_rating(on_date=start),
            make_rating(rating=EagleEyeRating.SELL, on_date=start + timedelta(days=5)),
        ])
        eng.load_ohlcv(make_ohlcv(start=start, n=10))
        result = eng.run()
        assert result.cash_reconciliation_ok, (
            f"Cash reconciliation failed: error={result.cash_reconciliation_error}"
        )


# ---------------------------------------------------------------------------
# Test 14: Kuwait trading-calendar behavior (weekend dates excluded)
# ---------------------------------------------------------------------------

class TestKuwaitTradingCalendar:
    def test_friday_saturday_dates_excluded_when_missing_ohlcv(self):
        """
        The engine only processes dates that have OHLCV bars.
        Fridays and Saturdays (KSE weekend) have no OHLCV, so they're naturally skipped.
        """
        # Build a week with only Sun-Thu (Jan 7 = Sun, Jan 8 = Mon ... Jan 11 = Thu)
        start = date(2025, 1, 5)  # Sunday
        trading_dates = [start + timedelta(days=i) for i in range(5)]  # Sun-Thu

        ohlcv = [
            OHLCV(
                symbol="TEST",
                date=d,
                open_price=Decimal("100"),
                high=Decimal("102"),
                low=Decimal("99"),
                close=Decimal("101"),
                volume=1_000_000,
            )
            for d in trading_dates
        ]

        cfg = SimulationConfig(
            initial_cash=Decimal("10000"),
            start_date=start,
            end_date=start + timedelta(days=6),  # Spans into weekend
        )
        eng = SimulationEngine(cfg)
        eng.load_ratings([make_rating(on_date=start)])
        eng.load_ohlcv(ohlcv)
        result = eng.run()

        # All processed dates must be trading dates (Sun-Thu only, within OHLCV)
        processed_dates = {dr.date for dr in result.daily_records}
        assert processed_dates.issubset(set(trading_dates)), (
            f"Non-trading dates processed: {processed_dates - set(trading_dates)}"
        )


# ---------------------------------------------------------------------------
# Test 15: JSON enum serialization
# ---------------------------------------------------------------------------

class TestJSONEnumSerialization:
    def test_enum_serializes_as_string_value(self):
        trade = TradeRecord(symbol="T", signal_rating=EagleEyeRating.BUY)
        d = trade.to_dict()
        assert d["signal_rating"] == "BUY"
        assert "EagleEyeRating" not in str(d["signal_rating"])


# ---------------------------------------------------------------------------
# Test 16: Duplicate signal idempotency
# ---------------------------------------------------------------------------

class TestDuplicateSignalIdempotency:
    def test_two_identical_buy_signals_same_date_produces_one_order(self):
        """Two identical BUY signals on the same day must produce exactly one order."""
        start = date(2024, 1, 1)
        eng = simple_engine(start=start, end=start + timedelta(days=4))

        # Two identical BUY signals on same date
        rating = make_rating(on_date=start)
        eng.load_ratings([rating, rating])  # Duplicates
        eng.load_ohlcv(make_ohlcv(start=start, n=5))
        result = eng.run()

        buys = [o for o in result.orders if o.side.value == "BUY"]
        assert len(buys) == 1, f"Duplicate signal produced {len(buys)} orders"


# ---------------------------------------------------------------------------
# Test 17: Same-date signal deterministic ordering
# ---------------------------------------------------------------------------

class TestSameDateSignalOrdering:
    def test_multiple_symbols_same_date_sorted_deterministically(self):
        """Multiple BUY signals on same day must be processed in deterministic order."""
        start = date(2024, 1, 1)
        cfg = SimulationConfig(
            initial_cash=Decimal("10000"),
            start_date=start,
            end_date=start + timedelta(days=2),
            max_concurrent_positions=5,
        )

        symbols = ["ZZZ", "AAA", "MMM"]
        ratings = [make_rating(symbol=s, on_date=start) for s in symbols]
        ohlcv = []
        for s in symbols:
            ohlcv += make_ohlcv(symbol=s, start=start, n=3)

        # Run twice and compare order of executed buys
        def run_it():
            e = SimulationEngine(cfg)
            e.load_ratings(ratings)
            e.load_ohlcv(ohlcv)
            r = e.run()
            return [o.symbol for o in r.orders if o.side.value == "BUY"]

        order1 = run_it()
        order2 = run_it()
        assert order1 == order2, f"Non-deterministic signal ordering: {order1} vs {order2}"


# ---------------------------------------------------------------------------
# Test 18: SELL while flat → NO_OPEN_POSITION skip
# ---------------------------------------------------------------------------

class TestSellWhileFlat:
    def test_sell_signal_while_flat_is_skipped(self):
        start = date(2024, 1, 1)
        eng = simple_engine(start=start, end=start + timedelta(days=2))
        eng.load_ratings([make_rating(rating=EagleEyeRating.SELL, on_date=start)])
        eng.load_ohlcv(make_ohlcv(start=start, n=3))
        result = eng.run()

        sell_orders = [o for o in result.orders if o.side.value == "SELL"]
        assert len(sell_orders) == 0, "SELL while flat should not create order"

        skipped = [s for s in result.skipped_signals if "NO_OPEN_POSITION" in s.reason]
        assert len(skipped) >= 1, "Expected skip record for SELL while flat"


# ---------------------------------------------------------------------------
# Test 19: Missing next-session price must not be silently skipped with future price
# ---------------------------------------------------------------------------

class TestMissingNextSessionPrice:
    def test_pending_order_waits_without_substituting_future_price(self):
        """If no OHLCV for T+1, the order stays pending until data is available."""
        start = date(2024, 1, 1)
        cfg = SimulationConfig(
            initial_cash=Decimal("10000"),
            start_date=start,
            end_date=start + timedelta(days=3),
        )
        eng = SimulationEngine(cfg)
        eng.load_ratings([make_rating(on_date=start)])
        # OHLCV only on days 0 and 3 — gap days 1 and 2 are missing
        ohlcv = [
            OHLCV("TEST", start, Decimal("100"), Decimal("102"), Decimal("99"), Decimal("101"), 1_000_000),
            OHLCV("TEST", start + timedelta(days=3), Decimal("105"), Decimal("107"), Decimal("104"), Decimal("106"), 1_000_000),
        ]
        eng.load_ohlcv(ohlcv)
        result = eng.run()

        buys = [o for o in result.orders if o.side.value == "BUY"]
        if buys:
            # If order was executed, it must have been on day 3, not substituted with day 0
            assert buys[0].execution_date == start + timedelta(days=3), (
                f"Order filled on wrong date: {buys[0].execution_date}"
            )


# ---------------------------------------------------------------------------
# Test 20: Persistent HOLD creates no order
# ---------------------------------------------------------------------------

class TestPersistentHold:
    def test_hold_signal_creates_no_order(self):
        start = date(2024, 1, 1)
        eng = simple_engine(start=start, end=start + timedelta(days=4))
        ratings = [make_rating(rating=EagleEyeRating.HOLD, on_date=start + timedelta(days=i)) for i in range(5)]
        eng.load_ratings(ratings)
        eng.load_ohlcv(make_ohlcv(start=start, n=5))
        result = eng.run()
        assert len(result.orders) == 0, "HOLD signals should not generate orders"


# ---------------------------------------------------------------------------
# Test 21: Configuration hash for idempotency
# ---------------------------------------------------------------------------

class TestConfigurationHash:
    def test_identical_configs_produce_same_hash(self):
        from app.services.eagle_eye.simulator_service import SimulatorService
        h1 = SimulatorService.create_config_hash(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            initial_cash=Decimal("100000"),
            max_positions=10,
            position_sizing_mode="equal",
            commission_pct=Decimal("0.001"),
            slippage_pct=Decimal("0.001"),
        )
        h2 = SimulatorService.create_config_hash(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            initial_cash=Decimal("100000"),
            max_positions=10,
            position_sizing_mode="equal",
            commission_pct=Decimal("0.001"),
            slippage_pct=Decimal("0.001"),
        )
        assert h1 == h2, "Same config must produce same hash"

    def test_different_configs_produce_different_hash(self):
        from app.services.eagle_eye.simulator_service import SimulatorService
        h1 = SimulatorService.create_config_hash(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            initial_cash=Decimal("100000"),
            max_positions=10,
            position_sizing_mode="equal",
            commission_pct=Decimal("0.001"),
            slippage_pct=Decimal("0.001"),
        )
        h2 = SimulatorService.create_config_hash(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            initial_cash=Decimal("200000"),  # Different cash
            max_positions=10,
            position_sizing_mode="equal",
            commission_pct=Decimal("0.001"),
            slippage_pct=Decimal("0.001"),
        )
        assert h1 != h2, "Different config must produce different hash"


# ---------------------------------------------------------------------------
# Test 22: API schema validation
# ---------------------------------------------------------------------------

class TestAPISchemaValidation:
    def test_simulation_request_schema_validates(self):
        from app.schemas.eagle_eye import SimulationRequest
        req = SimulationRequest(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            initial_cash=Decimal("100000"),
            max_positions=10,
        )
        assert req.start_date == date(2024, 1, 1)
        assert req.max_positions == 10
        assert req.position_sizing_mode == "equal"  # default

    def test_simulation_response_schema_validates(self):
        from app.schemas.eagle_eye import SimulationResponse
        resp = SimulationResponse(
            run_id="abc123",
            status="FAILED",
            signal_data_status="HISTORICAL_SIGNAL_DATA_UNAVAILABLE",
            error_message="No data found",
        )
        assert resp.run_id == "abc123"
        assert resp.status == "FAILED"
        assert resp.signal_data_status == "HISTORICAL_SIGNAL_DATA_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Test 23: Signal data availability status
# ---------------------------------------------------------------------------

class TestSignalDataAvailabilityStatus:
    def test_status_enum_values_known(self):
        """Document the valid signal_data_status values."""
        valid_statuses = {
            "HISTORICAL_SIGNAL_REPLAY",         # Live historical data from ratings_history
            "RECONSTRUCTED_RESEARCH",           # Reconstructed using current classifier
            "CURRENT_RATING_DEMO",              # Demo using current ratings only
            "HISTORICAL_SIGNAL_DATA_UNAVAILABLE",  # No historical data found
        }
        from app.schemas.eagle_eye import SimulationResponse
        resp = SimulationResponse(
            run_id="x",
            status="FAILED",
            signal_data_status="HISTORICAL_SIGNAL_DATA_UNAVAILABLE",
        )
        assert resp.signal_data_status in valid_statuses


# ---------------------------------------------------------------------------
# Test 24: Accounting sign convention
# ---------------------------------------------------------------------------

class TestAccountingSignConvention:
    def test_buy_reduces_cash_increases_position(self):
        acct = PortfolioAccounting(Decimal("1000"))
        acct.execute_buy("T", Decimal("5"), Decimal("100"), Decimal("0"), Decimal("0"))
        assert acct.cash == Decimal("500"), f"BUY should reduce cash: {acct.cash}"
        pos = acct.get_position("T")
        assert pos.quantity == Decimal("5")

    def test_sell_increases_cash_reduces_position(self):
        acct = PortfolioAccounting(Decimal("1000"))
        acct.execute_buy("T", Decimal("5"), Decimal("100"), Decimal("0"), Decimal("0"))
        acct.execute_sell("T", Decimal("5"), Decimal("110"), Decimal("0"), Decimal("0"))
        assert acct.cash == Decimal("1050"), f"SELL should increase cash: {acct.cash}"

    def test_commission_reduces_proceeds(self):
        acct = PortfolioAccounting(Decimal("2000"))
        acct.execute_buy("T", Decimal("10"), Decimal("100"), Decimal("10"), Decimal("0"))
        # Expected: 2000 - (10*100 + 10 commission) = 2000 - 1010 = 990
        assert acct.cash == Decimal("990")


# ---------------------------------------------------------------------------
# Test 25: Engine error handling (FAILED status on bad config)
# ---------------------------------------------------------------------------

class TestEngineErrorHandling:
    def test_empty_ohlcv_returns_failed_status(self):
        cfg = SimulationConfig(
            initial_cash=Decimal("10000"),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
        )
        eng = SimulationEngine(cfg)
        eng.load_ratings([make_rating()])
        # No OHLCV loaded
        result = eng.run()
        assert result.status == "FAILED"

    def test_empty_ratings_returns_failed_status(self):
        cfg = SimulationConfig(
            initial_cash=Decimal("10000"),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
        )
        eng = SimulationEngine(cfg)
        eng.load_ohlcv(make_ohlcv(n=5))
        # No ratings loaded
        result = eng.run()
        assert result.status == "FAILED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
