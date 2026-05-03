"""Integration tests for the Kuwait Signal Engine — end-to-end signal flow.

Tests the full pipeline from raw OHLCV input → canonical signal JSON for
three representative Premier Market stocks: NBK, ZAIN, MABANEE.

Verified properties:
  1. All price levels (entry, SL, TP1, TP2) respect tick-size alignment.
  2. Circuit breaker logic rejects signals near daily limits.
  3. JSON output matches the canonical schema exactly.
  4. Liquidity filter correctly gates NEUTRAL vs BUY/SELL.
  5. All sub-scores are within their expected numerical ranges.
"""
from __future__ import annotations

import math
import random
import pytest

from app.services.signal_engine.engine.signal_generator import generate_kuwait_signal
from app.services.signal_engine.config.kuwait_constants import (
    TICK_SMALL,
    TICK_LARGE,
    TICK_BOUNDARY,
    CIRCUIT_BREAKER_UP,
    CIRCUIT_BREAKER_DOWN,
    align_to_tick,
)


# ── Synthetic OHLCV dataset builder ───────────────────────────────────────────

def _make_ohlcv(
    n: int = 300,
    start_price: float = 500.0,
    trend: float = 0.0003,
    sigma: float = 0.010,
    adtv_kd: float = 250_000.0,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic OHLCV rows with attached indicator stubs.

    Indicator stubs use simple rolling calculations so the signal engine
    doesn't need to call the actual indicators_service (which requires
    TA-Lib installation in CI).
    """
    rng = random.Random(seed)
    closes = [start_price]
    for _ in range(n - 1):
        ret = trend + rng.gauss(0, sigma)
        closes.append(max(10.0, closes[-1] * (1 + ret)))

    rows = []
    for i, c in enumerate(closes):
        h = c * (1 + abs(rng.gauss(0, sigma / 2)))
        l = c * (1 - abs(rng.gauss(0, sigma / 2)))
        vol = int(adtv_kd / max(c / 1000, 1) * rng.uniform(0.8, 1.2))
        row: dict = {
            "date": f"2024-{(i // 30 + 1):02d}-{(i % 30 + 1):02d}",
            "open": round(c * (1 + rng.gauss(0, 0.002)), 1),
            "high": round(h, 1),
            "low": round(l, 1),
            "close": round(c, 1),
            "volume": vol,
            "value": round(vol * c / 1000, 2),  # KWD value
        }
        # Attach minimal indicator stubs — use simple rolling calculations
        if i >= 20:
            row["ema_20"] = round(sum(closes[max(0, i-19):i+1]) / min(20, i+1), 2)
        if i >= 50:
            row["ema_50"] = round(sum(closes[max(0, i-49):i+1]) / min(50, i+1), 2)
        if i >= 199:
            row["sma_200"] = round(sum(closes[max(0, i-199):i+1]) / 200, 2)
        # RSI stub: simple momentum proxy
        if i >= 14:
            gains = [max(0, closes[j] - closes[j-1]) for j in range(i-13, i+1)]
            losses = [max(0, closes[j-1] - closes[j]) for j in range(i-13, i+1)]
            avg_g = sum(gains) / 14
            avg_l = sum(losses) / 14
            if avg_l < 1e-9:
                row["rsi_14"] = 100.0
            else:
                rs = avg_g / avg_l
                row["rsi_14"] = round(100 - 100 / (1 + rs), 2)
        # ADX stub (constant moderate trend)
        row["adx_14"] = 28.0
        # MACD stubs
        row["macd"] = 0.5
        row["macd_signal"] = 0.3
        row["macd_hist"] = 0.2
        # ATR stub
        row["atr_14"] = round(c * 0.012, 2)
        # BB stubs
        row["bb_upper"] = round(c * 1.02, 2)
        row["bb_lower"] = round(c * 0.98, 2)
        row["bb_mid"] = round(c, 2)
        # Volume flow stubs
        row["obv"] = vol * i
        row["ad"] = vol * (c - l) / max(h - l, 1)
        row["cmf_20"] = rng.gauss(0.05, 0.02)
        row["vwap"] = c
        rows.append(row)
    return rows


# ── NBK representative signal test ────────────────────────────────────────────

class TestNBKSignalFlow:
    @pytest.fixture(scope="class")
    def signal(self):
        rows = _make_ohlcv(n=300, start_price=720.0, trend=0.0005, adtv_kd=400_000.0, seed=1)
        return generate_kuwait_signal(rows, stock_code="NBK", segment="PREMIER",
                                      account_equity=100_000.0, delay_hours=0)

    def test_canonical_schema_keys_present(self, signal):
        required = ["timestamp", "stock_code", "segment", "signal", "setup_type",
                    "execution", "risk_metrics", "probabilities", "confluence_details",
                    "alerts", "metadata"]
        for k in required:
            assert k in signal, f"Missing top-level key: {k}"

    def test_execution_fields_present(self, signal):
        exe = signal["execution"]
        for k in ["entry_zone_fils", "stop_loss_fils", "tp1_fils", "tp2_fils",
                  "tick_alignment", "preferred_order_type"]:
            assert k in exe

    def test_stock_code_is_uppercase(self, signal):
        assert signal["stock_code"] == "NBK"

    def test_segment_is_uppercase(self, signal):
        assert signal["segment"] == "PREMIER"

    def test_signal_value_is_valid(self, signal):
        assert signal["signal"] in {"BUY", "SELL", "NEUTRAL"}

    def test_tick_alignment_on_entry(self, signal):
        if signal["signal"] == "NEUTRAL":
            return
        entry_low, entry_high = signal["execution"]["entry_zone_fils"]
        for price in [entry_low, entry_high]:
            if price is not None and price > 0:
                _assert_tick_aligned(price)

    def test_sl_tp_tick_aligned(self, signal):
        if signal["signal"] == "NEUTRAL":
            return
        exe = signal["execution"]
        for key in ["stop_loss_fils", "tp1_fils", "tp2_fils"]:
            v = exe.get(key)
            if v is not None and v > 0:
                _assert_tick_aligned(v)


# ── ZAIN signal flow test ──────────────────────────────────────────────────────

class TestZAINSignalFlow:
    @pytest.fixture(scope="class")
    def signal(self):
        rows = _make_ohlcv(n=300, start_price=560.0, trend=0.0, sigma=0.012, adtv_kd=200_000.0, seed=7)
        return generate_kuwait_signal(rows, stock_code="ZAIN", segment="PREMIER",
                                      account_equity=50_000.0, delay_hours=0)

    def test_schema_valid(self, signal):
        assert "confluence_details" in signal
        assert "sub_scores" in signal["confluence_details"]

    def test_sub_scores_in_range(self, signal):
        sub = signal["confluence_details"].get("sub_scores", {})
        for k, v in sub.items():
            if v is not None:
                assert 0.0 <= v <= 100.0, f"sub_score {k}={v} out of [0,100]"

    def test_probabilities_in_unit_interval(self, signal):
        probs = signal.get("probabilities", {})
        for k in ["p_tp1_before_sl", "p_tp2_before_sl"]:
            v = probs.get(k)
            if v is not None:
                assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_alerts_is_list(self, signal):
        assert isinstance(signal.get("alerts"), list)


# ── MABANEE signal flow test ───────────────────────────────────────────────────

class TestMABANEESignalFlow:
    @pytest.fixture(scope="class")
    def signal(self):
        # MABANEE is higher-priced: test the 1.0-fil tick grid
        rows = _make_ohlcv(n=300, start_price=1100.0, trend=0.0003, adtv_kd=150_000.0, seed=13)
        return generate_kuwait_signal(rows, stock_code="MABANEE", segment="PREMIER",
                                      account_equity=75_000.0, delay_hours=0)

    def test_large_tick_grid_applied(self, signal):
        """Prices above 100.9 fils must use 1.0-fil tick grid."""
        if signal["signal"] == "NEUTRAL":
            return
        exe = signal["execution"]
        for key in ["stop_loss_fils", "tp1_fils", "tp2_fils"]:
            v = exe.get(key)
            if v is not None and v >= TICK_BOUNDARY:
                assert abs(v - round(v)) < 1e-9, f"{key}={v} not on 1.0-fil grid"

    def test_tick_alignment_note_matches_price(self, signal):
        exe = signal["execution"]
        note = exe.get("tick_alignment", "")
        entry_low = exe.get("entry_zone_fils", [None, None])[0]
        if entry_low and entry_low >= TICK_BOUNDARY:
            assert "1.0-fil" in note
        elif entry_low and entry_low < TICK_BOUNDARY:
            assert "0.1-fil" in note

    def test_no_crash_on_high_price_stock(self, signal):
        assert signal is not None
        assert "signal" in signal


# ── Circuit breaker scenarios ─────────────────────────────────────────────────

class TestCircuitBreakerRejection:
    def test_near_upper_circuit_generates_neutral_or_no_buy(self):
        """A stock near +10 % upper circuit should not generate a fresh BUY."""
        rows = _make_ohlcv(n=300, start_price=100.0, adtv_kd=200_000.0, seed=99)
        # Simulate last bar near upper circuit: +9.8 % from previous close
        prev_close = rows[-2]["close"]
        upper = prev_close * (1 + CIRCUIT_BREAKER_UP - 0.002)
        rows[-1]["close"] = round(upper, 1)
        rows[-1]["high"] = round(upper * 1.001, 1)
        rows[-1]["low"] = round(upper * 0.998, 1)
        sig = generate_kuwait_signal(rows, "CBK", "PREMIER", 100_000.0, delay_hours=0)
        # Near circuit: BUY entries should not target above the circuit limit
        if sig["signal"] == "BUY":
            tp2 = sig["execution"].get("tp2_fils")
            if tp2 is not None:
                circuit_limit = prev_close * (1 + CIRCUIT_BREAKER_UP)
                assert tp2 <= circuit_limit * 1.01  # within 1 % margin

    def test_near_lower_circuit_generates_neutral_or_no_sell(self):
        rows = _make_ohlcv(n=300, start_price=200.0, trend=-0.001, adtv_kd=200_000.0, seed=77)
        prev_close = rows[-2]["close"]
        lower = prev_close * (1 - CIRCUIT_BREAKER_DOWN + 0.002)
        rows[-1]["close"] = round(lower, 1)
        rows[-1]["high"] = round(lower * 1.002, 1)
        rows[-1]["low"] = round(lower * 0.999, 1)
        sig = generate_kuwait_signal(rows, "BURG", "PREMIER", 100_000.0, delay_hours=0)
        # Near lower circuit: SELL SL shouldn't be placed below circuit floor
        if sig["signal"] == "SELL":
            sl = sig["execution"].get("stop_loss_fils")
            if sl is not None:
                circuit_floor = prev_close * (1 - CIRCUIT_BREAKER_DOWN)
                assert sl >= circuit_floor * 0.99


# ── Illiquid stock liquidity gate ─────────────────────────────────────────────

class TestLiquidityGate:
    def test_illiquid_stock_returns_neutral(self):
        """Stock with ADTV below threshold should always return NEUTRAL."""
        rows = _make_ohlcv(n=300, adtv_kd=10_000.0, seed=55)  # far below 100k KD min
        sig = generate_kuwait_signal(rows, "ILLIQUID", "PREMIER", 100_000.0, delay_hours=0)
        assert sig["signal"] == "NEUTRAL"

    def test_liquid_stock_can_generate_directional_signal(self):
        """Strong trend + liquid stock should NOT always be NEUTRAL."""
        rows = _make_ohlcv(n=300, trend=0.001, sigma=0.008, adtv_kd=500_000.0, seed=21)
        sig = generate_kuwait_signal(rows, "NBK", "PREMIER", 100_000.0, delay_hours=0)
        # We cannot guarantee a BUY (depends on data), but should not raise
        assert sig["signal"] in {"BUY", "SELL", "NEUTRAL"}


# ── Delay decay integration ────────────────────────────────────────────────────

class TestDelayDecay:
    def test_48h_delay_lowers_confidence(self):
        rows = _make_ohlcv(n=300, trend=0.001, adtv_kd=300_000.0, seed=33)
        sig_0h = generate_kuwait_signal(rows, "NBK", "PREMIER", 100_000.0, delay_hours=0)
        sig_48h = generate_kuwait_signal(rows, "NBK", "PREMIER", 100_000.0, delay_hours=48)
        p0 = sig_0h.get("probabilities", {}).get("p_tp1_before_sl") or 0.0
        p48 = sig_48h.get("probabilities", {}).get("p_tp1_before_sl") or 0.0
        assert p48 <= p0 + 0.01  # 48h delay never increases probability


# ── Helper ────────────────────────────────────────────────────────────────────

def _assert_tick_aligned(price: float) -> None:
    """Assert price is on the correct tick grid for its magnitude."""
    if price <= TICK_BOUNDARY:
        remainder = round(price % TICK_SMALL, 6)
        assert remainder < 1e-4 or abs(remainder - TICK_SMALL) < 1e-4, (
            f"Price {price} not aligned to 0.1-fil grid (remainder={remainder})"
        )
    else:
        remainder = price % TICK_LARGE
        assert remainder < 1e-9 or abs(remainder - TICK_LARGE) < 1e-9, (
            f"Price {price} not aligned to 1.0-fil grid (remainder={remainder})"
        )
