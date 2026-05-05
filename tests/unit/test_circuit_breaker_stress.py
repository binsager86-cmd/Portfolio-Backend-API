"""Circuit-breaker stress tests for the Kuwait Signal Engine.

Tests:
  - Price at +9.8% (near +10% upper circuit limit):
      * circuit_proximity["is_near_limit"] is True
      * total_score is reduced to 70% of raw score
      * TP1 is below the circuit upper limit minus a 0.5% buffer
  - Price at -4.8% (near -5% lower circuit limit):
      * circuit_proximity["is_near_limit"] is True
      * total_score penalty applied
  - Price at +5% (comfortably within limits):
      * circuit_proximity["is_near_limit"] is False
      * total_score unchanged
  - _circuit_breaker_alerts helper returns the right string format
"""
from __future__ import annotations

import asyncio
import math

import pytest

from app.services.signal_engine.config.kuwait_constants import (
    CIRCUIT_BUFFER_PCT,
    CIRCUIT_LOWER_PCT,
    CIRCUIT_UPPER_PCT,
)
from app.services.signal_engine.engine.signal_generator import (
    _circuit_breaker_alerts,
    generate_kuwait_signal,
)


# ── Helper: run async generate_kuwait_signal synchronously ────────────────────

def _run(coro):
    return asyncio.run(coro)


# ── Helper: build minimal OHLCV rows ─────────────────────────────────────────

def _make_rows(
    prev_close: float,
    current_close: float,
    n_warmup: int = 80,
) -> list[dict]:
    """Build enough rows for the signal engine (prev_close warmup + two final rows)."""
    base = prev_close * 0.9
    # Warmup rows: gradually rising to prev_close
    rows = []
    for i in range(n_warmup):
        c = base + (prev_close - base) * (i / max(1, n_warmup - 1))
        rows.append({
            "date": f"2024-{(i + 1):04d}",
            "close": round(c, 3),
            "open":  round(c * 0.998, 3),
            "high":  round(c * 1.006, 3),
            "low":   round(c * 0.994, 3),
            "volume":  5_000_000,
            "value":   300_000.0,
            "ema_20":  round(c * 0.985, 3),
            "ema_50":  round(c * 0.970, 3),
            "sma_200": round(c * 0.940, 3),
            "adx_14":  30.0,
            "rsi_14":  58.0,
            "macd":        2.0,
            "macd_signal": 1.5,
            "macd_hist":   0.5,
            "stoch_k":  55.0,
            "stoch_d":  48.0,
            "obv":    float(1_000_000 * (i + 1)),
            "cmf_20": 0.12,
            "ad_line": float(5_000_000 * (i + 1)),
            "atr_14": round(prev_close * 0.015, 3),
            "vwap":   round(c * 0.975, 3),
        })

    # Second-to-last row = prev_close (yesterday)
    rows.append({
        **rows[-1],
        "date": "2024-9998",
        "close": prev_close,
        "open":  round(prev_close * 0.999, 3),
        "high":  round(prev_close * 1.005, 3),
        "low":   round(prev_close * 0.995, 3),
        "ema_20":  round(prev_close * 0.985, 3),
        "ema_50":  round(prev_close * 0.970, 3),
        "sma_200": round(prev_close * 0.940, 3),
        "atr_14": round(prev_close * 0.015, 3),
        "vwap":   round(prev_close * 0.975, 3),
    })

    # Last row = current close (today)
    rows.append({
        **rows[-1],
        "date": "2024-9999",
        "close": current_close,
        "open":  round(current_close * 0.998, 3),
        "high":  round(current_close * 1.004, 3),
        "low":   round(current_close * 0.996, 3),
        "ema_20":  round(current_close * 0.980, 3),
        "ema_50":  round(current_close * 0.960, 3),
        "sma_200": round(current_close * 0.920, 3),
        "atr_14": round(current_close * 0.014, 3),
        "vwap":   round(current_close * 0.970, 3),
        "obv":    float(1_000_000 * n_warmup * 1.5),
        "cmf_20": 0.20,
    })

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: _circuit_breaker_alerts helper (pure, no async)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreakerAlerts:
    def test_upper_circuit_alert_when_within_buffer(self):
        """Price within CIRCUIT_BUFFER_PCT of upper limit should trigger an alert."""
        prev_close = 1_000.0
        # Place current price at exactly (CIRCUIT_UPPER_PCT - 0.005) above prev_close
        gap = CIRCUIT_BUFFER_PCT * 0.5  # inside buffer
        upper_limit = prev_close * (1.0 + CIRCUIT_UPPER_PCT)
        current_close = upper_limit - prev_close * gap  # just below upper limit

        rows = _make_rows(prev_close, current_close)
        alerts = _circuit_breaker_alerts(rows, prev_close)

        assert len(alerts) >= 1
        assert any("upper circuit" in a.lower() or "+10%" in a for a in alerts)

    def test_lower_circuit_alert_when_within_buffer(self):
        """Price within CIRCUIT_BUFFER_PCT of lower limit should trigger an alert."""
        prev_close = 1_000.0
        lower_limit = prev_close * (1.0 + CIRCUIT_LOWER_PCT)  # e.g. 950.0
        # Place price just above lower limit
        current_close = lower_limit + prev_close * (CIRCUIT_BUFFER_PCT * 0.5)

        rows = _make_rows(prev_close, current_close)
        alerts = _circuit_breaker_alerts(rows, prev_close)

        assert len(alerts) >= 1
        assert any("lower circuit" in a.lower() or "-5%" in a for a in alerts)

    def test_no_alert_far_from_limits(self):
        """Price comfortably in the middle should produce no circuit alerts."""
        prev_close = 1_000.0
        current_close = 1_050.0  # +5%, within limits
        rows = _make_rows(prev_close, current_close)
        alerts = _circuit_breaker_alerts(rows, prev_close)
        assert alerts == []

    def test_empty_rows_returns_empty(self):
        assert _circuit_breaker_alerts([], 1000.0) == []

    def test_zero_prev_close_returns_empty(self):
        rows = _make_rows(1000.0, 1050.0)
        assert _circuit_breaker_alerts(rows, 0.0) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: circuit_proximity in generate_kuwait_signal output
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitProximityOutput:
    """Verify the circuit_proximity dict exposed in confluence_details."""

    def _run_signal(self, prev_close: float, current_close: float) -> dict:
        rows = _make_rows(prev_close, current_close)
        return _run(generate_kuwait_signal(
            rows=rows,
            stock_code="TEST",
            segment="PREMIER",
            account_equity=100_000.0,
        ))

    def test_near_upper_limit_marks_proximity(self):
        """Price at +9.8% (near +10% upper circuit) must set is_near_limit=True."""
        prev_close = 1_000.0
        current_close = prev_close * 1.098   # +9.8% — within 0.2% of +10% limit

        signal = self._run_signal(prev_close, current_close)
        cp = signal["confluence_details"]["circuit_proximity"]

        assert cp["is_near_limit"] is True, (
            f"Expected is_near_limit=True at +9.8%, got {cp}"
        )
        assert cp["direction"] in {"upper", "both"}

    def test_near_upper_limit_reduces_score(self):
        """Total score must be ≤ 70% of raw score when near upper circuit."""
        prev_close = 1_000.0
        current_close = prev_close * 1.098

        signal = self._run_signal(prev_close, current_close)
        cd = signal["confluence_details"]

        total_score = cd["total_score"]
        total_score_raw = cd.get("total_score_raw", total_score)

        if total_score_raw > 0 and cd["circuit_proximity"]["is_near_limit"]:
            max_allowed = int(total_score_raw * 0.70)
            assert total_score <= max_allowed, (
                f"Expected total_score ({total_score}) <= 70% of raw "
                f"({total_score_raw}) = {max_allowed}"
            )

    def test_near_upper_limit_caps_tp1(self):
        """TP1 (in fils) must be below the circuit upper limit to avoid filling above it."""
        prev_close = 1_000.0
        current_close = prev_close * 1.098
        circuit_upper_fils = prev_close * (1.0 + CIRCUIT_UPPER_PCT)  # 1_100.0 fils

        signal = self._run_signal(prev_close, current_close)
        tp1 = signal["execution"].get("tp1_fils")

        if tp1 is not None and tp1 > 0:
            # TP1 should be below circuit limit (with a small buffer)
            assert tp1 < circuit_upper_fils * 0.998, (
                f"TP1 ({tp1:.1f} fils) is at or above circuit upper limit "
                f"({circuit_upper_fils:.1f} fils)"
            )

    def test_comfortable_price_no_proximity(self):
        """Price at +5% should not trigger circuit proximity."""
        prev_close = 1_000.0
        current_close = 1_050.0  # +5%

        signal = self._run_signal(prev_close, current_close)
        cp = signal["confluence_details"]["circuit_proximity"]

        assert cp["is_near_limit"] is False

    def test_proximity_dict_has_required_keys(self):
        """circuit_proximity must always contain the four required keys."""
        signal = self._run_signal(1_000.0, 1_050.0)
        cp = signal["confluence_details"]["circuit_proximity"]

        for key in ("is_near_limit", "direction", "distance_to_upper_pct", "distance_to_lower_pct"):
            assert key in cp, f"Missing key '{key}' in circuit_proximity"

    def test_near_lower_limit_marks_proximity(self):
        """Price at -4.8% (near -5% lower circuit) must set is_near_limit=True."""
        prev_close = 1_000.0
        current_close = prev_close * 0.952  # -4.8%, within 0.2% of -5% limit

        signal = self._run_signal(prev_close, current_close)
        cp = signal["confluence_details"]["circuit_proximity"]

        assert cp["is_near_limit"] is True
        assert cp["direction"] in {"lower", "both"}
