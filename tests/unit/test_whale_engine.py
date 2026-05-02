"""Tests for the whale-flow proxy engine.

Covers the 8 mandatory test suites:
1. Edge cases  2. Boundaries  3. Liquidity gate  4. Action logic
5. Multi-horizon  6. Persistence  7. JSON safety  8. Scenario tests A/B/C
"""

from __future__ import annotations

import copy
import json
import math
from typing import Dict, List

import pytest

from app.services.whale_flow_engine import (
    calculate_factors,
    calculate_persistence_bonus,
    calculate_scores,
    check_liquidity_gate,
    determine_action,
    determine_alignment,
    determine_bias,
    estimate_whale_flow_proxy,
    normalize_factor,
    percentile_rank,
    run_whale_engine,
    safe_divide,
    select_history,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _ramp(low: float, high: float, n: int = 90) -> List[float]:
    """Simple linear ramp – useful as deterministic history."""
    if n == 1:
        return [low]
    step = (high - low) / (n - 1)
    return [round(low + step * i, 6) for i in range(n)]


def _base_input() -> Dict:
    """A neutral-ish baseline input. Tests mutate copies of this."""
    return {
        "ticker": "TEST.KW",
        "timeframe": "daily",
        "as_of_date": "2026-05-02",
        "open_price": 100.0,
        "high_price": 102.0,
        "low_price": 99.0,
        "close_price": 101.0,
        "current_price": 101.0,
        "volume_today": 1_000_000.0,
        "volume_20d_avg": 1_000_000.0,
        "total_traded_value": 5_000_000.0,
        "minimum_total_traded_value": 1_000_000.0,
        "net_liquidity_3d_avg": 0.0,
        "atr_20d": 3.0,
        "price_range_today": 3.0,
        "anchored_vwap": {
            "value": 100.5,
            "slope_5d": 0.0,
            "time_above_vwap_ratio_10d": 0.5,
        },
        "ad_line_slope_5d": 0.0,
        "cmf_10d": 0.0,
        "market_context": {"market_bias": "neutral"},
        "history_90d": {
            "net_liquidity": _ramp(-1.0, 1.0),
            "rel_volume": _ramp(0.5, 2.0),
            "vwap_position": _ramp(-0.05, 0.05),
            "vwap_slope": _ramp(-0.5, 0.5),
            "time_above_vwap": _ramp(0.0, 1.0),
            "ad_slope": _ramp(-1.0, 1.0),
            "range_compression": _ramp(0.1, 5.0),
            "range_expansion": _ramp(0.1, 3.0),
            "downside_pressure": _ramp(0.0, 1.0),
        },
        "history_252d": None,
        "normalization_horizon": "90d",
        "long_term_regime": "unknown",
        "data_quality": "direct",
        "higher_timeframe_bias": "neutral",
    }


# ---------------------------------------------------------------------------
# 1. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_percentile_rank_empty_history(self):
        assert percentile_rank(1.0, []) == 0.5

    def test_percentile_rank_single_value_history(self):
        assert percentile_rank(1.0, [42.0]) == 0.5

    def test_percentile_rank_constant_history(self):
        assert percentile_rank(1.0, [5.0, 5.0, 5.0, 5.0]) == 0.5

    def test_safe_divide_zero_denominator(self):
        assert safe_divide(10.0, 0.0) == 0.0
        assert safe_divide(10.0, 0.0, default=42.0) == 42.0
        assert safe_divide(10.0, 1e-15, default=7.0) == 7.0

    def test_factors_with_zero_atr_and_volume(self):
        inp = _base_input()
        inp["atr_20d"] = 0.0
        inp["price_range_today"] = 0.0
        inp["volume_today"] = 0.0
        inp["volume_20d_avg"] = 0.0
        # Should not crash and all factors must be in [0.05, 0.95].
        out = calculate_factors(inp)
        for k, v in out.items():
            assert 0.05 <= v <= 0.95, f"{k}={v} outside clip range"


# ---------------------------------------------------------------------------
# 2. Boundaries
# ---------------------------------------------------------------------------

class TestBoundaries:
    def test_scores_within_range_and_integer(self):
        inp = _base_input()
        # Push everything as bullish as possible.
        inp["net_liquidity_3d_avg"] = 1e9
        inp["volume_today"] = 1e9
        inp["current_price"] = 200.0
        inp["anchored_vwap"]["value"] = 50.0
        inp["anchored_vwap"]["slope_5d"] = 1e9
        inp["anchored_vwap"]["time_above_vwap_ratio_10d"] = 1.0
        inp["ad_line_slope_5d"] = 1e9
        inp["cmf_10d"] = 1.0
        out = run_whale_engine(inp)
        assert isinstance(out["accumulation_score"], int)
        assert isinstance(out["distribution_score"], int)
        assert 0 <= out["accumulation_score"] <= 100
        assert 0 <= out["distribution_score"] <= 100

    def test_scores_lower_bound(self):
        inp = _base_input()
        inp["net_liquidity_3d_avg"] = -1e9
        inp["volume_today"] = 0.0
        inp["current_price"] = 1.0
        inp["anchored_vwap"]["value"] = 1000.0
        inp["anchored_vwap"]["slope_5d"] = -1e9
        inp["anchored_vwap"]["time_above_vwap_ratio_10d"] = 0.0
        inp["ad_line_slope_5d"] = -1e9
        inp["cmf_10d"] = -1.0
        out = run_whale_engine(inp)
        assert 0 <= out["accumulation_score"] <= 100
        assert 0 <= out["distribution_score"] <= 100


# ---------------------------------------------------------------------------
# 3. Liquidity gate
# ---------------------------------------------------------------------------

class TestLiquidityGate:
    def test_check_liquidity_gate(self):
        assert check_liquidity_gate(100.0, 50.0) is True
        assert check_liquidity_gate(50.0, 50.0) is True
        assert check_liquidity_gate(49.0, 50.0) is False

    def test_failed_gate_forces_wait(self):
        inp = _base_input()
        # Build a clean BUY setup but starve liquidity.
        inp["net_liquidity_3d_avg"] = 1e9
        inp["current_price"] = 200.0
        inp["anchored_vwap"]["value"] = 100.0
        inp["anchored_vwap"]["slope_5d"] = 1e9
        inp["anchored_vwap"]["time_above_vwap_ratio_10d"] = 1.0
        inp["ad_line_slope_5d"] = 1e9
        inp["cmf_10d"] = 1.0
        inp["higher_timeframe_bias"] = "accumulation"
        inp["total_traded_value"] = 10.0
        inp["minimum_total_traded_value"] = 1_000_000.0
        out = run_whale_engine(inp)
        assert out["liquidity_gate_passed"] is False
        assert out["action"] == "WAIT"
        assert out["alert"]["liquidity_status"] == "FAILED"


# ---------------------------------------------------------------------------
# 4. Action logic
# ---------------------------------------------------------------------------

class TestActionLogic:
    def _factors(self, V: float = 0.5) -> Dict[str, float]:
        return {"N": 0.8, "V": V, "W": 0.8, "A": 0.8, "C": 0.8, "R": 0.8, "D": 0.5}

    def test_buy_when_all_conditions_met(self):
        assert (
            determine_action(
                bias="accumulation",
                accum_score=80,
                dist_score=20,
                price=101.0,
                vwap_value=100.0,
                ad_slope=0.5,
                alignment="aligned",
                liquidity_passed=True,
                factors=self._factors(),
            )
            == "BUY"
        )

    def test_buy_blocked_when_below_vwap(self):
        assert (
            determine_action(
                "accumulation", 80, 20, 99.0, 100.0, 0.5, "aligned", True, self._factors()
            )
            == "WAIT"
        )

    def test_buy_blocked_when_score_too_low(self):
        assert (
            determine_action(
                "accumulation", 65, 20, 101.0, 100.0, 0.5, "aligned", True, self._factors()
            )
            == "WAIT"
        )

    def test_buy_blocked_when_conflicting_alignment(self):
        assert (
            determine_action(
                "accumulation", 80, 20, 101.0, 100.0, 0.5, "conflicting", True, self._factors()
            )
            == "WAIT"
        )

    def test_sell_when_all_conditions_met(self):
        assert (
            determine_action(
                "distribution", 20, 80, 99.0, 100.0, -0.5, "aligned", True, self._factors(V=0.5)
            )
            == "SELL"
        )

    def test_sell_blocked_when_volume_exhausted(self):
        # V > 0.92 → capitulation guard.
        assert (
            determine_action(
                "distribution", 20, 80, 99.0, 100.0, -0.5, "aligned", True, self._factors(V=0.95)
            )
            == "WAIT"
        )

    def test_wait_when_liquidity_failed(self):
        assert (
            determine_action(
                "accumulation", 90, 20, 101.0, 100.0, 0.5, "aligned", False, self._factors()
            )
            == "WAIT"
        )


# ---------------------------------------------------------------------------
# 5. Multi-horizon
# ---------------------------------------------------------------------------

class TestMultiHorizon:
    def test_252d_used_when_present_and_long_enough(self):
        h90 = {"net_liquidity": [1.0, 2.0]}
        h252 = {"net_liquidity": [9.0] * 200}
        chosen = select_history(h90, h252, "252d", "unknown", "net_liquidity")
        assert chosen is h252["net_liquidity"]

    def test_falls_back_to_90d_when_252d_too_short(self):
        h90 = {"net_liquidity": [1.0, 2.0, 3.0]}
        h252 = {"net_liquidity": [9.0] * 50}
        chosen = select_history(h90, h252, "252d", "unknown", "net_liquidity")
        assert chosen == [1.0, 2.0, 3.0]

    def test_auto_uses_252d_only_for_qualifying_regime(self):
        h90 = {"net_liquidity": [1.0]}
        h252 = {"net_liquidity": [9.0] * 200}
        assert select_history(h90, h252, "auto", "trending_up", "net_liquidity") == [1.0]
        assert select_history(h90, h252, "auto", "base_building", "net_liquidity") is h252["net_liquidity"]
        assert select_history(h90, h252, "auto", "choppy", "net_liquidity") is h252["net_liquidity"]

    def test_missing_h252_safe(self):
        h90 = {"net_liquidity": [1.0, 2.0]}
        assert select_history(h90, None, "252d", "base_building", "net_liquidity") == [1.0, 2.0]


# ---------------------------------------------------------------------------
# 6. Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_ten_consecutive_above_threshold_yields_bonus(self):
        hist = [0.1] * 20 + [0.7] * 10
        bonus = calculate_persistence_bonus(hist)
        # consec=10 → (10-8+1)*0.01 = 0.03
        assert bonus == pytest.approx(0.03)

    def test_five_consecutive_no_bonus(self):
        hist = [0.1] * 25 + [0.7] * 5
        assert calculate_persistence_bonus(hist) == 0.0

    def test_bonus_capped_at_max(self):
        hist = [0.7] * 30
        assert calculate_persistence_bonus(hist) == pytest.approx(0.10)

    def test_short_history_no_bonus(self):
        assert calculate_persistence_bonus([0.9, 0.9]) == 0.0


# ---------------------------------------------------------------------------
# 7. JSON safety
# ---------------------------------------------------------------------------

def _walk(obj):
    """Yield every leaf value in a nested dict/list structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


class TestJSONSafety:
    def test_output_is_serializable(self):
        out = run_whale_engine(_base_input())
        # Must serialize without raising.
        s = json.dumps(out)
        assert isinstance(s, str) and len(s) > 0

    def test_no_tuples_nan_inf_or_none_leaves(self):
        out = run_whale_engine(_base_input())
        # No tuples anywhere.
        for v in _walk(out):
            assert not isinstance(v, tuple)
            if isinstance(v, float):
                assert not math.isnan(v)
                assert not math.isinf(v)
        # Required fields are not None.
        assert out["accumulation_score"] is not None
        assert out["distribution_score"] is not None
        assert out["bias"] is not None
        assert out["action"] is not None
        assert out["estimated_whale_flow_proxy_range"] is not None

    def test_flow_range_is_two_element_list(self):
        out = run_whale_engine(_base_input())
        flow = out["estimated_whale_flow_proxy_range"]
        assert isinstance(flow, list)
        assert len(flow) == 2
        assert flow[0] <= flow[1]

    def test_no_institutional_language(self):
        out = run_whale_engine(_base_input())
        s = json.dumps(out).lower()
        assert "institutional flow" not in s


# ---------------------------------------------------------------------------
# 8. Scenario tests
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_a_strong_silent_accumulation(self):
        inp = _base_input()
        # Quiet body, close near high, price above VWAP, A/D rising, persistent N.
        inp["open_price"] = 100.0
        inp["high_price"] = 100.5
        inp["low_price"] = 99.8
        inp["close_price"] = 100.45
        inp["current_price"] = 100.45
        inp["price_range_today"] = 0.7  # << ATR → high R
        inp["atr_20d"] = 3.0
        inp["volume_today"] = 1_200_000.0  # moderate, not exhaustion
        inp["volume_20d_avg"] = 1_000_000.0
        inp["net_liquidity_3d_avg"] = 0.95  # near top of ramp [-1, 1]
        inp["anchored_vwap"] = {
            "value": 99.5,
            "slope_5d": 0.45,
            "time_above_vwap_ratio_10d": 0.95,
        }
        inp["ad_line_slope_5d"] = 0.9
        inp["cmf_10d"] = 0.25
        inp["higher_timeframe_bias"] = "accumulation"
        inp["market_context"]["market_bias"] = "bullish"
        # Last 30 days of net_liquidity all above threshold to trigger persistence.
        inp["history_90d"]["net_liquidity"] = (
            inp["history_90d"]["net_liquidity"][:60] + [0.7] * 30
        )

        out = run_whale_engine(inp)
        assert out["bias"] == "accumulation"
        assert out["accumulation_score"] >= out["distribution_score"]
        assert out["persistence_bonus"] > 0.0
        # Phase should reflect either silent_accumulation or active.
        assert out["alert"]["phase"] in ("silent_accumulation", "active", "building")
        assert out["liquidity_gate_passed"] is True

    def test_b_strong_distribution(self):
        inp = _base_input()
        inp["open_price"] = 101.0
        inp["high_price"] = 101.5
        inp["low_price"] = 98.0
        inp["close_price"] = 98.2
        inp["current_price"] = 98.2
        inp["price_range_today"] = 3.5
        inp["atr_20d"] = 3.0
        inp["volume_today"] = 1_500_000.0  # elevated but not > 0.92 of ramp top (=2.0)
        inp["volume_20d_avg"] = 1_000_000.0
        inp["net_liquidity_3d_avg"] = -0.9
        inp["anchored_vwap"] = {
            "value": 100.5,
            "slope_5d": -0.4,
            "time_above_vwap_ratio_10d": 0.05,
        }
        inp["ad_line_slope_5d"] = -0.9
        inp["cmf_10d"] = -0.25
        inp["higher_timeframe_bias"] = "distribution"
        inp["market_context"]["market_bias"] = "bearish"

        out = run_whale_engine(inp)
        assert out["bias"] == "distribution"
        assert out["distribution_score"] >= out["accumulation_score"]
        assert out["liquidity_gate_passed"] is True
        # Action is SELL unless V > 0.92 exhaustion guard kicks in.
        assert out["action"] in ("SELL", "WAIT")

    def test_c_illiquid_forces_wait(self):
        inp = _base_input()
        # Strong-looking accumulation, but starved of liquidity.
        inp["net_liquidity_3d_avg"] = 0.95
        inp["current_price"] = 105.0
        inp["anchored_vwap"]["value"] = 100.0
        inp["anchored_vwap"]["slope_5d"] = 0.5
        inp["anchored_vwap"]["time_above_vwap_ratio_10d"] = 0.95
        inp["ad_line_slope_5d"] = 0.8
        inp["cmf_10d"] = 0.3
        inp["higher_timeframe_bias"] = "accumulation"
        inp["total_traded_value"] = 100.0
        inp["minimum_total_traded_value"] = 5_000_000.0

        out = run_whale_engine(inp)
        assert out["liquidity_gate_passed"] is False
        assert out["action"] == "WAIT"
        assert out["alert"]["liquidity_status"] == "FAILED"


# ---------------------------------------------------------------------------
# Misc internal sanity
# ---------------------------------------------------------------------------

def test_determine_bias_thresholds():
    assert determine_bias(80, 50) == "accumulation"
    assert determine_bias(50, 80) == "distribution"
    assert determine_bias(60, 55) == "neutral"
    assert determine_bias(60, 40) == "neutral"  # diff exactly 20 → neutral
    assert determine_bias(60, 39) == "accumulation"


def test_determine_alignment():
    assert determine_alignment("accumulation", "accumulation") == "aligned"
    assert determine_alignment("accumulation", "distribution") == "conflicting"
    assert determine_alignment("neutral", "accumulation") == "mixed"
    assert determine_alignment("accumulation", "neutral") == "mixed"


def test_estimate_flow_range_ordering():
    flow = estimate_whale_flow_proxy(
        total_traded_value=1_000_000.0,
        accum_score=85,
        dist_score=20,
        confidence=0.9,
        market_bias="bullish",
        active_bias="accumulation",
    )
    assert isinstance(flow, list) and len(flow) == 2
    assert flow[0] < flow[1]


def test_normalize_factor_alias_matches_percentile():
    h = [1.0, 2.0, 3.0, 4.0]
    assert normalize_factor(2.5, h) == percentile_rank(2.5, h)


def test_run_whale_engine_missing_field_raises():
    inp = _base_input()
    del inp["ticker"]
    with pytest.raises(ValueError, match="ticker"):
        run_whale_engine(inp)


def test_run_whale_engine_deterministic():
    inp = _base_input()
    a = run_whale_engine(copy.deepcopy(inp))
    b = run_whale_engine(copy.deepcopy(inp))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_calculate_scores_returns_int_tuple():
    factors = {"N": 0.5, "V": 0.5, "W": 0.5, "A": 0.5, "C": 0.5, "R": 0.5, "D": 0.5}
    acc, dist = calculate_scores(factors)
    assert isinstance(acc, int) and isinstance(dist, int)
    assert 0 <= acc <= 100 and 0 <= dist <= 100
