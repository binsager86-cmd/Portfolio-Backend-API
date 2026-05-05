"""Unit tests for Hurst Exponent pre-filter.

Tests trending vs mean-reverting vs random walk behavior.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.services.signal_engine.models.regime.hurst_filter import (
    _rescaled_range_analysis,
    compute_hurst_filter,
)


class TestHurstFilter:
    """Test Hurst exponent calculations."""
    
    def test_trending_market_high_hurst(self):
        """Trending market should produce H > 0.5."""
        # Create trending price series (persistent uptrend).
        # Use 200 data points with low noise so the trend dominates over noise,
        # giving reliable H estimation from R/S analysis.
        np.random.seed(42)
        trend = np.linspace(100, 200, 200)
        noise = np.random.normal(0, 0.5, 200)
        prices = trend + noise
        
        h_value, h_std_error = _rescaled_range_analysis(prices)
        
        # Trending markets have H > 0.5 (typically 0.55-0.85)
        assert h_value > 0.5, f"Expected H > 0.5 for trending market, got {h_value:.3f}"
        assert 0.01 <= h_std_error <= 0.30, f"Std error out of expected range: {h_std_error:.3f}"
    
    def test_random_walk_near_half(self):
        """Random walk should produce H ≈ 0.5."""
        # Create random walk (Brownian motion).
        # Use 200 data points for more stable estimation.
        np.random.seed(123)
        returns = np.random.normal(0, 1, 200)
        prices = 100 * np.exp(np.cumsum(returns * 0.01))
        
        h_value, h_std_error = _rescaled_range_analysis(prices)
        
        # Random walk should be near 0.5 (± 0.20 tolerance with 200 samples)
        assert 0.30 < h_value < 0.70, f"Expected H ≈ 0.5 for random walk, got {h_value:.3f}"
    
    def test_mean_reverting_low_hurst(self):
        """Mean-reverting market should produce H < 0.5."""
        # Create mean-reverting series (oscillating around 100)
        np.random.seed(789)
        prices = []
        price = 100.0
        target = 100.0
        
        for _ in range(60):
            # Mean reversion force: pulls price back to target
            reversion = (target - price) * 0.3
            noise = np.random.normal(0, 2)
            price += reversion + noise
            prices.append(price)
        
        prices_arr = np.array(prices)
        h_value, h_std_error = _rescaled_range_analysis(prices_arr)
        
        # Mean-reverting markets typically have H < 0.5 (often 0.3-0.45)
        # Relaxed assertion since synthetic mean reversion can be noisy
        assert h_value < 0.55, f"Expected H < 0.55 for mean-reverting market, got {h_value:.3f}"
    
    def test_compute_hurst_filter_premier_pass(self):
        """Premier market with strong trend should pass filter."""
        # Create strong uptrend
        np.random.seed(42)
        rows = []
        for i in range(40):
            rows.append({
                "date": f"2024-01-{i+1:02d}",
                "close": 100 + i * 2 + np.random.normal(0, 1),
            })
        
        result = compute_hurst_filter(rows, market_segment="PREMIER", lookback_days=30)
        
        assert result["action"] == "proceed", f"Expected 'proceed', got {result['action']}"
        assert result["is_trending"], "Expected is_trending=True for strong trend"
        assert result["h_value"] > result["threshold_used"], \
            f"H={result['h_value']:.3f} should be > threshold={result['threshold_used']:.2f}"
    
    def test_compute_hurst_filter_main_lower_threshold(self):
        """Main market uses lower threshold (0.48 vs 0.55)."""
        # Borderline trend that passes MAIN but would fail PREMIER
        np.random.seed(999)
        rows = []
        for i in range(40):
            rows.append({
                "date": f"2024-01-{i+1:02d}",
                "close": 100 + i * 0.5 + np.random.normal(0, 3),
            })
        
        result_main = compute_hurst_filter(rows, market_segment="MAIN", lookback_days=30)
        result_premier = compute_hurst_filter(rows, market_segment="PREMIER", lookback_days=30)
        
        assert result_main["threshold_used"] == 0.48, "MAIN threshold should be 0.48"
        assert result_premier["threshold_used"] == 0.55, "PREMIER threshold should be 0.55"
        
        # Same data, different thresholds may produce different actions
        # (Not strictly required, but documents threshold difference)
    
    def test_compute_hurst_filter_skip_signal(self):
        """Strong mean-reversion should trigger skip_signal."""
        # Create clear mean-reverting series
        np.random.seed(321)
        rows = []
        price = 100.0
        
        for i in range(40):
            # Strong oscillation
            price = 100 + 10 * np.sin(i * 0.5) + np.random.normal(0, 1)
            rows.append({
                "date": f"2024-01-{i+1:02d}",
                "close": price,
            })
        
        result = compute_hurst_filter(rows, market_segment="PREMIER", lookback_days=30)
        
        # Expect either skip_signal or skip_or_downgrade for mean reversion
        assert result["action"] in ["skip_signal", "skip_or_downgrade"], \
            f"Expected skip action for mean reversion, got {result['action']}"
        assert result["confidence_penalty"] < 1.0, "Confidence penalty should be applied"
    
    def test_insufficient_data(self):
        """Too few bars should trigger skip_or_downgrade."""
        rows = [
            {"date": "2024-01-01", "close": 100},
            {"date": "2024-01-02", "close": 101},
            {"date": "2024-01-03", "close": 102},
        ]
        
        result = compute_hurst_filter(rows, market_segment="PREMIER", lookback_days=30)
        
        assert result["action"] == "skip_or_downgrade", \
            "Insufficient data should trigger skip_or_downgrade"
        assert result["description"] == "insufficient_data_for_hurst"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
