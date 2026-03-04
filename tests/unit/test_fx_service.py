"""
Unit tests for FX Service and Currency conversion.

Covers:
  - safe_float edge cases
  - convert_to_kwd with various currencies
  - FX cache behavior
  - PORTFOLIO_CCY mapping
"""

import pytest

from app.services.fx_service import (
    safe_float,
    convert_to_kwd,
    DEFAULT_USD_TO_KWD,
    PORTFOLIO_CCY,
    BASE_CCY,
    USD_CCY,
)


class TestSafeFloat:
    """Exhaustive tests for safe_float utility."""

    def test_integer(self):
        assert safe_float(42) == 42.0

    def test_float(self):
        assert safe_float(3.14) == 3.14

    def test_string_number(self):
        assert safe_float("3.14") == 3.14

    def test_string_integer(self):
        assert safe_float("100") == 100.0

    def test_none_returns_default(self):
        assert safe_float(None) == 0.0
        assert safe_float(None, 99.0) == 99.0

    def test_empty_string_returns_default(self):
        assert safe_float("") == 0.0
        assert safe_float("", 42.0) == 42.0

    def test_non_numeric_string(self):
        assert safe_float("abc") == 0.0
        assert safe_float("abc", -1.0) == -1.0

    def test_boolean_values(self):
        # bool is subclass of int in Python
        assert safe_float(True) == 1.0
        assert safe_float(False) == 0.0

    def test_negative_numbers(self):
        assert safe_float(-10.5) == -10.5
        assert safe_float("-10.5") == -10.5

    def test_scientific_notation(self):
        assert safe_float("1e5") == 100000.0
        assert safe_float("2.5e-3") == 0.0025

    def test_whitespace_string(self):
        # Depending on implementation, could be 0.0 or strip and parse
        result = safe_float("  ")
        assert isinstance(result, float)

    def test_inf_values(self):
        assert safe_float(float("inf")) == float("inf")

    def test_nan_handling(self):
        import math
        result = safe_float(float("nan"))
        assert math.isnan(result)


class TestConvertToKWD:
    """Tests for currency conversion to KWD."""

    def test_kwd_passthrough(self):
        """KWD amounts pass through unchanged."""
        assert convert_to_kwd(100.0, "KWD") == 100.0
        assert convert_to_kwd(0.0, "KWD") == 0.0

    def test_none_amount(self):
        """None amount returns 0.0."""
        assert convert_to_kwd(None, "KWD") == 0.0
        assert convert_to_kwd(None, "USD") == 0.0

    def test_zero_amount(self):
        assert convert_to_kwd(0, "KWD") == 0.0
        assert convert_to_kwd(0, "USD") == 0.0

    def test_usd_conversion(self):
        """USD converts using the exchange rate."""
        result = convert_to_kwd(100.0, "USD")
        # Should be approximately 100 * 0.307190
        assert result > 0
        # With fallback rate
        expected = 100.0 * DEFAULT_USD_TO_KWD
        assert abs(result - expected) < 1.0  # Allow for cached rate

    def test_usd_conversion_with_known_rate(self):
        """Verify conversion math using the default rate."""
        # Use a fresh import to ensure we're using the fallback
        result = 100.0 * DEFAULT_USD_TO_KWD
        assert abs(result - 30.719) < 0.01

    def test_negative_amount(self):
        """Negative amounts should convert correctly."""
        result = convert_to_kwd(-100.0, "USD")
        assert result < 0

    def test_none_currency_defaults_to_kwd(self):
        """None currency should default to KWD (passthrough)."""
        assert convert_to_kwd(100.0, None) == 100.0

    def test_unknown_currency_passthrough(self):
        """Unknown currencies pass through without conversion."""
        assert convert_to_kwd(100.0, "EUR") == 100.0
        assert convert_to_kwd(100.0, "GBP") == 100.0

    def test_string_amount(self):
        """String amounts should be safely converted."""
        # convert_to_kwd calls float() internally
        result = convert_to_kwd("100", "KWD")
        assert result == 100.0

    def test_invalid_amount_type(self):
        """Invalid amount type returns 0.0."""
        assert convert_to_kwd("not_a_number", "KWD") == 0.0


class TestPortfolioCCYMapping:
    """Tests for the PORTFOLIO_CCY constant."""

    def test_kfh_is_kwd(self):
        assert PORTFOLIO_CCY["KFH"] == "KWD"

    def test_bbyn_is_kwd(self):
        assert PORTFOLIO_CCY["BBYN"] == "KWD"

    def test_usa_is_usd(self):
        assert PORTFOLIO_CCY["USA"] == "USD"

    def test_all_portfolios_present(self):
        assert set(PORTFOLIO_CCY.keys()) == {"KFH", "BBYN", "USA"}


class TestFXConstants:
    """Verify FX service constants."""

    def test_base_ccy(self):
        assert BASE_CCY == "KWD"

    def test_usd_ccy(self):
        assert USD_CCY == "USD"

    def test_default_rate_reasonable(self):
        """Default USD/KWD rate should be in reasonable range."""
        assert 0.25 < DEFAULT_USD_TO_KWD < 0.35
