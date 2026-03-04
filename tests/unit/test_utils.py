"""
Unit tests for core utility functions.
"""


def test_safe_float():
    from app.utils.currency import safe_float

    assert safe_float(42) == 42.0
    assert safe_float("3.14") == 3.14
    assert safe_float(None) == 0.0
    assert safe_float("", 99.0) == 99.0
    assert safe_float("abc") == 0.0


def test_convert_to_kwd_passthrough():
    from app.utils.currency import convert_to_kwd

    assert convert_to_kwd(100.0, "KWD") == 100.0
    assert convert_to_kwd(None, "KWD") == 0.0
    assert convert_to_kwd(0, "KWD") == 0.0


def test_convert_to_kwd_usd():
    from app.utils.currency import convert_to_kwd, DEFAULT_USD_TO_KWD

    result = convert_to_kwd(100.0, "USD", usd_kwd_rate=DEFAULT_USD_TO_KWD)
    assert abs(result - 100.0 * DEFAULT_USD_TO_KWD) < 0.001


def test_validate_stock_symbol():
    from app.utils.validators import validate_stock_symbol

    assert validate_stock_symbol("AAPL") == "AAPL"
    assert validate_stock_symbol("GFH.KW") == "GFH.KW"
    assert validate_stock_symbol("") is None
    assert validate_stock_symbol("A" * 25) is None  # too long


def test_parse_date_iso():
    from app.utils.date_parser import parse_date

    assert parse_date("2024-01-15") == "2024-01-15"
    assert parse_date(None) is None
    assert parse_date("") is None


def test_parse_date_european():
    from app.utils.date_parser import parse_date

    assert parse_date("15/01/2024") == "2024-01-15"
    assert parse_date("31/12/2023") == "2023-12-31"
