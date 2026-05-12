from app.data.stock_lists import resolve_yf_ticker_from_lists


def test_resolve_yf_ticker_prefers_kuwait_for_kwd_duplicate_symbol() -> None:
    assert resolve_yf_ticker_from_lists("KRE", "KWD") == "KRE.KW"


def test_resolve_yf_ticker_prefers_us_for_usd_duplicate_symbol() -> None:
    assert resolve_yf_ticker_from_lists("KRE", "USD") == "KRE"
