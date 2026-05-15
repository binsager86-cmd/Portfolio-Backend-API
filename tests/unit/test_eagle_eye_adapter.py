from __future__ import annotations

from app.data.stock_lists import KUWAIT_STOCKS
from app.services.eagle_eye.adapter import TickerChartAdapter


def test_list_stocks_keeps_market_wide_universe_when_analysis_stocks_is_partial(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.core.database.query_all",
        lambda *_args, **_kwargs: [
            {
                "symbol": "NBK",
                "company_name": "NBK From Analysis Stocks",
                "exchange": "KW",
                "currency": "KWD",
                "sector": "Banking",
            }
        ],
    )

    stocks = TickerChartAdapter().list_stocks()
    by_ticker = {s.ticker: s for s in stocks}

    kuwait_tickers = {str(s["symbol"]).upper() for s in KUWAIT_STOCKS}
    assert kuwait_tickers.issubset(set(by_ticker.keys()))
    assert by_ticker["NBK"].name_en == "NBK From Analysis Stocks"


def test_list_stocks_includes_db_only_kuwait_tickers(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.core.database.query_all",
        lambda *_args, **_kwargs: [
            {
                "symbol": "CUSTOMKW",
                "company_name": "Custom Kuwait Co",
                "exchange": "KW",
                "currency": "KWD",
                "sector": "Custom Sector",
            }
        ],
    )

    stocks = TickerChartAdapter().list_stocks()
    by_ticker = {s.ticker: s for s in stocks}

    assert "CUSTOMKW" in by_ticker
    assert by_ticker["CUSTOMKW"].name_en == "Custom Kuwait Co"
    assert by_ticker["CUSTOMKW"].sector == "Custom Sector"
