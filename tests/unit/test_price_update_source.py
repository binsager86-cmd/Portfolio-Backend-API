from app.core.database import query_one
from app.services import price_service
from tests.helpers import create_buy, create_stock


def test_update_all_prices_uses_tickerchart_source(test_client, monkeypatch):
    stock_id = create_stock(symbol="TCFAST", portfolio="KFH", currency="KWD", current_price=0.1)
    create_buy(user_id=1, portfolio="KFH", symbol="TCFAST", shares=10, cost=1.0)

    async def fake_snapshot(symbol: str, currency: str = "KWD") -> dict:
        return {
            "symbol": symbol,
            "price": 0.222,
            "previous_close": 0.2,
            "pe_ratio": 11.5,
            "currency": currency,
            "source": "tickerchart",
        }

    monkeypatch.setattr(price_service, "get_price_snapshot", fake_snapshot)

    result = price_service.update_all_prices(user_id=1)

    assert result.updated >= 1
    row = query_one("SELECT current_price, previous_close, pe_ratio, price_source FROM stocks WHERE id = ?", (stock_id,))
    assert round(float(row[0]), 6) == 0.222
    assert round(float(row[1]), 6) == 0.2
    assert round(float(row[2]), 6) == 11.5
    assert row[3] == "TICKERCHART"
