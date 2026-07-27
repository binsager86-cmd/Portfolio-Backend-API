from app.core.database import query_one
from app.api import portfolio as legacy_portfolio_api
from app.services import price_service
from tests.helpers import create_buy, create_stock


def test_update_all_prices_uses_tickerchart_source(test_client, monkeypatch):
    stock_id = create_stock(symbol="TCFAST", portfolio="KFH", currency="KWD", current_price=0.1)
    create_buy(user_id=1, portfolio="KFH", symbol="TCFAST", shares=10, cost=1.0)

    async def fake_snapshot(symbol: str, currency: str = "KWD", force_refresh: bool = False) -> dict:
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


def test_update_all_prices_forces_fresh_snapshot(test_client, monkeypatch):
    create_stock(symbol="TCFRESH", portfolio="KFH", currency="KWD", current_price=0.1)
    create_buy(user_id=1, portfolio="KFH", symbol="TCFRESH", shares=10, cost=1.0)
    calls = []

    async def fake_snapshot(symbol: str, currency: str = "KWD", force_refresh: bool = False) -> dict:
        calls.append({"symbol": symbol, "force_refresh": force_refresh})
        return {
            "symbol": symbol,
            "price": 0.333,
            "previous_close": 0.3,
            "currency": currency,
            "source": "tickerchart",
        }

    monkeypatch.setattr(price_service, "get_price_snapshot", fake_snapshot)

    price_service.update_all_prices(user_id=1)

    assert any(call == {"symbol": "TCFRESH", "force_refresh": True} for call in calls)


def test_legacy_holdings_endpoint_overlays_live_tickerchart_snapshot(test_client, auth_headers, monkeypatch):
    create_stock(symbol="TCVIEW", portfolio="KFH", currency="KWD", current_price=0.1)
    create_buy(user_id=1, portfolio="KFH", symbol="TCVIEW", shares=10, cost=1.0)

    async def fake_snapshot(symbol: str, currency: str = "KWD") -> dict:
        return {
            "symbol": symbol,
            "price": 0.222,
            "previous_close": 0.2,
            "pe_ratio": 11.5,
            "currency": currency,
            "source": "tickerchart",
        }

    monkeypatch.setattr(legacy_portfolio_api, "get_price_snapshot", fake_snapshot)

    resp = test_client.get("/api/portfolio/holdings", headers=auth_headers)

    assert resp.status_code == 200
    holdings = resp.json()["data"]["holdings"]
    row = next(h for h in holdings if h["symbol"] == "TCVIEW")
    assert round(float(row["market_price"]), 6) == 0.222
    assert round(float(row["previous_close"]), 6) == 0.2
    assert round(float(row["market_value_kwd"]), 6) == 2.22
