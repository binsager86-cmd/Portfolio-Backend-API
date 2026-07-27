from tests.helpers import create_buy, create_stock


def test_v1_holdings_consolidates_same_symbol_across_portfolios(test_client, auth_headers):
    create_stock(symbol="KRE", portfolio="KFH", currency="KWD", current_price=0.338)
    create_stock(symbol="KRE", portfolio="BBYN", currency="KWD", current_price=0.338)

    create_buy(user_id=1, portfolio="KFH", symbol="KRE", shares=139_847, cost=0.350)
    create_buy(user_id=1, portfolio="BBYN", symbol="KRE", shares=107_529, cost=0.352)

    resp = test_client.get("/api/v1/portfolio/holdings", headers=auth_headers)

    assert resp.status_code == 200
    payload = resp.json()["data"]
    rows = [h for h in payload["holdings"] if h.get("symbol") == "KRE"]

    assert len(rows) == 1
    assert round(float(rows[0]["shares_qty"]), 3) == round(139_847 + 107_529, 3)