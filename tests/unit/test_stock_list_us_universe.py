import json

from app.api.v1 import stocks as stocks_api


def test_get_cached_us_universe_prefers_persisted_cache_when_live_feeds_fail(monkeypatch, tmp_path):
    cache_path = tmp_path / "us_stock_universe.json"
    cache_path.write_text(
        json.dumps(
            {
                "updated_at": 1,
                "count": 3,
                "stocks": [
                    {"symbol": "AAPL", "name": "Apple Inc.", "yf_ticker": "AAPL"},
                    {"symbol": "MSFT", "name": "Microsoft Corporation", "yf_ticker": "MSFT"},
                    {"symbol": "ZZZZ", "name": "Test Corp", "yf_ticker": "ZZZZ"},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(stocks_api, "_US_STOCKS_DISK_CACHE_PATH", cache_path)
    monkeypatch.setattr(stocks_api, "_US_STOCKS_CACHE", {"expires_at": 0.0, "stocks": []})

    class _FailingResponse:
        def raise_for_status(self):
            raise RuntimeError("network disabled")

    def _failing_get(*_args, **_kwargs):
        return _FailingResponse()

    import requests

    monkeypatch.setattr(requests, "get", _failing_get)

    universe = stocks_api._get_cached_us_universe()

    assert len(universe) >= 3
    assert {row["symbol"] for row in universe}.issuperset({"AAPL", "MSFT", "ZZZZ"})