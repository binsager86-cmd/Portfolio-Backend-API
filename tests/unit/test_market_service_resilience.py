from __future__ import annotations

import json
import sys
from types import ModuleType

from app.services import market_service


def _install_fake_database_module(monkeypatch, fallback_row):
    fake_module = ModuleType("app.core.database")

    def query_one(sql, params=None):  # noqa: ARG001
        if "WHERE trade_date = ?" in sql:
            return None
        return fallback_row

    def exec_sql(sql, params=None):  # noqa: ARG001
        raise AssertionError("exec_sql should not be called when scraping fails")

    fake_module.query_one = query_one
    fake_module.exec_sql = exec_sql
    monkeypatch.setitem(sys.modules, "app.core.database", fake_module)


def test_get_market_data_returns_stale_cache_when_scrape_fails(monkeypatch):
    fallback_payload = {"indices": [{"name": "Premier Market"}], "market_summary": {"gainers": 1}}
    fallback_row = {"data_json": json.dumps(fallback_payload), "fetched_at": 123456}
    _install_fake_database_module(monkeypatch, fallback_row)

    def _raise_scrape_failure():
        msg = "BrowserType.launch: Executable doesn't exist at /workspace/.cache/ms-playwright/chromium"
        raise RuntimeError(msg)

    monkeypatch.setattr(market_service, "_scrape_market_data", _raise_scrape_failure)

    data = market_service.get_market_data(force_refresh=True)

    assert data["_cached"] is True
    assert data["_stale"] is True
    assert data["_fetched_at"] == 123456
    assert data["indices"] == fallback_payload["indices"]


def test_get_market_data_returns_degraded_payload_when_cache_missing(monkeypatch):
    _install_fake_database_module(monkeypatch, fallback_row=None)

    def _raise_scrape_failure():
        msg = "BrowserType.launch: Executable doesn't exist at /workspace/.cache/ms-playwright/chromium"
        raise RuntimeError(msg)

    monkeypatch.setattr(market_service, "_scrape_market_data", _raise_scrape_failure)

    data = market_service.get_market_data(force_refresh=True)

    assert data["_degraded"] is True
    assert data["_stale"] is True
    assert data["_cached"] is False
    assert data["status"] == "unavailable"
    assert data["indices"] == []
