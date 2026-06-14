from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest

from app.services import market_service


def _install_fake_database_module(monkeypatch, fallback_row):
    fake_module = ModuleType("app.core.database")

    def query_one(sql, params=None):  # noqa: ARG001
        if "WHERE trade_date = ?" in sql:
            return None
        return fallback_row

    def exec_sql(sql, params=None):  # noqa: ARG001
        raise AssertionError("exec_sql should not be called when market fetch fails")

    fake_module.query_one = query_one
    fake_module.exec_sql = exec_sql
    monkeypatch.setitem(sys.modules, "app.core.database", fake_module)


def _install_failing_market_fetch(monkeypatch):
    fake_stock_list = ModuleType("app.data.stock_lists")
    fake_stock_list.KUWAIT_STOCKS = [{"symbol": "KFH", "name": "Kuwait Finance House"}]
    monkeypatch.setitem(sys.modules, "app.data.stock_lists", fake_stock_list)

    fake_tickerchart = ModuleType("app.services.tickerchart_service")

    async def _raise_fetch_failure(symbols, stock_name_map):  # noqa: ARG001
        msg = "TickerChart unavailable"
        raise RuntimeError(msg)

    fake_tickerchart.fetch_kse_market_snapshot = _raise_fetch_failure
    monkeypatch.setitem(sys.modules, "app.services.tickerchart_service", fake_tickerchart)


@pytest.mark.asyncio
async def test_get_market_data_returns_stale_cache_when_live_fetch_fails(monkeypatch):
    fallback_payload = {"indices": [{"name": "Premier Market"}], "market_summary": {"gainers": 1}}
    fallback_row = {"data_json": json.dumps(fallback_payload), "fetched_at": 123456}
    _install_fake_database_module(monkeypatch, fallback_row)
    _install_failing_market_fetch(monkeypatch)

    data = await market_service.get_market_data(force_refresh=True)

    assert data["_cached"] is True
    assert data["_stale"] is True
    assert data["_fetched_at"] == 123456
    assert data["indices"] == fallback_payload["indices"]


@pytest.mark.asyncio
async def test_get_market_data_returns_degraded_payload_when_cache_missing(monkeypatch):
    _install_fake_database_module(monkeypatch, fallback_row=None)
    _install_failing_market_fetch(monkeypatch)

    data = await market_service.get_market_data(force_refresh=True)

    assert data["_degraded"] is True
    assert data["_stale"] is True
    assert data["_cached"] is False
    assert data["status"] == "unavailable"
    assert data["indices"] == []
