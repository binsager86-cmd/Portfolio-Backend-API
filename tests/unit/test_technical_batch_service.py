from __future__ import annotations

import asyncio

from app.services import technical_batch_service as tbs


def test_score_one_symbol_uses_sync_signal_generator(monkeypatch):
    async def _fake_fetch_ohlcv(base, market, from_d=None, to_d=None):
        del base, market, from_d, to_d
        return [{"date": "2026-05-10", "close": 100.0}]

    monkeypatch.setattr(
        "app.services.tickerchart_service.split_symbol",
        lambda symbol, exchange, market: ("NBK", "KSE"),
    )
    monkeypatch.setattr(
        "app.services.tickerchart_service.fetch_ohlcv",
        _fake_fetch_ohlcv,
    )
    monkeypatch.setattr("app.services.indicators_service.attach_indicators", lambda rows: rows)
    monkeypatch.setattr(
        "app.services.signal_engine.data.preprocessing.forward_fill_gaps",
        lambda rows: rows,
    )

    def _fake_generate_signal(**kwargs):
        del kwargs
        return {
            "signal": "BUY",
            "reason": "ok",
            "confluence_details": {
                "raw_sub_scores": {
                    "trend": 72,
                    "momentum": 64,
                    "volume_flow": 59,
                    "support_resistance": 61,
                },
            },
            "combined_score_adjusted_directional": 67,
            "combined_score_unadjusted_directional": 70,
        }

    monkeypatch.setattr(
        "app.services.signal_engine.engine.signal_generator.generate_kuwait_signal",
        _fake_generate_signal,
    )

    result = asyncio.run(
        tbs._score_one_symbol(
            symbol="NBK",
            company_name="National Bank of Kuwait",
            segment="PREMIER",
            account_equity=100000.0,
        )
    )

    assert result["error"] is None
    assert result["signal"] == "BUY"
    assert result["trend_score"] == 72
    assert result["momentum_score"] == 64
    assert result["buying_pressure_score"] == 59
    assert result["key_price_level_score"] == 61
    assert result["overall_score"] == 67
    assert result["risk_adjusted_score"] == 67
    assert result["raw_technical_score"] == 70
