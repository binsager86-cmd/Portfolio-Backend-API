"""Unit tests for the Kuwait Signal Engine — Momentum Score.

Validates:
  - RSI in bull zone (50-65) → high RSI sub-score
  - RSI overbought → penalised
  - MACD bullish accelerating → high MACD sub-score
  - ROC positive → higher ROC sub-score
  - Stochastic K > D in 40-70 zone → highest Stoch sub-score
  - compute_momentum_score returns (int, dict) with 'stoch_pts' in details
  - Score always in [0, 100]
"""
from __future__ import annotations

import pytest

from app.services.signal_engine.models.technical.momentum_score import compute_momentum_score


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(close: float = 500.0, rsi: float | None = 58.0,
         macd: float | None = 2.0, macd_sig: float | None = 1.5,
         macd_hist: float | None = 0.5, stoch_k: float | None = 55.0,
         stoch_d: float | None = 48.0) -> dict:
    return {
        "close": close, "open": close - 2, "high": close + 5, "low": close - 5,
        "volume": 1_000_000,
        "rsi_14": rsi,
        "macd": macd, "macd_signal": macd_sig, "macd_hist": macd_hist,
        "stoch_k": stoch_k, "stoch_d": stoch_d,
    }


def _rows(n: int = 30, rsi: float = 58.0, **kw) -> list[dict]:
    rows = [_row(close=450.0 + i, rsi=rsi, **kw) for i in range(n)]
    return rows


# ── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_tuple(self):
        r, d = compute_momentum_score(_rows())
        assert isinstance(r, int) and isinstance(d, dict)

    def test_score_in_range(self):
        score, _ = compute_momentum_score(_rows())
        assert 0 <= score <= 100

    def test_details_has_stoch_key(self):
        _, details = compute_momentum_score(_rows())
        assert "stoch_pts" in details
        assert "stoch_desc" in details

    def test_empty_rows_returns_neutral(self):
        score, d = compute_momentum_score([])
        assert 0 <= score <= 100


# ── RSI ───────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_bull_zone_rsi_gives_max_rsi_pts(self):
        rows = _rows(rsi=58.0)   # 50-65 is optimal
        _, d = compute_momentum_score(rows)
        assert d["rsi_pts"] == 25

    def test_overbought_rsi_penalised(self):
        rows = _rows(rsi=75.0)
        _, d_ob = compute_momentum_score(rows)
        rows_ok = _rows(rsi=58.0)
        _, d_ok = compute_momentum_score(rows_ok)
        assert d_ob["rsi_pts"] < d_ok["rsi_pts"]

    def test_deeply_oversold_gives_low_pts(self):
        rows = _rows(rsi=20.0)
        _, d = compute_momentum_score(rows)
        assert d["rsi_pts"] <= 5

    def test_missing_rsi_returns_neutral_pts(self):
        rows = _rows()
        rows[-1]["rsi_14"] = None
        _, d = compute_momentum_score(rows)
        assert d["rsi_pts"] == 12


# ── MACD ─────────────────────────────────────────────────────────────────────

class TestMACD:
    def test_bullish_accelerating_macd_gives_max(self):
        rows = _rows(macd=2.0, macd_sig=1.5, macd_hist=0.5)
        _, d = compute_momentum_score(rows)
        assert d["macd_pts"] == 40

    def test_bearish_macd_gives_low_pts(self):
        rows = _rows(macd=-1.0, macd_sig=0.5, macd_hist=-0.5)
        _, d = compute_momentum_score(rows)
        assert d["macd_pts"] == 5

    def test_missing_macd_returns_neutral_pts(self):
        rows = _rows()
        rows[-1]["macd"] = None
        _, d = compute_momentum_score(rows)
        assert d["macd_pts"] == 17


# ── ROC ───────────────────────────────────────────────────────────────────────

class TestROC:
    def test_positive_roc_above_5pct_gives_max(self):
        closes = [400.0 + i * 4.0 for i in range(30)]  # >5% gain over 10 bars
        rows = [_row(close=c) for c in closes]
        _, d = compute_momentum_score(rows)
        assert d["roc_pts"] == 25

    def test_negative_roc_gives_zero(self):
        closes = [500.0 - i * 5.0 for i in range(30)]  # steep decline
        rows = [_row(close=c) for c in closes]
        _, d = compute_momentum_score(rows)
        assert d["roc_pts"] == 0


# ── Stochastic ────────────────────────────────────────────────────────────────

class TestStochastic:
    def test_k_above_d_in_bull_zone_gives_max(self):
        rows = _rows(stoch_k=55.0, stoch_d=45.0)  # K > D, K in 40-70
        _, d = compute_momentum_score(rows)
        assert d["stoch_pts"] == 10

    def test_k_below_d_bearish_gives_zero(self):
        rows = _rows(stoch_k=35.0, stoch_d=50.0)  # K < D, K < 40
        _, d = compute_momentum_score(rows)
        assert d["stoch_pts"] == 0

    def test_overbought_stoch_gives_reduced_pts(self):
        rows = _rows(stoch_k=85.0, stoch_d=70.0)  # K > D but overbought
        _, d = compute_momentum_score(rows)
        assert d["stoch_pts"] == 3

    def test_missing_stoch_returns_neutral(self):
        rows = _rows()
        rows[-1]["stoch_k"] = None
        _, d = compute_momentum_score(rows)
        assert d["stoch_pts"] == 5


# ── Score ceiling ─────────────────────────────────────────────────────────────

class TestScoreCeiling:
    def test_perfect_conditions_capped_at_100(self):
        # RSI=25 + MACD=40 + ROC=25 + Stoch=10 = 100
        closes = [400.0 + i * 4.0 for i in range(30)]
        rows = [_row(close=c, rsi=58.0, macd=3.0, macd_sig=1.0, macd_hist=2.0,
                     stoch_k=55.0, stoch_d=45.0) for c in closes]
        score, _ = compute_momentum_score(rows)
        assert score == 100

    def test_details_has_component_keys(self):
        _, d = compute_momentum_score(_rows())
        for key in ("rsi_pts", "macd_pts", "roc_pts", "stoch_pts", "raw_score"):
            assert key in d, f"missing key: {key}"

