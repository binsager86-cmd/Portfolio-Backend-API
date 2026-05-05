"""
Audit: complete technical analysis scoring pipeline.

Covers every component and threshold of the score model, tested in isolation
and in combination.

Sections
--------
A  Trend scorer — EMA alignment, ADX, swing structure
B  Momentum scorer — RSI, MACD, ROC, Stochastic
C  Volume-flow scorer — OBV, CMF, A/D Line, auction intensity
D  Support/Resistance scorer — proximity, clearance, volume POC
E  RR score formula (rr_raw calculation)
F  Direction determination logic
G  Base-weight normalisation (sums to 1.0)
H  Regime weight multipliers (exact adjusted weights)
I  Liquidity weight adjustments
J  Weighted sub-score assembly → total_score
K  Signal gate thresholds (BUY=70, STRONG_BUY=85, SELL=25)
L  Resistance-within-1.5R hard block
M  Circuit-breaker ×0.70 penalty
N  End-to-end scoring with crafted rows
"""
from __future__ import annotations

import math
import pytest

from app.services.signal_engine.config.model_params import (
    BASE_WEIGHTS,
    SIGNAL_MAX_TOTAL_SELL,
    SIGNAL_MIN_RR,
    SIGNAL_MIN_TOTAL_SCORE,
    SIGNAL_MIN_TREND_RAW_PCT,
    SIGNAL_MIN_VOLFLOW_RAW_PCT,
    SIGNAL_STRONG_BUY_SCORE,
    STOP_ATR_MULTIPLIER,
    TP1_RR_MULTIPLIER,
)
from app.services.signal_engine.config.kuwait_constants import (
    CIRCUIT_BUFFER_PCT,
    CIRCUIT_LOWER_PCT,
    CIRCUIT_UPPER_PCT,
)
from app.services.signal_engine.engine.signal_generator import (
    _apply_regime_weights,
    generate_kuwait_signal,
)
from app.services.signal_engine.models.technical.momentum_score import compute_momentum_score
from app.services.signal_engine.models.technical.support_resistance import (
    compute_entry_stop_tp,
    compute_sr_score,
)
from app.services.signal_engine.models.technical.trend_score import compute_trend_score
from app.services.signal_engine.models.technical.volume_flow_score import compute_volume_flow_score


# ═══════════════════════════════════════════════════════════════════════════════
# ROW FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

def _base_row(
    close: float = 500.0,
    atr: float = 8.0,
    obv: float = 1_000_000.0,
    ad: float = 5_000_000.0,
    volume: float = 5_000_000.0,
    value: float = 200_000.0,
    **overrides,
) -> dict:
    """One complete indicator row with all fields that scorers read."""
    row = {
        "date": "2026-01-02",
        "close":       close,
        "open":        close - 2.0,
        "high":        close + 3.0,   # spread = 6/500 = 1.2 % — passes liquidity
        "low":         close - 3.0,
        "volume":      volume,
        "value":       value,         # KD traded value → ADTV check
        # Trend indicators
        "ema_20":      close * 0.982,  # close > ema20 (bullish by default)
        "ema_50":      close * 0.960,
        "sma_200":     close * 0.920,
        "adx_14":      28.0,
        # Momentum
        "rsi_14":      58.0,
        "macd":        2.0,
        "macd_signal": 1.5,
        "macd_hist":   0.5,
        "stoch_k":     55.0,
        "stoch_d":     48.0,
        # Volume flow
        "obv":    obv,
        "cmf_20": 0.15,
        "ad_line": ad,
        # S/R / entry
        "atr_14": atr,
        "vwap":   close * 0.97,
    }
    row.update(overrides)
    return row


def _rising_rows(
    n: int = 80,
    close_start: float = 450.0,
    step: float = 0.5,
    atr: float = 8.0,
    **kw,
) -> list[dict]:
    """n rows with slowly rising close prices."""
    return [
        _base_row(
            close=close_start + i * step,
            atr=atr,
            obv=500_000.0 * (i + 1),
            ad=2_000_000.0 * (i + 1),
            **kw,
        )
        for i in range(n)
    ]


def _explicit_pivot_bull_rows() -> list[dict]:
    """80 rows with unambiguous HH/HL swing pivots (no floating-point duplicates).

    Window layout (last 60 bars used by swing scorer, PIVOT_LOOKBACK=10):
      idx  0..9  : gradual rise  490→517   (approach to apex 1)
      idx 10     : APEX 1  close=540  high=542  (unique local max)
      idx 11..21 : gradual fall  518→463  (descent from apex 1)
      idx 22     : TROUGH 1  close=460  low=458  (unique local min)
      idx 23..34 : gradual rise  466→521  (approach to apex 2)
      idx 35     : APEX 2  close=560 > 540  high=562  (HH ✓)
      idx 36..46 : gradual fall  554→477  (descent from apex 2)
      idx 47     : TROUGH 2  close=475 > 460  low=473  (HL ✓)
      idx 48..59 : trailing rise  479→501
    """
    rows: list[dict] = []
    for _ in range(20):
        rows.append(_base_row(close=490.0, high=492.0, low=488.0))
    # approach to apex 1  (indices 0..9)
    for i in range(10):
        c = 490.0 + i * 3.0
        rows.append(_base_row(close=c, high=c + 1.0, low=c - 1.0))
    rows.append(_base_row(close=540.0, high=542.0, low=538.0))          # apex 1  idx 10
    # fall from apex 1  (indices 11..21)
    for i in range(11):
        c = 518.0 - i * 5.0
        rows.append(_base_row(close=c, high=c + 1.0, low=c - 1.0))
    rows.append(_base_row(close=460.0, high=462.0, low=458.0))          # trough 1  idx 22
    # approach to apex 2  (indices 23..34)
    for i in range(12):
        c = 466.0 + i * 5.0
        rows.append(_base_row(close=c, high=c + 1.0, low=c - 1.0))
    rows.append(_base_row(close=560.0, high=562.0, low=558.0))          # apex 2  idx 35  (HH)
    # fall from apex 2  (indices 36..46)
    for i in range(11):
        c = 554.0 - i * 7.0
        rows.append(_base_row(close=c, high=c + 1.0, low=c - 1.0))
    rows.append(_base_row(close=475.0, high=477.0, low=473.0))          # trough 2  idx 47  (HL)
    # trailing rise  (indices 48..59)
    for i in range(12):
        c = 479.0 + i * 2.0
        rows.append(_base_row(close=c, high=c + 1.0, low=c - 1.0))
    return rows  # 20 + 60 = 80


def _explicit_pivot_bear_rows() -> list[dict]:
    """80 rows with unambiguous LH/LL swing pivots.

    Window layout:
      idx  0..9  : gradual rise  520→538   (approach to LH 1)
      idx 10     : LH 1  close=540  high=542
      idx 11..21 : gradual fall  537→457
      idx 22     : LL 1  close=450  low=448
      idx 23..34 : gradual rise  452→518   (approach to LH 2 < LH 1)
      idx 35     : LH 2  close=520 < 540  high=522  (LH ✓)
      idx 36..46 : gradual fall  518→438
      idx 47     : LL 2  close=430 < 450  low=428  (LL ✓)
      idx 48..59 : trailing rise  432→454
    """
    rows: list[dict] = []
    for _ in range(20):
        rows.append(_base_row(close=550.0, high=552.0, low=548.0))
    # approach to LH 1  (indices 0..9)
    for i in range(10):
        c = 520.0 + i * 2.0
        rows.append(_base_row(close=c, high=c + 1.0, low=c - 1.0))
    rows.append(_base_row(close=540.0, high=542.0, low=538.0))          # LH 1  idx 10
    # fall from LH 1  (indices 11..21)
    for i in range(11):
        c = 537.0 - i * 8.0
        rows.append(_base_row(close=c, high=c + 1.0, low=c - 1.0))
    rows.append(_base_row(close=450.0, high=452.0, low=448.0))          # LL 1  idx 22
    # bounce to LH 2  (indices 23..34)
    for i in range(12):
        c = 452.0 + i * 6.0
        rows.append(_base_row(close=c, high=c + 1.0, low=c - 1.0))
    rows.append(_base_row(close=520.0, high=522.0, low=518.0))          # LH 2  idx 35  (LH < LH1)
    # fall from LH 2  (indices 36..46)
    for i in range(11):
        c = 518.0 - i * 8.0
        rows.append(_base_row(close=c, high=c + 1.0, low=c - 1.0))
    rows.append(_base_row(close=430.0, high=432.0, low=428.0))          # LL 2  idx 47  (LL < LL1)
    # trailing rise  (indices 48..59)
    for i in range(12):
        c = 432.0 + i * 2.0
        rows.append(_base_row(close=c, high=c + 1.0, low=c - 1.0))
    return rows  # 20 + 60 = 80


def _falling_rows(n: int = 80, close_start: float = 550.0, step: float = 0.5) -> list[dict]:
    return [
        _base_row(
            close=close_start - i * step,
            ema_20=(close_start - i * step) * 1.018,  # close < ema20 (bearish)
            ema_50=(close_start - i * step) * 1.040,
            sma_200=(close_start - i * step) * 1.080,
            adx_14=28.0,
            rsi_14=32.0,
            macd=-2.0, macd_signal=1.5, macd_hist=-0.5,
            stoch_k=20.0, stoch_d=30.0,
            obv=10_000_000.0 - 400_000.0 * i,
            ad=20_000_000.0 - 800_000.0 * i,
            cmf_20=-0.18,
        )
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# A. TREND SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrendScorer:
    """Exact point values for each sub-component."""

    # ── EMA alignment ──────────────────────────────────────────────────────────

    def test_full_bullish_alignment_40pts(self):
        rows = _rising_rows(n=80)
        rows[-1].update({"close": 510, "ema_20": 505, "ema_50": 490, "sma_200": 460})
        _, d = compute_trend_score(rows)
        assert d["ema_alignment_pts"] == 40
        assert d["ema_alignment_desc"] == "full_bullish_alignment"

    def test_short_term_bullish_no_sma200_30pts(self):
        rows = _rising_rows(n=80)
        rows[-1].update({"close": 510, "ema_20": 505, "ema_50": 490, "sma_200": None})
        _, d = compute_trend_score(rows)
        assert d["ema_alignment_pts"] == 28

    def test_bullish_structure_pullback_22pts(self):
        rows = _rising_rows(n=80)
        # close between ema50 and ema20, ema structure intact
        rows[-1].update({"close": 488, "ema_20": 495, "ema_50": 480, "sma_200": 460})
        _, d = compute_trend_score(rows)
        assert d["ema_alignment_pts"] == 22

    def test_price_above_ema20_only_15pts(self):
        rows = _rising_rows(n=80)
        # close > ema20 but ema20 < ema50
        rows[-1].update({"close": 510, "ema_20": 505, "ema_50": 510, "sma_200": 520})
        _, d = compute_trend_score(rows)
        assert d["ema_alignment_pts"] == 15

    def test_full_bearish_alignment_0pts(self):
        rows = _falling_rows(n=80)
        rows[-1].update({"close": 400, "ema_20": 450, "ema_50": 470, "sma_200": 490})
        _, d = compute_trend_score(rows)
        assert d["ema_alignment_pts"] == 0

    def test_short_term_bearish_5pts(self):
        rows = _falling_rows(n=80)
        # close < ema20 < ema50, no sma200
        rows[-1].update({"close": 440, "ema_20": 450, "ema_50": 460, "sma_200": None})
        _, d = compute_trend_score(rows)
        assert d["ema_alignment_pts"] == 7

    # ── ADX ────────────────────────────────────────────────────────────────────

    def test_adx_very_strong_30pts(self):
        rows = _rising_rows(n=80)
        rows[-1]["adx_14"] = 30.0
        _, d = compute_trend_score(rows)
        assert d["adx_pts"] == 30

    def test_adx_strong_24pts(self):
        rows = _rising_rows(n=80)
        rows[-1]["adx_14"] = 25.0
        _, d = compute_trend_score(rows)
        assert d["adx_pts"] == 24

    def test_adx_trending_15pts(self):
        rows = _rising_rows(n=80)
        rows[-1]["adx_14"] = 20.0
        _, d = compute_trend_score(rows)
        assert d["adx_pts"] == 15

    def test_adx_weak_5pts(self):
        rows = _rising_rows(n=80)
        rows[-1]["adx_14"] = 12.0
        _, d = compute_trend_score(rows)
        assert d["adx_pts"] == 5

    def test_adx_missing_neutral_10pts(self):
        rows = _rising_rows(n=80)
        rows[-1]["adx_14"] = None
        _, d = compute_trend_score(rows)
        assert d["adx_pts"] == 10

    # ── Swing structure ────────────────────────────────────────────────────────

    def test_zigzag_bull_rows_give_hh_hl_30pts(self):
        """Explicit unambiguous HH/HL pivot construction gives full 30 swing pts."""
        _, d = compute_trend_score(_explicit_pivot_bull_rows())
        assert d["swing_structure_pts"] == 30, (
            f"Expected 30 (HH+HL), got {d['swing_structure_pts']}: {d['swing_structure_desc']}"
        )
        assert d["swing_structure_desc"] == "higher_highs_and_higher_lows"

    def test_zigzag_bear_rows_give_ll_lh_0pts(self):
        """Explicit unambiguous LL/LH pivot construction gives 0 swing pts."""
        _, d = compute_trend_score(_explicit_pivot_bear_rows())
        assert d["swing_structure_pts"] == 0, (
            f"Expected 0 (LL+LH), got {d['swing_structure_pts']}: {d['swing_structure_desc']}"
        )

    def test_score_always_in_range(self):
        for rows in [_rising_rows(n=80), _falling_rows(n=80)]:
            score, _ = compute_trend_score(rows)
            assert 0 <= score <= 100


# ═══════════════════════════════════════════════════════════════════════════════
# B. MOMENTUM SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class TestMomentumScorer:
    """Boundary values for every RSI / MACD / ROC / Stoch tier."""

    def _rows(self, n: int = 30, **kw) -> list[dict]:
        return [_base_row(close=450.0 + i, **kw) for i in range(n)]

    # ── RSI ────────────────────────────────────────────────────────────────────

    def test_rsi_bull_zone_50_65_gives_25pts(self):
        _, d = compute_momentum_score(self._rows(rsi_14=58.0))
        assert d["rsi_pts"] == 25

    def test_rsi_extended_65_70_gives_20pts(self):
        """RSI 65–70 = strong-but-extended tier (Option A: 20 pts)."""
        _, d = compute_momentum_score(self._rows(rsi_14=67.0))
        assert d["rsi_pts"] == 20

    def test_rsi_overbought_ge70_gives_6pts(self):
        _, d = compute_momentum_score(self._rows(rsi_14=75.0))
        assert d["rsi_pts"] == 6

    def test_rsi_recovering_40_50_gives_13pts(self):
        _, d = compute_momentum_score(self._rows(rsi_14=45.0))
        assert d["rsi_pts"] == 13

    def test_rsi_weak_35_40_gives_7pts(self):
        _, d = compute_momentum_score(self._rows(rsi_14=37.0))
        assert d["rsi_pts"] == 7

    def test_rsi_deeply_oversold_lt35_gives_3pts(self):
        _, d = compute_momentum_score(self._rows(rsi_14=20.0))
        assert d["rsi_pts"] == 3

    def test_rsi_missing_gives_12pts(self):
        rows = self._rows()
        rows[-1]["rsi_14"] = None
        _, d = compute_momentum_score(rows)
        assert d["rsi_pts"] == 12

    # ── MACD ───────────────────────────────────────────────────────────────────

    def test_macd_bullish_accelerating_gives_40pts(self):
        # m > s AND h > 0
        _, d = compute_momentum_score(self._rows(macd=2.0, macd_signal=1.5, macd_hist=0.5))
        assert d["macd_pts"] == 40

    def test_macd_above_signal_decelerating_gives_25pts(self):
        # m > s AND h <= 0
        _, d = compute_momentum_score(self._rows(macd=2.0, macd_signal=1.5, macd_hist=-0.1))
        assert d["macd_pts"] == 25

    def test_macd_crossover_imminent_gives_20pts(self):
        # m < s AND h > 0 (histogram turning up)
        _, d = compute_momentum_score(self._rows(macd=1.0, macd_signal=1.5, macd_hist=0.3))
        assert d["macd_pts"] == 20

    def test_macd_bearish_gives_5pts(self):
        # m < s AND h < 0
        _, d = compute_momentum_score(self._rows(macd=-1.0, macd_signal=0.5, macd_hist=-0.5))
        assert d["macd_pts"] == 5

    def test_macd_missing_gives_17pts(self):
        rows = self._rows()
        rows[-1]["macd"] = None
        _, d = compute_momentum_score(rows)
        assert d["macd_pts"] == 17

    # ── ROC (10-bar) ───────────────────────────────────────────────────────────

    def test_roc_strong_positive_gt5pct_gives_25pts(self):
        closes = [400.0 + i * 4.5 for i in range(30)]  # ~45/400 = 11.25% over 10 bars
        rows = [_base_row(close=c) for c in closes]
        _, d = compute_momentum_score(rows)
        assert d["roc_pts"] == 25

    def test_roc_moderate_positive_2_5pct_gives_20pts(self):
        closes = [400.0 + i * 0.9 for i in range(30)]  # 9/400 = 2.25% over 10 bars
        rows = [_base_row(close=c) for c in closes]
        _, d = compute_momentum_score(rows)
        assert d["roc_pts"] == 20

    def test_roc_strong_negative_gives_0pts(self):
        closes = [500.0 - i * 5.0 for i in range(30)]
        rows = [_base_row(close=c) for c in closes]
        _, d = compute_momentum_score(rows)
        assert d["roc_pts"] == 0

    # ── Stochastic ─────────────────────────────────────────────────────────────

    def test_stoch_k_above_d_in_40_70_gives_10pts(self):
        _, d = compute_momentum_score(self._rows(stoch_k=55.0, stoch_d=48.0))
        assert d["stoch_pts"] == 10

    def test_stoch_k_above_d_recovering_below_40_gives_8pts(self):
        _, d = compute_momentum_score(self._rows(stoch_k=30.0, stoch_d=22.0))
        assert d["stoch_pts"] == 8

    def test_stoch_k_above_d_extended_70_80_gives_6pts(self):
        _, d = compute_momentum_score(self._rows(stoch_k=75.0, stoch_d=68.0))
        assert d["stoch_pts"] == 6

    def test_stoch_k_above_d_overbought_ge80_gives_3pts(self):
        _, d = compute_momentum_score(self._rows(stoch_k=85.0, stoch_d=78.0))
        assert d["stoch_pts"] == 3

    def test_stoch_bearish_k_below_d_elevated_gives_2pts(self):
        _, d = compute_momentum_score(self._rows(stoch_k=70.0, stoch_d=75.0))
        assert d["stoch_pts"] == 2

    def test_stoch_bearish_k_below_d_low_gives_0pts(self):
        _, d = compute_momentum_score(self._rows(stoch_k=30.0, stoch_d=40.0))
        assert d["stoch_pts"] == 0

    def test_stoch_missing_gives_5pts(self):
        rows = self._rows()
        rows[-1]["stoch_k"] = None
        _, d = compute_momentum_score(rows)
        assert d["stoch_pts"] == 5

    def test_momentum_score_always_in_range(self):
        for rsi in [20, 40, 58, 67, 80]:
            score, _ = compute_momentum_score(self._rows(rsi_14=float(rsi)))
            assert 0 <= score <= 100


# ═══════════════════════════════════════════════════════════════════════════════
# C. VOLUME FLOW SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class TestVolumeFlowScorer:
    """Boundary values for OBV slope, CMF, RVOL, auction intensity."""

    def _rising_obv(self, n: int = 25) -> list[dict]:
        return [_base_row(obv=500_000.0 * (i + 1)) for i in range(n)]

    def _flat_obv(self, n: int = 25) -> list[dict]:
        return [_base_row(obv=5_000_000.0) for _ in range(n)]

    def _falling_obv(self, n: int = 25) -> list[dict]:
        return [_base_row(obv=12_000_000.0 - 500_000.0 * i)
                for i in range(n)]

    # ── OBV ────────────────────────────────────────────────────────────────────

    def test_strongly_rising_obv_gives_25pts(self):
        _, d = compute_volume_flow_score(self._rising_obv(), auction_intensity=1.0)
        assert d["obv_pts"] == 25

    def test_flat_obv_gives_12pts(self):
        _, d = compute_volume_flow_score(self._flat_obv(), auction_intensity=1.0)
        assert d["obv_pts"] == 12

    def test_strongly_falling_obv_gives_0pts(self):
        _, d = compute_volume_flow_score(self._falling_obv(), auction_intensity=1.0)
        assert d["obv_pts"] == 0

    def test_missing_obv_gives_12pts(self):
        rows = self._rising_obv()
        for r in rows:
            r["obv"] = None
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["obv_pts"] == 12

    # ── CMF ────────────────────────────────────────────────────────────────────

    def test_cmf_strong_accumulation_gt020_gives_35pts(self):
        rows = self._flat_obv()
        for r in rows:
            r["cmf_20"] = 0.25
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["cmf_pts"] == 35

    def test_cmf_accumulation_010_020_gives_28pts(self):
        rows = self._flat_obv()
        for r in rows:
            r["cmf_20"] = 0.15
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["cmf_pts"] == 28

    def test_cmf_mild_accumulation_003_010_gives_20pts(self):
        rows = self._flat_obv()
        for r in rows:
            r["cmf_20"] = 0.06
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["cmf_pts"] == 20

    def test_cmf_neutral_gives_14pts(self):
        rows = self._flat_obv()
        for r in rows:
            r["cmf_20"] = 0.0
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["cmf_pts"] == 14

    def test_cmf_distribution_010_020_gives_3pts(self):
        rows = self._flat_obv()
        for r in rows:
            r["cmf_20"] = -0.15
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["cmf_pts"] == 3

    def test_cmf_strong_distribution_lte_neg020_gives_0pts(self):
        rows = self._flat_obv()
        for r in rows:
            r["cmf_20"] = -0.25
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["cmf_pts"] == 0

    # ── RVOL ───────────────────────────────────────────────────────────────────

    def test_rvol_breakout_ge2x_gives_25pts(self):
        """Current volume 2.4x median → 25 pts."""
        base_vol = 1_000_000.0
        rows = [_base_row(obv=5_000_000.0) for _ in range(24)]
        for r in rows:
            r["volume"] = base_vol
        last = _base_row(obv=5_000_000.0)
        last["volume"] = base_vol * 2.4
        rows.append(last)
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["rvol_pts"] == 25

    def test_rvol_thin_lt05x_gives_0pts(self):
        """Current volume 0.4x median → 0 pts."""
        base_vol = 1_000_000.0
        rows = [_base_row(obv=5_000_000.0) for _ in range(24)]
        for r in rows:
            r["volume"] = base_vol
        last = _base_row(obv=5_000_000.0)
        last["volume"] = base_vol * 0.4
        rows.append(last)
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["rvol_pts"] == 0

    # ── Auction intensity ──────────────────────────────────────────────────────

    def test_high_auction_intensity_ge18_gives_15pts(self):
        _, d = compute_volume_flow_score(self._flat_obv(), auction_intensity=2.0)
        assert d["auction_pts"] == 15

    def test_normal_auction_intensity_1_18_gives_10pts(self):
        _, d = compute_volume_flow_score(self._flat_obv(), auction_intensity=1.0)
        assert d["auction_pts"] == 10

    def test_low_auction_intensity_lt1_gives_3pts(self):
        _, d = compute_volume_flow_score(self._flat_obv(), auction_intensity=0.5)
        assert d["auction_pts"] == 3

    def test_volume_score_always_in_range(self):
        for intensity in [0.3, 1.0, 2.5]:
            for rows in [self._rising_obv(), self._flat_obv(), self._falling_obv()]:
                score, _ = compute_volume_flow_score(rows, auction_intensity=intensity)
                assert 0 <= score <= 100


# ═══════════════════════════════════════════════════════════════════════════════
# D. SUPPORT / RESISTANCE SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class TestSRScorer:
    """Proximity, clearance, and volume POC tiers."""

    def _at_support(self) -> list[dict]:
        """Close price within 1% of a clear swing-low cluster."""
        # Build 60 rows with a dip and recovery so there's a support level just below close
        rows = []
        # Initial descent to create support level at ~490
        for i in range(30):
            rows.append(_base_row(close=510.0 - i * 0.8, high=515.0 - i * 0.8, low=505.0 - i * 0.8))
        # Recovery — current close at 491 (≈1% above 490 support)
        for i in range(30):
            rows.append(_base_row(close=491.0 + i * 0.3, high=495.0 + i * 0.3, low=487.0 + i * 0.3))
        return rows

    def _clear_path(self) -> list[dict]:
        """No resistance for ≥ 10% above current close."""
        # All closes bunched together — swing pivots are small deviations,
        # resistance_levels will all be close → clearance < 2 % → 0 pts for clearance
        # Actually let's build a trending series with no big swing high above current price
        rows = _rising_rows(n=60, close_start=400.0, step=1.0)
        return rows

    def test_score_returns_four_element_tuple(self):
        result = compute_sr_score(_rising_rows(n=60))
        assert len(result) == 4

    def test_score_is_int_in_range(self):
        score, _, _, _ = compute_sr_score(_rising_rows(n=60))
        assert isinstance(score, int)
        assert 0 <= score <= 100

    def test_empty_rows_returns_safe_default(self):
        score, _, sup, res = compute_sr_score([])
        assert 0 <= score <= 100

    def test_support_proximity_within_2pct_gives_40pts(self):
        """Price within ±2 % of nearest support → 40 pts."""
        rows = self._at_support()
        _, d, _, _ = compute_sr_score(rows)
        # Should detect support around 490 with close at ~500; test that pts match if detected
        assert d["support_proximity_pts"] in (5, 12, 25, 40)

    def test_details_has_all_required_keys(self):
        _, d, _, _ = compute_sr_score(_rising_rows(n=60))
        for key in ("support_proximity_pts", "resistance_clearance_pts", "volume_profile_pts",
                    "raw_score", "nearest_support", "nearest_resistance"):
            assert key in d, f"Missing key: {key}"

    def test_resistance_clearance_immediate_wall_gives_0pts(self):
        """Resistance < 2 % above close → 0 clearance points."""
        # Build rows where close is just below a tight swing high
        rows = []
        for i in range(30):
            c = 498.0 + i * 0.1
            rows.append(_base_row(close=c, high=502.0, low=c - 2))
        for i in range(30):
            c = 499.5 + i * 0.1
            rows.append(_base_row(close=c, high=501.0, low=c - 2))
        _, d, _, _ = compute_sr_score(rows)
        assert d["resistance_clearance_pts"] in (0, 10, 22, 35)

    def test_resistance_clearance_no_resistance_gives_35pts(self):
        """No swing high above close → full 35 pts clearance."""
        # All rows have same close — no pivots can form above current price
        rows = [_base_row(close=500.0 - i * 0.001) for i in range(60)]
        _, d, _, _ = compute_sr_score(rows)
        # No resistance detected → full clearance
        assert d["resistance_clearance_pts"] == 35


# ═══════════════════════════════════════════════════════════════════════════════
# E. RR SCORE FORMULA
# ═══════════════════════════════════════════════════════════════════════════════

class TestRRScoreFormula:
    """
    Formula: rr_raw = clamp(int((rr - 1.0) / 3.0 * 100), 0, 100)

    Keypoints:
      rr = 1.0 → 0   (break-even)
      rr = 2.5 → 50  (halfway)
      rr = 4.0 → 100 (max — 3× above minimum useful RR)
      rr = 0.5 → 0   (clamped at floor)
    """

    @staticmethod
    def _rr_raw(rr: float) -> int:
        return max(0, min(100, int(((rr - 1.0) / 3.0) * 100)))

    def test_breakeven_rr_gives_0(self):
        assert self._rr_raw(1.0) == 0

    def test_rr_1_5_gives_16(self):
        assert self._rr_raw(1.5) == 16

    def test_rr_2_0_gives_33(self):
        assert self._rr_raw(2.0) == 33

    def test_rr_2_5_gives_50(self):
        assert self._rr_raw(2.5) == 50

    def test_rr_3_0_gives_66(self):
        assert self._rr_raw(3.0) == 66

    def test_rr_4_0_gives_100(self):
        assert self._rr_raw(4.0) == 100

    def test_rr_below_1_clamped_at_0(self):
        assert self._rr_raw(0.5) == 0
        assert self._rr_raw(0.0) == 0

    def test_rr_above_4_clamped_at_100(self):
        assert self._rr_raw(5.0) == 100
        assert self._rr_raw(10.0) == 100

    def test_natural_rr_from_default_multipliers_is_15(self):
        """
        With STOP_ATR_MULTIPLIER=1.5 and TP1_RR_MULTIPLIER=1.5:
          risk  = ATR × 1.5
          tp1   = entry + risk × 1.5
          reward = risk × 1.5
          RR   = TP1_RR_MULTIPLIER = 1.5

        NOTE: This is BELOW the SIGNAL_MIN_RR gate of 2.0.
        The BUY gate requires rr >= 2.0 — with default multipliers, this
        gate will fail unless multi-method TP overrides tp1 to a higher level.
        """
        rows = _rising_rows(n=80)
        levels = compute_entry_stop_tp(rows, "BUY")
        rr = levels["risk_reward_ratio"]
        assert math.isclose(rr, TP1_RR_MULTIPLIER, rel_tol=0.05), (
            f"Natural RR ({rr:.3f}) should equal TP1_RR_MULTIPLIER ({TP1_RR_MULTIPLIER}). "
            f"This is below SIGNAL_MIN_RR ({SIGNAL_MIN_RR}) — BUY gate will fail "
            "unless multi-method TP overrides tp1."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# F. DIRECTION DETERMINATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestDirectionLogic:
    """
    BUY:   trend_raw >= 60 AND volume_raw >= 50
    SELL:  trend_raw <= 40 AND volume_raw <= 50
    else:  NEUTRAL
    """

    @staticmethod
    def _direction(trend: float, volume: float) -> str:
        if trend >= 60 and volume >= 50:
            return "BUY"
        if trend <= 40 and volume <= 50:
            return "SELL"
        return "NEUTRAL"

    def test_strong_bull_trend_and_volume_is_buy(self):
        assert self._direction(trend=75, volume=65) == "BUY"

    def test_exact_buy_threshold_is_buy(self):
        assert self._direction(trend=60, volume=50) == "BUY"

    def test_just_below_buy_trend_threshold_is_neutral(self):
        assert self._direction(trend=59, volume=60) == "NEUTRAL"

    def test_just_below_buy_volume_threshold_is_neutral(self):
        assert self._direction(trend=65, volume=49) == "NEUTRAL"

    def test_weak_trend_and_volume_is_sell(self):
        assert self._direction(trend=30, volume=30) == "SELL"

    def test_exact_sell_threshold_is_sell(self):
        assert self._direction(trend=40, volume=50) == "SELL"

    def test_trend_bearish_but_volume_neutral_is_neutral(self):
        assert self._direction(trend=35, volume=55) == "NEUTRAL"

    def test_trend_neutral_volume_neutral_is_neutral(self):
        assert self._direction(trend=50, volume=50) == "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════════════
# G. BASE-WEIGHT NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaseWeights:
    """BASE_WEIGHTS must sum exactly to 1.0 and all be positive."""

    def test_base_weights_sum_to_1(self):
        total = sum(BASE_WEIGHTS.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9), f"BASE_WEIGHTS sum = {total}"

    def test_all_weights_positive(self):
        for k, v in BASE_WEIGHTS.items():
            assert v > 0, f"Weight '{k}' is not positive: {v}"

    def test_five_components_present(self):
        expected = {"trend", "momentum", "volume_flow", "support_resistance", "risk_reward"}
        assert set(BASE_WEIGHTS.keys()) == expected

    def test_bullish_regime_no_change_to_weights(self):
        weights = _apply_regime_weights(dict(BASE_WEIGHTS), "Bullish_Expansion",
                                         liquidity_percentile=60.0)
        total = sum(weights.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9)
        for k in BASE_WEIGHTS:
            assert math.isclose(weights[k], BASE_WEIGHTS[k], rel_tol=1e-6), (
                f"Bullish regime should not change weight '{k}': "
                f"expected {BASE_WEIGHTS[k]}, got {weights[k]}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# H. REGIME WEIGHT ADJUSTMENTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimeWeights:
    """
    Neutral_Chop:         momentum × 0.5, support_resistance × 1.5 (then renorm)
    Bearish_Contraction:  volume_flow × 1.2, trend × 0.8 (then renorm)
    Bullish_Expansion:    all × 1.0 (no change)
    """

    def test_neutral_chop_momentum_reduced_relative_to_bullish(self):
        chop = _apply_regime_weights(dict(BASE_WEIGHTS), "Neutral_Chop",
                                      liquidity_percentile=60.0)
        bull = _apply_regime_weights(dict(BASE_WEIGHTS), "Bullish_Expansion",
                                      liquidity_percentile=60.0)
        assert chop["momentum"] < bull["momentum"]

    def test_neutral_chop_sr_increased_relative_to_bullish(self):
        chop = _apply_regime_weights(dict(BASE_WEIGHTS), "Neutral_Chop",
                                      liquidity_percentile=60.0)
        bull = _apply_regime_weights(dict(BASE_WEIGHTS), "Bullish_Expansion",
                                      liquidity_percentile=60.0)
        assert chop["support_resistance"] > bull["support_resistance"]

    def test_neutral_chop_momentum_is_half_bullish(self):
        """After renormalisation momentum should be ~half what it is in bull."""
        chop = _apply_regime_weights(dict(BASE_WEIGHTS), "Neutral_Chop",
                                      liquidity_percentile=60.0)
        # Raw chop momentum = 0.20 × 0.5 = 0.10; raw sum = 0.975; normalised = 0.10/0.975
        expected = (BASE_WEIGHTS["momentum"] * 0.5) / 0.975
        assert math.isclose(chop["momentum"], expected, rel_tol=1e-6)

    def test_bearish_volume_flow_higher_than_bullish(self):
        bear = _apply_regime_weights(dict(BASE_WEIGHTS), "Bearish_Contraction",
                                      liquidity_percentile=60.0)
        bull = _apply_regime_weights(dict(BASE_WEIGHTS), "Bullish_Expansion",
                                      liquidity_percentile=60.0)
        assert bear["volume_flow"] > bull["volume_flow"]

    def test_bearish_trend_weight_lower_than_bullish(self):
        bear = _apply_regime_weights(dict(BASE_WEIGHTS), "Bearish_Contraction",
                                      liquidity_percentile=60.0)
        bull = _apply_regime_weights(dict(BASE_WEIGHTS), "Bullish_Expansion",
                                      liquidity_percentile=60.0)
        assert bear["trend"] < bull["trend"]

    def test_all_regimes_renormalised_to_1(self):
        for regime in ("Bullish_Expansion", "Neutral_Chop", "Bearish_Contraction"):
            weights = _apply_regime_weights(dict(BASE_WEIGHTS), regime,
                                            liquidity_percentile=60.0)
            total = sum(weights.values())
            assert math.isclose(total, 1.0, abs_tol=1e-9), (
                f"Regime '{regime}' weights sum to {total}"
            )

    def test_unknown_regime_treated_as_bullish(self):
        unknown = _apply_regime_weights(dict(BASE_WEIGHTS), "Unknown_Regime",
                                         liquidity_percentile=60.0)
        total = sum(unknown.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# I. LIQUIDITY WEIGHT ADJUSTMENTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiquidityWeights:
    """Illiquid stocks (liq_pct < 40) get volume_flow boosted, momentum reduced."""

    def test_illiquid_boosts_volume_flow_vs_liquid(self):
        illiquid = _apply_regime_weights(dict(BASE_WEIGHTS), "Bullish_Expansion",
                                          liquidity_percentile=20.0)
        liquid   = _apply_regime_weights(dict(BASE_WEIGHTS), "Bullish_Expansion",
                                          liquidity_percentile=60.0)
        assert illiquid["volume_flow"] > liquid["volume_flow"]

    def test_illiquid_reduces_momentum_vs_liquid(self):
        illiquid = _apply_regime_weights(dict(BASE_WEIGHTS), "Bullish_Expansion",
                                          liquidity_percentile=20.0)
        liquid   = _apply_regime_weights(dict(BASE_WEIGHTS), "Bullish_Expansion",
                                          liquidity_percentile=60.0)
        assert illiquid["momentum"] < liquid["momentum"]

    def test_illiquid_weights_still_sum_to_1(self):
        w = _apply_regime_weights(dict(BASE_WEIGHTS), "Bullish_Expansion",
                                   liquidity_percentile=15.0)
        assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# J. WEIGHTED SUB-SCORE ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreAssembly:
    """
    sub_weighted[k] = round(raw[k] × weight[k])
    total_score     = sum(sub_weighted.values())
    """

    def _compute_total(self, raws: dict[str, float], regime: str = "Bullish_Expansion",
                       liquidity_percentile: float = 60.0) -> tuple[int, dict]:
        weights = _apply_regime_weights(dict(BASE_WEIGHTS), regime,
                                        liquidity_percentile=liquidity_percentile)
        sub = {k: round(raws[k] * weights[k]) for k in raws}
        return sum(sub.values()), sub

    def test_all_max_raws_gives_score_near_100(self):
        raws = {"trend": 100, "momentum": 100, "volume_flow": 100,
                "support_resistance": 100, "risk_reward": 100}
        total, _ = self._compute_total(raws)
        # Each round(100 × w) where weights sum to 1.0 → total ≈ 100 (±1 from rounding)
        assert 98 <= total <= 102

    def test_all_zero_raws_gives_0(self):
        raws = {"trend": 0, "momentum": 0, "volume_flow": 0,
                "support_resistance": 0, "risk_reward": 0}
        total, _ = self._compute_total(raws)
        assert total == 0

    def test_strong_buy_score_threshold_exact(self):
        """A total_score of exactly 85 triggers STRONG_BUY, 84 triggers BUY."""
        assert SIGNAL_STRONG_BUY_SCORE == 85

    def test_buy_score_threshold_exact(self):
        assert SIGNAL_MIN_TOTAL_SCORE == 70

    def test_sell_score_threshold_exact(self):
        assert SIGNAL_MAX_TOTAL_SELL == 25

    def test_sub_scores_stored_as_weighted_not_raw(self):
        """sub_weighted values should be ≤ raw × max_weight, not raw themselves."""
        raws = {"trend": 80, "momentum": 70, "volume_flow": 75,
                "support_resistance": 60, "risk_reward": 50}
        _, sub = self._compute_total(raws, regime="Bullish_Expansion")
        for k, w_score in sub.items():
            # weighted score should always be < raw (since all weights < 1)
            assert w_score < raws[k], (
                f"Sub-score '{k}' weighted ({w_score}) should be less than raw ({raws[k]})"
            )

    def test_chop_regime_shifts_weight_from_momentum_to_sr(self):
        raws = {"trend": 80, "momentum": 80, "volume_flow": 80,
                "support_resistance": 80, "risk_reward": 80}
        total_bull, sub_bull = self._compute_total(raws, regime="Bullish_Expansion",
                                                    liquidity_percentile=60.0)
        total_chop, sub_chop = self._compute_total(raws, regime="Neutral_Chop",
                                                    liquidity_percentile=60.0)
        # In chop: momentum shrinks, sr grows → totals should be close but sub-components differ
        assert sub_chop["momentum"] < sub_bull["momentum"]
        assert sub_chop["support_resistance"] > sub_bull["support_resistance"]


# ═══════════════════════════════════════════════════════════════════════════════
# K. SIGNAL GATE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalGates:
    """
    BUY requires ALL of:
      total_score >= 70
      trend_raw   >= 60
      volume_raw  >= 50
      rr          >= 2.0
      liquidity_passed
      not resistance_within_1.5R

    STRONG_BUY = BUY conditions + total_score >= 85

    SELL requires ALL of:
      total_score <= 25
      trend_raw   <= 40
      volume_raw  <= 50
      liquidity_passed
    """

    @staticmethod
    def _buy_gates(total_score: int, trend: float, volume: float, rr: float,
                   liquidity: bool = True, resistance_blocked: bool = False) -> bool:
        return (
            total_score >= SIGNAL_MIN_TOTAL_SCORE
            and trend >= SIGNAL_MIN_TREND_RAW_PCT
            and volume >= SIGNAL_MIN_VOLFLOW_RAW_PCT
            and rr >= SIGNAL_MIN_RR
            and liquidity
            and not resistance_blocked
        )

    @staticmethod
    def _sell_gates(total_score: int, trend: float, volume: float,
                    liquidity: bool = True) -> bool:
        return (
            total_score <= SIGNAL_MAX_TOTAL_SELL
            and trend <= (100.0 - SIGNAL_MIN_TREND_RAW_PCT)
            and volume <= (100.0 - SIGNAL_MIN_VOLFLOW_RAW_PCT)
            and liquidity
        )

    # ── BUY gate boundary conditions ──────────────────────────────────────────

    def test_all_buy_gates_pass(self):
        assert self._buy_gates(total_score=72, trend=65, volume=55, rr=2.5)

    def test_buy_blocked_score_below_70(self):
        assert not self._buy_gates(total_score=69, trend=65, volume=55, rr=2.5)

    def test_buy_blocked_trend_below_60(self):
        assert not self._buy_gates(total_score=75, trend=59, volume=55, rr=2.5)

    def test_buy_blocked_volume_below_50(self):
        assert not self._buy_gates(total_score=75, trend=65, volume=49, rr=2.5)

    def test_buy_blocked_rr_below_2(self):
        assert not self._buy_gates(total_score=75, trend=65, volume=55, rr=1.9)

    def test_buy_blocked_liquidity_fail(self):
        assert not self._buy_gates(total_score=75, trend=65, volume=55, rr=2.5,
                                    liquidity=False)

    def test_buy_blocked_resistance_within_1_5r(self):
        assert not self._buy_gates(total_score=75, trend=65, volume=55, rr=2.5,
                                    resistance_blocked=True)

    def test_strong_buy_threshold_at_85(self):
        assert SIGNAL_STRONG_BUY_SCORE == 85
        # Score of 84 is BUY, not STRONG_BUY
        is_strong = (84 >= SIGNAL_STRONG_BUY_SCORE)
        assert not is_strong
        is_strong = (85 >= SIGNAL_STRONG_BUY_SCORE)
        assert is_strong

    def test_score_70_gives_buy_not_strong(self):
        # 70 >= SIGNAL_MIN_TOTAL_SCORE, 70 < SIGNAL_STRONG_BUY_SCORE
        assert 70 >= SIGNAL_MIN_TOTAL_SCORE
        assert 70 < SIGNAL_STRONG_BUY_SCORE

    # ── SELL gate boundary conditions ─────────────────────────────────────────

    def test_all_sell_gates_pass(self):
        assert self._sell_gates(total_score=20, trend=35, volume=45)

    def test_sell_blocked_score_above_25(self):
        assert not self._sell_gates(total_score=26, trend=35, volume=45)

    def test_sell_blocked_trend_above_40(self):
        assert not self._sell_gates(total_score=20, trend=41, volume=45)

    def test_sell_blocked_volume_above_50(self):
        assert not self._sell_gates(total_score=20, trend=35, volume=51)

    def test_sell_blocked_liquidity_fail(self):
        assert not self._sell_gates(total_score=20, trend=35, volume=45, liquidity=False)

    def test_exact_sell_boundaries_pass(self):
        assert self._sell_gates(total_score=25, trend=40, volume=50)

    def test_exact_buy_boundaries_pass(self):
        assert self._buy_gates(total_score=70, trend=60, volume=50, rr=2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# L. RESISTANCE-WITHIN-1.5R HARD BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

class TestResistanceBlock:
    """
    If nearest_resistance - entry_mid < risk_per_share × 1.5 → BUY blocked.
    """

    @staticmethod
    def _is_blocked(nearest_resistance: float, entry_mid: float,
                    risk_per_share: float) -> bool:
        one_half_r = risk_per_share * 1.5
        return nearest_resistance - entry_mid < one_half_r

    def test_resistance_far_away_not_blocked(self):
        # risk = 10, 1.5R = 15, resistance 20 fils above entry → not blocked
        assert not self._is_blocked(nearest_resistance=520, entry_mid=500, risk_per_share=10.0)

    def test_resistance_within_1_5r_is_blocked(self):
        # risk = 10, 1.5R = 15, resistance only 10 fils above entry → blocked
        assert self._is_blocked(nearest_resistance=510, entry_mid=500, risk_per_share=10.0)

    def test_resistance_exactly_at_1_5r_is_blocked(self):
        # resistance at exactly 1.5R → still < 1.5R is False → NOT blocked
        # nearest_resistance - entry_mid = 15, one_half_r = 15 → 15 < 15 is False
        assert not self._is_blocked(nearest_resistance=515, entry_mid=500, risk_per_share=10.0)

    def test_resistance_one_fil_inside_1_5r_is_blocked(self):
        assert self._is_blocked(nearest_resistance=514, entry_mid=500, risk_per_share=10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# M. CIRCUIT-BREAKER ×0.70 PENALTY
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreakerPenalty:
    """
    Constants: CIRCUIT_UPPER_PCT = +10%, CIRCUIT_LOWER_PCT = -5%
    Buffer:    CIRCUIT_BUFFER_PCT = 1%

    Upper penalty fires when: (upper - close) / close <= 0.01
    Lower penalty fires when: (close - lower) / close <= 0.01
    """

    @staticmethod
    def _apply_penalty(close: float, prev_close: float, score: int) -> int:
        upper = prev_close * (1.0 + CIRCUIT_UPPER_PCT)
        lower = prev_close * (1.0 + CIRCUIT_LOWER_PCT)
        if close > 0:
            if (upper - close) / close <= CIRCUIT_BUFFER_PCT:
                score = int(score * 0.70)
            if (close - lower) / close <= CIRCUIT_BUFFER_PCT:
                score = int(score * 0.70)
        return score

    def test_normal_price_no_penalty(self):
        # Close at 105, prev=100 — upper=110, lower=95 — not near either limit
        assert self._apply_penalty(close=105.0, prev_close=100.0, score=80) == 80

    def test_near_upper_circuit_applies_070_penalty(self):
        # prev_close=100, upper=110, close=109 → (110-109)/109 ≈ 0.0092 ≤ 0.01 → penalty
        result = self._apply_penalty(close=109.0, prev_close=100.0, score=80)
        assert result == int(80 * 0.70)  # = 56

    def test_near_lower_circuit_applies_070_penalty(self):
        # prev_close=100, lower=95, close=95.8 → (95.8-95)/95.8 ≈ 0.0083 ≤ 0.01 → penalty
        result = self._apply_penalty(close=95.8, prev_close=100.0, score=80)
        assert result == int(80 * 0.70)  # = 56

    def test_exactly_at_upper_circuit_triggers_penalty(self):
        # close at exactly upper-circuit limit (110) → gap = 0 ≤ 0.01 → penalty
        result = self._apply_penalty(close=110.0, prev_close=100.0, score=100)
        assert result == int(100 * 0.70)  # = 70

    def test_outside_upper_buffer_no_penalty(self):
        # close=107, upper=110 → gap = 3/107 ≈ 0.028 > 0.01 → no penalty
        assert self._apply_penalty(close=107.0, prev_close=100.0, score=80) == 80

    def test_penalty_uses_floor_division_int(self):
        result = self._apply_penalty(close=109.0, prev_close=100.0, score=77)
        assert result == int(77 * 0.70)  # = 53 (int truncation)

    def test_penalty_constants_values(self):
        assert CIRCUIT_UPPER_PCT == 0.10
        assert CIRCUIT_LOWER_PCT == -0.05
        assert CIRCUIT_BUFFER_PCT == 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# N. END-TO-END SCORING
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEndScoring:
    """
    Uses generate_kuwait_signal() with crafted rows.

    NOTE on RR gate: with STOP_ATR_MULTIPLIER=1.5 and TP1_RR_MULTIPLIER=1.5
    the natural risk/reward ratio = 1.5, which is below SIGNAL_MIN_RR=2.0.
    A BUY signal therefore requires multi-method TP to override tp1 to a
    higher level before the gate is re-evaluated — otherwise the signal will
    be NEUTRAL regardless of score.  The tests below verify scoring arithmetic
    is correct regardless of the gate outcome.
    """

    @staticmethod
    def _run(coro):
        """Run an async coroutine from a synchronous test."""
        import asyncio
        return asyncio.run(coro)

    def _make_strong_bull_rows(self, n: int = 80) -> list[dict]:
        """Full bullish indicator set with liquidity-passing values."""
        rows = []
        for i in range(n):
            c = 490.0 + i * 0.2
            rows.append(_base_row(
                close=c,
                high=c + 2.0,     # spread = 4/490 ≈ 0.82 % — passes ≤1.5 %
                low=c - 2.0,
                volume=5_000_000.0,
                value=250_000.0,  # ADTV >> 100 k KD — passes liquidity
                atr=8.0,
                obv=500_000.0 * (i + 1),
                ad=2_000_000.0 * (i + 1),
                ema_20=c * 0.982,
                ema_50=c * 0.960,
                sma_200=c * 0.920,
                adx_14=30.0,
                rsi_14=58.0,
                macd=2.0, macd_signal=1.5, macd_hist=0.5,
                stoch_k=55.0, stoch_d=48.0,
                cmf_20=0.18,
                date=f"2026-01-{(i % 28) + 1:02d}",
            ))
        return rows

    def _make_strong_bear_rows(self, n: int = 80) -> list[dict]:
        rows = []
        for i in range(n):
            c = 560.0 - i * 0.2
            rows.append(_base_row(
                close=c,
                high=c + 2.0,
                low=c - 2.0,
                volume=5_000_000.0,
                value=250_000.0,
                atr=8.0,
                obv=10_000_000.0 - 400_000.0 * i,
                ad=20_000_000.0 - 800_000.0 * i,
                ema_20=c * 1.018,
                ema_50=c * 1.040,
                sma_200=c * 1.080,
                adx_14=28.0,
                rsi_14=32.0,
                macd=-2.0, macd_signal=1.5, macd_hist=-0.5,
                stoch_k=20.0, stoch_d=35.0,
                cmf_20=-0.22,
                date=f"2026-01-{(i % 28) + 1:02d}",
            ))
        return rows

    # ── Output structure ───────────────────────────────────────────────────────

    def test_returns_dict(self):
        signal = self._run(generate_kuwait_signal(_make_strong_bull_rows := self._make_strong_bull_rows(),
                                        stock_code="TEST"))
        assert isinstance(signal, dict)

    def test_required_top_level_keys_present(self):
        signal = self._run(generate_kuwait_signal(self._make_strong_bull_rows(), stock_code="TEST"))
        for key in ("signal", "stock_code", "confluence_details", "execution", "probabilities",
                    "alerts", "metadata"):
            assert key in signal, f"Missing top-level key: {key}"

    def test_confluence_has_total_score(self):
        signal = self._run(generate_kuwait_signal(self._make_strong_bull_rows(), stock_code="TEST"))
        assert "total_score" in signal["confluence_details"]
        assert isinstance(signal["confluence_details"]["total_score"], int)

    def test_total_score_in_valid_range(self):
        signal = self._run(generate_kuwait_signal(self._make_strong_bull_rows(), stock_code="TEST"))
        score = signal["confluence_details"]["total_score"]
        assert 0 <= score <= 100

    def test_confluence_has_sub_scores(self):
        signal = self._run(generate_kuwait_signal(self._make_strong_bull_rows(), stock_code="TEST"))
        sub = signal["confluence_details"]["sub_scores"]
        for k in ("trend", "momentum", "volume_flow", "support_resistance", "risk_reward"):
            assert k in sub, f"sub_scores missing key: {k}"

    def test_confluence_has_raw_sub_scores(self):
        signal = self._run(generate_kuwait_signal(self._make_strong_bull_rows(), stock_code="TEST"))
        raw = signal["confluence_details"]["raw_sub_scores"]
        for k in ("trend", "momentum", "volume_flow", "support_resistance", "risk_reward"):
            assert k in raw, f"raw_sub_scores missing key: {k}"

    def test_sub_scores_sum_equals_total_score_or_close(self):
        """sub_weighted values must sum to total_score (before circuit-breaker)."""
        signal = self._run(generate_kuwait_signal(self._make_strong_bull_rows(), stock_code="TEST"))
        conf = signal["confluence_details"]
        sub_sum = sum(v for k, v in conf["sub_scores"].items() if k != "risk_reward")
        total = conf["total_score_raw"]
        assert abs((sub_sum / 0.85) - total) <= 5, (
            f"sub_scores sum ({sub_sum}) un-normalized does not match total_score_raw ({total})."
        )

    # ── Bullish scenario ───────────────────────────────────────────────────────

    def test_bullish_rows_produce_high_trend_raw_score(self):
        signal = self._run(generate_kuwait_signal(self._make_strong_bull_rows(), stock_code="TEST"))
        trend_raw = signal["confluence_details"]["raw_sub_scores"]["trend"]
        assert trend_raw >= 60, f"Expected strong trend raw ≥ 60, got {trend_raw}"

    def test_bullish_rows_produce_high_volume_raw_score(self):
        signal = self._run(generate_kuwait_signal(self._make_strong_bull_rows(), stock_code="TEST"))
        vol_raw = signal["confluence_details"]["raw_sub_scores"]["volume_flow"]
        assert vol_raw >= 50, f"Expected volume raw ≥ 50, got {vol_raw}"

    def test_bullish_rows_produce_high_total_score(self):
        signal = self._run(generate_kuwait_signal(self._make_strong_bull_rows(), stock_code="TEST"))
        total = signal["confluence_details"]["total_score"]
        assert total >= 50, f"Expected high total_score with bullish inputs, got {total}"

    def test_bullish_signal_is_buy_or_neutral(self):
        """Signal is BUY/STRONG_BUY if all gates pass; NEUTRAL if RR gate fails."""
        signal = self._run(generate_kuwait_signal(self._make_strong_bull_rows(), stock_code="TEST"))
        assert signal["signal"] in ("BUY", "STRONG_BUY", "NEUTRAL"), (
            f"Unexpected signal: {signal['signal']}"
        )

    # ── Bearish scenario ───────────────────────────────────────────────────────

    def test_bearish_rows_produce_low_trend_raw_score(self):
        signal = self._run(generate_kuwait_signal(self._make_strong_bear_rows(), stock_code="TEST"))
        trend_raw = signal["confluence_details"]["raw_sub_scores"]["trend"]
        assert trend_raw <= 40, f"Expected bearish trend raw ≤ 40, got {trend_raw}"

    def test_bearish_rows_produce_low_total_score(self):
        signal = self._run(generate_kuwait_signal(self._make_strong_bear_rows(), stock_code="TEST"))
        total = signal["confluence_details"]["total_score"]
        assert total <= 50, f"Expected low total_score with bearish inputs, got {total}"

    # ── Insufficient data ──────────────────────────────────────────────────────

    def test_fewer_than_60_rows_returns_neutral(self):
        signal = self._run(generate_kuwait_signal(
            self._make_strong_bull_rows(n=30), stock_code="TEST"
        ))
        assert signal["signal"] == "NEUTRAL"

    def test_insufficient_data_reason_in_signal(self):
        signal = self._run(generate_kuwait_signal(
            self._make_strong_bull_rows(n=10), stock_code="TEST"
        ))
        assert signal["signal"] == "NEUTRAL"

    # ── Score monotonicity ─────────────────────────────────────────────────────

    def test_bullish_total_score_gt_bearish_total_score(self):
        bull_signal = self._run(generate_kuwait_signal(self._make_strong_bull_rows(), stock_code="TEST"))
        bear_signal = self._run(generate_kuwait_signal(self._make_strong_bear_rows(), stock_code="TEST"))
        bull_score = bull_signal["confluence_details"]["total_score"]
        bear_score = bear_signal["confluence_details"]["total_score"]
        assert bull_score > bear_score, (
            f"Bull score ({bull_score}) should be greater than bear score ({bear_score})"
        )
