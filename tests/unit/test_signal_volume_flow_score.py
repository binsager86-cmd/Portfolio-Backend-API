"""Unit tests for the Kuwait Signal Engine — Volume/Flow Score.

Validates:
  - Rising OBV slope → high OBV sub-score
  - Positive CMF → high CMF sub-score
  - Rising AD Line slope → high AD sub-score
  - High auction intensity → high auction sub-score
  - Details dict contains 'ad_pts' key
  - Score always in [0, 100]
"""
from __future__ import annotations

import pytest

from app.services.signal_engine.models.technical.volume_flow_score import compute_volume_flow_score


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(close: float = 500.0, obv: float = 1_000_000.0,
         cmf: float = 0.15, ad: float = 5_000_000.0, vol: float = 2_000_000.0) -> dict:
    return {
        "close": close, "open": close - 2, "high": close + 5, "low": close - 5,
        "volume": vol,
        "obv": obv,
        "cmf_20": cmf,
        "ad_line": ad,
    }


def _rows_rising_obv(n: int = 12) -> list[dict]:
    return [_row(obv=500_000.0 * (i + 1), ad=2_000_000.0 * (i + 1)) for i in range(n)]


def _rows_falling_obv(n: int = 12) -> list[dict]:
    return [_row(obv=12_000_000.0 - 500_000.0 * i, ad=24_000_000.0 - 1_000_000.0 * i) for i in range(n)]


# ── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_tuple(self):
        r, d = compute_volume_flow_score(_rows_rising_obv(), auction_intensity=1.0)
        assert isinstance(r, int) and isinstance(d, dict)

    def test_score_in_range(self):
        score, _ = compute_volume_flow_score(_rows_rising_obv(), auction_intensity=1.0)
        assert 0 <= score <= 100

    def test_details_has_ad_key(self):
        _, d = compute_volume_flow_score(_rows_rising_obv(), auction_intensity=1.0)
        assert "ad_pts" in d
        assert "ad_desc" in d

    def test_empty_rows_returns_neutral(self):
        score, _ = compute_volume_flow_score([], auction_intensity=1.0)
        assert 0 <= score <= 100


# ── OBV ───────────────────────────────────────────────────────────────────────

class TestOBV:
    def test_strongly_rising_obv_gives_max_pts(self):
        _, d = compute_volume_flow_score(_rows_rising_obv(), auction_intensity=1.0)
        assert d["obv_pts"] == 35

    def test_falling_obv_gives_low_pts(self):
        _, d = compute_volume_flow_score(_rows_falling_obv(), auction_intensity=1.0)
        assert d["obv_pts"] <= 5

    def test_missing_obv_gives_neutral(self):
        rows = _rows_rising_obv()
        for r in rows:
            r["obv"] = None
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["obv_pts"] == 15


# ── CMF ───────────────────────────────────────────────────────────────────────

class TestCMF:
    def test_strong_accumulation_cmf_gives_max(self):
        rows = _rows_rising_obv()
        for r in rows:
            r["cmf_20"] = 0.25
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["cmf_pts"] == 25

    def test_strong_distribution_cmf_gives_zero(self):
        rows = _rows_rising_obv()
        for r in rows:
            r["cmf_20"] = -0.25
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["cmf_pts"] == 0

    def test_missing_cmf_gives_neutral(self):
        rows = _rows_rising_obv()
        rows[-1]["cmf_20"] = None
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["cmf_pts"] == 10


# ── AD Line ───────────────────────────────────────────────────────────────────

class TestADLine:
    def test_strongly_rising_ad_gives_max(self):
        _, d = compute_volume_flow_score(_rows_rising_obv(), auction_intensity=1.0)
        assert d["ad_pts"] == 20

    def test_falling_ad_gives_low_pts(self):
        _, d = compute_volume_flow_score(_rows_falling_obv(), auction_intensity=1.0)
        assert d["ad_pts"] <= 3

    def test_missing_ad_gives_neutral(self):
        rows = _rows_rising_obv()
        for r in rows:
            r["ad_line"] = None
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["ad_pts"] == 8


# ── Auction ───────────────────────────────────────────────────────────────────

class TestAuction:
    def test_high_intensity_gives_max_auction_pts(self):
        _, d = compute_volume_flow_score(_rows_rising_obv(), auction_intensity=2.0)
        assert d["auction_pts"] == 30

    def test_low_intensity_gives_low_pts(self):
        _, d = compute_volume_flow_score(_rows_rising_obv(), auction_intensity=0.3)
        assert d["auction_pts"] == 5

    def test_score_capped_at_100(self):
        rows = _rows_rising_obv()
        for r in rows:
            r["cmf_20"] = 0.30
        score, _ = compute_volume_flow_score(rows, auction_intensity=2.0)
        assert score == 100
