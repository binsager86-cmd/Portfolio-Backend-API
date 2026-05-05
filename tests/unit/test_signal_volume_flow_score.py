"""Unit tests for the Kuwait Signal Engine — Volume/Flow Score.

Validates:
  - CMF (35%) primary flow signal
  - OBV slope (25%) trend alignment
  - RVOL (25%) breakout confirmation (replaces A/D Line)
  - Auction intensity (15%) confirmation
  - Details dict contains 'rvol_pts' key (A/D Line removed)
  - Score always in [0, 100]
"""
from __future__ import annotations

from app.services.signal_engine.models.technical.volume_flow_score import compute_volume_flow_score


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(close: float = 500.0, obv: float = 1_000_000.0,
         cmf: float = 0.15, vol: float = 2_000_000.0) -> dict:
    return {
        "close": close, "open": close - 2, "high": close + 5, "low": close - 5,
        "volume": vol,
        "obv": obv,
        "cmf_20": cmf,
    }


def _rows_rising_obv(n: int = 25) -> list[dict]:
    """25 rows so RVOL has enough data; OBV strongly rising."""
    return [_row(obv=500_000.0 * (i + 1)) for i in range(n)]


def _rows_falling_obv(n: int = 25) -> list[dict]:
    """25 rows; OBV strongly falling."""
    return [_row(obv=12_000_000.0 - 500_000.0 * i) for i in range(n)]


# ── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_tuple(self):
        r, d = compute_volume_flow_score(_rows_rising_obv(), auction_intensity=1.0)
        assert isinstance(r, int) and isinstance(d, dict)

    def test_score_in_range(self):
        score, _ = compute_volume_flow_score(_rows_rising_obv(), auction_intensity=1.0)
        assert 0 <= score <= 100

    def test_details_has_rvol_key(self):
        _, d = compute_volume_flow_score(_rows_rising_obv(), auction_intensity=1.0)
        assert "rvol_pts" in d
        assert "rvol_desc" in d

    def test_empty_rows_returns_neutral(self):
        score, _ = compute_volume_flow_score([], auction_intensity=1.0)
        assert 0 <= score <= 100


# ── OBV ───────────────────────────────────────────────────────────────────────

class TestOBV:
    def test_strongly_rising_obv_gives_max_pts(self):
        _, d = compute_volume_flow_score(_rows_rising_obv(), auction_intensity=1.0)
        assert d["obv_pts"] == 25

    def test_falling_obv_gives_low_pts(self):
        _, d = compute_volume_flow_score(_rows_falling_obv(), auction_intensity=1.0)
        assert d["obv_pts"] <= 5

    def test_missing_obv_gives_neutral(self):
        rows = _rows_rising_obv()
        for r in rows:
            r["obv"] = None
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["obv_pts"] == 12


# ── CMF ───────────────────────────────────────────────────────────────────────

class TestCMF:
    def test_strong_accumulation_cmf_gives_max(self):
        rows = _rows_rising_obv()
        for r in rows:
            r["cmf_20"] = 0.25
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["cmf_pts"] == 35

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
        assert d["cmf_pts"] == 14


# ── RVOL ──────────────────────────────────────────────────────────────────────

class TestRVOL:
    def test_high_breakout_volume_gives_max(self):
        """Current volume 2.4x median → 25 pts."""
        base_vol = 1_000_000.0
        rows = [_row(vol=base_vol) for _ in range(24)]
        rows.append(_row(vol=base_vol * 2.4))  # current day breakout
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["rvol_pts"] == 25

    def test_normal_volume_gives_mid_pts(self):
        """Current volume equal to median → 10 pts (0.8x–1.2x range)."""
        base_vol = 1_000_000.0
        rows = [_row(vol=base_vol) for _ in range(25)]
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["rvol_pts"] == 10

    def test_thin_volume_trap_gives_zero(self):
        """Current volume 0.4x median → 0 pts."""
        base_vol = 1_000_000.0
        rows = [_row(vol=base_vol) for _ in range(24)]
        rows.append(_row(vol=base_vol * 0.4))  # thin current volume
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["rvol_pts"] == 0

    def test_insufficient_data_gives_neutral(self):
        """< 21 rows → 12 pts neutral."""
        rows = [_row() for _ in range(15)]
        _, d = compute_volume_flow_score(rows, auction_intensity=1.0)
        assert d["rvol_pts"] == 12


# ── Auction ───────────────────────────────────────────────────────────────────

class TestAuction:
    def test_high_intensity_gives_max_auction_pts(self):
        _, d = compute_volume_flow_score(_rows_rising_obv(), auction_intensity=2.0)
        assert d["auction_pts"] == 15

    def test_low_intensity_gives_low_pts(self):
        _, d = compute_volume_flow_score(_rows_rising_obv(), auction_intensity=0.3)
        assert d["auction_pts"] == 3

    def test_score_capped_at_100(self):
        """CMF=35 + OBV=25 + RVOL=25 + Auction=15 = 100 exactly."""
        base_vol = 1_000_000.0
        rows = [_row(obv=500_000.0 * (i + 1), cmf=0.30, vol=base_vol) for i in range(24)]
        rows.append(_row(obv=500_000.0 * 25, cmf=0.30, vol=base_vol * 2.5))
        score, _ = compute_volume_flow_score(rows, auction_intensity=2.0)
        assert score == 100
