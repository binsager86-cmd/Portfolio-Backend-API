"""
Regression tests for Eagle Eye Behavioral DNA move statistics.
"""
from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd
import pytest

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from app.services.eagle_eye.dna_extractor import extract_dna
from app.services.eagle_eye.move_detector import MoveEvent
from app.services.eagle_eye.recorder import ForensicSnapshot


def _snapshot(
    event_id: str,
    *,
    threshold_pct: float,
    gain_pct: float,
    signals: list[str],
    is_fakeout: bool = False,
    failed_at_pct: float | None = None,
) -> ForensicSnapshot:
    event = MoveEvent(
        ticker="TEST",
        event_id=event_id,
        direction="UP",
        threshold_pct=threshold_pct,
        start_date=date(2024, 1, 1),
        acceleration_date=date(2024, 1, 10),
        peak_date=date(2024, 2, 1),
        start_price=1.0,
        acceleration_price=1.05,
        peak_price=1.0 + gain_pct / 100,
        gain_pct=gain_pct,
        duration_days=20,
        days_consolidating_before=12,
        pre_move_volatility_pct=5.0,
        is_fakeout=is_fakeout,
        failed_at_pct=failed_at_pct,
    )
    sequence = [
        {
            "signal": signal,
            "days_before_acceleration": 5 + idx,
            "fired_on": f"2024-01-{idx + 1:02d}",
        }
        for idx, signal in enumerate(signals)
    ]
    return ForensicSnapshot(event=event, indicator_snapshots={}, signal_sequence=sequence)


def _indicator_row(*, close: float, setup_active: bool) -> dict:
    return {
        "close": close,
        "volume": 1000.0,
        "obv_slope_60": 1.0 if setup_active else -1.0,
        "accumulation_score": 80.0 if setup_active else 40.0,
        "supertrend": 1 if setup_active else 0,
    }


def test_extract_dna_uses_forward_setup_base_rates():
    snapshots = [
        _snapshot(
            "e1",
            threshold_pct=10.0,
            gain_pct=12.0,
            signals=[
                "obv_60d_slope_strongly_positive",
                "accumulation_above_75",
                "supertrend_bullish",
            ],
        ),
        _snapshot(
            "e2",
            threshold_pct=25.0,
            gain_pct=30.0,
            signals=[
                "obv_60d_slope_strongly_positive",
                "accumulation_above_75",
                "supertrend_bullish",
            ],
        ),
        _snapshot(
            "e3",
            threshold_pct=50.0,
            gain_pct=60.0,
            signals=[
                "obv_60d_slope_strongly_positive",
                "accumulation_above_75",
                "supertrend_bullish",
            ],
        ),
    ]

    indicators_df = pd.DataFrame(
        [
            _indicator_row(close=100.0, setup_active=True),
            _indicator_row(close=105.0, setup_active=False),
            _indicator_row(close=130.0, setup_active=False),
            _indicator_row(close=100.0, setup_active=True),
            _indicator_row(close=108.0, setup_active=False),
            _indicator_row(close=112.0, setup_active=False),
            _indicator_row(close=100.0, setup_active=True),
            _indicator_row(close=95.0, setup_active=False),
            _indicator_row(close=100.0, setup_active=True),
            _indicator_row(close=120.0, setup_active=False),
            _indicator_row(close=160.0, setup_active=False),
            _indicator_row(close=150.0, setup_active=True),
        ],
        index=pd.date_range("2024-01-01", periods=12, freq="D"),
    )

    dna = extract_dna(
        "TEST",
        snapshots,
        [],
        indicators_df=indicators_df,
        horizon_days=2,
        min_setup_occurrences=4,
    )

    assert dna is not None
    assert dna.history_status == "ok"
    assert dna.total_events_studied == 4
    assert set(dna.setup_signals) == {
        "accumulation_above_75",
        "obv_60d_slope_strongly_positive",
        "supertrend_bullish",
    }

    profiles = {int(p.threshold_pct): p for p in dna.profiles_by_threshold}
    assert profiles[10].occurrences == 3
    assert profiles[15].occurrences == 2
    assert profiles[25].occurrences == 2
    assert profiles[50].occurrences == 1
    assert profiles[100].occurrences == 0

    success_rates = [profiles[t].success_rate for t in (10, 15, 25, 50, 100)]
    assert success_rates == pytest.approx([75.0, 50.0, 50.0, 25.0, 0.0], abs=0.05)
    assert success_rates == sorted(success_rates, reverse=True)

    assert profiles[10].sample_count == 4
    assert profiles[10].avg_gain_all_pct == pytest.approx(25.5)
    assert profiles[10].avg_gain_on_hits_pct == pytest.approx(34.0)
    assert profiles[50].avg_gain_on_hits_pct == pytest.approx(60.0)
    assert profiles[100].avg_gain_on_hits_pct is None

    sig_stats = {s.signal: s for s in dna.signal_stats}
    assert sig_stats["accumulation_above_75"].fired_count == 4
    assert sig_stats["accumulation_above_75"].total_setups == 4
    assert sig_stats["accumulation_above_75"].presence_pct == pytest.approx(100.0)


def test_extract_dna_marks_insufficient_history_when_setup_count_is_too_small():
    snapshots = [
        _snapshot(
            "e1",
            threshold_pct=10.0,
            gain_pct=12.0,
            signals=["obv_60d_slope_strongly_positive", "accumulation_above_75"],
        ),
        _snapshot(
            "e2",
            threshold_pct=25.0,
            gain_pct=30.0,
            signals=["obv_60d_slope_strongly_positive", "accumulation_above_75"],
        ),
        _snapshot(
            "e3",
            threshold_pct=50.0,
            gain_pct=60.0,
            signals=["obv_60d_slope_strongly_positive", "accumulation_above_75"],
        ),
    ]

    indicators_df = pd.DataFrame(
        [
            _indicator_row(close=100.0, setup_active=True),
            _indicator_row(close=110.0, setup_active=False),
            _indicator_row(close=100.0, setup_active=True),
            _indicator_row(close=102.0, setup_active=False),
            _indicator_row(close=100.0, setup_active=True),
            _indicator_row(close=104.0, setup_active=False),
            _indicator_row(close=103.0, setup_active=True),
        ],
        index=pd.date_range("2024-02-01", periods=7, freq="D"),
    )

    dna = extract_dna(
        "TEST",
        snapshots,
        [],
        indicators_df=indicators_df,
        horizon_days=1,
        min_setup_occurrences=4,
    )

    assert dna is not None
    assert dna.history_status == "INSUFFICIENT_HISTORY"
    assert dna.total_events_studied == 3
    assert dna.profiles_by_threshold == []


def test_extract_dna_builds_multi_window_profiles_and_examples():
    snapshots = [
        _snapshot(
            "e1",
            threshold_pct=10.0,
            gain_pct=12.0,
            signals=[
                "obv_60d_slope_strongly_positive",
                "accumulation_above_75",
                "supertrend_bullish",
            ],
        ),
        _snapshot(
            "e2",
            threshold_pct=25.0,
            gain_pct=30.0,
            signals=[
                "obv_60d_slope_strongly_positive",
                "accumulation_above_75",
                "supertrend_bullish",
            ],
        ),
        _snapshot(
            "e3",
            threshold_pct=50.0,
            gain_pct=60.0,
            signals=[
                "obv_60d_slope_strongly_positive",
                "accumulation_above_75",
                "supertrend_bullish",
            ],
        ),
    ]

    indicators_df = pd.DataFrame(
        [
            _indicator_row(close=100.0, setup_active=True),
            _indicator_row(close=105.0, setup_active=False),
            _indicator_row(close=110.0, setup_active=False),
            _indicator_row(close=130.0, setup_active=False),
            _indicator_row(close=100.0, setup_active=True),
            _indicator_row(close=108.0, setup_active=False),
            _indicator_row(close=112.0, setup_active=False),
            _indicator_row(close=118.0, setup_active=False),
            _indicator_row(close=100.0, setup_active=True),
            _indicator_row(close=120.0, setup_active=False),
            _indicator_row(close=140.0, setup_active=False),
            _indicator_row(close=160.0, setup_active=False),
            _indicator_row(close=150.0, setup_active=True),
            _indicator_row(close=148.0, setup_active=False),
            _indicator_row(close=152.0, setup_active=False),
            _indicator_row(close=154.0, setup_active=False),
        ],
        index=pd.date_range("2024-01-01", periods=16, freq="D"),
    )

    dna = extract_dna(
        "TEST",
        snapshots,
        [],
        indicators_df=indicators_df,
        horizon_days=2,
        window_days=(2, 4),
        min_setup_occurrences=4,
    )

    assert dna is not None
    assert dna.default_window_days == 2
    assert dna.available_window_days == [2, 4]

    window_profiles = {profile.horizon_days: profile for profile in dna.window_profiles}
    assert window_profiles[2].setup_count == 4
    assert window_profiles[2].percentages_visible is True
    assert window_profiles[4].setup_count == 4
    assert window_profiles[4].history_status == "ok"

    assert len(dna.setup_examples) == 3
    first_example = dna.setup_examples[0]
    assert first_example.bars
    assert first_example.forward_outcomes["2"].completed is True
    assert first_example.forward_outcomes["4"].horizon_days == 4
    assert first_example.observations