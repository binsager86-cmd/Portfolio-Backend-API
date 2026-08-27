"""
Regression tests for rating_engine.py stage-name handling.

The rating engine's internal lookup tables were built with two different
naming conventions for the same lifecycle stages: the raw names returned by
stage_classifier.classify_stage_with_confidence() ("MARKUP", "ACCUMULATION",
"DISTRIBUTION", "MARKDOWN", "EARLY_MARKUP", "NEUTRAL_AMBIGUOUS") and a set of
legacy alias names ("MARKUP_TRENDING", "STEALTH_ACCUMULATION",
"DISTRIBUTION_TOPPING", "MARKDOWN_DECLINE", "EARLY_BREAKOUT", "DORMANT").

Several lookup tables in compute_confidence() and compute_final_confidence()
only had the alias keys, so callers passing the raw stage name (the only
names ever actually produced by the live classifier) silently fell through
to the generic default instead of the intended stage-specific behavior.
"""
from __future__ import annotations

import os
import sys

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from app.services.eagle_eye.rating_engine import compute_confidence, compute_final_confidence

RAW_ALIAS_PAIRS = [
    ("MARKDOWN", "MARKDOWN_DECLINE"),
    ("DISTRIBUTION", "DISTRIBUTION_TOPPING"),
    ("MARKUP", "MARKUP_TRENDING"),
    ("ACCUMULATION", "STEALTH_ACCUMULATION"),
    ("EARLY_MARKUP", "EARLY_BREAKOUT"),
    ("NEUTRAL_AMBIGUOUS", "DORMANT"),
]


def test_compute_confidence_raw_and_alias_stage_names_agree():
    """The raw stage name and its legacy alias must yield identical confidence."""
    for raw, alias in RAW_ALIAS_PAIRS:
        raw_conf = compute_confidence({}, raw, dna=None)
        alias_conf = compute_confidence({}, alias, dna=None)
        assert raw_conf == alias_conf, f"{raw} vs {alias}: {raw_conf} != {alias_conf}"


def test_compute_confidence_caps_bearish_stage_via_ml_score():
    """A very high predicted gain must still be capped for a markdown/distribution stock."""
    markdown_conf = compute_confidence({}, "MARKDOWN", dna=None, ml_score=95.0)
    distribution_conf = compute_confidence({}, "DISTRIBUTION", dna=None, ml_score=95.0)
    assert markdown_conf <= 35.0
    assert distribution_conf <= 35.0


def test_compute_confidence_does_not_cap_bullish_stage_via_ml_score():
    """The bearish-stage cap must not incorrectly apply to a markup stock."""
    markup_conf = compute_confidence({}, "MARKUP", dna=None, ml_score=95.0)
    assert markup_conf > 35.0


def test_compute_confidence_caps_bearish_stage_via_ml_proba():
    markdown_conf = compute_confidence(
        {}, "MARKDOWN", dna=None, ml_proba={"buy": 0.95, "sell": 0.0, "hold": 0.05}
    )
    assert markdown_conf <= 30.0


def test_compute_final_confidence_raw_and_alias_stage_names_agree():
    for raw, alias in RAW_ALIAS_PAIRS:
        raw_conf, raw_rating = compute_final_confidence(50.0, {}, raw)
        alias_conf, alias_rating = compute_final_confidence(50.0, {}, alias)
        assert raw_conf == alias_conf, f"{raw} vs {alias}: {raw_conf} != {alias_conf}"
        assert raw_rating == alias_rating
