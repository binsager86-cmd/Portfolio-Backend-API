"""
tests/test_leakage_audit.py

Phase 1 verification — Check 1: Leakage audit framework catches known-bad features.

All 6 test cases must pass for Phase 2 to proceed.
"""
from __future__ import annotations

import textwrap
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the module under test (path-independent — run from backend-api/)
# ---------------------------------------------------------------------------
from app.services.eagle_eye.ml.leakage_audit import (
    LeakageAuditor,
    scan_source_for_leakage,
)


# ===========================================================================
# Helper factories
# ===========================================================================

def _make_df(
    n: int = 120,
    *,
    feature_train_fn=None,
    feature_test_fn=None,
    target_fn=None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a tiny feature+label DataFrame for statistical audit tests."""
    rng = np.random.default_rng(seed)
    base_date = date(2022, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n)]

    y = rng.integers(0, 2, size=n).astype(float)

    if target_fn is not None:
        y = target_fn(rng, n)

    if feature_train_fn is None and feature_test_fn is None:
        # Default: pure noise
        f = rng.standard_normal(n)
    else:
        split = int(n * 0.75)
        f = np.empty(n)
        if feature_train_fn is not None:
            f[:split] = feature_train_fn(rng, split)
        else:
            f[:split] = rng.standard_normal(split)
        if feature_test_fn is not None:
            f[split:] = feature_test_fn(rng, n - split)
        else:
            f[split:] = rng.standard_normal(n - split)

    return pd.DataFrame({"event_date": dates, "y": y, "feature": f})


# ===========================================================================
# Test 1 — Centered rolling window AST detection
# ===========================================================================

def test_centered_rolling_window_flagged():
    """
    AST scanner must detect `rolling(window=10, center=True)` and return
    at least one issue mentioning 'center=True'.
    """
    leaky_source = textwrap.dedent("""
        import pandas as pd

        def compute_bad_feature(df):
            # This looks ahead — centered window is not causal
            return df['x'].rolling(window=10, center=True).mean()
    """)

    issues = scan_source_for_leakage(leaky_source)

    assert issues, "Expected at least one leakage issue, got none"
    full_text = "\n".join(issues).lower()
    assert "center" in full_text, (
        f"Expected 'center' in leakage report. Got: {issues}"
    )
    print(f"\n[Test 1] PASS — centered rolling window flagged: {issues}")


# ===========================================================================
# Test 2 — Negative shift AST/text detection
# ===========================================================================

def test_negative_shift_flagged():
    """
    Text scanner must detect `df['x'].shift(-5)` (pulls future values into past).
    """
    leaky_source = textwrap.dedent("""
        def compute_leaky(df):
            # Peeks 5 days into the future — definitional leakage
            return df['x'].shift(-5)
    """)

    issues = scan_source_for_leakage(leaky_source)

    assert issues, "Expected leakage issue for shift(-5), got none"
    full_text = "\n".join(issues).lower()
    assert "shift" in full_text or "negative" in full_text, (
        f"Expected 'shift' or 'negative' in report. Got: {issues}"
    )
    print(f"\n[Test 2] PASS — negative shift flagged: {issues}")


# ===========================================================================
# Test 3 — Back-fill (bfill) AST/text detection
# ===========================================================================

def test_bfill_flagged():
    """
    Text scanner must detect `df['x'].bfill()` (propagates future values backward).
    """
    leaky_source = textwrap.dedent("""
        def impute_badly(df):
            # bfill propagates FUTURE known values into PAST rows — leakage
            return df['x'].bfill()
    """)

    issues = scan_source_for_leakage(leaky_source)

    assert issues, "Expected leakage issue for bfill(), got none"
    full_text = "\n".join(issues).lower()
    assert "bfill" in full_text or "back" in full_text, (
        f"Expected 'bfill' or 'back' in report. Got: {issues}"
    )
    print(f"\n[Test 3] PASS — bfill flagged: {issues}")


# ===========================================================================
# Test 4 — Target-correlation statistical detection
# ===========================================================================

def test_target_correlation_detected():
    """
    Statistical auditor must:
    - Flag `f1` (= target * 0.99 + noise) as LEAKY (|r| ≥ 0.95)
    - Flag `f2` (= pure noise) as CLEAN
    """
    rng = np.random.default_rng(0)
    n = 200
    base_date = date(2021, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n)]

    y = rng.integers(0, 2, size=n).astype(float)
    f1 = y * 0.99 + rng.standard_normal(n) * 0.01   # near-perfect target proxy
    f2 = rng.standard_normal(n)                       # pure noise

    df = pd.DataFrame({
        "event_date": dates,
        "y": y,
        "f1": f1,
        "f2": f2,
    })

    auditor = LeakageAuditor()
    report = auditor.audit_dataframe(df, date_col="event_date", target_col="y")

    verdicts = {r.feature_name: r.verdict for r in report.results}
    corrs    = {r.feature_name: r.checks.get("target_corr") for r in report.results}

    print(f"\n[Test 4] verdicts={verdicts}, corrs={corrs}")

    assert verdicts.get("f1") == "LEAKY", (
        f"Expected f1=LEAKY, got {verdicts.get('f1')} (corr={corrs.get('f1'):.3f})"
    )
    assert verdicts.get("f2") == "CLEAN", (
        f"Expected f2=CLEAN, got {verdicts.get('f2')}"
    )
    print("[Test 4] PASS — f1=LEAKY, f2=CLEAN")


# ===========================================================================
# Test 5 — Train/test distribution shift (scale-fit leak signal)
# ===========================================================================

def test_distribution_shift_flagged():
    """
    Statistical auditor must flag a feature whose test distribution is
    dramatically shifted vs train (z-score > 5) as REVIEW or LEAKY.

    This catches normalizers that were fit on train+test jointly.
    """
    rng = np.random.default_rng(1)
    n_train, n_test = 90, 30
    n = n_train + n_test
    base_date = date(2020, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n)]
    y = rng.integers(0, 2, size=n).astype(float)

    # Train: N(0, 1) — after joint normalization, test should look like N(100, 1)
    feature = np.concatenate([
        rng.standard_normal(n_train),          # train window: mean≈0
        rng.standard_normal(n_test) + 100.0,   # test window: mean≈100 → z=100
    ])

    df = pd.DataFrame({"event_date": dates, "y": y, "shifted_feature": feature})

    auditor = LeakageAuditor()
    report = auditor.audit_dataframe(df, date_col="event_date", target_col="y")

    verdicts = {r.feature_name: r.verdict for r in report.results}
    z_scores = {r.feature_name: r.checks.get("test_z_vs_train") for r in report.results}

    print(f"\n[Test 5] verdicts={verdicts}, z_scores={z_scores}")

    assert verdicts.get("shifted_feature") in ("REVIEW", "LEAKY"), (
        f"Expected REVIEW or LEAKY for shifted_feature, got {verdicts.get('shifted_feature')}"
        f" (z={z_scores.get('shifted_feature')})"
    )
    assert (z_scores.get("shifted_feature") or 0) > 5, (
        f"Expected z > 5, got {z_scores.get('shifted_feature')}"
    )
    print(f"[Test 5] PASS — shifted_feature flagged {verdicts.get('shifted_feature')} "
          f"(z={z_scores.get('shifted_feature'):.1f})")


# ===========================================================================
# Test 6 — Clean source + clean feature both pass
# ===========================================================================

def test_clean_feature_passes():
    """
    A feature computed with `rolling(window=10).mean().shift(1)` is a
    trailing window lagged by 1 bar — definitionally point-in-time clean.
    Both the AST scan AND the statistical audit must pass as CLEAN.
    """
    # ── 6a: source-level scan ─────────────────────────────────────────
    clean_source = textwrap.dedent("""
        def compute_clean_feature(df):
            # Trailing 10-bar mean, then lag by 1 so we never use the current bar
            return df['close'].rolling(window=10).mean().shift(1)
    """)

    issues = scan_source_for_leakage(clean_source)
    # rolling(center=False) — no look-ahead, no negative shift, no bfill
    assert not issues, (
        f"Expected no leakage issues for clean source, got: {issues}"
    )
    print(f"\n[Test 6a] PASS — clean source has 0 issues")

    # ── 6b: statistical audit ─────────────────────────────────────────
    rng = np.random.default_rng(7)
    n = 200
    base_date = date(2022, 6, 1)
    dates = [base_date + timedelta(days=i) for i in range(n)]
    y = rng.integers(0, 2, size=n).astype(float)

    # Generate the feature the same way the clean source would:
    close = pd.Series(100.0 + rng.standard_normal(n).cumsum())
    clean_feature = close.rolling(window=10).mean().shift(1).values

    df = pd.DataFrame({"event_date": dates, "y": y, "clean_ma": clean_feature})

    auditor = LeakageAuditor()
    report = auditor.audit_dataframe(df, date_col="event_date", target_col="y")

    verdicts = {r.feature_name: r.verdict for r in report.results}
    print(f"[Test 6b] verdicts={verdicts}")

    assert verdicts.get("clean_ma") == "CLEAN", (
        f"Expected clean_ma=CLEAN, got {verdicts.get('clean_ma')}: "
        f"{[r for r in report.results if r.feature_name == 'clean_ma']}"
    )
    print("[Test 6] PASS — clean feature passes AST and statistical audit")


# ===========================================================================
# Bonus: Confirm `assert_clean` raises on LEAKY report
# ===========================================================================

def test_assert_clean_raises_on_leaky():
    """assert_clean() must raise ValueError when LEAKY features exist."""
    rng = np.random.default_rng(0)
    n = 200
    base_date = date(2021, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n)]
    y = rng.integers(0, 2, size=n).astype(float)
    f_leaky = y * 0.99 + rng.standard_normal(n) * 0.01

    df = pd.DataFrame({"event_date": dates, "y": y, "f_leaky": f_leaky})

    auditor = LeakageAuditor()
    report = auditor.audit_dataframe(df, date_col="event_date", target_col="y")

    with pytest.raises(ValueError, match="LEAKY"):
        auditor.assert_clean(report)
    print("\n[Bonus] PASS — assert_clean raises on LEAKY report")
