"""Risk management parameters for the Kuwait Signal Engine."""
from __future__ import annotations

# ── CVaR (Conditional Value-at-Risk) ─────────────────────────────────────────
CVAR_ALPHA: float = 0.05            # 5 % tail → 95th-percentile CVaR
CVAR_LOOKBACK_DAYS: int = 250       # rolling window for historical simulation
CVAR_ILLIQUID_WIDEN_FACTOR: float = 1.25  # widen CVaR 25 % when ADTV < 100k KD

# Trigger extra position-size reduction if CVaR is an outlier vs VaR
CVAR_VAR_RATIO_REDUCE_THRESHOLD: float = 2.0
CVAR_VAR_REDUCE_FACTOR: float = 0.70

# ── Position Sizing ───────────────────────────────────────────────────────────
RISK_PER_TRADE: float = 0.02        # 2 % of account equity per trade (baseline)
LIQUIDITY_THRESHOLD_KD: float = 100_000.0
KELLY_MAX_FRACTION: float = 0.25    # never exceed 25 % of account on one trade
USE_HALF_KELLY: bool = True         # halve Kelly fraction for risk management

# ── Transaction Costs (Kuwait, Premier Market) ───────────────────────────────
TC_COMMISSION: float = 0.0015       # 0.15 % each leg
TC_SLIPPAGE_PREMIER: float = 0.0010 # 0.10 % estimated slippage
TC_SLIPPAGE_MAIN: float = 0.0030
TC_SLIPPAGE_AUCTION: float = 0.0080

# ── Confidence Decay Schedule ─────────────────────────────────────────────────
# Keys are hours since signal generation; values are multiplier on final probability.
# At T+72 the signal is invalidated (multiplier = 0).
DECAY_SCHEDULE: dict[int, float] = {
    0:  1.00,
    24: 0.85,
    48: 0.65,
    72: 0.00,
}

# ── Probability Calibration (pre-seeded lookup table) ────────────────────────
# Maps total_score bucket lower-bound → expected TP1 win rate.
# Derived from the spec target (≥ 68 % win rate at score ≥ 75).
# Refined via isotonic regression once 50+ live trades are accumulated.
SCORE_TO_WIN_RATE: dict[int, float] = {
    0:  0.35,
    25: 0.42,
    50: 0.52,
    65: 0.60,
    70: 0.64,
    75: 0.68,
    80: 0.71,
    85: 0.74,
    90: 0.77,
    95: 0.80,
}

# TP2 win rate is approximately 65 % of TP1 win rate
TP2_WIN_RATE_FRACTION: float = 0.65

# Bayesian prior pseudo-observation count
BAYES_PRIOR_PSEUDO_OBS: int = 50

# Isotonic regression minimum training size
ISO_MIN_SAMPLES: int = 30
