"""Model hyperparameters for the Kuwait Signal Engine.

All thresholds and indicator periods live here — no hard-coded magic
numbers in the scoring modules.
"""
from __future__ import annotations

# ── HMM Regime Detector ───────────────────────────────────────────────────────
HMM_N_STATES: int = 3
HMM_COVARIANCE_TYPE: str = "diag"   # diagonal for stability with limited data
HMM_RANDOM_STATE: int = 42
HMM_N_ITER: int = 100
HMM_MIN_TRAIN_BARS: int = 120       # minimum bars required to train HMM

# Regime names (must correspond to state ordering by mean return, ascending)
REGIME_BEAR: str = "Bearish_Contraction"
REGIME_CHOP: str = "Neutral_Chop"
REGIME_BULL: str = "Bullish_Expansion"

# ── Feature ATR Percentile Window ────────────────────────────────────────────
ATR_PERCENTILE_WINDOW: int = 250    # bars used for percentile ranking

# ── Indicator Periods ─────────────────────────────────────────────────────────
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL_PERIOD: int = 9
ATR_PERIOD: int = 14
ADX_PERIOD: int = 14
CMF_PERIOD: int = 20
ROC_PERIOD: int = 10                # rate of change lookback for momentum
OBV_SLOPE_BARS: int = 5             # bars used for OBV linear-slope calculation
VWAP_ANCHOR_LOOKBACK: int = 60      # bars for anchored-VWAP calculation

# ── Trend Score Thresholds ────────────────────────────────────────────────────
ADX_TRENDING_MIN: float = 20.0
ADX_STRONG_MIN: float = 25.0
ADX_VERY_STRONG_MIN: float = 30.0

# ── RSI Thresholds ────────────────────────────────────────────────────────────
RSI_OVERSOLD: float = 35.0
RSI_OVERBOUGHT: float = 65.0
RSI_BULL_MOMENTUM_LOW: float = 50.0
RSI_BULL_MOMENTUM_HIGH: float = 65.0

# ── Swing Pivot Detection ─────────────────────────────────────────────────────
PIVOT_LOOKBACK: int = 10    # bars each side of a swing pivot candidate
PIVOT_CLUSTER_PCT: float = 0.015   # levels within 1.5 % count as same cluster
SR_PROXIMITY_PCT: float = 0.02     # "at level" if price within ± 2 %

# ── Support/Resistance Resistance-Check ──────────────────────────────────────
RESISTANCE_WITHIN_1_5R_BLOCK: bool = True  # hard-block BUY if strong res < 1.5R

# ── Stop & Target Multipliers (ATR-based) ────────────────────────────────────
STOP_ATR_MULTIPLIER: float = 1.5    # stop = entry – 1.5 × ATR14
TP1_RR_MULTIPLIER: float = 1.5     # TP1 reward = risk × 1.5
TP2_RR_MULTIPLIER: float = 3.0     # TP2 reward = risk × 3.0
TP3_RR_MULTIPLIER: float = 4.0     # TP3 reward = risk × 4.0  (aggressive target)
ENTRY_BUFFER_PCT: float = 0.005    # entry zone half-width (± 0.5 % of close)

# ── Signal Confluence Thresholds ─────────────────────────────────────────────
SIGNAL_STRONG_BUY_SCORE: int = 85      # total weighted score for STRONG_BUY
SIGNAL_MIN_TOTAL_SCORE: int = 70       # total weighted score for BUY
SIGNAL_MAX_TOTAL_SELL: int = 25        # total weighted score for SELL
SIGNAL_MIN_TREND_RAW_PCT: float = 60.0 # trend raw ≥ 60 % of max for BUY
SIGNAL_MIN_VOLFLOW_RAW_PCT: float = 50.0
SIGNAL_MIN_RR: float = 2.0             # minimum risk/reward ratio

# Calibrated TP1 probability floors (post auction-adj + decay).
# A signal must clear these to qualify as BUY / STRONG_BUY, regardless of score.
SIGNAL_MIN_P_TP1_BUY: float = 0.55
SIGNAL_MIN_P_TP1_STRONG_BUY: float = 0.65
# Mirror floor for SELL — minimum p_tp1 on the short side.
SIGNAL_MIN_P_TP1_SELL: float = 0.50

# ── Base Confluence Weights (regime-neutral) ──────────────────────────────────
BASE_WEIGHTS: dict[str, float] = {
    "trend":              0.25,
    "momentum":           0.20,
    "volume_flow":        0.25,
    "support_resistance": 0.15,
    "risk_reward":        0.15,
}

# ── Entry Trigger Thresholds ──────────────────────────────────────────────────
PULLBACK_EMA_PROXIMITY_PCT: float = 0.01   # price within ±1 % of EMA-20
PULLBACK_LOOKBACK_BARS: int = 5            # bars to check EMA slope
PULLBACK_STOCH_MAX: float = 50.0           # stoch %K must be below this

BREAKOUT_RANGE_BARS: int = 8               # consolidation window
BREAKOUT_RANGE_ATR_MULT_MAX: float = 1.8   # range ≤ 1.8 × ATR = "tight"
BREAKOUT_VOLUME_MULT_MIN: float = 1.5      # volume ≥ 1.5 × 20-bar avg
BREAKOUT_VOLUME_AVG_BARS: int = 20         # lookback for avg volume

ACCUMULATION_OBV_MIN_SLOPE_PCT: float = 0.3  # OBV slope ≥ 0.3 % per bar
ACCUMULATION_CMF_MIN: float = 0.05            # CMF ≥ 0.05

# ── Missing Data Handling ─────────────────────────────────────────────────────
MAX_FORWARD_FILL_DAYS: int = 3      # forward-fill gaps up to 3 days, then NaN
MIN_BARS_FOR_SIGNAL: int = 60       # need at least 60 valid bars
