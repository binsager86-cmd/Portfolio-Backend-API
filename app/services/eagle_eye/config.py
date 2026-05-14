"""
Kuwait Stock Analysis Engine — Central Configuration
All tunable parameters live here. Change them here, not scattered in code.
"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass(frozen=True)
class EngineConfig:
    # --- Historical analysis window ---
    HISTORY_YEARS: int = 3
    MIN_HISTORY_DAYS_REQUIRED: int = 500  # ~2 trading years minimum to issue ratings

    # --- Move detection thresholds (multi-threshold learning) ---
    MOVE_THRESHOLDS_PCT: tuple = (10.0, 15.0, 25.0, 50.0, 100.0)

    # Minimum days a move must take to be "real" (filters out single-day spikes)
    MIN_MOVE_DURATION_DAYS: int = 3
    # Maximum days to look forward for a move to complete from a starting low
    MAX_MOVE_LOOKAHEAD_DAYS: int = 180

    # --- Pre-move forensic snapshot lookbacks (in trading days) ---
    PRE_MOVE_LOOKBACK_DAYS: tuple = (90, 60, 30, 14, 7, 3, 1, 0)

    # --- Indicator parameters ---
    EMA_PERIODS: tuple = (8, 21, 50, 100, 200)
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    ATR_PERIOD: int = 14
    BB_PERIOD: int = 20
    BB_STDDEV: float = 2.0
    ADX_PERIOD: int = 14
    STOCH_K: int = 14
    STOCH_D: int = 3
    CCI_PERIOD: int = 20
    MFI_PERIOD: int = 14
    CMF_PERIOD: int = 20
    OBV_SLOPE_PERIOD: int = 20
    VOLUME_AVG_PERIOD: int = 20
    DONCHIAN_PERIOD: int = 20
    KELTNER_PERIOD: int = 20
    SUPERTREND_PERIOD: int = 10
    SUPERTREND_MULTIPLIER: float = 3.0

    # --- Volume Profile (Kuwait-adapted: longer window for lumpier volume) ---
    VOLUME_PROFILE_DAYS: int = 90
    VOLUME_PROFILE_BUCKETS: int = 50

    # --- Stage classification thresholds ---
    DORMANT_ATR_PCTILE_MAX: float = 25.0
    DORMANT_BB_WIDTH_PCTILE_MAX: float = 25.0
    ACCUMULATION_SCORE_THRESHOLD: float = 60.0
    BREAKOUT_VOLUME_MULTIPLIER: float = 2.0
    CLIMAX_RSI_THRESHOLD: float = 80.0
    CAPITULATION_RSI_THRESHOLD: float = 25.0
    TRENDING_ADX_THRESHOLD: float = 25.0

    # --- Position sizing & safety ---
    LIQUIDITY_CAP_PCT_OF_DAILY_TURNOVER: float = 10.0
    CIRCUIT_BREAKER_DRAWDOWN_PCT: float = 25.0
    CONFIRMATION_MODAL_THRESHOLD_PCT: float = 30.0
    HALF_KELLY_MULTIPLIER: float = 0.5
    SECTOR_EXPOSURE_CAP_PCT: float = 35.0

    # --- Rating thresholds (asymmetric — need more proof to issue STRONG BUY) ---
    STRONG_BUY_CONFIDENCE: float = 85.0
    BUY_CONFIDENCE: float = 70.0
    HOLD_CONFIDENCE: float = 45.0
    SELL_CONFIDENCE: float = 25.0

    # --- Kuwait market specifics ---
    MIN_DAILY_TURNOVER_KWD: float = 25_000.0
    KUWAIT_TIMEZONE: str = "Asia/Kuwait"
    TRADING_DAYS: tuple = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday")

    # --- Regime detection ---
    REGIME_INDEX_TICKER: str = "PMI"
    REGIME_EMA_PERIOD: int = 50
    BRENT_TICKER: str = "BRENT"
    BRENT_LOOKBACK_DAYS: int = 30

    # --- Retraining cadence ---
    RETRAIN_DAY: str = "Friday"
    BEHAVIORAL_DNA_REFRESH_DAYS: int = 7


# Indicator category mapping
INDICATOR_CATEGORIES: Dict[str, List[str]] = {
    "trend": [
        "ema_8", "ema_21", "ema_50", "ema_100", "ema_200",
        "ema_ribbon_aligned", "macd_line", "macd_histogram", "macd_signal_cross",
        "adx", "plus_di", "minus_di", "supertrend", "parabolic_sar",
        "ichimoku_cloud_position", "ichimoku_tk_cross", "hull_ma_slope",
        "linear_regression_slope",
    ],
    "momentum": [
        "rsi", "rsi_divergence", "stoch_k", "stoch_d", "stoch_rsi",
        "williams_r", "cci", "roc", "tsi", "awesome_oscillator",
        "connors_rsi", "momentum_divergence_obv", "momentum_divergence_macd",
    ],
    "volatility": [
        "atr", "atr_percentile_252", "bb_percent_b", "bb_bandwidth",
        "bb_squeeze", "keltner_position", "donchian_position",
        "chaikin_volatility", "historical_volatility_30d",
    ],
    "volume_flow": [
        "obv", "obv_slope_20d", "obv_divergence", "ad_line", "ad_line_slope",
        "cmf", "mfi", "vwap", "vwap_distance", "anchored_vwap_distance",
        "volume_relative_20d", "force_index", "ease_of_movement",
        "klinger", "vw_macd",
    ],
    "structure": [
        "nearest_support_distance", "nearest_resistance_distance",
        "support_strength", "resistance_strength",
        "vp_poc_distance", "vp_vah_distance", "vp_val_distance",
        "vp_hvn_proximity", "vp_lvn_proximity",
        "fib_38_distance", "fib_50_distance", "fib_61_distance",
        "pivot_classic", "pivot_fib", "pivot_camarilla",
        "swing_high_distance", "swing_low_distance",
    ],
    "institutional": [
        "wyckoff_phase", "accumulation_score", "block_trade_pressure",
        "up_down_volume_ratio_30d", "closing_strength_30d",
        "narrowing_range_rising_volume", "insider_filing_score",
    ],
    "statistical": [
        "zscore_vs_20ma", "zscore_vs_50ma", "zscore_vs_200ma",
        "hurst_exponent", "rsi_percentile_252",
        "atr_zscore", "price_zscore_vwap",
    ],
    "regime": [
        "regime_state", "sector_relative_strength", "index_correlation_60d",
        "beta_vs_premier_index", "brent_correlation_60d",
        "market_breadth_pct_above_50ma",
    ],
}


# Stock lifecycle stages
STAGES = [
    "DORMANT",
    "STEALTH_ACCUMULATION",
    "EARLY_BREAKOUT",
    "MARKUP_TRENDING",
    "ACCELERATION_CLIMAX",
    "DISTRIBUTION_TOPPING",
    "MARKDOWN_DECLINE",
    "CAPITULATION_EXHAUSTION",
]

# Stage colors for UI
STAGE_COLORS = {
    "DORMANT":                "#5F5E5A",
    "STEALTH_ACCUMULATION":   "#185FA5",
    "EARLY_BREAKOUT":         "#0F6E56",
    "MARKUP_TRENDING":        "#3B6D11",
    "ACCELERATION_CLIMAX":    "#BA7517",
    "DISTRIBUTION_TOPPING":   "#993C1D",
    "MARKDOWN_DECLINE":       "#A32D2D",
    "CAPITULATION_EXHAUSTION": "#534AB7",
}

# Ratings
RATINGS = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", "INSUFFICIENT_DATA"]


# Singleton
CONFIG = EngineConfig()
