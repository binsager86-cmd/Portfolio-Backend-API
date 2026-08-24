"""
Eagle Eye schemas — request/response models for the Eagle Eye API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SupportResistanceLevel(BaseModel):
    price: float
    strength: float          # 0-100
    method: str              # "swing_low", "vp_poc", "fib_61.8", etc.


class SignalBreakdown(BaseModel):
    signal: str
    fired: bool
    value: Optional[float] = None
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Scanner endpoint
# ---------------------------------------------------------------------------

class VolumeContextSummary(BaseModel):
    """Subset of volume context surfaced in scanner rows."""
    relative_volume: float = 1.0
    liquidity_tier: str = "TRADEABLE"
    is_volume_confirmed: bool = True
    volume_character: str = "NEUTRAL"
    volume_trend_5d: str = "NEUTRAL"
    today_volume: Optional[float] = None
    avg_20d_volume: Optional[float] = None


class RatedStock(BaseModel):
    ticker: str
    name_en: str
    sector: str
    stage: str
    rating: str
    confidence: float         # 0-100 - latest/live confidence (fresh computation or today's cache)
    confidence_yesterday: Optional[float] = None  # 0-100 - yesterday's cached confidence (before today's refresh)
    thesis: str
    entry_primary: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    last_price: Optional[float] = None
    average_volume: Optional[float] = None
    latest_volume: Optional[float] = None
    book_value_per_share: Optional[float] = None
    pe_ratio: Optional[float] = None
    computed_at: Optional[str] = None   # ISO date string
    volume_context: Optional[VolumeContextSummary] = None
    continue_rising: bool = False
    continue_rising_badge: Optional[str] = None
    continue_rising_label: Optional[str] = None
    continue_rising_reason: Optional[str] = None
    continue_rising_exhaustion_count: int = 0
    continue_rising_exhaustion_signals: List[str] = []
    risk_warning_score: int = 0
    risky_near_resistance: bool = False
    risk_reward_ratio: Optional[float] = None


class ScannerResponse(BaseModel):
    status: str = "ok"
    count: int
    stocks: List[RatedStock]
    progress_phase: Optional[str] = None
    progress_message: Optional[str] = None
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None
    progress_percent: Optional[int] = None


# ---------------------------------------------------------------------------
# Full stock analysis
# ---------------------------------------------------------------------------

class FullStockAnalysis(BaseModel):
    ticker: str
    name_en: str
    sector: str

    # Stage & rating
    stage: str
    rating: str
    confidence: float
    thesis: str
    continue_rising: bool = False
    continue_rising_badge: Optional[str] = None
    continue_rising_label: Optional[str] = None
    continue_rising_reason: Optional[str] = None
    continue_rising_exhaustion_count: int = 0
    continue_rising_exhaustion_signals: List[str] = []
    risk_warning_score: int = 0
    risky_near_resistance: bool = False

    # SR levels
    supports: List[SupportResistanceLevel] = []
    resistances: List[SupportResistanceLevel] = []

    # Entry / stop / targets
    entry_primary: Optional[float] = None
    entry_aggressive: Optional[float] = None
    entry_conservative: Optional[float] = None
    plan_state: str = "ACTIVE"
    plan_reason: Optional[str] = None
    conditional_entry: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp1_probability: Optional[float] = None
    tp2: Optional[float] = None
    tp2_probability: Optional[float] = None
    tp3: Optional[float] = None
    tp3_probability: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    gain_pct_to_tp1: Optional[float] = None

    # Position sizing
    position_size_pct: Optional[float] = None
    position_size_kwd: Optional[float] = None
    liquidity_capped: Optional[bool] = None
    requires_confirmation: Optional[bool] = None

    # Signals
    signals: List[SignalBreakdown] = []

    # Meta
    computed_at: Optional[str] = None
    days_of_history: Optional[int] = None


class StockAnalysisResponse(BaseModel):
    status: str = "ok"
    data: FullStockAnalysis


# ---------------------------------------------------------------------------
# Behavioral DNA
# ---------------------------------------------------------------------------

class SignalReliabilityResponse(BaseModel):
    signal: str
    reliability_pct: Optional[float] = None
    presence_pct: Optional[float] = None
    fired_count: int
    total_events: Optional[int] = None
    total_setups: Optional[int] = None
    avg_lead_days: Optional[float] = None
    false_positive_rate: Optional[float] = None
    discriminative_power: Optional[float] = None
    lift: Optional[float] = None

class ThresholdProfileResponse(BaseModel):
    threshold_pct: float
    success_rate: float
    sample_count: int
    total_count: Optional[int] = None
    hits: Optional[int] = None
    total_setups: Optional[int] = None
    median_bars_to_hit: Optional[float] = None
    avg_win_pct: Optional[float] = None
    avg_loss_pct: Optional[float] = None
    avg_gain_all_pct: Optional[float] = None
    avg_gain_on_hits_pct: Optional[float] = None


class DNAWindowProfileResponse(BaseModel):
    horizon_days: int
    setup_count: int
    history_status: str = "ok"
    confidence_floor: int = 5
    confidence_tier: str = "TOO_THIN"
    confidence_label: str = "Too thin"
    percentages_visible: bool = False
    raw_setup_count: Optional[int] = None
    threshold_profiles: List[ThresholdProfileResponse] = []


class DNASetupObservationResponse(BaseModel):
    date: str
    signal: str
    label: str
    detail: str
    value: Optional[float] = None


class DNASetupForwardOutcomeResponse(BaseModel):
    horizon_days: int
    completed: bool = False
    max_gain_pct: Optional[float] = None
    max_gain_date: Optional[str] = None
    threshold_hits: List[float] = []


class DNASetupBarResponse(BaseModel):
    date: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    rel_volume: Optional[float] = None
    rsi: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None


class DNASetupExampleResponse(BaseModel):
    setup_date: str
    setup_window_start_date: str
    setup_window_end_date: str
    setup_bar_index: int
    setup_window_start_index: int
    setup_window_end_index: int
    available_forward_bars: int
    bars: List[DNASetupBarResponse] = []
    observations: List[DNASetupObservationResponse] = []
    forward_outcomes: Dict[str, DNASetupForwardOutcomeResponse] = {}


class DNAPullbackEntryProfileResponse(BaseModel):
    median_pullback_pct: float
    max_pullback_pct: float
    pullback_within_days: int
    recovery_days: int
    pullback_success_rate: float
    sample_count: int


class DNAHistoricalTargetClusterResponse(BaseModel):
    gain_pct_from_entry: float
    cluster_strength: int
    avg_days_to_reach: float
    hit_rate: float


class DNACycleProfileResponse(BaseModel):
    period_days: float
    std_days: float
    period_confidence: str
    sample_count: int
    days_since_last: int
    days_to_next: float


class DNAPreDropSignalResponse(BaseModel):
    signal: str
    label: str
    description: str
    fired_before_drop_pct: float
    avg_bars_before_drop: float
    sample_count: int


class DNAPostDropBehaviorResponse(BaseModel):
    classification: str
    bounce_rate_pct: float
    avg_recovery_days: float
    avg_continuation_pct: float
    bounce_window_days: int
    sample_count: int
    classification_reason: str


class DNAExitSignalProfileResponse(BaseModel):
    drop_threshold_pct: float
    historical_drop_events: int
    avg_drop_magnitude_pct: float
    avg_days_peak_to_trough: float
    pre_drop_signals: List[DNAPreDropSignalResponse] = []
    post_drop_behavior: Optional[DNAPostDropBehaviorResponse] = None


class BehavioralDNAResponse(BaseModel):
    ticker: str
    total_events_analyzed: int
    history_status: str = "ok"
    setup_signals: List[str] = []
    setup_horizon_days: Optional[int] = None
    default_window_days: Optional[int] = None
    available_window_days: List[int] = []
    confidence_floor: int = 5
    most_reliable_signals: List[str] = []
    signal_stats: List[SignalReliabilityResponse] = []
    threshold_profiles: List[ThresholdProfileResponse] = []
    window_profiles: List[DNAWindowProfileResponse] = []
    setup_examples: List[DNASetupExampleResponse] = []
    dominant_pattern: Optional[str] = None
    computed_at: Optional[str] = None
    optimal_hold_window_days: Optional[int] = None
    avg_entry_quality_score: Optional[float] = None
    pullback_entry_profile: Optional[DNAPullbackEntryProfileResponse] = None
    historical_target_clusters: List[DNAHistoricalTargetClusterResponse] = []
    cycle_profile: Optional[DNACycleProfileResponse] = None
    exit_signal_profile: Optional[DNAExitSignalProfileResponse] = None


class DNAResponse(BaseModel):
    status: str = "ok"
    data: BehavioralDNAResponse


# ---------------------------------------------------------------------------
# Historical move events
# ---------------------------------------------------------------------------

class MoveEventResponse(BaseModel):
    date: str
    event_type: str           # "breakout", "breakdown", "reversal", "fakeout"
    magnitude_pct: float
    duration_bars: int
    volume_confirmation: bool
    description: Optional[str] = None


class EventsListResponse(BaseModel):
    status: str = "ok"
    ticker: str
    count: int
    events: List[MoveEventResponse]


# ---------------------------------------------------------------------------
# Refresh (background recompute)
# ---------------------------------------------------------------------------

class RefreshRequest(BaseModel):
    tickers: List[str]


class RefreshResponse(BaseModel):
    status: str = "ok"
    job_id: str
    tickers_queued: int
    estimated_minutes: float


# ---------------------------------------------------------------------------
# Market regime
# ---------------------------------------------------------------------------

class RegimeResponse(BaseModel):
    status: str = "ok"
    regime: str                       # "RISK_ON" / "NEUTRAL" / "RISK_OFF"
    pmi_trend: Optional[str] = None   # "expanding" / "contracting" / "neutral"
    brent_trend: Optional[str] = None
    breadth_pct_above_50ma: Optional[float] = None
    last_updated: Optional[str] = None
