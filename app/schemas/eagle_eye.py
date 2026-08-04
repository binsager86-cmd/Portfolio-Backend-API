"""
Eagle Eye schemas — request/response models for the Eagle Eye API.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    latest_volume: Optional[float] = None
    average_volume: Optional[float] = None


class RatedStock(BaseModel):
    ticker: str
    name_en: str
    sector: str
    stage: str
    rating: str
    confidence: float         # 0-100
    thesis: str
    entry_primary: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    average_volume: Optional[float] = None
    latest_volume: Optional[float] = None
    last_price: Optional[float] = None
    book_value_per_share: Optional[float] = None
    pe_ratio: Optional[float] = None
    computed_at: Optional[str] = None   # ISO date string
    volume_context: Optional[VolumeContextSummary] = None


class ScannerResponse(BaseModel):
    status: str = "ok"
    count: int
    stocks: List[RatedStock]


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


# ---------------------------------------------------------------------------
# Strategy Simulator
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    """Request body for POST /api/v1/eagle-eye/simulations."""
    start_date: date
    end_date: date
    initial_cash: Decimal = Field(default=Decimal("100000.00"), gt=0)
    max_positions: int = Field(default=10, ge=1, le=50)
    position_sizing_mode: str = Field(default="equal")     # equal|fixed|percentage
    fixed_position_size: Optional[Decimal] = None
    position_size_pct: Decimal = Field(default=Decimal("10.0"), ge=1, le=100)
    commission_pct: Decimal = Field(default=Decimal("0.001"), ge=0)
    slippage_pct: Decimal = Field(default=Decimal("0.001"), ge=0)
    execution_rule: str = Field(default="next_open")        # next_open|same_close
    allow_pyramiding: bool = False
    include_dividends: bool = True
    universe: Optional[List[str]] = None     # None = all available symbols
    benchmark_symbol: Optional[str] = None


class DailyLedgerRow(BaseModel):
    """One daily record in the equity curve."""
    date: str
    cash: str
    invested_value: str
    total_equity: str
    positions_count: int


class TradeLedgerRow(BaseModel):
    """One completed trade."""
    symbol: str
    entry_date: str
    entry_price: str
    exit_date: Optional[str] = None
    exit_price: Optional[str] = None
    quantity: str
    realized_pnl_gross: str
    realized_pnl_pct: str
    holding_days: int


class SkippedSignalRow(BaseModel):
    """One skipped or rejected signal."""
    symbol: str
    signal_date: str
    rating: str
    reason: str
    quantity_requested: Optional[str] = None


class SimulationResponse(BaseModel):
    """Response for simulation endpoints."""
    run_id: str
    status: str  # COMPLETED|FAILED|RUNNING

    # Data availability
    signal_data_status: str = "HISTORICAL_SIGNAL_DATA_UNAVAILABLE"
    # HISTORICAL_SIGNAL_REPLAY | RECONSTRUCTED_RESEARCH | CURRENT_RATING_DEMO

    error_message: Optional[str] = None

    # Configuration (if completed)
    config: Optional[Any] = None

    # Summary metrics
    ending_equity: Optional[Decimal] = None
    ending_cash: Optional[Decimal] = None
    total_return_pct: Optional[Decimal] = None
    max_drawdown_pct: Optional[Decimal] = None
    trades_count: Optional[int] = None
    win_rate_pct: Optional[Decimal] = None
    profit_factor: Optional[Decimal] = None
    buy_signals_executed: Optional[int] = None
    sell_signals_executed: Optional[int] = None

    # Validation
    cash_reconciliation_ok: Optional[bool] = None
    equity_reconciliation_ok: Optional[bool] = None
    validation_warnings: Optional[List[str]] = None

    # Timestamps
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_seconds: Optional[float] = None

    # Ledger data (only in /result endpoint, not summary)
    daily_ledger: Optional[List[DailyLedgerRow]] = None
    trades: Optional[List[TradeLedgerRow]] = None
    skipped_signals: Optional[List[SkippedSignalRow]] = None
