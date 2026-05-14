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
    last_price: Optional[float] = None
    computed_at: Optional[str] = None   # ISO date string


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
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp1_probability: Optional[float] = None
    tp2: Optional[float] = None
    tp2_probability: Optional[float] = None
    tp3: Optional[float] = None
    tp3_probability: Optional[float] = None

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

class ThresholdProfileResponse(BaseModel):
    threshold_pct: float
    success_rate: float
    sample_count: int
    median_bars_to_hit: Optional[float] = None
    avg_win_pct: Optional[float] = None
    avg_loss_pct: Optional[float] = None


class BehavioralDNAResponse(BaseModel):
    ticker: str
    total_events_analyzed: int
    most_reliable_signals: List[str] = []
    threshold_profiles: List[ThresholdProfileResponse] = []
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
