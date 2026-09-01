"""
Trend-Hold Book — response schemas.

Own file, deliberately not appended to app/schemas/eagle_eye.py's
RatedStock/ScannerResponse -- this is a separate subsystem (a virtual-money
paper-trading ledger fed by trend_hold_engine decisions), not an extension
of the scanner response.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class TrendHoldBookPortfolio(BaseModel):
    cash_kwd: float
    starting_capital_kwd: float
    equity_kwd: float
    total_return_pct: float
    open_position_count: int
    as_of: Optional[str] = None


class TrendHoldBookPosition(BaseModel):
    ticker: str
    quantity: float
    avg_cost: float
    latest_close: Optional[float] = None
    market_value_kwd: Optional[float] = None
    unrealized_pnl_kwd: Optional[float] = None
    opened_date: Optional[str] = None


class TrendHoldBookPositionsResponse(BaseModel):
    positions: List[TrendHoldBookPosition]


class TrendHoldBookTrade(BaseModel):
    id: int
    ticker: str
    side: str  # BUY / SCALE_OUT / EXIT
    trade_date: str
    quantity: float
    price: float
    gross_kwd: float
    commission_kwd: float
    realized_pnl_kwd: Optional[float] = None
    reason: Optional[str] = None
    # 0-100 signal-strength score computed at the moment this decision fired
    # (see trend_hold_engine.py's _entry_confidence/_exit_confidence) --
    # BUY and EXIT only; null for SCALE_OUT (a fixed profit-milestone rule,
    # not a judged signal).
    confidence: Optional[float] = None


class TrendHoldBookTradesResponse(BaseModel):
    trades: List[TrendHoldBookTrade]


class TrendHoldBookNavPoint(BaseModel):
    nav_date: str
    cash_kwd: float
    equity_kwd: float
    open_position_count: int


class TrendHoldBookNavHistoryResponse(BaseModel):
    points: List[TrendHoldBookNavPoint]


class TrendHoldDecisionLogEntry(BaseModel):
    ticker: str
    trade_date: str
    decision: str
    reason: Optional[str] = None
    position_state: Optional[str] = None
    close: Optional[float] = None
    structural_stop: Optional[float] = None
    confidence: Optional[float] = None


class TrendHoldDecisionLogResponse(BaseModel):
    entries: List[TrendHoldDecisionLogEntry]


class TrendHoldBookLesson(BaseModel):
    ticker: str
    trade_date: str
    side: str  # SCALE_OUT / EXIT
    classification: str
    outcome: str  # WIN / LOSS / PARTIAL / UNKNOWN
    mae_pct: Optional[float] = None
    mfe_pct: Optional[float] = None
    giveback_pct: Optional[float] = None
    holding_days: Optional[int] = None
    reason: str
    enhancement: str


class TrendHoldBookLessonsResponse(BaseModel):
    lessons: List[TrendHoldBookLesson]


class TrendHoldBookLessonsSummary(BaseModel):
    total_closed: int
    by_classification: dict
    by_outcome: dict
    avg_loss_mae_pct: Optional[float] = None
    avg_win_giveback_pct: Optional[float] = None


class TrendHoldBookPerformance(BaseModel):
    total_closed: int
    win_count: int
    loss_count: int
    win_rate_pct: Optional[float] = None
    total_realized_pnl_kwd: float
    max_profit_kwd: Optional[float] = None
    max_loss_kwd: Optional[float] = None
    avg_win_kwd: Optional[float] = None
    avg_loss_kwd: Optional[float] = None
    profit_factor: Optional[float] = None
    expectancy_kwd: Optional[float] = None
    total_commission_paid_kwd: float
