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


class TrendHoldDecisionLogResponse(BaseModel):
    entries: List[TrendHoldDecisionLogEntry]
