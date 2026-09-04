"""
Trend-Hold Book — response schemas.

Own file, deliberately not appended to app/schemas/eagle_eye.py's
RatedStock/ScannerResponse -- this is a separate subsystem (a virtual-money
paper-trading ledger fed by trend_hold_engine decisions), not an extension
of the scanner response.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class TrendHoldBookPortfolio(BaseModel):
    cash_kwd: float
    starting_capital_kwd: float
    equity_kwd: float
    total_return_pct: float
    open_position_count: int
    # P&L, split by source so "how am I doing" isn't one ambiguous number:
    #   realized_pnl_kwd   — booked gain/loss from closed legs only (SCALE_OUT
    #                        + EXIT), i.e. TrendHoldBookPerformance.total_realized_pnl_kwd
    #   unrealized_pnl_kwd — mark-to-market gain/loss on currently open
    #                        positions only, using the same price source as
    #                        /positions (never touches the decision engine)
    #   net_pnl_kwd        — realized_pnl_kwd + unrealized_pnl_kwd, the true
    #                        net portfolio performance (ties out to
    #                        equity_kwd - starting_capital_kwd)
    realized_pnl_kwd: float = 0.0
    unrealized_pnl_kwd: float = 0.0
    net_pnl_kwd: float = 0.0
    as_of: Optional[str] = None


class TrendHoldBookPosition(BaseModel):
    ticker: str
    quantity: float
    avg_cost: float
    latest_close: Optional[float] = None
    market_value_kwd: Optional[float] = None
    unrealized_pnl_kwd: Optional[float] = None
    opened_date: Optional[str] = None
    # What triggered this position's own BUY, straight from
    # trend_hold_engine's entry-gate snapshot (see trend_hold_engine.py's
    # _build_entry_gate) -- this is what answers "what triggered the buy"
    # and "target price" (there isn't one) for a still-open trade.
    entry_path: Optional[str] = None  # "DONCHIAN" | "EMA_CROSS"
    entry_confidence: Optional[float] = None
    breakout_margin_pct: Optional[float] = None
    rel_volume_entry: Optional[float] = None
    cmf10_entry: Optional[float] = None
    adx14_entry: Optional[float] = None
    sma200_slope_entry: Optional[float] = None
    atr14_entry: Optional[float] = None
    # Today's live trailing-stop level (ee_trend_hold_state.structural_stop)
    # -- the only "target" concept this system has for an open trade: not a
    # take-profit price, just where it currently exits if broken.
    structural_stop: Optional[float] = None


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
    # Buy/sell price, size, and realized P&L for this closing leg -- denormalized
    # from ee_trend_hold_book_trades so a lesson card is a self-contained,
    # full trade record (see paper_book_store.py's _lesson_insert_statement).
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    quantity: Optional[float] = None
    realized_pnl_kwd: Optional[float] = None
    commission_kwd: Optional[float] = None
    # Entry-quality / regime context, straight from trend_hold_engine's own
    # reading at the moment it fired the BUY. Null for any signal source
    # with no equivalent concept (e.g. the V1 Rating Book).
    entry_path: Optional[str] = None  # "DONCHIAN" | "EMA_CROSS"
    entry_confidence: Optional[float] = None  # 0-100
    breakout_margin_pct: Optional[float] = None
    rel_volume_entry: Optional[float] = None
    cmf10_entry: Optional[float] = None
    adx14_entry: Optional[float] = None
    sma200_slope_entry: Optional[float] = None
    atr14_entry: Optional[float] = None
    # Exit-side gate context -- what actually governed the close.
    adx14_exit: Optional[float] = None
    atr14_exit: Optional[float] = None
    structural_stop_at_exit: Optional[float] = None
    # Tested upside missed (pp) -- WIN/PARTIAL-side classifications only
    # (CLEAN_WIN, GAVE_BACK_GAINS, PROFIT_MILESTONE). This system's stated
    # priority is maximizing profit and win rate, so losses keep their
    # classification + reason without an equivalent field.
    pct_left_on_table: Optional[float] = None
    # Forward-look: did the stock keep running after this system exited it?
    # Computed on demand from real OHLCV that arrived *after* trade_date --
    # null until at least 5 trading sessions (1 week) have actually passed,
    # since the answer doesn't exist before then. See
    # trend_hold_lessons.py::compute_forward_look.
    forward_1w_available: bool = False
    forward_1w_price: Optional[float] = None
    forward_1w_return_pct: Optional[float] = None
    forward_peak_20d_pct: Optional[float] = None
    # How many post-exit sessions of OHLCV actually exist right now -- lets
    # the UI say "3 of 5 sessions so far" instead of a bare not-yet-available.
    forward_sessions_available: Optional[int] = None


class TrendHoldBookLessonsResponse(BaseModel):
    lessons: List[TrendHoldBookLesson]


class TrendHoldBookEntryPathStats(BaseModel):
    closed: int
    wins: int
    losses: int
    # Withheld (null) below a minimum-sample floor (see paper_book_store.py's
    # MIN_BUCKET_SAMPLE) rather than shown as a misleadingly precise rate off
    # a handful of trades. Same shape reused for ADX-regime buckets.
    win_rate_pct: Optional[float] = None


class TrendHoldBookLessonsSummary(BaseModel):
    total_closed: int
    by_classification: dict
    by_outcome: dict
    avg_loss_mae_pct: Optional[float] = None
    avg_win_giveback_pct: Optional[float] = None
    # Profit-maximization headline -- the average measurable upside missed
    # across WIN/PARTIAL trades, and how many had any (> 0.5pp). This
    # system's stated priority is maximizing profit and win rate, ahead of
    # loss diagnostics, so this leads the KPI strip in the rendered report.
    avg_pct_left_on_table: Optional[float] = None
    trades_with_room_to_improve: int = 0
    by_entry_path: Dict[str, TrendHoldBookEntryPathStats] = {}
    # ADX14-at-entry regime bands: WEAK_LT20 | MODERATE_20_40 | STRONG_GT40.
    by_adx_bucket: Dict[str, TrendHoldBookEntryPathStats] = {}
    avg_entry_confidence_win: Optional[float] = None
    avg_entry_confidence_loss: Optional[float] = None


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


class BookComparisonResponse(BaseModel):
    """Both paper books' scorecards side by side -- the head-to-head "which one is best" view."""
    trend_hold: TrendHoldBookPerformance
    v1_rating: TrendHoldBookPerformance
    # Portfolio-level view (realized + unrealized + net) for each book, so
    # the comparison strip isn't limited to realized-only performance stats.
    trend_hold_portfolio: TrendHoldBookPortfolio
    v1_rating_portfolio: TrendHoldBookPortfolio
