"""
Portfolio schemas — holdings, overview, transactions.

Matches the openapi.yaml contract for Portfolio + Transactions tags.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Holding Row ──────────────────────────────────────────────────────

class HoldingRow(BaseModel):
    """Single stock holding — one row in the portfolio table."""
    company: str
    symbol: str
    shares_qty: float
    avg_cost: float
    total_cost: float
    previous_close: Optional[float] = None
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    cash_dividends: float
    reinvested_dividends: float
    bonus_dividend_shares: float
    dividend_yield_on_cost_pct: float
    total_pnl: float
    pnl_pct: float
    currency: str
    # KWD conversions
    market_value_kwd: float
    unrealized_pnl_kwd: float
    total_pnl_kwd: float
    total_cost_kwd: float
    weight_by_cost: Optional[float] = None
    allocation_pct: Optional[float] = None
    weighted_dividend_yield_on_cost: Optional[float] = None


class HoldingsTotals(BaseModel):
    """Aggregate totals across all holdings."""
    total_market_value_kwd: float = 0.0
    total_cost_kwd: float = 0.0
    total_unrealized_pnl_kwd: float = 0.0
    total_realized_pnl_kwd: float = 0.0
    total_pnl_kwd: float = 0.0
    total_dividends_kwd: float = 0.0


class HoldingsResponse(BaseModel):
    """GET /portfolio/holdings response."""
    status: str = "ok"
    data: Optional[Dict[str, Any]] = None  # {holdings, totals, usd_kwd_rate, count}


class PortfolioTableResponse(BaseModel):
    """GET /portfolio/table/{name} response."""
    status: str = "ok"
    data: Optional[Dict[str, Any]] = None  # {portfolio, currency, holdings, count, usd_kwd_rate}


# ── Overview ─────────────────────────────────────────────────────────

class PortfolioOverview(BaseModel):
    """Complete portfolio overview — overview tab payload."""
    # Transaction aggregates (all KWD)
    total_deposits: float = 0.0
    total_withdrawals: float = 0.0
    net_deposits: float = 0.0
    total_invested: float = 0.0
    total_divested: float = 0.0
    total_dividends: float = 0.0
    total_fees: float = 0.0
    transaction_count: int = 0
    # Current values
    portfolio_value: float = 0.0
    cash_balance: float = 0.0
    total_value: float = 0.0
    # Calculated metrics
    total_gain: float = 0.0
    roi_percent: float = 0.0
    # Breakdowns
    by_portfolio: Dict[str, Any] = {}
    portfolio_values: Dict[str, Any] = {}
    accounts: List[Dict[str, Any]] = []
    # FX
    usd_kwd_rate: float = 0.0


# ── Transactions CRUD ────────────────────────────────────────────────

class TransactionCreate(BaseModel):
    """Create a new transaction."""
    portfolio: str = Field(..., description="Portfolio name: KFH, BBYN, or USA")
    stock_symbol: str = Field(..., min_length=1, max_length=50)
    txn_date: str = Field(..., description="YYYY-MM-DD", pattern=r"^\d{4}-\d{2}-\d{2}$")
    txn_type: str = Field(..., description="Buy, Sell, or DIVIDEND_ONLY")
    shares: float = Field(0, ge=0, description="Shares (0 allowed for DIVIDEND_ONLY)")
    purchase_cost: Optional[float] = Field(None, ge=0, description="Required for Buy")
    sell_value: Optional[float] = Field(None, ge=0, description="Required for Sell")
    bonus_shares: Optional[float] = Field(None, ge=0)
    cash_dividend: Optional[float] = Field(None, ge=0)
    reinvested_dividend: Optional[float] = Field(None, ge=0)
    fees: Optional[float] = Field(None, ge=0)
    price_override: Optional[float] = None
    planned_cum_shares: Optional[float] = None
    broker: Optional[str] = Field(None, max_length=100)
    reference: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = None


class TransactionUpdate(BaseModel):
    """Update an existing transaction (partial)."""
    stock_symbol: Optional[str] = Field(None, min_length=1, max_length=50)
    txn_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    txn_type: Optional[str] = None
    shares: Optional[float] = Field(None, ge=0)
    purchase_cost: Optional[float] = Field(None, ge=0)
    sell_value: Optional[float] = Field(None, ge=0)
    bonus_shares: Optional[float] = Field(None, ge=0)
    cash_dividend: Optional[float] = Field(None, ge=0)
    reinvested_dividend: Optional[float] = Field(None, ge=0)
    fees: Optional[float] = Field(None, ge=0)
    price_override: Optional[float] = None
    planned_cum_shares: Optional[float] = None
    broker: Optional[str] = Field(None, max_length=100)
    reference: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = None


class TransactionResponse(BaseModel):
    """Single transaction."""
    id: int
    user_id: int
    portfolio: str
    stock_symbol: str
    txn_date: Optional[str] = None
    txn_type: str
    shares: Optional[float] = None
    purchase_cost: Optional[float] = None
    sell_value: Optional[float] = None
    bonus_shares: Optional[float] = None
    cash_dividend: Optional[float] = None
    reinvested_dividend: Optional[float] = None
    fees: Optional[float] = None
    price_override: Optional[float] = None
    planned_cum_shares: Optional[float] = None
    broker: Optional[str] = None
    reference: Optional[str] = None
    notes: Optional[str] = None
    category: Optional[str] = "portfolio"
    is_deleted: Optional[int] = 0
    created_at: Optional[int] = None


class TransactionListResponse(BaseModel):
    """Paginated transaction list."""
    status: str = "ok"
    data: Optional[Dict[str, Any]] = None  # {transactions, count, pagination}
