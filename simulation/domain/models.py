"""Domain models for Eagle Eye strategy simulator."""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4


class PositionState(Enum):
    """Position state machine."""
    FLAT = "FLAT"                      # No position held
    PENDING_BUY = "PENDING_BUY"        # Buy order submitted, awaiting fill
    OPEN = "OPEN"                      # Position held
    PENDING_SELL = "PENDING_SELL"      # Sell order submitted, awaiting fill
    CLOSED = "CLOSED"                  # Position fully sold
    ERROR = "ERROR"                    # Error during execution


class OrderSide(Enum):
    """Order side: BUY or SELL."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


class EagleEyeRating(Enum):
    """Eagle Eye rating values from canonical model."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    NEUTRAL = "NEUTRAL"


class WyckoffPhase(Enum):
    """Wyckoff market phase classification."""
    DORMANT = "DORMANT"
    STEALTH_ACCUMULATION = "STEALTH_ACCUMULATION"
    EARLY_BREAKOUT = "EARLY_BREAKOUT"
    MARKUP_TRENDING = "MARKUP_TRENDING"
    CLIMAX = "CLIMAX"
    CAPITULATION = "CAPITULATION"
    RECOVERY_BASE = "RECOVERY_BASE"
    NEUTRAL = "NEUTRAL"


class ExecutionRule(Enum):
    """When to execute orders relative to signal."""
    NEXT_SESSION_OPEN = "NEXT_SESSION_OPEN"      # Default: execute at next open
    SAME_SESSION_CLOSE = "SAME_SESSION_CLOSE"    # Not allowed (look-ahead violation)
    LIMIT_ORDER = "LIMIT_ORDER"                  # Use limit price from signal


class PositionSizingMode(Enum):
    """Position sizing calculation mode."""
    EQUAL_ALLOCATION = "EQUAL_ALLOCATION"        # Equal % across max positions
    FIXED_AMOUNT = "FIXED_AMOUNT"                # Fixed currency per position
    PERCENTAGE_EQUITY = "PERCENTAGE_EQUITY"      # % of current equity


@dataclass
class SimulationConfig:
    """Simulation run configuration."""
    run_id: str = field(default_factory=lambda: str(uuid4()))
    start_date: date = field(default_factory=date.today)
    end_date: date = field(default_factory=date.today)
    initial_cash: Decimal = Decimal("100000.00")
    max_concurrent_positions: int = 10
    position_sizing_mode: PositionSizingMode = PositionSizingMode.EQUAL_ALLOCATION
    fixed_position_size: Optional[Decimal] = None  # For FIXED_AMOUNT mode
    position_size_pct: Decimal = Decimal("10.0")   # % of equity per position
    commission_pct: Decimal = Decimal("0.001")     # 0.1% commission
    slippage_pct: Decimal = Decimal("0.001")       # 0.1% slippage
    execution_rule: ExecutionRule = ExecutionRule.NEXT_SESSION_OPEN
    include_dividends: bool = True
    allow_partial_exits: bool = True
    allow_pyramiding: bool = False
    enable_short_selling: bool = False
    benchmark_symbol: Optional[str] = None
    universe: Optional[List[str]] = None           # If None, use all available
    model_version: str = "EE_R14"
    data_cutoff_date: Optional[date] = None
    created_at: datetime = field(default_factory=datetime.now)
    config_hash: str = ""  # Populated by service

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "run_id": self.run_id,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_cash": float(self.initial_cash),
            "max_concurrent_positions": self.max_concurrent_positions,
            "position_sizing_mode": self.position_sizing_mode.value,
            "commission_pct": float(self.commission_pct),
            "slippage_pct": float(self.slippage_pct),
            "execution_rule": self.execution_rule.value,
            "model_version": self.model_version,
            "data_cutoff_date": self.data_cutoff_date.isoformat() if self.data_cutoff_date else None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class EagleEyeRatingRecord:
    """Historical Eagle Eye model output for a symbol at a point in time."""
    symbol: str
    rating_date: date
    rating_timestamp: datetime
    rating: EagleEyeRating
    confidence: Decimal  # 0-100
    stage: WyckoffPhase
    thesis: str  # Recommended action/thesis text
    entry_primary: Optional[Decimal] = None
    entry_aggressive: Optional[Decimal] = None
    entry_conservative: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    tp1: Optional[Decimal] = None  # Target price 1
    tp2: Optional[Decimal] = None
    tp3: Optional[Decimal] = None
    model_version: str = "EE_R14"
    model_input_snapshot_id: Optional[str] = None


@dataclass
class OHLCV:
    """Market data: Open, High, Low, Close, Volume."""
    symbol: str
    date: date
    open_price: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    adjustment_flag: str = "UNADJUSTED"  # "UNADJUSTED", "SPLIT_ADJUSTED", "DIVIDEND_ADJUSTED"
    source: str = "TICKERCHART"


@dataclass
class Order:
    """Order record."""
    order_id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity_requested: Decimal = Decimal("0")
    quantity_filled: Decimal = Decimal("0")
    price_limit: Optional[Decimal] = None  # Execution price rule
    status: OrderStatus = OrderStatus.PENDING
    signal_date: date = field(default_factory=date.today)
    execution_date: Optional[date] = None
    execution_price: Optional[Decimal] = None
    gross_amount: Decimal = Decimal("0")  # qty * price
    commission: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    rejection_reason: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None

    def get_total_cost(self) -> Decimal:
        """Total cash impact for this order."""
        if self.side == OrderSide.BUY:
            return self.gross_amount + self.commission + self.slippage
        else:  # SELL
            return self.gross_amount - self.commission - self.slippage


@dataclass
class TradeRecord:
    """Completed round-trip trade."""
    trade_id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    signal_rating: EagleEyeRating = EagleEyeRating.HOLD
    entry_signal_date: date = field(default_factory=date.today)
    entry_date: date = field(default_factory=date.today)
    entry_price: Decimal = Decimal("0")
    entry_quantity: Decimal = Decimal("0")
    exit_signal_date: Optional[date] = None
    exit_date: Optional[date] = None
    exit_price: Optional[Decimal] = None
    exit_quantity: Decimal = Decimal("0")
    exit_reason: str = "HOLDING"  # "SELL_SIGNAL", "MAX_LOSS", "PROFIT_TARGET", "MANUAL", "HOLDING"
    
    gross_entry_cost: Decimal = Decimal("0")
    entry_commission: Decimal = Decimal("0")
    entry_slippage: Decimal = Decimal("0")
    
    gross_exit_proceeds: Decimal = Decimal("0")
    exit_commission: Decimal = Decimal("0")
    exit_slippage: Decimal = Decimal("0")
    
    realized_pnl_gross: Decimal = Decimal("0")  # Before costs
    realized_pnl_net: Decimal = Decimal("0")    # After costs
    realized_pnl_pct: Decimal = Decimal("0")    # %
    
    holding_days: int = 0
    status: str = "OPEN"  # "OPEN", "CLOSED"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "signal_rating": self.signal_rating.value,
            "entry_signal_date": self.entry_signal_date.isoformat(),
            "entry_date": self.entry_date.isoformat(),
            "entry_price": float(self.entry_price),
            "entry_quantity": float(self.entry_quantity),
            "exit_date": self.exit_date.isoformat() if self.exit_date else None,
            "exit_price": float(self.exit_price) if self.exit_price else None,
            "exit_quantity": float(self.exit_quantity),
            "realized_pnl_net": float(self.realized_pnl_net),
            "realized_pnl_pct": float(self.realized_pnl_pct),
            "holding_days": self.holding_days,
            "status": self.status,
        }


@dataclass
class Position:
    """Current open position."""
    symbol: str
    quantity: Decimal = Decimal("0")
    cost_basis: Decimal = Decimal("0")  # Total acquisition cost (qty * avg_price + fees)
    average_cost: Decimal = Decimal("0")  # Per-share cost
    current_price: Decimal = Decimal("0")
    current_value: Decimal = Decimal("0")  # qty * current_price
    unrealized_pnl: Decimal = Decimal("0")
    unrealized_pnl_pct: Decimal = Decimal("0")
    date_acquired: date = field(default_factory=date.today)
    state: PositionState = PositionState.FLAT
    open_orders: List[Order] = field(default_factory=list)

    def update_market_price(self, price: Decimal) -> None:
        """Mark position to market."""
        self.current_price = price
        if self.quantity > 0:
            self.current_value = self.quantity * price
            self.unrealized_pnl = self.current_value - self.cost_basis
            if self.cost_basis > 0:
                self.unrealized_pnl_pct = (self.unrealized_pnl / self.cost_basis) * Decimal("100")
        else:
            self.current_value = Decimal("0")
            self.unrealized_pnl = Decimal("0")
            self.unrealized_pnl_pct = Decimal("0")


@dataclass
class DailyRecord:
    """Daily portfolio snapshot."""
    date: date
    cash: Decimal = Decimal("0")
    invested_value: Decimal = Decimal("0")
    total_equity: Decimal = Decimal("0")
    daily_return_pct: Decimal = Decimal("0")
    cumulative_return_pct: Decimal = Decimal("0")
    max_drawdown_pct: Decimal = Decimal("0")
    positions_count: int = 0
    positions: Dict[str, Position] = field(default_factory=dict)
    trades_today: List[TradeRecord] = field(default_factory=list)
    orders_today: List[Order] = field(default_factory=list)
    dividends_received: Decimal = Decimal("0")
    commissions_paid: Decimal = Decimal("0")
    slippage_paid: Decimal = Decimal("0")


@dataclass
class SkippedSignalRecord:
    """Record of a signal that could not be executed."""
    symbol: str
    signal_date: date
    signal_rating: EagleEyeRating
    reason: str  # "INSUFFICIENT_CASH", "ALREADY_OPEN", "NO_MARKET_DATA", "ALREADY_SOLD", etc.
    quantity_requested: Decimal = Decimal("0")
    price_available: Optional[Decimal] = None
    details: Optional[str] = None


@dataclass
class SimulationResult:
    """Complete simulation result."""
    run_id: str
    config: SimulationConfig
    status: str = "COMPLETED"  # "COMPLETED", "FAILED", "RUNNING"
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    execution_seconds: float = 0.0

    # Input data
    ratings_loaded: int = 0
    ohlcv_rows_loaded: int = 0

    # Execution statistics
    buy_signals_total: int = 0
    buy_signals_executed: int = 0
    buy_signals_skipped: int = 0
    sell_signals_total: int = 0
    sell_signals_executed: int = 0
    sell_signals_skipped: int = 0

    # Portfolio performance
    initial_cash: Decimal = Decimal("0")
    ending_cash: Decimal = Decimal("0")
    ending_equity: Decimal = Decimal("0")
    total_profit_loss: Decimal = Decimal("0")
    total_return_pct: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    max_drawdown_pct: Decimal = Decimal("0")
    total_commissions: Decimal = Decimal("0")
    total_slippage: Decimal = Decimal("0")

    # Trade statistics
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: Decimal = Decimal("0")
    avg_holding_days: float = 0.0
    largest_winner: Decimal = Decimal("0")
    largest_loser: Decimal = Decimal("0")
    avg_winner: Decimal = Decimal("0")
    avg_loser: Decimal = Decimal("0")
    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")

    # Reconciliation
    cash_reconciliation_ok: bool = False
    cash_reconciliation_error: Decimal = Decimal("0")
    equity_reconciliation_ok: bool = False
    equity_reconciliation_error: Decimal = Decimal("0")

    # Detailed records
    daily_records: List[DailyRecord] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    orders: List[Order] = field(default_factory=list)
    skipped_signals: List[SkippedSignalRecord] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "ending_equity": float(self.ending_equity),
            "total_return_pct": float(self.total_return_pct),
            "max_drawdown_pct": float(self.max_drawdown_pct),
            "trades_count": self.trades_count,
            "win_rate_pct": float(self.win_rate_pct),
            "profit_factor": float(self.profit_factor),
            "realized_pnl": float(self.realized_pnl),
            "total_commissions": float(self.total_commissions),
            "buy_signals_executed": self.buy_signals_executed,
            "sell_signals_executed": self.sell_signals_executed,
            "cash_reconciliation_ok": self.cash_reconciliation_ok,
            "equity_reconciliation_ok": self.equity_reconciliation_ok,
        }
