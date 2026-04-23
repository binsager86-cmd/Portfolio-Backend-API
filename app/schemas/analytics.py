"""
Analytics schemas — performance metrics, TWR, MWRR, Sharpe, Sortino, realized P&L.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PerformanceMetrics(BaseModel):
    """Portfolio performance metrics response."""
    status: str = "ok"
    data: Optional[Dict[str, Any]] = None
    # data shape: {period, start_date, end_date, twr_percent, mwrr_percent,
    #              roi_percent, total_gain_kwd, starting_value, ending_value, net_deposits}


class SnapshotListResponse(BaseModel):
    """Portfolio snapshot list."""
    status: str = "ok"
    data: Optional[Dict[str, Any]] = None  # {snapshots, count}


class PositionSnapshotListResponse(BaseModel):
    """Position snapshot list."""
    status: str = "ok"
    data: Optional[Dict[str, Any]] = None  # {snapshots, count}


# ── Risk Metrics ─────────────────────────────────────────────────────

class RiskMetricsResponse(BaseModel):
    """Sharpe & Sortino risk-adjusted return metrics."""
    status: str = "ok"
    data: Optional[Dict[str, Any]] = None
    # data shape: {sharpe_ratio, sortino_ratio, rf_rate, mar}


# ── Realized Profit ──────────────────────────────────────────────────

class RealizedProfitDetail(BaseModel):
    """One sell transaction's realized P&L detail."""
    id: int
    symbol: str
    portfolio: str
    txn_date: Optional[str] = None
    shares: float = 0.0
    sell_value: float = 0.0
    avg_cost_at_txn: float = 0.0
    realized_pnl: float = 0.0
    realized_pnl_kwd: float = 0.0
    dividends_allocated_kwd: float = 0.0
    net_pnl_kwd: float = 0.0
    currency: str = "KWD"
    source: str = "calculated"  # "stored" | "calculated"


class RealizedProfitResponse(BaseModel):
    """Aggregate realized profit response."""
    status: str = "ok"
    data: Optional[Dict[str, Any]] = None
    # data shape: {total_realized_kwd, total_profit_kwd, total_loss_kwd, details}


# ── Cash Reconciliation ─────────────────────────────────────────────

class CashReconciliationResponse(BaseModel):
    """Cash reconciliation (5-source UNION ALL) response."""
    status: str = "ok"
    data: Optional[Dict[str, float]] = None  # {portfolio_name: balance, ...}
