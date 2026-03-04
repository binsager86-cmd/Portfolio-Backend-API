"""
Pydantic schemas for the Financial Integrity API (Phase 3.2).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


# ── Component-level schemas ──────────────────────────────────────────

class CashComponents(BaseModel):
    deposits: str
    buys: str
    sells: str
    dividends: str
    fees: str


class CashCheckResult(BaseModel):
    portfolio: str
    expected_balance: str
    stored_balance: Optional[str] = None
    discrepancy: Optional[str] = None
    is_valid: Optional[bool] = None
    tolerance: str
    components: CashComponents


class PositionMismatch(BaseModel):
    symbol: str
    agg_shares: float
    wac_shares: float
    share_diff: float
    agg_total_bought_cost: float
    wac_remaining_cost: float


class PositionCheckResult(BaseModel):
    portfolio: str
    total_symbols: int
    matched: int
    mismatches: List[PositionMismatch] = []
    details: List[PositionMismatch] = []
    is_valid: bool


class SnapshotCheckResult(BaseModel):
    portfolio: str
    has_snapshots: bool
    is_fresh: bool
    is_valid: Optional[bool] = None
    message: Optional[str] = None
    latest_date: Optional[str] = None
    days_since_snapshot: Optional[int] = None
    snapshot_value: Optional[float] = None
    live_value: Optional[float] = None
    drift_pct: Optional[float] = None


class Anomaly(BaseModel):
    txn_id: Optional[int] = None
    type: str
    detail: str
    severity: str  # "error" | "warning" | "info"


class AnomalyReport(BaseModel):
    anomalies: List[Anomaly] = []
    count: int = 0
    errors: int = 0
    warnings: int = 0
    is_valid: bool = True


class CompletenessIssue(BaseModel):
    type: str
    detail: str
    severity: str


class CompletenessReport(BaseModel):
    portfolios_found: int = 0
    orphan_symbols: int = 0
    zero_price_symbols: int = 0
    portfolios_without_deposits: List[str] = []
    issues: List[CompletenessIssue] = []
    is_valid: bool = True


class IntegritySummary(BaseModel):
    cash_checks: int = 0
    position_checks: int = 0
    snapshot_checks: int = 0
    anomaly_errors: int = 0
    anomaly_warnings: int = 0
    completeness_issues: int = 0


# ── Top-level report ─────────────────────────────────────────────────

class IntegrityReport(BaseModel):
    user_id: int
    timestamp: str
    overall_valid: Optional[bool] = None
    summary: IntegritySummary
    cash: Dict[str, Any] = {}
    positions: Dict[str, Any] = {}
    snapshots: Dict[str, Any] = {}
    anomalies: AnomalyReport
    completeness: CompletenessReport


class IntegrityResponse(BaseModel):
    status: str = "ok"
    data: IntegrityReport
