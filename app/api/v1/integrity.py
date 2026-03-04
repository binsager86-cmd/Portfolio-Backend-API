"""
Financial Integrity API — Phase 3.2

Endpoints to run data-integrity checks on portfolio calculations.

All endpoints are authenticated (current user only — row-level security).
"""

import logging

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.services.integrity_service import IntegrityService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integrity", tags=["Integrity"])


# ── Full sweep ───────────────────────────────────────────────────────

@router.get("/check")
async def full_integrity_check(
    current_user: TokenData = Depends(get_current_user),
):
    """
    Run ALL integrity checks for the authenticated user:
      - Cash balance verification per portfolio
      - Position cross-check (aggregate SQL vs WAC engine)
      - Snapshot freshness & consistency
      - Transaction anomaly scan
      - Data completeness
    """
    svc = IntegrityService(current_user.user_id)
    report = svc.run_full_integrity_check()
    return {"status": "ok", "data": report}


# ── Individual checks ────────────────────────────────────────────────

@router.get("/cash/{portfolio}")
async def check_cash_balance(
    portfolio: str,
    current_user: TokenData = Depends(get_current_user),
):
    """Verify cash balance for a single portfolio."""
    svc = IntegrityService(current_user.user_id)
    result = svc.verify_cash_balance(portfolio)
    return {"status": "ok", "data": result}


@router.get("/positions/{portfolio}")
async def check_positions(
    portfolio: str,
    current_user: TokenData = Depends(get_current_user),
):
    """Cross-check holdings (aggregate vs WAC engine) for a portfolio."""
    svc = IntegrityService(current_user.user_id)
    result = svc.verify_positions(portfolio)
    return {"status": "ok", "data": result}


@router.get("/snapshots/{portfolio}")
async def check_snapshots(
    portfolio: str,
    current_user: TokenData = Depends(get_current_user),
):
    """Check snapshot freshness & consistency for a portfolio."""
    svc = IntegrityService(current_user.user_id)
    result = svc.verify_snapshots(portfolio)
    return {"status": "ok", "data": result}


@router.get("/anomalies")
async def check_anomalies(
    current_user: TokenData = Depends(get_current_user),
):
    """Scan all transactions for anomalies (duplicates, over-sells, etc.)."""
    svc = IntegrityService(current_user.user_id)
    result = svc.scan_transaction_anomalies()
    return {"status": "ok", "data": result}


@router.get("/completeness")
async def check_completeness(
    current_user: TokenData = Depends(get_current_user),
):
    """Check for missing stock entries, zero prices, etc."""
    svc = IntegrityService(current_user.user_id)
    result = svc.verify_data_completeness()
    return {"status": "ok", "data": result}
