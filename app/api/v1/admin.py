"""
Admin API v1 — user management and activity monitoring.

All endpoints require admin privileges (is_admin=1).
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from app.api.deps import require_admin
from app.core.security import TokenData, hash_password
from app.core.database import query_all, query_val, query_df, exec_sql, table_exists, transaction
from app.schemas.user import _validate_strong_password
from app.services.audit_service import ADMIN_ACTION, log_event
from app.services.password_service import change_user_password
from app.services.portfolio_service import PortfolioService
from app.services.user_onboarding import setup_new_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


# ── Response models ──────────────────────────────────────────────────

class AdminUserRow(BaseModel):
    id: int
    username: str
    name: Optional[str] = None
    created_at: Optional[int] = None
    last_login: Optional[int] = None
    stocks_value: float = 0.0
    cash_balance: float = 0.0
    total_value: float = 0.0
    portfolio_value: float = 0.0
    growth_value: float = 0.0
    daily_change: float = 0.0
    transaction_count: int = 0


class AdminUsersResponse(BaseModel):
    status: str = "ok"
    count: int
    users: list[AdminUserRow]


class AdminActivityRow(BaseModel):
    id: int
    user_id: int
    username: str
    txn_date: Optional[str] = None
    txn_type: str
    stock_symbol: str
    portfolio: str
    shares: float = 0.0
    value: float = 0.0
    price: float = 0.0
    created_at: Optional[int] = None


class AdminActivitiesResponse(BaseModel):
    status: str = "ok"
    count: int
    total: int
    activities: list[AdminActivityRow]


# ── Endpoints ────────────────────────────────────────────────────────

@router.get("/users", response_model=AdminUsersResponse)
async def list_users(current_user: TokenData = Depends(require_admin)):
    """
    List all registered users with aggregated portfolio data.

    Returns: registered date, name, username, last login,
    portfolio value (sum of market values), growth (market - cost),
    and transaction count per user.
    """
    # Fetch all users
    users_raw = query_all(
        "SELECT id, username, name, created_at FROM users ORDER BY created_at DESC"
    )

    result = []
    for row in users_raw:
        uid, username, name, created_at = row

        # Transaction count
        txn_count = query_val(
            "SELECT COUNT(*) FROM transactions WHERE user_id = ? AND COALESCE(is_deleted, 0) = 0",
            (uid,),
        ) or 0

        # get_total_portfolio_value() returns stocks_kwd/cash_kwd/total_value_kwd —
        # this used to read stocks_value_kwd/portfolio_value_kwd/total_cost_kwd (keys
        # that don't exist on that dict), so stocks/growth silently computed as 0 for
        # every user. get_overview() has the right keys plus total_gain (lifetime P&L
        # vs net deposits, cash included) and daily_movement (change vs yesterday's
        # snapshot), which is what "growth" and "updated daily" should reflect.
        try:
            overview = PortfolioService(uid).get_overview()
        except Exception:
            overview = {}
        market_val = float(overview.get("portfolio_value") or 0.0)
        cash_bal = float(overview.get("cash_balance") or 0.0)
        total_val = float(overview.get("total_value") or (market_val + cash_bal))
        growth_val = float(overview.get("total_gain") or 0.0)
        daily_change_val = float(overview.get("daily_movement") or 0.0)

        # Last login: most recent auth activity. Password login and Google login
        # cover explicit sign-ins; token_refresh covers silent re-auth on app open,
        # which is how most returning sessions actually happen (mobile apps keep
        # users signed in and just rotate the refresh token instead of re-hitting
        # /login), so restricting this to 'auth.login' alone made active users show
        # as "Never" once their initial session token was refreshed.
        last_login = query_val(
            "SELECT MAX(created_at) FROM audit_log WHERE user_id = ? "
            "AND action IN ('auth.login', 'auth.google_login', 'auth.token_refresh')",
            (uid,),
        )

        stocks_val = round(market_val, 2)

        result.append(AdminUserRow(
            id=uid,
            username=username,
            name=name,
            created_at=created_at,
            last_login=last_login,
            stocks_value=stocks_val,
            cash_balance=round(cash_bal, 2),
            total_value=round(total_val, 2),
            portfolio_value=stocks_val,
            growth_value=round(growth_val, 2),
            daily_change=round(daily_change_val, 2),
            transaction_count=txn_count,
        ))

    return AdminUsersResponse(count=len(result), users=result)


@router.get("/activities", response_model=AdminActivitiesResponse)
async def list_activities(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    txn_type: Optional[str] = Query(None, description="Filter by type: buy, sell, dividend, deposit"),
    stock_symbol: Optional[str] = Query(None, description="Filter by stock symbol (partial match)"),
    date_from: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    current_user: TokenData = Depends(require_admin),
):
    """
    List all client transactions across all users.

    Returns: date, type (buy/sell/deposit/dividend), stock name,
    quantity, value. Supports pagination and filtering.
    """
    # Build WHERE clause
    conditions = ["COALESCE(t.is_deleted, 0) = 0"]
    params: list = []

    if user_id is not None:
        conditions.append("t.user_id = ?")
        params.append(user_id)
    if txn_type:
        conditions.append("LOWER(t.txn_type) = ?")
        params.append(txn_type.lower())
    if stock_symbol:
        conditions.append("LOWER(t.stock_symbol) LIKE ?")
        params.append(f"%{stock_symbol.lower()}%")
    if date_from:
        conditions.append("t.txn_date >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("t.txn_date <= ?")
        params.append(date_to)

    where = " AND ".join(conditions)

    # Total count
    total = query_val(
        f"SELECT COUNT(*) FROM transactions t WHERE {where}",
        tuple(params),
    ) or 0

    # Paginated results with username join
    offset = (page - 1) * per_page
    rows = query_all(
        f"SELECT t.id, t.user_id, u.username, t.txn_date, t.txn_type, "
        f"t.stock_symbol, t.portfolio, t.shares, "
        f"CASE "
        f"  WHEN LOWER(t.txn_type) = 'buy' THEN t.purchase_cost "
        f"  WHEN LOWER(t.txn_type) = 'sell' THEN t.sell_value "
        f"  WHEN LOWER(t.txn_type) LIKE '%dividend%' THEN t.cash_dividend "
        f"  ELSE 0 "
        f"END as value, "
        f"t.created_at "
        f"FROM transactions t "
        f"JOIN users u ON u.id = t.user_id "
        f"WHERE {where} "
        f"ORDER BY t.created_at DESC "
        f"LIMIT ? OFFSET ?",
        tuple(params) + (per_page, offset),
    )

    activities = []
    for row in rows:
        shares = float(row[7] or 0)
        value = float(row[8] or 0)
        price = round(value / shares, 4) if shares > 0 else 0.0
        activities.append(AdminActivityRow(
            id=row[0],
            user_id=row[1],
            username=row[2],
            txn_date=row[3],
            txn_type=row[4],
            stock_symbol=row[5] or "",
            portfolio=row[6] or "",
            shares=shares,
            value=round(value, 3),
            price=price,
            created_at=row[9],
        ))

    return AdminActivitiesResponse(
        count=len(activities),
        total=total,
        activities=activities,
    )


# ── Request / Response models for user management ───────────────────

class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    name: str | None = None


class UpdateUsernameRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)


class UpdatePasswordRequest(BaseModel):
    password: str = Field(..., min_length=8)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _validate_strong_password(v)


class AdminMessageResponse(BaseModel):
    status: str = "ok"
    message: str


# ── User CRUD endpoints ─────────────────────────────────────────────

@router.post("/users", response_model=AdminMessageResponse, status_code=status.HTTP_201_CREATED)
async def create_user(body: CreateUserRequest, current_user: TokenData = Depends(require_admin)):
    """Create a new user (admin only)."""
    existing = query_val(
        "SELECT id FROM users WHERE username = ?", (body.username,)
    )
    if existing:
        raise HTTPException(status_code=409, detail=f"Username '{body.username}' already exists")

    hashed = hash_password(body.password)
    now = int(time.time())

    exec_sql(
        "INSERT INTO users (username, email, password_hash, name, created_at, failed_login_attempts) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        (body.username, body.username, hashed, body.name, now),
    )

    user_id = query_val("SELECT id FROM users WHERE username = ?", (body.username,))
    setup_new_user(user_id, body.username)

    return AdminMessageResponse(message=f"User '{body.username}' created successfully")


@router.put("/users/{user_id}/username", response_model=AdminMessageResponse)
async def update_username(
    user_id: int, body: UpdateUsernameRequest,
    current_user: TokenData = Depends(require_admin),
):
    """Change a user's username (admin only)."""
    user = query_val("SELECT id FROM users WHERE id = ?", (user_id,))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    conflict = query_val(
        "SELECT id FROM users WHERE username = ? AND id != ?", (body.username, user_id)
    )
    if conflict:
        raise HTTPException(status_code=409, detail=f"Username '{body.username}' already taken")

    exec_sql("UPDATE users SET username = ? WHERE id = ?", (body.username, user_id))
    return AdminMessageResponse(message=f"Username updated to '{body.username}'")


@router.put("/users/{user_id}/password", response_model=AdminMessageResponse)
async def update_password(
    user_id: int,
    body: UpdatePasswordRequest,
    request: Request,
    current_user: TokenData = Depends(require_admin),
):
    """Reset a user's password (admin only — no current password required)."""
    user = query_val("SELECT id FROM users WHERE id = ?", (user_id,))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    change_user_password(
        user_id,
        body.password,
        request=request,
        actor_user_id=current_user.user_id,
        audit_action=ADMIN_ACTION,
        audit_details={"action": "admin.password_reset"},
    )
    return AdminMessageResponse(message="Password updated successfully")


@router.delete("/users/{user_id}", response_model=AdminMessageResponse)
async def delete_user(
    user_id: int, current_user: TokenData = Depends(require_admin),
):
    """Delete a user and all related data (admin only)."""
    user = query_val("SELECT username FROM users WHERE id = ?", (user_id,))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Prevent admin from deleting themselves
    if user_id == current_user.user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    # Delete related data in correct order. Preserve audit_log for retention/compliance history.
    tables = [
        "daily_snapshots", "portfolio_snapshots", "position_snapshots",
        "pfm_assets", "pfm_liabilities", "pfm_income_expenses", "pfm_snapshots",
        "portfolio_transactions", "external_accounts",
        "securities_master", "security_aliases",
        "ledger_entries", "push_tokens", "portfolio_news_dispatches",
        "stocks", "transactions", "cash_deposits",
        "portfolio_cash", "portfolios", "user_settings",
        "token_blacklist",
    ]
    deleted = {}
    with transaction():
        log_event(
            ADMIN_ACTION,
            user_id=current_user.user_id,
            resource_type="user",
            resource_id=user_id,
            details={"action": "admin.delete_user", "target_username": user},
        )
        for table in tables:
            if not table_exists(table):
                deleted[table] = "absent"
                continue
            exec_sql(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
            deleted[table] = "cleared"

        exec_sql("DELETE FROM users WHERE id = ?", (user_id,))
    return AdminMessageResponse(message=f"User '{user}' deleted successfully")
