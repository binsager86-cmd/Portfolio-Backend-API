"""
Admin API v1 — user management and activity monitoring.

All endpoints require admin privileges (is_admin=1).
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import require_admin
from app.core.security import TokenData, hash_password
from app.core.database import query_all, query_val, query_df, exec_sql
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

        # Portfolio value & cost — try daily_snapshots first, then stocks+txns
        market_val = 0.0
        cost_val = 0.0
        try:
            snapshot = query_all(
                "SELECT SUM(mkt_value_base), SUM(cost_value_base) "
                "FROM daily_snapshots WHERE snapshot_date = ("
                "  SELECT MAX(snapshot_date) FROM daily_snapshots"
                ") AND asset_id IN ("
                "  SELECT id FROM stocks WHERE user_id = ?"
                ")",
                (uid,),
            )
            if snapshot and snapshot[0] and snapshot[0][0] is not None:
                market_val = float(snapshot[0][0] or 0)
                cost_val = float(snapshot[0][1] or 0)
        except Exception:
            pass

        # If no snapshot, try computing from transactions (LEFT JOIN stocks for price)
        if market_val == 0:
            try:
                df = query_df(
                    "SELECT t.stock_symbol, "
                    "  COALESCE(s.current_price, 0) * "
                    "    SUM(CASE WHEN LOWER(t.txn_type) IN ('buy','bonus shares','bonus') THEN t.shares "
                    "              WHEN LOWER(t.txn_type) = 'sell' THEN -t.shares ELSE 0 END) "
                    "  as market_value, "
                    "  SUM(CASE WHEN LOWER(t.txn_type) = 'buy' THEN COALESCE(t.purchase_cost, 0) "
                    "           WHEN LOWER(t.txn_type) = 'sell' THEN -COALESCE(t.sell_value, 0) ELSE 0 END) "
                    "  as cost_basis "
                    "FROM transactions t "
                    "LEFT JOIN stocks s ON s.user_id = t.user_id AND s.symbol = t.stock_symbol "
                    "WHERE t.user_id = ? AND COALESCE(t.is_deleted, 0) = 0 "
                    "GROUP BY t.stock_symbol",
                    (uid,),
                )
                if not df.empty:
                    market_val = float(df["market_value"].sum())
                    cost_val = float(df["cost_basis"].sum())
            except Exception:
                pass  # graceful fallback

        # Last login: most recent audit log entry for login
        last_login = query_val(
            "SELECT MAX(created_at) FROM audit_log WHERE user_id = ? AND action = 'auth.login'",
            (uid,),
        )

        # Cash balance: Deposits - Buys + Sells + Dividends - Fees
        cash_bal = 0.0
        try:
            cash_row = query_val(
                "SELECT COALESCE(SUM(net_change), 0) FROM ("
                "  SELECT COALESCE(amount, 0) AS net_change FROM cash_deposits"
                "    WHERE user_id = ? AND COALESCE(include_in_analysis,1) = 1"
                "    AND COALESCE(is_deleted,0) = 0"
                "  UNION ALL"
                "  SELECT -1 * COALESCE(purchase_cost, 0) FROM transactions"
                "    WHERE user_id = ? AND txn_type = 'Buy'"
                "    AND COALESCE(is_deleted,0) = 0"
                "  UNION ALL"
                "  SELECT COALESCE(sell_value, 0) FROM transactions"
                "    WHERE user_id = ? AND txn_type = 'Sell'"
                "    AND COALESCE(is_deleted,0) = 0"
                "  UNION ALL"
                "  SELECT COALESCE(cash_dividend, 0) FROM transactions"
                "    WHERE user_id = ? AND COALESCE(cash_dividend,0) > 0"
                "    AND COALESCE(is_deleted,0) = 0"
                "  UNION ALL"
                "  SELECT -1 * COALESCE(fees, 0) FROM transactions"
                "    WHERE user_id = ? AND COALESCE(fees,0) > 0"
                "    AND COALESCE(is_deleted,0) = 0"
                ") AS cash_movements",
                (uid, uid, uid, uid, uid),
            )
            cash_bal = float(cash_row or 0)
        except Exception:
            pass

        stocks_val = round(market_val, 2)
        total_val = round(market_val + cash_bal, 2)

        result.append(AdminUserRow(
            id=uid,
            username=username,
            name=name,
            created_at=created_at,
            last_login=last_login,
            stocks_value=stocks_val,
            cash_balance=round(cash_bal, 2),
            total_value=total_val,
            portfolio_value=stocks_val,
            growth_value=round(market_val - cost_val, 2),
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
    user_id: int, body: UpdatePasswordRequest,
    current_user: TokenData = Depends(require_admin),
):
    """Reset a user's password (admin only — no current password required)."""
    user = query_val("SELECT id FROM users WHERE id = ?", (user_id,))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    hashed = hash_password(body.password)
    exec_sql("UPDATE users SET password_hash = ? WHERE id = ?", (hashed, user_id))
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

    # Delete related data in correct order
    for table in [
        "daily_snapshots", "portfolio_snapshots", "position_snapshots",
        "pfm_assets", "pfm_liabilities", "pfm_income_expenses", "pfm_snapshots",
        "portfolio_transactions", "external_accounts",
        "securities_master", "security_aliases",
        "stocks", "transactions", "cash_deposits",
        "portfolio_cash", "portfolios", "user_settings",
        "token_blacklist", "audit_log",
    ]:
        try:
            exec_sql(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
        except Exception:
            pass  # table may not exist

    exec_sql("DELETE FROM users WHERE id = ?", (user_id,))
    return AdminMessageResponse(message=f"User '{user}' deleted successfully")
