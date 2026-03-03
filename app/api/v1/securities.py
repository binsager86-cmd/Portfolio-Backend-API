"""
Securities Master API v1 — CRUD for canonical securities and aliases.

Mirrors the Streamlit ``ui_securities_master()`` logic.
"""

import time
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.exceptions import NotFoundError, BadRequestError, ConflictError
from app.core.database import query_df, query_one, query_val, exec_sql, get_connection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/securities", tags=["Securities Master"])


# ── Schemas ──────────────────────────────────────────────────────────

class SecurityCreate(BaseModel):
    canonical_ticker: str = Field(..., min_length=1, max_length=50)
    exchange: str = Field(..., min_length=1, max_length=30)
    display_name: Optional[str] = Field(None, max_length=200)
    isin: Optional[str] = Field(None, max_length=20)
    currency: str = Field("KWD", max_length=10)
    country: str = Field("KW", max_length=5)
    sector: Optional[str] = Field(None, max_length=100)
    status: str = Field("active", max_length=20)
    aliases: Optional[list[str]] = Field(default_factory=list)


class SecurityUpdate(BaseModel):
    canonical_ticker: Optional[str] = Field(None, max_length=50)
    exchange: Optional[str] = Field(None, max_length=30)
    display_name: Optional[str] = Field(None, max_length=200)
    isin: Optional[str] = Field(None, max_length=20)
    currency: Optional[str] = Field(None, max_length=10)
    country: Optional[str] = Field(None, max_length=5)
    sector: Optional[str] = Field(None, max_length=100)
    status: Optional[str] = Field(None, max_length=20)


class AliasCreate(BaseModel):
    alias_name: str = Field(..., min_length=1, max_length=100)
    alias_type: str = Field("user_input", description="user_input, broker_format, official, or legacy")
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None


# ── Securities list ──────────────────────────────────────────────────

@router.get("")
async def list_securities(
    exchange: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None, description="Search ticker or name"),
    current_user: TokenData = Depends(get_current_user),
):
    """List all securities in the master table with optional filters."""
    conditions = ["user_id = ?"]
    params: list = [current_user.user_id]

    if exchange:
        conditions.append("exchange = ?")
        params.append(exchange)
    if status:
        conditions.append("status = ?")
        params.append(status)
    if search:
        conditions.append("(canonical_ticker LIKE ? OR display_name LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])

    where = " AND ".join(conditions)
    df = query_df(
        f"""
        SELECT security_id, exchange, canonical_ticker, display_name,
               isin, currency, country, status, sector
        FROM securities_master
        WHERE {where}
        ORDER BY exchange, canonical_ticker
        """,
        tuple(params),
    )

    records = df.to_dict(orient="records") if not df.empty else []

    return {
        "status": "ok",
        "data": {
            "securities": records,
            "count": len(records),
        },
    }


# ── Get single security ─────────────────────────────────────────────

@router.get("/{security_id}")
async def get_security(
    security_id: str,
    current_user: TokenData = Depends(get_current_user),
):
    """Get a security with its aliases."""
    row = query_one(
        "SELECT * FROM securities_master WHERE security_id = ? AND user_id = ?",
        (security_id, current_user.user_id),
    )
    if not row:
        raise NotFoundError("Security", security_id)

    security = dict(row)

    # Fetch aliases
    alias_df = query_df(
        "SELECT alias_name, alias_type, valid_from, valid_until "
        "FROM security_aliases WHERE security_id = ? AND user_id = ?",
        (security_id, current_user.user_id),
    )
    security["aliases"] = alias_df.to_dict(orient="records") if not alias_df.empty else []

    return {"status": "ok", "data": security}


# ── Create security ──────────────────────────────────────────────────

@router.post("", status_code=201)
async def create_security(
    body: SecurityCreate,
    current_user: TokenData = Depends(get_current_user),
):
    """Create a new security in the master table with optional aliases."""
    uid = current_user.user_id
    ticker = body.canonical_ticker.strip().upper()
    exchange = body.exchange.strip().upper()

    # Generate security_id  (e.g. "KSE:HUMANSOFT")
    security_id = f"{exchange}:{ticker}"

    # Check for conflict
    existing = query_val(
        "SELECT 1 FROM securities_master WHERE security_id = ? AND user_id = ?",
        (security_id, uid),
    )
    if existing:
        raise ConflictError(f"Security '{security_id}' already exists")

    now = int(time.time())
    exec_sql(
        """INSERT INTO securities_master
           (security_id, user_id, exchange, canonical_ticker, display_name,
            isin, currency, country, status, sector, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            security_id, uid, exchange, ticker,
            body.display_name or ticker,
            body.isin, body.currency, body.country,
            body.status, body.sector, now,
        ),
    )

    # Create aliases (include the ticker itself)
    aliases_to_create = set(body.aliases or [])
    aliases_to_create.add(ticker)
    if body.display_name:
        aliases_to_create.add(body.display_name)

    with get_connection() as conn:
        cur = conn.cursor()
        for alias in aliases_to_create:
            alias_clean = alias.strip()
            if not alias_clean:
                continue
            try:
                cur.execute(
                    """INSERT OR IGNORE INTO security_aliases
                       (security_id, user_id, alias_name, alias_type, created_at)
                       VALUES (?, ?, ?, 'user_input', ?)""",
                    (security_id, uid, alias_clean, now),
                )
            except Exception:
                pass  # skip duplicates
        conn.commit()

    return {
        "status": "ok",
        "data": {
            "security_id": security_id,
            "message": "Security created",
        },
    }


# ── Update security ──────────────────────────────────────────────────

@router.put("/{security_id}")
async def update_security(
    security_id: str,
    body: SecurityUpdate,
    current_user: TokenData = Depends(get_current_user),
):
    """Update a security's metadata."""
    existing = query_one(
        "SELECT security_id FROM securities_master WHERE security_id = ? AND user_id = ?",
        (security_id, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("Security", security_id)

    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        raise BadRequestError("No valid fields to update")

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    params = list(updates.values()) + [security_id, current_user.user_id]

    exec_sql(
        f"UPDATE securities_master SET {set_clause} WHERE security_id = ? AND user_id = ?",
        tuple(params),
    )

    return {"status": "ok", "data": {"security_id": security_id, "message": "Security updated"}}


# ── Delete security ──────────────────────────────────────────────────

@router.delete("/{security_id}")
async def delete_security(
    security_id: str,
    current_user: TokenData = Depends(get_current_user),
):
    """Delete a security and its aliases."""
    existing = query_one(
        "SELECT security_id FROM securities_master WHERE security_id = ? AND user_id = ?",
        (security_id, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("Security", security_id)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM security_aliases WHERE security_id = ? AND user_id = ?",
            (security_id, current_user.user_id),
        )
        cur.execute(
            "DELETE FROM securities_master WHERE security_id = ? AND user_id = ?",
            (security_id, current_user.user_id),
        )
        conn.commit()

    return {"status": "ok", "data": {"security_id": security_id, "message": "Security deleted"}}


# ── Alias CRUD ───────────────────────────────────────────────────────

@router.get("/{security_id}/aliases")
async def list_aliases(
    security_id: str,
    current_user: TokenData = Depends(get_current_user),
):
    """List all aliases for a security."""
    df = query_df(
        "SELECT alias_name, alias_type, valid_from, valid_until "
        "FROM security_aliases WHERE security_id = ? AND user_id = ?",
        (security_id, current_user.user_id),
    )
    records = df.to_dict(orient="records") if not df.empty else []
    return {"status": "ok", "data": {"aliases": records, "count": len(records)}}


@router.post("/{security_id}/aliases", status_code=201)
async def add_alias(
    security_id: str,
    body: AliasCreate,
    current_user: TokenData = Depends(get_current_user),
):
    """Add an alias to a security."""
    uid = current_user.user_id

    # Verify security exists
    sec = query_one(
        "SELECT security_id FROM securities_master WHERE security_id = ? AND user_id = ?",
        (security_id, uid),
    )
    if not sec:
        raise NotFoundError("Security", security_id)

    # Check for duplicate alias
    existing = query_val(
        "SELECT 1 FROM security_aliases WHERE alias_name = ? AND user_id = ?",
        (body.alias_name.strip(), uid),
    )
    if existing:
        raise ConflictError(f"Alias '{body.alias_name}' already exists")

    now = int(time.time())
    exec_sql(
        """INSERT INTO security_aliases
           (security_id, user_id, alias_name, alias_type, valid_from, valid_until, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (security_id, uid, body.alias_name.strip(), body.alias_type,
         body.valid_from, body.valid_until, now),
    )

    return {"status": "ok", "data": {"alias_name": body.alias_name, "message": "Alias added"}}


@router.delete("/{security_id}/aliases/{alias_name}")
async def delete_alias(
    security_id: str,
    alias_name: str,
    current_user: TokenData = Depends(get_current_user),
):
    """Remove an alias from a security."""
    existing = query_val(
        "SELECT 1 FROM security_aliases WHERE security_id = ? AND alias_name = ? AND user_id = ?",
        (security_id, alias_name, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("Alias", alias_name)

    exec_sql(
        "DELETE FROM security_aliases WHERE security_id = ? AND alias_name = ? AND user_id = ?",
        (security_id, alias_name, current_user.user_id),
    )

    return {"status": "ok", "data": {"alias_name": alias_name, "message": "Alias deleted"}}
