"""
Fundamental Analysis API v1 — Stock profiles, financial statements,
metrics, valuations, and scoring.

Mirrors the Streamlit ``stock_analysis`` package using raw SQL through
the backend's database helpers (same SQLite/PG database).
"""

import hashlib
import json
import math
import os
import time
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.exceptions import NotFoundError, BadRequestError, ConflictError
from app.core.database import query_all, query_one, query_val, query_df, exec_sql, get_connection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fundamental", tags=["Fundamental Analysis"])

# ── Ensure analysis tables exist ─────────────────────────────────────

_SCHEMA_INIT = False


def _ensure_schema() -> None:
    """Create analysis tables if they don't exist (idempotent)."""
    global _SCHEMA_INIT
    if _SCHEMA_INIT:
        return

    from app.core.config import get_settings
    _s = get_settings()
    _PK = "SERIAL PRIMARY KEY" if _s.use_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"

    _TABLES = [
        f"""CREATE TABLE IF NOT EXISTS analysis_stocks (
                id {_PK},
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                company_name TEXT NOT NULL,
                exchange TEXT DEFAULT 'NYSE',
                currency TEXT DEFAULT 'USD',
                sector TEXT,
                industry TEXT,
                country TEXT,
                isin TEXT,
                cik TEXT,
                description TEXT,
                website TEXT,
                outstanding_shares REAL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(user_id, symbol)
            )""",
        f"""CREATE TABLE IF NOT EXISTS financial_statements (
                id {_PK},
                stock_id INTEGER NOT NULL,
                statement_type TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                fiscal_quarter INTEGER,
                period_end_date TEXT NOT NULL,
                filing_date TEXT,
                source_file TEXT,
                extracted_by TEXT DEFAULT 'manual',
                confidence_score REAL,
                verified_by_user BOOLEAN DEFAULT 0,
                notes TEXT,
                created_at INTEGER NOT NULL,
                UNIQUE(stock_id, statement_type, period_end_date),
                FOREIGN KEY (stock_id) REFERENCES analysis_stocks(id)
            )""",
        f"""CREATE TABLE IF NOT EXISTS financial_line_items (
                id {_PK},
                statement_id INTEGER NOT NULL,
                line_item_code TEXT NOT NULL,
                line_item_name TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT DEFAULT 'USD',
                order_index INTEGER,
                parent_item_id INTEGER,
                is_total BOOLEAN DEFAULT 0,
                manually_edited BOOLEAN DEFAULT 0,
                edited_by_user_id INTEGER,
                edited_at INTEGER,
                FOREIGN KEY (statement_id) REFERENCES financial_statements(id)
            )""",
        f"""CREATE TABLE IF NOT EXISTS stock_metrics (
                id {_PK},
                stock_id INTEGER NOT NULL,
                fiscal_year INTEGER NOT NULL,
                fiscal_quarter INTEGER,
                period_end_date TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                created_at INTEGER NOT NULL,
                UNIQUE(stock_id, metric_name, period_end_date),
                FOREIGN KEY (stock_id) REFERENCES analysis_stocks(id)
            )""",
        f"""CREATE TABLE IF NOT EXISTS valuation_models (
                id {_PK},
                stock_id INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                valuation_date TEXT NOT NULL,
                intrinsic_value REAL,
                parameters JSON,
                assumptions JSON,
                created_by_user_id INTEGER,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (stock_id) REFERENCES analysis_stocks(id)
            )""",
        f"""CREATE TABLE IF NOT EXISTS stock_scores (
                id {_PK},
                stock_id INTEGER NOT NULL,
                scoring_date TEXT NOT NULL,
                overall_score REAL,
                fundamental_score REAL,
                valuation_score REAL,
                growth_score REAL,
                quality_score REAL,
                details JSON,
                analyst_notes TEXT,
                created_by_user_id INTEGER,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (stock_id) REFERENCES analysis_stocks(id)
            )""",
        f"""CREATE TABLE IF NOT EXISTS analysis_audit_log (
                id {_PK},
                user_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id INTEGER,
                old_value TEXT,
                new_value TEXT,
                reason TEXT,
                details TEXT,
                created_at INTEGER NOT NULL
            )""",
        f"""CREATE TABLE IF NOT EXISTS pdf_uploads (
                id {_PK},
                stock_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                pdf_hash TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (stock_id) REFERENCES analysis_stocks(id)
            )""",
        "CREATE INDEX IF NOT EXISTS idx_analysis_stocks_user ON analysis_stocks(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_analysis_stocks_symbol ON analysis_stocks(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_financial_statements_stock ON financial_statements(stock_id)",
        "CREATE INDEX IF NOT EXISTS idx_line_items_statement ON financial_line_items(statement_id)",
        "CREATE INDEX IF NOT EXISTS idx_stock_metrics_stock ON stock_metrics(stock_id)",
        "CREATE INDEX IF NOT EXISTS idx_valuation_models_stock ON valuation_models(stock_id)",
        "CREATE INDEX IF NOT EXISTS idx_stock_scores_stock ON stock_scores(stock_id)",
        "CREATE INDEX IF NOT EXISTS idx_pdf_uploads_stock ON pdf_uploads(stock_id)",
    ]

    try:
        for ddl in _TABLES:
            exec_sql(ddl)
    except Exception as e:
        logger.warning("⚠️  Analysis schema creation skipped: %s", e)

    _SCHEMA_INIT = True


# ── PDF file storage ─────────────────────────────────────────────────

_PDF_UPLOAD_DIR = Path(__file__).resolve().parents[2] / "uploads" / "pdfs"


def _get_pdf_dir(stock_id: int) -> Path:
    """Return (and create) the per-stock PDF directory."""
    d = _PDF_UPLOAD_DIR / str(stock_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_pdf_file(
    stock_id: int, user_id: int, pdf_bytes: bytes, original_name: str,
) -> int:
    """Save PDF to disk and record in pdf_uploads table. Returns the row id."""
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Deduplicate: skip if same file already saved for this stock
    existing = query_one(
        "SELECT id FROM pdf_uploads WHERE stock_id = ? AND pdf_hash = ?",
        (stock_id, pdf_hash),
    )
    if existing:
        return existing["id"] if isinstance(existing, dict) else existing[0]

    now = int(time.time())
    safe_name = "".join(
        c if (c.isalnum() or c in "._- ") else "_" for c in original_name
    ).strip()
    disk_name = f"{now}_{pdf_hash[:12]}_{safe_name}"

    target = _get_pdf_dir(stock_id) / disk_name
    target.write_bytes(pdf_bytes)

    exec_sql(
        """INSERT INTO pdf_uploads
           (stock_id, user_id, filename, original_name, pdf_hash, file_size, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (stock_id, user_id, disk_name, original_name, pdf_hash, len(pdf_bytes), now),
    )
    row_id = query_val(
        "SELECT id FROM pdf_uploads WHERE stock_id = ? AND pdf_hash = ?",
        (stock_id, pdf_hash),
    )
    return row_id


# ── Pydantic Schemas ─────────────────────────────────────────────────

class StockCreate(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    company_name: str = Field(..., min_length=1, max_length=200)
    exchange: str = Field("NYSE", max_length=30)
    currency: str = Field("USD", max_length=10)
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    isin: Optional[str] = None
    cik: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None
    outstanding_shares: Optional[float] = None


class StockUpdate(BaseModel):
    company_name: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    isin: Optional[str] = None
    cik: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None
    outstanding_shares: Optional[float] = None


class StatementCreate(BaseModel):
    statement_type: str = Field(..., description="income, balance, cashflow, equity")
    fiscal_year: int
    fiscal_quarter: Optional[int] = None
    period_end_date: str = Field(..., description="ISO date YYYY-MM-DD")
    filing_date: Optional[str] = None
    source_file: Optional[str] = None
    extracted_by: str = Field("manual")
    confidence_score: Optional[float] = None
    notes: Optional[str] = None
    line_items: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Array of {code, name, amount, currency?, order?, is_total?}",
    )


class DeletePeriodsRequest(BaseModel):
    periods: List[str] = Field(..., description="List of period_end_date values to delete")

class LineItemUpdate(BaseModel):
    amount: float


class MetricsCalculateRequest(BaseModel):
    period_end_date: str
    fiscal_year: int
    fiscal_quarter: Optional[int] = None


class GrahamRequest(BaseModel):
    eps: float
    book_value_per_share: float
    multiplier: float = 22.5


class DCFRequest(BaseModel):
    fcf: float
    growth_rate_stage1: float
    growth_rate_stage2: float
    discount_rate: float
    stage1_years: int = 5
    stage2_years: int = 5
    terminal_growth: float = 0.025
    shares_outstanding: float = 1.0


class DDMRequest(BaseModel):
    last_dividend: float
    growth_rate: float
    required_return: float
    high_growth_years: int = 5
    high_growth_rate: Optional[float] = None


class MultiplesRequest(BaseModel):
    metric_value: float
    peer_multiple: float
    multiple_type: str = "P/E"
    shares_outstanding: float = 1.0


# ════════════════════════════════════════════════════════════════════
# STOCK PROFILES
# ════════════════════════════════════════════════════════════════════

@router.get("/stocks")
async def list_stocks(
    search: Optional[str] = Query(None),
    current_user: TokenData = Depends(get_current_user),
):
    """List analysis stocks for the current user."""
    _ensure_schema()
    uid = current_user.user_id

    if search:
        df = query_df(
            """SELECT * FROM analysis_stocks
               WHERE user_id = ? AND (symbol LIKE ? OR company_name LIKE ?)
               ORDER BY symbol""",
            (uid, f"%{search}%", f"%{search}%"),
        )
    else:
        df = query_df(
            "SELECT * FROM analysis_stocks WHERE user_id = ? ORDER BY symbol",
            (uid,),
        )

    stocks = df.to_dict(orient="records") if not df.empty else []
    return {"status": "ok", "data": {"stocks": stocks, "count": len(stocks)}}


@router.get("/stocks/{stock_id}")
async def get_stock(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Get a single analysis stock with summary counts."""
    _ensure_schema()

    row = query_one(
        "SELECT * FROM analysis_stocks WHERE id = ? AND user_id = ?",
        (stock_id, current_user.user_id),
    )
    if not row:
        raise NotFoundError("Analysis Stock", str(stock_id))

    stock = dict(row)

    # Attach summary
    stock["statement_count"] = query_val(
        "SELECT COUNT(*) FROM financial_statements WHERE stock_id = ?",
        (stock_id,),
    ) or 0
    stock["metric_count"] = query_val(
        "SELECT COUNT(*) FROM stock_metrics WHERE stock_id = ?",
        (stock_id,),
    ) or 0
    stock["valuation_count"] = query_val(
        "SELECT COUNT(*) FROM valuation_models WHERE stock_id = ?",
        (stock_id,),
    ) or 0

    # Latest score
    score_row = query_one(
        """SELECT overall_score, fundamental_score, valuation_score,
                  growth_score, quality_score, scoring_date
           FROM stock_scores WHERE stock_id = ?
           ORDER BY scoring_date DESC LIMIT 1""",
        (stock_id,),
    )
    stock["latest_score"] = dict(score_row) if score_row else None

    return {"status": "ok", "data": stock}


@router.post("/stocks", status_code=201)
async def create_stock(
    body: StockCreate,
    current_user: TokenData = Depends(get_current_user),
):
    """Create a new analysis stock."""
    _ensure_schema()
    uid = current_user.user_id
    symbol = body.symbol.strip().upper()

    # Check uniqueness
    existing = query_val(
        "SELECT id FROM analysis_stocks WHERE user_id = ? AND symbol = ?",
        (uid, symbol),
    )
    if existing:
        raise ConflictError(f"Analysis stock '{symbol}' already exists.")

    now = int(time.time())
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO analysis_stocks
               (user_id, symbol, company_name, exchange, currency,
                sector, industry, country, isin, cik,
                description, website, outstanding_shares,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                uid, symbol, body.company_name, body.exchange, body.currency,
                body.sector, body.industry, body.country, body.isin, body.cik,
                body.description, body.website, body.outstanding_shares,
                now, now,
            ),
        )
        conn.commit()
        new_id = cur.lastrowid

    return {
        "status": "ok",
        "data": {"id": new_id, "symbol": symbol, "message": "Stock created."},
    }


@router.put("/stocks/{stock_id}")
async def update_stock(
    stock_id: int,
    body: StockUpdate,
    current_user: TokenData = Depends(get_current_user),
):
    """Update an analysis stock."""
    _ensure_schema()

    row = query_val(
        "SELECT id FROM analysis_stocks WHERE id = ? AND user_id = ?",
        (stock_id, current_user.user_id),
    )
    if not row:
        raise NotFoundError("Analysis Stock", str(stock_id))

    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise BadRequestError("No fields to update.")

    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [int(time.time()), stock_id]
    exec_sql(
        f"UPDATE analysis_stocks SET {sets}, updated_at = ? WHERE id = ?",
        tuple(vals),
    )
    return {"status": "ok", "data": {"message": "Stock updated."}}


@router.delete("/stocks/{stock_id}")
async def delete_stock(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Delete an analysis stock and all related data (cascade)."""
    _ensure_schema()

    row = query_val(
        "SELECT id FROM analysis_stocks WHERE id = ? AND user_id = ?",
        (stock_id, current_user.user_id),
    )
    if not row:
        raise NotFoundError("Analysis Stock", str(stock_id))

    with get_connection() as conn:
        cur = conn.cursor()
        # Cascade deletes
        stmt_ids = [
            r[0]
            for r in cur.execute(
                "SELECT id FROM financial_statements WHERE stock_id = ?",
                (stock_id,),
            ).fetchall()
        ]
        for sid in stmt_ids:
            cur.execute("DELETE FROM financial_line_items WHERE statement_id = ?", (sid,))
        cur.execute("DELETE FROM financial_statements WHERE stock_id = ?", (stock_id,))
        cur.execute("DELETE FROM stock_metrics WHERE stock_id = ?", (stock_id,))
        cur.execute("DELETE FROM valuation_models WHERE stock_id = ?", (stock_id,))
        cur.execute("DELETE FROM stock_scores WHERE stock_id = ?", (stock_id,))
        cur.execute("DELETE FROM analysis_stocks WHERE id = ?", (stock_id,))
        conn.commit()

    return {"status": "ok", "data": {"message": "Stock and related data deleted."}}


# ════════════════════════════════════════════════════════════════════
# FINANCIAL STATEMENTS & LINE ITEMS
# ════════════════════════════════════════════════════════════════════

@router.get("/stocks/{stock_id}/statements")
async def list_statements(
    stock_id: int,
    statement_type: Optional[str] = Query(None),
    current_user: TokenData = Depends(get_current_user),
):
    """List financial statements for a stock, with line items."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    if statement_type:
        df = query_df(
            """SELECT * FROM financial_statements
               WHERE stock_id = ? AND statement_type = ?
               ORDER BY period_end_date DESC""",
            (stock_id, statement_type),
        )
    else:
        df = query_df(
            """SELECT * FROM financial_statements
               WHERE stock_id = ?
               ORDER BY period_end_date DESC""",
            (stock_id,),
        )

    statements = df.to_dict(orient="records") if not df.empty else []

    # Attach line items to each statement
    for stmt in statements:
        items_df = query_df(
            """SELECT * FROM financial_line_items
               WHERE statement_id = ?
               ORDER BY order_index""",
            (stmt["id"],),
        )
        stmt["line_items"] = items_df.to_dict(orient="records") if not items_df.empty else []

    return {"status": "ok", "data": {"statements": statements, "count": len(statements)}}


@router.post("/stocks/{stock_id}/statements", status_code=201)
async def create_statement(
    stock_id: int,
    body: StatementCreate,
    current_user: TokenData = Depends(get_current_user),
):
    """Create or upsert a financial statement with optional line items."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    now = int(time.time())

    # Upsert: check existing by (stock_id, statement_type, period_end_date)
    existing = query_one(
        """SELECT id FROM financial_statements
           WHERE stock_id = ? AND statement_type = ? AND period_end_date = ?""",
        (stock_id, body.statement_type, body.period_end_date),
    )

    with get_connection() as conn:
        cur = conn.cursor()
        if existing:
            stmt_id = existing["id"]
            cur.execute(
                """UPDATE financial_statements
                   SET fiscal_year=?, fiscal_quarter=?, period_end_date=?,
                       filing_date=?, source_file=?, extracted_by=?,
                       confidence_score=?, notes=?, created_at=?
                   WHERE id=?""",
                (
                    body.fiscal_year, body.fiscal_quarter, body.period_end_date,
                    body.filing_date, body.source_file, body.extracted_by,
                    body.confidence_score, body.notes, now, stmt_id,
                ),
            )
            # Clear existing line items if new ones provided
            if body.line_items:
                cur.execute("DELETE FROM financial_line_items WHERE statement_id = ?", (stmt_id,))
        else:
            cur.execute(
                """INSERT INTO financial_statements
                   (stock_id, statement_type, fiscal_year, fiscal_quarter,
                    period_end_date, filing_date, source_file, extracted_by,
                    confidence_score, notes, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    stock_id, body.statement_type, body.fiscal_year,
                    body.fiscal_quarter, body.period_end_date,
                    body.filing_date, body.source_file, body.extracted_by,
                    body.confidence_score, body.notes, now,
                ),
            )
            stmt_id = cur.lastrowid

        # Insert line items
        if body.line_items:
            for idx, item in enumerate(body.line_items, 1):
                code = (item.get("code") or item.get("name", "UNKNOWN")).upper().replace(" ", "_")
                name_ = item.get("name", item.get("code", "Unknown"))
                amount_ = float(item.get("amount", 0))
                cur.execute(
                    """INSERT INTO financial_line_items
                       (statement_id, line_item_code, line_item_name,
                        amount, currency, order_index, parent_item_id, is_total)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        stmt_id, code, name_, amount_,
                        item.get("currency", "USD"),
                        item.get("order", idx),
                        item.get("parent_item_id"),
                        item.get("is_total", False),
                    ),
                )
        conn.commit()

    return {
        "status": "ok",
        "data": {"id": stmt_id, "message": "Statement saved."},
    }


@router.delete("/stocks/{stock_id}/statements/{statement_id}")
async def delete_statement(
    stock_id: int,
    statement_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Delete a financial statement and its line items."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    row = query_val(
        "SELECT id FROM financial_statements WHERE id = ? AND stock_id = ?",
        (statement_id, stock_id),
    )
    if not row:
        raise NotFoundError("Statement", str(statement_id))

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM financial_line_items WHERE statement_id = ?", (statement_id,))
        cur.execute("DELETE FROM financial_statements WHERE id = ?", (statement_id,))
        conn.commit()

    return {"status": "ok", "data": {"message": "Statement deleted."}}


@router.post("/stocks/{stock_id}/statements/delete-periods")
async def delete_statements_by_period(
    stock_id: int,
    body: DeletePeriodsRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Delete all statements (and their line items) for the given periods."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    if not body.periods:
        raise BadRequestError("No periods provided.")

    placeholders = ",".join("?" for _ in body.periods)
    rows = query_all(
        f"SELECT id FROM financial_statements WHERE stock_id = ? AND period_end_date IN ({placeholders})",
        [stock_id] + body.periods,
    )
    if not rows:
        raise NotFoundError("Statements", ", ".join(body.periods))

    stmt_ids = [r["id"] if isinstance(r, dict) else r[0] for r in rows]
    id_placeholders = ",".join("?" for _ in stmt_ids)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM financial_line_items WHERE statement_id IN ({id_placeholders})", stmt_ids)
        cur.execute(f"DELETE FROM financial_statements WHERE id IN ({id_placeholders})", stmt_ids)
        conn.commit()

    return {
        "status": "ok",
        "data": {"message": f"{len(stmt_ids)} statement(s) deleted.", "deleted_count": len(stmt_ids)},
    }


@router.put("/line-items/{item_id}")
async def update_line_item(
    item_id: int,
    body: LineItemUpdate,
    current_user: TokenData = Depends(get_current_user),
):
    """Update a single line item amount."""
    _ensure_schema()

    # Verify the line item exists AND belongs to the current user
    # by joining through financial_statements → analysis_stocks
    row = query_one(
        """SELECT li.id
           FROM financial_line_items li
           JOIN financial_statements fs ON li.statement_id = fs.id
           JOIN analysis_stocks       s  ON fs.stock_id    = s.id
           WHERE li.id = ? AND s.user_id = ?""",
        (item_id, current_user.user_id),
    )
    if not row:
        raise NotFoundError("Line Item", str(item_id))

    exec_sql(
        """UPDATE financial_line_items
           SET amount = ?, manually_edited = 1,
               edited_by_user_id = ?, edited_at = ?
           WHERE id = ?""",
        (body.amount, current_user.user_id, int(time.time()), item_id),
    )
    return {"status": "ok", "data": {"message": "Line item updated."}}


# ════════════════════════════════════════════════════════════════════
# AI-POWERED PDF UPLOAD — Gemini Vision Extraction
# ════════════════════════════════════════════════════════════════════

# Map AI statement_type → DB statement_type
_AI_STMT_TYPE_MAP = {
    "balance_sheet": "balance",
    "income_statement": "income",
    "cash_flow": "cashflow",
    "equity_statement": "equity",
    # Accept direct names too
    "income": "income",
    "balance": "balance",
    "cashflow": "cashflow",
    "equity": "equity",
}

# Standard line item codes for normalization
_STANDARD_CODES: Dict[str, str] = {
    # Income statement
    "revenue": "REVENUE", "total_revenue": "REVENUE", "net_revenue": "REVENUE",
    "sales": "REVENUE", "net_sales": "REVENUE",
    "cost_of_revenue": "COST_OF_REVENUE", "cost_of_sales": "COST_OF_REVENUE",
    "cost_of_goods_sold": "COST_OF_REVENUE", "cogs": "COST_OF_REVENUE",
    "gross_profit": "GROSS_PROFIT",
    "selling_general_administrative": "SGA", "sga": "SGA",
    "selling_general_and_administrative": "SGA",
    "research_and_development": "R&D", "r_and_d": "R&D", "r&d": "R&D",
    "operating_expenses": "OPERATING_EXPENSES", "total_operating_expenses": "OPERATING_EXPENSES",
    "operating_income": "OPERATING_INCOME", "operating_profit": "OPERATING_INCOME",
    "income_from_operations": "OPERATING_INCOME",
    "interest_expense": "INTEREST_EXPENSE", "finance_costs": "INTEREST_EXPENSE",
    "finance_cost": "INTEREST_EXPENSE",
    "other_income": "OTHER_INCOME", "other_income_expense": "OTHER_INCOME",
    "income_before_tax": "INCOME_BEFORE_TAX", "profit_before_tax": "INCOME_BEFORE_TAX",
    "income_tax": "INCOME_TAX", "income_tax_expense": "INCOME_TAX",
    "tax_expense": "INCOME_TAX", "taxation": "INCOME_TAX",
    "net_income": "NET_INCOME", "net_profit": "NET_INCOME",
    "profit_for_the_year": "NET_INCOME", "profit_for_the_period": "NET_INCOME",
    "net_income_attributable_to_shareholders": "NET_INCOME",
    "eps_basic": "EPS_BASIC", "basic_eps": "EPS_BASIC",
    "basic_earnings_per_share": "EPS_BASIC", "earnings_per_share_basic": "EPS_BASIC",
    "eps_diluted": "EPS_DILUTED", "diluted_eps": "EPS_DILUTED",
    "diluted_earnings_per_share": "EPS_DILUTED", "earnings_per_share_diluted": "EPS_DILUTED",
    "ebitda": "EBITDA",
    "depreciation_and_amortization": "DEPRECIATION_AMORTIZATION",
    "depreciation_amortization": "DEPRECIATION_AMORTIZATION",
    # Balance sheet
    "cash_and_cash_equivalents": "CASH_EQUIVALENTS", "cash_equivalents": "CASH_EQUIVALENTS",
    "cash_and_bank_balances": "CASH_EQUIVALENTS", "cash_and_balances_with_banks": "CASH_EQUIVALENTS",
    "accounts_receivable": "ACCOUNTS_RECEIVABLE", "trade_receivables": "ACCOUNTS_RECEIVABLE",
    "receivables": "ACCOUNTS_RECEIVABLE", "trade_and_other_receivables": "ACCOUNTS_RECEIVABLE",
    "inventory": "INVENTORY", "inventories": "INVENTORY",
    "other_current_assets": "OTHER_CURRENT_ASSETS",
    "total_current_assets": "TOTAL_CURRENT_ASSETS",
    "property_plant_equipment": "PPE_NET", "property_plant_and_equipment": "PPE_NET",
    "ppe_net": "PPE_NET", "fixed_assets": "PPE_NET",
    "goodwill": "GOODWILL",
    "intangible_assets": "INTANGIBLE_ASSETS", "intangibles": "INTANGIBLE_ASSETS",
    "total_non_current_assets": "TOTAL_NON_CURRENT_ASSETS",
    "total_assets": "TOTAL_ASSETS",
    "accounts_payable": "ACCOUNTS_PAYABLE", "trade_payables": "ACCOUNTS_PAYABLE",
    "trade_and_other_payables": "ACCOUNTS_PAYABLE",
    "short_term_debt": "SHORT_TERM_DEBT", "current_portion_of_debt": "SHORT_TERM_DEBT",
    "short_term_borrowings": "SHORT_TERM_DEBT",
    "total_current_liabilities": "TOTAL_CURRENT_LIABILITIES",
    "long_term_debt": "LONG_TERM_DEBT", "long_term_borrowings": "LONG_TERM_DEBT",
    "non_current_borrowings": "LONG_TERM_DEBT",
    "total_non_current_liabilities": "TOTAL_NON_CURRENT_LIABILITIES",
    "total_liabilities": "TOTAL_LIABILITIES",
    "common_stock": "COMMON_STOCK", "share_capital": "COMMON_STOCK",
    "issued_capital": "COMMON_STOCK",
    "retained_earnings": "RETAINED_EARNINGS",
    "total_equity": "TOTAL_EQUITY", "total_shareholders_equity": "TOTAL_EQUITY",
    "total_stockholders_equity": "TOTAL_EQUITY",
    "equity_attributable_to_shareholders": "TOTAL_EQUITY",
    "total_liabilities_and_equity": "TOTAL_LIABILITIES_EQUITY",
    "total_liabilities_and_shareholders_equity": "TOTAL_LIABILITIES_EQUITY",
    # Cash flow
    "net_income_cf": "NET_INCOME_CF",
    "cash_from_operations": "CASH_FROM_OPERATIONS",
    "cash_from_operating_activities": "CASH_FROM_OPERATIONS",
    "net_cash_from_operating_activities": "CASH_FROM_OPERATIONS",
    "capital_expenditures": "CAPITAL_EXPENDITURES", "capex": "CAPITAL_EXPENDITURES",
    "purchase_of_property_plant_equipment": "CAPITAL_EXPENDITURES",
    "other_investing_activities": "OTHER_INVESTING",
    "cash_from_investing": "CASH_FROM_INVESTING",
    "cash_from_investing_activities": "CASH_FROM_INVESTING",
    "net_cash_from_investing_activities": "CASH_FROM_INVESTING",
    "debt_issued": "DEBT_ISSUED", "proceeds_from_borrowings": "DEBT_ISSUED",
    "debt_repaid": "DEBT_REPAID", "repayment_of_borrowings": "DEBT_REPAID",
    "dividends_paid": "DIVIDENDS_PAID", "dividend_paid": "DIVIDENDS_PAID",
    "cash_from_financing": "CASH_FROM_FINANCING",
    "cash_from_financing_activities": "CASH_FROM_FINANCING",
    "net_cash_from_financing_activities": "CASH_FROM_FINANCING",
    "net_change_in_cash": "NET_CHANGE_CASH",
    "net_increase_decrease_in_cash": "NET_CHANGE_CASH",
    "changes_in_working_capital": "CHANGES_WORKING_CAPITAL",
    # Equity statement
    "share_premium": "SHARE_PREMIUM",
    "statutory_reserve": "STATUTORY_RESERVE",
    "voluntary_reserve": "VOLUNTARY_RESERVE",
    "general_reserve": "GENERAL_RESERVE",
    "treasury_shares": "TREASURY_SHARES_EQUITY",
    "shares_outstanding": "SHARES_DILUTED", "share_count": "SHARE_COUNT",
    "shares_basic": "SHARES_BASIC", "shares_diluted": "SHARES_DILUTED",
    # GCC / Kuwait specific
    "cost_of_operations": "COST_OF_REVENUE",
    "general_and_administrative_expenses": "GENERAL_AND_ADMIN",
    "general_and_administrative_expens": "GENERAL_AND_ADMIN",
    "selling_expenses": "SELLING_EXPENSES",
    "selling_and_distribution_expenses": "SELLING_EXPENSES",
    "finance_charges": "FINANCE_CHARGES",
    "finance_charge": "FINANCE_CHARGES",
    "profit_before_contribution_to_kfas": "PROFIT_BEFORE_DEDUCTIONS",
    "profit_before_contribution_to_kuwait_foundation_for_advancement_of_sciences": "PROFIT_BEFORE_DEDUCTIONS",
    "profit_before_contribution_to_kuwait_foundation_for_the_advancement_of_sciences": "PROFIT_BEFORE_DEDUCTIONS",
    "contribution_to_kfas": "CONTRIBUTION_KFAS",
    "contribution_to_kuwait_foundation_for_advancement_of_sciences": "CONTRIBUTION_KFAS",
    "contribution_to_kuwait_foundation_for_the_advancement_of_sciences": "CONTRIBUTION_KFAS",
    "national_labour_support_tax": "NLST", "nlst": "NLST",
    "national_labor_support_tax": "NLST",
    "zakat": "ZAKAT",
    "directors_remuneration": "DIRECTORS_REMUNERATION",
    "directors_fees": "DIRECTORS_REMUNERATION",
    "board_of_directors_remuneration": "DIRECTORS_REMUNERATION",
    "basic_and_diluted_earnings_per_share": "EPS_BASIC",
    "basic_and_diluted_earnings_per_sha": "EPS_BASIC",
    "earnings_per_share": "EPS_BASIC",
    "share_of_profit_of_associates": "SHARE_RESULTS_ASSOCIATES",
    "share_of_loss_of_associates": "SHARE_RESULTS_ASSOCIATES",
    "share_of_results_of_associates": "SHARE_RESULTS_ASSOCIATES",
    "share_of_profit_loss_of_associates": "SHARE_RESULTS_ASSOCIATES",
    # Cash flow additional
    "net_cash_used_in_operating_activities": "CASH_FROM_OPERATIONS",
    "net_cash_used_in_investing_activities": "CASH_FROM_INVESTING",
    "net_cash_used_in_financing_activities": "CASH_FROM_FINANCING",
    "cash_used_in_financing_activities": "CASH_FROM_FINANCING",
    "cash_used_in_investing_activities": "CASH_FROM_INVESTING",
    "cash_used_in_operating_activities": "CASH_FROM_OPERATIONS",
}

# Batch extraction prompt (from Streamlit ai_vision_extractor.py)
_BATCH_EXTRACT_PROMPT = """\
You are a financial statement extraction engine.
I am giving you {n_pages} page(s) from a financial report.

TASK — do BOTH classification AND extraction in a single pass:
1) Identify which financial statement each page shows:
   - balance_sheet (Statement of Financial Position / الميزانية العمومية)
   - income_statement (Profit or Loss / قائمة الدخل)
   - cash_flow (Cash Flows / التدفقات النقدية)
   - equity_statement (Changes in Equity / التغيرات في حقوق الملكية)
   Pages that belong to the SAME statement type should be merged.
2) Extract ALL line items with both year columns for each statement.
3) Detect currency and unit scale (e.g. KD, KD'000, USD millions).
4) Parentheses or negative signs → NEGATIVE numbers.
5) Dash or blank → null.
6) Do NOT invent lines that aren't visible.

CLASSIFICATION HINTS:
- "Operating Activities", "Investing Activities", "Financing Activities"
  (or Arabic: "أنشطة تشغيلية", "أنشطة استثمارية", "أنشطة تمويلية") → cash_flow
- "Total Assets", "Total Equity", "Liabilities" → balance_sheet
- "Revenue", "Net Income", "Earnings per share" → income_statement
- If the page has multiple equity component columns (Share Capital,
  Statutory Reserve, Retained Earnings, etc.) with opening/closing
  balances → equity_statement

OUTPUT THIS EXACT JSON — an ARRAY of statement objects:
[
  {{
    "statement_type": "balance_sheet",
    "source_pages": [1],
    "currency": "KWD",
    "unit_scale": 1,
    "periods": [
      {{"label": "2025-12-31", "col_name": "2025"}},
      {{"label": "2024-12-31", "col_name": "2024"}}
    ],
    "items": [
      {{
        "label_raw": "Cash and bank balances",
        "key": "cash_and_bank_balances",
        "values": {{"2025-12-31": 67007011, "2024-12-31": 74286447}},
        "is_total": false
      }}
    ]
  }}
]

RULES:
- "values" must contain numbers or null — never strings.
- Parentheses (1,234) → -1234.
- Dash or blank → null.
- Detect unit_scale: if header says "KD'000" set unit_scale=1000.
- period labels should be ISO dates if year is visible, otherwise "col_1"/"col_2".
- is_total=true for subtotals and totals.
- source_pages is 1-indexed (page 1, 2, 3…).
- If TWO pages belong to the same statement (e.g. cash flow spans 2 pages),
  merge them into ONE entry with source_pages=[2,3].
- Return ONLY the JSON array. No markdown fences, no explanation.

CRITICAL — COMPLETENESS & PRECISION:
- You MUST extract EVERY SINGLE line item visible in the statement — do NOT
  skip, summarize, or omit any row, even if it looks like a sub-item or note.
- Count the number of rows in the source image and verify your output has
  the SAME number of items (excluding blank separator rows).
- Copy every number EXACTLY as printed — do not round, estimate, or approximate.
- Double-check each extracted number against the source image before returning.
- If a line item appears with a zero value, include it with value 0 — do NOT omit it.
- Include ALL subtotals, totals, and grand totals as separate items with is_total=true.
- For statements spanning multiple pages, ensure NO line items are lost at page breaks.
"""


def _normalize_key(raw_key: str) -> str:
    """Map an AI-extracted key to a standard line item code."""
    k = raw_key.strip().lower().replace(" ", "_").replace("-", "_")
    # Direct lookup
    if k in _STANDARD_CODES:
        return _STANDARD_CODES[k]
    # Try without trailing _total, _net etc.
    for suffix in ("_total", "_net", "_and_equivalents"):
        trimmed = k.rstrip(suffix) if k.endswith(suffix) else None
        if trimmed and trimmed in _STANDARD_CODES:
            return _STANDARD_CODES[trimmed]
    # Fallback: uppercase the raw key
    return raw_key.strip().upper().replace(" ", "_").replace("-", "_")


def _parse_ai_json(text: str) -> list:
    """Parse JSON from Gemini response with multi-stage repair."""
    import re
    # Strip markdown fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fix trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    raise BadRequestError("AI returned invalid JSON. Please try again with a clearer PDF.")


@router.post("/stocks/{stock_id}/upload-statement")
async def upload_financial_statement(
    stock_id: int,
    file: UploadFile = File(...),
    force: bool = Query(False, description="Skip cache and re-extract from scratch"),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Upload a financial report PDF and extract statements using Gemini AI.

    Self-Reflective Pipeline:
    PDF → 300 DPI images → AI reasoning + extraction → arithmetic audit
    → retry on mismatches → cache result → persist to DB.
    """
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    # ── 1. Validate file ─────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise BadRequestError("Only PDF files are supported.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) < 100:
        raise BadRequestError("File is too small to be a valid PDF.")
    if len(pdf_bytes) > 50_000_000:
        raise BadRequestError("File exceeds 50 MB limit.")

    # ── 2. Get Gemini API key ────────────────────────────────────────
    from app.core.config import get_settings
    settings = get_settings()
    api_key = settings.GEMINI_API_KEY

    # Check per-user key
    try:
        from app.core.database import add_column_if_missing
        add_column_if_missing("users", "gemini_api_key", "TEXT")
        row = query_one(
            "SELECT gemini_api_key FROM users WHERE id = ?",
            (current_user.user_id,),
        )
        if row and row[0]:
            api_key = row[0]
    except Exception:
        pass

    if not api_key:
        raise BadRequestError(
            "AI extraction requires a Gemini API key. "
            "Set GEMINI_API_KEY in .env or add it in Settings."
        )

    # ── 3. Run self-reflective extraction pipeline ───────────────────
    from app.services.extraction_service import extract_financials

    try:
        result = await extract_financials(
            pdf_bytes=pdf_bytes,
            stock_id=stock_id,
            api_key=api_key,
            filename=file.filename or "upload.pdf",
            use_cache=not force,
        )
    except ValueError as exc:
        raise BadRequestError(str(exc))
    except Exception as exc:
        logger.error("Extraction pipeline error: %s", exc)
        raise BadRequestError(f"AI extraction failed: {exc}")

    # ── 4. Persist extracted data to DB ──────────────────────────────
    now = int(time.time())
    created_statements = []
    total_items = 0

    stock_row = query_one(
        "SELECT currency FROM analysis_stocks WHERE id = ?",
        (stock_id,),
    )
    stock_currency = stock_row["currency"] if stock_row else "USD"

    logger.info(
        "Persisting extraction: %d statements from pipeline",
        len(result.statements),
    )
    for _dbg_s in result.statements:
        _dbg_labels = [p.get("label", p.get("col_name", "?")) for p in _dbg_s.periods]
        _sample_keys = [it.key for it in _dbg_s.items[:3]]
        _sample_vals = {it.key: list(it.values.keys())[:4] for it in _dbg_s.items[:3]}
        logger.info(
            "  stmt type=%s  periods=%s  items=%d  sample_keys=%s  sample_val_keys=%s",
            _dbg_s.statement_type, _dbg_labels, len(_dbg_s.items), _sample_keys, _sample_vals,
        )

    with get_connection() as conn:
        cur = conn.cursor()

        for stmt in result.statements:
            currency = stmt.currency or stock_currency

            if not stmt.periods or not stmt.items:
                logger.warning(
                    "Skipping stmt type=%s: periods=%d, items=%d",
                    stmt.statement_type, len(stmt.periods), len(stmt.items),
                )
                continue

            for period_info in stmt.periods:
                period_label = period_info.get("label", "")
                col_name = period_info.get("col_name", period_label)

                # Derive fiscal year
                fiscal_year = None
                if period_label and len(period_label) >= 4:
                    try:
                        fiscal_year = int(period_label[:4])
                    except ValueError:
                        try:
                            fiscal_year = int(period_label[-4:])
                        except ValueError:
                            fiscal_year = date.today().year
                if not fiscal_year:
                    fiscal_year = date.today().year

                period_end_date = period_label
                if len(period_end_date) == 4:
                    period_end_date = f"{period_end_date}-12-31"

                # Upsert statement
                existing = query_one(
                    """SELECT id FROM financial_statements
                       WHERE stock_id = ? AND statement_type = ? AND period_end_date = ?""",
                    (stock_id, stmt.statement_type, period_end_date),
                )

                if existing:
                    stmt_id = existing["id"]
                    cur.execute(
                        """UPDATE financial_statements
                           SET fiscal_year=?, source_file=?, extracted_by=?,
                               confidence_score=?, created_at=?
                           WHERE id=?""",
                        (fiscal_year, file.filename, "gemini-ai-pipeline",
                         result.confidence, now, stmt_id),
                    )
                    cur.execute(
                        "DELETE FROM financial_line_items WHERE statement_id = ?",
                        (stmt_id,),
                    )
                else:
                    cur.execute(
                        """INSERT INTO financial_statements
                           (stock_id, statement_type, fiscal_year, fiscal_quarter,
                            period_end_date, filing_date, source_file, extracted_by,
                            confidence_score, notes, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            stock_id, stmt.statement_type, fiscal_year, None,
                            period_end_date, None, file.filename, "gemini-ai-pipeline",
                            result.confidence,
                            f"AI-extracted (pipeline, retries={result.retry_count})",
                            now,
                        ),
                    )
                    stmt_id = cur.lastrowid

                # Insert line items for this period
                _MISSING = object()
                seen_codes: set = set()
                for item in stmt.items:
                    code = _normalize_key(item.key)
                    if code in seen_codes:
                        continue  # already stored for this period
                    amount = item.values.get(period_label, _MISSING)
                    if amount is _MISSING:
                        amount = item.values.get(col_name, _MISSING)
                    if amount is _MISSING:
                        all_vals = list(item.values.values())
                        pidx = stmt.periods.index(period_info)
                        if pidx < len(all_vals):
                            amount = all_vals[pidx]
                    if amount is _MISSING:
                        continue  # not extracted for this period — skip
                    if amount is None:
                        amount = 0.0  # document showed dash/blank

                    cur.execute(
                        """INSERT INTO financial_line_items
                           (statement_id, line_item_code, line_item_name,
                            amount, currency, order_index, is_total)
                           VALUES (?,?,?,?,?,?,?)""",
                        (stmt_id, code, item.label_raw, float(amount),
                         currency, item.order_index, item.is_total),
                    )
                    seen_codes.add(code)
                    total_items += 1

                created_statements.append({
                    "statement_id": stmt_id,
                    "statement_type": stmt.statement_type,
                    "period_end_date": period_end_date,
                    "fiscal_year": fiscal_year,
                    "line_items_count": len(stmt.items),
                    "currency": currency,
                })

        conn.commit()

    # ── 4b. Save PDF file for future reference ───────────────────────
    try:
        _save_pdf_file(stock_id, current_user.user_id, pdf_bytes, file.filename or "upload.pdf")
    except Exception as e:
        logger.warning("PDF file save failed (non-fatal): %s", e)

    # ── 5. Build audit summary ───────────────────────────────────────
    audit_summary = {
        "checks_total": len(result.audit_checks),
        "checks_passed": sum(1 for c in result.audit_checks if c.passed),
        "checks_failed": sum(1 for c in result.audit_checks if not c.passed),
        "retries_used": result.retry_count,
        "validation_corrections": result.validation_corrections,
        "details": [
            {
                "statement_type": c.statement_type,
                "period": c.period,
                "rule": c.total_label,
                "expected": c.computed_sum,
                "actual": c.total_value,
                "passed": c.passed,
                "detail": c.detail,
                "discrepancy": c.discrepancy,
            }
            for c in result.audit_checks
        ],
    }

    logger.info(
        "Pipeline complete for stock %d: %d statements, %d items, "
        "confidence=%.1f%%, retries=%d, validation_corrections=%d, cached=%s",
        stock_id, len(created_statements), total_items,
        result.confidence * 100, result.retry_count,
        result.validation_corrections, result.cached,
    )

    return {
        "status": "ok",
        "data": {
            "message": (
                f"Successfully extracted {len(created_statements)} statements "
                f"with {total_items} line items."
            ),
            "statements": created_statements,
            "source_file": file.filename,
            "pages_processed": result.pages_processed,
            "model": result.model_used,
            "confidence": result.confidence,
            "cached": result.cached,
            "audit": audit_summary,
        },
    }


@router.post("/stocks/{stock_id}/validate-statement")
async def validate_financial_statement(
    stock_id: int,
    file: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Step 2: Validate previously extracted financial data.

    Sends the cached extraction + PDF back to AI for completeness check.
    Applies corrections (missing items, wrong values) and updates DB + cache.
    """
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise BadRequestError("Only PDF files are supported.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) < 100:
        raise BadRequestError("File is too small to be a valid PDF.")

    # Get Gemini API key
    from app.core.config import get_settings
    settings = get_settings()
    api_key = settings.GEMINI_API_KEY

    try:
        from app.core.database import add_column_if_missing
        add_column_if_missing("users", "gemini_api_key", "TEXT")
        row = query_one(
            "SELECT gemini_api_key FROM users WHERE id = ?",
            (current_user.user_id,),
        )
        if row and row[0]:
            api_key = row[0]
    except Exception:
        pass

    if not api_key:
        raise BadRequestError("Gemini API key required for validation.")

    # Run validation pipeline
    from app.services.extraction_service import validate_extraction

    try:
        result = await validate_extraction(
            pdf_bytes=pdf_bytes,
            stock_id=stock_id,
            api_key=api_key,
            filename=file.filename or "upload.pdf",
        )
    except ValueError as exc:
        raise BadRequestError(str(exc))
    except Exception as exc:
        logger.error("Validation pipeline error: %s", exc)
        raise BadRequestError(f"AI validation failed: {exc}")

    if result.validation_corrections == 0:
        return {
            "status": "ok",
            "data": {
                "message": "Validation complete — no corrections needed.",
                "corrections_applied": 0,
                "confidence": result.confidence,
            },
        }

    # Re-persist corrected data to DB
    now = int(time.time())
    updated_statements = []
    total_items = 0

    stock_row = query_one(
        "SELECT currency FROM analysis_stocks WHERE id = ?",
        (stock_id,),
    )
    stock_currency = stock_row["currency"] if stock_row else "USD"

    with get_connection() as conn:
        cur = conn.cursor()

        for stmt in result.statements:
            currency = stmt.currency or stock_currency

            if not stmt.periods or not stmt.items:
                continue

            for period_info in stmt.periods:
                period_label = period_info.get("label", "")
                col_name = period_info.get("col_name", period_label)

                fiscal_year = None
                if period_label and len(period_label) >= 4:
                    try:
                        fiscal_year = int(period_label[:4])
                    except ValueError:
                        try:
                            fiscal_year = int(period_label[-4:])
                        except ValueError:
                            fiscal_year = date.today().year
                if not fiscal_year:
                    fiscal_year = date.today().year

                period_end_date = period_label
                if len(period_end_date) == 4:
                    period_end_date = f"{period_end_date}-12-31"

                # Upsert statement
                existing = query_one(
                    """SELECT id FROM financial_statements
                       WHERE stock_id = ? AND statement_type = ? AND period_end_date = ?""",
                    (stock_id, stmt.statement_type, period_end_date),
                )

                if existing:
                    stmt_id = existing["id"]
                    cur.execute(
                        """UPDATE financial_statements
                           SET confidence_score=?, created_at=?
                           WHERE id=?""",
                        (result.confidence, now, stmt_id),
                    )
                    cur.execute(
                        "DELETE FROM financial_line_items WHERE statement_id = ?",
                        (stmt_id,),
                    )
                else:
                    cur.execute(
                        """INSERT INTO financial_statements
                           (stock_id, statement_type, fiscal_year, fiscal_quarter,
                            period_end_date, filing_date, source_file, extracted_by,
                            confidence_score, notes, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            stock_id, stmt.statement_type, fiscal_year, None,
                            period_end_date, None, file.filename,
                            "gemini-ai-validated",
                            result.confidence,
                            f"AI-validated (corrections={result.validation_corrections})",
                            now,
                        ),
                    )
                    stmt_id = cur.lastrowid

                _MISSING = object()
                seen_codes: set = set()
                for item in stmt.items:
                    code = _normalize_key(item.key)
                    if code in seen_codes:
                        continue
                    amount = item.values.get(period_label, _MISSING)
                    if amount is _MISSING:
                        amount = item.values.get(col_name, _MISSING)
                    if amount is _MISSING:
                        all_vals = list(item.values.values())
                        pidx = stmt.periods.index(period_info)
                        if pidx < len(all_vals):
                            amount = all_vals[pidx]
                    if amount is _MISSING:
                        continue
                    if amount is None:
                        amount = 0.0

                    cur.execute(
                        """INSERT INTO financial_line_items
                           (statement_id, line_item_code, line_item_name,
                            amount, currency, order_index, is_total)
                           VALUES (?,?,?,?,?,?,?)""",
                        (stmt_id, code, item.label_raw, float(amount),
                         currency, item.order_index, item.is_total),
                    )
                    seen_codes.add(code)
                    total_items += 1

                updated_statements.append({
                    "statement_id": stmt_id,
                    "statement_type": stmt.statement_type,
                    "period_end_date": period_end_date,
                    "fiscal_year": fiscal_year,
                    "line_items_count": len(stmt.items),
                    "currency": currency,
                })

        conn.commit()

    logger.info(
        "Validation persisted for stock %d: %d corrections, %d statements, %d items",
        stock_id, result.validation_corrections,
        len(updated_statements), total_items,
    )

    return {
        "status": "ok",
        "data": {
            "message": (
                f"Validation complete — {result.validation_corrections} corrections "
                f"applied across {len(updated_statements)} statements."
            ),
            "corrections_applied": result.validation_corrections,
            "statements": updated_statements,
            "confidence": result.confidence,
        },
    }


@router.post("/stocks/{stock_id}/verify-placement")
async def verify_statement_placement(
    stock_id: int,
    file: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Step 3: Verify every line item is placed in the correct statement type
    with the correct key and is_total flag.
    """
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise BadRequestError("Only PDF files are supported.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) < 100:
        raise BadRequestError("File is too small to be a valid PDF.")

    # Get Gemini API key
    from app.core.config import get_settings
    settings = get_settings()
    api_key = settings.GEMINI_API_KEY

    try:
        from app.core.database import add_column_if_missing
        add_column_if_missing("users", "gemini_api_key", "TEXT")
        row = query_one(
            "SELECT gemini_api_key FROM users WHERE id = ?",
            (current_user.user_id,),
        )
        if row and row[0]:
            api_key = row[0]
    except Exception:
        pass

    if not api_key:
        raise BadRequestError("Gemini API key required for placement verification.")

    from app.services.extraction_service import verify_placement

    try:
        result = await verify_placement(
            pdf_bytes=pdf_bytes,
            stock_id=stock_id,
            api_key=api_key,
            filename=file.filename or "upload.pdf",
        )
    except ValueError as exc:
        raise BadRequestError(str(exc))
    except Exception as exc:
        logger.error("Placement verification error: %s", exc)
        raise BadRequestError(f"Placement verification failed: {exc}")

    if result.placement_corrections == 0:
        return {
            "status": "ok",
            "data": {
                "message": "Placement verified — all items correctly placed.",
                "corrections_applied": 0,
                "confidence": result.confidence,
            },
        }

    # Re-persist corrected data to DB
    now = int(time.time())
    updated_statements = []
    total_items = 0

    stock_row = query_one(
        "SELECT currency FROM analysis_stocks WHERE id = ?",
        (stock_id,),
    )
    stock_currency = stock_row["currency"] if stock_row else "USD"

    with get_connection() as conn:
        cur = conn.cursor()

        for stmt in result.statements:
            currency = stmt.currency or stock_currency

            if not stmt.periods or not stmt.items:
                continue

            for period_info in stmt.periods:
                period_label = period_info.get("label", "")
                col_name = period_info.get("col_name", period_label)

                fiscal_year = None
                if period_label and len(period_label) >= 4:
                    try:
                        fiscal_year = int(period_label[:4])
                    except ValueError:
                        try:
                            fiscal_year = int(period_label[-4:])
                        except ValueError:
                            fiscal_year = date.today().year
                if not fiscal_year:
                    fiscal_year = date.today().year

                period_end_date = period_label
                if len(period_end_date) == 4:
                    period_end_date = f"{period_end_date}-12-31"

                existing = query_one(
                    """SELECT id FROM financial_statements
                       WHERE stock_id = ? AND statement_type = ? AND period_end_date = ?""",
                    (stock_id, stmt.statement_type, period_end_date),
                )

                if existing:
                    stmt_id = existing["id"]
                    cur.execute(
                        """UPDATE financial_statements
                           SET confidence_score=?, extracted_by=?, created_at=?
                           WHERE id=?""",
                        (result.confidence, "gemini-ai-verified", now, stmt_id),
                    )
                    cur.execute(
                        "DELETE FROM financial_line_items WHERE statement_id = ?",
                        (stmt_id,),
                    )
                else:
                    cur.execute(
                        """INSERT INTO financial_statements
                           (stock_id, statement_type, fiscal_year, fiscal_quarter,
                            period_end_date, filing_date, source_file, extracted_by,
                            confidence_score, notes, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            stock_id, stmt.statement_type, fiscal_year, None,
                            period_end_date, None, file.filename,
                            "gemini-ai-verified",
                            result.confidence,
                            f"AI-verified (placement={result.placement_corrections})",
                            now,
                        ),
                    )
                    stmt_id = cur.lastrowid

                for item in stmt.items:
                    code = _normalize_key(item.key)
                    _MISSING = object()
                    amount = item.values.get(period_label, _MISSING)
                    if amount is _MISSING:
                        amount = item.values.get(col_name, _MISSING)
                    if amount is _MISSING:
                        all_vals = list(item.values.values())
                        pidx = stmt.periods.index(period_info)
                        if pidx < len(all_vals):
                            amount = all_vals[pidx]
                    if amount is _MISSING:
                        continue
                    if amount is None:
                        amount = 0.0

                    cur.execute(
                        """INSERT INTO financial_line_items
                           (statement_id, line_item_code, line_item_name,
                            amount, currency, order_index, is_total)
                           VALUES (?,?,?,?,?,?,?)""",
                        (stmt_id, code, item.label_raw, float(amount),
                         currency, item.order_index, item.is_total),
                    )
                    total_items += 1

                updated_statements.append({
                    "statement_id": stmt_id,
                    "statement_type": stmt.statement_type,
                    "period_end_date": period_end_date,
                    "fiscal_year": fiscal_year,
                    "line_items_count": len(stmt.items),
                    "currency": currency,
                })

        conn.commit()

    logger.info(
        "Placement persisted for stock %d: %d corrections, %d statements, %d items",
        stock_id, result.placement_corrections,
        len(updated_statements), total_items,
    )

    return {
        "status": "ok",
        "data": {
            "message": (
                f"Placement verified — {result.placement_corrections} corrections "
                f"applied across {len(updated_statements)} statements."
            ),
            "corrections_applied": result.placement_corrections,
            "statements": updated_statements,
            "confidence": result.confidence,
        },
    }


@router.post("/stocks/{stock_id}/ai-attribute")
async def ai_attribute_statement(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Step 4: AI Attribution Expert — user-triggered.

    Uses the cached extraction result (no PDF re-upload needed).
    Sends extracted data to AI for expert review of item attribution:
    statement-type placement, key naming, is_total flags, value signs,
    and cross-statement consistency.
    """
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    # Get Gemini API key
    from app.core.config import get_settings
    settings = get_settings()
    api_key = settings.GEMINI_API_KEY

    try:
        from app.core.database import add_column_if_missing
        add_column_if_missing("users", "gemini_api_key", "TEXT")
        row = query_one(
            "SELECT gemini_api_key FROM users WHERE id = ?",
            (current_user.user_id,),
        )
        if row and row[0]:
            api_key = row[0]
    except Exception:
        pass

    if not api_key:
        raise BadRequestError("Gemini API key required for AI attribution.")

    from app.services.extraction_service import ai_attribute_extraction

    try:
        result = await ai_attribute_extraction(
            stock_id=stock_id,
            api_key=api_key,
        )
    except ValueError as exc:
        raise BadRequestError(str(exc))
    except Exception as exc:
        logger.error("AI attribution error: %s", exc)
        raise BadRequestError(f"AI attribution failed: {exc}")

    if result.validation_corrections == 0:
        return {
            "status": "ok",
            "data": {
                "message": "AI attribution reviewed — all items correctly attributed.",
                "corrections_applied": 0,
                "confidence": result.confidence,
            },
        }

    # Re-persist corrected data to DB
    now = int(time.time())
    updated_statements = []
    total_items = 0

    stock_row = query_one(
        "SELECT currency FROM analysis_stocks WHERE id = ?",
        (stock_id,),
    )
    stock_currency = stock_row["currency"] if stock_row else "USD"

    with get_connection() as conn:
        cur = conn.cursor()

        for stmt in result.statements:
            currency = stmt.currency or stock_currency

            if not stmt.periods or not stmt.items:
                continue

            for period_info in stmt.periods:
                period_label = period_info.get("label", "")
                col_name = period_info.get("col_name", period_label)

                fiscal_year = None
                if period_label and len(period_label) >= 4:
                    try:
                        fiscal_year = int(period_label[:4])
                    except ValueError:
                        try:
                            fiscal_year = int(period_label[-4:])
                        except ValueError:
                            fiscal_year = date.today().year
                if not fiscal_year:
                    fiscal_year = date.today().year

                period_end_date = period_label
                if len(period_end_date) == 4:
                    period_end_date = f"{period_end_date}-12-31"

                existing = query_one(
                    """SELECT id FROM financial_statements
                       WHERE stock_id = ? AND statement_type = ? AND period_end_date = ?""",
                    (stock_id, stmt.statement_type, period_end_date),
                )

                if existing:
                    stmt_id = existing["id"]
                    cur.execute(
                        """UPDATE financial_statements
                           SET confidence_score=?, extracted_by=?, created_at=?
                           WHERE id=?""",
                        (result.confidence, "gemini-ai-attributed", now, stmt_id),
                    )
                    cur.execute(
                        "DELETE FROM financial_line_items WHERE statement_id = ?",
                        (stmt_id,),
                    )
                else:
                    cur.execute(
                        """INSERT INTO financial_statements
                           (stock_id, statement_type, fiscal_year, fiscal_quarter,
                            period_end_date, filing_date, source_file, extracted_by,
                            confidence_score, notes, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            stock_id, stmt.statement_type, fiscal_year, None,
                            period_end_date, None, None,
                            "gemini-ai-attributed",
                            result.confidence,
                            f"AI-attributed (corrections={result.validation_corrections})",
                            now,
                        ),
                    )
                    stmt_id = cur.lastrowid

                for item in stmt.items:
                    code = _normalize_key(item.key)
                    _MISSING = object()
                    amount = item.values.get(period_label, _MISSING)
                    if amount is _MISSING:
                        amount = item.values.get(col_name, _MISSING)
                    if amount is _MISSING:
                        all_vals = list(item.values.values())
                        pidx = stmt.periods.index(period_info)
                        if pidx < len(all_vals):
                            amount = all_vals[pidx]
                    if amount is _MISSING:
                        continue
                    if amount is None:
                        amount = 0.0

                    cur.execute(
                        """INSERT INTO financial_line_items
                           (statement_id, line_item_code, line_item_name,
                            amount, currency, order_index, is_total)
                           VALUES (?,?,?,?,?,?,?)""",
                        (stmt_id, code, item.label_raw, float(amount),
                         currency, item.order_index, item.is_total),
                    )
                    total_items += 1

                updated_statements.append({
                    "statement_id": stmt_id,
                    "statement_type": stmt.statement_type,
                    "period_end_date": period_end_date,
                    "fiscal_year": fiscal_year,
                    "line_items_count": len(stmt.items),
                    "currency": currency,
                })

        conn.commit()

    logger.info(
        "Attribution persisted for stock %d: %d corrections, %d statements, %d items",
        stock_id, result.validation_corrections,
        len(updated_statements), total_items,
    )

    return {
        "status": "ok",
        "data": {
            "message": (
                f"AI attribution applied — {result.validation_corrections} corrections "
                f"across {len(updated_statements)} statements."
            ),
            "corrections_applied": result.validation_corrections,
            "statements": updated_statements,
            "confidence": result.confidence,
        },
    }


@router.delete("/stocks/{stock_id}/extraction-cache")
async def clear_extraction_cache(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Clear cached extraction results so next upload re-runs the AI pipeline."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    from app.services.extraction_service import _ensure_cache_table
    _ensure_cache_table()

    exec_sql(
        "DELETE FROM extraction_cache WHERE stock_id = ?",
        (stock_id,),
    )

    return {"status": "ok", "data": {"message": "Extraction cache cleared."}}


# ════════════════════════════════════════════════════════════════════
# METRICS & RATIOS
# ════════════════════════════════════════════════════════════════════

@router.get("/stocks/{stock_id}/metrics")
async def get_metrics(
    stock_id: int,
    metric_type: Optional[str] = Query(None),
    current_user: TokenData = Depends(get_current_user),
):
    """Get stored metrics for a stock."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    if metric_type:
        df = query_df(
            """SELECT * FROM stock_metrics
               WHERE stock_id = ? AND metric_type = ?
               ORDER BY period_end_date DESC""",
            (stock_id, metric_type),
        )
    else:
        df = query_df(
            "SELECT * FROM stock_metrics WHERE stock_id = ? ORDER BY period_end_date DESC",
            (stock_id,),
        )

    metrics = df.to_dict(orient="records") if not df.empty else []

    # Group by category for frontend convenience
    grouped: Dict[str, List[Dict]] = {}
    for m in metrics:
        cat = m.get("metric_type", "other")
        grouped.setdefault(cat, []).append(m)

    return {"status": "ok", "data": {"metrics": metrics, "grouped": grouped, "count": len(metrics)}}


@router.post("/stocks/{stock_id}/metrics/calculate")
async def calculate_metrics(
    stock_id: int,
    body: MetricsCalculateRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Calculate all financial metrics for a given period and persist."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    results = _calculate_all_metrics(
        stock_id,
        body.period_end_date,
        body.fiscal_year,
        body.fiscal_quarter,
    )

    if not results:
        raise BadRequestError("No line items found for that period. Upload statements first.")

    return {"status": "ok", "data": {"metrics": results}}


# ════════════════════════════════════════════════════════════════════
# GROWTH ANALYSIS
# ════════════════════════════════════════════════════════════════════

@router.get("/stocks/{stock_id}/growth")
async def get_growth(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Get YoY growth rates for key items across all periods."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    growth = _calculate_growth(stock_id)
    return {"status": "ok", "data": {"growth": growth}}


# ════════════════════════════════════════════════════════════════════
# STOCK SCORE
# ════════════════════════════════════════════════════════════════════

@router.get("/stocks/{stock_id}/score")
async def get_score(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Get the latest stored score and/or compute a new one."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    score = _compute_stock_score(stock_id, current_user.user_id)
    return {"status": "ok", "data": score}


@router.get("/stocks/{stock_id}/scores/history")
async def get_score_history(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Get all historical scores for a stock."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    df = query_df(
        "SELECT * FROM stock_scores WHERE stock_id = ? ORDER BY scoring_date DESC",
        (stock_id,),
    )
    scores = df.to_dict(orient="records") if not df.empty else []
    # Parse JSON details
    for s in scores:
        if isinstance(s.get("details"), str):
            try:
                s["details"] = json.loads(s["details"])
            except (json.JSONDecodeError, TypeError):
                pass

    return {"status": "ok", "data": {"scores": scores, "count": len(scores)}}


# ════════════════════════════════════════════════════════════════════
# VALUATION MODELS
# ════════════════════════════════════════════════════════════════════

@router.get("/stocks/{stock_id}/valuations")
async def get_valuations(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Get saved valuation results for a stock."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    df = query_df(
        "SELECT * FROM valuation_models WHERE stock_id = ? ORDER BY valuation_date DESC",
        (stock_id,),
    )
    valuations = df.to_dict(orient="records") if not df.empty else []
    for v in valuations:
        for field in ("parameters", "assumptions"):
            if isinstance(v.get(field), str):
                try:
                    v[field] = json.loads(v[field])
                except (json.JSONDecodeError, TypeError):
                    pass

    return {"status": "ok", "data": {"valuations": valuations, "count": len(valuations)}}


@router.post("/stocks/{stock_id}/valuations/graham")
async def run_graham(
    stock_id: int,
    body: GrahamRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Run Graham Number valuation."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    result = _graham_number(body.eps, body.book_value_per_share, body.multiplier)
    _save_valuation(stock_id, result, current_user.user_id)
    return {"status": "ok", "data": result}


@router.post("/stocks/{stock_id}/valuations/dcf")
async def run_dcf(
    stock_id: int,
    body: DCFRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Run Two-Stage DCF valuation."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    result = _dcf(
        body.fcf, body.growth_rate_stage1, body.growth_rate_stage2,
        body.discount_rate, body.stage1_years, body.stage2_years,
        body.terminal_growth, body.shares_outstanding,
    )
    _save_valuation(stock_id, result, current_user.user_id)
    return {"status": "ok", "data": result}


@router.post("/stocks/{stock_id}/valuations/ddm")
async def run_ddm(
    stock_id: int,
    body: DDMRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Run Dividend Discount Model."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    result = _ddm(
        body.last_dividend, body.growth_rate, body.required_return,
        body.high_growth_years, body.high_growth_rate,
    )
    _save_valuation(stock_id, result, current_user.user_id)
    return {"status": "ok", "data": result}


@router.post("/stocks/{stock_id}/valuations/multiples")
async def run_multiples(
    stock_id: int,
    body: MultiplesRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Run Comparable Multiples valuation."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    result = _comparable_multiples(
        body.metric_value, body.peer_multiple,
        body.multiple_type, body.shares_outstanding,
    )
    _save_valuation(stock_id, result, current_user.user_id)
    return {"status": "ok", "data": result}


# ════════════════════════════════════════════════════════════════════
# PDF FILE MANAGEMENT
# ════════════════════════════════════════════════════════════════════

@router.get("/stocks/{stock_id}/pdfs")
async def list_stock_pdfs(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """List all saved PDF files for a stock."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    rows = query_all(
        """SELECT id, original_name, file_size, created_at
           FROM pdf_uploads
           WHERE stock_id = ? AND user_id = ?
           ORDER BY created_at DESC""",
        (stock_id, current_user.user_id),
    )
    pdfs = []
    for r in rows:
        if isinstance(r, dict):
            pdfs.append(r)
        else:
            pdfs.append({
                "id": r[0],
                "original_name": r[1],
                "file_size": r[2],
                "created_at": r[3],
            })
    return {"status": "ok", "data": pdfs}


@router.get("/stocks/{stock_id}/pdfs/{pdf_id}/download")
async def download_stock_pdf(
    stock_id: int,
    pdf_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Download a previously uploaded PDF."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    row = query_one(
        """SELECT filename, original_name
           FROM pdf_uploads
           WHERE id = ? AND stock_id = ? AND user_id = ?""",
        (pdf_id, stock_id, current_user.user_id),
    )
    if not row:
        raise NotFoundError("PDF Upload", str(pdf_id))

    disk_name = row["filename"] if isinstance(row, dict) else row[0]
    original = row["original_name"] if isinstance(row, dict) else row[1]
    path = _get_pdf_dir(stock_id) / disk_name

    if not path.is_file():
        raise NotFoundError("PDF file on disk", str(pdf_id))

    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=original,
    )


@router.delete("/stocks/{stock_id}/pdfs/{pdf_id}")
async def delete_stock_pdf(
    stock_id: int,
    pdf_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Delete a saved PDF file."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    row = query_one(
        """SELECT filename FROM pdf_uploads
           WHERE id = ? AND stock_id = ? AND user_id = ?""",
        (pdf_id, stock_id, current_user.user_id),
    )
    if not row:
        raise NotFoundError("PDF Upload", str(pdf_id))

    disk_name = row["filename"] if isinstance(row, dict) else row[0]
    path = _get_pdf_dir(stock_id) / disk_name
    if path.is_file():
        path.unlink()

    exec_sql("DELETE FROM pdf_uploads WHERE id = ?", (pdf_id,))
    return {"status": "ok", "message": "PDF deleted."}


# ════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ════════════════════════════════════════════════════════════════════

def _verify_stock_owner(stock_id: int, user_id: int) -> None:
    row = query_val(
        "SELECT id FROM analysis_stocks WHERE id = ? AND user_id = ?",
        (stock_id, user_id),
    )
    if not row:
        raise NotFoundError("Analysis Stock", str(stock_id))


# ── Metrics calculation (mirrors MetricsCalculator) ──────────────────

def _load_items_for_period(stock_id: int, period_end_date: str) -> Dict[str, float]:
    """Flatten all line items across all statement types for one period."""
    rows = query_all(
        """SELECT li.line_item_code, li.amount
           FROM financial_line_items li
           JOIN financial_statements fs ON fs.id = li.statement_id
           WHERE fs.stock_id = ? AND fs.period_end_date = ?""",
        (stock_id, period_end_date),
    )
    return {r[0] if isinstance(r, (tuple, list)) else r["line_item_code"]: (r[1] if isinstance(r, (tuple, list)) else r["amount"]) for r in rows}


def _upsert_metric(
    stock_id: int, fiscal_year: int, period_end_date: str,
    metric_type: str, metric_name: str, metric_value: float,
    fiscal_quarter: Optional[int] = None,
) -> None:
    now = int(time.time())
    existing = query_val(
        "SELECT id FROM stock_metrics WHERE stock_id = ? AND metric_name = ? AND period_end_date = ?",
        (stock_id, metric_name, period_end_date),
    )
    if existing:
        exec_sql(
            """UPDATE stock_metrics
               SET fiscal_year=?, fiscal_quarter=?, metric_type=?, metric_value=?, created_at=?
               WHERE id=?""",
            (fiscal_year, fiscal_quarter, metric_type, metric_value, now, existing),
        )
    else:
        exec_sql(
            """INSERT INTO stock_metrics
               (stock_id, fiscal_year, fiscal_quarter, period_end_date,
                metric_type, metric_name, metric_value, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (stock_id, fiscal_year, fiscal_quarter, period_end_date,
             metric_type, metric_name, metric_value, now),
        )


def _calculate_all_metrics(
    stock_id: int, period_end_date: str, fiscal_year: int,
    fiscal_quarter: Optional[int] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Calculate every metric for one period (mirrors MetricsCalculator)."""
    items = _load_items_for_period(stock_id, period_end_date)
    if not items:
        return {}

    def _get(code: str) -> Optional[float]:
        return items.get(code)

    results: Dict[str, Dict[str, Optional[float]]] = {}

    # ── profitability
    prof: Dict[str, Optional[float]] = {}
    revenue = _get("REVENUE")
    gross_profit = _get("GROSS_PROFIT")
    operating_income = _get("OPERATING_INCOME")
    net_income = _get("NET_INCOME")
    total_assets = _get("TOTAL_ASSETS")
    total_equity = _get("TOTAL_EQUITY")

    if revenue and revenue != 0:
        if gross_profit is not None:
            prof["Gross Margin"] = gross_profit / revenue
        if operating_income is not None:
            prof["Operating Margin"] = operating_income / revenue
        if net_income is not None:
            prof["Net Margin"] = net_income / revenue
    if total_assets and total_assets != 0 and net_income is not None:
        prof["ROA"] = net_income / total_assets
    if total_equity and total_equity != 0 and net_income is not None:
        prof["ROE"] = net_income / total_equity
    if all(v is not None and v != 0 for v in [net_income, revenue, total_assets, total_equity]):
        prof["DuPont ROE"] = (net_income / revenue) * (revenue / total_assets) * (total_assets / total_equity)

    ebitda = _get("EBITDA")
    if ebitda is None:
        da = _get("DEPRECIATION_AMORTIZATION")
        if operating_income is not None and da is not None:
            ebitda = operating_income + abs(da)
    if ebitda is not None and revenue and revenue != 0:
        prof["EBITDA Margin"] = ebitda / revenue
    results["profitability"] = prof

    # ── liquidity
    liq: Dict[str, Optional[float]] = {}
    current_assets = _get("TOTAL_CURRENT_ASSETS")
    current_liab = _get("TOTAL_CURRENT_LIABILITIES")
    inventory = _get("INVENTORY")
    cash = _get("CASH_EQUIVALENTS")
    if current_assets is not None and current_liab and current_liab != 0:
        liq["Current Ratio"] = current_assets / current_liab
        if inventory is not None:
            liq["Quick Ratio"] = (current_assets - inventory) / current_liab
        if cash is not None:
            liq["Cash Ratio"] = cash / current_liab
    results["liquidity"] = liq

    # ── leverage
    lev: Dict[str, Optional[float]] = {}
    total_liab = _get("TOTAL_LIABILITIES")
    lt_debt = _get("LONG_TERM_DEBT")
    st_debt = _get("SHORT_TERM_DEBT")
    interest_expense = _get("INTEREST_EXPENSE")
    if total_liab is not None and total_equity and total_equity != 0:
        lev["Debt-to-Equity"] = total_liab / total_equity
    if total_liab is not None and total_assets and total_assets != 0:
        lev["Debt-to-Assets"] = total_liab / total_assets
    total_debt = (lt_debt or 0) + (st_debt or 0)
    if total_debt and total_equity and total_equity != 0:
        lev["Total Debt / Equity"] = total_debt / total_equity
    if ebitda and ebitda != 0 and total_debt:
        lev["Debt / EBITDA"] = total_debt / ebitda
    if interest_expense and interest_expense != 0 and operating_income is not None:
        lev["Interest Coverage"] = operating_income / abs(interest_expense)
    if total_assets and total_equity and total_equity != 0:
        lev["Equity Multiplier"] = total_assets / total_equity
    results["leverage"] = lev

    # ── efficiency
    eff: Dict[str, Optional[float]] = {}
    ar = _get("ACCOUNTS_RECEIVABLE")
    ap = _get("ACCOUNTS_PAYABLE")
    cogs = _get("COST_OF_REVENUE")
    if revenue and total_assets and total_assets != 0:
        eff["Asset Turnover"] = revenue / total_assets
    if revenue and ar and ar != 0:
        eff["Receivables Turnover"] = revenue / ar
        eff["Days Sales Outstanding"] = 365.0 / (revenue / ar)
    if cogs and inventory and inventory != 0:
        eff["Inventory Turnover"] = cogs / inventory
        eff["Days Inventory"] = 365.0 / (cogs / inventory)
    if cogs and ap and ap != 0:
        eff["Payables Turnover"] = cogs / ap
        eff["Days Payable"] = 365.0 / (cogs / ap)
    dso = eff.get("Days Sales Outstanding")
    dio = eff.get("Days Inventory")
    dpo = eff.get("Days Payable")
    if all(v is not None for v in [dso, dio, dpo]):
        eff["Cash Conversion Cycle"] = dso + dio - dpo
    results["efficiency"] = eff

    # ── valuation (per-share)
    val: Dict[str, Optional[float]] = {}
    eps = _get("EPS_DILUTED") or _get("EPS_BASIC")
    shares = _get("SHARES_DILUTED") or _get("SHARES_BASIC") or _get("SHARE_COUNT")
    if total_equity is not None and shares and shares != 0:
        val["Book Value / Share"] = total_equity / shares
    if eps is not None:
        val["EPS"] = eps
    dividends_paid = _get("DIVIDENDS_PAID")
    if dividends_paid is not None and shares and shares != 0:
        dps = abs(dividends_paid) / shares
        val["Dividends / Share"] = dps
        if eps and eps != 0:
            val["Payout Ratio"] = dps / eps
    results["valuation"] = val

    # ── cash flow
    cfm: Dict[str, Optional[float]] = {}
    cfo = _get("CASH_FROM_OPERATIONS")
    capex = _get("CAPITAL_EXPENDITURES") or _get("CAPEX")
    fcf = _get("FCF")
    if fcf is None and cfo is not None and capex is not None:
        fcf = cfo - abs(capex)
    if fcf is not None:
        cfm["Free Cash Flow"] = fcf
        if revenue and revenue != 0:
            cfm["FCF Margin"] = fcf / revenue
        if shares and shares != 0:
            cfm["FCF / Share"] = fcf / shares
    if cfo is not None and net_income and net_income != 0:
        cfm["CFO / Net Income"] = cfo / net_income
    results["cashflow"] = cfm

    # ── persist
    for category, metrics in results.items():
        for name, value in metrics.items():
            if value is not None:
                _upsert_metric(stock_id, fiscal_year, period_end_date, category, name, value, fiscal_quarter)

    return results


# ── Growth calculation ───────────────────────────────────────────────

def _calculate_growth(stock_id: int) -> Dict[str, List[Dict[str, Any]]]:
    growth: Dict[str, List[Dict[str, Any]]] = {}
    growth_items = [
        ("REVENUE", "Revenue Growth", "income"),
        ("NET_INCOME", "Net Income Growth", "income"),
        ("EPS_DILUTED", "EPS Growth", "income"),
        ("TOTAL_ASSETS", "Total Assets Growth", "balance"),
        ("CASH_FROM_OPERATIONS", "CFO Growth", "cashflow"),
    ]
    for code, label, stmt_type in growth_items:
        rows = query_all(
            """SELECT fs.period_end_date AS period, fs.fiscal_year, li.amount
               FROM financial_line_items li
               JOIN financial_statements fs ON fs.id = li.statement_id
               WHERE fs.stock_id = ? AND fs.statement_type = ? AND li.line_item_code = ?
               ORDER BY fs.period_end_date""",
            (stock_id, stmt_type, code),
        )
        if len(rows) < 2:
            continue
        periods = []
        for r in rows:
            if isinstance(r, (tuple, list)):
                periods.append({"period": r[0], "fiscal_year": r[1], "amount": r[2]})
            else:
                periods.append({"period": r["period"], "fiscal_year": r["fiscal_year"], "amount": r["amount"]})

        rates: List[Dict[str, Any]] = []
        for i in range(1, len(periods)):
            prev = periods[i - 1]
            curr = periods[i]
            if prev["amount"] and prev["amount"] != 0:
                g = (curr["amount"] - prev["amount"]) / abs(prev["amount"])
                rates.append({
                    "period": curr["period"],
                    "prev_period": prev["period"],
                    "growth": round(g, 4),
                })
                _upsert_metric(
                    stock_id, curr.get("fiscal_year", 0), curr["period"],
                    "growth", label, round(g, 4),
                )
        if rates:
            growth[label] = rates
    return growth


# ── Score calculation (mirrors MetricsCalculator.compute_stock_score)

def _compute_stock_score(stock_id: int, user_id: int) -> Dict[str, Any]:
    rows = query_all(
        "SELECT metric_name, metric_value FROM stock_metrics WHERE stock_id = ? ORDER BY period_end_date DESC",
        (stock_id,),
    )
    if not rows:
        return {"overall_score": None, "error": "No metrics available. Calculate metrics first."}

    latest: Dict[str, float] = {}
    for r in rows:
        name = r[0] if isinstance(r, (tuple, list)) else r["metric_name"]
        val = r[1] if isinstance(r, (tuple, list)) else r["metric_value"]
        if name not in latest:
            latest[name] = val

    fund = _score_fundamentals(latest)
    val = _score_valuation(latest)
    growth = _score_growth(latest)
    quality = _score_quality(latest)
    overall = fund * 0.30 + val * 0.25 + growth * 0.25 + quality * 0.20

    result = {
        "overall_score": round(overall, 1),
        "fundamental_score": round(fund, 1),
        "valuation_score": round(val, 1),
        "growth_score": round(growth, 1),
        "quality_score": round(quality, 1),
        "details": latest,
    }

    # Persist
    now = int(time.time())
    exec_sql(
        """INSERT INTO stock_scores
           (stock_id, scoring_date, overall_score, fundamental_score,
            valuation_score, growth_score, quality_score, details,
            created_by_user_id, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            stock_id, date.today().isoformat(), result["overall_score"],
            result["fundamental_score"], result["valuation_score"],
            result["growth_score"], result["quality_score"],
            json.dumps(latest), user_id, now,
        ),
    )
    return result


def _score_fundamentals(m: Dict[str, float]) -> float:
    score = 50.0
    roe = m.get("ROE")
    if roe is not None:
        if roe > 0.20:
            score += 15
        elif roe > 0.12:
            score += 10
        elif roe > 0.05:
            score += 5
        elif roe < 0:
            score -= 15
    cr = m.get("Current Ratio")
    if cr is not None:
        if 1.5 <= cr <= 3.0:
            score += 10
        elif cr >= 1.0:
            score += 5
        else:
            score -= 10
    de = m.get("Debt-to-Equity")
    if de is not None:
        if de < 0.5:
            score += 10
        elif de < 1.0:
            score += 5
        elif de > 2.0:
            score -= 10
    nm = m.get("Net Margin")
    if nm is not None:
        if nm > 0.15:
            score += 10
        elif nm > 0.05:
            score += 5
        elif nm < 0:
            score -= 10
    ic = m.get("Interest Coverage")
    if ic is not None:
        if ic > 5:
            score += 5
        elif ic < 1.5:
            score -= 10
    return max(0.0, min(100.0, score))


def _score_valuation(m: Dict[str, float]) -> float:
    score = 50.0
    pr = m.get("Payout Ratio")
    if pr is not None:
        if 0.20 <= pr <= 0.60:
            score += 10
        elif pr > 1.0:
            score -= 10
    bvps = m.get("Book Value / Share")
    if bvps is not None and bvps > 0:
        score += 5
    return max(0.0, min(100.0, score))


def _score_growth(m: Dict[str, float]) -> float:
    score = 50.0
    rg = m.get("Revenue Growth")
    if rg is not None:
        if rg > 0.10:
            score += 15
        elif rg > 0.03:
            score += 10
        elif rg < -0.05:
            score -= 15
    eg = m.get("EPS Growth")
    if eg is not None:
        if eg > 0.10:
            score += 15
        elif eg > 0:
            score += 5
        elif eg < -0.10:
            score -= 15
    return max(0.0, min(100.0, score))


def _score_quality(m: Dict[str, float]) -> float:
    score = 50.0
    cfoni = m.get("CFO / Net Income")
    if cfoni is not None:
        if cfoni > 1.0:
            score += 15
        elif cfoni > 0.8:
            score += 5
        elif cfoni < 0.5:
            score -= 10
    fcf_m = m.get("FCF Margin")
    if fcf_m is not None:
        if fcf_m > 0.10:
            score += 10
        elif fcf_m > 0:
            score += 5
        else:
            score -= 10
    return max(0.0, min(100.0, score))


# ── Valuation model helpers ──────────────────────────────────────────

def _graham_number(eps: float, bvps: float, multiplier: float = 22.5) -> Dict[str, Any]:
    if eps <= 0 or bvps <= 0:
        return {"model": "graham", "intrinsic_value": None, "error": "EPS and BVPS must be positive.",
                "parameters": {"eps": eps, "bvps": bvps, "multiplier": multiplier}}
    iv = math.sqrt(multiplier * eps * bvps)
    return {"model": "graham", "intrinsic_value": round(iv, 2),
            "parameters": {"eps": eps, "bvps": bvps, "multiplier": multiplier},
            "assumptions": {"max_pe": 15, "max_pb": 1.5}}


def _dcf(fcf, g1, g2, dr, s1=5, s2=5, tg=0.025, shares=1.0):
    if dr <= tg:
        return {"model": "dcf", "intrinsic_value": None, "error": "Discount rate must exceed terminal growth."}
    projected = []
    cf = fcf
    for yr in range(1, s1 + 1):
        cf *= 1 + g1
        projected.append(cf / ((1 + dr) ** yr))
    for yr in range(s1 + 1, s1 + s2 + 1):
        cf *= 1 + g2
        projected.append(cf / ((1 + dr) ** yr))
    tv = cf * (1 + tg) / (dr - tg)
    pv_tv = tv / ((1 + dr) ** (s1 + s2))
    ev = sum(projected) + pv_tv
    ps = ev / shares if shares else 0
    return {"model": "dcf", "intrinsic_value": round(ps, 2), "enterprise_value": round(ev, 2),
            "pv_terminal": round(pv_tv, 2), "pv_fcfs": round(sum(projected), 2),
            "parameters": {"fcf": fcf, "growth_stage1": g1, "growth_stage2": g2, "discount_rate": dr,
                           "stage1_years": s1, "stage2_years": s2, "terminal_growth": tg, "shares_outstanding": shares},
            "assumptions": {"method": "Two-stage DCF with Gordon Growth terminal value"}}


def _ddm(div, gr, rr, hgy=5, hgr=None):
    if rr <= gr:
        return {"model": "ddm", "intrinsic_value": None, "error": "Required return must exceed stable growth."}
    if hgr is None:
        iv = div * (1 + gr) / (rr - gr)
        return {"model": "ddm", "intrinsic_value": round(iv, 2),
                "parameters": {"last_dividend": div, "growth_rate": gr, "required_return": rr},
                "assumptions": {"method": "Gordon Growth (single stage)"}}
    pv_div = 0.0
    d = div
    for yr in range(1, hgy + 1):
        d *= 1 + hgr
        pv_div += d / ((1 + rr) ** yr)
    tv = d * (1 + gr) / (rr - gr)
    pv_tv = tv / ((1 + rr) ** hgy)
    iv = pv_div + pv_tv
    return {"model": "ddm", "intrinsic_value": round(iv, 2), "pv_dividends": round(pv_div, 2),
            "pv_terminal": round(pv_tv, 2),
            "parameters": {"last_dividend": div, "growth_rate": gr, "required_return": rr,
                           "high_growth_years": hgy, "high_growth_rate": hgr},
            "assumptions": {"method": "Two-stage DDM"}}


def _comparable_multiples(mv, pm, mt="P/E", shares=1.0):
    implied = mv * pm
    ps = implied / shares if shares else 0
    return {"model": "multiples", "intrinsic_value": round(ps, 2), "implied_total": round(implied, 2),
            "parameters": {"metric_value": mv, "peer_multiple": pm, "multiple_type": mt, "shares_outstanding": shares},
            "assumptions": {"method": f"Comparable {mt} multiple"}}


def _save_valuation(stock_id: int, result: Dict[str, Any], user_id: int) -> None:
    now = int(time.time())
    exec_sql(
        """INSERT INTO valuation_models
           (stock_id, model_type, valuation_date, intrinsic_value,
            parameters, assumptions, created_by_user_id, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            stock_id, result["model"], date.today().isoformat(),
            result.get("intrinsic_value"),
            json.dumps(result.get("parameters", {})),
            json.dumps(result.get("assumptions", {})),
            user_id, now,
        ),
    )
