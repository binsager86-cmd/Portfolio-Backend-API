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

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, UploadFile, File
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
                summary_margin_of_safety REAL DEFAULT 15.0,
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
                risk_score REAL,
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
        # ── Cash flow staging tables ──────────────────────────────────
        f"""CREATE TABLE IF NOT EXISTS cashflow_extraction_runs (
                id {_PK},
                stock_id INTEGER NOT NULL,
                pdf_upload_id INTEGER,
                source_file TEXT,
                status TEXT NOT NULL DEFAULT 'extracted',
                periods_json TEXT,
                raw_model_response TEXT,
                reconciliation_summary TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (stock_id) REFERENCES analysis_stocks(id)
            )""",
        f"""CREATE TABLE IF NOT EXISTS cashflow_staged_rows (
                id {_PK},
                run_id INTEGER NOT NULL,
                row_order INTEGER NOT NULL,
                label_raw TEXT NOT NULL,
                normalized_code TEXT,
                section TEXT NOT NULL DEFAULT 'unknown',
                row_kind TEXT NOT NULL DEFAULT 'item',
                values_json TEXT NOT NULL DEFAULT '{{}}',
                is_total BOOLEAN DEFAULT 0,
                confidence REAL,
                is_accepted BOOLEAN DEFAULT 0,
                rejection_reason TEXT,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (run_id) REFERENCES cashflow_extraction_runs(id)
            )""",
        # ── Extraction jobs (async upload pipeline) ─────────────────────
        f"""CREATE TABLE IF NOT EXISTS extraction_jobs (
                id {_PK},
                stock_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                pdf_upload_id INTEGER,
                pdf_hash TEXT,
                source_file TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                stage TEXT NOT NULL DEFAULT 'uploading',
                pages_processed INTEGER DEFAULT 0,
                total_pages INTEGER DEFAULT 0,
                progress_percent REAL DEFAULT 0,
                model TEXT DEFAULT 'gemini-2.5-flash',
                error_message TEXT,
                result_payload TEXT,
                attempt_count INTEGER DEFAULT 1,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                started_at INTEGER,
                last_heartbeat_at INTEGER,
                completed_at INTEGER,
                FOREIGN KEY (stock_id) REFERENCES analysis_stocks(id)
            )""",
        "CREATE INDEX IF NOT EXISTS idx_extraction_jobs_stock ON extraction_jobs(stock_id)",
        "CREATE INDEX IF NOT EXISTS idx_extraction_jobs_hash ON extraction_jobs(stock_id, pdf_hash)",
        "CREATE INDEX IF NOT EXISTS idx_cf_runs_stock ON cashflow_extraction_runs(stock_id)",
        "CREATE INDEX IF NOT EXISTS idx_cf_rows_run ON cashflow_staged_rows(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_analysis_stocks_user ON analysis_stocks(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_analysis_stocks_symbol ON analysis_stocks(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_financial_statements_stock ON financial_statements(stock_id)",
        "CREATE INDEX IF NOT EXISTS idx_line_items_statement ON financial_line_items(statement_id)",
        "CREATE INDEX IF NOT EXISTS idx_stock_metrics_stock ON stock_metrics(stock_id)",
        "CREATE INDEX IF NOT EXISTS idx_valuation_models_stock ON valuation_models(stock_id)",
        "CREATE INDEX IF NOT EXISTS idx_stock_scores_stock ON stock_scores(stock_id)",
        "CREATE INDEX IF NOT EXISTS idx_pdf_uploads_stock ON pdf_uploads(stock_id)",
        f"""CREATE TABLE IF NOT EXISTS peer_companies (
                id {_PK},
                stock_id INTEGER NOT NULL,
                peer_symbol TEXT NOT NULL,
                peer_name TEXT NOT NULL,
                sector TEXT,
                pe REAL, pb REAL, ps REAL, pcf REAL, ev_ebitda REAL,
                eps REAL, price REAL,
                fetched_at INTEGER NOT NULL,
                UNIQUE(stock_id, peer_symbol),
                FOREIGN KEY (stock_id) REFERENCES analysis_stocks(id)
            )""",
        "CREATE INDEX IF NOT EXISTS idx_peer_companies_stock ON peer_companies(stock_id)",
    ]

    try:
        for ddl in _TABLES:
            try:
                exec_sql(ddl)
            except Exception as e:
                logger.warning("⚠️  DDL skipped: %s", e)
    except Exception as e:
        logger.warning("⚠️  Analysis schema creation skipped: %s", e)

    # ── Schema migrations (add columns to existing tables) ──
    _MIGRATIONS = [
        "ALTER TABLE stock_scores ADD COLUMN risk_score REAL",
        "ALTER TABLE analysis_stocks ADD COLUMN summary_margin_of_safety REAL DEFAULT 15.0",
    ]
    for mig in _MIGRATIONS:
        try:
            exec_sql(mig)
        except Exception:
            pass  # column already exists

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


# ── Extraction job helpers ───────────────────────────────────────────

# Stale-job threshold: jobs without heartbeat for this long are marked failed.
STALE_JOB_THRESHOLD_SECONDS = 15 * 60  # 15 minutes


def _log_job(job_id: int, stock_id: int, event: str, **extra: Any) -> None:
    """Emit a structured extraction-job log line.

    Every message includes job_id and stock_id; callers add stage, duration_ms,
    filename, hash, error, etc. via **extra.
    """
    parts = [f"extraction job_id={job_id} stock_id={stock_id} event={event}"]
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    logger.info(" ".join(parts))


def _update_job(job_id: int, **fields: Any) -> None:
    """Update extraction_jobs fields atomically."""
    fields["updated_at"] = int(time.time())
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [job_id]
    exec_sql(f"UPDATE extraction_jobs SET {sets} WHERE id = ?", tuple(vals))


def recover_stale_jobs() -> int:
    """Mark stale running/queued jobs as failed.

    Called at app startup and can be called periodically.
    Returns the number of jobs recovered.
    """
    now = int(time.time())
    cutoff = now - STALE_JOB_THRESHOLD_SECONDS
    stale = query_all(
        """SELECT id, stock_id, status, last_heartbeat_at, updated_at FROM extraction_jobs
           WHERE status IN ('queued', 'running')
           AND COALESCE(last_heartbeat_at, updated_at, created_at) < ?""",
        (cutoff,),
    )
    count = 0
    for row in (stale or []):
        r = dict(row) if not isinstance(row, dict) else row
        job_id = r["id"]
        logger.warning(
            "Recovering stale extraction job %d (status=%s, last_heartbeat=%s)",
            job_id, r["status"], r.get("last_heartbeat_at"),
        )
        _log_job(job_id, r.get("stock_id", 0), "stale_recovery",
                 prev_status=r["status"],
                 last_heartbeat=r.get("last_heartbeat_at"))
        _update_job(
            job_id,
            status="failed",
            error_message="Job timed out — server restart or stale heartbeat.",
            completed_at=now,
        )
        count += 1
    if count:
        logger.info("Recovered %d stale extraction job(s).", count)
    return count


HEARTBEAT_INTERVAL_SECONDS = 30  # Periodic heartbeat interval


def _start_heartbeat_thread(job_id: int, interval: int = HEARTBEAT_INTERVAL_SECONDS):
    """Spawn a daemon thread that updates last_heartbeat_at every *interval* seconds.

    Returns (thread, stop_event).  Call ``stop_event.set()`` to terminate.
    """
    import threading

    stop_event = threading.Event()

    def _heartbeat_loop():
        while not stop_event.wait(timeout=interval):
            try:
                _update_job(job_id, last_heartbeat_at=int(time.time()))
            except Exception:
                pass  # best-effort; don't crash the heartbeat thread

    t = threading.Thread(target=_heartbeat_loop, daemon=True, name=f"hb-job-{job_id}")
    t.start()
    return t, stop_event


def _run_extraction_job_sync(
    job_id: int,
    stock_id: int,
    user_id: int,
    pdf_bytes: bytes,
    filename: str,
    model: str,
    force: bool,
    api_key: str,
    existing_codes: List[Dict[str, str]],
) -> None:
    """Background extraction worker — runs the full AI pipeline for a job.

    This is a synchronous wrapper executed via FastAPI BackgroundTasks.
    The actual AI call (extract_financials) is async, so we run it in
    an event loop.  A periodic heartbeat thread keeps *last_heartbeat_at*
    fresh so stale-job recovery doesn't kill long-running jobs.
    """
    import asyncio
    from app.services.extraction_service import extract_financials

    start_ts = time.time()
    now = int(start_ts)
    _update_job(job_id, status="running", stage="extracting", started_at=now, last_heartbeat_at=now)

    _log_job(job_id, stock_id, "started", filename=filename, model=model)

    # ── Start periodic heartbeat thread ──────────────────────────────
    hb_thread, hb_stop = _start_heartbeat_thread(job_id)
    loop: Optional[asyncio.AbstractEventLoop] = None

    try:
        # Run the async extraction in a new event loop (BackgroundTasks are sync threads)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(extract_financials(
            pdf_bytes=pdf_bytes,
            stock_id=stock_id,
            api_key=api_key,
            filename=filename,
            model_name=model,
            use_cache=not force,
            existing_codes=existing_codes if existing_codes else None,
        ))

        extract_ms = int((time.time() - start_ts) * 1000)
        _log_job(job_id, stock_id, "extraction_complete", filename=filename,
                 duration_ms=extract_ms, pages=result.pages_processed)

        # Heartbeat after extraction
        _update_job(
            job_id,
            stage="saving",
            total_pages=result.pages_processed,
            pages_processed=result.pages_processed,
            progress_percent=80,
            last_heartbeat_at=int(time.time()),
        )

        # ── Persist (must succeed for status=done) ───────────────────
        stock_row = query_one(
            "SELECT currency FROM analysis_stocks WHERE id = ?", (stock_id,),
        )
        stock_currency = stock_row["currency"] if stock_row else "USD"

        created_statements, total_items = _persist_extraction_result(
            stock_id=stock_id,
            result=result,
            stock_currency=stock_currency,
            source_file=filename,
            extracted_by="gemini-ai-pipeline",
            notes_template=f"AI-extracted (pipeline, retries={result.retry_count})",
        )

        # Save PDF (non-fatal)
        try:
            _save_pdf_file(stock_id, user_id, pdf_bytes, filename)
        except Exception as e:
            logger.warning("PDF file save failed (non-fatal): %s", e)

        # Build audit summary
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

        total_ms = int((time.time() - start_ts) * 1000)
        result_payload = {
            "message": (
                f"Successfully extracted {len(created_statements)} statements "
                f"with {total_items} line items."
            ),
            "statements": created_statements,
            "source_file": filename,
            "pages_processed": result.pages_processed,
            "model": result.model_used,
            "confidence": result.confidence,
            "cached": result.cached,
            "audit": audit_summary,
            "duration_ms": total_ms,
        }

        # Attach raw AI text for debugging when 0 statements extracted
        if not created_statements and getattr(result, "raw_ai_text", ""):
            result_payload["raw_ai_response_preview"] = result.raw_ai_text[:3000]

        # ── Finalize atomically: status=done only after all persistence ──
        _update_job(
            job_id,
            status="done",
            stage="done",
            progress_percent=100,
            pages_processed=result.pages_processed,
            total_pages=result.pages_processed,
            result_payload=json.dumps(result_payload, default=str),
            last_heartbeat_at=int(time.time()),
            completed_at=int(time.time()),
        )

        _log_job(job_id, stock_id, "done", filename=filename,
                 duration_ms=total_ms, statements=len(created_statements),
                 items=total_items)

    except Exception as exc:
        err_ms = int((time.time() - start_ts) * 1000)
        _log_job(job_id, stock_id, "failed", filename=filename,
                 duration_ms=err_ms, error=str(exc)[:200])
        _update_job(
            job_id,
            status="failed",
            error_message=str(exc)[:2000],
            last_heartbeat_at=int(time.time()),
            completed_at=int(time.time()),
        )
    finally:
        # ── Cleanup: stop heartbeat, close event loop ────────────────
        hb_stop.set()
        hb_thread.join(timeout=5)
        if loop is not None:
            try:
                loop.close()
            except Exception:
                pass


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
    summary_margin_of_safety: Optional[float] = None


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
    statement_type: Optional[str] = Field(None, description="Optional: only delete this statement type (income, balance, cashflow, equity)")

class LineItemUpdate(BaseModel):
    amount: float


class ReorderItem(BaseModel):
    id: int
    order_index: int


class ReorderItemsRequest(BaseModel):
    items: List[ReorderItem]


class CreateLineItemRequest(BaseModel):
    statement_id: int
    line_item_code: str
    line_item_name: str
    amount: float = 0.0
    order_index: Optional[int] = None


class MergeLineItemsRequest(BaseModel):
    """Merge two line items into one.

    keep_code: the line_item_code to keep (target)
    remove_code: the line_item_code to merge into keep_code and delete
    """
    keep_code: str
    remove_code: str


class MetricsCalculateRequest(BaseModel):
    period_end_date: str
    fiscal_year: int
    fiscal_quarter: Optional[int] = None


class GrahamRequest(BaseModel):
    eps: float
    growth_rate: float = 0.0
    corporate_yield: float = 4.4
    margin_of_safety: float = 25.0
    current_price: Optional[float] = None


class DCFRequest(BaseModel):
    fcf: float
    growth_rate_stage1: float
    growth_rate_stage2: float
    discount_rate: float
    stage1_years: int = 5
    stage2_years: int = 5
    terminal_growth: float = 0.025
    shares_outstanding: float = 1.0
    cash: float = 0.0
    debt: float = 0.0
    # Optional WACC components (for display in result card)
    wacc_used: bool = False
    wacc_risk_free_rate: Optional[float] = None
    wacc_beta: Optional[float] = None
    wacc_equity_risk_premium: Optional[float] = None
    wacc_cost_of_equity: Optional[float] = None
    wacc_cost_of_debt: Optional[float] = None
    wacc_tax_rate: Optional[float] = None
    wacc_weight_equity: Optional[float] = None
    wacc_weight_debt: Optional[float] = None


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
    if body.statement_type:
        rows = query_all(
            f"SELECT id FROM financial_statements WHERE stock_id = ? AND statement_type = ? AND period_end_date IN ({placeholders})",
            [stock_id, body.statement_type] + body.periods,
        )
    else:
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


@router.delete("/stocks/{stock_id}/statements")
async def delete_all_statements(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Hard-delete ALL financial statements and their line items for a stock."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    rows = query_all(
        "SELECT id FROM financial_statements WHERE stock_id = ?",
        (stock_id,),
    )
    if not rows:
        raise NotFoundError("No statements found for this stock.")

    stmt_ids = [r["id"] if isinstance(r, dict) else r[0] for r in rows]
    id_placeholders = ",".join("?" for _ in stmt_ids)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM financial_line_items WHERE statement_id IN ({id_placeholders})", stmt_ids)
        cur.execute(f"DELETE FROM financial_statements WHERE id IN ({id_placeholders})", stmt_ids)
        conn.commit()

    return {
        "status": "ok",
        "data": {"message": f"All {len(stmt_ids)} statement(s) deleted.", "deleted_count": len(stmt_ids)},
    }


# ── Helper: convert camelCase SA keys to readable display names ──────

_SA_METADATA_KEYS = {"datekey", "fiscalYear", "fiscalQuarter"}

# SA keys that represent ratios/growth metrics — exclude from statement line items
_SA_RATIO_KEYS = {
    "revenueGrowth", "netIncomeGrowth", "epsGrowth", "fcfGrowth",
    "dividendGrowth", "grossMargin", "operatingMargin", "profitMargin",
    "fcfMargin", "ebitdaMargin", "ebitMargin", "effectiveTaxRate",
    "sharesYoY", "ocfGrowth", "netCashGrowth", "totalcashGrowth",
}


def _camel_to_display(key: str) -> str:
    """Convert camelCase key to 'Title Case Display Name'.

    e.g. 'stockBasedCompensation' -> 'Stock Based Compensation'
         'netIncomeCF' -> 'Net Income CF'
         'totalOpex' -> 'Total Opex'
    """
    import re as _re
    # Insert space before uppercase letters that follow lowercase letters or
    # before a run of uppercase letters followed by a lowercase letter
    spaced = _re.sub(r'([a-z])([A-Z])', r'\1 \2', key)
    spaced = _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    return spaced[:1].upper() + spaced[1:]


# ── Fetch statements from macrotrends.net (US stocks, 10+ years) ─────

_MT_FIELD_MAP_INCOME = {
    "Revenue": ("revenue", "Revenue"),
    "Cost Of Goods Sold": ("cost_of_revenue", "Cost of Revenue"),
    "Gross Profit": ("gross_profit", "Gross Profit"),
    "Research And Development Expenses": ("research_development", "Research & Development"),
    "SG&A Expenses": ("sga", "SG&A"),
    "Other Operating Income Or Expenses": ("other_operating_expenses", "Other Operating Expenses"),
    "Operating Expenses": ("total_operating_expenses", "Total Operating Expenses"),
    "Operating Income": ("operating_income", "Operating Income"),
    "Total Non-Operating Income/Expense": ("non_operating_income", "Non-Operating Income"),
    "Pre-Tax Income": ("income_before_tax", "Pretax Income"),
    "Income Taxes": ("income_tax", "Income Tax"),
    "Income After Taxes": ("income_after_tax", "Income After Taxes"),
    "Other Income": ("other_income", "Other Income"),
    "Income From Continuous Operations": ("earnings_continuing_ops", "Earnings from Continuing Ops"),
    "Income From Discontinued Operations": ("earnings_discontinued_ops", "Earnings from Discontinued Ops"),
    "Net Income": ("net_income", "Net Income"),
    "EBITDA": ("ebitda", "EBITDA"),
    "EBIT": ("ebit", "EBIT"),
    "Basic Shares Outstanding": ("shares_outstanding", "Shares Outstanding"),
    "Shares Outstanding": ("shares_diluted", "Shares Diluted"),
    "Basic EPS": ("eps_basic", "EPS (Basic)"),
    "EPS - Earnings Per Share": ("eps_diluted", "EPS (Diluted)"),
}

_MT_FIELD_MAP_BALANCE = {
    "Cash On Hand": ("cash", "Cash & Equivalents"),
    "Receivables": ("accounts_receivable", "Accounts Receivable"),
    "Inventory": ("inventory", "Inventory"),
    "Pre-Paid Expenses": ("prepaid_expenses", "Prepaid Expenses"),
    "Other Current Assets": ("other_current_assets", "Other Current Assets"),
    "Total Current Assets": ("total_current_assets", "Total Current Assets"),
    "Property, Plant, And Equipment": ("ppe_net", "Property, Plant & Equipment"),
    "Long-Term Investments": ("long_term_investments", "Long-Term Investments"),
    "Goodwill And Intangible Assets": ("goodwill", "Goodwill & Intangible Assets"),
    "Other Long-Term Assets": ("other_long_term_assets", "Other Long-Term Assets"),
    "Total Long-Term Assets": ("total_long_term_assets", "Total Long-Term Assets"),
    "Total Assets": ("total_assets", "Total Assets"),
    "Total Current Liabilities": ("total_current_liabilities", "Total Current Liabilities"),
    "Long Term Debt": ("long_term_debt", "Long-Term Debt"),
    "Other Non-Current Liabilities": ("other_non_current_liabilities", "Other Non-Current Liabilities"),
    "Total Long-Term Liabilities": ("total_long_term_liabilities", "Total Long-Term Liabilities"),
    "Total Liabilities": ("total_liabilities", "Total Liabilities"),
    "Common Stock Net": ("share_capital", "Share Capital"),
    "Retained Earnings (Accumulated Deficit)": ("retained_earnings", "Retained Earnings"),
    "Comprehensive Income": ("comprehensive_income", "Comprehensive Income"),
    "Other Share Holders Equity": ("other_equity", "Other Shareholders' Equity"),
    "Share Holder Equity": ("total_equity", "Total Equity"),
    "Total Liabilities And Share Holders Equity": ("total_liabilities_equity", "Total Liabilities & Equity"),
}

_MT_FIELD_MAP_CASHFLOW = {
    "Net Income/Loss": ("net_income_cf", "Net Income"),
    "Total Depreciation And Amortization - Cash Flow": ("depreciation_cf", "Depreciation & Amortization"),
    "Other Non-Cash Items": ("other_non_cash", "Other Non-Cash Items"),
    "Total Non-Cash Items": ("total_non_cash", "Total Non-Cash Items"),
    "Change In Accounts Receivable": ("change_receivables", "Change in Receivables"),
    "Change In Inventories": ("change_inventories", "Change in Inventories"),
    "Change In Accounts Payable": ("change_payables", "Change in Payables"),
    "Change In Assets/Liabilities": ("change_assets_liabilities", "Change in Assets/Liabilities"),
    "Total Change In Assets/Liabilities": ("total_change_assets_liabilities", "Total Change in Assets/Liabilities"),
    "Cash Flow From Operating Activities": ("cash_from_operations", "Cash from Operations"),
    "Net Change In Property, Plant, And Equipment": ("capital_expenditures", "Capital Expenditures"),
    "Net Change In Intangible Assets": ("change_intangibles", "Change in Intangibles"),
    "Net Acquisitions/Divestitures": ("acquisitions", "Acquisitions/Divestitures"),
    "Net Change In Short-term Investments": ("change_short_term_investments", "Change in Short-Term Investments"),
    "Net Change In Long-Term Investments": ("change_long_term_investments", "Change in Long-Term Investments"),
    "Net Change In Investments - Total": ("change_investments_total", "Change in Investments Total"),
    "Investing Activities - Other": ("investing_other", "Investing Other"),
    "Cash Flow From Investing Activities": ("cash_from_investing", "Cash from Investing"),
    "Net Long-Term Debt": ("net_debt_issued", "Net Long-Term Debt"),
    "Net Current Debt": ("net_current_debt", "Net Current Debt"),
    "Debt Issuance/Retirement Net - Total": ("net_debt_total", "Net Debt Total"),
    "Net Common Equity Issued/Repurchased": ("shares_repurchased", "Shares Issued/Repurchased"),
    "Net Total Equity Issued/Repurchased": ("net_equity_total", "Net Equity Total"),
    "Total Common And Preferred Stock Dividends Paid": ("dividends_paid", "Dividends Paid"),
    "Financial Activities - Other": ("financing_other", "Financing Other"),
    "Cash Flow From Financial Activities": ("cash_from_financing", "Cash from Financing"),
    "Net Cash Flow": ("net_change_in_cash", "Net Change in Cash"),
    "Free Cash Flow": ("free_cash_flow", "Free Cash Flow"),
}

_MT_STMT_MAP = {
    "income":   ("https://www.macrotrends.net/stocks/charts/{sym}/{slug}/income-statement", _MT_FIELD_MAP_INCOME),
    "balance":  ("https://www.macrotrends.net/stocks/charts/{sym}/{slug}/balance-sheet", _MT_FIELD_MAP_BALANCE),
    "cashflow": ("https://www.macrotrends.net/stocks/charts/{sym}/{slug}/cash-flow-statement", _MT_FIELD_MAP_CASHFLOW),
}

# Maximum years to fetch from macrotrends (user requested 10)
_MT_MAX_YEARS = 10


def _mt_resolve_slug(symbol: str, client: "httpx.Client") -> Optional[str]:
    """Get the macrotrends company slug by following the redirect.

    macrotrends.net/stocks/charts/AAPL/ → 301 → .../AAPL/apple/
    """
    url = f"https://www.macrotrends.net/stocks/charts/{symbol.upper()}/"
    try:
        # HEAD first (lighter)
        resp = client.head(url, follow_redirects=False)
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("location", "")
            parts = loc.rstrip("/").split("/")
            if len(parts) >= 2:
                return parts[-1]

        # Fallback: GET without following redirects
        resp = client.get(url, follow_redirects=False)
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("location", "")
            parts = loc.rstrip("/").split("/")
            if len(parts) >= 2:
                return parts[-1]

        # Last resort: follow redirect and parse final URL
        if resp.status_code == 200:
            final = str(resp.url).rstrip("/").split("/")
            if len(final) >= 2:
                return final[-1]
    except Exception as exc:
        logger.warning("macrotrends slug resolution failed for %s: %s", symbol, exc)
    return None


def _mt_parse_financial_data(html: str) -> Optional[list]:
    """Extract originalData from macrotrends.net page.

    Returns list of dicts like:
    [{"field_name": "<a href='...'>Revenue</a>", "2024-09-30": "391035.00", ...}, ...]
    """
    import re as _re
    import json as _json

    m = _re.search(r'var\s+originalData\s*=\s*(\[.*?\]);\s*$', html, _re.MULTILINE | _re.DOTALL)
    if not m:
        m = _re.search(r'var\s+originalData\s*=\s*(\[.*?\]);', html, _re.DOTALL)
    if not m:
        return None
    try:
        return _json.loads(m.group(1))
    except (ValueError, _json.JSONDecodeError) as exc:
        logger.warning("macrotrends JSON parse error: %s", exc)
        return None


def _fetch_us_statements(stock_id: int, symbol: str, user_id: int) -> dict:
    """Fetch financial statements from stockanalysis.com for a US stock.

    Uses the same SvelteKit page format as Kuwait stocks, but with
    /stocks/{sym}/ URLs instead of /quote/kwse/{sym}/.

    Returns {"status": "ok", "data": {...}} or raises HTTPException.
    """
    import httpx as _httpx
    import re as _re

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    now = int(time.time())
    saved_summary = []
    base = symbol.upper()

    # US stocks use /stocks/{sym}/ on stockanalysis.com
    us_stmt_map = {
        "income":   (f"https://stockanalysis.com/stocks/{base.lower()}/financials/", _SA_FIELD_MAP_INCOME),
        "balance":  (f"https://stockanalysis.com/stocks/{base.lower()}/financials/balance-sheet/", _SA_FIELD_MAP_BALANCE),
        "cashflow": (f"https://stockanalysis.com/stocks/{base.lower()}/financials/cash-flow-statement/", _SA_FIELD_MAP_CASHFLOW),
    }

    for stmt_type, (url, field_map) in us_stmt_map.items():
        try:
            resp = _httpx.get(url, headers=headers, timeout=20, follow_redirects=True)
            if resp.status_code != 200:
                logger.warning("stockanalysis.com %s returned %s for US/%s", stmt_type, resp.status_code, base)
                continue
        except Exception as exc:
            logger.warning("stockanalysis.com fetch error for US/%s/%s: %s", base, stmt_type, exc)
            continue

        data = _sa_parse_financial_data(resp.text)
        if not data:
            logger.warning("stockanalysis.com: no financialData for US/%s/%s", base, stmt_type)
            continue

        datekeys = data.get("datekey", [])
        fiscal_years = data.get("fiscalYear", [])

        # Process each period (skip TTM), limit to 10 years
        periods_saved = 0
        for idx, dk in enumerate(datekeys):
            if dk == "TTM":
                continue
            if periods_saved >= _MT_MAX_YEARS:
                break

            period_end_date = dk  # e.g. "2024-12-31"
            try:
                fy = int(fiscal_years[idx])
            except (IndexError, ValueError):
                continue

            # Collect line items for this period
            line_items = []
            seen_codes = set()
            order = 1
            for sa_key, (canonical_code, display_name) in field_map.items():
                if sa_key not in data:
                    continue
                if canonical_code in seen_codes:
                    continue
                vals = data[sa_key]
                if idx >= len(vals) or vals[idx] is None:
                    continue
                seen_codes.add(canonical_code)
                line_items.append({
                    "code": canonical_code,
                    "name": display_name,
                    "amount": vals[idx],
                    "currency": "USD",
                    "order": order,
                    "is_total": canonical_code in (
                        "total_assets", "total_equity", "total_liabilities",
                        "total_current_assets", "total_current_liabilities",
                        "total_liabilities_equity", "total_operating_expenses",
                        "total_common_equity",
                    ),
                })
                order += 1

            # ── Include ALL remaining fields not in field_map ────────
            mapped_sa_keys = set(field_map.keys())
            for extra_key, vals in data.items():
                if extra_key in _SA_METADATA_KEYS:
                    continue
                if extra_key in _SA_RATIO_KEYS:
                    continue
                if extra_key in mapped_sa_keys:
                    continue
                if not isinstance(vals, list):
                    continue
                if extra_key in seen_codes:
                    continue
                if idx >= len(vals) or vals[idx] is None:
                    continue
                seen_codes.add(extra_key)
                line_items.append({
                    "code": extra_key,
                    "name": _camel_to_display(extra_key),
                    "amount": vals[idx],
                    "currency": "USD",
                    "order": order,
                    "is_total": False,
                })
                order += 1

            if not line_items:
                continue

            # Upsert: check existing statement
            existing = query_one(
                """SELECT id FROM financial_statements
                   WHERE stock_id = ? AND statement_type = ? AND period_end_date = ?""",
                (stock_id, stmt_type, period_end_date),
            )

            source_label = f"stockanalysis.com/stocks/{base.lower()}"

            with get_connection() as conn:
                cur = conn.cursor()
                if existing:
                    stmt_id = existing["id"] if isinstance(existing, dict) else existing[0]
                    cur.execute(
                        """UPDATE financial_statements
                           SET fiscal_year=?, extracted_by=?, source_file=?,
                               confidence_score=?, notes=?, created_at=?
                           WHERE id=?""",
                        (fy, "stockanalysis.com", source_label,
                         1.0, "Fetched from stockanalysis.com", now, stmt_id),
                    )
                    cur.execute("DELETE FROM financial_line_items WHERE statement_id = ?", (stmt_id,))
                else:
                    cur.execute(
                        """INSERT INTO financial_statements
                           (stock_id, statement_type, fiscal_year, fiscal_quarter,
                            period_end_date, source_file, extracted_by,
                            confidence_score, notes, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (stock_id, stmt_type, fy, None, period_end_date,
                         source_label, "stockanalysis.com",
                         1.0, "Fetched from stockanalysis.com", now),
                    )
                    stmt_id = cur.lastrowid

                for item in line_items:
                    cur.execute(
                        """INSERT INTO financial_line_items
                           (statement_id, line_item_code, line_item_name,
                            amount, currency, order_index, is_total)
                           VALUES (?,?,?,?,?,?,?)""",
                        (stmt_id, item["code"], item["name"],
                         item["amount"], item["currency"],
                         item["order"], item["is_total"]),
                    )
                conn.commit()
            periods_saved += 1

        if periods_saved > 0:
            saved_summary.append({
                "statement_type": stmt_type,
                "periods_saved": periods_saved,
            })

    # ── Supplement with macrotrends.net for older years (up to 10 total) ──
    # Also merges missing line items into existing sparse periods from SA.
    try:
        # Build lookup of existing periods → (statement_id, set_of_codes)
        period_info_by_type: Dict[str, Dict[str, tuple]] = {}  # stmt_type → {period → (stmt_id, {codes})}
        for s in saved_summary:
            stmts = query_df(
                "SELECT id, period_end_date FROM financial_statements WHERE stock_id = ? AND statement_type = ?",
                (stock_id, s["statement_type"]),
            )
            info: Dict[str, tuple] = {}
            if not stmts.empty:
                for _, row in stmts.iterrows():
                    sid = row["id"]
                    ped = row["period_end_date"]
                    codes_df = query_df(
                        "SELECT line_item_code FROM financial_line_items WHERE statement_id = ?",
                        (sid,),
                    )
                    existing_codes = set(codes_df["line_item_code"].tolist()) if not codes_df.empty else set()
                    info[ped] = (sid, existing_codes)
            period_info_by_type[s["statement_type"]] = info

        slug = _mt_resolve_slug(base, _httpx.Client(headers=headers, timeout=20, follow_redirects=True))
        if slug:
            mt_saved = 0
            for stmt_type, (url_tpl, field_map) in _MT_STMT_MAP.items():
                period_info = period_info_by_type.get(stmt_type, {})
                existing_years = {p[:4] for p in period_info.keys()}
                total_years = len(existing_years)

                url = url_tpl.format(sym=base, slug=slug)
                try:
                    resp = _httpx.get(url, headers=headers, timeout=20, follow_redirects=True)
                    if resp.status_code != 200:
                        continue
                except Exception:
                    continue

                rows_data = _mt_parse_financial_data(resp.text)
                if not rows_data:
                    continue

                # Extract date columns from first row
                date_cols = [k for k in rows_data[0].keys() if k not in ("field_name", "popup_icon")]
                date_cols.sort(reverse=True)

                import re as _re

                for date_col in date_cols:
                    year_str = date_col[:4]

                    # For NEW years: check total limit
                    is_new_year = year_str not in existing_years
                    if is_new_year and total_years >= _MT_MAX_YEARS:
                        break

                    # Find matching existing period for this year
                    existing_stmt_id = None
                    existing_codes: set = set()
                    for ep, (sid, codes) in period_info.items():
                        if ep == date_col or ep.startswith(year_str):
                            existing_stmt_id = sid
                            existing_codes = codes
                            break

                    try:
                        fy = int(year_str)
                    except ValueError:
                        continue

                    line_items = []
                    order = 1
                    for row in rows_data:
                        raw_name = _re.sub(r'<[^>]+>', '', row.get("field_name", "")).strip()
                        val_str = row.get(date_col, "")
                        if not val_str or val_str.strip() == "":
                            continue
                        try:
                            amount = float(val_str.replace(",", ""))
                        except (ValueError, TypeError):
                            continue

                        if raw_name in field_map:
                            code, display = field_map[raw_name]
                        else:
                            code = raw_name.lower().replace(" ", "_").replace(",", "").replace("-", "_")
                            display = raw_name

                        # Skip codes already present in this period (from SA)
                        if code in existing_codes:
                            continue

                        # Per-share items are NOT in millions on macrotrends
                        _MT_NO_SCALE = {
                            "eps_basic", "eps_diluted",
                            "dividends_per_share", "book_value_per_share",
                        }
                        if code in _MT_NO_SCALE:
                            scaled = amount
                        else:
                            scaled = amount * 1_000_000  # macrotrends values are in millions
                        line_items.append({
                            "code": code, "name": display,
                            "amount": scaled,
                            "currency": "USD", "order": order, "is_total": False,
                        })
                        order += 1

                    if not line_items:
                        continue

                    source_label = f"macrotrends.net/{base}/{slug}"
                    with get_connection() as conn:
                        cur = conn.cursor()
                        if existing_stmt_id:
                            # Merge: add missing line items to existing statement
                            stmt_id = existing_stmt_id
                            # Get max existing order_index
                            max_ord = query_one(
                                "SELECT MAX(order_index) as mx FROM financial_line_items WHERE statement_id = ?",
                                (stmt_id,),
                            )
                            start_order = ((max_ord["mx"] if isinstance(max_ord, dict) else max_ord[0]) or 0) + 1
                        else:
                            # Create new statement
                            cur.execute(
                                """INSERT INTO financial_statements
                                   (stock_id, statement_type, fiscal_year, fiscal_quarter,
                                    period_end_date, source_file, extracted_by,
                                    confidence_score, notes, created_at)
                                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                                (stock_id, stmt_type, fy, None, date_col,
                                 source_label, "macrotrends.net",
                                 0.9, "Fetched from macrotrends.net (older years)", now),
                            )
                            stmt_id = cur.lastrowid
                            start_order = 1

                        for i, item in enumerate(line_items):
                            cur.execute(
                                """INSERT INTO financial_line_items
                                   (statement_id, line_item_code, line_item_name,
                                    amount, currency, order_index, is_total)
                                   VALUES (?,?,?,?,?,?,?)""",
                                (stmt_id, item["code"], item["name"],
                                 item["amount"], item["currency"],
                                 start_order + i, item["is_total"]),
                            )
                        conn.commit()

                    if is_new_year:
                        existing_years.add(year_str)
                        total_years += 1
                    mt_saved += 1

                # Update saved_summary
                if mt_saved > 0:
                    existing_entry = next((s for s in saved_summary if s["statement_type"] == stmt_type), None)
                    if existing_entry:
                        existing_entry["periods_saved"] += mt_saved
                    else:
                        saved_summary.append({"statement_type": stmt_type, "periods_saved": mt_saved})
    except Exception as exc:
        logger.warning("macrotrends supplement failed for %s: %s", base, exc)

    if not saved_summary:
        raise HTTPException(
            status_code=404,
            detail=f"No financial data found for {base}.",
        )

    total_periods = sum(s["periods_saved"] for s in saved_summary)
    return {
        "status": "ok",
        "data": {
            "message": f"Fetched {total_periods} period(s) across {len(saved_summary)} statement type(s).",
            "summary": saved_summary,
            "source": f"stockanalysis.com + macrotrends.net",
        },
    }


# ── Fetch statements from stockanalysis.com ──────────────────────────

# Field mapping: stockanalysis.com JS key -> (canonical_code, display_name)
_SA_FIELD_MAP_INCOME = {
    # Revenue
    "revenue": ("revenue", "Revenue"),
    "revenueRE": ("revenue", "Revenue"),
    "rentalRevenue": ("rental_revenue", "Rental Revenue"),
    # Cost of revenue
    "costrev": ("cost_of_revenue", "Cost of Revenue"),
    "cor": ("cost_of_revenue", "Cost of Revenue"),
    # Gross profit
    "grossProfit": ("gross_profit", "Gross Profit"),
    # SG&A / R&D
    "sgna": ("sga", "SG&A"),
    "sgnaRE": ("sga", "SG&A"),
    "rnd": ("research_development", "Research & Development"),
    "propertyExpenses": ("property_expenses", "Property Expenses"),
    # Operating expenses
    "otherOpex": ("other_operating_expenses", "Other Operating Expenses"),
    "otherOperatingExpensesRE": ("other_operating_expenses", "Other Operating Expenses"),
    "totalOpex": ("total_operating_expenses", "Total Operating Expenses"),
    "totalOperatingExpensesRE": ("total_operating_expenses", "Total Operating Expenses"),
    "totalOperatingExpenses": ("total_operating_expenses", "Total Operating Expenses"),
    # Operating income
    "opinc": ("operating_income", "Operating Income"),
    "operatingIncomeRE": ("operating_income", "Operating Income"),
    "operatingIncome": ("operating_income", "Operating Income"),
    # Non-operating income
    "totalNonOperatingIncome": ("non_operating_income", "Non-Operating Income"),
    "intexp": ("interest_expense", "Interest Expense"),
    # Pretax / tax
    "pretax": ("income_before_tax", "Pretax Income"),
    "taxexp": ("income_tax", "Income Tax"),
    "income_statement_provision_for_income_taxes": ("income_tax", "Income Tax"),
    # Net income
    "netinc": ("net_income", "Net Income"),
    "netIncome": ("net_income", "Net Income"),
    "netinccmn": ("net_income_common", "Net Income to Common"),
    # EPS
    "epsBasic": ("eps_basic", "EPS (Basic)"),
    "epsdil": ("eps_diluted", "EPS (Diluted)"),
    "epsDiluted": ("eps_diluted", "EPS (Diluted)"),
    # EBITDA / EBIT
    "ebitda": ("ebitda", "EBITDA"),
    "ebit": ("ebit", "EBIT"),
    # Shares
    "sharesBasic": ("shares_outstanding", "Shares Outstanding"),
    "sharesDiluted": ("shares_diluted", "Shares Diluted"),
    # Per-share
    "dps": ("dividends_per_share", "Dividends Per Share"),
    # Other
    "depAmorEbitda": ("depreciation_amortization", "Depreciation & Amortization"),
    "minorityInterest": ("minority_interest", "Minority Interest"),
    "earningContinuing": ("earnings_continuing_ops", "Earnings from Continuing Ops"),
    # Derived amounts (not ratios)
    "fcf": ("free_cash_flow", "Free Cash Flow"),
    "fcfps": ("fcf_per_share", "FCF Per Share"),
}

_SA_FIELD_MAP_BALANCE = {
    # Cash & equivalents
    "cashneq": ("cash", "Cash & Equivalents"),
    "totalcash": ("cash", "Cash & Equivalents"),
    # Investments
    "shortTermInvestments": ("short_term_investments", "Short-Term Investments"),
    "investmentsc": ("short_term_investments", "Short-Term Investments"),
    "longTermInvestmentsRE": ("long_term_investments", "Long-Term Investments"),
    "tradingAssetSecurities": ("trading_securities", "Trading Securities"),
    # Current assets
    "accountsReceivable": ("accounts_receivable", "Accounts Receivable"),
    "balance_sheet_accounts_receivable": ("accounts_receivable", "Accounts Receivable"),
    "inventory": ("inventory", "Inventory"),
    "assetsc": ("total_current_assets", "Total Current Assets"),
    "totalCurrentAssets": ("total_current_assets", "Total Current Assets"),
    "balance_sheet_other_current_assets": ("other_current_assets", "Other Current Assets"),
    # Long-term assets
    "netPPE": ("ppe_net", "Property, Plant & Equipment"),
    "balance_sheet_net_property_plant_and_equipment": ("ppe_net", "Property, Plant & Equipment"),
    "goodwill": ("goodwill", "Goodwill"),
    "balance_sheet_goodwill": ("goodwill", "Goodwill"),
    "intangibles": ("intangible_assets", "Intangible Assets"),
    "otherIntangibles": ("other_intangibles", "Other Intangibles"),
    "balance_sheet_other_long_term_assets": ("other_long_term_assets", "Other Long-Term Assets"),
    "balance_sheet_long_term_investments": ("long_term_investments", "Long-Term Investments"),
    # Total assets
    "assets": ("total_assets", "Total Assets"),
    # Current liabilities
    "accountsPayable": ("accounts_payable", "Accounts Payable"),
    "balance_sheet_accounts_payable": ("accounts_payable", "Accounts Payable"),
    "shortTermDebt": ("short_term_debt", "Short-Term Debt"),
    "balance_sheet_current_portion_of_long_term_debt": ("short_term_debt", "Current Portion of Long-Term Debt"),
    "balance_sheet_accrued_expenses": ("accrued_expenses", "Accrued Expenses"),
    "balance_sheet_unearned_revenue": ("unearned_revenue", "Unearned Revenue"),
    "balance_sheet_other_current_liabilities": ("other_current_liabilities", "Other Current Liabilities"),
    "liabilitiesc": ("total_current_liabilities", "Total Current Liabilities"),
    "totalCurrentLiabilities": ("total_current_liabilities", "Total Current Liabilities"),
    # Long-term liabilities
    "longTermDebt": ("long_term_debt", "Long-Term Debt"),
    "balance_sheet_long_term_debt": ("long_term_debt", "Long-Term Debt"),
    "longTermLeases": ("long_term_leases", "Long-Term Leases"),
    "balance_sheet_other_long_term_liabilities": ("other_non_current_liabilities", "Other Non-Current Liabilities"),
    "balance_sheet_total_long_term_liabilities": ("total_long_term_liabilities", "Total Long-Term Liabilities"),
    # Total liabilities
    "totalLiabilities": ("total_liabilities", "Total Liabilities"),
    "totalLiabilitiesRE": ("total_liabilities", "Total Liabilities"),
    "liabilities": ("total_liabilities", "Total Liabilities"),
    # Equity
    "commonStock": ("share_capital", "Share Capital"),
    "balance_sheet_common_stock": ("share_capital", "Share Capital"),
    "additionalPaidInCapital": ("additional_paid_in_capital", "Additional Paid-In Capital"),
    "retearn": ("retained_earnings", "Retained Earnings"),
    "balance_sheet_retained_earnings": ("retained_earnings", "Retained Earnings"),
    "balance_sheet_accumulated_other_comprehensive_income": ("comprehensive_income", "Accumulated Other Comprehensive Income"),
    "totalCommonEquity": ("total_common_equity", "Total Common Equity"),
    "equity": ("total_equity", "Total Equity"),
    "treasuryStock": ("treasury_stock", "Treasury Stock"),
    "minorityInterestBS": ("minority_interest_bs", "Minority Interest"),
    # Totals
    "liabilitiesequity": ("total_liabilities_equity", "Total Liabilities & Equity"),
    # Derived amounts
    "debt": ("total_debt", "Total Debt"),
    "netcash": ("net_cash", "Net Cash / Debt"),
    "netCash": ("net_cash", "Net Cash / Debt"),
    "capitalLeases": ("capital_leases", "Capital Leases"),
    # Per-share
    "bvps": ("book_value_per_share", "Book Value Per Share"),
    "bookValuePerShare": ("book_value_per_share", "Book Value Per Share"),
    "bookValue": ("book_value", "Book Value"),
    "tangibleBookValue": ("tangible_book_value", "Tangible Book Value"),
    "tangibleBookValuePerShare": ("tangible_book_value_per_share", "Tangible Book Value Per Share"),
    "netCashPerShare": ("net_cash_per_share", "Net Cash Per Share"),
}

_SA_FIELD_MAP_CASHFLOW = {
    # Net income
    "netIncomeCF": ("net_income_cf", "Net Income"),
    "cash_flow_statement_net_income": ("net_income_cf", "Net Income"),
    # Depreciation
    "totalDepAmorCF": ("depreciation_cf", "Depreciation & Amortization"),
    "cash_flow_statement_depreciation_and_amortization": ("depreciation_cf", "Depreciation & Amortization"),
    # Stock-based compensation
    "sbcomp": ("stock_based_compensation", "Stock-Based Compensation"),
    # Working capital changes
    "changeWorkingCapital": ("change_working_capital", "Change in Working Capital"),
    "changeInReceivables": ("change_receivables", "Change in Receivables"),
    "cash_flow_statement_changes_in_accounts_payable": ("change_payables", "Change in Payables"),
    "cash_flow_statement_changes_in_inventories": ("change_inventories", "Change in Inventories"),
    "cash_flow_statement_changes_in_income_taxes_payable": ("change_taxes_payable", "Change in Taxes Payable"),
    "cash_flow_statement_changes_in_unearned_revenue": ("change_unearned_revenue", "Change in Unearned Revenue"),
    "cash_flow_statement_changes_in_other_operating_activities": ("other_operating_activities", "Other Operating Activities"),
    "cash_flow_statement_other_adjustments": ("other_adjustments", "Other Adjustments"),
    # Operating cash flow
    "ncfo": ("cash_from_operations", "Cash from Operations"),
    # Investing
    "capex": ("capital_expenditures", "Capital Expenditures"),
    "cash_flow_statement_payments_for_business_acquisitions": ("acquisitions", "Acquisitions"),
    "cash_flow_statement_purchases_of_investments": ("purchases_investments", "Purchases of Investments"),
    "cash_flow_statement_proceeds_from_sale_of_investments": ("sales_investments", "Sales of Investments"),
    "cash_flow_statement_other_investing_activities": ("investing_other", "Other Investing Activities"),
    "ncfi": ("cash_from_investing", "Cash from Investing"),
    # Financing
    "debtIssuedTotal": ("debt_issued", "Debt Issued"),
    "debtissuedlongterm": ("debt_issued_long_term", "Long-Term Debt Issued"),
    "debtissuedshortterm": ("debt_issued_short_term", "Short-Term Debt Issued"),
    "debtRepaidTotal": ("debt_repaid", "Debt Repaid"),
    "debtrepaidlongterm": ("debt_repaid_long_term", "Long-Term Debt Repaid"),
    "netDebtIssued": ("net_debt_issued", "Net Debt Issued"),
    "netdebtissuedlongterm": ("net_debt_issued_long_term", "Net Long-Term Debt Issued"),
    "netdebtissuedshortterm": ("net_debt_issued_short_term", "Net Short-Term Debt Issued"),
    "commonDividendCF": ("dividends_paid", "Dividends Paid"),
    "commondividendcf": ("dividends_paid", "Dividends Paid"),
    "commonIssued": ("shares_issued", "Shares Issued"),
    "commonissued": ("shares_issued", "Shares Issued"),
    "commonRepurchased": ("shares_repurchased", "Shares Repurchased"),
    "commonrepurchased": ("shares_repurchased", "Shares Repurchased"),
    "netstockissued": ("net_equity_total", "Net Stock Issued / Repurchased"),
    "otherfinancing": ("financing_other", "Other Financing Activities"),
    "ncff": ("cash_from_financing", "Cash from Financing"),
    # Exchange rate effect
    "cash_flow_statement_effect_of_exchange_rate_changes_on_cash_and_cash_equivalents": ("exchange_rate_effect", "Effect of Exchange Rate Changes"),
    # Net change in cash
    "ncf": ("net_change_in_cash", "Net Change in Cash"),
    # Free cash flow
    "leveredFCF": ("free_cash_flow", "Free Cash Flow"),
    "unleveredFCF": ("unlevered_fcf", "Unlevered Free Cash Flow"),
    # Other
    "cashInterestPaid": ("interest_paid", "Interest Paid"),
    "acquisitionRealEstateAssets": ("acquisition_re_assets", "RE Asset Acquisitions"),
    "saleRealEstateAssets": ("sale_re_assets", "RE Asset Sales"),
}

_SA_STMT_MAP = {
    "income":   ("https://stockanalysis.com/quote/kwse/{sym}/financials/", _SA_FIELD_MAP_INCOME),
    "balance":  ("https://stockanalysis.com/quote/kwse/{sym}/financials/balance-sheet/", _SA_FIELD_MAP_BALANCE),
    "cashflow": ("https://stockanalysis.com/quote/kwse/{sym}/financials/cash-flow-statement/", _SA_FIELD_MAP_CASHFLOW),
}


def _sa_parse_financial_data(html: str) -> Optional[Dict]:
    """Extract the financialData object from stockanalysis.com SvelteKit page."""
    import re as _re

    m = _re.search(r'financialData:\{(.*?)\},(?:mapData|columns)', html, _re.DOTALL)
    if not m:
        m = _re.search(r'financialData:\{(.+)', html, _re.DOTALL)
        if not m:
            return None

    raw = m.group(1)
    result: Dict[str, Any] = {}

    dk = _re.search(r'datekey:\[([^\]]+)\]', raw)
    if dk:
        result["datekey"] = [s.strip().strip('"') for s in dk.group(1).split(",")]

    fy = _re.search(r'fiscalYear:\[([^\]]+)\]', raw)
    if fy:
        result["fiscalYear"] = [s.strip().strip('"') for s in fy.group(1).split(",")]

    for fm in _re.finditer(r'(\w+):\[([^\]]*)\]', raw):
        name = fm.group(1)
        if name in ("datekey", "fiscalYear", "fiscalQuarter"):
            continue
        vals = []
        for v in fm.group(2).split(","):
            v = v.strip()
            if v in ("null", "void 0", ""):
                vals.append(None)
            else:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(None)
        result[name] = vals

    return result if result.get("datekey") else None


@router.post("/stocks/{stock_id}/fetch-statements-online")
async def fetch_statements_online(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Fetch financial statements online for a stock.

    - Kuwait stocks: scraped from stockanalysis.com (/quote/kwse/{sym}/)
    - US stocks: scraped from stockanalysis.com (/stocks/{sym}/)

    Upserts into financial_statements + financial_line_items.
    """
    import httpx as _httpx

    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    # Resolve symbol from stock_id
    row = query_one("SELECT symbol FROM analysis_stocks WHERE id = ?", (stock_id,))
    if not row:
        raise NotFoundError("Analysis Stock", str(stock_id))
    symbol = row["symbol"] if isinstance(row, dict) else row[0]

    # Determine yf_ticker to check if it's a Kuwait stock
    resolved = _resolve_yf_ticker(symbol, current_user.user_id)
    is_kuwait = resolved.upper().endswith(".KW")

    # Route US stocks to stockanalysis.com (/stocks/ path)
    if not is_kuwait:
        return _fetch_us_statements(stock_id, symbol, current_user.user_id)

    base = resolved.replace(".KW", "").upper()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    now = int(time.time())
    saved_summary = []

    for stmt_type, (url_tpl, field_map) in _SA_STMT_MAP.items():
        url = url_tpl.format(sym=base)
        try:
            resp = _httpx.get(url, headers=headers, timeout=20, follow_redirects=True)
            if resp.status_code != 200:
                logger.warning("stockanalysis.com %s returned %s for %s", stmt_type, resp.status_code, base)
                continue
        except Exception as exc:
            logger.warning("stockanalysis.com fetch error for %s/%s: %s", base, stmt_type, exc)
            continue

        data = _sa_parse_financial_data(resp.text)
        if not data:
            logger.warning("stockanalysis.com: no financialData for %s/%s", base, stmt_type)
            continue

        datekeys = data.get("datekey", [])
        fiscal_years = data.get("fiscalYear", [])

        # Process each period (skip TTM)
        periods_saved = 0
        for idx, dk in enumerate(datekeys):
            if dk == "TTM":
                continue

            period_end_date = dk  # e.g. "2024-12-31"
            try:
                fy = int(fiscal_years[idx])
            except (IndexError, ValueError):
                continue

            # Collect line items for this period
            line_items = []
            seen_codes = set()
            order = 1
            for sa_key, (canonical_code, display_name) in field_map.items():
                if sa_key not in data:
                    continue
                # Skip duplicate canonical codes (e.g. revenue vs revenueRE)
                if canonical_code in seen_codes:
                    continue
                vals = data[sa_key]
                if idx >= len(vals) or vals[idx] is None:
                    continue
                seen_codes.add(canonical_code)
                line_items.append({
                    "code": canonical_code,
                    "name": display_name,
                    "amount": vals[idx],
                    "currency": "KWD",
                    "order": order,
                    "is_total": canonical_code in (
                        "total_assets", "total_equity", "total_liabilities",
                        "total_current_assets", "total_current_liabilities",
                        "total_liabilities_equity", "total_operating_expenses",
                        "total_common_equity",
                    ),
                })
                order += 1

            # ── Include ALL remaining fields not in field_map ────────
            mapped_sa_keys = set(field_map.keys())
            for extra_key, vals in data.items():
                if extra_key in _SA_METADATA_KEYS:
                    continue
                if extra_key in _SA_RATIO_KEYS:
                    continue
                if extra_key in mapped_sa_keys:
                    continue
                if not isinstance(vals, list):
                    continue
                if extra_key in seen_codes:
                    continue
                if idx >= len(vals) or vals[idx] is None:
                    continue
                seen_codes.add(extra_key)
                line_items.append({
                    "code": extra_key,
                    "name": _camel_to_display(extra_key),
                    "amount": vals[idx],
                    "currency": "KWD",
                    "order": order,
                    "is_total": False,
                })
                order += 1

            if not line_items:
                continue

            # Upsert: check existing statement
            existing = query_one(
                """SELECT id FROM financial_statements
                   WHERE stock_id = ? AND statement_type = ? AND period_end_date = ?""",
                (stock_id, stmt_type, period_end_date),
            )

            with get_connection() as conn:
                cur = conn.cursor()
                if existing:
                    stmt_id = existing["id"] if isinstance(existing, dict) else existing[0]
                    cur.execute(
                        """UPDATE financial_statements
                           SET fiscal_year=?, extracted_by=?, source_file=?,
                               confidence_score=?, notes=?, created_at=?
                           WHERE id=?""",
                        (fy, "stockanalysis.com", f"stockanalysis.com/kwse/{base}",
                         1.0, "Fetched from stockanalysis.com", now, stmt_id),
                    )
                    cur.execute("DELETE FROM financial_line_items WHERE statement_id = ?", (stmt_id,))
                else:
                    cur.execute(
                        """INSERT INTO financial_statements
                           (stock_id, statement_type, fiscal_year, fiscal_quarter,
                            period_end_date, source_file, extracted_by,
                            confidence_score, notes, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (stock_id, stmt_type, fy, None, period_end_date,
                         f"stockanalysis.com/kwse/{base}", "stockanalysis.com",
                         1.0, "Fetched from stockanalysis.com", now),
                    )
                    stmt_id = cur.lastrowid

                for item in line_items:
                    cur.execute(
                        """INSERT INTO financial_line_items
                           (statement_id, line_item_code, line_item_name,
                            amount, currency, order_index, is_total)
                           VALUES (?,?,?,?,?,?,?)""",
                        (stmt_id, item["code"], item["name"],
                         item["amount"], item["currency"],
                         item["order"], item["is_total"]),
                    )
                conn.commit()
            periods_saved += 1

        if periods_saved > 0:
            saved_summary.append({
                "statement_type": stmt_type,
                "periods_saved": periods_saved,
            })

    if not saved_summary:
        raise HTTPException(
            status_code=404,
            detail=f"No financial data found on stockanalysis.com for {base}.",
        )

    total_periods = sum(s["periods_saved"] for s in saved_summary)
    return {
        "status": "ok",
        "data": {
            "message": f"Fetched {total_periods} period(s) across {len(saved_summary)} statement type(s) from stockanalysis.com.",
            "summary": saved_summary,
            "source": f"https://stockanalysis.com/quote/kwse/{base}/financials/",
        },
    }


@router.post("/stocks/{stock_id}/statements/reorder-items")
async def reorder_line_items(
    stock_id: int,
    body: ReorderItemsRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Bulk-update order_index for line items (drag-and-drop reorder)."""
    _ensure_schema()
    if not body.items:
        raise BadRequestError("No items provided.")

    with get_connection() as conn:
        cur = conn.cursor()
        updated = 0
        for item in body.items:
            # Verify ownership via join chain
            cur.execute(
                """SELECT li.id FROM financial_line_items li
                   JOIN financial_statements fs ON li.statement_id = fs.id
                   JOIN analysis_stocks s ON fs.stock_id = s.id
                   WHERE li.id = ? AND s.id = ? AND s.user_id = ?""",
                (item.id, stock_id, current_user.user_id),
            )
            if cur.fetchone():
                cur.execute(
                    "UPDATE financial_line_items SET order_index = ? WHERE id = ?",
                    (item.order_index, item.id),
                )
                updated += 1
        conn.commit()
    return {"status": "ok", "data": {"message": f"{updated} item(s) reordered.", "updated": updated}}


@router.post("/stocks/{stock_id}/line-items", status_code=201)
async def create_line_item(
    stock_id: int,
    body: CreateLineItemRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Create a single line item in an existing statement (e.g. filling a dash)."""
    _ensure_schema()
    # Verify the statement belongs to this stock and user
    row = query_one(
        """SELECT fs.id FROM financial_statements fs
           JOIN analysis_stocks s ON fs.stock_id = s.id
           WHERE fs.id = ? AND s.id = ? AND s.user_id = ?""",
        (body.statement_id, stock_id, current_user.user_id),
    )
    if not row:
        raise NotFoundError("Statement", str(body.statement_id))

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO financial_line_items
               (statement_id, line_item_code, line_item_name, amount,
                currency, order_index, is_total, manually_edited,
                edited_by_user_id, edited_at)
               VALUES (?, ?, ?, ?, 'KWD', ?, 0, 1, ?, ?)""",
            (
                body.statement_id, body.line_item_code, body.line_item_name,
                body.amount, body.order_index,
                current_user.user_id, int(time.time()),
            ),
        )
        new_id = cur.lastrowid
        conn.commit()
    return {"status": "ok", "data": {"id": new_id, "message": "Line item created."}}


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


@router.delete("/line-items/{item_id}")
async def delete_line_item(
    item_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Delete a single line item (row removal from statement table)."""
    _ensure_schema()
    row = query_one(
        """SELECT li.id, li.line_item_code
           FROM financial_line_items li
           JOIN financial_statements fs ON li.statement_id = fs.id
           JOIN analysis_stocks       s  ON fs.stock_id    = s.id
           WHERE li.id = ? AND s.user_id = ?""",
        (item_id, current_user.user_id),
    )
    if not row:
        raise NotFoundError("Line Item", str(item_id))
    exec_sql("DELETE FROM financial_line_items WHERE id = ?", (item_id,))
    return {"status": "ok", "data": {"message": "Line item deleted."}}


@router.post("/stocks/{stock_id}/merge-line-items")
async def merge_line_items(
    stock_id: int,
    body: MergeLineItemsRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Merge two line items across all periods.

    For each period/statement: if keep_code has no value (or is 0/dash) but
    remove_code does, copy remove_code's amount into keep_code.  Then delete
    all remove_code items.
    """
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    keep_code = body.keep_code
    remove_code = body.remove_code
    if keep_code == remove_code:
        raise BadRequestError("Cannot merge a line item with itself.")

    now = int(time.time())
    merged_count = 0
    deleted_count = 0

    with get_connection() as conn:
        cur = conn.cursor()

        # Get all statements for this stock
        stmt_rows = cur.execute(
            "SELECT id FROM financial_statements WHERE stock_id = ?",
            (stock_id,),
        ).fetchall()

        for stmt_row in stmt_rows:
            sid = stmt_row["id"] if isinstance(stmt_row, dict) else stmt_row[0]

            # Find keep_code item in this statement
            keep_item = cur.execute(
                "SELECT id, amount FROM financial_line_items WHERE statement_id = ? AND line_item_code = ?",
                (sid, keep_code),
            ).fetchone()

            # Find remove_code item in this statement
            remove_item = cur.execute(
                "SELECT id, amount, line_item_name, order_index, is_total FROM financial_line_items WHERE statement_id = ? AND line_item_code = ?",
                (sid, remove_code),
            ).fetchone()

            if remove_item is None:
                continue

            r_id = remove_item["id"] if isinstance(remove_item, dict) else remove_item[0]
            r_amount = remove_item["amount"] if isinstance(remove_item, dict) else remove_item[1]
            r_name = remove_item["line_item_name"] if isinstance(remove_item, dict) else remove_item[2]
            r_order = remove_item["order_index"] if isinstance(remove_item, dict) else remove_item[3]
            r_is_total = remove_item["is_total"] if isinstance(remove_item, dict) else remove_item[4]

            if keep_item is None:
                # No keep_code in this period — re-label the remove item
                cur.execute(
                    "UPDATE financial_line_items SET line_item_code = ? WHERE id = ?",
                    (keep_code, r_id),
                )
                merged_count += 1
            else:
                k_id = keep_item["id"] if isinstance(keep_item, dict) else keep_item[0]
                k_amount = keep_item["amount"] if isinstance(keep_item, dict) else keep_item[1]

                # If keep has 0/null but remove has a real value, copy it over
                if (k_amount is None or k_amount == 0) and r_amount is not None and r_amount != 0:
                    cur.execute(
                        "UPDATE financial_line_items SET amount = ?, manually_edited = 1, edited_at = ? WHERE id = ?",
                        (r_amount, now, k_id),
                    )
                    merged_count += 1

                # Delete the remove item
                cur.execute("DELETE FROM financial_line_items WHERE id = ?", (r_id,))
                deleted_count += 1

        conn.commit()

    return {
        "status": "ok",
        "data": {
            "message": f"Merged '{remove_code}' into '{keep_code}': {merged_count} values merged, {deleted_count} duplicates removed.",
            "merged_count": merged_count,
            "deleted_count": deleted_count,
        },
    }


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
# Canonical code map — all values are lowercase for consistency across years.
# Uses the same canonical forms as _MERGE_CODES in extraction_service.py.
_CANONICAL_CODES: Dict[str, str] = {
    # Income statement
    "revenue": "revenue", "total_revenue": "revenue", "net_revenue": "revenue",
    "sales": "revenue", "net_sales": "revenue",
    "cost_of_revenue": "cost_of_revenue", "cost_of_sales": "cost_of_revenue",
    "cost_of_goods_sold": "cost_of_revenue", "cogs": "cost_of_revenue",
    "gross_profit": "gross_profit",
    "selling_general_administrative": "sga", "sga": "sga",
    "selling_general_and_administrative": "sga",
    "research_and_development": "r_and_d", "r_and_d": "r_and_d", "r&d": "r_and_d",
    "operating_expenses": "operating_expenses", "total_operating_expenses": "operating_expenses",
    "operating_income": "operating_income", "operating_profit": "operating_income",
    "income_from_operations": "operating_income",
    "interest_expense": "interest_expense", "finance_costs": "interest_expense",
    "finance_cost": "interest_expense",
    "other_income": "other_income", "other_income_expense": "other_income",
    "income_before_tax": "income_before_tax", "profit_before_tax": "income_before_tax",
    "income_tax": "income_tax", "income_tax_expense": "income_tax",
    "tax_expense": "income_tax", "taxation": "income_tax",
    "net_income": "net_income", "net_profit": "net_income",
    "profit_for_the_year": "net_income", "profit_for_the_period": "net_income",
    "profit_for_year": "net_income",
    "net_income_attributable_to_shareholders": "net_income",
    "eps_basic": "eps_basic", "basic_eps": "eps_basic",
    "basic_earnings_per_share": "eps_basic", "earnings_per_share_basic": "eps_basic",
    "basic_and_diluted_earnings_per_share": "eps_basic",
    "basic_and_diluted_earnings_per_sha": "eps_basic",
    "earnings_per_share": "eps_basic",
    "eps_diluted": "eps_diluted", "diluted_eps": "eps_diluted",
    "diluted_earnings_per_share": "eps_diluted", "earnings_per_share_diluted": "eps_diluted",
    "ebitda": "ebitda",
    "depreciation_and_amortization": "depreciation_amortization",
    "depreciation_amortization": "depreciation_amortization",
    # Balance sheet
    "cash": "cash", "cash_and_cash_equivalents": "cash", "cash_equivalents": "cash",
    "cash_and_bank_balances": "cash", "cash_and_balances_with_banks": "cash",
    "accounts_receivable": "accounts_receivable", "trade_receivables": "accounts_receivable",
    "receivables": "accounts_receivable", "trade_and_other_receivables": "accounts_receivable",
    "inventory": "inventory", "inventories": "inventory",
    "other_current_assets": "other_current_assets",
    "total_current_assets": "total_current_assets",
    "property_plant_equipment": "ppe_net", "property_plant_and_equipment": "ppe_net",
    "ppe_net": "ppe_net", "fixed_assets": "ppe_net",
    "property_and_equipment": "ppe_net",
    "goodwill": "goodwill",
    "intangible_assets": "intangible_assets", "intangibles": "intangible_assets",
    "total_non_current_assets": "total_non_current_assets",
    "total_assets": "total_assets",
    "accounts_payable": "accounts_payable", "trade_payables": "accounts_payable",
    "trade_and_other_payables": "accounts_payable",
    "short_term_debt": "short_term_debt", "current_portion_of_debt": "short_term_debt",
    "current_portion_of_long_term_debt": "short_term_debt",
    "short_term_borrowings": "short_term_debt",
    "total_current_liabilities": "total_current_liabilities",
    "long_term_debt": "long_term_debt", "long_term_borrowings": "long_term_debt",
    "non_current_borrowings": "long_term_debt",
    "total_non_current_liabilities": "total_non_current_liabilities",
    "total_liabilities": "total_liabilities",
    "common_stock": "share_capital", "share_capital": "share_capital",
    "issued_capital": "share_capital",
    "retained_earnings": "retained_earnings",
    "share_premium": "share_premium",
    "statutory_reserve": "statutory_reserve",
    "voluntary_reserve": "voluntary_reserve",
    "general_reserve": "general_reserve",
    "treasury_shares": "treasury_shares", "treasury_shares_equity": "treasury_shares",
    "total_equity": "total_equity", "total_shareholders_equity": "total_equity",
    "total_stockholders_equity": "total_equity",
    "equity_attributable_to_shareholders": "total_equity",
    "total_liabilities_and_equity": "total_liabilities_and_equity",
    "total_liabilities_and_shareholders_equity": "total_liabilities_and_equity",
    "total_liabilities_equity": "total_liabilities_and_equity",
    # Cash flow
    "net_income_cf": "net_income_cf",
    "cash_from_operations": "cash_from_operations",
    "cash_from_operating_activities": "cash_from_operations",
    "net_cash_from_operating_activities": "cash_from_operations",
    "net_cash_used_in_operating_activities": "cash_from_operations",
    "cash_used_in_operating_activities": "cash_from_operations",
    "capital_expenditures": "capital_expenditures", "capex": "capital_expenditures",
    "purchase_of_property_plant_equipment": "capital_expenditures",
    "purchase_of_property_plant_and_equipment": "capital_expenditures",
    "other_investing_activities": "other_investing",
    "other_investing": "other_investing",
    "cash_from_investing": "cash_from_investing",
    "cash_from_investing_activities": "cash_from_investing",
    "net_cash_from_investing_activities": "cash_from_investing",
    "net_cash_used_in_investing_activities": "cash_from_investing",
    "cash_used_in_investing_activities": "cash_from_investing",
    "debt_issued": "debt_issued", "proceeds_from_borrowings": "debt_issued",
    "debt_repaid": "debt_repaid", "repayment_of_borrowings": "debt_repaid",
    "dividends_paid": "dividends_paid", "dividend_paid": "dividends_paid",
    "cash_from_financing": "cash_from_financing",
    "cash_from_financing_activities": "cash_from_financing",
    "net_cash_from_financing_activities": "cash_from_financing",
    "net_cash_used_in_financing_activities": "cash_from_financing",
    "cash_used_in_financing_activities": "cash_from_financing",
    "net_change_in_cash": "net_change_in_cash",
    "net_change_cash": "net_change_in_cash",
    "net_increase_decrease_in_cash": "net_change_in_cash",
    "changes_in_working_capital": "changes_in_working_capital",
    "changes_working_capital": "changes_in_working_capital",
    # Free Cash Flow variants
    "free_cash_flow": "free_cash_flow",
    "fcf": "free_cash_flow",
    "unlevered_free_cash_flow": "free_cash_flow",
    "levered_free_cash_flow": "levered_free_cash_flow",
    "free_cash_flow_margin": "free_cash_flow_margin",
    "fcf_margin": "free_cash_flow_margin",
    "free_cash_flow_per_share": "free_cash_flow_per_share",
    "fcf_per_share": "free_cash_flow_per_share",
    "free_cash_flow_growth": "free_cash_flow_growth",
    "operating_cash_flow": "cash_from_operations",
    # Equity statement
    "shares_outstanding": "shares_diluted", "share_count": "share_count",
    "shares_basic": "shares_basic", "shares_diluted": "shares_diluted",
    # GCC / Kuwait specific
    "cost_of_operations": "cost_of_revenue",
    "general_and_administrative_expenses": "general_and_admin",
    "general_and_administrative_expens": "general_and_admin",
    "selling_expenses": "selling_expenses",
    "selling_and_distribution_expenses": "selling_expenses",
    "finance_charges": "finance_charges", "finance_charge": "finance_charges",
    "profit_before_contribution_to_kfas": "profit_before_deductions",
    "profit_before_contribution_to_kuwait_foundation_for_advancement_of_sciences": "profit_before_deductions",
    "profit_before_contribution_to_kuwait_foundation_for_the_advancement_of_sciences": "profit_before_deductions",
    "contribution_to_kfas": "contribution_to_kfas",
    "contribution_kfas": "contribution_to_kfas",
    "contribution_to_kuwait_foundation_for_advancement_of_sciences": "contribution_to_kfas",
    "contribution_to_kuwait_foundation_for_the_advancement_of_sciences": "contribution_to_kfas",
    "national_labour_support_tax": "nlst", "nlst": "nlst",
    "national_labor_support_tax": "nlst",
    "zakat": "zakat",
    "directors_remuneration": "directors_remuneration",
    "directors_fees": "directors_remuneration",
    "board_of_directors_remuneration": "directors_remuneration",
    "share_of_profit_of_associates": "share_results_associates",
    "share_of_loss_of_associates": "share_results_associates",
    "share_of_results_of_associates": "share_results_associates",
    "share_of_profit_loss_of_associates": "share_results_associates",
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
    """Map an AI-extracted key to a canonical lowercase line item code."""
    import re as _re
    k = raw_key.strip().lower().replace(" ", "_").replace("-", "_")
    k = _re.sub(r"_+", "_", k).strip("_")
    # Direct lookup in canonical map
    if k in _CANONICAL_CODES:
        return _CANONICAL_CODES[k]
    # Try without trailing _total, _net etc.
    for suffix in ("_total", "_net", "_and_equivalents"):
        if k.endswith(suffix):
            trimmed = k[: -len(suffix)].rstrip("_")
            if trimmed in _CANONICAL_CODES:
                return _CANONICAL_CODES[trimmed]
    # Fallback: clean lowercase snake_case (never uppercase)
    return k


def _reconcile_code_with_existing(
    new_code: str,
    new_name: str,
    existing_codes_map: Dict[str, str],
) -> str:
    """Fuzzy-match a newly extracted code against existing codes for the same stock.

    All comparisons are case-insensitive. If the new_code already exists
    (case-insensitively), return the existing version. Otherwise, check if
    any existing code shares enough word overlap to be the same financial
    concept.

    Returns the best matching existing code, or the original new_code if no match.
    """
    if not existing_codes_map:
        return new_code

    # Build a lowercase → original-code lookup for case-insensitive matching
    lower_map: Dict[str, str] = {k.lower(): k for k in existing_codes_map}

    # Case-insensitive exact match
    new_lower = new_code.lower()
    if new_lower in lower_map:
        return lower_map[new_lower]

    # Check if both map to the same canonical code
    new_canonical = _CANONICAL_CODES.get(new_lower)
    if new_canonical:
        for ex_code in existing_codes_map:
            ex_canonical = _CANONICAL_CODES.get(ex_code.lower())
            if ex_canonical and ex_canonical == new_canonical:
                logger.info(
                    "Code reconciled via canonical map: '%s' → '%s' (canonical=%s)",
                    new_code, ex_code, new_canonical,
                )
                return ex_code

    # Build word sets for fuzzy comparison
    new_words = set(new_lower.replace("_", " ").split())
    filler = {"for", "the", "of", "and", "in", "on", "to", "a", "an", "total"}
    new_meaningful = new_words - filler

    if not new_meaningful:
        return new_code

    best_code = new_code
    best_score = 0.0

    for ex_code in existing_codes_map:
        ex_words = set(ex_code.lower().replace("_", " ").split())
        ex_meaningful = ex_words - filler

        if not ex_meaningful:
            continue

        # Jaccard-like overlap on meaningful words
        intersection = new_meaningful & ex_meaningful
        union = new_meaningful | ex_meaningful
        score = len(intersection) / len(union) if union else 0.0

        if score > best_score:
            best_score = score
            best_code = ex_code

    # Threshold: 60% word overlap = same concept
    if best_score >= 0.6:
        logger.info(
            "Code reconciled: '%s' → '%s' (score=%.2f)",
            new_code, best_code, best_score,
        )
        return best_code

    return new_code


def _persist_extraction_result(
    stock_id: int,
    result,
    stock_currency: str,
    source_file: Optional[str],
    extracted_by: str,
    notes_template: str = "AI-extracted",
) -> tuple:
    """Shared persist logic for all extraction/validation endpoints.

    Returns (created_statements, total_items).
    Uses _resolve_amount (key-based matching) instead of unsafe positional fallback.

    **Cash flow statements** are routed through a staging→reconcile→commit pipeline
    instead of being saved directly. Only reconciled, accepted rows reach
    financial_line_items.
    """
    from app.services.extraction_service import _resolve_amount, _MISSING

    now = int(time.time())
    created_statements = []
    total_items = 0

    # Build map of existing codes for this stock (for fuzzy reconciliation)
    existing_codes_map: Dict[str, str] = {}
    try:
        rows = query_all(
            """SELECT DISTINCT li.line_item_code, fs.statement_type
               FROM financial_line_items li
               JOIN financial_statements fs ON li.statement_id = fs.id
               WHERE fs.stock_id = ?""",
            (stock_id,),
        )
        for r in rows:
            existing_codes_map[r["line_item_code"]] = r["statement_type"]
    except Exception:
        pass

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

            # ── CASH FLOW → staging pipeline ─────────────────────────
            if stmt.statement_type == "cashflow":
                cf_result = _stage_and_reconcile_cashflow(
                    stock_id=stock_id,
                    stmt=stmt,
                    result=result,
                    stock_currency=currency,
                    source_file=source_file,
                    extracted_by=extracted_by,
                    notes_template=notes_template,
                    existing_codes_map=existing_codes_map,
                    cur=cur,
                    now=now,
                )
                created_statements.extend(cf_result["created_statements"])
                total_items += cf_result["total_items"]
                continue

            # ── Non-cashflow: direct persist (income / balance / equity) ─
            for period_info in stmt.periods:
                period_label = period_info.get("label", "")

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
                        (fiscal_year, source_file, extracted_by,
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
                            period_end_date, None, source_file, extracted_by,
                            result.confidence, notes_template, now,
                        ),
                    )
                    stmt_id = cur.lastrowid

                # Insert line items — key-based matching only (no positional fallback)
                seen_codes: set = set()
                for item in stmt.items:
                    code = _normalize_key(item.key)
                    # Reconcile with existing codes to prevent duplicates
                    code = _reconcile_code_with_existing(
                        code, item.label_raw, existing_codes_map,
                    )
                    if code in seen_codes:
                        continue
                    amount = _resolve_amount(item, period_info, stmt.periods)
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

                created_statements.append({
                    "statement_id": stmt_id,
                    "statement_type": stmt.statement_type,
                    "period_end_date": period_end_date,
                    "fiscal_year": fiscal_year,
                    "line_items_count": len(stmt.items),
                    "currency": currency,
                })

        conn.commit()

    return created_statements, total_items


def _stage_and_reconcile_cashflow(
    stock_id: int,
    stmt,
    result,
    stock_currency: str,
    source_file: Optional[str],
    extracted_by: str,
    notes_template: str,
    existing_codes_map: Dict[str, str],
    cur,
    now: int,
) -> Dict[str, Any]:
    """I) Stage cash flow rows, reconcile, and commit ONLY if reconciliation passes.

    If reconciliation fails (needs_review):
     - rows are persisted in staging
     - run is marked needs_review
     - NO rows are committed to financial_line_items
     - enough metadata is preserved for UI/debugging

    Returns dict with created_statements and total_items.
    """
    from app.services.extraction_service import _resolve_amount, _MISSING
    from app.services.cashflow_reconciler import (
        reconcile_cashflow, ReconcileResult, compute_validated_cashflow_metrics,
    )

    created_statements = []
    total_items = 0

    # Build period labels list
    period_labels = [p.get("label", p.get("col_name", "")) for p in stmt.periods]

    # ── 1. Create extraction run ─────────────────────────────────────
    cur.execute(
        """INSERT INTO cashflow_extraction_runs
           (stock_id, source_file, status, periods_json, created_at, updated_at)
           VALUES (?,?,?,?,?,?)""",
        (stock_id, source_file, "extracted",
         json.dumps(period_labels), now, now),
    )
    run_id = cur.lastrowid

    # ── 2. Build raw items with resolved values per period ───────────
    raw_items_for_reconcile: List[Dict[str, Any]] = []
    for idx, item in enumerate(stmt.items):
        resolved_values: Dict[str, Optional[float]] = {}
        for period_info in stmt.periods:
            period_label = period_info.get("label", "")
            amount = _resolve_amount(item, period_info, stmt.periods)
            if amount is _MISSING:
                resolved_values[period_label] = None
            else:
                resolved_values[period_label] = float(amount) if amount is not None else None

        raw_item = {
            "label_raw": item.label_raw,
            "key": item.key,
            "values": resolved_values,
            "is_total": item.is_total,
            "order_index": item.order_index or (idx + 1),
            "section": getattr(item, "section", "unknown"),
            "row_kind": getattr(item, "row_kind", "item"),
        }
        raw_items_for_reconcile.append(raw_item)

    # ── 3. Run deterministic reconciliation ──────────────────────────
    recon: ReconcileResult = reconcile_cashflow(raw_items_for_reconcile, period_labels)

    # ── 4. Stage ALL rows (raw + reconciliation results) ─────────────
    for row in recon.rows:
        cur.execute(
            """INSERT INTO cashflow_staged_rows
               (run_id, row_order, label_raw, normalized_code, section,
                row_kind, values_json, is_total, confidence,
                is_accepted, rejection_reason, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, row.row_order, row.label_raw,
                row.normalized_code, row.section, row.row_kind,
                json.dumps(row.values), row.is_total, row.confidence,
                row.is_accepted, row.rejection_reason, now,
            ),
        )

    # ── 5. Compute validated metrics from accepted rows ──────────────
    cf_metrics = compute_validated_cashflow_metrics(recon.rows, period_labels)

    # ── 6. Update run with reconciliation summary + metrics ──────────
    recon_summary = {
        "status": recon.status,
        "summary": recon.summary,
        "warnings": recon.warnings,
        "errors": recon.errors,
        "validated_metrics": {
            p: {k: v for k, v in m.items() if not str(k).startswith("_")}
            for p, m in cf_metrics.items()
        },
    }
    new_status = "reconciled" if recon.status == "reconciled" else "needs_review"
    cur.execute(
        """UPDATE cashflow_extraction_runs
           SET status=?, reconciliation_summary=?, updated_at=?
           WHERE id=?""",
        (new_status, json.dumps(recon_summary), now, run_id),
    )

    # ── I) Log reconciliation issues but always commit accepted rows ─
    if recon.status != "reconciled":
        logger.warning(
            "Cash flow run %d: reconciliation=%s — committing accepted rows despite errors. "
            "Errors: %s",
            run_id, recon.status, recon.errors,
        )

    accepted_rows = [r for r in recon.rows if r.is_accepted]
    if not accepted_rows:
        logger.warning("Cash flow run %d: no accepted rows to commit", run_id)
        created_statements.append({
            "statement_type": "cashflow",
            "staged_run_id": run_id,
            "reconcile_status": recon.status,
            "needs_review": recon.status != "reconciled",
            "errors": recon.errors[:5],
        })
        return {"created_statements": created_statements, "total_items": 0}

    # ── 7. Commit accepted rows to financial_line_items ──────────────
    for period_info in stmt.periods:
        period_label = period_info.get("label", "")
        period_end_date = period_label
        if len(period_end_date) == 4:
            period_end_date = f"{period_end_date}-12-31"

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

        # Upsert statement
        existing = query_one(
            """SELECT id FROM financial_statements
               WHERE stock_id = ? AND statement_type = ? AND period_end_date = ?""",
            (stock_id, "cashflow", period_end_date),
        )
        if existing:
            stmt_id = existing["id"]
            cur.execute(
                """UPDATE financial_statements
                   SET fiscal_year=?, source_file=?, extracted_by=?,
                       confidence_score=?, notes=?, created_at=?
                   WHERE id=?""",
                (fiscal_year, source_file, extracted_by,
                 result.confidence,
                 f"{notes_template} (staged run #{run_id}, {new_status})",
                 now, stmt_id),
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
                    stock_id, "cashflow", fiscal_year, None,
                    period_end_date, None, source_file, extracted_by,
                    result.confidence,
                    f"{notes_template} (staged run #{run_id}, {new_status})",
                    now,
                ),
            )
            stmt_id = cur.lastrowid

        # I) Insert only accepted rows — preserve section, ordering, raw label, normalized code
        seen_codes: set = set()
        for row in accepted_rows:
            # G) Headers must never be committed as financial line items
            if row.row_kind == "header":
                continue

            code = _normalize_key(row.normalized_code or row.label_raw)
            code = _reconcile_code_with_existing(
                code, row.label_raw, existing_codes_map,
            )
            if code in seen_codes:
                continue

            amount = row.values.get(period_label)
            if amount is None:
                amount = 0.0

            cur.execute(
                """INSERT INTO financial_line_items
                   (statement_id, line_item_code, line_item_name,
                    amount, currency, order_index, is_total)
                   VALUES (?,?,?,?,?,?,?)""",
                (stmt_id, code, row.label_raw, float(amount),
                 stock_currency, row.row_order, row.is_total),
            )
            seen_codes.add(code)
            total_items += 1

        created_statements.append({
            "statement_id": stmt_id,
            "statement_type": "cashflow",
            "period_end_date": period_end_date,
            "fiscal_year": fiscal_year,
            "line_items_count": total_items,
            "currency": stock_currency,
            "staged_run_id": run_id,
            "reconcile_status": recon.status,
        })

    # Update run status to committed
    cur.execute(
        """UPDATE cashflow_extraction_runs
           SET status='committed', updated_at=?
           WHERE id=?""",
        (now, run_id),
    )

    logger.info(
        "Cash flow staged run #%d: %d/%d rows accepted → committed to %d period(s)",
        run_id, len(accepted_rows), len(recon.rows), len(stmt.periods),
    )

    return {"created_statements": created_statements, "total_items": total_items}


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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    force: bool = Query(False, description="Skip cache and re-extract from scratch"),
    model: str = Query("gemini-2.5-flash", description="Gemini model to use: gemini-2.5-flash or gemini-2.5-pro"),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Upload a financial report PDF and start async AI extraction.

    Phase 1 (fast): Validate file, save PDF, create extraction job.
    Phase 2 (async): BackgroundTasks runs the full AI pipeline.
    Frontend polls GET /extraction-status/{job_id} for progress.
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

    # Validate model name
    allowed_models = {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-pro-preview-03-25"}
    if model not in allowed_models:
        raise BadRequestError(f"Invalid model. Choose from: {', '.join(sorted(allowed_models))}")

    # ── 3. Deduplicate: reject if same file already has an active job ─
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    active = query_one(
        """SELECT id FROM extraction_jobs
           WHERE stock_id = ? AND pdf_hash = ? AND status IN ('queued', 'running')
           ORDER BY created_at DESC LIMIT 1""",
        (stock_id, pdf_hash),
    )
    if active:
        active_id = active["id"] if isinstance(active, dict) else active[0]
        return {
            "status": "ok",
            "data": {
                "job_id": active_id,
                "upload_id": str(active_id),
                "status": "running",
                "message": "An extraction job is already in progress for this file.",
                "source_file": file.filename,
            },
        }

    # ── 4. Save PDF immediately ──────────────────────────────────────
    try:
        pdf_upload_id = _save_pdf_file(
            stock_id, current_user.user_id, pdf_bytes,
            file.filename or "upload.pdf",
        )
    except Exception as e:
        logger.warning("PDF save failed: %s", e)
        pdf_upload_id = None

    # ── 5. Fetch existing codes for AI reuse ─────────────────────────
    existing_codes: List[Dict[str, str]] = []
    try:
        rows = query_all(
            """SELECT DISTINCT li.line_item_code, li.line_item_name, fs.statement_type
               FROM financial_line_items li
               JOIN financial_statements fs ON li.statement_id = fs.id
               WHERE fs.stock_id = ?
               ORDER BY fs.statement_type, li.order_index""",
            (stock_id,),
        )
        seen: set = set()
        for r in rows:
            code = r["line_item_code"]
            if code not in seen:
                existing_codes.append({
                    "code": code,
                    "name": r["line_item_name"],
                    "type": r["statement_type"],
                })
                seen.add(code)
    except Exception as exc:
        logger.warning("Could not fetch existing codes: %s", exc)

    # ── 6. Create extraction job record ──────────────────────────────
    now = int(time.time())
    exec_sql(
        """INSERT INTO extraction_jobs
           (stock_id, user_id, pdf_upload_id, pdf_hash, source_file, status,
            stage, model, attempt_count, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (stock_id, current_user.user_id, pdf_upload_id, pdf_hash,
         file.filename, "queued", "uploading", model, 1, now, now),
    )
    job_id = query_val(
        """SELECT id FROM extraction_jobs
           WHERE stock_id = ? AND user_id = ? AND created_at = ?
           ORDER BY id DESC LIMIT 1""",
        (stock_id, current_user.user_id, now),
    )

    # ── 7. Launch background extraction via BackgroundTasks ──────────
    background_tasks.add_task(
        _run_extraction_job_sync,
        job_id=job_id,
        stock_id=stock_id,
        user_id=current_user.user_id,
        pdf_bytes=pdf_bytes,
        filename=file.filename or "upload.pdf",
        model=model,
        force=force,
        api_key=api_key,
        existing_codes=existing_codes,
    )

    _log_job(job_id, stock_id, "created", filename=file.filename,
             model=model, pdf_hash=pdf_hash[:12])

    return {
        "status": "ok",
        "data": {
            "job_id": job_id,
            "upload_id": str(job_id),
            "status": "queued",
            "message": "File uploaded successfully. Extraction is running…",
            "source_file": file.filename,
        },
    }


@router.get("/extraction-status/{job_id}")
async def get_extraction_status(
    job_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Poll extraction job progress.

    Returns current status, stage, progress, and result when done.
    """
    _ensure_schema()

    row = query_one(
        "SELECT * FROM extraction_jobs WHERE id = ?",
        (job_id,),
    )
    if not row:
        raise NotFoundError("Extraction job not found.")

    r = dict(row) if not isinstance(row, dict) else row

    # Verify ownership
    if r.get("user_id") != current_user.user_id:
        raise NotFoundError("Extraction job not found.")

    # Parse result_payload if done
    result_data = None
    if r.get("result_payload"):
        try:
            result_data = json.loads(r["result_payload"])
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "status": "ok",
        "data": {
            "job_id": r["id"],
            "upload_id": str(r["id"]),
            "status": r["status"],
            "stage": r.get("stage", "uploading"),
            "pages_processed": r.get("pages_processed", 0),
            "total_pages": r.get("total_pages", 0),
            "progress_percent": r.get("progress_percent", 0),
            "model": r.get("model"),
            "error_message": r.get("error_message"),
            "result": result_data,
            "source_file": r.get("source_file"),
            "pdf_hash": r.get("pdf_hash"),
            "attempt_count": r.get("attempt_count", 1),
            "created_at": r.get("created_at"),
            "started_at": r.get("started_at"),
            "updated_at": r.get("updated_at"),
            "last_heartbeat_at": r.get("last_heartbeat_at"),
            "completed_at": r.get("completed_at"),
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
    stock_row = query_one(
        "SELECT currency FROM analysis_stocks WHERE id = ?",
        (stock_id,),
    )
    stock_currency = stock_row["currency"] if stock_row else "USD"

    updated_statements, total_items = _persist_extraction_result(
        stock_id=stock_id,
        result=result,
        stock_currency=stock_currency,
        source_file=file.filename,
        extracted_by="gemini-ai-validated",
        notes_template=f"AI-validated (corrections={result.validation_corrections})",
    )

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
    stock_row = query_one(
        "SELECT currency FROM analysis_stocks WHERE id = ?",
        (stock_id,),
    )
    stock_currency = stock_row["currency"] if stock_row else "USD"

    updated_statements, total_items = _persist_extraction_result(
        stock_id=stock_id,
        result=result,
        stock_currency=stock_currency,
        source_file=file.filename,
        extracted_by="gemini-ai-verified",
        notes_template=f"AI-verified (placement={result.placement_corrections})",
    )

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
    stock_row = query_one(
        "SELECT currency FROM analysis_stocks WHERE id = ?",
        (stock_id,),
    )
    stock_currency = stock_row["currency"] if stock_row else "USD"

    updated_statements, total_items = _persist_extraction_result(
        stock_id=stock_id,
        result=result,
        stock_currency=stock_currency,
        source_file=None,
        extracted_by="gemini-ai-attributed",
        notes_template=f"AI-attributed (corrections={result.validation_corrections})",
    )

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


class AiRearrangeRequest(BaseModel):
    statement_type: str = Field(..., description="income, balance, cashflow, or equity")
    periods: Optional[List[str]] = Field(None, description="Specific period_end_dates to check")
    pdf_id: Optional[int] = Field(None, description="Saved PDF id to include images for AI context")


@router.post("/stocks/{stock_id}/ai-rearrange")
async def ai_rearrange(
    stock_id: int,
    body: AiRearrangeRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """
    AI Reconcile — act as auditor to get names, sequence & values right
    by comparing stored data against the saved PDF.
    """
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)
    logger.info("AI-RECONCILE: stock=%d type=%s pdf_id=%s body_pdf=%s", stock_id, body.statement_type, body.pdf_id, body)

    # Auto-select latest saved PDF if none specified
    pdf_id = body.pdf_id
    if pdf_id is None:
        latest = query_one(
            "SELECT id FROM pdf_uploads WHERE stock_id = ? ORDER BY created_at DESC LIMIT 1",
            (stock_id,),
        )
        logger.info("AI-RECONCILE: auto-detect latest=%s", latest)
        if latest:
            pdf_id = latest["id"] if isinstance(latest, dict) else latest[0]
    if pdf_id is None:
        raise BadRequestError("No saved PDF found. Upload a PDF first so the AI can audit against it.")

    logger.info("AI-RECONCILE: using pdf_id=%s", pdf_id)

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
        logger.error("AI-RECONCILE: No Gemini API key found")
        raise BadRequestError("Gemini API key required for AI reconciliation.")

    logger.info("AI-RECONCILE: calling ai_rearrange_statement with key=%s...", api_key[:8] if api_key else "NONE")

    from app.services.extraction_service import ai_rearrange_statement

    try:
        result = await ai_rearrange_statement(
            stock_id=stock_id,
            statement_type=body.statement_type,
            api_key=api_key,
            periods=body.periods,
            pdf_id=pdf_id,
        )
    except ValueError as exc:
        logger.error("AI-RECONCILE ValueError: %s", exc)
        raise BadRequestError(str(exc))
    except Exception as exc:
        logger.error("AI-RECONCILE error: %s", exc, exc_info=True)
        raise BadRequestError(f"AI reconciliation failed: {exc}")

    return {"status": "ok", "data": result}


# ════════════════════════════════════════════════════════════════════
# CASH FLOW STAGING
# ════════════════════════════════════════════════════════════════════

@router.get("/stocks/{stock_id}/cashflow-runs")
async def list_cashflow_runs(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """List all cash flow extraction runs for a stock."""
    _ensure_schema()
    runs = query_all(
        """SELECT id, stock_id, pdf_upload_id, source_file, status,
                  periods_json, reconciliation_summary, created_at, updated_at
           FROM cashflow_extraction_runs
           WHERE stock_id = ?
           ORDER BY created_at DESC""",
        (stock_id,),
    )
    data = []
    for r in runs:
        row = dict(r)
        row["periods"] = json.loads(row.pop("periods_json", "[]") or "[]")
        summary = row.pop("reconciliation_summary", None)
        row["reconciliation_summary"] = json.loads(summary) if summary else None
        data.append(row)
    return {"status": "ok", "data": data}


@router.get("/stocks/{stock_id}/cashflow-runs/{run_id}/rows")
async def get_cashflow_staged_rows(
    stock_id: int,
    run_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Get staged rows for a specific cash flow extraction run."""
    _ensure_schema()
    # Verify ownership
    run = query_one(
        "SELECT id FROM cashflow_extraction_runs WHERE id = ? AND stock_id = ?",
        (run_id, stock_id),
    )
    if not run:
        raise NotFoundError(f"Cash flow run {run_id} not found for stock {stock_id}")

    rows = query_all(
        """SELECT id, run_id, row_order, label_raw, normalized_code, section,
                  row_kind, values_json, is_total, confidence,
                  is_accepted, rejection_reason, created_at
           FROM cashflow_staged_rows
           WHERE run_id = ?
           ORDER BY row_order""",
        (run_id,),
    )
    data = []
    for r in rows:
        row = dict(r)
        row["values"] = json.loads(row.pop("values_json", "{}") or "{}")
        data.append(row)
    return {"status": "ok", "data": data}


class CashflowRowPatch(BaseModel):
    is_accepted: Optional[bool] = None
    rejection_reason: Optional[str] = None


@router.patch("/stocks/{stock_id}/cashflow-staged-rows/{row_id}")
async def patch_cashflow_staged_row(
    stock_id: int,
    row_id: int,
    body: CashflowRowPatch,
    current_user: TokenData = Depends(get_current_user),
):
    """Manually accept or reject a staged cash flow row."""
    _ensure_schema()
    # Verify ownership through join
    row = query_one(
        """SELECT sr.id, sr.run_id
           FROM cashflow_staged_rows sr
           JOIN cashflow_extraction_runs cr ON sr.run_id = cr.id
           WHERE sr.id = ? AND cr.stock_id = ?""",
        (row_id, stock_id),
    )
    if not row:
        raise NotFoundError(
            f"Staged row {row_id} not found for stock {stock_id}"
        )

    updates = []
    params = []
    if body.is_accepted is not None:
        updates.append("is_accepted = ?")
        params.append(body.is_accepted)
    if body.rejection_reason is not None:
        updates.append("rejection_reason = ?")
        params.append(body.rejection_reason)

    if not updates:
        return {"status": "ok", "message": "No changes"}

    params.append(row_id)
    exec_sql(
        f"UPDATE cashflow_staged_rows SET {', '.join(updates)} WHERE id = ?",
        tuple(params),
    )
    return {"status": "ok", "message": "Row updated"}


@router.post("/stocks/{stock_id}/cashflow-runs/{run_id}/commit")
async def commit_cashflow_run(
    stock_id: int,
    run_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Commit accepted rows from a staged cash flow run to financial_line_items.

    This replaces any existing cashflow line items for the corresponding periods.
    """
    _ensure_schema()
    run = query_one(
        """SELECT id, stock_id, source_file, periods_json, status
           FROM cashflow_extraction_runs
           WHERE id = ? AND stock_id = ?""",
        (run_id, stock_id),
    )
    if not run:
        raise NotFoundError(f"Cash flow run {run_id} not found for stock {stock_id}")

    if run["status"] == "committed":
        raise BadRequestError("Run already committed")

    periods = json.loads(run["periods_json"] or "[]")
    source_file = run["source_file"]

    # Get accepted rows
    accepted = query_all(
        """SELECT row_order, label_raw, normalized_code, values_json,
                  is_total, confidence
           FROM cashflow_staged_rows
           WHERE run_id = ? AND is_accepted = 1
           ORDER BY row_order""",
        (run_id,),
    )
    if not accepted:
        raise BadRequestError("No accepted rows to commit")

    # Find stock currency
    stock = query_one("SELECT currency FROM stocks WHERE id = ?", (stock_id,))
    stock_currency = stock["currency"] if stock else "SAR"

    now = int(time.time())

    # Build existing codes map
    existing_codes_map: Dict[str, str] = {}
    try:
        code_rows = query_all(
            """SELECT DISTINCT li.line_item_code, fs.statement_type
               FROM financial_line_items li
               JOIN financial_statements fs ON li.statement_id = fs.id
               WHERE fs.stock_id = ?""",
            (stock_id,),
        )
        for r in code_rows:
            existing_codes_map[r["line_item_code"]] = r["statement_type"]
    except Exception:
        pass

    created_stmts = []
    total_items = 0

    with get_connection() as conn:
        cur = conn.cursor()

        for period_label in periods:
            period_end_date = period_label
            if len(period_end_date) == 4:
                period_end_date = f"{period_end_date}-12-31"

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

            existing_stmt = query_one(
                """SELECT id FROM financial_statements
                   WHERE stock_id = ? AND statement_type = 'cashflow'
                   AND period_end_date = ?""",
                (stock_id, period_end_date),
            )

            if existing_stmt:
                stmt_id = existing_stmt["id"]
                cur.execute(
                    """UPDATE financial_statements
                       SET fiscal_year=?, source_file=?, extracted_by='staged_commit',
                           notes=?, created_at=?
                       WHERE id=?""",
                    (fiscal_year, source_file,
                     f"Committed from staged run #{run_id}", now, stmt_id),
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
                        stock_id, "cashflow", fiscal_year, None,
                        period_end_date, None, source_file, "staged_commit",
                        None, f"Committed from staged run #{run_id}", now,
                    ),
                )
                stmt_id = cur.lastrowid

            seen_codes: set = set()
            for row in accepted:
                # G) Headers must never be committed as financial line items
                if row["row_kind"] == "header":
                    continue

                vals = json.loads(row["values_json"] or "{}")
                code = _normalize_key(row["normalized_code"] or row["label_raw"])
                code = _reconcile_code_with_existing(
                    code, row["label_raw"], existing_codes_map,
                )
                if code in seen_codes:
                    continue

                amount = vals.get(period_label, 0.0)
                if amount is None:
                    amount = 0.0

                cur.execute(
                    """INSERT INTO financial_line_items
                       (statement_id, line_item_code, line_item_name,
                        amount, currency, order_index, is_total)
                       VALUES (?,?,?,?,?,?,?)""",
                    (stmt_id, code, row["label_raw"], float(amount),
                     stock_currency, row["row_order"], row["is_total"]),
                )
                seen_codes.add(code)
                total_items += 1

            created_stmts.append({
                "statement_id": stmt_id,
                "period_end_date": period_end_date,
                "fiscal_year": fiscal_year,
                "line_items_count": len(accepted),
            })

        # Update run status
        cur.execute(
            """UPDATE cashflow_extraction_runs
               SET status='committed', updated_at=?
               WHERE id=?""",
            (now, run_id),
        )
        conn.commit()

    return {
        "status": "ok",
        "message": f"Committed {total_items} items across {len(created_stmts)} period(s)",
        "data": {
            "statements": created_stmts,
            "total_items": total_items,
        },
    }


# ════════════════════════════════════════════════════════════════════
# L) CASH FLOW DEBUG / REVIEW SURFACE
# ════════════════════════════════════════════════════════════════════

@router.get("/stocks/{stock_id}/cashflow-status")
async def get_cashflow_status(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """L) Lightweight debug/review surface for cash flow extraction status.

    Returns:
    - latest extraction run status
    - reconciliation status and summary
    - accepted totals (CFO/CFI/CFF/beginning/ending)
    - needs_review flag
    - rejection reasons and errors if failed
    """
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    # Get latest run
    latest = query_one(
        """SELECT id, status, periods_json, reconciliation_summary,
                  source_file, created_at, updated_at
           FROM cashflow_extraction_runs
           WHERE stock_id = ?
           ORDER BY created_at DESC LIMIT 1""",
        (stock_id,),
    )

    if not latest:
        return {
            "status": "ok",
            "data": {
                "has_extraction": False,
                "message": "No cash flow extraction runs found for this stock.",
            },
        }

    run = dict(latest)
    run_id = run["id"]
    run["periods"] = json.loads(run.pop("periods_json", "[]") or "[]")
    recon_raw = run.pop("reconciliation_summary", None)
    recon_summary = json.loads(recon_raw) if recon_raw else {}

    # Get row counts
    row_counts = query_one(
        """SELECT
             COUNT(*) as total,
             SUM(CASE WHEN is_accepted = 1 THEN 1 ELSE 0 END) as accepted,
             SUM(CASE WHEN is_accepted = 0 THEN 1 ELSE 0 END) as rejected
           FROM cashflow_staged_rows WHERE run_id = ?""",
        (run_id,),
    )

    # Get rejected rows with reasons
    rejected_rows = query_all(
        """SELECT label_raw, section, row_kind, rejection_reason
           FROM cashflow_staged_rows
           WHERE run_id = ? AND is_accepted = 0
           ORDER BY row_order""",
        (run_id,),
    )

    # Get accepted subtotals for quick view
    accepted_subtotals = query_all(
        """SELECT label_raw, normalized_code, section, row_kind, values_json
           FROM cashflow_staged_rows
           WHERE run_id = ? AND is_accepted = 1
             AND (row_kind IN ('subtotal', 'total') OR section = 'cash_bridge')
           ORDER BY row_order""",
        (run_id,),
    )
    subtotal_data = []
    for r in accepted_subtotals:
        vals = json.loads(r["values_json"] or "{}")
        subtotal_data.append({
            "label": r["label_raw"],
            "normalized_code": r["normalized_code"],
            "section": r["section"],
            "row_kind": r["row_kind"],
            "values": vals,
        })

    return {
        "status": "ok",
        "data": {
            "has_extraction": True,
            "run_id": run_id,
            "extraction_status": run["status"],
            "needs_review": run["status"] == "needs_review",
            "source_file": run.get("source_file"),
            "periods": run["periods"],
            "created_at": run.get("created_at"),
            "row_counts": dict(row_counts) if row_counts else {},
            "reconciliation": {
                "status": recon_summary.get("status"),
                "warnings": recon_summary.get("warnings", []),
                "errors": recon_summary.get("errors", []),
                "summary": recon_summary.get("summary", {}),
            },
            "validated_metrics": recon_summary.get("validated_metrics", {}),
            "accepted_subtotals": subtotal_data,
            "rejected_rows": [dict(r) for r in rejected_rows],
        },
    }


# K) Validated FCF endpoint for DCF integration
@router.get("/stocks/{stock_id}/validated-fcf")
async def get_validated_fcf(
    stock_id: int,
    period_end_date: Optional[str] = Query(None),
    current_user: TokenData = Depends(get_current_user),
):
    """K) Get validated FCF from the cash flow reconciliation pipeline.

    Returns validated CFO, capex, and FCF — computed deterministically
    from committed/reconciled cash flow data, not raw Gemini output.

    If period_end_date is not provided, returns the latest available.
    """
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    # Get the latest committed run with metrics
    latest = query_one(
        """SELECT id, periods_json, reconciliation_summary, status
           FROM cashflow_extraction_runs
           WHERE stock_id = ? AND status IN ('committed', 'reconciled')
           ORDER BY created_at DESC LIMIT 1""",
        (stock_id,),
    )

    result: Dict[str, Any] = {"source": "validated_cashflow_pipeline"}

    if latest:
        recon_raw = latest["reconciliation_summary"]
        recon_summary = json.loads(recon_raw) if recon_raw else {}
        metrics_by_period = recon_summary.get("validated_metrics", {})

        if period_end_date and period_end_date in metrics_by_period:
            result["period"] = period_end_date
            result["metrics"] = metrics_by_period[period_end_date]
        elif metrics_by_period:
            # Return all periods
            result["metrics_by_period"] = metrics_by_period
        else:
            result["metrics"] = None
            result["message"] = "No validated metrics in latest run"
    else:
        result["metrics"] = None
        result["message"] = "No committed cash flow runs found"

    # Fallback: try from financial_line_items directly
    if not result.get("metrics") and not result.get("metrics_by_period"):
        items = _load_items_for_period(stock_id, period_end_date or "")
        cfo = items.get("CASH_FROM_OPERATIONS")
        capex = items.get("CAPITAL_EXPENDITURES") or items.get("CAPEX")
        if cfo is not None:
            fcf = (cfo - abs(capex)) if capex is not None else None
            result["metrics"] = {
                "cash_from_operations": cfo,
                "capex_ppe_cash": capex,
                "free_cash_flow": fcf,
            }
            result["source"] = "financial_line_items_fallback"

    return {"status": "ok", "data": result}


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

    # Auto-compute growth-derived metrics (CAGR, stability, trends) if missing
    has_growth = query_val(
        "SELECT COUNT(*) FROM stock_metrics WHERE stock_id = ? AND metric_type = 'growth' AND metric_name LIKE '%CAGR%'",
        (stock_id,),
    )
    if not has_growth:
        try:
            _calculate_growth(stock_id)
        except Exception:
            pass

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

    # Also compute growth-derived metrics (CAGR, stability, margin trends)
    # so they appear in the metrics table without requiring a separate Growth tab visit.
    try:
        _calculate_growth(stock_id)
    except Exception:
        pass  # non-fatal — per-period metrics already saved

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


def _fetch_beta_from_stockanalysis(symbol: str) -> Optional[float]:
    """Scrape beta from stockanalysis.com overview page for a KW stock.

    URL: https://stockanalysis.com/quote/kwse/{SYMBOL}/
    The page embeds beta in the statistics table.
    Returns None on any failure (network, parsing, not found).
    """
    import re as _re
    try:
        import httpx as _httpx
        url = f"https://stockanalysis.com/quote/kwse/{symbol}/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = _httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            logger.warning("stockanalysis.com beta: %s returned %s", symbol, resp.status_code)
            return None
        # Beta appears in the page data — look for "Beta" followed by a number
        m = _re.search(r'Beta[^0-9\-]*?([\-]?[0-9]+\.?[0-9]*)', resp.text)
        if m:
            val = float(m.group(1))
            logger.info("stockanalysis.com beta for %s = %.4f", symbol, val)
            return round(val, 4)
        logger.warning("stockanalysis.com: beta not found in page for %s", symbol)
        return None
    except Exception as exc:
        logger.warning("stockanalysis.com beta fetch error for %s: %s", symbol, exc)
        return None


def _compute_tax_rate_from_statements(stock_id: int, user_id: int) -> Optional[float]:
    """Compute effective tax rate from the user's uploaded income statements.

    Looks for income_tax and income_before_tax (or pretax_income) line items
    in the most recent annual period.  Returns decimal (e.g. 0.15 for 15%).
    """
    try:
        rows = query_all(
            """
            SELECT li.code, li.value, fs.period_end
            FROM financial_line_items li
            JOIN financial_statements fs ON li.statement_id = fs.id
            WHERE fs.stock_id = ? AND fs.user_id = ? AND fs.statement_type = 'income'
              AND li.code IN ('income_tax', 'income_before_tax', 'pretax_income')
            ORDER BY fs.period_end DESC
            """,
            (stock_id, user_id),
        )
        if not rows:
            return None
        # Group by period — take values from the latest period
        latest_period = rows[0]["period_end"] if isinstance(rows[0], dict) else rows[0][2]
        tax_val = None
        pretax_val = None
        for r in rows:
            p = r["period_end"] if isinstance(r, dict) else r[2]
            if p != latest_period:
                break
            code = r["code"] if isinstance(r, dict) else r[0]
            val = r["value"] if isinstance(r, dict) else r[1]
            if code == "income_tax" and val is not None:
                tax_val = float(val)
            if code in ("income_before_tax", "pretax_income") and val is not None:
                pretax_val = float(val)
        if tax_val is not None and pretax_val and pretax_val > 0:
            return round(abs(tax_val) / pretax_val, 4)
        return None
    except Exception as exc:
        logger.warning("tax rate from statements failed for stock %s: %s", stock_id, exc)
        return None


@router.get("/stocks/{stock_id}/valuation-defaults")
async def get_valuation_defaults(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Return auto-computed default inputs for all 4 valuation models."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    # Gather latest metrics
    rows = query_all(
        "SELECT metric_name, metric_value FROM stock_metrics WHERE stock_id = ? ORDER BY period_end_date DESC",
        (stock_id,),
    )
    latest: Dict[str, float] = {}
    for r in rows:
        name = r[0] if isinstance(r, (tuple, list)) else r["metric_name"]
        val = r[1] if isinstance(r, (tuple, list)) else r["metric_value"]
        if name not in latest:
            latest[name] = val

    # Shares outstanding, exchange, and summary MoS from analysis_stocks table
    stock_row = query_one("SELECT outstanding_shares, summary_margin_of_safety, exchange FROM analysis_stocks WHERE id = ?", (stock_id,))
    shares = (stock_row[0] if isinstance(stock_row, (tuple, list)) else stock_row.get("outstanding_shares", None)) if stock_row else None
    summary_mos = (stock_row[1] if isinstance(stock_row, (tuple, list)) else stock_row.get("summary_margin_of_safety", 15.0)) if stock_row else 15.0
    stock_exchange = (stock_row[2] if isinstance(stock_row, (tuple, list)) else stock_row.get("exchange", "US")) if stock_row else "US"

    # FCF from cash flow
    fcf_row = query_one(
        """SELECT li.amount FROM financial_line_items li
           JOIN financial_statements fs ON fs.id = li.statement_id
           WHERE fs.stock_id = ? AND fs.statement_type = 'cashflow'
             AND fs.fiscal_quarter IS NULL
             AND UPPER(li.line_item_code) IN ('FREE_CASH_FLOW', 'UNLEVERED_FREE_CASH_FLOW', 'LEVERED_FREE_CASH_FLOW')
           ORDER BY fs.fiscal_year DESC LIMIT 1""",
        (stock_id,),
    )
    fcf_val = (fcf_row[0] if isinstance(fcf_row, (tuple, list)) else fcf_row.get("amount")) if fcf_row else None

    # If no explicit FCF, compute from CFO - CapEx
    if fcf_val is None:
        cfo_row = query_one(
            """SELECT li.amount FROM financial_line_items li
               JOIN financial_statements fs ON fs.id = li.statement_id
               WHERE fs.stock_id = ? AND fs.statement_type = 'cashflow'
                 AND fs.fiscal_quarter IS NULL
                 AND UPPER(li.line_item_code) = 'CASH_FROM_OPERATIONS'
               ORDER BY fs.fiscal_year DESC LIMIT 1""",
            (stock_id,),
        )
        capex_row = query_one(
            """SELECT li.amount FROM financial_line_items li
               JOIN financial_statements fs ON fs.id = li.statement_id
               WHERE fs.stock_id = ? AND fs.statement_type = 'cashflow'
                 AND fs.fiscal_quarter IS NULL
                 AND UPPER(li.line_item_code) IN ('CAPITAL_EXPENDITURES', 'CAPEX')
               ORDER BY fs.fiscal_year DESC LIMIT 1""",
            (stock_id,),
        )
        cfo = (cfo_row[0] if isinstance(cfo_row, (tuple, list)) else cfo_row.get("amount")) if cfo_row else None
        capex = (capex_row[0] if isinstance(capex_row, (tuple, list)) else capex_row.get("amount")) if capex_row else 0
        if cfo is not None:
            fcf_val = cfo - abs(capex or 0)

    # ── FCF History (multi-year) for DCF section ──────────────────
    fcf_history = []
    avg_fcf_growth = None
    # Method 1: explicit FREE_CASH_FLOW per year
    fcf_hist_rows = query_all(
        """SELECT fs.fiscal_year, li.amount FROM financial_line_items li
           JOIN financial_statements fs ON fs.id = li.statement_id
           WHERE fs.stock_id = ? AND fs.statement_type = 'cashflow'
             AND fs.fiscal_quarter IS NULL
             AND UPPER(li.line_item_code) IN ('FREE_CASH_FLOW', 'UNLEVERED_FREE_CASH_FLOW', 'LEVERED_FREE_CASH_FLOW')
             AND li.amount IS NOT NULL
           ORDER BY fs.fiscal_year""",
        (stock_id,),
    )
    if fcf_hist_rows:
        for fr in fcf_hist_rows:
            fy = fr[0] if isinstance(fr, (tuple, list)) else fr["fiscal_year"]
            amt = fr[1] if isinstance(fr, (tuple, list)) else fr["amount"]
            fcf_history.append({"year": fy, "fcf": round(float(amt), 2)})
    else:
        # Method 2: compute CFO - |CapEx| per year
        cfo_rows = query_all(
            """SELECT fs.fiscal_year, li.amount FROM financial_line_items li
               JOIN financial_statements fs ON fs.id = li.statement_id
               WHERE fs.stock_id = ? AND fs.statement_type = 'cashflow'
                 AND fs.fiscal_quarter IS NULL
                 AND UPPER(li.line_item_code) IN ('CASH_FROM_OPERATIONS', 'OPERATING_CASH_FLOW')
                 AND li.amount IS NOT NULL
               ORDER BY fs.fiscal_year""",
            (stock_id,),
        )
        capex_rows = query_all(
            """SELECT fs.fiscal_year, li.amount FROM financial_line_items li
               JOIN financial_statements fs ON fs.id = li.statement_id
               WHERE fs.stock_id = ? AND fs.statement_type = 'cashflow'
                 AND fs.fiscal_quarter IS NULL
                 AND UPPER(li.line_item_code) IN ('CAPITAL_EXPENDITURES', 'CAPEX')
                 AND li.amount IS NOT NULL
               ORDER BY fs.fiscal_year""",
            (stock_id,),
        )
        capex_by_year = {}
        for cr in (capex_rows or []):
            fy = cr[0] if isinstance(cr, (tuple, list)) else cr["fiscal_year"]
            amt = cr[1] if isinstance(cr, (tuple, list)) else cr["amount"]
            capex_by_year[fy] = float(amt)
        for cr in (cfo_rows or []):
            fy = cr[0] if isinstance(cr, (tuple, list)) else cr["fiscal_year"]
            cfo_amt = float(cr[1] if isinstance(cr, (tuple, list)) else cr["amount"])
            capex_amt = capex_by_year.get(fy, 0)
            fcf_history.append({"year": fy, "fcf": round(cfo_amt - abs(capex_amt), 2)})

    # Compute average YoY FCF growth rate
    if len(fcf_history) >= 2:
        fcf_growth_rates = []
        for i in range(1, len(fcf_history)):
            prev = fcf_history[i - 1]["fcf"]
            curr = fcf_history[i]["fcf"]
            if prev and prev != 0:
                fcf_growth_rates.append((curr - prev) / abs(prev))
        if fcf_growth_rates:
            avg_fcf_growth = round(sum(fcf_growth_rates) / len(fcf_growth_rates), 4)

    # Dividends per share history (for DDM growth calculation)
    div_rows = query_all(
        """SELECT fs.fiscal_year, li.amount FROM financial_line_items li
           JOIN financial_statements fs ON fs.id = li.statement_id
           WHERE fs.stock_id = ? AND fs.statement_type = 'cashflow'
             AND fs.fiscal_quarter IS NULL
             AND UPPER(li.line_item_code) = 'DIVIDENDS_PAID'
           ORDER BY fs.fiscal_year""",
        (stock_id,),
    )
    dps_history = []
    div_growth_rates = []
    if div_rows and shares and shares > 0:
        for dr_row in div_rows:
            fy = dr_row[0] if isinstance(dr_row, (tuple, list)) else dr_row["fiscal_year"]
            amt = dr_row[1] if isinstance(dr_row, (tuple, list)) else dr_row["amount"]
            if amt is not None:
                dps = abs(amt) / shares
                dps_history.append({"year": fy, "dps": round(dps, 4)})
        # Compute growth rates between consecutive years
        for i in range(1, len(dps_history)):
            prev_dps = dps_history[i - 1]["dps"]
            curr_dps = dps_history[i]["dps"]
            if prev_dps > 0:
                gr = (curr_dps - prev_dps) / prev_dps
                div_growth_rates.append(gr)

    avg_div_growth = sum(div_growth_rates) / len(div_growth_rates) if div_growth_rates else None

    # Revenue growth (for DCF stage1 default)
    rev_growth = latest.get("Revenue Growth")

    # Debt and cash (for DCF enterprise-to-equity bridge)
    # ── Cash: pick best single cash line item from latest balance sheet ──
    # Priority: TOTAL_DEBT (pre-computed) > CASH_EQUIVALENTS/CASH_SHORT_TERM_INVESTMENTS > lowercase 'cash'
    total_cash = None
    cash_row = query_one(
        """SELECT li.amount FROM financial_line_items li
           JOIN financial_statements fs ON fs.id = li.statement_id
           WHERE fs.stock_id = ? AND fs.statement_type = 'balance'
             AND fs.fiscal_quarter IS NULL
             AND UPPER(li.line_item_code) IN (
               'CASH_AND_EQUIVALENTS','CASH_AND_CASH_EQUIVALENTS',
               'CASH_EQUIVALENTS','CASH_SHORT_TERM_INVESTMENTS',
               'CASH','CASH_BALANCES')
             AND li.amount IS NOT NULL
           ORDER BY fs.fiscal_year DESC,
                    CASE UPPER(li.line_item_code)
                      WHEN 'CASH_SHORT_TERM_INVESTMENTS' THEN 1
                      WHEN 'CASH_EQUIVALENTS' THEN 2
                      WHEN 'CASH_AND_CASH_EQUIVALENTS' THEN 3
                      WHEN 'CASH_AND_EQUIVALENTS' THEN 4
                      WHEN 'CASH_BALANCES' THEN 5
                      WHEN 'CASH' THEN 6
                      ELSE 7 END
           LIMIT 1""",
        (stock_id,),
    )
    if cash_row:
        total_cash = cash_row[0] if isinstance(cash_row, (tuple, list)) else cash_row.get("amount")

    # ── Debt: prefer pre-computed TOTAL_DEBT, else sum individual debt items ──
    total_debt = None
    td_row = query_one(
        """SELECT li.amount FROM financial_line_items li
           JOIN financial_statements fs ON fs.id = li.statement_id
           WHERE fs.stock_id = ? AND fs.statement_type = 'balance'
             AND fs.fiscal_quarter IS NULL
             AND UPPER(li.line_item_code) = 'TOTAL_DEBT'
             AND li.amount IS NOT NULL
           ORDER BY fs.fiscal_year DESC LIMIT 1""",
        (stock_id,),
    )
    if td_row:
        total_debt = td_row[0] if isinstance(td_row, (tuple, list)) else td_row.get("amount")
    else:
        # Sum individual debt line items from latest year
        debt_sum_row = query_one(
            """SELECT SUM(li.amount) as debt_total FROM financial_line_items li
               JOIN financial_statements fs ON fs.id = li.statement_id
               WHERE fs.stock_id = ? AND fs.statement_type = 'balance'
                 AND fs.fiscal_quarter IS NULL
                 AND fs.fiscal_year = (
                   SELECT MAX(fs2.fiscal_year) FROM financial_statements fs2
                   WHERE fs2.stock_id = ? AND fs2.statement_type = 'balance' AND fs2.fiscal_quarter IS NULL)
                 AND (UPPER(li.line_item_code) IN (
                       'LONG_TERM_DEBT','SHORT_TERM_DEBT','CURRENT_PORTION_OF_LONG_TERM_DEBT',
                       'CURRENT_PORTION_LT_DEBT','LONG_TERM_DEBTS','CURRENT_PORTION_OF_LONG_TERM_DEBTS')
                      OR LOWER(li.line_item_code) LIKE '%borrowing%'
                      OR LOWER(li.line_item_code) LIKE '%bank_overdraft%'
                      OR LOWER(li.line_item_code) LIKE '%overdraft%'
                      OR LOWER(li.line_item_code) LIKE '%bank_facilit%'
                      OR LOWER(li.line_item_code) LIKE '%islamic_payable%'
                      OR LOWER(li.line_item_code) LIKE '%due_to_bank%'
                      OR LOWER(li.line_item_code) LIKE '%short_term_loan%'
                      OR LOWER(li.line_item_code) LIKE '%murabaha%')
                 AND li.amount IS NOT NULL""",
            (stock_id, stock_id),
        )
        if debt_sum_row:
            val = debt_sum_row[0] if isinstance(debt_sum_row, (tuple, list)) else debt_sum_row.get("debt_total")
            if val is not None:
                total_debt = abs(val)

    # ── Graham-specific defaults: historical EPS avg growth & yfinance data ───
    # Helper: detect subunit codes (fils/cents/halala) for division by 1000
    def _is_subunit_code(code_str: str) -> bool:
        if not isinstance(code_str, str):
            return False
        low = code_str.lower()
        return 'fils' in low or 'cents' in low or 'halala' in low

    eps_ttm = latest.get("EPS")
    # Fallback: get EPS directly from income statement line items
    if eps_ttm is None:
        eps_li_row = query_one(
            """SELECT li.line_item_code, li.amount FROM financial_line_items li
               JOIN financial_statements fs ON fs.id = li.statement_id
               WHERE fs.stock_id = ? AND fs.statement_type = 'income'
                 AND fs.fiscal_quarter IS NULL
                 AND fs.period_end_date NOT LIKE '%/_3M' ESCAPE '/'
                 AND fs.period_end_date NOT LIKE '%/_6M' ESCAPE '/'
                 AND fs.period_end_date NOT LIKE '%/_9M' ESCAPE '/'
                 AND (UPPER(li.line_item_code) IN ('EPS_DILUTED','EPS_BASIC')
                      OR LOWER(li.line_item_code) LIKE '%earnings_per_share%'
                      OR LOWER(li.line_item_code) LIKE '%eps_%')
               ORDER BY fs.fiscal_year DESC LIMIT 1""",
            (stock_id,),
        )
        if eps_li_row:
            code = eps_li_row.get("line_item_code", "")
            val = eps_li_row.get("amount")
            if val is not None and _is_subunit_code(code):
                val = val / 1000.0
            eps_ttm = val
    # Last fallback: compute from Net Income / Shares Outstanding
    if eps_ttm is None and shares and shares > 0:
        ni_row = query_one(
            """SELECT li.amount FROM financial_line_items li
               JOIN financial_statements fs ON fs.id = li.statement_id
               WHERE fs.stock_id = ? AND fs.statement_type = 'income'
                 AND fs.fiscal_quarter IS NULL
                 AND fs.period_end_date NOT LIKE '%/_3M' ESCAPE '/'
                 AND fs.period_end_date NOT LIKE '%/_6M' ESCAPE '/'
                 AND fs.period_end_date NOT LIKE '%/_9M' ESCAPE '/'
                 AND UPPER(li.line_item_code) = 'NET_INCOME'
               ORDER BY fs.fiscal_year DESC LIMIT 1""",
            (stock_id,),
        )
        if ni_row:
            ni_val = ni_row[0] if isinstance(ni_row, (tuple, list)) else ni_row.get("amount")
            if ni_val is not None:
                eps_ttm = round(ni_val / shares, 4)
    graham_growth_avg = None
    eps_history = []
    # Fetch multi-year EPS from stock_metrics
    eps_rows = query_all(
        """SELECT fiscal_year, metric_value FROM stock_metrics
           WHERE stock_id = ? AND metric_name = 'EPS'
             AND fiscal_quarter IS NULL AND metric_value IS NOT NULL
           ORDER BY fiscal_year""",
        (stock_id,),
    )
    # Fallback: pull EPS directly from income statement line items
    if not eps_rows:
        eps_rows_raw = query_all(
            """SELECT fs.fiscal_year, li.line_item_code, li.amount FROM financial_line_items li
               JOIN financial_statements fs ON fs.id = li.statement_id
               WHERE fs.stock_id = ? AND fs.statement_type = 'income'
                 AND fs.fiscal_quarter IS NULL
                 AND fs.period_end_date NOT LIKE '%/_3M' ESCAPE '/'
                 AND fs.period_end_date NOT LIKE '%/_6M' ESCAPE '/'
                 AND fs.period_end_date NOT LIKE '%/_9M' ESCAPE '/'
                 AND (UPPER(li.line_item_code) IN ('EPS_DILUTED','EPS_BASIC')
                      OR LOWER(li.line_item_code) LIKE '%earnings_per_share%'
                      OR LOWER(li.line_item_code) LIKE '%eps_%')
                 AND li.amount IS NOT NULL
               ORDER BY fs.fiscal_year""",
            (stock_id,),
        )
        # Dedupe: keep one EPS per fiscal_year, convert fils/cents if needed
        seen_years = set()
        eps_rows = []
        for er in (eps_rows_raw or []):
            fy = er[0] if isinstance(er, (tuple, list)) else er["fiscal_year"]
            if fy not in seen_years:
                seen_years.add(fy)
                code = er[1] if isinstance(er, (tuple, list)) else er.get("line_item_code", "")
                val = er[2] if isinstance(er, (tuple, list)) else er.get("amount")
                if val is not None and _is_subunit_code(code):
                    val = val / 1000.0
                eps_rows.append((fy, val))
    if eps_rows:
        for er in eps_rows:
            fy = er[0] if isinstance(er, (tuple, list)) else er.get("fiscal_year", er.get("fiscal_year"))
            val = er[1] if isinstance(er, (tuple, list)) else (er.get("metric_value") or er.get("amount"))
            eps_history.append({"year": fy, "eps": round(val, 4) if val else None})
        # Average year-over-year EPS growth rate
        yoy_growth_rates = []
        for i in range(1, len(eps_history)):
            prev = eps_history[i - 1]["eps"]
            curr = eps_history[i]["eps"]
            if prev and prev > 0 and curr is not None:
                yoy_growth_rates.append(((curr - prev) / prev) * 100)
        if yoy_growth_rates:
            avg_raw = sum(yoy_growth_rates) / len(yoy_growth_rates)
            # Graham conservatism: cap at 15%, floor at 0%
            graham_growth_avg = round(min(max(avg_raw, 0), 15), 2)

    # Fetch current price, bond yield & WACC components from yfinance
    current_price_yf = None
    bond_yield_yf = None
    wacc_data: Dict[str, Any] = {}
    stock_sym_row = query_one("SELECT symbol FROM analysis_stocks WHERE id = ?", (stock_id,))
    ticker_sym = (stock_sym_row[0] if isinstance(stock_sym_row, (tuple, list)) else stock_sym_row.get("symbol")) if stock_sym_row else None
    raw_symbol = ticker_sym  # preserve original symbol before yf resolution
    if ticker_sym:
        ticker_sym = _resolve_yf_ticker(ticker_sym, current_user.user_id)
        _kw = ticker_sym.upper().endswith(".KW")
        try:
            import yfinance as yf
            # Current stock price
            tk = yf.Ticker(ticker_sym)
            hist = tk.history(period="5d")
            if not hist.empty:
                _cprice = float(hist["Close"].iloc[-1])
                if _kw:
                    _cprice = _cprice / 1000.0
                current_price_yf = round(_cprice, 2)
            # 10Y Treasury yield (proxy for AAA bond yield / risk-free rate)
            tnx = yf.Ticker("^TNX")
            tnx_hist = tnx.history(period="5d")
            if not tnx_hist.empty:
                bond_yield_yf = round(float(tnx_hist["Close"].iloc[-1]), 2)

            # ── WACC components ──────────────────────────────────
            info = tk.info or {}
            beta = info.get("beta")

            # For Kuwait stocks, prefer beta from stockanalysis.com
            # (yfinance often has unreliable beta for KW tickers)
            if _kw and raw_symbol:
                sa_beta = _fetch_beta_from_stockanalysis(raw_symbol.replace(".KW", "").upper())
                if sa_beta is not None:
                    beta = sa_beta

            market_cap = info.get("marketCap")
            yf_total_debt = info.get("totalDebt")
            interest_expense = info.get("interestExpense")  # annual
            income_before_tax = info.get("incomeBeforeTax")
            income_tax = info.get("incomeTaxExpense")

            risk_free = (bond_yield_yf / 100.0) if bond_yield_yf else None
            equity_risk_premium = 0.055  # historical average ~5.5%

            # Cost of equity (CAPM)
            cost_of_equity = None
            if risk_free is not None and beta is not None:
                cost_of_equity = round(risk_free + float(beta) * equity_risk_premium, 4)

            # Cost of debt (interest / total debt)
            cost_of_debt = None
            if interest_expense and yf_total_debt and yf_total_debt > 0:
                cost_of_debt = round(abs(float(interest_expense)) / float(yf_total_debt), 4)

            # Tax rate
            tax_rate = None
            if income_before_tax and income_tax and income_before_tax > 0:
                tax_rate = round(abs(float(income_tax)) / float(income_before_tax), 4)
            # Fallback: compute tax rate from user's uploaded income statements
            if tax_rate is None:
                tax_rate = _compute_tax_rate_from_statements(stock_id, current_user.user_id)

            # Weights
            weight_equity = None
            weight_debt = None
            if market_cap and yf_total_debt is not None:
                total_value = float(market_cap) + float(yf_total_debt or 0)
                if total_value > 0:
                    weight_equity = round(float(market_cap) / total_value, 4)
                    weight_debt = round(float(yf_total_debt or 0) / total_value, 4)

            # WACC = (E/V × Re) + (D/V × Rd × (1-T))
            wacc_value = None
            if cost_of_equity is not None and weight_equity is not None:
                eq_part = weight_equity * cost_of_equity
                debt_part = 0.0
                if cost_of_debt is not None and weight_debt is not None and tax_rate is not None:
                    debt_part = weight_debt * cost_of_debt * (1 - tax_rate)
                wacc_value = round(eq_part + debt_part, 4)

            wacc_data = {
                "wacc": wacc_value,
                "wacc_risk_free_rate": risk_free,
                "wacc_beta": round(float(beta), 4) if beta is not None else None,
                "wacc_equity_risk_premium": equity_risk_premium,
                "wacc_cost_of_equity": cost_of_equity,
                "wacc_cost_of_debt": cost_of_debt,
                "wacc_tax_rate": tax_rate,
                "wacc_weight_equity": weight_equity,
                "wacc_weight_debt": weight_debt,
            }
        except Exception as e:
            logger.warning("yfinance fetch failed for Graham defaults: %s", e)

    defaults = {
        "eps": eps_ttm,
        "book_value_per_share": latest.get("Book Value / Share"),
        "dividends_per_share": latest.get("Dividends / Share"),
        "fcf": fcf_val,
        "fcf_history": fcf_history,
        "avg_fcf_growth": avg_fcf_growth,
        "shares_outstanding": shares,
        "revenue_growth": rev_growth,
        "avg_dividend_growth": round(avg_div_growth, 4) if avg_div_growth is not None else None,
        "dps_history": dps_history,
        "total_debt": total_debt,
        "total_cash": total_cash,
        "net_margin": latest.get("Net Margin"),
        "roe": latest.get("ROE"),
        # Graham-specific
        "graham_growth_cagr": graham_growth_avg,
        "eps_history": eps_history,
        "current_price": current_price_yf,
        "bond_yield": bond_yield_yf,
        "summary_margin_of_safety": summary_mos if summary_mos is not None else 15.0,
        "exchange": stock_exchange or "US",
        **wacc_data,
    }

    return {"status": "ok", "data": defaults}


@router.get("/stocks/{stock_id}/peer-multiples")
async def get_peer_multiples(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Return stored peer companies with their multiples for a stock."""
    _ensure_schema()
    uid = current_user.user_id
    _verify_stock_owner(stock_id, uid)

    rows = query_df(
        "SELECT id, peer_symbol, peer_name, sector, pe, pb, ps, pcf, ev_ebitda, eps, price "
        "FROM peer_companies WHERE stock_id = ? ORDER BY peer_symbol",
        (stock_id,),
    )
    peers = []
    if not rows.empty:
        for _, r in rows.iterrows():
            peers.append({
                "stock_id": int(r["id"]),
                "symbol": r["peer_symbol"],
                "company_name": r["peer_name"],
                "pe": r["pe"], "pb": r["pb"], "ps": r["ps"],
                "pcf": r["pcf"], "ev_ebitda": r["ev_ebitda"],
                "eps": r["eps"], "price": r["price"],
            })
    return {"status": "ok", "data": {"peers": peers, "count": len(peers)}}


def _stockanalysis_multiples(symbol: str) -> Dict[str, Any]:
    """Fetch multiples from stockanalysis.com for a Kuwait (KWSE) stock.

    Parses the embedded SvelteKit bootstrap data from the statistics page.
    Returns dict with pe, pb, ps, pcf, ev_ebitda, eps, price, company_name.
    """
    import re as _re
    import httpx

    base = _re.sub(r'\.KW$', '', symbol, flags=_re.IGNORECASE).upper()
    url = f"https://stockanalysis.com/quote/kwse/{base}/statistics/"

    entry: Dict[str, Any] = {
        "pe": None, "pb": None, "ps": None, "pcf": None,
        "ev_ebitda": None, "eps": None, "price": None,
        "company_name": None,
    }

    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })
        if resp.status_code != 200:
            logger.warning("stockanalysis.com returned %s for %s", resp.status_code, base)
            return entry

        text = resp.text

        m = _re.search(
            r'data:\s*(\[\{type:"data".*?\}]),\s*form:\s*null', text, _re.DOTALL,
        )
        if not m:
            logger.warning("stockanalysis.com: no SvelteKit data found for %s", base)
            return entry

        raw_js = m.group(1)

        def _get(field_id: str) -> Optional[float]:
            pat = _re.compile(r'\{id:"' + _re.escape(field_id) + r'"[^}]*hover:"([^"]*)"')
            match = pat.search(raw_js)
            if match:
                v = match.group(1).replace(",", "").replace("%", "").strip()
                if v and v.lower() not in ("n/a", "\u2014"):
                    try:
                        return float(v)
                    except ValueError:
                        pass
            return None

        def _r(v: Optional[float], dp: int = 2) -> Optional[float]:
            return round(v, dp) if v is not None else None

        entry["pe"] = _r(_get("pe"))
        entry["pb"] = _r(_get("pb"))
        entry["ps"] = _r(_get("ps"))
        entry["pcf"] = _r(_get("pocf"))
        entry["ev_ebitda"] = _r(_get("evEbitda"))
        entry["eps"] = _r(_get("eps"), 4)

        price_m = _re.search(r'quote:\{[^}]*\bp:([\d.]+)', raw_js)
        if price_m:
            entry["price"] = round(float(price_m.group(1)), 4)

        name_m = _re.search(r'nameFull:"([^"]+)"', raw_js)
        if name_m:
            entry["company_name"] = name_m.group(1)

    except Exception as e:
        logger.warning("stockanalysis.com fetch failed for %s: %s", base, e)

    return entry


def _yf_multiples(sym: str) -> Dict[str, Any]:
    """Fetch multiples + eps + price for a single symbol.

    For Kuwait (.KW) stocks, uses stockanalysis.com as the primary source
    (more accurate ratios, no fils/KWD conversion issues).
    Falls back to yfinance if stockanalysis.com fails.

    Returns dict with keys: pe, pb, ps, pcf, ev_ebitda, eps, price,
    and optionally company_name (from stockanalysis.com).
    """
    import yfinance as yf
    resolved = _resolve_yf_ticker(sym)
    _kw = resolved.upper().endswith(".KW")

    # Try stockanalysis.com first for Kuwait stocks
    if _kw:
        sa = _stockanalysis_multiples(resolved)
        # Check if we got meaningful data (at least P/E or P/B)
        if sa.get("pe") is not None or sa.get("pb") is not None:
            return {
                "pe": sa["pe"], "pb": sa["pb"], "ps": sa["ps"],
                "pcf": sa["pcf"], "ev_ebitda": sa["ev_ebitda"],
                "eps": sa["eps"], "price": sa["price"],
                "company_name": sa.get("company_name"),
            }
        logger.info("stockanalysis.com had no data for %s, falling back to yfinance", sym)

    entry: Dict[str, Any] = {
        "pe": None, "pb": None, "ps": None, "pcf": None,
        "ev_ebitda": None, "eps": None, "price": None,
    }
    try:
        tk = yf.Ticker(resolved)
        info = tk.info or {}
        pe = info.get("trailingPE") or info.get("forwardPE")
        entry["pe"] = round(float(pe), 2) if pe is not None else None
        pb = info.get("priceToBook")
        if pb is not None:
            pb = float(pb)
            if _kw:
                pb = pb / 1000.0  # yfinance mixes fils price with KWD book value
            entry["pb"] = round(pb, 2)
        ps = info.get("priceToSalesTrailing12Months")
        if ps is not None:
            entry["ps"] = round(float(ps), 2)
        ocf = info.get("operatingCashflow")
        mcap = info.get("marketCap")
        if ocf and mcap and ocf > 0:
            entry["pcf"] = round(float(mcap) / float(ocf), 2)
        ev = info.get("enterpriseValue")
        ebitda = info.get("ebitda")
        if ev is not None and ebitda and ebitda > 0:
            entry["ev_ebitda"] = round(float(ev) / float(ebitda), 2)
        eps_val = info.get("trailingEps") or info.get("forwardEps")
        entry["eps"] = round(float(eps_val), 2) if eps_val is not None else None
        price_val = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        if price_val is not None:
            p = float(price_val)
            if _kw:
                p = p / 1000.0
            entry["price"] = round(p, 2)
    except Exception as e:
        logger.warning("yfinance multiples fetch failed for %s: %s", sym, e)
    return entry


@router.post("/stocks/{stock_id}/peer-multiples/fetch")
async def fetch_sector_peers(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Discover peer companies in the same sector via yfinance and persist them."""
    _ensure_schema()
    uid = current_user.user_id
    _verify_stock_owner(stock_id, uid)

    # Get the stock's symbol and stored sector
    stock_row = query_one(
        "SELECT symbol, sector FROM analysis_stocks WHERE id = ? AND user_id = ?",
        (stock_id, uid),
    )
    if not stock_row:
        raise NotFoundError("Stock not found")

    symbol = stock_row["symbol"]
    import yfinance as yf

    # Get the stock's sector from yfinance (authoritative)
    yf_sym = _resolve_yf_ticker(symbol, uid)
    tk = yf.Ticker(yf_sym)
    info = tk.info or {}
    sector = info.get("sector") or stock_row.get("sector") or ""
    industry = info.get("industry") or ""

    if not sector:
        raise BadRequestError("Cannot determine sector for this stock. Update the stock's sector or check the ticker symbol.")

    # Update the stock's sector/industry in DB if we got it from yfinance
    if info.get("sector"):
        exec_sql(
            "UPDATE analysis_stocks SET sector = ?, industry = ? WHERE id = ?",
            (sector, industry, stock_id),
        )

    # Use yfinance screener to find companies in the same sector & country
    from yfinance import EquityQuery

    peer_symbols: List[str] = []
    stock_exchange = info.get("exchange", "")
    stock_country = info.get("country", "")

    # Map country name → Yahoo region code
    _COUNTRY_TO_REGION = {
        "United States": "us", "Canada": "ca", "United Kingdom": "gb",
        "Germany": "de", "France": "fr", "Switzerland": "ch",
        "Japan": "jp", "China": "cn", "Hong Kong": "hk",
        "India": "in", "Australia": "au", "South Korea": "kr",
        "Brazil": "br", "Mexico": "mx", "Singapore": "sg",
        "Taiwan": "tw", "Netherlands": "nl", "Sweden": "se",
        "Norway": "no", "Denmark": "dk", "Finland": "fi",
        "Spain": "es", "Italy": "it", "Ireland": "ie",
        "Israel": "il", "South Africa": "za", "New Zealand": "nz",
        "Belgium": "be", "Austria": "at", "Portugal": "pt",
        "Indonesia": "id", "Malaysia": "my", "Thailand": "th",
        "Philippines": "ph", "Turkey": "tr", "Poland": "pl",
        "Saudi Arabia": "sa", "United Arab Emirates": "ae",
        "Kuwait": "kw", "Qatar": "qa", "Bahrain": "bh",
        "Oman": "om", "Argentina": "ar", "Chile": "cl",
        "Colombia": "co", "Peru": "pe", "Egypt": "eg",
    }
    region = _COUNTRY_TO_REGION.get(stock_country, "")

    # Build base filters: same sector + same country/region
    def _build_query(extra_filters: list) -> "EquityQuery":
        filters = list(extra_filters)
        if region:
            filters.append(EquityQuery("eq", ["region", region]))
        else:
            # Fallback: restrict to same exchange if country unknown
            filters.append(EquityQuery("eq", ["exchange", stock_exchange]))
        return EquityQuery("and", filters)

    # Step 1: Try industry peers first (most specific)
    try:
        if industry:
            q = _build_query([
                EquityQuery("eq", ["industry", industry]),
                EquityQuery("gt", ["intradaymarketcap", 100_000_000]),
            ])
            resp = yf.screen(q, size=15, sortField="intradaymarketcap", sortAsc=False)
            if resp and "quotes" in resp:
                peer_symbols = [qt["symbol"] for qt in resp["quotes"]
                                if qt.get("symbol") and qt["symbol"] != symbol][:10]
    except Exception as e:
        logger.warning("yfinance industry screener failed for %s: %s", symbol, e)

    # Step 2: Fill remaining slots with sector-level peers
    if len(peer_symbols) < 10:
        try:
            q = _build_query([
                EquityQuery("eq", ["sector", sector]),
                EquityQuery("gt", ["intradaymarketcap", 500_000_000]),
            ])
            resp = yf.screen(q, size=20, sortField="intradaymarketcap", sortAsc=False)
            if resp and "quotes" in resp:
                existing = set(peer_symbols) | {symbol}
                for qt in resp["quotes"]:
                    s = qt.get("symbol")
                    if s and s not in existing:
                        peer_symbols.append(s)
                        if len(peer_symbols) >= 10:
                            break
        except Exception as e:
            logger.warning("yfinance sector screener fallback failed for %s: %s", symbol, e)

    if not peer_symbols:
        return {"status": "ok", "data": {"peers": [], "count": 0}}

    # Fetch multiples for each peer and persist
    now = int(time.time())
    saved_peers = []
    for psym in peer_symbols:
        mults = _yf_multiples(psym)
        # Use company_name from stockanalysis.com if available
        peer_name = mults.pop("company_name", None) or psym
        if peer_name == psym:
            try:
                tk_peer = yf.Ticker(_resolve_yf_ticker(psym))
                peer_info = tk_peer.info or {}
                peer_name = peer_info.get("shortName") or peer_info.get("longName") or psym
            except Exception:
                pass

        exec_sql(
            "INSERT INTO peer_companies (stock_id, peer_symbol, peer_name, sector, pe, pb, ps, pcf, ev_ebitda, eps, price, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(stock_id, peer_symbol) DO UPDATE SET "
            "peer_name=excluded.peer_name, sector=excluded.sector, pe=excluded.pe, pb=excluded.pb, "
            "ps=excluded.ps, pcf=excluded.pcf, ev_ebitda=excluded.ev_ebitda, "
            "eps=excluded.eps, price=excluded.price, fetched_at=excluded.fetched_at",
            (stock_id, psym, peer_name, sector,
             mults["pe"], mults["pb"], mults["ps"], mults["pcf"], mults["ev_ebitda"],
             mults["eps"], mults["price"], now),
        )

    # Return stored peers
    rows = query_df(
        "SELECT id, peer_symbol, peer_name, sector, pe, pb, ps, pcf, ev_ebitda, eps, price "
        "FROM peer_companies WHERE stock_id = ? ORDER BY peer_symbol",
        (stock_id,),
    )
    for _, r in rows.iterrows():
        saved_peers.append({
            "stock_id": int(r["id"]),
            "symbol": r["peer_symbol"],
            "company_name": r["peer_name"],
            "pe": r["pe"], "pb": r["pb"], "ps": r["ps"],
            "pcf": r["pcf"], "ev_ebitda": r["ev_ebitda"],
            "eps": r["eps"], "price": r["price"],
        })
    return {"status": "ok", "data": {"peers": saved_peers, "count": len(saved_peers)}}


@router.delete("/stocks/{stock_id}/peer-multiples/{peer_id}")
async def delete_peer_company(
    stock_id: int,
    peer_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Remove a specific peer company from a stock's peer list."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)
    exec_sql("DELETE FROM peer_companies WHERE id = ? AND stock_id = ?", (peer_id, stock_id))
    return {"status": "ok", "data": {"message": "Peer removed"}}


class AddPeerBody(BaseModel):
    symbol: str


@router.post("/stocks/{stock_id}/peer-multiples/add")
async def add_peer_company(
    stock_id: int,
    body: AddPeerBody,
    current_user: TokenData = Depends(get_current_user),
):
    """Add a single peer company by symbol — fetch its multiples."""
    _ensure_schema()
    uid = current_user.user_id
    _verify_stock_owner(stock_id, uid)

    import yfinance as yf

    sym = body.symbol.strip().upper()
    if not sym:
        raise BadRequestError("Symbol is required")

    mults = _yf_multiples(sym)
    # Use company_name from stockanalysis.com if available
    peer_name = mults.pop("company_name", None) or sym
    sector = ""

    # Fallback to yfinance for name and sector
    if peer_name == sym or not sector:
        try:
            resolved = _resolve_yf_ticker(sym)
            tk = yf.Ticker(resolved)
            info = tk.info or {}
            if peer_name == sym:
                peer_name = info.get("shortName") or info.get("longName") or sym
            sector = info.get("sector") or sector
        except Exception:
            pass

    now = int(time.time())
    exec_sql(
        "INSERT INTO peer_companies (stock_id, peer_symbol, peer_name, sector, pe, pb, ps, pcf, ev_ebitda, eps, price, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(stock_id, peer_symbol) DO UPDATE SET "
        "peer_name=excluded.peer_name, sector=excluded.sector, pe=excluded.pe, pb=excluded.pb, "
        "ps=excluded.ps, pcf=excluded.pcf, ev_ebitda=excluded.ev_ebitda, "
        "eps=excluded.eps, price=excluded.price, fetched_at=excluded.fetched_at",
        (stock_id, sym, peer_name, sector,
         mults["pe"], mults["pb"], mults["ps"], mults["pcf"], mults["ev_ebitda"],
         mults["eps"], mults["price"], now),
    )

    # Return updated peer list
    rows = query_df(
        "SELECT id, peer_symbol, peer_name, sector, pe, pb, ps, pcf, ev_ebitda, eps, price "
        "FROM peer_companies WHERE stock_id = ? ORDER BY peer_symbol",
        (stock_id,),
    )
    peers = []
    if not rows.empty:
        for _, r in rows.iterrows():
            peers.append({
                "stock_id": int(r["id"]),
                "symbol": r["peer_symbol"],
                "company_name": r["peer_name"],
                "pe": r["pe"], "pb": r["pb"], "ps": r["ps"],
                "pcf": r["pcf"], "ev_ebitda": r["ev_ebitda"],
                "eps": r["eps"], "price": r["price"],
            })
    return {"status": "ok", "data": {"peers": peers, "count": len(peers)}}


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


@router.delete("/stocks/{stock_id}/valuations")
async def delete_all_valuations(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Delete all saved valuations for a stock."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)
    exec_sql("DELETE FROM valuation_models WHERE stock_id = ?", (stock_id,))
    return {"status": "ok", "message": "All valuations deleted."}


@router.delete("/stocks/{stock_id}/valuations/{valuation_id}")
async def delete_single_valuation(
    stock_id: int,
    valuation_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Delete a single saved valuation."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)
    row = query_one("SELECT id FROM valuation_models WHERE id = ? AND stock_id = ?", (valuation_id, stock_id))
    if not row:
        raise NotFoundError("Valuation not found.")
    exec_sql("DELETE FROM valuation_models WHERE id = ? AND stock_id = ?", (valuation_id, stock_id))
    return {"status": "ok", "message": "Valuation deleted."}


@router.post("/stocks/{stock_id}/valuations/graham")
async def run_graham(
    stock_id: int,
    body: GrahamRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Run Graham Number valuation."""
    _ensure_schema()
    _verify_stock_owner(stock_id, current_user.user_id)

    result = _graham_number(body.eps, body.growth_rate,
                             body.corporate_yield, body.margin_of_safety,
                             body.current_price)
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
        body.cash, body.debt,
        wacc_components={
            "risk_free_rate": body.wacc_risk_free_rate,
            "beta": body.wacc_beta,
            "equity_risk_premium": body.wacc_equity_risk_premium,
            "cost_of_equity": body.wacc_cost_of_equity,
            "cost_of_debt": body.wacc_cost_of_debt,
            "tax_rate": body.wacc_tax_rate,
            "weight_equity": body.wacc_weight_equity,
            "weight_debt": body.wacc_weight_debt,
        } if body.wacc_used else None,
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
    """Flatten all line items across all statement types for one period.

    Keys are uppercased so callers can use ``_get("REVENUE")`` regardless
    of whether the DB stores ``revenue`` or ``REVENUE``.
    """
    rows = query_all(
        """SELECT li.line_item_code, li.amount
           FROM financial_line_items li
           JOIN financial_statements fs ON fs.id = li.statement_id
           WHERE fs.stock_id = ? AND fs.period_end_date = ?""",
        (stock_id, period_end_date),
    )
    items: Dict[str, float] = {}
    for r in rows:
        code = (r[0] if isinstance(r, (tuple, list)) else r["line_item_code"]).upper()
        amount = r[1] if isinstance(r, (tuple, list)) else r["amount"]
        # Keep first occurrence (don't overwrite with duplicates)
        if code not in items:
            items[code] = amount
    return items


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
    revenue = _get("REVENUE") or _get("TOTAL_REVENUE") or _get("NET_REVENUE") or _get("TOTAL_SALES")
    gross_profit = _get("GROSS_PROFIT")
    operating_income = _get("OPERATING_INCOME")
    net_income = _get("NET_INCOME")
    total_assets = _get("TOTAL_ASSETS")
    total_equity = _get("TOTAL_EQUITY") or _get("SHAREHOLDERS_EQUITY")

    if revenue and revenue != 0:
        if gross_profit is not None:
            prof["Gross Margin"] = gross_profit / revenue
        if operating_income is not None:
            prof["Operating Margin"] = operating_income / revenue
        if net_income is not None:
            prof["Net Margin"] = net_income / revenue
    # Fallback: use pre-computed margin line items from the statement
    if "Net Margin" not in prof:
        pm = _get("PROFIT_MARGIN") or _get("NET_MARGIN")
        if pm is not None:
            prof["Net Margin"] = pm
    if "Operating Margin" not in prof:
        om = _get("OPERATING_MARGIN")
        if om is not None:
            prof["Operating Margin"] = om
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
    short_term_inv = _get("SHORT_TERM_INVESTMENTS")
    ar = _get("ACCOUNTS_RECEIVABLE")

    # Synthesize current assets/liabilities from components when aggregates are missing
    if current_assets is None:
        ca_parts = [
            cash, short_term_inv, ar,
            _get("OTHER_RECEIVABLES"), _get("PREPAID_EXPENSES"),
            _get("OTHER_CURRENT_ASSETS"), inventory,
        ]
        known = [v for v in ca_parts if v is not None]
        if known:
            current_assets = sum(known)
    if current_liab is None:
        cl_parts = [
            _get("ACCOUNTS_PAYABLE"), _get("CURRENT_PORTION_OF_LONG_TERM_DEBT"),
            _get("CURRENT_PORTION_OF_LEASES"), _get("CURRENT_INCOME_TAXES_PAYABLE"),
            _get("CURRENT_UNEARNED_REVENUE"), _get("OTHER_CURRENT_LIABILITIES"),
        ]
        known = [v for v in cl_parts if v is not None]
        if known:
            current_liab = sum(known)
    if current_assets is not None and current_liab and current_liab != 0:
        liq["Current Ratio"] = current_assets / current_liab
        # CFA: Quick Ratio = (Cash + ST Investments + Receivables) / CL
        quick_assets = (cash or 0) + (short_term_inv or 0) + (ar or 0)
        if quick_assets > 0:
            liq["Quick Ratio"] = quick_assets / current_liab
        elif inventory is not None:
            liq["Quick Ratio"] = (current_assets - inventory) / current_liab
        # CFA: Cash Ratio = (Cash + ST Marketable Securities) / CL
        cash_plus_st = (cash or 0) + (short_term_inv or 0)
        if cash_plus_st > 0:
            liq["Cash Ratio"] = cash_plus_st / current_liab
        elif cash is not None:
            liq["Cash Ratio"] = cash / current_liab
    results["liquidity"] = liq

    # ── leverage (CFA: "debt" = interest-bearing ST + LT debt, NOT total liabilities)
    lev: Dict[str, Optional[float]] = {}
    total_liab = _get("TOTAL_LIABILITIES")
    lt_debt = _get("LONG_TERM_DEBT")
    st_debt = _get("SHORT_TERM_DEBT")
    interest_expense = _get("INTEREST_EXPENSE")
    has_debt_data = lt_debt is not None or st_debt is not None
    total_debt = (lt_debt or 0) + (st_debt or 0)
    # CFA: Debt-to-Equity = Total Debt / Total Equity
    if has_debt_data and total_equity and total_equity != 0:
        lev["Debt-to-Equity"] = total_debt / total_equity
    # CFA: Debt-to-Assets = Total Debt / Total Assets
    if has_debt_data and total_assets and total_assets != 0:
        lev["Debt-to-Assets"] = total_debt / total_assets
    # CFA: Debt-to-Capital = Total Debt / (Total Debt + Total Equity)
    if has_debt_data and total_equity is not None and (total_debt + total_equity) != 0:
        lev["Debt-to-Capital"] = total_debt / (total_debt + total_equity)
    # CFA: Financial Leverage = Total Liabilities / Total Equity
    if total_liab is not None and total_equity and total_equity != 0:
        lev["Financial Leverage"] = total_liab / total_equity
    if ebitda and ebitda != 0 and has_debt_data and total_debt:
        lev["Debt / EBITDA"] = total_debt / ebitda
    # CFA: Interest Coverage = EBIT / Interest Expense
    if interest_expense and interest_expense != 0 and operating_income is not None:
        lev["Interest Coverage"] = operating_income / abs(interest_expense)
    # CFA: Equity Multiplier = Total Assets / Total Equity
    if total_assets and total_equity and total_equity != 0:
        lev["Equity Multiplier"] = total_assets / total_equity
    results["leverage"] = lev

    # CFA: ROIC = NOPAT / Invested Capital (placed after leverage so total_debt is available)
    tax_rate = _get("EFFECTIVE_TAX_RATE")
    if tax_rate is None:
        income_tax = _get("INCOME_TAX_EXPENSE")
        pretax = _get("PRETAX_INCOME")
        if income_tax is not None and pretax and pretax != 0:
            tax_rate = income_tax / pretax
    if operating_income is not None and tax_rate is not None:
        nopat = operating_income * (1.0 - min(max(tax_rate, 0), 1.0))
        invested_capital = (total_equity or 0) + total_debt - (cash or 0) - (short_term_inv or 0)
        if invested_capital and invested_capital > 0:
            prof["ROIC"] = nopat / invested_capital
            # Re-persist profitability since we added ROIC late
            results["profitability"] = prof

    # ── efficiency / activity
    eff: Dict[str, Optional[float]] = {}
    if ar is None:
        ar = _get("ACCOUNTS_RECEIVABLE")
    ap = _get("ACCOUNTS_PAYABLE")
    cogs = _get("COST_OF_REVENUE")
    ppe = _get("NET_FIXED_ASSETS") or _get("PROPERTY_PLANT_EQUIPMENT")
    if revenue and total_assets and total_assets != 0:
        eff["Asset Turnover"] = revenue / total_assets
    # CFA: Fixed Asset Turnover = Revenue / Net PP&E
    if revenue and ppe and ppe != 0:
        eff["Fixed Asset Turnover"] = revenue / ppe
    # CFA: Working Capital Turnover = Revenue / Working Capital
    if revenue and current_assets is not None and current_liab is not None:
        working_capital = current_assets - current_liab
        if working_capital and working_capital != 0:
            eff["Working Capital Turnover"] = revenue / working_capital
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
    eps = _get("EPS_DILUTED") or _get("EPS_BASIC") or _get("eps_basic")
    shares = (
        _get("SHARES_DILUTED") or _get("SHARES_BASIC") or _get("SHARE_COUNT")
        or _get("SHARES_OUTSTANDING_DILUTED") or _get("SHARES_OUTSTANDING_BASIC")
        or _get("TOTAL_COMMON_SHARES_OUTSTANDING")
        or _get("FILING_DATE_SHARES_OUTSTANDING")
    )
    if total_equity is not None and shares and shares != 0:
        val["Book Value / Share"] = total_equity / shares
    if eps is not None:
        val["EPS"] = eps

    # --- Payout Ratio (CFA approach with multiple fallbacks) ---
    # 1) Try total dividends paid from cash-flow statement
    dividends_paid = (
        _get("DIVIDENDS_PAID") or _get("COMMON_DIVIDENDS_PAID")
        or _get("dividends_paid")
    )
    dps: Optional[float] = None
    if dividends_paid is not None and shares and shares != 0:
        dps = abs(dividends_paid) / shares
    # 2) Fall back to DIVIDEND_PER_SHARE line item if available
    if dps is None:
        dps = _get("DIVIDEND_PER_SHARE")
    if dps is not None:
        val["Dividends / Share"] = dps

    # CFA Payout Ratio = DPS / EPS
    payout: Optional[float] = None
    if dps is not None and eps and eps != 0:
        payout = dps / eps
    # 3) Final fallback: Total Dividends Paid / Net Income
    if payout is None and dividends_paid is not None and net_income and net_income != 0:
        payout = abs(dividends_paid) / net_income

    if payout is not None:
        val["Payout Ratio"] = payout
        # CFA: Retention Rate = 1 - Payout Ratio
        val["Retention Rate"] = 1.0 - payout
        # CFA: Sustainable Growth Rate = ROE × Retention Rate
        roe = prof.get("ROE")
        if roe is not None:
            val["Sustainable Growth Rate"] = roe * (1.0 - payout)
    results["valuation"] = val

    # ── cash flow (J/K: use validated CFO and validated capex only)
    cfm: Dict[str, Optional[float]] = {}
    cfo = _get("CASH_FROM_OPERATIONS") or _get("OPERATING_CASH_FLOW")
    cfi = _get("CASH_FROM_INVESTING") or _get("INVESTING_CASH_FLOW")
    cff = _get("CASH_FROM_FINANCING") or _get("FINANCING_CASH_FLOW")

    # J) Capex: look for both PPE and intangibles capex from committed rows
    capex_ppe = _get("CAPITAL_EXPENDITURES") or _get("CAPEX")
    capex_intang = None
    # Try normalized codes that the reconciler would have set
    for code_name in items:
        low = code_name.lower()
        if "intangible" in low and ("capex" in low or "purchase" in low or "payment" in low):
            capex_intang = items[code_name]
            break

    total_capex: Optional[float] = None
    if capex_ppe is not None:
        total_capex = abs(capex_ppe)
        cfm["CAPEX PPE"] = capex_ppe
    if capex_intang is not None:
        total_capex = (total_capex or 0.0) + abs(capex_intang)
        cfm["CAPEX Intangibles"] = capex_intang

    # J) FCF = validated CFO - total cash CAPEX (deterministic, not from raw Gemini)
    fcf: Optional[float] = None
    if cfo is not None and total_capex is not None:
        fcf = cfo - total_capex
    elif cfo is not None and capex_ppe is not None:
        fcf = cfo - abs(capex_ppe)

    if cfo is not None:
        cfm["Cash from Operations"] = cfo
    if cfi is not None:
        cfm["Cash from Investing"] = cfi
    if cff is not None:
        cfm["Cash from Financing"] = cff
    if fcf is not None:
        cfm["Free Cash Flow"] = fcf
        if revenue and revenue != 0:
            cfm["FCF Margin"] = fcf / revenue
        if shares and shares != 0:
            cfm["FCF / Share"] = fcf / shares
    if cfo is not None and net_income and net_income != 0:
        cfm["CFO / Net Income"] = cfo / net_income

    # Free Cash Flow — also pick up from statement line items if not already set
    if "Free Cash Flow" not in cfm:
        stmt_fcf = _get("FREE_CASH_FLOW") or _get("UNLEVERED_FREE_CASH_FLOW") or _get("LEVERED_FREE_CASH_FLOW")
        if stmt_fcf is not None:
            cfm["Free Cash Flow"] = stmt_fcf

    # FCF Margin fallback: compute from whatever FCF we have
    if "FCF Margin" not in cfm:
        final_fcf = cfm.get("Free Cash Flow")
        if final_fcf is not None and revenue and revenue != 0:
            cfm["FCF Margin"] = final_fcf / revenue
    # Also try the pre-computed line item from the statement
    if "FCF Margin" not in cfm:
        stmt_fcf_margin = _get("FREE_CASH_FLOW_MARGIN")
        if stmt_fcf_margin is not None:
            cfm["FCF Margin"] = stmt_fcf_margin

    # FCF / Share fallback
    if "FCF / Share" not in cfm:
        final_fcf = cfm.get("Free Cash Flow")
        if final_fcf is not None and shares and shares != 0:
            cfm["FCF / Share"] = final_fcf / shares

    # Levered Free Cash Flow — pick up directly from statement line items
    lfcf = _get("LEVERED_FREE_CASH_FLOW")
    if lfcf is not None:
        cfm["Levered Free Cash Flow"] = lfcf

    results["cashflow"] = cfm

    # ── quality / supplemental
    qual: Dict[str, Optional[float]] = {}
    # CFA: Accruals Ratio = (Net Income − CFO) / Total Assets — lower is better
    if net_income is not None and cfo is not None and total_assets and total_assets != 0:
        qual["Accruals Ratio"] = (net_income - cfo) / total_assets
    # Net Debt / EBITDA (for risk scoring)
    net_debt = total_debt - (cash or 0) - (short_term_inv or 0)
    if ebitda and ebitda != 0:
        qual["Net Debt / EBITDA"] = net_debt / ebitda
    results["quality"] = qual

    # ── persist
    for category, metrics in results.items():
        for name, value in metrics.items():
            if value is not None:
                _upsert_metric(stock_id, fiscal_year, period_end_date, category, name, value, fiscal_quarter)

    return results


# ── Growth calculation ───────────────────────────────────────────────

def _calculate_growth(stock_id: int) -> Dict[str, List[Dict[str, Any]]]:
    import statistics
    growth: Dict[str, List[Dict[str, Any]]] = {}
    growth_items = [
        ("REVENUE", "Revenue Growth", "income"),  # also searches TOTAL_REVENUE below
        ("NET_INCOME", "Net Income Growth", "income"),
        ("EPS_DILUTED", "EPS Growth", "income"),
        ("TOTAL_ASSETS", "Total Assets Growth", "balance"),
        ("CASH_FROM_OPERATIONS", "Operating Cash Flow Growth", "cashflow"),
        ("FREE_CASH_FLOW", "FCF Growth", "cashflow"),
    ]

    # FCF has many names across different statements/sources;
    # try all variants when looking up a code.
    _FCF_ALIASES = [
        "FREE_CASH_FLOW",
        "UNLEVERED_FREE_CASH_FLOW",
        "LEVERED_FREE_CASH_FLOW",
    ]

    # Helper: fetch by-year series for a line item code.
    # If `codes` (list) is provided, tries each code in order and merges results.
    def _fetch_series(code: str, stmt_type: str, codes: list = None) -> Dict[int, float]:
        search_codes = codes or [code]
        by_year: Dict[int, Dict[str, Any]] = {}
        for c in search_codes:
            rows = query_all(
                """SELECT fs.period_end_date AS period, fs.fiscal_year, li.amount
                   FROM financial_line_items li
                   JOIN financial_statements fs ON fs.id = li.statement_id
                   WHERE fs.stock_id = ? AND fs.statement_type = ?
                     AND UPPER(li.line_item_code) = UPPER(?)
                   ORDER BY fs.fiscal_year, fs.period_end_date""",
                (stock_id, stmt_type, c),
            )
            for r in rows:
                if isinstance(r, (tuple, list)):
                    rec = {"period": r[0], "fiscal_year": r[1], "amount": r[2]}
                else:
                    rec = {"period": r["period"], "fiscal_year": r["fiscal_year"], "amount": r["amount"]}
                fy = rec["fiscal_year"]
                # Keep first code's data for each year (don't overwrite)
                if fy is not None and fy not in by_year:
                    by_year[fy] = rec
        return by_year

    # ── A) Standard YoY growth rates
    for code, label, stmt_type in growth_items:
        # For FCF + OCF, try multiple alias codes to cover naming variations
        if code == "FREE_CASH_FLOW":
            by_year = _fetch_series(code, stmt_type, codes=_FCF_ALIASES)
        elif code == "CASH_FROM_OPERATIONS":
            by_year = _fetch_series(code, stmt_type, codes=["CASH_FROM_OPERATIONS", "OPERATING_CASH_FLOW"])
        elif code == "REVENUE":
            by_year = _fetch_series(code, stmt_type, codes=["REVENUE", "TOTAL_REVENUE", "NET_REVENUE"])
        else:
            by_year = _fetch_series(code, stmt_type)
        sorted_years = sorted(by_year.keys())
        if len(sorted_years) < 2:
            continue

        rates: List[Dict[str, Any]] = []
        for i in range(1, len(sorted_years)):
            prev_fy = sorted_years[i - 1]
            curr_fy = sorted_years[i]
            if curr_fy - prev_fy != 1:
                continue
            prev = by_year[prev_fy]
            curr = by_year[curr_fy]
            if prev["amount"] and prev["amount"] != 0:
                g = (curr["amount"] - prev["amount"]) / abs(prev["amount"])
                rates.append({
                    "period": curr["period"],
                    "prev_period": prev["period"],
                    "growth": round(g, 4),
                })
                _upsert_metric(
                    stock_id, curr_fy, curr["period"],
                    "growth", label, round(g, 4),
                )
        if rates:
            growth[label] = rates

    # ── A2) FCF fallback: compute as CFO - CapEx when FREE_CASH_FLOW line item is missing
    if "FCF Growth" not in growth:
        cfo_by_year = _fetch_series("CASH_FROM_OPERATIONS", "cashflow")
        capex_by_year = _fetch_series("CAPITAL_EXPENDITURES", "cashflow")
        common_years = sorted(set(cfo_by_year.keys()) & set(capex_by_year.keys()))
        if len(common_years) >= 2:
            fcf_by_year = {}
            for fy in common_years:
                cfo_amt = cfo_by_year[fy]["amount"]
                capex_amt = capex_by_year[fy]["amount"]
                if cfo_amt is not None and capex_amt is not None:
                    fcf_by_year[fy] = {
                        "period": cfo_by_year[fy]["period"],
                        "fiscal_year": fy,
                        "amount": cfo_amt - abs(capex_amt),
                    }
            sorted_fy = sorted(fcf_by_year.keys())
            rates_fcf: List[Dict[str, Any]] = []
            for i in range(1, len(sorted_fy)):
                prev_fy = sorted_fy[i - 1]
                curr_fy = sorted_fy[i]
                if curr_fy - prev_fy != 1:
                    continue
                prev_amt = fcf_by_year[prev_fy]["amount"]
                curr_amt = fcf_by_year[curr_fy]["amount"]
                if prev_amt and prev_amt != 0:
                    g = (curr_amt - prev_amt) / abs(prev_amt)
                    rates_fcf.append({
                        "period": fcf_by_year[curr_fy]["period"],
                        "prev_period": fcf_by_year[prev_fy]["period"],
                        "growth": round(g, 4),
                    })
                    _upsert_metric(
                        stock_id, curr_fy, fcf_by_year[curr_fy]["period"],
                        "growth", "FCF Growth", round(g, 4),
                    )
            if rates_fcf:
                growth["FCF Growth"] = rates_fcf

    # Get newest fiscal year from the latest period
    latest_period = query_one(
        """SELECT MAX(period_end_date) as p, MAX(fiscal_year) as fy
           FROM financial_statements WHERE stock_id = ?""",
        (stock_id,),
    )
    if not latest_period:
        return growth
    latest_fy = latest_period[1] if isinstance(latest_period, (tuple, list)) else latest_period["fy"]
    latest_pd = latest_period[0] if isinstance(latest_period, (tuple, list)) else latest_period["p"]
    if not latest_fy or not latest_pd:
        return growth

    # ── B) CAGRs (3Y and 5Y) for Revenue and EPS
    cagr_items = [
        ("REVENUE", "Revenue", "income"),
        ("EPS_DILUTED", "EPS", "income"),
    ]
    for code, metric_label, stmt_type in cagr_items:
        if code == "REVENUE":
            by_year = _fetch_series(code, stmt_type, codes=["REVENUE", "TOTAL_REVENUE", "NET_REVENUE"])
        else:
            by_year = _fetch_series(code, stmt_type)
        sorted_years = sorted(by_year.keys())
        for n_years in [3, 5]:
            target_fy = latest_fy - n_years
            if target_fy in by_year and latest_fy in by_year:
                start_val = by_year[target_fy]["amount"]
                end_val = by_year[latest_fy]["amount"]
                # CAGR valid when both values are positive
                if start_val and end_val and start_val > 0 and end_val > 0:
                    cagr = (end_val / start_val) ** (1.0 / n_years) - 1.0
                    metric_name = f"{metric_label} CAGR {n_years}Y"
                    _upsert_metric(stock_id, latest_fy, latest_pd,
                                   "growth", metric_name, round(cagr, 4))
                # Handle negative-to-positive or positive-to-negative via
                # simple annualised growth rate (not geometric CAGR)
                elif start_val and end_val and start_val != 0:
                    annualised = ((end_val - start_val) / abs(start_val)) / n_years
                    metric_name = f"{metric_label} CAGR {n_years}Y"
                    _upsert_metric(stock_id, latest_fy, latest_pd,
                                   "growth", metric_name, round(annualised, 4))

    # ── C) Growth stability (stdev of YoY rates)
    for label_stub in ["Revenue Growth", "EPS Growth"]:
        rates_list = growth.get(label_stub, [])
        # Use last 5 rates max
        recent_rates = [r["growth"] for r in rates_list[-5:]]
        if len(recent_rates) >= 3:
            stdev = statistics.stdev(recent_rates)
            _upsert_metric(stock_id, latest_fy, latest_pd,
                           "growth", f"{label_stub} Stability", round(stdev, 4))

    # ── D) Margin trend (3Y delta for Net Margin and Operating Margin)
    for code, metric_label, stmt_type in [
        ("NET_INCOME", "Net Margin", "income"),
        ("OPERATING_INCOME", "Operating Margin", "income"),
    ]:
        margin_by_year = _fetch_series(code, stmt_type)
        revenue_by_year = _fetch_series("REVENUE", "income", codes=["REVENUE", "TOTAL_REVENUE", "NET_REVENUE"])
        if latest_fy in margin_by_year and latest_fy in revenue_by_year:
            for n_years in [3]:
                start_fy = latest_fy - n_years
                if (start_fy in margin_by_year and start_fy in revenue_by_year
                        and revenue_by_year.get(start_fy, {}).get("amount")
                        and revenue_by_year[start_fy]["amount"] != 0
                        and revenue_by_year[latest_fy]["amount"]
                        and revenue_by_year[latest_fy]["amount"] != 0):
                    margin_now = margin_by_year[latest_fy]["amount"] / revenue_by_year[latest_fy]["amount"]
                    margin_then = margin_by_year[start_fy]["amount"] / revenue_by_year[start_fy]["amount"]
                    delta = margin_now - margin_then  # positive = improving
                    _upsert_metric(stock_id, latest_fy, latest_pd,
                                   "growth", f"{metric_label} Trend 3Y", round(delta, 4))

    # ── E) Profit-aware growth flag: revenue growing but margins falling
    rev_rates = growth.get("Revenue Growth", [])
    if rev_rates:
        latest_rev_g = rev_rates[-1]["growth"]
        # Check if revenue grew but net margin declined
        nm_trend = None
        nm_trend_rows = query_all(
            "SELECT metric_value FROM stock_metrics WHERE stock_id = ? AND metric_name = ? ORDER BY period_end_date DESC LIMIT 1",
            (stock_id, "Net Margin Trend 3Y"),
        )
        if nm_trend_rows:
            nm_trend = nm_trend_rows[0][0] if isinstance(nm_trend_rows[0], (tuple, list)) else nm_trend_rows[0]["metric_value"]
        if latest_rev_g > 0.03 and nm_trend is not None and nm_trend < -0.02:
            # Revenue growing > 3% but net margin declined > 2pp → penalize
            _upsert_metric(stock_id, latest_fy, latest_pd,
                           "growth", "Growth Without Profit", 1.0)
        else:
            _upsert_metric(stock_id, latest_fy, latest_pd,
                           "growth", "Growth Without Profit", 0.0)

    return growth


# ── Score calculation (mirrors MetricsCalculator.compute_stock_score)

def _compute_stock_score(stock_id: int, user_id: int) -> Dict[str, Any]:
    # ── Auto-recalculate all metrics from financial statements before scoring ──
    # This ensures newly added formulas (ROIC, Accruals Ratio, Net Debt/EBITDA, etc.)
    # are applied even if the user hasn't manually re-triggered metric calculation.
    periods = query_all(
        """SELECT DISTINCT period_end_date, fiscal_year, fiscal_quarter
           FROM financial_statements
           WHERE stock_id = ?
           ORDER BY period_end_date""",
        (stock_id,),
    )
    for p in periods:
        if isinstance(p, (tuple, list)):
            ped, fy, fq = p[0], p[1], p[2]
        else:
            ped, fy, fq = p["period_end_date"], p["fiscal_year"], p.get("fiscal_quarter")
        if ped and fy:
            try:
                _calculate_all_metrics(stock_id, ped, fy, fq)
            except Exception:
                pass  # non-fatal, keep going

    # Recalculate growth (CAGRs, stability, trends, profit-aware growth)
    try:
        _calculate_growth(stock_id)
    except Exception:
        pass

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

    # Enrich with yfinance current price + volatility/beta
    symbol_row = query_one(
        "SELECT symbol FROM analysis_stocks WHERE id = ?", (stock_id,)
    )
    symbol = None
    if symbol_row:
        symbol = symbol_row[0] if isinstance(symbol_row, (tuple, list)) else symbol_row.get("symbol")
    if symbol:
        yf_ticker = _resolve_yf_ticker(symbol, user_id)
        yf_data = _fetch_yfinance_risk_data(yf_ticker)
        for k, v in yf_data.items():
            if k not in latest:  # don't overwrite DB metrics
                latest[k] = v

        # Derive ratios from DB metrics + yfinance price
        cp = latest.get("Current Price")
        if cp and cp > 0:
            bvps = latest.get("Book Value / Share")
            if bvps and bvps > 0:
                latest["P/B"] = round(cp / bvps, 4)

            eps = latest.get("EPS")
            if eps and eps > 0:
                latest["Earnings Yield"] = round(eps / cp, 6)

            # Fetch shares, EBIT, total debt, cash from balance sheet line items
            li_row = query_one("""
                SELECT
                    (SELECT li2.amount FROM financial_line_items li2
                     JOIN financial_statements fs2 ON li2.statement_id = fs2.id
                     WHERE fs2.stock_id = ? AND li2.line_item_code IN
                           ('TOTAL_COMMON_SHARES_OUTSTANDING','DILUTED_SHARES_OUTSTANDING','BASIC_SHARES_OUTSTANDING')
                     ORDER BY fs2.period_end_date DESC LIMIT 1) AS shares,
                    (SELECT li2.amount FROM financial_line_items li2
                     JOIN financial_statements fs2 ON li2.statement_id = fs2.id
                     WHERE fs2.stock_id = ? AND li2.line_item_code = 'EBIT'
                     ORDER BY fs2.period_end_date DESC LIMIT 1) AS ebit,
                    (SELECT li2.amount FROM financial_line_items li2
                     JOIN financial_statements fs2 ON li2.statement_id = fs2.id
                     WHERE fs2.stock_id = ? AND li2.line_item_code = 'TOTAL_DEBT'
                     ORDER BY fs2.period_end_date DESC LIMIT 1) AS total_debt,
                    (SELECT li2.amount FROM financial_line_items li2
                     JOIN financial_statements fs2 ON li2.statement_id = fs2.id
                     WHERE fs2.stock_id = ? AND li2.line_item_code = 'CASH_EQUIVALENTS'
                     ORDER BY fs2.period_end_date DESC LIMIT 1) AS cash
            """, (stock_id, stock_id, stock_id, stock_id))
            if li_row:
                shares = li_row[0] if isinstance(li_row, (tuple, list)) else li_row.get("shares")
                ebit = li_row[1] if isinstance(li_row, (tuple, list)) else li_row.get("ebit")
                total_debt = li_row[2] if isinstance(li_row, (tuple, list)) else li_row.get("total_debt")
                cash = li_row[3] if isinstance(li_row, (tuple, list)) else li_row.get("cash")

                if shares and shares > 0:
                    latest["Market Cap"] = cp * shares

                    if ebit and ebit > 0:
                        ev = cp * shares + (total_debt or 0) - (cash or 0)
                        latest["EV/EBIT"] = round(ev / ebit, 2)

    # Also try to compute Discount to Intrinsic Value from latest valuation
    _enrich_intrinsic_discount(stock_id, latest)

    fund, fund_breakdown = _score_fundamentals_detailed(latest)
    val, val_breakdown = _score_valuation_detailed(latest)
    growth, growth_breakdown = _score_growth_detailed(latest)
    quality, quality_breakdown = _score_quality_detailed(latest)
    risk, risk_breakdown = _score_risk_detailed(latest)

    # Positive pillars sum to 100%: Fund 30%, Quality 25%, Growth 25%, Valuation 20%
    base_score = fund * 0.30 + quality * 0.25 + growth * 0.25 + val * 0.20
    # Risk is a deduction only (up to -15%): higher risk_score = safer = less penalty
    # risk=100 → 0% penalty, risk=50 → 7.5% penalty, risk=0 → 15% penalty
    risk_penalty = (1.0 - risk / 100.0) * 0.15
    overall = base_score * (1.0 - risk_penalty)

    result = {
        "overall_score": round(overall, 1),
        "fundamental_score": round(fund, 1),
        "valuation_score": round(val, 1),
        "growth_score": round(growth, 1),
        "quality_score": round(quality, 1),
        "risk_score": round(risk, 1),
        "risk_penalty_pct": round(risk_penalty * 100, 1),
        "details": latest,
        "score_breakdown": {
            "fundamental": fund_breakdown,
            "valuation": val_breakdown,
            "growth": growth_breakdown,
            "quality": quality_breakdown,
            "risk": risk_breakdown,
        },
    }

    # Persist (risk_score goes into details JSON since column may not exist yet)
    now = int(time.time())
    # Try inserting with risk_score column, fall back to without
    try:
        exec_sql(
            """INSERT INTO stock_scores
               (stock_id, scoring_date, overall_score, fundamental_score,
                valuation_score, growth_score, quality_score, risk_score, details,
                created_by_user_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                stock_id, date.today().isoformat(), result["overall_score"],
                result["fundamental_score"], result["valuation_score"],
                result["growth_score"], result["quality_score"],
                result["risk_score"],
                json.dumps(latest), user_id, now,
            ),
        )
    except Exception:
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


def _enrich_intrinsic_discount(stock_id: int, latest: Dict[str, float]) -> None:
    """Compute discount/premium using average IV across all models (matches frontend)."""
    try:
        # Get latest IV per model_type (same logic as frontend avgIV)
        rows = query_all(
            """SELECT model_type, intrinsic_value, parameters
               FROM valuation_models
               WHERE stock_id = ? AND intrinsic_value IS NOT NULL
               ORDER BY created_at DESC""",
            (stock_id,),
        )
        if not rows:
            return

        # Keep only the latest per model_type
        seen: Dict[str, float] = {}
        price_candidates: list = []
        for r in rows:
            mt = r[0] if isinstance(r, (tuple, list)) else r.get("model_type")
            iv_val = r[1] if isinstance(r, (tuple, list)) else r.get("intrinsic_value")
            p_str = r[2] if isinstance(r, (tuple, list)) else r.get("parameters")
            if mt in seen or not iv_val or iv_val <= 0:
                continue
            seen[mt] = float(iv_val)
            # Collect price from parameters for fallback
            if p_str:
                params = json.loads(p_str) if isinstance(p_str, str) else p_str
                p = params.get("price") or params.get("current_price")
                if p and float(p) > 0:
                    price_candidates.append(float(p))

        if not seen:
            return

        avg_iv = sum(seen.values()) / len(seen)

        # Determine current price: prefer live price already enriched, then param fallback
        price = latest.get("Current Price")
        if (not price or price <= 0) and price_candidates:
            price = price_candidates[0]

        if price and price > 0:
            discount = (avg_iv - price) / avg_iv  # positive = undervalued
            latest["Discount to Intrinsic Value"] = round(discount, 4)
            latest["Intrinsic Value"] = round(avg_iv, 2)
    except Exception:
        pass


def _score_fundamentals(m: Dict[str, float]) -> float:
    return _score_fundamentals_detailed(m)[0]


def _score_fundamentals_detailed(m: Dict[str, float]):
    """Score fundamental strength: ROIC, ROE, margins, leverage, coverage, margin trend."""
    score = 50.0
    breakdown = []

    def _add(metric, value, pts, reason):
        nonlocal score
        score += pts
        breakdown.append({"metric": metric, "value": value, "points": pts, "reason": reason})

    # ── ROIC (primary return-on-capital) ─ max +15 / min -12
    roic = m.get("ROIC")
    if roic is not None:
        if roic > 0.20:   _add("ROIC", round(roic, 4), 15, "> 20% (exceptional)")
        elif roic > 0.12: _add("ROIC", round(roic, 4), 10, "> 12% (strong)")
        elif roic > 0.06: _add("ROIC", round(roic, 4), 4, "> 6% (adequate)")
        elif roic < 0:    _add("ROIC", round(roic, 4), -12, "< 0% (destroying value)")
        else:             _add("ROIC", round(roic, 4), 0, "0–6% (weak)")
    else:
        _add("ROIC", None, 0, "N/A")

    # ── ROE (complementary) ─ max +8 / min -8
    roe = m.get("ROE")
    if roe is not None:
        if roe > 0.20:   _add("ROE", round(roe, 4), 8, "> 20% (excellent)")
        elif roe > 0.12: _add("ROE", round(roe, 4), 5, "> 12% (good)")
        elif roe > 0.05: _add("ROE", round(roe, 4), 2, "> 5% (fair)")
        elif roe < 0:    _add("ROE", round(roe, 4), -8, "< 0% (negative)")
        else:            _add("ROE", round(roe, 4), 0, "0–5% (weak)")
    else:
        _add("ROE", None, 0, "N/A")

    # ── Net Margin ─ max +8 / min -8
    nm = m.get("Net Margin")
    if nm is not None:
        if nm > 0.20:   _add("Net Margin", round(nm, 4), 8, "> 20% (wide moat)")
        elif nm > 0.10: _add("Net Margin", round(nm, 4), 5, "> 10% (healthy)")
        elif nm > 0.03: _add("Net Margin", round(nm, 4), 2, "> 3% (thin)")
        elif nm < 0:    _add("Net Margin", round(nm, 4), -8, "< 0% (loss-making)")
        else:           _add("Net Margin", round(nm, 4), 0, "0–3% (very thin)")
    else:
        _add("Net Margin", None, 0, "N/A")

    # ── Net Margin Trend 3Y ─ max +6 / min -6
    nm_trend = m.get("Net Margin Trend 3Y")
    if nm_trend is not None:
        if nm_trend > 0.03:    _add("Margin Trend 3Y", round(nm_trend, 4), 6, "Expanding > 3pp")
        elif nm_trend > 0:     _add("Margin Trend 3Y", round(nm_trend, 4), 3, "Slightly expanding")
        elif nm_trend < -0.03: _add("Margin Trend 3Y", round(nm_trend, 4), -6, "Contracting > 3pp")
        elif nm_trend < 0:     _add("Margin Trend 3Y", round(nm_trend, 4), -2, "Slightly contracting")
        else:                  _add("Margin Trend 3Y", round(nm_trend, 4), 0, "Stable")
    else:
        _add("Margin Trend 3Y", None, 0, "N/A")

    # ── Current Ratio ─ max +6 / min -6
    cr = m.get("Current Ratio")
    if cr is not None:
        if 1.5 <= cr <= 3.0: _add("Current Ratio", round(cr, 2), 6, "1.5–3.0× (healthy)")
        elif cr >= 1.0:      _add("Current Ratio", round(cr, 2), 3, "≥ 1.0× (adequate)")
        else:                _add("Current Ratio", round(cr, 2), -6, "< 1.0× (liquidity risk)")
    else:
        _add("Current Ratio", None, 0, "N/A")

    # ── Debt-to-Equity ─ max +6 / min -8
    de = m.get("Debt-to-Equity")
    if de is not None:
        if de < 0.3:     _add("Debt-to-Equity", round(de, 2), 6, "< 0.3× (conservative)")
        elif de < 0.7:   _add("Debt-to-Equity", round(de, 2), 3, "< 0.7× (moderate)")
        elif de < 1.5:   _add("Debt-to-Equity", round(de, 2), 0, "0.7–1.5× (acceptable)")
        elif de > 2.5:   _add("Debt-to-Equity", round(de, 2), -8, "> 2.5× (high leverage)")
        else:            _add("Debt-to-Equity", round(de, 2), -3, "1.5–2.5× (elevated)")
    else:
        _add("Debt-to-Equity", None, 0, "N/A")

    # ── Interest Coverage ─ max +5 / min -8
    ic = m.get("Interest Coverage")
    if ic is not None:
        if ic > 8:       _add("Interest Coverage", round(ic, 2), 5, "> 8× (very safe)")
        elif ic > 3:     _add("Interest Coverage", round(ic, 2), 2, "> 3× (adequate)")
        elif ic < 1.5:   _add("Interest Coverage", round(ic, 2), -8, "< 1.5× (distress risk)")
        else:            _add("Interest Coverage", round(ic, 2), 0, "1.5–3× (tight)")
    else:
        _add("Interest Coverage", None, 0, "N/A")

    return max(0.0, min(100.0, score)), {"base": 50, "metrics": breakdown}


def _score_valuation(m: Dict[str, float]) -> float:
    return _score_valuation_detailed(m)[0]


def _score_valuation_detailed(m: Dict[str, float]):
    """Score valuation: earnings yield, P/B, payout ratio, EV/EBIT, discount to intrinsic value."""
    score = 50.0
    breakdown = []

    def _add(metric, value, pts, reason):
        nonlocal score
        score += pts
        breakdown.append({"metric": metric, "value": value, "points": pts, "reason": reason})

    # ── Earnings Yield (= 1/PE, from EPS & price) ─ max +12 / min -10
    ey = m.get("Earnings Yield")
    if ey is not None:
        if ey > 0.08:    _add("Earnings Yield", round(ey, 4), 12, "> 8% (cheap)")
        elif ey > 0.05:  _add("Earnings Yield", round(ey, 4), 6, "> 5% (fair)")
        elif ey > 0.02:  _add("Earnings Yield", round(ey, 4), 0, "2–5% (market average)")
        elif ey > 0:     _add("Earnings Yield", round(ey, 4), -5, "< 2% (expensive)")
        else:            _add("Earnings Yield", round(ey, 4), -10, "Negative earnings")
    else:
        _add("Earnings Yield", None, 0, "N/A")

    # ── EV / EBIT ─ max +8 / min -8
    ev_ebit = m.get("EV/EBIT")
    if ev_ebit is not None and ev_ebit > 0:
        if ev_ebit < 10:   _add("EV/EBIT", round(ev_ebit, 1), 8, "< 10× (cheap)")
        elif ev_ebit < 15: _add("EV/EBIT", round(ev_ebit, 1), 4, "10–15× (reasonable)")
        elif ev_ebit < 25: _add("EV/EBIT", round(ev_ebit, 1), 0, "15–25× (full)")
        elif ev_ebit < 40: _add("EV/EBIT", round(ev_ebit, 1), -4, "25–40× (rich)")
        else:              _add("EV/EBIT", round(ev_ebit, 1), -8, "> 40× (very expensive)")
    else:
        _add("EV/EBIT", None, 0, "N/A")

    # ── Price / Book ─ max +6 / min -6
    pb = m.get("P/B")
    if pb is not None and pb > 0:
        if pb < 1.0:     _add("P/B", round(pb, 2), 6, "< 1× (below book)")
        elif pb < 2.0:   _add("P/B", round(pb, 2), 3, "1–2× (reasonable)")
        elif pb < 5.0:   _add("P/B", round(pb, 2), 0, "2–5× (growth priced in)")
        else:            _add("P/B", round(pb, 2), -6, "> 5× (premium)")
    else:
        _add("P/B", None, 0, "N/A")

    # ── Intrinsic vs Price (from last valuation model) ─ max +18 / min -15
    disc = m.get("Discount to Intrinsic Value")
    iv = m.get("Intrinsic Value")
    cp = m.get("Current Price")
    iv_str = f" — IV {iv:.2f} vs Price {cp:.2f}" if iv and cp else ""
    if disc is not None:
        # disc > 0 = undervalued, disc < 0 = overvalued (as fraction, e.g. 0.20 = 20% discount)
        if disc > 0.40:      _add("Intrinsic vs Price", round(disc, 4), 18, f"> 40% undervalued (deep value){iv_str}")
        elif disc > 0.20:    _add("Intrinsic vs Price", round(disc, 4), 12, f"20–40% undervalued (attractive){iv_str}")
        elif disc > 0.10:    _add("Intrinsic vs Price", round(disc, 4), 6, f"10–20% undervalued (moderate margin){iv_str}")
        elif disc > -0.10:   _add("Intrinsic vs Price", round(disc, 4), 0, f"Within ±10% (fairly valued){iv_str}")
        elif disc > -0.20:   _add("Intrinsic vs Price", round(disc, 4), -5, f"10–20% overvalued{iv_str}")
        elif disc > -0.40:   _add("Intrinsic vs Price", round(disc, 4), -10, f"20–40% overvalued (expensive){iv_str}")
        else:                _add("Intrinsic vs Price", round(disc, 4), -15, f"> 40% overvalued (very expensive){iv_str}")
    else:
        _add("Intrinsic vs Price", None, 0, "N/A — run a valuation model first")

    # ── Payout Ratio ─ max +6 / min -6
    pr = m.get("Payout Ratio")
    if pr is not None:
        if 0.20 <= pr <= 0.60: _add("Payout Ratio", round(pr, 4), 6, "20–60% (sustainable)")
        elif pr > 1.0:         _add("Payout Ratio", round(pr, 4), -6, "> 100% (unsustainable)")
        elif pr >= 0:          _add("Payout Ratio", round(pr, 4), 0, "Outside ideal range")
        else:                  _add("Payout Ratio", round(pr, 4), -3, "Negative (no earnings)")
    else:
        _add("Payout Ratio", None, 0, "N/A")

    # ── Book Value / Share ─ max +4 / min 0
    bvps = m.get("Book Value / Share")
    if bvps is not None:
        if bvps > 0: _add("Book Value / Share", round(bvps, 3), 4, "Positive book value")
        else:        _add("Book Value / Share", round(bvps, 3), 0, "≤ 0 (negative equity)")
    else:
        _add("Book Value / Share", None, 0, "N/A")

    return max(0.0, min(100.0, score)), {"base": 50, "metrics": breakdown}


def _score_growth(m: Dict[str, float]) -> float:
    return _score_growth_detailed(m)[0]


def _score_growth_detailed(m: Dict[str, float]):
    """Score growth: CAGRs, trailing YoY, stability, profit-aware quality."""
    score = 50.0
    breakdown = []

    def _add(metric, value, pts, reason):
        nonlocal score
        score += pts
        breakdown.append({"metric": metric, "value": value, "points": pts, "reason": reason})

    # ── Revenue CAGR (prefer 5Y, fallback 3Y) ─ max +10 / min -8
    rev_cagr = m.get("Revenue CAGR 5Y") or m.get("Revenue CAGR 3Y")
    label = "Revenue CAGR 5Y" if m.get("Revenue CAGR 5Y") is not None else "Revenue CAGR 3Y"
    if rev_cagr is not None:
        if rev_cagr > 0.15:   _add(label, round(rev_cagr, 4), 10, "> 15% (high growth)")
        elif rev_cagr > 0.07: _add(label, round(rev_cagr, 4), 6, "> 7% (solid)")
        elif rev_cagr > 0.02: _add(label, round(rev_cagr, 4), 2, "> 2% (slow growth)")
        elif rev_cagr < -0.05:_add(label, round(rev_cagr, 4), -8, "< −5% (shrinking)")
        else:                 _add(label, round(rev_cagr, 4), 0, "−5% to 2% (stagnant)")
    else:
        _add("Revenue CAGR", None, 0, "N/A")

    # ── EPS CAGR (prefer 5Y, fallback 3Y) ─ max +10 / min -8
    eps_cagr = m.get("EPS CAGR 5Y") or m.get("EPS CAGR 3Y")
    label2 = "EPS CAGR 5Y" if m.get("EPS CAGR 5Y") is not None else "EPS CAGR 3Y"
    if eps_cagr is not None:
        if eps_cagr > 0.15:   _add(label2, round(eps_cagr, 4), 10, "> 15% (high growth)")
        elif eps_cagr > 0.07: _add(label2, round(eps_cagr, 4), 6, "> 7% (solid)")
        elif eps_cagr > 0:    _add(label2, round(eps_cagr, 4), 2, "> 0% (positive)")
        elif eps_cagr < -0.10:_add(label2, round(eps_cagr, 4), -8, "< −10% (declining fast)")
        else:                 _add(label2, round(eps_cagr, 4), -2, "Slightly declining")
    else:
        _add("EPS CAGR", None, 0, "N/A")

    # ── Trailing Revenue Growth (YoY) ─ max +8 / min -8
    rg = m.get("Revenue Growth")
    if rg is not None:
        if rg > 0.10:     _add("Revenue Growth YoY", round(rg, 4), 8, "> 10% (strong)")
        elif rg > 0.03:   _add("Revenue Growth YoY", round(rg, 4), 4, "> 3% (moderate)")
        elif rg < -0.05:  _add("Revenue Growth YoY", round(rg, 4), -8, "< −5% (declining)")
        else:             _add("Revenue Growth YoY", round(rg, 4), 0, "−5% to 3% (flat)")
    else:
        _add("Revenue Growth YoY", None, 0, "N/A")

    # ── Trailing EPS Growth (YoY) ─ max +8 / min -8
    eg = m.get("EPS Growth")
    if eg is not None:
        if eg > 0.10:     _add("EPS Growth YoY", round(eg, 4), 8, "> 10% (strong)")
        elif eg > 0:      _add("EPS Growth YoY", round(eg, 4), 3, "> 0% (positive)")
        elif eg < -0.10:  _add("EPS Growth YoY", round(eg, 4), -8, "< −10% (sharp decline)")
        else:             _add("EPS Growth YoY", round(eg, 4), -2, "Slightly declining")
    else:
        _add("EPS Growth YoY", None, 0, "N/A")

    # ── Revenue Growth Stability (stdev) ─ max +5 / min -5
    stab = m.get("Revenue Growth Stability")
    if stab is not None:
        if stab < 0.05:    _add("Revenue Stability", round(stab, 4), 5, "< 5% σ (very consistent)")
        elif stab < 0.10:  _add("Revenue Stability", round(stab, 4), 2, "5–10% σ (consistent)")
        elif stab > 0.25:  _add("Revenue Stability", round(stab, 4), -5, "> 25% σ (erratic)")
        else:              _add("Revenue Stability", round(stab, 4), 0, "10–25% σ (moderate)")
    else:
        _add("Revenue Stability", None, 0, "N/A")

    # ── Profit-Aware Growth penalty ─ max 0 / min -6
    gwp = m.get("Growth Without Profit")
    if gwp is not None and gwp > 0:
        _add("Profit-Aware Growth", 1, -6, "Revenue growing but margins falling")
    else:
        _add("Profit-Aware Growth", 0, 0, "No margin deterioration detected")

    return max(0.0, min(100.0, score)), {"base": 50, "metrics": breakdown}


def _score_quality(m: Dict[str, float]) -> float:
    return _score_quality_detailed(m)[0]


def _score_quality_detailed(m: Dict[str, float]):
    """Score earnings quality: cash conversion, FCF margin, accruals, ROIC level."""
    score = 50.0
    breakdown = []

    def _add(metric, value, pts, reason):
        nonlocal score
        score += pts
        breakdown.append({"metric": metric, "value": value, "points": pts, "reason": reason})

    # ── CFO / Net Income (cash conversion) ─ max +12 / min -10
    cfoni = m.get("CFO / Net Income")
    if cfoni is not None:
        if cfoni > 1.2:     _add("CFO / Net Income", round(cfoni, 2), 12, "> 1.2× (excellent cash conversion)")
        elif cfoni > 0.8:   _add("CFO / Net Income", round(cfoni, 2), 6, "> 0.8× (solid)")
        elif cfoni > 0.5:   _add("CFO / Net Income", round(cfoni, 2), 0, "0.5–0.8× (moderate)")
        else:               _add("CFO / Net Income", round(cfoni, 2), -10, "< 0.5× (poor cash conversion)")
    else:
        _add("CFO / Net Income", None, 0, "N/A")

    # ── FCF Margin ─ max +10 / min -8
    fcf_m = m.get("FCF Margin")
    if fcf_m is not None:
        if fcf_m > 0.15:    _add("FCF Margin", round(fcf_m, 4), 10, "> 15% (exceptional)")
        elif fcf_m > 0.08:  _add("FCF Margin", round(fcf_m, 4), 5, "> 8% (healthy)")
        elif fcf_m > 0:     _add("FCF Margin", round(fcf_m, 4), 2, "> 0% (positive)")
        else:               _add("FCF Margin", round(fcf_m, 4), -8, "≤ 0% (cash burn)")
    else:
        _add("FCF Margin", None, 0, "N/A")

    # ── Accruals Ratio (NI − CFO) / Total Assets ─ max +10 / min -10
    # Lower (more negative) = higher quality; high positive = aggressive accounting
    ar = m.get("Accruals Ratio")
    if ar is not None:
        if ar < -0.05:      _add("Accruals Ratio", round(ar, 4), 10, "< −5% (high quality earnings)")
        elif ar < 0.03:     _add("Accruals Ratio", round(ar, 4), 4, "< 3% (clean)")
        elif ar < 0.10:     _add("Accruals Ratio", round(ar, 4), 0, "3–10% (moderate)")
        else:               _add("Accruals Ratio", round(ar, 4), -10, "> 10% (aggressive accruals)")
    else:
        _add("Accruals Ratio", None, 0, "N/A")

    # ── ROIC level (quality of returns) ─ max +8 / min -6
    roic = m.get("ROIC")
    if roic is not None:
        if roic > 0.15:     _add("ROIC (Quality)", round(roic, 4), 8, "> 15% (value creator)")
        elif roic > 0.08:   _add("ROIC (Quality)", round(roic, 4), 4, "> 8% (above WACC)")
        elif roic > 0:      _add("ROIC (Quality)", round(roic, 4), 0, "0–8% (marginal)")
        else:               _add("ROIC (Quality)", round(roic, 4), -6, "< 0% (value destroyer)")
    else:
        _add("ROIC (Quality)", None, 0, "N/A")

    # ── Operating Margin Trend 3Y ─ max +5 / min -5
    om_trend = m.get("Operating Margin Trend 3Y")
    if om_trend is not None:
        if om_trend > 0.02:    _add("Op Margin Trend 3Y", round(om_trend, 4), 5, "Expanding > 2pp")
        elif om_trend > 0:     _add("Op Margin Trend 3Y", round(om_trend, 4), 2, "Slightly expanding")
        elif om_trend < -0.02: _add("Op Margin Trend 3Y", round(om_trend, 4), -5, "Contracting > 2pp")
        else:                  _add("Op Margin Trend 3Y", round(om_trend, 4), 0, "Stable / slight decline")
    else:
        _add("Op Margin Trend 3Y", None, 0, "N/A")

    return max(0.0, min(100.0, score)), {"base": 50, "metrics": breakdown}


def _resolve_yf_ticker(symbol: str, user_id: int = None) -> str:
    """Resolve a raw symbol to its yfinance-compatible ticker.

    Priority: 1) stocks table yf_ticker, 2) KUWAIT_STOCKS list, 3) raw symbol.
    """
    # Try the stocks table (populated when user added the stock)
    if user_id:
        row = query_one(
            "SELECT yf_ticker FROM stocks WHERE symbol = ? AND user_id = ? AND yf_ticker IS NOT NULL",
            (symbol, user_id),
        )
    else:
        row = query_one(
            "SELECT yf_ticker FROM stocks WHERE symbol = ? AND yf_ticker IS NOT NULL LIMIT 1",
            (symbol,),
        )
    if row:
        yft = row[0] if isinstance(row, (tuple, list)) else row.get("yf_ticker")
        if yft:
            return yft

    # Check the hardcoded stock lists (covers peers not in the user's portfolio)
    from app.data.stock_lists import KUWAIT_STOCKS
    upper = symbol.upper()
    for s in KUWAIT_STOCKS:
        if s["symbol"].upper() == upper:
            return s["yf_ticker"]

    return symbol


def _fetch_yfinance_risk_data(symbol: str) -> Dict[str, float]:
    """Fetch current price, beta, volatility & drawdown from yfinance.

    All ratios (P/B, Earnings Yield, EV/EBIT, etc.) are computed from
    DB data in _compute_stock_score — yfinance only provides price +
    price-history-derived metrics.
    """
    import math
    import signal
    import threading

    _kw = symbol.upper().endswith(".KW")
    data: Dict[str, float] = {}

    def _fetch():
        nonlocal data
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            # Current price (only price gets /1000 for .KW)
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if price and price > 0:
                price = float(price)
                if _kw:
                    price = price / 1000.0
                data["Current Price"] = round(price, 2)

            # Beta
            beta = info.get("beta")
            if beta is not None:
                data["Beta"] = float(beta)

            # 1Y price history → volatility & max drawdown
            hist = ticker.history(period="1y")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()
                if len(closes) > 20:
                    returns = closes.pct_change().dropna()
                    ann_vol = float(returns.std()) * math.sqrt(252)
                    data["1Y Volatility"] = round(ann_vol, 4)

                    # Max drawdown
                    cummax = closes.cummax()
                    drawdown = (closes - cummax) / cummax
                    data["Max Drawdown 1Y"] = round(float(drawdown.min()), 4)
        except Exception:
            pass  # yfinance failures are non-fatal

    # Run with a 15-second timeout to avoid hanging the request
    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(timeout=15)
    return data


def _score_risk_detailed(m: Dict[str, float]):
    """Score risk / downside: volatility, drawdown, balance sheet risk, earnings risk, size."""
    score = 50.0
    breakdown = []

    def _add(metric, value, pts, reason):
        nonlocal score
        score += pts
        breakdown.append({"metric": metric, "value": value, "points": pts, "reason": reason})

    # ── 1Y Volatility (annualized) ─ max +8 / min -10
    vol = m.get("1Y Volatility")
    if vol is not None:
        if vol < 0.15:      _add("1Y Volatility", round(vol, 4), 8, "< 15% (low risk)")
        elif vol < 0.25:    _add("1Y Volatility", round(vol, 4), 3, "15–25% (moderate)")
        elif vol < 0.40:    _add("1Y Volatility", round(vol, 4), -3, "25–40% (elevated)")
        else:               _add("1Y Volatility", round(vol, 4), -10, "> 40% (very volatile)")
    else:
        _add("1Y Volatility", None, 0, "N/A")

    # ── Max Drawdown 1Y ─ max +6 / min -8
    dd = m.get("Max Drawdown 1Y")
    if dd is not None:
        # dd is negative (e.g. -0.20 = -20%)
        if dd > -0.10:      _add("Max Drawdown 1Y", round(dd, 4), 6, "< 10% (resilient)")
        elif dd > -0.20:    _add("Max Drawdown 1Y", round(dd, 4), 2, "10–20% (normal)")
        elif dd > -0.35:    _add("Max Drawdown 1Y", round(dd, 4), -3, "20–35% (significant)")
        else:               _add("Max Drawdown 1Y", round(dd, 4), -8, "> 35% (severe)")
    else:
        _add("Max Drawdown 1Y", None, 0, "N/A")

    # ── Net Debt / EBITDA ─ max +8 / min -10
    nd_ebitda = m.get("Net Debt / EBITDA")
    if nd_ebitda is not None:
        if nd_ebitda < 0:       _add("Net Debt / EBITDA", round(nd_ebitda, 2), 8, "Net cash position")
        elif nd_ebitda < 1.5:   _add("Net Debt / EBITDA", round(nd_ebitda, 2), 5, "< 1.5× (conservative)")
        elif nd_ebitda < 3.0:   _add("Net Debt / EBITDA", round(nd_ebitda, 2), 0, "1.5–3× (moderate)")
        elif nd_ebitda < 5.0:   _add("Net Debt / EBITDA", round(nd_ebitda, 2), -5, "3–5× (elevated)")
        else:                   _add("Net Debt / EBITDA", round(nd_ebitda, 2), -10, "> 5× (high leverage risk)")
    else:
        _add("Net Debt / EBITDA", None, 0, "N/A")

    # ── Interest Coverage (risk lens) ─ max +5 / min -8
    ic = m.get("Interest Coverage")
    if ic is not None:
        if ic > 8:          _add("Interest Coverage (Risk)", round(ic, 2), 5, "> 8× (no debt risk)")
        elif ic > 3:        _add("Interest Coverage (Risk)", round(ic, 2), 2, "> 3× (manageable)")
        elif ic > 1.5:      _add("Interest Coverage (Risk)", round(ic, 2), -3, "1.5–3× (tight)")
        else:               _add("Interest Coverage (Risk)", round(ic, 2), -8, "< 1.5× (debt distress)")
    else:
        _add("Interest Coverage (Risk)", None, 0, "N/A")

    # ── Revenue Growth Stability (risk lens) ─ max +5 / min -5
    stab = m.get("Revenue Growth Stability")
    if stab is not None:
        if stab < 0.05:     _add("Revenue Variability", round(stab, 4), 5, "< 5% σ (predictable)")
        elif stab < 0.10:   _add("Revenue Variability", round(stab, 4), 2, "5–10% σ (stable)")
        elif stab > 0.25:   _add("Revenue Variability", round(stab, 4), -5, "> 25% σ (unpredictable)")
        else:               _add("Revenue Variability", round(stab, 4), 0, "10–25% σ (normal)")
    else:
        _add("Revenue Variability", None, 0, "N/A")

    # ── Market Cap (size premium) ─ max +5 / min -3
    mcap = m.get("Market Cap")
    if mcap is not None:
        if mcap > 50e9:         _add("Market Cap", round(mcap / 1e9, 1), 5, "> $50B (mega cap)")
        elif mcap > 10e9:       _add("Market Cap", round(mcap / 1e9, 1), 3, "$10–50B (large cap)")
        elif mcap > 2e9:        _add("Market Cap", round(mcap / 1e9, 1), 0, "$2–10B (mid cap)")
        elif mcap > 300e6:      _add("Market Cap", round(mcap / 1e9, 1), -2, "$300M–2B (small cap)")
        else:                   _add("Market Cap", round(mcap / 1e9, 1), -3, "< $300M (micro cap risk)")
    else:
        _add("Market Cap", None, 0, "N/A")

    # ── Beta ─ max +4 / min -5
    beta = m.get("Beta")
    if beta is not None:
        if beta < 0.7:      _add("Beta", round(beta, 2), 4, "< 0.7 (defensive)")
        elif beta < 1.2:    _add("Beta", round(beta, 2), 2, "0.7–1.2 (market-like)")
        elif beta < 1.8:    _add("Beta", round(beta, 2), -2, "1.2–1.8 (above-market)")
        else:               _add("Beta", round(beta, 2), -5, "> 1.8 (high systematic risk)")
    else:
        _add("Beta", None, 0, "N/A")

    return max(0.0, min(100.0, score)), {"base": 50, "metrics": breakdown}


# ── Valuation model helpers ──────────────────────────────────────────

def _graham_number(eps: float, growth_rate: float = 0.0,
                   corporate_yield: float = 4.4, margin_of_safety: float = 25.0,
                   current_price: float | None = None) -> Dict[str, Any]:
    """Graham valuation — computes both Original (8.5+2g) and Revised (7+1g)."""
    # ── Edge case: EPS ≤ 0 → model not applicable
    if eps <= 0:
        return {"model": "graham", "intrinsic_value": None,
                "error": "N/A – Unprofitable (EPS ≤ 0)",
                "verdict": "N/A - Unprofitable",
                "parameters": {"eps": eps, "growth_rate": growth_rate,
                               "corporate_yield": corporate_yield}}

    # ── Growth rate: already whole-number (e.g. 5 for 5%)
    # Apply Graham conservatism: cap at 15%, floor at 0%
    g = min(max(growth_rate, 0), 15)

    # ── Corporate bond yield (Y)
    # If user passed decimal < 1 interpret as percentage (e.g. 0.042 → 4.2)
    y_val = corporate_yield if corporate_yield > 1 else corporate_yield * 100
    if y_val <= 0:
        y_val = 4.4  # fallback to 1962 baseline

    # ── Original formula: V = EPS × (8.5 + 2g) × 4.4 / Y
    implied_pe_original = 8.5 + 2 * g
    iv_original = eps * implied_pe_original * 4.4 / y_val

    # ── Revised formula: V* = EPS × (7 + 1g) × 4.4 / Y
    implied_pe_revised = 7 + 1 * g
    iv_revised = eps * implied_pe_revised * 4.4 / y_val

    # Primary intrinsic value = revised (more conservative)
    intrinsic_value = iv_revised

    # ── Margin of safety (as percentage, e.g. 25 for 25%)
    mos_decimal = margin_of_safety / 100
    buy_price_target = intrinsic_value * (1 - mos_decimal)

    # ── Verdict (requires current price)
    verdict = "N/A - No Current Price"
    if current_price is not None and intrinsic_value > 0:
        if current_price <= buy_price_target:
            verdict = "Undervalued (Buy)"
        elif current_price <= intrinsic_value:
            verdict = "Fair Value (Hold)"
        else:
            verdict = "Overvalued (Sell/Avoid)"

    return {
        "model": "graham",
        "intrinsic_value": round(intrinsic_value, 4),
        "iv_original": round(iv_original, 4),
        "iv_revised": round(iv_revised, 4),
        "implied_pe_original": round(implied_pe_original, 2),
        "implied_pe_revised": round(implied_pe_revised, 2),
        "buy_price_target": round(buy_price_target, 4),
        "current_price": round(current_price, 4) if current_price is not None else None,
        "verdict": verdict,
        "acceptable_buy_price": round(buy_price_target, 4),
        "parameters": {
            "eps": eps,
            "growth_rate": g,
            "aaa_yield": round(y_val, 4),
            "margin_of_safety": margin_of_safety,
            "iv_original": round(iv_original, 4),
            "iv_revised": round(iv_revised, 4),
            "price": round(current_price, 4) if current_price is not None else None,
        },
        "assumptions": {
            "formula_original": "V = EPS × (8.5 + 2g) × 4.4 / Y",
            "formula_revised": "V* = EPS × (7 + 1g) × 4.4 / Y",
            "base_pe_original": 8.5,
            "base_pe_revised": 7,
            "no_growth_yield": 4.4,
            "growth_cap": 15,
        },
    }


def _dcf(fcf, g1, g2, dr, s1=5, s2=5, tg=0.025, shares=1.0,
         cash=0.0, debt=0.0, wacc_components=None):
    if dr <= tg:
        return {"model": "dcf", "intrinsic_value": None, "error": "Discount rate must exceed terminal growth."}
    projections = []  # year-by-year table for UI
    projected_pvs = []
    cf = fcf
    for yr in range(1, s1 + 1):
        cf *= 1 + g1
        pv = cf / ((1 + dr) ** yr)
        projected_pvs.append(pv)
        projections.append({"year": yr, "stage": 1, "fcf": round(cf, 2), "pv": round(pv, 2)})
    for yr in range(s1 + 1, s1 + s2 + 1):
        cf *= 1 + g2
        pv = cf / ((1 + dr) ** yr)
        projected_pvs.append(pv)
        projections.append({"year": yr, "stage": 2, "fcf": round(cf, 2), "pv": round(pv, 2)})
    tv = cf * (1 + tg) / (dr - tg)
    pv_tv = tv / ((1 + dr) ** (s1 + s2))
    sum_pv_fcfs = sum(projected_pvs)
    ev = sum_pv_fcfs + pv_tv
    # Enterprise-to-equity bridge
    equity_value = ev + (cash or 0) - (debt or 0)
    ps = equity_value / shares if shares else 0
    tv_pct = (pv_tv / ev * 100) if ev > 0 else 0
    params = {"fcf": fcf, "growth_stage1": g1, "growth_stage2": g2, "discount_rate": dr,
              "stage1_years": s1, "stage2_years": s2, "terminal_growth": tg,
              "shares_outstanding": shares, "cash": cash, "debt": debt}
    if wacc_components:
        params["wacc"] = wacc_components
    return {"model": "dcf", "intrinsic_value": round(ps, 2), "enterprise_value": round(ev, 2),
            "equity_value": round(equity_value, 2),
            "pv_terminal": round(pv_tv, 2), "pv_fcfs": round(sum_pv_fcfs, 2),
            "terminal_value": round(tv, 2), "tv_pct_of_ev": round(tv_pct, 1),
            "cash": round(cash or 0, 2), "debt": round(debt or 0, 2),
            "projections": projections,
            "parameters": params,
            "assumptions": {"method": "Two-stage DCF with Gordon Growth terminal value"}}


def _ddm(div, gr, rr, hgy=5, hgr=None):
    if rr <= gr:
        return {"model": "ddm", "intrinsic_value": None, "error": "Required return must exceed stable growth."}
    d1 = div * (1 + gr)
    if hgr is None:
        iv = d1 / (rr - gr)
        return {"model": "ddm", "intrinsic_value": round(iv, 2),
                "d1": round(d1, 4), "spread": round(rr - gr, 4),
                "parameters": {"last_dividend": div, "growth_rate": gr, "required_return": rr},
                "assumptions": {"method": "Gordon Growth (single stage)",
                                "formula": "D₁ / (r − g)"}}
    # Two-stage DDM
    projections = []
    pv_div = 0.0
    d = div
    for yr in range(1, hgy + 1):
        d *= 1 + hgr
        pv = d / ((1 + rr) ** yr)
        pv_div += pv
        projections.append({"year": yr, "dividend": round(d, 4), "pv": round(pv, 4)})
    tv = d * (1 + gr) / (rr - gr)
    pv_tv = tv / ((1 + rr) ** hgy)
    iv = pv_div + pv_tv
    return {"model": "ddm", "intrinsic_value": round(iv, 2), "pv_dividends": round(pv_div, 2),
            "pv_terminal": round(pv_tv, 2), "d1": round(d1, 4),
            "projections": projections,
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
    # Merge computed output into assumptions so history cards can render full detail
    enriched_assumptions = dict(result.get("assumptions", {}))
    for key in ("projections", "pv_fcfs", "pv_terminal", "terminal_value",
                "tv_pct_of_ev", "enterprise_value", "equity_value", "cash", "debt"):
        if key in result:
            enriched_assumptions[key] = result[key]
    exec_sql(
        """INSERT INTO valuation_models
           (stock_id, model_type, valuation_date, intrinsic_value,
            parameters, assumptions, created_by_user_id, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            stock_id, result["model"], date.today().isoformat(),
            result.get("intrinsic_value"),
            json.dumps(result.get("parameters", {})),
            json.dumps(enriched_assumptions),
            user_id, now,
        ),
    )
