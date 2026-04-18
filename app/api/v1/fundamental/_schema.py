"""
Fundamental Analysis — schema initialization.

Creates all analysis tables on first use (idempotent).
Extracted from the monolithic fundamental.py.
"""

import logging

from app.core.database import exec_sql

logger = logging.getLogger(__name__)

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
