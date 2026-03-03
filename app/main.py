"""
Mobile Migration — FastAPI Backend  (v1 architecture)

Main application entry point.
Run with:  uvicorn app.main:app --reload --port 8002
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import get_settings
from app.core.database import check_db_exists
from app.core.limiter import limiter
from app.core.exceptions import APIError, api_error_handler, unhandled_exception_handler
from app.core.middleware import SecurityHeadersMiddleware, RequestSizeLimitMiddleware
from app.core.json_response import SafeJSONResponse

# Versioned API router (all /api/v1/* routes)
from app.api.v1 import v1_router

# Legacy flat routers kept for backward-compat on old prefixes
from app.api.auth import router as auth_router_legacy
from app.api.portfolio import router as portfolio_router_legacy
from app.api.cron import router as cron_router_legacy

# Cron scheduler
from app.cron.scheduler import start_scheduler, stop_scheduler

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


# ── Lifespan (startup / shutdown) ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not check_db_exists():
        logger.error(
            "⛔  dev_portfolio.db NOT FOUND at %s\n"
            "    → Copy your portfolio.db into mobile-migration/ first:\n"
            "      copy portfolio.db mobile-migration/dev_portfolio.db",
            settings.database_abs_path,
        )
    else:
        logger.info("✅  Database found: %s", settings.database_abs_path)
        # Additive migrations — safe to run every startup
        from app.core.database import add_column_if_missing, exec_sql, query_df
        add_column_if_missing("stocks", "yf_ticker", "TEXT")

        # Ensure PFM tables exist (additive — safe to run every startup)
        exec_sql("""
            CREATE TABLE IF NOT EXISTS pfm_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                snapshot_date TEXT NOT NULL,
                notes TEXT,
                total_assets REAL DEFAULT 0,
                total_liabilities REAL DEFAULT 0,
                net_worth REAL DEFAULT 0,
                created_at INTEGER
            )
        """)
        exec_sql("""
            CREATE TABLE IF NOT EXISTS pfm_assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                asset_type TEXT NOT NULL,
                category TEXT NOT NULL,
                name TEXT NOT NULL,
                quantity REAL,
                price REAL,
                currency TEXT DEFAULT 'KWD',
                value_kwd REAL DEFAULT 0,
                created_at INTEGER
            )
        """)
        exec_sql("""
            CREATE TABLE IF NOT EXISTS pfm_liabilities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                amount_kwd REAL DEFAULT 0,
                is_current INTEGER DEFAULT 0,
                is_long_term INTEGER DEFAULT 0,
                created_at INTEGER
            )
        """)
        exec_sql("""
            CREATE TABLE IF NOT EXISTS pfm_income_expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                category TEXT NOT NULL,
                monthly_amount REAL DEFAULT 0,
                is_finance_cost INTEGER DEFAULT 0,
                is_gna INTEGER DEFAULT 0,
                sort_order INTEGER DEFAULT 0,
                created_at INTEGER
            )
        """)
        logger.info("✅  PFM tables ensured")

        # User settings table (stores rf_rate, etc.)
        exec_sql("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER NOT NULL,
                setting_key TEXT NOT NULL,
                setting_value TEXT NOT NULL,
                updated_at INTEGER,
                PRIMARY KEY (user_id, setting_key)
            )
        """)
        logger.info("✅  user_settings table ensured")

        # Backfill yf_ticker for existing stocks that don't have one yet
        try:
            from app.data.stock_lists import KUWAIT_STOCKS, US_STOCKS
            ticker_map = {}
            for s in KUWAIT_STOCKS:
                ticker_map[s["symbol"].upper()] = s["yf_ticker"]
            for s in US_STOCKS:
                ticker_map[s["symbol"].upper()] = s["yf_ticker"]
            missing = query_df(
                "SELECT id, symbol FROM stocks WHERE yf_ticker IS NULL OR yf_ticker = ''"
            )
            if missing is not None and not missing.empty:
                updated = 0
                for _, row in missing.iterrows():
                    sym = str(row["symbol"]).strip().upper()
                    yf = ticker_map.get(sym)
                    if yf:
                        exec_sql("UPDATE stocks SET yf_ticker = ? WHERE id = ?", (yf, row["id"]))
                        updated += 1
                if updated:
                    logger.info("Backfilled yf_ticker for %d existing stocks", updated)
        except Exception as e:
            logger.warning("yf_ticker backfill skipped: %s", e)
    start_scheduler()

    # ── Production security audit ────────────────────────────────────
    if settings.is_production:
        _issues = []
        if settings.SECRET_KEY in ("change_this_to_a_random_string_before_production", ""):
            _issues.append("SECRET_KEY is still the default — change it!")
        if not settings.CRON_SECRET_KEY:
            _issues.append("CRON_SECRET_KEY is empty — cron endpoint is disabled.")
        if "*" in settings.CORS_ORIGINS or "localhost" in settings.CORS_ORIGINS:
            _issues.append("CORS_ORIGINS contains wildcard or localhost.")
        if _issues:
            for issue in _issues:
                logger.warning("🔒 SECURITY: %s", issue)
    else:
        logger.info("🔧 Running in DEVELOPMENT mode (CORS=*, verbose errors)")

    logger.info("🚀  Backend API starting on http://localhost:8002")
    logger.info("📖  Swagger docs at http://localhost:8002/docs")

    yield  # app is running

    # Shutdown
    stop_scheduler()
    logger.info("👋  Backend API shutting down")


# ── App factory ──────────────────────────────────────────────────────

app = FastAPI(
    title="Portfolio Mobile API",
    version="1.0.0",
    default_response_class=SafeJSONResponse,
    description=(
        "REST API for the Portfolio Mobile Migration.\n\n"
        "**Versioned API:** All endpoints live under `/api/v1/`.\n\n"
        "**Auth:** POST `/api/v1/auth/login` (JSON) or `/api/v1/auth/login/form` (OAuth2) to get a JWT.\n"
        "Then click **Authorize** (top-right) and paste the token."
    ),
    lifespan=lifespan,
)

# ── Exception handlers ──────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


# ── Security Middleware ──────────────────────────────────────────────
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestSizeLimitMiddleware)

# ── CORS Middleware ──────────────────────────────────────────────────
# NOTE: allow_origins=["*"] + allow_credentials=True is spec-invalid.
# Starlette echoes origin on preflight but returns "*" on actual requests,
# causing browsers to reject credentialed responses with "Network Error".
# Fix: list explicit dev origins so the browser always sees the real origin.
_dev_origins = [
    "http://localhost:8081",   # Expo web
    "http://localhost:19006",  # Expo web (alt port)
    "http://localhost:3000",   # dev fallback
    "http://localhost:8004",   # Swagger UI
    "http://192.168.1.5:8081", # LAN mobile browser
    "http://127.0.0.1:8081",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list if settings.is_production else _dev_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
)


# ── Routes ───────────────────────────────────────────────────────────

# v1 versioned API (primary)
app.include_router(v1_router)

# Legacy unversioned routes (kept for backward compat — will be removed)
app.include_router(auth_router_legacy)
app.include_router(portfolio_router_legacy)
app.include_router(cron_router_legacy)


# ── Health check (no auth) ──────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "db_exists": check_db_exists(),
        "db_path": settings.database_abs_path,
    }
