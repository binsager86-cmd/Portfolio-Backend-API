"""
Mobile Migration — FastAPI Backend  (v1 architecture)

Main application entry point.
Run with:  uvicorn app.main:app --reload --port 8004
"""

import logging
import os
import json
import asyncio
from collections.abc import Sequence
from contextlib import asynccontextmanager

import tracemalloc

import pandas as pd
pd.set_option("future.no_silent_downcasting", True)

import sentry_sdk
from prometheus_fastapi_instrumentator import Instrumentator
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import get_settings
from app.core.database import check_db_exists
from app.core.limiter import limiter
from app.core.exceptions import APIError, api_error_handler, unhandled_exception_handler
from app.core.middleware import (
    SecurityHeadersMiddleware,
    RequestSizeLimitMiddleware,
    PrivateNetworkAccessMiddleware,
    CorrelationIDMiddleware,
    RequestTimingMiddleware,
    LegacyDeprecationMiddleware,
    AssetCacheMiddleware,
)
from app.core.json_response import SafeJSONResponse
from app.core.feature_flags import get_flags

# Versioned API router (all /api/v1/* routes)
from app.api.v1 import v1_router

# Legacy flat routers kept for backward-compat on old prefixes
from app.api.auth import router as auth_router_legacy
from app.api.portfolio import router as portfolio_router_legacy
from app.api.cron import router as cron_router_legacy

# Cron scheduler
from app.cron.scheduler import start_scheduler, stop_scheduler

# ── Logging ──────────────────────────────────────────────────────────
from app.core.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()


def _join_router_prefixes(*parts: str) -> str:
    normalized_parts = []
    for part in parts:
        if not part:
            continue
        normalized_parts.append(part if part.startswith("/") else f"/{part}")
    return "".join(normalized_parts)


def _annotate_included_router_paths(routes: Sequence[object]) -> None:
    for route in routes:
        original_router = getattr(route, "original_router", None)
        include_context = getattr(route, "include_context", None)
        if original_router is None or include_context is None:
            continue

        prefix = _join_router_prefixes(
            getattr(include_context, "prefix", ""),
            getattr(original_router, "prefix", ""),
        )
        route.path = prefix
        route.path_format = prefix

        child_routes = getattr(original_router, "routes", None)
        if child_routes:
            _annotate_included_router_paths(child_routes)


# Sentry Init (Production Only)
if os.getenv("ENVIRONMENT") == "production" and os.getenv("SENTRY_DSN"):
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        integrations=[FastApiIntegration(), SqlalchemyIntegration()],
        traces_sample_rate=0.2,
        environment="production",
        release=os.getenv("GIT_COMMIT_SHA", "dev"),
    )


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

        # ── [B-3] Schema managed by Alembic — skipped in-process to avoid
        # SQLite lock contention on OneDrive/networked filesystems.
        # Run `alembic upgrade head` from the CLI before starting the server
        # when schema changes are needed.
        logger.info("ℹ️  Alembic in-process migration skipped (run CLI manually if needed)")

        # ── Ensure core tables exist for legacy/stamped databases ──
        # Some local DBs can be stamped to head without all table DDL actually
        # present (for example after manual DB swaps). This idempotent schema
        # initializer backfills missing tables like user_settings so runtime
        # endpoints do not fail with OperationalError.
        try:
            from app.core.schema import ensure_all_tables
            ensure_all_tables()
            logger.info("✅  Core schema ensured (idempotent)")
        except Exception as schema_err:
            logger.warning("⚠️  Core schema ensure failed: %s", schema_err)

        # ── Eagle Eye tables ──────────────────────────────────────────────
        try:
            from app.services.eagle_eye.ingest import init_schema as _ee_init
            _ee_init()
            logger.info("✅  Eagle Eye schema ensured (idempotent)")
        except Exception as ee_err:
            logger.warning("⚠️  Eagle Eye schema init failed: %s", ee_err)

        # ── Simulator tables (idempotent — must exist before first request) ──
        try:
            from app.services.eagle_eye.simulator import ensure_simulator_tables as _sim_init
            _sim_init()
            logger.info("✅  Simulator schema ensured (idempotent)")
        except Exception as sim_err:
            logger.warning("⚠️  Simulator schema init failed: %s", sim_err)

        # ── Eagle Eye cache warmup: if ratings cache is cold (<50 rows),
        # trigger a full background recompute so the scanner shows all
        # ~141 Kuwait stocks instead of only the on-demand-fetched ones.
        try:
            from app.services.eagle_eye.store import load_all_ratings as _ee_load_ratings
            _ee_cached = _ee_load_ratings()
            if len(_ee_cached) < 50:
                import threading
                from app.services.eagle_eye.ingest import run_nightly_recompute as _ee_recompute
                _ee_warmup = threading.Thread(
                    target=_ee_recompute,
                    kwargs={"dna_refresh": False, "verbose": False},
                    daemon=True,
                    name="ee_startup_warmup",
                )
                _ee_warmup.start()
                logger.info(
                    "🔥  Eagle Eye ratings cache is cold (%d rows) — "
                    "background warmup started for all Kuwait stocks",
                    len(_ee_cached),
                )
            else:
                logger.info("✅  Eagle Eye ratings cache warm (%d stocks)", len(_ee_cached))
        except Exception as _ee_warmup_err:
            logger.warning("⚠️  Eagle Eye startup warmup skipped: %s", _ee_warmup_err)

        # ── Additive migration: portfolios.currency (missing in early prod DBs) ──
        try:
            from app.core.database import add_column_if_missing
            add_column_if_missing("portfolios", "currency", "VARCHAR(10) NOT NULL DEFAULT 'KWD'")
        except Exception as e:
            logger.warning("portfolios.currency migration skipped: %s", e)

        # ── Additive migration: news_articles.content_hash for dedupe fallback ──
        try:
            from app.core.database import add_column_if_missing, exec_sql
            add_column_if_missing("news_articles", "content_hash", "TEXT")
            exec_sql("CREATE INDEX IF NOT EXISTS ix_news_articles_content_hash ON news_articles(content_hash)")
        except Exception as e:
            logger.warning("news_articles.content_hash migration skipped: %s", e)

        # ── Extraction jobs: ensure table + recover stale jobs ───
        try:
            from app.api.v1.fundamental import _ensure_schema as _ensure_fundamental_schema
            from app.api.v1.fundamental import recover_stale_jobs
            _ensure_fundamental_schema()
            recovered = recover_stale_jobs()
            if recovered:
                logger.info("♻️  Recovered %d stale extraction job(s) at startup", recovered)
        except Exception as e:
            logger.warning("Extraction job recovery skipped: %s", e)

        # ── Backfill yf_ticker for existing stocks ───────────────
        try:
            from app.core.database import exec_sql, query_df
            from app.data.stock_lists import resolve_yf_ticker_from_lists
            missing = query_df(
                "SELECT id, symbol, currency FROM stocks WHERE yf_ticker IS NULL OR yf_ticker = ''"
            )
            if missing is not None and not missing.empty:
                updated = 0
                for _, row in missing.iterrows():
                    sym = str(row["symbol"]).strip().upper()
                    ccy = str(row.get("currency") or "KWD").strip().upper()
                    yf = resolve_yf_ticker_from_lists(sym, ccy)
                    if yf:
                        exec_sql("UPDATE stocks SET yf_ticker = ? WHERE id = ?", (yf, row["id"]))
                        updated += 1
                if updated:
                    logger.info("Backfilled yf_ticker for %d existing stocks", updated)
        except Exception as e:
            logger.warning("yf_ticker backfill skipped: %s", e)

        # ── Correct mismatched yf_ticker for known stocks ────────
        # Symbols that exist in both the Kuwait and US reference lists (e.g. KRE)
        # may have been stored with the wrong market ticker. Re-resolve and fix.
        try:
            from app.core.database import exec_sql, query_df
            from app.data.stock_lists import resolve_yf_ticker_from_lists
            all_stocks = query_df(
                "SELECT id, symbol, currency, yf_ticker FROM stocks"
                " WHERE yf_ticker IS NOT NULL AND yf_ticker != ''"
            )
            if all_stocks is not None and not all_stocks.empty:
                corrected = 0
                for _, row in all_stocks.iterrows():
                    sym = str(row["symbol"]).strip().upper()
                    ccy = str(row.get("currency") or "KWD").strip().upper()
                    stored = str(row["yf_ticker"]).strip()
                    resolved = resolve_yf_ticker_from_lists(sym, ccy)
                    if resolved and resolved != stored:
                        exec_sql(
                            "UPDATE stocks SET yf_ticker = ? WHERE id = ?",
                            (resolved, row["id"]),
                        )
                        corrected += 1
                        logger.info(
                            "Corrected yf_ticker for %s (%s): %s → %s",
                            sym, ccy, stored, resolved,
                        )
                if corrected:
                    logger.info("Corrected yf_ticker for %d stock(s)", corrected)
        except Exception as e:
            logger.warning("yf_ticker correction skipped: %s", e)

    # ── Warm FX cache once at startup to avoid first-request cold fetch cost ──
    try:
        from app.services.fx_service import get_usd_kwd_rate
        fx_rate = await asyncio.to_thread(get_usd_kwd_rate)
        logger.info("✅  FX cache warmed at startup (USD/KWD=%s)", round(float(fx_rate), 6))
    except Exception as fx_err:
        logger.warning("⚠️  FX startup warmup skipped: %s", fx_err)

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

    logger.info("🚀  Backend API starting on http://localhost:8004")
    logger.info("📖  Swagger docs at http://localhost:8004/docs")

    # Dev-only: track memory allocations for profiling
    if not settings.is_production:
        tracemalloc.start()
        logger.info("🔍  tracemalloc enabled (development mode)")

    import app.core.ai_metrics  # noqa: F401 — side-effect import registers counters

    yield  # app is running

    # Shutdown
    try:
        from app.core.http_client import close_client
        await close_client()
    except Exception as exc:
        logger.debug("HTTP client shutdown skipped: %s", exc)

    # Cancel any in-flight background technical-batch tasks so they can mark
    # their runs as "failed" before the process exits.  Without this, a
    # graceful restart leaves runs stuck in "running" status, which causes the
    # technical-analysis page to spin indefinitely on the next server start.
    try:
        from app.services.technical_batch_service import _BACKGROUND_TASKS
        pending = list(_BACKGROUND_TASKS)
        if pending:
            logger.info("Cancelling %d in-flight technical batch task(s) for clean shutdown", len(pending))
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    except Exception as exc:
        logger.debug("Background task cleanup skipped: %s", exc)

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


# ── Observability (Prometheus) ─────────────────────────────────────
# Must be called at module level — add_middleware() cannot be called
# after the application has started (i.e. inside lifespan).
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ── Exception handlers ──────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


# ── Security Middleware ──────────────────────────────────────────────
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestSizeLimitMiddleware)
# ── Compression Middleware ────────────────────────────────────────────
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=512)
# ── Observability Middleware ─────────────────────────────────────────
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(LegacyDeprecationMiddleware)
app.add_middleware(AssetCacheMiddleware)

# ── CORS Middleware ──────────────────────────────────────────────────
# NOTE: allow_origins=["*"] + allow_credentials=True is spec-invalid.
# Starlette echoes origin on preflight but returns "*" on actual requests,
# causing browsers to reject credentialed responses with "Network Error".
# Fix: list explicit dev origins so the browser always sees the real origin.
_dev_origins = [
    "http://localhost:8081",   # Expo web
    "http://localhost:8082",   # Expo web (alt port)
    "http://localhost:19006",  # Expo web (alt port)
    "http://localhost:3000",   # dev fallback
    "http://127.0.0.1:8004",  # Local backend
    "http://localhost:8004",   # Local backend (alt)
    "http://192.168.1.5:8081", # LAN mobile browser
    "http://127.0.0.1:8081",
    "http://127.0.0.1:8082",
]

# In production, match any *.ondigitalocean.app subdomain + explicit CORS_ORIGINS
_prod_origin_regex = r"https://.*\.ondigitalocean\.app"

# In development, allow localhost/127.0.0.1 on any port so Expo can
# run on alternate ports (8083, 8084, etc.) without CORS failures.
_dev_origin_regex = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list if settings.is_production else _dev_origins,
    allow_origin_regex=_prod_origin_regex if settings.is_production else _dev_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
    # Chrome Private Network Access: localhost:8081 → 127.0.0.1:8004
    allow_private_network=not settings.is_production,
)


@app.middleware("http")
async def inject_flags(request, call_next):
    response = await call_next(request)
    response.headers["X-Feature-Flags"] = json.dumps(get_flags())
    return response


# ── Routes ───────────────────────────────────────────────────────────

# v1 versioned API (primary)
app.include_router(v1_router)

# Legacy unversioned routes (kept for backward compat — will be removed)
# Guard against accidental `_IncludedRouter` objects to avoid runtime error:
# AttributeError: '_IncludedRouter' object has no attribute 'path'
for _legacy_router in (auth_router_legacy, portfolio_router_legacy, cron_router_legacy):
    if isinstance(_legacy_router, FastAPI):
        logger.warning("Skipping invalid legacy router %r: got FastAPI app instead of APIRouter", _legacy_router)
        continue
    if not hasattr(_legacy_router, "routes"):
        logger.warning("Skipping invalid legacy router %r: missing .routes (likely _IncludedRouter)", _legacy_router)
        continue
    app.include_router(_legacy_router)

_annotate_included_router_paths(app.router.routes)


# ── Health check (no auth) ──────────────────────────────────────────
# Exposed at both /health (legacy) and /api/health (for DO App Platform
# routing where only /api/* is forwarded to this service).

@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "deploy": "2026-03-06-combined-app-spa-fix",
        "db_mode": "postgresql" if settings.use_postgres else "sqlite",
        "db_connected": check_db_exists(),
        "environment": "production" if settings.is_production else "development",
    }


@app.get("/health/tables", tags=["System"])
@app.get("/api/health/tables", tags=["System"])
async def health_tables():
    """Diagnostic: list all tables that exist in the database."""
    from app.core.database import query_df
    try:
        if settings.use_postgres:
            df = query_df(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )
            tables = df["table_name"].tolist() if not df.empty else []
        else:
            df = query_df(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = df["name"].tolist() if not df.empty else []
        return {"status": "ok", "table_count": len(tables), "tables": tables}
    except Exception as e:
        return {"status": "error", "error": str(e)}
