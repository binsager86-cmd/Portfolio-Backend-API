"""
Application Configuration — loaded from .env file.
All settings are read once at startup and cached.
"""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings

# Resolve project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend-api/
# Prefer .env, fall back to .env.production (deployed repos may lack .env)
_env = BASE_DIR / ".env"
ENV_FILE = _env if _env.exists() else BASE_DIR / ".env.production"


class Settings(BaseSettings):
    """Application settings with .env support."""

    # Environment
    ENVIRONMENT: str = "development"  # "development" | "production"

    # Database — dual-mode: SQLite (dev) or PostgreSQL (prod)
    DATABASE_PATH: str = "../dev_portfolio.db"      # SQLite file (used when DATABASE_URL is empty)
    DATABASE_URL: str = ""                           # PostgreSQL URL — set for production
    # Example: postgresql://user:pass@localhost:5432/portfolio

    # Security
    SECRET_KEY: str = "change_this_to_a_random_string_before_production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 360              # Access tokens last 6 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30        # Long-lived refresh tokens
    BCRYPT_ROUNDS: int = 12                    # bcrypt work factor
    LEGACY_PLAINTEXT_LOGIN: bool = False       # Allow plaintext password fallback (dev migration only)
    ACCOUNT_LOCKOUT_ATTEMPTS: int = 5          # Lock after N failed logins
    ACCOUNT_LOCKOUT_MINUTES: int = 15          # Lockout duration

    # Field-level encryption key (Fernet, for sensitive fields like API keys at rest)
    FIELD_ENCRYPTION_KEY: str = ""             # Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

    # Request limits
    MAX_REQUEST_BODY_BYTES: int = 52_428_800    # 50 MB (for PDF uploads)
    TRUSTED_PROXY_IPS: str = ""                 # Comma-separated proxy IPs allowed to supply X-Forwarded-For

    # CORS
    CORS_ORIGINS: str = "http://localhost:19006,http://localhost:8081,http://localhost:3000"

    # KFH read-only synchronization is deny-by-default and additionally scoped
    # to an explicit test-user allowlist during controlled rollout.
    KFH_AUTO_SYNC_ENABLED: bool = False
    KFH_AUTO_SYNC_TEST_USER_IDS: str = ""
    # Manual localhost-only statement review. This never enables scheduled or
    # automatic KFH synchronization and is rejected in production.
    KFH_LOCAL_TEST_ENABLED: bool = False

    # Redis / shared infrastructure
    REDIS_URL: str = ""                         # Used by Celery and shared rate limiting when configured
    ENABLE_PUBLIC_DOCS: bool = False             # Production docs/OpenAPI exposure opt-in
    ENABLE_PUBLIC_METRICS: bool = False          # Production /metrics exposure opt-in

    # FX
    FX_CACHE_TTL: int = 3600  # 1 hour cache for USD/KWD rate

    # Cron / Scheduler
    CRON_SECRET_KEY: str = ""           # Required for POST /api/cron/update-prices
    PRICE_UPDATE_HOUR: int = 13         # Hour (24h) in Asia/Kuwait to run daily
    PRICE_UPDATE_MINUTE: int = 20
    PRICE_UPDATE_ENABLED: bool = True   # Set False to disable the built-in scheduler
    TECHNICAL_BATCH_ENABLED: bool = True
    TECHNICAL_BATCH_HOUR: int = 14      # Daily technical universe scoring (Asia/Kuwait)
    TECHNICAL_BATCH_MINUTE: int = 5
    TECHNICAL_BATCH_MAX_CONCURRENCY: int = 4
    TECHNICAL_BATCH_SEGMENT: str = "PREMIER"
    STOCKANALYSIS_MAX_WORKERS: int = 4  # bounded concurrency for daily fundamentals scrape
    PUSH_NOTIFICATIONS_ENABLED: bool = True
    FIREBASE_SERVICE_ACCOUNT_FILE: str = ""  # Deployment-only Firebase Admin JSON credential path

    # App version gating
    MIN_APP_VERSION: str = "1.2.0"      # Oldest version allowed to connect
    LATEST_APP_VERSION: str = "1.3.1"   # Shown in /system/version-check

    # AI / Gemini (optional)
    GEMINI_API_KEY: str = ""            # Google Gemini API key for AI analysis
    GOOGLE_CLIENT_IDS: str = ""         # Comma-separated OAuth client IDs allowed for Google Sign-In

    # Market data (Whale Tracker)
    EODHD_API_TOKEN: str = ""           # EODHD token for /api/v1/trade-signals/whale-candles (legacy)
    TICKERCHART_USERNAME: str = ""     # TickerChart Live account (replaces EODHD)
    TICKERCHART_PASSWORD: str = ""     # TickerChart Live password (plaintext; sent base64-encoded)
    TICKERCHART_MARKET_INFO_PATH: str = ""  # Optional full path to MarketInfo.json override
    TICKERCHART_FLATFILES_PATH: str = ""    # Optional full path to TickerChart FlatFiles root

    # Eagle Eye — ML artifact paths (relative to backend-api/ or absolute)
    ML_MATRIX_ROOT: str = "./ml_training_matrix"
    ML_MODEL_ROOT: str = "./ml_models"
    ML_REPORTS_ROOT: str = "./reports"
    EAGLE_EYE_LOG_LEVEL: str = "INFO"

    # Eagle Eye — Phase 3: Shadow runner / band display settings
    ENABLE_ML_DISPLAY: bool = True        # Set False to hide all ML bands site-wide
    ML_SHADOW_HOUR: int = 14              # Hour to run daily shadow scoring (Asia/Kuwait)
    ML_SHADOW_MINUTE: int = 30
    ML_WEEKLY_REVIEW_DAY: str = "sun"     # Day of week for weekly review job
    ALERT_WEBHOOK_URL: str = ""           # Optional webhook URL for auto-disable alerts

    # SMTP / Email (for password reset OTP)
    SMTP_HOST: str = ""                 # e.g. "smtp.gmail.com"
    SMTP_PORT: int = 587                # 587 for STARTTLS, 465 for SSL
    SMTP_USER: str = ""                 # e.g. "you@gmail.com"
    SMTP_PASSWORD: str = ""             # Gmail App Password or SMTP password
    SMTP_FROM_EMAIL: str = ""           # Sender address (defaults to SMTP_USER)
    SMTP_FROM_NAME: str = "Portfolio Tracker"
    SMTP_USE_TLS: bool = True           # True=STARTTLS (587), False=SSL (465)

    # Password reset
    OTP_EXPIRE_MINUTES: int = 10        # OTP code validity window
    OTP_MAX_ATTEMPTS: int = 5           # Max verification attempts per OTP

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def use_postgres(self) -> bool:
        """True when DATABASE_URL is a PostgreSQL connection string."""
        url = (self.DATABASE_URL or "").strip()
        return url.startswith("postgresql://") or url.startswith("postgres://")

    class Config:
        env_file = str(ENV_FILE)
        env_file_encoding = "utf-8"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def google_client_ids_list(self) -> list[str]:
        return [client_id.strip() for client_id in self.GOOGLE_CLIENT_IDS.split(",") if client_id.strip()]

    @property
    def kfh_auto_sync_test_user_ids(self) -> set[int]:
        return {
            int(user_id)
            for user_id in self.KFH_AUTO_SYNC_TEST_USER_IDS.split(",")
            if user_id.strip().isdigit()
        }

    @property
    def database_abs_path(self) -> str:
        """Resolve DATABASE_PATH relative to backend-api/ directory."""
        p = Path(self.DATABASE_PATH)
        if p.is_absolute():
            return str(p)
        return str((BASE_DIR / p).resolve())

    @property
    def sqlalchemy_url(self) -> str:
        """
        Canonical SQLAlchemy connection URL.

        Uses DATABASE_URL (PostgreSQL) when set, otherwise falls back to
        SQLite file from DATABASE_PATH.
        """
        if self.use_postgres:
            # Some platforms still provide postgres://; SQLAlchemy expects postgresql://
            url = self.DATABASE_URL.strip()
            if url.startswith("postgres://"):
                return "postgresql://" + url[len("postgres://"):]
            return url
        return f"sqlite:///{self.database_abs_path}"


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    # [Phase-1 Security] LEGACY_PLAINTEXT_LOGIN must never be enabled in production.
    # This plaintext fallback exists only to ease dev-time password migration and
    # must be removed before any traffic hits real user data.
    if settings.is_production and settings.LEGACY_PLAINTEXT_LOGIN:
        raise RuntimeError(
            "LEGACY_PLAINTEXT_LOGIN=True is forbidden in production. "
            "Set LEGACY_PLAINTEXT_LOGIN=False in your .env before deploying."
        )
    return settings
