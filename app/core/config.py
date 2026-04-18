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

    # CORS
    CORS_ORIGINS: str = "http://localhost:19006,http://localhost:8081,http://localhost:3000"

    # FX
    FX_CACHE_TTL: int = 3600  # 1 hour cache for USD/KWD rate

    # Cron / Scheduler
    CRON_SECRET_KEY: str = ""           # Required for POST /api/cron/update-prices
    PRICE_UPDATE_HOUR: int = 14         # Hour (24h) in Asia/Kuwait to run daily
    PRICE_UPDATE_MINUTE: int = 0
    PRICE_UPDATE_ENABLED: bool = True   # Set False to disable the built-in scheduler

    # AI / Gemini (optional)
    GEMINI_API_KEY: str = ""            # Google Gemini API key for AI analysis

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
    return Settings()
