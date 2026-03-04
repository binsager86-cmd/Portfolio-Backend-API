"""
Alembic env.py — migration environment configuration.

Reads the database URL from app settings and discovers all
SQLAlchemy models via app.models (which imports them all).

Supports both SQLite (dev) and PostgreSQL (prod).
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Import our models so `target_metadata` sees all tables
from app.core.database import Base
import app.models  # noqa: F401  — force model registration

from app.core.config import get_settings

# ── Alembic Config ───────────────────────────────────────────────────
config = context.config

# Override sqlalchemy.url from app settings (supports both SQLite & PostgreSQL)
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.sqlalchemy_url)

# Setup Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Detect if we're running against SQLite (needs batch mode for ALTER TABLE)
_is_sqlite = not settings.use_postgres


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — generates SQL script."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=_is_sqlite,  # Required for SQLite ALTER TABLE
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — connects to DB."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=_is_sqlite,  # Required for SQLite ALTER TABLE
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
