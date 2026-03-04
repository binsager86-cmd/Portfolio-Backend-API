# ─────────────────────────────────────────────────────────────────────
# Portfolio Mobile API — Production Dockerfile
# ─────────────────────────────────────────────────────────────────────
# Build:  docker build -t portfolio-api .
# Run:    docker run -p 8002:8002 --env-file .env portfolio-api
# ─────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc / buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install OS-level dependencies ────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libffi-dev curl && \
    rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ──────────────────────────────────────
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ───────────────────────────────────────────
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# ── Health check ─────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8002}/health || exit 1

# ── Default port ─────────────────────────────────────────────────────
EXPOSE 8002

# ── Start server ─────────────────────────────────────────────────────
CMD ["sh", "-c", "gunicorn app.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8002} --timeout 120 --access-logfile -"]
