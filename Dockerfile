# Production image with Playwright browsers + OS dependencies preinstalled
FROM mcr.microsoft.com/playwright/python:v1.59.0-jammy
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8004 \
  PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
  EAGLE_EYE_ML_MODELS_DIR=/data/ml_models \
  EAGLE_EYE_ML_FEATURE_STORE_DIR=/data/ml_feature_store

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m playwright install chromium

# Copy app code & set permissions
COPY . .
RUN mkdir -p /data/ml_models /data/ml_feature_store \
    && chown -R pwuser:pwuser /data \
    && chown -R pwuser:pwuser /app \
    && chmod -R 555 /app \
    && chmod -R 755 /data

USER pwuser
EXPOSE 8004

# Healthcheck (FastAPI /health endpoint)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.getenv('PORT', '8004')}/health\")"

CMD ["sh", "-c", "python -m alembic upgrade head && gunicorn app.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8004} --timeout 300"]
