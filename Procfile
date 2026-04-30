# [B-3] Run Alembic migrations before starting the web worker
release: python -m alembic upgrade head
web: gunicorn app.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8004} --timeout 300
worker: celery -A app.core.celery_app.celery_app worker --concurrency=4 --loglevel=info --queues=default,heavy_ai,polling
