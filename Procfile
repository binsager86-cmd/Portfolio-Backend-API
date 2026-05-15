release: PLAYWRIGHT_BROWSERS_PATH=.cache/ms-playwright python -m alembic upgrade head
web: PLAYWRIGHT_BROWSERS_PATH=.cache/ms-playwright gunicorn app.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8004} --timeout 300
worker: PLAYWRIGHT_BROWSERS_PATH=.cache/ms-playwright celery -A app.core.celery_app.celery_app worker --concurrency=4 --loglevel=info --queues=default,heavy_ai,polling
