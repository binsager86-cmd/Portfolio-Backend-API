import multiprocessing
import os

# Worker Configuration
workers = int(os.getenv("GUNICORN_WORKERS", min(multiprocessing.cpu_count() * 2 + 1, 8)))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
backlog = 2048

# Timeouts & Recycling
timeout = 120
graceful_timeout = 30
keepalive = 5
max_requests = int(os.getenv("MAX_REQUESTS", "2000"))
max_requests_jitter = 200

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Network & Security
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
forwarded_allow_ips = "*"
preload_app = True