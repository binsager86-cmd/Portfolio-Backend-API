import os

from celery import Celery
from kombu import Queue

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "portfolio_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks.fundamental", "app.tasks.news", "app.tasks.prices"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kuwait",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_default_queue="default",
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("heavy_ai", routing_key="heavy_ai"),
        Queue("polling", routing_key="polling"),
    ),
    task_routes={
        "app.tasks.fundamental.*": {"queue": "heavy_ai"},
        "app.tasks.news.*": {"queue": "polling"},
        "app.tasks.prices.*": {"queue": "default"},
    },
    task_annotations={
        "*": {"rate_limit": "10/m"},
        "app.tasks.fundamental.*": {"rate_limit": "2/m"},
    },
)
