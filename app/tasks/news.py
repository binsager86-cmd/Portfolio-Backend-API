import asyncio
import logging

from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="news.fetch_and_ingest")
def fetch_and_ingest_news_task(lang: str = "en") -> dict:
    """Fetch Boursa announcements and persist them using existing news pipeline helpers."""
    from app.api.v1.news import (
        _fetch_all_boursa_sources,
        _map_item,
        _persist_articles,
        news_cache,
    )
    from app.core.database import SessionLocal

    boursa_lang = "A" if lang == "ar" else "E"
    raw_items = asyncio.run(_fetch_all_boursa_sources(boursa_lang))
    mapped_items = [_map_item(item, boursa_lang) for item in raw_items]

    db = SessionLocal()
    try:
        ingested = _persist_articles(db, mapped_items)
    finally:
        db.close()

    news_cache.clear()
    logger.info("Celery news ingestion complete: %d new articles", ingested)
    return {"ingested": ingested, "status": "success"}
