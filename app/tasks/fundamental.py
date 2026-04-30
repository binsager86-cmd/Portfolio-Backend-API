import logging

from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="fundamental.extract_pdf", max_retries=3)
def extract_pdf_task(
    self,
    *,
    job_id: int,
    stock_id: int,
    user_id: int,
    pdf_bytes: bytes,
    filename: str,
    model: str,
    force: bool,
    api_key: str,
    existing_codes: list[dict],
):
    """Run the existing synchronous fundamental extraction worker in Celery."""
    from app.api.v1.fundamental import _run_extraction_job_sync

    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "job_id": job_id})
        _run_extraction_job_sync(
            job_id=job_id,
            stock_id=stock_id,
            user_id=user_id,
            pdf_bytes=pdf_bytes,
            filename=filename,
            model=model,
            force=force,
            api_key=api_key,
            existing_codes=existing_codes,
        )
        self.update_state(state="PROGRESS", meta={"progress": 100, "job_id": job_id})
        return {"status": "completed", "job_id": job_id}
    except Exception as exc:
        logger.error("PDF extraction failed for job %s: %s", job_id, exc)
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))
