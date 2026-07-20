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
    pdf_upload_id: int,
    model: str,
    force: bool,
    api_key: str,
    existing_codes: list[dict],
):
    """Run the existing synchronous fundamental extraction worker in Celery."""
    from app.api.v1.fundamental_legacy import _PDF_UPLOAD_DIR, _run_extraction_job_sync
    from app.core.database import query_one

    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "job_id": job_id})
        upload = query_one(
            "SELECT filename, original_name FROM pdf_uploads WHERE id = ? AND stock_id = ? AND user_id = ?",
            (pdf_upload_id, stock_id, user_id),
        )
        if not upload:
            raise RuntimeError(f"PDF upload {pdf_upload_id} not found for extraction job {job_id}")
        pdf_path = _PDF_UPLOAD_DIR / str(stock_id) / upload["filename"]
        pdf_bytes = pdf_path.read_bytes()
        filename = upload["original_name"] or upload["filename"]
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
