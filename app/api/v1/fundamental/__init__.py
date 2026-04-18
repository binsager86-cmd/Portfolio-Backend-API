"""
Fundamental Analysis API v1 — package entry point.

Transitional: imports everything from the legacy monolith so existing
``from app.api.v1.fundamental import router`` keeps working.
Sub-modules will be split out incrementally.
"""

from app.api.v1.fundamental_legacy import (  # noqa: F401
    router,
    _ensure_schema,
    recover_stale_jobs,
    _run_extraction_job_sync,
    _update_job,
    _start_heartbeat_thread,
    _log_job,
)

__all__ = [
    "router", "_ensure_schema", "recover_stale_jobs",
    "_run_extraction_job_sync", "_update_job", "_start_heartbeat_thread", "_log_job",
]
