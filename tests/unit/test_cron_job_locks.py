import pytest

from app.cron.job_locks import run_with_job_lock


def test_job_lock_rejects_overlapping_run():
    def inner():
        with pytest.raises(RuntimeError, match="already running"):
            run_with_job_lock("price-refresh", lambda: "second")
        return "first"

    assert run_with_job_lock("price-refresh", inner) == "first"


def test_job_lock_releases_after_failure():
    def failing():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_with_job_lock("price-refresh-after-error", failing)

    assert run_with_job_lock("price-refresh-after-error", lambda: "ok") == "ok"
