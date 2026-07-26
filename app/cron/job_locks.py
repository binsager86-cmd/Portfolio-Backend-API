"""Lightweight in-process locks for cron/scheduler job orchestration."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _get_lock(name: str) -> threading.Lock:
    with _registry_lock:
        lock = _locks.get(name)
        if lock is None:
            lock = threading.Lock()
            _locks[name] = lock
        return lock


def run_with_job_lock(name: str, fn: Callable[[], T]) -> T:
    """Run a job only if no same-named job is already active in this process."""
    lock = _get_lock(name)
    if not lock.acquire(blocking=False):
        raise RuntimeError(f"Job '{name}' is already running")
    try:
        return fn()
    finally:
        lock.release()
