"""
Rate limiter — shared slowapi instance.

Kept in its own module to avoid circular imports between main.py and routers.
"""

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

_enabled = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() != "false"

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["120/minute"],          # global default
    storage_uri="memory://",
    enabled=_enabled,
)
