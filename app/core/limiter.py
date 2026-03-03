"""
Rate limiter — shared slowapi instance.

Kept in its own module to avoid circular imports between main.py and routers.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["120/minute"],          # global default
    storage_uri="memory://",
)
