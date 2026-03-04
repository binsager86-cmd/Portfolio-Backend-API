"""
Custom JSON response that safely handles NaN, Infinity, and -Infinity
by converting them to None (JSON null).

Python's json module rejects these values by default because they
are not valid JSON.  Pandas DataFrames and financial calculations
frequently produce them (division by zero, missing data, etc.).
"""

import json
import math
from typing import Any

from starlette.responses import JSONResponse as _StarletteJSONResponse


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN / ±Infinity floats with None."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(item) for item in obj]
    return obj


class SafeJSONResponse(_StarletteJSONResponse):
    """JSONResponse subclass that sanitises NaN / Inf before encoding."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            _sanitize(content),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
