from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EngineConfigUpdateRequest(BaseModel):
    target_area: str = Field(..., min_length=2, max_length=80)
    change_request_id: int
    values: dict[str, Any]


class ScanRunRequest(BaseModel):
    source: str = "manual"


class Pagination(BaseModel):
    total: int
    limit: int
    offset: int
