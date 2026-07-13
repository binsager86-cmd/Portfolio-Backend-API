from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EngineConfigUpdateRequest(BaseModel):
    target_area: str = Field(..., min_length=2, max_length=80)
    change_request_id: int
    values: dict[str, Any]


class ScanRunRequest(BaseModel):
    source: str = "manual"


class TickerChartIngestRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    start: str | None = None
    end: str | None = None
    source: str = "manual"


class DataQualityClearRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=24)
    change_request_id: int


class PipelineModeUpdateRequest(BaseModel):
    mode: str = Field(..., min_length=4, max_length=8)
    change_request_id: int


class Pagination(BaseModel):
    total: int
    limit: int
    offset: int
