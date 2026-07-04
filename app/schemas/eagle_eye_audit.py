"""Eagle Eye audit/change-management schemas."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class AuditEventCreate(BaseModel):
    action: str = Field(..., min_length=2, max_length=120)
    entity_type: str = Field(..., min_length=2, max_length=80)
    entity_id: Optional[str] = Field(default=None, max_length=120)
    change_type: Literal["operation", "config", "model", "workflow", "data"] = "operation"
    before_state: Optional[dict[str, Any]] = None
    after_state: Optional[dict[str, Any]] = None
    rationale: Optional[str] = Field(default=None, max_length=2000)
    risk_level: Literal["low", "medium", "high", "critical"] = "low"
    trace_id: Optional[str] = Field(default=None, max_length=120)
    source: Literal["api", "scheduler", "manual", "system"] = "api"
    metadata: Optional[dict[str, Any]] = None
    concept_version: Optional[str] = Field(default=None, max_length=40)
    requires_follow_up: bool = False


class ChangeRequestCreate(BaseModel):
    title: str = Field(..., min_length=4, max_length=180)
    description: str = Field(..., min_length=8, max_length=5000)
    target_area: Literal[
        "scanner",
        "rating_engine",
        "entry_exit",
        "risk_management",
        "ml_overlay",
        "scheduler",
        "api_contract",
        "data_pipeline",
        "other",
    ]
    change_category: Literal["bugfix", "enhancement", "policy", "experiment", "breaking"]
    proposed_payload: Optional[dict[str, Any]] = None
    status: Literal["draft", "proposed"] = "draft"
    effective_from: Optional[int] = None
    effective_to: Optional[int] = None
    supersedes_request_id: Optional[int] = None


class ChangeReviewRequest(BaseModel):
    decision: Literal["approved", "rejected", "needs_changes"]
    review_notes: str = Field(..., min_length=3, max_length=3000)


class ChangeTransitionRequest(BaseModel):
    new_status: Literal["proposed", "cancelled", "implemented"]
    note: Optional[str] = Field(default=None, max_length=1000)
