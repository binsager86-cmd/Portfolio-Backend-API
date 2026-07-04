"""Eagle Eye audit/change-management API endpoints."""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_current_user, require_admin
from app.core.security import TokenData
from app.schemas.eagle_eye_audit import (
    AuditEventCreate,
    ChangeRequestCreate,
    ChangeReviewRequest,
    ChangeTransitionRequest,
)
from app.services.eagle_eye.audit_service import (
    ALLOWED_TRANSITIONS,
    AUDIT_STATUSES,
    change_status,
    create_change_request,
    create_event,
    get_change_request,
    get_design,
    get_summary,
    list_change_requests,
    list_events,
)

router = APIRouter(prefix="/eagle-eye/audit", tags=["Eagle Eye Audit"])


def _viewer(user: TokenData) -> dict:
    return {
        "user_id": user.user_id,
        "username": user.username,
        "is_admin": user.is_admin,
    }


@router.get("/design")
def api_get_design(current_user: TokenData = Depends(get_current_user)):
    return {
        "status": "ok",
        "data": {
            **get_design(),
            "lifecycle": sorted(AUDIT_STATUSES),
            "allowed_transitions": {k: sorted(v) for k, v in ALLOWED_TRANSITIONS.items()},
            "viewer": _viewer(current_user),
        },
    }


@router.post("/events")
def api_create_event(payload: AuditEventCreate, current_user: TokenData = Depends(get_current_user)):
    event = create_event(payload.model_dump(), current_user)
    return {"status": "ok", "data": event}


@router.get("/events")
def api_list_events(
    action: Optional[str] = Query(default=None),
    entity_type: Optional[str] = Query(default=None),
    risk_level: Optional[Literal["low", "medium", "high", "critical"]] = Query(default=None),
    since: Optional[int] = Query(default=None, description="Unix timestamp lower bound"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: TokenData = Depends(get_current_user),
):
    items, total = list_events(
        action=action,
        entity_type=entity_type,
        risk_level=risk_level,
        since=since,
        limit=limit,
        offset=offset,
    )
    return {
        "status": "ok",
        "data": {
            "items": items,
            "pagination": {"total": total, "limit": limit, "offset": offset},
            "viewer": _viewer(current_user),
        },
    }


@router.post("/change-requests")
def api_create_change_request(payload: ChangeRequestCreate, current_user: TokenData = Depends(get_current_user)):
    request = create_change_request(payload.model_dump(), current_user)
    return {"status": "ok", "data": request}


@router.get("/change-requests")
def api_list_change_requests(
    status: Optional[str] = Query(default=None),
    target_area: Optional[str] = Query(default=None),
    requested_by_user_id: Optional[int] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: TokenData = Depends(get_current_user),
):
    items, total = list_change_requests(
        status=status,
        target_area=target_area,
        requested_by_user_id=requested_by_user_id,
        limit=limit,
        offset=offset,
    )
    return {
        "status": "ok",
        "data": {
            "items": items,
            "pagination": {"total": total, "limit": limit, "offset": offset},
            "viewer": _viewer(current_user),
        },
    }


@router.get("/change-requests/{request_id}")
def api_get_change_request(request_id: int, current_user: TokenData = Depends(get_current_user)):
    request, history = get_change_request(request_id)
    return {
        "status": "ok",
        "data": {
            "request": request,
            "history": history,
            "viewer": _viewer(current_user),
        },
    }


@router.post("/change-requests/{request_id}/review")
def api_review_change_request(
    request_id: int,
    payload: ChangeReviewRequest,
    admin_user: TokenData = Depends(require_admin),
):
    updated = change_status(
        request_id=request_id,
        actor=admin_user,
        new_status=payload.decision,
        note=payload.review_notes,
        set_reviewer=True,
    )
    return {"status": "ok", "data": updated}


@router.post("/change-requests/{request_id}/transition")
def api_transition_change_request(
    request_id: int,
    payload: ChangeTransitionRequest,
    current_user: TokenData = Depends(get_current_user),
):
    request, _ = get_change_request(request_id)

    if not current_user.is_admin and int(request.get("requested_by_user_id") or 0) != current_user.user_id:
        raise HTTPException(status_code=403, detail="Only the requester or an admin may transition this request")

    target = payload.new_status

    if target in {"implemented", "cancelled"} and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required for this transition")

    updated = change_status(
        request_id=request_id,
        actor=current_user,
        new_status=target,
        note=payload.note,
    )
    return {"status": "ok", "data": updated}


@router.get("/summary")
def api_get_summary(
    days: int = Query(default=30, ge=1, le=365),
    current_user: TokenData = Depends(get_current_user),
):
    return {
        "status": "ok",
        "data": {
            **get_summary(days),
            "viewer": _viewer(current_user),
        },
    }