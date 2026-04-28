"""
Notifications API v1 — push token registration and notification management.

Endpoints:
  POST /register-token   — Register an Expo push token for the current user
  DELETE /unregister-token — Remove a push token
  GET  /status            — Check news polling status
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.security import TokenData
from app.models.push_token import PushToken
from app.services import notification_prefs as prefs_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notifications", tags=["Notifications"])


# ── Schemas ──────────────────────────────────────────────────────────

class RegisterTokenRequest(BaseModel):
    token: str
    platform: str = "unknown"  # ios / android / web


class RegisterTokenResponse(BaseModel):
    ok: bool
    message: str


class NotificationPrefs(BaseModel):
    newsNotifications: Optional[bool] = None
    portfolioUpdates: Optional[bool] = None
    priceAlerts: Optional[bool] = None
    dailyPriceUpdates: Optional[bool] = None


def _is_valid_expo_push_token(token: str) -> bool:
    """Accept current and legacy Expo push token formats."""
    if not token:
        return False
    t = token.strip()
    return t.startswith("ExponentPushToken[") or t.startswith("ExpoPushToken[")


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/register-token", response_model=RegisterTokenResponse)
async def register_push_token(
    body: RegisterTokenRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Register an Expo push token for receiving push notifications."""
    token = (body.token or "").strip()
    if not _is_valid_expo_push_token(token):
        raise HTTPException(status_code=400, detail="Invalid Expo push token format")

    # Check if token already exists
    existing = db.query(PushToken).filter(PushToken.token == token).first()
    if existing:
        # Update user_id and platform if changed
        existing.user_id = current_user.user_id
        existing.platform = body.platform
        existing.updated_at = datetime.utcnow()
        db.commit()
        return RegisterTokenResponse(ok=True, message="Token updated")

    # Create new token record
    push_token = PushToken(
        user_id=current_user.user_id,
        token=token,
        platform=body.platform,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(push_token)
    db.commit()
    logger.info("Push token registered for user %d (%s)", current_user.user_id, body.platform)
    return RegisterTokenResponse(ok=True, message="Token registered")


@router.delete("/unregister-token")
async def unregister_push_token(
    body: RegisterTokenRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove a push token (e.g., on logout)."""
    deleted = db.query(PushToken).filter(
        PushToken.token == body.token,
        PushToken.user_id == current_user.user_id,
    ).delete()
    db.commit()
    return {"ok": True, "deleted": deleted}


@router.get("/preferences")
async def get_notification_preferences(
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the current user's notification preferences (defaults if none set)."""
    return prefs_service.get_prefs(db, current_user.user_id)


@router.put("/preferences")
async def update_notification_preferences(
    body: NotificationPrefs,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upsert the current user's notification preferences (partial allowed)."""
    partial = {k: v for k, v in body.dict().items() if v is not None}
    return prefs_service.set_prefs(db, current_user.user_id, partial)


@router.get("/status")
async def notification_status(
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Check push notification status and poller health for the current user."""
    token_count = db.query(PushToken).filter(
        PushToken.user_id == current_user.user_id
    ).count()

    from app.cron.news_poller import get_poller_status
    poller = get_poller_status()

    return {
        "push_tokens_registered": token_count,
        "poller": poller,
    }
