"""
Users API v1 — per-user UI preferences (expertise level, language, flags).

Notification toggles live under /notifications/preferences; this endpoint
covers everything else stored in mobile-app's userPrefsStore.
"""

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.security import TokenData
from app.services import user_prefs as prefs_service

router = APIRouter(prefix="/users", tags=["Users"])


class UserPrefsPayload(BaseModel):
    expertiseLevel: Optional[str] = None
    language: Optional[str] = None
    showAdvancedMetrics: Optional[bool] = None
    enableShariaFilter: Optional[bool] = None
    dividendFocus: Optional[bool] = None


@router.get("/me/preferences")
async def get_user_preferences(
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the current user's UI preferences (defaults if none set)."""
    return prefs_service.get_prefs(db, current_user.user_id)


@router.put("/me/preferences")
async def update_user_preferences(
    body: UserPrefsPayload,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upsert the current user's UI preferences (partial allowed)."""
    partial = {k: v for k, v in body.dict().items() if v is not None}
    return prefs_service.set_prefs(db, current_user.user_id, partial)
