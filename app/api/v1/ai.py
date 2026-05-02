"""
AI Analyst API v1 — Gemini-powered portfolio analysis.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["AI Analyst"])


class AIAnalysisRequest(BaseModel):
    """Request body for AI analysis."""
    prompt: Optional[str] = None
    include_holdings: bool = True
    include_transactions: bool = False
    include_performance: bool = True
    language: str = "en"  # en | ar


@router.post("/analyze")
async def analyze_portfolio(
    body: AIAnalysisRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """
    AI-powered portfolio analysis using Google Gemini.

    Gathers portfolio data and sends it to Gemini for analysis.
    Supports English and Arabic responses.
    """
    from app.services.ai_service import analyze_portfolio as ai_analyze

    try:
        result = await ai_analyze(
            user_id=current_user.user_id,
            prompt=body.prompt,
            include_holdings=body.include_holdings,
            include_transactions=body.include_transactions,
            include_performance=body.include_performance,
            language=body.language,
        )
    except ValueError as e:
        raise ServiceUnavailableError(str(e))

    return {"status": "ok", "data": result}


@router.get("/status")
async def ai_status(current_user: TokenData = Depends(get_current_user)):
    """Check if the AI service is configured and available."""
    from app.core.config import get_settings
    settings = get_settings()

    # Check per-user key first, then server-wide key
    has_key = bool(settings.GEMINI_API_KEY)
    try:
        from app.core.database import query_one, add_column_if_missing
        # Ensure column exists (additive migration) before querying
        add_column_if_missing("users", "gemini_api_key", "TEXT")
        row = query_one(
            "SELECT gemini_api_key FROM users WHERE id = ?",
            (current_user.user_id,),
        )
        if row and row[0]:
            has_key = True
    except Exception as exc:
        logger.warning("AI status check error: %s", exc)

    return {
        "status": "ok",
        "data": {
            "configured": has_key,
            "model": "gemini-2.5-flash",
        },
    }


# ── Whale Radar dedicated chat (deep-thinking) ────────────────────────


class WhaleChatRequest(BaseModel):
    """Single-turn prompt for the Whale Radar AI analyst chat."""
    prompt: str


@router.post("/whale-chat")
async def whale_chat(
    body: WhaleChatRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Dedicated Gemini endpoint for Whale Radar conversations.

    Uses ``gemini-2.5-pro`` with thinking enabled (no budget cap) for
    in-depth, multi-paragraph reasoning. Falls back to flash if pro
    is unavailable in the caller's region.
    """
    from app.services.ai_service import whale_chat as ai_whale_chat

    try:
        result = await ai_whale_chat(
            user_id=current_user.user_id,
            prompt=body.prompt,
        )
    except ValueError as e:
        raise ServiceUnavailableError(str(e))

    return {"status": "ok", "data": result}
