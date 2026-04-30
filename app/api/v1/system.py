"""
System API — app version gating and health metadata.
"""

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(prefix="/system", tags=["system"])


@router.get(
    "/version-check",
    summary="Check minimum required app version",
)
async def version_check():
    """
    Returns version gating information for the mobile app.

    The mobile client compares ``current_version`` against ``min_version``
    and blocks usage when the installed version is below the minimum.
    """
    settings = get_settings()
    return {
        "min_version": settings.MIN_APP_VERSION,
        "latest_version": settings.LATEST_APP_VERSION,
        "update_required": False,
        "update_url": "https://play.google.com/store/apps/details?id=com.yourname.portfolio",
    }
