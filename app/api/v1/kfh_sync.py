"""KFH read-only synchronization staging and confirmation endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.security import TokenData
from app.schemas.kfh_sync import KfhConfirmRequest, KfhPreviewRequest
from app.services.kfh_sync_service import (
    confirm_batch,
    create_preview,
    disconnect_connection,
    get_connection,
    get_preview,
)

router = APIRouter(prefix="/kfh-sync", tags=["KFH Sync"])
LOCAL_SAHAM_ORIGIN = "http://localhost:8081"


def require_kfh_auto_sync(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> TokenData:
    settings = get_settings()
    controlled_rollout = (
        settings.KFH_AUTO_SYNC_ENABLED
        and current_user.user_id in settings.kfh_auto_sync_test_user_ids
    )
    manual_local_test = (
        settings.ENVIRONMENT.lower() != "production"
        and settings.KFH_LOCAL_TEST_ENABLED
        and request.headers.get("origin") == LOCAL_SAHAM_ORIGIN
    )
    if not controlled_rollout and not manual_local_test:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="KFH sync is unavailable")
    return current_user


@router.get("/connection")
async def get_kfh_connection(
    current_user: TokenData = Depends(require_kfh_auto_sync),
):
    """Return masked KFH connection metadata; credentials and sessions are never returned."""
    return {"status": "ok", "data": get_connection(current_user.user_id)}


@router.delete("/connection/{connection_id}")
async def disconnect_kfh_connection(
    connection_id: int,
    current_user: TokenData = Depends(require_kfh_auto_sync),
):
    """Mark a KFH connection disconnected without exposing or deleting financial history."""
    return {
        "status": "ok",
        "data": disconnect_connection(current_user.user_id, connection_id),
    }


@router.post("/batches/preview", status_code=201)
async def preview_kfh_batch(
    body: KfhPreviewRequest,
    current_user: TokenData = Depends(require_kfh_auto_sync),
):
    """Persist raw broker records and return a non-mutating classified preview."""
    return {"status": "ok", "data": create_preview(current_user.user_id, body)}


@router.get("/batches/{batch_id}")
async def get_kfh_batch(
    batch_id: int,
    current_user: TokenData = Depends(require_kfh_auto_sync),
):
    """Return a staged or confirmed KFH import batch owned by the user."""
    return {"status": "ok", "data": get_preview(current_user.user_id, batch_id)}


@router.post("/batches/{batch_id}/confirm")
async def confirm_kfh_batch(
    batch_id: int,
    body: KfhConfirmRequest,
    current_user: TokenData = Depends(require_kfh_auto_sync),
):
    """Atomically commit explicitly selected NEW transactions."""
    return {
        "status": "ok",
        "data": confirm_batch(
            current_user.user_id,
            batch_id,
            body.selected_item_ids,
            update_cash_balance=body.update_cash_balance,
        ),
    }
