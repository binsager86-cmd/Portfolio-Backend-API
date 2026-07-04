from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_current_user, require_admin
from app.core.security import TokenData
from app.schemas.eagle_eye_signals import EngineConfigUpdateRequest, ScanRunRequest
from app.services.eagle_eye.entry_exit_service import get_position_state
from app.services.eagle_eye.market_data_service import ensure_schema, get_config_with_meta, update_config
from app.services.eagle_eye.rating_service import load_rating_history
from app.services.eagle_eye.scanner_service import get_symbol_state, list_watchlist
from app.services.eagle_eye.scheduler_service import (
    get_signal_detail,
    performance_summary,
    query_signals,
    run_eod_pipeline,
)

router = APIRouter(prefix="/eagle-eye/signals", tags=["Eagle Eye Signals"])


def _viewer(user: TokenData) -> dict:
    return {
        "user_id": user.user_id,
        "username": user.username,
        "is_admin": user.is_admin,
    }


@router.get("/watchlist")
def api_watchlist(current_user: TokenData = Depends(get_current_user)):
    ensure_schema()
    items = list_watchlist()
    return {
        "status": "ok",
        "data": {
            "items": items,
            "viewer": _viewer(current_user),
            "advice": False,
        },
    }


@router.get("/signals")
def api_signals(
    signal_type: Optional[str] = Query(default=None),
    symbol: Optional[str] = Query(default=None),
    since: Optional[int] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: TokenData = Depends(get_current_user),
):
    ensure_schema()
    items, total = query_signals(signal_type=signal_type, symbol=symbol, since=since, limit=limit, offset=offset)
    return {
        "status": "ok",
        "data": {
            "items": items,
            "pagination": {"total": total, "limit": limit, "offset": offset},
            "viewer": _viewer(current_user),
            "advice": False,
        },
    }


@router.get("/signals/{signal_id}")
def api_signal_detail(signal_id: int, current_user: TokenData = Depends(get_current_user)):
    ensure_schema()
    item = get_signal_detail(signal_id)
    if not item:
        raise HTTPException(status_code=404, detail="Signal not found")
    return {
        "status": "ok",
        "data": {
            **item,
            "viewer": _viewer(current_user),
            "advice": False,
        },
    }


@router.get("/ratings/{symbol}")
def api_rating_history(symbol: str, current_user: TokenData = Depends(get_current_user)):
    ensure_schema()
    history = load_rating_history(symbol.upper())
    return {
        "status": "ok",
        "data": {
            "symbol": symbol.upper(),
            "history": history,
            "viewer": _viewer(current_user),
            "advice": False,
        },
    }


@router.get("/state/{symbol}")
def api_symbol_state(symbol: str, current_user: TokenData = Depends(get_current_user)):
    ensure_schema()
    state = get_symbol_state(symbol.upper())
    return {
        "status": "ok",
        "data": {
            "symbol": symbol.upper(),
            "state": state,
            "position": get_position_state(symbol.upper()),
            "viewer": _viewer(current_user),
            "advice": False,
        },
    }


@router.post("/scan/run")
def api_scan_run(
    payload: ScanRunRequest,
    admin_user: TokenData = Depends(require_admin),
):
    ensure_schema()
    result = run_eod_pipeline(source=payload.source or "manual", actor=admin_user)
    return result


@router.get("/config")
def api_get_config(current_user: TokenData = Depends(get_current_user)):
    ensure_schema()
    return {
        "status": "ok",
        "data": {
            **get_config_with_meta(),
            "viewer": _viewer(current_user),
            "advice": False,
        },
    }


@router.put("/config")
def api_put_config(
    payload: EngineConfigUpdateRequest,
    admin_user: TokenData = Depends(require_admin),
):
    ensure_schema()
    updated = update_config(
        values=payload.values,
        target_area=payload.target_area,
        change_request_id=payload.change_request_id,
        actor=admin_user,
    )
    return {"status": "ok", "data": {**updated, "advice": False}}


@router.get("/performance")
def api_performance(current_user: TokenData = Depends(get_current_user)):
    ensure_schema()
    perf = performance_summary()
    return {
        "status": "ok",
        "data": {
            **perf,
            "viewer": _viewer(current_user),
            "advice": False,
        },
    }
