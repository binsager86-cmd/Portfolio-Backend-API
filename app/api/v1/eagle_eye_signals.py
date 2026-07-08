from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_current_user, require_admin
from app.core.database import exec_sql, query_all
from app.core.security import TokenData
from app.schemas.eagle_eye_signals import EngineConfigUpdateRequest, ScanRunRequest
from app.services.eagle_eye.entry_exit_service import get_position_state
from app.services.eagle_eye.market_data_service import ensure_schema, get_config_with_meta, load_ohlcv_csv, update_config
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


def _scan_preview_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "kse"


def _reset_preview_tables() -> None:
    for table in [
        "ee_ohlcv",
        "ee_indicators",
        "ee_symbol_state",
        "ee_signals",
        "ee_ratings",
        "ee_positions",
        "ee_audit_events",
    ]:
        exec_sql(f"DELETE FROM {table}", ())


def _full_watchlist_items() -> list[dict]:
    rows = query_all(
        """
        SELECT s.symbol, s.phase, s.base_high, s.base_low, s.last_score, s.updated_at,
               r.band, r.score,
               i.payload_json,
               ls.trade_date AS latest_signal_trade_date,
               ls.signal_type AS latest_signal_type,
               ls.evidence_json AS latest_signal_evidence_json
        FROM ee_symbol_state s
        LEFT JOIN ee_ratings r
          ON r.symbol = s.symbol AND r.trade_date = (
              SELECT MAX(trade_date) FROM ee_ratings r2 WHERE r2.symbol = s.symbol
          )
        LEFT JOIN ee_indicators i
          ON i.symbol = s.symbol AND i.trade_date = (
              SELECT MAX(trade_date) FROM ee_indicators i2 WHERE i2.symbol = s.symbol
          )
        LEFT JOIN ee_signals ls
          ON ls.id = (
              SELECT id FROM ee_signals s2 WHERE s2.symbol = s.symbol ORDER BY trade_date DESC, id DESC LIMIT 1
          )
        WHERE s.phase IN ('ACCUMULATION', 'BREAKOUT_WATCH')
        ORDER BY COALESCE(r.score, s.last_score, 0) DESC, s.symbol ASC
        """,
        (),
    )

    out: list[dict] = []
    for row in rows or []:
        indicator_evidence = json.loads(str(row.get("payload_json") or "{}"))
        signal_evidence = json.loads(str(row.get("latest_signal_evidence_json") or "{}"))
        out.append(
            {
                "symbol": row.get("symbol"),
                "phase": row.get("phase"),
                "score": row.get("score") if row.get("score") is not None else row.get("last_score"),
                "band": row.get("band"),
                "base_high": row.get("base_high"),
                "base_low": row.get("base_low"),
                "updated_at": row.get("updated_at"),
                "latest_signal": {
                    "trade_date": row.get("latest_signal_trade_date"),
                    "signal_type": row.get("latest_signal_type"),
                    "evidence": signal_evidence,
                },
                "evidence": indicator_evidence,
                "advice": False,
            }
        )
    return out


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


@router.get("/scan-preview")
def api_scan_preview(admin_user: TokenData = Depends(require_admin)):
    ensure_schema()
    data_dir = _scan_preview_dir()
    if not data_dir.exists() or not data_dir.is_dir():
        return {
            "status": "ok",
            "data": {
                "items": [],
                "loaded_symbols": [],
                "message": f"Directory not found: {data_dir}",
                "advice": False,
            },
        }

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        return {
            "status": "ok",
            "data": {
                "items": [],
                "loaded_symbols": [],
                "message": f"No CSV files found under {data_dir}",
                "advice": False,
            },
        }

    _reset_preview_tables()
    loaded: list[str] = []
    for path in csv_files:
        symbol = path.stem.upper()
        load_ohlcv_csv(str(path), symbol)
        loaded.append(symbol)

    run = run_eod_pipeline(source="scan-preview", actor=admin_user)
    return {
        "status": "ok",
        "data": {
            "items": _full_watchlist_items(),
            "loaded_symbols": loaded,
            "run": run.get("data", run),
            "advice": False,
        },
    }


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
