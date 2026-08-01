from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from time import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.eagle_eye_v2.simulator.constants import INITIAL_CAPITAL_KWD, MANIFEST_PATH, SIMULATOR_ROOT
from app.services.eagle_eye_v2.simulator.projection import SQL_MAP_POSTGRES, table_name

router = APIRouter(prefix="/simulator", tags=["Simulator v2"])

BOOKS = ("BUY", "WATCHLIST")
CACHE_SECONDS = 60
DAY_ZERO_INVENTORY_PATH = SIMULATOR_ROOT / "day_zero_state_inventory.json"
SQL_MAP = SQL_MAP_POSTGRES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _cache_response(response: Response) -> None:
    response.headers["Cache-Control"] = f"private, max-age={CACHE_SECONDS}"


def _book(value: str) -> str:
    book = value.upper()
    if book not in BOOKS:
        raise HTTPException(status_code=404, detail="unknown simulator book")
    return book


@lru_cache(maxsize=1)
def _day_zero_inventory_cached(mtime_ns: int) -> dict[str, Any]:
    _ = mtime_ns
    if not DAY_ZERO_INVENTORY_PATH.exists():
        return {"symbols": {}}
    return json.loads(DAY_ZERO_INVENTORY_PATH.read_text(encoding="utf-8"))


def _day_zero_inventory() -> dict[str, Any]:
    mtime_ns = DAY_ZERO_INVENTORY_PATH.stat().st_mtime_ns if DAY_ZERO_INVENTORY_PATH.exists() else 0
    return _day_zero_inventory_cached(mtime_ns)


def _verify_simulator_seals() -> dict[str, Any]:
    manifest_path = Path(os.environ.get("SIMULATOR_MANIFEST_PATH", str(MANIFEST_PATH)))
    started = time()
    failures: list[dict[str, str]] = []
    code_entries = 0
    if not manifest_path.exists():
        return {
            "pass": False,
            "checked_at": _checked_at(),
            "duration_ms": int((time() - started) * 1000),
            "code_entries": 0,
            "failures": [{"path": str(manifest_path), "reason": "manifest missing"}],
        }
    archive_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest.get("simulator_artifacts", []):
        rel = str(entry.get("archive_relative_path", ""))
        if not rel.startswith("simulator/code_genesis/"):
            continue
        code_entries += 1
        path = archive_root / rel
        expected = str(entry.get("sha256", ""))
        if not path.exists():
            failures.append({"path": rel, "reason": "missing"})
            continue
        actual = _sha256(path)
        if actual != expected:
            failures.append({"path": rel, "reason": f"sha256 mismatch: {actual}"})
    return {
        "pass": len(failures) == 0,
        "checked_at": _checked_at(),
        "duration_ms": int((time() - started) * 1000),
        "code_entries": code_entries,
        "failures": failures,
    }


def _checked_at() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _rows(db: Session, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in db.execute(text(sql), params or {}).mappings().all()]
    except Exception as exc:
        raise HTTPException(status_code=503, detail="simulator projection is not available") from exc


def _row(db: Session, sql: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    rows = _rows(db, sql, params)
    return rows[0] if rows else None


@router.get("/portfolios")
def get_portfolios(response: Response, db: Session = Depends(get_db)) -> dict[str, Any]:
    _cache_response(response)
    return {
        "portfolios": _rows(
            db,
            f"""
            SELECT book, nav_kwd, cash_kwd, invested_kwd, open_position_count,
                   total_pnl_kwd, change_since_inception_pct, inception_date
            FROM {table_name('sim_portfolios')} ORDER BY book
            """,
        ),
        "sql_key": "GET /api/v2/simulator/portfolios",
    }


@router.get("/portfolios/{book}/positions")
def get_positions(book: str, response: Response, db: Session = Depends(get_db)) -> dict[str, Any]:
    book = _book(book)
    _cache_response(response)
    positions = _rows(
        db,
        f"""
        SELECT symbol, entry_date, entry_price, entry_reason, sessions_held, last_close,
               unrealized_pnl_pct, unrealized_pnl_kwd, current_lifecycle, avoid_tier
        FROM {table_name('sim_positions')} WHERE book = :book ORDER BY symbol
        """,
        {"book": book},
    )
    return {"book": book, "positions": positions, "sql_key": "GET /api/v2/simulator/portfolios/{book}/positions"}


@router.get("/portfolios/{book}/nav")
def get_nav(book: str, response: Response, days: int = Query(60, ge=1, le=1000), db: Session = Depends(get_db)) -> dict[str, Any]:
    book = _book(book)
    _cache_response(response)
    rows = _rows(
        db,
        f"""
        SELECT session, nav_kwd, cash_kwd, invested_kwd
        FROM {table_name('sim_nav_daily')} WHERE book = :book ORDER BY session DESC LIMIT :days
        """,
        {"book": book, "days": days},
    )
    series = list(reversed(rows))
    if not series:
        series = [{"session": None, "nav_kwd": INITIAL_CAPITAL_KWD, "cash_kwd": INITIAL_CAPITAL_KWD, "invested_kwd": 0.0}]
    return {"book": book, "series": series, "sql_key": "GET /api/v2/simulator/portfolios/{book}/nav"}


@router.get("/transactions")
def get_transactions(
    response: Response,
    book: str | None = Query(None),
    symbol: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _cache_response(response)
    normalized_book = _book(book) if book else None
    normalized_symbol = symbol.upper() if symbol else None
    rows = _rows(
        db,
        f"""
        SELECT id, created_at, portfolio, transaction_type, symbol, quantity, price,
               gross_value_kwd, commission_kwd, net_cash_delta_kwd, decision_session,
               fill_session, source_event_id, reason, status, voids_transaction_id,
               suspension_gap_sessions, state_snapshot_json
        FROM {table_name('sim_transactions')}
        WHERE (:book IS NULL OR portfolio = :book) AND (:symbol IS NULL OR symbol = :symbol)
        ORDER BY id DESC LIMIT :limit
        """,
        {"book": normalized_book, "symbol": normalized_symbol, "limit": limit},
    )
    return {"transactions": rows, "sql_key": "GET /api/v2/simulator/transactions"}


@router.get("/decisions")
def get_decisions(response: Response, symbol: str | None = Query(None), limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)) -> dict[str, Any]:
    _cache_response(response)
    normalized_symbol = symbol.upper() if symbol else None
    rows = _rows(
        db,
        f"""
        SELECT id, created_at, symbol, decision_session, kind, reason, portfolio,
               frozen_action_json, state_snapshot_json, veto_tier,
               would_have_entry_reason, disposition, tier
        FROM {table_name('sim_decisions')}
        WHERE (:symbol IS NULL OR symbol = :symbol) ORDER BY id DESC LIMIT :limit
        """,
        {"symbol": normalized_symbol, "limit": limit},
    )
    decisions = []
    for item in rows:
        item["state_snapshot"] = _json_object(item.pop("state_snapshot_json", None))
        item["frozen_action"] = _json_object(item.pop("frozen_action_json", None))
        decisions.append(item)
    return {"decisions": decisions, "sql_key": "GET /api/v2/simulator/decisions"}


@router.get("/symbols/state")
def get_symbols_state(response: Response, db: Session = Depends(get_db)) -> dict[str, Any]:
    _cache_response(response)
    rows = _rows(db, f"SELECT symbol, lifecycle, tier, session, source FROM {table_name('sim_symbol_state')} ORDER BY symbol")
    states = {row["symbol"]: row for row in rows}
    if not states:
        for symbol, state in _day_zero_inventory().get("symbols", {}).items():
            if not isinstance(state, dict):
                continue
            states[symbol] = {
                "symbol": symbol,
                "lifecycle": state.get("lifecycle") or "NEUTRAL",
                "tier": state.get("tier") or state.get("avoid_tier") or "NONE",
                "session": state.get("last_sealed_session"),
                "source": "day_zero_inventory",
            }
    return {"symbols": states, "sql_key": "GET /api/v2/simulator/symbols/state"}


@router.get("/system/integrity")
def get_system_integrity(response: Response, db: Session = Depends(get_db)) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    integrity = _row(db, f"SELECT * FROM {table_name('sim_integrity')} WHERE id = 1")
    if integrity is None:
        raise HTTPException(status_code=503, detail="simulator projection integrity is not available")
    row_counts = _json_object(integrity.get("postgres_row_counts_json"))
    source_counts = _json_object(integrity.get("sqlite_row_counts_json"))
    return {
        "seal_verification": _verify_simulator_seals(),
        "guard_trips_count": int(integrity.get("guard_trips_count") or 0),
        "last_session_processed": integrity.get("last_projected_session"),
        "projection_status": integrity.get("status"),
        "projection_stale": integrity.get("status") != "FRESH",
        "projection_stale_reason": integrity.get("stale_reason"),
        "projection_row_count_match": bool(integrity.get("row_count_match")),
        "projection_checksum_match": bool(integrity.get("checksum_match")),
        "row_counts": row_counts,
        "source_row_counts": source_counts,
        "ledger_sha256": integrity.get("ledger_sha256"),
        "sql_key": "GET /api/v2/simulator/system/integrity",
    }


@router.get("/sql-map")
def get_sql_map(response: Response) -> dict[str, str]:
    _cache_response(response)
    return SQL_MAP
