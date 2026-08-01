from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from functools import lru_cache
from pathlib import Path
from time import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response

from app.services.eagle_eye_v2.simulator.constants import INITIAL_CAPITAL_KWD, LEDGER_PATH, MANIFEST_PATH, SIMULATOR_ROOT

router = APIRouter(prefix="/simulator", tags=["Simulator v2"])

BOOKS = ("BUY", "WATCHLIST")
CACHE_SECONDS = 60
DAY_ZERO_INVENTORY_PATH = SIMULATOR_ROOT / "day_zero_state_inventory.json"

SQL_MAP: dict[str, str] = {
    "GET /api/v2/simulator/portfolios": """
WITH latest_session AS (
  SELECT portfolio, MAX(session) AS session FROM daily_valuations GROUP BY portfolio
), latest AS (
  SELECT d.* FROM daily_valuations d
  JOIN latest_session s ON s.portfolio = d.portfolio AND s.session = d.session
), positions AS (
  SELECT portfolio, symbol, SUM(CASE transaction_type WHEN 'BUY' THEN quantity WHEN 'SELL' THEN -quantity ELSE 0 END) AS quantity
  FROM transactions WHERE status = 'POSTED' GROUP BY portfolio, symbol
), open_counts AS (
  SELECT portfolio, COUNT(*) AS open_position_count FROM positions WHERE quantity > 0.000001 GROUP BY portfolio
), realized AS (
  SELECT portfolio, SUM(net_cash_delta_kwd) AS net_cash FROM transactions WHERE status = 'POSTED' GROUP BY portfolio
)
SELECT b.book AS book, COALESCE(MAX(l.nav_kwd), ?) AS nav_kwd, COALESCE(MAX(l.cash_kwd), ? + COALESCE(r.net_cash, 0), ?) AS cash_kwd,
       COALESCE(SUM(l.market_value_kwd), 0) AS invested_kwd, COALESCE(o.open_position_count, 0) AS open_position_count,
       COALESCE(MAX(l.nav_kwd), ?) - ? AS total_pnl_kwd, MIN(COALESCE(l.session, t.fill_session)) AS inception_date
FROM (SELECT 'BUY' AS book UNION ALL SELECT 'WATCHLIST') b
LEFT JOIN latest l ON l.portfolio = b.book
LEFT JOIN open_counts o ON o.portfolio = b.book
LEFT JOIN realized r ON r.portfolio = b.book
LEFT JOIN transactions t ON t.portfolio = b.book AND t.status = 'POSTED'
GROUP BY b.book
""",
    "GET /api/v2/simulator/portfolios/{book}/positions": """
WITH tx AS (
  SELECT portfolio, symbol,
         SUM(CASE transaction_type WHEN 'BUY' THEN quantity WHEN 'SELL' THEN -quantity ELSE 0 END) AS quantity,
         SUM(CASE WHEN transaction_type = 'BUY' THEN gross_value_kwd ELSE 0 END) AS buy_gross,
         MIN(CASE WHEN transaction_type = 'BUY' THEN fill_session END) AS entry_date,
         MIN(CASE WHEN transaction_type = 'BUY' THEN price END) AS entry_price,
         MIN(CASE WHEN transaction_type = 'BUY' THEN reason END) AS entry_reason
  FROM transactions WHERE status = 'POSTED' AND portfolio = ? GROUP BY portfolio, symbol
), latest_val AS (
  SELECT d.* FROM daily_valuations d
  JOIN (SELECT portfolio, symbol, MAX(session) AS session FROM daily_valuations WHERE portfolio = ? GROUP BY portfolio, symbol) s
    ON s.portfolio = d.portfolio AND s.symbol = d.symbol AND s.session = d.session
)
SELECT tx.symbol, tx.entry_date, tx.entry_price, tx.entry_reason, tx.quantity,
       latest_val.session AS last_session, latest_val.close_price AS last_close, latest_val.state_snapshot_json
FROM tx LEFT JOIN latest_val ON latest_val.portfolio = tx.portfolio AND latest_val.symbol = tx.symbol
WHERE tx.quantity > 0.000001 ORDER BY tx.symbol
""",
    "GET /api/v2/simulator/portfolios/{book}/nav": """
SELECT session, MAX(nav_kwd) AS nav_kwd, MAX(cash_kwd) AS cash_kwd, SUM(market_value_kwd) AS invested_kwd
FROM daily_valuations WHERE portfolio = ? GROUP BY session ORDER BY session DESC LIMIT ?
""",
    "GET /api/v2/simulator/transactions": """
SELECT * FROM transactions
WHERE (? IS NULL OR portfolio = ?) AND (? IS NULL OR symbol = ?)
ORDER BY id DESC LIMIT ?
""",
    "GET /api/v2/simulator/decisions": """
SELECT * FROM decision_log WHERE (? IS NULL OR symbol = ?) ORDER BY id DESC LIMIT ?
""",
    "GET /api/v2/simulator/symbols/state": """
SELECT d.symbol, d.decision_session, d.kind, d.reason, d.veto_tier, d.state_snapshot_json
FROM decision_log d JOIN (SELECT symbol, MAX(id) AS id FROM decision_log GROUP BY symbol) latest ON latest.id = d.id
ORDER BY d.symbol
""",
    "GET /api/v2/simulator/system/integrity": """
SELECT 'transactions' AS table_name, COUNT(*) AS row_count FROM transactions
UNION ALL SELECT 'daily_valuations', COUNT(*) FROM daily_valuations
UNION ALL SELECT 'decision_log', COUNT(*) FROM decision_log
UNION ALL SELECT 'guard_trips', COUNT(*) FROM guard_trips
UNION ALL SELECT 'monthly_hashes', COUNT(*) FROM monthly_hashes
""",
}


def _ledger_path() -> Path:
    return Path(os.environ.get("SIMULATOR_LEDGER_PATH", str(LEDGER_PATH)))


def _readonly_uri(path: Path) -> str:
    return f"file:{path.resolve().as_posix()}?mode=ro"


def _connect_ro() -> sqlite3.Connection:
    path = _ledger_path()
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"simulator ledger not found: {path}")
    conn = sqlite3.connect(_readonly_uri(path), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


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


def _pct(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return (numerator / denominator) * 100.0


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


@router.get("/portfolios")
def get_portfolios(response: Response) -> dict[str, Any]:
    _cache_response(response)
    with _connect_ro() as conn:
        rows = conn.execute(SQL_MAP["GET /api/v2/simulator/portfolios"], (INITIAL_CAPITAL_KWD,) * 5).fetchall()
    portfolios = []
    for row in rows:
        nav = float(row["nav_kwd"] or INITIAL_CAPITAL_KWD)
        inception = row["inception_date"]
        portfolios.append({
            "book": row["book"],
            "nav_kwd": nav,
            "cash_kwd": float(row["cash_kwd"] or INITIAL_CAPITAL_KWD),
            "invested_kwd": float(row["invested_kwd"] or 0.0),
            "open_position_count": int(row["open_position_count"] or 0),
            "total_pnl_kwd": float(row["total_pnl_kwd"] or 0.0),
            "change_since_inception_pct": _pct(nav - INITIAL_CAPITAL_KWD, INITIAL_CAPITAL_KWD),
            "inception_date": inception,
        })
    return {"portfolios": portfolios, "sql_key": "GET /api/v2/simulator/portfolios"}


@router.get("/portfolios/{book}/positions")
def get_positions(book: str, response: Response) -> dict[str, Any]:
    book = _book(book)
    _cache_response(response)
    with _connect_ro() as conn:
        rows = conn.execute(SQL_MAP["GET /api/v2/simulator/portfolios/{book}/positions"], (book, book)).fetchall()
    positions = []
    for row in rows:
        state = _json_object(row["state_snapshot_json"])
        quantity = float(row["quantity"] or 0.0)
        entry_price = float(row["entry_price"] or 0.0)
        last_close = float(row["last_close"] or entry_price)
        unrealized = (last_close - entry_price) * quantity
        positions.append({
            "symbol": row["symbol"],
            "entry_date": row["entry_date"],
            "entry_price": entry_price,
            "entry_reason": row["entry_reason"],
            "sessions_held": _sessions_between(row["entry_date"], row["last_session"]),
            "last_close": last_close,
            "unrealized_pnl_pct": _pct(last_close - entry_price, entry_price),
            "unrealized_pnl_kwd": unrealized,
            "current_lifecycle": state.get("lifecycle_state") or state.get("lifecycle") or state.get("lifecycle_status"),
            "avoid_tier": state.get("avoid_tier") or state.get("tier") or "NONE",
        })
    return {"book": book, "positions": positions, "sql_key": "GET /api/v2/simulator/portfolios/{book}/positions"}


@router.get("/portfolios/{book}/nav")
def get_nav(book: str, response: Response, days: int = Query(60, ge=1, le=1000)) -> dict[str, Any]:
    book = _book(book)
    _cache_response(response)
    with _connect_ro() as conn:
        rows = conn.execute(SQL_MAP["GET /api/v2/simulator/portfolios/{book}/nav"], (book, days)).fetchall()
    series = [
        {
            "session": row["session"],
            "nav_kwd": float(row["nav_kwd"]),
            "cash_kwd": float(row["cash_kwd"]),
            "invested_kwd": float(row["invested_kwd"] or 0.0),
        }
        for row in reversed(rows)
    ]
    if not series:
        series = [{"session": None, "nav_kwd": INITIAL_CAPITAL_KWD, "cash_kwd": INITIAL_CAPITAL_KWD, "invested_kwd": 0.0}]
    return {"book": book, "series": series, "sql_key": "GET /api/v2/simulator/portfolios/{book}/nav"}


@router.get("/transactions")
def get_transactions(
    response: Response,
    book: str | None = Query(None),
    symbol: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    _cache_response(response)
    normalized_book = _book(book) if book else None
    normalized_symbol = symbol.upper() if symbol else None
    with _connect_ro() as conn:
        rows = conn.execute(
            SQL_MAP["GET /api/v2/simulator/transactions"],
            (normalized_book, normalized_book, normalized_symbol, normalized_symbol, limit),
        ).fetchall()
    return {"transactions": [dict(row) for row in rows], "sql_key": "GET /api/v2/simulator/transactions"}


@router.get("/decisions")
def get_decisions(response: Response, symbol: str | None = Query(None), limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
    _cache_response(response)
    normalized_symbol = symbol.upper() if symbol else None
    with _connect_ro() as conn:
        rows = conn.execute(SQL_MAP["GET /api/v2/simulator/decisions"], (normalized_symbol, normalized_symbol, limit)).fetchall()
    decisions = []
    for row in rows:
        item = dict(row)
        item["state_snapshot"] = _json_object(item.pop("state_snapshot_json", None))
        item["frozen_action"] = _json_object(item.pop("frozen_action_json", None))
        item["disposition"] = item.get("kind")
        item["tier"] = item.get("veto_tier") or item["state_snapshot"].get("avoid_tier") or item["state_snapshot"].get("tier")
        decisions.append(item)
    return {"decisions": decisions, "sql_key": "GET /api/v2/simulator/decisions"}


@router.get("/symbols/state")
def get_symbols_state(response: Response) -> dict[str, Any]:
    _cache_response(response)
    states: dict[str, dict[str, Any]] = {}
    with _connect_ro() as conn:
        rows = conn.execute(SQL_MAP["GET /api/v2/simulator/symbols/state"]).fetchall()
    for row in rows:
        state = _json_object(row["state_snapshot_json"])
        states[row["symbol"]] = {
            "symbol": row["symbol"],
            "lifecycle": state.get("lifecycle_state") or state.get("lifecycle") or "NEUTRAL",
            "tier": state.get("avoid_tier") or state.get("tier") or row["veto_tier"] or "NONE",
            "session": row["decision_session"],
            "source": "decision_log",
        }
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
def get_system_integrity(response: Response) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    ledger_path = _ledger_path()
    with _connect_ro() as conn:
        row_counts = {row["table_name"]: int(row["row_count"]) for row in conn.execute(SQL_MAP["GET /api/v2/simulator/system/integrity"]).fetchall()}
        last_session = conn.execute("SELECT MAX(session) FROM daily_valuations").fetchone()[0]
        guard_trips = conn.execute("SELECT COUNT(*) FROM guard_trips").fetchone()[0]
    return {
        "seal_verification": _verify_simulator_seals(),
        "guard_trips_count": int(guard_trips or 0),
        "last_session_processed": last_session,
        "row_counts": row_counts,
        "ledger_sha256": _sha256(ledger_path),
        "sql_key": "GET /api/v2/simulator/system/integrity",
    }


@router.get("/sql-map")
def get_sql_map(response: Response) -> dict[str, str]:
    _cache_response(response)
    return SQL_MAP


def _sessions_between(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    from datetime import date

    try:
        return max(0, (date.fromisoformat(end) - date.fromisoformat(start)).days)
    except ValueError:
        return None