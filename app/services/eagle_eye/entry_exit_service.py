from __future__ import annotations

import json
from typing import Any

from app.core.database import exec_sql, query_one


def _json_load(raw: Any) -> dict[str, Any]:
    try:
        return json.loads(str(raw or "{}"))
    except Exception:
        return {}


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def maybe_open_or_add_position(
    symbol: str,
    signal_type: str,
    signal_id: int,
    trade_date: int,
    price: float,
    stop_price: float,
    pilot_enabled: bool,
) -> None:
    open_row = query_one(
        "SELECT id, tranches_json, avg_entry, stop_price FROM ee_positions WHERE symbol = ? AND status = 'open' ORDER BY id DESC LIMIT 1",
        (symbol,),
    )

    if signal_type == "ACCUMULATION_ALERT" and not pilot_enabled:
        return

    if signal_type == "ACCUMULATION_ALERT":
        tranche = {"name": "T1", "size_pct": 25, "price": price, "date": trade_date}
    elif signal_type == "BREAKOUT_CONFIRMED":
        tranche = {"name": "T2", "size_pct": 50, "price": price, "date": trade_date}
    elif signal_type == "ADD_ON_PULLBACK":
        tranche = {"name": "T3", "size_pct": 25, "price": price, "date": trade_date}
    else:
        return

    if not open_row:
        tranches = [tranche]
        avg_entry = price
        exec_sql(
            """
            INSERT INTO ee_positions (
                symbol, opened_at, status, tranches_json, avg_entry, stop_price, trail_price, signal_id
            ) VALUES (?, ?, 'open', ?, ?, ?, ?, ?)
            """,
            (symbol, trade_date, _json_dump(tranches), avg_entry, stop_price, stop_price, signal_id),
        )
        return

    tranches = _json_load(open_row.get("tranches_json"))
    if not isinstance(tranches, list):
        tranches = []

    current = {t.get("name") for t in tranches if isinstance(t, dict)}
    if tranche["name"] in current:
        return

    tranches.append(tranche)
    total_weight = sum(float(t.get("size_pct") or 0.0) for t in tranches)
    weighted_price = sum((float(t.get("price") or 0.0) * float(t.get("size_pct") or 0.0)) for t in tranches)
    avg_entry = weighted_price / total_weight if total_weight > 0 else price

    exec_sql(
        """
        UPDATE ee_positions
        SET tranches_json = ?, avg_entry = ?, stop_price = ?
        WHERE id = ?
        """,
        (_json_dump(tranches), avg_entry, stop_price, open_row.get("id")),
    )


def update_trailing_stop(symbol: str, trail_price: float) -> None:
    exec_sql(
        """
        UPDATE ee_positions
        SET trail_price = ?
        WHERE symbol = ? AND status = 'open'
        """,
        (trail_price, symbol),
    )


def close_open_position(symbol: str, trade_date: int, reason: str, exit_price: float) -> bool:
    row = query_one(
        """
        SELECT id, avg_entry
        FROM ee_positions
        WHERE symbol = ? AND status = 'open'
        ORDER BY id DESC
        LIMIT 1
        """,
        (symbol,),
    )
    if not row:
        return False

    avg_entry = float(row.get("avg_entry") or 0.0)
    realized = None
    if avg_entry > 0:
        realized = (exit_price - avg_entry) / avg_entry

    exec_sql(
        """
        UPDATE ee_positions
        SET status = 'closed',
            closed_at = ?,
            exit_reason = ?,
            realized_return = ?
        WHERE id = ?
        """,
        (trade_date, reason, realized, row.get("id")),
    )
    return True


def get_position_state(symbol: str) -> dict[str, Any] | None:
    row = query_one(
        """
        SELECT id, symbol, opened_at, closed_at, status, tranches_json,
               avg_entry, stop_price, trail_price, exit_reason, realized_return, signal_id
        FROM ee_positions
        WHERE symbol = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (symbol,),
    )
    if not row:
        return None

    return {
        "id": row.get("id"),
        "symbol": row.get("symbol"),
        "opened_at": row.get("opened_at"),
        "closed_at": row.get("closed_at"),
        "status": row.get("status"),
        "tranches": _json_load(row.get("tranches_json")),
        "avg_entry": row.get("avg_entry"),
        "stop_price": row.get("stop_price"),
        "trail_price": row.get("trail_price"),
        "exit_reason": row.get("exit_reason"),
        "realized_return": row.get("realized_return"),
        "signal_id": row.get("signal_id"),
    }
