from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.services.eagle_eye_v2.simulator.constants import SIMULATOR_ROOT
from app.services.eagle_eye_v2.simulator.ledger import SimulatorLedger


def write_weekly_owner_report(ledger: SimulatorLedger, week_start: str, week_end: str, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or SIMULATOR_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_weekly_owner_report(ledger, week_start, week_end)
    path = output_dir / f"sim1_weekly_owner_report_{week_start}_to_{week_end}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    digest = sha256_file(path)
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return {"path": str(path), "sha256": digest, "week_start": week_start, "week_end": week_end}


def build_weekly_owner_report(ledger: SimulatorLedger, week_start: str, week_end: str) -> dict[str, Any]:
    with ledger.connect() as conn:
        transactions = [
            dict(row)
            for row in conn.execute(
                """
                SELECT * FROM transactions
                WHERE fill_session BETWEEN ? AND ?
                ORDER BY fill_session, id
                """,
                (week_start, week_end),
            )
        ]
        nav_series = [
            dict(row)
            for row in conn.execute(
                """
                SELECT portfolio, session, MAX(nav_kwd) AS nav_kwd
                FROM daily_valuations
                WHERE session BETWEEN ? AND ?
                GROUP BY portfolio, session
                ORDER BY session, portfolio
                """,
                (week_start, week_end),
            )
        ]
        guard_trips = [
            dict(row)
            for row in conn.execute(
                """
                SELECT * FROM guard_trips
                WHERE decision_session BETWEEN ? AND ?
                ORDER BY decision_session, id
                """,
                (week_start, week_end),
            )
        ]
        suspension_gaps = [row for row in transactions if int(row.get("suspension_gap_sessions") or 0) > 0]
        latest_positions = [
            dict(row)
            for row in conn.execute(
                """
                WITH latest AS (
                    SELECT portfolio, symbol, quantity, close_price, market_value_kwd, cash_kwd, nav_kwd, session,
                           ROW_NUMBER() OVER (PARTITION BY portfolio, symbol ORDER BY session DESC, id DESC) AS rn
                    FROM daily_valuations
                    WHERE session <= ?
                )
                SELECT * FROM latest WHERE rn = 1 AND quantity > 0 ORDER BY portfolio, symbol
                """,
                (week_end,),
            )
        ]
    return {
        "schema": "SIM-1_WEEKLY_OWNER_REPORT_V1",
        "week_start": week_start,
        "week_end": week_end,
        "positions": latest_positions,
        "fills": [row for row in transactions if row["transaction_type"] == "BUY"],
        "exits": [row for row in transactions if row["transaction_type"] == "SELL"],
        "nav_series": nav_series,
        "buy_vs_watchlist_divergence": _divergence(nav_series),
        "guard_trips": guard_trips,
        "suspension_gaps": suspension_gaps,
    }


def _divergence(nav_series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_session: dict[str, dict[str, float]] = {}
    for row in nav_series:
        by_session.setdefault(str(row["session"]), {})[str(row["portfolio"])] = float(row["nav_kwd"] or 0.0)
    out = []
    for session, values in sorted(by_session.items()):
        buy = values.get("BUY")
        watch = values.get("WATCHLIST")
        if buy is not None and watch is not None:
            out.append({"session": session, "buy_nav_kwd": buy, "watchlist_nav_kwd": watch, "difference_kwd": buy - watch})
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
