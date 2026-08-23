from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

from app.services.eagle_eye_v2.simulator.accounting import PaperPortfolioEngine, SessionExecutionResult, canonical_entry_reason
from app.services.eagle_eye_v2.simulator.constants import ARCHIVE_ROOT, ENTRY_REASONS, FORWARD_SURFACE_DB, FROZEN_VARIANT, RELEASE_ROOT, SIMULATOR_ROOT
from app.services.eagle_eye_v2.simulator.ledger import SimulatorLedger
from app.services.eagle_eye_v2.simulator.market_data_source import SealedReplayMarketDataSource, resolve_market_data_source
from app.services.eagle_eye_v2.simulator.models import DecisionKind, FrozenEvent, MarketSession
from app.services.eagle_eye_v2.simulator.sealed_imports import verify_frozen_imports

DAY_ZERO_SOURCE_DB = ARCHIVE_ROOT / "v5x_candidates" / "harness_dbs" / "harness_v53A_2026-07-27T150230_580976Z.db"
RUN_KEY = "R16_3_HARNESS_V53_A"


class SimulatorRunner:
    def __init__(self, ledger: SimulatorLedger | None = None, *, mode: str = "sealed", source_db: Path | str | None = None, live_db_path: Path | str | None = None, expected_symbol_count: int | None = None) -> None:
        self.ledger = ledger or SimulatorLedger()
        self.engine = PaperPortfolioEngine(self.ledger)
        self.frozen_hashes = verify_frozen_imports()
        self.mode = mode
        self.expected_symbol_count = expected_symbol_count
        self.market_data_source = resolve_market_data_source(
            mode,
            source_db=source_db or DAY_ZERO_SOURCE_DB,
            db_path=live_db_path,
            expected_symbol_count=expected_symbol_count,
            surface_db_path=FORWARD_SURFACE_DB,
        )

    def load_market_sessions(self, session_date: str, *, mode: str | None = None, expected_symbol_count: int | None = None) -> dict[str, MarketSession]:
        source_mode = mode or self.mode
        source = self.market_data_source if source_mode == self.mode and (source_mode == "sealed" or source_mode == "live") else resolve_market_data_source(
            source_mode,
            source_db=DAY_ZERO_SOURCE_DB,
            db_path=self.market_data_source.db_path if hasattr(self.market_data_source, "db_path") else None,
            expected_symbol_count=expected_symbol_count,
            surface_db_path=getattr(self.market_data_source, "surface_db_path", FORWARD_SURFACE_DB),
        )
        return source.load_session_rows(session_date=session_date, expected_symbol_count=expected_symbol_count)

    @staticmethod
    def load_replay_window(symbol: str, start_date: str, end_date: str, forward_db: Path | str | None = None) -> list[dict[str, Any]]:
        release_scripts = RELEASE_ROOT / "scripts"
        if str(release_scripts) not in sys.path:
            sys.path.insert(0, str(release_scripts))
        import forward_replay

        load_from, effective_end = forward_replay.symbol_replay_window(symbol, start_date, end_date)
        return forward_replay.continuous_symbol_rows(
            symbol,
            load_from,
            effective_end,
            forward_replay.resolve_forward_db(forward_db),
        )

    def ingest_session(
        self,
        *,
        session: str,
        market_sessions: dict[str, MarketSession],
        frozen_events: Iterable[FrozenEvent],
    ) -> SessionExecutionResult:
        normalized = {symbol.upper(): row for symbol, row in market_sessions.items()}
        return self.engine.process_session(session, normalized, frozen_events)

    @staticmethod
    def events_from_actions(symbol: str, decision_session: str, actions: Iterable[dict[str, Any]], state_snapshot: dict[str, Any]) -> list[FrozenEvent]:
        events: list[FrozenEvent] = []
        for action in actions:
            action_type = str(action.get("type") or "")
            if action_type == "OPEN_POSITION":
                reason = canonical_entry_reason(str(action.get("entry_reason") or ""))
                if reason in ENTRY_REASONS:
                    events.append(FrozenEvent(symbol.upper(), decision_session, DecisionKind.ENTRY, reason, dict(action), state_snapshot))
            elif action_type == "CLOSE_POSITION":
                events.append(FrozenEvent(symbol.upper(), decision_session, DecisionKind.EXIT, str(action.get("exit_reason") or "EXIT"), dict(action), state_snapshot))
            elif action_type == "DAILY_STATE":
                events.append(FrozenEvent(symbol.upper(), decision_session, DecisionKind.DAILY_STATE, str(action.get("avoid_tier") or "NONE"), dict(action), state_snapshot))
        return events

    @staticmethod
    def veto_event(
        *,
        symbol: str,
        decision_session: str,
        would_have_entry_reason: str,
        veto_tier: str,
        state_snapshot: dict[str, Any],
        action: dict[str, Any] | None = None,
    ) -> FrozenEvent:
        return FrozenEvent(
            symbol=symbol.upper(),
            decision_session=decision_session,
            kind=DecisionKind.VETO,
            reason=f"{veto_tier}_VETO",
            action=action or {},
            state_snapshot=state_snapshot,
            would_have_entry_reason=canonical_entry_reason(would_have_entry_reason),
            veto_tier=veto_tier,
        )

    def write_day_zero_snapshot(self, output_path: Path | None = None, source_db: Path = DAY_ZERO_SOURCE_DB) -> dict[str, Any]:
        snapshot = build_day_zero_inventory(source_db)
        output_path = output_path or SIMULATOR_ROOT / "day_zero_state_inventory.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
        digest = sha256_file(output_path)
        Path(str(output_path) + ".sha256").write_text(f"{digest}  {output_path.name}\n", encoding="ascii")
        return {"path": str(output_path), "sha256": digest, "symbols": len(snapshot["symbols"]), "source_db": str(source_db)}


def build_day_zero_inventory(source_db: Path = DAY_ZERO_SOURCE_DB) -> dict[str, Any]:
    if not source_db.exists():
        raise FileNotFoundError(f"sealed v5.3-A replay DB not found: {source_db}")
    query = """
    WITH ranked AS (
        SELECT symbol, trade_date, row_json,
               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date DESC) AS rn
        FROM r16_daily_rows
        WHERE run_key = ?
    )
    SELECT symbol, trade_date, row_json FROM ranked WHERE rn = 1 ORDER BY symbol
    """
    symbols: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True) as conn:
        for symbol, trade_date, row_json in conn.execute(query, (RUN_KEY,)):
            row = json.loads(row_json)
            symbols[str(symbol)] = {
                "last_sealed_session": trade_date,
                "lifecycle": row.get("state"),
                "tier": row.get("disposition_state") or row.get("avoid_tier"),
                "avoid_tier": row.get("avoid_tier"),
                "confirmation_state": row.get("confirmation_state"),
                "candidate_intent_state": row.get("candidate_intent_state"),
                "position": row.get("position"),
            }
    return {
        "schema": "SIM-1_DAY_ZERO_STATE_INVENTORY_V1",
        "authority": "Freeze V3 governs all decision logic; SIM-1 adds accounting/execution only.",
        "frozen_variant": FROZEN_VARIANT,
        "source_db": str(source_db),
        "run_key": RUN_KEY,
        "symbols": symbols,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
