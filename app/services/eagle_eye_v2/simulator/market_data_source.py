from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.eagle_eye_v2.simulator.constants import ARCHIVE_ROOT
from app.services.eagle_eye_v2.simulator.forward_surface import FORWARD_SURFACE_RUN_KEY
from app.services.eagle_eye_v2.simulator.models import MarketSession

DEFAULT_SESSION_CLOSE = "15:00:00+00:00"
DEFAULT_EXPECTED_SYMBOL_COUNT = 139


class MarketDataSource(ABC):
    """Abstract market-data source for simulator sessions."""

    @abstractmethod
    def load_session_rows(self, session_date: str | None = None, *, expected_symbol_count: int | None = None) -> dict[str, MarketSession]:
        raise NotImplementedError

    @staticmethod
    def schema_compatibility_map() -> dict[str, Any]:
        return {
            "ee_ohlcv_cache_columns": {
                "ticker": {"source": "ee_ohlcv_cache.ticker", "required_by": ["MarketSession.symbol"], "supplied_as": "symbol"},
                "bar_date": {"source": "ee_ohlcv_cache.bar_date", "required_by": ["MarketSession.session", "date guard"], "supplied_as": "session"},
                "open": {"source": "ee_ohlcv_cache.open", "required_by": ["Layer-1 open price input"], "supplied_as": "open_price"},
                "high": {"source": "ee_ohlcv_cache.high", "required_by": ["Layer-1 range/volatility inputs"], "supplied_as": "high_price"},
                "low": {"source": "ee_ohlcv_cache.low", "required_by": ["Layer-1 range/volatility inputs"], "supplied_as": "low_price"},
                "close": {"source": "ee_ohlcv_cache.close", "required_by": ["Layer-1 decision close price", "PortfolioEngine._write_valuations"], "supplied_as": "close_price"},
                "volume": {"source": "ee_ohlcv_cache.volume", "required_by": ["Layer-1 participation / conviction inputs"], "supplied_as": "volume"},
                "turnover_kwd": {"source": "ee_ohlcv_cache.turnover_kwd", "required_by": ["Layer-1 value/turnover inputs"], "supplied_as": "turnover_kwd"},
                "fetched_at": {"source": "ee_ohlcv_cache.fetched_at", "required_by": ["ingestion-time guard", "backfill parity"], "supplied_as": "ingestion_ts"},
            },
            "frozen_layer1_requirements": {
                "symbol": "required",
                "session": "required",
                "open_price": "required",
                "close_price": "required",
                "ingestion_ts": "required before decision_close_ts",
                "decision_close_ts": "required for guard parity",
            },
            "blocker_policy": "Any missing Layer-1 input or row set that fails the session/date guard is a hard blocker; no approximation is allowed.",
        }


class SealedReplayMarketDataSource(MarketDataSource):
    def __init__(self, source_db: Path | str | None = None):
        self.source_db = Path(source_db) if source_db is not None else ARCHIVE_ROOT / "v5x_candidates" / "harness_dbs" / "harness_v53A_2026-07-27T150230_580976Z.db"

    def load_session_rows(self, session_date: str | None = None, *, expected_symbol_count: int | None = None) -> dict[str, MarketSession]:
        if not self.source_db.exists():
            raise FileNotFoundError(f"sealed v5.3-A replay DB not found: {self.source_db}")

        query = """
            WITH ranked AS (
                SELECT symbol, trade_date, row_json,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date DESC) AS rn
                FROM r16_daily_rows
                WHERE run_key = ?
            )
            SELECT symbol, trade_date, row_json FROM ranked WHERE rn = 1 ORDER BY symbol
        """
        run_key = "R16_3_HARNESS_V53_A"
        symbols: dict[str, MarketSession] = {}
        with sqlite3.connect(f"file:{self.source_db.as_posix()}?mode=ro", uri=True) as conn:
            for symbol, trade_date, row_json in conn.execute(query, (run_key,)):
                row = json.loads(row_json)
                symbols[str(symbol)] = MarketSession(
                    symbol=str(symbol),
                    session=str(trade_date),
                    open_price=float(row.get("open") or 0.0),
                    close_price=float(row.get("close") or 0.0),
                    ingestion_ts=f"{trade_date}T12:00:00+00:00",
                    decision_close_ts=f"{trade_date}T{DEFAULT_SESSION_CLOSE}",
                )
        if expected_symbol_count is not None and len(symbols) != expected_symbol_count:
            raise RuntimeError(f"sealed source expected symbol count {expected_symbol_count} but found {len(symbols)}")
        return symbols


class LiveMarketDataSource(MarketDataSource):
    def __init__(self, db_path: Path | str | None = None, *, session_date: str | None = None, expected_symbol_count: int | None = None, surface_db_path: Path | str | None = None):
        if db_path is None:
            settings = get_settings()
            db_path = settings.database_abs_path
        self.db_path = Path(db_path)
        self.session_date = session_date
        self.expected_symbol_count = expected_symbol_count
        self.surface_db_path = Path(surface_db_path) if surface_db_path is not None else None

    def load_session_rows(self, session_date: str | None = None, *, expected_symbol_count: int | None = None) -> dict[str, MarketSession]:
        target_session = str(session_date or self.session_date or "").strip()
        if not target_session:
            raise RuntimeError("LiveMarketDataSource requires a session_date to load rows")
        expected = expected_symbol_count if expected_symbol_count is not None else self.expected_symbol_count

        if self.surface_db_path is None:
            raise RuntimeError(
                "forward surface is required for live decision runs; raw ee_ohlcv_cache is not a valid decision surface and is never used as a degraded fallback."
            )
        if not self.surface_db_path.exists():
            raise RuntimeError(
                f"forward surface is missing for session {target_session}: {self.surface_db_path}. No raw-cache fallback is allowed."
            )

        rows = self._load_from_surface(target_session)
        actual = len(rows)
        if expected is not None and actual != expected:
            raise RuntimeError(f"live source expected symbol count {expected} but found {actual} for session {target_session}")
        if actual == 0:
            raise RuntimeError(f"forward surface has no rows for session {target_session} in {self.surface_db_path}")
        for symbol, market in rows.items():
            if str(market.session) != target_session:
                raise RuntimeError(f"live source row mismatch for symbol {symbol}: session {market.session} != {target_session}")
            if not market.ingestion_ts:
                raise RuntimeError(f"live source missing ingestion timestamp for {symbol}")
            if self._clock_order_violation(market.ingestion_ts, market.decision_close_ts):
                raise RuntimeError(f"live source backfill guard failed for {symbol}: ingestion {market.ingestion_ts} is not before decision close {market.decision_close_ts}")

        return rows

    def _load_from_live_cache(self, target_session: str) -> dict[str, MarketSession]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"live market DB not found: {self.db_path}")

        query = """
            SELECT ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at
            FROM ee_ohlcv_cache
            WHERE bar_date = ?
            ORDER BY ticker
        """
        rows: dict[str, MarketSession] = {}
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(query, (target_session,)):
                symbol = str(row["ticker"]).upper()
                rows[symbol] = MarketSession(
                    symbol=symbol,
                    session=str(row["bar_date"]),
                    open_price=float(row["open"] or 0.0),
                    close_price=float(row["close"] or 0.0),
                    ingestion_ts=str(row["fetched_at"] or ""),
                    decision_close_ts=f"{target_session}T{DEFAULT_SESSION_CLOSE}",
                )
        return rows

    def _load_from_surface(self, target_session: str) -> dict[str, MarketSession]:
        query = """
            SELECT symbol, trade_date, row_json, calendar_version_id, mask_manifest_version_id
            FROM forward_surface_rows
            WHERE run_key = ? AND trade_date = ?
            ORDER BY symbol
        """
        rows: dict[str, MarketSession] = {}
        with sqlite3.connect(str(self.surface_db_path)) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(query, (FORWARD_SURFACE_RUN_KEY, target_session)):
                payload = json.loads(row["row_json"])
                symbol = str(row["symbol"]).upper()
                if str(row["calendar_version_id"]) != "BK_CAL_V4_1783783330":
                    raise RuntimeError(f"forward surface authority mismatch for {symbol}: {row['calendar_version_id']}")
                if str(row["mask_manifest_version_id"]) != "R12_MASKED_INTERVALS_MANIFEST_V4_3_FINAL":
                    raise RuntimeError(f"forward surface mask manifest mismatch for {symbol}: {row['mask_manifest_version_id']}")
                rows[symbol] = MarketSession(
                    symbol=symbol,
                    session=str(row["trade_date"]),
                    open_price=float(payload.get("open") or 0.0),
                    close_price=float(payload.get("close") or 0.0),
                    ingestion_ts=f"{target_session}T12:00:00+00:00",
                    decision_close_ts=f"{target_session}T{DEFAULT_SESSION_CLOSE}",
                )
        return rows

    @staticmethod
    def _clock_order_violation(ingestion_ts: str, decision_close_ts: str) -> bool:
        try:
            ingestion_dt = datetime.fromisoformat(ingestion_ts.replace("Z", "+00:00"))
            decision_dt = datetime.fromisoformat(decision_close_ts.replace("Z", "+00:00"))
        except ValueError:
            return True
        return ingestion_dt >= decision_dt


def resolve_market_data_source(mode: str = "sealed", **kwargs) -> MarketDataSource:
    mode_key = (mode or "sealed").strip().lower()
    if mode_key == "live":
        filtered = {key: value for key, value in kwargs.items() if key in {"db_path", "session_date", "expected_symbol_count", "surface_db_path"}}
        return LiveMarketDataSource(**filtered)
    if mode_key in {"sealed", "genesis", "replay"}:
        filtered = {key: value for key, value in kwargs.items() if key in {"source_db"}}
        return SealedReplayMarketDataSource(**filtered)
    raise ValueError(f"unknown market data mode: {mode}")
