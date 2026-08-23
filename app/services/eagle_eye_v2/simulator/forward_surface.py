from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

FORWARD_SURFACE_RUN_KEY = "FORWARD_SURFACE"
CALENDAR_VERSION_ID = "BK_CAL_V4_1783783330"
MASK_MANIFEST_VERSION_ID = "R12_MASKED_INTERVALS_MANIFEST_V4_3_FINAL"
SEALED_AUTHORITY_DB = Path(r"C:\Users\Sager\OneDrive\Desktop\portfolio_app\mobile-migration\backend-api-main-release\artifacts\preview1a_prestart\review_final\r12_exam_surface_v4_5_runtime.db")
EXPECTED_SEGMENT_MAP_ROWS = 309


class ForwardSurfaceBuilder:
    """Append a session from the live OHLCV cache into the sealed forward surface."""

    def __init__(
        self,
        *,
        live_db_path: Path | str,
        sealed_db_path: Path | str | None = None,
        surface_db_path: Path | str | None = None,
        run_key: str = FORWARD_SURFACE_RUN_KEY,
        calendar_version_id: str = CALENDAR_VERSION_ID,
        mask_manifest_version_id: str = MASK_MANIFEST_VERSION_ID,
    ) -> None:
        self.live_db_path = Path(live_db_path)
        self.sealed_db_path = Path(sealed_db_path) if sealed_db_path is not None else SEALED_AUTHORITY_DB
        self.surface_db_path = Path(surface_db_path) if surface_db_path is not None else Path("F:/eagle_eye_archive/forward_surface/ee_forward_surface.db")
        self.run_key = run_key
        self.surface_authority_ids = {
            "calendar_version_id": calendar_version_id,
            "mask_manifest_version_id": mask_manifest_version_id,
        }

    def ensure_surface_db(self) -> None:
        self.surface_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.surface_db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forward_surface_rows (
                    run_key TEXT,
                    symbol TEXT,
                    trade_date TEXT,
                    row_json TEXT,
                    calendar_version_id TEXT,
                    mask_manifest_version_id TEXT,
                    status TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forward_surface_quarantine (
                    run_key TEXT,
                    symbol TEXT,
                    session TEXT,
                    reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_forward_surface_symbol_date ON forward_surface_rows(run_key, symbol, trade_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_forward_surface_trade_date ON forward_surface_rows(run_key, trade_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_forward_surface_quarantine_session ON forward_surface_quarantine(run_key, session, symbol)"
            )
            conn.commit()

    @staticmethod
    def iter_bk_cal_v4_sessions(start_date: str, end_date: str) -> list[str]:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        current = start
        sessions: list[str] = []
        while current <= end:
            if current.weekday() not in {4, 5}:
                sessions.append(current.isoformat())
            current += timedelta(days=1)
        return sessions

    @staticmethod
    def _segment_resolution_map(sealed_db_path: Path) -> dict[str, list[dict[str, Any]]]:
        if not sealed_db_path.exists():
            raise FileNotFoundError(f"sealed authority DB not found: {sealed_db_path}")

        with sqlite3.connect(f"file:{sealed_db_path.as_posix()}?mode=ro", uri=True) as conn:
            table_exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ee_symbol_segment_map'").fetchone()
            if table_exists is None:
                raise RuntimeError(f"sealed authority DB is missing ee_symbol_segment_map: {sealed_db_path}")

            row_count = conn.execute("SELECT COUNT(*) FROM ee_symbol_segment_map").fetchone()[0]
            if int(row_count) != EXPECTED_SEGMENT_MAP_ROWS:
                raise RuntimeError(
                    f"sealed authority DB segment map row count mismatch: expected {EXPECTED_SEGMENT_MAP_ROWS}, found {int(row_count)} in {sealed_db_path}"
                )

            rows = conn.execute(
                "SELECT original_symbol, segment_symbol, segment_id, start_trade_date, end_trade_date FROM ee_symbol_segment_map ORDER BY original_symbol, segment_id",
            ).fetchall()

        resolved: dict[str, list[dict[str, Any]]] = {}
        for original_symbol, segment_symbol, segment_id, start_trade_date, end_trade_date in rows:
            key = str(original_symbol).upper()
            resolved.setdefault(key, []).append({
                "segment_symbol": str(segment_symbol),
                "segment_id": int(segment_id),
                "start_trade_date": int(start_trade_date),
                "end_trade_date": int(end_trade_date),
            })
        return resolved

    @staticmethod
    def _session_epoch(session_date: str) -> int:
        try:
            return int(datetime.fromisoformat(session_date).timestamp())
        except ValueError:
            return int(datetime.strptime(session_date, "%Y-%m-%d").timestamp())

    @staticmethod
    def _sealed_window_end(segment_map: dict[str, list[dict[str, Any]]]) -> int | None:
        max_end: int | None = None
        for rows in segment_map.values():
            for row in rows:
                end_value = int(row["end_trade_date"])
                if max_end is None or end_value > max_end:
                    max_end = end_value
        return max_end

    def _resolve_market_symbol(self, canonical_symbol: str, segment_map: dict[str, list[dict[str, Any]]], session_date: str) -> str:
        key = canonical_symbol.upper()
        rows = segment_map.get(key, [])
        if not rows:
            raise RuntimeError(f"canonical symbol {canonical_symbol} is unmapped in ee_symbol_segment_map and must be rejected")

        session_epoch = self._session_epoch(session_date)
        sealed_window_end = self._sealed_window_end(segment_map)
        if sealed_window_end is None:
            raise RuntimeError(f"segment map is empty; cannot resolve {canonical_symbol} for {session_date}")

        active_rows = [
            row for row in rows
            if (
                int(row["start_trade_date"]) <= session_epoch <= int(row["end_trade_date"])
            )
        ]
        if active_rows:
            if len(active_rows) > 1:
                raise RuntimeError(
                    f"canonical symbol {canonical_symbol} has a segment change requiring owner ruling for {session_date}; active segments: {sorted({r['segment_symbol'] for r in active_rows})}"
                )
            return str(active_rows[0]["segment_symbol"])

        if session_epoch > sealed_window_end:
            latest = max(rows, key=lambda r: int(r["end_trade_date"]))
            if int(latest["end_trade_date"]) < sealed_window_end:
                raise RuntimeError(
                    f"canonical symbol {canonical_symbol} is closed before the sealed-window end {datetime.utcfromtimestamp(sealed_window_end).strftime('%Y-%m-%d')} and requires owner ruling"
                )
            if int(latest["end_trade_date"]) == sealed_window_end:
                return str(latest["segment_symbol"])
            raise RuntimeError(
                f"canonical symbol {canonical_symbol} has no valid open segment for {session_date}; latest segment ends {datetime.utcfromtimestamp(int(latest['end_trade_date'])).strftime('%Y-%m-%d')}"
            )

        raise RuntimeError(f"canonical symbol {canonical_symbol} has no active segment for {session_date}; owner ruling required")

    def append_session_rows(self, session_date: str, *, expected_symbol_count: int | None = None) -> dict[str, Any]:
        if not self.live_db_path.exists():
            raise FileNotFoundError(f"live market DB not found: {self.live_db_path}")

        self.ensure_surface_db()
        segment_map = self._segment_resolution_map(self.sealed_db_path)

        query = """
            SELECT ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at
            FROM ee_ohlcv_cache
            WHERE bar_date = ?
            ORDER BY ticker
        """
        with sqlite3.connect(str(self.live_db_path)) as source_conn:
            source_conn.row_factory = sqlite3.Row
            rows = list(source_conn.execute(query, (session_date,)))

        if not rows:
            return {
                "session_date": session_date,
                "rows_written": 0,
                "quarantined": 0,
                "total": 0,
                "run_key": self.run_key,
                "surface_db": str(self.surface_db_path),
                "calendar_version_id": self.surface_authority_ids["calendar_version_id"],
                "mask_manifest_version_id": self.surface_authority_ids["mask_manifest_version_id"],
            }

        inserted = 0
        quarantined = 0
        with sqlite3.connect(str(self.surface_db_path)) as conn:
            for row in rows:
                canonical_symbol = str(row["ticker"]).upper()
                try:
                    resolved_symbol = self._resolve_market_symbol(canonical_symbol, segment_map, session_date)
                except Exception as exc:
                    conn.execute(
                        "INSERT INTO forward_surface_quarantine (run_key, symbol, session, reason) VALUES (?, ?, ?, ?)",
                        (self.run_key, canonical_symbol, session_date, str(exc)),
                    )
                    quarantined += 1
                    continue

                payload = {
                    "symbol": resolved_symbol,
                    "session": str(row["bar_date"]),
                    "open": float(row["open"] or 0.0),
                    "high": float(row["high"] or 0.0),
                    "low": float(row["low"] or 0.0),
                    "close": float(row["close"] or 0.0),
                    "volume": float(row["volume"] or 0.0),
                    "turnover_kwd": float(row["turnover_kwd"] or 0.0),
                    "fetched_at": str(row["fetched_at"] or ""),
                    "calendar_version_id": self.surface_authority_ids["calendar_version_id"],
                    "mask_manifest_version_id": self.surface_authority_ids["mask_manifest_version_id"],
                }
                existing = conn.execute(
                    "SELECT 1 FROM forward_surface_rows WHERE run_key = ? AND symbol = ? AND trade_date = ?",
                    (self.run_key, resolved_symbol, session_date),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO forward_surface_rows (run_key, symbol, trade_date, row_json, calendar_version_id, mask_manifest_version_id, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            self.run_key,
                            resolved_symbol,
                            session_date,
                            json.dumps(payload, separators=(",", ":"), sort_keys=True),
                            self.surface_authority_ids["calendar_version_id"],
                            self.surface_authority_ids["mask_manifest_version_id"],
                            "READY",
                        ),
                    )
                    inserted += 1
            conn.commit()

        if expected_symbol_count is not None and inserted != expected_symbol_count:
            raise RuntimeError(f"forward surface expected {expected_symbol_count} rows for {session_date} but wrote {inserted}")

        return {
            "session_date": session_date,
            "rows_written": inserted,
            "quarantined": quarantined,
            "total": len(rows),
            "run_key": self.run_key,
            "surface_db": str(self.surface_db_path),
            "calendar_version_id": self.surface_authority_ids["calendar_version_id"],
            "mask_manifest_version_id": self.surface_authority_ids["mask_manifest_version_id"],
        }
