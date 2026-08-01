from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import r14e_module_e_lifecycle_intent_harness_v7 as v7


ROOT = Path(__file__).resolve().parents[1]
SEALED_DB = ROOT / "artifacts" / "preview1a_prestart" / "review_final" / "r12_exam_surface_v4_5_runtime.db"
SOURCE_REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
SANDBOX = Path(r"C:\ee_sandbox\harness_v41")
REQUIRED_SHA256 = "480e4186c17f2e4b20c3e8f2674f2f84c4bd4127eb25d5ca648d057a57bb037d"
RUN_NONCE = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
RUN_KEY = "R14E_MODULE_E_HARNESS_V41A"


def ro_uri(path: Path) -> str:
    return "file:" + path.resolve().as_posix() + "?mode=ro"


def connect_sealed_ro() -> sqlite3.Connection:
    return sqlite3.connect(ro_uri(SEALED_DB), uri=True)


def load_segment_map() -> dict[str, list[dict[str, Any]]]:
    with connect_sealed_ro() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT original_symbol, segment_symbol, segment_id, start_trade_date, end_trade_date, bars_count
            FROM ee_symbol_segment_map
            ORDER BY original_symbol, segment_id
            """
        ).fetchall()
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row["original_symbol"]), []).append(dict(row))
    return out


SEGMENT_MAP = load_segment_map()


def assert_sealed_input() -> dict[str, Any]:
    actual = v7.sha256_file(SEALED_DB)
    with connect_sealed_ro() as conn:
        canonical_count = conn.execute("SELECT COUNT(DISTINCT original_symbol) FROM ee_symbol_segment_map").fetchone()[0]
        segment_count = conn.execute("SELECT COUNT(DISTINCT segment_symbol) FROM ee_symbol_segment_map").fetchone()[0]
        absent = {
            symbol: conn.execute("SELECT COUNT(*) FROM ee_symbol_segment_map WHERE original_symbol = ?", (symbol,)).fetchone()[0]
            for symbol in ("ERESCO", "ONON")
        }
    if actual.lower() != REQUIRED_SHA256:
        raise RuntimeError(f"sealed input SHA-256 mismatch: actual={actual} required={REQUIRED_SHA256}")
    if int(canonical_count) != 139 or int(segment_count) != 309:
        raise RuntimeError(f"universe assertion failed: canonical={canonical_count} segment={segment_count}")
    if any(absent.values()):
        raise RuntimeError(f"ERESCO/ONON absence assertion failed: {absent}")
    return {
        "sealed_db": str(SEALED_DB),
        "required_sha256": REQUIRED_SHA256,
        "actual_sha256": actual.lower(),
        "sqlite_readonly_uri": ro_uri(SEALED_DB),
        "canonical_count": int(canonical_count),
        "segment_count": int(segment_count),
        "absent_assertions": absent,
        "target_segment_map": {symbol: SEGMENT_MAP.get(symbol, []) for symbol in ("SANAM", "TIJARA", "MABANEE")},
    }


def fetch_indicator_payload(conn: sqlite3.Connection, segment_symbol: str, trade_date: int) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT payload_json
        FROM ee_indicators
        WHERE symbol = ? AND trade_date = ?
        """,
        (segment_symbol, int(trade_date)),
    ).fetchone()
    if row is None or row[0] is None:
        return {}
    try:
        return json.loads(str(row[0]))
    except json.JSONDecodeError:
        return {}


def load_window(symbol: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    segments = [row["segment_symbol"] for row in SEGMENT_MAP.get(symbol, [])]
    if not segments:
        return []
    placeholders = ",".join("?" for _ in segments)
    with connect_sealed_ro() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT symbol, trade_date, open, high, low, close, volume, value_kwd
            FROM ee_ohlcv
            WHERE symbol IN ({placeholders})
              AND date(trade_date, 'unixepoch') BETWEEN ? AND ?
            ORDER BY trade_date ASC, symbol ASC
            """,
            (*segments, start_date, end_date),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            ts = int(row["trade_date"])
            segment_symbol = str(row["symbol"])
            out.append(
                {
                    "symbol": symbol,
                    "segment_symbol": segment_symbol,
                    "trade_date": v7.to_date_text(ts),
                    "trade_date_ts": ts,
                    "open": float(row["open"] or 0.0),
                    "high": float(row["high"] or 0.0),
                    "low": float(row["low"] or 0.0),
                    "close": float(row["close"] or 0.0),
                    "volume": float(row["volume"] or 0.0),
                    "value_kwd": float(row["value_kwd"] or 0.0),
                    "indicator_payload": fetch_indicator_payload(conn, segment_symbol, ts),
                }
            )
        return out


def count_ee_signals(symbol: str, start_date: str, end_date: str) -> dict[str, int]:
    segments = [row["segment_symbol"] for row in SEGMENT_MAP.get(symbol, [])]
    if not segments:
        return {"total_rows": 0, "avoid_stream_rows": 0}
    placeholders = ",".join("?" for _ in segments)
    start_i = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp())
    end_i = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp())
    with connect_sealed_ro() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM ee_signals WHERE symbol IN ({placeholders}) AND trade_date BETWEEN ? AND ?",
            (*segments, start_i, end_i),
        ).fetchone()[0]
        avoid = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM ee_signals
            WHERE symbol IN ({placeholders})
              AND trade_date BETWEEN ? AND ?
              AND (
                UPPER(COALESCE(signal_type, '')) LIKE '%AVOID%'
                OR UPPER(COALESCE(phase_from, '')) = 'AVOID'
                OR UPPER(COALESCE(phase_to, '')) = 'AVOID'
              )
            """,
            (*segments, start_i, end_i),
        ).fetchone()[0]
    return {"total_rows": int(total), "avoid_stream_rows": int(avoid)}


def r12_avoid_intervals(symbol: str) -> list[dict[str, Any]]:
    segments = [row["segment_symbol"] for row in SEGMENT_MAP.get(symbol, [])]
    if not segments:
        return []
    placeholders = ",".join("?" for _ in segments)
    with connect_sealed_ro() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT trade_date, signal_type, phase_from, phase_to
            FROM ee_signals
            WHERE symbol IN ({placeholders})
              AND (
                UPPER(COALESCE(signal_type, '')) LIKE '%AVOID%'
                OR UPPER(COALESCE(phase_from, '')) = 'AVOID'
                OR UPPER(COALESCE(phase_to, '')) = 'AVOID'
              )
            ORDER BY trade_date ASC
            """,
            tuple(segments),
        ).fetchall()
    intervals: list[dict[str, Any]] = []
    open_start: str | None = None
    for row in rows:
        date_text = v7.to_date_text(int(row["trade_date"]))
        phase_to = str(row["phase_to"] or "").upper()
        phase_from = str(row["phase_from"] or "").upper()
        signal_type = str(row["signal_type"] or "").upper()
        if phase_to == "AVOID" or "AVOID_SET" in signal_type:
            open_start = date_text
        elif phase_from == "AVOID" and open_start is not None:
            intervals.append({"start": open_start, "end": date_text, "clear_event": date_text})
            open_start = None
    if open_start is not None:
        intervals.append({"start": open_start, "end": None, "clear_event": None})
    return intervals


def owner_windows() -> dict[str, dict[str, Any]]:
    return {
        "MABANEE": {
            "owner_start": "2025-12-01",
            "owner_end": "2026-07-09",
            "replay_start": "2025-07-01",
            "replay_end": "2026-07-09",
            "source_artifact": "v4.1-A known-answer refresh: decline terminal zone approx 919-920; avoid-plane acceptance required.",
        },
        "SANAM": {
            "owner_start": "2026-04-01",
            "owner_end": "2026-07-09",
            "replay_start": "2025-10-01",
            "replay_end": "2026-07-09",
            "source_artifact": "v4.1-A known-answer refresh: close approx 360 on 2026-07-09; markup origin approx 207 in April 2026.",
        },
        "TIJARA": {
            "owner_start": "2026-01-01",
            "owner_end": "2026-07-09",
            "replay_start": "2025-07-01",
            "replay_end": "2026-07-09",
            "source_artifact": "v4.1-A known-answer refresh: close approx 181 on 2026-07-09; 185 print on 2026-07-07; shakeout stress-test reporting stands.",
        },
    }


def configure_v7() -> dict[str, Any]:
    SANDBOX.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_REVIEW / "r14b_parameter_freeze_v2.json", SANDBOX / "r14b_parameter_freeze_v2.json")
    shutil.copy2(SOURCE_REVIEW / "r14b_parameter_freeze_v2.sha256", SANDBOX / "r14b_parameter_freeze_v2.sha256")

    v7.REVIEW = SANDBOX
    v7.RUNTIME_DB = SEALED_DB
    v7.HARNESS_DB = SANDBOX / f"harness_v41_{RUN_NONCE.replace(':', '').replace('.', '_')}.db"
    v7.FREEZE_JSON = SANDBOX / "r14b_parameter_freeze_v2.json"
    v7.FREEZE_SHA = SANDBOX / "r14b_parameter_freeze_v2.sha256"
    v7.RUN_NONCE = RUN_NONCE
    v7.RUN_KEY = RUN_KEY
    v7.load_window = load_window
    v7.count_ee_signals = count_ee_signals
    v7.r12_avoid_intervals = r12_avoid_intervals
    v7.owner_windows = owner_windows
    return {"sandbox": str(SANDBOX), "harness_db": str(v7.HARNESS_DB), "run_nonce": RUN_NONCE}


def main() -> None:
    sealed_attestation = assert_sealed_input()
    config = configure_v7()
    print("PHASE_0_3_A_SEALED_INPUT")
    print(json.dumps(sealed_attestation, ensure_ascii=True, indent=2, sort_keys=True))
    print("HARNESS_V41A_CONFIG")
    print(json.dumps(config, ensure_ascii=True, indent=2, sort_keys=True))
    v7.main()


if __name__ == "__main__":
    main()