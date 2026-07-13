from __future__ import annotations

import sqlite3
from pathlib import Path

from app.services.eagle_eye.candidate_v2_service import (
    FAILPOINTS,
    LineageInfo,
    ensure_schema,
    ingest_symbol_rows,
)


def _sample_rows(n: int = 20) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "date": f"2025-01-{(i % 28) + 1:02d}",
                "open": 100 + i,
                "high": 101 + i,
                "low": 99 + i,
                "close": 100 + i,
                "volume": 1000 + i,
                "value": (100 + i) * (1000 + i),
            }
        )
    return rows


def _conn(tmp_path: Path) -> sqlite3.Connection:
    db = tmp_path / "candidate_v2_atomicity.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def test_failure_injection_points_are_supported() -> None:
    assert "25pct" in FAILPOINTS
    assert "50pct" in FAILPOINTS
    assert "90pct" in FAILPOINTS
    assert "after_raw_before_lineage" in FAILPOINTS
    assert "after_anomaly_before_completion" in FAILPOINTS


def test_failure_injection_rolls_back_all_partial_state(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    lineage = LineageInfo(commit="abc123", dirty=False, diff_hash=None)

    for fp in sorted(FAILPOINTS):
        result = ingest_symbol_rows(
            conn,
            symbol="TSTFAIL",
            rows=_sample_rows(30),
            lineage=lineage,
            failpoint=fp,
            environment="test",
            request_hash="reqhash",
        )

        assert result.status == "failed_transaction"

        run_id = result.run_id
        raw = conn.execute("SELECT COUNT(1) FROM ee_ohlcv_raw WHERE run_id=?", (run_id,)).fetchone()[0]
        adj = conn.execute(
            "SELECT COUNT(1) FROM ee_ohlcv_adjusted WHERE source_raw_identity LIKE ?",
            (f"TSTFAIL:%",),
        ).fetchone()[0]
        ind = conn.execute("SELECT COUNT(1) FROM ee_indicators_v2 WHERE symbol='TSTFAIL'").fetchone()[0]
        incomplete = conn.execute(
            "SELECT COUNT(1) FROM ee_symbol_reconciliation_v2 WHERE symbol='TSTFAIL'"
        ).fetchone()[0]

        assert raw == 0
        assert adj == 0
        assert ind == 0
        assert incomplete == 0

        status = conn.execute(
            "SELECT status FROM ee_ingestion_runs_v2 WHERE run_id=?",
            (run_id,),
        ).fetchone()[0]
        assert status != "completed"

        audit = conn.execute(
            "SELECT COUNT(1) FROM ee_failure_audit_v2 WHERE run_id=?",
            (run_id,),
        ).fetchone()[0]
        assert audit == 1

    conn.close()
