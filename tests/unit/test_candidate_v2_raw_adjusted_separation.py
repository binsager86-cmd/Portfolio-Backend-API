from __future__ import annotations

import sqlite3
from pathlib import Path

from app.services.eagle_eye.candidate_v2_service import (
    IDENTITY_ADJUSTMENT_VERSION,
    LineageInfo,
    ensure_schema,
    ingest_symbol_rows,
)


def _conn(tmp_path: Path) -> sqlite3.Connection:
    db = tmp_path / "candidate_v2_sep.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def _rows() -> list[dict]:
    return [
        {"date": "2025-01-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000, "value": 100000},
        {"date": "2025-01-02", "open": 102, "high": 103, "low": 101, "close": 102, "volume": 1100, "value": 112200},
        {"date": "2025-01-03", "open": 104, "high": 105, "low": 103, "close": 104, "volume": 1200, "value": 124800},
    ]


def test_raw_adjusted_physically_coexist(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    lineage = LineageInfo(commit="abc123", dirty=False, diff_hash=None)

    result = ingest_symbol_rows(
        conn,
        symbol="TSTSEP",
        rows=_rows(),
        lineage=lineage,
        failpoint=None,
        environment="test",
        request_hash="reqhash",
    )

    assert result.status == "completed"

    raw_count = conn.execute("SELECT COUNT(1) FROM ee_ohlcv_raw WHERE symbol='TSTSEP'").fetchone()[0]
    adj_count = conn.execute(
        "SELECT COUNT(1) FROM ee_ohlcv_adjusted WHERE symbol='TSTSEP' AND adjustment_version=?",
        (IDENTITY_ADJUSTMENT_VERSION,),
    ).fetchone()[0]

    assert raw_count == 3
    assert adj_count == 3

    conn.execute(
        """
        INSERT OR REPLACE INTO ee_ohlcv_adjusted (
            symbol, trade_date, adjustment_version, open, high, low, close,
            volume, value_kwd, source_raw_identity, source_raw_hash,
            corporate_action_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))
        """,
        (
            "TSTSEP",
            1735776000,
            "manual_adjustment_v2",
            50.0,
            50.5,
            49.5,
            50.0,
            1000.0,
            50000.0,
            "TSTSEP:1735776000:tickerchart:vendor_raw_v1",
            "h",
            "ca_v2",
        ),
    )
    conn.commit()

    raw_still = conn.execute(
        "SELECT close FROM ee_ohlcv_raw WHERE symbol='TSTSEP' AND trade_date=1735776000"
    ).fetchone()[0]
    adj_versions = conn.execute(
        "SELECT COUNT(DISTINCT adjustment_version) FROM ee_ohlcv_adjusted WHERE symbol='TSTSEP' AND trade_date=1735776000"
    ).fetchone()[0]

    assert float(raw_still) == 102.0
    assert adj_versions == 2

    conn.close()
