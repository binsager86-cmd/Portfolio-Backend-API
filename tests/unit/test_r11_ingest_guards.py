from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.core.database import exec_sql, query_all, query_one, query_val
from app.services.eagle_eye.audit_service import ensure_schema as ensure_audit_schema
from app.services.eagle_eye import market_data_service as mds


@pytest.fixture(autouse=True)
def _reset_ingest_tables() -> None:
    mds.ensure_schema()
    ensure_audit_schema()
    for table in [
        "ee_ohlcv",
        "ee_ingestion_runs",
        "ee_data_quality_quarantine",
        "ee_audit_events",
    ]:
        try:
            exec_sql(f"DELETE FROM {table}", ())
        except Exception:
            pass


def _write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    header = "date,open,high,low,close,volume,value\n"
    body = "\n".join(
        f"{r['date']},{r['open']},{r['high']},{r['low']},{r['close']},{r['volume']},{r['value']}" for r in rows
    )
    path.write_text(header + body + "\n", encoding="utf-8")


def _trusted_upsert(symbol: str, trade_date: int, close_v: float, payload_hash: str, adjustment_status: str = "raw_unadjusted") -> bool:
    run_id = mds._begin_ingestion_run(
        source_type="vendor_raw",
        source_ref="test:trusted",
        payload_hash=payload_hash,
        request_parameters_hash="req-hash",
        synthetic_flag=0,
    )
    wrote = mds._upsert_ohlcv_row(
        symbol=symbol,
        trade_date=trade_date,
        open_v=close_v,
        high_v=close_v,
        low_v=close_v,
        close_v=close_v,
        volume_v=100.0,
        value_kwd_v=100.0,
        source="tickerchart",
        source_type="vendor_raw",
        source_ref="test:trusted",
        run_id=run_id,
        request_parameters_hash="req-hash",
        payload_hash=payload_hash,
        synthetic_flag=0,
        adjustment_status=adjustment_status,
        corporate_action_version="none",
    )
    mds._finalize_ingestion_run(run_id, 1 if wrote else 0, status="completed")
    return wrote


def test_synthetic_source_real_symbol_rejected(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic_real_symbol.csv"
    _write_csv(csv_path, [{"date": "01/01/2024", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000, "value": 10000}])

    with pytest.raises(HTTPException, match="Synthetic/debug fixture symbol is not allowed"):
        mds.load_ohlcv_csv(str(csv_path), "TIJARA", source="synthetic_loader")


def test_debug_source_real_symbol_rejected_even_with_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_DEBUG_FIXTURE_WRITE", "1")
    csv_path = tmp_path / "debug_rows.csv"
    _write_csv(csv_path, [{"date": "01/01/2024", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000, "value": 10000}])

    with pytest.raises(HTTPException, match="Synthetic/debug fixture symbol is not allowed"):
        mds.load_ohlcv_csv(str(csv_path), "BPCC", source="debug_feed")


def test_api_csv_import_path_rejects_synthetic_source_on_real_symbol(tmp_path: Path) -> None:
    csv_path = tmp_path / "api_upload.csv"
    _write_csv(csv_path, [{"date": "01/01/2024", "open": 20, "high": 22, "low": 19, "close": 21, "volume": 2000, "value": 20000}])

    with pytest.raises(HTTPException, match="Synthetic/debug fixture symbol is not allowed"):
        mds.load_ohlcv_csv(str(csv_path), "ZAIN", source="synthetic_api")


def test_existing_tickerchart_row_preserved_against_fixture_overwrite(tmp_path: Path) -> None:
    td = 1704067200
    exec_sql(
        """
        INSERT INTO ee_ohlcv (
            symbol, trade_date, open, high, low, close, raw_close, adjusted_close,
            volume, value_kwd, value_unit, source, source_type, source_ref,
            data_environment, ingestion_run_id, request_parameters_hash,
            payload_hash, code_commit, parser_version, synthetic_flag,
            adjustment_status, corporate_action_version, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "TST_KEEP",
            td,
            100.0,
            101.0,
            99.0,
            100.0,
            100.0,
            100.0,
            1000.0,
            500.0,
            "kwd",
            "tickerchart",
            "vendor_raw",
            "tc:seed",
            "production",
            "seed-run",
            "seed-req",
            "seed-hash",
            "seed-commit",
            "seed-parser",
            0,
            "raw_unadjusted",
            "none",
            mds.now_ts(),
        ),
    )

    csv_path = tmp_path / "synthetic_tst_keep.csv"
    _write_csv(csv_path, [{"date": "01/01/2024", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000, "value": 10000}])
    out = mds.load_ohlcv_csv(str(csv_path), "TST_KEEP", source="synthetic_fixture")

    assert out["rows"] == 0
    row = query_one("SELECT source_type, close FROM ee_ohlcv WHERE symbol='TST_KEEP' AND trade_date=?", (td,))
    assert row is not None
    assert str(row["source_type"]) == "vendor_raw"
    assert float(row["close"]) == 100.0


def test_rejected_overwrite_creates_audit_event(tmp_path: Path) -> None:
    td = 1704067200
    _trusted_upsert("TST_AUDIT", td, 100.0, payload_hash="trusted-hash")

    csv_path = tmp_path / "synthetic_tst_audit.csv"
    _write_csv(csv_path, [{"date": "01/01/2024", "open": 9, "high": 10, "low": 8, "close": 9.5, "volume": 1000, "value": 10000}])
    mds.load_ohlcv_csv(str(csv_path), "TST_AUDIT", source="synthetic_fixture")

    event = query_one(
        "SELECT action, metadata_json FROM ee_audit_events WHERE action='data_ingest_rejected' ORDER BY id DESC LIMIT 1",
        (),
    )
    assert event is not None
    assert str(event["action"]) == "data_ingest_rejected"
    assert "trusted_row_preserved_against_fixture_or_debug" in str(event["metadata_json"])


def test_trusted_reingestion_identical_payload_is_idempotent() -> None:
    td = 1704067200
    payload = hashlib.sha256(b"same-payload").hexdigest()

    wrote1 = _trusted_upsert("TST_IDEMP", td, 100.0, payload_hash=payload)
    wrote2 = _trusted_upsert("TST_IDEMP", td, 100.0, payload_hash=payload)

    assert wrote1 is True
    assert wrote2 is False
    count = int(query_val("SELECT COUNT(1) FROM ee_ohlcv WHERE symbol='TST_IDEMP' AND trade_date=?", (td,)) or 0)
    assert count == 1


def test_conflicting_trusted_payload_is_quarantined() -> None:
    td = 1704067200
    h1 = hashlib.sha256(b"trusted-v1").hexdigest()
    h2 = hashlib.sha256(b"trusted-v2").hexdigest()

    wrote1 = _trusted_upsert("TST_CONFLICT", td, 100.0, payload_hash=h1)
    wrote2 = _trusted_upsert("TST_CONFLICT", td, 110.0, payload_hash=h2)

    assert wrote1 is True
    assert wrote2 is False

    q = query_one("SELECT status, reason_json FROM ee_data_quality_quarantine WHERE symbol='TST_CONFLICT'", ())
    assert q is not None
    assert str(q["status"]) == "quarantined"
    reason = json.loads(str(q["reason_json"]))
    assert reason and reason[0]["type"] == "trusted_payload_conflict"


def test_raw_and_adjusted_rows_do_not_cross_overwrite() -> None:
    td = 1704067200
    h1 = hashlib.sha256(b"raw").hexdigest()
    h2 = hashlib.sha256(b"adjusted").hexdigest()

    wrote1 = _trusted_upsert("TST_ADJ", td, 100.0, payload_hash=h1, adjustment_status="raw_unadjusted")
    wrote2 = _trusted_upsert("TST_ADJ", td, 105.0, payload_hash=h2, adjustment_status="adjusted")

    assert wrote1 is True
    assert wrote2 is False
    row = query_one("SELECT close, adjustment_status FROM ee_ohlcv WHERE symbol='TST_ADJ' AND trade_date=?", (td,))
    assert row is not None
    assert float(row["close"]) == 100.0
    assert str(row["adjustment_status"]) == "raw_unadjusted"


def test_failed_csv_batch_rolls_back_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "tst_batch.csv"
    _write_csv(
        csv_path,
        [
            {"date": "01/01/2024", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000, "value": 10000},
            {"date": "02/01/2024", "open": 11, "high": 12, "low": 10, "close": 11.5, "volume": 1200, "value": 12000},
        ],
    )
    monkeypatch.setenv("EE_FAIL_BATCH_AFTER_ROWS", "1")

    with pytest.raises(RuntimeError, match="EE_FORCED_BATCH_FAILURE"):
        mds.load_ohlcv_csv(str(csv_path), "TST_BATCH", source="csv")

    count = int(query_val("SELECT COUNT(1) FROM ee_ohlcv WHERE symbol='TST_BATCH'", ()) or 0)
    assert count == 0
    run = query_one(
        "SELECT status, rows_written FROM ee_ingestion_runs WHERE source_ref LIKE ? ORDER BY started_at DESC LIMIT 1",
        (f"file:{csv_path.resolve().as_posix()}",),
    )
    assert run is not None
    assert str(run["status"]) == "failed"
    assert int(run["rows_written"] or 0) == 0
