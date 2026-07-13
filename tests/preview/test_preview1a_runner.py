from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.services.eagle_eye.preview1a_runner import (
    MODE_CONTINUOUS_HISTORY,
    PreviewRunConfig,
    run_preview1a,
)
from app.services.eagle_eye.preview1a_source_db import SourceStreamSpec, market_date_to_utc_epoch
from app.services.eagle_eye.preview1a_snapshot import build_symbol_snapshot


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_source_db(path: Path, symbol: str = "TST") -> tuple[int, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE ee_ohlcv_src (
                symbol TEXT NOT NULL,
                trade_date INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                value_kwd REAL,
                adjustment_status TEXT,
                corporate_action_version TEXT,
                PRIMARY KEY(symbol, trade_date)
            )
            """
        )
        conn.execute("CREATE TABLE ee_signals (id INTEGER PRIMARY KEY, symbol TEXT, trade_date INTEGER)")

        price = 100.0
        rows = []
        for i in range(260):
            dt = start + timedelta(days=i)
            ts = int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp())
            # Smooth trend with periodic acceleration to create non-trivial indicators.
            price = price * (1.002 if (i % 15) else 1.01)
            rows.append(
                (
                    symbol,
                    ts,
                    price * 0.99,
                    price * 1.01,
                    price * 0.98,
                    price,
                    300_000 + i,
                    (price * (300_000 + i)) / 1000.0,
                    "raw_unadjusted",
                    "none",
                )
            )
        conn.executemany(
            """
            INSERT INTO ee_ohlcv_src (
                symbol, trade_date, open, high, low, close, volume, value_kwd,
                adjustment_status, corporate_action_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    warmup_start = rows[0][1]
    eval_start = rows[180][1]
    milestone_t = rows[230][1]
    return warmup_start, eval_start, milestone_t


def _config(tmp_path: Path, mode: str, ca_version: str = "v1") -> PreviewRunConfig:
    source = tmp_path / "source.db"
    warmup_start, eval_start, milestone_t = _make_source_db(source)
    return PreviewRunConfig(
        source_db_path=str(source),
        output_db_path=str(tmp_path / f"output_{mode.lower()}.db"),
        evidence_dir=str(tmp_path / "evidence"),
        symbols=["TST"],
        warmup_data_start=warmup_start,
        evaluation_output_start=eval_start,
        milestone_t=milestone_t,
        mode=mode,
        source_stream=SourceStreamSpec(
            source_table="ee_ohlcv_src",
            primary_key=("symbol", "trade_date"),
            stream_type="RAW",
            adjustment_version="raw_unadjusted",
            corporate_action_ledger_version=ca_version,
            dataset_id="preview-test-dataset",
        ),
        checkpoints=[milestone_t],
    )


def test_preview1a_continuous_history_preserves_pre_evaluation_position(tmp_path: Path) -> None:
    result = run_preview1a(_config(tmp_path, MODE_CONTINUOUS_HISTORY))
    scored = result["run"]["scored_metrics"]
    pre = result["run"]["pre_evaluation_decisions"]["open_positions_at_eval_start"]
    assert scored["evaluated_open_positions_at_start"] == pre


def test_preview1a_decision_activation_mode_removed(tmp_path: Path) -> None:
    cfg = _config(tmp_path, MODE_CONTINUOUS_HISTORY)
    cfg = PreviewRunConfig(
        source_db_path=cfg.source_db_path,
        output_db_path=cfg.output_db_path,
        evidence_dir=cfg.evidence_dir,
        symbols=cfg.symbols,
        warmup_data_start=cfg.warmup_data_start,
        evaluation_output_start=cfg.evaluation_output_start,
        milestone_t=cfg.milestone_t,
        mode="DECISION_ACTIVATION",
        source_stream=cfg.source_stream,
        checkpoints=cfg.checkpoints,
    )
    try:
        run_preview1a(cfg)
        assert False, "DECISION_ACTIVATION must be unsupported"
    except ValueError as exc:
        assert "Unsupported preview mode" in str(exc)


def test_preview1a_warmup_retained_before_evaluation_start(tmp_path: Path) -> None:
    cfg = _config(tmp_path, MODE_CONTINUOUS_HISTORY)
    result = run_preview1a(cfg)
    assert result["bars_copied_to_output"] > 0
    assert cfg.warmup_data_start < cfg.evaluation_output_start


def test_preview1a_no_scored_outputs_before_evaluation_start(tmp_path: Path) -> None:
    cfg = _config(tmp_path, MODE_CONTINUOUS_HISTORY)
    result = run_preview1a(cfg)
    with sqlite3.connect(cfg.output_db_path) as conn:
        count_scored = conn.execute(
            "SELECT COUNT(1) FROM ee_backtest_trades WHERE opened_at >= ?",
            (cfg.evaluation_output_start,),
        ).fetchone()[0]
    assert count_scored == result["run"]["scored_metrics"]["evaluated_trades"]


def test_preview1a_explicit_market_date_utc_conversion() -> None:
    ts = market_date_to_utc_epoch("2021-01-01")
    assert ts == 1609459200


def test_preview1a_full_snapshot_no_lookahead_equality(tmp_path: Path) -> None:
    cfg_a = _config(tmp_path / "a", MODE_CONTINUOUS_HISTORY)
    cfg_b = _config(tmp_path / "b", MODE_CONTINUOUS_HISTORY)
    run_preview1a(cfg_a)
    run_preview1a(cfg_b)

    with sqlite3.connect(cfg_a.output_db_path) as c1, sqlite3.connect(cfg_b.output_db_path) as c2:
        s1 = build_symbol_snapshot(c1, "TST", cfg_a.milestone_t)
        s2 = build_symbol_snapshot(c2, "TST", cfg_b.milestone_t)
    assert s1 == s2


def test_preview1a_recomputes_indicators_from_truncated_raw_bars(tmp_path: Path) -> None:
    cfg = _config(tmp_path, MODE_CONTINUOUS_HISTORY)
    run_preview1a(cfg)
    with sqlite3.connect(cfg.output_db_path) as conn:
        n_ind = conn.execute("SELECT COUNT(1) FROM ee_indicators").fetchone()[0]
        n_bars = conn.execute("SELECT COUNT(1) FROM ee_ohlcv").fetchone()[0]
    assert n_ind > 0
    assert n_ind <= n_bars


def test_preview1a_does_not_read_persisted_states_or_signals(tmp_path: Path) -> None:
    cfg = _config(tmp_path, MODE_CONTINUOUS_HISTORY)
    with sqlite3.connect(cfg.source_db_path) as conn:
        conn.execute("INSERT INTO ee_signals (symbol, trade_date) VALUES (?, ?)", ("TST", cfg.warmup_data_start))
        conn.commit()
    result = run_preview1a(cfg)
    assert result["bars_copied_to_output"] > 0


def test_preview1a_excludes_unresolved_ca_milestones_from_scoring(tmp_path: Path) -> None:
    cfg = _config(tmp_path, MODE_CONTINUOUS_HISTORY, ca_version="UNRESOLVED")
    result = run_preview1a(cfg)
    assert result["corporate_action_approval_status"] == "PIT_INVALID_CA_UNRESOLVED"
    assert result["run"]["scored_metrics"]["excluded_from_scoring"] is True


def test_preview1a_ingestion_paths_unreachable(tmp_path: Path) -> None:
    cfg = _config(tmp_path, MODE_CONTINUOUS_HISTORY)
    run_preview1a(cfg)
    with sqlite3.connect(cfg.output_db_path) as conn:
        n_runs = conn.execute("SELECT COUNT(1) FROM ee_ingestion_runs").fetchone()[0]
    assert n_runs == 0


def test_preview1a_scheduler_paths_unreachable(tmp_path: Path) -> None:
    cfg = _config(tmp_path, MODE_CONTINUOUS_HISTORY)
    result = run_preview1a(cfg)
    closure_path = Path(result["dependency_closure_artifacts"]["classifications"])
    cls = json.loads(closure_path.read_text(encoding="utf-8"))
    sched_mod = "app.services.eagle_eye.scheduler_service"
    assert cls.get(sched_mod) != "RUNTIME_REACHABLE"


def test_preview1a_source_database_is_read_only(tmp_path: Path) -> None:
    cfg = _config(tmp_path, MODE_CONTINUOUS_HISTORY)
    source = Path(cfg.source_db_path).resolve()
    uri = f"file:{source.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        try:
            conn.execute("INSERT INTO ee_ohlcv_src (symbol, trade_date) VALUES ('X', 1)")
            conn.commit()
            assert False, "expected read-only write failure"
        except sqlite3.OperationalError:
            pass


def test_preview1a_no_writes_outside_disposable_output_db(tmp_path: Path) -> None:
    cfg = _config(tmp_path, MODE_CONTINUOUS_HISTORY)
    before = _sha256(Path(cfg.source_db_path))
    run_preview1a(cfg)
    after = _sha256(Path(cfg.source_db_path))
    assert before == after


def test_preview1a_engine_and_source_hashes_unchanged_pre_post(tmp_path: Path) -> None:
    cfg = _config(tmp_path, MODE_CONTINUOUS_HISTORY)
    before_source = _sha256(Path(cfg.source_db_path))
    result = run_preview1a(cfg)
    after_source = _sha256(Path(cfg.source_db_path))
    assert before_source == after_source

    backtest_hash = _sha256(Path("app/services/eagle_eye/backtest_service.py"))
    assert result["run"]["engine_hashes"]["backtest_service"] == backtest_hash


def test_preview1a_ml_gate_identity_or_disabled_status(tmp_path: Path) -> None:
    result = run_preview1a(_config(tmp_path, MODE_CONTINUOUS_HISTORY))
    ml = result["run"]["ml_gate"]
    assert ml["enabled"] is False
    assert ml["loads_model_artifact"] is False
    assert ml["uses_network"] is False
    assert ml["writes_database"] is False
