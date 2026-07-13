from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class WorkerSpec:
    output_db_path: str
    symbols: list[str]
    warmup_data_start: int
    evaluation_output_start: int
    milestone_t: int
    mode: str
    checkpoints: list[int]


def _hash_file(path: str) -> str:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_spec(path: str) -> WorkerSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return WorkerSpec(**payload)


def _collect_node_ids() -> list[str]:
    from app.core.database import query_all

    rows = query_all("SELECT id FROM ee_signals ORDER BY id")
    return [f"signal::{int(r['id'])}" for r in rows]


def run_worker(spec: WorkerSpec) -> dict[str, Any]:
    os.environ["DATABASE_PATH"] = spec.output_db_path
    os.environ["ENVIRONMENT"] = "test"
    pre_imports = set(sys.modules.keys())

    from app.core.schema import ensure_all_tables
    from app.core.database import query_all, query_val
    from app.services.eagle_eye.market_data_service import ensure_schema, get_active_config
    from app.services.eagle_eye.audit_service import ensure_schema as ensure_audit_schema
    from app.services.eagle_eye.backtest_service import run_backtest
    from app.services.eagle_eye.preview1a_ml_gate import classify_ml_gate_behavior, hash_ml_gate_identity

    ensure_all_tables()
    ensure_schema()
    ensure_audit_schema()

    cfg = get_active_config()
    ml_identity = classify_ml_gate_behavior(cfg)

    report = run_backtest(
        symbols=spec.symbols,
        start=int(spec.warmup_data_start),
        end=int(spec.milestone_t),
    )
    post_imports = set(sys.modules.keys())
    loaded = sorted(post_imports - pre_imports)
    runtime_import_trace = {
        "new_modules": loaded,
        "app_modules": [m for m in loaded if m.startswith("app.")],
    }

    pre_eval_signals = int(
        query_val(
            "SELECT COUNT(1) FROM ee_signals WHERE trade_date < ?",
            (int(spec.evaluation_output_start),),
        )
        or 0
    )
    pre_eval_open_positions = int(
        query_val(
            "SELECT COUNT(1) FROM ee_backtest_trades WHERE opened_at < ? AND (closed_at IS NULL OR closed_at >= ?)",
            (int(spec.evaluation_output_start), int(spec.evaluation_output_start)),
        )
        or 0
    )

    evaluated_trade_rows = query_all(
        """
        SELECT opened_at, closed_at, net_return
        FROM ee_backtest_trades
        WHERE opened_at >= ?
        ORDER BY opened_at
        """,
        (int(spec.evaluation_output_start),),
    )

    evaluated_returns = [float(r.get("net_return") or 0.0) for r in evaluated_trade_rows]
    scored = {
        "evaluated_trades": len(evaluated_trade_rows),
        "evaluated_expectancy": (sum(evaluated_returns) / len(evaluated_returns)) if evaluated_returns else 0.0,
        "evaluated_open_positions_at_start": pre_eval_open_positions,
    }

    return {
        "mode": spec.mode,
        "report": report,
        "scored_metrics": scored,
        "pre_evaluation_decisions": {
            "signals": pre_eval_signals,
            "open_positions_at_eval_start": pre_eval_open_positions,
        },
        "ml_gate": {
            **ml_identity.__dict__,
            "identity_hash": hash_ml_gate_identity(ml_identity),
        },
        "runtime_import_trace": runtime_import_trace,
        "node_ids": _collect_node_ids(),
        "engine_hashes": {
            "backtest_service": _hash_file("app/services/eagle_eye/backtest_service.py"),
            "pipeline": _hash_file("app/services/eagle_eye/pipeline.py"),
            "scanner_service": _hash_file("app/services/eagle_eye/scanner_service.py"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PREVIEW-1A worker")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = run_worker(_load_spec(args.spec))
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
