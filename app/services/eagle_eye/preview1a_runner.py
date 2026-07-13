from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.eagle_eye.preview1a_dependency_closure import write_dependency_closure_artifacts
from app.services.eagle_eye.preview1a_source_db import (
    SourceStreamSpec,
    extract_bars_from_source,
    initialize_output_ohlcv,
    load_bars_into_output,
)

MODE_CONTINUOUS_HISTORY = "CONTINUOUS_HISTORY"

CA_STATUS_ADJUSTED_APPROVED = "ADJUSTED_APPROVED"
CA_STATUS_RAW_DIAGNOSTIC_ONLY = "RAW_DIAGNOSTIC_ONLY"
CA_STATUS_PIT_INVALID = "PIT_INVALID_CA_UNRESOLVED"
CA_STATUS_DISTORTED = "DISTORTED_BY_UNADJUSTED_CA"


@dataclass(frozen=True)
class PreviewRunConfig:
    source_db_path: str
    output_db_path: str
    evidence_dir: str
    symbols: list[str]
    warmup_data_start: int
    evaluation_output_start: int
    milestone_t: int
    mode: str
    source_stream: SourceStreamSpec
    checkpoints: list[int]


def _run_worker(config: PreviewRunConfig) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="preview1a_spec_") as td:
        spec_path = Path(td) / "spec.json"
        out_path = Path(td) / "result.json"
        spec_payload = {
            "output_db_path": config.output_db_path,
            "symbols": [s.upper() for s in config.symbols],
            "warmup_data_start": int(config.warmup_data_start),
            "evaluation_output_start": int(config.evaluation_output_start),
            "milestone_t": int(config.milestone_t),
            "mode": config.mode,
            "checkpoints": [int(x) for x in config.checkpoints],
        }
        spec_path.write_text(json.dumps(spec_payload, ensure_ascii=True), encoding="utf-8")

        env = dict(os.environ)
        env["DATABASE_PATH"] = config.output_db_path
        env["ENVIRONMENT"] = "test"
        env["PYTEST_ALLOW_NON_TEMP_DB"] = "1"

        cmd = [
            sys.executable,
            "-m",
            "app.services.eagle_eye.preview1a_worker",
            "--spec",
            str(spec_path),
            "--out",
            str(out_path),
        ]
        cp = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
        if cp.returncode != 0:
            raise RuntimeError(
                "preview1a_worker failed"
                f"\nexit={cp.returncode}"
                f"\nstdout:\n{cp.stdout}"
                f"\nstderr:\n{cp.stderr}"
            )
        result = json.loads(out_path.read_text(encoding="utf-8"))
        result["worker_stdout"] = cp.stdout
        result["worker_stderr"] = cp.stderr
        result["worker_exit_code"] = cp.returncode
        return result


def run_preview1a(config: PreviewRunConfig) -> dict[str, Any]:
    if config.mode != MODE_CONTINUOUS_HISTORY:
        raise ValueError(f"Unsupported preview mode: {config.mode}")

    Path(config.evidence_dir).mkdir(parents=True, exist_ok=True)

    bars = extract_bars_from_source(
        source_db_path=config.source_db_path,
        stream=config.source_stream,
        symbols=config.symbols,
        warmup_data_start=config.warmup_data_start,
        milestone_t=config.milestone_t,
    )

    initialize_output_ohlcv(config.output_db_path)
    inserted = load_bars_into_output(config.output_db_path, bars)

    run_result = _run_worker(config)
    runtime_trace_path = Path(config.evidence_dir) / "preview1a_runtime_import_trace.json"
    runtime_trace_path.write_text(
        json.dumps(run_result.get("runtime_import_trace", {}), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    closure_paths = write_dependency_closure_artifacts(
        repo_root=str(Path(__file__).resolve().parents[3]),
        entry_modules=["app.services.eagle_eye.backtest_service", "app.services.eagle_eye.pipeline"],
        out_dir=config.evidence_dir,
    )

    if config.source_stream.corporate_action_ledger_version.upper() in {"UNRESOLVED", "UNKNOWN", "PENDING"}:
        ca_status = CA_STATUS_PIT_INVALID
    elif config.source_stream.stream_type.upper().startswith("RAW"):
        ca_status = CA_STATUS_RAW_DIAGNOSTIC_ONLY
    else:
        ca_status = CA_STATUS_ADJUSTED_APPROVED

    if ca_status == CA_STATUS_PIT_INVALID:
        run_result.setdefault("scored_metrics", {})["excluded_from_scoring"] = True

    package = {
        "status": "ok",
        "classification": "HISTORICAL_REPLAY_NON_CANONICAL",
        "preview_report_header": {
            "ml_gate_default_behavior": "INERT_UNDER_DEFAULTS_NO_LABELED_SIGNALS",
            "ml_gate_default_explanation": "ml_gate_enabled defaults to false; apply_ml_gate returns (True, None), so baseline behavior does not block or alter entry/exit decisions.",
        },
        "mode": config.mode,
        "source_stream": {
            "source_table": config.source_stream.source_table,
            "primary_key": list(config.source_stream.primary_key),
            "stream_type": config.source_stream.stream_type,
            "adjustment_version": config.source_stream.adjustment_version,
            "corporate_action_ledger_version": config.source_stream.corporate_action_ledger_version,
            "dataset_id": config.source_stream.dataset_id,
        },
        "bars_copied_to_output": inserted,
        "run": run_result,
        "corporate_action_approval_status": ca_status,
        "corporate_action_knowledge_as_of": config.source_stream.corporate_action_ledger_version,
        "calculation_as_of": int(config.milestone_t),
        "dependency_closure_artifacts": closure_paths,
        "runtime_import_trace_artifact": str(runtime_trace_path),
        "source_database_read_only": True,
        "production_write_block": True,
    }

    out = Path(config.evidence_dir) / f"preview1a_result_{config.mode.lower()}.json"
    out.write_text(json.dumps(package, indent=2, ensure_ascii=True), encoding="utf-8")
    return package
