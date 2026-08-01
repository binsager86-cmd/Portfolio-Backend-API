from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import r16_3_candidate_state_machine as sm
import r16_3_harness_v53 as h

SWEEP_SANDBOX = Path(r"C:\ee_sandbox\harness_v53_rat1")
SWEEP_EXPORT = SWEEP_SANDBOX / "export"
ARCHIVE_DIR = Path(r"F:\eagle_eye_archive\rat1_sweep")
RATIFIED = {
    "pivot_k": 3,
    "pivot_sig_atr": 1.5,
    "mfe_waiver": 0.08,
    "c3_recovery_sessions": 1,
}


@dataclass(frozen=True)
class SweepPoint:
    label: str
    pivot_k: int = 3
    pivot_sig_atr: float = 1.5
    mfe_waiver: float = 0.08
    c3_recovery_sessions: int = 1


def sweep_points() -> list[SweepPoint]:
    points = [SweepPoint("RATIFIED")]
    points.extend(SweepPoint(f"PIVOT_K={value}", pivot_k=value) for value in (2, 4))
    points.extend(SweepPoint(f"PIVOT_SIG_ATR={value}", pivot_sig_atr=value) for value in (1.2, 1.8))
    points.extend(SweepPoint(f"MFE_WAIVER={value:.0%}", mfe_waiver=value) for value in (0.06, 0.10))
    points.extend(SweepPoint(f"C3_RECOVERY_SESSIONS={value}", c3_recovery_sessions=value) for value in (0, 2))
    return points


def configure(point: SweepPoint) -> None:
    h.PIVOT_CONFIRMATION_LAG_SESSIONS = point.pivot_k
    h.SIGNIFICANT_PIVOT_ATR_MULT = point.pivot_sig_atr
    sm.TIME_STOP_MFE_WAIVER_PCT = point.mfe_waiver
    sm.C3_REENTRY_RECOVERY_SESSIONS = point.c3_recovery_sessions
    h.SANDBOX = SWEEP_SANDBOX
    h.EXPORT = SWEEP_EXPORT
    h.RUN_KEY_PREFIX = "R16_RAT_1_SWEEP_V53A"


def parse_universe_lines(lines: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in lines:
        if line.startswith("F5_ROW|"):
            payload = json.loads(line.split("|", 1)[1])
            out[str(payload["symbol"])] = payload
    return out


def collect_metrics(result: dict[str, Any], baseline_sum: dict[str, Any]) -> dict[str, Any]:
    with sqlite3.connect(result["harness_db"]) as conn:
        universe = parse_universe_lines(h.universe_rows(conn, result["run_key"])).get("GLOBAL", {})
        mabanee = h.load_daily(conn, result["run_key"], "MABANEE", "2025-12-01", "2026-07-09")
        first_soft = next((row["date"] for row in mabanee if row.get("avoid_tier") == "AVOID_SOFT"), None)
        exposure = [row for row in mabanee if row.get("position")]
        exposure_below_1000 = [row for row in exposure if float(row.get("close") or 0.0) < 1000.0]
        sanam_daily = h.load_daily(conn, result["run_key"], "SANAM", "2026-01-01", "2026-07-09")
        sanam_events = h.load_events(conn, result["run_key"], "SANAM", "2026-01-01", "2026-07-09")
        sanam_first = next((event for event in sanam_events if event.get("event_type") == "POSITION_OPENED"), None)
        sanam_capture = h.position_capture(sanam_events, sanam_daily, "SANAM")
        tijara_daily = h.load_daily(conn, result["run_key"], "TIJARA", "2026-01-01", "2026-07-09")
        tijara_events_2026 = h.load_events(conn, result["run_key"], "TIJARA", "2026-01-01", "2026-07-09")
        tijara_capture = h.position_capture(tijara_events_2026, tijara_daily, "TIJARA")
        tijara_verdict = h.tijara_may_verdict(tijara_events_2026)
        tijara_closed_2021_2025 = [event for event in h.load_events(conn, result["run_key"], "TIJARA", "2021-01-01", "2025-12-31") if str(event.get("event_type", "")).startswith("EXIT")]
    total_rows = max(int(result.get("row_count") or 0), 1)
    candidate_sum = float(universe.get("sum_pnl_pct") or 0.0)
    total_open_sessions = float(universe.get("total_open_sessions") or 0.0)
    g1_pass = first_soft is not None and first_soft <= "2026-02-28"
    g2_pass = not exposure_below_1000
    g3_pass = sanam_first is not None and sanam_first.get("date") == "2026-04-15" and abs(float(sanam_first.get("entry_close") or 0.0) - 229.0) < 1e-9 and sanam_capture is not None and sanam_capture >= 57.2
    g4_pass = tijara_capture is not None and tijara_capture >= 101.1 and tijara_verdict in {"SURVIVED"} | {f"EXITED+REENTERED_WITHIN_{idx}" for idx in range(0, 31)}
    return {
        "run_key": result["run_key"],
        "run_nonce": result["run_nonce"],
        "harness_db": result["harness_db"],
        "row_count": result["row_count"],
        "sealed_sha_pass": result["sealed"]["actual_sha256"] == h.REQUIRED_SHA256,
        "freeze_sha_pass": bool(result["freeze_byte_match"]),
        "g1_date": first_soft,
        "g1_pass": g1_pass,
        "g2_pass": g2_pass,
        "g3_capture": sanam_capture,
        "g3_first_entry_date": None if sanam_first is None else sanam_first.get("date"),
        "g3_first_entry_price": None if sanam_first is None else sanam_first.get("entry_close"),
        "g3_pass": g3_pass,
        "g4_capture": tijara_capture,
        "g4_verdict": tijara_verdict,
        "g4_pass": g4_pass,
        "g5_count": len(tijara_closed_2021_2025),
        "efficiency": None if total_open_sessions <= 0.0 else candidate_sum / total_open_sessions,
        "sum_pnl_pct": candidate_sum,
        "worst_position_pnl_pct": float(universe.get("worst_position_pnl_pct") or 0.0),
        "clock_share": total_open_sessions / total_rows,
        "total_open_sessions": int(total_open_sessions),
        "baseline_sum_pnl_pct": baseline_sum["baseline_sum_pnl_pct"],
    }


def format_line(row: dict[str, Any]) -> str:
    return (
        f"SWEEP|{row['label']}|"
        f"G1={row['g1_date']}|"
        f"G2={'PASS' if row['g2_pass'] else 'FAIL'}|"
        f"G3_CAPTURE={row['g3_capture']}|"
        f"G4_CAPTURE={row['g4_capture']}|G4_VERDICT={row['g4_verdict']}|"
        f"G5_COUNT={row['g5_count']}|"
        f"EFFICIENCY={row['efficiency']}|"
        f"SUM={row['sum_pnl_pct']}|"
        f"WORST={row['worst_position_pnl_pct']}"
    )


def write_artifact(path: Path, text: str) -> str:
    return h.write_text_with_sidecar(path, text if text.endswith("\n") else text + "\n")


def cleanup_sweep_dbs() -> None:
    SWEEP_SANDBOX.mkdir(parents=True, exist_ok=True)
    for path in SWEEP_SANDBOX.glob("harness_v53A_*.db*"):
        path.unlink(missing_ok=True)


def run_worker_point(index: int, output_path: Path) -> None:
    point = sweep_points()[index]
    configure(point)
    baseline_sum = h.compute_baseline_sum()
    result = h.run_variant("A")
    metrics = collect_metrics(result, baseline_sum)
    row = {**asdict(point), **metrics}
    output_path.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")


def archive_result_db(row: dict[str, Any]) -> dict[str, Any]:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(str(row["harness_db"]))
    if not source.exists():
        return {"archived": False, "reason": "source_missing", "harness_db": str(source)}
    target = ARCHIVE_DIR / source.name
    if target.exists():
        target.unlink()
    shutil.move(str(source), str(target))
    digest = h.sha256_file(target)
    (target.with_suffix(target.suffix + ".sha256")).write_text(f"{digest}  {target.name}\n", encoding="ascii")
    for suffix in ("-wal", "-shm"):
        side = source.with_name(source.name + suffix)
        if side.exists():
            side.unlink(missing_ok=True)
    row["archived_harness_db"] = str(target)
    row["archived_harness_db_sha256"] = digest
    return {"archived": True, "source": str(source), "target": str(target), "sha256": digest}


def run_single_point(index: int, output_path: Path) -> None:
    worker_output = output_path.with_name(output_path.stem + "_worker.json")
    worker_output.unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(index), str(worker_output)]
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    row = json.loads(worker_output.read_text(encoding="utf-8"))
    archive = archive_result_db(row)
    row["archive"] = archive
    output_path.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
    worker_output.unlink(missing_ok=True)


def run_point_child(index: int, point: SweepPoint) -> dict[str, Any]:
    output_path = SWEEP_SANDBOX / f"sweep_point_{index:02d}.json"
    output_path.unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--single", str(index), str(output_path)]
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"sweep point failed: {point.label} rc={completed.returncode}")
    row = json.loads(output_path.read_text(encoding="utf-8"))
    output_path.unlink(missing_ok=True)
    cleanup_sweep_dbs()
    return row


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        run_worker_point(int(sys.argv[2]), Path(sys.argv[3]))
        return
    if len(sys.argv) == 4 and sys.argv[1] == "--single":
        run_single_point(int(sys.argv[2]), Path(sys.argv[3]))
        return
    cleanup_sweep_dbs()
    SWEEP_EXPORT.mkdir(parents=True, exist_ok=True)
    h.SANDBOX = SWEEP_SANDBOX
    h.EXPORT = SWEEP_EXPORT
    baseline_sum = h.compute_baseline_sum()
    points = sweep_points()
    rows: list[dict[str, Any]] = []
    print("R16_RAT_1_SWEEP_PLAN|" + json.dumps({"unique_runs": len(points), "directive_claimed_runs": 11, "note": "Enumerated one-at-a-time values with ratified point counted once resolve to 9 unique runs."}, sort_keys=True))
    for index, point in enumerate(points):
        print("R16_RAT_1_RUN_START|" + json.dumps(asdict(point), sort_keys=True), flush=True)
        row = run_point_child(index, point)
        rows.append(row)
        print(format_line(row), flush=True)
    ratified_sum = next(row["sum_pnl_pct"] for row in rows if row["label"] == "RATIFIED")
    violations = []
    for row in rows:
        if not (row["g1_pass"] and row["g2_pass"] and row["g3_pass"] and row["g4_pass"]):
            violations.append({"label": row["label"], "type": "G1_G4_ROBUSTNESS", "g1": row["g1_pass"], "g2": row["g2_pass"], "g3": row["g3_pass"], "g4": row["g4_pass"]})
        if row["label"] != "RATIFIED" and float(row["sum_pnl_pct"] or 0.0) > 1.15 * ratified_sum:
            violations.append({"label": row["label"], "type": "SUM_IMPROVES_GT_15PCT", "sum_pnl_pct": row["sum_pnl_pct"], "ratified_sum_pnl_pct": ratified_sum})
    summary = {
        "ratified_sum_pnl_pct": ratified_sum,
        "unique_runs": len(rows),
        "directive_claimed_runs": 11,
        "all_g1_g4_pass": not any(v["type"] == "G1_G4_ROBUSTNESS" for v in violations),
        "no_sum_improvement_gt_15pct": not any(v["type"] == "SUM_IMPROVES_GT_15PCT" for v in violations),
        "violations": violations,
    }
    table_lines = ["R16_RAT_1_SWEEP_TABLE"] + [format_line(row) for row in rows] + ["R16_RAT_1_SUMMARY|" + json.dumps(summary, sort_keys=True)]
    json_lines = [json.dumps(row, sort_keys=True) for row in rows]
    hashes = {
        "r16_rat_1_sweep_table.txt": write_artifact(SWEEP_EXPORT / "r16_rat_1_sweep_table.txt", "\n".join(table_lines) + "\n"),
        "r16_rat_1_sweep_rows.jsonl": write_artifact(SWEEP_EXPORT / "r16_rat_1_sweep_rows.jsonl", "\n".join(json_lines) + "\n"),
        "r16_rat_1_sweep_summary.json": write_artifact(SWEEP_EXPORT / "r16_rat_1_sweep_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"),
    }
    if h.R3_INV_SOURCE.exists():
        hashes["r3_inv_diff.txt"] = h.export_existing_artifact(h.R3_INV_SOURCE)
        hashes["r3_inv_diff.txt.sha256"] = h.export_existing_artifact(h.R3_INV_SOURCE.with_suffix(h.R3_INV_SOURCE.suffix + ".sha256"))
    print("R16_RAT_1_EXPORT_HASHES")
    print(json.dumps(hashes, indent=2, sort_keys=True))
    print("R16_RAT_1_SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if violations:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
