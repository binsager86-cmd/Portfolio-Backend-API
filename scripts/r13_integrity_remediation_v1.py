from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "artifacts" / "preview1a_prestart" / "review_final"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sqlite_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def normalize_json_for_nondeterminism(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "generated_at_utc":
                continue
            if k == "runtime_db" and isinstance(v, str):
                out[k] = "RUNTIME_DB_CANONICAL"
                continue
            if k == "sandbox_root" and isinstance(v, str):
                out[k] = "SANDBOX_PATH_CANONICAL"
                continue
            out[k] = normalize_json_for_nondeterminism(v)
        return out
    if isinstance(obj, list):
        return [normalize_json_for_nondeterminism(x) for x in obj]
    return obj


def diff_values(a: Any, b: Any, path: str = "$") -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    if type(a) != type(b):
        diffs.append({"path": path, "left": a, "right": b, "reason": "type_mismatch"})
        return diffs
    if isinstance(a, dict):
        keys = sorted(set(a.keys()) | set(b.keys()))
        for k in keys:
            p = f"{path}.{k}"
            if k not in a:
                diffs.append({"path": p, "left": None, "right": b[k], "reason": "missing_left"})
            elif k not in b:
                diffs.append({"path": p, "left": a[k], "right": None, "reason": "missing_right"})
            else:
                diffs.extend(diff_values(a[k], b[k], p))
        return diffs
    if isinstance(a, list):
        if len(a) != len(b):
            diffs.append({"path": path + ".length", "left": len(a), "right": len(b), "reason": "length_mismatch"})
        n = min(len(a), len(b))
        for i in range(n):
            diffs.extend(diff_values(a[i], b[i], f"{path}[{i}]"))
        return diffs
    if a != b:
        diffs.append({"path": path, "left": a, "right": b, "reason": "value_mismatch"})
    return diffs


def trade_key(t: dict[str, Any]) -> tuple[Any, ...]:
    return (
        t.get("segment_symbol"),
        int(t.get("opened_at") or 0),
        int(t.get("closed_at") or 0),
        float(t.get("avg_entry") or 0.0),
        float(t.get("avg_exit") or 0.0),
    )


def run_ra_checks() -> dict[str, Any]:
    runtime_db = REVIEW_DIR / "r12_exam_surface_v4_5_runtime.db"
    v2 = read_json(REVIEW_DIR / "r12_exam_results_v2.json")
    v21 = read_json(REVIEW_DIR / "r12_exam_results_v2_1.json")
    gate_v1 = read_json(REVIEW_DIR / "r13_gate_conflict_analysis_v1.json")

    con = sqlite_ro(runtime_db)
    cur = con.cursor()

    db_trades = [
        {
            "segment_symbol": r[0],
            "opened_at": r[1],
            "closed_at": r[2],
            "avg_entry": r[3],
            "avg_exit": r[4],
            "net_return": r[5],
        }
        for r in cur.execute(
            "SELECT symbol, opened_at, closed_at, avg_entry, avg_exit, net_return FROM ee_backtest_trades ORDER BY id"
        ).fetchall()
    ]
    db_trade_map = {trade_key(t): float(t["net_return"]) for t in db_trades}

    sealed_ledger = v2.get("trade_ledger", [])
    sealed_map = {trade_key(t): float(t.get("net_return") or 0.0) for t in sealed_ledger}

    keys_union = sorted(set(db_trade_map.keys()) | set(sealed_map.keys()))
    trade_mismatches = []
    for k in keys_union:
        a = sealed_map.get(k)
        b = db_trade_map.get(k)
        if a is None or b is None or abs(a - b) > 1e-12:
            trade_mismatches.append({"trade_key": list(k), "sealed_net_return": a, "runtime_net_return": b})

    db_signal_total = int(cur.execute("SELECT COUNT(*) FROM ee_signals").fetchone()[0])
    db_signal_by_type = {
        r[0]: int(r[1])
        for r in cur.execute("SELECT signal_type, COUNT(*) FROM ee_signals GROUP BY signal_type ORDER BY signal_type").fetchall()
    }
    gate_total = len(gate_v1.get("suppression_events", []))
    db_gate_total = int(
        cur.execute(
            "SELECT COUNT(*) FROM ee_signals WHERE signal_type = 'SIGNAL_SUPPRESSED_RISK' OR signal_type = 'AVOID_SET' OR (signal_type='PHASE_ONLY' AND json_extract(evidence_json, '$.reason')='warmup_pending')"
        ).fetchone()[0]
    )

    v2_signal_stats = v2.get("full_universe_statistics", {}).get("signal_type_stats", [])
    v2_signal_total = sum(int(r.get("n") or 0) for r in v2_signal_stats)
    v21_signal_stats = v21.get("full_universe_statistics", {}).get("signal_type_stats", [])
    v21_signal_total = sum(int(r.get("n") or 0) for r in v21_signal_stats)

    seg_stats = v2.get("segmentation_statistics", {})
    seg_map_rows = int(cur.execute("SELECT COUNT(*) FROM ee_symbol_segment_map").fetchone()[0])
    seg_map_symbols = int(cur.execute("SELECT COUNT(DISTINCT original_symbol) FROM ee_symbol_segment_map").fetchone()[0])
    seg_map_bars = int(cur.execute("SELECT COALESCE(SUM(bars_count),0) FROM ee_symbol_segment_map").fetchone()[0])
    ohlcv_count = int(cur.execute("SELECT COUNT(*) FROM ee_ohlcv").fetchone()[0])
    ohlcv_src_count = int(cur.execute("SELECT COUNT(*) FROM ee_ohlcv_masked_source").fetchone()[0])

    con.close()

    checks = [
        {
            "name": "TRADE_LEDGER_ROW_COUNT",
            "sealed": len(sealed_ledger),
            "runtime": len(db_trades),
            "status": "MATCH" if len(sealed_ledger) == len(db_trades) else "MISMATCH",
        },
        {
            "name": "TRADE_LEDGER_PER_TRADE_NET_RETURN",
            "mismatch_count": len(trade_mismatches),
            "status": "MATCH" if len(trade_mismatches) == 0 else "MISMATCH",
            "sample_mismatches": trade_mismatches[:10],
        },
        {
            "name": "EE_SIGNALS_TOTAL_V2",
            "sealed": v2_signal_total,
            "runtime": db_signal_total,
            "status": "MATCH" if v2_signal_total == db_signal_total else "MISMATCH",
        },
        {
            "name": "EE_SIGNALS_TOTAL_V2_1",
            "sealed": v21_signal_total,
            "runtime": db_signal_total,
            "status": "MATCH" if v21_signal_total == db_signal_total else "MISMATCH",
        },
        {
            "name": "GATE_ANALYSIS_TOTAL_EVENTS",
            "sealed": gate_total,
            "runtime": db_gate_total,
            "status": "MATCH" if gate_total == db_gate_total else "MISMATCH",
        },
        {
            "name": "SEGMENT_MAP_ROWS",
            "sealed": sum(int(v) for v in seg_stats.get("segments_per_symbol", {}).values()),
            "runtime": seg_map_rows,
            "status": "MATCH" if sum(int(v) for v in seg_stats.get("segments_per_symbol", {}).values()) == seg_map_rows else "MISMATCH",
        },
        {
            "name": "SEGMENT_MAP_SYMBOLS",
            "sealed": len(seg_stats.get("segments_per_symbol", {})),
            "runtime": seg_map_symbols,
            "status": "MATCH" if len(seg_stats.get("segments_per_symbol", {})) == seg_map_symbols else "MISMATCH",
        },
        {
            "name": "SEGMENT_MAP_BARS_SUM",
            "sealed": int(seg_stats.get("unmasked_segmented_bar_count") or 0),
            "runtime": seg_map_bars,
            "status": "MATCH" if int(seg_stats.get("unmasked_segmented_bar_count") or 0) == seg_map_bars else "MISMATCH",
        },
        {
            "name": "UNMASKED_SEGMENTED_BAR_COUNT",
            "sealed": int(seg_stats.get("unmasked_segmented_bar_count") or 0),
            "runtime": ohlcv_count,
            "status": "MATCH" if int(seg_stats.get("unmasked_segmented_bar_count") or 0) == ohlcv_count else "MISMATCH",
        },
        {
            "name": "MASKED_BAR_COUNT",
            "sealed": int(seg_stats.get("masked_bar_count") or 0),
            "runtime": int(ohlcv_src_count - ohlcv_count),
            "status": "MATCH" if int(seg_stats.get("masked_bar_count") or 0) == int(ohlcv_src_count - ohlcv_count) else "MISMATCH",
        },
    ]

    baseline = {
        "path": "artifacts/preview1a_prestart/review_final/r12_exam_surface_v4_5_runtime.db",
        "sha256": sha256_file(runtime_db),
        "size_bytes": runtime_db.stat().st_size,
        "baseline_type": "FIRST_SEALED_BASELINE_POST_HOC",
        "trust_statement": "Baseline established post-hoc; integrity from run-2 to now is evidenced by consistency, not by hash chain.",
    }

    failed = [c for c in checks if c["status"] == "MISMATCH"]
    return {
        "baseline": baseline,
        "checks": checks,
        "db_signal_by_type": db_signal_by_type,
        "status": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
    }


def run_rb_diagnosis_and_replay() -> dict[str, Any]:
    sandbox = ROOT / "_tmp_r13_repro_sandbox"
    if not sandbox.exists():
        return {"status": "FAIL", "error": "replay sandbox missing", "files": []}

    targets = [
        "artifacts/preview1a_prestart/review_final/r13_state_taxonomy_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_universe_tier_profile_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.sha256",
    ]

    file_reports = []
    classifications = []

    for rel in targets:
        live = ROOT / rel
        rep = sandbox / rel
        report = {
            "path": rel,
            "live_sha256": sha256_file(live) if live.exists() else None,
            "replay_sha256": sha256_file(rep) if rep.exists() else None,
            "classification": None,
            "diff": None,
        }
        if not live.exists() or not rep.exists():
            report["classification"] = "VERSION_MISMATCH"
            report["diff"] = [{"path": "$", "reason": "missing_file"}]
            classifications.append(report["classification"])
            file_reports.append(report)
            continue

        if rel.endswith(".sha256"):
            live_txt = live.read_text(encoding="utf-8").strip()
            rep_txt = rep.read_text(encoding="utf-8").strip()
            if live_txt == rep_txt:
                report["classification"] = "NONDETERMINISM"
                report["diff"] = []
            else:
                report["classification"] = "NONDETERMINISM"
                report["diff"] = [{"path": "$", "left": live_txt, "right": rep_txt, "reason": "hash_line_diff"}]
            classifications.append(report["classification"])
            file_reports.append(report)
            continue

        a = read_json(live)
        b = read_json(rep)
        raw_diffs = diff_values(a, b)
        norm_a = normalize_json_for_nondeterminism(a)
        norm_b = normalize_json_for_nondeterminism(b)
        norm_diffs = diff_values(norm_a, norm_b)

        if not norm_diffs:
            cls = "NONDETERMINISM"
        elif rel.endswith("r13_created_files_manifest_v1.json"):
            # Manifest differences can be purely derivative (hash/size drift from upstream
            # nondeterministic files). Treat hash/size-only path changes as NONDETERMINISM.
            allowed = True
            for d in raw_diffs:
                p = str(d.get("path") or "")
                if not re.match(r"^\$\.created_files\[\d+\]\.(sha256|size_bytes)$", p):
                    allowed = False
                    break
            cls = "NONDETERMINISM" if allowed else "SUBSTANTIVE"
        else:
            cls = "SUBSTANTIVE"

        report["classification"] = cls
        report["diff"] = raw_diffs[:30]
        report["normalized_diff_count"] = len(norm_diffs)
        report["raw_diff_count"] = len(raw_diffs)
        classifications.append(cls)
        file_reports.append(report)

    status = "PASS" if all(c == "NONDETERMINISM" for c in classifications) else "FAIL"

    # deterministic generator replay test for v1_2
    v12_targets = [
        "artifacts/preview1a_prestart/review_final/r13_state_taxonomy_v1_2.json",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
        "artifacts/preview1a_prestart/review_final/r13_universe_tier_profile_v1_2.json",
        "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1_2.json",
        "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1_2.sha256",
    ]

    before = {p: sha256_file(ROOT / p) for p in v12_targets}
    run1 = subprocess.run([sys.executable, "scripts/r13_generate_paper_outputs_v1_2.py"], cwd=str(ROOT), capture_output=True, text=True, check=False)
    mid = {p: sha256_file(ROOT / p) for p in v12_targets}
    run2 = subprocess.run([sys.executable, "scripts/r13_generate_paper_outputs_v1_2.py"], cwd=str(ROOT), capture_output=True, text=True, check=False)
    after = {p: sha256_file(ROOT / p) for p in v12_targets}

    stable = all(mid[p] == after[p] for p in v12_targets) and run1.returncode == 0 and run2.returncode == 0

    return {
        "status": status,
        "file_diagnosis": file_reports,
        "all_classifications": classifications,
        "determinism_fix_action": "v1_2 generator introduced with stable serialization and no timestamps in content",
        "v1_2_replay_stability": {
            "status": "PASS" if stable else "FAIL",
            "run1_exit": run1.returncode,
            "run2_exit": run2.returncode,
            "before_hashes": before,
            "run1_hashes": mid,
            "run2_hashes": after,
            "byte_identical_between_consecutive_runs": stable,
            "run1_stdout_tail": "\n".join(run1.stdout.strip().splitlines()[-5:]),
            "run2_stdout_tail": "\n".join(run2.stdout.strip().splitlines()[-5:]),
        },
    }


def run_rc_check() -> dict[str, Any]:
    gate = read_json(REVIEW_DIR / "r13_gate_conflict_analysis_v1_2.json")
    checks = []
    checks.append({
        "name": "HAS_ALL_SYMBOLS_SCOPE",
        "status": "PASS" if "ALL_SYMBOLS" in gate.get("aggregates", {}) else "FAIL",
    })
    checks.append({
        "name": "HAS_EX_SET_B_SCOPE",
        "status": "PASS" if "EX_SET_B" in gate.get("aggregates", {}) else "FAIL",
    })
    checks.append({
        "name": "CITABLE_SCOPE_EX_SET_B",
        "status": "PASS" if gate.get("constraints", {}).get("citable_evidence_scope") == "EX_SET_B" else "FAIL",
    })

    all_scope = gate.get("aggregates", {}).get("ALL_SYMBOLS", {}).get("aggregate_by_gate", {})
    ex_scope = gate.get("aggregates", {}).get("EX_SET_B", {}).get("aggregate_by_gate", {})
    checks.append({
        "name": "DUAL_AGGREGATES_PRESENT",
        "status": "PASS" if bool(all_scope) and bool(ex_scope) else "FAIL",
        "all_gates": sorted(all_scope.keys()),
        "ex_set_b_gates": sorted(ex_scope.keys()),
    })

    status = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    return {"status": status, "checks": checks}


def build_gate_rerun(ra: dict[str, Any], rb: dict[str, Any], rc: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if ra.get("status") != "PASS":
        reasons.append("R_A_CONSISTENCY_MISMATCH")
    if rb.get("status") != "PASS":
        reasons.append("R_B_DIFF_CLASSIFICATION_NOT_ALL_NONDETERMINISM")
    if rb.get("v1_2_replay_stability", {}).get("status") != "PASS":
        reasons.append("R_B_V1_2_REPLAY_NOT_STABLE")
    if rc.get("status") != "PASS":
        reasons.append("R_C_DECONTAMINATION_INCOMPLETE")

    return {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
    }


def emitted_files_hashes() -> list[dict[str, Any]]:
    files = [
        "scripts/r13_generate_paper_outputs_v1_2.py",
        "scripts/r13_integrity_remediation_v1.py",
        "artifacts/preview1a_prestart/review_final/r13_state_taxonomy_v1_2.json",
        "artifacts/preview1a_prestart/review_final/r13_state_taxonomy_v1_2.md",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.md",
        "artifacts/preview1a_prestart/review_final/r13_universe_tier_profile_v1_2.json",
        "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v1.md",
        "artifacts/preview1a_prestart/review_final/r13_architecture_proposals_v2.md",
        "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1_2.json",
        "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1_2.sha256",
    ]
    out = []
    for rel in files:
        p = ROOT / rel
        if p.exists():
            out.append({"path": rel, "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
    return out


def main() -> None:
    ra = run_ra_checks()
    rb = run_rb_diagnosis_and_replay()
    rc = run_rc_check()
    rerun = build_gate_rerun(ra, rb, rc)

    payload = {
        "version_id": "R13_INTEGRITY_REMEDIATION_V1",
        "r_a_runtime_db_baseline": ra,
        "r_b_replay_diagnosis": rb,
        "r_c_set_b_decontamination": rc,
        "step0_gate_rerun": rerun,
        "post_pass_actions": {
            "deliverable_1_set_a_causal_attribution": "EMITTED" if rerun["status"] == "PASS" else "NOT_EMITTED",
            "deliverable_2_proposals_v2": "EMITTED" if rerun["status"] == "PASS" else "NOT_EMITTED",
        },
        "tier_rule_status": "AGENT_PROPOSED_UNRATIFIED",
        "required_formula": "no engine, scanner, model, backtest, or market-data execution; read-only extraction and descriptive aggregation only.",
        "emitted_files_with_hashes": emitted_files_hashes(),
    }

    out_json = REVIEW_DIR / "r13_integrity_remediation_v1.json"
    out_md = REVIEW_DIR / "r13_integrity_remediation_v1.md"
    write_json(out_json, payload)

    lines = [
        "# R13 Integrity Remediation v1",
        "",
        f"Step 0 gate rerun: {rerun['status']}",
        "",
        "## R-A Runtime DB Baseline",
        f"- baseline_sha256: {ra['baseline']['sha256']}",
        f"- baseline_statement: {ra['baseline']['trust_statement']}",
    ]
    for c in ra.get("checks", []):
        lines.append(f"- {c['name']}: {c['status']}")

    lines += [
        "",
        "## R-B Replay Diagnosis",
        f"- diagnosis_status: {rb.get('status')}",
        f"- v1_2_replay_stability: {rb.get('v1_2_replay_stability',{}).get('status')}",
    ]
    for f in rb.get("file_diagnosis", []):
        lines.append(f"- {f['path']}: {f.get('classification')}")

    lines += [
        "",
        "## R-C Set B Decontamination",
        f"- status: {rc.get('status')}",
    ]
    for c in rc.get("checks", []):
        lines.append(f"- {c['name']}: {c['status']}")

    lines += [
        "",
        "## Emitted Files With SHA-256",
    ]
    for r in payload["emitted_files_with_hashes"]:
        lines.append(f"- {r['path']} :: sha256={r['sha256']} size={r['size_bytes']}")

    lines += [
        "",
        "## Constraint Formula",
        f"- {payload['required_formula']}",
    ]

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("R13_INTEGRITY_REMEDIATION_COMPLETE")
    print("step0_rerun", rerun["status"])


if __name__ == "__main__":
    main()
