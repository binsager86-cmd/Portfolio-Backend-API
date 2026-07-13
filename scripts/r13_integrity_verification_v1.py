from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "artifacts" / "preview1a_prestart" / "review_final"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_size(path: Path) -> int:
    return path.stat().st_size


def sqlite_ro_connect(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def ts_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def line_citations(path: Path, patterns: list[str]) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        for pat in patterns:
            if re.search(pat, line):
                out.append({
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "line": idx,
                    "text": line.strip(),
                    "pattern": pat,
                })
    return out


def parse_generator_return_code_quotes(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    end = None
    for i, line in enumerate(lines, start=1):
        if line.startswith("def future_close_metrics"):
            start = i
        if start is not None and line.startswith("def liquidity_tier"):
            end = i - 1
            break
    if start is None:
        return {"error": "future_close_metrics not found"}
    if end is None:
        end = min(start + 120, len(lines))
    snippet = "\n".join(lines[start - 1 : end])
    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "start_line": start,
        "end_line": end,
        "snippet": snippet,
    }


def load_reference_hashes() -> dict[str, str]:
    refs: dict[str, str] = {}
    seal = read_json(REVIEW_DIR / "r12_pre_exam_surface_seal_v4_4.json")
    for row in seal.get("ratified_artifact_references", []):
        p = row.get("path")
        h = row.get("sha256")
        if p and h:
            refs[p] = h

    add10 = read_json(REVIEW_DIR / "r12a_created_files_manifest_v4_5_addendum_10.json")
    for row in add10.get("created_files", []):
        p = row.get("path")
        h = row.get("sha256")
        if p and h:
            refs[p] = h

    add11 = read_json(REVIEW_DIR / "r12a_created_files_manifest_v2_1_addendum_11.json")
    for row in add11.get("created_files", []):
        p = row.get("path")
        h = row.get("sha256")
        if p and h:
            refs[p] = h

    return refs


def compare_hash(path_rel: str, expected: str | None) -> dict[str, Any]:
    abs_path = ROOT / Path(path_rel)
    exists = abs_path.exists()
    current = sha256_file(abs_path) if exists else None
    status = "NO_BASELINE"
    if expected is not None and current is not None:
        status = "MATCH" if expected == current else "MISMATCH"
    return {
        "path": path_rel,
        "exists": exists,
        "expected_sha256": expected,
        "current_sha256": current,
        "status": status,
        "size_bytes": file_size(abs_path) if exists else None,
    }


def base_symbol(symbol: str) -> str:
    s = str(symbol)
    return s.split("__SEG", 1)[0] if "__SEG" in s else s


def mean(vals: list[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def verify_seam_safety(conf: dict[str, Any], runtime_db: Path) -> dict[str, Any]:
    con = sqlite_ro_connect(runtime_db)
    cur = con.cursor()

    rows = cur.execute("SELECT symbol, trade_date, close FROM ee_ohlcv ORDER BY symbol, trade_date").fetchall()
    by_base: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    by_seg: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for sym, td, close in rows:
        s = str(sym)
        b = base_symbol(s)
        by_base[b].append((int(td), float(close or 0.0), s))
        by_seg[s].append((int(td), float(close or 0.0)))

    events = [r for r in conf.get("suppression_events", []) if r.get("gate") in {"AVOID_GATE", "RISK_SUPPRESSION", "WARMUP_GATE"}]

    horizons = [5, 20, 60]
    cross_counts = Counter()
    trunc_counts = Counter()
    corrected_by_gate: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for ev in events:
        gate = str(ev.get("gate"))
        b = str(ev.get("symbol"))
        seg = str(ev.get("symbol_segment"))
        td = int(ev.get("trade_date") or 0)
        entry = float(ev.get("evidence_values", {}).get("close") or 0.0)

        base_future = [(d, c, ss) for (d, c, ss) in by_base.get(b, []) if d > td]
        seg_future = [(d, c) for (d, c) in by_seg.get(seg, []) if d > td]

        for h in horizons:
            if entry <= 0:
                continue
            if len(base_future) >= h:
                _, _, seg_h = base_future[h - 1]
                if seg_h != seg:
                    cross_counts[(gate, h)] += 1
                    cross_counts[("ALL", h)] += 1

            if not seg_future:
                continue
            idx = min(h, len(seg_future)) - 1
            close_h = seg_future[idx][1]
            ret = (close_h / entry) - 1.0
            corrected_by_gate[gate][f"ret_{h}"].append(ret)
            if len(seg_future) < h:
                trunc_counts[(gate, h)] += 1
                trunc_counts[("ALL", h)] += 1

    con.close()

    old_agg = conf.get("aggregate_by_gate", {})
    corrected_agg: dict[str, Any] = {}
    for gate in sorted(corrected_by_gate.keys()):
        corrected_agg[gate] = {
            "count": int(old_agg.get(gate, {}).get("count", 0)),
            "mean_ret_5": mean(corrected_by_gate[gate].get("ret_5", [])),
            "mean_ret_20": mean(corrected_by_gate[gate].get("ret_20", [])),
            "mean_ret_60": mean(corrected_by_gate[gate].get("ret_60", [])),
            "truncations": {
                "ret_5": int(trunc_counts.get((gate, 5), 0)),
                "ret_20": int(trunc_counts.get((gate, 20), 0)),
                "ret_60": int(trunc_counts.get((gate, 60), 0)),
            },
            "horizon_rule": "seam-safe same-segment forward return; horizon truncated at segment end",
        }

    contamination = {
        "cross_segment_window_counts": {
            "ret_5": int(cross_counts.get(("ALL", 5), 0)),
            "ret_20": int(cross_counts.get(("ALL", 20), 0)),
            "ret_60": int(cross_counts.get(("ALL", 60), 0)),
        },
        "cross_segment_window_counts_by_gate": {
            gate: {
                "ret_5": int(cross_counts.get((gate, 5), 0)),
                "ret_20": int(cross_counts.get((gate, 20), 0)),
                "ret_60": int(cross_counts.get((gate, 60), 0)),
            }
            for gate in ["AVOID_GATE", "RISK_SUPPRESSION", "WARMUP_GATE"]
        },
        "contamination_detected": any(cross_counts.get(("ALL", h), 0) > 0 for h in horizons),
    }

    return {
        "original_aggregate_by_gate": old_agg,
        "contamination": contamination,
        "corrected_seam_safe_aggregate_by_gate": corrected_agg,
    }


def sandbox_regeneration_compare() -> dict[str, Any]:
    tmp = ROOT / "_tmp_r13_repro_sandbox"
    if tmp.exists():
        shutil.rmtree(tmp)

    # Minimal tree needed for generator rerun without touching sealed artifacts.
    (tmp / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp / "app" / "services" / "eagle_eye").mkdir(parents=True, exist_ok=True)
    (tmp / "artifacts" / "preview1a_prestart" / "review_final").mkdir(parents=True, exist_ok=True)

    copy_files = [
        "scripts/r13_generate_paper_outputs_v1.py",
        "app/services/eagle_eye/scanner_service.py",
        "app/services/eagle_eye/risk_service.py",
        "artifacts/preview1a_prestart/review_final/r12_exam_results_v2_1.json",
        "artifacts/preview1a_prestart/review_final/r12_breach_triage_v4_2_FINAL.json",
        "artifacts/preview1a_prestart/review_final/r12_ca_ledger_v0_2.json",
        "artifacts/preview1a_prestart/review_final/r12_masked_intervals_manifest_v4_3_final.json",
        "artifacts/preview1a_prestart/review_final/r12_pre_exam_surface_seal_v4_4.json",
        "artifacts/preview1a_prestart/review_final/r12_exam_surface_v4_5_runtime.db",
    ]

    for rel in copy_files:
        src = ROOT / rel
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    proc = subprocess.run(
        [sys.executable, "scripts/r13_generate_paper_outputs_v1.py"],
        cwd=str(tmp),
        capture_output=True,
        text=True,
        check=False,
    )

    generated = [
        "artifacts/preview1a_prestart/review_final/r13_state_taxonomy_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_state_taxonomy_v1.md",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1.md",
        "artifacts/preview1a_prestart/review_final/r13_architecture_proposals_v1.md",
        "artifacts/preview1a_prestart/review_final/r13_universe_tier_profile_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.sha256",
    ]

    comparisons = []
    all_identical = proc.returncode == 0
    for rel in generated:
        live = ROOT / rel
        repro = tmp / rel
        live_hash = sha256_file(live) if live.exists() else None
        repro_hash = sha256_file(repro) if repro.exists() else None
        identical = bool(live_hash and repro_hash and live_hash == repro_hash)
        all_identical = all_identical and identical
        comparisons.append(
            {
                "path": rel,
                "live_sha256": live_hash,
                "repro_sha256": repro_hash,
                "byte_identical": identical,
                "live_size": file_size(live) if live.exists() else None,
                "repro_size": file_size(repro) if repro.exists() else None,
            }
        )

    # keep sandbox for audit evidence
    return {
        "sandbox_root": str(tmp),
        "generator_exit_code": int(proc.returncode),
        "generator_stdout_tail": "\n".join(proc.stdout.strip().splitlines()[-10:]),
        "generator_stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-10:]),
        "all_byte_identical": all_identical,
        "comparisons": comparisons,
    }


def distinct_runtime_values(runtime_db: Path) -> dict[str, Any]:
    con = sqlite_ro_connect(runtime_db)
    cur = con.cursor()
    signal_types = [r[0] for r in cur.execute("SELECT DISTINCT signal_type FROM ee_signals ORDER BY signal_type").fetchall()]
    phase_from = [r[0] for r in cur.execute("SELECT DISTINCT phase_from FROM ee_signals ORDER BY phase_from").fetchall()]
    phase_to = [r[0] for r in cur.execute("SELECT DISTINCT phase_to FROM ee_signals ORDER BY phase_to").fetchall()]
    evidence_rows = cur.execute("SELECT evidence_json FROM ee_signals").fetchall()
    con.close()

    attempted = set()
    suppressed = set()
    for (raw,) in evidence_rows:
        try:
            j = json.loads(str(raw or "{}"))
        except Exception:
            j = {}
        a = j.get("attempted_signal_type")
        s = j.get("suppressed_reason")
        if a is not None:
            attempted.add(str(a))
        if s is not None:
            suppressed.add(str(s))

    return {
        "signal_type": signal_types,
        "phase_from": phase_from,
        "phase_to": phase_to,
        "attempted_signal_type": sorted(attempted),
        "suppressed_reason": sorted(suppressed),
    }


def source_vocabulary_with_citations() -> dict[str, Any]:
    scanner = ROOT / "app" / "services" / "eagle_eye" / "scanner_service.py"
    risk = ROOT / "app" / "services" / "eagle_eye" / "risk_service.py"

    scanner_lines = scanner.read_text(encoding="utf-8").splitlines()

    phase_entries: list[str] = []
    phase_citations: list[dict[str, Any]] = []
    in_phases = False
    for idx, line in enumerate(scanner_lines, start=1):
        if line.strip().startswith("PHASES = {"):
            in_phases = True
        if in_phases:
            m = re.search(r'"([A-Z_]+)"', line)
            if m:
                phase_entries.append(m.group(1))
                phase_citations.append({"path": "app/services/eagle_eye/scanner_service.py", "line": idx, "text": line.strip()})
            if line.strip().startswith("}"):
                in_phases = False

    signal_assignments = line_citations(
        scanner,
        [
            r'signal_type\s*=\s*"[A-Z_]+"',
            r'effective_signal_type\s*=\s*"SIGNAL_SUPPRESSED_RISK"',
            r'signal_type=\"PHASE_ONLY\"',
        ],
    )
    suppression_reason_lines = line_citations(risk, [r'return False, "[a-z_]+"', r'return True, "ok"'])

    signal_values = sorted({
        re.search(r'"([A-Z_]+)"', r["text"]).group(1)
        for r in signal_assignments
        if re.search(r'"([A-Z_]+)"', r["text"]) is not None
    })

    suppression_values = sorted({
        re.search(r'"([a-z_]+|ok)"', r["text"]).group(1)
        for r in suppression_reason_lines
        if re.search(r'"([a-z_]+|ok)"', r["text"]) is not None
    })

    gates = [
        {"name": "AVOID_CONDITION", "citation": {"path": "app/services/eagle_eye/scanner_service.py", "line": 526}},
        {"name": "BREAKOUT_MANDATORY_M1_M5", "citation": {"path": "app/services/eagle_eye/scanner_service.py", "line": 707}},
        {"name": "ML_GATE", "citation": {"path": "app/services/eagle_eye/scanner_service.py", "line": 750}},
        {"name": "RISK_SUPPRESSION", "citation": {"path": "app/services/eagle_eye/scanner_service.py", "line": 992}},
        {"name": "LIQUIDITY_FILTER", "citation": {"path": "app/services/eagle_eye/risk_service.py", "line": 11}},
        {"name": "WARMUP_PENDING", "citation": {"path": "app/services/eagle_eye/scanner_service.py", "line": 390}},
    ]

    return {
        "phases": sorted(set(phase_entries)),
        "phases_citations": phase_citations,
        "signal_types_emittable": signal_values,
        "signal_type_citations": signal_assignments,
        "suppression_reasons": suppression_values,
        "suppression_reason_citations": suppression_reason_lines,
        "gates": gates,
    }


def set_b_derivation_audit(conf: dict[str, Any], arch_md: str) -> dict[str, Any]:
    events = conf.get("suppression_events", [])
    set_b_events = [e for e in events if e.get("cohort") == "set_b"]
    non_set_b_events = [e for e in events if e.get("cohort") != "set_b"]

    def agg(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
        by_gate: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_gate[str(r.get("gate"))].append(r)
        out: dict[str, dict[str, float | int | None]] = {}
        for gate, grows in sorted(by_gate.items()):
            out[gate] = {
                "count": len(grows),
                "mean_ret_5": mean([x["subsequent_unmasked_outcome"]["ret_5"] for x in grows if x["subsequent_unmasked_outcome"]["ret_5"] is not None]),
                "mean_ret_20": mean([x["subsequent_unmasked_outcome"]["ret_20"] for x in grows if x["subsequent_unmasked_outcome"]["ret_20"] is not None]),
                "mean_ret_60": mean([x["subsequent_unmasked_outcome"]["ret_60"] for x in grows if x["subsequent_unmasked_outcome"]["ret_60"] is not None]),
            }
        return out

    all_agg = conf.get("aggregate_by_gate", {})
    excl_set_b_agg = agg(non_set_b_events)

    deriving_fields = []
    for gate, row in all_agg.items():
        row2 = excl_set_b_agg.get(gate, {})
        if (
            row.get("count") != row2.get("count")
            or row.get("mean_ret_5") != row2.get("mean_ret_5")
            or row.get("mean_ret_20") != row2.get("mean_ret_20")
            or row.get("mean_ret_60") != row2.get("mean_ret_60")
        ):
            deriving_fields.append({"gate": gate, "all": row, "excluding_set_b": row2})

    arch_mentions_set_b = [ln for ln in arch_md.splitlines() if "set b" in ln.lower()]

    return {
        "set_b_event_count": len(set_b_events),
        "set_b_derivation_flag": len(deriving_fields) > 0,
        "derived_fields": deriving_fields,
        "architecture_set_b_lines": arch_mentions_set_b,
        "finding": "AGGREGATES_DERIVE_FROM_SET_B" if deriving_fields else "NO_SET_B_DERIVATION_DETECTED",
    }


def create_manifest_v1_1(files: list[str]) -> tuple[Path, Path, dict[str, Any]]:
    rows = []
    for rel in files:
        p = ROOT / rel
        rows.append({
            "path": rel,
            "sha256": sha256_file(p),
            "size_bytes": file_size(p),
        })

    manifest = {
        "scope": "R13 integrity verification addendum and generator sealing",
        "r13_step0_status": "INTEGRITY_GATE_EMITTED",
        "no_self_reference": True,
        "created_files": rows,
    }

    manifest_path = REVIEW_DIR / "r13_created_files_manifest_v1_1.json"
    sha_path = REVIEW_DIR / "r13_created_files_manifest_v1_1.sha256"
    write_json(manifest_path, manifest)
    msha = sha256_file(manifest_path)
    sha_path.write_text(
        f"{msha}  artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1_1.json\n",
        encoding="utf-8",
    )
    return manifest_path, sha_path, manifest


def main() -> None:
    conf = read_json(REVIEW_DIR / "r13_gate_conflict_analysis_v1.json")
    arch_md = (REVIEW_DIR / "r13_architecture_proposals_v1.md").read_text(encoding="utf-8")
    runtime_db = Path(conf.get("runtime_db") or (REVIEW_DIR / "r12_exam_surface_v4_5_runtime.db"))

    # 0a seam-safety
    code_quote = parse_generator_return_code_quotes(ROOT / "scripts" / "r13_generate_paper_outputs_v1.py")
    seam = verify_seam_safety(conf, runtime_db)

    # 0b read-only hash proof
    refs = load_reference_hashes()
    read_artifacts = [
        "artifacts/preview1a_prestart/review_final/r12_exam_surface_v4_5_runtime.db",
        "artifacts/preview1a_prestart/review_final/r12_exam_surface_v4_5.db",
        "artifacts/preview1a_prestart/review_final/r12_exam_results_v2_1.json",
        "artifacts/preview1a_prestart/review_final/r12_breach_triage_v4_2_FINAL.json",
        "artifacts/preview1a_prestart/review_final/r12_ca_ledger_v0_2.json",
        "artifacts/preview1a_prestart/review_final/r12_masked_intervals_manifest_v4_3_final.json",
        "artifacts/preview1a_prestart/review_final/r12_pre_exam_surface_seal_v4_4.json",
    ]

    baseline_overrides = {
        # runtime DB hash baseline was not sealed in addendum-10/11 or seal-v4.4
        "artifacts/preview1a_prestart/review_final/r12_exam_surface_v4_5_runtime.db": None,
        "artifacts/preview1a_prestart/review_final/r12_exam_surface_v4_5.db": refs.get("artifacts/preview1a_prestart/review_final/r12_exam_surface_v4_5.db"),
        "artifacts/preview1a_prestart/review_final/r12_exam_results_v2_1.json": refs.get("artifacts/preview1a_prestart/review_final/r12_exam_results_v2_1.json"),
        "artifacts/preview1a_prestart/review_final/r12_breach_triage_v4_2_FINAL.json": refs.get("artifacts/preview1a_prestart/review_final/r12_breach_triage_v4_2_FINAL.json"),
        "artifacts/preview1a_prestart/review_final/r12_ca_ledger_v0_2.json": refs.get("artifacts/preview1a_prestart/review_final/r12_ca_ledger_v0_2.json"),
        "artifacts/preview1a_prestart/review_final/r12_masked_intervals_manifest_v4_3_final.json": refs.get("artifacts/preview1a_prestart/review_final/r12_masked_intervals_manifest_v4_3_final.json"),
        "artifacts/preview1a_prestart/review_final/r12_pre_exam_surface_seal_v4_4.json": refs.get("artifacts/preview1a_prestart/review_final/r12_pre_exam_surface_seal_v4_4.json"),
    }

    hash_checks = [compare_hash(p, baseline_overrides.get(p)) for p in read_artifacts]

    # 0c generator sealing and replay check
    generator_paths = [
        "scripts/r13_generate_paper_outputs_v1.py",
        "scripts/r13_generate_addendum_1.py",
    ]
    generator_hashes = [
        {
            "path": p,
            "sha256": sha256_file(ROOT / p),
            "size_bytes": file_size(ROOT / p),
        }
        for p in generator_paths
    ]

    replay = sandbox_regeneration_compare()

    # 0d tier provenance
    tier_rule = {
        "rule_source_file": "scripts/r13_generate_paper_outputs_v1.py",
        "rule_function": "liquidity_tier",
        "rule": "HIGH if median_daily_value_traded_kwd >= 500000; MID if >= 100000; else LOW",
        "status": "AGENT_PROPOSED_UNRATIFIED",
        "alternative_for_owner_ratification": "terciles over median_daily_value_traded_kwd across full 139-symbol universe",
    }

    # 0e taxonomy completeness
    source_vocab = source_vocabulary_with_citations()
    runtime_vocab = distinct_runtime_values(runtime_db)

    mapped_rows = {
        "AVOID",
        "BASE_FORMING",
        "ACCUMULATION",
        "BREAKOUT_WATCH",
        "BREAKOUT_CONFIRMED",
        "SIGNAL_SUPPRESSED_RISK",
        "DISTRIBUTION_WARNING/EXIT",
    }

    coverage = {
        "phases": [
            {
                "item": p,
                "covered_in_r13_v1_mapping": p in mapped_rows,
                "status": "MAPPED" if p in mapped_rows else "N_A_OR_MISSING",
                "note": "Covered directly" if p in mapped_rows else "Not explicit in 7-row table; requires extension",
            }
            for p in source_vocab.get("phases", [])
        ],
        "signal_types_emittable": [
            {
                "item": s,
                "covered_in_r13_v1_mapping": s in {"SIGNAL_SUPPRESSED_RISK"},
                "status": "MAPPED" if s in {"SIGNAL_SUPPRESSED_RISK"} else "N_A_OR_MISSING",
                "note": "Mapped as gate-output row" if s == "SIGNAL_SUPPRESSED_RISK" else "Not explicit in 7-row table",
            }
            for s in source_vocab.get("signal_types_emittable", [])
        ],
        "suppression_reasons": [
            {
                "item": r,
                "covered_in_r13_v1_mapping": False,
                "status": "N_A_OR_MISSING",
                "note": "Suppression reason vocabulary not explicitly mapped in v1 table",
            }
            for r in source_vocab.get("suppression_reasons", [])
        ],
    }

    # 0f set B derivation audit
    setb_audit = set_b_derivation_audit(conf, arch_md)

    # 0g handoff standard
    standard_formula = "no engine, scanner, model, backtest, or market-data execution; read-only extraction and descriptive aggregation only."

    mismatch_exists = any(r["status"] == "MISMATCH" for r in hash_checks)
    seam_contamination = seam["contamination"]["contamination_detected"]
    runtime_no_baseline = any(r["path"].endswith("r12_exam_surface_v4_5_runtime.db") and r["status"] == "NO_BASELINE" for r in hash_checks)

    gate_pass = not mismatch_exists and replay.get("all_byte_identical", False) and not runtime_no_baseline

    integrity = {
        "version_id": "R13_INTEGRITY_VERIFICATION_V1",
        "generated_at_utc": ts_now(),
        "step0_status": "PASSED" if gate_pass else "FAILED",
        "blocking_reasons": [
            x for x in [
                "HASH_MISMATCH_DETECTED" if mismatch_exists else None,
                "RUNTIME_DB_BASELINE_HASH_NOT_RECORDED" if runtime_no_baseline else None,
                "GENERATOR_REPLAY_NOT_BYTE_IDENTICAL" if not replay.get("all_byte_identical", False) else None,
            ] if x is not None
        ],
        "0a_seam_safety": {
            "generator_code_quote": code_quote,
            **seam,
            "citable_original_figure_status": "REPLACED_BY_SEAM_SAFE_CORRECTED" if seam_contamination else "SURVIVES_SEAM_CHECK",
        },
        "0b_read_only_proof": {
            "analysis_connection_mode": "sqlite_uri_mode_ro",
            "hash_checks": hash_checks,
            "reference_sources": [
                "artifacts/preview1a_prestart/review_final/r12_pre_exam_surface_seal_v4_4.json",
                "artifacts/preview1a_prestart/review_final/r12a_created_files_manifest_v4_5_addendum_10.json",
                "artifacts/preview1a_prestart/review_final/r12a_created_files_manifest_v2_1_addendum_11.json",
            ],
        },
        "0c_generator_sealing": {
            "generator_hashes": generator_hashes,
            "replay_check": replay,
        },
        "0d_tier_boundary_provenance": tier_rule,
        "0e_taxonomy_completeness": {
            "source_vocabulary": source_vocab,
            "runtime_distinct_values": runtime_vocab,
            "coverage_vs_7_row_mapping": coverage,
        },
        "0f_set_b_derivation_audit": setb_audit,
        "0g_handoff_standard": {
            "required_formula": standard_formula,
            "future_reporting_requirement": "Each emitted file must include inline individual SHA-256.",
        },
    }

    out_json = REVIEW_DIR / "r13_integrity_verification_v1.json"
    write_json(out_json, integrity)

    md_lines = [
        "# R13 Integrity Verification v1",
        "",
        f"Step 0 status: {integrity['step0_status']}",
        "",
        "## Blocking Reasons",
    ]
    if integrity["blocking_reasons"]:
        for r in integrity["blocking_reasons"]:
            md_lines.append(f"- {r}")
    else:
        md_lines.append("- NONE")

    md_lines += [
        "",
        "## 0a Seam Safety",
        f"- contamination_detected: {str(seam_contamination).lower()}",
        f"- cross_segment_window_counts: {json.dumps(seam['contamination']['cross_segment_window_counts'], ensure_ascii=True)}",
        "- corrected_seam_safe_aggregate_by_gate:",
        f"{json.dumps(seam['corrected_seam_safe_aggregate_by_gate'], ensure_ascii=True, indent=2)}",
        "",
        "## 0b Read-only Proof",
        "- connection mode: file:...?mode=ro",
    ]
    for row in hash_checks:
        md_lines.append(
            f"- {row['path']} :: expected={row['expected_sha256']} current={row['current_sha256']} status={row['status']}"
        )

    md_lines += [
        "",
        "## 0c Generator Sealing",
    ]
    for g in generator_hashes:
        md_lines.append(f"- {g['path']} :: sha256={g['sha256']} size={g['size_bytes']}")
    md_lines.append(f"- replay all_byte_identical: {str(replay.get('all_byte_identical', False)).lower()}")

    md_lines += [
        "",
        "## 0d Tier Provenance",
        f"- rule: {tier_rule['rule']}",
        f"- status: {tier_rule['status']}",
        f"- alternative: {tier_rule['alternative_for_owner_ratification']}",
        "",
        "## 0e Taxonomy Completeness",
        f"- phases(source): {json.dumps(source_vocab.get('phases', []), ensure_ascii=True)}",
        f"- signal_types_emittable(source): {json.dumps(source_vocab.get('signal_types_emittable', []), ensure_ascii=True)}",
        f"- suppression_reasons(source): {json.dumps(source_vocab.get('suppression_reasons', []), ensure_ascii=True)}",
        f"- runtime distinct signal_type: {json.dumps(runtime_vocab.get('signal_type', []), ensure_ascii=True)}",
        f"- runtime distinct attempted_signal_type: {json.dumps(runtime_vocab.get('attempted_signal_type', []), ensure_ascii=True)}",
        f"- runtime distinct suppressed_reason: {json.dumps(runtime_vocab.get('suppressed_reason', []), ensure_ascii=True)}",
        "",
        "## 0f Set B Derivation Audit",
        f"- finding: {setb_audit['finding']}",
        f"- set_b_event_count: {setb_audit['set_b_event_count']}",
        "",
        "## 0g Permanent Standard",
        f"- {standard_formula}",
    ]

    out_md = REVIEW_DIR / "r13_integrity_verification_v1.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # manifest v1.1 (exclude self-reference intentionally)
    files_for_manifest = [
        "scripts/r13_generate_paper_outputs_v1.py",
        "scripts/r13_generate_addendum_1.py",
        "scripts/r13_integrity_verification_v1.py",
        "artifacts/preview1a_prestart/review_final/r13_state_taxonomy_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_state_taxonomy_v1.md",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1.md",
        "artifacts/preview1a_prestart/review_final/r13_architecture_proposals_v1.md",
        "artifacts/preview1a_prestart/review_final/r13_universe_tier_profile_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.sha256",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_addendum_1.json",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_addendum_1.md",
        "artifacts/preview1a_prestart/review_final/r13_integrity_verification_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_integrity_verification_v1.md",
    ]
    mpath, spath, _manifest = create_manifest_v1_1(files_for_manifest)

    print("R13_INTEGRITY_VERIFICATION_COMPLETE")
    print("step0_status", integrity["step0_status"])
    print("manifest", mpath)
    print("manifest_sidecar", spath)


if __name__ == "__main__":
    main()
