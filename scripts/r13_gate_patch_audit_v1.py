from __future__ import annotations

import hashlib
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
SANDBOX = ROOT / "_tmp_r13_repro_sandbox" / "artifacts" / "preview1a_prestart" / "review_final"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_text_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def line_diffs(a_lines: list[str], b_lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, (la, lb) in enumerate(zip_longest(a_lines, b_lines, fillvalue=None), start=1):
        if la != lb:
            out.append({"line": i, "sealed": la, "replay": lb})
    return out


def main() -> None:
    script_path = ROOT / "scripts" / "r13_integrity_remediation_v1.py"
    manifest_live = REVIEW / "r13_created_files_manifest_v1.json"
    manifest_rep = SANDBOX / "r13_created_files_manifest_v1.json"
    side_live = REVIEW / "r13_created_files_manifest_v1.sha256"
    side_rep = SANDBOX / "r13_created_files_manifest_v1.sha256"
    gate_v12_path = REVIEW / "r13_gate_conflict_analysis_v1_2.json"
    seta_path = REVIEW / "r13_set_a_causal_attribution_v1.json"
    prop_path = REVIEW / "r13_architecture_proposals_v2.md"

    gate_v12 = json.loads(gate_v12_path.read_text(encoding="utf-8"))
    seta = json.loads(seta_path.read_text(encoding="utf-8"))
    proposals_text = prop_path.read_text(encoding="utf-8")
    manifest_live_obj = json.loads(manifest_live.read_text(encoding="utf-8"))
    manifest_rep_obj = json.loads(manifest_rep.read_text(encoding="utf-8"))

    patch1_before = [
        "        if not norm_diffs:",
        '            cls = "NONDETERMINISM"',
        "        else:",
        '            cls = "SUBSTANTIVE"',
    ]
    patch1_after = [
        "        if not norm_diffs:",
        '            cls = "NONDETERMINISM"',
        '        elif rel.endswith("r13_created_files_manifest_v1.json"):',
        "            # Manifest differences can be purely derivative (hash/size drift from upstream",
        "            # nondeterministic files). Treat hash/size-only path changes as NONDETERMINISM.",
        "            allowed = True",
        "            for d in raw_diffs:",
        '                p = str(d.get("path") or "")',
        '                if not re.match(r"^\\$\\.created_files\\[\\d+\\]\\.(sha256|size_bytes)$", p):',
        "                    allowed = False",
        "                    break",
        '            cls = "NONDETERMINISM" if allowed else "SUBSTANTIVE"',
        "        else:",
        '            cls = "SUBSTANTIVE"',
    ]
    patch2_before = ["import hashlib", "import json", "import sqlite3"]
    patch2_after = ["import hashlib", "import json", "import re", "import sqlite3"]

    manifest_line_diffs = line_diffs(read_text_lines(manifest_live), read_text_lines(manifest_rep))
    side_line_diffs = line_diffs(read_text_lines(side_live), read_text_lines(side_rep))

    live_cf = manifest_live_obj.get("created_files", [])
    rep_cf = manifest_rep_obj.get("created_files", [])

    obj_diffs: list[dict[str, Any]] = []
    for i in range(max(len(live_cf), len(rep_cf))):
        lv = live_cf[i] if i < len(live_cf) else {}
        rv = rep_cf[i] if i < len(rep_cf) else {}
        for field in sorted(set(lv.keys()) | set(rv.keys())):
            if lv.get(field) != rv.get(field):
                obj_diffs.append(
                    {
                        "path": f"$.created_files[{i}].{field}",
                        "sealed": lv.get(field),
                        "replay": rv.get(field),
                        "referenced_file_path": lv.get("path") or rv.get("path"),
                    }
                )

    allowed_manifest_fields = {"sha256", "size_bytes"}
    manifest_derivative_only = all(
        d["path"].startswith("$.created_files[") and d["path"].split(".")[-1] in allowed_manifest_fields for d in obj_diffs
    )

    referenced_changed_files = sorted({d["referenced_file_path"] for d in obj_diffs})
    side_ref_line = side_line_diffs[0]["sealed"] if side_line_diffs else ""
    side_ref_target = side_ref_line.split("  ", 1)[1] if "  " in side_ref_line else None

    nondeterminism_files_from_remediation = [
        "artifacts/preview1a_prestart/review_final/r13_state_taxonomy_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_universe_tier_profile_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.json",
    ]

    all_referenced = sorted(set(referenced_changed_files + ([side_ref_target] if side_ref_target else [])))
    all_referenced_in_nondet = all(x in nondeterminism_files_from_remediation for x in all_referenced)

    ex_gate = gate_v12["aggregates"]["EX_SET_B"]["aggregate_by_gate"]
    all_gate = gate_v12["aggregates"]["ALL_SYMBOLS"]["aggregate_by_gate"]
    ex_gate_tier = gate_v12["aggregates"]["EX_SET_B"]["aggregate_by_gate_and_liquidity_tier"]
    all_gate_tier = gate_v12["aggregates"]["ALL_SYMBOLS"]["aggregate_by_gate_and_liquidity_tier"]

    seta_forensics = [r for r in seta.get("set_a_no_trade_forensics_reference", []) if r.get("symbol") in {"TIJARA", "BPCC", "SANAM"}]
    five_categories = ["AVOID_SET", "DISTRIBUTION_WARNING", "EXIT", "PHASE_ONLY", "SIGNAL_SUPPRESSED_RISK"]
    counts_five: list[dict[str, Any]] = []
    for r in sorted(seta_forensics, key=lambda x: x["symbol"]):
        c = r.get("signal_type_counts", {})
        counts_five.append(
            {
                "symbol": r["symbol"],
                "signal_type_counts_verbatim": c,
                "five_category_projection": {k: int(c.get(k, 0)) for k in five_categories},
            }
        )

    predicate_defs = {
        "simple_breakout": {
            "coded_predicate": "close_i > max_prev",
            "threshold_form": "close_minus_prev20_high > 0",
        },
        "price_plus_relative_volume": {
            "coded_predicate": "close_i > max_prev AND vol_i > avg_vol",
            "threshold_form": "close_minus_prev20_high > 0 AND relative_volume_minus_1 > 0",
        },
    }

    trigger_eval: list[dict[str, Any]] = []
    for r in sorted(seta_forensics, key=lambda x: x["symbol"]):
        sym = r["symbol"]
        for chk in r.get("benchmark_active_day_checks", []):
            if not chk.get("active"):
                continue
            b = chk["benchmark"]
            close_delta = chk.get("close_minus_prev20_high")
            rv_delta = chk.get("relative_volume_minus_1")
            if b == "simple_breakout":
                pred_ok = close_delta is not None and close_delta > 0
            else:
                pred_ok = close_delta is not None and close_delta > 0 and rv_delta is not None and rv_delta > 0

            trigger_eval.append(
                {
                    "symbol": sym,
                    "benchmark": b,
                    "benchmark_trigger_date": chk.get("benchmark_trigger_date"),
                    "failed_transition_predicate_as_coded": "benchmark triggered but actual_trade_count == 0 (status FAIL/NO_SIGNAL)",
                    "coded_trigger_predicate": predicate_defs[b]["coded_predicate"],
                    "threshold_form": predicate_defs[b]["threshold_form"],
                    "computed_values": {
                        "close_minus_prev20_high": close_delta,
                        "relative_volume_minus_1": rv_delta,
                    },
                    "threshold_evaluation": {
                        "close_minus_prev20_high_gt_0": None if close_delta is None else bool(close_delta > 0),
                        "relative_volume_minus_1_gt_0": None if rv_delta is None else bool(rv_delta > 0),
                        "coded_trigger_predicate_true": bool(pred_ok),
                    },
                    "observed_engine_outcome": {
                        "trade_count": r.get("trade_count"),
                        "primary_blocker": r.get("primary_blocker"),
                        "nearest_engine_signal_date": chk.get("nearest_engine_signal_date"),
                        "nearest_engine_signal_type": chk.get("nearest_engine_signal_type"),
                        "nearest_engine_phase_to": chk.get("nearest_engine_phase_to"),
                    },
                }
            )

    citation_index = {
        "Proposal A - Sequential Tri-Model With Hard Data-Surface Gate": [
            {
                "symbol": None,
                "date": None,
                "value": gate_v12["aggregates"]["EX_SET_B"]["aggregate_by_gate"]["AVOID_GATE"]["mean_ret_60"],
                "metric": "EX_SET_B.AVOID_GATE.mean_ret_60",
                "source_artifact": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
            }
        ],
        "Proposal B - Parallel Specialists + Evidence Council": [
            {
                "symbol": None,
                "date": None,
                "value": gate_v12["aggregates"]["EX_SET_B"]["aggregate_by_gate"]["RISK_SUPPRESSION"]["mean_ret_20"],
                "metric": "EX_SET_B.RISK_SUPPRESSION.mean_ret_20",
                "source_artifact": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
            },
            {
                "symbol": "BPCC",
                "date": "2021-08-26",
                "value": 3.630416211257269,
                "metric": "relative_volume_minus_1 on benchmark-active trigger day",
                "source_artifact": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v1.json",
            },
        ],
        "Proposal C - Regime-First Controller": [
            {
                "symbol": None,
                "date": None,
                "value": gate_v12["aggregates"]["EX_SET_B"]["aggregate_by_gate"]["WARMUP_GATE"]["mean_ret_60"],
                "metric": "EX_SET_B.WARMUP_GATE.mean_ret_60",
                "source_artifact": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
            },
            {
                "symbol": "TIJARA",
                "date": "2021-09-09",
                "value": 0.5255211376257112,
                "metric": "relative_volume_minus_1 on benchmark-active trigger day",
                "source_artifact": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v1.json",
            },
            {
                "symbol": "SANAM",
                "date": "2021-03-23",
                "value": 1.2828780700349434,
                "metric": "relative_volume_minus_1 on benchmark-active trigger day",
                "source_artifact": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v1.json",
            },
        ],
    }

    hashes: dict[str, dict[str, Any]] = {}
    for rel in [
        "scripts/r13_integrity_remediation_v1.py",
        "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.sha256",
        "_tmp_r13_repro_sandbox/artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.json",
        "_tmp_r13_repro_sandbox/artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.sha256",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
        "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_architecture_proposals_v2.md",
    ]:
        p = ROOT / rel
        hashes[rel] = {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}

    payload = {
        "version_id": "R13_GATE_PATCH_AUDIT_V1",
        "governance_rule_acknowledgement": {
            "rule": "verification logic is frozen once a gate run begins; any proposed classifier change is reported with evidence and awaits directive before re-run.",
            "acknowledged": True,
            "effective": "PERMANENT",
        },
        "scope": "report-only audit; no regeneration, no reruns, no new classification changes",
        "file_hashes_inline": hashes,
        "classifier_patch_evidence": {
            "patch_1_manifest_reclassification_logic": {
                "before_code": patch1_before,
                "after_code": patch1_after,
            },
            "patch_2_missing_import_for_classifier_regex": {
                "before_code": patch2_before,
                "after_code": patch2_after,
            },
        },
        "manifest_replay_diff_evidence": {
            "manifest_live_path": "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.json",
            "manifest_replay_path": "_tmp_r13_repro_sandbox/artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.json",
            "manifest_line_diffs_all": manifest_line_diffs,
            "manifest_object_diffs_all": obj_diffs,
            "sidecar_live_path": "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.sha256",
            "sidecar_replay_path": "_tmp_r13_repro_sandbox/artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.sha256",
            "sidecar_line_diffs_all": side_line_diffs,
        },
        "derivative_reclassification_proof": {
            "non_determinism_reference_files": nondeterminism_files_from_remediation,
            "referenced_files_changed_by_manifest_diff": referenced_changed_files,
            "referenced_file_changed_by_sidecar_diff": side_ref_target,
            "all_referenced_changed_files_union": all_referenced,
            "manifest_diffs_hash_size_only": manifest_derivative_only,
            "all_referenced_files_preclassified_nondeterminism": all_referenced_in_nondet,
            "non_derivative_line_detected": not (manifest_derivative_only and all_referenced_in_nondet),
            "gate_status_if_non_derivative_present": "FAILED",
        },
        "citable_evidence_surface": {
            "source_artifact": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
            "constraints_verbatim": gate_v12["constraints"],
            "aggregate_counts_by_gate_verbatim": gate_v12["aggregate_counts_by_gate"],
            "EX_SET_B_aggregate_by_gate_verbatim": ex_gate,
            "ALL_SYMBOLS_aggregate_by_gate_verbatim": all_gate,
            "EX_SET_B_aggregate_by_gate_and_tier_verbatim": ex_gate_tier,
            "ALL_SYMBOLS_aggregate_by_gate_and_tier_verbatim": all_gate_tier,
            "tier_rule_verbatim": gate_v12.get("tier_rule", {}),
            "alternate_tier_gate_matrix_present_in_source": False,
        },
        "deliverable_1_surface": {
            "source_artifact": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v1.json",
            "per_symbol_day_classification_counts": counts_five,
            "trigger_day_failed_transition_predicates": trigger_eval,
        },
        "deliverable_2_surface": {
            "source_artifact": "artifacts/preview1a_prestart/review_final/r13_architecture_proposals_v2.md",
            "proposals_text_verbatim": proposals_text,
            "citation_index": citation_index,
        },
        "authorization_status": "R14_NOT_AUTHORIZED",
    }

    out_json = REVIEW / "r13_gate_patch_audit_v1.json"
    out_md = REVIEW / "r13_gate_patch_audit_v1.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# R13 Gate Patch Audit v1")
    lines.append("")
    lines.append(
        "Rule acknowledgement: verification logic is frozen once a gate run begins; any proposed classifier change is reported with evidence and awaits directive before re-run."
    )
    lines.append("")
    lines.append("R14 remains NOT AUTHORIZED.")
    lines.append("")
    lines.append("## Per-file hashes (inline)")
    for rel, hv in hashes.items():
        lines.append(f"- {rel} :: sha256={hv['sha256']} size_bytes={hv['size_bytes']}")
    lines.append("")
    lines.append("## 1) Classifier patches before/after")
    lines.append("Patch 1 (manifest classification logic) BEFORE:")
    lines.append("```python")
    lines.extend(patch1_before)
    lines.append("```")
    lines.append("Patch 1 AFTER:")
    lines.append("```python")
    lines.extend(patch1_after)
    lines.append("```")
    lines.append("Patch 2 (missing regex import) BEFORE:")
    lines.append("```python")
    lines.extend(patch2_before)
    lines.append("```")
    lines.append("Patch 2 AFTER:")
    lines.append("```python")
    lines.extend(patch2_after)
    lines.append("```")
    lines.append("")
    lines.append("## 1b) Manifest + sidecar actual differing lines (all)")
    lines.append("Manifest differing lines:")
    for d in manifest_line_diffs:
        lines.append(f"- line {d['line']}:")
        lines.append(f"  sealed: {d['sealed']}")
        lines.append(f"  replay: {d['replay']}")
    lines.append("Sidecar differing lines:")
    for d in side_line_diffs:
        lines.append(f"- line {d['line']}:")
        lines.append(f"  sealed: {d['sealed']}")
        lines.append(f"  replay: {d['replay']}")
    lines.append("")
    lines.append("Derivative proof summary:")
    lines.append(f"- manifest_diffs_hash_size_only={manifest_derivative_only}")
    lines.append(f"- all_referenced_files_preclassified_nondeterminism={all_referenced_in_nondet}")
    lines.append(f"- non_derivative_line_detected={not (manifest_derivative_only and all_referenced_in_nondet)}")
    if not (manifest_derivative_only and all_referenced_in_nondet):
        lines.append("- classification verdict: WRONG; gate must revert to FAILED")
    else:
        lines.append("- classification verdict: DERIVATIVE-ONLY (no key/path/count/scope drift detected)")
    lines.append("")
    lines.append("## 2) Citable evidence base (verbatim blocks + ALL_SYMBOLS equivalents)")
    lines.append("aggregate_counts_by_gate (verbatim):")
    lines.append("```json")
    lines.append(json.dumps(gate_v12["aggregate_counts_by_gate"], ensure_ascii=True, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("EX_SET_B aggregate_by_gate (verbatim):")
    lines.append("```json")
    lines.append(json.dumps(ex_gate, ensure_ascii=True, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("ALL_SYMBOLS aggregate_by_gate (equivalent):")
    lines.append("```json")
    lines.append(json.dumps(all_gate, ensure_ascii=True, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("EX_SET_B aggregate_by_gate_and_liquidity_tier (verbatim):")
    lines.append("```json")
    lines.append(json.dumps(ex_gate_tier, ensure_ascii=True, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("ALL_SYMBOLS aggregate_by_gate_and_liquidity_tier (equivalent):")
    lines.append("```json")
    lines.append(json.dumps(all_gate_tier, ensure_ascii=True, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("tier_rule (verbatim):")
    lines.append("```json")
    lines.append(json.dumps(gate_v12.get("tier_rule", {}), ensure_ascii=True, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## 3) Deliverable 1 surfacing")
    lines.append("Per-symbol day-classification counts (TIJARA/BPCC/SANAM):")
    lines.append("```json")
    lines.append(json.dumps(counts_five, ensure_ascii=True, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("Named trigger-day failed transition predicate, coded form, computed values vs thresholds:")
    lines.append("```json")
    lines.append(json.dumps(trigger_eval, ensure_ascii=True, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## 4) Deliverable 2 surfacing")
    lines.append("Complete proposals text (verbatim):")
    lines.append("```markdown")
    lines.append(proposals_text.rstrip("\n"))
    lines.append("```")
    lines.append("Citation index per proposal:")
    lines.append("```json")
    lines.append(json.dumps(citation_index, ensure_ascii=True, indent=2, sort_keys=True))
    lines.append("```")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("R13_GATE_PATCH_AUDIT_V1_WRITTEN")


if __name__ == "__main__":
    main()
