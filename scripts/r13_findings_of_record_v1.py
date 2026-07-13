from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_blocking_count(d1: dict[str, Any], symbol: str, term: str) -> int:
    cnt = 0
    for row in d1.get("owner_pattern_window_audit", {}).get("day_level", []):
        if row.get("symbol") == symbol and row.get("blocking_term", {}).get("term") == term:
            cnt += 1
    return cnt


def first_matching(rows: list[dict[str, Any]], symbol: str, date: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("symbol") == symbol and row.get("date") == date:
            return row
    return None


def main() -> None:
    d1 = read_json(REVIEW / "r13_set_a_causal_attribution_v3.json")
    vol = read_json(REVIEW / "r13_volume_arrival_audit_v1.json")
    gate = read_json(REVIEW / "r13_gate_conflict_analysis_v1_2.json")

    hi25_days = []
    for rows in vol.get("rel_volume_ge_2_5", {}).get("per_symbol_days", {}).values():
        hi25_days.extend(rows)

    bpcc_nearmiss = first_matching(hi25_days, "BPCC", "2025-04-22")
    tijara_hi25 = [r for r in hi25_days if r.get("symbol") == "TIJARA"]
    sanam_hi25 = [r for r in hi25_days if r.get("symbol") == "SANAM"]

    f3_confirmed = any(r.get("disposition") == "M1_close_gt_base" for r in tijara_hi25 + sanam_hi25)
    f3_status = "CONFIRMED" if f3_confirmed else "REFUTED"

    warm_rows = {r["symbol"]: r for r in d1.get("warmup_cost_quantification", {}).get("per_symbol", [])}
    totals = d1.get("per_symbol_category_counts_total", {})
    ex_set_b = gate.get("aggregates", {}).get("EX_SET_B", {}).get("aggregate_by_gate", {})

    md = []
    md.append("# R13 Findings Of Record v1")
    md.append("")
    md.append("## F1 Confirmation-Predicate Defect")
    md.append("- Statement: Breakout confirmation is dominated by same-day M2 relative-volume gating on trending Set A names.")
    md.append("- Code: [scanner_service.py#L713](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L713) through [scanner_service.py#L718](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L718)")
    if bpcc_nearmiss is not None:
        md.append(f"- BPCC 2025-04-22: rel_volume={bpcc_nearmiss.get('rel_volume')} vs threshold=2.5, close={bpcc_nearmiss.get('close')}, disposition={bpcc_nearmiss.get('disposition')} :: source [r13_volume_arrival_audit_v1.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json)")
    md.append(f"- Blocking frequency: TIJARA M2_rel_volume={find_blocking_count(d1,'TIJARA','M2_rel_volume')}, ZAIN M2_rel_volume={find_blocking_count(d1,'ZAIN','M2_rel_volume')}, BPCC M2_rel_volume={find_blocking_count(d1,'BPCC','M2_rel_volume')} :: source [r13_set_a_causal_attribution_v3.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json)")
    md.append("")
    md.append("## F2 Base-Geometry Defect")
    md.append("- Statement: base_max_width_pct=0.18 rejects wide Boursa bases, most visibly SANAM.")
    md.append("- Code: [scanner_service.py#L859](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L859)")
    md.append(f"- SANAM width<=0.18 blocked days in owner window: {find_blocking_count(d1,'SANAM','width <= base_max_width_pct')} :: source [r13_set_a_causal_attribution_v3.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json)")
    md.append("")
    md.append("## F3 Base-Reference Lifecycle Defect")
    md.append(f"- Status: {f3_status}")
    if f3_confirmed:
        md.append("- Evidence: On high-volume days where rel_volume finally arrives, M1_close_gt_base appears as the blocker, indicating missing/stale base reference lifecycle.")
    else:
        md.append("- Evidence: In the surfaced rel_volume>=2.5 window days for Set A, M1_close_gt_base does not dominate; the audit did not confirm a broad lifecycle failure as the primary explanation.")
    md.append(f"- TIJARA high-volume day dispositions include {len([r for r in tijara_hi25 if r.get('disposition')=='M1_close_gt_base'])} M1_close_gt_base days and {len([r for r in tijara_hi25 if r.get('disposition')=='M2_rel_volume'])} M2_rel_volume days :: source [r13_volume_arrival_audit_v1.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json)")
    md.append(f"- SANAM high-volume day dispositions include {len([r for r in sanam_hi25 if r.get('disposition')=='M1_close_gt_base'])} M1_close_gt_base days and {len([r for r in sanam_hi25 if r.get('disposition')=='M2_rel_volume'])} M2_rel_volume days :: source [r13_volume_arrival_audit_v1.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json)")
    md.append("")
    md.append("## F4 Warmup Structural Blindness")
    md.append("- Statement: Structural blindness consumes a material share of each symbol's history, especially SANAM.")
    for sym in ["BPCC","MABANEE","SANAM","TIJARA","ZAIN"]:
        row = warm_rows.get(sym)
        if row:
            md.append(f"- {sym}: blind_sessions={row['blind_sessions_from_segment_warmup']} blind_share={row['blind_share_unmasked']} :: source [r13_set_a_causal_attribution_v3.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json)")
    md.append("")
    md.append("## F5 Avoid Logic Validated")
    md.append("- Statement: Avoid logic behaves correctly on MABANEE's decline and should be preserved.")
    md.append(f"- MABANEE avoid-condition holds in owner window: {find_blocking_count(d1,'MABANEE','avoid_condition close < sma200 and sma200_slope < 0 and ema10 < ema30')} :: source [r13_set_a_causal_attribution_v3.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json)")
    md.append("- Code: [scanner_service.py#L463](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L463) and [scanner_service.py#L679](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L679)")
    md.append("")
    md.append("## F6 Gate-Warfare Hypothesis Retired")
    bpcc = totals.get('BPCC', {})
    tij = totals.get('TIJARA', {})
    zai = totals.get('ZAIN', {})
    md.append("- Statement: The dominant failure mode is no-candidate persistence, not gate warfare.")
    md.append(f"- BPCC no-candidate={bpcc.get('PHASE_PROGRESSED_NO_CANDIDATE')} vs risk_veto={bpcc.get('CANDIDATE_VETOED(RISK_SUPPRESSION)',0)} vs avoid_veto={bpcc.get('CANDIDATE_VETOED(AVOID_GATE)',0)}")
    md.append(f"- TIJARA no-candidate={tij.get('PHASE_PROGRESSED_NO_CANDIDATE')} vs avoid_veto={tij.get('CANDIDATE_VETOED(AVOID_GATE)',0)}")
    md.append(f"- ZAIN no-candidate={zai.get('PHASE_PROGRESSED_NO_CANDIDATE')} vs risk_veto={zai.get('CANDIDATE_VETOED(RISK_SUPPRESSION)',0)} vs avoid_veto={zai.get('CANDIDATE_VETOED(AVOID_GATE)',0)}")
    md.append(f"- Set A total risk vetoes={sum(v.get('CANDIDATE_VETOED(RISK_SUPPRESSION)',0) for v in totals.values())}; total avoid vetoes={sum(v.get('CANDIDATE_VETOED(AVOID_GATE)',0) for v in totals.values())} :: source [r13_set_a_causal_attribution_v3.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json)")
    md.append("")
    md.append("## F7 Telemetry Gap")
    md.append("- Statement: base_high_ref and liquidity_ok are not durably persisted at day-level in sealed records, making M1/M4/M5 unresolved for non-signal owner-window days.")
    md.append("- State persistence stores current state only: [scanner_service.py#L108](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L108) through [scanner_service.py#L136](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L136)")
    md.append("- Signal persistence stores evidence only when a signal is emitted: [scanner_service.py#L288](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L288) through [scanner_service.py#L340](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L340)")
    md.append("- base_high_ref is injected into payload only inside breakout confirmation evaluation: [scanner_service.py#L770](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L770)")
    md.append("- M5_liquidity depends on liquidity_ok runtime boolean in mandatory block: [scanner_service.py#L718](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L718)")
    md.append("")
    md.append("## Surviving Suppression-Cost Evidence")
    md.append(f"- EX_SET_B RISK_SUPPRESSION mean_ret_20={ex_set_b.get('RISK_SUPPRESSION',{}).get('mean_ret_20')} mean_ret_60={ex_set_b.get('RISK_SUPPRESSION',{}).get('mean_ret_60')} truncations={ex_set_b.get('RISK_SUPPRESSION',{}).get('truncations')} :: source [r13_gate_conflict_analysis_v1_2.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json)")
    md.append("")
    md.append("## Governance Findings")
    md.append("- Gate patch occurred during gate run and was later audited/reinstated under evidence review.")
    md.append("- Self-certified compliance failures occurred around citation validation and earlier deliverable gating.")
    md.append("- Temp script usage occurred in prior report-only surfacing runs; permanent-script rule now extends to all executed scripts.")
    md.append("")
    md.append("R14 remains NOT AUTHORIZED.")
    md.append("")

    out = REVIEW / "r13_findings_of_record_v1.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("R13_FINDINGS_OF_RECORD_V1_COMPLETE")
    print("sha256", sha256_file(out))


if __name__ == "__main__":
    main()
