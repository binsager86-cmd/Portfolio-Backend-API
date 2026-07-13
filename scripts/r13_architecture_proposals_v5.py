from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_citations(d1: dict[str, Any], vol: dict[str, Any], gate: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    def first_day(sym: str, disposition: str | None = None) -> dict[str, Any]:
        rows = [r for r in d1.get("owner_pattern_window_audit", {}).get("day_level", []) if r.get("symbol") == sym]
        if disposition is not None:
            rows = [r for r in rows if r.get("blocking_term", {}).get("term") == disposition]
        if not rows:
            raise ValueError(f"No day rows for {sym} disposition={disposition}")
        return rows[0]

    def vol_day(sym: str, threshold_key: str) -> dict[str, Any]:
        rows = vol.get(threshold_key, {}).get("per_symbol_days", {}).get(sym, [])
        if not rows:
            raise ValueError(f"No volume rows for {sym} {threshold_key}")
        return rows[0]

    def warm_row(sym: str) -> dict[str, Any]:
        for r in d1.get("warmup_cost_quantification", {}).get("per_symbol", []):
            if r.get("symbol") == sym:
                return r
        raise ValueError(sym)

    ex = gate.get("aggregates", {}).get("EX_SET_B", {}).get("aggregate_by_gate", {})

    return {
        "Proposal A": [
            {
                "symbol": "BPCC",
                "date": first_day("BPCC", "M2_rel_volume").get("trade_date_iso"),
                "value": first_day("BPCC", "M2_rel_volume").get("blocking_term", {}).get("term"),
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
                "metric": "owner-window blocker",
                "level": "day",
            },
            {
                "symbol": "BPCC",
                "date": vol_day("BPCC", "rel_volume_ge_2_0").get("date"),
                "value": vol_day("BPCC", "rel_volume_ge_2_0").get("rel_volume"),
                "source": "artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json",
                "metric": "volume-arrival rel_volume",
                "level": "day",
            },
            {
                "symbol": "BPCC",
                "date": "2025-04-22",
                "value": 2.1563101859019342,
                "source": "artifacts/preview1a_prestart/review_final/r13_findings_of_record_v1.md",
                "metric": "F1 BPCC near-miss rel_volume",
                "level": "finding",
            },
            {
                "symbol": "ALL_EX_SET_B",
                "date": None,
                "date_exempt_reason": "aggregate citation",
                "value": ex.get("RISK_SUPPRESSION", {}).get("mean_ret_20"),
                "source": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
                "metric": "EX_SET_B.RISK_SUPPRESSION.mean_ret_20",
                "level": "aggregate",
            },
        ],
        "Proposal B": [
            {
                "symbol": "SANAM",
                "date": first_day("SANAM", "width <= base_max_width_pct").get("trade_date_iso"),
                "value": first_day("SANAM", "width <= base_max_width_pct").get("blocking_term", {}).get("term"),
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
                "metric": "owner-window blocker",
                "level": "day",
            },
            {
                "symbol": "SANAM",
                "date": vol_day("SANAM", "rel_volume_ge_2_5").get("date"),
                "value": vol_day("SANAM", "rel_volume_ge_2_5").get("disposition"),
                "source": "artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json",
                "metric": "volume-arrival disposition",
                "level": "day",
            },
            {
                "symbol": "SANAM",
                "date": first_day("SANAM", "width <= base_max_width_pct").get("trade_date_iso"),
                "value": 95,
                "source": "artifacts/preview1a_prestart/review_final/r13_findings_of_record_v1.md",
                "metric": "F2 blocked-day count",
                "level": "finding",
            },
        ],
        "Proposal C": [
            {
                "symbol": "TIJARA",
                "date": first_day("TIJARA", "M2_rel_volume").get("trade_date_iso"),
                "value": first_day("TIJARA", "M2_rel_volume").get("blocking_term", {}).get("term"),
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
                "metric": "owner-window blocker",
                "level": "day",
            },
            {
                "symbol": "TIJARA",
                "date": vol_day("TIJARA", "rel_volume_ge_2_5").get("date") if vol.get("rel_volume_ge_2_5", {}).get("per_symbol_days", {}).get("TIJARA") else vol_day("TIJARA", "rel_volume_ge_2_0").get("date"),
                "value": vol_day("TIJARA", "rel_volume_ge_2_5").get("disposition") if vol.get("rel_volume_ge_2_5", {}).get("per_symbol_days", {}).get("TIJARA") else vol_day("TIJARA", "rel_volume_ge_2_0").get("disposition"),
                "source": "artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json",
                "metric": "volume-arrival disposition",
                "level": "day",
            },
            {
                "symbol": "TIJARA",
                "date": first_day("TIJARA", "M2_rel_volume").get("trade_date_iso"),
                "value": next(r.get("blind_share_unmasked") for r in d1.get("warmup_cost_quantification", {}).get("per_symbol", []) if r.get("symbol") == "TIJARA"),
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
                "metric": "warmup blindness share",
                "level": "symbol_metric",
            },
            {
                "symbol": "TIJARA",
                "date": "2025-04-23",
                "value": 154,
                "source": "artifacts/preview1a_prestart/review_final/r13_findings_of_record_v1.md",
                "metric": "F1 TIJARA M2 blocked-day count",
                "level": "finding",
            },
        ],
    }


def validate_gate(citations: dict[str, list[dict[str, Any]]], set_b: set[str]) -> dict[str, Any]:
    checks = []
    aggregate_proposals = 0
    for proposal, rows in citations.items():
        checks.append({"check": f"{proposal}_min_3_citations", "pass": len(rows) >= 3, "value": len(rows)})
        for idx, row in enumerate(rows):
            missing = [k for k in ["symbol", "value", "source", "metric", "level"] if k not in row]
            checks.append({"check": f"{proposal}_citation_{idx}_required_fields", "pass": len(missing) == 0, "missing": missing})
            is_aggregate = str(row.get("level")) == "aggregate"
            date = row.get("date")
            date_ok = is_aggregate or (date not in {None, "", "UNKNOWN"})
            value = row.get("value")
            value_ok = value is not None and value != "" and not (isinstance(value, list) and len(value) == 0)
            checks.append({"check": f"{proposal}_citation_{idx}_date_gate", "pass": date_ok, "date": date, "aggregate_exempt": is_aggregate})
            checks.append({"check": f"{proposal}_citation_{idx}_value_gate", "pass": value_ok, "value": value})
            sym = str(row.get("symbol"))
            checks.append({"check": f"{proposal}_citation_{idx}_no_set_b", "pass": sym not in set_b, "symbol": sym})
        has_finding = any("findings_of_record" in str(r.get("source")) for r in rows)
        has_d1 = any("set_a_causal_attribution_v3" in str(r.get("source")) for r in rows)
        has_vol = any("volume_arrival_audit" in str(r.get("source")) for r in rows)
        checks.append({"check": f"{proposal}_has_d1", "pass": has_d1})
        checks.append({"check": f"{proposal}_has_volume_audit", "pass": has_vol})
        checks.append({"check": f"{proposal}_has_findings_record", "pass": has_finding})
        if any(str(r.get("level")) == "aggregate" for r in rows):
            aggregate_proposals += 1
    checks.append({"check": "aggregate_level_citations_at_most_one_proposal", "pass": aggregate_proposals <= 1, "value": aggregate_proposals})
    ok = all(bool(c.get("pass")) for c in checks)
    return {"status": "PASS" if ok else "FAIL", "checks": checks}


def build_text(d1: dict[str, Any], vol: dict[str, Any], gate: dict[str, Any], citations: dict[str, list[dict[str, Any]]], gatecheck: dict[str, Any]) -> str:
    tier_rule = gate.get("tier_rule", {})
    warm_tier = d1.get("warmup_cost_quantification", {}).get("per_liquidity_tier", {})
    f3_status = "CONFIRMED" if any(r.get("disposition") == "M1_close_gt_base" for rows in vol.get("rel_volume_ge_2_5", {}).get("per_symbol_days", {}).values() for r in rows if r.get("symbol") in {"TIJARA","SANAM","BPCC"}) else "REFUTED"
    lines = [
        "# R13 Three-Model Architecture Proposals v5",
        "",
        "Universe assumptions (dual-stated):",
        f"- Proposed threshold rule: {tier_rule.get('rule')}",
        f"- Tercile alternative: {tier_rule.get('alternative')}",
        "",
        "Named findings addressed:",
        "- F1 confirmation-predicate defect",
        "- F2 base-geometry defect",
        f"- F3 base-reference lifecycle defect: {f3_status}",
        "- F4 warmup structural blindness",
        "- F5 preserve avoid logic",
        "- F7 telemetry gap",
        "",
        "## Proposal A - Flow-Aware Confirmation Architecture",
        "Answer to F1:",
        "- Replace single-day multiple as sole confirmation gate with accumulated flow evidence already computed in ee_indicators (obv_slope_40, anv_slope_40, accumulation_divergence) plus breakout structure.",
        "Answer to F2:",
        "- No direct geometry change; proposal assumes base geometry remains external and is paired with Proposal B if needed.",
        "Answer to F3:",
        f"- Under current evidence status {f3_status}, base-reference lifecycle is not the primary mechanism to solve first; confirmation redesign is primary.",
        "Answer to F7:",
        "- Persist term-level confirmation traces on every evaluation day, not only on signal days.",
        "Preserve F5:",
        "- Leave avoid-condition logic intact as a separate safety plane.",
        "Address F4:",
        "- Surface readiness state explicitly so warmup blindness is auditable and not conflated with rejection.",
        "Falsifiable predictions:",
        "- TIJARA 2024-2026 PHASE_PROGRESSED_NO_CANDIDATE share falls materially under a confirmation re-exam using flow-aware confirmation.",
        "- BPCC-like near-miss days should stop failing solely on M2 when EMA structure is already aligned.",
        "",
        "## Proposal B - Adaptive Base Geometry Architecture",
        "Answer to F1:",
        "- Confirmation remains multi-input, but geometry must stop discarding valid wide Boursa bases before confirmation even has a chance.",
        "Answer to F2:",
        "- Replace fixed-width base geometry with adaptive, regime-aware geometry derived from local volatility and listing behavior, without hard-coding a new numeric threshold here.",
        "Answer to F3:",
        f"- If F3 remains {f3_status}, lifecycle handling is a secondary concern after geometry admits valid base states.",
        "Answer to F7:",
        "- Persist frozen base references and base validity rationale day-by-day.",
        "Preserve F5:",
        "- Avoid logic remains unchanged and continues to suppress obvious downtrends like MABANEE's decline.",
        "Address F4:",
        "- Adaptive base geometry must be warmup-aware and segment-aware so new listings and segment restarts do not inherit impossible geometry prerequisites.",
        "Falsifiable predictions:",
        "- SANAM owner-window width-block share declines materially under re-exam.",
        "- Geometry-admitted windows should produce fewer NEUTRAL/BASE_FORMING stalls on progressing patterns.",
        "",
        "## Proposal C - Stateful Lifecycle + Deferred Intent Architecture",
        "Answer to F1:",
        "- Confirmation and execution are decoupled: intent can survive delayed volume arrival instead of requiring same-bar alignment of all mandatory terms.",
        "Answer to F2:",
        "- Frozen base references persist through quiet markup transitions until explicitly invalidated.",
        "Answer to F3:",
        f"- Directly addresses lifecycle failure mode, whether {f3_status.lower()} as dominant or secondary, by persisting base reference state across delayed-volume arrival.",
        "Answer to F7:",
        "- Persist base_high_ref, gap_pct_base, liquidity_ok, and per-term pass/fail as daily telemetry.",
        "Preserve F5:",
        "- Avoid plane remains separate from intent queue and can still veto execution in true downtrends.",
        "Address F4:",
        "- Readiness and segment resets become explicit state transitions rather than silent blind periods.",
        "Falsifiable predictions:",
        "- High-volume arrival days that currently fail for unresolved lifecycle reasons should convert to explicit deferred-intent or confirmed states in replay.",
        "- The share of unresolved M1/M4/M5 owner-window rows falls toward zero once telemetry is persisted.",
        "",
        "## Citation Gate Output",
        "```json",
        json.dumps(gatecheck, ensure_ascii=True, indent=2, sort_keys=True),
        "```",
        "",
        "## Citation Index",
    ]
    for name in ["Proposal A", "Proposal B", "Proposal C"]:
        lines.append(f"### {name}")
        for row in citations[name]:
            lines.append("- " + json.dumps(row, ensure_ascii=True, sort_keys=True))
        lines.append("")
    lines += [
        "Warmup tier summary:",
        json.dumps(warm_tier, ensure_ascii=True, sort_keys=True),
        "",
        "Constraint formula: no engine, scanner, model, backtest, or market-data execution; read-only extraction and descriptive aggregation only.",
        "R14 remains NOT AUTHORIZED.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    d1 = read_json(REVIEW / "r13_set_a_causal_attribution_v3.json")
    vol = read_json(REVIEW / "r13_volume_arrival_audit_v1.json")
    gate = read_json(REVIEW / "r13_gate_conflict_analysis_v1_2.json")
    citations = build_citations(d1, vol, gate)
    set_b = set(gate.get("set_membership", {}).get("set_b", []))
    gatecheck = validate_gate(citations, set_b)
    if gatecheck.get("status") != "PASS":
        fail = REVIEW / "r13_architecture_proposals_v5_gatecheck_fail.json"
        fail.write_text(json.dumps(gatecheck, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("R13_ARCHITECTURE_PROPOSALS_V5_GATECHECK_FAIL")
        raise SystemExit(1)
    out = REVIEW / "r13_architecture_proposals_v5.md"
    out.write_text(build_text(d1, vol, gate, citations, gatecheck), encoding="utf-8")
    print("R13_ARCHITECTURE_PROPOSALS_V5_COMPLETE")
    print("sha256", sha256_file(out))


if __name__ == "__main__":
    main()
