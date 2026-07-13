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


def find_d1_row(d1: dict[str, Any], symbol: str, date_iso: str) -> dict[str, Any] | None:
    for r in d1.get("benchmark_active_day_rows", []):
        if r.get("symbol") == symbol and r.get("trade_date_iso") == date_iso:
            return r
    return None


def build_citations(gate: dict[str, Any], d1: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    set_b = set(gate.get("set_membership", {}).get("set_b", []))

    d1_tijara = find_d1_row(d1, "TIJARA", "2021-09-09")
    d1_bpcc = find_d1_row(d1, "BPCC", "2021-08-26")
    d1_sanam = find_d1_row(d1, "SANAM", "2021-03-23")

    ev = gate.get("suppression_events", [])

    def first_event(symbol: str, gate_name: str) -> dict[str, Any] | None:
        for row in ev:
            if row.get("symbol") == symbol and row.get("gate") == gate_name and row.get("cohort") != "set_b":
                return row
        return None

    bpcc_risk = first_event("BPCC", "RISK_SUPPRESSION")
    sanam_avoid = first_event("SANAM", "AVOID_GATE")
    tijara_avoid = first_event("TIJARA", "AVOID_GATE")

    citations = {
        "Proposal A": [
            {
                "symbol": "ALL_EX_SET_B",
                "date": "AGGREGATE",
                "value": gate["aggregates"]["EX_SET_B"]["aggregate_by_gate"]["RISK_SUPPRESSION"]["mean_ret_20"],
                "metric": "EX_SET_B.RISK_SUPPRESSION.mean_ret_20",
                "source": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
                "level": "aggregate",
            },
            {
                "symbol": "BPCC",
                "date": "2021-08-26",
                "value": None if d1_bpcc is None else d1_bpcc.get("classification"),
                "metric": "D1-v2 day classification",
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v2.json",
                "level": "day",
            },
            {
                "symbol": "BPCC",
                "date": "2024-06-24" if bpcc_risk is None else bpcc_risk.get("trade_date_iso") or "UNKNOWN",
                "value": None if bpcc_risk is None else bpcc_risk.get("subsequent_unmasked_outcome", {}).get("ret_20"),
                "metric": "RISK_SUPPRESSION event ret_20",
                "source": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
                "level": "event",
            },
        ],
        "Proposal B": [
            {
                "symbol": "TIJARA",
                "date": "2021-09-09",
                "value": None if d1_tijara is None else d1_tijara.get("classification"),
                "metric": "D1-v2 day classification",
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v2.json",
                "level": "day",
            },
            {
                "symbol": "TIJARA",
                "date": "UNKNOWN" if tijara_avoid is None else tijara_avoid.get("trade_date_iso") or "UNKNOWN",
                "value": None if tijara_avoid is None else tijara_avoid.get("subsequent_unmasked_outcome", {}).get("ret_60"),
                "metric": "AVOID_GATE event ret_60",
                "source": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
                "level": "event",
            },
            {
                "symbol": "TIJARA",
                "date": "2021-09-09",
                "value": None
                if d1_tijara is None
                else d1_tijara.get("classification_evidence", {}).get("warmup_basis"),
                "metric": "Warmup exclusion basis",
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v2.json",
                "level": "day",
            },
        ],
        "Proposal C": [
            {
                "symbol": "SANAM",
                "date": "2021-03-23",
                "value": None if d1_sanam is None else d1_sanam.get("classification"),
                "metric": "D1-v2 day classification",
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v2.json",
                "level": "day",
            },
            {
                "symbol": "SANAM",
                "date": "UNKNOWN" if sanam_avoid is None else sanam_avoid.get("trade_date_iso") or "UNKNOWN",
                "value": None if sanam_avoid is None else sanam_avoid.get("subsequent_unmasked_outcome", {}).get("ret_20"),
                "metric": "AVOID_GATE event ret_20",
                "source": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
                "level": "event",
            },
            {
                "symbol": "SANAM",
                "date": "2021-03-23",
                "value": None
                if d1_sanam is None
                else d1_sanam.get("classification_evidence", {}).get("masked_interval_hits", []),
                "metric": "Masked/warmup evidence on target day",
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v2.json",
                "level": "day",
            },
        ],
    }

    # Hard rule: no Set B symbol citations.
    for plist in citations.values():
        for c in plist:
            if c["symbol"] in set_b:
                raise ValueError(f"Set B citation detected: {c}")

    return citations


def validate_requirements(citations: dict[str, list[dict[str, Any]]]) -> None:
    aggregate_proposals = 0
    for name, plist in citations.items():
        if len(plist) < 3:
            raise ValueError(f"{name} has fewer than 3 citations")
        if any(p.get("source") == "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v2.json" for p in plist) is False:
            raise ValueError(f"{name} missing citation from Deliverable-1-v2")
        for p in plist:
            for k in ("symbol", "date", "value", "source"):
                if k not in p:
                    raise ValueError(f"{name} citation missing required field {k}")
        if any(p.get("level") == "aggregate" for p in plist):
            aggregate_proposals += 1
    if aggregate_proposals > 1:
        raise ValueError("Aggregate-level citations present in more than one proposal")


def build_text(gate: dict[str, Any], d1: dict[str, Any], citations: dict[str, list[dict[str, Any]]]) -> str:
    proposed_counts = gate.get("aggregates", {}).get("ALL_SYMBOLS", {}).get("aggregate_by_gate", {})
    ex = gate.get("aggregates", {}).get("EX_SET_B", {}).get("aggregate_by_gate", {})
    tier_rule = gate.get("tier_rule", {})

    lines: list[str] = [
        "# R13 Three-Model Architecture Proposals v3",
        "",
        "Governance: built from sealed artifacts only; no engine contact, no reruns.",
        "Tier rule status: AGENT_PROPOSED_UNRATIFIED.",
        "",
        "Universe assumptions (dual-stated):",
        f"- Proposed threshold rule: {tier_rule.get('rule')}",
        f"- Tercile alternative: {tier_rule.get('alternative')}",
        "",
        "Failure modes addressed in all proposals:",
        "- Failure mode A (phase-transition failure): D1-v2 day-level classifications and predicate-term audits.",
        "- Failure mode B (suppression cost): EX_SET_B seam-safe suppression outcomes.",
        "",
        "## Proposal A - Sequential Gate-First Controller",
        "Design intent:",
        "- Force explicit warmup/coverage readiness before permitting BASE_FORMING and downstream transition checks.",
        "- Keep risk suppression active but require a veto ledger entry on every candidate veto decision.",
        "Quantified, falsifiable predictions:",
        "- Within next controlled replay campaign, BPCC-like windows classified as MASKED_OR_WARMUP_EXCLUDED should drop by at least 50%.",
        f"- EX_SET_B RISK_SUPPRESSION mean_ret_20 baseline={ex.get('RISK_SUPPRESSION', {}).get('mean_ret_20')}; prediction is non-negative drift and <= +0.01 absolute delta.",
        "- If warmup guard instrumentation is added and no classification shift occurs, proposal is falsified.",
        "",
        "## Proposal B - Transition Diagnostics Layer",
        "Design intent:",
        "- Persist per-day predicate-term outcomes for BASE_FORMING, ACCUMULATION, BREAKOUT_WATCH to remove non-recoverable terms.",
        "- Keep execution unchanged while adding explicit term-level explainability logs.",
        "Quantified, falsifiable predictions:",
        "- Unknown/unrecoverable term rate in D1-style audits should fall below 10% of audited terms.",
        "- TIJARA-class windows currently tagged MASKED_OR_WARMUP_EXCLUDED should either remain unchanged with explicit proof or transition to PHASE_PROGRESSED_NO_CANDIDATE with term evidence.",
        f"- AVOID_GATE event-level ret_60 citations for TIJARA remain benchmarked to current seam-safe evidence; if sign flips broadly without gate changes, proposal is falsified.",
        "",
        "## Proposal C - Capacity-Aware Candidate Router",
        "Design intent:",
        "- Separate candidate-quality scoring from capacity decisions; route vetoed candidates into deferred queue with trace IDs.",
        "- Maintain existing phase machine, but convert suppression into auditable deferred-intent decisions.",
        "Quantified, falsifiable predictions:",
        "- For SANAM/BPCC-class no-trade windows, CANDIDATE_VETOED incidence should decline by at least 30% while preserving max-position constraints.",
        f"- EX_SET_B suppression-cost baseline remains anchored to seam-safe outcomes (ALL_SYMBOLS warmup count={proposed_counts.get('WARMUP_GATE', {}).get('count')}).",
        "- If deferred queue does not reduce veto concentration in top symbols, proposal is falsified.",
        "",
        "## Citation Index",
    ]

    for name in ["Proposal A", "Proposal B", "Proposal C"]:
        lines.append(f"### {name}")
        for c in citations[name]:
            lines.append(
                "- "
                + json.dumps(
                    {
                        "symbol": c["symbol"],
                        "date": c["date"],
                        "value": c["value"],
                        "source": c["source"],
                        "metric": c.get("metric"),
                        "level": c.get("level"),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
        lines.append("")

    lines += [
        "Constraint formula: no engine, scanner, model, backtest, or market-data execution; read-only extraction and descriptive aggregation only.",
        "R14 remains NOT AUTHORIZED.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    gate = read_json(REVIEW / "r13_gate_conflict_analysis_v1_2.json")
    d1 = read_json(REVIEW / "r13_set_a_causal_attribution_v2.json")

    citations = build_citations(gate, d1)
    validate_requirements(citations)
    text = build_text(gate, d1, citations)

    out_path = REVIEW / "r13_architecture_proposals_v3.md"
    out_path.write_text(text, encoding="utf-8")

    print("R13_ARCHITECTURE_PROPOSALS_V3_COMPLETE")
    print("sha256", sha256_file(out_path))


if __name__ == "__main__":
    main()
