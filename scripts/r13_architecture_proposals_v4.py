from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
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


def ts_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def first_nonempty_day_row(d1: dict[str, Any], symbol: str) -> dict[str, Any]:
    rows = [r for r in d1.get("owner_pattern_window_audit", {}).get("day_level", []) if r.get("symbol") == symbol]
    if not rows:
        raise ValueError(f"No owner-window rows for {symbol}")
    return rows[0]


def first_gate_event(gate: dict[str, Any], symbol: str, gate_name: str) -> dict[str, Any]:
    for r in gate.get("suppression_events", []):
        if r.get("symbol") == symbol and r.get("gate") == gate_name and r.get("cohort") != "set_b":
            return r
    raise ValueError(f"No suppression event for {symbol} gate={gate_name}")


def warmup_cost_symbol(d1: dict[str, Any], symbol: str) -> dict[str, Any]:
    for r in d1.get("warmup_cost_quantification", {}).get("per_symbol", []):
        if r.get("symbol") == symbol:
            return r
    raise ValueError(f"No warmup cost row for {symbol}")


def build_citations(d1: dict[str, Any], gate: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    bpcc_day = first_nonempty_day_row(d1, "BPCC")
    tijara_day = first_nonempty_day_row(d1, "TIJARA")
    sanam_day = first_nonempty_day_row(d1, "SANAM")

    bpcc_evt = first_gate_event(gate, "BPCC", "RISK_SUPPRESSION")
    tijara_evt = first_gate_event(gate, "TIJARA", "AVOID_GATE")
    sanam_evt = first_gate_event(gate, "SANAM", "AVOID_GATE")

    bpcc_warm = warmup_cost_symbol(d1, "BPCC")
    tijara_warm = warmup_cost_symbol(d1, "TIJARA")
    sanam_warm = warmup_cost_symbol(d1, "SANAM")

    citations = {
        "Proposal A": [
            {
                "symbol": "BPCC",
                "date": str(bpcc_day.get("trade_date_iso")),
                "value": str(bpcc_day.get("blocking_term", {}).get("term")),
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
                "metric": "owner-window blocking term",
                "level": "day",
            },
            {
                "symbol": "BPCC",
                "date": ts_to_iso(int(bpcc_evt.get("trade_date"))),
                "value": bpcc_evt.get("subsequent_unmasked_outcome", {}).get("ret_20"),
                "source": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
                "metric": "RISK_SUPPRESSION event ret_20",
                "level": "event",
            },
            {
                "symbol": "ALL_EX_SET_B",
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "value": gate.get("aggregates", {}).get("EX_SET_B", {}).get("aggregate_by_gate", {}).get("RISK_SUPPRESSION", {}).get("mean_ret_20"),
                "source": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
                "metric": "EX_SET_B.RISK_SUPPRESSION.mean_ret_20",
                "level": "aggregate",
            },
        ],
        "Proposal B": [
            {
                "symbol": "TIJARA",
                "date": str(tijara_day.get("trade_date_iso")),
                "value": str(tijara_day.get("blocking_term", {}).get("term")),
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
                "metric": "owner-window blocking term",
                "level": "day",
            },
            {
                "symbol": "TIJARA",
                "date": ts_to_iso(int(tijara_evt.get("trade_date"))),
                "value": tijara_evt.get("subsequent_unmasked_outcome", {}).get("ret_60"),
                "source": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
                "metric": "AVOID_GATE event ret_60",
                "level": "event",
            },
            {
                "symbol": "TIJARA",
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "value": tijara_warm.get("blind_share_unmasked"),
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
                "metric": "warmup blindness share",
                "level": "symbol_metric",
            },
        ],
        "Proposal C": [
            {
                "symbol": "SANAM",
                "date": str(sanam_day.get("trade_date_iso")),
                "value": str(sanam_day.get("blocking_term", {}).get("term")),
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
                "metric": "owner-window blocking term",
                "level": "day",
            },
            {
                "symbol": "SANAM",
                "date": ts_to_iso(int(sanam_evt.get("trade_date"))),
                "value": sanam_evt.get("subsequent_unmasked_outcome", {}).get("ret_20"),
                "source": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
                "metric": "AVOID_GATE event ret_20",
                "level": "event",
            },
            {
                "symbol": "SANAM",
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "value": sanam_warm.get("blind_sessions_from_segment_warmup"),
                "source": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
                "metric": "warmup blindness sessions",
                "level": "symbol_metric",
            },
        ],
    }
    return citations


def validate_gate(citations: dict[str, list[dict[str, Any]]], set_b: set[str]) -> dict[str, Any]:
    checks = []
    aggregate_count = 0

    for proposal, rows in citations.items():
        checks.append({"check": f"{proposal}_min_3_citations", "pass": len(rows) >= 3, "value": len(rows)})
        has_d1 = any(r.get("source") == "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json" for r in rows)
        checks.append({"check": f"{proposal}_has_d1_v3_citation", "pass": has_d1})

        for idx, r in enumerate(rows):
            missing = [k for k in ["symbol", "date", "value", "source"] if k not in r]
            checks.append({"check": f"{proposal}_citation_{idx}_required_fields", "pass": len(missing) == 0, "missing": missing})

            val = r.get("value")
            bad_date = r.get("date") in {None, "", "UNKNOWN"}
            bad_value = val is None or val == "" or (isinstance(val, list) and len(val) == 0)
            checks.append({"check": f"{proposal}_citation_{idx}_date_resolvable", "pass": not bad_date, "value": r.get("date")})
            checks.append({"check": f"{proposal}_citation_{idx}_value_nonempty", "pass": not bad_value, "value": val})

            sym = str(r.get("symbol"))
            checks.append({"check": f"{proposal}_citation_{idx}_no_set_b", "pass": sym not in set_b, "symbol": sym})
            if str(r.get("level")) == "aggregate":
                aggregate_count += 1

    checks.append({"check": "aggregate_level_citations_at_most_one_proposal", "pass": aggregate_count <= 1, "value": aggregate_count})

    passed = all(bool(c.get("pass")) for c in checks)
    return {"status": "PASS" if passed else "FAIL", "checks": checks}


def build_text(d1: dict[str, Any], gate: dict[str, Any], citations: dict[str, list[dict[str, Any]]], gatecheck: dict[str, Any]) -> str:
    tier_rule = gate.get("tier_rule", {})
    ex = gate.get("aggregates", {}).get("EX_SET_B", {}).get("aggregate_by_gate", {})
    warm_tier = d1.get("warmup_cost_quantification", {}).get("per_liquidity_tier", {})

    lines = [
        "# R13 Three-Model Architecture Proposals v4",
        "",
        "Universe assumptions (dual-stated):",
        f"- Proposed threshold rule: {tier_rule.get('rule')}",
        f"- Tercile alternative: {tier_rule.get('alternative')}",
        "",
        "Sharpened failure modes:",
        "- (i) Phase-transition blocking in 2025-2026 owner-pattern windows from D1-v3 owner-window audit.",
        "- (ii) Seam-safe EX_SET_B suppression cost (event + aggregate where allowed).",
        "- (iii) Warmup blindness cost from D1-v3 warmup quantification.",
        "",
        "## Proposal A - Warmup-Readiness Promotion Gate",
        "Design:",
        "- Promote and persist warmup readiness state explicitly before base-entry eligibility.",
        "- Keep suppression policy unchanged but require explicit veto trace rows for all candidate vetoes.",
        "Falsifiable predictions:",
        "- BPCC owner-window blocking term concentration should shift away from neutral-hold terms by >=40% in next controlled replay campaign.",
        f"- EX_SET_B RISK_SUPPRESSION mean_ret_20 baseline={ex.get('RISK_SUPPRESSION', {}).get('mean_ret_20')}; expected delta in [-0.01, +0.01].",
        "- If blocking-term concentration does not move while readiness state is persisted, proposal fails.",
        "",
        "## Proposal B - Predicate-Term Telemetry Layer",
        "Design:",
        "- Persist per-day predicate term outcomes for base, accumulation, and watch transitions.",
        "- Require term-level provenance in review artifacts for any phase hold/revert under pattern progress.",
        "Falsifiable predictions:",
        "- Unrecoverable-term share in owner-window audits should be <=10%.",
        "- TIJARA-like holds should map to explicit failed terms rather than unresolved state references within one release cycle.",
        "- If unresolved term share remains >10%, proposal fails.",
        "",
        "## Proposal C - Capacity Router With Deferred Candidate Queue",
        "Design:",
        "- Separate candidate quality from capacity veto outcome with deferred queue and replayable intent records.",
        "- Preserve risk caps but reduce repeated veto concentration in single symbols.",
        "Falsifiable predictions:",
        "- SANAM/BPCC owner-window candidate-veto-like blockage frequency should reduce by >=30% without increasing max-position breaches.",
        f"- Warmup blindness tier summary baseline={json.dumps(warm_tier, ensure_ascii=True, sort_keys=True)}",
        "- If veto concentration does not reduce after deferred routing, proposal fails.",
        "",
        "## Citation Gate Output",
        "```json",
        json.dumps(gatecheck, ensure_ascii=True, indent=2, sort_keys=True),
        "```",
        "",
        "## Citation Index",
    ]

    for p in ["Proposal A", "Proposal B", "Proposal C"]:
        lines.append(f"### {p}")
        for c in citations[p]:
            lines.append("- " + json.dumps(c, ensure_ascii=True, sort_keys=True))
        lines.append("")

    lines += [
        "Constraint formula: no engine, scanner, model, backtest, or market-data execution; read-only extraction and descriptive aggregation only.",
        "R14 remains NOT AUTHORIZED.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    d1 = read_json(REVIEW / "r13_set_a_causal_attribution_v3.json")
    gate = read_json(REVIEW / "r13_gate_conflict_analysis_v1_2.json")

    set_b = set(gate.get("set_membership", {}).get("set_b", []))
    citations = build_citations(d1, gate)
    gatecheck = validate_gate(citations, set_b)

    if gatecheck.get("status") != "PASS":
        out_fail = REVIEW / "r13_architecture_proposals_v4_gatecheck_fail.json"
        out_fail.write_text(json.dumps(gatecheck, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("R13_ARCHITECTURE_PROPOSALS_V4_GATECHECK_FAIL")
        raise SystemExit(1)

    text = build_text(d1, gate, citations, gatecheck)
    out_md = REVIEW / "r13_architecture_proposals_v4.md"
    out_md.write_text(text, encoding="utf-8")

    print("R13_ARCHITECTURE_PROPOSALS_V4_COMPLETE")
    print("sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()
