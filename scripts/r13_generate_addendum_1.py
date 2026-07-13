from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "artifacts" / "preview1a_prestart" / "review_final"


def main() -> None:
    conf_path = REVIEW_DIR / "r13_gate_conflict_analysis_v1.json"
    arch_path = REVIEW_DIR / "r13_architecture_proposals_v1.md"
    out_json = REVIEW_DIR / "r13_gate_conflict_analysis_v1_addendum_1.json"
    out_md = REVIEW_DIR / "r13_gate_conflict_analysis_v1_addendum_1.md"

    conf = json.loads(conf_path.read_text(encoding="utf-8"))
    arch = arch_path.read_text(encoding="utf-8")

    runtime = Path(conf["runtime_db"])
    con = sqlite3.connect(runtime)
    cur = con.cursor()
    seg_rows = cur.execute(
        "SELECT original_symbol, segment_symbol, segment_id, bars_count, start_trade_date, end_trade_date FROM ee_symbol_segment_map"
    ).fetchall()
    con.close()

    seg_map = {}
    for osym, ssym, sid, bars, sdt, edt in seg_rows:
        seg_map[str(ssym)] = {
            "original_symbol": str(osym),
            "segment_id": int(sid),
            "bars_count": int(bars),
            "start_trade_date": int(sdt) if sdt is not None else None,
            "end_trade_date": int(edt) if edt is not None else None,
        }

    warm = [r for r in conf.get("suppression_events", []) if r.get("gate") == "WARMUP_GATE"]

    split_counts = Counter()
    per_tier = defaultdict(Counter)
    rows = []
    unknown = []
    for r in warm:
        ss = str(r.get("symbol_segment") or "")
        tier = str(r.get("liquidity_tier") or "UNKNOWN")
        td = int(r.get("trade_date") or 0)
        seg = seg_map.get(ss)
        if seg is None:
            category = "UNRESOLVED"
            unknown.append(ss)
            seg_id = None
            at_segment_start = None
        else:
            seg_id = int(seg["segment_id"])
            sdt = seg["start_trade_date"]
            at_segment_start = sdt is not None and td == int(sdt)
            category = "NATURAL" if seg_id == 1 else "MASK_INDUCED"
        split_counts[category] += 1
        per_tier[tier][category] += 1
        rows.append(
            {
                "signal_id": r.get("signal_id"),
                "symbol": r.get("symbol"),
                "symbol_segment": ss,
                "trade_date": r.get("trade_date"),
                "liquidity_tier": tier,
                "category": category,
                "segment_id": seg_id,
                "at_segment_start": at_segment_start,
            }
        )

    proposal_warmup_mentions = []
    for proposal in ["Proposal A", "Proposal B", "Proposal C"]:
        start = arch.find("## " + proposal)
        if start == -1:
            proposal_warmup_mentions.append({"proposal": proposal, "mentions_warmup": False, "mention_lines": []})
            continue
        nxt = arch.find("\n## ", start + 1)
        section = arch[start:nxt] if nxt != -1 else arch[start:]
        lines = [ln for ln in section.splitlines() if "warmup" in ln.lower()]
        proposal_warmup_mentions.append({"proposal": proposal, "mentions_warmup": len(lines) > 0, "mention_lines": lines})

    payload = {
        "version_id": "R13_GATE_CONFLICT_ANALYSIS_V1_ADDENDUM_1",
        "scope": "WARMUP_GATE decomposition only; append-only addendum",
        "source_artifact": "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1.json",
        "runtime_db": str(runtime),
        "classification_rule": {
            "MASK_INDUCED": "WARMUP_GATE event where symbol_segment maps to ee_symbol_segment_map.segment_id > 1 (segment follows prior masked boundary).",
            "NATURAL": "WARMUP_GATE event where symbol_segment maps to ee_symbol_segment_map.segment_id == 1 (start of symbol history).",
            "UNRESOLVED": "symbol_segment not found in ee_symbol_segment_map.",
        },
        "warmup_gate_total": len(warm),
        "split_counts": dict(split_counts),
        "split_by_liquidity_tier": {tier: dict(cnt) for tier, cnt in sorted(per_tier.items())},
        "segment_start_alignment": {
            "events_at_exact_segment_start": sum(1 for x in rows if x.get("at_segment_start") is True),
            "events_not_at_exact_segment_start": sum(1 for x in rows if x.get("at_segment_start") is False),
            "events_unknown_alignment": sum(1 for x in rows if x.get("at_segment_start") is None),
        },
        "proposal_text_warmup_mentions": proposal_warmup_mentions,
        "proposal_dependency_statement": "Proposal text references warmup only in Proposal B (veto codebook includes warmup invalid). No proposal text cites MASK_INDUCED vs NATURAL category burden explicitly.",
        "unresolved_symbol_segments": sorted(set(unknown)),
        "rows": rows,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    md: list[str] = []
    md.append("# R13 Gate Conflict Analysis v1 Addendum 1")
    md.append("")
    md.append("Scope: WARMUP_GATE decomposition only. No reruns, no model changes.")
    md.append("")
    md.append(f"WARMUP_GATE total: {len(warm)}")
    md.append("")
    md.append("## Split Counts")
    for k in ["MASK_INDUCED", "NATURAL", "UNRESOLVED"]:
        if k in split_counts:
            md.append(f"- {k}: {split_counts[k]}")
    md.append("")
    md.append("## Split by Liquidity Tier")
    for tier in sorted(per_tier):
        c = per_tier[tier]
        md.append(
            f"- {tier}: MASK_INDUCED={c.get('MASK_INDUCED', 0)}, NATURAL={c.get('NATURAL', 0)}, UNRESOLVED={c.get('UNRESOLVED', 0)}"
        )
    md.append("")
    md.append("## Classification Rule")
    md.append("- MASK_INDUCED: segment_id > 1 in ee_symbol_segment_map for symbol_segment.")
    md.append("- NATURAL: segment_id == 1 in ee_symbol_segment_map for symbol_segment.")
    md.append("- UNRESOLVED: symbol_segment not found in ee_symbol_segment_map.")
    md.append("")
    md.append("## Proposal Warmup Citation Check")
    for row in proposal_warmup_mentions:
        md.append(f"- {row['proposal']}: mentions_warmup={str(row['mentions_warmup']).lower()}")
        if row["mention_lines"]:
            for ln in row["mention_lines"]:
                md.append(f"  - {ln}")
    md.append("")
    md.append("Statement: Proposal text does not cite category-specific warmup burden (MASK_INDUCED vs NATURAL).")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("R13_ADDENDUM_1_WRITTEN")


if __name__ == "__main__":
    main()
