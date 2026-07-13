from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
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
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def base_symbol(symbol: str) -> str:
    s = str(symbol)
    return s.split("__SEG", 1)[0] if "__SEG" in s else s


def sqlite_ro_connect(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def mean(vals: list[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def liquidity_tier(median_value_kwd: float) -> str:
    if median_value_kwd >= 500000.0:
        return "HIGH"
    if median_value_kwd >= 100000.0:
        return "MID"
    return "LOW"


def set_symbols(v21: dict[str, Any]) -> tuple[set[str], set[str]]:
    set_a: set[str] = set()
    set_b: set[str] = set()
    rows = v21.get("benchmark_parity_suite_completion", {}).get("random_top_k_portfolio_rows", [])
    for row in rows:
        syms = row.get("symbols", [])
        if row.get("set") == "set_a":
            set_a.update(syms)
        elif row.get("set") == "set_b":
            set_b.update(syms)
    return set_a, set_b


def state_taxonomy_v1_2(v1: dict[str, Any]) -> dict[str, Any]:
    out = dict(v1)
    out["version_id"] = "R13_STATE_TAXONOMY_V1_2"
    out.pop("generated_at_utc", None)
    out["deterministic_build"] = True
    out["tier_rule_status"] = "AGENT_PROPOSED_UNRATIFIED"
    return out


def universe_profile_v1_2(runtime_db: Path, mask_manifest: dict[str, Any], triage: dict[str, Any]) -> dict[str, Any]:
    con = sqlite_ro_connect(runtime_db)
    cur = con.cursor()

    src = cur.execute(
        "SELECT symbol, volume, COALESCE(value_kwd,0.0), COALESCE(is_masked,0) FROM ee_ohlcv_masked_source ORDER BY symbol, trade_date"
    ).fetchall()

    by_symbol: dict[str, dict[str, Any]] = defaultdict(lambda: {"vals": [], "vols": [], "n": 0, "masked": 0})
    for sym, vol, val, is_masked in src:
        r = by_symbol[str(sym)]
        r["vals"].append(float(val or 0.0))
        r["vols"].append(float(vol or 0.0))
        r["n"] += 1
        r["masked"] += int(is_masked or 0)

    mask_counts = Counter()
    for row in mask_manifest.get("intervals", []):
        sym = row.get("symbol")
        if sym:
            mask_counts[str(sym)] += 1

    breach_counts = Counter()
    final = triage.get("final", {})
    for entry in final.get("ca_ledger_v0_1", {}).get("entries", []):
        sym = entry.get("symbol")
        if sym:
            breach_counts[str(sym)] += int(entry.get("event_count") or 0)

    rows = []
    for sym in sorted(by_symbol.keys()):
        r = by_symbol[sym]
        n = int(r["n"])
        med = float(median(r["vals"])) if r["vals"] else 0.0
        zero_share = (sum(1 for v in r["vols"] if v <= 0.0) / n) if n else 0.0
        mask_burden = (int(r["masked"]) / n) if n else 0.0
        breaches = int(breach_counts.get(sym, 0))
        rows.append(
            {
                "symbol": sym,
                "liquidity_tier": liquidity_tier(med),
                "median_daily_value_traded_kwd": med,
                "zero_volume_session_share": zero_share,
                "masked_interval_count": int(mask_counts.get(sym, 0)),
                "masked_interval_burden": mask_burden,
                "breach_event_count": breaches,
                "breach_event_density": (breaches / n) if n else 0.0,
                "total_sessions": n,
            }
        )

    tiers = Counter(r["liquidity_tier"] for r in rows)
    con.close()

    return {
        "version_id": "R13_UNIVERSE_TIER_PROFILE_V1_2",
        "deterministic_build": True,
        "runtime_db": "artifacts/preview1a_prestart/review_final/r12_exam_surface_v4_5_runtime.db",
        "tier_rule": {
            "rule": "HIGH if median_daily_value_traded_kwd >= 500000; MID if >= 100000; else LOW",
            "status": "AGENT_PROPOSED_UNRATIFIED",
            "alternative": "terciles over median_daily_value_traded_kwd across full universe",
        },
        "rows": rows,
        "tier_summary": {
            "counts": dict(sorted(tiers.items())),
            "symbol_count": len(rows),
        },
    }


def seam_safe_metrics(seg_rows: list[tuple[int, float]], trade_date: int, entry_close: float) -> tuple[dict[str, float | None], dict[str, int]]:
    future = [(d, c) for (d, c) in seg_rows if d > trade_date]
    out: dict[str, float | None] = {}
    trunc: dict[str, int] = {"ret_5": 0, "ret_20": 0, "ret_60": 0}
    for h in (5, 20, 60):
        key = f"ret_{h}"
        if entry_close <= 0 or not future:
            out[key] = None
            continue
        idx = min(h, len(future)) - 1
        out[key] = (future[idx][1] / entry_close) - 1.0
        if len(future) < h:
            trunc[key] = 1
    if future and entry_close > 0:
        window = future[:60]
        closes = [c for _, c in window]
        out["max_upside_60"] = (max(closes) / entry_close) - 1.0
        out["max_drawdown_60"] = (min(closes) / entry_close) - 1.0
    else:
        out["max_upside_60"] = None
        out["max_drawdown_60"] = None
    out["horizon_basis"] = "seam_safe_same_segment"
    return out, trunc


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_gate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_gate[str(r["gate"])].append(r)
    out: dict[str, Any] = {}
    for gate in sorted(by_gate.keys()):
        grows = by_gate[gate]
        vals5 = [x["subsequent_unmasked_outcome"]["ret_5"] for x in grows if x["subsequent_unmasked_outcome"]["ret_5"] is not None]
        vals20 = [x["subsequent_unmasked_outcome"]["ret_20"] for x in grows if x["subsequent_unmasked_outcome"]["ret_20"] is not None]
        vals60 = [x["subsequent_unmasked_outcome"]["ret_60"] for x in grows if x["subsequent_unmasked_outcome"]["ret_60"] is not None]
        out[gate] = {
            "count": len(grows),
            "mean_ret_5": mean(vals5),
            "mean_ret_20": mean(vals20),
            "mean_ret_60": mean(vals60),
            "truncations": {
                "ret_5": sum(int(x.get("truncation", {}).get("ret_5") or 0) for x in grows),
                "ret_20": sum(int(x.get("truncation", {}).get("ret_20") or 0) for x in grows),
                "ret_60": sum(int(x.get("truncation", {}).get("ret_60") or 0) for x in grows),
            },
        }
    return out


def gate_conflict_v1_2(runtime_db: Path, v21: dict[str, Any], universe_profile: dict[str, Any]) -> dict[str, Any]:
    set_a, set_b = set_symbols(v21)
    set_a_detail = v21.get("set_a_no_trade_forensics", [])
    tier_by_symbol = {r["symbol"]: r["liquidity_tier"] for r in universe_profile.get("rows", [])}

    con = sqlite_ro_connect(runtime_db)
    cur = con.cursor()

    seg_prices: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for sym, td, close in cur.execute("SELECT symbol, trade_date, close FROM ee_ohlcv ORDER BY symbol, trade_date").fetchall():
        seg_prices[str(sym)].append((int(td), float(close or 0.0)))

    rows_db = cur.execute(
        "SELECT id, symbol, trade_date, signal_type, phase_from, phase_to, score, price, evidence_json FROM ee_signals ORDER BY id"
    ).fetchall()
    con.close()

    events = []
    gate_counts = Counter()
    gate_tier_counts = Counter()
    for sid, seg_sym, td, stype, pfrom, pto, score, price, evidence_raw in rows_db:
        evidence = json.loads(str(evidence_raw or "{}")) if evidence_raw else {}
        gate = None
        if stype == "SIGNAL_SUPPRESSED_RISK":
            gate = "RISK_SUPPRESSION"
        elif stype == "AVOID_SET" or pto == "AVOID":
            gate = "AVOID_GATE"
        elif stype == "PHASE_ONLY" and evidence.get("reason") == "warmup_pending":
            gate = "WARMUP_GATE"
        if gate is None:
            continue

        seg = str(seg_sym)
        base = base_symbol(seg)
        metrics, trunc = seam_safe_metrics(seg_prices.get(seg, []), int(td), float(price or evidence.get("close") or 0.0))
        cohort = "set_a" if base in set_a else ("set_b" if base in set_b else "universe")
        tier = tier_by_symbol.get(base, "UNKNOWN")

        ev = {
            "signal_id": int(sid),
            "symbol": base,
            "symbol_segment": seg,
            "trade_date": int(td),
            "cohort": cohort,
            "gate": gate,
            "signal_type_recorded": str(stype),
            "attempted_signal_type": evidence.get("attempted_signal_type"),
            "suppressed_reason": evidence.get("suppressed_reason"),
            "phase_from": pfrom,
            "phase_to": pto,
            "score": float(score or 0.0),
            "liquidity_tier": tier,
            "evidence_values": {
                "close": evidence.get("close"),
                "rel_volume": evidence.get("rel_volume"),
                "rsi_14": evidence.get("rsi_14"),
                "adx_19": evidence.get("adx_19"),
                "sma200": evidence.get("sma200"),
                "ema10": evidence.get("ema10"),
                "ema30": evidence.get("ema30"),
                "suppressed_reason": evidence.get("suppressed_reason"),
                "attempted_signal_type": evidence.get("attempted_signal_type"),
                "confirming_c_score": evidence.get("confirming_c_score"),
            },
            "subsequent_unmasked_outcome": metrics,
            "truncation": trunc,
        }
        events.append(ev)
        gate_counts[gate] += 1
        gate_tier_counts[(gate, tier)] += 1

    all_agg = aggregate(events)
    ex_set_b = [e for e in events if e["cohort"] != "set_b"]
    ex_set_b_agg = aggregate(ex_set_b)

    by_gate_tier_all = []
    by_gate_tier_ex = []
    for (gate, tier), c in sorted(gate_tier_counts.items()):
        by_gate_tier_all.append({"scope": "ALL_SYMBOLS", "gate": gate, "liquidity_tier": tier, "count": int(c)})
    ex_counts = Counter((e["gate"], e["liquidity_tier"]) for e in ex_set_b)
    for (gate, tier), c in sorted(ex_counts.items()):
        by_gate_tier_ex.append({"scope": "EX_SET_B", "gate": gate, "liquidity_tier": tier, "count": int(c)})

    out = {
        "version_id": "R13_GATE_CONFLICT_ANALYSIS_V1_2",
        "deterministic_build": True,
        "runtime_db": "artifacts/preview1a_prestart/review_final/r12_exam_surface_v4_5_runtime.db",
        "constraints": {
            "set_b_rows_carried_only": True,
            "citable_evidence_scope": "EX_SET_B",
            "seam_safe_rule": "same-segment forward returns; horizon truncated at segment end",
        },
        "set_membership": {"set_a": sorted(set_a), "set_b": sorted(set_b)},
        "suppression_events": events,
        "aggregate_counts_by_gate": dict(sorted(gate_counts.items())),
        "aggregates": {
            "ALL_SYMBOLS": {
                "aggregate_by_gate": all_agg,
                "aggregate_by_gate_and_liquidity_tier": by_gate_tier_all,
            },
            "EX_SET_B": {
                "aggregate_by_gate": ex_set_b_agg,
                "aggregate_by_gate_and_liquidity_tier": by_gate_tier_ex,
            },
        },
        "set_a_benchmark_day_detail": set_a_detail,
        "tier_rule": {
            "rule": "HIGH if median_daily_value_traded_kwd >= 500000; MID if >= 100000; else LOW",
            "status": "AGENT_PROPOSED_UNRATIFIED",
            "alternative": "terciles over median_daily_value_traded_kwd across full universe",
        },
    }
    return out


def md_state(payload: dict[str, Any]) -> str:
    lines = ["# R13 State-Taxonomy Mapping v1.2", "", "Deterministic build; content-stable serialization.", ""]
    lines.append("## Mapping Table")
    for r in payload.get("state_mapping_table", []):
        lines.append(f"- {r['engine_state']}: {r['definition_as_coded']}")
    lines.append("")
    lines.append("## Gap List")
    for r in payload.get("gap_list_market_states_without_engine_representation", []):
        lines.append(f"- {r['taxonomy_state']}: {r['evidence']}")
    return "\n".join(lines) + "\n"


def md_gate(payload: dict[str, Any]) -> str:
    lines = ["# R13 Gate-Conflict Analysis v1.2", "", "Seam-safe outcomes with dual scopes: ALL_SYMBOLS and EX_SET_B.", ""]
    for scope in ["ALL_SYMBOLS", "EX_SET_B"]:
        lines.append(f"## {scope}")
        ag = payload.get("aggregates", {}).get(scope, {}).get("aggregate_by_gate", {})
        for gate in sorted(ag.keys()):
            row = ag[gate]
            lines.append(
                f"- {gate}: count={row['count']} mean_ret_5={row['mean_ret_5']} mean_ret_20={row['mean_ret_20']} mean_ret_60={row['mean_ret_60']} trunc={row['truncations']}"
            )
        lines.append("")
    lines.append("Citable scope: EX_SET_B only.")
    return "\n".join(lines) + "\n"


def md_arch_v2(gate_v12: dict[str, Any], tier_counts: dict[str, int], tercile_claims: dict[str, Any]) -> str:
    ex = gate_v12.get("aggregates", {}).get("EX_SET_B", {}).get("aggregate_by_gate", {})
    return "\n".join([
        "# R13 Three-Model Architecture Proposals v2",
        "",
        "Evidence basis: EX_SET_B seam-safe aggregates from r13_gate_conflict_analysis_v1_2.",
        "Tier rule status: AGENT_PROPOSED_UNRATIFIED.",
        f"Tier counts under proposed rule: {json.dumps(tier_counts, ensure_ascii=True, sort_keys=True)}",
        f"Tier counts under terciles: {json.dumps(tercile_claims.get('tercile_counts', {}), ensure_ascii=True, sort_keys=True)}",
        "",
        "## Proposal A - Sequential Tri-Model With Hard Data-Surface Gate",
        "Universe assumption:",
        "- Full universe with tier-aware sizing only (no exclusions).",
        "Testable predictions:",
        f"- EX_SET_B AVOID_GATE mean_ret_60 remains positive/flat after seam-safe correction baseline: {ex.get('AVOID_GATE',{}).get('mean_ret_60')}",
        "- Set A AVOID_GATE share declines in future exam under explicit veto audit.",
        "Tier-dependent claims shown under both proposed thresholds and tercile alternative.",
        "",
        "## Proposal B - Parallel Specialists + Evidence Council",
        "Universe assumption:",
        "- Tiered treatment: HIGH/MID full participation; LOW constrained execution but retained analytics.",
        "Testable predictions:",
        f"- EX_SET_B RISK_SUPPRESSION mean_ret_20 baseline: {ex.get('RISK_SUPPRESSION',{}).get('mean_ret_20')}",
        "- Lower null-attempt suppression frequency on Set A benchmark-active windows.",
        "Tier-dependent claims shown under both proposed thresholds and tercile alternative.",
        "",
        "## Proposal C - Regime-First Controller",
        "Universe assumption:",
        "- Analytics full universe; execution may use explicit exclusion tier pending ratification.",
        "Testable predictions:",
        f"- EX_SET_B WARMUP_GATE mean_ret_60 baseline: {ex.get('WARMUP_GATE',{}).get('mean_ret_60')}",
        "- Improved attribution split regime-closed vs risk-closed for Set A no-trade cases.",
        "Tier-dependent claims shown under both proposed thresholds and tercile alternative.",
        "",
        "Constraint formula: no engine, scanner, model, backtest, or market-data execution; read-only extraction and descriptive aggregation only.",
    ]) + "\n"


def set_a_causal_v1(v21: dict[str, Any], gate_v12: dict[str, Any], tercile_map: dict[str, str]) -> dict[str, Any]:
    set_a = set(gate_v12.get("set_membership", {}).get("set_a", []))
    events = [e for e in gate_v12.get("suppression_events", []) if e.get("cohort") == "set_a"]
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_symbol[str(e["symbol"])].append(e)

    rows = []
    for sym in sorted(set_a):
        evs = by_symbol.get(sym, [])
        gates = Counter(e["gate"] for e in evs)
        attempted = Counter(str(e.get("attempted_signal_type")) for e in evs if e.get("attempted_signal_type") is not None)
        suppress = Counter(str(e.get("suppressed_reason")) for e in evs if e.get("suppressed_reason") is not None)
        rows.append({
            "symbol": sym,
            "event_count": len(evs),
            "gate_counts": dict(sorted(gates.items())),
            "attempted_signal_counts": dict(sorted(attempted.items())),
            "suppressed_reason_counts": dict(sorted(suppress.items())),
            "primary_gate": gates.most_common(1)[0][0] if gates else None,
            "tercile_tier": tercile_map.get(sym),
        })

    return {
        "version_id": "R13_SET_A_CAUSAL_ATTRIBUTION_V1",
        "evidence_scope": "EX_SET_B seam-safe aggregates + Set A event rows",
        "set_a_rows": rows,
        "set_a_no_trade_forensics_reference": v21.get("set_a_no_trade_forensics", []),
    }


def md_set_a(payload: dict[str, Any]) -> str:
    lines = ["# R13 Set A Causal Attribution v1", "", f"Evidence scope: {payload.get('evidence_scope')}", ""]
    for r in payload.get("set_a_rows", []):
        lines.append(f"- {r['symbol']}: primary_gate={r['primary_gate']} event_count={r['event_count']} gate_counts={r['gate_counts']} attempted={r['attempted_signal_counts']} suppressed={r['suppressed_reason_counts']} tercile_tier={r.get('tercile_tier')}")
    return "\n".join(lines) + "\n"


def tercile_tiers(universe_rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, int]]:
    sorted_rows = sorted(universe_rows, key=lambda r: (float(r["median_daily_value_traded_kwd"]), str(r["symbol"])))
    n = len(sorted_rows)
    c1 = n // 3
    c2 = (2 * n) // 3
    mapping = {}
    counts = Counter()
    for i, r in enumerate(sorted_rows):
        if i < c1:
            t = "LOW_TERCILE"
        elif i < c2:
            t = "MID_TERCILE"
        else:
            t = "HIGH_TERCILE"
        mapping[str(r["symbol"])] = t
        counts[t] += 1
    return mapping, dict(sorted(counts.items()))


def main() -> None:
    v1_state = read_json(REVIEW_DIR / "r13_state_taxonomy_v1.json")
    v21 = read_json(REVIEW_DIR / "r12_exam_results_v2_1.json")
    triage = read_json(REVIEW_DIR / "r12_breach_triage_v4_2_FINAL.json")
    mask_manifest = read_json(REVIEW_DIR / "r12_masked_intervals_manifest_v4_3_final.json")
    runtime_db = REVIEW_DIR / "r12_exam_surface_v4_5_runtime.db"

    state_v12 = state_taxonomy_v1_2(v1_state)
    uni_v12 = universe_profile_v1_2(runtime_db, mask_manifest, triage)
    gate_v12 = gate_conflict_v1_2(runtime_db, v21, uni_v12)

    tercile_map, tercile_counts = tercile_tiers(uni_v12.get("rows", []))
    set_a_payload = set_a_causal_v1(v21, gate_v12, tercile_map)
    arch_v2_md = md_arch_v2(gate_v12, uni_v12.get("tier_summary", {}).get("counts", {}), {"tercile_counts": tercile_counts})

    p_state_json = REVIEW_DIR / "r13_state_taxonomy_v1_2.json"
    p_state_md = REVIEW_DIR / "r13_state_taxonomy_v1_2.md"
    p_gate_json = REVIEW_DIR / "r13_gate_conflict_analysis_v1_2.json"
    p_gate_md = REVIEW_DIR / "r13_gate_conflict_analysis_v1_2.md"
    p_uni_json = REVIEW_DIR / "r13_universe_tier_profile_v1_2.json"
    p_seta_json = REVIEW_DIR / "r13_set_a_causal_attribution_v1.json"
    p_seta_md = REVIEW_DIR / "r13_set_a_causal_attribution_v1.md"
    p_arch_v2 = REVIEW_DIR / "r13_architecture_proposals_v2.md"

    write_json(p_state_json, state_v12)
    p_state_md.write_text(md_state(state_v12), encoding="utf-8")
    write_json(p_gate_json, gate_v12)
    p_gate_md.write_text(md_gate(gate_v12), encoding="utf-8")
    write_json(p_uni_json, uni_v12)
    write_json(p_seta_json, set_a_payload)
    p_seta_md.write_text(md_set_a(set_a_payload), encoding="utf-8")
    p_arch_v2.write_text(arch_v2_md, encoding="utf-8")

    created = [
        "scripts/r13_generate_paper_outputs_v1_2.py",
        "artifacts/preview1a_prestart/review_final/r13_state_taxonomy_v1_2.json",
        "artifacts/preview1a_prestart/review_final/r13_state_taxonomy_v1_2.md",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.json",
        "artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1_2.md",
        "artifacts/preview1a_prestart/review_final/r13_universe_tier_profile_v1_2.json",
        "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v1.json",
        "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v1.md",
        "artifacts/preview1a_prestart/review_final/r13_architecture_proposals_v2.md",
    ]

    rows = []
    for rel in created:
        ap = ROOT / rel
        rows.append({"path": rel, "sha256": sha256_file(ap), "size_bytes": ap.stat().st_size})

    manifest = {
        "version_id": "R13_CREATED_FILES_MANIFEST_V1_2",
        "scope": "R13 deterministic v1_2 regeneration and post-step0 outputs",
        "supersession": {
            "supersedes_manifest": "artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.json",
            "v1_status": "SUPERSEDED",
            "v1_on_disk": True,
        },
        "created_files": rows,
    }
    p_manifest = REVIEW_DIR / "r13_created_files_manifest_v1_2.json"
    p_manifest_sha = REVIEW_DIR / "r13_created_files_manifest_v1_2.sha256"
    write_json(p_manifest, manifest)
    p_manifest_sha.write_text(
        f"{sha256_file(p_manifest)}  artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1_2.json\n",
        encoding="utf-8",
    )

    print("R13_V1_2_GENERATION_COMPLETE")


if __name__ == "__main__":
    main()
