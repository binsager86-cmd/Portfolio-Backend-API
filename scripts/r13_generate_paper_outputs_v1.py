from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
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


def ts_to_iso(ts: int | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()


def safe_json_load(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(str(raw))
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def future_close_metrics(rows: list[tuple[int, float]], trade_date: int, entry_close: float, horizons: list[int]) -> dict[str, Any]:
    future = [(d, c) for (d, c) in rows if d > trade_date]
    out: dict[str, Any] = {}
    for h in horizons:
        key = f"ret_{h}"
        if len(future) >= h and entry_close > 0:
            out[key] = (future[h - 1][1] / entry_close) - 1.0
        else:
            out[key] = None
    window = future[:60]
    if window and entry_close > 0:
        closes = [c for _, c in window]
        out["max_upside_60"] = (max(closes) / entry_close) - 1.0
        out["max_drawdown_60"] = (min(closes) / entry_close) - 1.0
    else:
        out["max_upside_60"] = None
        out["max_drawdown_60"] = None
    out["horizon_basis"] = "unmasked_ee_ohlcv"
    return out


def liquidity_tier(median_value_kwd: float) -> str:
    if median_value_kwd >= 500000.0:
        return "HIGH"
    if median_value_kwd >= 100000.0:
        return "MID"
    return "LOW"


def base_symbol(symbol: str) -> str:
    s = str(symbol)
    return s.split("__SEG", 1)[0] if "__SEG" in s else s


def load_set_symbols(v21: dict[str, Any]) -> tuple[set[str], set[str]]:
    set_a: set[str] = set()
    set_b: set[str] = set()
    rows = v21.get("benchmark_parity_suite_completion", {}).get("random_top_k_portfolio_rows", [])
    for row in rows:
        symbols = row.get("symbols", [])
        row_set = row.get("set")
        if row_set == "set_a":
            set_a.update(symbols)
        elif row_set == "set_b":
            set_b.update(symbols)
    # fallback to known benchmark rows if random-top-k rows are absent
    if not set_a:
        for row in v21.get("benchmark_parity_suite_completion", {}).get("set_a_symbol_rows", []):
            sym = row.get("symbol")
            if sym:
                set_a.add(sym)
    if not set_b:
        for row in v21.get("benchmark_parity_suite_completion", {}).get("set_b_symbol_rows", []):
            sym = row.get("symbol")
            if sym:
                set_b.add(sym)
    return set_a, set_b


def extract_taxonomy_classes(triage: dict[str, Any], ca_v02: dict[str, Any]) -> dict[str, Any]:
    dispositions: set[str] = set()
    final_classes: set[str] = set()
    classification_labels: set[str] = set()
    suspected_actions: set[str] = set()

    final = triage.get("final", {})
    for key in ("disposition_counts", "summary_counts"):
        counts = final.get(key, {})
        if isinstance(counts, dict):
            dispositions.update(str(k) for k in counts.keys())

    decisions = final.get("decisions", [])
    if isinstance(decisions, list):
        for d in decisions:
            if not isinstance(d, dict):
                continue
            if d.get("disposition"):
                dispositions.add(str(d["disposition"]))
            if d.get("final_class_v4_2"):
                final_classes.add(str(d["final_class_v4_2"]))
            if d.get("original_class_v4"):
                classification_labels.add(str(d["original_class_v4"]))
            if d.get("annotation"):
                classification_labels.add(str(d["annotation"]))

    for row in ca_v02.get("corrected_annotations", []):
        action = row.get("suspected_action")
        if action:
            suspected_actions.add(str(action))

    taxonomy_classes = sorted(
        dispositions
        | final_classes
        | classification_labels
        | suspected_actions
        | {
            "MASKED_INTERVAL_SEGMENT",
            "POST_SUSPENSION_REPRICING",
            "TRUE_CONSECUTIVE",
            "ACCUMULATION_MARKUP_BENCHMARK_LIFECYCLE",
        }
    )

    return {
        "dispositions": sorted(dispositions),
        "final_classes": sorted(final_classes),
        "classification_labels": sorted(classification_labels),
        "suspected_actions": sorted(suspected_actions),
        "taxonomy_classes": taxonomy_classes,
    }


def load_engine_vocabulary() -> dict[str, Any]:
    scanner_path = ROOT / "app" / "services" / "eagle_eye" / "scanner_service.py"
    risk_path = ROOT / "app" / "services" / "eagle_eye" / "risk_service.py"
    scanner_text = scanner_path.read_text(encoding="utf-8")
    risk_text = risk_path.read_text(encoding="utf-8")

    # Keep this explicit and aligned to frozen code constants.
    phases = [
        "NEUTRAL",
        "BASE_FORMING",
        "ACCUMULATION",
        "BREAKOUT_WATCH",
        "BREAKOUT_CONFIRMED",
        "MARKUP",
        "DISTRIBUTION_WARNING",
        "EXIT",
        "AVOID",
    ]

    return {
        "source_files": [
            {
                "path": "app/services/eagle_eye/scanner_service.py",
                "sha256": sha256_file(scanner_path),
            },
            {
                "path": "app/services/eagle_eye/risk_service.py",
                "sha256": sha256_file(risk_path),
            },
        ],
        "phases": phases,
        "known_signal_types_runtime": [],
        "gates_in_code": [
            {
                "gate": "AVOID_CONDITION",
                "coded_definition": "close < sma200 and sma200_slope < 0 and ema10 < ema30 sets phase AVOID",
                "source": "scanner_service.evaluate_symbol",
            },
            {
                "gate": "BREAKOUT_MANDATORY_M1_M5",
                "coded_definition": "M1 close>base, M2 rel_volume, M3 ema10>ema30, M4 chase_guard gap<=8%, M5 liquidity",
                "source": "scanner_service.evaluate_symbol",
            },
            {
                "gate": "ML_GATE",
                "coded_definition": "apply_ml_gate must pass for BREAKOUT_CONFIRMED",
                "source": "scanner_service.evaluate_symbol",
            },
            {
                "gate": "RISK_SUPPRESSION",
                "coded_definition": "can_open_new_position(score,max_positions) can suppress to SIGNAL_SUPPRESSED_RISK",
                "source": "scanner_service.evaluate_symbol + risk_service.can_open_new_position",
            },
            {
                "gate": "LIQUIDITY_FILTER",
                "coded_definition": "median value threshold, min price, zero-volume count constraints",
                "source": "risk_service.liquidity_filter_at",
            },
            {
                "gate": "WARMUP_PENDING",
                "coded_definition": "insufficient indicator warmup forces NEUTRAL/PHASE_ONLY",
                "source": "scanner_service._resolve_warmup_context/evaluate_symbol",
            },
        ],
        "scanner_contains_apply_ml_gate": "apply_ml_gate" in scanner_text,
        "risk_contains_can_open_new_position": "def can_open_new_position" in risk_text,
    }


def build_state_taxonomy(v21: dict[str, Any], triage: dict[str, Any], ca_v02: dict[str, Any]) -> dict[str, Any]:
    tax = extract_taxonomy_classes(triage, ca_v02)
    vocab = load_engine_vocabulary()

    phase_mapping = [
        {
            "engine_state": "AVOID",
            "state_kind": "phase",
            "definition_as_coded": "Trend-down safety phase set by avoid condition.",
            "market_phenomenon_intended": "Persistent bearish regime where entries are discouraged.",
            "misreadable_taxonomy_classes": [
                "TRUE_CONSECUTIVE",
                "POST_SUSPENSION_REPRICING",
                "MASKED_INTERVAL_SEGMENT",
            ],
            "why_misread_possible": "No explicit suspension/CA/masked-seam tag in phase state; only price/MA geometry is observed.",
        },
        {
            "engine_state": "BASE_FORMING",
            "state_kind": "phase",
            "definition_as_coded": "Range-width and sessions-in-range conditions freeze base bounds.",
            "market_phenomenon_intended": "Consolidation before accumulation or breakout.",
            "misreadable_taxonomy_classes": [
                "TRUE_CONSECUTIVE_EXTREME",
                "POST_SUSPENSION_REPRICING",
                "CAPITAL_DECREASE",
            ],
            "why_misread_possible": "Base detection is geometric and does not carry adjudicated corporate-action labels.",
        },
        {
            "engine_state": "ACCUMULATION",
            "state_kind": "phase",
            "definition_as_coded": "Divergence/volume/squeeze gate from BASE_FORMING.",
            "market_phenomenon_intended": "Early accumulation with improving internals.",
            "misreadable_taxonomy_classes": [
                "TRUE_CONSECUTIVE",
                "SUSPECTED_CORPORATE_ACTION",
            ],
            "why_misread_possible": "Feature set does not encode adjudicated event disposition.",
        },
        {
            "engine_state": "BREAKOUT_WATCH",
            "state_kind": "phase",
            "definition_as_coded": "Near-base and relative-volume build-up trigger watch state.",
            "market_phenomenon_intended": "Pre-breakout staging.",
            "misreadable_taxonomy_classes": [
                "LIMIT_DAY_SEQUENCE",
                "POST_SUSPENSION_REPRICING",
            ],
            "why_misread_possible": "No dedicated exchange-state marker for limit-day microstructure or suspension resumption.",
        },
        {
            "engine_state": "BREAKOUT_CONFIRMED",
            "state_kind": "phase/signal",
            "definition_as_coded": "Mandatory breakout gates + confirmatory flags + ml_gate + score>=70.",
            "market_phenomenon_intended": "Validated breakout suitable for entry.",
            "misreadable_taxonomy_classes": [
                "TRUE_CONSECUTIVE_EXTREME",
                "DEFERRED_TO_CA_LEDGER",
                "MASKED_INTERVAL_SEGMENT",
            ],
            "why_misread_possible": "Breakout logic consumes unmasked segmented price series without explicit event-class fields.",
        },
        {
            "engine_state": "SIGNAL_SUPPRESSED_RISK",
            "state_kind": "gate_output",
            "definition_as_coded": "Attempted entry signal vetoed by can_open_new_position.",
            "market_phenomenon_intended": "Portfolio-level risk budget protection.",
            "misreadable_taxonomy_classes": [
                "ACCUMULATION_MARKUP_BENCHMARK_LIFECYCLE",
                "TRUE_CONSECUTIVE",
            ],
            "why_misread_possible": "Suppression is capacity/score-driven and independent of benchmark lifecycle class.",
        },
        {
            "engine_state": "DISTRIBUTION_WARNING/EXIT",
            "state_kind": "phase/signal",
            "definition_as_coded": "Distribution features, EMA/trail/time-stop exits.",
            "market_phenomenon_intended": "Markup exhaustion and liquidation.",
            "misreadable_taxonomy_classes": [
                "POST_SUSPENSION_REPRICING",
                "MASKED_INTERVAL_SEGMENT",
            ],
            "why_misread_possible": "Exit triggers are technical and do not tag suspension/corporate-action semantics.",
        },
    ]

    represented_taxonomy = {
        "ACCUMULATION_MARKUP_BENCHMARK_LIFECYCLE",
        "TRUE_CONSECUTIVE",
    }
    missing_taxonomy = sorted(set(tax["taxonomy_classes"]) - represented_taxonomy)

    return {
        "version_id": "R13_STATE_TAXONOMY_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "r12_exam_results_v2_1": "artifacts/preview1a_prestart/review_final/r12_exam_results_v2_1.json",
            "r12_breach_triage_v4_2_FINAL": "artifacts/preview1a_prestart/review_final/r12_breach_triage_v4_2_FINAL.json",
            "r12_ca_ledger_v0_2": "artifacts/preview1a_prestart/review_final/r12_ca_ledger_v0_2.json",
            "r12_pre_exam_surface_seal_v4_4": "artifacts/preview1a_prestart/review_final/r12_pre_exam_surface_seal_v4_4.json",
        },
        "source_trace": vocab,
        "taxonomy_inventory": tax,
        "state_mapping_table": phase_mapping,
        "gap_list_market_states_without_engine_representation": [
            {
                "taxonomy_state": "DEFERRED_TO_CA_LEDGER",
                "evidence": "Disposition exists in sealed triage and mask manifests but not as scanner phase/gate state.",
            },
            {
                "taxonomy_state": "MASKED_INTERVAL_SEGMENT",
                "evidence": "Segmentation is implemented at exam-surface level, not represented as runtime phase in scanner state machine.",
            },
            {
                "taxonomy_state": "POST_SUSPENSION_REPRICING",
                "evidence": "No explicit suspension-resumption state variable in scanner/risk phase vocabulary.",
            },
            {
                "taxonomy_state": "CAPITAL_DECREASE",
                "evidence": "Corporate-action suspicion is adjudication metadata, not a first-class runtime engine state.",
            },
            {
                "taxonomy_state": "OWNER_VERIFIED_MISSING_SESSION_SEMANTICS",
                "evidence": "Calendar adjudication drives surface preparation but is not encoded in per-symbol scanner state.",
            },
        ],
        "missing_taxonomy_classes_by_set_difference": missing_taxonomy,
        "constraints": {
            "paper_only": True,
            "no_engine_changes": True,
            "no_runs": True,
        },
    }


def build_universe_profile(runtime_db: Path, mask_manifest: dict[str, Any], triage: dict[str, Any]) -> dict[str, Any]:
    con = sqlite3.connect(runtime_db)
    cur = con.cursor()

    source_rows = cur.execute(
        "SELECT symbol, trade_date, close, volume, COALESCE(value_kwd,0.0), COALESCE(is_masked,0) FROM ee_ohlcv_masked_source ORDER BY symbol, trade_date"
    ).fetchall()

    by_symbol: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "value_kwd": [],
        "volume": [],
        "total_sessions": 0,
        "masked_sessions_source": 0,
    })
    for sym, _td, _close, vol, val, is_masked in source_rows:
        rec = by_symbol[sym]
        rec["value_kwd"].append(float(val or 0.0))
        rec["volume"].append(float(vol or 0.0))
        rec["total_sessions"] += 1
        rec["masked_sessions_source"] += int(is_masked or 0)

    mask_interval_count_by_symbol = Counter()
    for row in mask_manifest.get("intervals", []):
        sym = row.get("symbol")
        if sym:
            mask_interval_count_by_symbol[str(sym)] += 1

    breach_count_by_symbol = Counter()
    final = triage.get("final", {})
    for entry in final.get("ca_ledger_v0_1", {}).get("entries", []):
        sym = entry.get("symbol")
        cnt = int(entry.get("event_count") or 0)
        if sym:
            breach_count_by_symbol[str(sym)] += cnt
    decisions = final.get("decisions", [])
    if isinstance(decisions, list):
        for d in decisions:
            if not isinstance(d, dict):
                continue
            sym = d.get("symbol")
            if sym:
                breach_count_by_symbol[str(sym)] += 1

    profile_rows: list[dict[str, Any]] = []
    for sym in sorted(by_symbol.keys()):
        rec = by_symbol[sym]
        total = int(rec["total_sessions"])
        vals = rec["value_kwd"]
        vols = rec["volume"]
        med_val = float(median(vals)) if vals else 0.0
        zero_share = (sum(1 for v in vols if v <= 0.0) / total) if total else 0.0
        masked_share = (int(rec["masked_sessions_source"]) / total) if total else 0.0
        breach_events = int(breach_count_by_symbol.get(sym, 0))
        breach_density = (breach_events / total) if total else 0.0
        tier = liquidity_tier(med_val)
        profile_rows.append(
            {
                "symbol": sym,
                "liquidity_tier": tier,
                "median_daily_value_traded_kwd": med_val,
                "zero_volume_session_share": zero_share,
                "masked_interval_count": int(mask_interval_count_by_symbol.get(sym, 0)),
                "masked_interval_burden": masked_share,
                "breach_event_count": breach_events,
                "breach_event_density": breach_density,
                "total_sessions": total,
            }
        )

    tier_counts = Counter(r["liquidity_tier"] for r in profile_rows)
    con.close()

    return {
        "version_id": "R13_UNIVERSE_TIER_PROFILE_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_db": str(runtime_db),
        "rows": profile_rows,
        "tier_summary": {
            "counts": dict(sorted(tier_counts.items())),
            "symbol_count": len(profile_rows),
        },
    }


def build_gate_conflicts(
    runtime_db: Path,
    v21: dict[str, Any],
    universe_profile: dict[str, Any],
) -> dict[str, Any]:
    set_a, set_b = load_set_symbols(v21)
    set_bench_forensics = v21.get("set_a_no_trade_forensics", [])

    tier_by_symbol = {r["symbol"]: r["liquidity_tier"] for r in universe_profile.get("rows", [])}

    con = sqlite3.connect(runtime_db)
    cur = con.cursor()
    signal_rows = cur.execute(
        "SELECT id, symbol, trade_date, signal_type, phase_from, phase_to, score, price, evidence_json FROM ee_signals ORDER BY id"
    ).fetchall()

    prices_by_symbol: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for sym, td, close in cur.execute("SELECT symbol, trade_date, close FROM ee_ohlcv ORDER BY symbol, trade_date").fetchall():
        prices_by_symbol[base_symbol(str(sym))].append((int(td), float(close or 0.0)))

    profile_symbols = [r.get("symbol") for r in universe_profile.get("rows", []) if r.get("symbol")]
    event_rows: list[dict[str, Any]] = []
    gate_counts = Counter()
    gate_tier_counts = Counter()
    symbol_gate_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for sid, sym, td, stype, phase_from, phase_to, score, price, evidence_raw in signal_rows:
        sym_raw = str(sym)
        sym_base = base_symbol(sym_raw)
        evidence = safe_json_load(evidence_raw)
        gate: str | None = None
        attempted = evidence.get("attempted_signal_type")
        suppressed_reason = evidence.get("suppressed_reason")

        if stype == "SIGNAL_SUPPRESSED_RISK":
            gate = "RISK_SUPPRESSION"
        elif stype == "AVOID_SET" or phase_to == "AVOID":
            gate = "AVOID_GATE"
        elif stype == "PHASE_ONLY" and evidence.get("reason") == "warmup_pending":
            gate = "WARMUP_GATE"

        if gate is None:
            continue

        entry_price = float(price or evidence.get("close") or 0.0)
        metrics = future_close_metrics(prices_by_symbol.get(sym_base, []), int(td), entry_price, [5, 20, 60])
        cohort = "set_a" if sym_base in set_a else ("set_b" if sym_base in set_b else "universe")
        tier = tier_by_symbol.get(sym_base, "UNKNOWN")
        row = {
            "signal_id": int(sid),
            "symbol": sym_base,
            "symbol_segment": sym_raw,
            "trade_date": int(td),
            "trade_date_iso": ts_to_iso(int(td)),
            "cohort": cohort,
            "gate": gate,
            "signal_type_recorded": str(stype),
            "attempted_signal_type": attempted,
            "suppressed_reason": suppressed_reason,
            "phase_from": phase_from,
            "phase_to": phase_to,
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
                "suppressed_reason": suppressed_reason,
                "attempted_signal_type": attempted,
                "confirming_c_score": evidence.get("confirming_c_score"),
            },
            "subsequent_unmasked_outcome": metrics,
        }
        event_rows.append(row)
        gate_counts[gate] += 1
        gate_tier_counts[(gate, tier)] += 1
        symbol_gate_counts[sym_base][gate] += 1

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        by_gate: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_gate[r["gate"]].append(r)
        for gate, grows in sorted(by_gate.items()):
            ret5 = [x["subsequent_unmasked_outcome"]["ret_5"] for x in grows if x["subsequent_unmasked_outcome"]["ret_5"] is not None]
            ret20 = [x["subsequent_unmasked_outcome"]["ret_20"] for x in grows if x["subsequent_unmasked_outcome"]["ret_20"] is not None]
            ret60 = [x["subsequent_unmasked_outcome"]["ret_60"] for x in grows if x["subsequent_unmasked_outcome"]["ret_60"] is not None]
            out[gate] = {
                "count": len(grows),
                "mean_ret_5": (sum(ret5) / len(ret5)) if ret5 else None,
                "mean_ret_20": (sum(ret20) / len(ret20)) if ret20 else None,
                "mean_ret_60": (sum(ret60) / len(ret60)) if ret60 else None,
            }
        return out

    all_aggregate = aggregate(event_rows)
    set_a_rows = [r for r in event_rows if r["cohort"] == "set_a"]
    set_b_rows = [r for r in event_rows if r["cohort"] == "set_b"]

    by_gate_by_tier = []
    for (gate, tier), c in sorted(gate_tier_counts.items()):
        by_gate_by_tier.append({"gate": gate, "liquidity_tier": tier, "count": int(c)})

    symbol_summary: list[dict[str, Any]] = []
    for sym in sorted(profile_symbols):
        counts = symbol_gate_counts.get(sym, Counter())
        total = sum(int(v) for v in counts.values())
        symbol_summary.append(
            {
                "symbol": sym,
                "cohort": "set_a" if sym in set_a else ("set_b" if sym in set_b else "universe"),
                "liquidity_tier": tier_by_symbol.get(sym, "UNKNOWN"),
                "total_block_or_suppression_events": total,
                "by_gate": dict(sorted((k, int(v)) for k, v in counts.items())),
            }
        )
    symbols_without_events = [r["symbol"] for r in symbol_summary if r["total_block_or_suppression_events"] == 0]

    con.close()

    return {
        "version_id": "R13_GATE_CONFLICT_ANALYSIS_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_db": str(runtime_db),
        "constraints": {
            "set_b_rows_included_for_descriptive_only": True,
            "set_b_not_used_for_design_implications": True,
        },
        "set_membership": {
            "set_a": sorted(set_a),
            "set_b": sorted(set_b),
        },
        "suppression_events": event_rows,
        "aggregate_counts_by_gate": dict(sorted(gate_counts.items())),
        "aggregate_by_gate": all_aggregate,
        "aggregate_by_gate_and_liquidity_tier": by_gate_by_tier,
        "set_a_aggregate_by_gate": aggregate(set_a_rows),
        "set_b_descriptive_aggregate_by_gate": aggregate(set_b_rows),
        "symbol_gate_coverage": symbol_summary,
        "symbols_with_zero_block_or_suppression_events": symbols_without_events,
        "set_a_benchmark_day_detail": set_bench_forensics,
    }


def build_architecture_md(gate_analysis: dict[str, Any], universe_profile: dict[str, Any]) -> str:
    tier_counts = universe_profile.get("tier_summary", {}).get("counts", {})
    gate_counts = gate_analysis.get("aggregate_counts_by_gate", {})

    return "\n".join(
        [
            "# R13 Three-Model Architecture Proposals v1",
            "",
            "Scope: paper-only proposal set. No code, no simulation, no rerun.",
            "",
            "## Inputs",
            "- R12 run-2 recorded artifacts and runtime DB evidence.",
            "- R13 gate-conflict aggregates and universe liquidity-tier profile.",
            f"- Liquidity tiers (symbol counts): {json.dumps(tier_counts, ensure_ascii=True)}",
            f"- Observed gate counts: {json.dumps(gate_counts, ensure_ascii=True)}",
            "",
            "## Proposal A - Sequential Tri-Model With Hard Data-Surface Gate",
            "Model boundaries and responsibilities:",
            "- Model 1 (Detection): pattern-state candidate generation from unmasked segmented bars.",
            "- Model 2 (Regime-State): validate candidate against regime persistence and phase continuity.",
            "- Model 3 (Risk/Execution): capacity, liquidity, and position-budget arbitration only.",
            "Data consumed from sealed surface:",
            "- Segmented unmasked bars, mask metadata, owner-verified calendar, adjudicated event dispositions.",
            "Gate authority arbitration:",
            "- Model 1 proposes; Model 2 can downgrade/hold; Model 3 can veto entries only on explicit capacity/liquidity evidence.",
            "Universe assumption:",
            "- Full 139-symbol universe preserved; risk layer applies liquidity-tiered position sizing rather than exclusion.",
            "R12 failure addressed:",
            "- Prevents detection-vs-risk hidden coupling by constraining veto reasons to explicit evidence classes.",
            "Testable predictions for future exam:",
            "- Lower AVOID/RISK suppression share on Set A benchmark-active windows.",
            "- Higher ratio of attempted detections reaching executable-state on MID/HIGH tiers.",
            "Honest costs/risks:",
            "- Increased orchestration complexity and audit payload size.",
            "- More arbitration states to govern and monitor.",
            "",
            "## Proposal B - Parallel Specialist Models With Evidence Council",
            "Model boundaries and responsibilities:",
            "- Specialist Detector A: accumulation/base pathways.",
            "- Specialist Detector B: breakout/continuation pathways.",
            "- Specialist Controller C: risk/execution council with immutable veto taxonomy.",
            "Data consumed from sealed surface:",
            "- Same sealed bars/calendar/dispositions; each specialist receives identical surface and emits normalized evidence vectors.",
            "Gate authority arbitration:",
            "- Council vote requires explicit veto codebook (capacity, liquidity, drawdown budget, warmup invalid).",
            "- Uncoded veto is disallowed.",
            "Universe assumption:",
            "- Liquidity-tiered treatment: HIGH/MID run full participation; LOW tier requires stricter execution-capacity proofs but remains in study reporting.",
            "R12 failure addressed:",
            "- Separates accumulation detection from breakout confirmation to reduce state-machine collapse into AVOID-only outcomes.",
            "Testable predictions for future exam:",
            "- Improved Set A signal continuity through benchmark lifecycle phases.",
            "- Reduced false blanket suppression where attempted_signal_type is repeatedly null.",
            "Honest costs/risks:",
            "- Potential disagreement churn between specialist detectors.",
            "- Governance overhead for veto codebook maintenance.",
            "",
            "## Proposal C - Regime-First Controller With Detection as Conditional Worker",
            "Model boundaries and responsibilities:",
            "- Model 1 (Regime Controller): establishes tradable regime envelope and phase admissibility.",
            "- Model 2 (Detection Worker): only runs when regime envelope is open.",
            "- Model 3 (Execution Guard): maps signals to executable orders with tier-aware constraints.",
            "Data consumed from sealed surface:",
            "- Owner-verified calendar and segmented bars first; adjudicated dispositions inform regime envelope boundaries.",
            "Gate authority arbitration:",
            "- Regime controller has first-pass veto on inadmissible windows.",
            "- Execution guard has second-pass veto limited to liquidity/capacity evidence.",
            "Universe assumption:",
            "- Explicit exclusion tier allowed only for execution (not analytics): all symbols remain scored and reported.",
            "R12 failure addressed:",
            "- Makes AVOID-like regime logic explicit and measurable before detection, reducing silent suppression post-detection.",
            "Testable predictions for future exam:",
            "- Fewer late-stage risk suppressions after high-confidence detections.",
            "- Clearer attribution of no-trade outcomes to regime-closed vs risk-closed causes.",
            "Honest costs/risks:",
            "- Regime misclassification can starve detector opportunity.",
            "- Added dependency on regime quality and transition governance.",
            "",
            "## Comparative Notes",
            "- All proposals preserve frozen-baseline principle for R13: document-only, no implementation in this phase.",
            "- Set B outcomes remain carried evidence and are not used as design-justification basis.",
        ]
    ) + "\n"


def md_from_state_taxonomy(payload: dict[str, Any]) -> str:
    lines = [
        "# R13 State-Taxonomy Mapping v1",
        "",
        "## Scope",
        "Read-only mapping between frozen engine vocabulary and adjudicated market-event taxonomy.",
        "",
        "## Source Trace",
    ]
    for row in payload.get("source_trace", {}).get("source_files", []):
        lines.append(f"- {row['path']} :: {row['sha256']}")

    lines += ["", "## Mapping Table"]
    for row in payload.get("state_mapping_table", []):
        lines += [
            f"- State: {row['engine_state']} ({row['state_kind']})",
            f"  - Definition: {row['definition_as_coded']}",
            f"  - Intended phenomenon: {row['market_phenomenon_intended']}",
            f"  - Misreadable taxonomy classes: {', '.join(row['misreadable_taxonomy_classes'])}",
            f"  - Why: {row['why_misread_possible']}",
        ]

    lines += ["", "## Gap List"]
    for row in payload.get("gap_list_market_states_without_engine_representation", []):
        lines.append(f"- {row['taxonomy_state']}: {row['evidence']}")

    lines += ["", "## Constraints", "- Paper-only phase", "- No engine modifications", "- No statistical runs"]
    return "\n".join(lines) + "\n"


def md_from_gate_conflicts(payload: dict[str, Any]) -> str:
    lines = [
        "# R13 Gate-Conflict Analysis v1",
        "",
        "## Scope",
        "Read-only analysis from run-2 recorded signals and evidence payloads.",
        "",
        "## Aggregate Counts by Gate",
    ]
    for gate, count in payload.get("aggregate_counts_by_gate", {}).items():
        lines.append(f"- {gate}: {count}")

    lines += ["", "## Aggregate by Gate and Liquidity Tier"]
    for row in payload.get("aggregate_by_gate_and_liquidity_tier", []):
        lines.append(f"- gate={row['gate']} tier={row['liquidity_tier']} count={row['count']}")

    lines += ["", "## Set A Benchmark-Day Detail"]
    for row in payload.get("set_a_benchmark_day_detail", []):
        lines.append(f"- {row.get('symbol')}: primary_blocker={row.get('primary_blocker')} signal_counts={row.get('signal_type_counts')} ")

    lines += [
        "",
        "## Set B Handling Constraint",
        "- Set B rows are included for descriptive carrying statistics only.",
        "- No threshold finding or design implication is derived from Set B outcomes in this artifact.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    v21 = read_json(REVIEW_DIR / "r12_exam_results_v2_1.json")
    triage = read_json(REVIEW_DIR / "r12_breach_triage_v4_2_FINAL.json")
    mask_manifest = read_json(REVIEW_DIR / "r12_masked_intervals_manifest_v4_3_final.json")
    ca_v02 = read_json(REVIEW_DIR / "r12_ca_ledger_v0_2.json")

    runtime_db_raw = (
        v21.get("runtime_bindings", {}).get("runtime_db_path")
        or v21.get("containment", {}).get("runtime_db_path")
        or str(REVIEW_DIR / "r12_exam_surface_v4_5_runtime.db")
    )
    runtime_db = Path(runtime_db_raw)
    if not runtime_db.exists():
        runtime_db = REVIEW_DIR / "r12_exam_surface_v4_5_runtime.db"
    if not runtime_db.exists():
        raise FileNotFoundError(f"Runtime DB not found: {runtime_db_raw}")

    state_taxonomy = build_state_taxonomy(v21, triage, ca_v02)
    universe_profile = build_universe_profile(runtime_db, mask_manifest, triage)
    gate_conflicts = build_gate_conflicts(runtime_db, v21, universe_profile)
    architecture_md = build_architecture_md(gate_conflicts, universe_profile)

    out_state_json = REVIEW_DIR / "r13_state_taxonomy_v1.json"
    out_state_md = REVIEW_DIR / "r13_state_taxonomy_v1.md"
    out_gate_json = REVIEW_DIR / "r13_gate_conflict_analysis_v1.json"
    out_gate_md = REVIEW_DIR / "r13_gate_conflict_analysis_v1.md"
    out_arch_md = REVIEW_DIR / "r13_architecture_proposals_v1.md"
    out_universe_json = REVIEW_DIR / "r13_universe_tier_profile_v1.json"

    write_json(out_state_json, state_taxonomy)
    out_state_md.write_text(md_from_state_taxonomy(state_taxonomy), encoding="utf-8")

    write_json(out_gate_json, gate_conflicts)
    out_gate_md.write_text(md_from_gate_conflicts(gate_conflicts), encoding="utf-8")

    out_arch_md.write_text(architecture_md, encoding="utf-8")
    write_json(out_universe_json, universe_profile)

    manifest_files = [
        Path("scripts/r13_generate_paper_outputs_v1.py"),
        Path("artifacts/preview1a_prestart/review_final/r13_state_taxonomy_v1.json"),
        Path("artifacts/preview1a_prestart/review_final/r13_state_taxonomy_v1.md"),
        Path("artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1.json"),
        Path("artifacts/preview1a_prestart/review_final/r13_gate_conflict_analysis_v1.md"),
        Path("artifacts/preview1a_prestart/review_final/r13_architecture_proposals_v1.md"),
        Path("artifacts/preview1a_prestart/review_final/r13_universe_tier_profile_v1.json"),
    ]

    created_rows = []
    for rel in manifest_files:
        abs_path = ROOT / rel
        created_rows.append(
            {
                "path": rel.as_posix(),
                "sha256": sha256_file(abs_path),
                "size_bytes": abs_path.stat().st_size,
            }
        )

    manifest = {
        "scope": "R13 paper-only deliverables",
        "r13_execution_status": "DELIVERED_PAPER_ONLY",
        "created_files": created_rows,
    }

    manifest_path = REVIEW_DIR / "r13_created_files_manifest_v1.json"
    manifest_sha_path = REVIEW_DIR / "r13_created_files_manifest_v1.sha256"
    write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    manifest_sha_path.write_text(
        f"{manifest_sha}  artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1.json\n",
        encoding="utf-8",
    )

    print("R13_PAPER_OUTPUTS_COMPLETE")
    print("runtime_db", runtime_db)
    print("manifest_sha256", manifest_sha)


if __name__ == "__main__":
    main()
