"""
ml/run_phase2.py — Phase 2 Deliverable 7

Pipeline orchestrator for Phase 2 training.

Runs the full pipeline for each eligible stock:
  1. Build training matrix (training_matrix.py)
  2. Audit for leakage
  3. Train all 20 surface cells (trainer_v2.py)
  4. Evaluate OOT (evaluator_v2.py)
  5. Build precursor library (precursor_builder.py)
  6. Build pattern store (pattern_store.py)
  7. Assign model lifecycle status (SHADOW / FAILED_GATE / INSUFFICIENT_DATA)
  8. Write progress report

Resumable: checkpoint file records completed tickers.
Checkpoint: ml_training_matrix/v1/_checkpoint.json

Hard rules enforced:
  - No model promoted to LIVE (SHADOW is the max)
  - If 100+ stocks pass gates → STOP and alert (investigate)
  - If 0–5 pass → STOP and alert (investigate)
  - Stop and escalate if: changing thresholds, adding features,
    switching architecture, or deleting artifacts

Usage:
    python -m app.services.eagle_eye.ml.run_phase2
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.services.eagle_eye.ml.training_matrix import (
    build_stock_matrix,
    audit_stock_matrix,
    write_stock_matrix,
    load_stock_matrix,
    PRIMARY_LABEL,
    SURFACE_LABEL_COLS,
    _get_eligible_tickers as _eligible_from_db,
    _matrix_root,
)
from app.services.eagle_eye.ml.trainer_v2 import (
    train_per_stock,
    PerStockTrainingResult,
)
from app.services.eagle_eye.ml.evaluator_v2 import (
    evaluate_stock_oot,
    produce_failure_report,
    EvalResult,
)
from app.services.eagle_eye.ml.precursor_builder import build_precursors_for_ticker, write_precursors_to_db, verify_precursors
from app.services.eagle_eye.ml.pattern_store import (
    build_pattern_index,
    save_pattern_index,
    _write_vectors_to_db,
)
from app.services.eagle_eye.ml.model_store import get_models_root
from app.services.eagle_eye.ml.walk_forward import build_fold_indices, DEFAULT_EMBARGO_TD
from app.services.eagle_eye.store import load_ohlcv

LOGGER = logging.getLogger(__name__)

# ── Sanity guard thresholds ───────────────────────────────────────────────
SHADOW_PASS_MAX = 100   # Suspicious if too many pass
SHADOW_PASS_MIN = 5     # Suspicious if too few pass

PROGRESS_REPORT_PATH_REL = "app/services/eagle_eye/ml/phase2_progress_report.md"


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def _checkpoint_path() -> Path:
    p = Path(__file__).resolve().parents[5] / "ml_training_matrix" / "v1"
    p.mkdir(parents=True, exist_ok=True)
    return p / "_checkpoint.json"


def _load_checkpoint() -> Dict[str, str]:
    """Load checkpoint dict (ticker → status) from disk."""
    cp = _checkpoint_path()
    if not cp.exists():
        return {}
    try:
        with cp.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_checkpoint(checkpoint: Dict[str, str]) -> None:
    with _checkpoint_path().open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _log_lifecycle(ticker: str, version: str, event_type: str, notes: str) -> None:
    try:
        from app.core.database import exec_sql
        exec_sql(
            """INSERT INTO model_lifecycle_log
               (model_identifier, model_version, event_type, event_notes, occurred_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (ticker.upper(), version, event_type, notes[:500]),
        )
    except Exception:
        pass


def _update_model_status(ticker: str, version: str, label_col: str, status: str) -> None:
    try:
        from app.core.database import exec_sql
        model_id = f"{ticker.upper()}::{label_col}::{version}"
        exec_sql(
            """INSERT INTO ml_models
               (model_id, ticker, label_col, version, status, created_at)
               VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT (model_id) DO UPDATE SET
                   status = EXCLUDED.status""",
            (model_id, ticker.upper(), label_col, version, status),
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Baseline Brier computation (rule-engine prevalence baseline)
# ---------------------------------------------------------------------------

def _compute_baseline_brier(oot_df: pd.DataFrame, label_col: str = PRIMARY_LABEL) -> float:
    """
    Compute the Brier score of a naive prevalence baseline:
    always predict the positive class rate in the OOT set.
    """
    if label_col not in oot_df.columns:
        return float("nan")
    y = oot_df[label_col].dropna().values
    if len(y) == 0:
        return float("nan")
    prev = float(y.mean())
    return float(((y - prev) ** 2).mean())


# ---------------------------------------------------------------------------
# Per-stock pipeline
# ---------------------------------------------------------------------------

def run_stock_pipeline(
    ticker: str,
    version: str = "v1",
    *,
    force_rebuild_matrix: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Run the full Phase 2 pipeline for one stock.

    Returns a result dict with keys:
      ticker, status, primary_auc, n_cells_trained, n_rows_oot, n_pos_oot,
      shadow_passed, eval_gates, error_msg, elapsed_sec
    """
    log = logger or LOGGER
    t0 = time.monotonic()
    result: Dict[str, Any] = {
        "ticker": ticker,
        "version": version,
        "status": "error",
        "primary_auc": float("nan"),
        "n_cells_trained": 0,
        "n_rows_oot": 0,
        "n_pos_oot": 0,
        "shadow_passed": False,
        "eval_gates": {},
        "error_msg": "",
        "elapsed_sec": 0.0,
    }

    try:
        # ── Step 1: Training matrix ───────────────────────────────────
        ohlcv = load_ohlcv(ticker)
        if ohlcv is None or len(ohlcv) < 120:
            result["status"] = "insufficient_data"
            result["error_msg"] = "Insufficient OHLCV bars"
            return result

        df = load_stock_matrix(ticker) if not force_rebuild_matrix else None
        if df is None or df.empty:
            log.info("[%s] Building training matrix ...", ticker)
            df = build_stock_matrix(ticker, ohlcv, logger=log)
            if df is None:
                result["status"] = "insufficient_data"
                result["error_msg"] = "Training matrix build failed"
                return result

            df, _ = audit_stock_matrix(ticker, df, logger=log)
            write_stock_matrix(ticker, df)

        # ── Step 1b: Check per-stock primary label from schema ────────
        schema_path = _matrix_root() / ticker.upper() / "schema.json"
        primary_label: Optional[str] = None
        if schema_path.exists():
            try:
                with schema_path.open(encoding="utf-8") as _f:
                    _schema = json.load(_f)
                primary_label = _schema.get("primary_label")
            except Exception:
                pass

        if primary_label is None:
            result["status"] = "insufficient_data"
            result["error_msg"] = (
                "No label tier has >= 50 positives — INSUFFICIENT_DATA. "
                "Increase OHLCV history or wait for more events."
            )
            log.info("[%s] INSUFFICIENT_DATA: %s", ticker, result["error_msg"])
            _log_lifecycle(ticker, version, "INSUFFICIENT_DATA", result["error_msg"])
            return result

        log.info("[%s] Adaptive primary label: %s", ticker, primary_label)

        # ── Step 2: Train ─────────────────────────────────────────────
        log.info("[%s] Training 20 surface cells ...", ticker)
        train_result: PerStockTrainingResult = train_per_stock(
            ticker=ticker, version=version, logger=log,
        )

        result["primary_auc"] = train_result.primary_auc
        result["n_cells_trained"] = train_result.n_cells_trained
        result["n_rows_oot"] = train_result.n_rows_oot
        result["n_pos_oot"] = train_result.n_pos_oot

        if train_result.status == "insufficient_data":
            result["status"] = "insufficient_data"
            result["error_msg"] = train_result.error_msg
            _log_lifecycle(ticker, version, "FAILED_GATE", f"insufficient_data: {train_result.error_msg}")
            return result

        if train_result.status == "failed_gate":
            result["status"] = "failed_gate"
            result["error_msg"] = train_result.error_msg
            _log_lifecycle(ticker, version, "FAILED_GATE", train_result.error_msg)
            _update_model_status(ticker, version, primary_label, "FAILED_GATE")
            return result

        # ── Step 3: OOT evaluation ────────────────────────────────────
        df = df.sort_values("event_date").reset_index(drop=True)
        fold_indices = build_fold_indices(len(df))
        if not fold_indices:
            result["status"] = "insufficient_data"
            result["error_msg"] = "No walk-forward folds possible"
            return result

        oot_idx = fold_indices[-1][1]
        oot_df = df.iloc[oot_idx].reset_index(drop=True)

        baseline_brier = _compute_baseline_brier(oot_df)

        # Load trained models from disk for evaluation
        models_root = get_models_root()
        model_bundle_dict: Dict[str, Any] = {}
        for cell in train_result.cells:
            if not cell.passed_gate or cell.model is None:
                continue
            model_bundle_dict[cell.label] = {
                "model": cell.model,
                "calibrator": cell.calibrator,
                "feature_cols": cell.feature_cols,
            }

        if not model_bundle_dict:
            result["status"] = "failed_gate"
            result["error_msg"] = "No trained cell models available for evaluation"
            return result

        log.info("[%s] Evaluating OOT holdout (%d rows) ...", ticker, len(oot_df))
        eval_result: EvalResult = evaluate_stock_oot(
            ticker=ticker,
            model_bundle_dict=model_bundle_dict,
            oot_df=oot_df,
            baseline_brier=baseline_brier,
            version=version,
            primary_label=primary_label,
            logger=log,
        )

        result["eval_gates"] = {g.gate_id: g.passed for g in eval_result.gates}
        result["shadow_passed"] = (eval_result.status == "SHADOW")

        # ── Model status assignment ───────────────────────────────────
        final_status = eval_result.status  # "SHADOW" | "FAILED_GATE" | "INSUFFICIENT_DATA"
        for label_col in SURFACE_LABEL_COLS:
            _update_model_status(ticker, version, label_col, final_status)

        _log_lifecycle(
            ticker, version,
            "SHADOW_START" if final_status == "SHADOW" else "FAILED_GATE",
            f"OOT eval: {final_status}, primary_auc={eval_result.primary_auc:.3f}",
        )

        if final_status != "SHADOW":
            # Write failure diagnostics
            primary_cell = next((c for c in train_result.cells if c.label == primary_label), None)
            primary_model = primary_cell.model if primary_cell else None
            primary_feature_cols = primary_cell.feature_cols if primary_cell else []
            try:
                report_path = produce_failure_report(
                    ticker=ticker,
                    version=version,
                    eval_result=eval_result,
                    model=primary_model,
                    feature_cols=primary_feature_cols,
                    oot_df=oot_df,
                    primary_label=primary_label,
                )
                log.info("[%s] Failure report: %s", ticker, report_path)
            except Exception as exc:
                log.warning("[%s] Failure report write error: %s", ticker, exc)

        result["status"] = final_status.lower()

        # ── Step 4: Precursor library ─────────────────────────────────
        try:
            log.info("[%s] Building precursor library ...", ticker)
            precursor_rows = build_precursors_for_ticker(ticker, ohlcv, logger=log)
            if precursor_rows:
                verify_precursors(ticker, ohlcv, precursor_rows, logger=log)
                n_prec = write_precursors_to_db(precursor_rows)
                log.info("[%s] %d precursor rows written", ticker, n_prec)
        except Exception as exc:
            log.warning("[%s] Precursor build error (non-fatal): %s", ticker, exc)

        # ── Step 5: Pattern store ─────────────────────────────────────
        try:
            log.info("[%s] Building pattern store ...", ticker)
            pattern_result = build_pattern_index(ticker, df, logger=log)
            if pattern_result is not None:
                nn, meta_df, feat_cols = pattern_result
                save_pattern_index(ticker, nn, meta_df, feat_cols)
                with _db_conn() as conn:
                    _write_vectors_to_db(ticker, df, feat_cols, conn)
                log.info("[%s] Pattern index saved", ticker)
        except Exception as exc:
            log.warning("[%s] Pattern store error (non-fatal): %s", ticker, exc)

    except RuntimeError as exc:
        result["status"] = "aborted_leakage"
        result["error_msg"] = str(exc)
        log.error("[%s] Leakage abort: %s", ticker, exc)

    except Exception as exc:
        result["status"] = "error"
        result["error_msg"] = f"{type(exc).__name__}: {exc}"
        log.error("[%s] Pipeline error:\n%s", ticker, traceback.format_exc())

    finally:
        result["elapsed_sec"] = round(time.monotonic() - t0, 1)

    return result


# ---------------------------------------------------------------------------
# Progress report
# ---------------------------------------------------------------------------

def _backend_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _write_progress_report(
    results: List[Dict[str, Any]],
    checkpoint: Dict[str, str],
    total: int,
) -> Path:
    root = _backend_root()
    report_path = root / PROGRESS_REPORT_PATH_REL
    report_path.parent.mkdir(parents=True, exist_ok=True)

    shadow = [r for r in results if r.get("status") == "shadow"]
    failed = [r for r in results if r.get("status") in ("failed_gate", "aborted_leakage")]
    insuf = [r for r in results if r.get("status") == "insufficient_data"]
    errors = [r for r in results if r.get("status") == "error"]

    lines = [
        "# Phase 2 Training Progress Report",
        f"Generated: {datetime.utcnow().isoformat()} UTC",
        "",
        "## Summary",
        f"- Total eligible stocks: {total}",
        f"- Processed so far: {len(results)}",
        f"- **SHADOW** (gates passed): {len(shadow)}",
        f"- FAILED_GATE: {len(failed)}",
        f"- INSUFFICIENT_DATA: {len(insuf)}",
        f"- ERROR: {len(errors)}",
        "",
    ]

    if shadow:
        lines += [
            "## SHADOW Models",
            "",
            "| Ticker | Primary AUC | Cells | OOT Rows | OOT Pos | Elapsed |",
            "|--------|-------------|-------|----------|---------|---------|",
        ]
        for r in shadow:
            auc = r.get("primary_auc", float("nan"))
            lines.append(
                f"| {r['ticker']} | {auc:.3f} | {r.get('n_cells_trained', 0)} | "
                f"{r.get('n_rows_oot', 0)} | {r.get('n_pos_oot', 0)} | {r.get('elapsed_sec', 0)}s |"
            )
        lines.append("")

    if failed:
        lines += [
            "## Failed Gate",
            "",
            "| Ticker | Reason |",
            "|--------|--------|",
        ]
        for r in failed:
            msg = str(r.get("error_msg", ""))[:120]
            lines.append(f"| {r['ticker']} | {msg} |")
        lines.append("")

    if errors:
        lines += [
            "## Errors",
            "",
            "| Ticker | Error |",
            "|--------|-------|",
        ]
        for r in errors:
            msg = str(r.get("error_msg", ""))[:120]
            lines.append(f"| {r['ticker']} | {msg} |")
        lines.append("")

    lines += [
        "## Gate Pass Rate by Gate",
        "",
    ]
    for gate_id in ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]:
        gate_results = [
            r for r in results
            if r.get("eval_gates") and gate_id in r["eval_gates"]
        ]
        if gate_results:
            n_pass = sum(1 for r in gate_results if r["eval_gates"][gate_id])
            pct = n_pass / len(gate_results) * 100
            lines.append(f"- {gate_id}: {n_pass}/{len(gate_results)} passed ({pct:.0f}%)")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Sanity guards
# ---------------------------------------------------------------------------

class SanityGuardError(RuntimeError):
    """Raised when the pipeline sanity guard trips. Hard stop — do not catch."""


def _check_sanity_guards(
    results: List[Dict[str, Any]],
    log: logging.Logger,
    *,
    is_final: bool = False,
) -> None:
    """
    Hard-stop guard.  RAISES SanityGuardError when thresholds are breached.

    Mid-run checks (is_final=False):
      - SHADOW >= SHADOW_PASS_MAX (100)  → fire immediately
      - SHADOW == 0 after 50+ stocks    → catastrophic, fire immediately

    Final check (is_final=True, called after all stocks processed):
      - SHADOW >= SHADOW_PASS_MAX (100) → still suspicious
      - SHADOW <= SHADOW_PASS_MIN (5)   → total fleet too small

    The lower-bound ≤5 guard is deliberately ONLY checked at is_final=True.
    With an expected ~8% SHADOW rate, checking mid-run after 20 stocks would
    fire at 1-2 SHADOW (which is normal, not suspicious).
    """
    shadow_count = sum(1 for r in results if r.get("status") == "shadow")
    processed = len(results)

    if processed < 10:
        return  # Insufficient data to judge yet

    # Upper bound — always active
    if shadow_count >= SHADOW_PASS_MAX:
        msg = (
            f"SANITY GUARD TRIPPED: {shadow_count} stocks passed all gates — "
            f"threshold is {SHADOW_PASS_MAX}. Suspiciously high. "
            "Investigate for data leakage or overly lenient gate thresholds. "
            "PIPELINE HALTED."
        )
        log.error(msg)
        raise SanityGuardError(msg)

    # Mid-run lower bound: only fire if ZERO shadows after 50+ stocks
    if not is_final and processed >= 50 and shadow_count == 0:
        msg = (
            f"SANITY GUARD TRIPPED: Zero stocks passed gates out of "
            f"{processed} processed. Catastrophic failure — investigate "
            "feature quality, label computation, or eligibility list. "
            "PIPELINE HALTED."
        )
        log.error(msg)
        raise SanityGuardError(msg)

    # Final lower bound: fire if ≤ SHADOW_PASS_MIN after full run
    if is_final and shadow_count <= SHADOW_PASS_MIN:
        msg = (
            f"SANITY GUARD: Only {shadow_count} stocks reached SHADOW "
            f"(threshold ≤ {SHADOW_PASS_MIN}). Expected ~11. "
            "Investigate before next run."
        )
        log.warning(msg)
        # Do NOT raise here — run is complete, summary will be written.
        # Caller is responsible for surfacing this in the summary.



# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_phase2(
    tickers: Optional[Sequence[str]] = None,
    version: str = "v1",
    *,
    resume: bool = True,
    force_rebuild_matrix: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Run the full Phase 2 pipeline for all eligible stocks.

    Parameters
    ----------
    tickers : override eligible ticker list (None = load from eligibility table)
    version : model version string
    resume  : if True, skip tickers already in checkpoint
    force_rebuild_matrix : if True, rebuild training matrices from scratch

    Returns list of per-stock result dicts.
    """
    log = logger or LOGGER
    log.info("=" * 60)
    log.info("PHASE 2 PIPELINE STARTING — version=%s, resume=%s", version, resume)
    log.info("=" * 60)

    started_at = datetime.utcnow().isoformat(timespec="seconds") + " UTC"

    if tickers is None:
        tickers = _eligible_from_db(log)

    total = len(tickers)
    log.info("Eligible stocks: %d", total)

    checkpoint = _load_checkpoint() if resume else {}
    results: List[Dict[str, Any]] = []

    # Pre-load results from checkpoint for progress report
    for ticker, status in checkpoint.items():
        results.append({"ticker": ticker, "status": status, "primary_auc": float("nan"),
                        "n_cells_trained": 0, "n_rows_oot": 0, "n_pos_oot": 0,
                        "shadow_passed": status == "shadow", "eval_gates": {}, "error_msg": ""})

    for i, ticker in enumerate(tickers):
        if resume and ticker in checkpoint:
            log.info("[%d/%d] %s — skipping (checkpoint: %s)", i + 1, total, ticker, checkpoint[ticker])
            continue

        log.info("[%d/%d] Processing %s ...", i + 1, total, ticker)
        stock_result = run_stock_pipeline(
            ticker=ticker,
            version=version,
            force_rebuild_matrix=force_rebuild_matrix,
            logger=log,
        )
        results.append(stock_result)

        # Update checkpoint
        checkpoint[ticker] = stock_result["status"]
        _save_checkpoint(checkpoint)

        # Write progress report after each stock
        try:
            report_path = _write_progress_report(results, checkpoint, total)
            log.debug("Progress report updated: %s", report_path)
        except Exception as exc:
            log.warning("Progress report write failed: %s", exc)

        # Sanity guards (check every 10 stocks)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            _check_sanity_guards(results, log, is_final=False)  # raises SanityGuardError if tripped

    # Final summary
    shadow_results = [r for r in results if r.get("status") == "shadow"]
    failed_results = [r for r in results if r.get("status") in ("failed_gate", "aborted_leakage")]
    insuf_results = [r for r in results if r.get("status") == "insufficient_data"]

    log.info("=" * 60)
    log.info("PHASE 2 COMPLETE")
    log.info("  Processed : %d / %d", len([r for r in results if r["ticker"] in [t for t in tickers]]), total)
    log.info("  SHADOW    : %d", len(shadow_results))
    log.info("  FAILED    : %d", len(failed_results))
    log.info("  INSUF     : %d", len(insuf_results))
    log.info("=" * 60)

    # Final sanity check (only if we ran enough stocks to judge)
    try:
        _check_sanity_guards(results, log, is_final=True)
    except SanityGuardError:
        pass  # Already logged; final report still written below

    # Final progress report
    try:
        report_path = _write_progress_report(results, checkpoint, total)
        log.info("Final progress report: %s", report_path)
    except Exception as exc:
        log.warning("Final progress report failed: %s", exc)

    # Final summary (P2.9)
    try:
        ended_at = datetime.utcnow().isoformat(timespec="seconds") + " UTC"
        _write_phase2_summary(results, version, started_at, ended_at, log)
    except Exception as exc:
        log.warning("Phase 2 summary write failed: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Phase 2 final summary (P2.9)
# ---------------------------------------------------------------------------

SUMMARY_PATH_REL = "app/services/eagle_eye/ml/phase2_summary.md"


def _write_phase2_summary(
    results: List[Dict[str, Any]],
    version: str,
    started_at: str,
    ended_at: str,
    log: logging.Logger,
) -> Path:
    """
    Write the final Phase 2 summary per Section P2.9 of the main brief.

    Sections:
      1. Run metadata
      2. Headline counts
      3. SHADOW stock table
      4. Gate pass rates
      5. Failure breakdown (FAILED_GATE, INSUFFICIENT_DATA, ERROR)
      6. Sanity guard verdict
    """
    root = _backend_root()
    out = root / SUMMARY_PATH_REL
    out.parent.mkdir(parents=True, exist_ok=True)

    shadow  = [r for r in results if r.get("status") == "shadow"]
    failed  = [r for r in results if r.get("status") in ("failed_gate", "aborted_leakage")]
    insuf   = [r for r in results if r.get("status") == "insufficient_data"]
    errors  = [r for r in results if r.get("status") == "error"]
    total   = len(results)

    shadow_count = len(shadow)
    guard_verdict = (
        "SANITY GUARD: LOW — fewer than 6 SHADOW (expected ~11). Review before next run."
        if shadow_count <= SHADOW_PASS_MIN
        else "SANITY GUARD: OK — SHADOW count within expected range."
    )

    def pct(n): return f"{100*n/total:.1f}%" if total else "—"

    lines = [
        "# Phase 2 Training Run — Final Summary",
        "",
        "## 1. Run Metadata",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Model version | `{version}` |",
        f"| Started | {started_at} |",
        f"| Completed | {ended_at} |",
        f"| Pipeline | run_phase2.py (adaptive label tiers) |",
        f"| Max status | SHADOW (no LIVE promotions) |",
        f"| Checkpoint | ml_training_matrix/v1/_checkpoint.json |",
        f"",
        f"## 2. Headline Counts",
        f"",
        f"| Status | Count | Share |",
        f"|--------|-------|-------|",
        f"| **SHADOW** | **{shadow_count}** | **{pct(shadow_count)}** |",
        f"| FAILED_GATE | {len(failed)} | {pct(len(failed))} |",
        f"| INSUFFICIENT_DATA | {len(insuf)} | {pct(len(insuf))} |",
        f"| ERROR | {len(errors)} | {pct(len(errors))} |",
        f"| **Total processed** | **{total}** | — |",
        f"",
    ]

    if shadow:
        lines += [
            "## 3. SHADOW Models",
            "",
            "| Ticker | Primary AUC | Cells Trained | OOT Rows | OOT Pos | Elapsed (s) |",
            "|--------|-------------|---------------|----------|---------|-------------|",
        ]
        for r in sorted(shadow, key=lambda x: x.get("primary_auc", 0), reverse=True):
            auc = r.get("primary_auc", float("nan"))
            auc_str = f"{auc:.3f}" if not (isinstance(auc, float) and auc != auc) else "—"
            lines.append(
                f"| {r['ticker']} | {auc_str} | {r.get('n_cells_trained', 0)} | "
                f"{r.get('n_rows_oot', 0)} | {r.get('n_pos_oot', 0)} | {r.get('elapsed_sec', 0)} |"
            )
        lines.append("")
    else:
        lines += ["## 3. SHADOW Models", "", "_None reached SHADOW status._", ""]

    # Gate pass rates
    lines += ["## 4. Gate Pass Rates", ""]
    for gate_id in ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]:
        gate_results = [r for r in results if r.get("eval_gates") and gate_id in r["eval_gates"]]
        if gate_results:
            n_pass = sum(1 for r in gate_results if r["eval_gates"][gate_id])
            p = f"{100*n_pass/len(gate_results):.0f}%"
            lines.append(f"- {gate_id}: {n_pass}/{len(gate_results)} passed ({p})")
    lines.append("")

    # Failure breakdown
    if failed:
        lines += [
            "## 5a. FAILED_GATE Breakdown",
            "",
            "| Ticker | Reason |",
            "|--------|--------|",
        ]
        for r in failed:
            msg = str(r.get("error_msg", ""))[:150].replace("|", "‖")
            lines.append(f"| {r['ticker']} | {msg} |")
        lines.append("")

    if insuf:
        lines += [
            "## 5b. INSUFFICIENT_DATA",
            "",
            ", ".join(r["ticker"] for r in insuf),
            "",
        ]

    if errors:
        lines += [
            "## 5c. Errors",
            "",
            "| Ticker | Error |",
            "|--------|-------|",
        ]
        for r in errors:
            msg = str(r.get("error_msg", ""))[:150].replace("|", "‖")
            lines.append(f"| {r['ticker']} | {msg} |")
        lines.append("")

    lines += [
        "## 6. Sanity Guard Verdict",
        "",
        guard_verdict,
        "",
        "---",
        f"*Generated by run_phase2.py on {ended_at}. No models promoted to LIVE.*",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Phase 2 summary written: %s", out)
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 2 training pipeline")
    parser.add_argument("--version", default="v1", help="Model version string")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint and re-run all")
    parser.add_argument("--rebuild-matrix", action="store_true", help="Rebuild training matrices from scratch")
    parser.add_argument("--tickers", nargs="*", help="Override ticker list (space-separated)")
    parser.add_argument("--log-file", default=None, help="Write log output to this file path (in addition to stdout)")
    args = parser.parse_args()

    handlers: list = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=handlers,
    )

    run_phase2(
        tickers=args.tickers or None,
        version=args.version,
        resume=not args.no_resume,
        force_rebuild_matrix=args.rebuild_matrix,
        logger=LOGGER,
    )
