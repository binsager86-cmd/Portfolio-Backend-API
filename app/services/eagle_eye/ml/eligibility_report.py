"""
ml/eligibility_report.py — Addendum A.1: ML eligibility coverage report.

Produces ``reports/ml_eligibility.md`` after every eligibility screen run.
Also writes a structured summary dict suitable for the frontend Settings page
("X of 139 stocks are ML-eligible. Y are on rules-only. Z are watch-only.").

Runs automatically after DataPipeline.run_eligibility_screen().
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[4] / "reports"


@dataclass
class EligibilitySummary:
    total_stocks: int
    ml_eligible: int
    rules_only: int
    watch_only: int
    ineligible_detail: List[Dict[str, Any]]
    generated_at: str


def generate_eligibility_report(
    eligibility_records: list,  # list of StockEligibility from data_pipeline
    output_path: Optional[Path] = None,
) -> EligibilitySummary:
    """
    Build the eligibility coverage report and write it to a Markdown file.

    Parameters
    ----------
    eligibility_records : list of StockEligibility
    output_path         : override default path if needed
    """
    from app.services.eagle_eye.ml.data_pipeline import (
        MIN_MOVE_EVENTS,
        MIN_TRADING_DAYS,
        WATCH_ONLY_VOLUME_THRESHOLD,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or REPORTS_DIR / "ml_eligibility.md"

    total = len(eligibility_records)
    ml_eligible = [r for r in eligibility_records if r.eligible and not r.watch_only]
    watch_only  = [r for r in eligibility_records if r.eligible and r.watch_only]
    rules_only  = [r for r in eligibility_records if not r.eligible]

    ineligible_detail = [
        {
            "ticker":       r.ticker,
            "reason":       r.reason,
            "n_moves":      r.n_move_events,
            "n_days":       r.n_trading_days,
            "tier":         r.liquidity_tier,
            "median_vol":   r.median_daily_vol,
        }
        for r in rules_only
    ]

    summary = EligibilitySummary(
        total_stocks=total,
        ml_eligible=len(ml_eligible),
        rules_only=len(rules_only),
        watch_only=len(watch_only),
        ineligible_detail=ineligible_detail,
        generated_at=datetime.utcnow().isoformat(),
    )

    # ── Render Markdown ───────────────────────────────────────────────
    lines = [
        "# Eagle Eye ML Eligibility Report",
        f"\n_Generated: {summary.generated_at} UTC_",
        f"\n**Thresholds:** min move events = {MIN_MOVE_EVENTS}, "
        f"min trading days = {MIN_TRADING_DAYS}, "
        f"watch-only volume threshold = {WATCH_ONLY_VOLUME_THRESHOLD:,} shares/day",
        "\n---\n",
        "## High-Level Summary\n",
        f"| Category | Count |",
        f"| --- | --- |",
        f"| Total stocks screened | {total} |",
        f"| **ML-eligible (full training)** | **{len(ml_eligible)}** |",
        f"| Watch-only (ML where possible) | {len(watch_only)} |",
        f"| Rules-only (ineligible for ML) | {len(rules_only)} |",
        "\n---\n",
        "## Tier Breakdown\n",
    ]

    # Group by liquidity tier
    tier_groups: Dict[str, list] = {}
    for r in eligibility_records:
        tier_groups.setdefault(r.liquidity_tier, []).append(r)

    lines.append("| Tier | Total | ML-Eligible | Rules-Only |")
    lines.append("| --- | --- | --- | --- |")
    for tier, group in sorted(tier_groups.items()):
        elig = sum(1 for r in group if r.eligible)
        inelig = sum(1 for r in group if not r.eligible)
        lines.append(f"| {tier} | {len(group)} | {elig} | {inelig} |")

    lines += ["\n---\n", "## Ineligible Stocks Detail\n"]

    if not ineligible_detail:
        lines.append("_All screened stocks are ML-eligible._\n")
    else:
        lines.append("| Ticker | Reason | Move Events | Trading Days | Tier |")
        lines.append("| --- | --- | --- | --- | --- |")
        for d in sorted(ineligible_detail, key=lambda x: x["ticker"]):
            reason_short = d["reason"].split(":")[0]
            lines.append(
                f"| {d['ticker']} | {reason_short} | {d['n_moves']} | "
                f"{d['n_days']} | {d['tier']} |"
            )

    lines += ["\n---\n", "_This report is regenerated automatically on every retrain cycle._\n"]

    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Eligibility report written to %s", out)
    return summary


def get_eligibility_summary_for_frontend() -> Dict[str, Any]:
    """
    Return the latest eligibility counts from the DB for the frontend
    Settings page.  Falls back to zeros if the table doesn't exist yet.
    """
    try:
        from app.core.database import exec_sql_fetch
        rows = exec_sql_fetch(
            """
            SELECT
                COUNT(*)                                         AS total,
                SUM(CASE WHEN eligible=1 AND watch_only=0 THEN 1 ELSE 0 END) AS ml_eligible,
                SUM(CASE WHEN eligible=0 THEN 1 ELSE 0 END)    AS rules_only,
                SUM(CASE WHEN eligible=1 AND watch_only=1 THEN 1 ELSE 0 END) AS watch_only
            FROM ml_stock_eligibility
            """,
            (),
        )
        if rows:
            row = rows[0]
            return {
                "total":       int(row[0] or 0),
                "ml_eligible": int(row[1] or 0),
                "rules_only":  int(row[2] or 0),
                "watch_only":  int(row[3] or 0),
            }
    except Exception as exc:  # noqa: BLE001
        logger.debug("eligibility summary query failed: %s", exc)
    return {"total": 0, "ml_eligible": 0, "rules_only": 0, "watch_only": 0}
