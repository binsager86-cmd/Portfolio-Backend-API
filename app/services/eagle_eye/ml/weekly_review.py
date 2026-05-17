"""
ml/weekly_review.py — Phase 3: Weekly shadow-run review report.

Runs every Sunday at 15:00 Asia/Kuwait.  Generates a Markdown review
report covering the 3 flagged stocks (URC, JAZEERA, KCEM) plus a
summary table for all SHADOW-roster stocks.

Output: reports/weekly_flagged_review_{DATE}.md
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger(__name__)

FLAGGED_STOCKS: List[str] = ["URC", "JAZEERA", "KCEM"]


def run_weekly_review(review_date: Optional[str] = None) -> str:
    """
    Generate weekly review Markdown report.

    Parameters
    ----------
    review_date : ISO date string (default: today).

    Returns
    -------
    Absolute path to the written report file.
    """
    from app.core.config import get_settings
    from app.core.database import query_all, query_one
    from app.services.eagle_eye.ml.shadow_runner import SHADOW_ROSTER

    settings = get_settings()
    today_str = review_date or date.today().isoformat()
    today = date.fromisoformat(today_str)
    window_start = (today - timedelta(days=7)).isoformat()

    report_dir = Path(settings.ML_REPORTS_ROOT)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"weekly_flagged_review_{today_str}.md"

    # ── Pull data ─────────────────────────────────────────────────────────
    all_rows = query_all(
        """
        SELECT s.stock_ticker, s.log_date, s.band_label,
               s.calibrated_prob, s.raw_prob, s.rule_stage, s.rule_confidence,
               s.ml_bucket
          FROM ml_shadow_log s
         WHERE s.log_date >= ?
           AND s.log_date <= ?
         ORDER BY s.stock_ticker, s.log_date
        """,
        (window_start, today_str),
    )

    display_state = query_one("SELECT auto_disabled, disabled_reason FROM ml_display_state WHERE id = 1", ())
    disabled = bool(display_state and display_state["auto_disabled"])

    # ── Build report ──────────────────────────────────────────────────────
    lines: List[str] = []
    lines.append(f"# Eagle Eye ML Weekly Review — {today_str}")
    lines.append("")
    lines.append(f"> Generated: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("")

    if disabled:
        lines.append(f"⚠️ **ML display is currently AUTO-DISABLED.** Reason: {display_state['disabled_reason']}")
        lines.append("")

    lines.append("## Summary — All SHADOW Stocks")
    lines.append("")
    lines.append("| Ticker | Days Scored | Last Band | Avg Cal Prob | Agreement Rate |")
    lines.append("|--------|-------------|-----------|--------------|----------------|")

    by_ticker: dict = {}
    for row in all_rows:
        tk = row["stock_ticker"]
        by_ticker.setdefault(tk, []).append(row)

    for ticker in SHADOW_ROSTER:
        rows = by_ticker.get(ticker, [])
        n = len(rows)
        last_band = rows[-1]["band_label"] if rows else "—"
        probs = [float(r["calibrated_prob"]) for r in rows if r["calibrated_prob"] is not None]
        avg_prob = f"{sum(probs)/len(probs):.3f}" if probs else "—"
        agreements = [r for r in rows if r.get("rule_confidence") is not None]
        # Simple agreement: calibrated_prob vs rule_confidence direction
        agree_n = sum(
            1 for r in agreements
            if (float(r["calibrated_prob"]) >= 0.5) == (float(r["rule_confidence"] or 0) >= 0.5)
        )
        agree_rate = f"{agree_n}/{len(agreements)}" if agreements else "—"
        lines.append(f"| {ticker} | {n} | {last_band} | {avg_prob} | {agree_rate} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Flagged Stocks Deep Dive")
    lines.append("")

    for ticker in FLAGGED_STOCKS:
        rows = by_ticker.get(ticker, [])
        lines.append(f"### {ticker}")
        if not rows:
            lines.append("_No shadow data this week._")
            lines.append("")
            continue

        lines.append("")
        lines.append("| Date | Band | Cal Prob | Raw Prob | Rule Stage | Rule Conf |")
        lines.append("|------|------|----------|----------|------------|-----------|")
        for r in rows:
            cal = f"{float(r['calibrated_prob']):.3f}" if r["calibrated_prob"] else "—"
            raw = f"{float(r['raw_prob']):.3f}" if r["raw_prob"] else "—"
            rc = f"{float(r['rule_confidence']):.2f}" if r["rule_confidence"] else "—"
            lines.append(
                f"| {r['log_date']} | {r['band_label'] or '—'} | {cal} | {raw} | {r['rule_stage'] or '—'} | {rc} |"
            )
        lines.append("")

        # Band distribution
        bands = [r["band_label"] for r in rows if r["band_label"]]
        if bands:
            from collections import Counter
            dist = Counter(bands)
            lines.append(f"**Band distribution:** {dict(dist)}")
            lines.append("")

    lines.append("---")
    lines.append("_No model will be promoted to LIVE during Phase 3 evaluation._")
    lines.append("")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    LOGGER.info("weekly_review: report written to %s", report_path)
    return str(report_path)
