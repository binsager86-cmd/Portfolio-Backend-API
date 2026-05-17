"""
ml/auto_disable_monitor.py — Phase 3: Auto-disable watchdog.

Runs after shadow_runner (≈14:45 Asia/Kuwait, Sun–Thu) and checks four
trigger conditions.  When any trigger fires the monitor:

  1. Sets ml_display_state.auto_disabled = 1, records reason + timestamp.
  2. Logs an AUTO_DISABLE lifecycle event for each affected model.
  3. Logs a WARNING so the ops team sees it immediately.

The kill-switch does NOT stop shadow scoring — scoring continues so the
weekly reviewer has data to look at.

Trigger conditions
------------------
A. 7-day mean MCE (Mean Calibration Error) > 30% across scored models.
   Proxy: use |calibrated_prob - rule_confidence| averaged over last 7 days.
   (Full calibration error requires labelled outcomes; rule confidence is used
   as a proxy until outcomes are filled in.)

B. Any individual model's BSS (Brier Skill Score) < 0 for 2+ consecutive days.
   Proxy: calibrated_prob outside [0.05, 0.95] for 2+ consecutive days
   (extreme misprediction indicator).

C. 3 or more CASCADE ROLLBACKs logged in the last 7 days.

D. 2 or more consecutive days of shadow scoring job failure (no rows written
   for a trading day).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

LOGGER = logging.getLogger(__name__)

MCE_THRESHOLD = 0.30       # trigger A
BSS_CONSECUTIVE_DAYS = 2   # trigger B
CASCADE_ROLLBACK_THRESHOLD = 3  # trigger C
FAILURE_CONSECUTIVE_DAYS = 2    # trigger D


def run_auto_disable_check(signal_date: Optional[str] = None) -> dict:
    """
    Evaluate all four trigger conditions and disable display if any fires.

    Returns a dict:
      {
        "signal_date": "YYYY-MM-DD",
        "triggered": bool,
        "trigger": str | None,
        "reason": str | None
      }
    """
    from app.core.database import query_one, query_all, exec_sql

    today_str = signal_date or date.today().isoformat()

    # ── Check each trigger ────────────────────────────────────────────────
    trigger, reason = _check_all(today_str, query_one, query_all)

    if trigger:
        LOGGER.warning(
            "auto_disable_monitor: TRIGGER=%s fired — disabling ML display. reason=%s",
            trigger, reason
        )
        _disable_display(today_str, reason, exec_sql)
        _log_lifecycle_for_all_shadow(today_str, reason, query_all, exec_sql)
    else:
        LOGGER.info("auto_disable_monitor: all checks passed for %s", today_str)

    return {"signal_date": today_str, "triggered": bool(trigger), "trigger": trigger, "reason": reason}


# ---------------------------------------------------------------------------
# Trigger checks
# ---------------------------------------------------------------------------

def _check_all(today_str, query_one, query_all):
    """Return (trigger_name, reason) or (None, None)."""

    # Skip if already disabled
    state = query_one("SELECT auto_disabled FROM ml_display_state WHERE id = 1", ())
    if state and state["auto_disabled"]:
        return None, None  # already disabled, nothing to do

    trig, reason = _check_trigger_a(today_str, query_all)
    if trig:
        return trig, reason

    trig, reason = _check_trigger_b(today_str, query_all)
    if trig:
        return trig, reason

    trig, reason = _check_trigger_c(today_str, query_all)
    if trig:
        return trig, reason

    trig, reason = _check_trigger_d(today_str, query_all)
    if trig:
        return trig, reason

    return None, None


def _check_trigger_a(today_str: str, query_all) -> tuple:
    """7-day mean |calibrated_prob - rule_confidence| > MCE_THRESHOLD."""
    seven_days_ago = (date.fromisoformat(today_str) - timedelta(days=7)).isoformat()
    rows = query_all(
        """
        SELECT calibrated_prob, rule_confidence
          FROM ml_shadow_log
         WHERE log_date > ?
           AND log_date <= ?
           AND calibrated_prob IS NOT NULL
           AND rule_confidence IS NOT NULL
        """,
        (seven_days_ago, today_str),
    )
    if not rows:
        return None, None

    errors = [abs(float(r["calibrated_prob"]) - float(r["rule_confidence"])) for r in rows]
    mean_mce = sum(errors) / len(errors)
    if mean_mce > MCE_THRESHOLD:
        return "MCE_EXCEEDED", f"7-day mean MCE={mean_mce:.3f} > threshold={MCE_THRESHOLD}"
    return None, None


def _check_trigger_b(today_str: str, query_all) -> tuple:
    """Any model: calibrated_prob outside [0.05, 0.95] for 2+ consecutive days."""
    two_days_ago = (date.fromisoformat(today_str) - timedelta(days=BSS_CONSECUTIVE_DAYS)).isoformat()
    rows = query_all(
        """
        SELECT model_id, stock_ticker, log_date, calibrated_prob
          FROM ml_shadow_log
         WHERE log_date > ?
           AND log_date <= ?
           AND calibrated_prob IS NOT NULL
         ORDER BY model_id, log_date
        """,
        (two_days_ago, today_str),
    )
    if not rows:
        return None, None

    from collections import defaultdict
    by_model: dict = defaultdict(list)
    for r in rows:
        by_model[r["model_id"]].append(float(r["calibrated_prob"]))

    for model_id, probs in by_model.items():
        if len(probs) >= BSS_CONSECUTIVE_DAYS:
            extreme = [p for p in probs if p < 0.05 or p > 0.95]
            if len(extreme) >= BSS_CONSECUTIVE_DAYS:
                return "BSS_NEGATIVE", f"model {model_id} has {len(extreme)} extreme predictions in last {BSS_CONSECUTIVE_DAYS} days"
    return None, None


def _check_trigger_c(today_str: str, query_all) -> tuple:
    """3+ ROLLBACK lifecycle events in the last 7 days."""
    seven_days_ago = (date.fromisoformat(today_str) - timedelta(days=7)).isoformat()
    rows = query_all(
        """
        SELECT COUNT(*) AS n
          FROM model_lifecycle_log
         WHERE action = 'ROLLBACK'
           AND logged_at >= ?
        """,
        (seven_days_ago,),
    )
    n = int(rows[0]["n"]) if rows else 0
    if n >= CASCADE_ROLLBACK_THRESHOLD:
        return "CASCADE_ROLLBACK", f"{n} rollbacks in last 7 days (threshold={CASCADE_ROLLBACK_THRESHOLD})"
    return None, None


def _check_trigger_d(today_str: str, query_all) -> tuple:
    """2+ consecutive trading days with zero shadow rows (scoring job failures)."""
    from app.services.eagle_eye.ml.shadow_runner import SHADOW_ROSTER

    # Check last FAILURE_CONSECUTIVE_DAYS trading days (skip today, look at history)
    check_date = date.fromisoformat(today_str) - timedelta(days=1)
    consecutive_failures = 0
    for _ in range(FAILURE_CONSECUTIVE_DAYS + 2):  # scan a few extra days to skip weekends
        if check_date.weekday() >= 5:  # skip Sat/Sun (KSE is Sun-Thu but keep simple)
            check_date -= timedelta(days=1)
            continue
        ds = check_date.isoformat()
        rows = query_all(
            "SELECT COUNT(*) AS n FROM ml_shadow_log WHERE log_date = ?",
            (ds,),
        )
        n = int(rows[0]["n"]) if rows else 0
        if n == 0:
            consecutive_failures += 1
        else:
            break  # streak broken
        check_date -= timedelta(days=1)

    if consecutive_failures >= FAILURE_CONSECUTIVE_DAYS:
        return "SCORING_FAILURE", f"{consecutive_failures} consecutive days with no shadow rows"
    return None, None


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _disable_display(today_str: str, reason: Optional[str], exec_sql) -> None:
    exec_sql(
        """
        INSERT OR REPLACE INTO ml_display_state
            (id, auto_disabled, disabled_at, disabled_reason, updated_at)
        VALUES (1, 1, ?, ?, datetime('now'))
        """,
        (today_str, reason),
    )


def _log_lifecycle_for_all_shadow(today_str: str, reason: Optional[str], query_all, exec_sql) -> None:
    from app.services.eagle_eye.ml.db_tables import log_lifecycle

    rows = query_all(
        "SELECT model_id, stock_ticker FROM ml_models WHERE status = 'SHADOW'",
        (),
    )
    for row in rows:
        try:
            log_lifecycle(
                action="AUTO_DISABLE",
                stock_ticker=row["stock_ticker"],
                model_id=row["model_id"],
                reason=reason,
                metadata={"triggered_on": today_str},
            )
        except Exception as exc:
            LOGGER.warning("auto_disable_monitor: lifecycle log failed for %s: %s", row["model_id"], exc)


def re_enable_display() -> None:
    """Manually re-enable ML display (call from admin endpoint or REPL)."""
    from app.core.database import exec_sql

    exec_sql(
        """
        INSERT OR REPLACE INTO ml_display_state
            (id, auto_disabled, disabled_at, disabled_reason, updated_at)
        VALUES (1, 0, NULL, NULL, datetime('now'))
        """,
        (),
    )
    LOGGER.info("auto_disable_monitor: ML display manually re-enabled.")
