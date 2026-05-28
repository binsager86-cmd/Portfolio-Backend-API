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
    Uses multiclass BUY-class calibration error: compare predicted P(BUY)
    against realised BUY outcomes on labelled rows only.

B. Any individual model's BSS (Brier Skill Score) < 0 for 2+ consecutive days.
   Proxy: calibrated_prob outside [0.05, 0.95] for 2+ consecutive days
   (extreme misprediction indicator).
    Advisory-only in v8: logged for observability but does not auto-disable ML.

C. 3 or more CASCADE ROLLBACKs logged in the last 7 days.

D. 2 or more consecutive days of shadow scoring job failure (no rows written
   for a trading day).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Iterable, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

MCE_THRESHOLD = 0.30       # trigger A
MIN_SCORED_ROWS_FOR_MCE = 14
OUTCOME_LABEL_LOOKAHEAD_DAYS = 60
BSS_CONSECUTIVE_DAYS = 2   # trigger B
CASCADE_ROLLBACK_THRESHOLD = 3  # trigger C
FAILURE_CONSECUTIVE_DAYS = 2    # trigger D


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _extract_buy_probability(prediction: Any) -> Optional[float]:
    p_buy: Optional[float]

    if isinstance(prediction, dict):
        p_buy = _safe_float(prediction.get("buy"))
    elif isinstance(prediction, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(prediction, dtype=float).reshape(-1)
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.size >= 3:
            # Multiclass convention in this pipeline: [sell, hold, buy].
            p_buy = _safe_float(arr[2])
        else:
            p_buy = _safe_float(arr[0])
    else:
        p_buy = _safe_float(prediction)

    if p_buy is None:
        return None

    # Legacy compatibility: some historical rows store 0-100 scalar scores.
    if p_buy > 1.0:
        p_buy = p_buy / 100.0

    return float(np.clip(p_buy, 0.0, 1.0))


def _extract_buy_actual(actual: Any) -> Optional[float]:
    if isinstance(actual, str):
        token = actual.strip().upper()
        if token == "BUY":
            return 1.0
        if token in {"HOLD", "SELL"}:
            return 0.0

    value = _safe_float(actual)
    if value is None:
        return None

    # Native multiclass labels are {-1, 0, 1}.
    if value in (-1.0, 0.0, 1.0):
        return 1.0 if value == 1.0 else 0.0

    # Legacy realised-return fallback.
    return 1.0 if value > 0.0 else 0.0


def compute_multiclass_calibration_error(
    predictions: Iterable[Any],
    actuals: Iterable[Any],
    n_bins: int = 10,
) -> float:
    """Compute BUY-class mean calibration error in [0, 1]."""
    buy_probs: list[float] = []
    buy_actuals: list[float] = []

    for pred, actual in zip(predictions, actuals):
        p_buy = _extract_buy_probability(pred)
        y_buy = _extract_buy_actual(actual)
        if p_buy is None or y_buy is None:
            continue
        buy_probs.append(p_buy)
        buy_actuals.append(y_buy)

    if len(buy_probs) < MIN_SCORED_ROWS_FOR_MCE:
        return 0.0

    probs = np.asarray(buy_probs, dtype=float)
    actual_arr = np.asarray(buy_actuals, dtype=float)
    edges = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)

    weighted_abs_error = 0.0
    n_used = 0
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            continue

        mean_pred = float(probs[mask].mean())
        mean_actual = float(actual_arr[mask].mean())
        weighted_abs_error += abs(mean_pred - mean_actual) * count
        n_used += count

    if n_used == 0:
        return 0.0
    return float(weighted_abs_error / n_used)


def _backfill_shadow_actual_outcomes(today_str: str, query_all, exec_sql) -> int:
    """Populate ml_shadow_log.actual_outcome from v6 BUY/HOLD/SELL labels."""
    from app.services.eagle_eye.ml.labelers import detect_buy_sell_points
    from app.services.eagle_eye.store import load_ohlcv

    cutoff = (date.fromisoformat(today_str) - timedelta(days=OUTCOME_LABEL_LOOKAHEAD_DAYS)).isoformat()
    pending_rows = query_all(
        """
        SELECT id, stock_ticker, log_date
          FROM ml_shadow_log
         WHERE calibrated_prob IS NOT NULL
           AND (outcome_filled = 0 OR actual_outcome IS NULL)
           AND log_date <= ?
         ORDER BY stock_ticker, log_date
        """,
        (cutoff,),
    )
    if not pending_rows:
        return 0

    rows_by_ticker: dict[str, list[dict]] = {}
    for row in pending_rows:
        ticker = str(row["stock_ticker"]).upper()
        rows_by_ticker.setdefault(ticker, []).append(row)

    updated = 0
    for ticker, ticker_rows in rows_by_ticker.items():
        try:
            ohlcv = load_ohlcv(ticker)
        except Exception as exc:
            LOGGER.debug(
                "auto_disable_monitor: OHLCV load failed while backfilling %s: %s",
                ticker,
                exc,
            )
            continue

        if ohlcv is None or ohlcv.empty:
            continue

        try:
            labels = detect_buy_sell_points(ohlcv)
        except Exception as exc:
            LOGGER.debug(
                "auto_disable_monitor: label generation failed while backfilling %s: %s",
                ticker,
                exc,
            )
            continue

        label_by_date: dict[str, int] = {}
        for idx, value in labels.items():
            try:
                if hasattr(idx, "date"):
                    key = idx.date().isoformat()
                else:
                    key = str(idx)[:10]
                label_by_date[key] = int(value)
            except (TypeError, ValueError):
                continue

        for row in ticker_rows:
            log_date = str(row["log_date"])
            label_val = label_by_date.get(log_date)
            if label_val is None:
                continue
            exec_sql(
                """
                UPDATE ml_shadow_log
                   SET actual_outcome = ?, outcome_filled = 1
                 WHERE id = ?
                """,
                (int(label_val), int(row["id"])),
            )
            updated += 1

    return updated


def run_auto_disable_check(signal_date: str | None = None) -> dict:
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
    from app.core.database import exec_sql, query_all, query_one

    today_str = signal_date or date.today().isoformat()

    # Update labelled outcomes so trigger-A evaluates multiclass reality,
    # not stale proxy values from the pre-v6 architecture.
    try:
        filled = _backfill_shadow_actual_outcomes(today_str, query_all, exec_sql)
        if filled > 0:
            LOGGER.info("auto_disable_monitor: backfilled %d shadow outcomes", filled)
    except Exception as exc:
        LOGGER.warning("auto_disable_monitor: outcome backfill failed (non-fatal): %s", exc)

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

    trig, reason = _check_trigger_a(today_str, query_one, query_all)
    if trig:
        return trig, reason

    trig, reason = _check_trigger_b(today_str, query_all)
    if trig:
        LOGGER.warning(
            "auto_disable_monitor: advisory-only trigger %s reason=%s",
            trig,
            reason,
        )

    trig, reason = _check_trigger_c(today_str, query_all)
    if trig:
        return trig, reason

    trig, reason = _check_trigger_d(today_str, query_all)
    if trig:
        return trig, reason

    return None, None


def _check_trigger_a(today_str: str, query_one, query_all) -> tuple:
    """7-day mean multiclass BUY-class MCE > MCE_THRESHOLD."""
    scored_row_count = query_one(
        """
        SELECT COUNT(*) AS n
          FROM ml_shadow_log
         WHERE calibrated_prob IS NOT NULL
           AND outcome_filled = 1
           AND actual_outcome IS NOT NULL
        """,
        (),
    )
    if not scored_row_count or int(scored_row_count["n"]) < MIN_SCORED_ROWS_FOR_MCE:
        # Temporary bypass until enough labelled multiclass outcomes exist.
        return None, None

    seven_days_ago = (date.fromisoformat(today_str) - timedelta(days=7)).isoformat()
    rows = query_all(
        """
        SELECT calibrated_prob, actual_outcome
          FROM ml_shadow_log
         WHERE log_date > ?
           AND log_date <= ?
           AND calibrated_prob IS NOT NULL
           AND outcome_filled = 1
           AND actual_outcome IS NOT NULL
        """,
        (seven_days_ago, today_str),
    )
    if not rows or len(rows) < MIN_SCORED_ROWS_FOR_MCE:
        return None, None

    mean_mce = compute_multiclass_calibration_error(
        [r["calibrated_prob"] for r in rows],
        [r["actual_outcome"] for r in rows],
    )
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

def _disable_display(today_str: str, reason: str | None, exec_sql) -> None:
    exec_sql(
        """
        INSERT INTO ml_display_state
            (id, auto_disabled, disabled_at, disabled_reason, updated_at)
        VALUES (1, 1, ?, ?, datetime('now'))
        ON CONFLICT (id) DO UPDATE SET
            auto_disabled    = 1,
            disabled_at      = EXCLUDED.disabled_at,
            disabled_reason  = EXCLUDED.disabled_reason,
            updated_at       = CURRENT_TIMESTAMP
        """,
        (today_str, reason),
    )


def _log_lifecycle_for_all_shadow(today_str: str, reason: str | None, query_all, exec_sql) -> None:
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
        INSERT INTO ml_display_state
            (id, auto_disabled, disabled_at, disabled_reason, updated_at)
        VALUES (1, 0, NULL, NULL, datetime('now'))
        ON CONFLICT (id) DO UPDATE SET
            auto_disabled   = 0,
            disabled_at     = NULL,
            disabled_reason = NULL,
            updated_at      = CURRENT_TIMESTAMP
        """,
        (),
    )
    LOGGER.info("auto_disable_monitor: ML display manually re-enabled.")
