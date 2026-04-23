"""
Portfolio alert push notifications.

Triggered after the daily price update + snapshot save, this module sends
per-user push notifications based on the user's notification preferences:

  * ``dailyPriceUpdates`` — always send a daily summary push when enabled.
  * ``portfolioUpdates``  — send a push when the absolute portfolio
                            change_percent crosses ``PORTFOLIO_THRESHOLD_PCT``.

Per-symbol ``priceAlerts`` would require a price-history table; not
implemented in this iteration. The pref is still respected by future code.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import desc

logger = logging.getLogger(__name__)

# Significant portfolio move threshold (absolute percent).
PORTFOLIO_THRESHOLD_PCT = 2.0


def _format_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def _format_kwd(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}KD {abs(value):,.3f}"


def notify_portfolio_update(user_id: int) -> dict:
    """
    Send portfolio-update push(es) to a single user, honoring their prefs.

    Reads the latest two ``PortfolioSnapshot`` rows where ``portfolio IS NULL``
    (the per-user totals row) and computes day-over-day change. Sends:

      * a "daily update" push if ``dailyPriceUpdates`` is enabled, AND
      * a "portfolio moved" push if ``portfolioUpdates`` is enabled and the
        absolute change crosses ``PORTFOLIO_THRESHOLD_PCT``.

    Returns a dict summary; never raises (errors are logged).
    """
    from app.core.database import SessionLocal
    from app.models.push_token import PushToken
    from app.models.snapshot import PortfolioSnapshot
    from app.services.notification_prefs import get_prefs
    from app.services.push_service import send_push_notifications

    summary: dict = {"user_id": user_id, "sent": 0, "skipped": []}
    db = SessionLocal()
    try:
        # Fetch the two most recent total-portfolio snapshots for this user.
        snaps = (
            db.query(PortfolioSnapshot)
            .filter(
                PortfolioSnapshot.user_id == user_id,
                PortfolioSnapshot.portfolio.is_(None),
            )
            .order_by(desc(PortfolioSnapshot.snapshot_date))
            .limit(2)
            .all()
        )
        if not snaps:
            summary["skipped"].append("no_snapshots")
            return summary

        latest = snaps[0]
        prev = snaps[1] if len(snaps) > 1 else None

        latest_value = float(latest.portfolio_value or 0.0)

        # Prefer the persisted change_percent; fall back to recomputing from
        # the previous snapshot if the saver hasn't populated it yet.
        change_pct: Optional[float] = (
            float(latest.change_percent)
            if latest.change_percent is not None
            else None
        )
        daily_movement: Optional[float] = (
            float(latest.daily_movement)
            if latest.daily_movement is not None
            else None
        )
        if change_pct is None and prev is not None:
            prev_value = float(prev.portfolio_value or 0.0)
            if prev_value > 0:
                change_pct = ((latest_value - prev_value) / prev_value) * 100.0
                if daily_movement is None:
                    daily_movement = latest_value - prev_value

        # Pull tokens once.
        tokens = [
            t[0]
            for t in db.query(PushToken.token)
            .filter(PushToken.user_id == user_id)
            .all()
        ]
        if not tokens:
            summary["skipped"].append("no_tokens")
            return summary

        prefs = get_prefs(db, user_id)

        # ── 1. Daily update push (always, when enabled) ─────────────
        if prefs.get("dailyPriceUpdates", True):
            value_str = f"KD {latest_value:,.3f}"
            if change_pct is not None:
                title = f"📊 Portfolio: {_format_pct(change_pct)}"
                body = f"Today's value: {value_str}"
                if daily_movement is not None:
                    body += f"  ({_format_kwd(daily_movement)})"
            else:
                title = "📊 Portfolio updated"
                body = f"Today's value: {value_str}"

            data = {
                "type": "portfolio_update",
                "subtype": "daily",
                "snapshotDate": latest.snapshot_date,
                "value": latest_value,
                "changePct": change_pct,
            }
            res = send_push_notifications(tokens, title, body, data)
            summary["daily"] = res
            summary["sent"] += int(res.get("sent", 0) or 0)
        else:
            summary["skipped"].append("dailyPriceUpdates_off")

        # ── 2. Threshold-crossing alert ─────────────────────────────
        if (
            prefs.get("portfolioUpdates", True)
            and change_pct is not None
            and abs(change_pct) >= PORTFOLIO_THRESHOLD_PCT
        ):
            direction = "up" if change_pct >= 0 else "down"
            emoji = "🚀" if direction == "up" else "⚠️"
            title = f"{emoji} Portfolio {direction} {_format_pct(change_pct)}"
            body = f"Today's value: KD {latest_value:,.3f}"
            if daily_movement is not None:
                body += f"  ({_format_kwd(daily_movement)})"

            data = {
                "type": "portfolio_update",
                "subtype": "threshold",
                "snapshotDate": latest.snapshot_date,
                "value": latest_value,
                "changePct": change_pct,
                "thresholdPct": PORTFOLIO_THRESHOLD_PCT,
            }
            res = send_push_notifications(tokens, title, body, data)
            summary["threshold"] = res
            summary["sent"] += int(res.get("sent", 0) or 0)
        elif change_pct is None:
            summary["skipped"].append("no_change_pct")
        elif not prefs.get("portfolioUpdates", True):
            summary["skipped"].append("portfolioUpdates_off")
        else:
            summary["skipped"].append(
                f"below_threshold({change_pct:.2f}%<{PORTFOLIO_THRESHOLD_PCT}%)"
            )

        return summary
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("notify_portfolio_update failed for user %s: %s", user_id, e)
        return {"user_id": user_id, "sent": 0, "error": str(e)}
    finally:
        db.close()


def notify_portfolio_updates_for_users(user_ids: list[int]) -> dict:
    """Convenience: run :func:`notify_portfolio_update` for each user_id."""
    results: dict[int, dict] = {}
    total_sent = 0
    for uid in user_ids:
        try:
            r = notify_portfolio_update(uid)
            results[uid] = r
            total_sent += int(r.get("sent", 0) or 0)
        except Exception as e:  # pragma: no cover
            logger.warning("portfolio alert dispatch failed for user %s: %s", uid, e)
            results[uid] = {"user_id": uid, "sent": 0, "error": str(e)}
    return {"total_sent": total_sent, "users": results}
