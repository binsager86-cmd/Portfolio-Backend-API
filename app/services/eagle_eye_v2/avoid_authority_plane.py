from __future__ import annotations

from typing import Any


AVOID_SOURCE_VERBATIM = "close < sma200 and sma200_slope < 0 and ema10 < ema30; clear via reclaim/20-session fallback"


class AvoidAuthorityPlane:
    """SMA200 avoid authority plane promoted from the v7 module-(e) harness."""

    def evaluate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        state: dict[str, Any] = {"phase": "NONE", "avoid_clear_streak": 0, "avoid_reclaim_streak": 0, "avoid_until": None}
        out: list[dict[str, Any]] = []
        for day in rows:
            payload = dict(day.get("indicator_payload") or {})
            close = float(day.get("close") or 0.0)
            ema10 = float(payload.get("ema10") or 0.0)
            ema30 = float(payload.get("ema30") or 0.0)
            sma200 = float(payload.get("sma200") or 0.0)
            sma200_slope = float(payload.get("sma200_slope") or 0.0)

            avoid_now = close < sma200 and sma200_slope < 0 and ema10 < ema30
            if avoid_now:
                if state["phase"] != "AVOID":
                    state["avoid_clear_streak"] = 0
                    state["avoid_reclaim_streak"] = 0
                state["phase"] = "AVOID"
            elif state["phase"] == "AVOID":
                state["avoid_clear_streak"] = int(state["avoid_clear_streak"] or 0) + 1
                if close > sma200:
                    state["avoid_reclaim_streak"] = int(state["avoid_reclaim_streak"] or 0) + 1
                else:
                    state["avoid_reclaim_streak"] = 0
                if int(state["avoid_reclaim_streak"] or 0) >= 2 or int(state["avoid_clear_streak"] or 0) >= 20:
                    state["phase"] = "NONE"
                    state["avoid_until"] = str(day["trade_date"])
                    state["avoid_clear_streak"] = 0
                    state["avoid_reclaim_streak"] = 0

            out.append(
                {
                    "avoid_state": state["phase"],
                    "avoid_active": state["phase"] == "AVOID",
                    "avoid_until": state["avoid_until"],
                    "avoid_clear_streak": int(state["avoid_clear_streak"] or 0),
                    "avoid_reclaim_streak": int(state["avoid_reclaim_streak"] or 0),
                    "close": close,
                    "sma200": sma200,
                    "sma200_slope": sma200_slope,
                    "ema10": ema10,
                    "ema30": ema30,
                    "avoid_entry_predicate": avoid_now,
                    "avoid_source": AVOID_SOURCE_VERBATIM,
                }
            )
        return out


def derive_avoid_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return AvoidAuthorityPlane().evaluate(rows)