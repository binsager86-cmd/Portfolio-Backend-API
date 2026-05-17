"""
ml/corporate_events.py — Phase 1: Structured corporate event features.

Ingests four event types from Boursa Kuwait disclosures:
  DIVIDEND, CAPITAL_INCREASE, AGM_EGM, RESULTS

Each event is stored in ml_corporate_events with a point-in-time
announcement_date (NOT the period it refers to).

Feature generation (Section 1.4 — Corporate event features):
  - days_until_next_dividend_ex_date
  - days_since_last_dividend
  - is_in_pre_dividend_window_N
  - days_since_last_capital_increase
  - days_until_next_results
  - is_in_pre_results_window_5d
  - is_in_results_blackout
  - days_since_last_agm

These are all computable at time T using only announcement_date < T.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-results blackout window (market convention for KSE)
# ---------------------------------------------------------------------------
RESULTS_BLACKOUT_DAYS_BEFORE = 30   # trading halt / blackout window before results
PRE_RESULTS_WINDOW_DAYS = 5         # "imminent results" flag
PRE_DIVIDEND_WINDOW_DAYS = 14       # flag stock is approaching ex-date


# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------

def ingest_event(
    *,
    stock_ticker: str,
    event_type: str,
    announcement_date: str,
    event_date: Optional[str] = None,
    ex_date: Optional[str] = None,
    amount: Optional[float] = None,
    notes: Optional[str] = None,
    raw: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Upsert one corporate event.  Silently skips duplicates (already-indexed
    events with the same (ticker, type, announcement_date)).
    """
    from app.core.database import exec_sql

    raw_json = json.dumps(raw, default=str) if raw else None
    try:
        exec_sql(
            """
            INSERT INTO ml_corporate_events
                (stock_ticker, event_type, announcement_date, event_date,
                 ex_date, amount, notes, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (stock_ticker, event_type, announcement_date) DO NOTHING
            """,
            (
                stock_ticker.upper(), event_type, announcement_date,
                event_date, ex_date, amount, notes, raw_json,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Corporate event ingest failed (%s %s %s): %s",
                       stock_ticker, event_type, announcement_date, exc)


def load_events_for_ticker(ticker: str) -> pd.DataFrame:
    """Load all corporate events for a ticker, ordered by announcement_date."""
    try:
        from app.core.database import exec_sql_fetch
        rows = exec_sql_fetch(
            """
            SELECT stock_ticker, event_type, announcement_date, event_date,
                   ex_date, amount, notes
            FROM ml_corporate_events
            WHERE stock_ticker = ?
            ORDER BY announcement_date
            """,
            (ticker.upper(),),
        )
        if not rows:
            return pd.DataFrame()
        cols = ["stock_ticker", "event_type", "announcement_date",
                "event_date", "ex_date", "amount", "notes"]
        return pd.DataFrame(rows, columns=cols)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load corporate events for %s: %s", ticker, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------

class CorporateEventFeatureBuilder:
    """
    Generates point-in-time corporate event features for each row in a
    feature DataFrame.

    All features use ONLY information available at or before the row's date,
    satisfying the leakage rule in Section 0.7.

    Parameters
    ----------
    ticker : stock ticker
    """

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker.upper()
        self._events: Optional[pd.DataFrame] = None

    def _get_events(self) -> pd.DataFrame:
        if self._events is None:
            self._events = load_events_for_ticker(self.ticker)
            if not self._events.empty:
                for col in ("announcement_date", "event_date", "ex_date"):
                    if col in self._events.columns:
                        self._events[col] = pd.to_datetime(
                            self._events[col], errors="coerce"
                        )
        return self._events

    def build_features(
        self, df: pd.DataFrame, date_col: str = "event_date"
    ) -> pd.DataFrame:
        """
        Add corporate event feature columns to ``df``.

        Features added (all CLEAN — use only past announcements):
        - days_since_last_dividend
        - days_until_next_dividend_ex_date
        - is_in_pre_dividend_window_14d
        - days_since_last_capital_increase
        - days_since_last_results_announcement
        - days_until_next_results
        - is_in_pre_results_window_5d
        - is_in_results_blackout_30d
        - days_since_last_agm
        """
        df = df.copy()
        events = self._get_events()

        if events.empty or date_col not in df.columns:
            # No events available — fill with neutral defaults
            for col in _EVENT_FEATURE_COLS:
                df[col] = _EVENT_FEATURE_DEFAULTS[col]
            return df

        # Pre-compute per-type event date lists
        div_rows = events[events["event_type"] == "DIVIDEND"].copy()
        cap_rows = events[events["event_type"] == "CAPITAL_INCREASE"].copy()
        res_rows = events[events["event_type"] == "RESULTS"].copy()
        agm_rows = events[events["event_type"] == "AGM_EGM"].copy()

        rows_out = []
        for _, row in df.iterrows():
            t = pd.Timestamp(row[date_col])
            feats = self._compute_row_features(t, div_rows, cap_rows, res_rows, agm_rows)
            rows_out.append(feats)

        feat_df = pd.DataFrame(rows_out, index=df.index)
        return pd.concat([df, feat_df], axis=1)

    def _compute_row_features(
        self,
        t: pd.Timestamp,
        div_rows: pd.DataFrame,
        cap_rows: pd.DataFrame,
        res_rows: pd.DataFrame,
        agm_rows: pd.DataFrame,
    ) -> Dict[str, Any]:
        feats: Dict[str, Any] = {}

        # ── Dividend features ─────────────────────────────────────────
        past_divs = div_rows[div_rows["announcement_date"] < t]
        # CRITICAL: future_divs must also require announcement_date < t
        # so we never see an ex_date that hasn't been publicly announced yet.
        future_divs = div_rows[
            (div_rows["announcement_date"] < t)          # announced before today
            & div_rows["ex_date"].notna()
            & (div_rows["ex_date"] > t)                  # ex-date still in future
        ]

        if past_divs.empty:
            feats["days_since_last_dividend"] = 365 * 5
        else:
            last_div = past_divs["announcement_date"].max()
            feats["days_since_last_dividend"] = (t - last_div).days

        if future_divs.empty:
            feats["days_until_next_dividend_ex_date"] = 365
            feats["is_in_pre_dividend_window_14d"] = 0
        else:
            next_ex = future_divs["ex_date"].min()
            days_to_ex = (next_ex - t).days
            feats["days_until_next_dividend_ex_date"] = int(max(0, days_to_ex))
            feats["is_in_pre_dividend_window_14d"] = int(
                0 <= days_to_ex <= PRE_DIVIDEND_WINDOW_DAYS
            )

        # ── Capital increase features ─────────────────────────────────
        past_cap = cap_rows[cap_rows["announcement_date"] < t]
        if past_cap.empty:
            feats["days_since_last_capital_increase"] = 365 * 5
        else:
            feats["days_since_last_capital_increase"] = (
                t - past_cap["announcement_date"].max()
            ).days

        # ── Results announcement features ─────────────────────────────
        past_res = res_rows[res_rows["announcement_date"] < t]
        # CRITICAL: same gate — only see future results that have been announced
        future_res = res_rows[
            (res_rows["announcement_date"] < t)          # announced before today
            & res_rows["event_date"].notna()
            & (res_rows["event_date"] > t)               # results date still in future
        ]

        if past_res.empty:
            feats["days_since_last_results_announcement"] = 365 * 5
        else:
            feats["days_since_last_results_announcement"] = (
                t - past_res["announcement_date"].max()
            ).days

        if future_res.empty:
            feats["days_until_next_results"] = 180
            feats["is_in_pre_results_window_5d"] = 0
            feats["is_in_results_blackout_30d"] = 0
        else:
            next_res_date = future_res["event_date"].min()
            days_to_res = (next_res_date - t).days
            feats["days_until_next_results"] = int(max(0, days_to_res))
            feats["is_in_pre_results_window_5d"] = int(
                0 <= days_to_res <= PRE_RESULTS_WINDOW_DAYS
            )
            feats["is_in_results_blackout_30d"] = int(
                0 <= days_to_res <= RESULTS_BLACKOUT_DAYS_BEFORE
            )

        # ── AGM/EGM features ──────────────────────────────────────────
        past_agm = agm_rows[agm_rows["announcement_date"] < t]
        if past_agm.empty:
            feats["days_since_last_agm"] = 365 * 5
        else:
            feats["days_since_last_agm"] = (
                t - past_agm["announcement_date"].max()
            ).days

        return feats


# Column names and defaults (used for padding when no events exist)
_EVENT_FEATURE_COLS = [
    "days_since_last_dividend",
    "days_until_next_dividend_ex_date",
    "is_in_pre_dividend_window_14d",
    "days_since_last_capital_increase",
    "days_since_last_results_announcement",
    "days_until_next_results",
    "is_in_pre_results_window_5d",
    "is_in_results_blackout_30d",
    "days_since_last_agm",
]

_EVENT_FEATURE_DEFAULTS: Dict[str, Any] = {
    "days_since_last_dividend": 365 * 5,
    "days_until_next_dividend_ex_date": 365,
    "is_in_pre_dividend_window_14d": 0,
    "days_since_last_capital_increase": 365 * 5,
    "days_since_last_results_announcement": 365 * 5,
    "days_until_next_results": 180,
    "is_in_pre_results_window_5d": 0,
    "is_in_results_blackout_30d": 0,
    "days_since_last_agm": 365 * 5,
}
