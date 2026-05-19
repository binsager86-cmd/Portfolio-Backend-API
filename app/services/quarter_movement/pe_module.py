"""
Module 2: Quarterly P/E Ratio Movement.

Computes quarterly highest/lowest daily P/E ratios for each calendar quarter
from 2023-Q1 to the current date.

Daily P/E formula (spec §3.1):
    daily_price_to_earnings_ratio = closing_price / trailing_twelve_months_eps

If TTM EPS is zero or negative the daily P/E is marked undefined and excluded
from quarterly aggregation (spec §3.1 and §7.2).

Input:
    daily_records: list of dicts with keys: date (ISO str), close (float).
    eps_snapshots: list of dicts {period_end_date: str (ISO), eps_value: float}.
                   Represents TTM EPS readings at discrete fiscal period end dates.
                   Applied as a step function: for each trading day the most
                   recent snapshot whose period_end_date <= trading_date is used.
    today: current system date.

Output dict keys:
    years              — list[int]
    pe_movement_table  — {year_str: {q1..q4: QuarterPECell | None}}
    pe_movement_means  — {q1..q4: QuarterPEMeanCell}
    ttm_eps            — most recent EPS snapshot value (float | None)
    eps_coverage       — "full" | "latest_only" | "none"

QuarterPECell:
    highest_pe      — max daily P/E within the quarter, 2 d.p., or None
    lowest_pe       — min daily P/E within the quarter, 2 d.p., or None
    in_progress     — True when quarter end is after today
    insufficient_data — True when no valid P/E could be computed

Known limitations:
    * If only the most recent TTM EPS is available, it is applied to all
      historical quarters as a constant (eps_coverage = "latest_only").
    * In-progress quarter cells are computed but excluded from mean calculation.
    * Quarters with fewer than two complete historical instances carry the
      reduced_sample flag in the mean cell (spec §7.3).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from app.services.quarter_movement.price_module import (
    _QUARTERS_ORDERED,
    _active_quarter_for_date,
    _enumerate_quarters_since_2023,
    _quarter_end_date,
    _quarter_start_date,
)


class QuarterlyPERatioMovementModule:
    """
    Module 2: Quarterly P/E Ratio Movement.
    Computes per-quarter highest/lowest daily P/E ratios using the step-function
    TTM EPS series provided by the caller.
    """

    def compute(
        self,
        daily_records: List[Dict],
        eps_snapshots: List[Dict],
        today: date,
        price_divisor: float = 1.0,
    ) -> Dict:
        """
        Build the quarterly P/E movement table and per-quarter-type means.

        Args:
            daily_records:  OHLCV records. Needs: date (ISO str), close (float).
            eps_snapshots:  Chronologically ordered list of TTM EPS readings.
                            Each entry: {period_end_date: str, eps_value: float}.
                            May be empty — returns "none" coverage in that case.
            today:          Current system date.
            price_divisor:  Optional price normalization divisor. Kuwait daily
                            closes are stored in fils, so callers pass 1000.0
                            when EPS is stored in KWD.

        Returns:
            Dict with keys: years, pe_movement_table, pe_movement_means,
            ttm_eps, eps_coverage.
        """
        # ── Build date-indexed close price lookup ─────────────────────────
        close_lookup: Dict[str, float] = {}
        for record in daily_records:
            date_str = record.get("date")
            close_val = record.get("close")
            if not date_str or close_val is None:
                continue
            try:
                close_lookup[date_str] = float(close_val)
            except (TypeError, ValueError):
                continue

        all_trading_dates = sorted(close_lookup.keys())

        # ── Determine EPS coverage mode and build step-function lookup ────
        # Sorted EPS snapshots: [(period_end_date_iso, eps_value), ...]
        sorted_eps_snapshots: List[Tuple[str, float]] = []
        for snap in sorted(eps_snapshots, key=lambda s: s.get("period_end_date", "")):
            period_end = snap.get("period_end_date")
            eps_val = snap.get("eps_value")
            if period_end and eps_val is not None:
                try:
                    sorted_eps_snapshots.append((period_end, float(eps_val)))
                except (TypeError, ValueError):
                    continue

        if not sorted_eps_snapshots:
            eps_coverage = "none"
            most_recent_ttm_eps: Optional[float] = None
        elif len(sorted_eps_snapshots) == 1:
            eps_coverage = "latest_only"
            most_recent_ttm_eps = sorted_eps_snapshots[-1][1]
        else:
            eps_coverage = "full"
            most_recent_ttm_eps = sorted_eps_snapshots[-1][1]

        def resolve_ttm_eps_for_date(trading_date_iso: str) -> Optional[float]:
            """
            Return the most recent TTM EPS whose period_end_date <= trading_date.
            Falls back to the earliest snapshot if none precede the date.
            """
            if not sorted_eps_snapshots:
                return None
            result: Optional[float] = None
            for period_end_iso, eps_val in sorted_eps_snapshots:
                if period_end_iso <= trading_date_iso:
                    result = eps_val
                else:
                    break
            # If no snapshot precedes the date, use the earliest one available
            if result is None:
                result = sorted_eps_snapshots[0][1]
            return result

        # ── Iterate all quarters ──────────────────────────────────────────
        quarters = _enumerate_quarters_since_2023(today)
        active_year, _ = _active_quarter_for_date(today)
        years_in_range = sorted({y for y, _ in quarters})

        pe_movement_table: Dict[str, Dict[str, Optional[Dict]]] = {
            str(year): {q: None for q in _QUARTERS_ORDERED}
            for year in years_in_range
        }

        for year, quarter_key in quarters:
            if eps_coverage == "none":
                pe_movement_table[str(year)][quarter_key] = {
                    "highest_pe": None,
                    "lowest_pe": None,
                    "in_progress": False,
                    "insufficient_data": True,
                }
                continue

            quarter_start = _quarter_start_date(year, quarter_key)
            quarter_end = _quarter_end_date(year, quarter_key)
            is_in_progress = quarter_end > today

            quarter_start_iso = quarter_start.isoformat()
            effective_end_iso = min(quarter_end.isoformat(), today.isoformat())

            # ── Compute daily P/E values within the quarter ───────────────
            quarterly_pe_values: List[float] = []

            for trading_date in all_trading_dates:
                if not (quarter_start_iso <= trading_date <= effective_end_iso):
                    continue

                closing_price = close_lookup[trading_date]
                trailing_twelve_months_eps = resolve_ttm_eps_for_date(trading_date)

                # §3.1: exclude zero/negative EPS
                if trailing_twelve_months_eps is None or trailing_twelve_months_eps <= 0:
                    continue

                normalized_closing_price = (
                    closing_price / price_divisor
                    if price_divisor and price_divisor > 0
                    else closing_price
                )
                daily_price_to_earnings_ratio = normalized_closing_price / trailing_twelve_months_eps
                quarterly_pe_values.append(daily_price_to_earnings_ratio)

            if not quarterly_pe_values:
                pe_movement_table[str(year)][quarter_key] = {
                    "highest_pe": None,
                    "lowest_pe": None,
                    "in_progress": is_in_progress,
                    "insufficient_data": True,
                }
                continue

            # §3.2: quarterly highest and lowest P/E
            quarterly_highest_price_to_earnings_ratio = round(max(quarterly_pe_values), 2)
            quarterly_lowest_price_to_earnings_ratio = round(min(quarterly_pe_values), 2)

            pe_movement_table[str(year)][quarter_key] = {
                "highest_pe": quarterly_highest_price_to_earnings_ratio,
                "lowest_pe": quarterly_lowest_price_to_earnings_ratio,
                "in_progress": is_in_progress,
                "insufficient_data": False,
            }

        # ── Arithmetic means per quarter type (spec §3.3) ─────────────────
        # Exclude in-progress and insufficient cells.
        pe_movement_means: Dict[str, Dict] = {}
        for q_key in _QUARTERS_ORDERED:
            highest_pe_values: List[float] = []
            lowest_pe_values: List[float] = []

            for year in years_in_range:
                cell = pe_movement_table[str(year)].get(q_key)
                if cell is None or cell.get("insufficient_data") or cell.get("in_progress"):
                    continue
                h_pe = cell.get("highest_pe")
                l_pe = cell.get("lowest_pe")
                if h_pe is not None:
                    highest_pe_values.append(h_pe)
                if l_pe is not None:
                    lowest_pe_values.append(l_pe)

            historical_arithmetic_mean_highest_price_to_earnings_ratio = (
                round(sum(highest_pe_values) / len(highest_pe_values), 2)
                if highest_pe_values else None
            )
            historical_arithmetic_mean_lowest_price_to_earnings_ratio = (
                round(sum(lowest_pe_values) / len(lowest_pe_values), 2)
                if lowest_pe_values else None
            )
            reduced_sample = len(highest_pe_values) < 2

            pe_movement_means[q_key] = {
                "highest_pe_mean": historical_arithmetic_mean_highest_price_to_earnings_ratio,
                "lowest_pe_mean": historical_arithmetic_mean_lowest_price_to_earnings_ratio,
                "reduced_sample": reduced_sample,
                "sample_count": len(highest_pe_values),
            }

        return {
            "years": years_in_range,
            "pe_movement_table": pe_movement_table,
            "pe_movement_means": pe_movement_means,
            "ttm_eps": most_recent_ttm_eps,
            "eps_coverage": eps_coverage,
        }

    def compute_from_pe_series(
        self,
        pe_series: Dict,
        today: date,
    ) -> Dict:
        """
        Build the quarterly P/E movement table directly from a pre-computed
        daily PE series (e.g. from TickerChart's local FlatFiles cache).

        Args:
            pe_series: dict mapping datetime.date → daily PE close value (float).
                       Typically sourced from `fetch_pe_from_flatfiles()`.
            today:     Current system date.

        Returns:
            Same structure as `compute()`, with eps_coverage="flatfiles" and
            ttm_eps set to the most recently available PE value.
            Quarters beyond the last date in pe_series will be marked
            insufficient_data=True.
        """
        quarters = _enumerate_quarters_since_2023(today)
        years_in_range = sorted({y for y, _ in quarters})

        pe_movement_table: Dict[str, Dict[str, Optional[Dict]]] = {
            str(year): {q: None for q in _QUARTERS_ORDERED}
            for year in years_in_range
        }

        last_available_date: Optional[date] = max(pe_series.keys()) if pe_series else None

        for year, quarter_key in quarters:
            quarter_start = _quarter_start_date(year, quarter_key)
            quarter_end = _quarter_end_date(year, quarter_key)
            is_in_progress = quarter_end > today

            effective_end = min(quarter_end, today)

            # Quarter entirely beyond our PE data range
            if last_available_date is None or quarter_start > last_available_date:
                pe_movement_table[str(year)][quarter_key] = {
                    "highest_pe": None,
                    "lowest_pe": None,
                    "in_progress": is_in_progress,
                    "insufficient_data": True,
                }
                continue

            quarterly_high_values: List[float] = []
            quarterly_low_values: List[float] = []

            for d, pe_val in pe_series.items():
                if not (quarter_start <= d <= effective_end):
                    continue
                # pe_val is (high_pe, low_pe) tuple — flatfile dates use (close, close)
                if isinstance(pe_val, tuple):
                    h_pe, l_pe = float(pe_val[0]), float(pe_val[1])
                else:
                    h_pe = l_pe = float(pe_val)
                quarterly_high_values.append(h_pe)
                quarterly_low_values.append(l_pe)

            if not quarterly_high_values:
                pe_movement_table[str(year)][quarter_key] = {
                    "highest_pe": None,
                    "lowest_pe": None,
                    "in_progress": is_in_progress,
                    "insufficient_data": True,
                }
                continue

            pe_movement_table[str(year)][quarter_key] = {
                "highest_pe": round(max(quarterly_high_values), 2),
                "lowest_pe": round(min(quarterly_low_values), 2),
                "in_progress": is_in_progress,
                "insufficient_data": False,
            }

        # ── Arithmetic means per quarter type ─────────────────────────
        pe_movement_means: Dict[str, Dict] = {}
        for q_key in _QUARTERS_ORDERED:
            highest_values: List[float] = []
            lowest_values: List[float] = []
            for year in years_in_range:
                cell = pe_movement_table[str(year)].get(q_key)
                if cell is None or cell.get("insufficient_data") or cell.get("in_progress"):
                    continue
                if cell.get("highest_pe") is not None:
                    highest_values.append(cell["highest_pe"])
                if cell.get("lowest_pe") is not None:
                    lowest_values.append(cell["lowest_pe"])

            pe_movement_means[q_key] = {
                "highest_pe_mean": (
                    round(sum(highest_values) / len(highest_values), 2) if highest_values else None
                ),
                "lowest_pe_mean": (
                    round(sum(lowest_values) / len(lowest_values), 2) if lowest_values else None
                ),
                "reduced_sample": len(highest_values) < 2,
                "sample_count": len(highest_values),
            }

        return {
            "years": years_in_range,
            "pe_movement_table": pe_movement_table,
            "pe_movement_means": pe_movement_means,
            "ttm_eps": None,   # EPS must be injected by caller after this call
            "eps_coverage": "flatfiles",
        }
