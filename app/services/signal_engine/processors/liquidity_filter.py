"""Liquidity filter for the Kuwait Signal Engine.

Enforces the minimum tradability criteria for Premier Market stocks.
All thresholds are defined in kuwait_constants.py — no magic numbers here.
"""
from __future__ import annotations

import statistics
from typing import Any

from app.services.signal_engine.config.kuwait_constants import (
    PREMIER_ACTIVE_DAYS_MIN_PCT,
    PREMIER_ADTV_MIN_KD,
    PREMIER_SPREAD_PROXY_MAX,
    PREMIER_VOLUME_CONCENTRATION_MAX,
)


def is_tradable(rows: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    """Evaluate whether a stock meets Premier Market tradability criteria.

    Criteria (ALL must pass):
    1. 20-day median traded value ≥ KD 100,000
    2. Bid-ask spread proxy (high-low)/close ≤ 1.5 %
    3. Active trading days in last 30 sessions ≥ 80 %
    4. Max single-day volume concentration ≤ 40 % of 20-day sum (wash-trade filter)

    Args:
        rows: OHLCV rows sorted ascending by date.  Each row must contain at
              minimum: high, low, close, volume, value (traded_value_kd).

    Returns:
        Tuple of (passed: bool, details: dict with individual check results).
    """
    details: dict[str, Any] = {
        "adtv_20d_kd":          None,
        "adtv_fallback_used":   False,
        "spread_proxy_pct":     None,
        "active_days_30d_pct":  None,
        "volume_concentration": None,
        "pass_adtv":            False,
        "pass_spread":          False,
        "pass_active_days":     False,
        "pass_concentration":   False,
    }

    if len(rows) < 20:
        return False, details

    recent_20 = rows[-20:]
    recent_30 = rows[-30:] if len(rows) >= 30 else rows

    # ── 1. ADTV: 20-day median of traded_value_kd ─────────────────────────────
    # Primary source: r["value"] (pre-computed traded value in KWD).
    # Fallback: volume (shares) × close (fils) ÷ 1000 → KWD when value is zero
    # or missing (e.g. data pipeline gaps, delayed delivery).
    _adtv_fallback_used = False
    values_kd = []
    for r in recent_20:
        v = float(r.get("value") or 0.0)
        if v <= 0:
            _close = float(r.get("close") or 0.0)
            _vol   = float(r.get("volume") or 0.0)
            v = _vol * _close / 1000.0   # fils × shares → KWD
            if v > 0:
                _adtv_fallback_used = True
        values_kd.append(v)
    values_kd_nonzero = [v for v in values_kd if v > 0]  # exclude true zeros
    try:
        adtv = statistics.median(values_kd_nonzero) if values_kd_nonzero else 0.0
    except statistics.StatisticsError:
        adtv = 0.0
    details["adtv_20d_kd"] = round(adtv, 2)
    details["adtv_fallback_used"] = _adtv_fallback_used
    details["pass_adtv"] = adtv >= PREMIER_ADTV_MIN_KD

    # ── 2. Spread proxy: median (high-low)/close over 20 days ─────────────────
    # Capped at 10% to guard against corrupt H/L data causing absurd display values.
    _SPREAD_CAP: float = 0.10
    spreads = []
    for r in recent_20:
        h = float(r.get("high") or 0.0)
        l_val = float(r.get("low") or 0.0)
        c = float(r.get("close") or 0.0)
        if c > 0 and h >= l_val:
            spreads.append(min((h - l_val) / c, _SPREAD_CAP))
    spread_proxy = statistics.median(spreads) if spreads else 1.0
    details["spread_proxy_pct"] = round(spread_proxy * 100, 3)   # stored as percent (e.g. 1.5 = 1.5%)
    details["pass_spread"] = spread_proxy <= PREMIER_SPREAD_PROXY_MAX

    # ── 3. Active days: non-zero volume days in last 30 sessions ──────────────
    active = sum(1 for r in recent_30 if float(r.get("volume") or 0.0) > 0)
    active_pct = active / len(recent_30) if recent_30 else 0.0
    details["active_days_30d_pct"] = round(active_pct * 100, 1)
    details["pass_active_days"] = active_pct >= PREMIER_ACTIVE_DAYS_MIN_PCT

    # ── 4. Volume concentration: max single day / 20-day sum (wash-trade) ─────
    vols = [float(r.get("volume") or 0.0) for r in recent_20]
    total_vol = sum(vols)
    max_day_vol = max(vols) if vols else 0.0
    concentration = (max_day_vol / total_vol) if total_vol > 0 else 1.0
    details["volume_concentration"] = round(concentration * 100, 1)
    details["pass_concentration"] = concentration <= PREMIER_VOLUME_CONCENTRATION_MAX

    passed = (
        details["pass_adtv"]
        and details["pass_spread"]
        and details["pass_active_days"]
        and details["pass_concentration"]
    )
    return passed, details
