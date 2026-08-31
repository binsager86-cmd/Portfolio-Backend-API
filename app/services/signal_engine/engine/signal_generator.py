"""Kuwait Signal Engine — main orchestration entry point.

Provides generate_kuwait_signal() which takes pre-computed OHLCV + indicator
rows (from TickerChart → attach_indicators) and produces the canonical signal
JSON in one call.

Usage:
    from app.services.signal_engine.engine.signal_generator import generate_kuwait_signal

    signal = generate_kuwait_signal(
        rows=rows_with_indicators,
        stock_code="NBK",
        segment="PREMIER",
        account_equity=100_000.0,
        delay_hours=0,
    )
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from app.services.signal_engine.config.kuwait_constants import (
    align_to_tick,
    CIRCUIT_BUFFER_PCT,
    CIRCUIT_LOWER_PCT,
    CIRCUIT_UPPER_PCT,
    PREMIER_ADTV_MIN_KD,
)
from app.services.signal_engine.config.model_params import (
    BASE_WEIGHTS,
    MIN_INDICATOR_COVERAGE,
    MIN_BARS_FOR_SIGNAL,
    MIN_REQUIRED_DATA_QUALITY,
    REGIME_BULL,
    SIGNAL_MAX_TOTAL_SELL,
    SIGNAL_MIN_P_TP1_BUY,
    SIGNAL_MIN_P_TP1_SELL,
    SIGNAL_MIN_P_TP1_STRONG_BUY,
    SIGNAL_MIN_RR,
    SIGNAL_MIN_TOTAL_SCORE,
    SIGNAL_MIN_TREND_RAW_PCT,
    SIGNAL_MIN_VOLFLOW_RAW_PCT,
    SIGNAL_STRONG_BUY_SCORE,
)
from app.services.signal_engine.engine.output_formatter import classify_setup_type, format_signal
from app.services.signal_engine.engine.probability_calibrator import calibrate_probabilities
from app.services.signal_engine.models.regime.hmm_regime_detector import predict_regime
from app.services.signal_engine.models.regime.transition_monitor import (
    detect_transition_alerts,
    get_regime_weight_adjustment,
)
from app.services.signal_engine.models.risk.confluence_decay import adjust_confidence_for_delay
from app.services.signal_engine.models.risk.cvar_calculator import calculate_cvar
from app.services.signal_engine.models.risk.position_sizer import calculate_position_size
from app.services.signal_engine.config.risk_config import TC_COMMISSION, TC_SLIPPAGE_MAIN, TC_SLIPPAGE_PREMIER
from app.services.signal_engine.models.technical.entry_trigger import evaluate_entry_trigger
from app.services.signal_engine.models.technical.four_score_engine import compute_all_four_scores
from app.services.signal_engine.models.technical.momentum_score import compute_momentum_score
from app.services.signal_engine.models.technical.support_resistance import (
    compute_entry_stop_tp,
    compute_sr_score,
    compute_tp_methods,
)
from app.services.signal_engine.processors.sr_engine import calculate_full_sr_levels
from app.services.signal_engine.processors.volume_profile import calculate_volume_profile
from app.services.signal_engine.models.technical.trend_score import compute_trend_score
from app.services.signal_engine.models.technical.volume_flow_score import compute_volume_flow_score
from app.services.signal_engine.processors.auction_proxy import (
    auction_confidence_adjustment,
    calculate_auction_intensity,
)
from app.services.signal_engine.processors.liquidity_filter import is_tradable

logger = logging.getLogger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _compute_data_quality_score(
    rows: list[dict[str, Any]],
    *,
    min_bars_required: int,
) -> tuple[float, list[str]]:
    """Score market-data readiness from observable runtime inputs only.

    The score falls when:
      - history coverage is short,
      - key indicators are missing on the latest bar,
      - OHLCV/value fields are missing or invalid,
      - corporate-actions metadata is absent,
      - market/session metadata is absent.
    """
    if not rows:
        return 0.0, ["no_rows"]

    latest = rows[-1]
    reasons: list[str] = []
    score = 100.0

    history_ratio = _clamp(len(rows) / float(max(1, min_bars_required)), 0.0, 1.0)
    history_penalty = (1.0 - history_ratio) * 35.0
    if history_penalty > 0:
        reasons.append("short_history")
        score -= history_penalty

    required_price_fields = ("open", "high", "low", "close", "volume", "value")
    missing_price_fields = [field for field in required_price_fields if latest.get(field) in (None, "")]
    if missing_price_fields:
        reasons.append("missing_market_fields")
        score -= min(20.0, 4.0 * len(missing_price_fields))

    if float(latest.get("close") or 0.0) <= 0.0:
        reasons.append("non_positive_close")
        score -= 15.0

    if float(latest.get("volume") or 0.0) <= 0.0:
        reasons.append("non_positive_volume")
        score -= 10.0

    invalid_ohlc_bars = 0
    for row in rows:
        try:
            open_px = float(row.get("open"))
            high_px = float(row.get("high"))
            low_px = float(row.get("low"))
            close_px = float(row.get("close"))
            volume = float(row.get("volume"))
            if (
                not all(math.isfinite(v) for v in (open_px, high_px, low_px, close_px, volume))
                or min(open_px, high_px, low_px, close_px) <= 0
                or high_px < max(open_px, close_px)
                or low_px > min(open_px, close_px)
                or high_px < low_px
                or volume < 0
            ):
                invalid_ohlc_bars += 1
        except (TypeError, ValueError):
            invalid_ohlc_bars += 1
    if invalid_ohlc_bars:
        reasons.append(f"invalid_ohlc_bars_{invalid_ohlc_bars}")
        score = min(score - min(30.0, invalid_ohlc_bars * 5.0), MIN_REQUIRED_DATA_QUALITY - 0.1)

    indicator_keys = (
        "ema_20",
        "ema_50",
        "sma_200",
        "adx_14",
        "rsi_14",
        "macd",
        "macd_signal",
        "atr_14",
        "cmf_20",
    )
    missing_indicators = [key for key in indicator_keys if latest.get(key) in (None, "")]
    if missing_indicators:
        reasons.extend(f"missing_indicator_{key}" for key in missing_indicators)
        score -= min(35.0, 4.0 * len(missing_indicators))

    indicator_coverage = sum(latest.get(key) not in (None, "") for key in indicator_keys) / len(indicator_keys)
    if indicator_coverage < MIN_INDICATOR_COVERAGE:
        reasons.append(f"indicator_coverage_below_{MIN_INDICATOR_COVERAGE:.0%}")
        score = min(score, MIN_REQUIRED_DATA_QUALITY - 0.1)

    if latest.get("corporate_actions") is None:
        reasons.append("corporate_actions_unavailable")
        score -= 5.0

    if latest.get("market") is None and latest.get("exchange") is None:
        reasons.append("market_context_missing")
        score -= 5.0

    return round(_clamp(score, 0.0, 100.0), 1), reasons


def _compute_recommendation_contract(
    *,
    final_signal: str,
    direction_score: float,
    setup_quality_score: float,
    timing_score: float,
    data_quality_score: float,
    expected_value_r: float | None,
    entry_trigger_action: str,
    neutral_reason: str | None,
    probability_status: str | None = None,
) -> dict[str, Any]:
    """Builds the explicit direction/quality/timing/action recommendation matrix.

    Decision matrix:
      1) INSUFFICIENT_DATA when data quality is too low or the engine reports it.
      2) AVOID for weak-quality or negative-EV setups.
      3) Direction bucket from signed direction_score:
         LONG >= +15, SHORT <= -15, else NEUTRAL.
      4) LONG path: STRONG_BUY / BUY when quality + timing + trigger agree;
         otherwise WATCH_LONG.
      5) SHORT path: SELL when quality + timing + trigger agree; otherwise WATCH_SHORT.
      6) HOLD for neutral direction with adequate data quality.
    """
    normalized_signal = str(final_signal or "NEUTRAL").upper()
    trigger_action = str(entry_trigger_action or "HOLD").upper()
    reason = str(neutral_reason or "").lower()

    direction = "NEUTRAL"
    if direction_score >= 15.0:
        direction = "LONG"
    elif direction_score <= -15.0:
        direction = "SHORT"

    recommendation = "HOLD"
    actionable = False

    if data_quality_score < 35.0 or "insufficient_data" in reason:
        recommendation = "INSUFFICIENT_DATA"
        actionable = False
    elif setup_quality_score < 40.0 or (expected_value_r is not None and expected_value_r < 0.0):
        recommendation = "AVOID"
        actionable = False
    elif probability_status is not None and probability_status != "CALIBRATED":
        recommendation = "WATCH_LONG" if direction == "LONG" else "WATCH_SHORT" if direction == "SHORT" else "HOLD"
        actionable = False
    elif direction == "LONG":
        if (
            normalized_signal == "STRONG_BUY"
            and setup_quality_score >= 78.0
            and timing_score >= 72.0
            and trigger_action == "ENTER"
            and (expected_value_r is None or expected_value_r >= 0.35)
        ):
            recommendation = "STRONG_BUY"
            actionable = True
        elif (
            normalized_signal in {"BUY", "STRONG_BUY"}
            and setup_quality_score >= 62.0
            and timing_score >= 58.0
            and trigger_action in {"ENTER", "WATCH"}
            and (expected_value_r is None or expected_value_r >= 0.10)
        ):
            recommendation = "BUY"
            actionable = True
        else:
            recommendation = "WATCH_LONG"
            actionable = False
    elif direction == "SHORT":
        if (
            normalized_signal == "SELL"
            and setup_quality_score >= 62.0
            and timing_score >= 58.0
            and trigger_action in {"ENTER", "WATCH", "HOLD"}
            and (expected_value_r is None or expected_value_r >= 0.10)
        ):
            recommendation = "SELL"
            actionable = True
        else:
            recommendation = "WATCH_SHORT"
            actionable = False
    else:
        recommendation = "HOLD"
        actionable = False

    return {
        "direction": direction,
        "direction_score": int(round(_clamp(direction_score, -100.0, 100.0))),
        "setup_quality_score": int(round(_clamp(setup_quality_score, 0.0, 100.0))),
        "timing_score": int(round(_clamp(timing_score, 0.0, 100.0))),
        "data_quality_score": round(_clamp(data_quality_score, 0.0, 100.0), 1),
        "expected_value_r": None if expected_value_r is None else round(float(expected_value_r), 3),
        "recommendation": recommendation,
        "actionable": bool(actionable),
    }


def _build_indicator_breakdown(
    trend_details: dict[str, Any] | None,
    momentum_details: dict[str, Any] | None,
    volume_details: dict[str, Any] | None,
    sr_details: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a frontend-friendly indicator breakdown payload."""
    if not any((trend_details, momentum_details, volume_details, sr_details)):
        return None

    trend = None
    if trend_details:
        multipliers = trend_details.get("multipliers") or {
            "efficiency_ratio": trend_details.get("er_mult", 1.0),
            "trend_age": trend_details.get("age_mult", 1.0),
            "ema_stretch": trend_details.get("stretch_mult", 1.0),
            "sector_lead_lag": trend_details.get("sector_mult", 1.0),
        }
        trend = {
            "base_raw": trend_details.get("base_raw", trend_details.get("raw_score", 0)),
            "final_adjusted": trend_details.get("final_adjusted", trend_details.get("raw_score", 0)),
            "adjustment_factor": trend_details.get("adjustment_factor", 1.0),
            "multipliers": multipliers,
            "ema_pts": trend_details.get("ema_pts", trend_details.get("ema_alignment_pts")),
            "ema_desc": trend_details.get("ema_desc", trend_details.get("ema_alignment_desc")),
            "adx_pts": trend_details.get("adx_pts"),
            "adx_desc": trend_details.get("adx_desc"),
            "swing_pts": trend_details.get("swing_pts", trend_details.get("swing_structure_pts")),
            "swing_desc": trend_details.get("swing_desc", trend_details.get("swing_structure_desc")),
            "raw_score": trend_details.get("raw_score"),
        }

    return {
        "trend": trend,
        "momentum": momentum_details,
        "volume_flow": volume_details,
        "support_resistance": sr_details,
    }


def _make_blocked_four_scores(
    rows: list[dict[str, Any]],
    adtv_kwd: float,
    spread_pct: float,
) -> dict[str, Any]:
    """Return a deterministic no-trade four-score profile."""
    del rows, adtv_kwd, spread_pct
    return {
        "potential": {"score": 0, "tier": "Strong Sell", "description": "blocked"},
        "timing": {"score": 0, "tier": "Strong Sell", "description": "blocked"},
        "risk": {"score": 0, "level": "High Risk", "tier": "Sell", "description": "blocked"},
        "overall": {
            "base_score": 0,
            "score": 0,
            "tier": "Strong Sell",
            "description": "blocked",
            "risk_multiplier": 0.0,
            "adjustment_factor": 0.0,
        },
        "position_action": {"action": "NO_TRADE", "max_position_pct": 0.0},
    }


def _apply_regime_weights(
    base_weights: dict[str, float],
    regime: str,
    liquidity_percentile: float,
) -> dict[str, float]:
    """Return effective weights after regime and liquidity adjustments.

    Weights are then re-normalised so they still sum to 1.0.
    """
    adjustments = get_regime_weight_adjustment(regime)
    weights: dict[str, float] = {}
    for k, base_w in base_weights.items():
        adj = adjustments.get(k, 1.0)
        # Spec: illiquid stocks → stronger volume filter, weaker momentum
        if liquidity_percentile < 40.0:
            if k == "volume_flow":
                adj *= 1.4
            elif k == "momentum":
                adj *= 0.7
        weights[k] = base_w * adj

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def _circuit_breaker_alerts(rows: list[dict[str, Any]], prev_close: float) -> list[str]:
    """Check if current price is near circuit-breaker limits."""
    if not rows or prev_close <= 0:
        return []
    close = float(rows[-1].get("close") or 0.0)
    upper = prev_close * (1.0 + CIRCUIT_UPPER_PCT)
    lower = prev_close * (1.0 + CIRCUIT_LOWER_PCT)
    alerts: list[str] = []
    gap_to_upper = (upper - close) / close if close > 0 else 1.0
    gap_to_lower = (close - lower) / close if close > 0 else 1.0
    if gap_to_upper <= CIRCUIT_BUFFER_PCT:
        alerts.append(f"WARNING: Price within {gap_to_upper*100:.1f}% of upper circuit-breaker limit (+10%)")
    if gap_to_lower <= CIRCUIT_BUFFER_PCT:
        alerts.append(f"WARNING: Price within {gap_to_lower*100:.1f}% of lower circuit-breaker limit (-5%)")
    return alerts


def _liquidity_percentile(adtv_kd: float | None) -> float:
    """Map ADTV to a rough liquidity percentile (0-100) for weight adjustment."""
    if not adtv_kd or adtv_kd <= 0:
        return 20.0
    if adtv_kd >= 1_000_000:
        return 95.0
    if adtv_kd >= 500_000:
        return 80.0
    if adtv_kd >= 200_000:
        return 60.0
    if adtv_kd >= 100_000:
        return 40.0
    return 15.0


def _evaluate_short_entry_trigger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Short/exit timing trigger parallel to the long entry trigger.

    SELL semantics in this service are EXIT-oriented for long-only books.
    This trigger still describes bearish timing quality for consistency.
    """
    if len(rows) < 5:
        return {
            "action": "HOLD",
            "trigger": "none",
            "pullback": {"triggered": False, "reason": "insufficient_data"},
            "breakout": {"triggered": False, "reason": "insufficient_data"},
            "accumulation": {"state": "absent", "obv_slope_pct": None, "cmf": None},
            "short_breakdown": {"triggered": False, "reason": "insufficient_data"},
            "failed_rally": {"triggered": False, "reason": "insufficient_data"},
            "distribution": {"state": "absent", "obv_slope_pct": None, "cmf": None},
        }

    last = rows[-1]
    prev = rows[-2]
    close = float(last.get("close") or 0.0)
    open_px = float(last.get("open") or 0.0)
    low_prev = float(prev.get("low") or close)
    high_prev = float(prev.get("high") or close)
    ema20 = float(last.get("ema_20") or close)
    cmf = float(last.get("cmf_20") or 0.0)
    rsi = float(last.get("rsi_14") or 50.0)

    vol_window = [float(r.get("volume") or 0.0) for r in rows[-21:-1]]
    avg_vol = sum(vol_window) / len(vol_window) if vol_window else 0.0
    vol_mult = (float(last.get("volume") or 0.0) / avg_vol) if avg_vol > 0 else 0.0

    bearish_candle = close < open_px
    short_breakdown = close < low_prev and vol_mult >= 1.2 and bearish_candle
    failed_rally = high_prev >= ema20 and close < ema20 and bearish_candle
    distribution_active = cmf < -0.05 or rsi < 45.0

    if short_breakdown:
        action = "ENTER"
        trigger = "breakdown"
    elif failed_rally or distribution_active:
        action = "WATCH"
        trigger = "failed_rally" if failed_rally else "distribution"
    else:
        action = "HOLD"
        trigger = "none"

    return {
        "action": action,
        "trigger": trigger,
        "pullback": {"triggered": False, "reason": "long_only_trigger"},
        "breakout": {"triggered": False, "reason": "long_only_trigger"},
        "accumulation": {"state": "absent", "obv_slope_pct": None, "cmf": None},
        "short_breakdown": {
            "triggered": short_breakdown,
            "reason": "breakdown_confirmed" if short_breakdown else "not_confirmed",
        },
        "failed_rally": {
            "triggered": failed_rally,
            "reason": "rejection_at_ema20" if failed_rally else "not_confirmed",
        },
        "distribution": {
            "state": "active" if distribution_active else "absent",
            "obv_slope_pct": None,
            "cmf": round(cmf, 4),
        },
    }


def _component_contract(
    *,
    direction: float,
    quality: float,
    confidence: float,
    available: bool,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "direction": round(_clamp(direction, -1.0, 1.0), 3),
        "quality": round(_clamp(quality, 0.0, 1.0), 3),
        "confidence": round(_clamp(confidence, 0.0, 1.0), 3),
        "available": bool(available),
        "details": details,
    }


async def generate_kuwait_signal(
    rows: list[dict[str, Any]],
    stock_code: str,
    segment: str = "PREMIER",
    account_equity: float = 100_000.0,
    delay_hours: int = 0,
    recent_performance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a full multi-factor trade signal for a Kuwait Premier Market stock.

    Args:
        rows:               OHLCV rows with attached TA-Lib indicators, sorted
                            ascending by date.  Minimum MIN_BARS_FOR_SIGNAL bars.
        stock_code:         Stock ticker (e.g. "NBK").
        segment:            "PREMIER" | "MAIN" | "AUCTION".
        account_equity:     Account size in KWD for position sizing.
        delay_hours:        Hours since signal generation (for confidence decay).
        recent_performance: Optional {"wins": int, "total": int} for Bayesian
                            calibration update.

    Returns:
        Full signal dict matching the spec JSON schema.
        Signal is "NEUTRAL" when thresholds are not met or liquidity fails.
    """
    data_as_of = rows[-1]["date"] if rows else "unknown"

    # ── Guard: minimum data ───────────────────────────────────────────────────
    data_quality_score, data_quality_reasons = _compute_data_quality_score(
        rows,
        min_bars_required=MIN_BARS_FOR_SIGNAL,
    )
    if len(rows) < MIN_BARS_FOR_SIGNAL:
        return _neutral_signal(
            stock_code,
            segment,
            data_as_of,
            "insufficient_data",
            data_quality_score=data_quality_score,
            data_quality_reasons=data_quality_reasons,
        )

    # ── 1. Liquidity filter ───────────────────────────────────────────────────
    liquidity_passed, liq_details = is_tradable(rows)
    adtv_kd = liq_details.get("adtv_20d_kd") or 0.0
    liq_pct = _liquidity_percentile(adtv_kd)

    alerts: list[str] = []
    if data_quality_reasons:
        alerts.append("DATA QUALITY WARNING: " + ", ".join(data_quality_reasons))
    if not liquidity_passed:
        alerts.append("LIQUIDITY FAIL: stock does not meet Premier Market tradability criteria")

    # ── 2. Auction intensity ──────────────────────────────────────────────────
    auction_intensity = calculate_auction_intensity(rows)
    auction_adj = auction_confidence_adjustment(auction_intensity)
    alerts.append("AUCTION ANALYSIS UNAVAILABLE: daily bars contain no intraday auction volume")

    # ── 3. Regime detection ───────────────────────────────────────────────────
    regime_result = predict_regime(rows)
    regime = regime_result.get("current_regime", "Neutral_Chop")
    regime_confidence = regime_result.get("regime_confidence") or 0.5
    regime_alerts = detect_transition_alerts(regime_result)
    alerts.extend(regime_alerts)

    # ── 4. Technical scoring ──────────────────────────────────────────────────
    trend_raw, trend_details = compute_trend_score(rows)
    # Base trend score before directional context multipliers (ER, age, stretch, sector).
    # Used to produce the unadjusted combined score so the frontend can show both.
    trend_base_raw = trend_details.get("base_raw", trend_raw)
    momentum_raw, momentum_details = compute_momentum_score(rows)
    volume_raw, volume_details = compute_volume_flow_score(rows, auction_intensity)
    sr_raw, sr_details, support_levels, resistance_levels = compute_sr_score(rows)

    nearest_support = support_levels[0] if support_levels else None
    nearest_resistance = resistance_levels[0] if resistance_levels else None
    spread_pct = 0.0
    if rows and rows[-1].get("close"):
        close_px = float(rows[-1].get("close") or 0.0)
        hi = float(rows[-1].get("high") or close_px)
        lo = float(rows[-1].get("low") or close_px)
        if close_px > 0:
            spread_pct = ((hi - lo) / close_px) * 100.0
    # ── 4b. Volume profile ────────────────────────────────────────────────────
    try:
        volume_profile = calculate_volume_profile(rows)
    except Exception:  # noqa: BLE001
        logger.exception("Volume profile calculation failed")
        volume_profile = {}
    # ── 5. Determine provisional direction (explicit BUY / SELL / NEUTRAL only) ─
    last = rows[-1]
    close_now = float(last.get("close") or 0.0)
    ema20 = float(last.get("ema_20") or close_now)
    ema50 = float(last.get("ema_50") or close_now)
    rsi_now = float(last.get("rsi_14") or 50.0)
    cmf_now = float(last.get("cmf_20") or 0.0)

    bullish_structure = trend_raw >= 58 and ema20 >= ema50
    bearish_structure = trend_raw <= 42 and ema20 <= ema50
    positive_momentum = momentum_raw >= 52 and rsi_now >= 50.0
    negative_momentum = momentum_raw <= 48 and rsi_now <= 50.0
    accumulation_confirm = volume_raw >= 52 and cmf_now >= 0.0
    distribution_confirm = volume_raw <= 48 and cmf_now <= 0.0

    long_geometry_ok = bool(
        nearest_support is not None
        and (
            nearest_resistance is None
            or (nearest_resistance - close_now) / max(close_now, 1.0) >= 0.015
        )
    )
    short_geometry_ok = bool(
        nearest_resistance is not None
        and (
            nearest_support is None
            or (close_now - nearest_support) / max(close_now, 1.0) >= 0.015
        )
    )

    long_direction_ready = bullish_structure and positive_momentum and accumulation_confirm and long_geometry_ok
    short_direction_ready = bearish_structure and negative_momentum and distribution_confirm and short_geometry_ok

    if long_direction_ready and not short_direction_ready:
        provisional_direction = "BUY"
    elif short_direction_ready and not long_direction_ready:
        provisional_direction = "SELL"
    else:
        provisional_direction = "NEUTRAL"

    # ── 6. Scenario levels (diagnostic) and executable level candidate ───────
    if provisional_direction in {"BUY", "SELL"}:
        levels = compute_entry_stop_tp(rows, provisional_direction, nearest_resistance, nearest_support)
    else:
        levels = compute_entry_stop_tp(rows, "NEUTRAL", nearest_resistance, nearest_support)

    scenario_levels = {
        "direction": "LONG" if provisional_direction == "BUY" else "SHORT" if provisional_direction == "SELL" else "NEUTRAL",
        "entry_zone_fils": [levels.get("entry_low"), levels.get("entry_high")],
        "stop_loss_fils": levels.get("stop_loss"),
        "tp1_fils": levels.get("tp1"),
        "tp2_fils": levels.get("tp2"),
        "tp3_fils": levels.get("tp3"),
        "risk_reward_ratio": levels.get("risk_reward_ratio"),
        "assumptions": {
            "bullish_structure": bullish_structure,
            "bearish_structure": bearish_structure,
            "positive_momentum": positive_momentum,
            "negative_momentum": negative_momentum,
            "accumulation_confirm": accumulation_confirm,
            "distribution_confirm": distribution_confirm,
            "long_geometry_ok": long_geometry_ok,
            "short_geometry_ok": short_geometry_ok,
        },
    }

    selected_avwap = (
        sr_details.get("anchored_vwap_long")
        if provisional_direction == "BUY"
        else sr_details.get("anchored_vwap_short")
        if provisional_direction == "SELL"
        else None
    )
    sr_details["anchored_vwap"] = selected_avwap

    rr = levels.get("risk_reward_ratio") or 0.0

    # ── 6b. Rich S/R map + multi-method TP ───────────────────────────────────
    entry_mid = levels.get("entry_mid") or float(rows[-1].get("close") or 0.0)
    try:
        rich_sr = calculate_full_sr_levels(rows, volume_profile, entry_mid)
    except Exception:  # noqa: BLE001
        logger.exception("Rich S/R calculation failed")
        rich_sr = {"resistance": [], "support": [], "nearest_resistance": None, "nearest_support": None}

    try:
        if provisional_direction in {"BUY", "SELL"}:
            tp_methods = compute_tp_methods(
                rows=rows,
                direction=provisional_direction,
                entry_mid=entry_mid,
                stop_loss=levels.get("stop_loss") or 0.0,
                volume_profile=volume_profile,
                nearest_sr=rich_sr,
            )
        else:
            tp_methods = {}
    except Exception:  # noqa: BLE001
        logger.exception("compute_tp_methods failed")
        tp_methods = {}

    # Merge multi-method TPs into levels and recompute RR using the actual served TP1.
    # Without this recompute, the gate below would test the original simple-RR TP1 while
    # the user is shown the (possibly tighter) multi-method TP1 — letting through "BUY"
    # signals whose effective RR has dropped under SIGNAL_MIN_RR.
    if tp_methods:
        levels["tp3"] = tp_methods.get("tp3")
        if tp_methods.get("tp1"):
            levels["tp1"] = tp_methods["tp1"]
        if tp_methods.get("tp2"):
            levels["tp2"] = tp_methods["tp2"]
        levels["tp_methods"] = {
            "tp1": tp_methods.get("tp1_methods"),
            "tp2": tp_methods.get("tp2_methods"),
            "tp3": tp_methods.get("tp3_methods"),
            "tp1_confluence": tp_methods.get("tp1_confluence"),
            "tp2_confluence": tp_methods.get("tp2_confluence"),
            "tp3_confluence": tp_methods.get("tp3_confluence"),
        }
        new_entry = levels.get("entry_mid")
        new_stop = levels.get("stop_loss")
        new_tp1 = levels.get("tp1")
        if new_entry and new_stop and new_tp1:
            new_risk = abs(new_entry - new_stop)
            new_reward = abs(new_tp1 - new_entry)
            if new_risk > 0:
                levels["risk_per_share"] = round(new_risk, 1)
                levels["risk_reward_ratio"] = round(new_reward / new_risk, 2)

    rr = levels.get("risk_reward_ratio") or 0.0  # gross RR from actual rounded levels
    entry_plan = float(levels.get("entry_mid") or 0.0)
    stop_plan = float(levels.get("stop_loss") or 0.0)
    risk_fils = abs(entry_plan - stop_plan)
    slippage_rate = TC_SLIPPAGE_PREMIER if segment.upper() == "PREMIER" else TC_SLIPPAGE_MAIN
    entry_cost_fils = entry_plan * (TC_COMMISSION + slippage_rate + max(0.0, spread_pct) / 200.0)
    stop_cost_fils = stop_plan * (TC_COMMISSION + slippage_rate + max(0.0, spread_pct) / 200.0)
    gap_samples = [
        abs(float(row.get("open") or 0.0) - float(previous.get("close") or 0.0))
        for previous, row in zip(rows[-21:-1], rows[-20:])
        if float(row.get("open") or 0.0) > 0 and float(previous.get("close") or 0.0) > 0
    ]
    gap_risk_fils = float(np.median(gap_samples)) if gap_samples else 0.0
    net_risk_fils = risk_fils + entry_cost_fils + stop_cost_fils + gap_risk_fils
    served_targets = {key: levels.get(key) for key in ("tp1", "tp2", "tp3")}
    for target_key, target_price in served_targets.items():
        if target_price is None or risk_fils <= 0:
            continue
        target_cost_fils = float(target_price) * (TC_COMMISSION + slippage_rate + max(0.0, spread_pct) / 200.0)
        gross_reward_fils = abs(float(target_price) - entry_plan)
        net_reward_fils = gross_reward_fils - entry_cost_fils - target_cost_fils
        levels.setdefault("target_metrics", {})[target_key] = {
            "price": float(target_price),
            "methods": (levels.get("tp_methods") or {}).get(target_key) or levels.get(f"{target_key}_methods"),
            "confidence": levels.get(f"{target_key}_confluence"),
            "distance_fils": round(gross_reward_fils, 2),
            "gross_rr": round(gross_reward_fils / risk_fils, 3),
            "net_rr": round(net_reward_fils / net_risk_fils, 3) if net_risk_fils > 0 else None,
        }
    levels["gross_rr"] = round(rr, 3)
    levels["net_rr"] = round(
        (abs(float(levels.get("tp1") or 0.0) - entry_plan) - entry_cost_fils - (float(levels.get("tp1") or 0.0) * (TC_COMMISSION + slippage_rate + max(0.0, spread_pct) / 200.0))) / net_risk_fils,
        3,
    ) if levels.get("tp1") is not None and net_risk_fils > 0 else None
    levels["costs"] = {
        "commission_rate_each_leg": TC_COMMISSION,
        "slippage_rate_each_leg": slippage_rate,
        "spread_pct_estimate": max(0.0, spread_pct),
        "entry_cost_fils": round(entry_cost_fils, 3),
        "stop_exit_cost_fils": round(stop_cost_fils, 3),
        "gap_risk_fils": round(gap_risk_fils, 3),
        "net_risk_fils": round(net_risk_fils, 3),
    }
    net_rr = levels.get("net_rr") or 0.0

    # ── 7. Risk/Reward score (0-100 raw → 0-15 weighted) ─────────────────────
    rr_raw = max(0, min(100, int(((rr - 1.0) / 3.0) * 100)))

    # ── 8. Apply regime + liquidity weight adjustments ────────────────────────
    weights_regime = regime if liquidity_passed else "Neutral_Chop"
    weights = _apply_regime_weights(dict(BASE_WEIGHTS), weights_regime, liq_pct)

    # ── 9. Weighted sub-scores (each 0-max_weight*100) ───────────────────────
    w_trend = weights["trend"]
    w_mom = weights["momentum"]
    w_vol = weights["volume_flow"]
    w_sr = weights["support_resistance"]
    w_rr = weights["risk_reward"]

    component_available = {
        "trend": bool(trend_details.get("available", True)),
        "momentum": bool(momentum_details.get("available", True)),
        "volume_flow": bool(volume_details.get("available", True)),
        "support_resistance": "error" not in sr_details,
        "risk_reward": rr > 0.0,
    }
    available_weight = sum(
        component_weight
        for component_name, component_weight in weights.items()
        if component_available.get(component_name, False)
    )
    coverage_ratio = sum(component_available.values()) / len(component_available)
    if coverage_ratio >= MIN_INDICATOR_COVERAGE and available_weight > 0:
        effective_weights = {
            name: (weight / available_weight if component_available.get(name, False) else 0.0)
            for name, weight in weights.items()
        }
    else:
        effective_weights = {name: (weight if component_available.get(name, False) else 0.0) for name, weight in weights.items()}

    sub_weighted = {
        "trend":              round(trend_raw * effective_weights["trend"]),
        "momentum":           round(momentum_raw * effective_weights["momentum"]),
        "volume_flow":        round(volume_raw * effective_weights["volume_flow"]),
        "support_resistance": round(sr_raw * effective_weights["support_resistance"]),
        "risk_reward":        round(rr_raw * effective_weights["risk_reward"]),
    }
    total_score = sum(sub_weighted.values())
    combined_score_adjusted_directional = int(
        (
            sub_weighted["trend"]
            + sub_weighted["momentum"]
            + sub_weighted["volume_flow"]
            + sub_weighted["support_resistance"]
        ) / 0.85
    )

    # Unadjusted combined score: same weights but uses the pure base trend score
    # (before directional context multipliers such as ER, trend-age, EMA-stretch).
    # Computed explicitly from all sub-scores so it stays correct if total_score changes.
    sub_weighted_unadjusted = {
        "trend":              round(trend_base_raw * w_trend),
        "momentum":           round(momentum_raw * w_mom),
        "volume_flow":        round(volume_raw * w_vol),
        "support_resistance": round(sr_raw * w_sr),
        "risk_reward":        round(rr_raw * w_rr),
    }
    total_score_unadjusted = sum(sub_weighted_unadjusted.values())
    combined_score_unadjusted_directional = int(
        (
            sub_weighted_unadjusted["trend"]
            + sub_weighted_unadjusted["momentum"]
            + sub_weighted_unadjusted["volume_flow"]
            + sub_weighted_unadjusted["support_resistance"]
        ) / 0.85
    )

    circuit_proximity = {
        "is_near_limit": False,
        "direction": "none",
        "distance_to_upper_pct": None,
        "distance_to_lower_pct": None,
    }
    combined_score_adjusted_directional_raw = combined_score_adjusted_directional

    # ── 10. Circuit-breaker score haircut (BEFORE classification) ────────────
    # Applied here so the displayed score equals the score the gate evaluates.
    if len(rows) >= 2:
        prev_close = float(rows[-2].get("close") or 0.0)
        alerts.extend(_circuit_breaker_alerts(rows, prev_close))
        close_now = float(rows[-1].get("close") or 0.0)
        if close_now > 0 and prev_close > 0:
            upper = prev_close * (1.0 + CIRCUIT_UPPER_PCT)
            lower = prev_close * (1.0 + CIRCUIT_LOWER_PCT)
            dist_upper = (upper - close_now) / close_now
            dist_lower = (close_now - lower) / close_now
            near_upper = dist_upper <= CIRCUIT_BUFFER_PCT
            near_lower = dist_lower <= CIRCUIT_BUFFER_PCT

            circuit_direction = "none"
            if near_upper and near_lower:
                circuit_direction = "both"
            elif near_upper:
                circuit_direction = "upper"
            elif near_lower:
                circuit_direction = "lower"

            circuit_proximity = {
                "is_near_limit": bool(near_upper or near_lower),
                "direction": circuit_direction,
                "distance_to_upper_pct": round(dist_upper * 100.0, 3),
                "distance_to_lower_pct": round(dist_lower * 100.0, 3),
            }

            if near_upper and levels.get("tp1"):
                capped_tp1 = align_to_tick(upper * 0.997)
                levels["tp1"] = min(levels["tp1"], capped_tp1)

            if near_upper or near_lower:
                total_score = int(total_score * 0.70)
                total_score_unadjusted = int(total_score_unadjusted * 0.70)
                combined_score_adjusted_directional = int(combined_score_adjusted_directional * 0.70)
                combined_score_unadjusted_directional = int(combined_score_unadjusted_directional * 0.70)

    # ── 11. CVaR ──────────────────────────────────────────────────────────────
    cvar_result = calculate_cvar(rows, adtv_kd=adtv_kd)

    # ── 12. Probability calibration ───────────────────────────────────────────
    # The calibrator's score→win-rate table is in BUY-polarity (high score → high p).
    # For a SELL setup, low total_score is the *good* signal, so we look up the
    # mirrored quality score so p_tp1 is meaningful for shorts too.
    calib_score = total_score if provisional_direction != "SELL" else max(0, 100 - total_score)
    prob_result = calibrate_probabilities(
        total_score=calib_score,
        regime=regime,
        recent_performance=recent_performance or {},
    )

    # ── 13. Auction confidence adjustment to probabilities ───────────────────
    raw_p_tp1 = prob_result.get("p_tp1_before_sl")
    raw_p_tp2 = prob_result.get("p_tp2_before_sl")
    if isinstance(raw_p_tp1, (int, float)):
        prob_result["p_tp1_before_sl"] = round(min(0.95, raw_p_tp1 * auction_adj), 3)
    if isinstance(raw_p_tp2, (int, float)):
        prob_result["p_tp2_before_sl"] = round(min(0.90, raw_p_tp2 * auction_adj), 3)

    # ── 14. Confidence decay ──────────────────────────────────────────────────
    prob_result = adjust_confidence_for_delay(prob_result, delay_hours)
    if prob_result.get("decay_factor") == 0.0:
        alerts.append(
            "Signal invalidated: ≥ 72 hours since generation — require new confirmation candle"
        )

    quality_factor = _clamp(data_quality_score / 100.0, 0.0, 1.0)
    if quality_factor < 1.0:
        for probability_key in ("p_tp1_before_sl", "p_tp2_before_sl", "p_tp3_before_sl"):
            probability = prob_result.get(probability_key)
            if isinstance(probability, (int, float)):
                prob_result[probability_key] = round(float(probability) * quality_factor, 3)
        expected_return = prob_result.get("expected_return_r_multiple")
        if isinstance(expected_return, (int, float)):
            prob_result["expected_return_r_multiple"] = round(float(expected_return) * quality_factor, 3)
        alerts.append(f"DATA QUALITY WARNING: calibrated confidence scaled by {quality_factor:.0%}")

    if data_quality_score < MIN_REQUIRED_DATA_QUALITY:
        prob_result = {
            **prob_result,
            "p_tp1_before_sl": None,
            "p_tp2_before_sl": None,
            "p_tp3_before_sl": None,
            "confidence_interval_95": None,
            "expected_return_r_multiple": None,
            "calibration_method": "unavailable_insufficient_data_quality",
        }

    # ── 15. Position sizing (single call with post-decay win_prob) ────────────
    position_result = calculate_position_size(
        account_equity=account_equity,
        entry_price=levels.get("entry_mid") or 0.0,
        stop_loss=levels.get("stop_loss") or 0.0,
        adtv_kd=adtv_kd,
        win_probability=prob_result.get("p_tp1_before_sl"),
        cvar_reduction=cvar_result.get("position_size_reduction") or 1.0,
        net_rr=net_rr,
        probability_status=prob_result.get("probability_status"),
        segment=segment,
        spread_pct=spread_pct,
        gap_risk_fils=gap_risk_fils,
    )

    # ── 16. Final signal determination — score AND probability gates ──────────
    trend_pct = trend_raw
    vol_pct = volume_raw
    post_decay_p_tp1 = prob_result.get("p_tp1_before_sl") or 0.0

    resistance_within_1_5r = False
    if nearest_resistance and levels.get("entry_mid") and levels.get("risk_per_share"):
        one_half_r = levels["risk_per_share"] * 1.5
        if nearest_resistance - levels["entry_mid"] < one_half_r:
            resistance_within_1_5r = True

    support_within_1_5r = False
    if nearest_support and levels.get("entry_mid") and levels.get("risk_per_share"):
        one_half_r = levels["risk_per_share"] * 1.5
        if levels["entry_mid"] - nearest_support < one_half_r:
            support_within_1_5r = True

    buy_gates = (
        data_quality_score >= 45.0
        and long_direction_ready
        and total_score >= SIGNAL_MIN_TOTAL_SCORE
        and trend_pct >= SIGNAL_MIN_TREND_RAW_PCT
        and vol_pct >= SIGNAL_MIN_VOLFLOW_RAW_PCT
        and net_rr >= SIGNAL_MIN_RR
        and liquidity_passed
        and not resistance_within_1_5r
        and post_decay_p_tp1 >= SIGNAL_MIN_P_TP1_BUY
    )
    sell_gates = (
        data_quality_score >= 45.0
        and short_direction_ready
        and total_score <= SIGNAL_MAX_TOTAL_SELL
        and trend_pct <= (100.0 - SIGNAL_MIN_TREND_RAW_PCT)
        and vol_pct <= (100.0 - SIGNAL_MIN_VOLFLOW_RAW_PCT)
        and net_rr >= SIGNAL_MIN_RR
        and liquidity_passed
        and not support_within_1_5r
        and post_decay_p_tp1 >= SIGNAL_MIN_P_TP1_SELL
    )

    if provisional_direction == "BUY" and buy_gates:
        final_signal = (
            "STRONG_BUY"
            if total_score >= SIGNAL_STRONG_BUY_SCORE
               and post_decay_p_tp1 >= SIGNAL_MIN_P_TP1_STRONG_BUY
            else "BUY"
        )
    elif provisional_direction == "SELL" and sell_gates:
        final_signal = "SELL"
    else:
        final_signal = "NEUTRAL"

    if resistance_within_1_5r and provisional_direction == "BUY":
        alerts.append("Major resistance detected within 1.5R — BUY signal blocked")
    if support_within_1_5r and provisional_direction == "SELL":
        alerts.append("Major support detected within 1.5R — SELL signal blocked")

    # ── 16b. Entry trigger evaluation ─────────────────────────────────────
    if final_signal in ("BUY", "STRONG_BUY"):
        score_tier = "Strong Buy" if final_signal == "STRONG_BUY" else "Buy"
        entry_trigger = evaluate_entry_trigger(rows, score_tier)
    elif final_signal == "SELL":
        entry_trigger = _evaluate_short_entry_trigger(rows)
    else:
        entry_trigger = {"action": "HOLD", "trigger": "none",
                         "pullback": {"triggered": False, "reason": "non_actionable"},
                         "breakout": {"triggered": False, "reason": "non_actionable"},
                         "accumulation": {"state": "absent", "obv_slope_pct": None, "cmf": None},
                         "short_breakdown": {"triggered": False, "reason": "non_actionable"},
                         "failed_rally": {"triggered": False, "reason": "non_actionable"},
                         "distribution": {"state": "absent", "obv_slope_pct": None, "cmf": None}}

    # ── 17. Setup type classification ─────────────────────────────────────────
    setup_type = classify_setup_type(rows, final_signal, trend_raw, momentum_raw, sr_details)

    # ── 18. Resistance / psychological level alerts ───────────────────────────
    if nearest_resistance and levels.get("tp2") and nearest_resistance <= (levels["tp2"] * 1.02):
        alerts.append(f"Psychological resistance near TP2 ({nearest_resistance:.1f} fils) — monitor TP2 execution")
    if nearest_support and provisional_direction == "BUY":
        alerts.append(f"Key support at {nearest_support:.1f} fils confirms entry zone")

    # ── 19. Assemble final output ─────────────────────────────────────────────
    trend_dir = _clamp((float(trend_raw) - 50.0) / 50.0, -1.0, 1.0)
    momentum_dir = _clamp((float(momentum_raw) - 50.0) / 50.0, -1.0, 1.0)
    volume_dir = _clamp((float(volume_raw) - 50.0) / 50.0, -1.0, 1.0)
    sr_dir = 0.0
    if nearest_support is not None and nearest_resistance is not None and close_now > 0:
        span = max(nearest_resistance - nearest_support, 1e-6)
        rel = (close_now - nearest_support) / span
        sr_dir = _clamp((rel - 0.5) * 2.0, -1.0, 1.0)

    rr_quality = _clamp(rr / 3.0, 0.0, 1.0)
    rr_dir = 0.0
    if provisional_direction == "BUY":
        rr_dir = 0.5 if rr > 1.0 else 0.0
    elif provisional_direction == "SELL":
        rr_dir = -0.5 if rr > 1.0 else 0.0

    regime_direction = 0.0
    regime_upper = str(regime or "").upper()
    if "UP" in regime_upper:
        regime_direction = 0.6
    elif "DOWN" in regime_upper:
        regime_direction = -0.6

    four_scores = compute_all_four_scores(
        rows=rows,
        trend_raw=trend_raw,
        momentum_raw=momentum_raw,
        volume_raw=volume_raw,
        sr_details=sr_details,
        auction_intensity=auction_intensity,
        rr_ratio=rr,
        adtv_kwd=float(adtv_kd or 0.0),
        spread_pct=float(spread_pct),
        circuit_result={"nearest_circuit_pct": 5.0},
    )
    timing_raw = (four_scores.get("timing") or {}).get("score")
    timing_score = float(timing_raw) if isinstance(timing_raw, (int, float)) else 0.0

    timing_quality = _clamp(timing_score / 100.0, 0.0, 1.0)
    timing_direction = 0.0
    if provisional_direction == "BUY":
        timing_direction = 0.5
    elif provisional_direction == "SELL":
        timing_direction = -0.5

    trigger_action = str((entry_trigger or {}).get("action") or "HOLD").upper()
    if trigger_action == "HOLD":
        timing_direction = 0.0

    # Correlation attenuation to avoid over-counting tightly related families.
    directional_sign_agreement = abs((trend_dir + momentum_dir + volume_dir) / 3.0)
    correlation_penalty = _clamp((directional_sign_agreement - 0.7) / 0.3, 0.0, 1.0) * 0.15
    effective_directional_weight = round(1.0 - correlation_penalty, 3)

    component_scores = {
        "trend": _component_contract(
            direction=trend_dir,
            quality=float(trend_raw) / 100.0,
            confidence=min(1.0, max(0.0, float(regime_confidence) * effective_directional_weight)),
            available=True,
            details={"raw": trend_raw, "adjusted_weighted": sub_weighted.get("trend")},
        ),
        "momentum": _component_contract(
            direction=momentum_dir,
            quality=float(momentum_raw) / 100.0,
            confidence=0.75 * effective_directional_weight,
            available=True,
            details={"raw": momentum_raw, "adjusted_weighted": sub_weighted.get("momentum")},
        ),
        "volume_flow": _component_contract(
            direction=volume_dir,
            quality=float(volume_raw) / 100.0,
            confidence=0.7 * effective_directional_weight,
            available=True,
            details={"raw": volume_raw, "adjusted_weighted": sub_weighted.get("volume_flow")},
        ),
        "support_resistance": _component_contract(
            direction=sr_dir,
            quality=float(sr_raw) / 100.0,
            confidence=0.7,
            available=True,
            details={
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "adjusted_weighted": sub_weighted.get("support_resistance"),
            },
        ),
        "risk_reward": _component_contract(
            direction=rr_dir,
            quality=rr_quality,
            confidence=1.0 if rr > 0 else 0.0,
            available=rr > 0,
            details={"rr_ratio": rr, "adjusted_weighted": sub_weighted.get("risk_reward")},
        ),
        "regime": _component_contract(
            direction=regime_direction,
            quality=_clamp(float(regime_confidence), 0.0, 1.0),
            confidence=_clamp(float(regime_confidence), 0.0, 1.0),
            available=True,
            details={"name": regime, "liquidity_passed": liquidity_passed},
        ),
        "entry_timing": _component_contract(
            direction=timing_direction,
            quality=timing_quality,
            confidence=0.8 if trigger_action in {"ENTER", "WATCH"} else 0.5,
            available=True,
            details={"trigger_action": trigger_action, "trigger": (entry_trigger or {}).get("trigger")},
        ),
    }

    scoring_model = {
        "version": "2.0.0",
        "weights": {
            "trend": w_trend,
            "momentum": w_mom,
            "volume_flow": w_vol,
            "support_resistance": w_sr,
            "risk_reward": w_rr,
        },
        "effective_weights": effective_weights,
        "component_availability": component_available,
        "coverage_ratio": round(coverage_ratio, 3),
        "rationale": "Directional components are polarity-signed and quality-normalized; risk/reward is quality-first with directional context.",
        "correlation_controls": {
            "directional_family": ["trend", "momentum", "volume_flow"],
            "sign_agreement": round(directional_sign_agreement, 3),
            "penalty": round(correlation_penalty, 3),
            "effective_directional_weight": effective_directional_weight,
        },
    }

    risk_merged = {**position_result, **cvar_result}
    confluence = {
        "total_score": total_score,
        "total_score_raw": combined_score_adjusted_directional_raw,
        "regime": regime,
        "regime_confidence": regime_confidence,
        "auction_intensity": auction_intensity,
        "sub_scores": sub_weighted,
        "raw_sub_scores": {
            "trend": trend_raw,
            "momentum": momentum_raw,
            "volume_flow": volume_raw,
            "support_resistance": sr_raw,
            "risk_reward": rr_raw,
        },
        "liquidity_passed": liquidity_passed,
        "liquidity_details": liq_details,
        "scenario_levels": scenario_levels,
        "circuit_proximity": circuit_proximity,
        # Price level arrays for UI price ladder (up to 3 nearest levels each)
        "support_levels": sr_details.get("support_levels", [])[:3],
        "resistance_levels": sr_details.get("resistance_levels", [])[:3],
        "vwap": sr_details.get("anchored_vwap"),
        # Rich S/R map (for UI S/R Map section)
        "rich_sr": rich_sr,
        # Volume profile summary
        "volume_profile": {
            "poc": volume_profile.get("poc"),
            "value_area_high": volume_profile.get("value_area_high"),
            "value_area_low": volume_profile.get("value_area_low"),
            "hvn_levels": volume_profile.get("hvn_levels", [])[:5],
            "lvn_levels": volume_profile.get("lvn_levels", [])[:5],
        },
        "indicator_breakdown": _build_indicator_breakdown(
            trend_details=trend_details,
            momentum_details=momentum_details,
            volume_details=volume_details,
            sr_details=sr_details,
        ),
        "four_scores": four_scores,
        "component_scores": component_scores,
        "scoring_model": scoring_model,
        "component_availability": component_available,
        "component_coverage": round(coverage_ratio, 3),
        "effective_weights": effective_weights,
    }

    # New explicit direction/quality/timing contract (keeps legacy `signal` for compatibility).
    trend_strength = abs(float(trend_raw) - 50.0) * 2.0
    volume_strength = abs(float(volume_raw) - 50.0) * 2.0
    momentum_strength = abs(float(momentum_raw) - 50.0) * 2.0
    setup_quality_score = (
        0.30 * trend_strength
        + 0.20 * volume_strength
        + 0.15 * momentum_strength
        + 0.20 * float(sr_raw)
        + 0.15 * float(rr_raw)
    )
    if not liquidity_passed:
        setup_quality_score *= 0.6

    direction_score = (
        (float(trend_raw) - 50.0) * 1.0
        + (float(volume_raw) - 50.0) * 0.7
        + (float(momentum_raw) - 50.0) * 0.3
    )
    direction_score = _clamp(direction_score * 2.0 / 2.0, -100.0, 100.0)

    expected_value_r = prob_result.get("expected_return_r_multiple")
    if not isinstance(expected_value_r, (int, float)):
        expected_value_r = None

    recommendation_contract = _compute_recommendation_contract(
        final_signal=final_signal,
        direction_score=direction_score,
        setup_quality_score=setup_quality_score,
        timing_score=timing_score,
        data_quality_score=data_quality_score,
        expected_value_r=expected_value_r,
        entry_trigger_action=str((entry_trigger or {}).get("action") or "HOLD"),
        neutral_reason=";".join(alerts),
        probability_status=prob_result.get("probability_status"),
    )

    gate_audit = {
        "data_quality": data_quality_score >= 45.0,
        "directional_agreement": long_direction_ready if provisional_direction == "BUY" else short_direction_ready if provisional_direction == "SELL" else False,
        "entry_trigger": str((entry_trigger or {}).get("action") or "HOLD").upper() in {"ENTER", "WATCH"},
        "structure_stop": bool(levels.get("stop_loss")),
        "profitable_target": bool(levels.get("tp1")),
        "minimum_net_rr": net_rr >= SIGNAL_MIN_RR,
        "liquidity": liquidity_passed,
        "probability_calibrated": prob_result.get("probability_status") == "CALIBRATED",
        "stale_signal": prob_result.get("decay_factor") != 0.0,
        "passed": bool(recommendation_contract.get("actionable")),
    }
    confluence["gate_audit"] = gate_audit

    # ── §8 Runtime Monitoring — log required metrics for every signal ─────────
    from datetime import datetime, timezone  # noqa: PLC0415 (local import to avoid cycle)
    _friction_pct = round(
        (2 * 0.0015 + 2 * (0.0010 if segment.upper() == "PREMIER" else 0.0030)) * 100, 3
    )
    logger.info(
        "[SIGNAL] ts=%s  stock=%s  signal=%s  data_as_of=%s  delay_h=%d  "
        "regime=%s  regime_conf=%.2f  score=%d  p_tp1=%.3f  friction_pct=%.3f%%",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        stock_code,
        final_signal,
        data_as_of,
        delay_hours,
        regime,
        regime_confidence,
        total_score,
        prob_result.get("p_tp1_before_sl") or 0.0,
        _friction_pct,
    )

    signal_out = format_signal(
        stock_code=stock_code,
        segment=segment,
        signal_direction=final_signal,
        setup_type=setup_type,
        levels=levels,
        risk_metrics=risk_merged,
        probabilities=prob_result,
        confluence=confluence,
        alerts=alerts,
        data_as_of=data_as_of,
        entry_trigger=entry_trigger,
        recommendation_contract=recommendation_contract,
        data_quality_reasons=data_quality_reasons,
    )
    # Attach dual combined scores so the daily batch can persist them separately:
    #   combined_score_adjusted_directional   — total_score using trend WITH directional multipliers
    #   combined_score_unadjusted_directional — total_score using base trend WITHOUT multipliers
    signal_out["combined_score_adjusted_directional"] = combined_score_adjusted_directional
    signal_out["combined_score_unadjusted_directional"] = combined_score_unadjusted_directional
    signal_out["raw_technical_score"] = combined_score_adjusted_directional
    # Per-stock trend directional haircut: the combined multiplier applied to the base trend
    # score (e.g. 0.87 means the trend score was reduced to 87% of its raw structural value).
    # Component multipliers are exposed separately for transparency.
    signal_out["trend_directional_factor"] = trend_details.get("adjustment_factor")
    signal_out["trend_directional_multipliers"] = trend_details.get("multipliers")
    return signal_out


def _neutral_signal(
    stock_code: str,
    segment: str,
    data_as_of: str,
    reason: str,
    *,
    data_quality_score: float = 0.0,
    data_quality_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Return a minimal NEUTRAL signal with the given reason in alerts."""
    blocked_four = _make_blocked_four_scores(rows=[], adtv_kwd=0.0, spread_pct=0.0)
    recommendation_contract = {
        "direction": "NEUTRAL",
        "direction_score": 0,
        "setup_quality_score": 0,
        "timing_score": 0,
        "data_quality_score": round(_clamp(data_quality_score, 0.0, 100.0), 1),
        "expected_value_r": None,
        "recommendation": "INSUFFICIENT_DATA" if reason == "insufficient_data" else "HOLD",
        "actionable": False,
    }

    return format_signal(
        stock_code=stock_code,
        segment=segment,
        signal_direction="NEUTRAL",
        setup_type="No_Signal",
        levels={
            "entry_low": None, "entry_mid": None, "entry_high": None,
            "stop_loss": None, "tp1": None, "tp2": None,
            "risk_per_share": None, "risk_reward_ratio": None,
        },
        risk_metrics={"equity_pct": None, "cvar_fils": None, "liquidity_factor": None},
        probabilities={
            "p_tp1_before_sl": None, "p_tp2_before_sl": None,
            "confidence_interval_95": None, "expected_return_r_multiple": None,
            "calibration_method": "n/a",
        },
        confluence={
            "total_score": 0, "total_score_raw": 0, "regime": "Neutral_Chop",
            "regime_confidence": None, "auction_intensity": None,
            "sub_scores": {}, "raw_sub_scores": {},
            "liquidity_passed": False, "liquidity_details": {},
            "circuit_proximity": {
                "is_near_limit": False,
                "direction": "none",
                "distance_to_upper_pct": None,
                "distance_to_lower_pct": None,
            },
            "indicator_breakdown": None,
            "four_scores": blocked_four,
        },
        alerts=[f"No signal: {reason}"],
        data_as_of=data_as_of,
        entry_trigger={"action": "HOLD", "trigger": "none",
                       "pullback": {"triggered": False, "reason": reason},
                       "breakout": {"triggered": False, "reason": reason},
                       "accumulation": {"state": "absent", "obv_slope_pct": None, "cmf": None}},
        recommendation_contract=recommendation_contract,
        data_quality_reasons=data_quality_reasons or [],
    )
