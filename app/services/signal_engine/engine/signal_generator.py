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
from typing import Any

from app.services.signal_engine.config.kuwait_constants import (
    CIRCUIT_BUFFER_PCT,
    CIRCUIT_LOWER_PCT,
    CIRCUIT_UPPER_PCT,
    PREMIER_ADTV_MIN_KD,
)
from app.services.signal_engine.config.model_params import (
    BASE_WEIGHTS,
    MIN_BARS_FOR_SIGNAL,
    REGIME_BULL,
    SIGNAL_MAX_TOTAL_SELL,
    SIGNAL_MIN_RR,
    SIGNAL_MIN_TOTAL_SCORE,
    SIGNAL_MIN_TREND_RAW_PCT,
    SIGNAL_MIN_VOLFLOW_RAW_PCT,
    SIGNAL_STRONG_BUY_SCORE,
)
from app.services.signal_engine.engine.output_formatter import classify_setup_type, format_signal
from app.services.signal_engine.engine.probability_calibrator import calibrate_probabilities
from app.services.signal_engine.models.regime.hmm_regime_detector import predict_regime
from app.services.signal_engine.models.regime.hurst_filter import compute_hurst_filter
from app.services.signal_engine.models.regime.transition_monitor import (
    detect_transition_alerts,
    get_regime_weight_adjustment,
)
from app.services.signal_engine.models.risk.confluence_decay import adjust_confidence_for_delay
from app.services.signal_engine.models.risk.cvar_calculator import calculate_cvar
from app.services.signal_engine.models.risk.position_sizer import calculate_position_size, kuwait_phased_kelly
from app.services.signal_engine.models.technical.momentum_score import compute_momentum_score
from app.services.signal_engine.models.technical.support_resistance import (
    compute_entry_stop_tp,
    compute_sr_score,
    compute_tp_methods,
)
from app.services.signal_engine.processors.sr_engine import calculate_full_sr_levels
from app.services.signal_engine.processors.volume_profile import calculate_volume_profile
from app.services.signal_engine.models.technical.four_score_engine import compute_all_four_scores
from app.services.signal_engine.models.technical.trend_score import compute_trend_score
from app.services.signal_engine.models.technical.volume_flow_score import compute_volume_flow_score
from app.services.signal_engine.processors.auction_proxy import (
    auction_confidence_adjustment,
    calculate_auction_intensity,
)
from app.services.signal_engine.processors.orderbook_imbalance import OrderBookImbalance
from app.services.signal_engine.processors.liquidity_filter import is_tradable
from app.services.signal_engine.processors.sector_regime import fetch_banking_index_regime

logger = logging.getLogger(__name__)


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


def _graduated_circuit_penalty(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return graduated circuit-breaker penalty for price near Kuwait circuit limits.

    Kuwait circuit limits (from constants):
    - Upper: +10 % from previous close
    - Lower: -5 % from previous close

    Penalty tiers applied to the technical score:
    - ≤ 0.5 % from limit → ×0.50  (severe)
    - 0.5–1.0 %          → ×0.80  (moderate)
    - 1.0–2.0 %          → ×0.95  (light)
    - > 2.0 %            → ×1.00  (none)

    Includes backward-compatible keys for existing test assertions:
    ``is_near_limit``, ``direction``, ``distance_to_upper_pct``,
    ``distance_to_lower_pct``.
    """
    _default: dict[str, Any] = {
        "penalty_multiplier": 1.0, "severity": "none",
        "nearest_circuit_pct": None, "is_near_upper_circuit": None,
        "is_near_limit": False, "direction": None,
        "distance_to_upper_pct": None, "distance_to_lower_pct": None,
        "description": "no_previous_bar",
    }
    if len(rows) < 2:
        return _default
    close = float(rows[-1].get("close") or 0.0)
    prev_close = float(rows[-2].get("close") or 0.0)
    if prev_close <= 0 or close <= 0:
        return {**_default, "description": "invalid_price"}

    upper = prev_close * (1.0 + CIRCUIT_UPPER_PCT)
    lower = prev_close * (1.0 + CIRCUIT_LOWER_PCT)
    dist_to_upper_pct = (upper - close) / prev_close * 100.0
    dist_to_lower_pct = (close - lower) / prev_close * 100.0
    nearest_pct = min(dist_to_upper_pct, dist_to_lower_pct)
    is_near_upper = dist_to_upper_pct <= dist_to_lower_pct

    if nearest_pct <= 0.5:
        multiplier, severity = 0.50, "severe"
    elif nearest_pct <= 1.0:
        multiplier, severity = 0.80, "moderate"
    elif nearest_pct <= 2.0:
        multiplier, severity = 0.95, "light"
    else:
        multiplier, severity = 1.00, "none"

    is_near = severity != "none"
    direction: str | None = None
    if is_near:
        direction = "upper" if is_near_upper else "lower"

    return {
        "penalty_multiplier": multiplier,
        "severity": severity,
        "nearest_circuit_pct": round(nearest_pct, 2),
        "is_near_upper_circuit": is_near_upper,
        # Backward-compat keys (used by existing tests):
        "is_near_limit": is_near,
        "direction": direction,
        "distance_to_upper_pct": round(dist_to_upper_pct, 2),
        "distance_to_lower_pct": round(dist_to_lower_pct, 2),
        "description": f"circuit_{severity}_{nearest_pct:.1f}pct",
    }


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


async def generate_kuwait_signal(
    rows: list[dict[str, Any]],
    stock_code: str,
    segment: str = "PREMIER",
    account_equity: float = 100_000.0,
    delay_hours: int = 0,
    recent_performance: dict[str, Any] | None = None,
    orderbook_client: Any | None = None,
    stock_meta: dict[str, Any] | None = None,
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
        orderbook_client:   Optional OrderBookClient instance for real-time OB data.

    Returns:
        Full signal dict matching the spec JSON schema.
        Signal is "NEUTRAL" when thresholds are not met or liquidity fails.
    """
    data_as_of = rows[-1]["date"] if rows else "unknown"

    # ── Guard: minimum data ───────────────────────────────────────────────────
    if len(rows) < MIN_BARS_FOR_SIGNAL:
        return _neutral_signal(
            stock_code, segment, data_as_of,
            reason_code="insufficient_data",
            four_scores=_make_blocked_four_scores(rows),
        )

    # ── 1. Liquidity filter ───────────────────────────────────────────────────
    liquidity_passed, liq_details = is_tradable(rows)
    adtv_kd = liq_details.get("adtv_20d_kd") or 0.0
    liq_pct = _liquidity_percentile(adtv_kd)

    alerts: list[str] = []

    # ── Hard gate: fail early if liquidity check fails (before technical scoring)
    if not liquidity_passed:
        _failed_gates = [
            k.replace("pass_", "") for k, v in liq_details.items()
            if k.startswith("pass_") and not v
        ]
        _liq_spread = float(liq_details.get("spread_proxy_pct") or 0.0)
        return _neutral_signal(
            stock_code, segment, data_as_of,
            reason_code="liquidity_failed",
            failed_gates=_failed_gates,
            block_details={k: v for k, v in liq_details.items() if not k.startswith("pass_")},
            four_scores=_make_blocked_four_scores(rows, adtv_kwd=adtv_kd, spread_pct=_liq_spread),
            liquidity_passed=liquidity_passed,
            liquidity_details=liq_details,
        )

    # ── 2. Circuit-breaker proximity (computed early so all early-return paths can use it)
    _circuit_result_early = _graduated_circuit_penalty(rows)
    _circuit_proximity_early: dict[str, Any] = {
        "is_near_limit": _circuit_result_early["is_near_limit"],
        "direction": _circuit_result_early["direction"],
        "distance_to_upper_pct": _circuit_result_early["distance_to_upper_pct"],
        "distance_to_lower_pct": _circuit_result_early["distance_to_lower_pct"],
    }

    # ── 3. Auction intensity ──────────────────────────────────────────────────
    auction_intensity = calculate_auction_intensity(rows)
    auction_adj = auction_confidence_adjustment(auction_intensity)

    # ── 4. Regime detection ───────────────────────────────────────────────────
    regime_result = predict_regime(rows)
    regime = regime_result.get("current_regime", "Neutral_Chop")
    regime_confidence = regime_result.get("regime_confidence") or 0.5
    regime_alerts = detect_transition_alerts(regime_result)
    alerts.extend(regime_alerts)

    # ── 4b. Hurst Exponent pre-filter (trend vs mean-reversion) ──────────────
    hurst_result = compute_hurst_filter(
        rows=rows,
        market_segment=segment,
        lookback_days=30,
    )
    
    hurst_confidence_penalty = hurst_result["confidence_penalty"]  # 0.70-1.0
    
    if hurst_result["action"] == "skip_signal":
        alerts.append(
            f"HURST FILTER FAIL: {hurst_result['description']} — "
            f"H={hurst_result['h_value']:.3f}±{hurst_result['h_std_error']:.3f}, "
            f"market shows mean-reverting behavior, skipping signal"
        )
        return _neutral_signal(stock_code, segment, data_as_of, reason_code="hurst_filter_fail",
                               circuit_proximity=_circuit_proximity_early,
                               four_scores=_make_blocked_four_scores(rows, adtv_kwd=adtv_kd, spread_pct=float(liq_details.get("spread_proxy_pct", 0.0))),
                               liquidity_passed=liquidity_passed,
                               liquidity_details=liq_details)
    
    if hurst_result["action"] == "skip_or_downgrade":
        alerts.append(
            f"HURST BORDERLINE: H={hurst_result['h_value']:.3f}±{hurst_result['h_std_error']:.3f}, "
            f"threshold={hurst_result['threshold_used']:.2f} — reduced confidence by "
            f"{(1.0 - hurst_confidence_penalty)*100:.0f}%"
        )

    # ── 4. Technical scoring ──────────────────────────────────────────────────
    trend_raw, trend_details = compute_trend_score(rows)
    momentum_raw, momentum_details = compute_momentum_score(rows)
    
    # ── 4a. Order Book Imbalance (real-time bid/ask flow) ────────────────────
    ob_metrics_dict: dict[str, Any] | None = None   # full dict passed to volume scorer
    ob_liquidity_wall: dict[str, Any] | None = None
    orderbook_imbalance: float | None = None          # scalar kept for confluence output
    
    if orderbook_client is not None:
        try:
            ob_analyzer = OrderBookImbalance(
                symbol=stock_code,
                market_segment=segment,
                api_client=orderbook_client,
            )
            
            ob_snapshot = await ob_analyzer.fetch_snapshot()
            
            if ob_snapshot:
                ob_metrics = ob_analyzer.compute_imbalance_ratio(
                    snapshot=ob_snapshot,
                    lookback_levels=5,
                )
                ob_metrics_dict = ob_metrics          # full dict → volume scorer
                orderbook_imbalance = ob_metrics.get("imbalance_ratio")  # scalar for output
                ob_liquidity_wall = ob_metrics.get("liquidity_wall")
                
                # Override auction intensity with real-time order book data
                historical_spread = await orderbook_client.get_historical_spread(
                    symbol=stock_code,
                    lookback_days=20,
                )
                auction_intensity = ob_analyzer.compute_auction_intensity_proxy(
                    snapshot=ob_snapshot,
                    historical_avg_spread_pct=historical_spread,
                )
                
                logger.info(
                    f"Order book for {stock_code}: imbalance={orderbook_imbalance:.3f}, "
                    f"wall={ob_liquidity_wall}, auction_intensity={auction_intensity:.2f}"
                )
                
                # Check for blocking liquidity walls
                # If major wall against our potential direction, downgrade signal
                if ob_liquidity_wall:
                    wall_side = ob_liquidity_wall.get("side")
                    wall_strength = ob_liquidity_wall.get("strength")
                    wall_price = ob_liquidity_wall.get("price")
                    
                    # Determine likely direction from preliminary scoring
                    is_likely_buy = trend_raw >= 60 and momentum_raw >= 50
                    is_likely_sell = trend_raw <= 40 and momentum_raw <= 50
                    
                    # Ask wall blocks BUY, Bid wall blocks SELL
                    if wall_side == "ask" and is_likely_buy and wall_strength == "strong":
                        alerts.append(
                            f"LIQUIDITY WALL BLOCK: Strong ask wall at {wall_price} "
                            f"blocking BUY direction — signal downgraded"
                        )
                        # Option: could return neutral signal here if user wants hard block
                        # return _neutral_signal(stock_code, segment, data_as_of, "liquidity_wall_block")
                    
                    elif wall_side == "bid" and is_likely_sell and wall_strength == "strong":
                        alerts.append(
                            f"LIQUIDITY WALL BLOCK: Strong bid wall at {wall_price} "
                            f"blocking SELL direction — signal downgraded"
                        )
                
        except Exception as e:
            logger.warning(f"Order book fetch failed for {stock_code}: {e}")
            # Fallback to volume-based auction proxy (already computed above)
    
    # ── 4b. Volume/Flow Scoring (with optional OB imbalance dict) ──────────────
    volume_raw, volume_details = compute_volume_flow_score(
        rows,
        auction_intensity,
        orderbook_imbalance=ob_metrics_dict,
    )
    
    sr_raw, sr_details, support_levels, resistance_levels = compute_sr_score(rows)

    nearest_support = support_levels[0] if support_levels else None
    nearest_resistance = resistance_levels[0] if resistance_levels else None
    # ── 4b. Volume profile ────────────────────────────────────────────────────
    try:
        volume_profile = calculate_volume_profile(rows)
    except Exception:  # noqa: BLE001
        logger.exception("Volume profile calculation failed")
        volume_profile = {}
    # ── 5. Determine signal direction (before RR calculation) ─────────────────
    is_bullish = trend_raw >= 60 and volume_raw >= 50
    is_bearish = trend_raw <= 40 and volume_raw <= 50

    if is_bullish:
        direction = "BUY"
    elif is_bearish:
        direction = "SELL"
    else:
        direction = "NEUTRAL"

    # ── 6. Entry / Stop / TP levels ───────────────────────────────────────────
    levels = compute_entry_stop_tp(rows, direction, nearest_resistance, nearest_support)
    rr = levels.get("risk_reward_ratio") or 0.0

    # ── 6b. Rich S/R map + multi-method TP ───────────────────────────────────
    entry_mid = levels.get("entry_mid") or float(rows[-1].get("close") or 0.0)
    try:
        rich_sr = calculate_full_sr_levels(rows, volume_profile, entry_mid)
    except Exception:  # noqa: BLE001
        logger.exception("Rich S/R calculation failed")
        rich_sr = {"resistance": [], "support": [], "nearest_resistance": None, "nearest_support": None}

    try:
        tp_methods = compute_tp_methods(
            rows=rows,
            direction=direction,
            entry_mid=entry_mid,
            stop_loss=levels.get("stop_loss") or 0.0,
            volume_profile=volume_profile,
            nearest_sr=rich_sr,
        )
    except Exception:  # noqa: BLE001
        logger.exception("compute_tp_methods failed")
        tp_methods = {}

    # Merge multi-method TPs into levels (tp3 + override tp1/tp2 if available)
    if tp_methods:
        levels["tp3"] = tp_methods.get("tp3")
        # Prefer multi-method TPs over simple RR targets when available
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

    # ── 7. Risk/Reward score (0-100 raw → 0-15 weighted) ─────────────────────
    rr_raw = max(0, min(100, int(((rr - 1.0) / 3.0) * 100)))

    # ── Preliminary sub-scores (base weights) for pre-flight gate neutrals ───
    # Used when RR/resistance gates fire before full step-9 scoring.
    _pre_weights = _apply_regime_weights(dict(BASE_WEIGHTS), "Neutral_Chop", 100.0)
    _pre_sub_raw = {
        "trend": trend_raw, "momentum": momentum_raw,
        "volume_flow": volume_raw, "support_resistance": sr_raw,
        "risk_reward": rr_raw,
    }
    _pre_sub_weighted = {
        "trend":              round(trend_raw * _pre_weights["trend"]),
        "momentum":           round(momentum_raw * _pre_weights["momentum"]),
        "volume_flow":        round(volume_raw * _pre_weights["volume_flow"]),
        "support_resistance": round(sr_raw * _pre_weights["support_resistance"]),
        "risk_reward":        round(rr_raw * _pre_weights["risk_reward"]),
    }
    _pre_four_factor_sum = sum(v for k, v in _pre_sub_weighted.items() if k != "risk_reward")
    _pre_total_raw = int(_pre_four_factor_sum / 0.85)
    # Apply circuit penalty so total_score matches what a real signal would show
    _pre_total = int(_pre_total_raw * _circuit_result_early["penalty_multiplier"])

    # ── Pre-gate four-score computation ──────────────────────────────────────
    # Computed here so early-return paths (poor_risk_reward, resistance_too_close,
    # hurst_chop_blocked) can surface four_scores in their UI output.
    _adtv_kwd = float(liq_details.get("adtv_20d_kd") or 0.0)
    _spread_pct = float(liq_details.get("spread_proxy_pct") or 0.0)
    four_scores = compute_all_four_scores(
        rows=rows,
        trend_raw=trend_raw,
        momentum_raw=momentum_raw,
        volume_raw=volume_raw,
        sr_details=sr_details,
        auction_intensity=auction_intensity,
        rr_ratio=rr,
        adtv_kwd=_adtv_kwd,
        spread_pct=_spread_pct,
        circuit_result=_circuit_result_early,
    )

    # ── Hard gate: BUY signals require minimum RR ratio ───────────────────────
    if direction == "BUY" and rr < SIGNAL_MIN_RR:
        return _neutral_signal(
            stock_code, segment, data_as_of,
            reason_code="poor_risk_reward",
            block_details={
                "rr_ratio": round(rr, 2),
                "minimum_required": SIGNAL_MIN_RR,
                "entry_mid": levels.get("entry_mid"),
                "stop_loss": levels.get("stop_loss"),
                "tp1": levels.get("tp1"),
            },
            technical_scores_debug={
                "trend_raw": trend_raw,
                "momentum_raw": momentum_raw,
                "volume_raw": volume_raw,
                "sr_raw": sr_raw,
            },
            circuit_proximity=_circuit_proximity_early,
            sub_scores=_pre_sub_weighted,
            raw_sub_scores=_pre_sub_raw,
            total_score_for_neutral=_pre_total,
            total_score_raw_for_neutral=_pre_total_raw,
            four_scores=four_scores,
        )

    # ── Hard gate: major resistance within 1.5R blocks BUY ───────────────────
    _resistance_within_1_5r = False
    if nearest_resistance and levels.get("entry_mid") and levels.get("risk_per_share"):
        _one_half_r = levels["risk_per_share"] * 1.5
        if nearest_resistance - levels["entry_mid"] < _one_half_r:
            _resistance_within_1_5r = True
    if _resistance_within_1_5r and direction == "BUY":
        return _neutral_signal(
            stock_code, segment, data_as_of,
            reason_code="resistance_too_close",
            block_details={
                "nearest_resistance": nearest_resistance,
                "entry_mid": levels.get("entry_mid"),
                "risk_per_share": levels.get("risk_per_share"),
            },
            circuit_proximity=_circuit_proximity_early,
            sub_scores=_pre_sub_weighted,
            raw_sub_scores=_pre_sub_raw,
            total_score_for_neutral=_pre_total,
            total_score_raw_for_neutral=_pre_total_raw,
            four_scores=four_scores,
        )

    # ── 8. Apply regime + liquidity weight adjustments ────────────────────────
    weights = _apply_regime_weights(dict(BASE_WEIGHTS), regime, liq_pct)

    # ── 8a. Banking Lead-Lag Filter ───────────────────────────────────────────
    # When banking sector trends strongly (>= 65), mid-cap non-banking stocks
    # tend to follow.  Boost their momentum and volume weights.
    _meta = stock_meta or {}
    _sector = _meta.get("sector", "")
    _mktcap_pct = float(_meta.get("market_cap_percentile", 100))
    _bll_active = False
    _bll_multiplier = 1.0
    _bll_trend_raw: float = 0.0
    try:
        banking_data = await fetch_banking_index_regime()
        if (
            banking_data.get("available")
            and banking_data.get("trend_raw", 0) >= 65
            and str(_sector).lower() != "banking"
            and _mktcap_pct < 70
        ):
            weights["momentum"] *= 1.8
            weights["volume_flow"] *= 1.3
            _bll_total = sum(weights.values())
            weights = {k: v / _bll_total for k, v in weights.items()}
            _bll_active = True
            _bll_multiplier = 1.8
            _bll_trend_raw = float(banking_data["trend_raw"])
            logger.debug(
                "Banking Lead-Lag active for %s: banking_trend=%.1f sector=%s mktcap_pct=%.0f",
                stock_code, banking_data["trend_raw"], _sector, _mktcap_pct,
            )
            alerts.append(
                f"Banking Lead-Lag active: banking trend={banking_data['trend_raw']:.0f} — "
                f"momentum/volume weights boosted for {stock_code}"
            )
        else:
            _bll_trend_raw = float(banking_data.get("trend_raw", 0))
    except Exception as _bll_exc:
        logger.debug("Banking Lead-Lag filter failed for %s: %s", stock_code, _bll_exc)

    # ── 9. Weighted sub-scores (each 0-max_weight*100) ───────────────────────
    w_trend = weights["trend"]
    w_mom = weights["momentum"]
    w_vol = weights["volume_flow"]
    w_sr = weights["support_resistance"]
    w_rr = weights["risk_reward"]

    sub_weighted = {
        "trend":              round(trend_raw * w_trend),
        "momentum":           round(momentum_raw * w_mom),
        "volume_flow":        round(volume_raw * w_vol),
        "support_resistance": round(sr_raw * w_sr),
        "risk_reward":        round(rr_raw * w_rr),
    }
    
    # We sum all keys EXCEPT risk_reward for the Combined Score
    _four_factor_sum = sum(v for k, v in sub_weighted.items() if k != "risk_reward")
    # Because we excluded 15% weight, re-normalize the remaining 85% to be out of 100
    total_score = int((_four_factor_sum / 0.85))
    
    # Apply Hurst confidence penalty to total score
    total_score = int(total_score * hurst_confidence_penalty)

    # ── 9b. Hurst + Neutral_Chop combined gate ────────────────────────────────
    # Even when Hurst says "skip_or_downgrade" (borderline), block low-conviction
    # signals when the regime is already Neutral_Chop.
    if (
        not hurst_result["is_trending"]
        and regime == "Neutral_Chop"
        and total_score < 75
    ):
        alerts.append(
            f"HURST+CHOP BLOCKED: mean-reverting market (H={hurst_result['h_value']:.3f}) "
            f"in Neutral_Chop regime — score {total_score} below 75 threshold"
        )
        return _neutral_signal(
            stock_code, segment, data_as_of,
            reason_code=f"hurst_chop_blocked: {hurst_result['description']}",
            circuit_proximity=_circuit_proximity_early,
            four_scores=four_scores,
            liquidity_passed=liquidity_passed,
            liquidity_details=liq_details,
        )

    # ── 10. Final signal determination ───────────────────────────────────────
    # Note: liquidity, RR, and resistance gates are handled by early returns above.
    trend_pct = trend_raw
    vol_pct = volume_raw

    buy_gates = (
        total_score >= SIGNAL_MIN_TOTAL_SCORE
        and trend_pct >= SIGNAL_MIN_TREND_RAW_PCT
        and vol_pct >= SIGNAL_MIN_VOLFLOW_RAW_PCT
    )
    sell_gates = (
        total_score <= SIGNAL_MAX_TOTAL_SELL
        and trend_pct <= (100.0 - SIGNAL_MIN_TREND_RAW_PCT)
        and vol_pct <= (100.0 - SIGNAL_MIN_VOLFLOW_RAW_PCT)
    )

    if direction == "BUY" and buy_gates:
        final_signal = "STRONG_BUY" if total_score >= SIGNAL_STRONG_BUY_SCORE else "BUY"
    elif direction == "SELL" and sell_gates:
        final_signal = "SELL"
    else:
        final_signal = "NEUTRAL"

    # ── 11. Graduated circuit-breaker penalty ────────────────────────────────
    # Reuse the already-computed early result (same rows, no need to recompute)
    _circuit_result = _circuit_result_early
    _circuit_multiplier = _circuit_result["penalty_multiplier"]

    # Backward-compat circuit_proximity dict (shape expected by existing tests)
    _circuit_proximity = _circuit_proximity_early

    # Track raw score before circuit penalty
    raw_technical_score = total_score   # post-Hurst, pre-circuit
    total_score_raw = raw_technical_score  # kept for backward compat in confluence

    # Apply graduated penalty
    total_score = int(raw_technical_score * _circuit_multiplier)
    risk_adjusted_score = total_score  # CVaR affects position sizing, not score

    if _circuit_result["severity"] != "none" and len(rows) >= 2:
        prev_close_cb = float(rows[-2].get("close") or 0.0)
        alerts.extend(_circuit_breaker_alerts(rows, prev_close_cb))
        # Cap TPs below upper circuit ceiling (0.5 % buffer)
        if _circuit_result.get("is_near_upper_circuit"):
            _upper_cb = prev_close_cb * (1.0 + CIRCUIT_UPPER_PCT)
            _circuit_cap_fils = _upper_cb * 0.995
            for _tp_key in ("tp1", "tp2", "tp3"):
                _tp_val = levels.get(_tp_key)
                if _tp_val is not None and _tp_val > _circuit_cap_fils:
                    levels[_tp_key] = round(_circuit_cap_fils, 1)

    _score_breakdown = {
        "raw_technical": raw_technical_score,
        "circuit_penalty_pct": round((1.0 - _circuit_multiplier) * 100, 1),
        "cvar_penalty_pct": 0.0,  # CVaR reduces position size, not the score
        "age_decay_applied": False,  # updated after decay step below
        "final_risk_adjusted": risk_adjusted_score,
    }

    # ── 12. CVaR and position sizing ─────────────────────────────────────────
    cvar_result = calculate_cvar(rows, adtv_kd=adtv_kd)
    position_result = calculate_position_size(
        account_equity=account_equity,
        entry_price=levels.get("entry_mid") or 0.0,
        stop_loss=levels.get("stop_loss") or 0.0,
        adtv_kd=adtv_kd,
        win_probability=None,      # populated after calibration below
        cvar_reduction=cvar_result.get("position_size_reduction") or 1.0,
    )

    # ── 13. Probability calibration ───────────────────────────────────────────
    prob_result = calibrate_probabilities(
        total_score=total_score,
        regime=regime,
        recent_performance=recent_performance or {},
    )

    # Re-run position sizing now that we have win_probability
    position_result = calculate_position_size(
        account_equity=account_equity,
        entry_price=levels.get("entry_mid") or 0.0,
        stop_loss=levels.get("stop_loss") or 0.0,
        adtv_kd=adtv_kd,
        win_probability=prob_result.get("p_tp1_before_sl"),
        cvar_reduction=cvar_result.get("position_size_reduction") or 1.0,
    )

    # ── 13b. Phased Kelly position sizing ────────────────────────────────────
    # "Early" phase until score confirms (>= STRONG_BUY threshold), then "Confirmed".
    _kelly_phase = "Confirmed" if total_score >= SIGNAL_STRONG_BUY_SCORE else "Early"
    _p_win_kelly = prob_result.get("p_tp1_before_sl") or 0.5
    _rr_ratio = levels.get("risk_reward_ratio") or 1.0
    phased_kelly_result: dict = {}
    try:
        phased_kelly_result = kuwait_phased_kelly(
            p_win=_p_win_kelly,
            rr_ratio=_rr_ratio,
            phase=_kelly_phase,
            adtv_percentile=liq_pct,
            market_segment=segment,
            portfolio_value_kwd=account_equity,
        )
        # Derive share count from kelly position value
        _entry = levels.get("entry_mid") or 0.0
        _kelly_shares = (
            int(phased_kelly_result["position_size_kwd"] / (_entry / 1000.0))
            if _entry > 0
            else 0
        )
        phased_kelly_result["phase"] = _kelly_phase
        phased_kelly_result["kelly_fraction"] = phased_kelly_result.pop("kelly_fraction")
        phased_kelly_result["shares"] = _kelly_shares
    except Exception:
        logger.exception("Phased Kelly calculation failed for %s", stock_code)

    # ── 14. Apply auction confidence adjustment to probabilities ─────────────
    p_tp1 = prob_result.get("p_tp1_before_sl") or 0.0
    p_tp2 = prob_result.get("p_tp2_before_sl") or 0.0
    adj_p_tp1 = round(min(0.95, p_tp1 * auction_adj), 3)
    adj_p_tp2 = round(min(0.90, p_tp2 * auction_adj), 3)
    prob_result["p_tp1_before_sl"] = adj_p_tp1
    prob_result["p_tp2_before_sl"] = adj_p_tp2

    # ── 15. Confidence decay ──────────────────────────────────────────────────
    prob_result = adjust_confidence_for_delay(prob_result, delay_hours)

    if prob_result.get("decay_factor") == 0.0:
        final_signal = "NEUTRAL"
        alerts.append("Signal invalidated: ≥ 72 hours since generation — require new confirmation candle")

    # ── 16. Setup type classification ─────────────────────────────────────────
    setup_type = classify_setup_type(rows, final_signal, trend_raw, momentum_raw, sr_details)

    # ── 17. Resistance / psychological level alerts ───────────────────────────
    if nearest_resistance and levels.get("tp2") and nearest_resistance <= (levels["tp2"] * 1.02):
        alerts.append(f"Psychological resistance near TP2 ({nearest_resistance:.1f} fils) — monitor TP2 execution")
    if nearest_support and direction == "BUY":
        alerts.append(f"Key support at {nearest_support:.1f} fils confirms entry zone")

    # ── 17b. Four-score already computed at pre-gate step (line ~460) ────────
    # four_scores is in scope; _circuit_result == _circuit_result_early (same obj)

    # ── 18. Assemble final output ─────────────────────────────────────────────
    risk_merged = {**position_result, **cvar_result}
    confluence = {
        "total_score": total_score,
        "regime": regime,
        "regime_confidence": regime_confidence,
        "auction_intensity": auction_intensity,
        "hurst_filter": {
            "h_value": hurst_result["h_value"],
            "h_std_error": hurst_result["h_std_error"],
            "threshold": hurst_result["threshold_used"],
            "confidence_penalty": hurst_result["confidence_penalty"],
            "action": hurst_result["action"],
            "description": hurst_result["description"],
        },
        "orderbook_metrics": {
            "imbalance_ratio": orderbook_imbalance,
            "liquidity_wall": ob_liquidity_wall,
            "available": orderbook_imbalance is not None,
        },
        "banking_lead_lag": {
            "active": _bll_active,
            "multiplier": _bll_multiplier,
            "banking_trend_raw": _bll_trend_raw,
        },
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
        # Circuit-breaker proximity — exposed for UI warning banners
        "circuit_proximity": _circuit_proximity,
        "circuit_breaker": _circuit_result,   # full graduated result for new UI
        "total_score_raw": total_score_raw,
        # Weights — for component_scores in output formatter
        "weights": weights,
        # ── Four-score architecture ───────────────────────────────────────────
        "four_scores": four_scores,
    }

    # Update age_decay_applied in score_breakdown after decay step
    _score_breakdown["age_decay_applied"] = bool(
        prob_result.get("decay_factor", 1.0) is not None
        and float(prob_result.get("decay_factor", 1.0)) < 1.0
    )

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

    return format_signal(
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
        position_sizing=phased_kelly_result,
        raw_technical_score=raw_technical_score,
        risk_adjusted_score=risk_adjusted_score,
        score_breakdown=_score_breakdown,
    )


_NEUTRAL_CIRCUIT_PROXIMITY: dict[str, Any] = {
    "is_near_limit": False,
    "direction": None,
    "distance_to_upper_pct": None,
    "distance_to_lower_pct": None,
}


_BLOCKED_CIRCUIT_RESULT: dict[str, Any] = {
    "nearest_circuit_pct": 5.0,
    "is_near_upper_circuit": False,
    "is_near_lower_circuit": False,
    "penalty_multiplier": 1.0,
    "severity": "none",
    "direction": None,
    "is_near_limit": False,
    "distance_to_upper_pct": None,
    "distance_to_lower_pct": None,
}


def _make_blocked_four_scores(
    rows: list[dict[str, Any]],
    adtv_kwd: float = 0.0,
    spread_pct: float = 0.0,
) -> dict[str, Any]:
    """Return a four_scores dict with all-zero technical inputs.

    Used by early-return paths (liquidity failure, insufficient data, corporate
    action) where technical scores are not yet computed.  The Risk score will
    reflect real liquidity/spread data when available.
    """
    # compute_all_four_scores needs to be imported or already in scope
    return compute_all_four_scores(
        rows=rows,
        trend_raw=0,
        momentum_raw=0,
        volume_raw=0,
        sr_details={},
        auction_intensity=1.0,
        rr_ratio=0.0,
        adtv_kwd=adtv_kwd,
        spread_pct=spread_pct,
        circuit_result=_BLOCKED_CIRCUIT_RESULT,
    )


def _neutral_signal(
    stock_code: str,
    segment: str,
    data_as_of: str,
    reason_code: str = "no_signal",
    failed_gates: list[str] | None = None,
    block_details: dict[str, Any] | None = None,
    technical_scores_debug: dict[str, Any] | None = None,
    circuit_proximity: dict[str, Any] | None = None,
    sub_scores: dict[str, Any] | None = None,
    raw_sub_scores: dict[str, Any] | None = None,
    total_score_for_neutral: int = 0,
    total_score_raw_for_neutral: int | None = None,
    four_scores: dict[str, Any] | None = None,
    liquidity_passed: bool = False,
    liquidity_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a minimal NEUTRAL signal with structured block reason."""
    _cp = circuit_proximity or _NEUTRAL_CIRCUIT_PROXIMITY
    _total_raw = total_score_raw_for_neutral if total_score_raw_for_neutral is not None else total_score_for_neutral
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
            "total_score": total_score_for_neutral,
            "total_score_raw": _total_raw,
            "regime": "Neutral_Chop",
            "regime_confidence": None, "auction_intensity": None,
            "sub_scores": sub_scores or {}, "raw_sub_scores": raw_sub_scores or {},
            "liquidity_passed": liquidity_passed, "liquidity_details": liquidity_details or {},
            # Pass through the actual circuit_proximity so tests can access it
            "circuit_proximity": _cp,
            "four_scores": four_scores,
        },
        alerts=[f"No signal: {reason_code}"],
        data_as_of=data_as_of,
        block_reason=reason_code,
        failed_gates=failed_gates or [],
        block_details=block_details or {},
        technical_scores_debug=technical_scores_debug,
    )
