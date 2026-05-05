"""Four-Score Architecture for Kuwait Signal Engine.

Implements: Potential, Timing, Risk, Overall with 5-tier classification.

Tier thresholds (shared across all four scores):
  >= 85 → "Strong Buy"
  >= 70 → "Buy"
  >= 40 → "Hold"
  >= 15 → "Sell"
   < 15 → "Strong Sell"

Weights:
  Potential: Trend 40% | Momentum 25% | Volume Flow 35%
  Timing:    S/R Proximity 35% | Resistance Clearance 20% | Volume POC 30% | Auction 15%
  Risk:      RR 40% | Volatility 25% | Liquidity 20% | Circuit 15%
  Overall:   Potential 50% | Timing 50%  (risk-gated + risk multiplier)
"""
from __future__ import annotations

from typing import Any


# ── Tier Classifier (Shared) ─────────────────────────────────────────────────

def _classify_tier(score: int) -> tuple[str, str]:
    if score >= 85:
        return "Strong Buy", "maximum_conviction_aligned"
    if score >= 70:
        return "Buy", "good_setup_acceptable_edge"
    if score >= 40:
        return "Hold", "neutral_mixed_signals_wait"
    if score >= 15:
        return "Sell", "weak_setup_avoid_or_reduce"
    return "Strong Sell", "dangerous_no_edge_block"


# ── 1. POTENTIAL SCORE (Directional Upside) ───────────────────────────────────
# Weights: Trend 40% | Momentum 25% | Volume Flow 35%

def compute_potential_score(
    trend_raw: int,
    momentum_raw: int,
    volume_raw: int,
) -> tuple[int, str, str]:
    """Directional upside: Trend (40%) + Momentum (25%) + Volume Flow (35%).

    Returns:
        (score 0-100, tier, description)
    """
    raw = int((trend_raw * 0.40) + (momentum_raw * 0.25) + (volume_raw * 0.35))
    raw = max(0, min(100, raw))
    tier, desc = _classify_tier(raw)
    return raw, tier, desc


# ── 2. TIMING SCORE (Entry Quality) ─────────────────────────────────────────
# Weights: S/R Proximity 35% | Resistance Clearance 20% | Volume POC 30% | Auction 15%

def compute_timing_score(
    sr_details: dict[str, Any],
    auction_intensity: float,
    close: float,
    atr_14: float,
    atr_60: float | None = None,
) -> tuple[int, str, str]:
    """Entry quality: S/R proximity, resistance clearance, POC distance, auction.

    Args:
        sr_details:        Details dict from compute_sr_score() — must contain
                           ``support_proximity_pts`` (max 40),
                           ``resistance_clearance_pts`` (max 35),
                           ``volume_poc`` (price or None).
        auction_intensity: Auction intensity proxy, typical range 0.5–2.5.
        close:             Current close price in fils.
        atr_14:            14-period ATR in fils.
        atr_60:            60-period ATR in fils (optional; falls back to atr_14).

    Returns:
        (score 0-100, tier, description)
    """
    # Normalize S/R sub-scores to 0-100 scale
    sr_prox_norm   = (sr_details.get("support_proximity_pts", 0) / 40.0) * 100
    res_clear_norm = (sr_details.get("resistance_clearance_pts", 0) / 35.0) * 100

    # Volume POC distance → discrete score
    poc = sr_details.get("volume_poc")
    poc_dist = abs(poc - close) / close if poc and close > 0 else 1.0
    poc_pts = (
        100 if poc_dist <= 0.02 else
        70  if poc_dist <= 0.05 else
        40  if poc_dist <= 0.10 else
        15
    )

    # Auction intensity → 0-100  (0.5 base, 2.5 ceiling)
    auc_norm = min(100, max(0, (auction_intensity - 0.5) / 2.0 * 100))

    # Volatility compression (computed for reference; not in weighted sum per spec)
    _atr60 = atr_60 if atr_60 and atr_60 > 0 else atr_14
    comp_ratio = atr_14 / _atr60 if _atr60 > 0 else 1.0
    _comp_pts = (  # noqa: F841
        100 if comp_ratio < 0.7 else
        75  if comp_ratio < 0.9 else
        50  if comp_ratio < 1.1 else
        25
    )

    raw = int(
        (sr_prox_norm   * 0.35) +
        (res_clear_norm * 0.20) +
        (poc_pts        * 0.30) +
        (auc_norm       * 0.15)
    )
    raw = max(0, min(100, raw))
    tier, desc = _classify_tier(raw)
    return raw, tier, desc


# ── 3. RISK SCORE (Downside Protection & Gate) ───────────────────────────────
# Weights: RR 40% | Volatility 25% | Liquidity 20% | Circuit 15%

def compute_risk_score(
    rr_ratio: float,
    atr_pct: float,
    adtv_kwd: float,
    spread_pct: float,
    circuit_distance_pct: float,
) -> tuple[int, str, str, bool, str]:
    """Downside protection: RR (40%) + Volatility (25%) + Liquidity (20%) + Circuit (15%).

    Args:
        rr_ratio:             Risk-reward ratio (e.g. 2.5).
        atr_pct:              ATR(14) as a percentage of close (e.g. 1.8 for 1.8%).
        adtv_kwd:             Average daily traded value in KWD.
        spread_pct:           Bid-ask spread proxy as % (e.g. 1.2).
        circuit_distance_pct: Distance to nearest circuit limit in % of price.

    Returns:
        (score 0-100, tier, description, is_blocked, block_reason)
    """
    # RR (non-linear)
    rr_pts = (
        0   if rr_ratio < 1.5 else
        30  if rr_ratio < 2.0 else
        60  if rr_ratio < 2.5 else
        80  if rr_ratio < 3.0 else
        100
    )

    # Volatility (lower ATR% = better)
    vol_pts = (
        100 if atr_pct < 1.0 else
        80  if atr_pct < 2.0 else
        60  if atr_pct < 3.0 else
        40  if atr_pct < 5.0 else
        20
    )

    # Liquidity
    liq_pts = 100
    if adtv_kwd < 100_000:
        liq_pts -= 50
    elif adtv_kwd < 200_000:
        liq_pts -= 20
    if spread_pct > 1.5:
        liq_pts -= 30
    elif spread_pct > 1.0:
        liq_pts -= 10
    liq_pts = max(0, liq_pts)

    # Circuit proximity
    circuit_pts = (
        100 if circuit_distance_pct > 2.0 else
        85  if circuit_distance_pct > 1.0 else
        60  if circuit_distance_pct > 0.5 else
        30
    )

    raw = int(
        (rr_pts      * 0.40) +
        (vol_pts     * 0.25) +
        (liq_pts     * 0.20) +
        (circuit_pts * 0.15)
    )
    raw = max(0, min(100, raw))
    tier, desc = _classify_tier(raw)

    is_blocked = tier in ("Sell", "Strong Sell")
    block_reason = "risk_gate_triggered" if is_blocked else ""

    return raw, tier, desc, is_blocked, block_reason


# ── 4. OVERALL SCORE (Composite Decision) ────────────────────────────────────

def compute_overall_score(
    potential_raw: int,
    timing_raw: int,
    risk_raw: int,
    risk_tier: str,
) -> tuple[int, str, str, float]:
    """Risk-gated composite: Potential (50%) + Timing (50%) × risk multiplier.

    Returns:
        (score 0-100, tier, description, risk_multiplier)
    """
    # Hard gate
    if risk_tier in ("Sell", "Strong Sell"):
        return 0, "Strong Sell", "blocked_by_risk_gate", 0.0

    base_confluence = (potential_raw * 0.50) + (timing_raw * 0.50)

    # Risk multiplier — any risk_raw < 40 maps to Sell/Strong Sell and is
    # already caught by the hard gate above, so mult=0.0 is a safety net only.
    if risk_raw >= 85:
        mult = 1.00
    elif risk_raw >= 70:
        mult = 0.90
    elif risk_raw >= 60:
        mult = 0.75
    elif risk_raw >= 40:
        mult = 0.50
    else:  # pragma: no cover  — hard gate above blocks risk_raw < 40
        mult = 0.0

    raw = int(base_confluence * mult)
    raw = max(0, min(100, raw))
    tier, desc = _classify_tier(raw)
    return raw, tier, desc, mult


# ── DECISION MATRIX → position action ────────────────────────────────────────

_TIER_RANK: dict[str, int] = {
    "Strong Buy": 5, "Buy": 4, "Hold": 3, "Sell": 2, "Strong Sell": 1,
}


def compute_position_action(
    potential_tier: str,
    timing_tier: str,
    risk_tier: str,
    overall_tier: str,
) -> dict[str, Any]:
    """Map the four tiers to a position action and max position size."""
    if risk_tier == "Strong Sell":
        return {"action": "BLOCK_DANGEROUS", "label": "BLOCK — Dangerous", "max_position_pct": 0.0}

    if risk_tier == "Sell":
        return {"action": "BLOCK_DO_NOT_TRADE", "label": "BLOCK — Do not trade", "max_position_pct": 0.0}

    if all(t == "Strong Buy" for t in (potential_tier, timing_tier, risk_tier, overall_tier)):
        return {"action": "MAXIMUM_SIZE", "label": "Maximum size (2.5%)", "max_position_pct": 2.5}

    if all(t == "Buy" for t in (potential_tier, timing_tier, risk_tier, overall_tier)):
        return {"action": "STANDARD_SIZE", "label": "Standard size (1.5%)", "max_position_pct": 1.5}

    if (potential_tier == "Buy" and timing_tier == "Hold"
            and risk_tier == "Buy" and overall_tier == "Hold"):
        return {"action": "REDUCE_SIZE", "label": "Reduce size (0.75%)", "max_position_pct": 0.75}

    if (potential_tier == "Hold" and timing_tier == "Buy"
            and risk_tier == "Buy" and overall_tier == "Hold"):
        return {"action": "WAIT_PULLBACK", "label": "Wait for pullback", "max_position_pct": 0.0}

    if (potential_tier == "Buy" and timing_tier == "Buy"
            and risk_tier == "Hold" and overall_tier == "Hold"):
        return {"action": "WAIT_BETTER_RR", "label": "Wait for better RR", "max_position_pct": 0.0}

    overall_rank = _TIER_RANK.get(overall_tier, 0)
    if overall_rank >= 4:
        return {"action": "STANDARD_SIZE", "label": "Standard size (1.5%)", "max_position_pct": 1.5}
    if overall_rank == 3:
        return {"action": "REDUCE_SIZE", "label": "Reduce size (0.5%)", "max_position_pct": 0.5}

    return {"action": "NO_TRADE", "label": "No trade — no edge", "max_position_pct": 0.0}


# ── HELPER: Simple ATR ────────────────────────────────────────────────────────

def _simple_atr(rows: list[dict], period: int) -> float | None:
    """Simple average true range over the last ``period`` bars."""
    if len(rows) < period + 1:
        return None
    tr_values: list[float] = []
    for i in range(1, len(rows)):
        high = float(rows[i].get("high") or 0.0)
        low  = float(rows[i].get("low")  or 0.0)
        prev = float(rows[i - 1].get("close") or 0.0)
        if prev <= 0:
            continue
        tr_values.append(max(high - low, abs(high - prev), abs(low - prev)))
    if len(tr_values) < period:
        return None
    return sum(tr_values[-period:]) / period


# ── CONVENIENCE WRAPPER ───────────────────────────────────────────────────────

def compute_all_four_scores(
    rows: list[dict],
    trend_raw: int,
    momentum_raw: int,
    volume_raw: int,
    sr_details: dict[str, Any],
    auction_intensity: float,
    rr_ratio: float,
    adtv_kwd: float,
    spread_pct: float,
    circuit_result: dict[str, Any],
) -> dict[str, Any]:
    """Compute all four scores and the decision-matrix position action.

    Returns a structured dict with keys:
    ``potential``, ``timing``, ``risk``, ``overall``, ``position_action``.
    """
    close = float(rows[-1].get("close") or 1.0) if rows else 1.0

    # ATR inputs for timing score
    atr_14_col = rows[-1].get("atr_14") if rows else None
    atr_14 = float(atr_14_col) if atr_14_col else (_simple_atr(rows, 14) or close * 0.015)
    atr_60 = _simple_atr(rows, 60)  # None if insufficient history → falls back inside timing

    # ATR as % of price for risk score
    atr_pct = (atr_14 / close * 100.0) if close > 0 else 2.0

    # Circuit distance
    circuit_dist = max(0.0, float(circuit_result.get("nearest_circuit_pct") or 5.0))

    # ── Compute ───────────────────────────────────────────────────────────────
    pot_score, pot_tier, pot_desc = compute_potential_score(trend_raw, momentum_raw, volume_raw)
    tim_score, tim_tier, tim_desc = compute_timing_score(
        sr_details, auction_intensity, close, atr_14, atr_60
    )
    risk_score, risk_tier, risk_desc, risk_blocked, risk_reason = compute_risk_score(
        rr_ratio, atr_pct, adtv_kwd, spread_pct, circuit_dist
    )
    ov_score, ov_tier, ov_desc, risk_mult = compute_overall_score(
        pot_score, tim_score, risk_score, risk_tier
    )
    action = compute_position_action(pot_tier, tim_tier, risk_tier, ov_tier)

    return {
        "potential": {
            "score": pot_score,
            "tier": pot_tier,
            "description": pot_desc,
        },
        "timing": {
            "score": tim_score,
            "tier": tim_tier,
            "description": tim_desc,
        },
        "risk": {
            "score": risk_score,
            "tier": risk_tier,
            "description": risk_desc,
            "is_blocked": risk_blocked,
            "block_reason": risk_reason,
        },
        "overall": {
            "score": ov_score,
            "tier": ov_tier,
            "description": ov_desc,
            "risk_multiplier": risk_mult,
        },
        "position_action": action,
    }
