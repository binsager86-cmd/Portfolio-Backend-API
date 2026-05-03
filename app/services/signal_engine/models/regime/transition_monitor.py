"""Regime transition monitoring and alerting.

Detects recent regime shifts and generates human-readable alerts for the
signal output.  No stochastic state — works on the sequence of regime
predictions derived from recent bars.
"""
from __future__ import annotations

from typing import Any

from app.services.signal_engine.config.model_params import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_CHOP,
)


def detect_transition_alerts(
    current_result: dict[str, Any],
    historical_regimes: list[str] | None = None,
) -> list[str]:
    """Generate plain-language alerts based on regime state and confidence.

    Args:
        current_result: Output from predict_regime().
        historical_regimes: Optional ordered list of recent regime labels
                            (oldest first) for streak / transition analysis.

    Returns:
        List of alert strings (empty if no notable events).
    """
    alerts: list[str] = []
    regime = current_result.get("current_regime", REGIME_CHOP)
    confidence = float(current_result.get("regime_confidence") or 0.0)
    days_in = int(current_result.get("days_in_current_regime") or 0)

    # ── Low confidence warning ────────────────────────────────────────────────
    if confidence < 0.60:
        alerts.append(
            f"Regime confidence low ({confidence:.0%}) — potential regime transition in progress"
        )

    # ── Regime change alert ───────────────────────────────────────────────────
    if historical_regimes and len(historical_regimes) >= 2:
        prev = historical_regimes[-2]
        if prev != regime:
            alerts.append(
                f"Regime shift detected: {prev} → {regime} (day {days_in} of new regime)"
            )

    # ── Extended chop warning ────────────────────────────────────────────────
    if regime == REGIME_CHOP and days_in >= 10:
        alerts.append(
            f"Extended neutral/chop regime ({days_in} days) — momentum signals unreliable"
        )

    # ── Bear regime caution ───────────────────────────────────────────────────
    if regime == REGIME_BEAR:
        alerts.append("Bear-regime active: only counter-trend SELL setups or cash preferred")

    # ── Bull regime confirmation ──────────────────────────────────────────────
    if regime == REGIME_BULL and days_in >= 5 and confidence >= 0.75:
        alerts.append(
            f"Bull-regime confirmed ({days_in} days, {confidence:.0%} confidence) — trend-following favoured"
        )

    return alerts


def get_regime_weight_adjustment(regime: str) -> dict[str, float]:
    """Return multiplicative weight adjustments per regime from model_params spec.

    Neutral_Chop:     momentum × 0.5, support_resistance × 1.5
    Bearish:          volume_flow × 1.2, trend × 0.8
    Bullish:          no adjustment (all × 1.0)

    Returns:
        Dict mapping component name → multiplier.
    """
    adjustments: dict[str, float] = {
        "trend":              1.0,
        "momentum":           1.0,
        "volume_flow":        1.0,
        "support_resistance": 1.0,
        "risk_reward":        1.0,
    }
    if regime == REGIME_CHOP:
        adjustments["momentum"] = 0.50
        adjustments["support_resistance"] = 1.50
    elif regime == REGIME_BEAR:
        adjustments["volume_flow"] = 1.20
        adjustments["trend"] = 0.80
    return adjustments
