"""Time-based signal confidence decay for the Kuwait Signal Engine.

Signals degrade over time as market conditions evolve.  The decay schedule
is defined in risk_config.DECAY_SCHEDULE.

T+0 h:  100 % confidence
T+24 h:  85 % confidence
T+48 h:  65 % confidence
T+72 h:   0 % (invalidated — require fresh confirmation candle)
"""
from __future__ import annotations

from app.services.signal_engine.config.risk_config import DECAY_SCHEDULE


def get_decay_factor(hours_since_generation: int) -> float:
    """Return the confidence multiplier for the given staleness.

    Linearly interpolates between schedule breakpoints.
    Returns 0.0 if the signal is invalidated (≥ 72 h).

    Args:
        hours_since_generation: Non-negative integer number of hours.

    Returns:
        Float multiplier in [0.0, 1.0].
    """
    h = max(0, hours_since_generation)
    checkpoints = sorted(DECAY_SCHEDULE.keys())

    if h == 0:
        return DECAY_SCHEDULE[0]

    # Check for invalidation
    max_h = max(checkpoints)
    if h >= max_h:
        return 0.0

    # Linear interpolation between adjacent checkpoints
    for i in range(len(checkpoints) - 1):
        t0, t1 = checkpoints[i], checkpoints[i + 1]
        if t0 <= h < t1:
            f0 = DECAY_SCHEDULE[t0]
            f1 = DECAY_SCHEDULE[t1]
            frac = (h - t0) / (t1 - t0)
            return round(f0 + (f1 - f0) * frac, 4)

    return DECAY_SCHEDULE[max_h]


def adjust_confidence_for_delay(
    probabilities: dict[str, float],
    hours_since_generation: int,
) -> dict[str, float]:
    """Multiply all probability values in the dict by the decay factor.

    Modifies:
      p_tp1_before_sl
      p_tp2_before_sl

    Args:
        probabilities: Dict from probability_calibrator output.
        hours_since_generation: Non-negative integer hours since signal was
                                first generated.

    Returns:
        New dict with decayed probabilities.  Adds 'decay_factor' key.
    """
    factor = get_decay_factor(hours_since_generation)
    out = dict(probabilities)
    for key in ("p_tp1_before_sl", "p_tp2_before_sl"):
        if key in out and out[key] is not None:
            out[key] = round(float(out[key]) * factor, 3)
    out["decay_factor"] = factor
    out["hours_since_generation"] = hours_since_generation
    return out
