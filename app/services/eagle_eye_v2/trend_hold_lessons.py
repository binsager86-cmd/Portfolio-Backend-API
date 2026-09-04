"""
Trend-Hold Book -- post-trade lesson analyzer ("trade autopsy").

Classifies each CLOSED leg of a trade (SCALE_OUT / EXIT) using the actual
OHLCV price path between entry and exit -- a small, auditable rules
engine grounded in concrete excursion metrics (MAE/MFE, holding period,
single-session-vs-grinding decline), not a black-box model. Every
classification ships with the exact numbers that produced it, and an
"enhancement" suggestion tied to a real, named engine parameter
(CHANDELIER_ATR_MULT, SCALE_OUT_GAIN_PCT, ...) rather than generic advice.

Deliberately does NOT feed back into trend_hold_engine.py automatically.
With only a handful of closed trades at any given time, auto-tuning a
live parameter from that sample is curve-fitting on noise -- exactly the
overfitting trap this project has already been burned by once (see
trend_hold_engine.py's revision notes). The whole point of the Book is
to accumulate enough real outcomes for a human to decide, with evidence,
whether a parameter genuinely needs to change. This module produces that
evidence; it never acts on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from app.services.eagle_eye_v2.trend_hold_engine import CHANDELIER_ATR_MULT, SCALE_OUT_GAIN_PCT

# ---- Classification thresholds (tunable, documented) ----

QUICK_STOP_MAX_DAYS = 5          # exit within this many sessions of entry -> "failed fast"
GAVE_BACK_THRESHOLD_PCT = 5.0    # peak-to-exit giveback (percentage points) worth flagging
WHIPSAW_SINGLE_SESSION_SHARE = 0.6  # one session's share of total drawdown -> "whipsaw" vs "grind"


@dataclass
class TradeLesson:
    classification: str
    outcome: str  # WIN | LOSS | PARTIAL | UNKNOWN
    mae_pct: Optional[float]
    mfe_pct: Optional[float]
    giveback_pct: Optional[float]
    holding_days: Optional[int]
    reason: str
    enhancement: str


def _window(ohlcv: pd.DataFrame, entry_date: str, exit_date: str) -> pd.DataFrame:
    """Slice ohlcv (DatetimeIndex) to the closed [entry_date, exit_date] session range."""
    if ohlcv is None or ohlcv.empty:
        return ohlcv
    mask = (ohlcv.index >= pd.Timestamp(entry_date)) & (ohlcv.index <= pd.Timestamp(exit_date))
    return ohlcv.loc[mask]


def compute_forward_look(
    ticker: str,
    exit_date: str,
    exit_price: float,
    ohlcv: pd.DataFrame,
    sessions: int = 5,
) -> Optional[dict]:
    """
    Did the stock have more room to run after this system exited it?

    Looks at the *sessions* (default 5 -- one trading week) sessions of real
    OHLCV that came after *exit_date*, plus a longer ~20-session (~1 month)
    window for the "how much further did it run at its best" question --
    "1 week" alone is a single noisy data point, the extended peak gives a
    steadier read on whether the exit was early or justified.

    Returns None when fewer than *sessions* sessions of forward data exist
    yet (the trade closed too recently) -- this is a hard limit, not a bug:
    the answer literally doesn't exist until real time passes. Never raises;
    a missing/malformed ohlcv just means no forward-look, same convention as
    analyze_trade()'s other None-safe paths.
    """
    if ohlcv is None or ohlcv.empty or not exit_date or not exit_price or exit_price <= 0:
        return None

    forward = ohlcv.loc[ohlcv.index > pd.Timestamp(exit_date)].sort_index()
    if len(forward) < sessions:
        return {"available": False, "sessions_available": int(len(forward))}

    price_1w = float(forward["close"].iloc[sessions - 1])
    return_1w_pct = (price_1w / exit_price - 1.0) * 100.0

    extended = forward.iloc[: min(len(forward), 20)]
    peak_extended = float(extended["high"].max())
    peak_extended_pct = (peak_extended / exit_price - 1.0) * 100.0

    return {
        "available": True,
        "sessions_available": int(len(forward)),
        "price_1w": price_1w,
        "return_1w_pct": round(return_1w_pct, 2),
        "peak_20d_pct": round(peak_extended_pct, 2),
    }


def analyze_trade(
    *,
    side: str,  # "SCALE_OUT" or "EXIT"
    entry_date: Optional[str],
    entry_price: Optional[float],
    exit_date: str,
    exit_price: float,
    ohlcv: pd.DataFrame,
    entry_gate: Optional[dict] = None,
) -> TradeLesson:
    """Classify one closed leg of a trade using its realized price path."""
    if not entry_date or not entry_price or entry_price <= 0:
        return TradeLesson(
            classification="UNKNOWN",
            outcome="UNKNOWN",
            mae_pct=None,
            mfe_pct=None,
            giveback_pct=None,
            holding_days=None,
            reason="Missing entry data -- cannot reconstruct this trade's price path.",
            enhancement="No action -- this is a data-completeness gap, not a strategy issue.",
        )

    window = _window(ohlcv, entry_date, exit_date)
    realized_pct = (exit_price / entry_price - 1.0) * 100.0
    holding_days = max(len(window) - 1, 0) if window is not None else None

    mae_pct: Optional[float] = None
    mfe_pct: Optional[float] = None
    giveback_pct: Optional[float] = None
    if window is not None and not window.empty:
        low_since_entry = float(window["low"].min())
        high_since_entry = float(window["high"].max())
        mae_pct = (entry_price - low_since_entry) / entry_price * 100.0
        mfe_pct = (high_since_entry - entry_price) / entry_price * 100.0
        giveback_pct = max(mfe_pct - realized_pct, 0.0)

    if side == "SCALE_OUT":
        return TradeLesson(
            classification="PROFIT_MILESTONE",
            outcome="PARTIAL",
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
            giveback_pct=giveback_pct,
            holding_days=holding_days,
            reason=(
                f"Scaled out at +{realized_pct:.1f}% once the {SCALE_OUT_GAIN_PCT:.0%} gain "
                f"milestone was reached -- the strategy's designed profit-banking step, not a mistake."
            ),
            enhancement=(
                "Working as intended. If scale-outs keep firing well below the eventual peak on "
                "winning trades, that's a signal SCALE_OUT_GAIN_PCT could be raised -- judge that "
                "from the pattern across many trades, not this one alone."
            ),
        )

    # side == "EXIT" -- a full close, win or loss.
    if realized_pct >= 0 and (giveback_pct is None or giveback_pct <= GAVE_BACK_THRESHOLD_PCT):
        classification = "CLEAN_WIN"
        reason = (
            f"Closed at +{realized_pct:.1f}% with minimal giveback from the peak "
            f"({giveback_pct:.1f}pp)."
            if giveback_pct is not None
            else f"Closed at +{realized_pct:.1f}%."
        )
        enhancement = "No change indicated -- the trailing stop captured this trend efficiently."

    elif realized_pct >= 0:
        classification = "GAVE_BACK_GAINS"
        reason = (
            f"Reached +{mfe_pct:.1f}% at the peak but gave back {giveback_pct:.1f}pp before the "
            f"{CHANDELIER_ATR_MULT}x-ATR trailing stop caught it, closing at +{realized_pct:.1f}%."
        )
        enhancement = (
            f"A tighter chandelier multiple than {CHANDELIER_ATR_MULT}x would bank more of the peak "
            f"on winners like this -- at the cost of getting stopped out sooner on trades that pull "
            f"back and then resume. Worth revisiting once several GAVE_BACK_GAINS trades accumulate, "
            f"not on this one alone."
        )

    elif holding_days is not None and holding_days <= QUICK_STOP_MAX_DAYS:
        classification = "QUICK_STOP"
        reason = (
            f"Stopped out {holding_days} session(s) after entry at {realized_pct:.1f}% -- "
            f"the breakout failed almost immediately."
        )
        if entry_gate and entry_gate.get("rel_volume") is not None and entry_gate.get("cmf10") is not None:
            rv = float(entry_gate["rel_volume"])
            rv_floor = float(entry_gate.get("rel_volume_floor") or 0.8)
            cmf10 = float(entry_gate["cmf10"])
            cmf_floor = float(entry_gate.get("cmf_floor") or -0.05)
            cmf_passed = cmf10 >= cmf_floor
            # "Marginal" means a gate barely cleared its own floor -- a
            # large margin means weak entry confirmation ISN'T what
            # explains this quick-stop, and claiming otherwise would
            # misattribute the loss. Only meaningful for a gate that
            # actually passed; a gate that failed (covered by the OTHER
            # condition) gets its own, differently-worded line below.
            rv_marginal = rv < rv_floor * 1.25
            cmf_marginal = cmf_passed and (cmf10 - cmf_floor) < 0.03
            gate_line = (
                f"Entry rel-volume was {rv:.2f} vs the {rv_floor:.2f} floor, and CMF10 was "
                f"{cmf10:.3f} vs the {cmf_floor:.2f} floor"
            )
            if entry_gate.get("flow_pass_via") == "CMF":
                if rv_marginal and cmf_marginal:
                    enhancement = (
                        f"{gate_line} -- both barely cleared, consistent with a weak-confirmation breakout."
                    )
                else:
                    enhancement = (
                        f"{gate_line} -- both well clear of their floors, so this wasn't a marginal-"
                        f"confirmation entry. The quick stop-out more likely reflects a single sharp "
                        f"adverse session than a lenient gate; check the price path itself before "
                        f"touching MIN_REL_VOLUME/CMF_FLOOR for cases like this."
                    )
            else:
                cmf_note = (
                    f"CMF actually failed its own floor by {(cmf_floor - cmf10):.3f} -- confirmation came "
                    f"entirely from OBV slope"
                    if not cmf_passed
                    else "CMF also passed, just not the stronger of the two signals that day"
                )
                enhancement = f"{gate_line} -- {cmf_note}. Worth checking whether OBV-only confirmations underperform CMF-confirmed ones across more trades."
        else:
            enhancement = (
                "A cluster of these suggests the entry's volume/flow confirmation (MIN_REL_VOLUME / "
                "CMF_FLOOR) may be too lenient for this kind of setup -- worth checking whether "
                "quick-stop trades share a common entry pattern."
            )

    else:
        single_session_share: Optional[float] = None
        if window is not None and not window.empty and len(window) >= 2:
            closes = window["close"]
            biggest_one_day_drop = float((closes.shift(1) - closes).clip(lower=0).max() or 0.0)
            total_drawdown = float(window["high"].max() - window["low"].min())
            if total_drawdown > 0:
                single_session_share = biggest_one_day_drop / total_drawdown

        if single_session_share is not None and single_session_share >= WHIPSAW_SINGLE_SESSION_SHARE:
            classification = "WHIPSAW_EXIT"
            reason = (
                f"Closed at {realized_pct:.1f}% after {holding_days} sessions -- most of the "
                f"drawdown happened in a single sharp session rather than a grinding decline."
            )
            enhancement = (
                f"A wider stop survives sessions like this, but also gives back more on genuine "
                f"reversals -- a real tradeoff, not a free fix. Track whether WHIPSAW_EXIT trades "
                f"cluster around news/earnings dates before touching {CHANDELIER_ATR_MULT}x."
            )
        else:
            classification = "TREND_REVERSAL"
            reason = (
                f"Closed at {realized_pct:.1f}% after a {holding_days}-session grinding decline -- "
                f"the trailing stop reacted to a genuine trend break, not noise."
            )
            enhancement = "No change indicated -- this is the stop doing its job correctly."

    return TradeLesson(
        classification=classification,
        outcome="WIN" if realized_pct >= 0 else "LOSS",
        mae_pct=mae_pct,
        mfe_pct=mfe_pct,
        giveback_pct=giveback_pct,
        holding_days=holding_days,
        reason=reason,
        enhancement=enhancement,
    )
