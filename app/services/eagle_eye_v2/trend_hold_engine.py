from __future__ import annotations

"""Trend entry/hold engine.

Replaces the broken V2 decision path (adaptive_base_geometry.py +
flow_confirmation_engine.py + lifecycle_intent_router.py) for the specific job
the user needs: detect a breakout from a real base, buy it, and then HOLD
through the trend using a trailing stop -- not a daily re-audition of strict
entry-grade oscillator gates.

Why not patch the old gates instead: their bug is structural, not a typo.
CONFIRM_CHASE_GUARD_OK requires base_validity_state == "VALID" every single
day, and AdaptiveBaseGeometry retires (nulls) the base once price extends far
enough above it ("RETIRED_SUPERSEDED_BY_MARKUP"), which is exactly when a
trend accelerates -- so the system loses its own hold authority right when it
should be holding hardest, and can't re-freeze a new base under trending
(non-squeeze) volatility. On top of that there is no persistent position
state anywhere in eagle_eye_v2 -- every day recomputes ADX/RSI/CMF/structure
from scratch, so a normal mid-trend pullback reads as "failed." Two separate
concerns (strict entry filter vs. lenient hold/trail) were being answered by
the same strict function. This module splits them.

Entry: Donchian-style lookback breakout (close makes a fresh
DONCHIAN_LOOKBACK_SESSIONS high) plus a light trend/flow sanity check.
Hold: chandelier trailing stop (highest close since entry, minus N x ATR14)
is the ONLY exit trigger. The stop only ever ratchets up. Position is held
for as long as price stays above it -- which is what "hold until it fails"
means mechanically.

Revision note (v2): the first version required a TIGHT 15-session box
(<=12% wide) immediately before the breakout. That was wrong -- real cycles
in this data launch out of long, often choppy multi-month bases with a wide
-range thrust candle, and that thrust candle itself blows out a 15-session
width filter the moment it fires, forcing the engine to wait for a second,
calmer mini-base further up the trend before it would enter (confirmed on
replay: it missed the first ~40-50 points of both the ZAIN and BPCC moves).
Requiring "calm" as an entry precondition is the same mistake the original
V2 base-geometry gate made, just reintroduced in a milder form. A Donchian
breakout has no shape precondition -- it only asks whether today cleared the
ceiling of however-long the preceding range was, tight or not -- so it
catches the actual origin of the cycle instead of a second, later leg of it.

Revision note (v3): v2 also used AvoidAuthorityPlane (SMA200 regime) as a
second, independent exit trigger. Dropped -- it is a lagging state machine
(needs a 2-session SMA200 reclaim, or 20 clear sessions, to clear its own
AVOID phase) and it stays "active" for a session or two right after a
legitimate breakout, when price is still below a still-declining SMA200.
On replay this fired five same-day-or-next-day exits immediately after
entry (e.g. BPCC bought 2026-04-13, stopped out 2026-04-14 by this rule
alone, re-bought 2026-04-15) -- contradicting the entry it had just
approved. The chandelier stop is already a price-driven trend-break
detector and doesn't have this lag; one exit mechanism, doing its job.

Revision note (v4): the Donchian breakout alone still enters later than the
trend visibly turns, because it waits for the ceiling of the PRIOR range to
be reclaimed rather than reacting to the turn itself. Added a second, faster
entry path: an EMA10-crosses-above-EMA30 bullish crossover, gated by
close > EMA50 so it only fires once price has already reclaimed the
intermediate trend (not a dead-cat bounce mid-decline). Entry fires on
EITHER path, whichever comes first. Tested against the full ZAIN/BPCC
history: this closed the BPCC entry gap from 603 to 575 (the EMA path fired
5 weeks earlier, right at the base breakout) with no change to ZAIN's entry
and no material increase in whipsaw severity -- a pure EMA10/30-crossover
entry on its own was tested and rejected: on ZAIN alone it was noisy enough
to fully exit and miss the current trend altogether, so it is used only as
an addition to the Donchian path, never a replacement for it. An EMA-based
alternative was also tried for the EXIT side (trailing on EMA30, or
requiring both EMA and chandelier to break) -- on this data the chandelier
stop was never the looser constraint during either live trade, so it made
no difference and was left out to keep the exit to one mechanism.

Revision note (v5): tried to make the single exit "smarter" about blow-off
tops (MABANEE peaked at 1149 on 2025-11-26 and the chandelier didn't confirm
until 1050 on 2025-12-04, giving back ~8.6%). Two indicator-based triggers
were tested and rejected: RSI14 >= 80 as a tightening signal, and a 3-day
parabolic-return threshold. Both fail the same way -- MABANEE's RSI first
crossed 80 on 2025-08-24 at 948, three months before the real top, during
what was just a healthy trend (a well-known trap: RSI overbought does not
mean "sell" in a strong trend). Using either as a sticky tighten-the-stop
trigger fragmented the single +15% trade into two smaller, worse ones. No
single indicator reliably separates "healthy strength" from "the real top"
in real time, so stop tightening it is. Instead, added a profit-milestone
partial scale-out: once unrealized gain from entry reaches SCALE_OUT_GAIN_PCT,
sell SCALE_OUT_FRACTION of the position (once per trade) and let the
remainder keep running on the untouched, already-validated chandelier stop.
This sidesteps the classification problem entirely -- it doesn't need to
guess whether any given extension is the top -- and on replay the milestone
happened to land exactly on MABANEE's peak session (2025-11-26, +25.8%),
banking half the position at the top while the runner still captures
whatever the trend does next.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from app.services.eagle_eye.indicators import adx, atr, cmf, ema, obv_slope, relative_volume, rsi

# ---- Tunable parameters (single source of truth for this engine) ----

DONCHIAN_LOOKBACK_SESSIONS = 40    # ~2 trading months; entry = fresh high over this window
CHANDELIER_ATR_MULT = 3.5          # trailing-stop distance, in ATR14s, off the post-entry high
MIN_REL_VOLUME = 0.8               # entry day just needs to not be a dead/no-volume day
CMF_FLOOR = -0.05                  # entry flow filter is lenient: "not distributing," not "must accumulate"
SCALE_OUT_GAIN_PCT = 0.20          # bank profit once unrealized gain from entry reaches this
SCALE_OUT_FRACTION = 0.5           # fraction of the position sold at that milestone (once per trade)


# ---- Signal-strength scoring (informational only) ----
#
# BUY and SELL_SIGNAL each get a 0-100 "confidence" score, built purely
# from the exact inputs that fired that decision. This score NEVER feeds
# back into the entry/exit rules above -- it is computed strictly after
# the decision already fired, as a diagnostic annotation for the trade
# log so a person can see how decisive a given signal was. Deliberately
# NOT computed for HOLD/WAIT (nothing decisive happened) or SCALE_OUT
# (a fixed profit-milestone rule, not a judged signal -- see
# _scale_out_confidence's docstring below for why "confidence" isn't a
# meaningful concept there).

def _clamp01_100(x: float) -> float:
    return max(0.0, min(100.0, x))


@dataclass
class EntryConfidence:
    """Blended 0-100 score plus the three sub-scores it's built from, and the
    raw breakout-margin % -- kept separate from the blended total so a
    trade-log consumer can see *which* leg of the entry was strong/weak,
    not just the combined number."""
    total: float
    breakout_score: float
    volume_score: float
    flow_score: float
    breakout_margin_pct: float


def _entry_confidence(
    donchian_fire: bool,
    close: float,
    donchian_high: float,
    ema50: float,
    rel_volume: float,
    cmf10: float,
    obv_slope40: float,
) -> EntryConfidence:
    """
    0-100 BUY signal strength: breakout strength (50%) + volume
    confirmation (25%) + flow confirmation (25%) -- the same three gates
    replay_symbol()'s entry_fire check already evaluates, scored instead
    of just pass/failed.
    """
    # Breakout strength: how decisively price cleared the trigger level
    # (the Donchian ceiling on that path, else the EMA50 reclaim level).
    if donchian_fire and pd.notna(donchian_high) and donchian_high > 0:
        excess_pct = (close - float(donchian_high)) / float(donchian_high) * 100.0
    elif pd.notna(ema50) and ema50 > 0:
        excess_pct = (close - float(ema50)) / float(ema50) * 100.0
    else:
        excess_pct = 0.0
    breakout_score = _clamp01_100(50.0 + excess_pct * 10.0)

    # Volume confirmation, relative to the MIN_REL_VOLUME pass threshold.
    if pd.isna(rel_volume):
        volume_score = 50.0  # entry_fire itself treats missing volume as neutral/pass
    else:
        volume_score = _clamp01_100((float(rel_volume) / MIN_REL_VOLUME) * 50.0)

    # Flow confirmation: entry_fire passes on EITHER condition, so score
    # is the stronger of the two independent pieces of evidence.
    cmf_score = _clamp01_100(50.0 + (float(cmf10) - CMF_FLOOR) * 250.0) if pd.notna(cmf10) else 0.0
    obv_score = 65.0 if (pd.notna(obv_slope40) and obv_slope40 > 0) else 0.0
    flow_score = max(cmf_score, obv_score)

    total = round(_clamp01_100(breakout_score * 0.5 + volume_score * 0.25 + flow_score * 0.25), 1)
    return EntryConfidence(
        total=total,
        breakout_score=round(breakout_score, 1),
        volume_score=round(volume_score, 1),
        flow_score=round(flow_score, 1),
        breakout_margin_pct=round(excess_pct, 2),
    )


def _exit_confidence(close: float, structural_stop: float) -> float:
    """0-100 SELL_SIGNAL strength: how decisively price closed below the trailing stop."""
    if not structural_stop or structural_stop <= 0:
        return 50.0
    overshoot_pct = (float(structural_stop) - close) / float(structural_stop) * 100.0
    return round(_clamp01_100(50.0 + overshoot_pct * 16.67), 1)


@dataclass
class TrendHoldState:
    position_state: str = "NO_POSITION"  # NO_POSITION | IN_POSITION | EXIT_SIGNAL
    entry_date: Optional[str] = None
    entry_price: Optional[float] = None
    base_high: Optional[float] = None
    highest_close_since_entry: Optional[float] = None
    structural_stop: Optional[float] = None
    exit_reason: Optional[str] = None
    position_fraction: float = 1.0   # remaining position size, 1.0 = full
    scaled_out: bool = False         # whether the profit-milestone scale-out already fired
    peak_date: Optional[str] = None  # session that set highest_close_since_entry (for exit_gate's days_since_peak)
    entry_gate: Optional[dict] = None  # full gate snapshot captured on entry, carried for the life of the trade


def _build_entry_gate(
    *,
    donchian_fire: bool,
    donchian_high: float,
    ema10: float,
    ema30: float,
    ema50: float,
    rel_volume: float,
    cmf10: float,
    obv_slope40: float,
    adx14: float,
    sma200: float,
    sma200_slope: float,
    atr14: float,
    conf: EntryConfidence,
) -> dict:
    """Structured record of every gate input that decided a BUY -- what
    actually fired it, not just the human-readable reason string. Persisted
    verbatim (as JSON) so the Lessons Learned report can show the exact
    numbers instead of a re-derived approximation."""
    cmf_pass = bool(pd.notna(cmf10) and float(cmf10) >= CMF_FLOOR)
    obv_pass = bool(pd.notna(obv_slope40) and float(obv_slope40) > 0)
    flow_pass_via = "CMF+OBV" if (cmf_pass and obv_pass) else ("CMF" if cmf_pass else ("OBV" if obv_pass else None))
    return {
        "entry_path": "DONCHIAN" if donchian_fire else "EMA_CROSS",
        "donchian_high": float(donchian_high) if pd.notna(donchian_high) else None,
        "ema10": float(ema10) if pd.notna(ema10) else None,
        "ema30": float(ema30) if pd.notna(ema30) else None,
        "ema50": float(ema50) if pd.notna(ema50) else None,
        "rel_volume": float(rel_volume) if pd.notna(rel_volume) else None,
        "rel_volume_floor": MIN_REL_VOLUME,
        "cmf10": float(cmf10) if pd.notna(cmf10) else None,
        "cmf_floor": CMF_FLOOR,
        "obv_slope40": float(obv_slope40) if pd.notna(obv_slope40) else None,
        "flow_pass_via": flow_pass_via,
        "adx14": float(adx14) if pd.notna(adx14) else None,
        "sma200": float(sma200) if pd.notna(sma200) else None,
        "sma200_slope": float(sma200_slope) if pd.notna(sma200_slope) else None,
        "atr14": float(atr14) if pd.notna(atr14) else None,
        "breakout_margin_pct": conf.breakout_margin_pct,
        "confidence": conf.total,
        "confidence_breakdown": {
            "breakout_score": conf.breakout_score,
            "volume_score": conf.volume_score,
            "flow_score": conf.flow_score,
        },
    }


def _build_exit_gate(
    *,
    trigger: str,
    structural_stop: float,
    highest_close_since_entry: float,
    peak_date: Optional[str],
    exit_date: str,
    atr14: float,
    adx14: float,
) -> dict:
    """Structured record of what fired a SELL_SIGNAL/SCALE_OUT -- the stop
    level and volatility/regime context at that moment, plus how many
    sessions passed between the trade's peak and the exit (a direct measure
    of how much the fixed chandelier multiple lagged the actual top)."""
    days_since_peak: Optional[int] = None
    if peak_date:
        try:
            days_since_peak = (pd.Timestamp(exit_date) - pd.Timestamp(peak_date)).days
        except (TypeError, ValueError):
            days_since_peak = None
    return {
        "trigger": trigger,  # "CHANDELIER_STOP" | "SCALE_OUT_MILESTONE"
        "structural_stop": float(structural_stop) if structural_stop is not None else None,
        "highest_close_since_entry": float(highest_close_since_entry) if highest_close_since_entry is not None else None,
        "peak_date": peak_date,
        "days_since_peak": days_since_peak,
        "atr14": float(atr14) if pd.notna(atr14) else None,
        "adx14": float(adx14) if pd.notna(adx14) else None,
    }


def compute_daily_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """ohlcv must have columns: trade_date, open, high, low, close, volume, value_kwd."""
    df = ohlcv.sort_values("trade_date").reset_index(drop=True).copy()

    df["ema10"] = ema(df, 10)
    df["ema30"] = ema(df, 30)
    df["ema50"] = ema(df, 50)
    df["sma200"] = df["close"].rolling(200, min_periods=200).mean()
    df["sma200_slope"] = df["sma200"].diff(10)
    df["atr14"] = atr(df, 14)
    df["rsi14"] = rsi(df, 14)
    adx14, _plus_di, _minus_di = adx(df, 14)
    df["adx14"] = adx14
    df["cmf10"] = cmf(df, 10)
    df["obv_slope40"] = obv_slope(df, 40)
    df["rel_volume"] = relative_volume(df, 20)

    # Donchian ceiling: highest HIGH over the prior N sessions, strictly excluding
    # today (shift(1)) -- "today closed above however-long the base has been."
    df["donchian_high"] = df["high"].rolling(DONCHIAN_LOOKBACK_SESSIONS).max().shift(1)

    # Faster entry path: the EMA10/EMA30 bullish crossover EVENT (not just the
    # state), gated by close > EMA50 downstream so it only counts once the
    # intermediate trend has actually turned.
    ema10_above = df["ema10"] > df["ema30"]
    df["ema_cross_up"] = ema10_above & ~ema10_above.shift(1, fill_value=False)
    return df


def replay_symbol(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Day-by-day BUY / HOLD / SELL_SIGNAL / WAIT decisions. df must already
    have compute_daily_features() applied."""
    state = TrendHoldState()
    results: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        close = float(row["close"])
        decision = "WAIT"
        reason = "no qualifying breakout"
        confidence: Optional[float] = None  # only scored for BUY / SELL_SIGNAL -- see module notes above
        gate_snapshot: Optional[dict] = None  # entry_gate on BUY, exit_gate on SELL_SIGNAL/SCALE_OUT

        if state.position_state != "IN_POSITION":
            donchian_high = row["donchian_high"]
            donchian_fire = bool(pd.notna(donchian_high) and close > float(donchian_high))
            ema_cross_fire = bool(
                row["ema_cross_up"] and pd.notna(row["ema50"]) and close > float(row["ema50"])
            )
            flow_ok = bool(
                (pd.notna(row["cmf10"]) and row["cmf10"] >= CMF_FLOOR)
                or (pd.notna(row["obv_slope40"]) and row["obv_slope40"] > 0)
            )
            vol_ok = bool(pd.isna(row["rel_volume"]) or row["rel_volume"] >= MIN_REL_VOLUME)

            entry_fire = bool((donchian_fire or ema_cross_fire) and flow_ok and vol_ok)

            if entry_fire:
                atr_val = float(row["atr14"]) if pd.notna(row["atr14"]) else 0.0
                trade_date_str = str(row["trade_date"])
                conf = _entry_confidence(
                    donchian_fire=donchian_fire,
                    close=close,
                    donchian_high=donchian_high,
                    ema50=row["ema50"],
                    rel_volume=row["rel_volume"],
                    cmf10=row["cmf10"],
                    obv_slope40=row["obv_slope40"],
                )
                entry_gate = _build_entry_gate(
                    donchian_fire=donchian_fire,
                    donchian_high=donchian_high,
                    ema10=row["ema10"],
                    ema30=row["ema30"],
                    ema50=row["ema50"],
                    rel_volume=row["rel_volume"],
                    cmf10=row["cmf10"],
                    obv_slope40=row["obv_slope40"],
                    adx14=row["adx14"],
                    sma200=row["sma200"],
                    sma200_slope=row["sma200_slope"],
                    atr14=row["atr14"],
                    conf=conf,
                )
                state = TrendHoldState(
                    position_state="IN_POSITION",
                    entry_date=trade_date_str,
                    entry_price=close,
                    base_high=float(donchian_high) if pd.notna(donchian_high) else None,
                    highest_close_since_entry=close,
                    structural_stop=close - CHANDELIER_ATR_MULT * atr_val,
                    peak_date=trade_date_str,
                    entry_gate=entry_gate,
                )
                decision = "BUY"
                if donchian_fire:
                    reason = (
                        f"fresh {DONCHIAN_LOOKBACK_SESSIONS}-session high, close {close:.3f} "
                        f"broke ceiling {float(donchian_high):.3f}"
                    )
                else:
                    reason = (
                        f"EMA10 crossed above EMA30 with close {close:.3f} above EMA50 "
                        f"{float(row['ema50']):.3f} (intermediate trend turned)"
                    )
                confidence = conf.total
                gate_snapshot = entry_gate
        else:
            if close > float(state.highest_close_since_entry):
                state.peak_date = str(row["trade_date"])
            state.highest_close_since_entry = max(float(state.highest_close_since_entry), close)
            atr_val = float(row["atr14"]) if pd.notna(row["atr14"]) else 0.0
            chandelier_stop = state.highest_close_since_entry - CHANDELIER_ATR_MULT * atr_val
            state.structural_stop = max(float(state.structural_stop), chandelier_stop)

            unrealized_gain = close / float(state.entry_price) - 1.0
            if (not state.scaled_out) and unrealized_gain >= SCALE_OUT_GAIN_PCT:
                decision = "SCALE_OUT"
                state.scaled_out = True
                state.position_fraction = 1.0 - SCALE_OUT_FRACTION
                reason = (
                    f"unrealized gain {unrealized_gain:.1%} reached the {SCALE_OUT_GAIN_PCT:.0%} milestone -- "
                    f"banking {SCALE_OUT_FRACTION:.0%} of the position, runner continues on the trailing stop"
                )
                gate_snapshot = _build_exit_gate(
                    trigger="SCALE_OUT_MILESTONE",
                    structural_stop=state.structural_stop,
                    highest_close_since_entry=state.highest_close_since_entry,
                    peak_date=state.peak_date,
                    exit_date=str(row["trade_date"]),
                    atr14=row["atr14"],
                    adx14=row["adx14"],
                )
            elif close < state.structural_stop:
                decision = "SELL_SIGNAL"
                reason = (
                    f"close {close:.3f} broke trailing stop {state.structural_stop:.3f} "
                    f"(chandelier {CHANDELIER_ATR_MULT}x ATR14 off high {state.highest_close_since_entry:.3f})"
                )
                confidence = _exit_confidence(close, state.structural_stop)
                gate_snapshot = _build_exit_gate(
                    trigger="CHANDELIER_STOP",
                    structural_stop=state.structural_stop,
                    highest_close_since_entry=state.highest_close_since_entry,
                    peak_date=state.peak_date,
                    exit_date=str(row["trade_date"]),
                    atr14=row["atr14"],
                    adx14=row["adx14"],
                )
                state.exit_reason = reason
                state.position_state = "EXIT_SIGNAL"
            else:
                decision = "HOLD"
                reason = f"close {close:.3f} above trailing stop {state.structural_stop:.3f}"

        results.append(
            {
                "trade_date": row["trade_date"],
                "close": close,
                "decision": decision,
                "reason": reason,
                "position_state": state.position_state,
                "entry_date": state.entry_date,
                "entry_price": state.entry_price,
                "structural_stop": state.structural_stop,
                "position_fraction": state.position_fraction,
                "confidence": confidence,
                "gate_snapshot": gate_snapshot,
                "entry_gate": state.entry_gate,
            }
        )

        if state.position_state == "EXIT_SIGNAL":
            state = TrendHoldState(position_state="NO_POSITION")

    return results
