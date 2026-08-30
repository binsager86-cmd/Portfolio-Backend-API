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

from dataclasses import dataclass
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
                state = TrendHoldState(
                    position_state="IN_POSITION",
                    entry_date=str(row["trade_date"]),
                    entry_price=close,
                    base_high=float(donchian_high) if pd.notna(donchian_high) else None,
                    highest_close_since_entry=close,
                    structural_stop=close - CHANDELIER_ATR_MULT * atr_val,
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
        else:
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
            elif close < state.structural_stop:
                decision = "SELL_SIGNAL"
                reason = (
                    f"close {close:.3f} broke trailing stop {state.structural_stop:.3f} "
                    f"(chandelier {CHANDELIER_ATR_MULT}x ATR14 off high {state.highest_close_since_entry:.3f})"
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
            }
        )

        if state.position_state == "EXIT_SIGNAL":
            state = TrendHoldState(position_state="NO_POSITION")

    return results
