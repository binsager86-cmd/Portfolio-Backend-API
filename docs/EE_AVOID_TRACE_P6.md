# EE AVOID Trace P6

## Evidence Scope

- Source of truth: `scripts/debug_gates.py` segment-anchored traces.
- Segment rule: only fixture segments were used for the diagnosis.

## BPCC: decline -> base -> breakout

Observed AVOID telemetry from `BPCC:decline` and `BPCC:breakout`:

- `1660262400` phase=`AVOID`, `close=597.131`, `sma200=600.925`, `sma200_slope<0=True`, `ema10<ema30=True`, `clear_streak=0/20`.
- `1660521600` phase=`AVOID`, `close=600.596`, `sma200=600.517`, `sma200_slope<0=True`, `ema10<ema30=False`, `clear_streak=1/20`.

Answer:

1. The clear streak does not reach 20 before the breakout window because the 200MA reclaim is intermittent and the AVOID condition resets when the entry predicate becomes false.
2. The trace shows the exact flicker: `close<sma200` remains true, but `ema10<ema30` flips false on the second bar, which resets the consecutive-false counter.
3. Result: AVOID persists through the breakout window even though WATCH passes.

## MABANEE: prefix -> first base -> breakout

Observed AVOID telemetry from `MABANEE:prefix` and `MABANEE:breakout`:

- Prefix bars show AVOID active very early, with `sma200_slope<0=True` and `ema10<ema30=True` while price remains under the falling 200MA.
- Example prefix line: `1655424000` phase=`AVOID`, `close=886.568`, `sma200=861.615`, `sma200_slope<0=True`, `ema10<ema30=False`, `clear_streak=3/20`.

Verdict:

1. This is not a generator defect.
2. The prefix is materially bearish, not nominally flat: the measured net drift is about `+0.85%`, while the local 200MA slope is negative on the inspected prefix bars.
3. The AVOID guard is correctly engaged because the prefix is already below a falling 200MA structure.
4. The remaining breakout-window miss is therefore a release-valve problem, not a false AVOID entry from fixture corruption.

## Trace Notes

- `avoid_until` is currently unset in the observed traces.
- The AVOID telemetry block now prints `close vs sma200`, `sma200_slope`, `ema10 vs ema30`, the clear-streak counter, and `avoid_until`.
