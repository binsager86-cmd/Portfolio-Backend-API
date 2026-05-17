# Eagle Eye ML — Improvements Backlog

Items here are identified but not yet implemented. Do not act on these until explicitly scheduled.

---

- **Dampener state granularity** — `dampener_fired` is a single boolean but has two distinct sub-states: (1) conditions met (`rel_liq < 0.5 AND today_ret > 0.02`) and (2) cap had actual effect (`confidence_pre_cap > 60`); for TRADEABLE stocks both are typically true together, but for WATCH_ONLY/ILLIQUID stocks (1) can be true while (2) is false (confidence already below 60 after Adj1). Separating these states in `ingest.py` and `considered_signals` would allow empirical analysis of whether the dampener improves win rate specifically for TRADEABLE stocks vs. lower-tier stocks where it is structurally inert.
