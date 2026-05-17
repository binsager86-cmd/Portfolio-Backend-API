# Simulator Pre-Phase-2 Reset Snapshot

_Captured: 2026-05-16 (before Phase 2 clean-baseline reset)_  
_Purpose: Historical record of simulator state under pre-Phase-1 scoring logic (pre-dampener, pre-corporate-events-fix)_

---

## Portfolio State at Reset

| Portfolio    | Strategy      | Cash (KWD)  | Total Value (KWD) | Inception   |
|--------------|---------------|-------------|-------------------|-------------|
| id=1         | CONSERVATIVE  | 6,245.33    | 9,999.96          | 2026-05-14  |
| id=2         | MODERATE      | 6,324.41    | 9,999.98          | 2026-05-14  |
| id=3         | AGGRESSIVE    | 6,324.41    | 9,999.98          | 2026-05-14  |

---

## Open Positions at Reset (11 total)

| Strategy      | Ticker   | Entry Date  | Entry Price (KWD) | Size (KWD) | Last Close (KWD) | Unrealized PnL |
|---------------|----------|-------------|-------------------|------------|------------------|----------------|
| AGGRESSIVE    | ENERGYH  | 2026-05-14  | 285.0000          | 877        | 285.0000         | 0.00%          |
| AGGRESSIVE    | ALDEERA  | 2026-05-14  | 584.0000          | 1,166      | 584.0000         | 0.00%          |
| AGGRESSIVE    | KFIC     | 2026-05-14  | 148.0000          | 421        | 148.0000         | 0.00%          |
| AGGRESSIVE    | CLEANING | 2026-05-14  | 264.0000          | 1,212      | 264.0000         | 0.00%          |
| CONSERVATIVE  | ENERGYH  | 2026-05-16  | 285.0000          | 1,011      | 285.0000         | 0.00%          |
| CONSERVATIVE  | ALDEERA  | 2026-05-16  | 584.0000          | 1,345      | 584.0000         | 0.00%          |
| CONSERVATIVE  | CLEANING | 2026-05-16  | 264.0000          | 1,398      | 264.0000         | 0.00%          |
| MODERATE      | ENERGYH  | 2026-05-14  | 285.0000          | 877        | 284.0000         | 0.00%          |
| MODERATE      | ALDEERA  | 2026-05-14  | 584.0000          | 1,166      | 584.0000         | 0.00%          |
| MODERATE      | KFIC     | 2026-05-14  | 148.0000          | 421        | 148.0000         | 0.00%          |
| MODERATE      | CLEANING | 2026-05-14  | 264.0000          | 1,212      | 264.0000         | 0.00%          |

_Note: Last close = 2026-05-14 (most recent OHLCV bar available). All positions entered at or after this date, so unrealized PnL = 0.00% across all strategies._

---

## Trade Statistics at Reset

| Strategy      | Total Trades | Closed | Wins | Win Rate | Total PnL (KWD) |
|---------------|--------------|--------|------|----------|-----------------|
| CONSERVATIVE  | 3            | 0      | 0    | N/A      | 0.00            |
| MODERATE      | 4            | 0      | 0    | N/A      | 0.00            |
| AGGRESSIVE    | 4            | 0      | 0    | N/A      | 0.00            |

_All positions were open at time of reset — no completed trades, no realized PnL._

---

## Reset Reason

Scoring logic updated since portfolio inception (2026-05-14):
1. **Thin-volume-on-rise dampener** added — caps confidence at 60 when `rel_liq < 0.5 AND today_ret > 0.02`
2. **Corporate events leakage bug fixed** — announcement_date filter applied before training window selection
3. **`dollar_volume` column added** to indicator output (required for dampener)

A clean baseline is required so all forward simulator trades run under the updated scoring logic.

---

## Archive Tables

Pre-reset data is preserved in:
- `simulator_positions_archive` — all 11 position rows (created during reset)
- `simulator_daily_snapshots_archive` — all 6 daily snapshot rows (created during reset)

This snapshot file is the human-readable complement to those archive tables.
