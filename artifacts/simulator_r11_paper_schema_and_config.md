# Simulator Rebuild R11 - Paper Only

## Scope Guard
- Paper money only
- Uses production legacy R11 rating outputs (`ratings_history` + fallback to `ee_ratings_cache` for current day)
- No `eagle_eye_v2` authority imports in simulator decision paths
- Campaign closure untouched

## DDL (Append-Only Ledger)

```sql
CREATE TABLE IF NOT EXISTS sim_position_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER NOT NULL,
    portfolio_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    snapshot_kind TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    snapshot_price REAL,
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sim_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL,
    position_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    event_kind TEXT NOT NULL,
    fraction_closed REAL NOT NULL,
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_snapshot_json TEXT NOT NULL,
    exit_date TEXT NOT NULL,
    exit_price REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    realized_pnl_pct REAL,
    holding_sessions INTEGER,
    mfe_pct REAL,
    mae_pct REAL,
    exit_snapshot_json TEXT,
    outcome_class TEXT,
    persisted_fields_json TEXT,
    flipped_fields_json TEXT,
    sessions_to_flip_json TEXT,
    mfe_gt_10 INTEGER NOT NULL DEFAULT 0,
    mfe_gt_20 INTEGER NOT NULL DEFAULT 0,
    attribution_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS trg_sim_transactions_block_update
BEFORE UPDATE ON sim_transactions
BEGIN
    SELECT RAISE(ABORT, 'append-only table: sim_transactions update blocked');
END;

CREATE TRIGGER IF NOT EXISTS trg_sim_transactions_block_delete
BEFORE DELETE ON sim_transactions
BEGIN
    SELECT RAISE(ABORT, 'append-only table: sim_transactions delete blocked');
END;

CREATE TRIGGER IF NOT EXISTS trg_sim_position_snapshots_block_update
BEFORE UPDATE ON sim_position_snapshots
BEGIN
    SELECT RAISE(ABORT, 'append-only table: sim_position_snapshots update blocked');
END;

CREATE TRIGGER IF NOT EXISTS trg_sim_position_snapshots_block_delete
BEFORE DELETE ON sim_position_snapshots
BEGIN
    SELECT RAISE(ABORT, 'append-only table: sim_position_snapshots delete blocked');
END;
```

## Two-Card Config

```json
{
  "cards": [
    {
      "name": "BUY",
      "paper_label": "PAPER - SIMULATION",
      "starting_capital_kwd": 100000,
      "position_size_pct_of_equity": 0.10,
      "max_concurrent_positions": 10,
      "one_position_per_symbol": true,
      "entry_rule": "rating transition to BUY or STRONG_BUY",
      "fill_rule": "next session open after signal day",
      "same_bar_fills": false,
      "exit_rules": {
        "full": ["SELL_TRANSITION", "TOPPING_SIGNAL", "AVOID_TRANSITION"],
        "partial_half": ["REDUCE_TRANSITION"]
      }
    },
    {
      "name": "WATCHLIST",
      "paper_label": "PAPER - SIMULATION",
      "starting_capital_kwd": 100000,
      "position_size_pct_of_equity": 0.10,
      "max_concurrent_positions": 10,
      "one_position_per_symbol": true,
      "entry_rule": "rating transition to WATCH or WATCHLIST",
      "fill_rule": "next session open after signal day",
      "same_bar_fills": false,
      "exit_rules": {
        "full": ["SELL_TRANSITION", "TOPPING_SIGNAL", "AVOID_TRANSITION"],
        "partial_half": ["REDUCE_TRANSITION"]
      }
    }
  ]
}
```

## Processing Cadence
- Daily job still executes after ratings refresh.
- `run_daily(..., backfill_missing=true)` processes missed trading days in ascending date order.
- `tools/backfill_simulator.py` is re-enabled for explicit date-range backfill.
