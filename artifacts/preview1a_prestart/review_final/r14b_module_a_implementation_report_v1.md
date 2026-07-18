# R14-B Module (a) Implementation Report v1

Boundary: PredicateTelemetryLedger (storage + integrity only)

## File Hashes
[
  {
    "path": "app/services/eagle_eye_v2/__init__.py",
    "sha256": "a040c3d5cfe6651a8d421038250d2a7a760095e55aaba4179c38a042d29d8a09",
    "size_bytes": 75
  },
  {
    "path": "app/services/eagle_eye_v2/telemetry_schema.py",
    "sha256": "138114b03de5f1d0da3912b66267c94b65df87f6fb7bc58ed842319689e69ff0",
    "size_bytes": 1859
  },
  {
    "path": "app/services/eagle_eye_v2/predicate_telemetry_ledger.py",
    "sha256": "f9427ee6bc9fac05b620a25efc94fd0eed195fee6f7bb56a11f1a44f49fa2f34",
    "size_bytes": 10684
  },
  {
    "path": "scripts/r14b_module_a_write_path_harness_v1.py",
    "sha256": "a450c410360a341b20603e713bf5eb54c187375ad25c63f35d213a1dca870265",
    "size_bytes": 9286
  }
]

## Schema DDL As Emitted
```sql

        CREATE TABLE IF NOT EXISTS daily_term_row (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            segment_id TEXT,
            phase_before TEXT,
            phase_after TEXT,
            readiness_state TEXT,
            base_reference_id TEXT,
            intent_id TEXT,
            predicate_namespace TEXT,
            predicate_name TEXT,
            predicate_value REAL,
            predicate_threshold_parameter TEXT,
            predicate_pass INTEGER,
            recoverability_state TEXT,
            recoverability_reason TEXT,
            source_payload_fields TEXT,
            base_reference_version TEXT,
            base_reference_origin TEXT,
            base_reference_current_flag INTEGER,
            extension_pct_vs_current_valid_reference REAL,
            chase_advisory_flag INTEGER,
            current_day_value_kwd REAL,
            trailing_liquidity_context_value REAL,
            early_tier_flag INTEGER,
            dead_money_sessions INTEGER,
            flow_obv_slope_40 REAL,
            flow_anv_slope_40 REAL,
            flow_accumulation_divergence REAL,
            accumulation_context_ok INTEGER,
            participation_cap_pct REAL,
            pilot_size_fraction REAL,
            time_stop_sessions INTEGER,
            entry_tier TEXT,
            flow_evidence_snapshot TEXT,
            current_valid_reference_value REAL
        )
        

CREATE INDEX IF NOT EXISTS idx_daily_term_row_symbol_date ON daily_term_row(symbol, trade_date)


        CREATE TABLE IF NOT EXISTS daily_state_snapshot (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            readiness_state TEXT,
            phase_state TEXT,
            base_reference_snapshot TEXT,
            intent_snapshot TEXT,
            avoid_state TEXT,
            risk_budget_state TEXT
        )
        

CREATE INDEX IF NOT EXISTS idx_daily_state_snapshot_symbol_date ON daily_state_snapshot(symbol, trade_date)


        CREATE TABLE IF NOT EXISTS execution_outcome_row (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            candidate_intent_state TEXT,
            execution_state TEXT,
            veto_plane TEXT,
            veto_reason TEXT,
            opened_trade_flag INTEGER,
            trade_id TEXT,
            chase_advisory_emitted INTEGER,
            chase_advisory_extension_pct REAL,
            entry_tier TEXT,
            dead_money_sessions INTEGER
        )
        

CREATE INDEX IF NOT EXISTS idx_execution_outcome_row_symbol_date ON execution_outcome_row(symbol, trade_date)


        CREATE TABLE IF NOT EXISTS ledger_daily_hash_chain (
            chain_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            previous_hash TEXT,
            chain_hash TEXT NOT NULL,
            emitted_at_utc TEXT NOT NULL
        )
        

CREATE INDEX IF NOT EXISTS idx_ledger_daily_hash_chain_date ON ledger_daily_hash_chain(trade_date)


        CREATE TRIGGER IF NOT EXISTS trg_daily_term_row_block_update
        BEFORE UPDATE ON daily_term_row
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: daily_term_row update blocked');
        END
        


        CREATE TRIGGER IF NOT EXISTS trg_daily_term_row_block_delete
        BEFORE DELETE ON daily_term_row
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: daily_term_row delete blocked');
        END
        


        CREATE TRIGGER IF NOT EXISTS trg_daily_state_snapshot_block_update
        BEFORE UPDATE ON daily_state_snapshot
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: daily_state_snapshot update blocked');
        END
        


        CREATE TRIGGER IF NOT EXISTS trg_daily_state_snapshot_block_delete
        BEFORE DELETE ON daily_state_snapshot
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: daily_state_snapshot delete blocked');
        END
        


        CREATE TRIGGER IF NOT EXISTS trg_execution_outcome_row_block_update
        BEFORE UPDATE ON execution_outcome_row
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: execution_outcome_row update blocked');
        END
        


        CREATE TRIGGER IF NOT EXISTS trg_execution_outcome_row_block_delete
        BEFORE DELETE ON execution_outcome_row
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: execution_outcome_row delete blocked');
        END
        


        CREATE TRIGGER IF NOT EXISTS trg_ledger_daily_hash_chain_block_update
        BEFORE UPDATE ON ledger_daily_hash_chain
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: ledger_daily_hash_chain update blocked');
        END
        


        CREATE TRIGGER IF NOT EXISTS trg_ledger_daily_hash_chain_block_delete
        BEFORE DELETE ON ledger_daily_hash_chain
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: ledger_daily_hash_chain delete blocked');
        END
        

```

## Interface Conformance Artifact
r14b_module_a_interface_conformance_v1.json

## Test Evidence Artifact
r14b_module_a_test_evidence_v1.json

## Test Harness Output (Verbatim)
```text
R14B_MODULE_A_HARNESS_START
DDL_APPLIED count=16 dialect=sqlite
WRITE_READ_OK
APPEND_ONLY_TRIGGER_CHECK_COMPLETE
SIDECAR_CHAIN_EMITTED trade_date=2026-07-13 chain_hash=c15b571c17f094c921f2be923a0075a12efafdea9be3fb5f397355aa532071a0
INTERFACE_CONFORMANCE pass=True

```
