# R11 Unfreeze SOP (Eagle Eye Data Tables)

## Scope
This SOP defines the only allowed process to remove R11 write-freeze triggers from Eagle Eye market-data tables after governance checks pass.

## Preconditions
- Formal approval is recorded in an approved change request.
- R11 forensic artifacts remain intact under `artifacts/r11/forensic_20260710T190941Z`.
- Guard matrix proof is green (`tests/unit/test_db_isolation_guards.py`).
- Environment isolation is validated for the active runtime.

## Trigger Inventory
Run before unfreeze:

```sql
SELECT name, tbl_name, sql
FROM sqlite_master
WHERE type = 'trigger'
  AND (
    name LIKE 'r11_freeze_%'
    OR sql LIKE '%ee_ohlcv%'
    OR sql LIKE '%ee_indicators%'
    OR sql LIKE '%ee_signals%'
    OR sql LIKE '%ee_symbol_state%'
  )
ORDER BY name;
```

Export results to an artifact file in the active R11 report directory.

## Pre-Unfreeze Hash Snapshot
1. Create a DB backup file.
2. Compute SHA256 for:
- Active DB file
- Backup file
- Trigger inventory export
3. Store these in `artifacts/r11/reports_<timestamp>/r11_unfreeze_pre_hashes.json`.

## Unfreeze Execution
Only execute approved trigger drops. Example template:

```sql
DROP TRIGGER IF EXISTS r11_freeze_ee_ohlcv_insert;
DROP TRIGGER IF EXISTS r11_freeze_ee_ohlcv_update;
DROP TRIGGER IF EXISTS r11_freeze_ee_ohlcv_delete;
```

Repeat for other Eagle Eye freeze triggers only if explicitly listed in the approved change request.

## Post-Unfreeze Validation
- Re-run trigger inventory query and save to `r11_unfreeze_post_trigger_inventory.json`.
- Run guard suite:
  - `python -m pytest tests/unit/test_db_isolation_guards.py -q`
- Run ingestion smoke check in non-production DB and confirm:
  - lineage columns populated
  - `ee_ingestion_runs` row created and finalized
  - source-priority overwrite guard behavior is active
- Compute post-change DB SHA256 and store in `r11_unfreeze_post_hashes.json`.

## Rollback Procedure
If any post-unfreeze validation fails:
1. Stop writes to Eagle Eye tables.
2. Restore database from pre-unfreeze backup.
3. Recreate freeze triggers from approved SQL definition set.
4. Re-run guard matrix and ingestion smoke checks.
5. Log incident in R11 artifact directory with timestamp and root cause notes.

## Change Log Requirements
Every unfreeze must include:
- Operator and approver identities
- Timestamp (UTC)
- Approved change request ID
- Exact SQL executed
- Pre/post hash bundle references
- Validation command outputs
