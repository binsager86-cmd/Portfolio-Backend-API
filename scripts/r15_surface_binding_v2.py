from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.services.eagle_eye_v2.predicate_telemetry_ledger import apply_schema_migration

REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
REVOKED_DB = REVIEW / "ee_v2_runtime_surface_r15_v1.db"
CANONICAL_V2_DB = REVIEW / "ee_v2_runtime_surface_r15_v2.db"
BINDING_V1 = REVIEW / "r15_surface_binding_v1.json"
REVOCATION_V1 = REVIEW / "r15_surface_revocation_v1.json"

REQUIRED_TRIGGERS = [
    "trg_daily_term_row_block_update",
    "trg_daily_term_row_block_delete",
    "trg_daily_state_snapshot_block_update",
    "trg_daily_state_snapshot_block_delete",
    "trg_execution_outcome_row_block_update",
    "trg_execution_outcome_row_block_delete",
    "trg_ledger_daily_hash_chain_block_update",
    "trg_ledger_daily_hash_chain_block_delete",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_triggers_read_only(path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name"
        ).fetchall()
        present = [str(r[0]) for r in rows]
        missing = [t for t in REQUIRED_TRIGGERS if t not in present]
        return {
            "target_db": str(path),
            "verification_mode": "read_only_sqlite_master",
            "required_triggers": REQUIRED_TRIGGERS,
            "present_triggers": present,
            "missing_triggers": missing,
            "pass": len(missing) == 0,
        }
    finally:
        conn.close()


def zero_row_check(path: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(path))
    try:
        out: dict[str, int] = {}
        for table in ["daily_term_row", "daily_state_snapshot", "execution_outcome_row", "ledger_daily_hash_chain"]:
            cnt = conn.execute(f"SELECT COUNT(1) FROM {table}").fetchone()
            out[table] = int(cnt[0]) if cnt else 0
        return out
    finally:
        conn.close()


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)

    binding_json = REVIEW / "r15_surface_binding_v2.json"
    binding_md = REVIEW / "r15_surface_binding_v2.md"

    if not REVOCATION_V1.exists():
        raise FileNotFoundError(f"Revocation artifact missing; run scripts/r15_surface_revocation_v1.py first: {REVOCATION_V1}")
    if CANONICAL_V2_DB.exists():
        raise FileExistsError(f"Canonical v2 DB already exists; refusing to mutate: {CANONICAL_V2_DB}")

    b1 = read_json(BINDING_V1) if BINDING_V1.exists() else {}
    revocation_payload = read_json(REVOCATION_V1)

    CANONICAL_V2_DB.touch()
    creation_hash = sha256_file(CANONICAL_V2_DB)

    os.environ["EE_V2_RUNTIME_DB_PATH"] = str(CANONICAL_V2_DB)
    os.environ["DATABASE_PATH"] = str(CANONICAL_V2_DB)
    get_settings.cache_clear()

    migration = apply_schema_migration()
    post_ddl_hash = sha256_file(CANONICAL_V2_DB)
    trigger_check = inspect_triggers_read_only(CANONICAL_V2_DB)
    row_counts = zero_row_check(CANONICAL_V2_DB)

    if any(v != 0 for v in row_counts.values()):
        raise RuntimeError(f"Canonical v2 must be zero-row after DDL, got: {row_counts}")

    binding_payload = {
        "version_id": "R15_SURFACE_BINDING_V2",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "surface_binding": {
            "canonical_surface_db_path": str(CANONICAL_V2_DB),
            "created_in_this_run": True,
            "creation_hash_sha256": creation_hash,
            "post_ddl_hash_sha256": post_ddl_hash,
            "revocation_reference": str(REVOCATION_V1),
            "binding_env": {
                "EE_V2_RUNTIME_DB_PATH": str(CANONICAL_V2_DB),
                "DATABASE_PATH": str(CANONICAL_V2_DB),
            },
            "standing_rule": "No harness or test process may ever write to canonical surface. Canonical verification is read-only sqlite_master/PRAGMA inspection only. All write-path testing must use dedicated harness DBs.",
        },
        "ddl_application": {
            "dialect": migration.get("dialect"),
            "ddl_statement_count": len(migration.get("ddl_emitted", [])),
            "zero_row_assertion": row_counts,
        },
        "revoked_surface": revocation_payload.get("revoked_surface", {}),
        "trigger_presence": trigger_check,
        "pass": bool(trigger_check.get("pass")) and all(v == 0 for v in row_counts.values()),
    }
    binding_json.write_text(json.dumps(binding_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    binding_md.write_text(
        "\n".join(["# R15 Surface Binding v2", "", json.dumps(binding_payload, ensure_ascii=True, indent=2, sort_keys=True), ""]),
        encoding="utf-8",
    )

    print("R15_SURFACE_BINDING_V2_COMPLETE")
    print("canonical_v2_db_sha256", sha256_file(CANONICAL_V2_DB))
    print("binding_json_sha256", sha256_file(binding_json))
    print("binding_md_sha256", sha256_file(binding_md))


if __name__ == "__main__":
    main()
