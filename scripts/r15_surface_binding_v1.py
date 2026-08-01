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
SURFACE_DB = REVIEW / "ee_v2_runtime_surface_r15_v1.db"


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


def trigger_presence(surface_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(surface_path))
    try:
        rows = conn.execute(
            "SELECT name, tbl_name FROM sqlite_master WHERE type='trigger' ORDER BY name"
        ).fetchall()
        present = [str(r[0]) for r in rows]
        missing = [name for name in REQUIRED_TRIGGERS if name not in present]
        return {
            "target_db": str(surface_path),
            "required_triggers": REQUIRED_TRIGGERS,
            "present_triggers": present,
            "missing_triggers": missing,
            "pass": len(missing) == 0,
        }
    finally:
        conn.close()


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)

    out_json = REVIEW / "r15_surface_binding_v1.json"
    out_md = REVIEW / "r15_surface_binding_v1.md"

    pre_exists = SURFACE_DB.exists()
    if not pre_exists:
        SURFACE_DB.touch()

    creation_hash = sha256_file(SURFACE_DB)

    os.environ["EE_V2_RUNTIME_DB_PATH"] = str(SURFACE_DB)
    os.environ["DATABASE_PATH"] = str(SURFACE_DB)
    get_settings.cache_clear()

    migration = apply_schema_migration()
    trigger_check = trigger_presence(SURFACE_DB)
    post_ddl_hash = sha256_file(SURFACE_DB)

    payload = {
        "version_id": "R15_SURFACE_BINDING_V1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "surface_binding": {
            "canonical_surface_db_path": str(SURFACE_DB),
            "created_in_this_run": not pre_exists,
            "creation_hash_sha256": creation_hash,
            "post_ddl_hash_sha256": post_ddl_hash,
            "binding_env": {
                "EE_V2_RUNTIME_DB_PATH": str(SURFACE_DB),
                "DATABASE_PATH": str(SURFACE_DB),
            },
            "execution_path_policy": "EE_V2 pipeline writes target canonical surface only; dev_portfolio.db excluded from EE_V2 execution path.",
        },
        "ddl_application": {
            "dialect": migration.get("dialect"),
            "ddl_statement_count": len(migration.get("ddl_emitted", [])),
        },
        "trigger_presence": trigger_check,
        "pass": bool(trigger_check.get("pass")),
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_md.write_text(
        "\n".join(
            [
                "# R15 Surface Binding v1",
                "",
                json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("R15_SURFACE_BINDING_V1_COMPLETE")
    print("surface_db_sha256", sha256_file(SURFACE_DB))
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()
