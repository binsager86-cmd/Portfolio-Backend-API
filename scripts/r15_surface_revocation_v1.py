from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
REVOKED_DB = REVIEW / "ee_v2_runtime_surface_r15_v1.db"
BINDING_V1 = REVIEW / "r15_surface_binding_v1.json"

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
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name").fetchall()
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


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)

    out_json = REVIEW / "r15_surface_revocation_v1.json"
    out_md = REVIEW / "r15_surface_revocation_v1.md"

    if out_json.exists() or out_md.exists():
        raise FileExistsError("Revocation artifacts already exist; refusing to overwrite append-only evidence")
    if not REVOKED_DB.exists():
        raise FileNotFoundError(f"Revoked surface DB not found: {REVOKED_DB}")

    b1 = read_json(BINDING_V1) if BINDING_V1.exists() else {}
    revoked_hash = sha256_file(REVOKED_DB)
    trigger_check = inspect_triggers_read_only(REVOKED_DB)

    payload = {
        "version_id": "R15_SURFACE_REVOCATION_V1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "revoked_surface": {
            "path": str(REVOKED_DB),
            "hash_sha256_at_revocation": revoked_hash,
            "prior_binding_hashes": {
                "creation_hash_sha256": b1.get("surface_binding", {}).get("creation_hash_sha256"),
                "post_ddl_hash_sha256": b1.get("surface_binding", {}).get("post_ddl_hash_sha256"),
            },
            "trigger_presence_read_only": trigger_check,
            "reason": "EVIDENCE_FABRICATION_REMEDIATION: synthetic evidence rows were persisted to ledger; canonical surface contaminated and revoked.",
        },
        "revocation_policy": {
            "canonical_status": "REVOKED",
            "append_only_note": "Rows are immutable under append-only triggers; revocation is lineage containment, not row deletion.",
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(["# R15 Surface Revocation v1", "", json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), ""]), encoding="utf-8")

    print("R15_SURFACE_REVOCATION_V1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()
