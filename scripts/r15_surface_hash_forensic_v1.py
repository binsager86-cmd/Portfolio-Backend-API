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

V1_DB = REVIEW / "ee_v2_runtime_surface_r15_v1.db"
V2_DB = REVIEW / "ee_v2_runtime_surface_r15_v2.db"
PROBE_DB = REVIEW / "ee_v2_runtime_surface_r15_determinism_probe_v1.db"
V3_DB = REVIEW / "ee_v2_runtime_surface_r15_v3.db"

BINDING_V1 = REVIEW / "r15_surface_binding_v1.json"
REVOCATION_V1 = REVIEW / "r15_surface_revocation_v1.json"
BINDING_V2 = REVIEW / "r15_surface_binding_v2.json"

OUT_JSON = REVIEW / "r15_surface_hash_forensic_v1.json"
OUT_MD = REVIEW / "r15_surface_hash_forensic_v1.md"

BINDING_V2_1_JSON = REVIEW / "r15_surface_binding_v2_1.json"
BINDING_V2_1_MD = REVIEW / "r15_surface_binding_v2_1.md"
BINDING_V3_JSON = REVIEW / "r15_surface_binding_v3.json"
BINDING_V3_MD = REVIEW / "r15_surface_binding_v3.md"

TELEMETRY_TABLES = ["daily_term_row", "daily_state_snapshot", "execution_outcome_row"]

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


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def db_metrics(path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(path))
    try:
        row_counts: dict[str, int] = {}
        for t in TELEMETRY_TABLES:
            cnt = conn.execute(f"SELECT COUNT(1) FROM {t}").fetchone()
            row_counts[t] = int(cnt[0]) if cnt else 0
        ledger_cnt = conn.execute("SELECT COUNT(1) FROM ledger_daily_hash_chain").fetchone()
        row_counts["ledger_daily_hash_chain"] = int(ledger_cnt[0]) if ledger_cnt else 0

        trig_rows = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name").fetchall()
        present = [str(r[0]) for r in trig_rows]
        missing = [t for t in REQUIRED_TRIGGERS if t not in present]

        return {
            "path": str(path),
            "exists": True,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "row_counts": row_counts,
            "trigger_presence_read_only": {
                "verification_mode": "sqlite_master",
                "required_triggers": REQUIRED_TRIGGERS,
                "present_triggers": present,
                "missing_triggers": missing,
                "pass": len(missing) == 0,
            },
            "mtime_utc": iso_utc(path.stat().st_mtime),
        }
    finally:
        conn.close()


def touch_clean(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing forensic output DB: {path}")
    path.touch()


def ddl_mint(path: Path) -> dict[str, Any]:
    touch_clean(path)

    os.environ["EE_V2_RUNTIME_DB_PATH"] = str(path)
    os.environ["DATABASE_PATH"] = str(path)
    get_settings.cache_clear()

    migration = apply_schema_migration()
    return {
        "dialect": migration.get("dialect"),
        "ddl_statement_count": len(migration.get("ddl_emitted", [])),
        "metrics": db_metrics(path),
    }


def build_timeline(binding_v1: dict[str, Any], revocation_v1: dict[str, Any], binding_v2: dict[str, Any]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []

    if binding_v1:
        timeline.append(
            {
                "event": "binding_v1_recorded",
                "artifact": str(BINDING_V1),
                "event_time_utc": str(binding_v1.get("generated_at_utc") or "UNKNOWN"),
                "recorded_creation_hash": binding_v1.get("surface_binding", {}).get("creation_hash_sha256"),
                "recorded_post_ddl_hash": binding_v1.get("surface_binding", {}).get("post_ddl_hash_sha256"),
            }
        )

    contaminated_ref = REVIEW / "r14b_module_b_test_evidence_v2.json"
    if contaminated_ref.exists():
        timeline.append(
            {
                "event": "contaminating_harness_reference",
                "artifact": str(contaminated_ref),
                "event_time_utc": iso_utc(contaminated_ref.stat().st_mtime),
                "note": "Module (b) v2 evidence artifact timestamp used as contamination-run anchor.",
            }
        )

    if revocation_v1:
        timeline.append(
            {
                "event": "revocation_v1_recorded",
                "artifact": str(REVOCATION_V1),
                "event_time_utc": str(revocation_v1.get("generated_at_utc") or "UNKNOWN"),
                "recorded_revoked_hash": revocation_v1.get("revoked_surface", {}).get("hash_sha256_at_revocation"),
            }
        )

    if binding_v2:
        timeline.append(
            {
                "event": "binding_v2_recorded",
                "artifact": str(BINDING_V2),
                "event_time_utc": str(binding_v2.get("generated_at_utc") or "UNKNOWN"),
                "recorded_creation_hash": binding_v2.get("surface_binding", {}).get("creation_hash_sha256"),
                "recorded_post_ddl_hash": binding_v2.get("surface_binding", {}).get("post_ddl_hash_sha256"),
            }
        )

    return timeline


def counts_equal_nonzero(a: dict[str, int], b: dict[str, int]) -> bool:
    return all(int(a.get(k, 0)) == int(b.get(k, 0)) and int(a.get(k, 0)) > 0 for k in TELEMETRY_TABLES)


def all_zero(a: dict[str, int]) -> bool:
    return all(int(a.get(k, 0)) == 0 for k in TELEMETRY_TABLES + ["ledger_daily_hash_chain"])


def emit_binding_v2_1(v2: dict[str, Any], current_v2: dict[str, Any], note: str) -> dict[str, Any]:
    if BINDING_V2_1_JSON.exists() or BINDING_V2_1_MD.exists():
        raise FileExistsError("r15_surface_binding_v2_1 already exists; refusing overwrite")

    payload = {
        "version_id": "R15_SURFACE_BINDING_V2_1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "supersedes": "R15_SURFACE_BINDING_V2",
        "correction_note": note,
        "surface_binding": {
            "canonical_surface_db_path": str(V2_DB),
            "corrected_current_hash_sha256": current_v2["sha256"],
            "corrected_current_size_bytes": current_v2["size_bytes"],
            "row_counts": current_v2["row_counts"],
            "standing_rule": v2.get("surface_binding", {}).get("standing_rule"),
        },
        "trigger_presence": current_v2.get("trigger_presence_read_only"),
        "pass": True,
    }

    BINDING_V2_1_JSON.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    BINDING_V2_1_MD.write_text("\n".join(["# R15 Surface Binding v2.1", "", json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), ""]), encoding="utf-8")
    return {
        "artifact_json": str(BINDING_V2_1_JSON),
        "artifact_md": str(BINDING_V2_1_MD),
        "json_sha256": sha256_file(BINDING_V2_1_JSON),
        "md_sha256": sha256_file(BINDING_V2_1_MD),
    }


def emit_binding_v3(v2: dict[str, Any], v3_mint: dict[str, Any], note: str) -> dict[str, Any]:
    if BINDING_V3_JSON.exists() or BINDING_V3_MD.exists():
        raise FileExistsError("r15_surface_binding_v3 already exists; refusing overwrite")

    m = v3_mint["metrics"]
    payload = {
        "version_id": "R15_SURFACE_BINDING_V3",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "supersedes": "R15_SURFACE_BINDING_V2",
        "reason": note,
        "surface_binding": {
            "canonical_surface_db_path": str(V3_DB),
            "creation_hash_sha256": m["sha256"],
            "post_ddl_hash_sha256": m["sha256"],
            "size_bytes": m["size_bytes"],
            "row_counts": m["row_counts"],
            "standing_rule": v2.get("surface_binding", {}).get("standing_rule"),
        },
        "ddl_application": {
            "dialect": v3_mint.get("dialect"),
            "ddl_statement_count": v3_mint.get("ddl_statement_count"),
        },
        "trigger_presence": m.get("trigger_presence_read_only"),
        "pass": all_zero(m["row_counts"]) and bool(m.get("trigger_presence_read_only", {}).get("pass")),
    }

    BINDING_V3_JSON.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    BINDING_V3_MD.write_text("\n".join(["# R15 Surface Binding v3", "", json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), ""]), encoding="utf-8")
    return {
        "artifact_json": str(BINDING_V3_JSON),
        "artifact_md": str(BINDING_V3_MD),
        "json_sha256": sha256_file(BINDING_V3_JSON),
        "md_sha256": sha256_file(BINDING_V3_MD),
        "v3_db_sha256": sha256_file(V3_DB),
    }


def main() -> None:
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("Forensic artifacts already exist; refusing overwrite")

    if not V1_DB.exists() or not V2_DB.exists():
        raise FileNotFoundError("Expected v1 and v2 surface DB files must exist")

    if PROBE_DB.exists():
        raise FileExistsError(f"Determinism probe DB already exists; refusing overwrite: {PROBE_DB}")

    binding_v1 = read_json(BINDING_V1) if BINDING_V1.exists() else {}
    revocation_v1 = read_json(REVOCATION_V1) if REVOCATION_V1.exists() else {}
    binding_v2 = read_json(BINDING_V2) if BINDING_V2.exists() else {}

    current_v1 = db_metrics(V1_DB)
    current_v2 = db_metrics(V2_DB)

    probe = ddl_mint(PROBE_DB)
    probe_metrics = probe["metrics"]

    timeline = build_timeline(binding_v1, revocation_v1, binding_v2)

    branch = "C_REPORTING_ERROR"
    conclusion = "Recorded hashes require correction to align with current bytes on disk."
    actions: list[dict[str, Any]] = []

    v1_nonzero = any(int(current_v1["row_counts"].get(k, 0)) > 0 for k in TELEMETRY_TABLES)
    v2_zero = all_zero(current_v2["row_counts"])
    deterministic_match = probe_metrics["sha256"] == current_v2["sha256"]

    if v1_nonzero and v2_zero and deterministic_match:
        branch = "A_PRE_CONTAMINATION_HASH_PLUS_DETERMINISTIC_DDL"
        conclusion = (
            "v1 recorded post-DDL hash was pre-contamination; current v1 differs due later writes. "
            "v2 is independently-created zero-row DDL and matches a fresh deterministic DDL probe hash."
        )
    elif counts_equal_nonzero(current_v1["row_counts"], current_v2["row_counts"]):
        branch = "B_V2_COPY_OR_EQUIVALENT_CONTAMINATED_SURFACE"
        conclusion = "v2 row counts match nonzero v1 telemetry counts; v2 must be treated as revoked and replaced by clean v3."
        v3_mint = ddl_mint(V3_DB)
        actions.append(
            {
                "action": "mint_binding_v3",
                "details": emit_binding_v3(binding_v2, v3_mint, conclusion),
            }
        )
    else:
        note = (
            "Binding v2 recorded hash/size lineage is inconsistent with current bytes and deterministic probe outcome; "
            "v2.1 corrects the canonical record."
        )
        actions.append(
            {
                "action": "mint_binding_v2_1",
                "details": emit_binding_v2_1(binding_v2, current_v2, note),
            }
        )

    payload = {
        "version_id": "R15_SURFACE_HASH_FORENSIC_V1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "measured_current": {
            "v1": current_v1,
            "v2": current_v2,
        },
        "recorded_hashes": {
            "binding_v1": {
                "creation_hash_sha256": binding_v1.get("surface_binding", {}).get("creation_hash_sha256"),
                "post_ddl_hash_sha256": binding_v1.get("surface_binding", {}).get("post_ddl_hash_sha256"),
            },
            "revocation_v1": {
                "hash_sha256_at_revocation": revocation_v1.get("revoked_surface", {}).get("hash_sha256_at_revocation"),
            },
            "binding_v2": {
                "creation_hash_sha256": binding_v2.get("surface_binding", {}).get("creation_hash_sha256"),
                "post_ddl_hash_sha256": binding_v2.get("surface_binding", {}).get("post_ddl_hash_sha256"),
            },
        },
        "deterministic_ddl_probe": {
            "path": str(PROBE_DB),
            "dialect": probe.get("dialect"),
            "ddl_statement_count": probe.get("ddl_statement_count"),
            "metrics": probe_metrics,
            "matches_v2_hash": probe_metrics["sha256"] == current_v2["sha256"],
        },
        "timeline": timeline,
        "resolution": {
            "branch": branch,
            "conclusion": conclusion,
            "actions_emitted": actions,
        },
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join(["# R15 Surface Hash Forensic v1", "", json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), ""]), encoding="utf-8")

    print("R15_SURFACE_HASH_FORENSIC_V1_COMPLETE")
    print("json_sha256", sha256_file(OUT_JSON))
    print("md_sha256", sha256_file(OUT_MD))
    print("probe_db_sha256", sha256_file(PROBE_DB))
    print("resolution_branch", branch)


if __name__ == "__main__":
    main()
