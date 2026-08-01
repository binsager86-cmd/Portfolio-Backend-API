from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    files = [
        "scripts/r14_design_spec_consolidated_v2_2.py",
        "scripts/r13_seal_batch_v1_13.py",
        "artifacts/preview1a_prestart/review_final/r14_design_spec_CONSOLIDATED_v2_2.json",
        "artifacts/preview1a_prestart/review_final/r14_design_spec_CONSOLIDATED_v2_2.md",
    ]
    created = []
    for rel in files:
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(rel)
        created.append({"path": rel, "sha256": sha256_file(p), "size_bytes": p.stat().st_size})

    payload: dict[str, Any] = {
        "version_id": "R13_CREATED_FILES_MANIFEST_V1_13",
        "policy_note": "All executed scripts are permanent under scripts/ and sealed in this manifest.",
        "created_files": created,
        "constraints": {
            "read_only_db_uri": True,
            "no_engine_contact": True,
            "no_reruns": True,
            "design_only": True,
            "append_only_lineage_enforced": True,
        },
        "authorization_status": {
            "R14_A": "AUTHORIZED",
            "R14_B": "NOT_AUTHORIZED",
            "R15": "NOT_AUTHORIZED",
        },
        "disposition_note": {
            "entry_1_finding_anchoring": "Anchored 8 finding-response entries with exact modules and underscore tokens from defined sets.",
            "entry_2_module_fix": "Resolved ExecutionLiquidityAssessment mismatch by defining module boundaries and preserving registry ownership.",
            "lineage_repair": "Recorded v2_1 original and overwritten variants as non-authoritative; v2_2 is authoritative append-only continuation.",
        },
        "lineage_context": {
            "v2_1_original_json_sha256": "aedc50fca01886727e5154af8445f3ca64327d6d5f12638f52395cc7cb7dd328",
            "v2_1_overwritten_json_sha256": "262b8c17c175475e9ad893a5794348a40db9dd5e0c7fdf5c37fa7cf75d02d7bc",
        },
    }

    out_json = REVIEW / "r13_created_files_manifest_v1_13.json"
    out_sha = REVIEW / "r13_created_files_manifest_v1_13.sha256"
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_hash = sha256_file(out_json)
    out_sha.write_text(
        f"{manifest_hash}  artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1_13.json\n",
        encoding="utf-8",
    )

    print("R13_SEAL_BATCH_V1_13_COMPLETE")
    print("manifest_sha256", manifest_hash)
    print("sidecar_sha256", sha256_file(out_sha))


if __name__ == "__main__":
    main()
