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
        "scripts/r13_f8_forensic_v1_1.py",
        "scripts/r13_findings_of_record_v1_3.py",
        "scripts/r14_design_spec_v1_1.py",
        "scripts/r13_seal_batch_v1_8.py",
        "artifacts/preview1a_prestart/review_final/r13_f8_forensic_v1_1.json",
        "artifacts/preview1a_prestart/review_final/r13_f8_forensic_v1_1.md",
        "artifacts/preview1a_prestart/review_final/r13_findings_of_record_v1_3.md",
        "artifacts/preview1a_prestart/review_final/r14_design_spec_v1_1.json",
        "artifacts/preview1a_prestart/review_final/r14_design_spec_v1_1.md",
    ]
    created = []
    for rel in files:
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(rel)
        created.append({"path": rel, "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
    payload: dict[str, Any] = {
        "version_id": "R13_CREATED_FILES_MANIFEST_V1_8",
        "policy_note": "All executed scripts are permanent under scripts/ and sealed in this manifest.",
        "created_files": created,
        "constraints": {
            "read_only_db_uri": True,
            "no_engine_contact": True,
            "no_reruns": True,
            "design_only": True,
        },
        "authorization_status": {
            "R14_A": "AUTHORIZED",
            "R14_B": "NOT_AUTHORIZED",
            "R15": "NOT_AUTHORIZED",
        },
    }
    out_json = REVIEW / "r13_created_files_manifest_v1_8.json"
    out_sha = REVIEW / "r13_created_files_manifest_v1_8.sha256"
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_hash = sha256_file(out_json)
    out_sha.write_text(f"{manifest_hash}  artifacts/preview1a_prestart/review_final/r13_created_files_manifest_v1_8.json\n", encoding="utf-8")
    print("R13_SEAL_BATCH_V1_8_COMPLETE")
    print("manifest_sha256", manifest_hash)
    print("sidecar_sha256", sha256_file(out_sha))


if __name__ == "__main__":
    main()
