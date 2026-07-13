from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.eagle_eye.candidate_v2_service import run_candidate_v2


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run R11 Phase 0 Candidate V2 remediation gate.")
    parser.add_argument("--existing-db", required=True, help="Path to immutable evidence DB (v1).")
    parser.add_argument("--new-db", required=True, help="Path for candidate v2 DB.")
    parser.add_argument("--output-dir", required=True, help="Output directory for candidate v2 artifacts.")
    args = parser.parse_args()

    existing_db = Path(args.existing_db).resolve()
    new_db = Path(args.new_db).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not existing_db.exists():
        raise SystemExit(f"existing evidence DB not found: {existing_db}")

    existing_hash = file_sha256(existing_db)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_v1_immutable_sha256.json").write_text(
        json.dumps(
            {
                "db_path": str(existing_db),
                "sha256": existing_hash,
                "immutable": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = run_candidate_v2(db_path=new_db, output_dir=output_dir)

    print(
        json.dumps(
            {
                "existing_db_sha256": existing_hash,
                "new_db": str(new_db),
                "output_dir": str(output_dir),
                "pass_recommendation": summary.get("pass_recommendation", False),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
