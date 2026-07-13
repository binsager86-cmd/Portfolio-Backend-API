from __future__ import annotations

import hashlib
from pathlib import Path

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
    base = (REVIEW / "r13_findings_of_record_v1_4.md").read_text(encoding="utf-8").rstrip() + "\n"
    add = []
    add.append("")
    add.append("## Amendment v1.5")
    add.append("")
    add.append("### Conduct Ledger")
    add.append("- Permanent-script rule remains in force for all extraction, forensic, and spec-surfacings scripts.")
    add.append("- This batch remains sealed and reproducible through v1_10.")
    add.append("")
    add.append("R14-B and R15 remain NOT AUTHORIZED.")
    add.append("")
    out = REVIEW / "r13_findings_of_record_v1_5.md"
    out.write_text(base + "\n".join(add), encoding="utf-8")
    print("R13_FINDINGS_OF_RECORD_V1_5_COMPLETE")
    print("sha256", sha256_file(out))


if __name__ == "__main__":
    main()
