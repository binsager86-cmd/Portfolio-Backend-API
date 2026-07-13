from __future__ import annotations

import hashlib
import json
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


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    base = (REVIEW / "r13_findings_of_record_v1_1.md").read_text(encoding="utf-8").rstrip() + "\n"
    f8 = read_json(REVIEW / "r13_f8_forensic_v1.json")

    sanam_rows = f8["sanam"]["window_rows"]
    tijara_rows = f8["tijara"]["window_rows"]

    ti_all_false = all(bool(r.get("M1_close_gt_base")) is False for r in tijara_rows)
    sanam_m1_pass = [r for r in sanam_rows if r.get("M1_close_gt_base") is True]
    sanam_m4_fail = [r for r in sanam_rows if r.get("M1_close_gt_base") is True and r.get("M4_chase_guard") is False]

    add = []
    add.append("")
    add.append("## Amendment v1.2")
    add.append("")
    add.append("### F8 Split Resolution")
    add.append("- F8 is split into F8a and F8b based on permanent forensic evidence from [r13_f8_forensic_v1.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_f8_forensic_v1.json).")
    add.append("")
    add.append("### F8a No-Freeze M1 Disarm (TIJARA)")
    add.append(f"- Status: {'CONFIRMED' if ti_all_false else 'NOT_CONFIRMED'}")
    add.append("- Statement: No persisted base reference exists for TIJARA in the 12 months preceding the cited 2025 high-volume window, so M1 remains false throughout the surfaced volume-arrival set.")
    add.append(f"- Evidence: reference_found={f8['tijara']['reference_found']} and all surfaced TIJARA rows have M1_close_gt_base=False :: source [r13_f8_forensic_v1.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_f8_forensic_v1.json)")
    add.append("")
    add.append("### F8b Stale-Reference Chase-Guard Race (SANAM)")
    add.append(f"- Status: {'CONFIRMED' if sanam_m4_fail else 'NOT_CONFIRMED'}")
    add.append("- Statement: SANAM carries a persisted frozen base_high_ref=233.0 from 2024-12-02; in May 2025 the stock eventually trades above that reference, and on 2025-05-21 the open gaps far enough above the stale reference to fail M4 while M1 is already true.")
    add.append(f"- Freeze evidence: base_high_ref={f8['sanam']['base_high_ref']} at freeze_event_date={f8['sanam']['freeze_event_date']} :: source [r13_f8_forensic_v1.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_f8_forensic_v1.json)")
    for row in sanam_m1_pass:
        add.append(f"- SANAM {row['date']}: close={row['close']} open={row['open']} M1={row['M1_close_gt_base']} gap_pct_base={row['gap_pct_base']} M4={row['M4_chase_guard']}")
    add.append("")
    add.append("### Interpretation")
    add.append("- TIJARA and SANAM diverge: TIJARA shows no-freeze M1 disarm; SANAM shows stale-reference race at volume arrival. These are distinct mechanisms and should remain separated in future design/test work.")
    add.append("")
    add.append("R14-B and R15 remain NOT AUTHORIZED.")
    add.append("")

    out = REVIEW / "r13_findings_of_record_v1_2.md"
    out.write_text(base + "\n".join(add), encoding="utf-8")
    print("R13_FINDINGS_OF_RECORD_V1_2_COMPLETE")
    print("sha256", sha256_file(out))


if __name__ == "__main__":
    main()
