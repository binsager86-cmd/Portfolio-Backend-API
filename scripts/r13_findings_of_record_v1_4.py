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
    base = (REVIEW / "r13_findings_of_record_v1_3.md").read_text(encoding="utf-8").rstrip() + "\n"
    m5 = read_json(REVIEW / "r13_m5_liquidity_forensic_v1.json")
    add = []
    add.append("")
    add.append("## Amendment v1.4")
    add.append("")
    add.append("### F9 Liquidity-Gate Temporal Lag")
    if m5['f9']['status'] == 'CONFIRMED':
        add.append("- F9 CONFIRMED.")
        add.append("- Statement: the liquidity gate measures the accumulation past, not the breakout present; current-day breakout liquidity may be strong while the trailing median still fails the sole veto threshold.")
    else:
        add.append("- F9 NOT CONFIRMED as a sole-veto mechanism under the implemented resolved-term test.")
        add.append(f"- Recorded mechanism instead: {m5['f9']['statement']}")
    sanam_0518 = next(r for r in m5['sanam_2025_05_08_to_2025_05_21'] if r['date']=='2025-05-18')
    add.append(f"- Canonical day SANAM 2025-05-18: same_day_value_kwd={sanam_0518['same_day_value_kwd']}, trailing_median_value_kwd={sanam_0518['liquidity_filter_inputs']['median_daily_value_kwd_20']}, threshold={m5['run2_config_values']['min_daily_value_kwd']}, M5_pass={sanam_0518['mandatory']['M5_liquidity']} :: source [r13_m5_liquidity_forensic_v1.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_m5_liquidity_forensic_v1.json)")
    add.append(f"- M5 sole surviving blocker counts by symbol: {json.dumps(m5['m5_sole_surviving_blocker_count_by_symbol'], ensure_ascii=True, sort_keys=True)}")
    add.append("")
    add.append("### Conduct Ledger")
    add.append("- Third permanent-script violation acknowledged: prior cycle used deleted temp probe scripts despite the permanent-script rule already being in force.")
    add.append("- Rule restated: ALL executed scripts, including read-only extraction and forensic scripts, must be permanent files under scripts/ and sealed in the manifest.")
    add.append("")
    add.append("R14-B and R15 remain NOT AUTHORIZED.")
    add.append("")
    out = REVIEW / "r13_findings_of_record_v1_4.md"
    out.write_text(base + "\n".join(add), encoding="utf-8")
    print("R13_FINDINGS_OF_RECORD_V1_4_COMPLETE")
    print("sha256", sha256_file(out))


if __name__ == "__main__":
    main()
