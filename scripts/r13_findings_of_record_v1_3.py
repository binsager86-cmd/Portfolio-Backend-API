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
    base = (REVIEW / "r13_findings_of_record_v1_2.md").read_text(encoding="utf-8").rstrip() + "\n"
    probe = read_json(REVIEW / "r13_f8_forensic_v1_1.json")["probe_2025_05_18"]

    add = []
    add.append("")
    add.append("## Amendment v1.3")
    add.append("")
    add.append("### 2025-05-18 Final Probe")
    add.append("- SANAM 2025-05-18 was probed through the full post-mandatory confirmation path using sealed runtime records and code-path reconstruction.")
    add.append(f"- M3 pass={probe['mandatory']['M3_ema10_gt_ema30']['pass']} value={json.dumps(probe['mandatory']['M3_ema10_gt_ema30']['value'], ensure_ascii=True)}")
    add.append(f"- M5 pass={probe['mandatory']['M5_liquidity']['pass']} value={json.dumps(probe['mandatory']['M5_liquidity']['value'], ensure_ascii=True)}")
    add.append(f"- C-score pass={probe['c_score']['pass']} value={probe['c_score']['value']}")
    add.append(f"- ML gate pass={probe['ml_gate']['pass']} details={json.dumps(probe['ml_gate'], ensure_ascii=True)}")
    add.append(f"- Score gate pass={probe['score_gate']['pass']} score={probe['score_gate']['score']}")
    add.append("")
    if probe['narrowed_conclusion']['identified_blocker'] == 'M5_liquidity':
        add.append("### 2025-05-18 Blocker Resolution")
        add.append("- Resolved blocker: M5_liquidity.")
        add.append("- This is not a new F8c mechanism. Per directive, the result is folded into existing findings rather than creating a forced third branch.")
        add.append("- Composite confirmation and score layers are not implicated on the sealed 2025-05-18 evidence; the post-mandatory resolved blocker is liquidity.")
    elif probe['narrowed_conclusion']['identified_blocker'] is None:
        add.append("### F8c Status")
        add.append("- F8c NOT ESTABLISHED.")
        add.append(f"- Reason: {probe['narrowed_conclusion']['residual_uncertainty']}")
        add.append("- The composite-layer block hypothesis is not supported by the sealed 2025-05-18 evidence; resolved post-mandatory elements all pass.")
        add.append("- The remaining blocker narrows to non-persisted current-valid-reference/state behavior, reinforcing F7 rather than creating a new confirmed mechanism.")
    else:
        add.append("### Final-Probe Result")
        add.append(f"- Established blocker: {probe['narrowed_conclusion']['identified_blocker']}")
    add.append("")
    add.append("### Implication For Spec")
    add.append("- The design spec must explicitly neutralize the stale-reference race and bring any remaining veto-capable post-mandatory authority under full telemetry.")
    add.append("")
    add.append("R14-B and R15 remain NOT AUTHORIZED.")
    add.append("")

    out = REVIEW / "r13_findings_of_record_v1_3.md"
    out.write_text(base + "\n".join(add), encoding="utf-8")
    print("R13_FINDINGS_OF_RECORD_V1_3_COMPLETE")
    print("sha256", sha256_file(out))


if __name__ == "__main__":
    main()
