from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from typing import Any

import r14e_module_e_lifecycle_intent_harness_v7 as v7
import r14e_module_e_lifecycle_intent_harness_v41a as v41a


RUN_NONCE = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
RUN_KEY = "R14E_MODULE_E_HARNESS_V41B"


def owner_windows() -> dict[str, dict[str, Any]]:
    windows: dict[str, dict[str, Any]] = {}
    for symbol, segments in v41a.SEGMENT_MAP.items():
        starts = [int(row["start_trade_date"]) for row in segments]
        ends = [int(row["end_trade_date"]) for row in segments]
        replay_start = v7.to_date_text(min(starts))
        replay_end = v7.to_date_text(max(ends))
        windows[symbol] = {
            "owner_start": replay_start,
            "owner_end": replay_end,
            "replay_start": replay_start,
            "replay_end": replay_end,
            "source_artifact": "v4.1-B full ratified baseline orchestration from ee_symbol_segment_map sealed window bounds.",
        }
    return dict(sorted(windows.items()))


def configure_v7() -> dict[str, Any]:
    v41a.SANDBOX.mkdir(parents=True, exist_ok=True)
    shutil.copy2(v41a.SOURCE_REVIEW / "r14b_parameter_freeze_v2.json", v41a.SANDBOX / "r14b_parameter_freeze_v2.json")
    shutil.copy2(v41a.SOURCE_REVIEW / "r14b_parameter_freeze_v2.sha256", v41a.SANDBOX / "r14b_parameter_freeze_v2.sha256")

    v7.REVIEW = v41a.SANDBOX
    v7.RUNTIME_DB = v41a.SEALED_DB
    v7.HARNESS_DB = v41a.SANDBOX / f"harness_v41B_{RUN_NONCE.replace(':', '').replace('.', '_')}.db"
    v7.FREEZE_JSON = v41a.SANDBOX / "r14b_parameter_freeze_v2.json"
    v7.FREEZE_SHA = v41a.SANDBOX / "r14b_parameter_freeze_v2.sha256"
    v7.RUN_NONCE = RUN_NONCE
    v7.RUN_KEY = RUN_KEY
    v7.load_window = v41a.load_window
    v7.count_ee_signals = v41a.count_ee_signals
    v7.r12_avoid_intervals = v41a.r12_avoid_intervals
    v7.owner_windows = owner_windows
    return {
        "sandbox": str(v41a.SANDBOX),
        "harness_db": str(v7.HARNESS_DB),
        "run_nonce": RUN_NONCE,
        "run_key": RUN_KEY,
        "canonical_symbols": len(owner_windows()),
        "window_mode": "full sealed window per canonical symbol",
    }


def main() -> None:
    sealed_attestation = v41a.assert_sealed_input()
    config = configure_v7()
    print("PHASE_0_3_B_SEALED_INPUT")
    print(json.dumps(sealed_attestation, ensure_ascii=True, indent=2, sort_keys=True))
    print("HARNESS_V41B_CONFIG")
    print(json.dumps(config, ensure_ascii=True, indent=2, sort_keys=True))
    v7.main()


if __name__ == "__main__":
    main()