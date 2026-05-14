from __future__ import annotations

import json
from pathlib import Path

from app.services.eagle_eye.ml.feature_builder_v2 import run_abk_clean_rebuild


def main() -> None:
    result = run_abk_clean_rebuild(ticker="ABK", seed=42)
    out_path = Path("tools") / "abk_clean_rebuild_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()