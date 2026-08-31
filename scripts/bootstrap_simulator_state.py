from __future__ import annotations

import json
import time
from pathlib import Path

from app.core.config import get_settings
from app.services.eagle_eye_v2.simulator import ForwardSurfaceBuilder, SimulatorRunner


def main() -> None:
    settings = get_settings()
    live_db = Path(settings.database_abs_path)
    sealed_db = Path(
        r"C:\Users\Sager\OneDrive\Desktop\portfolio_app\mobile-migration\backend-api-main-release\artifacts\preview1a_prestart\review_final\r12_exam_surface_v4_5_runtime.db"
    )
    surface_db = Path(
        r"C:\Users\Sager\OneDrive\Desktop\portfolio_app\mobile-migration\backend-api-main-release\artifacts\preview1a_prestart\review_final\forward_surface_gate_live_full.db"
    )
    import sqlite3

    with sqlite3.connect(str(live_db)) as conn:
        session = str(conn.execute("SELECT MAX(bar_date) FROM ee_ohlcv_cache").fetchone()[0])
    builder = ForwardSurfaceBuilder(live_db_path=live_db, sealed_db_path=sealed_db, surface_db_path=surface_db)
    market = builder
    runner = SimulatorRunner(mode="live", live_db_path=live_db, expected_symbol_count=None)
    sessions = runner.load_market_sessions(session, expected_symbol_count=None)
    started = time.perf_counter()
    completed = 0
    total = len(sessions)
    failures: list[dict[str, str]] = []
    print(json.dumps({"bootstrap_session": session, "symbols": total, "mode": "offline_full_replay"}), flush=True)
    for segment_symbol in sorted(sessions):
        canonical = segment_symbol.split("__SEG", 1)[0]
        completed += 1
        try:
            result = runner.bootstrap_machine_state(canonical, session, surface_db)
            print(json.dumps({"symbol": canonical, "status": "BOOTSTRAPPED", "completed": completed, "total": total, "elapsed_sec": result["elapsed_sec"]}), flush=True)
        except Exception as exc:
            failure = {"symbol": canonical, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(json.dumps({"symbol": canonical, "status": "FAILED", "completed": completed, "total": total, **failure}), flush=True)
    print(json.dumps({"status": "BOOTSTRAP_COMPLETE", "symbols": completed - len(failures), "failed": len(failures), "failures": failures, "elapsed_sec": round(time.perf_counter() - started, 3)}), flush=True)


if __name__ == "__main__":
    main()
