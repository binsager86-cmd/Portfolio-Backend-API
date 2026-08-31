"""
Eagle Eye - Scheduled Daily Recompute Runner

Runs the full nightly pipeline (fetch fresh OHLCV -> compute ratings ->
auto-snapshot via tracker hook) and logs the result to a dated log file
so missed or failed runs are visible.

Usage (invoked by Windows Task Scheduler):
    python scripts/run_scheduled_recompute.py          # daily (no DNA refresh)
    python scripts/run_scheduled_recompute.py --dna     # weekly (with DNA refresh)

Exit codes:
    0 = success
    1 = failure (check the log)
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

# Ensure backend-api root is on the path regardless of where the task runs from
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(_THIS_DIR)  # scripts/ -> backend-api/
sys.path.insert(0, _BACKEND_ROOT)
os.chdir(_BACKEND_ROOT)

LOG_DIR = os.path.join(_BACKEND_ROOT, "logs", "scheduled_recompute")
os.makedirs(LOG_DIR, exist_ok=True)


def _log(msg: str, log_path: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def main() -> int:
    with_dna = "--dna" in sys.argv
    mode = "WEEKLY (with DNA refresh)" if with_dna else "DAILY (no DNA refresh)"

    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"recompute_{today}.log")

    _log("=" * 60, log_path)
    _log(f"Eagle Eye scheduled recompute START - mode: {mode}", log_path)

    t0 = time.time()

    # Optional: skip on weekends if desired. Kuwait trades Sun-Thu.
    # Python weekday(): Mon=0 .. Sun=6. Kuwait weekend = Fri(4), Sat(5).
    weekday = datetime.now().weekday()
    if weekday in (4, 5):  # Friday, Saturday
        _log(f"Today is a Kuwait weekend (weekday={weekday}). Skipping recompute.", log_path)
        _log("Eagle Eye scheduled recompute SKIPPED (weekend).", log_path)
        return 0

    try:
        from app.services.eagle_eye.ingest import run_nightly_recompute

        _log("Calling run_nightly_recompute()...", log_path)
        result = run_nightly_recompute(dna_refresh=with_dna, verbose=False)

        elapsed = round(time.time() - t0, 1)
        ohlcv = result.get("ohlcv") if isinstance(result, dict) else {}
        ohlcv_ok = int((ohlcv or {}).get("ok", 0) or 0)
        ohlcv_errors = int((ohlcv or {}).get("errors", 0) or 0)
        status = str(result.get("status", "ok") if isinstance(result, dict) else "ok")

        _log(f"run_nightly_recompute returned: {result}", log_path)

        if status == "failure" or ohlcv_errors > 0 or ohlcv_ok == 0:
            _log(f"FAILURE in {elapsed}s - mode: {mode} - OHLCV ok={ohlcv_ok} errors={ohlcv_errors}", log_path)
            _log("Scheduler aborted because fresh market data was unavailable or ingest returned errors.", log_path)
            _log("=" * 60, log_path)
            return 1

        _log(f"SUCCESS in {elapsed}s - mode: {mode}", log_path)

        # Quick post-run verification: did a snapshot land for today?
        try:
            from app.core.database import query_val

            snap_count = query_val(
                "SELECT COUNT(*) FROM ee_rating_snapshots WHERE snapshot_date = ?",
                (today,),
            )
            _log(f"Snapshot rows for {today}: {snap_count}", log_path)
            if not snap_count:
                _log("WARNING: no snapshot rows captured for today - check tracker hook.", log_path)
        except Exception as exc:
            _log(f"Snapshot verification check failed (non-fatal): {exc}", log_path)

        _log("=" * 60, log_path)
        return 0

    except Exception as exc:
        elapsed = round(time.time() - t0, 1)
        _log(f"FAILED after {elapsed}s: {type(exc).__name__}: {exc}", log_path)
        import traceback

        _log(traceback.format_exc(), log_path)
        _log("=" * 60, log_path)
        return 1


if __name__ == "__main__":
    sys.exit(main())
