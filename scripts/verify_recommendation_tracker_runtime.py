"""
Quick runtime verification for Eagle Eye recommendation tracker integration.

Checks:
1) load_ohlcv date accessibility (column vs index)
2) query_all row shape (dict-like access)
3) tracker snapshot idempotency for today's date
4) signal tracker row count and sample

Usage:
    python scripts/verify_recommendation_tracker_runtime.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.database import query_all
from app.services.eagle_eye.recommendation_tracker import post_compute_snapshot
from app.services.eagle_eye.store import load_ohlcv


def _check_ohlcv_date_shape(ticker: str = "ZAIN") -> None:
    df = load_ohlcv(ticker)
    if df is None or df.empty:
        print(f"[WARN] load_ohlcv({ticker}) returned no data")
        return

    cols = list(df.columns)
    print(f"[OHLCV] {ticker} rows={len(df)}")
    print(f"[OHLCV] columns={cols}")
    print(f"[OHLCV] index_type={type(df.index).__name__} index_name={df.index.name}")

    date_candidates = {"date", "bar_date", "timestamp", "datetime", "index"}
    has_date_col = any(c in date_candidates for c in cols)
    has_date_index = str(getattr(df.index, "name", "") or "").lower() in date_candidates

    if has_date_col:
        print("[OK] Date is available as a column")
    elif has_date_index:
        print("[OK] Date is available via index name")
    else:
        print("[WARN] Date not clearly exposed as known column/index name")


def _check_query_all_shape() -> None:
    rows = query_all("SELECT ticker, rating, confidence FROM ee_ratings_cache LIMIT 3", ())
    print(f"[DB] query_all type={type(rows).__name__} len={len(rows) if rows else 0}")
    if not rows:
        print("[WARN] ee_ratings_cache is empty")
        return

    first = rows[0]
    has_get = hasattr(first, "get")
    has_keys = hasattr(first, "keys")
    print(f"[DB] first row type={type(first).__name__} has_get={has_get} has_keys={has_keys}")
    if has_get:
        print(f"[DB] first row ticker via .get -> {first.get('ticker')}")


def _check_snapshot_idempotency() -> None:
    run_date = datetime.now().strftime("%Y-%m-%d")
    run_id_1 = f"verify_1_{int(datetime.now().timestamp())}"
    run_id_2 = f"verify_2_{int(datetime.now().timestamp())}"

    s1 = post_compute_snapshot(run_id=run_id_1, run_date=run_date)
    s2 = post_compute_snapshot(run_id=run_id_2, run_date=run_date)

    print(f"[TRACKER] first call stats={s1}")
    print(f"[TRACKER] second call stats={s2}")

    snap_count = query_all(
        "SELECT COUNT(*) AS cnt FROM ee_rating_snapshots WHERE snapshot_date = ?",
        (run_date,),
    )
    cnt = int((snap_count or [{}])[0].get("cnt", 0))
    print(f"[TRACKER] snapshot rows today={cnt}")

    sig_count = query_all("SELECT COUNT(*) AS cnt FROM ee_signal_tracker", ())
    sig_cnt = int((sig_count or [{}])[0].get("cnt", 0))
    print(f"[TRACKER] signal tracker total rows={sig_cnt}")

    sample = query_all(
        """
        SELECT ticker, signal_type, signal_date, status, pnl_20d_pct
        FROM ee_signal_tracker
        ORDER BY id DESC LIMIT 5
        """,
        (),
    )
    print("[TRACKER] latest signals sample:")
    for r in sample or []:
        print(
            f"  - {r.get('ticker')} {r.get('signal_type')} {r.get('signal_date')} "
            f"status={r.get('status')} pnl20={r.get('pnl_20d_pct')}"
        )


def main() -> None:
    print("== Eagle Eye Recommendation Tracker Runtime Check ==")
    _check_ohlcv_date_shape("ZAIN")
    _check_query_all_shape()
    _check_snapshot_idempotency()


if __name__ == "__main__":
    main()
