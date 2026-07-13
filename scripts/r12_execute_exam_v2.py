from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SET_A = ["TIJARA", "BPCC", "ZAIN", "SANAM", "MABANEE"]
SET_B = ["KRE", "IFA", "SPEC", "CGC", "THURAYA"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def run_backtest_subprocess(runtime_db: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="r12_run2_") as td:
        out = Path(td) / "run.json"
        code = f"""
import json, sqlite3
from app.services.eagle_eye.backtest_service import run_backtest
from app.services.eagle_eye.market_data_service import get_active_config

with sqlite3.connect(r'{runtime_db.as_posix()}') as conn:
    mn,mx=conn.execute('SELECT MIN(trade_date),MAX(trade_date) FROM ee_ohlcv').fetchone()
    syms=[r[0] for r in conn.execute('SELECT DISTINCT symbol FROM ee_ohlcv ORDER BY symbol').fetchall()]

report=run_backtest(symbols=syms,start=int(mn),end=int(mx))
cfg=get_active_config()
print(json.dumps({{'report':report,'start':int(mn),'end':int(mx),'symbols':syms,'config':cfg}},ensure_ascii=True))
"""
        env = dict(os.environ)
        env["DATABASE_PATH"] = str(runtime_db)
        env["ENVIRONMENT"] = "test"
        cp = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, env=env)
        if cp.returncode != 0:
            return {
                "ok": False,
                "stdout": cp.stdout,
                "stderr": cp.stderr,
                "returncode": cp.returncode,
            }
        parsed = json.loads(cp.stdout.strip().splitlines()[-1])
        return {"ok": True, **parsed, "stdout": cp.stdout, "stderr": cp.stderr, "returncode": cp.returncode}


def summarize_symbol_trades(rows: list[sqlite3.Row], symbols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in symbols:
        rr = [r for r in rows if str(r["original_symbol"]) == s]
        vals = [float(r["net_return"] or 0.0) for r in rr]
        compounded = 1.0
        for v in vals:
            compounded *= 1.0 + v
        out[s] = {
            "trades": len(rr),
            "mean_net_return": (sum(vals) / len(vals)) if vals else None,
            "compounded_net_return": compounded - 1.0,
        }
    return out


def build_parity_suite(conn: sqlite3.Connection, symbols: list[str], fill_cost: float) -> dict[str, Any]:
    def load_unmasked(sym: str) -> list[tuple[int, float, float, float]]:
        return conn.execute(
            "SELECT trade_date, open, close, volume FROM ee_ohlcv_masked_source WHERE symbol=? AND is_masked=0 ORDER BY trade_date",
            (sym,),
        ).fetchall()

    def bh(sym: str) -> float | None:
        rows = load_unmasked(sym)
        if len(rows) < 2:
            return None
        e = float(rows[0][2])
        x = float(rows[-1][2])
        if e <= 0:
            return None
        return (x / e) - 1.0 - (2.0 * fill_cost)

    def rnd(sym: str, seed: int) -> float | None:
        rows = load_unmasked(sym)
        if len(rows) < 5:
            return None
        i = random.Random(seed).randint(1, len(rows) - 2)
        e = float(rows[i][1] or rows[i][2])
        x = float(rows[-1][2])
        if e <= 0:
            return None
        return (x / e) - 1.0 - (2.0 * fill_cost)

    vals_bh = [v for v in [bh(s) for s in symbols] if v is not None]
    vals_rnd = [v for i, s in enumerate(symbols) if (v := rnd(s, 20260711 + i)) is not None]

    return {
        "NO_TRADE_BENCHMARK": {"net_return": 0.0},
        "BUY_AND_HOLD_PER_ELIGIBLE_SYMBOL": {
            "eligible_symbol_count": len(vals_bh),
            "mean_net_return": (sum(vals_bh) / len(vals_bh)) if vals_bh else None,
        },
        "RANDOM_ELIGIBLE_ENTRY_BENCHMARK": {
            "eligible_symbol_count": len(vals_rnd),
            "mean_net_return": (sum(vals_rnd) / len(vals_rnd)) if vals_rnd else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="R12 run2 execution")
    parser.add_argument("--canonical-db", required=True)
    parser.add_argument("--review-dir", required=True)
    args = parser.parse_args()

    canonical_db = Path(args.canonical_db).resolve()
    review_dir = Path(args.review_dir).resolve()

    prep_path = review_dir / "r12_run2_preparation_v4_5.json"
    seal_path = review_dir / "r12_pre_exam_surface_seal_v4_4.json"
    exam_v45 = review_dir / "r12_exam_surface_v4_5.db"
    runtime_db = review_dir / "r12_exam_surface_v4_5_runtime.db"

    prep = load_json(prep_path)
    if not bool(prep.get("ready_for_run2")):
        raise RuntimeError("Run2 preparation gate did not pass")

    if runtime_db.exists():
        runtime_db.unlink()
    shutil.copy2(exam_v45, runtime_db)

    with sqlite3.connect(runtime_db) as conn:
        conn.execute("ALTER TABLE ee_ohlcv RENAME TO ee_ohlcv_masked_source")
        conn.execute("CREATE TABLE ee_ohlcv AS SELECT * FROM ee_ohlcv_unmasked_segmented")
        conn.commit()

    proc = run_backtest_subprocess(runtime_db)

    seal_hash = sha256_file(seal_path)
    v45_hash = sha256_file(exam_v45)

    if not proc.get("ok"):
        payload = {
            "version_id": "R12_EXAM_RESULTS_V2",
            "run_status": "FAILED_TECHNICAL",
            "authorization": {"r12": "AUTHORIZED", "r13": "NOT_AUTHORIZED"},
            "run_configuration": {
                "seal_v4_4_sha256": seal_hash,
                "exam_surface_v4_5_sha256": v45_hash,
                "runtime_db_path": str(runtime_db),
                "database_path_assertion": str(runtime_db),
                "set_a": SET_A,
                "set_b": SET_B,
            },
            "full_universe_statistics": None,
            "per_symbol_results": {"set_a": None, "set_b": None},
            "benchmark_parity_suite": None,
            "trade_ledger": [],
            "segmentation_statistics": prep.get("mask_semantics", {}).get("v45_surface", {}),
            "technical_anomalies": [
                "Run2 subprocess failed before report completion.",
                proc.get("stderr") or "",
            ],
        }
        dump_json(review_dir / "r12_exam_results_v2.json", payload)
        (review_dir / "r12_exam_results_v2.md").write_text(
            "# R12 Exam Results V2\n\n- run_status: FAILED_TECHNICAL\n- technical_anomalies: see JSON\n",
            encoding="utf-8",
            newline="\n",
        )
        print("R12_RUN2_STATUS FAILED_TECHNICAL")
        return 2

    report = proc["report"]
    cfg = proc["config"]

    with sqlite3.connect(runtime_db) as conn:
        conn.row_factory = sqlite3.Row
        trade_rows = conn.execute(
            """
            SELECT run_id, symbol AS segment_symbol,
                   CASE WHEN instr(symbol, '__SEG')>0 THEN substr(symbol,1,instr(symbol,'__SEG')-1) ELSE symbol END AS original_symbol,
                   opened_at, closed_at, side, tranches_json, avg_entry, avg_exit, gross_return, net_return, exit_reason
            FROM ee_backtest_trades
            WHERE run_id = ?
            ORDER BY opened_at, symbol
            """,
            (int(report["run_id"]),),
        ).fetchall()

        fill_cost = (float(cfg.get("bt_commission_bps", 25.0)) + float(cfg.get("bt_slippage_bps", 30.0))) / 10000.0

        symbols = sorted(str(r[0]) for r in conn.execute("SELECT DISTINCT CASE WHEN instr(symbol, '__SEG')>0 THEN substr(symbol,1,instr(symbol,'__SEG')-1) ELSE symbol END s FROM ee_ohlcv GROUP BY s ORDER BY s").fetchall())
        parity_full = build_parity_suite(conn, symbols, fill_cost)
        parity_set_a = build_parity_suite(conn, [s for s in SET_A if s in symbols], fill_cost)
        parity_set_b = build_parity_suite(conn, [s for s in SET_B if s in symbols], fill_cost)

        mabanee_rows = conn.execute(
            "SELECT trade_date, close FROM ee_ohlcv_masked_source WHERE symbol='MABANEE' AND is_masked=0 ORDER BY trade_date"
        ).fetchall()
        mabanee_bh = None
        if len(mabanee_rows) >= 2:
            e = float(mabanee_rows[0][1])
            x = float(mabanee_rows[-1][1])
            mabanee_bh = {
                "entry_trade_date": int(mabanee_rows[0][0]),
                "exit_trade_date": int(mabanee_rows[-1][0]),
                "net_return": (x / e) - 1.0 - (2.0 * fill_cost) if e > 0 else None,
            }

    payload = {
        "version_id": "R12_EXAM_RESULTS_V2",
        "run_status": "EXECUTED",
        "authorization": {"r12": "AUTHORIZED", "r13": "NOT_AUTHORIZED"},
        "run_configuration": {
            "seal_v4_4_sha256": seal_hash,
            "exam_surface_v4_5_sha256": v45_hash,
            "runtime_db_path": str(runtime_db),
            "database_path_assertion": str(runtime_db),
            "start_trade_date": int(proc["start"]),
            "end_trade_date": int(proc["end"]),
            "segment_symbol_count": len(proc["symbols"]),
            "set_a": SET_A,
            "set_b": SET_B,
            "real_costs": {
                "bt_commission_bps": float(cfg.get("bt_commission_bps", 25.0)),
                "bt_slippage_bps": float(cfg.get("bt_slippage_bps", 30.0)),
            },
        },
        "full_universe_statistics": report,
        "per_symbol_results": {
            "set_a": summarize_symbol_trades(trade_rows, SET_A),
            "set_b": summarize_symbol_trades(trade_rows, SET_B),
        },
        "benchmark_parity_suite": {
            "full_universe": parity_full,
            "set_a": parity_set_a,
            "set_b": parity_set_b,
            "mabanee_full_lifecycle_benchmark": mabanee_bh,
        },
        "trade_ledger": [
            {
                "run_id": int(r["run_id"]),
                "segment_symbol": str(r["segment_symbol"]),
                "original_symbol": str(r["original_symbol"]),
                "opened_at": int(r["opened_at"]),
                "closed_at": int(r["closed_at"]),
                "side": str(r["side"]),
                "tranches_json": str(r["tranches_json"]),
                "avg_entry": float(r["avg_entry"]),
                "avg_exit": float(r["avg_exit"]),
                "gross_return": float(r["gross_return"]),
                "net_return": float(r["net_return"]),
                "exit_reason": str(r["exit_reason"]),
            }
            for r in trade_rows
        ],
        "segmentation_statistics": prep.get("mask_semantics", {}).get("v45_surface", {}),
        "technical_anomalies": [],
    }

    out_json = review_dir / "r12_exam_results_v2.json"
    out_md = review_dir / "r12_exam_results_v2.md"
    dump_json(out_json, payload)

    md = [
        "# R12 Exam Results V2",
        "",
        f"- run_status: {payload['run_status']}",
        f"- seal_v4_4_sha256: {seal_hash}",
        f"- exam_surface_v4_5_sha256: {v45_hash}",
        f"- trades: {payload['full_universe_statistics']['trades']}",
        f"- win_rate: {payload['full_universe_statistics']['win_rate']}",
        f"- expectancy: {payload['full_universe_statistics']['expectancy']}",
        f"- max_drawdown: {payload['full_universe_statistics']['max_drawdown']}",
        "",
        "## Segmentation",
        f"- masked_bar_count: {payload['segmentation_statistics'].get('masked_bar_count')}",
        f"- unmasked_segmented_bar_count: {payload['segmentation_statistics'].get('unmasked_segmented_bar_count')}",
        "",
    ]
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")

    print("R12_RUN2_STATUS EXECUTED")
    print("SEAL_HASH", seal_hash)
    print("V45_HASH", v45_hash)
    print("HEADLINE", json.dumps({
        "trades": payload["full_universe_statistics"]["trades"],
        "win_rate": payload["full_universe_statistics"]["win_rate"],
        "expectancy": payload["full_universe_statistics"]["expectancy"],
        "max_drawdown": payload["full_universe_statistics"]["max_drawdown"],
    }, sort_keys=True))
    print("SET_A", json.dumps(payload["per_symbol_results"]["set_a"], sort_keys=True))
    print("SET_B", json.dumps(payload["per_symbol_results"]["set_b"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
