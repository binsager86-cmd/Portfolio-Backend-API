from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sqlite3
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.eagle_eye.backtest_service import run_backtest  # noqa: E402

SET_A = ["TIJARA", "BPCC", "ZAIN", "SANAM", "MABANEE"]
SET_B = ["KRE", "IFA", "SPEC", "CGC", "THURAYA"]


def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


def iso_to_ts(iso: str) -> int:
    return to_ts(date.fromisoformat(iso))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_symbol_bars(conn: sqlite3.Connection, symbol: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT trade_date, open, close, volume
        FROM ee_ohlcv
        WHERE symbol = ?
        ORDER BY trade_date
        """,
        (symbol,),
    ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "trade_date": int(r[0]),
                "open": float(r[1] if r[1] is not None else 0.0),
                "close": float(r[2] if r[2] is not None else 0.0),
                "volume": float(r[3] if r[3] is not None else 0.0),
            }
        )
    return out


def buy_hold_return(bars: list[dict[str, Any]], fill_cost: float) -> dict[str, Any] | None:
    if len(bars) < 2:
        return None
    entry = float(bars[0]["close"])
    exit_ = float(bars[-1]["close"])
    if entry <= 0:
        return None
    gross = (exit_ / entry) - 1.0
    net = gross - (2.0 * fill_cost)
    return {
        "entry_trade_date": int(bars[0]["trade_date"]),
        "entry_price": entry,
        "exit_trade_date": int(bars[-1]["trade_date"]),
        "exit_price": exit_,
        "gross_return": gross,
        "net_return": net,
    }


def breakout_return(bars: list[dict[str, Any]], fill_cost: float, with_volume_gate: bool) -> dict[str, Any] | None:
    if len(bars) < 25:
        return None
    entry_idx = None
    for i in range(20, len(bars) - 1):
        prev20 = bars[i - 20 : i]
        max_prev = max(float(x["close"]) for x in prev20)
        close_i = float(bars[i]["close"])
        cond = close_i > max_prev
        if with_volume_gate:
            avg_vol = sum(float(x["volume"]) for x in prev20) / 20.0
            cond = cond and float(bars[i]["volume"]) > avg_vol
        if cond:
            entry_idx = i + 1
            break
    if entry_idx is None or entry_idx >= len(bars):
        return None
    entry = float(bars[entry_idx]["open"] or bars[entry_idx]["close"])
    exit_ = float(bars[-1]["close"])
    if entry <= 0:
        return None
    gross = (exit_ / entry) - 1.0
    net = gross - (2.0 * fill_cost)
    return {
        "entry_trade_date": int(bars[entry_idx]["trade_date"]),
        "entry_price": entry,
        "exit_trade_date": int(bars[-1]["trade_date"]),
        "exit_price": exit_,
        "gross_return": gross,
        "net_return": net,
    }


def random_entry_return(bars: list[dict[str, Any]], fill_cost: float, seed: int) -> dict[str, Any] | None:
    if len(bars) < 5:
        return None
    rng = random.Random(seed)
    entry_idx = rng.randint(1, len(bars) - 2)
    entry = float(bars[entry_idx]["open"] or bars[entry_idx]["close"])
    exit_ = float(bars[-1]["close"])
    if entry <= 0:
        return None
    gross = (exit_ / entry) - 1.0
    net = gross - (2.0 * fill_cost)
    return {
        "entry_trade_date": int(bars[entry_idx]["trade_date"]),
        "entry_price": entry,
        "exit_trade_date": int(bars[-1]["trade_date"]),
        "exit_price": exit_,
        "gross_return": gross,
        "net_return": net,
    }


def mean_or_none(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def build_benchmark_suite(conn: sqlite3.Connection, symbols: list[str], fill_cost: float, seed_base: int) -> dict[str, Any]:
    bars_map = {s: load_symbol_bars(conn, s) for s in symbols}
    eligible = [s for s in symbols if len(bars_map[s]) >= 2]

    buy_hold_rows = {s: buy_hold_return(bars_map[s], fill_cost) for s in eligible}
    buy_hold_vals = [float(v["net_return"]) for v in buy_hold_rows.values() if v is not None]

    breakout_rows = {s: breakout_return(bars_map[s], fill_cost, with_volume_gate=False) for s in eligible}
    breakout_vals = [float(v["net_return"]) for v in breakout_rows.values() if v is not None]

    breakout_vol_rows = {s: breakout_return(bars_map[s], fill_cost, with_volume_gate=True) for s in eligible}
    breakout_vol_vals = [float(v["net_return"]) for v in breakout_vol_rows.values() if v is not None]

    random_rows = {s: random_entry_return(bars_map[s], fill_cost, seed_base + i) for i, s in enumerate(eligible)}
    random_vals = [float(v["net_return"]) for v in random_rows.values() if v is not None]

    rng = random.Random(seed_base + 997)
    k = min(5, len(eligible))
    topk_syms = sorted(rng.sample(eligible, k)) if k > 0 else []
    topk_vals = [float(buy_hold_rows[s]["net_return"]) for s in topk_syms if buy_hold_rows[s] is not None]

    return {
        "symbol_count": len(symbols),
        "eligible_symbol_count": len(eligible),
        "NO_TRADE_BENCHMARK": {"net_return": 0.0},
        "BUY_AND_HOLD_PER_ELIGIBLE_SYMBOL": {
            "mean_net_return": mean_or_none(buy_hold_vals),
            "symbol_details": {k: v for k, v in buy_hold_rows.items() if v is not None},
        },
        "SIMPLE_PRICE_BREAKOUT_BENCHMARK": {
            "mean_net_return": mean_or_none(breakout_vals),
            "triggered_symbol_count": len([1 for v in breakout_rows.values() if v is not None]),
        },
        "PRICE_PLUS_RELATIVE_VOLUME_BENCHMARK": {
            "mean_net_return": mean_or_none(breakout_vol_vals),
            "triggered_symbol_count": len([1 for v in breakout_vol_rows.values() if v is not None]),
        },
        "RANDOM_ELIGIBLE_ENTRY_BENCHMARK": {
            "seed_rule": f"seed={seed_base}+symbol_index",
            "mean_net_return": mean_or_none(random_vals),
        },
        "RANDOM_TOP_K_PORTFOLIO_BENCHMARK": {
            "seed": seed_base + 997,
            "k": k,
            "symbols": topk_syms,
            "mean_net_return": mean_or_none(topk_vals),
        },
    }


def summarize_symbol_trades(trade_rows: list[sqlite3.Row], symbols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in symbols:
        rows = [r for r in trade_rows if str(r["symbol"]) == s]
        rets = [float(r["net_return"] or 0.0) for r in rows]
        compounded = 1.0
        for rr in rets:
            compounded *= 1.0 + rr
        out[s] = {
            "trades": len(rows),
            "mean_net_return": mean_or_none(rets),
            "compounded_net_return": compounded - 1.0,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="R12 execution v1 (owner authorized)")
    parser.add_argument("--clean-db", required=True)
    parser.add_argument("--review-dir", required=True)
    args = parser.parse_args()

    clean_db = Path(args.clean_db).resolve()
    review_dir = Path(args.review_dir).resolve()

    seal_path = review_dir / "r12_pre_exam_surface_seal_v4_4.json"
    mask_path = review_dir / "r12_masked_intervals_manifest_v4_3_final.json"
    triage_path = review_dir / "r12_breach_triage_v4_2_FINAL.json"

    seal = load_json(seal_path)
    mask = load_json(mask_path)
    triage = load_json(triage_path)

    exam_db = review_dir / "r12_exam_surface_v4_4.db"
    shutil.copy2(clean_db, exam_db)

    masked_rows_deleted = 0
    with sqlite3.connect(exam_db) as conn:
        for m in mask["intervals"]:
            symbol = str(m["symbol"])
            start_ts = iso_to_ts(str(m["start_date"]))
            end_ts = iso_to_ts(str(m["end_date"]))
            cur = conn.execute(
                "DELETE FROM ee_ohlcv WHERE symbol = ? AND trade_date >= ? AND trade_date <= ?",
                (symbol, start_ts, end_ts),
            )
            masked_rows_deleted += int(cur.rowcount or 0)
        conn.commit()

    os.environ["DATABASE_PATH"] = str(exam_db)
    os.environ["ENVIRONMENT"] = "test"

    with sqlite3.connect(exam_db) as conn:
        conn.row_factory = sqlite3.Row
        bounds = conn.execute("SELECT MIN(trade_date) AS mn, MAX(trade_date) AS mx FROM ee_ohlcv").fetchone()
        if bounds is None or bounds["mn"] is None or bounds["mx"] is None:
            raise RuntimeError("No unmasked bars remain in exam surface")
        start_ts = int(bounds["mn"])
        end_ts = int(bounds["mx"])
        symbols = sorted(str(r[0]) for r in conn.execute("SELECT DISTINCT symbol FROM ee_ohlcv ORDER BY symbol").fetchall())

    report = run_backtest(symbols=symbols, start=start_ts, end=end_ts)

    with sqlite3.connect(exam_db) as conn:
        conn.row_factory = sqlite3.Row
        trade_rows = conn.execute(
            """
            SELECT run_id, symbol, opened_at, closed_at, side, tranches_json, avg_entry, avg_exit, gross_return, net_return, exit_reason
            FROM ee_backtest_trades
            WHERE run_id = ?
            ORDER BY opened_at, symbol
            """,
            (int(report["run_id"]),),
        ).fetchall()

        fill_cost = (float(conn.execute("SELECT 1").fetchone()[0]) * 0.0)
        # Real costs are sourced from the frozen engine config used by run_backtest.
        # For benchmark comparators, replicate cost burden from current config snapshot.
        from app.services.eagle_eye.market_data_service import get_active_config  # local import to avoid import-time side effects

        cfg = get_active_config()
        fill_cost = (float(cfg.get("bt_commission_bps", 25.0)) + float(cfg.get("bt_slippage_bps", 30.0))) / 10000.0

        benchmark_full = build_benchmark_suite(conn, symbols, fill_cost, seed_base=20260711)
        benchmark_set_a = build_benchmark_suite(conn, [s for s in SET_A if s in symbols], fill_cost, seed_base=20260712)
        benchmark_set_b = build_benchmark_suite(conn, [s for s in SET_B if s in symbols], fill_cost, seed_base=20260713)

        mabanee_bh = None
        if "MABANEE" in symbols:
            mabanee_bh = buy_hold_return(load_symbol_bars(conn, "MABANEE"), fill_cost)

    technical_anomalies: list[str] = []
    if len(mask["intervals"]) != int(seal["masked_interval_count"]):
        technical_anomalies.append("Seal masked_interval_count differs from v4.3 final mask manifest interval_count")

    disposition_counts = triage.get("final", {}).get("disposition_counts", {})

    result_payload = {
        "version_id": "R12_EXAM_RESULTS_V1",
        "run_status": "EXECUTED",
        "authorization": {
            "r12": "AUTHORIZED",
            "r13": "NOT_AUTHORIZED",
        },
        "frozen_constraints": {
            "single_run": True,
            "no_parameter_sweeps": True,
            "set_b_reported_as_is": True,
            "full_universe_unmasked_bars_only": True,
        },
        "input_surface": {
            "seal_path": "artifacts/preview1a_prestart/review_final/r12_pre_exam_surface_seal_v4_4.json",
            "seal_sha256": sha256_file(seal_path),
            "calendar_version_id": "BK_CAL_V4_1783783330",
            "mask_manifest_path": "artifacts/preview1a_prestart/review_final/r12_masked_intervals_manifest_v4_3_final.json",
            "masked_interval_count": int(mask["interval_count"]),
            "masked_rows_deleted_from_exam_db": masked_rows_deleted,
            "triage_disposition_counts": disposition_counts,
        },
        "run_configuration": {
            "database_path": str(exam_db),
            "start_trade_date": start_ts,
            "end_trade_date": end_ts,
            "symbol_count": len(symbols),
            "set_a": SET_A,
            "set_b": SET_B,
        },
        "full_universe_statistics": report,
        "per_symbol_results": {
            "set_a": summarize_symbol_trades(trade_rows, [s for s in SET_A if s in symbols]),
            "set_b": summarize_symbol_trades(trade_rows, [s for s in SET_B if s in symbols]),
        },
        "benchmark_parity_suite": {
            "full_universe": benchmark_full,
            "set_a": benchmark_set_a,
            "set_b": benchmark_set_b,
            "mabanee_full_lifecycle_benchmark": mabanee_bh,
        },
        "trade_ledger": [
            {
                "run_id": int(r["run_id"]),
                "symbol": str(r["symbol"]),
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
        "technical_anomalies": technical_anomalies,
    }

    out_json = review_dir / "r12_exam_results_v1.json"
    out_md = review_dir / "r12_exam_results_v1.md"
    dump_json(out_json, result_payload)

    md_lines = [
        "# R12 Exam Results V1",
        "",
        f"- run_status: {result_payload['run_status']}",
        f"- seal_sha256: {result_payload['input_surface']['seal_sha256']}",
        f"- symbol_count: {result_payload['run_configuration']['symbol_count']}",
        f"- trades: {result_payload['full_universe_statistics']['trades']}",
        f"- win_rate: {result_payload['full_universe_statistics']['win_rate']}",
        f"- expectancy: {result_payload['full_universe_statistics']['expectancy']}",
        f"- max_drawdown: {result_payload['full_universe_statistics']['max_drawdown']}",
        "",
        "## Set A Per-Symbol",
        json.dumps(result_payload["per_symbol_results"]["set_a"], ensure_ascii=True, indent=2),
        "",
        "## Set B Per-Symbol",
        json.dumps(result_payload["per_symbol_results"]["set_b"], ensure_ascii=True, indent=2),
        "",
        "## Technical Anomalies",
        json.dumps(result_payload["technical_anomalies"], ensure_ascii=True, indent=2),
        "",
    ]
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8", newline="\n")

    print("R12_EXECUTION_STATUS EXECUTED")
    print("SEAL_SHA256", result_payload["input_surface"]["seal_sha256"])
    print("HEADLINE", json.dumps({
        "trades": result_payload["full_universe_statistics"]["trades"],
        "win_rate": result_payload["full_universe_statistics"]["win_rate"],
        "expectancy": result_payload["full_universe_statistics"]["expectancy"],
        "max_drawdown": result_payload["full_universe_statistics"]["max_drawdown"],
    }, sort_keys=True))
    print("SET_A_SUMMARY", json.dumps(result_payload["per_symbol_results"]["set_a"], sort_keys=True))
    print("SET_B_SUMMARY", json.dumps(result_payload["per_symbol_results"]["set_b"], sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
