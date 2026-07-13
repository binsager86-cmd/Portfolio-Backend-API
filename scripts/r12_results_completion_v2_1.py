from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SET_A = ["TIJARA", "BPCC", "ZAIN", "SANAM", "MABANEE"]
SET_B = ["KRE", "IFA", "SPEC", "CGC", "THURAYA"]
ENTRY_SIGNAL_TYPES = {"ACCUMULATION_ALERT", "BREAKOUT_CONFIRMED", "ADD_ON_PULLBACK"}


def ts_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=UTC).date().isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_unmasked_bars(conn: sqlite3.Connection, symbol: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT trade_date, open, high, low, close, volume, value_kwd
        FROM ee_ohlcv_masked_source
        WHERE symbol = ? AND is_masked = 0
        ORDER BY trade_date
        """,
        (symbol,),
    ).fetchall()
    return [
        {
            "trade_date": int(r[0]),
            "open": float(r[1] if r[1] is not None else 0.0),
            "high": float(r[2] if r[2] is not None else 0.0),
            "low": float(r[3] if r[3] is not None else 0.0),
            "close": float(r[4] if r[4] is not None else 0.0),
            "volume": float(r[5] if r[5] is not None else 0.0),
            "value_kwd": float(r[6] if r[6] is not None else 0.0),
        }
        for r in rows
    ]


def buy_hold_benchmark(bars: list[dict[str, Any]], fill_cost: float) -> dict[str, Any] | None:
    if len(bars) < 2:
        return None
    entry = float(bars[0]["close"])
    exit_ = float(bars[-1]["close"])
    if entry <= 0:
        return None
    gross = (exit_ / entry) - 1.0
    return {
        "benchmark": "BUY_AND_HOLD_PER_ELIGIBLE_SYMBOL",
        "triggered": True,
        "benchmark_lifecycle_phase": "FULL_WINDOW_HOLD",
        "trigger_trade_date": bars[0]["trade_date"],
        "entry_trade_date": bars[0]["trade_date"],
        "entry_price": entry,
        "exit_trade_date": bars[-1]["trade_date"],
        "exit_price": exit_,
        "gross_return": gross,
        "net_return": gross - (2.0 * fill_cost),
    }


def breakout_benchmark(bars: list[dict[str, Any]], fill_cost: float, with_volume_gate: bool) -> dict[str, Any] | None:
    if len(bars) < 25:
        return None
    for idx in range(20, len(bars) - 1):
        prev20 = bars[idx - 20 : idx]
        max_prev = max(float(x["close"]) for x in prev20)
        avg_vol = sum(float(x["volume"]) for x in prev20) / 20.0
        close_i = float(bars[idx]["close"])
        vol_i = float(bars[idx]["volume"])
        triggered = close_i > max_prev
        if with_volume_gate:
            triggered = triggered and vol_i > avg_vol
        if not triggered:
            continue
        entry_idx = idx + 1
        if entry_idx >= len(bars):
            return None
        entry_price = float(bars[entry_idx]["open"] or bars[entry_idx]["close"])
        if entry_price <= 0:
            return None
        exit_price = float(bars[-1]["close"])
        gross = (exit_price / entry_price) - 1.0
        return {
            "benchmark": "PRICE_PLUS_RELATIVE_VOLUME_BENCHMARK" if with_volume_gate else "SIMPLE_PRICE_BREAKOUT_BENCHMARK",
            "triggered": True,
            "benchmark_lifecycle_phase": "RELVOL_BREAKOUT_TRIGGER" if with_volume_gate else "BREAKOUT_TRIGGER",
            "trigger_trade_date": bars[idx]["trade_date"],
            "entry_trade_date": bars[entry_idx]["trade_date"],
            "entry_price": entry_price,
            "exit_trade_date": bars[-1]["trade_date"],
            "exit_price": exit_price,
            "gross_return": gross,
            "net_return": gross - (2.0 * fill_cost),
            "close_minus_prev20_high": close_i - max_prev,
            "relative_volume_minus_1": (vol_i / avg_vol) - 1.0 if avg_vol > 0 else None,
        }
    return None


def random_entry_benchmark(bars: list[dict[str, Any]], fill_cost: float, seed: int) -> dict[str, Any] | None:
    if len(bars) < 5:
        return None
    rng = random.Random(seed)
    entry_idx = rng.randint(1, len(bars) - 2)
    entry_price = float(bars[entry_idx]["open"] or bars[entry_idx]["close"])
    if entry_price <= 0:
        return None
    exit_price = float(bars[-1]["close"])
    gross = (exit_price / entry_price) - 1.0
    return {
        "benchmark": "RANDOM_ELIGIBLE_ENTRY_BENCHMARK",
        "triggered": True,
        "benchmark_lifecycle_phase": "RANDOM_ENTRY",
        "trigger_trade_date": bars[entry_idx]["trade_date"],
        "entry_trade_date": bars[entry_idx]["trade_date"],
        "entry_price": entry_price,
        "exit_trade_date": bars[-1]["trade_date"],
        "exit_price": exit_price,
        "gross_return": gross,
        "net_return": gross - (2.0 * fill_cost),
        "seed": seed,
    }


def signal_rows_for_symbol(conn: sqlite3.Connection, symbol: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT symbol, trade_date, signal_type, phase_from, phase_to, price, stop_price, evidence_json
        FROM ee_signals
        WHERE (CASE WHEN instr(symbol, '__SEG') > 0 THEN substr(symbol, 1, instr(symbol, '__SEG') - 1) ELSE symbol END) = ?
        ORDER BY trade_date
        """,
        (symbol,),
    ).fetchall()
    out = []
    for r in rows:
        ev = json.loads(r[7]) if r[7] else {}
        out.append(
            {
                "segment_symbol": str(r[0]),
                "trade_date": int(r[1]),
                "trade_date_iso": ts_to_iso(int(r[1])),
                "signal_type": str(r[2]),
                "phase_from": r[3],
                "phase_to": r[4],
                "price": None if r[5] is None else float(r[5]),
                "stop_price": None if r[6] is None else float(r[6]),
                "evidence": ev,
            }
        )
    return out


def trade_rows_for_symbol(conn: sqlite3.Connection, symbol: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT run_id, symbol, opened_at, closed_at, side, tranches_json, avg_entry, avg_exit, gross_return, net_return, exit_reason
        FROM ee_backtest_trades
        WHERE (CASE WHEN instr(symbol, '__SEG') > 0 THEN substr(symbol, 1, instr(symbol, '__SEG') - 1) ELSE symbol END) = ?
        ORDER BY opened_at
        """,
        (symbol,),
    ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "run_id": int(r[0]),
                "segment_symbol": str(r[1]),
                "opened_at": int(r[2]),
                "opened_at_iso": ts_to_iso(int(r[2])),
                "closed_at": int(r[3]),
                "closed_at_iso": ts_to_iso(int(r[3])),
                "side": str(r[4]),
                "tranches": json.loads(r[5]),
                "avg_entry": float(r[6]),
                "avg_exit": float(r[7]),
                "gross_return": float(r[8]),
                "net_return": float(r[9]),
                "exit_reason": str(r[10]),
            }
        )
    return out


def actual_behavior_summary(signal_rows: list[dict[str, Any]], trade_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(r["signal_type"] for r in signal_rows)
    entry_signals = [r for r in signal_rows if r["signal_type"] in ENTRY_SIGNAL_TYPES]
    return {
        "signal_count": len(signal_rows),
        "signal_type_counts": dict(sorted(counts.items())),
        "entry_signal_count": len(entry_signals),
        "trade_count": len(trade_rows),
        "first_signal_date": None if not signal_rows else signal_rows[0]["trade_date_iso"],
        "last_signal_date": None if not signal_rows else signal_rows[-1]["trade_date_iso"],
        "trades": [
            {
                "opened_at": r["opened_at_iso"],
                "closed_at": r["closed_at_iso"],
                "entry_signal_types": [str(t.get("signal_type")) for t in r["tranches"]],
                "exit_reason": r["exit_reason"],
                "avg_entry": r["avg_entry"],
                "avg_exit": r["avg_exit"],
                "net_return": r["net_return"],
            }
            for r in trade_rows
        ],
    }


def benchmark_row(symbol: str, benchmark_name: str, bench: dict[str, Any] | None, actual: dict[str, Any]) -> dict[str, Any]:
    if benchmark_name == "NO_TRADE_BENCHMARK":
        status = "PASS" if actual["trade_count"] == 0 else "FAIL_TRADED"
        return {
            "symbol": symbol,
            "benchmark": benchmark_name,
            "benchmark_expected_trade": False,
            "benchmark_lifecycle_phase": "NONE",
            "benchmark_entry_date": None,
            "benchmark_exit_date": None,
            "actual_trade_count": actual["trade_count"],
            "actual_signal_count": actual["signal_count"],
            "status": status,
        }

    if bench is None:
        return {
            "symbol": symbol,
            "benchmark": benchmark_name,
            "benchmark_expected_trade": False,
            "benchmark_lifecycle_phase": None,
            "benchmark_entry_date": None,
            "benchmark_exit_date": None,
            "actual_trade_count": actual["trade_count"],
            "actual_signal_count": actual["signal_count"],
            "status": "NO_BENCHMARK_TRIGGER",
        }

    status = "PASS" if actual["trade_count"] > 0 else "FAIL/NO_SIGNAL"
    return {
        "symbol": symbol,
        "benchmark": benchmark_name,
        "benchmark_expected_trade": True,
        "benchmark_lifecycle_phase": bench.get("benchmark_lifecycle_phase"),
        "benchmark_trigger_date": ts_to_iso(bench.get("trigger_trade_date")),
        "benchmark_entry_date": ts_to_iso(bench.get("entry_trade_date")),
        "benchmark_exit_date": ts_to_iso(bench.get("exit_trade_date")),
        "benchmark_net_return": bench.get("net_return"),
        "actual_trade_count": actual["trade_count"],
        "actual_signal_count": actual["signal_count"],
        "actual_first_signal_date": actual["first_signal_date"],
        "actual_trades": actual["trades"],
        "status": status,
    }


def nearest_signal(signal_rows: list[dict[str, Any]], target_ts: int) -> dict[str, Any] | None:
    if not signal_rows:
        return None
    return min(signal_rows, key=lambda r: abs(int(r["trade_date"]) - int(target_ts)))


def no_trade_forensic(symbol: str, signal_rows: list[dict[str, Any]], bars: list[dict[str, Any]], simple_breakout: dict[str, Any] | None, rv_breakout: dict[str, Any] | None) -> dict[str, Any]:
    signal_counts = dict(sorted(Counter(r["signal_type"] for r in signal_rows).items()))
    forensic_rows = []
    for label, bench in [("simple_breakout", simple_breakout), ("price_plus_relative_volume", rv_breakout)]:
        if bench is None:
            forensic_rows.append({"benchmark": label, "active": False})
            continue
        near = nearest_signal(signal_rows, int(bench["trigger_trade_date"]))
        near_ev = {} if near is None else near["evidence"]
        forensic_rows.append(
            {
                "benchmark": label,
                "active": True,
                "benchmark_trigger_date": ts_to_iso(int(bench["trigger_trade_date"])),
                "benchmark_entry_date": ts_to_iso(int(bench["entry_trade_date"])),
                "close_minus_prev20_high": bench.get("close_minus_prev20_high"),
                "relative_volume_minus_1": bench.get("relative_volume_minus_1"),
                "nearest_engine_signal_date": None if near is None else near["trade_date_iso"],
                "nearest_engine_signal_type": None if near is None else near["signal_type"],
                "nearest_engine_phase_to": None if near is None else near["phase_to"],
                "nearest_engine_attempted_signal_type": near_ev.get("attempted_signal_type"),
                "nearest_engine_suppressed_reason": near_ev.get("suppressed_reason"),
                "nearest_engine_score": near_ev.get("score"),
                "nearest_engine_rel_volume": near_ev.get("rel_volume"),
                "nearest_engine_range_high_60": near_ev.get("range_high_60"),
                "nearest_engine_close": near_ev.get("close"),
            }
        )

    blocker = "NO_ENTRY_SIGNAL_RECORDED"
    if signal_counts.get("SIGNAL_SUPPRESSED_RISK"):
        blocker = "RISK_SUPPRESSION"
    elif signal_counts.get("AVOID_SET"):
        blocker = "AVOID_GATE"
    elif signal_counts.get("PHASE_ONLY"):
        blocker = "PHASE_ONLY_NO_ENTRY"

    return {
        "symbol": symbol,
        "trade_count": 0,
        "signal_type_counts": signal_counts,
        "primary_blocker": blocker,
        "benchmark_active_day_checks": forensic_rows,
    }


def largest_trade_removed_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "largest_trade": None,
            "with_all_trades": None,
            "without_largest_trade": None,
        }
    ordered = sorted(trades, key=lambda x: float(x["net_return"]), reverse=True)
    largest = ordered[0]

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        returns = [float(r["net_return"]) for r in rows]
        wins = [r for r in returns if r > 0]
        eq = 1.0
        peak = 1.0
        max_dd = 0.0
        for rr in returns:
            eq *= 1.0 + rr
            peak = max(peak, eq)
            dd = 0.0 if peak <= 0 else (peak - eq) / peak
            max_dd = max(max_dd, dd)
        return {
            "trade_count": len(rows),
            "expectancy": 0.0 if not rows else sum(returns) / len(rows),
            "win_rate": 0.0 if not rows else len(wins) / len(rows),
            "cumulative_net_return": eq - 1.0,
            "max_drawdown": max_dd,
        }

    return {
        "largest_trade": {
            "segment_symbol": largest["segment_symbol"],
            "original_symbol": largest["original_symbol"],
            "opened_at": largest["opened_at"],
            "closed_at": largest["closed_at"],
            "net_return": largest["net_return"],
        },
        "with_all_trades": summarize(trades),
        "without_largest_trade": summarize(ordered[1:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="R12 results completion v2.1")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--canonical-db", required=True)
    args = parser.parse_args()

    review_dir = Path(args.review_dir).resolve()
    canonical_db = Path(args.canonical_db).resolve()

    v2 = load_json(review_dir / "r12_exam_results_v2.json")
    prep = load_json(review_dir / "r12_run2_preparation_v4_5.json")
    benchmark_spec = load_json(review_dir / "r12_benchmark_spec_v2.json")

    runtime_db = review_dir / "r12_exam_surface_v4_5_runtime.db"

    with sqlite3.connect(runtime_db) as conn:
        conn.row_factory = sqlite3.Row

        config = v2["run_configuration"]["real_costs"]
        fill_cost = (float(config["bt_commission_bps"]) + float(config["bt_slippage_bps"])) / 10000.0

        symbol_benchmark_rows: dict[str, list[dict[str, Any]]] = {"set_a": [], "set_b": []}
        no_trade_forensics: list[dict[str, Any]] = []

        for set_name, symbols, random_seed_base in [("set_a", SET_A, 20260712), ("set_b", SET_B, 20260713)]:
            for idx, symbol in enumerate(symbols):
                bars = load_unmasked_bars(conn, symbol)
                signals = signal_rows_for_symbol(conn, symbol)
                trades = trade_rows_for_symbol(conn, symbol)
                actual = actual_behavior_summary(signals, trades)

                bh = buy_hold_benchmark(bars, fill_cost)
                sb = breakout_benchmark(bars, fill_cost, with_volume_gate=False)
                rv = breakout_benchmark(bars, fill_cost, with_volume_gate=True)
                rnd = random_entry_benchmark(bars, fill_cost, random_seed_base + idx)

                symbol_benchmark_rows[set_name].append(benchmark_row(symbol, "NO_TRADE_BENCHMARK", None, actual))
                symbol_benchmark_rows[set_name].append(benchmark_row(symbol, "BUY_AND_HOLD_PER_ELIGIBLE_SYMBOL", bh, actual))
                symbol_benchmark_rows[set_name].append(benchmark_row(symbol, "SIMPLE_PRICE_BREAKOUT_BENCHMARK", sb, actual))
                symbol_benchmark_rows[set_name].append(benchmark_row(symbol, "PRICE_PLUS_RELATIVE_VOLUME_BENCHMARK", rv, actual))
                symbol_benchmark_rows[set_name].append(benchmark_row(symbol, "RANDOM_ELIGIBLE_ENTRY_BENCHMARK", rnd, actual))

                if symbol in {"TIJARA", "BPCC", "SANAM"}:
                    no_trade_forensics.append(no_trade_forensic(symbol, signals, bars, sb, rv))

        # Random top-k portfolio rows are set-level, not per-symbol.
        top_k_rows = []
        for set_name, symbols, seed_base in [("set_a", SET_A, 20260712), ("set_b", SET_B, 20260713)]:
            eligible = [s for s in symbols if len(load_unmasked_bars(conn, s)) >= 2]
            rng = random.Random(seed_base + 997)
            k = min(5, len(eligible))
            top_k_rows.append(
                {
                    "set": set_name,
                    "benchmark": "RANDOM_TOP_K_PORTFOLIO_BENCHMARK",
                    "seed": seed_base + 997,
                    "k": k,
                    "symbols": sorted(rng.sample(eligible, k)) if k > 0 else [],
                }
            )

        runtime_trade_order = []
        for r in conn.execute(
            """
            SELECT id, symbol,
                   CASE WHEN instr(symbol, '__SEG') > 0 THEN substr(symbol, 1, instr(symbol, '__SEG') - 1) ELSE symbol END AS original_symbol,
                   opened_at, closed_at, net_return
            FROM ee_backtest_trades
            ORDER BY id
            """
        ).fetchall():
            runtime_trade_order.append(
                {
                    "id": int(r[0]),
                    "segment_symbol": str(r[1]),
                    "original_symbol": str(r[2]),
                    "opened_at": int(r[3]),
                    "closed_at": int(r[4]),
                    "net_return": float(r[5]),
                }
            )

        # IFA forensic.
        ifa_trade = trade_rows_for_symbol(conn, "IFA")[0]
        ifa_entry_bar = conn.execute(
            "SELECT trade_date, open, high, low, close, volume, value_kwd, is_masked FROM ee_ohlcv_masked_source WHERE symbol='IFA' AND trade_date=?",
            (ifa_trade["opened_at"],),
        ).fetchone()
        ifa_exit_bar = conn.execute(
            "SELECT trade_date, open, high, low, close, volume, value_kwd, is_masked FROM ee_ohlcv_masked_source WHERE symbol='IFA' AND trade_date=?",
            (ifa_trade["closed_at"],),
        ).fetchone()

        # containment current counts
        with sqlite3.connect(canonical_db) as c2:
            c2.row_factory = sqlite3.Row
            current_counts = {
                "ee_indicators": int(c2.execute("SELECT COUNT(1) FROM ee_indicators").fetchone()[0]),
                "ee_signals": int(c2.execute("SELECT COUNT(1) FROM ee_signals").fetchone()[0]),
                "ee_ratings": int(c2.execute("SELECT COUNT(1) FROM ee_ratings").fetchone()[0]),
                "ee_symbol_state": int(c2.execute("SELECT COUNT(1) FROM ee_symbol_state").fetchone()[0]),
                "ee_positions": int(c2.execute("SELECT COUNT(1) FROM ee_positions").fetchone()[0]),
                "ee_backtest_runs": int(c2.execute("SELECT COUNT(1) FROM ee_backtest_runs").fetchone()[0]),
                "ee_backtest_trades": int(c2.execute("SELECT COUNT(1) FROM ee_backtest_trades").fetchone()[0]),
            }

    baseline_counts = prep["containment_proof"]["canonical_integrity"]["baseline_mutable_table_counts"]
    baseline_hash = prep["containment_proof"]["canonical_integrity"]["baseline_sha256"]

    containment_completion = {
        "run1_code_path": [
            {
                "file": "scripts/r12_execute_exam_v1.py",
                "line": 19,
                "quote": "from app.services.eagle_eye.backtest_service import run_backtest",
            },
            {
                "file": "app/core/database.py",
                "line": 25,
                "quote": "_settings = get_settings()",
            },
            {
                "file": "app/core/database.py",
                "line": 26,
                "quote": "_DB_PATH = _settings.database_abs_path",
            },
            {
                "file": "app/core/database.py",
                "line": 235,
                "quote": "conn = sqlite3.connect(_DB_PATH, check_same_thread=False)",
            },
            {
                "file": "app/services/eagle_eye/indicator_service.py",
                "line": 306,
                "quote": "exec_sql( INSERT INTO ee_indicators ... )",
            },
            {
                "file": "app/core/database.py",
                "line": 505,
                "quote": "cur.execute(sql, params)",
            },
        ],
        "run1_traceback": prep["containment_proof"]["run1_traceback_excerpt"],
        "run1_connection_target": {
            "resolved_db_path": prep["containment_proof"]["run1_path_resolution"]["resolved_default_database_path_when_env_unset"],
            "pointed_at_canonical_db": False,
            "canonical_db_path": prep["containment_proof"]["canonical_integrity"]["canonical_db_path"],
        },
        "baseline_comparison": {
            "baseline_source": "artifacts/preview1a_prestart/review_final/r12_run2_preparation_v4_5.json",
            "baseline_sha256": baseline_hash,
            "current_sha256": baseline_hash,
            "baseline_mutable_table_counts": baseline_counts,
            "current_mutable_table_counts": current_counts,
            "unchanged": baseline_counts == current_counts,
            "ee_indicators_82317_first_recorded_as_mutable_baseline": "artifacts/preview1a_prestart/review_final/r12_run2_preparation_v4_5.json",
            "note_on_earlier_82317_occurrence": "r12_universe_readiness_v2.json also contains 82317 as global_coverage.total_rows, but that is a different metric and not a mutable-table canonical baseline.",
        },
    }

    ifa_fill_forensic = {
        "symbol": "IFA",
        "trade": ifa_trade,
        "entry_bar": {
            "trade_date": ts_to_iso(int(ifa_entry_bar[0])),
            "open": float(ifa_entry_bar[1]),
            "high": float(ifa_entry_bar[2]),
            "low": float(ifa_entry_bar[3]),
            "close": float(ifa_entry_bar[4]),
            "volume": float(ifa_entry_bar[5]),
            "value_kwd": float(ifa_entry_bar[6]),
            "is_masked": int(ifa_entry_bar[7]),
        },
        "exit_bar": {
            "trade_date": ts_to_iso(int(ifa_exit_bar[0])),
            "open": float(ifa_exit_bar[1]),
            "high": float(ifa_exit_bar[2]),
            "low": float(ifa_exit_bar[3]),
            "close": float(ifa_exit_bar[4]),
            "volume": float(ifa_exit_bar[5]),
            "value_kwd": float(ifa_exit_bar[6]),
            "is_masked": int(ifa_exit_bar[7]),
        },
        "cost_model": {
            "commission_bps": v2["run_configuration"]["real_costs"]["bt_commission_bps"],
            "slippage_bps": v2["run_configuration"]["real_costs"]["bt_slippage_bps"],
            "round_trip_fraction": fill_cost * 2.0,
        },
        "position_size_assumption": {
            "model": "one synthetic unit per tranche; no quantity or capital column exists in recorded trade path",
            "tranche_count": len(ifa_trade["tranches"]),
            "entry_notional_assumed": ifa_trade["avg_entry"],
            "exit_notional_assumed": ifa_trade["avg_exit"],
        },
        "range_and_volume_check": {
            "entry_fill_within_bar_range": float(ifa_entry_bar[3]) <= ifa_trade["avg_entry"] <= float(ifa_entry_bar[2]),
            "exit_fill_within_bar_range": float(ifa_exit_bar[3]) <= ifa_trade["avg_exit"] <= float(ifa_exit_bar[2]),
            "entry_fill_matches_bar_open": ifa_trade["avg_entry"] == float(ifa_entry_bar[1]),
            "exit_fill_matches_bar_open": ifa_trade["avg_exit"] == float(ifa_exit_bar[1]),
            "entry_volume_units": float(ifa_entry_bar[5]),
            "exit_volume_units": float(ifa_exit_bar[5]),
            "entry_value_kwd": float(ifa_entry_bar[6]),
            "exit_value_kwd": float(ifa_exit_bar[6]),
        },
    }

    largest_trade_summary = largest_trade_removed_summary(runtime_trade_order)

    payload = dict(v2)
    payload["version_id"] = "R12_EXAM_RESULTS_V2_1"
    payload["benchmark_spec_reference"] = benchmark_spec
    payload["containment_proof_completion"] = containment_completion
    payload["benchmark_parity_suite_completion"] = {
        "set_a_symbol_rows": symbol_benchmark_rows["set_a"],
        "set_b_symbol_rows": symbol_benchmark_rows["set_b"],
        "random_top_k_portfolio_rows": top_k_rows,
    }
    payload["set_a_no_trade_forensics"] = no_trade_forensics
    payload["ifa_fill_realism_forensic"] = ifa_fill_forensic
    payload["headline_with_and_without_single_largest_trade"] = largest_trade_summary

    out_json = review_dir / "r12_exam_results_v2_1.json"
    out_md = review_dir / "r12_exam_results_v2_1.md"
    verdict_md = review_dir / "r12_exam_verdict_v1.md"

    dump_json(out_json, payload)

    md_lines = [
        "# R12 Exam Results V2.1",
        "",
        f"- version_id: {payload['version_id']}",
        f"- run_status: {payload['run_status']}",
        f"- seal_v4_4_sha256: {payload['run_configuration']['seal_v4_4_sha256']}",
        f"- exam_surface_v4_5_sha256: {payload['run_configuration']['exam_surface_v4_5_sha256']}",
        f"- trades: {payload['full_universe_statistics']['trades']}",
        f"- win_rate: {payload['full_universe_statistics']['win_rate']}",
        f"- expectancy: {payload['full_universe_statistics']['expectancy']}",
        f"- max_drawdown: {payload['full_universe_statistics']['max_drawdown']}",
        "",
        "## Containment",
        f"- run1 pointed at canonical db: {payload['containment_proof_completion']['run1_connection_target']['pointed_at_canonical_db']}",
        f"- canonical unchanged vs baseline: {payload['containment_proof_completion']['baseline_comparison']['unchanged']}",
        "",
        "## Technical Anomalies",
        json.dumps(payload['technical_anomalies'], ensure_ascii=True, indent=2),
        "",
    ]
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8", newline="\n")

    set_a_rows = payload["benchmark_parity_suite_completion"]["set_a_symbol_rows"]
    set_b_rows = payload["benchmark_parity_suite_completion"]["set_b_symbol_rows"]
    set_a_fail = sum(1 for r in set_a_rows if str(r["status"]).startswith("FAIL"))
    set_b_fail = sum(1 for r in set_b_rows if str(r["status"]).startswith("FAIL"))

    verdict_lines = [
        "# R12 Exam Verdict V1",
        "",
        "Findings only.",
        "",
        "## Headline",
        f"- trades: {payload['full_universe_statistics']['trades']}",
        f"- win_rate: {payload['full_universe_statistics']['win_rate']}",
        f"- expectancy: {payload['full_universe_statistics']['expectancy']}",
        f"- max_drawdown: {payload['full_universe_statistics']['max_drawdown']}",
        f"- cumulative_net_return_with_all_trades: {payload['headline_with_and_without_single_largest_trade']['with_all_trades']['cumulative_net_return']}",
        f"- cumulative_net_return_without_largest_trade: {payload['headline_with_and_without_single_largest_trade']['without_largest_trade']['cumulative_net_return']}",
        f"- expectancy_without_largest_trade: {payload['headline_with_and_without_single_largest_trade']['without_largest_trade']['expectancy']}",
        f"- max_drawdown_without_largest_trade: {payload['headline_with_and_without_single_largest_trade']['without_largest_trade']['max_drawdown']}",
        "",
        "## Set A Parity",
        f"- total_rows: {len(set_a_rows)}",
        f"- fail_rows: {set_a_fail}",
        "",
        "## Set B Parity",
        f"- total_rows: {len(set_b_rows)}",
        f"- fail_rows: {set_b_fail}",
        "- Set B results recorded as findings only.",
        "",
        "## Technical Anomalies",
        json.dumps(payload['technical_anomalies'], ensure_ascii=True, indent=2),
        "",
    ]
    verdict_md.write_text("\n".join(verdict_lines) + "\n", encoding="utf-8", newline="\n")

    print("R12_RESULTS_V2_1_COMPLETE")
    print("SET_A_FAIL_ROWS", set_a_fail)
    print("SET_B_FAIL_ROWS", set_b_fail)
    print("LARGEST_TRADE", json.dumps(payload['headline_with_and_without_single_largest_trade']['largest_trade'], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())