"""Replay harness for app/services/eagle_eye_v2/trend_hold_engine.py.

This is the canonical validation tool for the trend-hold engine -- not a
one-off RC/R-series diagnostic. Re-run it whenever the engine's parameters
change. Reads real sealed OHLCV history straight from
eagle_eye_r11_clean_candidate.db (read-only) -- no shadow DBs, no synthetic
data.

Usage:
    python scripts/trend_hold_engine_replay.py ZAIN BPCC
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.eagle_eye_v2.trend_hold_engine import compute_daily_features, replay_symbol  # noqa: E402

DB_PATH = ROOT.parent / "eagle_eye_r11_clean_candidate.db"


def load_symbol(symbol: str) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT symbol, trade_date, open, high, low, close, volume, value_kwd "
            "FROM ee_ohlcv WHERE symbol = ? ORDER BY trade_date",
            conn,
            params=(symbol.upper(),),
        )
    finally:
        conn.close()
    df["trade_date"] = df["trade_date"].apply(
        lambda ts: datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    )
    return df


def summarize(symbol: str, rows: list[dict]) -> None:
    print(f"\n{'=' * 70}\n{symbol}  ({len(rows)} sessions)\n{'=' * 70}")
    trades: list[dict] = []
    open_trade: dict | None = None
    for r in rows:
        if r["decision"] == "BUY":
            open_trade = {"entry_date": r["trade_date"], "entry_price": r["close"]}
        elif r["decision"] == "SELL_SIGNAL" and open_trade is not None:
            ret_pct = (r["close"] / open_trade["entry_price"] - 1.0) * 100.0
            hold_days = (
                datetime.strptime(r["trade_date"], "%Y-%m-%d")
                - datetime.strptime(open_trade["entry_date"], "%Y-%m-%d")
            ).days
            trades.append(
                {
                    "entry_date": open_trade["entry_date"],
                    "entry_price": open_trade["entry_price"],
                    "exit_date": r["trade_date"],
                    "exit_price": r["close"],
                    "return_pct": ret_pct,
                    "hold_calendar_days": hold_days,
                    "exit_reason": r["reason"],
                }
            )
            open_trade = None

    if not trades and open_trade is None:
        print("No BUY signal fired across the whole replay window.")
    for t in trades:
        print(
            f"  BUY  {t['entry_date']} @ {t['entry_price']:.3f}  ->  "
            f"SELL {t['exit_date']} @ {t['exit_price']:.3f}   "
            f"return={t['return_pct']:+.1f}%   held={t['hold_calendar_days']}d"
        )
        print(f"       exit reason: {t['exit_reason']}")
    if open_trade is not None:
        last = rows[-1]
        ret_pct = (last["close"] / open_trade["entry_price"] - 1.0) * 100.0
        print(
            f"  BUY  {open_trade['entry_date']} @ {open_trade['entry_price']:.3f}  ->  "
            f"STILL HOLDING as of {last['trade_date']} @ {last['close']:.3f}   "
            f"unrealized={ret_pct:+.1f}%   stop={last['structural_stop']:.3f}"
        )

    hold_days_total = sum(1 for r in rows if r["decision"] == "HOLD")
    buy_days_total = sum(1 for r in rows if r["decision"] == "BUY")
    sell_days_total = sum(1 for r in rows if r["decision"] == "SELL_SIGNAL")
    print(f"\n  decision counts: BUY={buy_days_total}  HOLD={hold_days_total}  SELL_SIGNAL={sell_days_total}")


def main() -> None:
    symbols = sys.argv[1:] or ["ZAIN", "BPCC"]
    for symbol in symbols:
        raw = load_symbol(symbol)
        if raw.empty:
            print(f"{symbol}: no OHLCV rows found in {DB_PATH}")
            continue
        features = compute_daily_features(raw)
        rows = replay_symbol(features)
        summarize(symbol, rows)

        out_path = ROOT / "artifacts" / f"trend_hold_engine_{symbol.lower()}_replay.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  full daily log: {out_path}")


if __name__ == "__main__":
    main()
