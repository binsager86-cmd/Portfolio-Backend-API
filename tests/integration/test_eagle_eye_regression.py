from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.database import exec_sql, query_all, query_one, query_val
from app.services.eagle_eye.backtest_service import run_backtest
from app.services.eagle_eye.indicator_service import compute_and_store_symbol
from app.services.eagle_eye.market_data_service import DEFAULT_ENGINE_CONFIG, ensure_schema, load_ohlcv_csv, now_ts
from app.services.eagle_eye.audit_service import ensure_schema as ensure_audit_schema


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REAL_DATA = Path(__file__).resolve().parents[2] / "data" / "kse"
SYMBOLS = ["TIJARA", "BPCC", "ZAIN", "SANAM", "MABANEE"]
JOINER_SYMBOL = "JOINER"
ADVERSARIAL_SYMBOLS = ["CHOP", "FAKEOUT", "PUMP"]
SUITE_SYMBOLS = SYMBOLS + [JOINER_SYMBOL] + ADVERSARIAL_SYMBOLS
SYNTHETIC_OVERRIDES = {
    "min_daily_value_kwd": 100000.0,
}


def _reset_regression_tables_now() -> None:
    ensure_schema()
    ensure_audit_schema()
    for t in [
        "ee_ohlcv",
        "ee_indicators",
        "ee_symbol_state",
        "ee_signals",
        "ee_ratings",
        "ee_positions",
        "ee_backtest_runs",
        "ee_backtest_trades",
        "ee_change_status_history",
        "ee_change_requests",
        "ee_audit_events",
    ]:
        try:
            exec_sql(f"DELETE FROM {t}", ())
        except Exception:
            pass

    exec_sql("DELETE FROM ee_engine_config", ())
    ts = now_ts()
    for key, value in DEFAULT_ENGINE_CONFIG.items():
        exec_sql(
            """
            INSERT INTO ee_engine_config (key, value_json, updated_at, updated_by_user_id, change_request_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key, json.dumps(value, ensure_ascii=True), ts, 0, None),
        )


@pytest.fixture(scope="module", autouse=True)
def _reset_regression_tables():
    _reset_regression_tables_now()
    yield


@pytest.fixture(scope="module")
def _synthetic_suite() -> dict:
    mn, mx = _load_synthetic(SUITE_SYMBOLS)
    rep = run_backtest(SUITE_SYMBOLS, mn, mx, config_overrides=SYNTHETIC_OVERRIDES)
    signal_counts = query_all(
        """
        SELECT symbol, signal_type, COUNT(1) AS n
        FROM ee_signals
        GROUP BY symbol, signal_type
        ORDER BY symbol, signal_type
        """,
        (),
    )
    return {
        "mn": mn,
        "mx": mx,
        "report": rep,
        "signal_counts": {(str(r["symbol"]), str(r["signal_type"])): int(r["n"] or 0) for r in signal_counts},
    }


def _load_synthetic(symbols: list[str] | None = None) -> tuple[int, int]:
    symbols = symbols or SYMBOLS
    for s in symbols:
        load_ohlcv_csv(str(FIXTURES / f"synthetic_{s.lower()}.csv"), s)
        compute_and_store_symbol(s)
    row = query_one("SELECT MIN(trade_date) mn, MAX(trade_date) mx FROM ee_ohlcv", ())
    return int(row["mn"]), int(row["mx"])


def _run_fixture_backtest(mn: int, mx: int, symbols: list[str] | None = None):
    return run_backtest(symbols or SYMBOLS, mn, mx, config_overrides=SYNTHETIC_OVERRIDES)


def _signal_dates(symbol: str, signal_type: str) -> list[int]:
    rows = query_all(
        "SELECT trade_date FROM ee_signals WHERE symbol = ? AND signal_type = ? ORDER BY trade_date",
        (symbol, signal_type),
    )
    return [int(r["trade_date"]) for r in rows]


def _index_date(symbol: str, trade_date: int) -> int:
    rows = query_all(
        "SELECT trade_date FROM ee_ohlcv WHERE symbol = ? ORDER BY trade_date",
        (symbol,),
    )
    dates = [int(r["trade_date"]) for r in rows]
    return dates.index(trade_date)


def _trade(symbol: str):
    return query_one(
        "SELECT * FROM ee_backtest_trades WHERE symbol = ? ORDER BY id DESC LIMIT 1",
        (symbol,),
    )


def test_r1_tijara_regression_gate(_synthetic_suite):

    acc = _signal_dates("TIJARA", "ACCUMULATION_ALERT")
    brk = _signal_dates("TIJARA", "BREAKOUT_CONFIRMED")
    assert acc, "ACCUMULATION_ALERT missing for TIJARA"
    assert brk, "BREAKOUT_CONFIRMED missing for TIJARA"
    assert min(acc) < min(brk)

    t = _trade("TIJARA")
    assert t is not None
    assert float(t["net_return"] or 0.0) >= 0.40


def test_r2_bpcc_regression_gate(_synthetic_suite):

    brk = _signal_dates("BPCC", "BREAKOUT_CONFIRMED")
    assert brk

    t = _trade("BPCC")
    assert t is not None
    assert float(t["net_return"] or 0.0) >= 0.08

    avoid = _signal_dates("BPCC", "AVOID_SET")
    assert avoid, "Expected AVOID_SET during decline segment"

    early_cut = query_val(
        "SELECT trade_date FROM ee_ohlcv WHERE symbol = ? ORDER BY trade_date ASC LIMIT 1 OFFSET 80",
        ("BPCC",),
    )
    early_longs = query_all(
        """
        SELECT id FROM ee_signals
        WHERE symbol = ? AND trade_date <= ?
          AND signal_type IN ('ACCUMULATION_ALERT', 'BREAKOUT_CONFIRMED')
        """,
        ("BPCC", early_cut),
    )
    assert early_longs == []


def test_r3_zain_regression_gate(_synthetic_suite):

    brk_dates = _signal_dates("ZAIN", "BREAKOUT_CONFIRMED")
    assert brk_dates
    first_brk = brk_dates[0]

    pullbacks = query_all(
        """
        SELECT i.trade_date
        FROM ee_indicators i
        WHERE i.symbol = ? AND i.trade_date >= ?
          AND json_extract(i.payload_json, '$.low') <= json_extract(i.payload_json, '$.ema30')
          AND json_extract(i.payload_json, '$.close') >= json_extract(i.payload_json, '$.ema30')
        """,
        ("ZAIN", first_brk),
    )
    assert len(pullbacks) >= 2

    exits = _signal_dates("ZAIN", "EXIT")
    assert exits == []


def test_r4_sanam_regression_gate(_synthetic_suite):

    acc = _signal_dates("SANAM", "ACCUMULATION_ALERT")
    assert acc

    exit_rsi70 = query_all(
        """
        SELECT s.id
        FROM ee_signals s
        JOIN ee_indicators i ON i.symbol = s.symbol AND i.trade_date = s.trade_date
        WHERE s.symbol = 'SANAM'
          AND s.signal_type = 'EXIT'
          AND json_extract(i.payload_json, '$.rsi_14') > 70
        """,
        (),
    )
    assert exit_rsi70 == []

    t = _trade("SANAM")
    assert t is not None
    assert float(t["net_return"] or 0.0) >= 0.25


def test_r5_mabanee_regression_gate(_synthetic_suite):

    brk = _signal_dates("MABANEE", "BREAKOUT_CONFIRMED")
    assert brk, "R5a: BREAKOUT_CONFIRMED missing for MABANEE"

    first_brk = min(brk)
    pre_exit = query_all(
        """
        SELECT id
        FROM ee_signals
        WHERE symbol = 'MABANEE'
          AND trade_date >= ?
          AND signal_type IN ('EXIT', 'AVOID_SET')
        ORDER BY trade_date ASC
        LIMIT 1
        """,
        (first_brk,),
    )
    assert pre_exit, "R5b: expected EXIT/AVOID after breakout lifecycle"

    top_date = int(
        query_val(
            "SELECT trade_date FROM ee_ohlcv WHERE symbol = 'MABANEE' ORDER BY close DESC, trade_date ASC LIMIT 1",
            (),
        )
        or 0
    )
    assert top_date > 0

    warn = _signal_dates("MABANEE", "DISTRIBUTION_WARNING")
    assert warn
    assert (_index_date("MABANEE", warn[0]) - _index_date("MABANEE", top_date)) <= 15

    exit_dates = _signal_dates("MABANEE", "EXIT")
    assert exit_dates, "R5d: EXIT missing for MABANEE"

    rows = query_all(
        "SELECT trade_date, close FROM ee_ohlcv WHERE symbol = 'MABANEE' AND trade_date >= ? ORDER BY trade_date ASC",
        (first_brk,),
    )
    running_peak = 0.0
    ten_down_date = 0
    for r in rows:
        c = float(r["close"] or 0.0)
        running_peak = max(running_peak, c)
        if running_peak > 0 and c < (running_peak * 0.90):
            ten_down_date = int(r["trade_date"])
            break
    assert ten_down_date > 0
    assert min(exit_dates) <= ten_down_date

    t = _trade("MABANEE")
    assert t is not None
    assert float(t["net_return"] or 0.0) >= 0.18, "R5e: net return below +0.18"

    end_state = query_one("SELECT phase FROM ee_symbol_state WHERE symbol = 'MABANEE'", ())
    assert end_state and end_state["phase"] == "AVOID"

    reentries = query_all(
        """
        SELECT id
        FROM ee_signals
        WHERE symbol = 'MABANEE'
          AND trade_date > ?
          AND signal_type IN ('ACCUMULATION_ALERT', 'BREAKOUT_CONFIRMED')
        """,
        (min(exit_dates),),
    )
    assert reentries == [], "R5f: re-entry occurred after exit"


def test_r6_adversarial_rejection_gate(_synthetic_suite):
    near_misses = query_all(
        """
        SELECT symbol, trade_date, signal_type, evidence_json
        FROM ee_signals
        WHERE symbol IN ('CHOP', 'FAKEOUT', 'PUMP')
          AND signal_type IN ('PHASE_ONLY', 'DISTRIBUTION_WARNING', 'AVOID_SET')
        ORDER BY symbol, trade_date
        """,
        (),
    )
    _ = near_misses  # kept for external reporting; WATCH/revert activity is allowed.

    forbidden_signals = query_all(
        """
        SELECT symbol, signal_type, trade_date
        FROM ee_signals
        WHERE symbol IN ('CHOP', 'FAKEOUT', 'PUMP')
          AND signal_type IN ('ACCUMULATION_ALERT', 'BREAKOUT_CONFIRMED')
        """,
        (),
    )
    assert forbidden_signals == [], f"R6: adversarial fixtures emitted entry signals: {forbidden_signals}"

    opened = query_all(
        """
        SELECT symbol, id
        FROM ee_backtest_trades
        WHERE symbol IN ('CHOP', 'FAKEOUT', 'PUMP')
        """,
        (),
    )
    assert opened == [], f"R6: adversarial fixtures opened trades: {opened}"

    live_positions = query_all(
        """
        SELECT symbol, id
        FROM ee_positions
        WHERE symbol IN ('CHOP', 'FAKEOUT', 'PUMP')
        """,
        (),
    )
    assert live_positions == [], f"R6: adversarial fixtures left open positions: {live_positions}"


def test_r7_synthetic_joiner_trend_join_gate(_synthetic_suite):

    joined = query_all(
        """
        SELECT trade_date
        FROM ee_signals
        WHERE symbol = ?
          AND signal_type = 'PHASE_ONLY'
          AND phase_from = 'NEUTRAL'
          AND phase_to = 'MARKUP'
          AND COALESCE(json_extract(evidence_json, '$.joined_externally'), 0) = 1
        ORDER BY trade_date ASC
        LIMIT 1
        """,
        (JOINER_SYMBOL,),
    )
    assert joined, "R7: expected joined_externally trend join for synthetic_joiner"
    join_date = int(joined[0]["trade_date"])

    entries = query_all(
        """
        SELECT id
        FROM ee_signals
        WHERE symbol = ?
          AND signal_type IN ('ACCUMULATION_ALERT', 'BREAKOUT_CONFIRMED', 'ADD_ON_PULLBACK')
        """,
        (JOINER_SYMBOL,),
    )
    assert entries == [], "R7: trend-join path should not emit entry stack signals"

    warn = query_all(
        """
        SELECT trade_date
        FROM ee_signals
        WHERE symbol = ?
          AND signal_type = 'DISTRIBUTION_WARNING'
          AND trade_date > ?
        ORDER BY trade_date ASC
        LIMIT 1
        """,
        (JOINER_SYMBOL, join_date),
    )
    assert warn, "R7: expected DISTRIBUTION_WARNING after joined MARKUP"

    exit_row = query_all(
        """
        SELECT trade_date
        FROM ee_signals
        WHERE symbol = ?
          AND signal_type = 'EXIT'
          AND trade_date > ?
        ORDER BY trade_date ASC
        LIMIT 1
        """,
        (JOINER_SYMBOL, int(warn[0]["trade_date"])),
    )
    assert exit_row, "R7: expected EXIT after DISTRIBUTION_WARNING"

    avoid = query_all(
        """
        SELECT trade_date
        FROM ee_signals
        WHERE symbol = ?
          AND signal_type = 'AVOID_SET'
          AND trade_date > ?
        ORDER BY trade_date ASC
        LIMIT 1
        """,
        (JOINER_SYMBOL, int(exit_row[0]["trade_date"])),
    )
    assert avoid, "R7: expected AVOID_SET after EXIT"

    end_state = query_one("SELECT phase FROM ee_symbol_state WHERE symbol = ?", (JOINER_SYMBOL,))
    assert end_state and end_state["phase"] == "AVOID"

    src = (Path(__file__).resolve().parents[2] / "app" / "services" / "eagle_eye" / "scanner_service.py").read_text(encoding="utf-8")
    assert "MABANEE" not in src and "TIJARA" not in src and "BPCC" not in src


def test_real_data_statistical_gate_optional():
    _reset_regression_tables_now()
    missing = [s for s in SYMBOLS if not (REAL_DATA / f"{s}.csv").exists()]
    if missing:
        pytest.skip(f"Real KSE CSV data not found under data/kse: {', '.join(missing)}")

    for s in SYMBOLS:
        load_ohlcv_csv(str(REAL_DATA / f"{s}.csv"), s)
        compute_and_store_symbol(s)

    row = query_one("SELECT MIN(trade_date) mn, MAX(trade_date) mx FROM ee_ohlcv", ())
    rep = run_backtest(SYMBOLS, int(row["mn"]), int(row["mx"]))

    assert float(rep["expectancy"]) > 0

    breakout = query_one(
        """
        SELECT
            AVG(CASE WHEN outcome_label = 'WIN' THEN 1.0 ELSE 0.0 END) AS win_rate,
            AVG(CASE WHEN outcome_return > 0 THEN outcome_return END) AS avg_win,
            AVG(CASE WHEN outcome_return < 0 THEN outcome_return END) AS avg_loss
        FROM ee_signals
        WHERE signal_type = 'BREAKOUT_CONFIRMED' AND outcome_label IS NOT NULL
        """,
        (),
    )
    win_rate = float(breakout["win_rate"] or 0.0)
    avg_win = float(breakout["avg_win"] or 0.0)
    avg_loss = abs(float(breakout["avg_loss"] or 1.0))

    assert win_rate >= 0.45
    assert (avg_win / max(1e-9, avg_loss)) >= 1.8
