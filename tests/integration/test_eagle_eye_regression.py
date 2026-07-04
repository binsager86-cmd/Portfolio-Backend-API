from __future__ import annotations

from pathlib import Path

import pytest

from app.core.database import exec_sql, query_all, query_one, query_val
from app.services.eagle_eye.backtest_service import run_backtest
from app.services.eagle_eye.indicator_service import compute_and_store_symbol
from app.services.eagle_eye.market_data_service import ensure_schema, load_ohlcv_csv
from app.services.eagle_eye.audit_service import ensure_schema as ensure_audit_schema


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REAL_DATA = Path(__file__).resolve().parents[2] / "data" / "kse"
SYMBOLS = ["TIJARA", "BPCC", "ZAIN", "SANAM", "MABANEE"]
SYNTHETIC_OVERRIDES = {
    "base_max_width_pct": 0.25,
    "volume_breakout_mult": 1.2,
    "rsi_regime": 50,
    "adx_trigger": 15,
    "cmf_floor": 0.0,
    "accumulation_cmf_hits_min": 3,
    "accumulation_price_slope_max": 0.20,
    "accumulation_volume_slope_min": 0.02,
    "accumulation_min_score": 55,
    "breakout_min_score": 65,
    "min_daily_value_kwd": 1000.0,
}


@pytest.fixture(autouse=True)
def _reset_regression_tables():
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
    yield


def _load_synthetic() -> tuple[int, int]:
    for s in SYMBOLS:
        load_ohlcv_csv(str(FIXTURES / f"synthetic_{s.lower()}.csv"), s)
        compute_and_store_symbol(s)
    row = query_one("SELECT MIN(trade_date) mn, MAX(trade_date) mx FROM ee_ohlcv", ())
    return int(row["mn"]), int(row["mx"])


def _run_fixture_backtest(mn: int, mx: int):
    return run_backtest(SYMBOLS, mn, mx, config_overrides=SYNTHETIC_OVERRIDES)


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


def test_r1_tijara_regression_gate():
    mn, mx = _load_synthetic()
    _run_fixture_backtest(mn, mx)

    acc = _signal_dates("TIJARA", "ACCUMULATION_ALERT")
    brk = _signal_dates("TIJARA", "BREAKOUT_CONFIRMED")
    assert acc, "ACCUMULATION_ALERT missing for TIJARA"
    assert brk, "BREAKOUT_CONFIRMED missing for TIJARA"
    assert min(acc) < min(brk)

    t = _trade("TIJARA")
    assert t is not None
    assert float(t["net_return"] or 0.0) >= 0.40


def test_r2_bpcc_regression_gate():
    mn, mx = _load_synthetic()
    _run_fixture_backtest(mn, mx)

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


def test_r3_zain_regression_gate():
    mn, mx = _load_synthetic()
    _run_fixture_backtest(mn, mx)

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


def test_r4_sanam_regression_gate():
    mn, mx = _load_synthetic()
    _run_fixture_backtest(mn, mx)

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


def test_r5_mabanee_regression_gate():
    mn, mx = _load_synthetic()
    _run_fixture_backtest(mn, mx)

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

    longs_after_top = query_all(
        """
        SELECT id FROM ee_signals
        WHERE symbol = 'MABANEE'
          AND trade_date >= ?
          AND signal_type IN ('ACCUMULATION_ALERT', 'BREAKOUT_CONFIRMED')
        """,
        (top_date,),
    )
    assert longs_after_top == []

    exit_dates = _signal_dates("MABANEE", "EXIT")
    assert exit_dates
    peak = float(query_val("SELECT MAX(close) FROM ee_ohlcv WHERE symbol = 'MABANEE'", ()) or 0.0)
    ten_down_date = int(
        query_val(
            "SELECT trade_date FROM ee_ohlcv WHERE symbol = 'MABANEE' AND close <= ? ORDER BY trade_date ASC LIMIT 1",
            (peak * 0.90,),
        )
        or 0
    )
    assert ten_down_date > 0
    assert min(exit_dates) <= ten_down_date

    end_state = query_one("SELECT phase FROM ee_symbol_state WHERE symbol = 'MABANEE'", ())
    assert end_state and end_state["phase"] == "AVOID"


def test_real_data_statistical_gate_optional():
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
