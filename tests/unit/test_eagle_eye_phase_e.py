from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from app.core.database import exec_sql, query_all, query_one, query_val
from app.core.security import TokenData
from app.services.eagle_eye.backtest_service import run_backtest
from app.services.eagle_eye.indicator_service import compute_and_store_symbol
from app.services.eagle_eye.market_data_service import ConfigKeyMissing, DEFAULT_ENGINE_CONFIG, ensure_schema, get_active_config, load_ohlcv_csv
from app.services.eagle_eye.pipeline import process_bar
from app.services.eagle_eye.rating_service import compute_rating_from_indicator
from app.services.eagle_eye.scanner_service import assert_valid_transition, evaluate_symbol
from app.services.eagle_eye.scheduler_service import run_eod_pipeline
from app.services.eagle_eye.audit_service import ensure_schema as ensure_audit_schema
from app.services.eagle_eye.store import ensure_tables as ensure_legacy_tables


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(autouse=True)
def _reset_phase_e_tables():
    ensure_schema()
    ensure_audit_schema()
    ensure_legacy_tables()
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


def _load_fixture(symbol: str, low_liquidity: bool = False) -> None:
    path = FIXTURES / f"synthetic_{symbol.lower()}.csv"
    load_ohlcv_csv(str(path), symbol)
    # Mirror into legacy cache table so scheduler sync path can run idempotency tests.
    exec_sql(
        """
        INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at)
        SELECT symbol,
               strftime('%Y-%m-%d', trade_date, 'unixepoch') AS bar_date,
               open, high, low, close, volume, value_kwd, trade_date
        FROM ee_ohlcv
        WHERE symbol = ?
        ON CONFLICT(ticker, bar_date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            turnover_kwd = excluded.turnover_kwd,
            fetched_at = excluded.fetched_at
        """,
        (symbol,),
    )
    if low_liquidity:
        exec_sql("UPDATE ee_ohlcv SET value_kwd = 5000 WHERE symbol = ?", (symbol,))


def _bounds(symbols: list[str]) -> tuple[int, int]:
    q = ",".join(["?"] * len(symbols))
    row = query_one(
        f"SELECT MIN(trade_date) mn, MAX(trade_date) mx FROM ee_ohlcv WHERE symbol IN ({q})",
        tuple(symbols),
    )
    return int(row["mn"]), int(row["mx"])


def _seed_admin_user(test_client):
    exists = query_val("SELECT id FROM users WHERE username = ?", ("adminuser",))
    if not exists:
        pwd_hash = "$2b$12$drYtGzFmYlnMLLvZdo5nauyYZUN0slnBha1iCtgLqGghj/OfBHuwm"
        exec_sql(
            "INSERT INTO users (username, password_hash, name, is_admin, created_at) VALUES (?, ?, ?, ?, strftime('%s','now'))",
            ("adminuser", pwd_hash, "Admin User", 1),
        )
    resp = test_client.post("/api/v1/auth/login", json={"username": "adminuser", "password": "testpass123"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def _reset_driver_runtime_tables() -> None:
    for t in [
        "ee_symbol_state",
        "ee_signals",
        "ee_ratings",
        "ee_positions",
        "ee_backtest_runs",
        "ee_backtest_trades",
    ]:
        exec_sql(f"DELETE FROM {t}", ())


def _reset_driver_all_tables() -> None:
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
        exec_sql(f"DELETE FROM {t}", ())


def _load_driver_equivalence_fixtures(symbols: list[str]) -> tuple[int, int]:
    for symbol in symbols:
        _load_fixture(symbol)
        compute_and_store_symbol(symbol)
    return _bounds(symbols)


def test_u1_state_machine_transition_matrix():
    assert_valid_transition("NEUTRAL", "BASE_FORMING")
    assert_valid_transition("NEUTRAL", "MARKUP")
    assert_valid_transition("BASE_FORMING", "BREAKOUT_WATCH")
    assert_valid_transition("BREAKOUT_WATCH", "BREAKOUT_CONFIRMED")
    assert_valid_transition("MARKUP", "DISTRIBUTION_WARNING")
    assert_valid_transition("DISTRIBUTION_WARNING", "EXIT")

    with pytest.raises(ValueError):
        assert_valid_transition("NEUTRAL", "EXIT")

    with pytest.raises(ValueError):
        assert_valid_transition("AVOID", "MARKUP")


def test_u1b_trend_join_order_scope_and_avoid_precedence():
    symbol = "U1B"
    td = 1712000000
    # Coverage bars to satisfy trend_join_window checks.
    for i in range(50):
        t = td - (50 - i) * 86400
        exec_sql(
            "INSERT INTO ee_ohlcv (symbol, trade_date, open, high, low, close, volume, value_kwd, source, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', ?)",
            (symbol, t, 102.0, 104.0, 100.0, 103.0, 100000, 300000.0, t),
        )

    payload = {
        "trade_date": td,
        "open": 103.0,
        "high": 106.0,
        "low": 101.0,
        "close": 105.0,
        "volume": 120000,
        "value_kwd": 320000.0,
        "sma200": 95.0,
        "ema30": 101.0,
        "ema10": 104.0,
        "ema10_slope": 0.02,
        "sma200_slope": 0.01,
        "rsi_14": 58.0,
        "adx_19": 24.0,
        "plus_di": 30.0,
        "minus_di": 14.0,
        "macd_line": 0.8,
        "macd_signal": 0.4,
        "macd_hist": 0.4,
        "atr_14": 1.5,
        "cmf_10": 0.1,
        "rel_volume": 1.0,
        "range_high_60": 106.0,
        "range_low_60": 92.0,
        "range_high_120": 108.0,
        "range_low_120": 90.0,
        "range_width_pct": 0.14,
        "bb_width": 0.09,
        "atr_pct_percentile_252": 0.2,
        "price_slope_40": 0.0,
        "obv_slope_40": 0.2,
        "anv_slope_40": 0.2,
        "accumulation_divergence": True,
        "distribution_divergence": False,
    }
    exec_sql(
        "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
        (symbol, td, json.dumps(payload), "ee-2.1.2"),
    )

    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
    }

    # Trend-join must not fire from BASE_FORMING.
    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'BASE_FORMING', ?, 106.0, 92.0, ?, '{}')",
        (symbol, td - 86400, td),
    )
    res_base = evaluate_symbol(symbol, td, 80.0, cfg)
    assert res_base["phase"] != "MARKUP"

    # Trend-join must not fire from BREAKOUT_WATCH.
    exec_sql("DELETE FROM ee_symbol_state WHERE symbol = ?", (symbol,))
    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'BREAKOUT_WATCH', ?, 106.0, 92.0, ?, '{}')",
        (symbol, td - 86400, td),
    )
    res_watch = evaluate_symbol(symbol, td, 80.0, cfg)
    assert res_watch["phase"] != "MARKUP"

    # AVOID precedence should win before breakout/confirm checks.
    avoid_payload = dict(payload)
    avoid_payload.update({
        "open": 98.0,
        "close": 90.0,
        "low": 89.0,
        "high": 99.0,
        "sma200": 100.0,
        "sma200_slope": -0.02,
        "ema10": 94.0,
        "ema30": 95.0,
        "ema10_slope": 0.03,
        "range_high_60": 102.0,
        "range_low_60": 88.0,
        "range_high_120": 104.0,
        "range_low_120": 86.0,
        "rel_volume": 1.0,
        "accumulation_divergence": False,
        "distribution_divergence": False,
    })
    symbol = "U1I"
    base_td = 1712500000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "avoid_reclaim_clear_closes": 2,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
    }

    for i in range(220):
        td = base_td + i * 86400
        close = 160.0 - (i * 0.45)
        if i in {140, 155, 170}:
            close = 132.0 - (i * 0.02)
        payload = {
            "trade_date": td,
            "open": close + 0.4,
            "high": close + 1.0,
            "low": close - 1.2,
            "close": close,
            "volume": 110000,
            "value_kwd": 220000.0,
            "sma200": 150.0,
            "ema30": 148.0,
            "ema10": 147.0,
            "ema10_slope": -0.02,
            "sma200_slope": -0.03,
            "rsi_14": 42.0,
            "adx_19": 19.0,
            "plus_di": 18.0,
            "minus_di": 24.0,
            "macd_line": -0.2,
            "macd_signal": -0.1,
            "macd_hist": -0.1,
            "atr_14": 1.4,
            "cmf_10": -0.05,
            "rel_volume": 1.0,
            "range_high_60": close + 2.0,
            "range_low_60": close - 4.0,
            "range_high_120": close + 2.0,
            "range_low_120": close - 6.0,
            "range_width_pct": 0.08,
            "bb_width": 0.07,
            "atr_pct_percentile_252": 0.15,
            "price_slope_40": -0.03,
            "obv_slope_40": -0.2,
            "anv_slope_40": -0.2,
            "accumulation_divergence": False,
            "distribution_divergence": False,
        }
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(payload), "ee-2.1.2"),
        )
        exec_sql(
            "INSERT INTO ee_ohlcv (symbol, trade_date, open, high, low, close, volume, value_kwd, source, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume, value_kwd=excluded.value_kwd",
            (symbol, td, close + 0.4, close + 1.0, close - 1.2, close, 110000, 220000.0, td),
        )

    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'AVOID', ?, 125.0, 118.0, ?, '{}')",
        (symbol, base_td, base_td),
    )

    td = base_td + 219 * 86400
    res = evaluate_symbol(symbol, td, 68.0, cfg)
    assert res["phase"] == "AVOID"
    st = query_one("SELECT phase, state_json FROM ee_symbol_state WHERE symbol = ?", (symbol,))
    assert st is not None
    sj = json.loads(st["state_json"] or "{}")
    assert int(sj.get("avoid_clear_streak") or 0) == 0


def test_u1j_avoid_clears_on_two_consecutive_reclaim_closes_and_base_remains_reachable():
    symbol = "U1J"
    base_td = 1712600000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "avoid_reclaim_clear_closes": 2,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
    }

    for i in range(221):
        td = base_td + i * 86400
        close = 160.0 - (i * 0.55)
        if i in {218, 219}:
            close = 151.0 + ((i - 218) * 0.4)
        payload = {
            "trade_date": td,
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 110000,
            "value_kwd": 220000.0,
            "sma200": 150.0,
            "ema30": 148.0,
            "ema10": 147.0,
            "ema10_slope": -0.02,
            "sma200_slope": -0.03,
            "rsi_14": 45.0,
            "adx_19": 19.0,
            "plus_di": 18.0,
            "minus_di": 24.0,
            "macd_line": -0.2,
            "macd_signal": -0.1,
            "macd_hist": -0.1,
            "atr_14": 1.4,
            "cmf_10": -0.05,
            "rel_volume": 1.0,
            "range_high_60": close + 2.0,
            "range_low_60": close - 4.0,
            "range_high_120": close + 2.0,
            "range_low_120": close - 6.0,
            "range_width_pct": 0.08,
            "bb_width": 0.07,
            "atr_pct_percentile_252": 0.15,
            "price_slope_40": -0.03,
            "obv_slope_40": -0.2,
            "anv_slope_40": -0.2,
            "accumulation_divergence": False,
            "distribution_divergence": False,
        }
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(payload), "ee-2.1.2"),
        )
        exec_sql(
            "INSERT INTO ee_ohlcv (symbol, trade_date, open, high, low, close, volume, value_kwd, source, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume, value_kwd=excluded.value_kwd",
            (symbol, td, close - 0.1, close + 1.0, close - 1.0, close, 110000, 220000.0, td),
        )

    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'AVOID', ?, 125.0, 118.0, ?, '{}')",
        (symbol, base_td, base_td),
    )

    res_1 = evaluate_symbol(symbol, base_td + 218 * 86400, 70.0, cfg)
    res_2 = evaluate_symbol(symbol, base_td + 219 * 86400, 70.0, cfg)

    st = query_one("SELECT phase, base_high, base_low, state_json FROM ee_symbol_state WHERE symbol = ?", (symbol,))
    assert st is not None
    assert st["phase"] == "BASE_FORMING"
    assert st["base_high"] == 125.0
    sj = json.loads(st["state_json"] or "{}")
    assert int(sj.get("avoid_reclaim_streak") or 0) == 0
    assert any(e.get("action") == "avoid_cleared_resume" for e in (sj.get("phase_lifecycle_log") or []))


def test_u1i_avoid_persists_through_monotonic_decline_even_with_single_sma200_touches():
    symbol = "U1I"
    base_td = 1712500000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "avoid_reclaim_clear_closes": 2,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
    }

    for i in range(220):
        td = base_td + i * 86400
        close = 160.0 - (i * 0.45)
        if i in {140, 155, 170}:
            close = 132.0 - (i * 0.02)
        payload = {
            "trade_date": td,
            "open": close + 0.4,
            "high": close + 1.0,
            "low": close - 1.2,
            "close": close,
            "volume": 110000,
            "value_kwd": 220000.0,
            "sma200": 150.0,
            "ema30": 148.0,
            "ema10": 147.0,
            "ema10_slope": -0.02,
            "sma200_slope": -0.03,
            "rsi_14": 42.0,
            "adx_19": 19.0,
            "plus_di": 18.0,
            "minus_di": 24.0,
            "macd_line": -0.2,
            "macd_signal": -0.1,
            "macd_hist": -0.1,
            "atr_14": 1.4,
            "cmf_10": -0.05,
            "rel_volume": 1.0,
            "range_high_60": close + 2.0,
            "range_low_60": close - 4.0,
            "range_high_120": close + 2.0,
            "range_low_120": close - 6.0,
            "range_width_pct": 0.08,
            "bb_width": 0.07,
            "atr_pct_percentile_252": 0.15,
            "price_slope_40": -0.03,
            "obv_slope_40": -0.2,
            "anv_slope_40": -0.2,
            "accumulation_divergence": False,
            "distribution_divergence": False,
        }
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(payload), "ee-2.1.2"),
        )
        exec_sql(
            "INSERT INTO ee_ohlcv (symbol, trade_date, open, high, low, close, volume, value_kwd, source, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume, value_kwd=excluded.value_kwd",
            (symbol, td, close + 0.4, close + 1.0, close - 1.2, close, 110000, 220000.0, td),
        )

    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'AVOID', ?, 125.0, 118.0, ?, ?)",
        (symbol, base_td, base_td, json.dumps({"pre_avoid_phase": "BASE_FORMING", "pre_avoid_base_high": 125.0, "pre_avoid_base_low": 118.0})),
    )

    r = evaluate_symbol(symbol, base_td + 219 * 86400, 68.0, cfg)
    assert r["phase"] == "AVOID"
    st = query_one("SELECT phase, state_json FROM ee_symbol_state WHERE symbol = ?", (symbol,))
    assert st is not None
    sj = json.loads(st["state_json"] or "{}")
    assert int(sj.get("avoid_clear_streak") or 0) == 0 or int(sj.get("avoid_clear_streak") or 0) == 1


def test_u1l_avoid_lands_neutral_when_base_invalidated_during_avoid():
    symbol = "U1L"
    base_td = 1712800000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "base_drift_invalidate_sessions": 3,
        "avoid_reclaim_clear_closes": 2,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
    }

    for i in range(12):
        td = base_td + i * 86400
        if i < 8:
            close = 124.0 - (i * 0.2)
        elif i in {8, 9}:
            close = 115.0 - ((i - 8) * 0.6)
        else:
            close = 125.5 + ((i - 10) * 0.2)
        payload = {
            "trade_date": td,
            "open": 124.5,
            "high": 125.0,
            "low": 123.0,
            "close": close,
            "volume": 110000,
            "value_kwd": 220000.0,
            "sma200": 125.0,
            "ema30": 124.0,
            "ema10": 123.5,
            "ema10_slope": -0.02,
            "sma200_slope": -0.02,
            "rsi_14": 41.0,
            "adx_19": 19.0,
            "plus_di": 18.0,
            "minus_di": 24.0,
            "macd_line": -0.2,
            "macd_signal": -0.1,
            "macd_hist": -0.1,
            "atr_14": 1.4,
            "cmf_10": -0.05,
            "rel_volume": 1.0,
            "range_high_60": 126.0,
            "range_low_60": 118.0,
            "range_high_120": 127.0,
            "range_low_120": 117.0,
            "range_width_pct": 0.08,
            "bb_width": 0.07,
            "atr_pct_percentile_252": 0.15,
            "price_slope_40": -0.03,
            "obv_slope_40": -0.2,
            "anv_slope_40": -0.2,
            "accumulation_divergence": False,
            "distribution_divergence": False,
        }
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(payload), "ee-2.1.2"),
        )
        exec_sql(
            "INSERT INTO ee_ohlcv (symbol, trade_date, open, high, low, close, volume, value_kwd, source, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume, value_kwd=excluded.value_kwd",
            (symbol, td, 124.5, 125.0, 123.0, close, 110000, 220000.0, td),
        )

    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'AVOID', ?, 125.0, 118.0, ?, ?)",
        (symbol, base_td, base_td, json.dumps({"pre_avoid_phase": "BASE_FORMING", "pre_avoid_base_high": 125.0, "pre_avoid_base_low": 118.0})),
    )

    r1 = evaluate_symbol(symbol, base_td + 9 * 86400, 68.0, cfg)
    r_mid = evaluate_symbol(symbol, base_td + 10 * 86400, 68.0, cfg)
    r2 = evaluate_symbol(symbol, base_td + 11 * 86400, 68.0, cfg)
    assert r1["phase"] == "AVOID"
    assert r_mid["phase"] == "AVOID"
    assert r2["phase"] == "NEUTRAL"
    st = query_one("SELECT phase, base_high, base_low, state_json FROM ee_symbol_state WHERE symbol = ?", (symbol,))
    assert st is not None
    assert st["phase"] == "NEUTRAL"
    assert st["base_high"] is None and st["base_low"] is None


def test_u1k_reversal_entry_after_avoid_clear():
    symbol = "U1K"
    base_td = 1712700000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "avoid_reclaim_clear_closes": 2,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
    }

    for i in range(223):
        td = base_td + i * 86400
        if i < 150:
            close = 160.0 - (i * 0.55)
        elif i < 180:
            close = 98.0 + ((i - 150) * 0.25)
        elif i < 220:
            close = 126.0 + ((i - 180) * 0.8)
        else:
            close = 158.0 + ((i - 220) * 0.6)
        payload = {
            "trade_date": td,
            "open": close - 0.2,
            "high": close + 1.1,
            "low": close - 1.1,
            "close": close,
            "volume": 120000,
            "value_kwd": 260000.0,
            "sma200": 150.0 if i < 140 else 124.0,
            "ema30": 148.0,
            "ema10": 147.0,
            "ema10_slope": 0.02,
            "sma200_slope": -0.02 if i < 40 else 0.02,
            "rsi_14": 50.0 + min(10, i * 0.1),
            "adx_19": 24.0,
            "plus_di": 30.0,
            "minus_di": 18.0,
            "macd_line": 0.3,
            "macd_signal": 0.2,
            "macd_hist": 0.1,
            "atr_14": 1.5,
            "cmf_10": 0.08,
            "rel_volume": 2.8 if i >= 220 else 1.0,
            "range_high_60": close + 2.0,
            "range_low_60": close - 4.0,
            "range_high_120": close + 2.0,
            "range_low_120": close - 6.0,
            "range_width_pct": 0.08,
            "bb_width": 0.07,
            "atr_pct_percentile_252": 0.15,
            "price_slope_40": 0.01,
            "obv_slope_40": 0.2,
            "anv_slope_40": 0.2,
            "accumulation_divergence": True,
            "distribution_divergence": False,
        }
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(payload), "ee-2.1.2"),
        )
        exec_sql(
            "INSERT INTO ee_ohlcv (symbol, trade_date, open, high, low, close, volume, value_kwd, source, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume, value_kwd=excluded.value_kwd",
            (symbol, td, close - 0.2, close + 1.1, close - 1.1, close, 120000, 260000.0, td),
        )

    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'AVOID', ?, 125.0, 118.0, ?, ?)",
        (symbol, base_td, base_td, json.dumps({"pre_avoid_phase": "BASE_FORMING", "pre_avoid_base_high": 125.0, "pre_avoid_base_low": 118.0})),
    )

    p_watch = {
        "trade_date": base_td + 220 * 86400,
        "open": 125.9,
        "high": 127.1,
        "low": 125.2,
        "close": 126.4,
        "volume": 120000,
        "value_kwd": 260000.0,
        "sma200": 124.0,
        "ema30": 124.0,
        "ema10": 125.0,
        "ema10_slope": 0.04,
        "sma200_slope": 0.01,
        "rsi_14": 56.0,
        "adx_19": 24.5,
        "plus_di": 30.0,
        "minus_di": 18.0,
        "macd_line": 0.35,
        "macd_signal": 0.20,
        "macd_hist": 0.15,
        "atr_14": 1.5,
        "cmf_10": 0.09,
        "rel_volume": 2.9,
        "range_high_60": 127.0,
        "range_low_60": 120.0,
        "range_high_120": 128.0,
        "range_low_120": 118.0,
        "range_width_pct": 0.08,
        "bb_width": 0.07,
        "atr_pct_percentile_252": 0.15,
        "price_slope_40": 0.01,
        "obv_slope_40": 0.2,
        "anv_slope_40": 0.2,
        "accumulation_divergence": True,
        "distribution_divergence": False,
    }
    p_confirm1 = dict(p_watch)
    p_confirm1.update({"trade_date": base_td + 221 * 86400, "open": 126.5, "close": 127.1, "high": 127.6, "low": 126.1, "rsi_14": 57.0, "adx_19": 25.0, "macd_hist": 0.18})
    p_confirm2 = dict(p_watch)
    p_confirm2.update({"trade_date": base_td + 222 * 86400, "open": 127.0, "close": 127.8, "high": 128.3, "low": 126.6, "rsi_14": 58.0, "adx_19": 26.0, "macd_hist": 0.22})
    for p in [p_watch, p_confirm1, p_confirm2]:
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, p["trade_date"], json.dumps(p), "ee-2.1.2"),
        )

    for td, rv, close in [
        (base_td + 218 * 86400, 1.7, 125.2),
        (base_td + 219 * 86400, 2.9, 125.8),
    ]:
        payload = {
            "trade_date": td,
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 120000,
            "value_kwd": 260000.0,
            "sma200": 124.0,
            "ema30": 124.0,
            "ema10": 125.0,
            "ema10_slope": 0.03,
            "sma200_slope": 0.01,
            "rsi_14": 55.0,
            "adx_19": 24.0,
            "plus_di": 30.0,
            "minus_di": 18.0,
            "macd_line": 0.30,
            "macd_signal": 0.20,
            "macd_hist": 0.10,
            "atr_14": 1.5,
            "cmf_10": 0.08,
            "rel_volume": rv,
            "range_high_60": 127.0,
            "range_low_60": 120.0,
            "range_high_120": 128.0,
            "range_low_120": 118.0,
            "range_width_pct": 0.08,
            "bb_width": 0.07,
            "atr_pct_percentile_252": 0.15,
            "price_slope_40": 0.01,
            "obv_slope_40": 0.2,
            "anv_slope_40": 0.2,
            "accumulation_divergence": True,
            "distribution_divergence": False,
        }
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(payload), "ee-2.1.2"),
        )

    r_clear = evaluate_symbol(symbol, base_td + 218 * 86400, 78.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    assert r_clear["phase"] == "AVOID"
    assert r_clear["transition"] is None

    r_resume = evaluate_symbol(symbol, base_td + 219 * 86400, 78.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    assert r_resume["phase"] == "BREAKOUT_WATCH"
    assert r_resume["transition"] == ("ACCUMULATION", "BREAKOUT_WATCH")

    r_watch = evaluate_symbol(symbol, base_td + 220 * 86400, 78.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    assert r_watch["phase"] == "BREAKOUT_CONFIRMED"
    assert r_watch["signal_type"] == "BREAKOUT_CONFIRMED"

    r_confirm1 = evaluate_symbol(symbol, base_td + 221 * 86400, 78.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    assert r_confirm1["phase"] == "BREAKOUT_CONFIRMED"

    r_confirm2 = evaluate_symbol(symbol, base_td + 222 * 86400, 78.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    assert r_confirm2["phase"] == "BREAKOUT_CONFIRMED"

    st = query_one("SELECT phase, base_high, base_low, state_json FROM ee_symbol_state WHERE symbol = ?", (symbol,))
    assert st is not None
    assert st["phase"] == "BREAKOUT_CONFIRMED"
    sj = json.loads(st["state_json"] or "{}")
    log = sj.get("phase_lifecycle_log") if isinstance(sj.get("phase_lifecycle_log"), list) else []
    assert any(e.get("action") == "avoid_cleared_resume" for e in log)
    assert any((e.get("old") or {}).get("phase") == "AVOID" and (e.get("new") or {}).get("phase") == "BASE_FORMING" for e in log)


def test_u1d_join_preempts_base_when_both_true_inside_window():
    symbol = "U1D"
    td = 1712600000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
    }

    payload = {
        "trade_date": td,
        "open": 119.5,
        "high": 122.0,
        "low": 118.0,
        "close": 120.0,
        "volume": 120000,
        "value_kwd": 300000.0,
        "sma200": 100.0,
        "ema30": 115.0,
        "ema10": 121.0,
        "ema10_slope": 0.02,
        "sma200_slope": 0.01,
        "rsi_14": 58.0,
        "adx_19": 24.0,
        "plus_di": 30.0,
        "minus_di": 14.0,
        "macd_line": 0.8,
        "macd_signal": 0.4,
        "macd_hist": 0.4,
        "atr_14": 1.5,
        "cmf_10": 0.1,
        "rel_volume": 1.0,
        "range_high_60": 126.0,
        "range_low_60": 114.0,
        "range_high_120": 128.0,
        "range_low_120": 100.0,
        "range_width_pct": 0.10,
        "bb_width": 0.09,
        "atr_pct_percentile_252": 0.2,
        "price_slope_40": 0.0,
        "obv_slope_40": 0.2,
        "anv_slope_40": 0.2,
        "accumulation_divergence": True,
        "distribution_divergence": False,
    }

    history = []
    for i in range(70):
        row = dict(payload)
        row["trade_date"] = td - (69 - i) * 86400
        history.append(row)

    state = {
        "symbol": symbol,
        "phase": "NEUTRAL",
        "phase_since": td,
        "base_high": None,
        "base_low": None,
        "base_start": None,
        "last_score": None,
        "avoid_until": None,
        "updated_at": td,
        "state_json": {},
    }

    result = evaluate_symbol(
        symbol,
        td,
        82.0,
        cfg,
        indicator_payload=payload,
        indicator_history=history,
        state_override=state,
        persist_state=False,
        liquidity_snapshot=(True, {"source": "unit"}),
        coverage_start_date=td - 15 * 86400,
        coverage_sessions=15,
    )

    assert result["phase"] == "MARKUP"
    assert result["transition"] == ("NEUTRAL", "MARKUP")
    assert bool(result["state"]["state_json"].get("joined_externally")) is True


def test_u1e_no_phase_classification_before_warmup_ready():
    symbol = "U1E"
    base_td = 1715000000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
        "exit_cooldown_sessions": 10,
    }

    history: list[dict] = []
    for i in range(250):
        td = base_td + i * 86400
        row = {
            "trade_date": td,
            "open": 99.5,
            "high": 101.0,
            "low": 98.8,
            "close": 100.0 + (i * 0.05),
            "volume": 120000,
            "value_kwd": 300000.0,
            "ema10": 101.0,
            "ema30": 100.0,
            "sma200": 0.0 if i < 199 else 95.0,
            "sma200_slope": 0.01,
            "rsi_14": 58.0,
            "adx_19": 24.0,
            "plus_di": 30.0,
            "minus_di": 14.0,
            "macd_line": 0.8,
            "macd_signal": 0.4,
            "macd_hist": 0.4,
            "atr_14": 1.5,
            "cmf_10": 0.12,
            "rel_volume": 1.0,
            "range_high_60": 106.0,
            "range_low_60": 92.0,
            "range_high_120": 0.0 if i < 199 else 110.0,
            "range_low_120": 0.0 if i < 199 else 90.0,
            "range_width_pct": 0.10,
            "bb_width": 0.09,
            "atr_pct_percentile_252": 0.2,
            "price_slope_40": 0.0,
            "obv_slope_40": 0.2,
            "anv_slope_40": 0.2,
            "accumulation_divergence": True,
            "distribution_divergence": False,
        }
        history.append(row)

    pre_idx = 150
    pre = evaluate_symbol(
        symbol,
        history[pre_idx]["trade_date"],
        82.0,
        cfg,
        indicator_payload=history[pre_idx],
        indicator_history=history[: pre_idx + 1],
        state_override={
            "symbol": symbol,
            "phase": "NEUTRAL",
            "phase_since": history[pre_idx]["trade_date"],
            "base_high": None,
            "base_low": None,
            "base_start": None,
            "last_score": None,
            "avoid_until": None,
            "updated_at": history[pre_idx]["trade_date"],
            "state_json": {},
        },
        persist_state=False,
        liquidity_snapshot=(True, {"source": "unit"}),
    )
    assert pre["phase"] == "NEUTRAL"
    assert pre["transition"] is None
    assert pre["reason"] == "warmup_pending"

    warm_idx = 199
    warm = evaluate_symbol(
        symbol,
        history[warm_idx]["trade_date"],
        82.0,
        cfg,
        indicator_payload=history[warm_idx],
        indicator_history=history[: warm_idx + 1],
        state_override={
            "symbol": symbol,
            "phase": "NEUTRAL",
            "phase_since": history[warm_idx]["trade_date"],
            "base_high": None,
            "base_low": None,
            "base_start": None,
            "last_score": None,
            "avoid_until": None,
            "updated_at": history[warm_idx]["trade_date"],
            "state_json": {},
        },
        persist_state=False,
        liquidity_snapshot=(True, {"source": "unit"}),
    )
    assert warm["phase"] == "MARKUP"
    assert warm["transition"] == ("NEUTRAL", "MARKUP")


def test_u1f_exit_rearms_to_neutral_after_cooldown():
    symbol = "U1F"
    base_td = 1716000000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
        "exit_cooldown_sessions": 10,
    }

    history: list[dict] = []
    for i in range(30):
        td = base_td + i * 86400
        history.append(
            {
                "trade_date": td,
                "open": 101.0,
                "high": 103.0,
                "low": 100.0,
                "close": 102.0,
                "volume": 100000,
                "value_kwd": 250000.0,
                "sma200": 95.0,
                "ema30": 100.0,
                "ema10": 101.0,
                "ema10_slope": 0.01,
                "sma200_slope": 0.01,
                "rsi_14": 52.0,
                "adx_19": 22.0,
                "plus_di": 25.0,
                "minus_di": 18.0,
                "macd_line": 0.2,
                "macd_signal": 0.1,
                "macd_hist": 0.1,
                "atr_14": 1.4,
                "cmf_10": 0.07,
                "rel_volume": 1.1,
                "range_high_60": 106.0,
                "range_low_60": 92.0,
                "range_high_120": 108.0,
                "range_low_120": 90.0,
                "range_width_pct": 0.12,
                "bb_width": 0.09,
                "atr_pct_percentile_252": 0.2,
                "price_slope_40": 0.0,
                "obv_slope_40": 0.1,
                "anv_slope_40": 0.1,
                "accumulation_divergence": False,
                "distribution_divergence": False,
            }
        )

    idx = 15
    result = evaluate_symbol(
        symbol,
        history[idx]["trade_date"],
        72.0,
        cfg,
        indicator_payload=history[idx],
        indicator_history=history[: idx + 1],
        state_override={
            "symbol": symbol,
            "phase": "EXIT",
            "phase_since": history[0]["trade_date"],
            "base_high": None,
            "base_low": None,
            "base_start": None,
            "last_score": 70.0,
            "avoid_until": None,
            "updated_at": history[idx]["trade_date"],
            "state_json": {},
        },
        persist_state=False,
        liquidity_snapshot=(True, {"source": "unit"}),
    )

    assert result["phase"] == "NEUTRAL"
    assert result["transition"] == ("EXIT", "NEUTRAL")
    assert result["reason"] == "exit_cooldown_rearm"


def test_u1c_confirming_window_confirms_or_reverts():
    symbol = "U1C"
    base_td = 1713000000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
    }

    for i in range(70):
        td = base_td + i * 86400
        exec_sql(
            "INSERT INTO ee_ohlcv (symbol, trade_date, open, high, low, close, volume, value_kwd, source, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', ?)",
            (symbol, td, 100.0, 103.0, 98.0, 101.0, 100000, 250000.0, td),
        )

    def put_indicator(td: int, p: dict):
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(p), "ee-2.1.2"),
        )

    # Seed BREAKOUT_WATCH and run 3-bar confirming window: confirm on t+2.
    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'BREAKOUT_WATCH', ?, 100.0, 90.0, ?, '{}')",
        (symbol, base_td, base_td),
    )

    p0 = {
        "open": 101.0, "high": 105.0, "low": 99.0, "close": 102.0,
        "volume": 180000.0, "value_kwd": 300000.0,
        "sma200": 95.0, "ema30": 99.0, "ema10": 101.0, "ema10_slope": 0.01, "sma200_slope": 0.01,
        "rsi_14": 54.0, "adx_19": 23.0, "plus_di": 28.0, "minus_di": 20.0,
        "macd_line": 0.2, "macd_signal": 0.25, "macd_hist": -0.05,
        "atr_14": 1.5, "cmf_10": 0.1, "rel_volume": 2.8,
        "range_high_60": 100.0, "range_low_60": 90.0,
        "range_high_120": 101.0, "range_low_120": 88.0,
        "range_width_pct": 0.1, "bb_width": 0.08,
        "atr_pct_percentile_252": 0.2,
        "price_slope_40": 0.0, "obv_slope_40": 0.2, "anv_slope_40": 0.2,
        "accumulation_divergence": True, "distribution_divergence": False,
    }
    p1 = dict(p0)
    p1.update({"open": 101.8, "close": 102.4, "high": 105.4, "low": 100.0, "rsi_14": 54.5, "adx_19": 23.2, "macd_hist": -0.03})
    p2 = dict(p0)
    p2.update({"open": 102.2, "close": 103.5, "high": 104.0, "low": 100.5, "rsi_14": 57.0, "adx_19": 24.0, "macd_hist": 0.1})

    put_indicator(base_td, p0)
    put_indicator(base_td + 86400, p1)
    put_indicator(base_td + 2 * 86400, p2)

    evaluate_symbol(symbol, base_td, 80.0, cfg)
    evaluate_symbol(symbol, base_td + 86400, 80.0, cfg)
    r_confirm = evaluate_symbol(symbol, base_td + 2 * 86400, 80.0, cfg)
    assert r_confirm["signal_type"] == "BREAKOUT_CONFIRMED"

    # New symbol: confirming window never reaches 4/6 and reverts to watch.
    symbol2 = "U1C2"
    for i in range(70):
        td = base_td + i * 86400
        exec_sql(
            "INSERT INTO ee_ohlcv (symbol, trade_date, open, high, low, close, volume, value_kwd, source, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', ?)",
            (symbol2, td, 100.0, 103.0, 98.0, 101.0, 100000, 250000.0, td),
        )
    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'BREAKOUT_WATCH', ?, 100.0, 90.0, ?, '{}')",
        (symbol2, base_td, base_td),
    )
    q0 = dict(p0)
    q0.update({"macd_hist": -0.2, "rsi_14": 53.0})
    q1 = dict(q0)
    q2 = dict(q0)
    for j, q in enumerate([q0, q1, q2]):
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol2, base_td + j * 86400, json.dumps(q), "ee-2.1.2"),
        )
    evaluate_symbol(symbol2, base_td, 80.0, cfg)
    evaluate_symbol(symbol2, base_td + 86400, 80.0, cfg)
    r_revert = evaluate_symbol(symbol2, base_td + 2 * 86400, 80.0, cfg)
    assert r_revert["phase"] == "BREAKOUT_WATCH"
    assert r_revert["signal_type"] is None


def test_u1d_breakout_uses_frozen_base_high_not_rolling_120_high():
    symbol = "U1D"
    base_td = 1715000000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
    }

    for i in range(80):
        td = base_td + i * 86400
        exec_sql(
            "INSERT INTO ee_ohlcv (symbol, trade_date, open, high, low, close, volume, value_kwd, source, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', ?)",
            (symbol, td, 100.0, 106.0, 96.0, 100.0, 120000, 300000.0, td),
        )

    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'BASE_FORMING', ?, 100.0, 90.0, ?, '{}')",
        (symbol, base_td, base_td),
    )

    def put_indicator(td: int, payload: dict) -> None:
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(payload), "ee-2.1.2"),
        )

    base_payload = {
        "open": 99.5,
        "high": 100.8,
        "low": 98.8,
        "close": 99.2,
        "volume": 160000.0,
        "value_kwd": 250000.0,
        "sma200": 95.0,
        "ema30": 98.5,
        "ema10": 99.4,
        "ema10_slope": 0.01,
        "sma200_slope": 0.01,
        "rsi_14": 54.0,
        "adx_19": 23.0,
        "plus_di": 27.0,
        "minus_di": 20.0,
        "macd_line": 0.2,
        "macd_signal": 0.1,
        "macd_hist": 0.1,
        "atr_14": 1.5,
        "cmf_10": 0.1,
        "rel_volume": 1.1,
        "range_high_60": 100.0,
        "range_low_60": 90.0,
        "range_high_120": 106.0,
        "range_low_120": 88.0,
        "range_width_pct": 0.1,
        "bb_width": 0.08,
        "atr_pct_percentile_252": 0.2,
        "price_slope_40": 0.0,
        "obv_slope_40": 0.2,
        "anv_slope_40": 0.2,
        "accumulation_divergence": True,
        "distribution_divergence": False,
    }

    # Build rv-hit history so watch can open at close ~= 99 against frozen base_high=100.
    rel_vols = [1.0, 1.7, 1.2, 1.6, 1.1]
    for i, rv in enumerate(rel_vols):
        p = dict(base_payload)
        p["rel_volume"] = rv
        p["close"] = 99.0 + 0.05 * i
        p["open"] = p["close"] - 0.2
        p["high"] = p["close"] + 0.8
        p["low"] = p["close"] - 0.7
        put_indicator(base_td + i * 86400, p)

    r_acc = evaluate_symbol(symbol, base_td + 4 * 86400, 80.0, cfg)
    assert r_acc["phase"] == "ACCUMULATION"
    assert r_acc["transition"] == ("BASE_FORMING", "ACCUMULATION")

    # Order is frozen: BASE_FORMING->ACCUMULATION is evaluated before WATCH trigger.
    p_watch = dict(base_payload)
    p_watch.update({"open": 99.4, "close": 99.4, "high": 100.0, "low": 98.8, "rel_volume": 1.8})
    put_indicator(base_td + 5 * 86400, p_watch)
    r_watch = evaluate_symbol(symbol, base_td + 5 * 86400, 80.0, cfg)
    assert r_watch["phase"] == "BREAKOUT_WATCH"
    assert r_watch["transition"] == ("ACCUMULATION", "BREAKOUT_WATCH")

    # Next bars: close breaks frozen base_high(100) but remains below rolling 120-high(106).
    p_break = dict(base_payload)
    p_break.update({"open": 100.4, "close": 101.0, "high": 101.4, "low": 99.8, "rel_volume": 2.8})
    put_indicator(base_td + 6 * 86400, p_break)
    p_break2 = dict(p_break)
    p_break2.update({"open": 100.8, "close": 101.3, "high": 101.8, "low": 100.2, "rsi_14": 56.0, "macd_hist": 0.12})
    put_indicator(base_td + 7 * 86400, p_break2)
    p_break3 = dict(p_break)
    p_break3.update({"open": 101.0, "close": 101.6, "high": 102.0, "low": 100.4, "rsi_14": 57.0, "macd_hist": 0.15})
    put_indicator(base_td + 8 * 86400, p_break3)

    evaluate_symbol(symbol, base_td + 6 * 86400, 80.0, cfg)
    r_confirm = evaluate_symbol(symbol, base_td + 7 * 86400, 80.0, cfg)
    assert r_confirm["signal_type"] == "BREAKOUT_CONFIRMED"


def test_u1e_base_breakdown_invalidation_clears_frozen_base():
    symbol = "U1E"
    base_td = 1717000000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "base_drift_invalidate_sessions": 10,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
    }

    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'BASE_FORMING', ?, 100.0, 90.0, ?, '{}')",
        (symbol, base_td, base_td),
    )

    payload = {
        "open": 88.5,
        "high": 89.0,
        "low": 86.0,
        "close": 87.5,
        "volume": 100000.0,
        "value_kwd": 200000.0,
        "sma200": 80.0,
        "ema30": 86.0,
        "ema10": 87.0,
        "sma200_slope": 0.01,
        "rsi_14": 52.0,
        "adx_19": 23.0,
        "plus_di": 26.0,
        "minus_di": 18.0,
        "macd_line": 0.1,
        "macd_signal": 0.05,
        "macd_hist": 0.05,
        "atr_14": 1.2,
        "cmf_10": 0.02,
        "rel_volume": 1.0,
        "range_high_60": 100.0,
        "range_low_60": 90.0,
        "range_high_120": 102.0,
        "range_low_120": 88.0,
        "range_width_pct": 0.1,
        "bb_width": 0.08,
        "atr_pct_percentile_252": 0.15,
        "price_slope_40": 0.0,
        "obv_slope_40": 0.0,
        "anv_slope_40": 0.0,
        "accumulation_divergence": False,
        "distribution_divergence": False,
    }

    for i, c in enumerate([87.5, 87.0]):
        td = base_td + i * 86400
        p = dict(payload)
        p["close"] = c
        p["open"] = c + 0.2
        p["high"] = c + 1.0
        p["low"] = c - 1.3
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(p), "ee-2.1.2"),
        )

    evaluate_symbol(symbol, base_td, 72.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    r2 = evaluate_symbol(symbol, base_td + 86400, 72.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    assert r2["phase"] == "NEUTRAL"
    assert r2["signal_type"] == "PHASE_ONLY"

    st = query_one("SELECT phase, base_high, base_low, state_json FROM ee_symbol_state WHERE symbol = ?", (symbol,))
    assert st is not None
    assert st["phase"] == "NEUTRAL"
    assert st["base_high"] is None and st["base_low"] is None


def test_u1f_base_structure_invalidation_after_upward_drift_without_qualifying_breakout():
    symbol = "U1F"
    base_td = 1718000000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "base_drift_invalidate_sessions": 3,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
    }

    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'ACCUMULATION', ?, 100.0, 90.0, ?, '{}')",
        (symbol, base_td, base_td),
    )

    payload = {
        "open": 100.7,
        "high": 101.8,
        "low": 100.2,
        "close": 101.2,
        "volume": 100000.0,
        "value_kwd": 220000.0,
        "sma200": 95.0,
        "ema30": 99.0,
        "ema10": 100.5,
        "sma200_slope": 0.01,
        "rsi_14": 53.0,
        "adx_19": 24.0,
        "plus_di": 26.0,
        "minus_di": 18.0,
        "macd_line": 0.1,
        "macd_signal": 0.05,
        "macd_hist": 0.05,
        "atr_14": 1.1,
        "cmf_10": 0.01,
        "rel_volume": 1.0,
        "range_high_60": 100.0,
        "range_low_60": 90.0,
        "range_high_120": 103.0,
        "range_low_120": 88.0,
        "range_width_pct": 0.1,
        "bb_width": 0.08,
        "atr_pct_percentile_252": 0.15,
        "price_slope_40": 0.0,
        "obv_slope_40": 0.0,
        "anv_slope_40": 0.0,
        "accumulation_divergence": False,
        "distribution_divergence": False,
    }

    for i in range(3):
        td = base_td + i * 86400
        p = dict(payload)
        p["close"] = 101.1 + (0.1 * i)
        p["open"] = p["close"] - 0.2
        p["high"] = p["close"] + 0.8
        p["low"] = p["close"] - 0.7
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(p), "ee-2.1.2"),
        )

    evaluate_symbol(symbol, base_td, 74.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    evaluate_symbol(symbol, base_td + 86400, 74.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    r3 = evaluate_symbol(symbol, base_td + 2 * 86400, 74.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    assert r3["phase"] == "NEUTRAL"
    assert r3["signal_type"] == "PHASE_ONLY"


def test_u1g_exit_rearm_clears_frozen_base_and_breakout_state():
    symbol = "U1G"
    base_td = 1719000000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "base_drift_invalidate_sessions": 10,
        "exit_cooldown_sessions": 2,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
    }

    state_json = {
        "breakout_confirmed_at": base_td - 86400,
        "breakout_base_high": 100.0,
        "breakout_entry_price": 102.0,
        "confirming": {"bars": 1, "scores": []},
        "ema30_armed": True,
        "below_ema30_streak": 1,
    }
    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'EXIT', ?, 100.0, 90.0, ?, ?)",
        (symbol, base_td, base_td, json.dumps(state_json)),
    )

    payload = {
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 100000.0,
        "value_kwd": 220000.0,
        "sma200": 95.0,
        "ema30": 99.0,
        "ema10": 100.0,
        "sma200_slope": 0.01,
        "rsi_14": 55.0,
        "adx_19": 22.0,
        "plus_di": 24.0,
        "minus_di": 20.0,
        "macd_line": 0.1,
        "macd_signal": 0.05,
        "macd_hist": 0.05,
        "atr_14": 1.0,
        "cmf_10": 0.01,
        "rel_volume": 1.0,
        "range_high_60": 101.0,
        "range_low_60": 90.0,
        "range_high_120": 103.0,
        "range_low_120": 88.0,
        "range_width_pct": 0.1,
        "bb_width": 0.08,
        "atr_pct_percentile_252": 0.15,
        "price_slope_40": 0.0,
        "obv_slope_40": 0.0,
        "anv_slope_40": 0.0,
        "accumulation_divergence": False,
        "distribution_divergence": False,
    }

    for i in range(3):
        td = base_td + i * 86400
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(payload), "ee-2.1.2"),
        )

    evaluate_symbol(symbol, base_td, 75.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    evaluate_symbol(symbol, base_td + 86400, 75.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    r3 = evaluate_symbol(symbol, base_td + 2 * 86400, 75.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    assert r3["phase"] == "NEUTRAL"

    st = query_one("SELECT base_high, base_low, state_json FROM ee_symbol_state WHERE symbol = ?", (symbol,))
    assert st is not None
    assert st["base_high"] is None and st["base_low"] is None
    sj = json.loads(st["state_json"] or "{}")
    assert sj.get("breakout_confirmed_at") is None
    assert sj.get("breakout_base_high") is None
    assert sj.get("confirming") is None


def test_u1h_base_freeze_logs_provenance_event():
    symbol = "U1H"
    base_td = 1720000000
    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "base_drift_invalidate_sessions": 10,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
        "trend_join_window": 40,
    }

    for i in range(60):
        td = base_td + i * 86400
        payload = {
            "open": 95.0,
            "high": 96.0,
            "low": 94.0,
            "close": 95.0,
            "volume": 100000.0,
            "value_kwd": 220000.0,
            "sma200": 100.0,
            "ema30": 96.0,
            "ema10": 95.5,
            "sma200_slope": 0.01,
            "rsi_14": 50.0,
            "adx_19": 20.0,
            "plus_di": 24.0,
            "minus_di": 20.0,
            "macd_line": 0.1,
            "macd_signal": 0.05,
            "macd_hist": 0.05,
            "atr_14": 1.0,
            "cmf_10": 0.01,
            "rel_volume": 1.0,
            "range_high_60": 100.0,
            "range_low_60": 90.0,
            "range_high_120": 102.0,
            "range_low_120": 88.0,
            "range_width_pct": 0.1,
            "bb_width": 0.08,
            "atr_pct_percentile_252": 0.15,
            "price_slope_40": 0.0,
            "obv_slope_40": 0.0,
            "anv_slope_40": 0.0,
            "accumulation_divergence": False,
            "distribution_divergence": False,
        }
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(payload), "ee-2.1.2"),
        )

    result = evaluate_symbol(symbol, base_td + 59 * 86400, 72.0, cfg, liquidity_snapshot=(True, {"source": "unit"}))
    assert result["phase"] == "BASE_FORMING"
    assert result["transition"] == ("NEUTRAL", "BASE_FORMING")

    st = query_one("SELECT state_json FROM ee_symbol_state WHERE symbol = ?", (symbol,))
    assert st is not None
    sj = json.loads(st["state_json"] or "{}")
    ev = sj.get("base_lifecycle_last_event") or {}
    assert ev.get("action") == "base_freeze"
    assert ev.get("new", {}).get("base_high") == 100.0


def test_u2_no_lookahead_mutating_future_bar_keeps_prior_signals_identical():
    _load_fixture("TIJARA")
    compute_and_store_symbol("TIJARA")
    mn, mx = _bounds(["TIJARA"])

    report1 = run_backtest(["TIJARA"], mn, mx)
    assert isinstance(report1.get("equity_curve"), list)

    mutation_date = query_val(
        "SELECT trade_date FROM ee_ohlcv WHERE symbol = ? ORDER BY trade_date ASC LIMIT 1 OFFSET 120",
        ("TIJARA",),
    )
    baseline = query_all(
        "SELECT trade_date, evidence_json FROM ee_signals WHERE symbol = ? AND trade_date <= ? ORDER BY trade_date, id",
        ("TIJARA", mutation_date),
    )
    baseline_hashes = [(int(r["trade_date"]), _json_hash(json.loads(r["evidence_json"]))) for r in baseline]

    exec_sql(
        "UPDATE ee_ohlcv SET close = close * 1.25, high = high * 1.25, low = low * 1.25 WHERE symbol = ? AND trade_date > ?",
        ("TIJARA", mutation_date),
    )
    compute_and_store_symbol("TIJARA")

    report2 = run_backtest(["TIJARA"], mn, mx)
    assert isinstance(report2.get("equity_curve"), list)

    after = query_all(
        "SELECT trade_date, evidence_json FROM ee_signals WHERE symbol = ? AND trade_date <= ? ORDER BY trade_date, id",
        ("TIJARA", mutation_date),
    )
    after_hashes = [(int(r["trade_date"]), _json_hash(json.loads(r["evidence_json"]))) for r in after]
    assert baseline_hashes == after_hashes


def test_u3_eod_idempotency_same_date_same_signals_single_summary_event():
    _load_fixture("ZAIN")
    compute_and_store_symbol("ZAIN")

    r1 = run_eod_pipeline(source="scheduler")
    d = int(r1["data"]["run_date"])
    sig1 = query_all(
        "SELECT symbol, trade_date, signal_type, phase_from, phase_to, score, price, stop_price, evidence_json, config_hash FROM ee_signals WHERE trade_date = ? ORDER BY id",
        (d,),
    )
    sig1 = [dict(r) for r in sig1]

    r2 = run_eod_pipeline(source="scheduler")
    sig2 = query_all(
        "SELECT symbol, trade_date, signal_type, phase_from, phase_to, score, price, stop_price, evidence_json, config_hash FROM ee_signals WHERE trade_date = ? ORDER BY id",
        (d,),
    )
    sig2 = [dict(r) for r in sig2]

    assert sig1 == sig2
    n_summary = int(
        query_val(
            "SELECT COUNT(1) FROM ee_audit_events WHERE action = 'eod_pipeline_run' AND entity_id = ?",
            (f"eagle_eye:{d}",),
        )
        or 0
    )
    assert n_summary == 1


def test_u9_driver_equivalence():
    symbols = ["TIJARA", "BPCC", "ZAIN", "SANAM", "MABANEE", "JOINER"]

    mn, mx = _load_driver_equivalence_fixtures(symbols)
    run_backtest(symbols, mn, mx, config_overrides={"min_daily_value_kwd": 100000.0})

    path_a_signals = {
        symbol: [
            (int(row["trade_date"]), str(row["signal_type"]), str(row["phase_from"] or ""), str(row["phase_to"] or ""))
            for row in query_all(
                "SELECT trade_date, signal_type, phase_from, phase_to FROM ee_signals WHERE symbol = ? ORDER BY trade_date, id",
                (symbol,),
            )
        ]
        for symbol in symbols
    }
    path_a_states = {
        symbol: query_one(
            "SELECT symbol, phase, phase_since, base_high, base_low, base_start, last_score, avoid_until, updated_at, state_json FROM ee_symbol_state WHERE symbol = ?",
            (symbol,),
        )
        for symbol in symbols
    }

    _reset_driver_all_tables()
    mn, mx = _load_driver_equivalence_fixtures(symbols)
    _reset_driver_runtime_tables()

    cfg = get_active_config()
    indicators_by_symbol: dict[str, dict[int, dict]] = {}
    dates_by_symbol: dict[str, list[int]] = {}
    for symbol in symbols:
        rows = query_all(
            "SELECT trade_date, payload_json FROM ee_indicators WHERE symbol = ? ORDER BY trade_date ASC",
            (symbol,),
        )
        indicators_by_symbol[symbol] = {}
        dates_by_symbol[symbol] = []
        for row in rows:
            td = int(row["trade_date"])
            payload = json.loads(row["payload_json"] or "{}")
            payload["trade_date"] = td
            indicators_by_symbol[symbol][td] = payload
            dates_by_symbol[symbol].append(td)

    global_dates = sorted({d for dates in dates_by_symbol.values() for d in dates if mn <= d <= mx})
    for dt in global_dates:
        for symbol in sorted(symbols):
            payload = indicators_by_symbol.get(symbol, {}).get(dt)
            if not payload:
                continue
            score, band, components = compute_rating_from_indicator(payload)
            process_bar(
                symbol,
                dt,
                cfg,
                trace_id="u9-driver-equivalence",
                persist_state=True,
                indicator_payload=payload,
                score=score,
                band=band,
                components=components,
                persist_rating=True,
            )

    path_b_signals = {
        symbol: [
            (int(row["trade_date"]), str(row["signal_type"]), str(row["phase_from"] or ""), str(row["phase_to"] or ""))
            for row in query_all(
                "SELECT trade_date, signal_type, phase_from, phase_to FROM ee_signals WHERE symbol = ? ORDER BY trade_date, id",
                (symbol,),
            )
        ]
        for symbol in symbols
    }
    path_b_states = {
        symbol: query_one(
            "SELECT symbol, phase, phase_since, base_high, base_low, base_start, last_score, avoid_until, updated_at, state_json FROM ee_symbol_state WHERE symbol = ?",
            (symbol,),
        )
        for symbol in symbols
    }

    def _normalized_state(row: dict | None) -> dict:
        data = dict(row or {})
        state_json = json.loads(str(data.get("state_json") or "{}"))
        for k in ["warmup_ready_date", "warmup_sessions", "warmup_note_emitted", "last_phase_reason"]:
            state_json.pop(k, None)
        data["state_json"] = json.dumps(state_json, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return data

    for symbol in symbols:
        assert path_a_signals[symbol] == path_b_signals[symbol]
        assert _normalized_state(path_a_states[symbol]) == _normalized_state(path_b_states[symbol])


def test_u4_signals_api_auth_and_config_workflow(test_client, auth_headers):
    # Auth required on all signals endpoints.
    endpoints = [
        ("get", "/api/v1/eagle-eye/signals/watchlist"),
        ("get", "/api/v1/eagle-eye/signals/signals"),
        ("get", "/api/v1/eagle-eye/signals/signals/1"),
        ("get", "/api/v1/eagle-eye/signals/ratings/TIJARA"),
        ("get", "/api/v1/eagle-eye/signals/state/TIJARA"),
        ("post", "/api/v1/eagle-eye/signals/scan/run"),
        ("get", "/api/v1/eagle-eye/signals/config"),
        ("put", "/api/v1/eagle-eye/signals/config"),
        ("get", "/api/v1/eagle-eye/signals/performance"),
    ]
    for method, url in endpoints:
        if method == "get":
            resp = test_client.get(url)
        elif method == "post":
            resp = test_client.post(url, json={"source": "manual"})
        else:
            resp = test_client.put(url, json={"target_area": "scanner", "change_request_id": 1, "values": {"base_min_sessions": 61}})
        assert resp.status_code == 401

    _load_fixture("TIJARA")
    compute_and_store_symbol("TIJARA")

    cfg = test_client.get("/api/v1/eagle-eye/signals/config", headers=auth_headers)
    assert cfg.status_code == 200
    assert cfg.json()["data"]["advice"] is False

    put_non_admin = test_client.put(
        "/api/v1/eagle-eye/signals/config",
        headers=auth_headers,
        json={"target_area": "scanner", "change_request_id": 1, "values": {"base_min_sessions": 61}},
    )
    assert put_non_admin.status_code == 403

    admin_headers = _seed_admin_user(test_client)

    cr = test_client.post(
        "/api/v1/eagle-eye/audit/change-requests",
        headers=auth_headers,
        json={
            "title": "Phase E config test",
            "description": "Approve scanner config update for endpoint gate test",
            "target_area": "scanner",
            "change_category": "enhancement",
            "proposed_payload": {"base_min_sessions": 61},
            "status": "proposed",
        },
    )
    assert cr.status_code == 200, cr.text
    cr_id = int(cr.json()["data"]["id"])

    rv = test_client.post(
        f"/api/v1/eagle-eye/audit/change-requests/{cr_id}/review",
        headers=admin_headers,
        json={"decision": "approved", "review_notes": "approved for test"},
    )
    assert rv.status_code == 200, rv.text

    mismatch = test_client.put(
        "/api/v1/eagle-eye/signals/config",
        headers=admin_headers,
        json={
            "target_area": "risk_management",
            "change_request_id": cr_id,
            "values": {"base_min_sessions": 61},
        },
    )
    assert mismatch.status_code in (400, 409)

    put_ok = test_client.put(
        "/api/v1/eagle-eye/signals/config",
        headers=admin_headers,
        json={
            "target_area": "scanner",
            "change_request_id": cr_id,
            "values": {"base_min_sessions": 61},
        },
    )
    assert put_ok.status_code == 200, put_ok.text
    assert put_ok.json()["data"]["advice"] is False


    wl = test_client.get("/api/v1/eagle-eye/signals/watchlist", headers=auth_headers)
    assert wl.status_code == 200
    assert wl.json()["data"]["advice"] is False

    sigs = test_client.get("/api/v1/eagle-eye/signals/signals", headers=auth_headers)
    assert sigs.status_code == 200
    assert sigs.json()["data"]["advice"] is False

    st = test_client.get("/api/v1/eagle-eye/signals/state/TIJARA", headers=auth_headers)
    assert st.status_code == 200
    assert st.json()["data"]["advice"] is False

    perf = test_client.get("/api/v1/eagle-eye/signals/performance", headers=auth_headers)
    assert perf.status_code == 200
    assert perf.json()["data"]["advice"] is False


def test_u5_risk_suppression_emits_signal_and_skips_position():
    symbol = "NINE"
    # Seed liquidity-pass OHLCV rows.
    for i in range(80):
        td = 1700000000 + i * 86400
        exec_sql(
            "INSERT INTO ee_ohlcv (symbol, trade_date, open, high, low, close, volume, value_kwd, source, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', ?)",
            (symbol, td, 108.0, 121.0, 105.0, 110.0 + (i * 0.1), 100000, 250000.0, td),
        )
        payload = {
            "trade_date": td,
            "open": 111.0,
            "high": 121.0,
            "low": 118.0,
            "close": 120.0,
            "volume": 150000,
            "value_kwd": 260000,
            "sma200": 100.0,
            "ema30": 112.0,
            "ema10": 118.0,
            "ema10_slope": 0.02,
            "sma200_slope": 0.01,
            "rsi_14": 50.0 + (i * 0.2),
            "adx_19": 18.0 + (i % 10),
            "plus_di": 35.0,
            "minus_di": 18.0,
            "macd_line": 1.0,
            "macd_signal": 0.6,
            "macd_hist": 0.4,
            "atr_14": 2.0,
            "cmf_10": 0.12,
            "rel_volume": 3.0,
            "range_high_60": 110.0,
            "range_low_60": 90.0,
            "range_high_120": 110.0,
            "range_low_120": 90.0,
            "range_width_pct": 0.17,
            "bb_width": 0.09,
            "atr_pct_percentile_252": 0.10,
            "accumulation_divergence": True,
            "distribution_divergence": False,
        }
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(payload), "ee-2.1.1-hotfix"),
        )

    latest = query_val("SELECT MAX(trade_date) FROM ee_indicators WHERE symbol = ?", (symbol,))

    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'BREAKOUT_WATCH', ?, 110.0, 90.0, ?, '{}')",
        (symbol, latest - 86400, latest),
    )

    for i in range(8):
        exec_sql(
            "INSERT INTO ee_positions (symbol, opened_at, status, tranches_json, avg_entry, stop_price, trail_price, signal_id) VALUES (?, ?, 'open', '[]', 100, 95, 96, 1)",
            (f"OPEN{i}", latest - 1000),
        )

    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
    }
    evaluate_symbol(symbol, int(latest), 80.0, cfg)

    sup = query_all("SELECT * FROM ee_signals WHERE symbol = ? AND signal_type = 'SIGNAL_SUPPRESSED_RISK'", (symbol,))
    assert len(sup) == 1
    breakout = query_all("SELECT * FROM ee_signals WHERE symbol = ? AND signal_type = 'BREAKOUT_CONFIRMED'", (symbol,))
    assert breakout == []

    p = query_all("SELECT * FROM ee_positions WHERE symbol = ?", (symbol,))
    assert len(p) == 0


def test_u6_liquidity_filter_blocks_publishable_signals():
    _load_fixture("SANAM", low_liquidity=True)
    compute_and_store_symbol("SANAM")

    latest = int(query_val("SELECT MAX(trade_date) FROM ee_indicators WHERE symbol = ?", ("SANAM",)) or 0)
    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, updated_at, state_json) VALUES (?, 'BASE_FORMING', ?, ?, '{}')",
        ("SANAM", latest - 86400, latest),
    )

    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
    }
    evaluate_symbol("SANAM", latest, 82.0, cfg)

    longs = query_all(
        "SELECT signal_type FROM ee_signals WHERE symbol = ? AND signal_type IN ('ACCUMULATION_ALERT', 'BREAKOUT_CONFIRMED')",
        ("SANAM",),
    )
    assert longs == []


def test_u7_default_config_has_all_required_keys_for_scanner_paths():
    symbol = "CFGCHK"
    td = 1719000000
    exec_sql(
        "INSERT INTO ee_ohlcv (symbol, trade_date, open, high, low, close, volume, value_kwd, source, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', ?)",
        (symbol, td, 100.0, 103.0, 99.0, 101.0, 120000, 250000.0, td),
    )

    payload = {
        "trade_date": td,
        "open": 101.0,
        "high": 104.0,
        "low": 99.5,
        "close": 102.0,
        "volume": 120000,
        "value_kwd": 250000.0,
        "sma200": 95.0,
        "ema30": 100.0,
        "ema10": 101.0,
        "ema10_slope": 0.02,
        "sma200_slope": 0.01,
        "rsi_14": 58.0,
        "adx_19": 24.0,
        "plus_di": 30.0,
        "minus_di": 14.0,
        "macd_line": 0.8,
        "macd_signal": 0.4,
        "macd_hist": 0.4,
        "atr_14": 1.5,
        "cmf_10": 0.1,
        "rel_volume": 1.8,
        "range_high_60": 106.0,
        "range_low_60": 92.0,
        "range_high_120": 108.0,
        "range_low_120": 90.0,
        "range_width_pct": 0.14,
        "bb_width": 0.09,
        "atr_pct_percentile_252": 0.2,
        "price_slope_40": 0.0,
        "obv_slope_40": 0.2,
        "anv_slope_40": 0.2,
        "accumulation_divergence": True,
        "distribution_divergence": False,
    }

    cfg = dict(DEFAULT_ENGINE_CONFIG)
    phases = [
        "NEUTRAL",
        "BASE_FORMING",
        "ACCUMULATION",
        "BREAKOUT_WATCH",
        "BREAKOUT_CONFIRMED",
        "MARKUP",
        "DISTRIBUTION_WARNING",
        "EXIT",
        "AVOID",
    ]

    for phase in phases:
        state = {
            "symbol": symbol,
            "phase": phase,
            "phase_since": td,
            "base_high": 106.0,
            "base_low": 92.0,
            "base_start": td,
            "last_score": 75.0,
            "avoid_until": None,
            "updated_at": td,
            "state_json": {},
        }
        try:
            evaluate_symbol(
                symbol,
                td,
                75.0,
                cfg,
                indicator_payload=payload,
                indicator_history=[payload],
                state_override=state,
                persist_state=False,
                coverage_start_date=td,
            )
        except ConfigKeyMissing as exc:
            pytest.fail(f"Missing config key for phase {phase}: {exc}")


def test_u8_one_transition_emits_single_signal_row_when_suppressed():
    symbol = "DEDUP"
    for i in range(80):
        td = 1710000000 + i * 86400
        close = 100.0 + (i * 0.2)
        exec_sql(
            "INSERT INTO ee_ohlcv (symbol, trade_date, open, high, low, close, volume, value_kwd, source, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', ?)",
            (symbol, td, close - 0.5, close + 1.5, close - 1.0, close, 120000, 300000.0, td),
        )
        payload = {
            "trade_date": td,
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 0.6,
            "close": close,
            "volume": 120000,
            "value_kwd": 300000.0,
            "sma200": 95.0,
            "ema30": close - 0.5,
            "ema10": close - 0.2,
            "ema10_slope": 0.03,
            "sma200_slope": 0.01,
            "rsi_14": 55.0 + (i * 0.1),
            "adx_19": 28.0,
            "plus_di": 33.0,
            "minus_di": 14.0,
            "macd_line": 1.4,
            "macd_signal": 0.6,
            "macd_hist": 0.8,
            "atr_14": 1.8,
            "cmf_10": 0.12,
            "rel_volume": 3.0,
            "range_high_60": close - 1.5,
            "range_low_60": close - 10.0,
            "range_high_120": close - 1.5,
            "range_low_120": close - 15.0,
            "range_width_pct": 0.15,
            "bb_width": 0.08,
            "atr_pct_percentile_252": 0.11,
            "price_slope_40": 0.0,
            "obv_slope_40": 0.2,
            "anv_slope_40": 0.2,
            "accumulation_divergence": True,
            "distribution_divergence": False,
        }
        exec_sql(
            "INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version) VALUES (?, ?, ?, ?) ON CONFLICT(symbol, trade_date) DO UPDATE SET payload_json = excluded.payload_json",
            (symbol, td, json.dumps(payload), "ee-2.1.2"),
        )

    latest = int(query_val("SELECT MAX(trade_date) FROM ee_indicators WHERE symbol = ?", (symbol,)) or 0)
    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'BREAKOUT_WATCH', ?, ?, ?, ?, '{}')",
        (symbol, latest - 86400, 112.0, 95.0, latest),
    )

    for i in range(8):
        exec_sql(
            "INSERT INTO ee_positions (symbol, opened_at, status, tranches_json, avg_entry, stop_price, trail_price, signal_id) VALUES (?, ?, 'open', '[]', 100, 95, 96, 1)",
            (f"HOLD{i}", latest - 1000),
        )

    cfg = {
        "base_min_sessions": 60,
        "base_max_width_pct": 0.18,
        "volume_breakout_mult": 2.5,
        "rsi_regime": 55,
        "adx_trigger": 22,
        "cmf_floor": 0.05,
        "atr_squeeze_pctile": 0.20,
        "pilot_enabled": True,
        "max_positions": 8,
        "min_daily_value_kwd": 100000.0,
    }
    evaluate_symbol(symbol, latest, 82.0, cfg)

    rows = query_all("SELECT signal_type FROM ee_signals WHERE symbol = ? AND trade_date = ? ORDER BY id", (symbol, latest))
    assert len(rows) == 1
    assert rows[0]["signal_type"] == "SIGNAL_SUPPRESSED_RISK"


def test_lint_single_evaluate_symbol_callsite():
    root = Path(__file__).resolve().parents[2] / "app"
    callsites: list[str] = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(Path(__file__).resolve().parents[2]).as_posix()
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "evaluate_symbol":
                callsites.append(f"{rel}:{int(node.lineno)}")

    assert len(callsites) == 1
    assert callsites[0].startswith("app/services/eagle_eye/pipeline.py:")
