from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.core.database import exec_sql, query_all, query_one, query_val
from app.core.security import TokenData
from app.services.eagle_eye.backtest_service import run_backtest
from app.services.eagle_eye.indicator_service import compute_and_store_symbol
from app.services.eagle_eye.market_data_service import ensure_schema, load_ohlcv_csv
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


def test_u1_state_machine_transition_matrix():
    assert_valid_transition("NEUTRAL", "BASE_FORMING")
    assert_valid_transition("BREAKOUT_WATCH", "BREAKOUT_CONFIRMED")
    assert_valid_transition("MARKUP", "DISTRIBUTION_WARNING")
    assert_valid_transition("DISTRIBUTION_WARNING", "EXIT")

    with pytest.raises(ValueError):
        assert_valid_transition("NEUTRAL", "EXIT")

    with pytest.raises(ValueError):
        assert_valid_transition("AVOID", "MARKUP")


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

    r2 = run_eod_pipeline(source="scheduler")
    sig2 = query_all(
        "SELECT symbol, trade_date, signal_type, phase_from, phase_to, score, price, stop_price, evidence_json, config_hash FROM ee_signals WHERE trade_date = ? ORDER BY id",
        (d,),
    )

    assert sig1 == sig2
    n_summary = int(
        query_val(
            "SELECT COUNT(1) FROM ee_audit_events WHERE action = 'eod_pipeline_run' AND entity_id = ?",
            (f"eagle_eye:{d}",),
        )
        or 0
    )
    assert n_summary == 1


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
            (symbol, td, json.dumps(payload), "ee-2.1.0-verification"),
        )

    latest = query_val("SELECT MAX(trade_date) FROM ee_indicators WHERE symbol = ?", (symbol,))

    exec_sql(
        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, updated_at, state_json) VALUES (?, 'BREAKOUT_WATCH', ?, ?, '{}')",
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
    assert len(sup) >= 1

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
