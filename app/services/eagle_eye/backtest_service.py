from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from collections import deque
import statistics

from app.core.database import exec_sql, query_all, query_one
from app.core.security import TokenData
from app.services.eagle_eye.audit_service import create_event
from app.services.eagle_eye.indicator_service import compute_and_store_symbol
from app.services.eagle_eye.market_data_service import (
    CONCEPT_VERSION,
    get_active_config,
    get_config_hash,
    now_ts,
    set_now_ts_override,
    validate_engine_config_presence,
    validate_runtime_config_keys,
)
from app.services.eagle_eye.rating_service import compute_rating_from_indicator, store_rating
from app.services.eagle_eye.pipeline import process_bar
from app.services.eagle_eye.scanner_service import upsert_symbol_state


@dataclass
class PendingFill:
    symbol: str
    side: str
    tranche: str
    signal_id: int
    signal_type: str
    signal_date: int
    execute_date: int


def _system_actor() -> TokenData:
    return TokenData(user_id=0, username="system", is_admin=True)


def _ensure_backtest_tables() -> None:
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            started_at INTEGER NOT NULL,
            ended_at INTEGER,
            symbols_json TEXT NOT NULL,
            start_date INTEGER NOT NULL,
            end_date INTEGER NOT NULL,
            config_hash TEXT NOT NULL,
            concept_version TEXT NOT NULL,
            report_json TEXT,
            status TEXT NOT NULL DEFAULT 'running'
        )
        """,
        (),
    )
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_backtest_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            opened_at INTEGER NOT NULL,
            closed_at INTEGER,
            side TEXT NOT NULL DEFAULT 'long',
            tranches_json TEXT NOT NULL,
            avg_entry REAL,
            avg_exit REAL,
            gross_return REAL,
            net_return REAL,
            exit_reason TEXT,
            meta_json TEXT
        )
        """,
        (),
    )


def _load_bars(symbol: str, start: int, end: int) -> list[dict[str, Any]]:
    return query_all(
        """
        SELECT trade_date, open, high, low, close, volume, value_kwd
        FROM ee_ohlcv
        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date ASC
        """,
        (symbol, start, end),
    )


def _load_indicator_map(symbol: str, start: int, end: int) -> dict[int, dict[str, Any]]:
    rows = query_all(
        """
        SELECT trade_date, payload_json
        FROM ee_indicators
        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date ASC
        """,
        (symbol, start, end),
    )
    out: dict[int, dict[str, Any]] = {}
    for row in rows or []:
        trade_date = int(row.get("trade_date") or 0)
        try:
            payload = json.loads(str(row.get("payload_json") or "{}"))
        except Exception:
            payload = {}
        payload["trade_date"] = trade_date
        out[trade_date] = payload
    return out


def _precompute_liquidity_map(
    bars: list[dict[str, Any]],
    min_daily_value_kwd: float,
) -> dict[int, tuple[bool, dict[str, Any]]]:
    value_q: deque[float] = deque(maxlen=60)
    close_q: deque[float] = deque(maxlen=60)
    vol_q: deque[float] = deque(maxlen=60)
    out: dict[int, tuple[bool, dict[str, Any]]] = {}
    for bar in bars:
        td = int(bar.get("trade_date") or 0)
        value_q.append(float(bar.get("value_kwd") or 0.0))
        close_q.append(float(bar.get("close") or 0.0))
        vol_q.append(float(bar.get("volume") or 0.0))
        values = list(value_q)
        closes = list(close_q)
        vols = list(vol_q)
        median_val = float(statistics.median(values)) if values else 0.0
        min_price = min(closes) if closes else 0.0
        zero_vol = sum(1 for v in vols if v <= 0)
        ok = median_val >= float(min_daily_value_kwd) and min_price >= 50.0 and zero_vol <= 3
        out[td] = (
            ok,
            {
                "median_daily_value_kwd_20": median_val,
                "min_price_fils_60": min_price,
                "zero_volume_sessions_60": zero_vol,
            },
        )
    return out


def run_backtest(
    symbols: list[str],
    start: int,
    end: int,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_backtest_tables()
    validate_engine_config_presence()
    symbols_sorted = sorted({s.upper().strip() for s in symbols if s})
    cfg = get_active_config()
    if config_overrides:
        cfg.update(config_overrides)
    validate_runtime_config_keys(cfg)

    run_started = now_ts()
    config_hash = get_config_hash(cfg)
    run_id = 0

    exec_sql("DELETE FROM ee_signals WHERE trade_date BETWEEN ? AND ?", (start, end))
    exec_sql("DELETE FROM ee_ratings WHERE trade_date BETWEEN ? AND ?", (start, end))
    exec_sql("DELETE FROM ee_symbol_state", ())
    exec_sql("DELETE FROM ee_positions", ())

    run_id_row = query_one(
        """
        INSERT INTO ee_backtest_runs (
            created_at, started_at, symbols_json, start_date, end_date,
            config_hash, concept_version, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running')
        RETURNING id
        """,
        (
            run_started,
            run_started,
            json.dumps(symbols_sorted, ensure_ascii=True, separators=(",", ":")),
            start,
            end,
            config_hash,
            CONCEPT_VERSION,
        ),
    )
    if run_id_row:
        run_id = int(run_id_row.get("id") or 0)

    bars_by_symbol: dict[str, list[dict[str, Any]]] = {}
    indicators_by_symbol: dict[str, dict[int, dict[str, Any]]] = {}
    liquidity_by_symbol: dict[str, dict[int, tuple[bool, dict[str, Any]]]] = {}
    coverage_start_by_symbol: dict[str, int] = {}
    for symbol in symbols_sorted:
        compute_and_store_symbol(symbol)
        bars_by_symbol[symbol] = _load_bars(symbol, start, end)
        indicators_by_symbol[symbol] = _load_indicator_map(symbol, start, end)
        liquidity_by_symbol[symbol] = _precompute_liquidity_map(
            bars_by_symbol[symbol],
            float(cfg.get("min_daily_value_kwd", 100000.0)),
        )
        coverage_start = int(bars_by_symbol[symbol][0].get("trade_date") or start) if bars_by_symbol[symbol] else int(start)
        for td in sorted(indicators_by_symbol[symbol].keys()):
            p = indicators_by_symbol[symbol].get(td) or {}
            sma200 = float(p.get("sma200") or 0.0)
            range_low_120 = float(p.get("range_low_120") or 0.0)
            if sma200 > 0 and range_low_120 > 0:
                coverage_start = int(td)
                break
        coverage_start_by_symbol[symbol] = coverage_start

    pending: list[PendingFill] = []
    open_trades: dict[str, dict[str, Any]] = {}
    closed_trades: list[dict[str, Any]] = []
    commission_bps = float(cfg.get("bt_commission_bps", 25))
    slippage_bps = float(cfg.get("bt_slippage_bps", 30))
    fill_cost = (commission_bps + slippage_bps) / 10000.0

    global_dates = sorted({int(b.get("trade_date") or 0) for rows in bars_by_symbol.values() for b in rows if b.get("trade_date")})
    bar_index: dict[str, dict[int, dict[str, Any]]] = {
        s: {int(x.get("trade_date") or 0): x for x in rows}
        for s, rows in bars_by_symbol.items()
    }
    history_cache: dict[str, list[dict[str, Any]]] = {s: [] for s in symbols_sorted}
    coverage_count: dict[str, int] = {s: 0 for s in symbols_sorted}
    state_cache: dict[str, dict[str, Any]] = {}
    next_date_map: dict[str, dict[int, int]] = {}
    for symbol, rows in bars_by_symbol.items():
        dates = [int(x.get("trade_date") or 0) for x in rows if x.get("trade_date")]
        next_date_map[symbol] = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}

    for dt in global_dates:
        set_now_ts_override(dt)

        # Execute queued fills at next-session open.
        to_exec = [p for p in pending if p.execute_date == dt]
        pending = [p for p in pending if p.execute_date != dt]
        for pf in to_exec:
            bar = bar_index.get(pf.symbol, {}).get(dt)
            if not bar:
                continue
            px = float(bar.get("open") or 0.0)
            if px <= 0:
                continue
            if pf.side == "buy":
                t = open_trades.setdefault(
                    pf.symbol,
                    {
                        "symbol": pf.symbol,
                        "opened_at": dt,
                        "tranches": [],
                        "exit_reason": None,
                    },
                )
                t["tranches"].append(
                    {
                        "date": dt,
                        "type": pf.tranche,
                        "signal_type": pf.signal_type,
                        "signal_id": pf.signal_id,
                        "price": px,
                        "fill_cost": fill_cost,
                    }
                )
            elif pf.side == "sell" and pf.symbol in open_trades:
                t = open_trades.pop(pf.symbol)
                entries = [float(x.get("price") or 0.0) for x in t.get("tranches", [])]
                avg_entry = sum(entries) / max(1, len(entries))
                gross = (px - avg_entry) / avg_entry if avg_entry > 0 else 0.0
                # One exit side plus average entry side cost.
                net = gross - (2.0 * fill_cost)
                t.update(
                    {
                        "closed_at": dt,
                        "avg_entry": avg_entry,
                        "avg_exit": px,
                        "gross_return": gross,
                        "net_return": net,
                        "exit_reason": pf.signal_type,
                    }
                )
                closed_trades.append(t)

        for symbol in symbols_sorted:
            bar = bar_index.get(symbol, {}).get(dt)
            if not bar:
                continue

            payload = indicators_by_symbol.get(symbol, {}).get(dt)
            if not payload:
                continue
            feature_max_ts = int(payload.get("trade_date") or 0)
            assert feature_max_ts <= dt, f"No-lookahead violated for {symbol} at {dt}"  # nosec B101

            history_cache[symbol].append(payload)
            if dt >= int(coverage_start_by_symbol.get(symbol) or dt):
                coverage_count[symbol] = int(coverage_count.get(symbol, 0)) + 1
            if len(history_cache[symbol]) > 140:
                history_cache[symbol] = history_cache[symbol][-140:]

            score, band, components = compute_rating_from_indicator(payload)
            store_rating(symbol, dt, score, band, components)

            signal_result = process_bar(
                symbol,
                dt,
                cfg,
                trace_id=f"bt-{run_id}",
                indicator_payload=payload,
                indicator_history=history_cache[symbol],
                state_override=state_cache.get(symbol),
                persist_state=False,
                liquidity_snapshot=liquidity_by_symbol.get(symbol, {}).get(dt),
                coverage_start_date=coverage_start_by_symbol.get(symbol),
                coverage_sessions=coverage_count.get(symbol) if coverage_count.get(symbol, 0) > 0 else None,
                score=score,
                band=band,
                components=components,
                persist_rating=False,
            )
            if isinstance(signal_result.get("state"), dict):
                state_cache[symbol] = signal_result["state"]
            signal_id = int(signal_result.get("signal_id") or 0)
            if signal_id <= 0:
                continue
            stype = str(signal_result.get("signal_type") or "")

            # Queue fills on t+1 open only.
            next_dt = next_date_map.get(symbol, {}).get(dt)
            if next_dt is None:
                continue

            if stype == "ACCUMULATION_ALERT":
                pending.append(PendingFill(symbol, "buy", "T1", signal_id, stype, dt, next_dt))
            elif stype == "BREAKOUT_CONFIRMED":
                pending.append(PendingFill(symbol, "buy", "T2", signal_id, stype, dt, next_dt))
            elif stype == "ADD_ON_PULLBACK":
                pending.append(PendingFill(symbol, "buy", "T3", signal_id, stype, dt, next_dt))
            elif stype in {"EXIT", "BREAKOUT_FAILED"}:
                pending.append(PendingFill(symbol, "sell", "EXIT", signal_id, stype, dt, next_dt))

    set_now_ts_override(None)

    for state in state_cache.values():
        upsert_symbol_state(state)

    # Force-close at final close for open trades.
    for symbol, trade in list(open_trades.items()):
        rows = bars_by_symbol.get(symbol, [])
        if not rows:
            continue
        last = rows[-1]
        px = float(last.get("close") or 0.0)
        entries = [float(x.get("price") or 0.0) for x in trade.get("tranches", [])]
        avg_entry = sum(entries) / max(1, len(entries))
        gross = (px - avg_entry) / avg_entry if avg_entry > 0 else 0.0
        net = gross - (2.0 * fill_cost)
        trade.update(
            {
                "closed_at": int(last.get("trade_date") or end),
                "avg_entry": avg_entry,
                "avg_exit": px,
                "gross_return": gross,
                "net_return": net,
                "exit_reason": "force_close_end",
            }
        )
        closed_trades.append(trade)
        del open_trades[symbol]

    for t in closed_trades:
        exec_sql(
            """
            INSERT INTO ee_backtest_trades (
                run_id, symbol, opened_at, closed_at, side, tranches_json,
                avg_entry, avg_exit, gross_return, net_return, exit_reason, meta_json
            ) VALUES (?, ?, ?, ?, 'long', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                t.get("symbol"),
                t.get("opened_at"),
                t.get("closed_at"),
                json.dumps(t.get("tranches", []), ensure_ascii=True, separators=(",", ":")),
                t.get("avg_entry"),
                t.get("avg_exit"),
                t.get("gross_return"),
                t.get("net_return"),
                t.get("exit_reason"),
                json.dumps({"concept_version": CONCEPT_VERSION}, ensure_ascii=True, separators=(",", ":")),
            ),
        )

    # Build report
    trade_returns = [float(t.get("net_return") or 0.0) for t in closed_trades]
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r < 0]
    expectancy = sum(trade_returns) / max(1, len(trade_returns))
    win_rate = len(wins) / max(1, len(trade_returns))
    avg_win = sum(wins) / max(1, len(wins)) if wins else 0.0
    avg_loss = sum(losses) / max(1, len(losses)) if losses else 0.0

    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    equity_curve = []
    for r in trade_returns:
        eq *= (1.0 + r)
        peak = max(peak, eq)
        dd = 0.0 if peak <= 0 else (peak - eq) / peak
        max_dd = max(max_dd, dd)
        equity_curve.append(eq)

    signal_rows = query_all(
        """
        SELECT signal_type,
               COUNT(1) AS n,
               AVG(CASE WHEN outcome_label = 'WIN' THEN 1.0 ELSE 0.0 END) AS hit_rate,
               AVG(CASE WHEN outcome_return > 0 THEN outcome_return END) AS avg_win,
               AVG(CASE WHEN outcome_return < 0 THEN outcome_return END) AS avg_loss,
               AVG(outcome_return) AS expectancy
        FROM ee_signals
        WHERE trade_date BETWEEN ? AND ?
        GROUP BY signal_type
        ORDER BY signal_type
        """,
        (start, end),
    )

    report = {
        "run_id": run_id,
        "symbols": symbols_sorted,
        "start": start,
        "end": end,
        "config_hash": config_hash,
        "concept_version": CONCEPT_VERSION,
        "trades": len(closed_trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "max_drawdown": max_dd,
        "equity_curve": equity_curve,
        "signal_type_stats": [dict(r) for r in signal_rows or []],
        "advice": False,
    }

    ended = now_ts()
    exec_sql(
        """
        UPDATE ee_backtest_runs
        SET ended_at = ?, status = 'completed', report_json = ?
        WHERE id = ?
        """,
        (
            ended,
            json.dumps(report, ensure_ascii=True, separators=(",", ":")),
            run_id,
        ),
    )

    audit = create_event(
        {
            "action": "backtest_run",
            "entity_type": "backtest",
            "entity_id": str(run_id),
            "change_type": "workflow",
            "risk_level": "high",
            "source": "system",
            "metadata": {
                "config_hash": config_hash,
                "symbols": symbols_sorted,
                "summary": {
                    "trades": len(closed_trades),
                    "win_rate": win_rate,
                    "expectancy": expectancy,
                    "max_drawdown": max_dd,
                },
            },
            "concept_version": CONCEPT_VERSION,
        },
        _system_actor(),
    )
    report["audit_event_id"] = audit.get("id")
    return report


def run_regression_backtest() -> dict[str, Any]:
    symbols = ["TIJARA", "BPCC", "ZAIN", "SANAM", "MABANEE"]
    bounds = query_one("SELECT MIN(trade_date) AS mn, MAX(trade_date) AS mx FROM ee_ohlcv", ())
    if not bounds or bounds.get("mn") is None or bounds.get("mx") is None:
        return {"symbols": {}, "note": "No OHLCV data loaded", "advice": False}
    return run_backtest(symbols, int(bounds.get("mn")), int(bounds.get("mx")))
