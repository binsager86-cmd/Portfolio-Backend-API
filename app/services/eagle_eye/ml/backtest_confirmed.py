from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.services.eagle_eye.ml.data_pipeline import DataPipeline
from app.services.eagle_eye.ml.lifecycle_labeler import MIN_BARS_FOR_LABELS, label_lifecycle_states
from app.services.eagle_eye.store import list_tickers_with_ohlcv, load_ohlcv
from app.services.signal_engine.config.risk_config import TC_COMMISSION
from app.services.signal_engine.engine.backtester import _total_cost_factor


STARTING_CAPITAL_KWD = 100_000.0
TARGET_POSITION_PCT = 0.10
MIN_POSITION_KWD = 100.0
LIQUIDITY_CAP_PCT = 0.10
ADTV_LOOKBACK_DAYS = 20
STOP_LOSS_PCT = 0.08
TIME_EXIT_DAYS = 20
MIN_HISTORY_BARS = max(250, MIN_BARS_FOR_LABELS)
STAGE_A_SLEEP_SECONDS = 0.05
RANDOM_BASELINE_RUNS = 25
RANDOM_BASELINE_MAX_TRADES = 200
RANDOM_SEED = 42
REPORT_DATE = "2026_05_23"


@dataclass(frozen=True)
class CostScenario:
    name: str
    slippage_multiplier: float
    description: str


@dataclass(frozen=True)
class TradePath:
    ticker: str
    market_tier: str
    signal_date: str
    entry_date: str
    entry_index: int
    entry_open: float
    signal_close: float
    exit_date: str
    exit_index: int
    exit_price: float
    exit_reason: str
    liquidity_cap_kwd: float
    adtv20_kwd: float


@dataclass
class ExecutedTrade:
    ticker: str
    market_tier: str
    signal_date: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    exit_reason: str
    holding_days: int
    requested_size_kwd: float
    allowed_size_kwd: float
    size_kwd: float
    shrunk_for_liquidity: bool
    skipped_for_liquidity: bool
    entry_cost_kwd: float
    exit_cost_kwd: float
    total_cost_kwd: float
    commission_kwd: float
    slippage_kwd: float
    gross_pnl_kwd: float
    net_pnl_kwd: float
    net_return_pct: float


@dataclass
class SimulationResult:
    scenario_name: str
    executed_trades: list[ExecutedTrade]
    skipped_signals: list[dict[str, Any]]
    raw_signal_count: int
    tradeable_signal_count: int
    liquidity_shrink_count: int
    liquidity_skip_count: int
    equity_curve: pd.DataFrame
    equity_end_kwd: float
    total_net_pnl_kwd: float
    max_drawdown_pct: float
    expectancy_pct: float
    expectancy_kwd: float
    median_trade_return_pct: float
    mean_trade_return_pct: float
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    win_loss_ratio: Optional[float]
    tradeable_pct: float
    cagr_pct: Optional[float]
    big_loser_count: int
    outcome_buckets: dict[str, int]
    pnl_by_ticker: dict[str, float]


def _normalize_tier_value(tier: Optional[str]) -> str:
    value = (tier or "PREMIER").strip().upper()
    if value in {"PREMIER", "MAIN", "AUCTION"}:
        return value
    return "PREMIER"


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(parsed) or math.isinf(parsed):
        return 0.0
    return parsed


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


class ConfirmedBacktester:
    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[4]
        self.workspace_root = self.repo_root.parent.parent
        self.cache_dir = self.repo_root / "cache" / "l3_signals"
        self.report_path = self.repo_root / "reports" / f"lifecycle_L3_backtest_{REPORT_DATE}.md"
        self.index_proxy_path = self.workspace_root / "kse_index_proxy.csv"
        self._tier_infer = DataPipeline()._infer_liquidity_tier

    def stage_a_label_once(self, *, force: bool = False, sleep_seconds: float = STAGE_A_SLEEP_SECONDS) -> dict[str, Any]:
        tickers = self._universe_tickers()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        created = 0
        reused = 0
        for index, ticker in enumerate(tickers, start=1):
            cache_path = self._cache_path(ticker)
            if cache_path.exists() and not force:
                try:
                    cached = _read_json(cache_path)
                    reused += 1
                    print(f"{index}/{len(tickers)} {ticker}: cached {int(cached.get('signal_count') or 0)} signals", flush=True)
                    time.sleep(sleep_seconds)
                    continue
                except Exception:
                    pass

            payload = self._build_cache_payload(ticker)
            _write_json(cache_path, payload)
            created += 1
            print(f"{index}/{len(tickers)} {ticker}: {int(payload.get('signal_count') or 0)} signals", flush=True)
            time.sleep(sleep_seconds)

        summary = self.cache_status()
        summary["stage"] = "A"
        summary["created_files"] = created
        summary["reused_files"] = reused
        return summary

    def stage_b_simulate_from_cache(self) -> dict[str, Any]:
        stock_caches = self._load_cached_stocks(eligible_only=True)
        signals = self._confirmed_trade_paths(stock_caches)
        moderate = CostScenario(
            name="moderate",
            slippage_multiplier=1.0,
            description="Premier 0.15% commission + 0.10% slippage; Main 0.15% + 0.30%.",
        )
        result = self._simulate(signals, stock_caches, moderate, scenario_name="CONFIRMED moderate")
        return {
            "stage": "B",
            "eligible_cached_stocks": len(stock_caches),
            "raw_signal_count": result.raw_signal_count,
            "tradeable_signal_count": result.tradeable_signal_count,
            "executed_trade_count": len(result.executed_trades),
            "liquidity_shrink_count": result.liquidity_shrink_count,
            "liquidity_skip_count": result.liquidity_skip_count,
            "win_rate_pct": round(result.win_rate_pct, 4),
            "expectancy_pct": round(result.expectancy_pct, 6),
            "expectancy_kwd": round(result.expectancy_kwd, 4),
            "mean_trade_return_pct": round(result.mean_trade_return_pct, 6),
            "median_trade_return_pct": round(result.median_trade_return_pct, 6),
            "max_drawdown_pct": round(result.max_drawdown_pct, 6),
            "total_net_pnl_kwd": round(result.total_net_pnl_kwd, 4),
        }

    def stage_c_baselines_and_sensitivity(self) -> dict[str, Any]:
        stock_caches = self._load_cached_stocks(eligible_only=True)
        actual_signals = self._confirmed_trade_paths(stock_caches)

        moderate = CostScenario(
            name="moderate",
            slippage_multiplier=1.0,
            description="Premier 0.15% commission + 0.10% slippage; Main 0.15% + 0.30%.",
        )
        harsh = CostScenario(
            name="harsh",
            slippage_multiplier=2.0,
            description="Same commission, doubled slippage.",
        )

        actual_moderate = self._simulate(actual_signals, stock_caches, moderate, scenario_name="CONFIRMED moderate")
        actual_harsh = self._simulate(actual_signals, stock_caches, harsh, scenario_name="CONFIRMED harsh")

        best_ticker = self._best_ticker(actual_moderate)
        exclude_best_signals = [signal for signal in actual_signals if signal.ticker != best_ticker]
        exclude_best = self._simulate(
            exclude_best_signals,
            stock_caches,
            moderate,
            scenario_name=f"CONFIRMED moderate ex-{best_ticker}" if best_ticker else "CONFIRMED moderate ex-best",
        )

        random_baseline = self._random_baseline(stock_caches, actual_signals, moderate)
        index_baseline = self._index_baseline(actual_moderate)

        report = self._render_report(
            stock_caches=stock_caches,
            actual_moderate=actual_moderate,
            actual_harsh=actual_harsh,
            exclude_best=exclude_best,
            best_ticker=best_ticker,
            random_baseline=random_baseline,
            index_baseline=index_baseline,
        )
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(report, encoding="utf-8")

        return {
            "stage": "C",
            "report_path": str(self.report_path),
            "executed_trade_count": len(actual_moderate.executed_trades),
            "expectancy_pct": round(actual_moderate.expectancy_pct, 6),
            "harsh_expectancy_pct": round(actual_harsh.expectancy_pct, 6),
            "exclude_best_expectancy_pct": round(exclude_best.expectancy_pct, 6),
            "best_ticker": best_ticker,
            "random_baseline": random_baseline,
            "index_baseline": index_baseline,
        }

    def run_all(self, *, force_stage_a: bool = False, sleep_seconds: float = STAGE_A_SLEEP_SECONDS) -> dict[str, Any]:
        stage_a = self.stage_a_label_once(force=force_stage_a, sleep_seconds=sleep_seconds)
        stage_b = self.stage_b_simulate_from_cache()
        result = {"stage_a": stage_a, "stage_b": stage_b}
        if stage_b["executed_trade_count"] > 0 and stage_b["expectancy_pct"] > 0.0:
            result["stage_c"] = self.stage_c_baselines_and_sensitivity()
        else:
            result["stage_c"] = {
                "skipped": True,
                "reason": "Stage B was too weak to justify baselines yet.",
            }
        return result

    def cache_status(self) -> dict[str, Any]:
        files = sorted(self.cache_dir.glob("*.json")) if self.cache_dir.exists() else []
        eligible = 0
        total_signals = 0
        statuses: dict[str, int] = {}
        for file_path in files:
            try:
                payload = _read_json(file_path)
            except Exception:
                statuses["corrupt"] = statuses.get("corrupt", 0) + 1
                continue
            status = str(payload.get("status") or "unknown")
            statuses[status] = statuses.get(status, 0) + 1
            if payload.get("eligible"):
                eligible += 1
                total_signals += int(payload.get("signal_count") or 0)
        return {
            "cache_dir": str(self.cache_dir),
            "cache_file_count": len(files),
            "eligible_cached_stocks": eligible,
            "total_confirmed_signals": total_signals,
            "status_breakdown": statuses,
        }

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}.json"

    def _universe_tickers(self) -> list[str]:
        return sorted({ticker.upper() for ticker in list_tickers_with_ohlcv() if ticker.upper() != "KSE_ALL"})

    def _build_cache_payload(self, ticker: str) -> dict[str, Any]:
        frame = load_ohlcv(ticker)
        frame = frame.sort_index()
        frame.index = pd.to_datetime(frame.index)

        if frame.empty:
            return {
                "ticker": ticker.upper(),
                "status": "no_ohlcv",
                "eligible": False,
                "history_bar_count": 0,
                "signal_count": 0,
            }

        if len(frame) < MIN_HISTORY_BARS:
            return {
                "ticker": ticker.upper(),
                "status": "insufficient_history",
                "eligible": False,
                "history_bar_count": len(frame),
                "signal_count": 0,
            }

        try:
            labeled = label_lifecycle_states(ticker, ohlcv=frame)
        except Exception as exc:
            return {
                "ticker": ticker.upper(),
                "status": "label_error",
                "eligible": False,
                "history_bar_count": len(frame),
                "signal_count": 0,
                "error": str(exc),
            }

        median_vol = float(frame["volume"].median()) if "volume" in frame else 0.0
        market_tier = _normalize_tier_value(self._tier_infer(frame, median_vol))
        adtv20 = (frame["close"] * frame["volume"]).rolling(ADTV_LOOKBACK_DAYS).mean()
        transitions = (labeled["state"] == "CONFIRMED") & (labeled["state"].shift(1) != "CONFIRMED")

        signals: list[dict[str, Any]] = []
        transition_dates = [pd.Timestamp(idx) for idx in labeled.index[transitions.fillna(False)]]
        for signal_date in transition_dates:
            if signal_date not in frame.index:
                continue
            signal_index = int(frame.index.get_loc(signal_date))
            entry_index = signal_index + 1
            time_exit_index = entry_index + TIME_EXIT_DAYS
            if entry_index >= len(frame) or time_exit_index >= len(frame):
                continue

            signal_close = _safe_float(frame.iloc[signal_index].get("close"))
            entry_open = _safe_float(frame.iloc[entry_index].get("open"))
            adtv_value = _safe_float(adtv20.iloc[signal_index])
            if signal_close <= 0 or entry_open <= 0 or adtv_value <= 0:
                continue

            signals.append(
                {
                    "signal_date": str(signal_date.date()),
                    "signal_index": signal_index,
                    "signal_close": round(signal_close, 6),
                    "entry_date": str(frame.index[entry_index].date()),
                    "entry_index": entry_index,
                    "time_exit_index": time_exit_index,
                    "adtv20_kwd": round(adtv_value, 6),
                    "liquidity_cap_kwd": round(adtv_value * LIQUIDITY_CAP_PCT, 6),
                }
            )

        bars = [
            {
                "date": str(idx.date()),
                "open": round(_safe_float(row.get("open")), 6),
                "high": round(_safe_float(row.get("high")), 6),
                "low": round(_safe_float(row.get("low")), 6),
                "close": round(_safe_float(row.get("close")), 6),
                "volume": round(_safe_float(row.get("volume")), 6),
            }
            for idx, row in frame.iterrows()
        ]

        return {
            "ticker": ticker.upper(),
            "status": "ok",
            "eligible": True,
            "market_tier": market_tier,
            "history_bar_count": len(frame),
            "signal_count": len(signals),
            "bars": bars,
            "confirmed_signals": signals,
        }

    def _load_cached_stocks(self, *, eligible_only: bool) -> dict[str, dict[str, Any]]:
        stocks: dict[str, dict[str, Any]] = {}
        for file_path in sorted(self.cache_dir.glob("*.json")):
            payload = _read_json(file_path)
            if eligible_only and not payload.get("eligible"):
                continue
            stocks[str(payload.get("ticker") or file_path.stem).upper()] = payload
        return stocks

    def _confirmed_trade_paths(self, stock_caches: dict[str, dict[str, Any]]) -> list[TradePath]:
        trade_paths: list[TradePath] = []
        for stock in stock_caches.values():
            trade_paths.extend(self._confirmed_trade_paths_for_stock(stock))
        trade_paths.sort(key=lambda item: (item.entry_date, item.ticker))
        return trade_paths

    def _confirmed_trade_paths_for_stock(self, stock: dict[str, Any]) -> list[TradePath]:
        market_tier = _normalize_tier_value(stock.get("market_tier"))
        bars = stock.get("bars") or []
        signals = stock.get("confirmed_signals") or []
        trade_paths: list[TradePath] = []
        for signal in signals:
            trade_path = self._build_trade_path(
                ticker=str(stock.get("ticker") or ""),
                market_tier=market_tier,
                bars=bars,
                signal_date=str(signal.get("signal_date") or ""),
                signal_index=int(signal.get("signal_index") or 0),
                signal_close=_safe_float(signal.get("signal_close")),
                entry_index=int(signal.get("entry_index") or 0),
                adtv20_kwd=_safe_float(signal.get("adtv20_kwd")),
                liquidity_cap_kwd=_safe_float(signal.get("liquidity_cap_kwd")),
            )
            if trade_path is not None:
                trade_paths.append(trade_path)
        return trade_paths

    def _random_trade_pool(self, stock_caches: dict[str, dict[str, Any]]) -> list[TradePath]:
        pool: list[TradePath] = []
        earliest_signal_index = max(MIN_BARS_FOR_LABELS - 1, ADTV_LOOKBACK_DAYS - 1)
        for stock in stock_caches.values():
            bars = stock.get("bars") or []
            if len(bars) < MIN_HISTORY_BARS:
                continue
            market_tier = _normalize_tier_value(stock.get("market_tier"))
            latest_signal_index = len(bars) - TIME_EXIT_DAYS - 2
            if latest_signal_index < earliest_signal_index:
                continue

            traded_values = [_safe_float(bar.get("close")) * _safe_float(bar.get("volume")) for bar in bars]
            for signal_index in range(earliest_signal_index, latest_signal_index + 1):
                window = traded_values[max(0, signal_index - ADTV_LOOKBACK_DAYS + 1): signal_index + 1]
                if len(window) < ADTV_LOOKBACK_DAYS:
                    continue
                adtv20_kwd = sum(window) / len(window)
                if adtv20_kwd <= 0:
                    continue
                trade_path = self._build_trade_path(
                    ticker=str(stock.get("ticker") or ""),
                    market_tier=market_tier,
                    bars=bars,
                    signal_date=str((bars[signal_index] or {}).get("date") or ""),
                    signal_index=signal_index,
                    signal_close=_safe_float((bars[signal_index] or {}).get("close")),
                    entry_index=signal_index + 1,
                    adtv20_kwd=adtv20_kwd,
                    liquidity_cap_kwd=adtv20_kwd * LIQUIDITY_CAP_PCT,
                )
                if trade_path is not None:
                    pool.append(trade_path)
        return pool

    def _build_trade_path(
        self,
        *,
        ticker: str,
        market_tier: str,
        bars: list[dict[str, Any]],
        signal_date: str,
        signal_index: int,
        signal_close: float,
        entry_index: int,
        adtv20_kwd: float,
        liquidity_cap_kwd: float,
    ) -> Optional[TradePath]:
        time_exit_index = entry_index + TIME_EXIT_DAYS
        if signal_index < 0 or entry_index <= signal_index or time_exit_index >= len(bars):
            return None

        entry_bar = bars[entry_index]
        entry_open = _safe_float(entry_bar.get("open"))
        if entry_open <= 0 or signal_close <= 0:
            return None

        stop_price = entry_open * (1.0 - STOP_LOSS_PCT)
        exit_index = time_exit_index
        exit_bar = bars[time_exit_index]
        exit_date = str(exit_bar.get("date") or "")
        exit_price = _safe_float(exit_bar.get("open"))
        exit_reason = "TIME_EXIT"
        if exit_price <= 0:
            return None

        for idx in range(entry_index, time_exit_index):
            low = _safe_float((bars[idx] or {}).get("low"))
            if low <= stop_price:
                exit_index = idx
                exit_date = str((bars[idx] or {}).get("date") or "")
                exit_price = stop_price
                exit_reason = "STOP_LOSS"
                break

        return TradePath(
            ticker=ticker.upper(),
            market_tier=_normalize_tier_value(market_tier),
            signal_date=signal_date,
            entry_date=str(entry_bar.get("date") or ""),
            entry_index=entry_index,
            entry_open=entry_open,
            signal_close=signal_close,
            exit_date=exit_date,
            exit_index=exit_index,
            exit_price=exit_price,
            exit_reason=exit_reason,
            liquidity_cap_kwd=liquidity_cap_kwd,
            adtv20_kwd=adtv20_kwd,
        )

    def _simulate(
        self,
        signals: list[TradePath],
        stock_caches: dict[str, dict[str, Any]],
        cost_scenario: CostScenario,
        *,
        scenario_name: str,
    ) -> SimulationResult:
        signals = sorted(signals, key=lambda item: (item.entry_date, item.ticker))
        calendar = self._collect_calendar(signals, stock_caches)
        if not calendar:
            empty_curve = pd.DataFrame([{"date": None, "equity_kwd": STARTING_CAPITAL_KWD, "drawdown_pct": 0.0}])
            return self._empty_result(scenario_name, len(signals), empty_curve)

        entries_by_date: dict[str, list[TradePath]] = {}
        time_exits_by_date: dict[str, list[TradePath]] = {}
        stop_exits_by_date: dict[str, list[TradePath]] = {}
        for signal in signals:
            entries_by_date.setdefault(signal.entry_date, []).append(signal)
            if signal.exit_reason == "TIME_EXIT":
                time_exits_by_date.setdefault(signal.exit_date, []).append(signal)
            else:
                stop_exits_by_date.setdefault(signal.exit_date, []).append(signal)

        cash = STARTING_CAPITAL_KWD
        peak_equity = STARTING_CAPITAL_KWD
        open_positions: dict[tuple[str, str], dict[str, Any]] = {}
        executed_trades: list[ExecutedTrade] = []
        skipped_signals: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []
        liquidity_shrink_count = 0
        liquidity_skip_count = 0
        tradeable_signal_count = 0

        close_lookup = self._close_lookup(stock_caches)

        for current_date in calendar:
            date_str = str(current_date.date())

            for trade in time_exits_by_date.get(date_str, []):
                key = (trade.ticker, trade.entry_date)
                position = open_positions.pop(key, None)
                if position is None:
                    continue
                executed = self._close_trade(position, trade, cost_scenario)
                executed_trades.append(executed)
                cash += executed.size_kwd + executed.gross_pnl_kwd - executed.exit_cost_kwd

            equity_before_entries = cash + self._mark_to_market(open_positions, close_lookup, date_str)
            requested_size_kwd = equity_before_entries * TARGET_POSITION_PCT

            for trade in entries_by_date.get(date_str, []):
                allowed_size_kwd = min(requested_size_kwd, trade.liquidity_cap_kwd)
                if trade.liquidity_cap_kwd >= MIN_POSITION_KWD:
                    tradeable_signal_count += 1
                if trade.liquidity_cap_kwd < MIN_POSITION_KWD:
                    liquidity_skip_count += 1
                    skipped_signals.append(
                        {
                            "ticker": trade.ticker,
                            "signal_date": trade.signal_date,
                            "entry_date": trade.entry_date,
                            "reason": "LIQUIDITY_CAP_TOO_SMALL",
                            "requested_size_kwd": round(requested_size_kwd, 4),
                            "allowed_size_kwd": round(trade.liquidity_cap_kwd, 4),
                            "adtv20_kwd": round(trade.adtv20_kwd, 4),
                        }
                    )
                    continue
                if allowed_size_kwd < requested_size_kwd:
                    liquidity_shrink_count += 1

                leg_cost_factor = self._leg_cost_factor(trade.market_tier, cost_scenario)
                affordable_size_kwd = cash / (1.0 + leg_cost_factor) if (1.0 + leg_cost_factor) > 0 else 0.0
                actual_size_kwd = min(allowed_size_kwd, affordable_size_kwd)
                if actual_size_kwd < MIN_POSITION_KWD:
                    skipped_signals.append(
                        {
                            "ticker": trade.ticker,
                            "signal_date": trade.signal_date,
                            "entry_date": trade.entry_date,
                            "reason": "INSUFFICIENT_CASH",
                            "requested_size_kwd": round(requested_size_kwd, 4),
                            "allowed_size_kwd": round(allowed_size_kwd, 4),
                            "cash_kwd": round(cash, 4),
                        }
                    )
                    continue

                entry_costs = self._leg_cost_breakdown(actual_size_kwd, trade.market_tier, cost_scenario)
                shares = actual_size_kwd / trade.entry_open
                cash -= actual_size_kwd + entry_costs["total_cost_kwd"]
                open_positions[(trade.ticker, trade.entry_date)] = {
                    "trade": trade,
                    "shares": shares,
                    "size_kwd": actual_size_kwd,
                    "requested_size_kwd": requested_size_kwd,
                    "allowed_size_kwd": allowed_size_kwd,
                    "entry_costs": entry_costs,
                    "commission_entry_kwd": entry_costs["commission_kwd"],
                    "slippage_entry_kwd": entry_costs["slippage_kwd"],
                    "shrunk_for_liquidity": allowed_size_kwd < requested_size_kwd,
                }

            for trade in stop_exits_by_date.get(date_str, []):
                key = (trade.ticker, trade.entry_date)
                position = open_positions.pop(key, None)
                if position is None:
                    continue
                executed = self._close_trade(position, trade, cost_scenario)
                executed_trades.append(executed)
                cash += executed.size_kwd + executed.gross_pnl_kwd - executed.exit_cost_kwd

            equity = cash + self._mark_to_market(open_positions, close_lookup, date_str)
            peak_equity = max(peak_equity, equity)
            drawdown_pct = ((equity - peak_equity) / peak_equity * 100.0) if peak_equity > 0 else 0.0
            equity_rows.append({"date": date_str, "equity_kwd": round(equity, 4), "drawdown_pct": round(drawdown_pct, 4)})

        equity_curve = pd.DataFrame(equity_rows)
        return self._summarize_result(
            scenario_name=scenario_name,
            executed_trades=executed_trades,
            skipped_signals=skipped_signals,
            raw_signal_count=len(signals),
            tradeable_signal_count=tradeable_signal_count,
            liquidity_shrink_count=liquidity_shrink_count,
            liquidity_skip_count=liquidity_skip_count,
            equity_curve=equity_curve,
        )

    def _close_trade(self, position: dict[str, Any], trade: TradePath, cost_scenario: CostScenario) -> ExecutedTrade:
        gross_exit_value = position["shares"] * trade.exit_price
        exit_costs = self._leg_cost_breakdown(gross_exit_value, trade.market_tier, cost_scenario)
        gross_pnl = gross_exit_value - position["size_kwd"]
        net_pnl = gross_pnl - position["entry_costs"]["total_cost_kwd"] - exit_costs["total_cost_kwd"]
        net_return_pct = (net_pnl / position["size_kwd"] * 100.0) if position["size_kwd"] > 0 else 0.0
        holding_days = max(1, trade.exit_index - trade.entry_index + 1)
        return ExecutedTrade(
            ticker=trade.ticker,
            market_tier=trade.market_tier,
            signal_date=trade.signal_date,
            entry_date=trade.entry_date,
            exit_date=trade.exit_date,
            entry_price=trade.entry_open,
            exit_price=trade.exit_price,
            exit_reason=trade.exit_reason,
            holding_days=holding_days,
            requested_size_kwd=position["requested_size_kwd"],
            allowed_size_kwd=position["allowed_size_kwd"],
            size_kwd=position["size_kwd"],
            shrunk_for_liquidity=bool(position["shrunk_for_liquidity"]),
            skipped_for_liquidity=False,
            entry_cost_kwd=position["entry_costs"]["total_cost_kwd"],
            exit_cost_kwd=exit_costs["total_cost_kwd"],
            total_cost_kwd=position["entry_costs"]["total_cost_kwd"] + exit_costs["total_cost_kwd"],
            commission_kwd=position["commission_entry_kwd"] + exit_costs["commission_kwd"],
            slippage_kwd=position["slippage_entry_kwd"] + exit_costs["slippage_kwd"],
            gross_pnl_kwd=gross_pnl,
            net_pnl_kwd=net_pnl,
            net_return_pct=net_return_pct,
        )

    def _close_lookup(self, stock_caches: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
        lookup: dict[str, dict[str, float]] = {}
        for ticker, stock in stock_caches.items():
            lookup[ticker] = {str(bar.get("date") or ""): _safe_float(bar.get("close")) for bar in (stock.get("bars") or [])}
        return lookup

    def _mark_to_market(
        self,
        open_positions: dict[tuple[str, str], dict[str, Any]],
        close_lookup: dict[str, dict[str, float]],
        date_str: str,
    ) -> float:
        total = 0.0
        for position in open_positions.values():
            trade = position["trade"]
            close_price = _safe_float(close_lookup.get(trade.ticker, {}).get(date_str))
            if close_price <= 0:
                close_price = trade.entry_open
            total += position["shares"] * close_price
        return total

    def _summarize_result(
        self,
        *,
        scenario_name: str,
        executed_trades: list[ExecutedTrade],
        skipped_signals: list[dict[str, Any]],
        raw_signal_count: int,
        tradeable_signal_count: int,
        liquidity_shrink_count: int,
        liquidity_skip_count: int,
        equity_curve: pd.DataFrame,
    ) -> SimulationResult:
        if equity_curve.empty:
            return self._empty_result(scenario_name, raw_signal_count, pd.DataFrame())

        returns = [trade.net_return_pct for trade in executed_trades]
        wins = [value for value in returns if value > 0]
        losses = [value for value in returns if value <= 0]
        expectancy_pct = mean(returns) if returns else 0.0
        expectancy_kwd = mean([trade.net_pnl_kwd for trade in executed_trades]) if executed_trades else 0.0
        equity_end = float(equity_curve["equity_kwd"].iloc[-1])
        max_drawdown = float(equity_curve["drawdown_pct"].min()) if not equity_curve.empty else 0.0

        cagr = None
        start_value = equity_curve["date"].iloc[0]
        end_value = equity_curve["date"].iloc[-1]
        if start_value is not None and end_value is not None:
            start_date = pd.Timestamp(start_value)
            end_date = pd.Timestamp(end_value)
            years = max((end_date - start_date).days / 365.25, 0.0)
            if years >= 1e-6 and equity_end > 0 and STARTING_CAPITAL_KWD > 0:
                cagr = ((equity_end / STARTING_CAPITAL_KWD) ** (1.0 / years) - 1.0) * 100.0

        pnl_by_ticker: dict[str, float] = {}
        for trade in executed_trades:
            pnl_by_ticker[trade.ticker] = pnl_by_ticker.get(trade.ticker, 0.0) + trade.net_pnl_kwd

        buckets = {
            "<=-10%": sum(1 for value in returns if value <= -10.0),
            "-10% to -5%": sum(1 for value in returns if -10.0 < value <= -5.0),
            "-5% to 0%": sum(1 for value in returns if -5.0 < value <= 0.0),
            "0% to 5%": sum(1 for value in returns if 0.0 < value <= 5.0),
            ">5%": sum(1 for value in returns if value > 5.0),
        }

        win_loss_ratio = None
        if wins and losses and abs(mean(losses)) > 1e-9:
            win_loss_ratio = abs(mean(wins) / mean(losses))

        return SimulationResult(
            scenario_name=scenario_name,
            executed_trades=executed_trades,
            skipped_signals=skipped_signals,
            raw_signal_count=raw_signal_count,
            tradeable_signal_count=tradeable_signal_count,
            liquidity_shrink_count=liquidity_shrink_count,
            liquidity_skip_count=liquidity_skip_count,
            equity_curve=equity_curve,
            equity_end_kwd=equity_end,
            total_net_pnl_kwd=sum(trade.net_pnl_kwd for trade in executed_trades),
            max_drawdown_pct=max_drawdown,
            expectancy_pct=expectancy_pct,
            expectancy_kwd=expectancy_kwd,
            median_trade_return_pct=median(returns) if returns else 0.0,
            mean_trade_return_pct=mean(returns) if returns else 0.0,
            win_rate_pct=(len(wins) / len(returns) * 100.0) if returns else 0.0,
            avg_win_pct=mean(wins) if wins else 0.0,
            avg_loss_pct=mean(losses) if losses else 0.0,
            win_loss_ratio=win_loss_ratio,
            tradeable_pct=(tradeable_signal_count / raw_signal_count * 100.0) if raw_signal_count else 0.0,
            cagr_pct=cagr,
            big_loser_count=sum(1 for value in returns if value <= -8.0),
            outcome_buckets=buckets,
            pnl_by_ticker=pnl_by_ticker,
        )

    def _empty_result(self, scenario_name: str, raw_signal_count: int, equity_curve: pd.DataFrame) -> SimulationResult:
        return SimulationResult(
            scenario_name=scenario_name,
            executed_trades=[],
            skipped_signals=[],
            raw_signal_count=raw_signal_count,
            tradeable_signal_count=0,
            liquidity_shrink_count=0,
            liquidity_skip_count=0,
            equity_curve=equity_curve,
            equity_end_kwd=STARTING_CAPITAL_KWD,
            total_net_pnl_kwd=0.0,
            max_drawdown_pct=0.0,
            expectancy_pct=0.0,
            expectancy_kwd=0.0,
            median_trade_return_pct=0.0,
            mean_trade_return_pct=0.0,
            win_rate_pct=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            win_loss_ratio=None,
            tradeable_pct=0.0,
            cagr_pct=None,
            big_loser_count=0,
            outcome_buckets={"<=-10%": 0, "-10% to -5%": 0, "-5% to 0%": 0, "0% to 5%": 0, ">5%": 0},
            pnl_by_ticker={},
        )

    def _random_baseline(
        self,
        stock_caches: dict[str, dict[str, Any]],
        actual_signals: list[TradePath],
        cost_scenario: CostScenario,
    ) -> dict[str, Any]:
        pool = self._random_trade_pool(stock_caches)
        if not pool:
            return {
                "method": "entry-date matched random entry baseline",
                "runs": 0,
                "sample_size": 0,
                "matched_trade_count": 0,
                "month_fallback_count": 0,
                "shortfall_count": 0,
                "mean_expectancy_pct": 0.0,
                "median_expectancy_pct": 0.0,
                "mean_total_net_pnl_kwd": 0.0,
                "median_total_net_pnl_kwd": 0.0,
                "pct_positive_runs": 0.0,
            }

        actual_trade_keys = {(signal.ticker, signal.entry_date) for signal in actual_signals}
        actual_date_counts = Counter(signal.entry_date for signal in actual_signals)
        pool_by_date: dict[str, list[TradePath]] = defaultdict(list)
        pool_by_month: dict[str, list[TradePath]] = defaultdict(list)
        for trade in pool:
            if (trade.ticker, trade.entry_date) in actual_trade_keys:
                continue
            pool_by_date[trade.entry_date].append(trade)
            pool_by_month[trade.entry_date[:7]].append(trade)

        sample_size = min(RANDOM_BASELINE_MAX_TRADES, max(1, len(actual_signals)))
        rng = np.random.default_rng(RANDOM_SEED)
        run_summaries: list[dict[str, float]] = []
        matched_trade_counts: list[int] = []
        month_fallback_counts: list[int] = []
        shortfall_counts: list[int] = []
        for _ in range(RANDOM_BASELINE_RUNS):
            sampled: list[TradePath] = []
            used_keys: set[tuple[str, str]] = set()
            month_fallback_count = 0
            shortfall_count = 0

            for entry_date, count in sorted(actual_date_counts.items()):
                date_candidates = [
                    trade
                    for trade in pool_by_date.get(entry_date, [])
                    if (trade.ticker, trade.entry_date) not in used_keys
                ]
                take_from_date = min(count, len(date_candidates))
                if take_from_date > 0:
                    indices = rng.choice(len(date_candidates), size=take_from_date, replace=False)
                    chosen = [date_candidates[int(index)] for index in indices]
                    sampled.extend(chosen)
                    used_keys.update((trade.ticker, trade.entry_date) for trade in chosen)

                remaining = count - take_from_date
                if remaining <= 0:
                    continue

                month_candidates = [
                    trade
                    for trade in pool_by_month.get(entry_date[:7], [])
                    if trade.entry_date != entry_date and (trade.ticker, trade.entry_date) not in used_keys
                ]
                take_from_month = min(remaining, len(month_candidates))
                if take_from_month > 0:
                    indices = rng.choice(len(month_candidates), size=take_from_month, replace=False)
                    chosen = [month_candidates[int(index)] for index in indices]
                    sampled.extend(chosen)
                    used_keys.update((trade.ticker, trade.entry_date) for trade in chosen)
                    month_fallback_count += take_from_month

                shortfall_count += remaining - take_from_month

            sampled = sorted(sampled, key=lambda trade: (trade.entry_date, trade.ticker))[:sample_size]
            result = self._simulate(sampled, stock_caches, cost_scenario, scenario_name="RANDOM_BASELINE")
            run_summaries.append({
                "expectancy_pct": result.expectancy_pct,
                "total_net_pnl_kwd": result.total_net_pnl_kwd,
            })
            matched_trade_counts.append(len(sampled))
            month_fallback_counts.append(month_fallback_count)
            shortfall_counts.append(shortfall_count)

        expectancy_values = [row["expectancy_pct"] for row in run_summaries]
        total_pnl_values = [row["total_net_pnl_kwd"] for row in run_summaries]
        return {
            "method": "entry-date matched random entry baseline",
            "runs": RANDOM_BASELINE_RUNS,
            "sample_size": sample_size,
            "matched_trade_count": min(matched_trade_counts) if matched_trade_counts else 0,
            "month_fallback_count": int(round(mean(month_fallback_counts))) if month_fallback_counts else 0,
            "shortfall_count": int(round(mean(shortfall_counts))) if shortfall_counts else 0,
            "mean_expectancy_pct": mean(expectancy_values) if expectancy_values else 0.0,
            "median_expectancy_pct": median(expectancy_values) if expectancy_values else 0.0,
            "mean_total_net_pnl_kwd": mean(total_pnl_values) if total_pnl_values else 0.0,
            "median_total_net_pnl_kwd": median(total_pnl_values) if total_pnl_values else 0.0,
            "pct_positive_runs": (sum(1 for value in total_pnl_values if value > 0) / len(total_pnl_values) * 100.0) if total_pnl_values else 0.0,
        }

    def _index_baseline(self, result: SimulationResult) -> dict[str, Any]:
        if not self.index_proxy_path.exists() or result.equity_curve.empty:
            return {"available": False, "reason": "KSE proxy CSV not found or no trade span."}

        frame = pd.read_csv(self.index_proxy_path)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date")

        start_value = result.equity_curve["date"].iloc[0]
        end_value = result.equity_curve["date"].iloc[-1]
        if start_value is None or end_value is None:
            return {"available": False, "reason": "No valid trade dates."}

        start_date = pd.Timestamp(start_value)
        end_date = pd.Timestamp(end_value)
        start_rows = frame.loc[frame["date"] <= start_date]
        end_rows = frame.loc[frame["date"] <= end_date]
        if start_rows.empty or end_rows.empty:
            return {"available": False, "reason": "KSE proxy CSV does not cover the trade span."}

        start_level = float(start_rows.iloc[-1]["kse_index"])
        end_level = float(end_rows.iloc[-1]["kse_index"])
        hold_return_pct = ((end_level / start_level) - 1.0) * 100.0 if start_level > 0 else 0.0
        years = max((end_date - start_date).days / 365.25, 0.0)
        cagr = None
        if years > 1e-6 and start_level > 0 and end_level > 0:
            cagr = ((end_level / start_level) ** (1.0 / years) - 1.0) * 100.0
        return {
            "available": True,
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "start_level": start_level,
            "end_level": end_level,
            "hold_return_pct": hold_return_pct,
            "cagr_pct": cagr,
        }

    def _best_ticker(self, result: SimulationResult) -> Optional[str]:
        if not result.pnl_by_ticker:
            return None
        return max(result.pnl_by_ticker.items(), key=lambda item: item[1])[0]

    def _collect_calendar(self, signals: list[TradePath], stock_caches: dict[str, dict[str, Any]]) -> list[pd.Timestamp]:
        dates: set[pd.Timestamp] = set()
        for signal in signals:
            bars = (stock_caches.get(signal.ticker) or {}).get("bars") or []
            if not bars:
                continue
            for bar in bars[signal.entry_index: signal.exit_index + 1]:
                date_value = bar.get("date")
                if date_value:
                    dates.add(pd.Timestamp(str(date_value)))
        return sorted(dates)

    def _leg_cost_factor(self, tier: str, scenario: CostScenario) -> float:
        round_trip = _total_cost_factor(_normalize_tier_value(tier))
        base_leg = round_trip / 2.0
        base_slippage = max(0.0, base_leg - TC_COMMISSION)
        stressed_leg = TC_COMMISSION + base_slippage * scenario.slippage_multiplier
        return stressed_leg

    def _leg_cost_breakdown(self, notional_kwd: float, tier: str, scenario: CostScenario) -> dict[str, float]:
        commission_kwd = notional_kwd * TC_COMMISSION
        slippage_rate = max(0.0, self._leg_cost_factor(tier, scenario) - TC_COMMISSION)
        slippage_kwd = notional_kwd * slippage_rate
        return {
            "commission_kwd": commission_kwd,
            "slippage_kwd": slippage_kwd,
            "total_cost_kwd": commission_kwd + slippage_kwd,
        }

    def _render_report(
        self,
        *,
        stock_caches: dict[str, dict[str, Any]],
        actual_moderate: SimulationResult,
        actual_harsh: SimulationResult,
        exclude_best: SimulationResult,
        best_ticker: Optional[str],
        random_baseline: dict[str, Any],
        index_baseline: dict[str, Any],
    ) -> str:
        sample_trades = actual_moderate.executed_trades[:10]
        eligible_universe_size = len(stock_caches)
        verdict = self._verdict(actual_moderate, actual_harsh, exclude_best, random_baseline)

        lines = [
            "# Lifecycle L3 Backtest - CONFIRMED Entry",
            "",
            "Generated: 2026-05-23",
            "",
            "## 1. Exact Rules Used",
            "",
            "- Entry signal: transition into `state == CONFIRMED` from the default lifecycle labeler config, with no retuning.",
            "- Signal timing: signal on day D, entry at day D+1 open from OHLCV. No signal-day fills.",
            f"- Stop policy: fixed {STOP_LOSS_PCT * 100:.1f}% stop from entry. This keeps the exit explicit and avoids introducing another adaptive parameter while testing the entry.",
            f"- Time exit: next available open after {TIME_EXIT_DAYS} full trading days if the stop was not hit first.",
            "- Stop execution rule: if a post-entry bar traded through the stop (`low <= stop`), the exit filled at the stop price on that bar.",
            "- Cost model (moderate): Premier 0.15% commission + 0.10% slippage each leg; Main 0.15% + 0.30% each leg. Reused from the existing backtester.",
            "- Cost model (harsh): same commission, doubled slippage.",
            "- Random baseline: for each actual entry date, choose a random non-signal trade path from the same date when possible, with same-month fallback only if a date is undersupplied.",
            f"- Liquidity cap: {LIQUIDITY_CAP_PCT * 100:.0f}% of rolling {ADTV_LOOKBACK_DAYS}-day traded value (`close * volume`). Shrink above cap; skip below {MIN_POSITION_KWD:.0f} KWD.",
            f"- Portfolio model: starting equity {STARTING_CAPITAL_KWD:,.0f} KWD, target {TARGET_POSITION_PCT * 100:.0f}% of current equity per trade, no leverage.",
            "- Universe: all cached tickers with at least 250 bars. This remains survivor-biased because it depends on the current OHLCV cache.",
            "",
            "## 2. Tradeable Signal Count",
            "",
            f"- Eligible universe size: {eligible_universe_size} tickers.",
            f"- Raw CONFIRMED transitions: {actual_moderate.raw_signal_count}.",
            f"- Tradeable after liquidity: {actual_moderate.tradeable_signal_count} ({actual_moderate.tradeable_pct:.1f}%).",
            f"- Liquidity shrinks: {actual_moderate.liquidity_shrink_count}.",
            f"- Liquidity skips: {actual_moderate.liquidity_skip_count}.",
            "",
            "## 3. Full Real Metrics",
            "",
            self._metrics_block(actual_moderate),
            "",
            "### Outcome Distribution",
            "",
            *[f"- {bucket}: {count}" for bucket, count in actual_moderate.outcome_buckets.items()],
            f"- Big losers (<= -8% net): {actual_moderate.big_loser_count}",
            "",
            "### Sample Trades",
            "",
        ]

        if sample_trades:
            for trade in sample_trades:
                lines.append(
                    "- "
                    f"{trade.ticker}: signal {trade.signal_date}, entry {trade.entry_date} @ {trade.entry_price:.3f}, "
                    f"exit {trade.exit_date} @ {trade.exit_price:.3f}, {trade.exit_reason}, "
                    f"net {trade.net_return_pct:.2f}%, costs {trade.total_cost_kwd:.2f} KWD"
                )
        else:
            lines.append("- No executed trades.")

        lines.extend([
            "",
            "## 4. Baseline Comparison",
            "",
            f"- {random_baseline['method'].capitalize()} ({random_baseline['runs']} runs of {random_baseline['matched_trade_count']} trades): mean expectancy {random_baseline['mean_expectancy_pct']:.3f}% per trade, median expectancy {random_baseline['median_expectancy_pct']:.3f}%, positive total-P&L runs {random_baseline['pct_positive_runs']:.1f}%, average same-month fallbacks {random_baseline['month_fallback_count']}, average shortfall {random_baseline['shortfall_count']}.",
        ])
        if index_baseline.get("available"):
            lines.append(
                f"- KSE proxy buy-and-hold over {index_baseline['start_date']} to {index_baseline['end_date']}: {index_baseline['hold_return_pct']:.2f}% return"
                + (f", CAGR {index_baseline['cagr_pct']:.2f}%" if index_baseline.get("cagr_pct") is not None else "")
                + "."
            )
        else:
            lines.append(f"- Index baseline unavailable: {index_baseline.get('reason', 'unknown')}")

        lines.extend([
            "",
            "## 5. Sensitivity Checks",
            "",
            "### Cost Stress",
            "",
            self._metrics_block(actual_harsh),
            "",
            "### Exclude Best Stock",
            "",
            (f"- Best stock by net P&L in the moderate run: {best_ticker}" if best_ticker else "- Best stock could not be determined."),
            self._metrics_block(exclude_best),
            "",
            "## 6. Verdict",
            "",
            verdict,
            "",
            "## Limitations",
            "",
            "- The universe comes from the current OHLCV cache, so this remains survivor-biased.",
            "- Market tier uses the existing repo heuristic based on median daily volume because there is no full historical exchange-tier table in the local DB.",
            "- The index comparison uses a monthly proxy CSV because the KSE index series is absent from `ee_ohlcv_cache`.",
            "- If trade count is small, treat any apparent edge as weak evidence rather than proof.",
        ])
        return "\n".join(lines) + "\n"

    def _metrics_block(self, result: SimulationResult) -> str:
        trade_count = len(result.executed_trades)
        return "\n".join([
            f"- Scenario: {result.scenario_name}",
            f"- Executed trades: {trade_count}",
            f"- Win rate: {result.win_rate_pct:.2f}%",
            f"- Avg win: {result.avg_win_pct:.2f}%",
            f"- Avg loss: {result.avg_loss_pct:.2f}%",
            f"- Win/loss ratio: {result.win_loss_ratio:.3f}" if result.win_loss_ratio is not None else "- Win/loss ratio: n/a",
            f"- Expectancy: {result.expectancy_pct:.3f}% per trade ({result.expectancy_kwd:.2f} KWD)",
            f"- Mean net return: {result.mean_trade_return_pct:.3f}%",
            f"- Median net return: {result.median_trade_return_pct:.3f}%",
            f"- Total net P&L: {result.total_net_pnl_kwd:.2f} KWD",
            f"- Equity start/end: {STARTING_CAPITAL_KWD:.2f} -> {result.equity_end_kwd:.2f} KWD",
            f"- Max drawdown: {result.max_drawdown_pct:.2f}%",
            (f"- CAGR: {result.cagr_pct:.2f}%" if result.cagr_pct is not None else "- CAGR: n/a"),
        ])

    def _verdict(
        self,
        actual_moderate: SimulationResult,
        actual_harsh: SimulationResult,
        exclude_best: SimulationResult,
        random_baseline: dict[str, Any],
    ) -> str:
        trade_count = len(actual_moderate.executed_trades)
        beats_random = actual_moderate.expectancy_pct > float(random_baseline.get("mean_expectancy_pct") or 0.0)
        holds_up_to_costs = actual_harsh.expectancy_pct > 0.0 and actual_harsh.total_net_pnl_kwd > 0.0
        survives_ex_best = exclude_best.expectancy_pct > 0.0 and exclude_best.total_net_pnl_kwd > 0.0

        if trade_count < 30:
            return (
                f"The full-universe CONFIRMED backtest only produced {trade_count} executed trades, which is too small for statistical confidence. "
                f"Treat the result as directional only. Moderate-friction expectancy was {actual_moderate.expectancy_pct:.3f}% per trade with max drawdown {actual_moderate.max_drawdown_pct:.2f}%. "
                f"Compared with the random baseline, the edge {'did' if beats_random else 'did not'} clear the timing hurdle. "
                f"Under harsh costs the expectancy was {actual_harsh.expectancy_pct:.3f}%, and excluding the best stock produced {exclude_best.expectancy_pct:.3f}%. "
                f"The honest conclusion is that the sample is too thin to claim a robust strategy."
            )

        if actual_moderate.expectancy_pct <= 0.0:
            return (
                f"After realistic frictions, trading CONFIRMED would not have made money in this backtest. Expectancy was {actual_moderate.expectancy_pct:.3f}% per trade, "
                f"and the signal should be treated as unproven."
            )

        if not (beats_random and holds_up_to_costs and survives_ex_best):
            return (
                f"CONFIRMED shows a positive moderate-cost expectancy ({actual_moderate.expectancy_pct:.3f}% per trade), but the edge looks fragile. "
                f"It {'does' if beats_random else 'does not'} beat the random-entry baseline, "
                f"it {'does' if holds_up_to_costs else 'does not'} stay positive under harsh costs, and "
                f"it {'does' if survives_ex_best else 'does not'} survive removing the best stock. "
                f"That is not strong enough to call the edge robust."
            )

        return (
            f"On this historical price-only test, CONFIRMED remained profitable after realistic frictions and cleared the random-entry hurdle, "
            f"with expectancy {actual_moderate.expectancy_pct:.3f}% per trade and max drawdown {actual_moderate.max_drawdown_pct:.2f}%. "
            f"It also stayed positive under harsh costs and without the best stock, so the edge looks more robust than a one-name artifact."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Laptop-safe staged backtest for CONFIRMED lifecycle entries.")
    parser.add_argument(
        "stage",
        nargs="?",
        default="status",
        choices=["stage-a", "stage-b", "stage-c", "run-all", "status"],
        help="Which stage to run.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild existing Stage A cache files.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=STAGE_A_SLEEP_SECONDS,
        help="Sleep inserted between Stage A stocks to keep CPU usage down.",
    )
    args = parser.parse_args()

    backtester = ConfirmedBacktester()
    if args.stage == "stage-a":
        result = backtester.stage_a_label_once(force=args.force, sleep_seconds=args.sleep_seconds)
    elif args.stage == "stage-b":
        result = backtester.stage_b_simulate_from_cache()
    elif args.stage == "stage-c":
        result = backtester.stage_c_baselines_and_sensitivity()
    elif args.stage == "run-all":
        result = backtester.run_all(force_stage_a=args.force, sleep_seconds=args.sleep_seconds)
    else:
        result = backtester.cache_status()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
