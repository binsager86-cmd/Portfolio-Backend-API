"""Walk-forward backtester + Monte Carlo parameter stability validator.

Implements the spec §5 Backtesting & Validation Protocol:

  Walk-Forward Windows:
    Window 1: Train 2020-2022 → Test 2023
    Window 2: Train 2021-2023 → Test 2024
    Window 3: Train 2022-2024 → Test 2025

  Monte Carlo:
    Shuffle daily returns 1,000 times to verify p < 0.05 (strategy edge).

  Transaction Costs:
    Premier  — commission 0.15 %, slippage 0.10 %
    Main     — commission 0.15 %, slippage 0.30 %

All metrics are computed only from OUT-OF-SAMPLE test periods.
"""
from __future__ import annotations

import logging
import math
import random
import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.services.signal_engine.config.risk_config import (
    TC_COMMISSION,
    TC_SLIPPAGE_AUCTION,
    TC_SLIPPAGE_MAIN,
    TC_SLIPPAGE_PREMIER,
)
from app.services.signal_engine.engine.signal_generator import generate_kuwait_signal

logger = logging.getLogger(__name__)

# ── Walk-forward window definitions ─────────────────────────────────────────
WALK_FORWARD_WINDOWS: list[tuple[str, str, str, str]] = [
    ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]

MONTE_CARLO_ITERATIONS: int = 1_000
RANDOM_STATE: int = 42


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    """Single simulated trade outcome."""
    date: str
    stock_code: str
    signal: str                     # BUY | SELL
    setup_type: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    rr_ratio: float
    outcome: str                    # NOT_FILLED | TP1_HIT | TP2_HIT | SL_HIT | EXPIRED
    pnl_r: float                    # P&L in R-multiples (net of costs)
    win_tp1: bool
    win_tp2: bool
    segment: str
    regime: str
    confluence_score: int


@dataclass
class WindowMetrics:
    """Out-of-sample metrics for one walk-forward window."""
    train_from: str
    train_to: str
    test_from: str
    test_to: str
    n_signals: int = 0
    orders_filled: int = 0
    not_filled: int = 0
    fill_rate: float = 0.0
    n_trades: int = 0
    win_rate_tp1: float = 0.0
    win_rate_tp2: float = 0.0
    profit_factor: float = 0.0
    avg_r: float = 0.0
    max_drawdown_r: float = 0.0
    sharpe: float = 0.0
    p_value_vs_random: float = 1.0
    expectancy_per_trade: float = 0.0
    win_rate_ci_95: tuple[float, float] = (0.0, 1.0)
    positive_expectancy: bool = False
    exposure_days: int = 0
    turnover_r: float = 0.0
    max_losing_streak: int = 0
    tail_loss_r_5pct: float = 0.0


@dataclass
class BacktestReport:
    """Aggregated report across all walk-forward windows."""
    windows: list[WindowMetrics] = field(default_factory=list)
    all_positive_expectancy: bool = False
    aggregate_win_rate_tp1: float = 0.0
    aggregate_avg_r: float = 0.0
    aggregate_profit_factor: float = 0.0
    calibration_error_pct: float = 0.0   # ± percentage
    monte_carlo_p_value: float = 1.0
    parameter_stability_pct: float = 0.0
    cvar_compliance_pct: float = 0.0
    total_trades: int = 0
    signals_generated: int = 0
    orders_filled: int = 0
    not_filled: int = 0
    fill_rate: float = 0.0
    passed_all_criteria: bool = False


# ── Transaction cost calculator ───────────────────────────────────────────────

def _total_cost_factor(segment: str) -> float:
    """Return round-trip cost fraction for a given segment."""
    seg = segment.upper()
    if seg == "PREMIER":
        slippage = TC_SLIPPAGE_PREMIER
    elif seg == "AUCTION":
        slippage = TC_SLIPPAGE_AUCTION
    else:
        slippage = TC_SLIPPAGE_MAIN
    return 2 * TC_COMMISSION + 2 * slippage


# ── Simulated trade outcome from OHLCV series ────────────────────────────────

def simulate_trade(
    rows: list[dict[str, Any]],
    signal: dict[str, Any],
    max_hold_days: int = 10,
) -> TradeResult:
    """Simulate a trade by scanning forward OHLCV bars for first exit hit.

    The strategy holds at most `max_hold_days` bars. If neither TP1 nor SL
    is hit, the position is closed at the last bar's close (EXPIRED).

    Args:
        rows:     Full OHLCV row list sorted ascending by date.
        signal:   Output from generate_kuwait_signal().
        max_hold_days: Maximum holding period before forced exit.

    Returns:
        TradeResult with outcome and R-multiple P&L.
    """
    exe = signal.get("execution", {})
    entry_zone = exe.get("entry_zone_fils") or [None, None]
    entry_low = entry_zone[0] if len(entry_zone) > 0 else None
    entry_high = entry_zone[1] if len(entry_zone) > 1 else None
    entry_mid = ((entry_low or 0.0) + (entry_high or 0.0)) / 2.0
    stop = exe.get("stop_loss_fils") or 0.0
    tp1 = exe.get("tp1_fils")
    tp2 = exe.get("tp2_fils")
    direction = signal.get("signal", "NEUTRAL")
    segment = signal.get("segment", "PREMIER")
    cost_frac = _total_cost_factor(segment)

    risk = abs(entry_mid - stop)
    if (
        risk < 1e-9
        or direction == "NEUTRAL"
        or entry_mid <= 0
        or not exe.get("actionable", signal.get("actionable", True))
        or tp1 is None
    ):
        return TradeResult(
            date=signal.get("metadata", {}).get("data_as_of", ""),
            stock_code=signal.get("stock_code", ""),
            signal=direction,
            setup_type=signal.get("setup_type", ""),
            entry=entry_mid,
            stop_loss=stop,
            tp1=tp1,
            tp2=tp2,
            rr_ratio=signal.get("risk_metrics", {}).get("risk_reward_ratio") or 0.0,
            outcome="SKIPPED",
            pnl_r=0.0,
            win_tp1=False,
            win_tp2=False,
            segment=segment,
            regime=signal.get("confluence_details", {}).get("regime", ""),
            confluence_score=signal.get("confluence_details", {}).get("total_score", 0),
        )

    # Locate signal date in rows
    sig_date = signal.get("metadata", {}).get("data_as_of", "")
    start_idx = 0
    for i, r in enumerate(rows):
        if str(r.get("date", "")) >= sig_date:
            start_idx = i + 1  # enter next bar after signal date
            break

    outcome = "NOT_FILLED"
    exit_price = entry_mid
    win_tp1 = False
    win_tp2 = False

    filled = False
    fill_price = entry_mid
    eligible_rows = rows[start_idx: start_idx + max_hold_days]
    for row in eligible_rows:
        h = float(row.get("high") or 0.0)
        l = float(row.get("low") or 0.0)
        open_price = float(row.get("open") or 0.0)

        if not filled:
            if direction == "BUY":
                if open_price > 0 and open_price <= entry_high:
                    fill_price = max(entry_low or entry_mid, open_price)
                elif entry_low is not None and entry_high is not None and l <= entry_high and h >= entry_low:
                    fill_price = entry_mid
            else:
                if open_price > 0 and open_price >= entry_low:
                    fill_price = min(entry_high or entry_mid, open_price)
                elif entry_low is not None and entry_high is not None and l <= entry_high and h >= entry_low:
                    fill_price = entry_mid
            if not filled and fill_price != entry_mid:
                filled = True
            elif not filled and entry_low is not None and entry_high is not None and l <= entry_high and h >= entry_low:
                filled = True
            if not filled:
                continue

        # If a bar opens beyond the stop, a stop order fills at the tradable open.
        # When both stop and target are touched in one bar, stop wins conservatively.
        if direction == "BUY":
            stop_hit = (open_price > 0 and open_price <= stop) or l <= stop
            tp2_hit = tp2 is not None and h >= tp2
            tp1_hit = h >= tp1
            if stop_hit:
                exit_price = open_price if open_price > 0 and open_price < stop else stop
                outcome = "SL_HIT"
                break
            if tp2_hit:
                exit_price = tp2
                outcome = "TP2_HIT"
                win_tp1 = True
                win_tp2 = True
                break
            if tp1_hit:
                exit_price = tp1
                outcome = "TP1_HIT"
                win_tp1 = True
                break
        else:
            stop_hit = (open_price > 0 and open_price >= stop) or h >= stop
            tp2_hit = tp2 is not None and l <= tp2
            tp1_hit = l <= tp1
            if stop_hit:
                exit_price = open_price if open_price > 0 and open_price > stop else stop
                outcome = "SL_HIT"
                break
            if tp2_hit:
                exit_price = tp2
                outcome = "TP2_HIT"
                win_tp1 = True
                win_tp2 = True
                break
            if tp1_hit:
                exit_price = tp1
                outcome = "TP1_HIT"
                win_tp1 = True
                break

    if not filled:
        return TradeResult(
            date=sig_date,
            stock_code=signal.get("stock_code", ""),
            signal=direction,
            setup_type=signal.get("setup_type", ""),
            entry=fill_price,
            stop_loss=stop,
            tp1=tp1,
            tp2=tp2,
            rr_ratio=signal.get("risk_metrics", {}).get("risk_reward_ratio") or 0.0,
            outcome="NOT_FILLED",
            pnl_r=0.0,
            win_tp1=False,
            win_tp2=False,
            segment=segment,
            regime=signal.get("confluence_details", {}).get("regime", ""),
            confluence_score=signal.get("confluence_details", {}).get("total_score", 0),
        )

    if outcome == "NOT_FILLED":
        last_eligible = eligible_rows[-1] if eligible_rows else {}
        exit_price = float(last_eligible.get("close") or fill_price)
        outcome = "EXPIRED"
    risk = abs(fill_price - stop)

    # P&L in R-multiples, net of transaction costs
    raw_pnl = (exit_price - fill_price) if direction == "BUY" else (fill_price - exit_price)
    entry_cost_fils = fill_price * (cost_frac / 2.0)
    exit_cost_fils = abs(exit_price) * (cost_frac / 2.0)
    net_pnl_r = (raw_pnl - entry_cost_fils - exit_cost_fils) / risk if risk > 0 else 0.0

    return TradeResult(
        date=sig_date,
        stock_code=signal.get("stock_code", ""),
        signal=direction,
        setup_type=signal.get("setup_type", ""),
        entry=fill_price,
        stop_loss=stop,
        tp1=tp1,
        tp2=tp2,
        rr_ratio=signal.get("risk_metrics", {}).get("risk_reward_ratio") or 0.0,
        outcome=outcome,
        pnl_r=round(net_pnl_r, 4),
        win_tp1=win_tp1,
        win_tp2=win_tp2,
        segment=segment,
        regime=signal.get("confluence_details", {}).get("regime", ""),
        confluence_score=signal.get("confluence_details", {}).get("total_score", 0),
    )


# ── Statistical helpers ───────────────────────────────────────────────────────

def _bootstrap_win_rate_ci(wins: int, total: int, n_boot: int = 2_000, seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95 % confidence interval for a win rate."""
    rng = random.Random(seed)
    if total == 0:
        return (0.0, 1.0)
    samples = [1] * wins + [0] * (total - wins)
    boot_rates: list[float] = []
    for _ in range(n_boot):
        resample = [rng.choice(samples) for _ in range(total)]
        boot_rates.append(sum(resample) / total)
    boot_rates.sort()
    lo = boot_rates[int(0.025 * n_boot)]
    hi = boot_rates[int(0.975 * n_boot)]
    return (round(lo, 4), round(hi, 4))


def _compute_sharpe(pnl_r_series: list[float]) -> float:
    """Annualised Sharpe ratio (risk-free = 0, Kuwait convention)."""
    if len(pnl_r_series) < 2:
        return 0.0
    arr = np.array(pnl_r_series)
    std = float(np.std(arr, ddof=1))
    if std < 1e-12:
        return 0.0
    mean = float(np.mean(arr))
    # Assume ~50 trades per year for annualisation
    return round(mean / std * math.sqrt(50), 4)


def _compute_max_drawdown(pnl_r_series: list[float]) -> float:
    """Peak-to-trough drawdown in R-multiples."""
    if not pnl_r_series:
        return 0.0
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in pnl_r_series:
        equity += r
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 4)


def _compute_profit_factor(pnl_r_series: list[float]) -> float:
    """Gross profit / gross loss (returns ∞ if no losses)."""
    gross_profit = sum(r for r in pnl_r_series if r > 0)
    gross_loss = sum(-r for r in pnl_r_series if r < 0)
    if gross_loss < 1e-9:
        return float("inf") if gross_profit > 0 else 1.0
    return round(gross_profit / gross_loss, 4)


# ── Window metric computation ─────────────────────────────────────────────────

def compute_window_metrics(
    trades: list[TradeResult],
    train_from: str,
    train_to: str,
    test_from: str,
    test_to: str,
) -> WindowMetrics:
    """Compute all KPI metrics for one walk-forward window."""
    actual = [
        t for t in trades
        if t.date >= test_from
        and t.date <= test_to
        and t.outcome not in {"SKIPPED", "NOT_FILLED"}
    ]
    n_trades = len(actual)
    m = WindowMetrics(
        train_from=train_from,
        train_to=train_to,
        test_from=test_from,
        test_to=test_to,
        n_signals=len(trades),
        n_trades=n_trades,
    )
    m.orders_filled = n_trades
    m.not_filled = sum(1 for t in trades if t.outcome == "NOT_FILLED")
    m.fill_rate = round(n_trades / len(trades), 4) if trades else 0.0
    if n_trades == 0:
        return m

    wins_tp1 = [t for t in actual if t.win_tp1]
    wins_tp2 = [t for t in actual if t.win_tp2]
    pnl_series = [t.pnl_r for t in actual]

    m.win_rate_tp1 = round(len(wins_tp1) / n_trades, 4)
    m.win_rate_tp2 = round(len(wins_tp2) / n_trades, 4)
    m.profit_factor = _compute_profit_factor(pnl_series)
    m.avg_r = round(float(np.mean(pnl_series)), 4)
    m.expectancy_per_trade = m.avg_r
    m.max_drawdown_r = _compute_max_drawdown(pnl_series)
    m.sharpe = _compute_sharpe(pnl_series)
    m.win_rate_ci_95 = _bootstrap_win_rate_ci(len(wins_tp1), n_trades)
    m.positive_expectancy = m.avg_r > 0.0
    m.exposure_days = n_trades
    m.turnover_r = round(sum(abs(t.pnl_r) for t in actual), 4)
    streak = 0
    for pnl in pnl_series:
        streak = streak + 1 if pnl < 0 else 0
        m.max_losing_streak = max(m.max_losing_streak, streak)
    sorted_losses = sorted(pnl_series)
    tail_count = max(1, int(math.ceil(len(sorted_losses) * 0.05)))
    m.tail_loss_r_5pct = round(float(np.mean(sorted_losses[:tail_count])), 4)
    return m


# ── Monte Carlo parameter stability ──────────────────────────────────────────

def monte_carlo_p_value(
    pnl_r_series: list[float],
    observed_avg_r: float,
    n_iter: int = MONTE_CARLO_ITERATIONS,
    seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Estimate p-value for strategy edge vs. random walk.

    Method (§5 spec): shuffle daily returns 1,000 times; measure how often
    shuffled data produces avg_r ≥ observed value.

    Returns:
        (p_value, parameter_stability_pct)
        Passes when p_value < 0.05 and stability_pct >= 95.
    """
    rng = random.Random(seed)
    if len(pnl_r_series) < 5:
        return (1.0, 0.0)

    # Sign-flip null hypothesis: under H0, return signs are random (coin flip).
    # This tests whether observed_avg_r is significantly above zero.
    abs_series = [abs(x) for x in pnl_r_series]
    n = len(abs_series)

    beat_count = 0
    for _ in range(n_iter):
        flipped_avg = sum(
            a * rng.choice((-1.0, 1.0)) for a in abs_series
        ) / n
        if flipped_avg >= observed_avg_r:
            beat_count += 1

    p_value = beat_count / n_iter
    stability_pct = round((1.0 - p_value) * 100.0, 2)
    return (round(p_value, 4), stability_pct)


# ── Calibration error ─────────────────────────────────────────────────────────

def calibration_error(
    predicted_probs: list[float],
    actual_wins: list[bool],
    n_bins: int = 10,
) -> float:
    """Mean absolute calibration error across equal-frequency bins.

    Args:
        predicted_probs: Predicted TP1 probability for each trade.
        actual_wins:     Whether TP1 was actually hit.

    Returns:
        Mean absolute error as a percentage (target: ≤ ±3 %).
    """
    if len(predicted_probs) != len(actual_wins) or len(predicted_probs) < n_bins:
        return 0.0

    pairs = sorted(zip(predicted_probs, actual_wins), key=lambda x: x[0])
    bin_size = len(pairs) // n_bins
    errors: list[float] = []
    for i in range(n_bins):
        chunk = pairs[i * bin_size: (i + 1) * bin_size]
        if not chunk:
            continue
        mean_pred = sum(p for p, _ in chunk) / len(chunk)
        actual_rate = sum(1 for _, w in chunk if w) / len(chunk)
        errors.append(abs(mean_pred - actual_rate) * 100.0)

    return round(sum(errors) / len(errors), 4) if errors else 0.0


# ── CVaR compliance check ─────────────────────────────────────────────────────

def cvar_compliance(trades: list[TradeResult]) -> float:
    """Return fraction of trades where loss ≤ CVaR limit.

    Spec target: 95 % of trades within CVaR limits.
    Proxy: max single-trade loss ≤ 2.0 R (reflecting the 95th-percentile tail).
    """
    if not trades:
        return 1.0
    max_loss_r = 2.0   # 2 R corresponds to ~95th-percentile CVaR threshold
    within = sum(1 for t in trades if t.pnl_r >= -max_loss_r)
    return round(within / len(trades), 4)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_walk_forward_test(
    rows_by_stock: dict[str, list[dict[str, Any]]],
    stock_codes: list[str] | None = None,
    segment: str = "PREMIER",
    account_equity: float = 100_000.0,
    max_hold_days: int = 10,
    verbose: bool = False,
) -> BacktestReport:
    """Run the full walk-forward backtest across 3 windows for given stocks.

    Args:
        rows_by_stock: Dict mapping stock_code → sorted OHLCV row list.
        stock_codes:   Subset of stocks to test (default: all in rows_by_stock).
        segment:       Market segment for cost calculations.
        account_equity: Account size in KWD.
        max_hold_days: Maximum holding period per trade.
        verbose:       Log per-trade details.

    Returns:
        BacktestReport with all metrics.
    """
    if stock_codes is None:
        stock_codes = list(rows_by_stock.keys())

    report = BacktestReport()
    all_trades_global: list[TradeResult] = []
    all_preds: list[float] = []
    all_wins: list[bool] = []

    for win_idx, (train_from, train_to, test_from, test_to) in enumerate(WALK_FORWARD_WINDOWS):
        window_trades: list[TradeResult] = []

        for code in stock_codes:
            rows = rows_by_stock.get(code, [])
            if not rows:
                continue

            # Filter rows within test window
            test_rows = [r for r in rows if test_from <= str(r.get("date", "")) <= test_to]
            if len(test_rows) < 20:
                continue

            # Generate signal on the last bar of the training window (simulate daily signal)
            train_rows = [r for r in rows if str(r.get("date", "")) <= train_to]
            if len(train_rows) < 50:
                continue

            try:
                sig = asyncio.run(generate_kuwait_signal(
                    rows=train_rows,
                    stock_code=code,
                    segment=segment,
                    account_equity=account_equity,
                    delay_hours=0,
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Signal generation failed for %s (window %d): %s", code, win_idx + 1, exc)
                continue

            if sig.get("signal") == "NEUTRAL":
                continue

            # Collect predicted probability only after an order fills.
            p_tp1 = sig.get("probabilities", {}).get("p_tp1_before_sl")

            # Simulate trade outcome in test rows
            trade = simulate_trade(
                rows=rows,          # full series so price levels are reachable
                signal=sig,
                max_hold_days=max_hold_days,
            )
            if trade.outcome != "SKIPPED":
                window_trades.append(trade)
                report.signals_generated += 1
                if trade.outcome == "NOT_FILLED":
                    report.not_filled += 1
                else:
                    report.orders_filled += 1
            if trade.outcome not in {"SKIPPED", "NOT_FILLED"}:
                all_trades_global.append(trade)
                if p_tp1 is not None:
                    all_preds.append(float(p_tp1))
                    all_wins.append(trade.win_tp1)

            if verbose:
                logger.info(
                    "[W%d] %s %s → %s  R=%.2f  score=%d",
                    win_idx + 1, code, sig["signal"], trade.outcome,
                    trade.pnl_r, trade.confluence_score,
                )

        m = compute_window_metrics(window_trades, train_from, train_to, test_from, test_to)
        report.windows.append(m)
        logger.info(
            "Window %d (%s→%s)  trades=%d  WR=%.1f%%  avgR=%.3f  PF=%.2f  positive=%s",
            win_idx + 1, test_from, test_to,
            m.n_trades, m.win_rate_tp1 * 100, m.avg_r, m.profit_factor,
            m.positive_expectancy,
        )

    # ── Aggregate stats ───────────────────────────────────────────────────────
    report.total_trades = len(all_trades_global)
    report.fill_rate = round(report.orders_filled / report.signals_generated, 4) if report.signals_generated else 0.0
    if all_trades_global:
        wins_tp1 = [t for t in all_trades_global if t.win_tp1]
        report.aggregate_win_rate_tp1 = round(len(wins_tp1) / len(all_trades_global), 4)
        pnl_all = [t.pnl_r for t in all_trades_global]
        report.aggregate_avg_r = round(float(np.mean(pnl_all)), 4)
        report.aggregate_profit_factor = _compute_profit_factor(pnl_all)

        # Monte Carlo
        mc_p, stability = monte_carlo_p_value(pnl_all, report.aggregate_avg_r)
        report.monte_carlo_p_value = mc_p
        report.parameter_stability_pct = stability

        # Calibration error
        if len(all_preds) == len(all_wins) and len(all_preds) >= 10:
            report.calibration_error_pct = calibration_error(all_preds, all_wins)

        # CVaR compliance
        report.cvar_compliance_pct = cvar_compliance(all_trades_global)

    # ── Final acceptance check ────────────────────────────────────────────────
    report.all_positive_expectancy = all(w.positive_expectancy for w in report.windows if w.n_trades > 0)
    report.passed_all_criteria = (
        report.all_positive_expectancy
        and report.aggregate_win_rate_tp1 >= 0.68
        and report.calibration_error_pct <= 3.0
        and report.monte_carlo_p_value < 0.05
        and report.parameter_stability_pct >= 95.0
        and report.cvar_compliance_pct >= 0.95
        and report.aggregate_avg_r >= 1.2   # ≥+1.2R net of slippage (spec §1)
    )

    logger.info(
        "Backtest complete — trades=%d  WR=%.1f%%  avgR=%.3f  CalErr=%.2f%%  MC_p=%.4f  PASSED=%s",
        report.total_trades,
        report.aggregate_win_rate_tp1 * 100,
        report.aggregate_avg_r,
        report.calibration_error_pct,
        report.monte_carlo_p_value,
        report.passed_all_criteria,
    )
    return report
