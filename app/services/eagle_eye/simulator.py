"""
Eagle Eye Paper Trading Simulator — SimulatorEngine.

Three parallel strategies run daily (after ratings recompute):
  CONSERVATIVE: min_confidence=65, stages=[EARLY_BREAKOUT, MARKUP_TRENDING]
  MODERATE:     min_confidence=60, stages=[STEALTH_ACCUMULATION, EARLY_BREAKOUT, MARKUP_TRENDING]
  AGGRESSIVE:   min_confidence=55, stages=[STEALTH_ACCUMULATION, EARLY_BREAKOUT,
                                           MARKUP_TRENDING, CAPITULATION_EXHAUSTION]

Each strategy starts with 10,000 KWD fake capital, maintains independent positions,
and never shares positions across strategies.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MIN_TRADE_SIZE_KWD = 100.0      # ignore positions smaller than this
MAX_OPEN_POSITIONS = 10         # per strategy
SECTOR_CAP_PCT = 35.0           # max sector exposure %
TIME_STOP_TRADING_DAYS = 30     # close after 30 trading days if no exit hit
SCALE_OUT_FRACTION = 0.33       # fraction of remaining shares to close at TP1/TP2

# Stage sets
BEARISH_STAGES = {"DISTRIBUTION_TOPPING", "MARKDOWN_DECLINE"}
BULLISH_ENTRY_STAGES_ALL = {
    "STEALTH_ACCUMULATION", "EARLY_BREAKOUT",
    "MARKUP_TRENDING", "CAPITULATION_EXHAUSTION",
}

# ── Strategy Configs ─────────────────────────────────────────────────────────
@dataclass
class StrategyConfig:
    name: str
    min_confidence: float
    allowed_stages: set[str]
    portfolio_id: int


STRATEGIES = [
    StrategyConfig(
        name="CONSERVATIVE",
        min_confidence=65.0,
        allowed_stages={"EARLY_BREAKOUT", "MARKUP_TRENDING"},
        portfolio_id=1,
    ),
    StrategyConfig(
        name="MODERATE",
        min_confidence=60.0,
        allowed_stages={"STEALTH_ACCUMULATION", "EARLY_BREAKOUT", "MARKUP_TRENDING"},
        portfolio_id=2,
    ),
    StrategyConfig(
        name="AGGRESSIVE",
        min_confidence=55.0,
        allowed_stages={
            "STEALTH_ACCUMULATION", "EARLY_BREAKOUT",
            "MARKUP_TRENDING", "CAPITULATION_EXHAUSTION",
        },
        portfolio_id=3,
    ),
]


# ── Entry decision ───────────────────────────────────────────────────────────
@dataclass
class EntryDecision:
    should_enter: bool
    skip_reason: Optional[str] = None


def _skip(reason: str) -> EntryDecision:
    return EntryDecision(should_enter=False, skip_reason=reason)


def _enter() -> EntryDecision:
    return EntryDecision(should_enter=True)


# ── DB helpers (thin wrappers around app.core.database) ──────────────────────

def _exec(sql: str, params: tuple = ()) -> None:
    from app.core.database import exec_sql
    exec_sql(sql, params)


def _query_one(sql: str, params: tuple = ()) -> Optional[Any]:
    from app.core.database import query_one
    return query_one(sql, params)


def _query_all(sql: str, params: tuple = ()) -> list:
    from app.core.database import query_all
    rows = query_all(sql, params)
    return [dict(r.items()) for r in rows] if rows else []


def _query_val(sql: str, params: tuple = ()):
    from app.core.database import query_val
    return query_val(sql, params)


def _now_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


# ── Portfolio state helpers ──────────────────────────────────────────────────

def _get_portfolio(portfolio_id: int) -> Optional[dict]:
    row = _query_one(
        "SELECT * FROM simulator_portfolios WHERE id = ?",
        (portfolio_id,),
    )
    return dict(row.items()) if row else None


def _get_open_positions(portfolio_id: int) -> List[dict]:
    return _query_all(
        "SELECT * FROM simulator_positions WHERE portfolio_id = ? AND status = 'OPEN'",
        (portfolio_id,),
    )


def _get_ohlcv(ticker: str, bar_date: str) -> Optional[dict]:
    """Return OHLCV row for ticker on the given date (ISO string)."""
    row = _query_one(
        "SELECT open, high, low, close, volume FROM ee_ohlcv_cache WHERE ticker = ? AND bar_date = ?",
        (ticker.upper(), bar_date),
    )
    return dict(row.items()) if row else None


def _get_ohlcv_near(ticker: str, target_date: str) -> Optional[dict]:
    """
    Try today's date first; fall back to the most recent bar within ±5 days.
    Useful during backfill when not every ticker has data on every date.
    """
    row = _get_ohlcv(ticker, target_date)
    if row:
        return row
    # Scan ±5 calendar days
    d = date.fromisoformat(target_date)
    for delta in range(1, 6):
        for sign in (-1, +1):
            candidate = (d + timedelta(days=delta * sign)).isoformat()
            row = _get_ohlcv(ticker, candidate)
            if row:
                return row
    return None


def _get_sector(ticker: str) -> str:
    row = _query_one(
        "SELECT sector FROM ee_ratings_cache WHERE ticker = ?",
        (ticker.upper(),),
    )
    if row:
        v = dict(row.items()).get("sector")
        return v or "UNKNOWN"
    return "UNKNOWN"


def _trading_days_between(start_date: str, end_date: str) -> int:
    """Approximate trading days (Mon–Thu in Kuwait market, 5 days/week approx)."""
    try:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date)
        delta = (e - s).days
        # Rough: 5/7 trading days
        return max(0, round(delta * 5 / 7))
    except Exception:
        return 0


def _sector_exposure_pct(portfolio_id: int, sector: str, portfolio_value: float) -> float:
    """Percentage of portfolio value allocated to *sector* in open positions."""
    rows = _query_all(
        """SELECT sp.size_kwd FROM simulator_positions sp
           JOIN ee_ratings_cache rc ON rc.ticker = sp.ticker
           WHERE sp.portfolio_id = ? AND sp.status = 'OPEN'
             AND (rc.sector = ? OR ? = 'UNKNOWN')""",
        (portfolio_id, sector, sector),
    )
    total = sum(float(r.get("size_kwd") or 0) for r in rows)
    if portfolio_value <= 0:
        return 0.0
    return (total / portfolio_value) * 100.0


# ── Main Engine ──────────────────────────────────────────────────────────────

class SimulatorEngine:
    """Runs daily after the Eagle Eye rating recompute."""

    def run_daily(self, run_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Called once per trading day after market close.
        For each strategy: exits → entries → snapshot.
        """
        from app.services.eagle_eye.store import ensure_tables
        ensure_tables()
        self._ensure_simulator_tables()

        if run_date is None:
            run_date = date.today()

        date_str = run_date.isoformat()
        results: Dict[str, Any] = {}

        for strategy in STRATEGIES:
            portfolio = _get_portfolio(strategy.portfolio_id)
            if portfolio is None:
                logger.warning("Simulator: portfolio %d not found, skipping", strategy.portfolio_id)
                continue
            try:
                exits = self._process_exits(strategy, portfolio, date_str)
                entries = self._process_entries(strategy, portfolio, date_str)
                self._snapshot_portfolio(strategy, portfolio, date_str)
                results[strategy.name] = {
                    "exits": exits,
                    "entries": entries,
                    "date": date_str,
                }
                logger.info(
                    "Simulator %s [%s]: %d exits, %d entries",
                    strategy.name, date_str, len(exits), len(entries),
                )
            except Exception as exc:
                logger.exception("Simulator %s failed for %s: %s", strategy.name, date_str, exc)
                results[strategy.name] = {"error": str(exc)}

        return results

    # ── Exits ────────────────────────────────────────────────────────────

    def _process_exits(self, strategy: StrategyConfig, portfolio: dict, date_str: str) -> list:
        open_positions = _get_open_positions(strategy.portfolio_id)
        closed = []

        for pos in open_positions:
            ohlcv = _get_ohlcv_near(pos["ticker"], date_str)
            if ohlcv is None:
                # No price data — skip exit check
                continue

            h = float(ohlcv["high"] or 0)
            l = float(ohlcv["low"] or 0)
            c = float(ohlcv["close"] or 0)
            stop = float(pos["planned_stop_loss"] or 0)
            tp1 = float(pos["planned_tp1"] or 0)
            tp2 = float(pos["planned_tp2"] or 0)
            tp3 = float(pos["planned_tp3"] or 0)
            tp1_hit = bool(pos.get("tp1_hit"))
            tp2_hit = bool(pos.get("tp2_hit"))

            days_held = _trading_days_between(pos["entry_date"] or date_str, date_str)

            # Update MAE/MFE while still open
            self._update_excursion(pos, h, l)

            exit_triggered = False

            # 1. Stop loss
            if stop > 0 and l <= stop:
                self._close_position(pos, portfolio, min(c, stop), "STOP_HIT", date_str, days_held)
                closed.append({"ticker": pos["ticker"], "reason": "STOP_HIT"})
                exit_triggered = True

            # 2. TP3 (full close)
            elif tp3 > 0 and h >= tp3:
                self._close_position(pos, portfolio, tp3, "TP3_HIT", date_str, days_held)
                closed.append({"ticker": pos["ticker"], "reason": "TP3_HIT"})
                exit_triggered = True

            # 3. TP2 (scale out 33%)
            elif tp2 > 0 and h >= tp2 and not tp2_hit:
                self._partial_close(pos, portfolio, tp2, SCALE_OUT_FRACTION, "TP2_HIT", date_str)
                closed.append({"ticker": pos["ticker"], "reason": "TP2_HIT_PARTIAL"})

            # 4. TP1 (scale out 33%)
            elif tp1 > 0 and h >= tp1 and not tp1_hit:
                self._partial_close(pos, portfolio, tp1, SCALE_OUT_FRACTION, "TP1_HIT", date_str)
                closed.append({"ticker": pos["ticker"], "reason": "TP1_HIT_PARTIAL"})

            if exit_triggered:
                continue

            # 5. Stage transition to bearish
            if not exit_triggered:
                current_rating = self._get_current_rating(pos["ticker"])
                if (current_rating
                        and current_rating.get("stage") in BEARISH_STAGES
                        and pos.get("entry_stage") in {"EARLY_BREAKOUT", "MARKUP_TRENDING"}):
                    self._close_position(pos, portfolio, c, "STAGE_TRANSITION", date_str, days_held)
                    closed.append({"ticker": pos["ticker"], "reason": "STAGE_TRANSITION"})
                    exit_triggered = True

            # 6. Time stop
            if not exit_triggered and days_held >= TIME_STOP_TRADING_DAYS:
                self._close_position(pos, portfolio, c, "TIME_STOP", date_str, days_held)
                closed.append({"ticker": pos["ticker"], "reason": "TIME_STOP"})

        return closed

    # ── Entries ──────────────────────────────────────────────────────────

    def _process_entries(self, strategy: StrategyConfig, portfolio: dict, date_str: str) -> list:
        opened = []
        # Re-read portfolio after exits may have freed cash
        portfolio = _get_portfolio(strategy.portfolio_id) or portfolio
        todays_ratings = self._get_todays_ratings(date_str)

        for rating in todays_ratings:
            portfolio = _get_portfolio(strategy.portfolio_id) or portfolio
            decision = self._evaluate_entry(strategy, rating, portfolio)
            if not decision.should_enter:
                self._log_considered(strategy.portfolio_id, date_str, rating, decision.skip_reason)
                continue

            portfolio_value = float(portfolio.get("total_value_kwd") or 10000)
            position_size_kwd = self._compute_position_size(rating, portfolio_value)
            if position_size_kwd < MIN_TRADE_SIZE_KWD:
                self._log_considered(strategy.portfolio_id, date_str, rating, "POSITION_TOO_SMALL")
                continue

            cash = float(portfolio.get("cash_balance_kwd") or 0)
            if position_size_kwd > cash:
                position_size_kwd = cash  # use all available cash if smaller
            if position_size_kwd < MIN_TRADE_SIZE_KWD:
                self._log_considered(strategy.portfolio_id, date_str, rating, "INSUFFICIENT_CASH")
                continue

            self._open_position(strategy, rating, portfolio, position_size_kwd, date_str)
            opened.append({"ticker": rating.get("ticker"), "size_kwd": position_size_kwd})

        return opened

    # ── Evaluation ───────────────────────────────────────────────────────

    def _evaluate_entry(
        self, strategy: StrategyConfig, rating: dict, portfolio: dict
    ) -> EntryDecision:
        confidence = float(rating.get("confidence") or 0)
        stage = rating.get("stage") or ""
        ticker = rating.get("ticker") or ""
        sector = rating.get("sector") or "UNKNOWN"

        if confidence < strategy.min_confidence:
            return _skip("CONFIDENCE_BELOW_THRESHOLD")
        if stage not in strategy.allowed_stages:
            return _skip("STAGE_NOT_ALLOWED")
        if self._already_holding(strategy.portfolio_id, ticker):
            return _skip("ALREADY_HOLDING")

        cash = float(portfolio.get("cash_balance_kwd") or 0)
        if cash < MIN_TRADE_SIZE_KWD:
            return _skip("INSUFFICIENT_CASH")

        open_positions = _get_open_positions(strategy.portfolio_id)
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            return _skip("MAX_POSITIONS_REACHED")

        portfolio_value = float(portfolio.get("total_value_kwd") or 10000)
        if _sector_exposure_pct(strategy.portfolio_id, sector, portfolio_value) >= SECTOR_CAP_PCT:
            return _skip("SECTOR_CAP_REACHED")

        return _enter()

    # ── Position sizing ──────────────────────────────────────────────────

    def _compute_position_size(self, rating: dict, portfolio_value: float) -> float:
        from app.services.eagle_eye.rating_engine import compute_position_size

        entry = float(rating.get("entry_primary") or rating.get("last_price") or 0)
        stop = float(rating.get("stop_loss") or 0)
        confidence = float(rating.get("confidence") or 60)
        tp1 = float(rating.get("tp1") or 0) or None

        if entry <= 0 or stop <= 0 or stop >= entry:
            # Fallback: 5% of portfolio
            return round(portfolio_value * 0.05, 2)

        result = compute_position_size(
            confidence=confidence,
            entry=entry,
            stop=stop,
            portfolio_kwd=portfolio_value,
            avg_daily_turnover_kwd=portfolio_value * 2,  # assume 200% turnover proxy
            dna=None,
            regime_multiplier=1.0,
            tp1_price=tp1,
        )
        return float(result.get("suggested_kwd") or 0)

    # ── Open position ────────────────────────────────────────────────────

    def _open_position(
        self,
        strategy: StrategyConfig,
        rating: dict,
        portfolio: dict,
        size_kwd: float,
        date_str: str,
    ) -> None:
        entry_price = float(rating.get("entry_primary") or rating.get("last_price") or 0)
        if entry_price <= 0:
            return

        shares = round(size_kwd / entry_price, 4)
        portfolio_value = float(portfolio.get("total_value_kwd") or 10000)
        size_pct = (size_kwd / portfolio_value * 100) if portfolio_value > 0 else 0

        indicators = rating.get("indicators_json") or {}
        if isinstance(indicators, str):
            try:
                indicators = json.loads(indicators)
            except Exception:
                indicators = {}

        accumulation_score = float(indicators.get("accumulation_score") or 0) if indicators else 0

        signals = rating.get("signals_json") or []
        if isinstance(signals, str):
            try:
                signals = json.loads(signals)
            except Exception:
                signals = []

        _exec(
            """
            INSERT INTO simulator_positions (
                portfolio_id, ticker, status, entry_date, entry_price,
                shares, shares_remaining, size_kwd, size_pct_of_portfolio,
                entry_confidence, entry_stage, entry_rating, entry_thesis,
                entry_signal_breakdown, entry_accumulation_score, entry_indicators_snapshot,
                planned_stop_loss, planned_tp1, planned_tp2, planned_tp3,
                tp1_hit, tp2_hit,
                max_unrealized_gain_pct, max_unrealized_loss_pct,
                created_at, updated_at
            ) VALUES (
                ?, ?, 'OPEN', ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                0, 0,
                0.0, 0.0,
                ?, ?
            )
            """,
            (
                strategy.portfolio_id,
                rating.get("ticker", "").upper(),
                date_str,
                round(entry_price, 6),
                shares,
                shares,  # shares_remaining starts equal to shares
                round(size_kwd, 4),
                round(size_pct, 4),
                float(rating.get("confidence") or 0),
                rating.get("stage") or "",
                rating.get("rating") or "",
                rating.get("thesis") or "",
                json.dumps(signals),
                round(accumulation_score, 4),
                json.dumps(indicators),
                float(rating.get("stop_loss") or 0),
                float(rating.get("tp1") or 0),
                float(rating.get("tp2") or 0),
                float(rating.get("tp3") or 0),
                _now_ts(),
                _now_ts(),
            ),
        )

        # Deduct cash
        new_cash = float(portfolio.get("cash_balance_kwd") or 0) - size_kwd
        _exec(
            "UPDATE simulator_portfolios SET cash_balance_kwd = ?, updated_at = ? WHERE id = ?",
            (round(new_cash, 4), _now_ts(), strategy.portfolio_id),
        )

    # ── Close position (full) ────────────────────────────────────────────

    def _close_position(
        self,
        pos: dict,
        portfolio: dict,
        exit_price: float,
        reason: str,
        date_str: str,
        days_held: int,
    ) -> None:
        shares_remaining = float(pos.get("shares_remaining") or pos.get("shares") or 0)
        entry_price = float(pos.get("entry_price") or 0)
        size_kwd = float(pos.get("size_kwd") or 0)

        if entry_price <= 0:
            return

        proceeds = shares_remaining * exit_price
        cost_basis = shares_remaining * entry_price
        pnl_kwd = proceeds - cost_basis
        pnl_pct = (pnl_kwd / cost_basis * 100) if cost_basis > 0 else 0

        _exec(
            """
            UPDATE simulator_positions SET
                status = 'CLOSED', exit_date = ?, exit_price = ?,
                exit_reason = ?, pnl_kwd = ?, pnl_pct = ?,
                days_held = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                date_str, round(exit_price, 6),
                reason, round(pnl_kwd, 4), round(pnl_pct, 4),
                days_held, _now_ts(),
                pos["id"],
            ),
        )

        # Return cash to portfolio
        portfolio_id = pos["portfolio_id"]
        new_cash = float(portfolio.get("cash_balance_kwd") or 0) + proceeds
        _exec(
            "UPDATE simulator_portfolios SET cash_balance_kwd = ?, updated_at = ? WHERE id = ?",
            (round(new_cash, 4), _now_ts(), portfolio_id),
        )

    # ── Partial close ────────────────────────────────────────────────────

    def _partial_close(
        self,
        pos: dict,
        portfolio: dict,
        exit_price: float,
        fraction: float,
        reason: str,
        date_str: str,
    ) -> None:
        shares_remaining = float(pos.get("shares_remaining") or pos.get("shares") or 0)
        entry_price = float(pos.get("entry_price") or 0)
        shares_to_close = round(shares_remaining * fraction, 4)

        if shares_to_close <= 0 or entry_price <= 0:
            return

        proceeds = shares_to_close * exit_price
        cost_basis = shares_to_close * entry_price
        partial_pnl = proceeds - cost_basis

        new_remaining = shares_remaining - shares_to_close

        # Mark TP hit flag
        tp_flag_col = "tp1_hit" if reason == "TP1_HIT" else "tp2_hit"
        _exec(
            f"""
            UPDATE simulator_positions SET
                shares_remaining = ?,
                {tp_flag_col} = 1,
                updated_at = ?
            WHERE id = ?
            """,
            (round(new_remaining, 4), _now_ts(), pos["id"]),
        )

        # Return partial proceeds to cash
        portfolio_id = pos["portfolio_id"]
        new_cash = float(portfolio.get("cash_balance_kwd") or 0) + proceeds
        _exec(
            "UPDATE simulator_portfolios SET cash_balance_kwd = ?, updated_at = ? WHERE id = ?",
            (round(new_cash, 4), _now_ts(), portfolio_id),
        )

    # ── MFE/MAE tracking ────────────────────────────────────────────────

    def _update_excursion(self, pos: dict, high: float, low: float) -> None:
        entry_price = float(pos.get("entry_price") or 0)
        if entry_price <= 0:
            return
        current_gain = (high - entry_price) / entry_price * 100
        current_loss = (low - entry_price) / entry_price * 100
        best = max(float(pos.get("max_unrealized_gain_pct") or 0), current_gain)
        worst = min(float(pos.get("max_unrealized_loss_pct") or 0), current_loss)
        _exec(
            """UPDATE simulator_positions SET
               max_unrealized_gain_pct = ?, max_unrealized_loss_pct = ?, updated_at = ?
               WHERE id = ?""",
            (round(best, 4), round(worst, 4), _now_ts(), pos["id"]),
        )

    # ── Snapshot ─────────────────────────────────────────────────────────

    def _snapshot_portfolio(
        self, strategy: StrategyConfig, portfolio: dict, date_str: str
    ) -> None:
        # Re-read fresh portfolio state
        portfolio = _get_portfolio(strategy.portfolio_id) or portfolio
        open_positions = _get_open_positions(strategy.portfolio_id)

        cash = float(portfolio.get("cash_balance_kwd") or 0)

        # Mark-to-market open positions
        open_value = 0.0
        for pos in open_positions:
            ohlcv = _get_ohlcv_near(pos["ticker"], date_str)
            if ohlcv:
                price = float(ohlcv["close"] or pos.get("entry_price") or 0)
            else:
                price = float(pos.get("entry_price") or 0)
            shares_remaining = float(pos.get("shares_remaining") or pos.get("shares") or 0)
            open_value += shares_remaining * price

        total_value = cash + open_value
        starting_capital = float(portfolio.get("starting_capital_kwd") or 10000)
        cumulative_return_pct = ((total_value - starting_capital) / starting_capital * 100) if starting_capital > 0 else 0

        # Previous day's total value for daily P&L
        prev_row = _query_one(
            """SELECT total_value_kwd FROM simulator_daily_snapshots
               WHERE portfolio_id = ? ORDER BY date DESC LIMIT 1""",
            (strategy.portfolio_id,),
        )
        prev_total = float(dict(prev_row.items()).get("total_value_kwd") or starting_capital) if prev_row else starting_capital
        daily_pnl = total_value - prev_total

        # Max drawdown from peak
        peak_row = _query_one(
            "SELECT MAX(total_value_kwd) FROM simulator_daily_snapshots WHERE portfolio_id = ?",
            (strategy.portfolio_id,),
        )
        peak = float(peak_row[0] if peak_row and peak_row[0] else total_value)
        peak = max(peak, total_value, starting_capital)
        drawdown_pct = ((total_value - peak) / peak * 100) if peak > 0 else 0

        # Update portfolio totals
        _exec(
            "UPDATE simulator_portfolios SET total_value_kwd = ?, updated_at = ? WHERE id = ?",
            (round(total_value, 4), _now_ts(), strategy.portfolio_id),
        )

        # Upsert snapshot row
        existing = _query_one(
            "SELECT id FROM simulator_daily_snapshots WHERE portfolio_id = ? AND date = ?",
            (strategy.portfolio_id, date_str),
        )
        if existing:
            _exec(
                """UPDATE simulator_daily_snapshots SET
                   cash_balance_kwd = ?, open_positions_value_kwd = ?,
                   total_value_kwd = ?, daily_pnl_kwd = ?,
                   cumulative_return_pct = ?, drawdown_from_peak_pct = ?,
                   open_position_count = ?
                   WHERE portfolio_id = ? AND date = ?""",
                (
                    round(cash, 4), round(open_value, 4),
                    round(total_value, 4), round(daily_pnl, 4),
                    round(cumulative_return_pct, 4), round(drawdown_pct, 4),
                    len(open_positions),
                    strategy.portfolio_id, date_str,
                ),
            )
        else:
            _exec(
                """INSERT INTO simulator_daily_snapshots (
                   portfolio_id, date, cash_balance_kwd, open_positions_value_kwd,
                   total_value_kwd, daily_pnl_kwd, cumulative_return_pct,
                   drawdown_from_peak_pct, open_position_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    strategy.portfolio_id, date_str,
                    round(cash, 4), round(open_value, 4),
                    round(total_value, 4), round(daily_pnl, 4),
                    round(cumulative_return_pct, 4), round(drawdown_pct, 4),
                    len(open_positions),
                ),
            )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _already_holding(self, portfolio_id: int, ticker: str) -> bool:
        count = _query_val(
            "SELECT COUNT(*) FROM simulator_positions WHERE portfolio_id = ? AND ticker = ? AND status = 'OPEN'",
            (portfolio_id, ticker.upper()),
        )
        return bool(count and int(count) > 0)

    def _get_current_rating(self, ticker: str) -> Optional[dict]:
        row = _query_one(
            "SELECT stage, rating, confidence FROM ee_ratings_cache WHERE ticker = ?",
            (ticker.upper(),),
        )
        return dict(row.items()) if row else None

    def _get_todays_ratings(self, date_str: str) -> List[dict]:
        """Load all rated stocks.  For backfill, we accept any recent rating."""
        rows = _query_all(
            """SELECT ticker, name_en, sector, stage, rating, confidence, thesis,
                      entry_primary, stop_loss, tp1, tp2, tp3, last_price,
                      signals_json, indicators_json, computed_at
               FROM   ee_ratings_cache
               ORDER  BY confidence DESC""",
            (),
        )
        result = []
        for r in rows:
            indicators = r.get("indicators_json")
            if isinstance(indicators, str):
                try:
                    r["indicators_json"] = json.loads(indicators)
                except Exception:
                    r["indicators_json"] = {}
            result.append(r)
        return result

    def _log_considered(
        self, portfolio_id: int, date_str: str, rating: dict, reason: Optional[str]
    ) -> None:
        _exec(
            """INSERT INTO simulator_considered_trades
               (portfolio_id, date, ticker, confidence, stage, reason_skipped)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                portfolio_id, date_str,
                rating.get("ticker", "").upper(),
                float(rating.get("confidence") or 0),
                rating.get("stage") or "",
                reason or "",
            ),
        )

    def _ensure_simulator_tables(self) -> None:
        """Ensure tables exist (idempotent — called on first run)."""
        _exec(
            """CREATE TABLE IF NOT EXISTS simulator_portfolios (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               strategy_name TEXT NOT NULL,
               starting_capital_kwd REAL NOT NULL DEFAULT 10000,
               cash_balance_kwd REAL NOT NULL DEFAULT 10000,
               total_value_kwd REAL NOT NULL DEFAULT 10000,
               created_at TEXT, updated_at TEXT
            )""",
        )
        _exec(
            """CREATE TABLE IF NOT EXISTS simulator_positions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               portfolio_id INTEGER NOT NULL,
               ticker TEXT NOT NULL,
               status TEXT NOT NULL DEFAULT 'OPEN',
               entry_date TEXT, entry_price REAL,
               shares REAL, shares_remaining REAL,
               size_kwd REAL, size_pct_of_portfolio REAL,
               entry_confidence REAL, entry_stage TEXT, entry_rating TEXT,
               entry_thesis TEXT, entry_signal_breakdown TEXT,
               entry_accumulation_score REAL, entry_indicators_snapshot TEXT,
               planned_stop_loss REAL, planned_tp1 REAL, planned_tp2 REAL, planned_tp3 REAL,
               tp1_hit INTEGER NOT NULL DEFAULT 0,
               tp2_hit INTEGER NOT NULL DEFAULT 0,
               exit_date TEXT, exit_price REAL, exit_reason TEXT,
               pnl_kwd REAL, pnl_pct REAL, days_held INTEGER,
               max_unrealized_gain_pct REAL, max_unrealized_loss_pct REAL,
               created_at TEXT, updated_at TEXT
            )""",
        )
        _exec(
            """CREATE TABLE IF NOT EXISTS simulator_daily_snapshots (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               portfolio_id INTEGER NOT NULL,
               date TEXT NOT NULL,
               cash_balance_kwd REAL, open_positions_value_kwd REAL,
               total_value_kwd REAL, daily_pnl_kwd REAL,
               cumulative_return_pct REAL, drawdown_from_peak_pct REAL,
               open_position_count INTEGER,
               UNIQUE(portfolio_id, date)
            )""",
        )
        _exec(
            """CREATE TABLE IF NOT EXISTS simulator_considered_trades (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               portfolio_id INTEGER NOT NULL,
               date TEXT, ticker TEXT, confidence REAL, stage TEXT, reason_skipped TEXT
            )""",
        )
        # Seed portfolios if missing
        count = _query_val("SELECT COUNT(*) FROM simulator_portfolios", ())
        if not count or int(count) == 0:
            now = _now_ts()
            for strat in STRATEGIES:
                _exec(
                    """INSERT INTO simulator_portfolios
                       (strategy_name, starting_capital_kwd, cash_balance_kwd, total_value_kwd, created_at, updated_at)
                       VALUES (?, 10000, 10000, 10000, ?, ?)""",
                    (strat.name, now, now),
                )

    # ── Manual override ──────────────────────────────────────────────────

    def manual_override_close(self, position_id: int, current_price: float) -> dict:
        """User closes a position from the UI at the given price."""
        row = _query_one("SELECT * FROM simulator_positions WHERE id = ?", (position_id,))
        if row is None:
            raise ValueError(f"Position {position_id} not found")
        pos = dict(row.items())
        if pos.get("status") != "OPEN":
            raise ValueError(f"Position {position_id} is not OPEN (status={pos.get('status')})")

        portfolio = _get_portfolio(pos["portfolio_id"])
        if portfolio is None:
            raise ValueError(f"Portfolio {pos['portfolio_id']} not found")

        date_str = date.today().isoformat()
        days_held = _trading_days_between(pos.get("entry_date") or date_str, date_str)
        self._close_position(pos, portfolio, current_price, "MANUAL_OVERRIDE", date_str, days_held)

        # Set OVERRIDDEN status instead of CLOSED for UI distinction
        _exec(
            "UPDATE simulator_positions SET status = 'OVERRIDDEN', updated_at = ? WHERE id = ?",
            (_now_ts(), position_id),
        )
        return {"position_id": position_id, "exit_price": current_price, "status": "OVERRIDDEN"}


# Module-level singleton
_engine = SimulatorEngine()


def get_engine() -> SimulatorEngine:
    return _engine
