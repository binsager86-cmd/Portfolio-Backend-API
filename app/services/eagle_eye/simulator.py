"""
Eagle Eye Paper Trading Simulator — SimulatorEngine.

Two parallel rating-transition cards run daily (after ratings recompute):
    BUY:        enters current BUY/STRONG_BUY ratings
    WATCHLIST:  enters current WATCH/WATCHLIST ratings for paper tracking

Each card starts with 100,000 KWD fake capital, maintains independent positions,
and exits when Eagle Eye emits SELL/STRONG_SELL.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
STARTING_CAPITAL_KWD = 100000.0
MIN_TRADE_SIZE_KWD = 100.0      # ignore positions smaller than this
MAX_OPEN_POSITIONS = 10         # per strategy
SECTOR_CAP_PCT = 35.0           # max sector exposure %
SCALE_OUT_FRACTION = 0.33       # fraction of remaining shares to close at TP1/TP2
ADTV_LOOKBACK_DAYS = 20         # rolling lookback for liquidity cap
MAX_PARTICIPATION_RATE = 0.10   # max 10% of average daily traded value

# v9 ML decision thresholds
ML_MIN_CONFIDENCE = 55.0
ML_EXIT_CONFIDENCE_FLOOR = 25.0
ML_STOP_LOSS_PCT = 8.0
ML_PARTIAL_TAKE_PROFIT_PCT = 20.0
ML_PARTIAL_TAKE_PROFIT_FRACTION = 0.50

KUWAIT_TRADING_WEEKDAYS = {6, 0, 1, 2, 3}  # Sun-Thu using Python weekday()

# Stage sets
BEARISH_STAGES = {"DISTRIBUTION_TOPPING", "MARKDOWN_DECLINE"}
BULLISH_ENTRY_STAGES_ALL = {
    "STEALTH_ACCUMULATION", "EARLY_BREAKOUT",
    "MARKUP_TRENDING", "CAPITULATION_EXHAUSTION",
}

STAGE_ALIASES = {
    "ACCUMULATION": "STEALTH_ACCUMULATION",
    "EARLY_MARKUP": "EARLY_BREAKOUT",
    "BREAKOUT": "EARLY_BREAKOUT",
    "MARKUP": "MARKUP_TRENDING",
    "DISTRIBUTION": "DISTRIBUTION_TOPPING",
    "MARKDOWN": "MARKDOWN_DECLINE",
}


def _normalize_stage(stage: Any) -> str:
    raw = str(stage or "").strip().upper()
    return STAGE_ALIASES.get(raw, raw)

# ── Strategy Configs ─────────────────────────────────────────────────────────
@dataclass
class StrategyConfig:
    name: str
    entry_ratings: set[str]
    min_confidence: float
    allowed_stages: set[str]
    portfolio_id: int


STRATEGIES = [
    StrategyConfig(
        name="BUY",
        entry_ratings={"BUY", "STRONG_BUY"},
        min_confidence=55.0,
        allowed_stages=BULLISH_ENTRY_STAGES_ALL,
        portfolio_id=1,
    ),
    StrategyConfig(
        name="WATCHLIST",
        entry_ratings={"WATCH", "WATCHLIST"},
        min_confidence=0.0,
        allowed_stages=BULLISH_ENTRY_STAGES_ALL | {"NEUTRAL", "NEUTRAL_AMBIGUOUS"},
        portfolio_id=2,
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


def ensure_simulator_tables() -> None:
    """Idempotent DDL — creates all simulator tables. Called at app startup."""
    SimulatorEngine()._ensure_simulator_tables()


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


def _column_exists(table: str, column: str) -> bool:
    from app.core.database import column_exists
    return bool(column_exists(table, column))


def _now_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


_MARKET_TIER_CACHE: Optional[dict[str, str]] = None


def _normalize_market_tier(market_tier: Optional[str]) -> str:
    tier = (market_tier or "PREMIER").strip().upper()
    if tier not in {"PREMIER", "MAIN", "AUCTION"}:
        return "PREMIER"
    return tier


def _load_market_tier_cache() -> dict[str, str]:
    global _MARKET_TIER_CACHE
    if _MARKET_TIER_CACHE is None:
        from app.services.eagle_eye.adapter import TickerChartAdapter

        adapter = TickerChartAdapter()
        _MARKET_TIER_CACHE = {
            stock.ticker.upper(): _normalize_market_tier(stock.market_tier)
            for stock in adapter.list_stocks()
        }
    return _MARKET_TIER_CACHE


def _next_trading_day(date_str: str) -> str:
    current = date.fromisoformat(date_str)
    while True:
        current += timedelta(days=1)
        if current.weekday() in KUWAIT_TRADING_WEEKDAYS:
            return current.isoformat()


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


def _compute_atr(ticker: str, date_str: str, periods: int = 14) -> Optional[float]:
    """Compute Average True Range over the last `periods` bars ending on date_str."""
    rows = _query_all(
        """SELECT high, low, close FROM ee_ohlcv_cache
           WHERE ticker = ? AND bar_date <= ?
           ORDER BY bar_date DESC LIMIT ?""",
        (ticker.upper(), date_str, periods + 1),
    )
    if not rows or len(rows) < 2:
        return None
    bars = list(reversed(rows))  # oldest first
    trs: list[float] = []
    for i in range(1, len(bars)):
        h = float(bars[i].get("high") or 0)
        lo = float(bars[i].get("low") or 0)
        prev_c = float(bars[i - 1].get("close") or 0)
        tr = max(h - lo, abs(h - prev_c), abs(lo - prev_c))
        trs.append(tr)
    return sum(trs) / len(trs) if trs else None


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

    def is_running(self) -> bool:
        row = _query_one("SELECT value FROM simulator_control WHERE key = 'enabled'", ())
        if row is None:
            return True
        value = str(dict(row.items()).get("value") if hasattr(row, "items") else row[0]).strip().lower()
        return value not in {"0", "false", "no", "off", "stopped"}

    def control_status(self) -> dict:
        self._ensure_simulator_tables()
        row = _query_one("SELECT value, updated_at FROM simulator_control WHERE key = 'enabled'", ())
        running = self.is_running()
        updated_at = None
        if row is not None:
            data = dict(row.items()) if hasattr(row, "items") else {"value": row[0], "updated_at": row[1] if len(row) > 1 else None}
            updated_at = data.get("updated_at")
        return {"running": running, "updated_at": updated_at}

    def set_running(self, enabled: bool) -> dict:
        self._ensure_simulator_tables()
        now_ts = _now_ts()
        value = "1" if enabled else "0"
        existing = _query_one("SELECT key FROM simulator_control WHERE key = 'enabled'", ())
        if existing:
            _exec("UPDATE simulator_control SET value = ?, updated_at = ? WHERE key = 'enabled'", (value, now_ts))
        else:
            _exec("INSERT INTO simulator_control (key, value, updated_at) VALUES ('enabled', ?, ?)", (value, now_ts))
        if not enabled:
            _exec(
                """UPDATE simulator_pending_orders
                      SET status = 'CANCELLED', updated_at = ?, notes_json = ?
                    WHERE status = 'PENDING'""",
                (now_ts, json.dumps({"cancel_reason": "SIMULATOR_STOPPED"})),
            )
        return self.control_status()

    def _assert_live_forward_date(self, as_of_date: Optional[date | str]) -> str:
        if isinstance(as_of_date, date):
            target_date = as_of_date
        elif isinstance(as_of_date, str) and as_of_date:
            target_date = date.fromisoformat(as_of_date)
        else:
            target_date = date.today()

        today = date.today()
        if target_date != today:
            raise ValueError(
                "historical replay disabled: no point-in-time ratings available — look-ahead unsafe"
            )
        return target_date.isoformat()

    def run_daily(self, run_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Called once per trading day after market close.
        For each strategy: exits → entries → snapshot.
        """
        from app.services.eagle_eye.store import ensure_tables
        ensure_tables()
        self._ensure_simulator_tables()

        if not self.is_running():
            return {
                "status": "stopped",
                "running": False,
                "processed_dates": [],
                "target_date": self._assert_live_forward_date(run_date),
            }

        if run_date is None:
            run_date = date.today()

        date_str = self._assert_live_forward_date(run_date)
        results: Dict[str, Any] = {}

        for strategy in STRATEGIES:
            portfolio = _get_portfolio(strategy.portfolio_id)
            if portfolio is None:
                logger.warning("Simulator: portfolio %d not found, skipping", strategy.portfolio_id)
                continue
            try:
                filled_exits = self._execute_pending_exit_orders(strategy, date_str)
                portfolio = _get_portfolio(strategy.portfolio_id) or portfolio
                filled_entries = self._execute_pending_entry_orders(strategy, date_str)
                portfolio = _get_portfolio(strategy.portfolio_id) or portfolio
                exits = self._process_exits(strategy, portfolio, date_str)
                portfolio = _get_portfolio(strategy.portfolio_id) or portfolio
                entries, liquidity_summary = self._process_entries(strategy, portfolio, date_str)
                snapshot = self._snapshot_portfolio(strategy, portfolio, date_str)
                results[strategy.name] = {
                    "filled_exits": filled_exits,
                    "filled_entries": filled_entries,
                    "exits": exits,
                    "entries": entries,
                    "date": date_str,
                    "liquidity_shrink_count": liquidity_summary["liquidity_shrink_count"],
                    "liquidity_skip_count": liquidity_summary["liquidity_skip_count"],
                    "max_drawdown_pct": snapshot["drawdown_from_peak_pct"],
                    "total_value_kwd": snapshot["total_value_kwd"],
                }
                logger.info(
                    "Simulator %s [%s]: %d filled exits, %d filled entries, %d exit signals, %d entry signals",
                    strategy.name,
                    date_str,
                    len(filled_exits),
                    len(filled_entries),
                    len(exits),
                    len(entries),
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
            if self._has_pending_exit(pos["id"]):
                continue

            ohlcv = _get_ohlcv(pos["ticker"], date_str)
            if ohlcv is None:
                # No same-day bar — do not infer from nearby dates.
                continue

            h = float(ohlcv["high"] or 0)
            l = float(ohlcv["low"] or 0)
            c = float(ohlcv["close"] or 0)
            entry_price = float(pos.get("entry_price") or 0)

            days_held = _trading_days_between(pos["entry_date"] or date_str, date_str)

            # Update MAE/MFE while still open
            self._update_excursion(pos, h, l)

            if entry_price <= 0:
                continue

            # 1) Hard stop-loss: close immediately if intraday low breaches -8%.
            stop_price = entry_price * (1.0 - (ML_STOP_LOSS_PCT / 100.0))
            if l <= stop_price:
                self._close_position(pos, portfolio, stop_price, "ML_STOP_LOSS_8", date_str, days_held)
                closed.append({"ticker": pos["ticker"], "reason": "ML_STOP_LOSS_8"})
                continue

            # 2) ML-based exits from current rating cache.
            current_rating = self._get_current_rating(pos["ticker"], date_str)
            if current_rating:
                current_label = str(current_rating.get("rating") or "").upper()
                current_confidence = float(current_rating.get("confidence") or 0)

                if current_label in {"SELL", "STRONG_SELL"}:
                    scheduled_date = _next_trading_day(date_str)
                    self._queue_exit_order(pos, 1.0, "ML_SELL_SIGNAL", date_str, scheduled_date)
                    closed.append(
                        {"ticker": pos["ticker"], "reason": "ML_SELL_SIGNAL", "scheduled_date": scheduled_date}
                    )
                    continue

                if current_confidence < ML_EXIT_CONFIDENCE_FLOOR:
                    scheduled_date = _next_trading_day(date_str)
                    self._queue_exit_order(pos, 1.0, "ML_CONFIDENCE_BREAK", date_str, scheduled_date)
                    closed.append(
                        {
                            "ticker": pos["ticker"],
                            "reason": "ML_CONFIDENCE_BREAK",
                            "scheduled_date": scheduled_date,
                        }
                    )
                    continue

            # 3) Optional partial take-profit at +20% (one-time 50% scale-out).
            if not bool(pos.get("tp1_hit")) and c > 0:
                pnl_pct = (c / entry_price - 1.0) * 100.0
                if pnl_pct >= ML_PARTIAL_TAKE_PROFIT_PCT:
                    scheduled_date = _next_trading_day(date_str)
                    self._queue_exit_order(
                        pos,
                        ML_PARTIAL_TAKE_PROFIT_FRACTION,
                        "ML_TAKE_PROFIT_20",
                        date_str,
                        scheduled_date,
                    )
                    closed.append(
                        {
                            "ticker": pos["ticker"],
                            "reason": "ML_TAKE_PROFIT_20_PARTIAL",
                            "scheduled_date": scheduled_date,
                        }
                    )
                    continue

        return closed

    # ── Entries ──────────────────────────────────────────────────────────

    def _process_entries(self, strategy: StrategyConfig, portfolio: dict, date_str: str) -> tuple[list, dict[str, int]]:
        opened = []
        liquidity_shrink_count = 0
        liquidity_skip_count = 0
        # Re-read portfolio after exits may have freed cash
        portfolio = _get_portfolio(strategy.portfolio_id) or portfolio
        todays_ratings = self._get_todays_ratings(date_str)

        for rating in todays_ratings:
            portfolio = _get_portfolio(strategy.portfolio_id) or portfolio
            decision = self._evaluate_entry(strategy, rating, portfolio, date_str)
            if not decision.should_enter:
                self._log_considered(strategy.portfolio_id, date_str, rating, decision.skip_reason)
                self._try_log_ml_signal(
                    (rating.get("ticker") or "").upper(), date_str, rating,
                    would_have_entered=(
                        decision.skip_reason
                        not in {"CONFIDENCE_BELOW_THRESHOLD", "RATING_NOT_BUY", "ML_SCORE_MISSING"}
                    ),
                    skip_reason=decision.skip_reason,
                )
                continue

            # Use actual market price from OHLCV cache (not stale rating target)
            ticker = (rating.get("ticker") or "").upper()
            ohlcv = _get_ohlcv(ticker, date_str)
            if ohlcv is None:
                self._log_considered(strategy.portfolio_id, date_str, rating, "NO_PRICE_DATA")
                self._try_log_ml_signal(ticker, date_str, rating, would_have_entered=True, skip_reason="NO_PRICE_DATA")
                continue
            actual_price = float(ohlcv.get("close") or 0)
            if actual_price <= 0:
                self._log_considered(strategy.portfolio_id, date_str, rating, "NO_PRICE_DATA")
                self._try_log_ml_signal(ticker, date_str, rating, would_have_entered=True, skip_reason="NO_PRICE_DATA")
                continue

            portfolio_value = float(portfolio.get("total_value_kwd") or STARTING_CAPITAL_KWD)
            requested_size_kwd = self._compute_position_size(rating, portfolio_value, actual_price, date_str)
            if requested_size_kwd < MIN_TRADE_SIZE_KWD:
                self._log_considered(strategy.portfolio_id, date_str, rating, "POSITION_TOO_SMALL")
                self._try_log_ml_signal(ticker, date_str, rating, would_have_entered=True, skip_reason="POSITION_TOO_SMALL")
                continue

            cash = float(portfolio.get("cash_balance_kwd") or 0)
            reserved_cash = self._pending_entry_reserved_cash(strategy.portfolio_id)
            available_cash = max(0.0, cash - reserved_cash)
            if requested_size_kwd > available_cash:
                requested_size_kwd = available_cash
            if requested_size_kwd < MIN_TRADE_SIZE_KWD:
                self._log_considered(strategy.portfolio_id, date_str, rating, "INSUFFICIENT_CASH")
                self._try_log_ml_signal(ticker, date_str, rating, would_have_entered=True, skip_reason="INSUFFICIENT_CASH")
                continue

            adtv_kwd = self._average_daily_traded_value_kwd(ticker, date_str)
            liquidity_cap_kwd = round(adtv_kwd * MAX_PARTICIPATION_RATE, 4) if adtv_kwd > 0 else 0.0
            approved_size_kwd = requested_size_kwd
            if liquidity_cap_kwd < MIN_TRADE_SIZE_KWD:
                liquidity_skip_count += 1
                logger.info(
                    "Simulator %s [%s] skip %s liquidity: requested=%.4f allowed=%.4f adtv=%.4f",
                    strategy.name,
                    date_str,
                    ticker,
                    requested_size_kwd,
                    liquidity_cap_kwd,
                    adtv_kwd,
                )
                self._log_considered(strategy.portfolio_id, date_str, rating, "LIQUIDITY_CAP_TOO_SMALL")
                self._try_log_ml_signal(
                    ticker,
                    date_str,
                    rating,
                    would_have_entered=True,
                    skip_reason="LIQUIDITY_CAP_TOO_SMALL",
                )
                continue
            if approved_size_kwd > liquidity_cap_kwd:
                approved_size_kwd = liquidity_cap_kwd
                liquidity_shrink_count += 1
                logger.info(
                    "Simulator %s [%s] shrink %s liquidity: requested=%.4f allowed=%.4f adtv=%.4f",
                    strategy.name,
                    date_str,
                    ticker,
                    requested_size_kwd,
                    approved_size_kwd,
                    adtv_kwd,
                )

            scheduled_date = _next_trading_day(date_str)
            self._queue_entry_order(
                strategy,
                rating,
                requested_size_kwd=requested_size_kwd,
                approved_size_kwd=approved_size_kwd,
                signal_date=date_str,
                scheduled_date=scheduled_date,
                signal_price=actual_price,
                avg_daily_traded_value_kwd=adtv_kwd,
                liquidity_cap_kwd=liquidity_cap_kwd,
            )
            self._try_log_ml_signal(ticker, date_str, rating, would_have_entered=True, skip_reason=None)
            opened.append(
                {
                    "ticker": ticker,
                    "requested_size_kwd": round(requested_size_kwd, 4),
                    "approved_size_kwd": round(approved_size_kwd, 4),
                    "signal_date": date_str,
                    "scheduled_entry_date": scheduled_date,
                    "signal_close": round(actual_price, 6),
                }
            )

        return opened, {
            "liquidity_shrink_count": liquidity_shrink_count,
            "liquidity_skip_count": liquidity_skip_count,
        }

    # ── Evaluation ───────────────────────────────────────────────────────

    def _evaluate_entry(
        self, strategy: StrategyConfig, rating: dict, portfolio: dict, date_str: str = ""
    ) -> EntryDecision:
        confidence = float(rating.get("confidence") or 0)
        stage = _normalize_stage(rating.get("stage"))
        rating_label = str(rating.get("rating") or "HOLD").upper()
        ticker = rating.get("ticker") or ""
        sector = rating.get("sector") or "UNKNOWN"

        required_confidence = max(ML_MIN_CONFIDENCE, float(strategy.min_confidence))
        if rating_label not in strategy.entry_ratings:
            return _skip("RATING_NOT_CARD_ENTRY")
        if confidence < required_confidence:
            return _skip("CONFIDENCE_BELOW_THRESHOLD")

        # Enforce per-strategy lifecycle stage rules.
        # This was defined in StrategyConfig but not applied in entry decisions.
        if stage and stage not in strategy.allowed_stages:
            return _skip("STAGE_NOT_ALLOWED")

        if self._already_holding(strategy.portfolio_id, ticker):
            return _skip("ALREADY_HOLDING")
        if self._has_pending_entry(strategy.portfolio_id, ticker):
            return _skip("ALREADY_PENDING_ENTRY")
        if self._recently_stopped_out(strategy.portfolio_id, ticker, date_str):
            return _skip("RECENTLY_STOPPED_OUT")

        cash = float(portfolio.get("cash_balance_kwd") or 0)
        available_cash = max(0.0, cash - self._pending_entry_reserved_cash(strategy.portfolio_id))
        if available_cash < MIN_TRADE_SIZE_KWD:
            return _skip("INSUFFICIENT_CASH")

        open_positions = _get_open_positions(strategy.portfolio_id)
        if len(open_positions) + self._pending_entry_count(strategy.portfolio_id) >= MAX_OPEN_POSITIONS:
            return _skip("MAX_POSITIONS_REACHED")

        portfolio_value = float(portfolio.get("total_value_kwd") or STARTING_CAPITAL_KWD)
        if _sector_exposure_pct(strategy.portfolio_id, sector, portfolio_value) >= SECTOR_CAP_PCT:
            return _skip("SECTOR_CAP_REACHED")

        # ── Volume gates ─────────────────────────────────────────────────
        vc = rating.get("volume_context") or {}
        if vc:
            if vc.get("liquidity_tier") == "ILLIQUID":
                return _skip("ILLIQUID_STOCK")
            if stage == "EARLY_BREAKOUT" and not vc.get("is_volume_confirmed", True):
                return _skip("BREAKOUT_WITHOUT_VOLUME_CONFIRMATION")
            if float(vc.get("relative_volume") or 1.0) < 0.5:
                return _skip("EXTREMELY_LOW_VOLUME_DAY")

        return _enter()

    # ── Position sizing ──────────────────────────────────────────────────

    def _compute_position_size(
        self, rating: dict, portfolio_value: float, actual_price: float = 0, date_str: str = ""
    ) -> float:
        from app.services.eagle_eye.rating_engine import compute_position_size

        entry = actual_price if actual_price > 0 else float(rating.get("entry_primary") or rating.get("last_price") or 0)
        entry_primary = float(rating.get("entry_primary") or 0)
        stop_from_rating = float(rating.get("stop_loss") or 0)
        confidence = float(rating.get("confidence") or 60)
        tp1_from_rating = float(rating.get("tp1") or 0)

        # Prefer ATR-based stop (2.5× ATR) — absorbs Kuwait market intraday noise
        ticker = (rating.get("ticker") or "").upper()
        atr = _compute_atr(ticker, date_str) if (ticker and date_str and entry > 0) else None
        if atr and atr > 0:
            stop = entry - (2.5 * atr)
            if stop <= 0:
                stop = entry * 0.90  # fallback: 10% max stop
        elif entry_primary > 0 and stop_from_rating > 0:
            stop_pct = (entry_primary - stop_from_rating) / entry_primary
            stop = entry * (1 - stop_pct)
        else:
            stop = entry * 0.93  # 7% default stop

        # Rescale TP1 to actual entry price
        if entry_primary > 0 and tp1_from_rating > 0:
            tp1_pct = (tp1_from_rating - entry_primary) / entry_primary
            tp1: Optional[float] = entry * (1 + tp1_pct)
        else:
            tp1 = None

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
        entry_date: str,
        actual_price: float = 0,
        signal_date: Optional[str] = None,
        market_tier: Optional[str] = None,
    ) -> dict:
        entry_primary = float(rating.get("entry_primary") or rating.get("last_price") or 0)
        stop_from_rating = float(rating.get("stop_loss") or 0)
        tp1_from_rating = float(rating.get("tp1") or 0)
        tp2_from_rating = float(rating.get("tp2") or 0)
        tp3_from_rating = float(rating.get("tp3") or 0)

        # Use actual market price; fall back to rating's entry_primary
        entry_price = actual_price if actual_price > 0 else entry_primary
        if entry_price <= 0:
            return

        # Rescale stop and TPs proportionally to the actual entry price
        def _rescale_level(level_from_rating: float, is_stop: bool) -> float:
            if entry_primary <= 0 or level_from_rating <= 0:
                return entry_price * (0.93 if is_stop else 0)
            pct = (level_from_rating - entry_primary) / entry_primary
            return entry_price * (1 + pct)

        # ATR-based stop (2.5× ATR) — wider than rating %, absorbs noise
        ticker_for_atr = (rating.get("ticker") or "").upper()
        atr = _compute_atr(ticker_for_atr, entry_date) if (ticker_for_atr and entry_date) else None
        if atr and atr > 0:
            planned_stop = entry_price - (2.5 * atr)
            if planned_stop <= 0:
                planned_stop = entry_price * 0.90
        else:
            planned_stop = _rescale_level(stop_from_rating, is_stop=True)
        planned_tp1 = _rescale_level(tp1_from_rating, is_stop=False) if tp1_from_rating > 0 else 0
        planned_tp2 = _rescale_level(tp2_from_rating, is_stop=False) if tp2_from_rating > 0 else 0
        planned_tp3 = _rescale_level(tp3_from_rating, is_stop=False) if tp3_from_rating > 0 else 0

        shares = round(size_kwd / entry_price, 4)
        actual_notional = round(shares * entry_price, 4)
        tier = self._market_tier_for_ticker(ticker_for_atr, rating, market_tier)
        entry_costs = self._leg_cost_breakdown(actual_notional, tier)
        portfolio_value = float(portfolio.get("total_value_kwd") or STARTING_CAPITAL_KWD)
        size_pct = (actual_notional / portfolio_value * 100) if portfolio_value > 0 else 0

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
                portfolio_id, ticker, status, signal_date, entry_date, entry_price,
                shares, shares_remaining, size_kwd, size_pct_of_portfolio,
                entry_confidence, entry_stage, entry_rating, entry_thesis,
                entry_signal_breakdown, entry_accumulation_score, entry_indicators_snapshot,
                planned_stop_loss, planned_tp1, planned_tp2, planned_tp3,
                tp1_hit, tp2_hit,
                max_unrealized_gain_pct, max_unrealized_loss_pct,
                entry_relative_volume,
                entry_market_tier, entry_cost_kwd, exit_cost_kwd, total_cost_kwd,
                realized_pnl_kwd, commission_paid_kwd, slippage_paid_kwd,
                created_at, updated_at
            ) VALUES (
                ?, ?, 'OPEN', ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                0, 0,
                0.0, 0.0,
                ?,
                ?, ?, 0.0, ?, ?, ?, ?,
                ?, ?
            )
            """,
            (
                strategy.portfolio_id,
                rating.get("ticker", "").upper(),
                signal_date or entry_date,
                entry_date,
                round(entry_price, 6),
                shares,
                shares,  # shares_remaining starts equal to shares
                actual_notional,
                round(size_pct, 4),
                float(rating.get("confidence") or 0),
                rating.get("stage") or "",
                rating.get("rating") or "",
                rating.get("thesis") or "",
                json.dumps(signals),
                round(accumulation_score, 4),
                json.dumps(indicators),
                round(planned_stop, 6),
                round(planned_tp1, 6),
                round(planned_tp2, 6),
                round(planned_tp3, 6),
                float((rating.get("volume_context") or {}).get("relative_volume") or 1.0),
                tier,
                round(entry_costs["total_cost_kwd"], 4),
                round(entry_costs["total_cost_kwd"], 4),
                round(-entry_costs["total_cost_kwd"], 4),
                round(entry_costs["commission_kwd"], 4),
                round(entry_costs["slippage_kwd"], 4),
                _now_ts(),
                _now_ts(),
            ),
        )

        # Deduct cash including entry-leg costs.
        new_cash = float(portfolio.get("cash_balance_kwd") or 0) - actual_notional - entry_costs["total_cost_kwd"]
        _exec(
            "UPDATE simulator_portfolios SET cash_balance_kwd = ?, updated_at = ? WHERE id = ?",
            (round(new_cash, 4), _now_ts(), strategy.portfolio_id),
        )

        return {
            "ticker": ticker_for_atr,
            "signal_date": signal_date or entry_date,
            "entry_date": entry_date,
            "entry_price": round(entry_price, 6),
            "size_kwd": actual_notional,
            "entry_cost_kwd": round(entry_costs["total_cost_kwd"], 4),
            "commission_kwd": round(entry_costs["commission_kwd"], 4),
            "slippage_kwd": round(entry_costs["slippage_kwd"], 4),
            "market_tier": tier,
        }

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
        prior_realized_pnl = float(pos.get("realized_pnl_kwd") or pos.get("pnl_kwd") or 0)
        prior_exit_cost = float(pos.get("exit_cost_kwd") or 0)
        prior_commission = float(pos.get("commission_paid_kwd") or 0)
        prior_slippage = float(pos.get("slippage_paid_kwd") or 0)
        tier = self._market_tier_for_ticker(pos.get("ticker"), pos, pos.get("entry_market_tier"))

        if entry_price <= 0:
            return

        gross_proceeds = shares_remaining * exit_price
        exit_costs = self._leg_cost_breakdown(gross_proceeds, tier)
        proceeds = gross_proceeds - exit_costs["total_cost_kwd"]
        cost_basis = shares_remaining * entry_price
        leg_pnl_kwd = proceeds - cost_basis
        pnl_kwd = prior_realized_pnl + leg_pnl_kwd
        pnl_pct = (pnl_kwd / cost_basis * 100) if cost_basis > 0 else 0
        exit_cost_total = prior_exit_cost + exit_costs["total_cost_kwd"]
        total_cost = float(pos.get("entry_cost_kwd") or 0) + exit_cost_total

        _exec(
            """
            UPDATE simulator_positions SET
                status = 'CLOSED', exit_date = ?, exit_price = ?,
                exit_reason = ?, pnl_kwd = ?, pnl_pct = ?,
                realized_pnl_kwd = ?, exit_cost_kwd = ?, total_cost_kwd = ?,
                commission_paid_kwd = ?, slippage_paid_kwd = ?,
                days_held = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                date_str, round(exit_price, 6),
                reason, round(pnl_kwd, 4), round(pnl_pct, 4),
                round(pnl_kwd, 4), round(exit_cost_total, 4), round(total_cost, 4),
                round(prior_commission + exit_costs["commission_kwd"], 4),
                round(prior_slippage + exit_costs["slippage_kwd"], 4),
                days_held, _now_ts(),
                pos["id"],
            ),
        )

        # Return net cash to portfolio after exit-leg costs.
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
        prior_realized_pnl = float(pos.get("realized_pnl_kwd") or pos.get("pnl_kwd") or 0)
        prior_exit_cost = float(pos.get("exit_cost_kwd") or 0)
        prior_commission = float(pos.get("commission_paid_kwd") or 0)
        prior_slippage = float(pos.get("slippage_paid_kwd") or 0)
        tier = self._market_tier_for_ticker(pos.get("ticker"), pos, pos.get("entry_market_tier"))

        if shares_to_close <= 0 or entry_price <= 0:
            return

        gross_proceeds = shares_to_close * exit_price
        exit_costs = self._leg_cost_breakdown(gross_proceeds, tier)
        proceeds = gross_proceeds - exit_costs["total_cost_kwd"]
        cost_basis = shares_to_close * entry_price
        partial_pnl = proceeds - cost_basis
        total_realized_pnl = prior_realized_pnl + partial_pnl
        total_exit_cost = prior_exit_cost + exit_costs["total_cost_kwd"]
        total_cost = float(pos.get("entry_cost_kwd") or 0) + total_exit_cost

        new_remaining = shares_remaining - shares_to_close

        # Mark TP hit flag
        tp_flag_col = "tp1_hit" if reason in {"TP1_HIT", "ML_TAKE_PROFIT_20"} else "tp2_hit"
        _exec(
            f"""
            UPDATE simulator_positions SET
                shares_remaining = ?,
                {tp_flag_col} = 1,
                pnl_kwd = ?, pnl_pct = ?, realized_pnl_kwd = ?,
                exit_cost_kwd = ?, total_cost_kwd = ?,
                commission_paid_kwd = ?, slippage_paid_kwd = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                round(new_remaining, 4),
                round(total_realized_pnl, 4),
                round((total_realized_pnl / float(pos.get("size_kwd") or 1)) * 100, 4),
                round(total_realized_pnl, 4),
                round(total_exit_cost, 4),
                round(total_cost, 4),
                round(prior_commission + exit_costs["commission_kwd"], 4),
                round(prior_slippage + exit_costs["slippage_kwd"], 4),
                _now_ts(),
                pos["id"],
            ),
        )

        # Return net partial proceeds to cash.
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
    ) -> dict:
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
        starting_capital = float(portfolio.get("starting_capital_kwd") or STARTING_CAPITAL_KWD)
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

        return {
            "cash_balance_kwd": round(cash, 4),
            "open_positions_value_kwd": round(open_value, 4),
            "total_value_kwd": round(total_value, 4),
            "daily_pnl_kwd": round(daily_pnl, 4),
            "cumulative_return_pct": round(cumulative_return_pct, 4),
            "drawdown_from_peak_pct": round(drawdown_pct, 4),
            "open_position_count": len(open_positions),
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    def _already_holding(self, portfolio_id: int, ticker: str) -> bool:
        count = _query_val(
            "SELECT COUNT(*) FROM simulator_positions WHERE portfolio_id = ? AND ticker = ? AND status = 'OPEN'",
            (portfolio_id, ticker.upper()),
        )
        return bool(count and int(count) > 0)

    def _has_pending_entry(self, portfolio_id: int, ticker: str) -> bool:
        count = _query_val(
            """SELECT COUNT(*) FROM simulator_pending_orders
               WHERE portfolio_id = ? AND ticker = ? AND order_kind = 'ENTRY' AND status = 'PENDING'""",
            (portfolio_id, ticker.upper()),
        )
        return bool(count and int(count) > 0)

    def _has_pending_exit(self, position_id: int) -> bool:
        count = _query_val(
            """SELECT COUNT(*) FROM simulator_pending_orders
               WHERE position_id = ? AND order_kind = 'EXIT' AND status = 'PENDING'""",
            (position_id,),
        )
        return bool(count and int(count) > 0)

    def _pending_entry_count(self, portfolio_id: int) -> int:
        count = _query_val(
            """SELECT COUNT(*) FROM simulator_pending_orders
               WHERE portfolio_id = ? AND order_kind = 'ENTRY' AND status = 'PENDING'""",
            (portfolio_id,),
        )
        return int(count or 0)

    def _pending_entry_reserved_cash(self, portfolio_id: int) -> float:
        rows = _query_all(
            """SELECT approved_size_kwd, market_tier FROM simulator_pending_orders
               WHERE portfolio_id = ? AND order_kind = 'ENTRY' AND status = 'PENDING'""",
            (portfolio_id,),
        )
        reserved = 0.0
        for row in rows:
            approved_size_kwd = float(row.get("approved_size_kwd") or 0.0)
            if approved_size_kwd <= 0:
                continue
            entry_cost = self._leg_cost_breakdown(approved_size_kwd, row.get("market_tier"))["total_cost_kwd"]
            reserved += approved_size_kwd + entry_cost
        return reserved

    def _average_daily_traded_value_kwd(self, ticker: str, date_str: str) -> float:
        rows = _query_all(
            """SELECT close, volume FROM ee_ohlcv_cache
               WHERE ticker = ? AND bar_date <= ?
               ORDER BY bar_date DESC LIMIT ?""",
            (ticker.upper(), date_str, ADTV_LOOKBACK_DAYS),
        )
        if not rows:
            return 0.0
        traded_values = [float(r.get("close") or 0) * float(r.get("volume") or 0) for r in rows]
        traded_values = [value for value in traded_values if value > 0]
        if not traded_values:
            return 0.0
        return sum(traded_values) / len(traded_values)

    def _market_tier_for_ticker(
        self,
        ticker: Optional[str],
        rating: Optional[dict] = None,
        explicit_tier: Optional[str] = None,
    ) -> str:
        if explicit_tier:
            return _normalize_market_tier(explicit_tier)
        if rating and rating.get("market_tier"):
            return _normalize_market_tier(str(rating.get("market_tier")))
        cache = _load_market_tier_cache()
        if ticker:
            return cache.get(str(ticker).upper(), "PREMIER")
        return "PREMIER"

    def _leg_cost_breakdown(self, notional_kwd: float, market_tier: Optional[str]) -> dict[str, float]:
        if notional_kwd <= 0:
            return {
                "commission_kwd": 0.0,
                "slippage_kwd": 0.0,
                "total_cost_kwd": 0.0,
            }

        from app.services.signal_engine.config.risk_config import TC_COMMISSION
        from app.services.signal_engine.engine.backtester import _total_cost_factor

        tier = _normalize_market_tier(market_tier)
        leg_cost_factor = _total_cost_factor(tier) / 2.0
        slippage_factor = max(0.0, leg_cost_factor - TC_COMMISSION)
        commission_kwd = notional_kwd * TC_COMMISSION
        slippage_kwd = notional_kwd * slippage_factor
        return {
            "commission_kwd": commission_kwd,
            "slippage_kwd": slippage_kwd,
            "total_cost_kwd": commission_kwd + slippage_kwd,
        }

    def _queue_entry_order(
        self,
        strategy: StrategyConfig,
        rating: dict,
        requested_size_kwd: float,
        approved_size_kwd: float,
        signal_date: str,
        scheduled_date: str,
        signal_price: float,
        avg_daily_traded_value_kwd: float,
        liquidity_cap_kwd: float,
    ) -> None:
        ticker = (rating.get("ticker") or "").upper()
        notes = {
            "requested_size_kwd": round(requested_size_kwd, 4),
            "approved_size_kwd": round(approved_size_kwd, 4),
            "signal_price": round(signal_price, 6),
            "liquidity_shrunk": approved_size_kwd < requested_size_kwd,
        }
        _exec(
            """
            INSERT INTO simulator_pending_orders (
                portfolio_id, ticker, order_kind, status,
                signal_date, scheduled_date,
                requested_size_kwd, approved_size_kwd,
                reason, signal_price, market_tier,
                avg_daily_traded_value_kwd, liquidity_cap_kwd,
                rating_snapshot_json, notes_json,
                created_at, updated_at
            ) VALUES (
                ?, ?, 'ENTRY', 'PENDING',
                ?, ?,
                ?, ?,
                'ENTRY_SIGNAL', ?, ?,
                ?, ?,
                ?, ?,
                ?, ?
            )
            """,
            (
                strategy.portfolio_id,
                ticker,
                signal_date,
                scheduled_date,
                round(requested_size_kwd, 4),
                round(approved_size_kwd, 4),
                round(signal_price, 6),
                self._market_tier_for_ticker(ticker, rating),
                round(avg_daily_traded_value_kwd, 4),
                round(liquidity_cap_kwd, 4),
                json.dumps(rating),
                json.dumps(notes),
                _now_ts(),
                _now_ts(),
            ),
        )

    def _queue_exit_order(
        self,
        pos: dict,
        fraction: float,
        reason: str,
        signal_date: str,
        scheduled_date: str,
    ) -> None:
        _exec(
            """
            INSERT INTO simulator_pending_orders (
                portfolio_id, position_id, ticker, order_kind, status,
                signal_date, scheduled_date, fraction,
                reason, market_tier, created_at, updated_at
            ) VALUES (
                ?, ?, ?, 'EXIT', 'PENDING',
                ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            (
                pos["portfolio_id"],
                pos["id"],
                pos["ticker"],
                signal_date,
                scheduled_date,
                round(fraction, 6),
                reason,
                self._market_tier_for_ticker(pos.get("ticker"), pos, pos.get("entry_market_tier")),
                _now_ts(),
                _now_ts(),
            ),
        )

    def _execute_pending_entry_orders(self, strategy: StrategyConfig, date_str: str) -> list:
        rows = _query_all(
            """SELECT * FROM simulator_pending_orders
               WHERE portfolio_id = ? AND order_kind = 'ENTRY'
                 AND status = 'PENDING' AND scheduled_date = ?
               ORDER BY id ASC""",
            (strategy.portfolio_id, date_str),
        )
        filled = []
        for order in rows:
            bar = _get_ohlcv(order["ticker"], date_str)
            if bar is None or float(bar.get("open") or 0) <= 0:
                logger.info(
                    "Simulator %s [%s] pending entry left unfilled for %s: no exact open bar",
                    strategy.name,
                    date_str,
                    order["ticker"],
                )
                continue

            portfolio = _get_portfolio(strategy.portfolio_id)
            if portfolio is None:
                break

            fill_price = float(bar.get("open") or 0)
            available_cash = float(portfolio.get("cash_balance_kwd") or 0)
            tier = self._market_tier_for_ticker(order.get("ticker"), None, order.get("market_tier"))
            target_notional = float(order.get("approved_size_kwd") or 0)
            leg_cost_factor = 0.0
            if target_notional > 0:
                leg_cost_factor = self._leg_cost_breakdown(target_notional, tier)["total_cost_kwd"] / target_notional
            affordable_notional = available_cash / (1.0 + leg_cost_factor) if (1.0 + leg_cost_factor) > 0 else 0.0
            fill_notional = min(target_notional, affordable_notional)
            if fill_notional < MIN_TRADE_SIZE_KWD:
                _exec(
                    "UPDATE simulator_pending_orders SET status = 'SKIPPED', updated_at = ?, notes_json = ? WHERE id = ?",
                    (
                        _now_ts(),
                        json.dumps({"skip_reason": "INSUFFICIENT_CASH_AT_FILL", "available_cash": round(available_cash, 4)}),
                        order["id"],
                    ),
                )
                continue

            rating = json.loads(order.get("rating_snapshot_json") or "{}")
            fill_result = self._open_position(
                strategy,
                rating,
                portfolio,
                fill_notional,
                entry_date=date_str,
                actual_price=fill_price,
                signal_date=order.get("signal_date"),
                market_tier=tier,
            )
            _exec(
                """UPDATE simulator_pending_orders SET
                       status = 'FILLED', fill_date = ?, fill_price = ?, approved_size_kwd = ?, updated_at = ?
                   WHERE id = ?""",
                (date_str, round(fill_price, 6), round(fill_result["size_kwd"], 4), _now_ts(), order["id"]),
            )
            filled.append(fill_result)
        return filled

    def _execute_pending_exit_orders(self, strategy: StrategyConfig, date_str: str) -> list:
        rows = _query_all(
            """SELECT * FROM simulator_pending_orders
               WHERE portfolio_id = ? AND order_kind = 'EXIT'
                 AND status = 'PENDING' AND scheduled_date = ?
               ORDER BY id ASC""",
            (strategy.portfolio_id, date_str),
        )
        executed = []
        for order in rows:
            row = _query_one("SELECT * FROM simulator_positions WHERE id = ?", (order.get("position_id"),))
            if row is None:
                _exec("UPDATE simulator_pending_orders SET status = 'CANCELLED', updated_at = ? WHERE id = ?", (_now_ts(), order["id"]))
                continue
            pos = dict(row.items())
            if pos.get("status") != "OPEN":
                _exec("UPDATE simulator_pending_orders SET status = 'CANCELLED', updated_at = ? WHERE id = ?", (_now_ts(), order["id"]))
                continue

            bar = _get_ohlcv(pos["ticker"], date_str)
            if bar is None or float(bar.get("open") or 0) <= 0:
                logger.info(
                    "Simulator %s [%s] pending exit left unfilled for %s: no exact open bar",
                    strategy.name,
                    date_str,
                    pos["ticker"],
                )
                continue

            portfolio = _get_portfolio(strategy.portfolio_id)
            if portfolio is None:
                break

            fill_price = float(bar.get("open") or 0)
            days_held = _trading_days_between(pos.get("entry_date") or date_str, date_str)
            fraction = float(order.get("fraction") or 1.0)
            if fraction >= 0.999999:
                self._close_position(pos, portfolio, fill_price, order.get("reason") or "NEXT_OPEN_EXIT", date_str, days_held)
                executed.append({
                    "ticker": pos["ticker"],
                    "fill_date": date_str,
                    "fill_price": round(fill_price, 6),
                    "reason": order.get("reason") or "NEXT_OPEN_EXIT",
                    "signal_date": order.get("signal_date"),
                })
            else:
                self._partial_close(pos, portfolio, fill_price, fraction, order.get("reason") or "NEXT_OPEN_PARTIAL", date_str)
                executed.append({
                    "ticker": pos["ticker"],
                    "fill_date": date_str,
                    "fill_price": round(fill_price, 6),
                    "reason": order.get("reason") or "NEXT_OPEN_PARTIAL",
                    "fraction": round(fraction, 4),
                    "signal_date": order.get("signal_date"),
                })

            _exec(
                "UPDATE simulator_pending_orders SET status = 'FILLED', fill_date = ?, fill_price = ?, updated_at = ? WHERE id = ?",
                (date_str, round(fill_price, 6), _now_ts(), order["id"]),
            )

        return executed

    def _recently_stopped_out(self, portfolio_id: int, ticker: str, date_str: str, lookback_days: int = 15) -> bool:
        """Return True if this ticker had a STOP_HIT exit within the last ~3 weeks."""
        try:
            d = date.fromisoformat(date_str) if date_str else date.today()
        except Exception:
            d = date.today()
        # Convert trading days to calendar days (roughly 7/5)
        cutoff = (d - timedelta(days=int(lookback_days * 7 / 5))).isoformat()
        count = _query_val(
            """SELECT COUNT(*) FROM simulator_positions
               WHERE portfolio_id = ? AND ticker = ? AND status = 'CLOSED'
                 AND exit_reason = 'STOP_HIT' AND exit_date >= ?""",
            (portfolio_id, ticker.upper(), cutoff),
        )
        return bool(count and int(count) > 0)

    def _get_current_rating(self, ticker: str, date_str: Optional[str] = None) -> Optional[dict]:
        self._assert_live_forward_date(date_str)

        # Prefer today's computed snapshot to keep simulator actions strictly daily.
        # Fall back to latest row only when computed_date is unavailable in schema.
        if _column_exists("ee_ratings_cache", "computed_date") and date_str:
            row = _query_one(
                """SELECT stage, rating, confidence, ml_score, last_price, market_tier, computed_date, computed_at
                   FROM ee_ratings_cache
                   WHERE ticker = ? AND computed_date = ?""",
                (ticker.upper(), date_str),
            )
            if row is not None:
                return dict(row.items())

        row = _query_one(
            """SELECT stage, rating, confidence, ml_score, last_price, market_tier, computed_date, computed_at
               FROM ee_ratings_cache
               WHERE ticker = ?""",
            (ticker.upper(),),
        )
        return dict(row.items()) if row else None

    def _get_todays_ratings(self, date_str: str) -> List[dict]:
        """Load all rated stocks for live-forward trading on the current date only."""
        self._assert_live_forward_date(date_str)

        if _column_exists("ee_ratings_cache", "computed_date"):
            rows = _query_all(
                """SELECT ticker, name_en, sector, stage, rating, confidence, ml_score, thesis,
                          entry_primary, stop_loss, tp1, tp2, tp3, last_price,
                          market_tier,
                          signals_json, indicators_json, volume_context_json, computed_date, computed_at
                   FROM   ee_ratings_cache
                   WHERE  computed_date = ?
                   ORDER  BY confidence DESC""",
                (date_str,),
            )
        else:
            rows = _query_all(
                """SELECT ticker, name_en, sector, stage, rating, confidence, ml_score, thesis,
                          entry_primary, stop_loss, tp1, tp2, tp3, last_price,
                          market_tier,
                          signals_json, indicators_json, volume_context_json, computed_at
                   FROM   ee_ratings_cache
                   ORDER  BY confidence DESC""",
                (),
            )

        if not rows:
            logger.warning(
                "Simulator daily run skipped entries: no ee_ratings_cache snapshot for %s",
                date_str,
            )

        result = []
        for r in rows:
            indicators = r.get("indicators_json")
            if isinstance(indicators, str):
                try:
                    r["indicators_json"] = json.loads(indicators)
                except Exception:
                    r["indicators_json"] = {}
            vc_raw = r.get("volume_context_json")
            try:
                r["volume_context"] = json.loads(vc_raw) if vc_raw else {}
            except Exception:
                r["volume_context"] = {}
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

    # Mapping from simulator internal skip reasons to ML signal-logger vocabulary.
    _SKIP_TO_ML: Dict[str, str] = {
        "CONFIDENCE_BELOW_THRESHOLD": "BELOW_CONFIDENCE_THRESHOLD",
        "RATING_NOT_BUY": "OTHER",
        "ML_SCORE_MISSING": "OTHER",
        "STAGE_NOT_ALLOWED": "STAGE_NOT_ALLOWED",
        "SECTOR_CAP_REACHED": "SECTOR_CAP_REACHED",
        "ILLIQUID_STOCK": "LIQUIDITY_GATE",
        "BREAKOUT_WITHOUT_VOLUME_CONFIRMATION": "LIQUIDITY_GATE",
        "EXTREMELY_LOW_VOLUME_DAY": "LIQUIDITY_GATE",
    }

    def reset_all(self, reset_date: Optional[date | str] = None) -> dict:
        """Clear simulator state and restart all portfolios from today."""
        self._ensure_simulator_tables()
        date_str = self._assert_live_forward_date(reset_date)

        from app.core.config import get_settings

        settings = get_settings()
        if settings.use_postgres:
            table_rows = _query_all(
                """SELECT table_name AS name
                     FROM information_schema.tables
                    WHERE table_schema = current_schema()
                      AND table_type = 'BASE TABLE'
                      AND table_name LIKE 'simulator_%'
                    ORDER BY table_name""",
                (),
            )
        else:
            table_rows = _query_all(
                """SELECT name
                     FROM sqlite_master
                    WHERE type = 'table' AND name LIKE 'simulator_%'
                    ORDER BY name""",
                (),
            )

        simulator_tables = []
        for row in table_rows:
            table_name = str(row.get("name") or "").strip()
            if re.fullmatch(r"simulator_[a-z0-9_]+", table_name):
                simulator_tables.append(table_name)

        cleared_rows: dict[str, int] = {}
        for table_name in simulator_tables:
            if table_name in {"simulator_portfolios", "simulator_control"}:
                continue

            count_row = _query_one(f"SELECT COUNT(*) AS n FROM {table_name}", ())
            count_val = int(dict(count_row.items()).get("n") or 0) if count_row else 0
            _exec(f"DELETE FROM {table_name}", ())
            cleared_rows[table_name] = count_val

        now_ts = _now_ts()
        portfolio_summaries = []

        for strategy in STRATEGIES:
            existing = _get_portfolio(strategy.portfolio_id)
            starting_capital = STARTING_CAPITAL_KWD

            if existing:
                _exec(
                    """UPDATE simulator_portfolios
                          SET strategy_name = ?,
                              starting_capital_kwd = ?,
                              cash_balance_kwd = ?,
                              total_value_kwd = ?,
                              updated_at = ?
                        WHERE id = ?""",
                    (
                        strategy.name,
                        round(starting_capital, 4),
                        round(starting_capital, 4),
                        round(starting_capital, 4),
                        now_ts,
                        strategy.portfolio_id,
                    ),
                )
            else:
                _exec(
                    """INSERT INTO simulator_portfolios (
                           id, strategy_name, starting_capital_kwd,
                           cash_balance_kwd, total_value_kwd, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        strategy.portfolio_id,
                        strategy.name,
                        round(starting_capital, 4),
                        round(starting_capital, 4),
                        round(starting_capital, 4),
                        now_ts,
                        now_ts,
                    ),
                )

            _exec(
                """INSERT INTO simulator_daily_snapshots (
                       portfolio_id, date, cash_balance_kwd, open_positions_value_kwd,
                       total_value_kwd, daily_pnl_kwd, cumulative_return_pct,
                       drawdown_from_peak_pct, open_position_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(portfolio_id, date) DO UPDATE SET
                       cash_balance_kwd = excluded.cash_balance_kwd,
                       open_positions_value_kwd = excluded.open_positions_value_kwd,
                       total_value_kwd = excluded.total_value_kwd,
                       daily_pnl_kwd = excluded.daily_pnl_kwd,
                       cumulative_return_pct = excluded.cumulative_return_pct,
                       drawdown_from_peak_pct = excluded.drawdown_from_peak_pct,
                       open_position_count = excluded.open_position_count""",
                (
                    strategy.portfolio_id,
                    date_str,
                    round(starting_capital, 4),
                    0.0,
                    round(starting_capital, 4),
                    0.0,
                    0.0,
                    0.0,
                    0,
                ),
            )

            portfolio_summaries.append(
                {
                    "portfolio_id": strategy.portfolio_id,
                    "strategy_name": strategy.name,
                    "starting_capital_kwd": round(starting_capital, 4),
                }
            )

        return {
            "date": date_str,
            "running": self.is_running(),
            "tables_found": simulator_tables,
            "cleared_rows": cleared_rows,
            "portfolios": portfolio_summaries,
        }

    def _try_log_ml_signal(
        self,
        ticker: str,
        date_str: str,
        rating: dict,
        would_have_entered: bool,
        skip_reason: Optional[str],
    ) -> None:
        """Observation-only hook — writes to ML considered_signals table.

        Errors are caught and logged; they must never block entry decisions.
        """
        try:
            from app.services.eagle_eye.ml import log_considered_signal as _log_sig
            ml_reason = self._SKIP_TO_ML.get(skip_reason or "", "OTHER") if skip_reason else None
            features = {k: v for k, v in rating.items() if k != "ticker"}
            _log_sig(
                ticker=ticker,
                signal_date=date_str,
                rule_score=float(rating.get("confidence") or 0),
                would_have_entered=would_have_entered,
                skip_reason=ml_reason,
                features=features,
            )
        except Exception as _exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "log_considered_signal failed for %s/%s: %s", ticker, date_str, _exc
            )

    def _ensure_simulator_tables(self) -> None:
        """Ensure tables exist (idempotent — called on first run)."""
        _exec(
            """CREATE TABLE IF NOT EXISTS simulator_portfolios (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               strategy_name TEXT NOT NULL,
               starting_capital_kwd REAL NOT NULL DEFAULT 100000,
               cash_balance_kwd REAL NOT NULL DEFAULT 100000,
               total_value_kwd REAL NOT NULL DEFAULT 100000,
               created_at TEXT, updated_at TEXT
            )""",
        )
        _exec(
            """CREATE TABLE IF NOT EXISTS simulator_positions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               portfolio_id INTEGER NOT NULL,
               ticker TEXT NOT NULL,
               status TEXT NOT NULL DEFAULT 'OPEN',
               signal_date TEXT,
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
               entry_relative_volume NUMERIC(8,2),
               entry_market_tier TEXT,
               entry_cost_kwd REAL DEFAULT 0,
               exit_cost_kwd REAL DEFAULT 0,
               total_cost_kwd REAL DEFAULT 0,
               realized_pnl_kwd REAL DEFAULT 0,
               commission_paid_kwd REAL DEFAULT 0,
               slippage_paid_kwd REAL DEFAULT 0,
               created_at TEXT, updated_at TEXT
            )""",
        )
        # Additive migration for tables created before entry_relative_volume was added
        from app.core.database import add_column_if_missing as _acim
        _acim("simulator_positions", "signal_date", "TEXT")
        _acim("simulator_positions", "entry_relative_volume", "NUMERIC(8,2)")
        _acim("simulator_positions", "entry_market_tier", "TEXT")
        _acim("simulator_positions", "entry_cost_kwd", "REAL DEFAULT 0")
        _acim("simulator_positions", "exit_cost_kwd", "REAL DEFAULT 0")
        _acim("simulator_positions", "total_cost_kwd", "REAL DEFAULT 0")
        _acim("simulator_positions", "realized_pnl_kwd", "REAL DEFAULT 0")
        _acim("simulator_positions", "commission_paid_kwd", "REAL DEFAULT 0")
        _acim("simulator_positions", "slippage_paid_kwd", "REAL DEFAULT 0")
        _exec(
            """CREATE TABLE IF NOT EXISTS simulator_pending_orders (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               portfolio_id INTEGER NOT NULL,
               position_id INTEGER,
               ticker TEXT NOT NULL,
               order_kind TEXT NOT NULL,
               status TEXT NOT NULL DEFAULT 'PENDING',
               signal_date TEXT NOT NULL,
               scheduled_date TEXT NOT NULL,
               fill_date TEXT,
               requested_size_kwd REAL,
               approved_size_kwd REAL,
               fraction REAL,
               reason TEXT,
               signal_price REAL,
               fill_price REAL,
               market_tier TEXT,
               avg_daily_traded_value_kwd REAL,
               liquidity_cap_kwd REAL,
               rating_snapshot_json TEXT,
               notes_json TEXT,
               created_at TEXT,
               updated_at TEXT
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
        _exec(
            """CREATE TABLE IF NOT EXISTS simulator_control (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL,
               updated_at TEXT
            )""",
        )
        if _query_one("SELECT key FROM simulator_control WHERE key = 'enabled'", ()) is None:
            _exec(
                "INSERT INTO simulator_control (key, value, updated_at) VALUES ('enabled', '1', ?)",
                (_now_ts(),),
            )
        # Seed or repair the canonical strategy rows. Existing paper state is
        # only re-capitalized when that card has no positions or pending orders.
        now = _now_ts()
        for strat in STRATEGIES:
            existing = _get_portfolio(strat.portfolio_id)
            if existing:
                active_count = _query_val(
                    """SELECT COUNT(*) FROM simulator_positions WHERE portfolio_id = ?""",
                    (strat.portfolio_id,),
                ) or 0
                pending_count = _query_val(
                    """SELECT COUNT(*) FROM simulator_pending_orders WHERE portfolio_id = ?""",
                    (strat.portfolio_id,),
                ) or 0
                if int(active_count) == 0 and int(pending_count) == 0:
                    _exec(
                        """UPDATE simulator_portfolios
                              SET strategy_name = ?, starting_capital_kwd = ?,
                                  cash_balance_kwd = ?, total_value_kwd = ?, updated_at = ?
                            WHERE id = ?""",
                        (
                            strat.name,
                            STARTING_CAPITAL_KWD,
                            STARTING_CAPITAL_KWD,
                            STARTING_CAPITAL_KWD,
                            now,
                            strat.portfolio_id,
                        ),
                    )
                else:
                    _exec(
                        "UPDATE simulator_portfolios SET strategy_name = ?, updated_at = ? WHERE id = ?",
                        (strat.name, now, strat.portfolio_id),
                    )
            else:
                _exec(
                    """INSERT INTO simulator_portfolios
                       (id, strategy_name, starting_capital_kwd, cash_balance_kwd, total_value_kwd, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        strat.portfolio_id,
                        strat.name,
                        STARTING_CAPITAL_KWD,
                        STARTING_CAPITAL_KWD,
                        STARTING_CAPITAL_KWD,
                        now,
                        now,
                    ),
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


def run_daily_simulation(date_str: Optional[str] = None) -> Dict[str, Any]:
    """Public wrapper used by scripts/tests to run one live-forward simulation day."""
    if date_str:
        return _engine.run_daily(date.fromisoformat(date_str))
    return _engine.run_daily()


def reset_simulator_data(date_str: Optional[str] = None) -> dict:
    """Public wrapper used by scripts/admin routes to reset simulator state."""
    return _engine.reset_all(date_str)
