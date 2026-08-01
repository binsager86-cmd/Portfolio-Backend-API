"""
Eagle Eye Simulator Service.

Canonical executable-action mapping (source: ee_ratings_cache.rating):
  STRONG_BUY  -> BUY       (explicit buy signal)
  BUY         -> BUY       (explicit buy signal)
  HOLD        -> HOLD      (no new order)
  NEUTRAL     -> HOLD      (no action)
  WATCHLIST   -> HOLD      (monitoring only, confirmation required; NOT a BUY)
  REDUCE      -> HOLD      (advisory; NOT a SELL order)
  AVOID       -> HOLD      (avoidance advice; NOT a SELL order)
  SELL        -> SELL      (explicit sell signal)
  STRONG_SELL -> SELL      (explicit sell signal)
  <any other> -> HOLD      (safe default)

Signal source status values:
  STORED_POINT_IN_TIME_REPLAY         - immutable signals from ratings_history table
  FORWARD_PAPER_SIMULATION            - signals stored prospectively (ee_forward_signals)
  RECONSTRUCTED_RESEARCH_SIMULATION   - NOT available without owner directive
  CURRENT_SIGNAL_ORDER_PREVIEW        - latest signals only; no historical return
  HISTORICAL_SIGNAL_DATA_UNAVAILABLE  - no suitable signal data found

authoritative_model_version : CONCEPT_VERSION from market_data_service (e.g. ee-2.1.2)
rating_engine_fingerprint   : code_fingerprint column in ee_ratings_cache
simulation_engine_version   : ee-sim-1.0.0
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from app.core.database import exec_sql, query_one, query_all
from simulation.domain.models import (
    SimulationConfig, SimulationResult, EagleEyeRatingRecord,
    OHLCV, EagleEyeRating, WyckoffPhase, PositionSizingMode, ExecutionRule,
)
from simulation.engine.simulator import SimulationEngine

logger = logging.getLogger(__name__)

SIMULATION_ENGINE_VERSION = "ee-sim-1.0.0"


# ---------------------------------------------------------------------------
# Model/engine version helpers
# ---------------------------------------------------------------------------

def _get_authoritative_model_version() -> str:
    try:
        from app.services.eagle_eye import market_data_service
        return getattr(market_data_service, "CONCEPT_VERSION", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def _get_current_rating_engine_fingerprint() -> str:
    try:
        row = query_one(
            "SELECT code_fingerprint FROM ee_ratings_cache "
            "WHERE code_fingerprint IS NOT NULL ORDER BY computed_date DESC LIMIT 1"
        )
        return row[0] if row and row[0] else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Canonical action mapping
# ---------------------------------------------------------------------------

def produce_action_mapping() -> Dict[str, str]:
    """
    Return the canonical rating->executable_action mapping.

    WATCHLIST, REDUCE, AVOID are explicitly NOT mapped to BUY or SELL.
    The owner must issue a separate directive to change these mappings.
    """
    return {
        "STRONG_BUY":  "BUY",
        "BUY":         "BUY",
        "HOLD":        "HOLD",
        "NEUTRAL":     "HOLD",
        "WATCHLIST":   "HOLD",   # monitoring only
        "REDUCE":      "HOLD",   # advisory, not a SELL order
        "AVOID":       "HOLD",   # avoidance advice, not a SELL order
        "SELL":        "SELL",
        "STRONG_SELL": "SELL",
    }


_ACTION_MAP: Dict[str, str] = produce_action_mapping()


def rating_to_enum(rating_str: str) -> EagleEyeRating:
    """
    Map stored rating string to EagleEyeRating enum.
    WATCHLIST/REDUCE/AVOID -> HOLD (no order generated).
    """
    _MAP: Dict[str, EagleEyeRating] = {
        "STRONG_BUY":  EagleEyeRating.STRONG_BUY,
        "BUY":         EagleEyeRating.BUY,
        "HOLD":        EagleEyeRating.HOLD,
        "NEUTRAL":     EagleEyeRating.NEUTRAL,
        "WATCHLIST":   EagleEyeRating.HOLD,
        "REDUCE":      EagleEyeRating.HOLD,
        "AVOID":       EagleEyeRating.HOLD,
        "SELL":        EagleEyeRating.SELL,
        "STRONG_SELL": EagleEyeRating.STRONG_SELL,
    }
    mapped = _MAP.get(rating_str)
    if mapped is None:
        logger.warning("Unknown rating '%s' -> treating as HOLD", rating_str)
        return EagleEyeRating.HOLD
    return mapped


def stage_to_enum(stage_str: str) -> WyckoffPhase:
    _MAP: Dict[str, WyckoffPhase] = {
        "ACCUMULATION":          WyckoffPhase.STEALTH_ACCUMULATION,
        "EARLY_MARKUP":          WyckoffPhase.EARLY_BREAKOUT,
        "MARKUP":                WyckoffPhase.MARKUP_TRENDING,
        "DISTRIBUTION":          WyckoffPhase.CLIMAX,
        "MARKDOWN":              WyckoffPhase.CAPITULATION,
        "NEUTRAL_AMBIGUOUS":     WyckoffPhase.NEUTRAL,
        "NEUTRAL":               WyckoffPhase.NEUTRAL,
        "INACTIVE_OR_DELISTED":  WyckoffPhase.DORMANT,
        "INSUFFICIENT_HISTORY":  WyckoffPhase.DORMANT,
        "INDICATOR_UNAVAILABLE": WyckoffPhase.DORMANT,
    }
    return _MAP.get(stage_str, WyckoffPhase.NEUTRAL)


# ---------------------------------------------------------------------------
# Configuration hash
# ---------------------------------------------------------------------------

def create_config_hash(
    start_date: date,
    end_date: date,
    initial_cash: Decimal,
    max_positions: int,
    position_sizing_mode: str,
    commission_pct: Decimal,
    slippage_pct: Decimal,
    universe: Optional[List[str]] = None,
) -> str:
    """SHA-256 hash of simulation configuration for idempotency."""
    payload = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "initial_cash": str(initial_cash),
        "max_positions": max_positions,
        "position_sizing_mode": position_sizing_mode,
        "commission_pct": str(commission_pct),
        "slippage_pct": str(slippage_pct),
        "universe": sorted(universe) if universe else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Forward-signal table (FORWARD_PAPER_SIMULATION)
# ---------------------------------------------------------------------------

def ensure_forward_signal_table() -> None:
    """
    Create ee_forward_signals table.

    UNIQUE constraint (ticker, effective_ts, rating_engine_fingerprint) ensures
    that repeated scanner runs are fully idempotent (INSERT OR IGNORE).
    """
    exec_sql("""
        CREATE TABLE IF NOT EXISTS ee_forward_signals (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker                      TEXT    NOT NULL,
            rating                      TEXT    NOT NULL,
            executable_action           TEXT    NOT NULL,
            stage                       TEXT,
            confidence                  REAL,
            ml_score                    REAL,
            entry_primary               REAL,
            stop_loss                   REAL,
            computed_date               TEXT    NOT NULL,
            rating_calc_ts              TEXT    NOT NULL,
            effective_ts                TEXT    NOT NULL,
            data_cutoff_ts              TEXT,
            rating_engine_fingerprint   TEXT,
            authoritative_model_version TEXT,
            signal_source_status        TEXT    NOT NULL DEFAULT 'FORWARD_PAPER_SIMULATION',
            created_at                  TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE (ticker, effective_ts, rating_engine_fingerprint)
        )
    """)
    exec_sql(
        "CREATE INDEX IF NOT EXISTS ix_ee_fwd_date ON ee_forward_signals(computed_date)"
    )
    exec_sql(
        "CREATE INDEX IF NOT EXISTS ix_ee_fwd_ticker ON ee_forward_signals(ticker, computed_date)"
    )


def snapshot_forward_signals(computed_date_str: Optional[str] = None) -> int:
    """
    Save current ee_ratings_cache rows into ee_forward_signals.

    Guarantees:
      - Idempotent: INSERT OR IGNORE, repeated calls produce no duplicates.
      - Never overwrites existing historical records.
      - Future-dated effective_ts are rejected.
      - executable_action derived from canonical mapping only.

    Returns count of newly inserted rows.
    """
    ensure_forward_signal_table()
    if computed_date_str is None:
        computed_date_str = date.today().isoformat()
    now_ts = datetime.utcnow().isoformat()
    auth_version = _get_authoritative_model_version()

    rows = query_all(
        """
        SELECT ticker, rating, stage, confidence, ml_score,
               entry_primary, stop_loss, computed_at,
               code_fingerprint, computed_date
        FROM ee_ratings_cache
        WHERE computed_date = ?
        """,
        (computed_date_str,),
    )

    # Count rows before insertion to compute actual new inserts
    count_before = (
        query_one(
            "SELECT COUNT(*) FROM ee_forward_signals WHERE computed_date = ?",
            (computed_date_str,),
        ) or (0,)
    )[0]

    for row in rows:
        (
            ticker, rating_str, stage_str, confidence, ml_score,
            entry_primary, stop_loss, computed_at_str, code_fp, comp_date,
        ) = row

        executable_action = _ACTION_MAP.get(rating_str, "HOLD")
        rating_calc_ts = computed_at_str or now_ts
        effective_ts = rating_calc_ts

        # Reject future effective_ts (no look-ahead)
        try:
            eff_date = datetime.fromisoformat(
                effective_ts.replace("Z", "+00:00")
            ).date()
            if eff_date > date.today():
                logger.warning(
                    "Rejecting future signal for %s (effective_ts=%s)",
                    ticker, effective_ts,
                )
                continue
        except Exception:
            pass  # If parse fails, allow it

        try:
            exec_sql(
                """
                INSERT OR IGNORE INTO ee_forward_signals (
                    ticker, rating, executable_action, stage, confidence, ml_score,
                    entry_primary, stop_loss, computed_date,
                    rating_calc_ts, effective_ts, data_cutoff_ts,
                    rating_engine_fingerprint, authoritative_model_version,
                    signal_source_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ticker, rating_str, executable_action,
                    stage_str, confidence, ml_score,
                    entry_primary, stop_loss, comp_date,
                    rating_calc_ts, effective_ts, now_ts,
                    code_fp, auth_version,
                    "FORWARD_PAPER_SIMULATION",
                ),
            )
        except Exception as e:
            logger.warning("Could not insert forward signal for %s: %s", ticker, e)

    # Measure actual new rows inserted (INSERT OR IGNORE means 0 when already exists)
    count_after = (
        query_one(
            "SELECT COUNT(*) FROM ee_forward_signals WHERE computed_date = ?",
            (computed_date_str,),
        ) or (0,)
    )[0]
    inserted = int(count_after) - int(count_before)

    logger.info(
        "snapshot_forward_signals: %d new rows inserted for %s",
        inserted, computed_date_str,
    )
    return inserted


def get_first_forward_signal_date() -> Optional[str]:
    """Return the earliest computed_date in ee_forward_signals, or None."""
    ensure_forward_signal_table()
    row = query_one(
        "SELECT MIN(computed_date) FROM ee_forward_signals "
        "WHERE executable_action IN ('BUY','SELL')"
    )
    return row[0] if row and row[0] else None


# ---------------------------------------------------------------------------
# OHLCV loading
# ---------------------------------------------------------------------------

def load_ohlcv(
    start_date: date,
    end_date: date,
    universe: Optional[List[str]] = None,
) -> List[OHLCV]:
    """Load OHLCV bars from ee_ohlcv_cache."""
    query = (
        "SELECT ticker, bar_date, open, high, low, close, volume "
        "FROM ee_ohlcv_cache "
        "WHERE bar_date >= ? AND bar_date <= ?"
    )
    params: List[Any] = [str(start_date), str(end_date)]
    if universe:
        placeholders = ",".join("?" for _ in universe)
        query += f" AND ticker IN ({placeholders})"
        params.extend(universe)
    query += " ORDER BY bar_date, ticker"

    result: List[OHLCV] = []
    for row in query_all(query, tuple(params)):
        try:
            result.append(
                OHLCV(
                    symbol=row[0],
                    date=datetime.strptime(row[1], "%Y-%m-%d").date(),
                    open_price=Decimal(str(row[2] or 0)),
                    high=Decimal(str(row[3] or 0)),
                    low=Decimal(str(row[4] or 0)),
                    close=Decimal(str(row[5] or 0)),
                    volume=int(row[6] or 0),
                    source="EE_OHLCV_CACHE",
                )
            )
        except Exception as e:
            logger.warning("Bad OHLCV row %s: %s", row, e)
    return result


# ---------------------------------------------------------------------------
# Signal loading
# ---------------------------------------------------------------------------

def load_forward_ratings(
    start_date: date,
    end_date: date,
    universe: Optional[List[str]] = None,
) -> Tuple[List[EagleEyeRatingRecord], str, str]:
    """
    Load executable signals from ee_forward_signals.
    Only rows with executable_action IN ('BUY','SELL') are returned.
    """
    ensure_forward_signal_table()

    query = (
        "SELECT ticker, computed_date, stage, rating, confidence, "
        "       rating_engine_fingerprint, effective_ts "
        "FROM ee_forward_signals "
        "WHERE computed_date >= ? AND computed_date <= ? "
        "  AND executable_action IN ('BUY','SELL')"
    )
    params: List[Any] = [str(start_date), str(end_date)]
    if universe:
        placeholders = ",".join("?" for _ in universe)
        query += f" AND ticker IN ({placeholders})"
        params.extend(universe)
    query += " ORDER BY computed_date, ticker"

    records: List[EagleEyeRatingRecord] = []
    fps: set = set()

    for row in query_all(query, tuple(params)):
        try:
            ticker, comp_date, stage, rating_str, conf, fp, effective_ts = row
            fps.add(fp or "UNKNOWN")
            records.append(
                EagleEyeRatingRecord(
                    symbol=ticker,
                    rating_date=datetime.strptime(comp_date, "%Y-%m-%d").date(),
                    rating_timestamp=effective_ts,
                    rating=rating_to_enum(rating_str),
                    confidence=Decimal(str(conf or 0)),
                    stage=stage_to_enum(stage or ""),
                    thesis=f"Forward signal {comp_date}",
                )
            )
        except Exception as e:
            logger.warning("Bad forward signal row %s: %s", row, e)

    fp_str = ";".join(sorted(fps)) if fps else "UNKNOWN"
    status = "FORWARD_PAPER_SIMULATION" if records else "HISTORICAL_SIGNAL_DATA_UNAVAILABLE"
    return records, status, fp_str


def load_stored_historical_ratings(
    start_date: date,
    end_date: date,
    universe: Optional[List[str]] = None,
) -> Tuple[List[EagleEyeRatingRecord], str, str]:
    """
    Load from ratings_history (immutable point-in-time snapshots).
    Status is STORED_POINT_IN_TIME_REPLAY if rows found.
    """
    query = (
        "SELECT ticker, computed_date, stage, rating, confidence "
        "FROM ratings_history "
        "WHERE computed_date >= ? AND computed_date <= ?"
    )
    params: List[Any] = [str(start_date), str(end_date)]
    if universe:
        placeholders = ",".join("?" for _ in universe)
        query += f" AND ticker IN ({placeholders})"
        params.extend(universe)
    query += " ORDER BY computed_date, ticker"

    try:
        rows = query_all(query, tuple(params))
    except Exception:
        rows = []

    records: List[EagleEyeRatingRecord] = []
    for row in rows:
        try:
            ticker, comp_date, stage, rating_str, conf = row
            records.append(
                EagleEyeRatingRecord(
                    symbol=ticker,
                    rating_date=datetime.strptime(comp_date, "%Y-%m-%d").date(),
                    rating_timestamp=None,
                    rating=rating_to_enum(rating_str),
                    confidence=Decimal(str(conf or 0)),
                    stage=stage_to_enum(stage or ""),
                    thesis=f"Historical signal {comp_date}",
                )
            )
        except Exception as e:
            logger.warning("Bad historical rating row %s: %s", row, e)

    status = (
        "STORED_POINT_IN_TIME_REPLAY"
        if records
        else "HISTORICAL_SIGNAL_DATA_UNAVAILABLE"
    )
    return records, status, _get_current_rating_engine_fingerprint()


# ---------------------------------------------------------------------------
# Persistence schema
# ---------------------------------------------------------------------------

def ensure_simulator_tables() -> None:
    """Create all simulator tables with full schema and indexes."""
    exec_sql("""
        CREATE TABLE IF NOT EXISTS ee_simulations (
            run_id                       TEXT PRIMARY KEY,
            status                       TEXT NOT NULL DEFAULT 'PENDING',
            signal_source_status         TEXT NOT NULL DEFAULT 'HISTORICAL_SIGNAL_DATA_UNAVAILABLE',
            config_json                  TEXT,
            config_hash                  TEXT,
            initial_cash                 TEXT,
            authoritative_model_version  TEXT,
            rating_engine_fingerprint    TEXT,
            simulation_engine_version    TEXT,
            signal_source_table          TEXT,
            data_cutoff_ts               TEXT,
            ending_equity                TEXT,
            ending_cash                  TEXT,
            total_return_pct             TEXT,
            max_drawdown_pct             TEXT,
            realized_pnl                 TEXT,
            unrealized_pnl               TEXT,
            total_commissions            TEXT,
            total_slippage               TEXT,
            trades_count                 INTEGER DEFAULT 0,
            win_rate_pct                 TEXT,
            profit_factor                TEXT,
            buy_signals_executed         INTEGER DEFAULT 0,
            sell_signals_executed        INTEGER DEFAULT 0,
            buy_signals_skipped          INTEGER DEFAULT 0,
            sell_signals_skipped         INTEGER DEFAULT 0,
            cash_recon_ok                INTEGER DEFAULT 0,
            cash_recon_error             TEXT,
            equity_recon_ok              INTEGER DEFAULT 0,
            equity_recon_error           TEXT,
            truncation_parity_ok         INTEGER,
            validation_warnings_json     TEXT,
            error_message                TEXT,
            created_at                   TEXT NOT NULL DEFAULT (datetime('now')),
            completed_at                 TEXT,
            execution_seconds            REAL
        )
    """)
    exec_sql("CREATE INDEX IF NOT EXISTS ix_ee_sim_created ON ee_simulations(created_at DESC)")
    exec_sql("CREATE INDEX IF NOT EXISTS ix_ee_sim_hash    ON ee_simulations(config_hash)")

    exec_sql("""
        CREATE TABLE IF NOT EXISTS ee_simulations_daily (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,
            date            TEXT NOT NULL,
            cash            TEXT,
            invested_value  TEXT,
            total_equity    TEXT,
            positions_count INTEGER DEFAULT 0,
            UNIQUE(run_id, date),
            FOREIGN KEY(run_id) REFERENCES ee_simulations(run_id) ON DELETE CASCADE
        )
    """)
    exec_sql("CREATE INDEX IF NOT EXISTS ix_ee_sim_daily ON ee_simulations_daily(run_id, date)")

    exec_sql("""
        CREATE TABLE IF NOT EXISTS ee_simulations_trades (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              TEXT NOT NULL,
            symbol              TEXT NOT NULL,
            entry_date          TEXT,
            entry_price         TEXT,
            exit_date           TEXT,
            exit_price          TEXT,
            quantity            TEXT,
            cost_basis          TEXT,
            realized_pnl_gross  TEXT,
            realized_pnl_net    TEXT,
            realized_pnl_pct    TEXT,
            commission          TEXT,
            slippage            TEXT,
            holding_days        INTEGER DEFAULT 0,
            signal_rating       TEXT,
            FOREIGN KEY(run_id) REFERENCES ee_simulations(run_id) ON DELETE CASCADE
        )
    """)

    exec_sql("""
        CREATE TABLE IF NOT EXISTS ee_simulations_orders (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              TEXT NOT NULL,
            symbol              TEXT NOT NULL,
            side                TEXT NOT NULL,
            signal_date         TEXT,
            execution_date      TEXT,
            execution_price     TEXT,
            quantity_requested  TEXT,
            quantity_filled     TEXT,
            gross_amount        TEXT,
            commission          TEXT,
            slippage            TEXT,
            status              TEXT,
            rejection_reason    TEXT,
            FOREIGN KEY(run_id) REFERENCES ee_simulations(run_id) ON DELETE CASCADE
        )
    """)

    exec_sql("""
        CREATE TABLE IF NOT EXISTS ee_simulations_skipped (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            signal_date     TEXT,
            signal_rating   TEXT,
            reason          TEXT,
            qty_requested   TEXT,
            FOREIGN KEY(run_id) REFERENCES ee_simulations(run_id) ON DELETE CASCADE
        )
    """)


def save_simulation_result(
    result: SimulationResult,
    signal_source_status: str,
    rating_engine_fingerprint: str,
    signal_source_table: str,
    config_hash: str,
) -> None:
    """Persist all simulation artefacts."""
    ensure_simulator_tables()
    auth_version = _get_authoritative_model_version()

    exec_sql("""
        INSERT OR REPLACE INTO ee_simulations (
            run_id, status, signal_source_status,
            config_json, config_hash, initial_cash,
            authoritative_model_version, rating_engine_fingerprint,
            simulation_engine_version, signal_source_table, data_cutoff_ts,
            ending_equity, ending_cash, total_return_pct, max_drawdown_pct,
            realized_pnl, unrealized_pnl, total_commissions, total_slippage,
            trades_count, win_rate_pct, profit_factor,
            buy_signals_executed, sell_signals_executed,
            buy_signals_skipped, sell_signals_skipped,
            cash_recon_ok, cash_recon_error, equity_recon_ok, equity_recon_error,
            validation_warnings_json, error_message,
            created_at, completed_at, execution_seconds
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        result.run_id, result.status, signal_source_status,
        json.dumps(result.config.to_dict(), default=str), config_hash,
        str(result.config.initial_cash),
        auth_version, rating_engine_fingerprint,
        SIMULATION_ENGINE_VERSION, signal_source_table, date.today().isoformat(),
        str(result.ending_equity), str(result.ending_cash),
        str(result.total_return_pct), str(result.max_drawdown_pct),
        str(result.realized_pnl), str(result.unrealized_pnl),
        str(result.total_commissions), str(result.total_slippage),
        result.trades_count, str(result.win_rate_pct), str(result.profit_factor),
        result.buy_signals_executed, result.sell_signals_executed,
        result.buy_signals_skipped, result.sell_signals_skipped,
        int(result.cash_reconciliation_ok), str(result.cash_reconciliation_error),
        int(result.equity_reconciliation_ok), str(result.equity_reconciliation_error),
        json.dumps(result.validation_warnings or []), result.error_message,
        str(result.created_at), str(result.completed_at),
        result.execution_seconds,
    ))

    for daily in result.daily_records:
        exec_sql("""
            INSERT OR IGNORE INTO ee_simulations_daily
                (run_id, date, cash, invested_value, total_equity, positions_count)
            VALUES (?,?,?,?,?,?)
        """, (
            result.run_id, str(daily.date), str(daily.cash),
            str(daily.invested_value), str(daily.total_equity),
            daily.positions_count,
        ))

    for trade in result.trades:
        exec_sql("""
            INSERT INTO ee_simulations_trades
                (run_id, symbol, entry_date, entry_price, exit_date, exit_price,
                 quantity, cost_basis, realized_pnl_gross, realized_pnl_net,
                 realized_pnl_pct, commission, slippage, holding_days, signal_rating)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            result.run_id, trade.symbol,
            str(trade.entry_date), str(trade.entry_price),
            str(trade.exit_date) if trade.exit_date else None,
            str(trade.exit_price) if trade.exit_price else None,
            str(trade.quantity),
            str(getattr(trade, "cost_basis", "")),
            str(trade.realized_pnl_gross),
            str(getattr(trade, "realized_pnl_net", trade.realized_pnl_gross)),
            str(trade.realized_pnl_pct),
            str(getattr(trade, "commission", "")),
            str(getattr(trade, "slippage", "")),
            trade.holding_days,
            trade.signal_rating.value if trade.signal_rating else None,
        ))

    for order in result.orders:
        exec_sql("""
            INSERT INTO ee_simulations_orders
                (run_id, symbol, side, signal_date, execution_date, execution_price,
                 quantity_requested, quantity_filled, gross_amount,
                 commission, slippage, status, rejection_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            result.run_id, order.symbol, order.side.value,
            str(order.signal_date),
            str(order.execution_date) if order.execution_date else None,
            str(order.execution_price) if order.execution_price else None,
            str(order.quantity_requested), str(order.quantity_filled),
            str(order.gross_amount), str(order.commission), str(order.slippage),
            order.status.value, order.rejection_reason or None,
        ))

    for sk in result.skipped_signals:
        exec_sql("""
            INSERT INTO ee_simulations_skipped
                (run_id, symbol, signal_date, signal_rating, reason, qty_requested)
            VALUES (?,?,?,?,?,?)
        """, (
            result.run_id, sk.symbol, str(sk.signal_date),
            sk.signal_rating.value if sk.signal_rating else None,
            sk.reason,
            str(sk.quantity_requested) if sk.quantity_requested else None,
        ))

    logger.info("Persisted run %s (%s)", result.run_id, signal_source_status)


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

def run_simulation(request: Any) -> Dict[str, Any]:
    """
    Execute simulation using stored signals.

    Signal source priority:
      1. ratings_history -> STORED_POINT_IN_TIME_REPLAY
      2. ee_forward_signals -> FORWARD_PAPER_SIMULATION

    CURRENT_RATING_DEMO / retrospective demo is NOT supported.
    Using current ratings applied to past dates is temporally invalid.
    """
    auth_version = _get_authoritative_model_version()
    config_hash = create_config_hash(
        start_date=request.start_date,
        end_date=request.end_date,
        initial_cash=request.initial_cash,
        max_positions=request.max_positions,
        position_sizing_mode=request.position_sizing_mode,
        commission_pct=request.commission_pct,
        slippage_pct=request.slippage_pct,
        universe=request.universe,
    )

    # Priority 1: stored historical
    hist_ratings, hist_status, hist_fp = load_stored_historical_ratings(
        request.start_date, request.end_date, request.universe
    )
    if hist_ratings:
        ratings = hist_ratings
        signal_source_status = "STORED_POINT_IN_TIME_REPLAY"
        rating_engine_fp = hist_fp
        signal_source_table = "ratings_history"
    else:
        # Priority 2: forward signals
        fwd_ratings, fwd_status, fwd_fp = load_forward_ratings(
            request.start_date, request.end_date, request.universe
        )
        ratings = fwd_ratings
        signal_source_status = fwd_status
        rating_engine_fp = fwd_fp
        signal_source_table = "ee_forward_signals"

    if not ratings:
        return {
            "run_id": "N/A",
            "status": "FAILED",
            "signal_source_status": "HISTORICAL_SIGNAL_DATA_UNAVAILABLE",
            "error_message": (
                "Historical point-in-time Eagle Eye signals are unavailable. "
                "Historical performance cannot currently be calculated. "
                "Forward paper tracking can begin from the activation date."
            ),
            "authoritative_model_version": auth_version,
            "simulation_engine_version": SIMULATION_ENGINE_VERSION,
        }

    ohlcv = load_ohlcv(request.start_date, request.end_date, request.universe)
    if not ohlcv:
        return {
            "run_id": "N/A",
            "status": "FAILED",
            "signal_source_status": signal_source_status,
            "error_message": "No OHLCV data for requested date range.",
        }

    sizing_map = {
        "equal":      PositionSizingMode.EQUAL_ALLOCATION,
        "fixed":      PositionSizingMode.FIXED_AMOUNT,
        "percentage": PositionSizingMode.PERCENTAGE_EQUITY,
    }
    cfg = SimulationConfig(
        start_date=request.start_date,
        end_date=request.end_date,
        initial_cash=request.initial_cash,
        max_concurrent_positions=request.max_positions,
        position_sizing_mode=sizing_map.get(
            request.position_sizing_mode, PositionSizingMode.EQUAL_ALLOCATION
        ),
        commission_pct=request.commission_pct,
        slippage_pct=request.slippage_pct,
        execution_rule=ExecutionRule.NEXT_SESSION_OPEN,
        allow_pyramiding=getattr(request, "allow_pyramiding", False),
        model_version=auth_version,
        data_cutoff_date=date.today(),
    )

    engine = SimulationEngine(cfg)
    engine.load_ratings(ratings)
    engine.load_ohlcv(ohlcv)
    result = engine.run()

    save_simulation_result(
        result,
        signal_source_status=signal_source_status,
        rating_engine_fingerprint=rating_engine_fp,
        signal_source_table=signal_source_table,
        config_hash=config_hash,
    )

    # Historical return metrics are only valid for STORED_POINT_IN_TIME_REPLAY
    is_historical = signal_source_status == "STORED_POINT_IN_TIME_REPLAY"
    return {
        "run_id": result.run_id,
        "status": result.status,
        "signal_source_status": signal_source_status,
        "error_message": result.error_message,
        "authoritative_model_version": auth_version,
        "rating_engine_fingerprint": rating_engine_fp,
        "simulation_engine_version": SIMULATION_ENGINE_VERSION,
        "config_hash": config_hash,
        "ending_equity": str(result.ending_equity),
        "ending_cash": str(result.ending_cash),
        "total_return_pct": str(result.total_return_pct) if is_historical else None,
        "max_drawdown_pct": str(result.max_drawdown_pct) if is_historical else None,
        "win_rate_pct": str(result.win_rate_pct) if is_historical else None,
        "profit_factor": str(result.profit_factor) if is_historical else None,
        "trades_count": result.trades_count,
        "buy_signals_executed": result.buy_signals_executed,
        "sell_signals_executed": result.sell_signals_executed,
        "cash_reconciliation_ok": result.cash_reconciliation_ok,
        "equity_reconciliation_ok": result.equity_reconciliation_ok,
        "validation_warnings": result.validation_warnings,
        "created_at": str(result.created_at),
        "completed_at": str(result.completed_at),
        "execution_seconds": result.execution_seconds,
    }


# Convenience alias for tests
class SimulatorService:
    create_config_hash = staticmethod(create_config_hash)
    snapshot_forward_signals = staticmethod(snapshot_forward_signals)
    get_first_forward_signal_date = staticmethod(get_first_forward_signal_date)
    load_ohlcv = staticmethod(load_ohlcv)
    produce_action_mapping = staticmethod(produce_action_mapping)
    run_simulation = staticmethod(run_simulation)
