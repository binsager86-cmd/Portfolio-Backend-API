"""
ml/data_pipeline.py — Phase 1: Filtering, deduplication, and orchestration.

Implements Section 1.2 (filtering) and Section 1.3 (deduplication) from the
ML brief, with every action logged to data_lineage_log.

Entry point
-----------
    from app.services.eagle_eye.ml.data_pipeline import DataPipeline

    pipeline = DataPipeline()
    eligible = pipeline.run_eligibility_screen()   # evaluates all stocks
    report   = pipeline.build_feature_matrix(ticker="ZAIN", version="v1")

Per-stock eligibility rules (Section 1.2):
    - Skip if move events < MIN_MOVE_EVENTS (50)
    - Skip if trading days < MIN_TRADING_DAYS (500)
    - Skip if liquidity tier == ILLIQUID
    - Flag WATCH_ONLY if median daily volume < WATCH_ONLY_VOLUME_THRESHOLD
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.services.eagle_eye.store import load_ohlcv, list_tickers_with_ohlcv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Eligibility constants  (documented here, referenced in brief Section 1.2)
# ---------------------------------------------------------------------------

#: Minimum move events (≥10% directional) for a stock to be ML-eligible.
#: Rationale: fewer than 50 events → insufficient signal for a reliable model.
MIN_MOVE_EVENTS = 50

#: Minimum clean trading days needed.
#: Rationale: <500 bars → < 2 years of data; walk-forward CV needs more.
MIN_TRADING_DAYS = 500

#: Liquidity tiers that are excluded from ML training.
ILLIQUID_TIERS = {"ILLIQUID", "SUSPENDED", "DELISTED", "AUCTION"}

#: Stocks with median daily volume (in shares) below this are WATCH_ONLY.
#: Rationale: fills at any reasonable size become impossible below this level.
WATCH_ONLY_VOLUME_THRESHOLD = 25_000

#: Maximum consecutive NaN days before flagging a gap.
MAX_GAP_DAYS = 5

#: Maximum fraction of missing feature values before dropping a row.
MAX_ROW_MISSING_RATE = 0.20


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class StockEligibility:
    ticker: str
    eligible: bool
    reason: str
    n_move_events: int = 0
    n_trading_days: int = 0
    liquidity_tier: str = "UNKNOWN"
    median_daily_vol: float = 0.0
    watch_only: bool = False


@dataclass
class PipelineReport:
    ticker: str
    ran_at: str
    n_raw_rows: int
    n_rows_after_dedup: int
    n_rows_after_filter: int
    n_rows_dropped_missing: int
    n_data_gaps_flagged: int
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """
    Orchestrates data ingestion, filtering, deduplication, and label building
    for the Eagle Eye ML training pipeline.

    All actions are logged to data_lineage_log.
    """

    def __init__(self, feature_version: str = "v1") -> None:
        self.feature_version = feature_version

    # ------------------------------------------------------------------
    # Eligibility screening
    # ------------------------------------------------------------------

    def run_eligibility_screen(
        self, tickers: Optional[List[str]] = None
    ) -> List[StockEligibility]:
        """
        Evaluate all (or specified) tickers and persist results to
        ml_stock_eligibility.

        Returns list of StockEligibility records sorted by eligible first.
        """
        if tickers is None:
            tickers = list_tickers_with_ohlcv()

        results: list[StockEligibility] = []
        for ticker in tickers:
            elig = self._evaluate_eligibility(ticker)
            self._persist_eligibility(elig)
            results.append(elig)

        n_eligible = sum(1 for r in results if r.eligible)
        logger.info(
            "Eligibility screen complete: %d/%d stocks eligible",
            n_eligible, len(results),
        )
        results_sorted = sorted(results, key=lambda r: (not r.eligible, r.ticker))

        # Auto-generate the coverage report (Addendum A.1)
        try:
            from app.services.eagle_eye.ml.eligibility_report import generate_eligibility_report
            generate_eligibility_report(results_sorted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Eligibility report generation failed (non-fatal): %s", exc)

        return results_sorted

    def _evaluate_eligibility(self, ticker: str) -> StockEligibility:
        """Evaluate one stock against all hard-filter rules."""
        from app.services.eagle_eye.move_detector import detect_moves

        try:
            ohlcv = load_ohlcv(ticker)
        except Exception as exc:  # noqa: BLE001
            return StockEligibility(
                ticker=ticker, eligible=False,
                reason=f"OHLCV load failed: {exc}",
            )

        if ohlcv is None or ohlcv.empty:
            return StockEligibility(
                ticker=ticker, eligible=False, reason="NO_OHLCV",
            )

        n_days = len(ohlcv)
        median_vol = float(ohlcv["volume"].median()) if "volume" in ohlcv.columns else 0.0
        liquidity_tier = self._infer_liquidity_tier(ohlcv, median_vol)

        # Rule 1: illiquid
        if liquidity_tier in ILLIQUID_TIERS:
            self._log_lineage("eligibility", "FILTER", ticker, 1,
                              f"Illiquid tier={liquidity_tier}")
            return StockEligibility(
                ticker=ticker, eligible=False,
                reason=f"ILLIQUID_TIER:{liquidity_tier}",
                n_trading_days=n_days,
                liquidity_tier=liquidity_tier,
                median_daily_vol=median_vol,
            )

        # Rule 2: insufficient trading history
        if n_days < MIN_TRADING_DAYS:
            self._log_lineage("eligibility", "FILTER", ticker, 1,
                              f"Only {n_days} trading days (need {MIN_TRADING_DAYS})")
            return StockEligibility(
                ticker=ticker, eligible=False,
                reason=f"INSUFFICIENT_HISTORY:{n_days}d",
                n_trading_days=n_days,
                liquidity_tier=liquidity_tier,
                median_daily_vol=median_vol,
            )

        # Rule 3: move event count
        try:
            moves = detect_moves(ticker, ohlcv)
            n_moves = len(moves)
        except Exception as exc:  # noqa: BLE001
            n_moves = 0
            logger.warning("Move detection failed for %s: %s", ticker, exc)

        if n_moves < MIN_MOVE_EVENTS:
            self._log_lineage("eligibility", "FILTER", ticker, 1,
                              f"Only {n_moves} move events (need {MIN_MOVE_EVENTS})")
            return StockEligibility(
                ticker=ticker, eligible=False,
                reason=f"INSUFFICIENT_MOVES:{n_moves}",
                n_move_events=n_moves,
                n_trading_days=n_days,
                liquidity_tier=liquidity_tier,
                median_daily_vol=median_vol,
            )

        # WATCH_ONLY flag
        watch_only = median_vol < WATCH_ONLY_VOLUME_THRESHOLD
        if watch_only:
            self._log_lineage("eligibility", "FLAG", ticker, 1,
                              f"WATCH_ONLY: median vol={median_vol:.0f} < {WATCH_ONLY_VOLUME_THRESHOLD}")

        self._log_lineage("eligibility", "INGEST", ticker, 1,
                          f"Eligible: {n_moves} moves, {n_days}d, tier={liquidity_tier}")
        return StockEligibility(
            ticker=ticker, eligible=True,
            reason="OK",
            n_move_events=n_moves,
            n_trading_days=n_days,
            liquidity_tier=liquidity_tier,
            median_daily_vol=median_vol,
            watch_only=watch_only,
        )

    def _infer_liquidity_tier(self, ohlcv: pd.DataFrame, median_vol: float) -> str:
        """Simple heuristic tier from volume until real market-tier data is available."""
        if median_vol == 0:
            return "ILLIQUID"
        if median_vol < 5_000:
            return "ILLIQUID"
        if median_vol < 50_000:
            return "MAIN"
        return "PREMIER"

    # ------------------------------------------------------------------
    # Feature matrix assembly for one stock
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self, ticker: str, version: str = "v1"
    ) -> PipelineReport:
        """
        Build a clean, deduplicated feature matrix for one stock and
        save it to the FeatureStore.

        Steps:
        1. Load OHLCV + compute indicators
        2. Build move events → label rows
        3. Add supplementary features (market context, corporate events, DNA)
        4. Deduplicate on (ticker, event_date)
        5. Drop rows with > 20% missing features
        6. Flag large data gaps
        7. Save to FeatureStore
        """
        from app.services.eagle_eye.ml.feature_builder import (
            build_feature_matrix as _build_raw,
            get_feature_columns,
        )
        from app.services.eagle_eye.ml.feature_store import FeatureStore
        from app.services.eagle_eye.ml.market_context import MarketContextBuilder
        from app.services.eagle_eye.ml.labelers import build_labels

        ran_at = datetime.utcnow().isoformat()
        warnings: list[str] = []

        try:
            # ── Step 1: raw feature matrix from existing builder ──────
            raw_df = _build_raw(ticker)
        except Exception as exc:  # noqa: BLE001
            return PipelineReport(
                ticker=ticker, ran_at=ran_at,
                n_raw_rows=0, n_rows_after_dedup=0,
                n_rows_after_filter=0, n_rows_dropped_missing=0,
                n_data_gaps_flagged=0, error=str(exc),
            )

        if raw_df is None or raw_df.empty:
            return PipelineReport(
                ticker=ticker, ran_at=ran_at,
                n_raw_rows=0, n_rows_after_dedup=0,
                n_rows_after_filter=0, n_rows_dropped_missing=0,
                n_data_gaps_flagged=0, error="Empty feature matrix",
            )

        n_raw = len(raw_df)
        self._log_lineage("feature_builder", "INGEST", ticker, n_raw)

        # ── Step 2: deduplication (Section 1.3) ───────────────────────
        date_col = "event_date" if "event_date" in raw_df.columns else None
        if date_col:
            before = len(raw_df)
            raw_df = raw_df.sort_values(date_col).drop_duplicates(
                subset=["ticker", date_col] if "ticker" in raw_df.columns else [date_col],
                keep="last",
            ).reset_index(drop=True)
            n_dupes = before - len(raw_df)
            if n_dupes:
                self._log_lineage("dedup", "DEDUP", ticker, n_dupes,
                                  f"Removed {n_dupes} duplicate rows on (ticker, event_date)")

        n_after_dedup = len(raw_df)

        # ── Step 3: add market context features ───────────────────────
        try:
            ctx = MarketContextBuilder()
            raw_df = ctx.enrich(raw_df, date_col=date_col or "event_date")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"market context enrich failed: {exc}")

        # ── Step 3b: add Kuwait macro features (Addendum A.3) ────────
        try:
            from app.services.eagle_eye.ml.macro_features import MacroFeatureBuilder
            stock_close = None
            if "close" in raw_df.columns and date_col in raw_df.columns:
                import pandas as _pd
                _tmp = raw_df.set_index(date_col)["close"]
                _tmp.index = _pd.to_datetime(_tmp.index, errors="coerce")
                stock_close = _tmp.sort_index()
            macro = MacroFeatureBuilder()
            raw_df = macro.enrich(
                raw_df,
                date_col=date_col or "event_date",
                stock_close=stock_close,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"macro features enrich failed: {exc}")

        # ── Step 4: drop high-missingness rows (Section 1.2) ──────────
        feat_cols = get_feature_columns(raw_df)
        if feat_cols:
            miss_rates = raw_df[feat_cols].isna().mean(axis=1)
            mask_keep = miss_rates <= MAX_ROW_MISSING_RATE
            n_dropped_missing = int((~mask_keep).sum())
            if n_dropped_missing:
                self._log_lineage("missingness_filter", "DROP", ticker,
                                  n_dropped_missing,
                                  f"Rows dropped: >{MAX_ROW_MISSING_RATE:.0%} features missing")
                raw_df = raw_df[mask_keep].reset_index(drop=True)
        else:
            n_dropped_missing = 0

        n_after_filter = len(raw_df)

        # ── Step 5: flag large data gaps ──────────────────────────────
        n_gaps = 0
        if date_col and date_col in raw_df.columns:
            dates = pd.to_datetime(raw_df[date_col], errors="coerce").dropna()
            if len(dates) > 1:
                gaps = (dates.diff().dt.days > MAX_GAP_DAYS * 7).sum()
                n_gaps = int(gaps)
                if n_gaps:
                    warnings.append(
                        f"{n_gaps} data gaps > {MAX_GAP_DAYS} trading days detected"
                    )

        # ── Step 6: save to feature store ────────────────────────────
        fs = FeatureStore()
        fs.save(
            ticker=ticker, version=version, df=raw_df,
            meta={"n_raw_rows": n_raw, "built_at": ran_at},
        )
        self._log_lineage("feature_store", "INGEST", ticker, n_after_filter,
                          f"Saved to feature store version={version}")

        return PipelineReport(
            ticker=ticker, ran_at=ran_at,
            n_raw_rows=n_raw,
            n_rows_after_dedup=n_after_dedup,
            n_rows_after_filter=n_after_filter,
            n_rows_dropped_missing=n_dropped_missing,
            n_data_gaps_flagged=n_gaps,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Deduplication for corporate events (Section 1.3)
    # ------------------------------------------------------------------

    def dedup_corporate_events(self) -> int:
        """
        Remove duplicate corporate event rows per the dedup key:
        (stock_ticker, event_type, announcement_date).
        Returns number of rows removed.
        """
        from app.core.database import exec_sql, exec_sql_fetchone

        # SQLite: delete duplicates keeping the max id for each key
        exec_sql(
            """
            DELETE FROM ml_corporate_events
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM ml_corporate_events
                GROUP BY stock_ticker, event_type, announcement_date
            )
            """,
            (),
        )
        # We cannot easily get rowcount portably, so return 0 as sentinel
        self._log_lineage("dedup_corporate_events", "DEDUP", None, 0,
                          "Deduplicated ml_corporate_events on (ticker, type, announcement_date)")
        return 0

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_eligibility(self, elig: StockEligibility) -> None:
        try:
            from app.core.database import exec_sql
            exec_sql(
                """
                INSERT INTO ml_stock_eligibility
                    (stock_ticker, eligible, reason, n_move_events,
                     n_trading_days, liquidity_tier, median_daily_vol,
                     watch_only, last_evaluated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT (stock_ticker) DO UPDATE SET
                    eligible       = excluded.eligible,
                    reason         = excluded.reason,
                    n_move_events  = excluded.n_move_events,
                    n_trading_days = excluded.n_trading_days,
                    liquidity_tier = excluded.liquidity_tier,
                    median_daily_vol = excluded.median_daily_vol,
                    watch_only     = excluded.watch_only,
                    last_evaluated = excluded.last_evaluated
                """,
                (
                    elig.ticker, int(elig.eligible), elig.reason,
                    elig.n_move_events, elig.n_trading_days,
                    elig.liquidity_tier, elig.median_daily_vol,
                    int(elig.watch_only),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist eligibility for %s: %s", elig.ticker, exc)

    def _log_lineage(
        self,
        source: str,
        action: str,
        ticker: Optional[str],
        records: int,
        notes: str = "",
    ) -> None:
        try:
            from app.services.eagle_eye.ml.db_tables import log_data_lineage
            log_data_lineage(
                source=source, action=action,
                stock_ticker=ticker, records_affected=records,
                notes=notes or None,
            )
        except Exception:  # noqa: BLE001
            pass  # Lineage logging must never break the pipeline
