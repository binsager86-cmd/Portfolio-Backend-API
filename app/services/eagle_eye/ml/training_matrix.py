"""
ml/training_matrix.py — Phase 2 Deliverable 1

Per-stock training matrix builder.

For each eligible stock:
  1. Detect historical move events (v1 canonical pipeline — feature_builder_v2 is orphaned).
  2. Build feature rows at each event's prediction position using compute_all_indicators.
  3. Compute the full 5×4 binary label surface (5 return targets × 4 horizons = 20 labels).
  4. Add regime tag (BULL / BEAR / CHOPPY) and quality flags.
  5. Run leakage audit.  If > 5% of features fail → stop and report.
  6. Write per-stock parquet + schema.json.
  7. Write manifest.

Output layout
-------------
  ml_training_matrix/v1/{TICKER}/data.parquet
  ml_training_matrix/v1/{TICKER}/schema.json
  ml_training_matrix/v1/_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.services.eagle_eye.indicators import compute_all_indicators
from app.services.eagle_eye.ml.feature_builder import (
    build_event_feature_rows_for_ticker,
    build_feature_matrix,
    get_feature_columns,
    NON_FEATURE_COLUMNS,
)
from app.services.eagle_eye.ml.leakage_audit import LeakageAuditor
from app.services.eagle_eye.store import list_tickers_with_ohlcv, load_ohlcv

LOGGER = logging.getLogger(__name__)

# ── Label surface definition ──────────────────────────────────────────────
RETURN_TARGETS_PCT: Tuple[int, ...] = (5, 7, 10, 15, 20)
HORIZONS_TD: Tuple[int, ...] = (7, 14, 20, 40)
STOP_PCT: float = 5.0
PRIMARY_LABEL: str = "y_10pct_20d"  # global default; per-stock selection via select_primary_label()

# Adaptive label tiers — tightest label with enough positives wins.
# Order is fixed: do not reorder or add tiers without a separate decision.
LABEL_TIERS: List[Tuple[str, int]] = [
    ("y_10pct_20d", 50),   # aggressive movers
    ("y_7pct_20d",  50),   # moderate movers
    ("y_5pct_20d",  50),   # stable / low-beta stocks
]


def select_primary_label(df: pd.DataFrame) -> Optional[str]:
    """
    Return the tightest label column that has >= 50 positive examples.

    Returns None if no tier qualifies — stock should be marked INSUFFICIENT_DATA.
    The 50-positive threshold is a data-quality gate, not a feature; it only
    counts existing label column values and contains no forward-looking information.
    """
    for col, min_pos in LABEL_TIERS:
        if col in df.columns:
            n_pos = int((df[col] == 1).sum())
            if n_pos >= min_pos:
                return col
    return None


SURFACE_LABEL_COLS: List[str] = [
    f"y_{r}pct_{h}d"
    for r in RETURN_TARGETS_PCT
    for h in HORIZONS_TD
]

# ── Regime computation constants ──────────────────────────────────────────
SMA_PERIOD = 200
REGIME_SLOPE_WINDOW = 20

LEAKY_THRESHOLD_CORR = 0.95      # feature-target correlation → LEAKY
LEAKY_THRESHOLD_FRAC = 0.05      # > 5% features leaky → abort


# ---------------------------------------------------------------------------
# Surface label computation
# ---------------------------------------------------------------------------

def build_label_surface(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    stop_pct: float = 0.05,
    return_targets: Sequence[int] = RETURN_TARGETS_PCT,
    horizons: Sequence[int] = HORIZONS_TD,
) -> pd.DataFrame:
    """
    Standalone label surface builder.  Computes binary labels for each row
    using TRADING-DAY forward windows (positional offset in df, not calendar days).

    Label naming convention: ``label_{horizon}d_{return}pct``
    e.g. label_5d_10pct = 1 if price rises 10% within 5 trading days without
    first hitting the stop loss.

    Parameters
    ----------
    df : DataFrame with at minimum `close_col`, `high_col`, `low_col` columns
         and an optional date column. Rows must be in chronological order.
    date_col : name of the date column (used for output alignment only)
    close_col : column used for entry price (close at signal bar)
    high_col : column used to detect target hit
    low_col : column used to detect stop hit
    stop_pct : fractional stop loss threshold (default 0.05 = -5%)
    return_targets : return % targets (integers, e.g. [5, 10, 15])
    horizons : forward trading-day horizons (e.g. [5, 10, 20])

    Returns
    -------
    DataFrame of binary labels aligned to df's index.
    """
    from typing import Sequence as _Seq  # local to avoid shadowing outer import

    df = df.reset_index(drop=True)
    n = len(df)
    max_h = max(horizons)

    label_cols = {f"label_{h}d_{r}pct": np.zeros(n, dtype=int) for r in return_targets for h in horizons}

    highs = df[high_col].values.astype(float)
    lows = df[low_col].values.astype(float)
    closes = df[close_col].values.astype(float)

    for i in range(n):
        entry = closes[i]
        if math.isnan(entry) or entry <= 0:
            continue
        stop_price = entry * (1 - stop_pct)

        # Precompute stop day (horizon-independent, up to max horizon)
        stop_day: Optional[int] = None
        for offset in range(1, min(max_h + 1, n - i)):
            low_val = lows[i + offset]
            if not math.isnan(low_val) and low_val <= stop_price:
                stop_day = offset
                break

        for r in return_targets:
            target_price = entry * (1 + r / 100)
            for h in horizons:
                col = f"label_{h}d_{r}pct"
                # If stop fires before this horizon ends, label = 0
                if stop_day is not None and stop_day <= h:
                    label_cols[col][i] = 0
                    continue
                # Scan forward up to h trading days
                hit = False
                for offset in range(1, min(h + 1, n - i)):
                    high_val = highs[i + offset]
                    if not math.isnan(high_val) and high_val >= target_price:
                        hit = True
                        break
                label_cols[col][i] = int(hit)

    out = pd.DataFrame(label_cols, index=df.index)
    if date_col in df.columns:
        out.insert(0, date_col, df[date_col])
    return out


def _compute_surface_labels(
    ohlcv: pd.DataFrame,
    accel_pos: int,
    entry_price: float,
) -> Dict[str, int]:
    """
    Compute all 20 binary surface labels for one event.

    A label (r, h) = 1 if price rises r% within h trading days from the
    acceleration bar WITHOUT first hitting the -5% stop.

    Ambiguous same-bar touch (both stop and target) → stop wins (conservative).
    """
    labels: Dict[str, int] = {}

    if math.isnan(entry_price) or entry_price <= 0:
        for col in SURFACE_LABEL_COLS:
            labels[col] = 0
        return labels

    max_horizon = max(HORIZONS_TD)
    future_all = ohlcv.iloc[accel_pos + 1: accel_pos + max_horizon + 1]

    stop_price = entry_price * (1 - STOP_PCT / 100)

    # Precompute stop firing day (horizon-independent)
    stop_day: Optional[int] = None
    for day_num, (_, row) in enumerate(future_all.iterrows(), start=1):
        low_val = float(row.get("low", float("nan")))
        high_val = float(row.get("high", float("nan")))
        if not math.isnan(low_val) and low_val <= stop_price:
            # Check if target also hit on same bar — stop wins
            stop_day = day_num
            break

    for r in RETURN_TARGETS_PCT:
        target_price = entry_price * (1 + r / 100)
        for h in HORIZONS_TD:
            col = f"y_{r}pct_{h}d"
            # If stop already fired before this horizon ends, use stop_day
            effective_stop = stop_day if (stop_day is not None and stop_day <= h) else None
            if effective_stop is not None:
                labels[col] = 0
                continue
            # Scan up to horizon h (or up to stop if stop_day > h, meaning stop
            # fires AFTER the horizon window ends — stop doesn't affect this cell)
            hit = False
            future_h = future_all.iloc[:h]
            for _, row in future_h.iterrows():
                high_val = float(row.get("high", float("nan")))
                if not math.isnan(high_val) and high_val >= target_price:
                    hit = True
                    break
            labels[col] = int(hit)

    return labels


# ---------------------------------------------------------------------------
# Regime tagger
# ---------------------------------------------------------------------------

def _regime_at(close_series: pd.Series, pos: int) -> str:
    """
    BULL / BEAR / CHOPPY regime tag at position pos.
    Uses stock's own 200-day SMA and its 20-day slope — no future data.
    """
    if pos < SMA_PERIOD:
        return "CHOPPY"

    window = close_series.iloc[max(0, pos - SMA_PERIOD):pos]
    sma_now = float(window.mean())
    close_now = float(close_series.iloc[pos])

    slope_window = close_series.iloc[max(0, pos - SMA_PERIOD):max(1, pos - SMA_PERIOD + REGIME_SLOPE_WINDOW)]
    sma_prev = float(slope_window.mean()) if len(slope_window) > 0 else sma_now

    sma_rising = sma_now > sma_prev

    if close_now > sma_now and sma_rising:
        return "BULL"
    if close_now < sma_now and not sma_rising:
        return "BEAR"
    return "CHOPPY"


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------

def _quality_flags(ohlcv: pd.DataFrame, pos: int, ticker: str) -> Dict[str, int]:
    """
    Compute quality flags for a row at position pos.

    flag_low_volume: volume < 10th percentile of preceding 60d
    flag_corp_action: corporate event within 5 trading days of event date
    """
    flags: Dict[str, int] = {
        "flag_low_volume": 0,
        "flag_corp_action": 0,
    }

    # Low volume flag
    if pos >= 2:
        vol_hist = ohlcv["volume"].iloc[max(0, pos - 60):pos]
        if len(vol_hist) >= 10:
            p10 = float(np.percentile(vol_hist, 10))
            today_vol = float(ohlcv["volume"].iloc[pos])
            if today_vol < p10:
                flags["flag_low_volume"] = 1

    # Corporate action flag
    try:
        from app.core.database import query_one
        event_date = ohlcv.index[pos].date()
        check_start = str(date(event_date.year, event_date.month, event_date.day).__class__(
            event_date.year, event_date.month, event_date.day
        ) - pd.Timedelta(days=7))
        check_end = str(event_date + pd.Timedelta(days=5))
        row = query_one(
            """SELECT COUNT(*) FROM ml_corporate_events
               WHERE stock_ticker = ? AND announcement_date BETWEEN ? AND ?""",
            (ticker.upper(), check_start, check_end),
        )
        if row and row[0] > 0:
            flags["flag_corp_action"] = 1
    except Exception:
        pass  # No corp event table or DB access — leave as 0

    return flags


# ---------------------------------------------------------------------------
# Per-stock matrix builder
# ---------------------------------------------------------------------------

def build_stock_matrix(
    ticker: str,
    ohlcv: Optional[pd.DataFrame] = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    """
    Build the full training matrix for one stock.

    Returns a DataFrame indexed by event_date with:
      - Feature columns (from v1 canonical pipeline)
      - 20 surface label columns (y_Rpct_Hd)
      - regime column
      - quality flag columns
      - metadata columns (ticker, event_id, event_date)

    Returns None if the stock cannot produce a usable matrix.
    """
    log = logger or LOGGER

    if ohlcv is None:
        ohlcv = load_ohlcv(ticker)
    if ohlcv is None or len(ohlcv) < 120:
        log.warning("[%s] Skipping — insufficient OHLCV bars (%s)", ticker, len(ohlcv) if ohlcv is not None else 0)
        return None

    # Build event feature rows using the canonical v1 pipeline
    try:
        from app.services.eagle_eye.adapter import TickerChartAdapter
        meta_map = {}
        try:
            adapter = TickerChartAdapter()
            for m in adapter.list_stocks():
                meta_map[m.ticker.upper()] = m
        except Exception:
            pass

        from app.services.eagle_eye.ml.feature_builder import _build_regime_frame
        end = date.today()
        start = date(end.year - 6, end.month, min(end.day, 28))
        regime_frame = _build_regime_frame(start, end, logger=log)

        rows = build_event_feature_rows_for_ticker(
            ticker=ticker,
            ohlcv=ohlcv,
            stock_meta=meta_map.get(ticker.upper()),
            regime_frame=regime_frame,
            include_fakeouts=True,
        )
    except Exception as exc:
        log.warning("[%s] Feature row build failed: %s", ticker, exc)
        return None

    if not rows:
        log.warning("[%s] No event rows produced", ticker)
        return None

    # Build feature matrix (cleaning, imputation)
    result = build_feature_matrix(rows, logger=log)
    if result.frame.empty:
        log.warning("[%s] Feature matrix empty after filtering", ticker)
        return None

    df = result.frame.copy()

    # Compute indicators once for the surface label + regime computation
    try:
        ind_df = compute_all_indicators(ohlcv)
    except Exception as exc:
        log.warning("[%s] compute_all_indicators failed: %s", ticker, exc)
        return None

    close_series = ohlcv["close"]

    # Enrich each row with surface labels, regime, and quality flags
    surface_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        event_date = row.get("event_date")
        if event_date is None:
            surface_rows.append({col: 0 for col in SURFACE_LABEL_COLS})
            continue

        ts = pd.Timestamp(event_date)
        # Locate acceleration position in OHLCV
        # feature_builder uses pred_pos (day before acceleration); acceleration_date is in the row
        accel_date_str = row.get("acceleration_date") or str(ts.date())
        try:
            accel_ts = pd.Timestamp(accel_date_str)
            accel_pos = ohlcv.index.get_indexer([accel_ts], method="nearest")[0]
        except Exception:
            accel_pos = ohlcv.index.get_indexer([ts], method="nearest")[0]

        # acceleration_price is entry for label computation
        entry_price = float(row.get("acceleration_price") or row.get("close") or float("nan"))
        if math.isnan(entry_price) and accel_pos >= 0:
            entry_price = float(ohlcv["close"].iloc[accel_pos])

        surface_labels = _compute_surface_labels(ohlcv, accel_pos, entry_price)

        # Regime at prediction position (day before acceleration)
        pred_pos = max(0, accel_pos - 1)
        regime = _regime_at(close_series, pred_pos)

        # Quality flags
        qf = _quality_flags(ohlcv, pred_pos, ticker)

        combined = {**surface_labels, "regime": regime, **qf}
        surface_rows.append(combined)

    enrichment = pd.DataFrame(surface_rows, index=df.index)
    df = pd.concat([df, enrichment], axis=1)

    # Ensure event_date is a proper datetime column for time-based splits
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)

    # Drop rows with no primary label (should be rare)
    if PRIMARY_LABEL in df.columns:
        df = df.dropna(subset=[PRIMARY_LABEL])

    if len(df) < 10:
        log.warning("[%s] Fewer than 10 rows after enrichment — skipping", ticker)
        return None

    log.info("[%s] Matrix built: %d rows, %d features", ticker, len(df), len(get_feature_columns(df)))
    return df


# ---------------------------------------------------------------------------
# Leakage audit per stock
# ---------------------------------------------------------------------------

def audit_stock_matrix(
    ticker: str,
    df: pd.DataFrame,
    *,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Run leakage audit on a stock's training matrix.

    Returns:
      (cleaned_df, dropped_features)

    Raises RuntimeError if > 5% of features are leaky (structural problem).
    """
    log = logger or LOGGER
    auditor = LeakageAuditor()
    feat_cols = get_feature_columns(df)

    if not feat_cols or PRIMARY_LABEL not in df.columns:
        return df, []

    # audit_dataframe expects a numeric feature set + date col + target col
    try:
        report = auditor.audit_dataframe(
            df[feat_cols + ["event_date", PRIMARY_LABEL]],
            date_col="event_date",
            target_col=PRIMARY_LABEL,
        )
    except Exception as exc:
        log.warning("[%s] Leakage audit failed: %s", ticker, exc)
        return df, []

    leaky: List[str] = []
    if hasattr(report, "results"):
        for res in report.results:
            if getattr(res, "verdict", "CLEAN") in ("LEAKY",):
                leaky.append(getattr(res, "feature_name", ""))
    elif hasattr(report, "issues"):
        leaky = [i.get("feature", "") for i in report.issues if i.get("verdict") == "LEAKY"]

    leaky = [f for f in leaky if f]

    leaky_frac = len(leaky) / max(len(feat_cols), 1)
    if leaky_frac > LEAKY_THRESHOLD_FRAC:
        raise RuntimeError(
            f"[{ticker}] {len(leaky)}/{len(feat_cols)} features ({leaky_frac:.1%}) "
            f"failed leakage audit — structural problem. Stopping."
        )

    if leaky:
        log.info("[%s] Dropping %d leaky features: %s", ticker, len(leaky), leaky[:5])
        df = df.drop(columns=leaky, errors="ignore")

    # Log dropped features to features_audit table
    _log_features_audit(leaky, verdict="DROPPED", notes=f"leakage audit: {ticker}")

    return df, leaky


def _log_features_audit(
    features: List[str],
    verdict: str,
    notes: str,
) -> None:
    try:
        from app.core.database import exec_sql
        for feat in features:
            exec_sql(
                """INSERT INTO features_audit
                   (feature_name, feature_version, leakage_verdict, audit_notes, updated_at)
                   VALUES (?, 'v1', ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT (feature_name, feature_version) DO UPDATE SET
                       leakage_verdict = EXCLUDED.leakage_verdict,
                       audit_notes = EXCLUDED.audit_notes,
                       updated_at = EXCLUDED.updated_at""",
                (feat, verdict, notes),
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _matrix_root() -> Path:
    p = Path(__file__).resolve().parents[5] / "ml_training_matrix" / "v1"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _feature_hash(feat_cols: List[str]) -> str:
    joined = ",".join(sorted(feat_cols))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def write_stock_matrix(ticker: str, df: pd.DataFrame) -> Path:
    """Write per-stock parquet + schema.json. Returns directory path."""
    root = _matrix_root()
    out_dir = root / ticker.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "data.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    feat_cols = get_feature_columns(df)
    label_cols = [c for c in df.columns if c.startswith("y_")]
    # Per-stock adaptive label selection — written once at matrix build time.
    # At train time, read this value from schema.json — do not recompute.
    selected_primary = select_primary_label(df)
    schema = {
        "ticker": ticker.upper(),
        "n_rows": len(df),
        "n_features": len(feat_cols),
        "feature_cols": feat_cols,
        "label_cols": label_cols,
        "primary_label": selected_primary,   # None → INSUFFICIENT_DATA at train time
        "primary_label_status": "ok" if selected_primary else "INSUFFICIENT_DATA",
        "surface_labels": SURFACE_LABEL_COLS,
        "return_targets_pct": list(RETURN_TARGETS_PCT),
        "horizons_td": list(HORIZONS_TD),
        "stop_pct": STOP_PCT,
        "feature_audit_hash": _feature_hash(feat_cols),
        "date_range_start": str(df["event_date"].min().date()) if "event_date" in df else "",
        "date_range_end": str(df["event_date"].max().date()) if "event_date" in df else "",
        "built_at": datetime.utcnow().isoformat(),
    }
    with (out_dir / "schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return out_dir


def write_manifest(results: List[Dict[str, Any]]) -> Path:
    root = _matrix_root()
    manifest_path = root / "_manifest.json"
    manifest = {
        "built_at": datetime.utcnow().isoformat(),
        "version": "v1",
        "stocks": results,
        "total": len(results),
        "successful": sum(1 for r in results if r.get("status") == "ok"),
        "failed": sum(1 for r in results if r.get("status") != "ok"),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_all_matrices(
    tickers: Optional[Sequence[str]] = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Build training matrices for all eligible stocks.

    Returns a list of per-stock result dicts (for the manifest).
    """
    log = logger or LOGGER

    if tickers is None:
        tickers = _get_eligible_tickers(log)

    log.info("Building training matrices for %d stocks", len(tickers))
    results: List[Dict[str, Any]] = []

    for ticker in tickers:
        entry: Dict[str, Any] = {"ticker": ticker, "status": "error", "n_rows": 0, "n_features": 0}
        try:
            ohlcv = load_ohlcv(ticker)
            df = build_stock_matrix(ticker, ohlcv, logger=log)
            if df is None:
                entry["status"] = "skipped"
                entry["reason"] = "no_matrix"
                results.append(entry)
                continue

            df, dropped_feats = audit_stock_matrix(ticker, df, logger=log)

            out_dir = write_stock_matrix(ticker, df)
            feat_cols = get_feature_columns(df)
            entry.update({
                "status": "ok",
                "n_rows": len(df),
                "n_features": len(feat_cols),
                "n_dropped_features": len(dropped_feats),
                "date_start": str(df["event_date"].min().date()) if "event_date" in df else "",
                "date_end": str(df["event_date"].max().date()) if "event_date" in df else "",
                "path": str(out_dir),
            })
            log.info(
                "[%s] ✓ %d rows, %d features, %d dropped",
                ticker, len(df), len(feat_cols), len(dropped_feats),
            )

        except RuntimeError as exc:
            # Structural leakage — abort this stock
            entry["status"] = "aborted_leakage"
            entry["reason"] = str(exc)
            log.error("[%s] %s", ticker, exc)

        except Exception as exc:
            entry["status"] = "error"
            entry["reason"] = str(exc)[:200]
            log.error("[%s] Unexpected error: %s", ticker, exc)

        results.append(entry)

    manifest_path = write_manifest(results)
    log.info("Manifest written to %s", manifest_path)
    ok_count = sum(1 for r in results if r["status"] == "ok")
    log.info("Done: %d/%d stocks succeeded", ok_count, len(tickers))
    return results


def load_stock_matrix(ticker: str) -> Optional[pd.DataFrame]:
    """Load a previously built stock matrix from parquet."""
    p = _matrix_root() / ticker.upper() / "data.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p, engine="pyarrow")


def _get_eligible_tickers(log: logging.Logger) -> List[str]:
    """Load eligible tickers from ml_stock_eligibility table, or fall back to OHLCV list."""
    try:
        from app.core.database import query_all
        rows = query_all(
            "SELECT stock_ticker FROM ml_stock_eligibility WHERE eligible=1 AND (watch_only IS NULL OR watch_only=0)"
        )
        tickers = [r[0] for r in rows if r[0]]
        if tickers:
            log.info("Loaded %d eligible tickers from ml_stock_eligibility", len(tickers))
            return tickers
    except Exception as exc:
        log.warning("Could not load eligibility table: %s — using OHLCV list", exc)
    tickers = list_tickers_with_ohlcv()
    log.info("Falling back to %d tickers from OHLCV store", len(tickers))
    return tickers
