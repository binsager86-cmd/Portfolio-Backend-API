"""
ml/pattern_store.py — Phase 2 Deliverable 6

Per-stock NearestNeighbors cosine-similarity pattern store.

For each eligible stock:
  1. Load its training matrix.
  2. For each date in the matrix, compute a normalized feature vector.
  3. Build a per-stock sklearn NearestNeighbors(metric='cosine') index.
  4. Save the NN index to ml_models/pattern_indices/{TICKER}.pkl
  5. Write compact vector rows to the pattern_vector_store table (BLOB).

Query interface:
  query_similar(ticker, query_vector, query_date, top_k=5) →
    List[dict(date, similarity, forward_return_pct, label_hit)]

Test (3 stocks × recent date → top-5 analogues):
  run_sanity_test(tickers, n=3)
"""
from __future__ import annotations

import io
import logging
import pickle
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from app.core.config import get_settings
from app.services.eagle_eye.ml.training_matrix import (
    PRIMARY_LABEL,
    load_stock_matrix,
)
from app.services.eagle_eye.store import list_tickers_with_ohlcv

LOGGER = logging.getLogger(__name__)

INDEX_DIR_NAME = "pattern_indices"
VECTOR_VERSION = "v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patterns_root() -> Path:
    p = Path(__file__).resolve().parents[5] / "ml_models" / INDEX_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    from app.services.eagle_eye.ml.feature_builder import NON_FEATURE_COLUMNS
    from app.services.eagle_eye.ml.training_matrix import SURFACE_LABEL_COLS
    non_feat = set(NON_FEATURE_COLUMNS) | set(SURFACE_LABEL_COLS) | {
        "regime", "flag_low_volume", "flag_corp_action",
    }
    return [c for c in df.columns if c not in non_feat and not c.startswith("y_")]


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2-normalize a feature vector. Returns zero vector if norm is 0."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


def _build_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    # Row-wise L2 normalization for cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (X / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Build and save per-stock index
# ---------------------------------------------------------------------------

def build_pattern_index(
    ticker: str,
    df: Optional[pd.DataFrame] = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[Tuple[NearestNeighbors, pd.DataFrame, List[str]]]:
    """
    Build a NearestNeighbors cosine index for one stock.

    Returns (nn_model, metadata_df, feature_cols) or None on failure.
    metadata_df has columns: event_date, primary_label, forward_return_pct
    """
    log = logger or LOGGER

    if df is None:
        df = load_stock_matrix(ticker)
    if df is None or df.empty:
        log.warning("[%s] No training matrix for pattern store", ticker)
        return None

    feature_cols = _get_feature_cols(df)
    if len(feature_cols) < 3:
        log.warning("[%s] Too few features (%d) for pattern store", ticker, len(feature_cols))
        return None

    df = df.sort_values("event_date").reset_index(drop=True)
    X = _build_matrix(df, feature_cols)

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(20, len(df)))
    nn.fit(X)

    meta_df = pd.DataFrame({
        "event_date": pd.to_datetime(df["event_date"], errors="coerce"),
        "primary_label": df.get(PRIMARY_LABEL, pd.Series(np.nan, index=df.index)),
        "max_excursion_pct": df.get("max_excursion_pct", pd.Series(np.nan, index=df.index)),
    })

    log.info("[%s] Pattern index built: %d vectors, %d features", ticker, len(df), len(feature_cols))
    return nn, meta_df, feature_cols


def save_pattern_index(
    ticker: str,
    nn: NearestNeighbors,
    meta_df: pd.DataFrame,
    feature_cols: List[str],
) -> Path:
    """Persist the NN index bundle to disk."""
    bundle = {
        "ticker": ticker.upper(),
        "version": VECTOR_VERSION,
        "nn": nn,
        "meta_df": meta_df,
        "feature_cols": feature_cols,
        "built_at": datetime.utcnow().isoformat(),
    }
    path = _patterns_root() / f"{ticker.upper()}.pkl"
    with path.open("wb") as f:
        pickle.dump(bundle, f, protocol=4)
    return path


def load_pattern_index(ticker: str) -> Optional[Dict[str, Any]]:
    """Load a previously saved NN index bundle from disk."""
    path = _patterns_root() / f"{ticker.upper()}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# DB vector store
# ---------------------------------------------------------------------------

def _write_vectors_to_db(
    ticker: str,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> int:
    """Write compact normalized vector rows to pattern_vector_store."""
    from app.core.database import exec_sql
    inserted = 0
    X = _build_matrix(df, feature_cols)
    now_str = datetime.utcnow().isoformat()

    for i, (_, row) in enumerate(df.iterrows()):
        vec_bytes = X[i].tobytes()
        event_date = str(pd.Timestamp(row.get("event_date", "")).date()) if row.get("event_date") is not None else ""
        primary_lbl = row.get(PRIMARY_LABEL)
        label_val = int(primary_lbl) if primary_lbl is not None and not (isinstance(primary_lbl, float) and np.isnan(primary_lbl)) else None
        meta = {"vector_version": VECTOR_VERSION, "primary_label": label_val}
        import json as _json
        try:
            exec_sql(
                """INSERT INTO pattern_vector_store
                   (stock_ticker, vector_date, vector_blob, vector_dim, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT (stock_ticker, vector_date) DO UPDATE SET
                       vector_blob = EXCLUDED.vector_blob,
                       vector_dim = EXCLUDED.vector_dim,
                       metadata_json = EXCLUDED.metadata_json""",
                (ticker.upper(), event_date, vec_bytes, len(feature_cols), _json.dumps(meta), now_str),
            )
            inserted += 1
        except Exception:
            pass

    return inserted


# ---------------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------------

def query_similar(
    ticker: str,
    query_vector: np.ndarray,
    query_date: Optional[date] = None,
    top_k: int = 5,
    *,
    bundle: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Find the top_k most similar historical setups for a stock.

    Parameters
    ----------
    ticker : stock ticker
    query_vector : raw feature vector (will be L2-normalized internally)
    query_date : if provided, only return analogues with date < query_date (no lookahead)
    top_k : number of analogues to return

    Returns list of dicts:
      {date, similarity, primary_label, max_excursion_pct}
    """
    if bundle is None:
        bundle = load_pattern_index(ticker)
    if bundle is None:
        return []

    nn: NearestNeighbors = bundle["nn"]
    meta_df: pd.DataFrame = bundle["meta_df"]
    feature_cols: List[str] = bundle["feature_cols"]

    # Align query vector to stored feature set
    q = np.array(query_vector, dtype=np.float32)
    if len(q) != len(feature_cols):
        # Truncate or pad if dimensions mismatch
        if len(q) > len(feature_cols):
            q = q[:len(feature_cols)]
        else:
            q = np.pad(q, (0, len(feature_cols) - len(q)))

    q_norm = _normalize_vector(q).reshape(1, -1)

    # Get more candidates if we're going to filter by date
    n_candidates = min(top_k * 4 if query_date else top_k, len(meta_df))
    distances, indices = nn.kneighbors(q_norm, n_neighbors=n_candidates)

    results: List[Dict[str, Any]] = []
    for dist, idx in zip(distances[0], indices[0]):
        row = meta_df.iloc[idx]
        analogue_date = pd.Timestamp(row["event_date"]).date() if pd.notna(row["event_date"]) else None

        if query_date and analogue_date and analogue_date >= query_date:
            continue  # Strict temporal guard: no future analogues

        similarity = max(0.0, 1.0 - float(dist))  # cosine similarity from cosine distance
        results.append({
            "date": str(analogue_date) if analogue_date else None,
            "similarity": round(similarity, 4),
            "primary_label": int(row["primary_label"]) if pd.notna(row.get("primary_label")) else None,
            "max_excursion_pct": round(float(row["max_excursion_pct"]), 2) if pd.notna(row.get("max_excursion_pct")) else None,
        })

        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------------------------
# Sanity test
# ---------------------------------------------------------------------------

def run_sanity_test(
    tickers: Optional[Sequence[str]] = None,
    n: int = 3,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Sanity test: for n stocks × a recent query date → top-5 analogues.

    Prints results to logger.info. No assertions — visual inspection.
    """
    log = logger or LOGGER

    if tickers is None:
        tickers = list_tickers_with_ohlcv()[:n]

    for ticker in list(tickers)[:n]:
        bundle = load_pattern_index(ticker)
        if bundle is None:
            log.warning("[%s] No pattern index found — skipping sanity test", ticker)
            continue

        meta_df = bundle["meta_df"]
        feature_cols = bundle["feature_cols"]

        # Use the most recent event as the query
        if meta_df.empty:
            continue

        last_idx = meta_df["event_date"].idxmax()
        last_date = pd.Timestamp(meta_df.loc[last_idx, "event_date"]).date()

        # Dummy query vector: the second-to-last stored row
        nn: NearestNeighbors = bundle["nn"]
        if len(meta_df) < 3:
            continue

        df = load_stock_matrix(ticker)
        if df is None or df.empty:
            continue

        df = df.sort_values("event_date").reset_index(drop=True)
        X = _build_matrix(df, feature_cols)
        query_vec = X[-2]  # penultimate event

        analogues = query_similar(
            ticker, query_vec,
            query_date=last_date,
            top_k=5,
            bundle=bundle,
        )

        log.info("[%s] Top-5 analogues for %s:", ticker, last_date)
        for a in analogues:
            log.info(
                "  %s  sim=%.3f  label=%s  excursion=%.1f%%",
                a["date"], a["similarity"],
                a["primary_label"], a["max_excursion_pct"] or 0.0,
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_all_pattern_indices(
    tickers: Optional[Sequence[str]] = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """
    Build and persist pattern indices for all eligible stocks.

    Returns dict of ticker → index path (or error message).
    """
    log = logger or LOGGER

    if tickers is None:
        tickers = _get_eligible_tickers(log)

    results: Dict[str, str] = {}

    for ticker in tickers:
        try:
            df = load_stock_matrix(ticker)
            build_result = build_pattern_index(ticker, df, logger=log)
            if build_result is None:
                results[ticker] = "skipped"
                continue

            nn, meta_df, feature_cols = build_result

            # Save to disk
            path = save_pattern_index(ticker, nn, meta_df, feature_cols)

            # Write compact vectors to DB
            if df is not None:
                n_written = _write_vectors_to_db(ticker, df, feature_cols)
                log.info("[%s] %d vectors written to DB", ticker, n_written)

            results[ticker] = str(path)
            log.info("[%s] Pattern index saved: %s", ticker, path)
        except Exception as exc:
            results[ticker] = f"error: {exc}"
            log.error("[%s] Pattern index error: %s", ticker, exc)

    # Run sanity test on a few stocks
    ok_tickers = [t for t, v in results.items() if not v.startswith(("skipped", "error"))]
    if ok_tickers:
        run_sanity_test(ok_tickers, n=3, logger=log)

    log.info("Pattern index build complete: %d/%d succeeded", len(ok_tickers), len(tickers))
    return results


def _get_eligible_tickers(log: logging.Logger) -> List[str]:
    try:
        from app.core.database import query_all
        rows = query_all(
            "SELECT stock_ticker FROM ml_stock_eligibility WHERE eligible=1 AND (watch_only IS NULL OR watch_only=0)"
        )
        tickers = [r[0] for r in rows if r[0]]
        if tickers:
            return tickers
    except Exception as exc:
        log.warning("Eligibility table error: %s", exc)
    return list_tickers_with_ohlcv()
