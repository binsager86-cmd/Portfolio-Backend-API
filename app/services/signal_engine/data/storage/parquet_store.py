"""Parquet-backed columnar storage for backtesting OHLCV data.

Provides an efficient on-disk cache for historical OHLCV+indicators data
used in walk-forward backtests.  Files are stored under a configurable
base directory and partitioned by stock code.

Why Parquet?
------------
* Columnar format → fast aggregations on price/volume columns.
* Built-in compression (snappy) → typical 70 % size reduction vs CSV.
* Preserves dtypes across read/write cycles.
* pyarrow is already a transitive dependency of pandas in the project.

Directory layout:
    {BASE_DIR}/
        NBK.parquet
        KFH.parquet
        ZAIN.parquet
        ...

Row schema (all columns):
    date        : str   (YYYY-MM-DD)
    open        : float
    high        : float
    low         : float
    close       : float
    volume      : int
    value       : float   (KWD value)
    ema_20      : float | null
    ema_50      : float | null
    sma_200     : float | null
    rsi_14      : float | null
    adx_14      : float | null
    macd        : float | null
    macd_signal : float | null
    macd_hist   : float | null
    atr_14      : float | null
    bb_upper    : float | null
    bb_lower    : float | null
    bb_mid      : float | null
    obv         : float | null
    ad          : float | null
    cmf_20      : float | null
    vwap        : float | null
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default storage location — overridable via environment variable
_DEFAULT_BASE_DIR = Path(os.environ.get(
    "SIGNAL_PARQUET_DIR",
    Path(__file__).parent.parent.parent.parent.parent / "data" / "signal_parquet",
))

# ── Import guard for optional pyarrow ─────────────────────────────────────────
try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    _PARQUET_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PARQUET_AVAILABLE = False
    logger.warning("pyarrow/pandas not installed — ParquetStore will operate in passthrough mode.")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _stock_path(stock_code: str, base_dir: Path) -> Path:
    return base_dir / f"{stock_code.upper()}.parquet"


def _ensure_dir(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)


# ── Public API ────────────────────────────────────────────────────────────────

def save_rows(
    stock_code: str,
    rows: list[dict[str, Any]],
    base_dir: Path | None = None,
) -> bool:
    """Persist OHLCV+indicator rows for a stock to Parquet.

    If the file already exists the new rows are merged (union by date, with
    newer data winning on conflicts).

    Args:
        stock_code: KSE stock code (e.g. "NBK").
        rows:       List of row dicts with at minimum date/open/high/low/close/volume.
        base_dir:   Override storage directory (default: SIGNAL_PARQUET_DIR env var).

    Returns:
        True on success, False on failure.
    """
    if not _PARQUET_AVAILABLE:
        return False
    if not rows:
        return True

    base = base_dir or _DEFAULT_BASE_DIR
    _ensure_dir(base)
    path = _stock_path(stock_code, base)

    new_df = pd.DataFrame(rows)
    new_df["date"] = new_df["date"].astype(str)

    if path.exists():
        try:
            existing_df = pq.read_table(str(path)).to_pandas()
            combined = (
                pd.concat([existing_df, new_df], ignore_index=True)
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not merge existing parquet for %s (%s) — overwriting.", stock_code, exc)
            combined = new_df.sort_values("date").reset_index(drop=True)
    else:
        combined = new_df.sort_values("date").reset_index(drop=True)

    try:
        table = pa.Table.from_pandas(combined, preserve_index=False)
        pq.write_table(table, str(path), compression="snappy")
        logger.debug("Saved %d rows for %s to %s", len(combined), stock_code, path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write parquet for %s: %s", stock_code, exc)
        return False


def load_rows(
    stock_code: str,
    from_date: str | None = None,
    to_date: str | None = None,
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load stored OHLCV+indicator rows for a stock.

    Args:
        stock_code: KSE stock code.
        from_date:  Optional start date filter (YYYY-MM-DD, inclusive).
        to_date:    Optional end date filter (YYYY-MM-DD, inclusive).
        base_dir:   Override storage directory.

    Returns:
        List of row dicts sorted ascending by date.  Empty list if not found.
    """
    if not _PARQUET_AVAILABLE:
        return []

    base = base_dir or _DEFAULT_BASE_DIR
    path = _stock_path(stock_code, base)

    if not path.exists():
        return []

    try:
        table = pq.read_table(str(path))
        df = table.to_pandas()
        df["date"] = df["date"].astype(str)

        if from_date:
            df = df[df["date"] >= from_date]
        if to_date:
            df = df[df["date"] <= to_date]

        df = df.sort_values("date").reset_index(drop=True)
        # Replace NaN with None for downstream compatibility
        return df.where(pd.notna(df), other=None).to_dict(orient="records")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read parquet for %s: %s", stock_code, exc)
        return []


def delete_stock(stock_code: str, base_dir: Path | None = None) -> bool:
    """Delete the stored parquet file for a stock.

    Returns True if deleted, False if file did not exist or deletion failed.
    """
    base = base_dir or _DEFAULT_BASE_DIR
    path = _stock_path(stock_code, base)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except OSError as exc:
        logger.error("Failed to delete parquet for %s: %s", stock_code, exc)
        return False


def list_stored_stocks(base_dir: Path | None = None) -> list[str]:
    """Return list of stock codes that have stored parquet files."""
    base = base_dir or _DEFAULT_BASE_DIR
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.parquet"))


def get_date_range(
    stock_code: str,
    base_dir: Path | None = None,
) -> tuple[str | None, str | None]:
    """Return (first_date, last_date) of stored data for a stock.

    Returns (None, None) if no data exists.
    """
    rows = load_rows(stock_code, base_dir=base_dir)
    if not rows:
        return None, None
    dates = [str(r.get("date", "")) for r in rows if r.get("date")]
    if not dates:
        return None, None
    return min(dates), max(dates)
