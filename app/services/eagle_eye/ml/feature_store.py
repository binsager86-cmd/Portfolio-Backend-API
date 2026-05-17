"""
ml/feature_store.py — Phase 1: Versioned per-stock feature store.

Stores feature matrices as Parquet files on disk, one file per
(stock, schema_version).  A JSON schema sidecar records every column,
its dtype, and its leakage audit status.

Directory layout
----------------
<store_root>/
  v1/
    TICKER/
      features.parquet
      schema.json
    _index.json            ← all tickers available at this version

Usage
-----
    from app.services.eagle_eye.ml.feature_store import FeatureStore

    fs = FeatureStore()                           # uses default root
    fs.save(ticker="ZAIN", version="v1", df=features_df, meta={"n_events": 210})
    df = fs.load(ticker="ZAIN", version="v1")
    available = fs.list_tickers(version="v1")
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default root is sibling of the ml_models directory produced by model_store.py
_DEFAULT_STORE_ROOT = Path(__file__).resolve().parents[4] / "ml_feature_store"


class FeatureStore:
    """
    Versioned on-disk Parquet store for per-stock feature matrices.

    Parameters
    ----------
    root : path-like, optional
        Base directory.  Created if it does not exist.
    """

    def __init__(self, root: Optional[Path | str] = None) -> None:
        self.root = Path(root) if root is not None else _DEFAULT_STORE_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self,
        *,
        ticker: str,
        version: str,
        df: pd.DataFrame,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Persist a feature DataFrame for one stock.

        Parameters
        ----------
        ticker  : stock ticker (e.g. "ZAIN")
        version : schema version string (e.g. "v1")
        df      : feature DataFrame — index should be the event/bar dates
        meta    : extra metadata to store in schema.json
        """
        stock_dir = self._stock_dir(ticker, version)
        stock_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = stock_dir / "features.parquet"
        df.to_parquet(parquet_path, index=True, engine="pyarrow")

        # Schema sidecar
        schema = self._build_schema(df, ticker, version, meta or {})
        (stock_dir / "schema.json").write_text(
            json.dumps(schema, indent=2, default=str), encoding="utf-8"
        )

        # Update version index
        self._update_index(ticker, version)

        logger.info(
            "FeatureStore.save: ticker=%s version=%s rows=%d cols=%d → %s",
            ticker, version, len(df), len(df.columns), parquet_path,
        )
        return parquet_path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(
        self,
        *,
        ticker: str,
        version: str,
        columns: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load the feature DataFrame for a stock.

        Parameters
        ----------
        ticker     : stock ticker
        version    : schema version
        columns    : optional column subset to load (saves memory)
        start_date : optional ISO date lower bound (inclusive)
        end_date   : optional ISO date upper bound (inclusive)
        """
        parquet_path = self._stock_dir(ticker, version) / "features.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Feature store: no data for ticker={ticker} version={version}"
            )

        df = pd.read_parquet(parquet_path, columns=columns, engine="pyarrow")

        # Date filtering — works whether index is datetime or string
        if start_date or end_date:
            idx = pd.to_datetime(df.index, errors="coerce")
            if start_date:
                df = df[idx >= pd.Timestamp(start_date)]
            if end_date:
                df = df[idx <= pd.Timestamp(end_date)]

        return df

    # ------------------------------------------------------------------
    # Schema / meta helpers
    # ------------------------------------------------------------------

    def load_schema(self, *, ticker: str, version: str) -> Dict[str, Any]:
        """Return the schema.json dict for a stock/version."""
        schema_path = self._stock_dir(ticker, version) / "schema.json"
        if not schema_path.exists():
            return {}
        return json.loads(schema_path.read_text(encoding="utf-8"))

    def exists(self, *, ticker: str, version: str) -> bool:
        return (self._stock_dir(ticker, version) / "features.parquet").exists()

    def list_tickers(self, version: str = "v1") -> List[str]:
        """Return all tickers that have a saved feature file for this version."""
        index_path = self.root / version / "_index.json"
        if not index_path.exists():
            return []
        return json.loads(index_path.read_text(encoding="utf-8")).get("tickers", [])

    def delete(self, *, ticker: str, version: str) -> None:
        """Remove all feature files for one stock+version."""
        import shutil
        stock_dir = self._stock_dir(ticker, version)
        if stock_dir.exists():
            shutil.rmtree(stock_dir)
            logger.info("FeatureStore.delete: %s version=%s", ticker, version)
        # Remove from index
        index_path = self.root / version / "_index.json"
        if index_path.exists():
            idx_data = json.loads(index_path.read_text(encoding="utf-8"))
            tickers = [t for t in idx_data.get("tickers", []) if t != ticker.upper()]
            idx_data["tickers"] = tickers
            idx_data["updated_at"] = datetime.utcnow().isoformat()
            index_path.write_text(json.dumps(idx_data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stock_dir(self, ticker: str, version: str) -> Path:
        return self.root / version / ticker.upper()

    def _build_schema(
        self,
        df: pd.DataFrame,
        ticker: str,
        version: str,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        columns_info = {
            col: {
                "dtype": str(df[col].dtype),
                "n_non_null": int(df[col].notna().sum()),
                "nan_rate": round(float(df[col].isna().mean()), 4),
            }
            for col in df.columns
        }
        return {
            "ticker": ticker.upper(),
            "version": version,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "created_at": datetime.utcnow().isoformat(),
            "columns": columns_info,
            **meta,
        }

    def _update_index(self, ticker: str, version: str) -> None:
        index_path = self.root / version / "_index.json"
        if index_path.exists():
            idx_data = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            idx_data = {"version": version, "tickers": []}

        tickers: list[str] = idx_data.get("tickers", [])
        t_upper = ticker.upper()
        if t_upper not in tickers:
            tickers.append(t_upper)
            tickers.sort()
        idx_data["tickers"] = tickers
        idx_data["updated_at"] = datetime.utcnow().isoformat()

        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(idx_data, indent=2), encoding="utf-8")
