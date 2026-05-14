from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Optional

from app.core.config import get_settings

from .model_store import (
    ModelBundle,
    get_cache_root,
    load_model_bundle,
    model_exists,
    model_is_rejected,
)


def _load_event_index(models_root: Optional[str] = None) -> Dict[str, Any]:
    cache = get_cache_root(models_root)
    event_index = cache / "event_index.json"
    if not event_index.exists():
        return {}
    return json.loads(event_index.read_text(encoding="utf-8"))


def _load_event_count_from_db(ticker: str) -> int:
    settings = get_settings()
    conn = sqlite3.connect(settings.database_abs_path)
    cur = conn.cursor()

    candidates = ["ee_events_cache", "ee_forensic_events", "forensic_events", "eagle_eye_events"]
    table_name = None
    for table in candidates:
        row = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if row:
            table_name = table
            break

    if not table_name:
        conn.close()
        return 0

    try:
        count = cur.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE UPPER(ticker)=?",  # nosec B608
            (ticker.upper(),),
        ).fetchone()[0]
    except Exception:
        count = 0
    conn.close()
    return int(count or 0)


def resolve_model_for_ticker(
    ticker: str,
    models_root: Optional[str] = None,
) -> Optional[ModelBundle]:
    """
    Resolve best available ML model tier for a ticker.

    Tier policy:
      1) per_stock if >=100 events and accepted model exists
      2) per_sector if >=30 events and accepted sector model exists
      3) global baseline if accepted model exists
      4) None -> caller falls back to rules
    """
    symbol = ticker.upper().strip()
    index = _load_event_index(models_root=models_root)

    event_counts = {k.upper(): int(v) for k, v in index.get("event_counts_by_ticker", {}).items()}
    sector_map = {k.upper(): str(v) for k, v in index.get("ticker_sector_map", {}).items()}

    event_count = event_counts.get(symbol)
    if event_count is None:
        event_count = _load_event_count_from_db(symbol)

    sector = sector_map.get(symbol, "holding_misc")

    if event_count >= 100:
        if model_exists("per_stock", symbol, models_root=models_root) and not model_is_rejected("per_stock", symbol, models_root=models_root):
            bundle = load_model_bundle(tier="per_stock", identifier=symbol, version="current", models_root=models_root)
            if bundle is not None:
                bundle.metadata["resolved_tier"] = "per_stock"
                bundle.metadata["resolved_event_count"] = event_count
                return bundle

    if event_count >= 30:
        if model_exists("per_sector", sector, models_root=models_root) and not model_is_rejected("per_sector", sector, models_root=models_root):
            bundle = load_model_bundle(tier="per_sector", identifier=sector, version="current", models_root=models_root)
            if bundle is not None:
                bundle.metadata["resolved_tier"] = "per_sector"
                bundle.metadata["resolved_event_count"] = event_count
                return bundle

    if model_exists("global", "baseline", models_root=models_root) and not model_is_rejected("global", "baseline", models_root=models_root):
        bundle = load_model_bundle(tier="global", identifier="baseline", version="current", models_root=models_root)
        if bundle is not None:
            bundle.metadata["resolved_tier"] = "global"
            bundle.metadata["resolved_event_count"] = event_count
            return bundle

    return None
