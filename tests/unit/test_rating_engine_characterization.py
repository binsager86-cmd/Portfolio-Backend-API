from __future__ import annotations

import importlib
import json
import math
import os
import sqlite3
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytest

from app.services.eagle_eye.adapter import TickerChartAdapter
from app.services.eagle_eye.indicators import compute_all_indicators
from app.services.eagle_eye.ingest import (
    RATINGS_INDICATOR_LOOKBACK_BARS,
    _build_premier_market_proxy,
)
from app.services.eagle_eye.rating_engine import is_stock_active
from app.services.eagle_eye.scoring.family_scores import compute_family_scores
from app.services.eagle_eye.scoring.recommendation_engine import generate_recommendation
from app.services.eagle_eye.stage_classifier import classify_stage_with_confidence
from app.services.eagle_eye.store import list_tickers_with_ohlcv, load_ohlcv


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "phase0_ratings_golden.json"
)
BASELINE_DB_PATH = Path(__file__).resolve().parents[3] / "dev_portfolio.db"
# Frozen as-of date to make characterization deterministic and independent of
# newly arrived market bars in ee_ohlcv_cache.
AS_OF = "2026-06-10"
AS_OF_TS = pd.Timestamp(AS_OF)


def _is_baseline_available() -> bool:
    if not BASELINE_DB_PATH.exists():
        return False
    try:
        conn = sqlite3.connect(str(BASELINE_DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM ee_ohlcv_cache WHERE bar_date <= ?",
            (AS_OF,),
        )
        count = int(cur.fetchone()[0])
        conn.close()
        return count > 0
    except Exception:
        return False


def _safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _build_market_proxy() -> Optional[pd.Series]:
    adapter = TickerChartAdapter()
    stock_meta = adapter.list_stocks()
    premier_tickers = [
        s.ticker
        for s in stock_meta
        if str(getattr(s, "market_tier", "premier") or "premier").strip().upper()
        == "PREMIER"
    ]
    return _build_premier_market_proxy(premier_tickers)


def _normalize_record(latest: dict, stage: str, recommendation_payload: dict) -> dict:
    confidence = _safe_float(recommendation_payload.get("confidence"))
    rr = _safe_float(latest.get("risk_reward_ratio"))
    return {
        "stage": stage,
        "rating": str(recommendation_payload.get("recommendation")),
        "confidence": round(confidence, 1) if confidence is not None else None,
        "continue_rising": bool(recommendation_payload.get("continue_rising", False)),
        "risky_near_resistance": bool(
            recommendation_payload.get("risky_near_resistance", False)
        ),
        "risk_reward_ratio": round(rr, 2) if rr is not None else None,
    }


def _placeholder_record(reason: str) -> dict:
    stage_map = {
        "insufficient_history": "INSUFFICIENT_HISTORY",
        "inactive_or_delisted": "INACTIVE_OR_DELISTED",
        "indicator_unavailable": "INDICATOR_UNAVAILABLE",
    }
    return {
        "stage": stage_map.get(reason, "DATA_ISSUE"),
        "rating": "AVOID",
        "confidence": 0.0,
        "continue_rising": False,
        "risky_near_resistance": False,
        "risk_reward_ratio": None,
    }


def _recompute_snapshot() -> Dict[str, dict]:
    # tests/conftest.py redirects DATABASE_PATH to a temporary DB; characterization
    # must pin to the shared baseline snapshot used by ingest recompute.
    os.environ["DATABASE_PATH"] = str(BASELINE_DB_PATH)
    from app.core.config import get_settings
    import app.core.database as db_module

    get_settings.cache_clear()
    importlib.reload(db_module)

    market_proxy = _build_market_proxy()
    snapshot: Dict[str, dict] = {}
    for ticker in list_tickers_with_ohlcv():
        df = load_ohlcv(ticker)
        if df is not None and len(df) > 0:
            df = df[df.index <= AS_OF_TS]
        if df is None or len(df) < 50:
            snapshot[ticker.upper()] = _placeholder_record("insufficient_history")
            continue
        if not is_stock_active(ticker, df):
            snapshot[ticker.upper()] = _placeholder_record("inactive_or_delisted")
            continue

        indicator_input = (
            df.tail(RATINGS_INDICATOR_LOOKBACK_BARS)
            if len(df) > RATINGS_INDICATOR_LOOKBACK_BARS
            else df
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            ind_df = compute_all_indicators(indicator_input, market_close=market_proxy)
        if ind_df is None or len(ind_df) == 0:
            snapshot[ticker.upper()] = _placeholder_record("indicator_unavailable")
            continue

        latest = ind_df.iloc[-1].to_dict()
        family_scores = compute_family_scores(latest)
        stage, stage_conf = classify_stage_with_confidence(
            latest,
            family_scores=family_scores,
        )
        dq = _safe_float(latest.get("data_quality_score"))
        recommendation_payload = generate_recommendation(
            latest,
            family_scores=family_scores,
            total_score=float(family_scores.get("total_score", 50.0)),
            stage=stage,
            stage_conf=stage_conf,
            pattern_match=None,
            data_quality=dq if dq is not None else 50.0,
        )
        snapshot[ticker.upper()] = _normalize_record(latest, stage, recommendation_payload)

    return dict(sorted(snapshot.items()))


def test_rating_engine_characterization_baseline() -> None:
    if not _is_baseline_available():
        pytest.skip("baseline DB / as-of bars not present")

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    expected = fixture["ratings"]
    current = _recompute_snapshot()

    assert current == expected

    distribution = Counter(v["rating"] for v in current.values())
    assert fixture["as_of"] == AS_OF
    assert dict(distribution) == fixture["distribution"]
    assert sum(distribution.values()) == int(fixture["universe_size"])


def _build_fixture_document(snapshot: Dict[str, dict]) -> dict:
    distribution = Counter(v["rating"] for v in snapshot.values())
    return {
        "as_of": AS_OF,
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "distribution": dict(distribution),
        "universe_size": len(snapshot),
        "ratings": snapshot,
    }