from __future__ import annotations

import json
from typing import Any

from app.core.database import exec_sql, query_all
from app.services.eagle_eye.market_data_service import CONCEPT_VERSION


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:
            return default
        return out
    except Exception:
        return default


def compute_rating_from_indicator(payload: dict[str, Any], liquidity_score: float = 100.0) -> tuple[float, str, dict[str, Any]]:
    obv_s = _to_float(payload.get("obv_slope_40"))
    anv_s = _to_float(payload.get("anv_slope_40"))
    cmf = _to_float(payload.get("cmf_10"))
    width = _to_float(payload.get("range_width_pct"), 1.0)
    base_duration = 120.0 if width <= 0.18 else max(0.0, 120.0 * (0.30 - min(width, 0.30)) / 0.30)

    accumulation_quality = _clamp(
        35 * _clamp((obv_s + 0.2) * 250)
        / 100
        + 25 * _clamp((anv_s + 0.2) * 250)
        / 100
        + 20 * _clamp((cmf + 0.2) * 250)
        / 100
        + 10 * _clamp((0.20 - width) * 500)
        / 100
        + 10 * _clamp(base_duration * (100 / 120))
        / 100
    )

    close = _to_float(payload.get("close"), 1.0)
    sma200 = _to_float(payload.get("sma200"), close)
    sma200_slope = _to_float(payload.get("sma200_slope"))
    ema10 = _to_float(payload.get("ema10"), close)
    ema30 = _to_float(payload.get("ema30"), close)
    ema10_slope = _to_float(payload.get("ema10_slope"))
    dist_h = _to_float(payload.get("dist_52w_high"), 1.0)

    trend_structure = _clamp(
        (20 if close >= sma200 else 0)
        + _clamp((sma200_slope + 0.02) * 500, 0, 20)
        + (20 if ema10 >= ema30 else 0)
        + _clamp((ema10_slope + 0.03) * 500, 0, 20)
        + _clamp((0.50 - dist_h) * 200, 0, 20)
    )

    rsi = _to_float(payload.get("rsi_14"), 50)
    macd_hist = _to_float(payload.get("macd_hist"))
    adx = _to_float(payload.get("adx_19"))
    plus_di = _to_float(payload.get("plus_di"))
    minus_di = _to_float(payload.get("minus_di"))

    rsi_score = 100 - abs(rsi - 62) * 3
    if rsi > 80:
        rsi_score -= 20
    momentum = _clamp(rsi_score * 0.45 + _clamp((macd_hist + 0.5) * 100) * 0.25 + _clamp((adx - 10) * 4) * 0.20 + (10 if plus_di > minus_di else 0))

    rel_volume = _to_float(payload.get("rel_volume"), 1.0)
    lnv = _to_float(payload.get("liquidity_net_value"))
    volume_confirmation = _clamp(_clamp(rel_volume * 35) + _clamp((lnv / 100000.0) + 50) * 0.4 + liquidity_score * 0.25)

    atr_pct = _to_float(payload.get("atr_pct"), 0.03)
    atr_quality = 100 - _clamp(abs(atr_pct - 0.03) * 3000)
    risk_quality = _clamp(atr_quality * 0.45 + liquidity_score * 0.55)

    total = (
        accumulation_quality * 0.30
        + trend_structure * 0.25
        + momentum * 0.20
        + volume_confirmation * 0.15
        + risk_quality * 0.10
    )

    score = round(_clamp(total), 2)
    if score >= 75:
        band = "A"
    elif score >= 60:
        band = "B"
    elif score >= 40:
        band = "C"
    else:
        band = "D"

    components = {
        "accumulation_quality": round(accumulation_quality, 2),
        "trend_structure": round(trend_structure, 2),
        "momentum": round(momentum, 2),
        "volume_confirmation": round(volume_confirmation, 2),
        "risk_quality": round(risk_quality, 2),
        "liquidity_score": round(liquidity_score, 2),
        "weights": {
            "accumulation_quality": 0.30,
            "trend_structure": 0.25,
            "momentum": 0.20,
            "volume_confirmation": 0.15,
            "risk_quality": 0.10,
        },
        "fundamental_pe_component": {"value": 50.0, "stubbed": True},
        "news_density_component": {"value": 50.0, "stubbed": True},
    }
    return score, band, components


def store_rating(symbol: str, trade_date: int, score: float, band: str, components: dict[str, Any]) -> None:
    exec_sql(
        """
        INSERT INTO ee_ratings (symbol, trade_date, score, band, components_json, concept_version)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, trade_date) DO UPDATE SET
            score = excluded.score,
            band = excluded.band,
            components_json = excluded.components_json,
            concept_version = excluded.concept_version
        """,
        (
            symbol,
            trade_date,
            score,
            band,
            json.dumps(components, ensure_ascii=True, separators=(",", ":")),
            CONCEPT_VERSION,
        ),
    )


def load_rating_history(symbol: str, limit: int = 200) -> list[dict[str, Any]]:
    rows = query_all(
        """
        SELECT symbol, trade_date, score, band, components_json, concept_version
        FROM ee_ratings
        WHERE symbol = ?
        ORDER BY trade_date DESC
        LIMIT ?
        """,
        (symbol, limit),
    )
    out: list[dict[str, Any]] = []
    for row in rows or []:
        out.append(
            {
                "symbol": row.get("symbol"),
                "trade_date": row.get("trade_date"),
                "score": row.get("score"),
                "band": row.get("band"),
                "components": json.loads(str(row.get("components_json") or "{}")),
                "concept_version": row.get("concept_version"),
            }
        )
    return out
