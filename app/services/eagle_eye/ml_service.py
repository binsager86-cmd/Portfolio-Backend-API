from __future__ import annotations

from typing import Any

from app.core.database import exec_sql, query_all, query_val


def labeled_signal_count() -> int:
    return int(
        query_val(
            "SELECT COUNT(1) FROM ee_signals WHERE outcome_label IS NOT NULL",
            (),
        )
        or 0
    )


def estimate_ml_probability(evidence: dict[str, Any]) -> float:
    score = 0.50
    if bool(evidence.get("accumulation_divergence")):
        score += 0.08
    if float(evidence.get("cmf_10") or 0.0) > 0.05:
        score += 0.05
    if float(evidence.get("rel_volume") or 0.0) >= 2.5:
        score += 0.07
    if float(evidence.get("adx_19") or 0.0) >= 22:
        score += 0.05
    if bool(evidence.get("distribution_divergence")):
        score -= 0.20
    return max(0.0, min(1.0, score))


def apply_ml_gate(evidence: dict[str, Any], config: dict[str, Any]) -> tuple[bool, float | None]:
    enabled = bool(config.get("ml_gate_enabled", False))
    if not enabled:
        return True, None

    min_count = int(config.get("ml_min_labeled_signals", 150))
    if labeled_signal_count() < min_count:
        return True, None

    prob = estimate_ml_probability(evidence)
    min_prob = float(config.get("ml_prob_min", 0.45))
    return prob >= min_prob, prob


def resolve_labels(barrier_days: int = 60) -> int:
    rows = query_all(
        """
        SELECT id, symbol, trade_date, price
        FROM ee_signals
        WHERE signal_type IN ('ACCUMULATION_ALERT', 'BREAKOUT_CONFIRMED')
          AND outcome_label IS NULL
        """,
        (),
    )

    updated = 0
    for row in rows or []:
        signal_id = int(row.get("id") or 0)
        symbol = str(row.get("symbol") or "")
        trade_date = int(row.get("trade_date") or 0)
        entry_price = float(row.get("price") or 0.0)
        if signal_id <= 0 or not symbol or entry_price <= 0:
            continue

        bars = query_all(
            """
            SELECT trade_date, high, low, close
            FROM ee_ohlcv
            WHERE symbol = ? AND trade_date > ?
            ORDER BY trade_date ASC
            LIMIT ?
            """,
            (symbol, trade_date, barrier_days),
        )
        if not bars or len(bars) < barrier_days:
            continue

        atr_row = query_all(
            """
            SELECT payload_json
            FROM ee_indicators
            WHERE symbol = ? AND trade_date = ?
            LIMIT 1
            """,
            (symbol, trade_date),
        )
        atr = 0.0
        if atr_row:
            import json

            try:
                atr = float(json.loads(str(atr_row[0].get("payload_json") or "{}")).get("atr_14") or 0.0)
            except Exception:
                atr = 0.0
        if atr <= 0:
            continue

        upper = entry_price + (3.0 * atr)
        lower = entry_price - (1.5 * atr)

        label = "TIMEOUT"
        outcome_at = int(bars[-1].get("trade_date") or 0)
        outcome_return = (float(bars[-1].get("close") or entry_price) - entry_price) / entry_price

        for b in bars:
            high = float(b.get("high") or 0.0)
            low = float(b.get("low") or 0.0)
            if high >= upper:
                label = "WIN"
                outcome_at = int(b.get("trade_date") or 0)
                outcome_return = (upper - entry_price) / entry_price
                break
            if low <= lower:
                label = "LOSS"
                outcome_at = int(b.get("trade_date") or 0)
                outcome_return = (lower - entry_price) / entry_price
                break

        exec_sql(
            """
            UPDATE ee_signals
            SET outcome_label = ?, outcome_return = ?, outcome_at = ?
            WHERE id = ?
            """,
            (label, outcome_return, outcome_at, signal_id),
        )
        updated += 1

    return updated
