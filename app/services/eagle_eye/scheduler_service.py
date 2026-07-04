from __future__ import annotations

import uuid
from typing import Any

from app.core.database import exec_sql, query_all
from app.core.security import TokenData
from app.services.eagle_eye.audit_service import create_event
from app.services.eagle_eye.indicator_service import compute_and_store_symbol, load_latest_indicator
from app.services.eagle_eye.market_data_service import (
    CONCEPT_VERSION,
    ensure_schema,
    get_active_config,
    latest_trade_date,
    list_symbols,
    now_ts,
    sync_from_legacy_cache,
)
from app.services.eagle_eye.ml_service import resolve_labels
from app.services.eagle_eye.rating_service import compute_rating_from_indicator, store_rating
from app.services.eagle_eye.risk_service import liquidity_filter
from app.services.eagle_eye.scanner_service import evaluate_symbol


def _system_actor() -> TokenData:
    return TokenData(user_id=0, username="system", is_admin=True)


def run_eod_pipeline(source: str = "scheduler", actor: TokenData | None = None) -> dict[str, Any]:
    ensure_schema()
    actor = actor or _system_actor()
    trace_id = str(uuid.uuid4())
    run_ts = now_ts()

    copied = sync_from_legacy_cache(run_ts, source="feed")
    run_date = latest_trade_date()
    if run_date is None:
        return {"status": "ok", "data": {"trace_id": trace_id, "symbols": 0, "advice": False}}

    exec_sql("DELETE FROM ee_signals WHERE trade_date = ?", (run_date,))
    exec_sql(
        "DELETE FROM ee_audit_events WHERE action = 'eod_pipeline_run' AND entity_type = 'pipeline' AND entity_id = ?",
        (f"eagle_eye:{run_date}",),
    )

    symbols = list_symbols()
    cfg = get_active_config()

    indicator_updates = 0
    rating_updates = 0
    transitions = 0
    failures: list[dict[str, Any]] = []

    for symbol in symbols:
        try:
            indicator_updates += compute_and_store_symbol(symbol)
            payload = load_latest_indicator(symbol, run_date)
            if not payload:
                continue
            liq_ok, liq_meta = liquidity_filter(symbol, float(cfg.get("min_daily_value_kwd", 100000.0)))
            liquidity_score = 100.0 if liq_ok else 20.0
            if not liq_ok:
                payload["liquidity_fail"] = liq_meta

            score, band, components = compute_rating_from_indicator(payload, liquidity_score=liquidity_score)
            store_rating(symbol, run_date, score, band, components)
            rating_updates += 1

            result = evaluate_symbol(symbol, run_date, score, cfg, trace_id=trace_id)
            if result.get("transition"):
                transitions += 1
        except Exception as exc:
            failures.append({"symbol": symbol, "error": str(exc)[:240]})
            create_event(
                {
                    "action": "data_quality_alert",
                    "entity_type": "symbol",
                    "entity_id": symbol,
                    "change_type": "data",
                    "risk_level": "high",
                    "trace_id": trace_id,
                    "source": source,
                    "requires_follow_up": True,
                    "metadata": {"error": str(exc)[:240]},
                    "concept_version": CONCEPT_VERSION,
                },
                actor,
            )

    labels_updated = resolve_labels(60)

    summary_event = create_event(
        {
            "action": "eod_pipeline_run",
            "entity_type": "pipeline",
            "entity_id": f"eagle_eye:{run_date}",
            "change_type": "workflow",
            "risk_level": "low" if not failures else "high",
            "trace_id": trace_id,
            "source": source,
            "metadata": {
                "run_date": run_date,
                "copied_ohlcv_rows": copied,
                "symbols": len(symbols),
                "indicator_updates": indicator_updates,
                "rating_updates": rating_updates,
                "transitions": transitions,
                "labels_updated": labels_updated,
                "errors": failures,
            },
            "concept_version": CONCEPT_VERSION,
        },
        actor,
    )

    return {
        "status": "ok",
        "data": {
            "trace_id": trace_id,
            "run_date": run_date,
            "source": source,
            "symbols": len(symbols),
            "copied_ohlcv_rows": copied,
            "indicator_updates": indicator_updates,
            "rating_updates": rating_updates,
            "transitions": transitions,
            "labels_updated": labels_updated,
            "errors": failures,
            "summary_audit_event_id": summary_event.get("id"),
            "advice": False,
        },
    }


def query_signals(
    signal_type: str | None = None,
    symbol: str | None = None,
    since: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    where = []
    params: list[Any] = []
    if signal_type:
        where.append("signal_type = ?")
        params.append(signal_type)
    if symbol:
        where.append("symbol = ?")
        params.append(symbol.upper())
    if since:
        where.append("trade_date >= ?")
        params.append(since)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    total_rows = query_all(f"SELECT COUNT(1) AS c FROM ee_signals {where_sql}", tuple(params))
    total = int(total_rows[0].get("c") or 0) if total_rows else 0

    rows = query_all(
        f"""
        SELECT id, created_at, symbol, trade_date, signal_type, phase_from, phase_to,
               score, price, stop_price, evidence_json, concept_version, config_hash,
               audit_event_id, outcome_label, outcome_return, outcome_at
        FROM ee_signals
        {where_sql}
        ORDER BY trade_date DESC, id DESC
        LIMIT ? OFFSET ?
        """,
        tuple(params + [limit, offset]),
    )

    import json

    items: list[dict[str, Any]] = []
    for row in rows or []:
        items.append(
            {
                "id": row.get("id"),
                "created_at": row.get("created_at"),
                "symbol": row.get("symbol"),
                "trade_date": row.get("trade_date"),
                "signal_type": row.get("signal_type"),
                "phase_from": row.get("phase_from"),
                "phase_to": row.get("phase_to"),
                "score": row.get("score"),
                "price": row.get("price"),
                "stop_price": row.get("stop_price"),
                "evidence": json.loads(str(row.get("evidence_json") or "{}")),
                "concept_version": row.get("concept_version"),
                "config_hash": row.get("config_hash"),
                "audit_event_id": row.get("audit_event_id"),
                "outcome_label": row.get("outcome_label"),
                "outcome_return": row.get("outcome_return"),
                "outcome_at": row.get("outcome_at"),
                "advice": False,
            }
        )
    return items, total


def get_signal_detail(signal_id: int) -> dict[str, Any] | None:
    rows, _ = query_signals(limit=1, offset=0)
    row = query_all("SELECT * FROM ee_signals WHERE id = ? LIMIT 1", (signal_id,))
    if not row:
        return None
    import json

    s = row[0]
    event = query_all("SELECT * FROM ee_audit_events WHERE id = ?", (s.get("audit_event_id"),))
    return {
        "signal": {
            "id": s.get("id"),
            "created_at": s.get("created_at"),
            "symbol": s.get("symbol"),
            "trade_date": s.get("trade_date"),
            "signal_type": s.get("signal_type"),
            "phase_from": s.get("phase_from"),
            "phase_to": s.get("phase_to"),
            "score": s.get("score"),
            "price": s.get("price"),
            "stop_price": s.get("stop_price"),
            "evidence": json.loads(str(s.get("evidence_json") or "{}")),
            "config_hash": s.get("config_hash"),
            "concept_version": s.get("concept_version"),
            "outcome_label": s.get("outcome_label"),
            "outcome_return": s.get("outcome_return"),
            "outcome_at": s.get("outcome_at"),
        },
        "audit_event": event[0] if event else None,
        "advice": False,
    }


def performance_summary() -> dict[str, Any]:
    rows = query_all(
        """
        SELECT signal_type,
               COUNT(1) AS n,
               AVG(CASE WHEN outcome_label = 'WIN' THEN 1.0 ELSE 0.0 END) AS hit_rate,
               AVG(CASE WHEN outcome_return > 0 THEN outcome_return END) AS avg_winner,
               AVG(CASE WHEN outcome_return < 0 THEN outcome_return END) AS avg_loser,
               AVG(outcome_return) AS expectancy
        FROM ee_signals
        WHERE outcome_label IS NOT NULL
        GROUP BY signal_type
        ORDER BY n DESC
        """,
        (),
    )
    by_type = []
    for r in rows or []:
        by_type.append(
            {
                "signal_type": r.get("signal_type"),
                "count": int(r.get("n") or 0),
                "hit_rate": float(r.get("hit_rate") or 0.0),
                "avg_winner": float(r.get("avg_winner") or 0.0),
                "avg_loser": float(r.get("avg_loser") or 0.0),
                "expectancy": float(r.get("expectancy") or 0.0),
            }
        )

    total = sum(int(x["count"]) for x in by_type)
    weighted_hit_rate = 0.0
    weighted_expectancy = 0.0
    if total > 0:
        weighted_hit_rate = sum(x["hit_rate"] * x["count"] for x in by_type) / total
        weighted_expectancy = sum(x["expectancy"] * x["count"] for x in by_type) / total

    return {
        "total_labeled_signals": total,
        "hit_rate": weighted_hit_rate,
        "expectancy": weighted_expectancy,
        "by_signal_type": by_type,
        "advice": False,
    }
