from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _q1(cur: sqlite3.Cursor, sql: str, params: tuple = ()) -> sqlite3.Row | None:
    cur.execute(sql, params)
    return cur.fetchone()


def _qall(cur: sqlite3.Cursor, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    cur.execute(sql, params)
    return cur.fetchall()


def run_phase0(output_dir: Path) -> dict:
    from app.core.config import get_settings
    from app.data.stock_lists import KUWAIT_STOCKS
    from app.services.eagle_eye import audit_service
    from app.services.eagle_eye.market_data_service import ensure_schema, ingest_tickerchart, list_symbols
    from app.services.eagle_eye.indicator_service import compute_and_store_symbol

    settings = get_settings()
    candidate_db = Path(settings.database_abs_path).resolve()

    if candidate_db.exists():
        candidate_db.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    audit_service.ensure_schema()
    ensure_schema()

    symbols = sorted({str(s.get("symbol") or "").upper().strip() for s in KUWAIT_STOCKS if str(s.get("symbol") or "").strip()})

    ingest_totals = {
        "rows_upserted": 0,
        "anomalies_count": 0,
        "quarantined_symbols": 0,
    }
    ingest_processed: list[dict] = []
    for i, sym in enumerate(symbols, start=1):
        print(f"[INGEST] {i}/{len(symbols)} {sym}", flush=True)
        try:
            result = ingest_tickerchart(symbols=[sym], source="phase0_clean_candidate")
        except Exception as exc:  # noqa: BLE001
            ingest_processed.append({"symbol": sym, "error": str(exc)})
            print(f"[INGEST] {sym} ERROR: {exc}", flush=True)
            continue

        ingest_totals["rows_upserted"] += int(result.get("rows_upserted") or 0)
        ingest_totals["anomalies_count"] += int(result.get("anomalies_count") or 0)
        ingest_totals["quarantined_symbols"] += int(result.get("quarantined_symbols") or 0)
        ingest_processed.extend(result.get("processed") or [])

    indicator_rows = 0
    for i, sym in enumerate(symbols, start=1):
        print(f"[INDICATORS] {i}/{len(symbols)} {sym}", flush=True)
        try:
            indicator_rows += int(compute_and_store_symbol(sym))
        except Exception as exc:  # noqa: BLE001
            print(f"[INDICATORS] {sym} ERROR: {exc}", flush=True)

    conn = sqlite3.connect(candidate_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows_total = int((_q1(cur, "SELECT COUNT(1) AS c FROM ee_ohlcv") or {"c": 0})["c"])
    ind_total = int((_q1(cur, "SELECT COUNT(1) AS c FROM ee_indicators") or {"c": 0})["c"])
    sym_total = int((_q1(cur, "SELECT COUNT(DISTINCT symbol) AS c FROM ee_ohlcv") or {"c": 0})["c"])
    q_open = int((_q1(cur, "SELECT COUNT(1) AS c FROM ee_data_quality_quarantine WHERE status='quarantined'") or {"c": 0})["c"])

    run_rows = _qall(
        cur,
        "SELECT run_id, status, rows_written, source_type, source_ref, synthetic_flag FROM ee_ingestion_runs ORDER BY started_at",
    )

    missing_run_refs = int(
        (_q1(cur, "SELECT COUNT(1) AS c FROM ee_ohlcv o LEFT JOIN ee_ingestion_runs r ON r.run_id=o.ingestion_run_id WHERE r.run_id IS NULL") or {"c": 0})[
            "c"
        ]
    )
    failed_residue = int(
        (_q1(cur, "SELECT COUNT(1) AS c FROM ee_ohlcv o JOIN ee_ingestion_runs r ON r.run_id=o.ingestion_run_id WHERE r.status='failed'") or {"c": 0})[
            "c"
        ]
    )

    adjustment_breakdown = [dict(r) for r in _qall(cur, "SELECT adjustment_status, COUNT(1) AS rows FROM ee_ohlcv GROUP BY adjustment_status ORDER BY adjustment_status")]
    raw_adjusted_dupes = int(
        (_q1(cur, "SELECT COUNT(1) AS c FROM (SELECT symbol, trade_date, COUNT(DISTINCT adjustment_status) AS kinds FROM ee_ohlcv GROUP BY symbol, trade_date HAVING kinds > 1) t") or {"c": 0})[
            "c"
        ]
    )

    span_rows = _qall(cur, "SELECT symbol, MIN(trade_date) AS min_td, MAX(trade_date) AS max_td, COUNT(1) AS bars FROM ee_ohlcv GROUP BY symbol ORDER BY symbol")
    symbol_spans = []
    bar_counts = []
    for r in span_rows:
        bar_counts.append(int(r["bars"]))
        symbol_spans.append(
            {
                "symbol": r["symbol"],
                "min_trade_date": int(r["min_td"]),
                "max_trade_date": int(r["max_td"]),
                "bars": int(r["bars"]),
            }
        )

    bench = ["BPCC", "SANAM", "TIJARA", "ZAIN", "MABANEE"]
    pit = {}
    for sym in bench:
        td = [int(x[0]) for x in _qall(cur, "SELECT trade_date FROM ee_ohlcv WHERE symbol=? ORDER BY trade_date", (sym,))]
        if not td:
            pit[sym] = {"error": "missing_symbol"}
            continue

        idxs = sorted({max(0, int((len(td) - 1) * p)) for p in (0.25, 0.5, 0.75, 1.0)})
        cuts = [td[i] for i in idxs]
        snapshots = []

        for asof in cuts:
            pr = _q1(cur, "SELECT close, volume, value_kwd, adjustment_status FROM ee_ohlcv WHERE symbol=? AND trade_date=?", (sym, asof))
            ir = _q1(cur, "SELECT payload_json FROM ee_indicators WHERE symbol=? AND trade_date=?", (sym, asof))
            payload = json.loads(ir["payload_json"]) if ir and ir["payload_json"] else {}
            snapshots.append(
                {
                    "as_of_trade_date": asof,
                    "left_of_line_row_count": int((_q1(cur, "SELECT COUNT(1) AS c FROM ee_ohlcv WHERE symbol=? AND trade_date<=?", (sym, asof)) or {"c": 0})["c"]),
                    "close": float(pr["close"]) if pr and pr["close"] is not None else None,
                    "volume": float(pr["volume"]) if pr and pr["volume"] is not None else None,
                    "value_kwd": float(pr["value_kwd"]) if pr and pr["value_kwd"] is not None else None,
                    "adjustment_status": pr["adjustment_status"] if pr else None,
                    "indicators": {
                        "rsi_14": payload.get("rsi_14"),
                        "adx_19": payload.get("adx_19"),
                        "ema10": payload.get("ema10"),
                        "ema30": payload.get("ema30"),
                        "sma200": payload.get("sma200"),
                        "rel_volume": payload.get("rel_volume"),
                        "atr_pct": payload.get("atr_pct"),
                    },
                }
            )

        pit[sym] = {"total_rows": len(td), "as_of_boundaries": cuts, "snapshots": snapshots}

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": settings.ENVIRONMENT,
        "use_postgres": settings.use_postgres,
        "database_path": str(candidate_db),
        "symbols_requested": len(symbols),
        "symbols_loaded": len(list_symbols()),
        "ingest_result": {
            **ingest_totals,
            "processed": ingest_processed,
        },
        "indicator_rows_written": indicator_rows,
        "census": {
            "ohlcv_rows": rows_total,
            "indicator_rows": ind_total,
            "symbol_count": sym_total,
            "quarantine_open": q_open,
            "bar_count_min": min(bar_counts) if bar_counts else 0,
            "bar_count_median": statistics.median(bar_counts) if bar_counts else 0,
            "bar_count_max": max(bar_counts) if bar_counts else 0,
        },
        "ingestion_integrity": {
            "run_count": len(run_rows),
            "missing_run_references": missing_run_refs,
            "failed_run_row_residue": failed_residue,
            "runs": [dict(r) for r in run_rows],
        },
        "raw_adjusted_separation": {
            "distinct_adjustment_status": adjustment_breakdown,
            "mixed_status_same_symbol_trade_date": raw_adjusted_dupes,
        },
    }

    (output_dir / "phase0_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    (output_dir / "symbol_spans.json").write_text(json.dumps(symbol_spans, ensure_ascii=True, indent=2), encoding="utf-8")
    (output_dir / "pit_benchmark_snapshots.json").write_text(json.dumps(pit, ensure_ascii=True, indent=2), encoding="utf-8")
    (output_dir / "phase0_evidence.md").write_text(
        "\n".join(
            [
                "# Phase 0 Evidence",
                "",
                f"- Candidate DB: {candidate_db}",
                f"- Symbols requested: {len(symbols)}",
                f"- Symbols loaded: {len(list_symbols())}",
                f"- OHLCV rows: {rows_total}",
                f"- Indicator rows: {ind_total}",
                f"- Open quarantine rows: {q_open}",
                f"- Ingestion run refs missing: {missing_run_refs}",
                f"- Failed-run residue rows: {failed_residue}",
                f"- Mixed raw/adjusted rows (same symbol/date): {raw_adjusted_dupes}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    conn.close()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Eagle Eye Phase 0 clean candidate build and evidence pack.")
    parser.add_argument("--output-dir", required=True, help="Output directory for artifacts.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    summary = run_phase0(out)
    print(json.dumps({"output_dir": str(out), "symbols_loaded": summary.get("symbols_loaded")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
