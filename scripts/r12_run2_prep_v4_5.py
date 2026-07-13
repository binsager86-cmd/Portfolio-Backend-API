from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

MUTABLE_TABLES = [
    "ee_indicators",
    "ee_signals",
    "ee_ratings",
    "ee_symbol_state",
    "ee_positions",
    "ee_backtest_runs",
    "ee_backtest_trades",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_table(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(1) FROM {table}").fetchone()[0])


def drop_triggers(db_path: Path) -> dict[str, Any]:
    dropped: list[str] = []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
        for (name,) in rows:
            conn.execute(f'DROP TRIGGER IF EXISTS "{name}"')
            dropped.append(str(name))
        conn.commit()
    return {"dropped_count": len(dropped), "dropped_triggers": sorted(dropped)}


def ensure_masked_column(conn: sqlite3.Connection) -> None:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(ee_ohlcv)").fetchall()]
    if "is_masked" not in cols:
        conn.execute("ALTER TABLE ee_ohlcv ADD COLUMN is_masked INTEGER NOT NULL DEFAULT 0")


def build_v45_surface(canonical_db: Path, exam_v45: Path, mask_manifest: dict[str, Any]) -> dict[str, Any]:
    if exam_v45.exists():
        exam_v45.unlink()
    exam_v45.write_bytes(canonical_db.read_bytes())

    with sqlite3.connect(exam_v45) as conn:
        ensure_masked_column(conn)

        conn.execute("DROP TABLE IF EXISTS ee_mask_intervals")
        conn.execute(
            """
            CREATE TABLE ee_mask_intervals (
                symbol TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                source_final_class TEXT,
                source_rule TEXT
            )
            """
        )

        for m in mask_manifest["intervals"]:
            conn.execute(
                """
                INSERT INTO ee_mask_intervals(symbol, start_date, end_date, source_final_class, source_rule)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(m["symbol"]),
                    str(m["start_date"]),
                    str(m["end_date"]),
                    str(m.get("source_final_class") or ""),
                    str(m.get("source_rule") or ""),
                ),
            )

            conn.execute(
                """
                UPDATE ee_ohlcv
                SET is_masked = 1
                WHERE symbol = ? AND trade_date >= strftime('%s', ?) AND trade_date <= strftime('%s', ?)
                """,
                (str(m["symbol"]), str(m["start_date"]), str(m["end_date"])),
            )

        conn.execute("DROP TABLE IF EXISTS ee_ohlcv_unmasked_segmented")
        conn.execute("CREATE TABLE ee_ohlcv_unmasked_segmented AS SELECT * FROM ee_ohlcv WHERE 0")
        conn.execute("DROP TABLE IF EXISTS ee_symbol_segment_map")
        conn.execute(
            """
            CREATE TABLE ee_symbol_segment_map (
                original_symbol TEXT NOT NULL,
                segment_symbol TEXT NOT NULL,
                segment_id INTEGER NOT NULL,
                bars_count INTEGER NOT NULL,
                start_trade_date INTEGER,
                end_trade_date INTEGER
            )
            """
        )

        symbols = [str(r[0]) for r in conn.execute("SELECT DISTINCT symbol FROM ee_ohlcv ORDER BY symbol").fetchall()]
        segment_rows_inserted = 0
        segments_per_symbol: dict[str, int] = {}
        masked_bars = 0

        for sym in symbols:
            rows = conn.execute(
                "SELECT * FROM ee_ohlcv WHERE symbol = ? ORDER BY trade_date",
                (sym,),
            ).fetchall()
            if not rows:
                continue

            col_names = [d[1] for d in conn.execute("PRAGMA table_info(ee_ohlcv)").fetchall()]
            idx = {c: i for i, c in enumerate(col_names)}

            segment_id = 1
            current_rows: list[tuple[Any, ...]] = []
            saw_mask = False

            def flush_segment(rows_to_flush: list[tuple[Any, ...]], seg_id: int) -> None:
                nonlocal segment_rows_inserted
                if not rows_to_flush:
                    return
                seg_symbol = f"{sym}__SEG{seg_id:04d}"
                patched = []
                for row in rows_to_flush:
                    row_list = list(row)
                    row_list[idx["symbol"]] = seg_symbol
                    patched.append(tuple(row_list))
                placeholders = ",".join(["?"] * len(col_names))
                conn.executemany(
                    f"INSERT INTO ee_ohlcv_unmasked_segmented VALUES ({placeholders})",
                    patched,
                )
                tds = [int(r[idx["trade_date"]]) for r in rows_to_flush]
                conn.execute(
                    """
                    INSERT INTO ee_symbol_segment_map(original_symbol, segment_symbol, segment_id, bars_count, start_trade_date, end_trade_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (sym, seg_symbol, seg_id, len(rows_to_flush), min(tds), max(tds)),
                )
                segment_rows_inserted += len(rows_to_flush)

            for row in rows:
                is_masked = int(row[idx["is_masked"]] or 0) == 1
                if is_masked:
                    masked_bars += 1
                    flush_segment(current_rows, segment_id)
                    if current_rows:
                        segment_id += 1
                    current_rows = []
                    saw_mask = True
                    continue
                if saw_mask and not current_rows:
                    # Start a fresh segment immediately after a masked boundary.
                    saw_mask = False
                current_rows.append(row)

            flush_segment(current_rows, segment_id)

            seg_count = int(
                conn.execute(
                    "SELECT COUNT(1) FROM ee_symbol_segment_map WHERE original_symbol = ?",
                    (sym,),
                ).fetchone()[0]
            )
            segments_per_symbol[sym] = seg_count

        conn.commit()

        unmasked_rows = count_table(conn, "ee_ohlcv_unmasked_segmented")

    return {
        "exam_surface_db": str(exam_v45),
        "masked_interval_count": int(mask_manifest["interval_count"]),
        "masked_bar_count": masked_bars,
        "unmasked_segmented_bar_count": unmasked_rows,
        "segment_rows_inserted": segment_rows_inserted,
        "segments_per_symbol": segments_per_symbol,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="R12 run2 preparation (containment + v4.5 surface)")
    parser.add_argument("--canonical-db", required=True)
    parser.add_argument("--review-dir", required=True)
    args = parser.parse_args()

    canonical_db = Path(args.canonical_db).resolve()
    review_dir = Path(args.review_dir).resolve()

    exam_v44 = review_dir / "r12_exam_surface_v4_4.db"
    exam_v45 = review_dir / "r12_exam_surface_v4_5.db"

    mask_path = review_dir / "r12_masked_intervals_manifest_v4_3_final.json"
    seal_v44_path = review_dir / "r12_pre_exam_surface_seal_v4_4.json"
    add9_path = review_dir / "r12a_created_files_manifest_v4_4_addendum_9.json"

    mask_manifest = json.loads(mask_path.read_text(encoding="utf-8"))
    seal_v44 = json.loads(seal_v44_path.read_text(encoding="utf-8"))

    # Containment evidence from run1 implementation.
    run1_path_resolution = {
        "script": "scripts/r12_execute_exam_v1.py",
        "import_order_issue": "run_backtest imported before setting DATABASE_PATH in script runtime",
        "evidence_lines": {
            "import_run_backtest": 19,
            "set_database_path": 246,
            "run_backtest_call": 258,
        },
        "resolved_default_database_path_when_env_unset": str((Path(__file__).resolve().parents[1] / ".." / "dev_portfolio.db").resolve()),
    }

    traceback_excerpt = [
        "File scripts/r12_execute_exam_v1.py, line 258, in main -> report = run_backtest(...)",
        "File app/services/eagle_eye/backtest_service.py, line 202, in run_backtest -> compute_and_store_symbol(symbol)",
        "File app/services/eagle_eye/indicator_service.py, line 306, in store_indicator_results -> exec_sql(...)",
        "sqlite3.IntegrityError: R11_FREEZE_ACTIVE_ee_indicators",
    ]

    # Canonical integrity snapshot (used as pre-run2 baseline and post-prep check).
    with sqlite3.connect(canonical_db) as conn:
        baseline_counts = {t: count_table(conn, t) for t in MUTABLE_TABLES}
    baseline_hash = sha256_file(canonical_db)

    # Step 2: isolate exam v4.4 by dropping inherited triggers in the copy only.
    trigger_drop_result = drop_triggers(exam_v44)

    # Step 3: build v4.5 surface with masked bars flagged and segmented unmasked surface.
    engine_hash_before = {
        "pipeline.py": sha256_file(Path(__file__).resolve().parents[1] / "app/services/eagle_eye/pipeline.py"),
        "scanner_service.py": sha256_file(Path(__file__).resolve().parents[1] / "app/services/eagle_eye/scanner_service.py"),
    }

    segmentation = build_v45_surface(canonical_db, exam_v45, mask_manifest)

    engine_hash_after = {
        "pipeline.py": sha256_file(Path(__file__).resolve().parents[1] / "app/services/eagle_eye/pipeline.py"),
        "scanner_service.py": sha256_file(Path(__file__).resolve().parents[1] / "app/services/eagle_eye/scanner_service.py"),
    }

    with sqlite3.connect(canonical_db) as conn:
        post_prep_counts = {t: count_table(conn, t) for t in MUTABLE_TABLES}
    post_prep_hash = sha256_file(canonical_db)

    unchanged = (baseline_hash == post_prep_hash) and (baseline_counts == post_prep_counts)

    prep_payload = {
        "version_id": "R12_RUN2_PREP_V4_5",
        "scope": "Containment proof + exam-surface isolation + mask-semantics scaffolding",
        "authorization": {
            "run1_status": "FAILED_TECHNICAL_ACCEPTED",
            "run2_authorization_gate": "PASS_REQUIRED_ON_CONTAINMENT_AND_HASH_INVARIANCE",
        },
        "containment_proof": {
            "run1_path_resolution": run1_path_resolution,
            "run1_traceback_excerpt": traceback_excerpt,
            "canonical_integrity": {
                "baseline_source": "pre-run2 snapshot (accepted fallback to pre-run addendum-9 state)",
                "canonical_db_path": str(canonical_db),
                "baseline_sha256": baseline_hash,
                "baseline_mutable_table_counts": baseline_counts,
                "post_prep_sha256": post_prep_hash,
                "post_prep_mutable_table_counts": post_prep_counts,
                "unchanged": unchanged,
            },
            "abort_if_canonical_write_detected": not unchanged,
        },
        "exam_db_isolation": {
            "v44_copy_path": str(exam_v44),
            "trigger_drop_result": trigger_drop_result,
            "assertion": "Run2 runner will set DATABASE_PATH before importing engine modules and execute only against exam copy runtime DB.",
        },
        "mask_semantics": {
            "mask_manifest_path": str(mask_path),
            "mask_manifest_interval_count": int(mask_manifest["interval_count"]),
            "v45_surface": segmentation,
            "no_cross_seam_policy": {
                "no_return_across_mask": True,
                "no_indicator_lookback_across_mask": True,
                "force_flat_on_mask_boundary": True,
            },
        },
        "engine_hash_invariance": {
            "before": engine_hash_before,
            "after": engine_hash_after,
            "unchanged": engine_hash_before == engine_hash_after,
        },
        "input_seal_hash": {
            "seal_path": str(seal_v44_path),
            "seal_sha256": sha256_file(seal_v44_path),
            "addendum9_manifest_sha256": sha256_file(add9_path),
        },
        "ready_for_run2": bool(unchanged and (engine_hash_before == engine_hash_after)),
    }

    out_json = review_dir / "r12_run2_preparation_v4_5.json"
    out_md = review_dir / "r12_run2_preparation_v4_5.md"
    out_json.write_text(json.dumps(prep_payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")

    md_lines = [
        "# R12 Run-2 Preparation V4.5",
        "",
        f"- version_id: {prep_payload['version_id']}",
        f"- ready_for_run2: {prep_payload['ready_for_run2']}",
        "",
        "## Containment",
        f"- canonical_sha256_baseline: {baseline_hash}",
        f"- canonical_sha256_post_prep: {post_prep_hash}",
        f"- canonical_unchanged: {unchanged}",
        "",
        "## Isolation",
        f"- v44_trigger_drop_count: {trigger_drop_result['dropped_count']}",
        "",
        "## Segmentation",
        f"- masked_interval_count: {mask_manifest['interval_count']}",
        f"- masked_bar_count: {segmentation['masked_bar_count']}",
        f"- unmasked_segmented_bar_count: {segmentation['unmasked_segmented_bar_count']}",
        "",
        "## Engine Hash Invariance",
        f"- pipeline.py unchanged: {engine_hash_before['pipeline.py'] == engine_hash_after['pipeline.py']}",
        f"- scanner_service.py unchanged: {engine_hash_before['scanner_service.py'] == engine_hash_after['scanner_service.py']}",
        "",
    ]
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8", newline="\n")

    print("READY_FOR_RUN2", prep_payload["ready_for_run2"])
    print("CANONICAL_UNCHANGED", unchanged)
    print("ENGINE_HASH_UNCHANGED", prep_payload["engine_hash_invariance"]["unchanged"])
    print("V44_TRIGGERS_DROPPED", trigger_drop_result["dropped_count"])
    print("V45_MASKED_BARS", segmentation["masked_bar_count"])
    print("V45_SEGMENTED_ROWS", segmentation["unmasked_segmented_bar_count"])

    if prep_payload["containment_proof"]["abort_if_canonical_write_detected"]:
        return 2
    if not prep_payload["engine_hash_invariance"]["unchanged"]:
        return 3
    if not prep_payload["ready_for_run2"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
