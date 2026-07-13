from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_ACCEPTED = 212
EXPECTED_DEFERRED = 193
EXPECTED_TOTAL = 405
EXPECTED_MASKED = 193


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def build_mask_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    intervals: list[dict[str, Any]] = []
    for r in rows:
        if not bool(r.get("masked_interval")):
            continue
        intervals.append(
            {
                "symbol": str(r["symbol"]),
                "start_date": str(r["prior_bar_date"]),
                "end_date": str(r["event_bar_date"]),
                "source_final_class": str(r.get("final_class_v4_2") or "UNSPECIFIED"),
                "source_rule": str(r.get("annotation") or "UNSPECIFIED"),
            }
        )

    dedup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for m in intervals:
        key = (m["symbol"], m["start_date"], m["end_date"], m["source_final_class"])
        dedup[key] = m

    sorted_intervals = sorted(dedup.values(), key=lambda x: (x["symbol"], x["start_date"], x["end_date"]))
    return {
        "version_id": "R12_MASKED_INTERVALS_MANIFEST_V4_3_FINAL",
        "scope": "R-3 + R-7 + R-9 intervals; masked-as-is for exam surface",
        "interval_count": len(sorted_intervals),
        "intervals": sorted_intervals,
    }


def apply_r9(rows: list[dict[str, Any]], ca_v0_2: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cap_keys = {
        (str(r["symbol"]), str(r["prior_bar_date"]), str(r["event_bar_date"]))
        for r in ca_v0_2["corrected_annotations"]
        if str(r.get("suspected_action")) == "CAPITAL_DECREASE"
    }

    updated: list[dict[str, Any]] = []
    changed_events: list[dict[str, Any]] = []
    confirm_no_change: list[dict[str, Any]] = []

    for r in rows:
        out = dict(r)
        key = (str(out["symbol"]), str(out["prior_bar_date"]), str(out["event_bar_date"]))
        if key in cap_keys:
            prev_disp = str(out.get("disposition") or "")
            prev_mask = bool(out.get("masked_interval"))
            out["disposition"] = "DEFERRED_TO_CA_LEDGER"
            out["masked_interval"] = True
            out["annotation"] = "R-9 CAPITAL_DECREASE_OVERRIDE"
            if prev_disp != out["disposition"] or prev_mask is not True:
                changed_events.append(
                    {
                        "symbol": out["symbol"],
                        "prior_bar_date": out["prior_bar_date"],
                        "event_bar_date": out["event_bar_date"],
                        "previous_disposition": prev_disp,
                        "new_disposition": out["disposition"],
                        "previous_masked_interval": prev_mask,
                        "new_masked_interval": True,
                    }
                )
            else:
                confirm_no_change.append(
                    {
                        "symbol": out["symbol"],
                        "prior_bar_date": out["prior_bar_date"],
                        "event_bar_date": out["event_bar_date"],
                        "disposition": out["disposition"],
                    }
                )
        updated.append(out)

    return updated, {
        "capital_decrease_event_count": len(cap_keys),
        "changed_events": sorted(changed_events, key=lambda x: (x["symbol"], x["event_bar_date"])),
        "unchanged_events": sorted(confirm_no_change, key=lambda x: (x["symbol"], x["event_bar_date"])),
    }


def build_seal_v4_4(review_dir: Path, disposition_counts: dict[str, int], masked_count: int) -> dict[str, Any]:
    refs = [
        ("triage_v4_2_final", "r12_breach_triage_v4_2_FINAL.json"),
        ("mask_manifest_v4_3_final", "r12_masked_intervals_manifest_v4_3_final.json"),
        ("ca_ledger_v0_2", "r12_ca_ledger_v0_2.json"),
        ("calendar_owner_adjudication_v4", "r12_calendar_owner_adjudication_v4.json"),
        ("reconciliation_v4_3", "r12_gap_reconciliation_v4_3.json"),
        ("reconciliation_v4_3_addendum_8", "r12_gap_reconciliation_v4_3_addendum_8.json"),
        ("reconciliation_v4_3_addendum_9", "r12_gap_reconciliation_v4_3_addendum_9.json"),
    ]
    ref_rows = []
    for label, rel in refs:
        p = review_dir / rel
        ref_rows.append(
            {
                "label": label,
                "path": f"artifacts/preview1a_prestart/review_final/{rel}",
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }
        )

    return {
        "version_id": "R12_PRE_EXAM_SURFACE_SEAL_V4_4",
        "scope": "Closure statement with R-9 preflight correction applied; no new classification logic beyond owner ruling.",
        "ratified_artifact_references": ref_rows,
        "final_disposition_counts": disposition_counts,
        "masked_interval_count": masked_count,
        "r12_authorization_status": "AUTHORIZED_FOR_R12_EXECUTION",
    }


def md_for_seal(payload: dict[str, Any]) -> str:
    lines = [
        "# R12 Pre-Exam Surface Seal V4.4",
        "",
        f"- version_id: {payload['version_id']}",
        f"- scope: {payload['scope']}",
        f"- r12_authorization_status: {payload['r12_authorization_status']}",
        "",
        "## Ratified Artifacts",
        "",
        "| Label | Path | SHA256 | Size (bytes) |",
        "|---|---|---|---:|",
    ]
    for r in payload["ratified_artifact_references"]:
        lines.append(f"| {r['label']} | {r['path']} | {r['sha256']} | {r['size_bytes']} |")

    lines.extend(
        [
            "",
            "## Final Counts",
            f"- final_disposition_counts: {json.dumps(payload['final_disposition_counts'], sort_keys=True)}",
            f"- masked_interval_count: {payload['masked_interval_count']}",
            "",
        ]
    )
    return "\n".join(lines)


def build_reconciliation_addendum_9() -> dict[str, Any]:
    return {
        "version_id": "R12_GAP_RECONCILIATION_V4_3_ADDENDUM_9",
        "scope": "Owner supersession append; original adjudication preserved.",
        "append_only": True,
        "supersession": {
            "topic": "THURAYA PREVIEW-1A adjudication referent",
            "statement": "The PREVIEW-1A explanation '164 missing sessions' does not reproduce in either DB under OWNER_VERIFIED calendar BK_CAL_V4_1783783330 and is superseded.",
            "preserved_original_text": True,
            "owner_confirmed_status": {
                "event_date": "2025-07-20",
                "class": "TRUE_CONSECUTIVE",
                "disposition": "ACCEPTED_REAL",
                "basis": "R-5/R-6 owner-confirmed class",
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="R12 preflight fix + v4.4 seal")
    parser.add_argument("--review-dir", required=True)
    args = parser.parse_args()

    review_dir = Path(args.review_dir).resolve()

    triage_path = review_dir / "r12_breach_triage_v4_2_FINAL.json"
    ca_v0_2_path = review_dir / "r12_ca_ledger_v0_2.json"

    triage = load_json(triage_path)
    ca_v0_2 = load_json(ca_v0_2_path)

    rows = [dict(r) for r in triage["final"]["rows"]]
    updated_rows, r9_summary = apply_r9(rows, ca_v0_2)

    disposition_counts = dict(sorted(Counter(str(r.get("disposition") or "").strip() for r in updated_rows).items()))
    masked_manifest = build_mask_manifest(updated_rows)

    accepted = int(disposition_counts.get("ACCEPTED_REAL", 0))
    deferred = int(disposition_counts.get("DEFERRED_TO_CA_LEDGER", 0))
    total = accepted + deferred
    masked_count = int(masked_manifest["interval_count"])

    deviation = {
        "accepted_delta": accepted - EXPECTED_ACCEPTED,
        "deferred_delta": deferred - EXPECTED_DEFERRED,
        "total_delta": total - EXPECTED_TOTAL,
        "masked_delta": masked_count - EXPECTED_MASKED,
    }

    add9 = build_reconciliation_addendum_9()
    add9_json = review_dir / "r12_gap_reconciliation_v4_3_addendum_9.json"
    add9_md = review_dir / "r12_gap_reconciliation_v4_3_addendum_9.md"
    dump_json(add9_json, add9)
    add9_md.write_text(
        "\n".join(
            [
                "# R12 Gap Reconciliation V4.3 Addendum 9",
                "",
                f"- version_id: {add9['version_id']}",
                f"- scope: {add9['scope']}",
                "",
                "## Supersession Append",
                f"- statement: {add9['supersession']['statement']}",
                "- preserved_original_text: True",
                "- owner_confirmed_class: TRUE_CONSECUTIVE",
                "- owner_confirmed_disposition: ACCEPTED_REAL",
                "- owner_confirmed_event_date: 2025-07-20",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    mask_out = review_dir / "r12_masked_intervals_manifest_v4_3_final.json"
    dump_json(mask_out, masked_manifest)

    seal = build_seal_v4_4(review_dir, {"ACCEPTED_REAL": accepted, "DEFERRED_TO_CA_LEDGER": deferred}, masked_count)
    seal_json = review_dir / "r12_pre_exam_surface_seal_v4_4.json"
    seal_md = review_dir / "r12_pre_exam_surface_seal_v4_4.md"
    dump_json(seal_json, seal)
    seal_md.write_text(md_for_seal(seal), encoding="utf-8", newline="\n")

    print("R9_CHANGED_EVENTS", json.dumps(r9_summary["changed_events"], ensure_ascii=True, sort_keys=True))
    print("R9_UNCHANGED_EVENTS", json.dumps(r9_summary["unchanged_events"], ensure_ascii=True, sort_keys=True))
    print("PRECHECK_COUNTS", json.dumps({"accepted": accepted, "deferred": deferred, "total": total, "masked": masked_count}, sort_keys=True))
    print("EXPECTED_COUNTS", json.dumps({"accepted": EXPECTED_ACCEPTED, "deferred": EXPECTED_DEFERRED, "total": EXPECTED_TOTAL, "masked": EXPECTED_MASKED}, sort_keys=True))
    print("DEVIATION", json.dumps(deviation, sort_keys=True))

    if any(v != 0 for v in deviation.values()):
        print("PRECHECK_STATUS DEVIATION_DETECTED")
        return 2

    print("PRECHECK_STATUS OK")
    print("SEAL_V4_4_SHA256", sha256_file(seal_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
