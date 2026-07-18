from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

OWNER_VERIFIED_CALENDAR_ID = "BK_CAL_V4_1783783330"
MASK_MANIFEST_VERSION_ID = "R12_MASKED_INTERVALS_MANIFEST_V4_3_FINAL"


@dataclass(frozen=True)
class AdapterAuthorities:
    calendar_version_id: str
    mask_manifest_version_id: str


@dataclass(frozen=True)
class SegmentState:
    segment_id: str
    segment_day_index: int
    segment_restart_flag: bool


class DataSurfaceAdapter:
    """Normalize source bars into spec-bound payloads with explicit seam masking."""

    def __init__(self, calendar_context: dict[str, Any], mask_manifest: dict[str, Any]) -> None:
        cal_version = str(calendar_context.get("version_id") or "")
        mask_version = str(mask_manifest.get("version_id") or "")
        if cal_version != OWNER_VERIFIED_CALENDAR_ID:
            raise ValueError(f"Calendar authority mismatch: expected {OWNER_VERIFIED_CALENDAR_ID}, got {cal_version}")
        if mask_version != MASK_MANIFEST_VERSION_ID:
            raise ValueError(f"Mask authority mismatch: expected {MASK_MANIFEST_VERSION_ID}, got {mask_version}")

        self.authorities = AdapterAuthorities(
            calendar_version_id=cal_version,
            mask_manifest_version_id=mask_version,
        )
        self._mask_index = self._build_mask_index(mask_manifest)

    @staticmethod
    def _build_mask_index(mask_manifest: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        out: dict[str, list[dict[str, str]]] = {}
        for row in mask_manifest.get("intervals", []):
            sym = str(row.get("symbol") or "").upper()
            if not sym:
                continue
            out.setdefault(sym, []).append(
                {
                    "start_date": str(row.get("start_date")),
                    "end_date": str(row.get("end_date")),
                    "source_rule": str(row.get("source_rule")),
                    "source_final_class": str(row.get("source_final_class")),
                }
            )
        return out

    @staticmethod
    def _to_date_text(value: Any) -> str:
        if isinstance(value, int):
            return datetime.utcfromtimestamp(value).strftime("%Y-%m-%d")
        text = str(value)
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            return text[:10]
        if text.isdigit() and len(text) >= 10:
            return datetime.utcfromtimestamp(int(text)).strftime("%Y-%m-%d")
        raise ValueError(f"Unsupported trade_date value: {value}")

    def mask_context_for(self, symbol: str, trade_date: str) -> dict[str, Any]:
        sym = symbol.upper()
        intervals = []
        for interval in self._mask_index.get(sym, []):
            if interval["start_date"] <= trade_date <= interval["end_date"]:
                intervals.append(interval)
        return {
            "masked_flag": len(intervals) > 0,
            "matched_intervals": intervals,
            "mask_authority": self.authorities.mask_manifest_version_id,
            "drop_policy": "FLAG_ONLY_NEVER_DROP",
        }

    def next_segment_state(
        self,
        *,
        symbol: str,
        trade_date: str,
        prev_segment: SegmentState | None,
        prev_masked: bool,
        current_masked: bool,
    ) -> SegmentState:
        if prev_segment is None:
            return SegmentState(segment_id=f"{symbol.upper()}::SEG0001", segment_day_index=0, segment_restart_flag=True)

        seam_break = prev_masked or current_masked
        if seam_break:
            seq = int(prev_segment.segment_id.split("SEG")[-1]) + 1
            return SegmentState(
                segment_id=f"{symbol.upper()}::SEG{seq:04d}",
                segment_day_index=0,
                segment_restart_flag=True,
            )

        return SegmentState(
            segment_id=prev_segment.segment_id,
            segment_day_index=prev_segment.segment_day_index + 1,
            segment_restart_flag=False,
        )

    def normalize_day(
        self,
        *,
        ohlcv_day: dict[str, Any],
        indicator_day: dict[str, Any],
        segment_context: SegmentState,
        calendar_context: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        symbol = str(ohlcv_day["symbol"]).upper()
        trade_date = self._to_date_text(ohlcv_day["trade_date"])
        masked_context = self.mask_context_for(symbol, trade_date)

        normalized_day_payload = {
            "trade_date": trade_date,
            "symbol": symbol,
            "open": float(ohlcv_day.get("open") or 0.0),
            "high": float(ohlcv_day.get("high") or 0.0),
            "low": float(ohlcv_day.get("low") or 0.0),
            "close": float(ohlcv_day.get("close") or 0.0),
            "volume": float(ohlcv_day.get("volume") or 0.0),
            "value_kwd": float(ohlcv_day.get("value_kwd") or 0.0),
            "indicator_terms": indicator_day,
            "segment_id": segment_context.segment_id,
            "segment_day_index": segment_context.segment_day_index,
            "masked_context": masked_context,
        }

        readiness_context = {
            "calendar_authority": self.authorities.calendar_version_id,
            "calendar_scope": {
                "date_range": calendar_context.get("date_range"),
                "holiday_count": calendar_context.get("holiday_count"),
            },
            "mask_authority": self.authorities.mask_manifest_version_id,
            "segment_restart_flag": segment_context.segment_restart_flag,
            "masked_flag": masked_context["masked_flag"],
            "no_cross_seam_rule": "ENFORCED",
        }
        return normalized_day_payload, readiness_context


def load_default_calendar_context(root: Path) -> dict[str, Any]:
    path = root / "artifacts" / "preview1a_prestart" / "review_final" / "r12_calendar_derivation_v4.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_default_mask_manifest(root: Path) -> dict[str, Any]:
    path = root / "artifacts" / "preview1a_prestart" / "review_final" / "r12_masked_intervals_manifest_v4_3_final.json"
    return json.loads(path.read_text(encoding="utf-8"))
