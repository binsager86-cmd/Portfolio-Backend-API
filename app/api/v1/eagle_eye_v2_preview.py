from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_current_user
from app.core.security import TokenData


router = APIRouter(prefix="/eagle-eye/v2-preview", tags=["Eagle Eye v2 Preview"])

VALIDATION_STATUS = "UNVALIDATED_PRE_R15"
ROOT = Path(__file__).resolve().parents[3]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
R14E_EVIDENCE = REVIEW / "r14e_module_e_test_evidence_v7.json"
R14F_EVIDENCE = REVIEW / "r14f_module_f_avoid_authority_v1_evidence.json"
R14G_EVIDENCE = REVIEW / "r14g_module_g_forward_prediction_v1_evidence.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"sealed artifact not found: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"sealed artifact JSON parse failed: {path.name}") from exc


def _envelope(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "validation_status": VALIDATION_STATUS,
        "authority": "DISPLAY_ONLY_NO_LIVE_COMPUTE_NO_WRITE_PATHS",
        **payload,
    }


def _symbol_key(symbol: str) -> str:
    key = symbol.upper().strip()
    if key not in {"MABANEE", "SANAM", "TIJARA"}:
        raise HTTPException(status_code=404, detail="symbol not present in sealed R14 evidence")
    return key


@router.get("/contract")
def get_v2_preview_contract(_current_user: TokenData = Depends(get_current_user)) -> dict[str, Any]:
    return _envelope(
        {
            "endpoints": {
                "GET /api/v1/eagle-eye/v2-preview/contract": "API contract and guardrails",
                "GET /api/v1/eagle-eye/v2-preview/summary": "closure and acceptance summary",
                "GET /api/v1/eagle-eye/v2-preview/module-e/tables/{symbol}": "sealed v7 per-day lifecycle rows",
                "GET /api/v1/eagle-eye/v2-preview/module-e/positions/{symbol}": "sealed v7 position progression rows",
                "GET /api/v1/eagle-eye/v2-preview/module-f/avoid/{symbol}": "sealed r14f avoid authority rows",
                "GET /api/v1/eagle-eye/v2-preview/module-g/predictions": "sealed r14g prediction rows",
                "GET /api/v1/eagle-eye/v2-preview/module-g/grades": "sealed r14g grade rows",
            },
            "guardrails": [
                "display-only reads of sealed R14 artifacts",
                "no live Eagle Eye v2 computation from app requests",
                "no write paths",
                "no authority over ratings or signals",
                "frozen R11 engine and existing Eagle Eye endpoints are untouched",
                "every response carries validation_status=UNVALIDATED_PRE_R15",
            ],
            "symbols": ["MABANEE", "SANAM", "TIJARA"],
        }
    )


@router.get("/summary")
def get_v2_preview_summary(_current_user: TokenData = Depends(get_current_user)) -> dict[str, Any]:
    e = _load_json(R14E_EVIDENCE)
    f = _load_json(R14F_EVIDENCE)
    g = _load_json(R14G_EVIDENCE)
    return _envelope(
        {
            "modules": {
                "e": {
                    "status": "CLOSED_PASS",
                    "scope_note": "entry, holding, suppression, avoid-veto lifecycle evidenced; exit lifecycle out of scope and untested",
                    "acceptance_checks": e.get("acceptance_checks", {}),
                },
                "f": {
                    "status": "CLOSED_PASS",
                    "byte_equivalence": "649/649",
                    "r12_interval_overlap": "97.5%",
                    "boundary_semantics_note": "REGISTERED",
                    "acceptance_checks": f.get("acceptance_checks", {}),
                },
                "g": {
                    "status": g.get("overall_status"),
                    "prediction_count": g.get("prediction_count"),
                    "grade_count": g.get("grade_count"),
                    "acceptance_checks": g.get("acceptance_checks", {}),
                },
            },
            "findings_carried_to_r15": ["FLOW_CORE_LAG", "AVOID_ARM_LAG"],
        }
    )


@router.get("/module-e/tables/{symbol}")
def get_module_e_table(symbol: str, _current_user: TokenData = Depends(get_current_user)) -> dict[str, Any]:
    key = _symbol_key(symbol)
    evidence = _load_json(R14E_EVIDENCE)
    rows = evidence.get("per_day_intent_lifecycle_tables", {}).get(key, [])
    return _envelope({"artifact": R14E_EVIDENCE.name, "symbol": key, "rows": rows})


@router.get("/module-e/positions/{symbol}")
def get_module_e_positions(symbol: str, _current_user: TokenData = Depends(get_current_user)) -> dict[str, Any]:
    key = _symbol_key(symbol)
    evidence = _load_json(R14E_EVIDENCE)
    rows = evidence.get("position_progressions", {}).get(key, [])
    return _envelope({"artifact": R14E_EVIDENCE.name, "symbol": key, "rows": rows})


@router.get("/module-f/avoid/{symbol}")
def get_module_f_avoid(symbol: str, _current_user: TokenData = Depends(get_current_user)) -> dict[str, Any]:
    key = _symbol_key(symbol)
    evidence = _load_json(R14F_EVIDENCE)
    rows = evidence.get("per_day_avoid_tables", {}).get(key, [])
    return _envelope({"artifact": R14F_EVIDENCE.name, "symbol": key, "rows": rows})


@router.get("/module-g/predictions")
def get_module_g_predictions(
    symbol: str | None = Query(default=None),
    _current_user: TokenData = Depends(get_current_user),
) -> dict[str, Any]:
    evidence = _load_json(R14G_EVIDENCE)
    rows = evidence.get("predictions", [])
    if symbol:
        key = _symbol_key(symbol)
        rows = [row for row in rows if str(row.get("symbol") or "").upper() == key]
    return _envelope({"artifact": R14G_EVIDENCE.name, "symbol": symbol.upper().strip() if symbol else "ALL", "rows": rows})


@router.get("/module-g/grades")
def get_module_g_grades(
    symbol: str | None = Query(default=None),
    _current_user: TokenData = Depends(get_current_user),
) -> dict[str, Any]:
    evidence = _load_json(R14G_EVIDENCE)
    rows = evidence.get("grades", [])
    if symbol:
        key = _symbol_key(symbol)
        rows = [row for row in rows if str(row.get("symbol") or "").upper() == key]
    return _envelope({"artifact": R14G_EVIDENCE.name, "symbol": symbol.upper().strip() if symbol else "ALL", "rows": rows})