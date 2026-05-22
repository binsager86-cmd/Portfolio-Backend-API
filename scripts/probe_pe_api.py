#!/usr/bin/env python3
"""
Probe TickerChart's financial-field endpoint for P/E indicator time series.
Run from repo root:
  python mobile-migration/backend-api/scripts/probe_pe_api.py
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import httpx

from app.services.tickerchart_service import (
    _sign, _get_token, _common_params, _USER_AGENT,
    resolve_company_id, _LOGIN_HOST
)
from app.core.config import get_settings

# FactSet PE-related field codes to probe
PE_FIELD_IDS = [
    "ff_pe_basic_ltm",
    "ff_pe_basic",
    "ff_pe_diluted_ltm",
    "ff_pe_diluted",
    "ff_pe_normalized_ltm",
    "ff_price_to_earnings",
    "ff_pe",
    "ff_eps_basic_ltm",   # known to work - single LTM value
    "ff_eps_basic",       # known to work
]

REPORT_RANGES = [
    "ltm",        # Last Twelve Months
    "annual",
    "quarterly",
    "5years",
    "10years",
    "ann",
    "q",
    "ttm",
    "historical",
]


async def probe(company_id: int, field_id: str, report_range: str) -> dict:
    token = await _get_token()
    path = f"/m/v2/tickerchart/financial-field/company/"
    qs_pairs = [
        ("companyID", str(company_id)),
        ("financialIndicatorId", field_id),
        ("reportRange", report_range),
    ] + _common_params()
    final_qs, _ = _sign(path, qs_pairs)
    url = f"https://{_LOGIN_HOST}{path}?{final_qs}"
    
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Authorization": f"TcToken{token}",
            }
        )
    
    return {
        "field": field_id,
        "range": report_range,
        "status": resp.status_code,
        "body": resp.text[:300] if resp.status_code != 200 else resp.text[:1000]
    }


async def main():
    settings = get_settings()
    
    # Get NBK company ID
    company_id = await resolve_company_id("NBK", "KSE")
    if not company_id:
        print("Could not resolve NBK company ID")
        return
    print(f"NBK company_id = {company_id}")
    
    # First probe: known working field with different ranges
    print("\n=== Probing ff_eps_basic with different reportRanges ===")
    for rr in REPORT_RANGES:
        result = await probe(company_id, "ff_eps_basic", rr)
        status = result["status"]
        body = result["body"]
        print(f"  reportRange={rr:<15} status={status}  body={body[:150]!r}")
    
    # Second probe: different PE field IDs with ltm
    print("\n=== Probing different PE field IDs with reportRange=ltm ===")
    for field_id in PE_FIELD_IDS:
        result = await probe(company_id, field_id, "ltm")
        status = result["status"]
        body = result["body"]
        print(f"  fieldId={field_id:<35} status={status}  body={body[:150]!r}")


if __name__ == "__main__":
    asyncio.run(main())
