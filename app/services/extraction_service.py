"""
Self-Reflective Financial Extraction Pipeline
================================================

A multi-pass AI extraction system that:

1. **Image Prep** — converts PDF pages to 300 DPI images.
2. **Reasoning Pass** — AI performs scratchpad arithmetic before outputting JSON.
3. **Extraction & Audit** — AI extracts numbers, validates sums internally,
   and flags discrepancies with root-cause analysis.
4. **Verification Step** — code-level cross-checks totals vs. sum-of-parts;
   retries targeted columns if mismatches are found.
5. **Caching** — stores a content-hash of the PDF so re-uploads skip AI calls.

All public functions are stateless except for caching (DB-backed).
"""

import hashlib
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# TYPES
# ════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedLineItem:
    key: str
    label_raw: str
    values: Dict[str, Optional[float]]
    is_total: bool = False
    order_index: int = 0


@dataclass
class ExtractedStatement:
    statement_type: str  # income, balance, cashflow, equity
    source_pages: List[int] = field(default_factory=list)
    currency: str = "USD"
    unit_scale: int = 1
    periods: List[Dict[str, str]] = field(default_factory=list)
    items: List[ExtractedLineItem] = field(default_factory=list)


@dataclass
class AuditCheck:
    """Result of one total-vs-parts validation."""
    statement_type: str
    period: str
    total_label: str
    total_value: float
    computed_sum: float
    discrepancy: float
    passed: bool
    detail: str = ""


@dataclass
class ExtractionResult:
    statements: List[ExtractedStatement]
    audit_checks: List[AuditCheck]
    confidence: float          # 0-1 overall confidence
    retry_count: int = 0
    cached: bool = False
    model_used: str = ""
    pages_processed: int = 0
    pdf_hash: str = ""
    validation_corrections: int = 0  # Number of corrections from validation pass
    placement_corrections: int = 0   # Number of corrections from placement verification


# ════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════

_TOLERANCE_PCT = 0.02  # 2% tolerance for rounding in audit checks

# Map AI statement types → canonical DB types
_TYPE_MAP = {
    "balance_sheet": "balance",
    "income_statement": "income",
    "cash_flow": "cashflow",
    "equity_statement": "equity",
    "income": "income",
    "balance": "balance",
    "cashflow": "cashflow",
    "equity": "equity",
}

# Total → component-parts mapping for cross-checks
# Key = total line_item_key pattern, Value = list of component patterns
_AUDIT_RULES: Dict[str, Dict[str, List[str]]] = {
    "balance": {
        "total_assets": ["total_current_assets", "total_non_current_assets"],
        "total_liabilities_and_equity": ["total_liabilities", "total_equity"],
        "total_liabilities": ["total_current_liabilities", "total_non_current_liabilities"],
    },
    "income": {
        "gross_profit": ["revenue", "cost_of_revenue"],
        "operating_income": ["gross_profit", "operating_expenses"],
    },
    "cashflow": {
        "net_change_in_cash": [
            "cash_from_operations",
            "cash_from_investing",
            "cash_from_financing",
        ],
    },
}

MAX_RETRIES = 2
VALIDATION_ENABLED = True  # Post-extraction validation pass


# ════════════════════════════════════════════════════════════════════
# CACHE TABLE
# ════════════════════════════════════════════════════════════════════

_CACHE_SCHEMA_INIT = False


def _ensure_cache_table() -> None:
    """Create extraction_cache table if it doesn't exist."""
    global _CACHE_SCHEMA_INIT
    if _CACHE_SCHEMA_INIT:
        return

    from app.core.config import get_settings
    s = get_settings()
    pk = "SERIAL PRIMARY KEY" if s.use_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"

    from app.core.database import exec_sql
    exec_sql(f"""
        CREATE TABLE IF NOT EXISTS extraction_cache (
            id {pk},
            stock_id     INTEGER NOT NULL,
            pdf_hash     TEXT NOT NULL,
            filename     TEXT,
            result_json  TEXT NOT NULL,
            model_used   TEXT,
            pages        INTEGER,
            created_at   INTEGER NOT NULL,
            UNIQUE(stock_id, pdf_hash)
        )
    """)
    _CACHE_SCHEMA_INIT = True


def _pdf_hash(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def _get_cached(stock_id: int, h: str) -> Optional[ExtractionResult]:
    """Return a cached ExtractionResult if we've seen this exact PDF before."""
    _ensure_cache_table()
    from app.core.database import query_one
    row = query_one(
        "SELECT result_json, model_used, pages FROM extraction_cache "
        "WHERE stock_id = ? AND pdf_hash = ?",
        (stock_id, h),
    )
    if not row:
        return None

    try:
        data = json.loads(row["result_json"])
        result = _dict_to_result(data)
        result.cached = True
        result.pdf_hash = h
        result.model_used = row["model_used"] or ""
        result.pages_processed = row["pages"] or 0
        logger.info("Cache hit for stock %d (hash %s…)", stock_id, h[:12])
        return result
    except Exception:
        logger.warning("Corrupt cache entry for stock %d, ignoring", stock_id)
        return None


def _set_cache(
    stock_id: int, h: str, filename: str, result: ExtractionResult,
) -> None:
    _ensure_cache_table()
    from app.core.database import exec_sql
    result_json = json.dumps(_result_to_dict(result))
    now = int(time.time())
    # Upsert
    try:
        exec_sql(
            "INSERT INTO extraction_cache "
            "(stock_id, pdf_hash, filename, result_json, model_used, pages, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (stock_id, h, filename, result_json,
             result.model_used, result.pages_processed, now),
        )
    except Exception:
        # Duplicate — update
        exec_sql(
            "UPDATE extraction_cache SET result_json=?, model_used=?, "
            "pages=?, created_at=?, filename=? "
            "WHERE stock_id=? AND pdf_hash=?",
            (result_json, result.model_used, result.pages_processed,
             now, filename, stock_id, h),
        )


# ════════════════════════════════════════════════════════════════════
# PDF → IMAGES
# ════════════════════════════════════════════════════════════════════

def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[bytes]:
    """Convert each PDF page to a PNG image at *dpi* resolution."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: List[bytes] = []
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


# ════════════════════════════════════════════════════════════════════
# PROMPTS
# ════════════════════════════════════════════════════════════════════

def _build_extraction_prompt(n_pages: int) -> str:
    """
    The main self-reflective prompt.  Three sections:
    1. Classification + Extraction
    2. Internal Scratchpad arithmetic audit
    3. Strict JSON output with embedded audit results
    """
    return f"""\
You are a CERTIFIED FINANCIAL ANALYST and DATA EXTRACTION ENGINE.
I am giving you {n_pages} page(s) from a financial report.

═══ PHASE 1 — CLASSIFICATION & EXTRACTION ═══
Identify each financial statement type on these pages:
  • balance_sheet   (Statement of Financial Position / الميزانية العمومية)
  • income_statement (Profit or Loss / قائمة الدخل)
  • cash_flow       (Cash Flows / التدفقات النقدية)
  • equity_statement (Changes in Equity / التغيرات في حقوق الملكية)

Merge pages that belong to the SAME statement type.
Extract EVERY line item with ALL period columns.

═══ PHASE 2 — INTERNAL SCRATCHPAD AUDIT ═══
Before producing output, perform these checks IN YOUR REASONING:

For BALANCE SHEET:
  • total_current_assets + total_non_current_assets MUST = total_assets
  • total_current_liabilities + total_non_current_liabilities MUST = total_liabilities
  • total_liabilities + total_equity MUST = total_liabilities_and_equity (or total_assets)

For INCOME STATEMENT:
  • revenue − cost_of_revenue MUST = gross_profit
  • Check operating_income is consistent with gross_profit − operating_expenses

For CASH FLOW:
  • cash_from_operations + cash_from_investing + cash_from_financing MUST ≈ net_change_in_cash

If any check FAILS:
  1. Re-examine the relevant rows in the image
  2. Look for missing subtotals, OCR errors, or sign convention issues
  3. Correct the values before producing output
  4. Note the correction in the "audit_notes" field

═══ PHASE 3 — OUTPUT FORMAT ═══
Return ONLY this JSON (no markdown fences, no commentary):

[
  {{
    "statement_type": "balance_sheet",
    "source_pages": [1],
    "currency": "KWD",
    "unit_scale": 1,
    "periods": [
      {{"label": "2025-12-31", "col_name": "2025"}},
      {{"label": "2024-12-31", "col_name": "2024"}}
    ],
    "items": [
      {{
        "label_raw": "Cash and bank balances",
        "key": "cash_and_bank_balances",
        "values": {{"2025-12-31": 67007011, "2024-12-31": 74286447}},
        "is_total": false
      }}
    ],
    "audit": {{
      "checks_performed": [
        {{
          "rule": "total_assets = current + non_current",
          "expected": 500000,
          "actual": 500000,
          "passed": true
        }}
      ],
      "corrections_made": [],
      "audit_notes": ""
    }}
  }}
]

═══ RULES ═══
• "values" must contain numbers or null — never strings.
• Parentheses (1,234) → −1234.  Dash or blank → null.
• Detect unit_scale: "KD'000" → unit_scale=1000.  "millions" → 1000000.
• Period labels should be ISO dates if year is visible.
• is_total=true for subtotals and totals.
• source_pages is 1-indexed.
• YOU MUST EXTRACT EVERY SINGLE LINE ITEM — do NOT skip any row.
• Copy every number EXACTLY as printed — do not round or approximate.
• Include ALL subtotals, totals, and grand totals with is_total=true.
• Zero values → include as 0, NOT omit.
"""


def _build_retry_prompt(
    n_pages: int,
    failed_checks: List[AuditCheck],
) -> str:
    """
    Targeted retry prompt that tells the AI exactly which checks failed
    so it can focus on those specific areas.
    """
    failures = "\n".join(
        f"  • [{c.statement_type}] {c.period}: {c.total_label} "
        f"expected sum={c.computed_sum:.2f} but total={c.total_value:.2f} "
        f"(discrepancy={c.discrepancy:.2f})"
        for c in failed_checks
    )

    return f"""\
You are a CERTIFIED FINANCIAL ANALYST and DATA EXTRACTION ENGINE.
I am giving you {n_pages} page(s) from a financial report.

⚠ PREVIOUS EXTRACTION HAD ERRORS — Please fix these discrepancies:
{failures}

INSTRUCTIONS:
1. Re-examine the original images carefully for the failing statements.
2. Check for: missing rows, OCR mis-reads, sign errors, skipped subtotals.
3. Pay special attention to the columns/periods that failed.
4. Re-extract the COMPLETE statement(s) with corrected values.
5. Verify your corrections pass the arithmetic checks before outputting.

Return the SAME JSON format as before — a JSON array of statement objects.
Each statement MUST include an "audit" block with your verification.

OUTPUT RULES (same as before):
• Values must be numbers or null — never strings.
• Parentheses (1,234) → −1234.  Dash or blank → null.
• Detect unit_scale from headers.
• is_total=true for all totals and subtotals.
• EXTRACT EVERY LINE ITEM — do NOT skip rows.
• Return ONLY the JSON array, no markdown fences.
"""


def _build_validation_prompt(
    n_pages: int,
    extracted_data: List[ExtractedStatement],
) -> str:
    """
    Validation Pass: send extracted data back to AI alongside the PDF images
    to cross-check completeness and accuracy. AI compares every row in the
    original document against what was extracted and reports corrections.
    """
    # Serialize the extracted data into a readable summary for the AI
    data_summary_parts = []
    for stmt in extracted_data:
        period_labels = [p.get("label", p.get("col_name", "?")) for p in stmt.periods]
        lines = []
        for it in stmt.items:
            vals = ", ".join(
                f"{p}: {it.values.get(p, 'MISSING')}" for p in period_labels
            )
            total_tag = " [TOTAL]" if it.is_total else ""
            lines.append(f"    {it.label_raw} ({it.key}){total_tag}: {vals}")
        data_summary_parts.append(
            f"  Statement: {stmt.statement_type}\n"
            f"  Currency: {stmt.currency}, Scale: {stmt.unit_scale}\n"
            f"  Periods: {period_labels}\n"
            f"  Items ({len(stmt.items)} rows):\n" + "\n".join(lines)
        )

    data_summary = "\n\n".join(data_summary_parts)

    return f"""\
You are a CERTIFIED FINANCIAL ANALYST performing a VALIDATION AUDIT.
I am giving you {n_pages} page(s) from a financial report AND the data
that was previously extracted from these same pages.

═══ YOUR TASK ═══
Compare the EXTRACTED DATA below against the ORIGINAL DOCUMENT in the images.
Check for:
  1. **MISSING LINE ITEMS** — rows in the document that are NOT in the extracted data.
  2. **INCORRECT VALUES** — numbers that don't match what's printed.
  3. **MISSING PERIODS** — year columns present in the document but absent from extraction.
  4. **WRONG SIGNS** — values that should be negative (parenthesized) but are positive, or vice versa.
  5. **MISSING STATEMENTS** — entire statement types visible in the PDF but not extracted at all.

═══ EXTRACTED DATA TO VALIDATE ═══
{data_summary}

═══ OUTPUT FORMAT ═══
Return ONLY this JSON (no markdown fences):

{{
  "validation_passed": true/false,
  "corrections": [
    {{
      "statement_type": "balance",
      "action": "add" | "update" | "remove",
      "period": "2020-12-31",
      "key": "cash_and_bank_balances",
      "label_raw": "Cash and bank balances",
      "old_value": null,
      "new_value": 67007011,
      "reason": "Missing from extraction — visible on page 2"
    }}
  ],
  "missing_statements": [
    {{
      "statement_type": "cashflow",
      "reason": "Cash flow statement on pages 4-5 was not extracted"
    }}
  ],
  "notes": "Optional overall notes about the validation"
}}

═══ RULES ═══
• If everything is correct, return {{"validation_passed": true, "corrections": [], "missing_statements": [], "notes": ""}}
• For "add" actions: the item was missing and needs to be added
• For "update" actions: the value was wrong and needs correction (include old_value)
• For "remove" actions: the item doesn't exist in the document (rare)
• Values must be numbers or null — not strings
• Parentheses (1,234) → −1234.  Dash or blank → null
• Check EVERY SINGLE ROW in the document — do not skip any
• Pay special attention to cash balances, total equity, retained earnings
• Check ALL year columns — older years like 2019, 2020, 2021 are often missed
"""


def _apply_corrections(
    statements: List[ExtractedStatement],
    corrections: list,
) -> Tuple[List[ExtractedStatement], int]:
    """
    Apply corrections from the validation pass to the extracted data.
    Returns (updated_statements, number_of_corrections_applied).
    """
    applied = 0

    for corr in corrections:
        stmt_type = _TYPE_MAP.get(corr.get("statement_type", ""), corr.get("statement_type", ""))
        action = corr.get("action", "")
        period = corr.get("period", "")
        key = corr.get("key", "")
        new_value = corr.get("new_value")
        label_raw = corr.get("label_raw", key)
        reason = corr.get("reason", "")

        if not stmt_type or not key or not period:
            continue

        # Find the matching statement
        target_stmt = None
        for s in statements:
            if s.statement_type == stmt_type:
                target_stmt = s
                break

        if target_stmt is None:
            continue

        if action == "add":
            # Check if item already exists
            existing = None
            for it in target_stmt.items:
                if it.key.lower().replace(" ", "_") == key.lower().replace(" ", "_"):
                    existing = it
                    break

            if existing:
                # Item exists but period value is missing — add it
                if period not in existing.values or existing.values[period] is None:
                    existing.values[period] = float(new_value) if new_value is not None else None
                    applied += 1
                    logger.info("Validation: added period %s to %s = %s (%s)",
                                period, key, new_value, reason)
            else:
                # Brand new item
                new_item = ExtractedLineItem(
                    key=key,
                    label_raw=label_raw,
                    values={period: float(new_value) if new_value is not None else None},
                    is_total=False,
                    order_index=len(target_stmt.items) + 1,
                )
                target_stmt.items.append(new_item)
                applied += 1
                logger.info("Validation: added new item %s for %s = %s (%s)",
                            key, period, new_value, reason)

        elif action == "update":
            for it in target_stmt.items:
                if it.key.lower().replace(" ", "_") == key.lower().replace(" ", "_"):
                    old = it.values.get(period)
                    it.values[period] = float(new_value) if new_value is not None else None
                    applied += 1
                    logger.info("Validation: updated %s [%s] from %s to %s (%s)",
                                key, period, old, new_value, reason)
                    break

        elif action == "remove":
            target_stmt.items = [
                it for it in target_stmt.items
                if it.key.lower().replace(" ", "_") != key.lower().replace(" ", "_")
            ]
            applied += 1
            logger.info("Validation: removed %s (%s)", key, reason)

    return statements, applied


# ════════════════════════════════════════════════════════════════════
# JSON PARSING
# ════════════════════════════════════════════════════════════════════

def _parse_ai_json(text: str) -> list:
    """Parse JSON from Gemini response with multi-stage repair."""
    cleaned = text.strip()
    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Fast path
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract JSON array
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fix trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Fix NaN / Infinity literals
    fixed2 = re.sub(r'\bNaN\b', 'null', fixed)
    fixed2 = re.sub(r'\bInfinity\b', 'null', fixed2)
    fixed2 = re.sub(r'\b-Infinity\b', 'null', fixed2)
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    raise ValueError("AI returned unparseable JSON. Try again with a clearer PDF.")


def _parse_validation_json(text: str) -> dict:
    """Parse validation response JSON (a dict, not a list)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Extract JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Fix trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        result = json.loads(fixed)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    raise ValueError("Could not parse validation response as JSON dict.")


# ════════════════════════════════════════════════════════════════════
# VERIFICATION ENGINE
# ════════════════════════════════════════════════════════════════════

def _verify_statement(
    stmt: ExtractedStatement,
) -> List[AuditCheck]:
    """
    Code-level cross-checks: for each total-vs-parts rule,
    sum the component items and compare to the declared total.
    """
    rules = _AUDIT_RULES.get(stmt.statement_type, {})
    if not rules:
        return []

    # Build a fast lookup: key → {period → value}
    item_map: Dict[str, Dict[str, float]] = {}
    for it in stmt.items:
        k = it.key.lower().replace(" ", "_").replace("-", "_")
        item_map[k] = {}
        for p, v in it.values.items():
            item_map[k][p] = v if v is not None else 0.0

    checks: List[AuditCheck] = []

    for total_key, part_keys in rules.items():
        total_vals = item_map.get(total_key, {})
        if not total_vals:
            continue

        for period in total_vals:
            total_val = total_vals[period]

            # Collect component values
            parts_sum = 0.0
            parts_found = 0
            for pk in part_keys:
                if pk in item_map and period in item_map[pk]:
                    parts_sum += item_map[pk][period]
                    parts_found += 1

            # Only check if we found at least some components
            if parts_found == 0:
                continue

            # For income: gross_profit = revenue - COGS (subtraction)
            if total_key == "gross_profit" and stmt.statement_type == "income":
                rev = item_map.get("revenue", {}).get(period, 0)
                cogs = item_map.get("cost_of_revenue", {}).get(period, 0)
                parts_sum = rev - abs(cogs) if cogs < 0 else rev - cogs

            disc = abs(total_val - parts_sum)
            threshold = max(abs(total_val) * _TOLERANCE_PCT, 1.0)
            passed = disc <= threshold

            checks.append(AuditCheck(
                statement_type=stmt.statement_type,
                period=period,
                total_label=total_key,
                total_value=total_val,
                computed_sum=parts_sum,
                discrepancy=disc,
                passed=passed,
                detail=(
                    f"OK: {total_key}={total_val:.0f}"
                    if passed
                    else f"FAIL: {total_key}={total_val:.0f} but "
                         f"sum({'+'.join(part_keys)})={parts_sum:.0f} "
                         f"(diff={disc:.0f})"
                ),
            ))

    # Also check AI's own audit block if present — trust-but-verify
    return checks


def _verify_all(statements: List[ExtractedStatement]) -> List[AuditCheck]:
    """Run verification across all extracted statements."""
    all_checks: List[AuditCheck] = []
    for stmt in statements:
        all_checks.extend(_verify_statement(stmt))
    return all_checks


def _calculate_confidence(
    checks: List[AuditCheck],
    ai_confidence: float = 0.85,
) -> float:
    """
    Compute overall confidence from audit checks.

    Base = 0.70 (AI extraction baseline).
    Add up to 0.30 based on audit pass rate.
    """
    if not checks:
        return ai_confidence

    passed = sum(1 for c in checks if c.passed)
    total = len(checks)
    pass_rate = passed / total if total > 0 else 1.0

    # Weighted: base 0.70 + up to 0.30 for perfect audit
    return round(min(0.70 + 0.30 * pass_rate, 1.0), 3)


# ════════════════════════════════════════════════════════════════════
# GEMINI API CALL
# ════════════════════════════════════════════════════════════════════

async def _call_gemini(
    api_key: str,
    prompt: str,
    page_images: List[bytes],
    model_name: str = "gemini-2.5-flash",
) -> str:
    """Send images + prompt to Gemini and return raw text response."""
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    parts: list = [prompt]
    for png in page_images:
        parts.append(Image.open(io.BytesIO(png)))

    response = model.generate_content(
        parts,
        generation_config={
            "max_output_tokens": 32768,
            "temperature": 0.05,  # very low for precision
        },
        request_options={"timeout": 240},  # 4 min per AI call
    )
    if not response.text:
        raise ValueError("AI returned an empty response.")
    return response.text


# ════════════════════════════════════════════════════════════════════
# RAW JSON → TYPED DATACLASSES
# ════════════════════════════════════════════════════════════════════

def _raw_to_statements(raw: list) -> List[ExtractedStatement]:
    """Convert parsed JSON dicts to typed ExtractedStatement objects."""
    stmts: List[ExtractedStatement] = []
    for entry in raw:
        st_type = _TYPE_MAP.get(
            entry.get("statement_type", ""), entry.get("statement_type", "unknown"),
        )
        if st_type not in ("income", "balance", "cashflow", "equity"):
            logger.warning("Skipping unknown type: %s", entry.get("statement_type"))
            continue

        items = []
        for idx, it in enumerate(entry.get("items", []), 1):
            items.append(ExtractedLineItem(
                key=it.get("key", it.get("label_raw", "UNKNOWN")),
                label_raw=it.get("label_raw", it.get("key", "")),
                values={
                    k: (float(v) if v is not None else None)
                    for k, v in it.get("values", {}).items()
                },
                is_total=bool(it.get("is_total", False)),
                order_index=idx,
            ))

        stmts.append(ExtractedStatement(
            statement_type=st_type,
            source_pages=entry.get("source_pages", []),
            currency=entry.get("currency", "USD"),
            unit_scale=entry.get("unit_scale", 1),
            periods=entry.get("periods", []),
            items=items,
        ))
    return stmts


# ════════════════════════════════════════════════════════════════════
# SERIALIZATION (for cache)
# ════════════════════════════════════════════════════════════════════

def _result_to_dict(r: ExtractionResult) -> dict:
    return {
        "statements": [
            {
                "statement_type": s.statement_type,
                "source_pages": s.source_pages,
                "currency": s.currency,
                "unit_scale": s.unit_scale,
                "periods": s.periods,
                "items": [
                    {
                        "key": it.key,
                        "label_raw": it.label_raw,
                        "values": it.values,
                        "is_total": it.is_total,
                        "order_index": it.order_index,
                    }
                    for it in s.items
                ],
            }
            for s in r.statements
        ],
        "audit_checks": [
            {
                "statement_type": c.statement_type,
                "period": c.period,
                "total_label": c.total_label,
                "total_value": c.total_value,
                "computed_sum": c.computed_sum,
                "discrepancy": c.discrepancy,
                "passed": c.passed,
                "detail": c.detail,
            }
            for c in r.audit_checks
        ],
        "confidence": r.confidence,
        "retry_count": r.retry_count,
        "validation_corrections": r.validation_corrections,
        "placement_corrections": r.placement_corrections,
    }


def _dict_to_result(d: dict) -> ExtractionResult:
    stmts = []
    for s in d.get("statements", []):
        items = [
            ExtractedLineItem(
                key=it["key"], label_raw=it["label_raw"],
                values=it["values"], is_total=it["is_total"],
                order_index=it.get("order_index", 0),
            )
            for it in s.get("items", [])
        ]
        stmts.append(ExtractedStatement(
            statement_type=s["statement_type"],
            source_pages=s.get("source_pages", []),
            currency=s.get("currency", "USD"),
            unit_scale=s.get("unit_scale", 1),
            periods=s.get("periods", []),
            items=items,
        ))
    checks = [
        AuditCheck(**c) for c in d.get("audit_checks", [])
    ]
    return ExtractionResult(
        statements=stmts,
        audit_checks=checks,
        confidence=d.get("confidence", 0.85),
        retry_count=d.get("retry_count", 0),
        validation_corrections=d.get("validation_corrections", 0),
        placement_corrections=d.get("placement_corrections", 0),
    )


# ════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════

async def extract_financials(
    pdf_bytes: bytes,
    stock_id: int,
    api_key: str,
    filename: str = "upload.pdf",
    model_name: str = "gemini-2.5-flash",
    use_cache: bool = True,
) -> ExtractionResult:
    """
    Full self-reflective extraction pipeline.

    1. Hash PDF → check cache
    2. Convert PDF → 300 DPI images
    3. Send to AI with self-reflective prompt
    4. Parse JSON → verify arithmetic
    5. If verification fails, retry with targeted prompt (up to MAX_RETRIES)
    6. Cache final result
    7. Return ExtractionResult
    """

    h = _pdf_hash(pdf_bytes)

    # ── Step 1: Cache check ──────────────────────────────────────────
    if use_cache:
        cached = _get_cached(stock_id, h)
        if cached:
            return cached

    # ── Step 2: PDF → images ─────────────────────────────────────────
    page_images = pdf_to_images(pdf_bytes, dpi=300)
    if not page_images:
        raise ValueError("PDF has no pages.")

    logger.info(
        "Extraction pipeline: %s (%d pages, %.1f KB)",
        filename, len(page_images), len(pdf_bytes) / 1024,
    )

    # ── Step 3: First extraction pass ────────────────────────────────
    prompt = _build_extraction_prompt(len(page_images))
    raw_text = await _call_gemini(api_key, prompt, page_images, model_name)
    raw_json = _parse_ai_json(raw_text)

    if not isinstance(raw_json, list) or len(raw_json) == 0:
        raise ValueError("AI did not detect any financial statements.")

    statements = _raw_to_statements(raw_json)

    # Apply unit_scale
    for stmt in statements:
        if stmt.unit_scale and stmt.unit_scale != 1:
            for item in stmt.items:
                item.values = {
                    k: (v * stmt.unit_scale if v is not None else None)
                    for k, v in item.values.items()
                }

    # ── Step 4: Verification ─────────────────────────────────────────
    checks = _verify_all(statements)
    failed = [c for c in checks if not c.passed]
    retry_count = 0

    # ── Step 5: Retry loop ───────────────────────────────────────────
    while failed and retry_count < MAX_RETRIES:
        retry_count += 1
        logger.warning(
            "Extraction attempt %d: %d audit failures — retrying",
            retry_count, len(failed),
        )

        retry_prompt = _build_retry_prompt(len(page_images), failed)
        retry_text = await _call_gemini(api_key, retry_prompt, page_images, model_name)
        retry_json = _parse_ai_json(retry_text)

        if isinstance(retry_json, list) and len(retry_json) > 0:
            retry_stmts = _raw_to_statements(retry_json)

            # Apply unit_scale to retry results
            for stmt in retry_stmts:
                if stmt.unit_scale and stmt.unit_scale != 1:
                    for item in stmt.items:
                        item.values = {
                            k: (v * stmt.unit_scale if v is not None else None)
                            for k, v in item.values.items()
                        }

            retry_checks = _verify_all(retry_stmts)
            retry_failed = [c for c in retry_checks if not c.passed]

            # Accept retry results if they're better
            if len(retry_failed) < len(failed):
                # Merge: replace only the statement types that improved
                improved_types = {c.statement_type for c in failed} - {
                    c.statement_type for c in retry_failed
                }
                for rt_stmt in retry_stmts:
                    if rt_stmt.statement_type in improved_types:
                        statements = [
                            s for s in statements
                            if s.statement_type != rt_stmt.statement_type
                        ] + [rt_stmt]

                # If ALL checks now pass after full retry, take the whole set
                if not retry_failed:
                    statements = retry_stmts

                checks = _verify_all(statements)
                failed = [c for c in checks if not c.passed]
            else:
                logger.info("Retry %d did not improve results, keeping original", retry_count)
                break

    # ── Step 6: Compute confidence ───────────────────────────────────
    confidence = _calculate_confidence(checks)

    result = ExtractionResult(
        statements=statements,
        audit_checks=checks,
        confidence=confidence,
        retry_count=retry_count,
        cached=False,
        model_used=model_name,
        pages_processed=len(page_images),
        pdf_hash=h,
        validation_corrections=0,
    )

    # ── Step 7: Cache ────────────────────────────────────────────────
    if use_cache:
        try:
            _set_cache(stock_id, h, filename, result)
        except Exception as exc:
            logger.warning("Failed to cache extraction result: %s", exc)

    logger.info(
        "Extraction complete: %d statements, %d checks (%d passed), "
        "confidence=%.1f%%, retries=%d",
        len(statements), len(checks),
        sum(1 for c in checks if c.passed),
        confidence * 100, retry_count,
    )

    return result


# ════════════════════════════════════════════════════════════════════
# VALIDATION PIPELINE (Step 2 — separate call)
# ════════════════════════════════════════════════════════════════════

async def validate_extraction(
    pdf_bytes: bytes,
    stock_id: int,
    api_key: str,
    filename: str = "upload.pdf",
    model_name: str = "gemini-2.5-flash",
) -> ExtractionResult:
    """
    Validation pass — runs AFTER extract_financials().

    1. Load cached extraction result for this PDF
    2. Send extracted data + PDF images back to AI for completeness check
    3. Apply corrections (missing items, wrong values, missing periods)
    4. Update cache with corrected result
    5. Return updated ExtractionResult
    """

    h = _pdf_hash(pdf_bytes)

    # Load the cached extraction result
    cached = _get_cached(stock_id, h)
    if not cached:
        raise ValueError(
            "No cached extraction found. Run upload-statement first."
        )

    statements = cached.statements

    # Convert PDF → images for AI
    page_images = pdf_to_images(pdf_bytes, dpi=300)
    if not page_images:
        raise ValueError("PDF has no pages.")

    logger.info(
        "Validation pipeline: %s (%d pages, %d statements to validate)",
        filename, len(page_images), len(statements),
    )

    # Send extracted data + PDF to AI for cross-check
    val_prompt = _build_validation_prompt(len(page_images), statements)
    val_text = await _call_gemini(api_key, val_prompt, page_images, model_name)
    val_json = _parse_validation_json(val_text)

    validation_corrections = 0

    if not val_json.get("validation_passed", True):
        corrections = val_json.get("corrections", [])
        if corrections:
            logger.info("Validation found %d correction(s) — applying", len(corrections))
            statements, validation_corrections = _apply_corrections(
                statements, corrections,
            )

            # Apply unit_scale to any newly added items
            for stmt in statements:
                if stmt.unit_scale and stmt.unit_scale != 1:
                    for item in stmt.items:
                        item.values = {
                            k: (v * stmt.unit_scale if v is not None else None)
                            for k, v in item.values.items()
                        }
        else:
            logger.info("Validation found issues but no actionable corrections")
    else:
        logger.info("Validation passed — no corrections needed")

    # Re-run arithmetic verification
    checks = _verify_all(statements)
    confidence = _calculate_confidence(checks)

    result = ExtractionResult(
        statements=statements,
        audit_checks=checks,
        confidence=confidence,
        retry_count=cached.retry_count,
        cached=False,
        model_used=model_name,
        pages_processed=len(page_images),
        pdf_hash=h,
        validation_corrections=validation_corrections,
    )

    # Update cache with validated result
    try:
        _set_cache(stock_id, h, filename, result)
    except Exception as exc:
        logger.warning("Failed to update cache after validation: %s", exc)

    logger.info(
        "Validation complete: %d corrections applied, %d checks (%d passed), "
        "confidence=%.1f%%",
        validation_corrections, len(checks),
        sum(1 for c in checks if c.passed),
        confidence * 100,
    )

    return result


# ════════════════════════════════════════════════════════════════════
# PLACEMENT VERIFICATION (Step 3 — separate call)
# ════════════════════════════════════════════════════════════════════

# Canonical rules for which keys belong under which statement type
_PLACEMENT_RULES: Dict[str, List[str]] = {
    "balance": [
        "cash", "bank", "receivable", "inventory", "prepaid", "asset",
        "property", "equipment", "goodwill", "intangible", "investment",
        "payable", "accrued", "liability", "debt", "loan", "borrowing",
        "equity", "capital", "retained", "reserve", "share",
    ],
    "income": [
        "revenue", "sales", "income", "cost_of", "cogs", "gross_profit",
        "operating", "expense", "depreciation", "amortization", "interest",
        "tax", "profit", "loss", "earning", "eps", "dividend",
        "administrative", "general", "selling", "marketing",
    ],
    "cashflow": [
        "cash_from", "cash_flow", "operating_activities", "investing",
        "financing", "net_change", "beginning_cash", "ending_cash",
        "purchase_of", "proceeds", "repayment", "issuance",
    ],
    "equity": [
        "share_capital", "treasury", "retained_earnings", "reserve",
        "comprehensive", "contributed", "accumulated", "minority",
        "non_controlling",
    ],
}


def _build_placement_prompt(
    extracted_data: List[ExtractedStatement],
) -> str:
    """
    Step 3 prompt: Check every line item is correctly placed under the
    right statement type, has the right key/code, and is_total is set properly.
    """
    data_summary_parts = []
    for stmt in extracted_data:
        period_labels = [p.get("label", p.get("col_name", "?")) for p in stmt.periods]
        lines = []
        for it in stmt.items:
            vals = ", ".join(
                f"{p}: {it.values.get(p, 'MISSING')}" for p in period_labels
            )
            total_tag = " [TOTAL]" if it.is_total else ""
            lines.append(f"    {it.label_raw} (key={it.key}){total_tag}: {vals}")
        data_summary_parts.append(
            f"  Statement: {stmt.statement_type}\n"
            f"  Periods: {period_labels}\n"
            f"  Items ({len(stmt.items)} rows):\n" + "\n".join(lines)
        )

    data_summary = "\n\n".join(data_summary_parts)

    return f"""\
You are a CERTIFIED FINANCIAL ANALYST performing a PLACEMENT VERIFICATION.

═══ YOUR TASK ═══
Review the extracted financial data below and verify that EVERY line item
is correctly placed. Check:

1. **CORRECT STATEMENT TYPE** — Is each item under the right statement?
   • Balance sheet items (assets, liabilities, equity) must be under "balance"
   • Income/P&L items (revenue, expenses, profit) must be under "income"
   • Cash flow items must be under "cashflow"
   • Equity changes must be under "equity"

2. **CORRECT KEY/CODE** — Does the key accurately represent the line item?
   • "cash_and_bank_balances" for cash, not "other_assets"
   • "revenue" for top-line sales, not "income"
   • "total_assets" should have is_total=true

3. **is_total FLAG** — Are totals and subtotals correctly flagged?
   • Items like "Total Assets", "Total Liabilities", "Net Income" → is_total=true
   • Individual line items → is_total=false

4. **DUPLICATE ITEMS** — Are any items duplicated across statements?
   • Each line item should appear in exactly ONE statement type

5. **MISPLACED ITEMS** — Common mistakes:
   • "Net income" appearing in balance sheet (should be in income statement)
   • "Cash and bank balances" missing from balance sheet
   • "Depreciation" in balance sheet instead of income statement
   • Operating cash flow items mixed into income statement

═══ EXTRACTED DATA TO VERIFY ═══
{data_summary}

═══ OUTPUT FORMAT ═══
Return ONLY this JSON (no markdown fences):

{{
  "placement_correct": true/false,
  "corrections": [
    {{
      "action": "move",
      "key": "net_income",
      "label_raw": "Net income",
      "from_statement": "balance",
      "to_statement": "income",
      "reason": "Net income belongs in income statement, not balance sheet"
    }},
    {{
      "action": "rename_key",
      "statement_type": "balance",
      "old_key": "other_current_assets",
      "new_key": "cash_and_bank_balances",
      "new_label": "Cash and bank balances",
      "reason": "This item represents cash balances, key should reflect that"
    }},
    {{
      "action": "fix_total",
      "statement_type": "income",
      "key": "net_income",
      "should_be_total": true,
      "reason": "Net income is a total/subtotal line"
    }},
    {{
      "action": "remove_duplicate",
      "statement_type": "balance",
      "key": "depreciation_expense",
      "reason": "Depreciation expense belongs only in income statement"
    }}
  ],
  "notes": "Optional overall notes"
}}

═══ RULES ═══
• If everything is correct, return {{"placement_correct": true, "corrections": [], "notes": ""}}
• action types: "move", "rename_key", "fix_total", "remove_duplicate"
• Be conservative — only flag clear mistakes, not ambiguous cases
• Financial statements follow IFRS / standard accounting placement rules
"""


def _apply_placement_corrections(
    statements: List[ExtractedStatement],
    corrections: list,
) -> Tuple[List[ExtractedStatement], int]:
    """
    Apply placement corrections: moves items between statements,
    renames keys, fixes is_total flags, removes duplicates.
    Returns (updated_statements, count_applied).
    """
    applied = 0

    for corr in corrections:
        action = corr.get("action", "")

        if action == "move":
            key = corr.get("key", "")
            from_stmt = _TYPE_MAP.get(corr.get("from_statement", ""), corr.get("from_statement", ""))
            to_stmt = _TYPE_MAP.get(corr.get("to_statement", ""), corr.get("to_statement", ""))
            if not key or not from_stmt or not to_stmt:
                continue

            # Find and remove from source
            moved_item = None
            for s in statements:
                if s.statement_type == from_stmt:
                    for it in s.items:
                        if it.key.lower().replace(" ", "_") == key.lower().replace(" ", "_"):
                            moved_item = it
                            s.items = [i for i in s.items if i is not it]
                            break
                    break

            if moved_item is None:
                continue

            # Add to target
            for s in statements:
                if s.statement_type == to_stmt:
                    moved_item.order_index = len(s.items) + 1
                    s.items.append(moved_item)
                    applied += 1
                    logger.info("Placement: moved %s from %s → %s (%s)",
                                key, from_stmt, to_stmt, corr.get("reason", ""))
                    break

        elif action == "rename_key":
            stmt_type = _TYPE_MAP.get(corr.get("statement_type", ""), corr.get("statement_type", ""))
            old_key = corr.get("old_key", "")
            new_key = corr.get("new_key", "")
            new_label = corr.get("new_label", "")
            if not stmt_type or not old_key or not new_key:
                continue

            for s in statements:
                if s.statement_type == stmt_type:
                    for it in s.items:
                        if it.key.lower().replace(" ", "_") == old_key.lower().replace(" ", "_"):
                            it.key = new_key
                            if new_label:
                                it.label_raw = new_label
                            applied += 1
                            logger.info("Placement: renamed %s → %s (%s)",
                                        old_key, new_key, corr.get("reason", ""))
                            break
                    break

        elif action == "fix_total":
            stmt_type = _TYPE_MAP.get(corr.get("statement_type", ""), corr.get("statement_type", ""))
            key = corr.get("key", "")
            should_be_total = corr.get("should_be_total", False)
            if not stmt_type or not key:
                continue

            for s in statements:
                if s.statement_type == stmt_type:
                    for it in s.items:
                        if it.key.lower().replace(" ", "_") == key.lower().replace(" ", "_"):
                            if it.is_total != should_be_total:
                                it.is_total = should_be_total
                                applied += 1
                                logger.info("Placement: %s is_total → %s (%s)",
                                            key, should_be_total, corr.get("reason", ""))
                            break
                    break

        elif action == "remove_duplicate":
            stmt_type = _TYPE_MAP.get(corr.get("statement_type", ""), corr.get("statement_type", ""))
            key = corr.get("key", "")
            if not stmt_type or not key:
                continue

            for s in statements:
                if s.statement_type == stmt_type:
                    before = len(s.items)
                    s.items = [
                        it for it in s.items
                        if it.key.lower().replace(" ", "_") != key.lower().replace(" ", "_")
                    ]
                    if len(s.items) < before:
                        applied += 1
                        logger.info("Placement: removed duplicate %s from %s (%s)",
                                    key, stmt_type, corr.get("reason", ""))
                    break

    return statements, applied


async def verify_placement(
    pdf_bytes: bytes,
    stock_id: int,
    api_key: str,
    filename: str = "upload.pdf",
    model_name: str = "gemini-2.5-flash",
) -> ExtractionResult:
    """
    Step 3: Verify that every line item is placed in the correct statement
    type with the correct key and is_total flag.

    1. Load cached result
    2. Send to AI for placement verification (no PDF images needed)
    3. Apply placement corrections
    4. Update cache
    5. Return updated result
    """

    h = _pdf_hash(pdf_bytes)

    cached = _get_cached(stock_id, h)
    if not cached:
        raise ValueError(
            "No cached extraction found. Run upload-statement first."
        )

    statements = cached.statements

    logger.info(
        "Placement verification: %s (%d statements, %d total items)",
        filename, len(statements),
        sum(len(s.items) for s in statements),
    )

    # Step 3 does NOT need PDF images — it only analyzes the extracted data
    # This makes it fast and avoids the heavy image processing
    placement_prompt = _build_placement_prompt(statements)
    placement_text = await _call_gemini(api_key, placement_prompt, [], model_name)
    placement_json = _parse_validation_json(placement_text)

    placement_corrections = 0

    if not placement_json.get("placement_correct", True):
        corrections = placement_json.get("corrections", [])
        if corrections:
            logger.info("Placement found %d issue(s) — applying", len(corrections))
            statements, placement_corrections = _apply_placement_corrections(
                statements, corrections,
            )
        else:
            logger.info("Placement flagged issues but no actionable corrections")
    else:
        logger.info("Placement verification passed — all items correctly placed")

    # Re-run arithmetic verification
    checks = _verify_all(statements)
    confidence = _calculate_confidence(checks)

    result = ExtractionResult(
        statements=statements,
        audit_checks=checks,
        confidence=confidence,
        retry_count=cached.retry_count,
        cached=False,
        model_used=model_name,
        pages_processed=cached.pages_processed,
        pdf_hash=h,
        validation_corrections=cached.validation_corrections,
        placement_corrections=placement_corrections,
    )

    # Update cache
    try:
        _set_cache(stock_id, h, filename, result)
    except Exception as exc:
        logger.warning("Failed to update cache after placement: %s", exc)

    logger.info(
        "Placement complete: %d corrections, confidence=%.1f%%",
        placement_corrections, confidence * 100,
    )

    return result
