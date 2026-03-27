"""
Deterministic Cash Flow Statement Reconciler
=============================================

Takes raw AI-extracted cash flow rows and applies accounting rules to:
1. Classify rows by section (operating / investing / financing / cash_bridge)
2. Tag row_kind (header / item / subtotal / total)
3. Reject bad rows (zero-value headers, duplicate totals, misclassified items)
4. Validate section subtotals against sum of their items
5. Validate cash bridge: Beginning Cash + CFO + CFI + CFF + FX = Ending Cash
6. Select best-candidate subtotals when duplicates exist
7. Mark accepted rows vs needs_review

Public API:
  - detect_cashflow_pages(pdf_bytes) → list of 0-indexed page numbers
  - reconcile_cashflow(raw_items, periods) → ReconcileResult
  - validate_cashflow_metrics(rows, periods) → dict of validated CF metrics

This module is pure logic — no AI calls, no DB writes.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# CASH FLOW PAGE DETECTION
# ════════════════════════════════════════════════════════════════════

# Strong signals (one hit = likely CF page)
_CF_STRONG_SIGNALS: List[str] = [
    r"statement\s+of\s+cash\s*flow",
    r"cash\s+flow\s+statement",
    r"consolidated\s+statement\s+of\s+cash\s*flow",
    r"قائمة\s*التدفقات\s*النقدية",  # Arabic
    r"التدفقات\s*النقدية\s*الموحدة",
]

# Medium signals (need 3+ to qualify)
_CF_MEDIUM_SIGNALS: List[str] = [
    r"net\s+cash\s+(?:from|used\s+in|generated|provided)",
    r"cash\s+(?:from|used\s+in)\s+(?:operating|investing|financing)",
    r"cash\s+and\s+cash\s+equivalents?\s+at\s+(?:beginning|end)",
    r"(?:operating|investing|financing)\s+activities",
    r"effect\s+of\s+(?:exchange|foreign\s+currency)",
    r"net\s+(?:increase|decrease)\s+in\s+cash",
    r"capital\s+expenditure",
    r"dividends?\s+paid",
    r"proceeds?\s+from\s+(?:borrowing|loan|issuance)",
    r"repayment\s+of\s+(?:borrowing|loan|debt)",
    r"purchase\s+of\s+(?:property|equipment|intangible)",
    r"depreciation\s+and\s+amortiz",
    r"changes?\s+in\s+working\s+capital",
    r"أنشطة\s*(?:تشغيلية|استثمارية|تمويلية)",  # Arabic activities
]


def detect_cashflow_pages(pdf_bytes: bytes) -> List[int]:
    """Detect pages that likely contain a cash flow statement.

    Uses text extraction via PyMuPDF and keyword heuristics.
    Returns a list of 0-indexed page numbers sorted ascending.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    cf_pages: List[int] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text").lower()

        # Check strong signals first
        for pattern in _CF_STRONG_SIGNALS:
            if re.search(pattern, text):
                cf_pages.append(page_idx)
                break
        else:
            # Count medium signal hits
            hits = sum(
                1 for pattern in _CF_MEDIUM_SIGNALS
                if re.search(pattern, text)
            )
            if hits >= 3:
                cf_pages.append(page_idx)

    doc.close()

    # If we found CF pages, also include immediately adjacent pages
    # (CF statements often span 2 pages)
    if cf_pages:
        expanded = set(cf_pages)
        for p in cf_pages:
            expanded.add(p + 1)  # next page might be continuation
        # Filter to valid page range
        max_page = doc.page_count - 1 if hasattr(doc, 'page_count') else max(cf_pages) + 1
        # Re-open to get page count
        doc2 = fitz.open(stream=pdf_bytes, filetype="pdf")
        max_page = doc2.page_count - 1
        doc2.close()
        expanded = {p for p in expanded if 0 <= p <= max_page}
        cf_pages = sorted(expanded)

    logger.info("Cash flow page detection: found %d pages: %s", len(cf_pages), cf_pages)
    return cf_pages


# ════════════════════════════════════════════════════════════════════
# CLASSIFICATION DICTIONARIES
# ════════════════════════════════════════════════════════════════════

# Section headers — these row labels indicate a section title, NOT a numeric item.
_SECTION_HEADER_PATTERNS: List[Tuple[str, str]] = [
    # (regex pattern, section)
    (r"operating\s+activit", "operating"),
    (r"cash\s+flows?\s+from\s+operat", "operating"),
    (r"investing\s+activit", "investing"),
    (r"cash\s+flows?\s+from\s+invest", "investing"),
    (r"financing\s+activit", "financing"),
    (r"cash\s+flows?\s+from\s+financ", "financing"),
    # Arabic variants
    (r"أنشطة\s*تشغيلية", "operating"),
    (r"أنشطة\s*استثمارية", "investing"),
    (r"أنشطة\s*تمويلية", "financing"),
]

# Section subtotals / totals — mark as subtotal and infer section
_SECTION_TOTAL_PATTERNS: List[Tuple[str, str]] = [
    (r"net\s+cash\s+(?:from|used\s+in|generated\s+from|provided\s+by)\s+operat", "operating"),
    (r"cash\s+(?:from|used\s+in|generated\s+from|provided\s+by)\s+operat", "operating"),
    (r"net\s+cash\s+(?:from|used\s+in|generated\s+from|provided\s+by)\s+invest", "investing"),
    (r"cash\s+(?:from|used\s+in|generated\s+from|provided\s+by)\s+invest", "investing"),
    (r"net\s+cash\s+(?:from|used\s+in|generated\s+from|provided\s+by)\s+financ", "financing"),
    (r"cash\s+(?:from|used\s+in|generated\s+from|provided\s+by)\s+financ", "financing"),
]

# Cash bridge items (net change, opening/closing balances)
_CASH_BRIDGE_PATTERNS: List[Tuple[str, str]] = [
    # (pattern, bridge_role)
    (r"net\s+(?:increase|decrease|change)\s+in\s+cash", "net_change"),
    (r"cash\s+and\s+(?:cash\s+)?equivalents?\s+at\s+(?:the\s+)?(?:beginning|start)", "beginning_cash"),
    (r"cash\s+and\s+bank\s+balances?\s+at\s+(?:the\s+)?(?:beginning|start)", "beginning_cash"),
    (r"(?:opening)\s+(?:cash|balance)", "beginning_cash"),
    (r"cash\s+at\s+(?:the\s+)?beginning", "beginning_cash"),
    (r"cash\s+and\s+(?:cash\s+)?equivalents?\s+at\s+(?:the\s+)?end", "ending_cash"),
    (r"cash\s+and\s+bank\s+balances?\s+at\s+(?:the\s+)?end", "ending_cash"),
    (r"(?:closing)\s+(?:cash|balance)", "ending_cash"),
    (r"cash\s+at\s+(?:the\s+)?end", "ending_cash"),
    (r"effect\s+of\s+(?:exchange|foreign\s+currency)", "fx_effect"),
    (r"foreign\s+(?:exchange|currency)\s+(?:effect|adjustment|difference)", "fx_effect"),
    (r"translation\s+(?:adjustment|difference)", "fx_effect"),
]

# Operating section item keywords (for items that AI might not classify)
_OPERATING_KEYWORDS: List[str] = [
    "depreciation", "amortization", "amortisation",
    "impairment", "provision", "working capital",
    "receivables", "payables", "inventory", "inventories",
    "accrued", "prepaid", "tax paid", "interest paid",
    "interest received", "dividend received",
    "profit for", "net income", "net profit", "net loss",
    "adjustments for", "adjustment for",
    "share of", "unrealized", "unrealised",
    "gain on", "loss on", "write off", "write-off",
    "non-cash", "non cash", "staff indemnity",
    "end of service", "employee benefits",
]

# Investing section item keywords
_INVESTING_KEYWORDS: List[str] = [
    "purchase of property", "purchase of equipment",
    "purchase of intangible", "acquisition of",
    "capital expenditure", "capex",
    "proceeds from sale of", "disposal of",
    "investment in", "investments acquired",
    "deposits placed", "deposits matured",
    "term deposits", "fixed deposits",
    "purchase of investment", "sale of investment",
]

# Financing section item keywords
_FINANCING_KEYWORDS: List[str] = [
    "dividends paid", "dividend paid",
    "proceeds from borrowing", "repayment of borrowing",
    "proceeds from loan", "repayment of loan",
    "debt issued", "debt repaid",
    "lease payment", "lease liability",
    "treasury shares", "buy back", "buyback",
    "share issuance", "share capital",
    "proceeds from issuance",
]

# ════════════════════════════════════════════════════════════════════
# F) LABEL NORMALIZATION — internal codes without losing raw labels
# ════════════════════════════════════════════════════════════════════

# Maps regex patterns (matched against lowered label) → internal code.
# Order matters: first match wins.
_NORMALIZATION_RULES: List[Tuple[str, str]] = [
    # Subtotals
    (r"net\s+cash\s+(?:from|used\s+in|generated|provided)\s+operat", "cfo_reported"),
    (r"cash\s+(?:from|used\s+in|generated|provided)\s+operat", "cfo_reported"),
    (r"net\s+cash\s+(?:from|used\s+in|generated|provided)\s+invest", "cfi_reported"),
    (r"cash\s+(?:from|used\s+in|generated|provided)\s+invest", "cfi_reported"),
    (r"net\s+cash\s+(?:from|used\s+in|generated|provided)\s+financ", "cff_reported"),
    (r"cash\s+(?:from|used\s+in|generated|provided)\s+financ", "cff_reported"),
    # Cash bridge
    (r"cash\s+and\s+(?:cash\s+)?equivalents?\s+at\s+(?:the\s+)?(?:beginning|start)", "beginning_cash"),
    (r"cash\s+and\s+bank\s+balances?\s+at\s+(?:the\s+)?(?:beginning|start)", "beginning_cash"),
    (r"(?:opening)\s+(?:cash|balance)", "beginning_cash"),
    (r"cash\s+at\s+(?:the\s+)?beginning", "beginning_cash"),
    (r"cash\s+and\s+(?:cash\s+)?equivalents?\s+at\s+(?:the\s+)?end", "ending_cash"),
    (r"cash\s+and\s+bank\s+balances?\s+at\s+(?:the\s+)?end", "ending_cash"),
    (r"(?:closing)\s+(?:cash|balance)", "ending_cash"),
    (r"cash\s+at\s+(?:the\s+)?end", "ending_cash"),
    (r"effect\s+of\s+(?:exchange|foreign\s+currency)", "fx_effect"),
    (r"foreign\s+(?:exchange|currency)\s+(?:effect|adjustment|difference)", "fx_effect"),
    (r"translation\s+(?:adjustment|difference)", "fx_effect"),
    (r"net\s+(?:increase|decrease|change)\s+in\s+cash", "net_change_cash"),
    # Capex variants
    (r"purchase\s+of\s+property[,]?\s*(?:plant\s+)?(?:and\s+)?equipment", "capex_ppe_cash"),
    (r"purchase\s+of\s+(?:ppe|pp&e)", "capex_ppe_cash"),
    (r"additions?\s+to\s+property[,]?\s*(?:plant\s+)?(?:and\s+)?equipment", "capex_ppe_cash"),
    (r"capital\s+expenditure", "capex_ppe_cash"),
    (r"\bcapex\b", "capex_ppe_cash"),
    (r"purchase\s+of\s+intangible", "capex_intangibles_cash"),
    (r"payments?\s+for\s+intangible", "capex_intangibles_cash"),
    (r"additions?\s+to\s+intangible", "capex_intangibles_cash"),
    # Dividends
    (r"dividends?\s+paid", "dividends_paid"),
    # Borrowing
    (r"(?:proceeds|receipt)\s+(?:from|of)\s+(?:borrowing|loan|debt)", "proceeds_from_borrowing"),
    (r"repayment\s+of\s+(?:borrowing|loan|debt)", "repayment_of_borrowing"),
    (r"net\s+(?:borrowing|debt)", "net_borrowing"),
    # Common operating items
    (r"depreciation\s+and\s+amortiz", "depreciation_amortization"),
    (r"depreciation", "depreciation"),
    (r"amortiz", "amortization"),
    (r"impairment", "impairment"),
    (r"changes?\s+in\s+working\s+capital", "changes_working_capital"),
    (r"(?:net\s+)?(?:income|profit|loss)\s+(?:for|before|after)", "net_income_cf"),
    (r"profit\s+for\s+the\s+(?:year|period)", "net_income_cf"),
]

# G) Pure section header labels that must NEVER become numeric rows
_PURE_HEADER_LABELS: List[str] = [
    "operating activities",
    "investing activities",
    "financing activities",
    "cash flows from operating activities",
    "cash flows from investing activities",
    "cash flows from financing activities",
    "changes in:",
    "adjustment for:",
    "adjustments for:",
    "adjustments to reconcile",
    "non-cash items",
    "non cash items",
    "working capital changes",
]


def normalize_cashflow_label(label: str) -> Optional[str]:
    """Normalize a raw cash flow label to an internal code.

    Returns the internal code (e.g. 'cfo_reported') or None if no match.
    Does NOT overwrite the raw label — callers store both.
    """
    low = label.strip().lower()
    # Remove footnote numbers
    low = re.sub(r"\s+\d{1,2}\s*$", "", low)
    for pattern, code in _NORMALIZATION_RULES:
        if re.search(pattern, low):
            return code
    return None


# ════════════════════════════════════════════════════════════════════
# DATA TYPES
# ════════════════════════════════════════════════════════════════════

@dataclass
class CashFlowRow:
    """A single row from the cash flow extraction.

    label_raw: original extracted text (never overwritten)
    normalized_code: internal code from normalize_cashflow_label() (may be None)
    """
    row_order: int
    label_raw: str
    section: str           # operating | investing | financing | cash_bridge | unknown
    row_kind: str          # header | item | subtotal | total
    values: Dict[str, Optional[float]]  # period → amount
    normalized_code: Optional[str] = None
    is_total: bool = False
    confidence: float = 1.0
    is_accepted: bool = False
    rejection_reason: Optional[str] = None
    bridge_role: Optional[str] = None  # beginning_cash | ending_cash | net_change | fx_effect
    page: Optional[int] = None


@dataclass
class ReconcileResult:
    """Output of the reconciliation process."""
    rows: List[CashFlowRow]
    status: str            # reconciled | needs_review | failed
    summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def _normalize_label(label: str) -> str:
    """Lowercase, strip whitespace and footnote references."""
    s = label.strip().lower()
    # Remove trailing footnote numbers (e.g. "Cash and bank balances 4")
    s = re.sub(r"\s+\d{1,2}\s*$", "", s)
    return s


def _all_values_null_or_zero(values: Dict[str, Optional[float]]) -> bool:
    """Check if every value is None or 0."""
    return all(v is None or v == 0 or v == 0.0 for v in values.values())


def _is_pure_header(label: str) -> bool:
    """G) Check if a label is a pure section header that must never become a numeric row."""
    low = label.strip().lower()
    # Remove trailing colons/whitespace
    low = low.rstrip(":").strip()
    for h in _PURE_HEADER_LABELS:
        h_clean = h.rstrip(":").strip()
        if h.endswith(":"):
            # Colon-terminated entries → exact match only (e.g. "Changes in:" must not match "Changes in working capital")
            if low == h_clean:
                return True
        else:
            if low == h_clean or low.startswith(h_clean):
                return True
    return False


def _classify_section_from_label(label: str) -> Optional[str]:
    """Try to infer section from the label text using keyword matching."""
    low = label.lower()
    for kw in _OPERATING_KEYWORDS:
        if kw in low:
            return "operating"
    for kw in _INVESTING_KEYWORDS:
        if kw in low:
            return "investing"
    for kw in _FINANCING_KEYWORDS:
        if kw in low:
            return "financing"
    return None


def _get_val(row: CashFlowRow, period: str) -> Optional[float]:
    """Safe getter for a period value."""
    return row.values.get(period)


# ════════════════════════════════════════════════════════════════════
# STEP 1: CLASSIFY ROWS
# ════════════════════════════════════════════════════════════════════

def _classify_rows(rows: List[CashFlowRow]) -> List[CashFlowRow]:
    """
    Assign section, row_kind, bridge_role, and normalized_code to each row.
    Walks top-to-bottom, tracking the current section context.

    IMPORTANT: Check subtotals and bridge items BEFORE headers,
    because subtotal labels like "Net cash from operating activities"
    would falsely match header patterns like "operating activit".

    Also applies F) label normalization and G) pure-header safety.
    """
    current_section = "unknown"

    for row in rows:
        label_low = _normalize_label(row.label_raw)

        # F) Apply label normalization (always prefer label-based code over raw AI key)
        label_code = normalize_cashflow_label(row.label_raw)
        if label_code:
            row.normalized_code = label_code

        # G) Pure header safety — force to header regardless of AI classification
        if _is_pure_header(row.label_raw):
            for pattern, section in _SECTION_HEADER_PATTERNS:
                if re.search(pattern, label_low):
                    row.section = section
                    break
            if row.section == "unknown":
                # Infer section from keywords
                inferred = _classify_section_from_label(row.label_raw)
                if inferred:
                    row.section = inferred
            row.row_kind = "header"
            # Force all values to None for headers
            row.values = {k: None for k in row.values}
            current_section = row.section if row.section != "unknown" else current_section
            continue

        # Check cash bridge items FIRST (most specific)
        is_bridge = False
        for pattern, role in _CASH_BRIDGE_PATTERNS:
            if re.search(pattern, label_low):
                row.section = "cash_bridge"
                row.bridge_role = role
                if role == "net_change":
                    row.row_kind = "total"
                    row.is_total = True
                else:
                    row.row_kind = "item"
                is_bridge = True
                break

        if is_bridge:
            continue

        # Check section subtotals BEFORE headers
        is_subtotal = False
        for pattern, section in _SECTION_TOTAL_PATTERNS:
            if re.search(pattern, label_low):
                row.section = section
                row.row_kind = "subtotal"
                row.is_total = True
                is_subtotal = True
                # Update current_section for items that follow
                current_section = section
                break

        if is_subtotal:
            continue

        # Check if this is a section header
        is_header = False
        for pattern, section in _SECTION_HEADER_PATTERNS:
            if re.search(pattern, label_low):
                row.section = section
                row.row_kind = "header"
                current_section = section
                is_header = True
                break

        if is_header:
            continue

        # Regular item — inherit current section context
        if row.section == "unknown" or row.section == "":
            row.section = current_section

        # Try keyword-based section classification as a fallback
        if row.section == "unknown":
            inferred = _classify_section_from_label(row.label_raw)
            if inferred:
                row.section = inferred

        if row.row_kind not in ("subtotal", "total"):
            row.row_kind = "item"

    return rows


# ════════════════════════════════════════════════════════════════════
# STEP 2: REJECT BAD ROWS
# ════════════════════════════════════════════════════════════════════

def _reject_bad_rows(rows: List[CashFlowRow]) -> List[CashFlowRow]:
    """
    Mark rows that should NOT be committed to final storage.

    Rules (includes G) header row safety):
    - Section headers NEVER become numeric financial rows
    - Pure section headers with zero values → reject unconditionally
    - Items with all null values → reject
    - Duplicate subtotals → handled in _select_best_subtotals
    """
    for row in rows:
        # G) Headers should NEVER have numeric values or participate in totals
        if row.row_kind == "header":
            # Force values to null for safety
            row.values = {k: None for k in row.values}
            row.is_accepted = False
            row.is_total = False
            row.rejection_reason = "Section header — not a financial line item"
            continue

        # G) Additional safety: if a row looks like a pure header but wasn't
        # caught by _classify_rows (e.g. AI set row_kind=item for "OPERATING ACTIVITIES")
        if _is_pure_header(row.label_raw) and row.row_kind != "subtotal":
            row.row_kind = "header"
            row.values = {k: None for k in row.values}
            row.is_accepted = False
            row.is_total = False
            row.rejection_reason = "Pure section header reclassified — not a financial line item"
            continue

        # Rule: Items with all null values (not headers/totals)
        if row.row_kind == "item" and _all_values_null_or_zero(row.values):
            row.is_accepted = False
            row.rejection_reason = "All values are null or zero"
            continue

        # Default: accept
        row.is_accepted = True

    return rows


# ════════════════════════════════════════════════════════════════════
# STEP 3: SELECT BEST SUBTOTALS (duplicate resolution)
# ════════════════════════════════════════════════════════════════════

_TOLERANCE_PCT = 0.02  # 2% tolerance for rounding


def _select_best_subtotals(
    rows: List[CashFlowRow],
    periods: List[str],
) -> Tuple[List[str], Dict[str, CashFlowRow]]:
    """H) When multiple subtotal candidates exist for a section,
    pick the one that best matches the sum of its section's items.
    Also uses cash bridge to choose best CFO when possible.

    Returns (warnings, {section: chosen_subtotal_row}).
    """
    warnings: List[str] = []
    chosen: Dict[str, CashFlowRow] = {}

    # Gather bridge rows for chooseBestCfo logic
    beginning = _find_bridge_row(rows, "beginning_cash")
    ending = _find_bridge_row(rows, "ending_cash")
    fx_row = _find_bridge_row(rows, "fx_effect")

    # Process investing/financing BEFORE operating so bridge check has CFI/CFF
    for section in ("investing", "financing", "operating"):
        candidates = [
            r for r in rows
            if r.section == section and r.row_kind == "subtotal" and r.is_accepted
        ]

        if not candidates:
            continue

        if len(candidates) == 1:
            chosen[section] = candidates[0]
            continue

        # Multiple candidates — compute reconciliation error for each
        section_items = [
            r for r in rows
            if r.section == section and r.row_kind == "item" and r.is_accepted
        ]

        # Track per-candidate errors
        candidate_errors: Dict[int, float] = {}
        best_candidate = candidates[0]
        best_error = float("inf")

        for cand in candidates:
            total_error = 0.0
            for period in periods:
                item_sum = sum((_get_val(r, period) or 0.0) for r in section_items)
                subtotal_val = _get_val(cand, period)
                if subtotal_val is not None:
                    total_error += abs(item_sum - subtotal_val)

            candidate_errors[id(cand)] = total_error
            if total_error < best_error:
                best_error = total_error
                best_candidate = cand

        # H) For CFO, also try cash bridge: chooseBestCfo logic
        if section == "operating" and beginning and ending:
            cfi_row = chosen.get("investing")
            cff_row = chosen.get("financing")
            if cfi_row and cff_row:
                for period in periods:
                    beg = _get_val(beginning, period)
                    end = _get_val(ending, period)
                    cfi = _get_val(cfi_row, period)
                    cff = _get_val(cff_row, period)
                    fx = _get_val(fx_row, period) if fx_row else 0.0

                    if all(v is not None for v in [beg, end, cfi, cff]):
                        bridge_best = None
                        bridge_best_error = float("inf")
                        for cand in candidates:
                            cfo_val = _get_val(cand, period)
                            if cfo_val is not None:
                                calc_end = beg + cfo_val + cfi + cff + (fx or 0.0)
                                err = abs(calc_end - end)
                                if err < bridge_best_error:
                                    bridge_best_error = err
                                    bridge_best = cand
                        if bridge_best and bridge_best is not best_candidate:
                            # Bridge gives a clear winner — prefer it
                            if bridge_best_error < best_error:
                                best_candidate = bridge_best
                                best_error = bridge_best_error
                        break  # Only need one period for the bridge check

        # Accept the best, reject the rest with individual error info
        chosen[section] = best_candidate
        for cand in candidates:
            if cand is not best_candidate:
                cand_error = candidate_errors.get(id(cand), 0.0)
                cand.is_accepted = False
                cand.rejection_reason = (
                    f"Duplicate {section} subtotal rejected — "
                    f"better candidate chosen (this error={cand_error:,.0f}, "
                    f"chosen error={best_error:,.0f})"
                )

        warnings.append(
            f"{section.title()}: {len(candidates)} subtotal candidates found, "
            f"best match chosen (error={best_error:,.0f})"
        )

    return warnings, chosen


# ════════════════════════════════════════════════════════════════════
# STEP 4: VALIDATE SECTION TOTALS
# ════════════════════════════════════════════════════════════════════

def _validate_section_totals(
    rows: List[CashFlowRow],
    periods: List[str],
    chosen_subtotals: Dict[str, CashFlowRow],
) -> Tuple[List[str], List[str]]:
    """
    For each section with a subtotal, verify that subtotal ≈ sum of items.
    Returns (warnings, errors).
    """
    warnings: List[str] = []
    errors: List[str] = []

    for section in ("operating", "investing", "financing"):
        subtotal_row = chosen_subtotals.get(section)
        if not subtotal_row:
            continue

        section_items = [
            r for r in rows
            if r.section == section
            and r.row_kind == "item"
            and r.is_accepted
        ]

        for period in periods:
            item_sum = sum((_get_val(r, period) or 0.0) for r in section_items)
            subtotal_val = _get_val(subtotal_row, period)

            if subtotal_val is None:
                continue

            discrepancy = abs(item_sum - subtotal_val)
            threshold = max(abs(subtotal_val) * _TOLERANCE_PCT, 1.0)

            if discrepancy > threshold:
                msg = (
                    f"{section.title()} subtotal mismatch [{period}]: "
                    f"sum of items={item_sum:,.0f}, "
                    f"subtotal={subtotal_val:,.0f}, "
                    f"diff={discrepancy:,.0f}"
                )
                if discrepancy > abs(subtotal_val) * 0.10:
                    errors.append(msg)
                else:
                    warnings.append(msg)

    return warnings, errors


# ════════════════════════════════════════════════════════════════════
# STEP 5: VALIDATE CASH BRIDGE
# Beginning Cash + CFO + CFI + CFF + FX = Ending Cash
# ════════════════════════════════════════════════════════════════════

def _find_bridge_row(rows: List[CashFlowRow], role: str) -> Optional[CashFlowRow]:
    """Find the accepted cash bridge row with the given role."""
    for r in rows:
        if r.bridge_role == role and r.is_accepted:
            return r
    return None


def _validate_cash_bridge(
    rows: List[CashFlowRow],
    periods: List[str],
    chosen_subtotals: Dict[str, CashFlowRow],
) -> Tuple[List[str], List[str]]:
    """
    Validate the cash bridge identity:
      Beginning Cash + CFO + CFI + CFF + FX = Ending Cash

    Also validates: CFO + CFI + CFF ≈ Net Change in Cash (if available).

    Returns (warnings, errors).
    """
    warnings: List[str] = []
    errors: List[str] = []

    beginning = _find_bridge_row(rows, "beginning_cash")
    ending = _find_bridge_row(rows, "ending_cash")
    net_change = _find_bridge_row(rows, "net_change")
    fx_effect = _find_bridge_row(rows, "fx_effect")

    cfo_row = chosen_subtotals.get("operating")
    cfi_row = chosen_subtotals.get("investing")
    cff_row = chosen_subtotals.get("financing")

    # Check required rows exist
    missing_subtotals = []
    if not cfo_row:
        missing_subtotals.append("operating")
    if not cfi_row:
        missing_subtotals.append("investing")
    if not cff_row:
        missing_subtotals.append("financing")
    if missing_subtotals:
        warnings.append(f"Missing section subtotals: {missing_subtotals}")

    if not beginning:
        warnings.append("Beginning cash row not found — cash bridge cannot be fully validated")
    if not ending:
        warnings.append("Ending cash row not found — cash bridge cannot be fully validated")

    # Validate per period
    for period in periods:
        cfo = _get_val(cfo_row, period) if cfo_row else None
        cfi = _get_val(cfi_row, period) if cfi_row else None
        cff = _get_val(cff_row, period) if cff_row else None
        fx = _get_val(fx_effect, period) if fx_effect else 0.0
        beg = _get_val(beginning, period) if beginning else None
        end = _get_val(ending, period) if ending else None
        net = _get_val(net_change, period) if net_change else None

        # Check 1: CFO + CFI + CFF ≈ Net Change (if net_change exists)
        if all(v is not None for v in [cfo, cfi, cff]) and net is not None:
            computed = cfo + cfi + cff
            if fx is not None:
                computed += fx
            discrepancy = abs(computed - net)
            threshold = max(abs(net) * _TOLERANCE_PCT, 1.0) if net != 0 else 1.0

            if discrepancy > threshold:
                msg = (
                    f"Grand total mismatch [{period}]: "
                    f"CFO({cfo:,.0f}) + CFI({cfi:,.0f}) + CFF({cff:,.0f})"
                    f"{f' + FX({fx:,.0f})' if fx else ''}"
                    f" = {computed:,.0f}, net change={net:,.0f}, "
                    f"diff={discrepancy:,.0f}"
                )
                if discrepancy > max(abs(net) * 0.10, 1.0):
                    errors.append(msg)
                else:
                    warnings.append(msg)

        # Check 2: Beginning + Net Change + FX = Ending (full bridge)
        if beg is not None and end is not None and net is not None:
            fx_val = fx if fx is not None else 0.0
            expected_end = beg + net + fx_val
            # Only add FX if we didn't already include it in net_change
            # Some statements: net_change = CFO+CFI+CFF (without FX)
            # Others: net_change = CFO+CFI+CFF+FX
            # Try both ways and pick the better one
            discrepancy1 = abs(expected_end - end)
            discrepancy2 = abs(beg + net - end)  # FX already in net_change

            discrepancy = min(discrepancy1, discrepancy2)
            threshold = max(abs(end) * _TOLERANCE_PCT, 1.0) if end != 0 else 1.0

            if discrepancy > threshold:
                msg = (
                    f"Cash bridge mismatch [{period}]: "
                    f"Beginning({beg:,.0f}) + Net Change({net:,.0f})"
                    f"{f' + FX({fx_val:,.0f})' if fx_val else ''}"
                    f" ≠ Ending({end:,.0f}), diff={discrepancy:,.0f}"
                )
                if discrepancy > max(abs(end) * 0.10, 1.0):
                    errors.append(msg)
                else:
                    warnings.append(msg)

    return warnings, errors


# ════════════════════════════════════════════════════════════════════
# STEP 6: ENFORCE PER-PERIOD ACCEPTANCE RULES
# ════════════════════════════════════════════════════════════════════

def _enforce_acceptance_rules(
    rows: List[CashFlowRow],
    periods: List[str],
    chosen_subtotals: Dict[str, CashFlowRow],
) -> List[str]:
    """
    Final acceptance rules:
    - Exactly one accepted CFO subtotal per period
    - Exactly one accepted CFI subtotal per period
    - Exactly one accepted CFF subtotal per period
    - Beginning and ending cash must exist

    Returns list of error messages for violated rules.
    """
    rule_errors: List[str] = []

    for section_name in ("operating", "investing", "financing"):
        if section_name not in chosen_subtotals:
            rule_errors.append(
                f"No accepted {section_name} subtotal found"
            )

    beginning = _find_bridge_row(rows, "beginning_cash")
    ending = _find_bridge_row(rows, "ending_cash")

    if not beginning:
        rule_errors.append("No beginning cash row found")
    if not ending:
        rule_errors.append("No ending cash row found")

    return rule_errors


# ════════════════════════════════════════════════════════════════════
# MAIN RECONCILE FUNCTION
# ════════════════════════════════════════════════════════════════════

def reconcile_cashflow(
    raw_items: List[Dict[str, Any]],
    periods: List[str],
) -> ReconcileResult:
    """
    Main entry point: take raw AI-extracted rows and reconcile them.

    Args:
        raw_items: list of dicts with keys:
            label_raw, key, values (dict), is_total, order_index,
            and optionally: section, row_kind
        periods: list of period labels (e.g. ["2024-12-31", "2023-12-31"])

    Returns:
        ReconcileResult with classified, validated rows and summary.
    """
    import json as _json

    # Build CashFlowRow objects from raw AI output
    rows: List[CashFlowRow] = []
    for idx, item in enumerate(raw_items):
        values = item.get("values", {})
        if isinstance(values, str):
            try:
                values = _json.loads(values)
            except Exception:
                values = {}

        # Coerce values to float/None
        clean_values: Dict[str, Optional[float]] = {}
        for k, v in values.items():
            if v is None:
                clean_values[k] = None
            else:
                try:
                    clean_values[k] = float(v)
                except (ValueError, TypeError):
                    clean_values[k] = None

        row = CashFlowRow(
            row_order=item.get("order_index", idx + 1),
            label_raw=item.get("label_raw", ""),
            section=item.get("section", "unknown"),
            row_kind=item.get("row_kind", "item"),
            values=clean_values,
            normalized_code=item.get("key"),
            is_total=bool(item.get("is_total", False)),
        )
        rows.append(row)

    if not rows:
        return ReconcileResult(
            rows=[], status="failed",
            summary={"error": "No rows to reconcile"},
        )

    # Step 1: Classify sections, row kinds, bridge roles
    rows = _classify_rows(rows)

    # Step 2: Reject bad rows (headers with no values, all-null items)
    rows = _reject_bad_rows(rows)

    # Step 3: Select best subtotals when duplicates exist
    dup_warnings, chosen_subtotals = _select_best_subtotals(rows, periods)

    # Step 4: Validate section totals
    section_warnings, section_errors = _validate_section_totals(
        rows, periods, chosen_subtotals,
    )

    # Step 5: Validate cash bridge
    bridge_warnings, bridge_errors = _validate_cash_bridge(
        rows, periods, chosen_subtotals,
    )

    # Step 6: Enforce per-period acceptance rules
    rule_errors = _enforce_acceptance_rules(rows, periods, chosen_subtotals)

    all_warnings = dup_warnings + section_warnings + bridge_warnings
    all_errors = section_errors + bridge_errors + rule_errors

    # Determine status
    accepted_count = sum(1 for r in rows if r.is_accepted)
    rejected_count = sum(1 for r in rows if not r.is_accepted)

    if all_errors:
        status = "needs_review"
    elif all_warnings:
        status = "reconciled"
    else:
        status = "reconciled"

    # Build summary
    section_counts: Dict[str, int] = {}
    for r in rows:
        if r.is_accepted:
            section_counts[r.section] = section_counts.get(r.section, 0) + 1

    bridge_found = {
        "beginning_cash": _find_bridge_row(rows, "beginning_cash") is not None,
        "ending_cash": _find_bridge_row(rows, "ending_cash") is not None,
        "net_change": _find_bridge_row(rows, "net_change") is not None,
        "fx_effect": _find_bridge_row(rows, "fx_effect") is not None,
    }

    summary = {
        "total_rows": len(rows),
        "accepted": accepted_count,
        "rejected": rejected_count,
        "sections": section_counts,
        "bridge_found": bridge_found,
        "chosen_subtotals": {
            s: r.label_raw for s, r in chosen_subtotals.items()
        },
        "warnings_count": len(all_warnings),
        "errors_count": len(all_errors),
    }

    logger.info(
        "Cash flow reconciliation: %d rows, %d accepted, %d rejected, "
        "%d warnings, %d errors → status=%s",
        len(rows), accepted_count, rejected_count,
        len(all_warnings), len(all_errors), status,
    )

    return ReconcileResult(
        rows=rows,
        status=status,
        summary=summary,
        warnings=all_warnings,
        errors=all_errors,
    )


# ════════════════════════════════════════════════════════════════════
# J) VALIDATED CASH FLOW METRICS & FCF COMPUTATION
# ════════════════════════════════════════════════════════════════════

_CAPEX_PPE_CODES = {"capex_ppe_cash"}
_CAPEX_PPE_KEYWORDS = [
    "capital expenditure", "capex",
    "purchase of property", "purchase of equipment",
    "additions to property", "purchase of ppe",
    "payments for property",
]

_CAPEX_INTANGIBLES_CODES = {"capex_intangibles_cash"}
_CAPEX_INTANGIBLES_KEYWORDS = [
    "purchase of intangible", "payments for intangible",
    "additions to intangible",
]


def _find_capex_rows(
    accepted: List[CashFlowRow],
) -> Tuple[Optional[CashFlowRow], Optional[CashFlowRow]]:
    """Find capex PPE and capex intangibles rows using normalized codes + keywords."""
    ppe_row = None
    intang_row = None

    for r in accepted:
        if r.section != "investing" or r.row_kind != "item":
            continue
        code = r.normalized_code or ""
        low = r.label_raw.lower()

        if not ppe_row:
            if code in _CAPEX_PPE_CODES or any(kw in low for kw in _CAPEX_PPE_KEYWORDS):
                ppe_row = r
                continue

        if not intang_row:
            if code in _CAPEX_INTANGIBLES_CODES or any(kw in low for kw in _CAPEX_INTANGIBLES_KEYWORDS):
                intang_row = r

    return ppe_row, intang_row


def compute_validated_cashflow_metrics(
    rows: List[CashFlowRow],
    periods: List[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """J) Compute cash flow metrics from reconciled/accepted rows only.

    FCF = validated CFO - cash CAPEX (PPE + intangibles).
    Does NOT use raw Gemini output — only accepted rows.

    Returns {period: {metric_name: value, ..., "_evidence": {...}}}.
    """
    accepted = [r for r in rows if r.is_accepted]

    # Find key rows
    cfo_row = next(
        (r for r in accepted if r.section == "operating" and r.row_kind == "subtotal"),
        None,
    )
    cfi_row = next(
        (r for r in accepted if r.section == "investing" and r.row_kind == "subtotal"),
        None,
    )
    cff_row = next(
        (r for r in accepted if r.section == "financing" and r.row_kind == "subtotal"),
        None,
    )
    beg_row = next(
        (r for r in accepted if r.bridge_role == "beginning_cash"),
        None,
    )
    end_row = next(
        (r for r in accepted if r.bridge_role == "ending_cash"),
        None,
    )

    capex_ppe_row, capex_intang_row = _find_capex_rows(accepted)

    result: Dict[str, Dict[str, Optional[float]]] = {}

    for period in periods:
        metrics: Dict[str, Optional[float]] = {}

        cfo = _get_val(cfo_row, period) if cfo_row else None
        cfi = _get_val(cfi_row, period) if cfi_row else None
        cff = _get_val(cff_row, period) if cff_row else None
        beg = _get_val(beg_row, period) if beg_row else None
        end = _get_val(end_row, period) if end_row else None
        capex_ppe = _get_val(capex_ppe_row, period) if capex_ppe_row else None
        capex_intang = _get_val(capex_intang_row, period) if capex_intang_row else None

        metrics["cash_from_operations"] = cfo
        metrics["cash_from_investing"] = cfi
        metrics["cash_from_financing"] = cff
        metrics["beginning_cash"] = beg
        metrics["ending_cash"] = end
        metrics["capex_ppe_cash"] = capex_ppe
        metrics["capex_intangibles_cash"] = capex_intang

        # Total cash capex = |PPE capex| + |intangibles capex|
        total_capex: Optional[float] = None
        if capex_ppe is not None:
            total_capex = abs(capex_ppe)
        if capex_intang is not None:
            total_capex = (total_capex or 0.0) + abs(capex_intang)
        metrics["total_capex_cash"] = total_capex

        # J) FCF = validated CFO - cash CAPEX
        if cfo is not None and total_capex is not None:
            metrics["free_cash_flow"] = cfo - total_capex
        else:
            metrics["free_cash_flow"] = None

        # Evidence / confidence for downstream consumers
        metrics["_source_method"] = "validated_cashflow_reconciler"  # type: ignore[assignment]

        result[period] = metrics

    return result
