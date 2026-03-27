"""
Self-Reflective Financial Extraction Pipeline
================================================

A multi-pass AI extraction system that:

1. **Image Prep** — converts PDF pages to 250 DPI images.
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
    visual_anchor_score: float = 0.0  # Vertical position in the anchor (newest) document


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
    raw_ai_text: str = ""            # First-pass AI response (for debugging)


# ════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════


def _safe_float(v: Any) -> Optional[float]:
    """Convert a value to float, returning None for empty/invalid strings."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v = v.strip().replace(",", "")
        if not v or v in ("-", "\u2014", "N/A", "n/a", "nil", "null"):
            return None
        # Handle parenthesised negatives: (1234) -> -1234
        if v.startswith("(") and v.endswith(")"):
            v = "-" + v[1:-1]
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _get_safe_values(raw_values: Any) -> Dict[str, Optional[float]]:
    """Ensure values from AI are always a dict, never a list.

    Prevents the "'list' object has no attribute 'items'" crash when
    the AI model accidentally returns a list instead of a dict.
    """
    if isinstance(raw_values, dict):
        return {str(k): _safe_float(v) for k, v in raw_values.items()}
    if isinstance(raw_values, list):
        logger.warning(
            "AI returned values as list (%d items) instead of dict — "
            "mapping to Unknown_N keys", len(raw_values),
        )
        return {f"Unknown_{i}": _safe_float(v) for i, v in enumerate(raw_values)}
    return {}


# Sentinel for "no value found" — used by _resolve_amount
_MISSING = object()


def _resolve_amount(
    item: "ExtractedLineItem",
    period_info: Dict[str, str],
    stmt_periods: list,
) -> object:
    """Resolve the correct amount for *item* in *period_info* using explicit key matching only.

    Returns the numeric value (float | None) or _MISSING if no match is found.
    Uses period_label → col_name → fuzzy year substring match.
    NEVER falls back to positional indexing — that is the root cause of year-swap bugs.
    """
    period_label = period_info.get("label", "")
    col_name = period_info.get("col_name", period_label)

    # 1. Exact match on period_label
    if period_label in item.values:
        return item.values[period_label]

    # 2. Exact match on col_name
    if col_name and col_name != period_label and col_name in item.values:
        return item.values[col_name]

    # 3. Fuzzy year match — e.g. period "2023" matches key "FY2023" or "2023-12-31"
    year_str = ""
    for ch_group in (period_label, col_name):
        import re as _re
        m = _re.search(r"((?:19|20)\d{2})", ch_group)
        if m:
            year_str = m.group(1)
            break
    if year_str:
        for vk in item.values:
            if year_str in vk:
                return item.values[vk]

    return _MISSING


_TOLERANCE_PCT = 0.02  # 2% tolerance for rounding in audit checks

# Map AI statement types → canonical DB types
_TYPE_MAP = {
    # Short canonical names
    "balance_sheet": "balance",
    "income_statement": "income",
    "cash_flow": "cashflow",
    "equity_statement": "equity",
    "income": "income",
    "balance": "balance",
    "cashflow": "cashflow",
    "equity": "equity",
    # Common short forms the AI may return
    "balance sheet": "balance",
    "income statement": "income",
    "cash flow": "cashflow",
    "cash_flows": "cashflow",
    "equity statement": "equity",
    # Full descriptive names the AI sometimes returns
    "consolidated statement of financial position": "balance",
    "statement of financial position": "balance",
    "consolidated balance sheet": "balance",
    "consolidated statement of profit or loss": "income",
    "statement of profit or loss": "income",
    "consolidated statement of income": "income",
    "statement of income": "income",
    "consolidated income statement": "income",
    "profit or loss": "income",
    "profit_or_loss": "income",
    "profit and loss": "income",
    "profit_and_loss": "income",
    "consolidated statement of cash flows": "cashflow",
    "statement of cash flows": "cashflow",
    "consolidated cash flow statement": "cashflow",
    "cash flow statement": "cashflow",
    "cash_flow_statement": "cashflow",
    "consolidated statement of changes in equity": "equity",
    "statement of changes in equity": "equity",
    "changes in equity": "equity",
    "shareholders equity": "equity",
    "shareholders' equity": "equity",
    "consolidated statement of comprehensive income": "income",
    "statement of comprehensive income": "income",
}

# ── Label normalization maps (matches Streamlit's ai_vision_extractor) ──
# Maps raw label text → canonical key for each statement type.
# Catches AI drift where the model returns raw labels instead of keys.
_BALANCE_SHEET_LABEL_MAP: Dict[str, str] = {
    "cash and bank balances": "cash",
    "cash and cash equivalents": "cash",
    "cash & cash equivalents": "cash",
    "accounts receivable": "accounts_receivable",
    "trade receivables": "accounts_receivable",
    "trade and other receivables": "accounts_receivable",
    "inventories": "inventory",
    "inventory": "inventory",
    "total current assets": "total_current_assets",
    "property plant and equipment": "ppe_net",
    "property, plant and equipment": "ppe_net",
    "goodwill": "goodwill",
    "intangible assets": "intangible_assets",
    "total non-current assets": "total_non_current_assets",
    "total assets": "total_assets",
    "accounts payable": "accounts_payable",
    "trade payables": "accounts_payable",
    "trade and other payables": "accounts_payable",
    "total current liabilities": "total_current_liabilities",
    "total non-current liabilities": "total_non_current_liabilities",
    "total liabilities": "total_liabilities",
    "share capital": "share_capital",
    "common stock": "share_capital",
    "retained earnings": "retained_earnings",
    "total equity": "total_equity",
    "shareholders equity": "total_equity",
    "shareholders' equity": "total_equity",
    "total liabilities and equity": "total_liabilities_and_equity",
    "total liabilities and shareholders equity": "total_liabilities_and_equity",
    "total liabilities and shareholders' equity": "total_liabilities_and_equity",
    "short term debt": "short_term_debt",
    "long term debt": "long_term_debt",
    "short-term borrowings": "short_term_debt",
    "long-term borrowings": "long_term_debt",
    "investment properties": "investment_properties",
    "investments in associates": "investments_in_associates",
    "other current assets": "other_current_assets",
    "other non-current assets": "other_non_current_assets",
    "other current liabilities": "other_current_liabilities",
    "other non-current liabilities": "other_non_current_liabilities",
    "treasury stock": "treasury_shares",
    "treasury shares": "treasury_shares",
    "reserves": "reserves",
    "statutory reserve": "statutory_reserve",
    "voluntary reserve": "voluntary_reserve",
    "general reserve": "general_reserve",
    "fair value reserve": "fair_value_reserve",
    "foreign currency translation reserve": "foreign_currency_translation_reserve",
    "non-controlling interests": "non_controlling_interests",
    "minority interest": "non_controlling_interests",
    "share premium": "share_premium",
}

_INCOME_STATEMENT_LABEL_MAP: Dict[str, str] = {
    "revenue": "revenue",
    "total revenue": "revenue",
    "net revenue": "revenue",
    "sales": "revenue",
    "net sales": "revenue",
    "cost of revenue": "cost_of_revenue",
    "cost of sales": "cost_of_revenue",
    "cost of goods sold": "cost_of_revenue",
    "cost of operations": "cost_of_revenue",
    "gross profit": "gross_profit",
    "operating expenses": "operating_expenses",
    "total operating expenses": "operating_expenses",
    "selling general and administrative": "sga",
    "selling, general and administrative": "sga",
    "general and administrative expenses": "general_and_admin",
    "research and development": "r_and_d",
    "operating income": "operating_income",
    "operating profit": "operating_income",
    "profit from operations": "operating_income",
    "income from operations": "operating_income",
    "interest income": "interest_income",
    "interest expense": "interest_expense",
    "finance costs": "interest_expense",
    "finance cost": "interest_expense",
    "finance income": "interest_income",
    "other income": "other_income",
    "other expenses": "other_expenses",
    "income before tax": "income_before_tax",
    "profit before tax": "income_before_tax",
    "income tax": "income_tax",
    "income tax expense": "income_tax",
    "tax expense": "income_tax",
    "net income": "net_income",
    "net profit": "net_income",
    "profit for the period": "net_income",
    "profit for the year": "net_income",
    "earnings per share": "eps_basic",
    "basic earnings per share": "eps_basic",
    "basic eps": "eps_basic",
    "diluted earnings per share": "eps_diluted",
    "diluted eps": "eps_diluted",
    "basic and diluted earnings per share": "eps_basic",
    "ebitda": "ebitda",
    "depreciation and amortization": "depreciation_amortization",
    "depreciation": "depreciation",
    "amortization": "amortization",
    "share of results of associates": "share_results_associates",
    "share of profit of associates": "share_results_associates",
    "share of loss of associates": "share_results_associates",
    "dividend income": "dividend_income",
    "impairment losses": "impairment_losses",
    # Kuwait / GCC specific
    "contribution to kfas": "contribution_to_kfas",
    "national labour support tax": "nlst",
    "national labor support tax": "nlst",
    "zakat": "zakat",
    "directors remuneration": "directors_remuneration",
    "directors fees": "directors_remuneration",
    "board of directors remuneration": "directors_remuneration",
}

_CASH_FLOW_LABEL_MAP: Dict[str, str] = {
    "cash from operating activities": "cash_from_operations",
    "net cash from operating activities": "cash_from_operations",
    "cash used in operating activities": "cash_from_operations",
    "net cash used in operating activities": "cash_from_operations",
    "cash from investing activities": "cash_from_investing",
    "net cash from investing activities": "cash_from_investing",
    "cash used in investing activities": "cash_from_investing",
    "net cash used in investing activities": "cash_from_investing",
    "cash from financing activities": "cash_from_financing",
    "net cash from financing activities": "cash_from_financing",
    "cash used in financing activities": "cash_from_financing",
    "net cash used in financing activities": "cash_from_financing",
    "net change in cash": "net_change_in_cash",
    "net increase in cash": "net_change_in_cash",
    "net decrease in cash": "net_change_in_cash",
    "increase in cash and cash equivalents": "net_change_in_cash",
    "decrease in cash and cash equivalents": "net_change_in_cash",
    "cash at beginning of period": "beginning_cash",
    "cash at beginning of year": "beginning_cash",
    "cash and cash equivalents at beginning": "beginning_cash",
    "cash at end of period": "ending_cash",
    "cash at end of year": "ending_cash",
    "cash and cash equivalents at end": "ending_cash",
    "capital expenditures": "capital_expenditures",
    "purchase of property plant and equipment": "capital_expenditures",
    "dividends paid": "dividends_paid",
    "debt issued": "debt_issued",
    "debt repaid": "debt_repaid",
    "repayment of borrowings": "debt_repaid",
    "proceeds from borrowings": "debt_issued",
    "depreciation and amortization": "depreciation_amortization",
    "net income": "net_income_cf",
    "profit for the year": "net_income_cf",
    "changes in working capital": "changes_in_working_capital",
}

_LABEL_MAPS: Dict[str, Dict[str, str]] = {
    "balance": _BALANCE_SHEET_LABEL_MAP,
    "income": _INCOME_STATEMENT_LABEL_MAP,
    "cashflow": _CASH_FLOW_LABEL_MAP,
    "equity": {},
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

# Model fallback order — matches Streamlit's config
MODEL_FALLBACK_ORDER = [
    "gemini-2.5-flash",     # Primary — free tier compatible
    "gemini-2.5-pro",       # Fallback — complex documents
    "gemini-2.5-pro-preview-03-25",  # 3.1 Pro — latest model
]
RATE_LIMIT_DELAY = 30   # seconds to wait after 429
API_MAX_RETRIES = 3     # retries per model before fallback

# In-memory image cache: pdf_hash → list of PNG bytes
# Avoids re-rendering PDFs between extraction and validation steps.
_IMAGE_CACHE: Dict[str, List[bytes]] = {}
_IMAGE_CACHE_MAX = 5  # Keep at most 5 PDFs in memory


def _cache_images(h: str, images: List[bytes]) -> None:
    """Store rendered images in memory, evict oldest if over limit."""
    if len(_IMAGE_CACHE) >= _IMAGE_CACHE_MAX:
        oldest = next(iter(_IMAGE_CACHE))
        del _IMAGE_CACHE[oldest]
    _IMAGE_CACHE[h] = images


def _get_cached_images(h: str) -> Optional[List[bytes]]:
    """Retrieve cached images for a given PDF hash."""
    return _IMAGE_CACHE.get(h)


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

        # Reject cache entries with empty/corrupt statements
        if not result.statements or any(
            not s.items or not s.periods for s in result.statements
        ):
            logger.warning(
                "Cache entry for stock %d has empty statements — invalidating",
                stock_id,
            )
            from app.core.database import exec_sql
            exec_sql(
                "DELETE FROM extraction_cache WHERE stock_id = ? AND pdf_hash = ?",
                (stock_id, h),
            )
            return None

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

def pdf_to_images(pdf_bytes: bytes, dpi: int = 250) -> List[bytes]:
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

def _build_extraction_prompt(n_pages: int, existing_codes: Optional[List[Dict[str, str]]] = None) -> str:
    """
    The "Perfect" Self-Correcting Extraction Prompt with Native Visual
    Table Mapping.

    Forces the AI through 5 mandatory internal steps before producing output:
    0. Visual Anchor Identification — find the newest column
    1. Spatial Mapping — identify columns, detect "Notes" traps
    2. Draft Extraction — capture every line item (internal only)
    3. Automated Self-Audit — mandatory arithmetic checks
    4. Self-Correction Logic — fix failures or flag UNCALC_ERROR
    """
    return f"""\
### ROLE
Expert Financial Auditor & Data Engineer (Self-Correcting Mode).
You perform Native Visual Table Mapping — you preserve the visual "DNA"
of the table structure rather than treating it as an OCR text stream.

### OBJECTIVE
Extract every line item from {n_pages} financial statement page(s).
You must ensure 100% mathematical integrity AND preserve the exact
visual row ordering of each company's unique financial layout.

### STEP 0: VISUAL ANCHOR IDENTIFICATION (NEW — MANDATORY)
Before anything else, identify the Anchor Column:
- Find the column with the MOST RECENT year (e.g. FY2024). This is
  your "Anchor Column."
- The top-to-bottom label sequence in this Anchor Column defines the
  canonical row order for the entire extraction.
- Assign each label a visual_index (0, 1, 2, ...) based on its
  physical vertical position in the document.
- **ROW SEQUENCE LOCKING**: The order of items in your output MUST
  match the visual top-to-bottom order of the Anchor Column exactly.
  Do NOT re-sort alphabetically or by financial category — follow the
  document's own structure.
- **HISTORICAL MERGING**: When extracting older years from other pages
  or comparative columns, match each label to the visual_index of its
  corresponding Anchor Column row. If the label matches, keep the
  same position.
- **INSERTIONAL LOGIC**: If an older report has a row that does NOT
  exist in the Anchor Column (e.g. a discontinued item from 2016),
  insert it between its nearest visual neighbours from the old page
  to maintain financial logic. Place it AFTER all anchor items.
- **DIRECT PIXEL VERIFICATION**: Look at the visual coordinates of
  each label and its value. If a label and a number are NOT on the
  same horizontal line in the image, do NOT extract them as a pair.
  This prevents mis-alignment of values across rows.

### STEP 1: SPATIAL MAPPING
Before reading ANY numbers, map the document layout:
- Identify ALL column headers. Typical: Column 1 (latest year, e.g. 2024),
  Column 2 (prior year, e.g. 2023). There may be more columns.
- Identify and COMPLETELY IGNORE the "Notes" column.  The Notes column
  is typically positioned between the label column and the first year
  column.  It contains small reference integers like 4, 5, 7, 8, 10, 15
  that point to footnotes — these are NOT financial amounts.
- **CRITICAL RULE — NOTES COLUMN EXCLUSION**:
  • The Notes column must NEVER appear in your output — not as a period,
    not as a value, not anywhere.
  • Note references are ALWAYS small integers (single-digit or low
    double-digit: 1-30 range) sitting next to the row label text.
  • If you see a column of small numbers (3, 5, 7, 8, 10…) between
    the labels and the year columns, that ENTIRE column is Notes.
    Skip it completely.
  • A value like 5 or 8 next to "Trade receivables" is a footnote
    reference, NOT the trade receivables amount.
  • When in doubt: if a number is suspiciously small compared to other
    values in the same row, it is almost certainly a Note reference.
- Identify statement boundaries: where one statement type ends and
  another begins (e.g. "Statement of Financial Position" vs
  "Statement of Profit or Loss").
- Map each statement to its type:
  • balance_sheet   (Statement of Financial Position / الميزانية العمومية)
  • income_statement (Profit or Loss / قائمة الدخل)
  • cash_flow       (Cash Flows / التدفقات النقدية)
  • equity_statement (Changes in Equity / التغيرات في حقوق الملكية)
- **CRITICAL MERGE RULE**: If the SAME statement type appears on MULTIPLE
  pages or spans MULTIPLE year ranges (e.g. an income statement showing
  2015-2019 on one page and 2020-2025 on another), you MUST merge them
  into a SINGLE statement object with ALL periods and ALL line items
  combined. NEVER output two separate objects of the same statement_type.
  Each line item's "values" dict must contain entries for EVERY period
  column across ALL pages for that statement type.

### STEP 2: DRAFT EXTRACTION (INTERNAL ONLY — do NOT output this)
For each statement type found:
- Capture EVERY line item label and its corresponding numeric values
  for ALL period columns identified in Step 1.
- Parentheses (1,234) = NEGATIVE → record as −1234.
- Dash, blank, or "–" = null.
- Zero = record as 0, do NOT skip.
- "KD'000" or "In thousands" → set unit_scale = 1000.
  "In millions" → unit_scale = 1000000. Otherwise 1.
- Pay close attention to rows with slashes (/) such as
  "Share of profit/(loss)" — these often have sign ambiguity.
- Sub-labels indented under a parent are separate line items.
- **KEY CONSISTENCY RULE**: If the SAME line item appears on MULTIPLE
  pages with slightly different wording (e.g. "Profit for the year" on
  one page vs "Net profit" on another), you MUST use the SAME key for
  both.  Map all variants to a standard key:
    revenue, cost_of_revenue, gross_profit, operating_income,
    net_income (= profit for the year = net profit),
    cash_and_bank_balances, total_assets, total_equity,
    cash_from_operations, cash_from_investing, cash_from_financing,
    net_change_in_cash, etc.
  NEVER generate two different keys for the same financial concept.

### STEP 3: AUTOMATED SELF-AUDIT (MANDATORY — run BEFORE output)
Perform these arithmetic checks internally on your draft data:

FOR BALANCE SHEET:
  1. SUM(individual current asset items) vs "Total current assets" line.
  2. SUM(individual non-current asset items) vs "Total non-current assets" line.
  3. TOTAL ASSETS == (Total current assets + Total non-current assets).
  4. SUM(current liability items) vs "Total current liabilities" line.
  5. SUM(non-current liability items) vs "Total non-current liabilities" line.
  6. TOTAL LIABILITIES == (Total current + Total non-current liabilities).
  7. TOTAL LIABILITIES + TOTAL EQUITY == TOTAL ASSETS.

FOR INCOME STATEMENT:
  5. REVENUE − COST OF OPERATIONS/REVENUE == GROSS PROFIT.
  6. GROSS PROFIT − OPERATING EXPENSES == OPERATING INCOME/PROFIT.
  7. Check NET INCOME/PROFIT is consistent with the chain of additions
     and subtractions from revenue down.

FOR CASH FLOW STATEMENT:
  8. Cash from operations + Cash from investing + Cash from financing
     ≈ Net change in cash (allow ±2% tolerance for FX adjustments).

### CASH FLOW SPECIFIC RULES (CRITICAL)
When extracting cash flow statements, follow these additional rules:
- **SECTION HEADERS**: Preserve section header rows exactly as they appear.
  Mark them with is_total=false.  Typical headers:
  • "Cash flows from operating activities"
  • "Cash flows from investing activities"
  • "Cash flows from financing activities"
- **SECTION SUBTOTALS**: Every section typically ends with a subtotal like
  "Net cash from operating activities" or "Cash used in investing activities".
  Mark these with is_total=true.
- **SIGNS**: Cash outflows are typically shown in parentheses — e.g.
  "(53,200)" → record as −53200.  This is critical for investing and
  financing sections where payments, purchases, and repayments are negative.
- **BRIDGE ROWS**: Rows like "Net increase/(decrease) in cash",
  "Cash at beginning of period", "Cash at end of period" should all be
  extracted.  They form the cash reconciliation bridge.
- **DO NOT MERGE SUBTOTALS**: If "Cash from operations" appears as both a
  section total and a bridge line, extract BOTH rows — the reconciler will
  handle deduplication.
- **COMPLETE EXTRACTION**: Extract every single row including small items
  like "Effect of exchange rate changes", "Bank overdrafts", or
  "Restricted cash". Missing rows cause reconciliation failures.

### STEP 4: SELF-CORRECTION LOGIC
If ANY check in Step 3 FAILS:
- Re-scan the image for the specific rows that caused the failure.
- Look for:
  • Missing rows — small items hidden between larger ones.
  • OCR/reading errors — digits transposed, commas misread.
  • Sign errors — items in parentheses incorrectly read as positive.
  • Rows with slashes (/) or special characters that were skipped.
  • "Share of profit/loss of associates" or similar compound items.
- Adjust values until the mathematical audit passes.
- If a number is genuinely unreadable after re-scanning, record the
  value as null and add "UNCALC_ERROR: [reason]" to audit_notes.
- Record every correction you made in the "corrections_made" array.

### OUTPUT FORMAT
After ALL self-audit checks pass (or failures are flagged with UNCALC_ERROR),
return ONLY this JSON — no markdown fences, no commentary, no explanation:

[
  {{
    "statement_type": "balance_sheet",
    "source_pages": [1],
    "currency": "KWD",
    "unit_scale": 1,
    "periods": [
      {{"label": "2024-12-31", "col_name": "2024"}},
      {{"label": "2023-12-31", "col_name": "2023"}}
    ],
    "items": [
      {{
        "label_raw": "Cash and bank balances",
        "key": "cash_and_bank_balances",
        "values": {{"2024-12-31": 20150329, "2023-12-31": 18432100}},
        "is_total": false
      }},
      {{
        "label_raw": "Total current assets",
        "key": "total_current_assets",
        "values": {{"2024-12-31": 40293699, "2023-12-31": 37856200}},
        "is_total": true
      }}
    ],
    "audit": {{
      "checks_performed": [
        {{
          "rule": "total_assets = total_current_assets + total_non_current_assets",
          "expected": 500000,
          "actual": 500000,
          "passed": true
        }},
        {{
          "rule": "total_liabilities + total_equity = total_assets",
          "expected": 500000,
          "actual": 500000,
          "passed": true
        }}
      ],
      "corrections_made": [
        "Corrected 'Trade receivables' 2023 from 1234 to 12340 — missing trailing zero"
      ],
      "audit_notes": ""
    }}
  }}
]

### ABSOLUTE RULES
• "values" must contain numbers or null — NEVER strings.
• Parentheses (1,234) → −1234.  Dash or blank → null.
• Detect unit_scale from headers (e.g. "KD'000" → 1000).
• Period labels → ISO dates: "31 December 2024" → "2024-12-31".
• is_total=true ONLY for summation lines (subtotals, totals, grand totals).
• source_pages is 1-indexed.
• YOU MUST EXTRACT EVERY SINGLE LINE ITEM — do NOT skip any row.
• Copy every number EXACTLY as printed — do NOT round or approximate.
• Zero values → include as 0, NOT omit.
• **NOTES COLUMN = INVISIBLE**: Completely ignore the Notes/footnote
  reference column.  Never extract note numbers as amounts.  If a column
  contains only small integers (1-30), it is the Notes column — skip it.
  A note reference (e.g. "5") next to a label is NOT that item's value.
• ONE object per statement_type — merge multi-page/multi-year data into one.
  Each item's "values" must include ALL period columns found across all pages.
• EVERY item must have a value (number or null) for EVERY period — no partial
  coverage.  If page A has 2020-2025 and page B has 2015-2019, each item must
  have entries for ALL 11 years.
• Use CONSISTENT keys across pages.  Same concept = same key, always.
• **VISUAL ORDER IS LAW**: Items in the output MUST appear in the same
  top-to-bottom order as they do in the document image.  The Anchor Column
  (most recent year) defines this order.  Do NOT re-sort by category,
  alphabet, or any other scheme.  A bank's statement structure must look
  like a bank; an industrial company's must look like theirs.
• Verify horizontal alignment — a label and its value must be on the same
  row in the image.  If they are not visually aligned, do NOT pair them.
"""
    # Inject existing codes so the AI reuses them for continuity
    if existing_codes:
        codes_block = "\n".join(
            f'  - key: "{c["code"]}"  label: "{c["name"]}"  (type: {c["type"]})'
            for c in existing_codes
        )
        prompt += f"""

### EXISTING LINE ITEM KEYS (MANDATORY REUSE)
This stock ALREADY has financial data with these line item keys from
previously uploaded reports. You MUST reuse the EXACT same "key" value
for any line item that matches the same financial concept, even if the
wording in the new PDF differs slightly.

{codes_block}

RULES FOR REUSE:
• If a line item in the new PDF matches any of the above concepts,
  use the EXISTING key — do NOT invent a new variant.
• Example: if existing key is "provision_for_staff_indemnity" and the
  new PDF says "Provision for indemnity", use "provision_for_staff_indemnity".
• Only create a NEW key for line items that genuinely do not exist above.
• This ensures data continuity across multiple report periods.
"""
    return prompt


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
6. COMPLETELY IGNORE the Notes column (small reference integers like
   3, 5, 7, 8 next to labels). These are footnote references, NOT amounts.
   If a value seems suspiciously small (single or double digit) compared to
   other amounts in the same row, it is almost certainly a Note reference.

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
### ROLE
Senior Financial Auditor performing CROSS-VALIDATION AUDIT (Step 2 of 3).

### CONTEXT
I am giving you {n_pages} page(s) from a financial report AND the data
that was previously extracted from these same pages by an AI system.
Your job is to act as the SECOND PAIR OF EYES — like a financial auditor
reviewing a junior analyst's work product.

### STEP 1: LINE-BY-LINE COMPARISON
Go through the ORIGINAL DOCUMENT image(s) row by row, left to right.
For EACH line item visible in the document:
  1. Find the corresponding entry in the EXTRACTED DATA below.
  2. Compare the label, key, and EVERY period's numeric value.
  3. Verify the sign — parenthesized values must be negative.
  4. Check that "Notes" column numbers were NOT captured as amounts.
     The Notes column typically sits between labels and year columns and
     contains only small integers (1-30). These are footnote references.
     If ANY extracted value is a small integer that looks like a note
     number rather than a financial amount, flag it as a correction.

### STEP 2: COMPLETENESS CHECK
After the line-by-line sweep, verify:
  1. **MISSING LINE ITEMS** — rows visible in the document but absent
     from extracted data. Pay attention to:
     - Small items between major sections
     - Items with slashes like "Share of profit/(loss)"
     - Footnote-referenced adjustments
  2. **MISSING PERIODS** — year columns in the document header that
     are not in the extracted data. Check for comparative years.
  3. **MISSING STATEMENTS** — entire statement types visible in the
     PDF but not extracted (e.g., Notes to Financial Statements
     contains a separate schedule).

### STEP 3: ARITHMETIC RE-VERIFICATION
Re-run the same audit checks on the extracted data:
  - SUM(current assets) vs "Total current assets"
  - SUM(non-current assets) vs "Total non-current assets"
  - Total assets = current + non-current
  - Total liabilities + Total equity = Total assets
  - Revenue − Cost = Gross profit
  - Cash from ops + investing + financing ≈ Net change
If a check fails, find the specific row causing the discrepancy.

### STEP 4: PRODUCE CORRECTIONS
For each discrepancy found, produce a correction entry.

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
      "reason": "Missing from extraction — visible on page 2, row 3"
    }}
  ],
  "missing_statements": [
    {{
      "statement_type": "cashflow",
      "reason": "Cash flow statement on pages 4-5 was not extracted"
    }}
  ],
  "notes": "Overall audit notes"
}}

═══ RULES ═══
• If everything is correct: {{"validation_passed": true, "corrections": [], "missing_statements": [], "notes": ""}}
• "add" → item was missing, needs to be added with new_value
• "update" → value was wrong, include old_value and new_value
• "remove" → item doesn't exist in the document (rare)
• Values must be numbers or null — NEVER strings
• Parentheses (1,234) → −1234.  Dash or blank → null
• Check EVERY SINGLE ROW — act like an auditor, not a skimmer
• Pay special attention to: cash balances, total equity, retained earnings,
  minority interest, and items near section boundaries
• Check ALL year columns — older comparative years are often missed
• DO NOT approve data that has arithmetic failures — flag them
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
                    existing.values[period] = _safe_float(new_value)
                    applied += 1
                    logger.info("Validation: added period %s to %s = %s (%s)",
                                period, key, new_value, reason)
            else:
                # Brand new item
                new_item = ExtractedLineItem(
                    key=key,
                    label_raw=label_raw,
                    values={period: _safe_float(new_value)},
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
                    it.values[period] = _safe_float(new_value)
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

def _get_json_repair():
    """Lazy-import json_repair for robust JSON parsing."""
    try:
        import json_repair as jr
        return jr
    except ImportError:
        return None


def _parse_ai_json(text: str) -> list:
    """Parse JSON from Gemini response with multi-stage repair.

    Mirrors Streamlit's proven pipeline: direct parse → json_repair →
    bracket extraction → regex fixes → NaN handling.
    """
    cleaned = text.strip()
    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Stage 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Stage 2: json_repair library (matches Streamlit)
    jr = _get_json_repair()
    if jr is not None:
        try:
            repaired = jr.repair_json(cleaned, return_objects=True)
            if isinstance(repaired, list) and repaired:
                return repaired
            if isinstance(repaired, dict):
                return [repaired]
        except Exception:
            pass

    # Stage 3: strip leading non-JSON text (thinking/reasoning output)
    first_bracket = cleaned.find("[")
    if first_bracket > 0:
        stripped = cleaned[first_bracket:]
        depth = 0
        end_pos = -1
        for i, ch in enumerate(stripped):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end_pos = i
        if end_pos >= 0:
            candidate = stripped[: end_pos + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                if jr is not None:
                    try:
                        repaired = jr.repair_json(candidate, return_objects=True)
                        if repaired:
                            return repaired if isinstance(repaired, list) else [repaired]
                    except Exception:
                        pass

    # Stage 4: regex fixes — trailing commas, unquoted keys
    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    fixed = re.sub(r'(?m)^(\s*)([a-zA-Z_]\w*)\s*:', r'\1"\2":', fixed)

    # Balance braces
    opens = fixed.count("{") - fixed.count("}")
    if opens > 0:
        fixed += "}" * opens
    closeb = fixed.count("[") - fixed.count("]")
    if closeb > 0:
        fixed += "]" * closeb

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    if jr is not None:
        try:
            repaired = jr.repair_json(fixed, return_objects=True)
            if repaired:
                return repaired if isinstance(repaired, list) else [repaired]
        except Exception:
            pass

    # Stage 5: extract first JSON array/object
    for pattern in (r'\[[\s\S]*\]', r'\{[\s\S]*\}'):
        m = re.search(pattern, cleaned)
        if m:
            try:
                parsed = json.loads(m.group())
                return parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                if jr is not None:
                    try:
                        repaired = jr.repair_json(m.group(), return_objects=True)
                        if repaired:
                            return repaired if isinstance(repaired, list) else [repaired]
                    except Exception:
                        pass

    # Stage 6: fix NaN / Infinity
    fixed2 = re.sub(r'\bNaN\b', 'null', fixed)
    fixed2 = re.sub(r'\bInfinity\b', 'null', fixed2)
    fixed2 = re.sub(r'\b-Infinity\b', 'null', fixed2)
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    raise ValueError("AI returned unparseable JSON. Try again with a clearer PDF.")


def _unwrap_to_statement_list(data) -> list:
    """Normalise any parsed JSON into a flat list of statement dicts.

    Gemini sometimes returns:
      - A list of statement dicts (correct) → pass through
      - A wrapper dict  {"statements": [...]}  → unwrap
      - A list containing a wrapper dict  [{"statements": [...]}]  → unwrap
      - Nested lists  [[{...}, ...]]  → flatten
    """
    if isinstance(data, dict):
        # {"statements": [...]} wrapper
        for key in ("statements", "financial_statements", "data", "results"):
            if key in data and isinstance(data[key], list):
                return _unwrap_to_statement_list(data[key])
        # Single statement dict → wrap in list
        if "statement_type" in data:
            return [data]
        return []

    if isinstance(data, list):
        result: list = []
        for item in data:
            if isinstance(item, dict):
                # Check for wrapper dict inside list
                unwrapped = False
                for key in ("statements", "financial_statements", "data", "results"):
                    if key in item and isinstance(item[key], list):
                        result.extend(_unwrap_to_statement_list(item[key]))
                        unwrapped = True
                        break
                if not unwrapped:
                    result.append(item)
            elif isinstance(item, list):
                # Nested list → flatten
                result.extend(_unwrap_to_statement_list(item))
            # Skip non-dict/non-list items (strings, numbers, None)
        return result

    return []


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
    Also detects null/missing values in key items.
    """
    rules = _AUDIT_RULES.get(stmt.statement_type, {})
    if not rules:
        return []

    # Build a fast lookup: key → {period → value}
    # Also track original None values for diagnostics
    item_map: Dict[str, Dict[str, float]] = {}
    null_items: Dict[str, Dict[str, bool]] = {}  # key → {period → True if null}
    for it in stmt.items:
        k = it.key.lower().replace(" ", "_").replace("-", "_")
        item_map[k] = {}
        null_items[k] = {}
        for p, v in it.values.items():
            item_map[k][p] = v if v is not None else 0.0
            if v is None:
                null_items[k][p] = True

    checks: List[AuditCheck] = []

    for total_key, part_keys in rules.items():
        total_vals = item_map.get(total_key, {})
        if not total_vals:
            continue

        for period in total_vals:
            total_val = total_vals[period]

            # Collect component values and track nulls
            parts_sum = 0.0
            parts_found = 0
            component_details = []
            null_components = []
            missing_components = []
            for pk in part_keys:
                if pk in item_map and period in item_map[pk]:
                    parts_sum += item_map[pk][period]
                    parts_found += 1
                    component_details.append(f"{pk}={item_map[pk][period]:.0f}")
                    if pk in null_items and period in null_items[pk]:
                        null_components.append(pk)
                else:
                    missing_components.append(pk)

            # Only check if we found at least some components
            if parts_found == 0:
                continue

            # For income: gross_profit = revenue - COGS (subtraction)
            if total_key == "gross_profit" and stmt.statement_type == "income":
                rev = item_map.get("revenue", {}).get(period, 0)
                cogs = item_map.get("cost_of_revenue", {}).get(period, 0)
                parts_sum = rev - abs(cogs) if cogs < 0 else rev - cogs
                component_details = [f"revenue={rev:.0f}", f"cost_of_revenue={cogs:.0f}"]

            disc = abs(total_val - parts_sum)
            threshold = max(abs(total_val) * _TOLERANCE_PCT, 1.0)
            passed = disc <= threshold

            # Build a rich detail string with component breakdown
            if passed:
                detail = f"OK: {total_key}={total_val:.0f}"
            else:
                detail_parts = [
                    f"FAIL: {total_key}={total_val:.0f} but "
                    f"sum({'+'.join(part_keys)})={parts_sum:.0f} "
                    f"(diff={disc:.0f})"
                ]
                if component_details:
                    detail_parts.append(f"Components: {', '.join(component_details)}")
                if null_components:
                    detail_parts.append(
                        f"NULL values: {', '.join(null_components)} "
                        f"(returned as N/A or empty)"
                    )
                if missing_components:
                    detail_parts.append(
                        f"Missing items: {', '.join(missing_components)} "
                        f"(not found in extraction)"
                    )
                detail = " | ".join(detail_parts)

            checks.append(AuditCheck(
                statement_type=stmt.statement_type,
                period=period,
                total_label=total_key,
                total_value=total_val,
                computed_sum=parts_sum,
                discrepancy=disc,
                passed=passed,
                detail=detail,
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
# AI ATTRIBUTION EXPERT (inline post-extraction quality pass)
# ════════════════════════════════════════════════════════════════════

def _build_attribution_prompt(
    extracted_data: List[ExtractedStatement],
    n_pages: int,
) -> str:
    """
    Build a prompt that uses AI as a financial-statement attribution expert.

    Covers:
    - Correct statement-type placement (e.g. depreciation → income, not balance)
    - Canonical key naming (consistent, query-friendly keys)
    - is_total flag accuracy
    - Cross-statement value consistency
    - Missing item detection
    - Value sign corrections (parentheses = negative)
    """
    data_parts = []
    for stmt in extracted_data:
        period_labels = [p.get("label", p.get("col_name", "?")) for p in stmt.periods]
        lines = []
        for it in stmt.items:
            vals = ", ".join(
                f"{p}: {it.values.get(p, 'MISSING')}" for p in period_labels
            )
            total_tag = " [TOTAL]" if it.is_total else ""
            lines.append(f"    {it.label_raw} (key={it.key}){total_tag}: {vals}")
        data_parts.append(
            f"  Statement: {stmt.statement_type}\n"
            f"  Currency: {stmt.currency}, Scale: {stmt.unit_scale}\n"
            f"  Periods: {period_labels}\n"
            f"  Items ({len(stmt.items)} rows):\n" + "\n".join(lines)
        )

    data_summary = "\n\n".join(data_parts)

    return f"""\
### ROLE
You are a Senior Financial Analyst acting as an ATTRIBUTION EXPERT.
You are given {n_pages} page(s) of a financial report and the data that
was extracted from them. Your job is to FIX any attribution errors — act
as the final quality gate before this data enters a financial database.

### WHAT TO CHECK

**1. STATEMENT-TYPE PLACEMENT**
Every line item must be under the CORRECT statement:
  • Balance Sheet ("balance"): Assets, Liabilities, Equity items
  • Income Statement ("income"): Revenue, Costs, Expenses, Profit/Loss, EPS
  • Cash Flow ("cashflow"): Operating/Investing/Financing activities, Cash beginning/end
  • Equity Statement ("equity"): Share capital movements, retained earnings movements

Common misplacements:
  ✗ "Depreciation" in balance → belongs in income (or cashflow adjustments)
  ✗ "Net income" in balance → belongs in income
  ✗ Operating cash items mixed into income
  ✗ "Retained earnings" in income → belongs in balance/equity

**2. KEY ATTRIBUTION (Canonical Naming)**
Each item needs a machine-friendly key that matches standard financial taxonomy:
  • Revenue items: "revenue", "cost_of_revenue", "gross_profit"
  • Expense items: "operating_expenses", "sga", "r_and_d", "interest_expense"
  • Profit items: "operating_income", "income_before_tax", "net_income"
  • Asset items: "cash", "accounts_receivable", "inventory", "ppe_net", "total_assets"
  • Liability items: "accounts_payable", "total_liabilities", "short_term_debt"
  • Equity items: "share_capital", "retained_earnings", "total_equity"
  • Cash flow: "cash_from_operations", "cash_from_investing", "cash_from_financing"

Fix keys that are:
  - Too generic ("item_1", "other", "value")
  - Duplicated across statements with different meanings
  - Not snake_case or inconsistent naming

**3. is_total FLAG**
  • Total/subtotal rows (Total Assets, Gross Profit, Net Income) → is_total: true
  • Individual line items → is_total: false
  • Watch for: "Profit before tax" (TOTAL), "Profit for the year" (TOTAL),
    "Total comprehensive income" (TOTAL), "Total current assets" (TOTAL)

**4. VALUE CORRECTIONS**
  • Parenthesized amounts must be negative: (1,234) → -1234
  • Dashes or blanks → null (not 0)
  • Verify signs make sense (expenses should typically be positive or
    negative depending on presentation — match the document convention)
  • Check that comparative period values weren't swapped

**5. CROSS-STATEMENT CONSISTENCY**
  • "Cash" on balance sheet ≈ "Ending cash" on cash flow
  • "Net income" on income statement ≈ flows into equity statement
  • "Retained earnings" on balance sheet ≈ ending balance in equity
  • "Total equity" on balance sheet ≈ total in equity statement

**6. MISSING CRITICAL ITEMS**
Look at the PDF images and flag if any of these key items are in the
document but MISSING from the extraction:
  • Revenue, Cost of Revenue, Gross Profit, Net Income (income)
  • Total Assets, Total Liabilities, Total Equity (balance)
  • Cash from Operations, Investing, Financing (cashflow)

═══ EXTRACTED DATA TO REVIEW ═══
{data_summary}

═══ OUTPUT FORMAT ═══
Return ONLY this JSON (no markdown fences):

{{
  "attribution_correct": true/false,
  "corrections": [
    {{
      "action": "move",
      "key": "depreciation_expense",
      "label_raw": "Depreciation and amortization",
      "from_statement": "balance",
      "to_statement": "income",
      "reason": "Depreciation is an income statement expense"
    }},
    {{
      "action": "rename_key",
      "statement_type": "income",
      "old_key": "item_1",
      "new_key": "revenue",
      "new_label": "Revenue",
      "reason": "This is the company's top-line revenue"
    }},
    {{
      "action": "fix_total",
      "statement_type": "balance",
      "key": "total_assets",
      "should_be_total": true,
      "reason": "Total assets is a total row"
    }},
    {{
      "action": "update_value",
      "statement_type": "income",
      "period": "2024-12-31",
      "key": "cost_of_revenue",
      "old_value": 5000,
      "new_value": -5000,
      "reason": "Cost shown in parentheses in document, should be negative"
    }},
    {{
      "action": "add",
      "statement_type": "income",
      "period": "2024-12-31",
      "key": "gross_profit",
      "label_raw": "Gross Profit",
      "new_value": 15000,
      "is_total": true,
      "reason": "Gross profit visible on page 2 but missing from extraction"
    }},
    {{
      "action": "remove_duplicate",
      "statement_type": "balance",
      "key": "operating_expenses",
      "reason": "Operating expenses belongs only in income statement"
    }}
  ],
  "notes": "Summary of findings"
}}

═══ RULES ═══
• If everything is correct: {{"attribution_correct": true, "corrections": [], "notes": ""}}
• Be an EXPERT — catch subtle misplacements that automation would miss
• Compare extracted data against the ORIGINAL PDF images to verify
• Only flag CLEAR mistakes — not ambiguous border cases
• Values must be numbers or null — NEVER strings
• Focus on what matters for financial analysis: correct placement,
  correct keys, correct totals, correct signs
"""


def _apply_attribution_fixes(
    statements: List[ExtractedStatement],
    corrections: list,
) -> Tuple[List[ExtractedStatement], int]:
    """Apply attribution corrections from the AI expert.

    Handles: move, rename_key, fix_total, update_value, add, remove_duplicate.
    Returns (updated_statements, count_applied).
    """
    applied = 0

    for corr in corrections:
        if not isinstance(corr, dict):
            continue
        action = corr.get("action", "")

        if action == "move":
            key = corr.get("key", "")
            from_stmt = _TYPE_MAP.get(corr.get("from_statement", ""), corr.get("from_statement", ""))
            to_stmt = _TYPE_MAP.get(corr.get("to_statement", ""), corr.get("to_statement", ""))
            if not key or not from_stmt or not to_stmt:
                continue

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

            for s in statements:
                if s.statement_type == to_stmt:
                    moved_item.order_index = len(s.items) + 1
                    s.items.append(moved_item)
                    applied += 1
                    logger.info("Attribution: moved %s from %s → %s (%s)",
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
                            logger.info("Attribution: renamed %s → %s (%s)",
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
                                logger.info("Attribution: %s is_total → %s (%s)",
                                            key, should_be_total, corr.get("reason", ""))
                            break
                    break

        elif action == "update_value":
            stmt_type = _TYPE_MAP.get(corr.get("statement_type", ""), corr.get("statement_type", ""))
            period = corr.get("period", "")
            key = corr.get("key", "")
            new_value = corr.get("new_value")
            if not stmt_type or not key or not period:
                continue

            for s in statements:
                if s.statement_type == stmt_type:
                    for it in s.items:
                        if it.key.lower().replace(" ", "_") == key.lower().replace(" ", "_"):
                            old = it.values.get(period)
                            it.values[period] = _safe_float(new_value)
                            applied += 1
                            logger.info("Attribution: updated %s [%s] from %s to %s (%s)",
                                        key, period, old, new_value, corr.get("reason", ""))
                            break
                    break

        elif action == "add":
            stmt_type = _TYPE_MAP.get(corr.get("statement_type", ""), corr.get("statement_type", ""))
            period = corr.get("period", "")
            key = corr.get("key", "")
            label_raw = corr.get("label_raw", key)
            new_value = corr.get("new_value")
            is_total = bool(corr.get("is_total", False))
            if not stmt_type or not key:
                continue

            for s in statements:
                if s.statement_type == stmt_type:
                    existing = None
                    for it in s.items:
                        if it.key.lower().replace(" ", "_") == key.lower().replace(" ", "_"):
                            existing = it
                            break

                    if existing and period:
                        if period not in existing.values or existing.values[period] is None:
                            existing.values[period] = _safe_float(new_value)
                            applied += 1
                            logger.info("Attribution: added period %s to %s = %s (%s)",
                                        period, key, new_value, corr.get("reason", ""))
                    elif not existing:
                        values = {}
                        if period and new_value is not None:
                            values[period] = _safe_float(new_value)
                        new_item = ExtractedLineItem(
                            key=key,
                            label_raw=label_raw,
                            values=values,
                            is_total=is_total,
                            order_index=len(s.items) + 1,
                        )
                        s.items.append(new_item)
                        applied += 1
                        logger.info("Attribution: added new item %s = %s (%s)",
                                    key, new_value, corr.get("reason", ""))
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
                        logger.info("Attribution: removed %s from %s (%s)",
                                    key, stmt_type, corr.get("reason", ""))
                    break

    return statements, applied


async def _ai_attribution_pass(
    api_key: str,
    statements: List[ExtractedStatement],
    page_images: List[bytes],
    model_name: str = "gemini-2.5-flash",
) -> Tuple[List[ExtractedStatement], int]:
    """Run the AI attribution expert pass on extracted statements.

    Sends the extracted data + PDF images to AI for expert review.
    The AI checks placement, keys, totals, values, and consistency.
    Returns (corrected_statements, corrections_count).
    """
    if not statements:
        return statements, 0

    prompt = _build_attribution_prompt(statements, len(page_images))

    try:
        response_text = await _call_gemini(api_key, prompt, page_images, model_name)
        result_json = _parse_validation_json(response_text)
    except Exception as exc:
        logger.warning("AI attribution pass failed (non-fatal): %s", exc)
        return statements, 0

    if result_json.get("attribution_correct", True):
        logger.info("AI attribution expert: all items correctly attributed")
        return statements, 0

    corrections = result_json.get("corrections", [])
    if not corrections:
        logger.info("AI attribution flagged issues but no actionable corrections")
        return statements, 0

    logger.info("AI attribution expert found %d issue(s) — applying", len(corrections))
    notes = result_json.get("notes", "")
    if notes:
        logger.info("Attribution notes: %s", notes)

    statements, applied = _apply_attribution_fixes(statements, corrections)
    logger.info("AI attribution applied %d/%d corrections", applied, len(corrections))

    return statements, applied


# ════════════════════════════════════════════════════════════════════
# TARGETED EXTRACTION (fallback for missing statement types)
# ════════════════════════════════════════════════════════════════════

_STMT_HINTS = {
    "balance": "Balance Sheet / Statement of Financial Position",
    "income": "Income Statement / Statement of Profit or Loss",
    "cashflow": "Cash Flow Statement / Statement of Cash Flows",
    "equity": "Statement of Changes in Equity",
}

_VALID_EXTRACT_TYPES = {"balance", "income", "cashflow"}


def _build_targeted_prompt(stmt_type: str, n_pages: int) -> str:
    """Build a focused extraction prompt for a single statement type."""
    hint = _STMT_HINTS.get(stmt_type, stmt_type)
    return f"""\
You are a financial statement extraction engine.
I am giving you {n_pages} page(s) from a financial report.

TASK: Find and extract ONLY the **{hint}** from these pages.
Some pages may contain other statement types — IGNORE them.

If {hint} spans multiple pages, merge all line items into ONE object.

Return STRICT JSON — a single-element array:
[
  {{
    "statement_type": "{stmt_type}",
    "source_pages": [1],
    "currency": "KWD",
    "unit_scale": 1,
    "periods": [
      {{"label": "2024-12-31", "col_name": "2024"}},
      {{"label": "2023-12-31", "col_name": "2023"}}
    ],
    "items": [
      {{
        "label_raw": "Cash and bank balances",
        "key": "cash_and_bank_balances",
        "values": {{"2024-12-31": 67007011, "2023-12-31": 74286447}},
        "is_total": false
      }}
    ]
  }}
]

RULES:
- "values" must contain numbers or null — NEVER strings.
- Parentheses (1,234) → −1234. Dash or blank → null.
- Detect unit_scale from headers (KD'000 → 1000, millions → 1000000).
- Period labels → ISO dates. is_total=true for totals.
- EXTRACT EVERY line item. Copy numbers EXACTLY.
- Return ONLY the JSON array. No markdown, no explanation.
- If this statement type is NOT found, return: []
"""


def _build_cashflow_extraction_prompt(
    n_pages: int,
    existing_codes: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Dedicated prompt for cash flow statement extraction.

    Uses stricter rules than the generic prompt:
    - Preserves row order exactly
    - Does not calculate missing values
    - Does not convert text headers into zero
    - Does not merge duplicate labels
    - Does not infer totals not explicitly shown
    - Preserves negative signs exactly
    """
    prompt = f"""\
### ROLE
You are extracting a statement of cash flows from a company financial report.
You must extract the cash flow statement ONLY — ignore all other statement types.

### OBJECTIVE
Extract every row from the cash flow statement across {n_pages} page(s).
Return valid JSON only. Preserve row order exactly as it appears in the document.

### EXTRACTION RULES (MANDATORY)
1. **ROW ORDER**: Output rows in the exact top-to-bottom order they appear
   in the document. Do NOT re-sort or regroup.
2. **DO NOT CALCULATE**: Never compute missing values. If a value is not
   explicitly shown, use null.
3. **DO NOT CONVERT HEADERS TO ZERO**: If a row is a section header
   (e.g. "Cash flows from operating activities"), its numeric values must
   be null — never convert them to 0.
4. **DO NOT MERGE DUPLICATES**: If the same label appears twice (e.g.
   "Net cash from operating activities" as both a section subtotal and a
   bridge line), extract BOTH rows with their respective values.
5. **DO NOT INFER TOTALS**: Only extract totals that are explicitly printed
   in the document. Never calculate a total from sub-items.
6. **PRESERVE SIGNS EXACTLY**: Parentheses (1,234) means NEGATIVE → −1234.
   If a value is printed without parentheses, it is positive.
   Do not flip signs based on your understanding of the item.
7. **NOTES COLUMN**: Completely ignore any Notes/footnote reference column.
   Small integers (1-30) next to a label are footnote numbers, not amounts.

### SECTION CLASSIFICATION
For each row, identify which section it belongs to:
- **operating**: Items under "Cash flows from operating activities"
  (depreciation, working capital changes, tax paid, interest paid/received, etc.)
- **investing**: Items under "Cash flows from investing activities"
  (property purchases, disposals, investments, deposits, etc.)
- **financing**: Items under "Cash flows from financing activities"
  (borrowings, repayments, dividends paid, share issuance, lease payments, etc.)
- **cash_bridge**: Items that reconcile cash balances
  (net increase/decrease in cash, cash at beginning/end of period,
   effect of exchange rate changes)

### ROW KIND CLASSIFICATION
For each row, identify its kind:
- **header**: Section heading text only, no numeric values
  (e.g. "Cash flows from operating activities")
- **item**: A regular line item with financial amounts
- **subtotal**: A section's total line (e.g. "Net cash from operating activities")
- **total**: The grand total / net change line

### OUTPUT FORMAT
Return ONLY a JSON array (no markdown fences, no commentary):

[
  {{
    "statement_type": "cash_flow",
    "source_pages": [1, 2],
    "currency": "SAR",
    "unit_scale": 1,
    "periods": [
      {{"label": "2024-12-31", "col_name": "2024"}},
      {{"label": "2023-12-31", "col_name": "2023"}}
    ],
    "items": [
      {{
        "label_raw": "Cash flows from operating activities",
        "key": "cash_flows_from_operating_activities",
        "section": "operating",
        "row_kind": "header",
        "values": {{"2024-12-31": null, "2023-12-31": null}},
        "is_total": false
      }},
      {{
        "label_raw": "Profit for the year",
        "key": "net_income",
        "section": "operating",
        "row_kind": "item",
        "values": {{"2024-12-31": 5340000, "2023-12-31": 4890000}},
        "is_total": false
      }},
      {{
        "label_raw": "Net cash from operating activities",
        "key": "cash_from_operations",
        "section": "operating",
        "row_kind": "subtotal",
        "values": {{"2024-12-31": 8200000, "2023-12-31": 7100000}},
        "is_total": true
      }},
      {{
        "label_raw": "Net increase in cash and cash equivalents",
        "key": "net_change_in_cash",
        "section": "cash_bridge",
        "row_kind": "total",
        "values": {{"2024-12-31": 1500000, "2023-12-31": 900000}},
        "is_total": true
      }},
      {{
        "label_raw": "Cash and cash equivalents at beginning of year",
        "key": "beginning_cash",
        "section": "cash_bridge",
        "row_kind": "item",
        "values": {{"2024-12-31": 10000000, "2023-12-31": 9100000}},
        "is_total": false
      }},
      {{
        "label_raw": "Cash and cash equivalents at end of year",
        "key": "ending_cash",
        "section": "cash_bridge",
        "row_kind": "item",
        "values": {{"2024-12-31": 11500000, "2023-12-31": 10000000}},
        "is_total": false
      }}
    ],
    "audit": {{
      "checks_performed": [],
      "corrections_made": [],
      "audit_notes": ""
    }}
  }}
]

### ABSOLUTE RULES
• "values" must contain numbers or null — NEVER strings.
• Parentheses (1,234) → −1234. Dash or blank → null.
• Detect unit_scale from headers (KD'000 → 1000, millions → 1000000).
• Period labels → ISO dates: "31 December 2024" → "2024-12-31".
• is_total=true ONLY for subtotals and totals. Headers are NOT totals.
• Copy every number EXACTLY as printed — do NOT round.
• Zero values → include as 0. Missing values → null.
• Return ONE object with ALL periods merged if CF spans multiple pages.
• Each item's "values" must cover ALL periods.
• Include "section" and "row_kind" for every item.
"""

    if existing_codes:
        cf_codes = [c for c in existing_codes if c.get("type") == "cashflow"]
        if cf_codes:
            codes_block = "\n".join(
                f'  - key: "{c["code"]}"  label: "{c["name"]}"'
                for c in cf_codes
            )
            prompt += f"""
### EXISTING CASH FLOW KEYS (MANDATORY REUSE)
This stock already has cash flow data with these exact keys.
Reuse the EXACT same "key" for matching concepts:

{codes_block}

If the new PDF has the same concept with different wording, use the existing key.
Only create new keys for genuinely new line items.
"""

    return prompt


async def _targeted_extract(
    api_key: str,
    stmt_type: str,
    page_images: List[bytes],
    model_name: str = "gemini-2.5-flash",
) -> List[ExtractedStatement]:
    """Extract a single statement type from page images (fallback)."""
    prompt = _build_targeted_prompt(stmt_type, len(page_images))
    try:
        raw_text = await _call_gemini(api_key, prompt, page_images, model_name)
        raw_json = _unwrap_to_statement_list(_parse_ai_json(raw_text))
        if not raw_json:
            return []
        stmts = _raw_to_statements(raw_json)
        stmts = _normalize_statements(stmts)
        logger.info(
            "Targeted fallback extracted %s: %d items",
            stmt_type,
            sum(len(s.items) for s in stmts),
        )
        return stmts
    except Exception as exc:
        logger.error("Targeted extraction failed for %s: %s", stmt_type, exc)
        return []


async def _fallback_missing_types(
    api_key: str,
    existing_statements: List[ExtractedStatement],
    page_images: List[bytes],
    model_name: str = "gemini-2.5-flash",
) -> List[ExtractedStatement]:
    """Check for missing core statement types and extract them individually.

    Matches Streamlit's parallel fallback approach — any of the 3 core
    types (balance, income, cashflow) missing from batch extraction gets
    a targeted re-extraction.
    """
    import asyncio

    found_types = {s.statement_type for s in existing_statements}
    missing = _VALID_EXTRACT_TYPES - found_types

    if not missing:
        return existing_statements

    logger.info(
        "Missing from batch: %s — running targeted fallback",
        missing,
    )

    tasks = [
        _targeted_extract(api_key, st, page_images, model_name)
        for st in sorted(missing)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_stmts = list(existing_statements)
    for result in results:
        if isinstance(result, list):
            all_stmts.extend(result)
        elif isinstance(result, Exception):
            logger.error("Fallback task failed: %s", result)

    return all_stmts


# ════════════════════════════════════════════════════════════════════
# GEMINI API CALL
# ════════════════════════════════════════════════════════════════════

async def _call_gemini(
    api_key: str,
    prompt: str,
    page_images: List[bytes],
    model_name: str = "gemini-2.5-flash",
) -> str:
    """Send images + prompt to Gemini and return raw text response.

    Uses response_mime_type='application/json' to force clean JSON output.
    Model fallback and rate-limit retry matching Streamlit's approach.
    """
    import asyncio

    from google import genai
    from google.genai import types
    from PIL import Image

    client = genai.Client(api_key=api_key)

    parts: list = [prompt]
    for png in page_images:
        parts.append(Image.open(io.BytesIO(png)))

    models_to_try = (
        [model_name] if model_name not in MODEL_FALLBACK_ORDER
        else MODEL_FALLBACK_ORDER
    )

    last_error: Exception | None = None

    for m in models_to_try:
        for attempt in range(1, API_MAX_RETRIES + 1):
            try:
                response = await client.aio.models.generate_content(
                    model=m,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=65536,
                        response_mime_type="application/json",
                        http_options=types.HttpOptions(timeout=180_000),
                    ),
                )
                if not response.text:
                    raise ValueError(
                        f"Empty response from {m} "
                        f"(finish_reason="
                        f"{getattr(response.candidates[0], 'finish_reason', '?')})"
                    )
                return response.text

            except Exception as exc:
                last_error = exc
                err = str(exc).lower()

                # Model not found → skip to next model
                if any(s in err for s in ("404", "not found", "does not exist")):
                    logger.warning("Model %s unavailable — skipping", m)
                    break

                # Rate limit → wait and retry
                if any(s in err for s in ("429", "quota", "rate limit", "resource has been exhausted")):
                    wait = RATE_LIMIT_DELAY * attempt
                    logger.warning("Rate limit on %s, waiting %ds…", m, wait)
                    await asyncio.sleep(wait)
                else:
                    await asyncio.sleep(2 ** attempt)

        await asyncio.sleep(2)

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


# ════════════════════════════════════════════════════════════════════
# RAW JSON → TYPED DATACLASSES
# ════════════════════════════════════════════════════════════════════

def _fuzzy_statement_type(raw_type: str) -> Optional[str]:
    """Fuzzy-match a statement type string to a canonical type.

    Handles long descriptive names the AI may return that aren't
    in _TYPE_MAP (e.g. "Consolidated Statement of Financial Position
    as at 31 December 2023").
    """
    t = raw_type.lower().strip()

    # Exact map check first
    mapped = _TYPE_MAP.get(raw_type) or _TYPE_MAP.get(t)
    if mapped:
        return mapped

    # Keyword-based fallback
    if any(k in t for k in ("cash flow", "cash_flow", "cashflow")):
        return "cashflow"
    if any(k in t for k in ("financial position", "balance sheet")):
        return "balance"
    if any(k in t for k in (
        "profit or loss", "profit and loss", "income statement",
        "statement of income", "comprehensive income",
    )):
        return "income"
    if any(k in t for k in ("changes in equity", "shareholders", "equity")):
        return "equity"

    return None


def _raw_to_statements(raw: list) -> List[ExtractedStatement]:
    """Convert parsed JSON dicts to typed ExtractedStatement objects.

    Handles both the expected flat format (top-level items + periods)
    and a legacy nested format where line_items are embedded inside
    each period dict.
    """
    stmts: List[ExtractedStatement] = []
    for entry in raw:
        raw_type = entry.get("statement_type", "").strip()
        st_type = _fuzzy_statement_type(raw_type)
        if st_type not in ("income", "balance", "cashflow", "equity"):
            logger.warning("Skipping unknown type: %r (no fuzzy match)", raw_type)
            continue

        top_items = entry.get("items", [])
        periods_raw = entry.get("periods", [])

        # ── Legacy nested format: line_items embedded inside periods ──
        if not top_items and periods_raw and isinstance(periods_raw[0], dict) and "line_items" in periods_raw[0]:
            merged_values: Dict[str, Dict[str, Optional[float]]] = {}
            item_meta: Dict[str, dict] = {}
            order_counter = 0
            normalised_periods = []
            for p in periods_raw:
                plabel = p.get("label", p.get("period_end", ""))
                normalised_periods.append({"label": plabel, "col_name": plabel})
                for li in p.get("line_items", []):
                    raw_key = li.get("name", "UNKNOWN")
                    key = raw_key.lower().replace(" ", "_").replace("'", "").replace('"', '')
                    if key not in merged_values:
                        merged_values[key] = {}
                        order_counter += 1
                        item_meta[key] = {
                            "label_raw": raw_key,
                            "is_total": bool(li.get("is_total", False)),
                            "order_index": order_counter,
                        }
                    merged_values[key][plabel] = _safe_float(li.get("value"))
            items = [
                ExtractedLineItem(
                    key=k, label_raw=item_meta[k]["label_raw"],
                    values=v, is_total=item_meta[k]["is_total"],
                    order_index=item_meta[k]["order_index"],
                )
                for k, v in merged_values.items()
            ]
            stmts.append(ExtractedStatement(
                statement_type=st_type,
                source_pages=entry.get("source_pages", []),
                currency=entry.get("currency", "USD"),
                unit_scale=entry.get("unit_scale", 1),
                periods=normalised_periods,
                items=items,
            ))
        else:
            # ── Standard flat format ──
            items = []
            for idx, it in enumerate(top_items, 1):
                items.append(ExtractedLineItem(
                    key=it.get("key", it.get("label_raw", "UNKNOWN")),
                    label_raw=it.get("label_raw", it.get("key", "")),
                    values=_get_safe_values(it.get("values", {})),
                    is_total=bool(it.get("is_total", False)),
                    order_index=idx,
                ))
            stmts.append(ExtractedStatement(
                statement_type=st_type,
                source_pages=entry.get("source_pages", []),
                currency=entry.get("currency", "USD"),
                unit_scale=entry.get("unit_scale", 1),
                periods=periods_raw,
                items=items,
            ))
    return stmts


def _normalize_label_key(label_raw: str, stmt_type: str) -> str:
    """Map a raw label to a canonical key using fuzzy matching.

    Matches Streamlit's normalization approach: exact match → partial
    contains match → snake_case fallback.
    """
    label_map = _LABEL_MAPS.get(stmt_type, {})
    label_lower = label_raw.strip().lower()

    # Exact match
    if label_lower in label_map:
        return label_map[label_lower]

    # Partial / contains match
    for pattern, canonical in label_map.items():
        if pattern in label_lower or label_lower in pattern:
            return canonical

    # Fallback: snake_case of raw label
    slug = re.sub(r"[^a-z0-9]+", "_", label_lower).strip("_")
    return slug or "unknown_item"


def _normalize_statements(stmts: List[ExtractedStatement]) -> List[ExtractedStatement]:
    """Normalize all item keys using label maps.

    If the AI returned a key that looks like raw text (contains spaces
    or isn't in _MERGE_CODES), replace it with the canonical key from
    the label map.
    """
    for stmt in stmts:
        for item in stmt.items:
            existing_key = item.key.lower().replace(" ", "_")
            # Only normalize if key looks like raw label text (not canonical)
            if existing_key not in _MERGE_CODES and (
                " " in item.key or item.key != existing_key
            ):
                normalized = _normalize_label_key(item.label_raw, stmt.statement_type)
                if normalized and normalized != "unknown_item":
                    item.key = normalized
            elif existing_key in _MERGE_CODES:
                item.key = _MERGE_CODES[existing_key]
    return stmts


# ── Canonical merge codes — maps AI key variants to a single merge key ──
# This catches the many ways the AI might name the same line item across
# different pages/year-ranges of a multi-page financial report.
_MERGE_CODES: Dict[str, str] = {
    # Income
    "revenue": "revenue", "total_revenue": "revenue", "net_revenue": "revenue",
    "sales": "revenue", "net_sales": "revenue",
    "cost_of_revenue": "cost_of_revenue", "cost_of_sales": "cost_of_revenue",
    "cost_of_goods_sold": "cost_of_revenue", "cogs": "cost_of_revenue",
    "cost_of_operations": "cost_of_revenue",
    "gross_profit": "gross_profit",
    "operating_expenses": "operating_expenses",
    "total_operating_expenses": "operating_expenses",
    "general_and_administrative_expenses": "general_and_admin",
    "general_and_administrative_expens": "general_and_admin",
    "selling_general_administrative": "sga", "sga": "sga",
    "selling_general_and_administrative": "sga",
    "selling_expenses": "selling_expenses",
    "selling_and_distribution_expenses": "selling_expenses",
    "operating_income": "operating_income", "operating_profit": "operating_income",
    "income_from_operations": "operating_income",
    "interest_expense": "interest_expense", "finance_costs": "interest_expense",
    "finance_cost": "interest_expense", "finance_charges": "interest_expense",
    "other_income": "other_income", "other_income_expense": "other_income",
    "income_before_tax": "income_before_tax", "profit_before_tax": "income_before_tax",
    "income_tax": "income_tax", "income_tax_expense": "income_tax",
    "tax_expense": "income_tax", "taxation": "income_tax",
    "net_income": "net_income", "net_profit": "net_income",
    "profit_for_the_year": "net_income", "profit_for_the_period": "net_income",
    "profit_for_year": "net_income",
    "net_income_attributable_to_shareholders": "net_income",
    "eps_basic": "eps_basic", "basic_eps": "eps_basic",
    "basic_earnings_per_share": "eps_basic", "earnings_per_share_basic": "eps_basic",
    "basic_and_diluted_earnings_per_share": "eps_basic",
    "basic_and_diluted_earnings_per_sha": "eps_basic",
    "earnings_per_share": "eps_basic",
    "eps_diluted": "eps_diluted", "diluted_eps": "eps_diluted",
    "diluted_earnings_per_share": "eps_diluted",
    "ebitda": "ebitda",
    "depreciation_and_amortization": "depreciation_amortization",
    "depreciation_amortization": "depreciation_amortization",
    # Kuwait / GCC specific
    "contribution_to_kfas": "contribution_to_kfas",
    "contribution_to_kuwait_foundation_for_advancement_of_sciences": "contribution_to_kfas",
    "contribution_to_kuwait_foundation_for_the_advancement_of_sciences": "contribution_to_kfas",
    "profit_before_contribution_to_kfas": "profit_before_deductions",
    "profit_before_contribution_to_kuwait_foundation_for_advancement_of_sciences": "profit_before_deductions",
    "profit_before_contribution_to_kuwait_foundation_for_the_advancement_of_sciences": "profit_before_deductions",
    "national_labour_support_tax": "nlst", "nlst": "nlst",
    "national_labor_support_tax": "nlst",
    "zakat": "zakat",
    "directors_remuneration": "directors_remuneration",
    "directors_fees": "directors_remuneration",
    "board_of_directors_remuneration": "directors_remuneration",
    "share_of_profit_of_associates": "share_results_associates",
    "share_of_loss_of_associates": "share_results_associates",
    "share_of_results_of_associates": "share_results_associates",
    "share_of_profit_loss_of_associates": "share_results_associates",
    # Balance sheet
    "cash_and_cash_equivalents": "cash", "cash_equivalents": "cash",
    "cash_and_bank_balances": "cash", "cash_and_balances_with_banks": "cash",
    "accounts_receivable": "accounts_receivable", "trade_receivables": "accounts_receivable",
    "receivables": "accounts_receivable", "trade_and_other_receivables": "accounts_receivable",
    "inventory": "inventory", "inventories": "inventory",
    "other_current_assets": "other_current_assets",
    "total_current_assets": "total_current_assets",
    "property_plant_equipment": "ppe_net", "property_plant_and_equipment": "ppe_net",
    "ppe_net": "ppe_net", "fixed_assets": "ppe_net",
    "goodwill": "goodwill",
    "intangible_assets": "intangible_assets", "intangibles": "intangible_assets",
    "total_non_current_assets": "total_non_current_assets",
    "total_assets": "total_assets",
    "accounts_payable": "accounts_payable", "trade_payables": "accounts_payable",
    "trade_and_other_payables": "accounts_payable",
    "short_term_debt": "short_term_debt", "current_portion_of_debt": "short_term_debt",
    "short_term_borrowings": "short_term_debt",
    "total_current_liabilities": "total_current_liabilities",
    "long_term_debt": "long_term_debt", "long_term_borrowings": "long_term_debt",
    "non_current_borrowings": "long_term_debt",
    "total_non_current_liabilities": "total_non_current_liabilities",
    "total_liabilities": "total_liabilities",
    "common_stock": "share_capital", "share_capital": "share_capital",
    "issued_capital": "share_capital",
    "retained_earnings": "retained_earnings",
    "share_premium": "share_premium",
    "statutory_reserve": "statutory_reserve",
    "voluntary_reserve": "voluntary_reserve",
    "general_reserve": "general_reserve",
    "treasury_shares": "treasury_shares",
    "total_equity": "total_equity", "total_shareholders_equity": "total_equity",
    "total_stockholders_equity": "total_equity",
    "equity_attributable_to_shareholders": "total_equity",
    "total_liabilities_and_equity": "total_liabilities_and_equity",
    "total_liabilities_and_shareholders_equity": "total_liabilities_and_equity",
    # Cash flow
    "cash_from_operations": "cash_from_operations",
    "cash_from_operating_activities": "cash_from_operations",
    "net_cash_from_operating_activities": "cash_from_operations",
    "net_cash_used_in_operating_activities": "cash_from_operations",
    "capital_expenditures": "capital_expenditures", "capex": "capital_expenditures",
    "purchase_of_property_plant_equipment": "capital_expenditures",
    "purchase_of_property_plant_and_equipment": "capital_expenditures",
    "other_investing_activities": "other_investing",
    "cash_from_investing": "cash_from_investing",
    "cash_from_investing_activities": "cash_from_investing",
    "net_cash_from_investing_activities": "cash_from_investing",
    "net_cash_used_in_investing_activities": "cash_from_investing",
    "debt_issued": "debt_issued", "proceeds_from_borrowings": "debt_issued",
    "debt_repaid": "debt_repaid", "repayment_of_borrowings": "debt_repaid",
    "dividends_paid": "dividends_paid", "dividend_paid": "dividends_paid",
    "cash_from_financing": "cash_from_financing",
    "cash_from_financing_activities": "cash_from_financing",
    "net_cash_from_financing_activities": "cash_from_financing",
    "net_cash_used_in_financing_activities": "cash_from_financing",
    "net_change_in_cash": "net_change_in_cash",
    "net_increase_decrease_in_cash": "net_change_in_cash",
    "changes_in_working_capital": "changes_in_working_capital",
}


def _get_merge_key(raw_key: str) -> str:
    """Get a canonical merge key for matching items across pages/year-ranges."""
    k = raw_key.strip().lower().replace(" ", "_").replace("-", "_")
    k = re.sub(r"_+", "_", k).strip("_")
    if k in _MERGE_CODES:
        return _MERGE_CODES[k]
    for suffix in ("_total", "_net", "_and_equivalents"):
        if k.endswith(suffix):
            trimmed = k[: -len(suffix)]
            if trimmed in _MERGE_CODES:
                return _MERGE_CODES[trimmed]
    return k


def _merge_same_type_statements(
    stmts: List[ExtractedStatement],
) -> List[ExtractedStatement]:
    """
    Consolidate multiple ExtractedStatement objects of the same type
    using Anchored Visual Ordering.

    The "Master Statement" (containing the most recent date) defines
    the canonical row sequence — preserving the document's own visual
    structure.  Items only found in older reports are inserted between
    their nearest visual neighbours.
    """
    from collections import OrderedDict

    groups: Dict[str, List[ExtractedStatement]] = {}
    for s in stmts:
        groups.setdefault(s.statement_type, []).append(s)

    merged: List[ExtractedStatement] = []
    for st_type, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        logger.info(
            "Merging %d '%s' statement objects into one",
            len(group), st_type,
        )

        # ── FIND THE ANCHOR: statement with the latest period date ──
        def _max_period_label(stmt: ExtractedStatement) -> str:
            return max(
                (p.get("label", "") for p in stmt.periods),
                default="",
            )

        anchor_stmt = max(group, key=_max_period_label)
        logger.info(
            "Anchor statement for '%s': latest period = %s",
            st_type, _max_period_label(anchor_stmt),
        )

        # ── Merge periods (unique by label, newest first) ──
        seen_labels: set = set()
        all_periods: list = []
        for g in sorted(group, key=_max_period_label, reverse=True):
            for p in g.periods:
                lbl = p.get("label", "")
                if lbl not in seen_labels:
                    seen_labels.add(lbl)
                    all_periods.append(p)

        # ── LOCK ORDER from anchor statement ──
        items_by_key: OrderedDict[str, ExtractedLineItem] = OrderedDict()
        for idx, item in enumerate(anchor_stmt.items):
            mk = _get_merge_key(item.key)
            item.order_index = idx
            item.visual_anchor_score = float(idx)
            items_by_key[mk] = ExtractedLineItem(
                key=item.key,
                label_raw=item.label_raw,
                values=dict(item.values),
                is_total=item.is_total,
                order_index=idx,
                visual_anchor_score=float(idx),
            )

        # ── MERGE HISTORICAL DATA into anchor structure ──
        for g in group:
            if g is anchor_stmt:
                continue
            for item in g.items:
                mk = _get_merge_key(item.key)

                # Prefix-based fuzzy match
                matched_mk = mk if mk in items_by_key else None
                if matched_mk is None:
                    for existing_mk in items_by_key:
                        shorter, longer = sorted(
                            [mk, existing_mk], key=len,
                        )
                        if len(shorter) >= 15 and longer.startswith(shorter):
                            matched_mk = existing_mk
                            logger.debug(
                                "Fuzzy-matched '%s' → '%s'", mk, existing_mk,
                            )
                            break

                if matched_mk is not None:
                    # Item exists in anchor — fill missing periods
                    existing = items_by_key[matched_mk]
                    safe_vals = _get_safe_values(item.values)
                    for period_key, val in safe_vals.items():
                        if period_key not in existing.values or existing.values[period_key] is None:
                            existing.values[period_key] = val
                    if item.is_total:
                        existing.is_total = True
                else:
                    # NEW item only in older report — insertional logic:
                    # Place after all anchor items using offset of 500 +
                    # its original position so relative order among
                    # historical-only items is preserved.
                    item.order_index = 500 + item.order_index
                    item.visual_anchor_score = 500.0 + float(item.order_index)
                    items_by_key[mk] = ExtractedLineItem(
                        key=item.key,
                        label_raw=item.label_raw,
                        values=dict(item.values),
                        is_total=item.is_total,
                        order_index=item.order_index,
                        visual_anchor_score=item.visual_anchor_score,
                    )

        # Sort by order_index (anchor items 0..N, historical-only 500+)
        final_items = sorted(items_by_key.values(), key=lambda x: x.order_index)

        # Re-index sequentially for the frontend
        for i, it in enumerate(final_items, 1):
            it.order_index = i

        # Merge source_pages
        all_pages = sorted({p for g in group for p in g.source_pages})

        merged.append(ExtractedStatement(
            statement_type=st_type,
            source_pages=all_pages,
            currency=anchor_stmt.currency,
            unit_scale=anchor_stmt.unit_scale,
            periods=all_periods,
            items=final_items,
        ))

        logger.info(
            "Merged '%s': %d periods, %d items (anchor had %d)",
            st_type, len(all_periods), len(final_items),
            len(anchor_stmt.items),
        )

    return merged


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
                        "visual_anchor_score": it.visual_anchor_score,
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
        top_items = s.get("items", [])
        periods_raw = s.get("periods", [])

        # ── Handle legacy cache format where line_items are embedded
        #    inside each period dict instead of a top-level "items" list.
        if not top_items and periods_raw and "line_items" in periods_raw[0]:
            merged_values: Dict[str, Dict[str, Optional[float]]] = {}
            item_meta: Dict[str, dict] = {}
            order_counter = 0
            normalised_periods = []
            for p in periods_raw:
                plabel = p.get("label", p.get("period_end", ""))
                normalised_periods.append({
                    "label": plabel,
                    "col_name": plabel,
                })
                for li in p.get("line_items", []):
                    raw_key = li.get("name", "UNKNOWN")
                    key = raw_key.lower().replace(" ", "_").replace("'", "").replace('"', '')
                    if key not in merged_values:
                        merged_values[key] = {}
                        order_counter += 1
                        item_meta[key] = {
                            "label_raw": raw_key,
                            "is_total": bool(li.get("is_total", False)),
                            "order_index": order_counter,
                        }
                    merged_values[key][plabel] = _safe_float(li.get("value"))
            items = [
                ExtractedLineItem(
                    key=k,
                    label_raw=item_meta[k]["label_raw"],
                    values=v,
                    is_total=item_meta[k]["is_total"],
                    order_index=item_meta[k]["order_index"],
                )
                for k, v in merged_values.items()
            ]
            stmts.append(ExtractedStatement(
                statement_type=s["statement_type"],
                source_pages=s.get("source_pages", []),
                currency=s.get("currency", "USD"),
                unit_scale=s.get("unit_scale", 1),
                periods=normalised_periods,
                items=items,
            ))
        else:
            items = [
                ExtractedLineItem(
                    key=it["key"], label_raw=it["label_raw"],
                    values=it["values"], is_total=it["is_total"],
                    order_index=it.get("order_index", 0),
                    visual_anchor_score=it.get("visual_anchor_score", 0.0),
                )
                for it in top_items
            ]
            stmts.append(ExtractedStatement(
                statement_type=s["statement_type"],
                source_pages=s.get("source_pages", []),
                currency=s.get("currency", "USD"),
                unit_scale=s.get("unit_scale", 1),
                periods=periods_raw,
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
# CASH FLOW SPECIFIC EXTRACTION
# ════════════════════════════════════════════════════════════════════

async def extract_cashflow_to_stage(
    pdf_bytes: bytes,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    existing_codes: Optional[List[Dict[str, str]]] = None,
) -> List[ExtractedStatement]:
    """Extract ONLY the cash flow statement using dedicated page detection
    and a cash-flow-specific prompt.

    Steps:
    1. Detect cash flow pages via keyword heuristics
    2. Convert only those pages to images
    3. Send to Gemini with the dedicated CF prompt
    4. Parse and return statements (no caching, no arithmetic retry)

    The caller (persist layer) handles staging and reconciliation.
    Falls back to all pages if page detection finds nothing.
    """
    from app.services.cashflow_reconciler import detect_cashflow_pages

    # Step 1: Detect cash flow pages
    cf_page_indices = detect_cashflow_pages(pdf_bytes)

    # Step 2: Convert to images
    all_images = pdf_to_images(pdf_bytes)
    if not all_images:
        raise ValueError("PDF has no pages.")

    if cf_page_indices:
        # Use only CF pages
        cf_images = [
            all_images[i] for i in cf_page_indices
            if i < len(all_images)
        ]
        if not cf_images:
            cf_images = all_images  # fallback
        logger.info(
            "Cash flow extraction: using %d detected CF pages out of %d total",
            len(cf_images), len(all_images),
        )
    else:
        # Fallback: use all pages
        cf_images = all_images
        logger.warning(
            "Cash flow page detection found no CF pages — using all %d pages",
            len(all_images),
        )

    # Step 3: Build CF-specific prompt and extract
    prompt = _build_cashflow_extraction_prompt(len(cf_images), existing_codes)
    raw_text = await _call_gemini(api_key, prompt, cf_images, model_name)

    logger.info("CF extraction raw response: %d chars", len(raw_text))
    raw_json = _unwrap_to_statement_list(_parse_ai_json(raw_text))

    if not raw_json:
        logger.warning("CF-specific extraction returned no results")
        return []

    # Step 4: Parse into statements
    statements = _raw_to_statements(raw_json)
    statements = _normalize_statements(statements)

    # Only keep cashflow type
    cf_stmts = [s for s in statements if s.statement_type == "cashflow"]

    # Apply unit_scale
    for stmt in cf_stmts:
        if stmt.unit_scale and stmt.unit_scale != 1:
            for item in stmt.items:
                item.values = {
                    k: (v * stmt.unit_scale if v is not None else None)
                    for k, v in item.values.items()
                }

    logger.info(
        "CF extraction complete: %d cashflow statements, %d items",
        len(cf_stmts),
        sum(len(s.items) for s in cf_stmts),
    )

    return cf_stmts


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
    existing_codes: Optional[List[Dict[str, str]]] = None,
) -> ExtractionResult:
    """
    Full self-reflective extraction pipeline.

    1. Hash PDF → check cache
    2. Convert PDF → 200 DPI images
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
    page_images = pdf_to_images(pdf_bytes)
    if not page_images:
        raise ValueError("PDF has no pages.")
    _cache_images(h, page_images)  # cache for validation step

    logger.info(
        "Extraction pipeline: %s (%d pages, %.1f KB)",
        filename, len(page_images), len(pdf_bytes) / 1024,
    )

    # ── Step 3: First extraction pass ────────────────────────────────
    prompt = _build_extraction_prompt(len(page_images), existing_codes=existing_codes)
    raw_text = await _call_gemini(api_key, prompt, page_images, model_name)
    logger.info("Raw AI response length: %d chars", len(raw_text))
    logger.debug("Raw AI response (first 500): %s", raw_text[:500])
    raw_json = _unwrap_to_statement_list(_parse_ai_json(raw_text))

    # Preserve first-pass raw text for debugging zero-statement results
    _first_pass_raw_text = raw_text

    if not raw_json:
        raise ValueError("AI did not detect any financial statements.")

    logger.info(
        "Parsed %d JSON entries. Types: %s",
        len(raw_json),
        [e.get('statement_type', '?') for e in raw_json],
    )
    for idx, entry in enumerate(raw_json):
        logger.info(
            "  Entry %d: type=%s, periods=%d, items=%d",
            idx, entry.get('statement_type', '?'),
            len(entry.get('periods', [])), len(entry.get('items', [])),
        )

    statements = _raw_to_statements(raw_json)
    statements = _normalize_statements(statements)
    statements = _merge_same_type_statements(statements)

    if not statements and raw_json:
        raw_types = [e.get('statement_type', '?') for e in raw_json]
        raise ValueError(
            f"AI returned {len(raw_json)} entries with types {raw_types}, "
            f"but none matched known statement types "
            f"(income, balance, cashflow, equity). "
            f"Check the PDF contains standard financial statements."
        )

    logger.info(
        "After _raw_to_statements: %d statements",
        len(statements),
    )
    for stmt in statements:
        logger.info(
            "  stmt type=%s  periods=%d  items=%d",
            stmt.statement_type, len(stmt.periods), len(stmt.items),
        )

    # ── Step 3b: Fallback for missing statement types ────────────────
    statements = await _fallback_missing_types(
        api_key, statements, page_images, model_name,
    )
    statements = _merge_same_type_statements(statements)

    # ── Step 3c: Re-extract incomplete statement types ───────────────
    MIN_ITEMS = {"balance": 10, "cashflow": 8, "income": 8, "equity": 4}
    incomplete_types = [
        s.statement_type for s in statements
        if len(s.items) < MIN_ITEMS.get(s.statement_type, 8)
    ]
    if incomplete_types:
        import asyncio as _aio
        logger.warning(
            "Incomplete extraction detected for %s — re-extracting",
            incomplete_types,
        )
        re_tasks = [
            _targeted_extract(api_key, st, page_images, model_name)
            for st in incomplete_types
        ]
        re_results = await _aio.gather(*re_tasks, return_exceptions=True)
        for st_type, result in zip(incomplete_types, re_results):
            if isinstance(result, list) and result:
                new_count = sum(len(s.items) for s in result)
                old_count = next(
                    (len(s.items) for s in statements if s.statement_type == st_type), 0
                )
                if new_count > old_count:
                    logger.info(
                        "Re-extraction improved %s: %d → %d items",
                        st_type, old_count, new_count,
                    )
                    statements = [
                        s for s in statements if s.statement_type != st_type
                    ] + result
                else:
                    logger.info(
                        "Re-extraction did not improve %s (%d vs %d) — keeping original",
                        st_type, new_count, old_count,
                    )
            elif isinstance(result, Exception):
                logger.error("Re-extraction failed for %s: %s", st_type, result)
        statements = _merge_same_type_statements(statements)

    # ── Step 3d: Dedicated cash flow extraction ──────────────────────
    # Use the CF-specific prompt with page detection for better results.
    # Replace the generic cashflow extraction with the dedicated one.
    cf_existing = next(
        (s for s in statements if s.statement_type == "cashflow"), None,
    )
    cf_item_count = len(cf_existing.items) if cf_existing else 0

    try:
        cf_stmts = await extract_cashflow_to_stage(
            pdf_bytes=pdf_bytes,
            api_key=api_key,
            model_name=model_name,
            existing_codes=existing_codes,
        )
        if cf_stmts:
            new_cf_count = sum(len(s.items) for s in cf_stmts)
            if new_cf_count >= cf_item_count:
                # Replace generic CF with dedicated CF extraction
                statements = [
                    s for s in statements if s.statement_type != "cashflow"
                ] + cf_stmts
                logger.info(
                    "Dedicated CF extraction replaced generic: %d → %d items",
                    cf_item_count, new_cf_count,
                )
            else:
                logger.info(
                    "Dedicated CF extraction had fewer items (%d vs %d) — keeping generic",
                    new_cf_count, cf_item_count,
                )
    except Exception as exc:
        logger.warning("Dedicated CF extraction failed (non-fatal): %s", exc)

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
        retry_json = _unwrap_to_statement_list(_parse_ai_json(retry_text))

        if retry_json:
            retry_stmts = _raw_to_statements(retry_json)
            retry_stmts = _normalize_statements(retry_stmts)
            retry_stmts = _merge_same_type_statements(retry_stmts)

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

    # ── Fallback: if extraction produced 0 statements, try cache ─────
    if not statements and use_cache:
        cached_fallback = _get_cached(stock_id, h)
        if cached_fallback and cached_fallback.statements:
            logger.warning(
                "AI returned 0 statements but valid cache exists — "
                "using cached result (%d statements)",
                len(cached_fallback.statements),
            )
            cached_fallback.retry_count = retry_count
            cached_fallback.raw_ai_text = _first_pass_raw_text
            return cached_fallback

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
        raw_ai_text=_first_pass_raw_text if not statements else "",
    )

    # ── Step 7: Cache (only if we have statements) ───────────────────
    if use_cache and statements:
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

    # Reuse cached images from extraction step, or re-render
    page_images = _get_cached_images(h)
    if not page_images:
        page_images = pdf_to_images(pdf_bytes)
        if not page_images:
            raise ValueError("PDF has no pages.")
        _cache_images(h, page_images)

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
### ROLE
Senior Financial Auditor performing PLACEMENT & CONSISTENCY AUDIT (Step 3 of 3).

You are the FINAL GATE before this data enters a financial database.
Think like a Big-4 auditor signing off on an engagement — every item must
be in the correct statement, with the correct key, and internally consistent.

### STEP 1: STATEMENT-TYPE PLACEMENT AUDIT
For EACH line item in the data below, verify it is under the correct statement:

  BALANCE SHEET ("balance"):
    ✓ Assets: cash, receivables, inventory, prepaid, PPE, goodwill, intangibles
    ✓ Liabilities: payables, accrued, borrowings, deferred tax, provisions
    ✓ Equity: share capital, retained earnings, reserves, minority interest

  INCOME STATEMENT ("income"):
    ✓ Revenue, cost of sales, gross profit
    ✓ Operating expenses (admin, selling, G&A, depreciation, amortization)
    ✓ Finance income/costs, tax, net profit/loss, EPS

  CASH FLOW STATEMENT ("cashflow"):
    ✓ Operating/investing/financing activities
    ✓ Purchase/proceeds of assets, repayments, issuances
    ✓ Net change in cash, opening/closing cash

  STATEMENT OF CHANGES IN EQUITY ("equity"):
    ✓ Movements in share capital, treasury shares
    ✓ Comprehensive income, dividends declared

  Common misplacements to catch:
    ✗ "Net income" placed in balance sheet → belongs in income
    ✗ "Depreciation" in balance sheet → belongs in income
    ✗ Operating cash items mixed into income statement
    ✗ "Retained earnings" in income → belongs in balance/equity

### STEP 2: KEY CONSISTENCY & NAMING AUDIT
Like an auditor chasing consistency across periods and statements:
  1. Is the key a faithful representation of the line item?
     e.g. "cash_and_bank_balances" not "other_assets" for cash
  2. Are similar items across different statements using CONSISTENT key patterns?
     e.g. "depreciation" key used consistently, not "depr" in one and "depreciation_expense" in another
  3. Would this key match what a portfolio tracker expects?
     e.g. "revenue" or "total_revenue" for top-line, "net_income" for bottom-line

### STEP 3: is_total FLAG AUDIT
  • Subtotals and totals ("Total Assets", "Total Liabilities", "Net Income",
    "Gross Profit", "Total Equity") → is_total = true
  • Individual line items → is_total = false
  • Watch for: "Profit before tax" (TOTAL), "Profit for the year" (TOTAL),
    "Total comprehensive income" (TOTAL)

### STEP 4: CROSS-STATEMENT CONSISTENCY (FINANCIAL AUDIT)
Chase consistency like an auditor would:
  1. "Retained earnings" ending value in equity statement should ≈ match
     "Retained earnings" on balance sheet for the same period.
  2. "Net income" in income statement should ≈ flow into the equity roll-forward.
  3. "Ending cash" in cash flow should ≈ match "Cash" on balance sheet.
  4. If same item appears in multiple statements, values must MATCH.

### STEP 5: DUPLICATE DETECTION
  • No item should appear in more than one statement type.
  • If an item appears in both balance and income → flag for removal/move.

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
      "reason": "This item represents cash balances per document label"
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
  "consistency_warnings": [
    "Retained earnings on balance sheet (45,000) does not match equity statement ending balance (43,500)"
  ],
  "notes": "Overall audit notes"
}}

═══ RULES ═══
• If everything is correct: {{"placement_correct": true, "corrections": [], "consistency_warnings": [], "notes": ""}}
• action types: "move", "rename_key", "fix_total", "remove_duplicate"
• Be conservative — only flag CLEAR mistakes, not ambiguous border cases
• Follow IFRS / standard accounting placement conventions
• Chase data consistency across statements — an auditor never lets mismatches slide
• If you find a cross-statement mismatch, add it to consistency_warnings
  AND produce a correction if you can determine which value is wrong
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


# ════════════════════════════════════════════════════════════════════
# AI ATTRIBUTION (Step 4 — user-triggered separate call)
# ════════════════════════════════════════════════════════════════════

def _get_latest_cached(stock_id: int) -> Tuple[Optional[ExtractionResult], Optional[str]]:
    """Return the most recent cached ExtractionResult + pdf_hash for a stock."""
    _ensure_cache_table()
    from app.core.database import query_one
    row = query_one(
        "SELECT pdf_hash, result_json, model_used, pages FROM extraction_cache "
        "WHERE stock_id = ? ORDER BY created_at DESC LIMIT 1",
        (stock_id,),
    )
    if not row:
        return None, None

    try:
        data = json.loads(row["result_json"])
        result = _dict_to_result(data)
        h = row["pdf_hash"]
        result.cached = True
        result.pdf_hash = h
        result.model_used = row["model_used"] or ""
        result.pages_processed = row["pages"] or 0
        return result, h
    except Exception:
        logger.warning("Corrupt cache entry for stock %d, ignoring", stock_id)
        return None, None


async def ai_attribute_extraction(
    stock_id: int,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> ExtractionResult:
    """
    AI Attribution Expert — user-triggered pass.

    1. Load the latest cached extraction for this stock
    2. Optionally load cached PDF images
    3. Send extracted data to AI for expert attribution review
    4. Apply corrections
    5. Update cache
    6. Return updated ExtractionResult
    """
    cached, h = _get_latest_cached(stock_id)
    if not cached or not h:
        raise ValueError("No cached extraction found. Upload a financial report first.")

    statements = cached.statements
    logger.info(
        "AI attribution: stock %d (%d statements, %d total items)",
        stock_id, len(statements), sum(len(s.items) for s in statements),
    )

    # Try to include PDF images for richer review (non-fatal if missing)
    page_images = _get_cached_images(h) or []

    statements, attribution_count = await _ai_attribution_pass(
        api_key, statements, page_images, model_name,
    )

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
        validation_corrections=attribution_count,
        placement_corrections=getattr(cached, 'placement_corrections', 0),
    )

    # Update cache
    try:
        _set_cache(stock_id, h, "", result)
    except Exception as exc:
        logger.warning("Failed to update cache after attribution: %s", exc)

    logger.info(
        "Attribution complete: %d corrections, confidence=%.1f%%",
        attribution_count, confidence * 100,
    )

    return result


# ════════════════════════════════════════════════════════════════════
# AI RECONCILE — audit names, sequence & values against stored PDF
# ════════════════════════════════════════════════════════════════════


def _build_reconcile_prompt(
    statement_type: str,
    db_periods: list[dict],
) -> str:
    """Prompt that asks AI to act as auditor: fix names, order AND values."""
    period_block = json.dumps(db_periods, indent=2, ensure_ascii=False)
    return f"""You are an Autonomous Financial Data Analyst AND meticulous auditor
performing a digit-by-digit reconciliation of an **{statement_type}** statement.

I am giving you HIGH-RESOLUTION images of the original PDF financial report
AND the values currently stored in our database.

CURRENT DATABASE STATE:
{period_block}

## ADAPTIVE EXTRACTION STRATEGY (apply when reading the PDF):

1. **Header Identification**: Locate "Consolidated Statement" (or similar)
   titles to identify which table you are looking at (Balance Sheet,
   Profit or Loss, Cash Flow). Every company has a different structure and
   row naming — adapt to whatever layout you see.

2. **Column Mapping**: Dynamically identify the year columns. Financial
   statements usually place the "Current Year" in the first numerical
   column and the "Comparative Year" in the second. Always map values
   to their specific year. IGNORE any "Notes" column — it contains
   small reference integers (1-30), not financial amounts.

3. **Parentheses & Signs**:
   - Numbers in parentheses `(x,xxx)` MUST be read as **negative** `-xxxx`
   - Dashes `-` or empty cells = **0**
   - Rows like "Share of profit/(loss)" — the sign can vary by year.

4. **Row Label Cleaning**: Extract the full text of the row label. Remove
   leading/trailing whitespace and footnote references. "Cash and bank
   balances 4" → the label is "Cash and bank balances", the "4" is a
   footnote reference to ignore.

5. **Structure Preservation**: Maintain the hierarchy of "Current" vs
   "Non-Current" sections. Items like "Lease liabilities" can appear
   TWICE — once under Current and once under Non-Current. They are
   DIFFERENT line items with different values. Capture sub-totals
   (e.g. "Total Assets") exactly as they appear for cross-verification.

## YOUR MANDATORY PROCESS (follow in order):

### Step 1 — READ the PDF first (ignore the database)
For every line item visible in the PDF, read the number **digit by digit**.
Pay extreme attention to:
- Commas vs periods (thousands separators vs decimals)
- Whether a number is in parentheses (negative) or not
- The exact column each number belongs to (which fiscal year)
- Small digits that could be confused: 0/6/8, 1/7, 3/8, 5/6
- Numbers that look similar: 364,013 vs 364,913
- **DIRECT PIXEL VERIFICATION**: Ensure each number is on the same
  horizontal line as its row label — do not misalign values across rows.

### Step 2 — COMPARE each PDF value against the database value
For every line item in every period, compare:
- The value you read from the PDF (Step 1)
- The value stored in the database (shown above)
If they differ by even 1, flag it as a correction.

### Step 3 — CHECK names and order
- Verify each `line_item_name` matches the PDF text exactly
- Verify the order_index matches the PDF's top-to-bottom sequence

### Step 4 — SELF-AUDIT (mandatory before output)
Perform arithmetic cross-checks on your PDF-read values:
- **Balance sheet**: Total assets == Total current + Total non-current assets.
  Total liabilities + Total equity == Total assets.
- **Income**: Revenue - Cost == Gross Profit. Check full profit chain.
- **Cash flow**: Operating + Investing + Financing ≈ Net change in cash.
If any check fails, re-read that specific area digit by digit before output.

### Step 5 — OUTPUT corrections

OUTPUT EXACTLY THIS JSON (include ONLY items that need a fix):
{{
  "name_corrections": [
    {{"line_item_code": "<CODE>", "correct_name": "<exact text from PDF>"}}
  ],
  "order_corrections": [
    {{"line_item_code": "<CODE>", "correct_order_index": <int>}}
  ],
  "value_corrections": [
    {{"line_item_code": "<CODE>", "corrections": {{"<period_end_date>": <number>}} }}
  ]
}}

CRITICAL RULES:
- For value_corrections, provide the EXACT number you read from the PDF.
  This may be a completely different number from the database — that is OK.
  You are correcting the DB to match the PDF.
- Read EVERY digit carefully. A single wrong digit means the value is wrong.
- Use EMPTY arrays for categories that need no fix.
- Names must be the exact text from the PDF — do not translate or abbreviate.
- order_index is 1-based, matching the PDF's visual top-to-bottom order.
- Respond with ONLY the JSON object. No markdown fences, no explanation."""


async def ai_rearrange_statement(
    stock_id: int,
    statement_type: str,
    api_key: str,
    periods: list[str] | None = None,
    pdf_id: int | None = None,
) -> dict:
    """AI-powered reconciliation: fix names, order AND values against stored PDF.

    Returns dict with corrections_applied, names_fixed, items_reordered,
    values_corrected, message, confidence.
    """
    from google import genai
    from app.core.database import query_df, query_one as _q1, get_connection

    # ── 1. Load current DB data (including names + order) ────────────
    where = "WHERE fs.stock_id = ? AND fs.statement_type = ?"
    params: list = [stock_id, statement_type]
    if periods:
        placeholders = ",".join("?" for _ in periods)
        where += f" AND fs.period_end_date IN ({placeholders})"
        params.extend(periods)

    df = query_df(
        f"""SELECT fs.id AS stmt_id, fs.period_end_date, fs.fiscal_year,
                   fli.line_item_code, fli.line_item_name, fli.amount,
                   fli.order_index, fli.id AS li_id
            FROM financial_statements fs
            JOIN financial_line_items fli ON fli.statement_id = fs.id
            {where}
            ORDER BY fs.period_end_date, fli.order_index""",
        tuple(params),
    )
    if df.empty:
        raise ValueError(f"No {statement_type} statements found for stock {stock_id}")

    # ── 2. Build structured prompt data WITH names + order ───────────
    db_periods: list[dict] = []
    for ped, grp in df.groupby("period_end_date"):
        items = []
        for _, row in grp.iterrows():
            items.append({
                "line_item_code": row["line_item_code"],
                "line_item_name": row["line_item_name"],
                "amount": row["amount"],
                "order_index": int(row["order_index"]),
            })
        db_periods.append({
            "period_end_date": ped,
            "fiscal_year": int(grp.iloc[0]["fiscal_year"]),
            "items": items,
        })

    prompt = _build_reconcile_prompt(statement_type, db_periods)

    # ── 3. Load stored PDF images (required for audit) ───────────────
    parts: list = []
    if pdf_id is not None:
        pdf_row = _q1(
            "SELECT filename FROM pdf_uploads WHERE id = ? AND stock_id = ?",
            (pdf_id, stock_id),
        )
        if pdf_row:
            from pathlib import Path as _Path
            from PIL import Image as _Image
            disk_name = pdf_row["filename"] if isinstance(pdf_row, dict) else pdf_row[0]
            pdf_dir = _Path(__file__).resolve().parents[1] / "uploads" / "pdfs" / str(stock_id)
            pdf_path = pdf_dir / disk_name
            if pdf_path.is_file():
                pdf_bytes = pdf_path.read_bytes()
                try:
                    # Always render at high DPI (400) for reconcile — skip cache
                    images = pdf_to_images(pdf_bytes, dpi=400)
                    for png in images[:10]:
                        parts.append(_Image.open(io.BytesIO(png)))
                except Exception as e:
                    logger.warning("Could not load PDF images: %s", e)

    if not parts:
        raise ValueError(
            "A saved PDF is required for AI reconciliation. "
            "Please select a stored PDF so the AI can compare against it."
        )

    parts.append(prompt)

    # ── 4. Call Gemini 3× for multi-pass consensus ───────────────────
    import re as _re
    NUM_PASSES = 3

    def _parse_gemini_json(raw: str) -> dict:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = _re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = _re.sub(r"\s*```$", "", cleaned)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            m = _re.search(r"\{.*\}", cleaned, _re.DOTALL)
            obj = json.loads(m.group()) if m else {}
        return obj if isinstance(obj, dict) else {}

    client = genai.Client(api_key=api_key)
    all_results: list[dict] = []
    for pass_num in range(NUM_PASSES):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=parts,
            )
            parsed = _parse_gemini_json(resp.text or "{}")
            all_results.append(parsed)
            logger.info(
                "Reconcile pass %d/%d: %d name, %d order, %d value corrections",
                pass_num + 1, NUM_PASSES,
                len(parsed.get("name_corrections", [])),
                len(parsed.get("order_corrections", [])),
                len(parsed.get("value_corrections", [])),
            )
        except Exception as e:
            logger.warning("Reconcile pass %d failed: %s", pass_num + 1, e)

    if not all_results:
        raise ValueError("All AI reconciliation passes failed.")

    # ── 5. Build consensus (majority vote: 2-of-3 agreement) ────────
    from collections import Counter

    # 5a. Value corrections — key = (code, period), votes = [amount …]
    value_votes: dict[tuple, list] = {}
    for r in all_results:
        for vc in r.get("value_corrections", []):
            code = vc.get("line_item_code", "")
            for ped, val in vc.get("corrections", {}).items():
                if code and val is not None:
                    value_votes.setdefault((code, ped), []).append(float(val))

    consensus_values: list[dict] = []
    for (code, ped), vals in value_votes.items():
        rounded = [round(v) for v in vals]
        ctr = Counter(rounded)
        best_val, count = ctr.most_common(1)[0]
        if count >= 2:
            consensus_values.append({
                "line_item_code": code,
                "corrections": {ped: best_val},
            })

    # 5b. Name corrections — key = code, votes = [name …]
    name_votes: dict[str, list] = {}
    for r in all_results:
        for nc in r.get("name_corrections", []):
            code = nc.get("line_item_code", "")
            name = nc.get("correct_name", "")
            if code and name:
                name_votes.setdefault(code, []).append(name)

    consensus_names: list[dict] = []
    for code, names in name_votes.items():
        ctr = Counter(names)
        best_name, count = ctr.most_common(1)[0]
        if count >= 2:
            consensus_names.append({
                "line_item_code": code,
                "correct_name": best_name,
            })

    # 5c. Order corrections — key = code, votes = [index …]
    order_votes: dict[str, list] = {}
    for r in all_results:
        for oc in r.get("order_corrections", []):
            code = oc.get("line_item_code", "")
            idx = oc.get("correct_order_index")
            if code and idx is not None:
                order_votes.setdefault(code, []).append(int(idx))

    consensus_order: list[dict] = []
    for code, idxs in order_votes.items():
        ctr = Counter(idxs)
        best_idx, count = ctr.most_common(1)[0]
        if count >= 2:
            consensus_order.append({
                "line_item_code": code,
                "correct_order_index": best_idx,
            })

    result = {
        "name_corrections": consensus_names,
        "order_corrections": consensus_order,
        "value_corrections": consensus_values,
    }
    name_corrections = consensus_names
    order_corrections = consensus_order
    value_corrections = consensus_values

    logger.info(
        "Reconcile consensus (%d passes): %d name, %d order, %d value",
        len(all_results), len(name_corrections),
        len(order_corrections), len(value_corrections),
    )

    # ── 6. Build lookups ────────────────────────────────────────────
    # (period_end_date, line_item_code) → li_id  (for value fixes)
    li_lookup: dict = {}
    # line_item_code → [li_id, ...]  (for name/order fixes across all periods)
    code_to_ids: dict = {}
    for _, row in df.iterrows():
        li_lookup[(row["period_end_date"], row["line_item_code"])] = row["li_id"]
        code_to_ids.setdefault(row["line_item_code"], []).append(row["li_id"])

    # ── 7. Apply all corrections ────────────────────────────────────
    applied_names = 0
    applied_order = 0
    applied_values = 0

    with get_connection() as conn:
        cur = conn.cursor()

        # Fix names
        for nc in name_corrections:
            code = nc.get("line_item_code", "")
            new_name = nc.get("correct_name", "")
            if code and new_name and code in code_to_ids:
                for li_id in code_to_ids[code]:
                    cur.execute(
                        "UPDATE financial_line_items SET line_item_name = ? WHERE id = ?",
                        (new_name, int(li_id)),
                    )
                applied_names += 1

        # Fix order
        for oc in order_corrections:
            code = oc.get("line_item_code", "")
            new_idx = oc.get("correct_order_index")
            if code and new_idx is not None and code in code_to_ids:
                for li_id in code_to_ids[code]:
                    cur.execute(
                        "UPDATE financial_line_items SET order_index = ? WHERE id = ?",
                        (int(new_idx), int(li_id)),
                    )
                applied_order += 1

        # Fix values
        for vc in value_corrections:
            code = vc.get("line_item_code", "")
            for ped, new_val in vc.get("corrections", {}).items():
                li_id = li_lookup.get((ped, code))
                if li_id is not None and new_val is not None:
                    cur.execute(
                        "UPDATE financial_line_items SET amount = ? WHERE id = ?",
                        (float(new_val), int(li_id)),
                    )
                    applied_values += 1

        conn.commit()

    total = applied_names + applied_order + applied_values
    parts_msg = []
    if applied_names:
        parts_msg.append(f"{applied_names} names fixed")
    if applied_order:
        parts_msg.append(f"{applied_order} items reordered")
    if applied_values:
        parts_msg.append(f"{applied_values} values corrected")

    logger.info(
        "AI reconcile for stock %d %s: names=%d order=%d values=%d",
        stock_id, statement_type, applied_names, applied_order, applied_values,
    )

    return {
        "corrections_applied": total,
        "names_fixed": applied_names,
        "items_reordered": applied_order,
        "values_corrected": applied_values,
        "passes_completed": len(all_results),
        "corrections": result,
        "message": (
            f"Reconciliation complete ({len(all_results)}-pass consensus) — {', '.join(parts_msg)}."
            if total > 0
            else f"All items verified ({len(all_results)}-pass consensus) — names, order, and values match the PDF."
        ),
        "confidence": 0.90 if total == 0 else 0.80,
    }
