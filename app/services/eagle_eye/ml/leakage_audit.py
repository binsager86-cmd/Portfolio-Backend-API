"""
ml/leakage_audit.py — Phase 1: Leakage audit framework.

Provides a test harness that can detect three classes of time-series
leakage before any model is trained:

  1. CENTERED WINDOWS  — rolling operations that look ahead in time
     (e.g. df['x'].rolling(20, center=True))
  2. FUTURE LABELS     — features whose values are derived from data
     strictly after the row's prediction date
  3. TARGET ENCODING   — features whose values encode the label
     (correlation > threshold on the training set)

Usage
-----
    from app.services.eagle_eye.ml.leakage_audit import LeakageAuditor

    auditor = LeakageAuditor()
    report  = auditor.audit_dataframe(df, date_col="event_date", target_col="y")
    auditor.assert_clean(report)          # raises if any LEAKY features found
    auditor.write_registry(report)        # persists to features_audit DB table

Source-level checks are also available:
    issues = auditor.audit_source_text(source_code_str)
"""
from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Pearson |r| above this → flag as potential target-encoding leakage
TARGET_CORR_THRESHOLD = 0.95

#: Columns whose names should always be reviewed regardless of computed stats
ALWAYS_REVIEW_PATTERNS: tuple[str, ...] = (
    "future_",
    "_next_",
    "_forward_",
    "_fwd_",
    "_lookahead",
    "_t+",
)

#: Column name patterns that are known-non-feature bookkeeping columns
META_COLUMN_PATTERNS: tuple[str, ...] = (
    "ticker",
    "event_id",
    "event_date",
    "acceleration_date",
    "start_date",
    "peak_date",
    "sample_type",
    "bar_date",
    "trade_date",
    "label",
    "y_",
    "_y",
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FeatureAuditResult:
    feature_name: str
    verdict: str          # "CLEAN" | "LEAKY" | "REVIEW" | "DROPPED"
    checks: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class AuditReport:
    audited_at: str
    n_features: int
    n_clean: int
    n_leaky: int
    n_review: int
    results: List[FeatureAuditResult] = field(default_factory=list)
    source_issues: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.n_leaky == 0

    def summary(self) -> str:
        lines = [
            f"Leakage audit — {self.audited_at}",
            f"  features : {self.n_features}",
            f"  CLEAN    : {self.n_clean}",
            f"  LEAKY    : {self.n_leaky}",
            f"  REVIEW   : {self.n_review}",
        ]
        if self.source_issues:
            lines.append(f"  source issues : {len(self.source_issues)}")
            for issue in self.source_issues[:5]:
                lines.append(f"    - {issue}")
        if self.n_leaky:
            lines.append("  LEAKY features:")
            for r in self.results:
                if r.verdict == "LEAKY":
                    lines.append(f"    - {r.feature_name}: {r.notes}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AST-level source code scanner
# ---------------------------------------------------------------------------

class _CenteredWindowVisitor(ast.NodeVisitor):
    """Walk an AST and flag rolling(…, center=True) calls."""

    def __init__(self) -> None:
        self.issues: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # Look for .rolling(..., center=True, ...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "rolling":
            for kw in node.keywords:
                if kw.arg == "center":
                    val = kw.value
                    if isinstance(val, ast.Constant) and val.value is True:
                        lineno = getattr(node, "lineno", "?")
                        self.issues.append(
                            f"line {lineno}: centered rolling window detected — "
                            "this looks ahead into the future"
                        )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        # Flag iloc[i-w : i+w+1] style — centered slice pattern
        if isinstance(node.slice, ast.Slice):
            slc = node.slice
            upper = slc.upper
            if upper is not None and isinstance(upper, ast.BinOp):
                # Any addition on the upper slice bound is suspicious
                if isinstance(upper.op, ast.Add):
                    lineno = getattr(node, "lineno", "?")
                    self.issues.append(
                        f"line {lineno}: possible centered slice `iloc[i-w : i+w+N]` — "
                        "verify this is not a future-look"
                    )
        self.generic_visit(node)


def scan_source_for_leakage(source: str) -> list[str]:
    """Return list of leakage-risk descriptions found in Python source text."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError as exc:
        return [f"SyntaxError parsing source: {exc}"]

    visitor = _CenteredWindowVisitor()
    visitor.visit(tree)

    issues = list(visitor.issues)

    # Textual heuristics (complement AST scan)
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip().lower()
        if "shift(-" in stripped:
            issues.append(f"line {i}: negative shift detected — may introduce future data")
        if ".fillna(method='bfill')" in stripped or "bfill()" in stripped:
            issues.append(
                f"line {i}: back-fill (bfill) detected — can propagate future values backward"
            )

    return issues


# ---------------------------------------------------------------------------
# DataFrame-level auditor
# ---------------------------------------------------------------------------

class LeakageAuditor:
    """
    Audits a feature DataFrame for time-series leakage.

    Parameters
    ----------
    target_corr_threshold : float
        Pearson |r| above this triggers LEAKY verdict for that feature.
    always_review_patterns : sequence of str
        Any feature whose name contains one of these strings is marked REVIEW
        regardless of computed statistics.
    """

    def __init__(
        self,
        target_corr_threshold: float = TARGET_CORR_THRESHOLD,
        always_review_patterns: Sequence[str] = ALWAYS_REVIEW_PATTERNS,
    ) -> None:
        self.target_corr_threshold = target_corr_threshold
        self.always_review_patterns = tuple(always_review_patterns)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit_dataframe(
        self,
        df: pd.DataFrame,
        *,
        date_col: str = "event_date",
        target_col: str = "y",
        train_cutoff: Optional[str] = None,
    ) -> AuditReport:
        """
        Audit every numeric feature column in ``df``.

        Parameters
        ----------
        df            : feature + label DataFrame (rows ordered by date_col)
        date_col      : name of the date/timestamp column
        target_col    : name of the binary label column
        train_cutoff  : ISO date string; rows after this are the test set.
                        If None, the last 25 % of rows are treated as test.
        """
        df = df.copy()
        if date_col in df.columns:
            df = df.sort_values(date_col).reset_index(drop=True)

        # Determine train / test split
        if train_cutoff is not None and date_col in df.columns:
            is_train = df[date_col] <= train_cutoff
        else:
            split_idx = max(1, int(len(df) * 0.75))
            is_train = pd.Series([True] * split_idx + [False] * (len(df) - split_idx))

        train_df = df[is_train]
        test_df = df[~is_train]

        # Feature columns to audit
        meta_cols = {date_col, target_col} | self._meta_columns(df.columns)
        feature_cols = [
            c for c in df.columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        results: list[FeatureAuditResult] = []
        for col in feature_cols:
            result = self._audit_feature(col, df, train_df, test_df, target_col, date_col)
            results.append(result)

        n_clean = sum(1 for r in results if r.verdict == "CLEAN")
        n_leaky = sum(1 for r in results if r.verdict == "LEAKY")
        n_review = sum(1 for r in results if r.verdict == "REVIEW")

        return AuditReport(
            audited_at=datetime.utcnow().isoformat(),
            n_features=len(results),
            n_clean=n_clean,
            n_leaky=n_leaky,
            n_review=n_review,
            results=results,
        )

    def audit_source_text(self, source: str) -> list[str]:
        """Return AST + text leakage issues found in Python source."""
        return scan_source_for_leakage(source)

    def audit_callable(self, func: Callable) -> list[str]:
        """Inspect and audit the source of a Python function/class."""
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError) as exc:
            return [f"Could not retrieve source: {exc}"]
        return self.audit_source_text(source)

    def assert_clean(self, report: AuditReport) -> None:
        """Raise ValueError if any LEAKY features are present."""
        if not report.is_clean:
            leaky = [r.feature_name for r in report.results if r.verdict == "LEAKY"]
            raise ValueError(
                f"Leakage audit failed — {len(leaky)} LEAKY feature(s): {leaky}\n"
                + report.summary()
            )

    def write_registry(self, report: AuditReport, feature_version: str = "v1") -> None:
        """Persist audit results to the features_audit DB table."""
        try:
            from app.core.database import exec_sql
        except ImportError:
            logger.warning("DB not available — skipping registry write")
            return

        for r in report.results:
            exec_sql(
                """
                INSERT INTO features_audit
                    (feature_name, feature_version, leakage_verdict, audit_notes, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT (feature_name, feature_version) DO UPDATE SET
                    leakage_verdict = excluded.leakage_verdict,
                    audit_notes     = excluded.audit_notes,
                    updated_at      = excluded.updated_at
                """,
                (r.feature_name, feature_version, r.verdict, r.notes or ""),
            )
        logger.info(
            "Feature audit registry updated: %d features written (version=%s)",
            len(report.results),
            feature_version,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _meta_columns(self, columns) -> set[str]:
        meta: set[str] = set()
        for col in columns:
            lc = col.lower()
            if any(pat in lc for pat in META_COLUMN_PATTERNS):
                meta.add(col)
        return meta

    def _audit_feature(
        self,
        col: str,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        date_col: str,
    ) -> FeatureAuditResult:
        checks: dict[str, Any] = {}
        notes_parts: list[str] = []
        verdict = "CLEAN"

        lc = col.lower()

        # ── Check 1: Name-pattern review ──────────────────────────────
        if any(pat in lc for pat in self.always_review_patterns):
            verdict = "REVIEW"
            notes_parts.append(f"Name contains suspicious pattern: {col}")

        # ── Check 2: Target correlation (possible target encoding) ────
        if target_col in df.columns:
            try:
                corr = df[col].corr(df[target_col])
                checks["target_corr"] = round(float(corr) if not np.isnan(corr) else 0.0, 4)
                if abs(checks["target_corr"]) >= self.target_corr_threshold:
                    verdict = "LEAKY"
                    notes_parts.append(
                        f"Target correlation={checks['target_corr']:.3f} ≥ "
                        f"threshold {self.target_corr_threshold} — likely encodes answer"
                    )
            except Exception:  # noqa: BLE001
                checks["target_corr"] = None

        # ── Check 3: Train-scale leakage ──────────────────────────────
        # If the test mean is dramatically different from what the train
        # distribution would predict → suggests normalization used test data.
        if len(train_df) > 5 and len(test_df) > 5:
            try:
                train_mean = train_df[col].mean()
                train_std = train_df[col].std()
                test_mean = test_df[col].mean()
                if train_std and train_std > 0:
                    # z-score of test mean relative to train distribution
                    z = abs((test_mean - train_mean) / train_std)
                    checks["test_z_vs_train"] = round(z, 3)
                    if z > 5:
                        if verdict == "CLEAN":
                            verdict = "REVIEW"
                        notes_parts.append(
                            f"Test mean z-score={z:.2f} — "
                            "large shift vs train; check normalization"
                        )
            except Exception:  # noqa: BLE001
                checks["test_z_vs_train"] = None

        # ── Check 4: Monotone with time (possible future-price leakage) ─
        if date_col in df.columns and len(df) > 20:
            try:
                sample = df[[date_col, col]].dropna().head(200)
                time_idx = pd.to_datetime(sample[date_col]).astype(np.int64)
                corr_time = sample[col].corr(pd.Series(time_idx, index=sample.index))
                checks["time_corr"] = round(float(corr_time) if not np.isnan(corr_time) else 0.0, 4)
                if abs(checks["time_corr"]) > 0.98:
                    if verdict == "CLEAN":
                        verdict = "REVIEW"
                    notes_parts.append(
                        f"Near-monotone with time (|r|={abs(checks['time_corr']):.3f}) — "
                        "verify this is not a cumulative sum of future values"
                    )
            except Exception:  # noqa: BLE001
                checks["time_corr"] = None

        # ── Check 5: NaN rate ─────────────────────────────────────────
        nan_rate = df[col].isna().mean()
        checks["nan_rate"] = round(float(nan_rate), 4)
        if nan_rate > 0.20:
            if verdict == "CLEAN":
                verdict = "REVIEW"
            notes_parts.append(f"NaN rate={nan_rate:.1%} > 20% — flag for data quality")

        return FeatureAuditResult(
            feature_name=col,
            verdict=verdict,
            checks=checks,
            notes="; ".join(notes_parts) if notes_parts else "OK",
        )


# ---------------------------------------------------------------------------
# Convenience: audit a module's feature-builder functions
# ---------------------------------------------------------------------------

def audit_feature_builder_module(module_path: str) -> AuditReport:
    """
    Read a Python source file and run the AST leakage scanner on it.
    Returns a report with only source_issues populated (no DataFrame audit).
    """
    import pathlib

    src = pathlib.Path(module_path).read_text(encoding="utf-8")
    issues = scan_source_for_leakage(src)
    return AuditReport(
        audited_at=datetime.utcnow().isoformat(),
        n_features=0,
        n_clean=0,
        n_leaky=0,
        n_review=0,
        source_issues=issues,
    )
