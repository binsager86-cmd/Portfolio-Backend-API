"""
ml/band_display.py — Phase 3: Per-stock band computation.

Converts a calibrated probability into a human-readable band label
(LOW / MEDIUM / HIGH) using rolling 90-day percentiles from the
shadow log.  Returns special labels when data is insufficient:

  INSUFFICIENT_DATA — fewer than COLD_START_MIN historical rows
  NO_VARIANCE       — standard deviation of history < 1e-4

Band thresholds are recomputed on every call (no caching) so they
always reflect the latest 90-day window.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

LOGGER = logging.getLogger(__name__)

BAND_WINDOW = 90          # days of history to use for percentile computation
COLD_START_MIN = 30       # minimum rows before we emit a real band
NO_VARIANCE_EPS = 1e-4    # std-dev threshold below which we declare NO_VARIANCE

BandResult = Tuple[str, Optional[float], Optional[float]]


def compute_band(
    ticker: str,
    calibrated_prob: float,
    model_id: str,
    signal_date: str,
) -> BandResult:
    """
    Compute the display band for one (ticker, model, date) observation.

    Parameters
    ----------
    ticker          : stock ticker (informational, not used in query)
    calibrated_prob : calibrated probability from the model
    model_id        : ml_models.model_id UUID
    signal_date     : ISO date string of *today* (exclusive upper bound)

    Returns
    -------
    (band_label, low_threshold, high_threshold)
    where low_threshold is the 33rd-percentile cutoff and
    high_threshold is the 67th-percentile cutoff.  Both are None for
    INSUFFICIENT_DATA / NO_VARIANCE.
    """
    from app.core.database import query_all

    rows = query_all(
        """
        SELECT calibrated_prob
          FROM ml_shadow_log
         WHERE model_id = ?
           AND log_date < ?
           AND log_date >= date(?, '-{} days')
           AND calibrated_prob IS NOT NULL
         ORDER BY log_date DESC
        """.format(BAND_WINDOW),
        (model_id, signal_date, signal_date),
    )

    if not rows or len(rows) < COLD_START_MIN:
        return "INSUFFICIENT_DATA", None, None

    import numpy as np
    probs = [float(r["calibrated_prob"]) for r in rows]
    arr = np.array(probs, dtype=float)

    if float(arr.std()) < NO_VARIANCE_EPS:
        return "NO_VARIANCE", None, None

    p33 = float(np.percentile(arr, 33))
    p67 = float(np.percentile(arr, 67))

    if calibrated_prob < p33:
        label = "LOW"
    elif calibrated_prob < p67:
        label = "MEDIUM"
    else:
        label = "HIGH"

    return label, p33, p67


def band_for_display(band_label: str) -> dict:
    """
    Return a display-friendly dict for a band label, suitable for the API response.
    """
    mapping = {
        "LOW":               {"emoji": "🔴", "color": "#EF4444", "short": "Low"},
        "MEDIUM":            {"emoji": "🟡", "color": "#F59E0B", "short": "Mid"},
        "HIGH":              {"emoji": "🟢", "color": "#10B981", "short": "High"},
        "INSUFFICIENT_DATA": {"emoji": "—",  "color": "#9CA3AF", "short": "N/A"},
        "NO_VARIANCE":       {"emoji": "—",  "color": "#9CA3AF", "short": "N/A"},
    }
    return mapping.get(band_label, {"emoji": "—", "color": "#9CA3AF", "short": "N/A"})


DISCLAIMER_TEXT = (
    "⚠️ ML signal in evaluation — do not use for trading decisions yet."
)
