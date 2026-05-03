"""3-state Gaussian HMM regime detector for Kuwait equities.

States: Bullish_Expansion | Neutral_Chop | Bearish_Contraction

Primary path uses hmmlearn (GaussianHMM).  When hmmlearn is not installed,
falls back to a deterministic rule-based classifier that reads the
pre-computed indicator columns (ema_20, ema_50, adx_14) from the rows dict.

The model is trained on-the-fly from the passed rows (no serialised model
file required at runtime).  A simple module-level LRU cache prevents
re-training on identical windows.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Any

import numpy as np

from app.services.signal_engine.config.model_params import (
    ADX_TRENDING_MIN,
    HMM_COVARIANCE_TYPE,
    HMM_MIN_TRAIN_BARS,
    HMM_N_ITER,
    HMM_N_STATES,
    HMM_RANDOM_STATE,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_CHOP,
)
from app.services.signal_engine.processors.regime_features import extract_regime_features

logger = logging.getLogger(__name__)

# ── Optional hmmlearn import ──────────────────────────────────────────────────
try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM  # type: ignore
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    logger.info("hmmlearn not installed — using rule-based regime detector fallback")


# ── Module-level model cache (keyed by a hash of the training feature matrix) ─
_model_cache: dict[str, "_GaussianHMM"] = {}   # type: ignore[name-defined]


def _feature_hash(features: np.ndarray) -> str:
    """Produce a short hash of the feature matrix for cache keying."""
    return hashlib.md5(features.tobytes()).hexdigest()[:16]


def _assign_state_labels(model: "_GaussianHMM") -> dict[int, str]:  # type: ignore[name-defined]
    """Map HMM state indices → regime names by sorting on mean log_return.

    The first feature column is log_return, so states with higher means
    correspond to more bullish regimes.
    """
    mean_returns = model.means_[:, 0]       # log_return mean per state
    order = np.argsort(mean_returns)        # ascending: bear, chop, bull
    labels = [REGIME_BEAR, REGIME_CHOP, REGIME_BULL]
    return {int(state): labels[rank] for rank, state in enumerate(order)}


def _rule_based_regime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fallback rule-based regime classifier using pre-computed indicators."""
    last = rows[-1]
    ema20 = last.get("ema_20")
    ema50 = last.get("ema_50")
    close = float(last.get("close") or 0.0)
    adx = last.get("adx_14") or 0.0

    trending = float(adx) >= ADX_TRENDING_MIN

    if ema20 and ema50:
        if close > float(ema20) > float(ema50) and trending:
            regime = REGIME_BULL
        elif close < float(ema20) < float(ema50) and trending:
            regime = REGIME_BEAR
        else:
            regime = REGIME_CHOP
    else:
        regime = REGIME_CHOP

    # Assign rough probabilities
    probs = {REGIME_BULL: 0.10, REGIME_CHOP: 0.10, REGIME_BEAR: 0.10}
    probs[regime] = 0.80
    total = sum(probs.values())
    probs = {k: round(v / total, 3) for k, v in probs.items()}

    # Days in current regime: count backwards while same regime (proxy)
    days_in = 1
    if len(rows) > 1:
        for r in reversed(rows[:-1]):
            _prev_ema20 = r.get("ema_20")
            _prev_ema50 = r.get("ema_50")
            _prev_close = float(r.get("close") or 0.0)
            _prev_adx = float(r.get("adx_14") or 0.0)
            _prev_trending = _prev_adx >= ADX_TRENDING_MIN
            if _prev_ema20 and _prev_ema50:
                if _prev_close > float(_prev_ema20) > float(_prev_ema50) and _prev_trending:
                    prev_regime = REGIME_BULL
                elif _prev_close < float(_prev_ema20) < float(_prev_ema50) and _prev_trending:
                    prev_regime = REGIME_BEAR
                else:
                    prev_regime = REGIME_CHOP
            else:
                prev_regime = REGIME_CHOP
            if prev_regime == regime:
                days_in += 1
            else:
                break

    return {
        "current_regime": regime,
        "state_probabilities": [probs[REGIME_BEAR], probs[REGIME_CHOP], probs[REGIME_BULL]],
        "regime_confidence": probs[regime],
        "days_in_current_regime": days_in,
        "method": "rule_based",
    }


def predict_regime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect the current market regime from OHLCV + indicator rows.

    Args:
        rows: OHLCV rows with attached indicator columns, sorted ascending.

    Returns:
        Dict with keys: current_regime, state_probabilities, regime_confidence,
        days_in_current_regime, method.
    """
    if not rows:
        return {
            "current_regime": REGIME_CHOP,
            "state_probabilities": [0.5],
            "regime_confidence": 0.5,
            "days_in_current_regime": 0,
            "method": "empty_data_fallback",
        }
    if not _HMM_AVAILABLE or len(rows) < HMM_MIN_TRAIN_BARS:
        return _rule_based_regime(rows)

    features = extract_regime_features(rows)

    # Drop rows with NaN (warmup ATR rows)
    valid_mask = ~np.any(np.isnan(features), axis=1)
    clean_features = features[valid_mask]

    if len(clean_features) < HMM_MIN_TRAIN_BARS:
        return _rule_based_regime(rows)

    # Use last HMM_MIN_TRAIN_BARS * 2 bars to keep training fast
    train_window = min(len(clean_features), HMM_MIN_TRAIN_BARS * 2)
    train_features = clean_features[-train_window:]

    cache_key = _feature_hash(train_features)
    if cache_key not in _model_cache:
        try:
            model = _GaussianHMM(  # type: ignore[misc]
                n_components=HMM_N_STATES,
                covariance_type=HMM_COVARIANCE_TYPE,
                n_iter=HMM_N_ITER,
                random_state=HMM_RANDOM_STATE,
            )
            model.fit(train_features)
            _model_cache[cache_key] = model
            # Keep cache bounded to 64 entries
            if len(_model_cache) > 64:
                oldest = next(iter(_model_cache))
                del _model_cache[oldest]
        except Exception as exc:  # noqa: BLE001
            logger.warning("HMM training failed (%s) — falling back to rule-based", exc)
            return _rule_based_regime(rows)

    model = _model_cache[cache_key]
    state_labels = _assign_state_labels(model)

    try:
        state_seq = model.predict(clean_features)
        current_state = int(state_seq[-1])
        log_probs = model.predict_proba(clean_features[-1:])  # (1, n_states)
        state_probs = log_probs[0].tolist()
    except Exception as exc:  # noqa: BLE001
        logger.warning("HMM predict failed (%s) — falling back to rule-based", exc)
        return _rule_based_regime(rows)

    current_regime = state_labels[current_state]
    confidence = float(state_probs[current_state])

    # Count consecutive days in current state
    days_in = 1
    for s in reversed(state_seq[:-1]):
        if s == current_state:
            days_in += 1
        else:
            break

    # Reorder probs as [bear, chop, bull]
    label_to_idx = {v: k for k, v in state_labels.items()}
    ordered_probs = [
        round(float(state_probs[label_to_idx[REGIME_BEAR]]), 3),
        round(float(state_probs[label_to_idx[REGIME_CHOP]]), 3),
        round(float(state_probs[label_to_idx[REGIME_BULL]]), 3),
    ]

    return {
        "current_regime": current_regime,
        "state_probabilities": ordered_probs,
        "regime_confidence": round(confidence, 3),
        "days_in_current_regime": days_in,
        "method": "hmm",
    }
