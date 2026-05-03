"""Liquidity-adjusted position sizer for the Kuwait Signal Engine.

Implements the spec formula:

    Position_Size = (Account_Equity * Risk_Per_Trade * Liquidity_Factor)
                    / (Entry_Price - Stop_Loss)

    Liquidity_Factor = min(1.0, ADTV / LIQUIDITY_THRESHOLD_KD)

With optional half-Kelly scaling based on win probability.
"""
from __future__ import annotations

from app.services.signal_engine.config.risk_config import (
    KELLY_MAX_FRACTION,
    LIQUIDITY_THRESHOLD_KD,
    RISK_PER_TRADE,
    USE_HALF_KELLY,
)


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_loss: float,
    adtv_kd: float,
    win_probability: float | None = None,
    cvar_reduction: float = 1.0,
) -> dict[str, float]:
    """Calculate the recommended position size (number of shares and % of equity).

    Args:
        account_equity:  Total account value in KWD.
        entry_price:     Entry price in fils.
        stop_loss:       Stop-loss price in fils.
        adtv_kd:         20-day average daily traded value in KD.
        win_probability: Calibrated win probability [0, 1] (optional).
                         When provided and valid, applies half-Kelly scaling.
        cvar_reduction:  Additional reduction factor from CVaR check (0-1).

    Returns:
        Dict with keys: shares, equity_pct, position_value_kd.
    """
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0 or entry_price <= 0:
        return {"shares": 0, "equity_pct": 0.0, "position_value_kd": 0.0}

    # ── Liquidity factor ─────────────────────────────────────────────────────
    liquidity_factor = min(1.0, adtv_kd / LIQUIDITY_THRESHOLD_KD) if adtv_kd > 0 else 0.5

    # ── Base risk fraction ────────────────────────────────────────────────────
    risk_fraction = RISK_PER_TRADE * liquidity_factor * cvar_reduction

    # ── Optional half-Kelly scaling ──────────────────────────────────────────
    if win_probability is not None and 0.0 < win_probability < 1.0 and USE_HALF_KELLY:
        # Kelly fraction = (p * b - q) / b  where b = reward/risk ≈ 1.5
        b = 1.5   # assumed reward multiple for Kelly (TP1 target)
        kelly = (win_probability * b - (1.0 - win_probability)) / b
        kelly = max(0.0, kelly)
        half_kelly = kelly / 2.0
        # Use the smaller of base risk fraction and half-Kelly
        risk_fraction = min(risk_fraction, half_kelly)

    risk_fraction = min(risk_fraction, KELLY_MAX_FRACTION)

    # ── Position size in shares ──────────────────────────────────────────────
    # account_equity is in KWD, entry_price in fils → convert entry to KWD
    entry_kd = entry_price / 1000.0
    stop_kd = stop_loss / 1000.0
    risk_per_share_kd = abs(entry_kd - stop_kd)

    if risk_per_share_kd <= 0:
        return {"shares": 0, "equity_pct": 0.0, "position_value_kd": 0.0}

    max_risk_kd = account_equity * risk_fraction
    shares = int(max_risk_kd / risk_per_share_kd)
    position_value_kd = shares * entry_kd
    equity_pct = round(position_value_kd / account_equity * 100.0, 2) if account_equity > 0 else 0.0

    return {
        "shares": shares,
        "equity_pct": equity_pct,
        "position_value_kd": round(position_value_kd, 2),
        "liquidity_factor": round(liquidity_factor, 3),
        "risk_fraction_used": round(risk_fraction * 100.0, 2),
    }
