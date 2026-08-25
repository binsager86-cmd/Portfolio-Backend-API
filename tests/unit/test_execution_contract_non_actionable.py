from __future__ import annotations

from app.services.signal_engine.engine.output_formatter import format_signal


def test_non_actionable_execution_hides_trade_plan_but_keeps_scenario_levels() -> None:
    levels = {
        "entry_low": 100.0,
        "entry_high": 102.0,
        "stop_loss": 105.0,
        "tp1": 95.0,
        "tp2": 90.0,
        "tp3": 85.0,
        "risk_per_share": 3.0,
        "risk_reward_ratio": 1.67,
    }
    scenario_levels = {
        "direction": "SHORT",
        "entry_zone_fils": [100.0, 102.0],
        "stop_loss_fils": 105.0,
        "tp1_fils": 95.0,
        "tp2_fils": 90.0,
        "tp3_fils": 85.0,
        "risk_reward_ratio": 1.67,
        "assumptions": {"bearish_structure": True},
    }

    signal = format_signal(
        stock_code="TEST",
        segment="PREMIER",
        signal_direction="NEUTRAL",
        setup_type="No_Signal",
        levels=levels,
        risk_metrics={"equity_pct": None, "cvar_fils": None, "liquidity_factor": None},
        probabilities={
            "p_tp1_before_sl": 0.51,
            "p_tp2_before_sl": 0.31,
            "confidence_interval_95": [0.45, 0.57],
            "expected_return_r_multiple": -0.1,
            "calibration_method": "test",
        },
        confluence={
            "total_score": 49,
            "regime": "Neutral_Chop",
            "regime_confidence": 0.5,
            "auction_intensity": 1.0,
            "sub_scores": {},
            "raw_sub_scores": {},
            "liquidity_passed": True,
            "liquidity_details": {},
            "circuit_proximity": {},
            "support_levels": [],
            "resistance_levels": [],
            "vwap": None,
            "scenario_levels": scenario_levels,
        },
        alerts=["No signal: blocked_by_gates"],
        data_as_of="2026-01-01",
        entry_trigger={
            "action": "HOLD",
            "trigger": "none",
            "pullback": {"triggered": False, "reason": "non_actionable"},
            "breakout": {"triggered": False, "reason": "non_actionable"},
            "accumulation": {"state": "absent", "obv_slope_pct": None, "cmf": None},
        },
        recommendation_contract={
            "direction": "SHORT",
            "direction_score": -42,
            "setup_quality_score": 48,
            "timing_score": 44,
            "data_quality_score": 92.0,
            "expected_value_r": -0.1,
            "recommendation": "HOLD",
            "actionable": False,
        },
    )

    execution = signal["execution"]
    assert execution["actionable"] is False
    assert execution["direction"] == "SHORT"
    assert execution["entry_zone_fils"] is None
    assert execution["stop_loss_fils"] is None
    assert execution["tp1_fils"] is None
    assert execution["tp2_fils"] is None
    assert execution["tp3_fils"] is None
    assert execution["preferred_order_type"] is None
    assert execution["scenario_levels"] == scenario_levels
