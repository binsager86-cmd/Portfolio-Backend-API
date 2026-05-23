"""
Behavioral DNA Extractor.
For each stock, distill forensic event snapshots into the stock's
behavioral fingerprint for live engine comparisons and visual review.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from app.services.eagle_eye.config import CONFIG
from app.services.eagle_eye.recorder import ForensicSnapshot, SIGNAL_DEFS


DNA_DEFAULT_WINDOW_DAYS = 20
DNA_WINDOW_OPTIONS = (20, 60, 180)
DNA_CONFIDENCE_FLOOR = 5
DNA_BUILDING_FLOOR = 10
DNA_ESTABLISHED_FLOOR = 20
DNA_CONTEXT_LOOKBACK_BARS = 30
DNA_MAX_SETUP_EXAMPLES = 3
DNA_CHART_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rel_volume",
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "adx",
    "plus_di",
    "minus_di",
)


@dataclass
class SignalReliability:
    signal: str
    avg_lead_days: float
    median_lead_days: float
    reliability_pct: float
    false_positive_rate: float
    discriminative_power: float
    fired_count: int
    total_events: int


@dataclass
class SetupSignalStat:
    signal: str
    fired_count: int
    total_setups: int
    presence_pct: float


@dataclass
class ThresholdProfile:
    threshold_pct: float
    occurrences: int
    sample_count: int
    fakeouts: int
    success_rate: float
    avg_consolidation_days: float
    avg_duration_days: float
    avg_gain_pct: Optional[float]
    avg_gain_all_pct: float
    avg_gain_on_hits_pct: Optional[float]
    earliest_reliable_signals: List[SignalReliability] = field(default_factory=list)
    confirmation_signals: List[SignalReliability] = field(default_factory=list)


@dataclass
class SetupWindowProfile:
    horizon_days: int
    setup_count: int
    history_status: str
    confidence_floor: int
    confidence_tier: str
    confidence_label: str
    percentages_visible: bool
    threshold_profiles: List[ThresholdProfile] = field(default_factory=list)


@dataclass
class SetupObservation:
    date: str
    signal: str
    label: str
    detail: str
    value: Optional[float] = None


@dataclass
class SetupForwardOutcome:
    horizon_days: int
    completed: bool
    max_gain_pct: Optional[float]
    max_gain_date: Optional[str]
    threshold_hits: List[float] = field(default_factory=list)


@dataclass
class SetupExample:
    setup_date: str
    setup_window_start_date: str
    setup_window_end_date: str
    setup_bar_index: int
    setup_window_start_index: int
    setup_window_end_index: int
    available_forward_bars: int
    bars: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[SetupObservation] = field(default_factory=list)
    forward_outcomes: Dict[str, SetupForwardOutcome] = field(default_factory=dict)


@dataclass
class BehavioralDNA:
    ticker: str
    total_events_studied: int
    fakeouts_studied: int
    history_status: str
    profiles_by_threshold: List[ThresholdProfile]
    personality_tag: str
    avg_pre_move_consolidation_days: float
    avg_move_duration_days: float
    avg_move_magnitude_pct: float
    most_reliable_signals_overall: List[SignalReliability] = field(default_factory=list)
    fakeout_signatures: List[str] = field(default_factory=list)
    setup_signals: List[str] = field(default_factory=list)
    setup_horizon_days: int = DNA_DEFAULT_WINDOW_DAYS
    signal_stats: List[SetupSignalStat] = field(default_factory=list)
    pre_move_volume_profile: Dict[str, Any] = field(default_factory=dict)
    fakeout_volume_profile: Dict[str, Any] = field(default_factory=dict)
    available_window_days: List[int] = field(default_factory=list)
    default_window_days: int = DNA_DEFAULT_WINDOW_DAYS
    confidence_floor: int = DNA_CONFIDENCE_FLOOR
    window_profiles: List[SetupWindowProfile] = field(default_factory=list)
    setup_examples: List[SetupExample] = field(default_factory=list)


def _round_optional(value: Any, digits: int = 1) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric):
        return None
    return round(numeric, digits)


def _date_string(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return str(value)


def _normalize_windows(
    horizon_days: Optional[int],
    window_days: Optional[Sequence[int]],
) -> tuple[List[int], int]:
    if window_days:
        normalized = sorted({int(day) for day in window_days if int(day) > 0})
        default_window = int(horizon_days or DNA_DEFAULT_WINDOW_DAYS)
        if default_window not in normalized:
            normalized.insert(0, default_window)
            normalized = sorted(set(normalized))
        return normalized, default_window

    default_window = int(horizon_days or CONFIG.MAX_MOVE_LOOKAHEAD_DAYS)
    return [default_window], default_window


def _confidence_meta(setup_count: int, confidence_floor: int) -> tuple[str, str, bool, str]:
    if setup_count >= DNA_ESTABLISHED_FLOOR:
        return ("ESTABLISHED", "Established (20+ setups)", True, "ok")
    if setup_count >= DNA_BUILDING_FLOOR:
        return ("BUILDING", "Building confidence (10-19 setups)", True, "ok")
    if setup_count >= confidence_floor:
        return ("EARLY", f"Early read ({confidence_floor}-9 setups)", True, "ok")
    return ("TOO_THIN", f"Too thin (<{confidence_floor} setups)", False, "INSUFFICIENT_HISTORY")


def _aggregate_signal_reliability(
    snapshots: List[ForensicSnapshot],
    fakeout_snapshots: List[ForensicSnapshot],
) -> Dict[str, SignalReliability]:
    lead_times: Dict[str, List[int]] = defaultdict(list)
    fired_in_real: Counter = Counter()
    fired_in_fake: Counter = Counter()

    n_real = len(snapshots)
    n_fake = len(fakeout_snapshots)

    for snap in snapshots:
        seen_signals = set()
        for entry in snap.signal_sequence:
            sig = entry["signal"]
            lead_times[sig].append(entry["days_before_acceleration"])
            seen_signals.add(sig)
        for sig in seen_signals:
            fired_in_real[sig] += 1

    for snap in fakeout_snapshots:
        seen_signals = {entry["signal"] for entry in snap.signal_sequence}
        for sig in seen_signals:
            fired_in_fake[sig] += 1

    reliability: Dict[str, SignalReliability] = {}
    for sig, leads in lead_times.items():
        if not leads:
            continue
        rel_pct = (fired_in_real[sig] / n_real * 100) if n_real else 0.0
        fpr_pct = (fired_in_fake[sig] / n_fake * 100) if n_fake else 0.0
        reliability[sig] = SignalReliability(
            signal=sig,
            avg_lead_days=float(np.mean(leads)),
            median_lead_days=float(np.median(leads)),
            reliability_pct=rel_pct,
            false_positive_rate=fpr_pct,
            discriminative_power=rel_pct - fpr_pct,
            fired_count=int(fired_in_real[sig]),
            total_events=int(n_real),
        )
    return reliability


def _signals_fired(row: pd.Series) -> set[str]:
    fired: set[str] = set()
    for signal_name, signal_fn in SIGNAL_DEFS.items():
        try:
            if signal_fn(row):
                fired.add(signal_name)
        except Exception:
            continue
    return fired


def _signal_rank(
    signal: str,
    reliability: Dict[str, SignalReliability],
) -> tuple[float, float, int, str]:
    rel = reliability.get(signal)
    if rel is None:
        return (0.0, 0.0, 0, signal)
    return (
        float(rel.discriminative_power),
        float(rel.reliability_pct),
        int(rel.fired_count),
        signal,
    )


def _window_outcome(
    closes: np.ndarray,
    dates: Sequence[Any],
    pos: int,
    horizon_days: int,
    thresholds: Sequence[float],
) -> Optional[SetupForwardOutcome]:
    if pos + 1 >= len(closes):
        return None

    base_price = closes[pos]
    if base_price is None or np.isnan(base_price) or base_price <= 0:
        return None

    end_pos = min(len(closes), pos + horizon_days + 1)
    future_closes = closes[pos + 1:end_pos]
    if len(future_closes) == 0 or np.isnan(future_closes).all():
        return None

    relative_max_pos = int(np.nanargmax(future_closes))
    max_pos = pos + 1 + relative_max_pos
    max_close = float(future_closes[relative_max_pos])
    max_gain_pct = float((max_close - base_price) / base_price * 100)
    return SetupForwardOutcome(
        horizon_days=horizon_days,
        completed=True,
        max_gain_pct=max_gain_pct,
        max_gain_date=_date_string(dates[max_pos]),
        threshold_hits=[float(threshold) for threshold in thresholds if max_gain_pct >= float(threshold)],
    )


def _find_setup_matches(
    indicators_df: Optional[pd.DataFrame],
    setup_signals: List[str],
) -> List[Dict[str, Any]]:
    if indicators_df is None or indicators_df.empty or not setup_signals:
        return []

    matches: List[Dict[str, Any]] = []
    match_start: Optional[int] = None
    match_signals: set[str] = set()

    for pos in range(len(indicators_df)):
        row = indicators_df.iloc[pos]
        fired = _signals_fired(row)
        is_match = all(signal in fired for signal in setup_signals)

        if is_match and match_start is None:
            match_start = pos
            match_signals = fired
            continue

        if not is_match and match_start is not None:
            matches.append(
                {
                    "date": indicators_df.index[match_start],
                    "pos": match_start,
                    "end_pos": pos - 1,
                    "signals": set(match_signals),
                }
            )
            match_start = None
            match_signals = set()

    if match_start is not None:
        matches.append(
            {
                "date": indicators_df.index[match_start],
                "pos": match_start,
                "end_pos": len(indicators_df) - 1,
                "signals": set(match_signals),
            }
        )

    return matches


def _collect_setup_occurrences(
    indicators_df: Optional[pd.DataFrame],
    setup_matches: List[Dict[str, Any]],
    horizon_days: int,
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    if indicators_df is None or indicators_df.empty or not setup_matches:
        return []

    closes = indicators_df["close"].astype(float).to_numpy()
    dates = list(indicators_df.index)
    occurrences: List[Dict[str, Any]] = []

    for match in setup_matches:
        outcome = _window_outcome(closes, dates, match["pos"], horizon_days, thresholds)
        if outcome is None:
            continue
        occurrences.append(
            {
                **match,
                "forward_gain_pct": float(outcome.max_gain_pct or 0.0),
                "max_gain_date": outcome.max_gain_date,
                "threshold_hits": outcome.threshold_hits,
            }
        )

    return occurrences


def _select_setup_signals(
    indicators_df: Optional[pd.DataFrame],
    reliability: Dict[str, SignalReliability],
    min_setup_occurrences: int,
    horizon_days: int,
) -> tuple[List[str], List[Dict[str, Any]]]:
    if indicators_df is None or indicators_df.empty:
        return [], []

    thresholds = [float(t) for t in CONFIG.MOVE_THRESHOLDS_PCT if float(t) > 0]
    active_now = list(_signals_fired(indicators_df.iloc[-1]))
    if not active_now:
        ranked = sorted(reliability, key=lambda signal: _signal_rank(signal, reliability), reverse=True)
    else:
        ranked = sorted(active_now, key=lambda signal: _signal_rank(signal, reliability), reverse=True)

    if not ranked:
        return [], []

    max_core_signals = min(3, len(ranked))
    best_signals = [ranked[0]]
    best_matches = _find_setup_matches(indicators_df, best_signals)
    best_occurrences = _collect_setup_occurrences(indicators_df, best_matches, horizon_days, thresholds)

    for signal_count in range(max_core_signals, 0, -1):
        candidate = ranked[:signal_count]
        candidate_matches = _find_setup_matches(indicators_df, candidate)
        candidate_occurrences = _collect_setup_occurrences(
            indicators_df,
            candidate_matches,
            horizon_days,
            thresholds,
        )
        if len(candidate_occurrences) >= min_setup_occurrences:
            return candidate, candidate_occurrences
        if len(candidate_occurrences) > len(best_occurrences):
            best_signals = candidate
            best_occurrences = candidate_occurrences

    return best_signals, best_occurrences


def _build_setup_signal_stats(occurrences: List[Dict[str, Any]]) -> List[SetupSignalStat]:
    if not occurrences:
        return []

    counts: Counter = Counter()
    total = len(occurrences)
    for occurrence in occurrences:
        for signal in occurrence["signals"]:
            counts[signal] += 1

    return [
        SetupSignalStat(
            signal=signal,
            fired_count=int(fired_count),
            total_setups=total,
            presence_pct=float(fired_count / total * 100),
        )
        for signal, fired_count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _classify_personality(dna_inputs: Dict[str, Any]) -> str:
    avg_consol = dna_inputs["avg_consolidation"]
    avg_duration = dna_inputs["avg_duration"]
    avg_magnitude = dna_inputs["avg_magnitude"]

    if avg_consol > 40 and avg_duration > 60:
        return "slow_builder"
    if avg_consol < 15 and avg_duration < 30:
        return "volatile_burst"
    if avg_magnitude > 40 and avg_duration > 45:
        return "high_amplitude_trender"
    if avg_magnitude < 20:
        return "range_grinder"
    return "balanced_mover"


def _compute_volume_profile(snapshots: List[ForensicSnapshot]) -> Dict[str, Any]:
    lookbacks = [90, 60, 30, 14, 7, 3, 0]
    keys = {lb: f"avg_rel_vol_t{lb}" for lb in lookbacks}

    accum: Dict[int, List[float]] = {lb: [] for lb in lookbacks}
    for snap in snapshots:
        for lb in lookbacks:
            rv = (snap.indicator_snapshots.get(lb) or {}).get("rel_volume")
            if rv is not None and not (isinstance(rv, float) and np.isnan(rv)):
                accum[lb].append(float(rv))

    avgs = {}
    for lb in lookbacks:
        vals = accum[lb]
        avgs[lb] = round(float(np.mean(vals)), 3) if vals else None

    t90 = avgs.get(90)
    t30 = avgs.get(30)
    t7 = avgs.get(7)
    t0 = avgs.get(0)

    if t90 is None or t0 is None:
        pattern = "NO_CLEAR_PATTERN"
    elif t0 > (t7 or 0) > (t30 or 0) > (t90 or 0):
        pattern = "GRADUAL_BUILD"
    elif t0 is not None and (t7 or 0) < 1.0 and t0 > 1.5:
        pattern = "LATE_SPIKE"
    elif t30 is not None and t30 >= 1.2 and t0 is not None:
        pattern = "EARLY_SIGNAL"
    else:
        pattern = "NO_CLEAR_PATTERN"

    t0_vals = accum.get(0, [])
    min_rv = round(float(np.percentile(t0_vals, 10)), 3) if len(t0_vals) >= 5 else None

    profile: Dict[str, Any] = {key: avgs[lb] for lb, key in keys.items()}
    profile["volume_pattern"] = pattern
    profile["min_rel_vol_for_real_move"] = min_rv
    return profile


def _build_threshold_profiles(
    setup_occurrences: List[Dict[str, Any]],
    thresholds: Sequence[float],
    real_moves: List[ForensicSnapshot],
    fakeouts: List[ForensicSnapshot],
) -> List[ThresholdProfile]:
    if not setup_occurrences:
        return []

    avg_consolidation = float(np.mean([s.event.days_consolidating_before for s in real_moves])) if real_moves else 0.0
    avg_duration = float(np.mean([s.event.duration_days for s in real_moves])) if real_moves else 0.0
    setup_returns = [float(occurrence["forward_gain_pct"]) for occurrence in setup_occurrences]
    avg_gain_all = float(np.mean(setup_returns)) if setup_returns else 0.0

    profiles: List[ThresholdProfile] = []
    for threshold in thresholds:
        tier_hits = [occ for occ in setup_occurrences if float(occ["forward_gain_pct"]) >= threshold]
        profiles.append(
            ThresholdProfile(
                threshold_pct=float(threshold),
                occurrences=len(tier_hits),
                sample_count=len(setup_occurrences),
                fakeouts=len(fakeouts),
                success_rate=(len(tier_hits) / len(setup_occurrences) * 100) if setup_occurrences else 0.0,
                avg_consolidation_days=avg_consolidation,
                avg_duration_days=avg_duration,
                avg_gain_pct=float(np.mean([occ["forward_gain_pct"] for occ in tier_hits])) if tier_hits else None,
                avg_gain_all_pct=avg_gain_all,
                avg_gain_on_hits_pct=float(np.mean([occ["forward_gain_pct"] for occ in tier_hits])) if tier_hits else None,
            )
        )
    return profiles


def _observation_detail(signal: str, row: pd.Series) -> tuple[str, Optional[float]]:
    if signal == "volume_breakout_2x":
        value = _round_optional(row.get("rel_volume"), 2)
        return (f"Relative volume expanded to {value}x the recent norm." if value is not None else "Relative volume expanded above 2x normal.", value)
    if signal == "volume_breakout_15x":
        value = _round_optional(row.get("rel_volume"), 2)
        return (f"Relative volume reached {value}x normal participation." if value is not None else "Relative volume expanded above 1.5x normal.", value)
    if signal == "macd_histogram_turned_positive":
        value = _round_optional(row.get("macd_histogram"), 3)
        return (f"MACD histogram flipped positive at {value}." if value is not None else "MACD histogram flipped positive.", value)
    if signal == "rsi_in_bullish_zone":
        value = _round_optional(row.get("rsi"), 1)
        return (f"RSI entered the bullish 50-70 zone at {value}." if value is not None else "RSI entered the bullish zone.", value)
    if signal == "adx_strong_trend":
        value = _round_optional(row.get("adx"), 1)
        return (f"ADX reached {value}, confirming a stronger trend." if value is not None else "ADX confirmed a stronger trend.", value)
    if signal == "adx_crossed_20":
        value = _round_optional(row.get("adx"), 1)
        return (f"ADX crossed 20 and rose to {value}." if value is not None else "ADX crossed 20, showing trend strength is emerging.", value)
    if signal == "plus_di_dominates":
        plus_di = _round_optional(row.get("plus_di"), 1)
        minus_di = _round_optional(row.get("minus_di"), 1)
        if plus_di is not None and minus_di is not None:
            return (f"+DI ({plus_di}) moved above -DI ({minus_di}), shifting control to buyers.", plus_di)
        return ("+DI moved above -DI, shifting control to buyers.", plus_di)
    if signal == "supertrend_bullish":
        return ("Supertrend flipped bullish on this bar.", 1.0)
    if signal == "accumulation_above_75":
        value = _round_optional(row.get("accumulation_score"), 1)
        return (f"Accumulation score rose above 75 to {value}." if value is not None else "Accumulation score rose above 75.", value)
    if signal == "accumulation_above_65":
        value = _round_optional(row.get("accumulation_score"), 1)
        return (f"Accumulation score pushed through 65 to {value}." if value is not None else "Accumulation score pushed through 65.", value)
    return (signal.replace("_", " ").title(), None)


def _observation_label(signal: str) -> str:
    labels = {
        "volume_breakout_2x": "Volume surged",
        "volume_breakout_15x": "Volume expanded",
        "macd_histogram_turned_positive": "MACD turned positive",
        "rsi_in_bullish_zone": "RSI entered bullish zone",
        "adx_strong_trend": "ADX confirmed trend",
        "adx_crossed_20": "ADX crossed 20",
        "plus_di_dominates": "+DI took control",
        "supertrend_bullish": "Supertrend flipped bullish",
        "accumulation_above_75": "Accumulation spiked",
        "accumulation_above_65": "Accumulation improved",
    }
    return labels.get(signal, signal.replace("_", " ").title())


def _build_setup_observations(
    indicators_df: pd.DataFrame,
    context_start: int,
    setup_start: int,
) -> List[SetupObservation]:
    observations: List[SetupObservation] = []
    seen_groups: set[str] = set()

    def add_observation(pos: int, signal: str, group: str) -> None:
        if group in seen_groups:
            return
        row = indicators_df.iloc[pos]
        detail, value = _observation_detail(signal, row)
        observations.append(
            SetupObservation(
                date=_date_string(indicators_df.index[pos]),
                signal=signal,
                label=_observation_label(signal),
                detail=detail,
                value=value,
            )
        )
        seen_groups.add(group)

    for pos in range(context_start, setup_start + 1):
        row = indicators_df.iloc[pos]
        prev = indicators_df.iloc[pos - 1] if pos > 0 else None

        rel_volume = row.get("rel_volume")
        prev_rel_volume = prev.get("rel_volume") if prev is not None else None
        if pd.notna(rel_volume):
            if rel_volume > 2.0 and (prev is None or pd.isna(prev_rel_volume) or prev_rel_volume <= 2.0):
                add_observation(pos, "volume_breakout_2x", "volume")
            elif rel_volume > 1.5 and (prev is None or pd.isna(prev_rel_volume) or prev_rel_volume <= 1.5):
                add_observation(pos, "volume_breakout_15x", "volume")

        macd_histogram = row.get("macd_histogram")
        prev_histogram = prev.get("macd_histogram") if prev is not None else None
        if pd.notna(macd_histogram) and macd_histogram > 0 and (
            prev is None or pd.isna(prev_histogram) or prev_histogram <= 0
        ):
            add_observation(pos, "macd_histogram_turned_positive", "macd")

        rsi = row.get("rsi")
        prev_rsi = prev.get("rsi") if prev is not None else None
        if pd.notna(rsi) and 50 < rsi < 70 and (
            prev is None or pd.isna(prev_rsi) or not (50 < prev_rsi < 70)
        ):
            add_observation(pos, "rsi_in_bullish_zone", "rsi")

        adx = row.get("adx")
        prev_adx = prev.get("adx") if prev is not None else None
        if pd.notna(adx):
            if adx > 25 and (prev is None or pd.isna(prev_adx) or prev_adx <= 25):
                add_observation(pos, "adx_strong_trend", "adx")
            elif adx > 20 and (prev is None or pd.isna(prev_adx) or prev_adx <= 20):
                add_observation(pos, "adx_crossed_20", "adx")

        plus_di = row.get("plus_di")
        minus_di = row.get("minus_di")
        prev_plus_di = prev.get("plus_di") if prev is not None else None
        prev_minus_di = prev.get("minus_di") if prev is not None else None
        if (
            pd.notna(plus_di)
            and pd.notna(minus_di)
            and plus_di > minus_di
            and (
                prev is None
                or pd.isna(prev_plus_di)
                or pd.isna(prev_minus_di)
                or prev_plus_di <= prev_minus_di
            )
        ):
            add_observation(pos, "plus_di_dominates", "directional")

        supertrend = row.get("supertrend")
        prev_supertrend = prev.get("supertrend") if prev is not None else None
        if pd.notna(supertrend) and int(supertrend) == 1 and (
            prev is None or pd.isna(prev_supertrend) or int(prev_supertrend) != 1
        ):
            add_observation(pos, "supertrend_bullish", "supertrend")

    if len(observations) < 3:
        setup_row = indicators_df.iloc[setup_start]
        active_signals = [
            signal
            for signal in (
                "accumulation_above_75",
                "accumulation_above_65",
                "volume_breakout_2x",
                "volume_breakout_15x",
                "macd_histogram_turned_positive",
                "rsi_in_bullish_zone",
                "adx_strong_trend",
                "adx_crossed_20",
                "plus_di_dominates",
                "supertrend_bullish",
            )
            if signal in _signals_fired(setup_row)
        ]
        for signal in active_signals:
            if len(observations) >= 5:
                break
            group = signal.split("_")[0]
            if group in seen_groups:
                continue
            add_observation(setup_start, signal, group)

    return observations[:5]


def _serialize_setup_bars(
    indicators_df: pd.DataFrame,
    start_pos: int,
    end_pos: int,
) -> List[Dict[str, Any]]:
    bars: List[Dict[str, Any]] = []
    for pos in range(start_pos, end_pos + 1):
        row = indicators_df.iloc[pos]
        bar: Dict[str, Any] = {"date": _date_string(indicators_df.index[pos])}
        for column in DNA_CHART_COLUMNS:
            if column not in indicators_df.columns:
                continue
            value = row.get(column)
            if value is None:
                bar[column] = None
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                bar[column] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                bar[column] = None if np.isnan(value) else float(value)
            else:
                try:
                    bar[column] = float(value)
                except (TypeError, ValueError):
                    bar[column] = value
        bars.append(bar)
    return bars


def _build_setup_examples(
    indicators_df: Optional[pd.DataFrame],
    setup_matches: List[Dict[str, Any]],
    windows: Sequence[int],
    thresholds: Sequence[float],
) -> List[SetupExample]:
    if indicators_df is None or indicators_df.empty or not setup_matches:
        return []

    max_window = max(windows)
    closes = indicators_df["close"].astype(float).to_numpy()
    dates = list(indicators_df.index)

    ranked_matches = sorted(
        setup_matches,
        key=lambda match: (min(max_window, len(indicators_df) - match["pos"] - 1), match["pos"]),
        reverse=True,
    )

    examples: List[SetupExample] = []
    for match in ranked_matches[:DNA_MAX_SETUP_EXAMPLES]:
        context_start = max(0, match["pos"] - DNA_CONTEXT_LOOKBACK_BARS)
        context_end = min(len(indicators_df) - 1, match["pos"] + max_window)
        bars = _serialize_setup_bars(indicators_df, context_start, context_end)
        forward_outcomes: Dict[str, SetupForwardOutcome] = {}
        for window in windows:
            outcome = _window_outcome(closes, dates, match["pos"], int(window), thresholds)
            forward_outcomes[str(window)] = outcome or SetupForwardOutcome(
                horizon_days=int(window),
                completed=False,
                max_gain_pct=None,
                max_gain_date=None,
                threshold_hits=[],
            )

        examples.append(
            SetupExample(
                setup_date=_date_string(match["date"]),
                setup_window_start_date=_date_string(indicators_df.index[match["pos"]]),
                setup_window_end_date=_date_string(indicators_df.index[match["end_pos"]]),
                setup_bar_index=match["pos"] - context_start,
                setup_window_start_index=match["pos"] - context_start,
                setup_window_end_index=match["end_pos"] - context_start,
                available_forward_bars=max(0, len(indicators_df) - match["pos"] - 1),
                bars=bars,
                observations=_build_setup_observations(indicators_df, context_start, match["end_pos"]),
                forward_outcomes=forward_outcomes,
            )
        )

    return examples


def extract_dna(
    ticker: str,
    snapshots: List[ForensicSnapshot],
    fakeout_snapshots: List[ForensicSnapshot],
    indicators_df: Optional[pd.DataFrame] = None,
    horizon_days: Optional[int] = None,
    min_setup_occurrences: int = DNA_CONFIDENCE_FLOOR,
    window_days: Optional[Sequence[int]] = None,
) -> Optional[BehavioralDNA]:
    """Build the BehavioralDNA for one stock from its forensic event library."""

    if len(snapshots) < 3 and indicators_df is None:
        return None

    windows, default_window = _normalize_windows(horizon_days, window_days)
    thresholds = [float(t) for t in CONFIG.MOVE_THRESHOLDS_PCT if float(t) > 0]

    real_moves = [snapshot for snapshot in snapshots if not snapshot.event.is_fakeout]
    fakeouts = [snapshot for snapshot in snapshots if snapshot.event.is_fakeout] + fakeout_snapshots
    if not real_moves:
        return None

    overall_reliability = _aggregate_signal_reliability(real_moves, fakeouts)
    setup_signals, _ = _select_setup_signals(
        indicators_df=indicators_df,
        reliability=overall_reliability,
        min_setup_occurrences=min_setup_occurrences,
        horizon_days=default_window,
    )
    setup_matches = _find_setup_matches(indicators_df, setup_signals)

    window_profiles: List[SetupWindowProfile] = []
    occurrences_by_window: Dict[int, List[Dict[str, Any]]] = {}
    profiles_by_window: Dict[int, List[ThresholdProfile]] = {}

    for window in windows:
        setup_occurrences = _collect_setup_occurrences(indicators_df, setup_matches, window, thresholds)
        occurrences_by_window[window] = setup_occurrences
        confidence_tier, confidence_label, percentages_visible, history_status = _confidence_meta(
            len(setup_occurrences),
            min_setup_occurrences,
        )
        threshold_profiles = (
            _build_threshold_profiles(setup_occurrences, thresholds, real_moves, fakeouts)
            if percentages_visible
            else []
        )
        profiles_by_window[window] = threshold_profiles
        window_profiles.append(
            SetupWindowProfile(
                horizon_days=window,
                setup_count=len(setup_occurrences),
                history_status=history_status,
                confidence_floor=min_setup_occurrences,
                confidence_tier=confidence_tier,
                confidence_label=confidence_label,
                percentages_visible=percentages_visible,
                threshold_profiles=threshold_profiles,
            )
        )

    default_occurrences = occurrences_by_window.get(default_window, [])
    setup_signal_stats = _build_setup_signal_stats(default_occurrences)
    profiles = profiles_by_window.get(default_window, [])

    top_overall = sorted(
        overall_reliability.values(),
        key=lambda reliability: -reliability.discriminative_power,
    )[:8]

    fakeout_sigs = []
    for sig, reliability in overall_reliability.items():
        if reliability.false_positive_rate > 40 and reliability.discriminative_power < 10:
            fakeout_sigs.append(
                f"{sig} fires in {reliability.false_positive_rate:.0f}% of fakeouts vs {reliability.reliability_pct:.0f}% of real moves."
            )

    dna_inputs = {
        "avg_consolidation": float(np.mean([snapshot.event.days_consolidating_before for snapshot in real_moves])),
        "avg_duration": float(np.mean([snapshot.event.duration_days for snapshot in real_moves])),
        "avg_magnitude": float(np.mean([snapshot.event.gain_pct for snapshot in real_moves])),
    }

    pre_move_volume_profile = _compute_volume_profile(real_moves)
    fakeout_volume_profile = _compute_volume_profile(fakeouts) if fakeouts else {}

    default_history_status = next(
        (profile.history_status for profile in window_profiles if profile.horizon_days == default_window),
        "INSUFFICIENT_HISTORY",
    )

    return BehavioralDNA(
        ticker=ticker,
        total_events_studied=len(default_occurrences),
        fakeouts_studied=len(fakeouts),
        history_status=default_history_status,
        profiles_by_threshold=profiles,
        personality_tag=_classify_personality(dna_inputs),
        avg_pre_move_consolidation_days=dna_inputs["avg_consolidation"],
        avg_move_duration_days=dna_inputs["avg_duration"],
        avg_move_magnitude_pct=dna_inputs["avg_magnitude"],
        setup_signals=setup_signals,
        setup_horizon_days=default_window,
        signal_stats=setup_signal_stats,
        most_reliable_signals_overall=top_overall,
        fakeout_signatures=fakeout_sigs,
        pre_move_volume_profile=pre_move_volume_profile,
        fakeout_volume_profile=fakeout_volume_profile,
        available_window_days=list(windows),
        default_window_days=default_window,
        confidence_floor=min_setup_occurrences,
        window_profiles=window_profiles,
        setup_examples=_build_setup_examples(indicators_df, setup_matches, windows, thresholds),
    )


def _serialize_signal(signal: SignalReliability) -> Dict[str, Any]:
    return {
        "signal": signal.signal,
        "avg_lead_days": round(signal.avg_lead_days, 1),
        "fired_count": signal.fired_count,
        "total_events": signal.total_events,
        "reliability_pct": round(signal.reliability_pct, 1),
        "false_positive_rate": round(signal.false_positive_rate, 1),
        "discriminative_power": round(signal.discriminative_power, 1),
    }


def _serialize_threshold_profile(profile: ThresholdProfile) -> Dict[str, Any]:
    return {
        "threshold_pct": profile.threshold_pct,
        "occurrences": profile.occurrences,
        "sample_count": profile.sample_count,
        "fakeouts": profile.fakeouts,
        "success_rate": round(profile.success_rate, 1),
        "avg_consolidation_days": round(profile.avg_consolidation_days, 1),
        "avg_duration_days": round(profile.avg_duration_days, 1),
        "avg_gain_pct": _round_optional(profile.avg_gain_pct, 1),
        "avg_gain_all_pct": round(profile.avg_gain_all_pct, 1),
        "avg_gain_on_hits_pct": _round_optional(profile.avg_gain_on_hits_pct, 1),
        "earliest_reliable_signals": [_serialize_signal(signal) for signal in profile.earliest_reliable_signals],
        "confirmation_signals": [_serialize_signal(signal) for signal in profile.confirmation_signals],
    }


def dna_to_dict(dna: BehavioralDNA) -> Dict[str, Any]:
    """Serialize to a JSON-friendly dict."""
    return {
        "ticker": dna.ticker,
        "personality_tag": dna.personality_tag,
        "total_events_studied": dna.total_events_studied,
        "fakeouts_studied": dna.fakeouts_studied,
        "history_status": dna.history_status,
        "setup_signals": dna.setup_signals,
        "setup_horizon_days": dna.setup_horizon_days,
        "default_window_days": dna.default_window_days,
        "available_window_days": dna.available_window_days,
        "confidence_floor": dna.confidence_floor,
        "avg_pre_move_consolidation_days": round(dna.avg_pre_move_consolidation_days, 1),
        "avg_move_duration_days": round(dna.avg_move_duration_days, 1),
        "avg_move_magnitude_pct": round(dna.avg_move_magnitude_pct, 1),
        "signal_stats": [
            {
                "signal": stat.signal,
                "fired_count": stat.fired_count,
                "total_setups": stat.total_setups,
                "presence_pct": round(stat.presence_pct, 1),
            }
            for stat in dna.signal_stats
        ],
        "most_reliable_signals_overall": [_serialize_signal(signal) for signal in dna.most_reliable_signals_overall],
        "profiles_by_threshold": [_serialize_threshold_profile(profile) for profile in dna.profiles_by_threshold],
        "window_profiles": [
            {
                "horizon_days": profile.horizon_days,
                "setup_count": profile.setup_count,
                "history_status": profile.history_status,
                "confidence_floor": profile.confidence_floor,
                "confidence_tier": profile.confidence_tier,
                "confidence_label": profile.confidence_label,
                "percentages_visible": profile.percentages_visible,
                "threshold_profiles": [
                    _serialize_threshold_profile(threshold_profile)
                    for threshold_profile in profile.threshold_profiles
                ],
            }
            for profile in dna.window_profiles
        ],
        "setup_examples": [
            {
                "setup_date": example.setup_date,
                "setup_window_start_date": example.setup_window_start_date,
                "setup_window_end_date": example.setup_window_end_date,
                "setup_bar_index": example.setup_bar_index,
                "setup_window_start_index": example.setup_window_start_index,
                "setup_window_end_index": example.setup_window_end_index,
                "available_forward_bars": example.available_forward_bars,
                "bars": example.bars,
                "observations": [
                    {
                        "date": observation.date,
                        "signal": observation.signal,
                        "label": observation.label,
                        "detail": observation.detail,
                        "value": observation.value,
                    }
                    for observation in example.observations
                ],
                "forward_outcomes": {
                    key: {
                        "horizon_days": outcome.horizon_days,
                        "completed": outcome.completed,
                        "max_gain_pct": _round_optional(outcome.max_gain_pct, 1),
                        "max_gain_date": outcome.max_gain_date,
                        "threshold_hits": outcome.threshold_hits,
                    }
                    for key, outcome in example.forward_outcomes.items()
                },
            }
            for example in dna.setup_examples
        ],
        "fakeout_signatures": dna.fakeout_signatures,
        "pre_move_volume_profile": dna.pre_move_volume_profile,
        "fakeout_volume_profile": dna.fakeout_volume_profile,
    }
