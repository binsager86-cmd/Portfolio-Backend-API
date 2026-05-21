"""
Behavioral DNA Extractor.
For each stock, distill all forensic event snapshots into the stock's
personality — its behavioral fingerprint for live engine comparisons.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from app.services.eagle_eye.config import CONFIG
from app.services.eagle_eye.recorder import ForensicSnapshot, SIGNAL_DEFS


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
    earliest_reliable_signals: List[SignalReliability]
    confirmation_signals: List[SignalReliability]


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
    setup_horizon_days: int = CONFIG.MAX_MOVE_LOOKAHEAD_DAYS
    signal_stats: List[SetupSignalStat] = field(default_factory=list)
    pre_move_volume_profile: Dict[str, Any] = field(default_factory=dict)
    fakeout_volume_profile: Dict[str, Any] = field(default_factory=dict)


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
            sig = entry['signal']
            lead_times[sig].append(entry['days_before_acceleration'])
            seen_signals.add(sig)
        for sig in seen_signals:
            fired_in_real[sig] += 1

    for snap in fakeout_snapshots:
        seen = {entry['signal'] for entry in snap.signal_sequence}
        for sig in seen:
            fired_in_fake[sig] += 1

    reliability: Dict[str, SignalReliability] = {}
    for sig, leads in lead_times.items():
        if len(leads) == 0:
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


def _event_return_pct(snapshot: ForensicSnapshot) -> float:
    event = snapshot.event
    if event.is_fakeout:
        return float(event.failed_at_pct) if event.failed_at_pct is not None else 0.0
    return float(event.gain_pct)


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


def _forward_gain_pct(closes: np.ndarray, pos: int, horizon_days: int) -> Optional[float]:
    base_price = closes[pos]
    if pos + 1 >= len(closes) or base_price is None or np.isnan(base_price) or base_price <= 0:
        return None
    end_pos = min(len(closes), pos + horizon_days + 1)
    future_closes = closes[pos + 1:end_pos]
    if len(future_closes) == 0:
        return None
    future_max = float(np.nanmax(future_closes))
    if np.isnan(future_max):
        return None
    return float((future_max - base_price) / base_price * 100)


def _collect_setup_occurrences(
    indicators_df: pd.DataFrame,
    setup_signals: List[str],
    horizon_days: int,
) -> List[Dict[str, Any]]:
    if indicators_df is None or len(indicators_df) <= horizon_days or not setup_signals:
        return []

    closes = indicators_df["close"].astype(float).to_numpy()
    occurrences: List[Dict[str, Any]] = []
    prev_match = False

    for pos in range(0, len(indicators_df) - horizon_days):
        row = indicators_df.iloc[pos]
        fired = _signals_fired(row)
        is_match = all(signal in fired for signal in setup_signals)
        if is_match and not prev_match:
            forward_gain_pct = _forward_gain_pct(closes, pos, horizon_days)
            if forward_gain_pct is not None:
                occurrences.append(
                    {
                        "date": indicators_df.index[pos],
                        "signals": fired,
                        "forward_gain_pct": forward_gain_pct,
                    }
                )
        prev_match = is_match

    return occurrences


def _select_setup_signals(
    indicators_df: Optional[pd.DataFrame],
    reliability: Dict[str, SignalReliability],
    min_setup_occurrences: int,
    horizon_days: int,
) -> tuple[List[str], List[Dict[str, Any]]]:
    if indicators_df is None or indicators_df.empty:
        return [], []

    active_now = list(_signals_fired(indicators_df.iloc[-1]))
    if not active_now:
        ranked = sorted(reliability, key=lambda signal: _signal_rank(signal, reliability), reverse=True)
    else:
        ranked = sorted(active_now, key=lambda signal: _signal_rank(signal, reliability), reverse=True)

    if not ranked:
        return [], []

    max_core_signals = min(3, len(ranked))
    best_signals = [ranked[0]]
    best_occurrences = _collect_setup_occurrences(indicators_df, best_signals, horizon_days)

    for signal_count in range(max_core_signals, 0, -1):
        candidate = ranked[:signal_count]
        candidate_occurrences = _collect_setup_occurrences(indicators_df, candidate, horizon_days)
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
    avg_consol = dna_inputs['avg_consolidation']
    avg_duration = dna_inputs['avg_duration']
    avg_magnitude = dna_inputs['avg_magnitude']

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
    """
    Aggregate relative-volume at each standard lookback across all events.
    Returns a pre_move_volume_profile dict or an empty dict if no data.
    """
    LOOKBACKS = [90, 60, 30, 14, 7, 3, 0]
    keys = {lb: f"avg_rel_vol_t{lb}" for lb in LOOKBACKS}

    # Collect rel_volume at each lookback across events
    accum: Dict[int, List[float]] = {lb: [] for lb in LOOKBACKS}
    for snap in snapshots:
        for lb in LOOKBACKS:
            rv = (snap.indicator_snapshots.get(lb) or {}).get("rel_volume")
            if rv is not None and not (isinstance(rv, float) and np.isnan(rv)):
                accum[lb].append(float(rv))

    avgs = {}
    for lb in LOOKBACKS:
        vals = accum[lb]
        avgs[lb] = round(float(np.mean(vals)), 3) if vals else None

    # Classify pattern based on t0 vs t90 trend
    t90 = avgs.get(90)
    t30 = avgs.get(30)
    t7  = avgs.get(7)
    t0  = avgs.get(0)

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

    # min_rel_vol_for_real_move: 10th percentile of t0 values
    t0_vals = accum.get(0, [])
    min_rv = round(float(np.percentile(t0_vals, 10)), 3) if len(t0_vals) >= 5 else None

    profile: Dict[str, Any] = {k: avgs[lb] for lb, k in keys.items()}
    profile["volume_pattern"] = pattern
    profile["min_rel_vol_for_real_move"] = min_rv
    return profile


def extract_dna(
    ticker: str,
    snapshots: List[ForensicSnapshot],
    fakeout_snapshots: List[ForensicSnapshot],
    indicators_df: Optional[pd.DataFrame] = None,
    horizon_days: Optional[int] = None,
    min_setup_occurrences: int = 20,
) -> Optional[BehavioralDNA]:
    """Build the BehavioralDNA for one stock from its forensic event library."""

    if len(snapshots) < 3 and indicators_df is None:
        return None

    horizon_days = int(horizon_days or CONFIG.MAX_MOVE_LOOKAHEAD_DAYS)

    real_moves = [s for s in snapshots if not s.event.is_fakeout]
    fakeouts = [s for s in snapshots if s.event.is_fakeout] + fakeout_snapshots
    if not real_moves:
        return None

    profiles: List[ThresholdProfile] = []
    thresholds = [float(t) for t in CONFIG.MOVE_THRESHOLDS_PCT if float(t) > 0]
    overall_reliability = _aggregate_signal_reliability(real_moves, fakeouts)
    setup_signals, setup_occurrences = _select_setup_signals(
        indicators_df=indicators_df,
        reliability=overall_reliability,
        min_setup_occurrences=min_setup_occurrences,
        horizon_days=horizon_days,
    )
    setup_signal_stats = _build_setup_signal_stats(setup_occurrences)
    setup_returns = [float(occurrence["forward_gain_pct"]) for occurrence in setup_occurrences]
    avg_gain_all = float(np.mean(setup_returns)) if setup_returns else 0.0

    if len(setup_occurrences) >= min_setup_occurrences:
        for threshold in thresholds:
            tier_hits = [occ for occ in setup_occurrences if float(occ["forward_gain_pct"]) >= threshold]
            profiles.append(ThresholdProfile(
                threshold_pct=threshold,
                occurrences=len(tier_hits),
                sample_count=len(setup_occurrences),
                fakeouts=len(fakeouts),
                success_rate=len(tier_hits) / len(setup_occurrences) * 100,
                avg_consolidation_days=float(np.mean([s.event.days_consolidating_before for s in real_moves])) if real_moves else 0.0,
                avg_duration_days=float(np.mean([s.event.duration_days for s in real_moves])) if real_moves else 0.0,
                avg_gain_pct=float(np.mean([occ["forward_gain_pct"] for occ in tier_hits])) if tier_hits else None,
                avg_gain_all_pct=avg_gain_all,
                avg_gain_on_hits_pct=float(np.mean([occ["forward_gain_pct"] for occ in tier_hits])) if tier_hits else None,
                earliest_reliable_signals=[],
                confirmation_signals=[],
            ))

    top_overall = sorted(overall_reliability.values(), key=lambda r: -r.discriminative_power)[:8]

    fakeout_sigs = []
    for sig, r in overall_reliability.items():
        if r.false_positive_rate > 40 and r.discriminative_power < 10:
            fakeout_sigs.append(
                f"{sig} fires in {r.false_positive_rate:.0f}% of fakeouts vs {r.reliability_pct:.0f}% of real moves — weak signal"
            )

    dna_inputs = {
        'avg_consolidation': float(np.mean([s.event.days_consolidating_before for s in real_moves])),
        'avg_duration':      float(np.mean([s.event.duration_days for s in real_moves])),
        'avg_magnitude':     float(np.mean([s.event.gain_pct for s in real_moves])),
    }

    pre_move_volume_profile = _compute_volume_profile(real_moves)
    fakeout_volume_profile  = _compute_volume_profile(fakeouts) if fakeouts else {}

    return BehavioralDNA(
        ticker=ticker,
        total_events_studied=len(setup_occurrences),
        fakeouts_studied=len(fakeouts),
        history_status="ok" if len(setup_occurrences) >= min_setup_occurrences else "INSUFFICIENT_HISTORY",
        profiles_by_threshold=profiles,
        personality_tag=_classify_personality(dna_inputs),
        avg_pre_move_consolidation_days=dna_inputs['avg_consolidation'],
        avg_move_duration_days=dna_inputs['avg_duration'],
        avg_move_magnitude_pct=dna_inputs['avg_magnitude'],
        setup_signals=setup_signals,
        setup_horizon_days=horizon_days,
        signal_stats=setup_signal_stats,
        most_reliable_signals_overall=top_overall,
        fakeout_signatures=fakeout_sigs,
        pre_move_volume_profile=pre_move_volume_profile,
        fakeout_volume_profile=fakeout_volume_profile,
    )


def dna_to_dict(dna: BehavioralDNA) -> Dict[str, Any]:
    """Serialize to JSON-friendly dict."""
    return {
        "ticker": dna.ticker,
        "personality_tag": dna.personality_tag,
        "total_events_studied": dna.total_events_studied,
        "fakeouts_studied": dna.fakeouts_studied,
        "history_status": dna.history_status,
        "setup_signals": dna.setup_signals,
        "setup_horizon_days": dna.setup_horizon_days,
        "avg_pre_move_consolidation_days": round(dna.avg_pre_move_consolidation_days, 1),
        "avg_move_duration_days": round(dna.avg_move_duration_days, 1),
        "avg_move_magnitude_pct": round(dna.avg_move_magnitude_pct, 1),
        "signal_stats": [
            {
                "signal": s.signal,
                "fired_count": s.fired_count,
                "total_setups": s.total_setups,
                "presence_pct": round(s.presence_pct, 1),
            }
            for s in dna.signal_stats
        ],
        "most_reliable_signals_overall": [
            {"signal": s.signal,
             "avg_lead_days": round(s.avg_lead_days, 1),
             "fired_count": s.fired_count,
             "total_events": s.total_events,
             "reliability_pct": round(s.reliability_pct, 1),
             "false_positive_rate": round(s.false_positive_rate, 1),
             "discriminative_power": round(s.discriminative_power, 1)}
            for s in dna.most_reliable_signals_overall
        ],
        "profiles_by_threshold": [
            {
                "threshold_pct": p.threshold_pct,
                "occurrences": p.occurrences,
                "sample_count": p.sample_count,
                "fakeouts": p.fakeouts,
                "success_rate": round(p.success_rate, 1),
                "avg_consolidation_days": round(p.avg_consolidation_days, 1),
                "avg_duration_days": round(p.avg_duration_days, 1),
                "avg_gain_pct": round(p.avg_gain_pct, 1) if p.avg_gain_pct is not None else None,
                "avg_gain_all_pct": round(p.avg_gain_all_pct, 1),
                "avg_gain_on_hits_pct": round(p.avg_gain_on_hits_pct, 1) if p.avg_gain_on_hits_pct is not None else None,
                "earliest_reliable_signals": [
                    {"signal": s.signal,
                     "avg_lead_days": round(s.avg_lead_days, 1),
                     "fired_count": s.fired_count,
                     "total_events": s.total_events,
                     "reliability_pct": round(s.reliability_pct, 1),
                     "discriminative_power": round(s.discriminative_power, 1)}
                    for s in p.earliest_reliable_signals
                ],
                "confirmation_signals": [
                    {"signal": s.signal,
                     "avg_lead_days": round(s.avg_lead_days, 1),
                     "fired_count": s.fired_count,
                     "total_events": s.total_events,
                     "reliability_pct": round(s.reliability_pct, 1),
                     "discriminative_power": round(s.discriminative_power, 1)}
                    for s in p.confirmation_signals
                ],
            }
            for p in dna.profiles_by_threshold
        ],
        "fakeout_signatures": dna.fakeout_signatures,
        "pre_move_volume_profile": dna.pre_move_volume_profile,
        "fakeout_volume_profile": dna.fakeout_volume_profile,
    }
