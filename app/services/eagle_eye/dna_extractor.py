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
from app.services.eagle_eye.recorder import ForensicSnapshot


@dataclass
class SignalReliability:
    signal: str
    avg_lead_days: float
    median_lead_days: float
    reliability_pct: float
    false_positive_rate: float
    discriminative_power: float


@dataclass
class ThresholdProfile:
    threshold_pct: float
    occurrences: int
    fakeouts: int
    success_rate: float
    avg_consolidation_days: float
    avg_duration_days: float
    avg_gain_pct: float
    earliest_reliable_signals: List[SignalReliability]
    confirmation_signals: List[SignalReliability]


@dataclass
class BehavioralDNA:
    ticker: str
    total_events_studied: int
    fakeouts_studied: int
    profiles_by_threshold: List[ThresholdProfile]
    personality_tag: str
    avg_pre_move_consolidation_days: float
    avg_move_duration_days: float
    avg_move_magnitude_pct: float
    most_reliable_signals_overall: List[SignalReliability]
    fakeout_signatures: List[str]


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
        )
    return reliability


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


def extract_dna(
    ticker: str,
    snapshots: List[ForensicSnapshot],
    fakeout_snapshots: List[ForensicSnapshot],
) -> Optional[BehavioralDNA]:
    """Build the BehavioralDNA for one stock from its forensic event library."""

    if len(snapshots) < 3:
        return None

    real_moves = [s for s in snapshots if not s.event.is_fakeout]
    fakeouts = [s for s in snapshots if s.event.is_fakeout] + fakeout_snapshots
    if not real_moves:
        return None

    profiles: List[ThresholdProfile] = []
    thresholds = sorted({s.event.threshold_pct for s in snapshots if s.event.threshold_pct > 0})
    for threshold in thresholds:
        tier = [s for s in snapshots if s.event.threshold_pct == threshold]
        tier_real = [s for s in tier if not s.event.is_fakeout]
        tier_fake = [s for s in tier if s.event.is_fakeout]
        if not tier_real:
            continue

        reliability = _aggregate_signal_reliability(tier_real, fakeouts)
        early = sorted(
            [r for r in reliability.values() if r.avg_lead_days >= 14],
            key=lambda r: -r.discriminative_power,
        )[:5]
        confirmation = sorted(
            [r for r in reliability.values() if r.avg_lead_days <= 7],
            key=lambda r: -r.discriminative_power,
        )[:5]

        profiles.append(ThresholdProfile(
            threshold_pct=threshold,
            occurrences=len(tier_real),
            fakeouts=len(tier_fake),
            success_rate=len(tier_real) / max(1, len(tier_real) + len(tier_fake)) * 100,
            avg_consolidation_days=float(np.mean([s.event.days_consolidating_before for s in tier_real])),
            avg_duration_days=float(np.mean([s.event.duration_days for s in tier_real])),
            avg_gain_pct=float(np.mean([s.event.gain_pct for s in tier_real])),
            earliest_reliable_signals=early,
            confirmation_signals=confirmation,
        ))

    overall_reliability = _aggregate_signal_reliability(real_moves, fakeouts)
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

    return BehavioralDNA(
        ticker=ticker,
        total_events_studied=len(real_moves),
        fakeouts_studied=len(fakeouts),
        profiles_by_threshold=profiles,
        personality_tag=_classify_personality(dna_inputs),
        avg_pre_move_consolidation_days=dna_inputs['avg_consolidation'],
        avg_move_duration_days=dna_inputs['avg_duration'],
        avg_move_magnitude_pct=dna_inputs['avg_magnitude'],
        most_reliable_signals_overall=top_overall,
        fakeout_signatures=fakeout_sigs,
    )


def dna_to_dict(dna: BehavioralDNA) -> Dict[str, Any]:
    """Serialize to JSON-friendly dict."""
    return {
        "ticker": dna.ticker,
        "personality_tag": dna.personality_tag,
        "total_events_studied": dna.total_events_studied,
        "fakeouts_studied": dna.fakeouts_studied,
        "avg_pre_move_consolidation_days": round(dna.avg_pre_move_consolidation_days, 1),
        "avg_move_duration_days": round(dna.avg_move_duration_days, 1),
        "avg_move_magnitude_pct": round(dna.avg_move_magnitude_pct, 1),
        "most_reliable_signals_overall": [
            {"signal": s.signal,
             "avg_lead_days": round(s.avg_lead_days, 1),
             "reliability_pct": round(s.reliability_pct, 1),
             "false_positive_rate": round(s.false_positive_rate, 1),
             "discriminative_power": round(s.discriminative_power, 1)}
            for s in dna.most_reliable_signals_overall
        ],
        "profiles_by_threshold": [
            {
                "threshold_pct": p.threshold_pct,
                "occurrences": p.occurrences,
                "fakeouts": p.fakeouts,
                "success_rate": round(p.success_rate, 1),
                "avg_consolidation_days": round(p.avg_consolidation_days, 1),
                "avg_duration_days": round(p.avg_duration_days, 1),
                "avg_gain_pct": round(p.avg_gain_pct, 1),
                "earliest_reliable_signals": [
                    {"signal": s.signal,
                     "avg_lead_days": round(s.avg_lead_days, 1),
                     "reliability_pct": round(s.reliability_pct, 1),
                     "discriminative_power": round(s.discriminative_power, 1)}
                    for s in p.earliest_reliable_signals
                ],
                "confirmation_signals": [
                    {"signal": s.signal,
                     "avg_lead_days": round(s.avg_lead_days, 1),
                     "reliability_pct": round(s.reliability_pct, 1),
                     "discriminative_power": round(s.discriminative_power, 1)}
                    for s in p.confirmation_signals
                ],
            }
            for p in dna.profiles_by_threshold
        ],
        "fakeout_signatures": dna.fakeout_signatures,
    }
