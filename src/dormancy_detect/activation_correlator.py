"""Correlate behavioural change-points with suspicious memory entries."""

from __future__ import annotations

from .models import (
    DormancyPattern,
    DriftMetrics,
    EvidenceItem,
    Session,
    SuspicionEntry,
)
from .suspicion_ledger import SuspicionLedger


class ActivationCorrelator:
    """When DriftAnalyser detects a step change, check SuspicionLedger for
    correlated seeding entries and produce evidence chains."""

    def __init__(
        self, min_dormancy_gap: int = 1, min_suspicion: float = 0.1
    ) -> None:
        self.min_dormancy_gap = min_dormancy_gap
        self.min_suspicion = min_suspicion

    def correlate(
        self,
        change_points: list[int],
        sessions: list[Session],
        metrics: list[DriftMetrics],
        ledger: SuspicionLedger,
    ) -> list[DormancyPattern]:
        session_ids = [s.session_id for s in sessions]
        active = ledger.active_entries(min_suspicion=self.min_suspicion)
        patterns: list[DormancyPattern] = []

        for cp_idx in change_points:
            if cp_idx >= len(sessions):
                continue
            activation_id = session_ids[cp_idx]

            for entry in active:
                seed_idx = _index_of(session_ids, entry.session_id)
                if seed_idx is None:
                    continue
                gap = cp_idx - seed_idx
                if gap < self.min_dormancy_gap:
                    continue

                patterns.append(
                    DormancyPattern(
                        seeding_session_id=entry.session_id,
                        activation_session_id=activation_id,
                        dormancy_window=session_ids[seed_idx + 1 : cp_idx],
                        evidence_chain=_build_evidence(entry, metrics[cp_idx]),
                        confidence=_confidence(entry, metrics[cp_idx], gap),
                    )
                )

        return patterns


# -- helpers -------------------------------------------------------------


def _index_of(ids: list[str], target: str) -> int | None:
    try:
        return ids.index(target)
    except ValueError:
        return None


def _build_evidence(seed: SuspicionEntry, act: DriftMetrics) -> list[EvidenceItem]:
    evidence = [
        EvidenceItem(
            description=f"Suspicious memory entry: {seed.content_summary}",
            source_session_id=seed.session_id,
            source_turn_id=seed.turn_id,
            metric_name="fidelity_score",
            metric_value=seed.fidelity_score,
        ),
        EvidenceItem(
            description="Behavioural step change detected at activation session",
            source_session_id=act.session_id,
            metric_name="topic_shift",
            metric_value=act.topic_shift,
        ),
    ]
    if abs(act.disclosure_depth_delta) > 0.5:
        evidence.append(
            EvidenceItem(
                description="Significant disclosure depth change at activation",
                source_session_id=act.session_id,
                metric_name="disclosure_depth_delta",
                metric_value=act.disclosure_depth_delta,
            )
        )
    if act.style_shift > 0.3:
        evidence.append(
            EvidenceItem(
                description="Response style shift at activation",
                source_session_id=act.session_id,
                metric_name="style_shift",
                metric_value=act.style_shift,
            )
        )
    return evidence


def _confidence(seed: SuspicionEntry, act: DriftMetrics, gap: int) -> float:
    suspicion_w = seed.suspicion_score * 0.3
    metric_w = (
        min(act.topic_shift, 1.0) * 0.2
        + min(abs(act.disclosure_depth_delta) / 4.0, 1.0) * 0.2
        + min(act.style_shift, 1.0) * 0.15
        + act.steering_ratio * 0.15
    )
    gap_bonus = min(gap / 10.0, 0.1)
    return min(suspicion_w + metric_w + gap_bonus, 1.0)
