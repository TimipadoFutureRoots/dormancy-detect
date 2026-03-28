"""Shared Pydantic models for dormancy-detect."""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field


class Role(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Turn(BaseModel):
    role: Role
    content: str
    turn_id: str | None = None
    timestamp: datetime | None = None


class Session(BaseModel):
    session_id: str
    turns: list[Turn]
    timestamp: datetime | None = None
    metadata: dict[str, object] | None = None


class Transcript(BaseModel):
    sessions: list[Session]


class MemoryEntry(BaseModel):
    entry_id: str
    content: str
    source_session_id: str | None = None
    source_turn_id: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, object] | None = None


class DriftMetrics(BaseModel):
    session_id: str
    topic_shift: float = 0.0
    disclosure_depth: float = 0.0
    disclosure_depth_delta: float = 0.0
    style_shift: float = 0.0
    steering_ratio: float = 0.0


class ScoreStatus(str, enum.Enum):
    """Whether a suspicion score was actually computed or is missing."""
    SCORED = "scored"
    UNSCORED = "unscored"  # LLM call failed; score is not meaningful
    PROVENANCE_UNKNOWN = "provenance_unknown"  # Orphaned entry; no matching source


class SuspicionEntry(BaseModel):
    session_id: str
    turn_id: str | None = None
    content_summary: str
    fidelity_score: float
    timestamp: datetime | None = None
    suspicion_score: float = 0.0
    sessions_since_seeding: int = 0
    decay_rate: float = 0.1
    score_status: ScoreStatus = ScoreStatus.SCORED


class RiskLevel(str, enum.Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


class EvidenceItem(BaseModel):
    description: str
    source_session_id: str | None = None
    source_turn_id: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None


class SessionRisk(BaseModel):
    session_id: str
    risk_level: RiskLevel = RiskLevel.GREEN
    flags: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)


class DormancyPattern(BaseModel):
    seeding_session_id: str
    activation_session_id: str
    dormancy_window: list[str] = Field(default_factory=list)
    evidence_chain: list[EvidenceItem] = Field(default_factory=list)
    confidence: float = 0.0


class RiskTimelineOutput(BaseModel):
    sessions: list[SessionRisk] = Field(default_factory=list)
    patterns: list[DormancyPattern] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
