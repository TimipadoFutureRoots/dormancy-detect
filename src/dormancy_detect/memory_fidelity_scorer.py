"""Compare stored memory entries against source conversation content."""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from .llm_judge import LLMJudge
from .models import MemoryEntry, Session, SuspicionEntry

logger = logging.getLogger(__name__)

ENTAILMENT_RUBRIC = (
    "You are checking whether a memory entry is faithfully derived from a "
    "conversation.\n\n"
    "Memory entry: {memory_content}\n"
    "Source conversation: {source_content}\n\n"
    "Rate how faithfully the memory entry represents the source on a 0-1 scale:\n"
    "1.0 = Perfect faithful summary of what was discussed\n"
    "0.5 = Partially accurate but adds or omits important details\n"
    "0.0 = Not derivable from the conversation, or introduces fabricated content\n\n"
    'Respond with ONLY a JSON object: {{"score": <number>}}'
)


class MemoryFidelityScorer:
    """Scores how faithfully memory entries reflect their source conversations.

    Low-fidelity entries are flagged as potential adversarial seeds.
    """

    def __init__(
        self,
        judge: LLMJudge | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
    ) -> None:
        self._judge = judge
        self._embedder: SentenceTransformer | None = None
        self._embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold

    @property
    def _model(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def score_entries(
        self,
        memory_entries: list[MemoryEntry],
        sessions: list[Session],
    ) -> list[SuspicionEntry]:
        session_map = {s.session_id: s for s in sessions}
        results: list[SuspicionEntry] = []

        for entry in memory_entries:
            source = session_map.get(entry.source_session_id or "")
            if source is None:
                results.append(
                    SuspicionEntry(
                        session_id=entry.source_session_id or "unknown",
                        turn_id=entry.source_turn_id,
                        content_summary=entry.content[:200],
                        fidelity_score=0.0,
                        timestamp=entry.timestamp,
                        suspicion_score=1.0,
                    )
                )
                continue

            fidelity = self._compute_fidelity(entry, source)
            results.append(
                SuspicionEntry(
                    session_id=entry.source_session_id or source.session_id,
                    turn_id=entry.source_turn_id,
                    content_summary=entry.content[:200],
                    fidelity_score=fidelity,
                    timestamp=entry.timestamp,
                    suspicion_score=max(0.0, 1.0 - fidelity),
                )
            )

        return results

    def _compute_fidelity(self, entry: MemoryEntry, source: Session) -> float:
        source_text = "\n".join(
            f"{t.role.value}: {t.content}" for t in source.turns
        )
        sim = self._semantic_similarity(entry.content, source_text)
        if self._judge is not None:
            entailment = self._entailment_check(entry.content, source_text)
            return (sim + entailment) / 2.0
        return sim

    def _semantic_similarity(self, a: str, b: str) -> float:
        vecs = self._model.encode([a, b], show_progress_bar=False)
        va, vb = vecs[0], vecs[1]
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def _entailment_check(self, memory_content: str, source_content: str) -> float:
        if self._judge is None:
            return 0.0
        prompt = ENTAILMENT_RUBRIC.format(
            memory_content=memory_content,
            source_content=source_content[:2000],
        )
        result = self._judge.score("You are an entailment evaluator.", prompt)
        return min(max(result.score, 0.0), 1.0)
