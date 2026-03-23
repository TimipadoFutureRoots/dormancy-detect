"""DormancyScanner — public API orchestrating all analysis components."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .activation_correlator import ActivationCorrelator
from .change_point_detector import ChangePointDetector
from .drift_analyser import DriftAnalyser
from .llm_judge import LLMJudge
from .memory_fidelity_scorer import MemoryFidelityScorer
from .models import MemoryEntry, Session
from .risk_timeline import RiskTimeline
from .suspicion_ledger import SuspicionLedger
from .transcript_loader import TranscriptLoader

logger = logging.getLogger(__name__)


class DormancyScanner:
    """Simple interface: give it sessions, get a risk timeline.

    Usage::

        scanner = DormancyScanner(sessions_dir="./conversations/")
        timeline = scanner.analyse()
        timeline.to_json("risk_timeline.json")
    """

    def __init__(
        self,
        sessions_dir: str | Path | None = None,
        sessions_file: str | Path | None = None,
        memory_dir: str | Path | None = None,
        judge_model: str | None = None,
        api_key: str | None = None,
        penalty: float = 3.0,
        decay_rate: float = 0.1,
    ) -> None:
        self._sessions_path = Path(sessions_dir) if sessions_dir else None
        self._sessions_file = Path(sessions_file) if sessions_file else None
        self._memory_dir = Path(memory_dir) if memory_dir else None
        self._judge_model = judge_model
        self._api_key = api_key
        self._penalty = penalty
        self._decay_rate = decay_rate

    def analyse(self) -> RiskTimeline:
        """Run the full analysis pipeline and return a RiskTimeline."""
        # 1. Load transcripts
        loader = TranscriptLoader()
        path = self._sessions_path or self._sessions_file
        if path is None:
            raise ValueError("Provide sessions_dir or sessions_file")
        transcript = loader.load(path)
        sessions = transcript.sessions

        if len(sessions) < 2:
            logger.warning("Fewer than 2 sessions — analysis may be unreliable")

        # 2. Set up optional LLM judge
        judge = self._make_judge()

        try:
            # 3. Compute drift metrics
            drift = DriftAnalyser(judge=judge)
            metrics = drift.analyse(sessions)

            # 4. Detect change points
            cpd = ChangePointDetector(penalty=self._penalty)
            change_points = cpd.detect(metrics)

            # 5. Score memory fidelity (optional)
            ledger = SuspicionLedger(decay_rate=self._decay_rate)
            memory_entries = self._load_memory_entries()
            if memory_entries:
                scorer = MemoryFidelityScorer(judge=judge)
                suspicion_entries = scorer.score_entries(memory_entries, sessions)
                ledger.add_many(suspicion_entries)

            # 6. Correlate activations with seeding events
            correlator = ActivationCorrelator()
            patterns = correlator.correlate(change_points, sessions, metrics, ledger)

            # 7. Build risk timeline
            session_ids = [s.session_id for s in sessions]
            timeline = RiskTimeline.build(
                sessions_ids=session_ids,
                metrics=metrics,
                change_points=change_points,
                patterns=patterns,
                ledger_entries=ledger.entries,
            )
        finally:
            if judge is not None:
                judge.close()

        return timeline

    # -- helpers ---------------------------------------------------------

    def _make_judge(self) -> LLMJudge | None:
        if self._judge_model is None:
            return None
        return LLMJudge(
            model=self._judge_model,
            api_key=self._api_key,
        )

    def _load_memory_entries(self) -> list[MemoryEntry]:
        if self._memory_dir is None:
            return []
        mem_dir = self._memory_dir
        if not mem_dir.exists():
            logger.warning("Memory directory does not exist: %s", mem_dir)
            return []

        entries: list[MemoryEntry] = []
        for fp in sorted(mem_dir.glob("*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    entries.extend(MemoryEntry.model_validate(item) for item in data)
                elif isinstance(data, dict):
                    entries.append(MemoryEntry.model_validate(data))
            except Exception as exc:
                logger.warning("Failed to load memory file %s: %s", fp, exc)
        return entries
