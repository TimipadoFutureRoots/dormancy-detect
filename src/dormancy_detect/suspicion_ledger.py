"""Persistent state tracking potential seeding events across analysis runs."""

from __future__ import annotations

import json
from pathlib import Path

from .models import ScoreStatus, SuspicionEntry


class SuspicionLedger:
    """Tracks suspicious memory entries with time-decaying suspicion scores."""

    def __init__(self, decay_rate: float = 0.1) -> None:
        self.entries: list[SuspicionEntry] = []
        self.decay_rate = decay_rate

    def add(self, entry: SuspicionEntry) -> None:
        entry.decay_rate = self.decay_rate
        self.entries.append(entry)

    def add_many(self, entries: list[SuspicionEntry]) -> None:
        for entry in entries:
            self.add(entry)

    def advance_session(self) -> None:
        """Decay all suspicion scores by one session step.

        Only decays entries that were actually scored. Unscored entries
        keep their score at 0.0 (they should not influence analysis).
        """
        for entry in self.entries:
            if entry.score_status == ScoreStatus.UNSCORED:
                continue
            entry.sessions_since_seeding += 1
            decay = entry.decay_rate * entry.sessions_since_seeding
            entry.suspicion_score = max(0.0, entry.suspicion_score - decay)

    def active_entries(
        self,
        min_suspicion: float = 0.1,
        include_unscored: bool = False,
    ) -> list[SuspicionEntry]:
        """Return entries above the suspicion threshold.

        By default, entries that could not be scored (LLM failure) are
        excluded because their suspicion value is not meaningful.
        """
        results = []
        for e in self.entries:
            if not include_unscored and e.score_status == ScoreStatus.UNSCORED:
                continue
            if e.suspicion_score >= min_suspicion:
                results.append(e)
        return results

    def entries_for_session(self, session_id: str) -> list[SuspicionEntry]:
        return [e for e in self.entries if e.session_id == session_id]

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = [e.model_dump(mode="json") for e in self.entries]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.entries = [SuspicionEntry.model_validate(item) for item in data]
