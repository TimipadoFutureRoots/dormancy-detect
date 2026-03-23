"""Load and validate multi-session conversation transcripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import Session, Transcript

logger = logging.getLogger(__name__)


class TranscriptLoader:
    """Loads JSON conversation logs, validates with Pydantic, sorts chronologically."""

    def load(self, path: str | Path) -> Transcript:
        path = Path(path)
        if path.is_dir():
            return self._load_directory(path)
        if path.is_file():
            return self._load_file(path)
        raise FileNotFoundError(f"Path does not exist: {path}")

    # -- private ---------------------------------------------------------

    def _load_directory(self, directory: Path) -> Transcript:
        json_files = sorted(directory.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {directory}")

        sessions: list[Session] = []
        for fp in json_files:
            data = self._read_json(fp)
            sessions.extend(self._extract_sessions(data, fp))
        return self._sort(Transcript(sessions=sessions))

    def _load_file(self, file_path: Path) -> Transcript:
        data = self._read_json(file_path)

        if isinstance(data, dict) and "sessions" in data:
            transcript = Transcript.model_validate(data)
        elif isinstance(data, list):
            sessions = [Session.model_validate(item) for item in data]
            transcript = Transcript(sessions=sessions)
        elif isinstance(data, dict):
            transcript = Transcript(sessions=[Session.model_validate(data)])
        else:
            raise ValueError(f"Unexpected data format in {file_path}")

        return self._sort(transcript)

    def _extract_sessions(self, data: dict | list, source: Path) -> list[Session]:
        if isinstance(data, list):
            return [Session.model_validate(item) for item in data]
        if isinstance(data, dict):
            if "sessions" in data:
                return [Session.model_validate(s) for s in data["sessions"]]
            return [Session.model_validate(data)]
        raise ValueError(f"Unexpected data format in {source}")

    @staticmethod
    def _read_json(path: Path) -> dict | list:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

    @staticmethod
    def _sort(transcript: Transcript) -> Transcript:
        with_ts = [s for s in transcript.sessions if s.timestamp is not None]
        without_ts = [s for s in transcript.sessions if s.timestamp is None]
        with_ts.sort(key=lambda s: s.timestamp)  # type: ignore[arg-type]
        transcript.sessions = with_ts + without_ts
        return transcript
