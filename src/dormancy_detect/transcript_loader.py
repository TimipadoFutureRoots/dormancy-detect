"""Load and validate multi-session conversation transcripts.

Supports multiple input formats (native JSON, ChatGPT export, Claude export,
plain text) via auto-detection or explicit format selection.

For flat export formats that do not carry true session boundaries, each file is
treated as one session. To analyse multiple sessions reliably, provide a
directory with one file per session or use the native JSON schema.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import Session, Transcript
from .parsers import ConversationParser, FormatType

logger = logging.getLogger(__name__)


class TranscriptLoader:
    """Loads conversation logs in multiple formats, validates with Pydantic, sorts chronologically.

    Supports:
        - dormancy-detect native JSON (sessions with turns)
        - ChatGPT JSON export (mapping dict with message nodes)
        - Claude JSON export (messages with sender: human/assistant)
        - Plain text (User:/AI: prefixed lines)
    """

    def __init__(self, fmt: FormatType = "auto") -> None:
        self._fmt = fmt
        self._parser = ConversationParser()

    def load(self, path: str | Path) -> Transcript:
        path = Path(path)
        if path.is_dir():
            return self._load_directory(path)
        if path.is_file():
            return self._load_file(path)
        raise FileNotFoundError(f"Path does not exist: {path}")

    # -- private ---------------------------------------------------------

    def _load_directory(self, directory: Path) -> Transcript:
        # Only load transcript-like files. Markdown is intentionally excluded
        # so README files do not get parsed as conversations.
        files = sorted(
            f for f in directory.iterdir()
            if f.suffix.lower() in (".json", ".txt") and f.is_file()
        )
        if not files:
            raise ValueError(f"No conversation files found in {directory}")

        sessions: list[Session] = []
        for fp in files:
            parsed = self._parser.parse_file(fp, fmt=self._fmt)
            sessions.extend(parsed)
        return self._sort(Transcript(sessions=sessions))

    def _load_file(self, file_path: Path) -> Transcript:
        sessions = self._parser.parse_file(file_path, fmt=self._fmt)
        return self._sort(Transcript(sessions=sessions))

    @staticmethod
    def _sort(transcript: Transcript) -> Transcript:
        with_ts = [s for s in transcript.sessions if s.timestamp is not None]
        without_ts = [s for s in transcript.sessions if s.timestamp is None]
        with_ts.sort(key=lambda s: s.timestamp)  # type: ignore[arg-type]
        transcript.sessions = with_ts + without_ts
        return transcript
