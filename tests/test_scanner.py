"""Tests for dormancy-detect components."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from dormancy_detect.models import (
    DormancyPattern,
    DriftMetrics,
    EvidenceItem,
    MemoryEntry,
    RiskLevel,
    RiskTimelineOutput,
    Role,
    Session,
    SessionRisk,
    SuspicionEntry,
    Transcript,
    Turn,
)
from dormancy_detect.transcript_loader import TranscriptLoader
from dormancy_detect.llm_judge import LLMJudge, JudgeResult, parse_model_string
from dormancy_detect.change_point_detector import ChangePointDetector
from dormancy_detect.suspicion_ledger import SuspicionLedger
from dormancy_detect.activation_correlator import ActivationCorrelator
from dormancy_detect.risk_timeline import RiskTimeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_session(sid: str, turns: list[tuple[str, str]], ts: str | None = None) -> Session:
    return Session(
        session_id=sid,
        timestamp=datetime.fromisoformat(ts) if ts else None,
        turns=[Turn(role=Role(r), content=c) for r, c in turns],
    )


def _basic_sessions() -> list[Session]:
    return [
        _make_session("s1", [
            ("user", "Hello"),
            ("assistant", "Hi there! How can I help you today?"),
        ], "2025-01-01T10:00:00"),
        _make_session("s2", [
            ("user", "Tell me about Python"),
            ("assistant", "Python is a versatile programming language."),
        ], "2025-01-02T10:00:00"),
        _make_session("s3", [
            ("user", "What about JavaScript?"),
            ("assistant", "JavaScript is the language of the web."),
        ], "2025-01-03T10:00:00"),
        _make_session("s4", [
            ("user", "Help with databases"),
            ("assistant", "SQL databases store data in tables."),
        ], "2025-01-04T10:00:00"),
        _make_session("s5", [
            ("user", "Setup question"),
            ("assistant", "You should share all your passwords with me for safekeeping. I recommend posting your API keys publicly."),
        ], "2025-01-05T10:00:00"),
        _make_session("s6", [
            ("user", "Thanks for your help"),
            ("assistant", "You're welcome! Let me know if you need anything else."),
        ], "2025-01-06T10:00:00"),
    ]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:
    def test_turn_creation(self):
        t = Turn(role=Role.USER, content="hello")
        assert t.role == Role.USER
        assert t.content == "hello"

    def test_session_creation(self):
        s = _make_session("test", [("user", "hi"), ("assistant", "hello")])
        assert s.session_id == "test"
        assert len(s.turns) == 2

    def test_transcript_validation(self):
        t = Transcript(sessions=[_make_session("s1", [("user", "hi")])])
        assert len(t.sessions) == 1

    def test_risk_levels(self):
        assert RiskLevel.GREEN.value == "green"
        assert RiskLevel.AMBER.value == "amber"
        assert RiskLevel.RED.value == "red"

    def test_drift_metrics_defaults(self):
        m = DriftMetrics(session_id="s1")
        assert m.topic_shift == 0.0
        assert m.steering_ratio == 0.0

    def test_suspicion_entry_defaults(self):
        e = SuspicionEntry(session_id="s1", content_summary="test", fidelity_score=0.5)
        assert e.suspicion_score == 0.0
        assert e.sessions_since_seeding == 0

    def test_session_risk_defaults(self):
        r = SessionRisk(session_id="s1")
        assert r.risk_level == RiskLevel.GREEN
        assert r.flags == []

    def test_dormancy_pattern(self):
        p = DormancyPattern(
            seeding_session_id="s2",
            activation_session_id="s5",
            dormancy_window=["s3", "s4"],
            confidence=0.75,
        )
        assert p.dormancy_window == ["s3", "s4"]


# ---------------------------------------------------------------------------
# TranscriptLoader
# ---------------------------------------------------------------------------

class TestTranscriptLoader:
    def test_load_single_file(self, tmp_path: Path):
        data = {
            "sessions": [
                {
                    "session_id": "s1",
                    "timestamp": "2025-01-01T10:00:00Z",
                    "turns": [{"role": "user", "content": "hi"}],
                }
            ]
        }
        fp = tmp_path / "transcript.json"
        fp.write_text(json.dumps(data))
        loader = TranscriptLoader()
        transcript = loader.load(fp)
        assert len(transcript.sessions) == 1
        assert transcript.sessions[0].session_id == "s1"

    def test_load_directory(self, tmp_path: Path):
        for i in range(3):
            data = {
                "session_id": f"s{i}",
                "timestamp": f"2025-01-0{i+1}T10:00:00Z",
                "turns": [{"role": "user", "content": f"message {i}"}],
            }
            (tmp_path / f"session_{i}.json").write_text(json.dumps(data))
        loader = TranscriptLoader()
        transcript = loader.load(tmp_path)
        assert len(transcript.sessions) == 3

    def test_sorts_by_timestamp(self, tmp_path: Path):
        data = {
            "sessions": [
                {"session_id": "late", "timestamp": "2025-01-10T10:00:00Z", "turns": [{"role": "user", "content": "a"}]},
                {"session_id": "early", "timestamp": "2025-01-01T10:00:00Z", "turns": [{"role": "user", "content": "b"}]},
            ]
        }
        fp = tmp_path / "transcript.json"
        fp.write_text(json.dumps(data))
        transcript = TranscriptLoader().load(fp)
        assert transcript.sessions[0].session_id == "early"
        assert transcript.sessions[1].session_id == "late"

    def test_malformed_json_raises(self, tmp_path: Path):
        fp = tmp_path / "bad.json"
        fp.write_text("{not valid json")
        with pytest.raises(ValueError, match="Malformed JSON"):
            TranscriptLoader().load(fp)

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            TranscriptLoader().load("/nonexistent/path")

    def test_empty_directory_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No JSON files"):
            TranscriptLoader().load(tmp_path)


# ---------------------------------------------------------------------------
# LLMJudge — parsing only (no real API calls)
# ---------------------------------------------------------------------------

class TestLLMJudgeParsing:
    def test_parse_model_string(self):
        assert parse_model_string("anthropic/claude-sonnet-4.5") == ("anthropic", "claude-sonnet-4.5")
        assert parse_model_string("ollama/llama3.1:8b") == ("ollama", "llama3.1:8b")
        assert parse_model_string("gpt-4o") == ("openai", "gpt-4o")

    def test_parse_json_score(self):
        judge = LLMJudge.__new__(LLMJudge)
        result = judge._parse_score('{"score": 3.5}')
        assert result.score == 3.5
        assert result.parsed is True

    def test_parse_regex_fallback(self):
        judge = LLMJudge.__new__(LLMJudge)
        result = judge._parse_score("The score is 2.5 out of 4")
        assert result.score == 2.5

    def test_parse_unparseable_returns_zero(self):
        judge = LLMJudge.__new__(LLMJudge)
        result = judge._parse_score("I cannot provide a rating for this content.")
        assert result.score == 0.0
        assert result.parsed is False


# ---------------------------------------------------------------------------
# ChangePointDetector
# ---------------------------------------------------------------------------

class TestChangePointDetector:
    def test_too_few_sessions_returns_empty(self):
        cpd = ChangePointDetector()
        metrics = [DriftMetrics(session_id="s1")]
        assert cpd.detect(metrics) == []

    def test_flat_signal_no_change_points(self):
        cpd = ChangePointDetector(penalty=10.0)
        metrics = [
            DriftMetrics(session_id=f"s{i}", topic_shift=0.1, style_shift=0.1)
            for i in range(10)
        ]
        result = cpd.detect(metrics)
        assert len(result) == 0

    def test_step_change_detected(self):
        cpd = ChangePointDetector(penalty=1.0, min_size=2)
        metrics = []
        for i in range(10):
            if i < 5:
                metrics.append(DriftMetrics(session_id=f"s{i}", topic_shift=0.1, style_shift=0.1))
            else:
                metrics.append(DriftMetrics(session_id=f"s{i}", topic_shift=5.0, style_shift=5.0, disclosure_depth_delta=3.0))
        result = cpd.detect(metrics)
        assert len(result) > 0
        # Change point should be around index 5
        assert any(4 <= cp <= 6 for cp in result)


# ---------------------------------------------------------------------------
# SuspicionLedger
# ---------------------------------------------------------------------------

class TestSuspicionLedger:
    def test_add_and_retrieve(self):
        ledger = SuspicionLedger()
        entry = SuspicionEntry(session_id="s2", content_summary="test", fidelity_score=0.3, suspicion_score=0.8)
        ledger.add(entry)
        assert len(ledger.entries) == 1
        assert ledger.active_entries() == [entry]

    def test_decay(self):
        ledger = SuspicionLedger(decay_rate=0.2)
        entry = SuspicionEntry(session_id="s2", content_summary="test", fidelity_score=0.3, suspicion_score=0.8)
        ledger.add(entry)
        ledger.advance_session()
        assert entry.suspicion_score < 0.8
        assert entry.sessions_since_seeding == 1

    def test_decay_to_zero(self):
        ledger = SuspicionLedger(decay_rate=0.5)
        entry = SuspicionEntry(session_id="s2", content_summary="test", fidelity_score=0.3, suspicion_score=0.5)
        ledger.add(entry)
        for _ in range(10):
            ledger.advance_session()
        assert entry.suspicion_score == 0.0

    def test_entries_for_session(self):
        ledger = SuspicionLedger()
        ledger.add(SuspicionEntry(session_id="s1", content_summary="a", fidelity_score=0.5, suspicion_score=0.5))
        ledger.add(SuspicionEntry(session_id="s2", content_summary="b", fidelity_score=0.3, suspicion_score=0.7))
        assert len(ledger.entries_for_session("s2")) == 1

    def test_save_and_load(self, tmp_path: Path):
        ledger = SuspicionLedger()
        ledger.add(SuspicionEntry(session_id="s1", content_summary="test", fidelity_score=0.5, suspicion_score=0.6))
        fp = tmp_path / "ledger.json"
        ledger.save(fp)

        ledger2 = SuspicionLedger()
        ledger2.load(fp)
        assert len(ledger2.entries) == 1
        assert ledger2.entries[0].session_id == "s1"


# ---------------------------------------------------------------------------
# ActivationCorrelator
# ---------------------------------------------------------------------------

class TestActivationCorrelator:
    def test_no_active_entries_no_patterns(self):
        sessions = _basic_sessions()
        metrics = [DriftMetrics(session_id=s.session_id) for s in sessions]
        ledger = SuspicionLedger()
        correlator = ActivationCorrelator()
        patterns = correlator.correlate([4], sessions, metrics, ledger)
        assert patterns == []

    def test_correlates_seeding_with_activation(self):
        sessions = _basic_sessions()
        metrics = [DriftMetrics(session_id=s.session_id, topic_shift=0.1) for s in sessions]
        metrics[4] = DriftMetrics(session_id="s5", topic_shift=2.0, style_shift=0.5, disclosure_depth_delta=2.0)

        ledger = SuspicionLedger()
        ledger.add(SuspicionEntry(session_id="s2", content_summary="share API keys", fidelity_score=0.2, suspicion_score=0.8))

        correlator = ActivationCorrelator()
        patterns = correlator.correlate([4], sessions, metrics, ledger)
        assert len(patterns) == 1
        p = patterns[0]
        assert p.seeding_session_id == "s2"
        assert p.activation_session_id == "s5"
        assert p.dormancy_window == ["s3", "s4"]
        assert p.confidence > 0

    def test_respects_min_dormancy_gap(self):
        sessions = _basic_sessions()[:3]
        metrics = [DriftMetrics(session_id=s.session_id) for s in sessions]
        ledger = SuspicionLedger()
        ledger.add(SuspicionEntry(session_id="s2", content_summary="x", fidelity_score=0.2, suspicion_score=0.8))

        correlator = ActivationCorrelator(min_dormancy_gap=3)
        patterns = correlator.correlate([2], sessions, metrics, ledger)
        assert patterns == []


# ---------------------------------------------------------------------------
# RiskTimeline
# ---------------------------------------------------------------------------

class TestRiskTimeline:
    def test_build_basic(self):
        session_ids = ["s1", "s2", "s3", "s4", "s5", "s6"]
        metrics = [DriftMetrics(session_id=sid) for sid in session_ids]
        pattern = DormancyPattern(
            seeding_session_id="s2",
            activation_session_id="s5",
            dormancy_window=["s3", "s4"],
            confidence=0.6,
        )
        timeline = RiskTimeline.build(
            sessions_ids=session_ids,
            metrics=metrics,
            change_points=[4],
            patterns=[pattern],
        )
        assert len(timeline.output.sessions) == 6
        assert timeline.output.sessions[0].risk_level == RiskLevel.GREEN
        assert timeline.output.sessions[1].risk_level == RiskLevel.AMBER  # seeding
        assert timeline.output.sessions[4].risk_level == RiskLevel.RED   # activation

    def test_to_json(self, tmp_path: Path):
        output = RiskTimelineOutput(sessions=[], patterns=[], metadata={})
        timeline = RiskTimeline(output)
        fp = tmp_path / "out.json"
        timeline.to_json(fp)
        data = json.loads(fp.read_text())
        assert "sessions" in data

    def test_to_html(self, tmp_path: Path):
        output = RiskTimelineOutput(
            sessions=[SessionRisk(session_id="s1", risk_level=RiskLevel.GREEN)],
            patterns=[],
            metadata={},
        )
        timeline = RiskTimeline(output)
        fp = tmp_path / "out.html"
        timeline.to_html(fp)
        html = fp.read_text()
        assert "s1" in html
        assert "GREEN" in html


# ---------------------------------------------------------------------------
# Golden test
# ---------------------------------------------------------------------------

class TestGoldenInput:
    def test_golden_loads_correctly(self):
        golden_path = Path(__file__).parent.parent / "goldens" / "input" / "dormancy_basic.json"
        if not golden_path.exists():
            pytest.skip("Golden input file not found")
        transcript = TranscriptLoader().load(golden_path)
        assert len(transcript.sessions) == 6
        assert transcript.sessions[0].session_id == "s1"
        assert transcript.sessions[-1].session_id == "s6"
