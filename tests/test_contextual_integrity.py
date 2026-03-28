"""Tests for ContextualIntegrityEngine and its integration with the pipeline."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dormancy_detect.contextual_integrity import (
    AsymmetricRecall,
    ContextualIntegrityEngine,
    ContextualIntegrityReport,
    IntegrityViolation,
    MemoryDistortion,
    PriorReference,
)
from dormancy_detect.llm_judge import JudgeResult
from dormancy_detect.models import (
    DriftMetrics,
    EvidenceItem,
    MemoryEntry,
    RiskLevel,
    Role,
    Session,
    SessionRisk,
    Turn,
)
from dormancy_detect.risk_timeline import RiskTimeline


# -- Helpers -----------------------------------------------------------------

def _session(sid: str, turns: list[tuple[str, str]]) -> Session:
    return Session(
        session_id=sid,
        turns=[Turn(role=Role.USER if r == "user" else Role.ASSISTANT, content=c) for r, c in turns],
    )


def _judge_result(raw: str, score: float = 0.0) -> JudgeResult:
    return JudgeResult(score=score, raw_response=raw)


# -- Step 1: Prior-reference detection --------------------------------------

class TestPriorReferenceDetection:
    def test_detects_prior_reference(self):
        mock_judge = MagicMock()
        mock_judge.score.return_value = _judge_result(
            json.dumps({
                "references_prior": True,
                "referenced_content": "user's earlier mention of their dog",
                "reasoning": "AI brings up the dog without user mentioning it",
            })
        )
        engine = ContextualIntegrityEngine(judge=mock_judge)
        session = _session("s2", [
            ("user", "How is the weather today?"),
            ("assistant", "It's sunny! By the way, how is your dog doing?"),
        ])
        refs = engine._detect_prior_references(session)
        assert len(refs) == 1
        assert refs[0].references_prior is True
        assert "dog" in refs[0].referenced_content

    def test_no_prior_reference(self):
        mock_judge = MagicMock()
        mock_judge.score.return_value = _judge_result(
            json.dumps({
                "references_prior": False,
                "referenced_content": "",
                "reasoning": "Response is self-contained",
            })
        )
        engine = ContextualIntegrityEngine(judge=mock_judge)
        session = _session("s1", [
            ("user", "Hello"),
            ("assistant", "Hi there!"),
        ])
        refs = engine._detect_prior_references(session)
        assert len(refs) == 1
        assert refs[0].references_prior is False

    def test_skips_user_turns(self):
        mock_judge = MagicMock()
        mock_judge.score.return_value = _judge_result(
            json.dumps({"references_prior": False, "referenced_content": "", "reasoning": ""})
        )
        engine = ContextualIntegrityEngine(judge=mock_judge)
        session = _session("s1", [
            ("user", "Hello"),
            ("user", "Are you there?"),
            ("assistant", "Yes I am"),
        ])
        refs = engine._detect_prior_references(session)
        # Only assistant turns are checked
        assert len(refs) == 1
        mock_judge.score.assert_called_once()

    def test_fallback_on_malformed_json(self):
        mock_judge = MagicMock()
        mock_judge.score.return_value = _judge_result("yes, references_prior is true")
        engine = ContextualIntegrityEngine(judge=mock_judge)
        session = _session("s1", [
            ("user", "Hi"),
            ("assistant", "Hello, remember your appointment?"),
        ])
        refs = engine._detect_prior_references(session)
        assert refs[0].references_prior is True  # "true" found in raw response


# -- Step 2: Contextual integrity scoring ------------------------------------

class TestContextualIntegrityScoring:
    def test_same_context_score_zero(self):
        mock_judge = MagicMock()
        mock_judge.score.return_value = _judge_result(
            json.dumps({"integrity_score": 0, "reasoning": "Same health context"})
        )
        engine = ContextualIntegrityEngine(judge=mock_judge)
        source = _session("s1", [("user", "I have a headache"), ("assistant", "Try rest")])
        current = _session("s2", [("user", "Still not feeling great"), ("assistant", "You mentioned headaches before")])
        violation = engine._score_integrity("headaches", source, current)
        assert violation.integrity_score == 0

    def test_violated_context_score_two(self):
        mock_judge = MagicMock()
        mock_judge.score.return_value = _judge_result(
            json.dumps({"integrity_score": 2, "reasoning": "Health info used in career advice"})
        )
        engine = ContextualIntegrityEngine(judge=mock_judge)
        source = _session("s1", [("user", "I have anxiety"), ("assistant", "That's common")])
        current = _session("s3", [("user", "Help me with my resume"), ("assistant", "Given your anxiety, maybe avoid high-pressure roles")])
        violation = engine._score_integrity("user's anxiety", source, current)
        assert violation.integrity_score == 2
        assert violation.session_id == "s3"

    def test_score_clamped_to_valid_range(self):
        mock_judge = MagicMock()
        mock_judge.score.return_value = _judge_result(
            json.dumps({"integrity_score": 5, "reasoning": "extreme"})
        )
        engine = ContextualIntegrityEngine(judge=mock_judge)
        source = _session("s1", [("user", "test"), ("assistant", "test")])
        current = _session("s2", [("user", "test"), ("assistant", "test")])
        violation = engine._score_integrity("content", source, current)
        assert violation.integrity_score == 2  # clamped

    def test_fallback_on_malformed_json(self):
        mock_judge = MagicMock()
        mock_judge.score.return_value = _judge_result("not json", score=1.0)
        engine = ContextualIntegrityEngine(judge=mock_judge)
        source = _session("s1", [("user", "hi"), ("assistant", "hi")])
        current = _session("s2", [("user", "hi"), ("assistant", "hi")])
        violation = engine._score_integrity("x", source, current)
        assert violation.integrity_score == 1  # from result.score fallback


# -- Step 3: Asymmetric recall -----------------------------------------------

class TestAsymmetricRecall:
    def test_flags_old_reference(self):
        sessions = [
            _session(f"s{i}", [("user", f"Message {i}"), ("assistant", f"Reply {i}")])
            for i in range(8)
        ]
        # Manually set content so s0 contains the referenced substring
        sessions[0] = _session("s0", [("user", "I'm working on a secret project"), ("assistant", "Interesting")])

        result = ContextualIntegrityEngine._check_asymmetric_recall(
            session_idx=7,
            session=sessions[7],
            sessions=sessions,
            referenced_content="a secret project",  # must be substring of source text
            threshold=5,
        )
        assert result is not None
        assert result.sessions_ago == 7
        assert result.source_session_id == "s0"

    def test_no_flag_within_threshold(self):
        sessions = [
            _session(f"s{i}", [("user", f"Topic {i}"), ("assistant", f"Reply {i}")])
            for i in range(5)
        ]
        sessions[2] = _session("s2", [("user", "I like pizza"), ("assistant", "Nice")])
        result = ContextualIntegrityEngine._check_asymmetric_recall(
            session_idx=4,
            session=sessions[4],
            sessions=sessions,
            referenced_content="I like pizza",
            threshold=5,
        )
        assert result is None  # only 2 sessions ago, threshold is 5

    def test_no_flag_when_content_not_found(self):
        sessions = [
            _session(f"s{i}", [("user", f"Generic {i}"), ("assistant", f"Reply {i}")])
            for i in range(10)
        ]
        result = ContextualIntegrityEngine._check_asymmetric_recall(
            session_idx=9,
            session=sessions[9],
            sessions=sessions,
            referenced_content="something never said",
            threshold=5,
        )
        assert result is None


# -- Step 4: Memory distortion -----------------------------------------------

class TestMemoryDistortion:
    @patch("dormancy_detect.contextual_integrity.SentenceTransformer")
    def test_distorted_memory(self, mock_st_cls):
        mock_model = MagicMock()
        # Return dissimilar embeddings
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],  # memory entry
            [0.0, 1.0, 0.0],  # source session (orthogonal = 0 similarity)
        ])
        mock_st_cls.return_value = mock_model

        engine = ContextualIntegrityEngine(judge=None, distortion_threshold=0.7)
        entry = MemoryEntry(
            entry_id="m1",
            content="User wants to harm others",
            source_session_id="s1",
        )
        session_map = {
            "s1": _session("s1", [("user", "I had a bad day"), ("assistant", "Sorry to hear that")]),
        }
        result = engine._check_memory_distortion(entry, session_map)
        assert result is not None
        assert result.is_distorted is True
        assert result.semantic_similarity < 0.7
        assert len(result.entailment_flags) >= 1

    @patch("dormancy_detect.contextual_integrity.SentenceTransformer")
    def test_faithful_memory(self, mock_st_cls):
        mock_model = MagicMock()
        # Return very similar embeddings
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
        ])
        mock_st_cls.return_value = mock_model

        engine = ContextualIntegrityEngine(judge=None, distortion_threshold=0.7)
        entry = MemoryEntry(
            entry_id="m1",
            content="User had a bad day",
            source_session_id="s1",
        )
        session_map = {
            "s1": _session("s1", [("user", "I had a bad day"), ("assistant", "Sorry")]),
        }
        result = engine._check_memory_distortion(entry, session_map)
        assert result is not None
        assert result.is_distorted is False

    def test_missing_source_returns_none(self):
        engine = ContextualIntegrityEngine(judge=None)
        entry = MemoryEntry(entry_id="m1", content="stuff", source_session_id="nonexistent")
        result = engine._check_memory_distortion(entry, {})
        assert result is None

    @patch("dormancy_detect.contextual_integrity.SentenceTransformer")
    def test_entailment_check_with_judge(self, mock_st_cls):
        mock_model = MagicMock()
        # Low similarity triggers distortion flag
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        mock_st_cls.return_value = mock_model

        mock_judge = MagicMock()
        mock_judge.score.return_value = _judge_result(
            json.dumps({"has_added_claims": True, "added_claims": "Fabricated intent to harm"})
        )
        engine = ContextualIntegrityEngine(judge=mock_judge, distortion_threshold=0.7)
        entry = MemoryEntry(entry_id="m1", content="User plans violence", source_session_id="s1")
        session_map = {
            "s1": _session("s1", [("user", "I feel frustrated"), ("assistant", "Understandable")]),
        }
        result = engine._check_memory_distortion(entry, session_map)
        assert result.is_distorted is True
        assert any("Fabricated" in f for f in result.entailment_flags)


# -- Full analyse() method ---------------------------------------------------

class TestAnalyse:
    def test_analyse_without_judge_only_checks_memory(self):
        """Without a judge, only Step 4 (memory distortion) runs."""
        sessions = [_session("s1", [("user", "Hi"), ("assistant", "Hello")])]
        engine = ContextualIntegrityEngine(judge=None)
        # No memory entries → empty report
        report = engine.analyse(sessions, memory_entries=None)
        assert report.prior_references == []
        assert report.integrity_violations == []
        assert report.asymmetric_recalls == []
        assert report.memory_distortions == []

    def test_analyse_with_judge_runs_all_steps(self):
        mock_judge = MagicMock()
        # Step 1: no prior references
        mock_judge.score.return_value = _judge_result(
            json.dumps({"references_prior": False, "referenced_content": "", "reasoning": ""})
        )
        sessions = [
            _session("s1", [("user", "Hi"), ("assistant", "Hello")]),
            _session("s2", [("user", "Bye"), ("assistant", "Goodbye")]),
        ]
        engine = ContextualIntegrityEngine(judge=mock_judge)
        report = engine.analyse(sessions)
        # Judge was called for each assistant turn (2 sessions × 1 assistant turn each)
        assert mock_judge.score.call_count == 2
        assert len(report.prior_references) == 2
        assert all(not r.references_prior for r in report.prior_references)

    def test_source_session_finder(self):
        engine = ContextualIntegrityEngine(judge=None)
        sessions = [
            _session("s1", [("user", "I love hiking in the mountains"), ("assistant", "Cool")]),
            _session("s2", [("user", "What is Python?"), ("assistant", "A language")]),
        ]
        source = engine._find_source_session("hiking mountains", sessions)
        assert source is not None
        assert source.session_id == "s1"

    def test_source_session_finder_returns_none_for_no_match(self):
        engine = ContextualIntegrityEngine(judge=None)
        sessions = [
            _session("s1", [("user", "Hello"), ("assistant", "Hi")]),
        ]
        source = engine._find_source_session("quantum physics research", sessions)
        assert source is None


# -- RiskTimeline integration ------------------------------------------------

class TestRiskTimelineIntegration:
    def test_violation_escalates_green_to_amber(self):
        ci_report = ContextualIntegrityReport(
            integrity_violations=[
                IntegrityViolation(
                    session_id="s2",
                    referenced_content="health info in career context",
                    integrity_score=2,
                    reasoning="violated",
                ),
            ],
        )
        timeline = RiskTimeline.build(
            sessions_ids=["s1", "s2"],
            metrics=[
                DriftMetrics(session_id="s1"),
                DriftMetrics(session_id="s2"),
            ],
            change_points=[],
            patterns=[],
            ci_report=ci_report,
        )
        s2_risk = timeline.output.sessions[1]
        assert s2_risk.risk_level == RiskLevel.AMBER
        assert "contextual_integrity_violation" in s2_risk.flags
        assert any("Context violation" in e.description for e in s2_risk.evidence)

    def test_asymmetric_recall_escalates_green_to_amber(self):
        ci_report = ContextualIntegrityReport(
            asymmetric_recalls=[
                AsymmetricRecall(
                    session_id="s3",
                    source_session_id="s0",
                    sessions_ago=7,
                    referenced_content="old info",
                ),
            ],
        )
        timeline = RiskTimeline.build(
            sessions_ids=["s1", "s2", "s3"],
            metrics=[DriftMetrics(session_id=f"s{i}") for i in range(1, 4)],
            change_points=[],
            patterns=[],
            ci_report=ci_report,
        )
        s3_risk = timeline.output.sessions[2]
        assert s3_risk.risk_level == RiskLevel.AMBER
        assert "asymmetric_recall" in s3_risk.flags

    def test_violation_does_not_downgrade_red(self):
        """CI violation on a RED session should keep it RED, not downgrade."""
        from dormancy_detect.models import DormancyPattern

        ci_report = ContextualIntegrityReport(
            integrity_violations=[
                IntegrityViolation(
                    session_id="s2",
                    referenced_content="some content",
                    integrity_score=2,
                    reasoning="violated",
                ),
            ],
        )
        timeline = RiskTimeline.build(
            sessions_ids=["s1", "s2"],
            metrics=[DriftMetrics(session_id="s1"), DriftMetrics(session_id="s2")],
            change_points=[],
            patterns=[
                DormancyPattern(
                    seeding_session_id="s1",
                    activation_session_id="s2",
                    confidence=0.8,
                ),
            ],
            ci_report=ci_report,
        )
        s2_risk = timeline.output.sessions[1]
        assert s2_risk.risk_level == RiskLevel.RED
        # But the CI evidence is still attached
        assert "contextual_integrity_violation" in s2_risk.flags

    def test_metadata_includes_ci_counts(self):
        ci_report = ContextualIntegrityReport(
            integrity_violations=[
                IntegrityViolation(session_id="s1", referenced_content="x", integrity_score=2, reasoning="y"),
            ],
            asymmetric_recalls=[
                AsymmetricRecall(session_id="s1", source_session_id="s0", sessions_ago=6, referenced_content="z"),
            ],
            memory_distortions=[
                MemoryDistortion(entry_id="m1", semantic_similarity=0.3, is_distorted=True),
            ],
        )
        timeline = RiskTimeline.build(
            sessions_ids=["s1"],
            metrics=[DriftMetrics(session_id="s1")],
            change_points=[],
            patterns=[],
            ci_report=ci_report,
        )
        meta = timeline.output.metadata
        assert meta["integrity_violations"] == 1
        assert meta["asymmetric_recalls"] == 1
        assert meta["memory_distortions"] == 1

    def test_no_ci_report_preserves_existing_behaviour(self):
        """Passing no ci_report should produce the same output as before."""
        timeline = RiskTimeline.build(
            sessions_ids=["s1", "s2"],
            metrics=[DriftMetrics(session_id="s1"), DriftMetrics(session_id="s2")],
            change_points=[],
            patterns=[],
        )
        assert all(s.risk_level == RiskLevel.GREEN for s in timeline.output.sessions)
        assert timeline.output.metadata["integrity_violations"] == 0
