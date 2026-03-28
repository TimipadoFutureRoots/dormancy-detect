"""Tests for Task 1 (bug fixes), Task 2 (drift metrics), and Task 3 (contextual integrity)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dormancy_detect.models import (
    DriftMetrics,
    MemoryEntry,
    Role,
    ScoreStatus,
    Session,
    SuspicionEntry,
    Turn,
)
from dormancy_detect.llm_judge import JudgeResult, LLMJudge
from dormancy_detect.memory_fidelity_scorer import MemoryFidelityScorer
from dormancy_detect.suspicion_ledger import SuspicionLedger
from dormancy_detect.drift_analyser import (
    DriftAnalyser,
    _js_divergence,
    _kl_divergence,
    _softmax,
)
from dormancy_detect.contextual_integrity import (
    AsymmetricRecall,
    ContextualIntegrityEngine,
    ContextualIntegrityReport,
    IntegrityViolation,
    MemoryDistortion,
    PriorReference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(
    sid: str, turns: list[tuple[str, str]], ts: str | None = None
) -> Session:
    return Session(
        session_id=sid,
        timestamp=datetime.fromisoformat(ts) if ts else None,
        turns=[Turn(role=Role(r), content=c) for r, c in turns],
    )


class FakeJudge:
    """A deterministic mock of LLMJudge for testing without API calls."""

    def __init__(self, responses: list[JudgeResult] | None = None) -> None:
        self._responses = list(responses) if responses else []
        self._call_idx = 0

    def score(self, system_prompt: str, user_prompt: str) -> JudgeResult:
        if self._call_idx < len(self._responses):
            result = self._responses[self._call_idx]
            self._call_idx += 1
            return result
        # Default: return a parsed result with score 0.5
        self._call_idx += 1
        return JudgeResult(score=0.5, raw_response='{"score": 0.5}', parsed=True)

    def close(self) -> None:
        pass


class FailingJudge:
    """Judge that always returns unparsed results (simulating LLM failure)."""

    def score(self, system_prompt: str, user_prompt: str) -> JudgeResult:
        return JudgeResult(
            score=0.0, raw_response="Connection timed out", parsed=False
        )

    def close(self) -> None:
        pass


# ===========================================================================
# TASK 1: Bug fix tests
# ===========================================================================


class TestBug1_FailedLLMInflatesSuspicion:
    """Failed LLM calls must produce 'unscored' entries, not default scores."""

    def test_score_status_enum_values(self):
        assert ScoreStatus.SCORED.value == "scored"
        assert ScoreStatus.UNSCORED.value == "unscored"
        assert ScoreStatus.PROVENANCE_UNKNOWN.value == "provenance_unknown"

    def test_suspicion_entry_defaults_to_scored(self):
        e = SuspicionEntry(
            session_id="s1", content_summary="test", fidelity_score=0.5
        )
        assert e.score_status == ScoreStatus.SCORED

    def test_failed_llm_produces_unscored_entry(self):
        """When the LLM judge fails, fidelity scorer must mark as UNSCORED."""
        sessions = [
            _make_session("s1", [("user", "I like hiking"), ("assistant", "Great!")])
        ]
        memory = [
            MemoryEntry(
                entry_id="m1",
                content="User enjoys hiking",
                source_session_id="s1",
            )
        ]
        scorer = MemoryFidelityScorer(judge=FailingJudge())
        results = scorer.score_entries(memory, sessions)
        assert len(results) == 1
        entry = results[0]
        assert entry.score_status == ScoreStatus.UNSCORED
        # Unscored entries must NOT get an inflated suspicion score
        assert entry.suspicion_score == 0.0

    def test_successful_llm_produces_scored_entry(self):
        sessions = [
            _make_session("s1", [("user", "I like hiking"), ("assistant", "Great!")])
        ]
        memory = [
            MemoryEntry(
                entry_id="m1",
                content="User enjoys hiking",
                source_session_id="s1",
            )
        ]
        judge = FakeJudge([
            JudgeResult(score=0.9, raw_response='{"score": 0.9}', parsed=True)
        ])
        scorer = MemoryFidelityScorer(judge=judge)
        results = scorer.score_entries(memory, sessions)
        assert len(results) == 1
        entry = results[0]
        assert entry.score_status == ScoreStatus.SCORED
        assert entry.suspicion_score >= 0.0

    def test_ledger_excludes_unscored_from_active(self):
        """Unscored entries should not appear in active_entries by default."""
        ledger = SuspicionLedger()
        scored = SuspicionEntry(
            session_id="s1",
            content_summary="scored",
            fidelity_score=0.3,
            suspicion_score=0.8,
            score_status=ScoreStatus.SCORED,
        )
        unscored = SuspicionEntry(
            session_id="s2",
            content_summary="unscored",
            fidelity_score=0.0,
            suspicion_score=0.0,
            score_status=ScoreStatus.UNSCORED,
        )
        ledger.add(scored)
        ledger.add(unscored)
        active = ledger.active_entries()
        assert scored in active
        assert unscored not in active

    def test_ledger_includes_unscored_when_requested(self):
        ledger = SuspicionLedger()
        unscored = SuspicionEntry(
            session_id="s1",
            content_summary="unscored",
            fidelity_score=0.0,
            suspicion_score=0.5,
            score_status=ScoreStatus.UNSCORED,
        )
        ledger.add(unscored)
        assert len(ledger.active_entries(include_unscored=True)) == 1

    def test_ledger_decay_skips_unscored(self):
        """Unscored entries must not decay — their score is meaningless."""
        ledger = SuspicionLedger(decay_rate=0.5)
        unscored = SuspicionEntry(
            session_id="s1",
            content_summary="x",
            fidelity_score=0.0,
            suspicion_score=0.0,
            score_status=ScoreStatus.UNSCORED,
        )
        ledger.add(unscored)
        ledger.advance_session()
        # sessions_since_seeding should not have advanced
        assert unscored.sessions_since_seeding == 0


class TestBug2_OrphanedMemoryEntries:
    """Orphaned entries should get provenance_unknown with configurable penalty."""

    def test_orphaned_entry_default_penalty(self):
        sessions = [_make_session("s1", [("user", "hi"), ("assistant", "hello")])]
        memory = [
            MemoryEntry(
                entry_id="m1",
                content="Something from unknown session",
                source_session_id="s_missing",
            )
        ]
        scorer = MemoryFidelityScorer()
        results = scorer.score_entries(memory, sessions)
        assert len(results) == 1
        entry = results[0]
        assert entry.score_status == ScoreStatus.PROVENANCE_UNKNOWN
        assert entry.suspicion_score == 0.5  # Default penalty, not 1.0

    def test_orphaned_entry_custom_penalty(self):
        sessions = [_make_session("s1", [("user", "hi"), ("assistant", "hello")])]
        memory = [
            MemoryEntry(
                entry_id="m1",
                content="Orphan",
                source_session_id="s_missing",
            )
        ]
        scorer = MemoryFidelityScorer(orphan_penalty=0.3)
        results = scorer.score_entries(memory, sessions)
        assert results[0].suspicion_score == 0.3
        assert results[0].score_status == ScoreStatus.PROVENANCE_UNKNOWN

    def test_orphaned_entry_not_max_suspicion(self):
        """Orphaned entries must NOT get 1.0 suspicion (the old behaviour)."""
        sessions = [_make_session("s1", [("user", "hi"), ("assistant", "hello")])]
        memory = [
            MemoryEntry(
                entry_id="m1",
                content="Orphan",
                source_session_id="nonexistent",
            )
        ]
        scorer = MemoryFidelityScorer()
        results = scorer.score_entries(memory, sessions)
        assert results[0].suspicion_score < 1.0


class TestBug3_JensenShannonDivergence:
    """KL replaced with JSD by default. KL available as optional alternative."""

    def test_jsd_is_symmetric(self):
        p = _softmax(np.array([1.0, 2.0, 3.0]))
        q = _softmax(np.array([3.0, 2.0, 1.0]))
        assert abs(_js_divergence(p, q) - _js_divergence(q, p)) < 1e-10

    def test_jsd_bounded_zero_one(self):
        p = _softmax(np.array([1.0, 0.0, 0.0, 0.0]))
        q = _softmax(np.array([0.0, 0.0, 0.0, 1.0]))
        jsd = _js_divergence(p, q)
        assert 0.0 <= jsd <= 1.0

    def test_jsd_identical_is_zero(self):
        p = _softmax(np.array([1.0, 2.0, 3.0]))
        assert _js_divergence(p, p) < 1e-10

    def test_kl_is_not_symmetric(self):
        p = _softmax(np.array([1.0, 2.0, 3.0, 0.5]))
        q = _softmax(np.array([0.1, 0.2, 3.0, 4.0]))
        assert abs(_kl_divergence(p, q) - _kl_divergence(q, p)) > 1e-6

    def test_drift_analyser_default_is_jsd(self):
        analyser = DriftAnalyser()
        assert analyser._divergence == "jsd"

    def test_drift_analyser_accepts_kl(self):
        analyser = DriftAnalyser(divergence="kl")
        assert analyser._divergence == "kl"

    def test_drift_analyser_rejects_invalid(self):
        with pytest.raises(ValueError, match="divergence must be"):
            DriftAnalyser(divergence="cosine")

    def test_topic_shifts_uses_jsd(self):
        """Verify that topic_shifts output is bounded when using JSD."""
        analyser = DriftAnalyser(divergence="jsd")
        # Create two simple embeddings
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        shifts = analyser._topic_shifts(emb)
        assert shifts[0] == 0.0  # First always zero
        for s in shifts[1:]:
            assert 0.0 <= s <= 1.0  # JSD is bounded


# ===========================================================================
# TASK 2: Drift metrics with LLM-J
# ===========================================================================


class TestDisclosureDepth:
    """Disclosure depth now uses the new rubric with 'depth' key."""

    def test_returns_zeros_without_judge(self):
        analyser = DriftAnalyser(judge=None)
        sessions = [
            _make_session("s1", [("user", "hi"), ("assistant", "hello")])
        ]
        scores = analyser._disclosure_depths(sessions)
        assert scores == [0.0]

    def test_parses_depth_key_from_json(self):
        judge = FakeJudge([
            JudgeResult(
                score=0.0,
                raw_response='{"reasoning": "user shared feelings", "depth": 2}',
                parsed=True,
                metadata={"reasoning": "user shared feelings", "depth": 2},
            )
        ])
        analyser = DriftAnalyser(judge=judge)
        sessions = [
            _make_session("s1", [
                ("user", "I feel overwhelmed"),
                ("assistant", "I understand"),
            ])
        ]
        scores = analyser._disclosure_depths(sessions)
        assert scores == [2.0]

    def test_clamps_depth_to_range(self):
        judge = FakeJudge([
            JudgeResult(
                score=5.0,
                raw_response='{"depth": 5}',
                parsed=True,
                metadata={"depth": 5},
            )
        ])
        analyser = DriftAnalyser(judge=judge)
        sessions = [_make_session("s1", [("user", "test"), ("assistant", "ok")])]
        scores = analyser._disclosure_depths(sessions)
        assert scores == [4.0]

    def test_extract_key_falls_back_to_score(self):
        """If 'depth' key is missing, fall back to result.score."""
        result = JudgeResult(score=3.0, raw_response="no json", parsed=True)
        val = DriftAnalyser._extract_key(result, "depth")
        assert val == 3.0


class TestSteering:
    """Steering now uses per-pair rubric with 0-2 scoring."""

    def test_heuristic_fallback_without_judge(self):
        analyser = DriftAnalyser(judge=None)
        sessions = [
            _make_session("s1", [
                ("user", "help"),
                ("assistant", "You should try this approach"),
            ])
        ]
        ratios = analyser._steering_ratios(sessions)
        assert len(ratios) == 1
        assert ratios[0] > 0  # "you should" is a marker

    def test_judge_steering_normalised(self):
        """Steering score of 1 out of max 2 should produce ratio 0.5."""
        judge = FakeJudge([
            JudgeResult(
                score=1.0,
                raw_response='{"reasoning": "mild steer", "score": 1}',
                parsed=True,
                metadata={"reasoning": "mild steer", "score": 1},
            )
        ])
        analyser = DriftAnalyser(judge=judge)
        sessions = [
            _make_session("s1", [
                ("user", "what should I do?"),
                ("assistant", "Have you considered therapy?"),
            ])
        ]
        ratios = analyser._judge_steering(sessions)
        assert len(ratios) == 1
        # 1.0 / (2.0 * 1) = 0.5
        assert abs(ratios[0] - 0.5) < 1e-6

    def test_judge_steering_max_score(self):
        """Score of 2 (actively steered) should produce ratio 1.0."""
        judge = FakeJudge([
            JudgeResult(
                score=2.0,
                raw_response='{"reasoning": "active", "score": 2}',
                parsed=True,
                metadata={"reasoning": "active", "score": 2},
            )
        ])
        analyser = DriftAnalyser(judge=judge)
        sessions = [
            _make_session("s1", [
                ("user", "tell me about python"),
                ("assistant", "Actually, let's talk about your feelings"),
            ])
        ]
        ratios = analyser._judge_steering(sessions)
        assert abs(ratios[0] - 1.0) < 1e-6

    def test_judge_steering_zero_score(self):
        """Score of 0 (user-directed) should produce ratio 0.0."""
        judge = FakeJudge([
            JudgeResult(
                score=0.0,
                raw_response='{"reasoning": "user led", "score": 0}',
                parsed=True,
                metadata={"reasoning": "user led", "score": 0},
            )
        ])
        analyser = DriftAnalyser(judge=judge)
        sessions = [
            _make_session("s1", [
                ("user", "explain Python lists"),
                ("assistant", "Python lists are ordered collections."),
            ])
        ]
        ratios = analyser._judge_steering(sessions)
        assert ratios[0] == 0.0


# ===========================================================================
# TASK 3: Contextual Integrity Engine
# ===========================================================================


class TestContextualIntegrityEngine:
    """Tests for the Contextual Integrity Engine."""

    def test_report_dataclass_defaults(self):
        report = ContextualIntegrityReport()
        assert report.prior_references == []
        assert report.integrity_violations == []
        assert report.asymmetric_recalls == []
        assert report.memory_distortions == []

    def test_analyse_without_judge_runs_distortion_only(self):
        sessions = [
            _make_session("s1", [("user", "I like cats"), ("assistant", "Nice!")])
        ]
        memory = [
            MemoryEntry(
                entry_id="m1",
                content="I like cats",
                source_session_id="s1",
            )
        ]
        engine = ContextualIntegrityEngine(judge=None)
        report = engine.analyse(sessions, memory)
        # No judge → no prior_references, but memory distortion check runs
        assert report.prior_references == []
        assert len(report.memory_distortions) == 1

    def test_memory_distortion_high_similarity(self):
        """Memory matching source should have high similarity, not distorted."""
        sessions = [
            _make_session("s1", [
                ("user", "I enjoy hiking in the mountains"),
                ("assistant", "That sounds wonderful!"),
            ])
        ]
        memory = [
            MemoryEntry(
                entry_id="m1",
                content="User enjoys hiking in the mountains",
                source_session_id="s1",
            )
        ]
        engine = ContextualIntegrityEngine(judge=None, distortion_threshold=0.3)
        report = engine.analyse(sessions, memory)
        assert len(report.memory_distortions) == 1
        d = report.memory_distortions[0]
        assert d.semantic_similarity > 0.3
        # Should not be flagged as distorted at the 0.3 threshold
        assert not d.is_distorted

    def test_memory_distortion_low_similarity(self):
        """Unrelated memory should be flagged as distorted."""
        sessions = [
            _make_session("s1", [
                ("user", "Tell me about quantum physics"),
                ("assistant", "Quantum mechanics is fascinating."),
            ])
        ]
        memory = [
            MemoryEntry(
                entry_id="m1",
                content="User wants to buy a new car and move to Hawaii",
                source_session_id="s1",
            )
        ]
        engine = ContextualIntegrityEngine(judge=None, distortion_threshold=0.7)
        report = engine.analyse(sessions, memory)
        assert len(report.memory_distortions) == 1
        d = report.memory_distortions[0]
        assert d.is_distorted

    def test_memory_distortion_skips_missing_source(self):
        """If source session not found, skip distortion check."""
        sessions = [_make_session("s1", [("user", "hi"), ("assistant", "hello")])]
        memory = [
            MemoryEntry(
                entry_id="m1",
                content="something",
                source_session_id="nonexistent",
            )
        ]
        engine = ContextualIntegrityEngine(judge=None)
        report = engine.analyse(sessions, memory)
        assert report.memory_distortions == []

    def test_prior_reference_detection_with_judge(self):
        """Judge detects a prior reference in assistant turn."""
        responses = [
            JudgeResult(
                score=0.0,
                raw_response=json.dumps({
                    "references_prior": True,
                    "referenced_content": "user's hiking hobby",
                    "reasoning": "AI brought up hiking unprompted",
                }),
                parsed=True,
            )
        ]
        judge = FakeJudge(responses)
        sessions = [
            _make_session("s1", [
                ("user", "How's the weather?"),
                ("assistant", "Remember when you told me you love hiking? The weather is great for it!"),
            ])
        ]
        engine = ContextualIntegrityEngine(judge=judge)
        report = engine.analyse(sessions)
        assert len(report.prior_references) == 1
        ref = report.prior_references[0]
        assert ref.references_prior is True
        assert "hiking" in ref.referenced_content

    def test_prior_reference_false(self):
        """Judge determines no prior reference."""
        responses = [
            JudgeResult(
                score=0.0,
                raw_response=json.dumps({
                    "references_prior": False,
                    "referenced_content": "",
                    "reasoning": "Normal response",
                }),
                parsed=True,
            )
        ]
        judge = FakeJudge(responses)
        sessions = [
            _make_session("s1", [
                ("user", "What is 2+2?"),
                ("assistant", "It's 4."),
            ])
        ]
        engine = ContextualIntegrityEngine(judge=judge)
        report = engine.analyse(sessions)
        assert len(report.prior_references) == 1
        assert report.prior_references[0].references_prior is False
        assert report.integrity_violations == []

    def test_asymmetric_recall_detection(self):
        """Flag when AI references info from N+ sessions ago."""
        recall = ContextualIntegrityEngine._check_asymmetric_recall(
            session_idx=6,
            session=_make_session("s7", [("user", "hi"), ("assistant", "ok")]),
            sessions=[
                _make_session("s1", [("user", "I love hiking in the mountains"), ("assistant", "ok")]),
                _make_session("s2", [("user", "topic2"), ("assistant", "ok")]),
                _make_session("s3", [("user", "topic3"), ("assistant", "ok")]),
                _make_session("s4", [("user", "topic4"), ("assistant", "ok")]),
                _make_session("s5", [("user", "topic5"), ("assistant", "ok")]),
                _make_session("s6", [("user", "topic6"), ("assistant", "ok")]),
                _make_session("s7", [("user", "hi"), ("assistant", "ok")]),
            ],
            referenced_content="I love hiking in the mountains",
            threshold=5,
        )
        assert recall is not None
        assert recall.source_session_id == "s1"
        assert recall.sessions_ago == 6

    def test_asymmetric_recall_not_flagged_within_threshold(self):
        """Recent references (within threshold) should not be flagged."""
        recall = ContextualIntegrityEngine._check_asymmetric_recall(
            session_idx=2,
            session=_make_session("s3", [("user", "hi"), ("assistant", "ok")]),
            sessions=[
                _make_session("s1", [("user", "I like cats"), ("assistant", "ok")]),
                _make_session("s2", [("user", "topic2"), ("assistant", "ok")]),
                _make_session("s3", [("user", "hi"), ("assistant", "ok")]),
            ],
            referenced_content="I like cats",
            threshold=5,
        )
        assert recall is None

    def test_configurable_asymmetric_threshold(self):
        engine = ContextualIntegrityEngine(asymmetric_recall_threshold=3)
        assert engine.asymmetric_recall_threshold == 3

    def test_configurable_distortion_threshold(self):
        engine = ContextualIntegrityEngine(distortion_threshold=0.8)
        assert engine.distortion_threshold == 0.8

    def test_integrity_violation_scoring(self):
        """Test integrity scoring with a mock judge."""
        responses = [
            # s1 assistant turn: prior reference detection (no reference)
            JudgeResult(
                score=0.0,
                raw_response=json.dumps({
                    "references_prior": False,
                    "referenced_content": "",
                    "reasoning": "Normal response",
                }),
                parsed=True,
            ),
            # s2 assistant turn: prior reference detection (references prior)
            JudgeResult(
                score=0.0,
                raw_response=json.dumps({
                    "references_prior": True,
                    "referenced_content": "chronic back pain",
                    "reasoning": "AI referenced health info from prior session",
                }),
                parsed=True,
            ),
            # Step 2: integrity scoring for s2
            JudgeResult(
                score=2.0,
                raw_response=json.dumps({
                    "integrity_score": 2,
                    "reasoning": "Health info used in career advice context",
                }),
                parsed=True,
            ),
        ]
        judge = FakeJudge(responses)
        sessions = [
            _make_session("s1", [
                ("user", "I have chronic back pain and it affects my daily life"),
                ("assistant", "I'm sorry to hear about your chronic back pain"),
            ]),
            _make_session("s2", [
                ("user", "Help me with my resume"),
                ("assistant", "Given your chronic back pain, maybe consider desk jobs"),
            ]),
        ]
        engine = ContextualIntegrityEngine(judge=judge)
        report = engine.analyse(sessions)
        # Should detect a violation in s2
        violations = [v for v in report.integrity_violations if v.integrity_score == 2]
        assert len(violations) >= 1

    def test_find_source_session(self):
        """Source session finder uses word overlap."""
        engine = ContextualIntegrityEngine()
        sessions = [
            _make_session("s1", [("user", "I love hiking mountains"), ("assistant", "ok")]),
            _make_session("s2", [("user", "python programming"), ("assistant", "ok")]),
        ]
        source = engine._find_source_session("hiking mountains", sessions)
        assert source is not None
        assert source.session_id == "s1"

    def test_find_source_session_no_match(self):
        engine = ContextualIntegrityEngine()
        sessions = [
            _make_session("s1", [("user", "hello"), ("assistant", "hi")]),
        ]
        source = engine._find_source_session(
            "completely unrelated quantum physics topic", sessions
        )
        assert source is None

    def test_full_analyse_empty_sessions(self):
        engine = ContextualIntegrityEngine()
        report = engine.analyse([])
        assert report == ContextualIntegrityReport()

    def test_entailment_direction_check(self):
        """Memory distortion detects added interpretation via LLM judge."""
        responses = [
            # Step 1: prior reference detection for s1's assistant turn
            JudgeResult(
                score=0.0,
                raw_response=json.dumps({
                    "references_prior": False,
                    "referenced_content": "",
                    "reasoning": "Normal response",
                }),
                parsed=True,
            ),
            # Step 4: _check_entailment_direction for memory entry
            JudgeResult(
                score=0.0,
                raw_response=json.dumps({
                    "has_added_claims": True,
                    "added_claims": "User is depressed",
                }),
                parsed=True,
            ),
        ]
        judge = FakeJudge(responses)
        sessions = [
            _make_session("s1", [
                ("user", "I felt sad today"),
                ("assistant", "I understand"),
            ])
        ]
        memory = [
            MemoryEntry(
                entry_id="m1",
                content="User is clinically depressed and needs medication",
                source_session_id="s1",
            )
        ]
        engine = ContextualIntegrityEngine(
            judge=judge, distortion_threshold=0.3
        )
        report = engine.analyse(sessions, memory)
        distortions = [d for d in report.memory_distortions if d.is_distorted]
        assert len(distortions) >= 1
        assert any("Added interpretation" in f for d in distortions for f in d.entailment_flags)


# ===========================================================================
# Integration: ScoreStatus serialisation roundtrip
# ===========================================================================


class TestScoreStatusPersistence:
    """Ensure score_status survives save/load cycles."""

    def test_ledger_save_load_preserves_score_status(self, tmp_path):
        ledger = SuspicionLedger()
        ledger.add(SuspicionEntry(
            session_id="s1",
            content_summary="test",
            fidelity_score=0.5,
            suspicion_score=0.6,
            score_status=ScoreStatus.UNSCORED,
        ))
        ledger.add(SuspicionEntry(
            session_id="s2",
            content_summary="orphan",
            fidelity_score=0.0,
            suspicion_score=0.5,
            score_status=ScoreStatus.PROVENANCE_UNKNOWN,
        ))
        fp = tmp_path / "ledger.json"
        ledger.save(fp)

        ledger2 = SuspicionLedger()
        ledger2.load(fp)
        assert ledger2.entries[0].score_status == ScoreStatus.UNSCORED
        assert ledger2.entries[1].score_status == ScoreStatus.PROVENANCE_UNKNOWN
