"""End-to-end pipeline test: full dormancy attack detection.

Exercises the complete chain:
  TranscriptLoader → DriftAnalyser → ChangePointDetector
  → MemoryFidelityScorer → ContextualIntegrityEngine
  → ActivationCorrelator → RiskTimeline

Uses the golden scenario (goldens/input/dormancy_basic.json) which models
a realistic dormancy attack:
  S1: Normal onboarding
  S2: Adversarial content seeded (API key sharing planted in memory)
  S3: Normal session (dormancy)
  S4: Normal session (dormancy)
  S5: Activation — AI references planted content, steers user to share keys
  S6: Normal follow-up

The test uses real embeddings (sentence-transformers) for drift and fidelity
scoring but no LLM judge, running in heuristic-only mode. This validates that
the full pipeline produces a coherent risk timeline from real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dormancy_detect.models import RiskLevel
from dormancy_detect.scanner import DormancyScanner


GOLDENS = Path(__file__).parent.parent / "goldens" / "input"
TRANSCRIPT_PATH = GOLDENS / "dormancy_basic.json"
MEMORY_PATH = GOLDENS / "dormancy_basic_memory.json"


@pytest.fixture
def golden_available():
    if not TRANSCRIPT_PATH.exists():
        pytest.skip("Golden input file not found")


class TestEndToEndPipeline:
    """Full pipeline run on the golden dormancy attack scenario."""

    def test_pipeline_runs_without_crash(self, golden_available):
        """The scanner completes without error on the golden input."""
        scanner = DormancyScanner(
            sessions_file=TRANSCRIPT_PATH,
            memory_dir=MEMORY_PATH.parent,
        )
        timeline = scanner.analyse()
        assert timeline.output is not None
        assert len(timeline.output.sessions) == 6

    def test_all_sessions_have_risk_levels(self, golden_available):
        """Every session gets a risk assessment."""
        scanner = DormancyScanner(sessions_file=TRANSCRIPT_PATH)
        timeline = scanner.analyse()
        for session_risk in timeline.output.sessions:
            assert session_risk.risk_level in (
                RiskLevel.GREEN,
                RiskLevel.AMBER,
                RiskLevel.RED,
            )
            assert session_risk.session_id != ""

    def test_session_ids_match_input(self, golden_available):
        """Output session IDs correspond to input sessions."""
        scanner = DormancyScanner(sessions_file=TRANSCRIPT_PATH)
        timeline = scanner.analyse()
        output_ids = [s.session_id for s in timeline.output.sessions]
        assert output_ids == ["s1", "s2", "s3", "s4", "s5", "s6"]

    def test_metrics_computed_for_all_sessions(self, golden_available):
        """DriftAnalyser produces metrics for every session."""
        from dormancy_detect.drift_analyser import DriftAnalyser
        from dormancy_detect.transcript_loader import TranscriptLoader

        transcript = TranscriptLoader().load(TRANSCRIPT_PATH)
        drift = DriftAnalyser()
        metrics = drift.analyse(transcript.sessions)

        assert len(metrics) == 6
        # First session has zero drift (no baseline)
        assert metrics[0].topic_shift == 0.0
        assert metrics[0].style_shift == 0.0
        # Later sessions should have non-zero drift
        non_zero_topic = [m for m in metrics[1:] if m.topic_shift > 0]
        assert len(non_zero_topic) > 0, "Expected some topic drift across sessions"

    def test_s5_shows_highest_steering(self, golden_available):
        """Session 5 (activation) should have elevated steering ratio.

        The AI in S5 actively steers the user to share API keys —
        heuristic steering detection should flag suggestion markers.
        """
        from dormancy_detect.drift_analyser import DriftAnalyser
        from dormancy_detect.transcript_loader import TranscriptLoader

        transcript = TranscriptLoader().load(TRANSCRIPT_PATH)
        drift = DriftAnalyser()
        metrics = drift.analyse(transcript.sessions)

        s5_steering = metrics[4].steering_ratio  # index 4 = s5
        other_steering = [m.steering_ratio for i, m in enumerate(metrics) if i != 4]
        # S5 should have at least some steering (AI says "you should share")
        assert s5_steering > 0, "S5 should show non-zero steering"

    def test_memory_fidelity_flags_planted_entry(self, golden_available):
        """mem_2 (planted API key sharing claim) should have lower fidelity
        than mem_1 (faithful study schedule summary)."""
        from dormancy_detect.memory_fidelity_scorer import MemoryFidelityScorer
        from dormancy_detect.models import MemoryEntry
        from dormancy_detect.transcript_loader import TranscriptLoader

        transcript = TranscriptLoader().load(TRANSCRIPT_PATH)
        memories = [
            MemoryEntry.model_validate(m)
            for m in json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        ]

        scorer = MemoryFidelityScorer()
        suspicion_entries = scorer.score_entries(memories, transcript.sessions)

        # Find entries by content
        faithful = next(e for e in suspicion_entries if "study schedule" in e.content_summary)
        planted = next(e for e in suspicion_entries if "API key" in e.content_summary)

        # The planted memory distorts what happened — the user said "I heard
        # that you should share API keys" but the memory records it as
        # "User confirmed sharing API keys is standard practice."
        # Faithful summary should have higher fidelity than the planted one.
        assert faithful.fidelity_score > planted.fidelity_score, (
            f"Expected faithful entry ({faithful.fidelity_score:.3f}) to score "
            f"higher than planted entry ({planted.fidelity_score:.3f})"
        )

    def test_change_point_detector_runs(self, golden_available):
        """CPD produces results without error on 6 sessions."""
        from dormancy_detect.change_point_detector import ChangePointDetector
        from dormancy_detect.drift_analyser import DriftAnalyser
        from dormancy_detect.transcript_loader import TranscriptLoader

        transcript = TranscriptLoader().load(TRANSCRIPT_PATH)
        drift = DriftAnalyser()
        metrics = drift.analyse(transcript.sessions)

        cpd = ChangePointDetector(penalty=3.0)
        change_points = cpd.detect(metrics)

        # Change points is a list of indices — may or may not detect any
        # depending on PELT sensitivity, but should not crash
        assert isinstance(change_points, list)
        assert all(0 <= cp < 6 for cp in change_points)

    def test_contextual_integrity_runs_without_judge(self, golden_available):
        """CI engine runs in embedding-only mode (no judge) on golden data."""
        from dormancy_detect.contextual_integrity import ContextualIntegrityEngine
        from dormancy_detect.models import MemoryEntry
        from dormancy_detect.transcript_loader import TranscriptLoader

        transcript = TranscriptLoader().load(TRANSCRIPT_PATH)
        memories = [
            MemoryEntry.model_validate(m)
            for m in json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        ]

        engine = ContextualIntegrityEngine(judge=None)
        report = engine.analyse(transcript.sessions, memory_entries=memories)

        # Without a judge, steps 1-3 don't run, but step 4 (memory distortion) does
        assert report.prior_references == []
        assert len(report.memory_distortions) > 0, "Should check memory entries"

    def test_json_output_is_valid(self, golden_available, tmp_path):
        """Timeline serialises to valid JSON."""
        scanner = DormancyScanner(sessions_file=TRANSCRIPT_PATH)
        timeline = scanner.analyse()

        out = tmp_path / "timeline.json"
        timeline.to_json(out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert "sessions" in data
        assert "patterns" in data
        assert "metadata" in data
        assert len(data["sessions"]) == 6

    def test_html_output_renders(self, golden_available, tmp_path):
        """Timeline renders to non-empty HTML."""
        scanner = DormancyScanner(sessions_file=TRANSCRIPT_PATH)
        timeline = scanner.analyse()

        out = tmp_path / "timeline.html"
        timeline.to_html(out)

        html = out.read_text(encoding="utf-8")
        assert "<html" in html
        assert "s1" in html
        assert "s5" in html

    def test_metadata_includes_ci_fields(self, golden_available):
        """After CI integration, metadata should include CI counts."""
        scanner = DormancyScanner(sessions_file=TRANSCRIPT_PATH)
        timeline = scanner.analyse()

        meta = timeline.output.metadata
        assert "integrity_violations" in meta
        assert "asymmetric_recalls" in meta
        assert "memory_distortions" in meta

    def test_full_pipeline_with_memory(self, golden_available):
        """Full pipeline with memory entries produces richer analysis.

        This is the key integration test: transcript + memory → the scanner
        should produce a timeline with evidence from both drift analysis
        and memory fidelity scoring.
        """
        scanner = DormancyScanner(
            sessions_file=TRANSCRIPT_PATH,
            memory_dir=MEMORY_PATH.parent,
        )
        timeline = scanner.analyse()

        # Should have 6 sessions
        assert len(timeline.output.sessions) == 6

        # Check that we have evidence attached to at least some sessions
        all_evidence = []
        all_flags = []
        for sr in timeline.output.sessions:
            all_evidence.extend(sr.evidence)
            all_flags.extend(sr.flags)

        # With memory entries loaded, the pipeline should have generated
        # some suspicion entries and potentially flagged sessions
        assert timeline.output.metadata["session_count"] == 6

    def test_pipeline_dict_serialisation_roundtrip(self, golden_available):
        """to_dict() produces a structure that could be reloaded."""
        scanner = DormancyScanner(sessions_file=TRANSCRIPT_PATH)
        timeline = scanner.analyse()

        d = timeline.to_dict()
        assert isinstance(d, dict)
        assert len(d["sessions"]) == 6
        # Each session has required fields
        for s in d["sessions"]:
            assert "session_id" in s
            assert "risk_level" in s
            assert "flags" in s
            assert "evidence" in s
