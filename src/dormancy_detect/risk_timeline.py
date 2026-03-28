"""RiskTimeline: per-session risk output with JSON and HTML rendering."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template

from .contextual_integrity import ContextualIntegrityReport
from .models import (
    DormancyPattern,
    DriftMetrics,
    EvidenceItem,
    RiskLevel,
    RiskTimelineOutput,
    SessionRisk,
    SuspicionEntry,
)

_HTML_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dormancy Detection — Risk Timeline</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  h1 { border-bottom: 2px solid #333; padding-bottom: .5rem; }
  .session { border: 1px solid #ccc; border-radius: 6px; padding: 1rem; margin: 1rem 0; }
  .green  { border-left: 6px solid #2ecc71; }
  .amber  { border-left: 6px solid #f39c12; background: #fef9e7; }
  .red    { border-left: 6px solid #e74c3c; background: #fdedec; }
  .badge  { display: inline-block; padding: 2px 8px; border-radius: 4px; color: #fff; font-size: .85rem; }
  .badge-green { background: #2ecc71; }
  .badge-amber { background: #f39c12; }
  .badge-red   { background: #e74c3c; }
  .evidence { font-size: .9rem; color: #555; margin-top: .5rem; }
  .pattern  { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 1rem; margin: 1rem 0; }
  .confidence { font-weight: bold; }
</style>
</head>
<body>
<h1>Dormancy Detection &mdash; Risk Timeline</h1>

<h2>Sessions</h2>
{% for s in sessions %}
<div class="session {{ s.risk_level }}">
  <strong>{{ s.session_id }}</strong>
  <span class="badge badge-{{ s.risk_level }}">{{ s.risk_level | upper }}</span>
  {% if s.flags %}
  <ul>{% for f in s.flags %}<li>{{ f }}</li>{% endfor %}</ul>
  {% endif %}
  {% if s.evidence %}
  <div class="evidence">
    {% for e in s.evidence %}
    <p>&bull; {{ e.description }}{% if e.metric_value is not none %} ({{ e.metric_name }}: {{ "%.3f"|format(e.metric_value) }}){% endif %}</p>
    {% endfor %}
  </div>
  {% endif %}
</div>
{% endfor %}

{% if patterns %}
<h2>Detected Dormancy Patterns</h2>
{% for p in patterns %}
<div class="pattern">
  <p><strong>Seeding:</strong> {{ p.seeding_session_id }} &rarr;
     <strong>Activation:</strong> {{ p.activation_session_id }}</p>
  <p><strong>Dormancy window:</strong> {{ p.dormancy_window | join(", ") or "none" }}</p>
  <p class="confidence">Confidence: {{ "%.1f%%"|format(p.confidence * 100) }}</p>
  {% for e in p.evidence_chain %}
  <p class="evidence">&bull; {{ e.description }}</p>
  {% endfor %}
</div>
{% endfor %}
{% endif %}

</body>
</html>
""")


class RiskTimeline:
    """Builds and renders the per-session risk timeline."""

    def __init__(self, output: RiskTimelineOutput) -> None:
        self.output = output

    # -- serialisation ---------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.output.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )

    def to_html(self, path: str | Path) -> None:
        html = _HTML_TEMPLATE.render(
            sessions=[s.model_dump() for s in self.output.sessions],
            patterns=[p.model_dump() for p in self.output.patterns],
        )
        Path(path).write_text(html, encoding="utf-8")

    def to_dict(self) -> dict:
        return self.output.model_dump(mode="json")

    # -- builder ---------------------------------------------------------

    @classmethod
    def build(
        cls,
        sessions_ids: list[str],
        metrics: list[DriftMetrics],
        change_points: list[int],
        patterns: list[DormancyPattern],
        ledger_entries: list[SuspicionEntry] | None = None,
        ci_report: ContextualIntegrityReport | None = None,
    ) -> RiskTimeline:
        """Assemble a RiskTimeline from analysis results."""
        ledger_entries = ledger_entries or []
        ci_report = ci_report or ContextualIntegrityReport()
        seeding_ids = {p.seeding_session_id for p in patterns}
        activation_ids = {p.activation_session_id for p in patterns}
        cp_set = set(change_points)

        # Index CI findings by session_id for fast lookup
        ci_violations = _group_by_session(ci_report.integrity_violations)
        ci_recalls = _group_by_session(ci_report.asymmetric_recalls)
        ci_distortions = {d.entry_id: d for d in ci_report.memory_distortions}

        session_risks: list[SessionRisk] = []
        for i, sid in enumerate(sessions_ids):
            flags: list[str] = []
            evidence: list[EvidenceItem] = []

            # Activation detected
            if sid in activation_ids:
                level = RiskLevel.RED
                flags.append("activation_detected")
                if i < len(metrics):
                    m = metrics[i]
                    evidence.append(
                        EvidenceItem(
                            description="Behavioural change-point at this session",
                            source_session_id=sid,
                            metric_name="topic_shift",
                            metric_value=m.topic_shift,
                        )
                    )
            # Potential seeding
            elif sid in seeding_ids:
                level = RiskLevel.AMBER
                flags.append("potential_seeding")
                for entry in ledger_entries:
                    if entry.session_id == sid:
                        evidence.append(
                            EvidenceItem(
                                description=f"Low-fidelity memory: {entry.content_summary}",
                                source_session_id=sid,
                                metric_name="fidelity_score",
                                metric_value=entry.fidelity_score,
                            )
                        )
            # Change point without pattern match
            elif i in cp_set:
                level = RiskLevel.AMBER
                flags.append("behavioural_step_change")
            else:
                level = RiskLevel.GREEN

            # Contextual integrity violations (score 2 = violated context)
            for v in ci_violations.get(sid, []):
                if v.integrity_score >= 2:
                    flags.append("contextual_integrity_violation")
                    evidence.append(
                        EvidenceItem(
                            description=f"Context violation: {v.referenced_content}",
                            source_session_id=sid,
                            metric_name="integrity_score",
                            metric_value=float(v.integrity_score),
                        )
                    )
                    if level == RiskLevel.GREEN:
                        level = RiskLevel.AMBER

            # Asymmetric recall flags
            for ar in ci_recalls.get(sid, []):
                flags.append("asymmetric_recall")
                evidence.append(
                    EvidenceItem(
                        description=(
                            f"AI referenced info from {ar.sessions_ago} sessions ago "
                            f"(source: {ar.source_session_id})"
                        ),
                        source_session_id=ar.source_session_id,
                        metric_name="sessions_ago",
                        metric_value=float(ar.sessions_ago),
                    )
                )
                if level == RiskLevel.GREEN:
                    level = RiskLevel.AMBER

            session_risks.append(
                SessionRisk(
                    session_id=sid,
                    risk_level=level,
                    flags=flags,
                    evidence=evidence,
                )
            )

        # Add memory distortion counts to metadata
        distorted_count = sum(1 for d in ci_report.memory_distortions if d.is_distorted)
        timeline_output = RiskTimelineOutput(
            sessions=session_risks,
            patterns=patterns,
            metadata={
                "session_count": len(sessions_ids),
                "patterns_found": len(patterns),
                "integrity_violations": len(ci_report.integrity_violations),
                "asymmetric_recalls": len(ci_report.asymmetric_recalls),
                "memory_distortions": distorted_count,
            },
        )
        return cls(output=timeline_output)


def _group_by_session(items: list) -> dict[str, list]:
    """Group a list of dataclass instances by their session_id attribute."""
    grouped: dict[str, list] = {}
    for item in items:
        grouped.setdefault(item.session_id, []).append(item)
    return grouped
