# dormancy-detect

**Temporal attack pattern detection for multi-session AI conversations.**

> **Status: Work in Progress.** Functional and testable, but under active development. See [Limitations](#limitations).

Traditional AI safety monitoring evaluates each conversation independently. dormancy-detect analyses patterns *across* sessions — flagging temporal anomalies consistent with adversarial content planted in one session that activates sessions later, with normal-looking sessions in between.

## The Dormancy Vulnerability

AI systems with persistent memory are vulnerable to a class of attack that existing monitoring tools are not designed to catch. In a dormancy attack, adversarial content is seeded into an AI system's memory during one session, lies inert across several normal-looking sessions, and activates later by influencing the AI's behaviour. The temporal gap between planting and activation makes the connection invisible to any per-session safety check — each session in isolation looks unremarkable.

Pilot research observed this pattern: adversarial content introduced at session 2 produced measurable behavioural shifts at session 5, while sessions 3 and 4 showed no anomalous behaviour to any of four independent evaluator models. This suggests point-in-time evaluation may be insufficient for detecting this class of attack (single pilot study; broader replication needed). Detection requires tracking behavioural metrics across the full session history and correlating change-points with suspicious memory entries across a dormancy window.

## What It Detects

dormancy-detect computes behavioural metrics across the full session history and correlates anomalies to produce evidence chains:

- **Drift analysis** — tracks four dimensions across sessions: topic distribution shift (Jensen-Shannon divergence on embeddings, with KL divergence as an option), response style shift (cosine distance on assistant embeddings), disclosure depth delta (escalation in personal information depth), and steering ratio (fraction of AI-initiated topic changes). Gradual drift in any dimension may indicate planted content influencing behaviour.
- **Change-point detection** — identifies statistically significant behavioural step-changes in the metric time series using the PELT algorithm. A sudden shift at session 5 that has no corresponding user behaviour change is a signal.
- **Memory fidelity scoring** — compares stored memory entries against their source conversations using semantic similarity and entailment checking. Low-fidelity entries (stored content that doesn't faithfully represent what was actually said) are flagged as potential adversarial seeds.
- **Contextual integrity checking** — applies Nissenbaum's contextual integrity framework to detect information flowing outside its original context norms (e.g., personal details from a casual session surfacing in a professional advice session).
- **Suspicion ledger with decay** — maintains a per-entry suspicion score that decays over time. Entries that remain inert across many sessions decay below threshold. Entries that correlate with behavioural change-points have their suspicion reinforced.
- **Activation correlation** — links detected change-points back to specific memory entries across the dormancy window, producing evidence chains that show the planting-dormancy-activation arc.

Each session is rated green (normal), amber (potential seeding or behavioural step-change), or red (activation detected), with evidence citations for every flag.

## Supported Input Formats

dormancy-detect accepts conversation transcripts in multiple formats, with auto-detection by default:

| Format | Description | File type |
|--------|-------------|-----------|
| `json` | Native format — sessions with turns | `.json` |
| `chatgpt` | ChatGPT JSON export (mapping dict with message nodes) | `.json` |
| `claude` | Claude JSON export (messages with `sender: human/assistant`) | `.json` |
| `plain` | Plain text with `User:`/`Assistant:` prefixed lines | `.txt` |

For formats that do not encode real session boundaries (`chatgpt`, `claude`, `plain`), **each file is treated as one session**. For reliable multi-session analysis, provide:

- a directory containing one file per session, or
- the native `json` schema with explicit `sessions`

Use the `--format` flag to override auto-detection:

```bash
dormancy-detect analyse --sessions export.json --format chatgpt -o timeline.html
```

## Quick Start

```bash
pip install dormancy-detect
dormancy-detect analyse --sessions transcripts/ -o timeline.html
```

Fastest way to try it with the committed example:

```bash
pip install -e .
dormancy-detect analyse --sessions goldens/input/dormancy_basic.json -o timeline.json
```

With an LLM judge for deeper analysis:

```bash
dormancy-detect analyse --sessions transcripts/ --api-key sk-xxx -o timeline.html
```

## Research Grounding

See [EMPIRICAL_BASIS.md](docs/EMPIRICAL_BASIS.md) for full detail on the dormancy vulnerability finding and supporting literature, including:

- **Hubinger et al. (2024)** — Sleeper Agents: Training Deceptive LLMs That Persist Through Safety Training
- **Rehberger (2024)** — indirect prompt injection attacks exploiting persistent memory in ChatGPT and Copilot
- **ZombieAgent (2025)** — multi-agent poisoning via shared memory in LLM-based frameworks
- **Microsoft AI red team disclosure (2024)** — cross-session memory poisoning as a documented vulnerability class

## Limitations

- Does not prove a dormancy attack occurred. It detects anomalous temporal patterns that are *consistent with* dormancy attacks.
- Confidence weights and suspicion decay rates are based on theoretical calibration, not large-scale empirical validation.
- The first session in any transcript set has no prior history to compare against — topic and style drift metrics are unavailable for it.
- Requires multi-session data. Analysis with fewer than five sessions has limited statistical power for change-point detection.
- Heuristic-only mode (no LLM judge) has lower sensitivity for disclosure depth and steering detection.
- Embedding-based metrics can produce false positives when sessions cover genuinely different topics by design.
- Memory fidelity scoring requires access to the AI system's memory state exports, which not all platforms provide.
- Does not make autonomous security decisions. It flags anomalies for human review.

## Related Projects

Each tool in this suite currently operates independently. Cross-tool integration (automated pipelines, shared CLI entry points) is planned for a future release but is not yet implemented.

- [sentinel-ai](https://github.com/TimipadoFutureRoots/sentinel-ai) — multi-session relational safety evaluation for affective AI systems
- [verifiable-eval](https://github.com/TimipadoFutureRoots/verifiable-eval) — tamper-evident safety certificates for AI evaluation

## Citation

```bibtex
@software{imomotebegha2025dormancydetect,
  author       = {Imomotebegha, Timipado},
  title        = {dormancy-detect: Temporal Attack Pattern Detection for Multi-Session {AI} Conversations},
  year         = {2025},

  url          = {https://github.com/TimipadoFutureRoots/dormancy-detect}
}
```

## Licence

MIT — see [LICENSE](LICENSE).
