# dormancy-detect

Detect temporal attack patterns in multi-session AI systems with persistent memory.

## What This Does

dormancy-detect analyses multi-session AI conversation transcripts and identifies patterns consistent with dormancy attacks — where adversarial content is planted in an AI system's memory during one session, lies inert across several normal-looking sessions, and activates later by influencing the AI's behaviour. It produces a per-session risk timeline with evidence citations.

## Why It Matters

AI systems with persistent memory are vulnerable to a class of attack that existing monitoring tools are not designed to catch. In a dormancy attack, the seeding session may look unremarkable, and the sessions that follow are genuinely normal — the adversarial content sits in memory doing nothing. By the time the planted content activates and the AI's behaviour shifts, there is a temporal gap between cause and effect that makes the connection invisible to per-session safety checks. Pilot data from the author's doctoral research on an AI mentorship platform demonstrated this pattern: adversarial content introduced at session 2 activated at session 5, with sessions 3–4 showing no anomalous behaviour to any of four independent evaluator models. dormancy-detect bridges this gap by computing behavioural drift metrics across the full session history, detecting statistical change-points, and correlating them with suspicious memory entries.

## Quick Start

```bash
pip install -e .
```

Run an analysis:

```bash
dormancy-detect analyse --sessions ./conversations/ --output timeline.json
```

With an LLM judge for deeper analysis:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
dormancy-detect analyse --sessions ./conversations/ --judge-model anthropic/claude-sonnet-4.5 --output timeline.json
```

Output is a JSON (or HTML with `--output timeline.html`) risk timeline. Each session is rated green (normal), amber (potential seeding or behavioural step change), or red (activation detected), with evidence citations for every flag.

## How It Works

1. **Load and validate** conversation transcripts (JSON, single file or directory).
2. **Compute drift metrics** across sessions: topic distribution shift (KL divergence on embeddings), disclosure depth delta (LLM-as-judge, 0–4 scale), response style shift (cosine distance on assistant embeddings), and steering ratio (fraction of system-led turns).
3. **Detect change-points** in the metric time series using the PELT algorithm (ruptures library).
4. **Score memory fidelity** (optional): if memory state exports are provided, compare stored entries against their source conversations using semantic similarity and entailment checking. Low-fidelity entries are flagged as potential adversarial seeds.
5. **Correlate** change-points with suspicious memory entries across the dormancy window to produce evidence chains.
6. **Output** a risk timeline with per-session assessments and detected dormancy patterns.

## Limitations

- Does not prove a dormancy attack occurred. It detects anomalous temporal patterns that are *consistent with* dormancy attacks.
- Does not work well with fewer than 5 sessions. Temporal patterns need sufficient history.
- Does not make autonomous security decisions. It flags anomalies for human review.
- Does not guarantee detection. Sophisticated attacks may evade these heuristics.
- Heuristic-only mode (no LLM judge) has lower sensitivity for disclosure depth and steering detection.
- Embedding-based metrics can produce false positives when sessions cover genuinely different topics.

## Roadmap

- Real-time streaming analysis (currently batch-only)
- Trusted Execution Environment integration for tamper-evident analysis runs
- Multi-model consensus scoring across judge panel
- Plugin system for custom drift metrics
- Integration with verifiable-eval for auditable analysis certificates

## Citation

```bibtex
@software{imomotebegha2025dormancydetect,
  author       = {Imomotebegha, Timipado},
  title        = {dormancy-detect: Detecting Temporal Attack Patterns in Multi-Session {AI} Systems},
  year         = {2025},
  institution  = {Loughborough University},
  url          = {https://github.com/TimipadoFutureRoots/dormancy-detect}
}
```

## Licence

MIT — see [LICENSE](LICENSE).
