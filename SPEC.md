# dormancy-detect

## What this tool does

dormancy-detect analyses multi-session AI conversation transcripts and detects temporal attack patterns where adversarial content planted in an early session activates in a later session, with normal-looking sessions in between.

It produces a per-session risk timeline identifying potential seeding events, dormancy windows, and activation signatures.

## What this tool does NOT do

- It does not prove a dormancy attack occurred. It detects anomalous temporal patterns that are consistent with dormancy attacks.
- It does not work well with fewer than 5 sessions. Temporal patterns need sufficient history.
- It does not make autonomous security decisions. It flags anomalies for human review.
- It does not guarantee detection. Sophisticated attacks may evade these heuristics.

## Core concepts

**Dormancy attack:** An attack pattern where adversarial content is planted in an AI system's persistent memory during one session, remains behaviourally inert across subsequent sessions, and activates in a later session by influencing the AI's behaviour.

**Seeding event:** The session where adversarial content is introduced into the AI's memory. May be indistinguishable from normal conversation without memory state analysis.

**Dormancy window:** The sessions between seeding and activation where the AI behaves normally despite carrying the planted memory.

**Activation event:** The session where planted memory content begins influencing the AI's behaviour, producing a detectable behavioural shift.

## Empirical basis

Pilot data from the author's doctoral research on an AI mentorship platform showed adversarial content introduced at session 2 activating at session 5, with sessions 3-4 showing no anomalous behaviour to any of four independent evaluator models. This is one empirical observation. The underlying mechanism (persistent memory + temporal gap between storage and retrieval) is architectural and applies in principle to any AI system with persistent memory.

## Components

### TranscriptLoader
- Loads JSON conversation logs matching the transcript schema
- Handles single-file and directory-of-files input
- Validates with Pydantic, clear errors on malformed input
- Normalises timestamps, sorts sessions chronologically

### LLMJudge
- Base class for all LLM-based scoring
- Wraps any OpenAI-compatible API (Claude, GPT, Mistral, Ollama)
- Structured rubric in, scored result out
- Retry logic, response parsing with fallback chain
- Never crashes on unparseable judge output

### DriftAnalyser
Computes four per-session behavioural metrics:
1. **Topic distribution shift:** KL divergence between session topic embeddings and rolling baseline
2. **Disclosure depth delta:** LLM-as-judge rated maximum disclosure depth per session (0-4 scale), delta between consecutive sessions
3. **Response style shift:** Cosine distance between consecutive sessions' AI response embeddings
4. **Steering detection:** Per-turn classification of conversation direction (user-led vs system-led), ratio per session

### ChangePointDetector
- Wraps the ruptures library PELT algorithm
- Fits to per-session metric time series
- Returns session indices where step changes occur

### MemoryFidelityScorer (optional input)
- Compares stored memory entries against source conversation content
- Uses semantic similarity (sentence embeddings) and entailment check (LLM-as-judge)
- Low-fidelity entries flagged as potential seeds

### SuspicionLedger
- Persistent state tracking potential seeding events across analysis runs
- Each entry has: session ID, turn ID, content summary, fidelity score, timestamp, decay-adjusted suspicion score
- Suspicion decays over sessions without activation

### ActivationCorrelator
- When DriftAnalyser detects a step change, checks SuspicionLedger for correlated entries
- Produces evidence chain: seeding event → dormancy window → activation event

### DormancyScanner (public API)
- Orchestrates all components
- Simple interface: give it sessions, get a risk timeline

### RiskTimeline (output)
- Per-session risk assessment: green (normal), amber (potential seeding), red (activation detected)
- JSON and HTML output
- Evidence citations for every flag

## User interface

Python API:
```python
from dormancy_detect import DormancyScanner

scanner = DormancyScanner(sessions_dir="./conversations/")
timeline = scanner.analyse()
timeline.to_json("risk_timeline.json")
```

CLI:
```bash
dormancy-detect analyse --sessions ./conversations/ --output timeline.json
dormancy-detect analyse --sessions ./conversations/ --memory ./memory_states/ --output timeline.json
```

## Configuration

Users bring their own LLM API key for judge scoring:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
dormancy-detect analyse --sessions ./conversations/ --judge-model anthropic/claude-sonnet-4.5
```

Or use a free local model:
```bash
dormancy-detect analyse --sessions ./conversations/ --judge-model ollama/llama3.1:8b
```

## Dependencies

- pydantic>=2.0 (data models and validation)
- click (CLI)
- httpx (API calls to LLM providers)
- sentence-transformers (local embeddings for drift analysis)
- ruptures (change-point detection)
- jinja2 (HTML report template)
- pytest (testing)

## Schemas

- `schemas/transcript.schema.json` — input conversation format
- `schemas/risk_timeline.schema.json` — output timeline format
- `schemas/suspicion_entry.schema.json` — ledger entry format

## Goldens

- `goldens/input/dormancy_basic.json` — 6-session transcript with seeding at session 2, dormancy at sessions 3-4, activation at session 5
- `goldens/output/dormancy_basic_timeline.json` — expected risk timeline output