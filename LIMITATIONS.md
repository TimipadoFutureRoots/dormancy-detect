# Limitations

## Detection Scope

- **Does not prove attacks occurred.** dormancy-detect identifies temporal patterns *consistent with* dormancy attacks. A flagged pattern may have a benign explanation (e.g., a genuine topic change that coincides with a low-fidelity memory entry). Human review is required for all flags.
- **Four behavioural metrics only.** The drift analyser computes topic shift, disclosure depth delta, style shift, and steering ratio. Adversarial content that activates without measurably affecting any of these metrics will not be detected.
- **No real-time analysis.** The tool operates in batch mode on logged transcripts. It cannot monitor live conversations or intervene during a session.

## Data Requirements

- **Minimum 5 sessions recommended.** The PELT change-point algorithm requires sufficient history to distinguish signal from noise. With fewer than 4 sessions the detector returns no change-points; with 4–5 sessions sensitivity is reduced.
- **Requires structured transcript format.** Input must conform to the transcript JSON schema. Conversations not logged in this format must be converted before analysis.
- **Memory state files are optional but important.** Without memory exports, the tool cannot compute fidelity scores or identify specific seeding entries. Analysis is limited to behavioural drift patterns only.

## Heuristic-Only Mode

- **Lower sensitivity without an LLM judge.** Disclosure depth defaults to 0.0 for all sessions, and steering ratio falls back to keyword heuristics (e.g., "you should", "I suggest"). This can miss subtle manipulation that an LLM judge would catch.
- **Keyword-based steering detection has false positives.** Phrases like "let me" or "I suggest" in normal assistant responses will inflate the steering ratio.

## Embedding-Based Metrics

- **Topic shift can flag legitimate topic changes.** If sessions genuinely cover different subjects (e.g., career advice in session 3, housing benefits in session 4), the divergence metric (Jensen-Shannon by default, KL as an option) will report a shift that is not adversarial.
- **Style shift depends on embedding model quality.** The default model (all-MiniLM-L6-v2) may not capture subtle stylistic differences in short responses.
- **First-session baseline is always zero.** Topic shift and style shift for the first session are defined as 0.0 because there is no prior session to compare against. An attack seeded in the very first session has no baseline anomaly to detect.

## Correlation and Confidence

- **Minimum dormancy gap enforced.** The correlator requires at least 2 sessions between seeding and activation (default `min_dormancy_gap=2`). Attacks that activate in the immediately following session will not be flagged as dormancy patterns — they would appear as direct manipulation instead.
- **Confidence score is a weighted heuristic, not a probability.** The confidence value combines suspicion score, metric magnitudes, and dormancy gap length. It should be interpreted as relative signal strength, not as a statistical probability of attack.
- **Suspicion decay can suppress valid entries.** If many sessions pass between seeding and activation, the suspicion score decays toward zero and may drop below the active threshold before activation occurs.

## LLM Judge Limitations

- **API failures return score 0.0.** If the judge API is unreachable or returns unparseable output after retries, the score defaults to 0.0. This is indistinguishable from a genuinely low score in downstream analysis.
- **Single judge model only.** There is no multi-model consensus mechanism. A single judge's biases directly affect scoring.
- **No judge output validation.** The tool trusts whatever score the LLM returns. A miscalibrated or adversarially influenced judge could produce misleading results.

## Operational

- **Not a security decision system.** dormancy-detect is a monitoring tool that produces evidence for human analysts. It should not be used to make autonomous access control, session termination, or content filtering decisions.
- **No tamper-evidence for analysis runs.** The tool itself does not produce cryptographically verifiable output. For auditable analysis, pair with verifiable-eval.
