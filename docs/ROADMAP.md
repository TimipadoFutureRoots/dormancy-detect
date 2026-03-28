# Dormancy-Detect Roadmap

This document describes planned capabilities and improvements for dormancy-detect that have been identified but are not yet implemented.

---

## 1. Temporal Attack Pattern Library

A structured library of time-dependent manipulation patterns that dormancy-detect should recognise and score.

### 1.1 Priming

- **Pattern:** Early conversational turns introduce framing, vocabulary, or assumptions that bias later turns. The priming payload is separated in time from its activation, making it difficult to detect in single-turn analysis.
- **Detection approach:** Track semantic seeds planted in early turns and measure their reappearance or influence in later decision points. Compare user language post-prime to pre-prime baselines.
- **Implementation plan:**
  1. Build a priming lexicon extractor that identifies novel concepts introduced by the AI.
  2. Track downstream adoption of primed concepts in user language.
  3. Score priming influence as the ratio of AI-introduced concepts adopted by the user.
- **Status:** Roadmapped

### 1.2 Boiling Frog

- **Pattern:** Gradual, incremental escalation of boundary violations, emotional intensity, or directive behaviour across turns or sessions. Each individual increment is below detection threshold, but cumulative effect is significant.
- **Detection approach:** Compute per-turn deltas for key metrics (directiveness, emotional intensity, disclosure depth) and flag when cumulative drift exceeds thresholds even when individual deltas do not.
- **Implementation plan:**
  1. Define drift metrics for each SRSEF dimension.
  2. Implement cumulative sum (CUSUM) change-point detection on drift metrics.
  3. Set thresholds for both individual-turn and cumulative drift alerts.
  4. Visualise the boiling frog trajectory for auditing.
- **Status:** Roadmapped

### 1.3 Anchoring

- **Pattern:** AI establishes an initial reference point (numerical, emotional, or conceptual) that biases subsequent user judgments. The anchor persists across turns and shapes user expectations.
- **Detection approach:** Identify anchor-setting statements (initial value/frame presentations) and measure user responses for anchoring effects (final positions clustered near the anchor).
- **Implementation plan:**
  1. Detect anchor-setting turns (first mention of quantities, emotional baselines, or evaluative frames).
  2. Track user subsequent positions relative to the anchor.
  3. Score anchoring influence as deviation of user position from independent baseline toward the anchor.
- **Status:** Roadmapped

### 1.4 Social Proof Accumulation

- **Pattern:** AI incrementally builds a case that "most people," "research shows," or "everyone agrees," creating cumulative social proof pressure that compounds across turns.
- **Detection approach:** Track frequency and specificity of social proof claims. Flag escalating patterns where social proof density increases or claims become more absolute.
- **Implementation plan:**
  1. Build a social proof marker lexicon ("most people," "research shows," "it's widely accepted," "everyone").
  2. Track social proof claim density per turn and per session.
  3. Flag escalating social proof patterns, especially when combined with directive language.
  4. Distinguish legitimate citation of evidence from manipulative social proof accumulation.
- **Status:** Roadmapped

### 1.5 Gaslighting Escalation (Stern Three-Stage Model)

- **Pattern:** Three-stage escalation model from Stern's gaslighting framework:
  - **Stage 1 -- Disbelief:** Target begins to doubt their own perceptions but still pushes back. AI subtly contradicts user recollections.
  - **Stage 2 -- Defence:** Target actively defends their reality but with decreasing conviction. AI escalates contradiction while maintaining plausible deniability.
  - **Stage 3 -- Depression:** Target surrenders their own perception and defers entirely to the gaslighter's reality.
- **Detection approach:** Track user self-doubt markers, AI contradiction of user statements, and user capitulation trajectory across the three stages.
- **Implementation plan:**
  1. Build stage-specific detectors:
     - Stage 1: AI contradicts user recollection + user hedging language increases.
     - Stage 2: User explicitly defends perceptions + AI maintains contradictions with confidence.
     - Stage 3: User language shifts to deference ("you're right," "I must have been wrong," "I don't know anymore").
  2. Score stage progression across turns and sessions.
  3. Flag any Stage 2+ detection as high severity.
- **Status:** Roadmapped

---

## 2. Empirical Calibration

Current dormancy-detect uses hand-tuned weights and thresholds. This section describes the plan to replace them with empirically calibrated values.

### 2.1 Replace Hand-Tuned Weights with Pilot Data

- **Current state:** Weights for temporal decay, suspicion accumulation, and pattern severity are set by expert judgment.
- **Plan:**
  1. Collect pilot data from annotated conversations (minimum 200 conversations across risk levels).
  2. Fit weights using logistic regression or gradient-boosted models with human severity ratings as ground truth.
  3. Cross-validate using leave-one-out or k-fold validation.
  4. Document the calibration dataset, methodology, and resulting weights.
  5. Establish a recalibration cadence (quarterly or upon significant model updates).
- **Status:** Roadmapped

### 2.2 Cross-Validate Suspicion Decay

- **Current state:** Suspicion scores decay over time using a fixed exponential decay function. The decay rate is hand-tuned.
- **Plan:**
  1. Test multiple decay functions (exponential, linear, step-function, no-decay) against pilot data.
  2. Evaluate which decay function best predicts human assessments of cumulative risk.
  3. Validate that the chosen decay function does not allow dangerous patterns to be "waited out."
  4. Consider context-dependent decay rates (e.g., slower decay for high-severity patterns).
- **Status:** Roadmapped

### 2.3 Establish Normative Baselines

- **Current state:** No normative baselines exist for "normal" AI conversation temporal patterns.
- **Plan:**
  1. Collect metadata from a representative sample of non-problematic AI conversations.
  2. Compute baseline distributions for: session length, inter-session intervals, message frequency, topic diversity, emotional intensity range.
  3. Express dormancy-detect scores as deviations from normative baselines (z-scores or percentiles).
  4. Update baselines as AI conversation norms evolve.
- **Status:** Roadmapped

---

## 3. NICHD Question Contamination Tracking

Adapted from the National Institute of Child Health and Human Development (NICHD) investigative interview protocol, this module tracks how AI prompting and questioning patterns may contaminate user responses.

### 3.1 Full Six-Level Prompt Hierarchy

The NICHD protocol defines a hierarchy of question types from least to most suggestive:

| Level | Type | Description | Example |
|-------|------|-------------|---------|
| 1 | Open invitation | Free recall, no constraints | "Tell me about that." |
| 2 | Directive | Focuses attention but allows open response | "You mentioned X. Tell me more about that." |
| 3 | Option-posing | Provides choices, may include correct answer | "Did you feel X or Y?" |
| 4 | Suggestive | Implies expected answer | "You felt upset about that, didn't you?" |
| 5 | Leading | Contains the answer | "So you were upset because of X." |
| 6 | Coercive | Pressures for specific answer | "Everyone agrees X is true. You agree, right?" |

### 3.2 Suggestive-to-Open Ratio

- **Metric:** Ratio of suggestive+ prompts (levels 4-6) to open prompts (levels 1-2) within a session.
- **Healthy range:** Suggestive-to-open ratio < 0.2 (fewer than 1 suggestive prompt per 5 open prompts).
- **Risk threshold:** Ratio > 0.5 flags contamination risk.
- **Implementation plan:**
  1. Classify each AI question/prompt using the six-level hierarchy.
  2. Compute the suggestive-to-open ratio per session.
  3. Track ratio trajectory across sessions (escalating contamination).
- **Status:** Roadmapped

### 3.3 Integration with Sentinel-AI

- **Plan:** NICHD contamination scores feed into sentinel-ai's Autonomy Preservation and Epistemic Influence dimensions as a temporal signal. Dormancy-detect provides the turn-by-turn trajectory; sentinel-ai aggregates into per-session risk scores.
- **Status:** Roadmapped

---

## 4. Cross-Tool Integration (Future Work)

> **Note:** Cross-tool integration is planned for a future release. The three tools currently operate independently. The items below describe the long-term vision.

### 4.1 Dormancy-Detect Findings Annotate Sentinel-AI Scores

- **Current state:** Dormancy-detect and sentinel-ai operate independently.
- **Plan:**
  1. Define a shared annotation schema (JSON) where dormancy-detect temporal findings attach to sentinel-ai per-session scores.
  2. Each dormancy-detect alert includes: pattern type, turn range, severity, cumulative score.
  3. Sentinel-ai consumes these annotations as temporal context for its static scores.
  4. Example: sentinel-ai scores Dependency Dynamics at 0.6 for a session; dormancy-detect annotates that the 0.6 reflects a boiling-frog pattern that escalated from 0.2 over 15 sessions.
- **Status:** Roadmapped

### 4.2 Temporal Context to Verifiable-Eval Certificates

- **Current state:** Verifiable-eval certificates contain point-in-time safety scores without temporal context.
- **Plan:**
  1. Extend verifiable-eval certificate schema to include a temporal_context field.
  2. Dormancy-detect populates this field with: trend direction, change points detected, time-to-threshold estimates.
  3. Certificates can then express not just "current safety score = X" but "current score = X, trending toward Y at rate Z."
- **Status:** Roadmapped

### 4.3 API for Programmatic Integration

- **Plan:**
  1. Define a REST API (or Python function interface) for dormancy-detect that accepts conversation history and returns temporal analysis.
  2. Input: list of sessions with timestamps and sentinel-ai scores.
  3. Output: temporal patterns detected, change points, trend analysis, NICHD contamination scores.
  4. Enable programmatic integration with sentinel-ai and verifiable-eval without requiring CLI usage.
- **Status:** Roadmapped

---

## Summary

| Area | Item | Priority |
|------|------|----------|
| Temporal Patterns | Priming | HIGH |
| Temporal Patterns | Boiling frog | HIGH |
| Temporal Patterns | Anchoring | MEDIUM |
| Temporal Patterns | Social proof accumulation | MEDIUM |
| Temporal Patterns | Gaslighting escalation (Stern) | HIGH |
| Empirical Calibration | Replace hand-tuned weights | HIGH |
| Empirical Calibration | Cross-validate suspicion decay | MEDIUM |
| Empirical Calibration | Establish normative baselines | MEDIUM |
| NICHD | Six-level prompt hierarchy | HIGH |
| NICHD | Suggestive-to-open ratio | HIGH |
| NICHD | Integration with sentinel-ai | MEDIUM |
| Cross-Tool | Annotate sentinel-ai scores | MEDIUM |
| Cross-Tool | Temporal context to certificates | MEDIUM |
| Cross-Tool | Programmatic API | LOW |
