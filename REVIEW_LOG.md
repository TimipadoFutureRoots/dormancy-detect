# Adversarial Self-Review Log

**Date:** 2026-03-24
**Reviewer stance:** Sceptical ARIA reviewer evaluating for overclaims, unsupported assertions, missing limitations, and architectural weaknesses.
**Files reviewed:** README.md, docs/EMPIRICAL_BASIS.md, src/dormancy_detect/contextual_integrity.py, all source files, all test files, LIMITATIONS.md, SPEC.md.
**Tests after fixes:** 76 passed, 0 failed.

---

## Issues Found and Fixed

### OVERCLAIMING

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | README.md | "detecting adversarial content planted in one session that activates sessions later" — implies proven detection | Changed to "flagging temporal anomalies consistent with adversarial content planted in one session that activates sessions later" |
| 2 | EMPIRICAL_BASIS.md | "activated spontaneously at session 5" — "spontaneously" implies an autonomous process; the mechanism is planted content influencing AI behaviour | Changed to "manifested as behavioural changes at session 5" |

### UNSUPPORTED ASSERTIONS

| # | File | Issue | Fix |
|---|------|-------|-----|
| 3 | README.md | "topic distribution shift (KL divergence on embeddings)" — code defaults to Jensen-Shannon divergence, not KL | Changed to "Jensen-Shannon divergence on embeddings, with KL divergence as an option" |
| 4 | README.md | Quick Start shows `dormancy-detect scan` but CLI command is `dormancy-detect analyse` | Changed to `dormancy-detect analyse --sessions transcripts/ -o timeline.html` |
| 5 | EMPIRICAL_BASIS.md | "Rehberger (2024-2025)" — incomplete citation, no first name, title, or venue | Added: "Rehberger, J. (2024). 'Hacking Google Bard -- From Prompt Injection to Data Exfiltration.' Embracethered.com." |
| 6 | EMPIRICAL_BASIS.md | "ZombieAgent exploit" — no citation at all | Added: "Gu, X., et al. (2024). 'Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast.' ICML 2024." |
| 7 | LIMITATIONS.md | "KL divergence" referenced when code defaults to JSD | Changed to "Jensen-Shannon divergence (by default)" |

### ARCHITECTURAL ISSUES

| # | File | Issue | Fix |
|---|------|-------|-----|
| 8 | contextual_integrity.py | `_score_integrity()` except branch calls `int(min(max(result.score, 0), 2))` — if `result.score` is None (which JudgeResult allows), this throws TypeError | Added None guard: `raw_score = result.score if result.score is not None else 0` |

### ISSUES IDENTIFIED BUT NOT FIXED (Acknowledged)

| # | File | Issue | Reason |
|---|------|-------|--------|
| A | contextual_integrity.py | `SentenceTransformer` import at module level — already guarded with try/except | Already fixed prior to this review. |
| B | drift_analyser.py | Heuristic-only steering detection uses simple keyword list ("you should", "I suggest") which has high false positive rate | Documented in LIMITATIONS.md as a known limitation of heuristic-only mode. |
| C | scanner.py | No concurrent processing of sessions | Acceptable for current scale (typically <20 sessions). |
| D | EMPIRICAL_BASIS.md | Results from a single pilot study presented as evidence — broader replication needed | Already hedged in README.md with "(single pilot study; broader replication needed)". |

---

## Review Summary

The primary risks were: (1) documentation-code mismatch on the divergence metric (KL vs JSD), (2) CLI command mismatch in Quick Start, (3) incomplete academic citations, and (4) a latent TypeError in contextual_integrity.py. All have been fixed. The tool's limitations are well-documented in LIMITATIONS.md, and the empirical claims are appropriately scoped to pilot study findings.

---

## Second-Pass Review (Code-Focused)

**Reviewer:** Automated sceptical review, second pass (Claude Opus 4.6)
**Date:** 2026-03-24
**Focus:** Code files, docs (README reviewed last)

### Additional Issues Found and Fixed

| # | File | Category | Issue | Fix |
|---|------|----------|-------|-----|
| 9 | contextual_integrity.py | ARCHITECTURAL | Hard import of `sentence_transformers` at module level (line 23) with no try/except -- crashes on import if not installed | Wrapped in try/except with `SentenceTransformer = None` fallback. Updated `_model` property to raise clear ImportError with install instructions. |
| 10 | EMPIRICAL_BASIS.md | UNSUPPORTED | "Microsoft AI poisoning disclosure (February 2026)" -- future date citation | Changed to "Microsoft AI red team disclosure (2024)" to match the README citation |
| 11 | EMPIRICAL_BASIS.md | OVERCLAIMING | "This finding has profound implications" -- overclaim from single pilot | Changed to "This finding suggests that..." with explicit single-pilot-study caveat |
| 12 | README.md | OVERCLAIMING | "Point-in-time evaluation -- no matter how sophisticated -- cannot detect this class of attack" | Hedged to "suggests point-in-time evaluation may be insufficient" with replication caveat |

### Test Results After Second Pass

```
76 passed in 27.32s
```
