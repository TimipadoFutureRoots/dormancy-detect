# Empirical Basis for Temporal Attack Detection

## The Dormancy Vulnerability

During doctoral pilot studies at Loughborough University (2025-2026), the author observed that adversarial content introduced at session 2 of a multi-session AI mentoring interaction manifested as behavioural changes at session 5, with sessions 3-4 showing no anomalous behaviour detectable by any of four independent evaluator models (Claude Sonnet 4.5, DeepSeek V3.2, Qwen3 32B, Mistral Large 3).

This finding suggests that point-in-time safety evaluation may be insufficient for detecting temporal attacks. A system that passes every individual session evaluation could still be executing a multi-session attack pattern. Note: this was observed in a single pilot study with one attacker model and four evaluator models; broader replication is needed to establish generalisability.

## Attack Mechanism

The adversarial content was introduced through the AI system's persistent memory layer (Amazon Titan embeddings stored in DynamoDB). An attacker model (Llama 3.3 70B) generated conversational content in session 2 that:

- Planted emotionally relevant information (user vulnerabilities, preferences, relationship status)
- Encoded this information in the memory system through natural-seeming conversation
- Did not reference or activate this information in sessions 3-4
- Activated the planted information in session 5 to exert influence (emotional leverage, isolation, decision steering)

## Why Existing Monitoring Fails

Current AI safety monitoring operates at three levels, all of which miss temporal attacks:

1. **Content filtering**: Checks individual outputs for toxicity. Planted information is benign in isolation.
2. **Session-level evaluation**: Reviews each conversation independently. Sessions 3-4 are genuinely safe.
3. **Red-teaming**: Tests system responses to adversarial prompts. The attack doesn't use adversarial prompts — it uses the system's own memory.

## Supporting Literature

- **Hubinger et al. (2024).** "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training." Demonstrated that models can maintain hidden behaviours through safety fine-tuning. Our finding extends this to the memory/interaction layer.
- **Rehberger, J. (2024).** "Hacking Google Bard — From Prompt Injection to Data Exfiltration." Embracethered.com. Demonstrated that persistent memory in AI assistants creates indirect prompt injection attack surfaces. Our finding extends this to temporal activation patterns.
- **Gu, X., et al. (2024).** "Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast." *ICML 2024.* Demonstrated adversarial persistence through shared memory in multi-agent LLM frameworks (the "ZombieAgent" pattern).
- **Microsoft AI Red Team. (2024).** Documented cross-session memory poisoning as a vulnerability class in production AI systems.
- **MITRE ATLAS AML.T0080.** Categorises model manipulation through training/fine-tuning data.

## Implications for dormancy-detect

This tool exists because this attack class exists. The architectural requirement is: safety evaluation must operate across session boundaries, tracking what was planted, when it was planted, and when it activated.
