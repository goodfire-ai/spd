# Bio Investigation Append-Only Log

Purpose: shared running log for model-behavior investigations in `spd` by Oli, Codex, and Claude 4.6 Opus.

## Append-Only Rules
- Do not edit or delete existing entries.
- Add new entries at the end of this file only.
- Use UTC timestamps in ISO-8601 format.
- Keep entries factual and include enough detail to reproduce work.

## Entry Template

```md
## [YYYY-MM-DDTHH:MM:SSZ] Author
Context:
Actions:
Observations:
Artifacts:
Next:
```

---

## [2026-02-26T00:00:00Z] Codex
Context: Initialized shared append-only collaboration log for bio-style model behavior investigations.
Actions: Created this file and defined append-only usage rules + entry template.
Observations: `logs/` is a suitable location for durable project logs.
Artifacts: `logs/bio_investigation_append_only_log.md`
Next: Start appending investigation entries as work begins.

## [2026-02-26T11:05:55Z] Codex
Context: Corrective append-only entry; previous entry had shell-escaped markdown artifacts in field values.
Actions: Added `scripts/parse_transformer_circuits_post.py`; parsed the biology URL into markdown; downloaded local image assets.
Observations: Distill-style `d-article`/`d-appendix` parsing works; forcing UTF-8 decode and stable asset naming improves output quality and rerun behavior.
Artifacts: `scripts/parse_transformer_circuits_post.py`, `papers/biology_source/biology.md`, `papers/biology_source/assets/`
Next: Reuse script for other Transformer Circuits posts or extend asset handling beyond `img` tags if needed.
## [2026-02-26T11:36:33Z] Codex
Context: Consolidated framework discussion after reviewing Anthropic biology post and adapting to `spd` methodology.
Actions: Logged two levels of framing for future investigations.
Observations:
- CLT-style framing (Anthropic-like):
  - Goal: reverse-engineer internal computation, not just behavior.
  - Mechanisms as causal internal pathways for prompt -> token behavior.
  - Loop: interpretable approximation -> trace -> hypothesis -> intervention validation.
  - Claims treated as local existence proofs unless generalized.
  - Explicit uncertainty accounting and limits required.
- `spd`-adapted framing (preferred for this project):
  - Primary object: literal parameter chunks in the original model.
  - Mechanism for a prompt: minimal necessary parameter subgraph (chunks + dependencies) preserving target behavior.
  - Identification: stochastic causal importance masking under sparsity constraints.
  - Faithfulness: interventions/masking on real parameters support mechanistic-faithfulness claims (subject to optimization/sampling error).
  - Evidence standard: necessity + sufficiency + minimality.
  - Uncertainty: residual behavior, seed instability, and alternative near-minimal supports.
  - Generalization: test transfer across prompt families and derive reusable motif taxonomies.
  - Advantage vs CLT-only approach: less surrogate mismatch, stronger causal grounding on what the base model actually used.
Artifacts: `logs/bio_investigation_append_only_log.md`
Next: Use this as the standing conceptual framework for upcoming `spd` bio-style experiments.
