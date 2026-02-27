# VPD Model Editing: Process Learnings

Notes on what works, what doesn't, and what to watch out for when doing component-level model editing with VPD decompositions. Written after a day of intensive experimentation on `s-892f140b` (2-layer Llama, SimpleStories, 7104 components).

## Component Selection

### Use output PMI, not labels, for finding ablation targets

The single biggest methodological lesson. When you want to suppress token X, search for components that *predict* X (high output PMI), not components that *respond to* X (high input PMI) or components whose label mentions X.

Ablating input-side components destroys the model's ability to *process* X, causing massive collateral damage. Ablating output-side components specifically suppresses *production* of X.

Concrete example — quote suppression:
- Old labels (input/output confused): -86% P("), +57% dialogue PPL
- Output PMI search: -89% P("), +0.5% non-dialogue PPL

That's 5.6x less collateral damage with better suppression.

### Labels are lossy — always inspect before committing

Autointerp labels compress a rich prompt (token correlations, activation examples) into ~5 words. They routinely miss the most important information. Always call `inspect_component()` or look at the full prompt before ablating.

The dual-view autointerp strategy (`dual_view`) separates input and output function in labels, which eliminates the worst failure mode. But labels still lose nuance — components with similar labels can have very different causal roles (see attn_o:82 vs attn_o:208 below).

### Start with 1 component, then add more

Dose-response is non-linear. One component often captures most of the effect. Adding more has diminishing returns with growing collateral. For male pronouns: 1 component gets -84%, 3 gets -92%, 6 gets -96% but with 3x the PPL cost.

### Syntactic > semantic for editability

Syntactic/functional features (pronouns, punctuation, quotes) decompose into dedicated components and edit cleanly. Semantic topics (nature, food) are distributed across many components and resist clean ablation. This likely reflects VPD's masking objective rewarding sharp on/off firing patterns.

## Circuit Analysis

### Component polarity is arbitrary

A rank-1 matrix `U ⊗ V^T` is the same as `(-U) ⊗ (-V)^T`. The decomposition has no privileged sign convention. Empirically, 49% of components are "aligned" with their predicted tokens (positive activation → positive logit contribution) and 51% are "anti-aligned" (negative activation × negative cosine → positive logit contribution). Both work identically.

Don't interpret the sign of a cosine or activation in isolation. What matters is the *product* of activation × write-direction alignment, and how that product changes across contexts. A component with negative cosine to "he" that has negative activations in male contexts is *boosting* "he", not suppressing it.

### Activation sign carries the computation

While the overall polarity convention is arbitrary, the *context-dependent variation* in activation sign is meaningful. Components like `attn_o:82` use activation sign as a conditional switch: negative in male context → boosts male pronouns, positive in female context → boosts female pronouns. One rank-1 matrix implementing two-way conditional behavior via signed activations.

### Three circuit architectures exist

Not all circuits are the same:

1. **Geometric/residual**: Components aligned in weight space, communicate through the residual stream within a single forward pass. The pronoun circuit (attn→MLP chain) is this type. Identified by high cosine between one component's write direction and another's read direction.

2. **Parallel**: Independent components that each contribute to the same output via separate pathways. The quote circuit is this type. Low geometric coupling, each fires on punctuation independently.

3. **Token-mediated handoff**: Components communicate across positions through the discrete token stream. The ? → " circuit is this type: ? predictors produce ?, then " predictors fire on the ? token at the next position. Identified by one group having high output PMI for a token that the other group has high input PMI for.

### Similar labels ≠ similar causal roles

`attn_o:82` and `attn_o:208` are both labeled as "male pronoun" components. But `attn_o:208` is more critical for IOI coreference (6/15 accuracy when ablated) while `attn_o:82` is a general pronoun booster (9/15 when ablated). The label doesn't capture this distinction. Always test causally.

### Geometric alignment percentiles use the empirical population

When we say two components have "100th percentile alignment," that's computed over all ~1M actual component pairs in the same two layers, not against a random baseline. The population mean cosine is ~0.02-0.05 (close to random in high-dim space), so any cosine above ~0.15 is unusual and above ~0.3 is extreme.

## Tooling

### The `EditableModel` workflow

```python
em, tok = EditableModel.from_wandb("wandb:goodfire/spd/s-892f140b")

# Search
matches = search_interpretations(harvest, interp, r"male pronoun")
pmi_hits = search_by_token_pmi(harvest, [he_id], side="output")

# Inspect
inspect_component(harvest, interp, "h.1.mlp.down_proj:798", tok)

# Edit
edit_fn = em.make_edit_fn({"h.1.mlp.down_proj:798": 0.0})
generate(edit_fn, tokens, tok)

# Measure
measure_kl(em, edit_fn, token_seqs)
measure_token_probs(em, edit_fn, token_seqs, {"male": [he_id, him_id]})

# Circuit analysis
em.component_alignment("h.1.attn.o_proj:82", "h.1.mlp.c_fc:144")
em.unembed_alignment("h.1.mlp.down_proj:798", tok)
em.get_component_activations(tokens, "h.1.attn.o_proj:82")

# Permanent edit
edited = em.without_components(["h.1.mlp.down_proj:798"])
```

### Use AppTokenizer, not raw HuggingFace

HF's `tokenizer.encode()` silently appends EOS, making the model treat every prompt as a complete document. `AppTokenizer` uses `add_special_tokens=False` and exposes `eos_token_id` as a typed property.

### Unbatched convention

All `EditableModel` methods and free functions (`generate`, `measure_kl`, `measure_token_probs`) use unbatched tensors: `[seq]` not `[1, seq]`. The batch dimension is handled internally. This eliminates the `[0]` indexing and `[...]` wrapping noise throughout notebook code.

### Verifying the base model can do the task

Before trying to ablate a behavior, always verify the base model actually exhibits it. This 2-layer model can do IOI at 97% accuracy — surprisingly capable. But don't assume; test first with concrete prompts and measure P(correct) vs P(incorrect).
