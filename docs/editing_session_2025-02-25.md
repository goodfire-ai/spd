# Editing Session Notes — 2025-02-25

Run: s-17805b61 (4-layer Llama MLP+Attn, Pile, ~39K components)

## What we tried and what worked

### Token-level ablation (worked)
Searched graph-interp output labels for specific token types, ablated, measured. Semicolons (-88%), colons (-88%), question marks (-91%), open parens (-94%), exclamation (-49%), male pronouns (-47%), contrastive "but" (-49%). All <3.7% PPL. This is the bread-and-butter of SPD editing — reliable, measurable, low cost.

### Circuit optimization (mixed)
`optimize_circuit` on "The king summoned his most trusted knight. He told him that" → " he". Found o_proj:361 as causally necessary with stochastic P=0.988. Traced the attention circuit: v_proj:717 reads person tokens → k_proj selects masculine entities → o_proj:361 outputs gender signal.

But on other prompts (soap→water, dog→tail, cat→is) stochastic performance was terrible (<0.13) despite good CI-masked (>0.97). Only the pronoun circuit held up. Takeaway: always check stochastic, and most predictions don't compress into sparse circuits on this model.

### find_components_by_examples (disappointing)
Finds shared infrastructure (bias components, general machinery), not the differentiating features. Tried contrastively (he-examples minus she-examples) — got a tiny diff of generic components, none gender-specific. The critical masculine pronoun component (o_proj:361) didn't appear because it's only needed in male contexts, not uniformly across all "he" predictions.

### Higher-level semantic ablation (mostly didn't work)
- Negative emotion: 2 components, barely visible effect
- Modal verbs: 11 components, no visible generation change
- Narrative speech verbs: 9 components, messy results
- Second-person "you": 1 component, no effect (model doesn't use "you" much on Pile)
- Lists: 9 components, lists still appeared

Root cause: this is a 4-layer Pile model. It generates degenerate/repetitive text, making qualitative comparison of long generations impossible. High-level semantic features are distributed across too many components for small ablations to have visible effects.

### Evaluative adjectives (marginal)
24 components labeled "evaluative adjective", -62% sum P over 15 evaluative words after copular verbs. But only 1.3x selectivity (copular vs non-copular) — suppresses evaluative words everywhere, not concept-selectively. And when restricted to prompts where evaluative words were actually predicted (not noise-level), effect drops to -42% ± 26%. The story felt mushy.

### Directional adverbs (the best result)
Single component h.3.mlp.down_proj:649. Found via `unembed_alignment` — write direction points at {back, home, down, off, forward, south}. Ablating it: -38% directional adverbs after movement verbs, -1.5% in non-directional contexts. 36pp gap. +1.0% PPL. Concept-selective, semantic category, single component, clean mechanism.

### "that" disambiguation (good)
3 components with input function "verbs of assertion and reporting." Suppresses "that" after formal verbs (-35%) but boosts it after informal verbs (+28%). Same token, opposite direction. The selectivity comes from the components' narrow input functions — they fire specifically in formal attribution contexts.

### Factual knowledge (didn't work)
Tried suppressing "Romeo and Juliet" association. `find_components_by_examples` found 16 generic "proper noun suffix" components. Ablating them: -60 to -98% P(Juliet) but +64% PPL — catastrophic collateral damage. The Juliet knowledge is distributed across generic name-completion machinery, not stored in dedicated components.

## Key findings

### Token-level, not concept-level (mostly)
The "but" and "he" ablations suppress the token uniformly across all contexts — contrastive and non-contrastive "but", gendered and generic "he". Non-contrastive uses are actually suppressed MORE. This is because ablation removes a rank-1 contribution everywhere, regardless of linguistic function.

### Exception: concept-selectivity from narrow input functions
The "that" and directional adverb results show concept-selectivity IS possible when the ablated component has a narrow input function. The "that" components fire specifically on "verbs of assertion" (formal register). The directional component fires specifically after movement verbs (via the c_fc:2506 fan-out). Broad input function → token-level edit. Narrow input function → concept-selective edit.

### The MLP "fan-out" — corrected understanding
Initial finding: c_fc:2506 detects movement verbs, fans out to multiple down-projections (directional, manner, degree, temporal, prepositional). But ablation testing revealed this is WRONG as a causal story:

- Ablating c_fc:2506 alone: directional **+2%** (no effect!), manner **-63%**, degree **-40%**
- Ablating ALL 7 non-bias upstream c_fc components: directional **+10%** (still no effect!)
- Ablating just down_proj:649 alone: directional **-38%**

The directional signal doesn't flow through any identifiable c_fc component. It's distributed across the full c_fc layer (3072 components), so removing a few doesn't matter. The concept-selectivity of down_proj:649 comes entirely from its **write direction** — it reads a broad, distributed MLP hidden state and projects onto the directional adverb subspace in vocab space.

Corrected framing: down_proj:649 is a **readout direction**, not a narrow channel in a pipeline. The "fan-out" structure (graph-interp edges from c_fc:2506 to multiple down_proj) describes attribution flow but not causal necessity. Manner and degree branches DO depend on c_fc:2506, but the directional branch doesn't.

Lesson: graph-interp edges show attribution (correlation in gradient flow), not causal necessity. Always validate with ablation.

### Bias components
14/39K components have mean CI > 0.5 and fire on >60% of tokens. They're structural biases necessary for everything. Show up in every circuit. Filter them out when searching for editing targets.

### Measurement matters
- Single-position P(token) can overstate the effect. Colons: -88% at single position, -18% in generation.
- Generation-level counts are more honest. Pronouns: -47% single position but -73% in generation (compounds). Question marks: -100% in generation (zero produced in 600 tokens).
- PPL on 15 hand-picked texts vs 25K training tokens: similar but the latter is more defensible.
- Random N-component ablation achieves ~0% target suppression at similar PPL cost. The edits are targeted, not just small enough to be harmless. But this is the expected outcome (N/39K is tiny), not an impressive finding.
- Concentrated damage: targeted edits have 4-13x higher KL on target domain vs unrelated text. Random has ~1x. Targeted doesn't spare unrelated text — it concentrates extra damage in the target domain.

## Tool effectiveness

### Graph-interp label search
Primary discovery method for 5/7 token-level edits and the directional adverb result. Fast, broad coverage. Most effective when searching for specific output patterns. The separate input/output labels are crucial — output tells you what the component produces, input tells you when it fires.

### unembed_alignment
How we found the directional adverb component — the write direction formed a tight semantic cluster in vocab space. Also useful for understanding the MLP fan-out (applying it to sibling components). Underused tool — should be a standard part of the exploration workflow.

### Graph-interp edges
How we traced the MLP fan-out: upstream from down_proj:649 to c_fc:2506, then downstream from c_fc:2506 to siblings. Also used for the pronoun circuit analysis. These are dataset-level attributions (aggregated), not prompt-specific.

### optimize_circuit
Good for prompt-specific causal analysis when it works (pronoun circuit, stoch P=0.988). But most behaviors don't compress into sparse circuits on this model (stoch P < 0.13). Expensive (~15s per prompt). Use for validation/mechanistic understanding, not for search.

### find_components_by_examples
Disappointing for editing purposes. Finds shared infrastructure and bias components, not the differentiating features that matter for targeted editing. The contrastive approach (run on A, run on B, diff) produced noise. Might work better with more examples or on a better-decomposed model.

### inspect_component
Underused. Should have looked at activation examples more systematically earlier. The labels are lossy summaries — the actual examples show what the component really does.

### PMI search
Good for rare/specific tokens (pronouns, question marks). Bad for common tokens (periods, commas, "the") where PMI is noisy. Graph-interp labels are better for common tokens.

## What I'd do differently next time

1. Start with `unembed_alignment` on random components to find ones with coherent semantic write directions. This found our best result.
2. Use graph-interp edges to trace fan-out patterns in MLPs. The up→down decomposition is a systematic structure, not a one-off.
3. Don't waste time on generation-level evaluation for this model. It generates degenerate text. Stick to P(token) measurements and be honest about their limitations.
4. For concept-selectivity tests, the key is finding components with NARROW input functions (check the graph-interp input label). Broad input → token-level edit. Narrow input → concept-selective.
5. Skip `find_components_by_examples` for contrastive features. It finds the wrong thing.
6. Always measure on prompts where the target token is actually in the baseline top-5. Measuring suppression of noise-level probabilities is meaningless.

## Open questions

- Is the MLP fan-out pattern common? How many MLPs decompose cleanly into semantically distinct up→down pathways?
- Can we find concept-selective attention edits (not just MLP)? The pronoun component is in attention but we didn't test its concept-selectivity properly.
- Would a larger/better model show cleaner high-level edits? The 4-layer Pile model may just be too small for semantic editing.
- Can permanent weight editing (rank-1 subtraction) reproduce all these results? We only validated it for the pronoun case.
