# SPD Editing: Conceptual Notes

Working notes from editing experiments on s-17805b61 (4-layer Llama MLP+Attn, Pile, ~39K components) and comparison with s-892f140b (2-layer, SimpleStories, 7K components).

## Component activations have no privileged sign

A rank-1 component is a matrix V[:, c] @ U[c, :]. Since V ⊗ U^T = (-V) ⊗ (-U)^T, the decomposition has an arbitrary sign convention. Empirically ~50/50 "aligned" vs "anti-aligned" with their predicted tokens.

This means:
- A component's activation being negative at some position doesn't mean it's "inactive" or "suppressing" — it depends on the write direction sign.
- The product (activation × write-direction alignment) is what determines the logit contribution. Both signs matter together, neither alone.
- Causal Importance (CI) measures *necessity* — how much masking the component degrades performance — not the direction of its contribution. High CI means "this component matters here," not "this component activates positively here."

When we say a component "fires on" a token in harvest data, we really mean "has high CI on that token." The actual computation depends on activation sign × write direction.

## Finding editing targets: PMI vs circuit optimization vs graph-interp

Three methods for finding components to ablate, with different strengths:

**PMI search** (`search_by_token_pmi`): Finds components whose output token statistics correlate with the target. Fast, covers all components. But correlation ≠ causation — on s-17805b61, the top 13 pronoun PMI hits had near-zero editing effect. PMI also picks up structural components (newline predictors that happen to co-occur with "he" because "he" often starts lines).

**Circuit optimization** (`optimize_circuit`): Finds the minimal causal set for a *specific prediction at a specific position*. Expensive (~15 seconds per prompt), but gives ground-truth causal structure plus edges showing how components interact. On the "king" prompt, it found o_proj:361 as causally necessary with ci=1.0. The edges revealed the full attention circuit: v_proj reads person tokens, k_proj selects masculine entities, o_proj outputs the gender signal.

**Graph-interp labels**: Context-aware labels that distinguish input function (what triggers it) from output function (what it produces). The output label for o_proj:361 specifically says "third-person masculine" — distinguishing it from 170+ generic "pronoun" components. Graph-interp also provides attributed edges from dataset attributions, showing the typical (not prompt-specific) circuit structure.

**`find_components_by_examples`**: Runs circuit optimization across multiple prompts and finds components that appear consistently in sparse circuits (≥ min_frequency). Finds *shared infrastructure* — components needed for the general mechanism (e.g. pronoun production). Does NOT find differentiating components (e.g. the specifically *masculine* component). For contrastive features, PMI + graph-interp specificity is better.

Example: running on 6 male-pronoun prompts found 19 components — mostly bias components and generic formal-text machinery. The critical masculine component (o_proj:361) did not appear because it's only needed in male contexts, not uniformly across all "he" predictions.

Running contrastively (he-examples minus she-examples) yielded a small diff (4 he-only, 2 she-only) but none were specifically gendered — they were generic prose/reporting components that happened to differ slightly.

**Best workflow**: PMI search or graph-interp label search for initial candidates → filter by firing density (remove bias) → circuit optimization on representative prompts for causal confirmation → ablate and measure.

## What edits cleanly vs what doesn't

On the Pile model (s-17805b61):

| Feature | Components | Suppression | PPL cost | Discovery method |
|---------|-----------|-------------|----------|-----------------|
| Male pronouns (" he") | 1 (0.003%) | -47% | +1.7% | PMI + graph-interp |
| Question marks (?) | 4 (0.01%) | -86 to -94% | +2.1% | PMI search |
| Semicolons (;) | 8 (0.02%) | -81 to -99% | +1.1% | Graph-interp labels |
| Exclamation marks (!) | 1 (0.003%) | -50 to -68% | +1.8% | Graph-interp labels |
| Colons (:) | 11 (0.03%) | -89 to -99% | -0.1% | Graph-interp labels |
| Opening parens (() | 27 (0.07%) | -93.6% ± 6.9% | +2.2% | Graph-interp labels |
| Contrastive conj ("but") | 3 (0.008%) | -49.4% ± 10.6% | +1.5% | Graph-interp labels |
| Negation (" not") | 2 (0.005%) | -24 to -50% | +1.9% | Graph-interp labels |
| Quotation marks (") | 16 (0.04%) | -30 to -54% | — | PMI search |
| Past tense verbs | 3 (0.008%) | Minimal | — | Graph-interp labels |

All PPL costs < 3.7% on 25K tokens of training data. Results validated on held-out prompts (within error bars).

**Honest tiering by generation-level confirmation:**
- **Strong**: question marks (-94% pos, -100% gen), open parens (-94% pos, -93% gen)
- **Good**: male pronouns (-47% pos, -73% gen — compounds!), contrastive "but" (-49% pos, -64% gen, "but"→"and" 87%)
- **Moderate**: semicolons (-88% pos, -63% gen)
- **Overclaimed**: colons (-88% pos, -18% gen — model recovers), exclamation (-49% pos but baseline rate too low to be meaningful)

The pattern: syntactic/punctuation features with sharp on/off firing patterns decompose into dedicated components and edit cleanly (>80% suppression). Functional features like negation are partially editable (-50%). Semantic/distributed features (tense, quotes on Pile) resist clean ablation.

Graph-interp labels were the primary discovery method for 4 of the 6 cleanest edits (semicolons, exclamation, colons, negation). The workflow: search graph-interp output labels for the target pattern → filter to late layers → ablate → measure. This is faster and more targeted than PMI search for punctuation features because these tokens are too common for PMI to be discriminative.

Notable generation effects:
- Semicolons: `int x = 5;` → `int x = 5, y = 0...` (substitutes commas)
- Semicolons: `return result;` → `return result.get(0)` (substitutes method call)
- Colons: `The answer is simple:` → `The answer is simple.` (substitutes period)

The contrastive conjunction result is notable: ablating just 3 components causes "but" to be replaced by "and" in 87% of cases. This is a discourse-level behavioral shift — the model loses the ability to express contrast and defaults to coordination. In 40-token generations, contrastive words (but/however/yet/although) drop -64% while "and" usage doubles.

**Generation-level metrics strengthen single-position results.** Pronouns go from -47% (single position) to -73% (in generation) — the effect compounds over tokens. Question marks go to -100% in generation (zero produced in 600 tokens). But colons only show -18% in generation despite -88% at single position — the model finds alternative pathways after the first token.

Compared to SimpleStories (s-892f140b, 2-layer, 7K components): quotes edited cleanly there (-89% with 3 components) but not on Pile. The Pile model has more redundancy for common tokens.

## Measurement: what's principled and what isn't

Three measurement levels, all necessary:
1. **P(token) at single position**: fast, easy to interpret, but only measures one token at one position. Filtered to prompts where baseline P > threshold.
2. **Token counts in generation**: captures compounding effects over multiple tokens. But generation is stochastic — need enough prompts for stable counts.
3. **PPL on training data**: 25K tokens from Pile (not hand-picked). Global damage measure.

**Random baseline comparison**: ablating N random (non-bias) components achieves ~0% target suppression at similar PPL cost. This confirms the edits are targeted, not just small enough to be harmless. But it's the *expected* outcome — N/39K is tiny, of course global PPL is similar regardless of which N you pick.

**Concentrated damage**: targeted edits have 4-13x higher KL on the target domain (code/narrative) vs unrelated text. Random ablation has ~1x ratio (uniform damage). Targeted ablation concentrates its effect in the relevant domain but doesn't *spare* unrelated text — unrelated domain KL is roughly the same as random.

**What we can't claim**: that the edit is "free" or "zero cost" — it costs the same as random ablation, which is small but nonzero. The claim is: same damage budget, massively concentrated effect.

## Token-level, not concept-level surgery

Component ablation suppresses *tokens*, not *concepts*. Tested by checking whether edits distinguish between linguistic contexts:

**Contrastive conjunctions**: ablating 3 "contrastive conjunction" components suppresses " but" in contrastive contexts (-49%) but ALSO in non-contrastive contexts like "nothing but" (-97%) and "no choice but" (-87%). The non-contrastive uses are actually suppressed *more*. The components predict the token " but" regardless of its linguistic function.

**Male pronouns**: ablating o_proj:361 suppresses " he" in male-subject contexts (-44%) but ALSO in gender-ambiguous contexts like "The doctor told the nurse that" (-63%) and "The cat looked at the mouse and" (-77%). Again, more suppression in secondary contexts.

**Why secondary contexts are suppressed more**: the components have lower CI in secondary contexts (their contribution is a smaller fraction of the total), so removing them causes a larger *relative* drop. In primary contexts, other components partially compensate.

**Implication**: component ablation removes a rank-1 contribution to token prediction across all contexts. It doesn't distinguish linguistic function. The "discourse-level" and "gender-specific" framings are overclaims. The honest framing is: "ablating N components reduces P(token) by X% broadly, with <Y% PPL cost."

However, concept-selective editing IS possible when components have specific enough input functions. The "that" complementizer experiment demonstrates this:

**"that" disambiguation (3 components)**:
- Components: h.3.mlp.c_fc:737, :1150, :1703 — all have input labels mentioning "verbs of assertion/reporting/attribution"
- Formal attribution verbs (said, showed, argued, claimed): P(" that") **-35.1%** ± 20.6%
- Informal mental verbs (think, know, thought, realized): P(" that") **+27.8%** ± 60.1%
- Same token, opposite direction by context. PPL cost: +1.2%.

The key difference from the "but" case: the "but" components had broad input functions ("punctuation at clause boundaries" — fires for all uses of "but"), while these "that" components have narrow input functions ("verbs of assertion and reporting" — fires only in formal attribution contexts).

**Principle**: concept-selectivity of an edit depends on the specificity of the ablated components' input functions. Narrow input → concept-selective. Broad input → token-level. Graph-interp input labels predict which case you'll get.

## Word-class editing: evaluative adjectives

Beyond single-token and concept-selective editing, SPD components can encode *semantic categories*. Ablating 24 components labeled "evaluative adjective" by graph-interp suppresses an entire word class — good, great, bad, terrible, wonderful, excellent, awful, amazing, horrible, fantastic, poor, nice, beautiful, ugly, pleasant — by -62% after copular verbs, -46% in other contexts. Random 24-component ablation: -8% (noise level). PPL: +2.5%.

The top-5 distribution shifts are the most visually compelling part: "The food tasted" loses "good" and "great" from top predictions, replaced by functional words ("was", "in", "at"). "The experience was" shifts from evaluative ("good", "great") to factual/descriptive ("recorded", "first", "then").

Selectivity is modest (1.3x copular vs non-copular) — the components predict evaluative adjectives broadly, not just after linking verbs. But the edit operates on a *semantic category*, not a single token. This is the most abstract level of editing we've demonstrated.

**Abstraction hierarchy of edits found:**
1. **Token-level**: suppress ";", "?", "(" — one token, high suppression (>87%)
2. **Concept-selective**: suppress "that" after formal verbs but not informal — same token, context-dependent (-35% formal, +28% informal)
3. **Semantic category + concept-selective**: suppress directional adverbs (back, home, down, off, forward) after movement verbs — multiple tokens, 36pp gap between directional vs non-directional contexts, single component, +1.0% PPL
4. **Word-class**: suppress evaluative adjectives — 15+ distinct tokens, but only 1.3x context selectivity

## MLP fan-out: how SPD decomposes verb-complement prediction

The directional adverb component (`h.3.mlp.down_proj:649`) is part of a decomposed MLP computation. One up-projection (`h.3.mlp.c_fc:2506`, density 6.6%) detects movement/posture verbs (stumbled, crept, marched, knelt, glanced, frowned) and feeds into multiple down-projections, each producing a different type of adverbial completion:

- `down_proj:649` → directional (back, home, down, off, forward)
- `down_proj:1160` → manner (strongly, beautifully, perfectly, intensely)
- `down_proj:1121` → degree/scope (mainly, primarily, mostly, predominantly)
- `down_proj:3402` → temporal/numeric (early, five, six, seven)
- `down_proj:516` → prepositional (against, intra-)

This fan-out structure explains the concept-selectivity: `:649` is one branch of a parallel computation. Ablating it removes the directional output while leaving manner, degree, and temporal outputs intact. The selectivity comes from SPD decomposing the MLP into semantically distinct pathways.

This is arguably the most important mechanistic finding: SPD doesn't just find individual features, it reveals how MLPs decompose multi-output computations into parallel sub-functions via the up-projection → fan-out → down-projection architecture.

Found via `unembed_alignment` — inspecting the write direction in vocab space, which forms a tight cluster of directional adverbs.

## Ablation vs boosting asymmetry

Ablation (mask=0) reliably suppresses features. Boosting (mask>1) doesn't reliably amplify them. On s-17805b61, boosting question mark components 2-3x had no visible effect on generation. Same pattern observed on s-892f140b.

Likely reason: ablation is a clean operation (remove a rank-1 contribution), while boosting amplifies the component's contribution in a way that interacts unpredictably with the rest of the network. The model wasn't trained to be robust to amplified components.

## Circuit structure: the masculine pronoun circuit

On the prompt "The king summoned his most trusted knight. He told him that" → predicts " he":

```
v_proj:717 reads person tokens    k_proj:53 selects "He", "king"     k_proj:151 selects "He", "knight"
   ("knight", "him", "his")              │                                    │
            │                            │                                    │
            └────────────────────────────┼────────────────────────────────────┘
                                         │
                                   o_proj:361 @ "that"
                                   act = -3.49, write cos(" he") = -0.531
                                   net: +1.85 boost to P(" he")
                                         │
                                   downstream MLP
                                   c_fc:2660 ("third-person pronouns")
                                   c_fc:2907 ("personal pronouns")
```

v_proj:717's graph-interp input label is "Third-person pronouns and person-referencing entities" — exactly what the circuit shows it reading. The k_proj components attend to masculine-context tokens. o_proj:361 combines these signals into a gendered output.

The activation is graded by context: -3.64 at "boy" context, -1.82 at "girl" context, -0.51 at "food" context. So the single component implements a continuous gender detector, not a binary switch.

### CI-masked vs stochastic performance gap

Circuit optimization metrics report both CI-masked and stochastic-masked label probabilities. CI-masked is consistently optimistic:

| Prompt → target | L0 | CI-masked P | Stoch P |
|----------------|-----|------------|---------|
| King → " he" | 13 | 0.9995 | 0.9877 |
| Soap → " water" | 22 | 0.993 | 0.12 |
| Dog → " tail" | 31 | 0.979 | 0.02 |
| Cat → " is" | 12 | 0.996 | 0.13 |

Only the pronoun circuit held up under stochastic sampling. This suggests masculine pronouns are particularly well-decomposed by SPD (sharp, binary, concentrated in attention), while lexical associations (soap→water) and syntactic patterns (is/are agreement) may not compress cleanly. Always check stochastic performance, not just CI-masked.

## Bias components: high mean CI ≈ always on

A small number of components (~14 out of 39K on s-17805b61, 0.04%) have mean CI > 0.5 and firing density > 0.6. These fire on essentially every token — they're structural biases, not selective features.

Examples from s-17805b61:
- `h.2.attn.o_proj:443`: mean CI=0.957, density=1.00 — "General bias for formal and technical prose"
- `h.3.attn.o_proj:524`: mean CI=0.947, density=0.99 — "General background bias across all token contexts"
- `h.1.attn.k_proj:343`: mean CI=0.926, density=1.00 — "Broad bias for technical, multilingual, and symbolic text"

These show up in circuit optimization results (o_proj:524 had the largest activation magnitude in the pronoun circuit) because they're causally necessary for *everything* — they're part of the model's baseline computation. But ablating them would damage all predictions, not just the target behavior.

**Practical guidance**: filter out components with mean CI > 0.5 (or firing density > 0.5) when searching for editing targets. They're not useful for targeted interventions. The harvest data has `mean_activations["causal_importance"]` and `firing_density` per component.

Note: some k_proj bias components (like k_proj:151, k_proj:184) appeared in the pronoun circuit's edges. They participate in attention routing broadly, so their presence in a circuit doesn't mean they're pronoun-specific — they're the structural scaffolding that selective components (like o_proj:361) operate within.

## Graph-interp: when it helps and when it doesn't

**Helps**:
- Distinguishing specific from generic: o_proj:361 was labeled "masculine" while 13 other pronoun components were labeled generically. This alone would have saved the entire dose-response sweep.
- Separate input/output labels: input="action verbs and structural punctuation", output="masculine pronouns". This tells you the component fires on narrative transitions and produces gendered pronouns — more informative than a single merged label.
- Circuit context from edges: shows upstream (v_proj:717, k_proj components) and downstream (MLP pronoun producers) connections.

**Doesn't help**:
- Broad searches: "pronoun" matches 170 components. Graph-interp labels are descriptive, not ranked. You still need PMI or circuit optimization to narrow candidates.
- The labels can be noisy: some graph-interp output labels for clearly structural components mention "pronouns" incidentally. PMI provides a quantitative filter that labels can't.

**Conclusion**: Graph-interp is most valuable *after* you have candidates (from PMI or circuit optimization) and want to understand *what they do* and *how they fit together*. It's a comprehension tool more than a search tool.
