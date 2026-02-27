# Model Editing with SPD: Observations & Improvement Directions

From hands-on experiments with `goodfire/spd/s-892f140b` (2-layer Llama MLP-only on SimpleStories, 7104 components).

## 1. Input vs Output Function Confusion

**The single biggest source of error in component selection.**

Autointerp labels like "quotation marks and speech verbs" don't distinguish between:
- **Input function**: the component *fires when it sees* quotation marks
- **Output function**: the component *predicts* quotation marks

These are very different for editing. Ablating input-function components destroys the model's ability to *process* a feature (massive collateral damage). Ablating output-function components specifically suppresses *production* of that feature.

**Quantified example — quote suppression:**

| Approach | How selected | P(") drop | Dialogue PPL | Non-dialogue PPL | Components |
|----------|-------------|-----------|-------------|-----------------|------------|
| Label search | Regex on labels | -86% | +57.5% | +2.9% | 6 |
| Output PMI | `search_by_token_pmi(side="output")` | -89% | +10.2% | +0.5% | 3 |

The output-PMI approach achieves better suppression with **5.6x less collateral damage** using half the components.

**Implications for autointerp prompting:**
- Labels should explicitly distinguish input vs output function: "fires on X" vs "predicts X"
- Or better: generate separate input-function and output-function labels
- The current prompt includes both input and output PMI data, but the LLM often conflates them into a single label
- Consider structured output with separate `input_label` and `output_label` fields

**Implications for research agents:**
- Never select components for ablation based on labels alone
- Always verify with `inspect_component()` or check output PMI directly
- When suppressing token X, search with `search_by_token_pmi(token_ids, side="output")`
- When finding components that respond to token X (e.g. for understanding circuits), use `side="input"`


## 2. Autointerp Labels are Lossy — Use the Full Prompt

The autointerp label is a ~5-word compression of a rich prompt containing:
- Input token correlations (recall, precision, PMI)
- Output token correlations (precision, PMI)
- 10+ activation examples with highlighted firing positions

The label often misses the most important information. Examples:
- `h.1.mlp.c_fc:802` labeled "moral lessons and [EOS]" — label misses that its output PMI is dominated by abstract nouns (unity, bonds, friendship, acceptance, overcome)
- `h.1.mlp.c_fc:1010` labeled "sentence-ending punctuation and commas" — label misses that its highest output PMI is for `"` (3.39), making it a dialogue-boundary predictor

**Systematic audit** (30 random components): ~60% of labels describe input function only, ignoring output. The reasoning field is usually much better — it mentions output patterns — but the label compression step loses this.

**Root causes in the prompt structure:**
1. **Input data comes first** (recall, precision, PMI) — forms the LLM's first impression before it sees output data
2. **Task says "detects"** — "what this component detects" naturally reads as input function. "Detects" implies sensing, not producing
3. **Activation examples show input context only** — the `<<delimiters>>` highlight tokens where CI is high (input positions), not what the model predicts at those positions. 30 rich text examples all reinforce input-side understanding
4. **Output data is less salient** — just token lists with numbers, easy to skim vs 30 highlighted examples
5. **"2-5 word" constraint** forces lossy compression — reasoning captures both sides but the label can only fit one concept, defaults to the more salient input pattern

**Concrete failure examples:**
- `h.1.mlp.c_fc:564` "prepositions 'of', 'about', 'from'" — fires on prepositions, but PREDICTS their completions ("course", "afar", "inspiration"). Reasoning says this; label drops it
- `h.1.mlp.down_proj:532` "closing quotation mark" — fires on `"` but predicts names/reactions AFTER quotes (startled, lily). Ablating this won't suppress quote production
- `h.1.mlp.down_proj:666` "third-person pronouns and punctuation" — fires on they/punctuation but predicts action verbs (reached, spotted, met). Label completely misses output function
- `h.0.attn.o_proj:153` "quotation marks and speech verbs" — fires on speech verbs, but predicts interjections (wow, oh, hey). Reasoning mentions this; label doesn't

**Suggested prompt fixes:**
1. Ask for TWO labels: `input_label` (what it fires on) and `output_label` (what it predicts)
2. Or reorder: put output data FIRST since it's more actionable for editing
3. Change "detects" to "does" in the task instruction
4. Add output token examples: show actual predicted text at firing positions, not just PMI numbers
5. Increase label length to 1 sentence: "fires on [X], predicts [Y]"

**Implemented fix: `output_centric` strategy** (`spd/autointerp/strategies/output_centric.py`):
- Output PMI/precision data presented FIRST
- Dual-view activation examples: (a) fires on, (b) says (shifted right by 1)
- Canonical label forms: "says X", "predicts X after Y"
- Task asks "what does it predict" not "what does it detect"
- Tested on 25 components: 23/25 (92%) correctly describe output function vs 1/25 (4%) with old strategy. Zero regressions.

Use via config: `{"type": "output_centric"}` in `template_strategy`.

**Implications for research agents:**
- Always call `inspect_component()` before committing to an ablation
- Look at the full autointerp prompt via `interp.get_interpretation(key).prompt` for edge cases
- The `reasoning` field on `InterpretationResult` is more informative than `label` — use it


## 3. Editing Difficulty Varies by Feature Type

Syntactic/functional features decompose cleanly into dedicated components. Semantic topics are distributed broadly.

| Feature Type | Suppression | PPL Cost | Components | Why |
|-------------|-------------|----------|------------|-----|
| Pronouns (he/him/his) | -96% | +6% | 6 (0.08%) | Sharp functional distinction |
| Quotes (`"`) | -89% | +0.5% | 3 (0.04%) | Punctuation is discrete/sparse |
| Moral lessons | qualitative | +8% | 5 (0.07%) | Somewhat concentrated |
| Nature words | -25% | +3.6% | 5 (0.07%) | Broadly distributed |

**Why**: SPD's stochastic masking objective rewards sharp on/off firing patterns. Features that are binary (present/absent) like pronouns and punctuation get dedicated components. Semantic topics that shade gradually across many contexts get distributed representations.

**Implications for research:**
- Component editing is most powerful for syntactic/functional features
- For semantic steering, boosting may work better than ablation (amplify existing components rather than trying to remove distributed representations)
- The editing difficulty spectrum is itself an interesting research finding about what SPD decomposes cleanly


## 4. Dose-Response and CI-Guided Ablation

**Dose-response is non-linear.** For male pronoun ablation:
- 1 component: -84% suppression (the sweet spot for minimal side effects)
- 3 components: -95% suppression, but female pronouns start increasing (+35%)
- 6 components: -96% suppression, diminishing returns with growing collateral

**CI-guided ablation reduces collateral damage by ~36%** while retaining ~89% of the targeted effect. It works by only zeroing a component at positions where its CI exceeds a threshold, leaving it active elsewhere.

| Method | Male P(he) drop | PPL increase |
|--------|----------------|-------------|
| Blanket (6 comps) | -96% | +13.6% |
| CI-guided (threshold=0.1) | -85% | +8.3% |

**Implications for tooling:**
- Always try single-component ablation first before adding more
- `make_edit_fn(model, edits, ci_threshold=0.1)` should be the default recommendation for surgical edits
- Could build an automatic dose-response sweep utility


## 5. Component Boosting is Fragile

Amplifying components (mask > 1) can steer generation but is less reliable than ablation:
- 3x boost on "magical wish-related tokens": works, stories gain magical elements
- 5x boost on "treats and cold desserts": degenerates into repetition
- 5x boost on "animal and character names": degenerates

**Implications:**
- Boosting needs careful tuning per-component. There's no universal safe multiplier.
- Ablation is more robust because zeroing is a clean operation; amplification interacts unpredictably with the rest of the model
- A future improvement could be learned boost factors (optimize the multiplier to maximize some target metric while constraining KL)


## 6. Data Accessibility Friction

During the experiments, several data access patterns required workaround code:

**6a. No output-PMI search.** The harvest DB stores per-component output PMI, but there's no query API for "find components whose output PMI for token X is high." Had to write raw SQL. Now in `spd.editing.search_by_token_pmi()` but could also live on `HarvestRepo`.

**6b. No interpretation search.** `InterpRepo` has `get_all_interpretations()` and `get_interpretation(key)` but no search/filter. Regex search over labels is very common in exploratory work. Now in `spd.editing.search_interpretations()`.

**6c. `get_all_components()` is slow for large decompositions.** 7K components with full activation examples takes many seconds to deserialize. For PMI-only queries, a lighter SQL query is much faster. Consider exposing `get_all_pmi()` or similar bulk-but-lightweight accessors.

**6d. Legacy data layout.** `HarvestRepo.open_most_recent()` and `InterpRepo.open()` only find `h-*/a-*` subrun directories. Older runs have flat layouts. Migration script exists at `scripts/migrate_harvest_data.py` but ideally old data would just be migrated so this doesn't bite future users.


## 7. Suggested Improvements

### Autointerp Prompting
1. **Separate input/output labels**: Ask the LLM for distinct labels — "Input: fires on sentence-ending punctuation inside dialogue" / "Output: predicts opening quotation mark"
2. **Longer descriptions**: The 3-5 word label loses too much. A 1-sentence description alongside the label would help
3. **Output-weighted labeling**: Since output function is more actionable for editing, prompt the LLM to weight output patterns more heavily

### Interfaces / Data Access
4. **PMI search on HarvestRepo**: `harvest.search_by_pmi(token_ids, side="output")` would be natural
5. **Label search on InterpRepo**: `interp.search(pattern)` with regex support
6. **Lightweight bulk accessors**: `harvest.get_all_pmi()` returning just `{key: (input_pmi, output_pmi)}` without deserializing activation examples

### Research Agent Prompts
7. **Standard component selection workflow**: (a) search by output PMI for target tokens, (b) filter by firing density > 1%, (c) inspect top candidates with `inspect_component()`, (d) verify input vs output function, (e) test single-component ablation first
8. **Standard evaluation workflow**: (a) measure KL + PPL on general text, (b) measure token probability shifts for target tokens, (c) measure KL on "target domain" vs "non-target domain" text for specificity ratio, (d) generate qualitative examples


## 8. Permanent Weight Editing Works

SPD component ablation can be done as a **permanent modification to the target model's weight matrices**, not just runtime masking. Since each component is a rank-1 matrix (V[:, c] @ U[c, :]), removing a component means subtracting its rank-1 contribution from the weight matrix.

**Validation**: Comparing mask-based ablation (ComponentModel + mask_infos) to permanent weight editing (subtract rank-1 from nn.Linear weights) on male pronoun suppression:

| Metric | Mask-based | Weight edit |
|--------|-----------|------------|
| KL from baseline | 0.1248 | 0.1218 |
| P(he/him/his) change | -93.8% | -93.7% |
| Agreement KL (between them) | 0.005 | — |

The two approaches agree to within KL=0.005 (25x smaller than the edit effect). Generation outputs are token-for-token identical.

**Why this matters**: The weight-edited model is a standard transformer. No CI function, no mask computation, no SPD machinery at inference. You can export the edited weights and deploy them anywhere. This validates the VPD paper's claim that parameter decomposition "permits direct editing of the model parameters in interpretable ways."

**Implementation**: `spd.editing.make_weight_edited_model(model, ablate_keys)` returns a deep copy with components subtracted from the target model's weights.

**Caveat**: Weight editing is strictly ablation (removal). Boosting (mask > 1) doesn't have a clean weight-space analogue because it would require the CI function to determine where to amplify. CI-guided ablation also requires runtime CI computation. Only uniform ablation (mask = 0 everywhere) maps cleanly to weight subtraction.
