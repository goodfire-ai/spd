# Attention Head Characterization: s-275c8f21

Model: 4-layer, 6-head LlamaSimpleMLP (d_model=768, head_dim=128), pretrained model t-32d1bb3b.

All scripts live in `spd/scripts/detect_*/` and output to `spd/scripts/detect_*/out/s-275c8f21/`.

## Analyses

### Previous-Token Heads

**Method**: On real text (eval split, 100 batches of 32), extract the offset-1 diagonal of each head's attention matrix — i.e., `attn[i, i-1]` — and average across positions and batches.

**Results**:
| Head | Score |
|------|-------|
| L1H1 | 0.604 |
| L0H5 | 0.308 |

All other heads score below 0.1. L1H1 is a clear previous-token head, spending over 60% of its attention on the immediately preceding token. L0H5 is weaker but still notable.

### Induction Heads

**Method**: Synthetic data — repeated random token sequences `[A B C ... | A B C ...]`. Measures the "offset diagonal" of attention in the second half: at position `L+k`, how much attention goes to position `k+1` (the token that followed the current token's earlier occurrence). This is the textbook induction pattern. 100 batches of 32, half-sequence length 256.

**Results**:
| Head | Score |
|------|-------|
| L2H4 | 0.629 |

No other head scores above 0.1. L2H4 is a strong, clean induction head.

The L1H1 → L2H4 pairing forms the classic two-layer induction circuit: L1H1 shifts information one position back (previous-token), composing with L2H4's key-query matching to attend to "what came after this token last time."

### Duplicate-Token Heads

**Method**: On real text, build a boolean mask of positions where a prior token has the same ID, then measure mean attention to those same-token positions. Only positions with at least one prior duplicate contribute to the score, and batches are weighted by the number of valid positions.

**Results**:
| Head | Score |
|------|-------|
| L0H4 | 0.323 |
| L0H2 | 0.202 |

All other heads below 0.05. Both heads are in layer 0, suggesting duplicate-token detection happens early.

### Successor Heads

**Method**: Constructs ordinal sequences (digits, letters, number words, days, months) as comma-separated lists and measures attention from each element to its ordinal predecessor (2 positions back, since commas intervene). A control condition uses random words in place of ordinals, with the same positional structure. The "signal" is ordinal score minus control score, isolating semantic successor attention from positional artifacts.

**Results** (signal > 0.05):
| Head | Ordinal | Control | Signal |
|------|---------|---------|--------|
| L0H2 | 0.379 | 0.073 | +0.307 |
| L0H4 | 0.155 | 0.001 | +0.154 |
| L1H0 | 0.098 | 0.041 | +0.058 |
| L1H1 | 0.174 | 0.108 | +0.067 |
| L3H0 | 0.192 | 0.121 | +0.070 |
| L1H2 | 0.497 | 0.443 | +0.054 |

L0H2 is the standout successor head. L0H4 is secondary. Several other heads show modest signals.

Note that L1H2 has high ordinal attention (0.497) but nearly as high control attention (0.443), suggesting it attends strongly to position-2-back regardless of content. The control subtraction properly removes this.

### S-Inhibition Heads

**Method**: Two-pronged analysis using IOI (Indirect Object Identification) prompts of the form "When Alice and Bob went to the store, Bob gave a drink to" → Alice.

1. **Data-driven**: Measures attention from the final position to the second occurrence of the subject name (S2). High S2 attention means the head is "looking at" the repeated name.
2. **Weight-based**: Computes the OV copy score `W_U[t] @ W_O_h @ W_V_h @ W_E[t]` averaged over name tokens. Negative values indicate the head suppresses (rather than promotes) the attended token's logit.

An S-inhibition head should have high S2 attention *and* a negative copy score.

**Results** (candidates: attn > 0.1 and copy < 0):
| Head | Attn to S2 | OV Copy | Assessment |
|------|-----------|---------|------------|
| L3H2 | 0.377 | -0.029 | Strongest candidate |
| L2H1 | 0.151 | -0.001 | Weak candidate |

L3H2 is the clearest S-inhibition candidate: it strongly attends to the repeated subject and has a negative copy score (suppression). L2H1 attends to S2 moderately but its copy score is only marginally negative.

Several other heads have high S2 attention but positive copy scores (e.g., L3H0 at attn=0.156, copy=+0.007), suggesting they *copy* the subject rather than inhibit it — a different role in the IOI circuit.

### Delimiter Heads

**Method**: On real text, identifies delimiter token IDs (`.` `,` `;` `:` `!` `?` `\n` and multi-char variants) via the tokenizer, then measures the mean fraction of each head's attention landing on delimiter tokens. Compares to the baseline delimiter frequency in the data (~10.7%). Reports the ratio over baseline.

**Results**: No head exceeds 2.0x baseline. Highest ratios:
| Head | Raw Attn | Ratio |
|------|----------|-------|
| L0H5 | 0.187 | 1.74x |
| L1H0 | 0.184 | 1.72x |
| L1H4 | 0.178 | 1.66x |

This model does not appear to have dedicated delimiter heads. Most heads sit in the 1.0-1.7x range — modestly above baseline but not specialized. This could reflect the model's small size, or it could mean delimiter attention is distributed across heads rather than concentrated.

### Positional Heads

**Method**: On real text, builds a mean attention profile by relative offset for each head (offset = query_pos - key_pos). Also measures attention to absolute position 0 (BOS). A "positional head" has high attention concentrated at a specific offset; a "BOS head" attends heavily to position 0.

**Results — offset-based**:
| Head | Max Offset Score | Peak Offset |
|------|-----------------|-------------|
| L1H1 | 0.604 | 1 |
| L0H5 | 0.308 | 1 |
| L1H0 | 0.265 | 1 |
| L1H3 | 0.226 | 2 |

L1H1 and L0H5 are the same heads already identified as previous-token heads, confirming the result from a different angle. L1H3 peaks at offset 2 — it preferentially attends two positions back.

**Results — BOS attention**:
| Head | BOS Score |
|------|-----------|
| L2H4 | 0.489 |
| L2H5 | 0.355 |
| L2H3 | 0.337 |
| L2H1 | 0.318 |
| L2H0 | 0.232 |
| L2H2 | 0.217 |
| L3H3 | 0.248 |
| L3H2 | 0.208 |
| L3H4 | 0.206 |

All six heads in layer 2 have substantial BOS attention (0.22–0.49). Layer 3 also shows moderate BOS attention in several heads. Layers 0–1 show negligible BOS attention (< 0.01).

## Cross-Cutting Observations

### Multi-functional early heads

L0H2 and L0H4 both serve as duplicate-token *and* successor heads. These are layer-0 heads operating directly on token embeddings, suggesting the behaviors may share a mechanism: attending to tokens that are "similar" to the current one (exact match for duplicate-token, ordinal neighbor for successor). Whether this reflects a single underlying computation or two coincidentally co-located behaviors is unclear from these analyses alone.

**Hypothesis**: L0H2 and L0H4 may implement a general "embedding similarity" attention pattern that manifests as duplicate-token detection on repeated tokens and successor detection on ordinal sequences. Testing this would require measuring the correlation between these heads' attention weights and embedding cosine similarity.

### The induction circuit

The L1H1 (previous-token) → L2H4 (induction) circuit is clean and well-separated. L1H1 scores 0.604 on previous-token and L2H4 scores 0.629 on induction, with no other head approaching either score. This is the textbook two-layer induction circuit.

### Layer 2 as a BOS sink

The uniform high BOS attention across all of layer 2 is striking. L2H4 — the induction head — has the highest BOS score (0.489) despite also being the strongest induction head. This might seem contradictory, but BOS attention and induction attention operate on different token positions: BOS attention is measured as an average across *all* query positions, while induction attention is measured specifically at positions following repeated sequences. L2H4 likely defaults to BOS when there's no induction pattern to match, using position 0 as an attention sink.

**Hypothesis**: Layer 2's BOS attention may serve as a "no-op" or default state. When a head doesn't have a strong content-based signal, it parks attention on BOS rather than distributing it noisily. This is a known phenomenon in transformer models (sometimes called "attention sinking"), and BOS is a natural sink since it's always available and semantically neutral in context.

### S-inhibition is late and sparse

Only L3H2 shows a convincing S-inhibition signal (layer 3, near the output). This makes architectural sense: S-inhibition requires first identifying the repeated subject (which depends on earlier duplicate-token and induction mechanisms) before suppressing it. The fact that it appears in the final layer is consistent with it being a downstream consumer of earlier head outputs.

### No dedicated delimiter heads

The absence of strong delimiter heads is a genuine null result, not a limitation of the method. The method would have detected them if present (the baseline-ratio approach has no inherent ceiling). This model apparently handles structural boundaries through other means, or distributes delimiter attention diffusely.

### Caveats

- All data-driven scores are averages. A head with a moderate average score might be strongly specialized on a subset of inputs and inactive on others. Per-example distributions would be more informative but are not captured here.
- The IOI template is a single fixed pattern. S-inhibition scores might differ with varied sentence structures.
- The successor head control condition (random words) controls for positional patterns but not for all confounds — e.g., if the tokenizer assigns similar embeddings to ordinal tokens, heads might use embedding similarity rather than "knowing" ordinal structure.
- OV copy scores (used in S-inhibition) are a linear approximation. They measure the direct path through one head and don't account for nonlinear interactions or composition with other heads/layers.

## Summary Table

| Head | Primary Role(s) | Evidence Strength |
|------|----------------|-------------------|
| L0H2 | Successor, duplicate-token | Strong (signal=0.307, dup=0.202) |
| L0H4 | Duplicate-token, successor | Strong (dup=0.323, signal=0.154) |
| L0H5 | Previous-token | Moderate (0.308) |
| L1H1 | Previous-token | Strong (0.604) |
| L1H3 | Offset-2 positional | Moderate (0.226 at offset 2) |
| L2H1 | Weak S-inhibition candidate | Weak (attn=0.151, copy=-0.001) |
| L2H4 | Induction, BOS sink | Strong induction (0.629), strong BOS (0.489) |
| L2H* | BOS sink (all layer 2) | Strong (0.22–0.49 across all heads) |
| L3H2 | S-inhibition | Moderate (attn=0.377, copy=-0.029) |

Heads not listed individually (L0H1, L0H3, L1H0, L1H2, L1H4, L1H5, L3H0–L3H1, L3H3–L3H5) did not show strong specialization in any analysis, though several had modest signals across multiple categories. Layer-2 heads without individual roles (L2H0, L2H2, L2H3, L2H5) are covered by the L2H* BOS sink row.
