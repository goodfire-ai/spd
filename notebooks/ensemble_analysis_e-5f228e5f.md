# Ensemble Clustering Analysis Report

**Ensemble ID:** `e-5f228e5f`
**Dataset:** ss_llama_simple_mlp-1L
**Components:** 442 neurons from `h.0.mlp.down_proj`
**Ensemble:** 10 independent clustering runs
**Iterations:** 300 merges per run

---

## Key Finding: Phase Transition in Clustering Stability

The ensemble exhibits a clear phase transition around iteration 73, visible in the distance distribution plot where many run pairs maintain 0 Hamming distance until ~iteration 100.

### Early Phase (iter 0-73): Self-Correcting Regime

- Runs make different merge decisions but converge to identical partitions
- A "catch-up" mechanism operates: runs that diverge re-synchronize within a few iterations
- All 45 run pairs reach 0 Hamming distance periodically
- Clustering is dominated by one large cluster absorbing singletons
- First divergence occurs at iteration 2, but runs keep re-converging

**Convergence iterations (all 10 runs identical):**
0, 1, 3, 5, 6, 7, 8, 10, 11, 12, 15, 17, 18, 19, 20, 22, 25, 26, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 49, 50, 52, 53, 56, 57, 58, 63, 65, 68, 69, 70, 72, 73

### Transition Phase (iter 73-105): Bifurcation Point

- 2nd largest cluster reaches size 5 at iteration 71
- 3rd largest cluster reaches size 5 at iteration 80
- Multiple valid merge paths emerge with similar scores
- Partition divergence becomes chaotic (oscillates between 3-10 unique partitions)
- Last full convergence at iteration 73

| Iteration | Unique Partitions | Zero-Distance Pairs |
|-----------|-------------------|---------------------|
| 73 | 1 | 45 |
| 80 | 7 | 4 |
| 85 | 9 | 1 |
| 90 | 3 | 28 |
| 100 | 8 | 2 |
| 105 | 10 | 0 |

### Late Phase (iter 105+): Permanently Diverged

- All 10 runs have distinct partitions
- No zero-distance pairs remain after iteration 114
- Hamming distances grow approximately linearly

| Iteration | Min Distance | Max Distance | Mean Distance |
|-----------|--------------|--------------|---------------|
| 150 | 6 | 23 | 13.84 |
| 200 | 17 | 36 | 26.91 |
| 299 | 46 | 84 | 68.18 |

---

## Cluster Size Evolution

The phase transition coincides with the emergence of competing secondary clusters:

| Iteration | # Clusters | Largest | 2nd Largest | 3rd Largest | Singletons |
|-----------|------------|---------|-------------|-------------|------------|
| 0 | 441 | 2 | 1 | 1 | 440 |
| 25 | 416 | 27 | 1 | 1 | 415 |
| 50 | 391 | 52 | 1 | 1 | 390 |
| 73 | 368 | 64 | 5 | 3 | 361 |
| 100 | 341 | 65 | 12 | 10 | 329 |
| 150 | 291 | 67 | 15 | 13 | 262 |
| 299 | 142 | 68 | 16 | 14 | 47 |

---

## Structure of Disagreement

Despite divergence in Hamming distance, runs agree on most of the clustering structure.

### Pair-wise Co-clustering Stability (at iteration 150)

| Category | Count | Percentage |
|----------|-------|------------|
| Total component pairs | 97,461 | 100% |
| Always co-clustered (10/10 runs) | 2,521 | 2.6% |
| Never co-clustered (0/10 runs) | 94,749 | 97.2% |
| Contested (1-9/10 runs) | 191 | 0.2% |

**99.8% of pairs have stable co-clustering across all runs.**

### Contested Neurons

The 191 contested pairs involve 164 "boundary" neurons. Four neurons dominate the contestedness:

| Neuron | Contested Pairs |
|--------|-----------------|
| 598 | 70 |
| 24 | 68 |
| 310 | 68 |
| 60 | 68 |

### Neighbor Consistency

For each component, we measure what fraction of its cluster neighbors are consistent across runs:

| Consistency Level | # Components | Percentage |
|-------------------|--------------|------------|
| 100% consistent | 278 | 63% |
| >90% consistent | 355 | 80% |
| <50% consistent | 48 | 11% |

**Mean neighbor consistency: 0.85**

---

## Run Pair Analysis

### Pairs That Stayed Similar Longest

| Run Pair | Last Zero-Distance | First Divergence |
|----------|-------------------|------------------|
| Run 7 vs Run 2 | iter 114 | iter 2 |
| Run 3 vs Run 0 | iter 114 | iter 9 |
| Run 5 vs Run 4 | iter 112 | iter 9 |
| Run 6 vs Run 0 | iter 107 | iter 16 |
| Run 8 vs Run 3 | iter 103 | iter 9 |

Note: "First divergence" is early but pairs re-converge via the catch-up mechanism until the phase transition.

---

## Interpretation

The clustering algorithm finds a robust "core" structure early, but has ambiguity about ~37% of components (boundary neurons). These neurons could plausibly belong to multiple clusters, and stochastic tie-breaking determines their final assignment.

**Implications:**

1. **Dominant cluster (68 neurons):** Robust finding across all runs
2. **Secondary clusters (15, 13 neurons):** Mostly stable
3. **Boundary neurons (~164):** Uncertain cluster membership
4. **Ensemble methods:** Could be used to identify ambiguous neurons and quantify clustering confidence

The phase transition occurs when the clustering moves from a "greedy absorption" regime (one dominant cluster eating singletons) to a "competition" regime (multiple medium-sized clusters with similar merge scores).

---

## Generated Figures

- `plots/distances_perm_invariant_hamming.png` - Original distance distribution
- `plots/phase_transition_analysis.png` - Three-panel phase transition visualization
- `plots/ensemble_analysis_summary.png` - Four-panel summary figure
