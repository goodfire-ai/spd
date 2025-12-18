# Clustering Cost Examples

## Setup

- **1000 original subcomponents**, so initially k=1000 clusters
- After some merging, we now have **k=951 clusters**:
  - 1 large cluster of **rank 50** (contains 50 original subcomponents)
  - 950 singleton clusters of **rank 1**
- All activations are normalized by number of samples (so they're frequencies in [0,1])

## Cost Function Recap

```
F(i, j) = s_other + bits_local + alpha * penalty
```

Where:
- `s_other = (s_total - s_i - s_j) * log2((k-1)/k)`
- `bits_local = s_{i,j} * log2(k-1) - s_i * log2(k) - s_j * log2(k)`
- `penalty = s_{i,j} * (r_i + r_j) - s_i * r_i - s_j * r_j`
- `s_{i,j} = s_i + s_j - coact_ij` (OR-activation of merged pair)

With k=951:
- `log2(950/951) ≈ -0.00152`
- `log2(950) ≈ 9.89`
- `log2(951) ≈ 9.89`

---

## Example 1: Two Singletons That Always Coactivate

**Scenario**: Components A and B both activate 10% of the time, and they ALWAYS activate together.

| Parameter | Value |
|-----------|-------|
| s_A | 0.10 |
| s_B | 0.10 |
| coact_AB | 0.10 |
| r_A | 1 |
| r_B | 1 |

**Derived values**:
- `s_{A,B} = 0.10 + 0.10 - 0.10 = 0.10` (merged component fires 10% of time)
- `s_total ≈ 1.0` (assume total activation sums to ~1 for simplicity)

**Cost computation**:

```
s_other = (1.0 - 0.1 - 0.1) * (-0.00152) = -0.00122

bits_local = 0.10 * 9.89 - 0.10 * 9.89 - 0.10 * 9.89
           = 0.989 - 0.989 - 0.989 = -0.989

penalty = 0.10 * (1+1) - 0.10 * 1 - 0.10 * 1
        = 0.20 - 0.10 - 0.10 = 0.00
```

**Total cost (alpha=1)**: `-0.00122 - 0.989 + 1.0 * 0.00 = -0.99` ✅ **Strongly favored merge**

**Interpretation**: Perfect coactivation means zero penalty. The merge is driven by the `bits_local` savings (we stop sending two indices).

---

## Example 2: Two Singletons That Never Coactivate

**Scenario**: Components A and B both activate 10% of the time, but NEVER together.

| Parameter | Value |
|-----------|-------|
| s_A | 0.10 |
| s_B | 0.10 |
| coact_AB | 0.00 |
| r_A | 1 |
| r_B | 1 |

**Derived values**:
- `s_{A,B} = 0.10 + 0.10 - 0.00 = 0.20` (merged fires 20% - whenever either fires)

**Cost computation**:

```
s_other = (1.0 - 0.1 - 0.1) * (-0.00152) = -0.00122

bits_local = 0.20 * 9.89 - 0.10 * 9.89 - 0.10 * 9.89
           = 1.978 - 0.989 - 0.989 = 0.00

penalty = 0.20 * (1+1) - 0.10 * 1 - 0.10 * 1
        = 0.40 - 0.10 - 0.10 = 0.20
```

**Total cost (alpha=1)**: `-0.00122 + 0.00 + 1.0 * 0.20 = +0.199` ❌ **Unfavored merge**

**Interpretation**: Zero coactivation means maximum penalty. We'd be sending both weights every time either fires, wasting transmission.

---

## Example 3: Two Singletons With Independent Activation

**Scenario**: A and B both activate 10%, coactivating at the independent rate (1% = 10% × 10%).

| Parameter | Value |
|-----------|-------|
| s_A | 0.10 |
| s_B | 0.10 |
| coact_AB | 0.01 |
| r_A | 1 |
| r_B | 1 |

**Derived values**:
- `s_{A,B} = 0.10 + 0.10 - 0.01 = 0.19`

**Cost computation**:

```
s_other = -0.00122

bits_local = 0.19 * 9.89 - 0.10 * 9.89 - 0.10 * 9.89
           = 1.879 - 1.978 = -0.099

penalty = 0.19 * 2 - 0.10 - 0.10
        = 0.38 - 0.20 = 0.18
```

**Total cost (alpha=1)**: `-0.00122 - 0.099 + 0.18 = +0.08` ❌ **Slightly unfavored**

**Total cost (alpha=0.5)**: `-0.00122 - 0.099 + 0.5 * 0.18 = -0.01` ✅ **Slightly favored**

**Interpretation**: Independent components are borderline. Lower alpha tips the balance toward merging.

---

## Example 4: Merging a Singleton Into the Big Cluster (Rank 50)

**Scenario**: The big cluster B activates 40% of the time. Singleton A activates 10%. They coactivate 8% (fairly often when A fires, B is also firing).

| Parameter | Value |
|-----------|-------|
| s_A | 0.10 |
| s_B | 0.40 |
| coact_AB | 0.08 |
| r_A | 1 |
| r_B | 50 |

**Derived values**:
- `s_{A,B} = 0.10 + 0.40 - 0.08 = 0.42`

**Cost computation**:

```
s_other = (1.0 - 0.1 - 0.4) * (-0.00152) = -0.00076

bits_local = 0.42 * 9.89 - 0.10 * 9.89 - 0.40 * 9.89
           = 4.15 - 0.99 - 3.96 = -0.80

penalty = 0.42 * (1+50) - 0.10 * 1 - 0.40 * 50
        = 0.42 * 51 - 0.10 - 20.0
        = 21.42 - 0.10 - 20.0 = 1.32
```

**Total cost (alpha=1)**: `-0.00076 - 0.80 + 1.32 = +0.52` ❌ **Unfavored**

**Total cost (alpha=0.5)**: `-0.00076 - 0.80 + 0.66 = -0.14` ✅ **Favored**

**Interpretation**: The big cluster is "expensive" to activate (rank 50 means sending 50 weights). Even with decent coactivation, merging into it has a high penalty at alpha=1.

---

## Example 5: Merging a Singleton That's a Subset of the Big Cluster

**Scenario**: A activates 10%, B activates 40%, and A ALWAYS coactivates with B (A is a "subset" of B).

| Parameter | Value |
|-----------|-------|
| s_A | 0.10 |
| s_B | 0.40 |
| coact_AB | 0.10 |
| r_A | 1 |
| r_B | 50 |

**Derived values**:
- `s_{A,B} = 0.10 + 0.40 - 0.10 = 0.40` (same as B alone!)

**Cost computation**:

```
s_other = -0.00076

bits_local = 0.40 * 9.89 - 0.10 * 9.89 - 0.40 * 9.89
           = 3.96 - 0.99 - 3.96 = -0.99

penalty = 0.40 * 51 - 0.10 * 1 - 0.40 * 50
        = 20.4 - 0.10 - 20.0 = 0.30
```

**Total cost (alpha=1)**: `-0.00076 - 0.99 + 0.30 = -0.69` ✅ **Favored**

**Interpretation**: When A is a subset of B, merging makes sense even with the big cluster - we're only adding 1 to the rank, and A already fires when B fires anyway.

---

## Example 6: High-Frequency Singleton vs Low-Frequency Singleton

**Scenario**: A activates 50%, B activates 2%, they coactivate 1.5% (B usually fires when A is on).

| Parameter | Value |
|-----------|-------|
| s_A | 0.50 |
| s_B | 0.02 |
| coact_AB | 0.015 |
| r_A | 1 |
| r_B | 1 |

**Derived values**:
- `s_{A,B} = 0.50 + 0.02 - 0.015 = 0.505`

**Cost computation**:

```
s_other = (1.0 - 0.5 - 0.02) * (-0.00152) = -0.00073

bits_local = 0.505 * 9.89 - 0.50 * 9.89 - 0.02 * 9.89
           = 4.99 - 4.95 - 0.20 = -0.16

penalty = 0.505 * 2 - 0.50 * 1 - 0.02 * 1
        = 1.01 - 0.50 - 0.02 = 0.49
```

**Total cost (alpha=1)**: `-0.00073 - 0.16 + 0.49 = +0.33` ❌ **Unfavored**

**Interpretation**: The high-frequency component A would "carry" B along most of the time, sending B's weights unnecessarily. Even though B fires mostly when A is on (75% of B's activations overlap with A), the raw coactivation rate is low compared to A's frequency.

---

## Summary: What Gets Merged First?

Given our scenario (1 cluster of rank 50, 950 singletons), at **alpha=1**:

| Merge Type | Typical Cost | Priority |
|------------|--------------|----------|
| Two singletons, perfect coactivation | ~ -1.0 | **Highest** |
| Singleton subset of big cluster | ~ -0.7 | High |
| Two singletons, 50% coactivation | ~ -0.5 | Medium |
| Two singletons, independent | ~ +0.1 | Low |
| Singleton into big cluster (moderate coact) | ~ +0.5 | Very Low |
| Two singletons, never coactivate | ~ +0.2 | Very Low |

**Key insight**: The big cluster is "protected" from absorbing unrelated singletons because its high rank makes the penalty term large. It will only absorb components that are true subsets of its activation pattern.

---

## Effect of Alpha

| Alpha | Behavior |
|-------|----------|
| 0.0 | Big cluster grows aggressively (penalty ignored) |
| 0.5 | Moderate protection for big cluster |
| 1.0 | Strong protection; only subsets merge into big cluster |
| 2.0 | Very strong protection; even moderate coactivation won't justify merging into big cluster |

At **alpha=0**, Examples 4 and 6 would become favored merges (costs around -0.8), causing the big cluster to absorb nearby singletons regardless of coactivation patterns.
