# How Alpha Affects Clustering Decisions

## TL;DR

Alpha controls the **weight penalty** in the MDL (Minimum Description Length) cost function. Higher alpha means you need **more coactivation** between components to justify merging them. At alpha=0, merging is driven purely by dictionary size reduction; at high alpha, you need strong coactivation evidence.

## The Cost Function

The clustering algorithm uses `compute_merge_costs()` to calculate the **change in MDL cost** if two groups were merged. A merge happens if this cost is negative (beneficial) or among the lowest costs.

The cost formula (from `compute_costs.py:58-68`) is:

```
F(i, j) = s_other + bits_local + alpha * penalty
```

Where:
- `s_i`, `s_j` = activation frequencies of components i and j (diagonal of normalized coact matrix)
- `s_{i,j}` = activation frequency of merged component = `s_i + s_j - coact_ij` (OR-semantics)
- `coact_ij` = how often both i AND j activate together
- `r(P)` = rank = number of original subcomponents in a cluster
- `k` = current number of groups

### Term 1: Dictionary Size Benefit (`s_other`)

```python
s_other = (s_total - s_i - s_j) * log2((k-1)/k)
```

This is always **negative** (beneficial for merging). When we merge, all OTHER components benefit from a smaller dictionary (fewer bits to encode which component is active).

### Term 2: Local Index Cost (`bits_local`)

```python
bits_local = s_{i,j} * log2(k-1) - s_i * log2(k) - s_j * log2(k)
```

This captures the change in bits for encoding component indices. After merging, we need to send only 1 index when either (or both) components activate.

### Term 3: Weight Penalty (`penalty`) - **This is where alpha matters**

```python
penalty = s_{i,j} * r(P_{i,j}) - s_i * r(P_i) - s_j * r(P_j)
```

When a merged component activates, we must "send" all weights from both original components. This term penalizes unnecessary weight transmission.

## Alpha's Effect on Two Singleton Components

For two **singleton components** (rank 1 each, not yet merged with anything):
- `r(P_i) = 1`, `r(P_j) = 1`, `r(P_{i,j}) = 2`

The penalty simplifies to:

```
penalty = 2 * s_{i,j} - s_i - s_j
        = 2 * (s_i + s_j - coact_ij) - s_i - s_j
        = s_i + s_j - 2 * coact_ij
```

**Interpretation**: The penalty equals the "wasted activations" - how often we send weights for a component that didn't actually fire.

### When is penalty negative (good for merging)?

```
s_i + s_j - 2 * coact_ij < 0
coact_ij > (s_i + s_j) / 2
```

The penalty favors merging when the coactivation rate exceeds half the sum of individual activation rates.

## Answering Your Question

> "If we were only deciding whether to cluster two lone subcomponents, how often would they need to coactivate to be clustered at alpha=1?"

It's not just about pure coactivation threshold because **all three terms interact**:

1. **At alpha=0**: The penalty term vanishes. Merging is decided purely by `s_other + bits_local`. Since `s_other` is always negative (favoring merging), components tend to merge readily regardless of coactivation patterns.

2. **At alpha=1**: The penalty term is fully weighted. You need sufficient coactivation to offset the weight transmission cost. Roughly, pairs with `coact_ij > (s_i + s_j)/2` will have negative penalty, helping the merge case.

3. **At high alpha**: The penalty dominates. Only pairs with very strong coactivation (approaching `min(s_i, s_j)` - i.e., one always fires when the other does) will be favored for merging.

## Concrete Example

Say we have two components with:
- `s_i = 0.1` (activates 10% of the time)
- `s_j = 0.1` (activates 10% of the time)
- `k = 100` groups

**If they're independent** (coact_ij ≈ 0.01):
```
penalty = 0.1 + 0.1 - 2*0.01 = 0.18  (positive = bad for merge)
```

**If they always coactivate** (coact_ij = 0.1):
```
penalty = 0.1 + 0.1 - 2*0.1 = 0  (neutral)
```

**If one always triggers the other** (perfect subset, coact_ij = 0.1):
```
penalty = 0  (neutral on penalty, but s_other still favors merge)
```

## The Algorithm Always Merges

The `range_sampler` (and the algorithm generally) **always picks a merge at every step**. Alpha doesn't control *whether* to merge - it controls **which pairs** are most attractive to merge by changing the relative cost landscape.

With higher alpha:
- Pairs with high coactivation have much lower (more negative) costs
- Pairs with low coactivation have higher costs
- The algorithm will prefer to merge highly-coactivating pairs first

With lower alpha:
- The cost landscape is flatter
- Dictionary size benefits dominate
- The algorithm is less discriminating about coactivation patterns

## Summary Table

| Alpha Value | Effect | Behavior |
|-------------|--------|----------|
| 0 | Penalty ignored | Merge based on dictionary size savings only |
| 1 (default) | Balanced | Moderate coactivation required |
| High (>1) | Penalty dominates | Only strongly coactivating pairs merge early |

## Code References

- Cost computation: `spd/clustering/compute_costs.py:38-119`
- MDL cost: `spd/clustering/compute_costs.py:11-35`
- Merge iteration: `spd/clustering/merge.py:105-109`
- Alpha config: `spd/clustering/merge_config.py:51-54`
