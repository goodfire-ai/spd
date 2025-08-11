# SPD Clustering Module

Brief overview of Python files in `spd/clustering/`:

## `__init__.py`
Empty initialization file for the clustering module.

## `activations.py`
Handles component activation analysis (core functionality only - plotting moved to plotting/).

**Functions:**
- `component_activations()` - Gets component activations over a single batch from a ComponentModel
- `process_activations()` - Processes activations to compute coactivations, filter dead components, and optionally create plots

**Imports from clustering:**
- None (plotting functionality moved to plotting/ submodule)

## `compute_rank.py`
Efficient computation of matrix rank for sum of low-rank matrices.

**Functions:**
- `compute_rank_of_sum()` - Computes rank(P₁ + P₂) in O(d(r₁+r₂)²) time using SVD factors

**No imports from other clustering files.**

## `merge.py`
Core merging algorithm for grouping components based on coactivation patterns (core functionality only - plotting moved to plotting/).

**Functions:**
- `compute_merge_costs()` - Computes MDL costs for merge matrices
- `recompute_coacts_merge_pair()` - Updates coactivations after merging two groups
- `recompute_coacts_pop_group()` - Updates coactivations after splitting a component from its group
- `merge_iteration()` - Main iterative merging algorithm
- `merge_iteration_ensemble()` - Runs multiple merge iterations for ensemble analysis

**Classes:**
- `MergeConfig` - Configuration for merge algorithm parameters
- `MergePlotConfig` - Configuration for merge plotting
- `MergeHistory` - Tracks merge iteration history with plot() method that delegates to plotting/
- `MergeEnsemble` - Container for multiple merge histories

**Imports from clustering:**
- `GroupMerge` from `merge_matrix`

## `merge_matrix.py`
Data structures for representing component-to-group assignments.

**Classes:**
- `GroupMerge` - Canonical component-to-group assignment with merge operations
- `BatchedGroupMerge` - Batch of merge matrices for efficient processing

**Methods include:**
- Matrix conversion operations
- Group merging/splitting
- Distance calculation between assignments
- Visualization

**Imports from clustering:**
- `perm_invariant_hamming` from `perm_invariant_hamming`
- `add_component_labeling` from `plotting.activations`

## `merge_sweep.py`
Utilities for parameter sweeps in merge ensemble analysis.

**Functions:**
- `sweep_merge_parameter()` - Runs ensemble merge iterations for different values of a single parameter
- `sweep_multiple_parameters()` - Runs multiple parameter sweeps

**Imports from clustering:**
- `MergeConfig`, `MergeEnsemble`, `MergePlotConfig`, `merge_iteration_ensemble` from `merge`
- `plot_dists_distribution` from `plotting.merge`

## `perm_invariant_hamming.py`
Computes permutation-invariant Hamming distance between labelings.

**Functions:**
- `perm_invariant_hamming()` - Computes minimum Hamming distance between two labelings up to optimal relabeling using Hungarian algorithm

**No imports from other clustering files.**

## `sweep.py`
Comprehensive hyperparameter sweep and visualization utilities.

**Functions:**
- `run_hyperparameter_sweep()` - Runs sweep across all parameter combinations
- `plot_evolution_histories()` - Plots evolution histories with 3D parameter organization
- `create_heatmaps()` - Creates flexible heatmaps showing statistics across hyperparameter combinations
- `create_multiple_heatmaps()` - Creates multiple heatmaps with different statistics
- `create_smart_heatmap()` - Creates heatmap with automatic parameter selection
- Various statistic functions (convergence rate, cost reduction, merge efficiency)

**Classes:**
- `SweepConfig` - Configuration for hyperparameter sweep

**Imports from clustering:**
- `MergeConfig`, `MergeHistory`, `MergePlotConfig`, `merge_iteration` from `merge`

## `util.py`
Utility functions for the clustering module.

**Functions:**
- `named_lambda()` - Helper to create named lambda functions for sweeps
- `format_scientific_latex()` - Formats numbers in LaTeX scientific notation

**No imports from other clustering files.**

## `plotting/`
Plotting functionality separated from core logic.

### `plotting/__init__.py`
Exposes main plotting functions: `add_component_labeling`, `plot_activations`, `plot_dists_distribution`, `plot_merge_iteration`.

### `plotting/activations.py`
Activation visualization functions moved from `activations.py`.

**Functions:**
- `plot_activations()` - Plots activation visualizations including raw, concatenated, sorted, and coactivations
- `add_component_labeling()` - Adds component labeling to plot axes showing module boundaries

### `plotting/merge.py`
Merge visualization functions moved from `merge.py`.

**Functions:**
- `plot_merge_iteration()` - Visualizes merge iteration results with merge tree, coactivations, and costs
- `plot_dists_distribution()` - Plots distribution of pairwise distances in ensemble
- `plot_merge_history()` - Plots cost evolution from merge history






# Notes 2025-08-11 10:33

on mem usage bottleneck for coact matrix:
- randomly sample components across layers
- compute coacts chunkwise, filter anything below some threshold
- hierarchical -- do one modules/layer at a time first