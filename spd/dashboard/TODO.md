# general

- [x] **[BUG]** fix issue with broken histograms
- [ ] not just current token, but true next token and predicted next token when a given component is active
- [ ] embeddings view
- [ ] a view where we can see, for some selected components, their activations (both actual and DT predicted) on a random set of samples -- stacked vertically so that we can compare. kinda like https://attention-motifs.github.io/v1/vis/attnpedia/index.html

# index.html

- [ ] link to wandb run
- [ ] columns with measures of token entropy
- [ ] fix table csv export


# component.html

- [ ] proportion of all tokens that is any given token in table
- [ ] dists of activations in each sample
- [ ] hovering a token highlights all occurrences in the text
- [ ] token statistics -- highlight percentages by magnitude
- [ ] dists of activations per token?

# Dashboard Summary Code Implementation

## 1. Split conditional_matrices() in compute.py

- [ ] Extract shared _compute_activated_per_token() helper function
- [ ] Create p_activation_given_token() function (with Laplace smoothing)
- [ ] Create p_token_given_activation() function
- [ ] Keep conditional_matrices() as convenience wrapper that calls both efficiently

## 2. Add missing data classes to summary.py

- [ ] Add TopKSample class (token_strs, activations)
- [ ] Add TokenActivationsSummary class (top_tokens, entropy, concentration_ratio, etc.)
- [ ] Update SubcomponentSummary.stats type hint to include token_activations

## 3. Implement statistics computation in summary.py

- [ ] Stats dict: mean, std, min, max, median, q05, q25, q75, q95
- [ ] Histograms dict: all_activations, max_per_sample, mean_per_sample (using existing _make_histogram())
- [ ] Token stats list: Use hybrid approach:
  - [ ] Compute both probability matrices once for all components via conditional_matrices()
  - [ ] Extract per-component probabilities
  - [ ] Build unified TokenStat list (union of top N by each metric)
- [ ] Top samples: Find top K by max and mean activation (union, deduplicated)
- [ ] Token activations summary: Entropy, concentration ratio, top tokens

## 4. Update method signatures to accept config

- [ ] SubcomponentSummary.create() - add config: ComponentDashboardConfig parameter
- [ ] ActivationsSummary.from_activations() - add config: ComponentDashboardConfig parameter
- [ ] Use config values: hist_bins, hist_range, token_active_threshold, token_stats_top_n, etc.

## 5. Update ci_dt_min.py to pass config

- [ ] Import ComponentDashboardConfig from dashboard_config.py
- [ ] Create config instance
- [ ] Pass config when calling summary methods

**Files to modify:**
- spd/dashboard/core/compute.py (split conditional_matrices)
- spd/dashboard/core/summary.py (implement all computation logic, add classes)
- spd/dashboard/core/ci_dt_min.py (pass config to summary methods)