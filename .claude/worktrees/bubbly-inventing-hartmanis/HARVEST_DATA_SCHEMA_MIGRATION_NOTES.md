# Harvest DB Schema Migration: `mean_ci` to `firing_density` + `mean_activations`

**Date investigated**: 2026-02-18
**Current branch**: `feature/attn_plots`
**Status**: Not yet migrated. Workaround in place for plotting scripts.

## What happened

A colleague is generalizing the harvest pipeline to work across decomposition methods (SPD, MOLT, CLT, SAE) on branches `feature/harvest-generic` and `feature/autointerp-generic`. As part of this, the harvest DB schema changed. The key commit is `70eceb8f` ("Generalize harvest pipeline over decomposition methods") by Claude SPD1, dated 2026-02-16.

A new harvest sub-run for `s-275c8f21` was created on 2026-02-18 using code from `feature/harvest-generic`, producing data with the new schema. This data is incompatible with the code on `dev`, `main`, and `feature/attn_plots`, which still expect the old schema.

## Schema change details

### Old schema (on `dev`, `main`, `feature/attn_plots`)

```sql
CREATE TABLE components (
    component_key TEXT PRIMARY KEY,
    layer TEXT NOT NULL,
    component_idx INTEGER NOT NULL,
    mean_ci REAL NOT NULL,              -- mean causal importance across all tokens
    activation_examples TEXT NOT NULL,
    input_token_pmi TEXT NOT NULL,
    output_token_pmi TEXT NOT NULL
);
```

`ActivationExample` dataclass fields: `token_ids: list[int]`, `ci_values: list[float]`, `component_acts: list[float]`

### New schema (on `feature/harvest-generic`, commit `70eceb8f`)

```sql
CREATE TABLE components (
    component_key TEXT PRIMARY KEY,
    layer TEXT NOT NULL,
    component_idx INTEGER NOT NULL,
    firing_density REAL NOT NULL,       -- proportion of tokens where component fired (0-1)
    mean_activations TEXT NOT NULL,     -- JSON dict, e.g. {"causal_importance": 0.007}
    activation_examples TEXT NOT NULL,
    input_token_pmi TEXT NOT NULL,
    output_token_pmi TEXT NOT NULL
);
```

`ActivationExample` dataclass fields: `token_ids: list[int]`, `firings: list[bool]`, `activations: dict[str, list[float]]`

### Field mapping

| Old field | New field | Notes |
|---|---|---|
| `mean_ci` (float) | `mean_activations["causal_importance"]` (float, inside JSON dict) | Same semantic meaning for SPD runs |
| *(not present)* | `firing_density` (float) | New metric: proportion of tokens where component fired |
| `ci_values` (on ActivationExample) | `activations["causal_importance"]` (on ActivationExample) | Per-token CI values, now keyed by activation type |
| `component_acts` (on ActivationExample) | `activations["component_activation"]` (on ActivationExample) | Per-token component activations |
| *(not present)* | `firings` (on ActivationExample) | Boolean per-token firing indicators |

### Example new data row

From `s-275c8f21` sub-run `h-20260218_000000`:
```
firing_density: 0.011455078125
mean_activations: {"causal_importance": 0.007389060687273741}
```

## Branches involved

| Branch | Schema version | Status |
|---|---|---|
| `main` | Old (`mean_ci`) | Production |
| `dev` | Old (`mean_ci`) | Development |
| `feature/attn_plots` | Old (`mean_ci`) | Current work branch |
| `feature/harvest-generic` | New (`firing_density` + `mean_activations`) | Colleague's WIP |
| `feature/autointerp-generic` | New | Colleague's WIP |

The schema change commits (`70eceb8f`, `5e66fd49`, `ad68187d`) exist **only** on `feature/harvest-generic` and `feature/autointerp-generic`. They are NOT on `dev` or `main`.

## Current workaround

The broken sub-run was renamed so the `HarvestRepo` skips it:

```
_h-20260218_000000.bak    <-- new schema, renamed with _ prefix
h-20260212_150336          <-- old schema, now picked as "latest" by HarvestRepo
```

`HarvestRepo.open()` picks the latest `h-*` directory by lexicographic sort. The `_` prefix prevents the glob match on `d.name.startswith("h-")` (see `spd/harvest/repo.py:46`).

## Files that need updating when migrating

### Core harvest module (update schema definitions)

1. **`spd/harvest/schemas.py`** — `ComponentSummary` and `ComponentData` dataclasses: replace `mean_ci: float` with `firing_density: float` + `mean_activations: dict[str, float]`. Also update `ActivationExample`.
2. **`spd/harvest/db.py`** — SQL schema, `_serialize_component()`, `_deserialize_component()`, `get_summary()`, `get_all_components()`.
3. **`spd/harvest/harvester.py`** — `build_results()` yields `ComponentData`; needs to compute `firing_density` and `mean_activations` dict.

Reference implementation: `git show 70eceb8f:spd/harvest/schemas.py` and `git show 70eceb8f:spd/harvest/db.py`.

### App backend (API schemas + endpoints)

4. **`spd/app/backend/schemas.py`** — `SubcomponentMetadata.mean_ci` and `SubcomponentActivationContexts.mean_ci`.
5. **`spd/app/backend/routers/activation_contexts.py`** — Extracts and sorts by `mean_ci` in 3 endpoints.

### Autointerp module

6. **`spd/autointerp/interpret.py`** — Sorts components by `c.mean_ci` (line 116).
7. **`spd/autointerp/strategies/compact_skeptical.py`** — Uses `mean_ci * 100` and `1 / component.mean_ci` for LLM prompt formatting.

### Dataset attributions

8. **`spd/dataset_attributions/harvest.py`** — Filters alive components with `summary[key].mean_ci > ci_threshold` (line 90).

### Plotting scripts (9 files, all follow same pattern)

All have `MIN_MEAN_CI` constant and `_get_alive_indices()` that filters on `s.mean_ci > threshold`:

9. `spd/scripts/plot_qk_c_attention_contributions/plot_qk_c_attention_contributions.py`
10. `spd/scripts/attention_stories/attention_stories.py`
11. `spd/scripts/characterize_induction_components/characterize_induction_components.py`
12. `spd/scripts/plot_kv_vt_similarity/plot_kv_vt_similarity.py`
13. `spd/scripts/plot_attention_weights/plot_attention_weights.py`
14. `spd/scripts/plot_kv_coactivation/plot_kv_coactivation.py`
15. `spd/scripts/plot_per_head_component_activations/plot_per_head_component_activations.py`
16. `spd/scripts/plot_head_spread/plot_head_spread.py`
17. `spd/scripts/plot_component_head_norms/plot_component_head_norms.py`

### Dedicated CI visualization

18. **`spd/scripts/plot_mean_ci/plot_mean_ci.py`** — Entire script dedicated to visualizing `mean_ci` distributions. May need renaming or deprecation.

## Recommended migration approach

Once `feature/harvest-generic` is stable and merged to `dev`:

1. **Port core schema changes** from `feature/harvest-generic` (items 1-3 above). The reference implementation at commit `70eceb8f` has the complete updated `db.py` and `schemas.py`.

2. **Update consumers** (items 4-18). For filtering "alive" components, the equivalent of `mean_ci > threshold` is `mean_activations["causal_importance"] > threshold`. Consider whether `firing_density` would be a better filter (it's a cleaner concept: "does this component fire often enough?").

3. **No legacy fallback needed** — per repo conventions (CLAUDE.md: "Don't add legacy fallbacks or migration code"). Old harvest data should be re-harvested with new code if needed.

4. **Consider extracting `_get_alive_indices`** into a shared utility — it's duplicated across 9 plotting scripts with identical logic.
