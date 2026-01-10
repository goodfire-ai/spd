# Frontend Audit TODOs

## High Priority

- [x] **Interpretation caching inefficiency** (`useComponentData.svelte.ts`)
    - Check `runState.getInterpretation()` before API call
    - After generation, update dict directly instead of full reload

- [x] **Dual componentDetail loading** (`ActivationContextsViewer.svelte:39, 93-105`)
    - Remove manual `currentComponent` state
    - Use `componentData.componentDetail` directly

- [x] **clusterMapping not cleared on run change** (`LocalAttributionsTab.svelte:145`)
    - Absorbed clusterMapping into runState with auto-cleanup via $effect.pre

## Medium Priority

- [ ] **Correlations/token stats not cached** (`useComponentData.svelte.ts`)
    - Add runState caching similar to componentDetail

- [ ] **Activation context detail linear scan** (backend `activation_contexts.py:86-91`)
    - Add index file or split by layer for O(1) lookup

- [ ] **Nullable type semantics** (`useComponentData.svelte.ts:35-37`)
    - Resolve `Loadable<T | null>` ambiguity

## Low Priority

- [ ] Extract duplicated token list derivation to utility
- [ ] Bundle NodeTooltip props
- [ ] Cache token lookup by tokenizer name
- [ ] Add HarvestCache.add_interpretation() method

## Slightly different

- [ ]
