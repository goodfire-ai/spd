# Plan: Generalize App to Arbitrary Transformer Models

## Overview

Remove all model-specific, tokenizer-specific, and dataset-specific hardcoding from the app so
it works with arbitrary transformer language models. Introduce two new abstractions:
`AppTokenizer` (wraps HF tokenizer weirdness) and `ModelAdapter` (wraps model topology).

## Phase 1: `AppTokenizer`

Replace `build_token_lookup()` and the `token_strings: dict[int, str]` on `RunState`.

### 1.1 Create `spd/app/backend/app_tokenizer.py`

```python
class AppTokenizer:
    _tok: PreTrainedTokenizerBase
    _is_fast: bool

    def get_spans(self, token_ids: list[int]) -> list[str]
        # offset_mapping approach with overlap dedup, fallback for non-round-trippable
    def get_tok_display(self, token_id: int) -> str
        # Single token decode for vocab browsers / hover labels
    def encode(self, text: str) -> list[int]
```

### 1.2 Update `RunState` (state.py:108-119)

- Remove `token_strings: dict[int, str]`
- Add `app_tokenizer: AppTokenizer`

### 1.3 Update `load_run()` (runs.py:44-126)

- Remove `build_token_lookup()` call (line 112)
- Construct `AppTokenizer(loaded_tokenizer)` instead
- Pass to `RunState`

### 1.4 Migrate all `token_strings` callsites

These all do `loaded.token_strings[tid]` → replace with `loaded.app_tokenizer.get_tok_display(tid)`:

- `routers/prompts.py:66,202,264` — token list for prompt display; use `get_spans(token_ids)` instead
- `routers/graphs.py:182,217,275,284,358,380,409,508,509,566,596,673,678,700,714,719,766` — mix of per-token display and span construction
- `routers/activation_contexts.py:78,83-84,127` — per-token lookup for harvest examples
- `routers/agents.py:42` — graph construction helper

Two usage patterns:
1. **Span construction** (`[token_strings[t] for t in token_ids]`): Replace with `app_tokenizer.get_spans(token_ids)`
2. **Single token lookup** (`token_strings[tid]`): Replace with `app_tokenizer.get_tok_display(tid)`
3. **Full vocab iteration** (`for tid, tstr in token_strings.items()`): Replace with list comprehension over `range(tokenizer.vocab_size)` using `get_tok_display`, used in `graphs.py:284` for the token info endpoint

### 1.5 Update `TokenDropdown.svelte` (frontend)

- Remove the `formatTokenDisplay` function that hardcodes `##` prefix logic (lines 14-19)
- Display tokens as-is from the backend (backend now returns correct display strings)

### 1.6 Delete `build_token_lookup()` and `_PUNCT_NO_SPACE` from `utils.py`

### 1.7 Tests

- Unit test `AppTokenizer.get_spans` for GPT-2 and SimpleStories tokenizers
- Test the overlap-dedup logic with a multi-byte unicode string
- Test fallback path (non-round-trippable tokenizer)

---

## Phase 2: `ModelAdapter` — Backend Topology

Remove hardcoded layer names, block counting, and module access patterns from `compute.py`.

### 2.1 Create `spd/app/backend/model_adapter.py`

```python
@dataclass
class ModulePosition:
    block: int | None     # None for embed/unembed pseudo-layers
    role: str             # "q_proj", "c_fc", "embed", "unembed", "output"
    group: str | None     # "attn", "mlp", None

@dataclass
class ModelAdapter:
    target_module_paths: list[str]   # in execution order (from ComponentModel)
    embedding_path: str              # e.g. "wte", "model.embed_tokens"
    unembed_path: str | None         # e.g. "lm_head" (None if not decomposed)
    role_order: list[str]            # deduped roles in execution order
    role_groups: dict[str, list[str]] # e.g. {"qkv": ["q_proj", "k_proj", "v_proj"]}
    cross_seq_roles: tuple[set[str], set[str]]  # (kv_roles, o_roles) for attention detection
    display_names: dict[str, str]    # e.g. {"lm_head": "W_U"}

    def parse_module_path(self, path: str) -> ModulePosition
    def is_cross_seq_pair(self, source: str, target: str) -> bool
    def get_unembed_weight(self, model: nn.Module) -> Tensor
    def get_embedding_module(self, model: nn.Module) -> nn.Module
```

### 2.2 `build_model_adapter()` factory

Auto-derives most fields from `ComponentModel` + `Config`:
- `target_module_paths` from `model.target_module_paths`
- `parse_module_path`: first integer in dotted path = block, last segment = role, second-to-last = group
- `embedding_path`: find first `nn.Embedding` in `model.target_model.named_modules()`
- `unembed_path`: check if `"lm_head"` or similar in `target_module_paths`; fallback to last `nn.Linear` with `out_features == vocab_size`
- `role_order`: deduplicated roles from execution-ordered paths
- `role_groups`: detect via config field (new `Config` field) or default to `{"qkv": ["q_proj", "k_proj", "v_proj"]}` if those roles exist
- `cross_seq_roles`: from config field (new) or default heuristic (roles containing "k_proj"/"v_proj" + "o_proj")

### 2.3 Add optional config fields to `spd/configs.py`

```python
# In Config or a new AppConfig section:
attn_kv_roles: set[str] | None = None   # e.g. {"k_proj", "v_proj"}, auto-detected if None
attn_out_roles: set[str] | None = None  # e.g. {"o_proj"}, auto-detected if None
role_groups: dict[str, list[str]] | None = None  # visual grouping, auto-detected if None
```

These are optional — auto-detection works for standard transformer naming. Config overrides for exotic architectures.

### 2.4 Refactor `get_sources_by_target()` (compute.py:129-227)

Currently hardcodes layer list construction (lines 186-203). Replace:

**Before:**
```python
layers = ["wte"]
component_layer_names = ["attn.q_proj", "attn.k_proj", ...]
n_blocks = get_model_n_blocks(model.target_model)
for i in range(n_blocks):
    layers.extend([f"h.{i}.{name}" for name in component_layer_names])
```

**After:**
```python
layers = [adapter.embedding_path] + list(model.target_module_paths)
if adapter.unembed_path and adapter.unembed_path not in layers:
    layers.append(adapter.unembed_path)
layers.append("output")
```

### 2.5 Refactor `is_kv_to_o_pair()` (compute.py:112-126)

Replace string matching with `adapter.is_cross_seq_pair(source, target)`.

### 2.6 Refactor embedding hook registration (compute.py:169-170, 344-346)

Replace `model.target_model.wte` with `adapter.get_embedding_module(model.target_model)`.

### 2.7 Refactor `_get_w_unembed()` (dataset_attributions.py:86-90)

Replace `model.target_model.lm_head` with `adapter.get_unembed_weight(model.target_model)`.

### 2.8 Delete `get_model_n_blocks()` (compute.py:764-778)

No longer needed — layer enumeration comes from `model.target_module_paths`.

### 2.9 Update `RunState`

Add `adapter: ModelAdapter`. Thread it through to all callsites that currently hardcode topology.

### 2.10 Tests

- Test `build_model_adapter()` with a mock ComponentModel using Llama-style paths
- Test `build_model_adapter()` with GPT-2 style paths (`transformer.h.0.attn.c_attn`)
- Test `is_cross_seq_pair()` logic
- Test `parse_module_path()` on various naming conventions

---

## Phase 3: Frontend Generalization

Replace hardcoded `parseLayer`, `ROW_ORDER`, `QKV_SUBTYPES` with data from the backend.

### 3.1 New API endpoint: `GET /api/model_info`

Returns model topology for frontend layout:

```json
{
  "role_order": ["embed", "q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj", "unembed", "output"],
  "role_groups": {"qkv": ["q_proj", "k_proj", "v_proj"]},
  "display_names": {"lm_head": "W_U"},
  "module_paths": ["wte", "h.0.attn.q_proj", ...]
}
```

### 3.2 Refactor `parseLayer()` in `PromptAttributionsGraph.svelte` (line 112) and `InterventionsView.svelte` (line 213)

Replace hardcoded regex with a generic parser:
- Extract block index as first integer in the path (or -1/Infinity for embed/output)
- Extract role as last segment
- Extract group from the segment before the role

### 3.3 Replace `ROW_ORDER` and `QKV_SUBTYPES` constants

- Fetch from `/api/model_info` on run load
- Store in run context (already exists: `RunContext`)
- Use dynamic arrays instead of constants

This affects:
- `PromptAttributionsGraph.svelte:41-42` (constants)
- `PromptAttributionsGraph.svelte:128-133` (`getRowKey`)
- `PromptAttributionsGraph.svelte:135-142` (`getRowLabel`)
- `PromptAttributionsGraph.svelte:211-212` (sorting)
- `PromptAttributionsGraph.svelte:249,285-299` (QKV row merging)
- `InterventionsView.svelte:37-38,213-227,285,299,354-367` (same pattern, duplicated)

### 3.4 Extract shared graph layout logic

`parseLayer`, `getRowKey`, `getRowLabel`, `ROW_ORDER`, `QKV_SUBTYPES` are duplicated between `PromptAttributionsGraph.svelte` and `InterventionsView.svelte`. Extract into a shared module (e.g. `graphLayout.ts`) that takes `model_info` as a parameter.

### 3.5 Update `TokenDropdown.svelte` `formatTokenDisplay`

Remove the hardcoded `##` prefix logic (lines 14-19). Backend will send correct display strings.

### 3.6 Update `LAYER_DISPLAY_NAMES` in `promptAttributionsTypes.ts:203-205`

Replace hardcoded `{ lm_head: "W_U" }` with dynamic `display_names` from model info.

### 3.7 Update `NON_INTERVENTABLE_LAYERS` in `promptAttributionsTypes.ts:223`

Currently hardcoded `["wte", "output"]`. These should come from the model info
(the pseudo-layers that aren't decomposed components).

---

## Phase 4: Dataset Search Generalization

### 4.1 Refactor `dataset_search.py`

- Read dataset name and text column from `loaded.config.dataset_config` (already has `name` and `column_name` fields)
- Remove hardcoded `"lennart-finke/SimpleStories"` (line 78)
- Replace `x["story"]` with `x[text_column]` (line 85)
- Remove `topic`/`theme` metadata hardcoding (lines 97-98) — make metadata columns dynamic
- Generalize `DatasetSearchResult` schema: replace `story`/`topic`/`theme` with `text` + `metadata: dict[str, str]`

### 4.2 Frontend labels

- Replace "Search SimpleStories Dataset" text in `DatasetSearchTab.svelte:47,99` with dynamic dataset name from run context

---

## Phase 5: Cleanup

### 5.1 Update `spd/app/CLAUDE.md`

Document `AppTokenizer` and `ModelAdapter` abstractions.

### 5.2 Update `spd/data.py:230`

Remove the `to_lower = "SimpleStories" in dataset_config.name` heuristic. If lowercasing is needed,
it should be a config field, not a name check.

---

## Execution Order

Phases are mostly independent but natural ordering is:

1. **Phase 1** (AppTokenizer) — self-contained, no frontend changes except TokenDropdown
2. **Phase 2** (ModelAdapter backend) — self-contained backend refactor
3. **Phase 3** (Frontend) — depends on Phase 2 for the new `/api/model_info` endpoint
4. **Phase 4** (Dataset) — independent, can be done anytime
5. **Phase 5** (Cleanup) — last

Phases 1, 2, and 4 can be done in parallel. Phase 3 depends on Phase 2.

## What's NOT Changing

- `ComponentModel`, `Components`, CI computation — all model-agnostic already
- Harvest pipeline — already keyed by run_id
- Autointerp pipeline — excluded per request
- Loss functions, metrics, figures — not app concerns
- Config system — mostly model-agnostic, just adding optional fields
