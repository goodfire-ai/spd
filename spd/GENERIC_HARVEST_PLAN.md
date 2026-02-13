# Plan: Generic harvest + autointerp pipeline

Abstract the entire harvest → autointerp → evals pipeline over decomposition methods, so that adding a new method (CLT, MOLT, transcoder, SAE) requires implementing a single function.

## Core insight

The only method-specific part of harvest is computing per-token activations. Everything else — reservoir sampling, token PMI, correlations, activation example collection, autointerp, evals — is identical across methods.

## The one function you'd implement per method

```python
def get_component_activations(
    model: nn.Module,
    batch: dict[str, Tensor],  # tokenizer output
) -> dict[str, Float[Tensor, "seq"]]
    """Map a batch to per-token activation values for each component.

    Returns {component_key: activations} where activations is a 1D tensor
    of per-token activation strength for that component on this input.

    Examples:
        SPD: causal importance values per component
        SAE: latent activation magnitudes per feature
        CLT: encoder activation per cross-layer feature
        Transcoder: latent activation per feature
    """
```

## What becomes shared

Currently SPD-specific, would become generic:

| Step | Current location | What it does | What changes |
|------|-----------------|--------------|--------------|
| Reservoir sampling | `harvest/harvester.py` | Collect top-k activation windows per component | Replace CI tensor with generic activation tensor |
| Token PMI | `harvest/harvester.py` | Input/output token co-occurrence stats | Use `activation > threshold` instead of `ci > threshold` for firing detection |
| Correlations | `harvest/harvester.py` | Co-firing counts between components | Same change — threshold on generic activation |
| Token stats | `harvest/harvester.py` | Precision/recall of tokens given component firing | Same |
| Merge | `harvest/harvest.py` | Combine multi-GPU worker results | No change needed |
| DB storage | `harvest/db.py` | SQLite persistence | Rename `ci_values` → `activation_values` in schema |
| Autointerp | `autointerp/interpret.py` | LLM labeling | Already done in `autointerp_generic/` |
| Evals | `autointerp/scoring/`, `harvest/intruder.py` | Intruder, detection, fuzzing | Already done in `autointerp_generic/` |

## Architecture sketch

```
spd/decomp/                          # new top-level module
├── types.py                         # DecompositionSpec, ActivationFn protocol
├── harvest.py                       # generic harvester (current harvester.py, parametrised)
├── harvest_worker.py                # SLURM worker script
├── merge.py                         # merge multi-GPU results
├── db.py                            # storage (current harvest/db.py, generalised)
├── interp/                          # current autointerp_generic/ (moved here)
│   ├── interpret.py
│   ├── intruder.py
│   ├── detection.py
│   └── fuzzing.py
└── methods/                         # per-method activation functions
    ├── spd.py                       # SPD: CI computation
    ├── sae.py                       # SAE: latent activations
    ├── clt.py                       # CLT: encoder activations
    └── transcoder.py                # transcoder: latent activations
```

### DecompositionSpec

```python
@dataclass
class DecompositionSpec:
    """Everything needed to harvest + interpret a decomposition."""

    method: str
    decomposition_explanation: str
    component_explanations: dict[str, str]  # {key: explanation}
    model: nn.Module
    tokenizer: AppTokenizer
    dataset: Dataset
    activation_fn: ActivationFn
    binarise_threshold: float  # method-specific threshold for bold mask
```

### ActivationFn protocol

```python
class ActivationFn(Protocol):
    def __call__(
        self,
        model: nn.Module,
        batch: dict[str, Tensor],
    ) -> dict[str, Float[Tensor, "seq"]]: ...
```

### Entry point

```python
from spd.decomp import run_pipeline
from spd.decomp.methods.clt import clt_activation_fn

spec = DecompositionSpec(
    method="clt",
    decomposition_explanation="...",
    component_explanations={...},
    model=clt_model,
    tokenizer=app_tok,
    dataset=train_dataset,
    activation_fn=clt_activation_fn,
    binarise_threshold=0.1,
)

# runs harvest → merge → interpret → evals, returns all results
results = run_pipeline(spec, openrouter_api_key="sk-or-...", n_gpus=8)
```

## Migration path

1. **Phase 1 (done)**: `autointerp_generic/` — generic evals, caller provides pre-harvested data
2. **Phase 2**: Extract `Harvester` from `harvest/harvester.py`, parametrise on `ActivationFn` instead of hardcoded CI. Keep SPD-specific code as `methods/spd.py`.
3. **Phase 3**: Wire up SLURM orchestration (adapt `harvest/scripts/run_slurm.py` to accept a `DecompositionSpec`)
4. **Phase 4**: Migrate `spd/harvest/` and `spd/autointerp/` to use the generic pipeline internally. Delete duplicated code.

## Key refactoring decisions

### What stays method-specific
- Model loading (checkpoint formats differ)
- `ActivationFn` implementation
- `binarise_threshold` choice
- `component_explanations` text

### What becomes shared
- Reservoir sampling logic
- Token PMI / correlation accumulation
- Multi-GPU sharding + merge
- All of autointerp (interpret + evals)
- DB storage + repo pattern
- SLURM orchestration

### Naming: `ci_values` → `activation_values`
Rename throughout. CI is an SPD concept; the generic field is "activation strength" or just "activation". The `bold` mask (binarised) is already method-agnostic.

### Backward compatibility
None. This is research code — just change it. Old harvest data can be regenerated.

## Estimated scope

- Phase 2 is the bulk: ~500 lines of `harvester.py` to parametrise
- Phase 3 is plumbing: adapt SLURM scripts
- Phase 4 is cleanup: delete old code, update imports

The hardest part is ensuring the `Harvester` class works with arbitrary activation functions without losing its GPU optimisations (subsampling, on-GPU accumulation, reservoir sampling).
