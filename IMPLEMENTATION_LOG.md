# Implementation Log

## 2026-02-13: Generic autointerp interface for cross-method comparison

### Context

Oli is coordinating with Bart on autointerp for the paper. Currently the autointerp pipeline
(`spd/autointerp/`) is tightly coupled to SPD (CI values, `SPDRunInfo`, layer-indexed components).
We need a shared interface that works for SPD, CLTs, MOLTs, and transcoders.

### Decision: `spd/autointerp_generic/` — slim parallel module

Create a new `spd/autointerp_generic/` module that accepts a generic data interface and runs
interpret + evals. No harvest DB, no SPD-specific loading, no correlations/stats.

The existing `spd/autointerp/` stays as-is for now. Once the generic module is proven,
`spd/autointerp` can be migrated to use it under the hood.

### Agreed interface

```python
@dataclass
class ActivatingExample:
    """A token sequence that activates a component of some decomposition strategy."""
    tokens: list[int]
    bold: list[bool]  # caller binarises; leave open for 0-1 float later

@dataclass
class ComponentAutointerpData:
    key: str  # unique within method, free-form (e.g. "attn_0:42", "1:mlp->2:mlp:100")
    component_explanation: str  # e.g. "a rank-one component of the layer 2 mlp up projection"
    activating_examples: list[ActivatingExample]

@dataclass
class DecompositionAutointerpData:
    method: str  # e.g. "spd", "clt", "molt", "transcoder"
    decomposition_explanation: str
    components: list[ComponentAutointerpData]
    tokenizer: AppTokenizer  # instance, not name
```

### Key decisions

- **Binarisation at boundary**: Caller converts method-specific activation values to `bold: list[bool]`.
  Persisted data keeps raw values; binarisation happens at runtime in the mapper. Avoids
  cross-method scale issues (CI vs SAE latent activation vs MOLT transform activation).
- **Component key**: `(method, key)` tuple semantics. `method` lives on the top-level data class,
  `key` is per-component free-form string with method-dependent schema.
- **Tokenizer**: Accept `AppTokenizer` instance directly.
- **DensityIndex** (intruder eval): Works with binarised bold — just `sum(bold)/len(bold)`.

### Future idea: abstract harvest over methods too

The harvest pipeline (`spd/harvest/`) could also be made generic. The only method-specific part
is `batch -> {component_key: per_token_activations}` (for SPD: running the CI function).
Everything else — collecting top examples, computing token PMI, correlations, stats — is the same
regardless of decomposition method.

This would be a major refactor but would mean Bart (and others) only need to provide a single
activation function, and they get the full harvest + autointerp + evals pipeline for free.

Parking this for now — do the narrow `autointerp_generic` first.

### Status: DONE

Module created at `spd/autointerp_generic/`. Passes `make check` (pyright + ruff).

Files:
- `types.py` — data classes + configs
- `db.py` — `GenericInterpDB` (interpretations + scores in one SQLite DB)
- `prompt.py` — generic prompt builder (no PMI/token stats, just examples + explanations)
- `interpret.py` — `run_interpret()` pipeline
- `intruder.py` — `run_intruder_scoring()` (label-free decomposition quality)
- `detection.py` — `run_detection_scoring()` (label predictiveness)
- `fuzzing.py` — `run_fuzzing_scoring()` (label specificity)

Reuses `spd.autointerp.llm_api`, `spd.app.backend.app_tokenizer`, `spd.app.backend.utils.delimit_tokens`.
