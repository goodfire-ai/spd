# Generic Autointerp Module

Decomposition-agnostic autointerp pipeline. Works with SPD, CLTs, MOLTs, transcoders — anything that can produce activating token windows per component.

## Interface

You provide a `DecompositionAutointerpData` object. The module handles interpretation (LLM labeling) and all three evals (intruder, detection, fuzzing).

```python
from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp_generic import (
    ActivatingExample,
    ComponentAutointerpData,
    DecompositionAutointerpData,
)

data = DecompositionAutointerpData(
    method="clt",  # or "molt", "transcoder", "sae", etc.
    decomposition_explanation=(
        "Cross-layer transcoders decompose transformer computations into "
        "features that span multiple layers. Each feature has an encoder "
        "activation indicating how strongly it fires on a given input."
    ),
    components=[
        ComponentAutointerpData(
            key="feature_42",
            component_explanation="a cross-layer feature spanning layers 2-5",
            activating_examples=[
                ActivatingExample(
                    tokens=[2845, 310, 278, 4802, 338, 2215, 373],
                    bold=[False, False, False, True, False, False, False],
                ),
                # ... more examples (aim for 30-1000 per component)
            ],
        ),
        # ... more components
    ],
    tokenizer=AppTokenizer.from_pretrained("meta-llama/Llama-3.2-1B"),
)
```

### Key fields

- **`method`**: String tag for your decomposition type. Used for grouping in cross-method comparisons.
- **`decomposition_explanation`**: Goes into the LLM prompt as context. Describe what your decomposition does and what "activation" means.
- **`key`**: Unique identifier per component, free-form string. Use whatever scheme makes sense for your method (e.g. `"feature_42"`, `"1.mlp->2.mlp:100"`, `"encoder_layer3:77"`).
- **`component_explanation`**: Per-component context for the LLM (e.g. `"a feature in the layer 2 MLP"`, `"a cross-layer feature spanning layers 2-5"`).
- **`bold`**: Binary mask over the token window marking where the component fires. **You binarise** at whatever threshold makes sense for your activation metric (SAE latent activation, encoder activation, etc.). This avoids cross-method scale issues.

## Running evals

All functions take the `DecompositionAutointerpData` and an OpenRouter API key. Results are persisted to a SQLite DB for resumption.

### Intruder eval (label-free, tests decomposition quality)

```python
from pathlib import Path
from spd.autointerp_generic import IntruderConfig
from spd.autointerp_generic.intruder import run_intruder_scoring

results = run_intruder_scoring(
    data=data,
    openrouter_api_key="sk-or-...",
    db_path=Path("results/clt_interp.db"),
    config=IntruderConfig(),
    limit=100,  # optional: only score first N components
    cost_limit_usd=5.0,  # optional: budget cap
)

for r in results:
    print(f"{r.component_key}: {r.score:.2f} ({len(r.trials)} trials)")
```

The intruder eval doesn't need labels — it tests whether a component's activating examples are coherent by asking an LLM to spot an "intruder" example from a different component. Higher score = more coherent decomposition.

### Interpretation (LLM labeling)

```python
from spd.autointerp_generic import InterpretConfig
from spd.autointerp_generic.interpret import run_interpret

interp_results = run_interpret(
    data=data,
    openrouter_api_key="sk-or-...",
    db_path=Path("results/clt_interp.db"),
    config=InterpretConfig(),
)

labels = {r.component_key: r.label for r in interp_results}
```

### Detection + fuzzing evals (label-based)

These test label quality. Run after interpretation.

```python
from spd.autointerp_generic import EvalConfig
from spd.autointerp_generic.detection import run_detection_scoring
from spd.autointerp_generic.fuzzing import run_fuzzing_scoring
from spd.autointerp_generic.db import GenericInterpDB

# Get labels from the DB (or from run_interpret results above)
db = GenericInterpDB(Path("results/clt_interp.db"), readonly=True)
labels = db.get_labels()
db.close()

eval_config = EvalConfig()

detection_results = run_detection_scoring(
    data=data,
    labels=labels,
    openrouter_api_key="sk-or-...",
    db_path=Path("results/clt_interp.db"),
    config=eval_config,
)

fuzzing_results = run_fuzzing_scoring(
    data=data,
    labels=labels,
    openrouter_api_key="sk-or-...",
    db_path=Path("results/clt_interp.db"),
    config=eval_config,
)
```

## What you need to provide

The only work on your side is building the `DecompositionAutointerpData`. This means:

1. **Load your model/checkpoint** and run forward passes over training data
2. **Collect activating examples**: for each component, gather token windows where it fires strongly (reservoir sampling works well)
3. **Binarise activations**: convert your method-specific activation values to `bold: list[bool]` at a threshold that makes sense for your metric
4. **Write component/decomposition explanations**: short strings giving the LLM context

Everything downstream (LLM calls, prompt formatting, scoring, DB persistence, resumption) is handled by this module.

## Resumption

All `run_*` functions resume automatically. If a run is interrupted, re-run with the same `db_path` and it picks up where it left off. Already-scored components are skipped.

## Config defaults

| Config | Key params | Defaults |
|--------|-----------|----------|
| `InterpretConfig` | `model`, `max_examples`, `label_max_words` | gemini-3-flash, 30 examples, 5 words |
| `IntruderConfig` | `n_real`, `n_trials`, `density_tolerance` | 4 real + 1 intruder, 10 trials, 0.05 tolerance |
| `EvalConfig` | `detection_n_*`, `fuzzing_n_*` | 5 activating + 5 non-activating, 5 trials each |

All configs accept `max_requests_per_minute` for rate limiting and can be serialized to/from YAML via `config.to_file()` / `Config.from_file()`.
