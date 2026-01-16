# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup
**IMPORTANT**: Always activate the virtual environment before running Python or git operations:
```bash
source .venv/bin/activate
```

## Project Overview
SPD (Stochastic Parameter Decomposition) is a research framework for analyzing neural network components and their interactions through sparse parameter decomposition techniques. The codebase supports three experimental domains: TMS (Toy Model of Superposition), ResidualMLP (residual MLP analysis), and Language Models.

**Available experiments** (defined in `spd/registry.py`):

- **TMS (Toy Model of Superposition)**:
  - `tms_5-2` - TMS with 5 features, 2 hidden dimensions
  - `tms_5-2-id` - TMS with 5 features, 2 hidden dimensions (fixed identity in-between)
  - `tms_40-10` - TMS with 40 features, 10 hidden dimensions
  - `tms_40-10-id` - TMS with 40 features, 10 hidden dimensions (fixed identity in-between)
- **ResidualMLP**:
  - `resid_mlp1` - ResidualMLP with 1 layer
  - `resid_mlp2` - ResidualMLP with 2 layers
  - `resid_mlp3` - ResidualMLP with 3 layers
- **Language Models**:
  - `ss_llama_simple`, `ss_llama_simple-1L`, `ss_llama_simple-2L` - Simple Stories Llama variants
  - `ss_llama_simple_mlp`, `ss_llama_simple_mlp-1L`, `ss_llama_simple_mlp-2L` - Llama MLP-only variants
  - `ss_gpt2`, `ss_gpt2_simple`, `ss_gpt2_simple_noln` - Simple Stories GPT-2 variants
  - `ss_gpt2_simple-1L`, `ss_gpt2_simple-2L` - GPT-2 simple layer variants
  - `gpt2` - Standard GPT-2
  - `ts` - TinyStories

## Research Papers

This repository implements methods from two key research papers on parameter decomposition:

**Stochastic Parameter Decomposition (SPD)**

- [`papers/Stochastic_Parameter_Decomposition/spd_paper.md`](papers/Stochastic_Parameter_Decomposition/spd_paper.md)
- A version of this repository was used to run the experiments in this paper. But we continue to develop on the code, so it no longer is limited to the implementation used for this paper. 
- Introduces the core SPD framework
- Details the stochastic masking approach and optimization techniques used throughout the codebase
- Useful reading for understanding the implementation details, though may be outdated.

**Attribution-based Parameter Decomposition (APD)**

- [`papers/Attribution_based_Parameter_Decomposition/apd_paper.md`](papers/Attribution_based_Parameter_Decomposition/apd_paper.md)
- This paper was the first to introduce the concept of linear parameter decomposition. It's the precursor to SPD.
- Contains **high-level conceptual insights** of parameter decompositions
- Provides theoretical foundations and broader context for parameter decomposition approaches
- Useful for understanding the conceptual framework and motivation behind SPD

## Development Commands

**Setup:**

- `make install-dev` - Install package with dev dependencies and pre-commit hooks
- `make install` - Install package only (`pip install -e .`)
- `make install-app` - Install frontend dependencies (`npm install` in `spd/app/frontend/`)

**Code Quality:**

- `make check` - Run full pre-commit suite (basedpyright, ruff lint, ruff format)
- `make type` - Run basedpyright type checking only
- `make format` - Run ruff linter and formatter

**Frontend (when working on `spd/app/frontend/`):**

- `make check-app` - Run frontend checks (format, type check, lint)
- Or run individually from `spd/app/frontend/`:
  - `npm run format` - Format code with Prettier
  - `npm run check` - Run Svelte type checking
  - `npm run lint` - Run ESLint

**Testing:**

- `make test` - Run tests (excluding slow tests)
- `make test-all` - Run all tests including slow ones
- `python -m pytest tests/test_specific.py` - Run specific test file
- `python -m pytest tests/test_specific.py::test_function` - Run specific test

**Running the App:**

- `make app` - Launch the SPD visualization app (backend + frontend)

## Architecture Overview

**Core SPD Framework:**
- `spd/run_spd.py` - Main SPD optimization logic called by all experiments
- `spd/configs.py` - Pydantic config classes for all experiment types
- `spd/registry.py` - Centralized experiment registry with all experiment configurations
- `spd/models/component_model.py` - Core ComponentModel that wraps target models
- `spd/models/components.py` - Component types (LinearComponent, EmbeddingComponent, etc.)
- `spd/losses.py` - SPD loss functions (faithfulness, reconstruction, importance minimality)
- `spd/metrics.py` - Metrics for logging to WandB (e.g. CI-L0, KL divergence, etc.)
- `spd/figures.py` - Figures for logging to WandB (e.g. CI histograms, Identity plots, etc.)

**Experiment Structure:**

Each experiment (`spd/experiments/{tms,resid_mlp,lm}/`) contains:
- `models.py` - Experiment-specific model classes and pretrained loading
- `*_decomposition.py` - Main SPD execution script
- `train_*.py` - Training script for target models  
- `*_config.yaml` - Configuration files
- `plotting.py` - Visualization utilities

**Key Data Flow:**

1. Experiments load pretrained target models via WandB or local paths
2. Target models are wrapped in ComponentModel with specified target modules
3. SPD optimization runs via `spd.run_spd.optimize()` with config-driven loss combination
4. Results include component masks, causal importance scores, and visualizations

**Configuration System:**

- YAML configs define all experiment parameters
- Pydantic models provide type safety and validation  
- WandB integration for experiment tracking and model storage
- Supports both local paths and `wandb:project/runs/run_id` format for model loading
- Centralized experiment registry (`spd/registry.py`) manages all experiment configurations

**Component Analysis:**

- Components represent sparse decompositions of target model parameters
- Stochastic masking enables differentiable sparsity
- Causal importance quantifies component contributions to model outputs
- Multiple loss terms balance faithfulness, reconstruction quality, and sparsity

**Harvest & Autointerp Modules:**

- `spd/harvest/` - Offline GPU pipeline for collecting component statistics (correlations, token stats, activation examples)
- `spd/autointerp/` - LLM-based automated interpretation of components
- Data stored at `SPD_OUT_DIR/{harvest,autointerp}/<run_id>/`
- See `spd/harvest/CLAUDE.md` and `spd/autointerp/CLAUDE.md` for details

**Output Directory (`SPD_OUT_DIR`):**

- Defined in `spd/settings.py`
- On cluster: `/mnt/polished-lake/spd/`
- Off cluster: `~/spd_out/`
- Contains: runs, SLURM logs, sbatch scripts, clustering outputs, harvest data, autointerp results

**Environment setup:**

- Requires `.env` file with WandB credentials (see `.env.example`)
- Uses WandB for experiment tracking and model storage
- All runs generate timestamped output directories with configs, models, and plots

## Common Usage Patterns

### Running Experiments Locally (`spd-local`)

For collaborators and simple local execution, use `spd-local`:

```bash
spd-local tms_5-2           # Run on single GPU (default)
spd-local tms_5-2 --cpu     # Run on CPU
spd-local tms_5-2 --dp 4    # Run on 4 GPUs (single node DDP)
```

This runs experiments directly without SLURM, git snapshots, or W&B views/reports.

### Web App for Visualization

The SPD app provides interactive visualization of component decompositions and attributions:

```bash
make app                        # Launch backend + frontend dev servers
# or
python -m spd.app.run_app
```

The app has its own detailed documentation in `spd/app/CLAUDE.md` and `spd/app/README.md`.

### Harvesting Component Statistics (`spd-harvest`)

Collect component statistics (activation examples, correlations, token stats) for a run:

```bash
spd-harvest <wandb_path>              # Submit SLURM job to harvest statistics
```

See `spd/harvest/CLAUDE.md` for details.

### Automated Component Interpretation (`spd-autointerp`)

Generate LLM interpretations for harvested components:

```bash
spd-autointerp <wandb_path>            # Submit SLURM job to interpret components
```

Requires `OPENROUTER_API_KEY` env var. See `spd/autointerp/CLAUDE.md` for details.

### Running on SLURM Cluster (`spd-run`)

For the core team, `spd-run` provides full-featured SLURM orchestration:

```bash
spd-run --experiments tms_5-2                    # Run a specific experiment
spd-run --experiments tms_5-2,resid_mlp1         # Run multiple experiments
spd-run                                          # Run all experiments
```

All `spd-run` executions:
- Submit jobs to SLURM
- Create a git snapshot for reproducibility
- Create W&B workspace views

A run will output the important losses and the paths to which important figures are saved. Use these
to analyse the result of the runs.

**Metrics and Figures:**

Metrics and figures are defined in `spd/metrics.py` and `spd/figures.py`.  These files expose dictionaries of functions that can be selected and parameterized in
the config of a given experiment.  This allows for easy extension and customization of metrics and figures, without modifying the core framework code.

### Sweeps

Run hyperparameter sweeps on the GPU cluster:

```bash
spd-run --experiments <experiment_name> --sweep --n_agents <n-agents> [--cpu]
```

Examples:
```bash
spd-run --experiments tms_5-2 --sweep --n_agents 4            # Run TMS 5-2 sweep with 4 GPU agents
spd-run --experiments resid_mlp2 --sweep --n_agents 3 --cpu   # Run ResidualMLP2 sweep with 3 CPU agents
spd-run --sweep --n_agents 10                                 # Sweep all experiments with 10 agents
spd-run --experiments tms_5-2 --sweep custom.yaml --n_agents 2 # Use custom sweep params file
```

**Supported experiments:** All experiments in `spd/registry.py` (run `spd-local --help` to see available options)

**How it works:**

1. Creates a WandB sweep using parameters from `spd/scripts/sweep_params.yaml` (or custom file)
2. Deploys multiple SLURM agents as a job array to run the sweep
3. Each agent runs on a single GPU by default (use `--cpu` for CPU-only)
4. Creates a git snapshot to ensure consistent code across all agents

**Sweep parameters:**

- Default sweep parameters are loaded from `spd/scripts/sweep_params.yaml`
- You can specify a custom sweep parameters file by passing its path to `--sweep`
- Sweep parameters support both experiment-specific and global configurations:
  ```yaml
  # Global parameters applied to all experiments
  global:
    seed:
      values: [0, 1, 2]
    lr_schedule:
      start_val:
        values: [0.001, 0.01]

  # Experiment-specific parameters (override global)
  tms_5-2:
    seed:
      values: [100, 200]  # Overrides global seed
    task_config:
      feature_probability:
        values: [0.05, 0.1]
  ```

**Logs:** logs are found in `~/slurm_logs/slurm-<job_id>_<task_id>.out`

### Cluster Usage Guidelines

- DO NOT use more than 8 GPUs at one time
- This includes not setting off multiple sweeps/evals that total >8 GPUs
- Monitor jobs with: `squeue --format="%.18i %.9P %.15j %.12u %.12T %.10M %.9l %.6D %b %R" --me`

## github
- To view github issues and PRs, use the github cli (e.g. `gh issue view 28` or `gh pr view 30`).
- When making PRs, use the github template defined in `.github/pull_request_template.md`.
- Only commit the files that include the relevant changes, don't commit all files.
- Use branch names `refactor/X` or `feature/Y` or `fix/Z`.
- **Always commit after making code edits** - don't let changes accumulate without committing.

## Coding Guidelines

**This is research code, not production. Prioritize simplicity and fail-fast over defensive programming.**

Core principles:
- **Fail fast** - assert assumptions, crash on violations, don't silently recover
- **No backwards compat** - delete unused code, don't deprecate or add migration shims
- **Narrow types** - avoid `| None` unless null is semantically meaningful; use discriminated unions over bags of optional fields
- **No try/except for control flow** - check preconditions explicitly, then trust them
- **YAGNI** - don't add abstractions, config options, or flexibility for hypothetical futures

```python
# BAD - defensive, recovers silently, wide types
def get_config(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

config = get_config(path)
if config is not None:
    value = config.get("key", "default")

# GOOD - fail fast, narrow types, trust preconditions
def get_config(path: Path) -> Config:
    assert path.exists(), f"config not found: {path}"
    with open(path) as f:
        data = json.load(f)
    return Config(**data)  # pydantic validates

config = get_config(path)
value = config.key
```

More detail in STYLE.md

## Software Engineering Principles

- If you have an invariant in your head, assert it. Are you afraid to assert? sounds like your program might already be broken. Assert, assert, assert. Never soft fail
- never write: `if everythingIsOk: continueHappyPath()`. Instead do `assert everythingIsOk`
- You should have a VERY good reason to handle an error gracefully. If your program isn't working like it should then it shouldn't be running, you should be fixing it
- Write your invariants into types as much as possible.
  - if you either have a and b, or neither, don't make them both independently optional, put them in an optional tuple
- Don't use bare dictionaries for structures whose values aren't homogenous
  - good: { <id>: <val>}
  - bad: {"tokens": …, "loss": …}
- Keep I/O as high up as possible, make as many functions as possible pure.
- Default args are a good idea far less often than they're typically used
- You should have a very good reason for having a default value for an argument, especially if it's caller also defaults to the same thing
- Keep defaults high in the call stack.
- Delete unused code. If an argument is always x, strongly consider removing as an argument and just inlining
- Differentiate no data from empty collections. Often it's important to differentiate `None` from `[]`
- Do not write try catch blocks unless it absolutely makes sense
- Comments hide sloppy code. If you feel the need to write a comment, consider that you should instead
  - name your functions more clearly
  - name your variables more clearly
  - separate a chunk of logic into a function
  - seperate an inlined computation into a meaningfully named variable

Some other notes:

- Please don’t write dialogic / narrativised comments or code. Instead, write comments that describe
  the code as is, not the diff you're making.
  - These are examples of narrativising comments:
    - `# the function now uses y instead of x`
    - `# changed to be faster`
    - `# we now traverse in reverse`
  - Here's an example of a bad diff:
    ```diff
95 -      # Reservoir states
96 -      reservoir_states: list[ReservoirState]
95 +      # Reservoir state (tensor-based)
96 +      reservoir: TensorReservoirState
    ```
    This is bad because the new comment makes reference to a change in code, not just the state of
    the code.
