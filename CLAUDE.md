# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup
**IMPORTANT**: Always activate the virtual environment before running Python or git operations:
```bash
source .venv/bin/activate
```

## Project Overview
SPD (Stochastic Parameter Decomposition) is a research framework for analyzing neural network components and their interactions through sparse parameter decomposition techniques. The codebase supports three experimental domains: TMS (Toy Model of Superposition), ResidMLP (residual MLP analysis), and Language Models.

**Available experiments** (defined in `spd/registry.py`):
- `tms_5-2` - TMS with 5 features, 2 hidden dimensions
- `tms_5-2-id` - TMS with 5 features, 2 hidden dimensions (fixed identity in-between)
- `tms_40-10` - TMS with 40 features, 10 hidden dimensions  
- `tms_40-10-id` - TMS with 40 features, 10 hidden dimensions (fixed identity in-between)
- `resid_mlp1` - ResidMLP with 1 layer
- `resid_mlp2` - ResidMLP with 2 layers
- `resid_mlp3` - ResidMLP with 3 layers
- `ss_emb` - Language model experiments (loaded from HuggingFace)

## Research Papers

This repository implements methods from two key research papers on parameter decomposition:

**Stochastic Parameter Decomposition (SPD)** - [`papers/Stochastic_Parameter_Decomposition/spd_paper.md`](papers/Stochastic_Parameter_Decomposition/spd_paper.md)
- A version of this repository was used to run the experiments in this paper. But we continue to develop on the code, so it no longer is limited to the implementation used for this paper. 
- Introduces the core SPD framework
- Details the stochastic masking approach and optimization techniques used throughout the codebase
- Useful reading for understanding the implementation details, though may be outdated.

**Attribution-based Parameter Decomposition (APD)** - [`papers/Attribution_based_Parameter_Decomposition/apd_paper.md`](papers/Attribution_based_Parameter_Decomposition/apd_paper.md)
- This paper was the first to introduce the concept of linear parameter decomposition. It's the precursor to SPD.
- Contains **high-level conceptual insights** of parameter decompositions
- Provides theoretical foundations and broader context for parameter decomposition approaches
- Useful for understanding the conceptual framework and motivation behind SPD

## Development Commands

**Setup:**
- `make install-dev` - Install package with dev dependencies and pre-commit hooks
- `make install` - Install package only (`pip install -e .`)

Both installation commands automatically create `spd/user_metrics_and_figs.py` from `spd/user_metrics_and_figs.py.example` if it doesn't exist.

**Code Quality:**
- `make check` - Run full pre-commit suite (basedpyright, ruff lint, ruff format)
- `make type` - Run basedpyright type checking only
- `make format` - Run ruff linter and formatter

**Testing:**
- `make test` - Run tests (excluding slow tests)
- `make test-all` - Run all tests including slow ones
- `python -m pytest tests/test_specific.py` - Run specific test file
- `python -m pytest tests/test_specific.py::test_function` - Run specific test

## Architecture Overview

**Core SPD Framework:**
- `spd/run_spd.py` - Main SPD optimization logic called by all experiments
- `spd/configs.py` - Pydantic config classes for all experiment types
- `spd/registry.py` - Centralized experiment registry with all experiment configurations
- `spd/models/component_model.py` - Core ComponentModel that wraps target models
- `spd/models/components.py` - Component types (LinearComponent, EmbeddingComponent, etc.)
- `spd/losses.py` - SPD loss functions (faithfulness, reconstruction, importance minimality)
- `spd/metrics.py` - SPD metrics (faithfulness, reconstruction, importance minimality)
- `spd/figures.py` - SPD figures (faithfulness, reconstruction, importance minimality)

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

**Environment setup:**
- Requires `.env` file with WandB credentials (see `.env.example`)
- Uses WandB for experiment tracking and model storage
- All runs generate timestamped output directories with configs, models, and plots

## Common Usage Patterns

**Running SPD experiments:**

The unified `spd-run` command provides a single entry point for running SPD experiments, supporting fixed configurations, parameter sweeps, and evaluation runs:

```bash
spd-run --experiments tms_5-2                    # Run a specific experiment
spd-run --experiments tms_5-2,resid_mlp1         # Run multiple experiments
spd-run                                          # Run all experiments
```

Alternatively, you can run individual experiments directly:
```bash
uv run spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_5-2_config.yaml
uv run spd/experiments/resid_mlp/resid_mlp_decomposition.py spd/experiments/resid_mlp/resid_mlp1_config.yaml
uv run spd/experiments/lm/lm_decomposition.py spd/experiments/lm/ss_emb_config.yaml
```

A run will output the important losses and the paths to which important figures are saved. Use these
to analyse the result of the runs.

**Custom Metrics and Visualizations:**
The `spd/user_metrics_and_figs.py` file (automatically created from template during installation) allows adding custom metrics and visualizations without modifying core framework code. The file provides:
- `compute_user_metrics()` - Define metrics logged to WandB during optimization
- `create_user_figures()` - Create matplotlib figures saved during optimization

Both functions receive the component model, gates, causal importances, and other optimization state, allowing flexible analysis of SPD results.

**Sweeps**
Run hyperparameter sweeps using WandB on the GPU cluster:

```bash
spd-run --experiments <experiment_name> --sweep --n_agents <n_agents> [--cpu] [--job_suffix <suffix>]
```

Examples:
```bash
spd-run --experiments tms_5-2 --sweep --n_agents 4            # Run TMS 5-2 sweep with 4 GPU agents
spd-run --experiments resid_mlp2 --sweep --n_agents 3 --cpu   # Run ResidMLP2 sweep with 3 CPU agents
spd-run --sweep --n_agents 10                                 # Sweep all experiments with 10 agents
spd-run --experiments tms_5-2 --sweep custom.yaml --n_agents 2 # Use custom sweep params file
```

**Supported experiments:** tms_5-2, tms_5-2-id, tms_40-10, tms_40-10-id, resid_mlp1, resid_mlp2, resid_mlp3, ss_emb

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
    lr:
      values: [0.001, 0.01]
  
  # Experiment-specific parameters (override global)
  tms_5-2:
    seed:
      values: [100, 200]  # Overrides global seed
    task_config:
      feature_probability:
        values: [0.05, 0.1]
  ```

**Logs:** Agent logs are found in `~/slurm_logs/slurm-<job_id>_<task_id>.out`

**Evaluation Runs**
To run SPD with default hyperparameters for each experiment:

```bash
spd-run                                                    # Run all experiments
spd-run --experiments tms_5-2-id,resid_mlp2,resid_mlp3     # Run specific experiments
```

When multiple experiments are run without `--sweep`, it creates a W&B report with aggregated visualizations across all experiments.

**Additional Options:**
```bash
spd-run --project my-project                 # Use custom W&B project
spd-run --job_suffix test                    # Add suffix to SLURM job names
spd-run --no-create_report                   # Skip W&B report creation
```

**Cluster Usage Guidelines:**
- DO NOT use more than 8 GPUs at one time
- This includes not setting off multiple sweeps/evals that total >8 GPUs
- Monitor jobs with: `squeue --format="%.18i %.9P %.15j %.12u %.12T %.10M %.9l %.6D %b %R" --me`

## github
- To view github issues and PRs, use the github cli (e.g. `gh issue view 28` or `gh pr view 30`).
- When making PRs, use the github template defined in `.github/pull_request_template.md`.
- Only commit the files that include the relevant changes, don't commit all files.
- Use branch names `refactor/X` or `feature/Y` or `fix/Z`.

## Coding Guidelines
see @STYLE.md