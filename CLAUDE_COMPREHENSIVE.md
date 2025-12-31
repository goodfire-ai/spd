# CLAUDE_COMPREHENSIVE.md - Complete Development Guide for SPD

This guide covers everything needed to understand, develop, and contribute to the SPD (Stochastic Parameter Decomposition) codebase.

## 1. Introduction

For AI assistants and developers. Covers:
- Environment setup and project structure
- Development philosophy and coding standards
- Architecture patterns and design principles
- Common workflows and usage patterns
- Testing, deployment, and collaboration practices

### How to Use This Guide

**Two Documents:**
- **CLAUDE_COMPREHENSIVE.md** (this document) - Complete reference for understanding the codebase, architecture, and development practices. Read this to learn how the project works.
- **CLAUDE_CHECKLIST.md** - Pre-submission checklist for verifying your code changes meet SPD standards. Use this before committing to ensure your work follows all conventions.

**Workflow:** Read the comprehensive guide to understand context, then use the checklist to verify your changes before submission.

## 2. Environment Setup & Quick Start

**IMPORTANT**: Always activate the virtual environment before running Python or git operations:
```bash
source .venv/bin/activate
```

**Installation:**
```bash
make install-dev    # Install with dev dependencies and pre-commit hooks
make install        # Install package only (pip install -e .)
```

**Environment:**
- `.env` file with WandB credentials (see `.env.example`)
- WandB for experiment tracking and model storage
- Runs generate timestamped output directories (configs, models, plots)

## 3. Project Overview

SPD is a research framework for analyzing neural network components through sparse parameter decomposition. Supports experimental domains:
- **TMS** (Toy Model of Superposition)
- **ResidualMLP** (residual MLP analysis)
- **Language Models**
- **Identity Insertion**

### Available Experiments

Defined in `spd/registry.py`:

- `tms_5-2`, `tms_5-2-id` - TMS: 5 features, 2 hidden dims (id = fixed identity in-between)
- `tms_40-10`, `tms_40-10-id` - TMS: 40 features, 10 hidden dims
- `resid_mlp1`, `resid_mlp2`, `resid_mlp3` - ResidualMLP: 1-3 layers
- `ss_emb` - Language models (from HuggingFace)

### Research Papers

**Stochastic Parameter Decomposition (SPD)**
- [`papers/Stochastic_Parameter_Decomposition/spd_paper.md`](papers/Stochastic_Parameter_Decomposition/spd_paper.md)
- Introduces core SPD framework, stochastic masking, and optimization techniques
- Note: Development has continued beyond the paper implementation

**Attribution-based Parameter Decomposition (APD)**
- [`papers/Attribution_based_Parameter_Decomposition/apd_paper.md`](papers/Attribution_based_Parameter_Decomposition/apd_paper.md)
- Precursor to SPD, first linear parameter decomposition
- High-level conceptual insights and theoretical foundations

### Key Data Flow

1. Experiments load pretrained target models via WandB or local paths
2. Target models are wrapped in ComponentModel with specified target modules
3. SPD optimization runs via `spd.run_spd.optimize()` with config-driven loss combination
4. Results include component masks, causal importance scores, and visualizations

### Component Analysis

- Components = sparse decompositions of model parameters
- Stochastic masking enables differentiable sparsity
- Causal importance quantifies contributions
- Loss terms balance faithfulness, reconstruction, sparsity

## 4. Development Philosophy & Principles

### Core Principles (TLDR)

1. **Simplicity First** - Code for researchers with varying experience. Prioritize simple, straightforward code.

2. **Type Safety** - Use types, einops, jaxtyping, liberal assertions, Pydantic validation, strict pyright.

3. **Fail Fast** - Code fails immediately when wrong, not silently. Liberal assertions, clear errors, explicit types.

4. **Documentation** - Comments for complex logic only. Skip obvious comments.

5. **Modularity** - Registry pattern, abstract interfaces, protocols. Decouple metrics from core.

6. **Reproducibility** - Centralized configs, seed management, WandB tracking.

7. **Performance** - Distributed training, parallel testing, optimized CI/CD.

8. **Maintainability** - Consistent naming, clear architecture, comprehensive tooling.

## 5. Development Workflow & Commands

**Package Manager:** uv (NOT pip/poetry)

### Make Targets

```bash
make install          # Install package only
make install-dev      # Install with dev deps and pre-commit hooks
make check           # Run full pre-commit suite (format + type check)
make format          # Ruff lint + format
make type            # BasedPyright type checking
make test            # Run tests (excluding slow tests)
make test-all        # Run all tests including slow ones
make coverage        # Generate coverage reports
```

### Pre-commit Hooks

Automatically run `make format` and `make type` before commits (install with `make install-dev`)

### CI/CD Pipeline (GitHub Actions)

1. Checkout code
2. Set up Python 3.13 via uv
3. Install dependencies with CPU-only PyTorch
4. Run basedpyright type checking
5. Run ruff lint and format
6. Run pytest with parallel execution (max 4 workers)

**Special CI install:**
```bash
make install-ci  # Uses CPU wheels, unsafe-best-match index strategy
```

## 6. Code Style & Formatting

### Naming Conventions

- **Files & modules**: `snake_case.py` (e.g., `component_model.py`)
- **Functions & variables**: `snake_case` (e.g., `create_data_loader()`)
- **Classes**: `PascalCase` (e.g., `ComponentModel`)
- **Constants**: `UPPERCASE_WITH_UNDERSCORES` (e.g., `REPO_ROOT`)
- **Private functions**: Prefix with underscore (e.g., `_infer_backend()`)
- **Abbreviations**: Uppercase in variables (e.g., `CI`, `L0`, `KL`)

### Formatting Rules

- **Line length**: 100 characters (strict, enforced by ruff)
- **Formatter**: ruff (configured in pyproject.toml)
- **Import organization**: stdlib → third-party → local
- **Import sorting**: Handled by ruff/isort

**Ruff Configuration:**
- Enabled rules: pycodestyle (E), Pyflakes (F), pyupgrade (UP), flake8-bugbear (B), flake8-simplify (SIM), isort (I)
- Ignored: F722 (jaxtyping incompatibility), E731 (lambda functions allowed), E501 (long lines)

## 7. Type Annotations

### Core Principles

- Use **jaxtyping** for tensor shapes: `Float[Tensor, "... C d_in"]` (runtime checking not yet enabled)
- Use **PEP 604 union syntax**: `str | None` (NOT `Optional[str]`)
- Use **lowercase generic types**: `dict`, `list`, `tuple` (NOT `Dict`, `List`, `Tuple`)
- **Don't annotate when redundant**: `my_thing = Thing()` not `my_thing: Thing = Thing()`, or `name = "John"` not `name: str = "John"`

### Examples

```python
# Good - jaxtyping with explicit dimensions
def forward(self, x: Float[Tensor, "... C d_in"]) -> Float[Tensor, "... C d_out"]:
    return einops.einsum(x, self.W, "... C d_in, C d_in d_out -> ... C d_out") + self.b

# Good - PEP 604 union syntax
def load_model(path: str | Path) -> Model | None:
    pass

# Bad - old style
from typing import Optional, Dict
def load_model(path: Optional[str]) -> Dict[str, Any]:
    pass
```

### Type Checking

- Uses **basedpyright** (NOT mypy) - forked pyright for better performance
- Strict mode enabled: `strictListInference`, `strictDictionaryInference`, `strictSetInference`
- Reports: `MissingTypeArgument`, `UnknownParameterType`, `IncompatibleMethodOverride`, `ImportCycles`
- Excluded: `wandb` directory, third-party code, frontend
- Run with `make type`

## 8. Documentation & Comments

### Philosophy: Don't Write Obvious Comments

Your first instinct should be: **"If I couldn't write any comments, how would I write this code?"**

If code is self-explanatory, skip the comment. Only comment to explain complex logic, focusing on **"why" not "what"**.

If you find it helps you develop, you can write whatever comments you like when developing, so long as you remember to come back and fix them later. 

### Bad (Obvious):
```python
# get dataloader
dataloader = get_dataloader(config)
```

### Good (Explains Complex Logic):
```python
# We need to mask out future positions for causal attention
# Upper triangular matrix excludes the diagonal (hence k=1)
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
```

### Docstring Format

Use **Google-style** with `Args:`, `Returns:`, `Raises:` sections. Single-line for simple functions, multi-line for complex. Focus on non-obvious information.

```python
def tokenize_and_concatenate(dataset: Dataset, tokenizer: PreTrainedTokenizer, ...) -> Dataset:
    """Tokenize and concatenate a dataset of text.

    Args:
        dataset: HuggingFace dataset to tokenize
        tokenizer: Pretrained tokenizer to use
        ...

    Returns:
        Tokenized and concatenated dataset
    """
```

## 9. Architecture & Design Patterns

### Core Pattern: Wrapper + Registry + Config

1. **ComponentModel**: Wraps PyTorch models and injects components
2. **Registry** (`registry.py`): Centralized experiment configuration
3. **Config System** (Pydantic): Type-safe config loading/validation

### Design Principle: Decouple Metrics from Core

Metric and figure code encapsulated in `spd/metrics.py` and `spd/figures.py`.

### Key Design Patterns

**1. Abstract Base Classes for Interfaces**
```python
class LoadableModule(nn.Module, ABC):
    @classmethod
    @abstractmethod
    def from_pretrained(cls, _path: ModelPath) -> "LoadableModule":
        raise NotImplementedError("Subclasses must implement from_pretrained method.")
```

**2. Protocol-Based Design**
```python
class Metric(Protocol):
    slow: ClassVar[bool] = False
    metric_section: ClassVar[str]

    def update(...) -> None: ...
    def compute(self) -> Any: ...
```

**3. Dataclass-Based Configuration**
```python
@dataclass
class ExperimentConfig:
    task_name: TaskName
    decomp_script: Path
    config_path: Path
    expected_runtime: int
    canonical_run: str | None = None
```

**4. Pydantic for Runtime Validation**
```python
class BaseConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_file(cls, path: Path | str) -> Self:
        """Load config from path to a JSON or YAML file."""
```

### Core Architecture Components

- `spd/run_spd.py` - Main SPD optimization logic
- `spd/configs.py` - Pydantic config classes
- `spd/registry.py` - Centralized experiment registry
- `spd/models/component_model.py` - ComponentModel wrapper
- `spd/models/components.py` - Component types (Linear, Embedding, etc.)
- `spd/losses.py` - Loss functions (faithfulness, reconstruction, importance minimality)
- `spd/metrics.py` - Metrics (CI-L0, KL divergence, etc.)
- `spd/figures.py` - Figures (CI histograms, Identity plots, etc.)

## 10. Project Structure

```
spd/
├── spd/                      # Main package
│   ├── models/               # Core model classes
│   ├── metrics/              # Metric implementations
│   ├── utils/                # Utilities (distributed, logging, data)
│   ├── experiments/          # Experiment implementations
│   │   ├── tms/              # Toy Model of Superposition
│   │   ├── resid_mlp/        # Residual MLP
│   │   ├── lm/               # Language models
│   │   └── ih/               # Identity insertion
│   ├── app/                  # Streamlit application
│   │   ├── backend/
│   │   └── frontend/
│   ├── scripts/              # CLI entry points
│   └── [core modules]
├── tests/                    # Test suite
│   ├── metrics/              # Metric tests
│   ├── scripts_run/          # Integration tests
│   └── [unit tests]
├── papers/                   # Research papers (markdown)
├── typings/                  # Type stubs
└── [configuration files]
```

### Organizational Principles

- **Flat within experiments**: Each has `models.py`, `configs.py`, `{task}_decomposition.py`, `train_*.py`, `*_config.yaml`, `plotting.py`
- **Centralized registry**: `spd/registry.py` manages experiment configs
- **Clear separation**: Core logic vs metrics vs experiments
- **Modular metrics**: Each metric in its own file

## 11. Configuration System

### Multi-layered Configuration

1. **YAML config files** define experiment parameters
2. **Pydantic config classes** provide type safety and validation
3. **Environment variables** can override runtime settings
4. **Nested config objects** for task-specific configs

### Key Conventions

- Paths: relative to repo root or `"wandb:"` prefix for WandB paths
- Configs **immutable** (`frozen=True`) and **forbid extra fields** (`extra="forbid"`)
- `ModelPath` type validates and normalizes paths automatically
- Pydantic validators handle deprecated keys and path resolution

### Example Config

```yaml
wandb_project: spd
seed: 0
C: 1200
n_mask_samples: 1
ci_fn_type: "shared_mlp"
ci_fn_hidden_dims: [1000]
loss_metric_configs:
  - classname: "ImportanceMinimalityLoss"
    coeff: 0.004
    pnorm: 2.0
```


## 12. Error Handling & Fail Fast

### Fail-Fast Philosophy (Negative Space Programming)

Code should fail immediately when assumptions are violated, preventing bugs from propagating.

### Assertions 

**If there's an assumption you're making while writing code, assert it:**
- If you were right, then it won't matter. If you were wrong, then the code **should** fail

```python
assert component_params, "component_params is empty"
assert x.shape[-1] == 1, "Last dimension should be 1 after the final layer"
assert cfg.coeff is not None, "All loss metric configs must have a coeff"
```

### Explicit Error Types

```python
raise ValueError(f"Only (.json, .yaml, .yml) files are supported, got {path}")
raise NotImplementedError("Subclasses must implement from_pretrained method.")
raise RuntimeError("Embedding modules not supported for identity insertion")
```

### Try-Except for Expected Errors

```python
try:
    return path.relative_to(REPO_ROOT)
except ValueError:
    # If the path is not relative to REPO_ROOT, return the original path
    return path
```

## 13. Tensor Operations

### Use Einops for Clarity

- Try to use **einops** by default for clarity over raw einsum
- **Assert shapes liberally**
- **Document complex tensor manipulations**

**Example:**
```python
# Preferred - clear dimensions
result = einops.einsum(x, self.W, "... C d_in, C d_in d_out -> ... C d_out") + self.b

# Also good - assert shapes
assert x.shape[-1] == d_in, f"Expected last dim to be {d_in}, got {x.shape[-1]}"
```

## 14. Testing Strategy

### Testing Philosophy

Tests ensure code works as expected, not for production (no deployment). Focus on unit tests for core functionality. Don't worry about integration/end-to-end tests - too much overhead for research code. Interactive use catches issues at low cost.

**Framework:** pytest with pytest-xdist for parallel execution

### Test Organization

- **Test files**: `test_*.py`
- **Test functions**: `def test_*():` with descriptive names
- **Tests mirror source structure**: `tests/metrics/`, `tests/scripts_run/`
- **Fixtures centralized** in `conftest.py` and `metrics/fixtures.py`

### Test Markers

- `@pytest.mark.slow` - Excluded by default, run with `make test-all`
- `@pytest.mark.requires_wandb` - Tests requiring WandB access

## 15. Logging

Use `spd.log.logger` with special methods: `.info()`, `.warning()`, `.error()` (standard), `.values()` (dict of metrics), `.section()` (visual separator), `.set_format()` (swap formatter).

```python
from spd.log import logger
logger.values({"loss": 0.42}, msg="Training metrics")
logger.section("Evaluation Phase")
```

**Config:** Console (INFO), File (WARNING → `logs/logs.log`), named "spd"

## 16. Common Usage Patterns

### Running SPD Experiments

Use `spd-run` command:

```bash
spd-run --experiments tms_5-2                    # Specific experiment
spd-run --experiments tms_5-2,resid_mlp1         # Multiple experiments
spd-run                                          # All experiments
```

Or run directly:
```bash
uv run spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_5-2_config.yaml
```

Outputs: losses and figure paths for analysis.

### Metrics and Figures

Defined in `spd/metrics.py` and `spd/figures.py` as dictionaries of functions. Select and parameterize in experiment configs for easy extension without modifying core framework.

### Running Sweeps

Run hyperparameter sweeps using WandB on the GPU cluster:

```bash
spd-run --experiments <experiment_name> --sweep --n-agents <n-agents> [--cpu] [--job_suffix <suffix>]
```

**Examples:**
```bash
spd-run --experiments tms_5-2 --sweep --n-agents 4            # Run TMS 5-2 sweep with 4 GPU agents
spd-run --experiments resid_mlp2 --sweep --n-agents 3 --cpu   # Run ResidualMLP2 sweep with 3 CPU agents
spd-run --sweep --n-agents 10                                 # Sweep all experiments with 10 agents
spd-run --experiments tms_5-2 --sweep custom.yaml --n-agents 2 # Use custom sweep params file
```

**How it works:** Creates WandB sweep from `spd/scripts/sweep_params.yaml` (or custom), deploys SLURM agents (GPU by default, `--cpu` for CPU), git snapshot for consistency.

**Sweep parameters:** Load from `sweep_params.yaml` or custom file. Supports global and experiment-specific configs:

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

### Evaluation Runs

Run with default hyperparameters:

```bash
spd-run                                                    # All experiments
spd-run --experiments tms_5-2-id,resid_mlp2,resid_mlp3     # Specific experiments
```

Multiple experiments without `--sweep` creates W&B report with aggregated visualizations.

### Additional Options

```bash
spd-run --project my-project                 # Use custom W&B project
spd-run --job_suffix test                    # Add suffix to SLURM job names
spd-run --no-create_report                   # Skip W&B report creation
```

### Cluster Usage Guidelines

**IMPORTANT:**
- **DO NOT use more than 8 GPUs at one time**
- This includes not setting off multiple sweeps/evals that total >8 GPUs
- Monitor jobs with: `squeue --format="%.18i %.9P %.15j %.12u %.12T %.10M %.9l %.6D %b %R" --me`

## 17. Distributed Training

### DistributedState Management

```python
@dataclass(frozen=True, slots=True)
class DistributedState:
    rank: int
    world_size: int
    local_rank: int
    backend: Literal["nccl", "gloo"]
```

### Conventions

- **MPI-based** rank initialization
- **NCCL backend** for GPU, **gloo** for CPU
- Utilities in `spd/utils/distributed.py`: gradient sync, metric averaging, device detection
- `torch.nn.parallel.DistributedDataParallel` for multi-GPU training

## 18. Git & Pull Request Workflow

### Branch Naming

- `refactor/X` - Refactoring work
- `feature/Y` - New features
- `fix/Z` - Bug fixes

### Using GitHub CLI

- To view issues and PRs: `gh issue view 28` or `gh pr view 30`
- Use the PR template defined in `.github/pull_request_template.md`
- Important: You should almost never use --no-verify. The pre-commit checks are there for a reason. 

### PR Checklist

- Review every line of the diff
- All CI checks pass
- Merge latest changes from main branch
- Use "Closes #XX" format for issue linking
- Only commit files that include relevant changes, don't commit all files

### Commit Messages

Explain "what" and "why". Clear, descriptive, focused on relevant changes. Explain purpose, not just the diff.

### PR Template Sections

1. Description - What changed
2. Related Issue - Use "Closes #XX" format
3. Motivation and Context - Why needed
4. Testing - How tested
5. Breaking Changes

## 19. Key Dependencies & Tools

### Core Stack

- **PyTorch** (>=2.6)
- **Transformers** - HuggingFace models and tokenizers
- **WandB** (>=0.20.1) - Optional, disable with `wandb_project=None`
- **Pydantic** (<2.12)
- **jaxtyping** - Type annotations for tensors
- **einops** - Tensor operations (preferred over einsum)
- **Fire** - CLI argument parsing

### Development Tooling

- **ruff** - Linter and formatter (NOT black + flake8 + isort)
- **basedpyright** - Type checker (NOT mypy)
- **pytest + pytest-xdist** - Testing with parallelization
- **uv** - Package manager (NOT pip/poetry)
- **pre-commit** - Git hooks

### Additional Libraries

- **datasets** (>=2.21.0) - HuggingFace data loading
- **streamlit** - Web UI
- **python-dotenv** - Environment variables
- **torchvision** (>=0.23,<0.24)

## 20. Quick Reference

### Key Principles Summary

1. **Simplicity** - Code for researchers with varying experience
2. **Type Safety** - jaxtyping, Pydantic, strict basedpyright
3. **Fail Fast** - Liberal assertions, explicit errors
4. **Minimal Comments** - Complex logic only
5. **Modularity** - Registry pattern, interfaces, protocols
6. **Decouple Metrics** - Separate from core
7. **Reproducibility** - Centralized configs, seeds, WandB
8. **Research Testing** - Unit tests, minimal integration
9. **Clear Architecture** - Wrapper + Registry + Config
10. **Consistent Style** - 100 char, snake_case, PEP 604

### Common Commands Cheatsheet

```bash
# Setup
source .venv/bin/activate
make install-dev

# Development
make check           # Format + type check
make format          # Ruff lint and format
make type            # Type check only
make test            # Run tests (fast)
make test-all        # Run all tests

# Running experiments
spd-run --experiments tms_5-2
spd-run --experiments tms_5-2 --sweep --n-agents 4

# Git/GitHub
gh issue view 28
gh pr view 30
git checkout -b feature/my-feature

# Monitoring cluster
squeue --format="%.18i %.9P %.15j %.12u %.12T %.10M %.9l %.6D %b %R" --me
```

### File Locations Reference

- **Core SPD**: `spd/run_spd.py`, `spd/configs.py`, `spd/registry.py`
- **Models**: `spd/models/component_model.py`, `spd/models/components.py`
- **Metrics**: `spd/metrics.py`, `spd/figures.py`
- **Experiments**: `spd/experiments/{tms,resid_mlp,lm,ih}/`
- **Tests**: `tests/`, `tests/metrics/`, `tests/scripts_run/`
- **Configs**: `spd/experiments/*/\*_config.yaml`
- **Papers**: `papers/Stochastic_Parameter_Decomposition/`, `papers/Attribution_based_Parameter_Decomposition/`
