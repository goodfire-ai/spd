# SPD Development Style Guide

This document contains the coding conventions, architectural patterns, and development practices used in the SPD codebase. It provides detailed guidance on maintaining consistency when working on this project.

## Code Style & Formatting

### Naming Conventions
- **Files & modules**: `snake_case.py` (e.g., `component_model.py`, `general_utils.py`)
- **Functions & variables**: `snake_case` (e.g., `create_data_loader()`, `n_mask_samples`)
- **Classes**: `PascalCase` (e.g., `ComponentModel`, `LoadableModule`, `SPDRunInfo`)
- **Constants**: `UPPERCASE_WITH_UNDERSCORES` (e.g., `REPO_ROOT`, `DEFAULT_LOGFILE`)
- **Private functions**: Prefix with underscore (e.g., `_infer_default_backend()`)
- **Abbreviations**: Use uppercase in variable names (e.g., `CI`, `L0`, `KL`, `MLP`)

### Formatting Rules
- **Line length**: 100 characters (strict, enforced by ruff)
- **Formatter**: ruff (configured in pyproject.toml)
- **Import organization**: stdlib → third-party → local
- **Import sorting**: Handled by ruff/isort

### Type Annotations

**Core Principles:**
- Use **jaxtyping** for tensor shapes with explicit dimensions: `Float[Tensor, "... C d_in"]`
- Use **PEP 604 union syntax**: `str | None` (NOT `Optional[str]`)
- Use **lowercase generic types**: `dict`, `list`, `tuple` (NOT `Dict`, `List`, `Tuple`)
- **Don't annotate when redundant**: `my_thing = Thing()` NOT `my_thing: Thing = Thing()`
- **Avoid redundant annotations**: `name = "John"` NOT `name: str = "John"`

**Examples:**
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

**Type Checking:**
- Uses **basedpyright** (NOT mypy)
- Strict mode enabled for lists, dicts, sets
- Run with `make type`

## Documentation Philosophy

**Core Principle: Don't write obvious comments**

If code is self-explanatory, skip the comment. Only comment to explain complex logic, focusing on "why" not "what".

**Bad (obvious):**
```python
# get dataloader
dataloader = get_dataloader(config)
```

**Good (explains complex logic):**
```python
# We need to mask out future positions for causal attention
# Upper triangular matrix excludes the diagonal (hence k=1)
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
```

**Docstrings:**
- Use Google-style format with `Args:`, `Returns:`, `Raises:` sections
- Single-line docstrings for simple functions
- Multi-line with proper indentation for complex functions

```python
def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    column_name: str,
    max_length: int = 1024,
) -> Dataset:
    """Helper function to tokenize and concatenate a dataset of text.

    Args:
        dataset: HuggingFace dataset to tokenize
        tokenizer: Pretrained tokenizer to use
        column_name: Column containing text to tokenize
        max_length: Maximum sequence length

    Returns:
        Tokenized and concatenated dataset
    """
```

## Architecture & Design Patterns

### Core Pattern: Wrapper + Registry + Config

The SPD framework uses:
1. **ComponentModel**: Wraps arbitrary PyTorch models and injects components
2. **Registry** (`registry.py`): Centralized configuration of all experiments
3. **Config System** (Pydantic): Type-safe configuration loading and validation

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

## Configuration System

**Multi-layered Configuration:**
1. **YAML config files** define experiment parameters
2. **Pydantic config classes** provide type safety and validation
3. **Environment variables** can override runtime settings
4. **Nested config objects** for task-specific configs

**Key Conventions:**
- Paths can be relative to repo root or WandB paths prefixed with `"wandb:"`
- Configs are **immutable after creation**: `frozen=True`
- Extra fields are **forbidden**: `extra="forbid"`
- Pydantic validators handle deprecated keys and path resolution

```yaml
# Example config structure
wandb_project: spd
seed: 0
C: 1200
n_mask_samples: 1
ci_fn_type: "shared_mlp"
loss_metric_configs:
  - classname: "ImportanceMinimalityLoss"
    coeff: 0.004
    pnorm: 2.0
```

## Error Handling

### Fail-Fast Philosophy

Code should fail immediately when assumptions are violated, preventing bugs from propagating.

**Liberal Use of Assertions:**
```python
assert component_params, "component_params is empty"
assert x.shape[-1] == 1, "Last dimension should be 1 after the final layer"
assert cfg.coeff is not None, "All loss metric configs must have a coeff"
```

**Explicit Error Types:**
```python
raise ValueError(f"Only (.json, .yaml, .yml) files are supported, got {path}")
raise NotImplementedError("Subclasses must implement from_pretrained method.")
raise RuntimeError("Embedding modules not supported for identity insertion")
```

**Try-Except for Expected Errors:**
```python
try:
    return path.relative_to(REPO_ROOT)
except ValueError:
    # If the path is not relative to REPO_ROOT, return the original path
    return path
```

## Testing Strategy

**Framework:** pytest with pytest-xdist for parallel execution

**Test Organization:**
- Test files: `test_*.py`
- Test functions: `def test_*():` with descriptive names
- Tests mirror source structure: `tests/metrics/`, `tests/scripts_run/`
- Fixtures centralized in `conftest.py` and `metrics/fixtures.py`

**Markers:**
- `@pytest.mark.slow` - Excluded by default, run with `make test-all`
- `@pytest.mark.requires_wandb` - Tests requiring WandB access

**Example:**
```python
@pytest.mark.slow
def test_gpt_2_decomposition_happy_path() -> None:
    """Test that SPD decomposition works for GPT-2"""
    set_seed(0)
    device = "cpu"

    config = Config(
        wandb_project=None,  # Disable wandb for testing
        # ... rest of config
    )
```

**Testing Philosophy:**
> Focus on unit tests for core functionality. Don't worry about lots of larger integration/end-to-end tests - these often require too much overhead for what it's worth in research code.

## Logging

**Custom Logger:** `spd.log.logger` with special methods

```python
from spd.log import logger

logger.info("Processing batch")
logger.values({"loss": 0.42, "lr": 0.001}, msg="Training metrics")
logger.section("Evaluation Phase")
logger.warning("Deprecated parameter used")
```

**Configuration:**
- Console handler: INFO level
- File handler: WARNING level (logs to `logs/logs.log`)
- Module-level logger named "spd"

## Project Structure

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
│   ├── scripts/              # CLI entry points
│   └── [core modules]
├── tests/                    # Test suite
├── papers/                   # Research papers (markdown)
└── typings/                  # Type stubs
```

**Organizational Principles:**
- **Flat within experiments**: Each has `models.py`, `configs.py`, `{task}_decomposition.py`
- **Centralized registry**: `spd/registry.py` manages experiment configs
- **Clear separation**: Core logic vs metrics vs experiments
- **Modular metrics**: Each metric in its own file

## Development Workflow

**Package Management:** uv (NOT pip/poetry)

**Standard Make Targets:**
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

**Pre-commit Hooks:**
- Automatically run `make format` and `make type` before commits
- Install with `make install-dev`

**CI/CD (GitHub Actions):**
1. Checkout code
2. Set up Python 3.13 via uv
3. Install dependencies with CPU-only PyTorch
4. Run basedpyright type checking
5. Run ruff lint and format
6. Run pytest with parallel execution (max 4 workers)

## Key Dependencies

**Core Stack:**
- **PyTorch** (>=2.6) - Deep learning framework
- **Transformers** - Hugging Face models and tokenizers
- **WandB** (>=0.20.1) - Experiment tracking (optional, disable with `wandb_project=None`)
- **Pydantic** (<2.12) - Configuration validation
- **jaxtyping** - Type annotations for tensors
- **einops** - Tensor operations (preferred over raw einsum for clarity)
- **Fire** - CLI argument parsing

**Tooling:**
- **ruff** - Linter and formatter
- **basedpyright** - Type checker
- **pytest + pytest-xdist** - Testing framework with parallelization
- **uv** - Package manager

## Distributed Training

**DistributedState Management:**
```python
@dataclass(frozen=True, slots=True)
class DistributedState:
    rank: int
    world_size: int
    local_rank: int
    backend: Literal["nccl", "gloo"]
```

**Conventions:**
- MPI-based rank initialization
- NCCL backend for GPU, gloo for CPU
- Gradient synchronization utilities in `spd/utils/distributed.py`
- Metric averaging across processes

## Pull Request Standards

**Branch Naming:**
- `refactor/X` - Refactoring work
- `feature/Y` - New features
- `fix/Z` - Bug fixes

**PR Checklist:**
- Review every line of the diff
- All CI checks pass
- Merge latest changes from dev branch
- Use "Closes #XX" format for issue linking

**Commit Messages:**
- Explain "what" and "why"
- Clear, descriptive, focused on relevant changes

## Summary of Key Principles

1. **Simplicity First** - Code written for researchers with varying coding experience
2. **Type Safety** - Heavy use of type annotations, Pydantic validation, strict pyright checking
3. **Fail Fast** - Assertions are liberal, clear error messages, explicit error types
4. **Documentation** - Comments for complex logic only, not obvious code
5. **Modularity** - Registry pattern for experiments, abstract interfaces, protocol-based design
6. **Reproducibility** - Centralized configs, seed management, WandB tracking
7. **Performance** - Distributed training support, parallel testing, optimized CI/CD
8. **Maintainability** - Consistent naming, clear architecture, comprehensive tooling
