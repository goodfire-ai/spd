# CLAUDE_CHECKLIST.md - Pre-Submission Checklist

Use this checklist before submitting any code changes to ensure your contribution meets SPD repository standards.

As you work through this checklist, you might notice something and then get distracted when fixing it. You need to restart the checklist again after your fixes. You might therefore want to keep a running list of changes to make, then make them, then start the checklist again for all of them. 

## Code Style & Formatting

### Naming
- [ ] **Files & modules**: `snake_case.py`
- [ ] **Functions & variables**: `snake_case`
- [ ] **Classes**: `PascalCase`
- [ ] **Constants**: `UPPERCASE_WITH_UNDERSCORES`
- [ ] **Private functions**: Prefixed with `_`
- [ ] **Abbreviations**: Uppercase (e.g., `CI`, `L0`, `KL`)

### Type Annotations
- [ ] **Used jaxtyping for tensors** - `Float[Tensor, "... C d_in"]` format (runtime checking not yet enabled)
- [ ] **Used PEP 604 unions** - `str | None` NOT `Optional[str]`
- [ ] **Used lowercase generics** - `dict`, `list`, `tuple` NOT `Dict`, `List`, `Tuple`
- [ ] **Avoided redundant annotations** - Don't write `my_thing: Thing = Thing()` or `name: str = "John"`
- [ ] **Type checking passes with no errors** - Run `make type` successfully and fix all issues (uses basedpyright, NOT mypy)

### Comments & Documentation
- [ ] **No obvious comments** - If code is self-explanatory, no comment needed. (Temp comments during development are fine if cleaned up before committing)
- [ ] **Complex logic explained** - Comments focus on "why" not "what"
- [ ] **Google-style docstrings** - Used `Args:`, `Returns:`, `Raises:` sections where needed
- [ ] **Non-obvious information only** - Docstrings don't repeat what's obvious from signature

### Formatting
- [ ] **Ruff formatting applied** - Run `make format` before committing

## Code Quality

### Error Handling (Fail Fast)
- [ ] **Liberal assertions** - Assert all assumptions about data/state
- [ ] **Clear error messages** - Assertions include descriptive messages
- [ ] **Explicit error types** - Use `ValueError`, `NotImplementedError`, `RuntimeError` appropriately
- [ ] **Fail immediately** - Code fails when wrong, doesn't recover silently
- [ ] **Use try-except only for expected errors** - Assertions for invariants/assumptions. Try-except only when errors are expected and handled (e.g., path resolution, file not found)

### Tensor Operations
- [ ] **Used einops by default** - Preferred over raw einsum for clarity
- [ ] **Asserted shapes liberally** - Verify tensor dimensions
- [ ] **Documented complex operations** - Explain non-obvious tensor manipulations

### Design Patterns
- [ ] **Followed existing patterns** - Match architecture style of surrounding code (ABC for interfaces, Protocol for metrics, Pydantic for configs)
- [ ] **Metrics decoupled** - Each metric in its own file within `spd/metrics/` directory. Figures in `spd/figures.py`
- [ ] **Used Pydantic for configs** - Configs are frozen (`frozen=True`) and forbid extras (`extra="forbid"`)
- [ ] **Config paths handled correctly** - If handling paths in configs, support both relative paths and `wandb:` prefix format
- [ ] **New experiments registered** - If adding new experiment, added to `spd/registry.py` with proper structure
- [ ] **Experiment structure followed** - Experiments have `models.py`, `configs.py`, `{task}_decomposition.py` in flat structure

## Testing

- [ ] **Tests written** - Unit tests for new functionality. Regression tests for bug fixes.
- [ ] **Tests run successfully** - Run `make test` (or `make test-all` if relevant)
- [ ] **Test files named correctly** - `test_*.py` format
- [ ] **Test functions named correctly** - `def test_*():` with descriptive names
- [ ] **Slow tests marked** - Used `@pytest.mark.slow` for slow tests
- [ ] **Focus on unit tests** - Not production code (no deployment). Integration tests often too much overhead for research code. Interactive use catches issues at low cost. Add integration tests only if testing complex interactions that can't be validated in units.

## Pre-Commit Checks

- [ ] **Ran `make check`** - Full pre-commit suite passes (format + type check)
- [ ] **No type errors** - basedpyright reports no issues
- [ ] **No lint errors** - ruff reports no issues

## Git & Version Control

### Before Committing
- [ ] **Checked existing patterns** - If adding new files (docs, configs, tests, etc.), looked at similar existing files for formatting/structure conventions to follow
- [ ] **Reviewed every line of the diff** - Understand every change being committed
- [ ] **Only relevant files staged** - Don't commit unrelated changes or all files
- [ ] **No secrets committed** - No `.env`, `credentials.json`, or similar files
- [ ] **Used correct branch name** - Format: `refactor/X`, `feature/Y`, or `fix/Z`

### Commit Message
- [ ] **Explains "what" and "why"** - Not just describing the diff
- [ ] **Clear and descriptive** - Focused on relevant changes
- [ ] **Explains purpose** - Why this change is needed

### Committing
- [ ] **NOT using `--no-verify`** - Almost never appropriate. Pre-commit checks exist for a reason.
- [ ] **Pre-commit hooks run** - Automatically runs `make format` and `make type`
- [ ] **All hooks passed** - No failures from pre-commit checks

## Pull Request (if creating)

### PR Content
- [ ] **Analyzed all changes** - Reviewed git diff and git status before creating PR
- [ ] **Title is clear** - Concise summary of changes
- [ ] **Used PR template** - Filled out all sections in `.github/pull_request_template.md`:
  - Description - What changed
  - Related Issue - "Closes #XX" format if applicable
  - Motivation and Context - Why needed
  - Testing - How tested
  - Breaking Changes - Listed if any

### PR Quality
- [ ] **All CI checks pass** - GitHub Actions successful
- [ ] **Merged latest from main** - Branch is up to date
- [ ] **Only relevant files** - No unrelated changes included
- [ ] **Self-reviewed** - Went through diff yourself first

## Cluster Usage (if applicable)

If running experiments on the cluster:
- [ ] **NOT exceeding 8 GPUs total** - Including all sweeps/evals combined
- [ ] **Monitored jobs** - Used `squeue` to check current usage
- [ ] **Used appropriate resources** - GPU vs CPU flags set correctly

## Final Self-Review

- [ ] **Restarted checklist after any changes** - If you made ANY changes while going through this checklist, you MUST restart from the beginning. Did you restart? If not, STOP and restart now.
- [ ] **Code is simple** - Straightforward for researchers with varying experience
- [ ] **No over-engineering** - Only made changes directly requested or clearly necessary
- [ ] **No unnecessary features** - Didn't add extra functionality beyond the task
- [ ] **No premature abstraction** - Didn't create helpers/utilities for one-time operations
- [ ] **No backwards-compatibility hacks** - Removed unused code completely instead of commenting
- [ ] **Followed fail-fast principle** - Code fails immediately when assumptions violated
- [ ] **Type safety maintained** - All functions properly typed
- [ ] **Tests are sufficient** - Core functionality tested, not over-tested

## Common Mistakes to Avoid

- ❌ Forgetting to remove obvious comments like `# get dataloader`
- ❌ Committing without running `make check`
- ❌ Using `--no-verify` flag
- ❌ Recovering silently from errors instead of failing
- ❌ Adding type annotations to obvious assignments like `name: str = "John"`
- ❌ Committing all files instead of only relevant changes
- ❌ Using more than 8 GPUs on cluster (total across all jobs)
- ❌ Failing to consult CLAUDE_COMPREHENSIVE.md for clarification in cases where the checklist is unclear. 
- ❌ Starting this checklist, noticing an issue, fixing it, and then forgetting to start the checklist **from the start** again. 
