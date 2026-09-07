# setup
# ONE venv: `param_decomp` carries jax as a normal dependency, so a single `uv sync`
# installs everything into `.venv`. The CPU jax wheel is the base; a GPU host adds the
# `[cuda]` (or `[cuda13]`) extra.
.PHONY: install
install:
	uv sync --no-dev

.PHONY: install-dev
install-dev:
	uv sync
	uv run --no-sync pre-commit install

# special install for CI (GitHub Actions) that reduces disk usage and install time
# 1. create a fresh venv with `--clear` -- this is mostly only for local testing of the CI install
# 2. install with `uv sync` but with some special options:
#  > `--frozen` to enforce using the lock file for consistent dependency versions
#  > `--link-mode copy` because symlinks/hardlinks dont work half the time anyway
# Note: explored the `--compile-bytecode` option for test speedups, nothing came of it. see https://github.com/goodfire-ai/param-decomp/pull/187/commits/740f6a28f4d3378078c917125356b6466f155e71
.PHONY: install-ci
install-ci:
	uv venv --python 3.13 --clear
	uv sync \
		--frozen \
		--link-mode copy

# checks
.PHONY: type
type:
	uv run basedpyright

.PHONY: format
format:
	# Fix all autofixable problems (which sorts imports) then format errors
	uv run ruff check --fix
	uv run ruff format

.PHONY: check
check: format type

.PHONY: check-pre-commit
check-pre-commit:
	SKIP=no-commit-to-branch pre-commit run -a --hook-stage commit

# tests

# All Python tests live under `param_decomp/tests/`, mirroring the public package.
TEST_PATHS = param_decomp/tests/

# min(16, nproc). XLA already threads within each test, so once the workers saturate the
# box another one buys nothing — the cap only stops a large workstation spawning dozens for
# no gain. testmon is compatible: it ships its own xdist controller/worker sync.
NUM_PROCESSES ?= $(shell (nproc 2>/dev/null || sysctl -n hw.ncpu) | awk '{print ($$1<16?$$1:16)}')

.PHONY: test
test:
	uv run pytest $(TEST_PATHS) --testmon --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal

.PHONY: test-all
test-all:
	uv run pytest $(TEST_PATHS) --runslow --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal
	$(MAKE) test-multidevice

# CI shards: an exact partition of `test-all`, one CI job each, sized so every job runs
# well inside its budget cold — runner minutes are cheap, wall-clock is not. Groupings are
# semantic units (a package, or one subsystem's multidevice tests), never a balance of
# whatever happens to be slow this week. A timed-out job is killed before its post steps,
# so it never saves the JAX compile cache and every later run repeats the compile cost.
# The llama goldens stay apart: they dominate one xdist worker for ~8 min and co-schedule
# the heaviest memory peaks next to the recon end-to-end tests on a 16GB runner. The three
# core integration modules ride the lab-pipeline shard: moving their files must not move
# their large memory peaks back beside the rest of the core suite. The simulated-multidevice
# tests run single-process by construction, so each subsystem's slice is its own job.
# `param_decomp/tests/infra/test_ci_shards.py` pins the partition: every test file in
# exactly one xdist shard, every multidevice-marked file in exactly one multidevice shard.
LLAMA_GOLDEN_TEST_PATHS = param_decomp/tests/targets/test_llama31.py param_decomp/tests/targets/test_llama_simple_mlp.py
CORE_LAB_TEST_PATHS = \
	param_decomp/tests/core/test_hidden_acts_reconstruction.py \
	param_decomp/tests/core/test_no_checkpointing.py \
	param_decomp/tests/core/test_placed_eval_tiers.py
CORE_ENGINE_TEST_PATHS = param_decomp/tests/core/
CORE_TARGETS_TEST_PATHS = param_decomp/tests/routed/ param_decomp/tests/targets/
LAB_LM_TEST_PATHS = param_decomp/tests/experiments/lm/
LAB_PIPELINE_TEST_PATHS = \
	param_decomp/tests/experiments/ \
	param_decomp/tests/infra/ \
	param_decomp/tests/migrations/ \
	param_decomp/tests/target_ports/ \
	param_decomp/tests/topology/ \
	$(CORE_LAB_TEST_PATHS) \

# The multidevice slices: the files whose multidevice-marked tests belong to each
# subsystem (their other tests run in the xdist shards above). Placement is the presets
# and their census; worlds is topology invariance — padded stacks at non-dividing node
# counts, and owner/ddp trajectories against a single device.
MULTIDEVICE_PLACEMENT_TEST_PATHS = \
	param_decomp/tests/core/test_placement.py \
	param_decomp/tests/targets/test_llama_simple_mlp_placed.py \
	param_decomp/tests/targets/test_qwen36_dtype_census.py \
	param_decomp/tests/targets/test_qwen36_placement.py
MULTIDEVICE_WORLDS_TEST_PATHS = \
	param_decomp/tests/targets/test_stack_padding_placed.py \
	param_decomp/tests/targets/test_step_replicate_invariance.py
MULTIDEVICE_ROUTED_TEST_PATHS = \
	param_decomp/tests/core/test_expert_blocked.py \
	param_decomp/tests/core/test_expert_sources.py \
	param_decomp/tests/core/test_narrow_ci.py \
	param_decomp/tests/routed/test_experts.py \
	param_decomp/tests/targets/test_gated_delta_chunkwise.py
MULTIDEVICE_EVALS_TEST_PATHS = \
	param_decomp/tests/core/test_placed_eval_tiers.py \
	param_decomp/tests/experiments/lm/test_eval_context.py \
	param_decomp/tests/experiments/lm/test_well_temperedness.py \
	param_decomp/tests/targets/test_activation_capture.py
MULTIDEVICE_SUBSTRATE_TEST_PATHS = \
	param_decomp/tests/core/test_checkpoint.py \
	param_decomp/tests/core/test_checkpoint_production_topology.py \
	param_decomp/tests/core/test_masked_forward_remat.py \
	param_decomp/tests/core/test_optim_torch_parity.py \
	param_decomp/tests/core/test_sharding.py \
	param_decomp/tests/core/test_tp_boundary_topology.py \
	param_decomp/tests/experiments/lm/test_run_inline.py

# `--verbose`, never `-v`: tokamax parses sys.argv with absl flags on first use, and
# absl owns `-v` as `--verbosity`, swallowing the next token as its value.
XDIST_FLAGS = --runslow --verbose --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal

.PHONY: test-ci-llama-goldens
test-ci-llama-goldens:
	uv run pytest $(LLAMA_GOLDEN_TEST_PATHS) $(XDIST_FLAGS)

.PHONY: test-ci-core-engine
test-ci-core-engine:
	uv run pytest $(CORE_ENGINE_TEST_PATHS) $(addprefix --ignore=,$(CORE_LAB_TEST_PATHS)) $(XDIST_FLAGS)

.PHONY: test-ci-core-targets
test-ci-core-targets:
	uv run pytest $(CORE_TARGETS_TEST_PATHS) $(addprefix --ignore=,$(LLAMA_GOLDEN_TEST_PATHS)) $(XDIST_FLAGS)

.PHONY: test-ci-lab-lm
test-ci-lab-lm:
	uv run pytest $(LAB_LM_TEST_PATHS) $(XDIST_FLAGS)

.PHONY: test-ci-lab-pipeline
test-ci-lab-pipeline:
	uv run pytest $(LAB_PIPELINE_TEST_PATHS) $(addprefix --ignore=,$(LAB_LM_TEST_PATHS)) $(XDIST_FLAGS)

# Tests needing >1 device hang at the default 1, so run them on logical CPU devices.
# Eight is the suite-wide minimum: the faithfulness-fallback (2,2,2) mesh arm needs 8;
# tests wanting exactly a 2 x 2 x 1 topology slice jax.devices() themselves.
MULTIDEVICE_CPU_DEVICE_COUNT = 8
# XLA:CPU sizes its client thread pool from the host's CPU count and issues independent
# collectives in no fixed per-device order. Two collectives over the same device subgroup
# then need a spare executor thread per device to resolve; with one thread per device
# (a 4-vCPU host) they deadlock and the rendezvous watchdog aborts the process. `NPROC`
# is the pool-size override the client honours: two threads per simulated device.
MULTIDEVICE_CLIENT_THREADS = 16
MULTIDEVICE_PYTEST = NPROC=$(MULTIDEVICE_CLIENT_THREADS) XLA_FLAGS="--xla_force_host_platform_device_count=$(MULTIDEVICE_CPU_DEVICE_COUNT)" uv run pytest
MULTIDEVICE_FLAGS = -m multidevice --runmultidevice --verbose --durations 10 --capture=tee-sys

.PHONY: test-multidevice
test-multidevice:
	$(MULTIDEVICE_PYTEST) $(TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-placement
test-ci-multidevice-placement:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_PLACEMENT_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-worlds
test-ci-multidevice-worlds:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_WORLDS_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-routed
test-ci-multidevice-routed:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_ROUTED_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-evals
test-ci-multidevice-evals:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_EVALS_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-substrate
test-ci-multidevice-substrate:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_SUBSTRATE_TEST_PATHS) $(MULTIDEVICE_FLAGS)

COVERAGE_DIR=docs/coverage

.PHONY: coverage
coverage:
	uv run pytest $(TEST_PATHS) --cov=param_decomp --runslow
	mkdir -p $(COVERAGE_DIR)
	uv run python -m coverage report -m > $(COVERAGE_DIR)/coverage.txt
	uv run python -m coverage html --directory=$(COVERAGE_DIR)/html/


.PHONY: clean
clean:
	@echo "Cleaning Python cache and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .ruff_cache/ .pytest_cache/ .coverage

