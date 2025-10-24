# setup
.PHONY: install
install: copy-templates
	uv sync --no-dev

.PHONY: install-dev
install-dev: copy-templates
	uv sync
	pre-commit install


# special install for CI (GitHub Actions) that reduces disk usage and install time
# 1. delete the existing lock file, it messes with things
# 2. create a fresh venv with `--clear` -- this is mostly only for local testing of the CI install
# 3. install with `uv sync` but with some special options:
#  > `--link-mode copy` because:
#    - symlinks/hardlinks dont work half the time anyway
#    - we want to kill the cache after installing before we run the tests
#  > `--extra-index-url` to get cpu-only pytorch wheels. installing with just `uv sync` will download a bunch of cuda stuff we cannot use anyway, since there are no GPUs anyways. takes up a lot of space and makes the install take 5x as long
#  > `--index-strategy unsafe-best-match` because pytorch index won't have every version of each package we need. markupsafe is a particular pain point
# Note: explored the `--compile-bytecode` option for test speedups, nothing came of it. see https://github.com/goodfire-ai/spd/pull/187/commits/740f6a28f4d3378078c917125356b6466f155e71
.PHONY: install-ci
install-ci:
	rm -f uv.lock
	uv venv --python 3.13 --clear
	uv sync \
		--link-mode copy \
		--extra-index-url https://download.pytorch.org/whl/cpu \
		--index-strategy unsafe-best-match

.PHONY: copy-templates
copy-templates:
	@if [ ! -f spd/scripts/sweep_params.yaml ]; then \
		cp spd/scripts/sweep_params.yaml.example spd/scripts/sweep_params.yaml; \
		echo "Created spd/scripts/sweep_params.yaml from template"; \
	fi


# checks
.PHONY: type
type:
	basedpyright

.PHONY: format
format:
	# Fix all autofixable problems (which sorts imports) then format errors
	ruff check --fix
	ruff format

.PHONY: check
check: format type

.PHONY: check-pre-commit
check-pre-commit:
	SKIP=no-commit-to-branch pre-commit run -a --hook-stage commit

# tests

.PHONY: test
test:
	pytest tests/ --durations 10

# Use min(4, nproc) for numprocesses. Any more and it slows down the tests.
NUM_PROCESSES ?= $(shell nproc | awk '{print ($$1<4?$$1:4)}')

.PHONY: test-all
test-all:
	pytest tests/ --runslow --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal

COVERAGE_DIR=docs/coverage

.PHONY: coverage
coverage:
	uv run pytest tests/ --cov=spd --runslow
	mkdir -p $(COVERAGE_DIR)
	uv run python -m coverage report -m > $(COVERAGE_DIR)/coverage.txt
	uv run python -m coverage html --directory=$(COVERAGE_DIR)/html/


BUNDLED_DASHBOARD_DIR=spd/clustering/dashboard/_bundled

.PHONY: bundle-dashboard
bundle-dashboard:
	@mkdir -p $(BUNDLED_DASHBOARD_DIR)
	uv run python -m muutils.web.bundle_html \
		spd/clustering/dashboard/index.html \
		--output $(BUNDLED_DASHBOARD_DIR)/index.html \
		--source-dir spd/clustering/dashboard
	uv run python -m muutils.web.bundle_html \
		spd/clustering/dashboard/cluster.html \
		--output $(BUNDLED_DASHBOARD_DIR)/cluster.html \
		--source-dir spd/clustering/dashboard
	@echo "Bundled HTML files to $(BUNDLED_DASHBOARD_DIR)/"

.PHONY: clean-test-dashboard
clean-test-dashboard:
	rm -rf tests/.temp/dashboard-integration


.PHONY: install-playwright
install-playwright:
	@echo "Install Playwright browsers, used for dashboard tests"
	uv run --with playwright playwright install chromium

.PHONY: test-dashboard
test-dashboard: clean-test-dashboard bundle-dashboard
	pytest tests/clustering/dashboard/test_dashboard_integration.py --runslow -v --durations 10


.PHONY: clustering-dashboard
clustering-dashboard: bundle-dashboard
	uv run python spd/clustering/dashboard/run.py \
		spd/clustering/dashboard/dashboard_config.yaml

.PHONY: clustering-dashboard-profile
clustering-dashboard-profile: bundle-dashboard
	uv run python -m cProfile -o dashboard.prof spd/clustering/dashboard/run.py \
		spd/clustering/dashboard/dashboard_config.yaml
	@echo "\nProfile saved to dashboard.prof"
	@echo "View with: python -m pstats dashboard.prof"
	@echo "Or install snakeviz and run: snakeviz dashboard.prof"

.PHONY: clean
clean:
	@echo "Cleaning Python cache and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .ruff_cache/ .pytest_cache/ .coverage


.PHONY: clustering-dev
clustering-dev:
	uv run spd-cluster --local --config spd/clustering/configs/pipeline-dev-simplestories.yaml

