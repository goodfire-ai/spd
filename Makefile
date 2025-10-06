# setup
.PHONY: install
install: copy-templates
	uv sync --no-dev --extra cu128

.PHONY: install-dev
install-dev: copy-templates
	uv sync --extra cu128
	pre-commit install

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
NUM_PROCESSES ?= auto

.PHONY: test
test:
	pytest tests/ --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal

NUM_PROCESSES ?= auto

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


.PHONY: bundle-dashboard
bundle-dashboard:
	@mkdir -p spd/clustering/dashboard/_bundled
	python -m muutils.web.bundle_html \
		spd/clustering/dashboard/index.html \
		--output spd/clustering/dashboard/_bundled/index.html \
		--source-dir spd/clustering/dashboard
	python -m muutils.web.bundle_html \
		spd/clustering/dashboard/cluster.html \
		--output spd/clustering/dashboard/_bundled/cluster.html \
		--source-dir spd/clustering/dashboard
	@echo "Bundled HTML files to spd/clustering/dashboard/_bundled/"

.PHONY: clustering-dashboard
clustering-dashboard: bundle-dashboard
	python spd/clustering/dashboard/run.py \
		--wandb-run goodfire/spd-cluster/runs/j8dgvemf \
		--iteration 7000 \
		--n-samples 32 \
		--n-batches 4 \
		--batch-size 128


.PHONY: diskinfo
diskinfo:
	echo "### disk"
	df -h
	echo "### root"
	du -h -d 1 / 2>/dev/null || true
	echo "### spd"
	du -h -d 1
	echo "spd depth 2"
	du -h -d 2
