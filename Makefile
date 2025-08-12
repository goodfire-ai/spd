# setup
.PHONY: install
install: copy-templates
	uv sync --no-dev

.PHONY: install-dev
install-dev: copy-templates
	uv sync
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
.PHONY: test
test:
	pytest tests/

.PHONY: test-all
test-all:
	pytest tests/ --runslow

COVERAGE_DIR=docs/coverage

.PHONY: coverage
coverage:
	uv run pytest tests/ --cov=spd --runslow
	mkdir -p $(COVERAGE_DIR)
	uv run python -m coverage report -m > $(COVERAGE_DIR)/coverage.txt
	uv run python -m coverage html --directory=$(COVERAGE_DIR)/html/

DEP_GRAPH_DIR=docs/dep_graph

.PHONY: dep-graph
dep-graph:
	ruff analyze graph > $(DEP_GRAPH_DIR)/import_graph.json



.PHONY: clustering-ss
clustering-ss:
	uv run python -m spd.clustering.scripts.main \
	  --config spd/clustering/configs/simplestories_dev.json \
	  --devices cuda:0 \
	  --max-concurrency 1

.PHONY: clustering-resid_mlp1
clustering-resid_mlp1:
	uv run python -m spd.clustering.scripts.main \
	  --config spd/clustering/configs/resid_mlp1.json \
	  --devices cuda:0 \
	  --max-concurrency 8