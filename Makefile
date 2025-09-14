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

NUM_PROCESSES ?= auto

.PHONY: test-all
test-all:
	pytest tests/ --runslow -v --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal

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
	spd-cluster \
	  --config spd/clustering/configs/simplestories_dev.json \
	  --devices cuda:0 \
	  --max-concurrency 1

.PHONY: clustering-resid_mlp1
clustering-resid_mlp1:
	spd-cluster \
	  --config spd/clustering/configs/resid_mlp1.json \
	  --devices cuda:0 \
	  --max-concurrency 8

.PHONY: clustering-test
clustering-test:
	pytest tests/clustering/test_clustering_experiments.py --runslow -vvv --durations 10 --numprocesses $(NUM_PROCESSES)


.PHONY: clustering-dashboard
clustering-dashboard:
	python spd/clustering/dashboard/lm_max_activations.py wandb:goodfire/spd/runs/okd93sk2 data/clustering/task_lm-w_okd93sk2-a1-i5.0K-b4-n1-h_8295a2/merge_history/task_lm-w_okd93sk2-a1-i5.0K-b4-n1-h_8295a2-data_batch_00/merge_history.zip
