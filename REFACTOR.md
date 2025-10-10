# Clustering Pipeline Refactor Plan

## Overview

Refactor the clustering pipeline to use isolated SLURM jobs for each clustering run, similar to how `spd-run` manages distributed SPD experiments. This plan focuses **only** on isolated clustering runs - ensemble consolidation will be handled in a future stage.

## Current Architecture

### Entry Point
- `spd/clustering/scripts/main.py` - CLI that:
  - Loads `ClusteringRunConfig` from a config file
  - Calls `clustering_pipeline.main(config)`

### Execution Flow
1. **s1_split_dataset.py** - Splits dataset into `n_batches` and saves to disk
2. **distribute_clustering()** (in `dist_utils.py`):
   - Manually manages subprocess pool across devices
   - Each subprocess runs `s2_clustering.py` CLI on a single batch
   - Tracks concurrency with `workers_per_device`
   - Returns results from all batches
3. **s3_normalize_histories.py** - Normalizes merge histories
4. **s4_compute_distances.py** - Computes distances between clusterings

### Issues with Current Approach
- Manual distributed management is complex
- All batches use the same dataset (just split differently)
- Cannot run clustering standalone easily for testing
- No per-run wandb tracking
- Different architecture from regular SPD runs
- Mixed concerns: single run vs. ensemble consolidation

## Desired Architecture

### Key Principles
1. **SLURM-based distribution** - Use SLURM array jobs like `spd-run`
2. **Isolated clustering runs** - Each run is completely independent:
   - Own dataset (seeded by `idx_in_ensemble`)
   - Own SLURM job
   - Own WandB run (if enabled)
   - Own merge history output
3. **Standalone execution** - Can run manually: `python .../run_clustering.py config.yaml`
4. **No ensemble logic** - This stage doesn't know about ensembles, just individual runs
5. **Shared utilities** - Reuse `slurm_utils.py` where appropriate (without major changes)

## Proposed Changes

### 1. Config Structure

#### Single Config: `ClusteringRunConfig`
Simplified config for **one** clustering run:

```python
class ClusteringRunConfig(BaseModel):
    """Configuration for a single clustering run."""

    # Core parameters
    model_path: str = Field(
        description="WandB path to the decomposed model (format: wandb:entity/project/run_id)"
    )
    task_name: TaskName = Field(
        description="Task name (tms, resid_mlp, lm, ih)"
    )
    experiment_key: str | None = Field(
        default=None,
        description="Optional experiment key from EXPERIMENT_REGISTRY"
    )

    # Run parameters
    idx_in_ensemble: int = Field(
        description="Index of this run in the ensemble (used for seeding and naming)"
    )
    batch_size: int = Field(
        default=64,
        description="Batch size for processing"
    )

    # Merge algorithm
    merge_config: MergeConfig = Field(
        description="Merge algorithm configuration"
    )

    # Dataset seeding
    dataset_seed: int = Field(
        description="Seed for dataset generation/loading"
    )

    # Output
    output_dir: Path = Field(
        description="Directory to save merge history"
    )

    # WandB
    wandb_enabled: bool = Field(
        default=True,
        description="Enable WandB logging"
    )
    wandb_project: str = Field(
        default="spd-cluster",
        description="WandB project name"
    )
    wandb_group: str | None = Field(
        default=None,
        description="WandB group name for organizing related runs"
    )
    intervals: IntervalsDict = Field(
        default_factory=lambda: _DEFAULT_INTERVALS.copy(),
        description="Logging intervals"
    )

    @property
    def run_id(self) -> str:
        """Unique identifier for this clustering run.

        Format: {ensemble_hash}_{idx_in_ensemble}
        This allows grouping runs together while maintaining uniqueness.
        """
        # Create stable hash from key config parameters
        config_str = f"{self.model_path}_{self.merge_config.alpha}_{self.dataset_seed}"
        ensemble_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{ensemble_hash}_{self.idx_in_ensemble}"

    @property
    def wandb_decomp_model(self) -> str:
        """Extract the WandB run ID of the source decomposition."""
        parts = self.model_path.replace("wandb:", "").split("/")
        if len(parts) >= 3:
            return parts[-1] if parts[-1] != "runs" else parts[-2]
        raise ValueError(f"Invalid wandb path format: {self.model_path}")
```

**Key changes from old config:**
- Removed `n_batches` (ensemble size) - doesn't belong here
- Removed `workers_per_device`, `devices`, `base_path` - handled by submitter
- Removed `distances_method` - that's for ensemble consolidation stage
- Added `idx_in_ensemble` to identify position in ensemble
- Added `run_id` property that creates globally unique ID
- Simplified to focus on single run parameters

### 2. Clustering Script: `run_clustering.py`

Refactor `s2_clustering.py` → `run_clustering.py`:

```python
# spd/clustering/pipeline/run_clustering.py

def run_clustering(
    config: ClusteringRunConfig,
    device: str = "cuda",
) -> Path:
    """Run clustering on a single dataset batch.

    Args:
        config: Clustering run configuration
        device: Device to run on (e.g., 'cuda:0', 'cpu')

    Returns:
        Path to saved merge history file
    """
    logger.info(f"Starting clustering run {config.run_id}")

    # 1. Load dataset with run-specific seed
    logger.info(f"Loading dataset (seed={config.dataset_seed})")
    batch = load_dataset(
        model_path=config.model_path,
        task_name=config.task_name,
        batch_size=config.batch_size,
        seed=config.dataset_seed,
    )

    # 2. Setup WandB for this run
    run: Run | None = None
    if config.wandb_enabled:
        run = wandb.init(
            project=config.wandb_project,
            name=f"cluster-{config.run_id}",
            group=config.wandb_group,
            config=config.model_dump(mode="json"),
            tags=[
                "clustering",
                f"task:{config.task_name}",
                f"model:{config.wandb_decomp_model}",
                f"idx:{config.idx_in_ensemble}",
            ],
        )
        logger.info(f"WandB run: {run.url}")

    # 3. Load model
    spd_run = SPDRunInfo.from_path(config.model_path)
    model = ComponentModel.from_pretrained(spd_run.checkpoint_path).to(device)

    # 4. Compute activations
    activations_dict = component_activations(
        model=model,
        batch=batch,
        device=device,
        sigmoid_type=spd_run.config.sigmoid_type,
    )

    # 5. Process activations
    processed_activations = process_activations(
        activations=activations_dict,
        filter_dead_threshold=config.merge_config.filter_dead_threshold,
        seq_mode="concat" if config.task_name == "lm" else None,
        filter_modules=config.merge_config.filter_modules,
    )

    # 6. Log activations (if WandB enabled)
    if run is not None:
        plot_activations(processed_activations, wandb_run=run)
        wandb_log_tensor(run, processed_activations.activations, "activations", 0, single=True)

    # 7. Run merge iteration
    log_callback = partial(_log_callback, run=run, config=config) if run else None

    history = merge_iteration(
        merge_config=config.merge_config,
        activations=processed_activations.activations,
        component_labels=ComponentLabels(processed_activations.labels),
        log_callback=log_callback,
        batch_id=config.run_id,
    )

    # 8. Save merge history
    config.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = config.output_dir / f"history_{config.idx_in_ensemble}.npz"
    history.save(history_path, wandb_url=run.url if run else None)

    # 9. Log to WandB
    if run is not None:
        _log_merge_history_plots(run, history)
        _save_merge_history_artifact(run, history_path, config.run_id, history)
        run.finish()

    logger.info(f"Clustering complete: {history_path}")
    return history_path


def cli() -> None:
    """CLI for running a single clustering run."""
    parser = argparse.ArgumentParser(
        description="Run clustering on a single dataset"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to ClusteringRunConfig file (JSON/YAML/TOML)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )

    args = parser.parse_args()

    # Load config
    config = ClusteringRunConfig.read(args.config)

    # Run clustering
    history_path = run_clustering(config, device=args.device)

    print(f"✓ Merge history saved: {history_path}")


if __name__ == "__main__":
    cli()
```

**Key features:**
- Accepts `ClusteringRunConfig` for single run
- Loads dataset with `dataset_seed`
- Can be run standalone: `python run_clustering.py config.yaml`
- Creates its own WandB run with appropriate tags
- Saves merge history to `output_dir`
- No mention of ensemble, batches, or distribution

### 3. Dataset Loading

Extract dataset loading logic:

```python
# spd/clustering/pipeline/dataset.py

def load_dataset(
    model_path: str,
    task_name: TaskName,
    batch_size: int,
    seed: int,
) -> BatchTensor:
    """Load a single batch for clustering.

    Each run gets its own dataset batch, seeded by idx_in_ensemble.

    Args:
        model_path: Path to decomposed model
        task_name: Task type
        batch_size: Batch size
        seed: Random seed for dataset

    Returns:
        Single batch of data
    """
    match task_name:
        case "lm":
            return _load_lm_batch(model_path, batch_size, seed)
        case "resid_mlp":
            return _load_resid_mlp_batch(model_path, batch_size, seed)
        case _:
            raise ValueError(f"Unsupported task: {task_name}")


def _load_lm_batch(model_path: str, batch_size: int, seed: int) -> BatchTensor:
    """Load a batch for language model task."""
    spd_run = SPDRunInfo.from_path(model_path)
    cfg = spd_run.config

    assert isinstance(cfg.task_config, LMTaskConfig)

    dataset_config = DatasetConfig(
        name=cfg.task_config.dataset_name,
        hf_tokenizer_path=cfg.pretrained_model_name,
        split=cfg.task_config.train_data_split,
        n_ctx=cfg.task_config.max_seq_len,
        seed=seed,  # Use run-specific seed
        streaming=False,
        is_tokenized=False,
        column_name=cfg.task_config.column_name,
    )

    dataloader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=batch_size,
        buffer_size=cfg.task_config.buffer_size,
        global_seed=seed,  # Use run-specific seed
        ddp_rank=0,
        ddp_world_size=1,
    )

    # Get first batch
    batch = next(iter(dataloader))
    return batch["input_ids"]


def _load_resid_mlp_batch(model_path: str, batch_size: int, seed: int) -> BatchTensor:
    """Load a batch for ResidMLP task."""
    from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset

    spd_run = SPDRunInfo.from_path(model_path)
    cfg = spd_run.config
    component_model = ComponentModel.from_pretrained(spd_run.checkpoint_path)

    assert isinstance(cfg.task_config, ResidMLPTaskConfig)
    assert isinstance(component_model.target_model, ResidMLP)

    # Create dataset with run-specific seed
    dataset = ResidMLPDataset(
        n_features=component_model.target_model.config.n_features,
        feature_probability=cfg.task_config.feature_probability,
        device="cpu",
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=seed,  # Use run-specific seed
        label_coeffs=None,
        data_generation_type=cfg.task_config.data_generation_type,
    )

    # Generate batch
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch, _ = next(iter(dataloader))
    return batch
```

### 4. SLURM Submission Script: `spd-cluster`

Create new entry point `spd/clustering/scripts/run.py`:

```python
# spd/clustering/scripts/run.py

def generate_run_id_for_ensemble(config: ClusteringSubmitConfig) -> str:
    """Generate a unique ensemble identifier based on config.

    This is used as the ensemble_hash component in individual run IDs.
    Format: cluster_{timestamp}
    """
    return f"cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def generate_clustering_commands(
    submit_config: ClusteringSubmitConfig,
    ensemble_hash: str,
) -> list[str]:
    """Generate commands for each clustering run.

    Args:
        submit_config: Submission configuration
        ensemble_hash: Shared hash for this ensemble

    Returns:
        List of commands, one per run
    """
    commands = []

    base_output_dir = submit_config.base_path / "clustering" / "clustering_runs" / ensemble_hash

    for idx in range(submit_config.n_runs):
        # Create config for this specific run
        run_config = ClusteringRunConfig(
            model_path=submit_config.model_path,
            task_name=submit_config.task_name,
            experiment_key=submit_config.experiment_key,
            idx_in_ensemble=idx,
            batch_size=submit_config.batch_size,
            merge_config=submit_config.merge_config,
            dataset_seed=submit_config.dataset_seed_base + idx,
            output_dir=base_output_dir / "merge_histories",
            wandb_enabled=submit_config.wandb_enabled,
            wandb_project=submit_config.wandb_project,
            wandb_group=f"ensemble-{ensemble_hash}",
            intervals=submit_config.intervals,
        )

        # Serialize to JSON for passing to script
        config_json = f"json:{json.dumps(run_config.model_dump(mode='json'))}"

        # Build command
        command = f"python spd/clustering/pipeline/run_clustering.py '{config_json}'"
        commands.append(command)

    return commands


def main(
    config: Path,
    cpu: bool = False,
    partition: str = "h100-reserved",
    create_snapshot: bool = True,
) -> None:
    """Submit clustering runs to SLURM.

    Args:
        config: Path to ClusteringSubmitConfig file
        cpu: Use CPU instead of GPU
        partition: SLURM partition to use
        create_snapshot: Create git snapshot for reproducibility
    """
    logger.set_format("console", "default")

    # Load submission config
    submit_config = ClusteringSubmitConfig.read(config)
    logger.info(f"Loaded config: {submit_config.n_runs} runs")

    # Generate ensemble hash
    ensemble_hash = generate_run_id_for_ensemble(submit_config)
    logger.info(f"Ensemble hash: {ensemble_hash}")

    # Create git snapshot
    if create_snapshot:
        snapshot_branch, commit_hash = create_git_snapshot(branch_name_prefix="cluster")
        logger.info(f"Git snapshot: {snapshot_branch} ({commit_hash[:8]})")
    else:
        snapshot_branch = repo_current_branch()
        commit_hash = "none"
        logger.info(f"Using current branch: {snapshot_branch}")

    # Setup WandB (create workspace view)
    if submit_config.wandb_enabled:
        workspace_url = create_clustering_workspace_view(
            ensemble_hash=ensemble_hash,
            project=submit_config.wandb_project,
        )
        logger.info(f"WandB workspace: {workspace_url}")

    # Generate commands
    commands = generate_clustering_commands(submit_config, ensemble_hash)
    logger.info(f"Generated {len(commands)} commands")

    # Submit to SLURM
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = Path(temp_dir) / f"cluster_{ensemble_hash}.sh"

        create_slurm_array_script(
            script_path=script_path,
            job_name=f"spd-cluster-{ensemble_hash}",
            commands=commands,
            snapshot_branch=snapshot_branch,
            max_concurrent_tasks=submit_config.n_runs,  # Run all concurrently
            n_gpus_per_job=0 if cpu else 1,  # Always 1 GPU per run
            partition=partition,
        )

        array_job_id = submit_slurm_array(script_path)

        logger.section("Job submitted successfully!")
        logger.values({
            "Array Job ID": array_job_id,
            "Total runs": len(commands),
            "Ensemble hash": ensemble_hash,
            "Logs": f"~/slurm_logs/slurm-{array_job_id}_*.out",
        })


def cli():
    """CLI for spd-cluster command."""
    parser = argparse.ArgumentParser(
        prog="spd-cluster",
        description="Submit clustering runs to SLURM",
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to ClusteringSubmitConfig file",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="h100-reserved",
        help="SLURM partition to use",
    )
    parser.add_argument(
        "--create-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create git snapshot",
    )

    args = parser.parse_args()
    main(
        config=args.config,
        cpu=args.cpu,
        partition=args.partition,
        create_snapshot=args.create_snapshot,
    )


if __name__ == "__main__":
    cli()
```

**Key features:**
- No `--local` flag - if you want local, just run `run_clustering.py` directly
- No `n_agents` - we just submit all runs (let SLURM manage concurrency)
- Hardcoded 1 GPU per run
- Creates unique `ensemble_hash` for grouping runs
- Uses `ClusteringSubmitConfig` (separate from `ClusteringRunConfig`)

### 5. Submission Config

Create a config for the submission script (not the individual runs):

```python
# spd/clustering/clustering_submit_config.py

class ClusteringSubmitConfig(BaseModel):
    """Configuration for submitting clustering runs via spd-cluster.

    This config is used by the submission script to generate multiple
    individual ClusteringRunConfig instances.
    """

    # Model configuration
    model_path: str = Field(
        description="WandB path to decomposed model"
    )
    task_name: TaskName = Field(
        description="Task name"
    )
    experiment_key: str | None = Field(
        default=None,
        description="Optional experiment key from registry"
    )

    # Ensemble parameters
    n_runs: int = Field(
        default=10,
        description="Number of clustering runs to perform"
    )
    batch_size: int = Field(
        default=64,
        description="Batch size for each run"
    )
    dataset_seed_base: int = Field(
        default=42,
        description="Base seed for datasets (each run gets seed_base + idx_in_ensemble)"
    )

    # Merge configuration
    merge_config: MergeConfig = Field(
        description="Merge algorithm configuration"
    )

    # Output
    base_path: Path = Field(
        default=Path(".data/clustering"),
        description="Base directory for all clustering outputs"
    )

    # WandB
    wandb_enabled: bool = Field(
        default=True,
        description="Enable WandB logging"
    )
    wandb_project: str = Field(
        default="spd-cluster",
        description="WandB project name"
    )
    intervals: IntervalsDict = Field(
        default_factory=lambda: _DEFAULT_INTERVALS.copy(),
        description="Logging intervals"
    )

    @classmethod
    def read(cls, path: Path) -> "ClusteringSubmitConfig":
        """Load from JSON/YAML/TOML file."""
        # Same logic as ClusteringRunConfig.read()
        # Support experiment_key resolution via EXPERIMENT_REGISTRY
```

### 6. Storage Structure

Simplified storage structure:

```
.data/clustering/clustering_runs/
└── {ensemble_hash}/
    ├── clustering_config.json         # ClusteringSubmitConfig
    └── merge_histories/
        ├── history_0.npz
        ├── history_1.npz
        ├── history_2.npz
        └── ...
```

**Key simplifications:**
- No separate `batches` directory (data loaded on-the-fly)
- No `results` directory (history files are the output)
- Named `clustering_runs` to distinguish from future `ensemble_merging_runs`
- All histories in one flat directory

Simplify `ClusteringStorage`:

```python
# spd/clustering/pipeline/storage.py

class ClusteringStorage:
    """Storage manager for clustering runs."""

    def __init__(self, base_path: Path, ensemble_hash: str):
        self.base_path = base_path
        self.ensemble_hash = ensemble_hash
        self.run_path = base_path / "clustering_runs" / ensemble_hash
        self.histories_dir = self.run_path / "merge_histories"

    def save_config(self, config: ClusteringSubmitConfig) -> None:
        """Save submission config."""
        self.run_path.mkdir(parents=True, exist_ok=True)
        config_path = self.run_path / "clustering_config.json"
        config.save(config_path)

    def get_history_paths(self) -> list[Path]:
        """Get all merge history files."""
        return sorted(self.histories_dir.glob("history_*.npz"))
```

### 7. WandB Integration

Create workspace view for clustering runs:

```python
# spd/clustering/utils/wandb_utils.py

def create_clustering_workspace_view(
    ensemble_hash: str,
    project: str = "spd-cluster",
) -> str:
    """Create WandB workspace view for clustering runs.

    Args:
        ensemble_hash: Unique identifier for this ensemble
        project: WandB project name

    Returns:
        URL to workspace view
    """
    # Use a template workspace (similar to spd/utils/wandb_utils.py)
    template_url = "https://wandb.ai/goodfire/spd-cluster?nw=..."  # TODO: create template
    workspace = ws.Workspace.from_url(template_url)

    workspace.project = project
    workspace.name = f"Clustering - {ensemble_hash}"

    # Filter for runs with this ensemble_hash
    # Runs will be tagged with wandb_group=f"ensemble-{ensemble_hash}"
    workspace.runset_settings.filters = [
        ws.Tags("group").isin([f"ensemble-{ensemble_hash}"]),
    ]

    workspace.save_as_new_view()
    return workspace.url
```

### 8. File Structure After Refactor

```
spd/clustering/
├── pipeline/
│   ├── run_clustering.py          # Main clustering script (formerly s2_clustering.py)
│   ├── dataset.py                 # Dataset loading utilities
│   ├── storage.py                 # Simplified storage (remove batch management)
│   ├── s3_normalize_histories.py  # Future: ensemble consolidation stage
│   └── s4_compute_distances.py    # Future: ensemble consolidation stage
├── scripts/
│   └── run.py                     # New: spd-cluster entry point
├── clustering_run_config.py       # New: ClusteringRunConfig
├── clustering_submit_config.py    # New: ClusteringSubmitConfig
├── utils/
│   └── wandb_utils.py             # Clustering-specific WandB utilities
└── configs/
    └── example.yaml               # Example ClusteringSubmitConfig

# Files to DELETE:
# - spd/clustering/scripts/main.py
# - spd/clustering/pipeline/s1_split_dataset.py
# - spd/clustering/pipeline/s2_clustering.py
# - spd/clustering/pipeline/dist_utils.py
# - spd/clustering/pipeline/clustering_pipeline.py
# - spd/clustering/merge_run_config.py
```

## Usage Examples

### Submit clustering runs to SLURM
```bash
spd-cluster configs/example.yaml
spd-cluster configs/example.yaml --cpu
spd-cluster configs/example.yaml --partition h100-reserved
```

### Run single clustering locally (for testing)
```bash
# Create a config for a single run
cat > my_run.yaml <<EOF
model_path: "wandb:goodfire/spd/runs/abc123"
task_name: "resid_mlp"
idx_in_ensemble: 0
batch_size: 64
dataset_seed: 42
output_dir: ".data/test_run"
merge_config:
  activation_threshold: 0.01
  alpha: 1.0
  iters: 100
EOF

# Run it
python spd/clustering/pipeline/run_clustering.py my_run.yaml
```

### Example submit config
```yaml
# configs/example_submit.yaml

# Model
model_path: "wandb:goodfire/spd/runs/ioprgffh"
task_name: "lm"

# Ensemble
n_runs: 10
batch_size: 64
dataset_seed_base: 42

# WandB
wandb_enabled: true
wandb_project: "spd-cluster"

# Merge algorithm
merge_config:
  activation_threshold: 0.01
  alpha: 1.0
  iters: 100
  pop_component_prob: 0
  filter_dead_threshold: 0.001
  module_name_filter: null
  rank_cost_fn_name: "const_1"
  merge_pair_sampling_method: "range"
  merge_pair_sampling_kwargs:
    threshold: 0.05

# Output
base_path: ".data/clustering"

intervals:
  stat: 1
  tensor: 100
  plot: 100
  artifact: 100
```

## Implementation Plan

### Phase 1: Core Refactoring (Priority)
1. Create `ClusteringRunConfig` class
2. Create `ClusteringSubmitConfig` class
3. Extract dataset loading into `dataset.py`
4. Refactor `s2_clustering.py` → `run_clustering.py`:
   - Accept `ClusteringRunConfig`
   - Integrate dataset loading
   - Update WandB tagging
   - Support standalone CLI

### Phase 2: SLURM Submission (Priority)
5. Create `spd/clustering/scripts/run.py`:
   - Command generation logic
   - SLURM submission using existing `slurm_utils.py`
   - Git snapshot integration
6. Create `spd-cluster` CLI entry point
7. Add WandB workspace view creation

### Phase 3: Storage & Cleanup
8. Simplify `ClusteringStorage` class
9. Remove deprecated files:
   - `s1_split_dataset.py`
   - `s2_clustering.py`
   - `dist_utils.py`
   - `clustering_pipeline.py`
   - `main.py`
   - `merge_run_config.py`

### Phase 4: Testing & Documentation
10. Test standalone run: `python run_clustering.py config.yaml`
11. Test SLURM submission: `spd-cluster config.yaml`
12. Verify WandB integration
13. Update documentation

## Key Decisions Made (Based on Feedback)

1. **Terminology**: Use "run" and "clustering run" - no "ensemble" at this stage
2. **Naming**: Use `idx_in_ensemble` instead of `ensemble_idx`
3. **Config split**:
   - `ClusteringRunConfig` for individual runs
   - `ClusteringSubmitConfig` for SLURM submission
4. **Run ID format**: `{ensemble_hash}_{idx_in_ensemble}` for global uniqueness
5. **Storage path**: `.data/clustering/clustering_runs/{ensemble_hash}/merge_histories/`
6. **GPU allocation**: Always 1 GPU per run (hardcoded)
7. **CLI**: `spd-cluster` for submission, `run_clustering.py` for individual runs
8. **No local flag**: Run `run_clustering.py` directly for local testing
9. **WandB tags**: Always include "clustering" tag, plus task/model/idx tags
10. **Backwards compatibility**: None - clean break
11. **Shared utilities**: Reuse `slurm_utils.py` without major changes

## Success Criteria

1. ✅ Can run clustering standalone: `python run_clustering.py config.yaml`
2. ✅ Can submit to SLURM: `spd-cluster submit_config.yaml`
3. ✅ Each run has unique dataset (seeded by `idx_in_ensemble`)
4. ✅ Each run has own WandB run with "clustering" tag
5. ✅ Each run has unique `run_id` in format `{hash}_{idx}`
6. ✅ All runs grouped in single WandB workspace view
7. ✅ Merge histories saved to `.data/clustering/clustering_runs/{hash}/merge_histories/`
8. ✅ Similar architecture to `spd-run` for consistency
9. ✅ No manual process management (all via SLURM)
10. ✅ Clean separation from future ensemble consolidation stage

## Future Work (Out of Scope)

The following will be addressed in a separate refactor:
- Ensemble consolidation (combining merge histories)
- Distance computation between clusterings
- Ensemble analysis and visualization
- Storage for consolidated ensemble results
