# Multi-Batch Clustering Implementation Plan

## Implementation Status: ✅ ALL TASKS COMPLETE

**Date Completed:** 2025-10-27
**Branch:** `clustering/refactor-multi-batch`

All four tasks have been implemented. See task completion reports:
- `TASKS-claude-1.md` - Task 1 completion
- `TASKS-claude-2.md` - Task 2 completion
- `spd/clustering/TASKS-claude-3.md` - Task 3 completion
- `TASKS-claude-4.md` - Task 4 completion

## Overview

Implement multi-batch clustering to avoid keeping the model loaded during merge iterations. Instead, precompute multiple batches of activations and cycle through them, recomputing costs every `m` merges with a new batch.

**Core Principle: Keep It Minimal**

### Key Simplifications
- ✅ No dead component filtering (for now)
- ✅ No within-module clustering (defer to later)
- ✅ Always disk-based batch storage
- ✅ Model never kept during merge iteration
- ✅ NaN masking only (no separate boolean masks)
- ✅ Simple round-robin batch loading from disk

---

## Tasks (All Complete)

### ✅ Task 1: Batch Storage Infrastructure (COMPLETE)
**Files:** New file + config changes
**Dependencies:** None
**Estimated Lines:** ~180 lines (actual)

**Completed Components:**
- ✅ Created `spd/clustering/batched_activations.py` (~180 lines)
  - `ActivationBatch` class with save/load methods
  - `BatchedActivations` class for round-robin batch cycling
  - `precompute_batches_for_ensemble()` function
- ✅ Updated `merge_config.py`: Added `recompute_costs_every` field (default=1)
- ✅ Updated `clustering_run_config.py`: Added `precomputed_activations_dir` field
- ✅ Updated `run_clustering.py`: Added `--precomputed-activations-dir` CLI argument

**Implementation Notes:**
- File location: `spd/clustering/batched_activations.py`
- Default `recompute_costs_every=1` maintains backward compatibility
- When `precomputed_activations_dir=None`, single batch is computed on-the-fly

**See:** `TASKS-claude-1.md` for full implementation details

---

### ✅ Task 2: Core Merge Logic Refactor (COMPLETE)
**Files:** `spd/clustering/merge.py`, `spd/clustering/math/merge_pair_samplers.py`
**Dependencies:** Task 1 (needs `BatchedActivations` interface)
**Estimated Lines:** ~150 lines (actual)

**Completed Components:**
- ✅ Added `recompute_coacts_from_scratch()` helper function (`merge.py:33-61`)
- ✅ Refactored `merge_iteration()` to accept `BatchedActivations` parameter
- ✅ Implemented NaN masking for merged component rows/columns
- ✅ Added periodic batch recomputation logic (every `recompute_costs_every` iterations)
- ✅ Updated `range_sampler` in `merge_pair_samplers.py` to handle NaN values
- ✅ Updated `mcmc_sampler` in `merge_pair_samplers.py` to handle NaN values

**Key Implementation Details:**
- Function signature changed to `batched_activations: BatchedActivations`
- Loads first batch at start, cycles through batches during merge iteration
- NaN masking invalidates affected entries after each merge
- Full coactivation recomputation from fresh batch at specified intervals
- All merge pair samplers gracefully handle NaN entries

**See:** `TASKS-claude-2.md` for full implementation details

---

### ✅ Task 3: Update `run_clustering.py` (COMPLETE)
**Files:** `spd/clustering/scripts/run_clustering.py`
**Dependencies:** Task 1 (needs `BatchedActivations`)
**Estimated Lines:** ~100 lines (actual)

**Completed Components:**
- ✅ Refactored `main()` function to support two modes:
  - **Case 1:** Load precomputed batches from disk (`precomputed_activations_dir` provided)
  - **Case 2:** Compute single batch on-the-fly, save to temp directory (original behavior)
- ✅ Updated `merge_iteration()` call to use `batched_activations` parameter
- ✅ Added memory cleanup after activation computation
- ✅ Both modes wrap batches in `BatchedActivations` for unified interface

**Key Implementation Details:**
- When `precomputed_activations_dir=None`: Single batch computed and saved to temp directory
- When `precomputed_activations_dir` provided: All batches loaded from disk
- Component labels extracted from first batch in both cases
- Memory cleanup: model, batch, activations deleted after computation in Case 2

**See:** `spd/clustering/TASKS-claude-3.md` for full implementation details

---

### ✅ Task 4: Batch Precomputation in `run_pipeline.py` (COMPLETE)
**Files:** `spd/clustering/scripts/run_pipeline.py`
**Dependencies:** Task 1 (needs `ActivationBatch`)
**Estimated Lines:** ~50 lines modifications (actual)

**Note:** The `precompute_batches_for_ensemble()` function was implemented in Task 1 as part of `batched_activations.py`, not as a separate addition to `run_pipeline.py`.

**Completed Components:**
- ✅ Batch precomputation logic in `batched_activations.py` (`precompute_batches_for_ensemble()`)
- ✅ Updated `generate_clustering_commands()` to pass `--precomputed-activations-dir` argument
- ✅ Updated `main()` to call precomputation before generating commands
- ✅ Proper seeding strategy: `base_seed + run_idx * 1000 + batch_idx`

**Key Implementation Details:**
- Loads model once, generates all batches for all runs in ensemble
- Batches saved to: `<pipeline_output>/precomputed_batches/run_{idx}/batch_{idx}.pt`
- Returns `None` if `recompute_costs_every=1` (single-batch mode)
- Each run gets unique seed per batch to ensure different data

**See:** `TASKS-claude-4.md` for full implementation details

---

## Testing Plan

### Unit Tests
**File:** `tests/clustering/test_multi_batch.py` (TO BE CREATED)

**Required Tests:**
    """A single clustering run."""

    # Create ExecutionStamp and storage
    execution_stamp = ExecutionStamp.create(
        run_type="cluster",
        create_snapshot=False,
    )
    storage = ClusteringRunStorage(execution_stamp)
    clustering_run_id = execution_stamp.run_id
    logger.info(f"Clustering run ID: {clustering_run_id}")

    # Register with ensemble if this is part of a pipeline
    assigned_idx = None
    if run_config.ensemble_id:
        assigned_idx = register_clustering_run(
            pipeline_run_id=run_config.ensemble_id,
            clustering_run_id=clustering_run_id,
        )
        logger.info(
            f"Registered with pipeline {run_config.ensemble_id} at index {assigned_idx}"
        )
        # IMPORTANT: set dataset seed based on assigned index
        run_config = replace_pydantic_model(
            run_config,
            {"dataset_seed": run_config.dataset_seed + assigned_idx},
        )

    # Save config
    run_config.to_file(storage.config_path)
    logger.info(f"Config saved to {storage.config_path}")

    # Start
    logger.info("Starting clustering run")
    logger.info(f"Output directory: {storage.base_dir}")
    device = get_device()

    spd_run = SPDRunInfo.from_path(run_config.model_path)
    task_name = spd_run.config.task_config.task_name

    # Setup WandB for this run
    wandb_run = None
    if run_config.wandb_project is not None:
        wandb_run = wandb.init(
            id=clustering_run_id,
            entity=run_config.wandb_entity,
            project=run_config.wandb_project,
            group=run_config.ensemble_id,
            config=run_config.model_dump(mode="json"),
            tags=[
                "clustering",
                f"task:{task_name}",
                f"model:{run_config.wandb_decomp_model}",
                f"ensemble_id:{run_config.ensemble_id}",
                f"assigned_idx:{assigned_idx}",
            ],
        )

    # Load or compute activations
    # =====================================
    batched_activations: BatchedActivations
    component_labels: ComponentLabels

    if run_config.precomputed_activations_dir is not None:
        # Case 1: Use precomputed batches from disk
        logger.info(f"Loading precomputed batches from {run_config.precomputed_activations_dir}")
        batched_activations = BatchedActivations(run_config.precomputed_activations_dir)

        # Get labels from first batch
        first_batch = batched_activations.get_next_batch()
        component_labels = ComponentLabels(first_batch.labels)

        logger.info(f"Loaded {batched_activations.n_batches} precomputed batches")

    else:
        # Case 2: Compute single batch on-the-fly (original behavior)
        logger.info(f"Computing single batch (seed={run_config.dataset_seed})")

        # Load model
        logger.info("Loading model")
        model = ComponentModel.from_run_info(spd_run).to(device)

        # Load data
        logger.info("Loading dataset")
        load_dataset_kwargs = {}
        if run_config.dataset_streaming:
            logger.info("Using streaming dataset loading")
            load_dataset_kwargs["config_kwargs"] = dict(streaming=True)
            assert task_name == "lm", (
                f"Streaming dataset loading only supported for 'lm' task, got '{task_name = }'"
            )

        batch = load_dataset(
            model_path=run_config.model_path,
            task_name=task_name,
            batch_size=run_config.batch_size,
            seed=run_config.dataset_seed,
            **load_dataset_kwargs,
        ).to(device)

        # Compute activations
        logger.info("Computing activations")
        activations_dict = component_activations(
            model=model,
            batch=batch,
            device=device,
        )

        # Process (concat modules, NO FILTERING)
        logger.info("Processing activations")
        processed = process_activations(
            activations=activations_dict,
            filter_dead_threshold=0.0,  # NO FILTERING
            seq_mode="concat" if task_name == "lm" else None,
            filter_modules=None,
        )

        # Save as single batch to temp dir
        temp_batch_dir = storage.base_dir / "temp_batch"
        temp_batch_dir.mkdir(exist_ok=True)

        single_batch = ActivationBatch(
            activations=processed.activations,
            labels=list(processed.labels),
        )
        single_batch.save(temp_batch_dir / "batch_0.pt")

        batched_activations = BatchedActivations(temp_batch_dir)
        component_labels = processed.labels

        # Log activations to WandB (if enabled)
        if wandb_run is not None:
            logger.info("Plotting activations")
            plot_activations(
                processed_activations=processed,
                save_dir=None,
                n_samples_max=256,
                wandb_run=wandb_run,
            )
            wandb_log_tensor(
                wandb_run,
                processed.activations,
                "activations",
                0,
                single=True,
            )

        # Clean up memory
        del model, batch, activations_dict, processed
        gc.collect()

    # Run merge iteration
    # =====================================
    logger.info("Starting merging")
    log_callback = (
        partial(_log_callback, run=wandb_run, run_config=run_config)
        if wandb_run is not None
        else None
    )

    history = merge_iteration(
        merge_config=run_config.merge_config,
        batched_activations=batched_activations,
        component_labels=component_labels,
        log_callback=log_callback,
    )

    # Save merge history
    history.save(storage.history_path)
    logger.info(f"History saved to {storage.history_path}")

    # Log to WandB
    if wandb_run is not None:
        _log_merge_history_plots(wandb_run, history)
        _save_merge_history_artifact(wandb_run, storage.history_path, history)
        wandb_run.finish()
        logger.info("WandB run finished")

    return storage.history_path
```

---

### Task 4: Batch Precomputation in `run_pipeline.py`
**Files:** `spd/clustering/scripts/run_pipeline.py`
**Dependencies:** Task 1 (needs `ActivationBatch`)
**Estimated Lines:** ~200 lines

#### 4.1 Add Batch Precomputation Function

Add this function before `main()`:

```python
def precompute_batches_for_ensemble(
    pipeline_config: ClusteringPipelineConfig,
    pipeline_run_id: str,
    storage: ClusteringPipelineStorage,
) -> Path | None:
    """
    Precompute activation batches for all runs in ensemble.

    This loads the model ONCE and generates all batches for all runs,
    then saves them to disk. Each clustering run will load batches
    from disk without needing the model.

    Args:
        pipeline_config: Pipeline configuration
        pipeline_run_id: Unique ID for this pipeline run
        storage: Storage paths for pipeline outputs

    Returns:
        Path to base directory containing batches for all runs,
        or None if single-batch mode (recompute_costs_every=1)
    """
    clustering_run_config = ClusteringRunConfig.from_file(
        pipeline_config.clustering_run_config_path
    )

    # Check if multi-batch mode
    recompute_every = clustering_run_config.merge_config.recompute_costs_every
    if recompute_every == 1:
        logger.info("Single-batch mode (recompute_costs_every=1), skipping precomputation")
        return None

    logger.info("Multi-batch mode detected, precomputing activation batches")

    # Load model to determine number of components
    device = get_device()
    spd_run = SPDRunInfo.from_path(clustering_run_config.model_path)
    model = ComponentModel.from_run_info(spd_run).to(device)
    task_name = spd_run.config.task_config.task_name

    # Get number of components (no filtering, so just count from model)
    # Load a sample to count components
    logger.info("Loading sample batch to count components")
    sample_batch = load_dataset(
        model_path=clustering_run_config.model_path,
        task_name=task_name,
        batch_size=clustering_run_config.batch_size,
        seed=0,
    ).to(device)

    with torch.no_grad():
        sample_acts = component_activations(model, device, sample_batch)

    # Count total components across all modules
    n_components = sum(
        act.shape[-1] for act in sample_acts.values()
    )

    # Calculate number of iterations
    n_iters = clustering_run_config.merge_config.get_num_iters(n_components)

    # Calculate batches needed per run
    n_batches_needed = (n_iters + recompute_every - 1) // recompute_every

    logger.info(
        f"Precomputing {n_batches_needed} batches per run for {pipeline_config.n_runs} runs"
    )
    logger.info(f"Total: {n_batches_needed * pipeline_config.n_runs} batches")

    # Create batches directory
    batches_base_dir = storage.base_dir / "precomputed_batches"
    batches_base_dir.mkdir(exist_ok=True)

    # For each run in ensemble
    for run_idx in tqdm(range(pipeline_config.n_runs), desc="Ensemble runs"):
        run_batch_dir = batches_base_dir / f"run_{run_idx}"
        run_batch_dir.mkdir(exist_ok=True)

        # Generate batches for this run
        for batch_idx in tqdm(
            range(n_batches_needed),
            desc=f"  Run {run_idx} batches",
            leave=False
        ):
            # Use unique seed: base_seed + run_idx * 1000 + batch_idx
            # This ensures different data for each run and each batch
            seed = clustering_run_config.dataset_seed + run_idx * 1000 + batch_idx

            # Load data
            batch_data = load_dataset(
                model_path=clustering_run_config.model_path,
                task_name=task_name,
                batch_size=clustering_run_config.batch_size,
                seed=seed,
            ).to(device)

            # Compute activations
            with torch.no_grad():
                acts_dict = component_activations(model, device, batch_data)

            # Process (concat, NO FILTERING)
            processed = process_activations(
                activations=acts_dict,
                filter_dead_threshold=0.0,  # NO FILTERING
                seq_mode="concat" if task_name == "lm" else None,
                filter_modules=None,
            )

            # Save as ActivationBatch
            activation_batch = ActivationBatch(
                activations=processed.activations.cpu(),  # Move to CPU for storage
                labels=list(processed.labels),
            )
            activation_batch.save(run_batch_dir / f"batch_{batch_idx}.pt")

            # Clean up
            del batch_data, acts_dict, processed, activation_batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Clean up model
    del model, sample_batch, sample_acts
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info(f"All batches precomputed and saved to {batches_base_dir}")

    return batches_base_dir
```

#### 4.2 Update `generate_clustering_commands()`

```python
def generate_clustering_commands(
    pipeline_config: ClusteringPipelineConfig,
    pipeline_run_id: str,
    batches_base_dir: Path | None,  # NEW PARAMETER
    dataset_streaming: bool = False,
) -> list[str]:
    """Generate commands for each clustering run.

    Args:
        pipeline_config: Pipeline configuration
        pipeline_run_id: Pipeline run ID (each run will create its own ExecutionStamp)
        batches_base_dir: Path to precomputed batches directory, or None for single-batch mode
        dataset_streaming: Whether to use dataset streaming

    Returns:
        List of shell-safe command strings
    """
    commands = []

    for idx in range(pipeline_config.n_runs):
        cmd_parts = [
            "python",
            "spd/clustering/scripts/run_clustering.py",
            "--config",
            pipeline_config.clustering_run_config_path.as_posix(),
            "--pipeline-run-id",
            pipeline_run_id,
            "--idx-in-ensemble",
            str(idx),
            "--wandb-project",
            str(pipeline_config.wandb_project),
            "--wandb-entity",
            pipeline_config.wandb_entity,
        ]

        # Add precomputed batches path if available
        if batches_base_dir is not None:
            run_batch_dir = batches_base_dir / f"run_{idx}"
            cmd_parts.extend(["--precomputed-activations-dir", str(run_batch_dir)])

        if dataset_streaming:
            cmd_parts.append("--dataset-streaming")

        commands.append(shlex.join(cmd_parts))

    return commands
```

#### 4.3 Update `main()` to Call Precomputation

```python
def main(
    pipeline_config: ClusteringPipelineConfig,
    local: bool = False,
    local_clustering_parallel: bool = False,
    local_calc_distances_parallel: bool = False,
    dataset_streaming: bool = False,
    track_resources_calc_distances: bool = False,
) -> None:
    """Submit clustering runs to SLURM."""

    logger.set_format("console", "terse")

    # Validation
    if local_clustering_parallel or local_calc_distances_parallel or track_resources_calc_distances:
        assert local, (
            "local_clustering_parallel, local_calc_distances_parallel, track_resources_calc_distances "
            "can only be set when running locally"
        )

    # Create ExecutionStamp for pipeline
    execution_stamp = ExecutionStamp.create(
        run_type="ensemble",
        create_snapshot=pipeline_config.create_git_snapshot,
    )
    pipeline_run_id = execution_stamp.run_id
    logger.info(f"Pipeline run ID: {pipeline_run_id}")

    # Initialize storage
    storage = ClusteringPipelineStorage(execution_stamp)
    logger.info(f"Pipeline output directory: {storage.base_dir}")

    # Save pipeline config
    pipeline_config.to_file(storage.pipeline_config_path)
    logger.info(f"Pipeline config saved to {storage.pipeline_config_path}")

    # Create WandB workspace if requested
    if pipeline_config.wandb_project is not None:
        workspace_url = create_clustering_workspace_view(
            ensemble_id=pipeline_run_id,
            project=pipeline_config.wandb_project,
            entity=pipeline_config.wandb_entity,
        )
        logger.info(f"WandB workspace: {workspace_url}")

    # NEW: Precompute batches if multi-batch mode
    batches_base_dir = precompute_batches_for_ensemble(
        pipeline_config=pipeline_config,
        pipeline_run_id=pipeline_run_id,
        storage=storage,
    )

    # Generate commands for clustering runs
    clustering_commands = generate_clustering_commands(
        pipeline_config=pipeline_config,
        pipeline_run_id=pipeline_run_id,
        batches_base_dir=batches_base_dir,  # NEW
        dataset_streaming=dataset_streaming,
    )

    # Generate commands for calculating distances
    calc_distances_commands = generate_calc_distances_commands(
        pipeline_run_id=pipeline_run_id,
        distances_methods=pipeline_config.distances_methods,
    )

    # ... rest of submission logic unchanged
```

---

## Testing Plan

### Unit Tests
**File:** `tests/clustering/test_multi_batch.py` (new)

1. Test `ActivationBatch` save/load
2. Test `BatchedActivations` cycling through batches
3. Test `recompute_coacts_from_scratch` produces correct shapes
4. Test NaN handling in merge pair samplers
5. Test backward compatibility (single batch, `recompute_costs_every=1`)

### Integration Tests

1. **Single-batch mode (backward compatibility):**
   ```python
   config = ClusteringRunConfig(
       precomputed_activations_dir=None,
       merge_config=MergeConfig(recompute_costs_every=1),
       ...
   )
   # Should behave exactly as before
   ```

2. **Multi-batch mode:**
   ```python
   config = ClusteringRunConfig(
       precomputed_activations_dir=Path("batches/"),
       merge_config=MergeConfig(recompute_costs_every=20),
       ...
   )
   # Should use multiple batches
   ```

3. **Ensemble with precomputation:**
   - Run small ensemble (n=3) with multi-batch
   - Verify batches are created correctly
   - Verify clustering runs use precomputed batches

### Manual Testing Checklist

- [ ] Single run, single batch (original behavior)
- [ ] Single run, multi-batch with precomputed dir
- [ ] Ensemble run, single batch mode
- [ ] Ensemble run, multi-batch mode with precomputation
- [ ] Verify NaN masking doesn't break merge sampling
- [ ] Verify memory usage (model not kept during merge)
- [ ] Verify batch cycling works correctly

---

## Summary

**Total Changes:**
- New files: 1 (`spd/clustering/batched_activations.py`)
- Modified files: 5
- Total new code: ~500 lines
- Backward compatible: Yes (defaults to original behavior)

**Key Benefits:**
- Model loaded once, not kept during merge iteration
- Supports arbitrary number of batches
- Simple disk-based storage
- Minimal config changes (2 new fields)
- No complex scheduling or memory management

**Dependencies:** None (PR #227 not required for this simplified version)
