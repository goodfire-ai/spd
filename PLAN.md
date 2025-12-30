# Plan: Diagnose Slow DDP Startup

## Goal
Identify what's causing the ~5 minute delay when starting distributed training with `dp>1` on LM experiments.

## Approach
Create a series of diagnostic scripts that incrementally add complexity, profiling each operation. Start with the simplest possible distributed operation and gradually add more of the actual codebase until we find the bottleneck.

## Test Configuration
- **Nodes:** 2
- **GPUs per node:** 8 (16 total, respecting cluster limits)
- **Backend:** NCCL (GPU) - matching production setup

## Diagnostic Phases

### Phase 1: Bare Minimum Distributed Init
**Script:** `scripts/debug/phase1_minimal.py`

What it tests:
- `torch.distributed.init_process_group()` with NCCL backend
- Simple all_reduce of a tensor
- `dist.barrier()`

Expected time: Should be <10 seconds if NCCL is healthy.

### Phase 2: SPD Distributed Utils
**Script:** `scripts/debug/phase2_spd_init.py`

What it tests:
- `init_distributed()` from `spd/utils/distributed_utils.py`
- `sync_across_processes()` barrier

Comparison: If significantly slower than Phase 1, the issue is in our distributed utils.

### Phase 3: Model Loading
**Script:** `scripts/debug/phase3_model_loading.py`

What it tests:
- Phase 2 + pretrained model loading via `from_pretrained()`
- Test with `ensure_cached_and_call` vs direct loading
- Use actual config: `ss_llama_simple_mlp-1L` (small 4-layer model)

This will tell us if model loading is the bottleneck.

### Phase 4: Dataset Loading
**Script:** `scripts/debug/phase4_dataset.py`

What it tests:
- Phase 3 + `load_dataset()` from HuggingFace
- `AutoTokenizer.from_pretrained()`
- `create_data_loader()` function

HuggingFace operations can be surprisingly slow with concurrent access.

### Phase 5: Full Initialization (No Training)
**Script:** `scripts/debug/phase5_full_init.py`

What it tests:
- Everything up to and including DDP wrapping
- Stops just before the training loop
- Mirrors `lm_decomposition.py` initialization

This should reproduce the 5-minute delay if all prior phases are fast.

## Output Format
Each script will print:
```
[RANK X] Phase N - Operation: <name>
[RANK X] Time: <seconds>s
```

At the end, rank 0 will print a summary table of all timings.

## Execution Plan

1. Create the scripts in `spd/scripts/debug/`
2. Create a launcher script that submits SLURM jobs for each phase
3. Run phases sequentially, analyze results
4. If a phase is slow, add more granular timing within that phase

## Configuration (Confirmed)

- **Test config:** `ss_llama_simple_mlp-1L`
- **Dataset:** Already cached
- **Execution:** One phase at a time with review between phases
- **Launcher:** Standalone bash scripts (not spd-run)

## Scripts Created

All scripts are in `spd/scripts/debug/`:

| Script | Description |
|--------|-------------|
| `phase1_minimal.py` | Bare minimum NCCL init + all_reduce + barrier |
| `phase2_spd_init.py` | SPD's init_distributed() + sync_across_processes() |
| `phase3_model_loading.py` | Model loading with ensure_cached_and_call |
| `phase4_dataset.py` | Dataset and tokenizer loading |
| `phase5_full_init.py` | Full lm_decomposition.py initialization |
| `submit_phase.sh` | SLURM launcher script |

## Usage

```bash
cd /mnt/polished-lake/home/braun/spd-main/spd/scripts/debug

# Submit Phase 1 on 2 nodes (16 GPUs)
./submit_phase.sh 1 2

# Monitor output
tail -f ~/slurm_logs/slurm-<job_id>.out

# After Phase 1 completes, submit Phase 2
./submit_phase.sh 2 2

# Continue with phases 3-5 as needed
```

## Expected Results

- **Phase 1**: Should complete in <30 seconds if NCCL is healthy
- **Phase 2**: Should be similar to Phase 1 (minimal overhead from SPD utils)
- **Phase 3**: Model loading - should be fast for cached 4-layer model
- **Phase 4**: Dataset loading - should be fast for cached dataset
- **Phase 5**: Full init - if this is slow and prior phases are fast, the issue is in the combination

If a phase is slow, we'll add more granular timing within that phase to pinpoint the exact bottleneck.
