# Distributed Data Parallel (DDP) Implementation Plan for SPD

## Background

This document outlines the implementation of distributed data parallel training for the SPD codebase using PyTorch's DistributedDataParallel with mpirun as the process launcher. The implementation will start with language model experiments (`spd/experiments/lm/`) and be designed for extension to all experiment types.

### Why mpirun over torchrun?

- Better integration with HPC clusters and SLURM
- More flexible process management
- Standard in many research computing environments
- Cleaner handling of multi-node setups

### Key Requirements

1. **Backward Compatibility**: Single-GPU runs must work without any changes
2. **Global Batch Size**: Effective batch size remains constant regardless of GPU count
3. **Process Coordination**: Only rank 0 performs logging, evaluation, and checkpointing
4. **Multi-node Support**: Support both single-node and multi-node training
5. **Data Distribution**: Each process gets non-overlapping data subsets

## Architecture Overview

### Process Initialization

With mpirun, processes are launched differently than torchrun:
- mpirun starts N processes directly
- Each process determines its rank from MPI environment
- PyTorch's distributed backend connects processes using MPI initialization

### Data Flow

1. Each process loads a subset of data using `split_dataset_by_node`
2. Model is wrapped with DDP after initialization
3. Gradients are automatically synchronized by DDP
4. Only rank 0 saves outputs and logs

## Implementation TODO List

### Phase 1: Core Infrastructure ✓ = completed, ○ = pending

✓ **1.1 Create `spd/utils/distributed.py`**
- [x] Implement MPI-based initialization
- [x] Create rank/world_size getters
- [x] Add distributed state management
- [x] Implement device selection based on local rank
- [x] Add synchronization utilities

✓ **1.2 Update device management**
- [x] Modify `get_device()` in `general_utils.py` to import from distributed utils
- [x] Ensure backward compatibility when not distributed

### Phase 2: Experiment Integration

✓ **2.1 Update `spd/experiments/lm/lm_decomposition.py`**
- [x] Add distributed initialization at start
- [x] Adjust batch size calculation for world_size
- [x] Pass correct rank/world_size to data loaders
- [x] Condition WandB initialization on rank 0
- [x] Add distributed cleanup at end

✓ **2.2 Update `spd/run_spd.py`**
- [x] Add DDP model wrapping logic
- [x] Adjust gradient accumulation for global batch size
- [x] Condition all logging on `is_main_process()`
- [x] Condition evaluation on rank 0
- [x] Condition checkpointing on rank 0
- [x] Add synchronization before optimizer steps

✓ **2.3 Update data loading**
- [x] Verify `create_data_loader` correctly uses ddp_rank and ddp_world_size
- [x] Test data distribution across ranks

### Phase 3: SLURM Integration

○ **3.1 Update `spd/utils/slurm_utils.py`**
- [ ] Add support for multi-GPU allocation
- [ ] Implement mpirun command generation
- [ ] Handle node allocation for multi-node jobs
- [ ] Set up proper MPI environment in SLURM scripts

○ **3.2 Update `spd/scripts/run.py`**
- [ ] Add `--ddp` flag
- [ ] Add `--gpus_per_node` parameter
- [ ] Add `--nodes` parameter
- [ ] Modify command generation for DDP jobs
- [ ] Update SLURM script creation with DDP parameters

### Phase 4: Configuration

✓ **4.1 Update `spd/configs.py`**
- [x] Add `ddp_enabled` field
- [x] Add `ddp_backend` field (default: "nccl")
- [x] Add `ddp_find_unused_parameters` field
- [x] Ensure backward compatibility

○ **4.2 Update experiment configs**
- [ ] Add DDP fields to existing YAML configs with defaults
- [ ] Create example DDP config variants

### Phase 5: Testing

✓ **5.1 Create unit tests**
- [x] Test distributed utilities
- [x] Test rank/world_size detection
- [x] Test data distribution
- [x] Test backward compatibility

✓ **5.2 Create integration tests**
- [x] Single-node 2-GPU test (manual_test_ddp.py)
- [ ] Single-node 4-GPU test
- [ ] Multi-node test (if possible)
- [ ] Verify results match single-GPU baseline

✓ **5.3 Create test scripts**
- [x] Local mpirun test script (run_ddp_test.sh)
- [ ] SLURM submission test script
- [ ] Validation script to compare DDP vs single-GPU results

### Phase 6: Extension to Other Experiments

○ **6.1 Create reusable DDP wrapper**
- [ ] Abstract common DDP initialization pattern
- [ ] Create base class or mixin for DDP experiments

○ **6.2 Extend to TMS experiments**
- [ ] Update `tms_decomposition.py`
- [ ] Test with synthetic data

○ **6.3 Extend to ResidMLP experiments**
- [ ] Update `resid_mlp_decomposition.py`
- [ ] Test with appropriate data

### Phase 7: Documentation

○ **7.1 Update README**
- [ ] Add DDP usage examples
- [ ] Document mpirun requirements
- [ ] Add troubleshooting section

○ **7.2 Update CLAUDE.md**
- [ ] Add DDP development guidelines
- [ ] Document testing procedures

## Implementation Details

### MPI-based Initialization

```python
def init_distributed() -> tuple[int, int, int]:
    """Initialize distributed process group using MPI."""
    import os
    
    # Check if running under MPI
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        # OpenMPI
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    elif 'MV2_COMM_WORLD_SIZE' in os.environ:
        # MVAPICH2
        world_size = int(os.environ['MV2_COMM_WORLD_SIZE'])
        rank = int(os.environ['MV2_COMM_WORLD_RANK'])
        local_rank = int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
    elif 'SLURM_NTASKS' in os.environ:
        # SLURM with srun
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
    else:
        # Not distributed
        return 0, 1, 0
    
    # Initialize PyTorch distributed
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set CUDA device
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank
```

### Batch Size Calculation

```python
# Global batch size remains constant
global_batch_size = config.microbatch_size * config.gradient_accumulation_steps

# Per-process batch size
per_process_batch_size = config.microbatch_size // world_size

# Adjust gradient accumulation if needed
if per_process_batch_size == 0:
    per_process_batch_size = 1
    # Increase gradient accumulation to maintain global batch size
    gradient_accumulation_steps = global_batch_size // world_size
else:
    gradient_accumulation_steps = config.gradient_accumulation_steps
```

### SLURM Script with mpirun

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=72:00:00

# Load MPI module (cluster-specific)
module load openmpi/4.1.0

# Run with mpirun
mpirun -np $SLURM_NTASKS \
    --bind-to none \
    --map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH \
    python spd/experiments/lm/lm_decomposition.py config.yaml
```

## Testing Strategy

### Local Testing (2 GPUs)
```bash
mpirun -np 2 \
    -x CUDA_VISIBLE_DEVICES=0,1 \
    python spd/experiments/lm/lm_decomposition.py config.yaml
```

### Validation Script
```python
# Compare single-GPU vs DDP results
# 1. Run single-GPU baseline
# 2. Run 2-GPU DDP
# 3. Compare final losses and metrics
# 4. Verify checkpoints are compatible
```

## Common Issues and Solutions

1. **MPI not found**: Ensure MPI is loaded in SLURM script
2. **NCCL errors**: Check GPU visibility and MPI binding
3. **Hanging processes**: Ensure all ranks reach synchronization points
4. **Different results**: Check random seed adjustment per rank

## Success Metrics

- [ ] Single-GPU compatibility maintained
- [ ] Linear speedup with GPU count (>85% efficiency)
- [ ] Identical convergence to single-GPU baseline
- [ ] Clean process management and error handling
- [ ] Working multi-node training