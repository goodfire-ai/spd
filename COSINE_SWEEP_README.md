# Cosine Similarity Sweep

This sweep tests different cosine similarity coefficients for different C values on the TMS 5-2 model.

## Files Created

1. **`tms_5-2_cosine_sweep_config.yaml`** - Base configuration for the sweep
2. **`cosine_similarity_sweep_params.yaml`** - Sweep parameters defining the parameter combinations
3. **`spd/registry.py`** - Updated to include the new experiment

## Sweep Parameters

The sweep will test the following combinations:

- **C values**: [10, 20, 30] (number of subcomponents)
- **Cosine similarity coefficients**: [1e-4, 1e-3, 1e-2]
- **Total combinations**: 3 × 3 = 9 experiments

## How to Run the Sweep

### Option 1: Using the unified runner (recommended)
```bash
spd-run --experiments tms_5-2_cosine_sweep --sweep cosine_similarity_sweep_params.yaml --n_agents 9
```

### Option 2: Using the unified runner with custom project
```bash
spd-run --experiments tms_5-2_cosine_sweep --sweep cosine_similarity_sweep_params.yaml --n_agents 9 --project cosine-sweep
```

### Option 3: Manual execution (for testing individual runs)
```bash
# Test a single combination manually
python spd/experiments/tms/tms_decomposition.py tms_5-2_cosine_sweep_config.yaml
```

## Expected Results

The sweep will create 9 different experiments, each testing:
- How the modified cosine similarity loss (only penalizing negative similarities) performs
- How different C values affect the decomposition
- How different cosine similarity coefficients affect the training dynamics

## Monitoring

- **WandB Project**: `spd` (or custom project if specified)
- **Run Prefix**: `cosine_sweep_`
- **Expected Runtime**: ~8 minutes per experiment
- **Total Runtime**: ~72 minutes for all 9 experiments

## Key Metrics to Watch

1. **`loss/cosine_similarity`** - Should converge to 0.0 when orthogonality is achieved
2. **`linear1/ci_l0`** and **`linear2/ci_l0`** - Sparsity metrics
3. **`loss/total`** - Overall training loss
4. **Cosine similarity histograms** - Visual representation of subcomponent alignment

## Analysis

After the sweep completes, you can compare:
- How quickly different C values achieve orthogonality
- How different coefficients affect the final decomposition quality
- Whether higher C values lead to better or worse sparsity
- The trade-off between orthogonality and reconstruction quality 