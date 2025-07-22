# All Models Cosine Similarity Evaluation

This setup allows you to run comprehensive evaluations of the cosine similarity loss across all available models in the SPD registry.

## 📋 Updated Models

All model configs have been updated to include:
- **`cosine_similarity_coeff: 1e-3`** - Cosine similarity loss coefficient
- **`cosine_similarity_histograms`** - Added to figures_fns for visualization

### TMS Models
- `tms_5-2` - TMS with 5 features, 2 hidden dimensions
- `tms_5-2-id` - TMS with 5 features, 2 hidden dimensions (fixed identity)
- `tms_40-10` - TMS with 40 features, 10 hidden dimensions  
- `tms_40-10-id` - TMS with 40 features, 10 hidden dimensions (fixed identity)

### ResidMLP Models
- `resid_mlp1` - ResidMLP with 1 layer
- `resid_mlp2` - ResidMLP with 2 layers
- `resid_mlp3` - ResidMLP with 3 layers

## 🚀 How to Run Evaluations

### Option 1: Run All Models (Recommended)
```bash
spd-run --sweep all_models_cosine_eval_sweep_params.yaml --n_agents 21
```

This will run:
- 4 TMS models × 3 coefficients = 12 experiments
- 3 ResidMLP models × 3 coefficients = 9 experiments
- **Total: 21 experiments**

### Option 2: Run Specific Model Types
```bash
# Run only TMS models
spd-run --experiments tms_5-2,tms_5-2-id,tms_40-10,tms_40-10-id --sweep all_models_cosine_eval_sweep_params.yaml --n_agents 12

# Run only ResidMLP models
spd-run --experiments resid_mlp1,resid_mlp2,resid_mlp3 --sweep all_models_cosine_eval_sweep_params.yaml --n_agents 9
```

### Option 3: Run Individual Models
```bash
# Test a single model
spd-run --experiments tms_5-2 --sweep all_models_cosine_eval_sweep_params.yaml --n_agents 3

# Test with custom project
spd-run --experiments tms_5-2 --sweep all_models_cosine_eval_sweep_params.yaml --n_agents 3 --project cosine-eval
```

## 📊 Expected Results

### Cosine Similarity Coefficients Tested
- **1e-4**: Very weak penalty
- **1e-3**: Moderate penalty (default)
- **1e-2**: Strong penalty

### Key Metrics to Monitor
1. **`loss/cosine_similarity`** - Should decrease over time (encouraging alignment)
2. **`loss/total`** - Overall training loss
3. **`linear1/ci_l0`** and **`linear2/ci_l0`** - Sparsity metrics
4. **Cosine similarity histograms** - Visual representation of alignment

## ⏱️ Expected Runtime

- **TMS models**: ~4-5 minutes each
- **ResidMLP models**: ~3-60 minutes each (depending on complexity)
- **Total sweep**: ~2-3 hours for all 21 experiments

## 🔍 Analysis Questions

After the evaluations complete, you can compare:

1. **Model Complexity**: How does cosine similarity loss scale with model size?
2. **Coefficient Sensitivity**: Which coefficient works best for each model type?
3. **Sparsity Trade-offs**: How does cosine similarity affect sparsity vs. faithfulness?
4. **Convergence**: How quickly do different models achieve good cosine similarity?

## 📈 Monitoring

- **WandB Project**: `spd` (or custom project if specified)
- **Run Prefix**: Auto-generated based on experiment name
- **Key Plots**: Cosine similarity histograms for each model layer

## 🛠️ Customization

To test different coefficients or add more models:

1. **Edit sweep parameters**: Modify `all_models_cosine_eval_sweep_params.yaml`
2. **Add new models**: Update `spd/registry.py` and create config files
3. **Change coefficients**: Modify the `cosine_similarity_coeff` values in the sweep file

## 📝 Notes

- All models now use the **reverted cosine similarity loss** (encourages positive alignment)
- Cosine similarity histograms are generated for all experiments
- The sweep tests 3 different coefficient strengths for each model
- Results will be logged to WandB for easy comparison and analysis 