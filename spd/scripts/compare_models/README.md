# Model Comparison Script

This directory contains the model comparison script for geometric similarity analysis.

## Files

- `compare_models.py` - Main script for comparing two SPD models
- `compare_models_config.yaml` - Default configuration file
- `out/` - Output directory (created automatically when script runs)

## Usage

```bash
# Using config file
python spd/scripts/compare_models/compare_models.py spd/scripts/compare_models/compare_models_config.yaml

# Using command line arguments
python spd/scripts/compare_models/compare_models.py --current_model_path="wandb:..." --reference_model_path="wandb:..."
```

## Output

Results are saved to the `out/` directory relative to this script's location, ensuring consistent output placement regardless of where the script is invoked from.
