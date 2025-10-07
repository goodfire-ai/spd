# Causal Importance Sweep Analysis

This script creates plots showing how each component's causal importance responds as we gradually increase one input dimension from 0 to max_magnitude.

## Usage

### Using configuration file:
```bash
python spd/scripts/causal_importance_sweep/causal_importance_sweep.py spd/scripts/causal_importance_sweep/causal_importance_sweep_config.yaml
```


## Configuration

The script uses a YAML configuration file (`causal_importance_sweep_config.yaml`) with the following parameters:

- `model_path`: Path to the trained SPD model (wandb:project/run_id or local path)
- `feature_idx`: Which feature to activate (default: 0)
- `n_steps`: Number of steps from 0 to max_magnitude (default: 50)
- `max_magnitude`: Maximum input magnitude (default: 1.0)
- `ci_threshold`: CI threshold for active gates (default: 0.1)
- `figsize`: Figure size (width height) (default: [12, 8])
- `dpi`: DPI for figures (default: 150)
- `device`: Device to use (default: "auto")
- `output_dir`: Directory to save results (optional, defaults to 'out' directory relative to script location)

## Output

The script generates layer-based plots showing:
- How causal importance values change as input magnitude increases
- Component-specific causal importance curves
- Analysis of which components are most responsive to input changes

Results are saved to the `out/` directory by default, or to a custom directory if specified in the configuration.
