# Magnitude Sweep Analysis

This script creates plots showing how individual neuron activations and causal importance values respond as we gradually increase the magnitude of a one-hot input vector from 0 to max_magnitude.

## Usage

### Using configuration file (recommended):
```bash
python spd/scripts/magnitude_sweep/magnitude_sweep.py spd/scripts/magnitude_sweep/magnitude_sweep_config.yaml
```

### Using command line arguments:
```bash
python spd/scripts/magnitude_sweep/magnitude_sweep.py --model_path="wandb:goodfire/spd/runs/2ki9tfsx" --feature_idx=0 --n_steps=100 --max_magnitude=2.0
```

## Configuration

The script uses a YAML configuration file (`magnitude_sweep_config.yaml`) with the following parameters:

- `model_path`: Path to the trained SPD model (wandb:project/run_id or local path)
- `feature_idx`: Which feature to activate (default: 0)
- `n_steps`: Number of steps from 0 to max_magnitude (default: 100)
- `max_magnitude`: Maximum input magnitude (default: 2.0)
- `ci_threshold`: CI threshold for active gates (default: 0.1)
- `pre_activation`: Show pre-activation values (before ReLU) instead of post-activation (default: false)
- `figsize_per_subplot`: Figure size per subplot (width height) (default: [2, 1.5])
- `dpi`: DPI for figures (default: 150)
- `device`: Device to use (default: "auto")
- `output_dir`: Directory to save results (optional, defaults to 'out' directory relative to script location)

## Output

The script generates unified grid plots showing:
- Individual neuron activations in the ResidMLP layers
- Output responses
- Causal importance function values for gates that actually activate
- Pre-sigmoid gate outputs
- Gate inputs (inner acts)

Results are saved to the `out/` directory by default, or to a custom directory if specified in the configuration.
