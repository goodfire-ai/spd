# Forward Plot Script

This directory contains the forward plot script for generating forward pass visualizations of SPD models.

## Files

- `forward_plot.py` - Main script for generating forward pass plots
- `forward_plot_config.yaml` - Default configuration file
- `out/` - Output directory (created automatically when script runs)

## Usage

```bash
# Using config file
python spd/scripts/forward_plot/forward_plot.py spd/scripts/forward_plot/forward_plot_config.yaml

# Using default config file (if no argument provided)
python spd/scripts/forward_plot/forward_plot.py
```

## Configuration

The script uses a YAML configuration file with the following parameters:

- `model_path`: Path to the trained SPD model (supports wandb: and local paths)
- `figsize`: Figure size as [width, height]
- `dpi`: DPI for the output figures
- `device`: Device to use (auto, cpu, cuda, etc.)
- `output_dir`: Optional custom output directory

## Output

Results are saved to the `out/` directory relative to this script's location, with a subdirectory named after the model ID. The forward pass plot is saved as `forward_plot.png`.