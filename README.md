# SPD - Stochastic Parameter Decomposition

**Note: The [spd-paper](https://github.com/goodfire-ai/spd/tree/spd-paper) branch contains code used in the paper [Stochastic Parameter Decomposition](https://arxiv.org/abs/2506.20790). The main branch contains active work from Goodfire and collaborators since this paper's release. This is now an open source
research project. Please feel free to view the issues (or add to them) and make a PR!**

Weights and Bias [report](https://wandb.ai/goodfire/spd-tms/reports/SPD-paper-report--VmlldzoxMzE3NzU0MQ) accompanying the paper.

## Installation
From the root of the repository, run one of

```bash
make install-dev  # To install the package, dev requirements, pre-commit hooks, and create user files
make install  # To install the package (runs `pip install -e .`) and create user files
```

## Usage
Place your wandb information in a .env file. See .env.example for an example.

The repository consists of several `experiments`, each of which containing scripts to run SPD,
analyse results, and optionally a train a target model:
- `spd/experiments/tms` - Toy model of superposition
- `spd/experiments/resid_mlp` - Toy model of compressed computation and toy model of distributed
  representations
- `spd/experiments/ih` - Small model trained on a toy induction head task.
- `spd/experiments/lm` - Language model loaded from huggingface.

Note that the `lm` experiment allows for running SPD on any model from huggingface or that is
accessible to the environment, provided that you only need to decompose `nn.Linear`, `nn.Embedding`
or `transformers.modeling_utils.Conv1D` layers (other layer types are not yet supported, though
these should cover most modules). See `spd/experiments/lm/ss_gpt2_config.yaml` for an example which
loads from huggingface and `spd/experiments/lm/ss_gpt2_simple_config.yaml` for an example which
loads from https://github.com/goodfire-ai/simple_stories_train (with the model weights saved on
wandb).

### Run SPD

The unified `spd-run` command provides a single entry point for running SPD experiments, supporting
fixed configurations, parameter sweeps, and evaluation runs. All runs are tracked in W&B with
workspace views created for each experiment.

For full CLI usage and more examples, run `spd-run --help` or see `scripts/run.py`.

Below we show a small subset of the functionality provided in the CLI:
```bash
spd-run --experiments tms_5-2                    # Run a specific experiment
spd-run --experiments tms_5-2,resid_mlp1         # Run multiple experiments
spd-run                                          # Run all experiments
```

For running hyperparameter sweeps:

```bash
spd-run --experiments <experiment_name> --sweep --n-agents <n-agents> [--cpu]
```

**Sweep parameters:**
- Default sweep parameters are loaded from `spd/scripts/sweep_params.yaml`
- You can specify a custom sweep parameters file by passing its path to `--sweep`
- Sweep parameters support both experiment-specific and global configurations:
  ```yaml
  # Params used for all experiments
  global:
    seed:
      values: [0, 1]
    loss_metric_configs:
      - classname: "ImportanceMinimalityLoss"
        coeff:
          values: [0.1, 0.2]

  # Experiment-specific parameters (override global)
  tms_5-2:
    seed:
      values: [100, 200]  # Overrides global seed
    task_config:
      feature_probability:
        values: [0.05, 0.1]
  ```

Note that SPD can also be run by executing any of the `*_decomposition.py` scripts defined in the experiment
subdirectories, along with a corresponding config file. E.g.
```bash
python spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_5-2_config.yaml
```

### Model Comparison

Use the model comparison script to analyse (post hoc) the geometric similarities between subcomponents of two different models:

```bash
python spd/scripts/compare_models/compare_models.py spd/scripts/compare_models/compare_models_config.yaml
```

See `spd/scripts/compare_models/README.md` for detailed usage instructions.

## Development

Suggested extensions and settings for VSCode/Cursor are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

### Contributing

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project, including pull request requirements and review processes.

### Development Commands

There are various `make` commands that may be helpful.

```bash
make check  # Run pre-commit on all files (i.e. basedpyright, ruff linter, and ruff formatter)
make type  # Run basedpyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```
