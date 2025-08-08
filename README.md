# SPD - Stochastic Parameter Decomposition
**Note: The [dev](https://github.com/goodfire-ai/spd/tree/dev) branch contains active work from Goodfire and collaborators since this paper's release. This is now an open source
research project. Please feel free to view the issues (or add to them) and make a PR to the dev branch!**

The main branch contains code used in the paper [Stochastic Parameter Decomposition](https://arxiv.org/abs/2506.20790)

Weights and Bias [report](https://wandb.ai/goodfire/spd-tms/reports/SPD-paper-report--VmlldzoxMzE3NzU0MQ) accompanying the paper.

## Installation
From the root of the repository, run one of

```bash
make install-dev  # To install the package, dev requirements and pre-commit hooks
make install  # To just install the package (runs `pip install -e .`)
```

## Usage
Place your wandb information in a .env file. See .env.example for an example.

The repository consists of several `experiments`, each of which containing scripts to run SPD,
analyse results, and optionally a train a target model:
- `spd/experiments/tms` - Toy model of superposition
- `spd/experiments/resid_mlp` - Toy model of compressed computation and toy model of distributed
  representations
- `spd/experiments/lm` - Language model loaded from huggingface.

Note that the `lm` experiment allows for running SPD on any model pulled from huggingface, provided
you only need to decompose nn.Linear or nn.Embedding layers (other layer types would need to be
added).

### Run SPD
SPD can be run by executing any of the `*_decomposition.py` scripts defined in the experiment
subdirectories. A config file is required for each experiment, which can be found in the same
directory. For example:
```bash
python spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_config.yaml
```
will run SPD on TMS with the config file `tms_config.yaml` (which is the main config file used
for the TMS experiments in the paper).

Wandb sweep files are also provided in the experiment subdirectories, and can be run with e.g.:
```bash
wandb sweep spd/experiments/tms/tms_sweep_config.yaml
```

All experiments call the `optimize` function in `spd/run_spd.py`, which contains the main SPD logic.

## Development

Suggested extensions and settings for VSCode/Cursor are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

There are various `make` commands that may be helpful

```bash
make check  # Run pre-commit on all files (i.e. pyright, ruff linter, and ruff formatter)
make type  # Run pyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```
