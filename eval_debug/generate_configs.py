"""Generate test configs for eval metric debugging.

Creates variants of pile_llama_simple_mlp-4L with different eval_metric_configs
to isolate which metric is causing slow evaluations.
"""

import copy
from pathlib import Path

import yaml

BASE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "spd/experiments/lm/pile_llama_simple_mlp-4L.yaml"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "configs"


# The eval_metric_configs from the base config, for reference:
# - ComponentActivationDensity (slow=True)
# - CI_L0 (slow=False)
# - CEandKLLosses (slow=False)
# - CIMeanPerComponent (slow=True)
# - StochasticHiddenActsReconLoss (slow=False)
# - PGDReconLoss (slow=False, n_steps=20)

CHEAP_EVALS = [
    {"classname": "ComponentActivationDensity"},
    {
        "classname": "CI_L0",
        "groups": {
            "layer_0": ["h.0.*"],
            "layer_1": ["h.1.*"],
            "layer_2": ["h.2.*"],
            "layer_3": ["h.3.*"],
            "total": ["*"],
        },
    },
    {"classname": "CIMeanPerComponent"},
]

CEKL = [
    {"classname": "CEandKLLosses", "rounding_threshold": 0.0},
]

STOCH_HIDDEN = [
    {"classname": "StochasticHiddenActsReconLoss", "coeff": None},
]

PGD = [
    {
        "classname": "PGDReconLoss",
        "coeff": None,
        "init": "random",
        "step_size": 0.1,
        "n_steps": 20,
        "mask_scope": "shared_across_batch",
    },
]

TESTS: dict[str, list | None] = {
    "00_no_eval_metrics": [],
    "01_cheap_only": CHEAP_EVALS,
    "02_cekl_only": CEKL,
    "03_stoch_hidden_only": STOCH_HIDDEN,
    "04_pgd_only": PGD,
    "05_all_eval_metrics": None,  # Keep original eval_metric_configs
}


def main():
    with open(BASE_CONFIG_PATH) as f:
        base = yaml.safe_load(f)

    OUTPUT_DIR.mkdir(exist_ok=True)

    for name, eval_configs in TESTS.items():
        config = copy.deepcopy(base)
        config["steps"] = 5
        # Ensure wandb is disabled
        config.pop("wandb_project", None)
        config["save_freq"] = None

        if eval_configs is not None:
            config["eval_metric_configs"] = eval_configs

        out_path = OUTPUT_DIR / f"{name}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Created: {out_path}")

    print(f"\nGenerated {len(TESTS)} configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
