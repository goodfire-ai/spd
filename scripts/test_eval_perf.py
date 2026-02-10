"""Test which eval metric is slow in pile_llama_simple_mlp-4L.

Creates config variants with different eval metrics enabled/disabled
and times each one to identify the bottleneck.
"""

import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml

BASE_CONFIG_PATH = Path("spd/experiments/lm/pile_llama_simple_mlp-4L.yaml")
TEST_DIR = Path("/tmp/spd_eval_perf_tests")
TEST_DIR.mkdir(parents=True, exist_ok=True)

# Load base config
base_config = yaml.safe_load(BASE_CONFIG_PATH.read_text())

# Common overrides: disable wandb, run for 1 step only, disable slow_eval_on_first_step
# to test fast evals first, then we can test slow evals separately
COMMON_OVERRIDES = {
    "wandb_project": None,
    "wandb_run_name": None,
    "steps": 1,
    "eval_freq": 1,
    "train_log_freq": 1,
    "save_freq": None,
    # Match per-rank sizes from dp=8: batch_size=64/8=8, eval_batch_size=256/8=32
    "batch_size": 8,
    "eval_batch_size": 32,
}


def make_config(
    name: str,
    eval_metric_overrides: list[dict] | None = None,
    extra_overrides: dict | None = None,
) -> Path:
    """Create a test config variant and return its path."""
    cfg = deepcopy(base_config)
    cfg.update(COMMON_OVERRIDES)
    if eval_metric_overrides is not None:
        cfg["eval_metric_configs"] = eval_metric_overrides
    if extra_overrides:
        cfg.update(extra_overrides)

    path = TEST_DIR / f"{name}.yaml"
    path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    return path


def run_test(name: str, config_path: Path, timeout_s: int = 600) -> tuple[str, float | None, str]:
    """Run a test and return (name, elapsed_seconds_or_None, status)."""
    print(f"\n{'=' * 60}")
    print(f"Running test: {name}")
    print(f"Config: {config_path}")
    print(f"Timeout: {timeout_s}s")
    print(f"{'=' * 60}")

    cmd = [
        sys.executable,
        "spd/experiments/lm/lm_decomposition.py",
        str(config_path),
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout_s,
            capture_output=True,
            text=True,
            cwd="/mnt/polished-lake/home/braun/spd",
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            status = "SUCCESS"
        else:
            status = f"FAILED (rc={result.returncode})"
            # Print last 30 lines of stderr
            stderr_lines = result.stderr.strip().split("\n")
            print("Last 30 lines of stderr:")
            for line in stderr_lines[-30:]:
                print(f"  {line}")
        print(f"  -> {status} in {elapsed:.1f}s")
        return name, elapsed, status
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  -> TIMEOUT after {elapsed:.1f}s")
        return name, None, "TIMEOUT"


# ---- Define test variants ----

# The full eval_metric_configs from the original config
FULL_EVAL_METRICS = base_config["eval_metric_configs"]

# Individual metrics from the config
component_activation_density = {"classname": "ComponentActivationDensity"}
ci_l0 = {
    "classname": "CI_L0",
    "groups": {
        "layer_0": ["h.0.*"],
        "layer_1": ["h.1.*"],
        "layer_2": ["h.2.*"],
        "layer_3": ["h.3.*"],
        "total": ["*"],
    },
}
ce_and_kl = {"classname": "CEandKLLosses", "rounding_threshold": 0.0}
ci_mean_per_component = {"classname": "CIMeanPerComponent"}
stoch_hidden_acts = {"coeff": None, "classname": "StochasticHiddenActsReconLoss"}
pgd_recon = {
    "coeff": None,
    "init": "random",
    "step_size": 0.1,
    "n_steps": 20,
    "mask_scope": "shared_across_batch",
    "classname": "PGDReconLoss",
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test",
        choices=[
            "no_eval",
            "ci_l0_only",
            "ce_and_kl_only",
            "pgd_recon_only",
            "stoch_hidden_only",
            "comp_act_density_only",
            "ci_mean_only",
            "all_fast",
            "all_slow",
            "full",
            "full_small_batch",
            "loss_metrics_only",
            "full_eval64",
            "full_eval128",
            "full_eval256",
        ],
        help="Which test to run",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    args = parser.parse_args()

    test_name = args.test

    match test_name:
        case "no_eval":
            config_path = make_config("no_eval", eval_metric_overrides=[])
        case "ci_l0_only":
            config_path = make_config("ci_l0_only", eval_metric_overrides=[ci_l0])
        case "ce_and_kl_only":
            config_path = make_config("ce_and_kl_only", eval_metric_overrides=[ce_and_kl])
        case "pgd_recon_only":
            config_path = make_config("pgd_recon_only", eval_metric_overrides=[pgd_recon])
        case "stoch_hidden_only":
            config_path = make_config(
                "stoch_hidden_only", eval_metric_overrides=[stoch_hidden_acts]
            )
        case "comp_act_density_only":
            config_path = make_config(
                "comp_act_density_only",
                eval_metric_overrides=[component_activation_density],
            )
        case "ci_mean_only":
            config_path = make_config("ci_mean_only", eval_metric_overrides=[ci_mean_per_component])
        case "all_fast":
            # All non-slow metrics from eval_metric_configs
            config_path = make_config(
                "all_fast",
                eval_metric_overrides=[ci_l0, ce_and_kl, stoch_hidden_acts, pgd_recon],
            )
        case "all_slow":
            # All slow metrics
            config_path = make_config(
                "all_slow",
                eval_metric_overrides=[
                    component_activation_density,
                    ci_mean_per_component,
                ],
            )
        case "full":
            config_path = make_config("full")  # use original eval_metric_configs
        case "full_small_batch":
            config_path = make_config("full_small_batch", extra_overrides={"eval_batch_size": 32})
        case "loss_metrics_only":
            # No explicit eval metrics, but loss metrics still get added
            config_path = make_config("loss_metrics_only", eval_metric_overrides=[])
        case "full_eval64":
            config_path = make_config("full_eval64", extra_overrides={"eval_batch_size": 64})
        case "full_eval128":
            config_path = make_config("full_eval128", extra_overrides={"eval_batch_size": 128})
        case "full_eval256":
            config_path = make_config("full_eval256", extra_overrides={"eval_batch_size": 256})

    result = run_test(test_name, config_path, timeout_s=args.timeout)
    print(f"\n{'=' * 60}")
    print(
        f"RESULT: {result[0]} -> {result[2]} ({result[1]:.1f}s)"
        if result[1]
        else f"RESULT: {result[0]} -> {result[2]}"
    )
    print(f"{'=' * 60}")
