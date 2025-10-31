import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import torch

from spd.configs import Config, MaskScope, PGDReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.identity_insertion import insert_identity_operations_
from spd.interfaces import LoadableModule
from spd.metrics.pgd_masked_recon_loss import PGDReconLoss
from spd.models.component_model import (
    ComponentModel,
    OutputWithCache,
    SPDRunInfo,
    handle_deprecated_state_dict_keys_,
)
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import (
    extract_batch_data,
    replace_pydantic_model,
    resolve_class,
    set_seed,
)

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


class ModelSetup:
    """Container for loaded model and config."""

    def __init__(
        self,
        model: ComponentModel,
        config: Any,
        device: Any,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device


def load_model(run_info_path: str, target_module_patterns: list[str] | None) -> ModelSetup:
    """Load model. Call once per model type."""

    device = get_device()

    # Load run info and validate config
    run_info = SPDRunInfo.from_path(run_info_path)
    config = run_info.config
    task_config = config.task_config

    assert isinstance(task_config, LMTaskConfig), "task_config not LMTaskConfig"

    # Load model
    # Load the target model
    model_class = resolve_class(config.pretrained_model_class)
    if config.pretrained_model_name is not None:
        assert hasattr(model_class, "from_pretrained"), (
            f"Model class {model_class} should have a `from_pretrained` method"
        )
        target_model = model_class.from_pretrained(config.pretrained_model_name)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        assert issubclass(model_class, LoadableModule), (
            f"Model class {model_class} should be a subclass of LoadableModule which "
            "defines a `from_pretrained` method"
        )
        assert run_info.config.pretrained_model_path is not None
        target_model = model_class.from_pretrained(run_info.config.pretrained_model_path)

    target_model.eval()
    target_model.requires_grad_(False)

    if config.identity_module_patterns is not None:
        insert_identity_operations_(target_model, identity_patterns=config.identity_module_patterns)

    target_module_patterns = target_module_patterns or config.all_module_patterns
    comp_model = ComponentModel(
        target_model=target_model,
        target_module_patterns=target_module_patterns,
        C=config.C,
        ci_fn_hidden_dims=config.ci_fn_hidden_dims,
        ci_fn_type=config.ci_fn_type,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
        sigmoid_type=config.sigmoid_type,
    )

    comp_model_weights = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=True)

    handle_deprecated_state_dict_keys_(comp_model_weights)

    comp_model.load_state_dict(comp_model_weights, strict=False)  # TODO: Remove strict=False
    comp_model.to(device)
    comp_model.target_model.requires_grad_(False)
    # model.target_model.requires_grad_(False)
    return ModelSetup(comp_model, config, device)
    # model.target_model.requires_grad_(False)
    # model = ComponentModel.from_run_info(run_info)
    # model.to(device)
    #
    # return ModelSetup(model, config, device)


def run_experiment(
    model_setup: ModelSetup,
    seed: int,
    mask_scope: MaskScope,
    n_batches: int,
    batch_sizes: list[int],
    step_size: float,
    n_steps: int,
    max_seq_len: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run PGD reconstruction loss experiment for a single seed using pre-loaded model."""
    model = model_setup.model
    config: Config = model_setup.config

    # update the config to set use_delta_component to False
    # config = replace_pydantic_model(config, {"use_delta_component": False})
    device = model_setup.device

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig), "task_config not LMTaskConfig"

    # Run experiments for each batch size
    results: dict[str, float] = {}
    pgd_info: dict[str, Any] = {}

    for batch_size in batch_sizes:
        set_seed(seed)

        # Update task config with desired max_seq_len
        updated_task_config = replace_pydantic_model(
            task_config, {"max_seq_len": max_seq_len, "train_data_split": "train"}
        )

        train_data_config = DatasetConfig(
            name=updated_task_config.dataset_name,
            hf_tokenizer_path=config.tokenizer_name,
            # split=updated_task_config.train_data_split,
            split="train[:100]",
            n_ctx=updated_task_config.max_seq_len,
            is_tokenized=updated_task_config.is_tokenized,
            streaming=updated_task_config.streaming,
            column_name=updated_task_config.column_name,
            shuffle_each_epoch=updated_task_config.shuffle_each_epoch,
            seed=None,
        )

        data_loader, _tokenizer = create_data_loader(
            dataset_config=train_data_config,
            batch_size=batch_size,
            buffer_size=updated_task_config.buffer_size,
            global_seed=seed,
            ddp_rank=0,
            ddp_world_size=1,
        )

        pgd_config = PGDReconLossConfig(
            init="random",
            step_size=step_size,
            n_steps=n_steps,
            mask_scope=mask_scope,
        )

        pgd_recon = PGDReconLoss(
            model=model,
            device=device,
            output_loss_type=config.output_loss_type,
            pgd_config=pgd_config,
            use_delta_component=config.use_delta_component,
        )

        data_loader_iter = iter(data_loader)
        weight_deltas = model.calc_weight_deltas()

        for i in range(n_batches):
            try:
                batch = extract_batch_data(next(data_loader_iter)).to(device)
            except StopIteration:
                print(
                    f"Depleted the dataloader for seed {seed}, batch {i} of {n_batches}, stopping."
                )
                break
            target_model_output: OutputWithCache = model(batch, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=target_model_output.cache,
                detach_inputs=False,
                sampling=config.sampling,
            )
            pgd_info = pgd_recon.update(
                batch=batch,
                target_out=target_model_output.output,
                ci=ci,
                weight_deltas=weight_deltas,
                _tokenizer=_tokenizer,
            )

        loss = pgd_recon.compute().item()
        results[f"batch_size_{batch_size}"] = loss

    return results, pgd_info


def save_results(results: dict[str, dict[str, dict[str, float]]], results_path: Path) -> None:
    """Save results to JSON file."""
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


def load_results(results_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load results from JSON file."""
    with open(results_path) as f:
        results = json.load(f)
    print(f"Results loaded from {results_path}")
    return results


def plot_results(
    results: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
    log_scale: bool = True,
) -> None:
    """Plot results with subplots for each model."""
    n_models = len(results)
    if n_models == 0:
        print("No results to plot")
        return

    # Determine grid size dynamically to minimize empty space
    if n_models == 1:
        n_cols = 1
        n_rows = 1
    elif n_models <= 2:
        n_cols = 2
        n_rows = 1
    elif n_models <= 4:
        n_cols = 2
        n_rows = 2
    else:
        n_cols = 3
        n_rows = (n_models + 2) // 3

    fig_width = 7 * n_cols
    fig_height = 5 * n_rows

    if n_models == 1:
        _, ax = plt.subplots(figsize=(fig_width, fig_height))
        axes = [ax]
    else:
        _, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

    model_names = list(results.keys())
    colormap = matplotlib.colormaps.get_cmap("tab10")
    colors = [colormap(i / 6.0) for i in range(6)]  # 6 seeds

    for idx, model_name in enumerate(model_names):
        if idx >= len(axes):
            break

        ax = axes[idx]
        model_data = results[model_name]

        for seed in range(6):
            seed_key = f"seed_{seed}"
            if seed_key not in model_data:
                continue

            seed_data = model_data[seed_key]
            # Extract batch sizes and losses
            batch_sizes = sorted([int(k.split("_")[-1]) for k in seed_data])
            losses = [seed_data[f"batch_size_{bs}"] for bs in batch_sizes]

            ax.plot(
                batch_sizes,
                losses,
                label=f"Seed {seed}",
                color=colors[seed],
                alpha=0.7,
                marker="o",
                markersize=4,
            )

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model_name} {output_path.stem}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_xscale("log")

    # Hide unused subplots
    for idx in range(len(model_names), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_from_file(
    results_path: Path | str,
    output_path: Path | str | None = None,
    log_scale: bool = True,
) -> None:
    """Load results from file and plot them."""
    results_path = Path(results_path)
    results = load_results(results_path)

    output_path = results_path.with_suffix(".png") if output_path is None else Path(output_path)

    plot_results(results, output_path, log_scale=log_scale)


def save_per_token_results(
    results: dict[str, dict[str, dict[str, list[float]]]], results_path: Path
) -> None:
    """Save per-token results to JSON file."""
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Per-token results saved to {results_path}")


def load_per_token_results(
    results_path: Path,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Load per-token results from JSON file."""
    with open(results_path) as f:
        results = json.load(f)
    print(f"Per-token results loaded from {results_path}")
    return results


def plot_per_token_results(
    results: dict[str, dict[str, dict[str, list[float]]]], output_path: Path
) -> None:
    """For each model, create subplots per seed with two lines for mask scopes."""
    for model_name, model_data in results.items():
        seed_keys = sorted(
            list(model_data.keys()),
            key=lambda s: int(s.split("_")[-1]) if s.startswith("seed_") else 0,
        )

        n_seeds = len(seed_keys)
        if n_seeds == 0:
            print(f"No per-token results to plot for model {model_name}")
            continue

        if n_seeds == 1:
            n_cols = 1
            n_rows = 1
        elif n_seeds <= 2:
            n_cols = 2
            n_rows = 1
        elif n_seeds <= 4:
            n_cols = 2
            n_rows = 2
        else:
            n_cols = 3
            n_rows = (n_seeds + 2) // 3

        fig_width = 7 * n_cols
        fig_height = 5 * n_rows

        if n_seeds == 1:
            _, ax = plt.subplots(figsize=(fig_width, fig_height))
            axes = [ax]
        else:
            _, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()

        cmap = matplotlib.colormaps.get_cmap("tab10")
        mask_colors: dict[str, Any] = {
            "unique_per_datapoint": cmap(0.1),
            "shared_across_batch": cmap(0.6),
        }

        for idx, seed_key in enumerate(seed_keys):
            if idx >= len(axes):
                break
            ax = axes[idx]
            seed_data = model_data[seed_key]

            for mask_key in ["unique_per_datapoint", "shared_across_batch"]:
                if mask_key not in seed_data:
                    continue
                y = seed_data[mask_key]
                if len(y) > 0 and isinstance(y[0], list):
                    y = [v for sub in y for v in (sub if isinstance(sub, list) else [sub])]
                x = list(range(len(y)))
                ax.plot(
                    x,
                    y,
                    label=mask_key,
                    color=mask_colors[mask_key],
                    alpha=0.9,
                    marker="o",
                    markersize=3,
                )

            ax.set_xlabel("Token index")
            ax.set_ylabel("Per-token loss")
            ax.set_title(f"{model_name} {seed_key}")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(len(seed_keys), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        print(f"Per-token plot saved to {output_path.with_suffix('.png')}")
        plt.close()


def main() -> None:
    """Run experiments for all models and seeds, generate report."""
    model_configs: dict[str, str] = {
        "giuseppe": "wandb:goodfire/spd/runs/lxs77xye",
    }

    # target_module_patterns = ["model.layers.0.self_attn.v_proj"]
    target_module_patterns = ["model.layers.0.mlp.gate_proj.pre_identity"]
    mask_scopes: list[MaskScope] = ["shared_across_batch", "unique_per_datapoint"]
    n_batches = 1
    step_size = 0.1
    n_steps = 50
    seeds = list(range(6))
    # seeds = [0, 1, 2, 3]
    # batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # batch_sizes = [8]
    batch_sizes = [32]
    # batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    # n_ctx = 512
    # n_ctx = 8
    # n_ctx = 32
    n_ctx = 1
    per_token_results: dict[str, dict[str, dict[str, list[float]]]] = {}
    for mask_scope in mask_scopes:
        # n_batches = 1
        # step_size = 0.1
        # n_steps = 100
        # seeds = list(range(10))
        # batch_sizes = [8]
        # n_ctx = 1

        results: dict[str, dict[str, dict[str, float]]] = {}

        for model_name, run_info_path in model_configs.items():
            print(f"\n{'=' * 60}")
            print(f"Loading model: {model_name.upper()}")
            print(f"{'=' * 60}\n")

            model_setup = load_model(run_info_path, target_module_patterns=target_module_patterns)
            # Change the model

            results[model_name] = {}

            print(f"Running with: {mask_scope}\n")

            for seed in seeds:
                seed_results, pgd_info = run_experiment(
                    model_setup=model_setup,
                    seed=seed,
                    mask_scope=mask_scope,
                    n_batches=n_batches,
                    batch_sizes=batch_sizes,
                    step_size=step_size,
                    n_steps=n_steps,
                    max_seq_len=n_ctx,
                )
                results[model_name][f"seed_{seed}"] = seed_results
                print(f"seed {seed} per_token_loss: {pgd_info['per_token_loss'].tolist()}")
                # Collect per-token results for plotting across mask scopes
                if model_name not in per_token_results:
                    per_token_results[model_name] = {}
                seed_key = f"seed_{seed}"
                if seed_key not in per_token_results[model_name]:
                    per_token_results[model_name][seed_key] = {}
                per_token_loss = pgd_info["per_token_loss"]
                try:
                    per_token_list = per_token_loss.tolist()
                except Exception:
                    per_token_list = list(per_token_loss)
                if (
                    isinstance(per_token_list, list)
                    and len(per_token_list) > 0
                    and isinstance(per_token_list[0], list)
                ):
                    per_token_list = [
                        v
                        for sub in per_token_list
                        for v in (sub if isinstance(sub, list) else [sub])
                    ]
                per_token_results[model_name][seed_key][str(mask_scope)] = per_token_list
                # print(f"seed {seed} decoded_batch: {pgd_info['decoded_batch']}")
                # print(f"seed {seed} decoded_tokens: {pgd_info['decoded_tokens']}")
                # print(f"seed {seed} target_out_top_5_logits: {pgd_info['target_out_top_5_logits']}")
                # print(f"seed {seed} target_out_top_5_tokens: {pgd_info['target_out_top_5_tokens']}")
                # print(
                #     f"seed {seed} target_out_top_5_tokens_str: {pgd_info['target_out_top_5_tokens_str']}"
                # )
                # print(
                #     f"seed {seed} model_output_top_5_logits: {pgd_info['model_output_top_5_logits']}"
                # )
                # print(
                #     f"seed {seed} model_output_top_5_tokens: {pgd_info['model_output_top_5_tokens']}"
                # )
                # print(
                #     f"seed {seed} model_output_top_5_tokens_str: {pgd_info['model_output_top_5_tokens_str']}"
                # )

        # results_path = Path(f"lm_pretoken_{mask_scope}_{n_steps=}_{step_size=}_{n_ctx=}.json")

        # save_results(results, results_path)

        # plot_from_file(results_path, log_scale=True)

    # Save and plot per-token results aggregated across mask scopes
    per_token_results_path = Path(f"lm_per_token_loss_{n_steps=}_{step_size=}_{n_ctx=}.json")
    save_per_token_results(per_token_results, per_token_results_path)

    per_token_results = load_per_token_results(per_token_results_path)
    plot_per_token_results(results=per_token_results, output_path=per_token_results_path)


if __name__ == "__main__":
    main()
