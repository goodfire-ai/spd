"""Harvest worker: collects component statistics on a single GPU.

Usage:
    python -m spd.harvest.scripts.run_worker <wandb_path> --config_json '{"n_batches": 100}'
    python -m spd.harvest.scripts.run_worker <wandb_path> --config_json '...' --rank 0 --world_size 4 --subrun_id h-20260211_120000
"""

from datetime import datetime

import fire
import torch
from jaxtyping import Float
from torch import Tensor

from spd.data import train_loader_and_tokenizer
from spd.harvest.config import HarvestConfig
from spd.harvest.harvest import harvest
from spd.harvest.schemas import HarvestBatch, get_harvest_subrun_dir
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.topology import TransformerTopology
from spd.utils.distributed_utils import get_device
from spd.utils.wandb_utils import parse_wandb_run_path


def _compute_u_norms(model: ComponentModel) -> dict[str, Float[Tensor, " C"]]:
    """Compute ||U[c,:]|| for each component c in each layer.

    Component activations (v_i^T @ a) have a scale invariance: scaling V by alpha and U by 1/alpha
    leaves the weight matrix unchanged but scales component activations by alpha. To make component
    activations reflect actual output contribution, we multiply by the U row norms.
    """
    return {
        layer_name: component.U.norm(dim=1) for layer_name, component in model.components.items()
    }


def main(
    wandb_path: str,
    config_json: str | dict[str, object] | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    subrun_id: str | None = None,
) -> None:
    _, _, run_id = parse_wandb_run_path(wandb_path)

    if subrun_id is None:
        subrun_id = "h-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = get_harvest_subrun_dir(run_id, subrun_id)

    assert (rank is None) == (world_size is None), "rank and world_size must both be set or unset"

    match config_json:
        case str(json_str):
            config = HarvestConfig.model_validate_json(json_str)
        case dict(d):
            config = HarvestConfig.model_validate(d)
        case None:
            config = HarvestConfig()

    if world_size is not None:
        logger.info(
            f"Distributed harvest: {wandb_path} (rank {rank}/{world_size}, subrun {subrun_id})"
        )
    else:
        logger.info(f"Single-GPU harvest: {wandb_path} (subrun {subrun_id})")

    device = torch.device(get_device())
    logger.info(f"Loading model on {device}")

    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    train_loader, _ = train_loader_and_tokenizer(spd_config, config.batch_size)

    layer_names = list(model.target_module_paths)
    layers = [(name, model.module_to_c[name]) for name in layer_names]
    topology = TransformerTopology(model.target_model)
    vocab_size = topology.unembed_module.out_features
    activation_threshold = config.activation_threshold

    u_norms = _compute_u_norms(model)

    def spd_harvest_fn(batch_item: Tensor) -> HarvestBatch:
        from spd.utils.general_utils import extract_batch_data

        batch = extract_batch_data(batch_item).to(device)

        out = model(batch, cache_type="input")
        probs = torch.softmax(out.output, dim=-1)

        ci_dict = model.calc_causal_importances(
            pre_weight_acts=out.cache,
            detach_inputs=True,
            sampling=spd_config.sampling,
        ).lower_leaky

        per_layer_acts = model.get_all_component_acts(out.cache)
        normalized_acts = {
            layer: acts * u_norms[layer].to(acts.device) for layer, acts in per_layer_acts.items()
        }

        firings = {layer: ci_dict[layer] > activation_threshold for layer in layer_names}
        activations = {
            layer: {
                "causal_importance": ci_dict[layer],
                "component_activation": normalized_acts[layer],
            }
            for layer in layer_names
        }

        return HarvestBatch(
            tokens=batch,
            firings=firings,
            activations=activations,
            output_probs=probs,
        )

    harvest(
        harvest_fn=spd_harvest_fn,
        layers=layers,
        vocab_size=vocab_size,
        dataloader=train_loader,
        config=config,
        output_dir=output_dir,
        rank=rank,
        world_size=world_size,
        device=device,
    )


if __name__ == "__main__":
    fire.Fire(main)
