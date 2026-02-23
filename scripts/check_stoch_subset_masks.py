#!/usr/bin/env python3
"""Log stochastic subset masks across DDP ranks to check for identical RNG.

Run (CPU):
  torchrun --standalone --nproc_per_node=2 scripts/check_stoch_subset_masks.py --n-mask-samples 2

Run (GPU):
  torchrun --standalone --nproc_per_node=2 scripts/check_stoch_subset_masks.py --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import torch
import torch.distributed as dist

from spd.configs import StaticProbabilityRoutingConfig, UniformKSubsetRoutingConfig
from spd.routing import get_subset_router
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.distributed_utils import cleanup_distributed, get_device, init_distributed
from spd.utils.general_utils import set_seed


def _tensor_fingerprint(t: torch.Tensor) -> dict[str, Any]:
    t_cpu = t.detach().contiguous().cpu()
    h = hashlib.sha256(t_cpu.numpy().tobytes()).hexdigest()
    t_float = t_cpu.float()
    return {
        "shape": list(t_cpu.shape),
        "dtype": str(t_cpu.dtype),
        "mean": t_float.mean().item(),
        "std": t_float.std(unbiased=False).item(),
        "sha256": h,
    }


def _mask_fingerprints(mask_infos: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for module_name, info in mask_infos.items():
        routing = info.routing_mask
        if isinstance(routing, str):
            routing_fp: dict[str, Any] | str = routing
        else:
            routing_fp = _tensor_fingerprint(routing)

        weight_mask_fp: dict[str, Any] | None = None
        if info.weight_delta_and_mask is not None:
            _weight_delta, weight_mask = info.weight_delta_and_mask
            weight_mask_fp = _tensor_fingerprint(weight_mask)

        out[module_name] = {
            "component_mask": _tensor_fingerprint(info.component_mask),
            "routing_mask": routing_fp,
            "weight_delta_mask": weight_mask_fp,
        }
    return out


def _flatten_hashes(fps: dict[str, dict[str, Any]], sample_idx: int) -> dict[str, str]:
    flat: dict[str, str] = {}
    for module_name, fp in fps.items():
        comp_hash = fp["component_mask"]["sha256"]
        routing = fp["routing_mask"]
        if isinstance(routing, str):
            routing_hash = routing
        else:
            routing_hash = routing["sha256"]
        weight_mask = fp["weight_delta_mask"]
        weight_hash = "none" if weight_mask is None else weight_mask["sha256"]
        flat[f"sample={sample_idx}|module={module_name}|component"] = comp_hash
        flat[f"sample={sample_idx}|module={module_name}|routing"] = routing_hash
        flat[f"sample={sample_idx}|module={module_name}|weight_delta"] = weight_hash
    return flat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling", choices=["continuous", "binomial"], default="continuous")
    parser.add_argument("--n-mask-samples", type=int, default=1)
    parser.add_argument("--use-delta-component", action="store_true")
    parser.add_argument("--routing", choices=["uniform_k_subset", "static_probability"], default="uniform_k_subset")
    parser.add_argument("--routing-p", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=3)
    parser.add_argument("--c", type=int, default=4)
    parser.add_argument("--n-modules", type=int, default=2)
    parser.add_argument("--d-in", type=int, default=3)
    parser.add_argument("--d-out", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    dist_state = init_distributed()
    set_seed(args.seed)

    device = args.device or get_device()
    rank = 0
    world_size = 1
    if dist_state is not None:
        rank = dist_state.rank
        world_size = dist_state.world_size

    module_names = [f"layer_{i}" for i in range(args.n_modules)]
    ci: dict[str, torch.Tensor] = {}
    for name in module_names:
        # Deterministic CI values; avoid consuming RNG before mask sampling.
        ci[name] = torch.full((args.batch, args.seq, args.c), 0.5, device=device)

    weight_deltas: dict[str, torch.Tensor] | None = None
    if args.use_delta_component:
        weight_deltas = {name: torch.zeros((args.d_out, args.d_in), device=device) for name in module_names}

    if args.routing == "uniform_k_subset":
        routing_cfg = UniformKSubsetRoutingConfig()
    else:
        routing_cfg = StaticProbabilityRoutingConfig(p=args.routing_p)

    router = get_subset_router(routing_cfg, device=device)

    all_flat_hashes: dict[str, str] = {}
    for sample_idx in range(args.n_mask_samples):
        mask_infos = calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=args.sampling,
            weight_deltas=weight_deltas,
            router=router,
        )

        fps = _mask_fingerprints(mask_infos)
        for module_name, fp in fps.items():
            payload = {
                "rank": rank,
                "world_size": world_size,
                "sample_idx": sample_idx,
                "module": module_name,
                "component_mask": fp["component_mask"],
                "routing_mask": fp["routing_mask"],
                "weight_delta_mask": fp["weight_delta_mask"],
            }
            print(json.dumps(payload), flush=True)

        all_flat_hashes.update(_flatten_hashes(fps, sample_idx))

    if dist.is_initialized():
        gathered: list[dict[str, str]] = [dict() for _ in range(world_size)]
        dist.all_gather_object(gathered, all_flat_hashes)
        if rank == 0:
            # Compare hashes across ranks for each key
            keys = sorted(all_flat_hashes.keys())
            mismatches = []
            for key in keys:
                values = {g.get(key) for g in gathered}
                if len(values) > 1:
                    mismatches.append((key, values))
            if mismatches:
                print("MISMATCHES:")
                for key, values in mismatches:
                    print(f"  {key}: {sorted(values)}")
            else:
                print("All hashes identical across ranks.")

    cleanup_distributed()


if __name__ == "__main__":
    main()
