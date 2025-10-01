import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from spd.app.backend.services.run_context_service import RunContextService
from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.settings import SPD_CACHE_DIR
from spd.utils.component_utils import calc_ci_l_zero
from spd.utils.general_utils import extract_batch_data


class ActivationContext(BaseModel):
    raw_text: str
    offset_mapping: list[tuple[int, int]]
    token_ci_values: list[float]
    active_position: int
    ci_value: float


ActivationContextsByComponent = dict[int, list[ActivationContext]]

ActivationContextsByModule = dict[str, ActivationContextsByComponent]


def _default_output_path(wandb_id: str) -> Path:
    run_dir: Path = SPD_CACHE_DIR / "runs" / f"spd-{wandb_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "component_activation_contexts.json"


def _serialize_activation_contexts(
    data: dict[str, dict[int, list[ActivationContext]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Convert nested Pydantic models to JSON-serializable primitives.

    JSON keys must be strings, so component indices are stringified.
    """
    serialized: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for layer, components in data.items():
        layer_obj: dict[str, list[dict[str, Any]]] = {}
        for comp_idx, examples in components.items():
            layer_obj[str(comp_idx)] = [ex.model_dump() for ex in examples]
        serialized[layer] = layer_obj
    return serialized


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp_path: Path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def main(
    wandb_id: str,
    out: Path | None,
    n_prompts: int,
    n_tokens_either_side: int,
    n_steps: int,
    ci_threshold: float,
):
    out_path: Path = out or _default_output_path(wandb_id)
    lock_path: Path = out_path.with_suffix(out_path.suffix + ".lock")

    # Try to obtain a simple lock to avoid duplicate computation.
    try:
        with open(lock_path, "x") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        logger.info(f"Lock exists at {lock_path}, another worker may be running. Exiting.")
        return 0

    try:
        logger.info(f"Starting activation contexts computation for run {wandb_id} -> {out_path}")

        rcs = RunContextService()
        rcs.load_run_from_wandb_id(wandb_id)
        assert (rc := rcs.run_context) is not None, "Run context not found"

        results = find_component_activation_contexts(
            component_model=rc.cm,
            dataloader=rc.train_loader,
            run_config=rc.config,
            tokenizer=rc.tokenizer,
            causal_importance_threshold=ci_threshold,
            n_prompts=n_prompts,
            n_tokens_either_side=n_tokens_either_side,
            n_steps=n_steps,
        )

        payload = _serialize_activation_contexts(results)
        _write_json_atomic(out_path, payload)
        logger.info(f"Wrote activation contexts to {out_path}")
        return 0
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(f"Activation contexts worker failed: {e}")
        return 1
    finally:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:  # best-effort cleanup
            pass


def find_component_activation_contexts(
    component_model: ComponentModel,
    dataloader: DataLoader[Any],
    run_config: Config,
    tokenizer: PreTrainedTokenizer,
    causal_importance_threshold: float,
    n_prompts: int,
    n_tokens_either_side: int,
    n_steps: int,
) -> ActivationContextsByModule:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize tracking
    component_contexts_by_module: ActivationContextsByModule = {}
    l0_scores_sum = defaultdict[str, float](float)
    # l0_scores_count = 0

    data_iter = iter(dataloader)

    from tqdm import tqdm

    for _ in tqdm(range(n_steps)):
        batch = extract_batch_data(next(data_iter))
        batch = batch.to(device)

        ci_l_zero_vals = _process_batch_for_contexts(
            batch=batch,
            component_model=component_model,
            run_config=run_config,
            tokenizer=tokenizer,
            causal_importance_threshold=causal_importance_threshold,
            n_prompts=n_prompts,
            n_tokens_either_side=n_tokens_either_side,
            component_contexts_by_module=component_contexts_by_module,
        )

        for layer_name, layer_ci_l_zero in ci_l_zero_vals.items():
            l0_scores_sum[layer_name] += layer_ci_l_zero
        # l0_scores_count += 1

        if _check_all_components_have_enough_examples(
            component_contexts_by_module, n_prompts, component_model.C
        ):
            break

    return component_contexts_by_module


def _process_batch_for_contexts(
    *,
    component_model: ComponentModel,
    run_config: Config,
    batch: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    component_contexts_by_module: ActivationContextsByModule,
    causal_importance_threshold: float,
    n_prompts: int,
    n_tokens_either_side: int,
) -> dict[str, float]:
    """Process a single batch to find activation contexts."""
    # Get activations before each component
    with torch.no_grad():
        _, pre_weight_acts = component_model(
            batch, mode="input_cache", module_names=list(component_model.components.keys())
        )

        causal_importances, _ = component_model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type=run_config.sigmoid_type,
            detach_inputs=True,
            sampling=run_config.sampling,
        )

    # Calculate L0 scores
    ci_l_zero_vals: dict[str, float] = {}
    for module_name, ci in causal_importances.items():
        ci_l_zero_vals[module_name] = calc_ci_l_zero(ci, causal_importance_threshold)

    # Find activation contexts
    for module_name, ci in causal_importances.items():
        assert ci.ndim == 3, "CI must be 3D (batch, seq_len, C)"

        if module_name not in component_contexts_by_module:
            component_contexts_by_module[module_name] = {}

        # Find active components
        active_mask = ci > causal_importance_threshold

        # For each component
        for component_idx in range(component_model.C):
            if component_idx not in component_contexts_by_module[module_name]:
                component_contexts_by_module[module_name][component_idx] = []

            # Skip if we already have enough examples
            if len(component_contexts_by_module[module_name][component_idx]) >= n_prompts:
                continue

            # Get positions where this component is active
            component_active = active_mask[:, :, component_idx]

            # Find activations in this batch
            batch_idxs, seq_idxs = torch.where(component_active)

            for batch_idx, seq_idx in zip(batch_idxs.tolist(), seq_idxs.tolist(), strict=True):
                # Skip if we have enough examples
                if len(component_contexts_by_module[module_name][component_idx]) >= n_prompts:
                    break

                context = _extract_activation_context(
                    batch=batch,
                    batch_idx=batch_idx,
                    seq_idx=seq_idx,
                    ci=ci,
                    component_idx=component_idx,
                    component_active=component_active,
                    n_tokens_either_side=n_tokens_either_side,
                    tokenizer=tokenizer,
                )

                component_contexts_by_module[module_name][component_idx].append(context)

    return ci_l_zero_vals


def _check_all_components_have_enough_examples(
    component_contexts_by_module: ActivationContextsByModule,
    n_prompts: int,
    n_components: int,
) -> bool:
    """Check if all components have enough examples."""
    for module_name in component_contexts_by_module:
        for component_idx in range(n_components):
            if component_idx not in component_contexts_by_module[module_name]:
                return False
            if len(component_contexts_by_module[module_name][component_idx]) < n_prompts:
                return False
    return True


def _extract_activation_context(
    *,
    batch: torch.Tensor,
    batch_idx: int,
    seq_idx: int,
    ci: torch.Tensor,
    component_idx: int,
    component_active: torch.Tensor,
    n_tokens_either_side: int,
    tokenizer: Any,
) -> ActivationContext:
    """Extract activation context for a single position."""
    # Get the CI value at this position
    ci_value = ci[batch_idx, seq_idx, component_idx].item()

    # Get context window
    start_idx = max(0, seq_idx - n_tokens_either_side)
    end_idx = min(batch.shape[1], seq_idx + n_tokens_either_side + 1)

    # Get token IDs for the context window
    context_token_ids = batch[batch_idx, start_idx:end_idx].tolist()

    # Decode the entire context to get raw text and offset mappings
    raw_text = tokenizer.decode(context_token_ids)

    # Re-tokenize to get offset mappings
    context_tokenized = tokenizer(
        raw_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
        add_special_tokens=False,
    )

    offset_mapping = context_tokenized["offset_mapping"][0].tolist()

    # Calculate CI values for each token in context
    token_ci_values = []
    for i in range(len(offset_mapping)):
        if i < len(context_token_ids):  # Ensure we're within bounds
            if start_idx + i == seq_idx:
                token_ci_values.append(ci_value)
            else:
                # Get CI value for other tokens too if they're active
                if start_idx + i < ci.shape[1] and component_active[batch_idx, start_idx + i]:
                    token_ci_values.append(ci[batch_idx, start_idx + i, component_idx].item())
                else:
                    token_ci_values.append(0.0)
        else:
            token_ci_values.append(0.0)

    return ActivationContext(
        raw_text=raw_text,
        offset_mapping=offset_mapping,
        token_ci_values=token_ci_values,
        active_position=seq_idx - start_idx,  # Position of main active token in context
        ci_value=ci_value,
    )


# # %%

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Compute component activation contexts and write JSON cache."
#     )
#     parser.add_argument("--wandb-id", required=True, help="W&B run id (e.g. ry05f67a)")
#     parser.add_argument(
#         "--out",
#         type=Path,
#         default=None,
#         help="Output JSON path. Defaults to SPD cache dir for the run.",
#     )
#     parser.add_argument("--n-prompts", type=int, default=20)
#     parser.add_argument("--n-tokens-either-side", type=int, default=10)
#     parser.add_argument("--n-steps", type=int, default=20)
#     parser.add_argument("--ci-threshold", type=float, default=0.01)

#     args = parser.parse_args()
#     sys.exit(
#         main(
#             wandb_id=args.wandb_id,
#             out=args.out,
#             n_prompts=args.n_prompts,
#             n_tokens_either_side=args.n_tokens_either_side,
#             n_steps=args.n_steps,
#             ci_threshold=args.ci_threshold,
#         )
#     )
