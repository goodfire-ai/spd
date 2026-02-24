"""Generate text completions with and without ablation at a single position.

Shows how a single-position ablation affects the model's next prediction
and how that cascades through autoregressive generation.

Usage:
    python -m spd.scripts.attention_ablation_experiment.generate_with_ablation \
        wandb:goodfire/spd/runs/s-275c8f21 \
        --components "h.1.attn.q_proj:279,h.1.attn.k_proj:177" \
        --heads L1H1 \
        --n_samples 4 --prompt_len 128 --gen_len 32
"""

import random
from pathlib import Path

import fire
import torch
from jaxtyping import Int
from torch import Tensor

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import ComponentsMaskInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.attention_ablation_experiment.attention_ablation_experiment import (
    ComponentHeadAblation,
    _build_component_head_ablations,
    _build_deterministic_masks_multi_pos,
    _build_prev_token_component_positions,
    parse_components,
    parse_heads,
    patched_attention_forward,
)
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent


def _build_baseline_mask_infos(
    spd_model: ComponentModel,
    device: torch.device,
) -> dict[str, ComponentsMaskInfo]:
    """Build all-ones (batch=1, C) masks for SPD model baseline."""
    from spd.models.components import make_mask_infos

    masks = {name: torch.ones(1, c, device=device) for name, c in spd_model.module_to_c.items()}
    return make_mask_infos(masks)


def _generate_greedy(
    target_model: LlamaSimpleMLP,
    prompt_ids: Int[Tensor, "1 prompt_len"],
    gen_len: int,
    spd_model: ComponentModel | None = None,
    mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    head_pos_ablations: list[tuple[int, int, int]] | None = None,
    value_pos_ablations: list[tuple[int, int]] | None = None,
    value_head_pos_ablations: list[tuple[int, int, int]] | None = None,
    component_head_ablations: list[ComponentHeadAblation] | None = None,
    ablate_first_only: bool = True,
) -> list[int]:
    """Generate tokens greedily. Ablation applied only on the first forward pass."""
    baseline_mask_infos = (
        _build_baseline_mask_infos(spd_model, prompt_ids.device) if spd_model is not None else None
    )
    generated: list[int] = []
    input_ids = prompt_ids.clone()

    for step in range(gen_len):
        use_ablation = step == 0 or not ablate_first_only

        with patched_attention_forward(
            target_model,
            head_pos_ablations=head_pos_ablations if use_ablation else None,
            value_pos_ablations=value_pos_ablations if use_ablation else None,
            value_head_pos_ablations=value_head_pos_ablations if use_ablation else None,
            component_head_ablations=component_head_ablations if use_ablation else None,
        ):
            if spd_model is not None:
                step_masks = mask_infos if use_ablation else baseline_mask_infos
                out = spd_model(input_ids, mask_infos=step_masks)
                assert isinstance(out, Tensor)
                logits = out
            else:
                logits, _ = target_model(input_ids)
                assert logits is not None

        next_token = logits[0, -1].argmax().item()
        generated.append(int(next_token))
        next_tensor = torch.tensor([[next_token]], device=input_ids.device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)

    return generated


def generate_with_ablation(
    wandb_path: ModelPath,
    components: str | None = None,
    heads: str | None = None,
    restrict_to_heads: str | None = None,
    n_samples: int = 4,
    prompt_len: int = 128,
    gen_len: int = 32,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    parsed_components = parse_components(components) if components else []
    parsed_heads = parse_heads(heads) if heads else []
    parsed_restrict_heads = parse_heads(restrict_to_heads) if restrict_to_heads else []

    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)
    config = run_info.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spd_model = ComponentModel.from_run_info(run_info)
    spd_model.eval()
    spd_model = spd_model.to(device)
    target_model = spd_model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)
    for block in target_model._h:
        block.attn.flash_attention = False

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
    )
    loader, tokenizer = create_data_loader(
        dataset_config=dataset_config, batch_size=1, buffer_size=1000
    )
    decode = tokenizer.decode  # pyright: ignore[reportAttributeAccessIssue]

    out_dir = SCRIPT_DIR / "out" / run_id / "generations"
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if i >= n_samples:
                break

            input_ids: Int[Tensor, "1 seq"] = batch_data[task_config.column_name][
                :, :prompt_len
            ].to(device)

            rng = random.Random(i)
            t = rng.randint(1, min(prompt_len, 128) - 1)

            # Truncate prompt to t+1 tokens so generation starts right after ablation
            prompt_ids = input_ids[:, : t + 1]
            prompt_text = decode(prompt_ids[0].tolist())
            context_after = decode(input_ids[0, t + 1 :].tolist())

            # 1. Target model (no SPD)
            target_gen = _generate_greedy(target_model, prompt_ids, gen_len)

            # 2. SPD baseline
            spd_gen = _generate_greedy(target_model, prompt_ids, gen_len, spd_model=spd_model)

            # 3. Head ablation (if heads specified)
            head_gen: list[int] | None = None
            if parsed_heads:
                head_abl = [(layer, head, t) for layer, head in parsed_heads]
                head_gen = _generate_greedy(
                    target_model, prompt_ids, gen_len, head_pos_ablations=head_abl
                )

            # 4. Full component ablation (if components specified)
            comp_gen: list[int] | None = None
            if parsed_components:
                cp = _build_prev_token_component_positions(parsed_components, t)
                bs = (prompt_ids.shape[0], prompt_ids.shape[1])
                _, ablated_masks = _build_deterministic_masks_multi_pos(
                    spd_model, cp, bs, prompt_ids.device
                )
                comp_gen = _generate_greedy(
                    target_model,
                    prompt_ids,
                    gen_len,
                    spd_model=spd_model,
                    mask_infos=ablated_masks,
                )

            # 5. Per-head component ablation (if restrict_to_heads specified)
            perhead_gen: list[int] | None = None
            if parsed_components and parsed_restrict_heads:
                comp_head_abls = _build_component_head_ablations(
                    spd_model, parsed_components, parsed_restrict_heads, t
                )
                perhead_gen = _generate_greedy(
                    target_model,
                    prompt_ids,
                    gen_len,
                    spd_model=spd_model,
                    component_head_ablations=comp_head_abls,
                )

            # Format output
            lines = [
                f"{'=' * 80}",
                f"Sample {i} | ablation at t={t} | generating from t+1",
                f"{'=' * 80}",
                "",
                f"PROMPT ({t + 1} tokens, up to and including ablation position):",
                prompt_text,
                "",
                "ACTUAL CONTINUATION (from dataset):",
                context_after,
                "",
                f"--- Generated continuations ({gen_len} tokens, greedy) ---",
                "",
                f"Target model:     {decode(target_gen)}",
                f"SPD baseline:     {decode(spd_gen)}",
            ]
            if head_gen is not None:
                head_label = ",".join(f"L{ly}H{hd}" for ly, hd in parsed_heads)
                lines.append(f"Head ablated ({head_label} @t={t}):  {decode(head_gen)}")
            if comp_gen is not None:
                lines.append(f"Full comp ablated (@t={t}):  {decode(comp_gen)}")
            if perhead_gen is not None:
                rh_label = ",".join(f"L{ly}H{hd}" for ly, hd in parsed_restrict_heads)
                lines.append(f"Per-head comp ({rh_label} @t={t}):  {decode(perhead_gen)}")

            lines.append("")
            output = "\n".join(lines)
            logger.info(output)

            # Save to file
            with open(out_dir / f"sample{i}.txt", "w") as f:
                f.write(output)

    logger.info(f"Saved {n_samples} generation samples to {out_dir}")


if __name__ == "__main__":
    fire.Fire(generate_with_ablation)
