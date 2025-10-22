import uuid
from dataclasses import dataclass
from typing import cast

import torch
from jaxtyping import Float, Int
from torch._tensor import Tensor

from spd.app.backend.api import (
    AblationEffect,
    LayerAblationEffect,
    LayerCIs,
    MaskDTO,
    MatrixCausalImportances,
    OutputTokenLogit,
    RunResponse,
    SimulateMergeResponse,
    TokenAblationEffect,
)
from spd.app.backend.services.run_context_service import RunContextService
from spd.app.backend.utils import tensor_to_sparse_vector
from spd.log import logger
from spd.models.components import make_mask_infos
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import runtime_cast


@dataclass
class PromptContext:
    prompt: str
    input_token_ids: Int[torch.Tensor, " seq_len"]
    subcomponent_causal_importances: dict[str, Float[torch.Tensor, " seq_len C"]]


@dataclass
class Mask:
    id: str
    layer: str
    description: str | None
    combined_mask: Float[Tensor, " C"]

    def to_dto(self) -> MaskDTO:
        return MaskDTO(
            id=self.id,
            description=self.description,
            layer=self.layer,
            combined_mask=tensor_to_sparse_vector(self.combined_mask),
        )


DEVICE = get_device()
ACTIVE_THRESHOLD = 0.01


class AblationService:
    def __init__(
        self,
        run_context_service: RunContextService,
        *,
        dashboard_iteration: int = 3000,
        dashboard_n_samples: int = 16,
        dashboard_n_batches: int = 2,
        dashboard_batch_size: int = 64,
        dashboard_context_length: int = 64,
    ):
        self.run_context_service = run_context_service
        self.prompt_contexts: dict[str, PromptContext] = {}
        self.saved_masks: dict[str, Mask] = {}

    def _materialize_prompt(
        self, prompt: str | torch.Tensor
    ) -> tuple[str, list[str], torch.Tensor]:
        assert self.run_context_service.train_run_context is not None, "Run context not found"
        run = self.run_context_service.train_run_context

        match prompt:
            case str():
                inputs = runtime_cast(
                    torch.Tensor, run.tokenizer.encode(prompt, return_tensors="pt")
                )
                assert inputs.ndim == 2, "Inputs must be 2D (batch, seq_len)"
                assert inputs.shape[0] == 1, "batch size must be 1"
                inputs = inputs[0]

                prompt_str = prompt[:]
                assert prompt_str == prompt
            case torch.Tensor():
                inputs = prompt
                assert inputs.ndim == 1, "Inputs must be 1D (seq_len)"
                prompt_str = run.tokenizer.decode(inputs)  # pyright: ignore[reportAttributeAccessIssue]

        prompt_tokens = cast(list[str], run.tokenizer.batch_decode(inputs))  # pyright: ignore[reportAttributeAccessIssue]
        assert isinstance(prompt_tokens, list)
        assert isinstance(prompt_tokens[0], str)

        inputs = inputs.to(DEVICE)

        return prompt_str, prompt_tokens, inputs

    def run_prompt(self, prompt: str | torch.Tensor) -> RunResponse:
        assert (ctx := self.run_context_service.train_run_context) is not None

        prompt_str, prompt_tokens, inputs = self._materialize_prompt(prompt)

        logger.info(f"Inputs shape: {inputs.shape}")

        target_logits_out, pre_weight_acts = ctx.cm(inputs[None], cache_type="input")
        target_logits_out_nobatch = target_logits_out[0]

        logger.info(f"Pre-weight acts shape: {pre_weight_acts.keys()}")

        causal_importances = ctx.cm.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sampling=ctx.config.sampling,
        ).lower_leaky

        logger.info(f"Causal importances shape: {causal_importances.keys()}")

        # Run with overridden mask
        ci_masked_logits_nobatch = ctx.cm(
            inputs[None],  # add batch dim
            mask_infos=make_mask_infos(causal_importances),
        )[0]  # remove batch dim

        logger.info(f"CI masked logits shape: {ci_masked_logits_nobatch.shape}")

        # Store context
        # Generate a unique prompt ID
        prompt_id = str(uuid.uuid4())

        causal_importances_nobatch = {k: v[0] for k, v in causal_importances.items()}

        # Store the prompt context with overridden masks
        self.prompt_contexts[prompt_id] = PromptContext(
            prompt=prompt_str,
            input_token_ids=inputs,
            subcomponent_causal_importances=causal_importances_nobatch,
        )

        logger.info("saved")

        return RunResponse(
            prompt_id=prompt_id,
            prompt_tokens=prompt_tokens,
            layer_cis=self._to_layer_cis(causal_importances_nobatch),
            full_run_token_logits=self._logits_to_token_logits(target_logits_out_nobatch),
            ci_masked_token_logits=self._logits_to_token_logits(ci_masked_logits_nobatch),
        )

    def _to_layer_cis(
        self,
        causal_importances: dict[str, Float[Tensor, "seq_len C"]],
    ) -> list[LayerCIs]:
        layer_cis = []
        for module, layer_ci in causal_importances.items():
            token_cis = []
            for tok_layer_ci in layer_ci:
                matrix_ci = MatrixCausalImportances(
                    subcomponent_cis_sparse=tensor_to_sparse_vector(tok_layer_ci),
                    subcomponent_cis=tok_layer_ci.tolist(),
                )
                token_cis.append(matrix_ci)
            layer_cis.append(LayerCIs(module=module, token_cis=token_cis))
        return layer_cis

    def run_with_mask(
        self, prompt_id: str, mask_id: str
    ) -> tuple[list[list[OutputTokenLogit]], AblationEffect]:
        """Apply a saved mask override as an ablation to a specific prompt."""
        assert (ctx := self.run_context_service.train_run_context) is not None
        assert (prompt_context := self.prompt_contexts.get(prompt_id)) is not None
        assert (mask := self.saved_masks.get(mask_id)) is not None

        # Start with the prompt's causal importances
        original_cis = prompt_context.subcomponent_causal_importances.copy()
        ablated_cis = prompt_context.subcomponent_causal_importances.copy()

        # Override the specified layer with the saved mask for ALL tokens
        ablated_cis[mask.layer].copy_(mask.combined_mask)

        # Compute effect before running inference
        ablation_effect = self._compute_ablation_effect(original_cis, ablated_cis)

        # Run with the mask
        ci_masked_logits = ctx.cm(
            prompt_context.input_token_ids[None],  # add batch dim
            mask_infos=make_mask_infos(ablated_cis),
        )[0]  # remove batch dim

        return self._logits_to_token_logits(ci_masked_logits), ablation_effect

    def ablate_subcomponents(
        self, prompt_id: str, subcomponent_mask: dict[str, list[list[int]]]
    ) -> list[list[OutputTokenLogit]]:
        assert (ctx := self.run_context_service.train_run_context) is not None
        assert (prompt_context := self.prompt_contexts.get(prompt_id)) is not None

        masked_ci = prompt_context.subcomponent_causal_importances.copy()
        for module, token_subcomponent_masks in subcomponent_mask.items():
            for token_idx, token_subcomponent_mask in enumerate(token_subcomponent_masks):
                masked_ci[module][token_idx][token_subcomponent_mask] = 0

        ci_masked_logits = ctx.cm(
            prompt_context.input_token_ids[None],  # add batch dim
            mask_infos=make_mask_infos(masked_ci),
        )[0]  # remove batch dim

        return self._logits_to_token_logits(ci_masked_logits)

    TOP_K = 5

    def _logits_to_token_logits(
        self,
        logits: Float[torch.Tensor, "seq_len vocab"],
    ) -> list[list[OutputTokenLogit]]:
        assert (ctx := self.run_context_service.train_run_context) is not None, (
            "Run context not found"
        )

        assert logits.ndim == 2, "Logits must be 2D (seq_len, vocab)"
        assert logits.shape[1] == len(ctx.tokenizer.vocab), (  # pyright: ignore[reportAttributeAccessIssue]
            f"Logits must have the same length as the vocabulary, {logits.shape[2]} != {len(ctx.tokenizer.vocab)}"  # pyright: ignore[reportAttributeAccessIssue]
        )

        tokens_logits: list[list[OutputTokenLogit]] = []
        for token_logits in logits:
            assert token_logits.ndim == 1, f"Token logits must be 1D, got {token_logits.ndim}"
            token_probs = torch.softmax(token_logits, dim=-1)

            this_token_logits = []
            ordered_indices = torch.argsort(token_logits, dim=-1, descending=True)
            for tok_id in ordered_indices[: self.TOP_K]:
                tok_id = int(tok_id.item())
                tok_str = ctx.tokenizer.convert_ids_to_tokens([tok_id])[0]  # pyright: ignore[reportAttributeAccessIssue]
                logit = float(token_logits[tok_id])
                prob = float(token_probs[tok_id])
                this_token_logits.append(
                    OutputTokenLogit(token=tok_str, logit=logit, probability=prob)
                )

            tokens_logits.append(this_token_logits)
        return tokens_logits

    def run_prompt_by_index(self, dataset_index: int) -> RunResponse:
        """Run a specific prompt from the dataset by index."""
        assert (ctx := self.run_context_service.train_run_context) is not None, (
            "Run context not found"
        )

        if dataset_index >= len(ctx.train_loader.dataset):  # pyright: ignore[reportArgumentType]
            raise ValueError(
                f"Index {dataset_index} out of range for dataset of size {len(ctx.train_loader.dataset)}"  # pyright: ignore[reportArgumentType]
            )

        example = ctx.train_loader.dataset[dataset_index]["input_ids"]

        assert isinstance(example, torch.Tensor)
        assert example.ndim == 1, "Example must be 1D (seq_len)"
        logger.info(f"Running prompt by index: {dataset_index}, shape: {example.shape}")

        return self.run_prompt(example)

    def create_combined_mask(
        self,
        prompt_id: str,
        layer: str,
        token_indices: list[int],
        description: str | None = None,
    ) -> Mask:
        assert self.run_context_service.train_run_context is not None, "Run context not found"
        assert prompt_id in self.prompt_contexts, f"Prompt {prompt_id} not found"

        # Get the CIs for the specified layer
        prompt_context = self.prompt_contexts[prompt_id]
        layer_ci = prompt_context.subcomponent_causal_importances[layer]

        # Combine masks using element-wise max
        token_masks: Float[Tensor, "seq_len C"] = layer_ci[token_indices]
        combined_mask: Float[Tensor, " C"] = torch.max(token_masks, dim=0).values

        mask = Mask(
            id=str(uuid.uuid4()),
            layer=layer,
            combined_mask=combined_mask,
            description=description,
        )

        return mask

    def save_mask(self, mask: Mask):
        self.saved_masks[mask.id] = mask

    # change this to also return the metric
    def get_merge_l0(
        self, prompt_id: str, layer: str, token_indices: list[int]
    ) -> SimulateMergeResponse:
        mask = self.create_combined_mask(
            prompt_id=prompt_id,
            layer=layer,
            token_indices=token_indices,
            description=None,
        )
        l0 = tensor_to_sparse_vector(mask.combined_mask).l0

        # Compute the k-way Jaccard on the original token masks for this layer
        assert prompt_id in self.prompt_contexts, f"Prompt {prompt_id} not found"
        prompt_context = self.prompt_contexts[prompt_id]
        layer_ci = prompt_context.subcomponent_causal_importances[layer]  # (seq_len, C)
        token_masks: Float[Tensor, "m C"] = layer_ci[token_indices]

        # Weighted/tensor Jaccard (choose this or the set version below)
        jacc = k_way_weighted_jaccard(token_masks)

        # If you prefer the set/binary interpretation (presence > 0.0), use:
        # jacc = k_way_set_jaccard(token_masks, threshold=0.0)
        return SimulateMergeResponse(l0=l0, jacc=jacc)

    def _compute_ablation_effect(
        self,
        original_cis: dict[str, Float[Tensor, "seq_len C"]],
        ablated_cis: dict[str, Float[Tensor, "seq_len C"]],
    ) -> AblationEffect:
        """Compute statistics about what was ablated."""
        layer_effect = []
        for module in original_cis:
            original_layer_cis = original_cis[module]
            ablated_layer_cis = ablated_cis[module]
            seq_len = original_layer_cis.shape[0]

            token_effect = []
            for token_idx in range(seq_len):
                original_token_cis = original_layer_cis[token_idx]
                ablated_token_cis = ablated_layer_cis[token_idx]

                # Binary: count active components
                originally_active = original_token_cis > ACTIVE_THRESHOLD
                now_inactive = ablated_token_cis <= ACTIVE_THRESHOLD
                ablated_mask = originally_active & now_inactive

                original_active_count = int(originally_active.sum().item())
                ablated_count = int(ablated_mask.sum().item())

                # Magnitude: sum of CI lost
                ci_diff = original_token_cis - ablated_token_cis
                ablated_magnitude = float(ci_diff.clamp_min(0).sum().item())

                token_effect.append(
                    TokenAblationEffect(
                        original_active_count=original_active_count,
                        ablated_count=ablated_count,
                        ablated_magnitude=ablated_magnitude,
                    )
                )

            layer_effect.append(LayerAblationEffect(module=module, token_abl_effect=token_effect))

        return AblationEffect(layer_abl_effect=layer_effect)


def k_way_weighted_jaccard(token_masks: Float[Tensor, "m C"]) -> float:
    """
    Weighted (Ruzicka/Tanimoto) k-way Jaccard for nonnegative masks.
    token_masks: shape (m, C) where m=len(token_indices)
    Returns a scalar in [0, 1]. If the union is empty, returns 1.0 by convention.
    """
    assert token_masks.ndim == 2, "Expected (m, C)"
    # Assumes nonnegative entries; clamp to be safe against tiny negatives from numerics
    token_masks = token_masks.clamp_min(0.0)
    mins = token_masks.min(dim=0).values  # (C,)
    maxs = token_masks.max(dim=0).values  # (C,)
    numerator = mins.sum()
    denominator = maxs.sum()
    return float(numerator / denominator) if float(denominator) > 0.0 else 1.0
