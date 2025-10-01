import uuid
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from pydantic import BaseModel
from torch._tensor import Tensor

from spd.app.backend.services.run_context_service import RunContextService
from spd.models.components import make_mask_infos
from spd.utils.general_utils import runtime_cast


@dataclass
class PromptContext:
    prompt: str
    input_token_ids: Int[torch.Tensor, " seq_len"]
    causal_importances: dict[str, Float[torch.Tensor, " seq_len C"]]


class SparseVector(BaseModel):
    l0: int
    indices: list[int]
    values: list[float]

    @classmethod
    def from_tensor(cls, tensor: Float[torch.Tensor, " C"]) -> "SparseVector":
        assert tensor.ndim == 1, "Tensor must be 1D"
        l0 = (tensor > 0.0).float().sum().item()
        (nonzero_indices,) = tensor.nonzero(as_tuple=True)
        return cls(
            l0=int(l0),
            indices=nonzero_indices.tolist(),
            values=tensor[nonzero_indices].tolist(),
        )


class LayerCIs(BaseModel):
    module: str
    token_cis: list[SparseVector]


class OutputTokenLogit(BaseModel):
    token: str
    logit: float
    probability: float


class TokenLayerCosineSimilarityData(BaseModel):
    input_singular_vectors: list[list[float]]
    output_singular_vectors: list[list[float]]
    component_indices: list[int]


@dataclass
class MaskOverride:
    id: str
    description: str | None
    layer: str
    combined_mask: Float[Tensor, " C"]


class AblationService:
    def __init__(self, run_context_service: RunContextService):
        self.run_context_service = run_context_service
        self.prompt_contexts: dict[str, PromptContext] = {}  # Multiple prompts by ID
        self.mask_overrides: dict[str, MaskOverride] = {}

    def run_prompt(
        self, prompt: str | torch.Tensor
    ) -> tuple[
        str,
        list[str],
        list[LayerCIs],
        list[list[OutputTokenLogit]],
        list[list[OutputTokenLogit]],
    ]:
        assert self.run_context_service.run_context is not None, "Run context not found"
        run = self.run_context_service.run_context

        match prompt:
            case str():
                inputs = runtime_cast(
                    torch.Tensor, run.tokenizer.encode(prompt, return_tensors="pt")
                )
                assert inputs.ndim == 2, "Inputs must be 2D (batch, seq_len)"
                assert inputs.shape[0] == 1, "Inputs must not be batched"
                prompt_str = prompt[:]
                assert prompt_str == prompt
            case torch.Tensor():
                inputs = prompt
                assert inputs.ndim == 1, "Inputs must be 1D (seq_len)"
                prompt_str = run.tokenizer.decode(inputs)  # pyright: ignore[reportAttributeAccessIssue]
                inputs = inputs[None]

        prompt_tokens = cast(list[str], run.tokenizer.batch_decode(inputs[0]))  # pyright: ignore[reportAttributeAccessIssue]
        assert isinstance(prompt_tokens, list)
        assert isinstance(prompt_tokens[0], str)

        target_logits_out, pre_weight_acts = run.cm.forward(
            inputs,
            mode="input_cache",
            module_names=list(run.cm.components.keys()),
        )

        causal_importances, _ = run.cm.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type=run.config.sigmoid_type,
            sampling=run.config.sampling,
        )

        ci_masked_logits = run.cm(
            inputs,
            mode="components",
            mask_infos=make_mask_infos(causal_importances),
        )

        # Generate a unique prompt ID
        prompt_id = str(uuid.uuid4())

        # Store the prompt context with its ID
        prompt_context = PromptContext(
            prompt=prompt_str,
            input_token_ids=inputs[0],
            causal_importances={module: ci[0] for module, ci in causal_importances.items()},
        )
        self.prompt_contexts[prompt_id] = prompt_context

        layer_causal_importances = [
            LayerCIs(
                module=module,
                token_cis=[SparseVector.from_tensor(token_ci) for token_ci in ci],
            )
            for module, ci in prompt_context.causal_importances.items()
        ]

        return (
            prompt_id,  # Return the prompt ID first
            prompt_tokens,
            layer_causal_importances,
            self.logits_to_token_logits(target_logits_out),
            self.logits_to_token_logits(ci_masked_logits),
        )

    def run_prompt_with_mask_override(self, prompt: str | torch.Tensor, mask_override_id: str):
        """Run a prompt with a saved mask override applied to all tokens."""
        assert self.run_context_service.run_context is not None, "Run context not found"
        assert mask_override_id in self.mask_overrides, (
            f"Mask override {mask_override_id} not found"
        )

        run = self.run_context_service.run_context
        mask_override = self.mask_overrides[mask_override_id]

        # Process prompt as normal
        match prompt:
            case str():
                inputs = runtime_cast(
                    torch.Tensor, run.tokenizer.encode(prompt, return_tensors="pt")
                )
                assert inputs.ndim == 2, "Inputs must be 2D (batch, seq_len)"
                assert inputs.shape[0] == 1, "Inputs must not be batched"
                prompt_str = prompt[:]
                assert prompt_str == prompt
            case torch.Tensor():
                inputs = prompt
                assert inputs.ndim == 1, "Inputs must be 1D (seq_len)"
                prompt_str = run.tokenizer.decode(inputs)  # pyright: ignore[reportAttributeAccessIssue]
                inputs = inputs[None]

        prompt_tokens = cast(list[str], run.tokenizer.batch_decode(inputs[0]))  # pyright: ignore[reportAttributeAccessIssue]
        assert isinstance(prompt_tokens, list)
        assert isinstance(prompt_tokens[0], str)

        # Get normal forward pass
        target_logits_out, pre_weight_acts = run.cm.forward(
            inputs,
            mode="input_cache",
            module_names=list(run.cm.components.keys()),
        )

        # Calculate causal importances normally
        causal_importances, _ = run.cm.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type=run.config.sigmoid_type,
            sampling=run.config.sampling,
        )

        # Override the causal importances for the specified layer
        # Apply the mask to ALL tokens
        seq_len = inputs.shape[1]
        overridden_ci = causal_importances.copy()
        if mask_override.layer in overridden_ci:
            # Broadcast the saved mask to all token positions
            existing_shape = overridden_ci[mask_override.layer].shape
            new_mask = mask_override.combined_mask.unsqueeze(0).expand(seq_len, -1)
            assert new_mask.shape == existing_shape, (
                f"Expected {existing_shape}, got {new_mask.shape}"
            )
            overridden_ci[mask_override.layer] = new_mask

        # Run with overridden mask
        ci_masked_logits = run.cm(
            inputs,
            mode="components",
            mask_infos=make_mask_infos(overridden_ci),
        )

        # Store context
        # Generate a unique prompt ID
        prompt_id = str(uuid.uuid4())

        # Store the prompt context with overridden masks
        self.prompt_contexts[prompt_id] = PromptContext(
            prompt=prompt_str,
            input_token_ids=inputs[0],
            causal_importances={
                module: ci[0] if ci.ndim == 2 else ci for module, ci in overridden_ci.items()
            },
        )

        # Format response
        layer_causal_importances = [
            LayerCIs(
                module=module,
                token_cis=[SparseVector.from_tensor(token_ci) for token_ci in ci],
            )
            for module, ci in self.prompt_contexts[prompt_id].causal_importances.items()
        ]

        return (
            prompt_id,  # Return prompt ID first
            prompt_tokens,
            layer_causal_importances,
            self.logits_to_token_logits(target_logits_out),
            self.logits_to_token_logits(ci_masked_logits),
        )

    TOPK = 5

    def ablate_with_mask_override(
        self, prompt_id: str, mask_override_id: str
    ) -> list[list[OutputTokenLogit]]:
        """Apply a saved mask override as an ablation to a specific prompt."""
        assert self.run_context_service.run_context is not None, "Run context not found"
        assert prompt_id in self.prompt_contexts, f"Prompt {prompt_id} not found"
        assert mask_override_id in self.mask_overrides, (
            f"Mask override {mask_override_id} not found"
        )

        run = self.run_context_service.run_context
        mask_override = self.mask_overrides[mask_override_id]
        prompt_context = self.prompt_contexts[prompt_id]

        # Start with the prompt's causal importances
        masked_ci = prompt_context.causal_importances.copy()

        # Override the specified layer with the saved mask for ALL tokens
        if mask_override.layer in masked_ci:
            seq_len = len(prompt_context.input_token_ids)
            # Expand the mask to all token positions
            masked_ci[mask_override.layer] = mask_override.combined_mask.unsqueeze(0).expand(
                seq_len, -1
            )

        # Run with the mask override
        ci_masked_logits = run.cm(
            prompt_context.input_token_ids[None],
            mode="components",
            mask_infos=make_mask_infos(masked_ci),
        )

        return self.logits_to_token_logits(ci_masked_logits)

    def ablate_components(
        self, prompt_id: str, component_mask: dict[str, list[list[int]]]
    ) -> list[list[OutputTokenLogit]]:
        assert self.run_context_service.run_context is not None, "Run context not found"
        assert prompt_id in self.prompt_contexts, f"Prompt {prompt_id} not found"

        run = self.run_context_service.run_context
        prompt_context = self.prompt_contexts[prompt_id]

        masked_ci = prompt_context.causal_importances.copy()
        for module, token_masks in component_mask.items():
            for token_idx, token_mask in enumerate(token_masks):
                masked_ci[module][token_idx][token_mask] = 0

        ci_masked_logits = run.cm(
            prompt_context.input_token_ids[None],
            mode="components",
            mask_infos=make_mask_infos(masked_ci),
        )

        return self.logits_to_token_logits(ci_masked_logits)

    def logits_to_token_logits(self, logits: torch.Tensor) -> list[list[OutputTokenLogit]]:
        assert self.run_context_service.run_context is not None, "Run context not found"
        run = self.run_context_service.run_context

        assert logits.ndim == 3, "Logits must be 3D (batch, seq_len, vocab)"
        assert logits.shape[0] == 1, "Logits must not be batched"
        assert logits.shape[2] == len(run.tokenizer.vocab), (  # pyright: ignore[reportAttributeAccessIssue]
            f"Logits must have the same length as the vocabulary, {logits.shape[2]} != {len(run.tokenizer.vocab)}"  # pyright: ignore[reportAttributeAccessIssue]
        )

        tokens_logits: list[list[OutputTokenLogit]] = []
        for token_logits in logits[0]:
            assert token_logits.ndim == 1, "Token logits must be 1D (vocab)"
            token_probs = torch.softmax(token_logits, dim=-1)

            this_token_logits = []
            ordered_indices = torch.argsort(token_logits, dim=-1, descending=True)
            for tok_id in ordered_indices[: self.TOPK]:
                tok_id = int(tok_id)  # or tok_id.item()
                # tok_str = run.tokenizer.decode([tok_id], skip_special_tokens=False)  # see note below
                tok_str = run.tokenizer.convert_ids_to_tokens([tok_id])[0]  # pyright: ignore[reportAttributeAccessIssue]
                logit = float(token_logits[tok_id])
                prob = float(token_probs[tok_id])
                this_token_logits.append(
                    OutputTokenLogit(token=tok_str, logit=logit, probability=prob)
                )

            tokens_logits.append(this_token_logits)
        return tokens_logits

    def run_prompt_by_index(
        self, dataset_index: int
    ) -> tuple[
        str,
        list[str],
        list[LayerCIs],
        list[list[OutputTokenLogit]],
        list[list[OutputTokenLogit]],
    ]:
        """Run a specific prompt from the dataset by index."""
        assert (ctx := self.run_context_service.run_context) is not None, "Run context not found"

        if dataset_index >= len(ctx.train_loader.dataset):  # pyright: ignore[reportArgumentType]
            raise ValueError(
                f"Index {dataset_index} out of range for dataset of size {len(ctx.train_loader.dataset)}"  # pyright: ignore[reportArgumentType]
            )

        example = ctx.train_loader.dataset[dataset_index]["input_ids"]
        assert isinstance(example, torch.Tensor)
        assert example.ndim == 1, "Example must be 1D (seq_len)"

        return self.run_prompt(example)

    def get_cosine_similarities_of_active_components(
        self, prompt_id: str, layer: str, token_idx: int
    ) -> TokenLayerCosineSimilarityData:
        assert (run := self.run_context_service.run_context) is not None, "Run context not found"

        C = run.config.C

        assert layer in run.cm.components, f"Layer {layer} not found"

        pc = self.prompt_contexts[prompt_id]

        ci = pc.causal_importances[layer][token_idx]
        assert ci.shape == (C,), f"Expected {C} CI, got {ci.shape}"

        (nonzero_indices,) = ci.nonzero(as_tuple=True)
        assert nonzero_indices.ndim == 1, "Nonzero indices must be 1D"
        n_nonzero = nonzero_indices.shape[0]

        u_singular_vectors: Float[torch.Tensor, "C d"] = run.cm.components[layer].U[nonzero_indices]
        u_pairwise_cosine_similarities = pairwise_cosine_similarities(u_singular_vectors)
        assert u_pairwise_cosine_similarities.shape == (n_nonzero, n_nonzero), (
            f"Expected {n_nonzero}x{n_nonzero} shape, got {u_pairwise_cosine_similarities.shape}"
        )

        v_singular_vectors: Float[torch.Tensor, "d C"] = run.cm.components[layer].V[
            :, nonzero_indices
        ]
        v_pairwise_cosine_similarities = pairwise_cosine_similarities(v_singular_vectors.T)
        assert v_pairwise_cosine_similarities.shape == (n_nonzero, n_nonzero), (
            f"Expected {n_nonzero}x{n_nonzero} shape, got {v_pairwise_cosine_similarities.shape}"
        )

        return TokenLayerCosineSimilarityData(
            input_singular_vectors=u_pairwise_cosine_similarities.tolist(),
            output_singular_vectors=v_pairwise_cosine_similarities.tolist(),
            component_indices=nonzero_indices.tolist(),
        )

    def create_combined_mask(
        self,
        prompt_id: str,
        layer: str,
        token_indices: list[int],
        description: str | None = None,
        save: bool = True,
    ) -> MaskOverride:
        assert self.run_context_service.run_context is not None, "Run context not found"
        assert prompt_id in self.prompt_contexts, f"Prompt {prompt_id} not found"

        # Get the CIs for the specified layer
        prompt_context = self.prompt_contexts[prompt_id]
        layer_ci = prompt_context.causal_importances[layer]

        # Combine masks using element-wise max
        token_masks: Float[Tensor, "seq_len C"] = layer_ci[token_indices]
        combined_mask: Float[Tensor, " C"] = torch.max(token_masks, dim=0).values

        mask_override = MaskOverride(
            id=str(uuid.uuid4()),
            layer=layer,
            combined_mask=combined_mask,
            description=description,
        )

        if save:
            self.mask_overrides[mask_override.id] = mask_override

        return mask_override

    # change this to also return the metric
    def get_merge_l0(
        self, prompt_id: str, layer: str, token_indices: list[int]
    ) -> tuple[int, float]:
        mask_override = self.create_combined_mask(
            prompt_id=prompt_id,
            layer=layer,
            token_indices=token_indices,
            description=None,
            save=False,
        )
        l0 = SparseVector.from_tensor(mask_override.combined_mask).l0

        # Compute the k-way Jaccard on the original token masks for this layer
        assert prompt_id in self.prompt_contexts, f"Prompt {prompt_id} not found"
        prompt_context = self.prompt_contexts[prompt_id]
        layer_ci = prompt_context.causal_importances[layer]  # (seq_len, C)
        token_masks: Float[Tensor, "m C"] = layer_ci[token_indices]

        # Weighted/tensor Jaccard (choose this or the set version below)
        jacc = k_way_weighted_jaccard(token_masks)

        # If you prefer the set/binary interpretation (presence > 0.0), use:
        # jacc = k_way_set_jaccard(token_masks, threshold=0.0)

        return l0, jacc


def pairwise_cosine_similarities(vectors: Float[Tensor, "n d"]) -> Float[Tensor, "n n"]:
    return F.cosine_similarity(vectors[:, None, :], vectors[None, :, :], dim=-1)


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
