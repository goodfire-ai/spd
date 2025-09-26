#  %%
import uuid
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from pydantic import BaseModel
from torch._tensor import Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import make_mask_infos
from spd.utils.general_utils import runtime_cast


@dataclass
class PromptContext:
    prompt: str
    input_token_ids: Int[torch.Tensor, " seq_len"]
    causal_importances: dict[str, Float[torch.Tensor, " seq_len C"]]


class SparseVectorDTO(BaseModel):
    l0: int
    indices: list[int]
    values: list[float]

    @classmethod
    def from_tensor(cls, tensor: Float[torch.Tensor, " C"]) -> "SparseVectorDTO":
        assert tensor.ndim == 1, "Tensor must be 1D"
        l0 = (tensor > 0.0).float().sum().item()
        (nonzero_indices,) = tensor.nonzero(as_tuple=True)
        return cls(
            l0=int(l0),
            indices=nonzero_indices.tolist(),
            values=tensor[nonzero_indices].tolist(),
        )


class LayerCIsDTO(BaseModel):
    module: str
    token_cis: list[SparseVectorDTO]


class OutputTokenLogitDTO(BaseModel):
    token: str
    logit: float
    probability: float


class StatusDTO(BaseModel):
    loaded: bool
    run_id: str | None
    prompt: str | None


class TokenLayerCosineSimilarityDataDTO(BaseModel):
    input_singular_vectors: list[list[float]]
    output_singular_vectors: list[list[float]]
    component_indices: list[int]


@dataclass
class RunContext:
    wandb_id: str
    config: Config
    cm: ComponentModel
    tokenizer: PreTrainedTokenizer
    train_loader: DataLoader[Any]


@dataclass
class MaskOverrideDTO:
    id: str
    description: str | None
    layer: str
    combined_mask: SparseVectorDTO


class AvailablePromptDTO(BaseModel):
    index: int
    text: str
    full_text: str


@dataclass
class MaskOverride:
    id: str
    layer: str
    combined_mask: Float[Tensor, " C"]
    description: str | None

    def to_dto(self) -> MaskOverrideDTO:
        return MaskOverrideDTO(
            id=self.id,
            description=self.description,
            layer=self.layer,
            combined_mask=SparseVectorDTO.from_tensor(self.combined_mask),
        )


class AblationService:
    def __init__(self):
        self.run_context: RunContext | None = None
        self.prompt_contexts: dict[str, PromptContext] = {}  # Multiple prompts by ID
        self.mask_overrides: dict[str, MaskOverride] = {}

    def get_status(self) -> StatusDTO:
        if self.run_context is None:
            return StatusDTO(
                loaded=False,
                run_id=None,
                prompt=None,
            )
        return StatusDTO(
            loaded=True,
            run_id=self.run_context.wandb_id,
            prompt=None,  # No "current" prompt anymore
        )

    def run_prompt(
        self, prompt: str | torch.Tensor
    ) -> tuple[
        str,
        list[str],
        list[LayerCIsDTO],
        list[list[OutputTokenLogitDTO]],
        list[list[OutputTokenLogitDTO]],
    ]:
        assert self.run_context is not None, "Run context not found"
        run = self.run_context

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
            LayerCIsDTO(
                module=module,
                token_cis=[SparseVectorDTO.from_tensor(token_ci) for token_ci in ci],
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
        assert self.run_context is not None, "Run context not found"
        assert mask_override_id in self.mask_overrides, (
            f"Mask override {mask_override_id} not found"
        )

        run = self.run_context
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
            LayerCIsDTO(
                module=module,
                token_cis=[SparseVectorDTO.from_tensor(token_ci) for token_ci in ci],
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
    ) -> list[list[OutputTokenLogitDTO]]:
        """Apply a saved mask override as an ablation to a specific prompt."""
        assert self.run_context is not None, "Run context not found"
        assert prompt_id in self.prompt_contexts, f"Prompt {prompt_id} not found"
        assert mask_override_id in self.mask_overrides, (
            f"Mask override {mask_override_id} not found"
        )

        run = self.run_context
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
    ) -> list[list[OutputTokenLogitDTO]]:
        assert self.run_context is not None, "Run context not found"
        assert prompt_id in self.prompt_contexts, f"Prompt {prompt_id} not found"

        run = self.run_context
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

    def logits_to_token_logits(self, logits: torch.Tensor) -> list[list[OutputTokenLogitDTO]]:
        assert self.run_context is not None, "Run context not found"
        run = self.run_context

        assert logits.ndim == 3, "Logits must be 3D (batch, seq_len, vocab)"
        assert logits.shape[0] == 1, "Logits must not be batched"
        assert logits.shape[2] == len(run.tokenizer.vocab), (  # pyright: ignore[reportAttributeAccessIssue]
            f"Logits must have the same length as the vocabulary, {logits.shape[2]} != {len(run.tokenizer.vocab)}"  # pyright: ignore[reportAttributeAccessIssue]
        )

        tokens_logits: list[list[OutputTokenLogitDTO]] = []
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
                    OutputTokenLogitDTO(token=tok_str, logit=logit, probability=prob)
                )

            tokens_logits.append(this_token_logits)
        return tokens_logits

    def load_run_from_wandb_id(self, wandb_id: str):
        path = f"wandb:goodfire/spd/runs/{wandb_id}"
        run_info = SPDRunInfo.from_path(path)

        task_config = runtime_cast(LMTaskConfig, run_info.config.task_config)

        train_data_config = DatasetConfig(
            name=task_config.dataset_name,
            hf_tokenizer_path=run_info.config.tokenizer_name,
            split=task_config.train_data_split,
            n_ctx=task_config.max_seq_len,
            is_tokenized=task_config.is_tokenized,
            streaming=task_config.streaming,
            column_name=task_config.column_name,
            shuffle_each_epoch=task_config.shuffle_each_epoch,
            seed=None,
        )

        batch_size = 1

        train_loader, tokenizer = create_data_loader(
            dataset_config=train_data_config,
            batch_size=batch_size,
            buffer_size=task_config.buffer_size,
            global_seed=run_info.config.seed,
            ddp_rank=0,
            ddp_world_size=0,
        )

        self.run_context = RunContext(
            wandb_id=wandb_id,
            config=run_info.config,
            cm=ComponentModel.from_run_info(run_info),
            tokenizer=tokenizer,
            train_loader=train_loader,
        )

    def get_available_prompts(self) -> list[AvailablePromptDTO]:
        """Get first 100 prompts from the dataset with their indices and text."""
        assert (ctx := self.run_context) is not None, "Run context not found"

        prompts = []
        for idx in range(min(100, len(ctx.train_loader.dataset))):  # pyright: ignore[reportArgumentType]
            example = ctx.train_loader.dataset[idx]["input_ids"]
            assert isinstance(example, torch.Tensor)
            assert example.ndim == 1, "Example must be 1D (seq_len)"

            # Decode to text for display
            text = ctx.tokenizer.decode(example, skip_special_tokens=True)  # pyright: ignore[reportAttributeAccessIssue]

            prompts.append(
                AvailablePromptDTO(
                    index=idx,
                    text=text[:200] + "..." if len(text) > 200 else text,  # Truncate for display
                    full_text=text,
                )
            )

        return prompts

    def run_prompt_by_index(
        self, dataset_index: int
    ) -> tuple[
        str,
        list[str],
        list[LayerCIsDTO],
        list[list[OutputTokenLogitDTO]],
        list[list[OutputTokenLogitDTO]],
    ]:
        """Run a specific prompt from the dataset by index."""
        assert (ctx := self.run_context) is not None, "Run context not found"

        if dataset_index >= len(ctx.train_loader.dataset):  # pyright: ignore[reportArgumentType]
            raise ValueError(
                f"Index {dataset_index} out of range for dataset of size {len(ctx.train_loader.dataset)}"  # pyright: ignore[reportArgumentType]
            )

        example = ctx.train_loader.dataset[dataset_index]["input_ids"]
        assert isinstance(example, torch.Tensor)
        assert example.ndim == 1, "Example must be 1D (seq_len)"

        return self.run_prompt(example)

    def get_cosine_similarities(
        self, prompt_id: str, layer: str, token_idx: int
    ) -> TokenLayerCosineSimilarityDataDTO:
        assert (run := self.run_context) is not None, "Run context not found"

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

        return TokenLayerCosineSimilarityDataDTO(
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
        assert self.run_context is not None, "Run context not found"
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
        # Build the combined mask as before (element-wise max)
        mask_override = self.create_combined_mask(
            prompt_id=prompt_id,
            layer=layer,
            token_indices=token_indices,
            description=None,
            save=False,
        )
        l0 = SparseVectorDTO.from_tensor(mask_override.combined_mask).l0

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


# def create_data_loader(
#     dataset_config: DatasetConfig,
#     batch_size: int,
#     buffer_size: int,
#     global_seed: int = 0,
#     ddp_rank: int = 0,
#     ddp_world_size: int = 1,
#     to_lower: bool = True,
# ) -> tuple[DataLoader[Any], PreTrainedTokenizer]:
#     """Create a DataLoader for the given dataset.

#     Uses PyTorch's DistributedSampler to ensure each rank gets the correct
#     subset of data in distributed mode. The sampler ensures that:
#     - Each rank gets a different subset of the data
#     - Data is distributed deterministically based on rank
#     - No data is duplicated across ranks

#     For shuffling datasets between epochs:
#     - When dp>1 and streaming=True, we shard the dataset across ranks and use the default sampler.
#         If the dataset has fewer dataset shards than we have ddp ranks, we split up the dataset
#         by example. We also use dataset.set_epoch(epoch) during training.
#     - When dp>1 and streaming=False, we use a DistributedSampler and run
#         sampler.set_epoch(epoch) during training.
#     - When dp=1 and streaming=True, we use the default sampler and run
#         dataset.set_epoch(epoch) during training.
#     - When dp=1 and streaming=False, we use the default sampler and set shuffle=True on the
#         DataLoader.

#     Args:
#         dataset_config: The configuration for the dataset.
#         batch_size: The batch size.
#         buffer_size: The buffer size for streaming datasets.
#         global_seed: Used for shuffling if dataset_config.seed is None.
#         ddp_rank: The rank of the current process in DDP.
#         ddp_world_size: The world size in DDP.

#     Returns:
#         A tuple of the DataLoader and the tokenizer.
#     """

#     dataset = load_dataset(
#         dataset_config.name,
#         streaming=dataset_config.streaming,
#         split=dataset_config.split,
#         trust_remote_code=False,
#     )

#     seed = dataset_config.seed if dataset_config.seed is not None else global_seed

#     assert not dataset_config.streaming
#     assert isinstance(dataset, Dataset)
#     dataset = dataset.shuffle(seed=seed)

#     tokenizer = AutoTokenizer.from_pretrained(dataset_config.hf_tokenizer_path)

#     torch_dataset: Dataset
#     if dataset_config.is_tokenized:
#         torch_dataset = dataset.with_format("torch")
#         # Verify tokenization
#         sample = next(iter(torch_dataset))[dataset_config.column_name]  # pyright: ignore[reportCallIssue, reportArgumentType]
#         assert isinstance(sample, Tensor) and sample.ndim == 1, (
#             f"Expected the dataset to be tokenized. Got type {type(sample)}"
#         )
#         assert len(sample) == dataset_config.n_ctx, (
#             f"n_ctx ({dataset_config.n_ctx}) does not match the tokenized length ({len(sample)})."
#         )
#     else:
#         to_lower = "SimpleStories" in dataset_config.name
#         torch_dataset = tokenize_and_concatenate(
#             dataset,  # pyright: ignore[reportArgumentType]
#             tokenizer,
#             max_length=dataset_config.n_ctx,
#             column_name=dataset_config.column_name,
#             add_bos_token=False,
#             to_lower=to_lower,
#         )

#     sampler = None
#     if not dataset_config.streaming and is_ddp:
#         sampler = DistributedSampler(
#             torch_dataset,  # pyright: ignore[reportArgumentType]
#             num_replicas=ddp_world_size,
#             rank=ddp_rank,
#             shuffle=dataset_config.shuffle_each_epoch,
#             seed=seed,
#             drop_last=True,
#         )

#     # Ensure determinicity when not distributed and not streaming
#     generator = torch.Generator(device="cpu")
#     generator.manual_seed(seed)

#     loader = DataLoader[Dataset | IterableDataset](
#         torch_dataset,  # pyright: ignore[reportArgumentType]
#         batch_size=batch_size,
#         sampler=sampler,
#         shuffle=(
#             sampler is None and dataset_config.shuffle_each_epoch and not dataset_config.streaming
#         ),
#         drop_last=True,
#         generator=generator,
#     )
#     return loader, tokenizer
