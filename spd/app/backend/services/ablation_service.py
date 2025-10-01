# %%
import uuid
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from pydantic import BaseModel
from torch._tensor import Tensor

from spd.app.backend.services.run_context_service import RunContextService
from spd.log import logger
from spd.models.components import make_mask_infos
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import runtime_cast


@dataclass
class PromptContext:
    prompt: str
    input_token_ids: Int[torch.Tensor, " seq_len"]
    subcomponent_causal_importances: dict[str, Float[torch.Tensor, " seq_len C"]]


N_MOCK_COMPONENTS = 20


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


class Component(BaseModel):
    index: int
    subcomponent_indices: list[int]


class MatrixCausalImportances(BaseModel):
    subcomponent_cis: SparseVector
    """the CI values for each subcomponent"""

    # component_indices: list[int]
    # """len: C. the index of the component that the each subcomponent belongs to"""

    component_agg_cis: list[float]  # or SparseVector
    """len: M. For each component, the assigned CI value as aggregated from the subcomponents (usually via
    max)"""

    components: list[Component]


class LayerCIs(BaseModel):
    module: str
    token_cis: list[MatrixCausalImportances]


class OutputTokenLogit(BaseModel):
    token: str
    logit: float
    probability: float


class RunResponse(BaseModel):
    prompt_id: str
    prompt_tokens: list[str]
    layer_cis: list[LayerCIs]
    full_run_token_logits: list[list[OutputTokenLogit]]
    ci_masked_token_logits: list[list[OutputTokenLogit]]


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


DEVICE = get_device()


class AblationService:
    def __init__(self, run_context_service: RunContextService):
        self.run_context_service = run_context_service
        self.prompt_contexts: dict[str, PromptContext] = {}  # Multiple prompts by ID
        self.mask_overrides: dict[str, MaskOverride] = {}

        self.__mock_component_indices = None  # REMOVE ME

    def _mock_component_indices(self) -> Int[Tensor, " C"]:
        assert (ctx := self.run_context_service.run_context) is not None, "Run context not found"
        C = ctx.config.C
        # groups of N_MOCK_COMPONENTS subcomponents
        if self.__mock_component_indices is None:
            self.__mock_component_indices = torch.randint(0, N_MOCK_COMPONENTS, (C,)).to(DEVICE)
        return self.__mock_component_indices

    def mock_aggregate_cis(self, cis: Float[Tensor, "seq_len C"]) -> Float[Tensor, "seq_len M"]:
        # Pretend it's the same for each layer
        assert (cfg := self.run_context_service.run_context) is not None, "Run context not found"

        n_toks, n_subcomponents = cis.shape

        seq_len = 24
        assert n_toks == seq_len, "for now"
        assert n_subcomponents == cfg.config.C

        # TODO not sure this is right
        component_indices: Int[Tensor, " C"] = self._mock_component_indices()

        component_agg_cis = torch.full(
            (n_toks, N_MOCK_COMPONENTS), float("-inf"), dtype=cis.dtype, device=cis.device
        )
        component_agg_cis.scatter_reduce_(
            dim=1,
            index=component_indices.unsqueeze(0).repeat(n_toks, 1),
            src=cis,
            reduce="amax",
            include_self=True,
        )

        return component_agg_cis

    def _materialize_prompt(
        self, prompt: str | torch.Tensor
    ) -> tuple[str, list[str], torch.Tensor]:
        assert self.run_context_service.run_context is not None, "Run context not found"
        run = self.run_context_service.run_context

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
        assert (ctx := self.run_context_service.run_context) is not None

        prompt_str, prompt_tokens, inputs = self._materialize_prompt(prompt)

        logger.info(f"Inputs shape: {inputs.shape}")

        target_logits_out, pre_weight_acts = ctx.cm.forward(
            inputs[None],
            mode="input_cache",
            module_names=list(ctx.cm.components.keys()),
        )

        logger.info(f"Pre-weight acts shape: {pre_weight_acts.keys()}")

        causal_importances, _ = ctx.cm.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type=ctx.config.sigmoid_type,
            sampling=ctx.config.sampling,
        )

        logger.info(f"Causal importances shape: {causal_importances.keys()}")

        # Run with overridden mask
        ci_masked_logits = ctx.cm(
            inputs[None],
            mode="components",
            mask_infos=make_mask_infos(causal_importances),
        )

        logger.info(f"CI masked logits shape: {ci_masked_logits.shape}")

        # Store context
        # Generate a unique prompt ID
        prompt_id = str(uuid.uuid4())

        # Store the prompt context with overridden masks
        self.prompt_contexts[prompt_id] = PromptContext(
            prompt=prompt_str,
            input_token_ids=inputs,
            subcomponent_causal_importances={k: v[0] for k, v in causal_importances.items()},
        )

        logger.info("saved")

        return RunResponse(
            prompt_id=prompt_id,
            prompt_tokens=prompt_tokens,
            layer_cis=self._to_layer_cis({k: v[0] for k, v in causal_importances.items()}),
            full_run_token_logits=self._logits_to_token_logits(target_logits_out[0]),
            ci_masked_token_logits=self._logits_to_token_logits(ci_masked_logits[0]),
        )

    def _to_layer_cis(
        self,
        causal_importances: dict[str, Float[Tensor, "seq_len C"]],
    ) -> list[LayerCIs]:
        assert (ctx := self.run_context_service.run_context) is not None

        component_indices = self._mock_component_indices()

        layer_cis = []
        for module, layer_ci in causal_importances.items():
            component_agg_cis = self.mock_aggregate_cis(layer_ci)
            token_cis = []

            for tok_component_agg_ci, tok_layer_ci in zip(component_agg_cis, layer_ci, strict=True):
                assert tok_component_agg_ci.shape == (N_MOCK_COMPONENTS,)
                # assert tok_component_indices.shape == (ctx.config.C,), (
                #     f"got {tok_component_indices.shape}"
                # )

                assert tok_layer_ci.shape == (ctx.config.C,), f"got {tok_layer_ci.shape}"

                components = []
                for i in range(N_MOCK_COMPONENTS):
                    subcomponent_indices = torch.where(component_indices == i)[0].tolist()
                    components.append(Component(index=i, subcomponent_indices=subcomponent_indices))

                token_cis.append(
                    MatrixCausalImportances(
                        subcomponent_cis=SparseVector.from_tensor(tok_layer_ci),
                        # component_indices=tok_component_indices.tolist(),
                        component_agg_cis=tok_component_agg_ci.tolist(),
                        components=components,
                    )
                )

            layer_cis.append(LayerCIs(module=module, token_cis=token_cis))

        return layer_cis

    def run_with_mask_override(
        self, prompt_id: str, mask_override_id: str
    ) -> list[list[OutputTokenLogit]]:
        """Apply a saved mask override as an ablation to a specific prompt."""
        assert (ctx := self.run_context_service.run_context) is not None
        assert (prompt_context := self.prompt_contexts.get(prompt_id)) is not None
        assert (mask_override := self.mask_overrides.get(mask_override_id)) is not None

        # Start with the prompt's causal importances
        masked_ci = prompt_context.subcomponent_causal_importances.copy()

        # Override the specified layer with the saved mask for ALL tokens
        masked_ci[mask_override.layer].copy_(mask_override.combined_mask)

        # Run with the mask override
        ci_masked_logits = ctx.cm(
            prompt_context.input_token_ids[None],
            mode="components",
            mask_infos=make_mask_infos(masked_ci),
        )[0]

        return self._logits_to_token_logits(ci_masked_logits)

    def ablate_subcomponents(
        self, prompt_id: str, subcomponent_mask: dict[str, list[list[int]]]
    ) -> list[list[OutputTokenLogit]]:
        assert (ctx := self.run_context_service.run_context) is not None
        assert (prompt_context := self.prompt_contexts.get(prompt_id)) is not None

        masked_ci = prompt_context.subcomponent_causal_importances.copy()
        for module, token_subcomponent_masks in subcomponent_mask.items():
            for token_idx, token_subcomponent_mask in enumerate(token_subcomponent_masks):
                masked_ci[module][token_idx][token_subcomponent_mask] = 0

        ci_masked_logits = ctx.cm(
            prompt_context.input_token_ids[None],
            mode="components",
            mask_infos=make_mask_infos(masked_ci),
        )[0]

        return self._logits_to_token_logits(ci_masked_logits)

    def ablate_components(
        self, prompt_id: str, component_mask: dict[str, list[list[int]]]
    ) -> list[list[OutputTokenLogit]]:
        assert (ctx := self.run_context_service.run_context) is not None
        assert (prompt_context := self.prompt_contexts.get(prompt_id)) is not None

        component_indices = self._mock_component_indices()

        masked_ci = prompt_context.subcomponent_causal_importances.copy()
        for module, token_component_masks in component_mask.items():
            for token_idx, token_component_mask in enumerate(token_component_masks):
                for component_idx in token_component_mask:
                    component_subcomponent_indices = torch.where(
                        component_indices == component_idx
                    )[0].tolist()
                    masked_ci[module][token_idx][component_subcomponent_indices] = 0

        ci_masked_logits = ctx.cm(
            prompt_context.input_token_ids[None],
            mode="components",
            mask_infos=make_mask_infos(masked_ci),
        )[0]

        return self._logits_to_token_logits(ci_masked_logits)

    TOP_K = 5

    def _logits_to_token_logits(
        self,
        logits: Float[torch.Tensor, "seq_len vocab"],
    ) -> list[list[OutputTokenLogit]]:
        assert (ctx := self.run_context_service.run_context) is not None, "Run context not found"

        assert logits.ndim == 2, "Logits must be 2D (seq_len, vocab)"
        assert logits.shape[0] == 24
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
        assert (ctx := self.run_context_service.run_context) is not None, "Run context not found"

        if dataset_index >= len(ctx.train_loader.dataset):  # pyright: ignore[reportArgumentType]
            raise ValueError(
                f"Index {dataset_index} out of range for dataset of size {len(ctx.train_loader.dataset)}"  # pyright: ignore[reportArgumentType]
            )

        example = ctx.train_loader.dataset[dataset_index]["input_ids"]

        assert isinstance(example, torch.Tensor)
        assert example.ndim == 1, "Example must be 1D (seq_len)"
        logger.info(f"Running prompt by index: {dataset_index}, shape: {example.shape}")

        return self.run_prompt(example)

    def get_subcomponent_cosine_sims(
        self, layer: str, component_idx: int
    ) -> TokenLayerCosineSimilarityData:
        assert (run := self.run_context_service.run_context) is not None, "Run context not found"
        assert (components := run.cm.components.get(layer)) is not None, f"Layer {layer} not found"

        logger.info(f"component index: {component_idx}")
        component_subcomponent_indices = torch.where(
            self._mock_component_indices() == component_idx
        )[0]

        logger.info(f"component subcomponent indices: {component_subcomponent_indices.shape}")

        assert component_subcomponent_indices.ndim == 1, "Nonzero indices must be 1D"
        n_nonzero = component_subcomponent_indices.shape[0]

        u_singular_vectors: Float[torch.Tensor, "C d"] = components.U[
            component_subcomponent_indices.tolist()
        ]
        u_pairwise_cosine_similarities = pairwise_cosine_similarities(u_singular_vectors)
        assert u_pairwise_cosine_similarities.shape == (n_nonzero, n_nonzero), (
            f"Expected {n_nonzero}x{n_nonzero} shape, got {u_pairwise_cosine_similarities.shape}"
        )

        v_singular_vectors: Float[torch.Tensor, "d C"] = components.V[
            :, component_subcomponent_indices.tolist()
        ]
        v_pairwise_cosine_similarities = pairwise_cosine_similarities(v_singular_vectors.T)
        assert v_pairwise_cosine_similarities.shape == (n_nonzero, n_nonzero), (
            f"Expected {n_nonzero}x{n_nonzero} shape, got {v_pairwise_cosine_similarities.shape}"
        )

        logger.info(f"U pairwise cosine similarities: {u_pairwise_cosine_similarities.shape}")
        logger.info(f"V pairwise cosine similarities: {v_pairwise_cosine_similarities.shape}")
        logger.info(f"Component indices: {component_subcomponent_indices.tolist()}")

        return TokenLayerCosineSimilarityData(
            input_singular_vectors=u_pairwise_cosine_similarities.tolist(),
            output_singular_vectors=v_pairwise_cosine_similarities.tolist(),
            component_indices=component_subcomponent_indices.tolist(),
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
        layer_ci = prompt_context.subcomponent_causal_importances[layer]

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
        layer_ci = prompt_context.subcomponent_causal_importances[layer]  # (seq_len, C)
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

