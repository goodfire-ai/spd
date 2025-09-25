from dataclasses import dataclass
from typing import Self

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.configs import Config
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import make_mask_infos
from spd.utils.general_utils import runtime_cast


@dataclass
class PromptContext:
    prompt: str
    input_token_ids: Int[torch.Tensor, " seq_len"]
    causal_importances: dict[str, Float[torch.Tensor, " seq_len C"]]


class TokenCIsDTO(BaseModel):
    l0: float
    component_cis: list[float]
    indices: list[int]

    @classmethod
    def from_ci(cls, ci: Float[torch.Tensor, " C"]) -> "TokenCIsDTO":
        assert ci.ndim == 1, "CI must be 1D"
        l0 = (ci > 0.0).float().sum().item()
        (nonzero_indices,) = ci.nonzero(as_tuple=True)
        return cls(
            l0=l0,
            component_cis=ci[nonzero_indices].tolist(),
            indices=nonzero_indices.tolist(),
        )


class LayerCIsDTO(BaseModel):
    module: str
    token_cis: list[TokenCIsDTO]


class OutputTokenLogitDTO(BaseModel):
    token: str
    logit: float
    probability: float


class AblationService:
    def __init__(
        self,
        config: Config,
        cm: ComponentModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.config = config
        self.cm = cm
        self.tokenizer = tokenizer

        self.prompt_context: PromptContext | None = None

    def run_prompt(self, prompt: str):
        inputs = runtime_cast(torch.Tensor, self.tokenizer.encode(prompt, return_tensors="pt"))
        assert inputs.ndim == 2, "Inputs must be 2D (batch, seq_len)"
        assert inputs.shape[0] == 1, "Inputs must not be batched"

        prompt_tokens = self.tokenizer.batch_decode(inputs[0])

        target_logits_out, pre_weight_acts = self.cm.forward(
            inputs,
            mode="input_cache",
            module_names=list(self.cm.components.keys()),
        )

        causal_importances, _ = self.cm.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type=self.config.sigmoid_type,
            sampling=self.config.sampling,
        )

        prompt_context = PromptContext(
            prompt=prompt,
            input_token_ids=inputs[0],
            causal_importances={module: ci[0] for module, ci in causal_importances.items()},
        )

        ci_masked_logits = self.cm(
            inputs,
            mode="components",
            mask_infos=make_mask_infos(causal_importances),
        )

        self.prompt_context = prompt_context

        layer_causal_importances = [
            LayerCIsDTO(
                module=module,
                token_cis=[TokenCIsDTO.from_ci(token_ci) for token_ci in ci],
            )
            for module, ci in prompt_context.causal_importances.items()
        ]

        return (
            prompt_tokens,
            layer_causal_importances,
            self.logits_to_token_logits(target_logits_out),
            self.logits_to_token_logits(ci_masked_logits),
        )

    TOPK = 5

    def modify_components(
        self,
        prompt: str,
        component_mask: dict[str, list[list[int]]],
    ) -> list[list[OutputTokenLogitDTO]]:
        assert self.prompt_context is not None, "Prompt context not found"
        assert self.prompt_context == prompt, "Prompt mismatch"

        masked_ci = self.prompt_context.causal_importances.copy()
        for module, token_masks in component_mask.items():
            for token_idx, token_mask in enumerate(token_masks):
                masked_ci[module][token_idx][token_mask] = 0

        ci_masked_logits = self.cm(
            self.prompt_context.input_token_ids,
            mode="components",
            mask_infos=make_mask_infos(masked_ci),
        )

        return self.logits_to_token_logits(ci_masked_logits)

    def logits_to_token_logits(self, logits: torch.Tensor) -> list[list[OutputTokenLogitDTO]]:
        assert logits.ndim == 3, "Logits must be 3D (batch, seq_len, vocab)"
        assert logits.shape[0] == 1, "Logits must not be batched"
        assert logits.shape[2] == len(self.tokenizer.vocab), (  # pyright: ignore[reportArgumentType]
            f"Logits must have the same length as the vocabulary, {logits.shape[2]} != {len(self.tokenizer.vocab)}"  # pyright: ignore[reportArgumentType]
        )

        tokens_logits: list[list[OutputTokenLogitDTO]] = []
        for token_logits in logits[0]:
            assert token_logits.ndim == 1, "Token logits must be 1D (vocab)"
            token_probs = torch.softmax(token_logits, dim=-1)

            this_token_logits = []
            ordered_indices = token_logits.argsort(dim=-1, descending=True)
            for token_idx in ordered_indices[: self.TOPK]:
                tok_str = self.tokenizer.decode(ordered_indices[token_idx])
                logit = token_logits[token_idx].item()
                prob = token_probs[token_idx].item()
                token_logit = OutputTokenLogitDTO(token=tok_str, logit=logit, probability=prob)
                this_token_logits.append(token_logit)

            tokens_logits.append(this_token_logits)
        return tokens_logits


    @classmethod
    def default(cls) -> Self:
        DEMO_RUN = "wandb:goodfire/spd/runs/ry05f67a"
        run_info = SPDRunInfo.from_path(DEMO_RUN)

        return cls(
            config=run_info.config,
            cm=ComponentModel.from_run_info(run_info),
            tokenizer=AutoTokenizer.from_pretrained(run_info.config.tokenizer_name),  # pyright: ignore[reportArgumentType]
        )