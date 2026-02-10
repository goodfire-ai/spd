"""Canonical transformer topology.

Defines a normalized addressing scheme for transformer layers. All API
consumers use canonical addresses instead of raw module paths.

Canonical layer address format:
    "wte"              — embedding
    "output"           — unembed / logits
    "{block}.attn.{p}" — attention projection (p: q | k | v | qkv | o)
    "{block}.ffn.{p}"  — FFN projection (p: up | down | gate)

Node key format (layer address + position):
    "{layer_address}:{seq_pos}:{component_idx}"

Examples:
    "0.attn.q"         — block 0, attention Q projection
    "2.ffn.gate"       — block 2, SwiGLU gate projection
    "0.attn.q:3:5"     — block 0, Q proj, seq pos 3, component 5
    "wte:0:0"          — embedding, seq 0, pseudo-component 0
    "output:7:42"       — output, seq 7, token 42
"""

from dataclasses import dataclass
from typing import Literal

from torch import nn


@dataclass
class GLUPathSchema:
    base: str
    gate: str
    up: str
    down: str


@dataclass
class FFNPathSchema:
    base: str
    up: str
    down: str


@dataclass
class SeparateAttnPathSchema:
    base: str
    q: str
    k: str
    v: str
    o: str


@dataclass
class FusedAttentionPathSchema:
    base: str
    qkv: str
    o: str


@dataclass(frozen=True)
class PathSchema:
    """Maps concrete module path patterns to canonical projections."""

    embedding_path: str
    blocks: str
    attn: SeparateAttnPathSchema | FusedAttentionPathSchema
    mlp: GLUPathSchema | FFNPathSchema
    unembed_path: str


_LLAMA_SIMPLE_CONFIG = PathSchema(
    embedding_path="wte",
    blocks="h",
    attn=SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj"),
    mlp=GLUPathSchema(base="mlp", gate="gate_proj", up="up_proj", down="down_proj"),
    unembed_path="lm_head",
)

_LLAMA_SIMPLE_MLP_CONFIG = PathSchema(
    embedding_path="wte",
    blocks="h",
    attn=SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj"),
    mlp=FFNPathSchema(base="mlp", up="c_fc", down="down_proj"),
    unembed_path="lm_head",
)

_GPT2_SIMPLE_CONFIG = PathSchema(
    embedding_path="wte",
    blocks="h",
    attn=SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj"),
    mlp=FFNPathSchema(base="mlp", up="c_fc", down="down_proj"),
    unembed_path="lm_head",
)

_GPT2_CONFIG = PathSchema(
    embedding_path="wte",
    blocks="h_torch",
    attn=FusedAttentionPathSchema(base="attn", qkv="c_attn", o="c_proj"),
    mlp=FFNPathSchema(base="mlp", up="c_fc", down="c_proj"),
    unembed_path="lm_head",
)

_HF_GPT2_CONFIG = PathSchema(
    embedding_path="transformer.wte",
    blocks="transformer.h",
    attn=FusedAttentionPathSchema(base="attn", qkv="c_attn", o="c_proj"),
    mlp=FFNPathSchema(base="mlp", up="c_fc", down="c_proj"),
    unembed_path="lm_head",
)


def _get_path_schema(model: nn.Module) -> PathSchema:
    from transformers.models.gpt2 import GPT2LMHeadModel

    from spd.pretrain.models import GPT2, GPT2Simple, LlamaSimple, LlamaSimpleMLP

    match model:
        case LlamaSimple():
            return _LLAMA_SIMPLE_CONFIG
        case LlamaSimpleMLP():
            return _LLAMA_SIMPLE_MLP_CONFIG
        case GPT2Simple():
            return _GPT2_SIMPLE_CONFIG
        case GPT2():
            return _GPT2_CONFIG
        case GPT2LMHeadModel():
            return _HF_GPT2_CONFIG
        case _:
            raise ValueError(
                f"Unsupported model class {type(model).__name__}. Add a PathSchema in topology.py."
            )


# canonical representations
class Embed: ...


class Unembed: ...


class SeparateAttnWeight:
    weight: Literal["q", "k", "v", "o"]


class FusedAttnWeight:
    weight: Literal["qkv", "o"]


AttnWeight = FusedAttnWeight | SeparateAttnWeight


class GLUWeight:
    weight: Literal["up", "down", "gate"]


class MLPWeight:
    weight: Literal["up", "down"]


FFNWeight = GLUWeight | MLPWeight


@dataclass
class LayerWeight:
    layer_idx: int
    name: AttnWeight | FFNWeight


CanonicalWeight = Embed | LayerWeight | Unembed


class TransformerTopology:
    """Bidirectional mapping between canonical addresses and concrete module paths.

    Built from a target model (nn.Module). Independent of decomposition —
    discovers the full model structure via named_modules().
    """

    def __init__(self, target_model: nn.Module) -> None:
        self.target_model = target_model
        self.path_schema = _get_path_schema(target_model)
        # self.canonical_weights = self.build_canonical_weights()

    def get_target_module_path(self, canonical: CanonicalWeight) -> str:
        match canonical:
            case Embed():
                return self.path_schema.embedding_path
            case Unembed():
                return self.path_schema.unembed_path
            case LayerWeight(layer_idx=layer_idx, name=name):
                base = f"{self.path_schema.blocks}.{layer_idx}"
                match name:
                    case SeparateAttnWeight(weight=weight):
                        assert isinstance(self.path_schema.attn, SeparateAttnPathSchema)
                        match weight:
                            case "q":
                                return f"{base}.{self.path_schema.attn.q}"
                            case "k":
                                return f"{base}.{self.path_schema.attn.k}"
                            case "v":
                                return f"{base}.{self.path_schema.attn.v}"
                            case "o":
                                return f"{base}.{self.path_schema.attn.o}"
                    case FusedAttnWeight(weight=weight):
                        assert isinstance(self.path_schema.attn, FusedAttentionPathSchema)
                        match weight:
                            case "qkv":
                                return f"{base}.{self.path_schema.attn.qkv}"
                            case "o":
                                return f"{base}.{self.path_schema.attn.o}"
                    case GLUWeight(weight=weight):
                        assert isinstance(self.path_schema.mlp, GLUPathSchema)
                        match weight:
                            case "up":
                                return f"{base}.{self.path_schema.mlp.up}"
                            case "down":
                                return f"{base}.{self.path_schema.mlp.down}"
                            case "gate":
                                return f"{base}.{self.path_schema.mlp.gate}"
                    case MLPWeight(weight=weight):
                        assert isinstance(self.path_schema.mlp, FFNPathSchema)
                        match weight:
                            case "up":
                                return f"{base}.{self.path_schema.mlp.up}"
                            case "down":
                                return f"{base}.{self.path_schema.mlp.down}"

    def get_canonical_weight(self, target_module_path: str) -> CanonicalWeight:
        # TODO: parse path based on path_schema and return the corresponding CanonicalWeight
        raise NotImplementedError("Not implemented")

    @property
    def n_blocks(self) -> int:
        blocks = self.target_model.get_submodule(self.path_schema.blocks)
        assert isinstance(blocks, nn.ModuleList)
        return len(blocks)

    def is_cross_seq_pair(self, source: CanonicalWeight, target: CanonicalWeight) -> bool:
        """True if source is k/v and target is o in the same block.

        Takes canonical addresses.
        """
        match source, target:
            case (
                LayerWeight(layer_idx=source_idx, name=SeparateAttnWeight(weight="k")),
                LayerWeight(layer_idx=target_idx, name=SeparateAttnWeight(weight="q")),
            ):
                return source_idx < target_idx
            case (
                LayerWeight(layer_idx=source_idx, name=FusedAttnWeight(weight="qkv")),
                LayerWeight(layer_idx=target_idx, name=FusedAttnWeight(weight=_)),
            ):
                return source_idx < target_idx
            case _:
                return False

    def describe(self, canonical: CanonicalWeight) -> str:
        """Human-readable description for autointerp prompts."""
        match canonical:
            case Embed():
                return "embedding"
            case LayerWeight(name=name):
                match name:
                    case SeparateAttnWeight(weight=weight):
                        match weight:
                            case "q":
                                return "attention Q projection"
                            case "k":
                                return "attention K projection"
                            case "v":
                                return "attention V projection"
                            case "o":
                                return "attention output projection"
                    case FusedAttnWeight(weight=weight):
                        match weight:
                            case "qkv":
                                return "attention QKV projection (fused)"
                            case "o":
                                return "attention output projection"
                    case GLUWeight(weight=weight):
                        match weight:
                            case "up":
                                return "GLU up-projection"
                            case "down":
                                return "GLU down-projection"
                            case "gate":
                                return "GLU gate projection"
                    case MLPWeight(weight=weight):
                        match weight:
                            case "up":
                                return "MLP up-projection"
                            case "down":
                                return "MLP down-projection"
            case Unembed():
                return "unembedding"
