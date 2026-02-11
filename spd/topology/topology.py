"""Bidirectional mapping between canonical weights and concrete module paths.

Depends on torch.nn and specific model classes. For pure canonical types
without torch, use canonical.py directly.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Literal

from torch import nn

from spd.topology.canonical import (
    CanonicalWeight,
    Embed,
    FusedAttnWeight,
    GLUWeight,
    LayerWeight,
    MLPWeight,
    SeparateAttnWeight,
    Unembed,
)


@dataclass
class _SeparateAttnPathSchema:
    base: str
    q: str
    k: str
    v: str
    o: str

    def _lookup(self) -> dict[str, Literal["q", "k", "v", "o"]]:
        return {self.q: "q", self.k: "k", self.v: "v", self.o: "o"}

    def _reverse(self) -> dict[str, str]:
        return {"q": self.q, "k": self.k, "v": self.v, "o": self.o}

    def parse(self, projection_name: str, layer_idx: int) -> LayerWeight:
        table = self._lookup()
        assert projection_name in table, f"Unknown attn projection: {projection_name}"
        return LayerWeight(layer_idx, SeparateAttnWeight(table[projection_name]))

    def render(self, w: SeparateAttnWeight) -> str:
        return f"{self.base}.{self._reverse()[w.weight]}"


@dataclass
class _FusedAttnPathSchema:
    base: str
    qkv: str
    o: str

    def _lookup(self) -> dict[str, Literal["qkv", "o"]]:
        return {self.qkv: "qkv", self.o: "o"}

    def _reverse(self) -> dict[str, str]:
        return {"qkv": self.qkv, "o": self.o}

    def parse(self, projection_name: str, layer_idx: int) -> LayerWeight:
        table = self._lookup()
        assert projection_name in table, f"Unknown fused attn projection: {projection_name}"
        return LayerWeight(layer_idx, FusedAttnWeight(table[projection_name]))

    def render(self, w: FusedAttnWeight) -> str:
        return f"{self.base}.{self._reverse()[w.weight]}"


@dataclass
class _GLUPathSchema:
    base: str
    gate: str
    up: str
    down: str

    def _lookup(self) -> dict[str, Literal["up", "down", "gate"]]:
        return {self.gate: "gate", self.up: "up", self.down: "down"}

    def _reverse(self) -> dict[str, str]:
        return {"gate": self.gate, "up": self.up, "down": self.down}

    def parse(self, projection_name: str, layer_idx: int) -> LayerWeight:
        table = self._lookup()
        assert projection_name in table, f"Unknown GLU projection: {projection_name}"
        return LayerWeight(layer_idx, GLUWeight(table[projection_name]))

    def render(self, w: GLUWeight) -> str:
        return f"{self.base}.{self._reverse()[w.weight]}"


@dataclass
class _FFNPathSchema:
    base: str
    up: str
    down: str

    def _lookup(self) -> dict[str, Literal["up", "down"]]:
        return {self.up: "up", self.down: "down"}

    def _reverse(self) -> dict[str, str]:
        return {"up": self.up, "down": self.down}

    def parse(self, projection_name: str, layer_idx: int) -> LayerWeight:
        table = self._lookup()
        assert projection_name in table, f"Unknown MLP projection: {projection_name}"
        return LayerWeight(layer_idx, MLPWeight(table[projection_name]))

    def render(self, w: MLPWeight) -> str:
        return f"{self.base}.{self._reverse()[w.weight]}"


# ── Path schema (base + concrete subclasses) ──────────────────────────


class _PathSchema(ABC):
    embedding_path: str
    blocks: str
    attn: _SeparateAttnPathSchema | _FusedAttnPathSchema
    mlp: _GLUPathSchema | _FFNPathSchema
    unembed_path: str

    def parse_target_path(self, path: str) -> CanonicalWeight:
        if path == self.embedding_path:
            return Embed()
        if path == self.unembed_path:
            return Unembed()
        return self._parse_block_path(path)

    def render_canonical_weight(self, weight: CanonicalWeight) -> str:
        match weight:
            case Embed():
                return self.embedding_path
            case Unembed():
                return self.unembed_path
            case LayerWeight() as lw:
                return self._render_layer_weight(lw)
            case _:
                raise ValueError(f"Unknown canonical weight: {weight!r}")

    def _parse_block_path(self, path: str) -> LayerWeight:
        """Parse a block-level path like 'h.3.attn.q_proj' into a LayerWeight."""
        assert path.startswith(self.blocks + ".")
        remainder = path[len(self.blocks) + 1 :]
        dot = remainder.index(".")
        layer_idx = int(remainder[:dot])
        sublayer_and_proj = remainder[dot + 1 :]

        if sublayer_and_proj.startswith(self.attn.base + "."):
            proj = sublayer_and_proj[len(self.attn.base) + 1 :]
            return self.attn.parse(proj, layer_idx)

        assert sublayer_and_proj.startswith(self.mlp.base + ".")
        proj = sublayer_and_proj[len(self.mlp.base) + 1 :]
        return self.mlp.parse(proj, layer_idx)

    def _render_layer_weight(self, w: LayerWeight) -> str:
        """Render a LayerWeight into a concrete path."""
        base = f"{self.blocks}.{w.layer_idx}"
        match w.name:
            case SeparateAttnWeight() as attn_w:
                assert isinstance(self.attn, _SeparateAttnPathSchema)
                return f"{base}.{self.attn.render(attn_w)}"
            case FusedAttnWeight() as attn_w:
                assert isinstance(self.attn, _FusedAttnPathSchema)
                return f"{base}.{self.attn.render(attn_w)}"
            case GLUWeight() as ffn_w:
                assert isinstance(self.mlp, _GLUPathSchema)
                return f"{base}.{self.mlp.render(ffn_w)}"
            case MLPWeight() as ffn_w:
                assert isinstance(self.mlp, _FFNPathSchema)
                return f"{base}.{self.mlp.render(ffn_w)}"


class _LlamaSimplePathSchema(_PathSchema):
    embedding_path = "wte"
    blocks = "h"
    attn = _SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    mlp = _GLUPathSchema(base="mlp", gate="gate_proj", up="up_proj", down="down_proj")
    unembed_path = "lm_head"


class _LlamaSimpleMLPPathSchema(_PathSchema):
    embedding_path = "wte"
    blocks = "h"
    attn = _SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    mlp = _FFNPathSchema(base="mlp", up="c_fc", down="down_proj")
    unembed_path = "lm_head"


class _GPT2SimplePathSchema(_PathSchema):
    embedding_path = "wte"
    blocks = "h"
    attn = _SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    mlp = _FFNPathSchema(base="mlp", up="c_fc", down="down_proj")
    unembed_path = "lm_head"


class _GPT2PathSchema(_PathSchema):
    embedding_path = "wte"
    blocks = "h_torch"
    attn = _FusedAttnPathSchema(base="attn", qkv="c_attn", o="c_proj")
    mlp = _FFNPathSchema(base="mlp", up="c_fc", down="c_proj")
    unembed_path = "lm_head"


class _HFGpt2PathSchema(_PathSchema):
    embedding_path = "transformer.wte"
    blocks = "transformer.h"
    attn = _FusedAttnPathSchema(base="attn", qkv="c_attn", o="c_proj")
    mlp = _FFNPathSchema(base="mlp", up="c_fc", down="c_proj")
    unembed_path = "lm_head"


# ── Schema lookup ──────────────────────────────────────────────────────


def _get_path_schema(model: nn.Module) -> _PathSchema:
    from transformers.models.gpt2 import GPT2LMHeadModel

    from spd.pretrain.models import GPT2, GPT2Simple, LlamaSimple, LlamaSimpleMLP

    match model:
        case LlamaSimple():
            return _LlamaSimplePathSchema()
        case LlamaSimpleMLP():
            return _LlamaSimpleMLPPathSchema()
        case GPT2Simple():
            return _GPT2SimplePathSchema()
        case GPT2():
            return _GPT2PathSchema()
        case GPT2LMHeadModel():
            return _HFGpt2PathSchema()
        case _:
            raise ValueError(
                f"Unsupported model class _{type(model).__name__}. Add a _PathSchema in topology.py."
            )


# ── TransformerTopology ────────────────────────────────────────────────


class TransformerTopology:
    """Bidirectional mapping between canonical weights and concrete module paths.

    Built from a target model (nn.Module). Independent of decomposition.
    """

    def __init__(self, target_model: nn.Module) -> None:
        self.target_model = target_model
        self.path_schema = _get_path_schema(target_model)

    def canon_to_target(self, canonical: str) -> str:
        return self.path_schema.render_canonical_weight(CanonicalWeight.parse(canonical))

    def target_to_canon(self, target_module_path: str) -> str:
        return self.path_schema.parse_target_path(target_module_path).canonical_str()

    def _get_module(self, canonical: CanonicalWeight) -> nn.Module:
        target_path = self.path_schema.render_canonical_weight(canonical)
        return self.target_model.get_submodule(target_path)

    @property
    def embedding_module(self) -> nn.Embedding:
        mod = self._get_module(Embed())
        assert isinstance(mod, nn.Embedding)
        return mod

    @property
    def unembed_module(self) -> nn.Linear:
        mod = self._get_module(Unembed())
        assert isinstance(mod, nn.Linear)
        return mod

    @property
    def n_blocks(self) -> int:
        blocks = self.target_model.get_submodule(self.path_schema.blocks)
        assert isinstance(blocks, nn.ModuleList)
        return len(blocks)

    def get_unembed_weight(self):
        """Unembedding weight matrix transposed to [d_model, vocab]."""
        return self.unembed_module.weight.T.detach()

    def is_cross_seq_pair(self, source_canonical: str, target_canonical: str) -> bool:
        """True if source is k/v and target is o in the same block."""
        source = CanonicalWeight.parse(source_canonical)
        target = CanonicalWeight.parse(target_canonical)
        match source, target:
            case (
                LayerWeight(layer_idx=si, name=SeparateAttnWeight(weight="k" | "v")),
                LayerWeight(layer_idx=ti, name=SeparateAttnWeight(weight="o")),
            ):
                return si == ti
            case (
                LayerWeight(layer_idx=si, name=FusedAttnWeight(weight="qkv")),
                LayerWeight(layer_idx=ti, name=FusedAttnWeight(weight="o")),
            ):
                return si == ti
            case _:
                return False
