"""Canonical transformer topology.

Defines a normalized addressing scheme for transformer layers. All API
consumers use canonical addresses instead of raw module paths.

Canonical layer address format:
    "embed"            — embedding
    "output"           — unembed / logits
    "{block}.attn.{p}" — attention projection (p: q | k | v | qkv | o)
    "{block}.ffn.{p}"  — FFN projection (p: up | down | gate)

Node key format (layer address + position):
    "{layer_address}:{seq_pos}:{component_idx}"

Examples:
    "0.attn.q"         — block 0, attention Q projection
    "2.ffn.gate"       — block 2, SwiGLU gate projection
    "0.attn.q:3:5"     — block 0, Q proj, seq pos 3, component 5
    "embed:0:0"        — embedding, seq 0, pseudo-component 0
    "output:7:42"       — output, seq 7, token 42
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, override

from torch import nn

# ── Canonical weight types ──────────────────────────────────────────────

_CANONICAL_RE = re.compile(
    r"^(?:"
    r"(?P<embed>embed)"
    r"|(?P<output>output)"
    r"|(?P<layer>\d+)\.(?P<sublayer>attn|attn_fused|glu|mlp)\.(?P<proj>[a-z]+)"
    r")$"
)


class CanonicalWeight(ABC):
    @abstractmethod
    def canonical_str(self) -> str: ...

    @staticmethod
    def parse(s: str) -> CanonicalWeight:
        """Parse a canonical address string into a CanonicalWeight."""
        m = _CANONICAL_RE.match(s)
        assert m is not None, f"Invalid canonical address: {s!r}"

        if m.group("embed"):
            return Embed()
        if m.group("output"):
            return Unembed()

        layer_idx = int(m.group("layer"))
        sublayer = m.group("sublayer")
        proj = m.group("proj")

        cls, valid = _SUBLAYER_PROJECTIONS[sublayer]
        assert proj in valid, f"Invalid projection {proj!r} for {sublayer!r} in {s!r}"
        return LayerWeight(layer_idx, cls(weight=proj))


@dataclass(frozen=True)
class Embed(CanonicalWeight):
    @override
    def canonical_str(self) -> str:
        return "embed"


@dataclass(frozen=True)
class Unembed(CanonicalWeight):
    @override
    def canonical_str(self) -> str:
        return "output"


@dataclass(frozen=True)
class SeparateAttnWeight:
    weight: Literal["q", "k", "v", "o"]


@dataclass(frozen=True)
class FusedAttnWeight:
    weight: Literal["qkv", "o"]


AttnWeight = SeparateAttnWeight | FusedAttnWeight


@dataclass(frozen=True)
class GLUWeight:
    weight: Literal["up", "down", "gate"]


@dataclass(frozen=True)
class MLPWeight:
    weight: Literal["up", "down"]


FFNWeight = GLUWeight | MLPWeight


@dataclass(frozen=True)
class LayerWeight(CanonicalWeight):
    layer_idx: int
    name: AttnWeight | FFNWeight

    @override
    def canonical_str(self) -> str:
        match self.name:
            case SeparateAttnWeight(weight=p):
                return f"{self.layer_idx}.attn.{p}"
            case FusedAttnWeight(weight=p):
                return f"{self.layer_idx}.attn_fused.{p}"
            case GLUWeight(weight=p):
                return f"{self.layer_idx}.glu.{p}"
            case MLPWeight(weight=p):
                return f"{self.layer_idx}.mlp.{p}"


_SUBLAYER_PROJECTIONS: dict[str, tuple[type, tuple[str, ...]]] = {
    "attn": (SeparateAttnWeight, ("q", "k", "v", "o")),
    "attn_fused": (FusedAttnWeight, ("qkv", "o")),
    "glu": (GLUWeight, ("up", "down", "gate")),
    "mlp": (MLPWeight, ("up", "down")),
}


# ── Sublayer path schemas ──────────────────────────────────────────────


@dataclass
class SeparateAttnPathSchema:
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
class FusedAttnPathSchema:
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
class GLUPathSchema:
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
class FFNPathSchema:
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


class PathSchema(ABC):
    embedding_path: str
    blocks: str
    attn: SeparateAttnPathSchema | FusedAttnPathSchema
    mlp: GLUPathSchema | FFNPathSchema
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
                assert isinstance(self.attn, SeparateAttnPathSchema)
                return f"{base}.{self.attn.render(attn_w)}"
            case FusedAttnWeight() as attn_w:
                assert isinstance(self.attn, FusedAttnPathSchema)
                return f"{base}.{self.attn.render(attn_w)}"
            case GLUWeight() as ffn_w:
                assert isinstance(self.mlp, GLUPathSchema)
                return f"{base}.{self.mlp.render(ffn_w)}"
            case MLPWeight() as ffn_w:
                assert isinstance(self.mlp, FFNPathSchema)
                return f"{base}.{self.mlp.render(ffn_w)}"


class LlamaSimplePathSchema(PathSchema):
    embedding_path = "wte"
    blocks = "h"
    attn = SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    mlp = GLUPathSchema(base="mlp", gate="gate_proj", up="up_proj", down="down_proj")
    unembed_path = "lm_head"


class LlamaSimpleMLPPathSchema(PathSchema):
    embedding_path = "wte"
    blocks = "h"
    attn = SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    mlp = FFNPathSchema(base="mlp", up="c_fc", down="down_proj")
    unembed_path = "lm_head"


class GPT2SimplePathSchema(PathSchema):
    embedding_path = "wte"
    blocks = "h"
    attn = SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    mlp = FFNPathSchema(base="mlp", up="c_fc", down="down_proj")
    unembed_path = "lm_head"


class GPT2PathSchema(PathSchema):
    embedding_path = "wte"
    blocks = "h_torch"
    attn = FusedAttnPathSchema(base="attn", qkv="c_attn", o="c_proj")
    mlp = FFNPathSchema(base="mlp", up="c_fc", down="c_proj")
    unembed_path = "lm_head"


class HFGpt2PathSchema(PathSchema):
    embedding_path = "transformer.wte"
    blocks = "transformer.h"
    attn = FusedAttnPathSchema(base="attn", qkv="c_attn", o="c_proj")
    mlp = FFNPathSchema(base="mlp", up="c_fc", down="c_proj")
    unembed_path = "lm_head"


# ── Schema lookup ──────────────────────────────────────────────────────


def _get_path_schema(model: nn.Module) -> PathSchema:
    from transformers.models.gpt2 import GPT2LMHeadModel

    from spd.pretrain.models import GPT2, GPT2Simple, LlamaSimple, LlamaSimpleMLP

    match model:
        case LlamaSimple():
            return LlamaSimplePathSchema()
        case LlamaSimpleMLP():
            return LlamaSimpleMLPPathSchema()
        case GPT2Simple():
            return GPT2SimplePathSchema()
        case GPT2():
            return GPT2PathSchema()
        case GPT2LMHeadModel():
            return HFGpt2PathSchema()
        case _:
            raise ValueError(
                f"Unsupported model class {type(model).__name__}. Add a PathSchema in topology.py."
            )


# ── TransformerTopology ────────────────────────────────────────────────


class TransformerTopology:
    """Bidirectional mapping between canonical weights and concrete module paths.

    Built from a target model (nn.Module). Independent of decomposition.
    """

    def __init__(self, target_model: nn.Module) -> None:
        self.target_model = target_model
        self.path_schema = _get_path_schema(target_model)

    def get_target_module_path(self, canonical: CanonicalWeight) -> str:
        return self.path_schema.render_canonical_weight(canonical)

    def get_canonical_weight(self, target_module_path: str) -> CanonicalWeight:
        return self.path_schema.parse_target_path(target_module_path)

    def get_module(self, canonical: CanonicalWeight) -> nn.Module:
        return self.target_model.get_submodule(self.get_target_module_path(canonical))

    @property
    def embedding_module(self) -> nn.Embedding:
        mod = self.get_module(Embed())
        assert isinstance(mod, nn.Embedding)
        return mod

    @property
    def unembed_module(self) -> nn.Linear:
        mod = self.get_module(Unembed())
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

    # NOTE: is_cross_seq_pair logic needs review — stubbed for now
    def is_cross_seq_pair(self, source: CanonicalWeight, target: CanonicalWeight) -> bool:
        """True if source is k/v and target is o in the same block."""
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
