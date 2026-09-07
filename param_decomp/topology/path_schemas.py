"""Path schemas: bidirectional mapping between concrete module paths and canonical weights.

Each model family gets a PathSchema subclass that declares its concrete naming conventions,
selected by model-type name. The schemas are private; `path_schema_for_model_type` is the
public entry (torch-free — no live model, just the config's model-type string).
"""

import re
from abc import ABC
from dataclasses import dataclass
from typing import Literal

from param_decomp.topology.canonical import (
    CanonicalWeight,
    Embed,
    FFNWeight,
    FusedAttnWeight,
    GLUWeight,
    LayerWeight,
    MLPWeight,
    MoEExpertsWeight,
    MoESharedWeight,
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
    """One gate/up/down GLU projection group; `weight_type` names the canonical kind it
    parses to, so one block may carry several GLU groups (an MoE's fused-expert and
    shared-expert stacks) without their canonical addresses colliding."""

    base: str
    gate: str
    up: str
    down: str
    weight_type: type[GLUWeight | MoEExpertsWeight | MoESharedWeight]

    def _lookup(self) -> dict[str, Literal["up", "down", "gate"]]:
        return {self.gate: "gate", self.up: "up", self.down: "down"}

    def _reverse(self) -> dict[str, str]:
        return {"gate": self.gate, "up": self.up, "down": self.down}

    def handles(self, w: FFNWeight) -> bool:
        return isinstance(w, self.weight_type)

    def parse(self, projection_name: str, layer_idx: int) -> LayerWeight:
        table = self._lookup()
        assert projection_name in table, f"Unknown GLU projection: {projection_name}"
        return LayerWeight(layer_idx, self.weight_type(table[projection_name]))

    def render(self, w: GLUWeight | MoEExpertsWeight | MoESharedWeight) -> str:
        assert self.handles(w), (self.weight_type, w)
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

    def handles(self, w: FFNWeight) -> bool:
        return isinstance(w, MLPWeight)

    def parse(self, projection_name: str, layer_idx: int) -> LayerWeight:
        table = self._lookup()
        assert projection_name in table, f"Unknown MLP projection: {projection_name}"
        return LayerWeight(layer_idx, MLPWeight(table[projection_name]))

    def render(self, w: MLPWeight) -> str:
        return f"{self.base}.{self._reverse()[w.weight]}"


class _PathSchema(ABC):
    embedding_path: str
    blocks: str
    attn: _SeparateAttnPathSchema | _FusedAttnPathSchema
    ffns: tuple[_GLUPathSchema | _FFNPathSchema, ...]
    unembed_path: str
    _block_re: re.Pattern[str] | None = None

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
        if self._block_re is None:
            blocks = re.escape(self.blocks)
            groups = "|".join(
                rf"(?P<g{i}>{re.escape(group.base)})\.(?P<p{i}>\w+)"
                for i, group in enumerate((self.attn, *self.ffns))
            )
            self._block_re = re.compile(rf"^{blocks}\.(?P<idx>\d+)\.(?:{groups})$")

        m = self._block_re.match(path)
        assert m is not None, f"Invalid block path: {path!r}"

        layer_idx = int(m.group("idx"))
        for i, group in enumerate((self.attn, *self.ffns)):
            if m.group(f"g{i}"):
                return group.parse(m.group(f"p{i}"), layer_idx)
        raise AssertionError(f"Invalid block path: {path!r}")

    def _ffn_schema_for(self, w: FFNWeight) -> _GLUPathSchema | _FFNPathSchema:
        matches = [schema for schema in self.ffns if schema.handles(w)]
        assert len(matches) == 1, f"Expected exactly one FFN schema for {w!r}, got {matches}"
        return matches[0]

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
            case MLPWeight() as ffn_w:
                schema = self._ffn_schema_for(ffn_w)
                assert isinstance(schema, _FFNPathSchema)
                return f"{base}.{schema.render(ffn_w)}"
            case GLUWeight() | MoEExpertsWeight() | MoESharedWeight() as ffn_w:
                schema = self._ffn_schema_for(ffn_w)
                assert isinstance(schema, _GLUPathSchema)
                return f"{base}.{schema.render(ffn_w)}"


class _LlamaSimplePathSchema(_PathSchema):
    embedding_path = "wte"
    blocks = "h"
    attn = _SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    ffns = (
        _GLUPathSchema(
            base="mlp", gate="gate_proj", up="up_proj", down="down_proj", weight_type=GLUWeight
        ),
    )
    unembed_path = "lm_head"


class _LlamaSimpleMLPPathSchema(_PathSchema):
    embedding_path = "wte"
    blocks = "h"
    attn = _SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    ffns = (_FFNPathSchema(base="mlp", up="c_fc", down="down_proj"),)
    unembed_path = "lm_head"


class _GPT2SimplePathSchema(_PathSchema):
    embedding_path = "wte"
    blocks = "h"
    attn = _SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    ffns = (_FFNPathSchema(base="mlp", up="c_fc", down="down_proj"),)
    unembed_path = "lm_head"


class _GPT2PathSchema(_PathSchema):
    embedding_path = "wte"
    blocks = "h_torch"
    attn = _FusedAttnPathSchema(base="attn", qkv="c_attn", o="c_proj")
    ffns = (_FFNPathSchema(base="mlp", up="c_fc", down="c_proj"),)
    unembed_path = "lm_head"


class _HFGLUPathSchema(_PathSchema):
    """The raw-HF GLU-transformer site grammar (`layers.{i}.self_attn.q_proj`, …) the
    Llama/Qwen3 `DecomposedModel` targets emit (`param_decomp.targets.glu_transformer`)."""

    embedding_path = "embed_tokens"
    blocks = "layers"
    attn = _SeparateAttnPathSchema(base="self_attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    ffns = (
        _GLUPathSchema(
            base="mlp", gate="gate_proj", up="up_proj", down="down_proj", weight_type=GLUWeight
        ),
    )
    unembed_path = "lm_head"


class _Qwen36MoePathSchema(_PathSchema):
    """The qwen36_moe site grammar (`param_decomp.targets.qwen36_moe`): each MoE block
    carries the fused all-expert GLU stack (`mlp.experts.*`) and the shared expert's
    (`mlp.shared_expert.*`) as separate canonical kinds (`{i}.moe.*` / `{i}.moe_shared.*`)."""

    embedding_path = "embed_tokens"
    blocks = "layers"
    attn = _SeparateAttnPathSchema(base="self_attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    ffns = (
        _GLUPathSchema(
            base="mlp.experts",
            gate="gate_proj",
            up="up_proj",
            down="down_proj",
            weight_type=MoEExpertsWeight,
        ),
        _GLUPathSchema(
            base="mlp.shared_expert",
            gate="gate_proj",
            up="up_proj",
            down="down_proj",
            weight_type=MoESharedWeight,
        ),
    )
    unembed_path = "lm_head"


_MODEL_TYPE_PATH_SCHEMAS: dict[str, type[_PathSchema]] = {
    "LlamaSimple": _LlamaSimplePathSchema,
    "LlamaSimpleMLP": _LlamaSimpleMLPPathSchema,
    "GPT2Simple": _GPT2SimplePathSchema,
    "GPT2": _GPT2PathSchema,
    "Llama": _HFGLUPathSchema,
    "Qwen3": _HFGLUPathSchema,
    "Qwen3_5Moe": _Qwen36MoePathSchema,
}


def path_schema_for_model_type(model_type: str) -> _PathSchema:
    """Select a path schema by target-model class name — torch-free (no live model).

    Consumers reading a JAX run know the model type from config, never a live model."""
    schema_cls = _MODEL_TYPE_PATH_SCHEMAS.get(model_type)
    assert schema_cls is not None, (
        f"No path schema for model type {model_type!r}. Add one in path_schemas.py."
    )
    return schema_cls()
