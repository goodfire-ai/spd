"""Abstract transformer topology.

Maps concrete model paths onto a canonical abstract structure. Built once at
model load time, consumed by harvest, attributions, autointerp, and the app.

Usage:
    topology = TransformerTopology(model)
    topology.n_blocks               # 12
    topology.describe("h.1.mlp.down_proj")  # "MLP down-projection in layer 2 of 12"
    topology.is_cross_seq_pair(src, tgt)    # True if k/v -> o in same block
    topology.embedding.module       # nn.Embedding
    topology.unembed.module         # nn.Linear
"""

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Literal

from jaxtyping import Float
from torch import Tensor, nn

from spd.models.component_model import ComponentModel

# Construction-only role type — used to sort paths into struct fields during init.
# Not exposed on LayerInfo or any public API.
_Role = Literal["q", "k", "v", "qkv", "o", "up", "down", "gate"]
_ATTN_ROLES: set[_Role] = {"q", "k", "v", "qkv", "o"}


@dataclass(frozen=True)
class LayerInfo:
    path: str
    module: nn.Module


# --- Attention variants ---


@dataclass(frozen=True)
class SeparateAttention:
    q: LayerInfo
    k: LayerInfo
    v: LayerInfo
    o: LayerInfo


@dataclass(frozen=True)
class FusedAttention:
    qkv: LayerInfo
    o: LayerInfo


AttentionInfo = SeparateAttention | FusedAttention


# --- FFN variants ---


@dataclass(frozen=True)
class StandardFFN:
    up: LayerInfo
    down: LayerInfo


@dataclass(frozen=True)
class SwiGLUFFN:
    gate: LayerInfo
    up: LayerInfo
    down: LayerInfo


FFNInfo = StandardFFN | SwiGLUFFN


@dataclass(frozen=True)
class BlockInfo:
    index: int
    attn: AttentionInfo | None
    ffn: FFNInfo


# --- Architecture configs ---


@dataclass(frozen=True)
class ArchConfig:
    """Per-architecture topology declaration.

    role_mapping maps glob patterns to roles used during construction to sort
    paths into the correct struct fields. These roles are not exposed publicly.
    """

    embedding_path: str
    unembed_path: str
    role_mapping: dict[str, _Role]
    role_groups: dict[str, tuple[str, ...]]
    display_names: dict[str, str]


_LLAMA_SIMPLE_CONFIG = ArchConfig(
    embedding_path="wte",
    unembed_path="lm_head",
    role_mapping={
        "h.*.attn.q_proj": "q",
        "h.*.attn.k_proj": "k",
        "h.*.attn.v_proj": "v",
        "h.*.attn.o_proj": "o",
        "h.*.mlp.gate_proj": "gate",
        "h.*.mlp.up_proj": "up",
        "h.*.mlp.down_proj": "down",
    },
    role_groups={
        "qkv": ("q_proj", "k_proj", "v_proj"),
        "swiglu": ("gate_proj", "up_proj"),
    },
    display_names={"lm_head": "W_U"},
)

_LLAMA_SIMPLE_MLP_CONFIG = ArchConfig(
    embedding_path="wte",
    unembed_path="lm_head",
    role_mapping={
        "h.*.attn.q_proj": "q",
        "h.*.attn.k_proj": "k",
        "h.*.attn.v_proj": "v",
        "h.*.attn.o_proj": "o",
        "h.*.mlp.c_fc": "up",
        "h.*.mlp.down_proj": "down",
    },
    role_groups={"qkv": ("q_proj", "k_proj", "v_proj")},
    display_names={"lm_head": "W_U"},
)

_GPT2_SIMPLE_CONFIG = ArchConfig(
    embedding_path="wte",
    unembed_path="lm_head",
    role_mapping={
        "h.*.attn.q_proj": "q",
        "h.*.attn.k_proj": "k",
        "h.*.attn.v_proj": "v",
        "h.*.attn.o_proj": "o",
        "h.*.mlp.c_fc": "up",
        "h.*.mlp.down_proj": "down",
    },
    role_groups={"qkv": ("q_proj", "k_proj", "v_proj")},
    display_names={"lm_head": "W_U"},
)

_GPT2_CONFIG = ArchConfig(
    embedding_path="wte",
    unembed_path="lm_head",
    role_mapping={
        "h_torch.*.attn.c_attn": "qkv",
        "h_torch.*.attn.c_proj": "o",
        "h_torch.*.mlp.c_fc": "up",
        "h_torch.*.mlp.c_proj": "down",
    },
    role_groups={},
    display_names={"lm_head": "W_U"},
)

_HF_GPT2_CONFIG = ArchConfig(
    embedding_path="transformer.wte",
    unembed_path="lm_head",
    role_mapping={
        "transformer.h.*.attn.c_attn": "qkv",
        "transformer.h.*.attn.c_proj": "o",
        "transformer.h.*.mlp.c_fc": "up",
        "transformer.h.*.mlp.c_proj": "down",
    },
    role_groups={},
    display_names={"lm_head": "W_U"},
)


def _get_arch_config(model: nn.Module) -> ArchConfig:
    """Get the architecture config by matching the model class."""
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
                f"Unsupported model architecture: {type(model).__name__}. "
                f"Add an ArchConfig for this model class in topology.py."
            )


def _extract_block_index(path: str) -> int:
    """Extract the single integer segment from a dotted path.

    Asserts exactly one integer segment exists (e.g. "h.3.attn.q_proj" -> 3).
    """
    digits = [s for s in path.split(".") if s.isdigit()]
    assert len(digits) == 1, f"Expected exactly 1 integer segment in '{path}', got {digits}"
    return int(digits[0])


def _resolve_role(path: str, role_mapping: dict[str, _Role]) -> _Role:
    """Match a concrete path against role_mapping patterns."""
    matches: list[_Role] = [
        role for pattern, role in role_mapping.items() if fnmatch(path, pattern)
    ]
    assert len(matches) == 1, f"Expected exactly 1 role match for {path}, got {matches}"
    return matches[0]


# --- Descriptions derived from struct position ---


def _describe_attn_layer(attn: AttentionInfo, layer: LayerInfo) -> str:
    match attn:
        case SeparateAttention(q, k, v, o):
            if layer is q:
                return "attention Q projection"
            if layer is k:
                return "attention K projection"
            if layer is v:
                return "attention V projection"
            assert layer is o
            return "attention output projection"
        case FusedAttention(qkv, o):
            if layer is qkv:
                return "attention QKV projection (fused)"
            assert layer is o
            return "attention output projection"


def _describe_ffn_layer(ffn: FFNInfo, layer: LayerInfo) -> str:
    match ffn:
        case StandardFFN(up, down):
            if layer is up:
                return "MLP up-projection"
            assert layer is down
            return "MLP down-projection"
        case SwiGLUFFN(gate, up, down):
            if layer is gate:
                return "SwiGLU gate projection"
            if layer is up:
                return "SwiGLU up-projection"
            assert layer is down
            return "SwiGLU down-projection"


def _get_kv_paths(blocks: list[BlockInfo]) -> frozenset[str]:
    paths: set[str] = set()
    for block in blocks:
        match block.attn:
            case SeparateAttention(_, k, v, _):
                paths.add(k.path)
                paths.add(v.path)
            case FusedAttention(qkv, _):
                paths.add(qkv.path)
            case None:
                pass
    return frozenset(paths)


def _get_o_paths(blocks: list[BlockInfo]) -> frozenset[str]:
    paths: set[str] = set()
    for block in blocks:
        match block.attn:
            case SeparateAttention(_, _, _, o):
                paths.add(o.path)
            case FusedAttention(_, o):
                paths.add(o.path)
            case None:
                pass
    return frozenset(paths)


class TransformerTopology:
    """Abstract transformer topology resolved at model load time."""

    embedding: LayerInfo
    blocks: list[BlockInfo]
    unembed: LayerInfo

    def __init__(self, model: ComponentModel) -> None:
        target_model = model.target_model
        target_paths = model.target_module_paths
        arch = _get_arch_config(target_model)

        # Resolve embedding
        embedding_module = target_model.get_submodule(arch.embedding_path)
        assert isinstance(embedding_module, nn.Embedding), (
            f"Expected nn.Embedding at '{arch.embedding_path}', "
            f"got {type(embedding_module).__name__}"
        )
        self.embedding = LayerInfo(arch.embedding_path, embedding_module)

        # Resolve unembed
        unembed_module = target_model.get_submodule(arch.unembed_path)
        assert isinstance(unembed_module, nn.Linear), (
            f"Expected nn.Linear at '{arch.unembed_path}', got {type(unembed_module).__name__}"
        )
        self.unembed = LayerInfo(arch.unembed_path, unembed_module)

        # Group target paths by block index, assign roles, build LayerInfos
        block_layers: dict[int, dict[_Role, LayerInfo]] = {}
        self._path_to_block_idx: dict[str, int] = {}

        for path in target_paths:
            block_idx = _extract_block_index(path)
            role = _resolve_role(path, arch.role_mapping)
            layer_info = LayerInfo(path, target_model.get_submodule(path))
            block_layers.setdefault(block_idx, {})[role] = layer_info
            self._path_to_block_idx[path] = block_idx

        # Build blocks in order
        self.blocks = []
        for idx in sorted(block_layers):
            roles = block_layers[idx]
            attn = _build_attention(roles)
            ffn = _build_ffn(roles)
            self.blocks.append(BlockInfo(idx, attn, ffn))

        # Preserve original target path ordering from ComponentModel
        self._model_target_paths = target_paths

        # Derive kv/o path sets from block structs
        self.kv_paths = _get_kv_paths(self.blocks)
        self.o_paths = _get_o_paths(self.blocks)

        # Build path -> (block, layer_info) lookup for describe()
        self._path_to_layer: dict[str, tuple[BlockInfo, LayerInfo]] = {}
        for block in self.blocks:
            for layer in _block_layers(block):
                self._path_to_layer[layer.path] = (block, layer)

        # Role order: deduplicated concrete role names in execution order (for frontend layout)
        seen: set[str] = set()
        role_order: list[str] = []
        for path in target_paths:
            role_name = path.rsplit(".", maxsplit=1)[-1]
            if role_name not in seen:
                seen.add(role_name)
                role_order.append(role_name)
        self.role_order = role_order

        # Role groups: only include groups where >= 2 roles are present
        self.role_groups: dict[str, list[str]] = {}
        for group_name, roles in arch.role_groups.items():
            present = [r for r in roles if r in seen]
            if len(present) >= 2:
                self.role_groups[group_name] = present

        self._arch = arch

    @property
    def target_module_paths(self) -> list[str]:
        return self._model_target_paths

    @property
    def embedding_path(self) -> str:
        return self.embedding.path

    @property
    def embedding_module(self) -> nn.Embedding:
        assert isinstance(self.embedding.module, nn.Embedding)
        return self.embedding.module

    @property
    def unembed_path(self) -> str:
        return self.unembed.path

    @property
    def unembed_module(self) -> nn.Linear:
        assert isinstance(self.unembed.module, nn.Linear)
        return self.unembed.module

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def block_index(self, path: str) -> int:
        return self._path_to_block_idx[path]

    def describe(self, path: str) -> str:
        """Human-readable description for autointerp prompts."""
        block, layer = self._path_to_layer[path]
        if block.attn is not None and _layer_in_attn(block.attn, layer):
            desc = _describe_attn_layer(block.attn, layer)
        else:
            desc = _describe_ffn_layer(block.ffn, layer)
        return f"{desc} in layer {block.index + 1} of {self.n_blocks}"

    def is_cross_seq_pair(self, source: str, target: str) -> bool:
        """True if source is k/v and target is o in the same block."""
        if source not in self._path_to_block_idx or target not in self._path_to_block_idx:
            return False
        if source not in self.kv_paths or target not in self.o_paths:
            return False
        return self._path_to_block_idx[source] == self._path_to_block_idx[target]

    def ordered_layers(self) -> list[str]:
        """Full layer list for gradient pair testing: [wte, ...components..., output]."""
        layers = ["wte"]
        layers.extend(self._model_target_paths)
        layers.append("output")
        return layers

    def get_unembed_weight(self) -> Float[Tensor, "d_model vocab"]:
        """Get the unembedding weight matrix (transposed to [d_model, vocab])."""
        assert isinstance(self.unembed.module, nn.Linear)
        return self.unembed.module.weight.T.detach()

    @property
    def display_names(self) -> dict[str, str]:
        return self._arch.display_names


def _layer_in_attn(attn: AttentionInfo, layer: LayerInfo) -> bool:
    match attn:
        case SeparateAttention(q, k, v, o):
            return layer is q or layer is k or layer is v or layer is o
        case FusedAttention(qkv, o):
            return layer is qkv or layer is o


def _block_layers(block: BlockInfo) -> list[LayerInfo]:
    """All LayerInfos in a block, for building lookups."""
    layers: list[LayerInfo] = []
    if block.attn is not None:
        match block.attn:
            case SeparateAttention(q, k, v, o):
                layers.extend([q, k, v, o])
            case FusedAttention(qkv, o):
                layers.extend([qkv, o])
    match block.ffn:
        case StandardFFN(up, down):
            layers.extend([up, down])
        case SwiGLUFFN(gate, up, down):
            layers.extend([gate, up, down])
    return layers


def _build_attention(roles: dict[_Role, LayerInfo]) -> AttentionInfo | None:
    """Build attention info from collected roles. Returns None if no attention roles present."""
    attn_roles = {r: info for r, info in roles.items() if r in _ATTN_ROLES}
    if not attn_roles:
        return None

    if "qkv" in attn_roles:
        assert set(attn_roles) == {"qkv", "o"}, (
            f"Fused attention expects qkv + o, got {set(attn_roles)}"
        )
        return FusedAttention(qkv=attn_roles["qkv"], o=attn_roles["o"])

    assert set(attn_roles) >= {"q", "k", "v", "o"}, (
        f"Separate attention expects q, k, v, o; got {set(attn_roles)}"
    )
    return SeparateAttention(
        q=attn_roles["q"], k=attn_roles["k"], v=attn_roles["v"], o=attn_roles["o"]
    )


def _build_ffn(roles: dict[_Role, LayerInfo]) -> FFNInfo:
    """Build FFN info from collected roles."""
    ffn_roles = {r: info for r, info in roles.items() if r not in _ATTN_ROLES}
    assert "up" in ffn_roles and "down" in ffn_roles, (
        f"FFN requires up + down, got {set(ffn_roles)}"
    )

    if "gate" in ffn_roles:
        return SwiGLUFFN(gate=ffn_roles["gate"], up=ffn_roles["up"], down=ffn_roles["down"])
    return StandardFFN(up=ffn_roles["up"], down=ffn_roles["down"])
