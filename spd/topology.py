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

from torch import nn

from spd.models.component_model import ComponentModel

AttentionRole = Literal["q", "k", "v", "qkv", "o"]
FFNRole = Literal["up", "down", "gate"]
AbstractRole = AttentionRole | FFNRole

_ATTN_ROLES: set[AttentionRole] = {"q", "k", "v", "qkv", "o"}
_KV_ROLES: set[AttentionRole] = {"k", "v", "qkv"}
_O_ROLES: set[AttentionRole] = {"o"}


@dataclass(frozen=True)
class LayerInfo:
    path: str
    module: nn.Module
    role: AbstractRole


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

    role_mapping maps glob patterns to abstract roles. Every target module path
    must match exactly one pattern.
    """

    embedding_path: str
    unembed_path: str
    role_mapping: dict[str, AbstractRole]
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


def _extract_block_index(path: str) -> int | None:
    """Extract the first integer segment from a dotted path."""
    for segment in path.split("."):
        if segment.isdigit():
            return int(segment)
    return None


def _resolve_role(path: str, role_mapping: dict[str, AbstractRole]) -> AbstractRole:
    """Match a concrete path against role_mapping patterns."""
    matches: list[AbstractRole] = [
        role for pattern, role in role_mapping.items() if fnmatch(path, pattern)
    ]
    assert len(matches) == 1, f"Expected exactly 1 role match for {path}, got {matches}"
    return matches[0]


# --- Role descriptions for autointerp prompts ---

_ROLE_DESCRIPTIONS: dict[AbstractRole, str] = {
    "q": "attention Q projection",
    "k": "attention K projection",
    "v": "attention V projection",
    "qkv": "attention QKV projection (fused)",
    "o": "attention output projection",
    "up": "MLP up-projection",
    "down": "MLP down-projection",
    "gate": "MLP gate projection",
}


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
        self.embedding = LayerInfo(arch.embedding_path, embedding_module, "up")

        # Resolve unembed
        unembed_module = target_model.get_submodule(arch.unembed_path)
        assert isinstance(unembed_module, nn.Linear), (
            f"Expected nn.Linear at '{arch.unembed_path}', got {type(unembed_module).__name__}"
        )
        self.unembed = LayerInfo(arch.unembed_path, unembed_module, "down")

        # Group target paths by block index and assign roles
        block_layers: dict[int, dict[AbstractRole, LayerInfo]] = {}
        self._path_to_role: dict[str, AbstractRole] = {}
        self._path_to_block_idx: dict[str, int] = {}

        for path in target_paths:
            block_idx = _extract_block_index(path)
            assert block_idx is not None, f"No block index in path: {path}"
            role = _resolve_role(path, arch.role_mapping)

            module = target_model.get_submodule(path)
            layer_info = LayerInfo(path, module, role)

            block_layers.setdefault(block_idx, {})[role] = layer_info
            self._path_to_role[path] = role
            self._path_to_block_idx[path] = block_idx

        # Build blocks in order
        self.blocks = []
        for idx in sorted(block_layers):
            roles = block_layers[idx]
            attn = _build_attention(roles)
            ffn = _build_ffn(roles)
            self.blocks.append(BlockInfo(idx, attn, ffn))

        self._arch = arch

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def block_index(self, path: str) -> int:
        return self._path_to_block_idx[path]

    def role(self, path: str) -> AbstractRole:
        return self._path_to_role[path]

    def describe(self, path: str) -> str:
        role = self.role(path)
        block_idx = self.block_index(path)
        desc = _ROLE_DESCRIPTIONS[role]
        return f"{desc} in layer {block_idx + 1} of {self.n_blocks}"

    def is_cross_seq_pair(self, source: str, target: str) -> bool:
        """True if source is k/v and target is o in the same block."""
        src_role = self._path_to_role.get(source)
        tgt_role = self._path_to_role.get(target)
        if src_role is None or tgt_role is None:
            return False
        if src_role not in _KV_ROLES or tgt_role not in _O_ROLES:
            return False
        return self._path_to_block_idx[source] == self._path_to_block_idx[target]

    def ordered_paths(self) -> list[str]:
        """All paths in execution order: ["wte", ...components..., "output"]."""
        paths: list[str] = ["wte"]
        for block in self.blocks:
            if block.attn is not None:
                match block.attn:
                    case SeparateAttention(q, k, v, o):
                        paths.extend([q.path, k.path, v.path, o.path])
                    case FusedAttention(qkv, o):
                        paths.extend([qkv.path, o.path])
            match block.ffn:
                case StandardFFN(up, down):
                    paths.extend([up.path, down.path])
                case SwiGLUFFN(gate, up, down):
                    paths.extend([gate.path, up.path, down.path])
        paths.append("output")
        return paths

    @property
    def role_groups(self) -> dict[str, tuple[str, ...]]:
        return self._arch.role_groups

    @property
    def display_names(self) -> dict[str, str]:
        return self._arch.display_names


def _build_attention(roles: dict[AbstractRole, LayerInfo]) -> AttentionInfo | None:
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


def _build_ffn(roles: dict[AbstractRole, LayerInfo]) -> FFNInfo:
    """Build FFN info from collected roles."""
    ffn_roles = {r: info for r, info in roles.items() if r not in _ATTN_ROLES}
    assert "up" in ffn_roles and "down" in ffn_roles, (
        f"FFN requires up + down, got {set(ffn_roles)}"
    )

    if "gate" in ffn_roles:
        return SwiGLUFFN(gate=ffn_roles["gate"], up=ffn_roles["up"], down=ffn_roles["down"])
    return StandardFFN(up=ffn_roles["up"], down=ffn_roles["down"])
