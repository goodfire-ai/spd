"""Adapter that extracts model topology from a ComponentModel.

Each supported architecture has an explicit config that declares its embedding
path, unembed path, cross-sequence path patterns, role groupings, and display
names. No regex, string heuristics, or role-name sniffing.

Pseudo-layers:
- "wte": embedding layer (always a source, never a target)
- "output": logit output (always a target, never a source)
These are used as node keys in the attribution graph regardless of the
actual module names in the underlying model.
"""

from dataclasses import dataclass
from fnmatch import fnmatch

from jaxtyping import Float
from torch import Tensor, nn

from spd.models.component_model import ComponentModel


def _extract_block_index(path: str) -> int | None:
    """Extract the block index from a module path like 'h.3.attn.q_proj' or 'layers.12.mlp.up'.

    Returns None if no integer segment is found (e.g. 'lm_head', 'wte').
    """
    for segment in path.split("."):
        if segment.isdigit():
            return int(segment)
    return None


def _extract_role(path: str) -> str:
    """Extract the role (last segment) from a module path.

    'h.0.attn.q_proj' -> 'q_proj'
    'layers.0.self_attn.k_proj' -> 'k_proj'
    'lm_head' -> 'lm_head'
    """
    return path.rsplit(".", maxsplit=1)[-1]


@dataclass(frozen=True)
class ArchConfig:
    """Per-architecture topology declaration.

    Fields:
        embedding_path: Dotted path to the nn.Embedding (e.g. "wte", "transformer.wte")
        unembed_path: Dotted path to the lm_head Linear
        kv_patterns: Glob patterns for K/V projection paths (e.g. "h.*.attn.k_proj")
        o_patterns: Glob patterns for attention output projection paths (e.g. "h.*.attn.o_proj")
        role_groups: Named groups of roles for visual layout (e.g. {"qkv": ("q_proj", "k_proj", "v_proj")})
        display_names: Static display-name overrides (e.g. {"lm_head": "W_U"})
    """

    embedding_path: str
    unembed_path: str
    kv_patterns: tuple[str, ...]
    o_patterns: tuple[str, ...]
    role_groups: dict[str, tuple[str, ...]]
    display_names: dict[str, str]


# LlamaSimple: SwiGLU MLP (gate_proj, up_proj, down_proj), separate Q/K/V/O attention
_LLAMA_SIMPLE_CONFIG = ArchConfig(
    embedding_path="wte",
    unembed_path="lm_head",
    kv_patterns=("h.*.attn.k_proj", "h.*.attn.v_proj"),
    o_patterns=("h.*.attn.o_proj",),
    role_groups={
        "qkv": ("q_proj", "k_proj", "v_proj"),
        "swiglu": ("gate_proj", "up_proj"),
    },
    display_names={"lm_head": "W_U"},
)

# LlamaSimpleMLP: GELU MLP (c_fc, down_proj), separate Q/K/V/O attention
_LLAMA_SIMPLE_MLP_CONFIG = ArchConfig(
    embedding_path="wte",
    unembed_path="lm_head",
    kv_patterns=("h.*.attn.k_proj", "h.*.attn.v_proj"),
    o_patterns=("h.*.attn.o_proj",),
    role_groups={"qkv": ("q_proj", "k_proj", "v_proj")},
    display_names={"lm_head": "W_U"},
)

# GPT2Simple: GELU MLP (c_fc, down_proj), separate Q/K/V/O attention
_GPT2_SIMPLE_CONFIG = ArchConfig(
    embedding_path="wte",
    unembed_path="lm_head",
    kv_patterns=("h.*.attn.k_proj", "h.*.attn.v_proj"),
    o_patterns=("h.*.attn.o_proj",),
    role_groups={"qkv": ("q_proj", "k_proj", "v_proj")},
    display_names={"lm_head": "W_U"},
)

# GPT2 (custom): GELU MLP (c_fc, c_proj), fused c_attn QKV, c_proj attn output
# c_proj appears in both attn and mlp — fully-qualified patterns disambiguate
_GPT2_CONFIG = ArchConfig(
    embedding_path="wte",
    unembed_path="lm_head",
    kv_patterns=("h_torch.*.attn.c_attn",),
    o_patterns=("h_torch.*.attn.c_proj",),
    role_groups={},
    display_names={"lm_head": "W_U"},
)

# HuggingFace GPT2LMHeadModel (AutoModelForCausalLM resolving to GPT2)
_HF_GPT2_CONFIG = ArchConfig(
    embedding_path="transformer.wte",
    unembed_path="lm_head",
    kv_patterns=("transformer.h.*.attn.c_attn",),
    o_patterns=("transformer.h.*.attn.c_proj",),
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
                f"Add an ArchConfig for this model class in model_adapter.py."
            )


def _resolve_cross_seq_paths(
    target_module_paths: list[str],
    arch: ArchConfig,
) -> tuple[frozenset[str], frozenset[str]]:
    """Resolve full module paths for cross-sequence attention roles by matching against patterns."""
    kv = frozenset(
        p for p in target_module_paths if any(fnmatch(p, pat) for pat in arch.kv_patterns)
    )
    o = frozenset(p for p in target_module_paths if any(fnmatch(p, pat) for pat in arch.o_patterns))
    return kv, o


def _build_role_order(target_module_paths: list[str]) -> list[str]:
    """Deduplicated roles in execution order from target module paths."""
    seen: set[str] = set()
    order: list[str] = []
    for path in target_module_paths:
        role = _extract_role(path)
        if role not in seen:
            seen.add(role)
            order.append(role)
    return order


def _build_role_groups(
    role_order: list[str],
    arch: ArchConfig,
) -> dict[str, list[str]]:
    """Build role groups from the architecture config.

    Only includes groups where at least 2 roles are present in the target modules.
    """
    groups: dict[str, list[str]] = {}
    for group_name, roles in arch.role_groups.items():
        present = [r for r in roles if r in role_order]
        if len(present) >= 2:
            groups[group_name] = present
    return groups


@dataclass(frozen=True)
class ModelAdapter:
    """Model topology resolved at load time.

    Thread this through to all compute functions instead of hardcoding
    layer names, block counts, or module access patterns.
    """

    target_module_paths: list[str]
    embedding_path: str
    embedding_module: nn.Embedding
    unembed_path: str | None
    unembed_module: nn.Linear | None
    kv_paths: frozenset[str]
    o_paths: frozenset[str]
    role_order: list[str]  # Unique roles in execution order (e.g. ["q_proj", "k_proj", ...])
    role_groups: dict[str, list[str]]  # Grouped roles sharing a row (e.g. {"qkv": [...]})
    display_names: dict[str, str]  # Special layer display names (e.g. {"lm_head": "W_U"})

    def ordered_layers(self) -> list[str]:
        """Full layer list for gradient pair testing: [wte, ...components..., (unembed), output]."""
        layers = ["wte"]
        layers.extend(self.target_module_paths)
        layers.append("output")
        return layers

    def is_cross_seq_pair(self, source: str, target: str) -> bool:
        """Check if source -> target requires cross-sequence gradient computation.

        True when source is a k/v projection and target is an o projection
        in the same transformer block.
        """
        if source not in self.kv_paths or target not in self.o_paths:
            return False
        source_block = _extract_block_index(source)
        target_block = _extract_block_index(target)
        return source_block is not None and source_block == target_block

    def get_unembed_weight(self) -> Float[Tensor, "d_model vocab"]:
        """Get the unembedding weight matrix (transposed to [d_model, vocab])."""
        assert self.unembed_module is not None, "No unembed module found"
        return self.unembed_module.weight.T.detach()


def build_model_adapter(model: ComponentModel) -> ModelAdapter:
    """Build a ModelAdapter by matching the model's architecture to an explicit config."""
    target_model = model.target_model
    target_paths = model.target_module_paths

    arch = _get_arch_config(target_model)

    # Resolve embedding
    embedding_module = target_model.get_submodule(arch.embedding_path)
    assert isinstance(embedding_module, nn.Embedding), (
        f"Expected nn.Embedding at '{arch.embedding_path}', got {type(embedding_module).__name__}"
    )

    # Resolve unembed (only if it's among the decomposed targets)
    unembed_path: str | None = None
    unembed_module: nn.Linear | None = None
    if arch.unembed_path in target_paths:
        module = target_model.get_submodule(arch.unembed_path)
        assert isinstance(module, nn.Linear), (
            f"Expected nn.Linear at '{arch.unembed_path}', got {type(module).__name__}"
        )
        unembed_path = arch.unembed_path
        unembed_module = module

    kv_paths, o_paths = _resolve_cross_seq_paths(target_paths, arch)
    role_order = _build_role_order(target_paths)
    role_groups = _build_role_groups(role_order, arch)

    return ModelAdapter(
        target_module_paths=target_paths,
        embedding_path=arch.embedding_path,
        embedding_module=embedding_module,
        unembed_path=unembed_path,
        unembed_module=unembed_module,
        kv_paths=kv_paths,
        o_paths=o_paths,
        role_order=role_order,
        role_groups=role_groups,
        display_names=arch.display_names,
    )
