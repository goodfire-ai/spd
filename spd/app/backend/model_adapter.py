"""Adapter that extracts model topology from a ComponentModel.

Replaces hardcoded layer names, block counting, and module access patterns
with introspection-based discovery. All model-architecture assumptions
(embedding path, unembedding path, attention detection, block numbering)
are resolved once at load time and stored here.

Pseudo-layers:
- "wte": embedding layer (always a source, never a target)
- "output": logit output (always a target, never a source)
These are used as node keys in the attribution graph regardless of the
actual module names in the underlying model.
"""

import re
from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor, nn

from spd.models.component_model import ComponentModel


def _find_embedding_module(model: nn.Module) -> tuple[str, nn.Embedding]:
    """Find the first nn.Embedding in the model."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            return name, module
    raise ValueError("No nn.Embedding found in model")


def _find_unembed_module(
    model: nn.Module,
    target_module_paths: list[str],
    vocab_size: int,
) -> tuple[str, nn.Linear] | None:
    """Find the unembedding (lm_head) module if it's a decomposed target."""
    for path in target_module_paths:
        module = model.get_submodule(path)
        if isinstance(module, nn.Linear) and module.out_features == vocab_size:
            return path, module
    return None


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


_KV_ROLE_PATTERN = re.compile(r"k_proj|v_proj|key|value|c_attn")
_O_ROLE_PATTERN = re.compile(r"o_proj|out_proj|c_proj")
_QKV_ROLE_PATTERN = re.compile(r"q_proj|k_proj|v_proj|query|key|value")


def _detect_cross_seq_roles(
    target_module_paths: list[str],
) -> tuple[set[str], set[str]]:
    """Auto-detect (kv_paths, o_paths) for attention cross-sequence flow.

    Returns full module paths (not just roles) so same-block matching works.
    """
    kv_paths: set[str] = set()
    o_paths: set[str] = set()

    for path in target_module_paths:
        role = _extract_role(path)
        if _KV_ROLE_PATTERN.search(role):
            kv_paths.add(path)
        elif _O_ROLE_PATTERN.search(role):
            o_paths.add(path)

    return kv_paths, o_paths


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


def _detect_role_groups(role_order: list[str]) -> dict[str, list[str]]:
    """Auto-detect role groups (roles that share a visual row).

    Currently detects QKV-like groups: consecutive roles matching q/k/v patterns.
    """
    qkv_roles = [r for r in role_order if _QKV_ROLE_PATTERN.search(r)]
    if len(qkv_roles) >= 2:
        return {"qkv": qkv_roles}
    return {}


def _build_display_names(
    unembed_path: str | None,
) -> dict[str, str]:
    """Build display name mapping for special layers."""
    names: dict[str, str] = {}
    if unembed_path is not None:
        names[_extract_role(unembed_path)] = "W_U"
    return names


def build_model_adapter(model: ComponentModel) -> ModelAdapter:
    """Build a ModelAdapter by introspecting the ComponentModel."""
    target_model = model.target_model

    embedding_path, embedding_module = _find_embedding_module(target_model)
    vocab_size = embedding_module.num_embeddings

    unembed_result = _find_unembed_module(target_model, model.target_module_paths, vocab_size)
    unembed_path = unembed_result[0] if unembed_result else None
    unembed_module = unembed_result[1] if unembed_result else None

    kv_paths, o_paths = _detect_cross_seq_roles(model.target_module_paths)
    role_order = _build_role_order(model.target_module_paths)
    role_groups = _detect_role_groups(role_order)
    display_names = _build_display_names(unembed_path)

    return ModelAdapter(
        target_module_paths=model.target_module_paths,
        embedding_path=embedding_path,
        embedding_module=embedding_module,
        unembed_path=unembed_path,
        unembed_module=unembed_module,
        kv_paths=frozenset(kv_paths),
        o_paths=frozenset(o_paths),
        role_order=role_order,
        role_groups=role_groups,
        display_names=display_names,
    )
