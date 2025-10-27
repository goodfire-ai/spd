"""Generate feature names for decision tree visualization with activation and decoding info."""

import torch
from jaxtyping import Float
from torch import Tensor

from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponents, LinearComponents


def get_embed_unembed_matrices(
    model: ComponentModel,
) -> tuple[Float[Tensor, "vocab d_model"], Float[Tensor, "d_model vocab"]]:
    """Extract embedding and unembedding matrices from the target model.

    For GPT-2 style models, returns (wte.weight, lm_head.weight or wte.weight.T)
    For LLaMA style models, returns (embed_tokens.weight, lm_head.weight)

    Returns:
        embed: Embedding matrix [vocab_size, d_model]
        unembed: Unembedding matrix [d_model, vocab_size]
    """
    target_model = model.target_model

    # Try to find embedding layer (GPT-2 style)
    if hasattr(target_model, "transformer") and hasattr(target_model.transformer, "wte"):
        embed = target_model.transformer.wte.weight  # [vocab, d_model]
    # Try LLaMA style
    elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
        embed = target_model.model.embed_tokens.weight  # [vocab, d_model]
    else:
        raise ValueError(
            "Could not find embedding layer. Expected transformer.wte or model.embed_tokens"
        )

    # Try to find unembedding layer
    if hasattr(target_model, "lm_head"):
        unembed = target_model.lm_head.weight.T  # [d_model, vocab]
    elif hasattr(target_model, "transformer") and hasattr(target_model.transformer, "wte"):
        # For tied embeddings, unembed is transpose of embed
        unembed = target_model.transformer.wte.weight.T  # [d_model, vocab]
    else:
        raise ValueError("Could not find unembedding layer (lm_head)")

    return embed, unembed


def decode_direction_top_k(
    direction: Float[Tensor, " d_model"],
    embed: Float[Tensor, "vocab d_model"],
    unembed: Float[Tensor, "d_model vocab"],
    tokenizer,
    k: int = 3,
    use_embed: bool = True,
) -> str:
    """Decode a direction vector to top-k tokens.

    Args:
        direction: Direction vector in d_model space
        embed: Embedding matrix [vocab, d_model]
        unembed: Unembedding matrix [d_model, vocab]
        tokenizer: Tokenizer for converting token IDs to strings
        k: Number of top tokens to return
        use_embed: If True, use embed matrix; if False, use unembed matrix

    Returns:
        String representation of top-k tokens
    """
    if use_embed:
        # Project direction onto embedding space: compute cosine similarity
        # direction: [d_model], embed: [vocab, d_model]
        direction_norm = direction / (direction.norm() + 1e-8)
        embed_norm = embed / (embed.norm(dim=1, keepdim=True) + 1e-8)
        similarities = torch.matmul(embed_norm, direction_norm)  # [vocab]
    else:
        # Project direction onto unembedding space
        # direction: [d_model], unembed: [d_model, vocab]
        logits = torch.matmul(direction, unembed)  # [vocab]
        similarities = logits

    # Get top-k tokens
    top_k_values, top_k_indices = torch.topk(similarities, k)

    # Decode tokens
    tokens = []
    for idx, val in zip(top_k_indices.tolist(), top_k_values.tolist(), strict=False):
        token_str = tokenizer.decode([idx])
        # Clean up token string for display
        token_str = repr(token_str)[1:-1]  # Remove quotes and escape special chars
        tokens.append(f"{token_str}({val:.2f})")

    return ",".join(tokens)


def get_component_directions(
    component_model: ComponentModel,
    module_key: str,
    component_idx: int,
) -> tuple[Float[Tensor, " d_in"], Float[Tensor, " d_out"]]:
    """Get read (V) and write (U) direction vectors for a component.

    Args:
        component_model: The ComponentModel containing components
        module_key: Key identifying the module (e.g., "transformer.h.0.attn.c_attn")
        component_idx: Index of the component

    Returns:
        read_direction: V[:, component_idx] - the read direction [d_in]
        write_direction: U[component_idx, :] - the write direction [d_out]
    """
    # Get the component module
    component = component_model.components[module_key]

    assert isinstance(component, LinearComponents | EmbeddingComponents), (
        f"Expected LinearComponents or EmbeddingComponents, got {type(component)}"
    )

    # Extract V and U
    V = component.V  # [d_in, C] or [vocab, C] for embedding
    U = component.U  # [C, d_out]

    read_direction = V[:, component_idx]  # [d_in]
    write_direction = U[component_idx, :]  # [d_out]

    return read_direction, write_direction
