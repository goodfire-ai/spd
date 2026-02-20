"""
Model architecture configurations for RelP attribution.

This module provides architecture-specific parameters for different LLM models,
allowing the RelP pipeline to support multiple model families.
"""

from dataclasses import dataclass, field


@dataclass
class ModelArchConfig:
    """Architecture configuration for a specific model."""

    # Model identification
    name: str  # Short identifier (e.g., 'llama-3.1-8b')
    hf_model_id: str  # HuggingFace model path

    # Architecture parameters
    num_layers: int  # Number of transformer layers
    d_model: int  # Hidden dimension
    d_ffn: int  # FFN intermediate dimension (MLP width)
    num_attention_heads: int
    num_kv_heads: int | None = None  # For GQA models

    # RelP-specific parameters
    always_on_neurons: list[tuple[int, int]] = field(default_factory=list)
    # Format: [(layer, neuron_idx), ...] - neurons that activate on nearly all inputs

    # Chat template
    chat_template: str | None = None  # None = use tokenizer's built-in template

    # Module path patterns (for hook registration)
    mlp_module_pattern: str = "model.layers.{layer}.mlp"
    attn_module_pattern: str = "model.layers.{layer}.self_attn"


# =============================================================================
# Model Registry
# =============================================================================

# Llama 3.1 8B Instruct
LLAMA_3_1_8B = ModelArchConfig(
    name="llama-3.1-8b",
    hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
    num_layers=32,
    d_model=4096,
    d_ffn=14336,
    num_attention_heads=32,
    num_kv_heads=8,  # GQA
    # Always-on neurons from Marks et al. 2024 (specific to Llama 3.1-8B-Instruct)
    always_on_neurons=[
        (23, 306), (20, 3972), (18, 7417), (16, 1241),
        (13, 4208), (11, 11321), (10, 11570), (9, 4255),
        (7, 6673), (6, 5866), (5, 7012), (2, 4786)
    ],
    chat_template=None,  # Use tokenizer's built-in
)

# Qwen 3 32B (used in current 800K experiment)
QWEN_3_32B = ModelArchConfig(
    name="qwen-3-32b",
    hf_model_id="Qwen/Qwen3-32B",
    num_layers=64,
    d_model=5120,
    d_ffn=27648,
    num_attention_heads=40,
    num_kv_heads=8,  # GQA
    always_on_neurons=[],  # TODO: Compute for Qwen
    chat_template=None,
)

# OLMo 7B
OLMO_7B = ModelArchConfig(
    name="olmo-7b",
    hf_model_id="allenai/OLMo-7B",
    num_layers=32,
    d_model=4096,
    d_ffn=11008,  # Smaller than Llama (11008 vs 14336)
    num_attention_heads=32,
    num_kv_heads=None,  # MHA, not GQA
    always_on_neurons=[],  # TODO: Compute for OLMo
    chat_template=None,
)

# OLMo 2 7B (newer version)
OLMO_2_7B = ModelArchConfig(
    name="olmo-2-7b",
    hf_model_id="allenai/OLMo-2-1124-7B",
    num_layers=32,
    d_model=4096,
    d_ffn=11008,
    num_attention_heads=32,
    num_kv_heads=None,
    always_on_neurons=[],
    chat_template=None,
)

# OLMo 2 7B Instruct
OLMO_2_7B_INSTRUCT = ModelArchConfig(
    name="olmo-2-7b-instruct",
    hf_model_id="allenai/OLMo-2-1124-7B-Instruct",
    num_layers=32,
    d_model=4096,
    d_ffn=11008,
    num_attention_heads=32,
    num_kv_heads=None,
    always_on_neurons=[],
    chat_template=None,
)

# OLMo 3 7B Instruct (latest version)
# Uses hybrid attention: sliding window (4096) + full attention every 4th layer
OLMO_3_7B_INSTRUCT = ModelArchConfig(
    name="olmo-3-7b-instruct",
    hf_model_id="allenai/OLMo-3-7B-Instruct",
    num_layers=32,
    d_model=4096,
    d_ffn=11008,
    num_attention_heads=32,
    num_kv_heads=32,  # Full MHA (not GQA)
    always_on_neurons=[],  # TODO: Compute for OLMo 3
    chat_template=None,
)


# =============================================================================
# Registry and Lookup
# =============================================================================

MODEL_REGISTRY: dict[str, ModelArchConfig] = {
    # By short name
    "llama-3.1-8b": LLAMA_3_1_8B,
    "llama-3.1-8b-instruct": LLAMA_3_1_8B,
    "qwen-3-32b": QWEN_3_32B,
    "qwen3-32b": QWEN_3_32B,
    "olmo-7b": OLMO_7B,
    "olmo-2-7b": OLMO_2_7B,
    "olmo-2-7b-instruct": OLMO_2_7B_INSTRUCT,
    "olmo-3-7b-instruct": OLMO_3_7B_INSTRUCT,

    # By HuggingFace model ID
    "meta-llama/Llama-3.1-8B-Instruct": LLAMA_3_1_8B,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": LLAMA_3_1_8B,
    "Qwen/Qwen3-32B": QWEN_3_32B,
    "allenai/OLMo-7B": OLMO_7B,
    "allenai/OLMo-2-1124-7B": OLMO_2_7B,
    "allenai/OLMo-2-1124-7B-Instruct": OLMO_2_7B_INSTRUCT,
    "allenai/OLMo-3-7B-Instruct": OLMO_3_7B_INSTRUCT,
    "allenai/Olmo-3-7B-Instruct": OLMO_3_7B_INSTRUCT,  # Case variant
}


def get_model_config(model_name_or_path: str) -> ModelArchConfig | None:
    """
    Look up model configuration by name or HuggingFace path.

    Args:
        model_name_or_path: Short name (e.g., 'llama-3.1-8b') or
                           HF path (e.g., 'meta-llama/Llama-3.1-8B-Instruct')

    Returns:
        ModelArchConfig if found, None otherwise
    """
    # Direct lookup
    if model_name_or_path in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name_or_path]

    # Case-insensitive lookup
    lower_name = model_name_or_path.lower()
    for key, config in MODEL_REGISTRY.items():
        if key.lower() == lower_name:
            return config

    # Partial match on HF model ID
    for key, config in MODEL_REGISTRY.items():
        if "/" in key and model_name_or_path.lower() in key.lower():
            return config

    return None


def detect_model_family(model_name_or_path: str) -> str:
    """
    Detect the model family from a model name or path.

    Returns one of: 'llama', 'qwen', 'olmo', 'unknown'
    """
    lower = model_name_or_path.lower()

    if "llama" in lower:
        return "llama"
    elif "qwen" in lower:
        return "qwen"
    elif "olmo" in lower:
        return "olmo"
    else:
        return "unknown"


def get_chat_template(model_name_or_path: str) -> str | None:
    """
    Get the chat template for a model, if explicitly configured.

    For most models, returns None (use tokenizer's built-in template).
    """
    config = get_model_config(model_name_or_path)
    if config:
        return config.chat_template
    return None
