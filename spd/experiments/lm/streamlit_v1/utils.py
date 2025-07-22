"""
Shared utilities for the Streamlit app.
"""

import html
import re
from dataclasses import dataclass
from typing import cast

import streamlit as st
import torch
from datasets import load_dataset
from jaxtyping import Float, Int
from torch import Tensor
from transformers import AutoTokenizer

from spd.configs import Config, LMTaskConfig
from spd.data import DatasetConfig
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, GateMLP, LinearComponent, VectorGateMLP
from spd.utils.component_utils import calc_causal_importances

# ============================================================================
# Data Classes
# ============================================================================


@dataclass(frozen=True)
class ModelData:
    """Core model data that gets cached."""

    model: ComponentModel
    tokenizer: AutoTokenizer
    config: Config
    gates: dict[str, GateMLP | VectorGateMLP]
    components: dict[str, LinearComponent | EmbeddingComponent]
    layer_names: list[str]


@dataclass(frozen=True)
class PromptData:
    """Data for the current prompt."""

    text: str
    input_ids: Int[Tensor, "1 seq_len"]
    offset_mapping: list[tuple[int, int]]
    tokens: list[str]


# ============================================================================
# Utility Functions
# ============================================================================


def parse_wandb_url(url_or_path: str) -> str:
    """
    Parse various WandB formats into standard wandb:project/runs/run_id format.

    Accepts:
    - Full URLs: https://wandb.ai/project-name/entity/runs/run_id
    - WandB paths: wandb:project/runs/run_id
    - Just run IDs: run_id (uses default project)
    """
    if url_or_path.startswith("wandb:"):
        return url_or_path

    # Parse full WandB URL
    wandb_url_pattern = r"https://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?]+)"
    match = re.match(wandb_url_pattern, url_or_path)
    if match:
        _, project, run_id = match.groups()
        return f"wandb:{project}/runs/{run_id}"

    # Just a run ID
    if re.match(r"^[a-z0-9]+$", url_or_path):
        return f"wandb:spd/runs/{url_or_path}"

    return url_or_path


def render_prompt_with_tokens(
    text: str,
    offset_mapping: list[tuple[int, int]],
    selected_idx: int | None,
    max_height: int = 400,
) -> None:
    """Render prompt text with token boundaries and selection."""
    html_chunks = []
    cursor = 0
    rendered_tokens = 0
    skipped_tokens = []

    for idx, (start, end) in enumerate(offset_mapping):
        # Skip invalid mappings
        if start < 0 or end < 0 or start > len(text) or end > len(text):
            skipped_tokens.append(idx)
            continue

        # Add text before token (if any gap exists)
        if cursor < start and start > cursor:
            gap_text = text[cursor:start]
            if gap_text:
                html_chunks.append(html.escape(gap_text))

        token_text = html.escape(text[start:end]) if start < end else f"[T{idx}]"

        is_selected = idx == selected_idx
        border_style = "2px solid rgb(200,0,0)" if is_selected else "0.5px solid #aaa"
        bg_color = "rgba(255,200,200,0.3)" if is_selected else "transparent"

        html_chunks.append(
            f'<span style="border:{border_style}; border-radius:2px; '
            f"padding:1px 2px; margin:0 1px; background-color:{bg_color}; "
            f'display:inline-block; white-space:pre-wrap;">{token_text}</span>'
        )
        rendered_tokens += 1
        cursor = max(cursor, end)

    # Add remaining text if any
    if cursor < len(text):
        remaining = text[cursor:]
        if remaining:
            html_chunks.append(html.escape(remaining))

    # Add debug info
    total_tokens = len(offset_mapping)
    debug_info = []
    if rendered_tokens < total_tokens:
        debug_info.append(f"Rendered {rendered_tokens}/{total_tokens} tokens")
    if skipped_tokens:
        debug_info.append(
            f"Skipped tokens: {skipped_tokens[:5]}{'...' if len(skipped_tokens) > 5 else ''}"
        )

    if debug_info:
        html_chunks.append(
            f'<br><span style="color: #888; font-size: 0.8em; font-style: italic;">'
            f"{' | '.join(debug_info)}</span>"
        )

    # Wrap in scrollable container
    st.markdown(
        f'<div style="line-height:2.2; font-family:monospace; max-height:{max_height}px; '
        f"overflow-y:auto; padding:15px; border:1px solid #ddd; border-radius:5px; "
        f'background-color:#fafafa; word-wrap:break-word;">'
        f"{''.join(html_chunks)}</div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# Model Loading and Initialization
# ============================================================================


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: str) -> ModelData:
    """Load model and prepare components."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, _ = ComponentModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    # Extract components and gates
    gates = {
        k.removeprefix("gates.").replace("-", "."): cast(GateMLP | VectorGateMLP, v)
        for k, v in model.gates.items()
    }
    components = {
        k.removeprefix("components.").replace("-", "."): cast(
            LinearComponent | EmbeddingComponent, v
        )
        for k, v in model.components.items()
    }

    return ModelData(
        model=model,
        tokenizer=tokenizer,
        config=config,
        gates=gates,
        components=components,
        layer_names=sorted(list(components.keys())),
    )


def create_dataloader_iterator(model_data: ModelData):
    """Create iterator for evaluation data."""
    task_config = model_data.config.task_config
    assert isinstance(task_config, LMTaskConfig)

    eval_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=model_data.config.pretrained_model_name_hf,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=False,
        streaming=False,
        column_name=task_config.column_name,
    )

    dataset = load_dataset(
        eval_config.name,
        streaming=eval_config.streaming,
        split=eval_config.split,
        trust_remote_code=False,
    )

    for example in dataset:
        text = str(example[eval_config.column_name]) if isinstance(example, dict) else str(example)

        # First, tokenize the full text without truncation to get all tokens for display
        full_tokenized = model_data.tokenizer(  # pyright: ignore[reportCallIssue]
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=False,  # Don't truncate for display
            padding=False,
        )

        # Then tokenize with truncation for model input
        model_tokenized = model_data.tokenizer(  # pyright: ignore[reportCallIssue]
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=task_config.max_seq_len,
            padding=False,
        )

        # Use truncated input_ids for model but full offset mapping for display
        input_ids = model_tokenized["input_ids"]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Use full tokenization for display purposes
        full_input_ids = full_tokenized["input_ids"]
        if full_input_ids.dim() == 1:
            full_input_ids = full_input_ids.unsqueeze(0)

        offset_mapping = full_tokenized["offset_mapping"][0].tolist()
        tokens = [model_data.tokenizer.decode([token_id]) for token_id in full_input_ids[0]]  # pyright: ignore[reportAttributeAccessIssue]

        yield PromptData(
            text=text,  # Keep full text
            input_ids=input_ids,  # Truncated for model
            offset_mapping=offset_mapping,  # Full mapping for display
            tokens=tokens,  # Full tokens for display
        )


# ============================================================================
# Analysis Functions
# ============================================================================


@st.cache_data(show_spinner="Computing component masks...")
def compute_component_masks(
    _model_path: str,
    _prompt_text: str,
    _model_data: ModelData,
    _input_ids: Tensor,
) -> dict[str, Float[Tensor, "1 seq_len C"]]:
    """Compute component activation masks for all layers."""
    with torch.no_grad():
        _, pre_weight_acts = _model_data.model.forward_with_pre_forward_cache_hooks(
            _input_ids, module_names=list(_model_data.components.keys())
        )
        Vs = {name: comp.V for name, comp in _model_data.components.items()}
        masks, _ = calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            Vs=Vs,
            gates=_model_data.gates,
            detach_inputs=True,
        )
    return masks


# ============================================================================
# UI Components
# ============================================================================


def render_model_selector(current_model_path: str | None) -> str | None:
    """Render model selection UI in sidebar. Returns new model path if changed."""
    st.sidebar.header("Model Selection")

    if current_model_path:
        st.sidebar.info(f"Current: {current_model_path}")

    model_input = st.sidebar.text_input(
        "Enter WandB URL or path:",
        value=current_model_path or "",
        help="Examples:\n"
        "- https://wandb.ai/goodfire/spd/runs/snq4ojcy\n"
        "- wandb:spd/runs/snq4ojcy\n"
        "- 151bsctx (just the run ID)",
        placeholder="Paste WandB URL here...",
    )

    if st.sidebar.button("Load Model", type="primary"):
        if model_input:
            new_path = parse_wandb_url(model_input.strip())
            if new_path != current_model_path:
                return new_path
        else:
            st.sidebar.error("Please enter a model path or URL.")

    return None
