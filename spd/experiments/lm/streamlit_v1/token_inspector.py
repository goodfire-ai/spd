"""
Token Component Inspector tab for the Streamlit app.
"""

import html
from collections.abc import Iterator
from dataclasses import dataclass

import streamlit as st
import torch
from datasets import load_dataset
from jaxtyping import Float, Int
from torch import Tensor

from spd.data import DatasetConfig
from spd.experiments.lm.configs import LMTaskConfig
from spd.experiments.lm.streamlit_v1.utils import ModelData


@dataclass
class TokenActivationAnalysis:
    """Results from analyzing component activations for a specific token."""

    n_active: int
    active_indices: list[int]
    active_values: list[float]
    total_components: int


@dataclass(frozen=True)
class PromptData:
    """Data for the current prompt."""

    text: str
    input_ids: Int[Tensor, "1 seq_len"]
    offset_mapping: list[tuple[int, int]]
    tokens: list[str]


# ============================================================================
# Core logic
# ============================================================================


def create_dataloader_iterator(model_data: ModelData) -> Iterator[PromptData]:
    """Yield one PromptData per raw dataset example, limited to max_seq_len."""

    task_cfg = model_data.config.task_config
    assert isinstance(task_cfg, LMTaskConfig)

    eval_cfg = DatasetConfig(
        name=task_cfg.dataset_name,
        hf_tokenizer_path=model_data.config.pretrained_model_name_hf,
        split=task_cfg.eval_data_split,
        n_ctx=task_cfg.max_seq_len,
        is_tokenized=task_cfg.is_tokenized,
        streaming=task_cfg.streaming,
        column_name=task_cfg.column_name,
    )

    dataset = load_dataset(
        eval_cfg.name,
        streaming=eval_cfg.streaming,
        split=eval_cfg.split,
        trust_remote_code=False,
    )

    for example in dataset:
        text = str(example[eval_cfg.column_name]) if isinstance(example, dict) else str(example)

        tokenised = model_data.tokenizer(  # pyright: ignore[reportCallIssue]
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=task_cfg.max_seq_len,
            padding=False,
            add_special_tokens=False,
        )

        input_ids: Int[Tensor, "1 seq_len"] = tokenised["input_ids"]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        offset_mapping: list[tuple[int, int]] = tokenised["offset_mapping"][0].tolist()

        tokens = [model_data.tokenizer.decode([int(tok)]) for tok in input_ids[0]]  # pyright: ignore[reportAttributeAccessIssue]

        yield PromptData(
            text=text,
            input_ids=input_ids,
            offset_mapping=offset_mapping,
            tokens=tokens,
        )


@st.cache_data(show_spinner="Computing causal importances...")
def compute_causal_importances(
    _model_data: ModelData,
    _input_ids: Tensor,
) -> dict[str, Float[Tensor, "1 seq_len C"]]:
    """Compute causal importances for all layers."""
    with torch.no_grad():
        _, pre_weight_acts = _model_data.model(
            _input_ids,
            mode="pre_forward_cache",
            module_names=list(_model_data.components.keys()),
        )
        cis, _ = _model_data.model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            detach_inputs=True,
            sigmoid_type=_model_data.config.sigmoid_type,
        )
    return cis


@st.cache_data(show_spinner="Analyzing token activations...")
def analyze_token_activations(
    token_idx: int,
    layer_name: str,
    ci_threshold: float,
    _cis: dict[str, Tensor],
) -> TokenActivationAnalysis:
    """Analyze component activations for a specific token and layer.

    Note: Parameters prefixed with _ are Streamlit conventions indicating the parameter
    should not trigger cache invalidation when it changes.
    """
    layer_cis = _cis[layer_name]

    # Ensure token_idx is within bounds of the mask tensor
    if token_idx >= layer_cis.shape[1]:
        return TokenActivationAnalysis(
            n_active=0,
            active_indices=[],
            active_values=[],
            total_components=0,
        )

    token_ci = layer_cis[0, token_idx, :]

    # Find active components
    active_indices = torch.where(token_ci > ci_threshold)[0]
    active_values = token_ci[active_indices]

    # Sort by activation strength
    sorted_indices = torch.argsort(active_values, descending=True)
    active_indices = active_indices[sorted_indices]
    active_values = active_values[sorted_indices]

    return TokenActivationAnalysis(
        n_active=len(active_indices),
        active_indices=active_indices.cpu().numpy().tolist(),
        active_values=active_values.cpu().numpy().tolist(),
        total_components=token_ci.shape[0],
    )


def load_next_prompt(model_data: ModelData) -> None:
    """Load the next prompt from the dataloader."""
    if "dataloader_iter" not in st.session_state:
        st.session_state.dataloader_iter = create_dataloader_iterator(model_data)

    try:
        prompt_data = next(st.session_state.dataloader_iter)
        st.session_state.current_prompt_data = prompt_data
        # Reset token selection
        st.session_state.selected_token_idx = 0
    except StopIteration:
        # Reset iterator and try again
        st.session_state.dataloader_iter = create_dataloader_iterator(model_data)
        prompt_data = next(st.session_state.dataloader_iter)
        st.session_state.current_prompt_data = prompt_data
        st.session_state.selected_token_idx = 0


# ============================================================================
# UI Rendering Functions
# ============================================================================


def _render_prompt_with_tokens(
    *,
    raw_text: str,
    offset_mapping: list[tuple[int, int]],
    selected_idx: int | None,
) -> None:
    """
    Renders `raw_text` inside Streamlit with faint borders around each token.
    The selected token gets a subtle highlight that works in both light and dark modes.
    """
    html_chunks: list[str] = []
    cursor = 0

    def esc(s: str) -> str:
        return html.escape(s)

    for idx, (start, end) in enumerate(offset_mapping):
        if cursor < start:
            html_chunks.append(esc(raw_text[cursor:start]))

        token_substr = esc(raw_text[start:end])
        if token_substr:
            is_selected = idx == selected_idx

            # Faint border for all tokens, with subtle highlight for selected
            if is_selected:
                # Selected token: faint blue background and slightly darker border
                style = (
                    "background-color: rgba(100, 150, 255, 0.1); "
                    "padding: 2px 4px; "
                    "border-radius: 3px; "
                    "border: 1px solid rgba(128, 128, 128, 0.4); "
                    "box-shadow: 0 0 2px rgba(100, 150, 255, 0.3);"
                )
            else:
                # Regular tokens: just faint border
                style = (
                    "padding: 2px 4px; "
                    "border-radius: 3px; "
                    "border: 1px solid rgba(128, 128, 128, 0.2);"
                )

            html_chunks.append(
                f'<span style="{style}" title="Token index: {idx}">{token_substr}</span>'
            )
        cursor = end

    if cursor < len(raw_text):
        html_chunks.append(esc(raw_text[cursor:]))

    # Add CSS styles before rendering
    st.markdown(
        f'<div class="example-item" style="font-family: monospace; font-size: 14px; '
        f'line-height: 1.8; color: var(--text-color);">{"".join(html_chunks)}</div>',
        unsafe_allow_html=True,
    )


def _render_selector_controls(n_tokens: int, model_token_count: int) -> tuple[int, float]:
    """Render token and CI-threshold selection controls."""

    token_idx = st.session_state.get("selected_token_idx", 0)
    ci_threshold = st.session_state.get("ci_threshold", 0.01)

    # Group both sliders in a single expander so they share one section
    with st.expander("Token & Analysis parameters", expanded=True):
        if n_tokens > 0:
            # Token-index slider (top)
            token_idx = st.slider(
                "Token index",
                min_value=0,
                max_value=n_tokens - 1,
                step=1,
                key="selected_token_idx",
            )

            selected_token = st.session_state.current_prompt_data.tokens[token_idx]
            st.write(f"Selected token: {selected_token} (Index: {token_idx})")

            # Warn if token is beyond the model's processing range
            if token_idx >= model_token_count:
                st.warning("⚠️ This token is beyond the model's processing range")

        # CI-threshold slider (below token index)
        ci_threshold = st.slider(
            "Causal Importance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=ci_threshold,
            step=0.01,
            format="%.3f",
            key="ci_threshold",
            help="Minimum CI value for a component to be considered active",
        )

    return token_idx, ci_threshold


def _render_layer_selector(layer_names: list[str]) -> str | None:
    """Render layer selection controls and return selected layer."""
    with st.expander("Layer selector", expanded=True):
        layer_name = st.selectbox(
            "Select Layer to Inspect:",
            options=layer_names,
            key="selected_layer",
        )
    return layer_name


def _render_activation_analysis(analysis: TokenActivationAnalysis, layer_name: str) -> None:
    """Render the component activation analysis results."""
    # Use component section styling from component_activation_contexts.py
    st.markdown(
        f'<div class="component-section">'
        f'<div class="component-header">Active Components in {layer_name}</div>'
        f'<div class="examples-container">'
        f"Total active components: {analysis.n_active}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Active Component Indices")
    if analysis.n_active > 0:
        # Create DataFrame for better display
        import numpy as np

        active_indices_np = np.array(analysis.active_indices).reshape(-1, 1)
        st.dataframe(active_indices_np, height=300, use_container_width=False)
    else:
        st.write("No active components for this token in this layer.")


# ============================================================================
# Main UI Function
# ============================================================================


@st.fragment
def render_token_activations_tab(model_data: ModelData):
    """Render the token component inspection analysis."""
    # Initialize session state
    if "current_prompt_data" not in st.session_state:
        load_next_prompt(model_data)
    if "selected_token_idx" not in st.session_state:
        st.session_state.selected_token_idx = 0

    # Load next prompt button
    if st.button("Load Next Prompt", key="token_inspector_load_prompt"):
        load_next_prompt(model_data)
        st.rerun()

    prompt_data = st.session_state.current_prompt_data

    # Compute masks for current prompt
    cis = compute_causal_importances(
        model_data,
        prompt_data.input_ids.to(next(model_data.model.parameters()).device),
    )

    # Check if prompt was truncated for model
    task_config = model_data.config.task_config
    assert isinstance(task_config, LMTaskConfig)

    # Calculate actual model tokens (truncated)
    model_token_count = prompt_data.input_ids.shape[1]
    full_token_count = len(prompt_data.tokens)

    if model_token_count < full_token_count:
        st.warning(
            f"⚠️ This prompt contains {full_token_count} tokens, but only the first "
            f"{model_token_count} tokens are processed by the model (maximum sequence length: {task_config.max_seq_len}). "
            f"You can inspect all {full_token_count} tokens, but component activations are only available for the first {model_token_count}."
        )

    # Render the prompt with token highlighting
    _render_prompt_with_tokens(
        raw_text=prompt_data.text,
        offset_mapping=prompt_data.offset_mapping,
        selected_idx=st.session_state.get("selected_token_idx", 0),
    )

    # Token and CI-threshold selection controls
    n_tokens = len(prompt_data.tokens)
    token_idx, ci_threshold = _render_selector_controls(n_tokens, model_token_count)

    st.divider()

    # Only show analysis if token is within model's range and we have tokens
    if n_tokens > 0 and token_idx < model_token_count:
        layer_name = _render_layer_selector(model_data.layer_names)

        if layer_name:
            analysis = analyze_token_activations(
                token_idx=token_idx,
                layer_name=layer_name,
                ci_threshold=ci_threshold,
                _cis=cis,
            )

            _render_activation_analysis(analysis, layer_name)
    else:
        st.info(
            "Component activation analysis is not available for tokens beyond the model's maximum sequence length. "
            "Please select a token within the model's processing range to view activations."
        )
