"""
Token Component Inspector tab for the Streamlit app.
"""

import html
from typing import Any

import streamlit as st
import torch
from torch import Tensor

from spd.configs import LMTaskConfig
from spd.experiments.lm.streamlit_v1.component_activation_contexts import (
    _get_streamlit_css,
)
from spd.experiments.lm.streamlit_v1.utils import (
    ModelData,
    compute_component_masks,
    create_dataloader_iterator,
)


@st.cache_data(show_spinner="Analyzing token activations...")
def analyze_token_activations(
    _model_path: str,
    _prompt_text: str,
    token_idx: int,
    layer_name: str,
    _masks: dict[str, Tensor],
) -> dict[str, Any]:
    """Analyze component activations for a specific token and layer."""
    layer_mask = _masks[layer_name]

    # Ensure token_idx is within bounds of the mask tensor
    if token_idx >= layer_mask.shape[1]:
        return {
            "n_active": 0,
            "active_indices": [],
            "active_values": [],
            "total_components": 0,
        }

    token_mask = layer_mask[0, token_idx, :]

    # Find active components
    active_indices = torch.where(token_mask > 0)[0]
    active_values = token_mask[active_indices]

    # Sort by activation strength
    sorted_indices = torch.argsort(active_values, descending=True)
    active_indices = active_indices[sorted_indices]
    active_values = active_values[sorted_indices]

    return {
        "n_active": len(active_indices),
        "active_indices": active_indices.cpu().numpy(),
        "active_values": active_values.cpu().numpy(),
        "total_components": token_mask.shape[0],
    }


def render_prompt_with_tokens(
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
        f"<style>{_get_streamlit_css()}</style>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="example-item" style="font-family: monospace; font-size: 14px; '
        f'line-height: 1.8; color: var(--text-color);">{"".join(html_chunks)}</div>',
        unsafe_allow_html=True,
    )


def load_next_prompt(model_data: ModelData):
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
    masks = compute_component_masks(
        st.session_state.model_path,
        prompt_data.text,
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
    render_prompt_with_tokens(
        raw_text=prompt_data.text,
        offset_mapping=prompt_data.offset_mapping,
        selected_idx=st.session_state.get("selected_token_idx", 0),
    )

    # Token selection controls
    n_tokens = len(prompt_data.tokens)
    token_idx = st.session_state.get("selected_token_idx", 0)

    with st.expander("Token selector", expanded=True):
        if n_tokens > 0:
            token_idx = st.slider(
                "Token index",
                min_value=0,
                max_value=n_tokens - 1,
                step=1,
                key="selected_token_idx",
            )

            selected_token = prompt_data.tokens[token_idx]
            st.write(f"Selected token: {selected_token} (Index: {token_idx})")

            # Show if token is beyond model's processing range
            if token_idx >= model_token_count:
                st.warning("⚠️ This token is beyond the model's processing range")

    st.divider()

    # Only show analysis if token is within model's range and we have tokens
    if n_tokens > 0 and token_idx < model_token_count:
        # Layer selection
        with st.expander("Layer selector", expanded=True):
            layer_name = st.selectbox(
                "Select Layer to Inspect:",
                options=model_data.layer_names,
                key="selected_layer",
            )

        # Analyze activations
        if layer_name and token_idx is not None:
            analysis = analyze_token_activations(
                st.session_state.model_path,
                prompt_data.text,
                token_idx,
                layer_name,
                masks,
            )

            # Use component section styling from component_activation_contexts.py
            st.markdown(
                f'<div class="component-section">'
                f'<div class="component-header">Active Components in {layer_name}</div>'
                f'<div class="examples-container">'
                f"Total active components: {analysis['n_active']}"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            st.subheader("Active Component Indices")
            if analysis["n_active"] > 0:
                # Convert to NumPy array and reshape to a column vector (N x 1)
                active_indices_np = analysis["active_indices"].reshape(-1, 1)
                # Pass the NumPy array directly and configure the column header
                st.dataframe(active_indices_np, height=300, use_container_width=False)
            else:
                st.write("No active components for this token in this layer.")

            # Extensibility Placeholder
            st.subheader("Additional Layer/Token Analysis")
            st.write(
                "Future figures and analyses for this specific layer and token will appear here."
            )
    else:
        st.info(
            "Component activation analysis is not available for tokens beyond the model's maximum sequence length. "
            "Please select a token within the model's processing range to view activations."
        )
