"""
To run this app, run the following command:

```bash
    streamlit run spd/experiments/lm/app.py -- --model_path "wandb:spd-gf-lm/runs/151bsctx"
```
"""

import argparse
import html
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import streamlit as st
import torch
from datasets import load_dataset
from jaxtyping import Float, Int
from torch import Tensor
from transformers import AutoTokenizer

from spd.configs import Config, LMTaskConfig
from spd.data import DatasetConfig
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.spd_types import ModelPath

DEFAULT_MODEL_PATH: ModelPath = "wandb:spd-gf-lm/runs/151bsctx"


# -----------------------------------------------------------
# Dataclass holding everything the app needs
# -----------------------------------------------------------
@dataclass(frozen=True)
class AppData:
    model: ComponentModel
    tokenizer: AutoTokenizer
    config: Config
    dataloader_iter_fn: Callable[[], Iterator[dict[str, Any]]]
    target_layer_names: list[str]
    device: str


# --- Initialization and Data Loading ---
@st.cache_resource(show_spinner="Loading model and data...")
def initialize(model_path: ModelPath) -> AppData:
    """
    Loads the model, tokenizer, config, and evaluation dataloader.
    Cached by Streamlit based on the model_path.
    """
    device = "cpu"  # Use CPU for the Streamlit app
    logger.info(f"Initializing app with model: {model_path} on device: {device}")
    ss_model, config, _ = ComponentModel.from_pretrained(model_path)
    ss_model.to(device)
    ss_model.eval()

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig), "Task config must be LMTaskConfig for this app."

    # Derive tokenizer path (adjust if stored differently)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    # Create eval dataloader config
    eval_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.pretrained_model_name_hf,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=False,
        streaming=False,
        column_name=task_config.column_name,
    )

    # Create the dataloader iterator
    def create_dataloader_iter() -> Iterator[dict[str, Any]]:
        """
        Returns a *new* iterator each time it is called.
        Each element is a dict with:
            - "text": the raw document text
            - "input_ids": Int[Tensor, "1 seq_len"]
            - "offset_mapping": list[tuple[int, int]]
        """
        logger.info("Creating new dataloader iterator.")

        # Stream the HF dataset split
        dataset = load_dataset(
            eval_data_config.name,
            streaming=eval_data_config.streaming,
            split=eval_data_config.split,
            trust_remote_code=False,
        )

        dataset = dataset.with_format("torch")

        text_column = eval_data_config.column_name

        def tokenize_and_prepare(example: dict[str, Any]) -> dict[str, Any]:
            original_text: str = example[text_column]

            tokenized = tokenizer(
                original_text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=task_config.max_seq_len,
                padding=False,
            )

            input_ids: Int[Tensor, "1 seq_len"] = tokenized["input_ids"]
            if input_ids.dim() == 1:  # Ensure 2‑D [1, seq_len]
                input_ids = input_ids.unsqueeze(0)

            # HF returns offset_mapping as a list per sequence; batch size is 1
            offset_mapping: list[tuple[int, int]] = tokenized["offset_mapping"][0].tolist()

            return {
                "text": original_text,
                "input_ids": input_ids,
                "offset_mapping": offset_mapping,
            }

        # Map over the streaming dataset and return an iterator
        return map(tokenize_and_prepare, iter(dataset))  # pyright: ignore[reportArgumentType]

    target_layer_names = sorted(ss_model.target_module_paths)

    logger.info(f"Initialization complete for {model_path}.")
    return AppData(
        model=ss_model,
        tokenizer=tokenizer,
        config=config,
        dataloader_iter_fn=create_dataloader_iter,
        target_layer_names=target_layer_names,
        device=device,
    )


# -----------------------------------------------------------
# Utility: render the prompt with faint token outlines
# -----------------------------------------------------------
def render_prompt_with_tokens(
    *,
    raw_text: str,
    offset_mapping: list[tuple[int, int]],
    selected_idx: int | None,
) -> None:
    """
    Renders `raw_text` inside Streamlit, wrapping each token span with a thin
    border.  The currently‑selected token receives a thicker red border.
    All other tokens get a thin mid‑grey border (no background fill).
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
            border_style = (
                "2px solid rgb(200,0,0)" if is_selected else "0.5px solid #aaa"  # all other tokens
            )
            html_chunks.append(
                "<span "
                f'style="border:{border_style};'
                'border-radius:2px; padding:1px 2px; margin:0 1px;">'
                f"{token_substr}</span>"
            )
        cursor = end

    if cursor < len(raw_text):
        html_chunks.append(esc(raw_text[cursor:]))

    st.markdown(
        f'<div style="line-height:1.7; font-family:monospace;">{"".join(html_chunks)}</div>',
        unsafe_allow_html=True,
    )


def load_next_prompt() -> None:
    """Loads the next prompt, calculates masks, and prepares token data."""
    logger.info("Loading next prompt.")
    app_data: AppData = st.session_state.app_data
    dataloader_iter = st.session_state.dataloader_iter  # Get current iterator

    try:
        batch = next(dataloader_iter)
        input_ids: Int[Tensor, "1 seq_len"] = batch["input_ids"].to(app_data.device)
    except StopIteration:
        logger.warning("Dataloader iterator exhausted. Throwing error.")
        st.error("Failed to get data even after resetting dataloader.")
        return

    st.session_state.current_input_ids = input_ids

    # Store the original raw prompt text
    st.session_state.current_prompt_text = batch["text"]

    # Calculate activations and masks
    with torch.no_grad():
        _, pre_weight_acts = app_data.model.forward_with_pre_forward_cache_hooks(
            input_ids, module_names=app_data.target_layer_names
        )
        masks, _ = app_data.model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            detach_inputs=True,  # No gradients needed
        )
    st.session_state.current_masks = masks  # Dict[str, Float[Tensor, "1 seq_len C"]]

    # Prepare token data for display
    token_data = []
    tokenizer = app_data.tokenizer
    for i, token_id in enumerate(input_ids[0]):
        # Decode individual token - might differ slightly from full decode for spaces etc.
        decoded_token_str = tokenizer.decode([token_id])  # pyright: ignore[reportAttributeAccessIssue]
        token_data.append(
            {
                "id": token_id.item(),
                "text": decoded_token_str,
                "index": i,
                "offset": batch["offset_mapping"][i],  # (start, end)
            }
        )
    st.session_state.token_data = token_data

    # Reset selections
    st.session_state.selected_token_index = 0  # default: first token
    st.session_state.selected_layer_name = None
    logger.info("Finished loading next prompt and calculating masks.")


# --- Main App UI ---
def run_app(args: argparse.Namespace) -> None:
    """Sets up and runs the Streamlit application."""
    st.set_page_config(layout="wide")
    st.title("LM Component Activation Explorer")

    # Initialize model, data, etc. (cached)
    st.session_state.app_data = initialize(args.model_path)
    app_data: AppData = st.session_state.app_data
    st.caption(f"Model: {args.model_path}")

    # Initialize session state variables if they don't exist
    if "current_prompt_text" not in st.session_state:
        st.session_state.current_prompt_text = None
    if "token_data" not in st.session_state:
        st.session_state.token_data = None
    if "current_masks" not in st.session_state:
        st.session_state.current_masks = None
    if "selected_token_index" not in st.session_state:
        st.session_state.selected_token_index = None
    if "selected_layer_name" not in st.session_state:
        if app_data.target_layer_names:
            st.session_state.selected_layer_name = app_data.target_layer_names[0]
        else:
            st.session_state.selected_layer_name = None
    # Initialize the dataloader iterator in session state
    if "dataloader_iter" not in st.session_state:
        st.session_state.dataloader_iter = app_data.dataloader_iter_fn()

    if st.session_state.current_prompt_text is None:
        load_next_prompt()

    # Sidebar container and a single expander for all interactive controls
    sidebar = st.sidebar
    controls_expander = sidebar.expander("Controls", expanded=True)

    # ------------------------------------------------------------------
    # Sidebar – interactive controls
    # ------------------------------------------------------------------
    with controls_expander:
        st.button("Load Next Prompt", on_click=load_next_prompt)

    # Render the raw prompt with faint token borders
    if st.session_state.token_data and st.session_state.current_prompt_text:
        # st.subheader("Prompt")
        render_prompt_with_tokens(
            raw_text=st.session_state.current_prompt_text,
            offset_mapping=[t["offset"] for t in st.session_state.token_data],
            selected_idx=st.session_state.selected_token_index,
        )

        # Sidebar slider for token selection
        n_tokens = len(st.session_state.token_data)
        if n_tokens > 0:
            with controls_expander:
                st.header("Token selector")
                idx = st.slider(
                    "Token index",
                    min_value=0,
                    max_value=n_tokens - 1,
                    step=1,
                    key="selected_token_index",
                )

                selected_token = st.session_state.token_data[idx]
                st.write(f"Selected token: {selected_token['text']} (ID: {selected_token['id']})")

    st.divider()

    # --- Token Information Area ---
    if st.session_state.token_data:
        idx = st.session_state.selected_token_index
        # Ensure token_data is loaded before accessing
        if (
            st.session_state.token_data
            and idx is not None
            and idx < len(st.session_state.token_data)
        ):
            # Layer Selection Dropdown
            # Always default to the first layer if nothing is selected yet
            if st.session_state.selected_layer_name is None and app_data.target_layer_names:
                st.session_state.selected_layer_name = app_data.target_layer_names[0]

            with controls_expander:
                st.header("Layer selector")
                st.selectbox(
                    "Select Layer to Inspect:",
                    options=app_data.target_layer_names,
                    key="selected_layer_name",
                )

            # Display Layer-Specific Info if a layer is selected
            if st.session_state.selected_layer_name:
                layer_name = st.session_state.selected_layer_name
                logger.debug(f"Displaying info for token {idx}, layer {layer_name}")

                if st.session_state.current_masks is None:
                    st.warning("Masks not calculated yet. Please load a prompt.")
                    return

                layer_mask_tensor: Float[Tensor, "1 seq_len C"] = st.session_state.current_masks[
                    layer_name
                ]
                token_mask: Float[Tensor, " C"] = layer_mask_tensor[0, idx, :]

                # Find active components (mask > 0)
                active_indices_layer: Int[Tensor, " n_active"] = torch.where(token_mask > 0)[0]
                n_active_layer = len(active_indices_layer)

                st.metric(f"Active Components in {layer_name}", n_active_layer)

                st.subheader("Active Component Indices")
                if n_active_layer > 0:
                    # Convert to NumPy array and reshape to a column vector (N x 1)
                    active_indices_np = active_indices_layer.cpu().numpy().reshape(-1, 1)
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
            # Handle case where selected_token_index might be invalid after data reload
            st.warning("Selected token index is out of bounds. Please select a token again.")
            st.session_state.selected_token_index = None  # Reset selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Streamlit app to explore LM component activations."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path or W&B reference to the trained ComponentModel. Default: {DEFAULT_MODEL_PATH}",
    )
    args = parser.parse_args()

    run_app(args)
