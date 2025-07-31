"""
Streamlit app for exploring LM component activations.

This is a clean, production-grade modular app designed to be easily extended.
It features:
- Modular analysis functions split into separate files
- Tab-based UI for organizing different analyses
- Efficient caching of model and analysis results
- Clear structure with well-defined sections

To run:
    streamlit run spd/experiments/lm/streamlit_v1/app.py
    streamlit run spd/experiments/lm/streamlit_v1/app.py -- --model_path "wandb:goodfire/spd/runs/yer1jgd7"

The app supports loading models from the UI. You can paste:
- Full WandB URLs: https://wandb.ai/goodfire/spd/runs/yer1jgd7
- WandB paths: wandb:goodfire/spd/runs/yer1jgd7
- Just run IDs: 151bsctx

To add new analyses:
1. Create a new module in this directory
2. Add a tab in main() that imports and calls your render function
3. Use @st.cache_data for analysis functions and @st.fragment for UI components
"""

import argparse

import streamlit as st

from spd.experiments.lm.streamlit_v1.component_activation_contexts import (
    render_component_activation_contexts_tab,
)
from spd.experiments.lm.streamlit_v1.token_activation_table import render_component_token_table_tab
from spd.experiments.lm.streamlit_v1.token_inspector import render_token_activations_tab
from spd.experiments.lm.streamlit_v1.utils import (
    load_model,
    parse_wandb_url,
    render_model_selector,
)


def main():
    st.set_page_config(page_title="LM Component Explorer", layout="wide")
    st.title("🔍 LM Component Activation Explorer")

    # Initialize session state
    if "model_path" not in st.session_state:
        st.session_state.model_path = None

    # Model selection
    new_model_path = render_model_selector(st.session_state.model_path)
    if new_model_path:
        # Clear all caches when loading new model
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.model_path = new_model_path
        st.rerun()

    # Check if model is loaded
    if not st.session_state.model_path:
        st.warning("👈 Please load a model using the sidebar to begin.")
        st.stop()

    # Load model
    model_data = load_model(st.session_state.model_path)

    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(
        [
            "📈 Token Activation Table",
            "📋 Activation Contexts",
            "🎯 Token Inspector",
        ]
    )

    with tab1:
        render_component_token_table_tab(model_data)

    with tab2:
        render_component_activation_contexts_tab(model_data)

    with tab3:
        render_token_activations_tab(model_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlit app for LM component analysis")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Initial model path (can be changed in UI)",
    )
    args = parser.parse_args()

    if args.model_path:
        st.session_state.model_path = parse_wandb_url(args.model_path)

    main()
