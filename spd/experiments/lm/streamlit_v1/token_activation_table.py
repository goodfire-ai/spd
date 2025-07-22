"""
Component Token Table tab for the Streamlit app.
"""

from typing import Any

import pandas as pd
import streamlit as st
import torch

from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.streamlit_v1.utils import ModelData
from spd.utils.component_utils import calc_causal_importances, calc_ci_l_zero
from spd.utils.general_utils import extract_batch_data


@st.cache_data(show_spinner="Analyzing component token activations across dataset...")
def analyze_component_token_table(
    _model_path: str,
    _model_data: ModelData,
    dataset_name: str,
    dataset_split: str,
    column_name: str,
    causal_importance_threshold: float,
    n_steps: int,
    batch_size: int,
    max_seq_len: int,
) -> tuple[
    dict[str, dict[int, dict[int, int]]],
    dict[str, dict[int, dict[int, list[float]]]],
    int,
    dict[int, int],
    dict[str, float],
]:
    """Analyze which tokens activate each component across the dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataloader
    data_config = DatasetConfig(
        name=dataset_name,
        hf_tokenizer_path=_model_data.config.pretrained_model_name_hf,
        split=dataset_split,
        n_ctx=max_seq_len,
        is_tokenized=False,
        streaming=False,
        column_name=column_name,
    )

    dataloader, _ = create_data_loader(
        dataset_config=data_config,
        batch_size=batch_size,
        buffer_size=1000,
        global_seed=42,
        ddp_rank=0,
        ddp_world_size=1,
    )

    # Initialize token activation tracking
    component_token_activations: dict[str, dict[int, dict[int, int]]] = {}
    component_token_ci_values: dict[str, dict[int, dict[int, list[float]]]] = {}
    total_token_counts: dict[int, int] = {}  # Track total appearances of each token
    l0_scores_sum: dict[str, float] = {}  # Track sum of L0 scores for averaging
    l0_scores_count = 0  # Track number of batches for averaging

    total_tokens_processed = 0
    data_iter = iter(dataloader)

    progress_bar = st.progress(0)
    progress_text = st.empty()

    for step in range(n_steps):
        try:
            batch = extract_batch_data(next(data_iter))
            batch = batch.to(device)

            # Count tokens in this batch
            total_tokens_processed += batch.numel()

            # Count all tokens in this batch
            for token_id in batch.flatten().tolist():
                if token_id not in total_token_counts:
                    total_token_counts[token_id] = 0
                total_token_counts[token_id] += 1

            # Get activations before each component
            with torch.no_grad():
                _, pre_weight_acts = _model_data.model.forward_with_pre_forward_cache_hooks(
                    batch, module_names=list(_model_data.components.keys())
                )

                Vs = {module_name: v.V for module_name, v in _model_data.components.items()}

                causal_importances, _ = calc_causal_importances(
                    pre_weight_acts=pre_weight_acts,
                    Vs=Vs,
                    gates=_model_data.gates,
                    detach_inputs=True,
                )

            # Calculate L0 scores for this batch
            ci_l_zero = calc_ci_l_zero(causal_importances=causal_importances)
            for layer_name, layer_ci_l_zero in ci_l_zero.items():
                if layer_name not in l0_scores_sum:
                    l0_scores_sum[layer_name] = 0.0
                l0_scores_sum[layer_name] += layer_ci_l_zero
            l0_scores_count += 1

            for module_name, ci in causal_importances.items():
                assert ci.ndim == 3, "CI must be 3D (batch, seq_len, C)"

                # Find active components
                active_mask = ci > causal_importance_threshold

                # Get token IDs for this batch
                token_ids = batch

                # For each component, track which tokens it activates on
                for component_idx in range(_model_data.model.C):
                    # Get positions where this component is active
                    component_active = active_mask[:, :, component_idx]

                    # Get the tokens at those positions
                    active_tokens = token_ids[component_active]

                    # Get the CI values at those positions
                    active_ci_values = ci[:, :, component_idx][component_active]

                    # Count occurrences and store CI values
                    for token_id, ci_val in zip(
                        active_tokens.tolist(), active_ci_values.tolist(), strict=False
                    ):
                        # Initialize nested dicts if they don't exist
                        if module_name not in component_token_activations:
                            component_token_activations[module_name] = {}
                        if component_idx not in component_token_activations[module_name]:
                            component_token_activations[module_name][component_idx] = {}
                        if token_id not in component_token_activations[module_name][component_idx]:
                            component_token_activations[module_name][component_idx][token_id] = 0

                        if module_name not in component_token_ci_values:
                            component_token_ci_values[module_name] = {}
                        if component_idx not in component_token_ci_values[module_name]:
                            component_token_ci_values[module_name][component_idx] = {}
                        if token_id not in component_token_ci_values[module_name][component_idx]:
                            component_token_ci_values[module_name][component_idx][token_id] = []

                        component_token_activations[module_name][component_idx][token_id] += 1
                        component_token_ci_values[module_name][component_idx][token_id].append(
                            ci_val
                        )

            # Update progress
            progress = (step + 1) / n_steps
            progress_bar.progress(progress)
            progress_text.text(
                f"Processed {step + 1}/{n_steps} batches ({total_tokens_processed:,} tokens)"
            )

        except StopIteration:
            st.warning(f"Dataset exhausted after {step} batches. Returning results.")
            break

    progress_bar.empty()
    progress_text.empty()

    # Calculate average L0 scores
    avg_l0_scores: dict[str, float] = {}
    if l0_scores_count > 0:
        for layer_name, score_sum in l0_scores_sum.items():
            avg_l0_scores[layer_name] = score_sum / l0_scores_count

    return (
        component_token_activations,
        component_token_ci_values,
        total_tokens_processed,
        total_token_counts,
        avg_l0_scores,
    )


@st.fragment
def render_component_token_table_tab(model_data: ModelData):
    """Render the component token table analysis."""
    st.subheader("Component Token Activation Analysis")
    st.markdown(
        "This analysis shows which tokens most frequently activate each component across a dataset. "
        "Higher causal importance values indicate stronger component activation."
    )

    # Configuration options - wrap in a form to prevent reruns on every change
    with st.form("component_token_config"):
        with st.expander("Analysis Configuration", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                dataset_name = st.text_input(
                    "Dataset Name",
                    value="SimpleStories/SimpleStories",
                    help="HuggingFace dataset to analyze",
                )
                dataset_split = st.text_input(
                    "Dataset Split",
                    value="test",
                    help="Dataset split to analyze",
                )
                column_name = st.text_input(
                    "Text Column",
                    value="story",
                    help="Column containing the text to analyze",
                )
                causal_importance_threshold = st.slider(
                    "Causal Importance Threshold",
                    min_value=0.001,
                    max_value=0.5,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    help="Minimum CI value for a component to be considered active",
                )

            with col2:
                n_steps = st.number_input(
                    "Number of Batches",
                    min_value=1,
                    max_value=1000,
                    value=10,
                    help="Number of batches to process",
                )
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=512,
                    value=32,
                    help="Batch size for processing",
                )
                max_seq_len = st.number_input(
                    "Max Sequence Length",
                    min_value=128,
                    max_value=2048,
                    value=512,
                    help="Maximum sequence length for tokenization",
                )

        run_analysis = st.form_submit_button("Run Analysis", type="primary")

    if run_analysis:
        # Run the analysis
        activations, ci_values, total_tokens, token_counts, l0_scores = (
            analyze_component_token_table(
                _model_path=st.session_state.model_path,
                _model_data=model_data,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                column_name=column_name,
                causal_importance_threshold=causal_importance_threshold,
                n_steps=n_steps,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
            )
        )

        # Store results in session state
        st.session_state.token_activation_results = {
            "activations": activations,
            "ci_values": ci_values,
            "total_tokens": total_tokens,
            "token_counts": token_counts,
            "l0_scores": l0_scores,
        }

    # Display results if available
    if "token_activation_results" in st.session_state:
        results: dict[str, Any] = st.session_state.token_activation_results
        activations: dict[str, dict[int, dict[int, int]]] = results.get("activations", {})
        ci_values: dict[str, dict[int, dict[int, list[float]]]] = results.get("ci_values", {})
        total_tokens: int = results.get("total_tokens", 0)
        total_token_counts: dict[int, int] = results.get(
            "token_counts", {}
        )  # Rename to avoid shadowing
        l0_scores: dict[str, float] = results.get("l0_scores", {})

        st.success(f"Analysis complete! Processed {total_tokens:,} tokens.")

        # Display L0 scores as summary metrics
        if l0_scores:
            st.subheader("L0 over dataset")
            l0_cols = st.columns(min(len(l0_scores), 4))
            for idx, (module_name, score) in enumerate(l0_scores.items()):
                with l0_cols[idx % len(l0_cols)]:
                    st.metric(
                        label=module_name,
                        value=f"{score:.2f}",
                        help=f"Average number of active components in {module_name}",
                    )

        # Module selection
        if activations:
            module_names: list[str] = sorted(activations.keys())
            selected_module = st.selectbox(
                "Select Module", options=module_names, key="component_module_selector"
            )

            if selected_module and selected_module in activations:
                module_activations: dict[int, dict[int, int]] = activations[selected_module]
                module_ci_values: dict[int, dict[int, list[float]]] = ci_values.get(
                    selected_module, {}
                )

                # Prepare data for display
                table_data: list[dict[str, Any]] = []

                for component_id in sorted(module_activations.keys()):
                    token_counts: dict[int, int] = module_activations[component_id]
                    if not token_counts:
                        continue

                    # Create list of tokens with their mean CI values and counts
                    token_ci_count_tuples: list[tuple[str, float, int, int]] = []
                    for token_id, count in token_counts.items():
                        try:
                            token_text = model_data.tokenizer.decode([token_id])  # pyright: ignore[reportAttributeAccessIssue]
                        except Exception:
                            token_text = f"<token_{token_id}>"

                        # Clean up the token text
                        token_text = token_text.strip()
                        if token_text:  # Only add non-empty tokens
                            # Calculate mean CI value for this token
                            ci_vals: list[float] = module_ci_values.get(component_id, {}).get(
                                token_id, []
                            )
                            mean_ci = sum(ci_vals) / len(ci_vals) if ci_vals else 0.0
                            # Get total count for this token
                            total_count = total_token_counts.get(token_id, 0)
                            assert total_count >= count, (
                                f"Token {token_id} has more activations ({count}) than total appearances ({total_count})"
                            )
                            token_ci_count_tuples.append((token_text, mean_ci, count, total_count))

                    # Sort by count first (descending), then by mean CI value (descending)
                    sorted_tokens: list[tuple[str, float, int, int]] = sorted(
                        token_ci_count_tuples, key=lambda x: (x[2], x[1]), reverse=True
                    )

                    if sorted_tokens:
                        # Format tokens for display
                        formatted_tokens: list[str] = []
                        for token_text, mean_ci, count, total_count in sorted_tokens:
                            formatted_tokens.append(
                                f"{token_text} ({mean_ci:.2f}, {count}/{total_count})"
                            )

                        tokens_str = " • ".join(formatted_tokens)
                        table_data.append(
                            {
                                "Component": component_id,
                                "Activating Tokens (mean_ci, count/total)": tokens_str,
                                "Total Unique Tokens": len(
                                    token_counts
                                ),  # This is correct - refers to activation counts
                            }
                        )

                if table_data:
                    # Display as a dataframe
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True, height=600)

                    # Download option
                    # Create markdown table manually
                    markdown_lines = []
                    markdown_lines.append("# Component Token Activations")
                    markdown_lines.append(f"\n## Module: {selected_module}\n")

                    # Table header
                    markdown_lines.append(
                        "| Component | Activating Tokens (mean_ci, count/total) | Total Unique Tokens |"
                    )
                    markdown_lines.append(
                        "|-----------|-----------------------------------|---------------------|"
                    )

                    # Table rows
                    for _, row in df.iterrows():
                        component = row["Component"]
                        tokens = row["Activating Tokens (mean_ci, count/total)"]
                        total = row["Total Unique Tokens"]
                        markdown_lines.append(f"| {component} | {tokens} | {total} |")

                    markdown_content = "\n".join(markdown_lines)

                    st.download_button(
                        label="Download as Markdown",
                        data=markdown_content,
                        file_name=f"component_tokens_{selected_module}.md",
                        mime="text/markdown",
                    )
                else:
                    st.info("No components found with activations above the threshold.")
