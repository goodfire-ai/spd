"""
Component Activation Contexts tab for the Streamlit app.

Shows example prompts where components activate, with surrounding context tokens.
"""

import html
from typing import Any

import streamlit as st
import torch

from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.streamlit_v1.utils import ModelData
from spd.utils.component_utils import calc_causal_importances, calc_ci_l_zero
from spd.utils.general_utils import extract_batch_data


@st.cache_data(show_spinner="Finding component activation contexts...")
def find_component_activation_contexts(
    _model_path: str,
    _model_data: ModelData,
    dataset_name: str,
    dataset_split: str,
    column_name: str,
    causal_importance_threshold: float,
    n_steps: int,
    batch_size: int,
    max_seq_len: int,
    n_prompts: int,
    n_tokens_either_side: int,
) -> tuple[
    dict[str, dict[int, list[dict[str, Any]]]],
    dict[str, float],
]:
    """Find example prompts where each component activates with surrounding context."""
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

    # Initialize tracking
    component_contexts: dict[str, dict[int, list[dict[str, Any]]]] = {}
    l0_scores_sum: dict[str, float] = {}
    l0_scores_count = 0

    data_iter = iter(dataloader)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for step in range(n_steps):
        try:
            batch = extract_batch_data(next(data_iter))
            batch = batch.to(device)

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

            # Calculate L0 scores
            ci_l_zero = calc_ci_l_zero(causal_importances=causal_importances)
            for layer_name, layer_ci_l_zero in ci_l_zero.items():
                if layer_name not in l0_scores_sum:
                    l0_scores_sum[layer_name] = 0.0
                l0_scores_sum[layer_name] += layer_ci_l_zero
            l0_scores_count += 1

            # Find activation contexts
            for module_name, ci in causal_importances.items():
                assert ci.ndim == 3, "CI must be 3D (batch, seq_len, C)"

                if module_name not in component_contexts:
                    component_contexts[module_name] = {}

                # Find active components
                active_mask = ci > causal_importance_threshold

                # For each component
                for component_idx in range(_model_data.model.C):
                    if component_idx not in component_contexts[module_name]:
                        component_contexts[module_name][component_idx] = []

                    # Skip if we already have enough examples
                    if len(component_contexts[module_name][component_idx]) >= n_prompts:
                        continue

                    # Get positions where this component is active
                    component_active = active_mask[:, :, component_idx]

                    # Find activations in this batch
                    batch_idxs, seq_idxs = torch.where(component_active)

                    for batch_idx, seq_idx in zip(
                        batch_idxs.tolist(), seq_idxs.tolist(), strict=False
                    ):
                        # Skip if we have enough examples
                        if len(component_contexts[module_name][component_idx]) >= n_prompts:
                            break

                        # Get the CI value at this position
                        ci_value = ci[batch_idx, seq_idx, component_idx].item()

                        # Get context window
                        start_idx = max(0, seq_idx - n_tokens_either_side)
                        end_idx = min(batch.shape[1], seq_idx + n_tokens_either_side + 1)

                        # Get token IDs and decode
                        token_ids = batch[batch_idx, start_idx:end_idx].tolist()
                        tokens = []

                        for i, token_id in enumerate(token_ids):
                            try:
                                token_text = _model_data.tokenizer.decode([token_id])  # pyright: ignore[reportAttributeAccessIssue]
                            except Exception:
                                token_text = f"<token_{token_id}>"

                            # Mark the activating token
                            if start_idx + i == seq_idx:
                                tokens.append((token_text, ci_value, True))  # (text, ci, is_active)
                            else:
                                tokens.append((token_text, None, False))

                        # Store the context
                        component_contexts[module_name][component_idx].append(
                            {
                                "tokens": tokens,
                                "position": seq_idx,
                                "ci_value": ci_value,
                            }
                        )

            # Update progress
            progress = (step + 1) / n_steps
            progress_bar.progress(progress)
            progress_text.text(f"Processed {step + 1}/{n_steps} batches")

            # Check if we have enough examples for all components
            all_components_have_enough = True
            for module_name in component_contexts:
                for component_idx in range(_model_data.model.C):
                    if component_idx not in component_contexts[module_name]:
                        all_components_have_enough = False
                        break
                    if len(component_contexts[module_name][component_idx]) < n_prompts:
                        all_components_have_enough = False
                        break
                if not all_components_have_enough:
                    break

            if all_components_have_enough:
                st.info(f"Found enough examples for all components after {step + 1} batches.")
                break

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

    return component_contexts, avg_l0_scores


@st.fragment
def render_component_activation_contexts_tab(model_data: ModelData):
    """Render the component activation contexts analysis."""
    st.subheader("Component Activation Contexts")
    st.markdown(
        "This analysis shows example prompts where each component activates, "
        "with surrounding context tokens. The activating token is highlighted with its CI value."
    )

    # Configuration options
    with st.form("component_context_config"):
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
                n_prompts = st.number_input(
                    "Examples per Component",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="Number of example prompts to show per component",
                )
                n_tokens_either_side = st.number_input(
                    "Context Tokens Either Side",
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="Number of tokens to show on either side of the activating token",
                )
                n_steps = st.number_input(
                    "Max Batches to Process",
                    min_value=1,
                    max_value=1000,
                    value=100,
                    help="Maximum number of batches to process (stops early if enough examples found)",
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
        contexts, l0_scores = find_component_activation_contexts(
            _model_path=st.session_state.model_path,
            _model_data=model_data,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            column_name=column_name,
            causal_importance_threshold=causal_importance_threshold,
            n_steps=n_steps,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            n_prompts=n_prompts,
            n_tokens_either_side=n_tokens_either_side,
        )

        # Store results in session state
        st.session_state.component_context_results = {
            "contexts": contexts,
            "l0_scores": l0_scores,
        }

    # Display results if available
    if "component_context_results" in st.session_state:
        results: dict[str, Any] = st.session_state.component_context_results
        contexts: dict[str, dict[int, list[dict[str, Any]]]] = results.get("contexts", {})
        l0_scores: dict[str, float] = results.get("l0_scores", {})

        st.success("Analysis complete!")

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
        if contexts:
            module_names: list[str] = sorted(contexts.keys())
            selected_module = st.selectbox(
                "Select Module", options=module_names, key="context_module_selector"
            )

            if selected_module and selected_module in contexts:
                module_contexts: dict[int, list[dict[str, Any]]] = contexts[selected_module]

                # Prepare data for display
                table_data: list[dict[str, Any]] = []

                for component_id in sorted(module_contexts.keys()):
                    component_examples = module_contexts[component_id]
                    if not component_examples:
                        continue

                    # Format examples
                    examples_html = []
                    for i, example in enumerate(component_examples):
                        tokens = example["tokens"]
                        ci_value = example["ci_value"]

                        # Build HTML for this example
                        token_parts = []
                        for token_text, _, is_active in tokens:
                            # Escape HTML special characters
                            escaped_text = html.escape(token_text)
                            if is_active:
                                # Use outline style similar to token inspector
                                token_parts.append(
                                    f'<span style="border: 2px solid rgb(200,0,0); '
                                    f'border-radius: 2px; padding: 1px 2px; margin: 0 1px;">'
                                    f"{escaped_text}</span>"
                                )
                            else:
                                # Wrap non-active tokens in spans with minimal margins to preserve spacing
                                token_parts.append(
                                    f'<span style="margin: 0 1px;">{escaped_text}</span>'
                                )

                        example_html = f"{i + 1}. CI val {ci_value:.3f}: {''.join(token_parts)}"
                        examples_html.append(example_html)

                    # Join all examples
                    examples_str = "<br><br>".join(examples_html)

                    table_data.append(
                        {
                            "Component": component_id,
                            "Example Activation Contexts": examples_str,
                            "Total Examples": len(component_examples),
                        }
                    )

                if table_data:
                    # Download options at the top
                    st.subheader("Export Options")
                    col1, col2 = st.columns(2)

                    with col1:
                        # Original markdown format
                        markdown_lines = []
                        markdown_lines.append("# Component Activation Contexts")
                        markdown_lines.append(f"\n## Module: {selected_module}\n")

                        for row in table_data:
                            markdown_lines.append(f"### Component {row['Component']}")

                            # Convert HTML back to markdown for download
                            component_examples = module_contexts[row["Component"]]
                            for i, example in enumerate(component_examples):
                                tokens = example["tokens"]
                                ci_value = example["ci_value"]
                                token_parts = []
                                for token_text, _, is_active in tokens:
                                    if is_active:
                                        token_parts.append(f"**{token_text}**")
                                    else:
                                        token_parts.append(token_text)
                                markdown_lines.append(
                                    f"{i + 1}. CI val {ci_value:.3f}: {''.join(token_parts)}"
                                )
                            markdown_lines.append("")

                        markdown_content = "\n".join(markdown_lines)

                        st.download_button(
                            label="Download as Markdown",
                            data=markdown_content,
                            file_name=f"component_contexts_{selected_module}.md",
                            mime="text/markdown",
                        )

                    with col2:
                        # Markdown table format
                        table_lines = []
                        table_lines.append("# Component Activation Contexts")
                        table_lines.append(f"\n## Module: {selected_module}\n")
                        table_lines.append("| Component | Example # | CI Value | Context |")
                        table_lines.append("|-----------|-----------|----------|---------|")

                        for row in table_data:
                            component_examples = module_contexts[row["Component"]]
                            for i, example in enumerate(component_examples):
                                tokens = example["tokens"]
                                ci_value = example["ci_value"]
                                token_parts = []
                                for token_text, _, is_active in tokens:
                                    if is_active:
                                        token_parts.append(f"**{token_text}**")
                                    else:
                                        token_parts.append(token_text)
                                context = "".join(token_parts).replace(
                                    "|", "\\|"
                                )  # Escape pipes in table
                                table_lines.append(
                                    f"| {row['Component']} | {i + 1} | {ci_value:.3f} | {context} |"
                                )

                        table_content = "\n".join(table_lines)

                        st.download_button(
                            label="Download as Markdown Table",
                            data=table_content,
                            file_name=f"component_contexts_{selected_module}_table.md",
                            mime="text/markdown",
                        )

                    st.divider()

                    # Component contexts display in scrollable container
                    st.subheader("Component Activation Examples")

                    # Create a container with constrained height
                    container = st.container(height=600)

                    with container:
                        for row in table_data:
                            col1, col2 = st.columns([1, 11])
                            with col1:
                                st.markdown(f"**Component {row['Component']}**")
                            with col2:
                                # The content already has HTML formatting
                                st.markdown(
                                    f'<div style="line-height:1.7; font-family:monospace;">'
                                    f"{row['Example Activation Contexts']}</div>",
                                    unsafe_allow_html=True,
                                )
                            st.divider()
                else:
                    st.info("No components found with activations above the threshold.")
