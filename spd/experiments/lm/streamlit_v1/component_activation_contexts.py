"""
Component Activation Contexts tab for the Streamlit app.

Shows example prompts where components activate, with surrounding context tokens.
"""

import html
import io
import zipfile
from typing import Any

import streamlit as st
import torch

from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.streamlit_v1.utils import ModelData
from spd.utils.component_utils import calc_ci_l_zero
from spd.utils.general_utils import extract_batch_data


def _get_highlight_color(
    importance: float,
) -> str:
    """Get highlight color based on importance value.

    Uses semi-transparent green that works in both light and dark themes.
    """
    importance_norm = min(max(importance, 0), 1)  # Clamp to [0, 1]
    # Use green with varying opacity based on importance
    # This works well in both light and dark modes
    opacity = 0.15 + (importance_norm * 0.35)  # Range from 0.15 to 0.5
    return f"rgba(0, 200, 0, {opacity})"


def _render_text_with_token_highlights(
    *,
    raw_text: str,
    offset_mapping: list[tuple[int, int]],
    token_ci_values: list[float],
    active_position: int,
) -> str:
    """
    Render raw text with token highlights based on offset mappings.
    Preserves original spacing and applies gradient coloring based on CI values.
    """
    html_chunks: list[str] = []
    cursor = 0

    for idx, (start, end) in enumerate(offset_mapping):
        # Add any text between tokens
        if cursor < start:
            html_chunks.append(html.escape(raw_text[cursor:start]))

        # Get token text
        token_text = raw_text[start:end]
        if token_text:
            escaped_text = html.escape(token_text)
            ci_value = token_ci_values[idx] if idx < len(token_ci_values) else 0.0

            if ci_value > 0:
                # Apply gradient background based on CI value
                bg_color = _get_highlight_color(ci_value)
                # Add thicker border for the main active token
                border_style = (
                    "border: 2px solid rgba(255,100,0,0.6);" if idx == active_position else ""
                )
                html_chunks.append(
                    f'<span style="background-color:{bg_color}; padding: 2px 4px; '
                    f'border-radius: 3px; {border_style}" '
                    f'title="Importance: {ci_value:.3f}">{escaped_text}</span>'
                )
            else:
                # Regular token without highlighting
                html_chunks.append(f"<span>{escaped_text}</span>")

        cursor = end

    # Add any remaining text
    if cursor < len(raw_text):
        html_chunks.append(html.escape(raw_text[cursor:]))

    return "".join(html_chunks)


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

                causal_importances, _ = _model_data.model.calc_causal_importances(
                    pre_weight_acts=pre_weight_acts,
                    sigmoid_type=_model_data.config.sigmoid_type,
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

                        # Get token IDs for the context window
                        context_token_ids = batch[batch_idx, start_idx:end_idx].tolist()

                        # Decode the entire context to get raw text and offset mappings
                        raw_text = _model_data.tokenizer.decode(context_token_ids)  # pyright: ignore[reportAttributeAccessIssue]

                        # Re-tokenize to get offset mappings
                        context_tokenized = _model_data.tokenizer(  # pyright: ignore[reportCallIssue]
                            raw_text,
                            return_tensors="pt",
                            return_offsets_mapping=True,
                            truncation=False,
                            padding=False,
                        )

                        offset_mapping = context_tokenized["offset_mapping"][0].tolist()

                        # Remove the final offset mapping if it is [0,0], which happens for some
                        # unknown reason
                        if offset_mapping[-1] == [0, 0]:
                            offset_mapping = offset_mapping[:-1]

                        # Calculate CI values for each token in context
                        token_ci_values = []
                        for i in range(len(offset_mapping)):
                            if i < len(context_token_ids):  # Ensure we're within bounds
                                if start_idx + i == seq_idx:
                                    token_ci_values.append(ci_value)
                                else:
                                    # Get CI value for other tokens too if they're active
                                    if (
                                        start_idx + i < ci.shape[1]
                                        and component_active[batch_idx, start_idx + i]
                                    ):
                                        token_ci_values.append(
                                            ci[batch_idx, start_idx + i, component_idx].item()
                                        )
                                    else:
                                        token_ci_values.append(0.0)
                            else:
                                token_ci_values.append(0.0)

                        # Store the context with raw text and offset mappings
                        component_contexts[module_name][component_idx].append(
                            {
                                "raw_text": raw_text,
                                "offset_mapping": offset_mapping,
                                "token_ci_values": token_ci_values,
                                "active_position": seq_idx
                                - start_idx,  # Position of main active token in context
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
                    max_value=10000,
                    value=10,
                    help="Maximum number of batches to process (stops early if enough examples found)",
                )
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=16384,
                    value=64,
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
                        ci_value = example["ci_value"]

                        # Build HTML using offset mappings for proper spacing
                        html_example = _render_text_with_token_highlights(
                            raw_text=example["raw_text"],
                            offset_mapping=example["offset_mapping"],
                            token_ci_values=example["token_ci_values"],
                            active_position=example["active_position"],
                        )

                        # Wrap in example container
                        example_html = (
                            f'<div style="margin: 8px 0; font-family: monospace; font-size: 14px; '
                            f'line-height: 1.8; color: var(--text-color);">'
                            f"<strong>{i + 1}.</strong> "
                            f"{html_example}</div>"
                        )
                        examples_html.append(example_html)

                    # Join all examples without extra line breaks
                    examples_str = "".join(examples_html)

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
                                ci_value = example["ci_value"]
                                raw_text = example["raw_text"]
                                active_position = example["active_position"]
                                offset_mapping = example["offset_mapping"]

                                # Use offset mapping to correctly identify the active token
                                if 0 <= active_position < len(offset_mapping):
                                    start, end = offset_mapping[active_position]
                                    # Insert ** markers around the active token
                                    marked_text = (
                                        raw_text[:start]
                                        + "**"
                                        + raw_text[start:end]
                                        + "**"
                                        + raw_text[end:]
                                    )
                                else:
                                    marked_text = raw_text

                                markdown_lines.append(
                                    f"{i + 1}. CI val {ci_value:.3f}: {marked_text}"
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
                        # HTML download button for single layer
                        html_content = []
                        html_content.append("<!DOCTYPE html>")
                        html_content.append("<html>")
                        html_content.append("<head>")
                        html_content.append('<meta charset="utf-8">')
                        html_content.append(
                            f"<title>Component Activation Contexts - {selected_module}</title>"
                        )
                        html_content.append("<style>")
                        html_content.append("""
                        body {
                            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                            line-height: 1.6;
                            max-width: 1200px;
                            margin: 0 auto;
                            padding: 20px;
                            background-color: #ffffff;
                            color: #333333;
                        }
                        
                        h1, h2 {
                            color: #1a1a1a;
                        }
                        
                        .component-section {
                            background-color: #f8f9fa;
                            border-radius: 8px;
                            padding: 16px;
                            margin-bottom: 16px;
                            border: 1px solid rgba(128, 128, 128, 0.2);
                        }
                        
                        .component-header {
                            font-weight: 600;
                            color: #1a1a1a;
                            margin-bottom: 12px;
                            font-size: 16px;
                        }
                        
                        .examples-container {
                            background-color: #ffffff;
                            border-radius: 4px;
                            padding: 12px;
                            border: 1px solid rgba(128, 128, 128, 0.1);
                        }
                        
                        .example-item {
                            margin: 8px 0;
                            font-family: monospace;
                            font-size: 14px;
                            line-height: 1.8;
                            color: #333333;
                        }
                        
                        /* Highlighted spans */
                        span[title] {
                            position: relative;
                            cursor: help;
                        }
                        
                        span[title]:hover::after {
                            content: attr(title);
                            position: absolute;
                            bottom: 100%;
                            left: 50%;
                            transform: translateX(-50%);
                            background-color: rgba(40, 40, 40, 0.95);
                            color: rgba(255, 255, 255, 1);
                            padding: 2px 6px;
                            border-radius: 3px;
                            font-size: 0.75em;
                            white-space: nowrap;
                            z-index: 10000;
                            pointer-events: none;
                            margin-bottom: 5px;
                            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
                            font-weight: 500;
                            border: 1px solid rgba(255, 255, 255, 0.1);
                        }
                        
                        span[title]:hover::before {
                            content: "";
                            position: absolute;
                            bottom: 100%;
                            left: 50%;
                            transform: translateX(-50%);
                            border: 4px solid transparent;
                            border-top-color: rgba(40, 40, 40, 0.95);
                            z-index: 10000;
                            pointer-events: none;
                            margin-bottom: 1px;
                        }
                        
                        /* Dark mode support */
                        @media (prefers-color-scheme: dark) {
                            body {
                                background-color: #1a1a1a;
                                color: #e0e0e0;
                            }
                            
                            h1, h2 {
                                color: #f0f0f0;
                            }
                            
                            .component-section {
                                background-color: #2a2a2a;
                                border-color: rgba(200, 200, 200, 0.2);
                            }
                            
                            .component-header {
                                color: #f0f0f0;
                            }
                            
                            .examples-container {
                                background-color: #1a1a1a;
                                border-color: rgba(200, 200, 200, 0.1);
                            }
                            
                            .example-item {
                                color: #e0e0e0;
                            }
                            
                            span[title]:hover::after {
                                background-color: rgba(240, 240, 240, 0.95);
                                color: rgba(20, 20, 20, 1);
                                border-color: rgba(0, 0, 0, 0.1);
                            }
                            
                            span[title]:hover::before {
                                border-top-color: rgba(240, 240, 240, 0.95);
                            }
                        }
                    """)
                        html_content.append("</style>")
                        html_content.append("</head>")
                        html_content.append("<body>")
                        html_content.append("<h1>Component Activation Contexts</h1>")
                        html_content.append(f"<h2>Module: {selected_module}</h2>")

                        # Add all component sections
                        for row in table_data:
                            html_content.append('<div class="component-section">')
                            html_content.append(
                                f'<div class="component-header">Component {row["Component"]} '
                            )
                            html_content.append(
                                '<span style="font-weight: normal; opacity: 0.7; font-size: 14px;">'
                            )
                            html_content.append(f"({row['Total Examples']} examples)</span></div>")
                            html_content.append('<div class="examples-container">')
                            html_content.append(row["Example Activation Contexts"])
                            html_content.append("</div></div>")

                        html_content.append("</body>")
                        html_content.append("</html>")

                        html_str = "\n".join(html_content)

                        st.download_button(
                            label=f"Download HTML ({selected_module})",
                            data=html_str,
                            file_name=f"component_contexts_{selected_module}.html",
                            mime="text/html",
                        )

                    # Function to generate HTML for a single module
                    def generate_module_html(
                        module_name: str, module_contexts_data: dict[int, list[dict[str, Any]]]
                    ) -> str:
                        html_content = []
                        html_content.append("<!DOCTYPE html>")
                        html_content.append("<html>")
                        html_content.append("<head>")
                        html_content.append('<meta charset="utf-8">')
                        html_content.append(
                            f"<title>Component Activation Contexts - {module_name}</title>"
                        )
                        html_content.append("<style>")
                        html_content.append("""
                            body {
                                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                                line-height: 1.6;
                                max-width: 1200px;
                                margin: 0 auto;
                                padding: 20px;
                                background-color: #ffffff;
                                color: #333333;
                            }
                            
                            h1, h2 {
                                color: #1a1a1a;
                            }
                            
                            .component-section {
                                background-color: #f8f9fa;
                                border-radius: 8px;
                                padding: 16px;
                                margin-bottom: 16px;
                                border: 1px solid rgba(128, 128, 128, 0.2);
                            }
                            
                            .component-header {
                                font-weight: 600;
                                color: #1a1a1a;
                                margin-bottom: 12px;
                                font-size: 16px;
                            }
                            
                            .examples-container {
                                background-color: #ffffff;
                                border-radius: 4px;
                                padding: 12px;
                                border: 1px solid rgba(128, 128, 128, 0.1);
                            }
                            
                            .example-item {
                                margin: 8px 0;
                                font-family: monospace;
                                font-size: 14px;
                                line-height: 1.8;
                                color: #333333;
                            }
                            
                            /* Highlighted spans */
                            span[title] {
                                position: relative;
                                cursor: help;
                            }
                            
                            span[title]:hover::after {
                                content: attr(title);
                                position: absolute;
                                bottom: 100%;
                                left: 50%;
                                transform: translateX(-50%);
                                background-color: rgba(40, 40, 40, 0.95);
                                color: rgba(255, 255, 255, 1);
                                padding: 2px 6px;
                                border-radius: 3px;
                                font-size: 0.75em;
                                white-space: nowrap;
                                z-index: 10000;
                                pointer-events: none;
                                margin-bottom: 5px;
                                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
                                font-weight: 500;
                                border: 1px solid rgba(255, 255, 255, 0.1);
                            }
                            
                            span[title]:hover::before {
                                content: "";
                                position: absolute;
                                bottom: 100%;
                                left: 50%;
                                transform: translateX(-50%);
                                border: 4px solid transparent;
                                border-top-color: rgba(40, 40, 40, 0.95);
                                z-index: 10000;
                                pointer-events: none;
                                margin-bottom: 1px;
                            }
                            
                            /* Dark mode support */
                            @media (prefers-color-scheme: dark) {
                                body {
                                    background-color: #1a1a1a;
                                    color: #e0e0e0;
                                }
                                
                                h1, h2 {
                                    color: #f0f0f0;
                                }
                                
                                .component-section {
                                    background-color: #2a2a2a;
                                    border-color: rgba(200, 200, 200, 0.2);
                                }
                                
                                .component-header {
                                    color: #f0f0f0;
                                }
                                
                                .examples-container {
                                    background-color: #1a1a1a;
                                    border-color: rgba(200, 200, 200, 0.1);
                                }
                                
                                .example-item {
                                    color: #e0e0e0;
                                }
                                
                                span[title]:hover::after {
                                    background-color: rgba(240, 240, 240, 0.95);
                                    color: rgba(20, 20, 20, 1);
                                    border-color: rgba(0, 0, 0, 0.1);
                                }
                                
                                span[title]:hover::before {
                                    border-top-color: rgba(240, 240, 240, 0.95);
                                }
                            }
                        """)
                        html_content.append("</style>")
                        html_content.append("</head>")
                        html_content.append("<body>")
                        html_content.append("<h1>Component Activation Contexts</h1>")
                        html_content.append(f"<h2>Module: {module_name}</h2>")

                        # Add all component sections for this module
                        for component_id in sorted(module_contexts_data.keys()):
                            component_examples = module_contexts_data[component_id]
                            if not component_examples:
                                continue

                            # Format examples
                            examples_html = []
                            for i, example in enumerate(component_examples):
                                # Build HTML using offset mappings for proper spacing
                                html_example = _render_text_with_token_highlights(
                                    raw_text=example["raw_text"],
                                    offset_mapping=example["offset_mapping"],
                                    token_ci_values=example["token_ci_values"],
                                    active_position=example["active_position"],
                                )

                                # Wrap in example container
                                example_html = (
                                    f'<div style="margin: 8px 0; font-family: monospace; font-size: 14px; '
                                    f'line-height: 1.8; color: var(--text-color);">'
                                    f"<strong>{i + 1}.</strong> "
                                    f"{html_example}</div>"
                                )
                                examples_html.append(example_html)

                            # Join all examples
                            examples_str = "".join(examples_html)

                            html_content.append('<div class="component-section">')
                            html_content.append(
                                f'<div class="component-header">Component {component_id} '
                            )
                            html_content.append(
                                '<span style="font-weight: normal; opacity: 0.7; font-size: 14px;">'
                            )
                            html_content.append(
                                f"({len(component_examples)} examples)</span></div>"
                            )
                            html_content.append('<div class="examples-container">')
                            html_content.append(examples_str)
                            html_content.append("</div></div>")

                        html_content.append("</body>")
                        html_content.append("</html>")

                        return "\n".join(html_content)

                    # Download all layers as ZIP
                    if len(contexts) > 1:  # Only show if there are multiple modules
                        # Create a ZIP file in memory
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for module_name in sorted(contexts.keys()):
                                module_html = generate_module_html(
                                    module_name, contexts[module_name]
                                )
                                zip_file.writestr(
                                    f"component_contexts_{module_name}.html", module_html
                                )

                        zip_buffer.seek(0)

                        st.download_button(
                            label="Download HTML (all layers)",
                            data=zip_buffer.getvalue(),
                            file_name="component_contexts_all_layers.zip",
                            mime="application/zip",
                            key="download_all_layers_zip",
                        )

                    st.divider()

                    # Component contexts display with improved styling
                    st.subheader("Component Activation Examples")

                    # Add custom CSS for better presentation
                    st.markdown(
                        """
                    <style>
                    /* Instant tooltip for importance values */
                    span[title] {
                        position: relative;
                        cursor: help;
                    }
                    
                    span[title]:hover::after {
                        content: attr(title);
                        position: absolute;
                        bottom: 100%;
                        left: 50%;
                        transform: translateX(-50%);
                        background-color: rgba(40, 40, 40, 1);
                        color: rgba(255, 255, 255, 1);
                        padding: 2px 6px;
                        border-radius: 3px;
                        font-size: 0.75em;
                        white-space: nowrap;
                        z-index: 10000;
                        pointer-events: none;
                        margin-bottom: 5px;
                        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
                        font-weight: 500;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                    }
                    
                    /* Dark mode tooltip */
                    @media (prefers-color-scheme: dark) {
                        span[title]:hover::after {
                            background-color: rgba(240, 240, 240, 1);
                            color: rgba(20, 20, 20, 1);
                            border: 1px solid rgba(0, 0, 0, 0.1);
                        }
                    }
                    
                    span[title]:hover::before {
                        content: "";
                        position: absolute;
                        bottom: 100%;
                        left: 50%;
                        transform: translateX(-50%);
                        border: 4px solid transparent;
                        border-top-color: rgba(40, 40, 40, 1);
                        z-index: 10000;
                        pointer-events: none;
                        margin-bottom: 1px;
                    }
                    
                    /* Dark mode tooltip arrow */
                    @media (prefers-color-scheme: dark) {
                        span[title]:hover::before {
                            border-top-color: rgba(240, 240, 240, 1);
                        }
                    }
                    
                    /* Component section styling */
                    .component-section {
                        background-color: var(--secondary-background-color);
                        border-radius: 8px;
                        padding: 16px;
                        margin-bottom: 16px;
                        border: 1px solid rgba(128, 128, 128, 0.2);
                    }
                    
                    .component-header {
                        font-weight: 600;
                        color: var(--text-color);
                        margin-bottom: 12px;
                        font-size: 16px;
                    }
                    
                    .examples-container {
                        background-color: var(--background-color);
                        border-radius: 4px;
                        padding: 12px;
                        border: 1px solid rgba(128, 128, 128, 0.1);
                    }
                    </style>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Display components in a scrollable container using Streamlit's container
                    with st.container(height=1000):
                        for row in table_data:
                            st.markdown(
                                f'<div class="component-section">'
                                f'<div class="component-header">Component {row["Component"]} '
                                f'<span style="font-weight: normal; opacity: 0.7; font-size: 14px;">'
                                f"({row['Total Examples']} examples)</span></div>"
                                f'<div class="examples-container">'
                                f"{row['Example Activation Contexts']}"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("No components found with activations above the threshold.")
