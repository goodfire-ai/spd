"""
Component Activation Contexts tab for the Streamlit app.

Shows example prompts where components activate, with surrounding context tokens.
"""

import html
import io
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import streamlit as st
import torch

from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.streamlit_v1.utils import ModelData
from spd.utils.component_utils import calc_ci_l_zero
from spd.utils.general_utils import extract_batch_data


@dataclass
class AnalysisConfig:
    """Configuration for component activation analysis."""

    dataset_name: str
    dataset_split: str
    column_name: str
    is_tokenized: bool
    streaming: bool
    causal_importance_threshold: float
    n_steps: int
    batch_size: int
    max_seq_len: int
    n_prompts: int
    n_tokens_either_side: int
    seed: int


@dataclass
class ActivationContext:
    """Data for a single component activation context."""

    raw_text: str
    offset_mapping: list[tuple[int, int]]
    token_ci_values: list[float]
    active_position: int  # Position of main active token in context
    ci_value: float


# ============================================================================
# CSS Styling
# ============================================================================


def _get_all_css_styles() -> dict[str, str]:
    """Get all CSS styles organized by category."""
    return {
        "base": """
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
        """,
        "tooltip": """
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
        """,
        "dark_mode": """
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
        """,
    }


def _get_streamlit_css() -> str:
    """Get CSS for Streamlit display."""
    styles = _get_all_css_styles()
    # For Streamlit, we customize the tooltip styles and add CSS variables
    return f"""
        {styles["tooltip"]}
        
        /* Dark mode tooltip */
        @media (prefers-color-scheme: dark) {{
            {styles["tooltip"].replace("rgba(40, 40, 40", "rgba(240, 240, 240").replace("rgba(255, 255, 255", "rgba(20, 20, 20")}
        }}
        
        /* Component section styling */
        .component-section {{
            background-color: var(--secondary-background-color);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }}
        
        .component-header {{
            font-weight: 600;
            color: var(--text-color);
            margin-bottom: 12px;
            font-size: 16px;
        }}
        
        .examples-container {{
            background-color: var(--background-color);
            border-radius: 4px;
            padding: 12px;
            border: 1px solid rgba(128, 128, 128, 0.1);
        }}
    """


def _get_html_export_css() -> str:
    """Get complete CSS for HTML export."""
    styles = _get_all_css_styles()
    return styles["base"] + styles["tooltip"] + styles["dark_mode"]


# ============================================================================
# Rendering Helpers
# ============================================================================


def _get_highlight_color(importance: float) -> str:
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
    # Assert that offset_mapping and token_ci_values have the same length
    assert len(offset_mapping) == len(token_ci_values), (
        f"offset_mapping length ({len(offset_mapping)}) must equal "
        f"token_ci_values length ({len(token_ci_values)})"
    )

    html_chunks: list[str] = []
    cursor = 0

    assert len(offset_mapping) > 0, "offset_mapping must have at least one element"
    assert offset_mapping[0][0] == 0, "first token should start at position 0"

    for idx, (start, end) in enumerate(offset_mapping):
        # Add any text between tokens
        # Cursor should always be <= start (equal when tokens are adjacent)
        assert cursor <= start, f"cursor ({cursor}) should be <= start ({start})"
        if cursor < start:
            html_chunks.append(html.escape(raw_text[cursor:start]))

        escaped_text = html.escape(raw_text[start:end])
        # We already asserted that len(offset_mapping) == len(token_ci_values)
        assert idx < len(token_ci_values), f"idx ({idx}) out of bounds for token_ci_values"
        ci_value = token_ci_values[idx]

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
            html_chunks.append(escaped_text)

        cursor = end

    # Add any remaining text
    if cursor < len(raw_text):
        html_chunks.append(html.escape(raw_text[cursor:]))

    return "".join(html_chunks)


def _format_component_examples_html(component_id: int, contexts: list[ActivationContext]) -> str:
    """Format examples for a single component as HTML."""
    _ = component_id  # Unused but kept for API consistency
    examples_html = []
    for i, ctx in enumerate(contexts):
        # Build HTML using offset mappings for proper spacing
        html_example = _render_text_with_token_highlights(
            raw_text=ctx.raw_text,
            offset_mapping=ctx.offset_mapping,
            token_ci_values=ctx.token_ci_values,
            active_position=ctx.active_position,
        )

        # Wrap in example container
        example_html = (
            f'<div style="margin: 8px 0; font-family: monospace; font-size: 14px; '
            f'line-height: 1.8; color: var(--text-color);">'
            f"<strong>{i + 1}.</strong> "
            f"{html_example}</div>"
        )
        examples_html.append(example_html)

    return "".join(examples_html)


# ============================================================================
# HTML Export
# ============================================================================


def _generate_module_html(
    module_name: str, module_contexts: dict[int, list[ActivationContext]]
) -> str:
    """Generate HTML content for a single module's activation contexts."""
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Component Activation Contexts - {module_name}</title>",
        "<style>",
        _get_html_export_css(),
        "</style>",
        "</head>",
        "<body>",
        "<h1>Component Activation Contexts</h1>",
        f"<h2>Module: {module_name}</h2>",
    ]

    # Add all component sections for this module
    for component_id in sorted(module_contexts.keys()):
        contexts = module_contexts[component_id]
        if not contexts:
            continue

        # Format examples
        examples_html = _format_component_examples_html(component_id, contexts)

        html_content.extend(
            [
                '<div class="component-section">',
                f'<div class="component-header">Component {component_id} ',
                '<span style="font-weight: normal; opacity: 0.7; font-size: 14px;">',
                f"({len(contexts)} examples)</span></div>",
                '<div class="examples-container">',
                examples_html,
                "</div></div>",
            ]
        )

    html_content.extend(["</body>", "</html>"])
    return "\n".join(html_content)


def _create_all_layers_zip(contexts: dict[str, dict[int, list[ActivationContext]]]) -> bytes:
    """Create a ZIP file containing HTML files for all layers."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for module_name in sorted(contexts.keys()):
            module_html = _generate_module_html(module_name, contexts[module_name])
            zip_file.writestr(f"component_contexts_{module_name}.html", module_html)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ============================================================================
# UI Components
# ============================================================================


def _render_configuration_form() -> AnalysisConfig | None:
    """Render the configuration form and return config if submitted."""
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
                is_tokenized = st.checkbox(
                    "Is Tokenized",
                    value=False,
                    help="Whether the dataset is already tokenized",
                )
                streaming = st.checkbox(
                    "Streaming",
                    value=False,
                    help="Whether the dataset is streamed",
                )
                causal_importance_threshold = st.slider(
                    "Causal Importance Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.01,
                    step=0.01,
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
                seed = st.number_input(
                    "Seed",
                    min_value=0,
                    max_value=2147483647,
                    value=0,
                    help="Random seed for reproducible data sampling",
                )

        run_analysis = st.form_submit_button("Run Analysis", type="primary")

        if run_analysis:
            return AnalysisConfig(
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                column_name=column_name,
                causal_importance_threshold=causal_importance_threshold,
                n_steps=n_steps,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                n_prompts=n_prompts,
                n_tokens_either_side=n_tokens_either_side,
                seed=seed,
                is_tokenized=is_tokenized,
                streaming=streaming,
            )
    return None


def _render_l0_scores(l0_scores: dict[str, float]) -> None:
    """Render L0 scores as metrics."""
    st.subheader("L0 over dataset")
    l0_cols = st.columns(min(len(l0_scores), 4))
    for idx, (module_name, score) in enumerate(l0_scores.items()):
        with l0_cols[idx % len(l0_cols)]:
            st.metric(
                label=module_name,
                value=f"{score:.2f}",
                help=f"Average number of active components in {module_name}",
            )


def _render_export_options(
    selected_module: str,
    module_contexts: dict[int, list[ActivationContext]],
    all_contexts: dict[str, dict[int, list[ActivationContext]]],
) -> None:
    """Render export options for the activation contexts."""
    st.subheader("Export Options")

    # HTML download button for single layer
    html_str = _generate_module_html(selected_module, module_contexts)

    st.download_button(
        label=f"Download HTML ({selected_module})",
        data=html_str,
        file_name=f"component_contexts_{selected_module}.html",
        mime="text/html",
    )

    # Download all layers as ZIP
    if len(all_contexts) > 1:  # Only show if there are multiple modules
        zip_data = _create_all_layers_zip(all_contexts)
        st.download_button(
            label="Download HTML (all layers)",
            data=zip_data,
            file_name="component_contexts_all_layers.zip",
            mime="application/zip",
            key="download_all_layers_zip",
        )


def _render_component_contexts(module_contexts: dict[int, list[ActivationContext]]) -> None:
    """Render the component activation contexts."""
    st.subheader("Component Activation Examples")

    # Add custom CSS for better presentation
    st.markdown(
        f"<style>{_get_streamlit_css()}</style>",
        unsafe_allow_html=True,
    )

    # Display components in a scrollable container using Streamlit's container
    with st.container(height=1000):
        for component_id in sorted(module_contexts.keys()):
            contexts = module_contexts[component_id]
            if not contexts:
                continue

            examples_html = _format_component_examples_html(component_id, contexts)

            st.markdown(
                f'<div class="component-section">'
                f'<div class="component-header">Component {component_id}</div>'
                f'<div class="examples-container">'
                f"{examples_html}"
                f"</div></div>",
                unsafe_allow_html=True,
            )


# ============================================================================
# Analysis Helpers
# ============================================================================


def _extract_activation_context(
    *,
    batch: torch.Tensor,
    batch_idx: int,
    seq_idx: int,
    ci: torch.Tensor,
    component_idx: int,
    component_active: torch.Tensor,
    n_tokens_either_side: int,
    tokenizer: Any,
) -> ActivationContext:
    """Extract activation context for a single position."""
    # Get the CI value at this position
    ci_value = ci[batch_idx, seq_idx, component_idx].item()

    # Get context window
    start_idx = max(0, seq_idx - n_tokens_either_side)
    end_idx = min(batch.shape[1], seq_idx + n_tokens_either_side + 1)

    # Get token IDs for the context window
    context_token_ids = batch[batch_idx, start_idx:end_idx].tolist()

    # Decode the entire context to get raw text and offset mappings
    raw_text = tokenizer.decode(context_token_ids)

    # Re-tokenize to get offset mappings
    context_tokenized = tokenizer(
        raw_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
        add_special_tokens=False,
    )

    offset_mapping = context_tokenized["offset_mapping"][0].tolist()

    # Calculate CI values for each token in context
    token_ci_values = []
    for i in range(len(offset_mapping)):
        if i < len(context_token_ids):  # Ensure we're within bounds
            if start_idx + i == seq_idx:
                token_ci_values.append(ci_value)
            else:
                # Get CI value for other tokens too if they're active
                if start_idx + i < ci.shape[1] and component_active[batch_idx, start_idx + i]:
                    token_ci_values.append(ci[batch_idx, start_idx + i, component_idx].item())
                else:
                    token_ci_values.append(0.0)
        else:
            token_ci_values.append(0.0)

    return ActivationContext(
        raw_text=raw_text,
        offset_mapping=offset_mapping,
        token_ci_values=token_ci_values,
        active_position=seq_idx - start_idx,  # Position of main active token in context
        ci_value=ci_value,
    )


def _process_batch_for_contexts(
    *,
    batch: torch.Tensor,
    model_data: ModelData,
    component_contexts: dict[str, dict[int, list[ActivationContext]]],
    config: AnalysisConfig,
) -> dict[str, float]:
    """Process a single batch to find activation contexts."""
    # Get activations before each component
    with torch.no_grad():
        _, pre_weight_acts = model_data.model(
            batch, mode="pre_forward_cache", module_names=list(model_data.components.keys())
        )

        causal_importances, _ = model_data.model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type=model_data.config.sigmoid_type,
            detach_inputs=True,
        )

    # Calculate L0 scores
    ci_l_zero_vals: dict[str, float] = {}
    for module_name, ci in causal_importances.items():
        ci_l_zero_vals[module_name] = calc_ci_l_zero(ci, config.causal_importance_threshold)

    # Find activation contexts
    for module_name, ci in causal_importances.items():
        assert ci.ndim == 3, "CI must be 3D (batch, seq_len, C)"

        if module_name not in component_contexts:
            component_contexts[module_name] = {}

        # Find active components
        active_mask = ci > config.causal_importance_threshold

        # For each component
        for component_idx in range(model_data.model.C):
            if component_idx not in component_contexts[module_name]:
                component_contexts[module_name][component_idx] = []

            # Skip if we already have enough examples
            if len(component_contexts[module_name][component_idx]) >= config.n_prompts:
                continue

            # Get positions where this component is active
            component_active = active_mask[:, :, component_idx]

            # Find activations in this batch
            batch_idxs, seq_idxs = torch.where(component_active)

            for batch_idx, seq_idx in zip(batch_idxs.tolist(), seq_idxs.tolist(), strict=True):
                # Skip if we have enough examples
                if len(component_contexts[module_name][component_idx]) >= config.n_prompts:
                    break

                context = _extract_activation_context(
                    batch=batch,
                    batch_idx=batch_idx,
                    seq_idx=seq_idx,
                    ci=ci,
                    component_idx=component_idx,
                    component_active=component_active,
                    n_tokens_either_side=config.n_tokens_either_side,
                    tokenizer=model_data.tokenizer,
                )

                component_contexts[module_name][component_idx].append(context)

    return ci_l_zero_vals


def _calculate_average_l0_scores(
    l0_scores_sum: defaultdict[str, float], l0_scores_count: int
) -> dict[str, float]:
    """Calculate average L0 scores from accumulated sums."""
    avg_l0_scores: dict[str, float] = {}
    if l0_scores_count > 0:
        for layer_name, score_sum in l0_scores_sum.items():
            avg_l0_scores[layer_name] = score_sum / l0_scores_count
    return avg_l0_scores


def _check_all_components_have_enough_examples(
    component_contexts: dict[str, dict[int, list[ActivationContext]]],
    n_prompts: int,
    n_components: int,
) -> bool:
    """Check if all components have enough examples."""
    for module_name in component_contexts:
        for component_idx in range(n_components):
            if component_idx not in component_contexts[module_name]:
                return False
            if len(component_contexts[module_name][component_idx]) < n_prompts:
                return False
    return True


# ============================================================================
# Main Analysis Function
# ============================================================================


@st.cache_data(show_spinner="Finding component activation contexts...")
def find_component_activation_contexts(
    _model_data: ModelData,
    config: AnalysisConfig,
) -> tuple[
    dict[str, dict[int, list[ActivationContext]]],
    dict[str, float],
]:
    """Find example prompts where each component activates with surrounding context.

    Note: Parameters prefixed with _ are Streamlit conventions indicating the parameter
    should not trigger cache invalidation when it changes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataloader
    data_config = DatasetConfig(
        name=config.dataset_name,
        hf_tokenizer_path=_model_data.config.pretrained_model_name_hf,
        split=config.dataset_split,
        n_ctx=config.max_seq_len,
        is_tokenized=config.is_tokenized,
        streaming=config.streaming,
        column_name=config.column_name,
    )

    dataloader, _ = create_data_loader(
        dataset_config=data_config,
        batch_size=config.batch_size,
        buffer_size=1000,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )

    # Initialize tracking
    component_contexts: dict[str, dict[int, list[ActivationContext]]] = {}
    l0_scores_sum: defaultdict[str, float] = defaultdict(float)
    l0_scores_count = 0

    data_iter = iter(dataloader)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for step in range(config.n_steps):
        try:
            batch = extract_batch_data(next(data_iter))
            batch = batch.to(device)

            ci_l_zero_vals = _process_batch_for_contexts(
                batch=batch,
                model_data=_model_data,
                component_contexts=component_contexts,
                config=config,
            )

            for layer_name, layer_ci_l_zero in ci_l_zero_vals.items():
                l0_scores_sum[layer_name] += layer_ci_l_zero
            l0_scores_count += 1

            progress = (step + 1) / config.n_steps
            progress_bar.progress(progress)
            progress_text.text(f"Processed {step + 1}/{config.n_steps} batches")

            if _check_all_components_have_enough_examples(
                component_contexts, config.n_prompts, _model_data.model.C
            ):
                st.info(f"Found enough examples for all components after {step + 1} batches.")
                break

        except StopIteration:
            st.warning(f"Dataset exhausted after {step} batches. Returning results.")
            break

    progress_bar.empty()
    progress_text.empty()

    avg_l0_scores = _calculate_average_l0_scores(l0_scores_sum, l0_scores_count)

    return component_contexts, avg_l0_scores


# ============================================================================
# Main UI Function
# ============================================================================


def render_component_activation_contexts_tab(model_data: ModelData):
    """Render the component activation contexts analysis."""
    st.subheader("Component Activation Contexts")
    st.markdown(
        "This analysis shows example prompts where each component activates, "
        "with surrounding context tokens. The activating token is highlighted with its CI value."
    )

    # Configuration and run analysis
    config = _render_configuration_form()

    if config:
        # Run the analysis
        contexts, l0_scores = find_component_activation_contexts(
            _model_data=model_data,
            config=config,
        )

        # Store results in session state
        st.session_state.component_context_results = {
            "contexts": contexts,
            "l0_scores": l0_scores,
        }

    # Display results if available
    if "component_context_results" in st.session_state:
        results: dict[str, Any] = st.session_state.component_context_results
        contexts: dict[str, dict[int, list[ActivationContext]]] = results.get("contexts", {})
        l0_scores: dict[str, float] = results.get("l0_scores", {})

        st.success("Analysis complete!")

        # Display L0 scores as summary metrics
        if l0_scores:
            _render_l0_scores(l0_scores)

        # Module selection
        if contexts:
            module_names: list[str] = sorted(contexts.keys())
            selected_module = st.selectbox(
                "Select Module", options=module_names, key="context_module_selector"
            )

            if selected_module and selected_module in contexts:
                module_contexts: dict[int, list[ActivationContext]] = contexts[selected_module]

                if any(module_contexts.values()):
                    # Export options
                    _render_export_options(selected_module, module_contexts, contexts)
                    st.divider()

                    # Display contexts
                    _render_component_contexts(module_contexts)
                else:
                    st.info("No components found with activations above the threshold.")
