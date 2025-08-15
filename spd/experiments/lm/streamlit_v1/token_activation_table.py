"""
Component Token Table tab for the Streamlit app.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st
import torch

from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.experiments.lm.streamlit_v1.utils import ModelData
from spd.utils.component_utils import calc_ci_l_zero
from spd.utils.general_utils import extract_batch_data


@dataclass
class AnalysisConfig:
    """Configuration for token activation analysis."""

    dataset_name: str
    dataset_split: str
    column_name: str
    is_tokenized: bool
    streaming: bool
    causal_importance_threshold: float
    n_steps: int
    batch_size: int
    max_seq_len: int
    seed: int
    min_act_frequency: float


@dataclass
class TokenActivationData:
    """Data for a single token's activation statistics."""

    token_text: str
    token_id: int
    activation_count: int
    total_count: int
    mean_ci: float
    activation_fraction: float


@dataclass
class AnalysisResults:
    """Results from token activation analysis."""

    component_token_activations: dict[str, dict[int, dict[int, int]]]
    component_token_ci_values: dict[str, dict[int, dict[int, list[float]]]]
    total_tokens_processed: int
    total_token_counts: dict[int, int]
    avg_l0_scores: dict[str, float]


# ============================================================================
# UI Helper Functions
# ============================================================================


def _render_configuration_form() -> AnalysisConfig | None:
    """Render the configuration form and return config if submitted."""
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
                    max_value=0.5,
                    value=0.01,
                    step=0.01,
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
                seed = st.number_input(
                    "Seed",
                    min_value=0,
                    max_value=2147483647,
                    value=0,
                    help="Random seed for reproducible data sampling",
                )
                min_act_frequency = st.slider(
                    "Minimum Token Activation Frequency",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                    format="%.3f",
                    help="Minimum fraction of token appearances where component activates (0-1)",
                )

        run_analysis = st.form_submit_button("Run Analysis", type="primary")

        if run_analysis:
            return AnalysisConfig(
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                column_name=column_name,
                is_tokenized=is_tokenized,
                streaming=streaming,
                causal_importance_threshold=causal_importance_threshold,
                n_steps=n_steps,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                seed=seed,
                min_act_frequency=min_act_frequency,
            )
    return None


def _process_component_tokens(
    *,
    component_token_counts: dict[int, int],
    module_ci_values: dict[int, list[float]],
    total_token_counts: dict[int, int],
    min_act_frequency: float,
    tokenizer: Any,
) -> list[TokenActivationData]:
    """Process tokens for a single component and return activation data."""
    token_data_list: list[TokenActivationData] = []

    for token_id, count in component_token_counts.items():
        total_count = total_token_counts[token_id]

        activation_fraction = count / total_count

        if activation_fraction < min_act_frequency:
            continue

        token_text = tokenizer.decode([token_id])

        token_text = token_text.strip()
        if token_text:  # Only add non-empty tokens
            # Calculate mean CI value for this token
            ci_vals: list[float] = module_ci_values.get(token_id, [])
            mean_ci = sum(ci_vals) / len(ci_vals) if ci_vals else 0.0

            assert total_count >= count, (
                f"Token {token_id} has more activations ({count}) than total appearances ({total_count})"
            )

            token_data_list.append(
                TokenActivationData(
                    token_text=token_text,
                    token_id=token_id,
                    activation_count=count,
                    total_count=total_count,
                    mean_ci=mean_ci,
                    activation_fraction=activation_fraction,
                )
            )

    return token_data_list


def _format_token_display(tokens: list[TokenActivationData]) -> str:
    """Format token activation data for display."""
    # Sort by count first (descending), then by mean CI value (descending)
    sorted_tokens = sorted(tokens, key=lambda x: (x.activation_count, x.mean_ci), reverse=True)

    formatted_tokens: list[str] = []
    for token in sorted_tokens:
        formatted_tokens.append(
            f"{token.token_text} ({token.mean_ci:.2f}, {token.activation_count}/{token.total_count})"
        )

    return " • ".join(formatted_tokens)


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


def _create_markdown_export(df: pd.DataFrame, selected_module: str) -> str:
    """Create markdown table content for export."""
    markdown_lines = []
    markdown_lines.append("# Component Token Activations")
    markdown_lines.append(f"\n## Module: {selected_module}\n")

    # Table header
    markdown_lines.append(
        "| Component | Activating Tokens (mean_ci, count/total) | Total Unique Tokens |"
    )
    markdown_lines.append("|-----------|-----------------------------------|---------------------|")

    # Table rows
    for _, row in df.iterrows():
        component = row["Component"]
        tokens = row["Activating Tokens (mean_ci, count/total)"]
        total = row["Total Unique Tokens"]
        markdown_lines.append(f"| {component} | {tokens} | {total} |")

    return "\n".join(markdown_lines)


def _render_token_table(
    *,
    table_data: list[dict[str, Any]],
    selected_module: str,
) -> None:
    """Render the token activation table with export options."""
    if table_data:
        # Display as a dataframe
        df = pd.DataFrame(table_data)

        # Download option
        markdown_content = _create_markdown_export(df, selected_module)

        st.download_button(
            label="Download as Markdown",
            data=markdown_content,
            file_name=f"component_tokens_{selected_module}.md",
            mime="text/markdown",
        )

        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.info("No components found with activations above the threshold.")


# ============================================================================
# Analysis Helper Functions
# ============================================================================


def _calculate_average_l0_scores(
    l0_scores_sum: defaultdict[str, float], l0_scores_count: int
) -> dict[str, float]:
    """Calculate average L0 scores from accumulated sums."""
    avg_l0_scores: dict[str, float] = {}
    if l0_scores_count > 0:
        for layer_name, score_sum in l0_scores_sum.items():
            avg_l0_scores[layer_name] = score_sum / l0_scores_count
    return avg_l0_scores


def _process_batch_for_tokens(
    *,
    batch: torch.Tensor,
    model_data: ModelData,
    component_token_activations: dict[str, dict[int, dict[int, int]]],
    component_token_ci_values: dict[str, dict[int, dict[int, list[float]]]],
    total_token_counts: dict[int, int],
    config: AnalysisConfig,
) -> tuple[int, dict[str, float]]:
    """Process a single batch for token activations.

    Returns:
        Tuple of (tokens_processed, ci_l_zero_scores)
    """
    # Count tokens in this batch
    tokens_processed = batch.numel()

    # Count all tokens in this batch
    for token_id in batch.flatten().tolist():
        total_token_counts[token_id] += 1

    # Get activations before each component
    with torch.no_grad():
        _, pre_weight_acts = model_data.model(
            batch,
            mode="pre_forward_cache",
            module_names=model_data.model.target_module_paths,
        )

        causal_importances, _ = model_data.model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type=model_data.config.sigmoid_type,
            detach_inputs=True,
        )

    # Calculate L0 scores for this batch
    ci_l_zero_vals: dict[str, float] = {}
    for module_name, ci in causal_importances.items():
        ci_l_zero_vals[module_name] = calc_ci_l_zero(ci, config.causal_importance_threshold)

    for module_name, ci in causal_importances.items():
        assert ci.ndim == 3, "CI must be 3D (batch, seq_len, C)"

        # Find active components
        active_mask = ci > config.causal_importance_threshold

        # Get token IDs for this batch
        token_ids = batch

        # For each component, track which tokens it activates on
        for component_idx in range(model_data.model.C):
            # Get positions where this component is active
            component_active = active_mask[:, :, component_idx]

            # Get the tokens at those positions
            active_tokens = token_ids[component_active]

            # Get the CI values at those positions
            active_ci_values = ci[:, :, component_idx][component_active]

            # Count occurrences and store CI values
            for token_id, ci_val in zip(
                active_tokens.tolist(), active_ci_values.tolist(), strict=True
            ):
                # Defaultdicts automatically handle nested initialization
                component_token_activations[module_name][component_idx][token_id] += 1
                component_token_ci_values[module_name][component_idx][token_id].append(ci_val)

    return tokens_processed, ci_l_zero_vals


def _prepare_component_table_data(
    *,
    module_activations: dict[int, dict[int, int]],
    module_ci_values: dict[int, dict[int, list[float]]],
    total_token_counts: dict[int, int],
    min_act_frequency: float,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Prepare table data for a module's components."""
    table_data: list[dict[str, Any]] = []

    for component_id in sorted(module_activations.keys()):
        component_token_counts = module_activations[component_id]
        if not component_token_counts:
            continue

        # Process tokens for this component
        component_ci_values = module_ci_values.get(component_id, {})
        token_data = _process_component_tokens(
            component_token_counts=component_token_counts,
            module_ci_values=component_ci_values,
            total_token_counts=total_token_counts,
            min_act_frequency=min_act_frequency,
            tokenizer=tokenizer,
        )

        if token_data:
            # Format tokens for display
            tokens_str = _format_token_display(token_data)
            table_data.append(
                {
                    "Component": component_id,
                    "Activating Tokens (mean_ci, count/total)": tokens_str,
                    "Total Unique Tokens": len(token_data),
                }
            )

    return table_data


def _defaultdict_to_dict(obj: Any) -> Any:
    """Recursively convert `defaultdict` instances to regular `dict`s.

    Streamlit's caching relies on pickle, which cannot serialize lambdas used
    as ``default_factory`` inside ``defaultdict``. To ensure the analysis
    results are picklable we convert any (nested) ``defaultdict`` structures
    to plain ``dict`` objects before returning them.
    """
    if isinstance(obj, defaultdict):
        # Convert defaultdict to dict but recurse into its values
        return {k: _defaultdict_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: _defaultdict_to_dict(v) for k, v in obj.items()}
    return obj


# ============================================================================
# Main Analysis Function
# ============================================================================


@st.cache_data(show_spinner="Analyzing component token activations across dataset...")
def analyze_component_token_table(
    _model_data: ModelData,
    config: AnalysisConfig,
) -> AnalysisResults:
    """Analyze which tokens activate each component across the dataset (with progress UI).

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

    assert isinstance(_model_data.config.task_config, LMTaskConfig)
    dataloader, _ = create_data_loader(
        dataset_config=data_config,
        batch_size=config.batch_size,
        buffer_size=_model_data.config.task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )

    # Initialize token activation tracking
    component_token_activations: dict[str, dict[int, dict[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    component_token_ci_values: dict[str, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    total_token_counts: dict[int, int] = defaultdict(int)
    l0_scores_sum: defaultdict[str, float] = defaultdict(float)
    l0_scores_count = 0

    total_tokens_processed = 0
    data_iter = iter(dataloader)

    for _ in range(config.n_steps):
        try:
            batch_data = next(data_iter)
        except StopIteration:
            break

        batch = extract_batch_data(batch_data).to(device)

        tokens_in_batch, ci_l_zero_vals = _process_batch_for_tokens(
            batch=batch,
            model_data=_model_data,
            component_token_activations=component_token_activations,
            component_token_ci_values=component_token_ci_values,
            total_token_counts=total_token_counts,
            config=config,
        )

        total_tokens_processed += tokens_in_batch

        for layer_name, layer_ci_l_zero in ci_l_zero_vals.items():
            l0_scores_sum[layer_name] += layer_ci_l_zero
        l0_scores_count += 1

    avg_l0_scores = _calculate_average_l0_scores(l0_scores_sum, l0_scores_count)

    # Convert defaultdicts (with lambda default factories) to dicts for pickling
    component_token_activations = _defaultdict_to_dict(component_token_activations)
    component_token_ci_values = _defaultdict_to_dict(component_token_ci_values)
    total_token_counts = _defaultdict_to_dict(total_token_counts)

    return AnalysisResults(
        component_token_activations=component_token_activations,
        component_token_ci_values=component_token_ci_values,
        total_tokens_processed=total_tokens_processed,
        total_token_counts=total_token_counts,
        avg_l0_scores=avg_l0_scores,
    )


# ============================================================================
# Main UI Function
# ============================================================================


@st.fragment
def render_component_token_table_tab(model_data: ModelData):
    """Render the component token table analysis."""
    st.subheader("Component Token Activation Analysis")
    st.markdown(
        "This analysis shows which tokens most frequently activate each component across a dataset. "
        "Higher causal importance values indicate stronger component activation."
    )

    # Configuration and run analysis
    config = _render_configuration_form()

    if config:
        # Run the analysis
        analysis_results = analyze_component_token_table(
            _model_data=model_data,
            config=config,
        )

        # Store results in session state
        st.session_state.token_activation_results = {
            "activations": analysis_results.component_token_activations,
            "ci_values": analysis_results.component_token_ci_values,
            "total_tokens": analysis_results.total_tokens_processed,
            "token_counts": analysis_results.total_token_counts,
            "l0_scores": analysis_results.avg_l0_scores,
            "min_act_frequency": config.min_act_frequency,
        }

    # Display results if available
    if "token_activation_results" in st.session_state:
        results: dict[str, Any] = st.session_state.token_activation_results
        # Since results come from AnalysisResults, all keys are guaranteed to exist
        activations: dict[str, dict[int, dict[int, int]]] = results["activations"]
        ci_values: dict[str, dict[int, dict[int, list[float]]]] = results["ci_values"]
        total_tokens: int = results["total_tokens"]
        total_token_counts: dict[int, int] = results["token_counts"]
        l0_scores: dict[str, float] = results["l0_scores"]
        min_act_frequency: float = results["min_act_frequency"]

        st.success(f"Analysis complete! Processed {total_tokens:,} tokens.")

        # Display L0 scores as summary metrics
        if l0_scores:
            _render_l0_scores(l0_scores)

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
                table_data = _prepare_component_table_data(
                    module_activations=module_activations,
                    module_ci_values=module_ci_values,
                    total_token_counts=total_token_counts,
                    min_act_frequency=min_act_frequency,
                    tokenizer=model_data.tokenizer,
                )

                # Render the table
                _render_token_table(
                    table_data=table_data,
                    selected_module=selected_module,
                )
