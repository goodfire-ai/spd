"""
Shared utilities for the Streamlit app.
"""

import os
import re
from dataclasses import dataclass
from typing import cast

import streamlit as st
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.configs import Config
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import EmbeddingComponents, GateMLPs, LinearComponents, VectorGateMLPs

DEFAULT_WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "spd")


@dataclass(frozen=True)
class ModelData:
    """Core model data that gets cached."""

    model: ComponentModel
    tokenizer: PreTrainedTokenizer
    config: Config
    gates: dict[str, GateMLPs | VectorGateMLPs]
    components: dict[str, LinearComponents | EmbeddingComponents]
    layer_names: list[str]


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
        return f"wandb:{DEFAULT_WANDB_PROJECT}/runs/{url_or_path}"

    return url_or_path


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: str) -> ModelData:
    """Load model and prepare components."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_info = SPDRunInfo.from_path(model_path)
    model = ComponentModel.from_run_info(run_info)
    config = run_info.config
    model.to(device)
    model.eval()

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    # Extract components and gates
    gates = {
        k.removeprefix("gates.").replace("-", "."): cast(GateMLPs | VectorGateMLPs, v)
        for k, v in model.gates.items()
    }
    components = {
        k.removeprefix("components.").replace("-", "."): cast(
            LinearComponents | EmbeddingComponents, v
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
        "- wandb:goodfire/spd/runs/snq4ojcy\n"
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
