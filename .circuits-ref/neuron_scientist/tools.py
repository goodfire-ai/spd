"""MCP Tools for neuron investigation experiments.

These tools are exposed via MCP and can be called by the Claude Agent SDK.
"""

import asyncio
import concurrent.futures
import json
import os
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# =============================================================================
# Default Paths
# =============================================================================

DEFAULT_LABELS_PATH = "data/neuron_labels_combined.json"
DEFAULT_NEURONDB_CSV_PATH = "data/neurondb_labels.csv"  # Pre-extracted NeuronDB export (99.9% coverage)

# =============================================================================
# Model Configuration System
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a specific model architecture."""
    name: str  # Human-readable name
    model_path: str  # HuggingFace model path
    num_layers: int  # Number of transformer layers
    neurons_per_layer: int  # MLP intermediate dimension
    chat_template: str  # Chat template format string with {prompt} placeholder
    activation_threshold: float = 0.5  # Threshold for "activating"
    trust_remote_code: bool = False  # Whether to trust remote code
    duckdb_path: str = ""  # Path to DuckDB atlas for this model
    labels_path: str = ""  # Path to JSON labels file for this model
    edge_stats_path: str = ""  # Path to edge stats / enriched profiles
    graphs_dir: str = ""  # Directory containing RelP graph JSON files

    def format_prompt(self, prompt: str) -> str:
        """Format a prompt with this model's chat template."""
        return self.chat_template.format(prompt=prompt)


# Predefined model configurations
LLAMA_3_1_8B_CONFIG = ModelConfig(
    name="Llama-3.1-8B-Instruct",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    num_layers=32,
    neurons_per_layer=14336,
    chat_template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    duckdb_path="data/llama8b_neurons.duckdb",
    labels_path="data/neuron_labels_combined.json",
    edge_stats_path="data/fineweb_50k_edge_stats_enriched.json",
    graphs_dir="graphs/fabric_fineweb_50k",
)

QWEN3_32B_CONFIG = ModelConfig(
    name="Qwen3-32B",
    model_path="Qwen/Qwen3-32B",
    num_layers=64,
    neurons_per_layer=25600,
    # Qwen3 uses a simpler template - raw prompts work well
    chat_template="{prompt}",
    trust_remote_code=True,
    duckdb_path="data/qwen32b_neurons.duckdb",
    graphs_dir="graphs/qwen3_32b_800k",
)

QWEN3_8B_CONFIG = ModelConfig(
    name="Qwen3-8B",
    model_path="Qwen/Qwen3-8B",
    num_layers=36,
    neurons_per_layer=12288,
    chat_template="{prompt}",
    trust_remote_code=True,
)

OLMO_3_7B_CONFIG = ModelConfig(
    name="OLMo-3-7B-Instruct",
    model_path="allenai/OLMo-3-7B-Instruct",
    num_layers=32,
    neurons_per_layer=11008,
    chat_template="{prompt}",  # Uses tokenizer's built-in apply_chat_template
    duckdb_path="data/olmo3_neurons.duckdb",
    labels_path="data/olmo3_enriched_labels.json",
    edge_stats_path="data/olmo3_wiring_cache/",
)

# Registry of available models
MODEL_CONFIGS = {
    "llama-3.1-8b": LLAMA_3_1_8B_CONFIG,
    "qwen3-32b": QWEN3_32B_CONFIG,
    "qwen3-8b": QWEN3_8B_CONFIG,
    "olmo-3-7b": OLMO_3_7B_CONFIG,
    "olmo-3-7b-instruct": OLMO_3_7B_CONFIG,
}

# Current model configuration (can be set before loading)
_MODEL_CONFIG: ModelConfig | None = None


def set_model_config(config: ModelConfig | str) -> ModelConfig:
    """Set the model configuration to use.

    Args:
        config: Either a ModelConfig object or a string key from MODEL_CONFIGS

    Returns:
        The ModelConfig that was set
    """
    global _MODEL_CONFIG, _MODEL, _TOKENIZER

    if isinstance(config, str):
        if config not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {config}. Available: {list(MODEL_CONFIGS.keys())}")
        config = MODEL_CONFIGS[config]

    # If changing model, clear the cached model
    if _MODEL_CONFIG is not None and _MODEL_CONFIG.model_path != config.model_path:
        _MODEL = None
        _TOKENIZER = None

    _MODEL_CONFIG = config
    print(f"Model config set to: {config.name}")
    return config


def get_model_config() -> ModelConfig:
    """Get the current model configuration, defaulting to Llama if not set."""
    global _MODEL_CONFIG
    if _MODEL_CONFIG is None:
        _MODEL_CONFIG = LLAMA_3_1_8B_CONFIG
    return _MODEL_CONFIG

# =============================================================================
# Reproducibility: Seed Management
# =============================================================================

_GLOBAL_SEED = None


def set_seed(seed: int = 42) -> int:
    """Set seeds for reproducibility across all random sources.

    Args:
        seed: The seed value to use

    Returns:
        The seed that was set
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For full determinism (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def get_seed() -> int | None:
    """Get the current global seed, or None if not set."""
    return _GLOBAL_SEED


# =============================================================================
# Protocol State: Track completion of validation steps
# =============================================================================


@dataclass
class ProtocolState:
    """Track completion of validation steps during investigation.

    This is used to gate report saving and auto-downgrade confidence
    when validation is incomplete.
    """
    # Phase 0: Corpus context
    phase0_corpus_queried: bool = False
    phase0_graph_count: int = 0
    phase0_sample_prompts: list = field(default_factory=list)

    # Baseline validation
    baseline_comparison_done: bool = False
    baseline_zscore: float | None = None
    baseline_prompts_tested: int = 0

    # Dose-response validation
    dose_response_done: bool = False
    dose_response_monotonic: bool = False
    dose_response_kendall_tau: float | None = None
    dose_response_results: list = field(default_factory=list)  # Full dose-response data

    # Category selectivity validation (REQUIRED for selectivity claims)
    category_selectivity_done: bool = False
    category_selectivity_zscore_gap: float | None = None  # Target z-mean minus control z-mean
    category_selectivity_n_categories: int = 0

    # RelP validation
    relp_runs: int = 0
    relp_positive_control: bool = False  # Neuron found in activating context
    relp_negative_control: bool = False  # Neuron NOT found in non-activating context

    # Hypothesis tracking
    hypotheses_registered: int = 0
    hypotheses_updated: int = 0

    # =============================================================================
    # V4 Phase Tracking
    # =============================================================================

    # Phase completion status
    input_phase_complete: bool = False
    output_phase_complete: bool = False
    anomaly_phase_complete: bool = False

    # Anomaly phase tracking
    anomalies_identified: list[str] = field(default_factory=list)
    anomalies_investigated: list[dict[str, Any]] = field(default_factory=list)

    # Input phase tool completion tracking
    upstream_dependency_tested: bool = False  # Ablation test done
    upstream_steering_tested: bool = False    # Steering test done (provides RelP comparison)
    upstream_neurons_exist: bool = False  # Set when upstream neurons are found in prior knowledge

    # Output phase tool completion tracking
    multi_token_ablation_done: bool = False
    batch_ablation_done: bool = False  # REQUIRED: batch_ablate_and_generate called
    batch_steering_done: bool = False  # REQUIRED: batch_steer_and_generate called (with downstream slope computation)
    batch_ablation_prompt_count: int = 0  # Track how many prompts used
    batch_steering_prompt_count: int = 0  # Track how many prompts used
    intelligent_steering_runs: int = 0  # REQUIRED ≥1: intelligent_steering_analysis calls
    intelligent_steering_total_prompts: int = 0  # Total prompts across all runs
    downstream_dependency_tested: bool = False
    downstream_dependency_prompt_count: int = 0  # Track how many prompts used
    downstream_neurons_exist: bool = False  # Set by analyze_output_wiring when downstream neurons are found

    # Phase characterization storage (populated by complete_*_phase tools)
    input_characterization: dict[str, Any] = field(default_factory=lambda: {
        "summary": "",
        "triggers": [],
        "upstream_dependencies": [],
        "selectivity_data": {},
        "confidence": 0.0,
    })
    output_characterization: dict[str, Any] = field(default_factory=lambda: {
        "summary": "",
        "promotes": [],
        "suppresses": [],
        "multi_token_ablation": [],
        "multi_token_steering": [],
        "downstream_dependencies": [],
        "relp_evidence": [],
        "confidence": 0.0,
    })

    # V4 Results storage (for persisting to investigation JSON)
    multi_token_ablation_results: list[dict[str, Any]] = field(default_factory=list)
    multi_token_steering_results: list[dict[str, Any]] = field(default_factory=list)
    upstream_dependency_results: list[dict[str, Any]] = field(default_factory=list)
    upstream_steering_results: list[dict[str, Any]] = field(default_factory=list)  # V5: batch_steer_upstream_and_test
    downstream_dependency_results: list[dict[str, Any]] = field(default_factory=list)

    # Categorized prompts from category selectivity (for batch ablation/steering)
    categorized_prompts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Connectivity analysis results (for batch_ablate_upstream_and_test)
    connectivity_analyzed: bool = False
    connectivity_data: dict[str, Any] | None = None

    # RelP verification results (accumulated across all batch_relp_verify_connections calls)
    # Keyed by neuron_id → {"relp_confirmed": bool, "relp_strength": float}
    relp_verification_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Wiring analysis (REQUIRED EARLY STEP - weight-based connectivity with polarity)
    wiring_analysis_done: bool = False
    wiring_excitatory_count: int = 0
    wiring_inhibitory_count: int = 0
    wiring_data: dict[str, Any] | None = None

    # Output wiring analysis (weight-based downstream connectivity)
    output_wiring_done: bool = False
    output_wiring_excitatory_count: int = 0
    output_wiring_inhibitory_count: int = 0
    output_wiring_data: dict[str, Any] | None = None

    # SwiGLU operating regime (detected during category selectivity)
    operating_regime: str | None = None        # "standard", "inverted", "mixed"
    regime_confidence: float = 0.0                # fraction of activations in dominant regime
    regime_data: dict[str, Any] | None = None  # full decomposition stats
    firing_sign_stats: dict[str, Any] | None = None  # {positive_pct, negative_pct, ...}

    # Target neuron info (set at init)
    target_layer: int = 0
    target_neuron: int = 0

    def get_input_phase_missing(self) -> list[str]:
        """Return list of missing input phase requirements."""
        missing = []
        # Wiring analysis is REQUIRED FIRST - provides upstream polarity predictions
        if not self.wiring_analysis_done:
            missing.append("Wiring analysis not run (REQUIRED EARLY STEP). Call analyze_wiring() first!")
        if not self.category_selectivity_done:
            missing.append("Category selectivity test not run (REQUIRED). Call run_category_selectivity_test()")
        if self.hypotheses_registered == 0:
            missing.append("No hypotheses registered (need at least 1). Call register_hypothesis()")
        # Phase 0 corpus query is optional - agent can query during investigation if needed
        # Upstream dependency testing is REQUIRED when upstream neurons exist
        if self.upstream_neurons_exist and not self.upstream_dependency_tested:
            missing.append("Upstream ablation not tested (REQUIRED when upstream neurons exist). Call batch_ablate_upstream_and_test()")
        if self.upstream_neurons_exist and not self.upstream_steering_tested:
            missing.append("Upstream steering not tested (REQUIRED when upstream neurons exist). Call batch_steer_upstream_and_test()")
        return missing

    def get_output_phase_missing(self) -> list[str]:
        """Return list of missing output phase requirements."""
        missing = []
        if not self.input_phase_complete:
            missing.append("Input phase not complete (BLOCKED)")
        if not self.batch_ablation_done:
            missing.append("Batch ablation not run (REQUIRED). Call batch_ablate_and_generate(use_categorized_prompts=True)")
        elif self.batch_ablation_prompt_count < 50:
            missing.append(f"Batch ablation used only {self.batch_ablation_prompt_count} prompts (need ≥50). Call batch_ablate_and_generate(use_categorized_prompts=True)")
        if self.intelligent_steering_runs < 1:
            missing.append("Intelligent steering not run (REQUIRED ≥1). Call intelligent_steering_analysis()")
        if not self.batch_steering_done:
            missing.append("Batch steering not run (REQUIRED — computes downstream slopes). Call batch_steer_and_generate(use_categorized_prompts=True)")
        if self.downstream_neurons_exist and not self.downstream_dependency_tested:
            missing.append("Downstream dependency not tested (REQUIRED when downstream neurons exist). Call ablate_and_check_downstream()")
        elif self.downstream_neurons_exist and self.downstream_dependency_prompt_count < 20:
            missing.append(f"Downstream dependency used only {self.downstream_dependency_prompt_count} prompts (need ≥20). Call ablate_and_check_downstream with more prompts.")
        return missing

    def can_start_output_phase(self) -> bool:
        """Check if output phase can be started (input phase must be complete)."""
        return self.input_phase_complete

    def get_missing_validation(self) -> list:
        """Return list of missing validation steps."""
        missing = []
        # Phase 0 corpus query is optional - removed from required validation
        if not self.baseline_comparison_done:
            missing.append("Baseline: Did not run baseline comparison with random neurons")
        elif self.baseline_zscore is not None and self.baseline_zscore < 2.0:
            missing.append(f"Baseline: Z-score ({self.baseline_zscore:.2f}) below threshold (2.0)")
        if not self.dose_response_done:
            missing.append("Dose-response: Did not run steering dose-response test")
        elif not self.dose_response_monotonic:
            missing.append("Dose-response: Effect is not monotonic (weak causality)")
        if self.relp_runs == 0:
            missing.append("RelP: Did not run any RelP attribution")
        if not self.relp_positive_control:
            missing.append("RelP: No positive control (neuron not found in any graph)")
        if not self.relp_negative_control:
            missing.append("RelP: No negative control (should verify neuron absent in non-activating)")
        if not self.category_selectivity_done:
            missing.append("Category Selectivity: Did not run category selectivity test")
        elif self.category_selectivity_zscore_gap is not None and self.category_selectivity_zscore_gap < 1.0:
            missing.append(f"Category Selectivity: Z-score gap ({self.category_selectivity_zscore_gap:.2f}) below threshold (1.0) - weak selectivity")
        if self.hypotheses_registered == 0:
            missing.append("Pre-registration: Did not register any hypotheses before testing")
        return missing

    def compute_evidence_confidence(self) -> float:
        """Compute confidence score based on validation evidence.

        Returns:
            Float between 0 and 1 representing evidence-based confidence.
        """
        points = 0
        max_points = 100

        # Baseline z-score (0-30 points)
        if self.baseline_comparison_done and self.baseline_zscore is not None:
            if self.baseline_zscore >= 3.0:
                points += 30
            elif self.baseline_zscore >= 2.5:
                points += 25
            elif self.baseline_zscore >= 2.0:
                points += 20
            elif self.baseline_zscore >= 1.5:
                points += 10
            # Below 1.5 = 0 points

        # Dose-response validation (0-20 points)
        if self.dose_response_done:
            if self.dose_response_monotonic:
                if self.dose_response_kendall_tau is not None:
                    if self.dose_response_kendall_tau >= 0.7:
                        points += 20  # Strong monotonic
                    elif self.dose_response_kendall_tau >= 0.4:
                        points += 15  # Moderate monotonic
                    else:
                        points += 5   # Weak
                else:
                    points += 15  # Monotonic but no tau
            else:
                points += 5  # Ran but not monotonic

        # RelP validation (0-20 points)
        if self.relp_runs >= 3:
            points += 10
        elif self.relp_runs >= 1:
            points += 5
        if self.relp_positive_control:
            points += 5
        if self.relp_negative_control:
            points += 5

        # Hypothesis pre-registration (0-15 points)
        if self.hypotheses_registered >= 1 and self.hypotheses_updated >= 1:
            points += 15
        elif self.hypotheses_registered >= 1:
            points += 10

        # Phase 0 corpus context (0-15 points)
        if self.phase0_corpus_queried:
            if self.phase0_graph_count >= 10:
                points += 15
            elif self.phase0_graph_count >= 1:
                points += 10
            else:
                points += 5  # Queried but no graphs found

        # Category selectivity validation (0-15 points) - REQUIRED for selectivity claims
        if self.category_selectivity_done:
            if self.category_selectivity_zscore_gap is not None:
                if self.category_selectivity_zscore_gap >= 2.0:
                    points += 15  # Highly selective
                elif self.category_selectivity_zscore_gap >= 1.0:
                    points += 10  # Moderately selective
                elif self.category_selectivity_zscore_gap >= 0.5:
                    points += 5   # Weakly selective
                # Below 0.5 = 0 points
            else:
                points += 5  # Ran test but no gap computed

        # Max points = 115 (baseline 30 + dose 20 + relp 20 + hyp 15 + corpus 15 + selectivity 15)
        max_points = 115
        return min(points / max_points, 1.0)


# Global protocol state - reset per investigation
_PROTOCOL_STATE: ProtocolState | None = None


def init_protocol_state() -> ProtocolState:
    """Initialize/reset the protocol state for a new investigation."""
    global _PROTOCOL_STATE
    _PROTOCOL_STATE = ProtocolState()
    return _PROTOCOL_STATE


def get_protocol_state() -> ProtocolState:
    """Get the current protocol state, initializing if needed."""
    global _PROTOCOL_STATE
    if _PROTOCOL_STATE is None:
        _PROTOCOL_STATE = ProtocolState()
    return _PROTOCOL_STATE


def _apply_regime_correction(state: ProtocolState) -> None:
    """Apply retroactive polarity correction when operating regime is first detected.

    If the neuron operates in "inverted" regime (gate_pre < 0, up_pre < 0),
    all wiring polarity predictions need to be flipped because the standard
    polarity logic assumes gate_pre > 0, up_pre > 0.

    Called after regime detection during category selectivity.
    """
    regime = state.operating_regime
    if not regime:
        return

    if regime == "inverted" and state.wiring_data:
        stats = state.wiring_data.get("stats", {})

        # Don't double-flip
        if stats.get("regime_correction_applied"):
            return

        print("[PROTOCOL] Applying regime correction: inverted regime detected, flipping all upstream polarities")

        # Flip polarity labels and negate effective_strength in wiring_data
        for key in ("top_excitatory", "all_excitatory", "top_inhibitory", "all_inhibitory"):
            for conn in state.wiring_data.get(key, []):
                old_polarity = conn.get("predicted_polarity")
                if old_polarity == "excitatory":
                    conn["predicted_polarity"] = "inhibitory"
                    conn["effective_strength"] = -abs(conn.get("effective_strength", 0))
                elif old_polarity == "inhibitory":
                    conn["predicted_polarity"] = "excitatory"
                    conn["effective_strength"] = abs(conn.get("effective_strength", 0))
                # mixed stays mixed

        # Swap the excitatory/inhibitory lists
        old_exc = state.wiring_data.get("top_excitatory", [])
        old_inh = state.wiring_data.get("top_inhibitory", [])
        state.wiring_data["top_excitatory"] = old_inh
        state.wiring_data["top_inhibitory"] = old_exc

        old_all_exc = state.wiring_data.get("all_excitatory", [])
        old_all_inh = state.wiring_data.get("all_inhibitory", [])
        state.wiring_data["all_excitatory"] = old_all_inh
        state.wiring_data["all_inhibitory"] = old_all_exc

        # Update stats counts
        old_exc_count = stats.get("excitatory_count", 0)
        old_inh_count = stats.get("inhibitory_count", 0)
        stats["excitatory_count"] = old_inh_count
        stats["inhibitory_count"] = old_exc_count

        # Mark correction as applied
        stats["regime_correction_applied"] = True
        stats["target_regime"] = regime

        # Also fix connectivity_data upstream_neurons
        if state.connectivity_data:
            for u in state.connectivity_data.get("upstream_neurons", []):
                old_pol = u.get("polarity")
                if old_pol == "excitatory":
                    u["polarity"] = "inhibitory"
                    u["weight"] = -abs(u.get("weight", 0))
                    u["effective_strength"] = -abs(u.get("effective_strength", 0))
                elif old_pol == "inhibitory":
                    u["polarity"] = "excitatory"
                    u["weight"] = abs(u.get("weight", 0))
                    u["effective_strength"] = abs(u.get("effective_strength", 0))

        # Regenerate analysis summary
        neuron_id = state.wiring_data.get("neuron_id", f"L{state.target_layer}/N{state.target_neuron}")
        new_exc = state.wiring_data.get("top_excitatory", [])
        new_inh = state.wiring_data.get("top_inhibitory", [])
        exc_labels = [c.get("label", "") for c in new_exc[:20] if c.get("label")]
        inh_labels = [c.get("label", "") for c in new_inh[:20] if c.get("label")]
        coverage = state.wiring_data.get("label_coverage_pct", 0)

        state.wiring_data["analysis_summary"] = f"""## Wiring Analysis Summary for {neuron_id} (REGIME-CORRECTED)

**Operating Regime: INVERTED** (gate_pre < 0, up_pre < 0 at max activation)
Polarity labels have been flipped from the original weight-based predictions.

**Statistics:**
- Analyzed {stats.get('total_upstream_neurons', 0):,} upstream neurons
- {stats['excitatory_count']:,} excitatory, {stats['inhibitory_count']:,} inhibitory (after correction)
- Label coverage: {coverage:.0f}%

**Top Excitatory Themes** (neurons that INCREASE target activation):
{chr(10).join(f'- {l[:100]}' for l in exc_labels[:10])}

**Top Inhibitory Themes** (neurons that DECREASE target activation):
{chr(10).join(f'- {l[:100]}' for l in inh_labels[:10])}

**Questions to Consider:**
1. Do the excitatory inputs share a semantic theme? What would activate them together?
2. Do the inhibitory inputs represent contexts where this neuron should NOT fire?
3. Does this suggest the neuron's role is broader/narrower than the initial label?
4. Compare with RelP connectivity later - which of these actually influence in context?
"""

        print(f"[PROTOCOL] Regime correction applied: swapped {old_exc_count} excitatory <-> {old_inh_count} inhibitory")

    elif regime == "mixed" and state.wiring_data:
        stats = state.wiring_data.get("stats", {})
        if not stats.get("regime_warning"):
            stats["regime_warning"] = "mixed regime — polarity predictions unreliable"
            print("[PROTOCOL] Mixed regime detected — added polarity warning to wiring stats")


def update_protocol_state(**kwargs) -> ProtocolState:
    """Update protocol state with new values."""
    state = get_protocol_state()
    for key, value in kwargs.items():
        if hasattr(state, key):
            setattr(state, key, value)
        else:
            print(f"[PROTOCOL] Warning: Unknown state key '{key}'")
    return state


def restore_protocol_state_from_investigation(protocol_validation: dict) -> ProtocolState:
    """Restore protocol state from a prior investigation's protocol_validation dict.

    This is used during revision runs to preserve phase completion state.

    Args:
        protocol_validation: The protocol_validation dict from a prior investigation JSON

    Returns:
        The restored ProtocolState
    """
    global _PROTOCOL_STATE
    _PROTOCOL_STATE = ProtocolState()

    # Map protocol_validation fields to ProtocolState attributes
    field_mapping = {
        "phase0_corpus_queried": "phase0_corpus_queried",
        "phase0_graph_count": "phase0_graph_count",
        "baseline_comparison_done": "baseline_comparison_done",
        "baseline_zscore": "baseline_zscore",
        "category_selectivity_done": "category_selectivity_done",
        "category_selectivity_zscore_gap": "category_selectivity_zscore_gap",
        "category_selectivity_n_categories": "category_selectivity_n_categories",
        "dose_response_done": "dose_response_done",
        "dose_response_monotonic": "dose_response_monotonic",
        "dose_response_kendall_tau": "dose_response_kendall_tau",
        "relp_runs": "relp_runs",
        "relp_positive_control": "relp_positive_control",
        "relp_negative_control": "relp_negative_control",
        "hypotheses_registered": "hypotheses_registered",
        "hypotheses_updated": "hypotheses_updated",
        "input_phase_complete": "input_phase_complete",
        "output_phase_complete": "output_phase_complete",
        "upstream_dependency_tested": "upstream_dependency_tested",
        "upstream_steering_tested": "upstream_steering_tested",
        "upstream_neurons_exist": "upstream_neurons_exist",
        "batch_ablation_done": "batch_ablation_done",
        "multi_token_ablation_done": "multi_token_ablation_done",
        "downstream_dependency_tested": "downstream_dependency_tested",
        "downstream_neurons_exist": "downstream_neurons_exist",
        "wiring_analysis_done": "wiring_analysis_done",
        "output_wiring_done": "output_wiring_done",
        "operating_regime": "operating_regime",
        "regime_confidence": "regime_confidence",
    }

    for json_key, attr_name in field_mapping.items():
        if json_key in protocol_validation:
            value = protocol_validation[json_key]
            if hasattr(_PROTOCOL_STATE, attr_name):
                setattr(_PROTOCOL_STATE, attr_name, value)

    print("[PROTOCOL] Restored state from prior investigation:")
    print(f"[PROTOCOL]   - input_phase_complete: {_PROTOCOL_STATE.input_phase_complete}")
    print(f"[PROTOCOL]   - output_phase_complete: {_PROTOCOL_STATE.output_phase_complete}")
    print(f"[PROTOCOL]   - category_selectivity_done: {_PROTOCOL_STATE.category_selectivity_done}")

    return _PROTOCOL_STATE


# =============================================================================
# NeuronDB Integration: Label Lookup with Fallback
# =============================================================================

# Cache for NeuronDB connection
_NEURONDB_MANAGER = None
_NEURONDB_AVAILABLE = None  # None = not checked, True/False = checked


def _get_neurondb():
    """Get NeuronDB manager instance, or None if unavailable.

    Lazily initializes connection and caches availability status.
    """
    global _NEURONDB_MANAGER, _NEURONDB_AVAILABLE

    if _NEURONDB_AVAILABLE is False:
        return None

    if _NEURONDB_MANAGER is not None:
        return _NEURONDB_MANAGER

    try:
        import os
        import sys

        # NeuronDB is in observatory_repo - add to path without changing directory
        observatory_path = Path(__file__).parent.parent / "observatory_repo"
        if not observatory_path.exists():
            _NEURONDB_AVAILABLE = False
            return None

        # Add observatory_repo to path if not already there
        observatory_str = str(observatory_path.absolute())
        if observatory_str not in sys.path:
            sys.path.insert(0, observatory_str)

        # Load .env file from observatory_repo (without changing directory)
        # Use override=True AND manually set env vars to handle module caching issues
        env_file = observatory_path / ".env"
        if env_file.exists():
            # Manually parse and set env vars first (handles cached ENV singletons)
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value

            from dotenv import load_dotenv
            load_dotenv(env_file, override=True)

        # Clear any cached neurondb modules to pick up new env vars
        modules_to_remove = [m for m in sys.modules if m.startswith(('neurondb', 'util.env'))]
        for m in modules_to_remove:
            del sys.modules[m]

        from neurondb.postgres import DBManager

        # Clear any stale instances
        DBManager.instances.clear()

        _NEURONDB_MANAGER = DBManager.get_instance()
        _NEURONDB_AVAILABLE = True

        print("[NeuronDB] Connected successfully (458K neuron labels available)")
        return _NEURONDB_MANAGER

    except Exception as e:
        _NEURONDB_AVAILABLE = False
        print(f"[NeuronDB] Not available: {e}")
        import traceback
        traceback.print_exc()
        return None


def _query_neurondb_labels(neuron_ids: list[str]) -> dict[str, str]:
    """Query NeuronDB for labels of multiple neurons.

    Args:
        neuron_ids: List of neuron IDs in format "L{layer}/N{neuron}"

    Returns:
        Dict mapping neuron_id -> description (empty string if not found)
    """
    db = _get_neurondb()
    if db is None:
        return {}

    try:
        from neurondb.schemas import SQLANeuron, SQLANeuronDescription
        from sqlalchemy import and_

        results = {}

        # Parse neuron IDs into (layer, neuron) tuples
        layer_neuron_pairs = []
        for nid in neuron_ids:
            try:
                # Parse "L15/N6874" format
                parts = nid.replace("L", "").replace("N", "").split("/")
                if len(parts) == 2:
                    layer_neuron_pairs.append((int(parts[0]), int(parts[1]), nid))
            except (ValueError, IndexError):
                continue

        # Query NeuronDB for each neuron
        for layer, neuron_idx, nid in layer_neuron_pairs:
            try:
                rows = db.get(
                    [SQLANeuron.layer, SQLANeuron.neuron, SQLANeuronDescription.description],
                    joins=[(SQLANeuronDescription, SQLANeuron.id == SQLANeuronDescription.neuron_id)],
                    filter=and_(SQLANeuron.layer == layer, SQLANeuron.neuron == neuron_idx)
                )
                if rows:
                    _, _, desc = rows[0]
                    results[nid] = desc
            except Exception:
                pass  # Skip individual failures

        return results

    except Exception as e:
        print(f"[NeuronDB] Query error: {e}")
        return {}


# =============================================================================
# CSV-Based NeuronDB Labels (Pre-Extracted)
# =============================================================================

_NEURONDB_CSV_CACHE: dict[str, str] | None = None


def _load_neurondb_csv(csv_path: str = DEFAULT_NEURONDB_CSV_PATH) -> dict[str, str]:
    """Load pre-extracted NeuronDB labels from CSV.

    CSV format: layer,neuron,description
    Returns: Dict mapping "L{layer}/N{neuron}" -> description

    This is much faster than live NeuronDB queries and doesn't require
    the PostgreSQL server to be running.
    """
    global _NEURONDB_CSV_CACHE

    if _NEURONDB_CSV_CACHE is not None:
        return _NEURONDB_CSV_CACHE

    try:
        import csv
        labels = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = int(row['layer'])
                neuron = int(row['neuron'])
                desc = row['description']
                neuron_id = f"L{layer}/N{neuron}"
                # Keep first description per neuron (highest polarity score)
                if neuron_id not in labels:
                    labels[neuron_id] = desc

        _NEURONDB_CSV_CACHE = labels
        print(f"[NeuronDB CSV] Loaded {len(labels)} labels from {csv_path}")
        return labels

    except FileNotFoundError:
        print(f"[NeuronDB CSV] File not found: {csv_path}")
        _NEURONDB_CSV_CACHE = {}
        return {}
    except Exception as e:
        print(f"[NeuronDB CSV] Error loading: {e}")
        _NEURONDB_CSV_CACHE = {}
        return {}


def _query_neurondb_csv(neuron_ids: list[str]) -> dict[str, str]:
    """Query pre-extracted NeuronDB labels from CSV.

    Args:
        neuron_ids: List of neuron IDs in format "L{layer}/N{neuron}"

    Returns:
        Dict mapping neuron_id -> description
    """
    csv_labels = _load_neurondb_csv()
    return {nid: csv_labels[nid] for nid in neuron_ids if nid in csv_labels}


def get_neuron_label_with_fallback(
    neuron_id: str,
    labels_data: dict | None = None,
    labels_path: str = DEFAULT_LABELS_PATH,
) -> dict[str, Any]:
    """Get neuron label, using CSV as primary source.

    Priority order:
    1. CSV labels (pre-extracted NeuronDB, 99.9% coverage, fast)
    2. JSON labels file (custom/additional labels)

    Args:
        neuron_id: Neuron ID in format "L{layer}/N{neuron}"
        labels_data: Pre-loaded labels dict (optional, avoids re-reading file)
        labels_path: Path to labels JSON file

    Returns:
        Dict with label info. Always includes 'neuron_id' and 'found'.
        If found, includes 'label' (short description) and optionally
        'function_label', 'input_label', 'source'.
    """
    # Primary source: CSV labels (fast, 99.9% coverage)
    csv_labels = _query_neurondb_csv([neuron_id])
    if neuron_id in csv_labels and csv_labels[neuron_id]:
        return {
            "neuron_id": neuron_id,
            "found": True,
            "source": "csv",
            "label": csv_labels[neuron_id],
            "function_label": "",
            "input_label": "",
        }

    # Fallback: JSON labels file (for custom/additional labels)
    if labels_data is None:
        try:
            with open(labels_path) as f:
                labels_data = json.load(f)
        except FileNotFoundError:
            labels_data = {"neurons": {}}

    neurons = labels_data.get("neurons", {})

    if neuron_id in neurons:
        n = neurons[neuron_id]
        # Skip uninformative labels
        skip_labels = {"uninterpretable-routing", "uninterpretable", "unknown", ""}
        func_label = n.get("function_label", "")
        input_label = n.get("input_label", "")
        # Use function_label unless it's uninformative, then use input_label
        best_label = func_label if func_label.lower() not in skip_labels else input_label
        if best_label.lower() not in skip_labels:
            return {
                "neuron_id": neuron_id,
                "found": True,
                "source": "json",
                "label": best_label,
                "function_label": func_label,
                "function_description": n.get("function_description", ""),
                "input_label": input_label,
                "input_description": n.get("input_description", ""),
                "interpretability": n.get("interpretability", ""),
            }

    return {
        "neuron_id": neuron_id,
        "found": False,
        "source": None,
        "label": "",
    }


def _query_duckdb_labels(neuron_ids: list[str], db_path: str = "") -> dict[str, str]:
    """Query neuron labels from DuckDB atlas for the current model.

    Args:
        neuron_ids: List of neuron IDs in format "L{layer}/N{neuron}"
        db_path: Path to DuckDB database (auto-detected from model config if empty)

    Returns:
        Dict mapping neuron_id -> label string
    """
    if not neuron_ids:
        return {}
    if not db_path:
        db_path = get_model_config().duckdb_path
    if not db_path:
        return {}
    try:
        import duckdb
        db = duckdb.connect(db_path, read_only=True)
        # Parse neuron IDs into (layer, neuron) tuples
        pairs = []
        for nid in neuron_ids:
            parts = nid.replace("L", "").replace("N", "").split("/")
            if len(parts) == 2:
                pairs.append((int(parts[0]), int(parts[1])))
        if not pairs:
            return {}
        # Batch query
        results = {}
        for layer, neuron in pairs:
            row = db.execute(
                "SELECT label FROM neurons WHERE layer=? AND neuron=?",
                [layer, neuron],
            ).fetchone()
            if row and row[0]:
                results[f"L{layer}/N{neuron}"] = row[0]
        db.close()
        return results
    except Exception:
        # DuckDB not available or query failed — not fatal
        return {}


def batch_get_neuron_labels_with_fallback(
    neuron_ids: list[str],
    labels_path: str = "",
) -> dict[str, dict[str, Any]]:
    """Get labels for multiple neurons with multi-source fallback.

    Uses the current model config to determine the right data sources.

    Priority order:
    1. DuckDB atlas (auto-detected from model config, fast)
    2. CSV labels (pre-extracted NeuronDB, Llama-3.1-8B only)
    2. DuckDB atlas (Qwen3-32B autointerp labels, 1.6M neurons)
    3. JSON labels file (custom/additional labels)

    Args:
        neuron_ids: List of neuron IDs
        labels_path: Path to labels JSON file

    Returns:
        Dict mapping neuron_id -> label info dict
    """
    model_config = get_model_config()
    if not labels_path:
        labels_path = model_config.labels_path or DEFAULT_LABELS_PATH

    results = {}
    missing_ids = list(neuron_ids)

    # Primary source: DuckDB atlas (auto-detected from model config)
    if missing_ids:
        duckdb_labels = _query_duckdb_labels(missing_ids)
        still_missing = []
        for nid in missing_ids:
            if nid in duckdb_labels and duckdb_labels[nid]:
                results[nid] = {
                    "found": True,
                    "source": "duckdb",
                    "label": duckdb_labels[nid],
                    "function_label": "",
                    "input_label": "",
                }
            else:
                still_missing.append(nid)
        missing_ids = still_missing

    # Fallback 2: CSV labels (NeuronDB, Llama-3.1-8B only)
    if missing_ids:
        csv_labels = _query_neurondb_csv(missing_ids)
        still_missing = []
        for nid in missing_ids:
            if nid in csv_labels and csv_labels[nid]:
                results[nid] = {
                    "found": True,
                    "source": "csv",
                    "label": csv_labels[nid],
                    "function_label": "",
                    "input_label": "",
                }
            else:
                still_missing.append(nid)
        missing_ids = still_missing

    # Fallback 3: JSON labels file (for custom/additional labels)
    if missing_ids:
        try:
            with open(labels_path) as f:
                labels_data = json.load(f)
        except FileNotFoundError:
            labels_data = {"neurons": {}}

        neurons = labels_data.get("neurons", {})
        skip_labels = {"uninterpretable-routing", "uninterpretable", "unknown", ""}

        for nid in missing_ids:
            if nid in neurons:
                n = neurons[nid]
                func_label = n.get("function_label", "")
                input_label = n.get("input_label", "")
                best_label = func_label if func_label.lower() not in skip_labels else input_label
                if best_label.lower() not in skip_labels:
                    results[nid] = {
                        "found": True,
                        "source": "json",
                        "label": best_label,
                        "function_label": func_label,
                        "input_label": input_label,
                    }
                    continue
            # Not found in any source
            results[nid] = {
                "found": False,
                "source": None,
                "label": "",
            }

    return results


# Global model/tokenizer - loaded once per worker
_MODEL = None
_TOKENIZER = None

# Lock to serialize CUDA model operations across concurrent async tasks
# This prevents "double free or corruption" errors when multiple agents
# run experiments on the same GPU simultaneously.
# NOTE: Using RLock (reentrant lock) because some functions call get_model_and_tokenizer()
# while already holding the lock.
_MODEL_LOCK = threading.RLock()

# Dedicated single-thread executor for CUDA operations.
# CRITICAL: CUDA binds GPU context to the thread that first touches it.
# All CUDA operations must run on the same thread to avoid hangs.
# This executor ensures model loading AND inference happen on one thread.
_CUDA_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Remote GPU client (set when using GPU server mode)
_GPU_CLIENT: Optional["GPUClient"] = None


def set_gpu_client(client):
    """Set the GPU client for remote dispatch."""
    global _GPU_CLIENT
    _GPU_CLIENT = client


def get_gpu_client():
    """Get the GPU client (None if running locally)."""
    return _GPU_CLIENT


# =============================================================================
# Batching Configuration
# =============================================================================
# These control GPU memory usage vs throughput tradeoffs for batched operations
MAX_BATCH_SIZE = 16  # Max prompts per batch (tune based on prompt lengths)
MAX_BATCH_TOKENS = 4096  # Cap total tokens per batch to avoid OOM


def get_model_and_tokenizer():
    """Get or load the model and tokenizer.

    Thread-safe: Uses _MODEL_LOCK to prevent race conditions during loading.
    Uses the current ModelConfig (set via set_model_config()).
    """
    global _MODEL, _TOKENIZER
    with _MODEL_LOCK:
        if _MODEL is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            config = get_model_config()
            print(f"Loading model: {config.name} ({config.model_path})...")
            _MODEL = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=config.trust_remote_code,
            )
            _TOKENIZER = AutoTokenizer.from_pretrained(
                config.model_path,
                trust_remote_code=config.trust_remote_code,
            )
            # Set pad_token for batch processing (required for padding=True)
            if _TOKENIZER.pad_token is None:
                _TOKENIZER.pad_token = _TOKENIZER.eos_token
            print(f"Model loaded: {config.num_layers} layers, {config.neurons_per_layer} neurons/layer")
    return _MODEL, _TOKENIZER


# Legacy chat template (kept for backwards compatibility)
CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def format_prompt(prompt: str) -> str:
    """Format a prompt with the current model's chat template."""
    config = get_model_config()
    return config.format_prompt(prompt)


def format_steering_prompt(prompt_info: dict) -> str:
    """Format a prompt for steering experiments based on its format type.

    Args:
        prompt_info: Dict with 'format' key and format-specific fields:
            - chat_response: {'format': 'chat_response', 'user_message': '...'}
            - chat_continuation: {'format': 'chat_continuation', 'user_message': '...', 'assistant_prefix': '...'}
            - raw_continuation: {'format': 'raw_continuation', 'text': '...'}

            Also supports legacy format: {'user_message': '...', 'assistant_prefix': '...' or None}
            And old format: {'prompt': '...'}

    Returns:
        Formatted string ready for model input
    """
    config = get_model_config()

    # Determine format type
    fmt = prompt_info.get("format", "")

    # Handle legacy formats
    if not fmt:
        if "user_message" in prompt_info:
            # Legacy: user_message + optional assistant_prefix
            fmt = "chat_continuation" if prompt_info.get("assistant_prefix") else "chat_response"
        elif "prompt" in prompt_info:
            # Old format: just 'prompt' field - treat as chat_response
            fmt = "chat_response"
            prompt_info = {"format": fmt, "user_message": prompt_info["prompt"]}
        elif "text" in prompt_info:
            fmt = "raw_continuation"
        else:
            # Fallback: treat as raw text
            fmt = "raw_continuation"
            prompt_info = {"format": fmt, "text": str(prompt_info)}

    # Check if this is Llama-style model
    is_llama = "<|begin_of_text|>" in config.chat_template

    if fmt == "raw_continuation":
        # No chat template - just raw text
        return prompt_info.get("text", "")

    elif fmt == "chat_response":
        # Standard chat: user asks, assistant responds from scratch
        user_msg = prompt_info.get("user_message", "")
        if is_llama:
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            return config.format_prompt(user_msg)

    elif fmt == "chat_continuation":
        # Chat with assistant prefill
        user_msg = prompt_info.get("user_message", "Continue:")
        prefix = prompt_info.get("assistant_prefix", "")
        if is_llama:
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are completing text naturally.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{prefix}"""
        else:
            return config.format_prompt(user_msg) + prefix

    else:
        # Unknown format - treat as raw
        return prompt_info.get("text", prompt_info.get("user_message", str(prompt_info)))


def format_generation_prompt(prompt: str, generation_format: str = "continuation") -> str:
    """Format a prompt for generation based on the specified format.

    Args:
        prompt: The prompt text
        generation_format: One of "continuation", "chat", "raw"
            - "continuation": Prompt becomes assistant prefix (model continues text)
            - "chat": Prompt becomes user message (model responds)
            - "raw": No template wrapping
    """
    if generation_format == "raw":
        return prompt
    elif generation_format == "chat":
        return format_steering_prompt({"format": "chat_response", "user_message": prompt})
    else:  # "continuation" (default)
        return format_steering_prompt({
            "format": "chat_continuation",
            "user_message": "Continue this text naturally:",
            "assistant_prefix": prompt,
        })


def get_neuron_activation(
    layer: int,
    neuron_idx: int,
    text: str,
    position: int = -1,
) -> tuple[float, int, str]:
    """Get neuron activation for given text at specified position.

    Thread-safe: Uses _MODEL_LOCK to prevent concurrent CUDA operations.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index within layer
        text: Input text (already formatted or raw)
        position: Position to check (-1 for last)

    Returns:
        (activation, position, token_at_position)
    """
    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        if position < 0:
            position = seq_len + position

        # Capture intermediate activations
        hidden_states = []

        def capture_hook(module, input, output):
            hidden_states.append(input[0].detach())

        mlp = model.model.layers[layer].mlp
        handle = mlp.gate_proj.register_forward_hook(capture_hook)

        try:
            with torch.no_grad():
                model(**inputs)

            if hidden_states:
                hidden = hidden_states[0]
                gate = mlp.gate_proj(hidden)
                up = mlp.up_proj(hidden)
                intermediate = torch.nn.functional.silu(gate) * up

                neuron_act = intermediate[0, position, neuron_idx].item()
                token_id = input_ids[0, position].item()
                token = tokenizer.decode([token_id])

                return neuron_act, position, token
        finally:
            handle.remove()

        return 0.0, position, ""


def get_all_activations(
    layer: int,
    neuron_idx: int,
    text: str,
) -> list[tuple[int, float, str]]:
    """Get neuron activations at all positions.

    Thread-safe: Uses _MODEL_LOCK to prevent concurrent CUDA operations.

    Returns list of (position, activation, token).
    """
    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        hidden_states = []

        def capture_hook(module, input, output):
            hidden_states.append(input[0].detach())

        mlp = model.model.layers[layer].mlp
        handle = mlp.gate_proj.register_forward_hook(capture_hook)

        try:
            with torch.no_grad():
                model(**inputs)

            if hidden_states:
                hidden = hidden_states[0]
                gate = mlp.gate_proj(hidden)
                up = mlp.up_proj(hidden)
                intermediate = torch.nn.functional.silu(gate) * up

                results = []
                for pos in range(seq_len):
                    act = intermediate[0, pos, neuron_idx].item()
                    token_id = input_ids[0, pos].item()
                    token = tokenizer.decode([token_id])
                    results.append((pos, act, token))

                return results
        finally:
            handle.remove()

        return []


def get_gate_up_decomposition(
    layer: int,
    neuron_idx: int,
    texts: list[str],
) -> dict[str, Any]:
    """Decompose SwiGLU activation into gate and up components to detect operating regime.

    For each text, at the position of max |activation|, captures:
    - gate_pre, up_pre, activation (= SiLU(gate_pre) * up_pre)
    - quadrant: "standard" (++), "inverted" (--), "gate_neg" (-+), "up_neg" (+-)
    - firing_sign: "positive" or "negative"

    Aggregates across texts to determine the neuron's operating regime.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index within layer
        texts: List of input texts (already formatted or raw)

    Returns:
        Dict with per-text decomposition and aggregate regime analysis.
    """
    if not texts:
        return {"error": "No texts provided", "regime": "unknown", "regime_confidence": 0.0}

    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        per_text_results = []

        # Process in batches to match get_all_activations_batch pattern
        for batch_start in range(0, len(texts), MAX_BATCH_SIZE):
            batch_end = min(batch_start + MAX_BATCH_SIZE, len(texts))
            batch_texts = texts[batch_start:batch_end]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            captured_hidden = [None]

            def capture_hook(module, input, output):
                captured_hidden[0] = input[0].detach()

            mlp = model.model.layers[layer].mlp
            handle = mlp.gate_proj.register_forward_hook(capture_hook)

            try:
                with torch.no_grad():
                    model(**inputs)

                if captured_hidden[0] is None:
                    continue

                hidden = captured_hidden[0]
                gate_pre = mlp.gate_proj(hidden)
                up_pre = mlp.up_proj(hidden)
                intermediate = torch.nn.functional.silu(gate_pre) * up_pre

                for i in range(len(batch_texts)):
                    seq_len = (inputs.attention_mask[i] == 1).sum().item()
                    # Find position of max |activation| within valid tokens
                    acts = intermediate[i, :seq_len, neuron_idx]
                    max_abs_pos = torch.argmax(torch.abs(acts)).item()

                    act_val = acts[max_abs_pos].item()
                    gate_val = gate_pre[i, max_abs_pos, neuron_idx].item()
                    up_val = up_pre[i, max_abs_pos, neuron_idx].item()

                    # Determine quadrant
                    if gate_val >= 0 and up_val >= 0:
                        quadrant = "standard"
                    elif gate_val < 0 and up_val < 0:
                        quadrant = "inverted"
                    elif gate_val < 0 and up_val >= 0:
                        quadrant = "gate_neg"
                    else:
                        quadrant = "up_neg"

                    per_text_results.append({
                        "gate_pre": round(gate_val, 4),
                        "up_pre": round(up_val, 4),
                        "activation": round(act_val, 4),
                        "quadrant": quadrant,
                        "firing_sign": "positive" if act_val > 0 else "negative",
                        "position": max_abs_pos,
                    })

            finally:
                handle.remove()

    if not per_text_results:
        return {"error": "No results", "regime": "unknown", "regime_confidence": 0.0}

    # Aggregate statistics
    quadrant_counts = {"standard": 0, "inverted": 0, "gate_neg": 0, "up_neg": 0}
    positive_count = 0
    negative_count = 0
    gate_pre_vals = []
    up_pre_vals = []

    for r in per_text_results:
        quadrant_counts[r["quadrant"]] += 1
        if r["firing_sign"] == "positive":
            positive_count += 1
        else:
            negative_count += 1
        gate_pre_vals.append(r["gate_pre"])
        up_pre_vals.append(r["up_pre"])

    total = len(per_text_results)
    standard_frac = quadrant_counts["standard"] / total
    inverted_frac = quadrant_counts["inverted"] / total

    if standard_frac >= 0.7:
        regime = "standard"
        regime_confidence = standard_frac
    elif inverted_frac >= 0.7:
        regime = "inverted"
        regime_confidence = inverted_frac
    else:
        regime = "mixed"
        regime_confidence = max(standard_frac, inverted_frac)

    positive_pct = 100 * positive_count / total
    negative_pct = 100 * negative_count / total

    return {
        "regime": regime,
        "regime_confidence": round(regime_confidence, 3),
        "quadrant_counts": quadrant_counts,
        "total_samples": total,
        "positive_firing_pct": round(positive_pct, 1),
        "negative_firing_pct": round(negative_pct, 1),
        "mean_gate_pre": round(float(np.mean(gate_pre_vals)), 4),
        "mean_up_pre": round(float(np.mean(up_pre_vals)), 4),
        "per_text": per_text_results,  # Full decomposition per text
    }


def get_activations_batch(
    layer: int,
    neuron_idx: int,
    texts: list[str],
    positions: list[int] | None = None,
) -> list[tuple[float, int, str]]:
    """Get neuron activations for multiple texts in a single batched forward pass.

    This is the core batching optimization: instead of N sequential forward passes,
    we run one batched forward pass for N texts, significantly improving GPU utilization.

    Thread-safe: Uses _MODEL_LOCK to prevent concurrent CUDA operations.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index within layer
        texts: List of input texts (already formatted or raw)
        positions: Optional per-text positions to check (-1 for last token of each).
                   If None, defaults to -1 (last token) for all texts.

    Returns:
        List of (activation, position, token_at_position) tuples, one per input text.
    """
    if not texts:
        return []

    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        batch_size = len(texts)
        if positions is None:
            positions = [-1] * batch_size

        # Tokenize all texts with padding
        # Use right-padding for causal LM (left-padding would shift positions)
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # Cap to avoid OOM on very long prompts
        ).to(device)

        # Per-batch local hook state (not global to avoid race conditions)
        captured_hidden = [None]

        def capture_hook(module, input, output):
            # input[0] has shape: (batch, seq, hidden_dim)
            captured_hidden[0] = input[0].detach()

        mlp = model.model.layers[layer].mlp
        handle = mlp.gate_proj.register_forward_hook(capture_hook)

        try:
            with torch.no_grad():
                model(**inputs)

            if captured_hidden[0] is None:
                return [(0.0, 0, "")] * batch_size

            hidden = captured_hidden[0]  # (batch, seq, hidden_dim)
            gate = mlp.gate_proj(hidden)
            up = mlp.up_proj(hidden)
            intermediate = torch.nn.functional.silu(gate) * up  # (batch, seq, intermediate_dim)

            # Extract per-text activations
            results = []
            for i in range(batch_size):
                # Compute actual sequence length for this example (excluding padding)
                seq_len = (inputs.attention_mask[i] == 1).sum().item()

                # Map position (-1 means last non-padding token)
                pos = positions[i]
                if pos < 0:
                    pos = seq_len + pos
                pos = max(0, min(pos, seq_len - 1))  # Bounds check

                act = intermediate[i, pos, neuron_idx].item()
                token_id = inputs.input_ids[i, pos].item()
                token = tokenizer.decode([token_id])

                results.append((act, pos, token))

            return results

        finally:
            handle.remove()  # Always cleanup hook


def get_all_activations_batch(
    layer: int,
    neuron_idx: int,
    texts: list[str],
) -> list[list[tuple[int, float, str]]]:
    """Get neuron activations at all positions for multiple texts in a single batched forward pass.

    This is a batched version of get_all_activations() for higher throughput.

    Thread-safe: Uses _MODEL_LOCK to prevent concurrent CUDA operations.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index within layer
        texts: List of input texts (already formatted or raw)

    Returns:
        List of lists, where each inner list contains (position, activation, token) tuples
        for one input text.
    """
    if not texts:
        return []

    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        batch_size = len(texts)

        # Tokenize all texts with padding
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        # Per-batch local hook state
        captured_hidden = [None]

        def capture_hook(module, input, output):
            captured_hidden[0] = input[0].detach()

        mlp = model.model.layers[layer].mlp
        handle = mlp.gate_proj.register_forward_hook(capture_hook)

        try:
            with torch.no_grad():
                model(**inputs)

            if captured_hidden[0] is None:
                return [[]] * batch_size

            hidden = captured_hidden[0]
            gate = mlp.gate_proj(hidden)
            up = mlp.up_proj(hidden)
            intermediate = torch.nn.functional.silu(gate) * up

            # Extract per-text, per-position activations
            all_results = []
            for i in range(batch_size):
                # Actual sequence length (excluding padding)
                seq_len = (inputs.attention_mask[i] == 1).sum().item()

                text_results = []
                for pos in range(seq_len):
                    act = intermediate[i, pos, neuron_idx].item()
                    token_id = inputs.input_ids[i, pos].item()
                    token = tokenizer.decode([token_id])
                    text_results.append((pos, act, token))

                all_results.append(text_results)

            return all_results

        finally:
            handle.remove()


def run_ablation(
    layer: int,
    neuron_idx: int,
    text: str,
    top_k_logits: int = 10,
    ablation_method: str = "mean",
) -> dict[str, Any]:
    """Run ablation experiment - ablate neuron and measure logit shifts.

    Thread-safe: Uses _MODEL_LOCK to prevent concurrent CUDA operations.

    Args:
        layer: Layer number
        neuron_idx: Neuron index
        text: Formatted prompt text
        top_k_logits: Number of top logits to return
        ablation_method: "mean" (default, less OOD) or "zero" (traditional)

    Returns dict with original_logits, ablated_logits, and shifts.
    """
    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        inputs = tokenizer(text, return_tensors="pt").to(device)

        # Get original logits
        with torch.no_grad():
            outputs = model(**inputs)
            original_logits = outputs.logits[0, -1].float()

        # Get top tokens
        top_values, top_indices = torch.topk(original_logits, top_k_logits)
        original_top = {
            tokenizer.decode([idx.item()]): val.item()
            for idx, val in zip(top_indices, top_values)
        }

        # Determine ablation value
        if ablation_method == "mean":
            # Mean ablation: replace with mean activation across reference prompts
            # This is less out-of-distribution than zero ablation
            ablation_value = get_mean_activation(layer, neuron_idx)
        else:  # "zero"
            ablation_value = 0.0

        # Ablate by setting the column in down_proj to produce ablation_value output
        # Since down_proj output = down_proj @ activation, we scale the column
        mlp = model.model.layers[layer].mlp
        with torch.no_grad():
            down_weight = mlp.down_proj.weight.clone()

            if ablation_method == "mean" and ablation_value != 0:
                # For mean ablation, we use a hook to set the activation to mean value
                # instead of modifying weights (cleaner approach)
                def ablation_hook(module, args, kwargs):
                    x = args[0]  # Shape: (batch, seq_len, intermediate_dim)
                    modified = x.clone()
                    modified[:, :, neuron_idx] = ablation_value
                    return (modified,) + args[1:], kwargs

                handle = mlp.down_proj.register_forward_pre_hook(ablation_hook, with_kwargs=True)
                try:
                    outputs = model(**inputs)
                    ablated_logits = outputs.logits[0, -1].float()
                finally:
                    handle.remove()
            else:
                # Zero ablation: zero out the weight column
                mlp.down_proj.weight[:, neuron_idx] = 0
                outputs = model(**inputs)
                ablated_logits = outputs.logits[0, -1].float()
                # Restore weights
                mlp.down_proj.weight.copy_(down_weight)

        # Compute shifts
        ablated_top = {
            tokenizer.decode([idx.item()]): ablated_logits[idx.item()].item()
            for idx in top_indices
        }

        logit_shifts = {
            token: ablated_top.get(token, 0) - original_top.get(token, 0)
            for token in original_top
        }

        most_affected = max(logit_shifts.items(), key=lambda x: abs(x[1]))

        return {
            "original_logits": original_top,
            "ablated_logits": ablated_top,
            "logit_shifts": logit_shifts,
            "most_affected_token": most_affected[0],
            "max_shift": most_affected[1],
            "ablation_method": ablation_method,
            "ablation_value": ablation_value,
        }


# =============================================================================
# MCP Tool Implementations (called by agent)
# =============================================================================

def _sync_test_activation(layer: int, neuron_idx: int, prompt: str, activation_threshold: float) -> dict[str, Any]:
    """Synchronous implementation of test_activation. Runs on dedicated CUDA thread."""
    text = format_prompt(prompt)
    all_acts = get_all_activations(layer, neuron_idx, text)

    if not all_acts:
        return {"error": "Failed to get activations"}

    # Find max activation (global)
    max_pos, max_act, max_token = max(all_acts, key=lambda x: x[1])

    # Find where user content starts and ends (between user header and assistant header)
    # For Llama chat template: <|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>
    user_content_start = 0
    user_content_end = len(all_acts)
    current_section = None  # Track which section: "system", "user", "assistant"

    for pos, act, tok in all_acts:
        tok_stripped = tok.strip()
        # Track section by watching for section name tokens
        if tok_stripped in ("system", "user", "assistant"):
            current_section = tok_stripped

        # end_header_id marks start of content for current section
        if "end_header_id" in tok:
            if current_section == "user":
                user_content_start = pos + 1

        # eot_id marks end of content for current section
        if "eot_id" in tok:
            if current_section == "user" and user_content_start > 0:
                user_content_end = pos
                break

    # Find max activation WITHIN user content (for highlighting)
    content_acts = [(pos, act, tok) for pos, act, tok in all_acts
                   if user_content_start <= pos < user_content_end]
    if content_acts:
        content_max_pos, content_max_act, content_max_token = max(content_acts, key=lambda x: x[1])
        content_min_pos, content_min_act, content_min_token = min(content_acts, key=lambda x: x[1])
    else:
        content_max_pos, content_max_act, content_max_token = max_pos, max_act, max_token
        content_min_pos, content_min_act, content_min_token = max_pos, max_act, max_token

    # Find where assistant response starts (after last <|end_header_id|>)
    assistant_start = 0
    for pos, act, tok in all_acts:
        if "end_header_id" in tok:
            assistant_start = pos + 1

    # Build context showing prefix up to content activation (what model "sees" when firing)
    tokens_before = [tok for pos, act, tok in all_acts if pos <= content_max_pos]
    clean_tokens = [t.replace('Ġ', ' ').replace('▁', ' ').lstrip() for t in tokens_before]
    prefix_tokens = []
    in_content = False
    for t in clean_tokens:
        if 'end_header_id' in t or 'eot_id' in t:
            in_content = True
            prefix_tokens = []
            continue
        if in_content:
            prefix_tokens.append(t)
    prefix_text = ''.join(prefix_tokens).strip()
    if len(prefix_text) > 80:
        prefix_text = '...' + prefix_text[-77:]

    # Build context prefix for min activation position
    tokens_before_min = [tok for pos, act, tok in all_acts if pos <= content_min_pos]
    clean_tokens_min = [t.replace('Ġ', ' ').replace('▁', ' ').lstrip() for t in tokens_before_min]
    prefix_tokens_min = []
    in_content_min = False
    for t in clean_tokens_min:
        if 'end_header_id' in t or 'eot_id' in t:
            in_content_min = True
            prefix_tokens_min = []
            continue
        if in_content_min:
            prefix_tokens_min.append(t)
    prefix_text_min = ''.join(prefix_tokens_min).strip()
    if len(prefix_text_min) > 80:
        prefix_text_min = '...' + prefix_text_min[-77:]

    return {
        "prompt": prompt[:100],
        "max_activation": content_max_act,  # Use CONTENT max, not global (template tokens can have spurious high activation)
        "max_position": content_max_pos,  # Position within content
        "token_at_max": content_max_token,  # Content token for highlighting
        "fires_after": prefix_text,  # Shows prefix up to activation token - what model "sees" when neuron fires
        "activates": content_max_act > activation_threshold,  # Use content max for threshold check
        "activation_threshold": activation_threshold,
        "assistant_start": assistant_start,
        "total_tokens": len(all_acts),
        "global_max_token": max_token,  # Keep global max for reference
        "global_max_pos": max_pos,
        "global_max_activation": max_act,  # Global max for reference (may be on template tokens)
        # Negative (min) activation data
        "min_activation": content_min_act,
        "min_position": content_min_pos,
        "token_at_min": content_min_token,
        "fires_after_min": prefix_text_min,
        "negatively_activates": content_min_act < -activation_threshold,
    }


async def tool_test_activation(
    layer: int,
    neuron_idx: int,
    prompt: str,
    activation_threshold: float = 0.5,
) -> dict[str, Any]:
    """Test neuron activation on a single prompt.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index within layer
        prompt: The prompt to test
        activation_threshold: Threshold for considering the neuron "activated" (default 0.5).
            Lower values (0.1-0.3) catch weaker activations, higher values (1.0+) are more selective.

    Returns activation value, position, and token where max activation occurs.
    """
    # Record experiment for temporal enforcement
    record_experiment("activation_test", {
        "layer": layer,
        "neuron_idx": neuron_idx,
        "prompt": prompt[:100],
    })

    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        return await client.test_activation(
            layer=layer, neuron_idx=neuron_idx, prompt=prompt,
            activation_threshold=activation_threshold,
        )

    # Run on dedicated CUDA thread to avoid context switching issues
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _CUDA_EXECUTOR,
        _sync_test_activation,
        layer,
        neuron_idx,
        prompt,
        activation_threshold,
    )


def _sync_batch_activation_test(
    layer: int,
    neuron_idx: int,
    prompts: list[str],
    activation_threshold: float,
) -> dict[str, Any]:
    """Synchronous batched activation test. Runs on dedicated CUDA thread.

    Uses efficient GPU batching to process multiple prompts in fewer forward passes.
    """
    if not prompts:
        return {
            "total_tested": 0,
            "activating_count": 0,
            "non_activating_count": 0,
            "negatively_activating_count": 0,
            "activation_threshold": activation_threshold,
            "mean_activation": 0,
            "max_activation": 0,
            "min_activation": 0,
            "mean_min_activation": 0,
            "top_activating": [],
            "top_negatively_activating": [],
            "sample_non_activating": [],
        }

    # Format all prompts with chat template
    texts = [format_prompt(p) for p in prompts]

    activating = []
    non_activating = []
    negatively_activating = []
    all_activations = []
    all_min_activations = []
    errors = []

    # Process in GPU batches for efficiency
    for batch_start in range(0, len(texts), MAX_BATCH_SIZE):
        batch_end = min(batch_start + MAX_BATCH_SIZE, len(texts))
        batch_texts = texts[batch_start:batch_end]
        batch_prompts = prompts[batch_start:batch_end]

        try:
            # Single batched forward pass for this batch
            batch_results = get_all_activations_batch(layer, neuron_idx, batch_texts)

            for i, (prompt, all_acts) in enumerate(zip(batch_prompts, batch_results)):
                if not all_acts:
                    errors.append({"prompt": prompt[:80], "error": "Failed to get activations"})
                    continue

                # Find max activation (global) - may include template tokens
                max_pos, max_act, max_token = max(all_acts, key=lambda x: x[1])

                # Find where user content starts and ends (between user header and assistant header)
                # For Llama chat template: <|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>
                user_content_start = 0
                user_content_end = len(all_acts)
                current_section = None  # Track which section: "system", "user", "assistant"

                for pos, act, tok in all_acts:
                    tok_stripped = tok.strip()
                    # Track section by watching for section name tokens
                    if tok_stripped in ("system", "user", "assistant"):
                        current_section = tok_stripped

                    # end_header_id marks start of content for current section
                    if "end_header_id" in tok:
                        if current_section == "user":
                            user_content_start = pos + 1

                    # eot_id marks end of content for current section
                    if "eot_id" in tok:
                        if current_section == "user" and user_content_start > 0:
                            user_content_end = pos
                            break

                # Find max activation WITHIN user content (for highlighting)
                content_acts = [(pos, act, tok) for pos, act, tok in all_acts
                               if user_content_start <= pos < user_content_end]
                if content_acts:
                    content_max_pos, content_max_act, content_max_token = max(content_acts, key=lambda x: x[1])
                    content_min_pos, content_min_act, content_min_token = min(content_acts, key=lambda x: x[1])
                else:
                    content_max_pos, content_max_act, content_max_token = max_pos, max_act, max_token
                    content_min_pos, content_min_act, content_min_token = max_pos, max_act, max_token

                # Use content max for statistics (template tokens can have spurious high activation)
                all_activations.append(content_max_act)
                all_min_activations.append(content_min_act)

                # Find where assistant response starts (after assistant's end_header_id)
                assistant_start = 0
                current_section = None
                for pos, act, tok in all_acts:
                    tok_stripped = tok.strip()
                    if tok_stripped in ("system", "user", "assistant"):
                        current_section = tok_stripped
                    if "end_header_id" in tok and current_section == "assistant":
                        assistant_start = pos + 1
                        break

                # Build context showing prefix up to content activation (what model "sees" when firing)
                # Get tokens up to and including content activation position
                tokens_before = [tok for pos, act, tok in all_acts if pos <= content_max_pos]
                # Clean up tokens for display (remove BPE artifacts)
                clean_tokens = [t.replace('Ġ', ' ').replace('▁', ' ').lstrip() for t in tokens_before]
                # Skip template tokens and build prefix
                prefix_tokens = []
                in_content = False
                for t in clean_tokens:
                    if 'end_header_id' in t or 'eot_id' in t:
                        in_content = True
                        prefix_tokens = []  # Reset after header
                        continue
                    if in_content:
                        prefix_tokens.append(t)
                prefix_text = ''.join(prefix_tokens).strip()
                # Truncate if too long, keeping the end (near activation)
                if len(prefix_text) > 60:
                    prefix_text = '...' + prefix_text[-57:]

                # Common entry fields including min activation
                entry_base = {
                    "min_activation": content_min_act,
                    "min_position": content_min_pos,
                    "token_at_min": content_min_token,
                }

                # Use content max for classification (template tokens can have spurious high activation)
                if content_max_act > activation_threshold:
                    activating.append({
                        "prompt": prompt,
                        "activation": content_max_act,  # Use content max
                        "token": content_max_token,  # Use content token for highlighting
                        "position": content_max_pos,  # Position within content
                        "fires_after": prefix_text,  # Shows what model "sees" when neuron fires
                        "global_max_activation": max_act,  # Keep global max for reference
                        "global_max_token": max_token,
                        "global_max_pos": max_pos,
                        **entry_base,
                    })
                else:
                    non_activating.append({
                        "prompt": prompt,
                        "activation": content_max_act,  # Use content max
                        "position": content_max_pos,
                        "global_max_activation": max_act,  # Include global for reference
                        **entry_base,
                    })

                # Track negatively activating prompts (min < -threshold)
                if content_min_act < -activation_threshold:
                    negatively_activating.append({
                        "prompt": prompt,
                        "activation": content_min_act,
                        "token": content_min_token,
                        "position": content_min_pos,
                    })

        except Exception as e:
            # If batch fails, record error for each prompt in batch
            for prompt in batch_prompts:
                errors.append({"prompt": prompt[:80], "error": str(e)})

    return {
        "total_tested": len(prompts),
        "activating_count": len(activating),
        "non_activating_count": len(non_activating),
        "negatively_activating_count": len(negatively_activating),
        "activation_threshold": activation_threshold,
        "mean_activation": sum(all_activations) / len(all_activations) if all_activations else 0,
        "max_activation": max(all_activations) if all_activations else 0,
        "min_activation": min(all_min_activations) if all_min_activations else 0,
        "mean_min_activation": sum(all_min_activations) / len(all_min_activations) if all_min_activations else 0,
        "top_activating": sorted(activating, key=lambda x: -x["activation"])[:5],
        "top_negatively_activating": sorted(negatively_activating, key=lambda x: x["activation"])[:5],
        "sample_non_activating": non_activating[:5],
        "errors": errors if errors else None,
    }


async def tool_batch_activation_test(
    layer: int,
    neuron_idx: int,
    prompts: list[str],
    activation_threshold: float = 0.5,
) -> dict[str, Any]:
    """Test neuron activation across multiple prompts using efficient GPU batching.

    This tool uses batched forward passes for significantly higher throughput:
    - Instead of N sequential forward passes, runs ceil(N/16) batched passes
    - Typically 5-10x faster than sequential testing for large prompt sets
    - Memory-efficient: caps batch size to avoid OOM on long prompts

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index within layer
        prompts: List of prompts to test
        activation_threshold: Threshold for considering the neuron "activated" (default 0.5).
            Lower values (0.1-0.3) catch weaker activations, higher values (1.0+) are more selective.

    Returns summary statistics and per-prompt results.
    """
    # Record experiment for temporal enforcement
    record_experiment("batch_activation_test", {
        "layer": layer,
        "neuron_idx": neuron_idx,
        "num_prompts": len(prompts),
    })

    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        return await client.batch_activation_test(
            layer=layer, neuron_idx=neuron_idx, prompts=prompts,
            activation_threshold=activation_threshold,
        )

    # Run batched processing on dedicated CUDA thread
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _CUDA_EXECUTOR,
        _sync_batch_activation_test,
        layer,
        neuron_idx,
        prompts,
        activation_threshold,
    )


def _sync_run_ablation(layer: int, neuron_idx: int, prompt: str, ablation_method: str = "mean") -> dict[str, Any]:
    """Synchronous implementation of run_ablation. Runs on dedicated CUDA thread."""
    text = format_prompt(prompt)
    result = run_ablation(layer, neuron_idx, text, ablation_method=ablation_method)
    result["prompt"] = prompt[:100]
    return result


async def tool_run_ablation(
    layer: int,
    neuron_idx: int,
    prompt: str,
    ablation_method: str = "mean",
) -> dict[str, Any]:
    """Run ablation experiment on a prompt.

    Ablates the neuron and measures effect on output logits.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index within layer
        prompt: The prompt to test
        ablation_method: "mean" (default, replaces with mean activation - less OOD)
                        or "zero" (traditional zero ablation)

    Returns:
        Dict with original_logits, ablated_logits, logit_shifts, and ablation metadata.
    """
    # Record experiment for temporal enforcement
    record_experiment("ablation", {
        "layer": layer,
        "neuron_idx": neuron_idx,
        "prompt": prompt[:100],
        "ablation_method": ablation_method,
    })

    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        return await client.run_ablation(
            layer=layer, neuron_idx=neuron_idx, prompt=prompt,
            ablation_method=ablation_method,
        )

    # Run on dedicated CUDA thread to avoid context switching issues
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _CUDA_EXECUTOR,
        _sync_run_ablation,
        layer,
        neuron_idx,
        prompt,
        ablation_method,
    )


async def tool_batch_ablation(
    layer: int,
    neuron_idx: int,
    prompts: list[str],
    ablation_method: str = "mean",
) -> dict[str, Any]:
    """Run ablation experiments on multiple prompts.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index within layer
        prompts: List of prompts to test
        ablation_method: "mean" (default) or "zero"

    Returns consistent patterns across ablations.
    """
    results = []
    token_shifts = {}

    for prompt in prompts:
        try:
            result = await tool_run_ablation(layer, neuron_idx, prompt, ablation_method)
            results.append(result)

            for token, shift in result.get("logit_shifts", {}).items():
                if token not in token_shifts:
                    token_shifts[token] = []
                token_shifts[token].append(shift)
        except Exception as e:
            results.append({"prompt": prompt[:80], "error": str(e)})

    # Find consistent effects
    consistent_promotes = []
    consistent_suppresses = []

    for token, shifts in token_shifts.items():
        if len(shifts) < 2:
            continue
        avg = sum(shifts) / len(shifts)
        if all(s > 0 for s in shifts):
            consistent_promotes.append((token, avg))
        elif all(s < 0 for s in shifts):
            consistent_suppresses.append((token, avg))

    # Build individual results with per-prompt promotes/suppresses
    individual_results = []
    for r in results:
        if "error" in r:
            continue
        logit_shifts = r.get("logit_shifts", {})
        # Extract per-prompt promotes (positive shifts) and suppresses (negative shifts)
        per_prompt_promotes = [(tok, shift) for tok, shift in logit_shifts.items() if shift > 0]
        per_prompt_suppresses = [(tok, shift) for tok, shift in logit_shifts.items() if shift < 0]
        individual_results.append({
            "prompt": r.get("prompt", ""),
            "most_affected": r.get("most_affected_token", ""),
            "max_shift": r.get("max_shift", 0),
            "promotes": sorted(per_prompt_promotes, key=lambda x: -x[1])[:5],
            "suppresses": sorted(per_prompt_suppresses, key=lambda x: x[1])[:5],
        })

    return {
        "total_ablations": len([r for r in results if "error" not in r]),
        "consistent_promotes": sorted(consistent_promotes, key=lambda x: -x[1])[:5],
        "consistent_suppresses": sorted(consistent_suppresses, key=lambda x: x[1])[:5],
        "individual_results": individual_results[:10],
    }


def _sync_get_relp_connectivity(layer: int, neuron_idx: int, edge_stats_path: str) -> dict[str, Any]:
    """Get RelP-based connectivity from edge statistics (informational only).

    Returns upstream and downstream neurons observed in RelP attribution graphs.
    This shows CONTEXT-SPECIFIC connectivity from aggregated RelP runs, not
    weight-based "in potentia" connectivity.

    NOTE: This is informational only - it does NOT populate the connectivity
    field for the dashboard. Use analyze_wiring/analyze_output_wiring for that.
    """
    neuron_id = f"L{layer}/N{neuron_idx}"

    try:
        with open(edge_stats_path) as f:
            edge_stats = json.load(f)
    except FileNotFoundError:
        return {"error": f"Edge stats not found: {edge_stats_path}"}

    profile = None
    for p in edge_stats.get("profiles", []):
        if p.get("neuron_id") == neuron_id:
            profile = p
            break

    if not profile:
        return {"error": f"Neuron {neuron_id} not found in edge stats"}

    # Extract connections - this may block on model loading
    _, tokenizer = get_model_and_tokenizer()

    # Collect ALL upstream connections first, then sort by absolute weight
    all_upstream = []
    for u in profile.get("top_upstream_sources", []):
        source = u.get("source", "")
        parts = source.split("_")
        if len(parts) >= 2 and not source.startswith("E_"):
            all_upstream.append({
                "neuron_id": f"L{parts[0]}/N{parts[1]}",
                "weight": u.get("avg_weight", 0),
                "frequency": u.get("frequency", 0),
            })
    # Sort by absolute weight to show strongest connections regardless of sign
    upstream = sorted(all_upstream, key=lambda x: abs(x.get("weight", 0)), reverse=True)[:10]

    # Collect ALL downstream connections first, then sort by absolute weight
    # Filter out same-layer or earlier-layer neurons (architecturally impossible in feedforward)
    all_downstream = []
    for d in profile.get("top_downstream_targets", []):
        target = d.get("target", "")
        if target.startswith("L_"):
            # Logit target - always valid
            token_id = int(target.split("_")[1])
            token = tokenizer.decode([token_id])
            all_downstream.append({
                "target": f"LOGIT({token})",
                "weight": d.get("avg_weight", 0),
                "frequency": d.get("frequency", 0),
            })
        else:
            parts = target.split("_")
            if len(parts) >= 2:
                target_layer = int(parts[0])
                # Only include downstream neurons from LATER layers
                if target_layer > layer:
                    all_downstream.append({
                        "neuron_id": f"L{parts[0]}/N{parts[1]}",
                        "weight": d.get("avg_weight", 0),
                        "frequency": d.get("frequency", 0),
                    })
    # Sort by absolute weight to show strongest connections regardless of sign
    downstream = sorted(all_downstream, key=lambda x: abs(x["weight"]), reverse=True)[:10]

    # Look up labels for all connected neurons (with NeuronDB fallback)
    all_neuron_ids = [u["neuron_id"] for u in upstream]
    all_neuron_ids += [d["neuron_id"] for d in downstream if "neuron_id" in d]
    labels = batch_get_neuron_labels_with_fallback(all_neuron_ids) if all_neuron_ids else {}

    # Add labels to upstream neurons
    for u in upstream:
        nid = u["neuron_id"]
        label_info = labels.get(nid, {})
        u["label"] = label_info.get("label", "")
        u["label_source"] = label_info.get("source", None)

    # Add labels to downstream neurons (skip LOGIT targets)
    for d in downstream:
        if "neuron_id" in d:
            nid = d["neuron_id"]
            label_info = labels.get(nid, {})
            d["label"] = label_info.get("label", "")
            d["label_source"] = label_info.get("source", None)

    return {
        "neuron_id": neuron_id,
        "appearance_count": profile.get("appearance_count", 0),
        "transluce_label_positive": profile.get("transluce_label_positive", ""),
        "transluce_label_negative": profile.get("transluce_label_negative", ""),
        "upstream_neurons": upstream,
        "downstream_targets": downstream,
    }


async def tool_get_relp_connectivity(
    layer: int,
    neuron_idx: int,
    edge_stats_path: str,
) -> dict[str, Any]:
    """Get RelP-based connectivity from aggregated edge statistics (informational only).

    Returns upstream and downstream connections observed in RelP attribution graphs.
    This shows CONTEXT-SPECIFIC connectivity - what actually influences this neuron
    in real prompts, as opposed to weight-based "in potentia" connectivity.

    NOTE: This is INFORMATIONAL ONLY. It does NOT populate the connectivity field
    for the dashboard. Use analyze_wiring() and analyze_output_wiring() which
    auto-populate connectivity based on weight analysis.

    Use this tool to compare weight-based predictions with actual context usage:
    - Do the strongest weight-based connections appear in RelP?
    - Are there context-specific connections not predicted by weights?
    """
    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        result = await client.get_relp_connectivity(
            layer=layer, neuron_idx=neuron_idx, edge_stats_path=edge_stats_path,
        )
        # Log what was found (informational only - no state population)
        if "error" not in result:
            downstream_targets = result.get("downstream_targets", [])
            downstream_neurons = [d for d in downstream_targets if "neuron_id" in d]
            upstream_sources = result.get("upstream_neurons", [])
            print(f"[INFO] RelP connectivity: {len(upstream_sources)} upstream, {len(downstream_neurons)} downstream neurons observed in corpus")
        return result

    # Run on dedicated CUDA thread (uses model tokenizer)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _CUDA_EXECUTOR,
        _sync_get_relp_connectivity,
        layer,
        neuron_idx,
        edge_stats_path,
    )

    # Log what was found (informational only - no state population)
    if "error" not in result:
        downstream_targets = result.get("downstream_targets", [])
        downstream_neurons = [d for d in downstream_targets if "neuron_id" in d]
        upstream_sources = result.get("upstream_neurons", [])
        print(f"[INFO] RelP connectivity: {len(upstream_sources)} upstream, {len(downstream_neurons)} downstream neurons observed in corpus")

    return result


# =============================================================================
# Wiring Analysis Tool (REQUIRED EARLY STEP)
# =============================================================================


def _sync_analyze_wiring(
    layer: int,
    neuron_idx: int,
    top_k: int = 100,
    min_layer: int = 0,
    target_regime: str | None = None,
) -> dict[str, Any]:
    """Compute weight-based wiring analysis with polarity prediction.

    This analyzes ALL upstream MLP neurons by computing their weight-based
    connectivity to the target neuron. For SwiGLU MLPs:

        target_activation = SiLU(gate_pre) * up_pre

    where gate_pre and up_pre depend on upstream neuron outputs via:
        c_up = e_up · d_i   (contribution to up channel)
        c_gate = e_gate · d_i   (contribution to gate channel)

    Polarity prediction:
        - Both positive: EXCITATORY (increases target activation)
        - Both negative: INHIBITORY (decreases target activation)
        - Mixed signs: MIXED (context-dependent)

    If target_regime is "inverted", all polarity predictions are flipped because
    the target neuron operates with gate_pre < 0 and up_pre < 0.

    Returns:
        Dict with excitatory/inhibitory upstream neurons, stats, and labels.
    """
    neuron_id = f"L{layer}/N{neuron_idx}"

    model, _ = get_model_and_tokenizer()

    # Get target neuron's input weights
    target_mlp = model.model.layers[layer].mlp
    e_up = target_mlp.up_proj.weight[neuron_idx, :]   # stay in native dtype
    e_gate = target_mlp.gate_proj.weight[neuron_idx, :]  # stay in native dtype

    # Compute connectivity from all upstream layers
    config = get_model_config()

    # Vectorized: collect per-layer matmul results, then top-k across all layers
    all_layers = []
    all_c_up_list = []
    all_c_gate_list = []
    total_exc = 0
    total_inh = 0
    n_intermediate = None

    for up_layer in range(min_layer, layer):
        upstream_mlp = model.model.layers[up_layer].mlp
        # Stay in native dtype (no .float() — saves ~500MB/layer on large models)
        down_proj = upstream_mlp.down_proj.weight  # [hidden, intermediate]

        c_up_all = torch.matmul(e_up, down_proj)  # [intermediate]
        c_gate_all = torch.matmul(e_gate, down_proj)  # [intermediate]

        if n_intermediate is None:
            n_intermediate = c_up_all.shape[0]

        both_pos = (c_up_all > 0) & (c_gate_all > 0)
        both_neg = (c_up_all < 0) & (c_gate_all < 0)
        total_exc += both_pos.sum().item()
        total_inh += both_neg.sum().item()

        all_layers.append(up_layer)
        all_c_up_list.append(c_up_all.cpu())
        all_c_gate_list.append(c_gate_all.cpu())

    n_layers_up = len(all_layers)
    if n_layers_up == 0 or n_intermediate is None:
        all_connections = []
        excitatory = []
        inhibitory = []
        total_mixed = 0
        stats = {
            "total_upstream_neurons": 0,
            "excitatory_count": 0, "inhibitory_count": 0, "mixed_count": 0,
            "c_combined_percentiles": {"90": 0, "95": 0, "99": 0},
        }
    else:
        c_up_stacked = torch.stack(all_c_up_list)
        c_gate_stacked = torch.stack(all_c_gate_list)

        both_pos = (c_up_stacked > 0) & (c_gate_stacked > 0)
        both_neg = (c_up_stacked < 0) & (c_gate_stacked < 0)
        c_combined = c_up_stacked.abs() + c_gate_stacked.abs()
        effective = torch.where(both_pos, c_combined,
                   torch.where(both_neg, -c_combined,
                   torch.zeros_like(c_combined)))

        eff_flat = effective.reshape(-1)
        k_exc = min(top_k, (eff_flat > 0).sum().item())
        k_inh = min(top_k, (eff_flat < 0).sum().item())
        total_mixed = n_layers_up * n_intermediate - total_exc - total_inh

        def _idx_to_conn(flat_idx):
            li = flat_idx // n_intermediate
            ni = flat_idx % n_intermediate
            ul = all_layers[li]
            cu = c_up_stacked[li, ni].item()
            cg = c_gate_stacked[li, ni].item()
            cc = abs(cu) + abs(cg)
            if cu > 0 and cg > 0:
                pol, conf = "excitatory", min(abs(cu), abs(cg)) / max(abs(cu), abs(cg)) if max(abs(cu), abs(cg)) > 0 else 0
                eff = cc
            elif cu < 0 and cg < 0:
                pol, conf = "inhibitory", min(abs(cu), abs(cg)) / max(abs(cu), abs(cg)) if max(abs(cu), abs(cg)) > 0 else 0
                eff = -cc
            else:
                pol, conf, eff = "mixed", 0.5, 0
            return {
                "layer": ul, "neuron": int(ni), "neuron_id": f"L{ul}/N{ni}",
                "c_up": cu, "c_gate": cg, "c_combined": cc,
                "predicted_polarity": pol, "polarity_confidence": conf,
                "effective_strength": eff,
                "relp_confirmed": None, "relp_strength": None,
            }

        all_connections = []
        if k_exc > 0:
            _, top_exc_idx = torch.topk(eff_flat, k_exc)
            for idx in top_exc_idx.tolist():
                all_connections.append(_idx_to_conn(idx))
        if k_inh > 0:
            _, top_inh_idx = torch.topk(-eff_flat, k_inh)
            for idx in top_inh_idx.tolist():
                conn = _idx_to_conn(idx)
                if conn["predicted_polarity"] == "inhibitory":
                    all_connections.append(conn)

        all_connections.sort(key=lambda c: abs(c["effective_strength"]), reverse=True)
        excitatory = [c for c in all_connections if c["predicted_polarity"] == "excitatory"][:top_k]
        inhibitory = [c for c in all_connections if c["predicted_polarity"] == "inhibitory"][:top_k]

        c_combined_flat = c_combined.reshape(-1)
        stats = {
            "total_upstream_neurons": n_layers_up * n_intermediate,
            "excitatory_count": total_exc,
            "inhibitory_count": total_inh,
            "mixed_count": total_mixed,
            "c_combined_percentiles": {
                "90": float(torch.quantile(c_combined_flat.float(), 0.90).item()),
                "95": float(torch.quantile(c_combined_flat.float(), 0.95).item()),
                "99": float(torch.quantile(c_combined_flat.float(), 0.99).item()),
            },
        }
        del c_up_stacked, c_gate_stacked, c_combined_flat

    # Get labels for top neurons
    all_top_ids = [c["neuron_id"] for c in excitatory + inhibitory]
    labels = batch_get_neuron_labels_with_fallback(all_top_ids) if all_top_ids else {}

    # Add labels to connections
    for c in excitatory + inhibitory:
        label_info = labels.get(c["neuron_id"], {})
        c["label"] = label_info.get("label", "")[:150]  # Truncate
        c["label_source"] = label_info.get("source", None)

    # Label coverage
    labeled = sum(1 for c in excitatory + inhibitory if c.get("label"))
    coverage = labeled / len(all_top_ids) * 100 if all_top_ids else 0

    # Generate analysis summary for the agent
    exc_labels = [c.get("label", "") for c in excitatory[:20] if c.get("label")]
    inh_labels = [c.get("label", "") for c in inhibitory[:20] if c.get("label")]

    # Apply regime correction if target operates in inverted regime
    regime_note = ""
    if target_regime == "inverted":
        # Flip all polarity predictions
        for conn in all_connections:
            old_pol = conn.get("predicted_polarity")
            if old_pol == "excitatory":
                conn["predicted_polarity"] = "inhibitory"
                conn["effective_strength"] = -abs(conn.get("effective_strength", 0))
            elif old_pol == "inhibitory":
                conn["predicted_polarity"] = "excitatory"
                conn["effective_strength"] = abs(conn.get("effective_strength", 0))

        # Re-sort and re-split
        all_connections.sort(key=lambda c: abs(c["effective_strength"]), reverse=True)
        excitatory = [c for c in all_connections if c["predicted_polarity"] == "excitatory"][:top_k]
        inhibitory = [c for c in all_connections if c["predicted_polarity"] == "inhibitory"][:top_k]

        # Re-add labels (already on connections from above)
        # Update stats
        stats["excitatory_count"] = len([c for c in all_connections if c["predicted_polarity"] == "excitatory"])
        stats["inhibitory_count"] = len([c for c in all_connections if c["predicted_polarity"] == "inhibitory"])
        stats["regime_correction_applied"] = True
        stats["target_regime"] = "inverted"

        # Recompute label lists for summary
        exc_labels = [c.get("label", "") for c in excitatory[:20] if c.get("label")]
        inh_labels = [c.get("label", "") for c in inhibitory[:20] if c.get("label")]

        regime_note = "\n**Operating Regime: INVERTED** (gate_pre < 0, up_pre < 0 at max activation)\nPolarity labels have been flipped from the original weight-based predictions.\n"
    else:
        stats["target_regime"] = target_regime or "standard"

    analysis_summary = f"""## Wiring Analysis Summary for {neuron_id}
{regime_note}
**Statistics:**
- Analyzed {stats['total_upstream_neurons']:,} upstream neurons
- {stats['excitatory_count']:,} excitatory, {stats['inhibitory_count']:,} inhibitory
- Label coverage: {coverage:.0f}%

**Top Excitatory Themes** (neurons that INCREASE target activation):
{chr(10).join(f'- {l[:100]}' for l in exc_labels[:10])}

**Top Inhibitory Themes** (neurons that DECREASE target activation):
{chr(10).join(f'- {l[:100]}' for l in inh_labels[:10])}

**Questions to Consider:**
1. Do the excitatory inputs share a semantic theme? What would activate them together?
2. Do the inhibitory inputs represent contexts where this neuron should NOT fire?
3. Does this suggest the neuron's role is broader/narrower than the initial label?
4. Compare with RelP connectivity later - which of these actually influence in context?
"""

    return {
        "neuron_id": neuron_id,
        "layer": layer,
        "neuron_idx": neuron_idx,
        "stats": stats,
        "label_coverage_pct": coverage,
        "analysis_summary": analysis_summary,  # Text summary for agent
        "top_excitatory": excitatory[:30],  # Return top 30 for display
        "top_inhibitory": inhibitory[:30],
        "all_excitatory": excitatory,  # Full lists for analysis
        "all_inhibitory": inhibitory,
    }


async def tool_analyze_wiring(
    layer: int,
    neuron_idx: int,
    top_k: int = 100,
) -> dict[str, Any]:
    """Analyze weight-based wiring to identify upstream excitatory/inhibitory neurons.

    REQUIRED EARLY STEP: This tool must be called before other input phase tools.
    It computes weight-based connectivity from ALL upstream MLP neurons and predicts
    whether each would be excitatory (increases target activation) or inhibitory
    (decreases it) based on their connection to the target's up and gate channels.

    Args:
        layer: Target neuron layer
        neuron_idx: Target neuron index
        top_k: Number of top neurons to return per polarity (default 100)

    Returns:
        Dict with:
        - stats: Total neurons, polarity distribution, percentiles
        - top_excitatory: Top excitatory upstream neurons with labels
        - top_inhibitory: Top inhibitory upstream neurons with labels
        - label_coverage_pct: Percent of top neurons with NeuronDB labels

    Key insight: This reveals the "in potentia" wiring - what COULD influence
    the target neuron based on weights alone. Compare with RelP connectivity
    which shows what DOES influence it in specific contexts.
    """
    # Pass target regime if known (for inverted regime polarity correction)
    state = get_protocol_state()
    target_regime = state.operating_regime if state else None

    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        result = await client.analyze_wiring(
            layer=layer, neuron_idx=neuron_idx, top_k=top_k,
            min_layer=0, target_regime=target_regime,
        )
    else:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _CUDA_EXECUTOR,
            lambda: _sync_analyze_wiring(layer, neuron_idx, top_k, 0, target_regime),
        )

    # Update protocol state
    if state:
        stats = result.get("stats", {})

        # Auto-populate connectivity_data with top 10 upstream neurons by |weight|
        # This provides the dashboard with upstream connectivity from weight analysis
        all_upstream = result.get("all_excitatory", []) + result.get("all_inhibitory", [])
        # Sort by absolute effective strength and take top 10
        top_upstream = sorted(all_upstream, key=lambda x: abs(x.get("effective_strength", 0)), reverse=True)[:10]

        # Format for connectivity_data - preserve FULL SwiGLU data for dashboard
        upstream_for_connectivity = [
            {
                "neuron_id": u.get("neuron_id"),
                "weight": u.get("effective_strength", 0),  # Use effective strength as weight
                "label": u.get("label", ""),
                "label_source": u.get("label_source"),
                "polarity": u.get("predicted_polarity", "unknown"),
                # Full SwiGLU decomposition for wiring polarity tables
                "c_up": u.get("c_up", 0),
                "c_gate": u.get("c_gate", 0),
                "polarity_confidence": u.get("polarity_confidence", 0),
                "effective_strength": u.get("effective_strength", 0),
                "relp_confirmed": u.get("relp_confirmed"),  # Set by batch_relp_verify_connections
                "relp_strength": u.get("relp_strength"),
            }
            for u in top_upstream
        ]

        # Get current connectivity_data or initialize
        current_connectivity = state.connectivity_data or {}
        current_connectivity["upstream_neurons"] = upstream_for_connectivity

        # Mark upstream neurons exist for protocol validation
        if upstream_for_connectivity:
            state.upstream_neurons_exist = True
            print(f"[PROTOCOL] Auto-populated {len(upstream_for_connectivity)} upstream neurons from weight analysis")

        update_protocol_state(
            wiring_analysis_done=True,
            wiring_excitatory_count=stats.get("excitatory_count", 0),
            wiring_inhibitory_count=stats.get("inhibitory_count", 0),
            wiring_data=result,
            connectivity_data=current_connectivity,
            connectivity_analyzed=True,
        )
        print(f"[PROTOCOL] Wiring analysis complete: {stats.get('excitatory_count', 0)} excitatory, "
              f"{stats.get('inhibitory_count', 0)} inhibitory upstream neurons identified")

    return result


def _sync_analyze_output_wiring(
    layer: int,
    neuron_idx: int,
    top_k: int = 100,
    max_layer: int | None = None,
    include_logits: bool = True,
    top_logits: int = 50,
) -> dict[str, Any]:
    """Compute weight-based OUTPUT wiring analysis with polarity prediction.

    This analyzes ALL downstream MLP neurons by computing how the target neuron's
    output connects to their inputs. For SwiGLU MLPs:

        downstream_activation = SiLU(gate_pre) * up_pre

    where gate_pre and up_pre depend on target neuron output via:
        c_up = downstream_e_up · target_d   (contribution to downstream's up channel)
        c_gate = downstream_e_gate · target_d   (contribution to downstream's gate channel)

    Polarity prediction (same logic as input wiring):
        - Both positive: EXCITATORY (target increases downstream activation)
        - Both negative: INHIBITORY (target decreases downstream activation)
        - Mixed signs: MIXED (context-dependent)

    This reveals "in potentia" downstream connectivity - what the target neuron
    COULD influence based on weights alone. Compare with RelP connectivity which
    shows what it DOES influence in specific contexts.

    Note: Validated experimentally - steering the target neuron causes massive
    activation changes in the top excitatory downstream neurons (avg +6000%).

    Args:
        layer: Target neuron layer
        neuron_idx: Target neuron index
        top_k: Number of top neurons to return per polarity
        max_layer: Maximum layer to analyze (default: all layers)
        include_logits: Whether to compute direct logit projections
        top_logits: Number of top logit projections to return

    Returns:
        Dict with downstream excitatory/inhibitory neurons, stats, and logit projections.
    """
    neuron_id = f"L{layer}/N{neuron_idx}"
    print(f"  [OutputWiring] Analyzing downstream connections from {neuron_id}")

    model, tokenizer = get_model_and_tokenizer()
    config = get_model_config()

    n_layers = getattr(config, "num_layers", 32)
    if max_layer is None:
        max_layer = n_layers - 1
    max_layer = min(max_layer, n_layers - 1)  # Clamp to valid range (0-indexed)

    # Get target neuron's output projection (down_proj row)
    target_mlp = model.model.layers[layer].mlp
    target_d = target_mlp.down_proj.weight[:, neuron_idx]  # stay in native dtype

    # Compute connectivity to all downstream neurons using vectorized ops.
    # Collect per-layer tensors, then merge and top-k at the end to avoid
    # materializing 1.5M+ Python dicts (which takes 20+ min on large models).
    all_layers = []
    all_c_up = []
    all_c_gate = []
    total_exc = 0
    total_inh = 0

    n_intermediate = None

    for down_layer in range(layer + 1, max_layer + 1):
        downstream_mlp = model.model.layers[down_layer].mlp

        # Matmul in model's native dtype (avoids .float() copy — saves ~1GB/layer)
        c_up_all = torch.matmul(downstream_mlp.up_proj.weight, target_d)    # [intermediate]
        c_gate_all = torch.matmul(downstream_mlp.gate_proj.weight, target_d)  # [intermediate]

        if n_intermediate is None:
            n_intermediate = c_up_all.shape[0]

        # Vectorized polarity classification
        both_pos = (c_up_all > 0) & (c_gate_all > 0)
        both_neg = (c_up_all < 0) & (c_gate_all < 0)
        total_exc += both_pos.sum().item()
        total_inh += both_neg.sum().item()

        # Compute effective strength: |c_up|+|c_gate|, signed by polarity
        c_combined = c_up_all.abs() + c_gate_all.abs()
        effective = torch.where(both_pos, c_combined,
                   torch.where(both_neg, -c_combined,
                   torch.zeros_like(c_combined)))

        all_layers.append(down_layer)
        all_c_up.append(c_up_all.cpu())
        all_c_gate.append(c_gate_all.cpu())

    # Now find top_k excitatory and inhibitory across all layers
    # Stack per-layer results: [n_layers, n_intermediate]
    n_layers_down = len(all_layers)
    if n_layers_down == 0 or n_intermediate is None:
        all_connections = []
    else:
        c_up_stacked = torch.stack(all_c_up)      # [n_layers, intermediate]
        c_gate_stacked = torch.stack(all_c_gate)

        both_pos = (c_up_stacked > 0) & (c_gate_stacked > 0)
        both_neg = (c_up_stacked < 0) & (c_gate_stacked < 0)
        c_combined = c_up_stacked.abs() + c_gate_stacked.abs()
        effective = torch.where(both_pos, c_combined,
                   torch.where(both_neg, -c_combined,
                   torch.zeros_like(c_combined)))

        # Top-k excitatory (largest positive effective)
        eff_flat = effective.reshape(-1)
        k_exc = min(top_k, (eff_flat > 0).sum().item())
        k_inh = min(top_k, (eff_flat < 0).sum().item())

        all_connections = []

        def _idx_to_conn(flat_idx):
            li = flat_idx // n_intermediate
            ni = flat_idx % n_intermediate
            dl = all_layers[li]
            cu = c_up_stacked[li, ni].item()
            cg = c_gate_stacked[li, ni].item()
            cc = abs(cu) + abs(cg)
            if cu > 0 and cg > 0:
                pol, conf = "excitatory", min(abs(cu), abs(cg)) / max(abs(cu), abs(cg)) if max(abs(cu), abs(cg)) > 0 else 0
                eff = cc
            elif cu < 0 and cg < 0:
                pol, conf = "inhibitory", min(abs(cu), abs(cg)) / max(abs(cu), abs(cg)) if max(abs(cu), abs(cg)) > 0 else 0
                eff = -cc
            else:
                pol, conf, eff = "mixed", 0.5, 0
            return {
                "layer": dl, "neuron": int(ni), "neuron_id": f"L{dl}/N{ni}",
                "c_up": cu, "c_gate": cg, "c_combined": cc,
                "predicted_polarity": pol, "polarity_confidence": conf,
                "effective_strength": eff,
                "relp_confirmed": None, "relp_strength": None,
            }

        if k_exc > 0:
            top_exc_vals, top_exc_idx = torch.topk(eff_flat, k_exc)
            for idx in top_exc_idx.tolist():
                all_connections.append(_idx_to_conn(idx))

        if k_inh > 0:
            top_inh_vals, top_inh_idx = torch.topk(-eff_flat, k_inh)  # negate to find most negative
            for idx in top_inh_idx.tolist():
                conn = _idx_to_conn(idx)
                if conn["predicted_polarity"] == "inhibitory":
                    all_connections.append(conn)

        # Sort by absolute effective strength
        all_connections.sort(key=lambda c: abs(c["effective_strength"]), reverse=True)

    # Separate by polarity and take top-k of each
    excitatory = [c for c in all_connections if c["predicted_polarity"] == "excitatory"][:top_k]
    inhibitory = [c for c in all_connections if c["predicted_polarity"] == "inhibitory"][:top_k]

    total_mixed = n_layers_down * n_intermediate - total_exc - total_inh if n_intermediate else 0
    print(f"  [OutputWiring] Found {total_exc:,} excitatory, {total_inh:,} inhibitory connections")

    # Compute statistics from vectorized data
    if n_layers_down > 0 and n_intermediate:
        c_combined_flat = (c_up_stacked.abs() + c_gate_stacked.abs()).reshape(-1)
        stats = {
            "total_downstream_neurons": n_layers_down * n_intermediate,
            "excitatory_count": total_exc,
            "inhibitory_count": total_inh,
            "mixed_count": total_mixed,
            "c_combined_percentiles": {
                "90": float(torch.quantile(c_combined_flat.float(), 0.90).item()),
                "95": float(torch.quantile(c_combined_flat.float(), 0.95).item()),
                "99": float(torch.quantile(c_combined_flat.float(), 0.99).item()),
            },
        }
        del c_up_stacked, c_gate_stacked, c_combined_flat
    else:
        stats = {
            "total_downstream_neurons": 0,
            "excitatory_count": 0, "inhibitory_count": 0, "mixed_count": 0,
            "c_combined_percentiles": {"90": 0, "95": 0, "99": 0},
        }

    # Compute direct logit projections (no .float() copy — use native dtype)
    logit_projections = []
    if include_logits:
        print("  [OutputWiring] Computing direct logit projections...")
        lm_head = model.lm_head.weight  # [vocab_size, hidden_dim] — stay in native dtype

        # logit_contribution[i] = lm_head[i, :] · target_d
        logit_contributions = torch.matmul(lm_head, target_d)  # [vocab_size]

        # Get top positive and negative projections
        top_pos_indices = torch.topk(logit_contributions, top_logits).indices
        top_neg_indices = torch.topk(-logit_contributions, top_logits).indices

        for idx in top_pos_indices:
            idx = idx.item()
            token = tokenizer.decode([idx])
            logit_projections.append({
                "token_id": idx,
                "token": token,
                "projection": logit_contributions[idx].item(),
                "direction": "promotes",
            })

        for idx in top_neg_indices:
            idx = idx.item()
            token = tokenizer.decode([idx])
            logit_projections.append({
                "token_id": idx,
                "token": token,
                "projection": logit_contributions[idx].item(),
                "direction": "suppresses",
            })

        # Sort by absolute projection
        logit_projections.sort(key=lambda x: abs(x["projection"]), reverse=True)

    # Get labels for top downstream neurons
    all_top_ids = [c["neuron_id"] for c in excitatory + inhibitory]
    labels = batch_get_neuron_labels_with_fallback(all_top_ids) if all_top_ids else {}

    # Add labels to connections
    labeled_count = 0
    for c in excitatory + inhibitory:
        label_info = labels.get(c["neuron_id"], {})
        c["label"] = label_info.get("label", "")[:150]  # Truncate
        c["label_source"] = label_info.get("source", None)
        if c["label"]:
            labeled_count += 1

    coverage = labeled_count / len(all_top_ids) * 100 if all_top_ids else 0

    # Generate analysis summary for the agent
    exc_labels = [c.get("label", "") for c in excitatory[:20] if c.get("label")]
    inh_labels = [c.get("label", "") for c in inhibitory[:20] if c.get("label")]

    analysis_summary = f"""## Output Wiring Analysis Summary for {neuron_id}

**Statistics:**
- Analyzed {stats['total_downstream_neurons']:,} downstream neurons (layers {layer+1}-{max_layer})
- {stats['excitatory_count']:,} excitatory, {stats['inhibitory_count']:,} inhibitory
- Label coverage: {coverage:.0f}%

**Top Excitatory Downstream Themes** (neurons this neuron ACTIVATES):
{chr(10).join(f'- {l[:100]}' for l in exc_labels[:10])}

**Top Inhibitory Downstream Themes** (neurons this neuron SUPPRESSES):
{chr(10).join(f'- {l[:100]}' for l in inh_labels[:10])}

**Direct Logit Projections** (top tokens promoted/suppressed):
- Promotes: {', '.join(f"'{p['token']}'" for p in logit_projections[:5] if p['direction'] == 'promotes')}
- Suppresses: {', '.join(f"'{p['token']}'" for p in logit_projections[:5] if p['direction'] == 'suppresses')}

**Key Insight:** This reveals "in potentia" downstream wiring - what this neuron COULD
influence based on weights alone. These connections are validated by steering experiments
(avg +6000% activation in top excitatory targets). Compare with RelP connectivity which
shows what actually happens in specific contexts.

**Note:** Polarity predictions for downstream neurons assume they operate in the standard
SwiGLU regime (gate_pre > 0, up_pre > 0). If a downstream neuron operates in an
inverted regime, its actual response may differ from the prediction.
"""

    return {
        "neuron_id": neuron_id,
        "layer": layer,
        "neuron_idx": neuron_idx,
        "stats": stats,
        "label_coverage_pct": coverage,
        "analysis_summary": analysis_summary,
        "top_excitatory": excitatory[:30],
        "top_inhibitory": inhibitory[:30],
        "all_excitatory": excitatory,
        "all_inhibitory": inhibitory,
        "logit_projections": logit_projections[:top_logits * 2] if include_logits else [],
    }


async def tool_analyze_output_wiring(
    layer: int,
    neuron_idx: int,
    top_k: int = 100,
    max_layer: int | None = None,
    include_logits: bool = True,
) -> dict[str, Any]:
    """Analyze weight-based OUTPUT wiring to identify downstream neurons this neuron influences.

    This is the symmetric counterpart to analyze_wiring() which looks at upstream inputs.
    Here we compute what downstream neurons the target neuron COULD influence based on
    weights alone ("in potentia" connectivity).

    For SwiGLU MLPs, we compute how the target's down_proj output connects to each
    downstream neuron's up_proj and gate_proj inputs, predicting excitatory/inhibitory
    polarity using the same logic as input wiring.

    IMPORTANT: These predictions are validated experimentally - steering the target
    neuron causes massive activation changes in top excitatory downstream neurons
    (average +6000% increase). However, these connections may not appear in RelP
    edge stats if the corpus didn't strongly activate the target neuron.

    Args:
        layer: Target neuron layer
        neuron_idx: Target neuron index
        top_k: Number of top neurons to return per polarity (default 100)
        max_layer: Maximum downstream layer to analyze (default: all layers)
        include_logits: Whether to compute direct logit projections (default True)

    Returns:
        Dict with:
        - stats: Total neurons, polarity distribution, percentiles
        - top_excitatory: Top downstream neurons this neuron ACTIVATES (with labels)
        - top_inhibitory: Top downstream neurons this neuron SUPPRESSES (with labels)
        - logit_projections: Direct vocabulary projections (promotes/suppresses tokens)
        - analysis_summary: Text summary for agent interpretation

    Use Cases:
        - Understand output function: What downstream circuits does this neuron feed?
        - Predict ablation effects: Which neurons will be affected by ablating this one?
        - Compare with RelP: Do the weight-predicted connections match actual context usage?
        - Identify semantic themes in downstream targets
    """
    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        result = await client.analyze_output_wiring(
            layer=layer, neuron_idx=neuron_idx, top_k=top_k,
            max_layer=max_layer, include_logits=include_logits, top_logits=50,
        )
    else:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _CUDA_EXECUTOR,
            _sync_analyze_output_wiring,
            layer,
            neuron_idx,
            top_k,
            max_layer,
            include_logits,
            50,  # top_logits
        )

    # Update protocol state
    state = get_protocol_state()
    if state:
        stats = result.get("stats", {})

        # Auto-populate connectivity_data with top 10 downstream neurons by |weight|
        # This provides the dashboard with downstream connectivity from weight analysis
        all_downstream = result.get("all_excitatory", []) + result.get("all_inhibitory", [])
        # Sort by absolute effective strength and take top 10
        top_downstream = sorted(all_downstream, key=lambda x: abs(x.get("effective_strength", 0)), reverse=True)[:10]

        # Format for connectivity_data - preserve FULL SwiGLU data for dashboard
        downstream_for_connectivity = [
            {
                "neuron_id": d.get("neuron_id"),
                "weight": d.get("effective_strength", 0),  # Use effective strength as weight
                "label": d.get("label", ""),
                "label_source": d.get("label_source"),
                "polarity": d.get("predicted_polarity", "unknown"),
                # Full SwiGLU decomposition for wiring polarity tables
                "c_up": d.get("c_up", 0),
                "c_gate": d.get("c_gate", 0),
                "polarity_confidence": d.get("polarity_confidence", 0),
                "effective_strength": d.get("effective_strength", 0),
                "relp_confirmed": d.get("relp_confirmed"),  # Set by batch_relp_verify_connections
                "relp_strength": d.get("relp_strength"),
            }
            for d in top_downstream
        ]

        # Get current connectivity_data or initialize
        current_connectivity = state.connectivity_data or {}
        current_connectivity["downstream_targets"] = downstream_for_connectivity

        # Also add logit projections to connectivity for dashboard
        logit_projections = result.get("logit_projections", [])
        if logit_projections:
            current_connectivity["logit_projections"] = logit_projections[:20]  # Top 20

        # Mark downstream neurons exist for protocol validation
        if downstream_for_connectivity:
            state.downstream_neurons_exist = True
            print(f"[PROTOCOL] Auto-populated {len(downstream_for_connectivity)} downstream neurons from weight analysis")

        update_protocol_state(
            output_wiring_done=True,
            output_wiring_excitatory_count=stats.get("excitatory_count", 0),
            output_wiring_inhibitory_count=stats.get("inhibitory_count", 0),
            output_wiring_data=result,
            connectivity_data=current_connectivity,
        )
        print(f"[PROTOCOL] Output wiring analysis complete: {stats.get('excitatory_count', 0)} excitatory, "
              f"{stats.get('inhibitory_count', 0)} inhibitory downstream neurons identified")

    return result


async def tool_save_report(
    neuron_id: str,
    report: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    """Save investigation report to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_id = neuron_id.replace("/", "_")
    report_file = output_path / f"{safe_id}_investigation.json"

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    return {"saved_to": str(report_file)}


def _normalize_projection_item(item: Any) -> dict[str, Any]:
    """Normalize an output projection item while preserving metadata.

    Takes various input formats (string, dict with different field names) and
    normalizes to a consistent format while preserving the original data.

    Args:
        item: Can be a string token or dict with token/weight info

    Returns:
        Dict with normalized fields plus metadata:
        - token: The token string
        - weight: The projection weight (0 if not available)
        - source_field: Which field the weight came from (for debugging)
        - raw_data: Original dict data if input was a dict
    """
    if not isinstance(item, dict):
        # Plain string token
        return {"token": str(item), "weight": 0, "source_field": None, "raw_data": None}

    # Dictionary input - extract and normalize
    token = item.get("token", str(item))

    # Try different field names for weight
    weight_fields = ["projection_strength", "weight", "magnitude", "strength", "score"]
    weight = 0
    source_field = None
    for field in weight_fields:
        if item.get(field) is not None:
            weight = item[field]
            source_field = field
            break

    return {
        "token": token,
        "weight": weight,
        "source_field": source_field,
        "raw_data": item,  # Preserve original for debugging
    }


def _derive_connectivity_from_wiring(protocol_state: "ProtocolState") -> dict[str, Any]:
    """Derive evidence.connectivity from wiring analysis data.

    This ensures evidence.connectivity is always consistent with wiring_analysis
    and output_wiring_analysis, eliminating format mismatches.

    Args:
        protocol_state: Current protocol state with wiring_data and output_wiring_data

    Returns:
        Dict with upstream_neurons and downstream_targets in canonical format
    """
    result = {
        "upstream_neurons": [],
        "downstream_targets": [],
    }

    # Build lookup for RelP confirmation data from the global verification results dict
    # This captures ALL neurons verified across all batch_relp_verify_connections calls,
    # regardless of whether they're in the top-10 connectivity_data
    relp_lookup = {}
    for nid, data in protocol_state.relp_verification_results.items():
        relp_lookup[nid] = (data["relp_confirmed"], data.get("relp_strength"))

    # Extract upstream from wiring_data
    if protocol_state.wiring_data:
        wiring = protocol_state.wiring_data
        # Combine excitatory and inhibitory, sort by absolute strength, take top 10
        all_upstream = wiring.get("top_excitatory", []) + wiring.get("top_inhibitory", [])
        all_upstream.sort(key=lambda x: abs(x.get("effective_strength", x.get("c_combined", 0))), reverse=True)
        for n in all_upstream[:10]:
            nid = n.get("neuron_id", "")
            relp_confirmed, relp_strength = relp_lookup.get(nid, (n.get("relp_confirmed"), n.get("relp_strength")))
            result["upstream_neurons"].append({
                "neuron_id": nid,
                "label": n.get("label", ""),
                "weight": n.get("effective_strength", 0),
                "c_up": n.get("c_up", 0),
                "c_gate": n.get("c_gate", 0),
                "polarity": n.get("predicted_polarity", "unknown"),
                "polarity_confidence": n.get("polarity_confidence", 0),
                "relp_confirmed": relp_confirmed,
                "relp_strength": relp_strength,
            })

    # Extract downstream from output_wiring_data
    if protocol_state.output_wiring_data:
        out_wiring = protocol_state.output_wiring_data
        all_downstream = out_wiring.get("top_excitatory", []) + out_wiring.get("top_inhibitory", [])
        all_downstream.sort(key=lambda x: abs(x.get("effective_strength", x.get("c_combined", 0))), reverse=True)
        for n in all_downstream[:10]:
            nid = n.get("neuron_id", "")
            relp_confirmed, relp_strength = relp_lookup.get(nid, (n.get("relp_confirmed"), n.get("relp_strength")))
            result["downstream_targets"].append({
                "neuron_id": nid,
                "label": n.get("label", ""),
                "weight": n.get("effective_strength", 0),
                "c_up": n.get("c_up", 0),
                "c_gate": n.get("c_gate", 0),
                "polarity": n.get("predicted_polarity", "unknown"),
                "polarity_confidence": n.get("polarity_confidence", 0),
                "relp_confirmed": relp_confirmed,
                "relp_strength": relp_strength,
            })

    return result


def _extract_ablation_effects(
    ablation_details: list[dict[str, Any]],
    steering_details: list[dict[str, Any]],
    ablation_promotes: list[str],
    ablation_suppresses: list[str],
) -> list[dict[str, Any]]:
    """Extract ablation effects preserving per-prompt data for dashboard display.

    Priority:
    1. ablation_details - explicit ablation results with per-prompt data
    2. steering_details - steering results contain promotes/suppresses with shifts
    3. ablation_promotes/suppresses - fallback to token names with zero shifts

    Returns list of per-prompt entries with prompt, promotes, suppresses, etc.
    """
    # Strategy 1: Use ablation_details if available - preserve per-prompt structure
    if ablation_details:
        results = []
        for detail in ablation_details:
            # Skip malformed entries (must have prompt and both promotes/suppresses)
            if not isinstance(detail, dict):
                continue
            if "prompt" not in detail:
                # Entry missing prompt - likely malformed data
                continue
            if "promotes" not in detail and "suppresses" not in detail:
                # Entry has neither effect data - skip
                continue

            # Normalize promotes/suppresses format
            promotes = []
            for item in detail.get("promotes", []):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    promotes.append([item[0], item[1]])
                elif isinstance(item, dict):
                    promotes.append([item.get("token", ""), item.get("shift", 0)])

            suppresses = []
            for item in detail.get("suppresses", []):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    suppresses.append([item[0], item[1]])
                elif isinstance(item, dict):
                    suppresses.append([item.get("token", ""), item.get("shift", 0)])

            # Preserve per-prompt entry with all fields
            results.append({
                "prompt": detail.get("prompt", ""),
                "most_affected": detail.get("most_affected", ""),
                "max_shift": detail.get("max_shift", 0),
                "promotes": promotes[:5],
                "suppresses": suppresses[:5],
            })

        if results:
            return results[:10]  # Limit to 10 entries for dashboard

    # Strategy 2: Use steering_details as ablation-like data (has per-prompt structure)
    if steering_details:
        results = []
        for steering in steering_details:
            promotes = []
            for item in steering.get("promotes", []):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    promotes.append([item[0], item[1]])

            suppresses = []
            for item in steering.get("suppresses", []):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    suppresses.append([item[0], item[1]])

            results.append({
                "prompt": steering.get("prompt", ""),
                "steering_value": steering.get("steering_value", 0),
                "max_shift": steering.get("max_shift", 0),
                "promotes": promotes[:5],
                "suppresses": suppresses[:5],
            })

        if results:
            return results[:10]

    # Strategy 3: Fallback to aggregated token names (no per-prompt data available)
    return [
        {"prompt": "[Aggregated]", "promotes": [[t, 0] for t in ablation_promotes[:5]], "suppresses": []},
        {"prompt": "[Aggregated]", "promotes": [], "suppresses": [[t, 0] for t in ablation_suppresses[:5]]},
    ]


def _extract_upstream_dependency_evidence(protocol_state) -> list[dict[str, Any]]:
    """Extract upstream dependency evidence for dashboard display."""
    results = []
    for item in protocol_state.upstream_dependency_results:
        individual = item.get("individual_ablation", {})
        for neuron_id, stats in individual.items():
            results.append({
                "upstream_neuron": neuron_id,
                "mean_change_percent": stats.get("mean_change_percent", 0),
                "effect_type": stats.get("effect_type", "unknown"),
                "n_prompts": stats.get("n_prompts", 0),
            })
    return results


def _extract_upstream_steering_evidence(protocol_state) -> list[dict[str, Any]]:
    """Extract upstream steering evidence for dashboard display."""
    results = []
    for item in protocol_state.upstream_steering_results:
        relp_comparison = item.get("relp_comparison", {})
        upstream_results = item.get("upstream_results", {})
        for neuron_id in relp_comparison:
            relp_data = relp_comparison[neuron_id]
            steer_data = upstream_results.get(neuron_id, {})
            results.append({
                "upstream_neuron": neuron_id,
                "slope": steer_data.get("slope", 0),
                "r_squared": steer_data.get("r_squared", 0),
                "effect_direction": steer_data.get("effect_direction", "unknown"),
                "relp_weight": relp_data.get("relp_weight", 0),
                "relp_sign": relp_data.get("relp_sign", "unknown"),
                "relp_sign_match": relp_data.get("signs_match", False),
            })
    return results


def _extract_category_selectivity_evidence(protocol_state) -> dict[str, Any]:
    """Extract category selectivity evidence for dashboard display."""
    if not protocol_state.category_selectivity_done:
        return {}

    # Get category data from categorized_prompts
    category_stats = {}
    for category, prompts in protocol_state.categorized_prompts.items():
        if prompts:
            activations = [p.get("activation", 0) for p in prompts if isinstance(p, dict)]
            if activations:
                category_stats[category] = {
                    "mean_activation": sum(activations) / len(activations),
                    "max_activation": max(activations),
                    "n_prompts": len(activations),
                    "n_activating": sum(1 for a in activations if a > 0.5),
                }

    return {
        "z_score_gap": protocol_state.category_selectivity_zscore_gap,
        "n_categories": protocol_state.category_selectivity_n_categories,
        "categories": category_stats,
    }


async def tool_save_structured_report(
    neuron_id: str,
    layer: int,
    neuron_idx: int,
    # Characterization
    input_function: str,
    output_function: str,
    function_type: str,  # semantic, syntactic, routing, formatting, hybrid
    summary: str,
    # Findings
    key_findings: list[str],
    open_questions: list[str],
    # Evidence summaries
    activating_patterns: list[dict[str, Any]],  # [{prompt, activation, token}, ...]
    non_activating_patterns: list[dict[str, Any]],
    ablation_promotes: list[str],  # tokens promoted
    ablation_suppresses: list[str],  # tokens suppressed
    # Connectivity
    upstream_neurons: list[dict[str, Any]],  # [{neuron_id, label, weight}, ...]
    downstream_neurons: list[dict[str, Any]],
    # Metadata
    total_experiments: int,
    output_dir: str,
    # Optional: Original labels and statistics from prior analysis
    original_output_label: str = "",
    original_input_label: str = "",
    original_output_description: str = "",
    original_input_description: str = "",
    output_projections_promote: list[Any] = None,  # List of {token, frequency, count} dicts or strings
    output_projections_suppress: list[Any] = None,  # List of {token, frequency, count} dicts or strings
    # Optional: Detailed experimental results
    ablation_details: list[dict[str, Any]] = None,  # [{prompt, promotes: [{token, shift}], suppresses: [{token, shift}]}]
    steering_details: list[dict[str, Any]] = None,  # [{prompt, steering_value, promotes: [{token, shift}], suppresses: [{token, shift}]}]
    # Optional: RelP attribution results
    relp_results: list[dict[str, Any]] = None,  # [{prompt, neuron_found, relp_score, downstream_edges, upstream_edges}]
    # Optional: Validation metrics
    baseline_zscore: float = None,  # Z-score from run_baseline_comparison
    # Optional: Pre-registered hypotheses (auto-injected from registry)
    hypotheses_tested: list[dict[str, Any]] = None,
    # Optional: Visualization data (auto-injected from investigation tracker)
    categorized_prompts: dict[str, list[dict[str, Any]]] = None,  # {category: [{prompt, activation, position, token}, ...]}
    homograph_tests: list[dict[str, Any]] = None,  # [{word, contexts: [{label, example, activation, category}, ...]}]
    category_selectivity_data: Any = None,  # Dict - accumulated selectivity results
    # DEPRECATED: Overall confidence - now tracked per-hypothesis in hypotheses_tested
    confidence: str = "",  # Kept for backward compatibility, not used
    # Polarity mode: "positive" or "negative" - which firing direction this investigation covers
    polarity_mode: str = "positive",
) -> dict[str, Any]:
    """Save structured investigation report (consolidated format).

    This creates the investigation JSON which contains all data needed for dashboards.
    The agent should call this at the end of investigation with all findings.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_id = neuron_id.replace("/", "_")
    polarity_suffix = "_negative" if polarity_mode == "negative" else ""

    # =========================================================================
    # Protocol State Validation (confidence is now per-hypothesis only)
    # =========================================================================
    protocol_state = get_protocol_state()
    missing_validation = protocol_state.get_missing_validation()

    # Print protocol state summary
    print("\n[PROTOCOL] === Save Report Protocol Check ===")
    print(f"[PROTOCOL] input_phase_complete={protocol_state.input_phase_complete}, input_char_summary={repr(protocol_state.input_characterization.get('summary', 'MISSING')[:80])}")
    print(f"[PROTOCOL] output_phase_complete={protocol_state.output_phase_complete}, output_char_summary={repr(protocol_state.output_characterization.get('summary', 'MISSING')[:80])}")

    # Report missing validation steps
    if missing_validation:
        print(f"[PROTOCOL] ⚠️  Missing validation steps ({len(missing_validation)}):")
        for step in missing_validation:
            print(f"[PROTOCOL]   - {step}")
    else:
        print("[PROTOCOL] ✓ All validation steps complete")

    # Log protocol state details (Phase 0 corpus is now optional)
    print(f"[PROTOCOL] Baseline: {'✓' if protocol_state.baseline_comparison_done else '✗'} (z={protocol_state.baseline_zscore or 'N/A'})")
    print(f"[PROTOCOL] Category selectivity: {'✓' if protocol_state.category_selectivity_done else '✗'} (z-gap={protocol_state.category_selectivity_zscore_gap or 'N/A'})")
    print(f"[PROTOCOL] Dose-response: {'✓' if protocol_state.dose_response_done else '✗'} (monotonic={protocol_state.dose_response_monotonic})")
    print(f"[PROTOCOL] RelP runs: {protocol_state.relp_runs} (pos_ctrl={'✓' if protocol_state.relp_positive_control else '✗'}, neg_ctrl={'✓' if protocol_state.relp_negative_control else '✗'})")
    print(f"[PROTOCOL] Hypotheses: {protocol_state.hypotheses_registered} registered, {protocol_state.hypotheses_updated} updated")
    print(f"[PROTOCOL] Batch ablation: {'✓' if protocol_state.batch_ablation_done else '✗ (REQUIRED)'}")
    upstream_ablation_status = '✓' if protocol_state.upstream_dependency_tested else ('✗ (REQUIRED)' if protocol_state.upstream_neurons_exist else 'N/A (no upstream neurons)')
    print(f"[PROTOCOL] Upstream ablation: {upstream_ablation_status}")
    upstream_steering_status = '✓' if protocol_state.upstream_steering_tested else ('✗ (REQUIRED)' if protocol_state.upstream_neurons_exist else 'N/A (no upstream neurons)')
    print(f"[PROTOCOL] Upstream steering: {upstream_steering_status}")
    downstream_status = '✓' if protocol_state.downstream_dependency_tested else ('✗ (REQUIRED)' if protocol_state.downstream_neurons_exist else 'N/A (no downstream neurons)')
    print(f"[PROTOCOL] Downstream dependency: {downstream_status}")
    print("[PROTOCOL] =========================================\n")

    # =========================================================================
    # HARD ENFORCEMENT: Required validation tools
    # =========================================================================
    enforcement_errors = []

    # 1. Baseline comparison is REQUIRED
    if not protocol_state.baseline_comparison_done:
        enforcement_errors.append(
            "BLOCKED: run_baseline_comparison() not called. "
            "You MUST run baseline comparison before saving. "
            "Call: run_baseline_comparison(prompts=[your activating prompts], n_random_neurons=30)"
        )

    # 2. Category selectivity test is REQUIRED for selectivity claims
    if not protocol_state.category_selectivity_done:
        enforcement_errors.append(
            "BLOCKED: run_category_selectivity_test() not called. "
            "You MUST run category selectivity test to verify the neuron is truly selective. "
            "Call: run_category_selectivity_test(target_domain='...', target_categories=['...'])"
        )

    # 2.5. Wiring analysis is REQUIRED (provides SwiGLU polarity predictions)
    if not protocol_state.wiring_analysis_done:
        enforcement_errors.append(
            "BLOCKED: analyze_wiring() not called. "
            "You MUST run wiring analysis to determine upstream excitatory/inhibitory neurons. "
            "Call: analyze_wiring(layer=..., neuron_idx=..., top_k=100)"
        )

    # 2.6. Output wiring analysis is REQUIRED (provides downstream targets)
    if not protocol_state.output_wiring_done:
        enforcement_errors.append(
            "BLOCKED: analyze_output_wiring() not called. "
            "You MUST run output wiring analysis to determine downstream targets and logit projections. "
            "Call: analyze_output_wiring(layer=..., neuron_idx=..., top_k=100)"
        )

    # 3. BATCH ablation is REQUIRED (tests output effects across ALL category selectivity prompts)
    if not protocol_state.batch_ablation_done:
        enforcement_errors.append(
            "BLOCKED: batch_ablate_and_generate() not called. "
            "You MUST run BATCH ablation on category selectivity prompts. Single-prompt ablation is not sufficient. "
            "Call: batch_ablate_and_generate(use_categorized_prompts=True, max_new_tokens=10)"
        )

    # 4. Downstream dependency test is REQUIRED when downstream neurons exist
    if protocol_state.downstream_neurons_exist and not protocol_state.downstream_dependency_tested:
        enforcement_errors.append(
            "BLOCKED: ablate_and_check_downstream() not called. "
            "Downstream neurons were found - you MUST test how ablation affects them across multiple token positions. "
            "Call: ablate_and_check_downstream(layer=..., neuron_idx=..., prompts=[...], max_new_tokens=10)"
        )

    # 5. Upstream ablation test is REQUIRED when upstream neurons exist
    if protocol_state.upstream_neurons_exist and not protocol_state.upstream_dependency_tested:
        enforcement_errors.append(
            "BLOCKED: batch_ablate_upstream_and_test() not called. "
            "Upstream neurons were found - you MUST test if ablating them affects target neuron activation. "
            "Call: batch_ablate_upstream_and_test(use_categorized_prompts=True)"
        )

    # 6. Upstream steering test is REQUIRED when upstream neurons exist (provides RelP comparison)
    if protocol_state.upstream_neurons_exist and not protocol_state.upstream_steering_tested:
        enforcement_errors.append(
            "BLOCKED: batch_steer_upstream_and_test() not called. "
            "Upstream neurons were found - you MUST test if steering them affects target neuron activation. "
            "This provides slopes comparable to RelP edge weights for validation. "
            "Call: batch_steer_upstream_and_test(use_categorized_prompts=True)"
        )

    # 7. BATCH steering is REQUIRED (tests steering effects across all category selectivity prompts)
    if not protocol_state.batch_steering_done:
        enforcement_errors.append(
            "BLOCKED: batch_steer_and_generate() not called. "
            "You MUST run BATCH steering on category selectivity prompts. "
            "Call: batch_steer_and_generate(use_categorized_prompts=True, steering_value=10.0)"
        )

    # 8. Minimum prompt counts for batch experiments
    if protocol_state.batch_ablation_done and protocol_state.batch_ablation_prompt_count < 50:
        enforcement_errors.append(
            f"BLOCKED: batch_ablate_and_generate() used only {protocol_state.batch_ablation_prompt_count} prompts (need ≥50). "
            "Use use_categorized_prompts=True to automatically use all activating prompts from category selectivity."
        )

    if protocol_state.batch_steering_done and protocol_state.batch_steering_prompt_count < 20:
        enforcement_errors.append(
            f"BLOCKED: batch_steer_and_generate() used only {protocol_state.batch_steering_prompt_count} prompts (need ≥20). "
            "Use use_categorized_prompts=True to automatically use all activating prompts from category selectivity."
        )

    if protocol_state.downstream_neurons_exist and protocol_state.downstream_dependency_tested and protocol_state.downstream_dependency_prompt_count < 20:
        enforcement_errors.append(
            f"BLOCKED: ablate_and_check_downstream() used only {protocol_state.downstream_dependency_prompt_count} prompts (need ≥20). "
            "Provide more prompts to get statistically meaningful downstream dependency measurements."
        )

    # 9. Anomaly phase is REQUIRED
    if not protocol_state.anomaly_phase_complete:
        enforcement_errors.append(
            "BLOCKED: complete_anomaly_phase() not called. "
            "You MUST investigate anomalies from your accumulated evidence before saving. "
            "1. Review all evidence for contradictions, surprises, unexplained patterns. "
            "2. List all anomalies you identified. "
            "3. Investigate the top 3 most interesting ones. "
            "4. Call: complete_anomaly_phase(anomalies_identified=[...], anomalies_investigated=[...])"
        )

    # 10. If baseline was run but z-score is too low, warn but allow
    if protocol_state.baseline_comparison_done and protocol_state.baseline_zscore is not None:
        if protocol_state.baseline_zscore < 2.0:
            print(f"[PROTOCOL] WARNING: Baseline z-score ({protocol_state.baseline_zscore:.2f}) < 2.0. "
                  "Effects may not be statistically meaningful.")

    # 4. If category selectivity was run but z-gap is too low, warn but allow
    if protocol_state.category_selectivity_done and protocol_state.category_selectivity_zscore_gap is not None:
        if protocol_state.category_selectivity_zscore_gap < 1.0:
            print(f"[PROTOCOL] WARNING: Category selectivity z-gap ({protocol_state.category_selectivity_zscore_gap:.2f}) < 1.0. "
                  "Neuron may not be genuinely selective for target categories.")

    # Block save if enforcement errors
    if enforcement_errors:
        error_msg = "\n".join(enforcement_errors)
        print(f"\n[PROTOCOL] === SAVE BLOCKED ===\n{error_msg}\n")

        # Build list of missing tools
        missing_tools = []
        if not protocol_state.baseline_comparison_done:
            missing_tools.append("run_baseline_comparison")
        if not protocol_state.category_selectivity_done:
            missing_tools.append("run_category_selectivity_test")
        if not protocol_state.batch_ablation_done:
            missing_tools.append("batch_ablate_and_generate(use_categorized_prompts=True)")
        elif protocol_state.batch_ablation_prompt_count < 50:
            missing_tools.append(f"batch_ablate_and_generate (used {protocol_state.batch_ablation_prompt_count} prompts, need ≥50)")
        if not protocol_state.batch_steering_done:
            missing_tools.append("batch_steer_and_generate(use_categorized_prompts=True)")
        elif protocol_state.batch_steering_prompt_count < 20:
            missing_tools.append(f"batch_steer_and_generate (used {protocol_state.batch_steering_prompt_count} prompts, need ≥20)")
        if protocol_state.upstream_neurons_exist and not protocol_state.upstream_dependency_tested:
            missing_tools.append("batch_ablate_upstream_and_test")
        if protocol_state.upstream_neurons_exist and not protocol_state.upstream_steering_tested:
            missing_tools.append("batch_steer_upstream_and_test")
        if protocol_state.downstream_neurons_exist and not protocol_state.downstream_dependency_tested:
            missing_tools.append("ablate_and_check_downstream")
        elif protocol_state.downstream_neurons_exist and protocol_state.downstream_dependency_prompt_count < 20:
            missing_tools.append(f"ablate_and_check_downstream (used {protocol_state.downstream_dependency_prompt_count} prompts, need ≥20)")

        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "error": "validation_required",
                    "message": error_msg,
                    "missing_tools": missing_tools
                })
            }]
        }

    # Inject baseline zscore from protocol state if not provided
    if baseline_zscore is None and protocol_state.baseline_zscore is not None:
        baseline_zscore = protocol_state.baseline_zscore

    # Auto-inject hypothesis registry if not provided
    if hypotheses_tested is None:
        hypotheses_tested = get_hypothesis_registry()

    # Auto-inject RelP results if not provided or empty
    if not relp_results:
        relp_results = get_relp_registry()

    # ALWAYS use connectivity from protocol_state.connectivity_data (populated by wiring tools)
    # WIRING ANALYSIS IS THE ONLY SOURCE - defines the hypothesis space
    # RelP is supplemental/validation only, NOT used for connectivity
    # IMPORTANT: Override any agent-provided values with wiring data
    conn_data = protocol_state.connectivity_data or {}
    if conn_data.get("upstream_neurons"):
        # Always use wiring-based connectivity, even if agent provided different values
        if upstream_neurons and upstream_neurons != conn_data["upstream_neurons"]:
            print("  [Override] Replacing agent-provided upstream neurons with wiring analysis data")
        upstream_neurons = conn_data["upstream_neurons"]
        print(f"  [Auto-inject] {len(upstream_neurons)} upstream neurons from wiring analysis")
    if conn_data.get("downstream_neurons"):
        # Always use wiring-based connectivity, even if agent provided different values
        if downstream_neurons and downstream_neurons != conn_data["downstream_neurons"]:
            print("  [Override] Replacing agent-provided downstream neurons with wiring analysis data")
        downstream_neurons = conn_data["downstream_neurons"]
        print(f"  [Auto-inject] {len(downstream_neurons)} downstream neurons from wiring analysis")

    # WARN if wiring connectivity not available (no fallback to RelP)
    if not upstream_neurons:
        print("  [WARNING] No upstream neurons - analyze_wiring should be called before save_structured_report")
    if not downstream_neurons:
        print("  [WARNING] No downstream neurons - analyze_output_wiring should be called before save_structured_report")

    # Enrich connectivity labels from ground truth file
    if upstream_neurons or downstream_neurons:
        all_neuron_ids = [n["neuron_id"] for n in upstream_neurons + downstream_neurons]
        label_results = batch_get_neuron_labels_with_fallback(all_neuron_ids)
        for n in upstream_neurons + downstream_neurons:
            nid = n["neuron_id"]
            if nid in label_results and label_results[nid].get("found"):
                n["label"] = label_results[nid].get("label", "")

    # Auto-inject output projections from cache if not provided
    # This ensures projections are saved even if agent didn't explicitly pass them
    if not output_projections_promote and not output_projections_suppress:
        cached_projections = get_output_projections_cache()
        if cached_projections:
            output_projections_promote = cached_projections.get("promoted", [])
            output_projections_suppress = cached_projections.get("suppressed", [])
            print(f"  [Auto-inject] Output projections from cache: {len(output_projections_promote)} promoted, {len(output_projections_suppress)} suppressed")

    # Auto-generate key_findings if empty - based on evidence and protocol state
    if not key_findings:
        auto_findings = []
        # Add finding based on activation patterns
        if activating_patterns:
            top_act = activating_patterns[0].get("activation", 0)
            pattern_count = len(activating_patterns)
            auto_findings.append(f"Found {pattern_count} activating patterns (max activation: {top_act:.2f})")
        # Add finding based on RelP
        if protocol_state.relp_positive_control:
            relp_found = sum(1 for r in (relp_results or []) if r.get("neuron_found"))
            auto_findings.append(f"RelP confirmed neuron in {relp_found} causal pathway(s)")
        # Add finding based on output projections
        if output_projections_promote:
            top_tokens = [p.get("token", "") for p in output_projections_promote[:3] if isinstance(p, dict)]
            if top_tokens:
                auto_findings.append(f"Output strongly promotes: {', '.join(top_tokens)}")
        # Add finding based on steering
        if steering_details:
            auto_findings.append(f"Causal effects verified via {len(steering_details)} steering experiment(s)")
        key_findings = auto_findings if auto_findings else ["Investigation completed - see evidence section for details"]
        print(f"  [Auto-inject] Generated {len(key_findings)} key findings")

    # NOTE: Open questions are generated by the scientist agent during investigation.
    # We trust the agent's questions rather than auto-generating generic ones here.
    # Validation gaps are tracked separately in protocol_validation.missing_validation
    # If the agent didn't provide questions, that's fine - leave empty rather than
    # overwriting with generic defaults like "polysemantic behavior?" or "circuit role?"

    # Build investigation JSON
    # NOTE: Overall confidence removed - confidence is now per-hypothesis only (in hypotheses_tested)
    investigation = {
        "neuron_id": neuron_id,
        "layer": layer,
        "neuron_idx": neuron_idx,
        "polarity_mode": polarity_mode,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "total_experiments": total_experiments,
        "seed_used": get_seed(),  # Log seed for reproducibility
        "characterization": {
            "input_function": input_function,
            "output_function": output_function,
            "function_type": function_type,
            "final_hypothesis": summary,
        },
        "key_findings": key_findings,
        "open_questions": open_questions,
        "hypotheses_tested": hypotheses_tested,  # Pre-registered hypotheses with outcomes
        # Protocol validation state for peer review
        "protocol_validation": {
            "phase0_corpus_queried": protocol_state.phase0_corpus_queried,
            "phase0_graph_count": protocol_state.phase0_graph_count,
            "baseline_comparison_done": protocol_state.baseline_comparison_done,
            "baseline_zscore": protocol_state.baseline_zscore,
            # Wiring analysis tracking (REQUIRED for SwiGLU polarity predictions)
            "wiring_analysis_done": protocol_state.wiring_analysis_done,
            "output_wiring_done": protocol_state.output_wiring_done,
            "category_selectivity_done": protocol_state.category_selectivity_done,
            "category_selectivity_zscore_gap": protocol_state.category_selectivity_zscore_gap,
            "category_selectivity_n_categories": protocol_state.category_selectivity_n_categories,
            "dose_response_done": protocol_state.dose_response_done,
            "dose_response_monotonic": protocol_state.dose_response_monotonic,
            "dose_response_kendall_tau": protocol_state.dose_response_kendall_tau,
            "relp_runs": protocol_state.relp_runs,
            "relp_positive_control": protocol_state.relp_positive_control,
            "relp_negative_control": protocol_state.relp_negative_control,
            "hypotheses_registered": protocol_state.hypotheses_registered,
            "hypotheses_updated": protocol_state.hypotheses_updated,
            "missing_validation": missing_validation,
            # V4 Phase tracking
            "input_phase_complete": protocol_state.input_phase_complete,
            "output_phase_complete": protocol_state.output_phase_complete,
            "upstream_dependency_tested": protocol_state.upstream_dependency_tested,
            "upstream_steering_tested": protocol_state.upstream_steering_tested,
            "upstream_neurons_exist": protocol_state.upstream_neurons_exist,
            "batch_ablation_done": protocol_state.batch_ablation_done,
            "multi_token_ablation_done": protocol_state.multi_token_ablation_done,
            "downstream_dependency_tested": protocol_state.downstream_dependency_tested,
            "downstream_neurons_exist": protocol_state.downstream_neurons_exist,
            # V5 Intelligent steering tracking
            "batch_steering_done": protocol_state.batch_steering_done,
            "intelligent_steering_runs": protocol_state.intelligent_steering_runs,
            "intelligent_steering_total_prompts": protocol_state.intelligent_steering_total_prompts,
            # V5 Anomaly investigation tracking
            "anomaly_phase_complete": protocol_state.anomaly_phase_complete,
            "anomalies_identified_count": len(protocol_state.anomalies_identified),
            "anomalies_investigated_count": len(protocol_state.anomalies_investigated),
            # SwiGLU operating regime
            "operating_regime": protocol_state.operating_regime,
            "regime_confidence": protocol_state.regime_confidence,
        },
        # V4 Phase characterizations (if used)
        "input_characterization": protocol_state.input_characterization if protocol_state.input_phase_complete else None,
        "output_characterization": protocol_state.output_characterization if protocol_state.output_phase_complete else None,
        "evidence": {
            "activating_prompts": activating_patterns[:20],
            "non_activating_prompts": non_activating_patterns[:10],
            "ablation_effects": _extract_ablation_effects(
                ablation_details, steering_details, ablation_promotes, ablation_suppresses
            ),
            # Connectivity is derived from wiring_analysis for consistency
            # Dashboard should use wiring_analysis/output_wiring_analysis as primary source
            "connectivity": _derive_connectivity_from_wiring(protocol_state),
            # Show ALL relp_results, sorted to prioritize corpus evidence (neuron_found=True first)
            "relp_results": sorted(
                relp_results or [],
                key=lambda x: (not x.get("neuron_found", False), x.get("source") != "corpus")
            ),
            # V5: Upstream dependency evidence (from batch_ablate_upstream_and_test)
            "upstream_dependency": _extract_upstream_dependency_evidence(protocol_state),
            # V5: Upstream steering evidence (from batch_steer_upstream_and_test)
            "upstream_steering": _extract_upstream_steering_evidence(protocol_state),
            # V5: Category selectivity evidence
            "category_selectivity": _extract_category_selectivity_evidence(protocol_state),
        },
        "relp_results": relp_results or [],  # Full RelP results
        "steering_results": steering_details or [],  # Individual steering results
        "dose_response_results": [],  # Populated by protocol state below
        "output_projections": {
            "promote": [
                _normalize_projection_item(item) for item in (output_projections_promote or [])
            ],
            "suppress": [
                _normalize_projection_item(item) for item in (output_projections_suppress or [])
            ],
        },
    }

    # Add dose-response results from protocol state if available
    if protocol_state.dose_response_results:
        investigation["dose_response_results"] = protocol_state.dose_response_results

    # Add V4 multi-token results from protocol state if available
    if protocol_state.multi_token_ablation_results:
        investigation["multi_token_ablation_results"] = protocol_state.multi_token_ablation_results
    if protocol_state.multi_token_steering_results:
        investigation["multi_token_steering_results"] = protocol_state.multi_token_steering_results
    if protocol_state.upstream_dependency_results:
        investigation["upstream_dependency_results"] = protocol_state.upstream_dependency_results
    if protocol_state.upstream_steering_results:
        investigation["upstream_steering_results"] = protocol_state.upstream_steering_results
    if protocol_state.downstream_dependency_results:
        investigation["downstream_dependency_results"] = protocol_state.downstream_dependency_results

    # Add anomaly investigation results (V5)
    if protocol_state.anomaly_phase_complete:
        investigation["anomaly_investigation"] = {
            "anomalies_identified": protocol_state.anomalies_identified,
            "anomalies_investigated": protocol_state.anomalies_investigated,
        }

    # Add wiring analysis data (weight-based upstream connectivity with polarity)
    print(f"[DEBUG] Wiring data exists: {bool(protocol_state.wiring_data)}, done: {protocol_state.wiring_analysis_done}")
    if protocol_state.wiring_data:
        wiring = protocol_state.wiring_data
        n_exc = len(wiring.get("top_excitatory", []))
        n_inh = len(wiring.get("top_inhibitory", []))
        print(f"[DEBUG] Saving wiring_analysis: {n_exc} excitatory, {n_inh} inhibitory neurons")
        investigation["wiring_analysis"] = {
            "stats": wiring.get("stats", {}),
            "label_coverage_pct": wiring.get("label_coverage_pct", 0),
            "analysis_summary": wiring.get("analysis_summary", ""),  # Text summary
            "top_excitatory": wiring.get("top_excitatory", [])[:15],  # Top 15 for JSON
            "top_inhibitory": wiring.get("top_inhibitory", [])[:15],
        }
    else:
        print("[WARNING] No wiring_data in protocol state - wiring_analysis will be missing from investigation!")

    # Enrich wiring_analysis with RelP confirmation data from global verification results
    if "wiring_analysis" in investigation and protocol_state.relp_verification_results:
        for n in investigation["wiring_analysis"].get("top_excitatory", []) + investigation["wiring_analysis"].get("top_inhibitory", []):
            nid = n.get("neuron_id", "")
            if nid in protocol_state.relp_verification_results:
                rv = protocol_state.relp_verification_results[nid]
                n["relp_confirmed"] = rv["relp_confirmed"]
                n["relp_strength"] = rv.get("relp_strength")

    # Add output wiring analysis data (weight-based downstream connectivity)
    print(f"[DEBUG] Output wiring data exists: {bool(protocol_state.output_wiring_data)}, done: {protocol_state.output_wiring_done}")
    if protocol_state.output_wiring_data:
        out_wiring = protocol_state.output_wiring_data
        n_exc = len(out_wiring.get("top_excitatory", []))
        n_inh = len(out_wiring.get("top_inhibitory", []))
        print(f"[DEBUG] Saving output_wiring_analysis: {n_exc} excitatory, {n_inh} inhibitory neurons")
        investigation["output_wiring_analysis"] = {
            "stats": out_wiring.get("stats", {}),
            "label_coverage_pct": out_wiring.get("label_coverage_pct", 0),
            "analysis_summary": out_wiring.get("analysis_summary", ""),  # Text summary
            "top_excitatory": out_wiring.get("top_excitatory", [])[:15],  # Top 15 for JSON
            "top_inhibitory": out_wiring.get("top_inhibitory", [])[:15],
            "logit_projections": out_wiring.get("logit_projections", [])[:20],  # Top 20 logit projections
        }
    else:
        print("[WARNING] No output_wiring_data in protocol state - output_wiring_analysis will be missing from investigation!")

    # Enrich output_wiring_analysis with RelP confirmation data from global verification results
    if "output_wiring_analysis" in investigation and protocol_state.relp_verification_results:
        for n in investigation["output_wiring_analysis"].get("top_excitatory", []) + investigation["output_wiring_analysis"].get("top_inhibitory", []):
            nid = n.get("neuron_id", "")
            if nid in protocol_state.relp_verification_results:
                rv = protocol_state.relp_verification_results[nid]
                n["relp_confirmed"] = rv["relp_confirmed"]
                n["relp_strength"] = rv.get("relp_strength")

    # Add SwiGLU operating regime data
    if protocol_state.operating_regime:
        investigation["operating_regime"] = protocol_state.operating_regime
        investigation["regime_confidence"] = protocol_state.regime_confidence
        investigation["regime_data"] = protocol_state.regime_data
        investigation["firing_sign_stats"] = protocol_state.firing_sign_stats

    # Add visualization data if provided (from investigation tracker)
    if categorized_prompts:
        investigation["categorized_prompts"] = categorized_prompts
    if homograph_tests:
        investigation["homograph_tests"] = homograph_tests
    if category_selectivity_data:
        # Store as single accumulated dict (legacy list format handled by from_dict)
        if isinstance(category_selectivity_data, list):
            investigation["category_selectivity_data"] = merge_selectivity_runs(category_selectivity_data)
        else:
            investigation["category_selectivity_data"] = category_selectivity_data

    # Save investigation (with polarity suffix for negative mode)
    inv_file = output_path / f"{safe_id}{polarity_suffix}_investigation.json"
    with open(inv_file, "w") as f:
        json.dump(investigation, f, indent=2)

    # NOTE: HTML dashboards are generated separately by the PI agent using dashboard_agent_v2
    # and stored in neuron_reports/html/. The V2 agent reads directly from investigation.json.

    return {
        "saved_investigation": str(inv_file),
        "summary": summary[:100],
    }


# =============================================================================
# RelP Attribution Tool
# =============================================================================

# Global RelP attributor cache - keyed by (tau, k)
_RELP_ATTRIBUTORS = {}

# Global RelP results registry - auto-accumulates all RelP runs
_RELP_REGISTRY = []


def clear_relp_registry():
    """Clear the RelP registry (call at start of new investigation)."""
    global _RELP_REGISTRY
    _RELP_REGISTRY = []


def init_relp_registry(prior_results: list[dict[str, Any]]):
    """Initialize RelP registry from prior investigation results.

    This preserves RelP results across revision iterations, allowing the agent
    to access prior RelP data without re-running attributions.

    Args:
        prior_results: List of RelP result dicts from prior_investigation.relp_results
    """
    global _RELP_REGISTRY
    _RELP_REGISTRY = []

    for r in prior_results:
        # Normalize the structure to match what tool_run_relp produces
        _RELP_REGISTRY.append({
            "prompt": r.get("prompt", ""),
            "tau": r.get("tau", 0.01),
            "neuron_found": r.get("neuron_found", False),
            "in_causal_pathway": r.get("in_causal_pathway", False),
            "neuron_relp_score": r.get("relp_score") or r.get("neuron_relp_score"),
            "influence": r.get("influence"),
            "source": r.get("source", "prior"),  # Mark as from prior iteration
            "graph_path": r.get("graph_path"),
            "upstream_edges": r.get("upstream_edges", []),
            "downstream_edges": r.get("downstream_edges", []),
            "graph_stats": r.get("graph_stats"),
        })

    print(f"  [RelP Registry] Initialized with {len(_RELP_REGISTRY)} prior results")


def get_relp_registry() -> list[dict[str, Any]]:
    """Get all accumulated RelP results."""
    return _RELP_REGISTRY.copy()


# Global output projections cache - stores results from get_output_projections calls
_OUTPUT_PROJECTIONS_CACHE: dict[str, Any] | None = None


def clear_output_projections_cache():
    """Clear the output projections cache (call at start of new investigation)."""
    global _OUTPUT_PROJECTIONS_CACHE
    _OUTPUT_PROJECTIONS_CACHE = None


def get_output_projections_cache() -> dict[str, Any] | None:
    """Get cached output projections, if any."""
    return _OUTPUT_PROJECTIONS_CACHE


def init_output_projections_cache(prior_projections: dict[str, Any]):
    """Initialize output projections cache from prior investigation.

    Args:
        prior_projections: Output projections dict from prior investigation
                          (from NeuronInvestigation.output_projections)
    """
    global _OUTPUT_PROJECTIONS_CACHE
    if prior_projections and (prior_projections.get("promote") or prior_projections.get("suppress")):
        _OUTPUT_PROJECTIONS_CACHE = prior_projections


def _store_output_projections(result: dict[str, Any]):
    """Store output projections result in cache for later auto-injection."""
    global _OUTPUT_PROJECTIONS_CACHE
    _OUTPUT_PROJECTIONS_CACHE = {
        "promoted": result.get("promoted", []),
        "suppressed": result.get("suppressed", []),
        "stats": result.get("stats", {}),
    }


def lookup_relp_result(prompt: str, tau: float = None) -> dict[str, Any] | None:
    """Look up a cached RelP result by prompt (and optionally tau).

    Args:
        prompt: The prompt to look up
        tau: Optional tau value to match exactly

    Returns:
        The cached RelP result dict, or None if not found
    """
    for r in _RELP_REGISTRY:
        if r.get("prompt") == prompt:
            if tau is None or r.get("tau") == tau:
                return r
    return None


# =============================================================================
# Activation Cache - persists activation values across iterations
# =============================================================================

# Global activation cache - maps prompt -> activation result
_ACTIVATION_CACHE: dict[str, dict[str, Any]] = {}


def clear_activation_cache():
    """Clear the activation cache (call at start of new investigation)."""
    global _ACTIVATION_CACHE
    _ACTIVATION_CACHE = {}


def init_activation_cache(
    activating_prompts: list[dict[str, Any]],
    non_activating_prompts: list[dict[str, Any]],
):
    """Initialize activation cache from prior investigation results.

    This preserves activation values across revision iterations, allowing the agent
    to look up prior results without re-running activation tests.

    Args:
        activating_prompts: List of activating prompt dicts from prior investigation
        non_activating_prompts: List of non-activating prompt dicts from prior investigation
    """
    global _ACTIVATION_CACHE
    _ACTIVATION_CACHE = {}

    for p in activating_prompts:
        prompt = p.get("prompt", "")
        if prompt:
            _ACTIVATION_CACHE[prompt] = {
                "activation": p.get("activation", 0),
                "max_position": p.get("max_position") or p.get("position"),
                "max_token": p.get("max_token") or p.get("token"),
                "is_activating": True,
                "source": "prior",
            }

    for p in non_activating_prompts:
        prompt = p.get("prompt", "")
        if prompt:
            _ACTIVATION_CACHE[prompt] = {
                "activation": p.get("activation", 0),
                "max_position": p.get("max_position") or p.get("position"),
                "max_token": p.get("max_token") or p.get("token"),
                "is_activating": False,
                "source": "prior",
            }

    print(f"  [Activation Cache] Initialized with {len(_ACTIVATION_CACHE)} prior results")


def get_cached_activation(prompt: str) -> dict[str, Any] | None:
    """Look up a cached activation result by prompt.

    Args:
        prompt: The prompt to look up

    Returns:
        Dict with activation, max_position, max_token, is_activating, source
        or None if not found
    """
    return _ACTIVATION_CACHE.get(prompt)


def cache_activation_result(prompt: str, result: dict[str, Any], is_activating: bool):
    """Cache an activation result for later lookup.

    Args:
        prompt: The prompt that was tested
        result: The result dict from tool_test_activation
        is_activating: Whether this was classified as activating
    """
    global _ACTIVATION_CACHE
    _ACTIVATION_CACHE[prompt] = {
        "activation": result.get("activation", 0),
        "max_position": result.get("max_position"),
        "max_token": result.get("max_token"),
        "is_activating": is_activating,
        "source": "current",
    }


def get_activation_cache() -> dict[str, dict[str, Any]]:
    """Get the full activation cache."""
    return _ACTIVATION_CACHE.copy()


# =============================================================================
# Mean Activation Cache - for mean ablation
# =============================================================================

# Global cache for neuron mean activations (for mean ablation)
# Maps (layer, neuron_idx) -> mean_activation
_MEAN_ACTIVATION_CACHE: dict[tuple[int, int], float] = {}

# Default prompts for computing mean activations (diverse, short prompts)
_DEFAULT_MEAN_ACTIVATION_PROMPTS = [
    "The weather today is",
    "In the year 2020,",
    "Scientists have discovered that",
    "The capital of France is",
    "According to recent studies,",
    "The main purpose of",
    "When considering the",
    "It is important to note that",
    "The history of humanity shows",
    "In modern times, people",
]


def clear_mean_activation_cache():
    """Clear the mean activation cache."""
    global _MEAN_ACTIVATION_CACHE
    _MEAN_ACTIVATION_CACHE = {}


def get_mean_activation(layer: int, neuron_idx: int, prompts: list[str] | None = None) -> float:
    """Get or compute the mean activation for a neuron across reference prompts.

    Uses a cache to avoid recomputing. If not cached, computes the mean
    activation across the provided prompts (or default prompts).

    Args:
        layer: Layer number
        neuron_idx: Neuron index
        prompts: Optional list of prompts to compute mean over. If None, uses defaults.

    Returns:
        Mean activation value
    """
    cache_key = (layer, neuron_idx)

    if cache_key in _MEAN_ACTIVATION_CACHE:
        return _MEAN_ACTIVATION_CACHE[cache_key]

    # Compute mean activation
    if prompts is None:
        prompts = _DEFAULT_MEAN_ACTIVATION_PROMPTS

    activations = []
    model, tokenizer = get_model_and_tokenizer()
    device = next(model.parameters()).device

    for prompt in prompts:
        text = format_prompt(prompt)
        try:
            acts = get_all_activations(layer, neuron_idx, text)
            if acts:
                # Get max activation for this prompt
                max_act = max(a[1] for a in acts)
                activations.append(max_act)
        except Exception:
            continue

    if activations:
        mean_act = sum(activations) / len(activations)
    else:
        mean_act = 0.0

    _MEAN_ACTIVATION_CACHE[cache_key] = mean_act
    return mean_act


def set_mean_activation(layer: int, neuron_idx: int, mean_value: float):
    """Manually set the mean activation for a neuron (e.g., from external computation)."""
    _MEAN_ACTIVATION_CACHE[(layer, neuron_idx)] = mean_value


def get_relp_attributor(tau: float = 0.01, k: int = 5):
    """Get or create the RelP attributor with specified parameters."""
    global _RELP_ATTRIBUTORS
    cache_key = (tau, k)
    if cache_key not in _RELP_ATTRIBUTORS:
        from circuits.relp import RelPAttributor, RelPConfig
        model, tokenizer = get_model_and_tokenizer()
        config = RelPConfig(
            k=k,
            tau=tau,
            compute_edges=True,
            use_jacobian_edges=True,
            linearize=True,
        )
        _RELP_ATTRIBUTORS[cache_key] = RelPAttributor(model, tokenizer, config=config)
    return _RELP_ATTRIBUTORS[cache_key]


async def tool_run_relp(
    layer: int,
    neuron_idx: int,
    prompt: str,
    target_tokens: list[str] | None = None,
    tau: float = 0.01,
    k: int = 5,
    timeout: float = 120.0,  # Configurable timeout in seconds
    max_nodes: int = 500,  # Maximum nodes before returning error (prevents slow edge computation)
) -> dict[str, Any]:
    """Run RelP attribution and check for the target neuron's presence and edges.

    RelP (Relative Propagation) computes attribution graphs showing which neurons
    contribute to the model's output and how they connect to each other.

    Args:
        layer: Layer of the neuron to look for
        neuron_idx: Index of the neuron to look for
        prompt: The prompt to run attribution on
        target_tokens: Optional list of specific tokens to trace (e.g., [" dopamine"])
        tau: Threshold for node inclusion (default 0.01). Higher = fewer nodes, faster.
             Use 0.05-0.1 for quick exploration, 0.005-0.01 for detailed analysis.
        k: Number of top logits to trace if target_tokens not specified (default 5)
        max_nodes: Maximum number of nodes before returning an error (default 500).
                   Prevents very slow edge computation on large graphs. If exceeded,
                   returns an error telling you to increase tau.

    Returns:
        Dict with:
        - neuron_found: Whether the target neuron appears in the graph
        - neuron_position: Token position where neuron was found (if any)
        - neuron_relp_score: The neuron's RelP score (activation * gradient)
        - downstream_edges: List of edges FROM this neuron to later neurons/logits
        - upstream_edges: List of edges TO this neuron from earlier neurons
        - all_neurons_in_graph: Summary of all neurons in the graph
        - nodes_exceeded: Whether max_nodes was exceeded (edges will be empty if so)
    """
    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        return await client.run_relp(
            layer=layer, neuron_idx=neuron_idx, prompt=prompt,
            target_tokens=target_tokens, tau=tau, k=k,
            timeout=timeout, max_nodes=max_nodes,
        )

    text = format_prompt(prompt)
    attributor = get_relp_attributor(tau=tau, k=k)

    # Run RelP attribution with timeout
    import asyncio
    import concurrent.futures

    def run_attribution():
        # Acquire lock to prevent concurrent CUDA operations from other tasks
        # This serializes model access to avoid "double free or corruption" errors
        with _MODEL_LOCK:
            return attributor.compute_attributions(
                text,
                target_tokens=target_tokens,
                max_nodes=max_nodes,
            )

    import time
    start_time = time.time()

    try:
        # Run in thread pool with configurable timeout
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = loop.run_in_executor(pool, run_attribution)
            graph = await asyncio.wait_for(future, timeout=timeout)
        elapsed = time.time() - start_time
    except TimeoutError:
        return {
            "error": f"RelP attribution timed out after {timeout:.0f} seconds. Try increasing tau (e.g., tau=0.05) to reduce graph size.",
            "suggestion": "Use higher tau for faster exploration, lower tau only for detailed analysis of specific prompts.",
            "tau_used": tau,
            "timeout_used": timeout,
        }
    except Exception as e:
        return {"error": f"RelP attribution failed: {str(e)}"}

    # Find the target neuron in the graph
    target_node_prefix = f"{layer}_{neuron_idx}_"
    neuron_id = f"L{layer}/N{neuron_idx}"

    found_nodes = []
    for node in graph.get("nodes", []):
        node_id = node.get("node_id", "")
        if node_id.startswith(target_node_prefix):
            found_nodes.append({
                "node_id": node_id,
                "position": node.get("ctx_idx"),
                "relp_score": node.get("influence"),
                "activation": node.get("activation"),
            })

    # Find edges involving this neuron
    downstream_edges = []
    upstream_edges = []

    for link in graph.get("links", []):
        source = link.get("source", "")
        target = link.get("target", "")
        weight = link.get("weight", 0)

        if source.startswith(target_node_prefix):
            # Edge FROM our neuron to something else
            # Parse target to get info
            target_info = _parse_node_id(target, graph.get("nodes", []))
            # Only include edges to LATER layers (or logits)
            # Same-layer or earlier-layer edges are not "downstream" architecturally
            target_layer = target_info.get("layer")
            if target_layer is None or target_layer > layer:
                downstream_edges.append({
                    "source": source,
                    "target": target,
                    "target_info": target_info,
                    "weight": weight,
                })
        elif target.startswith(target_node_prefix):
            # Edge TO our neuron from something else
            source_info = _parse_node_id(source, graph.get("nodes", []))
            # Only include edges from EARLIER layers (or embeddings)
            # Same-layer or later-layer edges are not "upstream" architecturally
            source_layer = source_info.get("layer")
            if source_layer is None or source_layer < layer:
                upstream_edges.append({
                    "source": source,
                    "source_info": source_info,
                    "target": target,
                    "weight": weight,
                })

    # Sort edges by absolute weight
    downstream_edges.sort(key=lambda x: abs(x["weight"]), reverse=True)
    upstream_edges.sort(key=lambda x: abs(x["weight"]), reverse=True)

    # Summarize all neurons in graph
    mlp_neurons = []
    for node in graph.get("nodes", []):
        if node.get("feature_type") == "mlp_neuron":
            mlp_neurons.append({
                "node_id": node.get("node_id"),
                "layer": node.get("layer"),
                "neuron": node.get("feature"),
                "position": node.get("ctx_idx"),
                "relp_score": node.get("influence"),
            })

    # Sort by absolute relp_score
    mlp_neurons.sort(key=lambda x: abs(x.get("relp_score", 0) or 0), reverse=True)

    # Check if node limit was exceeded - return error immediately
    total_nodes = len(graph.get("nodes", []))
    total_edges = len(graph.get("links", []))  # RelP uses "links" not "edges"
    nodes_exceeded = graph.get("nodes_exceeded", False)

    if nodes_exceeded:
        return {
            "error": f"Too many nodes ({total_nodes}) in RelP graph - edge computation would be too slow. INCREASE TAU to reduce graph size.",
            "suggestion": f"Current tau={tau} produced {total_nodes} nodes. Try tau=0.02, 0.05, or higher to reduce node count below {max_nodes}.",
            "tau_used": tau,
            "node_count": total_nodes,
            "max_nodes": max_nodes,
            "prompt": prompt[:100],
            "neuron_id": f"L{layer}/N{neuron_idx}",
        }

    # Recommendation based on performance
    recommendation = None
    if elapsed > 60:
        recommendation = f"Graph took {elapsed:.1f}s. Consider using higher tau (current: {tau}) for faster exploration."
    elif total_nodes > 500:
        recommendation = f"Large graph ({total_nodes} nodes). Consider higher tau to reduce complexity."

    result = {
        "prompt": prompt[:100],
        "neuron_id": neuron_id,
        "neuron_found": len(found_nodes) > 0,
        "neuron_relp_score": found_nodes[0].get("relp_score") if found_nodes else None,
        "found_at_positions": found_nodes,
        "downstream_edges": downstream_edges[:20],  # Top 20
        "downstream_edge_count": len(downstream_edges),
        "upstream_edges": upstream_edges[:10],  # Top 10
        "upstream_edge_count": len(upstream_edges),
        "total_neurons_in_graph": len(mlp_neurons),
        "top_neurons_in_graph": mlp_neurons[:15],  # Top 15 by score
        "graph_stats": {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "elapsed_seconds": round(elapsed, 2),
            "tau_used": tau,
            "nodes_exceeded": nodes_exceeded,
        },
        "recommendation": recommendation,
        "nodes_exceeded": nodes_exceeded,
    }

    # Auto-register result to RelP registry
    # Simplify edges to just neuron IDs and weights for easy extraction
    # Select top 10 positive + top 10 negative by absolute weight
    def simplify_upstream(edges):
        # Filter to MLP neurons and add weight
        mlp_edges = []
        for e in edges:
            source_info = e.get("source_info", {})
            if source_info.get("type") == "mlp_neuron":
                mlp_edges.append({
                    "source": f"L{source_info['layer']}/N{source_info['feature']}",
                    "weight": e.get("weight", 0)
                })
        # Separate positive and negative
        positive = [e for e in mlp_edges if e["weight"] > 0]
        negative = [e for e in mlp_edges if e["weight"] < 0]
        # Sort by absolute weight and take top 10 of each
        positive_top = sorted(positive, key=lambda x: abs(x["weight"]), reverse=True)[:10]
        negative_top = sorted(negative, key=lambda x: abs(x["weight"]), reverse=True)[:10]
        return positive_top + negative_top

    def simplify_downstream(edges):
        # Filter to MLP neurons and add weight
        mlp_edges = []
        for e in edges:
            target_info = e.get("target_info", {})
            if target_info.get("type") == "mlp_neuron":
                mlp_edges.append({
                    "target": f"L{target_info['layer']}/N{target_info['feature']}",
                    "weight": e.get("weight", 0)
                })
        # Separate positive and negative
        positive = [e for e in mlp_edges if e["weight"] > 0]
        negative = [e for e in mlp_edges if e["weight"] < 0]
        # Sort by absolute weight and take top 10 of each
        positive_top = sorted(positive, key=lambda x: abs(x["weight"]), reverse=True)[:10]
        negative_top = sorted(negative, key=lambda x: abs(x["weight"]), reverse=True)[:10]
        return positive_top + negative_top

    global _RELP_REGISTRY
    _RELP_REGISTRY.append({
        "prompt": prompt,
        "target_tokens": target_tokens,
        "tau": tau,
        "neuron_found": result["neuron_found"],
        "neuron_relp_score": result.get("neuron_relp_score"),
        "downstream_edges": simplify_downstream(downstream_edges),
        "upstream_edges": simplify_upstream(upstream_edges),
        "graph_stats": result["graph_stats"],
        "in_causal_pathway": result["neuron_found"],
    })

    # Update protocol state for RelP tracking
    state = get_protocol_state()
    new_relp_runs = state.relp_runs + 1
    updates = {"relp_runs": new_relp_runs}

    # Track positive control: neuron found in at least one graph
    if result["neuron_found"]:
        updates["relp_positive_control"] = True

    # Note: negative control tracking (neuron NOT found when it shouldn't be)
    # is set by save_structured_report based on agent's claims about control prompts
    update_protocol_state(**updates)
    found_str = "FOUND" if result["neuron_found"] else "not found"
    exceeded_str = " (nodes_exceeded, edges skipped)" if nodes_exceeded else ""
    print(f"[PROTOCOL] RelP run #{new_relp_runs}: neuron {found_str} (tau={tau}){exceeded_str}")

    return result


def _parse_node_id(node_id: str, nodes: list[dict]) -> dict[str, Any]:
    """Parse a node ID to extract useful information."""
    _, tokenizer = get_model_and_tokenizer()

    # Find the node in the nodes list
    for node in nodes:
        if node.get("node_id") == node_id:
            result = {
                "type": node.get("feature_type"),
                "layer": node.get("layer"),
                "feature": node.get("feature"),
                "position": node.get("ctx_idx"),
                "clerp": node.get("clerp"),
            }
            # Add decoded label for logit nodes
            if node.get("feature_type") == "logit":
                token_id = node.get("feature")
                if token_id is not None:
                    result["token"] = tokenizer.decode([token_id])
                    result["label"] = f"LOGIT({result['token']})"
            return result

    # Parse from the ID if not found
    if node_id.startswith("E_"):
        parts = node_id.split("_")
        token_id = int(parts[1])
        token = tokenizer.decode([token_id])
        return {
            "type": "embedding",
            "token_id": token_id,
            "token": token,
            "position": int(parts[2]),
            "label": f"EMB({token})",
        }
    elif node_id.startswith("L_"):
        parts = node_id.split("_")
        token_id = int(parts[1])
        token = tokenizer.decode([token_id])
        return {
            "type": "logit",
            "token_id": token_id,
            "token": token,
            "position": int(parts[2]),
            "label": f"LOGIT({token})",
        }
    else:
        parts = node_id.split("_")
        if len(parts) >= 3:
            return {
                "type": "mlp_neuron",
                "layer": int(parts[0]),
                "neuron": int(parts[1]),
                "position": int(parts[2]),
                "label": f"L{parts[0]}/N{parts[1]}",
            }

    return {"raw": node_id}


def _sync_steer_neuron(
    layer: int,
    neuron_idx: int,
    prompt: str,
    steering_value: float,
    position: int = -1,
    top_k_logits: int = 10,
) -> dict[str, Any]:
    """Synchronous implementation of steer_neuron."""
    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        text = format_prompt(prompt)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        # Find assistant start position (after last end_header_id)
        assistant_start = 0
        tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]
        for i, tok in enumerate(tokens):
            if "end_header_id" in tok:
                assistant_start = i + 1

        # Determine positions to steer
        if position == -2:
            # All positions from assistant start
            positions_to_steer = list(range(assistant_start, seq_len))
        elif position < 0:
            positions_to_steer = [seq_len + position]
        else:
            positions_to_steer = [position]

        # Get original logits
        with torch.no_grad():
            outputs = model(**inputs)
            original_logits = outputs.logits[0, -1].float()

        top_values, top_indices = torch.topk(original_logits, top_k_logits)
        original_top = {
            tokenizer.decode([idx.item()]): val.item()
            for idx, val in zip(top_indices, top_values)
        }

        # Create steering hook
        def steering_hook(module, args, kwargs):
            x = args[0]  # Shape: (batch, seq_len, intermediate_dim)
            modified = x.clone()
            for pos in positions_to_steer:
                if pos < modified.shape[1] and neuron_idx < modified.shape[2]:
                    modified[0, pos, neuron_idx] += steering_value
            return (modified,) + args[1:], kwargs

        # Apply steering
        mlp = model.model.layers[layer].mlp
        handle = mlp.down_proj.register_forward_pre_hook(steering_hook, with_kwargs=True)

        try:
            with torch.no_grad():
                outputs = model(**inputs)
                steered_logits = outputs.logits[0, -1].float()
        finally:
            handle.remove()

        # Compute effects
        steered_top = {
            tokenizer.decode([idx.item()]): steered_logits[idx.item()].item()
            for idx in top_indices
        }

        logit_shifts = {
            token: steered_top.get(token, 0) - original_top.get(token, 0)
            for token in original_top
        }

        # Find tokens with largest shifts
        sorted_shifts = sorted(logit_shifts.items(), key=lambda x: x[1], reverse=True)
        promoted = [(t, s) for t, s in sorted_shifts if s > 0][:5]
        suppressed = [(t, s) for t, s in sorted_shifts if s < 0][-5:]

        return {
            "prompt": prompt[:100],
            "steering_value": steering_value,
            "positions_steered": positions_to_steer,
            "original_logits": original_top,
            "steered_logits": steered_top,
            "logit_shifts": logit_shifts,
            "promoted_tokens": promoted,
            "suppressed_tokens": suppressed,
            "max_shift": max(logit_shifts.values(), key=abs) if logit_shifts else 0,
        }


async def tool_steer_neuron(
    layer: int,
    neuron_idx: int,
    prompt: str,
    steering_value: float,
    position: int = -1,
    top_k_logits: int = 10,
) -> dict[str, Any]:
    """Steer a neuron by adding a fixed value to its activation and measure logit effects.

    This is a causal intervention: it tests what happens when you increase or decrease
    the neuron's activation. Positive steering_value amplifies the neuron's effect,
    negative dampens it.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index
        prompt: Input prompt
        steering_value: Value to add to the neuron's activation (positive = amplify, negative = suppress)
        position: Position to steer (-1 for last token, -2 for all positions after assistant start)
        top_k_logits: Number of top logits to return

    Returns:
        Dict with original_logits, steered_logits, and shifts
    """
    # Record experiment for temporal enforcement
    record_experiment("steering", {
        "layer": layer,
        "neuron_idx": neuron_idx,
        "steering_value": steering_value,
        "prompt": prompt[:100],
    })

    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        return await client.steer_neuron(
            layer=layer, neuron_idx=neuron_idx, prompt=prompt,
            steering_value=steering_value, position=position,
            top_k_logits=top_k_logits,
        )

    # Run on dedicated CUDA thread to avoid context switching issues
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _CUDA_EXECUTOR,
        _sync_steer_neuron,
        layer,
        neuron_idx,
        prompt,
        steering_value,
        position,
        top_k_logits,
    )


async def tool_patch_activation(
    layer: int,
    neuron_idx: int,
    source_prompt: str,
    target_prompt: str,
    position: int = -1,
    top_k_logits: int = 10,
) -> dict[str, Any]:
    """Patch a neuron's activation from source prompt into target prompt.

    This is a counterfactual intervention: it tests what happens when you replace
    the neuron's activation in one context with its activation from another context.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index
        source_prompt: Prompt to get activation FROM
        target_prompt: Prompt to patch activation INTO
        position: Position to patch (-1 for last, -2 for all positions)
        top_k_logits: Number of top logits to return

    Returns:
        Dict with source activation, target activation, baseline logits, patched logits, and shifts
    """
    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        return await client.patch_activation(
            layer=layer, neuron_idx=neuron_idx,
            source_prompt=source_prompt, target_prompt=target_prompt,
            position=position, top_k_logits=top_k_logits,
        )

    model, tokenizer = get_model_and_tokenizer()
    device = next(model.parameters()).device

    source_text = format_prompt(source_prompt)
    target_text = format_prompt(target_prompt)

    source_inputs = tokenizer(source_text, return_tensors="pt").to(device)
    target_inputs = tokenizer(target_text, return_tensors="pt").to(device)

    source_len = source_inputs["input_ids"].shape[1]
    target_len = target_inputs["input_ids"].shape[1]

    # Find positions to patch
    if position == -2:
        # Find assistant start in target
        tokens = [tokenizer.decode([tid]) for tid in target_inputs["input_ids"][0]]
        assistant_start = 0
        for i, tok in enumerate(tokens):
            if "end_header_id" in tok:
                assistant_start = i + 1
        positions = list(range(assistant_start, target_len))
    elif position < 0:
        positions = [target_len + position]
    else:
        positions = [position]

    # Step 1: Get source activations at ALL positions (we'll map to target later)
    source_activations = {}

    def capture_hook(module, input, output):
        hidden = input[0]
        gate = module.gate_proj(hidden)
        up = module.up_proj(hidden)
        intermediate = torch.nn.functional.silu(gate) * up
        # Capture at ALL source positions so we can map to any target position
        for pos in range(intermediate.shape[1]):
            source_activations[pos] = intermediate[0, pos, neuron_idx].item()

    mlp = model.model.layers[layer].mlp
    handle = mlp.register_forward_hook(capture_hook)

    with torch.no_grad():
        model(**source_inputs)
    handle.remove()

    # Step 2: Get baseline target logits
    with torch.no_grad():
        outputs = model(**target_inputs)
        baseline_logits = outputs.logits[0, -1].float()

    top_values, top_indices = torch.topk(baseline_logits, top_k_logits)
    baseline_top = {
        tokenizer.decode([idx.item()]): val.item()
        for idx, val in zip(top_indices, top_values)
    }

    # Step 3: Get target activations (for comparison)
    target_activations = {}

    def capture_target_hook(module, input, output):
        hidden = input[0]
        gate = module.gate_proj(hidden)
        up = module.up_proj(hidden)
        intermediate = torch.nn.functional.silu(gate) * up
        for pos in positions:
            if pos < intermediate.shape[1]:
                target_activations[pos] = intermediate[0, pos, neuron_idx].item()

    handle = mlp.register_forward_hook(capture_target_hook)
    with torch.no_grad():
        model(**target_inputs)
    handle.remove()

    # Step 4: Patch source activations into target
    # Warn if lengths differ significantly
    length_mismatch_warning = None
    if abs(source_len - target_len) > 3:
        length_mismatch_warning = (
            f"Warning: Source ({source_len} tokens) and target ({target_len} tokens) have different lengths. "
            f"Position mapping may not be semantically meaningful. Consider using same-length prompts."
        )

    def patch_hook(module, args, kwargs):
        x = args[0]  # Shape: (batch, seq_len, intermediate_dim)
        modified = x.clone()
        for pos in positions:
            if pos < modified.shape[1] and neuron_idx < modified.shape[2]:
                # Map target position to source position (scale proportionally)
                source_pos = int(pos * source_len / target_len) if target_len > 0 else pos
                # Clamp to valid source range
                source_pos = min(source_pos, source_len - 1)
                if source_pos in source_activations:
                    modified[0, pos, neuron_idx] = source_activations[source_pos]
        return (modified,) + args[1:], kwargs

    handle = mlp.down_proj.register_forward_pre_hook(patch_hook, with_kwargs=True)

    try:
        with torch.no_grad():
            outputs = model(**target_inputs)
            patched_logits = outputs.logits[0, -1].float()
    finally:
        handle.remove()

    # Compute effects
    patched_top = {
        tokenizer.decode([idx.item()]): patched_logits[idx.item()].item()
        for idx in top_indices
    }

    logit_shifts = {
        token: patched_top.get(token, 0) - baseline_top.get(token, 0)
        for token in baseline_top
    }

    sorted_shifts = sorted(logit_shifts.items(), key=lambda x: x[1], reverse=True)
    promoted = [(t, s) for t, s in sorted_shifts if s > 0][:5]
    suppressed = [(t, s) for t, s in sorted_shifts if s < 0][-5:]

    # For activation_delta, compute delta at patched positions
    # Map target positions to source positions for comparison
    activation_delta = {}
    for pos in positions:
        if pos in target_activations:
            source_pos = int(pos * source_len / target_len) if target_len > 0 else pos
            source_pos = min(source_pos, source_len - 1)
            activation_delta[pos] = source_activations.get(source_pos, 0) - target_activations.get(pos, 0)

    # For source_activations, only include the positions that were actually used for patching
    # (the mapped source positions corresponding to each target position)
    used_source_positions = set()
    for pos in positions:
        source_pos = int(pos * source_len / target_len) if target_len > 0 else pos
        source_pos = min(source_pos, source_len - 1)
        used_source_positions.add(source_pos)

    result = {
        "source_prompt": source_prompt[:100],
        "target_prompt": target_prompt[:100],
        "source_length": source_len,
        "target_length": target_len,
        "positions_patched": positions,
        "source_activations": {k: v for k, v in source_activations.items() if k in used_source_positions},
        "target_activations": target_activations,
        "activation_delta": activation_delta,
        "baseline_logits": baseline_top,
        "patched_logits": patched_top,
        "logit_shifts": logit_shifts,
        "promoted_tokens": promoted,
        "suppressed_tokens": suppressed,
        "max_shift": max(logit_shifts.values(), key=abs) if logit_shifts else 0,
    }

    if length_mismatch_warning:
        result["warning"] = length_mismatch_warning

    return result


async def tool_get_neuron_label(
    neuron_id: str,
    labels_path: str = DEFAULT_LABELS_PATH,
) -> dict[str, Any]:
    """Look up the label and description for any neuron.

    Uses JSON file first, then falls back to NeuronDB (458K neurons) if not found.

    Args:
        neuron_id: Neuron ID in format "L{layer}/N{neuron}" (e.g., "L15/N8545")
        labels_path: Path to the labels JSON file

    Returns:
        Dict with label info if found, or indication that neuron is unlabeled.
        If found in NeuronDB but not JSON, includes 'neurondb_description'.
    """
    # Load JSON file
    try:
        with open(labels_path) as f:
            labels_data = json.load(f)
    except FileNotFoundError:
        labels_data = {"neurons": {}}

    neurons = labels_data.get("neurons", {})

    # Check JSON first
    if neuron_id in neurons:
        n = neurons[neuron_id]
        # Also look up labels for upstream/downstream neurons
        upstream_ids = [u.get("neuron_id") for u in n.get("upstream_neurons", [])[:10] if u.get("neuron_id")]
        downstream_ids = [d.get("neuron_id") for d in n.get("downstream_neurons", [])[:10] if d.get("neuron_id")]
        all_related_ids = upstream_ids + downstream_ids

        # Get labels for related neurons (with NeuronDB fallback)
        related_labels = batch_get_neuron_labels_with_fallback(all_related_ids, labels_path) if all_related_ids else {}

        return {
            "neuron_id": neuron_id,
            "found": True,
            "source": "json",
            "function_label": n.get("function_label", ""),
            "function_description": n.get("function_description", ""),
            "function_type": n.get("function_type", ""),
            "interpretability": n.get("interpretability", ""),
            "input_label": n.get("input_label", ""),
            "input_description": n.get("input_description", ""),
            "input_type": n.get("input_type", ""),
            "input_interpretability": n.get("input_interpretability", ""),
            "downstream_neurons": [
                {
                    "neuron_id": d.get("neuron_id"),
                    "weight": d.get("weight"),
                    "frequency": d.get("frequency"),
                    "function_label": d.get("function_label") or related_labels.get(d.get("neuron_id"), {}).get("label", ""),
                }
                for d in n.get("downstream_neurons", [])[:10]  # Increased from 5 to 10
            ],
            "upstream_neurons": [
                {
                    "neuron_id": u.get("neuron_id"),
                    "weight": u.get("weight"),
                    "frequency": u.get("frequency"),
                    "function_label": u.get("function_label") or related_labels.get(u.get("neuron_id"), {}).get("label", ""),
                }
                for u in n.get("upstream_neurons", [])[:10]  # Increased from 5 to 10
            ],
        }

    # Fall back to NeuronDB
    result = get_neuron_label_with_fallback(neuron_id, labels_data, labels_path)

    if result["found"]:
        return {
            "neuron_id": neuron_id,
            "found": True,
            "source": "neurondb",
            "neurondb_description": result.get("neurondb_description", result.get("label", "")),
            "message": "Label found in NeuronDB (based on max-activating examples). Consider expanding to input/output function claims.",
            # No function_label/input_label since NeuronDB doesn't have that split
            "function_label": "",
            "input_label": "",
            "downstream_neurons": [],
            "upstream_neurons": [],
        }

    return {
        "neuron_id": neuron_id,
        "found": False,
        "source": None,
        "message": "No label available in JSON file or NeuronDB",
    }


async def tool_batch_get_neuron_labels(
    neuron_ids: list[str],
    labels_path: str = DEFAULT_LABELS_PATH,
) -> dict[str, Any]:
    """Look up labels for multiple neurons at once.

    Uses JSON file first, then falls back to NeuronDB (458K neurons) for any not found.

    Args:
        neuron_ids: List of neuron IDs (e.g., ["L15/N8545", "L21/N6856"])
        labels_path: Path to the labels JSON file

    Returns:
        Dict with:
            - total_queried: Number of neurons queried
            - found_count: Number found (in JSON or NeuronDB)
            - found_in_json: Number found in JSON file
            - found_in_neurondb: Number found via NeuronDB fallback
            - labels: Dict mapping neuron_id to label info
    """
    # Use the helper that handles fallback
    results = batch_get_neuron_labels_with_fallback(neuron_ids, labels_path)

    found_count = sum(1 for r in results.values() if r.get("found"))
    found_in_json = sum(1 for r in results.values() if r.get("source") == "json")
    found_in_neurondb = sum(1 for r in results.values() if r.get("source") == "neurondb")

    return {
        "total_queried": len(neuron_ids),
        "found_count": found_count,
        "found_in_json": found_in_json,
        "found_in_neurondb": found_in_neurondb,
        "not_found_count": len(neuron_ids) - found_count,
        "labels": results,
    }


async def tool_verify_downstream_connections(
    layer: int,
    neuron_idx: int,
    prompts: list[str],
    expected_downstream: list[str],
    tau: float = 0.005,  # Lower default for edge verification
) -> dict[str, Any]:
    """Verify that the neuron connects to expected downstream neurons across multiple prompts.

    This runs RelP on multiple prompts and checks how often edges appear from the
    target neuron to the expected downstream neurons.

    Args:
        layer: Layer of the neuron to check
        neuron_idx: Index of the neuron to check
        prompts: List of prompts to test
        expected_downstream: List of expected downstream neuron IDs (e.g., ["L5/N247", "L6/N12481"])

    Returns:
        Dict with verification results for each expected downstream neuron
    """
    neuron_id = f"L{layer}/N{neuron_idx}"

    # Parse expected downstream neurons
    expected_parsed = []
    for nid in expected_downstream:
        if nid.startswith("L") and "/N" in nid:
            parts = nid.split("/N")
            exp_layer = int(parts[0][1:])
            exp_neuron = int(parts[1])
            expected_parsed.append({
                "neuron_id": nid,
                "layer": exp_layer,
                "neuron": exp_neuron,
                "node_prefix": f"{exp_layer}_{exp_neuron}_",
            })

    results = {
        "source_neuron": neuron_id,
        "prompts_tested": len(prompts),
        "source_found_count": 0,
        "downstream_verification": {exp["neuron_id"]: {"found_count": 0, "edge_count": 0, "weights": []} for exp in expected_parsed},
        "unexpected_downstream": {},
        "failed_prompts": [],  # Track failures explicitly
        "relp_errors": [],  # Log error messages
    }

    for prompt in prompts:
        try:
            relp_result = await tool_run_relp(layer, neuron_idx, prompt, tau=tau)

            # Check for RelP-level errors (timeout, etc.)
            if relp_result.get("error"):
                results["failed_prompts"].append(prompt[:80])
                results["relp_errors"].append({
                    "prompt": prompt[:80],
                    "error": relp_result.get("error"),
                })
                continue

            if relp_result.get("neuron_found"):
                results["source_found_count"] += 1

                # Check each downstream edge
                for edge in relp_result.get("downstream_edges", []):
                    target = edge.get("target", "")
                    weight = edge.get("weight", 0)

                    # Check if it matches any expected downstream
                    matched = False
                    for exp in expected_parsed:
                        if target.startswith(exp["node_prefix"]):
                            results["downstream_verification"][exp["neuron_id"]]["found_count"] += 1
                            results["downstream_verification"][exp["neuron_id"]]["edge_count"] += 1
                            results["downstream_verification"][exp["neuron_id"]]["weights"].append(weight)
                            matched = True
                            break

                    # Track unexpected downstream connections
                    if not matched:
                        target_info = edge.get("target_info", {})
                        if target_info.get("type") == "mlp_neuron":
                            target_layer = target_info.get("layer")
                            target_neuron = target_info.get("feature")
                            if target_layer is not None and target_neuron is not None:
                                unexpected_id = f"L{target_layer}/N{target_neuron}"
                                if unexpected_id not in results["unexpected_downstream"]:
                                    results["unexpected_downstream"][unexpected_id] = {"count": 0, "weights": []}
                                results["unexpected_downstream"][unexpected_id]["count"] += 1
                                results["unexpected_downstream"][unexpected_id]["weights"].append(weight)

        except Exception as e:
            # Log the error instead of silently swallowing
            results["failed_prompts"].append(prompt[:80])
            results["relp_errors"].append({
                "prompt": prompt[:80],
                "error": str(e),
                "error_type": type(e).__name__,
            })

    # Compute summary statistics
    for nid, data in results["downstream_verification"].items():
        weights = data["weights"]
        if weights:
            data["avg_weight"] = sum(weights) / len(weights)
            data["max_weight"] = max(weights, key=abs)
        else:
            data["avg_weight"] = 0
            data["max_weight"] = 0
        data["verification_rate"] = data["found_count"] / results["source_found_count"] if results["source_found_count"] > 0 else 0

    # Sort unexpected by count
    results["unexpected_downstream"] = dict(
        sorted(results["unexpected_downstream"].items(), key=lambda x: -x[1]["count"])[:10]
    )

    # Add error rate summary
    n_failed = len(results["failed_prompts"])
    n_total = len(prompts)
    results["error_rate"] = n_failed / n_total if n_total > 0 else 0

    # Add warning if high failure rate
    if results["error_rate"] > 0.3:
        results["warning"] = (
            f"High RelP failure rate ({n_failed}/{n_total} = {results['error_rate']:.0%}). "
            "Results may be unreliable. Consider increasing tau or checking prompts."
        )

    return results


# =============================================================================
# Output Projection Analysis (from model weights)
# =============================================================================


def _sync_get_output_projections(layer: int, neuron_idx: int, top_k: int) -> dict[str, Any]:
    """Synchronous implementation of get_output_projections. Runs on dedicated CUDA thread."""
    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()

        # Get the neuron's column from down_proj: shape (hidden_dim,)
        down_proj = model.model.layers[layer].mlp.down_proj.weight  # (hidden_dim, intermediate_dim)
        neuron_contribution = down_proj[:, neuron_idx]  # (hidden_dim,)

        # Get lm_head weights: shape (vocab_size, hidden_dim)
        lm_head = model.lm_head.weight  # (vocab_size, hidden_dim)

        # Compute projection to vocabulary: how much this neuron contributes to each logit
        # projection[i] = dot(lm_head[i], neuron_contribution)
        with torch.no_grad():
            projection = lm_head @ neuron_contribution  # (vocab_size,)

        # Get top promoted (most positive) and suppressed (most negative)
        top_promoted_vals, top_promoted_ids = torch.topk(projection, top_k)
        top_suppressed_vals, top_suppressed_ids = torch.topk(-projection, top_k)

        promoted = []
        for val, idx in zip(top_promoted_vals.tolist(), top_promoted_ids.tolist()):
            token = tokenizer.decode([idx])
            promoted.append({
                "token": token,
                "token_id": idx,
                "projection_strength": round(val, 4),
            })

        suppressed = []
        for val, idx in zip(top_suppressed_vals.tolist(), top_suppressed_ids.tolist()):
            token = tokenizer.decode([idx])
            suppressed.append({
                "token": token,
                "token_id": idx,
                "projection_strength": round(-val, 4),  # Negative value
            })

        # Compute some statistics
        proj_mean = projection.mean().item()
        proj_std = projection.std().item()

        return {
            "layer": layer,
            "neuron_idx": neuron_idx,
            "promoted": promoted,
            "suppressed": suppressed,
            "stats": {
                "mean_projection": round(proj_mean, 6),
                "std_projection": round(proj_std, 4),
                "max_projection": round(projection.max().item(), 4),
                "min_projection": round(projection.min().item(), 4),
            },
        }


async def tool_get_output_projections(
    layer: int,
    neuron_idx: int,
    top_k: int = 10,
    polarity_mode: str = "positive",
) -> dict[str, Any]:
    """Get the actual output projections from the neuron's down_proj weights.

    This computes what tokens the neuron promotes/suppresses based on its
    contribution to the unembedding (down_proj @ lm_head).

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index
        top_k: Number of top promoted/suppressed tokens to return
        polarity_mode: "positive" (default) or "negative". When "negative",
            promoted and suppressed lists are swapped (since negative activation
            reverses the effect of output weights).

    Returns:
        Dict with top promoted tokens (positive weights) and suppressed tokens (negative weights)
    """
    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        result = await client.get_output_projections(
            layer=layer, neuron_idx=neuron_idx, top_k=top_k,
        )
    else:
        # Run on dedicated CUDA thread to ensure consistent CUDA context
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _CUDA_EXECUTOR,
            _sync_get_output_projections,
            layer,
            neuron_idx,
            top_k,
        )

    # Cache result for auto-injection into save_structured_report
    _store_output_projections(result)

    # When in negative polarity mode, swap promoted/suppressed lists
    # because negative activation * positive weight = negative contribution (suppression)
    # and negative activation * negative weight = positive contribution (promotion)
    if polarity_mode == "negative":
        promoted = result.get("promoted", [])
        suppressed = result.get("suppressed", [])
        result["promoted"] = suppressed
        result["suppressed"] = promoted
        result["regime_note"] = (
            "Projections shown for NEGATIVE firing (reversed from weight signs). "
            "When this neuron fires negatively, tokens listed as 'promoted' are "
            "actually promoted (negative activation × negative weight = positive logit shift)."
        )
        result["polarity_mode"] = "negative"
    else:
        # Add regime context from protocol state
        state = get_protocol_state()
        result["regime_note"] = (
            "Output projections assume POSITIVE firing (activation > 0). "
            "When this neuron fires NEGATIVELY, promotes/suppresses are REVERSED."
        )
        result["operating_regime"] = state.operating_regime if state else None
        result["firing_sign_stats"] = state.firing_sign_stats if state else None
        result["polarity_mode"] = "positive"

        # Add stronger warning if significant negative firing detected
        if state and state.firing_sign_stats:
            neg_pct = state.firing_sign_stats.get("negative_pct", 0)
            if neg_pct > 20:
                result["regime_warning"] = (
                    f"WARNING: This neuron fires negatively {neg_pct:.0f}% of the time. "
                    "When firing negatively, all promoted tokens become suppressed and vice versa. "
                    "Interpret output projections separately for positive vs negative activation contexts."
                )

    return result


# =============================================================================
# Direct Effect Ratio Tool
# =============================================================================


def _sync_compute_direct_effect_ratio(
    layer: int, neuron_idx: int, prompt: str, target_token: str, activation: float
) -> dict[str, Any]:
    """Synchronous implementation of direct effect ratio computation."""
    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Get target token ID
        target_ids = tokenizer.encode(target_token, add_special_tokens=False)
        if not target_ids:
            return {"error": f"Could not tokenize target token: {target_token}"}
        target_id = target_ids[0]

        # Get model components
        final_norm = model.model.norm
        lm_head = model.lm_head

        # Capture final hidden state before norm
        final_hidden = None

        def capture_pre_norm(module, args):
            nonlocal final_hidden
            hidden = args[0] if isinstance(args, tuple) else args
            final_hidden = hidden[:, -1, :].detach().clone()

        hook = final_norm.register_forward_pre_hook(capture_pre_norm)

        # Clean forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits_clean = outputs.logits[:, -1, :].clone()

        hook.remove()

        # Compute direct effect algebraically (no forward pass needed)
        # Direct effect = what happens if we add V to final hidden and freeze downstream
        down_proj = model.model.layers[layer].mlp.down_proj.weight
        V = down_proj[:, neuron_idx] * activation

        h_plus_V = final_hidden + V
        h_normed = final_norm(h_plus_V)
        logits_direct = lm_head(h_normed)
        direct_effect = (logits_direct - logits_clean)[0, target_id].item()

        # Compute total effect with perturbed forward pass
        def inject_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[:, -1, :] += V
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        hook = model.model.layers[layer].register_forward_hook(inject_hook)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_perturbed = outputs.logits[:, -1, :].clone()

        hook.remove()

        total_effect = (logits_perturbed - logits_clean)[0, target_id].item()
        indirect_effect = total_effect - direct_effect

        # Compute ratio
        if abs(direct_effect) + abs(indirect_effect) > 1e-8:
            direct_ratio = abs(direct_effect) / (abs(direct_effect) + abs(indirect_effect))
        else:
            direct_ratio = 0.5  # Neutral when both are near-zero

        return {
            "layer": layer,
            "neuron_idx": neuron_idx,
            "prompt": prompt,
            "target_token": target_token,
            "target_id": target_id,
            "activation": activation,
            "direct_effect": round(direct_effect, 4),
            "indirect_effect": round(indirect_effect, 4),
            "total_effect": round(total_effect, 4),
            "direct_effect_ratio": round(direct_ratio, 4),
            "interpretation": (
                "Mostly direct (output weights)" if direct_ratio > 0.7
                else "Mostly indirect (downstream neurons)" if direct_ratio < 0.3
                else "Mixed direct/indirect"
            ),
        }


async def tool_compute_direct_effect_ratio(
    layer: int,
    neuron_idx: int,
    prompt: str,
    target_token: str,
    activation: float = 1.0,
) -> dict[str, Any]:
    """Compute the direct vs indirect effect ratio for a neuron (INFORMATIONAL ONLY).

    IMPORTANT: DER is highly context-dependent and varies significantly by prompt
    and target token. It should NOT be used to categorize neurons as "projection-
    dominant" or "routing hubs". Use it only as one data point among many.

    This measures how much of the neuron's effect on output comes from its
    direct projection to logits (via down_proj @ lm_head) vs indirect effects
    through downstream neurons for a SPECIFIC (prompt, target_token) pair.

    Args:
        layer: Layer number
        neuron_idx: Neuron index
        prompt: The prompt to test on
        target_token: The token to measure effects on (e.g., ' wine')
        activation: Activation value to simulate (default 1.0)

    Returns:
        Dict with direct_effect, indirect_effect, total_effect, and direct_effect_ratio.
        Note: These values are specific to this (prompt, target_token) combination
        and may differ substantially for other contexts.
    """
    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        return await client.compute_direct_effect_ratio(
            layer=layer, neuron_idx=neuron_idx, prompt=prompt,
            target_token=target_token, activation=activation,
        )

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _CUDA_EXECUTOR,
        _sync_compute_direct_effect_ratio,
        layer,
        neuron_idx,
        prompt,
        target_token,
        activation,
    )


# =============================================================================
# Pre-Registration Tool
# =============================================================================

# Global hypothesis registry for tracking pre-registered hypotheses
_HYPOTHESIS_REGISTRY = []

# Global experiment registry for temporal enforcement
# Tracks when experiments are run to validate pre-registration
_EXPERIMENT_REGISTRY = []


def clear_hypothesis_registry():
    """Clear the hypothesis registry (call at start of new investigation)."""
    global _HYPOTHESIS_REGISTRY
    _HYPOTHESIS_REGISTRY = []


def clear_experiment_registry():
    """Clear the experiment registry (call at start of new investigation)."""
    global _EXPERIMENT_REGISTRY
    _EXPERIMENT_REGISTRY = []


def record_experiment(experiment_type: str, details: dict[str, Any]) -> str:
    """Record an experiment for temporal enforcement.

    Args:
        experiment_type: Type of experiment (e.g., 'activation_test', 'ablation', 'steering')
        details: Experiment details (prompt, layer, neuron, etc.)

    Returns:
        Experiment ID for reference
    """
    import datetime

    experiment_id = f"E{len(_EXPERIMENT_REGISTRY) + 1}"
    _EXPERIMENT_REGISTRY.append({
        "experiment_id": experiment_id,
        "type": experiment_type,
        "timestamp": datetime.datetime.now().isoformat(),
        "details": details,
    })
    return experiment_id


def get_experiment_registry() -> list[dict[str, Any]]:
    """Get all recorded experiments."""
    return _EXPERIMENT_REGISTRY.copy()


def get_experiments_before_hypothesis(hypothesis_timestamp: str) -> list[dict[str, Any]]:
    """Get all experiments that were run before a hypothesis was registered.

    Args:
        hypothesis_timestamp: ISO format timestamp of hypothesis registration

    Returns:
        List of experiments run before the hypothesis was registered
    """
    return [
        exp for exp in _EXPERIMENT_REGISTRY
        if exp["timestamp"] < hypothesis_timestamp
    ]


def get_hypothesis_registry() -> list[dict[str, Any]]:
    """Get all registered hypotheses."""
    return _HYPOTHESIS_REGISTRY.copy()


def init_hypothesis_registry(prior_hypotheses: list[dict[str, Any]]):
    """Initialize hypothesis registry from prior iteration.

    Loads hypotheses from a prior investigation, preserving IDs and status.
    New hypotheses will continue numbering from where the prior left off.

    Args:
        prior_hypotheses: List of hypothesis dicts from prior investigation
                          (from NeuronInvestigation.hypotheses_tested)
    """
    global _HYPOTHESIS_REGISTRY
    _HYPOTHESIS_REGISTRY = [
        {**h, "from_prior_iteration": True}
        for h in prior_hypotheses
    ]


def get_next_hypothesis_id() -> str:
    """Get the next hypothesis ID (e.g., 'H3' if H1 and H2 exist)."""
    return f"H{len(_HYPOTHESIS_REGISTRY) + 1}"


async def tool_register_hypothesis(
    hypothesis: str,
    confirmation_criteria: str,
    refutation_criteria: str,
    prior_probability: int,
    hypothesis_type: str = "activation",
) -> dict[str, Any]:
    """Register a hypothesis BEFORE running experiments to prevent p-hacking.

    This tool enforces the pre-registration protocol. Call this BEFORE running
    experiments to test a hypothesis. This creates an auditable record of what
    you predicted before seeing the results.

    Args:
        hypothesis: Specific, falsifiable hypothesis (e.g., "Neuron activates >1.0
            when 'dopamine' appears in scientific contexts")
        confirmation_criteria: What result would CONFIRM the hypothesis (e.g.,
            "Activation >1.0 on 80%+ of dopamine prompts")
        refutation_criteria: What result would REFUTE the hypothesis (e.g.,
            "Activation <0.5 on most dopamine prompts OR equal activation on
            serotonin/norepinephrine prompts")
        prior_probability: Your confidence (0-100%) BEFORE testing
        hypothesis_type: Type of hypothesis: "activation", "output", "causal", "connectivity"

    Returns:
        Dict with hypothesis ID and registration confirmation
    """
    import datetime

    hypothesis_id = f"H{len(_HYPOTHESIS_REGISTRY) + 1}"

    timestamp = datetime.datetime.now().isoformat()

    registered = {
        "hypothesis_id": hypothesis_id,
        "hypothesis": hypothesis,
        "confirmation_criteria": confirmation_criteria,
        "refutation_criteria": refutation_criteria,
        "prior_probability": prior_probability,
        "hypothesis_type": hypothesis_type,
        "registered_at": timestamp,
        "status": "registered",  # registered -> tested -> confirmed/refuted/inconclusive
        "posterior_probability": None,
        "evidence": [],
        # History tracks the evolution of this hypothesis
        "history": [
            {
                "timestamp": timestamp,
                "action": "registered",
                "prior": prior_probability,
                "status": "registered",
            }
        ],
    }

    _HYPOTHESIS_REGISTRY.append(registered)

    # Update protocol state
    state = get_protocol_state()
    update_protocol_state(hypotheses_registered=state.hypotheses_registered + 1)

    # Warn if registering output hypothesis before input phase is complete
    warning = ""
    if hypothesis_type == "output" and not state.input_phase_complete:
        warning = (
            " ⚠️ WARNING: You are registering an OUTPUT hypothesis before completing the Input Phase. "
            "Output hypotheses should be grounded in what you discovered during the Input Phase "
            "(category selectivity, activation patterns). Register INPUT hypotheses now; defer OUTPUT "
            "hypotheses until you know what the neuron fires on."
        )
        print(f"[PROTOCOL] ⚠️ Output hypothesis {hypothesis_id} registered BEFORE input phase complete — may be premature")
    else:
        print(f"[PROTOCOL] Hypothesis {hypothesis_id} registered (total: {state.hypotheses_registered + 1})")

    return {
        "hypothesis_id": hypothesis_id,
        "message": f"Hypothesis registered as {hypothesis_id}. Now run experiments to test it." + warning,
        "reminder": "After experiments, call update_hypothesis_status with your findings.",
        "registered": registered,
    }


async def tool_update_hypothesis_status(
    hypothesis_id: str,
    status: str,
    posterior_probability: int,
    evidence_summary: str,
) -> dict[str, Any]:
    """Update the status of a registered hypothesis after testing.

    Args:
        hypothesis_id: The hypothesis ID (e.g., "H1")
        status: New status: "confirmed", "refuted", or "inconclusive"
        posterior_probability: Your confidence (0-100%) AFTER testing
        evidence_summary: Brief summary of evidence supporting your conclusion

    Returns:
        Dict with updated hypothesis
    """
    import datetime

    for h in _HYPOTHESIS_REGISTRY:
        if h["hypothesis_id"] == hypothesis_id:
            timestamp = datetime.datetime.now().isoformat()

            # TEMPORAL ENFORCEMENT: Check for experiments run BEFORE hypothesis was registered
            hypothesis_registration_time = h.get("registered_at", "")
            experiments_before_registration = get_experiments_before_hypothesis(hypothesis_registration_time)
            temporal_warning = None

            if experiments_before_registration:
                # Count experiments by type
                exp_types = {}
                for exp in experiments_before_registration:
                    exp_type = exp.get("type", "unknown")
                    exp_types[exp_type] = exp_types.get(exp_type, 0) + 1

                exp_summary = ", ".join(f"{count} {etype}" for etype, count in exp_types.items())
                temporal_warning = (
                    f"⚠️ TEMPORAL ENFORCEMENT WARNING: {len(experiments_before_registration)} experiments "
                    f"({exp_summary}) were run BEFORE this hypothesis was registered at {hypothesis_registration_time}. "
                    f"This may indicate post-hoc hypothesis registration (p-hacking risk). "
                    f"Pre-registration is only valid if hypotheses are registered BEFORE seeing experimental results."
                )
                print(f"[PROTOCOL WARNING] {temporal_warning}")

            # Update current state
            h["status"] = status
            h["posterior_probability"] = posterior_probability
            h["evidence"].append(evidence_summary)

            # Add history entry tracking this update
            history_entry = {
                "timestamp": timestamp,
                "action": status,  # confirmed, refuted, inconclusive
                "prior": h.get("prior_probability"),
                "posterior": posterior_probability,
                "evidence": evidence_summary,
                "status": status,
            }

            # Ensure history list exists (for backwards compatibility)
            if "history" not in h:
                h["history"] = []
            h["history"].append(history_entry)

            # Record temporal enforcement metadata
            h["experiments_before_registration"] = len(experiments_before_registration)
            h["temporal_enforcement_passed"] = len(experiments_before_registration) == 0

            # Calculate Bayes factor (simple heuristic)
            prior = h["prior_probability"] / 100
            posterior = posterior_probability / 100
            if prior > 0 and prior < 1:
                # Likelihood ratio approximation
                if status == "confirmed":
                    bayes_factor = (posterior / prior) / ((1 - posterior) / (1 - prior)) if posterior < 1 else float('inf')
                else:
                    bayes_factor = ((1 - posterior) / (1 - prior)) / (posterior / prior) if posterior > 0 else float('inf')
            else:
                bayes_factor = None

            # Update protocol state
            state = get_protocol_state()
            update_protocol_state(hypotheses_updated=state.hypotheses_updated + 1)
            print(f"[PROTOCOL] Hypothesis {hypothesis_id} updated to {status} (total updates: {state.hypotheses_updated + 1})")

            result = {
                "hypothesis_id": hypothesis_id,
                "status": status,
                "prior_probability": h["prior_probability"],
                "posterior_probability": posterior_probability,
                "probability_shift": posterior_probability - h["prior_probability"],
                "bayes_factor": round(bayes_factor, 2) if bayes_factor and bayes_factor != float('inf') else bayes_factor,
                "interpretation": (
                    f"Updated {hypothesis_id} to {status}. "
                    f"Confidence shifted from {h['prior_probability']}% to {posterior_probability}% "
                    f"({'+' if posterior_probability > h['prior_probability'] else ''}{posterior_probability - h['prior_probability']}%)."
                ),
                "temporal_enforcement_passed": h["temporal_enforcement_passed"],
            }

            if temporal_warning:
                result["temporal_warning"] = temporal_warning

            return result

    return {"error": f"Hypothesis {hypothesis_id} not found in registry"}


# =============================================================================
# New Tools: Baseline Comparison, Adaptive RelP, Dose-Response Steering
# =============================================================================


async def tool_run_baseline_comparison(
    layer: int,
    neuron_idx: int,
    prompts: list[str],
    n_random_neurons: int = 30,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compare target neuron's effects against random neurons to calibrate effect sizes.

    This helps distinguish real effects from noise by showing how the target neuron
    compares to randomly selected neurons from the same layer.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Target neuron index
        prompts: List of prompts to test
        n_random_neurons: Number of random neurons to compare against (default 30)
        seed: Optional seed for reproducibility. If None, uses global seed or random.

    Returns:
        Dict with target neuron results, random neuron results, and z-scores
    """
    # Handle seed for reproducibility
    if seed is not None:
        set_seed(seed)
        used_seed = seed
    elif _GLOBAL_SEED is not None:
        used_seed = _GLOBAL_SEED
    else:
        # Generate and set a random seed for reproducibility logging
        used_seed = random.randint(0, 2**31 - 1)
        set_seed(used_seed)

    # Get random neurons from the same layer
    neurons_per_layer = get_model_config().neurons_per_layer
    random_indices = random.sample(
        [i for i in range(neurons_per_layer) if i != neuron_idx],
        min(n_random_neurons, neurons_per_layer - 1)
    )

    # Test target neuron
    target_results = await tool_batch_activation_test(layer, neuron_idx, prompts)
    target_activations = [
        r["activation"] for r in target_results.get("top_activating", [])
    ]
    target_mean = target_results.get("mean_activation", 0)
    target_max = target_results.get("max_activation", 0)

    # Test random neurons
    random_results = []
    all_random_activations = []

    for rand_idx in random_indices:
        rand_result = await tool_batch_activation_test(layer, rand_idx, prompts)
        random_results.append({
            "neuron_idx": rand_idx,
            "mean_activation": rand_result.get("mean_activation", 0),
            "max_activation": rand_result.get("max_activation", 0),
            "activating_count": rand_result.get("activating_count", 0),
        })
        all_random_activations.append(rand_result.get("mean_activation", 0))

    # Compute statistics
    if all_random_activations:
        baseline_mean = sum(all_random_activations) / len(all_random_activations)
        # Use sample variance (N-1) instead of population variance (N) for unbiased estimate
        n = len(all_random_activations)
        baseline_variance = sum((x - baseline_mean) ** 2 for x in all_random_activations) / (n - 1) if n > 1 else 0
        baseline_std = baseline_variance ** 0.5 if baseline_variance > 0 else 1.0

        # Z-score: how many standard deviations above baseline
        z_score = (target_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
    else:
        baseline_mean = 0
        baseline_std = 1.0
        z_score = 0

    # Determine if effect is meaningful (z > 2 is conventional threshold)
    is_meaningful = abs(z_score) > 2.0

    # Update protocol state
    update_protocol_state(
        baseline_comparison_done=True,
        baseline_zscore=round(z_score, 2),
        baseline_prompts_tested=len(prompts)
    )
    print(f"[PROTOCOL] Baseline comparison completed, z-score={z_score:.2f}")

    return {
        "target_neuron": {
            "layer": layer,
            "neuron_idx": neuron_idx,
            "mean_activation": target_mean,
            "max_activation": target_max,
            "activating_count": target_results.get("activating_count", 0),
        },
        "baseline": {
            "n_random_neurons": len(random_results),
            "random_neurons": random_results[:10],  # Truncate for readability, keep first 10
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
        },
        "comparison": {
            "z_score": round(z_score, 2),
            "is_meaningful": is_meaningful,
            "effect_vs_baseline": f"{target_mean / baseline_mean:.1f}x" if baseline_mean > 0 else "N/A",
            "interpretation": (
                f"Target neuron activates {abs(z_score):.1f} standard deviations "
                f"{'above' if z_score > 0 else 'below'} random neurons (n={len(random_results)}). "
                f"{'This is statistically meaningful (z > 2).' if is_meaningful else 'This may be noise (z < 2).'}"
            ),
        },
        "prompts_tested": len(prompts),
        "seed_used": used_seed,  # For reproducibility
        "random_neuron_indices": random_indices,  # Log which neurons were sampled
    }


async def tool_adaptive_relp(
    layer: int,
    neuron_idx: int,
    prompt: str,
    target_tokens: list[str] | None = None,
    max_time: float = 60.0,
    tau_schedule: list[float] | None = None,
) -> dict[str, Any]:
    """Run RelP with adaptive tau - start coarse and refine until neuron is found.

    This avoids the guesswork of choosing tau: it starts with fast, coarse graphs
    and progressively increases detail until the target neuron appears.

    Args:
        layer: Layer of the neuron to look for
        neuron_idx: Index of the neuron to look for
        prompt: The prompt to run attribution on
        target_tokens: Optional list of specific tokens to trace
        max_time: Maximum total time in seconds (default 60)
        tau_schedule: Optional list of tau values to try (default: [0.1, 0.05, 0.02, 0.01, 0.005])

    Returns:
        Dict with the result from the smallest tau where the neuron was found,
        or the most detailed result if neuron never found.
    """
    import time

    if tau_schedule is None:
        tau_schedule = [0.1, 0.05, 0.02, 0.01, 0.005]

    start_time = time.time()
    results_by_tau = []
    best_result = None

    for tau in tau_schedule:
        elapsed = time.time() - start_time
        remaining_time = max_time - elapsed

        if remaining_time <= 5:
            break  # Not enough time for another iteration

        result = await tool_run_relp(
            layer, neuron_idx, prompt,
            target_tokens=target_tokens,
            tau=tau,
            timeout=min(remaining_time, 30.0)  # Cap individual runs at 30s
        )

        nodes_exceeded = result.get("nodes_exceeded", False)

        results_by_tau.append({
            "tau": tau,
            "neuron_found": result.get("neuron_found", False),
            "total_nodes": result.get("graph_stats", {}).get("total_nodes", 0),
            "elapsed": result.get("graph_stats", {}).get("elapsed_seconds", 0),
            "nodes_exceeded": nodes_exceeded,
        })

        if result.get("neuron_found"):
            best_result = result
            best_result["adaptive_info"] = {
                "tau_found_at": tau,
                "tau_schedule": tau_schedule,
                "iterations_tried": len(results_by_tau),
                "total_elapsed": time.time() - start_time,
            }
            break
        elif nodes_exceeded:
            # Graph exceeded max_nodes - no point going to lower tau values
            best_result = result
            best_result["adaptive_info"] = {
                "tau_found_at": None,
                "tau_schedule": tau_schedule,
                "iterations_tried": len(results_by_tau),
                "total_elapsed": time.time() - start_time,
                "stopped_reason": f"nodes_exceeded at tau={tau}",
                "message": f"Graph exceeded max_nodes at tau={tau}. Neuron may not be in causal pathway for this prompt.",
            }
            break
        else:
            # Keep the most detailed result even if neuron not found
            best_result = result

    if best_result is None:
        return {"error": "No RelP results obtained within time limit"}

    if not best_result.get("neuron_found"):
        best_result["adaptive_info"] = {
            "tau_found_at": None,
            "tau_schedule": tau_schedule,
            "iterations_tried": len(results_by_tau),
            "total_elapsed": time.time() - start_time,
            "message": "Neuron not found at any tau level. It may not be in the causal pathway for this prompt.",
        }

    best_result["tau_progression"] = results_by_tau

    return best_result


async def tool_steer_dose_response(
    layer: int,
    neuron_idx: int,
    prompt: str,
    steering_values: list[float] | None = None,
    position: int = -1,
    top_k_logits: int = 10,
) -> dict[str, Any]:
    """Run steering at multiple values to generate a dose-response curve.

    This tests the causal effect of the neuron at various intensities,
    revealing whether effects are linear, threshold-based, or saturating.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index
        prompt: Input prompt
        steering_values: List of steering values to test (default: [-10, -5, -2, 0, 2, 5, 10])
        position: Position to steer (-1 for last token)
        top_k_logits: Number of top logits to return per steering value

    Returns:
        Dict with dose-response curve data for analysis
    """
    if steering_values is None:
        steering_values = [-10, -5, -2, 0, 2, 5, 10]

    results = []
    token_effects = {}  # Track how each token changes across steering values

    for value in steering_values:
        result = await tool_steer_neuron(
            layer, neuron_idx, prompt,
            steering_value=value,
            position=position,
            top_k_logits=top_k_logits
        )

        results.append({
            "steering_value": value,
            "promoted_tokens": result.get("promoted_tokens", []),
            "suppressed_tokens": result.get("suppressed_tokens", []),
            "max_shift": result.get("max_shift", 0),
        })

        # Track individual token trajectories
        for token, logit in result.get("steered_logits", {}).items():
            if token not in token_effects:
                token_effects[token] = {"values": [], "logits": []}
            token_effects[token]["values"].append(value)
            token_effects[token]["logits"].append(logit)

    # Analyze the dose-response pattern
    max_shifts_by_value = {r["steering_value"]: r["max_shift"] for r in results}

    # Check for linearity (simple heuristic)
    positive_values = [v for v in steering_values if v > 0]
    negative_values = [v for v in steering_values if v < 0]

    linearity_check = "unknown"
    if len(positive_values) >= 2:
        shifts = [max_shifts_by_value.get(v, 0) for v in sorted(positive_values)]
        # Check if roughly monotonic
        if all(shifts[i] <= shifts[i+1] for i in range(len(shifts)-1)):
            linearity_check = "monotonic_positive"
        elif all(shifts[i] >= shifts[i+1] for i in range(len(shifts)-1)):
            linearity_check = "monotonic_negative"
        else:
            linearity_check = "non_monotonic"

    # Find tokens with strongest dose-response
    responsive_tokens = []
    for token, data in token_effects.items():
        if len(data["logits"]) >= 3:
            logit_range = max(data["logits"]) - min(data["logits"])
            if logit_range > 0.5:  # Meaningful response
                responsive_tokens.append({
                    "token": token,
                    "logit_range": round(logit_range, 2),
                    "curve": list(zip(data["values"], [round(l, 2) for l in data["logits"]])),
                })

    responsive_tokens.sort(key=lambda x: -x["logit_range"])

    # Compute Kendall's tau for monotonicity measure
    # Using a simple implementation (scipy would be better but avoiding dep)
    kendall_tau = None
    if len(steering_values) >= 3 and len(results) >= 3:
        # Pairs analysis: count concordant vs discordant
        concordant = 0
        discordant = 0
        sorted_pairs = sorted(zip(steering_values, [r["max_shift"] for r in results]))
        for i in range(len(sorted_pairs)):
            for j in range(i + 1, len(sorted_pairs)):
                v1, s1 = sorted_pairs[i]
                v2, s2 = sorted_pairs[j]
                if (v2 - v1) * (s2 - s1) > 0:
                    concordant += 1
                elif (v2 - v1) * (s2 - s1) < 0:
                    discordant += 1
        total_pairs = concordant + discordant
        if total_pairs > 0:
            kendall_tau = (concordant - discordant) / total_pairs

    is_monotonic = linearity_check.startswith("monotonic")

    # Build result object for storage
    dose_response_result = {
        "prompt": prompt[:200],
        "steering_values": steering_values,
        "pattern": linearity_check,
        "kendall_tau": round(kendall_tau, 3) if kendall_tau is not None else None,
        "is_monotonic": is_monotonic,
        "responsive_tokens": responsive_tokens[:10],
        "dose_response_curve": results,
    }

    # Update protocol state with metadata and accumulate full results
    protocol_state = get_protocol_state()
    protocol_state.dose_response_done = True
    # Only update monotonic if this is the first or if still monotonic
    if is_monotonic:
        protocol_state.dose_response_monotonic = True
    # Keep the best (highest absolute) kendall_tau
    if kendall_tau is not None:
        if protocol_state.dose_response_kendall_tau is None or abs(kendall_tau) > abs(protocol_state.dose_response_kendall_tau):
            protocol_state.dose_response_kendall_tau = round(kendall_tau, 3)
    # Accumulate results
    protocol_state.dose_response_results.append(dose_response_result)

    tau_str = f"{kendall_tau:.2f}" if kendall_tau is not None else "N/A"
    print(f"[PROTOCOL] Dose-response completed, pattern={linearity_check}, Kendall's tau={tau_str}")

    return {
        "prompt": prompt[:100],
        "layer": layer,
        "neuron_idx": neuron_idx,
        "steering_values": steering_values,
        "dose_response_curve": results,
        "pattern": linearity_check,
        "kendall_tau": round(kendall_tau, 3) if kendall_tau is not None else None,
        "is_monotonic": is_monotonic,
        "responsive_tokens": responsive_tokens[:10],
        "summary": {
            "strongest_positive_effect": max(max_shifts_by_value.values()) if max_shifts_by_value else 0,
            "strongest_negative_effect": min(max_shifts_by_value.values()) if max_shifts_by_value else 0,
            "n_responsive_tokens": len([t for t in responsive_tokens if t["logit_range"] > 0.5]),
        },
    }


# =============================================================================
# Graph Index Tools - Query pre-indexed RelP graphs
# =============================================================================

async def tool_find_graphs_for_neuron(
    layer: int,
    neuron_idx: int,
    limit: int = 50,
    min_influence: float = 0.0,
) -> dict[str, Any]:
    """Find pre-computed RelP graphs where a specific neuron appears.

    This queries the neuron-to-graph index database to quickly find graphs
    containing the target neuron, without needing to generate new RelP graphs.
    Use this to discover what contexts/prompts activate the neuron.

    **Important**: All indexed graphs are computed via relevance patching (RelP)
    with respect to the **final token position**. This means the graphs show
    which neurons influence the model's prediction at the last token.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index within layer
        limit: Maximum number of graphs to return (default 50)
        min_influence: Minimum influence score to include (default 0)

    Returns:
        Dict with:
            - neuron_id: The neuron identifier (e.g., "L15/N7890")
            - total_graphs: Total number of graphs containing this neuron
            - graphs: List of graph entries with paths and influence scores
            - sample_prompts: Preview of prompts from top graphs (with target token info)
    """
    from neuron_scientist.graph_index import DEFAULT_DB_PATH, GraphIndexDB

    if not DEFAULT_DB_PATH.exists():
        return {
            "error": "Graph index database not found. Run scripts/build_neuron_index.py first.",
            "db_path": str(DEFAULT_DB_PATH),
        }

    db = GraphIndexDB()

    # Get frequency stats
    freq = db.get_neuron_frequency(layer, neuron_idx)

    # Get graph list
    graphs = db.get_graphs_for_neuron(
        layer, neuron_idx, limit=limit, min_influence=min_influence
    )

    # Get sample prompts from top graphs
    sample_prompts = []
    for g in graphs[:5]:
        meta = db.get_graph_metadata(g["graph_path"])
        if meta and meta.get("prompt_preview"):
            sample_prompts.append({
                "graph": g["graph_path"],
                "influence": g["influence_score"],
                "prompt": meta["prompt_preview"],
            })

    # Update protocol state - Phase 0 corpus queried
    update_protocol_state(
        phase0_corpus_queried=True,
        phase0_graph_count=freq["graph_count"],
        phase0_sample_prompts=[p["prompt"] for p in sample_prompts[:5]]
    )
    print(f"[PROTOCOL] Phase 0 corpus query completed: found {freq['graph_count']} graphs")

    return {
        "neuron_id": f"L{layer}/N{neuron_idx}",
        "layer": layer,
        "neuron_idx": neuron_idx,
        "total_graphs": freq["graph_count"],
        "avg_influence": round(freq["avg_influence"], 3) if freq["avg_influence"] else None,
        "max_influence": round(freq["max_influence"], 3) if freq["max_influence"] else None,
        "graphs": graphs,
        "sample_prompts": sample_prompts,
    }


async def tool_get_neuron_graph_stats(
    layer: int,
    neuron_idx: int,
) -> dict[str, Any]:
    """Get statistics about a neuron's presence across indexed graphs.

    Returns frequency of appearance, influence distribution, and
    neurons that frequently co-occur with this neuron (with labels).

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index within layer

    Returns:
        Dict with frequency stats and top co-occurring neurons (labeled)
    """
    from neuron_scientist.graph_index import DEFAULT_DB_PATH, GraphIndexDB

    if not DEFAULT_DB_PATH.exists():
        return {
            "error": "Graph index database not found. Run scripts/build_neuron_index.py first.",
            "db_path": str(DEFAULT_DB_PATH),
        }

    db = GraphIndexDB()

    # Get frequency stats
    freq = db.get_neuron_frequency(layer, neuron_idx)

    # Get co-occurring neurons
    cooccur = db.find_cooccurring_neurons(layer, neuron_idx, limit=20)

    # Get database stats for context
    total_graphs = db.get_total_graphs()

    # Look up labels for co-occurring neurons (with NeuronDB fallback)
    cooccur_ids = [c.get("neuron_id") for c in cooccur if c.get("neuron_id")]
    labels = batch_get_neuron_labels_with_fallback(cooccur_ids) if cooccur_ids else {}

    # Add labels to co-occurring neurons
    for c in cooccur:
        nid = c.get("neuron_id")
        if nid:
            label_info = labels.get(nid, {})
            c["label"] = label_info.get("label", "")
            c["label_source"] = label_info.get("source", None)

    # Update protocol state - Phase 0 also counts as queried via stats
    state = get_protocol_state()
    if not state.phase0_corpus_queried:
        update_protocol_state(
            phase0_corpus_queried=True,
            phase0_graph_count=freq["graph_count"],
        )
        print(f"[PROTOCOL] Phase 0 graph stats queried: {freq['graph_count']} graphs found")

    return {
        "neuron_id": f"L{layer}/N{neuron_idx}",
        "layer": layer,
        "neuron_idx": neuron_idx,
        "graph_count": freq["graph_count"],
        "total_indexed_graphs": total_graphs,
        "appearance_rate": round(freq["graph_count"] / total_graphs, 4) if total_graphs > 0 else 0,
        "avg_influence": round(freq["avg_influence"], 3) if freq["avg_influence"] else None,
        "max_influence": round(freq["max_influence"], 3) if freq["max_influence"] else None,
        "min_influence": round(freq["min_influence"], 3) if freq["min_influence"] else None,
        "top_cooccurring_neurons": cooccur,
    }


async def tool_load_graph_from_index(
    graph_path: str,
    graphs_base_dir: str = "graphs/fabric_fineweb_50k",
) -> dict[str, Any]:
    """Load and return a specific RelP graph from the index.

    Use this after finding graphs with tool_find_graphs_for_neuron to
    inspect the full graph structure, including all nodes and edges.

    **Important**: All graphs are computed via relevance patching (RelP) with
    respect to the **final token position**. The graph shows which neurons
    influence the model's prediction at the last token.

    Args:
        graph_path: Relative path to graph file (from index results)
        graphs_base_dir: Base directory for graphs (default: fabric_fineweb_50k)

    Returns:
        Dict with graph metadata, nodes (with labels), and edges
    """
    import json
    from pathlib import Path

    full_path = Path(graphs_base_dir) / graph_path
    if not full_path.exists():
        # Try absolute path
        full_path = Path(graph_path)
        if not full_path.exists():
            return {"error": f"Graph file not found: {graph_path}"}

    try:
        with open(full_path) as f:
            data = json.load(f)

        metadata = data.get("metadata", {})

        # Extract MLP neurons
        mlp_nodes = []
        for n in data.get("nodes", []):
            if n.get("feature_type") == "mlp_neuron":
                mlp_nodes.append({
                    "node_id": n["node_id"],
                    "layer": n["layer"],
                    "neuron_idx": n["feature"],
                    "neuron_id": f"L{n['layer']}/N{n['feature']}",
                    "ctx_idx": n.get("ctx_idx"),
                    "influence": n.get("influence"),
                    "activation": n.get("activation"),
                })

        # Look up labels for MLP neurons (with NeuronDB fallback)
        neuron_ids = [m["neuron_id"] for m in mlp_nodes]
        labels = batch_get_neuron_labels_with_fallback(neuron_ids) if neuron_ids else {}

        # Add labels to neurons
        for m in mlp_nodes:
            label_info = labels.get(m["neuron_id"], {})
            m["label"] = label_info.get("label", "")
            m["label_source"] = label_info.get("source", None)

        # Sort by influence (descending) and limit
        mlp_nodes = sorted(mlp_nodes, key=lambda x: abs(x.get("influence", 0)), reverse=True)

        # Extract logit nodes (target tokens the graph influences)
        logit_nodes = [
            {
                "node_id": n["node_id"],
                "token": n.get("clerp", ""),
                "ctx_idx": n.get("ctx_idx"),
            }
            for n in data.get("nodes", [])
            if n.get("feature_type") == "logit"
        ]

        # Summarize edges
        edges = data.get("links", [])

        # Extract target token from logit nodes
        target_tokens = [ln["token"] for ln in logit_nodes if ln.get("token")]

        return {
            "graph_path": str(graph_path),
            "prompt": metadata.get("original_prompt", metadata.get("prompt", ""))[:500],
            "target_tokens": target_tokens,  # What tokens the graph traces influence to
            "target_position": "last",  # All indexed graphs use last token position
            "source": metadata.get("source"),
            "num_nodes": len(data.get("nodes", [])),
            "num_edges": len(edges),
            "num_mlp_neurons": len(mlp_nodes),
            "mlp_neurons": mlp_nodes[:100],  # Increased from 50 to 100
            "logit_nodes": logit_nodes,
            "sample_edges": edges[:30],  # Increased from 20 to 30
        }

    except Exception as e:
        return {"error": f"Failed to load graph: {str(e)}"}


async def tool_batch_relp_verify_connections(
    layer: int,
    neuron_idx: int,
    upstream_neurons: list[str],
    downstream_neurons: list[str],
    max_graphs: int = 20,
    graphs_base_dir: str = "graphs/fabric_fineweb_50k",
) -> dict[str, Any]:
    """Verify weight-predicted wiring connections against corpus RelP graphs.

    Takes the target neuron + predicted upstream/downstream neuron IDs from wiring
    analysis, then checks actual RelP corpus graphs to see which predictions appear
    as real edge endpoints. This bridges weight-based predictions with empirical data.

    Args:
        layer: Target neuron layer (0-31)
        neuron_idx: Target neuron index
        upstream_neurons: List of predicted upstream neuron IDs, e.g. ["L14/N4466", "L12/N890"]
        downstream_neurons: List of predicted downstream neuron IDs, e.g. ["L20/N1234"]
        max_graphs: Maximum graphs to check (default 20)
        graphs_base_dir: Base directory for graph files

    Returns:
        Dict with per-connection verification results:
            - upstream_results: [{neuron_id, found_in_n_graphs, graphs_checked, relp_confirmed, avg_edge_weight}]
            - downstream_results: [{neuron_id, found_in_n_graphs, graphs_checked, relp_confirmed, avg_edge_weight}]
            - graphs_checked: Total graphs examined
            - summary: Text summary of verification results
    """
    import json
    from collections import defaultdict
    from pathlib import Path

    from neuron_scientist.graph_index import DEFAULT_DB_PATH, GraphIndexDB

    if not DEFAULT_DB_PATH.exists():
        return {"error": "Graph index database not found. Run scripts/build_neuron_index.py first."}

    db = GraphIndexDB()
    target_id = f"L{layer}/N{neuron_idx}"

    # Get graphs containing the target neuron
    graphs = db.get_graphs_for_neuron(layer, neuron_idx, limit=max_graphs, min_influence=0.0)
    if not graphs:
        return {
            "neuron_id": target_id,
            "graphs_checked": 0,
            "upstream_results": [],
            "downstream_results": [],
            "summary": f"No corpus graphs found containing {target_id}.",
        }

    # Parse neuron IDs into (layer, idx) tuples for fast matching
    def parse_neuron_id(nid: str):
        """Parse 'L14/N4466' -> (14, 4466)"""
        try:
            parts = nid.replace("L", "").replace("N", "").split("/")
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return None

    upstream_set = {}
    for nid in upstream_neurons:
        parsed = parse_neuron_id(nid)
        if parsed:
            upstream_set[parsed] = nid

    downstream_set = {}
    for nid in downstream_neurons:
        parsed = parse_neuron_id(nid)
        if parsed:
            downstream_set[parsed] = nid

    # Track per-connection results
    upstream_hits = defaultdict(lambda: {"count": 0, "edge_weights": []})
    downstream_hits = defaultdict(lambda: {"count": 0, "edge_weights": []})
    graphs_checked = 0

    for g in graphs:
        graph_path = g.get("graph_path", "")
        full_path = Path(graphs_base_dir) / graph_path
        if not full_path.exists():
            full_path = Path(graph_path)
            if not full_path.exists():
                continue

        try:
            with open(full_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        graphs_checked += 1

        # Build node lookup: node_id -> (layer, neuron_idx)
        node_map = {}
        for n in data.get("nodes", []):
            if n.get("feature_type") == "mlp_neuron":
                node_map[n["node_id"]] = (n["layer"], n["feature"])

        # Find target neuron's node_id(s) in this graph
        target_node_ids = set()
        for node_id, (nl, ni) in node_map.items():
            if nl == layer and ni == neuron_idx:
                target_node_ids.add(node_id)

        if not target_node_ids:
            continue

        # Check edges for upstream/downstream connections
        for edge in data.get("links", []):
            src = edge.get("source")
            tgt = edge.get("target")
            weight = abs(edge.get("weight", 0))

            # Upstream: src is a predicted upstream neuron, tgt is our target
            if tgt in target_node_ids and src in node_map:
                src_key = node_map[src]
                if src_key in upstream_set:
                    nid = upstream_set[src_key]
                    upstream_hits[nid]["count"] += 1
                    upstream_hits[nid]["edge_weights"].append(weight)

            # Downstream: src is our target, tgt is a predicted downstream neuron
            if src in target_node_ids and tgt in node_map:
                tgt_key = node_map[tgt]
                if tgt_key in downstream_set:
                    nid = downstream_set[tgt_key]
                    downstream_hits[nid]["count"] += 1
                    downstream_hits[nid]["edge_weights"].append(weight)

    # Build results
    upstream_results = []
    for nid in upstream_neurons:
        hits = upstream_hits.get(nid, {"count": 0, "edge_weights": []})
        avg_weight = sum(hits["edge_weights"]) / len(hits["edge_weights"]) if hits["edge_weights"] else 0.0
        upstream_results.append({
            "neuron_id": nid,
            "found_in_n_graphs": hits["count"],
            "graphs_checked": graphs_checked,
            "relp_confirmed": hits["count"] > 0,
            "avg_edge_weight": round(avg_weight, 4),
        })

    downstream_results = []
    for nid in downstream_neurons:
        hits = downstream_hits.get(nid, {"count": 0, "edge_weights": []})
        avg_weight = sum(hits["edge_weights"]) / len(hits["edge_weights"]) if hits["edge_weights"] else 0.0
        downstream_results.append({
            "neuron_id": nid,
            "found_in_n_graphs": hits["count"],
            "graphs_checked": graphs_checked,
            "relp_confirmed": hits["count"] > 0,
            "avg_edge_weight": round(avg_weight, 4),
        })

    # Build summary
    up_confirmed = sum(1 for r in upstream_results if r["relp_confirmed"])
    down_confirmed = sum(1 for r in downstream_results if r["relp_confirmed"])
    summary = (
        f"Checked {graphs_checked} corpus graphs for {target_id}. "
        f"Upstream: {up_confirmed}/{len(upstream_results)} confirmed. "
        f"Downstream: {down_confirmed}/{len(downstream_results)} confirmed."
    )

    print(f"[RELP VERIFY] {summary}")

    # Store ALL results in the global relp_verification_results dict
    # This persists across calls and is used at save time for enrichment
    state = get_protocol_state()
    for r in upstream_results + downstream_results:
        state.relp_verification_results[r["neuron_id"]] = {
            "relp_confirmed": r["relp_confirmed"],
            "relp_strength": r["avg_edge_weight"],
        }
    print(f"[RELP VERIFY] Stored {len(upstream_results) + len(downstream_results)} results "
          f"(total: {len(state.relp_verification_results)} neurons verified)")

    # Re-sort connectivity_data to prioritize RelP-confirmed neurons for experiments
    # Priority: RelP-confirmed first (by |weight|), then unchecked (by |weight|), then denied
    if state.connectivity_data:
        def _relp_priority_sort(neurons_list):
            """Sort neurons: RelP-confirmed first, then unchecked, then denied."""
            confirmed, unchecked, denied = [], [], []
            for n in neurons_list:
                nid = n.get("neuron_id", "")
                rv = state.relp_verification_results.get(nid)
                if rv and rv["relp_confirmed"]:
                    confirmed.append(n)
                elif rv and not rv["relp_confirmed"]:
                    denied.append(n)
                else:
                    unchecked.append(n)
            # Within each group, sort by |weight|
            key = lambda x: abs(x.get("effective_strength", x.get("weight", 0)))
            return sorted(confirmed, key=key, reverse=True) + sorted(unchecked, key=key, reverse=True) + sorted(denied, key=key, reverse=True)

        if state.connectivity_data.get("upstream_neurons"):
            state.connectivity_data["upstream_neurons"] = _relp_priority_sort(state.connectivity_data["upstream_neurons"])
        if state.connectivity_data.get("downstream_targets"):
            state.connectivity_data["downstream_targets"] = _relp_priority_sort(state.connectivity_data["downstream_targets"])
            n_conf = sum(1 for d in state.connectivity_data["downstream_targets"]
                        if state.relp_verification_results.get(d.get("neuron_id", ""), {}).get("relp_confirmed"))
            print(f"[RELP VERIFY] Re-sorted downstream: {n_conf} RelP-confirmed neurons prioritized for experiments")

    return {
        "neuron_id": target_id,
        "graphs_checked": graphs_checked,
        "upstream_results": upstream_results,
        "downstream_results": downstream_results,
        "summary": summary,
    }


async def tool_extract_corpus_relp_evidence(
    layer: int,
    neuron_idx: int,
    max_graphs: int = 10,
    graphs_base_dir: str = "",
) -> dict[str, Any]:
    """Extract aggregated RelP evidence from corpus graphs containing this neuron.

    Loads multiple graphs where the neuron appears and aggregates:
    - Upstream neurons that influence this neuron
    - Downstream neurons/logits this neuron influences
    - Token contexts where the neuron activates
    - Output token associations (what logits it promotes/suppresses)

    This provides pre-existing evidence without running new experiments.

    Args:
        layer: Layer number
        neuron_idx: Neuron index within layer
        max_graphs: Maximum number of graphs to analyze (default 10)
        graphs_base_dir: Base directory for graph files (auto-detected from model config if empty)

    Returns:
        Dict with aggregated upstream, downstream, and context evidence
    """
    import json
    from collections import defaultdict
    from pathlib import Path

    if not graphs_base_dir:
        graphs_base_dir = get_model_config().graphs_dir or "graphs/fabric_fineweb_50k"

    from neuron_scientist.graph_index import DEFAULT_DB_PATH, GraphIndexDB

    if not DEFAULT_DB_PATH.exists():
        return {"error": "Graph index database not found. Run scripts/data/build_neuron_index.py first."}

    db = GraphIndexDB()

    # Get top graphs containing this neuron
    graphs = db.get_graphs_for_neuron(layer, neuron_idx, limit=max_graphs, min_influence=0.1)

    if not graphs:
        return {
            "neuron_id": f"L{layer}/N{neuron_idx}",
            "graphs_analyzed": 0,
            "error": "No graphs found containing this neuron",
        }

    # Aggregators
    upstream_weights = defaultdict(lambda: {"total_weight": 0.0, "count": 0, "graphs": []})
    downstream_weights = defaultdict(lambda: {"total_weight": 0.0, "count": 0, "graphs": []})
    logit_effects = defaultdict(lambda: {"total_weight": 0.0, "count": 0, "sign": 0})
    context_tokens = []  # [(prompt, position, token, influence)]
    prompts_analyzed = []

    graphs_loaded = 0

    for g in graphs:
        graph_path = Path(graphs_base_dir) / g["graph_path"]
        if not graph_path.exists():
            continue

        try:
            with open(graph_path) as f:
                data = json.load(f)

            graphs_loaded += 1
            prompt = data.get("metadata", {}).get("original_prompt", "")[:200]
            prompts_analyzed.append({
                "prompt": prompt,
                "influence": g["influence_score"],
                "graph": g["graph_path"],
            })

            nodes = data.get("nodes", [])
            links = data.get("links", [])

            # Build node lookup
            node_lookup = {n.get("node_id"): n for n in nodes}

            # Find target neuron node IDs
            target_node_ids = [
                n.get("node_id")
                for n in nodes
                if n.get("layer") == layer and n.get("feature") == neuron_idx
            ]

            # Extract context tokens where neuron appears
            for n in nodes:
                if n.get("layer") == layer and n.get("feature") == neuron_idx:
                    pos = n.get("ctx_idx")
                    influence = n.get("influence", 0)
                    # Get token at position from prompt if available
                    tokens = data.get("tokens", [])
                    token = tokens[pos] if tokens and pos < len(tokens) else ""
                    context_tokens.append({
                        "prompt": prompt[:100],
                        "position": pos,
                        "token": token,
                        "influence": influence,
                        "activation": n.get("activation", 0),
                    })

            # Aggregate upstream connections (links TO target)
            for link in links:
                if link.get("target") in target_node_ids:
                    src_node = node_lookup.get(link.get("source"), {})
                    src_layer = src_node.get("layer")
                    src_feature = src_node.get("feature")
                    if src_layer is not None and src_feature is not None:
                        key = f"L{src_layer}/N{src_feature}"
                        weight = link.get("weight", 0)
                        upstream_weights[key]["total_weight"] += weight
                        upstream_weights[key]["count"] += 1
                        if g["graph_path"] not in upstream_weights[key]["graphs"]:
                            upstream_weights[key]["graphs"].append(g["graph_path"])

            # Aggregate downstream connections (links FROM target)
            for link in links:
                if link.get("source") in target_node_ids:
                    tgt_node = node_lookup.get(link.get("target"), {})
                    tgt_layer = tgt_node.get("layer")
                    tgt_feature = tgt_node.get("feature")
                    weight = link.get("weight", 0)

                    # Check if it's a logit node
                    if tgt_node.get("feature_type") == "logit":
                        token = tgt_node.get("clerp", f"token_{tgt_feature}")
                        logit_effects[token]["total_weight"] += weight
                        logit_effects[token]["count"] += 1
                        logit_effects[token]["sign"] += 1 if weight > 0 else -1
                    elif tgt_layer is not None and tgt_feature is not None:
                        key = f"L{tgt_layer}/N{tgt_feature}"
                        downstream_weights[key]["total_weight"] += weight
                        downstream_weights[key]["count"] += 1
                        if g["graph_path"] not in downstream_weights[key]["graphs"]:
                            downstream_weights[key]["graphs"].append(g["graph_path"])

        except Exception:
            continue

    # Sort and format results
    def format_connections(conn_dict, top_k=15):
        items = [
            {
                "neuron_id": k,
                "avg_weight": round(v["total_weight"] / v["count"], 4) if v["count"] > 0 else 0,
                "total_weight": round(v["total_weight"], 4),
                "graph_count": len(v["graphs"]),
            }
            for k, v in conn_dict.items()
        ]
        # Sort by absolute total weight
        items.sort(key=lambda x: abs(x["total_weight"]), reverse=True)
        return items[:top_k]

    # Get formatted connections
    upstream_result = format_connections(upstream_weights)
    downstream_result = format_connections(downstream_weights)

    # Look up labels for upstream and downstream neurons (with NeuronDB fallback)
    all_neuron_ids = [u["neuron_id"] for u in upstream_result] + [d["neuron_id"] for d in downstream_result]
    labels = batch_get_neuron_labels_with_fallback(all_neuron_ids) if all_neuron_ids else {}

    # Add labels to neurons
    for u in upstream_result:
        label_info = labels.get(u["neuron_id"], {})
        u["label"] = label_info.get("label", "")
        u["label_source"] = label_info.get("source", None)

    for d in downstream_result:
        label_info = labels.get(d["neuron_id"], {})
        d["label"] = label_info.get("label", "")
        d["label_source"] = label_info.get("source", None)

    # Format logit effects
    logit_promotes = []
    logit_suppresses = []
    for token, data in logit_effects.items():
        entry = {
            "token": token,
            "avg_weight": round(data["total_weight"] / data["count"], 4) if data["count"] > 0 else 0,
            "graph_count": data["count"],
        }
        if data["sign"] > 0:
            logit_promotes.append(entry)
        else:
            logit_suppresses.append(entry)

    logit_promotes.sort(key=lambda x: abs(x["avg_weight"]), reverse=True)
    logit_suppresses.sort(key=lambda x: abs(x["avg_weight"]), reverse=True)

    # Update protocol state - this counts as RelP evidence from corpus
    update_protocol_state(relp_positive_control=graphs_loaded > 0)
    if graphs_loaded > 0:
        print(f"[PROTOCOL] Corpus RelP evidence extracted from {graphs_loaded} graphs")

    result = {
        "neuron_id": f"L{layer}/N{neuron_idx}",
        "graphs_analyzed": graphs_loaded,
        "target_position": "last",  # All indexed graphs use last token position
        "prompts_analyzed": prompts_analyzed,
        "upstream_neurons": upstream_result,
        "downstream_neurons": downstream_result,
        "output_effects": {
            "promotes": logit_promotes[:10],
            "suppresses": logit_suppresses[:10],
        },
        "context_examples": sorted(context_tokens, key=lambda x: -abs(x["influence"]))[:20],
        "summary": {
            "total_upstream_connections": len(upstream_weights),
            "total_downstream_connections": len(downstream_weights),
            "total_logit_effects": len(logit_effects),
            "avg_influence": round(sum(g["influence_score"] for g in graphs[:graphs_loaded]) / graphs_loaded, 3) if graphs_loaded > 0 else 0,
        },
    }

    return result


# =============================================================================
# Category Selectivity Testing
# =============================================================================

def _sync_category_selectivity_test(
    layer: int,
    neuron_idx: int,
    categorized_prompts: dict[str, list[str]],
    category_types: dict[str, str],  # category_name -> "target" | "control" | "inhibitory" | "unrelated"
) -> dict[str, Any]:
    """Synchronous category selectivity test. Runs on dedicated CUDA thread.

    Tests neuron selectivity across categorized prompts and computes z-scores.
    """
    if not categorized_prompts:
        return {"error": "No prompts provided"}

    # Collect all activations with category labels
    all_activations = []  # [(max_act, prompt, category, max_pos, max_token, token_acts, min_act, min_pos, min_token), ...]
    category_activations = {}  # category -> [(max_act, prompt, max_pos, max_token, token_acts, min_act, min_pos, min_token), ...]
    errors = []

    for category, prompts in categorized_prompts.items():
        if category not in category_activations:
            category_activations[category] = []

        # Format prompts with chat template
        texts = [format_prompt(p) for p in prompts]

        # Process in batches
        for batch_start in range(0, len(texts), MAX_BATCH_SIZE):
            batch_end = min(batch_start + MAX_BATCH_SIZE, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_prompts = prompts[batch_start:batch_end]

            try:
                batch_results = get_all_activations_batch(layer, neuron_idx, batch_texts)

                for prompt, all_acts in zip(batch_prompts, batch_results):
                    if not all_acts:
                        errors.append({"prompt": prompt[:80], "category": category, "error": "No activations"})
                        continue

                    # Find max activation (global)
                    max_pos, max_act, max_token = max(all_acts, key=lambda x: x[1])

                    # Find where user content starts and ends (between user header and assistant header)
                    # For Llama chat template: <|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>
                    user_content_start = 0
                    user_content_end = len(all_acts)
                    current_section = None  # Track which section: "system", "user", "assistant"

                    for pos, act, tok in all_acts:
                        tok_stripped = tok.strip()
                        # Track section by watching for section name tokens
                        if tok_stripped in ("system", "user", "assistant"):
                            current_section = tok_stripped

                        # end_header_id marks start of content for current section
                        if "end_header_id" in tok:
                            if current_section == "user":
                                user_content_start = pos + 1

                        # eot_id marks end of content for current section
                        if "eot_id" in tok:
                            if current_section == "user" and user_content_start > 0:
                                user_content_end = pos
                                break

                    # Find max and min activation WITHIN user content
                    content_acts = [(pos, act, tok) for pos, act, tok in all_acts
                                   if user_content_start <= pos < user_content_end]
                    if content_acts:
                        content_max_pos, content_max_act, content_max_token = max(content_acts, key=lambda x: x[1])
                        content_min_pos, content_min_act, content_min_token = min(content_acts, key=lambda x: x[1])
                    else:
                        content_max_pos, content_max_act, content_max_token = max_pos, max_act, max_token
                        content_min_pos, content_min_act, content_min_token = max_pos, max_act, max_token

                    # Store top 10 token activations for multi-token highlighting
                    top_tokens = sorted(content_acts, key=lambda x: x[1], reverse=True)[:10]
                    token_acts_list = [{"position": p, "activation": round(a, 4), "token": t}
                                       for p, a, t in top_tokens]

                    # Use content max for statistics (template tokens can have spurious high activation)
                    all_activations.append((content_max_act, prompt, category, content_max_pos, content_max_token, token_acts_list, content_min_act, content_min_pos, content_min_token))
                    category_activations[category].append((content_max_act, prompt, content_max_pos, content_max_token, token_acts_list, content_min_act, content_min_pos, content_min_token))

            except Exception as e:
                for prompt in batch_prompts:
                    errors.append({"prompt": prompt[:80], "category": category, "error": str(e)})

    if not all_activations:
        return {"error": "No successful activation tests", "errors": errors}

    # Compute global statistics (positive = max activations)
    activations_only = [a[0] for a in all_activations]
    global_mean = float(np.mean(activations_only))
    global_std = float(np.std(activations_only))
    if global_std < 0.001:
        global_std = 1.0  # Prevent division by zero

    # Compute global negative statistics (min activations per prompt)
    all_min_acts = [a[6] for a in all_activations]  # index 6 = content_min_act
    neg_global_mean = float(np.mean(all_min_acts))
    neg_global_std = float(np.std(all_min_acts))
    if neg_global_std < 0.001:
        neg_global_std = 1.0

    # Compute z-scores and category statistics
    categories = {}
    for category, acts in category_activations.items():
        if not acts:
            continue

        cat_activations = [a[0] for a in acts]
        cat_mean = float(np.mean(cat_activations))
        cat_std = float(np.std(cat_activations))

        # Negative (min) activations per category
        cat_min_acts = [a[5] for a in acts]  # index 5 = content_min_act
        cat_neg_mean = float(np.mean(cat_min_acts))
        cat_neg_std = float(np.std(cat_min_acts))

        # Z-scores relative to global distribution (positive)
        z_scores = [(a - global_mean) / global_std for a in cat_activations]
        z_mean = float(np.mean(z_scores))

        # Negative z-scores (how negative relative to global negative distribution)
        neg_z_scores = [(a - neg_global_mean) / neg_global_std for a in cat_min_acts]
        neg_z_mean = float(np.mean(neg_z_scores))

        # Build prompt details with z-scores
        prompts_with_z = []
        for entry, z, neg_z in zip(acts, z_scores, neg_z_scores):
            act, prompt, pos, token = entry[0], entry[1], entry[2], entry[3]
            token_acts = entry[4] if len(entry) > 4 else []
            min_act = entry[5] if len(entry) > 5 else 0.0
            min_pos = entry[6] if len(entry) > 6 else 0
            min_token = entry[7] if len(entry) > 7 else ""
            prompts_with_z.append({
                "prompt": prompt[:200],
                "activation": round(act, 4),
                "z_score": round(z, 3),
                "position": pos,
                "token": token,
                "token_activations": token_acts,
                "min_activation": round(min_act, 4),
                "min_position": min_pos,
                "min_token": min_token,
                "neg_z_score": round(neg_z, 3),
            })

        categories[category] = {
            "type": category_types.get(category, "unknown"),
            "prompts": prompts_with_z,
            "count": len(acts),
            "mean": round(cat_mean, 4),
            "std": round(cat_std, 4),
            "z_mean": round(z_mean, 3),
            "z_std": round(float(np.std(z_scores)), 3),
            "min_activation": round(min(cat_activations), 4),
            "max_activation": round(max(cat_activations), 4),
            # Negative firing stats
            "neg_mean": round(cat_neg_mean, 4),
            "neg_std": round(cat_neg_std, 4),
            "neg_z_mean": round(neg_z_mean, 3),
            "neg_z_std": round(float(np.std(neg_z_scores)), 3),
            "neg_min_activation": round(min(cat_min_acts), 4),
            "neg_max_activation": round(max(cat_min_acts), 4),
        }

    # Find top activating prompts across all categories (positive)
    all_with_z = []
    for category, data in categories.items():
        for p in data["prompts"]:
            all_with_z.append({
                "prompt": p["prompt"],
                "activation": p["activation"],
                "z_score": p["z_score"],
                "category": category,
                "category_type": data["type"],
                # Include token info for highlighting in dashboard
                "token": p.get("token", ""),
                "position": p.get("position"),
            })
    top_activating = sorted(all_with_z, key=lambda x: -x["activation"])[:20]

    # Find top NEGATIVELY activating prompts (most negative min activation)
    all_neg = []
    for category, data in categories.items():
        for p in data["prompts"]:
            all_neg.append({
                "prompt": p["prompt"],
                "activation": p["min_activation"],
                "neg_z_score": p["neg_z_score"],
                "category": category,
                "category_type": data["type"],
                "token": p.get("min_token", ""),
                "position": p.get("min_position"),
            })
    top_negatively_activating = sorted(all_neg, key=lambda x: x["activation"])[:20]

    # Generate selectivity summary
    target_cats = [c for c, d in categories.items() if d["type"] == "target"]
    control_cats = [c for c, d in categories.items() if d["type"] in ["control", "unrelated"]]

    if target_cats and control_cats:
        target_z_mean = np.mean([categories[c]["z_mean"] for c in target_cats])
        control_z_mean = np.mean([categories[c]["z_mean"] for c in control_cats])
        selectivity = target_z_mean - control_z_mean

        if selectivity > 2.0:
            selectivity_summary = f"HIGHLY SELECTIVE: Target categories z-mean ({target_z_mean:.2f}) >> control categories ({control_z_mean:.2f}). Δz = {selectivity:.2f}"
        elif selectivity > 1.0:
            selectivity_summary = f"MODERATELY SELECTIVE: Target categories z-mean ({target_z_mean:.2f}) > control categories ({control_z_mean:.2f}). Δz = {selectivity:.2f}"
        elif selectivity > 0.5:
            selectivity_summary = f"WEAKLY SELECTIVE: Slight preference for target categories. Δz = {selectivity:.2f}"
        else:
            selectivity_summary = f"NOT SELECTIVE: No clear preference for target categories. Δz = {selectivity:.2f}"
    else:
        selectivity_summary = "Selectivity not computed (missing target or control categories)"

    # Polarity summary — characterize negative firing behavior
    n_strong_neg = sum(1 for a in all_min_acts if a < -0.5)
    top_neg_cats = sorted(
        [(c, d["neg_mean"]) for c, d in categories.items()],
        key=lambda x: x[1]
    )[:3]
    top_neg_cat_str = ", ".join(f"{c} ({v:.2f})" for c, v in top_neg_cats)
    polarity_summary = (
        f"Negative firing: min={min(all_min_acts):.2f}, mean={neg_global_mean:.2f}, "
        f"{n_strong_neg} prompts below -0.5. "
        f"Top negative categories: {top_neg_cat_str}"
    )

    return {
        "global_mean": round(global_mean, 4),
        "global_std": round(global_std, 4),
        "total_prompts": len(all_activations),
        "categories": categories,
        "top_activating": top_activating,
        "top_negatively_activating": top_negatively_activating,
        "selectivity_summary": selectivity_summary,
        "polarity_summary": polarity_summary,
        "neg_global_mean": round(neg_global_mean, 4),
        "neg_global_std": round(neg_global_std, 4),
        "errors": errors if errors else None,
    }


async def tool_run_category_selectivity_test(
    layer: int,
    neuron_idx: int,
    target_domain: str,
    target_categories: list[str],
    inhibitory_categories: list[str] | None = None,
    include_corpus: bool = True,
    n_generated_per_category: int = 30,
    corpus_categories: list[str] | None = None,
) -> dict[str, Any]:
    """Run comprehensive category selectivity test.

    Tests neuron selectivity by measuring activations across:
    1. Pre-built corpus categories (tech, sports, cooking, etc.)
    2. AI-generated domain-specific prompts (target, control, inhibitory)

    The test computes z-scores for each category, enabling visualization
    of selectivity patterns via stacked area charts.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index within layer
        target_domain: Domain name (e.g., "pharmacology", "malware")
        target_categories: Categories that SHOULD activate the neuron
            (e.g., ["mechanism_of_action", "receptor_binding"])
        inhibitory_categories: Categories that should SUPPRESS the neuron
            (e.g., ["neurotransmitter_release"])
        include_corpus: Whether to include pre-built corpus categories
        n_generated_per_category: Number of prompts to generate per category (default 30)
        corpus_categories: Which corpus categories to include. Default: common unrelated domains

    Returns:
        Dict with category-wise statistics including:
        - global_mean/std: Baseline statistics across all prompts
        - categories: Per-category stats with z-scores for each prompt
        - top_activating: Top 20 activating prompts across all categories
        - selectivity_summary: Human-readable selectivity assessment
    """
    # Collect prompts by category
    categorized_prompts = {}  # category -> [prompts]
    category_types = {}  # category -> "target" | "control" | "inhibitory" | "unrelated"

    # 1. Load corpus categories (unrelated domains)
    if include_corpus:
        try:
            from neuron_scientist.prompt_generator import load_corpus

            corpus = load_corpus()

            # Default corpus categories (common unrelated domains)
            if corpus_categories is None:
                corpus_categories = [
                    "technology", "sports", "cooking", "travel",
                    "history", "finance", "literature", "music", "geography"
                ]

            for cat_name in corpus_categories:
                if cat_name in corpus:
                    categorized_prompts[cat_name] = corpus[cat_name]
                    category_types[cat_name] = "unrelated"

            print(f"  [CategorySelectivity] Loaded {sum(len(p) for p in categorized_prompts.values())} prompts from corpus")

        except Exception as e:
            print(f"  [CategorySelectivity] Warning: Failed to load corpus: {e}")

    # 2. Generate domain-specific prompts
    if n_generated_per_category > 0:
        try:
            from neuron_scientist.prompt_generator import generate_prompts_for_neuron

            # Build hypothesis from domain and categories
            neuron_hypothesis = f"{target_domain} - detecting {', '.join(target_categories)}"

            # Gather evidence from activation cache and protocol state for richer prompt generation
            state = get_protocol_state()

            # Activating examples from the activation cache (populated during Phase 0/1)
            activating_examples = []
            for prompt_text, cached in sorted(
                _ACTIVATION_CACHE.items(),
                key=lambda x: x[1].get("activation", 0),
                reverse=True,
            ):
                if cached.get("is_activating") and cached.get("activation", 0) > 0:
                    activating_examples.append({
                        "prompt": prompt_text[:120],
                        "token": cached.get("max_token", "?"),
                        "activation": cached.get("activation", 0),
                    })
                    if len(activating_examples) >= 15:
                        break

            # Upstream wiring labels
            upstream_labels = []
            wiring = state.wiring_data if state else None
            if wiring:
                for n in wiring.get("top_excitatory", [])[:5]:
                    label = n.get("label", "")
                    if label:
                        upstream_labels.append(f"{n.get('neuron_id', '?')} (excitatory): {label[:80]}")
                for n in wiring.get("top_inhibitory", [])[:5]:
                    label = n.get("label", "")
                    if label:
                        upstream_labels.append(f"{n.get('neuron_id', '?')} (inhibitory): {label[:80]}")

            # Existing label (from initial autointerp)
            existing_label = getattr(state, 'initial_label', '') if state else ''

            # Input hypotheses from hypothesis tracking
            input_hypotheses = []
            hypotheses = getattr(state, 'hypotheses', []) if state else []
            for h in hypotheses:
                if isinstance(h, dict) and h.get('hypothesis_type') in ('activation', None):
                    input_hypotheses.append(f"[{h.get('status', '?')}] {h.get('hypothesis', '')[:100]}")

            if activating_examples:
                print(f"  [CategorySelectivity] Passing {len(activating_examples)} activating examples to prompt generator")
            if upstream_labels:
                print(f"  [CategorySelectivity] Passing {len(upstream_labels)} upstream labels to prompt generator")

            print(f"  [CategorySelectivity] Generating domain-specific prompts for: {neuron_hypothesis}")

            generated = await generate_prompts_for_neuron(
                neuron_hypothesis=neuron_hypothesis,
                target_categories=target_categories,
                inhibitory_categories=inhibitory_categories,
                n_per_category=n_generated_per_category,
                activating_examples=activating_examples,
                upstream_labels=upstream_labels,
                existing_label=existing_label,
                input_hypotheses=input_hypotheses,
            )

            # Add generated prompts with type labels
            for cat_name, prompts in generated.items():
                if not prompts:
                    continue

                # Determine category type from prefix
                if cat_name.startswith("target_"):
                    display_name = cat_name.replace("target_", "")
                    cat_type = "target"
                elif cat_name.startswith("inhibitory_"):
                    display_name = cat_name.replace("inhibitory_", "")
                    cat_type = "inhibitory"
                elif cat_name.startswith("control_"):
                    display_name = cat_name.replace("control_", "")
                    cat_type = "control"
                else:
                    display_name = cat_name
                    cat_type = "generated"

                # Use prefixed name to avoid collisions with corpus
                full_name = f"gen_{display_name}"
                categorized_prompts[full_name] = prompts
                category_types[full_name] = cat_type

            print(f"  [CategorySelectivity] Generated {sum(len(p) for cat, p in generated.items())} domain-specific prompts")

        except Exception as e:
            print(f"  [CategorySelectivity] Warning: Failed to generate prompts: {e}")
            # Fall back to simple hard-coded prompts for target categories
            for cat in target_categories:
                categorized_prompts[f"gen_{cat}"] = [
                    f"Example prompt for {cat} category",
                ]
                category_types[f"gen_{cat}"] = "target"

    if not categorized_prompts:
        return {"error": "No prompts available for testing"}

    total_prompts = sum(len(p) for p in categorized_prompts.values())
    print(f"  [CategorySelectivity] Testing {total_prompts} prompts across {len(categorized_prompts)} categories")

    # 3. Run batched activation test on all prompts
    loop = asyncio.get_running_loop()

    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        result = await client.category_selectivity_test(
            layer=layer, neuron_idx=neuron_idx,
            categorized_prompts=categorized_prompts,
            category_types=category_types,
        )
    else:
        result = await loop.run_in_executor(
            _CUDA_EXECUTOR,
            _sync_category_selectivity_test,
            layer,
            neuron_idx,
            categorized_prompts,
            category_types,
        )

    # 4. Update protocol state
    if result and "error" not in result:
        # Extract z-score gap from result
        categories = result.get("categories", {})
        target_z_means = [d["z_mean"] for c, d in categories.items() if d.get("type") == "target"]
        control_z_means = [d["z_mean"] for c, d in categories.items() if d.get("type") in ["control", "unrelated"]]

        zscore_gap = None
        if target_z_means and control_z_means:
            import numpy as np
            zscore_gap = float(np.mean(target_z_means) - np.mean(control_z_means))

        # Store categorized prompts with activations for batch ablation/steering
        # Union-merge: don't overwrite prompts from prior runs
        state = get_protocol_state()
        for cat_name, cat_data in categories.items():
            existing = state.categorized_prompts.get(cat_name, [])
            existing_texts = {p.get("prompt", "") for p in existing}
            new_prompts = [p for p in cat_data.get("prompts", [])
                           if p.get("prompt", "") not in existing_texts]
            state.categorized_prompts[cat_name] = existing + new_prompts

        update_protocol_state(
            category_selectivity_done=True,
            category_selectivity_zscore_gap=zscore_gap,
            category_selectivity_n_categories=len(categories),
        )

        # Count activating prompts
        activating_count = sum(
            1 for cat_data in categories.values()
            for p in cat_data.get("prompts", [])
            if p.get("activation", 0) > 0.5
        )
        total_count = sum(len(cat_data.get("prompts", [])) for cat_data in categories.values())

        z_gap_str = f"{zscore_gap:.2f}" if zscore_gap is not None else "N/A"
        print(f"[PROTOCOL] Category selectivity test complete: {len(categories)} categories, z-gap={z_gap_str}")
        print(f"[PROTOCOL] Stored {total_count} categorized prompts ({activating_count} activating) for batch ablation/steering")

        # Add z_score_gap to result for storage in investigation JSON
        result["z_score_gap"] = zscore_gap
        result["n_categories"] = len(categories)

        # 5. Run SwiGLU operating regime detection on top activating prompts
        # Skip when using GPU server — get_gate_up_decomposition needs local model access
        if client is None:
            try:
                top_activating = result.get("top_activating", [])
                regime_prompts = [p["prompt"] for p in top_activating[:50] if p.get("prompt")]
                if len(regime_prompts) < 10:
                    # Supplement with all activating prompts from target categories
                    for cat_name, cat_data in categories.items():
                        if cat_data.get("type") == "target":
                            for p in cat_data.get("prompts", []):
                                if p.get("prompt") and p["prompt"] not in regime_prompts:
                                    regime_prompts.append(p["prompt"])
                                if len(regime_prompts) >= 50:
                                    break

                if regime_prompts:
                    regime_texts = [format_prompt(p) for p in regime_prompts]
                    regime_analysis = await loop.run_in_executor(
                        _CUDA_EXECUTOR,
                        get_gate_up_decomposition,
                        layer,
                        neuron_idx,
                        regime_texts,
                    )

                    if regime_analysis and "error" not in regime_analysis:
                        result["regime_analysis"] = regime_analysis

                        # Update protocol state with regime info
                        regime = regime_analysis.get("regime", "unknown")
                        regime_conf = regime_analysis.get("regime_confidence", 0.0)

                        firing_sign_stats = {
                            "positive_pct": regime_analysis.get("positive_firing_pct", 0),
                            "negative_pct": regime_analysis.get("negative_firing_pct", 0),
                            "mean_gate_pre": regime_analysis.get("mean_gate_pre", 0),
                            "mean_up_pre": regime_analysis.get("mean_up_pre", 0),
                        }

                        update_protocol_state(
                            operating_regime=regime,
                            regime_confidence=regime_conf,
                            regime_data=regime_analysis,
                            firing_sign_stats=firing_sign_stats,
                        )

                        print(f"[PROTOCOL] SwiGLU regime detected: {regime} (confidence: {regime_conf:.1%})")
                        print(f"[PROTOCOL]   Quadrants: {regime_analysis.get('quadrant_counts', {})}")
                        print(f"[PROTOCOL]   Firing: {firing_sign_stats['positive_pct']:.0f}% positive, {firing_sign_stats['negative_pct']:.0f}% negative")

                        # Apply retroactive polarity correction if inverted regime detected
                        _apply_regime_correction(state)

            except Exception as e:
                print(f"[PROTOCOL] Warning: Regime detection failed: {e}")

    return result


async def tool_test_additional_prompts(
    layer: int,
    neuron_idx: int,
    prompts: list[str],
    category: str = "follow_up",
    category_type: str = "target",
) -> dict[str, Any]:
    """Test additional prompts and merge into existing category selectivity data.

    Use this to:
    - Test follow-up prompts after seeing initial selectivity results
    - Add minimal pairs or targeted probes
    - Test skeptic-designed adversarial prompts

    Results are automatically merged with existing selectivity data,
    recomputing z-scores against the full prompt set.

    Args:
        layer: Neuron layer
        neuron_idx: Neuron index
        prompts: List of prompt strings to test
        category: Category name for these prompts (e.g., "minimal_pairs_no_ratio")
        category_type: One of "target", "control", "inhibitory", "unrelated"

    Returns:
        Dict with per-prompt activations and updated category stats
    """
    if not prompts:
        return {"error": "No prompts provided"}

    record_experiment("test_additional_prompts", {
        "layer": layer, "neuron_idx": neuron_idx,
        "n_prompts": len(prompts), "category": category,
    })

    # Build categorized_prompts and category_types for the GPU function
    full_category_name = f"gen_{category}"
    categorized = {full_category_name: prompts}
    cat_types = {full_category_name: category_type}

    # Run on GPU
    client = get_gpu_client()
    if client is not None:
        result = await client.category_selectivity_test(
            layer=layer, neuron_idx=neuron_idx,
            categorized_prompts=categorized,
            category_types=cat_types,
        )
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _CUDA_EXECUTOR,
            _sync_category_selectivity_test,
            layer, neuron_idx, categorized, cat_types,
        )

    if not result or "error" in result:
        return result

    # Merge into existing selectivity data via protocol state
    state = get_protocol_state()
    if state:
        # Union-merge categorized prompts
        categories = result.get("categories", {})
        for cat_name, cat_data in categories.items():
            existing = state.categorized_prompts.get(cat_name, [])
            existing_texts = {p.get("prompt", "") for p in existing if isinstance(p, dict)}
            new_items = []
            for p in cat_data.get("prompts", []):
                prompt_text = p.get("prompt", "") if isinstance(p, dict) else ""
                if prompt_text and prompt_text not in existing_texts:
                    new_items.append(p)
            state.categorized_prompts[cat_name] = existing + new_items

    # Build summary for agent
    categories = result.get("categories", {})
    per_prompt = []
    for cat_name, cat_data in categories.items():
        for p in cat_data.get("prompts", []):
            per_prompt.append({
                "prompt": p.get("prompt", "")[:100],
                "activation": round(p.get("activation", 0), 3),
                "token": p.get("token", ""),
                "z_score": round(p.get("z_score", 0), 2),
                "category": cat_name,
            })

    # Sort by activation descending
    per_prompt.sort(key=lambda x: x["activation"], reverse=True)

    return {
        "n_tested": len(prompts),
        "category": full_category_name,
        "category_type": category_type,
        "mean_activation": round(sum(p["activation"] for p in per_prompt) / len(per_prompt), 3) if per_prompt else 0,
        "activating_count": sum(1 for p in per_prompt if p["activation"] > 0.5),
        "per_prompt_results": per_prompt,
        "note": "Results merged into existing selectivity data. Re-run category_selectivity_test to see updated z-scores.",
    }


# =============================================================================
# Category Selectivity Merge & Quality Assessment
# =============================================================================


def merge_selectivity_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple category selectivity run dicts into one unified view.

    Deduplicates prompts by text (keeps higher activation if duplicate),
    recomputes global stats and z-scores from the full union.

    Args:
        runs: List of selectivity run dicts from run_category_selectivity_test

    Returns:
        Single merged dict with unified categories, recomputed stats
    """
    if not runs:
        return {}
    if len(runs) == 1:
        return runs[0]

    # Collect all prompts per category, deduplicating by prompt text
    # Key: (category_name) -> {prompt_text -> best_entry}
    merged_categories: dict[str, dict[str, dict]] = {}
    category_types: dict[str, str] = {}

    for run in runs:
        for cat_name, cat_data in run.get("categories", {}).items():
            if not isinstance(cat_data, dict):
                continue
            if cat_name not in merged_categories:
                merged_categories[cat_name] = {}
            # Preserve category type
            if cat_name not in category_types:
                category_types[cat_name] = cat_data.get("type", "unknown")

            for p in cat_data.get("prompts", []):
                prompt_text = p.get("prompt", "")
                if not prompt_text:
                    continue
                existing = merged_categories[cat_name].get(prompt_text)
                if existing is None or p.get("activation", 0) > existing.get("activation", 0):
                    merged_categories[cat_name][prompt_text] = p

    if not merged_categories:
        return runs[-1]  # fallback to last run

    # Flatten to lists and recompute global stats
    all_activations = []
    category_prompts: dict[str, list[dict]] = {}
    for cat_name, prompt_map in merged_categories.items():
        prompts_list = list(prompt_map.values())
        category_prompts[cat_name] = prompts_list
        for p in prompts_list:
            all_activations.append(p.get("activation", 0))

    if not all_activations:
        return runs[-1]

    global_mean = float(np.mean(all_activations))
    global_std = float(np.std(all_activations))
    if global_std < 0.001:
        global_std = 1.0

    # Recompute per-category stats with new global baseline
    categories = {}
    all_with_z = []
    for cat_name, prompts_list in category_prompts.items():
        cat_activations = [p.get("activation", 0) for p in prompts_list]
        cat_mean = float(np.mean(cat_activations))
        cat_std = float(np.std(cat_activations))

        # Recompute z-scores
        prompts_with_z = []
        z_scores = []
        for p in prompts_list:
            act = p.get("activation", 0)
            z = (act - global_mean) / global_std
            z_scores.append(z)
            updated = dict(p)
            updated["z_score"] = round(z, 3)
            prompts_with_z.append(updated)
            all_with_z.append({
                "prompt": p.get("prompt", ""),
                "activation": p.get("activation", 0),
                "z_score": round(z, 3),
                "category": cat_name,
                "category_type": category_types.get(cat_name, "unknown"),
                "token": p.get("token", ""),
                "position": p.get("position"),
            })

        categories[cat_name] = {
            "type": category_types.get(cat_name, "unknown"),
            "prompts": prompts_with_z,
            "count": len(prompts_list),
            "mean": round(cat_mean, 4),
            "std": round(cat_std, 4),
            "z_mean": round(float(np.mean(z_scores)), 3),
            "z_std": round(float(np.std(z_scores)), 3),
            "min_activation": round(min(cat_activations), 4),
            "max_activation": round(max(cat_activations), 4),
        }

    top_activating = sorted(all_with_z, key=lambda x: -x["activation"])[:20]

    # Recompute selectivity summary
    target_cats = [c for c, d in categories.items() if d["type"] == "target"]
    control_cats = [c for c, d in categories.items() if d["type"] in ["control", "unrelated"]]

    if target_cats and control_cats:
        target_z_mean = float(np.mean([categories[c]["z_mean"] for c in target_cats]))
        control_z_mean = float(np.mean([categories[c]["z_mean"] for c in control_cats]))
        selectivity = target_z_mean - control_z_mean
        z_score_gap = selectivity

        if selectivity > 2.0:
            selectivity_summary = f"HIGHLY SELECTIVE: Target categories z-mean ({target_z_mean:.2f}) >> control categories ({control_z_mean:.2f}). Δz = {selectivity:.2f}"
        elif selectivity > 1.0:
            selectivity_summary = f"MODERATELY SELECTIVE: Target categories z-mean ({target_z_mean:.2f}) > control categories ({control_z_mean:.2f}). Δz = {selectivity:.2f}"
        elif selectivity > 0.5:
            selectivity_summary = f"WEAKLY SELECTIVE: Slight preference for target categories. Δz = {selectivity:.2f}"
        else:
            selectivity_summary = f"NOT SELECTIVE: No clear preference for target categories. Δz = {selectivity:.2f}"
    else:
        selectivity_summary = "Selectivity not computed (missing target or control categories)"
        z_score_gap = None

    return {
        "global_mean": round(global_mean, 4),
        "global_std": round(global_std, 4),
        "total_prompts": len(all_activations),
        "categories": categories,
        "top_activating": top_activating,
        "selectivity_summary": selectivity_summary,
        "z_score_gap": z_score_gap,
        "n_categories": len(categories),
        "n_runs_merged": len(runs),
    }


def assess_selectivity_quality(merged: dict[str, Any]) -> dict[str, Any]:
    """Assess quality of category selectivity data.

    Checks data sufficiency and warns about sparse or uninformative runs.

    Args:
        merged: Merged selectivity dict (from merge_selectivity_runs or single run)

    Returns:
        Dict with quality_score, warnings, is_informative, target_coverage
    """
    if not merged or "categories" not in merged:
        return {
            "quality_score": 0.0,
            "warnings": ["No selectivity data available"],
            "is_informative": False,
            "target_coverage": {},
        }

    warnings = []
    categories = merged.get("categories", {})
    total_prompts = merged.get("total_prompts", 0)
    global_std = merged.get("global_std", 1.0)

    # Check total prompt count
    if total_prompts < 50:
        warnings.append(f"Only {total_prompts} total prompts (recommend >= 50 for reliable statistics)")

    # Check control category coverage
    control_cats = [c for c, d in categories.items()
                    if isinstance(d, dict) and d.get("type") in ["control", "unrelated"]]
    if len(control_cats) < 3:
        warnings.append(f"Only {len(control_cats)} control/unrelated categories (recommend >= 3)")

    # Check target category data quality
    target_coverage = {}
    target_cats = [c for c, d in categories.items()
                   if isinstance(d, dict) and d.get("type") == "target"]

    for cat_name in target_cats:
        cat_data = categories[cat_name]
        prompts = cat_data.get("prompts", [])
        high_z_prompts = [p for p in prompts if p.get("z_score", 0) > 1.0]
        target_coverage[cat_name] = {
            "total_prompts": len(prompts),
            "high_z_count": len(high_z_prompts),
            "z_mean": cat_data.get("z_mean", 0),
        }
        if len(high_z_prompts) < 5:
            warnings.append(
                f"Target category '{cat_name}' has only {len(high_z_prompts)} prompts with z > 1.0 "
                f"(recommend >= 5 for reliable conclusions)"
            )

    # Check z-score gap
    z_gap = merged.get("z_score_gap")
    if z_gap is not None and z_gap < 0.5:
        warnings.append(f"z-score gap between target and control is only {z_gap:.2f} (recommend >= 0.5)")

    # Compute quality score (0-1)
    score = 1.0
    if total_prompts < 50:
        score -= 0.2
    if len(control_cats) < 3:
        score -= 0.15
    if z_gap is not None and z_gap < 0.5:
        score -= 0.2
    for cat_name in target_cats:
        tc = target_coverage.get(cat_name, {})
        if tc.get("high_z_count", 0) < 5:
            score -= 0.15
    score = max(0.0, score)

    return {
        "quality_score": round(score, 2),
        "warnings": warnings,
        "is_informative": len(warnings) == 0,
        "target_coverage": target_coverage,
    }


# =============================================================================
# V4 Multi-Token Generation Tools
# =============================================================================


def _sync_ablate_and_generate(
    layer: int,
    neuron_idx: int,
    prompt: str,
    max_new_tokens: int = 10,
    downstream_neurons: list[str] | None = None,
    top_k_logits: int = 10,
    ablation_method: str = "mean",
) -> dict[str, Any]:
    """Synchronous implementation of ablate_and_generate.

    Generates multiple tokens with neuron ablated vs baseline.
    Uses greedy decoding (do_sample=False) for deterministic results.

    Args:
        ablation_method: "mean" (default) or "zero"
    """
    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        text = format_prompt(prompt)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # Determine ablation value
        if ablation_method == "mean":
            ablation_value = get_mean_activation(layer, neuron_idx)
        else:
            ablation_value = 0.0

        # Create ablation hook (works for both mean and zero ablation)
        def ablation_hook(module, args, kwargs):
            x = args[0]  # Shape: (batch, seq_len, intermediate_dim)
            modified = x.clone()
            modified[:, :, neuron_idx] = ablation_value
            return (modified,) + args[1:], kwargs

        mlp = model.model.layers[layer].mlp

        # 1. Baseline generation (no ablation)
        with torch.no_grad():
            baseline_outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline_completion = tokenizer.decode(
            baseline_outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # 2. Ablated generation using hook
        handle = mlp.down_proj.register_forward_pre_hook(ablation_hook, with_kwargs=True)

        try:
            with torch.no_grad():
                ablated_outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            ablated_completion = tokenizer.decode(
                ablated_outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
        finally:
            handle.remove()

        # 3. Analyze per-position logit shifts (first position of generation)
        # Re-run single forward pass to get logits at generation start
        per_position_shifts = []
        with torch.no_grad():
            # Baseline logits
            baseline_out = model(**inputs)
            baseline_logits = baseline_out.logits[0, -1].float()

            # Ablated logits (using hook)
            handle = mlp.down_proj.register_forward_pre_hook(ablation_hook, with_kwargs=True)
            ablated_out = model(**inputs)
            ablated_logits = ablated_out.logits[0, -1].float()
            handle.remove()

        top_values, top_indices = torch.topk(baseline_logits, top_k_logits)
        baseline_top = {
            tokenizer.decode([idx.item()]): baseline_logits[idx.item()].item()
            for idx in top_indices
        }
        ablated_top = {
            tokenizer.decode([idx.item()]): ablated_logits[idx.item()].item()
            for idx in top_indices
        }
        logit_shifts = {
            tok: ablated_top.get(tok, 0) - baseline_top.get(tok, 0)
            for tok in baseline_top
        }
        per_position_shifts.append({
            "position": 0,
            "baseline_top": baseline_top,
            "ablated_top": ablated_top,
            "shifts": logit_shifts,
        })

        # 4. Check downstream neuron effects (if specified)
        downstream_effects = {}
        if downstream_neurons:
            for ds_neuron_id in downstream_neurons[:5]:  # Limit to 5
                try:
                    parts = ds_neuron_id.replace("L", "").replace("N", "").split("/")
                    ds_layer = int(parts[0])
                    ds_idx = int(parts[1])

                    # Get downstream activation with and without ablation
                    # Using a hook to capture activations
                    ds_acts = {"baseline": None, "ablated": None}

                    def capture_ds_hook(module, input, output, key="baseline"):
                        hidden = input[0]
                        gate = module.gate_proj(hidden)
                        up = module.up_proj(hidden)
                        intermediate = torch.nn.functional.silu(gate) * up
                        ds_acts[key] = intermediate[0, -1, ds_idx].item()

                    ds_mlp = model.model.layers[ds_layer].mlp

                    # Baseline
                    handle = ds_mlp.register_forward_hook(
                        lambda m, i, o, k="baseline": capture_ds_hook(m, i, o, k)
                    )
                    with torch.no_grad():
                        model(**inputs)
                    handle.remove()

                    # Ablated (using hook instead of weight modification)
                    ablation_handle = mlp.down_proj.register_forward_pre_hook(ablation_hook, with_kwargs=True)
                    handle = ds_mlp.register_forward_hook(
                        lambda m, i, o, k="ablated": capture_ds_hook(m, i, o, k)
                    )
                    with torch.no_grad():
                        model(**inputs)
                    handle.remove()
                    ablation_handle.remove()

                    baseline_act = ds_acts["baseline"]
                    ablated_act = ds_acts["ablated"]
                    # Only compute percentage if baseline has significant activation (>= 0.1)
                    # to avoid division-by-small-number artifacts (e.g., -700% from 0.01 baseline)
                    if baseline_act is not None and abs(baseline_act) >= 0.1:
                        change_pct = 100 * (ablated_act - baseline_act) / abs(baseline_act)
                    elif baseline_act is not None:
                        # Low baseline - report absolute change instead
                        change_pct = ablated_act - baseline_act  # Will be small since baseline is small
                    else:
                        change_pct = None

                    downstream_effects[ds_neuron_id] = {
                        "baseline_activation": baseline_act,
                        "ablated_activation": ablated_act,
                        "change_percent": change_pct,
                        "absolute_change": (ablated_act - baseline_act) if baseline_act is not None else None,
                    }
                except Exception as e:
                    downstream_effects[ds_neuron_id] = {"error": str(e)}

        return {
            "prompt": prompt[:100],
            "max_new_tokens": max_new_tokens,
            "baseline_completion": baseline_completion,
            "ablated_completion": ablated_completion,
            "completion_changed": baseline_completion != ablated_completion,
            "per_position_shifts": per_position_shifts,
            "downstream_effects": downstream_effects,
            "ablation_method": ablation_method,
            "ablation_value": ablation_value,
        }


async def tool_ablate_and_generate(
    layer: int,
    neuron_idx: int,
    prompt: str,
    max_new_tokens: int = 10,
    downstream_neurons: list[str] | None = None,
    top_k_logits: int = 10,
    ablation_method: str = "mean",
) -> dict[str, Any]:
    """DEPRECATED: Use batch_ablate_and_generate instead.

    This function only checks downstream at the last input position.
    batch_ablate_and_generate checks downstream at ALL generated positions
    and supports both single and multiple prompts.

    Example migration:
        # Old:
        result = await ablate_and_generate(layer, neuron, prompt, downstream_neurons=ds)

        # New:
        result = await batch_ablate_and_generate(layer, neuron, prompts=[prompt], downstream_neurons=ds)

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index within layer
        prompt: The prompt to test
        max_new_tokens: Number of tokens to generate (default 10)
        downstream_neurons: Optional list of downstream neuron IDs to check
        ablation_method: "mean" (default) or "zero"
        top_k_logits: Number of top logits to return per position

    Returns:
        Dict with baseline_completion, ablated_completion, per_position_shifts,
        and downstream_effects (if downstream_neurons specified).
    """
    import warnings
    warnings.warn(
        "ablate_and_generate is deprecated. Use batch_ablate_and_generate instead, "
        "which checks downstream neurons at ALL generated positions.",
        DeprecationWarning,
        stacklevel=2
    )
    print("[DEPRECATED] ablate_and_generate: Use batch_ablate_and_generate for multi-position downstream checking")

    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        result = await client.ablate_and_generate(
            layer=layer, neuron_idx=neuron_idx, prompt=prompt,
            max_new_tokens=max_new_tokens, downstream_neurons=downstream_neurons,
            top_k_logits=top_k_logits, ablation_method=ablation_method,
        )
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _CUDA_EXECUTOR,
            _sync_ablate_and_generate,
            layer,
            neuron_idx,
            prompt,
            max_new_tokens,
            downstream_neurons,
            top_k_logits,
            ablation_method,
        )

    # Update protocol state and store result
    state = get_protocol_state()
    state.multi_token_ablation_results.append({
        "prompt": prompt,
        "layer": layer,
        "neuron_idx": neuron_idx,
        "max_new_tokens": max_new_tokens,
        "ablation_method": ablation_method,
        **result,
    })
    update_protocol_state(multi_token_ablation_done=True)
    print(f"[PROTOCOL] Multi-token ablation complete: changed={result.get('completion_changed')} (method={ablation_method})")

    return result


def _sync_steer_and_generate(
    layer: int,
    neuron_idx: int,
    prompt: str,
    steering_value: float,
    max_new_tokens: int = 10,
    downstream_neurons: list[str] | None = None,
    top_k_logits: int = 10,
) -> dict[str, Any]:
    """Synchronous implementation of steer_and_generate.

    Generates multiple tokens with neuron steered vs baseline.
    Uses greedy decoding (do_sample=False) for deterministic results.
    """
    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        text = format_prompt(prompt)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # 1. Baseline generation (no steering)
        with torch.no_grad():
            baseline_outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline_completion = tokenizer.decode(
            baseline_outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # 2. Steered generation - add steering_value to neuron activation
        def steering_hook(module, args, kwargs):
            x = args[0]  # Shape: (batch, seq_len, intermediate_dim)
            modified = x.clone()
            # Steer at all positions
            if neuron_idx < modified.shape[2]:
                modified[:, :, neuron_idx] += steering_value
            return (modified,) + args[1:], kwargs

        mlp = model.model.layers[layer].mlp
        handle = mlp.down_proj.register_forward_pre_hook(steering_hook, with_kwargs=True)

        try:
            with torch.no_grad():
                steered_outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            steered_completion = tokenizer.decode(
                steered_outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
        finally:
            handle.remove()

        # 3. Analyze per-position logit shifts (first position)
        per_position_shifts = []
        with torch.no_grad():
            # Baseline logits
            baseline_out = model(**inputs)
            baseline_logits = baseline_out.logits[0, -1].float()

            # Steered logits
            handle = mlp.down_proj.register_forward_pre_hook(steering_hook, with_kwargs=True)
            steered_out = model(**inputs)
            steered_logits = steered_out.logits[0, -1].float()
            handle.remove()

        top_values, top_indices = torch.topk(baseline_logits, top_k_logits)
        baseline_top = {
            tokenizer.decode([idx.item()]): baseline_logits[idx.item()].item()
            for idx in top_indices
        }
        steered_top = {
            tokenizer.decode([idx.item()]): steered_logits[idx.item()].item()
            for idx in top_indices
        }
        logit_shifts = {
            tok: steered_top.get(tok, 0) - baseline_top.get(tok, 0)
            for tok in baseline_top
        }
        per_position_shifts.append({
            "position": 0,
            "baseline_top": baseline_top,
            "steered_top": steered_top,
            "shifts": logit_shifts,
        })

        # 4. Check downstream neuron effects (if specified)
        downstream_effects = {}
        if downstream_neurons:
            for ds_neuron_id in downstream_neurons[:5]:
                try:
                    parts = ds_neuron_id.replace("L", "").replace("N", "").split("/")
                    ds_layer = int(parts[0])
                    ds_idx = int(parts[1])

                    ds_acts = {"baseline": None, "steered": None}

                    def capture_ds_hook(module, input, output, key="baseline"):
                        hidden = input[0]
                        gate = module.gate_proj(hidden)
                        up = module.up_proj(hidden)
                        intermediate = torch.nn.functional.silu(gate) * up
                        ds_acts[key] = intermediate[0, -1, ds_idx].item()

                    ds_mlp = model.model.layers[ds_layer].mlp

                    # Baseline
                    handle = ds_mlp.register_forward_hook(
                        lambda m, i, o, k="baseline": capture_ds_hook(m, i, o, k)
                    )
                    with torch.no_grad():
                        model(**inputs)
                    handle.remove()

                    # Steered
                    steer_handle = mlp.down_proj.register_forward_pre_hook(steering_hook, with_kwargs=True)
                    capture_handle = ds_mlp.register_forward_hook(
                        lambda m, i, o, k="steered": capture_ds_hook(m, i, o, k)
                    )
                    with torch.no_grad():
                        model(**inputs)
                    capture_handle.remove()
                    steer_handle.remove()

                    baseline_act = ds_acts["baseline"]
                    steered_act = ds_acts["steered"]
                    # Only compute percentage if baseline has significant activation (>= 0.1)
                    # to avoid division-by-small-number artifacts (e.g., -700% from 0.01 baseline)
                    if baseline_act is not None and abs(baseline_act) >= 0.1:
                        change_pct = 100 * (steered_act - baseline_act) / abs(baseline_act)
                    elif baseline_act is not None:
                        # Low baseline - report absolute change instead
                        change_pct = steered_act - baseline_act
                    else:
                        change_pct = None

                    downstream_effects[ds_neuron_id] = {
                        "baseline_activation": baseline_act,
                        "steered_activation": steered_act,
                        "change_percent": change_pct,
                        "absolute_change": (steered_act - baseline_act) if baseline_act is not None else None,
                    }
                except Exception as e:
                    downstream_effects[ds_neuron_id] = {"error": str(e)}

        return {
            "prompt": prompt[:100],
            "steering_value": steering_value,
            "max_new_tokens": max_new_tokens,
            "baseline_completion": baseline_completion,
            "steered_completion": steered_completion,
            "completion_changed": baseline_completion != steered_completion,
            "per_position_shifts": per_position_shifts,
            "downstream_effects": downstream_effects,
        }


async def tool_steer_and_generate(
    layer: int,
    neuron_idx: int,
    prompt: str,
    steering_value: float,
    max_new_tokens: int = 10,
    downstream_neurons: list[str] | None = None,
    top_k_logits: int = 10,
) -> dict[str, Any]:
    """V4 Output Phase tool: Steer neuron and generate multiple tokens.

    Uses greedy decoding for deterministic results.

    Args:
        layer: Layer number (0-31)
        neuron_idx: Neuron index within layer
        prompt: The prompt to test
        steering_value: Value to add to neuron activation
        max_new_tokens: Number of tokens to generate (default 10)
        downstream_neurons: Optional list of downstream neuron IDs to check
        top_k_logits: Number of top logits to return per position

    Returns:
        Dict with baseline_completion, steered_completion, per_position_shifts,
        and downstream_effects (if downstream_neurons specified).
    """
    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        result = await client.steer_and_generate(
            layer=layer, neuron_idx=neuron_idx, prompt=prompt,
            steering_value=steering_value, max_new_tokens=max_new_tokens,
            downstream_neurons=downstream_neurons, top_k_logits=top_k_logits,
        )
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _CUDA_EXECUTOR,
            _sync_steer_and_generate,
            layer,
            neuron_idx,
            prompt,
            steering_value,
            max_new_tokens,
            downstream_neurons,
            top_k_logits,
        )

    # Store result in protocol state
    state = get_protocol_state()
    state.multi_token_steering_results.append({
        "prompt": prompt,
        "layer": layer,
        "neuron_idx": neuron_idx,
        "steering_value": steering_value,
        "max_new_tokens": max_new_tokens,
        **result,
    })
    print(f"[PROTOCOL] Multi-token steering complete: changed={result.get('completion_changed')}")

    return result


# =============================================================================
# BATCH ABLATION AND STEERING TOOLS
# =============================================================================


def _sync_batch_ablate_and_generate(
    layer: int,
    neuron_idx: int,
    prompt_data: list[dict[str, Any]],
    downstream_neurons: list[str] | None = None,
    max_new_tokens: int = 10,
    truncate_to_activation: bool = False,
    generation_format: str = "continuation",
) -> dict[str, Any]:
    """Synchronous implementation of batch_ablate_and_generate.

    Unified ablation tool that:
    - Runs ablation on multiple prompts
    - Generates completions (baseline vs ablated)
    - Optionally checks downstream neurons at ALL generated positions

    Args:
        layer: Target layer
        neuron_idx: Target neuron index
        prompt_data: List of dicts with keys: prompt, category (optional), position (optional)
        downstream_neurons: Optional list of downstream neuron IDs to check (e.g., ["L26/N1234"])
        max_new_tokens: Number of tokens to generate
        truncate_to_activation: If True and position is provided, truncate prompt to activation position

    Returns:
        Dict with per-prompt results, category stats, and downstream dependency summary
    """
    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        mlp = model.model.layers[layer].mlp
        original_weight = mlp.down_proj.weight[:, neuron_idx].clone()

        # Parse downstream neurons
        parsed_downstream = []
        if downstream_neurons:
            for ds_id in downstream_neurons:
                match = re.match(r"L(\d+)/N(\d+)", ds_id)
                if match:
                    ds_layer = int(match.group(1))
                    ds_idx = int(match.group(2))
                    if ds_layer > layer:  # Must be in later layer
                        parsed_downstream.append({
                            "id": ds_id,
                            "layer": ds_layer,
                            "idx": ds_idx,
                        })

        all_results = []

        # Process each prompt individually (not batched) to properly track downstream at each position
        for item in prompt_data:
            prompt = item.get("prompt", item) if isinstance(item, dict) else item
            category = item.get("category") if isinstance(item, dict) else None
            activation_position = item.get("position") if isinstance(item, dict) else None

            # Format prompt based on generation_format
            text = format_generation_prompt(prompt, generation_format)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            # If truncate_to_activation and we have a position, truncate the input
            actual_input_len = input_len
            if truncate_to_activation and activation_position is not None:
                # activation_position is from the raw prompt tokenization, but
                # the formatted text has extra chat template tokens prepended.
                # Compute the offset so we truncate at the right position.
                if generation_format != "raw":
                    raw_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
                    template_offset = input_len - raw_tokens
                else:
                    template_offset = 0
                # Truncate to position where neuron fires + 1 (include the activating token)
                truncate_pos = min(activation_position + 1 + template_offset, input_len)
                if truncate_pos < input_len:
                    inputs["input_ids"] = inputs["input_ids"][:, :truncate_pos]
                    if "attention_mask" in inputs:
                        inputs["attention_mask"] = inputs["attention_mask"][:, :truncate_pos]
                    actual_input_len = truncate_pos

            # Storage for downstream activations per position
            downstream_per_position = {ds["id"]: {"baseline": [], "ablated": []} for ds in parsed_downstream}

            # --- Baseline generation with downstream capture ---
            if parsed_downstream:
                baseline_hooks = []
                for ds in parsed_downstream:
                    ds_mlp = model.model.layers[ds["layer"]].mlp
                    captured = []

                    def make_hook(cap_list, ds_idx):
                        def hook(module, input, output):
                            hidden = input[0]
                            gate = module.gate_proj(hidden)
                            up = module.up_proj(hidden)
                            intermediate = torch.nn.functional.silu(gate) * up
                            cap_list.append(intermediate[0, -1, ds_idx].item())
                        return hook

                    handle = ds_mlp.register_forward_hook(make_hook(captured, ds["idx"]))
                    baseline_hooks.append((handle, ds["id"], captured))

            with torch.no_grad():
                baseline_outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            if parsed_downstream:
                for handle, ds_id, captured in baseline_hooks:
                    handle.remove()
                    downstream_per_position[ds_id]["baseline"] = captured.copy()

            baseline_text = tokenizer.decode(
                baseline_outputs[0][actual_input_len:], skip_special_tokens=True
            )

            # --- Ablated generation with downstream capture ---
            if parsed_downstream:
                ablated_hooks = []
                for ds in parsed_downstream:
                    ds_mlp = model.model.layers[ds["layer"]].mlp
                    captured = []

                    def make_hook(cap_list, ds_idx):
                        def hook(module, input, output):
                            hidden = input[0]
                            gate = module.gate_proj(hidden)
                            up = module.up_proj(hidden)
                            intermediate = torch.nn.functional.silu(gate) * up
                            cap_list.append(intermediate[0, -1, ds_idx].item())
                        return hook

                    handle = ds_mlp.register_forward_hook(make_hook(captured, ds["idx"]))
                    ablated_hooks.append((handle, ds["id"], captured))

            with torch.no_grad():
                mlp.down_proj.weight[:, neuron_idx] = 0
                ablated_outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                mlp.down_proj.weight[:, neuron_idx] = original_weight

            if parsed_downstream:
                for handle, ds_id, captured in ablated_hooks:
                    handle.remove()
                    downstream_per_position[ds_id]["ablated"] = captured.copy()

            ablated_text = tokenizer.decode(
                ablated_outputs[0][actual_input_len:], skip_special_tokens=True
            )

            # Get logit shifts for first position
            with torch.no_grad():
                baseline_out = model(**inputs)
                baseline_logits = baseline_out.logits[0, -1].float()

                mlp.down_proj.weight[:, neuron_idx] = 0
                ablated_out = model(**inputs)
                ablated_logits = ablated_out.logits[0, -1].float()
                mlp.down_proj.weight[:, neuron_idx] = original_weight

            top_k = 10
            top_values, top_indices = torch.topk(baseline_logits, top_k)
            logit_shifts = {}
            for idx in top_indices:
                token = tokenizer.decode([idx.item()])
                shift = (ablated_logits[idx.item()] - baseline_logits[idx.item()]).item()
                logit_shifts[token] = shift

            max_shift = max(abs(v) for v in logit_shifts.values()) if logit_shifts else 0

            # Compute downstream effects per neuron
            downstream_effects = {}
            per_position_effects = {}
            for ds in parsed_downstream:
                ds_id = ds["id"]
                baseline_acts = downstream_per_position[ds_id]["baseline"]
                ablated_acts = downstream_per_position[ds_id]["ablated"]

                n_positions = min(len(baseline_acts), len(ablated_acts), max_new_tokens)
                per_pos_changes = []

                # Find position with max baseline activation
                max_baseline_pos = -1
                max_baseline_val = 0
                for pos in range(n_positions):
                    baseline_val = baseline_acts[pos] if pos < len(baseline_acts) else 0
                    if abs(baseline_val) > abs(max_baseline_val):
                        max_baseline_val = baseline_val
                        max_baseline_pos = pos

                for pos in range(n_positions):
                    baseline_val = baseline_acts[pos] if pos < len(baseline_acts) else 0
                    ablated_val = ablated_acts[pos] if pos < len(ablated_acts) else 0

                    if abs(baseline_val) >= 0.1:
                        change_pct = 100 * (ablated_val - baseline_val) / abs(baseline_val)
                    else:
                        change_pct = 0

                    per_pos_changes.append({
                        "position": pos,
                        "baseline": baseline_val,
                        "ablated": ablated_val,
                        "change_percent": change_pct,
                        "absolute_change": ablated_val - baseline_val,
                    })

                # Aggregate: use max activation position
                if per_pos_changes and max_baseline_pos >= 0 and abs(max_baseline_val) >= 0.1:
                    mean_change = per_pos_changes[max_baseline_pos]["change_percent"]
                elif per_pos_changes:
                    significant_changes = [p["change_percent"] for p in per_pos_changes if abs(p["baseline"]) >= 0.1]
                    mean_change = sum(significant_changes) / len(significant_changes) if significant_changes else 0
                else:
                    mean_change = 0

                downstream_effects[ds_id] = {
                    "mean_change_percent": mean_change,
                    "n_positions_measured": n_positions,
                }
                per_position_effects[ds_id] = per_pos_changes

            result_item = {
                "prompt": prompt[:100] if isinstance(prompt, str) else str(prompt)[:100],
                "category": category,
                "baseline_completion": baseline_text,
                "ablated_completion": ablated_text,
                "completion_changed": baseline_text != ablated_text,
                "logit_shifts": logit_shifts,
                "max_shift": max_shift,
            }

            if parsed_downstream:
                result_item["downstream_effects"] = downstream_effects
                result_item["per_position_effects"] = per_position_effects

            all_results.append(result_item)

        # Compute category-level statistics
        categories = [item.get("category") if isinstance(item, dict) else None for item in prompt_data]
        category_stats = {}
        unique_categories = set(c for c in categories if c is not None)
        for cat in unique_categories:
            cat_results = [r for r in all_results if r["category"] == cat]
            if cat_results:
                changed_count = sum(1 for r in cat_results if r["completion_changed"])
                avg_max_shift = sum(r["max_shift"] for r in cat_results) / len(cat_results)
                category_stats[cat] = {
                    "total": len(cat_results),
                    "changed": changed_count,
                    "change_rate": changed_count / len(cat_results),
                    "avg_max_shift": avg_max_shift,
                }

        # Compute dependency summary across all prompts (if downstream checked)
        dependency_summary = {}
        if parsed_downstream:
            for ds in parsed_downstream:
                ds_id = ds["id"]
                all_mean_changes = [
                    r.get("downstream_effects", {}).get(ds_id, {}).get("mean_change_percent", 0)
                    for r in all_results
                ]
                overall_mean = sum(all_mean_changes) / len(all_mean_changes) if all_mean_changes else 0
                dependency_summary[ds_id] = {
                    "mean_change_percent": overall_mean,
                    "dependency_strength": "strong" if abs(overall_mean) > 30 else "moderate" if abs(overall_mean) > 10 else "weak",
                }

        total_changed = sum(1 for r in all_results if r["completion_changed"])

        result = {
            "layer": layer,
            "neuron_idx": neuron_idx,
            "total_prompts": len(prompt_data),
            "total_changed": total_changed,
            "change_rate": total_changed / len(prompt_data) if prompt_data else 0,
            "per_prompt_results": all_results,
            "category_stats": category_stats,
        }

        if parsed_downstream:
            result["dependency_summary"] = dependency_summary
            result["downstream_neurons_checked"] = [ds["id"] for ds in parsed_downstream]

        return result


async def tool_batch_ablate_and_generate(
    layer: int,
    neuron_idx: int,
    prompts: list[str] | None = None,
    use_categorized_prompts: bool = False,
    activation_threshold: float = 0.5,
    max_new_tokens: int = 10,
    max_prompts: int = 300,
    downstream_neurons: list[str] | None = None,
    truncate_to_activation: bool = False,
    generation_format: str = "continuation",
) -> dict[str, Any]:
    """Unified ablation tool: run ablation, generate completions, and check downstream neurons.

    This is the consolidated ablation tool that combines generation and downstream checking.
    Can use provided prompts or automatically extract activating prompts from category selectivity.

    Args:
        layer: Target layer
        neuron_idx: Target neuron index
        prompts: List of prompts to test (optional if use_categorized_prompts=True)
        use_categorized_prompts: If True, use activating prompts from protocol state
        activation_threshold: Minimum activation to include (for categorized prompts)
        max_new_tokens: Number of tokens to generate
        max_prompts: Maximum number of prompts to process
        downstream_neurons: Optional list of downstream neuron IDs to check (e.g., ["L26/N1234"]).
            If None and connectivity data exists, automatically uses downstream neurons from connectivity.
        truncate_to_activation: If True and using categorized prompts with position info,
            truncate each prompt to the position where the target neuron fires and generate from there.
            This tests what the neuron would output when it activates.

    Returns:
        Dict with:
        - per_prompt_results: Per-prompt breakdown with completions and downstream effects
        - category_stats: Category-level statistics
        - dependency_summary: Aggregated dependency strength per downstream neuron (if checked)
    """
    state = get_protocol_state()

    prompt_data_list = []  # List of dicts with prompt, category, position

    if use_categorized_prompts:
        # Extract activating prompts from protocol state with full metadata
        if not state.categorized_prompts:
            return {"error": "No categorized prompts available. Run category_selectivity first."}

        # First collect ALL prompts with their activations to compute z-score threshold
        all_items = []
        for category, cat_prompt_data in state.categorized_prompts.items():
            for item in cat_prompt_data:
                if isinstance(item, dict):
                    activation = item.get("activation", 0)
                    prompt = item.get("prompt", "")
                    position = item.get("position")
                else:
                    activation = 0
                    prompt = str(item)
                    position = None
                if prompt:
                    all_items.append({
                        "prompt": prompt,
                        "category": category,
                        "position": position,
                        "activation": activation,
                    })

        # Compute z-score threshold: use prompts > 1 std dev above mean activation
        # This adapts to the neuron's activation distribution instead of a fixed threshold
        activations = [x["activation"] for x in all_items]
        if activations:
            import math
            mean_act = sum(activations) / len(activations)
            var = sum((a - mean_act) ** 2 for a in activations) / len(activations)
            std_act = math.sqrt(var) if var > 0 else 0.1
            z_threshold = mean_act + std_act  # 1 std dev above mean
            # But never go below the explicit activation_threshold as a floor
            effective_threshold = max(z_threshold, activation_threshold)
        else:
            effective_threshold = activation_threshold

        for item in all_items:
            if item["activation"] >= effective_threshold:
                prompt_data_list.append(item)

        if not prompt_data_list:
            # Fall back to absolute threshold if z-score is too strict
            prompt_data_list = [x for x in all_items if x["activation"] >= activation_threshold]

        if not prompt_data_list:
            return {"error": f"No prompts with activation >= {activation_threshold} (z-threshold was {effective_threshold:.2f})"}

        # Sort by activation descending so highest-activation prompts are used first
        prompt_data_list.sort(key=lambda x: x.get("activation", 0), reverse=True)
        print(f"  [BatchAblation] Using {len(prompt_data_list)} prompts (z-threshold={effective_threshold:.2f}, mean={mean_act:.2f}, std={std_act:.2f})")

    elif prompts:
        prompt_data_list = [{"prompt": p, "category": None, "position": None} for p in prompts]
    else:
        return {"error": "Either provide prompts or set use_categorized_prompts=True"}

    # Limit prompts
    if len(prompt_data_list) > max_prompts:
        print(f"  [BatchAblation] Limiting to {max_prompts} prompts (from {len(prompt_data_list)})")
        prompt_data_list = prompt_data_list[:max_prompts]

    # Auto-populate downstream neurons from connectivity if not specified
    actual_downstream = downstream_neurons
    if actual_downstream is None and state.connectivity_analyzed and state.connectivity_data:
        downstream_list = state.connectivity_data.get("downstream_targets", [])
        actual_downstream = [
            d.get("neuron_id") for d in downstream_list
            if d.get("neuron_id") and not d.get("target", "").startswith("LOGIT")
        ]
        if actual_downstream:
            print(f"  [BatchAblation] Auto-using {len(actual_downstream)} downstream neurons from connectivity")

    print(f"  [BatchAblation] Processing {len(prompt_data_list)} prompts")
    if actual_downstream:
        print(f"  [BatchAblation] Checking {len(actual_downstream)} downstream neurons at all positions")
    if truncate_to_activation:
        print("  [BatchAblation] Truncating prompts to activation position for generation")

    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        result = await client.batch_ablate_and_generate(
            layer=layer, neuron_idx=neuron_idx, prompt_data=prompt_data_list,
            downstream_neurons=actual_downstream, max_new_tokens=max_new_tokens,
            truncate_to_activation=truncate_to_activation,
            generation_format=generation_format,
        )
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _CUDA_EXECUTOR,
            _sync_batch_ablate_and_generate,
            layer,
            neuron_idx,
            prompt_data_list,
            actual_downstream,
            max_new_tokens,
            truncate_to_activation,
            generation_format,
        )

    # Extract examples where output changed (up to 10)
    changed_examples = []
    for r in result.get("per_prompt_results", []):
        if r.get("completion_changed") and len(changed_examples) < 10:
            changed_examples.append({
                "prompt": r.get("prompt", "")[:200],
                "category": r.get("category"),
                "baseline_completion": r.get("baseline_completion", "")[:500],
                "ablated_completion": r.get("ablated_completion", "")[:500],
                "max_shift": r.get("max_shift", 0),
                "downstream_effects": r.get("downstream_effects", {}),
            })

    # Store results in protocol state
    stored_result = {
        "type": "batch",
        "layer": layer,
        "neuron_idx": neuron_idx,
        "total_prompts": result["total_prompts"],
        "total_changed": result["total_changed"],
        "change_rate": result["change_rate"],
        "category_stats": result.get("category_stats", {}),
        "max_new_tokens": max_new_tokens,
        "changed_examples": changed_examples,
        "truncate_to_activation": truncate_to_activation,
    }

    # Include downstream dependency data if checked
    if result.get("dependency_summary"):
        stored_result["dependency_summary"] = result["dependency_summary"]
        stored_result["downstream_neurons_checked"] = result.get("downstream_neurons_checked", [])

        # Extract per_prompt downstream effects for dashboard compatibility
        per_prompt_downstream = []
        for pr in result.get("per_prompt_results", []):
            if pr.get("downstream_effects"):
                per_prompt_downstream.append({
                    "prompt": pr.get("prompt", "")[:100],
                    "downstream_effects": pr.get("downstream_effects", {}),
                })

        # Store in downstream_dependency_results with full data for dashboard tables
        state.downstream_dependency_results.append({
            "target_layer": layer,
            "target_neuron": neuron_idx,
            "downstream_neurons": result.get("downstream_neurons_checked", []),
            "per_prompt_results": per_prompt_downstream,
            "dependency_summary": result["dependency_summary"],
        })
        update_protocol_state(
            downstream_dependency_tested=True,
            downstream_dependency_prompt_count=result["total_prompts"]
        )

    state.multi_token_ablation_results.append(stored_result)
    update_protocol_state(
        multi_token_ablation_done=True,
        batch_ablation_done=True,
        batch_ablation_prompt_count=result["total_prompts"]
    )

    print(f"  [BatchAblation] Complete: {result['total_changed']}/{result['total_prompts']} changed ({result['change_rate']:.1%})")
    if changed_examples:
        print(f"  [BatchAblation] Stored {len(changed_examples)} examples of changed completions")

    if result.get("dependency_summary"):
        print("  [BatchAblation] Downstream dependency summary:")
        for ds_id, summary in result["dependency_summary"].items():
            print(f"    - {ds_id}: {summary['dependency_strength']} ({summary['mean_change_percent']:.1f}% mean change)")

    if result.get("category_stats"):
        sorted_cats = sorted(
            result["category_stats"].items(),
            key=lambda x: x[1]["change_rate"],
            reverse=True
        )
        print("  [BatchAblation] Top affected categories:")
        for cat, stats in sorted_cats[:5]:
            print(f"    - {cat}: {stats['changed']}/{stats['total']} ({stats['change_rate']:.1%})")

    return result


def _sync_batch_steer_and_generate(
    layer: int,
    neuron_idx: int,
    prompts: list[str],
    categories: list[str] | None = None,
    steering_values: list[float] | None = None,
    max_new_tokens: int = 10,
    batch_size: int = 8,
    generation_format: str = "continuation",
    downstream_neurons: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Synchronous implementation of batch_steer_and_generate.

    Runs steering on multiple prompts efficiently using batched generation.
    Tests multiple steering values on all prompts for dose-response analysis.
    Uses left padding for correct decoder-only generation.
    Optionally monitors downstream neuron activations during generation.

    Args:
        layer: Target layer
        neuron_idx: Target neuron index
        prompts: List of prompts to test
        categories: Optional list of category labels (same length as prompts)
        steering_values: List of steering magnitudes to test (default: [0, 5, 10]).
                        Zero is treated as baseline.
        max_new_tokens: Number of tokens to generate
        batch_size: Batch size for processing (default 8 for correctness)
        downstream_neurons: Optional list of downstream neurons to monitor,
                           e.g. [{"id": "L26/N1234", "layer": 26, "neuron_idx": 1234}]

    Returns:
        Dict with per-prompt results organized by steering value, category-level statistics,
        and downstream dependency_summary (if downstream_neurons provided)
    """
    # Default steering values include baseline (0) and two positive values
    if steering_values is None:
        steering_values = [0, 5, 10]

    # Normalize to int when possible (JSON deserializes 0 as 0.0, causing
    # str(0.0)="0.0" vs hardcoded "0" key lookups)
    steering_values = [int(sv) if sv == int(sv) else sv for sv in steering_values]

    # Ensure 0 is in steering_values for baseline comparison
    if 0 not in steering_values:
        steering_values = [0] + list(steering_values)
    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        # Store original padding side and set to left
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        mlp = model.model.layers[layer].mlp

        # Steering hook factory - creates hook for specific steering value
        def make_steering_hook(sv: float):
            def steering_hook(module, args, kwargs):
                x = args[0]  # Shape: (batch, seq_len, intermediate_dim=14336)
                modified = x.clone()
                if neuron_idx < modified.shape[2]:
                    modified[:, :, neuron_idx] += sv
                return (modified,) + args[1:], kwargs
            return steering_hook

        all_results = []
        print(f"  [BatchSteering] Testing steering values: {steering_values}")

        # Process in batches
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_categories = categories[batch_start:batch_end] if categories else [None] * len(batch_prompts)

            # Format prompts based on generation_format
            texts = [format_generation_prompt(p, generation_format) for p in batch_prompts]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            input_len = inputs["input_ids"].shape[1]

            # Initialize result dicts for each prompt in batch
            batch_results = []
            for prompt, category in zip(batch_prompts, batch_categories):
                batch_results.append({
                    "prompt": prompt[:100],
                    "category": category,
                    "steering_results": {},  # Will hold results keyed by steering value
                })

            # Run generation and logit analysis for each steering value
            for sv in steering_values:
                if sv == 0:
                    # Baseline (no steering)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        logit_out = model(**inputs)
                        logits = logit_out.logits[:, -1].float()
                else:
                    # Steered generation
                    hook = make_steering_hook(sv)
                    handle = mlp.down_proj.register_forward_pre_hook(hook, with_kwargs=True)
                    try:
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                            logit_out = model(**inputs)
                            logits = logit_out.logits[:, -1].float()
                    finally:
                        handle.remove()

                # Decode and store results for each prompt
                for i in range(len(batch_prompts)):
                    completion = tokenizer.decode(
                        outputs[i][input_len:], skip_special_tokens=True
                    )

                    # Store steering result for this value
                    batch_results[i]["steering_results"][str(sv)] = {
                        "completion": completion,
                        "logits": logits[i].cpu() if sv == 0 else None,  # Store baseline logits for comparison
                    }

            # Now compute logit shifts relative to baseline for each prompt
            for i in range(len(batch_prompts)):
                baseline_logits = batch_results[i]["steering_results"]["0"]["logits"]
                baseline_completion = batch_results[i]["steering_results"]["0"]["completion"]

                # Get top-k baseline tokens for comparison
                top_k = 10
                top_values, top_indices = torch.topk(baseline_logits, top_k)

                # Compute shifts for each non-zero steering value
                any_changed = False
                for sv in steering_values:
                    if sv == 0:
                        continue

                    steered_completion = batch_results[i]["steering_results"][str(sv)]["completion"]
                    changed = steered_completion != baseline_completion
                    if changed:
                        any_changed = True

                    # We need to recompute logits for steered to get shifts
                    # (We only stored baseline logits to save memory)
                    batch_results[i]["steering_results"][str(sv)]["changed"] = changed

                # Clean up stored logits to save memory
                batch_results[i]["steering_results"]["0"]["logits"] = None
                batch_results[i]["baseline_completion"] = baseline_completion
                batch_results[i]["any_changed"] = any_changed

            all_results.extend(batch_results)

        # Restore tokenizer padding side
        tokenizer.padding_side = original_padding_side

        # Compute per-steering-value statistics
        steering_stats = {}
        for sv in steering_values:
            if sv == 0:
                continue
            sv_key = str(sv)
            changed_count = sum(
                1 for r in all_results
                if r["steering_results"].get(sv_key, {}).get("changed", False)
            )
            steering_stats[sv_key] = {
                "total": len(all_results),
                "changed": changed_count,
                "change_rate": changed_count / len(all_results) if all_results else 0,
            }

        # Compute category-level statistics (for each steering value)
        category_stats = {}
        if categories:
            for cat in set(categories):
                cat_results = [r for r in all_results if r["category"] == cat]
                if cat_results:
                    # Per-steering-value stats for this category
                    sv_stats = {}
                    for sv in steering_values:
                        if sv == 0:
                            continue
                        sv_key = str(sv)
                        changed_count = sum(
                            1 for r in cat_results
                            if r["steering_results"].get(sv_key, {}).get("changed", False)
                        )
                        sv_stats[sv_key] = {
                            "changed": changed_count,
                            "change_rate": changed_count / len(cat_results) if cat_results else 0,
                        }

                    # Also compute any_changed across all steering values
                    any_changed_count = sum(1 for r in cat_results if r.get("any_changed", False))

                    category_stats[cat] = {
                        "total": len(cat_results),
                        "any_changed": any_changed_count,
                        "any_change_rate": any_changed_count / len(cat_results) if cat_results else 0,
                        "per_steering_value": sv_stats,
                    }

        # Overall change count (any steering value changed the output)
        total_any_changed = sum(1 for r in all_results if r.get("any_changed", False))

        # Downstream neuron monitoring (if requested)
        # Test at ALL non-zero steering values and compute slope + R² per downstream neuron
        dependency_summary = {}
        if downstream_neurons:
            non_zero_svs = sorted([sv for sv in steering_values if sv != 0])
            print(f"  [BatchSteering] Monitoring {len(downstream_neurons)} downstream neurons across {len(non_zero_svs)} steering values")

            # Sample up to 30 prompts for downstream monitoring (multiple sv's = more compute)
            sample_prompts = prompts[:30]
            sample_texts = [format_generation_prompt(p, generation_format) for p in sample_prompts]

            for ds in downstream_neurons:
                ds_id = ds.get("id", "")
                ds_layer = ds.get("layer", 0)
                ds_idx = ds.get("neuron_idx", 0)

                if ds_layer >= len(model.model.layers):
                    continue

                ds_mlp = model.model.layers[ds_layer].mlp

                # Collect baseline activations first
                baseline_acts = []
                for b_start in range(0, len(sample_texts), 4):
                    b_texts = sample_texts[b_start:b_start + 4]
                    b_inputs = tokenizer(b_texts, return_tensors="pt", padding=True, truncation=True).to(device)

                    captured = []
                    def bl_hook(module, input, output, cap=captured, idx=ds_idx):
                        hidden = input[0]
                        gate = module.gate_proj(hidden)
                        up = module.up_proj(hidden)
                        intermediate = torch.nn.functional.silu(gate) * up
                        cap.append(intermediate[:, -1, idx].detach().cpu().tolist())
                        return output

                    handle = ds_mlp.register_forward_hook(bl_hook)
                    with torch.no_grad():
                        model(**b_inputs)
                    handle.remove()
                    if captured:
                        baseline_acts.extend(captured[-1])

                if not baseline_acts:
                    continue

                mean_baseline = sum(abs(a) for a in baseline_acts) / len(baseline_acts)

                # Test each steering value
                dose_response = []
                for sv in non_zero_svs:
                    sv_acts = []
                    for b_start in range(0, len(sample_texts), 4):
                        b_texts = sample_texts[b_start:b_start + 4]
                        b_inputs = tokenizer(b_texts, return_tensors="pt", padding=True, truncation=True).to(device)

                        captured_sv = []
                        def sv_hook(module, input, output, cap=captured_sv, idx=ds_idx):
                            hidden = input[0]
                            gate = module.gate_proj(hidden)
                            up = module.up_proj(hidden)
                            intermediate = torch.nn.functional.silu(gate) * up
                            cap.append(intermediate[:, -1, idx].detach().cpu().tolist())
                            return output

                        hook_steer = make_steering_hook(sv)
                        handle_steer = mlp.down_proj.register_forward_pre_hook(hook_steer, with_kwargs=True)
                        handle_ds = ds_mlp.register_forward_hook(sv_hook)
                        with torch.no_grad():
                            model(**b_inputs)
                        handle_steer.remove()
                        handle_ds.remove()
                        if captured_sv:
                            sv_acts.extend(captured_sv[-1])

                    if sv_acts:
                        mean_sv = sum(abs(a) for a in sv_acts) / len(sv_acts)
                        if mean_baseline > 0.01:
                            pct_change = ((mean_sv - mean_baseline) / mean_baseline) * 100
                        else:
                            pct_change = 0.0
                        dose_response.append({
                            "steering_value": sv,
                            "mean_change_percent": round(pct_change, 1),
                            "n_prompts": len(sv_acts),
                        })

                # Compute slope + R² from dose-response curve
                if len(dose_response) >= 2:
                    x_vals = [d["steering_value"] for d in dose_response]
                    y_vals = [d["mean_change_percent"] for d in dose_response]
                    x_mean = sum(x_vals) / len(x_vals)
                    y_mean = sum(y_vals) / len(y_vals)

                    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
                    denominator = sum((x - x_mean) ** 2 for x in x_vals)

                    if denominator > 0:
                        slope = numerator / denominator
                        intercept = y_mean - slope * x_mean
                        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_vals, y_vals))
                        ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
                        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    else:
                        slope = 0
                        r_squared = 0

                    effect_direction = "excitatory" if slope > 0 else "inhibitory" if slope < 0 else "neutral"

                    dependency_summary[ds_id] = {
                        "slope": round(slope, 4),
                        "r_squared": round(r_squared, 3),
                        "effect_direction": effect_direction,
                        "mean_baseline": round(mean_baseline, 4),
                        "dose_response_curve": dose_response,
                        "n_prompts_measured": len(baseline_acts),
                        "steering_values_tested": non_zero_svs,
                    }
                    strength = "strong" if abs(slope) > 5 else "moderate" if abs(slope) > 1 else "weak"
                    print(f"    {ds_id}: slope={slope:+.2f}, R²={r_squared:.3f} ({strength})")
                elif dose_response:
                    # Single steering value fallback
                    d = dose_response[0]
                    dependency_summary[ds_id] = {
                        "slope": None,
                        "r_squared": None,
                        "effect_direction": "unknown",
                        "mean_baseline": round(mean_baseline, 4),
                        "mean_change_percent": d["mean_change_percent"],
                        "dose_response_curve": dose_response,
                        "n_prompts_measured": len(baseline_acts),
                        "steering_values_tested": non_zero_svs,
                    }
                    print(f"    {ds_id}: {d['mean_change_percent']:+.1f}% (single sv, no slope)")

        result = {
            "layer": layer,
            "neuron_idx": neuron_idx,
            "steering_values": steering_values,
            "total_prompts": len(prompts),
            "total_changed": total_any_changed,  # Changed by any steering value
            "change_rate": total_any_changed / len(prompts) if prompts else 0,
            "per_steering_value": steering_stats,  # Stats per steering value
            "per_prompt_results": all_results,
            "category_stats": category_stats,
        }
        if dependency_summary:
            result["dependency_summary"] = dependency_summary
        return result


async def tool_batch_steer_and_generate(
    layer: int,
    neuron_idx: int,
    prompts: list[str] | None = None,
    use_categorized_prompts: bool = False,
    activation_threshold: float = 0.5,
    steering_values: list[float] | None = None,
    max_new_tokens: int = 10,
    batch_size: int = 8,
    max_prompts: int = 300,
    generation_format: str = "continuation",
    downstream_neurons: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run steering on multiple prompts efficiently using batched generation.

    Tests multiple steering values on all prompts for dose-response analysis.
    Can either use provided prompts or automatically extract activating prompts
    from the category selectivity results. Optionally monitors downstream neurons.

    Args:
        layer: Target layer
        neuron_idx: Target neuron index
        prompts: List of prompts to test (optional if use_categorized_prompts=True)
        use_categorized_prompts: If True, use activating prompts from protocol state
        activation_threshold: Minimum activation to include (for categorized prompts)
        steering_values: List of steering magnitudes to test (default: [0, 5, 10]).
                        Zero is baseline, negative values suppress the neuron.
        max_new_tokens: Number of tokens to generate
        batch_size: Batch size for processing (default 8 for correctness)
        max_prompts: Maximum number of prompts to process
        downstream_neurons: Optional list of downstream neurons to monitor,
                           e.g. [{"id": "L26/N1234", "layer": 26, "neuron_idx": 1234}]

    Returns:
        Dict with per-prompt results organized by steering value, category-level statistics,
        and downstream dependency_summary (if downstream_neurons provided)
    """
    # Default steering values for dose-response curve
    if steering_values is None:
        steering_values = [0, 5, 10]
    state = get_protocol_state()

    prompt_list = []
    category_list = []

    if use_categorized_prompts:
        # Extract activating prompts from protocol state
        if not state.categorized_prompts:
            return {"error": "No categorized prompts available. Run category_selectivity first."}

        # Collect ALL prompts with activations to compute z-score threshold
        _all_tuples = []
        for category, prompt_data in state.categorized_prompts.items():
            for item in prompt_data:
                if isinstance(item, dict):
                    activation = item.get("activation", 0)
                    prompt = item.get("prompt", "")
                else:
                    activation = 0
                    prompt = str(item)

                if prompt:
                    _all_tuples.append((activation, prompt, category))

        # Compute z-score threshold: prompts > 1 std dev above mean
        activations = [t[0] for t in _all_tuples]
        if activations:
            import math
            mean_act = sum(activations) / len(activations)
            var = sum((a - mean_act) ** 2 for a in activations) / len(activations)
            std_act = math.sqrt(var) if var > 0 else 0.1
            z_threshold = mean_act + std_act
            effective_threshold = max(z_threshold, activation_threshold)
        else:
            mean_act = std_act = 0
            effective_threshold = activation_threshold

        MIN_STEERING_PROMPTS = 20  # Minimum to satisfy save gate

        _prompt_tuples = [t for t in _all_tuples if t[0] >= effective_threshold]

        if len(_prompt_tuples) < MIN_STEERING_PROMPTS:
            # Z-threshold too strict — fall back to absolute threshold
            _prompt_tuples = [t for t in _all_tuples if t[0] >= activation_threshold]

        if len(_prompt_tuples) < MIN_STEERING_PROMPTS:
            # Still not enough — use all available prompts sorted by activation
            _prompt_tuples = sorted(_all_tuples, key=lambda x: x[0], reverse=True)
            print(f"  [BatchSteering] WARNING: Only {len(_prompt_tuples)} prompts available, using all (min={MIN_STEERING_PROMPTS})")

        if not _prompt_tuples:
            return {"error": f"No prompts with activation >= {activation_threshold} (z-threshold was {effective_threshold:.2f})"}

        # Sort by activation descending
        _prompt_tuples.sort(key=lambda x: x[0], reverse=True)
        prompt_list = [t[1] for t in _prompt_tuples]
        category_list = [t[2] for t in _prompt_tuples]

        print(f"  [BatchSteering] Using {len(prompt_list)} prompts (z-threshold={effective_threshold:.2f}, mean={mean_act:.2f}, std={std_act:.2f})")

    elif prompts:
        prompt_list = prompts
        category_list = None
    else:
        return {"error": "Either provide prompts or set use_categorized_prompts=True"}

    # Limit prompts
    if len(prompt_list) > max_prompts:
        print(f"  [BatchSteering] Limiting to {max_prompts} prompts (from {len(prompt_list)})")
        prompt_list = prompt_list[:max_prompts]
        if category_list:
            category_list = category_list[:max_prompts]

    print(f"  [BatchSteering] Processing {len(prompt_list)} prompts in batches of {batch_size}")
    print(f"  [BatchSteering] Testing steering values: {steering_values}")

    # Auto-populate downstream neurons from connectivity if not provided
    parsed_downstream = downstream_neurons
    if parsed_downstream is None and state.connectivity_data:
        downstream_list = state.connectivity_data.get("downstream_targets", [])
        if downstream_list:
            parsed_downstream = []
            for conn in downstream_list:
                nid = conn.get("neuron_id", "")
                if "/" in nid and not nid.startswith("LOGIT"):
                    try:
                        dl = int(nid.split("/")[0].replace("L", ""))
                        dn = int(nid.split("/")[1].replace("N", ""))
                        parsed_downstream.append({"id": nid, "layer": dl, "neuron_idx": dn})
                    except (ValueError, IndexError):
                        pass
            if parsed_downstream:
                print(f"  [BatchSteering] Auto-populated {len(parsed_downstream)} downstream neurons from connectivity")

    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        result = await client.batch_steer_and_generate(
            layer=layer, neuron_idx=neuron_idx, prompts=prompt_list,
            categories=category_list, steering_values=steering_values,
            max_new_tokens=max_new_tokens, batch_size=batch_size,
            generation_format=generation_format,
            downstream_neurons=parsed_downstream,
        )
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _CUDA_EXECUTOR,
            _sync_batch_steer_and_generate,
            layer,
            neuron_idx,
            prompt_list,
            category_list,
            steering_values,
            max_new_tokens,
            batch_size,
            generation_format,
            parsed_downstream,
        )

    # Extract examples where output changed (up to 10, showing all steering values)
    changed_examples = []
    for r in result.get("per_prompt_results", []):
        if r.get("any_changed") and len(changed_examples) < 10:
            # Build steering results dict showing completions for each value
            steering_completions = {}
            for sv_key, sv_result in r.get("steering_results", {}).items():
                steering_completions[sv_key] = {
                    "completion": sv_result.get("completion", "")[:500],
                    "changed": sv_result.get("changed", False) if sv_key != "0" else False,
                }
            changed_examples.append({
                "prompt": r.get("prompt", "")[:200],  # Truncate long prompts
                "category": r.get("category"),
                "baseline_completion": r.get("baseline_completion", "")[:500],
                "steering_completions": steering_completions,  # All steering values' results
            })

    # Store results in protocol state (including changed examples + downstream slopes)
    stored_result = {
        "type": "batch_multi_value",
        "layer": layer,
        "neuron_idx": neuron_idx,
        "steering_values": steering_values,
        "total_prompts": result["total_prompts"],
        "total_changed": result["total_changed"],
        "change_rate": result["change_rate"],
        "per_steering_value": result.get("per_steering_value", {}),
        "category_stats": result.get("category_stats", {}),
        "max_new_tokens": max_new_tokens,
        "changed_examples": changed_examples,
    }
    if result.get("dependency_summary"):
        stored_result["downstream_steering_slopes"] = result["dependency_summary"]
    state.multi_token_steering_results.append(stored_result)
    update_protocol_state(
        batch_steering_done=True,
        batch_steering_prompt_count=result["total_prompts"]
    )

    print(f"  [BatchSteering] Complete: {result['total_changed']}/{result['total_prompts']} changed by any steering value ({result['change_rate']:.1%})")
    # Print per-steering-value stats
    for sv_key, sv_stats in result.get("per_steering_value", {}).items():
        print(f"    sv={sv_key}: {sv_stats['changed']}/{sv_stats['total']} changed ({sv_stats['change_rate']:.1%})")
    if changed_examples:
        print(f"  [BatchSteering] Stored {len(changed_examples)} examples of changed completions")

    if result.get("category_stats"):
        # Sort by any_change_rate descending
        sorted_cats = sorted(
            result["category_stats"].items(),
            key=lambda x: x[1].get("any_change_rate", 0),
            reverse=True
        )
        print("  [BatchSteering] Top affected categories (by any steering value):")
        for cat, stats in sorted_cats[:5]:
            print(f"    - {cat}: {stats.get('any_changed', 0)}/{stats['total']} ({stats.get('any_change_rate', 0):.1%})")

    return result


# =============================================================================
# INTELLIGENT STEERING ANALYSIS (Sonnet-powered)
# =============================================================================

STEERING_PROMPT_GENERATION_SYSTEM = """You are an expert at designing steering experiments for neural network interpretability.

Your task: Generate diverse prompts that will REVEAL the effect of steering a neuron. The neuron could detect ANY concept - linguistic patterns, semantic domains, syntactic structures, formatting, emotions, entities, actions, or abstract concepts.

## Core Principles

1. **Decision boundaries**: Choose prompts where the next token could plausibly go multiple ways
2. **Avoid bias**: Don't use prompts that already strongly contain the target concept
3. **Test generalization**: Vary context, style, and domain to see if the effect is robust
4. **Include controls**: Some prompts should NOT be affected by steering (to confirm specificity)
5. **Test both directions**: Amplification (positive steering) and suppression (negative steering)

## CRITICAL: Understanding Prompt Formats

The target is Llama-3.1-8B-Instruct (chat-tuned). You have THREE format options for each prompt:

### FORMAT OPTIONS

**Format A: `chat_response`** - User asks, assistant responds from scratch
```json
{"format": "chat_response", "user_message": "What are common pain relievers?"}
```
- Standard chat interaction
- Model responds as helpful assistant
- Best for: Questions, requests, factual queries
- Display: "User: [question]" → "Assistant: [response]"

**Format B: `chat_continuation`** - User message + assistant prefill
```json
{"format": "chat_continuation", "user_message": "Continue:", "assistant_prefix": "The doctor prescribed"}
```
- Chat template with prefilled assistant response
- Model continues from the prefix as if it started writing
- Best for: Forcing specific continuations, avoiding refusals
- Display: "[prefix]" → "[continuation]"

**Format C: `raw_continuation`** - No chat template, direct text completion
```json
{"format": "raw_continuation", "text": "The enzyme inhibited by aspirin is"}
```
- No chat template at all - raw text completion
- Model continues text directly (like base model behavior)
- Best for: Pure factual completion, avoiding chat behaviors entirely
- Display: "[text]" → "[continuation]"

### WHEN TO USE EACH FORMAT

**Use `chat_response` when:**
- Testing how the model responds to natural user questions
- The query is unambiguous and won't trigger refusals
- You want to see the full assistant response style
- Examples: "What medications contain ibuprofen?", "List common NSAIDs"

**Use `chat_continuation` when:**
- Topic might trigger refusals in user turn
- You want to force continuation of specific text
- Testing style/content at decision boundaries
- Examples: user="Continue:" prefix="After taking aspirin, the patient"

**Use `raw_continuation` when:**
- You want pure text completion without chat behaviors
- Testing factual recall without assistant framing
- Avoiding "I'd be happy to help" or other chat artifacts
- Examples: "The mechanism of action of aspirin involves", "Common side effects include"

### AVOIDING REFUSALS

**High refusal risk** (use `chat_continuation` or `raw_continuation`):
- Medical advice contexts
- Incomplete sentences that seem "cut off"
- Anything that could be interpreted as asking for personal guidance

**Low refusal risk** (any format works):
- Factual questions with clear answers
- Code completion
- Creative writing continuations
- List completions

## PROMPT STRATEGIES - Use Multiple Approaches!

Different neurons respond to different prompt structures. Use a MIX of these strategies:

### 1. Incomplete Factual Statements
Best for: entities, facts, terminology, domain knowledge
- "Common examples of [domain] include"
- "The Wikipedia article describes this as"
- "Scientists classify this as a type of"

### 2. Creative Writing Continuations
Best for: style, tone, imagery, descriptive vocabulary
- "The old house creaked as"
- "She opened the letter and discovered"
- "In the distance, the mountains"

### 3. Dialogue/Conversation
Best for: emotions, social dynamics, speech patterns
- "He replied with a tone of"
- "'I never expected,' she said,"
- "The customer complained that"

### 4. Code/Technical Completions
Best for: programming concepts, syntax, technical terms
- "def calculate_total(items):"
- "The function returns a"
- "To fix this bug, you should"

### 5. List/Enumeration Contexts
Best for: categories, related concepts, taxonomies
- "Popular choices include"
- "The menu featured dishes like"
- "Key symptoms are"

### 6. Comparative/Contrastive
Best for: attributes, qualities, relationships
- "Unlike cats, dogs tend to"
- "This is faster than"
- "The main difference is that"

### 7. Sensory/Descriptive
Best for: colors, textures, sounds, physical properties
- "The fabric felt"
- "The sound was"
- "It tasted strongly of"

### 8. Temporal/Sequential
Best for: actions, processes, routines
- "First, you should"
- "After the meeting,"
- "The next step involves"

### 9. Causal/Explanatory
Best for: mechanisms, reasoning, cause-effect
- "This happens because"
- "The reason is that"
- "As a result,"

### 10. Hypothetical/Conditional
Best for: abstract reasoning, counterfactuals
- "If this were true, then"
- "Assuming the conditions are met,"
- "In that scenario,"

## EXAMPLE PROMPTS BY NEURON TYPE

**For a "water/liquid" neuron:**
- "The glass was filled with" (factual)
- "She dove into the" (narrative)
- "Common beverages include" (list)
- "The texture was" (sensory - control, shouldn't trigger water)

**For a "negation/contrast" neuron:**
- "This is not a" (direct negation)
- "Unlike the previous version," (contrast)
- "However, the results showed" (but-clause)
- "The sky was blue and" (control - no negation context)

**For a "formal/academic" style neuron:**
- "The study demonstrates that" (academic framing)
- "Furthermore, evidence suggests" (formal connective)
- "yo check out this" (control - informal)
- "According to the literature," (scholarly)

**For a "Python programming" neuron:**
- "def process_data(df):" (function def)
- "import pandas as" (import statement)
- "The recipe calls for" (control - cooking, not code)
- "for item in items:" (loop syntax)

**For an "emotion/sentiment" neuron:**
- "When I heard the news, I felt" (emotional context)
- "The movie made audiences" (reaction)
- "The quarterly report shows" (control - neutral business)
- "She couldn't hide her" (emotion reveal)

**For a "plural/quantity" neuron:**
- "Several people mentioned that" (plural subject)
- "The boxes contained" (plural object)
- "A single bird" (control - singular)
- "Many researchers believe" (quantity word)

## TEST TYPE DEFINITIONS (CRITICAL!)

The test types refer to HOW the prompt relates to the neuron's target concept:

### positive_amplify (50% of prompts)
**Prompts where the target concept is NOT the obvious completion.**
- The baseline completion should NOT contain the target content
- Positive steering should CAUSE the target content to appear where it wouldn't naturally
- These prompts should be NEUTRAL or about ADJACENT/COMPETING topics

Example for an NSAID neuron (promotes anti-inflammatory drug terms):
- GOOD: "The doctor recommended taking" → baseline: "some rest" → steered: "ibuprofen"
- GOOD: "For the muscle pain, she decided to" → baseline: "apply ice" → steered: "take an NSAID"
- BAD: "Common NSAIDs include" → baseline already primes for NSAIDs!
- BAD: "The active ingredient in Advil is" → already about a specific NSAID!

### positive_suppress (30% of prompts)
**Prompts where the target concept IS the obvious completion.**
- The baseline completion SHOULD contain the target content
- Negative steering should SUPPRESS the target content
- These prompts naturally prime for the target domain

Example for an NSAID neuron:
- GOOD: "Common anti-inflammatory medications include" → baseline: "ibuprofen..." → suppressed: different topic
- GOOD: "For arthritis pain, doctors often prescribe" → baseline: NSAIDs → suppressed: alternatives

### negative_control (20% of prompts)
**Prompts completely unrelated to the neuron's domain.**
- Steering should have NO effect
- Tests that the neuron is specific, not a general "change output" neuron

Example for an NSAID neuron:
- "Popular programming languages include" → unaffected by steering
- "The capital of France is" → unaffected by steering

## DIVERSITY REQUIREMENTS

You MUST include variety across:
1. **Test types**: Use the exact percentages above
2. **Expected effects**: Mix of strong_change, moderate_change, no_change
3. **Prompt strategies**: Use at least 4 different strategies from the list above
4. **Domains/contexts**: Don't repeat the same scenario - vary the setting

**MOST COMMON MISTAKE**: Making ALL prompts about the target domain. For `positive_amplify`, you MUST use prompts where the target content would NOT naturally appear!

## Output Format

Return a JSON object. Each prompt specifies its format and required fields:

{
  "steering_values": [0, -10, -5, 5, 10, 15],
  "prompts": [
    {
      "format": "chat_response" | "chat_continuation" | "raw_continuation",
      // For chat_response:
      "user_message": "The user's question or statement",
      // For chat_continuation:
      "user_message": "Brief instruction like 'Continue:'",
      "assistant_prefix": "Text the assistant has 'started'",
      // For raw_continuation:
      "text": "Raw text to continue (no chat template)",
      // All formats:
      "expected_effect": "strong_change" | "moderate_change" | "no_change",
      "test_type": "positive_amplify" | "positive_suppress" | "negative_control",
      "rationale": "Why this prompt tests the hypothesis"
    },
    ...
  ]
}

Examples:
- chat_response: {"format": "chat_response", "user_message": "What medications treat headaches?", ...}
- chat_continuation: {"format": "chat_continuation", "user_message": "Continue:", "assistant_prefix": "The doctor prescribed", ...}
- raw_continuation: {"format": "raw_continuation", "text": "Common pain relievers include", ...}"""

STEERING_ANALYSIS_SYSTEM = """You are an expert at analyzing neural network steering experiments.

Your task: Analyze the results of steering experiments and provide:
1. A concise summary of what the experiments reveal about the neuron's function
2. Key findings about promotes/suppresses behavior
3. Select 10 illustrative examples that best demonstrate the neuron's effect

## Understanding Test Types

Prompts have different test_type values that determine how to interpret results:

**positive_amplify**: Neutral prompts where baseline does NOT contain target content.
- SUCCESS = positive steering CAUSES target content to appear where it wouldn't naturally
- The BEST examples show target content appearing in unexpected contexts
- If baseline already contains target content, this is a BAD prompt (not useful for analysis)

**positive_suppress**: Primed prompts where baseline DOES contain target content.
- SUCCESS = negative steering SUPPRESSES target content from appearing
- Look for baseline having target content, then suppressed version having alternatives

**negative_control**: Unrelated prompts where steering should have NO effect.
- SUCCESS = no change at any steering value
- Change here suggests the neuron is not specific to the hypothesized function

## Quality Assessment

When assessing whether the hypothesis is supported:
1. Prioritize positive_amplify results - these show the neuron CAUSING content, not just correlating
2. Be skeptical of positive_amplify prompts that already prime for the target (bad experimental design)
3. The strongest evidence is target content appearing in contexts where it's surprising

## Handling Model Refusals

Some completions may be "refusals" where the model says things like:
- "I can't help with that"
- "I'm a language model..."
- "I don't have personal experiences"

When selecting illustrative examples:
1. STRONGLY PREFER examples with actual completions over refusals
2. A change from one refusal to another is NOT a meaningful steering effect
3. Only count as "hypothesis supported" if actual token generation changed meaningfully
4. Note if high refusal rate limits what can be concluded

When analyzing, report separately:
- Semantic change rate (actual content changed meaningfully)
- vs. Any change rate (includes refusal-to-refusal changes)"""


# Common refusal/meta-response patterns in instruction-tuned models
REFUSAL_PATTERNS = [
    # Direct refusals
    "i can't",
    "i cannot",
    "i'm not able to",
    "i am not able to",
    "i don't have",
    "i do not have",
    "i'm sorry",
    "i apologize",
    "i'm not sure i can",
    "i don't think i can",
    # AI self-identification
    "i'm a language model",
    "i am a language model",
    "i'm an ai",
    "i am an ai",
    "i'm just a",
    "i am just a",
    "as an ai",
    "as a language model",
    # Helper/assistant framing
    "i'm here to",
    "i am here to",
    "i'm happy to help",
    "i'd be happy to",
    "i'm not experiencing",
    "i don't experience",
    # Clarification requests (meta-responses)
    "could you clarify",
    "could you please",
    "could you provide",
    "can you clarify",
    "can you please",
    "i'm not certain",
    "it seems like you",
    "it looks like you",
    "it seems like your",
    "it looks like your",
    "it appears that",
    "it appears you",
    # Incomplete input responses
    "your sentence",
    "your question",
    "your message",
    "the sentence",
    "got cut off",
    "was cut off",
    "is incomplete",
    "seems incomplete",
    "you started to",
    "you were going to",
    "you mentioned",
]


def _is_refusal(text: str) -> bool:
    """Check if a completion is a refusal/meta-response rather than actual content."""
    if not text:
        return False
    text_lower = text.lower().strip()
    # Check if starts with common refusal patterns
    for pattern in REFUSAL_PATTERNS:
        if text_lower.startswith(pattern):
            return True
    return False


def _get_anthropic_api_key() -> str:
    """Get Anthropic API key from environment or .env file."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try to load from .env file
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("ANTHROPIC_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in environment or .env file")
    return api_key


def _call_sonnet_for_prompts(
    output_hypothesis: str,
    promotes: list[str],
    suppresses: list[str],
    n_prompts: int = 100,
    additional_instructions: str | None = None,
) -> dict[str, Any]:
    """Call Sonnet to generate steering test prompts."""
    from anthropic import Anthropic

    api_key = _get_anthropic_api_key()
    client = Anthropic(api_key=api_key)

    # Build additional instructions section if provided
    extra_section = ""
    if additional_instructions:
        extra_section = f"""
## Additional Focus/Instructions
{additional_instructions}
"""

    # Calculate diversity targets
    n_amplify = int(n_prompts * 0.50)  # 50% positive amplify
    n_suppress = int(n_prompts * 0.30)  # 30% positive suppress
    n_control = n_prompts - n_amplify - n_suppress  # 20% negative control

    user_prompt = f"""Generate {n_prompts} prompts to test this neuron's output hypothesis through steering:

## Output Hypothesis
{output_hypothesis}

## The neuron PROMOTES these tokens/concepts when active:
{', '.join(promotes[:15]) if promotes else 'Not specified'}

## The neuron SUPPRESSES these tokens/concepts when active:
{', '.join(suppresses[:15]) if suppresses else 'Not specified'}
{extra_section}
## STRICT Requirements

### Prompt Count by Type (MANDATORY)
- Exactly {n_amplify} prompts with test_type="positive_amplify"
- Exactly {n_suppress} prompts with test_type="positive_suppress"
- Exactly {n_control} prompts with test_type="negative_control"

### Expected Effect Distribution
- Mix of "strong_change", "moderate_change", and "no_change"
- negative_control prompts should have expected_effect="no_change"

### Prompt Design (CRITICAL - choose the best format for each prompt)

For EACH prompt, choose one of three formats:
- **chat_response**: User asks question, model responds (standard chat)
- **chat_continuation**: User message + assistant prefill (model continues from prefix)
- **raw_continuation**: No chat template, pure text completion

**positive_amplify prompts must NOT already prime for the target!**
- GOOD: {{"format": "chat_continuation", "user_message": "Continue:", "assistant_prefix": "The doctor recommended"}} → neutral
- GOOD: {{"format": "chat_response", "user_message": "What did she do after the workout?"}} → open question
- BAD: {{"format": "raw_continuation", "text": "Common NSAIDs include"}} → already primes for target!

**positive_suppress prompts SHOULD prime for the target:**
- GOOD: {{"format": "raw_continuation", "text": "For inflammation, anti-inflammatory drugs like"}}
- GOOD: {{"format": "chat_response", "user_message": "List common over-the-counter pain relievers"}}

**Avoid refusals - choose format wisely:**
- BAD: {{"format": "chat_response", "user_message": "What should I take for pain?"}} → refusal risk
- GOOD: {{"format": "chat_continuation", "user_message": "Continue:", "assistant_prefix": "For the headache, she took"}}
- GOOD: {{"format": "raw_continuation", "text": "After taking aspirin, the patient felt"}}

**Use variety across formats!** Don't use only one format - mix them based on what each prompt needs.

### Steering Values
Choose 5-6 values including 0 as baseline, e.g., [0, -10, -5, 5, 10, 15]

Return ONLY valid JSON with this exact structure:
{{
  "steering_values": [0, -10, -5, 5, 10, 15],
  "prompts": [
    {{"format": "chat_response", "user_message": "...", "expected_effect": "...", "test_type": "...", "rationale": "..."}},
    {{"format": "chat_continuation", "user_message": "...", "assistant_prefix": "...", "expected_effect": "...", "test_type": "...", "rationale": "..."}},
    {{"format": "raw_continuation", "text": "...", "expected_effect": "...", "test_type": "...", "rationale": "..."}},
    ...
  ]
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=12000,
        system=STEERING_PROMPT_GENERATION_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}]
    )

    # Parse JSON from response
    text = response.content[0].text.strip()
    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Find the JSON content between ``` markers
        json_lines = []
        in_json = False
        for line in lines:
            if line.startswith("```") and not in_json:
                in_json = True
                continue
            elif line.startswith("```") and in_json:
                break
            elif in_json:
                json_lines.append(line)
        text = "\n".join(json_lines)

    result = json.loads(text)

    # Validate and report prompt diversity
    prompts = result.get("prompts", [])
    if prompts:
        type_counts = {}
        effect_counts = {}
        for p in prompts:
            t = p.get("test_type", "unknown")
            e = p.get("expected_effect", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            effect_counts[e] = effect_counts.get(e, 0) + 1

        print(f"  [IntelligentSteering] Prompt diversity - test_types: {type_counts}")
        print(f"  [IntelligentSteering] Prompt diversity - expected_effects: {effect_counts}")

        # Warn if all same type (indicates Sonnet ignored diversity instructions)
        if len(type_counts) == 1:
            print(f"  [IntelligentSteering] WARNING: All prompts have same test_type={list(type_counts.keys())[0]}")
            print("  [IntelligentSteering] This may indicate Sonnet ignored diversity requirements")

    return result


def _call_sonnet_for_analysis(
    output_hypothesis: str,
    promotes: list[str],
    suppresses: list[str],
    steering_results: list[dict[str, Any]],
    steering_values: list[float],
    additional_instructions: str | None = None,
) -> dict[str, Any]:
    """Call Sonnet to analyze steering results and select illustrative examples."""
    from anthropic import Anthropic

    api_key = _get_anthropic_api_key()
    client = Anthropic(api_key=api_key)

    # Prepare results summary for Sonnet with refusal detection
    results_summary = []
    refusal_count = 0
    for r in steering_results[:50]:  # Limit to avoid token overflow
        baseline = r.get("baseline_completion", "")
        baseline_is_refusal = _is_refusal(baseline)
        if baseline_is_refusal:
            refusal_count += 1

        result_entry = {
            "prompt": r["prompt"][:150],
            "expected_effect": r.get("expected_effect", "unknown"),
            "test_type": r.get("test_type", "unknown"),
            "baseline_completion": baseline[:100],
            "baseline_is_refusal": baseline_is_refusal,
            "steering_effects": {}
        }
        for sv_key, sv_data in r.get("steering_results", {}).items():
            if sv_key != "0":
                steered = sv_data.get("completion", "")
                steered_is_refusal = _is_refusal(steered)
                result_entry["steering_effects"][sv_key] = {
                    "completion": steered[:100],
                    "changed": sv_data.get("changed", False),
                    "is_refusal": steered_is_refusal,
                    # Semantic change = changed AND not both refusals
                    "semantic_change": sv_data.get("changed", False) and not (baseline_is_refusal and steered_is_refusal)
                }
        results_summary.append(result_entry)

    # Compute aggregate statistics with refusal awareness
    total = len(steering_results)
    stats_by_sv = {}
    for sv in steering_values:
        if sv == 0:
            continue
        sv_key = str(sv)
        changed = 0
        semantic_changed = 0
        for r in steering_results:
            sv_data = r.get("steering_results", {}).get(sv_key, {})
            baseline = r.get("baseline_completion", "")
            steered = sv_data.get("completion", "")
            if sv_data.get("changed", False):
                changed += 1
                # Semantic change: not both refusals
                if not (_is_refusal(baseline) and _is_refusal(steered)):
                    semantic_changed += 1
        stats_by_sv[sv_key] = {
            "changed": changed,
            "semantic_changed": semantic_changed,
            "total": total,
            "rate": changed/total if total else 0,
            "semantic_rate": semantic_changed/total if total else 0,
        }

    # Report refusal statistics
    refusal_rate = refusal_count / min(len(steering_results), 50) if steering_results else 0
    print(f"  [IntelligentSteering] Refusal rate in baseline: {refusal_count}/{min(len(steering_results), 50)} ({refusal_rate:.1%})")
    if refusal_rate > 0.3:
        print("  [IntelligentSteering] WARNING: High refusal rate may limit steering analysis quality")

    # Build additional instructions section if provided
    extra_section = ""
    if additional_instructions:
        extra_section = f"""
## Additional Analysis Focus
{additional_instructions}
"""

    # Build refusal warning if needed
    refusal_warning = ""
    if refusal_rate > 0.2:
        refusal_warning = f"""
## WARNING: High Refusal Rate
{refusal_count} of {min(len(steering_results), 50)} prompts ({refusal_rate:.0%}) resulted in model refusals.
When selecting illustrative examples, STRONGLY PREFER examples with actual completions.
A change from one refusal to another does NOT demonstrate the neuron's function.
"""

    user_prompt = f"""Analyze these steering experiment results:

## Output Hypothesis Being Tested
{output_hypothesis}

## Expected Effects
- PROMOTES: {', '.join(promotes[:10]) if promotes else 'Not specified'}
- SUPPRESSES: {', '.join(suppresses[:10]) if suppresses else 'Not specified'}

## Steering Values Tested
{steering_values}

## Aggregate Statistics (change rates by steering value)
Note: "semantic_rate" excludes refusal-to-refusal changes
{json.dumps(stats_by_sv, indent=2)}
{refusal_warning}
## Detailed Results (first 50 of {total} total)
Note: "baseline_is_refusal" and "is_refusal" flags indicate model refusals
{json.dumps(results_summary, indent=2)}
{extra_section}
## Your Task
Provide a JSON response with:
1. "summary": A 2-3 paragraph analysis (NO special characters like colons in strings - use commas instead)
2. "key_findings": List of 3-5 bullet points about the neuron's behavior
3. "hypothesis_supported": true, false, or "partial" (as a string)
4. "effective_steering_range": The steering values that had the most effect (as a string like "5 to 15")
5. "illustrative_examples": Exactly 10 examples that best demonstrate the effect, each with:
   - "prompt": The test prompt (keep short, under 100 chars)
   - "baseline": Baseline completion (keep short, under 80 chars)
   - "steered": Best steered completion (keep short, under 80 chars)
   - "steering_value": The value used (as integer)
   - "why_illustrative": Brief explanation (keep short, under 100 chars, NO colons)

CRITICAL: Return ONLY valid JSON. Avoid special characters in strings that could break JSON parsing.
Use simple punctuation (periods, commas) instead of colons in explanatory text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        system=STEERING_ANALYSIS_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}]
    )

    # Parse JSON from response with robust error handling
    text = response.content[0].text.strip()

    # Strip markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.startswith("```") and not in_json:
                in_json = True
                continue
            elif line.startswith("```") and in_json:
                break
            elif in_json:
                json_lines.append(line)
        text = "\n".join(json_lines)

    # Try parsing JSON with multiple strategies
    result = None
    parse_error = None

    # Strategy 1: Direct parse
    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        parse_error = str(e)

    # Strategy 2: Try to find JSON object boundaries
    if result is None:
        try:
            # Find first { and last }
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try to fix common JSON issues
    if result is None:
        try:
            import re
            # Replace smart quotes
            fixed = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
            # Remove trailing commas before } or ]
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            result = json.loads(fixed)
        except (json.JSONDecodeError, Exception):
            pass

    # If all parsing failed, return error with partial data extraction
    if result is None:
        print(f"  [IntelligentSteering] JSON parsing failed: {parse_error}")
        print("  [IntelligentSteering] Attempting to extract partial data...")

        # Try to extract key_findings using regex
        key_findings = []
        findings_match = re.search(r'"key_findings"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if findings_match:
            try:
                findings_text = "[" + findings_match.group(1) + "]"
                key_findings = json.loads(findings_text)
            except:
                pass

        # Try to extract hypothesis_supported
        hypothesis_supported = None
        if '"hypothesis_supported": true' in text.lower():
            hypothesis_supported = True
        elif '"hypothesis_supported": false' in text.lower():
            hypothesis_supported = False
        elif '"hypothesis_supported": "partial"' in text.lower():
            hypothesis_supported = "partial"

        return {
            "summary": f"Analysis parsing failed: {parse_error}. Raw response may contain insights but JSON was malformed.",
            "key_findings": key_findings,
            "hypothesis_supported": hypothesis_supported,
            "effective_steering_range": None,
            "illustrative_examples": [],  # Will be filled by fallback in caller
            "_parse_error": parse_error,
            "_raw_response_preview": text[:500],
        }

    return result


def _sync_run_intelligent_steering(
    layer: int,
    neuron_idx: int,
    prompts: list[dict[str, Any]],
    steering_values: list[float],
    max_new_tokens: int = 15,
    batch_size: int = 8,
    downstream_neurons: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run steering experiments on GPU for intelligent steering prompts.

    Args:
        layer: Target layer
        neuron_idx: Target neuron index
        prompts: List of prompt dicts with 'prompt', 'expected_effect', 'test_type', 'rationale'
        steering_values: List of steering magnitudes to test
        max_new_tokens: Tokens to generate
        batch_size: Batch size for processing
        downstream_neurons: Optional list of downstream neuron IDs (e.g., ['L20/N1234'])
                           to measure activation changes

    Returns:
        List of result dicts, each containing steering_results and optionally downstream_effects
    """
    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        mlp = model.model.layers[layer].mlp

        # Parse downstream neurons if provided
        parsed_downstream = []
        if downstream_neurons:
            for ds_id in downstream_neurons[:6]:  # Limit to 6 downstream neurons
                try:
                    parts = ds_id.replace("L", "").replace("N", "").split("/")
                    ds_layer = int(parts[0])
                    ds_idx = int(parts[1])
                    if ds_layer > layer:  # Must be in later layer
                        parsed_downstream.append({"id": ds_id, "layer": ds_layer, "idx": ds_idx})
                except Exception:
                    pass

        def make_steering_hook(sv: float):
            def steering_hook(module, args, kwargs):
                x = args[0]
                modified = x.clone()
                if neuron_idx < modified.shape[2]:
                    modified[:, :, neuron_idx] += sv
                return (modified,) + args[1:], kwargs
            return steering_hook

        def measure_downstream_activations(text: str) -> dict[str, float]:
            """Measure downstream neuron activations for a given text."""
            if not parsed_downstream:
                return {}

            ds_activations = {}
            inputs = tokenizer(text, return_tensors="pt").to(device)

            # Create hooks to capture downstream activations at last position
            captured_acts = {}
            handles = []

            def make_capture_hook(ds_id: str, ds_idx: int):
                def hook(module, input, output):
                    hidden = input[0]
                    gate = module.gate_proj(hidden)
                    up = module.up_proj(hidden)
                    intermediate = torch.nn.functional.silu(gate) * up
                    # Capture activation at last position
                    captured_acts[ds_id] = intermediate[0, -1, ds_idx].item()
                return hook

            for ds in parsed_downstream:
                ds_mlp = model.model.layers[ds["layer"]].mlp
                handle = ds_mlp.register_forward_hook(make_capture_hook(ds["id"], ds["idx"]))
                handles.append(handle)

            try:
                with torch.no_grad():
                    model(**inputs)
            finally:
                for h in handles:
                    h.remove()

            return captured_acts

        all_results = []

        # Helper to get display text for a prompt (for results)
        def get_prompt_display(p):
            """Get display text showing the prompt format and content."""
            fmt = p.get("format", "")

            if fmt == "raw_continuation":
                return p.get("text", "")
            elif fmt == "chat_continuation":
                prefix = p.get("assistant_prefix", "")
                user_msg = p.get("user_message", "Continue:")
                return f"[{user_msg}] {prefix}"
            elif fmt == "chat_response":
                return f"User: {p.get('user_message', '')}"
            elif "user_message" in p:
                # Legacy format
                prefix = p.get("assistant_prefix") or ""
                if prefix:
                    return f"[{p['user_message']}] {prefix}"
                return f"User: {p['user_message']}"
            return p.get("prompt", p.get("text", str(p)))

        def get_prompt_format_info(p):
            """Get format info for card rendering."""
            fmt = p.get("format", "")
            if fmt == "raw_continuation":
                return {"format": "raw", "text": p.get("text", "")}
            elif fmt == "chat_continuation":
                return {"format": "continuation", "instruction": p.get("user_message", ""), "prefix": p.get("assistant_prefix", "")}
            elif fmt == "chat_response":
                return {"format": "response", "user_message": p.get("user_message", "")}
            elif "user_message" in p:
                if p.get("assistant_prefix"):
                    return {"format": "continuation", "instruction": p.get("user_message", ""), "prefix": p.get("assistant_prefix", "")}
                return {"format": "response", "user_message": p.get("user_message", "")}
            return {"format": "raw", "text": p.get("prompt", str(p))}

        # Process in batches
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            # Format prompts for the model using the new format_steering_prompt
            texts = [format_steering_prompt(p) for p in batch_prompts]

            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            input_len = inputs["input_ids"].shape[1]

            # Initialize results for this batch
            batch_results = []
            for prompt_info in batch_prompts:
                batch_results.append({
                    "prompt": get_prompt_display(prompt_info),
                    "prompt_format": get_prompt_format_info(prompt_info),
                    "expected_effect": prompt_info.get("expected_effect", "unknown"),
                    "test_type": prompt_info.get("test_type", "unknown"),
                    "rationale": prompt_info.get("rationale", ""),
                    "steering_results": {},
                })

            # Run for each steering value
            for sv in steering_values:
                if sv == 0:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                else:
                    hook = make_steering_hook(sv)
                    handle = mlp.down_proj.register_forward_pre_hook(hook, with_kwargs=True)
                    try:
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                    finally:
                        handle.remove()

                # Decode completions and measure downstream activations
                for i in range(len(batch_prompts)):
                    completion = tokenizer.decode(
                        outputs[i][input_len:], skip_special_tokens=True
                    )
                    # texts[i] already has the formatted prompt
                    full_text = texts[i] + completion

                    result_entry = {"completion": completion}

                    # Measure downstream activations if requested
                    if parsed_downstream:
                        # For baseline (sv=0), store as baseline_downstream
                        # For steered, compute change vs baseline
                        ds_acts = measure_downstream_activations(full_text)
                        if sv == 0:
                            batch_results[i]["baseline_downstream"] = ds_acts
                        result_entry["downstream_activations"] = ds_acts

                    batch_results[i]["steering_results"][str(sv)] = result_entry

            # Compute which changed relative to baseline and downstream effects
            for br in batch_results:
                baseline = br["steering_results"].get("0", {}).get("completion", "")
                br["baseline_completion"] = baseline
                baseline_ds = br.get("baseline_downstream", {})

                for sv_key in br["steering_results"]:
                    if sv_key != "0":
                        sv_data = br["steering_results"][sv_key]
                        steered = sv_data.get("completion", "")
                        sv_data["changed"] = (steered != baseline)

                        # Compute downstream effects (change from baseline)
                        if baseline_ds and sv_data.get("downstream_activations"):
                            ds_effects = {}
                            for ds_id, steered_act in sv_data["downstream_activations"].items():
                                baseline_act = baseline_ds.get(ds_id, 0)
                                if abs(baseline_act) >= 0.1:
                                    change_pct = 100 * (steered_act - baseline_act) / abs(baseline_act)
                                else:
                                    change_pct = steered_act - baseline_act  # Absolute change for low baseline
                                ds_effects[ds_id] = {
                                    "baseline_activation": baseline_act,
                                    "steered_activation": steered_act,
                                    "change_percent": change_pct,
                                }
                            sv_data["downstream_effects"] = ds_effects

            all_results.extend(batch_results)

        tokenizer.padding_side = original_padding_side
        return all_results


async def tool_intelligent_steering_analysis(
    layer: int,
    neuron_idx: int,
    output_hypothesis: str,
    promotes: list[str] | None = None,
    suppresses: list[str] | None = None,
    additional_instructions: str | None = None,
    n_prompts: int = 100,
    max_new_tokens: int = 25,
    batch_size: int = 8,
    downstream_neurons: list[str] | None = None,
) -> dict[str, Any]:
    """Run intelligent steering analysis using Sonnet sub-agent.

    This tool replaces batch_steer_and_generate with a more sophisticated approach:
    1. Sonnet generates ~100 prompts designed to test the output hypothesis
    2. Sonnet specifies steering values based on the hypothesis
    3. System runs steering experiments on GPU
    4. Sonnet analyzes results and selects 10 illustrative examples

    Can be called multiple times with different additional_instructions to explore
    different aspects of the hypothesis (e.g., "focus on question contexts" or
    "test formal vs informal register").

    Args:
        layer: Target layer
        neuron_idx: Target neuron index
        output_hypothesis: Description of the neuron's output function to test
        promotes: List of tokens/concepts the neuron promotes (from output projections)
        suppresses: List of tokens/concepts the neuron suppresses
        additional_instructions: Optional guidance for Sonnet on what to focus on
                                (e.g., "focus on edge cases", "test in formal contexts")
        n_prompts: Number of test prompts to generate (default 100)
        max_new_tokens: Tokens to generate per completion (default 25, shows more effect)
        batch_size: Batch size for GPU processing
        downstream_neurons: Optional list of downstream neuron IDs to track activation changes.
                           If None, auto-detects from connectivity_data.

    Returns:
        Dict with analysis summary, key findings, 10 illustrative examples, and downstream_effects_summary
    """
    promotes = promotes or []
    suppresses = suppresses or []

    # Auto-detect downstream neurons from connectivity_data if not provided
    state = get_protocol_state()
    if downstream_neurons is None and state.connectivity_data:
        downstream_list = state.connectivity_data.get("downstream_targets", [])
        downstream_neurons = [d.get("neuron_id") for d in downstream_list[:6] if d.get("neuron_id")]
        if downstream_neurons:
            print(f"  [IntelligentSteering] Auto-detected {len(downstream_neurons)} downstream neurons to track")

    instructions_note = f" (focus: {additional_instructions[:50]}...)" if additional_instructions else ""
    print(f"  [IntelligentSteering] Phase 1: Generating {n_prompts} test prompts with Sonnet{instructions_note}...")

    # Phase 1: Generate prompts
    try:
        prompt_result = _call_sonnet_for_prompts(
            output_hypothesis=output_hypothesis,
            promotes=promotes,
            suppresses=suppresses,
            n_prompts=n_prompts,
            additional_instructions=additional_instructions,
        )
        steering_values = prompt_result.get("steering_values", [0, -5, 5, 10])
        prompts = prompt_result.get("prompts", [])
        print(f"  [IntelligentSteering] Generated {len(prompts)} prompts, steering values: {steering_values}")
    except Exception as e:
        return {"error": f"Sonnet prompt generation failed: {str(e)}"}

    if not prompts:
        return {"error": "Sonnet generated no prompts"}

    # Ensure 0 is in steering values for baseline
    if 0 not in steering_values:
        steering_values = [0] + list(steering_values)

    # Phase 2: Run steering experiments
    ds_note = f" (tracking {len(downstream_neurons)} downstream neurons)" if downstream_neurons else ""
    print(f"  [IntelligentSteering] Phase 2: Running {len(prompts)} experiments on GPU{ds_note}...")

    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        steering_results = await client.run_intelligent_steering(
            layer=layer, neuron_idx=neuron_idx, prompts=prompts,
            steering_values=steering_values, max_new_tokens=max_new_tokens,
            batch_size=batch_size, downstream_neurons=downstream_neurons,
        )
    else:
        loop = asyncio.get_running_loop()
        # Use functools.partial to pass all args including downstream_neurons
        from functools import partial
        steering_func = partial(
            _sync_run_intelligent_steering,
            layer=layer,
            neuron_idx=neuron_idx,
            prompts=prompts,
            steering_values=steering_values,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            downstream_neurons=downstream_neurons,
        )
        steering_results = await loop.run_in_executor(_CUDA_EXECUTOR, steering_func)

    # Compute aggregate stats with refusal awareness
    total = len(steering_results)
    stats_by_sv = {}
    refusal_count = sum(1 for r in steering_results if _is_refusal(r.get("baseline_completion", "")))

    for sv in steering_values:
        if sv == 0:
            continue
        sv_key = str(sv)
        changed = 0
        semantic_changed = 0
        for r in steering_results:
            sv_data = r.get("steering_results", {}).get(sv_key, {})
            if sv_data.get("changed", False):
                changed += 1
                # Semantic change: not both refusals
                baseline = r.get("baseline_completion", "")
                steered = sv_data.get("completion", "")
                if not (_is_refusal(baseline) and _is_refusal(steered)):
                    semantic_changed += 1
        stats_by_sv[sv_key] = {
            "changed": changed,
            "semantic_changed": semantic_changed,
            "total": total,
            "rate": changed/total if total else 0,
            "semantic_rate": semantic_changed/total if total else 0,
        }

    rates_str = ", ".join(f"sv={k}: {v['rate']:.1%} (semantic: {v['semantic_rate']:.1%})" for k, v in stats_by_sv.items())
    print(f"  [IntelligentSteering] Change rates: {rates_str}")
    print(f"  [IntelligentSteering] Baseline refusal rate: {refusal_count}/{total} ({refusal_count/total:.1%})")

    # Phase 3: Analyze results with Sonnet
    print("  [IntelligentSteering] Phase 3: Analyzing results with Sonnet...")
    try:
        analysis = _call_sonnet_for_analysis(
            output_hypothesis=output_hypothesis,
            promotes=promotes,
            suppresses=suppresses,
            steering_results=steering_results,
            steering_values=steering_values,
            additional_instructions=additional_instructions,
        )
    except Exception as e:
        # Return partial results if analysis fails
        analysis = {
            "summary": f"Analysis failed: {str(e)}",
            "key_findings": [],
            "hypothesis_supported": None,
            "illustrative_examples": [],  # Will be filled below
        }

    # If no illustrative examples from Sonnet, create smart fallback
    if not analysis.get("illustrative_examples"):
        print("  [IntelligentSteering] Using fallback: selecting best examples from raw results...")
        # Prefer examples where: (1) not refusal, (2) changed at positive sv
        scored_results = []
        for r in steering_results:
            baseline = r.get("baseline_completion", "")
            baseline_refusal = _is_refusal(baseline)

            # Find best steering value for this example
            best_sv = None
            best_steered = ""
            for sv in ["15", "10", "5"]:  # Prefer higher positive sv
                sv_data = r.get("steering_results", {}).get(sv, {})
                if sv_data.get("changed", False):
                    best_sv = sv
                    best_steered = sv_data.get("completion", "")
                    break

            if best_sv is None:
                continue

            steered_refusal = _is_refusal(best_steered)

            # Score: prefer non-refusals
            score = 0
            if not baseline_refusal:
                score += 2
            if not steered_refusal:
                score += 2
            if baseline_refusal != steered_refusal:
                score += 1  # Changed refusal status is interesting

            scored_results.append((score, r, best_sv, best_steered))

        # Sort by score descending, take top 10
        scored_results.sort(key=lambda x: -x[0])
        fallback_examples = []
        for score, r, sv, steered in scored_results[:10]:
            fallback_examples.append({
                "prompt": r.get("prompt", ""),
                "baseline_completion": r.get("baseline_completion", ""),
                "steering_results": r.get("steering_results", {}),
                "rationale": r.get("rationale", f"Fallback selection (score={score})"),
                "expected_effect": r.get("expected_effect", "unknown"),
                "test_type": r.get("test_type", "unknown"),
            })

        analysis["illustrative_examples"] = fallback_examples
        print(f"  [IntelligentSteering] Selected {len(fallback_examples)} fallback examples")

    # Aggregate downstream effects across all prompts
    downstream_effects_summary = {}
    if downstream_neurons:
        # Find the most effective steering value for downstream analysis (highest magnitude non-zero)
        effective_sv = max([sv for sv in steering_values if sv != 0], key=abs, default=10)
        sv_key = str(effective_sv)

        for ds_id in downstream_neurons:
            all_changes = []
            for r in steering_results:
                sv_data = r.get("steering_results", {}).get(sv_key, {})
                ds_effects = sv_data.get("downstream_effects", {})
                if ds_id in ds_effects:
                    change = ds_effects[ds_id].get("change_percent")
                    if change is not None:
                        all_changes.append(change)

            if all_changes:
                mean_change = sum(all_changes) / len(all_changes)
                downstream_effects_summary[ds_id] = {
                    "mean_change_percent": mean_change,
                    "n_prompts_measured": len(all_changes),
                    "steering_value_used": effective_sv,
                }

        if downstream_effects_summary:
            print(f"  [IntelligentSteering] Downstream effects (sv={effective_sv}):")
            for ds_id, effects in sorted(downstream_effects_summary.items(),
                                         key=lambda x: abs(x[1]["mean_change_percent"]), reverse=True)[:5]:
                print(f"    {ds_id}: {effects['mean_change_percent']:.1f}% change")

    # Store in protocol state
    state = get_protocol_state()
    state.multi_token_steering_results.append({
        "type": "intelligent_steering",
        "layer": layer,
        "neuron_idx": neuron_idx,
        "output_hypothesis": output_hypothesis,
        "additional_instructions": additional_instructions,
        "n_prompts": len(prompts),
        "steering_values": steering_values,
        "stats_by_steering_value": stats_by_sv,
        "analysis_summary": analysis.get("summary", ""),
        "key_findings": analysis.get("key_findings", []),
        "hypothesis_supported": analysis.get("hypothesis_supported"),
        "effective_steering_range": analysis.get("effective_steering_range"),
        "illustrative_examples": analysis.get("illustrative_examples", []),
        "downstream_neurons_tracked": downstream_neurons or [],
        "downstream_effects_summary": downstream_effects_summary,
    })

    # Update protocol state counters
    current_runs = state.intelligent_steering_runs
    current_total = state.intelligent_steering_total_prompts
    update_protocol_state(
        intelligent_steering_runs=current_runs + 1,
        intelligent_steering_total_prompts=current_total + len(prompts),
    )

    run_num = current_runs + 1
    print(f"  [IntelligentSteering] Complete (run #{run_num}). Hypothesis supported: {analysis.get('hypothesis_supported')}")
    print(f"  [IntelligentSteering] Selected {len(analysis.get('illustrative_examples', []))} illustrative examples")

    return {
        "layer": layer,
        "neuron_idx": neuron_idx,
        "output_hypothesis": output_hypothesis,
        "n_prompts_tested": len(prompts),
        "steering_values": steering_values,
        "stats_by_steering_value": stats_by_sv,
        "analysis": analysis,
        "downstream_neurons_tracked": downstream_neurons or [],
        "downstream_effects_summary": downstream_effects_summary,
    }


def _sync_ablate_upstream_and_test(
    layer: int,
    neuron_idx: int,
    upstream_neurons: list[str],
    test_prompts: list[str],
    window_tokens: int = 10,
    freeze_attention: bool = True,
    freeze_intermediate_mlps: bool = True,
) -> dict[str, Any]:
    """Synchronous implementation of ablate_upstream_and_test.

    Tests if target neuron depends on specific upstream neurons by ablating them
    and measuring the change in target neuron activation.

    Args:
        freeze_attention: If True, cache attention patterns from baseline pass
            and reuse them during ablation passes. This isolates the direct
            MLP pathway effect from attention redistribution effects.
        freeze_intermediate_mlps: If True, also cache MLP outputs for all layers
            between the upstream neuron and target (exclusive). This isolates
            the truly direct pathway and better matches RelP's linearized model.
            Default True because experiments show this produces results that
            align with RelP edge weight predictions.
    """
    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        results = {
            "target_neuron": f"L{layer}/N{neuron_idx}",
            "upstream_neurons": upstream_neurons,
            "freeze_attention": freeze_attention,
            "freeze_intermediate_mlps": freeze_intermediate_mlps,
            "individual_ablation": {},
            "combined_ablation": {},
            "per_prompt_breakdown": [],
        }

        # Parse upstream neurons
        parsed_upstream = []
        for us_id in upstream_neurons:
            try:
                parts = us_id.replace("L", "").replace("N", "").split("/")
                us_layer = int(parts[0])
                us_idx = int(parts[1])
                if us_layer >= layer:
                    print(f"  [Warning] Upstream neuron {us_id} not in earlier layer, skipping")
                    continue
                parsed_upstream.append({"id": us_id, "layer": us_layer, "idx": us_idx})
            except Exception as e:
                print(f"  [Warning] Could not parse upstream neuron {us_id}: {e}")

        if not parsed_upstream:
            return {"error": "No valid upstream neurons to test"}

        # Find the minimum upstream layer (for MLP freezing range)
        min_upstream_layer = min(us["layer"] for us in parsed_upstream)

        for prompt in test_prompts:
            text = format_prompt(prompt)
            inputs = tokenizer(text, return_tensors="pt").to(device)

            prompt_result = {
                "prompt": prompt[:100],
                "baseline_activation": None,
                "individual_effects": {},
                "combined_effect": None,
            }

            # 1. Get baseline activation of target neuron AND upstream neurons
            target_mlp = model.model.layers[layer].mlp
            target_acts_all_positions = []  # Will store full tensor for all positions
            upstream_acts = {us["id"]: [] for us in parsed_upstream}

            # For freezing: cache outputs during baseline
            cached_attn_outputs = {}  # layer_idx -> tensor
            cached_mlp_outputs = {}   # layer_idx -> tensor

            # Capture target activation at ALL positions (not just -1)
            # We'll find the max position after the forward pass
            def capture_target_hook_all_positions(module, input, output):
                hidden = input[0]
                gate = module.gate_proj(hidden)
                up = module.up_proj(hidden)
                intermediate = torch.nn.functional.silu(gate) * up
                # Store activations at all positions for this neuron
                target_acts_all_positions.append(intermediate[0, :, neuron_idx].detach().cpu())

            # Create hooks for each upstream neuron to capture their activations at all positions
            def make_upstream_hook(us_id, us_idx):
                def hook(module, input, output):
                    hidden = input[0]
                    gate = module.gate_proj(hidden)
                    up = module.up_proj(hidden)
                    intermediate = torch.nn.functional.silu(gate) * up
                    # Store all positions so we can extract at max_position later
                    upstream_acts[us_id].append(intermediate[0, :, us_idx].detach().cpu())
                return hook

            # Create hooks to cache attention outputs
            def make_attn_cache_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        cached_attn_outputs[layer_idx] = output[0].clone()
                    else:
                        cached_attn_outputs[layer_idx] = output.clone()
                return hook

            # Create hooks to cache MLP outputs
            def make_mlp_cache_hook(layer_idx):
                def hook(module, input, output):
                    cached_mlp_outputs[layer_idx] = output.clone()
                return hook

            # Register all hooks for baseline pass
            handles = []
            handles.append(target_mlp.register_forward_hook(capture_target_hook_all_positions))
            for us in parsed_upstream:
                us_mlp = model.model.layers[us["layer"]].mlp
                handles.append(us_mlp.register_forward_hook(make_upstream_hook(us["id"], us["idx"])))

            # Cache attention outputs if freezing
            if freeze_attention:
                for layer_idx in range(layer + 1):
                    attn_module = model.model.layers[layer_idx].self_attn
                    handles.append(attn_module.register_forward_hook(make_attn_cache_hook(layer_idx)))

            # Cache intermediate MLP outputs if freezing
            # Cache layers between min_upstream_layer and target (exclusive of both endpoints)
            if freeze_intermediate_mlps:
                for layer_idx in range(min_upstream_layer + 1, layer):
                    mlp_module = model.model.layers[layer_idx].mlp
                    handles.append(mlp_module.register_forward_hook(make_mlp_cache_hook(layer_idx)))

            with torch.no_grad():
                model(**inputs)

            # Remove all hooks
            for h in handles:
                h.remove()

            # Find the position where target neuron fires most strongly
            # This is the key fix - measure at max activation position, not position -1
            if target_acts_all_positions:
                target_tensor = target_acts_all_positions[0]
                max_position = int(target_tensor.argmax().item())
                baseline_act = target_tensor[max_position].item()
            else:
                max_position = -1  # Fallback
                baseline_act = 0

            prompt_result["baseline_activation"] = baseline_act
            prompt_result["max_position"] = max_position  # Store for reference
            # Extract upstream activations at the same max_position
            prompt_result["upstream_activations"] = {
                us_id: acts[0][max_position].item() if acts and len(acts[0]) > max_position else 0
                for us_id, acts in upstream_acts.items()
            }

            # Helper to create attention injection hook
            def make_attn_inject_hook(layer_idx):
                def hook(module, input, output):
                    cached = cached_attn_outputs.get(layer_idx)
                    if cached is not None:
                        if isinstance(output, tuple):
                            return (cached,) + output[1:]
                        else:
                            return cached
                    return output
                return hook

            # Helper to create MLP injection hook
            def make_mlp_inject_hook(layer_idx):
                def hook(module, input, output):
                    cached = cached_mlp_outputs.get(layer_idx)
                    if cached is not None:
                        return cached
                    return output
                return hook

            # 2. Test each upstream neuron individually
            for us in parsed_upstream:
                us_mlp = model.model.layers[us["layer"]].mlp
                original_weight = us_mlp.down_proj.weight[:, us["idx"]].clone()

                ablated_target_acts = []  # Will store all positions
                ablation_handles = []

                # Hook to capture target at all positions during ablation
                def capture_ablated_target_hook(module, input, output):
                    hidden = input[0]
                    gate = module.gate_proj(hidden)
                    up = module.up_proj(hidden)
                    intermediate = torch.nn.functional.silu(gate) * up
                    ablated_target_acts.append(intermediate[0, :, neuron_idx].detach().cpu())

                with torch.no_grad():
                    us_mlp.down_proj.weight.data[:, us["idx"]] = 0
                    ablation_handles.append(target_mlp.register_forward_hook(capture_ablated_target_hook))

                    # Inject cached attention outputs
                    if freeze_attention:
                        for layer_idx in range(layer + 1):
                            attn_module = model.model.layers[layer_idx].self_attn
                            ablation_handles.append(attn_module.register_forward_hook(make_attn_inject_hook(layer_idx)))

                    # Inject cached MLP outputs for layers between this upstream and target
                    if freeze_intermediate_mlps:
                        for layer_idx in range(us["layer"] + 1, layer):
                            mlp_module = model.model.layers[layer_idx].mlp
                            ablation_handles.append(mlp_module.register_forward_hook(make_mlp_inject_hook(layer_idx)))

                    model(**inputs)

                    # Remove all ablation hooks
                    for h in ablation_handles:
                        h.remove()

                    us_mlp.down_proj.weight.data[:, us["idx"]] = original_weight

                # Extract ablated activation at the SAME max_position as baseline
                if ablated_target_acts and max_position < len(ablated_target_acts[0]):
                    ablated_act = ablated_target_acts[0][max_position].item()
                else:
                    ablated_act = 0
                change_pct = 100 * (ablated_act - baseline_act) / (abs(baseline_act) + 1e-6) if baseline_act != 0 else 0

                # Get the upstream neuron's activation for this prompt
                us_activation = prompt_result["upstream_activations"].get(us["id"], 0)

                prompt_result["individual_effects"][us["id"]] = {
                    "upstream_activation": us_activation,
                    "ablated_activation": ablated_act,
                    "change_percent": change_pct,
                    "dependency": abs(change_pct) > 10,
                }

            # 3. Test combined ablation (all upstream neurons at once)
            saved_weights = {}
            for us in parsed_upstream:
                us_mlp = model.model.layers[us["layer"]].mlp
                saved_weights[us["id"]] = us_mlp.down_proj.weight[:, us["idx"]].clone()
                us_mlp.down_proj.weight.data[:, us["idx"]] = 0

            combined_target_acts = []  # Store all positions
            combined_handles = []

            # Hook to capture at all positions for combined ablation
            def capture_combined_target_hook(module, input, output):
                hidden = input[0]
                gate = module.gate_proj(hidden)
                up = module.up_proj(hidden)
                intermediate = torch.nn.functional.silu(gate) * up
                combined_target_acts.append(intermediate[0, :, neuron_idx].detach().cpu())

            combined_handles.append(target_mlp.register_forward_hook(capture_combined_target_hook))

            # Inject cached attention outputs
            if freeze_attention:
                for layer_idx in range(layer + 1):
                    attn_module = model.model.layers[layer_idx].self_attn
                    combined_handles.append(attn_module.register_forward_hook(make_attn_inject_hook(layer_idx)))

            # Inject cached MLP outputs for intermediate layers
            if freeze_intermediate_mlps:
                for layer_idx in range(min_upstream_layer + 1, layer):
                    mlp_module = model.model.layers[layer_idx].mlp
                    combined_handles.append(mlp_module.register_forward_hook(make_mlp_inject_hook(layer_idx)))

            with torch.no_grad():
                model(**inputs)

            for h in combined_handles:
                h.remove()

            # Restore all weights
            for us in parsed_upstream:
                us_mlp = model.model.layers[us["layer"]].mlp
                us_mlp.down_proj.weight.data[:, us["idx"]] = saved_weights[us["id"]]

            # Extract combined activation at the same max_position as baseline
            if combined_target_acts and max_position < len(combined_target_acts[0]):
                combined_act = combined_target_acts[0][max_position].item()
            else:
                combined_act = 0
            combined_change = 100 * (combined_act - baseline_act) / (abs(baseline_act) + 1e-6) if baseline_act != 0 else 0
            prompt_result["combined_effect"] = {
                "ablated_activation": combined_act,
                "change_percent": combined_change,
            }

            # Clear cached outputs to free memory
            cached_attn_outputs.clear()
            cached_mlp_outputs.clear()

            results["per_prompt_breakdown"].append(prompt_result)

        # Aggregate individual effects across prompts
        for us in parsed_upstream:
            changes = [
                p["individual_effects"].get(us["id"], {}).get("change_percent", 0)
                for p in results["per_prompt_breakdown"]
            ]
            upstream_activations = [
                p["individual_effects"].get(us["id"], {}).get("upstream_activation", 0)
                for p in results["per_prompt_breakdown"]
            ]
            mean_change = sum(changes) / len(changes) if changes else 0
            mean_upstream_act = sum(upstream_activations) / len(upstream_activations) if upstream_activations else 0

            # Determine effect type based on ablation result
            # If ablating INCREASES target (positive change), upstream is inhibitory
            # If ablating DECREASES target (negative change), upstream is excitatory
            if mean_change > 10:
                effect_type = "inhibitory"
            elif mean_change < -10:
                effect_type = "excitatory"
            else:
                effect_type = "neutral"

            results["individual_ablation"][us["id"]] = {
                "mean_change_percent": mean_change,
                "dependency_strength": "strong" if abs(mean_change) > 30 else "moderate" if abs(mean_change) > 10 else "weak",
                "mean_upstream_activation": mean_upstream_act,
                "effect_type": effect_type,
                "activation_sign": "positive" if mean_upstream_act > 0.1 else "negative" if mean_upstream_act < -0.1 else "near_zero",
            }

        # Aggregate combined effect
        combined_changes = [
            p["combined_effect"]["change_percent"]
            for p in results["per_prompt_breakdown"]
        ]
        results["combined_ablation"] = {
            "mean_change_percent": sum(combined_changes) / len(combined_changes) if combined_changes else 0,
        }

        return results


def _sync_steer_upstream_and_test(
    layer: int,
    neuron_idx: int,
    upstream_neuron: str,
    test_prompts: list[str],
    steering_values: list[float] | None = None,
    freeze_attention: bool = True,
    freeze_intermediate_mlps: bool = True,
) -> dict[str, Any]:
    """Synchronous implementation of steer_upstream_and_test.

    Steers an upstream neuron at various values and measures the effect on
    target neuron activation. This provides a dose-response curve that can
    be compared to RelP edge weight predictions.

    Args:
        layer: Target neuron layer
        neuron_idx: Target neuron index
        upstream_neuron: Upstream neuron ID (e.g., "L12/N5432")
        test_prompts: Prompts to test
        steering_values: List of steering values (default: [-10, -5, -2, 0, 2, 5, 10])
        freeze_attention: If True (default), cache attention patterns from baseline pass
            and reuse them during steering.
        freeze_intermediate_mlps: If True (default), cache MLP outputs for layers
            between upstream and target. This isolates the direct pathway and
            produces results that align with RelP edge weight predictions.

    Returns:
        Dict with dose-response curve data, slope analysis, and per-prompt breakdown.
    """
    if steering_values is None:
        steering_values = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]

    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        # Parse upstream neuron
        try:
            parts = upstream_neuron.replace("L", "").replace("N", "").split("/")
            us_layer = int(parts[0])
            us_idx = int(parts[1])
            if us_layer >= layer:
                return {"error": f"Upstream neuron {upstream_neuron} must be in earlier layer than target L{layer}"}
        except Exception as e:
            return {"error": f"Could not parse upstream neuron {upstream_neuron}: {e}"}

        results = {
            "target_neuron": f"L{layer}/N{neuron_idx}",
            "upstream_neuron": upstream_neuron,
            "steering_values": steering_values,
            "freeze_attention": freeze_attention,
            "freeze_intermediate_mlps": freeze_intermediate_mlps,
            "dose_response_curve": [],  # Aggregated across prompts
            "per_prompt_breakdown": [],
            "slope_analysis": {},
        }

        # Accumulators for each steering value
        steering_accum = {v: {"target_acts": [], "changes": []} for v in steering_values}

        for prompt in test_prompts:
            text = format_prompt(prompt)
            inputs = tokenizer(text, return_tensors="pt").to(device)

            prompt_result = {
                "prompt": prompt[:100],
                "baseline_activation": None,
                "upstream_baseline_activation": None,
                "steering_effects": {},  # steering_value -> {target_act, change_percent}
            }

            # 1. Get baseline activations (target and upstream) + cache outputs if needed
            target_mlp = model.model.layers[layer].mlp
            us_mlp = model.model.layers[us_layer].mlp
            target_acts_all_positions = []  # Store all positions for target
            upstream_acts_all_positions = []  # Store all positions for upstream
            cached_attn_outputs = {}  # layer_idx -> tensor
            cached_mlp_outputs = {}   # layer_idx -> tensor

            # Capture at ALL positions to find max activation position
            def capture_target_hook_all(module, input, output):
                hidden = input[0]
                gate = module.gate_proj(hidden)
                up = module.up_proj(hidden)
                intermediate = torch.nn.functional.silu(gate) * up
                target_acts_all_positions.append(intermediate[0, :, neuron_idx].detach().cpu())

            def capture_upstream_hook_all(module, input, output):
                hidden = input[0]
                gate = module.gate_proj(hidden)
                up = module.up_proj(hidden)
                intermediate = torch.nn.functional.silu(gate) * up
                upstream_acts_all_positions.append(intermediate[0, :, us_idx].detach().cpu())

            def make_attn_cache_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        cached_attn_outputs[layer_idx] = output[0].clone()
                    else:
                        cached_attn_outputs[layer_idx] = output.clone()
                return hook

            def make_mlp_cache_hook(layer_idx):
                def hook(module, input, output):
                    cached_mlp_outputs[layer_idx] = output.clone()
                return hook

            # Register hooks for baseline
            handles = []
            handles.append(target_mlp.register_forward_hook(capture_target_hook_all))
            handles.append(us_mlp.register_forward_hook(capture_upstream_hook_all))

            if freeze_attention:
                for layer_idx in range(layer + 1):
                    attn_module = model.model.layers[layer_idx].self_attn
                    handles.append(attn_module.register_forward_hook(make_attn_cache_hook(layer_idx)))

            if freeze_intermediate_mlps:
                # Cache MLP outputs for layers between upstream and target (exclusive)
                for layer_idx in range(us_layer + 1, layer):
                    mlp_module = model.model.layers[layer_idx].mlp
                    handles.append(mlp_module.register_forward_hook(make_mlp_cache_hook(layer_idx)))

            with torch.no_grad():
                model(**inputs)

            for h in handles:
                h.remove()

            # Find max position where TARGET fires - this is where we measure and steer
            if target_acts_all_positions:
                target_tensor = target_acts_all_positions[0]
                max_position = int(target_tensor.argmax().item())
                baseline_target_act = target_tensor[max_position].item()
            else:
                max_position = -1
                baseline_target_act = 0

            # Get upstream activation at the same max_position
            if upstream_acts_all_positions and max_position < len(upstream_acts_all_positions[0]):
                baseline_upstream_act = upstream_acts_all_positions[0][max_position].item()
            else:
                baseline_upstream_act = 0

            prompt_result["baseline_activation"] = baseline_target_act
            prompt_result["upstream_baseline_activation"] = baseline_upstream_act
            prompt_result["max_position"] = max_position  # Store for reference

            # Helper for attention injection
            def make_attn_inject_hook(layer_idx):
                def hook(module, input, output):
                    cached = cached_attn_outputs.get(layer_idx)
                    if cached is not None:
                        if isinstance(output, tuple):
                            return (cached,) + output[1:]
                        else:
                            return cached
                    return output
                return hook

            # Helper for MLP injection
            def make_mlp_inject_hook(layer_idx):
                def hook(module, input, output):
                    cached = cached_mlp_outputs.get(layer_idx)
                    if cached is not None:
                        return cached
                    return output
                return hook

            # 2. Test each steering value
            for steer_val in steering_values:
                steered_target_acts = []  # Store all positions
                steer_handles = []

                # Hook to capture target at all positions during steering
                def capture_steered_target_hook(module, input, output):
                    hidden = input[0]
                    gate = module.gate_proj(hidden)
                    up = module.up_proj(hidden)
                    intermediate = torch.nn.functional.silu(gate) * up
                    steered_target_acts.append(intermediate[0, :, neuron_idx].detach().cpu())

                with torch.no_grad():
                    # Apply steering to upstream MLP's activation at MAX_POSITION (where target fires)
                    # We'll use a forward_pre_hook on down_proj to modify its input
                    def make_steer_down_hook(sv, steer_pos):
                        def hook(module, args):
                            x = args[0]  # intermediate activations before down_proj
                            # Shape: (batch, seq_len, intermediate_dim)
                            modified = x.clone()
                            if us_idx < modified.shape[2] and steer_pos < modified.shape[1]:
                                modified[0, steer_pos, us_idx] += sv  # Steer at max_position
                            return (modified,)
                        return hook

                    steer_handles.append(
                        us_mlp.down_proj.register_forward_pre_hook(make_steer_down_hook(steer_val, max_position))
                    )
                    steer_handles.append(target_mlp.register_forward_hook(capture_steered_target_hook))

                    if freeze_attention:
                        for layer_idx in range(layer + 1):
                            attn_module = model.model.layers[layer_idx].self_attn
                            steer_handles.append(attn_module.register_forward_hook(make_attn_inject_hook(layer_idx)))

                    if freeze_intermediate_mlps:
                        for layer_idx in range(us_layer + 1, layer):
                            mlp_module = model.model.layers[layer_idx].mlp
                            steer_handles.append(mlp_module.register_forward_hook(make_mlp_inject_hook(layer_idx)))

                    model(**inputs)

                for h in steer_handles:
                    h.remove()

                # Extract steered activation at the same max_position
                if steered_target_acts and max_position < len(steered_target_acts[0]):
                    steered_act = steered_target_acts[0][max_position].item()
                else:
                    steered_act = 0
                change_pct = 100 * (steered_act - baseline_target_act) / (abs(baseline_target_act) + 1e-6) if baseline_target_act != 0 else 0

                prompt_result["steering_effects"][steer_val] = {
                    "target_activation": steered_act,
                    "change_percent": change_pct,
                }

                steering_accum[steer_val]["target_acts"].append(steered_act)
                steering_accum[steer_val]["changes"].append(change_pct)

            # Clear cached outputs to free memory
            cached_attn_outputs.clear()
            cached_mlp_outputs.clear()

            results["per_prompt_breakdown"].append(prompt_result)

        # Aggregate dose-response curve
        for steer_val in steering_values:
            changes = steering_accum[steer_val]["changes"]
            mean_change = sum(changes) / len(changes) if changes else 0
            results["dose_response_curve"].append({
                "steering_value": steer_val,
                "mean_change_percent": mean_change,
                "std_change_percent": (sum((c - mean_change) ** 2 for c in changes) / len(changes)) ** 0.5 if len(changes) > 1 else 0,
                "n_prompts": len(changes),
            })

        # Compute slope (linear regression of change vs steering value)
        # This can be compared to RelP edge weight
        x_vals = steering_values
        y_vals = [d["mean_change_percent"] for d in results["dose_response_curve"]]

        if len(x_vals) >= 2:
            x_mean = sum(x_vals) / len(x_vals)
            y_mean = sum(y_vals) / len(y_vals)

            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
            denominator = sum((x - x_mean) ** 2 for x in x_vals)

            if denominator > 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean

                # R-squared
                ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_vals, y_vals))
                ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                results["slope_analysis"] = {
                    "slope": slope,  # % change per unit steering
                    "intercept": intercept,
                    "r_squared": r_squared,
                    "is_linear": r_squared > 0.8,
                    "effect_direction": "excitatory" if slope > 0 else "inhibitory" if slope < 0 else "neutral",
                }
            else:
                results["slope_analysis"] = {"error": "Cannot compute slope (zero variance)"}
        else:
            results["slope_analysis"] = {"error": "Need at least 2 steering values"}

        return results


async def tool_steer_upstream_and_test(
    layer: int,
    neuron_idx: int,
    upstream_neuron: str,
    test_prompts: list[str],
    steering_values: list[float] | None = None,
    freeze_attention: bool = True,
    freeze_intermediate_mlps: bool = True,
) -> dict[str, Any]:
    """Steer an upstream neuron and measure effect on target neuron activation.

    This provides a dose-response curve showing how steering the upstream neuron
    affects the target. The slope can be compared to RelP edge weight predictions.

    **RelP Comparison**: RelP uses frozen attention and a linearized model to compute
    edge weights. By default, this tool freezes attention AND intermediate MLPs,
    matching RelP's assumptions and producing results that align with RelP predictions.

    A positive slope means the upstream is excitatory (increasing it increases target),
    negative means inhibitory.

    Args:
        layer: Target neuron layer (0-31)
        neuron_idx: Target neuron index
        upstream_neuron: Upstream neuron ID (e.g., "L12/N5432")
        test_prompts: Prompts to test
        steering_values: List of steering values (default: [-10, -5, -2, 0, 2, 5, 10])
        freeze_attention: If True (default), freeze attention patterns from baseline.
        freeze_intermediate_mlps: If True (default), freeze MLP outputs for layers
            between upstream and target. This isolates the direct pathway and
            produces results that align with RelP edge weight predictions.

    Returns:
        Dict with:
        - dose_response_curve: Mean change % at each steering value
        - slope_analysis: Linear regression (slope comparable to RelP weight)
        - per_prompt_breakdown: Detailed results per prompt
    """
    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        return await client.steer_upstream_and_test(
            layer=layer, neuron_idx=neuron_idx,
            upstream_neuron=upstream_neuron, test_prompts=test_prompts,
            steering_values=steering_values, freeze_attention=freeze_attention,
            freeze_intermediate_mlps=freeze_intermediate_mlps,
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _CUDA_EXECUTOR,
        lambda: _sync_steer_upstream_and_test(
            layer, neuron_idx, upstream_neuron, test_prompts, steering_values,
            freeze_attention, freeze_intermediate_mlps
        ),
    )

    return result


async def tool_ablate_upstream_and_test(
    layer: int,
    neuron_idx: int,
    upstream_neurons: list[str],
    test_prompts: list[str],
    window_tokens: int = 10,
    freeze_attention: bool = True,
    freeze_intermediate_mlps: bool = True,
) -> dict[str, Any]:
    """V4 Input Phase tool: Test if target neuron depends on upstream neurons.

    Ablates upstream neurons individually and combined, measuring change in
    target neuron activation. By default, freezes attention patterns AND
    intermediate MLP outputs to isolate the direct pathway (matching RelP's
    linearized model assumptions).

    Args:
        layer: Target neuron layer
        neuron_idx: Target neuron index
        upstream_neurons: List of upstream neuron IDs (e.g., ["L12/N5432", "L14/N8901"])
        test_prompts: Prompts to test
        window_tokens: Window size for activation capture
        freeze_attention: If True (default), freeze attention patterns from baseline.
        freeze_intermediate_mlps: If True (default), freeze MLP outputs for layers
            between upstream and target. This isolates the direct pathway and
            produces results that align with RelP edge weight predictions.

    Returns:
        Dict with individual ablation results, combined ablation results,
        and per-prompt breakdown.
    """
    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        result = await client.ablate_upstream_and_test(
            layer=layer, neuron_idx=neuron_idx,
            upstream_neurons=upstream_neurons, test_prompts=test_prompts,
            window_tokens=window_tokens, freeze_attention=freeze_attention,
            freeze_intermediate_mlps=freeze_intermediate_mlps,
        )
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _CUDA_EXECUTOR,
            lambda: _sync_ablate_upstream_and_test(
                layer, neuron_idx, upstream_neurons, test_prompts, window_tokens,
                freeze_attention, freeze_intermediate_mlps
            ),
        )

    # Update protocol state and store result
    state = get_protocol_state()
    state.upstream_dependency_results.append({
        "target_layer": layer,
        "target_neuron": neuron_idx,
        "upstream_neurons": upstream_neurons,
        "test_prompts": test_prompts,
        **result,
    })
    update_protocol_state(upstream_dependency_tested=True)
    print("[PROTOCOL] Upstream dependency test complete")

    return result


async def tool_batch_ablate_upstream_and_test(
    use_categorized_prompts: bool = True,
    activation_threshold: float = 0.5,
    upstream_neurons: list[str] | None = None,
    max_prompts: int = 100,
    window_tokens: int = 10,
    freeze_attention: bool = True,
    freeze_intermediate_mlps: bool = True,
) -> dict[str, Any]:
    """Batch test upstream dependencies using prompts from category selectivity.

    Similar to batch_ablate_and_generate but for upstream testing.
    Uses activating prompts from category selectivity to test whether
    ablating upstream neurons reduces target neuron activation.

    By default, freezes attention AND intermediate MLP outputs to isolate
    the direct pathway from upstream to target (matching RelP's linearized
    model assumptions).

    Args:
        use_categorized_prompts: Use prompts from run_category_selectivity_test (recommended)
        activation_threshold: Minimum activation to include prompt
        upstream_neurons: Upstream neurons to test. If None, auto-detected from analyze_wiring
        max_prompts: Maximum prompts to test
        window_tokens: Number of tokens to check for activation
        freeze_attention: If True (default), freeze attention patterns from baseline.
        freeze_intermediate_mlps: If True (default), freeze MLP outputs for layers
            between upstream and target. This isolates the direct pathway and
            produces results that align with RelP edge weight predictions.

    Returns:
        Dict with:
        - total_prompts: Number tested
        - upstream_results: Per-upstream neuron dependency statistics
        - category_breakdown: How dependencies vary by category
    """
    state = get_protocol_state()

    # Get prompts from category selectivity
    if use_categorized_prompts:
        if not state.categorized_prompts:
            return {"error": "No categorized prompts available. Run run_category_selectivity_test first."}

        # Filter to activating prompts and sort by activation strength
        activating_prompts_with_act = []
        prompt_categories = {}
        for category, prompts in state.categorized_prompts.items():
            for p in prompts:
                if isinstance(p, dict):
                    act = p.get("activation", 0)
                    prompt_text = p.get("prompt", "")
                else:
                    act = 0
                    prompt_text = str(p)
                if act >= activation_threshold:
                    activating_prompts_with_act.append((prompt_text, act))
                    prompt_categories[prompt_text] = category

        if not activating_prompts_with_act:
            return {"error": f"No prompts with activation >= {activation_threshold}"}

        # Sort by activation (highest first) to get most reliable test prompts
        activating_prompts_with_act.sort(key=lambda x: x[1], reverse=True)

        # Limit prompts
        if len(activating_prompts_with_act) > max_prompts:
            activating_prompts_with_act = activating_prompts_with_act[:max_prompts]

        test_prompts = [p[0] for p in activating_prompts_with_act]
        print(f"[BatchUpstream] Selected {len(test_prompts)} prompts (activation range: {activating_prompts_with_act[-1][1]:.2f} - {activating_prompts_with_act[0][1]:.2f})")
    else:
        return {"error": "use_categorized_prompts=False not supported. Run category_selectivity first."}

    # ALWAYS use ALL connectivity neurons for consistency with circuit diagram
    # (ignore any explicit list the agent may have passed)
    if not state.connectivity_analyzed:
        return {"error": "Connectivity not analyzed. Call analyze_wiring first."}
    connectivity = state.connectivity_data or {}
    upstream_list = connectivity.get("upstream_neurons", [])
    if not upstream_list:
        return {"error": "No upstream neurons found in connectivity data."}
    # Use ALL connectivity neurons - ensures batch results match circuit diagram
    upstream_neurons = [u.get("neuron_id") for u in upstream_list if u.get("neuron_id")]

    if not upstream_neurons:
        return {"error": "No upstream neurons to test."}

    # Parse target neuron from state
    target_layer = state.target_layer
    target_neuron = state.target_neuron

    print(f"[BatchUpstream] Testing ALL {len(upstream_neurons)} upstream neurons from connectivity across {len(test_prompts)} prompts")

    # Run the upstream ablation test
    result = await tool_ablate_upstream_and_test(
        layer=target_layer,
        neuron_idx=target_neuron,
        upstream_neurons=upstream_neurons,
        test_prompts=test_prompts,
        window_tokens=window_tokens,
        freeze_attention=freeze_attention,
        freeze_intermediate_mlps=freeze_intermediate_mlps,
    )

    # Add per-neuron category breakdown if we have category info
    # Note: _sync_ablate_upstream_and_test returns "per_prompt_breakdown" with "individual_effects"
    if use_categorized_prompts and "per_prompt_breakdown" in result:
        # Build per-upstream-neuron category effects
        # Structure: {upstream_id: {category: [change_pcts]}}
        per_neuron_category_effects = {}

        for pr in result.get("per_prompt_breakdown", []):
            prompt = pr.get("prompt", "")
            cat = prompt_categories.get(prompt, "unknown")
            individual_effects = pr.get("individual_effects", {})

            for upstream_id, effect in individual_effects.items():
                if upstream_id not in per_neuron_category_effects:
                    per_neuron_category_effects[upstream_id] = {}
                if cat not in per_neuron_category_effects[upstream_id]:
                    per_neuron_category_effects[upstream_id][cat] = []
                per_neuron_category_effects[upstream_id][cat].append(effect.get("change_percent", 0))

        # Compute average per category per neuron and add to individual_ablation
        for upstream_id, cat_effects in per_neuron_category_effects.items():
            category_breakdown = {}
            for cat, changes in cat_effects.items():
                avg_change = sum(changes) / len(changes) if changes else 0
                effect_type = "inhibitory" if avg_change > 10 else "excitatory" if avg_change < -10 else "neutral"
                category_breakdown[cat] = {
                    "avg_change_percent": avg_change,
                    "effect_type": effect_type,
                    "n_prompts": len(changes),
                }

            # Add to individual_ablation results
            if upstream_id in result.get("individual_ablation", {}):
                result["individual_ablation"][upstream_id]["category_breakdown"] = category_breakdown

        # Also add overall category summary (averaged across all neurons)
        overall_category = {}
        for cat in set(c for effects in per_neuron_category_effects.values() for c in effects):
            all_changes = []
            for effects in per_neuron_category_effects.values():
                all_changes.extend(effects.get(cat, []))
            if all_changes:
                overall_category[cat] = sum(all_changes) / len(all_changes)
        result["overall_category_effects"] = overall_category

    result["total_prompts"] = len(test_prompts)
    print(f"[BatchUpstream] Complete: tested {len(upstream_neurons)} upstream neurons")

    return result


async def tool_batch_steer_upstream_and_test(
    use_categorized_prompts: bool = True,
    activation_threshold: float = 0.5,
    upstream_neurons: list[str] | None = None,
    max_prompts: int = 100,
    steering_values: list[float] | None = None,
    freeze_attention: bool = True,
    freeze_intermediate_mlps: bool = True,
) -> dict[str, Any]:
    """Batch test upstream steering using prompts from category selectivity.

    Runs steering dose-response curves for multiple upstream neurons across
    many prompts. Computes slopes that can be compared to RelP edge weights.

    By default, freezes attention AND intermediate MLP outputs to isolate
    the direct pathway from upstream to target (matching RelP's linearized
    model assumptions).

    Args:
        use_categorized_prompts: Use prompts from run_category_selectivity_test (recommended)
        activation_threshold: Minimum activation to include prompt
        upstream_neurons: Upstream neurons to test. If None, auto-detected from analyze_wiring
        max_prompts: Maximum prompts to test
        steering_values: List of steering values (default: [-10, -5, 0, 5, 10])
        freeze_attention: If True (default), freeze attention patterns from baseline.
        freeze_intermediate_mlps: If True (default), freeze MLP outputs for layers
            between upstream and target.

    Returns:
        Dict with:
        - total_prompts: Number tested
        - upstream_results: Per-upstream neuron steering slopes and RelP comparison
        - category_breakdown: How steering effects vary by category
    """
    if steering_values is None:
        steering_values = [-10.0, -5.0, 0.0, 5.0, 10.0]

    state = get_protocol_state()

    # Get prompts from category selectivity
    if use_categorized_prompts:
        if not state.categorized_prompts:
            return {"error": "No categorized prompts available. Run run_category_selectivity_test first."}

        # Filter to activating prompts and sort by activation strength
        activating_prompts_with_act = []
        prompt_categories = {}
        for category, prompts in state.categorized_prompts.items():
            for p in prompts:
                if isinstance(p, dict):
                    act = p.get("activation", 0)
                    prompt_text = p.get("prompt", "")
                else:
                    act = 0
                    prompt_text = str(p)
                if act >= activation_threshold:
                    activating_prompts_with_act.append((prompt_text, act))
                    prompt_categories[prompt_text] = category

        if not activating_prompts_with_act:
            return {"error": f"No prompts with activation >= {activation_threshold}"}

        # Sort by activation (highest first)
        activating_prompts_with_act.sort(key=lambda x: x[1], reverse=True)

        # Limit prompts
        if len(activating_prompts_with_act) > max_prompts:
            activating_prompts_with_act = activating_prompts_with_act[:max_prompts]

        test_prompts = [p[0] for p in activating_prompts_with_act]
        print(f"[BatchSteerUpstream] Selected {len(test_prompts)} prompts (activation range: {activating_prompts_with_act[-1][1]:.2f} - {activating_prompts_with_act[0][1]:.2f})")
    else:
        return {"error": "use_categorized_prompts=False not supported. Run category_selectivity first."}

    # ALWAYS use ALL connectivity neurons for consistency with circuit diagram
    # (ignore any explicit list the agent may have passed)
    if not state.connectivity_analyzed:
        return {"error": "Connectivity not analyzed. Call analyze_wiring first."}
    connectivity = state.connectivity_data or {}
    upstream_list = connectivity.get("upstream_neurons", [])
    if not upstream_list:
        return {"error": "No upstream neurons found in connectivity data."}
    # Use ALL connectivity neurons - ensures batch results match circuit diagram
    upstream_neurons = [u.get("neuron_id") for u in upstream_list if u.get("neuron_id")]

    if not upstream_neurons:
        return {"error": "No upstream neurons to test."}

    # Get RelP weights for comparison
    connectivity = state.connectivity_data or {}
    relp_weights = {}
    for u in connectivity.get("upstream_neurons", []):
        nid = u.get("neuron_id")
        if nid:
            relp_weights[nid] = u.get("weight", 0)

    # Parse target neuron from state
    target_layer = state.target_layer
    target_neuron = state.target_neuron

    print(f"[BatchSteerUpstream] Testing ALL {len(upstream_neurons)} upstream neurons from connectivity across {len(test_prompts)} prompts")

    # Run steering for each upstream neuron
    results = {
        "target_neuron": f"L{target_layer}/N{target_neuron}",
        "freeze_attention": freeze_attention,
        "freeze_intermediate_mlps": freeze_intermediate_mlps,
        "steering_values": steering_values,
        "upstream_results": {},
        "total_prompts": len(test_prompts),
        "relp_comparison": {},
    }

    for us_id in upstream_neurons:
        print(f"  Steering {us_id}...", end="", flush=True)

        steer_result = await tool_steer_upstream_and_test(
            layer=target_layer,
            neuron_idx=target_neuron,
            upstream_neuron=us_id,
            test_prompts=test_prompts,
            steering_values=steering_values,
            freeze_attention=freeze_attention,
            freeze_intermediate_mlps=freeze_intermediate_mlps,
        )

        if "error" in steer_result:
            print(f" error: {steer_result['error']}")
            results["upstream_results"][us_id] = {"error": steer_result["error"]}
            continue

        slope_info = steer_result.get("slope_analysis", {})
        slope = slope_info.get("slope", 0)
        r_squared = slope_info.get("r_squared", 0)
        effect_direction = slope_info.get("effect_direction", "unknown")

        # Compare to RelP weight
        relp_wt = relp_weights.get(us_id, 0)
        relp_sign = "excitatory" if relp_wt > 0 else "inhibitory" if relp_wt < 0 else "neutral"
        signs_match = (slope > 0 and relp_wt > 0) or (slope < 0 and relp_wt < 0) or (slope == 0 and relp_wt == 0)

        results["upstream_results"][us_id] = {
            "slope": slope,
            "r_squared": r_squared,
            "effect_direction": effect_direction,
            "dose_response_curve": steer_result.get("dose_response_curve", []),
        }

        results["relp_comparison"][us_id] = {
            "relp_weight": relp_wt,
            "relp_sign": relp_sign,
            "steering_slope": slope,
            "steering_sign": effect_direction,
            "signs_match": signs_match,
        }

        print(f" slope={slope:.4f}, R²={r_squared:.3f}, RelP match={'✓' if signs_match else '✗'}")

    # Compute per-category breakdown (steering effect by category)
    if use_categorized_prompts:
        # We need to re-aggregate by category from per-prompt results
        # For simplicity, compute average slope per category
        # This would require running steering per category, which is expensive
        # Instead, just note category distribution
        category_counts = {}
        for prompt in test_prompts:
            cat = prompt_categories.get(prompt, "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        results["category_distribution"] = category_counts

    # Summary statistics
    n_match = sum(1 for r in results["relp_comparison"].values() if r.get("signs_match"))
    n_total = len(results["relp_comparison"])
    summary = {
        "relp_sign_agreement": f"{n_match}/{n_total} ({100*n_match/n_total:.0f}%)" if n_total > 0 else "N/A",
        "upstream_neurons_tested": len(upstream_neurons),
        "prompts_tested": len(test_prompts),
    }

    # Add regime context to summary
    if state.operating_regime:
        summary["operating_regime"] = state.operating_regime
        if state.wiring_data and state.wiring_data.get("stats", {}).get("regime_correction_applied"):
            summary["regime_correction_applied"] = True
            summary["regime_note"] = (
                "Wiring polarity was regime-corrected (target operates in inverted SwiGLU regime). "
                "Sign agreement reflects corrected polarities."
            )
    elif state.operating_regime is None:
        summary["regime_warning"] = (
            "Operating regime not yet detected. Run category selectivity first for accurate polarity comparison."
        )

    results["summary"] = summary
    print(f"[BatchSteerUpstream] Complete: {n_match}/{n_total} match RelP sign")

    # Store results in protocol state
    state.upstream_steering_results.append({
        "target_neuron": f"L{target_layer}/N{target_neuron}",
        "upstream_neurons": upstream_neurons,
        "total_prompts": len(test_prompts),
        "freeze_attention": freeze_attention,
        "freeze_intermediate_mlps": freeze_intermediate_mlps,
        "steering_values": steering_values,
        "upstream_results": results.get("upstream_results", {}),
        "relp_comparison": results.get("relp_comparison", {}),
        "relp_sign_agreement": results.get("summary", {}).get("relp_sign_agreement", "N/A"),
    })

    # Update protocol state flag
    update_protocol_state(upstream_steering_tested=True)
    print("[PROTOCOL] Upstream steering test complete")

    return results


def _sync_ablate_and_check_downstream(
    layer: int,
    neuron_idx: int,
    prompts: list[str],
    downstream_neurons: list[str] | None,
    max_new_tokens: int = 10,
) -> dict[str, Any]:
    """Synchronous implementation of ablate_and_check_downstream.

    Ablates target neuron and checks how downstream neuron activations change
    across MULTIPLE generated token positions.

    This generates tokens and measures downstream effects at each position,
    not just at the last input position.
    """
    with _MODEL_LOCK:
        model, tokenizer = get_model_and_tokenizer()
        device = next(model.parameters()).device

        results = {
            "target_neuron": f"L{layer}/N{neuron_idx}",
            "downstream_neurons": downstream_neurons or [],
            "max_new_tokens": max_new_tokens,
            "per_prompt_results": [],
            "dependency_summary": {},
        }

        # If no downstream neurons specified, we can't test dependencies
        if not downstream_neurons:
            return {"error": "No downstream neurons specified. Use analyze_output_wiring to find them."}

        # Parse downstream neurons
        parsed_downstream = []
        for ds_id in downstream_neurons[:10]:  # Limit to 10
            try:
                parts = ds_id.replace("L", "").replace("N", "").split("/")
                ds_layer = int(parts[0])
                ds_idx = int(parts[1])
                if ds_layer <= layer:
                    print(f"  [Warning] Downstream neuron {ds_id} not in later layer, skipping")
                    continue
                parsed_downstream.append({"id": ds_id, "layer": ds_layer, "idx": ds_idx})
            except Exception as e:
                print(f"  [Warning] Could not parse downstream neuron {ds_id}: {e}")

        if not parsed_downstream:
            return {"error": "No valid downstream neurons to check (must be in later layers)"}

        target_mlp = model.model.layers[layer].mlp
        original_weight = target_mlp.down_proj.weight[:, neuron_idx].clone()

        for prompt in prompts:
            text = format_prompt(prompt)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            prompt_result = {
                "prompt": prompt[:100],
                "downstream_effects": {},
                "per_position_effects": {},
            }

            # Generate tokens with baseline and ablated, capturing downstream activations at each step
            for ds in parsed_downstream:
                ds_mlp = model.model.layers[ds["layer"]].mlp

                # Storage for per-position activations
                baseline_acts_per_pos = []
                ablated_acts_per_pos = []

                # --- Baseline generation with downstream capture ---
                captured_baseline = []

                def make_baseline_hook():
                    def hook(module, input, output):
                        hidden = input[0]
                        gate = module.gate_proj(hidden)
                        up = module.up_proj(hidden)
                        intermediate = torch.nn.functional.silu(gate) * up
                        # Capture activation at last position (the newly generated token)
                        captured_baseline.append(intermediate[0, -1, ds["idx"]].item())
                    return hook

                handle = ds_mlp.register_forward_hook(make_baseline_hook())
                with torch.no_grad():
                    baseline_outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                handle.remove()
                baseline_acts_per_pos = captured_baseline.copy()

                # --- Ablated generation with downstream capture ---
                captured_ablated = []

                def make_ablated_hook():
                    def hook(module, input, output):
                        hidden = input[0]
                        gate = module.gate_proj(hidden)
                        up = module.up_proj(hidden)
                        intermediate = torch.nn.functional.silu(gate) * up
                        captured_ablated.append(intermediate[0, -1, ds["idx"]].item())
                    return hook

                with torch.no_grad():
                    target_mlp.down_proj.weight[:, neuron_idx] = 0
                    handle = ds_mlp.register_forward_hook(make_ablated_hook())
                    ablated_outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    handle.remove()
                    target_mlp.down_proj.weight[:, neuron_idx] = original_weight
                ablated_acts_per_pos = captured_ablated.copy()

                # Compute per-position changes
                # Note: generation may produce different number of tokens, align by min length
                n_positions = min(len(baseline_acts_per_pos), len(ablated_acts_per_pos), max_new_tokens)
                per_pos_changes = []

                # Find position with max baseline activation (where downstream neuron actually fires)
                max_baseline_pos = -1
                max_baseline_val = 0
                for pos in range(n_positions):
                    baseline_val = baseline_acts_per_pos[pos] if pos < len(baseline_acts_per_pos) else 0
                    if abs(baseline_val) > abs(max_baseline_val):
                        max_baseline_val = baseline_val
                        max_baseline_pos = pos

                for pos in range(n_positions):
                    baseline_val = baseline_acts_per_pos[pos] if pos < len(baseline_acts_per_pos) else 0
                    ablated_val = ablated_acts_per_pos[pos] if pos < len(ablated_acts_per_pos) else 0

                    # Only compute meaningful percentage if baseline has significant activation
                    # Use threshold of 0.1 to avoid division-by-small-number artifacts
                    if abs(baseline_val) >= 0.1:
                        change_pct = 100 * (ablated_val - baseline_val) / abs(baseline_val)
                    else:
                        # For low baseline, report absolute change scaled to typical activation range
                        # This avoids -700% artifacts from dividing by 0.01
                        change_pct = 0  # Will use absolute change instead

                    per_pos_changes.append({
                        "position": pos,
                        "baseline": baseline_val,
                        "ablated": ablated_val,
                        "change_percent": change_pct,
                        "absolute_change": ablated_val - baseline_val,
                    })

                # Compute aggregate statistics - focus on position with meaningful baseline
                # This mirrors the upstream fix: measure at position where neuron fires
                if per_pos_changes and max_baseline_pos >= 0 and abs(max_baseline_val) >= 0.1:
                    # Use the change at the max activation position (where downstream neuron fires)
                    max_pos_data = per_pos_changes[max_baseline_pos]
                    mean_change = max_pos_data["change_percent"]
                    max_change = max_pos_data["change_percent"]
                elif per_pos_changes:
                    # Fallback: only consider positions with significant baseline activation
                    significant_changes = [p["change_percent"] for p in per_pos_changes if abs(p["baseline"]) >= 0.1]
                    if significant_changes:
                        mean_change = sum(significant_changes) / len(significant_changes)
                        max_change = max(significant_changes, key=abs)
                    else:
                        # No significant baseline activations - use absolute change sum
                        abs_changes = [p["absolute_change"] for p in per_pos_changes]
                        mean_change = sum(abs_changes) / len(abs_changes) if abs_changes else 0
                        max_change = max(abs_changes, key=abs) if abs_changes else 0
                else:
                    mean_change = 0
                    max_change = 0

                prompt_result["downstream_effects"][ds["id"]] = {
                    "mean_change_percent": mean_change,
                    "max_change_percent": max_change,
                    "n_positions_measured": n_positions,
                }
                prompt_result["per_position_effects"][ds["id"]] = per_pos_changes

            results["per_prompt_results"].append(prompt_result)

        # Aggregate dependency summary across all prompts
        for ds in parsed_downstream:
            all_mean_changes = [
                p["downstream_effects"].get(ds["id"], {}).get("mean_change_percent", 0)
                for p in results["per_prompt_results"]
            ]
            overall_mean = sum(all_mean_changes) / len(all_mean_changes) if all_mean_changes else 0
            results["dependency_summary"][ds["id"]] = {
                "mean_change_percent": overall_mean,
                "dependency_strength": "strong" if abs(overall_mean) > 30 else "moderate" if abs(overall_mean) > 10 else "weak",
            }

        return results


async def tool_ablate_and_check_downstream(
    layer: int,
    neuron_idx: int,
    prompts: list[str],
    downstream_neurons: list[str] | None = None,
    max_new_tokens: int = 10,
) -> dict[str, Any]:
    """DEPRECATED: Use batch_ablate_and_generate instead.

    batch_ablate_and_generate now includes downstream checking at all positions
    AND returns completions. This function only returns downstream effects.

    Example migration:
        # Old:
        result = await ablate_and_check_downstream(layer, neuron, prompts)

        # New:
        result = await batch_ablate_and_generate(layer, neuron, prompts=prompts)
        # downstream_neurons auto-populated from connectivity
        # Result includes both completions AND dependency_summary

    Args:
        layer: Target neuron layer
        neuron_idx: Target neuron index
        prompts: Prompts to test
        downstream_neurons: List of downstream neuron IDs (auto from connectivity if None)
        max_new_tokens: Number of tokens to generate (default 10). Downstream effects
            are measured at each generated position. Recommended: 10 for thorough
            analysis, 5 for quick checks.

    Returns:
        Dict with:
        - per_prompt_results: Per-prompt breakdown with per_position_effects
        - dependency_summary: Aggregated dependency strength per downstream neuron
    """
    import warnings
    warnings.warn(
        "ablate_and_check_downstream is deprecated. Use batch_ablate_and_generate instead, "
        "which returns both completions AND downstream effects.",
        DeprecationWarning,
        stacklevel=2
    )
    print("[DEPRECATED] ablate_and_check_downstream: Use batch_ablate_and_generate for combined completions + downstream checking")
    # ALWAYS use ALL connectivity neurons for consistency with circuit diagram
    # (ignore any explicit list the agent may have passed)
    state = get_protocol_state()
    if not state.connectivity_analyzed:
        return {"error": "Connectivity not analyzed. Call analyze_output_wiring first."}
    connectivity = state.connectivity_data or {}
    downstream_list = connectivity.get("downstream_targets", [])
    # Filter to actual neurons (not LOGIT targets)
    downstream_neurons = [
        d.get("neuron_id") for d in downstream_list
        if d.get("neuron_id") and not d.get("target", "").startswith("LOGIT")
    ]
    if not downstream_neurons:
        return {"error": "No downstream neurons found in connectivity data (only LOGIT targets)."}
    print(f"[DownstreamCheck] Testing ALL {len(downstream_neurons)} downstream neurons from connectivity")

    # Remote GPU dispatch
    client = get_gpu_client()
    if client is not None:
        result = await client.ablate_and_check_downstream(
            layer=layer, neuron_idx=neuron_idx, prompts=prompts,
            downstream_neurons=downstream_neurons, max_new_tokens=max_new_tokens,
        )
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _CUDA_EXECUTOR,
            _sync_ablate_and_check_downstream,
            layer,
            neuron_idx,
            prompts,
            downstream_neurons,
            max_new_tokens,
        )

    # Update protocol state and store result
    if "error" not in result:
        state = get_protocol_state()
        state.downstream_dependency_results.append({
            "target_layer": layer,
            "target_neuron": neuron_idx,
            "prompts": prompts,
            "downstream_neurons": downstream_neurons,
            **result,
        })
        update_protocol_state(
            downstream_dependency_tested=True,
            downstream_dependency_prompt_count=len(prompts)
        )
        print(f"[PROTOCOL] Downstream dependency check complete ({len(prompts)} prompts)")

    return result


async def tool_complete_input_phase(
    summary: str,
    triggers: list[str],
    confidence: float,
    upstream_dependencies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """V4 Phase Completion tool: Mark input phase as complete.

    Validates that all input phase requirements are met and stores the
    input characterization.

    Args:
        summary: Summary of what triggers the neuron
        triggers: List of identified triggers (patterns, tokens, concepts)
        confidence: Confidence level (0.0-1.0)
        upstream_dependencies: Optional list of upstream dependency results

    Returns:
        Dict with status and any missing requirements.
    """
    state = get_protocol_state()

    # Check requirements
    missing = state.get_input_phase_missing()
    if missing:
        return {
            "status": "blocked",
            "message": "Cannot complete input phase - requirements missing",
            "missing_requirements": missing,
        }

    # Store input characterization
    state.input_characterization = {
        "summary": summary,
        "triggers": triggers,
        "upstream_dependencies": upstream_dependencies or [],
        "selectivity_data": {},  # Populated from category_selectivity_test
        "confidence": confidence,
    }

    # Mark phase complete
    state.input_phase_complete = True
    print(f"[PROTOCOL] INPUT PHASE COMPLETE: {summary[:50]}...")

    return {
        "status": "complete",
        "message": "Input phase completed successfully",
        "input_characterization": state.input_characterization,
        "can_start_output_phase": True,
    }


async def tool_complete_output_phase(
    summary: str,
    promotes: list[str],
    suppresses: list[str],
    confidence: float,
) -> dict[str, Any]:
    """V4 Phase Completion tool: Mark output phase as complete.

    Validates that all output phase requirements are met and stores the
    output characterization.

    Args:
        summary: Summary of what the neuron does when active
        promotes: List of tokens/concepts the neuron promotes
        suppresses: List of tokens/concepts the neuron suppresses
        confidence: Confidence level (0.0-1.0)

    Returns:
        Dict with status and any missing requirements.
    """
    state = get_protocol_state()

    # Check requirements
    missing = state.get_output_phase_missing()

    # Also check downstream dependency if downstream neurons exist
    # This is a V4 requirement for completeness
    if state.downstream_neurons_exist and not state.downstream_dependency_tested:
        missing.append("Downstream dependency not tested (REQUIRED when downstream neurons exist). Call ablate_and_check_downstream()")

    if missing:
        return {
            "status": "blocked",
            "message": "Cannot complete output phase - requirements missing",
            "missing_requirements": missing,
        }

    # Store output characterization
    state.output_characterization = {
        "summary": summary,
        "promotes": promotes,
        "suppresses": suppresses,
        "multi_token_ablation": [],  # Will be populated from results
        "multi_token_steering": [],
        "downstream_dependencies": [],
        "relp_evidence": [],
        "confidence": confidence,
    }

    # Mark phase complete
    state.output_phase_complete = True
    print(f"[PROTOCOL] OUTPUT PHASE COMPLETE: {summary[:50]}...")

    return {
        "status": "complete",
        "message": "Output phase completed successfully",
        "output_characterization": state.output_characterization,
        "ready_for_anomaly_phase": True,
    }


async def tool_complete_anomaly_phase(
    anomalies_identified: list[str],
    anomalies_investigated: list[dict[str, Any]],
) -> dict[str, Any]:
    """V5 Phase Completion tool: Mark anomaly investigation phase as complete.

    This phase requires the agent to:
    1. Identify anomalies from accumulated evidence
    2. Investigate the top 3 most interesting anomalies
    3. Document findings and explanations

    Args:
        anomalies_identified: List of all anomalies identified from evidence
            (e.g., ["Wiring-ablation mismatch for L16/N9747", "Unexpected activation on 'risotto' prompt"])
        anomalies_investigated: List of investigated anomalies with details:
            [
                {
                    "anomaly": "Description of the anomaly",
                    "explanation": "What was learned about this anomaly",
                    "experiments_run": ["tool_name1", "tool_name2"],
                    "confidence": 0.8  # How confident in the explanation
                },
                ...
            ]

    Returns:
        Dict with status and stored anomaly investigation results.
    """
    state = get_protocol_state()

    # Require output phase to be complete first
    if not state.output_phase_complete:
        return {
            "status": "blocked",
            "message": "Cannot complete anomaly phase - output phase not complete",
            "missing_requirements": ["output_phase_complete"],
        }

    # Validate inputs
    if not anomalies_identified:
        return {
            "status": "blocked",
            "message": "Must identify at least one anomaly from evidence (or explicitly state no anomalies found)",
            "hint": "Review all accumulated evidence for: contradictions, surprises, unexplained patterns, hypothesis mismatches",
        }

    if len(anomalies_investigated) < min(3, len(anomalies_identified)):
        expected = min(3, len(anomalies_identified))
        return {
            "status": "blocked",
            "message": f"Must investigate at least {expected} anomalies (you investigated {len(anomalies_investigated)})",
            "anomalies_identified": anomalies_identified,
            "hint": "Select the most interesting anomalies and run targeted experiments to understand them",
        }

    # Validate each investigated anomaly has required fields
    for i, investigation in enumerate(anomalies_investigated):
        if not investigation.get("anomaly"):
            return {
                "status": "blocked",
                "message": f"Investigation {i+1} missing 'anomaly' field",
            }
        if not investigation.get("explanation"):
            return {
                "status": "blocked",
                "message": f"Investigation {i+1} missing 'explanation' field - what did you learn?",
            }
        if not investigation.get("experiments_run"):
            return {
                "status": "blocked",
                "message": f"Investigation {i+1} missing 'experiments_run' field - what tools did you use?",
            }

    # Store anomaly investigation results
    state.anomalies_identified = anomalies_identified
    state.anomalies_investigated = anomalies_investigated

    # Mark phase complete
    state.anomaly_phase_complete = True
    print(f"[PROTOCOL] ANOMALY PHASE COMPLETE: {len(anomalies_investigated)} anomalies investigated")

    return {
        "status": "complete",
        "message": "Anomaly investigation phase completed successfully",
        "anomalies_identified": len(anomalies_identified),
        "anomalies_investigated": len(anomalies_investigated),
        "investigations": anomalies_investigated,
        "ready_for_report": True,
    }
