"""Progressive two-pass auto-interpretation for neural network neurons.

Interprets neurons layer-by-layer using two passes:
  - Pass 1 (late-to-early): What does each unit DO? (output function labels)
  - Pass 2 (early-to-late): What TRIGGERS each unit? (input descriptions)

Core types:
  - TokenEffect: A token promoted/suppressed by a neuron
  - NeuronLink: A connection to another neuron with optional function label
  - NeuronFunction: Complete functional description of a neuron
  - NeuronFunctionDB: JSON-backed store of NeuronFunction objects

Main engine:
  - ProgressiveInterpreter: Two-pass labeling engine using model weights + LLM

Bridge functions:
  - neuron_function_to_unit: Convert NeuronFunction -> Unit (circuits.schemas)
  - unit_to_neuron_function: Convert Unit -> NeuronFunction

Usage:
    from circuits.autointerp import ProgressiveInterpreter, NeuronFunctionDB

    interp = ProgressiveInterpreter(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        edge_stats_path=Path("data/edge_stats.json"),
        db_path=Path("data/neuron_function_db.json"),
    )
    interp.process_all_connections(min_appearances=100)
    interp.llm_label_all_layers(start_layer=31, end_layer=0, passes=2)
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from circuits.schemas import LabelSource, TokenProjection, Unit, UnitLabel

# Optional: OpenAI for LLM-assisted labeling
try:
    from openai import AsyncOpenAI, OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TokenEffect:
    """A token promoted or suppressed by a neuron."""

    token: str
    token_id: int
    logit_contribution: float


@dataclass
class NeuronLink:
    """A connection to another neuron."""

    neuron_id: str
    layer: int
    neuron_idx: int
    weight: float
    frequency: float
    function_label: str | None = None  # Filled in during progressive interp


@dataclass
class NeuronFunction:
    """Complete functional description of a neuron."""

    neuron_id: str
    layer: int
    neuron_idx: int

    # Function description (OUTPUT - what firing does)
    function_label: str = ""  # Short label like "article->possessive switch"
    function_description: str = ""  # Longer description
    confidence: str = "unknown"  # low, medium, high
    interpretability: str = "unknown"  # low, medium, high
    function_type: str = ""  # semantic, formatting, structural, syntactic, etc.

    # Input description (INPUT - what makes it fire)
    input_label: str = ""  # Short label like "medical-context-detector"
    input_description: str = ""  # Longer description of activation conditions
    input_type: str = ""  # token-pattern, context, upstream-gated, etc.
    input_interpretability: str = "unknown"  # low, medium, high
    input_confidence: str = "unknown"  # llm-auto, human-reviewed

    # INPUT: What makes this fire
    activation_patterns: list[str] = field(default_factory=list)
    upstream_neurons: list[NeuronLink] = field(default_factory=list)

    # OUTPUT: What firing does
    direct_logit_effects: dict = field(
        default_factory=lambda: {
            "promotes": [],  # List of TokenEffect
            "suppresses": [],  # List of TokenEffect
        }
    )
    downstream_neurons: list[NeuronLink] = field(default_factory=list)

    # Computed metrics
    logit_effect_magnitude: float = 0.0  # Max abs logit contribution
    downstream_effect_magnitude: float = 0.0  # Max abs edge weight to neurons
    output_norm: float = 0.0  # Norm of down_proj column

    # Metadata
    appearance_count: int = 0
    domain_specificity: float = 0.0
    neurondb_label: str = ""

    def effect_ratio(self) -> float:
        """Ratio of logit effect to downstream effect. High = more direct output effect."""
        if self.downstream_effect_magnitude == 0:
            return float("inf") if self.logit_effect_magnitude > 0 else 0
        return self.logit_effect_magnitude / self.downstream_effect_magnitude

    def primary_effect_type(self) -> str:
        """Whether this neuron primarily affects logits or downstream neurons."""
        ratio = self.effect_ratio()
        if ratio > 2:
            return "logit-dominant"
        elif ratio < 0.5:
            return "routing-dominant"
        else:
            return "mixed"


# ---------------------------------------------------------------------------
# NeuronFunctionDB
# ---------------------------------------------------------------------------


class NeuronFunctionDB:
    """Database of neuron function labels for progressive interpretation.

    Stores NeuronFunction objects in a JSON file with get/set/save/load
    operations and layer/effect-type queries.
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path
        self.neurons: dict[str, NeuronFunction] = {}
        self._load()

    def _load(self):
        """Load existing database if it exists."""
        if self.db_path and self.db_path.exists():
            with open(self.db_path) as f:
                data = json.load(f)
            for neuron_id, ndata in data.get("neurons", {}).items():
                self.neurons[neuron_id] = self._dict_to_neuron(ndata)
            print(f"Loaded {len(self.neurons)} neurons from {self.db_path}")

    def _dict_to_neuron(self, d: dict) -> NeuronFunction:
        """Convert dict to NeuronFunction, handling nested structures."""
        upstream = [NeuronLink(**n) for n in d.get("upstream_neurons", [])]
        downstream = [NeuronLink(**n) for n in d.get("downstream_neurons", [])]

        logit_effects = d.get(
            "direct_logit_effects", {"promotes": [], "suppresses": []}
        )
        promotes = [
            TokenEffect(**t) if isinstance(t, dict) else t
            for t in logit_effects.get("promotes", [])
        ]
        suppresses = [
            TokenEffect(**t) if isinstance(t, dict) else t
            for t in logit_effects.get("suppresses", [])
        ]

        return NeuronFunction(
            neuron_id=d["neuron_id"],
            layer=d["layer"],
            neuron_idx=d["neuron_idx"],
            function_label=d.get("function_label", ""),
            function_description=d.get("function_description", ""),
            confidence=d.get("confidence", "unknown"),
            interpretability=d.get("interpretability", "unknown"),
            function_type=d.get("function_type", ""),
            input_label=d.get("input_label", ""),
            input_description=d.get("input_description", ""),
            input_type=d.get("input_type", ""),
            input_interpretability=d.get("input_interpretability", "unknown"),
            input_confidence=d.get("input_confidence", "unknown"),
            activation_patterns=d.get("activation_patterns", []),
            upstream_neurons=upstream,
            direct_logit_effects={"promotes": promotes, "suppresses": suppresses},
            downstream_neurons=downstream,
            logit_effect_magnitude=d.get("logit_effect_magnitude", 0.0),
            downstream_effect_magnitude=d.get("downstream_effect_magnitude", 0.0),
            output_norm=d.get("output_norm", 0.0),
            appearance_count=d.get("appearance_count", 0),
            domain_specificity=d.get("domain_specificity", 0.0),
            neurondb_label=d.get("neurondb_label", ""),
        )

    def _neuron_to_dict(self, n: NeuronFunction) -> dict:
        """Convert NeuronFunction to serializable dict."""
        d = asdict(n)
        d["direct_logit_effects"]["promotes"] = [
            asdict(t) if isinstance(t, TokenEffect) else t
            for t in d["direct_logit_effects"]["promotes"]
        ]
        d["direct_logit_effects"]["suppresses"] = [
            asdict(t) if isinstance(t, TokenEffect) else t
            for t in d["direct_logit_effects"]["suppresses"]
        ]
        return d

    def save(self, quiet: bool = False, batch_count: int = None):
        """Save database to disk.

        Args:
            quiet: If True, don't print status message.
            batch_count: If provided, show "Saved X neurons (Y total)" instead of just total.
        """
        if not self.db_path:
            return
        data = {
            "neurons": {
                nid: self._neuron_to_dict(n) for nid, n in self.neurons.items()
            },
            "metadata": {
                "total_neurons": len(self.neurons),
                "output_labeled": sum(
                    1 for n in self.neurons.values() if n.function_label
                ),
                "input_labeled": sum(
                    1 for n in self.neurons.values() if n.input_label
                ),
                "layers": sorted(set(n.layer for n in self.neurons.values())),
            },
        }
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)
        if not quiet:
            if batch_count is not None:
                print(
                    f"Saved {batch_count} neurons ({len(self.neurons)} total) to {self.db_path}"
                )
            else:
                print(f"Saved {len(self.neurons)} neurons to {self.db_path}")

    def get(self, neuron_id: str) -> NeuronFunction | None:
        """Get a neuron by ID."""
        return self.neurons.get(neuron_id)

    def get_function_label(self, neuron_id: str) -> str:
        """Get just the function label for a neuron."""
        n = self.neurons.get(neuron_id)
        return n.function_label if n else ""

    def set(self, neuron: NeuronFunction):
        """Add or update a neuron."""
        self.neurons[neuron.neuron_id] = neuron

    def get_layer(self, layer: int) -> list[NeuronFunction]:
        """Get all neurons in a layer."""
        return [n for n in self.neurons.values() if n.layer == layer]

    def get_by_effect_type(self, effect_type: str) -> list[NeuronFunction]:
        """Get neurons by their primary effect type."""
        return [
            n for n in self.neurons.values() if n.primary_effect_type() == effect_type
        ]


# ---------------------------------------------------------------------------
# ProgressiveInterpreter
# ---------------------------------------------------------------------------


class ProgressiveInterpreter:
    """Interprets neurons progressively from output to input layers.

    Process:
    1. Load edge statistics and output projections
    2. Start with the last layer (closest to output)
    3. For each neuron, compute:
       - Direct logit effects (output projection)
       - Downstream neuron effects (edge weights to later layers)
    4. Generate function labels (either automatically or with LLM assistance)
    5. Move to earlier layers, describing effects in terms of downstream functions
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        edge_stats_path: Path | None = None,
        db_path: Path | None = None,
    ):
        self.model_name = model_name
        self.edge_stats_path = edge_stats_path
        self.db = NeuronFunctionDB(db_path)

        self.model = None
        self.tokenizer = None
        self.edge_stats = None

        # Cache for output/input projections
        self._output_projections: dict[int, torch.Tensor] = {}

    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is not None:
            return

        print(f"Loading model {self.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("Model loaded.")

    def load_edge_stats(self):
        """Load edge statistics from file."""
        if self.edge_stats is not None:
            return

        if not self.edge_stats_path or not self.edge_stats_path.exists():
            print("No edge stats file specified or file doesn't exist.")
            return

        print(f"Loading edge stats from {self.edge_stats_path}...")
        with open(self.edge_stats_path) as f:
            self.edge_stats = json.load(f)
        print(f"Loaded {len(self.edge_stats.get('profiles', []))} neuron profiles.")

    # ---- Output projection methods ----

    def compute_output_projection(self, layer: int) -> torch.Tensor:
        """Compute output projection for all neurons in a layer.

        Returns tensor of shape [vocab_size, num_neurons] where each column
        is the logit contribution of that neuron.
        """
        if layer in self._output_projections:
            return self._output_projections[layer]

        self.load_model()

        down_proj = self.model.model.layers[layer].mlp.down_proj.weight.float()
        lm_head = self.model.lm_head.weight.float()

        # [vocab, d_model] @ [d_model, d_ffn] = [vocab, d_ffn]
        logit_contributions = lm_head @ down_proj

        self._output_projections[layer] = logit_contributions
        return logit_contributions

    def get_neuron_logit_effects(
        self, layer: int, neuron: int, top_k: int = 20
    ) -> tuple[list[TokenEffect], list[TokenEffect]]:
        """Get top promoted and suppressed tokens for a neuron."""
        self.load_model()

        logits = self.compute_output_projection(layer)[:, neuron]

        top_vals, top_idx = logits.topk(top_k)
        bot_vals, bot_idx = logits.topk(top_k, largest=False)

        promotes = []
        for val, idx in zip(top_vals, top_idx):
            token = self.tokenizer.decode([idx.item()])
            promotes.append(
                TokenEffect(
                    token=token,
                    token_id=idx.item(),
                    logit_contribution=val.item(),
                )
            )

        suppresses = []
        for val, idx in zip(bot_vals, bot_idx):
            token = self.tokenizer.decode([idx.item()])
            suppresses.append(
                TokenEffect(
                    token=token,
                    token_id=idx.item(),
                    logit_contribution=val.item(),
                )
            )

        return promotes, suppresses

    def get_neuron_output_norm(self, layer: int, neuron: int) -> float:
        """Get the norm of a neuron's output direction."""
        self.load_model()
        down_proj = self.model.model.layers[layer].mlp.down_proj.weight
        return down_proj[:, neuron].float().norm().item()

    # ---- Input projection methods ----

    def compute_input_projection(self, layer: int) -> torch.Tensor:
        """Compute input projection for all neurons in a layer.

        Returns tensor of shape [vocab_size, num_neurons] where each column
        shows how much each input token activates that neuron.

        Uses the combined formula: SiLU(embedding @ gate_proj) * (embedding @ up_proj)
        This accounts for Llama's gated MLP architecture where the final activation
        depends on both the gate and up projections.
        """
        cache_key = f"input_{layer}"
        if cache_key in self._output_projections:
            return self._output_projections[cache_key]

        self.load_model()

        up_proj = self.model.model.layers[layer].mlp.up_proj.weight.float()
        gate_proj = self.model.model.layers[layer].mlp.gate_proj.weight.float()
        embeddings = self.model.model.embed_tokens.weight.float()

        # Compute projections: [vocab_size, d_ffn]
        up_vals = embeddings @ up_proj.T
        gate_vals = embeddings @ gate_proj.T

        # Combined: SiLU(gate) * up
        input_sensitivity = torch.nn.functional.silu(gate_vals) * up_vals

        self._output_projections[cache_key] = input_sensitivity
        return input_sensitivity

    def get_neuron_input_sensitivity(
        self, layer: int, neuron: int, top_k: int = 20
    ) -> tuple[list[TokenEffect], list[TokenEffect]]:
        """Get top activating and suppressing input tokens for a neuron.

        This shows which vocabulary tokens, when present in input, would
        most strongly activate or suppress this neuron (based on up_proj weights).
        """
        self.load_model()

        sensitivity = self.compute_input_projection(layer)[:, neuron]

        top_vals, top_idx = sensitivity.topk(top_k)
        bot_vals, bot_idx = sensitivity.topk(top_k, largest=False)

        activates = []
        for val, idx in zip(top_vals, top_idx):
            token = self.tokenizer.decode([idx.item()])
            activates.append(
                TokenEffect(
                    token=token,
                    token_id=idx.item(),
                    logit_contribution=val.item(),
                )
            )

        suppresses = []
        for val, idx in zip(bot_vals, bot_idx):
            token = self.tokenizer.decode([idx.item()])
            suppresses.append(
                TokenEffect(
                    token=token,
                    token_id=idx.item(),
                    logit_contribution=val.item(),
                )
            )

        return activates, suppresses

    def get_neuron_input_norm(self, layer: int, neuron: int) -> float:
        """Get the norm of a neuron's input sensitivity direction (up_proj)."""
        self.load_model()
        up_proj = self.model.model.layers[layer].mlp.up_proj.weight
        return up_proj[neuron, :].float().norm().item()

    # ---- Processing methods (from edge stats) ----

    def process_neuron_from_edge_stats(self, profile: dict) -> NeuronFunction:
        """Process a single neuron profile from edge stats.

        Computes:
        - Direct logit effects (output projection)
        - Downstream neuron effects (from edge stats)
        - Effect magnitude metrics
        """
        neuron_id = profile["neuron_id"]
        parts = neuron_id.split("/")
        layer = int(parts[0][1:])  # "L31" -> 31
        neuron_idx = int(parts[1][1:])  # "N11000" -> 11000

        # Get output projection effects
        promotes, suppresses = self.get_neuron_logit_effects(layer, neuron_idx)
        output_norm = self.get_neuron_output_norm(layer, neuron_idx)

        # Get max logit effect magnitude
        logit_mag = max(
            max(abs(t.logit_contribution) for t in promotes) if promotes else 0,
            max(abs(t.logit_contribution) for t in suppresses) if suppresses else 0,
        )

        # Process upstream sources
        upstream = []
        for src in profile.get("top_upstream_sources", []):
            source_id = src["source"]
            src_parts = source_id.split("_")
            if len(src_parts) >= 2:
                src_layer = int(src_parts[0])
                src_neuron = int(src_parts[1])
                src_id = f"L{src_layer}/N{src_neuron}"

                func_label = self.db.get_function_label(src_id)

                upstream.append(
                    NeuronLink(
                        neuron_id=src_id,
                        layer=src_layer,
                        neuron_idx=src_neuron,
                        weight=src.get("avg_weight", 0),
                        frequency=src.get("frequency", 0),
                        function_label=func_label if func_label else None,
                    )
                )

        # Process downstream targets
        downstream = []
        max_downstream_weight = 0
        for tgt in profile.get("top_downstream_targets", []):
            target_id = tgt["target"]
            tgt_parts = target_id.split("_")

            if len(tgt_parts) >= 2:
                if target_id.startswith("L_"):
                    token_id = int(tgt_parts[1])
                    token = (
                        self.tokenizer.decode([token_id])
                        if self.tokenizer
                        else f"token_{token_id}"
                    )
                    tgt_id = f"LOGIT/{token_id}"
                    tgt_layer = 32
                    tgt_neuron = token_id
                    func_label = f"output token '{token}'"
                else:
                    tgt_layer = int(tgt_parts[0])
                    tgt_neuron = int(tgt_parts[1])
                    tgt_id = f"L{tgt_layer}/N{tgt_neuron}"
                    func_label = self.db.get_function_label(tgt_id)

                weight = abs(tgt.get("avg_weight", 0))
                max_downstream_weight = max(max_downstream_weight, weight)

                downstream.append(
                    NeuronLink(
                        neuron_id=tgt_id,
                        layer=tgt_layer if not target_id.startswith("L_") else 32,
                        neuron_idx=tgt_neuron,
                        weight=tgt.get("avg_weight", 0),
                        frequency=tgt.get("frequency", 0),
                        function_label=func_label if func_label else None,
                    )
                )

        return NeuronFunction(
            neuron_id=neuron_id,
            layer=layer,
            neuron_idx=neuron_idx,
            upstream_neurons=upstream,
            downstream_neurons=downstream,
            direct_logit_effects={"promotes": promotes, "suppresses": suppresses},
            logit_effect_magnitude=logit_mag,
            downstream_effect_magnitude=max_downstream_weight,
            output_norm=output_norm,
            appearance_count=profile.get("appearance_count", 0),
            domain_specificity=profile.get("domain_specificity", 0),
            neurondb_label=profile.get("max_act_label", ""),
        )

    def process_neuron_connections_only(self, profile: dict) -> NeuronFunction:
        """Process a neuron profile from edge stats WITHOUT loading the model.

        Only stores connection data (upstream/downstream neurons).
        Output projections can be computed later with compute_layer_projections().
        """
        neuron_id = profile["neuron_id"]
        parts = neuron_id.split("/")
        layer = int(parts[0][1:])
        neuron_idx = int(parts[1][1:])

        # Process upstream sources
        upstream = []
        for src in profile.get("top_upstream_sources", []):
            source_id = src["source"]
            src_parts = source_id.split("_")
            if len(src_parts) >= 2:
                if source_id.startswith("E_"):
                    src_layer = -1
                    src_neuron = int(src_parts[1])
                    src_id = f"EMB/{src_neuron}"
                    func_label = (
                        f"embedding token {self.tokenizer.decode([src_neuron]) if self.tokenizer and src_neuron < 128000 else src_neuron}"
                    )
                else:
                    src_layer = int(src_parts[0])
                    src_neuron = int(src_parts[1])
                    src_id = f"L{src_layer}/N{src_neuron}"
                    func_label = self.db.get_function_label(src_id)
                upstream.append(
                    NeuronLink(
                        neuron_id=src_id,
                        layer=src_layer,
                        neuron_idx=src_neuron,
                        weight=src.get("avg_weight", 0),
                        frequency=src.get("frequency", 0),
                        function_label=func_label if func_label else None,
                    )
                )

        # Process downstream targets
        downstream = []
        max_downstream_weight = 0.0
        for tgt in profile.get("top_downstream_targets", []):
            target_id = tgt["target"]
            tgt_parts = target_id.split("_")
            if len(tgt_parts) >= 2:
                if target_id.startswith("L_"):
                    tgt_layer = 32
                    tgt_neuron = int(tgt_parts[1])
                    tgt_id = f"LOGIT/{tgt_neuron}"
                    func_label = (
                        f"output token {self.tokenizer.decode([tgt_neuron]) if self.tokenizer else tgt_neuron}"
                    )
                else:
                    tgt_layer = int(tgt_parts[0])
                    tgt_neuron = int(tgt_parts[1])
                    tgt_id = f"L{tgt_layer}/N{tgt_neuron}"
                    func_label = self.db.get_function_label(tgt_id)

                weight = abs(tgt.get("avg_weight", 0))
                max_downstream_weight = max(max_downstream_weight, weight)

                downstream.append(
                    NeuronLink(
                        neuron_id=tgt_id,
                        layer=tgt_layer if not target_id.startswith("L_") else 32,
                        neuron_idx=tgt_neuron,
                        weight=tgt.get("avg_weight", 0),
                        frequency=tgt.get("frequency", 0),
                        function_label=func_label if func_label else None,
                    )
                )

        # Extract output projection data from profile
        output_proj = profile.get("output_projection", {})
        promotes_raw = output_proj.get("promotes", output_proj.get("promoted", []))
        suppresses_raw = output_proj.get(
            "suppresses", output_proj.get("suppressed", [])
        )

        promotes = []
        suppresses = []
        max_logit_effect = 0.0

        for t in promotes_raw:
            weight = t.get("logit_contribution", t.get("weight", 0))
            promotes.append(
                {
                    "token": t.get("token", ""),
                    "token_id": t.get("token_id", 0),
                    "logit_contribution": weight,
                }
            )
            max_logit_effect = max(max_logit_effect, abs(weight))

        for t in suppresses_raw:
            weight = t.get("logit_contribution", t.get("weight", 0))
            suppresses.append(
                {
                    "token": t.get("token", ""),
                    "token_id": t.get("token_id", 0),
                    "logit_contribution": weight,
                }
            )
            max_logit_effect = max(max_logit_effect, abs(weight))

        # Get neurondb label (try multiple field names)
        neurondb_label = (
            profile.get("transluce_label_positive", "")
            or profile.get("max_act_label", "")
            or profile.get("neurondb_label", "")
        )

        return NeuronFunction(
            neuron_id=neuron_id,
            layer=layer,
            neuron_idx=neuron_idx,
            upstream_neurons=upstream,
            downstream_neurons=downstream,
            direct_logit_effects={"promotes": promotes, "suppresses": suppresses},
            logit_effect_magnitude=max_logit_effect,
            downstream_effect_magnitude=max_downstream_weight,
            output_norm=output_proj.get("output_norm", 0.0),
            appearance_count=profile.get("appearance_count", 0),
            domain_specificity=profile.get("domain_specificity", 0.0),
            neurondb_label=neurondb_label,
        )

    # ---- Layer processing ----

    def process_layer(self, layer: int, min_appearances: int = 100):
        """Process all neurons in a layer from edge stats.

        Args:
            layer: Layer number to process.
            min_appearances: Minimum appearance count to include.
        """
        self.load_edge_stats()
        if not self.edge_stats:
            print("No edge stats loaded.")
            return

        profiles = self.edge_stats.get("profiles", [])
        layer_profiles = [
            p
            for p in profiles
            if p["neuron_id"].startswith(f"L{layer}/")
            and p.get("appearance_count", 0) >= min_appearances
        ]

        print(
            f"\nProcessing layer {layer}: {len(layer_profiles)} neurons "
            f"with >= {min_appearances} appearances"
        )

        for i, profile in enumerate(layer_profiles):
            neuron_func = self.process_neuron_from_edge_stats(profile)
            self.db.set(neuron_func)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(layer_profiles)} neurons...")

        print(
            f"  Done. Layer {layer} neurons in database: {len(self.db.get_layer(layer))}"
        )

    def process_all_layers(
        self,
        start_layer: int = 31,
        end_layer: int = 0,
        min_appearances: int = 100,
    ):
        """Process all layers from output to input.

        Args:
            start_layer: Start from this layer (typically 31, closest to output).
            end_layer: End at this layer (typically 0).
            min_appearances: Minimum appearance count to include.
        """
        for layer in range(start_layer, end_layer - 1, -1):
            self.process_layer(layer, min_appearances)
            self.db.save()

    def process_all_connections(self, min_appearances: int = 100):
        """Process ALL neurons from edge stats without loading the model.

        Just stores connection data. Call compute_layer_projections() after.
        """
        self.load_edge_stats()
        if not self.edge_stats:
            print("No edge stats loaded.")
            return

        if not self.tokenizer:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        profiles = self.edge_stats.get("profiles", [])
        filtered = [
            p for p in profiles if p.get("appearance_count", 0) >= min_appearances
        ]

        print(f"\nProcessing {len(filtered)} neurons (connections only, no model)...")

        for i, profile in enumerate(filtered):
            neuron_func = self.process_neuron_connections_only(profile)
            self.db.set(neuron_func)

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(filtered)} neurons...")
                self.db.save()

        self.db.save()
        print(f"Done. Total neurons in database: {len(self.db.neurons)}")

    # ---- Projection computation ----

    def compute_layer_projections(self, layer: int):
        """Compute output projections for all neurons in a specific layer.

        Clears GPU cache after processing to manage memory.
        """
        neurons = self.db.get_layer(layer)
        if not neurons:
            print(f"No neurons found in layer {layer}")
            return

        print(
            f"\nComputing output projections for layer {layer} ({len(neurons)} neurons)..."
        )

        self.load_model()

        for i, n in enumerate(neurons):
            promotes, suppresses = self.get_neuron_logit_effects(n.layer, n.neuron_idx)
            output_norm = self.get_neuron_output_norm(n.layer, n.neuron_idx)

            n.direct_logit_effects = {"promotes": promotes, "suppresses": suppresses}
            n.output_norm = output_norm
            n.logit_effect_magnitude = max(
                max(abs(t.logit_contribution) for t in promotes) if promotes else 0,
                max(abs(t.logit_contribution) for t in suppresses) if suppresses else 0,
            )
            self.db.set(n)

            if (i + 1) % 50 == 0:
                print(f"  Computed {i + 1}/{len(neurons)} neurons...")

        self.db.save()

        # Clear the cached projection for this layer to free memory
        if layer in self._output_projections:
            del self._output_projections[layer]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Done with layer {layer}")

    def compute_all_projections(self, start_layer: int = 31, end_layer: int = 0):
        """Compute output projections for all layers, one layer at a time.

        Manages GPU memory by clearing cache between layers.
        """
        for layer in range(start_layer, end_layer - 1, -1):
            neurons = self.db.get_layer(layer)
            if neurons:
                self.compute_layer_projections(layer)

    # ---- Label propagation ----

    def propagate_labels(self):
        """Propagate function labels through the network.

        For each neuron with connections, update the function_label field
        in NeuronLinks based on current database state.
        """
        updated = 0
        for neuron in self.db.neurons.values():
            for down in neuron.downstream_neurons:
                current_label = self.db.get_function_label(down.neuron_id)
                if current_label and down.function_label != current_label:
                    down.function_label = current_label
                    updated += 1

            for up in neuron.upstream_neurons:
                current_label = self.db.get_function_label(up.neuron_id)
                if current_label and up.function_label != current_label:
                    up.function_label = current_label
                    updated += 1

        print(f"Propagated {updated} labels")
        self.db.save()

    # ---- Auto labeling (rule-based) ----

    def auto_generate_label(self, neuron_id: str) -> str:
        """Auto-generate a function label based on output projection and downstream effects.

        For late layers: Focus on promoted/suppressed tokens.
        For earlier layers: Focus on downstream neuron effects.
        """
        n = self.db.get(neuron_id)
        if not n:
            return ""

        parts = []

        effect_type = n.primary_effect_type()

        if effect_type in ("logit-dominant", "mixed") and n.logit_effect_magnitude > 0.05:
            promotes = n.direct_logit_effects.get("promotes", [])
            suppresses = n.direct_logit_effects.get("suppresses", [])

            if promotes:
                top_promoted = [t.token.strip() for t in promotes[:3] if t.token.strip()]
                if top_promoted:
                    if all(len(t) == 1 and t.isupper() for t in top_promoted):
                        parts.append("capital letter promoter")
                    elif any(t.lower() in ("the", "a", "an") for t in top_promoted):
                        parts.append("article promoter")
                    elif any(
                        t.lower() in ("their", "its", "his", "her")
                        for t in top_promoted
                    ):
                        parts.append("possessive promoter")
                    elif any(t in ("(", "[", "{") for t in top_promoted):
                        parts.append("parenthetical promoter")
                    elif any(t in (".", "!", "?") for t in top_promoted):
                        parts.append("sentence-end promoter")
                    else:
                        parts.append(f"'{top_promoted[0]}' promoter")

            if suppresses:
                top_suppressed = [
                    t.token.strip() for t in suppresses[:3] if t.token.strip()
                ]
                if top_suppressed:
                    if any(t.lower() in ("the", "a", "an") for t in top_suppressed):
                        parts.append("article suppressor")

        if effect_type == "routing-dominant" or not parts:
            labeled_downstream = [
                d
                for d in n.downstream_neurons
                if d.function_label and not d.neuron_id.startswith("LOGIT/")
            ]

            if labeled_downstream:
                promotes_funcs = [d for d in labeled_downstream if d.weight > 0]
                suppresses_funcs = [d for d in labeled_downstream if d.weight < 0]

                if suppresses_funcs:
                    strongest = max(suppresses_funcs, key=lambda x: abs(x.weight))
                    func_name = strongest.function_label.replace(" promoter", "")
                    parts.append(f"suppresses {func_name}")

                if promotes_funcs:
                    strongest = max(promotes_funcs, key=lambda x: abs(x.weight))
                    func_name = strongest.function_label.replace(" promoter", "")
                    parts.append(f"promotes {func_name}")

        if not parts:
            return ""

        return " / ".join(parts[:2])

    def auto_label_layer(
        self, layer: int, min_magnitude: float = 0.1, overwrite: bool = False
    ):
        """Auto-generate labels for all neurons in a layer.

        Args:
            layer: Layer to process.
            min_magnitude: Minimum logit effect magnitude to label.
            overwrite: Whether to overwrite existing labels.
        """
        neurons = self.db.get_layer(layer)
        labeled = 0

        for n in neurons:
            if n.function_label and not overwrite:
                continue

            if (
                n.logit_effect_magnitude < min_magnitude
                and n.downstream_effect_magnitude < min_magnitude
            ):
                continue

            label = self.auto_generate_label(n.neuron_id)
            if label:
                n.function_label = label
                n.confidence = "auto"
                self.db.set(n)
                labeled += 1
                print(f"  {n.neuron_id}: {label}")

        print(f"Auto-labeled {labeled} neurons in layer {layer}")
        self.db.save()

    # ---- LLM labeling (single neuron) ----

    def llm_generate_label(self, neuron_id: str, model: str = "gpt-5-mini") -> str:
        """Use LLM to generate a function label for a neuron.

        For terminal layers: Uses output projection (promoted/suppressed tokens).
        For earlier layers: Also includes upstream/downstream neuron functions.
        """
        if not OPENAI_AVAILABLE:
            print("OpenAI not available. Install with: pip install openai")
            return ""

        n = self.db.get(neuron_id)
        if not n:
            return ""

        client = OpenAI()
        prompt = self._build_label_prompt(n)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2000,
            )
            label = response.choices[0].message.content.strip().strip("\"'")
            return label
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return ""

    def llm_generate_input_description(
        self, neuron_id: str, model: str = "gpt-5-mini"
    ) -> str:
        """Use LLM to generate an INPUT description -- what makes this neuron fire.

        Called on Pass 2 (early-to-late) when upstream neurons have labels.
        """
        if not OPENAI_AVAILABLE:
            print("OpenAI not available. Install with: pip install openai")
            return ""

        n = self.db.get(neuron_id)
        if not n:
            return ""

        client = OpenAI()
        prompt = self._build_input_description_prompt(n)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2000,
            )
            desc = response.choices[0].message.content.strip().strip("\"'")
            return desc
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return ""

    # ---- LLM labeling (async batch) ----

    async def llm_label_layer_async(
        self,
        layer: int,
        model: str = "gpt-5-mini",
        min_magnitude: float = 0.05,
        overwrite: bool = False,
        batch_size: int = 64,
    ):
        """Async batch LLM labeling for a layer."""
        import asyncio

        neurons = self.db.get_layer(layer)
        if not neurons:
            print(f"No neurons found in layer {layer}")
            return

        to_label = []
        for n in neurons:
            if n.function_label and not overwrite:
                continue
            if (
                n.logit_effect_magnitude < min_magnitude
                and n.downstream_effect_magnitude < min_magnitude
            ):
                continue
            to_label.append(n)

        if not to_label:
            print(f"Layer {layer}: No neurons need labeling")
            return

        print(f"\nAsync LLM labeling layer {layer} ({len(to_label)} neurons to label)...")

        client = AsyncOpenAI()

        async def label_neuron(n):
            prompt = self._build_label_prompt(n)
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=2000,
                )
                return n, response.choices[0].message.content.strip().strip("\"'")
            except Exception as e:
                print(f"  Error for {n.neuron_id}: {e}")
                return n, ""

        labeled = 0
        for i in range(0, len(to_label), batch_size):
            batch = to_label[i : i + batch_size]
            results = await asyncio.gather(*[label_neuron(n) for n in batch])

            for n, label in results:
                if label:
                    n.function_label = label
                    n.confidence = "llm-auto"
                    self.db.set(n)
                    labeled += 1
                    print(f"  {n.neuron_id}: {label}")

            print(f"  Batch {i // batch_size + 1}: {len(batch)} neurons processed")

        print(f"LLM-labeled {labeled} neurons in layer {layer}")
        self.db.save()

    async def llm_input_description_layer_async(
        self,
        layer: int,
        model: str = "gpt-5-mini",
        min_upstream_labels: int = 2,
        overwrite: bool = False,
        batch_size: int = 64,
    ):
        """Async batch input description generation for a layer."""
        import asyncio

        neurons = self.db.get_layer(layer)
        if not neurons:
            print(f"No neurons found in layer {layer}")
            return

        to_label = []
        for n in neurons:
            if n.input_description and not overwrite:
                continue
            labeled_upstream = sum(1 for u in n.upstream_neurons if u.function_label)
            if labeled_upstream < min_upstream_labels:
                continue
            to_label.append(n)

        if not to_label:
            print(f"Layer {layer}: No neurons need input descriptions")
            return

        print(
            f"\nAsync generating input descriptions for layer {layer} "
            f"({len(to_label)} neurons)..."
        )

        client = AsyncOpenAI()

        async def describe_neuron(n):
            prompt = self._build_input_description_prompt(n)
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=2000,
                )
                return n, response.choices[0].message.content.strip().strip("\"'")
            except Exception as e:
                print(f"  Error for {n.neuron_id}: {e}")
                return n, ""

        labeled = 0
        for i in range(0, len(to_label), batch_size):
            batch = to_label[i : i + batch_size]
            results = await asyncio.gather(*[describe_neuron(n) for n in batch])

            for n, desc in results:
                if desc:
                    n.input_description = desc
                    n.input_confidence = "llm-auto"
                    self.db.set(n)
                    labeled += 1
                    print(f"  {n.neuron_id}: {desc}")

            print(f"  Batch {i // batch_size + 1}: {len(batch)} neurons processed")

        print(f"Generated {labeled} input descriptions in layer {layer}")
        self.db.save()

    # ---- LLM labeling (sync wrappers) ----

    def llm_label_layer(
        self,
        layer: int,
        model: str = "gpt-5-mini",
        min_magnitude: float = 0.05,
        overwrite: bool = False,
        use_async: bool = True,
    ):
        """Use LLM to generate labels for all neurons in a layer.

        Args:
            layer: Layer to process.
            model: OpenAI model to use.
            min_magnitude: Minimum effect magnitude to label.
            overwrite: Whether to overwrite existing labels.
            use_async: Use async batch processing (faster).
        """
        if use_async:
            import asyncio

            asyncio.run(
                self.llm_label_layer_async(layer, model, min_magnitude, overwrite)
            )
            return

        neurons = self.db.get_layer(layer)
        if not neurons:
            print(f"No neurons found in layer {layer}")
            return

        print(f"\nLLM labeling layer {layer} ({len(neurons)} neurons)...")
        labeled = 0

        for n in neurons:
            if n.function_label and not overwrite:
                continue

            if (
                n.logit_effect_magnitude < min_magnitude
                and n.downstream_effect_magnitude < min_magnitude
            ):
                continue

            label = self.llm_generate_label(n.neuron_id, model=model)
            if label:
                n.function_label = label
                n.confidence = "llm-auto"
                self.db.set(n)
                labeled += 1
                print(f"  {n.neuron_id}: {label}")

        print(f"LLM-labeled {labeled} neurons in layer {layer}")
        self.db.save()

    def llm_input_description_layer(
        self,
        layer: int,
        model: str = "gpt-5-mini",
        min_upstream_labels: int = 2,
        overwrite: bool = False,
        use_async: bool = True,
    ):
        """Generate input descriptions for neurons in a layer.

        Only processes neurons that have enough labeled upstream neurons.
        """
        if use_async:
            import asyncio

            asyncio.run(
                self.llm_input_description_layer_async(
                    layer, model, min_upstream_labels, overwrite
                )
            )
            return

        neurons = self.db.get_layer(layer)
        if not neurons:
            print(f"No neurons found in layer {layer}")
            return

        print(f"\nGenerating input descriptions for layer {layer} ({len(neurons)} neurons)...")
        labeled = 0

        for n in neurons:
            if n.input_description and not overwrite:
                continue

            labeled_upstream = sum(1 for u in n.upstream_neurons if u.function_label)
            if labeled_upstream < min_upstream_labels:
                continue

            desc = self.llm_generate_input_description(n.neuron_id, model=model)
            if desc:
                n.input_description = desc
                n.input_confidence = "llm-auto"
                self.db.set(n)
                labeled += 1
                print(f"  {n.neuron_id}: {desc}")

        print(f"Generated {labeled} input descriptions in layer {layer}")
        self.db.save()

    # ---- Multi-pass orchestration ----

    def llm_label_all_layers(
        self,
        start_layer: int = 31,
        end_layer: int = 0,
        model: str = "gpt-5-mini",
        passes: int = 2,
    ):
        """Run multi-pass LLM labeling across all layers.

        Pass 1 (late-to-early): Generate function_labels (OUTPUT -- what neuron does).
        Pass 2 (early-to-late): Generate input_descriptions (INPUT -- what activates it).

        This is ADDITIVE -- Pass 2 adds input_description without changing function_label.

        Args:
            start_layer: Start from this layer (typically 31).
            end_layer: End at this layer (typically 0).
            model: OpenAI model to use.
            passes: Number of passes (2 recommended).
        """
        # PASS 1: Late-to-Early
        print(f"\n{'=' * 60}")
        print("PASS 1: Late-to-Early (generating function_labels)")
        print(f"{'=' * 60}")

        for layer in range(start_layer, end_layer - 1, -1):
            neurons = self.db.get_layer(layer)
            if not neurons:
                continue

            self.llm_label_layer(layer, model=model, overwrite=False)
            self.propagate_labels()

        if passes >= 2:
            # PASS 2: Early-to-Late
            print(f"\n{'=' * 60}")
            print("PASS 2: Early-to-Late (generating input_descriptions)")
            print(f"{'=' * 60}")

            for layer in range(end_layer, start_layer + 1):
                neurons = self.db.get_layer(layer)
                if not neurons:
                    continue

                self.llm_input_description_layer(layer, model=model, overwrite=False)
                self.propagate_labels()

        print(f"\nCompleted {passes} passes of LLM labeling")

        all_neurons = list(self.db.neurons.values())
        func_labeled = sum(1 for n in all_neurons if n.function_label)
        input_labeled = sum(1 for n in all_neurons if n.input_description)
        print(f"  Function labels: {func_labeled}/{len(all_neurons)}")
        print(f"  Input descriptions: {input_labeled}/{len(all_neurons)}")

    # ---- Pass 1 / Pass 2 standalone ----

    def run_output_pass(
        self,
        layers: list[int] | None = None,
        start_layer: int = 31,
        end_layer: int = 0,
        model: str = "gpt-5-mini",
    ):
        """Pass 1: Late-to-Early -- what does each unit DO?

        Args:
            layers: Specific layers to process (overrides start/end).
            start_layer: Start layer (inclusive).
            end_layer: End layer (inclusive).
            model: OpenAI model to use.
        """
        if layers is None:
            layers = list(range(start_layer, end_layer - 1, -1))

        print(f"\nOutput pass: {len(layers)} layers, late-to-early")
        for layer in layers:
            neurons = self.db.get_layer(layer)
            if not neurons:
                continue
            self.llm_label_layer(layer, model=model, overwrite=False)
            self.propagate_labels()

    def run_input_pass(
        self,
        layers: list[int] | None = None,
        start_layer: int = 0,
        end_layer: int = 31,
        model: str = "gpt-5-mini",
    ):
        """Pass 2: Early-to-Late -- what TRIGGERS each unit?

        Args:
            layers: Specific layers to process (overrides start/end).
            start_layer: Start layer (inclusive).
            end_layer: End layer (inclusive).
            model: OpenAI model to use.
        """
        if layers is None:
            layers = list(range(start_layer, end_layer + 1))

        print(f"\nInput pass: {len(layers)} layers, early-to-late")
        for layer in layers:
            neurons = self.db.get_layer(layer)
            if not neurons:
                continue
            self.llm_input_description_layer(layer, model=model, overwrite=False)
            self.propagate_labels()

    # ---- Prompt builders ----

    def _build_label_prompt(self, n: NeuronFunction) -> str:
        """Build the prompt for LLM labeling with layer-appropriate guidance."""
        context_parts = []
        layer = n.layer

        context_parts.append(f"Neuron: {n.neuron_id} (Layer {layer} of 31)")
        context_parts.append(f"Appears in {n.appearance_count} prompts")
        if n.neurondb_label:
            context_parts.append(
                f"Max-activation label from NeuronDB: {n.neurondb_label}"
            )

        downstream_with_labels = [
            d
            for d in n.downstream_neurons
            if not d.neuron_id.startswith("LOGIT/") and d.function_label
        ]
        downstream_without_labels = [
            d
            for d in n.downstream_neurons
            if not d.neuron_id.startswith("LOGIT/") and not d.function_label
        ]

        logit_connections = [
            d for d in n.downstream_neurons if d.neuron_id.startswith("LOGIT/")
        ]

        # For LATE layers (25-31): emphasize token effects
        if layer >= 25:
            if n.direct_logit_effects["promotes"]:
                promotes = [
                    f"'{t.token}'" for t in n.direct_logit_effects["promotes"][:10]
                ]
                context_parts.append(
                    f"\nDirect token effects - PROMOTES: {', '.join(promotes)}"
                )
            if n.direct_logit_effects["suppresses"]:
                suppresses = [
                    f"'{t.token}'" for t in n.direct_logit_effects["suppresses"][:10]
                ]
                context_parts.append(
                    f"Direct token effects - SUPPRESSES: {', '.join(suppresses)}"
                )
            if logit_connections:
                context_parts.append("\nStrong logit connections:")
                for d in sorted(logit_connections, key=lambda x: -abs(x.weight))[:5]:
                    token = (
                        d.function_label.replace("output token ", "")
                        if d.function_label
                        else d.neuron_id
                    )
                    sign = "+" if d.weight > 0 else "-"
                    context_parts.append(f"  {sign} {token}")

        # For EARLY/MID layers (0-24): emphasize downstream effects
        else:
            if downstream_with_labels:
                context_parts.append(
                    "\n*** KEY: Downstream neurons this ACTIVATES (with known functions) ***"
                )
                for d in sorted(downstream_with_labels, key=lambda x: -abs(x.weight))[
                    :8
                ]:
                    sign = "ACTIVATES" if d.weight > 0 else "INHIBITS"
                    context_parts.append(
                        f'  {sign}: "{d.function_label}" ({d.neuron_id}, w={d.weight:+.3f})'
                    )

            if n.direct_logit_effects["promotes"]:
                promotes = [
                    f"'{t.token}'" for t in n.direct_logit_effects["promotes"][:5]
                ]
                context_parts.append(
                    f"\n(Secondary - token effects: promotes {', '.join(promotes)})"
                )

        context = "\n".join(context_parts)

        if layer >= 25:
            layer_guidance = (
                "This is a LATE layer neuron (close to output). Describe its DIRECT effect on tokens.\n"
                "Focus on what tokens it promotes/suppresses in the output vocabulary."
            )
            examples = (
                '- "promotes dopamine and neurotransmitter tokens when medical context is active"\n'
                '- "suppresses common articles like \'the\' and \'a\' in favor of specific nouns"\n'
                '- "gates medical terminology completion by boosting drug name continuations"\n'
                '- "promotes sentence-ending punctuation and suppresses mid-sentence tokens"'
            )
        elif layer >= 10:
            layer_guidance = (
                "This is a MID layer neuron. Describe it as a BRIDGE or GATE between detection and output.\n"
                "Your label MUST reference what downstream functions it activates/inhibits."
            )
            examples = (
                '- "activates late-layer article promoters when sentence boundaries detected"\n'
                '- "gates the chemical-term to dopamine-output pathway based on medical context"\n'
                '- "bridges early entity detection to late-layer capitalization and formatting neurons"\n'
                '- "inhibits code-formatting neurons when natural language context is strong"'
            )
        else:
            layer_guidance = (
                "This is an EARLY layer neuron (close to input). Describe what DOWNSTREAM FUNCTIONS it triggers.\n"
                "Your label MUST reference the downstream neuron functions it activates - NOT just tokens."
            )
            examples = (
                '- "triggers mid-layer continuation promoters and sentence-boundary detectors"\n'
                '- "activates medical-term processing pathway that feeds into drug-name completion"\n'
                '- "feeds downstream sentence-boundary and punctuation-handling neurons"\n'
                '- "initiates formal academic style pathway by activating article and structure neurons"'
            )

        return f"""You are analyzing neurons in Llama-3.1-8B to understand their FUNCTIONAL ROLE in the network.

{context}

{layer_guidance}

Write a DETAILED functional description (2-3 sentences) of what this neuron does and how it contributes to the network's computation. Include:
1. What triggers/activates it
2. What downstream effects it has (what other neurons or outputs it influences)
3. Its role in the overall computation

Good examples:
{examples}

Write your description now (2-3 sentences, be specific):"""

    def _build_input_description_prompt(self, n: NeuronFunction) -> str:
        """Build the prompt for input description generation."""
        context_parts = []

        context_parts.append(f"Neuron: {n.neuron_id} (Layer {n.layer})")
        context_parts.append(f"Appears in {n.appearance_count} prompts")

        if n.neurondb_label:
            context_parts.append(f"NeuronDB max-activation label: {n.neurondb_label}")

        if n.function_label:
            context_parts.append(f"\nThis neuron's OUTPUT function: {n.function_label}")

        upstream_with_labels = [u for u in n.upstream_neurons if u.function_label]
        upstream_without_labels = [u for u in n.upstream_neurons if not u.function_label]

        if upstream_with_labels:
            context_parts.append(
                "\nUpstream neurons that ACTIVATE this neuron (with known functions):"
            )
            for u in sorted(upstream_with_labels, key=lambda x: -abs(x.weight))[:10]:
                sign = "EXCITED by" if u.weight > 0 else "INHIBITED by"
                context_parts.append(
                    f"  - {sign} '{u.function_label}' ({u.neuron_id}, weight={u.weight:+.4f})"
                )

        if upstream_without_labels:
            context_parts.append("\nOther upstream neurons (functions unknown):")
            for u in sorted(upstream_without_labels, key=lambda x: -abs(x.weight))[:5]:
                sign = "excited by" if u.weight > 0 else "inhibited by"
                context_parts.append(
                    f"  - {sign} {u.neuron_id} (weight={u.weight:+.4f})"
                )

        if n.direct_logit_effects["promotes"]:
            promotes = [t.token for t in n.direct_logit_effects["promotes"][:5]]
            context_parts.append(f"\n(For reference - promotes tokens: {promotes})")

        context = "\n".join(context_parts)

        return f"""You are analyzing what ACTIVATES a neuron in the Llama-3.1-8B language model.

{context}

Based on the UPSTREAM neurons that feed into this neuron, generate a SHORT (3-7 words) description of what INPUT CONDITIONS cause this neuron to fire.

Guidelines:
- Focus on WHAT ACTIVATES IT, not what it does
- Use the upstream neuron functions to infer activation conditions
- If excited by "sentence boundary activator", it fires at sentence boundaries
- If inhibited by "question context", it fires in non-question contexts
- Describe the CONTEXT or PATTERN that triggers activation

Examples of good input descriptions:
- "fires at sentence boundaries"
- "activated by formal/academic context"
- "triggered by list structure"
- "fires when code detected"
- "activated after possessive pronouns"

Respond with ONLY the description, no explanation."""

    # ---- Summary / description helpers ----

    def generate_layer_summary(self, layer: int) -> str:
        """Generate a summary of neurons in a layer."""
        neurons = self.db.get_layer(layer)
        if not neurons:
            return f"Layer {layer}: No neurons in database"

        logit_dominant = [
            n for n in neurons if n.primary_effect_type() == "logit-dominant"
        ]
        routing_dominant = [
            n for n in neurons if n.primary_effect_type() == "routing-dominant"
        ]
        mixed = [n for n in neurons if n.primary_effect_type() == "mixed"]

        logit_dominant.sort(key=lambda n: -n.logit_effect_magnitude)

        lines = [
            f"=== Layer {layer} Summary ===",
            f"Total neurons: {len(neurons)}",
            f"  Logit-dominant: {len(logit_dominant)}",
            f"  Routing-dominant: {len(routing_dominant)}",
            f"  Mixed: {len(mixed)}",
            "",
            "Top 10 by logit effect magnitude:",
        ]

        for n in sorted(neurons, key=lambda x: -x.logit_effect_magnitude)[:10]:
            lines.append(
                f"  {n.neuron_id}: logit={n.logit_effect_magnitude:.3f}, "
                f"downstream={n.downstream_effect_magnitude:.3f}, "
                f"type={n.primary_effect_type()}"
            )
            if n.direct_logit_effects["promotes"]:
                top_tokens = [t.token for t in n.direct_logit_effects["promotes"][:3]]
                lines.append(f"    promotes: {top_tokens}")

        return "\n".join(lines)

    def describe_neuron(self, neuron_id: str) -> str:
        """Generate a human-readable description of a neuron."""
        n = self.db.get(neuron_id)
        if not n:
            return f"Neuron {neuron_id} not found in database"

        lines = [
            f"=== {neuron_id} ===",
            f"NeuronDB label: {n.neurondb_label or '(none)'}",
            f"Function label: {n.function_label or '(not yet labeled)'}",
            "",
            f"Effect type: {n.primary_effect_type()}",
            f"  Logit effect magnitude: {n.logit_effect_magnitude:.4f}",
            f"  Downstream effect magnitude: {n.downstream_effect_magnitude:.4f}",
            f"  Output norm: {n.output_norm:.4f}",
            "",
            f"Appearances: {n.appearance_count}",
            f"Domain specificity: {n.domain_specificity:.2f}",
            "",
            "=== INPUT (what makes this fire) ===",
        ]

        if n.upstream_neurons:
            lines.append("Upstream neurons:")
            for up in sorted(n.upstream_neurons, key=lambda x: -abs(x.weight))[:10]:
                func = f" -> {up.function_label}" if up.function_label else ""
                lines.append(
                    f"  {up.neuron_id} (weight={up.weight:+.4f}, freq={up.frequency:.2f}){func}"
                )

        lines.append("")
        lines.append("=== OUTPUT (what firing does) ===")

        if n.direct_logit_effects["promotes"]:
            lines.append("Direct logit effects - PROMOTES:")
            for t in n.direct_logit_effects["promotes"][:10]:
                lines.append(
                    f"  {repr(t.token):20s} logit: {t.logit_contribution:+.4f}"
                )

        if n.direct_logit_effects["suppresses"]:
            lines.append("Direct logit effects - SUPPRESSES:")
            for t in n.direct_logit_effects["suppresses"][:10]:
                lines.append(
                    f"  {repr(t.token):20s} logit: {t.logit_contribution:+.4f}"
                )

        if n.downstream_neurons:
            lines.append("")
            lines.append("Downstream neurons:")
            for down in sorted(n.downstream_neurons, key=lambda x: -abs(x.weight))[
                :10
            ]:
                func = f" -> {down.function_label}" if down.function_label else ""
                sign = "+" if down.weight > 0 else ""
                lines.append(
                    f"  {down.neuron_id} (weight={sign}{down.weight:.4f}, freq={down.frequency:.2f}){func}"
                )

        return "\n".join(lines)

    def set_function_label(
        self,
        neuron_id: str,
        label: str,
        description: str = "",
        confidence: str = "medium",
    ):
        """Manually set a function label for a neuron."""
        n = self.db.get(neuron_id)
        if not n:
            print(f"Neuron {neuron_id} not found")
            return

        n.function_label = label
        n.function_description = description
        n.confidence = confidence
        self.db.set(n)
        print(f"Set function label for {neuron_id}: {label}")

    def generate_functional_summary(self, neuron_id: str) -> str:
        """Generate a natural language summary of a neuron's function,
        describing both INPUT and OUTPUT.
        """
        n = self.db.get(neuron_id)
        if not n:
            return f"Neuron {neuron_id} not found"

        lines = [f"**{neuron_id}**"]

        if n.function_label:
            lines.append(f"OUTPUT function: {n.function_label}")
        if n.input_description:
            lines.append(f"INPUT condition: {n.input_description}")

        effect_type = n.primary_effect_type()
        logit_mag = n.logit_effect_magnitude
        down_mag = n.downstream_effect_magnitude

        lines.append("")
        if effect_type == "logit-dominant":
            lines.append(
                f"Primary effect: Direct logit modification (mag={logit_mag:.3f})"
            )
        elif effect_type == "routing-dominant":
            lines.append(
                f"Primary effect: Routing to downstream neurons (mag={down_mag:.3f})"
            )
        else:
            lines.append(
                f"Mixed effects: logit={logit_mag:.3f}, downstream={down_mag:.3f}"
            )

        lines.append("")
        lines.append("**When this neuron fires:**")

        if n.direct_logit_effects["promotes"] and logit_mag > 0.05:
            top3 = [t.token for t in n.direct_logit_effects["promotes"][:3]]
            lines.append(f"  - Directly promotes tokens: {top3}")

        if n.direct_logit_effects["suppresses"] and logit_mag > 0.05:
            top3 = [t.token for t in n.direct_logit_effects["suppresses"][:3]]
            lines.append(f"  - Directly suppresses tokens: {top3}")

        labeled_downstream = [
            d
            for d in n.downstream_neurons
            if d.function_label and not d.neuron_id.startswith("LOGIT/")
        ]

        if labeled_downstream:
            promotes = [d for d in labeled_downstream if d.weight > 0]
            suppresses = [d for d in labeled_downstream if d.weight < 0]

            if promotes:
                for d in sorted(promotes, key=lambda x: -x.weight)[:3]:
                    lines.append(
                        f"  - PROMOTES '{d.function_label}' (weight={d.weight:+.3f})"
                    )

            if suppresses:
                for d in sorted(suppresses, key=lambda x: x.weight)[:3]:
                    lines.append(
                        f"  - SUPPRESSES '{d.function_label}' (weight={d.weight:+.3f})"
                    )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bridge functions: NeuronFunction <-> Unit
# ---------------------------------------------------------------------------


def neuron_function_to_unit(nf: NeuronFunction) -> Unit:
    """Convert a NeuronFunction to a Unit schema object.

    Maps:
      - NeuronFunction.neuron_id (L{layer}/N{index}) -> Unit.layer, Unit.index
      - NeuronFunction.function_label -> Unit.label, Unit.output_label
      - NeuronFunction.input_label / input_description -> Unit.input_label
      - NeuronFunction.direct_logit_effects -> Unit.output_projections
      - NeuronFunction.appearance_count -> Unit.appearance_count
      - NeuronFunction.output_norm -> Unit.output_norm
      - Labels are added to Unit.labels with appropriate LabelSource
    """
    # Build output projections
    output_projections = None
    dle = nf.direct_logit_effects
    if dle.get("promotes") or dle.get("suppresses"):
        output_projections = {}
        for key in ("promotes", "suppresses"):
            items = dle.get(key, [])
            tps = []
            for item in items:
                if isinstance(item, TokenEffect):
                    tps.append(
                        TokenProjection(
                            token=item.token,
                            token_id=item.token_id,
                            weight=item.logit_contribution,
                        )
                    )
                elif isinstance(item, dict):
                    tps.append(
                        TokenProjection(
                            token=item.get("token", ""),
                            token_id=item.get("token_id", 0),
                            weight=item.get("logit_contribution", 0.0),
                        )
                    )
            output_projections[key] = tps

    # Build labels list
    labels: list[UnitLabel] = []
    if nf.function_label:
        labels.append(
            UnitLabel(
                text=nf.function_label,
                source=LabelSource.PROGRESSIVE_OUTPUT,
                confidence=nf.confidence,
                description=nf.function_description,
            )
        )
    if nf.input_description:
        labels.append(
            UnitLabel(
                text=nf.input_description,
                source=LabelSource.PROGRESSIVE_INPUT,
                confidence=nf.input_confidence,
                description="",
            )
        )
    if nf.neurondb_label:
        labels.append(
            UnitLabel(
                text=nf.neurondb_label,
                source=LabelSource.NEURONDB,
                confidence="unknown",
            )
        )

    return Unit(
        layer=nf.layer,
        index=nf.neuron_idx,
        label=nf.function_label,
        input_label=nf.input_label or nf.input_description,
        output_label=nf.function_label,
        labels=labels,
        max_activation=None,
        appearance_count=nf.appearance_count,
        output_projections=output_projections,
        output_norm=nf.output_norm if nf.output_norm else None,
    )


def unit_to_neuron_function(unit: Unit) -> NeuronFunction:
    """Convert a Unit schema object to a NeuronFunction.

    Maps:
      - Unit.layer, Unit.index -> NeuronFunction.layer, neuron_idx, neuron_id
      - Unit.label -> NeuronFunction.function_label
      - Unit.input_label -> NeuronFunction.input_label
      - Unit.output_projections -> NeuronFunction.direct_logit_effects
      - Unit.appearance_count -> NeuronFunction.appearance_count
      - Unit.output_norm -> NeuronFunction.output_norm
      - NeuronDB labels are extracted from Unit.labels
    """
    neuron_id = f"L{unit.layer}/N{unit.index}"

    # Convert output projections -> direct_logit_effects
    direct_logit_effects: dict[str, list] = {"promotes": [], "suppresses": []}
    max_logit_effect = 0.0

    if unit.output_projections:
        for key in ("promotes", "suppresses"):
            for tp in unit.output_projections.get(key, []):
                te = TokenEffect(
                    token=tp.token,
                    token_id=tp.token_id,
                    logit_contribution=tp.weight,
                )
                direct_logit_effects[key].append(te)
                max_logit_effect = max(max_logit_effect, abs(tp.weight))

    # Extract neurondb label from labels list
    neurondb_label = ""
    for lbl in unit.labels:
        if lbl.source == LabelSource.NEURONDB:
            neurondb_label = lbl.text
            break

    # Extract confidence from output label
    confidence = "unknown"
    for lbl in unit.labels:
        if lbl.source == LabelSource.PROGRESSIVE_OUTPUT:
            confidence = lbl.confidence
            break

    return NeuronFunction(
        neuron_id=neuron_id,
        layer=unit.layer,
        neuron_idx=unit.index,
        function_label=unit.label or unit.output_label or "",
        input_label=unit.input_label or "",
        direct_logit_effects=direct_logit_effects,
        logit_effect_magnitude=max_logit_effect,
        output_norm=unit.output_norm or 0.0,
        appearance_count=unit.appearance_count,
        neurondb_label=neurondb_label,
        confidence=confidence,
    )
