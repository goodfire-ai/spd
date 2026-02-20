#!/usr/bin/env python3
"""
Progressive Interpretation System for Neural Network Attribution Graphs.

This system interprets neurons layer-by-layer from output to input, building
a hierarchical understanding where:
- Late layer functions are described in terms of token effects
- Earlier layer functions are described in terms of downstream neuron effects

For each neuron, we track:
1. INPUT: What makes it fire (contexts + upstream neurons)
2. OUTPUT: What firing does (downstream neurons + direct logit effects)

Key insight: As we go earlier in the network, direct logit effects typically
decrease in magnitude/importance, while routing/gating effects increase.
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

# Load environment variables
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# OpenAI for LLM-assisted labeling
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


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
    function_label: str = ""  # Short label like "article→possessive switch"
    function_description: str = ""  # Longer description
    confidence: str = "unknown"  # low, medium, high
    interpretability: str = "unknown"  # low, medium, high - how clear/confident the interpretation is
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
    direct_logit_effects: dict = field(default_factory=lambda: {
        "promotes": [],  # List of TokenEffect
        "suppresses": [],  # List of TokenEffect
    })
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
            return float('inf') if self.logit_effect_magnitude > 0 else 0
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


class NeuronFunctionDB:
    """Database of neuron function labels for progressive interpretation."""

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
        # Convert upstream/downstream neurons
        upstream = [NeuronLink(**n) for n in d.get("upstream_neurons", [])]
        downstream = [NeuronLink(**n) for n in d.get("downstream_neurons", [])]

        # Convert token effects
        logit_effects = d.get("direct_logit_effects", {"promotes": [], "suppresses": []})
        promotes = [TokenEffect(**t) if isinstance(t, dict) else t
                   for t in logit_effects.get("promotes", [])]
        suppresses = [TokenEffect(**t) if isinstance(t, dict) else t
                     for t in logit_effects.get("suppresses", [])]

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
        # Convert TokenEffect objects
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
            quiet: If True, don't print status message
            batch_count: If provided, show "Saved X neurons (Y total)" instead of just total
        """
        if not self.db_path:
            return
        data = {
            "neurons": {nid: self._neuron_to_dict(n) for nid, n in self.neurons.items()},
            "metadata": {
                "total_neurons": len(self.neurons),
                "output_labeled": sum(1 for n in self.neurons.values() if n.function_label),
                "input_labeled": sum(1 for n in self.neurons.values() if n.input_label),
                "layers": sorted(set(n.layer for n in self.neurons.values())),
            }
        }
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
        if not quiet:
            if batch_count is not None:
                print(f"Saved {batch_count} neurons ({len(self.neurons)} total) to {self.db_path}")
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
        return [n for n in self.neurons.values()
                if n.primary_effect_type() == effect_type]


class ProgressiveInterpreter:
    """
    Interprets neurons progressively from output to input layers.

    Process:
    1. Load edge statistics and output projections
    2. Start with layer 31 (closest to output)
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

        # Cache for output projections
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

    def compute_output_projection(self, layer: int) -> torch.Tensor:
        """
        Compute output projection for all neurons in a layer.

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
        self,
        layer: int,
        neuron: int,
        top_k: int = 20
    ) -> tuple[list[TokenEffect], list[TokenEffect]]:
        """Get top promoted and suppressed tokens for a neuron."""
        self.load_model()

        logits = self.compute_output_projection(layer)[:, neuron]

        top_vals, top_idx = logits.topk(top_k)
        bot_vals, bot_idx = logits.topk(top_k, largest=False)

        promotes = []
        for val, idx in zip(top_vals, top_idx):
            token = self.tokenizer.decode([idx.item()])
            promotes.append(TokenEffect(
                token=token,
                token_id=idx.item(),
                logit_contribution=val.item()
            ))

        suppresses = []
        for val, idx in zip(bot_vals, bot_idx):
            token = self.tokenizer.decode([idx.item()])
            suppresses.append(TokenEffect(
                token=token,
                token_id=idx.item(),
                logit_contribution=val.item()
            ))

        return promotes, suppresses

    def get_neuron_output_norm(self, layer: int, neuron: int) -> float:
        """Get the norm of a neuron's output direction."""
        self.load_model()
        down_proj = self.model.model.layers[layer].mlp.down_proj.weight
        return down_proj[:, neuron].float().norm().item()

    def compute_input_projection(self, layer: int) -> torch.Tensor:
        """
        Compute input projection for all neurons in a layer.

        Returns tensor of shape [vocab_size, num_neurons] where each column
        shows how much each input token activates that neuron.

        Uses the combined formula: SiLU(embedding @ gate_proj) * (embedding @ up_proj)
        This accounts for Llama's gated MLP architecture where the final activation
        depends on both the gate and up projections.
        """
        cache_key = f"input_{layer}"
        if cache_key in self._output_projections:  # Reuse same cache dict
            return self._output_projections[cache_key]

        self.load_model()

        # Get weights
        up_proj = self.model.model.layers[layer].mlp.up_proj.weight.float()  # [d_ffn, d_model]
        gate_proj = self.model.model.layers[layer].mlp.gate_proj.weight.float()  # [d_ffn, d_model]
        embeddings = self.model.model.embed_tokens.weight.float()  # [vocab_size, d_model]

        # Compute projections: [vocab_size, d_ffn]
        up_vals = embeddings @ up_proj.T
        gate_vals = embeddings @ gate_proj.T

        # Combined: SiLU(gate) * up - this is the actual gated MLP formula
        input_sensitivity = torch.nn.functional.silu(gate_vals) * up_vals

        self._output_projections[cache_key] = input_sensitivity
        return input_sensitivity

    def get_neuron_input_sensitivity(
        self,
        layer: int,
        neuron: int,
        top_k: int = 20
    ) -> tuple[list[TokenEffect], list[TokenEffect]]:
        """
        Get top activating and suppressing input tokens for a neuron.

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
            activates.append(TokenEffect(
                token=token,
                token_id=idx.item(),
                logit_contribution=val.item()  # Reusing field name for consistency
            ))

        suppresses = []
        for val, idx in zip(bot_vals, bot_idx):
            token = self.tokenizer.decode([idx.item()])
            suppresses.append(TokenEffect(
                token=token,
                token_id=idx.item(),
                logit_contribution=val.item()
            ))

        return activates, suppresses

    def get_neuron_input_norm(self, layer: int, neuron: int) -> float:
        """Get the norm of a neuron's input sensitivity direction (up_proj)."""
        self.load_model()
        up_proj = self.model.model.layers[layer].mlp.up_proj.weight
        return up_proj[neuron, :].float().norm().item()

    def process_neuron_from_edge_stats(self, profile: dict) -> NeuronFunction:
        """
        Process a single neuron profile from edge stats.

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
            # Parse source ID: "layer_neuron_position" e.g., "30_1234_23"
            src_parts = source_id.split("_")
            if len(src_parts) >= 2:
                src_layer = int(src_parts[0])
                src_neuron = int(src_parts[1])
                src_id = f"L{src_layer}/N{src_neuron}"

                # Look up function label if we have it
                func_label = self.db.get_function_label(src_id)

                upstream.append(NeuronLink(
                    neuron_id=src_id,
                    layer=src_layer,
                    neuron_idx=src_neuron,
                    weight=src.get("avg_weight", 0),
                    frequency=src.get("frequency", 0),
                    function_label=func_label if func_label else None,
                ))

        # Process downstream targets
        downstream = []
        max_downstream_weight = 0
        for tgt in profile.get("top_downstream_targets", []):
            target_id = tgt["target"]
            tgt_parts = target_id.split("_")

            if len(tgt_parts) >= 2:
                # Check if it's a logit target (starts with "L_")
                if target_id.startswith("L_"):
                    # Direct logit connection
                    token_id = int(tgt_parts[1])
                    token = self.tokenizer.decode([token_id]) if self.tokenizer else f"token_{token_id}"
                    tgt_id = f"LOGIT/{token_id}"
                    tgt_layer = 32  # Virtual "output layer"
                    tgt_neuron = token_id
                    func_label = f"output token '{token}'"
                else:
                    tgt_layer = int(tgt_parts[0])
                    tgt_neuron = int(tgt_parts[1])
                    tgt_id = f"L{tgt_layer}/N{tgt_neuron}"
                    func_label = self.db.get_function_label(tgt_id)

                weight = abs(tgt.get("avg_weight", 0))
                max_downstream_weight = max(max_downstream_weight, weight)

                downstream.append(NeuronLink(
                    neuron_id=tgt_id,
                    layer=tgt_layer if not target_id.startswith("L_") else 32,
                    neuron_idx=tgt_neuron,
                    weight=tgt.get("avg_weight", 0),
                    frequency=tgt.get("frequency", 0),
                    function_label=func_label if func_label else None,
                ))

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

    def process_layer(self, layer: int, min_appearances: int = 100):
        """
        Process all neurons in a layer from edge stats.

        Args:
            layer: Layer number to process
            min_appearances: Minimum appearance count to include
        """
        self.load_edge_stats()
        if not self.edge_stats:
            print("No edge stats loaded.")
            return

        profiles = self.edge_stats.get("profiles", [])
        layer_profiles = [
            p for p in profiles
            if p["neuron_id"].startswith(f"L{layer}/")
            and p.get("appearance_count", 0) >= min_appearances
        ]

        print(f"\nProcessing layer {layer}: {len(layer_profiles)} neurons with >= {min_appearances} appearances")

        for i, profile in enumerate(layer_profiles):
            neuron_func = self.process_neuron_from_edge_stats(profile)
            self.db.set(neuron_func)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(layer_profiles)} neurons...")

        print(f"  Done. Layer {layer} neurons in database: {len(self.db.get_layer(layer))}")

    def process_all_layers(self, start_layer: int = 31, end_layer: int = 0, min_appearances: int = 100):
        """
        Process all layers from output to input.

        Args:
            start_layer: Start from this layer (typically 31, closest to output)
            end_layer: End at this layer (typically 0)
            min_appearances: Minimum appearance count to include
        """
        for layer in range(start_layer, end_layer - 1, -1):
            self.process_layer(layer, min_appearances)
            self.db.save()

    def process_neuron_connections_only(self, profile: dict) -> NeuronFunction:
        """
        Process a neuron profile from edge stats WITHOUT loading the model.
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
                # Handle embedding sources (E_tokenid_position)
                if source_id.startswith("E_"):
                    src_layer = -1  # Embedding layer
                    src_neuron = int(src_parts[1])
                    src_id = f"EMB/{src_neuron}"
                    func_label = f"embedding token {self.tokenizer.decode([src_neuron]) if self.tokenizer and src_neuron < 128000 else src_neuron}"
                else:
                    src_layer = int(src_parts[0])
                    src_neuron = int(src_parts[1])
                    src_id = f"L{src_layer}/N{src_neuron}"
                    func_label = self.db.get_function_label(src_id)
                upstream.append(NeuronLink(
                    neuron_id=src_id,
                    layer=src_layer,
                    neuron_idx=src_neuron,
                    weight=src.get("avg_weight", 0),
                    frequency=src.get("frequency", 0),
                    function_label=func_label if func_label else None,
                ))

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
                    func_label = f"output token {self.tokenizer.decode([tgt_neuron]) if self.tokenizer else tgt_neuron}"
                else:
                    tgt_layer = int(tgt_parts[0])
                    tgt_neuron = int(tgt_parts[1])
                    tgt_id = f"L{tgt_layer}/N{tgt_neuron}"
                    func_label = self.db.get_function_label(tgt_id)

                weight = abs(tgt.get("avg_weight", 0))
                max_downstream_weight = max(max_downstream_weight, weight)

                downstream.append(NeuronLink(
                    neuron_id=tgt_id,
                    layer=tgt_layer if not target_id.startswith("L_") else 32,
                    neuron_idx=tgt_neuron,
                    weight=tgt.get("avg_weight", 0),
                    frequency=tgt.get("frequency", 0),
                    function_label=func_label if func_label else None,
                ))

        # Extract output projection data from profile
        output_proj = profile.get("output_projection", {})
        promotes_raw = output_proj.get("promotes", output_proj.get("promoted", []))
        suppresses_raw = output_proj.get("suppresses", output_proj.get("suppressed", []))

        # Convert to expected format
        promotes = []
        suppresses = []
        max_logit_effect = 0.0

        for t in promotes_raw:
            weight = t.get("logit_contribution", t.get("weight", 0))
            promotes.append({
                "token": t.get("token", ""),
                "token_id": t.get("token_id", 0),
                "logit_contribution": weight,
            })
            max_logit_effect = max(max_logit_effect, abs(weight))

        for t in suppresses_raw:
            weight = t.get("logit_contribution", t.get("weight", 0))
            suppresses.append({
                "token": t.get("token", ""),
                "token_id": t.get("token_id", 0),
                "logit_contribution": weight,
            })
            max_logit_effect = max(max_logit_effect, abs(weight))

        # Get neurondb label (try multiple field names)
        neurondb_label = (
            profile.get("transluce_label_positive", "") or
            profile.get("max_act_label", "") or
            profile.get("neurondb_label", "")
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

    def process_all_connections(self, min_appearances: int = 100):
        """
        Process ALL neurons from edge stats without loading the model.
        Just stores connection data. Call compute_layer_projections() after.
        """
        self.load_edge_stats()
        if not self.edge_stats:
            print("No edge stats loaded.")
            return

        # Load tokenizer for downstream target labels
        if not self.tokenizer:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

        profiles = self.edge_stats.get("profiles", [])
        filtered = [p for p in profiles if p.get("appearance_count", 0) >= min_appearances]

        print(f"\nProcessing {len(filtered)} neurons (connections only, no model)...")

        for i, profile in enumerate(filtered):
            neuron_func = self.process_neuron_connections_only(profile)
            self.db.set(neuron_func)

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(filtered)} neurons...")
                self.db.save()

        self.db.save()
        print(f"Done. Total neurons in database: {len(self.db.neurons)}")

    def compute_layer_projections(self, layer: int):
        """
        Compute output projections for all neurons in a specific layer.
        Clears GPU cache after processing to manage memory.
        """
        import gc

        neurons = self.db.get_layer(layer)
        if not neurons:
            print(f"No neurons found in layer {layer}")
            return

        print(f"\nComputing output projections for layer {layer} ({len(neurons)} neurons)...")

        # Load model if needed
        self.load_model()

        # Compute projections for this layer (reuses cached projection matrix)
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

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Done with layer {layer}")

    def compute_all_projections(self, start_layer: int = 31, end_layer: int = 0):
        """
        Compute output projections for all layers, one layer at a time.
        Manages GPU memory by clearing cache between layers.
        """
        for layer in range(start_layer, end_layer - 1, -1):
            neurons = self.db.get_layer(layer)
            if neurons:
                self.compute_layer_projections(layer)

    def generate_layer_summary(self, layer: int) -> str:
        """Generate a summary of neurons in a layer."""
        neurons = self.db.get_layer(layer)
        if not neurons:
            return f"Layer {layer}: No neurons in database"

        # Group by effect type
        logit_dominant = [n for n in neurons if n.primary_effect_type() == "logit-dominant"]
        routing_dominant = [n for n in neurons if n.primary_effect_type() == "routing-dominant"]
        mixed = [n for n in neurons if n.primary_effect_type() == "mixed"]

        # Sort by logit effect magnitude
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
            label = n.function_label or "(unlabeled)"
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
                func = f" → {up.function_label}" if up.function_label else ""
                lines.append(f"  {up.neuron_id} (weight={up.weight:+.4f}, freq={up.frequency:.2f}){func}")

        lines.append("")
        lines.append("=== OUTPUT (what firing does) ===")

        if n.direct_logit_effects["promotes"]:
            lines.append("Direct logit effects - PROMOTES:")
            for t in n.direct_logit_effects["promotes"][:10]:
                lines.append(f"  {repr(t.token):20s} logit: {t.logit_contribution:+.4f}")

        if n.direct_logit_effects["suppresses"]:
            lines.append("Direct logit effects - SUPPRESSES:")
            for t in n.direct_logit_effects["suppresses"][:10]:
                lines.append(f"  {repr(t.token):20s} logit: {t.logit_contribution:+.4f}")

        if n.downstream_neurons:
            lines.append("")
            lines.append("Downstream neurons:")
            for down in sorted(n.downstream_neurons, key=lambda x: -abs(x.weight))[:10]:
                func = f" → {down.function_label}" if down.function_label else ""
                sign = "+" if down.weight > 0 else ""
                lines.append(f"  {down.neuron_id} (weight={sign}{down.weight:.4f}, freq={down.frequency:.2f}){func}")

        return "\n".join(lines)

    def set_function_label(self, neuron_id: str, label: str, description: str = "", confidence: str = "medium"):
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

    def propagate_labels(self):
        """
        Propagate function labels through the network.

        For each neuron with downstream connections, update the function_label
        field in the downstream NeuronLinks based on current database state.
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

    def auto_generate_label(self, neuron_id: str) -> str:
        """
        Auto-generate a function label based on output projection and downstream effects.

        For late layers: Focus on promoted/suppressed tokens
        For earlier layers: Focus on downstream neuron effects
        """
        n = self.db.get(neuron_id)
        if not n:
            return ""

        parts = []

        # Check effect type and magnitude
        effect_type = n.primary_effect_type()

        if effect_type in ("logit-dominant", "mixed") and n.logit_effect_magnitude > 0.05:
            # Describe by token effects
            promotes = n.direct_logit_effects.get("promotes", [])
            suppresses = n.direct_logit_effects.get("suppresses", [])

            if promotes:
                top_promoted = [t.token.strip() for t in promotes[:3] if t.token.strip()]
                if top_promoted:
                    # Check for patterns
                    if all(len(t) == 1 and t.isupper() for t in top_promoted):
                        parts.append("capital letter promoter")
                    elif any(t.lower() in ("the", "a", "an") for t in top_promoted):
                        parts.append("article promoter")
                    elif any(t.lower() in ("their", "its", "his", "her") for t in top_promoted):
                        parts.append("possessive promoter")
                    elif any(t in ("(", "[", "{") for t in top_promoted):
                        parts.append("parenthetical promoter")
                    elif any(t in (".", "!", "?") for t in top_promoted):
                        parts.append("sentence-end promoter")
                    else:
                        parts.append(f"'{top_promoted[0]}' promoter")

            if suppresses:
                top_suppressed = [t.token.strip() for t in suppresses[:3] if t.token.strip()]
                if top_suppressed:
                    if any(t.lower() in ("the", "a", "an") for t in top_suppressed):
                        parts.append("article suppressor")

        # For routing-dominant or if we have labeled downstream neurons
        if effect_type == "routing-dominant" or not parts:
            labeled_downstream = [
                d for d in n.downstream_neurons
                if d.function_label and not d.neuron_id.startswith("LOGIT/")
            ]

            if labeled_downstream:
                # Group by sign (positive = promotes, negative = suppresses)
                promotes_funcs = [d for d in labeled_downstream if d.weight > 0]
                suppresses_funcs = [d for d in labeled_downstream if d.weight < 0]

                if suppresses_funcs:
                    # Take the strongest suppression
                    strongest = max(suppresses_funcs, key=lambda x: abs(x.weight))
                    func_name = strongest.function_label.replace(" promoter", "")
                    parts.append(f"suppresses {func_name}")

                if promotes_funcs:
                    strongest = max(promotes_funcs, key=lambda x: abs(x.weight))
                    func_name = strongest.function_label.replace(" promoter", "")
                    parts.append(f"promotes {func_name}")

        if not parts:
            return ""

        return " / ".join(parts[:2])  # Limit to 2 parts

    def auto_label_layer(self, layer: int, min_magnitude: float = 0.1, overwrite: bool = False):
        """
        Auto-generate labels for all neurons in a layer.

        Args:
            layer: Layer to process
            min_magnitude: Minimum logit effect magnitude to label
            overwrite: Whether to overwrite existing labels
        """
        neurons = self.db.get_layer(layer)
        labeled = 0

        for n in neurons:
            if n.function_label and not overwrite:
                continue

            # Only auto-label neurons with significant effects
            if n.logit_effect_magnitude < min_magnitude and n.downstream_effect_magnitude < min_magnitude:
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

    def llm_generate_label(self, neuron_id: str, model: str = "gpt-5-mini") -> str:
        """
        Use LLM to generate a function label for a neuron.

        For terminal layers (L31): Uses output projection (promoted/suppressed tokens)
        For earlier layers: Also includes upstream/downstream neuron functions
        """
        if not OPENAI_AVAILABLE:
            print("OpenAI not available. Install with: pip install openai")
            return ""

        n = self.db.get(neuron_id)
        if not n:
            return ""

        client = OpenAI()

        # Build context about this neuron
        context_parts = []

        # Basic info
        context_parts.append(f"Neuron: {neuron_id} (Layer {n.layer})")
        context_parts.append(f"Appears in {n.appearance_count} prompts")
        if n.neurondb_label:
            context_parts.append(f"Max-activation label from NeuronDB: {n.neurondb_label}")

        # Output projection effects (what tokens it promotes/suppresses)
        if n.direct_logit_effects["promotes"]:
            promotes = [f"'{t.token}' ({t.logit_contribution:+.3f})"
                       for t in n.direct_logit_effects["promotes"][:15]]
            context_parts.append(f"\nDirect token effects - PROMOTES: {', '.join(promotes)}")

        if n.direct_logit_effects["suppresses"]:
            suppresses = [f"'{t.token}' ({t.logit_contribution:+.3f})"
                         for t in n.direct_logit_effects["suppresses"][:15]]
            context_parts.append(f"Direct token effects - SUPPRESSES: {', '.join(suppresses)}")

        # Downstream neurons (what this neuron affects)
        downstream_with_labels = [
            d for d in n.downstream_neurons
            if not d.neuron_id.startswith("LOGIT/")
        ]
        if downstream_with_labels:
            context_parts.append("\nDownstream neurons this neuron AFFECTS:")
            for d in sorted(downstream_with_labels, key=lambda x: -abs(x.weight))[:10]:
                sign = "promotes" if d.weight > 0 else "suppresses"
                func = f" (function: {d.function_label})" if d.function_label else ""
                context_parts.append(f"  - {sign} {d.neuron_id} (weight={d.weight:+.3f}){func}")

        # Upstream neurons (what activates this neuron)
        upstream_with_labels = [u for u in n.upstream_neurons]
        if upstream_with_labels:
            context_parts.append("\nUpstream neurons that ACTIVATE this neuron:")
            for u in sorted(upstream_with_labels, key=lambda x: -abs(x.weight))[:10]:
                sign = "excited by" if u.weight > 0 else "inhibited by"
                func = f" (function: {u.function_label})" if u.function_label else ""
                context_parts.append(f"  - {sign} {u.neuron_id} (weight={u.weight:+.3f}){func}")

        # Logit connections
        logit_connections = [
            d for d in n.downstream_neurons
            if d.neuron_id.startswith("LOGIT/")
        ]
        if logit_connections:
            context_parts.append("\nDirect connections to output logits:")
            for d in sorted(logit_connections, key=lambda x: -abs(x.weight))[:5]:
                token = d.function_label.replace("output token ", "") if d.function_label else d.neuron_id
                sign = "promotes" if d.weight > 0 else "suppresses"
                context_parts.append(f"  - {sign} {token} (weight={d.weight:+.3f})")

        context = "\n".join(context_parts)

        # Create prompt
        prompt = f"""You are analyzing neurons in the Llama-3.1-8B language model to understand their function.

{context}

Based on this information, generate a SHORT (3-7 words) functional label that describes what this neuron DOES.

Guidelines:
- Focus on the FUNCTION, not just what activates it
- If it promotes/suppresses specific token patterns, describe that
- If it mainly affects other neurons, describe it in terms of those effects
- Use patterns like: "X promoter", "X→Y switch", "suppresses X", "gates X to Y"
- Be specific but concise

Examples of good labels:
- "article→possessive switch"
- "sentence-start capital promoter"
- "suppresses code artifacts"
- "formal/academic style marker"
- "continuation vs new-sentence gate"

Respond with ONLY the label, no explanation."""

        try:
            # Use max_completion_tokens for newer models like gpt-5-mini
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,  # GPT-5-mini needs tokens for reasoning
            )
            label = response.choices[0].message.content.strip().strip('"\'')
            return label
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return ""

    def llm_generate_input_description(self, neuron_id: str, model: str = "gpt-5-mini") -> str:
        """
        Use LLM to generate an INPUT description - what makes this neuron fire.

        This is called on Pass 2 (early-to-late) when upstream neurons have labels.
        """
        if not OPENAI_AVAILABLE:
            print("OpenAI not available. Install with: pip install openai")
            return ""

        n = self.db.get(neuron_id)
        if not n:
            return ""

        client = OpenAI()

        # Build context focused on INPUT (what activates this neuron)
        context_parts = []

        context_parts.append(f"Neuron: {neuron_id} (Layer {n.layer})")
        context_parts.append(f"Appears in {n.appearance_count} prompts")

        if n.neurondb_label:
            context_parts.append(f"NeuronDB max-activation label: {n.neurondb_label}")

        # What this neuron DOES (already determined)
        if n.function_label:
            context_parts.append(f"\nThis neuron's OUTPUT function: {n.function_label}")

        # Upstream neurons (what activates this neuron) - THIS IS THE KEY PART
        upstream_with_labels = [u for u in n.upstream_neurons if u.function_label]
        upstream_without_labels = [u for u in n.upstream_neurons if not u.function_label]

        if upstream_with_labels:
            context_parts.append("\nUpstream neurons that ACTIVATE this neuron (with known functions):")
            for u in sorted(upstream_with_labels, key=lambda x: -abs(x.weight))[:10]:
                sign = "EXCITED by" if u.weight > 0 else "INHIBITED by"
                context_parts.append(f"  - {sign} '{u.function_label}' ({u.neuron_id}, weight={u.weight:+.4f})")

        if upstream_without_labels:
            context_parts.append("\nOther upstream neurons (functions unknown):")
            for u in sorted(upstream_without_labels, key=lambda x: -abs(x.weight))[:5]:
                sign = "excited by" if u.weight > 0 else "inhibited by"
                context_parts.append(f"  - {sign} {u.neuron_id} (weight={u.weight:+.4f})")

        # Also include some output context for reference
        if n.direct_logit_effects["promotes"]:
            promotes = [t.token for t in n.direct_logit_effects["promotes"][:5]]
            context_parts.append(f"\n(For reference - promotes tokens: {promotes})")

        context = "\n".join(context_parts)

        prompt = f"""You are analyzing what ACTIVATES a neuron in the Llama-3.1-8B language model.

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

        try:
            # Use max_completion_tokens for newer models like gpt-5-mini
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,  # GPT-5-mini needs tokens for reasoning
            )
            desc = response.choices[0].message.content.strip().strip('"\'')
            return desc
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return ""

    async def llm_input_description_layer_async(
        self,
        layer: int,
        model: str = "gpt-5-mini",
        min_upstream_labels: int = 2,
        overwrite: bool = False,
        batch_size: int = 64
    ):
        """
        Async batch input description generation for a layer.
        """
        import asyncio

        from openai import AsyncOpenAI

        neurons = self.db.get_layer(layer)
        if not neurons:
            print(f"No neurons found in layer {layer}")
            return

        # Filter neurons that need descriptions
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

        print(f"\nAsync generating input descriptions for layer {layer} ({len(to_label)} neurons)...")

        client = AsyncOpenAI()

        async def describe_neuron(n):
            """Generate input description for a single neuron."""
            prompt = self._build_input_description_prompt(n)
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=2000,  # GPT-5-mini needs tokens for reasoning
                )
                return n, response.choices[0].message.content.strip().strip('"\'')
            except Exception as e:
                print(f"  Error for {n.neuron_id}: {e}")
                return n, ""

        # Process in batches
        labeled = 0
        for i in range(0, len(to_label), batch_size):
            batch = to_label[i:i + batch_size]
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
            context_parts.append("\nUpstream neurons that ACTIVATE this neuron (with known functions):")
            for u in sorted(upstream_with_labels, key=lambda x: -abs(x.weight))[:10]:
                sign = "EXCITED by" if u.weight > 0 else "INHIBITED by"
                context_parts.append(f"  - {sign} '{u.function_label}' ({u.neuron_id}, weight={u.weight:+.4f})")

        if upstream_without_labels:
            context_parts.append("\nOther upstream neurons (functions unknown):")
            for u in sorted(upstream_without_labels, key=lambda x: -abs(x.weight))[:5]:
                sign = "excited by" if u.weight > 0 else "inhibited by"
                context_parts.append(f"  - {sign} {u.neuron_id} (weight={u.weight:+.4f})")

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

    def llm_input_description_layer(
        self,
        layer: int,
        model: str = "gpt-5-mini",
        min_upstream_labels: int = 2,
        overwrite: bool = False,
        use_async: bool = True
    ):
        """
        Generate input descriptions for neurons in a layer.

        Only processes neurons that have enough labeled upstream neurons.
        """
        if use_async:
            import asyncio
            asyncio.run(self.llm_input_description_layer_async(
                layer, model, min_upstream_labels, overwrite
            ))
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

            # Only generate if we have enough upstream labels
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

    async def llm_label_layer_async(
        self,
        layer: int,
        model: str = "gpt-5-mini",
        min_magnitude: float = 0.05,
        overwrite: bool = False,
        batch_size: int = 64
    ):
        """
        Async batch LLM labeling for a layer.
        """
        import asyncio

        from openai import AsyncOpenAI

        neurons = self.db.get_layer(layer)
        if not neurons:
            print(f"No neurons found in layer {layer}")
            return

        # Filter neurons that need labeling
        to_label = []
        for n in neurons:
            if n.function_label and not overwrite:
                continue
            if (n.logit_effect_magnitude < min_magnitude and
                n.downstream_effect_magnitude < min_magnitude):
                continue
            to_label.append(n)

        if not to_label:
            print(f"Layer {layer}: No neurons need labeling")
            return

        print(f"\nAsync LLM labeling layer {layer} ({len(to_label)} neurons to label)...")

        client = AsyncOpenAI()

        async def label_neuron(n):
            """Generate label for a single neuron."""
            prompt = self._build_label_prompt(n)
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=2000,  # GPT-5-mini needs tokens for reasoning
                )
                return n, response.choices[0].message.content.strip().strip('"\'')
            except Exception as e:
                print(f"  Error for {n.neuron_id}: {e}")
                return n, ""

        # Process in batches
        labeled = 0
        for i in range(0, len(to_label), batch_size):
            batch = to_label[i:i + batch_size]
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

    def _build_label_prompt(self, n: NeuronFunction) -> str:
        """Build the prompt for LLM labeling with layer-appropriate guidance."""
        context_parts = []
        layer = n.layer

        context_parts.append(f"Neuron: {n.neuron_id} (Layer {layer} of 31)")
        context_parts.append(f"Appears in {n.appearance_count} prompts")
        if n.neurondb_label:
            context_parts.append(f"Max-activation label from NeuronDB: {n.neurondb_label}")

        # Get downstream neurons with labels
        downstream_with_labels = [
            d for d in n.downstream_neurons
            if not d.neuron_id.startswith("LOGIT/") and d.function_label
        ]
        downstream_without_labels = [
            d for d in n.downstream_neurons
            if not d.neuron_id.startswith("LOGIT/") and not d.function_label
        ]

        # Get logit connections
        logit_connections = [
            d for d in n.downstream_neurons
            if d.neuron_id.startswith("LOGIT/")
        ]

        # For LATE layers (25-31): emphasize token effects
        if layer >= 25:
            if n.direct_logit_effects["promotes"]:
                promotes = [f"'{t.token}'" for t in n.direct_logit_effects["promotes"][:10]]
                context_parts.append(f"\nDirect token effects - PROMOTES: {', '.join(promotes)}")
            if n.direct_logit_effects["suppresses"]:
                suppresses = [f"'{t.token}'" for t in n.direct_logit_effects["suppresses"][:10]]
                context_parts.append(f"Direct token effects - SUPPRESSES: {', '.join(suppresses)}")
            if logit_connections:
                context_parts.append("\nStrong logit connections:")
                for d in sorted(logit_connections, key=lambda x: -abs(x.weight))[:5]:
                    token = d.function_label.replace("output token ", "") if d.function_label else d.neuron_id
                    sign = "+" if d.weight > 0 else "-"
                    context_parts.append(f"  {sign} {token}")

        # For EARLY/MID layers (0-24): emphasize downstream effects
        else:
            if downstream_with_labels:
                context_parts.append("\n*** KEY: Downstream neurons this ACTIVATES (with known functions) ***")
                for d in sorted(downstream_with_labels, key=lambda x: -abs(x.weight))[:8]:
                    sign = "ACTIVATES" if d.weight > 0 else "INHIBITS"
                    context_parts.append(f"  {sign}: \"{d.function_label}\" ({d.neuron_id}, w={d.weight:+.3f})")

            # Show token effects as secondary info
            if n.direct_logit_effects["promotes"]:
                promotes = [f"'{t.token}'" for t in n.direct_logit_effects["promotes"][:5]]
                context_parts.append(f"\n(Secondary - token effects: promotes {', '.join(promotes)})")

        context = "\n".join(context_parts)

        # Layer-specific instructions
        if layer >= 25:
            layer_guidance = """This is a LATE layer neuron (close to output). Describe its DIRECT effect on tokens.
Focus on what tokens it promotes/suppresses in the output vocabulary."""
            examples = """- "promotes dopamine and neurotransmitter tokens when medical context is active"
- "suppresses common articles like 'the' and 'a' in favor of specific nouns"
- "gates medical terminology completion by boosting drug name continuations"
- "promotes sentence-ending punctuation and suppresses mid-sentence tokens\""""
        elif layer >= 10:
            layer_guidance = """This is a MID layer neuron. Describe it as a BRIDGE or GATE between detection and output.
Your label MUST reference what downstream functions it activates/inhibits."""
            examples = """- "activates late-layer article promoters when sentence boundaries detected"
- "gates the chemical-term to dopamine-output pathway based on medical context"
- "bridges early entity detection to late-layer capitalization and formatting neurons"
- "inhibits code-formatting neurons when natural language context is strong\""""
        else:
            layer_guidance = """This is an EARLY layer neuron (close to input). Describe what DOWNSTREAM FUNCTIONS it triggers.
Your label MUST reference the downstream neuron functions it activates - NOT just tokens."""
            examples = """- "triggers mid-layer continuation promoters and sentence-boundary detectors"
- "activates medical-term processing pathway that feeds into drug-name completion"
- "feeds downstream sentence-boundary and punctuation-handling neurons"
- "initiates formal academic style pathway by activating article and structure neurons\""""

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

    def llm_label_layer(
        self,
        layer: int,
        model: str = "gpt-5-mini",
        min_magnitude: float = 0.05,
        overwrite: bool = False,
        use_async: bool = True
    ):
        """
        Use LLM to generate labels for all neurons in a layer.

        Args:
            layer: Layer to process
            model: OpenAI model to use
            min_magnitude: Minimum effect magnitude to label
            overwrite: Whether to overwrite existing labels
            use_async: Use async batch processing (faster)
        """
        if use_async:
            import asyncio
            asyncio.run(self.llm_label_layer_async(
                layer, model, min_magnitude, overwrite
            ))
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

            # Skip neurons with very weak effects
            if (n.logit_effect_magnitude < min_magnitude and
                n.downstream_effect_magnitude < min_magnitude):
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

    def llm_label_all_layers(
        self,
        start_layer: int = 31,
        end_layer: int = 0,
        model: str = "gpt-5-mini",
        passes: int = 2
    ):
        """
        Run multi-pass LLM labeling across all layers.

        Pass 1 (late-to-early): Generate function_labels (OUTPUT - what neuron does)
        Pass 2 (early-to-late): Generate input_descriptions (INPUT - what activates it)

        This is ADDITIVE - Pass 2 adds input_description without changing function_label.

        Args:
            start_layer: Start from this layer (typically 31)
            end_layer: End at this layer (typically 0)
            model: OpenAI model to use
            passes: Number of passes (2 recommended)
        """
        # PASS 1: Late-to-Early - Generate function_labels
        print(f"\n{'='*60}")
        print("PASS 1: Late-to-Early (generating function_labels)")
        print(f"{'='*60}")

        for layer in range(start_layer, end_layer - 1, -1):
            neurons = self.db.get_layer(layer)
            if not neurons:
                continue

            self.llm_label_layer(layer, model=model, overwrite=False)
            self.propagate_labels()

        if passes >= 2:
            # PASS 2: Early-to-Late - Generate input_descriptions
            print(f"\n{'='*60}")
            print("PASS 2: Early-to-Late (generating input_descriptions)")
            print(f"{'='*60}")

            for layer in range(end_layer, start_layer + 1):
                neurons = self.db.get_layer(layer)
                if not neurons:
                    continue

                self.llm_input_description_layer(layer, model=model, overwrite=False)
                self.propagate_labels()

        print(f"\nCompleted {passes} passes of LLM labeling")

        # Summary
        all_neurons = list(self.db.neurons.values())
        func_labeled = sum(1 for n in all_neurons if n.function_label)
        input_labeled = sum(1 for n in all_neurons if n.input_description)
        print(f"  Function labels: {func_labeled}/{len(all_neurons)}")
        print(f"  Input descriptions: {input_labeled}/{len(all_neurons)}")

    def generate_functional_summary(self, neuron_id: str) -> str:
        """
        Generate a natural language summary of a neuron's function,
        describing both INPUT (what activates it) and OUTPUT (what it does).
        """
        n = self.db.get(neuron_id)
        if not n:
            return f"Neuron {neuron_id} not found"

        lines = [f"**{neuron_id}**"]

        # Show labels if we have them
        if n.function_label:
            lines.append(f"OUTPUT function: {n.function_label}")
        if n.input_description:
            lines.append(f"INPUT condition: {n.input_description}")

        # Effect type
        effect_type = n.primary_effect_type()
        logit_mag = n.logit_effect_magnitude
        down_mag = n.downstream_effect_magnitude

        lines.append("")
        if effect_type == "logit-dominant":
            lines.append(f"Primary effect: Direct logit modification (mag={logit_mag:.3f})")
        elif effect_type == "routing-dominant":
            lines.append(f"Primary effect: Routing to downstream neurons (mag={down_mag:.3f})")
        else:
            lines.append(f"Mixed effects: logit={logit_mag:.3f}, downstream={down_mag:.3f}")

        # Describe OUTPUT effects
        lines.append("")
        lines.append("**When this neuron fires:**")

        # Direct logit effects
        if n.direct_logit_effects["promotes"] and logit_mag > 0.05:
            top3 = [t.token for t in n.direct_logit_effects["promotes"][:3]]
            lines.append(f"  - Directly promotes tokens: {top3}")

        if n.direct_logit_effects["suppresses"] and logit_mag > 0.05:
            top3 = [t.token for t in n.direct_logit_effects["suppresses"][:3]]
            lines.append(f"  - Directly suppresses tokens: {top3}")

        # Downstream neuron effects (with function labels!)
        labeled_downstream = [
            d for d in n.downstream_neurons
            if d.function_label and not d.neuron_id.startswith("LOGIT/")
        ]

        if labeled_downstream:
            promotes = [d for d in labeled_downstream if d.weight > 0]
            suppresses = [d for d in labeled_downstream if d.weight < 0]

            if promotes:
                for d in sorted(promotes, key=lambda x: -x.weight)[:3]:
                    lines.append(f"  - PROMOTES '{d.function_label}' (weight={d.weight:+.3f})")

            if suppresses:
                for d in sorted(suppresses, key=lambda x: x.weight)[:3]:
                    lines.append(f"  - SUPPRESSES '{d.function_label}' (weight={d.weight:+.3f})")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Progressive Neural Interpretation")
    parser.add_argument("--edge-stats", type=Path, help="Path to edge stats JSON")
    parser.add_argument("--db", type=Path, default=Path("data/neuron_function_db.json"),
                       help="Path to function database")
    parser.add_argument("--process-layer", type=int, help="Process a specific layer")
    parser.add_argument("--process-all", action="store_true", help="Process all layers")
    parser.add_argument("--start-layer", type=int, default=31, help="Start layer for --process-all")
    parser.add_argument("--end-layer", type=int, default=0, help="End layer for --process-all")
    parser.add_argument("--min-appearances", type=int, default=100, help="Minimum appearances to include")
    parser.add_argument("--describe", type=str, help="Describe a specific neuron (e.g., L31/N11000)")
    parser.add_argument("--layer-summary", type=int, help="Generate summary for a layer")
    parser.add_argument("--set-label", nargs=3, metavar=("NEURON_ID", "LABEL", "CONFIDENCE"),
                       help="Set function label for a neuron")
    parser.add_argument("--propagate", action="store_true", help="Propagate labels through network")
    parser.add_argument("--auto-label", type=int, help="Auto-generate labels for a layer")
    parser.add_argument("--auto-label-all", action="store_true", help="Auto-label all layers")
    parser.add_argument("--summary", type=str, help="Generate functional summary for a neuron")
    parser.add_argument("--llm-label", type=str, help="Use LLM to label a specific neuron")
    parser.add_argument("--llm-label-layer", type=int, help="Use LLM to label all neurons in a layer")
    parser.add_argument("--llm-label-all", action="store_true", help="Run multi-pass LLM labeling on all layers")
    parser.add_argument("--llm-model", type=str, default="gpt-5-mini", help="OpenAI model for labeling")
    parser.add_argument("--passes", type=int, default=2, help="Number of passes for multi-pass labeling")
    parser.add_argument("--process-connections", action="store_true",
                       help="Process all neurons from edge stats (connections only, no model)")
    parser.add_argument("--compute-projections", action="store_true",
                       help="Compute output projections for all neurons in database")
    parser.add_argument("--compute-layer-projections", type=int,
                       help="Compute output projections for a specific layer")

    args = parser.parse_args()

    interpreter = ProgressiveInterpreter(
        edge_stats_path=args.edge_stats,
        db_path=args.db,
    )

    if args.process_layer is not None:
        interpreter.process_layer(args.process_layer, args.min_appearances)
        interpreter.db.save()

    elif args.process_all:
        interpreter.process_all_layers(args.start_layer, args.end_layer, args.min_appearances)

    elif args.describe:
        print(interpreter.describe_neuron(args.describe))

    elif args.layer_summary is not None:
        print(interpreter.generate_layer_summary(args.layer_summary))

    elif args.set_label:
        neuron_id, label, confidence = args.set_label
        interpreter.set_function_label(neuron_id, label, confidence=confidence)
        interpreter.db.save()

    elif args.propagate:
        interpreter.propagate_labels()

    elif args.auto_label is not None:
        interpreter.auto_label_layer(args.auto_label)

    elif args.auto_label_all:
        for layer in range(31, -1, -1):
            neurons = interpreter.db.get_layer(layer)
            if neurons:
                interpreter.auto_label_layer(layer)
                interpreter.propagate_labels()

    elif args.summary:
        print(interpreter.generate_functional_summary(args.summary))

    elif args.llm_label:
        label = interpreter.llm_generate_label(args.llm_label, model=args.llm_model)
        if label:
            interpreter.set_function_label(args.llm_label, label, confidence="llm")
            interpreter.db.save()
            interpreter.propagate_labels()

    elif args.llm_label_layer is not None:
        interpreter.llm_label_layer(args.llm_label_layer, model=args.llm_model)
        interpreter.propagate_labels()

    elif args.llm_label_all:
        # Get the layer range from database
        layers = interpreter.db.neurons.values()
        if layers:
            max_layer = max(n.layer for n in layers)
            min_layer = min(n.layer for n in layers)
            interpreter.llm_label_all_layers(
                start_layer=max_layer,
                end_layer=min_layer,
                model=args.llm_model,
                passes=args.passes
            )

    elif args.process_connections:
        interpreter.process_all_connections(args.min_appearances)

    elif args.compute_projections:
        interpreter.compute_all_projections(args.start_layer, args.end_layer)

    elif args.compute_layer_projections is not None:
        interpreter.compute_layer_projections(args.compute_layer_projections)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
