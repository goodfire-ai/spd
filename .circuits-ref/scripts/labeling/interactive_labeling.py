#!/usr/bin/env python3
"""
Interactive CLI for Progressive Neuron Labeling.

A human-in-the-loop tool for labeling neurons layer-by-layer (L31->L0)
with the option to auto-complete layers asynchronously.

Usage:
    python scripts/interactive_labeling.py
    python scripts/interactive_labeling.py --resume
    python scripts/interactive_labeling.py --start-layer 25 --min-appearances 200
"""

import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

try:
    from openai import AsyncOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import from existing progressive_interp module
from progressive_interp import (
    NeuronFunction,
    ProgressiveInterpreter,
)


class NeuronStatus(Enum):
    """Status of a neuron in the labeling workflow."""
    PENDING = "pending"
    LABELED = "labeled"
    SKIPPED = "skipped"


@dataclass
class LabelingSession:
    """Tracks state of an interactive labeling session."""
    edge_stats_path: Path
    db_path: Path
    state_path: Path

    current_layer: int = 31
    current_index: int = 0

    neuron_status: dict = field(default_factory=dict)

    total_labeled: int = 0
    total_skipped: int = 0

    def save_state(self):
        """Save session state for resume."""
        state = {
            "edge_stats_path": str(self.edge_stats_path),
            "db_path": str(self.db_path),
            "current_layer": self.current_layer,
            "current_index": self.current_index,
            "neuron_status": {k: v.value for k, v in self.neuron_status.items()},
            "total_labeled": self.total_labeled,
            "total_skipped": self.total_skipped,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> bool:
        """Load session state. Returns True if state was loaded."""
        if not self.state_path.exists():
            return False
        try:
            with open(self.state_path) as f:
                state = json.load(f)
            self.current_layer = state["current_layer"]
            self.current_index = state["current_index"]
            self.neuron_status = {
                k: NeuronStatus(v) for k, v in state.get("neuron_status", {}).items()
            }
            self.total_labeled = state.get("total_labeled", 0)
            self.total_skipped = state.get("total_skipped", 0)
            return True
        except (json.JSONDecodeError, KeyError):
            return False


class InteractiveLabeler:
    """Interactive CLI for progressive neuron labeling."""

    # Baseline neurons to exclude from labeling (original set)
    BASELINE_NEURONS = {"L0/N491", "L0/N8268", "L0/N10585", "L1/N2427"}

    # Hub neurons - extremely common downstream/upstream targets that add noise
    # These are excluded from labeling AND filtered from connection lists
    # Loaded from data/hub_neurons.json if available, otherwise use hardcoded set
    HUB_NEURONS = {
        # Upstream hubs (appear in >20K upstream lists)
        "L0/N491", "L0/N8268", "L0/N10585", "L1/N198", "L1/N2427",
        "L3/N6390", "L4/N12934", "L12/N13860", "L15/N11853", "L24/N5326",
        # Downstream hubs (appear in >10K downstream lists)
        "L16/N8705", "L17/N7437", "L21/N5779",
        "L23/N13591", "L25/N11584", "L25/N3897", "L26/N2589",
        "L27/N8140", "L27/N13857", "L28/N11478", "L28/N447", "L29/N12010",
        "L29/N3490", "L30/N10321", "L30/N11430", "L30/N11920", "L30/N12395",
        "L30/N3382", "L30/N5442", "L30/N936", "L30/N1457",
    }

    def __init__(
        self,
        edge_stats_path: Path,
        db_path: Path,
        state_path: Path,
        model: str = "gpt-5",
        min_appearances: int = 10,
        batch_size: int = 800,
        browse_mode: bool = False,
        label_pass: str = "output",
        baseline_path: Path | None = None,
    ):
        self.console = Console()
        self.browse_mode = browse_mode
        self.model = model
        self.min_appearances = min_appearances
        self.batch_size = batch_size
        self.label_pass = label_pass  # "output" or "input"
        self.baseline_path = baseline_path

        # Only require OpenAI if not in browse mode
        if not browse_mode:
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI package not available. Install with: pip install openai")
            self.client = OpenAI()
            self.async_client = AsyncOpenAI()
        else:
            self.client = None
            self.async_client = None

        # Initialize interpreter (reuse existing infrastructure)
        self.interpreter = ProgressiveInterpreter(
            edge_stats_path=edge_stats_path,
            db_path=db_path,
        )
        self.interpreter.load_edge_stats()

        # Load baseline data for domain specificity comparison
        self._baseline_profiles: dict[str, dict] = {}
        if baseline_path and baseline_path.exists():
            with open(baseline_path) as f:
                baseline_data = json.load(f)
            for p in baseline_data.get("profiles", []):
                self._baseline_profiles[p["neuron_id"]] = p

        # Load tokenizer for decoding tokens
        if not self.interpreter.tokenizer:
            from transformers import AutoTokenizer
            self.interpreter.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct"
            )

        # Session state
        self.session = LabelingSession(
            edge_stats_path=edge_stats_path,
            db_path=db_path,
            state_path=state_path,
        )

        # Cache layer neurons
        self._layer_neurons: dict[int, list[dict]] = {}
        # Index profiles by neuron_id for Transluce label fallback
        self._profile_by_id: dict[str, dict] = {}
        self._build_layer_cache()

    def _get_neuron_label_with_fallback(self, neuron_id: str) -> str | None:
        """
        Get label for a neuron, with fallback to Transluce labels.

        Priority:
        1. Two-pass label from JSON database (function_label)
        2. Transluce label from edge_stats profile (transluce_label_positive)
        3. NeuronDB max-activation label (max_act_label)

        This ensures upstream neurons have labels even if not in the 45k labeled set.
        """
        # First try: JSON database (two-pass labels)
        label = self.interpreter.db.get_function_label(neuron_id)
        if label:
            return label

        # Second try: Transluce label from edge_stats profile
        profile = self._profile_by_id.get(neuron_id)
        if profile:
            transluce_label = profile.get("transluce_label_positive")
            if transluce_label:
                return f"[Transluce] {transluce_label}"

            # Third try: NeuronDB max-activation label
            max_act = profile.get("max_act_label")
            if max_act:
                return f"[NeuronDB] {max_act}"

        return None

    def _build_layer_cache(self):
        """Build cache of neurons by layer from edge stats."""
        profiles = self.interpreter.edge_stats.get("profiles", [])

        # Build profile index for quick lookup (used for Transluce fallback)
        self._profile_by_id = {p["neuron_id"]: p for p in profiles}

        # Combined exclusion set: baseline + hub neurons
        excluded_neurons = self.BASELINE_NEURONS | self.HUB_NEURONS

        for p in profiles:
            if p.get("appearance_count", 0) < self.min_appearances:
                continue
            if p["neuron_id"] in excluded_neurons:
                continue

            layer = p["layer"]
            if layer not in self._layer_neurons:
                self._layer_neurons[layer] = []
            self._layer_neurons[layer].append(p)

        # Sort each layer by appearance count (most frequent first)
        for layer in self._layer_neurons:
            self._layer_neurons[layer].sort(
                key=lambda x: -x.get("appearance_count", 0)
            )

    def get_layer_neurons(self, layer: int) -> list[dict]:
        """Get neurons for a layer, filtered and sorted."""
        return self._layer_neurons.get(layer, [])

    def get_layer_progress(self, layer: int) -> tuple[int, int, int]:
        """Get (labeled, skipped, total) for a layer."""
        neurons = self.get_layer_neurons(layer)
        labeled = sum(
            1 for n in neurons
            if self.session.neuron_status.get(n["neuron_id"]) == NeuronStatus.LABELED
        )
        skipped = sum(
            1 for n in neurons
            if self.session.neuron_status.get(n["neuron_id"]) == NeuronStatus.SKIPPED
        )
        return labeled, skipped, len(neurons)

    def get_domain_specificity_info(self, profile: dict) -> str | None:
        """
        Compute domain specificity by comparing primary vs baseline corpus.

        Returns a description string if baseline data is available, else None.

        NOTE: This function uses generic language. Do NOT add domain-specific terms
        like "medical" to labels unless the actual projection tokens support it.
        """
        if not self._baseline_profiles:
            return None

        nid = profile["neuron_id"]
        primary_count = profile.get("appearance_count", 0)
        baseline_profile = self._baseline_profiles.get(nid)

        if not baseline_profile:
            # Neuron only appears in primary corpus, not baseline
            if primary_count >= 10:
                return f"DOMAIN SPECIFICITY: This neuron appears {primary_count}x in primary corpus but NOT in baseline - may be domain-specific. Label based on actual token patterns, not assumed domain."
            return None

        baseline_count = baseline_profile.get("appearance_count", 0)

        if baseline_count == 0:
            if primary_count >= 10:
                return f"DOMAIN SPECIFICITY: {primary_count}x in primary corpus, 0x in baseline - may be domain-specific. Label based on actual tokens."
            return None

        ratio = primary_count / baseline_count

        if ratio >= 5:
            return f"DOMAIN SPECIFICITY: {primary_count}x primary vs {baseline_count}x baseline (ratio={ratio:.1f}x) - corpus-biased pattern. Focus on what the actual promoted/suppressed tokens are, not the corpus domain."
        elif ratio <= 0.2:
            return f"DOMAIN SPECIFICITY: {primary_count}x primary vs {baseline_count}x baseline (ratio={ratio:.2f}x) - GENERAL pattern across corpora."
        elif 0.5 <= ratio <= 2:
            return f"DOMAIN SPECIFICITY: {primary_count}x primary vs {baseline_count}x baseline (ratio={ratio:.1f}x) - BALANCED pattern across corpora."

        return None

    def build_prompt_for_neuron(self, profile: dict) -> str:
        """
        Build the LLM prompt for a neuron based on labeling pass.

        Pass 1 (output): What does this neuron DO when it fires?
        Pass 2 (input): What TRIGGERS this neuron to fire?
        """
        neuron_id = profile["neuron_id"]

        # First check if we have pre-computed data in the database
        neuron_func = self.interpreter.db.get(neuron_id)

        if neuron_func is None:
            # Fall back to processing from edge stats (no output projection)
            neuron_func = self.interpreter.process_neuron_connections_only(profile)

        if self.label_pass == "input":
            return self._build_input_label_prompt(neuron_func, profile)
        else:
            return self._build_output_label_prompt(neuron_func, profile)

    def parse_structured_response(self, response: str) -> dict:
        """
        Parse the structured LLM response into a dict with fields.

        For output pass:
        - interpretability, function_type, short_label, output_function

        For input pass:
        - interpretability, function_type, short_label, input_function

        Handles markdown formatting (**, etc.) that LLMs often add.
        """
        import re

        result = {
            "interpretability": "medium",
            "function_type": "",
            "short_label": "",
            "output_function": "",
            "input_function": "",
        }

        # Try to find the "Final Answer" or "Final Assessment" section first
        final_section = response
        if "final a" in response.lower():  # Matches both "final answer" and "final assessment"
            parts = re.split(r'(?i)\*?\*?[Ff]inal [Aa](nswer|ssessment)\*?\*?:?', response)
            if len(parts) > 1:
                final_section = parts[-1]

        # Clean markdown formatting
        clean_response = re.sub(r'\*+', '', final_section)

        # Parse with flexible patterns
        # Note: short_label captures until end of line (labels should be single-line)
        # but allows quotes within the label (e.g., 'promotes "art" tokens')
        patterns = {
            "interpretability": r'(?i)INTERPRETABILITY[:\s]+(\w+)',
            "function_type": r'(?i)\bTYPE[:\s]+([\w-]+)',
            "short_label": r'(?i)SHORT[_\s]?LABEL[:\s]+([^\n]+)',
            "output_function": r'(?i)OUTPUT[_\s]?FUNCTION[:\s]+(.+?)(?=\n[A-Z_]+:|$)',
            "input_function": r'(?i)INPUT[_\s]?FUNCTION[:\s]+(.+?)(?=\n[A-Z_]+:|$)',
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, clean_response, re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Clean up quotes that may wrap the value
                if field == "short_label":
                    value = value.strip('"\'')
                    # Clean up common parsing artifacts
                    value = self._clean_label(value)
                if field == "interpretability":
                    value = value.lower()
                    if value not in ("low", "medium", "high"):
                        value = "medium"
                if field in ("output_function", "input_function"):
                    # Clean up description: strip leading dashes, normalize whitespace
                    value = value.strip()
                    value = value.lstrip('-•* ')
                    # Replace multiple whitespace with single space
                    value = ' '.join(value.split())
                result[field] = value

        # Fallback: if parsing failed, use raw response
        if not result["short_label"] and not result["output_function"] and not result["input_function"]:
            result["short_label"] = response.strip()[:50]
            if self.label_pass == "input":
                result["input_function"] = response.strip()
            else:
                result["output_function"] = response.strip()

        return result

    def _clean_label(self, label: str) -> str:
        """
        Clean up common parsing artifacts from labels.

        Handles:
        - Leading dashes or bullets ("- label" -> "label")
        - Explanatory preambles ("Based on the data, this is a..." -> extract actual label)
        - Overly long labels (truncate at reasonable word boundary)
        - Newlines (replace with space)
        """
        import re

        # Replace newlines with spaces
        label = label.replace('\n', ' ').replace('\r', ' ')

        # Strip leading dashes/bullets
        label = re.sub(r'^[-•*]\s*', '', label)

        # Remove common preambles
        preambles = [
            r'^Based on (?:the )?(?:promoted|suppressed|output|projection)[^:,]*[,:]\s*',
            r'^(?:The|This) neuron (?:is|functions as|acts as)[^:]*[,:]\s*',
            r'^Considering (?:the )?[^:,]*[,:]\s*',
            r'^Given (?:the )?[^:,]*[,:]\s*',
        ]
        for preamble in preambles:
            label = re.sub(preamble, '', label, flags=re.IGNORECASE)

        # If still starts with lowercase "the"/"this", strip it
        label = re.sub(r'^(?:the|this)\s+', '', label, flags=re.IGNORECASE)

        # Truncate if too long (>60 chars), try to cut at word boundary
        if len(label) > 60:
            # Find last space before char 60
            cut_point = label.rfind(' ', 0, 60)
            if cut_point > 30:  # Only cut at word if we keep >30 chars
                label = label[:cut_point]
            else:
                label = label[:60]

        return label.strip()

    def _build_output_label_prompt(self, n: NeuronFunction, profile: dict) -> str:
        """
        Build prompt for Pass 1 (late-to-early): OUTPUT function only.

        Describes what the neuron DOES when it fires, NOT what triggers it.
        """
        context_parts = []
        layer = n.layer

        # Basic info with layer context
        context_parts.append(f"Neuron: {n.neuron_id} (Layer {layer} of 31)")
        if layer == 31:
            context_parts.append("*** THIS IS THE FINAL LAYER - neurons here ONLY affect output logits directly ***")
            context_parts.append("*** There are NO downstream neurons. 'routing' type is NOT applicable for L31. ***")

        # Direct effect ratio - important for understanding neuron's role
        der = profile.get("direct_effect_ratio", {})
        if der:
            effect_type = der.get("effect_type", "unknown")
            mean = der.get("mean", 0)
            context_parts.append(f"Direct effect ratio: {mean:.1%} ({effect_type} neuron)")
            if effect_type == "routing":
                context_parts.append("  -> This neuron primarily affects output INDIRECTLY through downstream neurons")
            elif effect_type == "logit":
                context_parts.append("  -> This neuron primarily affects output DIRECTLY through logit contributions")

        # Domain specificity (medical vs general corpus comparison)
        domain_info = self.get_domain_specificity_info(profile)
        if domain_info:
            context_parts.append(f"\n{domain_info}")

        # Transluce labels (what contexts activate/suppress this neuron)
        transluce_pos = profile.get("transluce_label_positive", "")
        transluce_neg = profile.get("transluce_label_negative", "")
        if transluce_pos:
            context_parts.append(f"Activates on (input pattern): {transluce_pos}")
        if transluce_neg:
            context_parts.append(f"Suppressed by (input pattern): {transluce_neg}")

        # Fallback to old NeuronDB label format
        neurondb_label = profile.get("max_act_label", "") or n.neurondb_label
        if neurondb_label and not transluce_pos:
            context_parts.append(f"NeuronDB label (activation context): {neurondb_label}")

        # Get downstream neurons with labels (look up CURRENT labels from db, with Transluce fallback)
        # Deduplicate by neuron_id, keeping the strongest weight
        # FILTER OUT hub neurons - they're too common to be informative
        downstream_by_id = {}
        for d in n.downstream_neurons:
            if d.neuron_id.startswith("LOGIT/"):
                continue
            if d.neuron_id in self.HUB_NEURONS:
                continue  # Skip hub neurons
            # Look up label with fallback: JSON db -> Transluce -> NeuronDB
            current_label = self._get_neuron_label_with_fallback(d.neuron_id)
            if current_label:
                # Keep entry with strongest weight for each neuron_id
                if d.neuron_id not in downstream_by_id or abs(d.weight) > abs(downstream_by_id[d.neuron_id][0].weight):
                    downstream_by_id[d.neuron_id] = (d, current_label)
        downstream_with_labels = list(downstream_by_id.values())

        # === OUTPUT EFFECTS ===
        context_parts.append("\n=== OUTPUT EFFECTS (what this neuron DOES when it fires) ===")

        # Output projection (context-independent) - always show both promotes and suppresses
        # First try from NeuronFunction, then from profile's output_projection
        promotes = n.direct_logit_effects.get("promotes", [])
        suppresses = n.direct_logit_effects.get("suppresses", [])

        # If not in NeuronFunction, check profile (support both key formats)
        if not promotes and not suppresses:
            output_proj = profile.get("output_projection", {})
            promotes = output_proj.get("promotes", output_proj.get("promoted", []))
            suppresses = output_proj.get("suppresses", output_proj.get("suppressed", []))

        def get_token_contrib(t):
            """Extract token and contribution from TokenEffect or dict."""
            if hasattr(t, 'token'):
                return t.token, t.logit_contribution
            elif isinstance(t, dict):
                # Support both 'logit_contribution' and 'weight' keys
                return t.get('token', '?'), t.get('logit_contribution', t.get('weight', 0))
            return '?', 0

        if promotes or suppresses:
            output_norm = profile.get("output_projection", {}).get("output_norm", 0)
            context_parts.append(f"\nStatic output projection (context-independent, norm={output_norm:.3f}):")

            if promotes:
                context_parts.append("  PROMOTES:")
                for t in promotes[:5]:
                    token, contrib = get_token_contrib(t)
                    context_parts.append(f"    {repr(token):25s} ({contrib:+.4f})")

            if suppresses:
                context_parts.append("  SUPPRESSES:")
                for t in suppresses[:5]:
                    token, contrib = get_token_contrib(t)
                    context_parts.append(f"    {repr(token):25s} ({contrib:+.4f})")

        # Downstream neuron effects
        if downstream_with_labels:
            context_parts.append("\n*** Downstream neurons with KNOWN functions ***")
            for d, current_label in sorted(downstream_with_labels, key=lambda x: -abs(x[0].weight))[:8]:
                sign = "ACTIVATES" if d.weight > 0 else "INHIBITS"
                context_parts.append(f"  {sign}: \"{current_label}\" ({d.neuron_id}, w={d.weight:+.3f})")

        # Downstream without labels (for context) - check with Transluce fallback, deduplicated
        # FILTER OUT hub neurons
        unlabeled_by_id = {}
        for d in n.downstream_neurons:
            if d.neuron_id.startswith("LOGIT/"):
                continue
            if d.neuron_id in self.HUB_NEURONS:
                continue  # Skip hub neurons
            if self._get_neuron_label_with_fallback(d.neuron_id):
                continue  # Has a label (from JSON db or Transluce)
            if d.neuron_id not in unlabeled_by_id or abs(d.weight) > abs(unlabeled_by_id[d.neuron_id].weight):
                unlabeled_by_id[d.neuron_id] = d
        downstream_without_labels = list(unlabeled_by_id.values())

        # Check if this neuron only connects to hubs (hub-only neuron)
        is_hub_only = len(downstream_with_labels) == 0 and len(downstream_without_labels) == 0

        if downstream_without_labels and not downstream_with_labels:
            context_parts.append("\nDownstream neurons (not yet labeled):")
            for d in sorted(downstream_without_labels, key=lambda x: -abs(x.weight))[:5]:
                sign = "+" if d.weight > 0 else "-"
                context_parts.append(f"  {sign} {d.neuron_id} (w={d.weight:+.3f}, freq={d.frequency:.1%})")

        if is_hub_only:
            context_parts.append("\n*** This neuron only connects to common hub neurons (filtered out). ***")
            context_parts.append("*** Focus on the OUTPUT PROJECTION and Transluce labels above. ***")

        context = "\n".join(context_parts)

        # Compute max projection weight for interpretability hint
        max_proj_weight = 0.0
        output_proj = profile.get("output_projection") or {}
        for t in output_proj.get("promoted", [])[:10]:
            max_proj_weight = max(max_proj_weight, abs(t.get("weight", 0)))
        for t in output_proj.get("suppressed", [])[:10]:
            max_proj_weight = max(max_proj_weight, abs(t.get("weight", 0)))

        # Guidance based on direct effect ratio (not layer)
        der = profile.get("direct_effect_ratio") or {}
        effect_type = der.get("effect_type", "unknown")

        # Compute magnitude hint (used by multiple branches)
        if max_proj_weight >= 0.20:
            magnitude_hint = f"Projection magnitude is relatively HIGH ({max_proj_weight:.2f}) - there's a decent chance this neuron has an interpretable pattern."
        elif max_proj_weight >= 0.10:
            magnitude_hint = f"Projection magnitude is MODERATE ({max_proj_weight:.2f}) - look carefully for patterns, but they may be subtle."
        else:
            magnitude_hint = f"Projection magnitude is LOW ({max_proj_weight:.2f}) - patterns may be weak or absent. Consider marking as uninterpretable if tokens appear random."

        if effect_type == "logit":
            effect_guidance = f"""This neuron has HIGH direct effect on logits.
Focus on the OUTPUT PROJECTION tokens above - what vocabulary does this neuron promote/suppress?
Look for patterns in the actual tokens shown (semantic themes, formatting, letter patterns, morphemes, etc.).

Note: A neuron's effect may be primarily PROMOTION, primarily SUPPRESSION, or both.
Sometimes only one side has an interpretable pattern (e.g., clear promotions but random suppressions).
If only one side is interpretable, focus your description on that side.

{magnitude_hint}"""
        elif effect_type == "routing" and layer < 31:
            # Routing only applies to non-final layers
            # Check if we have labeled downstream neurons
            has_labeled_downstream = len(downstream_with_labels) > 0

            if has_labeled_downstream:
                effect_guidance = """This neuron has LOW direct effect (routing neuron).
IMPORTANT: The output projection tokens shown above are largely NOISE for routing neurons.
Do NOT describe the projection - it doesn't reflect what this neuron actually does.

Instead, focus on the DOWNSTREAM NEURONS with known functions listed above.
Describe how this neuron orchestrates or gates those downstream functions."""
            else:
                effect_guidance = """This neuron has LOW direct effect (routing neuron).
IMPORTANT: The output projection tokens shown above are largely NOISE for routing neurons.
Do NOT try to interpret the projection - it doesn't reflect what this neuron actually does.

This neuron works through downstream neurons, but none of them are labeled yet.
Without knowing what the downstream neurons do, this neuron is NOT YET INTERPRETABLE.
Mark as: INTERPRETABILITY: low, TYPE: routing, SHORT_LABEL: uninterpretable-routing, OUTPUT_FUNCTION: Routing neuron with unlabeled downstream targets."""
        elif layer == 31:
            # L31 is final layer - all neurons affect logits directly regardless of direct_effect_ratio
            effect_guidance = f"""This is the FINAL LAYER (L31) - this neuron can ONLY affect output logits directly.
There are no downstream neurons. Focus entirely on the OUTPUT PROJECTION tokens above.
Look for patterns in the actual tokens shown (semantic themes, formatting, letter patterns, morphemes, etc.).

Note: A neuron's effect may be primarily PROMOTION, primarily SUPPRESSION, or both.
Sometimes only one side has an interpretable pattern. If only one side is interpretable, focus on that.

{magnitude_hint}"""
        else:
            effect_guidance = """Look at both the output projection AND downstream connections.
Consider whether this neuron affects output directly or through other neurons."""

        return f"""You are analyzing a neuron in Llama-3.1-8B to understand its OUTPUT FUNCTION.

{context}

{effect_guidance}

Analyze the neuron and provide a structured response with these four fields:

1. INTERPRETABILITY: How confident are you that this neuron has a clear, UNIFYING pattern?
   - "high" = Clear, obvious unifying pattern (e.g., all promoted tokens are anatomy terms, or all start with "Ch-")
   - "medium" = Likely pattern but some noise mixed in (e.g., 4/5 tokens fit a theme, 1 is random)
   - "low" = No unifying pattern - tokens appear unrelated or random

   EXAMPLES:
   - HIGH: Suppresses [' muscle', ' abdominal', ' stomach', ' spinal'] → clear "anatomy terms" pattern
   - HIGH: Promotes [' Green', ' green', 'Green', 'GREEN'] → clear "green" pattern
   - MEDIUM: Promotes [' A', ' L', ' D', ' T', '.scalablytyped'] → mostly capital letters, one outlier
   - LOW: Promotes ['(水', '(日', 'jedn', '!\n\n\n\n', ' Vš'] → no unifying theme, just listing tokens

   IMPORTANT: If you cannot identify ONE unifying theme, mark as LOW.
   Do NOT describe neurons as "promotes X and Y and Z" listing unrelated tokens - that's LOW interpretability.

2. TYPE: What category best describes this neuron's OUTPUT effect on the vocabulary?
   - "semantic" = Promotes/suppresses tokens with shared meaning (topic words, emotions, concepts)
   - "formatting" = Affects punctuation, whitespace, newlines, capitalization
   - "structural" = Sentence boundaries, list markers, section breaks
   - "lexical" = Letter/character patterns (e.g., words starting with "Ch-", or single letters)
   - "routing" = Works through downstream neurons (ONLY for layers < 31, NOT for final layer)
   - "associative" = Connects one concept (upstream) to another concept (downstream)
   - "unknown" = Cannot determine
   NOTE: TYPE describes the OUTPUT effect, not what triggers/activates the neuron.

3. SHORT_LABEL: A brief 3-8 word label summarizing the neuron's function.
   - If interpretable: describe the pattern (e.g., "capital-letter-promoter", "ellipsis-continuation-marker")
   - If low interpretability: write "uninterpretable" or "uninterpretable-routing"
   - IMPORTANT: Label based on the ACTUAL tokens shown, not assumed domains.

4. OUTPUT_FUNCTION: A 1-2 sentence description of what this neuron DOES when it fires.
   - Focus on the OUTPUT effect (what tokens it promotes/suppresses)
   - A neuron's effect may be primarily PROMOTION, primarily SUPPRESSION, or both. Sometimes only one side has an interpretable pattern. If only one side is interpretable, focus on that.
   - Do NOT describe what activates the neuron (that's the input function, not output)
   - If uninterpretable, explain why (e.g., "No clear pattern in promoted tokens")

First, generate THREE hypotheses at different specificity levels:

HYPOTHESIS_SPECIFIC: [Most specific interpretation - e.g., "promotes 'green' color token"]
HYPOTHESIS_MEDIUM: [Medium specificity - e.g., "promotes color-related terms"]
HYPOTHESIS_GENERAL: [Most general interpretation - e.g., "promotes adjectives"]

For each hypothesis, what evidence would CONTRADICT it?
CONTRADICT_SPECIFIC: [What would disprove the specific hypothesis?]
CONTRADICT_MEDIUM: [What would disprove the medium hypothesis?]
CONTRADICT_GENERAL: [What would disprove the general hypothesis?]

Based on the evidence above, which hypothesis best fits ALL the data? Consider:
- Does the pattern hold across ALL promoted/suppressed tokens, or just some?
- Is the specificity justified by the evidence, or are you over-fitting to examples?
- Would a more general pattern explain the same observations?

Now provide your final answer:
INTERPRETABILITY: [low/medium/high]
TYPE: [semantic/formatting/structural/lexical/routing/associative/unknown]
SHORT_LABEL: [3-8 word label - use the chosen hypothesis level]
OUTPUT_FUNCTION: [detailed description based on chosen hypothesis]"""

    def _build_input_label_prompt(self, n: NeuronFunction, profile: dict) -> str:
        """
        Build prompt for Pass 2 (early-to-late): INPUT function.

        Describes what TRIGGERS this neuron to fire, NOT what it does when firing.
        Uses:
        - Transluce labels (activation patterns from max-activating examples)
        - Upstream neurons with their output labels from Pass 1
        - Input token associations from edge stats
        """
        context_parts = []
        layer = n.layer

        # Basic info
        context_parts.append(f"Neuron: {n.neuron_id} (Layer {layer} of 31)")

        # Show the OUTPUT function we already determined (for context)
        if n.function_label:
            context_parts.append(f"\n*** KNOWN OUTPUT FUNCTION: {n.function_label} ***")
            if n.function_description:
                context_parts.append(f"    {n.function_description}")
            context_parts.append("(Your task: determine what INPUT conditions trigger this output behavior)")

        # Domain specificity (medical vs general corpus comparison)
        domain_info = self.get_domain_specificity_info(profile)
        if domain_info:
            context_parts.append(f"\n{domain_info}")

        # === ACTIVATION PATTERNS (from Transluce) ===
        context_parts.append("\n=== ACTIVATION PATTERNS (what contexts activate this neuron) ===")

        transluce_pos = profile.get("transluce_label_positive", "")
        transluce_neg = profile.get("transluce_label_negative", "")

        if transluce_pos:
            context_parts.append(f"ACTIVATES on: {transluce_pos}")
        if transluce_neg:
            context_parts.append(f"SUPPRESSED by: {transluce_neg}")

        # Fallback to old NeuronDB label
        neurondb_label = profile.get("max_act_label", "") or n.neurondb_label
        if neurondb_label and not transluce_pos:
            context_parts.append(f"NeuronDB (max-activating contexts): {neurondb_label}")

        if not transluce_pos and not transluce_neg and not neurondb_label:
            context_parts.append("(No activation pattern data available)")

        # === INPUT PROJECTION (static token sensitivity) ===
        # Use precomputed projection from edge stats, or compute on-the-fly
        input_proj = profile.get("input_projection", {})
        activates = input_proj.get("activates", [])
        suppresses_input = input_proj.get("suppresses", [])

        if not activates and not suppresses_input:
            # Try to compute from model (if available) - uses SiLU(gate)*up formula
            try:
                act_effects, sup_effects = self.interpreter.get_neuron_input_sensitivity(
                    layer, n.neuron_idx, top_k=8
                )
                activates = [{"token": t.token, "weight": t.logit_contribution} for t in act_effects]
                suppresses_input = [{"token": t.token, "weight": t.logit_contribution} for t in sup_effects]
            except Exception:
                pass  # Model not loaded, skip input projection

        if activates or suppresses_input:
            context_parts.append("\n=== INPUT PROJECTION (static: SiLU(gate)*up formula) ===")
            context_parts.append("Which vocabulary tokens would most activate this neuron based on MLP weights:")

            if activates:
                context_parts.append("  ACTIVATING tokens:")
                for t in activates[:5]:
                    context_parts.append(f"    {repr(t['token']):25s} ({t['weight']:+.6f})")

            if suppresses_input:
                context_parts.append("  SUPPRESSING tokens:")
                for t in suppresses_input[:5]:
                    context_parts.append(f"    {repr(t['token']):25s} ({t['weight']:+.6f})")

        # === UPSTREAM NEURONS ===
        # Get upstream neurons with labels (look up CURRENT labels from db, or use embedded label for EMB)
        # FILTER OUT hub neurons - they're too common to be informative
        upstream_by_id = {}
        for u in n.upstream_neurons:
            # Skip hub neurons
            if u.neuron_id in self.HUB_NEURONS:
                continue
            # For embeddings, use the function_label set during processing (contains decoded token)
            # For neurons, look up label with fallback to Transluce labels
            if u.neuron_id.startswith("EMB/"):
                current_label = u.function_label or self._decode_embedding_label(u.neuron_id)
            else:
                # Use fallback: JSON db -> Transluce -> NeuronDB max_act
                current_label = self._get_neuron_label_with_fallback(u.neuron_id)
            if u.neuron_id not in upstream_by_id or abs(u.weight) > abs(upstream_by_id[u.neuron_id][0].weight):
                upstream_by_id[u.neuron_id] = (u, current_label)
        upstream_with_labels = [(u, lbl) for u, lbl in upstream_by_id.values() if lbl]
        upstream_without_labels = [(u, lbl) for u, lbl in upstream_by_id.values() if not lbl]

        if upstream_with_labels:
            # Filter to neuron sources only (embeddings use incorrect RelP shortcut)
            neuron_upstream = [(u, lbl) for u, lbl in upstream_with_labels
                              if not u.neuron_id.startswith("EMB/") and not u.neuron_id.startswith("embedding")]
            if neuron_upstream:
                context_parts.append("\n=== UPSTREAM NEURONS (RelP Jacobian-based) ===")
                context_parts.append("These neurons feed into this neuron. Weight sign indicates excitation (+) or inhibition (-).")
                for u, current_label in sorted(neuron_upstream, key=lambda x: -abs(x[0].weight))[:10]:
                    sign = "EXCITES" if u.weight > 0 else "INHIBITS"
                    context_parts.append(f"  {sign}: \"{current_label}\" ({u.neuron_id}, w={u.weight:+.3f}, freq={u.frequency:.1%})")

        if upstream_without_labels and not upstream_with_labels:
            # Filter to neuron sources only (embeddings use incorrect RelP shortcut)
            neuron_unlabeled = [(u, lbl) for u, lbl in upstream_without_labels
                               if not u.neuron_id.startswith("EMB/") and not u.neuron_id.startswith("embedding")]
            if neuron_unlabeled:
                context_parts.append("\n=== UPSTREAM NEURONS (not yet labeled) ===")
                for u, _ in sorted(neuron_unlabeled, key=lambda x: -abs(x[0].weight))[:5]:
                    sign = "+" if u.weight > 0 else "-"
                    context_parts.append(f"  {sign} {u.neuron_id} (w={u.weight:+.3f}, freq={u.frequency:.1%})")

        # === INPUT TOKEN ASSOCIATIONS ===
        input_tokens = profile.get("input_token_associations", [])
        if input_tokens:
            context_parts.append("\n=== INPUT TOKEN PATTERNS (tokens present when this neuron fires) ===")
            for t in input_tokens[:8]:
                context_parts.append(f"  {repr(t['token']):20s} freq={t['frequency']:.0%}")

        # === CO-OCCURRENCE INFO ===
        cooccur = profile.get("co_occurring_neurons", [])
        if cooccur:
            # Look up labels for co-occurring neurons (with Transluce fallback)
            cooccur_labeled = []
            for c in cooccur[:10]:
                cid = c.get("neuron_id", "")
                clabel = self._get_neuron_label_with_fallback(cid)
                if clabel:
                    cooccur_labeled.append((cid, clabel, c.get("frequency", 0)))

            if cooccur_labeled:
                context_parts.append("\n=== CO-OCCURRING NEURONS (fire together with this neuron) ===")
                for cid, clabel, freq in cooccur_labeled[:5]:
                    context_parts.append(f"  {cid}: \"{clabel}\" (co-occur {freq:.0%})")

        context = "\n".join(context_parts)

        # Build the full prompt
        return f"""You are analyzing a neuron in Llama-3.1-8B to understand its INPUT FUNCTION.

Note: Transluce labels come from a general dataset. RelP connections (upstream neurons, token
associations) come from the primary corpus. Base your labels on the actual patterns shown.

{context}

Your task: Determine what TRIGGERS this neuron to fire.

The neuron's OUTPUT function (what it does when it fires) is already known. Now we need to understand:
- What input patterns cause this neuron to activate?
- What upstream computations feed into it?
- What context or tokens must be present?

Analyze the evidence and provide a structured response with these four fields:

1. INTERPRETABILITY: How confident are you that there's a clear triggering pattern?
   - "high" = Clear, obvious trigger (e.g., "fires when specific topic/token appears")
   - "medium" = Likely pattern but incomplete evidence
   - "low" = Cannot determine what triggers this neuron

2. TYPE: What category best describes the INPUT trigger?
   - "token-pattern" = Triggered by specific tokens/words in input
   - "context" = Triggered by semantic context (topic, domain, sentiment)
   - "position" = Triggered by position in sequence (early, late, after punctuation)
   - "upstream-gated" = Primarily controlled by upstream neurons' activity
   - "combination" = Multiple factors combine to trigger activation
   - "unknown" = Cannot determine

3. SHORT_LABEL: A brief 3-8 word label for the input trigger.
   - Example: "question-word-detector"
   - Example: "sentence-start-context"
   - Example: "gated-by-topic-neurons"
   - If uninterpretable: "uninterpretable-trigger"
   - IMPORTANT: Label based on ACTUAL patterns, not assumed domains.

4. INPUT_FUNCTION: A 1-2 sentence description of what triggers this neuron.
   - Focus on the INPUT conditions, not what the neuron outputs
   - Reference specific evidence (Transluce labels, upstream neurons)

First, generate THREE hypotheses at different specificity levels:

HYPOTHESIS_SPECIFIC: [Most specific - e.g., "fires on 'green' color word"]
HYPOTHESIS_MEDIUM: [Medium specificity - e.g., "fires on color-related terms"]
HYPOTHESIS_GENERAL: [Most general - e.g., "fires on adjectives generally"]

For each hypothesis, what evidence would CONTRADICT it?
CONTRADICT_SPECIFIC: [What would disprove the specific hypothesis?]
CONTRADICT_MEDIUM: [What would disprove the medium hypothesis?]
CONTRADICT_GENERAL: [What would disprove the general hypothesis?]

Based on the evidence above, which hypothesis best fits ALL the data? Consider:
- Does the trigger pattern hold across ALL activating contexts, or just some?
- Is the specificity justified by the evidence, or are you over-fitting?
- Would a more general trigger explain the same observations?
- Don't be overly general either - if evidence points to a specific trigger, use that.

Now provide your final answer:
INTERPRETABILITY: [low/medium/high]
TYPE: [token-pattern/context/position/upstream-gated/combination/unknown]
SHORT_LABEL: [3-8 word label - use the chosen hypothesis level]
INPUT_FUNCTION: [detailed description based on chosen hypothesis]"""

    def _decode_embedding_label(self, emb_id: str) -> str:
        """Decode an embedding ID to a human-readable token label.

        Handles special tokens (>128000) and makes whitespace visible.
        """
        # EMB/22559 -> token 22559
        try:
            token_id = int(emb_id.split("/")[1])
            if self.interpreter.tokenizer:
                token = self.interpreter.tokenizer.decode([token_id])
                # Make whitespace-only tokens visible
                if token.strip() == "":
                    # Show repr for whitespace (e.g., '\n\n' instead of blank)
                    return f"embedding {repr(token)}"
                return f"embedding {repr(token)}"
            return f"embedding #{token_id}"
        except (IndexError, ValueError):
            return emb_id

    def format_target_id(self, target_id: str) -> str:
        """Format a target ID for display."""
        parts = target_id.split("_")
        if len(parts) < 2:
            return target_id

        if parts[0] == "L":
            # Logit target - decode token
            token_id = int(parts[1])
            try:
                token = self.interpreter.tokenizer.decode([token_id])
                return f"LOGIT/{token_id} ({repr(token)})"
            except:
                return f"LOGIT/{token_id}"
        else:
            return f"L{parts[0]}/N{parts[1]}"

    def display_progress(self):
        """Display session progress."""
        labeled, skipped, total = self.get_layer_progress(self.session.current_layer)

        table = Table(
            title="Session Progress",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Current Layer", f"L{self.session.current_layer}")
        table.add_row("Layer Progress", f"{labeled + skipped}/{total}")
        table.add_row("Labeled", f"[green]{labeled}[/green]")
        table.add_row("Skipped", f"[yellow]{skipped}[/yellow]")
        table.add_row("Remaining", f"[dim]{total - labeled - skipped}[/dim]")
        table.add_row("", "")
        table.add_row("Total Session Labeled", f"[bold green]{self.session.total_labeled}[/bold green]")
        table.add_row("Total Session Skipped", f"[yellow]{self.session.total_skipped}[/yellow]")

        self.console.print(table)

    def display_neuron_info(self, profile: dict):
        """Display neuron information panel."""
        nid = profile["neuron_id"]
        layer = profile["layer"]
        appearances = profile.get("appearance_count", 0)
        domain_spec = profile.get("domain_specificity", 0)

        # Header
        header_text = Text()
        header_text.append("Neuron: ", style="bold")
        header_text.append(f"{nid}", style="bold cyan")
        header_text.append(f"\nLayer: {layer}/31 | ")
        header_text.append(f"Appearances: {appearances} | ")
        header_text.append(f"Domain Specificity: {domain_spec:.1%}")

        # Add Transluce labels if available (positive = what activates, negative = what suppresses)
        transluce_pos = profile.get("transluce_label_positive", "")
        transluce_neg = profile.get("transluce_label_negative", "")

        if transluce_pos:
            header_text.append("\nActivates on: ", style="bold green")
            header_text.append(transluce_pos, style="italic")
        if transluce_neg:
            header_text.append("\nSuppressed by: ", style="bold red")
            header_text.append(transluce_neg, style="italic")

        # Fallback to old format
        neurondb_label = profile.get("max_act_label", "")
        if neurondb_label and not transluce_pos:
            header_text.append("\nNeuronDB Label: ", style="dim")
            header_text.append(neurondb_label, style="italic")

        self.console.print(Panel(
            header_text,
            title="[bold]Current Neuron[/bold]",
            border_style="blue"
        ))

    def display_existing_label(self, profile: dict):
        """Display existing label from database if present."""
        nid = profile["neuron_id"]
        neuron_func = self.interpreter.db.get(nid)

        if not neuron_func:
            return

        # Show OUTPUT label if exists
        if neuron_func.function_label:
            interp = neuron_func.interpretability
            if interp == "high":
                interp_style = "bold green"
            elif interp == "medium":
                interp_style = "bold yellow"
            else:
                interp_style = "bold red"

            label_text = Text()
            label_text.append("INTERPRETABILITY: ", style="bold")
            label_text.append(f"{interp.upper()}\n", style=interp_style)
            label_text.append("TYPE: ", style="bold")
            label_text.append(f"{neuron_func.function_type}\n", style="magenta")
            label_text.append("LABEL: ", style="bold")
            label_text.append(f"{neuron_func.function_label}\n", style="cyan")
            label_text.append("DESCRIPTION: ", style="bold")
            label_text.append(neuron_func.function_description, style="white")

            # Highlight if this is the current pass
            if self.label_pass == "output":
                title = f"[bold green]OUTPUT Label ({neuron_func.confidence})[/bold green]"
                border = "green"
            else:
                title = "[bold blue]OUTPUT Label (existing)[/bold blue]"
                border = "blue"

            self.console.print(Panel(label_text, title=title, border_style=border))

        # Show INPUT label if exists
        if neuron_func.input_label:
            interp = neuron_func.input_interpretability
            if interp == "high":
                interp_style = "bold green"
            elif interp == "medium":
                interp_style = "bold yellow"
            else:
                interp_style = "bold red"

            label_text = Text()
            label_text.append("INTERPRETABILITY: ", style="bold")
            label_text.append(f"{interp.upper()}\n", style=interp_style)
            label_text.append("TYPE: ", style="bold")
            label_text.append(f"{neuron_func.input_type}\n", style="magenta")
            label_text.append("LABEL: ", style="bold")
            label_text.append(f"{neuron_func.input_label}\n", style="cyan")
            label_text.append("DESCRIPTION: ", style="bold")
            label_text.append(neuron_func.input_description, style="white")

            # Highlight if this is the current pass
            if self.label_pass == "input":
                title = f"[bold green]INPUT Label ({neuron_func.input_confidence})[/bold green]"
                border = "green"
            else:
                title = "[bold blue]INPUT Label (existing)[/bold blue]"
                border = "blue"

            self.console.print(Panel(label_text, title=title, border_style=border))

    def display_upstream_sources(self, profile: dict):
        """Display upstream neuron sources table (for input pass).

        Only shows neuron-to-neuron edges, NOT embedding edges.
        Embedding edges in RelP are computed incorrectly (just use target.relp_score),
        so we rely on static input projection for token activation direction instead.

        For neuron-to-neuron edges, RelP computes proper Jacobians:
        edge_weight = source.activation × ∂target/∂source
        The sign indicates whether the source excites (+) or inhibits (-) this neuron.
        """
        sources = profile.get("top_upstream_sources", [])
        if not sources:
            return

        # Filter to neuron sources only (exclude embeddings - those use incorrect shortcut)
        neuron_sources = [s for s in sources if not s["source"].startswith("E_")]

        if not neuron_sources:
            self.console.print("[dim]No upstream neurons (only embedding sources, see static projection above)[/dim]")
            return

        table = Table(
            title="Upstream Neurons (RelP Jacobian: + = excites, - = inhibits)",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Source", style="cyan")
        table.add_column("Weight", justify="right")
        table.add_column("Freq", justify="right")
        table.add_column("Output Label", style="yellow", max_width=40)

        # Deduplicate by source_id
        seen = {}
        for src in neuron_sources:
            source_id = src["source"]
            if source_id not in seen or abs(src.get("avg_weight", 0)) > abs(seen[source_id].get("avg_weight", 0)):
                seen[source_id] = src

        for src in sorted(seen.values(), key=lambda x: -abs(x.get("avg_weight", 0)))[:10]:
            source_id = src["source"]

            # Parse source format: "0_491" -> "L0/N491"
            parts = source_id.split("_")
            if len(parts) >= 2:
                neuron_id = f"L{parts[0]}/N{parts[1]}"
                existing = self.interpreter.db.get_function_label(neuron_id) or "-"
            else:
                neuron_id = source_id
                existing = "-"

            # Truncate existing label if too long
            if len(existing) > 40:
                existing = existing[:37] + "..."

            weight = src.get("avg_weight", 0)
            # Weight sign = excitation/inhibition direction
            weight_style = "green" if weight > 0 else "red"

            table.add_row(
                neuron_id,
                f"[{weight_style}]{weight:+.4f}[/{weight_style}]",
                f"{src.get('frequency', 0):.1%}",
                existing
            )

        self.console.print(table)

    def display_downstream_targets(self, profile: dict):
        """Display downstream targets table."""
        targets = profile.get("top_downstream_targets", [])
        if not targets:
            return

        # Separate logit targets and neuron targets
        logit_targets = [t for t in targets if t["target"].startswith("L_")]
        neuron_targets = [t for t in targets if not t["target"].startswith("L_")]

        # For logits: deduplicate by token_id, keeping strongest weight
        seen_tokens = {}
        for t in logit_targets:
            token_id = t["target"].split("_")[1]
            if token_id not in seen_tokens or abs(t["avg_weight"]) > abs(seen_tokens[token_id]["avg_weight"]):
                seen_tokens[token_id] = t

        # Sort by weight (strongest first)
        logit_list = sorted(seen_tokens.values(), key=lambda x: -abs(x["avg_weight"]))[:8]
        neuron_list = sorted(neuron_targets, key=lambda x: -abs(x["avg_weight"]))[:6]

        table = Table(
            title="Top Downstream Targets (by weight)",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold"
        )
        table.add_column("Target", style="green")
        table.add_column("Weight", justify="right")
        table.add_column("Freq", justify="right")
        table.add_column("Existing Label", style="yellow", max_width=40)

        for tgt in logit_list + neuron_list:
            target_id = tgt["target"]
            formatted = self.format_target_id(target_id)

            # Check for existing label
            parts = target_id.split("_")
            if len(parts) >= 2 and parts[0] != "L":
                neuron_id = f"L{parts[0]}/N{parts[1]}"
                existing = self.interpreter.db.get_function_label(neuron_id) or "-"
            else:
                existing = "-"

            # Truncate existing label if too long
            if len(existing) > 40:
                existing = existing[:37] + "..."

            weight = tgt.get("avg_weight", 0)
            weight_style = "green" if weight > 0 else "red"

            table.add_row(
                formatted,
                f"[{weight_style}]{weight:+.4f}[/{weight_style}]",
                f"{tgt.get('frequency', 0):.1%}",
                existing
            )

        self.console.print(table)

    def display_output_projection(self, profile: dict):
        """Display output projection (promotes/suppresses)."""
        output_proj = profile.get("output_projection", {})
        # Support both key formats: promotes/suppresses and promoted/suppressed
        promotes = output_proj.get("promotes", output_proj.get("promoted", []))
        suppresses = output_proj.get("suppresses", output_proj.get("suppressed", []))

        if not promotes and not suppresses:
            return

        output_norm = output_proj.get("output_norm", 0)
        title = "Output Projection" + (f" (norm={output_norm:.3f})" if output_norm else "")

        table = Table(
            title=title,
            box=box.SIMPLE,
            show_header=True,
            header_style="bold"
        )
        table.add_column("PROMOTES", style="green", max_width=30)
        table.add_column("Weight", justify="right", style="green")
        table.add_column("SUPPRESSES", style="red", max_width=30)
        table.add_column("Weight", justify="right", style="red")

        max_rows = max(len(promotes), len(suppresses), 1)
        for i in range(min(max_rows, 8)):
            p_token = promotes[i]["token"] if i < len(promotes) else ""
            # Support both key formats: logit_contribution and weight
            p_weight_val = promotes[i].get('logit_contribution', promotes[i].get('weight', 0)) if i < len(promotes) else 0
            p_weight = f"+{p_weight_val:.3f}" if p_token else ""
            s_token = suppresses[i]["token"] if i < len(suppresses) else ""
            s_weight_val = suppresses[i].get('logit_contribution', suppresses[i].get('weight', 0)) if i < len(suppresses) else 0
            s_weight = f"{s_weight_val:.3f}" if s_token else ""

            table.add_row(repr(p_token) if p_token else "", p_weight, repr(s_token) if s_token else "", s_weight)

        self.console.print(table)

    def display_input_projection(self, profile: dict):
        """Display input projection (what tokens activate/suppress this neuron).

        Uses precomputed projections from edge stats if available, otherwise
        computes on-the-fly using SiLU(gate)*up formula.
        Shows which vocabulary tokens would most activate this neuron.
        """
        # Check if we have cached input projection in profile
        input_proj = profile.get("input_projection", {})
        activates = input_proj.get("activates", [])
        suppresses = input_proj.get("suppresses", [])

        # If not cached, try to compute from model
        if not activates and not suppresses:
            try:
                neuron_id = profile["neuron_id"]
                parts = neuron_id.split("/")
                layer = int(parts[0][1:])
                neuron_idx = int(parts[1][1:])

                # This will load the model if not already loaded
                with self.console.status("[bold cyan]Computing input projection...[/bold cyan]"):
                    act_effects, sup_effects = self.interpreter.get_neuron_input_sensitivity(
                        layer, neuron_idx, top_k=10
                    )

                activates = [{"token": t.token, "weight": t.logit_contribution} for t in act_effects]
                suppresses = [{"token": t.token, "weight": t.logit_contribution} for t in sup_effects]

            except Exception as e:
                self.console.print(f"[dim]Could not compute input projection: {e}[/dim]")
                return

        if not activates and not suppresses:
            return

        input_norm = input_proj.get("input_norm", 0)
        title = "Input Projection (SiLU(gate)*up)" + (f" (norm={input_norm:.3f})" if input_norm else "")

        table = Table(
            title=title,
            box=box.SIMPLE,
            show_header=True,
            header_style="bold"
        )
        table.add_column("ACTIVATES", style="green", max_width=30)
        table.add_column("Weight", justify="right", style="green")
        table.add_column("SUPPRESSES", style="red", max_width=30)
        table.add_column("Weight", justify="right", style="red")

        max_rows = max(len(activates), len(suppresses), 1)
        for i in range(min(max_rows, 8)):
            a_token = activates[i]["token"] if i < len(activates) else ""
            a_weight_val = activates[i].get('weight', 0) if i < len(activates) else 0
            a_weight = f"+{a_weight_val:.3f}" if a_token else ""
            s_token = suppresses[i]["token"] if i < len(suppresses) else ""
            s_weight_val = suppresses[i].get('weight', 0) if i < len(suppresses) else 0
            s_weight = f"{s_weight_val:.3f}" if s_token else ""

            table.add_row(repr(a_token) if a_token else "", a_weight, repr(s_token) if s_token else "", s_weight)

        self.console.print(table)

    def display_direct_effect_ratio(self, profile: dict):
        """Display direct effect ratio information."""
        der = profile.get("direct_effect_ratio", {})
        if not der:
            return

        effect_type = der.get("effect_type", "unknown")
        mean = der.get("mean", 0)
        std = der.get("std", 0)

        # Color based on effect type
        if effect_type == "logit":
            style = "bold yellow"
        elif effect_type == "routing":
            style = "bold blue"
        else:
            style = "bold white"

        self.console.print(
            f"[{style}]Direct Effect: {mean:.1%} ({effect_type})[/{style}] "
            f"[dim](std={std:.1%}, range={der.get('min', 0):.1%}-{der.get('max', 0):.1%})[/dim]"
        )

    def display_token_associations(self, profile: dict):
        """Display output token associations from edge stats."""
        tokens = profile.get("output_token_associations", [])
        if not tokens:
            return

        table = Table(
            title="Output Token Associations (from edge stats)",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold"
        )
        table.add_column("Token", style="cyan")
        table.add_column("Frequency", justify="right")
        table.add_column("Count", justify="right")

        for t in tokens[:10]:
            table.add_row(
                repr(t["token"]),
                f"{t['frequency']:.0%}",
                str(t["count"])
            )

        self.console.print(table)

    def display_prompt(self, prompt: str):
        """Display the LLM prompt."""
        self.console.print(Panel(
            prompt,
            title="[bold]LLM Prompt[/bold]",
            border_style="dim",
            padding=(0, 1)
        ))

    def display_response(self, response: str):
        """Display the LLM response with parsed fields highlighted."""
        parsed = self.parse_structured_response(response)

        # Build formatted display
        interp = parsed["interpretability"]
        if interp == "high":
            interp_style = "bold green"
        elif interp == "medium":
            interp_style = "bold yellow"
        else:
            interp_style = "bold red"

        formatted = Text()
        formatted.append("INTERPRETABILITY: ", style="bold")
        formatted.append(f"{interp.upper()}\n", style=interp_style)
        formatted.append("TYPE: ", style="bold")
        formatted.append(f"{parsed['function_type']}\n", style="magenta")
        formatted.append("SHORT_LABEL: ", style="bold")
        formatted.append(f"{parsed['short_label']}\n", style="cyan")

        if self.label_pass == "input":
            formatted.append("INPUT_FUNCTION: ", style="bold")
            formatted.append(parsed['input_function'], style="white")
        else:
            formatted.append("OUTPUT_FUNCTION: ", style="bold")
            formatted.append(parsed['output_function'], style="white")

        self.console.print(Panel(
            formatted,
            title="[bold green]LLM Response (Parsed)[/bold green]",
            border_style="green",
            padding=(0, 1)
        ))

    def display_commands(self):
        """Display available commands."""
        cmd_text = (
            "[cyan][a][/cyan] Accept  "
            "[cyan][r][/cyan] Retry  "
            "[cyan][e][/cyan] Edit  "
            "[cyan][s][/cyan] Skip  "
            "[cyan][f][/cyan] Finish Layer  "
            "[cyan][p][/cyan] Progress  "
            "[cyan][v][/cyan] View Full Prompt  "
            "[cyan][q][/cyan] Quit"
        )
        self.console.print(cmd_text)

    def call_llm(self, prompt: str) -> str:
        """Synchronous LLM call."""
        with self.console.status("[bold cyan]Calling LLM...[/bold cyan]"):
            # Use max_completion_tokens for newer models (gpt-5, etc.)
            # GPT-5 uses internal reasoning, so needs many tokens
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=8000,
            )
        return response.choices[0].message.content.strip()

    async def call_llm_async(self, prompt: str) -> str:
        """Async LLM call."""
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=8000,
        )
        return response.choices[0].message.content.strip()

    def validate_label_against_projection(self, profile: dict, label: dict) -> tuple[bool, str]:
        """
        Validate that a label doesn't contradict clear output projection signals.

        Returns (is_valid, warning_message).
        - If output projection has strong, clear tokens but label describes something
          completely different, returns (False, warning).
        - For routing neurons or weak projections, always returns (True, "").

        This is a SANITY CHECK, not a rejection criterion. Routing neurons may have
        noisy projections that don't match their functional description.
        """
        # Only check for output labels
        if self.label_pass != "output":
            return True, ""

        # Check direct effect ratio - skip sanity check for routing neurons
        der = profile.get("direct_effect_ratio", {})
        effect_type = der.get("effect_type", "unknown")
        if effect_type == "routing":
            return True, ""  # Routing neurons have noisy projections, skip check

        # Get output projection
        output_proj = profile.get("output_projection", {})
        promotes = output_proj.get("promotes", output_proj.get("promoted", []))
        suppresses = output_proj.get("suppresses", output_proj.get("suppressed", []))

        if not promotes and not suppresses:
            return True, ""  # No projection data

        # Get max projection weight
        max_weight = 0.0
        for t in promotes[:10]:
            max_weight = max(max_weight, abs(t.get("logit_contribution", t.get("weight", 0))))
        for t in suppresses[:10]:
            max_weight = max(max_weight, abs(t.get("logit_contribution", t.get("weight", 0))))

        # Only validate if projection is strong enough to be meaningful
        if max_weight < 0.15:
            return True, ""  # Weak projection, skip validation

        # Extract top tokens for comparison
        top_tokens = []
        for t in promotes[:5]:
            top_tokens.append(t.get("token", "").lower().strip())
        for t in suppresses[:5]:
            top_tokens.append(t.get("token", "").lower().strip())

        # Get label text for comparison
        short_label = label.get("short_label", "").lower()
        description = label.get("output_function", "").lower()
        label_text = f"{short_label} {description}"

        # Check for common contradictions:
        # 1. Label says "uninterpretable" but projection has clear pattern
        if "uninterpretable" in short_label or "unknown" in short_label:
            # Check if tokens share obvious patterns
            token_text = " ".join(top_tokens)
            # If tokens have common prefixes/patterns, flag for review
            if len(set(t[0] for t in top_tokens if t)) <= 2:  # Most start with same letter
                return False, f"Label says uninterpretable but tokens may share pattern: {top_tokens[:5]}"

        # 2. Label mentions specific domain but projection tokens don't match
        domain_keywords = {
            "medical": ["disease", "symptom", "treatment", "drug", "patient"],
            "anatomy": ["muscle", "bone", "organ", "tissue", "nerve"],
            "chemistry": ["molecule", "compound", "chemical", "reaction"],
            "financial": ["price", "market", "stock", "trade", "money"],
        }

        for domain, keywords in domain_keywords.items():
            if domain in label_text:
                # Check if any projection tokens match this domain
                token_text = " ".join(top_tokens)
                matches = sum(1 for kw in keywords if kw in token_text)
                if matches == 0:
                    return False, f"Label mentions '{domain}' but projection tokens don't match: {top_tokens[:5]}"

        return True, ""

    def accept_label(self, profile: dict, label: str | dict):
        """Accept and save a label.

        Args:
            profile: Neuron profile dict
            label: Either a string (legacy) or dict with keys:
                   For output pass: interpretability, function_type, short_label, output_function
                   For input pass: interpretability, function_type, short_label, input_function
        """
        nid = profile["neuron_id"]

        # Run sanity check for output labels
        if isinstance(label, dict) and self.label_pass == "output":
            is_valid, warning = self.validate_label_against_projection(profile, label)
            if not is_valid:
                self.console.print(f"[yellow]Warning: {warning}[/yellow]")
                self.console.print("[dim]Label will still be saved, but consider reviewing.[/dim]")

        # Get existing neuron or create new one (preserves existing labels)
        neuron_func = self.interpreter.db.get(nid)
        if neuron_func is None:
            neuron_func = self.interpreter.process_neuron_connections_only(profile)

        if isinstance(label, dict):
            if self.label_pass == "input":
                # Input pass - save to input fields
                neuron_func.input_interpretability = label.get("interpretability", "medium")
                neuron_func.input_type = label.get("function_type", "")
                neuron_func.input_label = label.get("short_label", "")
                neuron_func.input_description = label.get("input_function", "")
                neuron_func.input_confidence = "human-reviewed"
            else:
                # Output pass - save to output fields
                neuron_func.interpretability = label.get("interpretability", "medium")
                neuron_func.function_type = label.get("function_type", "")
                neuron_func.function_label = label.get("short_label", "")
                neuron_func.function_description = label.get("output_function", "")
                neuron_func.confidence = "human-reviewed"
        else:
            # Legacy string label
            if self.label_pass == "input":
                neuron_func.input_label = label
            else:
                neuron_func.function_label = label

        # Save to database
        self.interpreter.db.set(neuron_func)
        self.interpreter.db.save()

        # Update session state
        self.session.neuron_status[nid] = NeuronStatus.LABELED
        self.session.total_labeled += 1
        self.session.save_state()

        self.console.print(f"[green]Saved label for {nid}[/green]")

    def skip_neuron(self, profile: dict):
        """Skip a neuron without labeling."""
        nid = profile["neuron_id"]
        self.session.neuron_status[nid] = NeuronStatus.SKIPPED
        self.session.total_skipped += 1
        self.session.save_state()
        self.console.print(f"[yellow]Skipped {nid}[/yellow]")

    async def finish_layer_async(self, layer: int):
        """Complete remaining neurons in layer asynchronously."""
        import time

        neurons = self.get_layer_neurons(layer)
        remaining = [
            n for n in neurons
            if n["neuron_id"] not in self.session.neuron_status
            or self.session.neuron_status[n["neuron_id"]] == NeuronStatus.PENDING
        ]

        if not remaining:
            self.console.print("[yellow]No remaining neurons in this layer.[/yellow]")
            return

        layer_start_time = time.time()
        self.console.print(f"\n[bold]Finishing layer {layer}: {len(remaining)} neurons remaining (batch_size={self.batch_size})[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Processing L{layer}...", total=len(remaining))

            # Process in batches
            for i in range(0, len(remaining), self.batch_size):
                batch = remaining[i:i + self.batch_size]

                # Build prompts
                prompts = [self.build_prompt_for_neuron(n) for n in batch]

                # Call LLM in parallel
                tasks = [self.call_llm_async(p) for p in prompts]
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                # Save results
                batch_saved = 0
                warnings_count = 0
                for neuron, response in zip(batch, responses):
                    nid = neuron["neuron_id"]
                    if isinstance(response, Exception):
                        self.console.print(f"[red]Error for {nid}: {response}[/red]")
                        continue

                    # Parse structured response and save to database
                    parsed = self.parse_structured_response(response)

                    # Run sanity check for output labels
                    if self.label_pass == "output":
                        is_valid, warning = self.validate_label_against_projection(neuron, parsed)
                        if not is_valid:
                            warnings_count += 1
                            # Store warning in database for later review
                            parsed["sanity_warning"] = warning

                    # Get existing neuron or create new one (preserves existing labels)
                    neuron_func = self.interpreter.db.get(nid)
                    if neuron_func is None:
                        neuron_func = self.interpreter.process_neuron_connections_only(neuron)

                    if self.label_pass == "input":
                        # Input pass - save to input fields
                        neuron_func.input_interpretability = parsed["interpretability"]
                        neuron_func.input_type = parsed["function_type"]
                        neuron_func.input_label = parsed["short_label"]
                        neuron_func.input_description = parsed["input_function"]
                        neuron_func.input_confidence = "llm-auto"
                    else:
                        # Output pass - save to output fields
                        neuron_func.interpretability = parsed["interpretability"]
                        neuron_func.function_type = parsed["function_type"]
                        neuron_func.function_label = parsed["short_label"]
                        neuron_func.function_description = parsed["output_function"]
                        neuron_func.confidence = "llm-auto"
                    self.interpreter.db.set(neuron_func)

                    self.session.neuron_status[nid] = NeuronStatus.LABELED
                    self.session.total_labeled += 1
                    batch_saved += 1

                    progress.advance(task)

                # Save progress after each batch with count
                self.interpreter.db.save(batch_count=batch_saved)
                self.session.save_state()

                # Report sanity warnings
                if warnings_count > 0:
                    self.console.print(f"[yellow]Batch had {warnings_count} sanity warnings (labels may need review)[/yellow]")

                # Calculate and display ETA
                elapsed = time.time() - layer_start_time
                completed = i + len(batch)
                if completed > 0:
                    rate = completed / elapsed  # neurons per second
                    remaining_count = len(remaining) - completed
                    eta_seconds = remaining_count / rate if rate > 0 else 0
                    eta_minutes = eta_seconds / 60
                    self.console.print(
                        f"[dim]Layer progress: {completed}/{len(remaining)} "
                        f"({rate*60:.0f}/min, ETA: {eta_minutes:.1f} min)[/dim]"
                    )

        layer_elapsed = time.time() - layer_start_time
        self.console.print(f"[green]Completed layer {layer} in {layer_elapsed/60:.1f} minutes![/green]")

    def get_next_neuron(self) -> dict | None:
        """Get the next neuron to label."""
        while self.session.current_layer >= 0:
            neurons = self.get_layer_neurons(self.session.current_layer)

            # Find next unlabeled neuron in current layer
            while self.session.current_index < len(neurons):
                neuron = neurons[self.session.current_index]
                nid = neuron["neuron_id"]

                status = self.session.neuron_status.get(nid)
                if status not in (NeuronStatus.LABELED, NeuronStatus.SKIPPED):
                    return neuron

                self.session.current_index += 1

            # Move to next layer
            self.console.print(f"\n[bold cyan]Layer {self.session.current_layer} complete. Moving to layer {self.session.current_layer - 1}[/bold cyan]")

            # Propagate labels when moving layers
            self.interpreter.propagate_labels()

            self.session.current_layer -= 1
            self.session.current_index = 0
            self.session.save_state()

        return None

    def _run_browse_loop(self):
        """Browse mode loop - quick navigation without LLM calls."""
        # Build flat list of all neurons for easy navigation
        all_neurons = []
        for layer in range(31, -1, -1):
            if layer in self._layer_neurons:
                for p in self._layer_neurons[layer]:
                    all_neurons.append(p)

        if not all_neurons:
            self.console.print("[red]No neurons found matching criteria[/red]")
            return

        current_idx = 0

        # Find starting position based on session state
        for i, p in enumerate(all_neurons):
            if p['layer'] == self.session.current_layer:
                current_idx = i
                break

        while True:
            # Display current neuron
            profile = all_neurons[current_idx]

            self.console.print("\n" + "=" * 60)
            self.console.print(f"[dim]Neuron {current_idx + 1} of {len(all_neurons)}[/dim]\n")

            self.display_neuron_info(profile)
            self.display_existing_label(profile)  # Show generated label if exists
            self.display_direct_effect_ratio(profile)

            if self.label_pass == "input":
                self.display_input_projection(profile)
                self.display_upstream_sources(profile)
            else:
                self.display_output_projection(profile)
                self.display_token_associations(profile)
                self.display_downstream_targets(profile)

            # Get action
            action = Prompt.ask(
                "\n[bold]Action[/bold] [n/b/r/g/l/p/q]",
                default="n"
            ).lower().strip()

            if action in ("n", ""):
                # Next neuron
                current_idx = min(current_idx + 1, len(all_neurons) - 1)
                if current_idx == len(all_neurons) - 1:
                    self.console.print("[yellow]At last neuron[/yellow]")

            elif action == "b":
                # Previous neuron
                current_idx = max(current_idx - 1, 0)
                if current_idx == 0:
                    self.console.print("[yellow]At first neuron[/yellow]")

            elif action == "g":
                # Go to specific neuron
                target = Prompt.ask("Enter neuron ID (e.g., L18/N6721)")
                found = False
                for i, p in enumerate(all_neurons):
                    if p['neuron_id'] == target:
                        current_idx = i
                        found = True
                        break
                if not found:
                    self.console.print(f"[red]Neuron {target} not found[/red]")

            elif action == "l":
                # Jump to layer
                try:
                    target_layer = int(Prompt.ask("Enter layer number (0-31)"))
                    found = False
                    for i, p in enumerate(all_neurons):
                        if p['layer'] == target_layer:
                            current_idx = i
                            found = True
                            break
                    if not found:
                        self.console.print(f"[red]No neurons found in layer {target_layer}[/red]")
                except ValueError:
                    self.console.print("[red]Invalid layer number[/red]")

            elif action == "r":
                # Random neuron
                current_idx = random.randint(0, len(all_neurons) - 1)
                self.console.print("[cyan]Jumped to random neuron[/cyan]")

            elif action == "p":
                # Show progress/stats
                self.console.print("\n[bold]Browse Stats:[/bold]")
                self.console.print(f"  Current: {profile['neuron_id']}")
                self.console.print(f"  Position: {current_idx + 1} / {len(all_neurons)}")
                layer_counts = {}
                for p in all_neurons:
                    layer_counts[p['layer']] = layer_counts.get(p['layer'], 0) + 1
                self.console.print(f"  Layers with neurons: {len(layer_counts)}")
                self.console.print(f"  Total neurons: {len(all_neurons)}")

            elif action == "q":
                self.console.print("[green]Goodbye![/green]")
                return

    async def run_auto_all_layers(self, start_layer: int, end_layer: int = None):
        """Process all layers automatically.

        For output pass (late-to-early): start_layer → 0
        For input pass (early-to-late): start_layer → 31
        """
        import time

        if self.label_pass == "input":
            # Input pass: early to late (ascending)
            end = end_layer if end_layer is not None else 31
            layer_range = list(range(start_layer, end + 1))
            direction = f"{start_layer} → {end}"
        else:
            # Output pass: late to early (descending)
            end = end_layer if end_layer is not None else 0
            layer_range = list(range(start_layer, end - 1, -1))
            direction = f"{start_layer} → {end}"

        # Count total neurons across all layers
        total_neurons = 0
        for layer in layer_range:
            neurons = self.get_layer_neurons(layer)
            remaining = [
                n for n in neurons
                if n["neuron_id"] not in self.session.neuron_status
                or self.session.neuron_status[n["neuron_id"]] == NeuronStatus.PENDING
            ]
            total_neurons += len(remaining)

        self.console.print(Panel(
            f"[bold]Auto Mode ({self.label_pass.upper()} pass)[/bold]\n\n"
            f"Processing layers {direction} automatically.\n"
            f"Batch size: {self.batch_size}\n"
            f"Total neurons to process: {total_neurons}\n"
            f"Layers: {len(layer_range)}",
            title="Auto Labeling",
            border_style="green"
        ))

        pass_start_time = time.time()
        layers_completed = 0
        neurons_completed = 0

        for layer in layer_range:
            neurons = self.get_layer_neurons(layer)
            if not neurons:
                self.console.print(f"[dim]Layer {layer}: no neurons[/dim]")
                continue

            # Count remaining
            remaining = [
                n for n in neurons
                if n["neuron_id"] not in self.session.neuron_status
                or self.session.neuron_status[n["neuron_id"]] == NeuronStatus.PENDING
            ]

            if not remaining:
                self.console.print(f"[dim]Layer {layer}: already complete ({len(neurons)} neurons)[/dim]")
                layers_completed += 1
                continue

            self.console.print(f"\n[bold cyan]Layer {layer}: {len(remaining)} neurons to process[/bold cyan]")
            self.session.current_layer = layer
            await self.finish_layer_async(layer)

            layers_completed += 1
            neurons_completed += len(remaining)

            # Overall progress and ETA
            elapsed = time.time() - pass_start_time
            if neurons_completed > 0:
                rate = neurons_completed / elapsed
                remaining_neurons = total_neurons - neurons_completed
                eta_hours = (remaining_neurons / rate) / 3600 if rate > 0 else 0
                self.console.print(
                    f"[bold]Overall: {layers_completed}/{len(layer_range)} layers, "
                    f"{neurons_completed}/{total_neurons} neurons "
                    f"({rate*60:.0f}/min, ETA: {eta_hours:.1f} hours)[/bold]"
                )

        pass_elapsed = time.time() - pass_start_time
        self.console.print(f"\n[bold green]All layers complete in {pass_elapsed/3600:.1f} hours![/bold green]")
        self.display_progress()

    def run_interactive_loop(self):
        """Main interactive loop."""
        # Welcome message
        if self.browse_mode:
            self.console.print(Panel(
                "[bold]Neuron Browser[/bold]\n\n"
                "Browse neurons without LLM calls.\n\n"
                "Commands:\n"
                "  [cyan][n][/cyan] or [Enter] - Next neuron\n"
                "  [cyan][b][/cyan] - Previous neuron (back)\n"
                "  [cyan][r][/cyan] - Random neuron\n"
                "  [cyan][g][/cyan] - Go to specific neuron (e.g., L18/N6721)\n"
                "  [cyan][l][/cyan] - Jump to layer\n"
                "  [cyan][p][/cyan] - Show progress\n"
                "  [cyan][q][/cyan] - Quit",
                title="Browse Mode",
                border_style="cyan"
            ))
            self._run_browse_loop()
            return

        if self.label_pass == "input":
            pass_desc = (
                "[bold]Pass 2: INPUT Function Labeling[/bold]\n\n"
                "Determine what TRIGGERS each neuron to fire.\n"
                "Uses Transluce labels, upstream neurons, and input patterns.\n"
            )
            title = "Input Pass"
        else:
            pass_desc = (
                "[bold]Pass 1: OUTPUT Function Labeling[/bold]\n\n"
                "Determine what each neuron DOES when it fires.\n"
                "Uses output projections and downstream connections.\n"
            )
            title = "Output Pass"

        self.console.print(Panel(
            pass_desc +
            "\nCommands:\n"
            "  [cyan][a][/cyan] Accept - Save label and continue\n"
            "  [cyan][r][/cyan] Retry - Re-generate label with same prompt\n"
            "  [cyan][e][/cyan] Edit - Edit the label before saving\n"
            "  [cyan][s][/cyan] Skip - Skip this neuron\n"
            "  [cyan][f][/cyan] Finish Layer - Auto-complete remaining neurons\n"
            "  [cyan][p][/cyan] Progress - Show session progress\n"
            "  [cyan][v][/cyan] View - Show full prompt\n"
            "  [cyan][q][/cyan] Quit - Save and exit",
            title=title,
            border_style="blue" if self.label_pass == "output" else "green"
        ))

        self.display_progress()

        current_prompt = None
        current_response = None
        current_profile = None

        while True:
            # Get next neuron if needed
            if current_profile is None:
                current_profile = self.get_next_neuron()
                if current_profile is None:
                    self.console.print("[bold green]All neurons labeled![/bold green]")
                    break

                # Build prompt and get response
                self.console.print("\n" + "=" * 60 + "\n")
                self.display_neuron_info(current_profile)
                self.display_existing_label(current_profile)
                self.display_direct_effect_ratio(current_profile)

                if self.label_pass == "input":
                    # Input pass: show input projection and upstream sources
                    self.display_input_projection(current_profile)
                    self.display_upstream_sources(current_profile)
                else:
                    # Output pass: show output projection and downstream targets
                    self.display_output_projection(current_profile)
                    self.display_token_associations(current_profile)
                    self.display_downstream_targets(current_profile)

                current_prompt = self.build_prompt_for_neuron(current_profile)
                self.display_prompt(current_prompt)

                current_response = self.call_llm(current_prompt)
                self.display_response(current_response)

            # Get user action
            self.console.print()
            self.display_commands()
            action = Prompt.ask(
                "\n[bold]Action[/bold]",
                choices=["a", "r", "e", "s", "f", "p", "v", "q"],
                default="a"
            )

            if action == "a":
                parsed = self.parse_structured_response(current_response)
                self.accept_label(current_profile, parsed)
                self.session.current_index += 1
                current_profile = None

            elif action == "r":
                self.console.print("\n[dim]Retrying...[/dim]")
                current_response = self.call_llm(current_prompt)
                self.display_response(current_response)

            elif action == "e":
                self.console.print("\n[dim]Edit the structured response:[/dim]")
                parsed = self.parse_structured_response(current_response)
                interp = Prompt.ask("Interpretability (low/medium/high)", default=parsed["interpretability"])

                if self.label_pass == "input":
                    func_type = Prompt.ask("Type (token-pattern/context/position/upstream-gated/combination/unknown)", default=parsed["function_type"])
                    short_label = Prompt.ask("Short label", default=parsed["short_label"])
                    input_func = Prompt.ask("Input function", default=parsed["input_function"])
                    edited = {
                        "interpretability": interp.lower() if interp.lower() in ("low", "medium", "high") else "medium",
                        "function_type": func_type.lower(),
                        "short_label": short_label,
                        "input_function": input_func,
                    }
                else:
                    func_type = Prompt.ask("Type (semantic/formatting/structural/lexical/routing/unknown)", default=parsed["function_type"])
                    short_label = Prompt.ask("Short label", default=parsed["short_label"])
                    output_func = Prompt.ask("Output function", default=parsed["output_function"])
                    edited = {
                        "interpretability": interp.lower() if interp.lower() in ("low", "medium", "high") else "medium",
                        "function_type": func_type.lower(),
                        "short_label": short_label,
                        "output_function": output_func,
                    }
                self.accept_label(current_profile, edited)
                self.session.current_index += 1
                current_profile = None

            elif action == "s":
                self.skip_neuron(current_profile)
                self.session.current_index += 1
                current_profile = None

            elif action == "f":
                if Confirm.ask(f"Finish all remaining neurons in L{self.session.current_layer} automatically?"):
                    asyncio.run(self.finish_layer_async(self.session.current_layer))
                    current_profile = None

            elif action == "p":
                self.display_progress()

            elif action == "v":
                # Show full prompt
                self.console.print(Panel(
                    current_prompt,
                    title="[bold]Full LLM Prompt[/bold]",
                    border_style="cyan"
                ))

            elif action == "q":
                self.session.save_state()
                self.console.print("[green]Session saved. Goodbye![/green]")
                return

        self.console.print("\n[bold green]Labeling complete![/bold green]")
        self.display_progress()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI for progressive neuron labeling"
    )
    parser.add_argument(
        "--edge-stats",
        type=Path,
        default=Path("data/medical_edge_stats_v6_enriched.json"),
        help="Path to edge statistics JSON"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/interactive_labels.json"),
        help="Path to output label database"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("data/fineweb_edge_stats_500.json"),
        help="Path to baseline edge stats for domain specificity comparison"
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,  # Will be set based on pass type
        help="Path to session state file for resume (default: auto based on pass)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="OpenAI model to use for labeling"
    )
    parser.add_argument(
        "--min-appearances",
        type=int,
        default=10,
        help="Minimum appearances to include a neuron"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=800,
        help="Number of parallel API calls for async mode"
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        default=None,
        help="Layer to start from (default: 31 for output pass, 0 for input pass)"
    )
    parser.add_argument(
        "--end-layer",
        type=int,
        default=None,
        help="Layer to end at (default: 0 for output pass, 31 for input pass)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous session"
    )
    parser.add_argument(
        "--browse",
        action="store_true",
        help="Browse mode: view neurons without LLM calls"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto mode: process all layers automatically (direction based on --pass)"
    )
    parser.add_argument(
        "--pass",
        dest="label_pass",
        choices=["output", "input"],
        default="output",
        help="Labeling pass: 'output' (what neuron does) or 'input' (what triggers neuron)"
    )

    args = parser.parse_args()

    # Set default state file based on pass type
    if args.state is None:
        if args.label_pass == "input":
            args.state = Path("data/.labeling_session_state_input.json")
        else:
            args.state = Path("data/.labeling_session_state.json")

    # Set default start layer based on pass type
    if args.start_layer is None:
        args.start_layer = 0 if args.label_pass == "input" else 31

    # Validate edge stats file exists
    if not args.edge_stats.exists():
        print(f"Error: Edge stats file not found: {args.edge_stats}")
        sys.exit(1)

    # Initialize labeler
    labeler = InteractiveLabeler(
        edge_stats_path=args.edge_stats,
        db_path=args.db,
        state_path=args.state,
        model=args.model,
        min_appearances=args.min_appearances,
        batch_size=args.batch_size,
        browse_mode=args.browse,
        label_pass=args.label_pass,
        baseline_path=args.baseline if args.baseline.exists() else None,
    )

    # Handle resume
    if args.resume:
        if labeler.session.load_state():
            labeler.console.print(f"[green]Resumed session at L{labeler.session.current_layer}, index {labeler.session.current_index}[/green]")
            labeler.console.print(f"[dim]Previously labeled: {labeler.session.total_labeled}, skipped: {labeler.session.total_skipped}[/dim]")
        else:
            labeler.console.print("[yellow]No previous session found. Starting fresh.[/yellow]")
            labeler.session.current_layer = args.start_layer
    else:
        labeler.session.current_layer = args.start_layer

    # Run in appropriate mode
    if args.auto:
        asyncio.run(labeler.run_auto_all_layers(args.start_layer, args.end_layer))
    else:
        labeler.run_interactive_loop()


if __name__ == "__main__":
    main()
