#!/usr/bin/env python3
"""Select interesting neurons for deep investigation.

Filters and scores neurons based on multiple criteria to identify
the most interpretable and interesting candidates for the neuron
scientist agent.

Usage:
    # Select top 500 neurons from labels
    python scripts/select_interesting_neurons.py \
        --labels data/fabric_labels/labels.json \
        --edge-stats data/fabric_edge_stats.json \
        --output data/interesting_neurons.json \
        --top-k 500

    # With baseline for domain specificity
    python scripts/select_interesting_neurons.py \
        --labels data/fabric_labels/labels.json \
        --edge-stats data/medical_edge_stats.json \
        --baseline data/fineweb_edge_stats.json \
        --output data/interesting_neurons.json

    # Filter by minimum interestingness score
    python scripts/select_interesting_neurons.py \
        --labels data/fabric_labels/labels.json \
        --min-interestingness 7
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NeuronCandidate:
    """A neuron candidate for deep investigation."""
    neuron_id: str
    layer: int
    neuron_idx: int

    # Labels
    output_label: str = ""
    output_description: str = ""
    output_type: str = ""
    output_interpretability: str = "medium"
    input_label: str = ""
    input_description: str = ""
    input_type: str = ""
    input_interpretability: str = "medium"

    # Interestingness from labeling
    interestingness_score: int = 5
    interestingness_reason: str = ""

    # Computed scores
    composite_score: float = 0.0
    selection_reasons: list[str] = field(default_factory=list)

    # Profile data
    appearance_count: int = 0
    domain_specificity: float = 0.0
    direct_effect_ratio: float = 0.0
    downstream_semantic_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "neuron_id": self.neuron_id,
            "layer": self.layer,
            "neuron_idx": self.neuron_idx,
            "output_label": self.output_label,
            "output_description": self.output_description,
            "output_type": self.output_type,
            "output_interpretability": self.output_interpretability,
            "input_label": self.input_label,
            "input_description": self.input_description,
            "input_type": self.input_type,
            "input_interpretability": self.input_interpretability,
            "interestingness_score": self.interestingness_score,
            "interestingness_reason": self.interestingness_reason,
            "composite_score": self.composite_score,
            "selection_reasons": self.selection_reasons,
            "appearance_count": self.appearance_count,
            "domain_specificity": self.domain_specificity,
            "direct_effect_ratio": self.direct_effect_ratio,
            "downstream_semantic_count": self.downstream_semantic_count,
        }


# Types to EXCLUDE (not interesting for semantic interpretability)
EXCLUDE_TYPES = {
    "formatting",
    "structural",
    "lexical",
    "unknown",
}


def load_labels(labels_path: Path) -> dict[str, dict[str, Any]]:
    """Load neuron labels from JSON file."""
    with open(labels_path) as f:
        data = json.load(f)

    labels_by_id = {}
    for lbl in data.get("labels", []):
        nid = lbl.get("neuron_id", "")
        if nid:
            labels_by_id[nid] = lbl

    return labels_by_id


def load_edge_stats(edge_stats_path: Path) -> dict[str, dict[str, Any]]:
    """Load edge statistics and return profiles by neuron ID."""
    with open(edge_stats_path) as f:
        data = json.load(f)

    profiles_by_id = {}
    for p in data.get("profiles", []):
        nid = p.get("neuron_id", "")
        if nid:
            profiles_by_id[nid] = p

    return profiles_by_id


def compute_domain_specificity(
    profiles: dict[str, dict],
    baseline_profiles: dict[str, dict]
) -> dict[str, float]:
    """Compute domain specificity ratio for each neuron.

    Returns ratio of (domain_count / baseline_count).
    Higher ratio = more domain-specific.
    """
    specificity = {}

    for nid, profile in profiles.items():
        domain_count = profile.get("appearance_count", 0)
        baseline_profile = baseline_profiles.get(nid, {})
        baseline_count = baseline_profile.get("appearance_count", 0)

        if baseline_count == 0:
            # Not in baseline = highly domain-specific
            if domain_count >= 5:
                specificity[nid] = 10.0  # Cap at 10x
            else:
                specificity[nid] = 1.0
        else:
            ratio = domain_count / baseline_count
            specificity[nid] = min(10.0, ratio)  # Cap at 10x

    return specificity


def count_downstream_semantic(
    profile: dict[str, Any],
    labels: dict[str, dict]
) -> int:
    """Count downstream neurons that are semantic type."""
    count = 0
    downstream = profile.get("top_downstream_targets", [])

    for d in downstream:
        target = d.get("target", "")
        if target.startswith("L_"):
            continue  # Skip logit targets

        parts = target.split("_")
        if len(parts) >= 2:
            target_id = f"L{parts[0]}/N{parts[1]}"
            target_label = labels.get(target_id, {})
            target_type = target_label.get("output_type", "")
            if target_type == "semantic":
                count += 1

    return count


def score_neuron(
    candidate: NeuronCandidate,
    labels: dict[str, dict],
    profile: dict[str, Any],
    domain_specificity: dict[str, float],
) -> float:
    """Compute composite interestingness score for a neuron.

    Scoring criteria:
    - Semantic type: +3 points
    - High interpretability: +2 points
    - LLM interestingness >= 7: +bonus points
    - Domain specificity > 2x: +2 points
    - Strong causal effects (DER > 0.5): +1 point
    - Semantic routing (routing type with semantic downstream): +2 points

    Returns -1 if neuron should be excluded.
    """
    reasons = []

    # Check exclusion criteria
    output_type = candidate.output_type.lower() if candidate.output_type else ""
    if output_type in EXCLUDE_TYPES:
        return -1  # Excluded

    # Also check for "uninterpretable" in label
    if "uninterpretable" in candidate.output_label.lower():
        return -1

    score = 0.0

    # Semantic type bonus
    if output_type == "semantic":
        score += 3.0
        reasons.append("semantic_type")
    elif output_type == "associative":
        score += 2.5
        reasons.append("associative_type")
    elif output_type == "routing" and candidate.layer < 31:
        # Routing can be interesting if it has semantic downstream
        if candidate.downstream_semantic_count > 2:
            score += 2.0
            reasons.append(f"semantic_router({candidate.downstream_semantic_count}_downstream)")

    # High interpretability bonus
    if candidate.output_interpretability == "high":
        score += 2.0
        reasons.append("high_interpretability")
    elif candidate.output_interpretability == "medium":
        score += 0.5

    # LLM interestingness score
    if candidate.interestingness_score >= 8:
        score += (candidate.interestingness_score - 6)
        reasons.append(f"llm_interesting({candidate.interestingness_score})")
    elif candidate.interestingness_score >= 7:
        score += 1.0
        reasons.append(f"llm_interesting({candidate.interestingness_score})")

    # Domain specificity
    spec = domain_specificity.get(candidate.neuron_id, 1.0)
    if spec >= 5.0:
        score += 2.0
        reasons.append(f"domain_specific({spec:.1f}x)")
    elif spec >= 2.0:
        score += 1.0
        reasons.append(f"domain_enriched({spec:.1f}x)")

    # Causal impact (direct effect ratio)
    der = profile.get("direct_effect_ratio", {})
    der_mean = der.get("mean", 0)
    if der_mean > 0.5:
        score += 1.0
        reasons.append(f"high_causal_effect({der_mean:.0%})")

    # Input interpretability bonus (if we have input labels)
    if candidate.input_interpretability == "high":
        score += 1.0
        reasons.append("interpretable_trigger")

    # Layer-based adjustments
    # Early layers (L0-L5): Token detectors and aggregators
    # Middle layers (L6-L25): Routing and composition
    # Late layers (L26-L31): Output formation
    if candidate.layer <= 5 and output_type == "semantic":
        score += 0.5
        reasons.append("early_semantic")
    elif 15 <= candidate.layer <= 25 and output_type in ("semantic", "routing"):
        score += 0.5
        reasons.append("mid_layer_reasoning")

    candidate.composite_score = score
    candidate.selection_reasons = reasons

    return score


def select_interesting_neurons(
    labels_path: Path,
    edge_stats_path: Path,
    baseline_path: Path | None = None,
    top_k: int = 500,
    min_interestingness: int = 0,
    min_appearances: int = 10,
) -> list[NeuronCandidate]:
    """Select the most interesting neurons for investigation.

    Args:
        labels_path: Path to neuron labels JSON
        edge_stats_path: Path to edge statistics JSON
        baseline_path: Optional path to baseline edge stats for domain specificity
        top_k: Number of top neurons to select
        min_interestingness: Minimum LLM interestingness score (0-10)
        min_appearances: Minimum appearance count

    Returns:
        List of NeuronCandidate objects, sorted by composite score
    """
    print(f"Loading labels from {labels_path}...", file=sys.stderr)
    labels = load_labels(labels_path)
    print(f"  Loaded {len(labels)} labels", file=sys.stderr)

    print(f"Loading edge stats from {edge_stats_path}...", file=sys.stderr)
    profiles = load_edge_stats(edge_stats_path)
    print(f"  Loaded {len(profiles)} profiles", file=sys.stderr)

    # Compute domain specificity if baseline provided
    domain_specificity = {}
    if baseline_path and baseline_path.exists():
        print(f"Loading baseline from {baseline_path}...", file=sys.stderr)
        baseline_profiles = load_edge_stats(baseline_path)
        domain_specificity = compute_domain_specificity(profiles, baseline_profiles)
        print(f"  Computed specificity for {len(domain_specificity)} neurons", file=sys.stderr)

    # Build candidates
    candidates = []
    excluded_count = 0
    missing_profile = 0

    for nid, label in labels.items():
        # Get profile
        profile = profiles.get(nid, {})
        if not profile:
            missing_profile += 1
            continue

        # Check minimum appearances
        appearances = profile.get("appearance_count", 0)
        if appearances < min_appearances:
            continue

        # Check minimum interestingness
        interestingness = label.get("interestingness_score", 5)
        if interestingness < min_interestingness:
            continue

        # Parse neuron ID
        parts = nid.split("/")
        if len(parts) != 2:
            continue
        layer = int(parts[0][1:])
        neuron_idx = int(parts[1][1:])

        # Count downstream semantic neurons
        downstream_semantic = count_downstream_semantic(profile, labels)

        # Create candidate
        candidate = NeuronCandidate(
            neuron_id=nid,
            layer=layer,
            neuron_idx=neuron_idx,
            output_label=label.get("output_label", ""),
            output_description=label.get("output_description", ""),
            output_type=label.get("output_type", ""),
            output_interpretability=label.get("output_interpretability", "medium"),
            input_label=label.get("input_label", ""),
            input_description=label.get("input_description", ""),
            input_type=label.get("input_type", ""),
            input_interpretability=label.get("input_interpretability", "medium"),
            interestingness_score=interestingness,
            interestingness_reason=label.get("interestingness_reason", ""),
            appearance_count=appearances,
            domain_specificity=domain_specificity.get(nid, 1.0),
            direct_effect_ratio=profile.get("direct_effect_ratio", {}).get("mean", 0),
            downstream_semantic_count=downstream_semantic,
        )

        # Score the candidate
        score = score_neuron(candidate, labels, profile, domain_specificity)

        if score < 0:
            excluded_count += 1
            continue

        candidates.append(candidate)

    print("\nScoring results:", file=sys.stderr)
    print(f"  Total candidates: {len(candidates)}", file=sys.stderr)
    print(f"  Excluded (type): {excluded_count}", file=sys.stderr)
    print(f"  Missing profile: {missing_profile}", file=sys.stderr)

    # Sort by composite score
    candidates.sort(key=lambda x: -x.composite_score)

    # Take top k
    selected = candidates[:top_k]

    # Print score distribution
    if selected:
        scores = [c.composite_score for c in selected]
        print(f"\nSelected {len(selected)} neurons:", file=sys.stderr)
        print(f"  Score range: {min(scores):.1f} - {max(scores):.1f}", file=sys.stderr)
        print(f"  Mean score: {sum(scores)/len(scores):.1f}", file=sys.stderr)

        # Type distribution
        type_dist = defaultdict(int)
        for c in selected:
            type_dist[c.output_type or "unknown"] += 1
        print(f"  Type distribution: {dict(type_dist)}", file=sys.stderr)

        # Layer distribution
        layer_dist = defaultdict(int)
        for c in selected:
            bucket = c.layer // 8  # 0-7, 8-15, 16-23, 24-31
            layer_dist[f"L{bucket*8}-{(bucket+1)*8-1}"] += 1
        print(f"  Layer distribution: {dict(layer_dist)}", file=sys.stderr)

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Select interesting neurons for deep investigation"
    )
    parser.add_argument(
        "--labels", type=Path, required=True,
        help="Path to neuron labels JSON"
    )
    parser.add_argument(
        "--edge-stats", type=Path, required=True,
        help="Path to edge statistics JSON"
    )
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="Path to baseline edge stats for domain specificity"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/interesting_neurons.json"),
        help="Output path for selected neurons"
    )
    parser.add_argument(
        "--top-k", type=int, default=500,
        help="Number of top neurons to select (default: 500)"
    )
    parser.add_argument(
        "--min-interestingness", type=int, default=0,
        help="Minimum LLM interestingness score (default: 0)"
    )
    parser.add_argument(
        "--min-appearances", type=int, default=10,
        help="Minimum appearance count (default: 10)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.labels.exists():
        print(f"Error: Labels file not found: {args.labels}", file=sys.stderr)
        sys.exit(1)
    if not args.edge_stats.exists():
        print(f"Error: Edge stats file not found: {args.edge_stats}", file=sys.stderr)
        sys.exit(1)

    # Select neurons
    selected = select_interesting_neurons(
        labels_path=args.labels,
        edge_stats_path=args.edge_stats,
        baseline_path=args.baseline,
        top_k=args.top_k,
        min_interestingness=args.min_interestingness,
        min_appearances=args.min_appearances,
    )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "labels_source": str(args.labels),
            "edge_stats_source": str(args.edge_stats),
            "baseline_source": str(args.baseline) if args.baseline else None,
            "top_k": args.top_k,
            "min_interestingness": args.min_interestingness,
            "min_appearances": args.min_appearances,
            "total_selected": len(selected),
        },
        "neurons": [c.to_dict() for c in selected],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(selected)} neurons to {args.output}")

    # Print top 10 for preview
    print("\nTop 10 neurons:")
    for i, c in enumerate(selected[:10]):
        print(f"  {i+1}. {c.neuron_id} (score={c.composite_score:.1f})")
        print(f"      Label: {c.output_label}")
        print(f"      Reasons: {', '.join(c.selection_reasons)}")


if __name__ == "__main__":
    main()
