#!/usr/bin/env python3
"""
Find the cleanest (most monosemantic) facts from the SPD analysis.

A fact is "clean" if the components that fire on it are monosemantic.

For down_proj: A component is monosemantic if it responds to a single label.
For up_proj: A component is monosemantic if it responds to:
  - A single label, OR
  - A single input element at position 0, 1, or 2

We score each fact based on how monosemantic its firing components are.
"""

import re
from collections import Counter, defaultdict


def parse_analysis_file(filepath: str):
    """Parse the analysis.txt file to extract component and fact information."""

    with open(filepath) as f:
        lines = f.readlines()

    # Parse component-to-facts mapping (from the COMPONENT ACTIVATION ANALYSIS section)
    up_proj_components = defaultdict(list)  # component_id -> list of (fact_idx, input, label)
    down_proj_components = defaultdict(list)

    # Parse the per-fact analysis (from PER-FACT COMPONENT ANALYSIS section)
    up_proj_per_fact = {}  # fact_idx -> {inputs, label, components}
    down_proj_per_fact = {}

    current_module = None
    current_section = None  # 'component_analysis' or 'per_fact'
    current_component = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect section changes
        if "COMPONENT ACTIVATION ANALYSIS" in line:
            current_section = "component_analysis"
        elif "PER-FACT COMPONENT ANALYSIS" in line:
            current_section = "per_fact"
        elif "SUMMARY STATISTICS" in line:
            current_section = "summary"

        # Detect module changes
        if "MODULE: block.mlp.up_proj" in line:
            current_module = "up_proj"
        elif "MODULE: block.mlp.down_proj" in line:
            current_module = "down_proj"

        # Parse component activation analysis section
        if current_section == "component_analysis" and current_module:
            # Parse component header: [Rank X] Component Y (mean CI=Z): N facts above threshold
            comp_match = re.match(r"\[Rank \d+\] Component (\d+)", line)
            if comp_match:
                current_component = int(comp_match.group(1))

            # Parse fact line: Fact X: input=[a, b, c] → label=Y (CI=Z)
            fact_match = re.match(
                r"Fact\s+(\d+): input=\[(\d+), (\d+), (\d+)\] → label=(\d+)", line
            )
            if fact_match and current_component is not None:
                fact_idx = int(fact_match.group(1))
                inputs = [
                    int(fact_match.group(2)),
                    int(fact_match.group(3)),
                    int(fact_match.group(4)),
                ]
                label = int(fact_match.group(5))

                if current_module == "up_proj":
                    up_proj_components[current_component].append((fact_idx, inputs, label))
                else:
                    down_proj_components[current_component].append((fact_idx, inputs, label))

        # Parse per-fact analysis section
        if current_section == "per_fact" and current_module:
            # Parse fact line
            fact_match = re.match(
                r"Fact\s+(\d+): input=\[(\d+), (\d+), (\d+)\] → label=(\d+)", line
            )
            if fact_match:
                fact_idx = int(fact_match.group(1))
                inputs = [
                    int(fact_match.group(2)),
                    int(fact_match.group(3)),
                    int(fact_match.group(4)),
                ]
                label = int(fact_match.group(5))

                # Look for components in the next lines
                components = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()

                    # Check if we've hit the next fact or section
                    if (
                        next_line.startswith("Fact ")
                        or next_line.startswith("===")
                        or next_line.startswith("MODULE:")
                    ):
                        break

                    # Parse component activations like C206(1.000)
                    comp_matches = re.findall(r"C(\d+)\(([\d.]+)\)", next_line)
                    for comp_id, ci_score in comp_matches:
                        components.append((int(comp_id), float(ci_score)))

                    j += 1

                if current_module == "up_proj":
                    up_proj_per_fact[fact_idx] = {
                        "inputs": inputs,
                        "label": label,
                        "components": components,
                    }
                else:
                    down_proj_per_fact[fact_idx] = {
                        "inputs": inputs,
                        "label": label,
                        "components": components,
                    }

        i += 1

    return up_proj_components, down_proj_components, up_proj_per_fact, down_proj_per_fact


def compute_component_monosemanticity(component_facts: list) -> dict:
    """
    Compute monosemanticity scores for a component.
    """
    if not component_facts:
        return None

    labels = [f[2] for f in component_facts]
    pos0_vals = [f[1][0] for f in component_facts]
    pos1_vals = [f[1][1] for f in component_facts]
    pos2_vals = [f[1][2] for f in component_facts]

    label_counts = Counter(labels)
    pos0_counts = Counter(pos0_vals)
    pos1_counts = Counter(pos1_vals)
    pos2_counts = Counter(pos2_vals)

    n = len(component_facts)

    dominant_label, dominant_label_count = label_counts.most_common(1)[0]
    dominant_pos0, dominant_pos0_count = pos0_counts.most_common(1)[0]
    dominant_pos1, dominant_pos1_count = pos1_counts.most_common(1)[0]
    dominant_pos2, dominant_pos2_count = pos2_counts.most_common(1)[0]

    return {
        "n_facts": n,
        "n_unique_labels": len(label_counts),
        "dominant_label": dominant_label,
        "dominant_label_ratio": dominant_label_count / n,
        "n_unique_pos0": len(pos0_counts),
        "dominant_pos0": dominant_pos0,
        "dominant_pos0_ratio": dominant_pos0_count / n,
        "n_unique_pos1": len(pos1_counts),
        "dominant_pos1": dominant_pos1,
        "dominant_pos1_ratio": dominant_pos1_count / n,
        "n_unique_pos2": len(pos2_counts),
        "dominant_pos2": dominant_pos2,
        "dominant_pos2_ratio": dominant_pos2_count / n,
    }


def is_component_monosemantic(stats: dict, threshold: float = 0.9) -> tuple[bool, str]:
    """
    Determine if a component is monosemantic based on its statistics.
    Returns (is_monosemantic, reason)
    """
    if stats is None:
        return False, "no_data"

    # Check if it responds to a single label
    if stats["dominant_label_ratio"] >= threshold:
        return True, f"label_{stats['dominant_label']}"

    # Check if it responds to a single input element
    if stats["dominant_pos0_ratio"] >= threshold:
        return True, f"pos0_{stats['dominant_pos0']}"
    if stats["dominant_pos1_ratio"] >= threshold:
        return True, f"pos1_{stats['dominant_pos1']}"
    if stats["dominant_pos2_ratio"] >= threshold:
        return True, f"pos2_{stats['dominant_pos2']}"

    return False, "polysemantic"


def compute_monosemanticity_score(stats: dict) -> float:
    """
    Compute a monosemanticity score from 0 to 1.
    Higher score = more monosemantic.
    """
    if stats is None:
        return 0.0

    # The score is the maximum of all the dominant ratios
    return max(
        stats["dominant_label_ratio"],
        stats["dominant_pos0_ratio"],
        stats["dominant_pos1_ratio"],
        stats["dominant_pos2_ratio"],
    )


def score_fact(
    fact_info: dict,
    up_proj_mono_scores: dict,
    down_proj_mono_scores: dict,
    up_proj_stats: dict,
    down_proj_stats: dict,
) -> tuple[float, dict]:
    """
    Score a fact based on how monosemantic its firing components are.
    Returns (score, details)
    """
    up_components = fact_info.get("up_proj_components", [])
    down_components = fact_info.get("down_proj_components", [])

    if not up_components and not down_components:
        return 0.0, {
            "reason": "no_components",
            "up_proj_components": [],
            "down_proj_components": [],
            "n_components": 0,
        }

    # For each component, get its monosemanticity score
    up_scores = []
    for comp_id, ci_score in up_components:
        mono_score = up_proj_mono_scores.get(comp_id, 0.0)
        stats = up_proj_stats.get(comp_id)
        is_mono, reason = (
            is_component_monosemantic(stats, threshold=0.9) if stats else (False, "unknown")
        )
        up_scores.append((comp_id, mono_score, ci_score, is_mono, reason))

    down_scores = []
    for comp_id, ci_score in down_components:
        mono_score = down_proj_mono_scores.get(comp_id, 0.0)
        stats = down_proj_stats.get(comp_id)
        is_mono, reason = (
            is_component_monosemantic(stats, threshold=0.9) if stats else (False, "unknown")
        )
        down_scores.append((comp_id, mono_score, ci_score, is_mono, reason))

    # Compute fact score as minimum monosemanticity of all components
    all_mono_scores = [s[1] for s in up_scores] + [s[1] for s in down_scores]

    if not all_mono_scores:
        return 0.0, {
            "reason": "no_scores",
            "up_proj_components": up_scores,
            "down_proj_components": down_scores,
            "n_components": 0,
        }

    min_score = min(all_mono_scores)
    mean_score = sum(all_mono_scores) / len(all_mono_scores)

    # Count how many components are monosemantic
    n_mono = sum(1 for s in up_scores + down_scores if s[3])
    total = len(up_scores) + len(down_scores)

    return min_score, {
        "up_proj_components": up_scores,
        "down_proj_components": down_scores,
        "min_mono_score": min_score,
        "mean_mono_score": mean_score,
        "n_components": total,
        "n_mono_components": n_mono,
        "mono_ratio": n_mono / total if total > 0 else 0,
    }


def main():
    print("Parsing analysis.txt...")
    up_proj_comps, down_proj_comps, up_proj_facts, down_proj_facts = parse_analysis_file(
        "analysis.txt"
    )

    print(f"\nFound {len(up_proj_comps)} up_proj components with facts")
    print(f"Found {len(down_proj_comps)} down_proj components with facts")
    print(f"Found {len(up_proj_facts)} facts with up_proj info")
    print(f"Found {len(down_proj_facts)} facts with down_proj info")

    # Sample check
    if up_proj_facts:
        sample_fact = list(up_proj_facts.items())[0]
        print(f"\nSample up_proj fact: {sample_fact}")
    if down_proj_facts:
        sample_fact = list(down_proj_facts.items())[0]
        print(f"Sample down_proj fact: {sample_fact}")

    # Compute monosemanticity for each component
    print("\nComputing component monosemanticity...")

    up_proj_stats = {}
    up_proj_mono_scores = {}
    for comp_id, facts in up_proj_comps.items():
        stats = compute_component_monosemanticity(facts)
        up_proj_stats[comp_id] = stats
        up_proj_mono_scores[comp_id] = compute_monosemanticity_score(stats)

    down_proj_stats = {}
    down_proj_mono_scores = {}
    for comp_id, facts in down_proj_comps.items():
        stats = compute_component_monosemanticity(facts)
        down_proj_stats[comp_id] = stats
        down_proj_mono_scores[comp_id] = compute_monosemanticity_score(stats)

    # Print some example monosemantic components
    print("\n" + "=" * 80)
    print("MONOSEMANTIC UP_PROJ COMPONENTS (threshold >= 0.9)")
    print("=" * 80)
    mono_up = []
    for comp_id, stats in up_proj_stats.items():
        is_mono, reason = is_component_monosemantic(stats, threshold=0.9)
        if is_mono:
            mono_up.append((comp_id, stats, reason))

    mono_up.sort(key=lambda x: compute_monosemanticity_score(x[1]), reverse=True)
    for comp_id, stats, reason in mono_up[:20]:
        print(
            f"  Component {comp_id}: {reason}, score={compute_monosemanticity_score(stats):.3f}, n_facts={stats['n_facts']}"
        )
    print(f"  ... and {len(mono_up) - 20} more" if len(mono_up) > 20 else "")

    print(f"\nTotal monosemantic up_proj components: {len(mono_up)} / {len(up_proj_stats)}")

    print("\n" + "=" * 80)
    print("MONOSEMANTIC DOWN_PROJ COMPONENTS (threshold >= 0.9)")
    print("=" * 80)
    mono_down = []
    for comp_id, stats in down_proj_stats.items():
        is_mono, reason = is_component_monosemantic(stats, threshold=0.9)
        if is_mono:
            mono_down.append((comp_id, stats, reason))

    mono_down.sort(key=lambda x: compute_monosemanticity_score(x[1]), reverse=True)
    for comp_id, stats, reason in mono_down[:20]:
        print(
            f"  Component {comp_id}: {reason}, score={compute_monosemanticity_score(stats):.3f}, n_facts={stats['n_facts']}"
        )
    print(f"  ... and {len(mono_down) - 20} more" if len(mono_down) > 20 else "")

    print(f"\nTotal monosemantic down_proj components: {len(mono_down)} / {len(down_proj_stats)}")

    # Combine up_proj and down_proj info for each fact
    print("\n" + "=" * 80)
    print("SCORING FACTS BY MONOSEMANTICITY")
    print("=" * 80)

    all_facts = set(up_proj_facts.keys()) | set(down_proj_facts.keys())
    fact_scores = []

    for fact_idx in all_facts:
        up_info = up_proj_facts.get(fact_idx, {})
        down_info = down_proj_facts.get(fact_idx, {})

        # Get the inputs and label from either source
        inputs = up_info.get("inputs") or down_info.get("inputs", [])
        label = up_info.get("label", down_info.get("label", -1))

        combined_info = {
            "inputs": inputs,
            "label": label,
            "up_proj_components": up_info.get("components", []),
            "down_proj_components": down_info.get("components", []),
        }

        score, details = score_fact(
            combined_info,
            up_proj_mono_scores,
            down_proj_mono_scores,
            up_proj_stats,
            down_proj_stats,
        )

        fact_scores.append(
            {
                "fact_idx": fact_idx,
                "inputs": inputs,
                "label": label,
                "score": score,
                "details": details,
            }
        )

    # Sort by score (highest = cleanest), then by mono ratio, then by fewer components
    fact_scores.sort(
        key=lambda x: (
            x["score"],
            x["details"].get("mono_ratio", 0),
            -x["details"].get("n_components", 999),
        ),
        reverse=True,
    )

    # Print top cleanest facts
    print("\nTOP 50 CLEANEST FACTS (highest monosemanticity score):")
    print("-" * 80)

    for i, fs in enumerate(fact_scores[:50]):
        up_comps = fs["details"].get("up_proj_components", [])
        down_comps = fs["details"].get("down_proj_components", [])

        up_str = ", ".join([f"C{c[0]}({c[4]})" for c in up_comps])
        down_str = ", ".join([f"C{c[0]}({c[4]})" for c in down_comps])

        print(f"\n{i + 1}. Fact {fs['fact_idx']}: input={fs['inputs']} → label={fs['label']}")
        print(f"   Score: {fs['score']:.3f}, mono_ratio: {fs['details'].get('mono_ratio', 0):.2f}")
        print(f"   Up_proj ({len(up_comps)}): {up_str if up_str else 'none'}")
        print(f"   Down_proj ({len(down_comps)}): {down_str if down_str else 'none'}")

    # Find facts where ALL components are monosemantic
    print("\n" + "=" * 80)
    print("FACTS WHERE ALL COMPONENTS ARE MONOSEMANTIC")
    print("=" * 80)

    all_mono_facts = [
        fs
        for fs in fact_scores
        if fs["details"].get("n_components", 0) > 0 and fs["details"].get("mono_ratio", 0) == 1.0
    ]

    print(f"\nFound {len(all_mono_facts)} facts where ALL components are monosemantic:\n")

    for i, fs in enumerate(all_mono_facts[:30]):
        up_comps = fs["details"].get("up_proj_components", [])
        down_comps = fs["details"].get("down_proj_components", [])

        up_str = ", ".join([f"C{c[0]}({c[4]})" for c in up_comps])
        down_str = ", ".join([f"C{c[0]}({c[4]})" for c in down_comps])

        print(f"{i + 1}. Fact {fs['fact_idx']}: input={fs['inputs']} → label={fs['label']}")
        print(f"   Up_proj: {up_str if up_str else 'none'}")
        print(f"   Down_proj: {down_str if down_str else 'none'}")
        print()

    if len(all_mono_facts) > 30:
        print(f"   ... and {len(all_mono_facts) - 30} more")

    # Also show facts with only 1 component firing in up_proj
    print("\n" + "=" * 80)
    print("FACTS WITH ONLY 1 UP_PROJ COMPONENT FIRING")
    print("=" * 80)

    single_comp_facts = [
        fs for fs in fact_scores if len(fs["details"].get("up_proj_components", [])) == 1
    ]
    single_comp_facts.sort(key=lambda x: x["score"], reverse=True)

    print(f"\nFound {len(single_comp_facts)} facts with only 1 up_proj component:\n")

    for i, fs in enumerate(single_comp_facts[:30]):
        up_comps = fs["details"].get("up_proj_components", [])
        down_comps = fs["details"].get("down_proj_components", [])

        comp_id = up_comps[0][0]
        comp_stats = up_proj_stats.get(comp_id)
        is_mono, reason = (
            is_component_monosemantic(comp_stats, threshold=0.9)
            if comp_stats
            else (False, "unknown")
        )

        down_str = ", ".join([f"C{c[0]}({c[4]})" for c in down_comps])

        print(f"{i + 1}. Fact {fs['fact_idx']}: input={fs['inputs']} → label={fs['label']}")
        print(
            f"   Up_proj C{comp_id}: mono_score={fs['score']:.3f}, is_mono={is_mono}, reason={reason}"
        )
        print(f"   Down_proj: {down_str if down_str else 'none'}")
        if comp_stats:
            print(
                f"   Component stats: dominant_label={comp_stats['dominant_label']} ({comp_stats['dominant_label_ratio']:.1%})"
            )
        print()

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Count facts with at least one component
    facts_with_components = [fs for fs in fact_scores if fs["details"].get("n_components", 0) > 0]
    print(f"\nTotal facts with at least one component: {len(facts_with_components)}")

    score_thresholds = [1.0, 0.95, 0.9, 0.8, 0.5, 0.0]
    for thresh in score_thresholds:
        count = sum(1 for fs in facts_with_components if fs["score"] >= thresh)
        print(f"  Facts with monosemanticity score >= {thresh}: {count}")

    # Save results to a file
    print("\n\nSaving detailed results to clean_facts_ranking.txt...")
    with open("clean_facts_ranking.txt", "w") as f:
        f.write("FACTS RANKED BY MONOSEMANTICITY SCORE\n")
        f.write("=" * 80 + "\n\n")
        f.write("A fact is 'clean' if all components that fire on it are monosemantic.\n")
        f.write(
            "Monosemantic = responds to a single label or single input position value (>= 90%).\n\n"
        )

        f.write(f"Total facts with at least one component: {len(facts_with_components)}\n")
        f.write(f"Facts where ALL components are monosemantic: {len(all_mono_facts)}\n\n")

        f.write("=" * 80 + "\n")
        f.write("CLEANEST FACTS (all components monosemantic)\n")
        f.write("=" * 80 + "\n\n")

        for i, fs in enumerate(all_mono_facts):
            up_comps = fs["details"].get("up_proj_components", [])
            down_comps = fs["details"].get("down_proj_components", [])

            f.write(f"Rank {i + 1}: Fact {fs['fact_idx']}\n")
            f.write(f"  Input: {fs['inputs']} → Label: {fs['label']}\n")
            f.write(f"  Monosemanticity Score: {fs['score']:.4f}\n")
            f.write(f"  Up_proj components ({len(up_comps)}):\n")
            for comp_id, mono_score, ci_score, _is_mono, reason in up_comps:
                f.write(
                    f"    C{comp_id}: CI={ci_score:.3f}, mono={mono_score:.3f}, reason={reason}\n"
                )
            f.write(f"  Down_proj components ({len(down_comps)}):\n")
            for comp_id, mono_score, ci_score, _is_mono, reason in down_comps:
                f.write(
                    f"    C{comp_id}: CI={ci_score:.3f}, mono={mono_score:.3f}, reason={reason}\n"
                )
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("ALL FACTS RANKED\n")
        f.write("=" * 80 + "\n\n")

        for i, fs in enumerate(facts_with_components):
            up_comps = fs["details"].get("up_proj_components", [])
            down_comps = fs["details"].get("down_proj_components", [])

            f.write(f"Rank {i + 1}: Fact {fs['fact_idx']}\n")
            f.write(f"  Input: {fs['inputs']} → Label: {fs['label']}\n")
            f.write(f"  Min Monosemanticity Score: {fs['score']:.4f}\n")
            f.write(
                f"  Mono ratio: {fs['details'].get('mono_ratio', 0):.2f} ({fs['details'].get('n_mono_components', 0)}/{fs['details'].get('n_components', 0)})\n"
            )
            f.write(f"  Up_proj components ({len(up_comps)}):\n")
            for comp_id, mono_score, ci_score, is_mono, reason in up_comps:
                mono_marker = "✓" if is_mono else "✗"
                f.write(
                    f"    {mono_marker} C{comp_id}: CI={ci_score:.3f}, mono={mono_score:.3f}, reason={reason}\n"
                )
            f.write(f"  Down_proj components ({len(down_comps)}):\n")
            for comp_id, mono_score, ci_score, is_mono, reason in down_comps:
                mono_marker = "✓" if is_mono else "✗"
                f.write(
                    f"    {mono_marker} C{comp_id}: CI={ci_score:.3f}, mono={mono_score:.3f}, reason={reason}\n"
                )
            f.write("\n")

    print("Done!")


if __name__ == "__main__":
    main()
