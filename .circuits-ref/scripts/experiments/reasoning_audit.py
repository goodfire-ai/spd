#!/usr/bin/env python3
"""
Convert analysis JSON files to a format for auditing reasoning quality.

Identifies potential "incorrect reasoning" patterns:
- Spurious correlations (high-influence neurons unrelated to domain)
- Shortcuts (direct lexical → answer without semantic processing)
- Missing intermediate concepts
- Format/style modules driving content decisions
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReasoningAudit:
    """Audit result for a single analysis."""
    prompt: str
    traced_tokens: list[str]

    # Reasoning chain
    input_modules: list[dict]  # Early modules processing input
    reasoning_modules: list[dict]  # Mid modules doing inference
    output_modules: list[dict]  # Late modules producing output

    # Red flags
    red_flags: list[dict] = field(default_factory=list)

    # Key statistics
    total_modules: int = 0
    semantic_modules: int = 0  # Modules with domain-relevant neurons
    format_modules: int = 0  # Modules primarily about formatting

    # Flow analysis
    main_pathway: list[tuple] = field(default_factory=list)  # Key edges
    inhibited_modules: list[dict] = field(default_factory=list)


def extract_prompt_from_filename(filename: str) -> str:
    """Extract original prompt from analysis filename."""
    # Pattern: relp-{slugified-prompt}-analysis.json
    name = Path(filename).stem
    if name.endswith('-analysis'):
        name = name[:-9]
    if name.startswith('relp-'):
        name = name[5:]
    # Convert slug back to readable text
    return name.replace('-', ' ')


def classify_module_type(module: dict, llm_synthesis: str = "") -> str:
    """Classify module as semantic, format, or mixed."""
    function = module.get('function', '').lower()
    name = module.get('name', '').lower()

    format_keywords = [
        'format', 'punctuation', 'colon', 'delimiter', 'style',
        'scaffold', 'syntax', 'structure', 'answer header', 'termination',
        'citation', 'numeric', 'table', 'pipe'
    ]

    semantic_keywords = [
        'concept', 'semantic', 'detection', 'recognizer', 'domain',
        'consolidation', 'hub', 'assembly', 'integration'
    ]

    for kw in format_keywords:
        if kw in function or kw in name:
            return 'format'

    for kw in semantic_keywords:
        if kw in function or kw in name:
            return 'semantic'

    return 'mixed'


def identify_red_flags(analysis: dict) -> list[dict]:
    """Identify potential reasoning issues."""
    flags = []
    modules = analysis.get('module_summaries', [])
    flow_matrix = analysis.get('flow_matrix', [])
    synthesis = analysis.get('llm_synthesis', '')

    # Check for spurious correlations mentioned in synthesis
    spurious_patterns = [
        r'spurious',
        r'incidental',
        r'unrelated',
        r'irrelevant',
        r'artifact',
        r'noise',
        r'nuisance'
    ]

    for pattern in spurious_patterns:
        matches = re.findall(rf'Module \*?\*?(\d+)[^.]*{pattern}[^.]*\.', synthesis, re.IGNORECASE)
        for match in matches:
            module_id = int(match)
            if module_id < len(modules):
                mod = modules[module_id]
                flags.append({
                    'type': 'spurious_correlation',
                    'module_id': module_id,
                    'module_name': mod.get('name', f'Module {module_id}'),
                    'description': "Module mentioned as spurious/irrelevant in analysis",
                    'severity': 'medium'
                })

    # Check for high-influence format modules
    for mod in modules:
        mod_type = classify_module_type(mod)
        influence = mod.get('total_influence', 0)
        outgoing = mod.get('outgoing_flow', 0)

        if mod_type == 'format' and influence > 5.0 and outgoing > 1.0:
            flags.append({
                'type': 'format_driving_content',
                'module_id': mod.get('cluster_id'),
                'module_name': mod.get('name', f"Module {mod.get('cluster_id')}"),
                'description': f"Formatting module has high influence ({influence:.1f}) and outgoing flow ({outgoing:.1f})",
                'severity': 'low'
            })

    # Check for shortcuts (early module directly connecting to late output)
    for mod in modules:
        layer_mean = mod.get('layer_mean', 0)
        if layer_mean < 5:  # Early module
            neurons = mod.get('top_neurons', [])
            for neuron in neurons:
                label = neuron.get('label', '').lower()
                # Check if early module contains answer-related concepts
                if 'dopamine' in label or 'serotonin' in label or 'neurotransmitter' in label:
                    if neuron.get('influence', 0) > 1.0:
                        flags.append({
                            'type': 'shortcut_detected',
                            'module_id': mod.get('cluster_id'),
                            'module_name': mod.get('name', f"Module {mod.get('cluster_id')}"),
                            'description': f"Early module (layer ~{layer_mean:.0f}) contains answer concept: {label[:80]}...",
                            'severity': 'high'
                        })

    # Check for unexpected content in modules
    unexpected_content = [
        ('sexual', 'climax'),
        ('orgasm', 'sexual'),
    ]

    for mod in modules:
        neurons = mod.get('top_neurons', [])
        for neuron in neurons:
            label = neuron.get('label', '').lower()
            for keyword, context in unexpected_content:
                if keyword in label:
                    flags.append({
                        'type': 'unexpected_content',
                        'module_id': mod.get('cluster_id'),
                        'module_name': mod.get('name', f"Module {mod.get('cluster_id')}"),
                        'description': f"Unexpected content in module: '{keyword}' detected",
                        'severity': 'medium'
                    })

    return flags


def extract_main_pathway(analysis: dict) -> list[dict]:
    """Extract the main reasoning pathway from flow matrix."""
    modules = analysis.get('module_summaries', [])
    flow_matrix = analysis.get('flow_matrix', [])

    if not flow_matrix or not modules:
        return []

    n = len(flow_matrix)
    pathway = []

    # Find strongest edges
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and abs(flow_matrix[i][j]) > 0.3:
                edges.append({
                    'from': i,
                    'to': j,
                    'weight': flow_matrix[i][j],
                    'type': 'excitatory' if flow_matrix[i][j] > 0 else 'inhibitory'
                })

    # Sort by absolute weight
    edges.sort(key=lambda x: abs(x['weight']), reverse=True)

    # Take top 10 edges
    for edge in edges[:10]:
        from_mod = modules[edge['from']] if edge['from'] < len(modules) else {}
        to_mod = modules[edge['to']] if edge['to'] < len(modules) else {}

        pathway.append({
            'from_id': edge['from'],
            'from_name': from_mod.get('name', f"Module {edge['from']}"),
            'to_id': edge['to'],
            'to_name': to_mod.get('name', f"Module {edge['to']}"),
            'weight': edge['weight'],
            'type': edge['type']
        })

    return pathway


def categorize_modules_by_stage(modules: list[dict]) -> tuple[list, list, list]:
    """Split modules into input, reasoning, and output stages."""
    input_mods = []
    reasoning_mods = []
    output_mods = []

    for mod in modules:
        layer_mean = mod.get('layer_mean', 0)
        summary = {
            'id': mod.get('cluster_id'),
            'name': mod.get('name', f"Module {mod.get('cluster_id')}"),
            'function': mod.get('function', ''),
            'layer_range': mod.get('layer_range', []),
            'top_tokens': [t[0] for t in mod.get('top_tokens', [])[:3]],
            'influence': mod.get('total_influence', 0),
            'outgoing_flow': mod.get('outgoing_flow', 0),
            'type': classify_module_type(mod)
        }

        if layer_mean < 6:
            input_mods.append(summary)
        elif layer_mean < 20:
            reasoning_mods.append(summary)
        else:
            output_mods.append(summary)

    return input_mods, reasoning_mods, output_mods


def create_audit(analysis_path: Path) -> ReasoningAudit:
    """Create an audit from an analysis file."""
    with open(analysis_path) as f:
        analysis = json.load(f)

    # Extract prompt
    prompt = extract_prompt_from_filename(analysis_path.name)

    # Get traced tokens from module summaries
    traced_tokens = set()
    for mod in analysis.get('module_summaries', []):
        for token, _ in mod.get('top_tokens', []):
            traced_tokens.add(token)

    # Categorize modules
    input_mods, reasoning_mods, output_mods = categorize_modules_by_stage(
        analysis.get('module_summaries', [])
    )

    # Find red flags
    red_flags = identify_red_flags(analysis)

    # Extract main pathway
    main_pathway = extract_main_pathway(analysis)

    # Count module types
    modules = analysis.get('module_summaries', [])
    semantic_count = sum(1 for m in modules if classify_module_type(m) == 'semantic')
    format_count = sum(1 for m in modules if classify_module_type(m) == 'format')

    return ReasoningAudit(
        prompt=prompt,
        traced_tokens=list(traced_tokens),
        input_modules=input_mods,
        reasoning_modules=reasoning_mods,
        output_modules=output_mods,
        red_flags=red_flags,
        total_modules=len(modules),
        semantic_modules=semantic_count,
        format_modules=format_count,
        main_pathway=main_pathway
    )


def audit_to_markdown(audit: ReasoningAudit) -> str:
    """Convert audit to markdown format for review."""
    lines = []

    # Header
    lines.append(f"# Reasoning Audit: {audit.prompt[:60]}...")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append(f"- **Traced tokens**: {', '.join(audit.traced_tokens[:10])}")
    lines.append(f"- **Total modules**: {audit.total_modules}")
    lines.append(f"- **Semantic modules**: {audit.semantic_modules}")
    lines.append(f"- **Format modules**: {audit.format_modules}")
    lines.append("")

    # Red Flags
    if audit.red_flags:
        lines.append("## ⚠️ Red Flags")
        for flag in audit.red_flags:
            severity_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(flag['severity'], '⚪')
            lines.append(f"- {severity_icon} **{flag['type']}** (Module {flag['module_id']}: {flag['module_name']})")
            lines.append(f"  - {flag['description']}")
        lines.append("")
    else:
        lines.append("## ✅ No Red Flags Detected")
        lines.append("")

    # Reasoning Chain
    lines.append("## Reasoning Chain")
    lines.append("")

    lines.append("### Input Processing (Layers 0-5)")
    for mod in audit.input_modules:
        icon = '🔤' if mod['type'] == 'format' else '🧠'
        lines.append(f"- {icon} **{mod['name']}** (L{mod['layer_range'][0]}-{mod['layer_range'][1]})")
        lines.append(f"  - Tokens: {', '.join(mod['top_tokens'])}")
        lines.append(f"  - {mod['function'][:100]}...")
    lines.append("")

    lines.append("### Reasoning (Layers 6-19)")
    for mod in audit.reasoning_modules:
        icon = '🔤' if mod['type'] == 'format' else '🧠'
        lines.append(f"- {icon} **{mod['name']}** (L{mod['layer_range'][0]}-{mod['layer_range'][1]})")
        lines.append(f"  - Tokens: {', '.join(mod['top_tokens'])}")
        lines.append(f"  - {mod['function'][:100]}...")
    lines.append("")

    lines.append("### Output Generation (Layers 20+)")
    for mod in sorted(audit.output_modules, key=lambda x: x['outgoing_flow'], reverse=True)[:8]:
        icon = '🔤' if mod['type'] == 'format' else '🧠'
        lines.append(f"- {icon} **{mod['name']}** (L{mod['layer_range'][0]}-{mod['layer_range'][1]}, flow={mod['outgoing_flow']:.2f})")
        lines.append(f"  - {mod['function'][:100]}...")
    lines.append("")

    # Main Information Flow
    lines.append("## Key Information Pathways")
    for edge in audit.main_pathway[:10]:
        arrow = '→' if edge['type'] == 'excitatory' else '⊣'
        lines.append(f"- {edge['from_name']} {arrow} {edge['to_name']} (w={edge['weight']:.2f})")
    lines.append("")

    return '\n'.join(lines)


def audit_to_json(audit: ReasoningAudit) -> dict:
    """Convert audit to JSON-serializable dict."""
    return {
        'prompt': audit.prompt,
        'traced_tokens': audit.traced_tokens,
        'summary': {
            'total_modules': audit.total_modules,
            'semantic_modules': audit.semantic_modules,
            'format_modules': audit.format_modules,
            'red_flag_count': len(audit.red_flags),
            'high_severity_flags': sum(1 for f in audit.red_flags if f['severity'] == 'high')
        },
        'red_flags': audit.red_flags,
        'reasoning_chain': {
            'input': audit.input_modules,
            'reasoning': audit.reasoning_modules,
            'output': audit.output_modules
        },
        'main_pathway': audit.main_pathway
    }


def main():
    parser = argparse.ArgumentParser(description='Audit reasoning quality of analysis files')
    parser.add_argument('inputs', nargs='+', help='Analysis JSON files or directories')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('--format', choices=['markdown', 'json', 'summary'], default='markdown',
                       help='Output format')
    parser.add_argument('--flagged-only', action='store_true',
                       help='Only show analyses with red flags')
    args = parser.parse_args()

    # Collect all analysis files
    analysis_files = []
    for inp in args.inputs:
        path = Path(inp)
        if path.is_dir():
            analysis_files.extend(path.glob('*-analysis.json'))
        elif path.exists():
            analysis_files.append(path)

    # Process each file
    audits = []
    for f in sorted(analysis_files):
        try:
            audit = create_audit(f)
            if args.flagged_only and not audit.red_flags:
                continue
            audits.append((f, audit))
        except Exception as e:
            print(f"Error processing {f}: {e}", file=__import__('sys').stderr)

    # Format output
    if args.format == 'json':
        output = json.dumps([audit_to_json(a) for _, a in audits], indent=2)
    elif args.format == 'summary':
        lines = ["# Reasoning Audit Summary", ""]
        lines.append(f"Total analyses: {len(audits)}")
        flagged = sum(1 for _, a in audits if a.red_flags)
        lines.append(f"With red flags: {flagged}")
        lines.append("")
        lines.append("## Flagged Analyses")
        for f, audit in audits:
            if audit.red_flags:
                high = sum(1 for flag in audit.red_flags if flag['severity'] == 'high')
                lines.append(f"- **{audit.prompt[:50]}...** ({len(audit.red_flags)} flags, {high} high)")
                for flag in audit.red_flags:
                    if flag['severity'] == 'high':
                        lines.append(f"  - 🔴 {flag['type']}: {flag['description'][:60]}...")
        output = '\n'.join(lines)
    else:  # markdown
        output = '\n---\n\n'.join(audit_to_markdown(a) for _, a in audits)

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()
