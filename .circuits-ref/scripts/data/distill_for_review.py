#!/usr/bin/env python3
"""
Distill analysis JSON files into a compact format for LLM review.

Removes verbose/redundant data while preserving the essential reasoning chain
information needed to identify shortcuts, spurious correlations, and incorrect reasoning.
"""

import argparse
import json
from pathlib import Path


def extract_prompt_info(analysis: dict, filename: str) -> dict:
    """Extract prompt and answer information."""
    # Try to get from llm_prompt
    llm_prompt = analysis.get('llm_prompt', '')
    prompt = None
    if 'answering the prompt:' in llm_prompt:
        start = llm_prompt.find('answering the prompt:') + len('answering the prompt:')
        end = llm_prompt.find('\n\n', start)
        if end > start:
            prompt = llm_prompt[start:end].strip().strip('"')

    if not prompt:
        # Extract from filename
        name = Path(filename).stem
        if name.endswith('-analysis'):
            name = name[:-9]
        if name.startswith('relp-'):
            name = name[5:]
        prompt = name.replace('-', ' ')

    return {'prompt': prompt}


def distill_module(mod: dict, max_neurons: int = 3) -> dict:
    """Distill a module to its essential information."""
    # Get top neurons with truncated labels
    top_neurons = []
    for n in mod.get('top_neurons', [])[:max_neurons]:
        label = n.get('label', '')
        # Remove the L#/N#### prefix if present
        if ': ' in label:
            label = label.split(': ', 1)[1]
        # Truncate long labels
        if len(label) > 120:
            label = label[:120] + '...'
        top_neurons.append({
            'desc': label,
            'influence': round(n.get('influence', 0), 2)
        })

    # Get top tokens as simple list
    top_tokens = [t[0] for t in mod.get('top_tokens', [])[:3]]

    return {
        'id': mod.get('cluster_id'),
        'name': mod.get('name', f"Module {mod.get('cluster_id')}"),
        'layers': f"{mod.get('layer_range', [0,0])[0]}-{mod.get('layer_range', [0,0])[1]}",
        'tokens': top_tokens,
        'influence': round(mod.get('total_influence', 0), 1),
        'flow_out': round(mod.get('outgoing_flow', 0), 2),
        'flow_in': round(mod.get('incoming_flow', 0), 2),
        'function': mod.get('function', '')[:200] + ('...' if len(mod.get('function', '')) > 200 else ''),
        'neurons': top_neurons
    }


def extract_key_edges(flow_matrix: list, modules: list, top_k: int = 15) -> list:
    """Extract the most important edges from the flow matrix."""
    if not flow_matrix or not modules:
        return []

    n = len(flow_matrix)
    edges = []

    for i in range(n):
        for j in range(n):
            if i != j and abs(flow_matrix[i][j]) > 0.2:
                edges.append({
                    'from': i,
                    'to': j,
                    'weight': round(flow_matrix[i][j], 2)
                })

    # Sort by absolute weight and take top k
    edges.sort(key=lambda x: abs(x['weight']), reverse=True)
    edges = edges[:top_k]

    # Add module names
    for edge in edges:
        if edge['from'] < len(modules):
            edge['from_name'] = modules[edge['from']].get('name', f"M{edge['from']}")
        if edge['to'] < len(modules):
            edge['to_name'] = modules[edge['to']].get('name', f"M{edge['to']}")

    return edges


def categorize_modules(modules: list) -> dict:
    """Group modules by processing stage."""
    early = []  # layers 0-5
    mid = []    # layers 6-19
    late = []   # layers 20+

    for mod in modules:
        distilled = distill_module(mod)
        layer_mean = mod.get('layer_mean', 0)

        if layer_mean < 6:
            early.append(distilled)
        elif layer_mean < 20:
            mid.append(distilled)
        else:
            late.append(distilled)

    return {
        'input_processing': early,
        'reasoning': mid,
        'output_generation': late
    }


def distill_analysis(analysis: dict, filename: str) -> dict:
    """Distill a full analysis to compact form."""
    modules = analysis.get('module_summaries', [])

    # Build distilled structure
    distilled = extract_prompt_info(analysis, filename)
    distilled['n_modules'] = len(modules)

    # Categorize modules by stage
    distilled['modules'] = categorize_modules(modules)

    # Extract key information flow edges
    distilled['key_edges'] = extract_key_edges(
        analysis.get('flow_matrix', []),
        modules
    )

    # Include the LLM synthesis narrative (this is actually very informative)
    synthesis = analysis.get('llm_synthesis', '')
    if synthesis:
        # Extract just the circuit narrative section if present
        if '## 2)' in synthesis:
            start = synthesis.find('## 2)')
            end = synthesis.find('## 3)', start)
            if end > start:
                narrative = synthesis[start:end].strip()
                distilled['circuit_narrative'] = narrative
            else:
                distilled['circuit_narrative'] = synthesis[start:start+2000]
        else:
            # Just take a reasonable chunk
            distilled['circuit_narrative'] = synthesis[:2000]

    return distilled


def format_for_llm(distilled: dict) -> str:
    """Format distilled analysis as readable text for LLM input."""
    lines = []

    lines.append(f"## Prompt: \"{distilled['prompt']}\"")
    lines.append(f"Total modules: {distilled['n_modules']}")
    lines.append("")

    # Input processing modules
    lines.append("### Input Processing (Layers 0-5)")
    for mod in distilled['modules']['input_processing']:
        lines.append(f"**{mod['name']}** (L{mod['layers']}, tokens: {mod['tokens']}, influence: {mod['influence']})")
        lines.append(f"  Function: {mod['function']}")
        lines.append("  Key neurons:")
        for n in mod['neurons']:
            lines.append(f"    - {n['desc']} (inf={n['influence']})")
        lines.append("")

    # Reasoning modules
    lines.append("### Reasoning (Layers 6-19)")
    for mod in distilled['modules']['reasoning']:
        lines.append(f"**{mod['name']}** (L{mod['layers']}, tokens: {mod['tokens']}, influence: {mod['influence']})")
        lines.append(f"  Function: {mod['function']}")
        lines.append("  Key neurons:")
        for n in mod['neurons']:
            lines.append(f"    - {n['desc']} (inf={n['influence']})")
        lines.append("")

    # Output modules
    lines.append("### Output Generation (Layers 20+)")
    for mod in sorted(distilled['modules']['output_generation'], key=lambda x: x['flow_out'], reverse=True):
        lines.append(f"**{mod['name']}** (L{mod['layers']}, flow_out: {mod['flow_out']}, influence: {mod['influence']})")
        lines.append(f"  Function: {mod['function']}")
        lines.append("  Key neurons:")
        for n in mod['neurons']:
            lines.append(f"    - {n['desc']} (inf={n['influence']})")
        lines.append("")

    # Key edges
    lines.append("### Key Information Flow")
    for edge in distilled['key_edges']:
        arrow = '→' if edge['weight'] > 0 else '⊣'
        lines.append(f"  {edge.get('from_name', edge['from'])} {arrow} {edge.get('to_name', edge['to'])} (w={edge['weight']})")
    lines.append("")

    # Circuit narrative from original LLM analysis
    if 'circuit_narrative' in distilled:
        lines.append("### Circuit Narrative (from original analysis)")
        lines.append(distilled['circuit_narrative'])

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Distill analysis JSON for LLM review',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Distill single file to JSON
  %(prog)s analysis.json -o distilled.json

  # Distill directory to text format for LLM
  %(prog)s outputs/knowledge_circuits/ --format text -o review.txt

  # Batch multiple analyses into single file
  %(prog)s outputs/knowledge_circuits/ --format text --batch
"""
    )
    parser.add_argument('inputs', nargs='+', help='Analysis JSON files or directories')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('--format', choices=['json', 'text'], default='json',
                       help='Output format: json (structured) or text (for LLM input)')
    parser.add_argument('--batch', action='store_true',
                       help='Combine all analyses into single output')
    parser.add_argument('--max-neurons', type=int, default=3,
                       help='Max neurons per module (default: 3)')
    args = parser.parse_args()

    # Collect analysis files
    analysis_files = []
    for inp in args.inputs:
        path = Path(inp)
        if path.is_dir():
            analysis_files.extend(path.glob('*-analysis.json'))
        elif path.exists():
            analysis_files.append(path)

    # Process files
    results = []
    for f in sorted(analysis_files):
        try:
            with open(f) as fp:
                analysis = json.load(fp)
            distilled = distill_analysis(analysis, str(f))
            results.append((f.stem, distilled))
        except Exception as e:
            print(f"Error processing {f}: {e}", file=__import__('sys').stderr)

    # Format output
    if args.format == 'json':
        if args.batch:
            output = json.dumps([d for _, d in results], indent=2)
        else:
            output = '\n'.join(json.dumps(d, indent=2) for _, d in results)
    else:  # text
        if args.batch:
            sections = []
            for name, distilled in results:
                sections.append(format_for_llm(distilled))
            output = '\n\n---\n\n'.join(sections)
        else:
            output = '\n\n---\n\n'.join(format_for_llm(d) for _, d in results)

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Written {len(results)} analyses to {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()
