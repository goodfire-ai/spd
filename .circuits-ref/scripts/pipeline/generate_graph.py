#!/usr/bin/env python3
"""Generate RelP attribution graphs compatible with the Anthropic graph viewer.

Supported models:
    - Llama 3.1-8B-Instruct (default)
    - Qwen 3 32B
    - OLMo 7B / OLMo 2 7B

Usage:
    python scripts/generate_graph.py "What is the capital of France?"
    python scripts/generate_graph.py "What is the capital of France?" --slug france-capital
    python scripts/generate_graph.py "The capital of France is" --raw  # No chat template
    python scripts/generate_graph.py "Hello world" --no-labels  # Skip Transluce labels
    python scripts/generate_graph.py "Hello" --model allenai/OLMo-7B  # Use OLMo

Output:
    Graphs are saved to graphs/<slug>.json in Anthropic viewer format.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from circuits import RelPAttributor, RelPConfig
from circuits.model_configs import detect_model_family, get_model_config
from transformers import AutoModelForCausalLM, AutoTokenizer


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:50].strip('-')


def format_for_anthropic_viewer(graph: dict, slug: str, model_name: str = "llama-3.1-8b") -> dict:
    """Format the graph for the Anthropic attribution graph viewer.

    Args:
        graph: Raw graph from RelPAttributor
        slug: URL slug for the graph
        model_name: Model identifier (e.g., 'llama-3.1-8b', 'olmo-7b')
    """
    # Look up model config for canonical name
    model_config = get_model_config(model_name)
    scan_name = model_config.name if model_config else model_name

    # Update metadata
    graph['metadata']['slug'] = slug
    graph['metadata']['scan'] = scan_name

    # Fix layer values to strings and add required fields
    for node in graph['nodes']:
        if node['layer'] == 'embedding':
            node['layer'] = 'E'
        elif node['layer'] == 'logit':
            # Logit layer is max neuron layer + 1
            max_neuron_layer = max(
                int(n['layer']) for n in graph['nodes']
                if n['layer'] not in ['embedding', 'logit', 'E'] and isinstance(n['layer'], (int, str)) and str(n['layer']).isdigit()
            )
            node['layer'] = str(max_neuron_layer + 1)
            node['isLogit'] = True
            # Format clerp for logits (probability is already included from relp.py)
            if 'Logit:' in node.get('clerp', ''):
                # Remove "Logit: " prefix, keep token and probability
                node['clerp'] = node['clerp'].replace('Logit: ', '')
                node['feature_type'] = 'logit'
        else:
            node['layer'] = str(node['layer'])

        if 'isLogit' not in node:
            node['isLogit'] = False

    # Add qParams defaults
    graph['qParams'] = {
        'pinnedIds': [],
        'supernodes': [],
        'linkType': 'both'
    }

    # Add features array
    features = []
    seen_features = set()
    for node in graph['nodes']:
        feature_id = f"{node['layer']}_{node['feature']}"
        if feature_id not in seen_features:
            seen_features.add(feature_id)
            features.append({
                'featureId': feature_id,
                'featureIndex': node['feature'],
                'layer': node['layer'],
                'clerp': node['clerp'],
                'feature_type': node['feature_type']
            })
    graph['features'] = features

    return graph


def add_transluce_labels(graph_path: Path) -> None:
    """Add neuron labels from Transluce using the relabel script."""
    relabel_script = Path(__file__).parent.parent / 'relabel_neurons.py'

    try:
        result = subprocess.run(
            [sys.executable, str(relabel_script), str(graph_path), '--inplace'],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            # Count successful labels from output
            if 'Successfully fetched' in result.stdout:
                print(result.stdout.split('\n')[-3])  # Print success line
        else:
            print(f"Warning: Could not fetch labels: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("Warning: Label fetching timed out")
    except FileNotFoundError:
        print("Warning: relabel_neurons.py not found, skipping labels")


# Minimal Llama 3.1 chat template without date/knowledge cutoff info
# Note: We omit bos_token here since the tokenizer adds it automatically during encoding
LLAMA_MINIMAL_CHAT_TEMPLATE = '''{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set system_message = '' %}
    {%- set loop_messages = messages %}
{%- endif %}
<|start_header_id|>system<|end_header_id|>

{{ system_message }}<|eot_id|>
{%- for message in loop_messages %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] }}
{%- if not loop.last or add_generation_prompt %}<|eot_id|>{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}'''


def apply_chat_template(tokenizer, prompt: str, answer_prefix: str = "", model_name: str = None) -> str:
    """Apply chat template to a prompt.

    Uses the tokenizer's built-in template when available, falling back to
    model-specific templates for known models.

    Args:
        tokenizer: HuggingFace tokenizer
        prompt: User message content
        answer_prefix: Optional prefix for assistant response (e.g., "Answer:")
        model_name: Model name/path for template selection

    Returns:
        Formatted prompt string
    """
    messages = [{"role": "user", "content": prompt}]

    # Determine which template to use
    model_family = detect_model_family(model_name) if model_name else "llama"

    # For Llama, use our minimal template (avoids date/knowledge cutoff info)
    # For other models, try the tokenizer's built-in template first
    if model_family == "llama":
        chat_template = LLAMA_MINIMAL_CHAT_TEMPLATE
    else:
        # Use tokenizer's built-in template if available
        chat_template = None  # None = use tokenizer default

    if answer_prefix:
        messages.append({"role": "assistant", "content": answer_prefix})
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                chat_template=chat_template,
                continue_final_message=True
            )
        except TypeError:
            # Some tokenizers don't support continue_final_message
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                chat_template=chat_template,
            )
    else:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=chat_template
        )
    return formatted


def main():
    parser = argparse.ArgumentParser(description='Generate RelP attribution graph')
    parser.add_argument('prompt', help='Input prompt to attribute')
    parser.add_argument('--slug', help='URL slug for the graph (auto-generated if not provided)')
    parser.add_argument('--no-labels', action='store_true', help='Skip fetching Transluce neuron labels')
    parser.add_argument('--raw', action='store_true', help='Use raw prompt without chat template')
    parser.add_argument('--answer-prefix', default='', help='Prefill assistant response (e.g., "Answer:")')
    parser.add_argument('--k', type=int, default=5, help='Number of top logits (default: 5)')
    parser.add_argument('--tau', type=float, default=0.005, help='Node threshold (default: 0.005)')
    parser.add_argument('--target-tokens', type=str, nargs='+', help='Specific tokens to trace (e.g., " Yes" " No"). Overrides --k')
    parser.add_argument('--contrastive', type=str, nargs=2, metavar=('POS', 'NEG'),
                        help='Contrastive attribution: trace logit(POS) - logit(NEG). E.g., --contrastive Yes No')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory (default: graphs/)')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Model name or path (default: meta-llama/Llama-3.1-8B-Instruct)')
    args = parser.parse_args()

    # Determine output directory
    repo_root = Path(__file__).parent.parent
    output_dir = args.output_dir or (repo_root / 'graphs')
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = args.slug or f"relp-{slugify(args.prompt)}"

    print(f"Loading model: {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Apply chat template for instruct model (unless --raw)
    if args.raw:
        formatted_prompt = args.prompt
        print(f"Computing attributions for (raw): {args.prompt}", flush=True)
    else:
        formatted_prompt = apply_chat_template(tokenizer, args.prompt, args.answer_prefix, args.model)
        print(f"Computing attributions for: {args.prompt}", flush=True)
        if args.answer_prefix:
            print(f"Answer prefix: {args.answer_prefix}", flush=True)
        print(f"Formatted prompt length: {len(formatted_prompt)} chars", flush=True)

    config = RelPConfig(k=args.k, tau=args.tau, target_tokens=args.target_tokens,
                        contrastive_tokens=args.contrastive, use_neuron_labels=False)
    attributor = RelPAttributor(model, tokenizer, config=config, model_name=args.model)
    graph = attributor.compute_attributions(formatted_prompt, target_tokens=args.target_tokens,
                                           contrastive_tokens=args.contrastive)
    attributor.cleanup()

    # Store original prompt for display (not the formatted template version)
    graph['metadata']['prompt'] = args.prompt
    graph['metadata']['model'] = args.model

    print("Formatting for Anthropic viewer...", flush=True)
    graph = format_for_anthropic_viewer(graph, slug, model_name=args.model)

    # Save graph
    graph_path = output_dir / f'{slug}.json'
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"Saved graph to: {graph_path}")

    # Add Transluce labels
    if not args.no_labels:
        print("Fetching neuron labels from Transluce...")
        add_transluce_labels(graph_path)

    # Print summary
    with open(graph_path) as f:
        final_graph = json.load(f)

    print(f"\n{'='*50}")
    print("Graph generated successfully!")
    print(f"  Nodes: {len(final_graph['nodes'])}")
    print(f"  Links: {len(final_graph['links'])}")
    print(f"  Slug: {slug}")
    print(f"  Output: {graph_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
