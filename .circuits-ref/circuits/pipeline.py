"""Pipeline orchestrator for attribution graph analysis.

Runs the complete pipeline:
1. Generate attribution graph (RelP)
2. Label neurons from database
3. Cluster neurons (Infomap)
4. Analyze modules with LLM

Usage:
    from circuits.pipeline import run_pipeline, PipelineConfig

    # Single prompt
    result = run_pipeline("What is the capital of France?")

    # With custom config
    config = PipelineConfig(functional_split=True)
    result = run_pipeline("Your prompt", config=config)

    # Batch processing (serial)
    results = run_batch([
        {"prompt": "What is the capital of France?"},
        {"prompt": "The Eiffel Tower is in", "answer_prefix": " Paris"},
    ], config=config)

    # Batch processing with parallel LLM analysis
    results = run_batch([...], config=config, parallel_llm=True)

    # From config file
    results = run_from_config("config.yaml")
"""

import asyncio
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from .analysis import analyze_modules
from .clustering import cluster_graph
from .labeling import label_graph
from .relp import RelPAttributor, RelPConfig

# Default output directory
DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass
class PipelineConfig:
    """Configuration for the attribution pipeline.

    All settings can be overridden via config file or constructor arguments.
    Use PipelineConfig.generate_example() to create a sample config file.
    """

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Graph generation settings
    k: int = 5  # Number of top logits to trace
    tau: float = 0.005  # Node influence threshold
    target_tokens: list[str] = None  # Specific tokens to trace (e.g., [" Yes"]). Overrides k.
    contrastive_tokens: list[str] = None  # Contrastive pair [pos, neg]. Traces logit(pos) - logit(neg).

    # Template settings
    use_chat_template: bool = True  # Wrap prompt in Llama 3.1 Instruct template
    answer_prefix: str = ""  # Prefill assistant response

    # Labeling settings
    label_neurons: bool = True  # Fetch neuron descriptions from database

    # Clustering settings
    skip_special_tokens: bool = True  # Exclude BOS, headers from clustering
    min_cluster_size: int = 5  # Min size for cluster subdivision
    max_cluster_depth: int = 1  # Recursion depth (1 = subdivide once)

    # Analysis settings
    run_llm_analysis: bool = True  # Run LLM circuit synthesis
    llm_model: str = "gpt-5.2"  # LLM model (gpt-5.2, claude-*, etc.)
    llm_provider: str = "auto"  # Provider (auto, anthropic, openai)

    # Functional splitting settings
    functional_split: bool = True  # Split modules into functional sub-modules
    functional_split_min_size: int = 2  # Min neurons to consider splitting
    use_prompt_answer_split: bool = True  # Split by prompt vs answer tokens first
    use_position_split: bool = True  # Split by contiguous token position spans
    max_position_gap: int = 3  # Max gap between positions before splitting
    use_layer_split: bool = False  # Split by layer ranges (early/mid/late)
    use_semantic_split: bool = False  # Split by neuron label semantics (sentence-transformers). Don't use with llm_split.
    use_llm_split: bool = True  # Use LLM to intelligently split based on neuron functions
    use_llm_reassignment: bool = True  # Use LLM for final module cleanup/reassignment

    # Output settings
    output_dir: Path | None = None  # Default: outputs/
    save_intermediate: bool = True  # Save graph and cluster files

    # Batch processing settings
    parallel_llm: bool = True  # Run LLM analysis in parallel for batch processing

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary (for serialization)."""
        d = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Path):
                val = str(val)
            d[f.name] = val
        return d

    def to_yaml(self, path: Path | None = None) -> str:
        """Export config to YAML string or file."""
        content = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path:
            Path(path).write_text(content)
        return content

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary."""
        # Convert output_dir to Path if present
        if 'output_dir' in d and d['output_dir'] is not None:
            d['output_dir'] = Path(d['output_dir'])
        # Filter to only valid fields
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load config from YAML file."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d.get('config', d))

    @classmethod
    def generate_example(cls, path: Path | None = None) -> str:
        """Generate an example config file with comments."""
        example = '''# Attribution Graph Pipeline Configuration
# See PipelineConfig for all available options

config:
  # Model settings
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  device: "cuda"
  dtype: "bfloat16"

  # Graph generation
  k: 5                    # Number of top logits to trace
  tau: 0.005              # Node influence threshold (lower = more nodes)

  # Template
  use_chat_template: true # Wrap prompts in Llama 3.1 Instruct format

  # Labeling
  label_neurons: true     # Fetch neuron descriptions from database

  # Clustering
  skip_special_tokens: true
  min_cluster_size: 5
  max_cluster_depth: 1

  # LLM Analysis
  run_llm_analysis: true
  llm_model: "auto"       # auto, claude-sonnet-4-20250514, gpt-4o, etc.
  llm_provider: "auto"    # auto, anthropic, openai

  # Functional splitting (split large modules into sub-modules)
  functional_split: true
  functional_split_min_size: 10
  use_prompt_answer_split: true  # Split by prompt vs answer tokens first
  use_position_split: true       # Split by contiguous token position spans
  max_position_gap: 3            # Max gap between positions before splitting
  use_layer_split: true          # Split by layer ranges (early/mid/late)
  use_semantic_split: true       # Split by neuron label semantics (sentence-transformers)
  use_llm_split: false           # Use LLM to split by neuron function (expensive)

  # Output
  output_dir: "outputs/"
  save_intermediate: true

# Sequences to analyze (batch mode)
sequences:
  - prompt: "What is the capital of France?"

  - prompt: "The Eiffel Tower is located in"
    answer_prefix: " Paris"

  - prompt: "L-DOPA is used to treat a disease caused by deficiency of the neurotransmitter"

  - prompt: "Parkinson's disease involves degeneration of neurons in the"
'''
        if path:
            Path(path).write_text(example)
        return example

    def __str__(self) -> str:
        """Pretty print config."""
        lines = ["PipelineConfig:"]
        for f in fields(self):
            val = getattr(self, f.name)
            lines.append(f"  {f.name}: {val}")
        return "\n".join(lines)


# Minimal Llama 3.1 chat template
MINIMAL_CHAT_TEMPLATE = '''{%- if messages[0]['role'] == 'system' %}
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


def apply_chat_template(tokenizer, prompt: str, answer_prefix: str = "") -> str:
    """Apply minimal Llama 3.1 Instruct chat template to a prompt."""
    messages = [{"role": "user", "content": prompt}]

    if answer_prefix:
        messages.append({"role": "assistant", "content": answer_prefix})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            chat_template=MINIMAL_CHAT_TEMPLATE,
            continue_final_message=True
        )
    else:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=MINIMAL_CHAT_TEMPLATE
        )


def format_for_viewer(graph: dict, slug: str) -> dict:
    """Format the graph for the Anthropic attribution graph viewer."""
    graph['metadata']['slug'] = slug
    graph['metadata']['scan'] = 'llama-3.1-8b'

    for node in graph['nodes']:
        if node['layer'] == 'embedding':
            node['layer'] = 'E'
        elif node['layer'] == 'logit':
            max_neuron_layer = max(
                int(n['layer']) for n in graph['nodes']
                if n['layer'] not in ['embedding', 'logit', 'E'] and isinstance(n['layer'], (int, str)) and str(n['layer']).isdigit()
            )
            node['layer'] = str(max_neuron_layer + 1)
            node['isLogit'] = True
            if 'Logit:' in node.get('clerp', ''):
                node['clerp'] = node['clerp'].replace('Logit: ', '')
                node['feature_type'] = 'logit'
        else:
            node['layer'] = str(node['layer'])

        if 'isLogit' not in node:
            node['isLogit'] = False

    graph['qParams'] = {
        'pinnedIds': [],
        'supernodes': [],
        'linkType': 'both'
    }

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


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:50].strip('-')


def run_pipeline(
    prompt: str,
    config: PipelineConfig | None = None,
    model=None,
    tokenizer=None,
    verbose: bool = True
) -> dict[str, Any]:
    """Run the complete attribution analysis pipeline.

    Args:
        prompt: Input prompt to analyze
        config: Pipeline configuration (uses defaults if None)
        model: Pre-loaded model (will load if None)
        tokenizer: Pre-loaded tokenizer (will load if None)
        verbose: Print progress messages

    Returns:
        Dict containing:
        - graph: Raw attribution graph
        - clusters: Clustering results
        - analysis: Module analysis with LLM synthesis
        - output_path: Path to final output file (if saved)
    """
    config = config or PipelineConfig()

    def log(msg):
        if verbose:
            print(msg, flush=True)

    # Setup output directory
    output_dir = config.output_dir or DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = f"relp-{slugify(prompt)}"

    # Step 1: Load model if needed
    if model is None:
        log(f"Loading model: {config.model_name}")
        dtype = getattr(torch, config.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            device_map=config.device
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Step 2: Generate attribution graph
    log("\n[1/4] Generating attribution graph...")
    log(f"Prompt: {prompt}")

    if config.use_chat_template:
        formatted_prompt = apply_chat_template(tokenizer, prompt, config.answer_prefix)
    else:
        formatted_prompt = prompt

    relp_config = RelPConfig(k=config.k, tau=config.tau, target_tokens=config.target_tokens,
                             contrastive_tokens=config.contrastive_tokens, use_neuron_labels=False)
    attributor = RelPAttributor(model, tokenizer, config=relp_config)
    graph = attributor.compute_attributions(formatted_prompt, target_tokens=config.target_tokens,
                                           contrastive_tokens=config.contrastive_tokens)
    attributor.cleanup()

    # Store original prompt
    graph['metadata']['prompt'] = prompt

    # Format for viewer
    graph = format_for_viewer(graph, slug)

    log(f"Generated graph with {len(graph['nodes'])} nodes, {len(graph['links'])} edges")

    # Step 3: Label neurons from database
    if config.label_neurons:
        log("\n[2/4] Labeling neurons from database...")
        graph = label_graph(graph, verbose=verbose)
    else:
        log("\n[2/4] Skipping neuron labeling")

    # Save graph
    if config.save_intermediate:
        graph_path = output_dir / f"{slug}-graph.json"
        with open(graph_path, 'w') as f:
            json.dump(graph, f, indent=2)
        log(f"Saved graph to: {graph_path}")

    # Step 4: Cluster neurons
    log("\n[3/4] Clustering neurons...")
    clusters = cluster_graph(
        graph,
        skip_special_tokens=config.skip_special_tokens,
        min_cluster_size=config.min_cluster_size,
        max_depth=config.max_cluster_depth,
        verbose=verbose
    )

    log(f"Found {clusters['methods'][0]['n_clusters']} clusters")

    if config.save_intermediate:
        clusters_path = output_dir / f"{slug}-clusters.json"
        with open(clusters_path, 'w') as f:
            json.dump(clusters, f, indent=2)
        log(f"Saved clusters to: {clusters_path}")

    # Step 5: Analyze modules
    log("\n[4/4] Analyzing modules...")
    analysis = analyze_modules(
        clusters,
        graph=graph,
        run_llm=config.run_llm_analysis,
        model=config.llm_model,
        provider=config.llm_provider,
        functional_split=config.functional_split,
        functional_split_min_size=config.functional_split_min_size,
        use_prompt_answer_split=config.use_prompt_answer_split,
        use_position_split=config.use_position_split,
        max_position_gap=config.max_position_gap,
        use_layer_split=config.use_layer_split,
        use_semantic_split=config.use_semantic_split,
        use_llm_split=config.use_llm_split,
        use_llm_reassignment=config.use_llm_reassignment,
        verbose=verbose
    )

    # Save final output
    final_path = output_dir / f"{slug}-analysis.json"
    with open(final_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    log(f"\nSaved final analysis to: {final_path}")

    # Print summary
    if verbose:
        top_logits = analysis.get('top_logits', [])
        if top_logits:
            top = top_logits[0]
            log(f"\nTop prediction: '{top['token']}' ({top['probability']*100:.1f}%)")

        log(f"\n{'='*60}")
        log("Pipeline complete!")
        log(f"  Prompt: {prompt}")
        log(f"  Nodes: {len(graph['nodes'])}")
        log(f"  Modules: {analysis['n_modules']}")
        log(f"  Output: {final_path}")
        log(f"{'='*60}")

    return {
        'graph': graph,
        'clusters': clusters,
        'analysis': analysis,
        'output_path': final_path
    }


def run_pipeline_from_graph(
    graph_path: Path,
    config: PipelineConfig | None = None,
    verbose: bool = True
) -> dict[str, Any]:
    """Run pipeline starting from an existing graph file.

    Useful when you already have a graph and just want to re-run
    clustering and analysis.
    """
    config = config or PipelineConfig()

    def log(msg):
        if verbose:
            print(msg, flush=True)

    output_dir = config.output_dir or graph_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load graph
    log(f"Loading graph from: {graph_path}")
    with open(graph_path) as f:
        graph = json.load(f)

    slug = graph.get('metadata', {}).get('slug', graph_path.stem)

    # Label neurons if needed
    if config.label_neurons:
        log("\n[1/3] Labeling neurons from database...")
        graph = label_graph(graph, verbose=verbose)
    else:
        log("\n[1/3] Skipping neuron labeling")

    # Cluster
    log("\n[2/3] Clustering neurons...")
    clusters = cluster_graph(
        graph,
        skip_special_tokens=config.skip_special_tokens,
        min_cluster_size=config.min_cluster_size,
        max_depth=config.max_cluster_depth,
        verbose=verbose
    )

    if config.save_intermediate:
        clusters_path = output_dir / f"{slug}-clusters.json"
        with open(clusters_path, 'w') as f:
            json.dump(clusters, f, indent=2)
        log(f"Saved clusters to: {clusters_path}")

    # Analyze
    log("\n[3/3] Analyzing modules...")
    analysis = analyze_modules(
        clusters,
        graph=graph,
        run_llm=config.run_llm_analysis,
        model=config.llm_model,
        provider=config.llm_provider,
        functional_split=config.functional_split,
        functional_split_min_size=config.functional_split_min_size,
        use_prompt_answer_split=config.use_prompt_answer_split,
        use_position_split=config.use_position_split,
        max_position_gap=config.max_position_gap,
        use_layer_split=config.use_layer_split,
        use_semantic_split=config.use_semantic_split,
        use_llm_split=config.use_llm_split,
        use_llm_reassignment=config.use_llm_reassignment,
        verbose=verbose
    )

    final_path = output_dir / f"{slug}-analysis.json"
    with open(final_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    log(f"\nSaved final analysis to: {final_path}")

    return {
        'graph': graph,
        'clusters': clusters,
        'analysis': analysis,
        'output_path': final_path
    }


def _run_graph_stages(
    prompt: str,
    config: PipelineConfig,
    model,
    tokenizer,
    verbose: bool = True
) -> dict[str, Any]:
    """Run graph generation, labeling, and clustering stages (no LLM analysis).

    This is the first phase of the two-phase batch pipeline, running the
    GPU-bound stages that must be serial.

    Returns:
        Dict with graph, clusters, slug, and output_dir for later LLM analysis
    """
    def log(msg):
        if verbose:
            print(msg, flush=True)

    # Setup output directory
    output_dir = config.output_dir or DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = f"relp-{slugify(prompt)}"

    # Step 1: Generate attribution graph
    log("\n[1/3] Generating attribution graph...")
    log(f"Prompt: {prompt}")

    if config.use_chat_template:
        formatted_prompt = apply_chat_template(tokenizer, prompt, config.answer_prefix)
    else:
        formatted_prompt = prompt

    relp_config = RelPConfig(k=config.k, tau=config.tau, target_tokens=config.target_tokens,
                             contrastive_tokens=config.contrastive_tokens, use_neuron_labels=False)
    attributor = RelPAttributor(model, tokenizer, config=relp_config)
    graph = attributor.compute_attributions(formatted_prompt, target_tokens=config.target_tokens,
                                           contrastive_tokens=config.contrastive_tokens)
    attributor.cleanup()

    # Store original prompt
    graph['metadata']['prompt'] = prompt

    # Format for viewer
    graph = format_for_viewer(graph, slug)

    log(f"Generated graph with {len(graph['nodes'])} nodes, {len(graph['links'])} edges")

    # Step 2: Label neurons from database
    if config.label_neurons:
        log("\n[2/3] Labeling neurons from database...")
        graph = label_graph(graph, verbose=verbose)
    else:
        log("\n[2/3] Skipping neuron labeling")

    # Save graph
    if config.save_intermediate:
        graph_path = output_dir / f"{slug}-graph.json"
        with open(graph_path, 'w') as f:
            json.dump(graph, f, indent=2)
        log(f"Saved graph to: {graph_path}")

    # Step 3: Cluster neurons
    log("\n[3/3] Clustering neurons...")
    clusters = cluster_graph(
        graph,
        skip_special_tokens=config.skip_special_tokens,
        min_cluster_size=config.min_cluster_size,
        max_depth=config.max_cluster_depth,
        verbose=verbose
    )

    log(f"Found {clusters['methods'][0]['n_clusters']} clusters")

    if config.save_intermediate:
        clusters_path = output_dir / f"{slug}-clusters.json"
        with open(clusters_path, 'w') as f:
            json.dump(clusters, f, indent=2)
        log(f"Saved clusters to: {clusters_path}")

    return {
        'prompt': prompt,
        'graph': graph,
        'clusters': clusters,
        'slug': slug,
        'output_dir': output_dir
    }


async def _run_llm_analysis_async(
    stage_result: dict[str, Any],
    config: PipelineConfig,
    rate_limiter,
    verbose: bool = True
) -> dict[str, Any]:
    """Run LLM analysis stage asynchronously.

    This is the second phase of the two-phase batch pipeline, running
    the LLM-bound analysis that can be parallelized.
    """
    from .async_analysis import analyze_modules_async

    def log(msg):
        if verbose:
            print(msg, flush=True)

    graph = stage_result['graph']
    clusters = stage_result['clusters']
    slug = stage_result['slug']
    output_dir = stage_result['output_dir']
    prompt = stage_result['prompt']

    log(f"\n[LLM] Analyzing: {prompt[:50]}...")

    analysis = await analyze_modules_async(
        clusters,
        graph=graph,
        model=config.llm_model,
        provider=config.llm_provider,
        rate_limiter=rate_limiter,
        functional_split=config.functional_split,
        functional_split_min_size=config.functional_split_min_size,
        use_prompt_answer_split=config.use_prompt_answer_split,
        use_position_split=config.use_position_split,
        max_position_gap=config.max_position_gap,
        use_layer_split=config.use_layer_split,
        use_semantic_split=config.use_semantic_split,
        use_llm_split=config.use_llm_split,
        use_llm_reassignment=config.use_llm_reassignment,
        verbose=verbose
    )

    # Save final output
    final_path = output_dir / f"{slug}-analysis.json"
    with open(final_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    log(f"Saved analysis to: {final_path}")

    return {
        'graph': graph,
        'clusters': clusters,
        'analysis': analysis,
        'output_path': final_path
    }


def run_batch(
    sequences: list[dict[str, str]],
    config: PipelineConfig | None = None,
    verbose: bool = True,
    parallel_llm: bool = False
) -> list[dict[str, Any]]:
    """Run pipeline on multiple sequences, reusing the loaded model.

    Args:
        sequences: List of dicts with 'prompt' and optional 'answer_prefix'
        config: Pipeline configuration (shared across all sequences)
        verbose: Print progress
        parallel_llm: If True, run graph generation serially then LLM analysis
                     in parallel. More efficient for large batches.

    Returns:
        List of results, one per sequence

    Example:
        # Serial processing (default)
        results = run_batch([
            {"prompt": "What is the capital of France?"},
            {"prompt": "The Eiffel Tower is in", "answer_prefix": " Paris"},
        ])

        # Parallel LLM analysis
        results = run_batch([...], parallel_llm=True)
    """
    config = config or PipelineConfig()

    def log(msg):
        if verbose:
            print(msg, flush=True)

    log(f"Running batch of {len(sequences)} sequences")
    if parallel_llm:
        log("Mode: parallel LLM analysis (graphs serial, LLM parallel)")
    log(f"Loading model: {config.model_name}")

    # Load model once
    dtype = getattr(torch, config.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=config.device
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if parallel_llm and config.run_llm_analysis:
        # Two-phase processing: serial graph gen, parallel LLM analysis
        return _run_batch_parallel_llm(sequences, config, model, tokenizer, verbose)
    else:
        # Original serial processing
        return _run_batch_serial(sequences, config, model, tokenizer, verbose)


def _run_batch_serial(
    sequences: list[dict[str, str]],
    config: PipelineConfig,
    model,
    tokenizer,
    verbose: bool
) -> list[dict[str, Any]]:
    """Run batch serially (original behavior)."""
    def log(msg):
        if verbose:
            print(msg, flush=True)

    results = []
    for i, seq in enumerate(sequences):
        prompt = seq['prompt']
        answer_prefix = seq.get('answer_prefix', config.answer_prefix)

        log(f"\n{'='*60}")
        log(f"[{i+1}/{len(sequences)}] {prompt[:50]}...")
        log(f"{'='*60}")

        # Create per-sequence config with answer_prefix override
        seq_config = PipelineConfig(**{
            **config.to_dict(),
            'answer_prefix': answer_prefix
        })

        result = run_pipeline(
            prompt,
            config=seq_config,
            model=model,
            tokenizer=tokenizer,
            verbose=verbose
        )
        results.append(result)

    log(f"\n{'='*60}")
    log(f"Batch complete: {len(results)} sequences processed")
    log(f"{'='*60}")

    return results


def _run_batch_parallel_llm(
    sequences: list[dict[str, str]],
    config: PipelineConfig,
    model,
    tokenizer,
    verbose: bool
) -> list[dict[str, Any]]:
    """Run batch with parallel LLM analysis.

    Phase 1: Serial graph generation (GPU-bound)
    Phase 2: Parallel LLM analysis (API-bound)
    """
    from .async_analysis import RateLimiter

    def log(msg):
        if verbose:
            print(msg, flush=True)

    # Phase 1: Generate all graphs serially
    log(f"\n{'='*60}")
    log(f"Phase 1: Graph Generation ({len(sequences)} sequences)")
    log(f"{'='*60}")

    stage_results = []
    for i, seq in enumerate(sequences):
        prompt = seq['prompt']
        answer_prefix = seq.get('answer_prefix', config.answer_prefix)

        log(f"\n[{i+1}/{len(sequences)}] {prompt[:50]}...")

        # Create per-sequence config with answer_prefix override
        seq_config = PipelineConfig(**{
            **config.to_dict(),
            'answer_prefix': answer_prefix
        })

        stage_result = _run_graph_stages(
            prompt,
            config=seq_config,
            model=model,
            tokenizer=tokenizer,
            verbose=verbose
        )
        stage_results.append((stage_result, seq_config))

    # Phase 2: Run LLM analysis in parallel
    log(f"\n{'='*60}")
    log(f"Phase 2: Parallel LLM Analysis ({len(stage_results)} graphs)")
    log(f"{'='*60}")

    # Create rate limiter based on provider
    provider = config.llm_provider
    if provider == "auto":
        provider = "openai" if "gpt" in config.llm_model else "anthropic"

    if provider == "anthropic":
        rate_limiter = RateLimiter(requests_per_minute=50, burst_size=10)
    else:
        rate_limiter = RateLimiter(requests_per_minute=500, burst_size=50)

    async def run_all_analyses():
        tasks = [
            _run_llm_analysis_async(stage_result, seq_config, rate_limiter, verbose=verbose)
            for stage_result, seq_config in stage_results
        ]
        return await asyncio.gather(*tasks)

    # Run async analyses
    results = asyncio.run(run_all_analyses())

    log(f"\n{'='*60}")
    log(f"Batch complete: {len(results)} sequences processed")
    log(f"{'='*60}")

    return results


def run_from_config(
    config_path: str | Path,
    verbose: bool = True
) -> list[dict[str, Any]]:
    """Run pipeline from a YAML config file.

    Args:
        config_path: Path to YAML config file
        verbose: Print progress

    Returns:
        List of results (one per sequence if batch, single-item list otherwise)

    Config file format:
        config:
          model_name: "meta-llama/Llama-3.1-8B-Instruct"
          functional_split: true
          ...

        sequences:
          - prompt: "What is the capital of France?"
          - prompt: "The Eiffel Tower is in"
            answer_prefix: " Paris"
    """
    config_path = Path(config_path)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    log(f"Loading config from: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Parse config
    config = PipelineConfig.from_dict(data.get('config', {}))

    # Get sequences
    sequences = data.get('sequences', [])

    if not sequences:
        raise ValueError("Config file must contain 'sequences' list")

    log(f"Config: {config.model_name}, functional_split={config.functional_split}")
    log(f"Sequences: {len(sequences)}")
    if config.parallel_llm:
        log("Parallel LLM: enabled")

    return run_batch(sequences, config=config, verbose=verbose, parallel_llm=config.parallel_llm)
