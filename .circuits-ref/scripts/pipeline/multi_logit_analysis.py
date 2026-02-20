#!/usr/bin/env python3
"""Multi-logit RelP analysis pipeline.

Runs RelP attribution for each top logit separately, clusters each graph,
then produces a comparative analysis showing which neurons are shared vs
logit-specific.

Usage:
    python scripts/multi_logit_analysis.py "Parkinson's disease involves degeneration of neurons in the"
    python scripts/multi_logit_analysis.py "prompt" --num-logits 5 --output outputs/multi_logit/
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from circuits.clustering import cluster_graph
from circuits.labeling import is_database_available, label_graphs
from circuits.relp import RelPAttributor, RelPConfig

# Chat template for Llama 3.1 Instruct
MINIMAL_CHAT_TEMPLATE = '''{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set system_message = '' %}
    {%- set loop_messages = messages %}
{%- endif %}
<|start_header_id|>system<|end_header_id|}

{{ system_message }}<|eot_id|>
{%- for message in loop_messages %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] }}
{%- if not loop.last or add_generation_prompt %}<|eot_id|>{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}'''


@dataclass
class NeuronInfo:
    """Information about a neuron across multiple logit graphs."""
    layer: int
    neuron_idx: int
    position: int
    label: str = ""

    # Per-logit information
    logit_scores: dict[str, float] = field(default_factory=dict)  # logit_token -> relp_score
    logit_activations: dict[str, float] = field(default_factory=dict)
    module_ids: dict[str, int] = field(default_factory=dict)  # logit_token -> module_id

    @property
    def node_id(self) -> str:
        return f"{self.layer}_{self.neuron_idx}_{self.position}"

    @property
    def appears_in_logits(self) -> list[str]:
        return list(self.logit_scores.keys())

    @property
    def num_logits(self) -> int:
        return len(self.logit_scores)

    @property
    def is_shared(self) -> bool:
        """Returns True if neuron appears in multiple logit graphs."""
        return self.num_logits > 1

    @property
    def avg_score(self) -> float:
        if not self.logit_scores:
            return 0.0
        return sum(self.logit_scores.values()) / len(self.logit_scores)

    @property
    def max_score(self) -> float:
        if not self.logit_scores:
            return 0.0
        return max(abs(s) for s in self.logit_scores.values())


@dataclass
class ModuleComparison:
    """Comparison of a module across different logit graphs."""
    module_id: int
    logit_token: str
    neurons: list[NeuronInfo]
    edges_to_logit: list[dict]

    @property
    def size(self) -> int:
        return len(self.neurons)

    @property
    def shared_neurons(self) -> list[NeuronInfo]:
        return [n for n in self.neurons if n.is_shared]

    @property
    def unique_neurons(self) -> list[NeuronInfo]:
        return [n for n in self.neurons if not n.is_shared]


class SingleLogitAttributor(RelPAttributor):
    """RelP attributor that backprops from a single logit instead of top-k sum."""

    def compute_single_logit_attribution(
        self,
        input_text: str,
        target_logit_idx: int,
        tau: float | None = None,
        filter_always_on: bool | None = None
    ) -> dict[str, Any]:
        """Compute RelP attribution for a single target logit.

        Args:
            input_text: Input prompt
            target_logit_idx: Index of the logit to attribute to
            tau: Node filtering threshold
            filter_always_on: Whether to filter always-on neurons

        Returns:
            Attribution graph for this single logit
        """
        tau = tau if tau is not None else self.config.tau
        filter_always_on = filter_always_on if filter_always_on is not None else self.config.filter_always_on

        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        seq_len = inputs.input_ids.shape[1]

        # Clear caches
        self.scope.clear_relp_caches()
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]

        # Get the single target logit value
        target_logit = logits[0, target_logit_idx]
        target_prob = torch.softmax(logits, dim=-1)[0, target_logit_idx].item()
        target_token = self.tokenizer.decode([target_logit_idx])

        # Backward from single logit
        target_logit.backward()

        # Compute node attributions
        node_attributions = self._compute_node_attributions(seq_len)

        # Filter nodes
        threshold = tau * abs(target_logit.item())
        filtered_nodes = self._filter_nodes(node_attributions, threshold, filter_always_on)

        # Compute edges
        edge_attributions = []
        if self.config.compute_edges and len(filtered_nodes) > 0:
            if self.config.use_jacobian_edges:
                edge_attributions = self._compute_jacobian_edges(
                    filtered_nodes, tokens, inputs, seq_len
                )
            else:
                edge_attributions = self._compute_edge_attributions(
                    filtered_nodes, tokens, inputs
                )

        # Build graph with single logit
        graph = self._build_single_logit_graph(
            input_text=input_text,
            tokens=tokens,
            nodes=filtered_nodes,
            edges=edge_attributions,
            target_logit_idx=target_logit_idx,
            target_token=target_token,
            target_prob=target_prob,
            target_logit_value=target_logit.item(),
            tau=tau
        )

        return graph

    def _build_single_logit_graph(
        self,
        input_text: str,
        tokens: list[str],
        nodes,
        edges,
        target_logit_idx: int,
        target_token: str,
        target_prob: float,
        target_logit_value: float,
        tau: float
    ) -> dict[str, Any]:
        """Build graph for single logit attribution."""

        graph_nodes = []

        # Embedding nodes
        for pos, token in enumerate(tokens):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            node_id = f"E_{token_id}_{pos}"
            graph_nodes.append({
                "node_id": node_id,
                "feature": token_id,
                "layer": "embedding",
                "ctx_idx": pos,
                "feature_type": "embedding",
                "jsNodeId": node_id,
                "clerp": f"Token: {token}",
                "influence": None,
                "activation": 1.0
            })

        # Neuron nodes (labels added later via label_graphs())
        for node in nodes:
            node_dict = node.to_neuronpedia_node()
            graph_nodes.append(node_dict)

        # Single logit node
        logit_node_id = f"L_{target_logit_idx}_{len(tokens)-1}"
        graph_nodes.append({
            "node_id": logit_node_id,
            "feature": target_logit_idx,
            "layer": "logit",
            "ctx_idx": len(tokens) - 1,
            "feature_type": "logit",
            "jsNodeId": logit_node_id,
            "clerp": f"{target_token} (p={target_prob:.4f})",
            "influence": None,
            "activation": None,
            "isLogit": True
        })

        # Build links
        graph_links = []
        for src, tgt, weight in edges:
            graph_links.append({"source": src, "target": tgt, "weight": weight})

        # Add edges from neurons to the single logit
        for node in nodes:
            edge_weight = node.relp_score
            if abs(edge_weight) > 1e-6:
                graph_links.append({
                    "source": node.node_id,
                    "target": logit_node_id,
                    "weight": edge_weight
                })

        metadata = {
            "slug": f"relp-single-logit-{target_logit_idx}",
            "scan": "llama-3.1-8b-instruct",
            "prompt_tokens": tokens,
            "prompt": input_text,
            "node_threshold": tau,
            "target_logit": {
                "index": target_logit_idx,
                "token": target_token,
                "prob": target_prob,
                "logit_value": target_logit_value
            }
        }

        return {
            "metadata": metadata,
            "qParams": {},
            "nodes": graph_nodes,
            "links": graph_links
        }


def run_multi_logit_relp(
    prompt: str,
    num_logits: int = 5,
    model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
    answer_prefix: str = "",
    tau: float = 0.005,
    use_chat_template: bool = True
) -> tuple[list[dict], dict[str, Any]]:
    """Run RelP for each top logit and return all graphs.

    Returns:
        Tuple of (list of graphs, metadata with top logits info)
    """
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = MINIMAL_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Format prompt
    if use_chat_template:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if answer_prefix:
            formatted_prompt += answer_prefix
    else:
        formatted_prompt = prompt

    print(f"Prompt: {prompt[:80]}...")
    print(f"Answer prefix: {repr(answer_prefix)}")

    # First, get the top logits
    print("\nGetting top logits...", flush=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, num_logits)

    top_logits_info = []
    for i in range(num_logits):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        token = tokenizer.decode([idx])
        top_logits_info.append({
            "index": idx,
            "token": token,
            "prob": prob
        })
        print(f"  {i+1}. {repr(token)}: p={prob:.4f}")

    # Create attributor
    config = RelPConfig(
        k=1,  # Single logit
        tau=tau,
        compute_edges=True,
        use_jacobian_edges=True,
        linearize=True
    )

    attributor = SingleLogitAttributor(model, tokenizer, "cuda", config)

    # Run RelP for each top logit
    graphs = []
    for i, logit_info in enumerate(top_logits_info):
        print(f"\n[{i+1}/{num_logits}] Computing RelP for {repr(logit_info['token'])}...", flush=True)

        # Clear caches between runs
        attributor.scope.clear_relp_caches()
        model.zero_grad()

        graph = attributor.compute_single_logit_attribution(
            formatted_prompt,
            target_logit_idx=logit_info["index"],
            tau=tau
        )

        # Add logit info to metadata
        graph["metadata"]["target_logit"] = logit_info
        graphs.append(graph)

        print(f"      {len([n for n in graph['nodes'] if n.get('feature_type') == 'mlp_neuron'])} neurons, {len(graph['links'])} edges")

    attributor.cleanup()

    metadata = {
        "prompt": prompt,
        "formatted_prompt": formatted_prompt,
        "answer_prefix": answer_prefix,
        "num_logits": num_logits,
        "top_logits": top_logits_info
    }

    return graphs, metadata


def cluster_all_graphs(graphs: list[dict]) -> list[dict]:
    """Cluster each graph using Infomap."""
    clustered = []

    for i, graph in enumerate(graphs):
        logit_token = graph["metadata"]["target_logit"]["token"]
        print(f"Clustering graph for {repr(logit_token)}...", flush=True)

        result = cluster_graph(graph, min_cluster_size=2, verbose=False)

        # Clusters are nested inside methods[0]['clusters']
        clusters = []
        if result.get("methods"):
            clusters = result["methods"][0].get("clusters", [])

        # Attach cluster info to graph
        graph["clusters"] = clusters
        clustered.append(graph)

        print(f"  {len(clusters)} modules")

    return clustered


def build_comparative_analysis(
    graphs: list[dict],
    metadata: dict
) -> dict[str, Any]:
    """Build comparative analysis across all logit graphs.

    Returns a structured comparison showing:
    - Neurons that appear in multiple graphs (shared)
    - Neurons unique to each logit
    - Module structure comparison
    """

    # Build neuron registry across all graphs
    neuron_registry: dict[str, NeuronInfo] = {}  # node_id -> NeuronInfo

    for graph in graphs:
        logit_token = graph["metadata"]["target_logit"]["token"]
        clusters = graph.get("clusters", [])

        # Build cluster lookup
        node_to_cluster = {}
        for cluster in clusters:
            cluster_id = cluster["cluster_id"]
            for member in cluster["members"]:
                node_id = f"{member['layer']}_{member['neuron']}_{member['position']}"
                node_to_cluster[node_id] = cluster_id

        # Process nodes
        for node in graph["nodes"]:
            if node.get("feature_type") != "mlp_neuron":
                continue

            node_id = node["node_id"]
            layer = node["layer"]
            neuron_idx = node["feature"]
            position = node["ctx_idx"]

            if node_id not in neuron_registry:
                neuron_registry[node_id] = NeuronInfo(
                    layer=layer,
                    neuron_idx=neuron_idx,
                    position=position,
                    label=node.get("clerp", "")
                )

            info = neuron_registry[node_id]
            info.logit_scores[logit_token] = node.get("influence", 0)
            info.logit_activations[logit_token] = node.get("activation", 0)
            if node_id in node_to_cluster:
                info.module_ids[logit_token] = node_to_cluster[node_id]

    # Categorize neurons
    shared_neurons = [n for n in neuron_registry.values() if n.is_shared]

    logit_specific: dict[str, list[NeuronInfo]] = defaultdict(list)
    for n in neuron_registry.values():
        if n.num_logits == 1:
            logit = n.appears_in_logits[0]
            logit_specific[logit].append(n)

    # Sort by importance
    shared_neurons.sort(key=lambda n: n.max_score, reverse=True)
    for logit in logit_specific:
        logit_specific[logit].sort(key=lambda n: abs(n.logit_scores.get(logit, 0)), reverse=True)

    # Build module comparison
    module_comparison = {}
    for graph in graphs:
        logit_token = graph["metadata"]["target_logit"]["token"]
        clusters = graph.get("clusters", [])

        module_comparison[logit_token] = {
            "num_modules": len(clusters),
            "modules": []
        }

        for cluster in clusters:
            cluster_neurons = []
            for member in cluster["members"]:
                node_id = f"{member['layer']}_{member['neuron']}_{member['position']}"
                if node_id in neuron_registry:
                    cluster_neurons.append(neuron_registry[node_id])

            shared_in_cluster = [n for n in cluster_neurons if n.is_shared]
            unique_in_cluster = [n for n in cluster_neurons if not n.is_shared]

            module_comparison[logit_token]["modules"].append({
                "module_id": cluster["cluster_id"],
                "size": len(cluster_neurons),
                "num_shared": len(shared_in_cluster),
                "num_unique": len(unique_in_cluster),
                "top_neurons": [
                    {
                        "node_id": n.node_id,
                        "label": n.label,
                        "score": n.logit_scores.get(logit_token, 0),
                        "appears_in": n.appears_in_logits
                    }
                    for n in sorted(cluster_neurons, key=lambda x: abs(x.logit_scores.get(logit_token, 0)), reverse=True)[:5]
                ]
            })

    return {
        "metadata": metadata,
        "summary": {
            "total_neurons": len(neuron_registry),
            "shared_neurons": len(shared_neurons),
            "logit_specific_counts": {k: len(v) for k, v in logit_specific.items()}
        },
        "shared_neurons": [
            {
                "node_id": n.node_id,
                "layer": n.layer,
                "neuron": n.neuron_idx,
                "position": n.position,
                "label": n.label,
                "appears_in": n.appears_in_logits,
                "scores": n.logit_scores,
                "modules": n.module_ids
            }
            for n in shared_neurons[:50]  # Top 50 shared
        ],
        "logit_specific_neurons": {
            logit: [
                {
                    "node_id": n.node_id,
                    "layer": n.layer,
                    "neuron": n.neuron_idx,
                    "position": n.position,
                    "label": n.label,
                    "score": n.logit_scores.get(logit, 0)
                }
                for n in neurons[:20]  # Top 20 per logit
            ]
            for logit, neurons in logit_specific.items()
        },
        "module_comparison": module_comparison
    }


MULTI_LOGIT_ANALYSIS_SYSTEM = """You are an expert in neural network interpretability, analyzing multi-logit attribution data from a language model (Llama-3.1-8B-Instruct).

You are given comparative data from running RelP (Relevance Propagation) attribution separately for each top predicted token. This produces a separate attribution graph for each logit, showing which neurons contribute to that specific completion.

The key insight is that by comparing graphs across logits, we can identify:
1. **Shared neurons**: Contribute to multiple completions (general machinery)
2. **Logit-specific neurons**: Only activate for one completion (decision-specific)
3. **Discriminator neurons**: Promote one completion while suppressing others (opposite signs)

Neuron labels come from automated interpretability and describe what activates the neuron.
Edge weights show how much a neuron contributes to the final logit (positive = promotes, negative = suppresses).

Your task is to synthesize this into a comprehensive circuit analysis with:
- Detailed module descriptions (what each module computes)
- A circuit narrative explaining the full computation
- Specific, actionable hypotheses about neuron functions"""


MULTI_LOGIT_ANALYSIS_PROMPT = """Analyze this multi-logit attribution data and produce a detailed circuit analysis.

{llm_prompt_text}

# Your Analysis

Provide a comprehensive analysis with the following sections:

## 1. Circuit Overview
Summarize what this circuit computes. What is the model deciding between? What information flow leads to the final probability distribution?

## 2. Module Descriptions
For each major module (across all logits), provide:
- **Module ID and logit context**
- **Functional description**: What is this module computing?
- **Key neurons**: Which neurons are most important and why?
- **Evidence**: What neuron labels and edge weights support this interpretation?

Focus on modules that appear important for the decision (strong edges to logits).

## 3. Key Discriminator Neurons
Identify neurons that distinguish between completions:
- Neurons with **opposite signs** across logits (promotes one, suppresses another)
- Neurons **unique to one logit** with strong influence
- For each, explain what it might be detecting

## 4. Shared Machinery Analysis
What do the shared neurons represent?
- Are they "general sentence completion" neurons?
- Domain-specific (medical, scientific)?
- Grammatical/syntactic?

## 5. Circuit Narrative
Tell the story of how this circuit produces its output. Walk through:
- What happens at each position in the prompt
- How information accumulates across layers
- How the final decision is made

## 6. Causal Predictions
Make specific, testable predictions:
- If we ablate module X, what should happen to the probability distribution?
- Which neurons would flip the decision if amplified?
- What interventions would shift probability from completion A to B?

Be specific and cite neuron IDs and edge weights to support your analysis."""


def call_llm_for_analysis(
    prompt_text: str,
    provider: str = "openai",
    model: str = "gpt-5.2",
    max_tokens: int = 16384
) -> str:
    """Call LLM to analyze the multi-logit data."""
    import os

    from dotenv import load_dotenv
    load_dotenv()

    user_prompt = MULTI_LOGIT_ANALYSIS_PROMPT.format(llm_prompt_text=prompt_text)

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Handle newer models that use max_completion_tokens
        use_new_api = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")

        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": MULTI_LOGIT_ANALYSIS_SYSTEM},
                {"role": "user", "content": user_prompt}
            ]
        }
        if use_new_api:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=MULTI_LOGIT_ANALYSIS_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text

    else:
        raise ValueError(f"Unknown provider: {provider}")


def format_for_llm(analysis: dict, graphs: list[dict]) -> str:
    """Format the comparative analysis for LLM consumption.

    Includes full module details with neuron labels and edges to logits.
    """
    lines = []

    metadata = analysis["metadata"]
    summary = analysis["summary"]

    lines.append("# Multi-Logit Circuit Analysis")
    lines.append("")
    lines.append(f"**Prompt:** {metadata['prompt']}")
    if metadata.get("answer_prefix"):
        lines.append(f"**Answer prefix:** {repr(metadata['answer_prefix'])}")
    lines.append("")

    lines.append("## Top Logits Analyzed")
    lines.append("RelP attribution was run separately for each top logit, producing separate graphs.")
    lines.append("This reveals which neurons are shared across completions vs specific to each.")
    lines.append("")
    for i, logit in enumerate(metadata["top_logits"]):
        lines.append(f"{i+1}. {repr(logit['token'])}: p={logit['prob']:.4f}")
    lines.append("")

    lines.append("## Summary Statistics")
    lines.append(f"- Total unique neurons across all graphs: {summary['total_neurons']}")
    lines.append(f"- Neurons appearing in 2+ graphs (shared): {summary['shared_neurons']}")
    for logit, count in summary["logit_specific_counts"].items():
        lines.append(f"- Neurons unique to {repr(logit)}: {count}")
    lines.append("")

    # === Detailed module structure for each logit ===
    lines.append("=" * 60)
    lines.append("# DETAILED MODULE ANALYSIS PER LOGIT")
    lines.append("=" * 60)
    lines.append("")

    for graph in graphs:
        logit_info = graph["metadata"]["target_logit"]
        logit_token = logit_info["token"]
        logit_prob = logit_info["prob"]
        clusters = graph.get("clusters", [])

        # Build node lookup for labels
        node_lookup = {n["node_id"]: n for n in graph["nodes"]}

        # Build edge lookup (edges TO the target logit)
        edges_to_logit = []
        logit_node_id = None
        for node in graph["nodes"]:
            if node.get("isLogit") or node.get("feature_type") == "logit":
                logit_node_id = node["node_id"]
                break

        if logit_node_id:
            for link in graph.get("links", []):
                if link["target"] == logit_node_id:
                    edges_to_logit.append(link)

        # Sort edges by absolute weight
        edges_to_logit.sort(key=lambda e: abs(e["weight"]), reverse=True)

        lines.append(f"## Logit: {repr(logit_token)} (p={logit_prob:.4f})")
        lines.append(f"**{len(clusters)} modules detected**")
        lines.append("")

        # Top edges to this logit
        lines.append("### Top edges to logit (strongest influences):")
        for edge in edges_to_logit[:15]:
            src_node = node_lookup.get(edge["source"], {})
            label = src_node.get("clerp", "unknown")
            weight = edge["weight"]
            sign = "+" if weight > 0 else ""
            lines.append(f"- {edge['source']} → {repr(logit_token)}: **{sign}{weight:.3f}**")
            lines.append(f"  - {label}")
        lines.append("")

        # Detailed modules
        lines.append("### Modules:")
        lines.append("")

        for cluster in clusters[:10]:  # Top 10 modules
            cluster_id = cluster["cluster_id"]
            members = cluster["members"]

            # Count shared vs unique
            shared_count = 0
            unique_count = 0
            for m in members:
                node_id = f"{m['layer']}_{m['neuron']}_{m['position']}"
                # Check if in analysis shared neurons
                is_shared = any(
                    n["node_id"] == node_id
                    for n in analysis["shared_neurons"]
                )
                if is_shared:
                    shared_count += 1
                else:
                    unique_count += 1

            lines.append(f"#### Module {cluster_id} ({len(members)} neurons, {shared_count} shared, {unique_count} unique)")
            lines.append("")
            lines.append("**Neurons:**")

            for m in members[:8]:  # Top 8 neurons per module
                node_id = f"{m['layer']}_{m['neuron']}_{m['position']}"
                graph_node = node_lookup.get(node_id, {})
                label = graph_node.get("clerp", f"L{m['layer']}/N{m['neuron']}")
                if len(label) > 100:
                    label = label[:97] + "..."
                influence = graph_node.get("influence", m.get("influence", 0))
                activation = graph_node.get("activation", m.get("activation", 0))
                lines.append(f"- L{m['layer']}/N{m['neuron']} @ pos {m['position']}")
                lines.append(f"  - {label}")
                if influence and activation:
                    lines.append(f"  - Influence: {influence:.3f}, Activation: {activation:.1f}")

            if len(members) > 8:
                lines.append(f"- ... and {len(members) - 8} more neurons")

            # Find edges from this module to logit
            module_edges = []
            module_node_ids = {f"{m['layer']}_{m['neuron']}_{m['position']}" for m in members}
            for edge in edges_to_logit:
                if edge["source"] in module_node_ids:
                    module_edges.append(edge)

            if module_edges:
                lines.append("")
                lines.append("**Edges to logit from this module:**")
                for edge in module_edges[:5]:
                    src_node = node_lookup.get(edge["source"], {})
                    weight = edge["weight"]
                    sign = "+" if weight > 0 else ""
                    lines.append(f"- {edge['source']} → {repr(logit_token)}: {sign}{weight:.3f}")

            lines.append("")

        lines.append("-" * 40)
        lines.append("")

    # === Contrastive analysis ===
    lines.append("=" * 60)
    lines.append("# CONTRASTIVE ANALYSIS")
    lines.append("=" * 60)
    lines.append("")

    lines.append("## Key Shared Neurons (appear in multiple logit graphs)")
    lines.append("These neurons contribute to multiple possible completions:")
    lines.append("")
    for n in analysis["shared_neurons"][:25]:
        appears = ", ".join(repr(l) for l in n["appears_in"])
        scores_str = ", ".join(f"{repr(l)}:{s:.3f}" for l, s in n["scores"].items())
        lines.append(f"### L{n['layer']}/N{n['neuron']} @ pos {n['position']}")
        lines.append(f"- **Label:** {n['label']}")
        lines.append(f"- **Appears in:** {appears}")
        lines.append(f"- **Scores by logit:** {scores_str}")
        if n.get("modules"):
            modules_str = ", ".join(f"{repr(l)}:mod{m}" for l, m in n["modules"].items())
            lines.append(f"- **Module assignments:** {modules_str}")
        lines.append("")

    lines.append("## Logit-Specific Neurons")
    lines.append("These neurons ONLY appear when attributing to a specific logit:")
    lines.append("")
    for logit, neurons in analysis["logit_specific_neurons"].items():
        lines.append(f"### Unique to {repr(logit)}")
        for n in neurons[:8]:
            lines.append(f"- **L{n['layer']}/N{n['neuron']}** @ pos {n['position']}: score={n['score']:.3f}")
            lines.append(f"  - {n['label']}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-logit RelP analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("prompt", help="Input prompt to analyze")
    parser.add_argument("--num-logits", "-n", type=int, default=5,
                        help="Number of top logits to analyze")
    parser.add_argument("--output", "-o", default="outputs/multi_logit",
                        help="Output directory")
    parser.add_argument("--answer-prefix", default="",
                        help="Answer prefix (e.g., ' Answer:')")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Node filtering threshold")
    parser.add_argument("--raw", action="store_true",
                        help="Don't apply chat template")
    parser.add_argument("--llm-provider", default="openai",
                        choices=["openai", "anthropic"],
                        help="LLM provider for analysis")
    parser.add_argument("--llm-model", default="gpt-5.2",
                        help="LLM model for analysis")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM analysis step")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run multi-logit RelP
    graphs, metadata = run_multi_logit_relp(
        args.prompt,
        num_logits=args.num_logits,
        answer_prefix=args.answer_prefix,
        tau=args.tau,
        use_chat_template=not args.raw
    )

    # Cluster all graphs
    print("\nClustering graphs...", flush=True)
    graphs = cluster_all_graphs(graphs)

    # Add neuron labels from database
    if is_database_available():
        print("\nLabeling neurons from database...", flush=True)
        graphs = label_graphs(graphs, verbose=True)
    else:
        print("\nWarning: Neuron database not available, skipping labeling")

    # Save individual graphs
    for i, graph in enumerate(graphs):
        logit_token = graph["metadata"]["target_logit"]["token"]
        safe_token = "".join(c if c.isalnum() else "_" for c in logit_token)
        graph_file = output_dir / f"graph_logit_{i}_{safe_token}.json"
        with open(graph_file, "w") as f:
            json.dump(graph, f, indent=2)
        print(f"Saved: {graph_file}")

    # Build comparative analysis
    print("\nBuilding comparative analysis...", flush=True)
    analysis = build_comparative_analysis(graphs, metadata)

    # Save analysis
    analysis_file = output_dir / "comparative_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved: {analysis_file}")

    # Format for LLM
    llm_text = format_for_llm(analysis, graphs)
    llm_file = output_dir / "llm_prompt.md"
    with open(llm_file, "w") as f:
        f.write(llm_text)
    print(f"Saved: {llm_file}")

    # Call LLM for analysis
    llm_analysis = None
    if not args.skip_llm:
        print(f"\nCalling {args.llm_model} for analysis...", flush=True)
        try:
            llm_analysis = call_llm_for_analysis(
                llm_text,
                provider=args.llm_provider,
                model=args.llm_model
            )

            # Save LLM analysis
            llm_analysis_file = output_dir / "llm_analysis.md"
            with open(llm_analysis_file, "w") as f:
                f.write("# Multi-Logit Circuit Analysis\n\n")
                f.write(f"**Prompt:** {args.prompt}\n\n")
                f.write(f"**Model:** {args.llm_model}\n\n")
                f.write("---\n\n")
                f.write(llm_analysis)
            print(f"Saved: {llm_analysis_file}")

        except Exception as e:
            print(f"LLM analysis failed: {e}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total neurons: {analysis['summary']['total_neurons']}")
    print(f"Shared neurons: {analysis['summary']['shared_neurons']}")
    if llm_analysis:
        print(f"\nLLM Analysis saved to: {output_dir / 'llm_analysis.md'}")
    for logit, count in analysis['summary']['logit_specific_counts'].items():
        print(f"Unique to {repr(logit)}: {count}")
    print(f"\nLLM prompt saved to: {llm_file}")
    print(f"Token count estimate: ~{len(llm_text.split())} words")


if __name__ == "__main__":
    main()
