"""LLM-based circuit analysis for clustered attribution graphs.

Computes inter-module flow, applies functional splitting, and synthesizes circuit descriptions.
"""

import os
import re
from collections import defaultdict
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# Functional Splitting (split modules into semantic sub-modules)
# ============================================================================

def get_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Get embeddings for a list of texts using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(texts, show_progress_bar=False)
    except ImportError:
        # Fallback: simple TF-IDF + SVD
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf = vectorizer.fit_transform(texts)

        n_components = min(50, tfidf.shape[0] - 1, tfidf.shape[1])
        if n_components > 1:
            svd = TruncatedSVD(n_components=n_components)
            return svd.fit_transform(tfidf)
        return tfidf.toarray()


def split_by_prompt_answer(
    neurons: list[dict],
    answer_start_pos: int,
    min_group_size: int = 3
) -> list[list[dict]]:
    """Split neurons into prompt-responding vs answer-responding groups.

    Args:
        neurons: List of neuron dicts with 'position' field
        answer_start_pos: Token position where the answer begins
        min_group_size: Minimum neurons per group

    Returns:
        List of [prompt_neurons, answer_neurons] if both large enough
    """
    if not neurons or answer_start_pos <= 0:
        return [neurons]

    prompt_neurons = []
    answer_neurons = []

    for n in neurons:
        pos = n.get('position', 0)
        # Convert position to int for comparison
        if isinstance(pos, str):
            try:
                pos = int(pos)
            except ValueError:
                pos = 0
        if pos < answer_start_pos:
            prompt_neurons.append(n)
        else:
            answer_neurons.append(n)

    result = []
    if len(prompt_neurons) >= min_group_size:
        result.append(prompt_neurons)
    if len(answer_neurons) >= min_group_size:
        result.append(answer_neurons)

    # If only one group is large enough, return all neurons together
    if len(result) < 2:
        return [neurons]

    return result


def split_by_position(
    neurons: list[dict],
    min_group_size: int = 3,
    max_gap: int = 3
) -> list[list[dict]]:
    """Split neurons into contiguous token position spans.

    Groups neurons by their token position, splitting when there's a gap
    larger than max_gap between positions.

    Args:
        neurons: List of neuron dicts with 'position' field
        min_group_size: Minimum neurons per group (smaller groups merged)
        max_gap: Maximum gap between positions before splitting

    Returns:
        List of neuron groups, split by position gaps
    """
    if not neurons:
        return [neurons]

    # Group neurons by position (convert to int for sorting)
    pos_groups = {}
    for n in neurons:
        pos = n.get('position', 0)
        # Convert position to int
        if isinstance(pos, str):
            try:
                pos = int(pos)
            except ValueError:
                pos = 0
        if pos not in pos_groups:
            pos_groups[pos] = []
        pos_groups[pos].append(n)

    if len(pos_groups) <= 1:
        return [neurons]

    positions = sorted(pos_groups.keys())

    # Split into contiguous spans based on gaps
    spans = []
    current_span = []
    last_pos = None

    for pos in positions:
        if last_pos is None or pos - last_pos <= max_gap:
            current_span.extend(pos_groups[pos])
        else:
            # Gap exceeded - start new span
            if current_span:
                spans.append(current_span)
            current_span = pos_groups[pos].copy()
        last_pos = pos

    if current_span:
        spans.append(current_span)

    # Filter out tiny spans, merge them into neighbors
    result = []
    for span in spans:
        if len(span) >= min_group_size:
            result.append(span)
        elif result:
            # Merge tiny span into previous
            result[-1].extend(span)
        # else: will be merged into next span or dropped

    return result if len(result) > 1 else [neurons]


def split_by_layer_range(neurons: list[dict], max_span: int = 10) -> list[list[dict]]:
    """Split neurons if they span too many layers."""
    if not neurons:
        return [neurons]

    layers = [int(n.get('layer', 0)) for n in neurons if str(n.get('layer', '')).isdigit()]
    if not layers:
        return [neurons]

    min_layer, max_layer = min(layers), max(layers)

    if max_layer - min_layer <= max_span:
        return [neurons]

    # Split into early, mid, late
    mid_layer = (min_layer + max_layer) // 2
    early = [n for n in neurons if str(n.get('layer', '')).isdigit() and int(n.get('layer', 0)) <= mid_layer - 3]
    late = [n for n in neurons if str(n.get('layer', '')).isdigit() and int(n.get('layer', 0)) >= mid_layer + 3]
    mid = [n for n in neurons if str(n.get('layer', '')).isdigit() and mid_layer - 3 < int(n.get('layer', 0)) < mid_layer + 3]

    result = []
    for group in [early, mid, late]:
        if len(group) >= 3:
            result.append(group)

    return result if result else [neurons]


def split_module_semantically(
    neurons: list[dict],
    min_size: int = 3,
    max_clusters: int = 5
) -> list[list[dict]]:
    """Split a module's neurons into semantic sub-clusters."""
    if len(neurons) < min_size * 2:
        return [neurons]

    # Get labels
    labels = []
    for n in neurons:
        label = n.get('label', '')
        if ':' in label:
            label = label.split(':', 1)[1].strip()
        labels.append(label if label else 'unknown')

    # Get embeddings
    embeddings = get_embeddings(labels)

    # Determine optimal number of clusters
    n_clusters = min(max_clusters, len(neurons) // min_size)
    n_clusters = max(2, n_clusters)

    # Hierarchical clustering
    try:
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(embeddings)
    except:
        try:
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clustering.fit_predict(embeddings)
        except:
            return [neurons]

    # Group neurons by cluster
    sub_modules = {}
    for i, neuron in enumerate(neurons):
        c = cluster_labels[i]
        if c not in sub_modules:
            sub_modules[c] = []
        sub_modules[c].append(neuron)

    # Filter out tiny clusters
    result = [neurons_list for neurons_list in sub_modules.values() if len(neurons_list) >= min_size]

    if len(result) == 0:
        return [neurons]

    return result


def split_by_llm(
    neurons: list[dict],
    module_name: str = "",
    module_context: str = "",
    top_logits: list[str] = None,
    min_size: int = 3,
    max_groups: int = 4,
    model: str = "auto",
    provider: str = "auto"
) -> list[list[dict]]:
    """Use LLM to intelligently split a module based on neuron functions.

    Args:
        neurons: List of neuron dicts with 'label' field
        module_name: Name of the module being split
        module_context: Description of input/output modules and flow
        top_logits: List of top output tokens (alternatives the model is considering)
        min_size: Minimum neurons per group
        max_groups: Maximum number of groups to create
        model: LLM model to use
        provider: LLM provider

    Returns:
        List of neuron groups based on LLM's functional analysis
    """
    if len(neurons) < min_size * 2:
        return [neurons]

    # Build neuron list for prompt
    neuron_list = []
    for i, n in enumerate(neurons):
        label = n.get('label', '')
        if ':' in label:
            label = label.split(':', 1)[1].strip()
        layer = n.get('layer', '?')
        neuron_id = n.get('neuron', '?')
        neuron_list.append(f"  {i}. L{layer}/N{neuron_id}: {label or 'unlabeled'}")

    # Build alternatives context
    alternatives_context = ""
    if top_logits:
        alternatives_context = f"""
**Output alternatives the model is considering:** {', '.join(repr(t) for t in top_logits[:5])}
Look for neurons that might specifically encode one of these alternatives vs another!
"""

    neurons_text = '\n'.join(neuron_list)

    prompt = f"""Analyze this neural network module and split it into functionally coherent sub-groups.

Module: {module_name or 'Unnamed module'}
{f'Context: {module_context}' if module_context else ''}
{alternatives_context}

Neurons in this module:
{neurons_text}

Task: Group neurons that serve the SAME function together. Each group should be coherent - neurons that work together toward the same goal.

Split criteria (in rough priority order):

1. **Semantic coherence**: Neurons about the same concept belong together
   - Group "dopamine-related" neurons separately from "serotonin-related" neurons
   - Group "disease" neurons separately from "treatment" neurons
   - If you see competing alternatives in the output, neurons encoding different alternatives should be in separate groups

2. **Functional role**: Neurons doing the same job belong together
   - Content/meaning neurons vs formatting/syntax neurons
   - Detection/recognition neurons vs output generation neurons

3. **Spurious/off-topic neurons**: If some neurons seem unrelated to the main task, group them separately
   - This helps isolate the "real" circuit from noise

Output format:
GROUP 1: [what function/concept do these neurons share?]
Neurons: 0, 3, 5, 7

GROUP 2: [what function/concept do these neurons share?]
Neurons: 1, 2, 4, 6

Rules:
- Create 2-{max_groups} groups (only if genuinely distinct functions exist)
- Each group needs at least {min_size} neurons
- Every neuron must be assigned to exactly one group
- Use neuron indices (0, 1, 2, ...) from the list above
- Prioritize semantic coherence: neurons about the same concept should be together
"""

    try:
        response = call_llm(prompt, model=model, provider=provider)

        # Parse response to extract groups
        groups = []
        current_indices = []

        for line in response.split('\n'):
            line = line.strip()
            # Remove markdown bold formatting for matching
            clean_line = line.replace('**', '').replace('*', '')
            if clean_line.lower().startswith('neurons:') or clean_line.lower().startswith('indices:'):
                # Extract indices from this line
                indices_part = clean_line.split(':', 1)[1]
                indices = []
                for part in indices_part.replace(',', ' ').split():
                    try:
                        idx = int(part.strip())
                        if 0 <= idx < len(neurons):
                            indices.append(idx)
                    except ValueError:
                        continue
                if indices:
                    current_indices = indices
            elif ('group' in clean_line.lower() and clean_line.lower().lstrip('#').strip().startswith('group')) and current_indices:
                # Save previous group and start new one
                if current_indices:
                    groups.append(current_indices)
                current_indices = []

        # Don't forget the last group
        if current_indices:
            groups.append(current_indices)

        # Convert indices to neuron lists
        result = []
        used_indices = set()
        for indices in groups:
            group_neurons = [neurons[i] for i in indices if i not in used_indices]
            used_indices.update(indices)
            if len(group_neurons) >= min_size:
                result.append(group_neurons)

        # Add any unassigned neurons to the largest group
        unassigned = [neurons[i] for i in range(len(neurons)) if i not in used_indices]
        if unassigned and result:
            largest_group = max(result, key=len)
            largest_group.extend(unassigned)

        if len(result) > 1:
            return result

    except Exception as e:
        import sys
        print(f"LLM split failed: {e}", file=sys.stderr)

    return [neurons]


def reassign_modules_with_llm(
    module_summaries: list[dict],
    flow_matrix: list[list[float]] = None,
    top_logits: list[str] = None,
    model: str = "auto",
    provider: str = "auto",
    verbose: bool = True
) -> tuple[list[dict], dict]:
    """Use LLM to perform final module reassignment for coherent groupings.

    This is a cleanup step after initial clustering and splitting. The LLM reviews
    all modules and their neurons, then reassigns neurons to create more coherent
    functional groupings.

    Args:
        module_summaries: Current module summaries with neurons
        flow_matrix: Inter-module flow matrix (optional, for context)
        top_logits: Top output tokens (for context about what model is computing)
        model: LLM model to use
        provider: LLM provider
        verbose: Print progress

    Returns:
        Tuple of (new_module_summaries, reassignment_info)
    """
    import json as json_module

    if not module_summaries:
        return module_summaries, {"reassigned": False}

    # Build module descriptions for prompt
    module_descriptions = []
    all_neurons = []  # Track all neurons with their current module
    neuron_index = 0

    for mod in module_summaries:
        mod_id = mod.get('cluster_id', '?')
        neurons = mod.get('top_neurons', mod.get('neurons', []))

        neuron_lines = []
        for n in neurons:
            layer = n.get('layer', '?')
            neuron_id = n.get('neuron', '?')
            pos = n.get('position', '?')
            label = n.get('label', '')
            if ':' in str(label):
                label = label.split(':', 1)[1].strip()

            neuron_lines.append(f"    N{neuron_index}: L{layer}/N{neuron_id}@pos{pos} - {label or 'unlabeled'}")
            all_neurons.append({
                'index': neuron_index,
                'original_module': mod_id,
                'neuron_data': n
            })
            neuron_index += 1

        module_descriptions.append(f"Module {mod_id} ({len(neurons)} neurons):\n" + "\n".join(neuron_lines))

    # Build context about output
    output_context = ""
    if top_logits:
        output_context = f"\nOutput tokens being predicted: {', '.join(repr(t) for t in top_logits[:5])}\n"

    # Build flow context if available
    flow_context = ""
    if flow_matrix and len(flow_matrix) == len(module_summaries):
        significant_flows = []
        for i, row in enumerate(flow_matrix):
            for j, flow in enumerate(row):
                if abs(flow) > 0.1 and i != j:
                    src_id = module_summaries[i].get('cluster_id', i)
                    dst_id = module_summaries[j].get('cluster_id', j)
                    significant_flows.append(f"  M{src_id} → M{dst_id}: {flow:.2f}")
        if significant_flows:
            flow_context = "\nSignificant inter-module flows:\n" + "\n".join(significant_flows[:20]) + "\n"

    prompt = f"""Review these neural network modules and reassign neurons to create more coherent functional groupings.
{output_context}{flow_context}
Current modules:
{chr(10).join(module_descriptions)}

## Layer Roles (important for grouping)
Neurons at different layer depths typically serve different functional roles:
- **Early layers (0-8)**: Pattern recognition and feature detection - identify specific tokens, n-grams, or syntactic patterns
- **Middle layers (9-20)**: Conceptual processing - build semantic representations, connect related concepts, reasoning
- **Late layers (21-31)**: Output preparation - select responses, format outputs, final decision-making

Neurons spanning many layers (e.g., L3 to L28) are almost certainly doing different jobs and should usually be in separate modules, even if they relate to the same topic.

## Position Considerations
- Neurons on the same physical token position often work together
- Neurons spanning many positions (10+ tokens apart) need strong semantic justification to be grouped
- Prompt-region neurons (earlier positions) vs answer-region neurons (later positions) often have different roles

Your task: Reassign neurons to create coherent modules where:
1. Neurons with similar functions AND similar layer depths are grouped together
2. Avoid grouping neurons that span more than ~10 layers unless they're clearly the same feature
3. Each module should have a clear, unified purpose at a specific processing stage
4. Spurious/off-topic neurons can be grouped into a "misc" module

You may:
- Merge modules (or parts of modules) that have similar functions at similar layers
- Split modules where neurons span too many layers or have clearly different functions
- Move individual neurons between modules
- Create new modules if needed

Output a JSON object with this structure:
```json
{{
  "modules": [
    {{
      "id": "0",
      "name": "Brief descriptive name",
      "function": "What this module does",
      "neurons": [0, 3, 5, 7]
    }},
    {{
      "id": "1",
      "name": "Another module name",
      "function": "What this module does",
      "neurons": [1, 2, 4, 6]
    }}
  ],
  "rationale": "Brief explanation of major changes made"
}}
```

Rules:
- Use neuron indices (N0, N1, etc.) from the list above
- Every neuron must be assigned to exactly one module
- Each module needs at least 2 neurons
- Module IDs MUST be numeric strings: "0", "1", "2", etc. (not descriptive names)
- Put the descriptive name in the "name" field instead
- Prefer more, smaller modules over fewer, larger ones when in doubt

**STRICT CONSTRAINTS - MUST BE FOLLOWED:**
1. **MAX LAYER SPAN = 12**: No module can have neurons spanning more than 12 layers. If min layer is L5, max layer must be ≤L17. SPLIT modules that violate this.
2. **MAX MODULE SIZE = 12**: No module can have more than 12 neurons. SPLIT larger groups by layer tier.
3. **SPLIT CATCH-ALL MODULES**: Do NOT create a single "misc" or "general" module. Instead create: misc_early (L0-10), misc_mid (L11-20), misc_late (L21-31).

Before finalizing, CHECK each module's layer range. If any module spans >12 layers, split it.
"""

    try:
        if verbose:
            print("  Calling LLM for module reassignment...")

        response = call_llm(prompt, model=model, provider=provider)

        # Parse JSON from response
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                if verbose:
                    print("  Warning: Could not parse LLM reassignment response")
                return module_summaries, {"reassigned": False, "error": "parse_failed"}

        result = json_module.loads(json_str)

        # Build new module summaries from reassignment
        new_modules = []
        assigned_neurons = set()

        for mod_spec in result.get('modules', []):
            mod_id = mod_spec.get('id', str(len(new_modules)))
            # Convert to int if numeric (for viewer compatibility)
            if isinstance(mod_id, str) and mod_id.isdigit():
                mod_id = int(mod_id)
            neuron_indices = mod_spec.get('neurons', [])
            # Convert indices to int (LLM may return strings like "0", "N0", or ints)
            parsed_indices = []
            for i in neuron_indices:
                if isinstance(i, int):
                    parsed_indices.append(i)
                elif isinstance(i, str):
                    # Handle "N0", "N1" format
                    clean = i.strip().upper()
                    if clean.startswith('N'):
                        clean = clean[1:]
                    if clean.isdigit():
                        parsed_indices.append(int(clean))
            neuron_indices = parsed_indices

            # Collect neurons for this module
            mod_neurons = []
            for idx in neuron_indices:
                if isinstance(idx, int) and idx < len(all_neurons) and idx not in assigned_neurons:
                    mod_neurons.append(all_neurons[idx]['neuron_data'])
                    assigned_neurons.add(idx)

            if mod_neurons:
                # Compute module statistics from neurons
                # Convert position and layer to int (may be stored as string)
                positions = []
                layers = []
                for n in mod_neurons:
                    pos = n.get('position', 0)
                    layer = n.get('layer', 0)
                    # Convert position
                    if isinstance(pos, str):
                        try:
                            positions.append(int(pos))
                        except ValueError:
                            positions.append(0)
                    else:
                        positions.append(int(pos) if pos else 0)
                    # Convert layer
                    if isinstance(layer, str):
                        try:
                            layers.append(int(layer))
                        except ValueError:
                            layers.append(0)
                    else:
                        layers.append(int(layer) if layer else 0)
                tokens = [n.get('token', '') for n in mod_neurons if n.get('token')]
                influences = [abs(n.get('influence', 0) or 0) for n in mod_neurons]

                new_modules.append({
                    'cluster_id': mod_id,
                    'name': mod_spec.get('name', f'Module {mod_id}'),
                    'function': mod_spec.get('function', ''),
                    'top_neurons': mod_neurons,
                    'size': len(mod_neurons),
                    # Add computed fields for compatibility
                    'position_span': [min(positions), max(positions)] if positions else [0, 0],
                    'position_range': [min(positions), max(positions)] if positions else [0, 0],
                    'layer_range': [min(layers), max(layers)] if layers else [0, 0],
                    'top_tokens': [(t, 1) for t in list(set(tokens))[:5]],
                    'total_influence': sum(influences),
                    'mean_influence': sum(influences) / len(influences) if influences else 0,
                    'outgoing_flow': 0.0,  # Will need to recompute if important
                    'incoming_flow': 0.0,
                })

        # Handle any unassigned neurons
        unassigned = [all_neurons[i]['neuron_data'] for i in range(len(all_neurons)) if i not in assigned_neurons]
        if verbose:
            print(f"  Total neurons: {len(all_neurons)}, Assigned: {len(assigned_neurons)}, Unassigned: {len(unassigned)}")
        if unassigned:
            # Convert position and layer to int (may be stored as string)
            positions = []
            layers = []
            for n in unassigned:
                pos = n.get('position', 0)
                layer = n.get('layer', 0)
                # Convert position
                if isinstance(pos, str):
                    try:
                        positions.append(int(pos))
                    except ValueError:
                        positions.append(0)
                else:
                    positions.append(int(pos) if pos else 0)
                # Convert layer
                if isinstance(layer, str):
                    try:
                        layers.append(int(layer))
                    except ValueError:
                        layers.append(0)
                else:
                    layers.append(int(layer) if layer else 0)
            tokens = [n.get('token', '') for n in unassigned if n.get('token')]
            influences = [abs(n.get('influence', 0) or 0) for n in unassigned]
            new_modules.append({
                'cluster_id': 'misc',
                'name': 'Miscellaneous',
                'function': 'Unassigned neurons',
                'top_neurons': unassigned,
                'size': len(unassigned),
                'position_span': [min(positions), max(positions)] if positions else [0, 0],
                'position_range': [min(positions), max(positions)] if positions else [0, 0],
                'layer_range': [min(layers), max(layers)] if layers else [0, 0],
                'top_tokens': [(t, 1) for t in list(set(tokens))[:5]],
                'total_influence': sum(influences),
                'mean_influence': sum(influences) / len(influences) if influences else 0,
                'outgoing_flow': 0.0,
                'incoming_flow': 0.0,
            })

        # Enforce constraints by splitting modules that violate them
        pre_enforce_count = len(new_modules)
        pre_enforce_neurons = sum(len(m.get('top_neurons', [])) for m in new_modules)
        new_modules = enforce_module_constraints(new_modules, max_layer_span=12, max_size=12, verbose=verbose)
        post_enforce_neurons = sum(len(m.get('top_neurons', [])) for m in new_modules)
        if verbose and pre_enforce_neurons != post_enforce_neurons:
            print(f"  WARNING: Neurons lost in constraint enforcement: {pre_enforce_neurons} → {post_enforce_neurons}")

        # Renumber all modules with sequential integer IDs
        for i, mod in enumerate(new_modules):
            mod['cluster_id'] = i

        # Compute layer_mean for each module (needed for viewer)
        for mod in new_modules:
            layers = []
            for n in mod.get('top_neurons', []):
                layer = n.get('layer', 0)
                if isinstance(layer, str):
                    try:
                        layers.append(int(layer))
                    except ValueError:
                        pass
                else:
                    layers.append(int(layer) if layer else 0)
            mod['layer_mean'] = sum(layers) / len(layers) if layers else 0

        reassignment_info = {
            'reassigned': True,
            'original_modules': len(module_summaries),
            'new_modules': len(new_modules),
            'rationale': result.get('rationale', '')
        }

        if verbose:
            print(f"  Reassigned: {len(module_summaries)} → {pre_enforce_count} → {len(new_modules)} modules (after constraint enforcement)")
            if result.get('rationale'):
                print(f"  Rationale: {result['rationale'][:100]}...")

        return new_modules, reassignment_info

    except Exception as e:
        import sys
        print(f"  LLM reassignment failed: {e}", file=sys.stderr)
        return module_summaries, {"reassigned": False, "error": str(e)}


def enforce_module_constraints(
    modules: list[dict],
    max_layer_span: int = 12,
    max_size: int = 12,
    verbose: bool = False
) -> list[dict]:
    """Split modules that violate layer span or size constraints.

    Args:
        modules: List of module summaries
        max_layer_span: Maximum allowed layer span (e.g., 12 means L5-L17 is ok)
        max_size: Maximum neurons per module
        verbose: Print debug info

    Returns:
        List of modules with constraints enforced
    """
    result_modules = []
    stragglers = []  # Collect neurons from small tiers to avoid dropping them
    next_id = 0

    # Find the highest existing numeric ID to continue from
    for m in modules:
        cid = m.get('cluster_id')
        if isinstance(cid, int):
            next_id = max(next_id, cid + 1)
        elif isinstance(cid, str) and cid.isdigit():
            next_id = max(next_id, int(cid) + 1)

    for mod in modules:
        neurons = mod.get('top_neurons', [])
        if not neurons:
            continue

        # Get layer info for each neuron
        neuron_layers = []
        for n in neurons:
            layer = n.get('layer', 0)
            # Convert layer to int (may be stored as string)
            if isinstance(layer, str):
                try:
                    layer = int(layer)
                except ValueError:
                    layer = 0
            elif isinstance(layer, float):
                layer = int(layer)
            elif not isinstance(layer, int):
                layer = 0
            neuron_layers.append((layer, n))

        # Check if module violates constraints
        layers_only = [l for l, _ in neuron_layers]
        layer_span = max(layers_only) - min(layers_only) if layers_only else 0

        if layer_span <= max_layer_span and len(neurons) <= max_size:
            # Module is fine, keep as-is
            result_modules.append(mod)
            continue

        # Need to split this module
        if verbose:
            print(f"  Splitting M{mod.get('cluster_id')} (span={layer_span}, size={len(neurons)})")

        # Split by layer tiers: early (0-10), mid (11-20), late (21-31)
        tiers = {'early': [], 'mid': [], 'late': []}
        for layer, neuron in neuron_layers:
            if layer <= 10:
                tiers['early'].append(neuron)
            elif layer <= 20:
                tiers['mid'].append(neuron)
            else:
                tiers['late'].append(neuron)

        base_name = mod.get('name', 'Module')

        for tier_name, tier_neurons in tiers.items():
            if len(tier_neurons) < 2:
                # Collect stragglers instead of dropping them
                stragglers.extend(tier_neurons)
                continue

            # Further split if still too large
            while tier_neurons:
                chunk = tier_neurons[:max_size]
                tier_neurons = tier_neurons[max_size:]

                if len(chunk) < 2:
                    # Collect stragglers instead of dropping
                    stragglers.extend(chunk)
                    continue

                # Compute stats for new module - convert to int for min/max
                positions = []
                for n in chunk:
                    pos = n.get('position', 0)
                    if isinstance(pos, str):
                        try:
                            positions.append(int(pos))
                        except ValueError:
                            positions.append(0)
                    else:
                        positions.append(int(pos) if pos else 0)
                chunk_layers = []
                for n in chunk:
                    layer = n.get('layer', 0)
                    if isinstance(layer, str) and layer.isdigit() or isinstance(layer, (int, float)):
                        chunk_layers.append(int(layer))
                tokens = [n.get('token', '') for n in chunk if n.get('token')]
                influences = [abs(n.get('influence', 0) or 0) for n in chunk]

                result_modules.append({
                    'cluster_id': next_id,
                    'name': f"{base_name} ({tier_name})",
                    'function': mod.get('function', ''),
                    'top_neurons': chunk,
                    'size': len(chunk),
                    'position_span': [min(positions), max(positions)] if positions else [0, 0],
                    'position_range': [min(positions), max(positions)] if positions else [0, 0],
                    'layer_range': [min(chunk_layers), max(chunk_layers)] if chunk_layers else [0, 0],
                    'top_tokens': [(t, 1) for t in list(set(tokens))[:5]],
                    'total_influence': sum(influences),
                    'mean_influence': sum(influences) / len(influences) if influences else 0,
                    'outgoing_flow': 0.0,
                    'incoming_flow': 0.0,
                })
                next_id += 1

    # Handle stragglers: merge into existing modules by layer proximity or create new module
    if stragglers:
        if verbose:
            print(f"  Collecting {len(stragglers)} straggler neurons")

        # Group stragglers by layer tier
        straggler_tiers = {'early': [], 'mid': [], 'late': []}
        for n in stragglers:
            layer = n.get('layer', 0)
            if isinstance(layer, str):
                try:
                    layer = int(layer)
                except ValueError:
                    layer = 0
            if layer <= 10:
                straggler_tiers['early'].append(n)
            elif layer <= 20:
                straggler_tiers['mid'].append(n)
            else:
                straggler_tiers['late'].append(n)

        # For each tier, either merge into an existing module or create new one
        for tier_name, tier_stragglers in straggler_tiers.items():
            if not tier_stragglers:
                continue

            # Find an existing module in this tier to merge into
            merged = False
            for mod in result_modules:
                mod_neurons = mod.get('top_neurons', [])
                if not mod_neurons:
                    continue
                # Check if module is in same tier
                sample_layer = mod_neurons[0].get('layer', 0)
                if isinstance(sample_layer, str):
                    try:
                        sample_layer = int(sample_layer)
                    except ValueError:
                        sample_layer = 0
                in_tier = (tier_name == 'early' and sample_layer <= 10) or \
                          (tier_name == 'mid' and 11 <= sample_layer <= 20) or \
                          (tier_name == 'late' and sample_layer > 20)
                if in_tier and mod['size'] + len(tier_stragglers) <= max_size:
                    mod['top_neurons'].extend(tier_stragglers)
                    mod['size'] = len(mod['top_neurons'])
                    merged = True
                    break

            if not merged:
                # Create new module for these stragglers
                positions = []
                chunk_layers = []
                for n in tier_stragglers:
                    pos = n.get('position', 0)
                    layer = n.get('layer', 0)
                    if isinstance(pos, str):
                        try:
                            positions.append(int(pos))
                        except ValueError:
                            positions.append(0)
                    else:
                        positions.append(int(pos) if pos else 0)
                    if isinstance(layer, str):
                        try:
                            chunk_layers.append(int(layer))
                        except ValueError:
                            chunk_layers.append(0)
                    else:
                        chunk_layers.append(int(layer) if layer else 0)
                tokens = [n.get('token', '') for n in tier_stragglers if n.get('token')]
                influences = [abs(n.get('influence', 0) or 0) for n in tier_stragglers]

                result_modules.append({
                    'cluster_id': next_id,
                    'name': f"Misc ({tier_name})",
                    'function': 'Collected straggler neurons',
                    'top_neurons': tier_stragglers,
                    'size': len(tier_stragglers),
                    'position_span': [min(positions), max(positions)] if positions else [0, 0],
                    'position_range': [min(positions), max(positions)] if positions else [0, 0],
                    'layer_range': [min(chunk_layers), max(chunk_layers)] if chunk_layers else [0, 0],
                    'top_tokens': [(t, 1) for t in list(set(tokens))[:5]],
                    'total_influence': sum(influences),
                    'mean_influence': sum(influences) / len(influences) if influences else 0,
                    'outgoing_flow': 0.0,
                    'incoming_flow': 0.0,
                })
                next_id += 1

    return result_modules


def recompute_flow_matrix(
    module_summaries: list[dict],
    graph: dict[str, Any],
    verbose: bool = False
) -> tuple[list[list[float]], list[dict]]:
    """Recompute inter-module flow matrix after module reassignment.

    Args:
        module_summaries: List of module summaries with top_neurons containing node_id
        graph: Original graph with links
        verbose: Print debug info

    Returns:
        Tuple of (flow_matrix, updated_module_summaries with flow stats)
    """
    if not module_summaries or not graph:
        return [], module_summaries

    # Build node_id -> module_index mapping
    node_to_module = {}
    for mod_idx, mod in enumerate(module_summaries):
        neurons = mod.get('top_neurons', mod.get('neurons', []))
        for neuron in neurons:
            node_id = neuron.get('node_id')
            if node_id:
                node_to_module[node_id] = mod_idx

    if verbose:
        print(f"  Building flow matrix for {len(module_summaries)} modules, {len(node_to_module)} nodes")

    n_modules = len(module_summaries)
    flow_matrix = [[0.0] * n_modules for _ in range(n_modules)]

    # Compute flows from graph links
    links = graph.get('links', [])
    for link in links:
        src = link.get('source')
        tgt = link.get('target')
        weight = link.get('weight', 1.0)

        src_mod = node_to_module.get(src)
        tgt_mod = node_to_module.get(tgt)

        if src_mod is not None and tgt_mod is not None and src_mod != tgt_mod:
            flow_matrix[src_mod][tgt_mod] += weight

    # Update module summaries with new flow stats
    for mod_idx, mod in enumerate(module_summaries):
        outgoing = sum(flow_matrix[mod_idx])
        incoming = sum(flow_matrix[i][mod_idx] for i in range(n_modules))
        mod['outgoing_flow'] = float(outgoing)
        mod['incoming_flow'] = float(incoming)

    return flow_matrix, module_summaries


def apply_functional_split(
    module_summaries: list[dict],
    min_module_size: int = 10,
    use_prompt_answer_split: bool = True,
    answer_start_pos: int = 0,
    use_position_split: bool = True,
    max_position_gap: int = 3,
    use_layer_split: bool = True,
    use_semantic_split: bool = True,
    use_llm_split: bool = False,
    llm_model: str = "auto",
    llm_provider: str = "auto",
    top_logits: list[str] = None
) -> tuple[list[dict], dict]:
    """Apply functional splitting to module summaries.

    Splits are applied in order:
    1. Prompt vs Answer split (if enabled) - first divide by prompt/answer tokens
    2. Position span split (if enabled) - split by contiguous token spans
    3. Layer split (if enabled) - split by early/mid/late layers
    4. Semantic split (if enabled) - split by neuron label embeddings
    5. LLM split (if enabled) - use LLM to intelligently split remaining groups

    Args:
        module_summaries: List of module summary dicts
        min_module_size: Only split modules with at least this many neurons
        use_prompt_answer_split: Split by prompt vs answer tokens first
        answer_start_pos: Token position where answer begins
        use_position_split: Split by contiguous token position spans
        max_position_gap: Max gap between positions before splitting
        use_layer_split: Split by layer ranges
        use_semantic_split: Split by neuron label semantics (sentence-transformers)
        use_llm_split: Use LLM to intelligently split based on neuron functions
        llm_model: LLM model for splitting
        llm_provider: LLM provider for splitting
        top_logits: List of top output tokens for context (helps LLM identify alternatives)

    Returns:
        Tuple of (new_modules, split_info)
    """
    new_modules = []
    split_info = {
        'original_n_modules': len(module_summaries),
        'splits': []
    }

    for module in module_summaries:
        neurons = module.get('top_neurons', [])
        size = module.get('size', len(neurons))

        if size < min_module_size:
            new_modules.append(module)
            continue

        all_submodules = [neurons]

        # Step 0: Prompt vs Answer split (first split)
        if use_prompt_answer_split and answer_start_pos > 0:
            new_groups = []
            for group in all_submodules:
                if len(group) >= min_module_size:
                    split_result = split_by_prompt_answer(group, answer_start_pos, min_group_size=3)
                    new_groups.extend(split_result)
                else:
                    new_groups.append(group)
            all_submodules = new_groups

        # Step 1: Position split (by contiguous spans)
        if use_position_split:
            new_groups = []
            for group in all_submodules:
                if len(group) >= min_module_size:
                    split_result = split_by_position(group, min_group_size=3, max_gap=max_position_gap)
                    new_groups.extend(split_result)
                else:
                    new_groups.append(group)
            all_submodules = new_groups

        # Step 2: Layer split
        if use_layer_split:
            new_groups = []
            for group in all_submodules:
                if len(group) >= 6:
                    split_result = split_by_layer_range(group)
                    new_groups.extend(split_result)
                else:
                    new_groups.append(group)
            all_submodules = new_groups

        # Step 3: Semantic split (sentence-transformers embeddings)
        if use_semantic_split:
            new_groups = []
            for group in all_submodules:
                if len(group) >= 6:
                    split_result = split_module_semantically(group)
                    new_groups.extend(split_result)
                else:
                    new_groups.append(group)
            all_submodules = new_groups

        # Step 4: LLM-based functional split
        if use_llm_split:
            new_groups = []
            module_name = module.get('name', f"Module {module.get('cluster_id', '?')}")
            for group in all_submodules:
                if len(group) >= 6:
                    split_result = split_by_llm(
                        group,
                        module_name=module_name,
                        top_logits=top_logits,
                        min_size=3,
                        model=llm_model,
                        provider=llm_provider
                    )
                    new_groups.extend(split_result)
                else:
                    new_groups.append(group)
            all_submodules = new_groups

        # Filter tiny groups
        all_submodules = [g for g in all_submodules if len(g) >= 3]

        if len(all_submodules) <= 1:
            new_modules.append(module)
        else:
            # Record the split for tracking (but create new modules, not submodules)
            base_id = module['cluster_id']
            split_info['splits'].append({
                'original_id': base_id,
                'original_name': module.get('name', ''),
                'split_into': len(all_submodules)
            })

            for j, sub_neurons in enumerate(all_submodules):
                # Convert positions to int for min/max comparison
                positions = set()
                for n in sub_neurons:
                    pos = n.get('position', 0)
                    if isinstance(pos, str):
                        try:
                            positions.add(int(pos))
                        except ValueError:
                            positions.add(0)
                    else:
                        positions.add(int(pos) if pos else 0)
                layers = [int(n.get('layer', 0)) for n in sub_neurons if str(n.get('layer', '')).isdigit()]
                tokens = [n.get('token', '') for n in sub_neurons if n.get('token')]
                unique_tokens = list(set(tokens))[:3]
                influences = [n.get('influence', 0) or 0 for n in sub_neurons]

                # Create a new module (not a submodule) - ID will be reassigned later
                new_module = {
                    'cluster_id': f"{base_id}.{j}",  # Temporary ID, will be renumbered
                    '_original_parent_id': base_id,  # Keep for edge mapping
                    'size': len(sub_neurons),
                    'layer_range': [min(layers) if layers else 0, max(layers) if layers else 0],
                    'layer_mean': np.mean(layers) if layers else 0,
                    'position_range': [min(positions) if positions else 0, max(positions) if positions else 0],
                    'top_tokens': [(t, 1) for t in unique_tokens],
                    'total_influence': sum(influences),
                    'mean_influence': np.mean(influences) if influences else 0,
                    'outgoing_flow': module.get('outgoing_flow', 0) / len(all_submodules),
                    'incoming_flow': module.get('incoming_flow', 0) / len(all_submodules),
                    'top_neurons': sub_neurons,
                }
                new_modules.append(new_module)

    # Renumber all modules with sequential integer IDs
    # Build mapping from old IDs to new IDs for edge remapping
    old_to_new_id = {}
    for i, module in enumerate(new_modules):
        old_id = module['cluster_id']
        old_to_new_id[old_id] = i
        # Also map parent ID to this new ID (for edge remapping)
        if '_original_parent_id' in module:
            parent_id = module['_original_parent_id']
            if parent_id not in old_to_new_id:
                old_to_new_id[parent_id] = i  # First split module inherits parent's edges
        module['cluster_id'] = i

    # Clean up temporary fields
    for module in new_modules:
        module.pop('_original_parent_id', None)

    split_info['new_n_modules'] = len(new_modules)
    split_info['id_mapping'] = old_to_new_id
    return new_modules, split_info


def get_answer_start_position(graph: dict[str, Any] | None) -> int:
    """Detect the token position where the model's answer begins.

    For Llama 3.1 Instruct format, this is after the assistant header:
    <|start_header_id|>assistant<|end_header_id|>

    Args:
        graph: Attribution graph with metadata containing prompt_tokens

    Returns:
        Token position where answer begins, or 0 if not detectable
    """
    if not graph:
        return 0

    metadata = graph.get('metadata', {})
    tokens = metadata.get('prompt_tokens', [])

    if not tokens:
        return 0

    # Look for the assistant header end marker
    # In Llama 3.1 format: <|start_header_id|>assistant<|end_header_id|>\n\n
    for i, tok in enumerate(tokens):
        # The token after "<|end_header_id|>" following "assistant" is the answer start
        if '<|end_header_id|>' in tok and i > 0:
            # Check if previous token was 'assistant'
            prev_tok = tokens[i-1] if i > 0 else ''
            if 'assistant' in prev_tok.lower():
                # Answer starts at the next token (i+1)
                return i + 1

    # Fallback: look for common answer prefix patterns
    # Sometimes the template adds a newline token after the header
    for i, tok in enumerate(tokens):
        if tok == 'assistant':
            # Skip header tokens and find actual answer start
            for j in range(i+1, min(i+5, len(tokens))):
                if tokens[j] not in ['<|end_header_id|>', '\n', '']:
                    return j

    return 0


def build_node_to_cluster_map(method_data: dict[str, Any]) -> dict[str, int]:
    """Map node IDs to their cluster assignments."""
    node_to_cluster = {}
    for cluster in method_data['clusters']:
        cluster_id = cluster['cluster_id']
        for member in cluster['members']:
            node_to_cluster[member['node_id']] = cluster_id
    return node_to_cluster


def compute_module_flow(
    graph: dict[str, Any] | None,
    method_data: dict[str, Any],
    clusters_data: dict[str, Any]
) -> dict[str, Any]:
    """Compute inter-module flow matrix and statistics."""
    node_to_cluster = build_node_to_cluster_map(method_data)
    n_clusters = method_data['n_clusters']
    cluster_ids = sorted(set(node_to_cluster.values()))

    flow_matrix = np.zeros((n_clusters, n_clusters))
    edge_count = np.zeros((n_clusters, n_clusters), dtype=int)
    inter_cluster_edges = defaultdict(list)

    links = graph.get('links', []) if graph else []
    intra_cluster_flow = defaultdict(float)

    for link in links:
        src = link['source']
        tgt = link['target']
        weight = link.get('weight', 1.0)

        src_cluster = node_to_cluster.get(src)
        tgt_cluster = node_to_cluster.get(tgt)

        if src_cluster is None or tgt_cluster is None:
            continue

        if src_cluster != tgt_cluster:
            flow_matrix[src_cluster, tgt_cluster] += weight
            edge_count[src_cluster, tgt_cluster] += 1
            inter_cluster_edges[(src_cluster, tgt_cluster)].append({
                'source': src, 'target': tgt, 'weight': weight
            })
        else:
            intra_cluster_flow[src_cluster] += abs(weight)

    # Compute module summaries
    module_summaries = []
    for cluster in method_data['clusters']:
        cid = cluster['cluster_id']
        members = cluster['members']

        layers = [int(m['layer']) for m in members if m['layer'].isdigit()]
        positions = [m['position'] for m in members]
        tokens = [m.get('token', '') for m in members]
        labels = [m.get('label', '') for m in members]
        influences = [m.get('influence') or 0 for m in members]

        token_counts = defaultdict(int)
        for t in tokens:
            token_counts[t] += 1
        top_tokens = sorted(token_counts.items(), key=lambda x: -x[1])[:5]

        outgoing_flow = flow_matrix[cid, :].sum()
        incoming_flow = flow_matrix[:, cid].sum()

        sorted_members = sorted(members, key=lambda x: -abs(x.get('influence') or 0))

        module_summaries.append({
            'cluster_id': cid,
            'size': len(members),
            'layer_range': [min(layers) if layers else 0, max(layers) if layers else 0],
            'layer_mean': np.mean(layers) if layers else 0,
            'position_range': [min(positions) if positions else 0, max(positions) if positions else 0],
            'top_tokens': top_tokens,
            'total_influence': sum(influences),
            'mean_influence': np.mean(influences) if influences else 0,
            'outgoing_flow': float(outgoing_flow),
            'incoming_flow': float(incoming_flow),
            'top_neurons': [
                {
                    'node_id': m['node_id'],
                    'layer': m['layer'],
                    'neuron': m['neuron'],
                    'position': m['position'],
                    'token': m.get('token', ''),
                    'label': m.get('label', ''),
                    'influence': m.get('influence') or 0
                }
                for m in sorted_members  # Keep all neurons, not just top 10
            ]
        })

    # Build edge list
    module_edges = []
    for (src, tgt), edges in inter_cluster_edges.items():
        total_weight = sum(e['weight'] for e in edges)
        module_edges.append({
            'source': src,
            'target': tgt,
            'weight': total_weight,
            'edge_count': len(edges)
        })
    module_edges.sort(key=lambda x: -abs(x['weight']))

    # Extract tokens from graph metadata
    tokens_sequence = []
    if graph:
        metadata = graph.get('metadata', {})
        prompt_tokens = metadata.get('prompt_tokens', [])
        for i, tok in enumerate(prompt_tokens):
            clean_tok = tok
            if clean_tok.startswith('\u0120') or clean_tok.startswith('Ġ'):
                clean_tok = clean_tok[1:]
            if clean_tok.startswith('\u010a') or clean_tok.startswith('Ċ'):
                clean_tok = '\\n'
            tokens_sequence.append({
                'position': i,
                'token': clean_tok,
                'raw_token': tok
            })

    # Token → module mapping
    token_to_modules = defaultdict(list)
    for m in module_summaries:
        pos_min, pos_max = m['position_range']
        for pos in range(pos_min, pos_max + 1):
            if pos not in token_to_modules or m['cluster_id'] not in [x['module_id'] for x in token_to_modules[pos]]:
                token_to_modules[pos].append({
                    'module_id': m['cluster_id'],
                    'influence': m['total_influence']
                })

    for tok in tokens_sequence:
        tok['modules'] = [x['module_id'] for x in token_to_modules.get(tok['position'], [])]

    return {
        'method': method_data['method'],
        'n_modules': n_clusters,
        'flow_matrix': flow_matrix.tolist(),
        'edge_count_matrix': edge_count.tolist(),
        'module_summaries': module_summaries,
        'module_edges': module_edges[:50],
        'prompt': clusters_data.get('prompt', ''),
        'top_logits': clusters_data.get('top_logits', []),
        'tokens': tokens_sequence,
        'token_to_modules': {k: v for k, v in token_to_modules.items()},
    }


def generate_llm_prompt(module_analysis: dict[str, Any]) -> str:
    """Generate a prompt for LLM circuit synthesis."""
    prompt_text = module_analysis.get('prompt', 'Unknown prompt')

    module_descs = []

    def format_module(m):
        tokens_str = ', '.join([f"'{t[0]}'" for t in m['top_tokens'][:3]]) if m['top_tokens'] else 'none'

        neuron_examples = []
        for n in m['top_neurons'][:3]:
            label = n['label']
            if label and 'L' in label and '/N' in label:
                desc = label.split(': ', 1)[1] if ': ' in label else label
            else:
                desc = label or f"L{n['layer']}/N{n['neuron']}"
            neuron_examples.append(f"      - {desc} (inf={n['influence']:.2f})")

        neurons_text = '\n'.join(neuron_examples) if neuron_examples else "      (no labeled neurons)"

        pos_info = ""
        if m.get('position_range'):
            pos_range = m['position_range']
            pos_info = f", positions={pos_range[0]}-{pos_range[1]}"

        return f"""**Module {m['cluster_id']}** (size={m['size']}, layers={m['layer_range'][0]}-{m['layer_range'][1]}{pos_info}, influence={m['total_influence']:.2f})
    - Most active at tokens: {tokens_str}
    - Flow: incoming={m['incoming_flow']:.2f}, outgoing={m['outgoing_flow']:.2f}
    - Example neurons:
{neurons_text}"""

    # Format all modules (sorted by layer mean for logical ordering)
    sorted_modules = sorted(module_analysis['module_summaries'], key=lambda m: m.get('layer_mean', 0))
    for m in sorted_modules:
        module_descs.append(format_module(m))

    edge_descs = []
    for e in module_analysis['module_edges'][:20]:
        direction = "→" if e['weight'] > 0 else "⊣"
        edge_descs.append(
            f"  Module {e['source']} {direction} Module {e['target']}: "
            f"weight={e['weight']:.3f}, edges={e['edge_count']}"
        )

    # Note about functional splitting if it occurred
    n_total_modules = len(module_analysis['module_summaries'])
    split_note = ""
    split_info = module_analysis.get('functional_split', {})
    if split_info.get('splits'):
        n_splits = len(split_info['splits'])
        split_note = f"""

Note: {n_splits} large modules were split into smaller functional groups based on:
- **Position**: Which input tokens the neurons respond to
- **Layer**: Early (input), mid (reasoning), or late (output) layers
- **Semantics**: Conceptual similarity of neuron descriptions
"""

    return f"""Analyze this neural network circuit extracted from a language model answering the prompt:
"{prompt_text}"

## Modules (Clusters of Neurons)
The circuit has been clustered into {n_total_modules} modules. Each module contains neurons that work together. Modules are sorted by average layer (early layers process input, late layers produce output).{split_note}

{chr(10).join(module_descs)}

## Inter-Module Information Flow
These edges show how information flows between modules. Positive = excitatory, negative = inhibitory.

{chr(10).join(edge_descs)}

## Your Task
Analyze this circuit and provide:

1. **Module Labels and Roles**: For EACH module, provide:
   - **Label**: A SHORT descriptive name (3-5 words) like "Diagnostic term detection" or "Movement interpretation"
   - **Function**: 1-2 sentence description of what the module does based on:
     * Which tokens activate it
     * Its layer range (early=input, mid=reasoning, late=output)
     * Its neuron descriptions (what concepts are they detecting?)
     * Its connections (flow in/out)

   Format as:
   **Module N: [Label]**
   - **Function:** [Description]

2. **Circuit Narrative**: Explain how information flows through this circuit to produce the answer.
   - Start from input token processing
   - Trace the key pathways to the output
   - Highlight any interesting patterns (parallel paths, inhibition, bottlenecks)

3. **Key Insights**: What's interesting or notable about this circuit? Any bottlenecks, parallel pathways, or unexpected connections?"""


SYSTEM_MSG = """You are an expert in neural network interpretability, specializing in analyzing circuits in language models. You understand how information flows through transformer layers and how different components contribute to model outputs.

When analyzing circuits:
- Pay attention to layer positions (early layers do input processing, late layers do output generation)
- Consider how neuron descriptions relate to the task
- Look for functional groupings (modules that work together)
- Identify key pathways and bottlenecks

Provide clear, specific analysis grounded in the data provided."""


def call_llm(
    prompt: str,
    model: str = "auto",
    provider: str = "auto"
) -> str:
    """Call an LLM to generate circuit analysis."""
    if provider == "auto":
        if model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith("gpt") or model.startswith("o1"):
            provider = "openai"
        else:
            if os.environ.get('ANTHROPIC_API_KEY'):
                provider = "anthropic"
                model = "claude-sonnet-4-20250514"
            elif os.environ.get('OPENAI_API_KEY'):
                provider = "openai"
                model = "gpt-5.2"
            else:
                return "Error: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set"

    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            return "Error: anthropic package not installed"

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return "Error: ANTHROPIC_API_KEY not set"

        client = anthropic.Anthropic(api_key=api_key)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=6000,  # Increased from 2000
                system=SYSTEM_MSG,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error calling Anthropic: {e}"

    elif provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            return "Error: openai package not installed"

        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return "Error: OPENAI_API_KEY not set"

        client = OpenAI(api_key=api_key)

        try:
            is_new_model = model.startswith("o1") or model.startswith("gpt-5")
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt}
                ],
            }

            if is_new_model:
                kwargs["max_completion_tokens"] = 16000
            else:
                kwargs["max_tokens"] = 6000  # Increased from 2000
                kwargs["temperature"] = 0.3

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI: {e}"

    else:
        return f"Error: Unknown provider '{provider}'"


def make_short_name(func_desc: str) -> str:
    """Create a short name from a function description."""
    desc = func_desc.strip()
    for prefix in ['This module ', 'This module is ', 'This module appears to ',
                   'Focuses on ', 'Handles ', 'Processes ', 'Responsible for ',
                   'Appears to ', 'Involves ']:
        if desc.lower().startswith(prefix.lower()):
            desc = desc[len(prefix):]
            break

    if desc:
        desc = desc[0].upper() + desc[1:]

    if len(desc) > 45:
        cut = desc[:42].rfind(' ')
        if cut > 20:
            desc = desc[:cut] + '...'
        else:
            desc = desc[:40] + '...'

    return desc


def parse_module_id(m_id_str: str):
    """Parse module ID string, returning int for numeric IDs or string for others."""
    m_id_str = m_id_str.strip()
    # Keep as string for sub-modules like "1.0" or descriptive IDs like "dopamine_core"
    if '.' in m_id_str or not m_id_str.isdigit():
        return m_id_str
    return int(m_id_str)


def extract_module_names(synthesis: str, module_summaries: list[dict]) -> dict[str, Any]:
    """Extract module labels and functions from LLM synthesis."""

    # Extract module labels and functions
    # Format: **Module N: [Label]**\n- **Function:** [description]
    # Or: **Module N.X: [Label]**\n- **Function:** [description]
    # Or: **Module name_id: [Label]**\n- **Function:** [description]
    # Labels may contain special chars like -, +, /, etc.
    module_labels = {}
    module_functions = {}

    # ID pattern matches: numeric (0, 1.2) or descriptive (dopamine_core, misc_formatting)
    id_pattern = r'[\w]+(?:\.[\w]+)?'

    # Pattern 1: **Module ID: Label** followed by Function on next line
    # Use greedy match up to closing ** to capture full label with special chars
    pattern1 = rf'\*\*Module ({id_pattern}):\s*(.+?)\*\*[^\n]*\n\s*[-–]\s*\*\*Function:\*\*\s*(.+?)(?=\n\n|\n\*\*Module|\n###|\Z)'
    matches = re.findall(pattern1, synthesis, re.DOTALL)
    for m_id, label, func in matches:
        parsed_id = parse_module_id(m_id)
        # Clean up label - remove any trailing punctuation or formatting
        clean_label = label.strip().rstrip('*').strip()
        module_labels[parsed_id] = clean_label
        # Get first line of function description
        func_first_line = func.strip().split('\n')[0].strip()
        module_functions[parsed_id] = func_first_line

    # Pattern 2: Simpler format without bold - Module ID: Label\nFunction: desc
    if not module_labels:
        pattern2 = rf'Module ({id_pattern}):\s*([^\n]+?)\n\s*[-–]?\s*\*?\*?Function:\*?\*?\s*(.+?)(?=\nModule|\n\n##|\n###|\Z)'
        matches = re.findall(pattern2, synthesis, re.DOTALL | re.IGNORECASE)
        for m_id, label, func in matches:
            parsed_id = parse_module_id(m_id)
            if parsed_id not in module_labels:
                clean_label = label.strip().rstrip('*').strip()
                module_labels[parsed_id] = clean_label
                func_text = func.strip().split('\n')[0].strip()
                module_functions[parsed_id] = func_text

    # Pattern 3: Inline format - **Module ID: Label** - Function: desc (all on one/two lines)
    if not module_labels:
        pattern3 = rf'\*\*Module ({id_pattern}):\s*([^*]+?)\*\*\s*[-–]?\s*(?:\*\*)?Function:(?:\*\*)?\s*(.+?)(?=\n\*\*Module|\n\n|\n###|\Z)'
        matches = re.findall(pattern3, synthesis, re.DOTALL | re.IGNORECASE)
        for m_id, label, func in matches:
            parsed_id = parse_module_id(m_id)
            if parsed_id not in module_labels:
                clean_label = label.strip()
                module_labels[parsed_id] = clean_label
                func_text = func.strip().split('\n')[0].strip()
                module_functions[parsed_id] = func_text

    # Assign labels and functions to modules
    for m in module_summaries:
        m_id = m['cluster_id']
        # Try both string and int versions of the ID for lookup
        # (reassignment uses string IDs like '0', but parse_module_id returns int)
        lookup_ids = [m_id]
        if isinstance(m_id, str) and m_id.isdigit():
            lookup_ids.append(int(m_id))
        elif isinstance(m_id, int):
            lookup_ids.append(str(m_id))

        # Use extracted label, or derive from function, or fallback
        found_label = None
        found_func = None
        for lid in lookup_ids:
            if lid in module_labels:
                found_label = module_labels[lid]
            if lid in module_functions:
                found_func = module_functions[lid]

        if found_label:
            m['name'] = found_label
        elif found_func:
            m['name'] = make_short_name(found_func)
        else:
            top_tok = m.get('top_tokens', [['']])[0][0] if m.get('top_tokens') else ''
            m['name'] = f"Token: {top_tok}" if top_tok else f"Module {m_id}"

        # Store full function description
        m['function'] = found_func or ''

    return {
        'module_summaries': module_summaries,
    }


def analyze_modules(
    clusters_data: dict[str, Any],
    graph: dict[str, Any] | None = None,
    method: str = "infomap",
    run_llm: bool = True,
    model: str = "auto",
    provider: str = "auto",
    functional_split: bool = False,
    functional_split_min_size: int = 10,
    use_prompt_answer_split: bool = True,
    answer_start_pos: int | None = None,
    use_position_split: bool = True,
    max_position_gap: int = 3,
    use_layer_split: bool = True,
    use_semantic_split: bool = True,
    use_llm_split: bool = False,
    use_llm_reassignment: bool = True,
    verbose: bool = True
) -> dict[str, Any]:
    """Analyze clustered modules and optionally synthesize with LLM.

    Args:
        clusters_data: Clustering results from cluster_graph()
        graph: Original attribution graph (optional, for edge info)
        method: Clustering method name
        run_llm: Whether to call LLM for synthesis
        model: LLM model name
        provider: LLM provider (anthropic, openai, or auto)
        functional_split: Whether to split large modules into functional sub-modules
        functional_split_min_size: Only split modules with at least this many neurons
        use_prompt_answer_split: Split by prompt vs answer tokens first
        answer_start_pos: Token position where answer begins (auto-detected if None)
        use_position_split: Split by contiguous token position spans
        max_position_gap: Max gap between positions before splitting
        use_layer_split: Split by layer ranges
        use_semantic_split: Split by neuron label semantics (sentence-transformers)
        use_llm_split: Use LLM to intelligently split based on neuron functions
        use_llm_reassignment: Use LLM for final module cleanup/reassignment
        verbose: Print progress

    Returns:
        Analysis results with module summaries and LLM synthesis
    """
    # Find the method data
    method_data = None
    for m in clusters_data.get('methods', []):
        if m['method'] == method:
            method_data = m
            break

    if method_data is None:
        available = [m['method'] for m in clusters_data.get('methods', [])]
        if available:
            method_data = clusters_data['methods'][0]
            if verbose:
                print(f"Method '{method}' not found, using '{method_data['method']}'")
        else:
            raise ValueError("No clustering methods found in data")

    if verbose:
        print("Computing inter-module flow...")

    module_analysis = compute_module_flow(graph, method_data, clusters_data)

    if verbose:
        print(f"Found {module_analysis['n_modules']} modules")

    # Apply functional splitting if enabled
    if functional_split:
        if verbose:
            print("Applying functional sub-splitting...")

        # Auto-detect answer start position if not provided
        effective_answer_pos = answer_start_pos
        if effective_answer_pos is None and use_prompt_answer_split:
            effective_answer_pos = get_answer_start_position(graph)
            if verbose and effective_answer_pos > 0:
                print(f"  Detected answer start position: {effective_answer_pos}")

        # Extract top logits for LLM split context (helps identify competing alternatives)
        top_logit_tokens = None
        if use_llm_split and graph:
            import re
            top_logit_tokens = []
            for node in graph.get("nodes", []):
                if node.get("isLogit") or node.get("node_id", "").startswith("L_"):
                    clerp = node.get("clerp", "")
                    # Parse token from clerp like " yes (p=0.4668)"
                    match = re.match(r'(.+?) \(p=([\d.]+)\)', clerp)
                    if match:
                        token = match.group(1)
                        prob = float(match.group(2))
                        top_logit_tokens.append((token, prob))
            # Sort by probability and extract just tokens
            top_logit_tokens.sort(key=lambda x: -x[1])
            top_logit_tokens = [t[0] for t in top_logit_tokens[:10]]
            if verbose and top_logit_tokens:
                print(f"  Top output alternatives: {top_logit_tokens[:5]}")

        new_modules, split_info = apply_functional_split(
            module_analysis['module_summaries'],
            min_module_size=functional_split_min_size,
            use_prompt_answer_split=use_prompt_answer_split,
            answer_start_pos=effective_answer_pos or 0,
            use_position_split=use_position_split,
            max_position_gap=max_position_gap,
            use_layer_split=use_layer_split,
            use_semantic_split=use_semantic_split,
            use_llm_split=use_llm_split,
            llm_model=model,
            llm_provider=provider,
            top_logits=top_logit_tokens
        )

        module_analysis['module_summaries'] = new_modules
        module_analysis['functional_split'] = split_info
        module_analysis['n_modules'] = len(new_modules)

        # Remap edges to use new module IDs
        id_mapping = split_info.get('id_mapping', {})
        if id_mapping and 'module_edges' in module_analysis:
            remapped_edges = []
            for edge in module_analysis['module_edges']:
                src = edge['source']
                tgt = edge['target']
                # Map old IDs to new IDs
                new_src = id_mapping.get(src, id_mapping.get(str(src), src))
                new_tgt = id_mapping.get(tgt, id_mapping.get(str(tgt), tgt))
                remapped_edges.append({
                    **edge,
                    'source': new_src,
                    'target': new_tgt
                })
            module_analysis['module_edges'] = remapped_edges

        if verbose:
            n_splits = len(split_info.get('splits', []))
            print(f"Split {n_splits} modules → {len(new_modules)} total modules")

        # Recompute flow matrix for new module structure
        if graph:
            if verbose:
                print("  Recomputing flow matrix for split modules...")
            new_flow_matrix, updated_modules = recompute_flow_matrix(
                new_modules, graph, verbose=verbose
            )
            module_analysis['flow_matrix'] = new_flow_matrix
            module_analysis['module_summaries'] = updated_modules

            # Rebuild module_edges from new flow matrix
            n_modules = len(updated_modules)
            new_module_edges = []
            for i in range(n_modules):
                for j in range(n_modules):
                    if i != j and abs(new_flow_matrix[i][j]) > 0.001:
                        new_module_edges.append({
                            'source': i,
                            'target': j,
                            'weight': float(new_flow_matrix[i][j]),
                            'edge_count': 1,
                        })
            new_module_edges.sort(key=lambda x: -abs(x['weight']))
            module_analysis['module_edges'] = new_module_edges[:50]

            if verbose:
                print(f"  Flow matrix: {n_modules}x{n_modules}, {len(new_module_edges)} edges")

    # LLM reassignment step - final cleanup of module assignments
    if use_llm_reassignment:
        if verbose:
            print("Performing LLM module reassignment...")

        # Extract top logits for context
        top_logit_tokens = None
        if graph:
            import re
            top_logit_tokens = []
            for node in graph.get("nodes", []):
                if node.get("isLogit") or node.get("node_id", "").startswith("L_"):
                    clerp = node.get("clerp", "")
                    match = re.match(r'(.+?) \(p=([\d.]+)\)', clerp)
                    if match:
                        token = match.group(1)
                        prob = float(match.group(2))
                        top_logit_tokens.append((token, prob))
            top_logit_tokens.sort(key=lambda x: -x[1])
            top_logit_tokens = [t[0] for t in top_logit_tokens[:10]]

        reassigned_modules, reassignment_info = reassign_modules_with_llm(
            module_analysis['module_summaries'],
            flow_matrix=module_analysis.get('flow_matrix'),
            top_logits=top_logit_tokens,
            model=model,
            provider=provider,
            verbose=verbose
        )

        if reassignment_info.get('reassigned'):
            module_analysis['module_summaries'] = reassigned_modules
            module_analysis['n_modules'] = len(reassigned_modules)
            module_analysis['llm_reassignment'] = reassignment_info

            # Recompute flow matrix for new module structure
            if graph:
                if verbose:
                    print("  Recomputing flow matrix for reassigned modules...")
                new_flow_matrix, updated_modules = recompute_flow_matrix(
                    reassigned_modules, graph, verbose=verbose
                )
                module_analysis['flow_matrix'] = new_flow_matrix
                module_analysis['module_summaries'] = updated_modules

                # Rebuild module_edges from new flow matrix
                n_modules = len(updated_modules)
                new_module_edges = []
                for i in range(n_modules):
                    for j in range(n_modules):
                        if i != j and abs(new_flow_matrix[i][j]) > 0.001:
                            new_module_edges.append({
                                'source': i,
                                'target': j,
                                'weight': float(new_flow_matrix[i][j]),
                                'edge_count': 1  # Not tracking individual edges after recompute
                            })
                new_module_edges.sort(key=lambda x: -abs(x['weight']))
                module_analysis['module_edges'] = new_module_edges[:50]

    if run_llm:
        if verbose:
            print("Generating LLM prompt...")

        llm_prompt = generate_llm_prompt(module_analysis)

        if verbose:
            print("Calling LLM for circuit synthesis...")

        synthesis = call_llm(llm_prompt, model, provider)

        if verbose and synthesis.startswith("Error"):
            print(f"Warning: {synthesis}")

        module_analysis['llm_synthesis'] = synthesis
        module_analysis['llm_prompt'] = llm_prompt

        # Extract module names from synthesis
        result = extract_module_names(synthesis, module_analysis['module_summaries'])
        module_analysis['module_summaries'] = result['module_summaries']

        if verbose:
            print("LLM synthesis complete")

    return module_analysis
