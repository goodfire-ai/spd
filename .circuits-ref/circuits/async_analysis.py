"""Async LLM analysis with rate limiting for parallel processing.

Provides async versions of LLM API calls with:
- Rate limiting (token bucket algorithm)
- Concurrent request limiting (semaphore)
- Retry with exponential backoff
- Support for both Anthropic and OpenAI

Usage:
    import asyncio
    from circuits.async_analysis import call_llm_async, RateLimiter

    # Create rate limiter
    limiter = RateLimiter(requests_per_minute=50)

    # Async call
    response = await call_llm_async(prompt, rate_limiter=limiter)

    # Or use analyze_modules_async for full analysis
    analysis = await analyze_modules_async(clusters, graph, rate_limiter=limiter)
"""

import asyncio
import os
import sys
import time
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Implements a token bucket algorithm where tokens are refilled at a
    constant rate. Each API call consumes one token. If no tokens are
    available, the caller waits until a token is available.

    Args:
        requests_per_minute: Maximum requests per minute
        burst_size: Maximum burst size (default: requests_per_minute)
    """

    def __init__(self, requests_per_minute: int, burst_size: int | None = None):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.max_tokens = burst_size or requests_per_minute
        self.tokens = self.max_tokens
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

                # Calculate wait time for next token
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)


# Default rate limiters for each provider
_rate_limiters: dict[str, RateLimiter] = {}


def get_rate_limiter(provider: str) -> RateLimiter:
    """Get or create a rate limiter for a provider."""
    if provider not in _rate_limiters:
        if provider == "anthropic":
            # Anthropic tier 1: 50 requests/min
            _rate_limiters[provider] = RateLimiter(50, burst_size=10)
        elif provider == "openai":
            # OpenAI tier 1: ~500 requests/min (varies by model)
            _rate_limiters[provider] = RateLimiter(300, burst_size=50)
        else:
            # Default conservative rate
            _rate_limiters[provider] = RateLimiter(30, burst_size=5)
    return _rate_limiters[provider]


# Semaphore to limit concurrent requests
_max_concurrent = asyncio.Semaphore(10)


SYSTEM_MSG = """You are an expert in neural network interpretability, specializing in analyzing circuits in language models. You understand how information flows through transformer layers and how different components contribute to model outputs.

When analyzing circuits:
- Pay attention to layer positions (early layers do input processing, late layers do output generation)
- Consider how neuron descriptions relate to the task
- Look for functional groupings (modules that work together)
- Identify key pathways and bottlenecks
- Critically evaluate which neurons are appropriate for the task vs spurious/off-topic

Provide clear, specific analysis grounded in the data provided. Focus on semantic/conceptual correctness when identifying spurious neurons."""


async def call_llm_async(
    prompt: str,
    model: str = "auto",
    provider: str = "auto",
    rate_limiter: RateLimiter | None = None,
    max_retries: int = 3,
) -> str:
    """Async LLM call with rate limiting and retry.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (auto, claude-*, gpt-*)
        provider: Provider (auto, anthropic, openai)
        rate_limiter: Optional rate limiter (uses default if not provided)
        max_retries: Maximum retry attempts

    Returns:
        The LLM response text
    """
    # Resolve auto settings
    if provider == "auto":
        if model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith("gpt") or model.startswith("o1"):
            provider = "openai"
        else:
            if os.environ.get("ANTHROPIC_API_KEY"):
                provider = "anthropic"
                model = "claude-sonnet-4-20250514"
            elif os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
                model = "gpt-4o"
            else:
                return "Error: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set"

    # Get rate limiter
    if rate_limiter is None:
        rate_limiter = get_rate_limiter(provider)

    # Acquire rate limit token
    await rate_limiter.acquire()

    # Limit concurrent requests
    async with _max_concurrent:
        for attempt in range(max_retries):
            try:
                if provider == "anthropic":
                    return await _call_anthropic_async(prompt, model)
                elif provider == "openai":
                    return await _call_openai_async(prompt, model)
                else:
                    return f"Error: Unknown provider '{provider}'"

            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error after {max_retries} retries: {e}"

                # Exponential backoff
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)

    return "Error: Max retries exceeded"


async def _call_anthropic_async(prompt: str, model: str) -> str:
    """Make async call to Anthropic API."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        return "Error: anthropic package not installed"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not set"

    client = AsyncAnthropic(api_key=api_key)

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=6000,
            system=SYSTEM_MSG,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        raise RuntimeError(f"Anthropic API error: {e}") from e


async def _call_openai_async(prompt: str, model: str) -> str:
    """Make async call to OpenAI API."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return "Error: openai package not installed"

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not set"

    client = AsyncOpenAI(api_key=api_key)

    try:
        is_new_model = model.startswith("o1") or model.startswith("gpt-5")
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
        }

        if is_new_model:
            kwargs["max_completion_tokens"] = 16000
        else:
            kwargs["max_tokens"] = 6000
            kwargs["temperature"] = 0.3

        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}") from e


async def split_by_llm_async(
    neurons: list[dict],
    module_name: str = "",
    module_context: str = "",
    top_logits: list[str] = None,
    min_size: int = 3,
    max_groups: int = 4,
    model: str = "auto",
    provider: str = "auto",
    rate_limiter: RateLimiter | None = None,
    verbose: bool = False,
) -> list[list[dict]]:
    """Async version of split_by_llm - use LLM to intelligently split a module.

    Args:
        neurons: List of neuron dicts with 'label' field
        module_name: Name of the module being split
        module_context: Description of input/output modules and flow
        top_logits: List of top output tokens (alternatives the model is considering)
        min_size: Minimum neurons per group
        max_groups: Maximum number of groups to create
        model: LLM model to use
        provider: LLM provider
        rate_limiter: Optional rate limiter for API calls
        verbose: Print timing information

    Returns:
        List of neuron groups based on LLM's functional analysis
    """
    if len(neurons) < min_size * 2:
        return [neurons]

    start_time = time.monotonic() if verbose else None

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
        response = await call_llm_async(
            prompt,
            model=model,
            provider=provider,
            rate_limiter=rate_limiter,
        )

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

        if verbose and start_time:
            elapsed = time.monotonic() - start_time
            print(f"    LLM split completed in {elapsed:.1f}s ({len(result)} groups)")

        if len(result) > 1:
            return result

    except Exception as e:
        print(f"LLM split failed: {e}", file=sys.stderr)

    if verbose and start_time:
        elapsed = time.monotonic() - start_time
        print(f"    LLM split completed in {elapsed:.1f}s (no split)")

    return [neurons]


async def apply_functional_split_async(
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
    top_logits: list[str] = None,
    rate_limiter: RateLimiter | None = None,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Async version of apply_functional_split.

    Splits are applied in order:
    1. Prompt vs Answer split (if enabled) - first divide by prompt/answer tokens
    2. Position span split (if enabled) - split by contiguous token spans
    3. Layer split (if enabled) - split by early/mid/late layers
    4. Semantic split (if enabled) - split by neuron label embeddings
    5. LLM split (if enabled) - use LLM to intelligently split remaining groups (ASYNC)

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
        rate_limiter: Optional rate limiter for API calls

    Returns:
        Tuple of (new_modules, split_info)
    """
    # Import sync helper functions (these don't use LLMs)
    from .analysis import (
        split_by_layer_range,
        split_by_position,
        split_by_prompt_answer,
        split_module_semantically,
    )

    new_modules = []
    split_info = {
        'original_n_modules': len(module_summaries),
        'splits': []
    }

    # Collect all LLM split tasks to run in parallel
    llm_split_tasks = []

    for module in module_summaries:
        neurons = module.get('top_neurons', [])
        size = module.get('size', len(neurons))

        if size < min_module_size:
            new_modules.append(module)
            continue

        all_submodules = [neurons]

        # Step 0: Prompt vs Answer split (first split) - sync
        if use_prompt_answer_split and answer_start_pos > 0:
            new_groups = []
            for group in all_submodules:
                if len(group) >= min_module_size:
                    split_result = split_by_prompt_answer(group, answer_start_pos, min_group_size=3)
                    new_groups.extend(split_result)
                else:
                    new_groups.append(group)
            all_submodules = new_groups

        # Step 1: Position split (by contiguous spans) - sync
        if use_position_split:
            new_groups = []
            for group in all_submodules:
                if len(group) >= min_module_size:
                    split_result = split_by_position(group, min_group_size=3, max_gap=max_position_gap)
                    new_groups.extend(split_result)
                else:
                    new_groups.append(group)
            all_submodules = new_groups

        # Step 2: Layer split - sync
        if use_layer_split:
            new_groups = []
            for group in all_submodules:
                if len(group) >= 6:
                    split_result = split_by_layer_range(group)
                    new_groups.extend(split_result)
                else:
                    new_groups.append(group)
            all_submodules = new_groups

        # Step 3: Semantic split (sentence-transformers embeddings) - sync
        if use_semantic_split:
            new_groups = []
            for group in all_submodules:
                if len(group) >= 6:
                    split_result = split_module_semantically(group)
                    new_groups.extend(split_result)
                else:
                    new_groups.append(group)
            all_submodules = new_groups

        # Step 4: LLM-based functional split - collect tasks for parallel execution
        if use_llm_split:
            module_name = module.get('name', f"Module {module.get('cluster_id', '?')}")
            for group in all_submodules:
                if len(group) >= 6:
                    # Create async task for this split
                    task = split_by_llm_async(
                        group,
                        module_name=module_name,
                        top_logits=top_logits,
                        min_size=3,
                        model=llm_model,
                        provider=llm_provider,
                        rate_limiter=rate_limiter,
                        verbose=verbose,
                    )
                    llm_split_tasks.append((module, group, task))
                else:
                    # Small group - no LLM split needed, store with None task
                    llm_split_tasks.append((module, group, None))
        else:
            # No LLM split - just store the submodules
            for group in all_submodules:
                llm_split_tasks.append((module, group, None))

    # Execute all LLM splits in parallel
    if any(task is not None for _, _, task in llm_split_tasks):
        # Gather results for tasks that need LLM
        tasks_to_run = [task for _, _, task in llm_split_tasks if task is not None]
        llm_results = await asyncio.gather(*tasks_to_run)

        # Map results back to modules
        result_idx = 0
        module_to_submodules = {}  # module -> list of submodule neuron groups

        for module, group, task in llm_split_tasks:
            mod_id = id(module)
            if mod_id not in module_to_submodules:
                module_to_submodules[mod_id] = {'module': module, 'submodules': []}

            if task is not None:
                # Get result from LLM
                split_result = llm_results[result_idx]
                result_idx += 1
                module_to_submodules[mod_id]['submodules'].extend(split_result)
            else:
                # No LLM task - use group as-is
                module_to_submodules[mod_id]['submodules'].append(group)

        # Now process all modules with their submodules
        for mod_id, data in module_to_submodules.items():
            module = data['module']
            all_submodules = data['submodules']

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
    else:
        # No LLM splits at all - process modules directly
        for module, group, _ in llm_split_tasks:
            # Each (module, group) pair represents a submodule
            # We need to group by original module
            pass  # This case is handled above when tasks is empty

        # Actually, if no LLM splits, we need to recreate the original flow
        # Re-process without LLM
        new_modules = []
        for module in module_summaries:
            neurons = module.get('top_neurons', [])
            size = module.get('size', len(neurons))

            if size < min_module_size:
                new_modules.append(module)
                continue

            all_submodules = [neurons]

            # Apply sync splits
            if use_prompt_answer_split and answer_start_pos > 0:
                new_groups = []
                for group in all_submodules:
                    if len(group) >= min_module_size:
                        split_result = split_by_prompt_answer(group, answer_start_pos, min_group_size=3)
                        new_groups.extend(split_result)
                    else:
                        new_groups.append(group)
                all_submodules = new_groups

            if use_position_split:
                new_groups = []
                for group in all_submodules:
                    if len(group) >= min_module_size:
                        split_result = split_by_position(group, min_group_size=3, max_gap=max_position_gap)
                        new_groups.extend(split_result)
                    else:
                        new_groups.append(group)
                all_submodules = new_groups

            if use_layer_split:
                new_groups = []
                for group in all_submodules:
                    if len(group) >= 6:
                        split_result = split_by_layer_range(group)
                        new_groups.extend(split_result)
                    else:
                        new_groups.append(group)
                all_submodules = new_groups

            if use_semantic_split:
                new_groups = []
                for group in all_submodules:
                    if len(group) >= 6:
                        split_result = split_module_semantically(group)
                        new_groups.extend(split_result)
                    else:
                        new_groups.append(group)
                all_submodules = new_groups

            # Filter tiny groups
            all_submodules = [g for g in all_submodules if len(g) >= 3]

            if len(all_submodules) <= 1:
                new_modules.append(module)
            else:
                base_id = module['cluster_id']
                split_info['splits'].append({
                    'original_id': base_id,
                    'original_name': module.get('name', ''),
                    'split_into': len(all_submodules)
                })

                for j, sub_neurons in enumerate(all_submodules):
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

                    new_module = {
                        'cluster_id': f"{base_id}.{j}",
                        '_original_parent_id': base_id,
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


async def analyze_modules_async(
    clusters_data: dict[str, Any],
    graph: dict[str, Any] | None = None,
    method: str = "infomap",
    model: str = "auto",
    provider: str = "auto",
    rate_limiter: RateLimiter | None = None,
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
    verbose: bool = False,
) -> dict[str, Any]:
    """Async version of analyze_modules.

    Same interface as analyze_modules but uses async LLM calls.
    This enables better throughput when processing multiple graphs
    in parallel.

    Args:
        clusters_data: Clustering results from cluster_graph()
        graph: Original attribution graph
        method: Clustering method name
        model: LLM model name
        provider: LLM provider
        rate_limiter: Optional rate limiter for API calls
        functional_split: Whether to split large modules
        functional_split_min_size: Min neurons to consider splitting
        use_prompt_answer_split: Split by prompt vs answer tokens
        answer_start_pos: Token position where answer begins
        use_position_split: Split by contiguous token position spans
        max_position_gap: Max gap between positions before splitting
        use_layer_split: Split by layer ranges
        use_semantic_split: Split by neuron label semantics
        use_llm_split: Use LLM to split (expensive)
        use_llm_reassignment: Use LLM for final module cleanup/reassignment
        verbose: Print progress

    Returns:
        Analysis results with module summaries and LLM synthesis
    """
    # Import the sync analysis module for helper functions
    from .analysis import (
        compute_module_flow,
        extract_module_names,
        generate_llm_prompt,
        get_answer_start_position,
        recompute_flow_matrix,
    )

    # Find the method data
    method_data = None
    for m in clusters_data.get("methods", []):
        if m["method"] == method:
            method_data = m
            break

    if method_data is None:
        available = [m["method"] for m in clusters_data.get("methods", [])]
        if available:
            method_data = clusters_data["methods"][0]
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
                    match = re.match(r'(.+?) \(p=([\d.]+)\)', clerp)
                    if match:
                        token = match.group(1)
                        prob = float(match.group(2))
                        top_logit_tokens.append((token, prob))
            top_logit_tokens.sort(key=lambda x: -x[1])
            top_logit_tokens = [t[0] for t in top_logit_tokens[:10]]
            if verbose and top_logit_tokens:
                print(f"  Top output alternatives: {top_logit_tokens[:5]}")

        new_modules, split_info = await apply_functional_split_async(
            module_analysis["module_summaries"],
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
            top_logits=top_logit_tokens,
            rate_limiter=rate_limiter,
            verbose=verbose,
        )

        module_analysis["module_summaries"] = new_modules
        module_analysis["functional_split"] = split_info
        module_analysis["n_modules"] = len(new_modules)

        if verbose:
            n_splits = len(split_info.get("splits", []))
            print(f"Split {n_splits} modules → {len(new_modules)} total modules")

        # Recompute flow matrix for new module structure
        if graph:
            if verbose:
                print("  Recomputing flow matrix for split modules...")
            new_flow_matrix, updated_modules = recompute_flow_matrix(
                new_modules, graph, verbose=verbose
            )
            module_analysis["flow_matrix"] = new_flow_matrix
            module_analysis["module_summaries"] = updated_modules

            # Rebuild module_edges from new flow matrix
            n_modules = len(updated_modules)
            new_module_edges = []
            for i in range(n_modules):
                for j in range(n_modules):
                    if i != j and abs(new_flow_matrix[i][j]) > 0.001:
                        new_module_edges.append({
                            "source": i,
                            "target": j,
                            "weight": float(new_flow_matrix[i][j]),
                            "edge_count": 1,
                        })
            new_module_edges.sort(key=lambda x: -abs(x["weight"]))
            module_analysis["module_edges"] = new_module_edges[:50]

            if verbose:
                print(f"  Flow matrix: {n_modules}x{n_modules}, {len(new_module_edges)} edges")

    # Generate LLM prompt and call async
    if verbose:
        print("Generating LLM prompt...")

    llm_prompt = generate_llm_prompt(module_analysis)

    if verbose:
        print("Calling LLM for circuit synthesis (async)...")

    synthesis = await call_llm_async(
        llm_prompt,
        model=model,
        provider=provider,
        rate_limiter=rate_limiter,
    )

    if verbose and synthesis.startswith("Error"):
        print(f"Warning: {synthesis}")

    module_analysis["llm_synthesis"] = synthesis
    module_analysis["llm_prompt"] = llm_prompt

    # Extract module names from synthesis
    result = extract_module_names(synthesis, module_analysis["module_summaries"])
    module_analysis["module_summaries"] = result["module_summaries"]

    if verbose:
        print("LLM synthesis complete")

    return module_analysis


async def batch_analyze_async(
    analyses_to_run: list,
    rate_limiter: RateLimiter | None = None,
    max_concurrent: int = 10,
) -> list:
    """Run multiple analyses concurrently with rate limiting.

    Args:
        analyses_to_run: List of (clusters_data, graph, kwargs) tuples
        rate_limiter: Shared rate limiter
        max_concurrent: Maximum concurrent analyses

    Returns:
        List of analysis results in same order as input
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(clusters, graph, kwargs):
        async with semaphore:
            return await analyze_modules_async(
                clusters,
                graph,
                rate_limiter=rate_limiter,
                **kwargs,
            )

    tasks = [
        run_one(clusters, graph, kwargs)
        for clusters, graph, kwargs in analyses_to_run
    ]

    return await asyncio.gather(*tasks)
