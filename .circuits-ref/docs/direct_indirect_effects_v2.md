# Direct vs Indirect Effect Analysis: Revised Plan (v2)

Based on review feedback, this document outlines an improved approach for measuring direct vs indirect effects for the 8K neuron collection.

## Key Corrections from Review

### 1. Additivity Clarification
- **Vectors add exactly**: `Δlogits_total = Δlogits_direct + Δlogits_indirect` (always true)
- **Magnitudes don't add**: `|Δ_total| ≠ |Δ_direct| + |Δ_indirect|` (due to interference)

### 2. Major Efficiency Gain: No Second Forward Pass for Direct Effect
Since freezing downstream guarantees `h_final_direct = h_final_clean + V`, we can compute:
```python
logits_direct = lm_head(final_norm(h_final_clean + V))
direct_effect = logits_direct - logits_clean
```
This eliminates one forward pass per neuron.

### 3. Scale Should Match Actual Activation
For meaningful "this neuron's effect" (not "per-unit effect"), use:
```python
scale = actual_activation  # from clean forward pass
V = down_proj[:, neuron] * scale
```

### 4. Use Specific Logit Metrics
Instead of L2 norm over entire vocab, use task-relevant metrics:
- `logit(target_token)`
- `logit(target) - logit(distractor)`

---

## Revised Computational Strategy

### Cost Analysis (8,177 neurons)

**Original approach**: 3 forwards per neuron = 24,531 forwards

**Revised approach**:
- 1 clean forward per prompt (shared across all neurons)
- 1 perturbed forward per neuron for total effect
- 0 forwards for direct effect (algebraic computation)

**Total**: 1 + 8,177 = 8,178 forwards per prompt

**Further optimization with batching**:
- Batch multiple neuron perturbations if they're at different positions
- Or batch across prompts for same neuron

---

## Implementation Plan

### Phase 1: Infrastructure Setup

#### 1.1 Efficient Forward Pass with Caching
```python
class CachedForward:
    """Single clean forward that caches everything needed."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.final_norm = model.model.norm
        self.lm_head = model.lm_head

    def clean_forward(self, prompt: str) -> dict:
        """Run clean forward, cache final hidden state and logits."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        final_hidden = None
        def capture_final(module, input, output):
            nonlocal final_hidden
            # Capture pre-norm hidden state
            final_hidden = input[0][:, -1, :].detach().clone()

        hook = self.model.model.norm.register_forward_pre_hook(capture_final)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_clean = outputs.logits[:, -1, :].clone()

        hook.remove()

        return {
            'inputs': inputs,
            'final_hidden': final_hidden,  # Pre-norm, last position
            'logits_clean': logits_clean,
            'tokens': inputs['input_ids'][0]
        }
```

#### 1.2 Direct Effect Computation (No Forward Pass)
```python
def compute_direct_effect(self, cache: dict, layer: int, neuron: int,
                          activation: float) -> torch.Tensor:
    """Compute direct effect algebraically - no forward pass needed."""

    # Get neuron's output direction
    down_proj = self.model.model.layers[layer].mlp.down_proj.weight
    V = down_proj[:, neuron] * activation

    # Direct effect: what if V went straight to output?
    h_plus_V = cache['final_hidden'] + V

    # Apply final norm and lm_head
    h_normed = self.final_norm(h_plus_V)
    logits_direct = self.lm_head(h_normed)

    direct_effect = logits_direct - cache['logits_clean']
    return direct_effect
```

#### 1.3 Total Effect Computation (One Forward Pass)
```python
def compute_total_effect(self, cache: dict, layer: int, neuron: int,
                         activation: float) -> torch.Tensor:
    """Compute total effect with one perturbed forward pass."""

    down_proj = self.model.model.layers[layer].mlp.down_proj.weight
    V = down_proj[:, neuron] * activation

    def inject_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] += V
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    hook = self.model.model.layers[layer].register_forward_hook(inject_hook)

    with torch.no_grad():
        outputs = self.model(**cache['inputs'])
        logits_perturbed = outputs.logits[:, -1, :].clone()

    hook.remove()

    total_effect = logits_perturbed - cache['logits_clean']
    return total_effect
```

### Phase 2: Metric Definition

#### 2.1 Task-Relevant Metrics
For medical domain prompts, define metrics based on actual task:

```python
class EffectMetrics:
    """Compute meaningful effect metrics."""

    @staticmethod
    def logit_diff(effect: torch.Tensor, target_id: int,
                   distractor_id: int = None) -> float:
        """Effect on target token logit (optionally vs distractor)."""
        target_effect = effect[0, target_id].item()
        if distractor_id is not None:
            distractor_effect = effect[0, distractor_id].item()
            return target_effect - distractor_effect
        return target_effect

    @staticmethod
    def top_k_effect(effect: torch.Tensor, k: int = 10) -> dict:
        """Effect magnitude on top-k affected tokens."""
        vals, ids = effect[0].abs().topk(k)
        return {
            'top_k_mean': vals.mean().item(),
            'top_k_max': vals.max().item(),
            'top_tokens': ids.tolist()
        }

    @staticmethod
    def target_prob_change(logits_clean: torch.Tensor,
                           effect: torch.Tensor,
                           target_id: int) -> float:
        """Change in probability of target token."""
        prob_clean = torch.softmax(logits_clean, dim=-1)[0, target_id]
        prob_new = torch.softmax(logits_clean + effect, dim=-1)[0, target_id]
        return (prob_new - prob_clean).item()
```

#### 2.2 Effect Ratio with Proper Interpretation
```python
def compute_effect_ratio(direct: torch.Tensor, total: torch.Tensor,
                         target_ids: list) -> dict:
    """
    Compute direct/total ratio for specific tokens.

    Note: We use the VECTOR components, not magnitudes, for interpretability.
    """
    indirect = total - direct  # Exact vector decomposition

    results = {}
    for tid in target_ids:
        d = direct[0, tid].item()
        t = total[0, tid].item()
        i = indirect[0, tid].item()

        # Ratio based on absolute contributions
        if abs(t) > 1e-6:
            direct_ratio = abs(d) / (abs(d) + abs(i))
        else:
            direct_ratio = 0.5  # Undefined, use neutral

        # Also track sign alignment (do direct and indirect agree?)
        sign_aligned = (d * i) > 0

        results[tid] = {
            'direct': d,
            'indirect': i,
            'total': t,
            'direct_ratio': direct_ratio,
            'sign_aligned': sign_aligned
        }

    return results
```

### Phase 3: Activation Extraction

#### 3.1 Get Actual Activations from Clean Forward
```python
def get_neuron_activations(model, inputs, layer: int,
                           neurons: list) -> dict:
    """Extract actual post-gating activations for specified neurons."""

    activations = {}

    def capture_hook(module, input, output):
        # For Llama: output of gate * up before down_proj
        # This is the actual scalar that multiplies down_proj columns
        gate_out = output  # [batch, seq, intermediate_size]
        for neuron in neurons:
            activations[neuron] = gate_out[0, -1, neuron].item()

    # Hook the MLP internals (after SiLU gate)
    # Note: exact hook point depends on model architecture
    hook = model.model.layers[layer].mlp.register_forward_hook(capture_hook)

    with torch.no_grad():
        _ = model(**inputs)

    hook.remove()
    return activations
```

### Phase 4: Batch Processing Pipeline

#### 4.1 Process All Neurons for One Prompt
```python
def analyze_prompt(model, tokenizer, prompt: str,
                   neuron_profiles: list,
                   target_token: str = None) -> dict:
    """
    Analyze all relevant neurons for a single prompt.

    Workflow:
    1. One clean forward (cache everything)
    2. Extract activations for all neurons
    3. Compute direct effects (algebraic, fast)
    4. Compute total effects (one forward each)
    5. Compute metrics
    """

    cache = CachedForward(model, tokenizer).clean_forward(prompt)

    # Determine target token(s) for metrics
    if target_token:
        target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
    else:
        # Use top predicted token as target
        target_id = cache['logits_clean'].argmax().item()

    results = {}

    for profile in neuron_profiles:
        layer = profile['layer']
        neuron = profile['neuron']
        nid = profile['neuron_id']

        # Get actual activation (or use mean from profile)
        activation = profile.get('mean_activation', 1.0)

        # Compute effects
        direct = compute_direct_effect(cache, layer, neuron, activation)
        total = compute_total_effect(cache, layer, neuron, activation)
        indirect = total - direct

        # Compute metrics
        metrics = compute_effect_ratio(direct, total, [target_id])

        results[nid] = {
            'layer': layer,
            'neuron': neuron,
            'activation': activation,
            'target_token': tokenizer.decode([target_id]),
            'direct_effect': metrics[target_id]['direct'],
            'indirect_effect': metrics[target_id]['indirect'],
            'total_effect': metrics[target_id]['total'],
            'direct_ratio': metrics[target_id]['direct_ratio'],
            'sign_aligned': metrics[target_id]['sign_aligned']
        }

    return results
```

#### 4.2 Batch Across Prompts with SLURM
```python
# For 8K neurons × 100 prompts = 800K total effect computations
# Split across SLURM array jobs

def create_slurm_job(neuron_batch: list, prompt_batch: list,
                     output_dir: str, job_id: int):
    """Generate SLURM job for a batch of neurons and prompts."""

    script = f"""#!/bin/bash
#SBATCH --job-name=effect-{job_id}
#SBATCH --partition=h200-reserved
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output={output_dir}/logs/job_{job_id}.out

cd /mnt/polished-lake/home/ctigges/code/attribution-graphs
source .venv/bin/activate

python scripts/compute_effects_batch.py \\
    --neurons {output_dir}/neurons_batch_{job_id}.json \\
    --prompts {output_dir}/prompts_batch_{job_id}.json \\
    --output {output_dir}/results_batch_{job_id}.json
"""
    return script
```

### Phase 5: Output Format

#### 5.1 Per-Neuron Summary (aggregated across prompts)
```python
{
    "neuron_id": "L15/N1234",
    "layer": 15,
    "neuron": 1234,
    "n_prompts": 100,
    "effect_summary": {
        "mean_direct_ratio": 0.35,
        "std_direct_ratio": 0.12,
        "median_direct_ratio": 0.33,
        "classification": "routing-dominant",  # <0.3: routing, 0.3-0.7: mixed, >0.7: logit-dominant
        "consistency": 0.85  # fraction of prompts agreeing with classification
    },
    "sign_alignment": {
        "aligned_fraction": 0.72,  # direct and indirect effects usually agree
        "interpretation": "reinforcing"  # vs "opposing"
    },
    "per_prompt_breakdown": [
        {"prompt_id": 0, "direct_ratio": 0.31, "target": "dopamine", ...},
        ...
    ]
}
```

---

## Execution Plan

### Step 1: Validate on Small Sample (Day 1)
- [ ] Implement CachedForward class
- [ ] Implement algebraic direct effect computation
- [ ] Verify: `direct_algebraic ≈ direct_frozen` (should match within numerical precision)
- [ ] Test on 10 neurons × 5 prompts

### Step 2: Define Metrics for Medical Domain (Day 1)
- [ ] Analyze prompt structure to identify natural target tokens
- [ ] Implement target token extraction heuristics
- [ ] Validate metrics make sense on sample outputs

### Step 3: Extract Actual Activations (Day 2)
- [ ] Implement activation capture hooks for Llama MLP
- [ ] Run clean forward on 100 prompts, capture activations for all 8K neurons
- [ ] Store activation statistics per neuron

### Step 4: Full Computation (Day 2-3)
- [ ] Create SLURM array job (e.g., 100 jobs, each handles ~82 neurons × 100 prompts)
- [ ] Estimated time: ~2 hours per job (8K forwards per job)
- [ ] Total GPU-hours: ~200 H200 GPU-hours

### Step 5: Aggregate and Analyze (Day 3)
- [ ] Merge results from all jobs
- [ ] Compute per-neuron summaries
- [ ] Generate classification labels
- [ ] Validate against manual inspection of sample neurons

### Step 6: Integration (Day 4)
- [ ] Add `direct_ratio` and `effect_type` to neuron profiles
- [ ] Update compositional labeling schema
- [ ] Generate report with layer-wise statistics

---

## Expected Outputs

1. **Per-neuron effect profiles** (8,177 entries):
   - Direct/indirect ratio (mean, std across prompts)
   - Effect type classification
   - Sign alignment statistics

2. **Layer-wise summary**:
   - Distribution of effect types per layer
   - Validate earlier finding: early layers routing-dominant, late layers logit-dominant

3. **Updated neuron labels**:
   - Add `effect_type` field to `data/medical_edge_stats_v2_enriched.json`
   - Add `direct_ratio` for downstream use in labeling

---

## Relation to Standard Methods

This approach is equivalent to **causal mediation analysis / activation patching** with:
- **Treatment**: Add neuron's output vector V to residual stream
- **Mediators**: All downstream layer outputs
- **Direct effect**: Controlled direct effect (mediators fixed to clean + V)
- **Indirect effect**: Natural indirect effect (mediators respond to V)

The algebraic shortcut for direct effect is valid because freezing downstream guarantees `h_final = h_clean + V`.

For neurons where this triage identifies interesting patterns, we can follow up with:
- **Path patching**: Identify which specific downstream components carry the indirect effect
- **Attribution patching / RelP**: Scalable approximation for broader analysis
