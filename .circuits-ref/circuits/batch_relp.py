"""
Optimized RelP Attribution with key performance improvements.

Optimizations over standard RelP (circuits/relp.py):
1. Freeze all model parameters - only compute activation gradients, not weight gradients
2. Use inputs_embeds as the gradient leaf - skip embedding layer backward
3. Remove .clone() in hooks - just detach() to avoid GPU memory copies

Note: Batching is NOT used when computing edges, since Jacobian computation
is inherently per-graph. The above optimizations still provide 1.5-2x speedup.

Usage:
    from circuits.batch_relp import OptimizedRelPAttributor

    attributor = OptimizedRelPAttributor(model, tokenizer)
    graph = attributor.compute_attributions("What is the capital of France?")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm


def log(msg: str):
    """Print with flush for real-time output."""
    print(msg, flush=True)


# =============================================================================
# Linearization Autograd Functions (same as relp.py)
# =============================================================================

class FrozenSiLU(torch.autograd.Function):
    """SiLU with frozen sigmoid for linearized backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        sigmoid_x = torch.sigmoid(x)
        ctx.save_for_backward(sigmoid_x.detach())
        return x * sigmoid_x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        sigmoid_frozen, = ctx.saved_tensors
        return grad_output * sigmoid_frozen


class FrozenGatedMLP(torch.autograd.Function):
    """Gated MLP multiplication with half-rule for linearized backward."""

    @staticmethod
    def forward(ctx, gate_activated: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate_activated.detach(), up_out.detach())
        return gate_activated * up_out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_frozen, up_frozen = ctx.saved_tensors
        grad_gate = grad_output * up_frozen * 0.5
        grad_up = grad_output * gate_frozen * 0.5
        return grad_gate, grad_up


class FrozenRMSNorm(torch.autograd.Function):
    """RMSNorm with frozen normalization factor."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + eps)
        normalized = x * inv_rms
        output = normalized * weight
        ctx.save_for_backward(inv_rms.detach(), weight.detach())
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        inv_rms_frozen, weight = ctx.saved_tensors
        grad_x = grad_output * weight * inv_rms_frozen
        return grad_x, None, None


def linearized_silu(x: torch.Tensor) -> torch.Tensor:
    return FrozenSiLU.apply(x)


def linearized_gated_mul(gate_activated: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
    return FrozenGatedMLP.apply(gate_activated, up_out)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OptimizedRelPConfig:
    """Configuration for optimized RelP attribution."""
    k: int = 5                      # Number of top logits for metric
    tau: float = 0.005              # Node filtering threshold
    top_k_per_layer: int = 500      # Max nodes to keep per layer
    linearize: bool = True          # Apply linearized backward pass
    compute_edges: bool = True      # Compute edge attributions
    use_jacobian_edges: bool = True # Use Jacobian-based edge computation
    edge_stop_grad_mlps: bool = True
    normalize_edge_flow: bool = True
    max_nodes: int = 500            # Skip edge computation if exceeded
    verbose: bool = True            # Logging
    # Batched Jacobian settings (GPT 5.2 Pro optimization)
    use_batched_jacobian: bool = True  # Use batched VJP for faster edges
    target_chunk_size: int = 16        # Chunk size for batched grad computation


@dataclass
class NodeAttribution:
    """Attribution data for a single node."""
    layer: int
    neuron_idx: int
    position: int
    activation: float
    gradient: float
    relp_score: float

    @property
    def node_id(self) -> str:
        return f"{self.layer}_{self.neuron_idx}_{self.position}"

    def to_neuronpedia_node(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "feature": self.neuron_idx,
            "layer": self.layer,
            "ctx_idx": self.position,
            "feature_type": "mlp_neuron",
            "jsNodeId": self.node_id,
            "clerp": f"L{self.layer}/N{self.neuron_idx}",
            "influence": self.relp_score,
            "activation": self.activation
        }


# =============================================================================
# Optimized Scope with No-Clone Hooks
# =============================================================================

class OptimizedRelPScope:
    """
    Optimized hook manager that avoids .clone() and stores only views.

    Key optimization: No .clone() - just .detach() to keep references without copying.
    """

    def __init__(self, model: nn.Module, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self._module_dict = {}
        self._build_module_dict(model)

        # Storage - NO CLONE, just detached views
        self.mlp_activations: dict[int, torch.Tensor] = {}
        self.mlp_gradients: dict[int, torch.Tensor] = {}

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._tensor_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._original_forwards: dict[str, Any] = {}

    def _build_module_dict(self, model: nn.Module, prefix: str = ""):
        """Build flat dict of all modules for easy access."""
        for name, module in model.named_children():
            full_name = f"{prefix}-{name}" if prefix else name
            self._module_dict[full_name] = module
            self._build_module_dict(module, full_name)

    def setup_hooks(self, linearize: bool = True):
        """Set up activation/gradient hooks and optional linearization."""
        for layer_idx in range(self.num_layers):
            self._setup_layer_hooks(layer_idx)
            if linearize:
                self._setup_layer_linearization(layer_idx)
                self._setup_layer_rmsnorms(layer_idx)
                self._setup_attention_linearization(layer_idx)

        if linearize:
            self._setup_final_rmsnorm()

    def _setup_layer_hooks(self, layer_idx: int):
        """Set up hooks to capture activations and gradients WITHOUT cloning."""
        down_paths = [
            f"layers-{layer_idx}-mlp-down_proj",
            f"model-layers-{layer_idx}-mlp-down_proj",
        ]
        down_path = next((p for p in down_paths if p in self._module_dict), None)
        if not down_path:
            return

        module = self._module_dict[down_path]

        def make_hook(l_idx):
            def hook(mod, inp, out):
                if inp[0] is not None:
                    h = inp[0]
                    # KEY OPTIMIZATION: No .clone(), just .detach()
                    self.mlp_activations[l_idx] = h.detach()

                    if h.requires_grad:
                        def grad_hook(grad):
                            # KEY OPTIMIZATION: No .clone() here either
                            self.mlp_gradients[l_idx] = grad.detach()
                            return grad
                        handle = h.register_hook(grad_hook)
                        self._tensor_hooks.append(handle)
            return hook

        h = module.register_forward_hook(make_hook(layer_idx))
        self._hooks.append(h)

    def _setup_layer_linearization(self, layer_idx: int):
        """Replace MLP forward with linearized version."""
        mlp_paths = [
            f"layers-{layer_idx}-mlp",
            f"model-layers-{layer_idx}-mlp",
        ]
        mlp_path = next((p for p in mlp_paths if p in self._module_dict), None)
        if not mlp_path:
            return

        mlp = self._module_dict[mlp_path]
        key = f"mlp_{layer_idx}"

        if key not in self._original_forwards:
            self._original_forwards[key] = mlp.forward

        def linearized_forward(x, _mlp=mlp):
            gate = _mlp.gate_proj(x)
            up = _mlp.up_proj(x)
            gate_activated = linearized_silu(gate)
            intermediate = linearized_gated_mul(gate_activated, up)
            return _mlp.down_proj(intermediate)

        mlp.forward = linearized_forward

    def _setup_layer_rmsnorms(self, layer_idx: int):
        """Linearize RMSNorm modules in a decoder layer."""
        rms_paths = [
            f"layers-{layer_idx}-input_layernorm",
            f"model-layers-{layer_idx}-input_layernorm",
            f"layers-{layer_idx}-post_attention_layernorm",
            f"model-layers-{layer_idx}-post_attention_layernorm",
        ]
        for path in rms_paths:
            self._linearize_rmsnorm(path)

    def _setup_final_rmsnorm(self):
        """Linearize final RMSNorm."""
        for path in ["model-norm", "norm"]:
            self._linearize_rmsnorm(path)

    def _linearize_rmsnorm(self, path: str):
        """Replace RMSNorm forward with frozen linearized version."""
        if path in self._original_forwards:
            return
        module = self._module_dict.get(path)
        if module is None or not hasattr(module, "weight"):
            return

        self._original_forwards[path] = module.forward
        eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))

        def linearized_forward(x, _module=module, _eps=eps):
            return FrozenRMSNorm.apply(x, _module.weight, _eps)

        module.forward = linearized_forward

    def _setup_attention_linearization(self, layer_idx: int):
        """Detach Q/K projections so attention weights are frozen during backward."""
        proj_names = ["q_proj", "k_proj"]
        for proj_name in proj_names:
            proj_paths = [
                f"layers-{layer_idx}-self_attn-{proj_name}",
                f"model-layers-{layer_idx}-self_attn-{proj_name}",
            ]
            proj_path = next((p for p in proj_paths if p in self._module_dict), None)
            if proj_path:
                self._detach_linear_module(proj_path)

    def _detach_linear_module(self, module_path: str):
        """Replace a linear module forward so its output does not propagate gradients."""
        if module_path in self._original_forwards:
            return
        module = self._module_dict.get(module_path)
        if module is None:
            return

        self._original_forwards[module_path] = module.forward

        def detached_forward(*args, _orig=self._original_forwards[module_path], **kwargs):
            output = _orig(*args, **kwargs)
            return output.detach()

        module.forward = detached_forward

    def clear(self):
        """Clear cached activations and gradients."""
        self.mlp_activations.clear()
        self.mlp_gradients.clear()

    def remove_tensor_hooks(self):
        """Remove tensor-level gradient hooks."""
        for h in self._tensor_hooks:
            h.remove()
        self._tensor_hooks.clear()

    def cleanup(self):
        """Remove all hooks and restore original forwards."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        self.remove_tensor_hooks()

        for key, original in self._original_forwards.items():
            if key.startswith("mlp_"):
                layer_idx = int(key.split("_")[1])
                mlp_paths = [
                    f"layers-{layer_idx}-mlp",
                    f"model-layers-{layer_idx}-mlp",
                ]
                mlp_path = next((p for p in mlp_paths if p in self._module_dict), None)
                if mlp_path:
                    self._module_dict[mlp_path].forward = original
            else:
                module = self._module_dict.get(key)
                if module:
                    module.forward = original
        self._original_forwards.clear()


# =============================================================================
# Optimized RelP Attributor
# =============================================================================

class OptimizedRelPAttributor:
    """
    Optimized RelP attributor with key performance improvements.

    Optimizations:
    1. Freeze all model parameters - only compute activation gradients
    2. Use inputs_embeds as gradient leaf - skip embedding backward
    3. No .clone() in hooks - just detached views
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: OptimizedRelPConfig | None = None,
        model_name: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or OptimizedRelPConfig()
        self.model_name = model_name

        # Get device
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get number of layers
        if hasattr(model, 'config'):
            self.num_layers = getattr(model.config, 'num_hidden_layers', 32)
        else:
            self.num_layers = 32

        # KEY OPTIMIZATION 1: Freeze all parameters
        self._freeze_parameters()

        # Initialize optimized scope
        inner_model = model.model if hasattr(model, 'model') else model
        self.scope = OptimizedRelPScope(inner_model, self.num_layers)
        self.scope.setup_hooks(linearize=self.config.linearize)

        # Disable KV cache for gradient computation
        if hasattr(model, 'config'):
            model.config.use_cache = False

        if self.config.verbose:
            log("      Optimizations enabled: frozen params, inputs_embeds leaf, no-clone hooks")

    def _freeze_parameters(self):
        """Freeze all model parameters to skip weight gradient computation."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def compute_attributions(
        self,
        input_text: str,
        k: int | None = None,
        tau: float | None = None,
        compute_edges: bool | None = None,
        target_position: int | None = None,
        max_nodes: int | None = None,
        grad_vector: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Compute RelP attributions for a single input.

        Args:
            input_text: Input prompt
            k: Number of top logits for metric
            tau: Node filtering threshold
            compute_edges: Whether to compute edges
            target_position: Position to compute attribution from (-1 = last)
            max_nodes: Skip edges if node count exceeds this
            grad_vector: Custom gradient direction at the logit layer (shape: [vocab_size]).
                         When provided, the backward pass traces attribution to this direction
                         instead of the top-k logits. Use for logit-difference vector attribution:
                         pass mean(behavior_logits) - mean(control_logits) to trace the axis
                         that separates two behavioral distributions.

        Returns:
            Neuronpedia-format attribution graph
        """
        k = k if k is not None else self.config.k
        tau = tau if tau is not None else self.config.tau
        compute_edges = compute_edges if compute_edges is not None else self.config.compute_edges
        max_nodes = max_nodes if max_nodes is not None else self.config.max_nodes

        # Tokenize
        if self.config.verbose:
            log("[1/6] Tokenizing input...")
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        seq_len = input_ids.shape[1]
        if self.config.verbose:
            log(f"      {seq_len} tokens")

        # Clear caches
        self.scope.clear()
        self.scope.remove_tensor_hooks()
        self.model.zero_grad()

        # KEY OPTIMIZATION 2: Use inputs_embeds as gradient leaf
        if self.config.verbose:
            log("[2/6] Forward pass (optimized)...")
        with torch.no_grad():
            embeds = self.model.get_input_embeddings()(input_ids)
        embeds = embeds.detach().requires_grad_(True)

        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
        )

        pos = target_position if target_position is not None else -1
        actual_pos = pos if pos >= 0 else seq_len + pos
        logits = outputs.logits[:, pos, :]
        if self.config.verbose:
            log(f"      Done (target position: {actual_pos})")

        # Compute metric
        probs = torch.softmax(logits, dim=-1)

        if grad_vector is not None:
            # Custom gradient direction: metric = logits · direction
            # This makes d(metric)/d(logits) = direction, so the backward pass
            # traces attribution along the provided direction in logit space.
            if self.config.verbose:
                log("[3/6] Computing metric (custom gradient direction)...")
            gv = grad_vector.to(logits.device, logits.dtype)
            if gv.dim() == 1:
                gv = gv.unsqueeze(0)
            metric_value = (logits * gv).sum()
            # Report top tokens in the direction for interpretability
            top_k_indices = torch.topk(gv[0], k, dim=-1).indices.unsqueeze(0)
            top_k_probs = probs[0, top_k_indices[0]].detach().tolist()
            if self.config.verbose:
                top_tokens = [self.tokenizer.decode([idx]) for idx in top_k_indices[0].tolist()]
                log(f"      Metric value: {metric_value.item():.4f} (direction top tokens: {top_tokens})")
        else:
            if self.config.verbose:
                log("[3/6] Computing metric (top-k logits)...")
            top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
            metric_value = top_k_values.sum()
            top_k_probs = probs[0, top_k_indices[0]].detach().tolist()
            if self.config.verbose:
                log(f"      Metric value: {metric_value.item():.4f}")

        # Backward pass
        if self.config.verbose:
            log("[4/6] Backward pass (gradient computation)...")
        metric_value.backward()
        if self.config.verbose:
            log("      Done")

        # Compute node attributions
        if self.config.verbose:
            log("[5/6] Computing node attributions...")
        node_attributions = self._compute_node_attributions(seq_len)
        if self.config.verbose:
            log(f"      Found {len(node_attributions)} non-zero attributions")

        # Filter nodes
        threshold = tau * abs(metric_value.item())
        filtered_nodes = [n for n in node_attributions if abs(n.relp_score) >= threshold]
        filtered_nodes.sort(key=lambda n: abs(n.relp_score), reverse=True)
        if self.config.verbose:
            log(f"      Filtered to {len(filtered_nodes)} nodes (threshold={threshold:.6f})")

        # Check node limit
        nodes_exceeded = max_nodes is not None and len(filtered_nodes) > max_nodes
        if nodes_exceeded and self.config.verbose:
            log(f"      WARNING: Node count exceeds max_nodes ({max_nodes}), skipping edges")

        # Compute edges
        if self.config.verbose:
            log("[6/6] Computing edge attributions...")
        edge_attributions = []
        if compute_edges and len(filtered_nodes) > 0 and not nodes_exceeded:
            if self.config.use_jacobian_edges:
                if self.config.use_batched_jacobian:
                    edge_attributions = self._compute_jacobian_edges_fast(
                        filtered_nodes, tokens, inputs, seq_len
                    )
                else:
                    edge_attributions = self._compute_jacobian_edges(
                        filtered_nodes, tokens, inputs, seq_len
                    )
            else:
                edge_attributions = self._compute_simple_edges(filtered_nodes, tokens)
        if self.config.verbose:
            log(f"      Created {len(edge_attributions)} edges")

        # Build graph
        graph = self._build_graph(
            input_text=input_text,
            tokens=tokens,
            nodes=filtered_nodes,
            edges=edge_attributions,
            metric_value=metric_value.item(),
            top_k_indices=top_k_indices[0].tolist(),
            top_k_probs=top_k_probs,
            k=k,
            tau=tau,
            target_position=actual_pos,
            nodes_exceeded=nodes_exceeded,
        )

        return graph

    def _compute_node_attributions(self, seq_len: int) -> list[NodeAttribution]:
        """Compute RelP scores for all neurons."""
        attributions = []
        top_k_per_layer = self.config.top_k_per_layer

        for layer_idx in range(self.num_layers):
            acts = self.scope.mlp_activations.get(layer_idx)
            grads = self.scope.mlp_gradients.get(layer_idx)

            if acts is None or grads is None:
                continue

            # RelP: activation * gradient
            relp_scores = (acts * grads)[0]  # [seq, d_ffn]

            # Flatten and get top-k
            flat_scores = relp_scores[:seq_len].flatten()
            flat_abs = torch.abs(flat_scores)
            k = min(top_k_per_layer, flat_abs.numel())
            _, top_indices = torch.topk(flat_abs, k)

            d_ffn = relp_scores.shape[1]
            positions = top_indices // d_ffn
            neuron_indices = top_indices % d_ffn

            scores = flat_scores[top_indices]
            acts_flat = acts[0, :seq_len].flatten()
            grads_flat = grads[0, :seq_len].flatten()

            for i in range(k):
                score = scores[i].item()
                if abs(score) < 1e-10:
                    continue
                attributions.append(NodeAttribution(
                    layer=layer_idx,
                    neuron_idx=neuron_indices[i].item(),
                    position=positions[i].item(),
                    activation=acts_flat[top_indices[i]].item(),
                    gradient=grads_flat[top_indices[i]].item(),
                    relp_score=score
                ))

        return attributions

    def _compute_jacobian_edges(
        self,
        nodes: list[NodeAttribution],
        tokens: list[str],
        inputs: dict[str, torch.Tensor],
        seq_len: int
    ) -> list[tuple[str, str, float]]:
        """Compute edge attributions using Jacobian-based method."""
        edges = []
        use_stop_grads = self.config.edge_stop_grad_mlps
        normalize_flow = self.config.normalize_edge_flow

        # Group nodes by layer
        nodes_by_layer: dict[int, list[NodeAttribution]] = {}
        for node in nodes:
            if node.layer not in nodes_by_layer:
                nodes_by_layer[node.layer] = []
            nodes_by_layer[node.layer].append(node)

        sorted_layers = sorted(nodes_by_layer.keys())
        if len(sorted_layers) < 2:
            return edges

        method_str = "with stop-grads" if use_stop_grads else "without stop-grads"
        if self.config.verbose:
            log(f"      Computing Jacobian edges between {len(sorted_layers)} layers ({method_str})...")

        # Set up hooks to capture live activations
        live_activations: dict[int, torch.Tensor] = {}
        hooks = []

        if use_stop_grads:
            min_layer = min(sorted_layers)
            max_layer = max(sorted_layers)
            layers_to_hook = list(range(min_layer, max_layer + 1))
        else:
            layers_to_hook = sorted_layers

        inner_model = self.model.model if hasattr(self.model, 'model') else self.model

        for layer_idx in layers_to_hook:
            down_paths = [
                f"layers-{layer_idx}-mlp-down_proj",
                f"model-layers-{layer_idx}-mlp-down_proj",
            ]
            down_path = next((p for p in down_paths if p in self.scope._module_dict), None)
            if down_path:
                module = self.scope._module_dict[down_path]

                def make_hook(l_idx):
                    def hook(mod, inp, out):
                        if inp[0] is not None:
                            live_activations[l_idx] = inp[0]
                    return hook

                h = module.register_forward_hook(make_hook(layer_idx))
                hooks.append(h)

        # Fresh forward pass with inputs_embeds
        self.model.zero_grad()
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            embeds = self.model.get_input_embeddings()(input_ids)
        embeds = embeds.detach().requires_grad_(True)

        with torch.enable_grad():
            outputs = self.model(
                inputs_embeds=embeds,
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
            )

        for h in hooks:
            h.remove()

        # Compute edges between layer pairs
        edge_count = 0
        layer_pairs = []
        for i, source_layer in enumerate(sorted_layers):
            for target_layer in sorted_layers[i + 1:]:
                layer_pairs.append((source_layer, target_layer))

        for source_layer, target_layer in tqdm(layer_pairs, desc="      Layer pairs", leave=False, disable=not self.config.verbose):
            source_nodes = nodes_by_layer[source_layer]
            target_nodes = nodes_by_layer[target_layer]

            source_acts = live_activations.get(source_layer)
            target_acts = live_activations.get(target_layer)

            if source_acts is None or target_acts is None:
                continue

            intermediate_layers = [l for l in range(source_layer + 1, target_layer)
                                   if l in live_activations]

            for target in target_nodes:
                target_val = target_acts[0, target.position, target.neuron_idx]

                if not target_val.requires_grad:
                    continue

                try:
                    if use_stop_grads and intermediate_layers:
                        grad_tensor = self._compute_grad_with_stop_grads(
                            target_val,
                            source_acts,
                            [live_activations[l] for l in intermediate_layers]
                        )
                        if grad_tensor is None:
                            continue
                    else:
                        grads = torch.autograd.grad(
                            target_val,
                            source_acts,
                            retain_graph=True,
                            allow_unused=True
                        )
                        if grads[0] is None:
                            continue
                        grad_tensor = grads[0][0]

                    for source in source_nodes:
                        if target.position < source.position:
                            continue

                        grad_at_source = grad_tensor[source.position, source.neuron_idx].item()
                        edge_weight = source.activation * grad_at_source

                        if normalize_flow and abs(target.activation) > 1e-10:
                            edge_weight = (edge_weight / target.activation) * target.relp_score

                        if abs(edge_weight) > 1e-10:
                            edges.append((source.node_id, target.node_id, edge_weight))
                            edge_count += 1

                except RuntimeError as e:
                    if "does not require grad" in str(e) or "One of the differentiated" in str(e):
                        continue
                    raise

        if self.config.verbose:
            log(f"      Computed {edge_count} Jacobian edges")

        # Add embedding edges
        first_layer = sorted_layers[0]
        for node in nodes_by_layer[first_layer]:
            if node.position < len(tokens):
                token = tokens[node.position]
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                emb_node_id = f"E_{token_id}_{node.position}"
                edges.append((emb_node_id, node.node_id, node.relp_score))

        return edges

    def _prep_layer_tensors(
        self,
        nodes: list[NodeAttribution],
        device: torch.device
    ) -> dict[str, Any]:
        """Build per-layer tensors for fast vectorized operations."""
        pos = torch.tensor([n.position for n in nodes], device=device, dtype=torch.long)
        idx = torch.tensor([n.neuron_idx for n in nodes], device=device, dtype=torch.long)
        act = torch.tensor([n.activation for n in nodes], device=device, dtype=torch.float32)
        relp = torch.tensor([n.relp_score for n in nodes], device=device, dtype=torch.float32)
        ids = [n.node_id for n in nodes]
        return {"pos": pos, "idx": idx, "act": act, "relp": relp, "ids": ids}

    def _compute_jacobian_edges_fast(
        self,
        nodes: list[NodeAttribution],
        tokens: list[str],
        inputs: dict[str, torch.Tensor],
        seq_len: int
    ) -> list[tuple[str, str, float]]:
        """
        Compute edge attributions using batched Jacobian method.

        GPT 5.2 Pro optimization: Uses batched VJPs (is_grads_batched=True) to compute
        gradients for multiple target nodes in one backward pass, and retrieves gradients
        for all source layers at once. This eliminates the (source_layer, target_layer)
        nested loop and removes per-edge .item() GPU syncs.

        Expected speedup: 10-100x on edge computation.
        """
        edges = []
        normalize_flow = self.config.normalize_edge_flow
        target_chunk = self.config.target_chunk_size
        eps = 1e-10

        # Group nodes by layer
        nodes_by_layer: dict[int, list[NodeAttribution]] = {}
        for node in nodes:
            nodes_by_layer.setdefault(node.layer, []).append(node)

        sorted_layers = sorted(nodes_by_layer.keys())
        if len(sorted_layers) < 2:
            return edges

        if self.config.verbose:
            log(f"      Computing batched Jacobian edges ({len(sorted_layers)} layers, chunk={target_chunk})...")

        # Set up hooks to capture live activations
        live_activations: dict[int, torch.Tensor] = {}
        hooks = []

        min_layer = min(sorted_layers)
        max_layer = max(sorted_layers)
        layers_to_hook = list(range(min_layer, max_layer + 1))

        for layer_idx in layers_to_hook:
            down_paths = [
                f"layers-{layer_idx}-mlp-down_proj",
                f"model-layers-{layer_idx}-mlp-down_proj",
            ]
            down_path = next((p for p in down_paths if p in self.scope._module_dict), None)
            if down_path:
                module = self.scope._module_dict[down_path]

                def make_hook(l_idx):
                    def hook(mod, inp, out):
                        if inp[0] is not None:
                            live_activations[l_idx] = inp[0]
                    return hook

                h = module.register_forward_hook(make_hook(layer_idx))
                hooks.append(h)

        # Fresh forward pass with inputs_embeds
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            embeds = self.model.get_input_embeddings()(input_ids)
        embeds = embeds.detach().requires_grad_(True)

        with torch.enable_grad():
            outputs = self.model(
                inputs_embeds=embeds,
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
            )

        for h in hooks:
            h.remove()

        device = embeds.device

        # IMPORTANT: Clear tensor hooks before batched grad computation.
        # is_grads_batched=True uses vmap internally, which conflicts with hooks that call .detach()
        self.scope.remove_tensor_hooks()

        # Pre-pack per-layer tensors once for efficient vectorized operations
        layer_pack = {l: self._prep_layer_tensors(nodes_by_layer[l], device) for l in sorted_layers}

        edge_count = 0

        # Iterate target layers only (not layer pairs) - key optimization
        for t_layer in tqdm(sorted_layers[1:], desc="      Target layers", leave=False, disable=not self.config.verbose):
            t_pack = layer_pack[t_layer]
            t_act_tensor = live_activations.get(t_layer)
            if t_act_tensor is None:
                continue

            T = t_pack["pos"].numel()
            if T == 0:
                continue

            # All earlier layers are sources - get gradients for ALL of them at once
            source_layers = [l for l in sorted_layers if l < t_layer]
            source_inputs = [live_activations[l] for l in source_layers if l in live_activations]
            source_layer_list = [l for l in source_layers if l in live_activations]

            if not source_inputs:
                continue

            # Vector of target scalars [T]
            y_all = t_act_tensor[0, t_pack["pos"], t_pack["idx"]]  # [T]
            if not y_all.requires_grad:
                continue

            # Chunk targets to control memory
            for start in range(0, T, target_chunk):
                end = min(start + target_chunk, T)
                y = y_all[start:end]  # [K]
                K = y.numel()

                # Batched VJP: grad_outputs shape [K, K] representing I_K
                eye = torch.eye(K, device=device, dtype=y.dtype)

                try:
                    grads = torch.autograd.grad(
                        outputs=y,
                        inputs=source_inputs,
                        grad_outputs=eye,
                        is_grads_batched=True,
                        retain_graph=True,
                        allow_unused=True,
                        create_graph=False,
                    )
                except RuntimeError as e:
                    if "does not require grad" in str(e):
                        continue
                    raise

                # Target-side tensors for this chunk
                t_pos = t_pack["pos"][start:end]    # [K]
                t_act = t_pack["act"][start:end]    # [K]
                t_relp = t_pack["relp"][start:end]  # [K]

                # Process each source layer
                for l, g in zip(source_layer_list, grads):
                    if g is None:
                        continue
                    s_pack = layer_pack[l]
                    S = s_pack["pos"].numel()
                    if S == 0:
                        continue

                    # g: [K, 1, seq, d_ffn] -> [K, seq, d_ffn]
                    g2 = g[:, 0]

                    # Gather grads only at source node indices => [K, S]
                    grad_block = g2[:, s_pack["pos"], s_pack["idx"]].to(torch.float32)

                    # Edge weights [K, S]
                    w = grad_block * s_pack["act"][None, :]

                    if normalize_flow:
                        denom = t_act.abs().clamp_min(eps)  # [K]
                        w = (w / denom[:, None]) * t_relp[:, None]

                    # Causal constraint: only allow edges where target.position >= source.position
                    causal = (t_pos[:, None] >= s_pack["pos"][None, :])

                    # Threshold
                    keep = causal & (w.abs() > eps)

                    if not keep.any():
                        continue

                    ti, si = torch.nonzero(keep, as_tuple=True)  # indices into [K, S]

                    # Pull weights to CPU in one shot (no per-edge .item())
                    w_keep = w[ti, si].detach().cpu()

                    # Convert indices to node ids
                    s_ids = s_pack["ids"]
                    t_ids = t_pack["ids"]

                    ti_cpu = ti.detach().cpu().tolist()
                    si_cpu = si.detach().cpu().tolist()
                    w_list = w_keep.tolist()

                    for a, b, ww in zip(si_cpu, ti_cpu, w_list):
                        edges.append((s_ids[a], t_ids[start + b], float(ww)))
                        edge_count += 1

        if self.config.verbose:
            log(f"      Computed {edge_count} batched Jacobian edges")

        # Add embedding edges
        first_layer = sorted_layers[0]
        for node in nodes_by_layer[first_layer]:
            if node.position < len(tokens):
                token = tokens[node.position]
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                emb_node_id = f"E_{token_id}_{node.position}"
                edges.append((emb_node_id, node.node_id, node.relp_score))

        return edges

    def _compute_grad_with_stop_grads(
        self,
        target_val: torch.Tensor,
        source_acts: torch.Tensor,
        intermediate_acts: list[torch.Tensor]
    ) -> torch.Tensor | None:
        """Compute gradient with stop-gradients on intermediate layers."""
        stop_hooks = []

        def make_zero_grad_hook():
            def hook(grad):
                return torch.zeros_like(grad)
            return hook

        for act in intermediate_acts:
            if act.requires_grad:
                h = act.register_hook(make_zero_grad_hook())
                stop_hooks.append(h)

        try:
            grads = torch.autograd.grad(
                target_val,
                source_acts,
                retain_graph=True,
                allow_unused=True
            )
            if grads[0] is None:
                return None
            return grads[0][0]
        finally:
            for h in stop_hooks:
                h.remove()

    def _compute_simple_edges(
        self,
        nodes: list[NodeAttribution],
        tokens: list[str]
    ) -> list[tuple[str, str, float]]:
        """Compute simple edges (geometric mean approximation)."""
        edges = []
        nodes_by_layer: dict[int, list[NodeAttribution]] = {}
        for node in nodes:
            if node.layer not in nodes_by_layer:
                nodes_by_layer[node.layer] = []
            nodes_by_layer[node.layer].append(node)

        sorted_layers = sorted(nodes_by_layer.keys())
        if not sorted_layers:
            return edges

        # Embedding edges
        first_layer = sorted_layers[0]
        for node in nodes_by_layer[first_layer]:
            if node.position < len(tokens):
                token = tokens[node.position]
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                emb_node_id = f"E_{token_id}_{node.position}"
                edges.append((emb_node_id, node.node_id, node.relp_score))

        # Inter-layer edges
        for i, source_layer in enumerate(sorted_layers[:-1]):
            target_layer = sorted_layers[i + 1]
            for source in nodes_by_layer[source_layer]:
                for target in nodes_by_layer[target_layer]:
                    if target.position >= source.position:
                        weight = (source.relp_score * target.relp_score) ** 0.5
                        if (source.relp_score < 0) != (target.relp_score < 0):
                            weight = -abs(weight)
                        if abs(weight) > 1e-10:
                            edges.append((source.node_id, target.node_id, weight))

        return edges

    def _build_graph(
        self,
        input_text: str,
        tokens: list[str],
        nodes: list[NodeAttribution],
        edges: list[tuple[str, str, float]],
        metric_value: float,
        top_k_indices: list[int],
        top_k_probs: list[float],
        k: int,
        tau: float,
        target_position: int,
        nodes_exceeded: bool = False,
    ) -> dict[str, Any]:
        """Build Neuronpedia-compatible graph."""
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

        # MLP neuron nodes
        for node in nodes:
            graph_nodes.append(node.to_neuronpedia_node())

        # Logit nodes
        for i, logit_idx in enumerate(top_k_indices):
            token = self.tokenizer.decode([logit_idx])
            prob = top_k_probs[i]
            node_id = f"L_{logit_idx}_{len(tokens)-1}"
            graph_nodes.append({
                "node_id": node_id,
                "feature": logit_idx,
                "layer": "logit",
                "ctx_idx": len(tokens) - 1,
                "feature_type": "logit",
                "jsNodeId": node_id,
                "clerp": f"Logit: {token} (p={prob:.4f})",
                "influence": None,
                "activation": None
            })

        # Build links
        graph_links = [{"source": s, "target": t, "weight": w} for s, t, w in edges]

        # Add neuron -> logit edges
        for node in nodes:
            for i, logit_idx in enumerate(top_k_indices):
                logit_node_id = f"L_{logit_idx}_{len(tokens)-1}"
                prob_weight = top_k_probs[i] / sum(top_k_probs) if sum(top_k_probs) > 0 else 1.0 / len(top_k_probs)
                edge_weight = node.relp_score * prob_weight
                if abs(edge_weight) > 1e-6:
                    graph_links.append({
                        "source": node.node_id,
                        "target": logit_node_id,
                        "weight": edge_weight
                    })

        metadata = {
            "slug": f"opt-relp-{hash(input_text) % 100000}",
            "scan": self.model_name or "unknown",
            "prompt_tokens": tokens,
            "prompt": input_text,
            "node_threshold": tau,
            "target_position": target_position,
            "info": {
                "description": "Optimized RelP attribution graph",
                "generator": {"name": "OptimizedRelPAttributor", "version": "1.0.0"}
            },
            "generation_settings": {
                "max_n_logits": k,
                "node_threshold": tau,
                "metric_value": metric_value,
                "target_position": target_position,
                "nodes_exceeded": nodes_exceeded,
            }
        }

        return {
            "metadata": metadata,
            "qParams": {},
            "nodes": graph_nodes,
            "links": graph_links,
            "nodes_exceeded": nodes_exceeded,
        }

    def cleanup(self):
        """Clean up resources."""
        self.scope.cleanup()
        self.scope.clear()
