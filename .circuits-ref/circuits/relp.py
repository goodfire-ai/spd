"""RelP (Relative Propagation) attribution for neuron circuits.

Implements the attribution method from "Language Model Circuits Are Sparse in the Neuron Basis"
(Arora et al., 2025) for non-paired inputs on transformer models.

Key concepts:
- Linearization: Freeze nonlinearities during backward pass
- Node attribution: RelP_v(x) = activation_v(x) * gradient_v(x)
- Edge attribution: RelP_{s->t}(x) = v_s(x) * dv_t/dv_s

Supported models:
- Llama 3.1-8B-Instruct
- Qwen 3 32B
- OLMo 7B / OLMo 2 7B
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from .hooks import LlamaScope
from .model_configs import ModelArchConfig, get_model_config


def log(msg: str):
    """Print with flush for real-time output."""
    print(msg, flush=True)


# =============================================================================
# Linearization Context Manager
# =============================================================================

class LinearizationContext:
    """Context for managing linearized backward pass.

    Tracks frozen values from forward pass and provides hooks to modify gradients.
    """

    def __init__(self):
        self.enabled = False
        # Cache frozen values per layer
        self.frozen_sigmoid: dict[int, torch.Tensor] = {}  # For SiLU
        self.frozen_up: dict[int, torch.Tensor] = {}  # For gated MLP
        self.frozen_gate_activated: dict[int, torch.Tensor] = {}  # SiLU(gate)
        self.frozen_rms_scale: dict[int, torch.Tensor] = {}  # For RMSNorm
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def clear(self):
        """Clear all cached values."""
        self.frozen_sigmoid.clear()
        self.frozen_up.clear()
        self.frozen_gate_activated.clear()
        self.frozen_rms_scale.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

# Optional: Neuron labels from database
try:
    from .labeling import NeuronInfo, fetch_descriptions, is_database_available
    LABELS_AVAILABLE = True
except ImportError:
    LABELS_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RelPConfig:
    """Configuration for RelP attribution."""
    k: int = 5                    # Number of top logits for metric
    target_tokens: list[str] = None  # Specific tokens to trace (e.g., ["Yes", "No"]). If set, overrides k.
    contrastive_tokens: list[str] = None  # Contrastive pair: [positive, negative]. Traces logit(pos) - logit(neg).
    tau: float = 0.005            # Node filtering threshold
    compute_edges: bool = True    # Whether to compute edge attributions
    use_jacobian_edges: bool = True  # Use Jacobian-based edge computation (more accurate)
    linearize: bool = True        # Apply linearized backward pass (paper's method)
    filter_always_on: bool = True # Filter always-on neurons
    use_neuron_labels: bool = True  # Fetch labels from Transluce database
    # Edge computation options (paper describes both variants):
    # - True: Apply stop-gradients on intermediate MLPs for "direct effect" edges
    # - False: No stop-gradients, computes "total effect" including all paths
    edge_stop_grad_mlps: bool = True
    # Whether to normalize edge weights to attribution flow (paper's Flow formula)
    normalize_edge_flow: bool = True
    # Always-on neurons to filter out (model-specific)
    # If None, will be populated from model config in RelPAttributor.__init__
    # Format: [(layer, neuron_idx), ...]
    always_on_neurons: list[tuple[int, int]] | None = None


# =============================================================================
# Linearization Autograd Functions
# =============================================================================

class FrozenSiLU(torch.autograd.Function):
    """SiLU with frozen sigmoid for linearized backward pass.

    Forward: y = x * sigmoid(x)
    Backward (linearized): dy/dx = frozen_sigmoid (no derivative of sigmoid)

    The paper freezes sigma(x) at its forward-pass value, treating it as a constant
    during backprop. The linearized derivative is simply the frozen sigmoid value.

    NOTE: The half-rule is applied separately at the gated MLP multiplication
    (gate_activated * up), NOT here at SiLU. See FrozenGatedMLP for the half-rule.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        sigmoid_x = torch.sigmoid(x)
        ctx.save_for_backward(sigmoid_x.detach())
        return x * sigmoid_x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        sigmoid_frozen, = ctx.saved_tensors
        # Linearized backward: treat sigmoid(x) as frozen constant
        # Gradient is simply frozen_sigmoid (paper's Table: x_i * Freeze(σ(x_i)))
        return grad_output * sigmoid_frozen


class FrozenGatedMLP(torch.autograd.Function):
    """Gated MLP multiplication with half-rule for linearized backward.

    Forward: y = gate_activated * up_out  (where gate_activated = SiLU(gate_proj(x)))
    Backward (linearized): Apply half-rule to both branches

    The half-rule ensures attribution conservation: total attribution to both
    branches sums to the total attribution flowing in.
    """

    @staticmethod
    def forward(ctx, gate_activated: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate_activated.detach(), up_out.detach())
        return gate_activated * up_out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_frozen, up_frozen = ctx.saved_tensors
        # Half-rule: each branch gets half the gradient (with cross-term)
        grad_gate = grad_output * up_frozen * 0.5
        grad_up = grad_output * gate_frozen * 0.5
        return grad_gate, grad_up


class FrozenRMSNorm(torch.autograd.Function):
    """RMSNorm with frozen normalization factor for linearized backward.

    Forward: y = (x / sqrt(mean(x^2) + eps)) * weight
    Backward (linearized): dy/dx = weight * frozen_inv_rms

    Freezing inv_rms means we treat the normalization as a fixed scaling,
    removing the complex derivative that involves interactions between elements.
    """

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
        # Linearized: gradient passes through with frozen normalization
        grad_x = grad_output * weight * inv_rms_frozen
        return grad_x, None, None


def linearized_silu(x: torch.Tensor) -> torch.Tensor:
    """Apply SiLU with linearized backward pass."""
    return FrozenSiLU.apply(x)


def linearized_gated_mul(gate_activated: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
    """Apply gated multiplication with half-rule."""
    return FrozenGatedMLP.apply(gate_activated, up_out)


# =============================================================================
# Node Attribution Data Structure
# =============================================================================

@dataclass
class NodeAttribution:
    """Attribution data for a single node (neuron)."""
    layer: int
    neuron_idx: int
    position: int           # Token position (ctx_idx)
    activation: float
    gradient: float
    relp_score: float       # activation * gradient

    @property
    def node_id(self) -> str:
        """Neuronpedia-compatible node ID."""
        return f"{self.layer}_{self.neuron_idx}_{self.position}"

    def to_neuronpedia_node(self) -> dict[str, Any]:
        """Convert to Neuronpedia node format."""
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
# RelPScope - Extended Hook Manager with Linearization
# =============================================================================

class RelPScope(LlamaScope):
    """Extended LlamaScope with linearization support for RelP attribution.

    Adds hooks to:
    1. Cache MLP activations (post-gating, which is the "neuron" basis)
    2. Cache gradients during backward pass
    3. Optionally apply linearized backward through nonlinearities

    The key insight from the paper is to use MLP activations (h^(i)) rather than
    MLP outputs (m^(i)). The activations are the post-nonlinearity hidden states
    within the MLP, providing a "privileged basis" due to the element-wise nonlinearity.
    """

    def __init__(self, model: nn.Module, num_layers: int = 32):
        super().__init__(model)
        self.num_layers = num_layers
        self.mlp_activations: dict[int, torch.Tensor] = {}
        self.mlp_gradients: dict[int, torch.Tensor] = {}
        self._linearization_enabled = False
        self._linearization_ctx = LinearizationContext()
        # For linearization: cache sigmoid values for SiLU
        self._sigmoid_cache: dict[int, torch.Tensor] = {}
        self._tensor_hooks: list[torch.utils.hooks.RemovableHandle] = []
        # Store original MLP forward methods for restoration
        self._original_mlp_forwards: dict[int, Callable] = {}
        # Store original RMSNorm and attention projection forwards
        self._original_rms_forwards: dict[str, Callable] = {}
        self._original_attn_proj_forwards: dict[str, Callable] = {}

    def setup_mlp_hooks(self, layers: list[int] | None = None, linearize: bool = False):
        """Set up activation and gradient caching hooks for MLP layers.

        For Llama/Mistral models with SwiGLU MLP:
        - gate_proj: Linear(d_model -> d_ffn) for gating
        - up_proj: Linear(d_model -> d_ffn) for values
        - down_proj: Linear(d_ffn -> d_model) output projection
        - Activation: down_proj(SiLU(gate_proj(x)) * up_proj(x))

        The "neurons" are the intermediate activations of dimension d_ffn (14336 for Llama 3.1-8B).
        We compute: h = SiLU(gate_proj(x)) * up_proj(x)

        Args:
            layers: List of layer indices to hook. If None, hooks all layers.
            linearize: If True, apply linearized backward pass.
        """
        if layers is None:
            layers = list(range(self.num_layers))

        self._linearization_enabled = linearize

        for layer_idx in layers:
            # Hook gate_proj and up_proj to compute intermediate activations
            gate_paths = [
                f"layers-{layer_idx}-mlp-gate_proj",
                f"model-layers-{layer_idx}-mlp-gate_proj",
            ]
            up_paths = [
                f"layers-{layer_idx}-mlp-up_proj",
                f"model-layers-{layer_idx}-mlp-up_proj",
            ]
            down_paths = [
                f"layers-{layer_idx}-mlp-down_proj",
                f"model-layers-{layer_idx}-mlp-down_proj",
            ]

            gate_path = next((p for p in gate_paths if p in self._module_dict), None)
            up_path = next((p for p in up_paths if p in self._module_dict), None)
            down_path = next((p for p in down_paths if p in self._module_dict), None)

            if gate_path and up_path and down_path:
                self._add_intermediate_hooks(layer_idx, gate_path, up_path, down_path)

            # Apply linearization if requested
            if linearize:
                self._setup_layer_linearization(layer_idx)
                self._setup_layer_rmsnorms(layer_idx)
                self._setup_attention_linearization(layer_idx)

        if linearize:
            self._setup_final_rmsnorm()

    def _add_intermediate_hooks(self, layer_idx: int, gate_path: str, up_path: str, down_path: str):
        """Add hooks to capture intermediate MLP activations (the neuron basis).

        The intermediate activation h = SiLU(gate_proj(x)) * up_proj(x) is the
        "neuron basis" with dimension d_ffn. This is what the paper calls h^(i).
        """
        # Storage for intermediate values
        gate_output = [None]
        up_output = [None]

        def gate_hook(module, input, output):
            gate_output[0] = output  # Don't detach yet - need for backward

        def up_hook(module, input, output):
            up_output[0] = output

        def down_hook(module, input, output):
            # Input to down_proj is the intermediate activation h
            # input[0] has shape [batch, seq, d_ffn]
            if input[0] is not None:
                h = input[0]
                self.mlp_activations[layer_idx] = h.detach().clone()

                # Register gradient hook on the intermediate activation
                if h.requires_grad:
                    def grad_hook(grad):
                        self.mlp_gradients[layer_idx] = grad.detach().clone()
                        return grad
                    handle = h.register_hook(grad_hook)
                    self._tensor_hooks.append(handle)

        self.add_hook(gate_hook, gate_path, f'relp_gate_{layer_idx}')
        self.add_hook(up_hook, up_path, f'relp_up_{layer_idx}')
        self.add_hook(down_hook, down_path, f'relp_down_{layer_idx}')

    def _setup_layer_linearization(self, layer_idx: int):
        """Set up linearization for a single layer by monkey-patching MLP forward.

        This replaces SiLU and gated multiplication with linearized versions that
        use the half-rule during backward pass.
        """
        # Find the MLP module
        mlp_paths = [
            f"layers-{layer_idx}-mlp",
            f"model-layers-{layer_idx}-mlp",
        ]
        mlp_path = next((p for p in mlp_paths if p in self._module_dict), None)
        if mlp_path is None:
            return

        mlp = self._module_dict[mlp_path]

        # Store original forward
        if layer_idx not in self._original_mlp_forwards:
            self._original_mlp_forwards[layer_idx] = mlp.forward

        # Create linearized forward
        original_forward = self._original_mlp_forwards[layer_idx]
        ctx = self._linearization_ctx

        def linearized_forward(x):
            # Llama MLP: down_proj(act_fn(gate_proj(x)) * up_proj(x))
            gate = mlp.gate_proj(x)
            up = mlp.up_proj(x)

            # Apply linearized SiLU (with frozen sigmoid and half-rule)
            gate_activated = linearized_silu(gate)

            # Apply linearized gated multiplication (with half-rule)
            intermediate = linearized_gated_mul(gate_activated, up)

            # down_proj is linear, no modification needed
            return mlp.down_proj(intermediate)

        # Monkey-patch the forward method
        mlp.forward = linearized_forward

    def _setup_layer_rmsnorms(self, layer_idx: int):
        """Linearize the RMSNorm modules in a decoder layer."""
        rms_paths = [
            f"layers-{layer_idx}-input_layernorm",
            f"model-layers-{layer_idx}-input_layernorm",
            f"layers-{layer_idx}-post_attention_layernorm",
            f"model-layers-{layer_idx}-post_attention_layernorm",
        ]
        for path in rms_paths:
            self._linearize_rmsnorm_module(path)

    def _setup_final_rmsnorm(self):
        """Linearize the final RMSNorm (if present)."""
        final_paths = ["model-norm", "norm"]
        for path in final_paths:
            self._linearize_rmsnorm_module(path)

    def _linearize_rmsnorm_module(self, module_path: str):
        """Replace RMSNorm forward with frozen linearized version."""
        if module_path in self._original_rms_forwards:
            return
        module = self._module_dict.get(module_path)
        if module is None or not hasattr(module, "weight"):
            return

        original_forward = module.forward
        self._original_rms_forwards[module_path] = original_forward
        eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))

        def linearized_forward(x: torch.Tensor, *, _module=module, _eps=eps):
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
        if module_path in self._original_attn_proj_forwards:
            return
        module = self._module_dict.get(module_path)
        if module is None:
            return

        original_forward = module.forward
        self._original_attn_proj_forwards[module_path] = original_forward

        def detached_forward(*args, _orig=original_forward, **kwargs):
            output = _orig(*args, **kwargs)
            if isinstance(output, tuple):
                # Linear layers should not return tuples, but guard just in case
                return tuple(o.detach() if torch.is_tensor(o) else o for o in output)
            return output.detach()

        module.forward = detached_forward

    def setup_linearization(self, layers: list[int] | None = None):
        """Set up linearization hooks for the backward pass.

        This implements the key insight from the paper: replace nonlinearities
        with frozen versions during backward, so gradients reflect local linear
        approximation rather than full nonlinear derivative.

        For SiLU: dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x))) becomes
                 dy/dx = frozen_sigmoid * 0.5 (half-rule for multiplicative)

        Args:
            layers: Layers to linearize. If None, linearizes all layers.
        """
        if layers is None:
            layers = list(range(self.num_layers))

        self._linearization_enabled = True

        for layer_idx in layers:
            self._setup_layer_linearization(layer_idx)

    def restore_original_forwards(self):
        """Restore original MLP forward methods after linearization."""
        for layer_idx, original_forward in self._original_mlp_forwards.items():
            mlp_paths = [
                f"layers-{layer_idx}-mlp",
                f"model-layers-{layer_idx}-mlp",
            ]
            mlp_path = next((p for p in mlp_paths if p in self._module_dict), None)
            if mlp_path:
                self._module_dict[mlp_path].forward = original_forward
        self._original_mlp_forwards.clear()

        for module_path, original_forward in self._original_rms_forwards.items():
            module = self._module_dict.get(module_path)
            if module:
                module.forward = original_forward
        self._original_rms_forwards.clear()

        for module_path, original_forward in self._original_attn_proj_forwards.items():
            module = self._module_dict.get(module_path)
            if module:
                module.forward = original_forward
        self._original_attn_proj_forwards.clear()

    def clear_relp_caches(self):
        """Clear all RelP-specific caches."""
        self.mlp_activations.clear()
        self.mlp_gradients.clear()
        self._sigmoid_cache.clear()
        self._linearization_ctx.clear()

    def get_mlp_activations(self, layer: int) -> torch.Tensor | None:
        """Get cached MLP activations for a layer."""
        return self.mlp_activations.get(layer)

    def get_mlp_gradients(self, layer: int) -> torch.Tensor | None:
        """Get cached MLP gradients for a layer."""
        return self.mlp_gradients.get(layer)

    def remove_tensor_hooks(self):
        """Remove all tensor-level hooks."""
        for hook in self._tensor_hooks:
            hook.remove()
        self._tensor_hooks.clear()
        self._linearization_ctx.remove_hooks()


# =============================================================================
# RelP Attributor - Main Attribution Class
# =============================================================================

class RelPAttributor:
    """Computes RelP attributions for neuron circuits in transformer models.

    Supported models:
    - Llama 3.1-8B-Instruct
    - Qwen 3 32B
    - OLMo 7B / OLMo 2 7B

    Usage:
        attributor = RelPAttributor(model, tokenizer, model_name="llama-3.1-8b")
        graph = attributor.compute_attributions("What is the capital of France?")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
        config: RelPConfig | None = None,
        model_name: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name

        # Get device from model parameters, fallback to provided device
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = device
        self.config = config or RelPConfig()

        # Look up model architecture config
        self.model_config: ModelArchConfig | None = None
        if model_name:
            self.model_config = get_model_config(model_name)
            if self.model_config:
                log(f"      Using model config: {self.model_config.name}")

        # Populate always_on_neurons from model config if not explicitly set
        if self.config.always_on_neurons is None:
            if self.model_config and self.model_config.always_on_neurons:
                self.config.always_on_neurons = self.model_config.always_on_neurons
                log(f"      Loaded {len(self.config.always_on_neurons)} always-on neurons from model config")
            else:
                # No model config or empty always_on_neurons - use empty list
                self.config.always_on_neurons = []
                if self.config.filter_always_on:
                    log("      Warning: No always-on neurons configured for this model")

        # Determine number of layers
        if hasattr(model, 'config'):
            self.num_layers = getattr(model.config, 'num_hidden_layers', 32)
        else:
            self.num_layers = 32

        # Initialize scope
        if hasattr(model, 'model'):
            # HuggingFace model with inner model
            self.scope = RelPScope(model.model, self.num_layers)
        else:
            self.scope = RelPScope(model, self.num_layers)

        # Set up hooks (with optional linearization)
        self.scope.setup_mlp_hooks(linearize=self.config.linearize)
        if self.config.linearize:
            log("      Linearization enabled (half-rule for SiLU and gated MLP)")

    def compute_attributions(
        self,
        input_text: str,
        k: int | None = None,
        tau: float | None = None,
        compute_edges: bool | None = None,
        filter_always_on: bool | None = None,
        target_tokens: list[str] | None = None,
        contrastive_tokens: list[str] | None = None,
        target_position: int | None = None,
        max_nodes: int | None = 500,
    ) -> dict[str, Any]:
        """Compute RelP attributions and return Neuronpedia-format graph.

        Args:
            input_text: Input prompt to attribute
            k: Number of top logits for metric (default from config)
            tau: Node filtering threshold (default from config)
            compute_edges: Whether to compute edges (default from config)
            filter_always_on: Whether to filter always-on neurons (default from config)
            target_tokens: Specific tokens to trace (e.g., [" Yes"]). Overrides k if set.
            contrastive_tokens: Pair of tokens [positive, negative] for contrastive attribution.
                               Traces logit(positive) - logit(negative). Overrides target_tokens and k.
            target_position: Position in sequence to compute attribution from (default: -1, last position).
                            Use this to trace attribution to predicting the token at position target_position+1.
            max_nodes: Maximum number of nodes before skipping edge computation (default 500).
                       If exceeded, returns nodes_exceeded=True and no edges.
                       Set to None to disable limit.

        Returns:
            Dictionary in Neuronpedia attribution graph format
        """
        # Use config defaults if not specified
        k = k if k is not None else self.config.k
        tau = tau if tau is not None else self.config.tau
        compute_edges = compute_edges if compute_edges is not None else self.config.compute_edges
        filter_always_on = filter_always_on if filter_always_on is not None else self.config.filter_always_on
        target_tokens = target_tokens if target_tokens is not None else self.config.target_tokens
        contrastive_tokens = contrastive_tokens if contrastive_tokens is not None else self.config.contrastive_tokens

        # Tokenize input
        log("[1/6] Tokenizing input...")
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        seq_len = inputs.input_ids.shape[1]
        log(f"      {seq_len} tokens")

        # Clear caches
        self.scope.clear_relp_caches()
        self.model.zero_grad()

        # Forward pass
        log("[2/6] Forward pass...")
        outputs = self.model(**inputs)
        # Use target_position if specified, otherwise last position (-1)
        pos = target_position if target_position is not None else -1
        logits = outputs.logits[:, pos, :]  # Logits at target position
        log(f"      Done (target position: {pos if pos >= 0 else seq_len + pos})")

        # Compute metric: sum of target logits (used for both backprop and threshold)
        # Paper: m(M,x) = Σᵢ₌₁ᵏ [M(x)]ᵢ
        probs = torch.softmax(logits, dim=-1)

        # Helper to resolve token to id
        def resolve_token_id(tok):
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            if tok_id == self.tokenizer.unk_token_id:
                # Try with Ġ prefix (common in BPE tokenizers)
                tok_id = self.tokenizer.convert_tokens_to_ids('Ġ' + tok.strip())
            if tok_id == self.tokenizer.unk_token_id:
                # Try encoding the token directly
                encoded = self.tokenizer.encode(tok, add_special_tokens=False)
                if encoded:
                    tok_id = encoded[0]
            return tok_id

        is_contrastive = False
        if contrastive_tokens:
            # Contrastive attribution: logit(positive) - logit(negative)
            if len(contrastive_tokens) != 2:
                raise ValueError(f"contrastive_tokens must have exactly 2 tokens [positive, negative], got {len(contrastive_tokens)}")

            pos_tok, neg_tok = contrastive_tokens
            pos_id = resolve_token_id(pos_tok)
            neg_id = resolve_token_id(neg_tok)

            if pos_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Positive token '{pos_tok}' not found in vocabulary")
            if neg_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Negative token '{neg_tok}' not found in vocabulary")

            log(f"[3/6] Computing metric (contrastive: {pos_tok} vs {neg_tok})...")
            log(f"      Positive '{pos_tok}' -> id {pos_id}, logit={logits[0, pos_id].item():.4f}, p={probs[0, pos_id].item():.4f}")
            log(f"      Negative '{neg_tok}' -> id {neg_id}, logit={logits[0, neg_id].item():.4f}, p={probs[0, neg_id].item():.4f}")

            # Metric is the difference: logit(pos) - logit(neg)
            metric_value = logits[0, pos_id] - logits[0, neg_id]
            is_contrastive = True

            # For graph output, we track both tokens
            top_k_indices = torch.tensor([[pos_id, neg_id]], device=logits.device)
            top_k_probs = [probs[0, pos_id].item(), probs[0, neg_id].item()]

            log(f"      Contrastive metric (logit diff): {metric_value.item():.4f}")

        elif target_tokens:
            # Use specific target tokens instead of top-k
            log(f"[3/6] Computing metric (target tokens: {target_tokens})...")
            target_indices = []
            for tok in target_tokens:
                tok_id = resolve_token_id(tok)
                if tok_id != self.tokenizer.unk_token_id:
                    target_indices.append(tok_id)
                    log(f"      Token '{tok}' -> id {tok_id}")
                else:
                    log(f"      WARNING: Token '{tok}' not found in vocabulary")

            if not target_indices:
                raise ValueError(f"None of the target tokens {target_tokens} found in vocabulary")

            top_k_indices = torch.tensor([target_indices], device=logits.device)
            top_k_values = logits[0, target_indices]
            metric_value = top_k_values.sum()
            top_k_probs = probs[0, target_indices].detach().tolist()
            log(f"      Metric value: {metric_value.item():.4f}")

        else:
            # Use top-k logits
            log("[3/6] Computing metric (top-k logits)...")
            top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
            metric_value = top_k_values.sum()  # Sum of top-k logits
            top_k_probs = probs[0, top_k_indices[0]].detach().tolist()
            log(f"      Metric value: {metric_value.item():.4f}")

        # Backward pass
        log("[4/6] Backward pass (gradient computation)...")
        metric_value.backward()
        log("      Done")

        # Compute node attributions
        log("[5/6] Computing node attributions...")
        node_attributions = self._compute_node_attributions(seq_len)
        log(f"      Found {len(node_attributions)} non-zero attributions")

        # Filter nodes: V(x) = {v ∈ V : |RelP_v(x)| ≥ τ · m(M,x)}
        # Paper uses sum of top-k logits as m(M,x)
        threshold = tau * abs(metric_value.item())
        filtered_nodes = self._filter_nodes(
            node_attributions,
            threshold,
            filter_always_on
        )
        log(f"      Filtered to {len(filtered_nodes)} nodes (threshold={threshold:.6f})")

        # Check if we exceeded max_nodes - if so, skip edge computation
        nodes_exceeded = max_nodes is not None and len(filtered_nodes) > max_nodes
        if nodes_exceeded:
            log(f"      WARNING: Node count ({len(filtered_nodes)}) exceeds max_nodes ({max_nodes}), skipping edge computation")

        # Compute edge attributions (only between filtered nodes)
        log("[6/6] Computing edge attributions...")
        edge_attributions = []
        if compute_edges and len(filtered_nodes) > 0 and not nodes_exceeded:
            if self.config.use_jacobian_edges:
                edge_attributions = self._compute_jacobian_edges(
                    filtered_nodes, tokens, inputs, seq_len
                )
            else:
                edge_attributions = self._compute_edge_attributions(
                    filtered_nodes, tokens, inputs
                )
        log(f"      Created {len(edge_attributions)} edges")

        # Build Neuronpedia graph
        graph = self._build_neuronpedia_graph(
            input_text=input_text,
            tokens=tokens,
            nodes=filtered_nodes,
            edges=edge_attributions,
            metric_value=metric_value.item(),
            top_k_indices=top_k_indices[0].tolist(),
            top_k_probs=top_k_probs,
            k=k,
            tau=tau,
            nodes_exceeded=nodes_exceeded,
        )

        return graph

    def _compute_node_attributions(self, seq_len: int) -> list[NodeAttribution]:
        """Compute RelP scores for all neurons.

        Optimized with vectorized operations - collects top-k per layer.
        """
        attributions = []
        # Keep only top-k neurons per layer to avoid processing millions
        top_k_per_layer = 500

        for layer_idx in tqdm(range(self.num_layers), desc="      Layers", leave=False):
            acts = self.scope.get_mlp_activations(layer_idx)
            grads = self.scope.get_mlp_gradients(layer_idx)

            if acts is None or grads is None:
                continue

            # acts shape: [batch, seq, d_ffn] where d_ffn=14336 for Llama 3.1-8B
            # Compute RelP: activation * gradient (vectorized)
            relp_scores = (acts * grads)[0]  # [seq, d_ffn]

            # Flatten and get top-k by absolute value
            flat_scores = relp_scores[:seq_len].flatten()  # [seq_len * d_ffn]
            flat_abs = torch.abs(flat_scores)

            # Get top-k indices
            k = min(top_k_per_layer, flat_abs.numel())
            _, top_indices = torch.topk(flat_abs, k)

            # Convert flat indices back to (position, neuron_idx)
            d_ffn = relp_scores.shape[1]
            positions = top_indices // d_ffn
            neuron_indices = top_indices % d_ffn

            # Extract values in batch
            scores = flat_scores[top_indices]
            acts_flat = acts[0, :seq_len].flatten()
            grads_flat = grads[0, :seq_len].flatten()
            activations = acts_flat[top_indices]
            gradients = grads_flat[top_indices]

            # Create attributions (this is fast since k is small)
            for i in range(k):
                score = scores[i].item()
                if abs(score) < 1e-10:
                    continue
                attributions.append(NodeAttribution(
                    layer=layer_idx,
                    neuron_idx=neuron_indices[i].item(),
                    position=positions[i].item(),
                    activation=activations[i].item(),
                    gradient=gradients[i].item(),
                    relp_score=score
                ))

        return attributions

    def _filter_nodes(
        self,
        attributions: list[NodeAttribution],
        threshold: float,
        filter_always_on: bool
    ) -> list[NodeAttribution]:
        """Filter nodes by threshold and always-on list."""
        filtered = []
        always_on_set = set(self.config.always_on_neurons) if filter_always_on else set()

        for node in attributions:
            # Check threshold
            if abs(node.relp_score) < threshold:
                continue

            # Check always-on filter
            if (node.layer, node.neuron_idx) in always_on_set:
                continue

            filtered.append(node)

        # Sort by absolute RelP score (descending)
        filtered.sort(key=lambda n: abs(n.relp_score), reverse=True)

        return filtered

    def _compute_edge_attributions(
        self,
        nodes: list[NodeAttribution],
        tokens: list[str],
        _inputs: dict[str, torch.Tensor]  # Reserved for future Jacobian computation
    ) -> list[tuple[str, str, float]]:
        """Compute edge attributions between filtered nodes.

        Creates edges connecting:
        1. Embeddings to first-layer neurons (at same position)
        2. Neurons in adjacent layers (approximated weights)
        3. Last-layer neurons to logits (added separately in _build_neuronpedia_graph)
        """
        edges = []

        # Group nodes by layer
        nodes_by_layer: dict[int, list[NodeAttribution]] = {}
        for node in nodes:
            if node.layer not in nodes_by_layer:
                nodes_by_layer[node.layer] = []
            nodes_by_layer[node.layer].append(node)

        sorted_layers = sorted(nodes_by_layer.keys())
        if not sorted_layers:
            return edges

        # 1. Connect embeddings to earliest layer neurons
        first_layer = sorted_layers[0]
        for node in nodes_by_layer[first_layer]:
            # Get the token at this position
            if node.position < len(tokens):
                token = tokens[node.position]
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                emb_node_id = f"E_{token_id}_{node.position}"
                # Weight: use the neuron's RelP score (how much this neuron contributes)
                weight = node.relp_score
                edges.append((emb_node_id, node.node_id, weight))

        # 2. Connect neurons between adjacent layers
        # For efficiency, only connect to the next layer with neurons (not all pairs)
        for i, source_layer in enumerate(sorted_layers[:-1]):
            target_layer = sorted_layers[i + 1]

            for source in nodes_by_layer[source_layer]:
                for target in nodes_by_layer[target_layer]:
                    # Only connect nodes at same or later positions
                    if target.position >= source.position:
                        # Approximate edge weight using geometric mean of RelP scores
                        # This captures that both nodes are important in the circuit
                        weight = (source.relp_score * target.relp_score) ** 0.5
                        # Preserve sign based on whether contributions align
                        if (source.relp_score < 0) != (target.relp_score < 0):
                            weight = -abs(weight)

                        if abs(weight) > 1e-10:
                            edges.append((
                                source.node_id,
                                target.node_id,
                                weight
                            ))

        return edges

    def _compute_jacobian_edges(
        self,
        nodes: list[NodeAttribution],
        tokens: list[str],
        inputs: dict[str, torch.Tensor],
        seq_len: int
    ) -> list[tuple[str, str, float]]:
        """Compute edge attributions using Jacobian-based method.

        The paper defines edge attribution as:
            RelP_{s→t}(x) = v_s(x) * ∂v_t/∂v_s

        And attribution flow (normalized edge weight):
            Flow_{s→t} = RelP_{s→t}(x) / v_t(x) * RelP_{v_t}(x)

        This requires computing the gradient of target activations w.r.t. source activations.
        We do this by:
        1. Fresh forward pass with gradient-tracked activations
        2. For each target node, backward to get gradients at source nodes
        3. Edge weight = source_activation * gradient
        4. Optionally normalize to attribution flow

        When edge_stop_grad_mlps=True (default), we apply stop-gradients on intermediate
        MLP layers to compute "direct effects". When False, we compute "total effects"
        including all paths through intermediate layers.

        Args:
            nodes: Filtered nodes from node attribution
            tokens: Input tokens
            inputs: Tokenized inputs
            seq_len: Sequence length

        Returns:
            List of (source_id, target_id, weight) tuples
        """
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
        log(f"      Computing Jacobian edges between {len(sorted_layers)} layers ({method_str})...")

        # For Jacobian computation, we need to capture activations WITHOUT detaching
        # Store live (non-detached) activation tensors during forward pass
        live_activations: dict[int, torch.Tensor] = {}

        # Set up hooks to capture live activations for ALL layers (not just sorted_layers)
        # This is needed for stop-gradient computation on intermediate layers
        hooks = []
        all_layers_with_nodes = set(sorted_layers)

        # If using stop-grads, we need to track ALL intermediate layers
        if use_stop_grads:
            min_layer = min(sorted_layers)
            max_layer = max(sorted_layers)
            layers_to_hook = list(range(min_layer, max_layer + 1))
        else:
            layers_to_hook = sorted_layers

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
                            # Keep the live tensor (not detached!)
                            live_activations[l_idx] = inp[0]
                    return hook

                h = module.register_forward_hook(make_hook(layer_idx))
                hooks.append(h)

        # Fresh forward pass to capture live activations
        self.model.zero_grad()
        with torch.enable_grad():
            outputs = self.model(**inputs)

        # Remove temporary hooks
        for h in hooks:
            h.remove()

        # Build a lookup from node_id to NodeAttribution for flow normalization
        node_lookup = {node.node_id: node for node in nodes}

        # Now compute edges between all layer pairs (not just adjacent)
        # This allows skip connections across multiple layers as in the paper
        edge_count = 0

        # Build all (source_layer, target_layer) pairs where source < target
        layer_pairs = []
        for i, source_layer in enumerate(sorted_layers):
            for target_layer in sorted_layers[i + 1:]:
                layer_pairs.append((source_layer, target_layer))

        log(f"      Computing Jacobian edges for {len(layer_pairs)} layer pairs...")

        for source_layer, target_layer in tqdm(layer_pairs, desc="      Layer pairs", leave=False):
            source_nodes = nodes_by_layer[source_layer]
            target_nodes = nodes_by_layer[target_layer]

            # Get live activations (with gradients attached)
            source_acts = live_activations.get(source_layer)
            target_acts = live_activations.get(target_layer)

            if source_acts is None or target_acts is None:
                continue

            # Identify intermediate layers for stop-gradient
            intermediate_layers = [l for l in range(source_layer + 1, target_layer)
                                   if l in live_activations]

            # For each target node, compute gradient w.r.t source layer
            for target in target_nodes:
                # Get the specific target activation value
                target_val = target_acts[0, target.position, target.neuron_idx]

                if not target_val.requires_grad:
                    continue

                # Compute gradient of this target w.r.t. source layer activations
                try:
                    if use_stop_grads and intermediate_layers:
                        # Apply stop-gradients on intermediate MLP activations
                        # We compute gradients while blocking flow through intermediate layers
                        # This gives "direct effect" of source on target
                        grad_tensor = self._compute_grad_with_stop_grads(
                            target_val,
                            source_acts,
                            [live_activations[l] for l in intermediate_layers]
                        )
                        if grad_tensor is None:
                            continue
                    else:
                        # Standard gradient computation (total effect)
                        grads = torch.autograd.grad(
                            target_val,
                            source_acts,
                            retain_graph=True,
                            allow_unused=True
                        )

                        if grads[0] is None:
                            continue

                        grad_tensor = grads[0][0]  # [seq, d_ffn]

                    # For each source node, compute edge weight
                    for source in source_nodes:
                        # Only connect if position allows information flow
                        if target.position < source.position:
                            continue

                        # Get gradient at source location
                        grad_at_source = grad_tensor[source.position, source.neuron_idx].item()

                        # Edge weight: v_s * ∂v_t/∂v_s
                        edge_weight = source.activation * grad_at_source

                        # Apply attribution flow normalization if enabled
                        # Flow_{s→t} = RelP_{s→t}(x) / v_t(x) * RelP_{v_t}(x)
                        if normalize_flow and abs(target.activation) > 1e-10:
                            edge_weight = (edge_weight / target.activation) * target.relp_score

                        if abs(edge_weight) > 1e-10:
                            edges.append((
                                source.node_id,
                                target.node_id,
                                edge_weight
                            ))
                            edge_count += 1

                except RuntimeError as e:
                    # Gradient computation can fail for disconnected layers
                    if "does not require grad" in str(e) or "One of the differentiated" in str(e):
                        continue
                    raise

        log(f"      Computed {edge_count} Jacobian edges")

        # Add embedding to first-layer edges
        first_layer = sorted_layers[0]
        for node in nodes_by_layer[first_layer]:
            if node.position < len(tokens):
                token = tokens[node.position]
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                emb_node_id = f"E_{token_id}_{node.position}"
                # For embedding edges, use the node's RelP score (already normalized)
                edges.append((emb_node_id, node.node_id, node.relp_score))

        return edges

    def _compute_grad_with_stop_grads(
        self,
        target_val: torch.Tensor,
        source_acts: torch.Tensor,
        intermediate_acts: list[torch.Tensor]
    ) -> torch.Tensor | None:
        """Compute gradient with stop-gradients on intermediate layers.

        This implements the "direct effect" edge computation by blocking gradient
        flow through intermediate MLP activations. The gradient only flows through
        the residual stream and attention, not through intermediate MLP computations.

        Args:
            target_val: Scalar target activation to differentiate
            source_acts: Source layer activations [batch, seq, d_ffn]
            intermediate_acts: List of intermediate layer activations to block

        Returns:
            Gradient tensor [seq, d_ffn] or None if computation fails
        """
        # We use a technique where we compute the gradient but zero out
        # contributions from intermediate layers by using backward hooks

        # Register hooks to zero out gradients at intermediate layers
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

            return grads[0][0]  # [seq, d_ffn]

        finally:
            # Always remove the hooks
            for h in stop_hooks:
                h.remove()

    def _build_neuronpedia_graph(
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
        nodes_exceeded: bool = False,
    ) -> dict[str, Any]:
        """Build Neuronpedia-compatible graph dictionary."""

        # Fetch neuron labels if available and enabled
        neuron_labels: dict[tuple[int, int], str] = {}
        if self.config.use_neuron_labels and LABELS_AVAILABLE and is_database_available():
            try:
                neuron_infos = [
                    NeuronInfo(layer=n.layer, neuron=n.neuron_idx, node_id=f"{n.layer}_{n.neuron_idx}_{n.position}")
                    for n in nodes
                ]
                desc_map = fetch_descriptions(neuron_infos)
                # Convert node_id-keyed map to (layer, neuron)-keyed map
                for n in nodes:
                    nid = f"{n.layer}_{n.neuron_idx}_{n.position}"
                    if nid in desc_map:
                        neuron_labels[(n.layer, n.neuron_idx)] = desc_map[nid]
            except Exception as e:
                print(f"Warning: Could not fetch neuron labels: {e}")

        # Build node list
        graph_nodes = []

        # Add embedding nodes
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

        # Add MLP neuron nodes with labels
        for node in nodes:
            node_dict = node.to_neuronpedia_node()
            # Override clerp with label if available
            label_key = (node.layer, node.neuron_idx)
            if label_key in neuron_labels:
                label = neuron_labels[label_key]
                # Truncate long labels
                if len(label) > 60:
                    label = label[:57] + "..."
                node_dict["clerp"] = f"L{node.layer}/N{node.neuron_idx}: {label}"
            graph_nodes.append(node_dict)

        # Add logit nodes for top-k predictions
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

        # Build link list with raw edge weights (viewer computes percentages)
        graph_links = []
        for src, tgt, weight in edges:
            graph_links.append({"source": src, "target": tgt, "weight": weight})

        # Add edges from ALL neurons to logits (not just last layer)
        # Edge weight = node's relp_score weighted by logit's probability share
        for node in nodes:
            for i, logit_idx in enumerate(top_k_indices):
                logit_node_id = f"L_{logit_idx}_{len(tokens)-1}"
                # Weight by probability share among top-k
                prob_weight = top_k_probs[i] / sum(top_k_probs) if sum(top_k_probs) > 0 else 1.0 / len(top_k_probs)
                edge_weight = node.relp_score * prob_weight

                if abs(edge_weight) > 1e-6:
                    graph_links.append({
                        "source": node.node_id,
                        "target": logit_node_id,
                        "weight": edge_weight
                    })

        # Build metadata
        actual_position = pos if pos >= 0 else seq_len + pos
        metadata = {
            "slug": f"relp-{hash(input_text) % 100000}",
            "scan": "llama-3.1-8b-instruct",
            "prompt_tokens": tokens,
            "prompt": input_text,
            "node_threshold": tau,
            "target_position": actual_position,
            "info": {
                "description": "RelP attribution graph for neuron circuits",
                "generator": {
                    "name": "RelP Attributor",
                    "version": "1.0.0"
                }
            },
            "generation_settings": {
                "max_n_logits": k,
                "node_threshold": tau,
                "metric_value": metric_value,
                "target_position": actual_position,
                "nodes_exceeded": nodes_exceeded,
            }
        }

        return {
            "metadata": metadata,
            "qParams": {},
            "nodes": graph_nodes,
            "links": graph_links,
            "nodes_exceeded": nodes_exceeded,  # Also at top level for easy access
        }

    def cleanup(self):
        """Remove all hooks and clean up."""
        self.scope.remove_tensor_hooks()
        self.scope.remove_all_hooks()
        self.scope.clear_relp_caches()
        # Restore original MLP forward methods if linearization was used
        self.scope.restore_original_forwards()


# =============================================================================
# Convenience Functions
# =============================================================================

def attribute(
    model: nn.Module,
    tokenizer,
    input_text: str,
    device: str = "cuda",
    **kwargs
) -> dict[str, Any]:
    """Convenience function for one-off attribution.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        input_text: Text to attribute
        device: Device to run on
        **kwargs: Additional arguments passed to compute_attributions

    Returns:
        Neuronpedia-format attribution graph
    """
    attributor = RelPAttributor(model, tokenizer, device)
    try:
        return attributor.compute_attributions(input_text, **kwargs)
    finally:
        attributor.cleanup()


def save_graph(graph: dict[str, Any], path: str):
    """Save attribution graph to JSON file."""
    import json
    with open(path, 'w') as f:
        json.dump(graph, f, indent=2)


def validate_graph(graph: dict[str, Any]) -> bool:
    """Basic validation of graph structure.

    For full validation, use https://www.neuronpedia.org/graph/validator
    """
    required_keys = ["metadata", "qParams", "nodes", "links"]
    if not all(k in graph for k in required_keys):
        return False

    metadata_required = ["slug", "scan", "prompt_tokens", "prompt"]
    if not all(k in graph["metadata"] for k in metadata_required):
        return False

    node_required = ["node_id", "feature", "layer", "ctx_idx", "feature_type", "jsNodeId", "clerp"]
    for node in graph["nodes"]:
        if not all(k in node for k in node_required):
            return False

    link_required = ["source", "target", "weight"]
    for link in graph["links"]:
        if not all(k in link for k in link_required):
            return False

    return True
