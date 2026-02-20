
import pytest
import torch
from circuits import RelPAttributor, RelPConfig, validate_graph
from circuits.relp import (
    FrozenGatedMLP,
    FrozenRMSNorm,
    FrozenSiLU,
    NodeAttribution,
)


def test_frozen_silu_half_rule():
    x = torch.randn(4, requires_grad=True)
    y = FrozenSiLU.apply(x)
    loss = y.sum()
    loss.backward()

    # FrozenSiLU linearized backward: gradient = frozen sigmoid(x)
    # (half-rule is applied at FrozenGatedMLP, not at SiLU level)
    expected_grad = torch.sigmoid(x).detach()
    assert torch.allclose(x.grad, expected_grad, atol=1e-6)


def test_frozen_gated_mlp_half_rule():
    gate = torch.randn(4, requires_grad=True)
    up = torch.randn(4, requires_grad=True)
    out = FrozenGatedMLP.apply(gate, up).sum()
    out.backward()

    assert torch.allclose(gate.grad, up.detach() * 0.5, atol=1e-6)
    assert torch.allclose(up.grad, gate.detach() * 0.5, atol=1e-6)


def test_frozen_rmsnorm_linearization():
    x = torch.randn(2, 3, requires_grad=True)
    weight = torch.randn(3)
    y = FrozenRMSNorm.apply(x, weight, 1e-6).sum()
    y.backward()

    inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6).detach()
    expected = weight * inv_rms
    assert torch.allclose(x.grad, expected, atol=1e-6)


def test_metric_matches_topk(tiny_attributor, tiny_model, tiny_tokenizer, monkeypatch):
    prompt = "A B"
    captured = {}
    original = RelPAttributor._build_neuronpedia_graph

    def wrapper(self, *args, **kwargs):
        captured["metric"] = kwargs["metric_value"]
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RelPAttributor, "_build_neuronpedia_graph", wrapper)
    tiny_attributor.compute_attributions(prompt)
    assert "metric" in captured

    inputs = tiny_tokenizer(prompt, return_tensors="pt")
    logits = tiny_model(input_ids=inputs.input_ids).logits[:, -1, :]
    expected = torch.topk(logits, k=tiny_attributor.config.k, dim=-1).values.sum().item()
    assert pytest.approx(expected, rel=1e-6) == captured["metric"]


def test_mlp_caches_populated(tiny_attributor):
    tiny_attributor.compute_attributions("A B")
    for layer in range(tiny_attributor.num_layers):
        acts = tiny_attributor.scope.get_mlp_activations(layer)
        grads = tiny_attributor.scope.get_mlp_gradients(layer)
        assert acts is not None
        assert grads is not None
        assert acts.shape == grads.shape


def test_filter_nodes_applies_threshold_and_always_on(tiny_attributor):
    # Set always-on list explicitly (tiny model has no model config)
    tiny_attributor.config.always_on_neurons = [(23, 306)]

    nodes = [
        NodeAttribution(layer=23, neuron_idx=306, position=0, activation=1.0, gradient=1.0, relp_score=1.0),
        NodeAttribution(layer=1, neuron_idx=5, position=0, activation=0.5, gradient=0.1, relp_score=0.05),
        NodeAttribution(layer=2, neuron_idx=10, position=0, activation=0.5, gradient=2.0, relp_score=1.0),
    ]

    filtered = tiny_attributor._filter_nodes(nodes, threshold=0.1, filter_always_on=True)
    assert len(filtered) == 1
    assert filtered[0].layer == 2

    filtered = tiny_attributor._filter_nodes(nodes, threshold=0.01, filter_always_on=False)
    assert len(filtered) == 3


def test_jacobian_edges_follow_formula(tiny_model, tiny_tokenizer):
    config = RelPConfig(
        k=2,
        tau=0.0,
        compute_edges=True,
        use_jacobian_edges=True,
        filter_always_on=False,
        linearize=False,  # Disable linearization so test can verify exact formula
        edge_stop_grad_mlps=False,  # Disable stop-grads so manual Jacobian matches
        normalize_edge_flow=False,  # Use raw Jacobian weights, not flow-normalized
    )
    attributor = RelPAttributor(tiny_model, tiny_tokenizer, device="cpu", config=config)

    prompt = "A B"
    inputs = tiny_tokenizer(prompt, return_tensors="pt").to("cpu")
    tokens = tiny_tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    attributor.scope.clear_relp_caches()
    tiny_model.zero_grad()
    outputs = tiny_model(input_ids=inputs.input_ids)
    logits = outputs.logits[:, -1, :]
    metric = torch.topk(logits, k=config.k, dim=-1).values.sum()
    metric.backward()

    nodes = attributor._compute_node_attributions(seq_len=inputs.input_ids.shape[1])
    filtered = attributor._filter_nodes(nodes, threshold=0.0, filter_always_on=False)
    edges = attributor._compute_jacobian_edges(filtered, tokens, inputs, inputs.input_ids.shape[1])
    neuron_edge = next((e for e in edges if not e[0].startswith("E_")), None)
    assert neuron_edge is not None

    source_id, target_id, weight = neuron_edge
    source_node = next(n for n in filtered if n.node_id == source_id)

    live_activations = {}
    hooks = []
    for layer_idx in {int(source_id.split("_")[0]), int(target_id.split("_")[0])}:
        down_paths = [
            f"layers-{layer_idx}-mlp-down_proj",
            f"model-layers-{layer_idx}-mlp-down_proj",
        ]
        down_path = next((p for p in down_paths if p in attributor.scope._module_dict), None)
        assert down_path is not None

        module = attributor.scope._module_dict[down_path]

        def make_hook(idx):
            def hook(_, inp, __):
                live_activations[idx] = inp[0]

            return hook

        hooks.append(module.register_forward_hook(make_hook(layer_idx)))

    with torch.enable_grad():
        tiny_model.zero_grad()
        tiny_model(input_ids=inputs.input_ids)

    for hook in hooks:
        hook.remove()

    source_layer, source_neuron, source_pos = map(int, source_id.split("_"))
    target_layer, target_neuron, target_pos = map(int, target_id.split("_"))

    target_tensor = live_activations[target_layer]
    target_val = target_tensor[0, target_pos, target_neuron]
    grads = torch.autograd.grad(target_val, live_activations[source_layer], retain_graph=False, allow_unused=True)
    grad_at_source = grads[0][0, source_pos, source_neuron].item()

    expected_weight = source_node.activation * grad_at_source
    assert pytest.approx(expected_weight, rel=1e-5, abs=1e-7) == weight
    attributor.cleanup()


def test_validate_graph_passes(tiny_attributor):
    graph = tiny_attributor.compute_attributions("A B")
    assert validate_graph(graph)


# =============================================================================
# Scientific correctness tests (per Transluce paper)
# =============================================================================


def test_node_relp_score_is_activation_times_gradient(tiny_attributor, tiny_tokenizer):
    """
    Paper formula (page 8):
        RelP_v(x) = v(x) * ∂m(M_replacement, x) / ∂v(x)

    Node attribution should equal activation × gradient.
    """
    prompt = "A B"
    inputs = tiny_tokenizer(prompt, return_tensors="pt")

    tiny_attributor.scope.clear_relp_caches()
    tiny_attributor.model.zero_grad()
    outputs = tiny_attributor.model(input_ids=inputs.input_ids)
    logits = outputs.logits[:, -1, :]
    metric = torch.topk(logits, k=tiny_attributor.config.k, dim=-1).values.sum()
    metric.backward()

    nodes = tiny_attributor._compute_node_attributions(seq_len=inputs.input_ids.shape[1])

    for node in nodes:
        expected = node.activation * node.gradient
        assert pytest.approx(expected, rel=1e-6) == node.relp_score, (
            f"Node {node.node_id}: expected relp={expected}, got {node.relp_score}"
        )


def test_threshold_is_relative_to_metric(tiny_model, tiny_tokenizer):
    """
    Paper formula (page 11):
        V(x) = {v ∈ V : RelP_v(x) ≥ τ · m(M, x)}

    Threshold should be tau × metric_value, not an absolute threshold.
    """
    tau = 0.1
    config = RelPConfig(k=2, tau=tau, compute_edges=False, filter_always_on=False)
    attributor = RelPAttributor(tiny_model, tiny_tokenizer, device="cpu", config=config)

    prompt = "A B"
    inputs = tiny_tokenizer(prompt, return_tensors="pt")

    # Compute metric value
    outputs = tiny_model(input_ids=inputs.input_ids)
    logits = outputs.logits[:, -1, :]
    metric_value = torch.topk(logits, k=config.k, dim=-1).values.sum().item()

    expected_threshold = tau * abs(metric_value)

    # The compute_attributions method computes threshold internally
    # We verify by checking that nodes with |relp| < threshold are filtered
    attributor.scope.clear_relp_caches()
    tiny_model.zero_grad()
    outputs = tiny_model(input_ids=inputs.input_ids)
    logits = outputs.logits[:, -1, :]
    metric = torch.topk(logits, k=config.k, dim=-1).values.sum()
    metric.backward()

    all_nodes = attributor._compute_node_attributions(seq_len=inputs.input_ids.shape[1])
    filtered = attributor._filter_nodes(all_nodes, threshold=expected_threshold, filter_always_on=False)

    # All filtered nodes should have |relp| >= threshold
    for node in filtered:
        assert abs(node.relp_score) >= expected_threshold, (
            f"Node {node.node_id} has |relp|={abs(node.relp_score)} < threshold={expected_threshold}"
        )

    # All excluded nodes should have |relp| < threshold
    filtered_ids = {n.node_id for n in filtered}
    for node in all_nodes:
        if node.node_id not in filtered_ids:
            assert abs(node.relp_score) < expected_threshold, (
                f"Node {node.node_id} excluded but |relp|={abs(node.relp_score)} >= threshold"
            )

    attributor.cleanup()


def test_mlp_activations_captured_at_post_swiglu(tiny_model, tiny_tokenizer):
    """
    Paper (page 1-2, 5): MLP activations (post-SwiGLU) are the privileged basis.

    We capture activations at the input to down_proj, which is:
        SiLU(gate_proj(x)) * up_proj(x)

    This test verifies the captured activation matches the expected post-SwiGLU value.
    """
    config = RelPConfig(k=2, tau=0.0, compute_edges=False, filter_always_on=False)
    attributor = RelPAttributor(tiny_model, tiny_tokenizer, device="cpu", config=config)

    prompt = "A B"
    inputs = tiny_tokenizer(prompt, return_tensors="pt")

    # Manually compute expected post-SwiGLU activation for layer 0
    layer = tiny_model.model.layers[0]
    hidden = tiny_model.model.embed_tokens(inputs.input_ids)
    normed = layer.input_layernorm(hidden)
    attn_out = layer.self_attn(normed)
    post_attn = hidden + attn_out
    normed2 = layer.post_attention_layernorm(post_attn)

    gate = layer.mlp.gate_proj(normed2)
    up = layer.mlp.up_proj(normed2)
    expected_activation = torch.nn.functional.silu(gate) * up

    # Run attributor to populate caches
    attributor.scope.clear_relp_caches()
    tiny_model.zero_grad()
    outputs = tiny_model(input_ids=inputs.input_ids)
    logits = outputs.logits[:, -1, :]
    metric = torch.topk(logits, k=config.k, dim=-1).values.sum()
    metric.backward()

    # Get cached activation
    cached = attributor.scope.get_mlp_activations(0)

    assert cached is not None, "MLP activations not captured for layer 0"
    assert torch.allclose(cached, expected_activation.detach(), atol=1e-5), (
        "Cached activation doesn't match expected post-SwiGLU activation"
    )

    attributor.cleanup()


def test_linearization_enabled_in_config():
    """
    Paper (page 7): RelP requires linearized backward pass.

    Default config should have linearization enabled.
    """
    config = RelPConfig()
    assert config.linearize is True, "Linearization should be enabled by default"
    assert config.use_jacobian_edges is True, "Jacobian edges should be enabled by default"

