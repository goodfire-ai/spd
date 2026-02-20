from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

# Import circuits only if available (some tests don't need it)
try:
    from circuits import RelPAttributor, RelPConfig
    CIRCUITS_AVAILABLE = True
except ImportError:
    CIRCUITS_AVAILABLE = False
    RelPAttributor = None
    RelPConfig = None


class TokenizerOutput(dict):
    """Minimal dict-like tokenizer output compatible with HF BatchEncoding."""

    def __init__(self, input_ids: torch.Tensor):
        super().__init__({"input_ids": input_ids})
        self.input_ids = input_ids

    def to(self, device: str):
        self["input_ids"] = self["input_ids"].to(device)
        self.input_ids = self["input_ids"]
        return self


class TinyTokenizer:
    """Whitespace tokenizer with a fixed vocab for tests."""

    def __init__(self):
        self.vocab = {"A": 0, "B": 1, "C": 2}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

    def __call__(self, text: str, return_tensors: str = "pt"):
        tokens = [self.vocab[token] for token in text.split()]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        return TokenizerOutput(input_ids)

    def convert_ids_to_tokens(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return [self.id_to_token[i] for i in ids]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab[token]

    def decode(self, ids) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self.id_to_token[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Minimal compatibility for generate_graph (not used in tests but keeps API parity)
        return messages[-1]["content"]


class TinyRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.variance_epsilon)
        return normed * self.weight


class TinyMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TinySelfAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = scores.softmax(dim=-1)
        out = torch.matmul(attn, v)
        return self.o_proj(out)


class TinyDecoderLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(dim)
        self.self_attn = TinySelfAttention(dim)
        self.post_attention_layernorm = TinyRMSNorm(dim)
        self.mlp = TinyMLP(dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class TinyBackbone(nn.Module):
    def __init__(self, vocab_size: int, dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [TinyDecoderLayer(dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = TinyRMSNorm(dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 3, dim: int = 4, hidden_dim: int = 8, num_layers: int = 2):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=num_layers)
        self.model = TinyBackbone(vocab_size, dim, hidden_dim, num_layers)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(logits=logits)


@pytest.fixture
def tiny_tokenizer():
    return TinyTokenizer()


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return TinyCausalLM()


@pytest.fixture
def tiny_config():
    return RelPConfig(k=2, tau=0.0, compute_edges=False, use_jacobian_edges=False, filter_always_on=False)


@pytest.fixture
def tiny_attributor(tiny_model, tiny_tokenizer, tiny_config):
    attributor = RelPAttributor(tiny_model, tiny_tokenizer, device="cpu", config=tiny_config)
    yield attributor
    attributor.cleanup()


# =============================================================================
# Neuron Scientist / Investigation Fixtures
# =============================================================================

@pytest.fixture
def sample_investigation_dict():
    """A valid investigation dictionary for testing."""
    return {
        "neuron_id": "L4/N10555",
        "layer": 4,
        "neuron_idx": 10555,
        "timestamp": "2026-01-19T00:00:00",
        "total_experiments": 45,
        "confidence": 0.75,
        "initial_label": "Neurotransmitter detector",
        "initial_hypothesis": "Activates on brain chemistry mentions",
        "characterization": {
            "input_function": "Activates on neurotransmitter mentions",
            "output_function": "Promotes brain chemistry tokens",
            "function_type": "semantic",
            "final_hypothesis": "Neurotransmitter detector",
        },
        "evidence": {
            "activating_prompts": [
                {"prompt": "Dopamine is important", "activation": 1.5, "position": 3, "token": "amine"},
                {"prompt": "Serotonin regulates mood", "activation": 2.0, "position": 2, "token": "onin"},
            ],
            "non_activating_prompts": [
                {"prompt": "Weather is nice", "activation": 0.1, "position": 2, "token": "is"},
            ],
            "ablation_effects": [
                {"prompt": "test", "promotes": [("dopamine", 0.5)], "suppresses": [("random", -0.2)]},
            ],
            "connectivity": {
                "upstream_neurons": [{"neuron_id": "L3/N9778", "label": "Technical terms", "weight": 0.15}],
                "downstream_neurons": [{"neuron_id": "L15/N7890", "label": "Reward", "weight": 0.35}],
            },
            "relp_results": [
                {"prompt": "test", "neuron_found": True, "tau": 0.01, "edges": []},
            ],
        },
        "relp_results": [
            {"prompt": "test", "neuron_found": True, "tau": 0.01, "edges": []},
        ],
        "hypotheses_tested": [
            {
                "hypothesis_id": "H1",
                "hypothesis": "Activates on neurotransmitters",
                "status": "confirmed",
                "posterior_probability": 0.85,
            }
        ],
        "key_findings": [
            "Strong activation on dopamine and serotonin prompts",
            "Z-score of 3.5 vs baseline",
        ],
        "open_questions": [
            "Does it distinguish between different neurotransmitters?",
        ],
        "agent_reasoning": "Detailed reasoning about the investigation...",
    }


@pytest.fixture
def protocol_state_factory():
    """Factory for creating protocol states with specific configurations."""
    from neuron_scientist.tools import ProtocolState

    def create(**kwargs):
        defaults = {
            "phase0_corpus_queried": False,
            "phase0_graph_count": 0,
            "baseline_comparison_done": False,
            "baseline_zscore": None,
            "dose_response_done": False,
            "dose_response_monotonic": False,
            "relp_runs": 0,
            "relp_positive_control": False,
            "relp_negative_control": False,
            "hypotheses_registered": 0,
            "hypotheses_updated": 0,
        }
        defaults.update(kwargs)
        return ProtocolState(**defaults)

    return create


@pytest.fixture
def complete_protocol_state(protocol_state_factory):
    """A protocol state with all validation complete."""
    return protocol_state_factory(
        phase0_corpus_queried=True,
        phase0_graph_count=10,
        baseline_comparison_done=True,
        baseline_zscore=3.5,
        dose_response_done=True,
        dose_response_monotonic=True,
        dose_response_kendall_tau=0.8,
        relp_runs=5,
        relp_positive_control=True,
        relp_negative_control=True,
        hypotheses_registered=3,
        hypotheses_updated=3,
    )


@pytest.fixture
def incomplete_protocol_state(protocol_state_factory):
    """A protocol state with minimal validation."""
    return protocol_state_factory(
        baseline_comparison_done=True,
        baseline_zscore=2.0,
    )
