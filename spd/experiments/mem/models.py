"""MemTransformer: A single-block LLaMA-style transformer for memorization tasks."""

from dataclasses import dataclass
from pathlib import Path
from typing import override

import torch
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F
from wandb.apis.public import Run

from spd.experiments.mem.configs import MemModelConfig, MemTaskConfig, MemTrainConfig
from spd.interfaces import LoadableModule, RunInfo
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.run_utils import check_run_exists
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


@dataclass
class MemTargetRunInfo(RunInfo[MemTrainConfig]):
    """Run info from training a MemTransformer."""

    n_facts: int

    @override
    @classmethod
    def from_path(cls, path: ModelPath) -> "MemTargetRunInfo":
        """Load the run info from a wandb run or a local path to a checkpoint."""
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            # Check if run exists in shared filesystem first
            run_dir = check_run_exists(path)
            if run_dir:
                # Use local files from shared filesystem
                mem_train_config_path = run_dir / "mem_train_config.yaml"
                checkpoint_path = run_dir / "mem.pth"
            else:
                # Download from wandb
                wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
                mem_train_config_path, checkpoint_path = MemTransformer._download_wandb_files(
                    wandb_path
                )
        else:
            # `path` should be a local path to a checkpoint
            mem_train_config_path = Path(path).parent / "mem_train_config.yaml"
            checkpoint_path = Path(path)

        with open(mem_train_config_path) as f:
            mem_train_config_dict = yaml.safe_load(f)

        mem_train_config = MemTrainConfig(**mem_train_config_dict)
        return cls(
            checkpoint_path=checkpoint_path,
            config=mem_train_config,
            n_facts=mem_train_config.n_facts,
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA-style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @override
    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:  # noqa: F821
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for LLaMA-style attention."""

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for all positions
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        self.cos_cached: Tensor
        self.sin_cached: Tensor

    @override
    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Float[Tensor, "B H S D"],  # noqa: F821
    k: Float[Tensor, "B H S D"],  # noqa: F821
    cos: Float[Tensor, "S D"],  # noqa: F821
    sin: Float[Tensor, "S D"],  # noqa: F821
) -> tuple[Float[Tensor, "B H S D"], Float[Tensor, "B H S D"]]:  # noqa: F821
    """Apply rotary position embeddings to Q and K."""
    # Expand cos and sin to match batch and head dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """LLaMA-style multi-head self-attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len

        # LLaMA uses no bias in attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rotary_emb = RotaryEmbedding(self.d_head, max_seq_len=max_seq_len)

    @override
    def forward(self, x: Float[Tensor, "B S D"]) -> Float[Tensor, "B S D"]:  # noqa: F821
        b, s, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [B, H, S, D_head]
        q = q.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(s)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention with causal mask
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.o_proj(attn_out)


class LlamaMLP(nn.Module):
    """LLaMA-style MLP with SwiGLU activation."""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp

        # LLaMA uses no bias and SwiGLU activation
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    @override
    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:  # noqa: F821
        # SwiGLU: down(silu(gate(x)) * up(x))
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaBlock(nn.Module):
    """A single LLaMA transformer block."""

    def __init__(self, d_model: int, d_mlp: int, n_heads: int, max_seq_len: int = 512):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = LlamaAttention(d_model, n_heads, max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.mlp = LlamaMLP(d_model, d_mlp)

    @override
    def forward(self, x: Float[Tensor, "B S D"]) -> Float[Tensor, "B S D"]:  # noqa: F821
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MemTransformer(LoadableModule):
    """A single-block LLaMA-style transformer for memorization tasks.

    The embedding and unembedding matrices are NOT tied, allowing the model
    to learn different representations for input and output tokens.
    """

    def __init__(self, config: MemModelConfig):
        super().__init__()
        self.config = config

        # Embedding layer (NOT tied with unembedding)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Single transformer block
        self.block = LlamaBlock(
            d_model=config.d_model,
            d_mlp=config.d_mlp,
            n_heads=config.n_heads,
            max_seq_len=config.seq_len,
        )

        # Final layer norm
        self.ln_f = RMSNorm(config.d_model)

        # Unembedding layer (separate from embedding, NOT tied)
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using standard transformer initialization."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    @override
    def forward(
        self,
        tokens: Float[Tensor, "B S"],
        **_,  # noqa: F821
    ) -> Float[Tensor, "B S V"]:  # noqa: F821
        """Forward pass returning logits at all positions.

        Args:
            tokens: Input token indices [batch, seq_len]

        Returns:
            logits: Logits for next token prediction [batch, seq_len, vocab_size]
        """
        x = self.embed(tokens)
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.unembed(x)
        return logits

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> tuple[Path, Path]:
        """Download the relevant files from a wandb run.

        Returns:
            - mem_train_config_path: Path to the mem_train_config.yaml file
            - checkpoint_path: Path to the checkpoint file
        """
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run)
        run_dir = fetch_wandb_run_dir(run.id)

        task_name = MemTaskConfig.model_fields["task_name"].default
        mem_train_config_path = download_wandb_file(run, run_dir, f"{task_name}_train_config.yaml")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return mem_train_config_path, checkpoint_path

    @classmethod
    @override
    def from_run_info(cls, run_info: RunInfo[MemTrainConfig]) -> "MemTransformer":
        """Load a pretrained model from a run info object."""
        mem_model = cls(config=run_info.config.mem_model_config)
        mem_model.load_state_dict(
            torch.load(run_info.checkpoint_path, weights_only=True, map_location="cpu")
        )
        return mem_model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "MemTransformer":
        """Fetch a pretrained model from wandb or a local path to a checkpoint."""
        run_info = MemTargetRunInfo.from_path(path)
        return cls.from_run_info(run_info)
