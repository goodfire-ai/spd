"""MemTransformer: A single-block GPT-style transformer for memorization tasks."""

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


class GPTAttention(nn.Module):
    """GPT-style multi-head self-attention with learned position embeddings."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # GPT uses bias in attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

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

        # Scaled dot-product attention with causal mask
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.o_proj(attn_out)


class GPTMLP(nn.Module):
    """GPT-style MLP with GELU activation."""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp

        # GPT uses bias and GELU activation
        self.up_proj = nn.Linear(d_model, d_mlp)
        self.down_proj = nn.Linear(d_mlp, d_model)

    @override
    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:  # noqa: F821
        x = self.up_proj(x)
        x = F.gelu(x)
        return self.down_proj(x)


class GPTBlock(nn.Module):
    """A single GPT transformer block with optional pre-norm architecture."""

    def __init__(self, d_model: int, d_mlp: int, n_heads: int, use_layer_norm: bool = True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
        self.attn = GPTAttention(d_model, n_heads)
        self.mlp = GPTMLP(d_model, d_mlp)

    @override
    def forward(self, x: Float[Tensor, "B S D"]) -> Float[Tensor, "B S D"]:  # noqa: F821
        if self.use_layer_norm:
            # Pre-norm architecture (GPT-2 style)
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        else:
            # No normalization
            x = x + self.attn(x)
            x = x + self.mlp(x)
        return x


class MemTransformer(LoadableModule):
    """A single-block GPT-style transformer for memorization tasks.

    The embedding and unembedding matrices are NOT tied, allowing the model
    to learn different representations for input and output tokens.
    """

    def __init__(self, config: MemModelConfig):
        super().__init__()
        self.config = config

        # Token embedding layer (NOT tied with unembedding)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Learned position embeddings (GPT style)
        self.pos_embed = nn.Embedding(config.seq_len, config.d_model)

        # Single transformer block
        self.block = GPTBlock(
            d_model=config.d_model,
            d_mlp=config.d_mlp,
            n_heads=config.n_heads,
            use_layer_norm=config.use_layer_norm,
        )

        # Final layer norm (optional)
        if config.use_layer_norm:
            self.ln_f = nn.LayerNorm(config.d_model)

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
                # unembed has bias=False, so bias can be None
                if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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
        _, s = tokens.shape

        # Token + position embeddings
        positions = torch.arange(s, device=tokens.device)
        x = self.embed(tokens) + self.pos_embed(positions)

        x = self.block(x)
        if self.config.use_layer_norm:
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
