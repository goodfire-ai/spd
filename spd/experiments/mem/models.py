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


def expand_model(
    model: "MemTransformer",
    d_model_new: int,
    d_mlp_new: int,
) -> "MemTransformer":
    """Expand model dimensions by padding weights with zeros.

    Creates a new MemTransformer with expanded dimensions where:
    - All d_model dimensions are expanded to d_model_new
    - All d_mlp dimensions are expanded to d_mlp_new
    - Original weights are preserved in the top-left corner
    - Padded dimensions are filled with zeros (except LayerNorm weights which use 1s)

    Note: When use_layer_norm=True, the expanded model's behavior may differ slightly
    from the original because LayerNorm statistics are computed over all dimensions.
    For exact behavior preservation, use expand with use_layer_norm=False.

    Args:
        model: The original MemTransformer model
        d_model_new: New residual stream dimension (must be >= original d_model)
        d_mlp_new: New MLP hidden dimension (must be >= original d_mlp)

    Returns:
        A new MemTransformer with expanded dimensions
    """
    old_config = model.config
    d_model_old = old_config.d_model
    d_mlp_old = old_config.d_mlp

    assert d_model_new >= d_model_old, (
        f"d_model_new ({d_model_new}) must be >= d_model ({d_model_old})"
    )
    assert d_mlp_new >= d_mlp_old, f"d_mlp_new ({d_mlp_new}) must be >= d_mlp ({d_mlp_old})"

    # n_heads must divide d_model_new evenly
    assert d_model_new % old_config.n_heads == 0, (
        f"d_model_new ({d_model_new}) must be divisible by n_heads ({old_config.n_heads})"
    )

    # Create new config with expanded dimensions
    new_config = MemModelConfig(
        vocab_size=old_config.vocab_size,
        d_model=d_model_new,
        d_mlp=d_mlp_new,
        n_heads=old_config.n_heads,
        seq_len=old_config.seq_len,
        use_layer_norm=old_config.use_layer_norm,
        device=old_config.device,
    )

    # Create new model (with random initialization)
    new_model = MemTransformer(new_config)

    # Helper functions for padding
    def pad_2d(
        old_tensor: Tensor, new_shape: tuple[int, int], pad_value: float = 0.0
    ) -> Tensor:
        """Pad a 2D tensor, preserving original values in top-left corner."""
        new_tensor = torch.full(new_shape, pad_value, dtype=old_tensor.dtype)
        new_tensor[: old_tensor.shape[0], : old_tensor.shape[1]] = old_tensor
        return new_tensor

    def pad_1d(old_tensor: Tensor, new_size: int, pad_value: float = 0.0) -> Tensor:
        """Pad a 1D tensor, preserving original values at the start."""
        new_tensor = torch.full((new_size,), pad_value, dtype=old_tensor.dtype)
        new_tensor[: old_tensor.shape[0]] = old_tensor
        return new_tensor

    with torch.no_grad():
        # Token embedding: [vocab_size, d_model] -> [vocab_size, d_model_new]
        new_model.embed.weight.copy_(
            pad_2d(model.embed.weight, (old_config.vocab_size, d_model_new))
        )

        # Position embedding: [seq_len, d_model] -> [seq_len, d_model_new]
        new_model.pos_embed.weight.copy_(
            pad_2d(model.pos_embed.weight, (old_config.seq_len, d_model_new))
        )

        # Attention projections (all d_model -> d_model)
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            old_proj = getattr(model.block.attn, proj_name)
            new_proj = getattr(new_model.block.attn, proj_name)
            # weight: [d_model, d_model] -> [d_model_new, d_model_new]
            new_proj.weight.copy_(pad_2d(old_proj.weight, (d_model_new, d_model_new)))
            # bias: [d_model] -> [d_model_new]
            new_proj.bias.copy_(pad_1d(old_proj.bias, d_model_new))

        # MLP up_proj: [d_mlp, d_model] -> [d_mlp_new, d_model_new]
        new_model.block.mlp.up_proj.weight.copy_(
            pad_2d(model.block.mlp.up_proj.weight, (d_mlp_new, d_model_new))
        )
        new_model.block.mlp.up_proj.bias.copy_(
            pad_1d(model.block.mlp.up_proj.bias, d_mlp_new)
        )

        # MLP down_proj: [d_model, d_mlp] -> [d_model_new, d_mlp_new]
        new_model.block.mlp.down_proj.weight.copy_(
            pad_2d(model.block.mlp.down_proj.weight, (d_model_new, d_mlp_new))
        )
        new_model.block.mlp.down_proj.bias.copy_(
            pad_1d(model.block.mlp.down_proj.bias, d_model_new)
        )

        # LayerNorm layers (if present)
        if old_config.use_layer_norm:
            # ln1: [d_model] -> [d_model_new]
            # Weight padded with 1s (identity for normalization), bias with 0s
            new_model.block.ln1.weight.copy_(
                pad_1d(model.block.ln1.weight, d_model_new, pad_value=1.0)
            )
            new_model.block.ln1.bias.copy_(pad_1d(model.block.ln1.bias, d_model_new))

            # ln2: [d_model] -> [d_model_new]
            new_model.block.ln2.weight.copy_(
                pad_1d(model.block.ln2.weight, d_model_new, pad_value=1.0)
            )
            new_model.block.ln2.bias.copy_(pad_1d(model.block.ln2.bias, d_model_new))

            # ln_f: [d_model] -> [d_model_new]
            new_model.ln_f.weight.copy_(
                pad_1d(model.ln_f.weight, d_model_new, pad_value=1.0)
            )
            new_model.ln_f.bias.copy_(pad_1d(model.ln_f.bias, d_model_new))

        # Unembed: [vocab_size, d_model] -> [vocab_size, d_model_new]
        new_model.unembed.weight.copy_(
            pad_2d(model.unembed.weight, (old_config.vocab_size, d_model_new))
        )

    return new_model
