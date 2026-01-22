"""
Multi-Input BatchTopK SAE

A variant of BatchTopK SAE that can use activations from multiple locations
in the network as encoder input, while reconstructing only the primary target.

The hypothesis is that additional context from other network locations
(e.g., the next layer, different parts of the MLP) might help learn better
sparse representations.

Architecture:
- Encoder input: concat([primary_input, aux_input_1, aux_input_2, ...])
- Encoder output: sparse latent features
- Decoder output: reconstruction of primary_input only
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "batchtopk"))

from collections.abc import Iterator
from functools import partial

from config import get_default_cfg, post_init_cfg
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer

from logs import init_wandb, save_checkpoint


class MultiInputActivationsStore:
    """
    Activation store that fetches activations from multiple hook points.

    The first hook point is the "primary" target for reconstruction.
    Additional hook points provide auxiliary context for the encoder.
    """

    def __init__(
        self,
        model: HookedTransformer,
        cfg: dict,
        hook_points: list[str],
    ):
        self.model = model
        self.dataset = iter(load_dataset(cfg["dataset_path"], split="train", streaming=True))
        self.hook_points = hook_points
        self.primary_hook_point = hook_points[0]
        self.context_size = min(cfg["seq_len"], model.cfg.n_ctx)
        self.model_batch_size = cfg["model_batch_size"]
        self.device = cfg["device"]
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.tokens_column = self._get_tokens_column()
        self.cfg = cfg
        self.tokenizer = model.tokenizer

        # Determine the maximum layer we need to run to
        self.max_layer = self._get_max_layer()

        # Initialize dataloader state (will be populated on first next_batch call)
        self.activation_buffer: dict[str, torch.Tensor] | None = None
        self.dataloader: DataLoader | None = None
        self.dataloader_iter: Iterator[tuple[torch.Tensor, ...]] | None = None

    def _get_tokens_column(self):
        sample = next(self.dataset)
        if "tokens" in sample:
            return "tokens"
        elif "input_ids" in sample:
            return "input_ids"
        elif "text" in sample:
            return "text"
        else:
            raise ValueError("Dataset must have a 'tokens', 'input_ids', or 'text' column.")

    def _get_max_layer(self) -> int:
        """Determine the maximum layer needed based on hook points."""
        max_layer = 0
        for hp in self.hook_points:
            # Parse layer from hook point name (e.g., "blocks.8.hook_mlp_out" -> 8)
            parts = hp.split(".")
            for i, part in enumerate(parts):
                if part == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer = int(parts[i + 1])
                    max_layer = max(max_layer, layer)
        return max_layer + 1  # +1 because stop_at_layer is exclusive

    def get_batch_tokens(self):
        all_tokens = []
        while len(all_tokens) < self.model_batch_size * self.context_size:
            batch = next(self.dataset)
            if self.tokens_column == "text":
                tokens = self.model.to_tokens(
                    batch["text"], truncate=True, move_to_device=True, prepend_bos=True
                ).squeeze(0)
            else:
                tokens = batch[self.tokens_column]
            all_tokens.extend(tokens)
        token_tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.device)[
            : self.model_batch_size * self.context_size
        ]
        return token_tensor.view(self.model_batch_size, self.context_size)

    def get_activations(self, batch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get activations from all hook points."""
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=self.hook_points,
                stop_at_layer=self.max_layer,
            )
        return {hp: cache[hp] for hp in self.hook_points}

    def _fill_buffer(self) -> dict[str, torch.Tensor]:
        """Fill buffer with activations from all hook points."""
        all_activations = {hp: [] for hp in self.hook_points}

        for _ in range(self.num_batches_in_buffer):
            batch_tokens = self.get_batch_tokens()
            activations_dict = self.get_activations(batch_tokens)

            for hp in self.hook_points:
                act = activations_dict[hp]
                # Flatten to [batch * seq, act_size]
                act = act.reshape(-1, act.shape[-1])
                all_activations[hp].append(act)

        return {hp: torch.cat(all_activations[hp], dim=0) for hp in self.hook_points}

    def _get_dataloader(self):
        """Create dataloader with aligned activations from all hook points."""
        tensors = [self.activation_buffer[hp] for hp in self.hook_points]
        return DataLoader(TensorDataset(*tensors), batch_size=self.cfg["batch_size"], shuffle=True)

    def _refill_buffer(self) -> None:
        """Refill the activation buffer and reset the dataloader iterator."""
        self.activation_buffer = self._fill_buffer()
        self.dataloader = self._get_dataloader()
        self.dataloader_iter = iter(self.dataloader)

    def next_batch(self) -> tuple[torch.Tensor, ...]:
        """
        Get next batch of activations.

        Returns a tuple of tensors, one per hook point.
        The first tensor is the primary target for reconstruction.
        """
        # Initialize on first call
        if self.dataloader_iter is None:
            self._refill_buffer()
            assert self.dataloader_iter is not None

        # Get next batch, refilling buffer if exhausted
        batch = next(self.dataloader_iter, None)
        if batch is None:
            self._refill_buffer()
            assert self.dataloader_iter is not None
            batch = next(self.dataloader_iter, None)
            assert batch is not None, "Empty dataloader after refill - buffer too small"

        return batch


class MultiInputBatchTopKSAE(nn.Module):
    """
    BatchTopK SAE that uses multiple inputs for encoding but reconstructs
    only the primary input.

    The encoder sees: concat([primary_input, aux_input_1, aux_input_2, ...])
    The decoder outputs: reconstruction of primary_input
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.get("seed", 42))

        # Primary input size (what we reconstruct)
        self.primary_size = cfg["primary_act_size"]
        # Total encoder input size (primary + all auxiliary)
        self.encoder_input_size = cfg["encoder_input_size"]
        # Dictionary size
        self.dict_size = cfg["dict_size"]

        # Encoder: maps concatenated inputs to features
        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(self.encoder_input_size, self.dict_size))
        )
        self.b_enc = nn.Parameter(torch.zeros(self.dict_size))

        # Decoder: maps features back to primary input only
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(self.dict_size, self.primary_size))
        )
        self.b_dec = nn.Parameter(torch.zeros(self.primary_size))

        # Initialize decoder from encoder (using primary input portion)
        with torch.no_grad():
            # Use the primary portion of W_enc for initialization
            self.W_dec.data = self.W_enc.data[: self.primary_size, :].T.clone()
            self.W_dec.data = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        # Track dead features
        self.num_batches_not_active = torch.zeros(self.dict_size, device=cfg.get("device", "cpu"))

        self.to(cfg.get("dtype", torch.float32)).to(cfg.get("device", "cpu"))

    def forward(
        self, primary_input: torch.Tensor, aux_inputs: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            primary_input: The main input to reconstruct [batch, primary_size]
            aux_inputs: List of auxiliary inputs [batch, aux_size_i]

        Returns:
            Dict with loss, reconstruction, features, etc.
        """
        # Concatenate all inputs for the encoder
        encoder_input = torch.cat([primary_input] + aux_inputs, dim=-1)

        # Encode
        acts = F.relu(encoder_input @ self.W_enc + self.b_enc)

        # BatchTopK
        acts_topk = torch.topk(acts.flatten(), self.cfg["top_k"] * primary_input.shape[0], dim=-1)
        acts_topk = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )

        # Decode to primary input space only
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        # Update dead feature tracking
        self.update_inactive_features(acts_topk)

        # Compute losses
        output = self.get_loss_dict(primary_input, x_reconstruct, acts, acts_topk)
        return output

    def get_loss_dict(
        self,
        x: torch.Tensor,
        x_reconstruct: torch.Tensor,
        acts: torch.Tensor,
        acts_topk: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg.get("l1_coeff", 0.0) * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss

        num_dead_features = (
            self.num_batches_not_active > self.cfg.get("n_batches_to_dead", 5)
        ).sum()

        return {
            "sae_out": x_reconstruct,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }

    def get_auxiliary_loss(
        self, x: torch.Tensor, x_reconstruct: torch.Tensor, acts: torch.Tensor
    ) -> torch.Tensor:
        dead_features = self.num_batches_not_active >= self.cfg.get("n_batches_to_dead", 5)
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg.get("top_k_aux", 512), dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg.get("aux_penalty", 1 / 32)
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        return torch.tensor(0, dtype=x.dtype, device=x.device)

    def update_inactive_features(self, acts: torch.Tensor) -> None:
        with torch.no_grad():
            active = acts.sum(0) > 0
            self.num_batches_not_active += (~active).float()
            self.num_batches_not_active[active] = 0

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        if self.W_dec.grad is not None:
            W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
            self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed


def _reconstr_hook(activation, hook, sae_out):
    return sae_out


def _zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)


def _mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)


@torch.no_grad()
def log_multi_input_model_performance(
    wandb_run,
    step: int,
    model: HookedTransformer,
    activation_store: MultiInputActivationsStore,
    sae: MultiInputBatchTopKSAE,
    index: str | None = None,
) -> None:
    """Log model performance metrics for multi-input SAE."""
    cfg = sae.cfg
    batch_tokens = activation_store.get_batch_tokens()[: cfg["batch_size"] // cfg["seq_len"]]

    # Get activations from all hook points
    activations_dict = activation_store.get_activations(batch_tokens)

    # Prepare inputs for SAE
    primary_hp = activation_store.primary_hook_point
    primary_batch = activations_dict[primary_hp].reshape(-1, cfg["primary_act_size"])
    aux_batches = [
        activations_dict[hp].reshape(-1, cfg["primary_act_size"])
        for hp in activation_store.hook_points[1:]
    ]

    # Get SAE reconstruction
    sae_output = sae(primary_batch, aux_batches)["sae_out"]
    sae_output = sae_output.reshape(batch_tokens.shape[0], batch_tokens.shape[1], -1)

    # Compute losses
    original_loss = model(batch_tokens, return_type="loss").item()
    reconstr_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(primary_hp, partial(_reconstr_hook, sae_out=sae_output))],
        return_type="loss",
    ).item()
    zero_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(primary_hp, _zero_abl_hook)],
        return_type="loss",
    ).item()
    mean_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(primary_hp, _mean_abl_hook)],
        return_type="loss",
    ).item()

    ce_degradation = original_loss - reconstr_loss
    zero_degradation = original_loss - zero_loss
    mean_degradation = original_loss - mean_loss

    log_dict = {
        "performance/ce_degradation": ce_degradation,
        "performance/recovery_from_zero": (reconstr_loss - zero_loss) / zero_degradation,
        "performance/recovery_from_mean": (reconstr_loss - mean_loss) / mean_degradation,
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)


def train_multi_input_sae(
    sae: MultiInputBatchTopKSAE,
    activation_store: MultiInputActivationsStore,
    model: HookedTransformer,
    cfg: dict,
) -> None:
    """Training loop for multi-input SAE."""
    num_batches = cfg["num_tokens"] // cfg["batch_size"]

    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"],
    )

    pbar = tqdm.trange(num_batches)
    wandb_run = init_wandb(cfg)

    for i in pbar:
        batch = activation_store.next_batch()
        primary_input = batch[0]
        aux_inputs = list(batch[1:])

        sae_output = sae(primary_input, aux_inputs)

        log_dict = {
            "loss": sae_output["loss"].item(),
            "l2_loss": sae_output["l2_loss"].item(),
            "l0_norm": sae_output["l0_norm"].item(),
            "l1_norm": sae_output["l1_norm"].item(),
            "aux_loss": sae_output["aux_loss"].item(),
            "num_dead_features": sae_output["num_dead_features"].item(),
            "n_dead_in_batch": (sae_output["feature_acts"].sum(0) == 0).sum().item(),
        }
        wandb_run.log(log_dict, step=i)

        if i % cfg.get("perf_log_freq", 1000) == 0:
            log_multi_input_model_performance(wandb_run, i, model, activation_store, sae)

        if i % cfg.get("checkpoint_freq", 10000) == 0:
            save_checkpoint(wandb_run, sae, cfg, i)

        loss = sae_output["loss"]
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "L0": f"{sae_output['l0_norm'].item():.1f}",
                "L2": f"{sae_output['l2_loss'].item():.4f}",
            }
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.get("max_grad_norm", 1.0))
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

    save_checkpoint(wandb_run, sae, cfg, num_batches)
    print("Training complete!")


def main(
    # Training settings
    num_tokens: int = 100_000_000,
    batch_size: int = 4096,
    lr: float = 3e-4,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
    # Model settings
    model_name: str = "gpt2-small",
    layer: int = 8,
    site: str = "mlp_out",
    dict_size: int = 768 * 4,
    # Auxiliary inputs - specify as "layer:site" pairs (use tuple for immutable default)
    aux_inputs: tuple[str, ...] = ("8:resid_mid", "8:resid_post"),
    # aux_inputs: tuple[str, ...] = (),
    # TopK settings
    top_k: int = 32,
    aux_penalty: float = 1 / 32,
    # Other
    seed: int = 42,
    device: str = "cuda:0",
    wandb_project: str = "multi_input_sae",
) -> None:
    """
    Train a Multi-Input BatchTopK SAE.

    This SAE uses activations from multiple network locations as encoder input,
    but reconstructs only the primary target location.

    Args:
        num_tokens: Total tokens to train on
        batch_size: Batch size
        lr: Learning rate
        beta1: Adam beta1
        beta2: Adam beta2
        weight_decay: Weight decay
        model_name: Transformer model name
        layer: Primary layer to extract activations from
        site: Primary hook site (mlp_out, resid_pre, resid_post, etc.)
        dict_size: SAE dictionary size
        aux_inputs: Auxiliary inputs as "layer:site" pairs, e.g. ("9:mlp_out", "8:resid_pre")
        top_k: Number of top-k features to keep
        aux_penalty: Auxiliary loss penalty for dead features
        seed: Random seed
        device: Device to train on
        wandb_project: W&B project name
    """
    # Build config from defaults
    cfg = get_default_cfg()
    cfg["seed"] = seed
    cfg["num_tokens"] = num_tokens
    cfg["batch_size"] = batch_size
    cfg["lr"] = lr
    cfg["beta1"] = beta1
    cfg["beta2"] = beta2
    cfg["weight_decay"] = weight_decay
    cfg["model_name"] = model_name
    cfg["layer"] = layer
    cfg["site"] = site
    cfg["dict_size"] = dict_size
    cfg["top_k"] = top_k
    cfg["top_k_aux"] = min(512, dict_size)
    cfg["aux_penalty"] = aux_penalty
    cfg["device"] = device
    cfg["wandb_project"] = wandb_project
    cfg["sae_type"] = "multi_input_batchtopk"
    cfg["l1_coeff"] = 0.0

    cfg = post_init_cfg(cfg)

    # Load model
    model = HookedTransformer.from_pretrained(model_name).to(device)

    # Build hook points list
    primary_hook_point = cfg["hook_point"]
    hook_points = [primary_hook_point]

    aux_hook_points = []
    for aux_spec in aux_inputs:
        aux_layer, aux_site = aux_spec.split(":")
        aux_layer = int(aux_layer)
        aux_hp = f"blocks.{aux_layer}.hook_{aux_site}"
        aux_hook_points.append(aux_hp)
        hook_points.append(aux_hp)

    # Determine activation sizes
    # For now, assume all hook points have the same size (d_model)
    # This is true for resid_pre, resid_post, mlp_out in standard transformers
    primary_act_size = model.cfg.d_model
    aux_act_sizes = [model.cfg.d_model for _ in aux_hook_points]
    total_encoder_input_size = primary_act_size + sum(aux_act_sizes)

    cfg["primary_act_size"] = primary_act_size
    cfg["encoder_input_size"] = total_encoder_input_size
    cfg["act_size"] = primary_act_size  # For compatibility
    cfg["aux_hook_points"] = aux_hook_points

    print("Multi-Input BatchTopK SAE")
    print(f"  Model: {model_name}")
    print(f"  Primary: layer {layer} ({site}) - {primary_act_size} dims")
    print("  Auxiliary inputs:")
    for aux_hp, aux_size in zip(aux_hook_points, aux_act_sizes, strict=True):
        print(f"    - {aux_hp} - {aux_size} dims")
    print(f"  Total encoder input: {total_encoder_input_size} dims")
    print(f"  Dictionary size: {dict_size}")
    print(f"  top_k: {top_k}")
    print(f"  batch_size: {batch_size}, lr: {lr}")
    print(f"  num_tokens: {num_tokens:,}")

    # Create activation store
    activation_store = MultiInputActivationsStore(model, cfg, hook_points)

    # Create SAE
    sae = MultiInputBatchTopKSAE(cfg)
    print(f"  SAE parameters: {sum(p.numel() for p in sae.parameters()):,}")

    # Train
    train_multi_input_sae(sae, activation_store, model, cfg)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
