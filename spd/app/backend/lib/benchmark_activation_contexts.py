"""Benchmark script for activation context collection.

Usage:
    source .venv/bin/activate
    python spd/app/backend/lib/benchmark_activation_contexts.py

This script profiles the `get_activations_data_streaming` function to identify
bottlenecks. It uses a mock LM-like model that produces outputs with shape (B, S, C)
to match real language model usage.
"""

import cProfile
import pstats
import time
from dataclasses import dataclass
from io import StringIO
from typing import Any, override

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from spd.app.backend.lib.activation_contexts import get_activations_data_streaming
from spd.app.backend.services.run_context_service import TrainRunContext
from spd.configs import Config, ImportanceMinimalityLossConfig, StochasticReconLossConfig
from spd.experiments.lm.configs import LMTaskConfig
from spd.interfaces import LoadableModule, RunInfo
from spd.models.component_model import ComponentModel
from spd.spd_types import ModelPath
from spd.utils.general_utils import set_seed


class MockTokenizer:
    """Mock tokenizer for benchmarking without HuggingFace."""

    pad_token_id = 0
    vocab_size = 1000

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return f"tok_{ids}"
        return [f"tok_{i}" for i in ids]


class MockLMModel(LoadableModule):
    """Mock LM model that takes token IDs and produces (B, S, d_model) output.

    This mimics a transformer-based LM where:
    - Input: token IDs of shape (B, S)
    - Embedding: maps to (B, S, d_model)
    - MLP layers: (B, S, d_model) -> (B, S, d_mlp) -> (B, S, d_model)
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 64,
        d_mlp: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "mlp_in": nn.Linear(d_model, d_mlp),
                        "mlp_out": nn.Linear(d_mlp, d_model),
                    }
                )
            )

        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size)

    @override
    def forward(self, input_ids: Int[Tensor, "B S"]) -> Float[Tensor, "B S vocab"]:
        x = self.embedding(input_ids)  # (B, S, d_model)

        for layer in self.layers:
            # Simple MLP block (no attention for simplicity)
            mlp_out = layer["mlp_out"](torch.relu(layer["mlp_in"](x)))
            x = x + mlp_out  # residual

        x = self.ln_final(x)
        return self.unembed(x)  # (B, S, vocab)

    @classmethod
    @override
    def from_run_info(cls, run_info: RunInfo[Any]) -> "MockLMModel":
        return cls()

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "MockLMModel":
        return cls()


class MockDataLoader:
    """Mock dataloader that yields random token ID batches."""

    def __init__(self, vocab_size: int, seq_len: int, device: str):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device

    def __iter__(self):
        while True:
            # Yield batch size 1 (matching real app usage)
            yield torch.randint(1, self.vocab_size, (1, self.seq_len), device=self.device)


def create_mock_run_context(
    vocab_size: int = 1000,
    d_model: int = 64,
    d_mlp: int = 256,
    n_layers: int = 2,
    seq_len: int = 128,
    C: int = 50,
    device: str = "cpu",
) -> TrainRunContext:
    """Create a mock TrainRunContext for benchmarking with LM-like shapes."""
    set_seed(42)

    config = Config(
        wandb_project=None,
        wandb_run_name=None,
        wandb_run_name_prefix="",
        seed=42,
        C=C,
        n_mask_samples=1,
        ci_fn_type="vector_mlp",  # Use vector_mlp for LM-like models
        ci_fn_hidden_dims=[32],
        loss_metric_configs=[
            ImportanceMinimalityLossConfig(coeff=3e-3, pnorm=0.9, eps=1e-12),
            StochasticReconLossConfig(coeff=1.0),
        ],
        target_module_patterns=["layers.*.mlp_in", "layers.*.mlp_out"],
        identity_module_patterns=None,
        output_loss_type="mse",
        lr=1e-3,
        batch_size=32,
        steps=100,
        lr_schedule="cosine",
        lr_exponential_halflife=None,
        lr_warmup_pct=0.01,
        n_eval_steps=1,
        eval_freq=10,
        eval_batch_size=32,
        slow_eval_freq=10,
        slow_eval_on_first_step=True,
        train_log_freq=50,
        save_freq=None,
        ci_alive_threshold=0.1,
        n_examples_until_dead=200,
        pretrained_model_class="MockLMModel",
        pretrained_model_path=None,
        pretrained_model_name=None,
        pretrained_model_output_attr=None,
        tokenizer_name=None,
        task_config=LMTaskConfig(
            task_name="lm",
            max_seq_len=seq_len,
        ),
    )

    target_model = MockLMModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_mlp=d_mlp,
        n_layers=n_layers,
    ).to(device)
    target_model.requires_grad_(False)

    cm = ComponentModel(
        target_model=target_model,
        target_module_patterns=config.target_module_patterns,
        C=config.C,
        ci_fn_type=config.ci_fn_type,
        ci_fn_hidden_dims=config.ci_fn_hidden_dims,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
        sigmoid_type="leaky_hard",
    )
    cm.to(device)

    train_loader = MockDataLoader(vocab_size=vocab_size, seq_len=seq_len, device=device)

    return TrainRunContext(
        wandb_id="benchmark",
        wandb_path="benchmark/test",
        config=config,
        cm=cm,
        tokenizer=MockTokenizer(),  # type: ignore
        train_loader=train_loader,  # type: ignore
    )


@dataclass
class BenchmarkParams:
    n_batches: int = 10
    batch_size: int = 32
    importance_threshold: float = 0.1
    n_tokens_either_side: int = 5
    topk_examples: int = 20


def run_benchmark(run_context: TrainRunContext, params: BenchmarkParams) -> dict[str, Any]:
    """Run the activation collection and return timing info."""
    start = time.perf_counter()

    results = list(
        get_activations_data_streaming(
            run_context=run_context,
            importance_threshold=params.importance_threshold,
            n_batches=params.n_batches,
            n_tokens_either_side=params.n_tokens_either_side,
            batch_size=params.batch_size,
            topk_examples=params.topk_examples,
        )
    )

    elapsed = time.perf_counter() - start

    # Extract the final result
    complete_result = next((r for t, r in results if t == "complete"), None)

    return {
        "elapsed_seconds": elapsed,
        "n_batches": params.n_batches,
        "batch_size": params.batch_size,
        "result": complete_result,
    }


def profile_with_cprofile(run_context: TrainRunContext, params: BenchmarkParams) -> str:
    """Profile with cProfile and return formatted stats."""
    profiler = cProfile.Profile()
    profiler.enable()

    list(
        get_activations_data_streaming(
            run_context=run_context,
            importance_threshold=params.importance_threshold,
            n_batches=params.n_batches,
            n_tokens_either_side=params.n_tokens_either_side,
            batch_size=params.batch_size,
            topk_examples=params.topk_examples,
        )
    )

    profiler.disable()

    # Format stats
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(40)

    return stream.getvalue()


def main():
    print("=" * 60)
    print("Activation Context Collection Benchmark")
    print("=" * 60)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")

    # Model config - roughly matches a small LM
    model_kwargs = {
        "vocab_size": 1000,
        "d_model": 64,
        "d_mlp": 256,
        "n_layers": 2,
        "seq_len": 128,
        "C": 50,
        "device": device,
    }

    # Create mock context
    print("\nCreating mock run context...")
    print(
        f"  Model: vocab={model_kwargs['vocab_size']}, d_model={model_kwargs['d_model']}, "
        f"d_mlp={model_kwargs['d_mlp']}, n_layers={model_kwargs['n_layers']}"
    )
    print(f"  Sequence length: {model_kwargs['seq_len']}")
    print(f"  Components (C): {model_kwargs['C']}")

    run_context = create_mock_run_context(**model_kwargs)

    params = BenchmarkParams(
        n_batches=20,
        batch_size=32,
        importance_threshold=0.1,
        n_tokens_either_side=10,
        topk_examples=50,
    )

    print("\nBenchmark params:")
    print(f"  n_batches: {params.n_batches}")
    print(f"  batch_size: {params.batch_size}")
    print(f"  importance_threshold: {params.importance_threshold}")
    print(f"  n_tokens_either_side: {params.n_tokens_either_side}")
    print(f"  topk_examples: {params.topk_examples}")
    print(f"  Total tokens: {params.n_batches * params.batch_size * model_kwargs['seq_len']:,}")

    # Warmup
    print("\nWarmup run...")
    warmup_params = BenchmarkParams(n_batches=2, batch_size=8)
    run_benchmark(run_context, warmup_params)

    # Basic timing
    print("\n" + "-" * 40)
    print("Basic Timing")
    print("-" * 40)
    result = run_benchmark(run_context, params)
    print(f"Total time: {result['elapsed_seconds']:.3f}s")
    print(f"Time per batch: {result['elapsed_seconds'] / params.n_batches * 1000:.1f}ms")
    tokens_per_sec = (
        params.n_batches * params.batch_size * model_kwargs["seq_len"] / result["elapsed_seconds"]
    )
    print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")

    # cProfile output
    print("\n" + "-" * 40)
    print("cProfile Output (top 40 functions)")
    print("-" * 40)

    run_context = create_mock_run_context(**model_kwargs)

    profile_output = profile_with_cprofile(run_context, params)
    print(profile_output)


if __name__ == "__main__":
    main()
