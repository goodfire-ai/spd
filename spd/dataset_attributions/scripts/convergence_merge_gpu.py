"""GPU-accelerated convergence merge: distance to final only.

Single pass: accumulate all rank files to get final, then replay accumulation
comparing each prefix to final. Copies files to local SSD for fast reads.
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from spd.log import logger

TOP_K_VALUES = [2, 4, 6, 8]


def compute_metrics(
    current: Float[Tensor, "s c"],
    reference: Float[Tensor, "s c"],
    device: torch.device,
) -> dict:
    col_norms = current.norm(dim=0)
    ref_col_norms = reference.norm(dim=0)
    active = (col_norms > 1e-10) & (ref_col_norms > 1e-10)
    assert active.any()

    ca = current[:, active].to(device)
    ra = reference[:, active].to(device)

    result: dict = {}
    for k in TOP_K_VALUES:
        ct = ca.abs().topk(k, dim=0).indices
        rt = ra.abs().topk(k, dim=0).indices
        matches = (ct.unsqueeze(1) == rt.unsqueeze(0)).any(dim=0).sum(dim=0).float() / k
        result[f"mean_top{k}_overlap"] = float(matches.mean().item())
        result[f"p1_top{k}_overlap"] = float(matches.quantile(0.01).item())
        result[f"min_top{k}_overlap"] = float(matches.min().item())

    return result


def load_rank_comp(path: Path) -> tuple[Tensor, int, int]:
    """Load only source_to_component + metadata from a rank file."""
    d = torch.load(path, weights_only=False)
    return d["source_to_component"], d["n_tokens_processed"], d["n_batches_processed"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("worker_dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    rank_files = sorted(args.worker_dir.glob("dataset_attributions_rank_*.pt"))
    assert rank_files, f"No rank files in {args.worker_dir}"
    logger.info(f"Found {len(rank_files)} rank files")

    # Copy to local SSD if total size fits (< 100GB to avoid filling local disk)
    total_size_gb = sum(rf.stat().st_size for rf in rank_files) / 1e9
    if total_size_gb < 100:
        local_dir = Path(tempfile.mkdtemp(prefix="da_merge_"))
        logger.info(f"Copying {total_size_gb:.0f}GB to local storage ({local_dir})...")
        t0 = time.perf_counter()
        local_files = []
        for rf in rank_files:
            local_path = local_dir / rf.name
            shutil.copy2(rf, local_path)
            local_files.append(local_path)
        logger.info(f"Copied {len(rank_files)} files in {time.perf_counter() - t0:.1f}s")
    else:
        logger.info(f"Skipping local copy ({total_size_gb:.0f}GB too large), reading from NFS")
        local_files = rank_files
        local_dir = None

    # Pass 1: accumulate everything to get final
    logger.info("Pass 1: accumulate all ranks for final reference...")
    t0 = time.perf_counter()
    comp0, tokens0, batches0 = load_rank_comp(local_files[0])
    total_comp = (comp0 * tokens0).double()
    total_tokens = tokens0
    total_batches = batches0
    batch_counts = [(batches0, tokens0)]

    for i, lf in enumerate(local_files[1:], 1):
        comp, tokens, batches = load_rank_comp(lf)
        total_comp += (comp * tokens).double()
        total_tokens += tokens
        total_batches += batches
        batch_counts.append((batches, tokens))
        if (i + 1) % 10 == 0:
            logger.info(f"  loaded {i + 1}/{len(local_files)}")

    final_comp = (total_comp / total_tokens).float()
    logger.info(
        f"Pass 1 done in {time.perf_counter() - t0:.1f}s ({total_batches} batches, {total_tokens:,} tokens)"
    )

    # Pass 2: replay accumulation, compute vs_final at each step
    logger.info("Pass 2: compute vs_final at each prefix...")
    t0 = time.perf_counter()
    comp0, tokens0, _ = load_rank_comp(local_files[0])
    accum = (comp0 * tokens0).double()
    accum_tokens = tokens0
    accum_batches = batch_counts[0][0]

    snapshots: list[dict] = []
    for i, lf in enumerate(local_files):
        if i > 0:
            comp, tokens, batches = load_rank_comp(lf)
            accum += (comp * tokens).double()
            accum_tokens += tokens
            accum_batches += batch_counts[i][0]

        if i == len(local_files) - 1:
            # Last one IS the final
            snapshots.append(
                {
                    "n_ranks_merged": i + 1,
                    "n_batches": accum_batches,
                    "n_tokens": accum_tokens,
                }
            )
            continue

        norm = (accum / accum_tokens).float()
        m = compute_metrics(norm, final_comp, device)
        snapshots.append(
            {
                "n_ranks_merged": i + 1,
                "n_batches": accum_batches,
                "n_tokens": accum_tokens,
                "vs_final": m,
            }
        )

        logger.info(
            f"Rank {i:>3} ({accum_batches:>5} batches): "
            + "  ".join(f"top{k}={m[f'mean_top{k}_overlap']:.3f}" for k in TOP_K_VALUES)
        )

        # Stream
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {"method": "incremental_merge", "n_ranks": len(rank_files), "snapshots": snapshots},
                f,
                indent=2,
            )

    # Final write
    with open(args.output, "w") as f:
        json.dump(
            {"method": "incremental_merge", "n_ranks": len(rank_files), "snapshots": snapshots},
            f,
            indent=2,
        )
    logger.info(f"Pass 2 done in {time.perf_counter() - t0:.1f}s")
    logger.info(f"Results saved to {args.output}")

    # Cleanup
    if local_dir is not None:
        shutil.rmtree(local_dir)
        logger.info(f"Cleaned up {local_dir}")


if __name__ == "__main__":
    main()
